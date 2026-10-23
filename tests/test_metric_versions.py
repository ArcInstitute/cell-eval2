import argparse
import dataclasses
import json
import pathlib

import pytest

from cell_eval2 import catalog
from cell_eval2.baseline import _baseline_policy_dict, build_run_meta, config_digest
from cell_eval2.catalog import CATALOG, METRICS_VERSION, PROFILES, resolve_metrics
from cell_eval2.catalog_text import text_key
from cell_eval2.config import DEParams, EvalConfig
from cell_eval2.evaluation_results import (SCALAR_METRICS, extend_ledger, headline_statistic,
                                           identity_violations, metric_axis, metric_identity)

SNAPSHOT = pathlib.Path(__file__).parent / "fixtures" / "metric_identity.json"


def _design_class(name):
    continuous = (name.startswith(("expr_", "delta_", "cell_"))
                  or text_key(name) in ("lfc_nmae", "pr_auc", "roc_auc"))
    return "continuous" if continuous else "exact"


def test_every_metric_starts_at_version_one_in_its_design_class():
    for n, s in CATALOG.items():
        assert type(s.version) is int and s.version == 1, n
        assert s.tolerance == _design_class(n), n


def test_de_family_siblings_share_version_and_tolerance():
    for n, s in CATALOG.items():
        if n.startswith("de_wilcoxon_"):
            t = CATALOG[n.replace("de_wilcoxon_", "de_deseq2_", 1)]
            assert (t.version, t.tolerance) == (s.version, s.tolerance), n


@pytest.mark.parametrize("bad", [0, -1, True, 1.0, "1"])
def test_version_must_be_an_int_of_at_least_one(bad):
    with pytest.raises(ValueError, match="version"):
        dataclasses.replace(CATALOG["expr_mae"], version=bad)


def test_tolerance_must_be_a_known_class():
    with pytest.raises(ValueError, match="tolerance"):
        dataclasses.replace(CATALOG["expr_mae"], tolerance="loose")


def test_counter_starts_at_one():
    assert METRICS_VERSION == 1


def test_scalar_metrics_are_the_nsig_spearman_pair():
    from cell_eval2.metrics.de import de_nsig_spearman
    assert SCALAR_METRICS == {n for n, s in CATALOG.items() if s.func is de_nsig_spearman}
    assert headline_statistic("expr_mse_unbiased_capped_norm") == "ratio_of_sums"
    assert headline_statistic("de_wilcoxon_nsig_spearman") == "value"
    assert headline_statistic("pds_cosine") == "mean"
    assert metric_axis("pds_cosine") == "perturbation"
    assert metric_axis("de_wilcoxon_nsig_spearman") is None
    assert metric_axis("expr_mse_unbiased_capped_norm") is None


def test_identity_snapshot_holds():
    assert identity_violations(json.loads(SNAPSHOT.read_text()), metric_identity()) == []


def test_tolerance_change_requires_a_metric_version_bump():
    current = metric_identity()
    ledger = extend_ledger(None, current)
    current["metrics"]["pds_l1"]["tolerance"] = "continuous"
    message = "pds_l1 v1: tolerance changed without a version bump"
    assert identity_violations(ledger, current) == [message]
    with pytest.raises(ValueError, match=message):
        extend_ledger(ledger, current)


def test_metrics_version_cannot_decrease_with_unchanged_identities():
    current = metric_identity()
    ledger = extend_ledger(None, current)
    ledger["metrics_version"] = 2
    message = "metrics_version decreased: 2 -> 1"
    assert identity_violations(ledger, current) == [message]
    with pytest.raises(ValueError, match=message):
        extend_ledger(ledger, current)


def _set(d, k, v):
    d[k] = v


@pytest.mark.parametrize("mutate, expected", [
    (lambda s, c: c["metrics"].pop("pds_l1"), "metrics_version bump"),
    (lambda s, c: _set(c["metrics"], "pds_new", dict(c["metrics"]["pds_l1"])), "metrics_version bump"),
    (lambda s, c: _set(c["metrics"]["pds_l1"], "version", 2), "metrics_version bump"),
    (lambda s, c: (_set(c["metrics"]["pds_l1"], "version", 2), _set(c, "metrics_version", 2)), None),
    (lambda s, c: _set(c["metrics"]["pds_l1"], "statistic", "median"), "without a version bump"),
    (lambda s, c: _set(c["metrics"]["pds_l1"], "axis", None), "without a version bump"),
    (lambda s, c: _set(c["metrics"]["pds_l1"], "axis", "context"), "without a version bump"),
    (lambda s, c: _set(s["history"]["pds_l1"], "2", dict(s["history"]["pds_l1"]["1"])),
     "version decreased"),
])
def test_identity_rules_discriminate(mutate, expected):
    snap, cur = extend_ledger(None, metric_identity()), metric_identity()
    mutate(snap, cur)
    got = identity_violations(snap, cur)
    assert (got == []) if expected is None else any(expected in v for v in got), got


def test_a_retired_identity_keeps_its_shape():
    a = metric_identity()
    b = {"metrics_version": a["metrics_version"] + 1,
         "metrics": {k: v for k, v in a["metrics"].items() if k != "pds_l1"}}
    ledger = extend_ledger(extend_ledger(None, a), b)             # release B retires pds_l1
    assert "pds_l1" not in ledger["active"] and "1" in ledger["history"]["pds_l1"]
    c = {"metrics_version": b["metrics_version"] + 1,
         "metrics": {**b["metrics"], "pds_l1": {**a["metrics"]["pds_l1"], "axis": None}}}
    assert any(v.startswith("pds_l1") for v in identity_violations(ledger, c))  # back, reshaped
    with pytest.raises(ValueError, match="pds_l1"):
        extend_ledger(ledger, c)                                   # a release cannot record it
    c["metrics"]["pds_l1"] = a["metrics"]["pds_l1"]
    assert identity_violations(ledger, c) == []                    # back unchanged: fine


def _cfg(profile):
    return EvalConfig(metrics=profile, device="cpu", de=DEParams(backend="pdex"))


def test_no_metric_version_term_while_every_version_is_one():
    for prof in PROFILES:
        names = resolve_metrics(prof)[0]
        assert "metric_version" not in _baseline_policy_dict(names, comparator="bulk_lognorm")


def test_a_selected_metric_at_version_two_moves_config_digest(monkeypatch):
    before = config_digest(_cfg("vcc2026"), comparator="bulk_lognorm")
    spec = CATALOG["de_wilcoxon_lfc_nmae"]
    monkeypatch.setitem(catalog.CATALOG, spec.name, dataclasses.replace(spec, version=2))
    assert config_digest(_cfg("vcc2026"), comparator="bulk_lognorm") != before


def test_an_unselected_metric_at_version_two_moves_nothing(monkeypatch):
    before = config_digest(_cfg("vcc2026"), comparator="bulk_lognorm")
    spec = CATALOG["de_wilcoxon_overlap"]          # not a vcc2026 member
    monkeypatch.setitem(catalog.CATALOG, spec.name, dataclasses.replace(spec, version=2))
    assert config_digest(_cfg("vcc2026"), comparator="bulk_lognorm") == before


@pytest.mark.parametrize("bumped, recomputes", [("expr_mae", True),    # a `minimal` member
                                                ("pds_l2", False)])   # not selected
def test_a_selected_version_bump_misses_the_warm_result_cache(monkeypatch, tmp_path, bumped,
                                                              recomputes):
    """A warm result cache must not serve a pre-bump frame under the new version (Copilot, #418)."""
    from cell_eval2 import run
    from _edge_fixtures import config, scenario
    pred, real, _ = scenario("random")
    cfg = dataclasses.replace(config("minimal"), cache_pred=str(tmp_path / "cache"))
    run.compute_metrics(pred, real, config=cfg)                  # cold: fills the result cache
    calls = []
    dispatch = run.dispatch_anndata_metrics

    def spy(*a, **k):
        calls.append(1)
        return dispatch(*a, **k)
    monkeypatch.setattr(run, "dispatch_anndata_metrics", spy)
    monkeypatch.setitem(catalog.CATALOG, bumped,
                        dataclasses.replace(CATALOG[bumped], version=2))
    run.compute_metrics(pred, real, config=cfg)
    assert bool(calls) is recomputes


def test_run_meta_records_metrics_version_and_nothing_compares_it(synthetic_counts_pair, tmp_path):
    from cell_eval2 import cli, real_bundle
    pred, real = synthetic_counts_pair
    meta = build_run_meta(EvalConfig(metrics=["expr_mae"], device="cpu"), real, pred)
    assert meta["metrics_version"] == METRICS_VERSION
    assert "metrics_version" not in real_bundle.SUBMISSION_PEERS + real_bundle.MANIFEST_RECORDED_ONLY
    (tmp_path / "base.json").write_text(json.dumps(meta))
    (tmp_path / "user.json").write_text(json.dumps({**meta, "metrics_version": METRICS_VERSION + 1}))
    cli._check_baseline_config(argparse.Namespace(
        baseline_meta=str(tmp_path / "base.json"), baseline_agg=str(tmp_path / "b.csv"),
        user_meta=str(tmp_path / "user.json"), user_agg=str(tmp_path / "u.csv"),
        allow_config_mismatch=False))   # returns: metrics_version is recorded, never compared
