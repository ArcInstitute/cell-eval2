"""Tests for the ``de_deseq2_*`` metric family + backend-driven relabel.

``test_de_wilcoxon_family_is_unchanged`` is the parity gate: the method-parameterized
``_register_de_family`` refactor must reproduce every pre-refactor ``de_wilcoxon_*`` spec
byte-identically (every MetricSpec field + full callable binding + CATALOG order). The
remaining tests drive the new deseq2 behavior. Raw-count toy AnnData comes from the shared
``toy_de_adata`` factory fixture (tests/conftest.py).
"""
from __future__ import annotations

import dataclasses

import pytest

from cell_eval2.catalog import CATALOG, PROFILES


def _func_id(spec):
    """Behavioral identity of a metric's func: underlying function + bound keywords. Object
    identity (``is``) does NOT hold across families — each ``_register_de_family`` call builds
    fresh ``functools.partial`` instances — so compare by (qualname, keywords)."""
    fn = getattr(spec.func, "func", spec.func)
    return (fn.__qualname__, tuple(sorted(getattr(spec.func, "keywords", {}).items())))


def _spec_fingerprint(spec):
    """Airtight identity for the parity gate: EVERY MetricSpec field + the full callable
    binding (module, qualname, positional args, keywords). Object identity won't survive the
    builder refactor (fresh partials), so bind by value; ``name`` is included first so the
    ordered-list comparison also pins CATALOG insertion order.

    ⚠️ ``spec.scoring`` is the LAST element, and it is the authoritative one (issue #212).
    Until it was added this docstring's "EVERY MetricSpec field" was false about the field
    that decides how a metric SCORES: `best_value` (field 2) is a deprecated, lossy, derived
    property collapsing to three tokens, while `Scoring` carries `scored`, `direction`,
    `anchor`, `penalty`, `penalty_exponent`, `penalty_cap`, `clamp_low`, `clamp_high`,
    `allow_negative_baseline` and `metric_min` (the last element of each scoring tuple --
    added when the four bounded `vcc2026` members lost their clip at 0). Flipping one of the anchorless DE metrics to `anchor=1.0`, or
    giving an error metric a different `penalty_cap`, changed how it scored and left this
    golden green. `best_value` is KEPT rather than replaced: it is derived, so it adds no
    coverage, but removing it would renumber every field index the comments below cite.

    ⚠️ DECLARED, not resolved. `dataclasses.astuple` records a `None` `penalty_cap` as
    `None`, not as `DEFAULT_PENALTY_CAP`. That is the right scope for a gate about
    deliberate CATALOG edits; a change to a module-level default is covered instead by
    `competition_payload`, which resolves before hashing."""
    fn = getattr(spec.func, "func", spec.func)
    args = tuple(getattr(spec.func, "args", ()))
    kwargs = tuple(sorted(getattr(spec.func, "keywords", {}).items()))
    return (spec.name, spec.v1_name, spec.best_value, spec.worst_value, tuple(spec.profiles),
            tuple(spec.aliases), spec.kind, spec.normalization, spec.agg, spec.v1_available,
            fn.__module__, fn.__qualname__, args, kwargs, spec.needs_moments,
            dataclasses.astuple(spec.scoring))


# Ordered golden snapshot of the de_wilcoxon_* family. Keep every spec exact and ordered so
# additions are deliberate and the deseq2 mirror remains mechanically faithful.
GOLDEN_WILCOXON = [
    ('de_wilcoxon_overlap', 'overlap_at_N', 'one', None, ('full', 'minimal', 'de', 'vcc'), (), 'de', None, 'mean', True, 'cell_eval2.metrics.de', 'de_overlap', (), (('k', None), ('metric', 'overlap')), False, (True, 'higher', 1.0, 'none', None, None, 0.0, None, False, None)),
    ('de_wilcoxon_overlap_top50', 'overlap_at_50', 'one', None, ('full', 'de'), (), 'de', None, 'mean', True, 'cell_eval2.metrics.de', 'de_overlap', (), (('k', 50), ('metric', 'overlap')), False, (True, 'higher', 1.0, 'none', None, None, 0.0, None, False, None)),
    ('de_wilcoxon_overlap_top100', 'overlap_at_100', 'one', None, ('full', 'de'), (), 'de', None, 'mean', True, 'cell_eval2.metrics.de', 'de_overlap', (), (('k', 100), ('metric', 'overlap')), False, (True, 'higher', 1.0, 'none', None, None, 0.0, None, False, None)),
    ('de_wilcoxon_overlap_top200', 'overlap_at_200', 'one', None, ('full', 'de'), (), 'de', None, 'mean', True, 'cell_eval2.metrics.de', 'de_overlap', (), (('k', 200), ('metric', 'overlap')), False, (True, 'higher', 1.0, 'none', None, None, 0.0, None, False, None)),
    ('de_wilcoxon_overlap_top500', 'overlap_at_500', 'one', None, ('full', 'de'), (), 'de', None, 'mean', True, 'cell_eval2.metrics.de', 'de_overlap', (), (('k', 500), ('metric', 'overlap')), False, (True, 'higher', 1.0, 'none', None, None, 0.0, None, False, None)),
    ('de_wilcoxon_precision', 'precision_at_N', 'one', None, ('full', 'minimal', 'de'), (), 'de', None, 'mean', True, 'cell_eval2.metrics.de', 'de_overlap', (), (('k', None), ('metric', 'precision')), False, (True, 'higher', 1.0, 'none', None, None, 0.0, None, False, None)),
    ('de_wilcoxon_precision_top50', 'precision_at_50', 'one', None, ('full', 'de'), (), 'de', None, 'mean', True, 'cell_eval2.metrics.de', 'de_overlap', (), (('k', 50), ('metric', 'precision')), False, (True, 'higher', 1.0, 'none', None, None, 0.0, None, False, None)),
    ('de_wilcoxon_precision_top100', 'precision_at_100', 'one', None, ('full', 'de'), (), 'de', None, 'mean', True, 'cell_eval2.metrics.de', 'de_overlap', (), (('k', 100), ('metric', 'precision')), False, (True, 'higher', 1.0, 'none', None, None, 0.0, None, False, None)),
    ('de_wilcoxon_precision_top200', 'precision_at_200', 'one', None, ('full', 'de'), (), 'de', None, 'mean', True, 'cell_eval2.metrics.de', 'de_overlap', (), (('k', 200), ('metric', 'precision')), False, (True, 'higher', 1.0, 'none', None, None, 0.0, None, False, None)),
    ('de_wilcoxon_precision_top500', 'precision_at_500', 'one', None, ('full', 'de'), (), 'de', None, 'mean', True, 'cell_eval2.metrics.de', 'de_overlap', (), (('k', 500), ('metric', 'precision')), False, (True, 'higher', 1.0, 'none', None, None, 0.0, None, False, None)),
    ('de_wilcoxon_nsig_counts_real', 'de_nsig_counts_real', 'none', None, ('full', 'minimal', 'de'), (), 'de', None, 'mean', True, 'cell_eval2.metrics.de', 'de_nsig_counts', (), (('side', 'real'),), False, (False, None, None, 'none', None, None, 0.0, None, False, None)),
    ('de_wilcoxon_nsig_counts_pred', 'de_nsig_counts_pred', 'none', None, ('full', 'minimal', 'de'), (), 'de', None, 'mean', True, 'cell_eval2.metrics.de', 'de_nsig_counts', (), (('side', 'pred'),), False, (False, None, None, 'none', None, None, 0.0, None, False, None)),
    ('de_wilcoxon_nsig_spearman', 'de_spearman_sig', 'one', -1.0, ('full', 'de'), (), 'de', None, 'mean', True, 'cell_eval2.metrics.de', 'de_nsig_spearman', (), (), False, (True, 'higher', 1.0, 'none', None, None, 0.0, None, False, None)),
    ('de_wilcoxon_sig_recall', 'de_sig_genes_recall', 'one', 0.0, ('full', 'de'), (), 'de', None, 'mean', True, 'cell_eval2.metrics.de', 'de_sig_recall', (), (), False, (True, 'higher', 1.0, 'none', None, None, 0.0, None, False, None)),
    ('de_wilcoxon_direction_match', 'de_direction_match', 'one', 0.0, ('full', 'de'), (), 'de', None, 'mean', True, 'cell_eval2.metrics.de', 'de_direction_match', (), (), False, (True, 'higher', 1.0, 'none', None, None, 0.0, None, False, None)),
    ('de_wilcoxon_model_direction_match', 'de_model_direction_match', 'one', 0.0, ('full', 'de'), (), 'de', None, 'mean', True, 'cell_eval2.metrics.de', 'de_model_direction_match', (), (), False, (True, 'higher', 1.0, 'none', None, None, 0.0, None, False, None)),
    ('de_wilcoxon_lfc_spearman', 'de_spearman_lfc_sig', 'one', -1.0, ('full', 'de'), (), 'de', None, 'mean', True, 'cell_eval2.metrics.de', 'de_lfc_spearman', (), (), False, (True, 'higher', 1.0, 'none', None, None, 0.0, None, False, None)),
    ('de_wilcoxon_lfc_spearman_pos', 'de_spearman_pos_lfc_sig', 'one', -1.0, ('full', 'de'), (), 'de', None, 'mean', True, 'cell_eval2.metrics.de', 'de_lfc_spearman', (), (('lfc_direction', 'pos'),), False, (True, 'higher', 1.0, 'none', None, None, 0.0, None, False, None)),
    ('de_wilcoxon_lfc_spearman_neg', 'de_spearman_neg_lfc_sig', 'one', -1.0, ('full', 'de'), (), 'de', None, 'mean', True, 'cell_eval2.metrics.de', 'de_lfc_spearman', (), (('lfc_direction', 'neg'),), False, (True, 'higher', 1.0, 'none', None, None, 0.0, None, False, None)),
    # #208. best_value (field 2) is 'zero': scored + direction='lower' maps to the old
    # error token. v1_available (field 9) is False -- v2-native, no cell-eval equivalent.
    ('de_wilcoxon_lfc_nmae', None, 'zero', None, ('full', 'de', 'vcc2026'), (), 'de', None, 'mean', False, 'cell_eval2.metrics.de', 'de_lfc_nmae', (), (), False, (True, 'lower', 0.0, 'none', None, None, -6.0, None, False, None)),
    ('de_wilcoxon_pr_auc', 'pr_auc', 'one', 0.0, ('full', 'de'), (), 'de', None, 'mean', True, 'cell_eval2.metrics.de', 'de_pr_auc', (), (), False, (True, 'higher', 1.0, 'none', None, None, 0.0, None, False, None)),
    ('de_wilcoxon_roc_auc', 'roc_auc', 'one', 0.0, ('full', 'de'), (), 'de', None, 'mean', True, 'cell_eval2.metrics.de', 'de_roc_auc', (), (), False, (True, 'higher', 1.0, 'none', None, None, 0.0, None, False, None)),
    # v1_available (field 9) is False for all seven: they are v2-native (v1_name=None), and
    # v1 availability is now derived from v1_name rather than hand-flagged.
    ('de_wilcoxon_overlap_adjusted', None, 'one', None, ('full', 'de'), (), 'de', None, 'mean', False, 'cell_eval2.metrics.de', 'de_overlap_adjusted', (), (), False, (True, 'higher', 1.0, 'none', None, None, 0.0, None, False, None)),
    ('de_wilcoxon_precision_adjusted', None, 'one', None, ('full', 'de'), (), 'de', None, 'mean', False, 'cell_eval2.metrics.de', 'de_sig_agreement', (), (('measure', 'markedness'),), False, (True, 'higher', 1.0, 'none', None, None, 0.0, None, False, None)),
    ('de_wilcoxon_sig_recall_adjusted', None, 'one', None, ('full', 'de'), (), 'de', None, 'mean', False, 'cell_eval2.metrics.de', 'de_sig_agreement', (), (('measure', 'informedness'),), False, (True, 'higher', 1.0, 'none', None, None, 0.0, None, False, None)),
    ('de_wilcoxon_sig_mcc', None, 'one', None, ('full', 'de'), (), 'de', None, 'mean', False, 'cell_eval2.metrics.de', 'de_sig_agreement', (), (('measure', 'mcc'),), False, (True, 'higher', 1.0, 'none', None, None, 0.0, None, False, None)),
    ('de_wilcoxon_direction_precision', None, 'one', 0.0, ('full', 'de'), (), 'de', None, 'mean', False, 'cell_eval2.metrics.direction', 'de_direction_precision', (), (), False, (True, 'higher', 1.0, 'none', None, None, 0.0, None, False, None)),
    ('de_wilcoxon_direction_sensitivity', None, 'one', 0.0, ('full', 'de'), (), 'de', None, 'mean', False, 'cell_eval2.metrics.direction', 'de_direction_sensitivity', (), (('universe', 'adjudicated'),), False, (True, 'higher', 1.0, 'none', None, None, 0.0, None, False, None)),
    # best_value (field 2) is 'one' for all twelve now: every directional metric is scored,
    # and the derived token maps scored+higher to 'one'. Only the two nsig counts above keep
    # 'none', because they are the only entries with no direction at all.
    ('de_wilcoxon_direction_sensitivity_universe', None, 'one', 0.0, ('full', 'de'), (), 'de', None, 'mean', False, 'cell_eval2.metrics.direction', 'de_direction_sensitivity', (), (('universe', 'all'),), False, (True, 'higher', None, 'none', None, None, -2.0, 2.0, False, None)),
    ('de_wilcoxon_direction_fidelity', None, 'one', None, ('full', 'de'), (), 'de', None, 'mean', False, 'cell_eval2.metrics.direction', 'de_direction_fidelity', (), (), False, (True, 'higher', 1.0, 'none', None, None, 0.0, None, False, None)),
    ('de_wilcoxon_direction_fidelity_raw', None, 'one', None, ('full', 'de'), (), 'de', None, 'mean', False, 'cell_eval2.metrics.direction', 'de_direction_fidelity_raw', (), (), False, (True, 'higher', 1.0, 'none', None, None, 0.0, None, False, None)),
    ('de_wilcoxon_direction_coverage', None, 'one', None, ('full', 'de'), (), 'de', None, 'mean', False, 'cell_eval2.metrics.direction', 'de_direction_coverage', (), (), False, (True, 'higher', None, 'none', None, None, -2.0, 2.0, False, None)),
    ('de_wilcoxon_direction_yield', None, 'one', None, ('full', 'de'), (), 'de', None, 'mean', False, 'cell_eval2.metrics.direction', 'de_direction_yield', (), (), False, (True, 'higher', None, 'none', None, None, -2.0, 2.0, True, None)),
    ('de_wilcoxon_direction_yield_raw', None, 'one', None, ('full', 'de'), (), 'de', None, 'mean', False, 'cell_eval2.metrics.direction', 'de_direction_yield_raw', (), (), False, (True, 'higher', None, 'none', None, None, -2.0, 2.0, False, None)),
    ('de_wilcoxon_direction_fidelity_yield', None, 'one', None, ('full', 'de'), (), 'de', None, 'mean', False, 'cell_eval2.metrics.direction', 'de_direction_fidelity_yield', (), (), False, (True, 'higher', 1.0, 'none', None, None, 0.0, None, False, None)),
    ('de_wilcoxon_direction_fidelity_yield_raw', None, 'one', None, ('full', 'de', 'vcc2026'), (), 'de', None, 'mean', False, 'cell_eval2.metrics.direction', 'de_direction_fidelity_yield_raw', (), (), False, (True, 'higher', 1.0, 'none', None, None, None, None, False, 0.0)),
    ('de_wilcoxon_direction_reach', None, 'one', None, ('full', 'de'), (), 'de', None, 'mean', False, 'cell_eval2.metrics.direction', 'de_direction_reach', (), (('corrected', True), ('universe', 'adjudicated')), False, (True, 'higher', 1.0, 'none', None, None, 0.0, None, False, None)),
    ('de_wilcoxon_direction_reach_raw', None, 'one', None, ('full', 'de', 'vcc2026'), (), 'de', None, 'mean', False, 'cell_eval2.metrics.direction', 'de_direction_reach', (), (('corrected', False), ('universe', 'adjudicated')), False, (True, 'higher', 1.0, 'none', None, None, None, None, False, 0.0)),
    ('de_wilcoxon_direction_reach_unbounded', None, 'one', None, ('full', 'de'), (), 'de', None, 'mean', False, 'cell_eval2.metrics.direction', 'de_direction_reach', (), (('corrected', True), ('universe', 'all')), False, (True, 'higher', None, 'none', None, None, -2.0, 2.0, False, None)),
    ('de_wilcoxon_direction_reach_unbounded_raw', None, 'one', None, ('full', 'de'), (), 'de', None, 'mean', False, 'cell_eval2.metrics.direction', 'de_direction_reach', (), (('corrected', False), ('universe', 'all')), False, (True, 'higher', None, 'none', None, None, -2.0, 2.0, False, None)),
    # Every row above carries agg='mean' (#231): the catalog has exactly one aggregation
    # statistic, so a competition profile can no longer average across two of them. The nine
    # that held 'median' through v0.7.0 moved in both families at once, `add` passing one
    # `agg` per wilcoxon/deseq2 sibling pair. `vcc2026` names the RAW direction pair since
    # #231; the chance-corrected pair stays in full/de.
    # Raw counterpart of sig_mcc, appended last to preserve the existing family order.
    ('de_wilcoxon_sig_jaccard', None, 'one', None, ('full', 'de', 'vcc2026'), (), 'de', None, 'mean', False, 'cell_eval2.metrics.de', 'de_sig_jaccard', (), (), False, (True, 'higher', 1.0, 'none', None, None, None, None, False, 0.0)),
]


def test_de_wilcoxon_family_is_unchanged():
    got = [_spec_fingerprint(s) for n, s in CATALOG.items() if n.startswith("de_wilcoxon_")]
    assert got == GOLDEN_WILCOXON   # ordered list -> also catches a reordering of the family


# ---- the de_deseq2_* family + name-map -----------------------------------------------

def test_deseq2_family_mirrors_wilcoxon_v2_only():
    wil = [n for n in CATALOG if n.startswith("de_wilcoxon_")]
    assert wil, "wilcoxon family missing"
    for wname in wil:
        dname = wname.replace("de_wilcoxon_", "de_deseq2_", 1)
        assert dname in CATALOG, f"{dname} not registered"
        w, d = CATALOG[wname], CATALOG[dname]
        assert _func_id(d) == _func_id(w)          # same metric function + bound params
        # #212: the authoritative scoring policy, not just the token derived from it.
        # `best_value` is kept alongside because it is what the pre-#203 mirror asserted and
        # an equal `scoring` already implies it -- a redundant assertion, deliberately.
        assert d.scoring == w.scoring
        assert d.best_value == w.best_value
        assert d.worst_value == w.worst_value
        assert d.kind == "de" and d.normalization is None
        assert d.v1_name is None                   # v2-only
        assert d.profiles == ()                    # not auto-selected by any profile


def test_deseq2_family_absent_from_profiles():
    for names in PROFILES.values():
        assert not any(n.startswith("de_deseq2_") for n in names)


def test_deseq2_metric_name_maps_and_is_idempotent():
    from cell_eval2.catalog import deseq2_metric_name
    assert deseq2_metric_name("de_wilcoxon_overlap") == "de_deseq2_overlap"
    assert deseq2_metric_name("de_wilcoxon_sig_mcc") == "de_deseq2_sig_mcc"
    assert (deseq2_metric_name("de_wilcoxon_model_direction_match")
            == "de_deseq2_model_direction_match")
    assert deseq2_metric_name("de_deseq2_overlap") == "de_deseq2_overlap"  # idempotent


# ---- backend-driven relabel + reverse-mislabel guard ----------------------------------

def test_effective_de_spec_relabels_under_deseq2():
    from cell_eval2.catalog import deseq2_metric_name
    from cell_eval2.run import _effective_de_spec
    for name in [n for n in CATALOG if n.startswith("de_wilcoxon_")]:
        assert _effective_de_spec(name, "deseq2").name == deseq2_metric_name(name)
        assert _effective_de_spec(name, "pdex").name == name      # rank backend: unchanged
        assert _effective_de_spec(name, "auto").name == name


def test_guard_rejects_deseq2_name_without_deseq2_backend():
    from cell_eval2.run import _guard_deseq2_metric_selection
    with pytest.raises(ValueError, match="require de.backend"):
        _guard_deseq2_metric_selection(["de_deseq2_overlap"], "pdex")
    _guard_deseq2_metric_selection(["de_deseq2_overlap"], "deseq2")   # ok: matching backend
    _guard_deseq2_metric_selection(["de_wilcoxon_overlap"], "pdex")   # ok: normal case


def test_dispatch_de_metrics_guards_all_paths():
    # the guard lives in dispatch_de_metrics — the universal DE chokepoint every path
    # (main / streaming / partition) funnels through — so it fires before prepared_de is
    # touched (passing None proves it guards first).
    from cell_eval2 import EvalConfig
    from cell_eval2.config import DEParams
    from cell_eval2.run import dispatch_de_metrics
    cfg = EvalConfig(pert_col="t", control="ctrl", de=DEParams(backend="pdex"))
    with pytest.raises(ValueError, match="require de.backend"):
        dispatch_de_metrics(["de_deseq2_overlap"], None, cfg)


def test_dispatch_relabels_and_dedups_engine_free():
    # Positive relabel + sibling-dedup WITHOUT the private engine: prepare_de builds the
    # PreparedDE from supplied DE tables, so dispatch only names/dedups (backend picks the
    # label). Covers in CI what the engine-gated end-to-end test can't.
    import polars as pl

    from cell_eval2 import EvalConfig
    from cell_eval2.config import DEParams
    from cell_eval2.de import prepare_de
    from cell_eval2.run import dispatch_de_metrics

    genes = ["g0", "g1", "g2", "g3"]
    tbl = pl.DataFrame({"target": ["P1"] * 4, "feature": genes,
                        "log2_fold_change": [3.0, 2.0, 1.0, 0.2],
                        "p_adj": [0.001, 0.001, 0.001, 0.9]})
    prep = prepare_de(tbl, tbl, control="non-targeting", p_adj_threshold=0.05)

    def run(backend, names):
        cfg = EvalConfig(pert_col="target", control="non-targeting", input_type="counts",
                         metrics="de", de=DEParams(backend=backend, replicate_col="g"))
        return dispatch_de_metrics(names, prep, cfg)

    # rank backend keeps de_wilcoxon_*; deseq2 relabels to de_deseq2_*
    assert {r["metric"] for r in run("pdex", ["de_wilcoxon_overlap"])} == {"de_wilcoxon_overlap"}
    assert {r["metric"] for r in run("deseq2", ["de_wilcoxon_overlap"])} == {"de_deseq2_overlap"}
    # both siblings under deseq2 -> a single de_deseq2_overlap per pert (no duplicate rows)
    rows = run("deseq2", ["de_wilcoxon_overlap", "de_deseq2_overlap"])
    assert [(r["perturbation"], r["metric"]) for r in rows] == [("P1", "de_deseq2_overlap")]


def test_deseq2_and_rank_backends_use_distinct_de_cache_method():
    # Regression (Codex P1): the DE cache keys embed de.method (de_<method>_rank/_table,
    # stream_de_<method>) and the rank cache's params carry no backend. Before method tracked
    # the backend, a deseq2 run keyed under 'wilcoxon' and could load a structurally-identical
    # Wilcoxon rank from a shared non-strict cache. The keys are now disjoint.
    from cell_eval2.config import DEParams
    deseq2_method = DEParams(backend="deseq2", replicate_col="g").method
    assert deseq2_method == "deseq2"
    for rank_backend in ("auto", "gpudge", "pdex", "scanpy"):
        assert DEParams(backend=rank_backend).method != deseq2_method


# ---- end-to-end deseq2 run emits only de_deseq2_* (engine-gated, CPU) -----------------

def test_compute_metrics_deseq2_emits_de_deseq2_names(toy_de_adata):
    pytest.importorskip("deseq2_gpu")
    from cell_eval2 import EvalConfig, compute_metrics
    from cell_eval2.config import DEParams

    real = toy_de_adata(n_ctrl_guides=3, n_pert=2, cells_per=6, n_genes=6, seed=0)
    pred = real.copy()
    cfg = EvalConfig(
        pert_col="target_gene", control="non-targeting", input_type="counts",
        control_source="real", device="cpu", metrics="de",
        de=DEParams(backend="deseq2", replicate_col="guide"),
    )
    df = compute_metrics(pred, real, config=cfg)
    de_metrics = {m for m in df["metric"].unique().to_list() if m.startswith("de_")}
    assert de_metrics, "no DE metrics produced"
    assert all(m.startswith("de_deseq2_") for m in de_metrics), sorted(de_metrics)
    assert not any(m.startswith("de_wilcoxon_") for m in de_metrics)
    # The chance-corrected direction metrics deliberately retain unscoreable NaNs
    # (worst_value=None); every legacy metric in this fixture must remain finite.
    nonfinite = set(df.filter(~df["value"].is_finite())["metric"].to_list())
    assert all(
        m.startswith("de_deseq2_direction_")
        and CATALOG[m].v1_available is False
        and CATALOG[m].worst_value is None
        for m in nonfinite
    )


# ---- deseq2 GPU path (skips without a JAX GPU; runs on a cluster GPU node) -------------

def test_deseq2_gpu_matches_cpu(toy_de_adata):
    pytest.importorskip("deseq2_gpu")
    jax = pytest.importorskip("jax")
    import numpy as np
    try:
        gpus = jax.devices("gpu")
    except RuntimeError:
        gpus = []
    if not gpus:
        pytest.skip("no JAX GPU visible")
    from cell_eval2.deseq2_de import run_deseq2_de

    a = toy_de_adata(n_ctrl_guides=4, n_pert=3, cells_per=8, n_genes=12, seed=1)
    kw = dict(pert_col="target_gene", control="non-targeting",
              replicate_col="guide", input_type="counts")
    cpu = run_deseq2_de(a, **kw, use_gpu=False).sort(["target", "feature"])
    gpu = run_deseq2_de(a, **kw, use_gpu=True).sort(["target", "feature"])
    assert cpu.columns == gpu.columns
    # same (target, feature) rows in the same order -> every comparison below is row-aligned
    assert cpu.select(["target", "feature"]).rows() == gpu.select(["target", "feature"]).rows()
    for col in ("log2_fold_change", "p_value"):
        x, y = cpu[col].to_numpy(), gpu[col].to_numpy()
        # identical finite masks: a CPU-finite/GPU-NaN divergence must FAIL, not be masked away
        assert np.array_equal(np.isfinite(x), np.isfinite(y)), f"finite mask differs: {col}"
        finite = np.isfinite(x)
        assert finite.any(), f"no finite values for {col} — parity test would pass vacuously"
        np.testing.assert_allclose(x[finite], y[finite], rtol=1e-3, atol=1e-4)
    # p_adj also involves independent filtering (discrete gene selection), so compare its
    # values on the finite intersection rather than requiring bit-identical filtering masks.
    pa, pb = cpu["p_adj"].to_numpy(), gpu["p_adj"].to_numpy()
    m = np.isfinite(pa) & np.isfinite(pb)
    assert m.any(), "no finite p_adj to compare"
    np.testing.assert_allclose(pa[m], pb[m], rtol=1e-3, atol=1e-4)
