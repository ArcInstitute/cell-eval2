"""`prep-real-bundle` end to end, on the synthetic fixture pair.

⚠️ TEST DISCIPLINE FOR THIS WHOLE FILE: build artifacts with the real producer and read the
sidecars back off disk. #276 part C-1 shipped three tests that could not fail -- a membership
check handed a hand-built tuple in profile order, a config_hash fixture that took its expected
value from the payload under test, and an identity test that disabled the clamps it existed to
exercise. All three passed while their subject was broken.
"""
import json
import os

import polars as pl
import pytest

from cell_eval2 import competition
from cell_eval2.config import EvalConfig
from cell_eval2.real_bundle import (BASELINE_AGG, BASELINE_META, MANIFEST,
                                    build_real_bundle, read_real_bundle)


def _cfg(**over):
    """The competition preset, shrunk to what the fixture can satisfy. `metrics` stays
    `vcc2026` and `cache_strict` stays True -- the two fields that decide the rule state."""
    from dataclasses import replace

    return replace(EvalConfig.from_preset("vcc2026"), **over)


def _build(tmp_path, real, baseline_pred, *, cfg=None, **kw):
    return build_real_bundle(real, baseline_pred, config=cfg or _cfg(),
                             outdir=str(tmp_path / "b"), bundle_id="test-bundle-r1", **kw)


def test_the_fixture_yields_a_usable_scale(counts_bundle_inputs):
    """Guard the guard. If the fixture cannot define a baseline -> replicate scale for all six
    members, every other test in this file and in Task 8 fails for a reason that has nothing
    to do with the feature under test -- and it fails as a RAISE from deep inside the anchor,
    which reads like a product bug."""
    import math
    from dataclasses import replace

    from cell_eval2.anchor import compute_replicate_anchor
    from cell_eval2.run import aggregate_metrics_wide, compute_metrics, metric_output_names
    from cell_eval2.score import _replicate_entries

    baseline_pred, real, _sub = counts_bundle_inputs
    cfg = _cfg()
    agg = aggregate_metrics_wide(
        compute_metrics(baseline_pred, real,
                        config=replace(cfg, allow_fractional_counts=True)),
        metrics=metric_output_names(cfg))
    row = agg.filter(pl.col("statistic") == "mean")
    # FINITE, not merely non-null: `aggregate_metrics_wide` writes NaN, not null, for an
    # undefined metric (including a derived value, `run.py:1492`).
    bad = [c for c in metric_output_names(cfg)
           if row[c].item() is None or not math.isfinite(float(row[c].item()))]
    assert not bad, f"the baseline leg yields no usable value for {bad}"

    _splits, frame = compute_replicate_anchor(real, config=cfg, base_seed=0, n_splits=5)
    base_by_name = {c: row[c].item() for c in agg.columns if c != "statistic"}
    entries = _replicate_entries(base_by_name, frame)
    assert sorted(entries) == sorted(competition.competition_members()), (
        f"the fixture defines a usable scale for only {sorted(entries)}")


def test_a_competition_bundle_carries_every_file_and_a_rule_digest(
        counts_bundle_inputs, tmp_path):
    baseline_pred, real, _sub = counts_bundle_inputs
    man = _build(tmp_path, real, baseline_pred)
    root = tmp_path / "b"
    for f in (MANIFEST, BASELINE_AGG, BASELINE_META, "anchor_agg.parquet",
              "anchor_splits.parquet", "anchor_meta.json", "config.yaml"):
        assert (root / f).exists(), f
    assert man["rule_digest"] == competition.competition_digest()
    assert man["rule_mismatches"] == []
    assert man["real_bundle_id"] == "test-bundle-r1"
    assert man["manifest_version"] == 1


def test_the_manifest_carries_the_submission_peers_verbatim(
        counts_bundle_inputs, tmp_path):
    """Every peer is COPIED from the baseline leg's run_meta, never recomputed -- a second
    computation is a second thing that can disagree.

    `input_type_pred_effective` is in this list although #192 removed it from the COMPARISON:
    it is still recorded, and a bundle whose provenance is poorer than the frozen ones' is
    the thing that change must not become."""
    baseline_pred, real, _sub = counts_bundle_inputs
    man = _build(tmp_path, real, baseline_pred)
    meta = json.loads((tmp_path / "b" / BASELINE_META).read_text())
    for field in ("cell_eval2_version", "config_digest", "comparator", "source_fingerprint",
                  "source_fingerprint_strict", "resolved_device", "resolved_de_backend",
                  "input_type_real_effective", "input_type_pred_effective",
                  "de_real_fingerprint"):
        assert man[field] == meta[field], field


def test_the_input_path_is_scrubbed_from_the_bundle(counts_bundle_inputs, tmp_path):
    """The official bundles are distributed; `build_run_meta` stamps the absolute source path,
    and `source` is not in the pairing comparison, so it is replaced with the bundle id.

    ⚠️ The real side must be a PATH ON DISK. `build_run_meta` stamps
    `"<in-memory AnnData>"` for an AnnData object (`baseline.py:770`), so passing the fixture
    directly makes the "no tmp_path in the metadata" assertion pass with the feature absent."""
    baseline_pred, real, _sub = counts_bundle_inputs
    real_path = tmp_path / "CCL_x.real.h5ad"
    real.write_h5ad(real_path)
    build_real_bundle(str(real_path), baseline_pred, config=_cfg(),
                      outdir=str(tmp_path / "b"),
                      bundle_id="test-bundle-r1")
    meta = json.loads((tmp_path / "b" / BASELINE_META).read_text())
    assert meta["source"] == "test-bundle-r1"
    assert "CCL_x.real.h5ad" not in json.dumps(meta)
    assert str(tmp_path) not in json.dumps(meta)


def test_a_non_competition_profile_produces_a_DIAGNOSTIC_bundle(
        counts_bundle_inputs, tmp_path):
    """Built, not refused (Alex 2026-08-13) -- and the reason is recorded so the state is
    diagnosable from the artifact alone.

    ⚠️ `anndata`, not `full`, and the difference is measured rather than stylistic. `full`
    carries `de_wilcoxon_lfc_spearman_neg`, whose split-half replicate on this fixture is
    NEGATIVE (-0.414) while the no-skill baseline arm is positive (+0.105) -- a scale running
    backwards, which `_replicate_entries` correctly refuses for a decisive metric, so the
    build ABORTS before any rule check runs. That refusal is the design and not a bug here:
    `score_metrics` calls the same builder on the same frame, so a bundle whose anchor cannot
    define a scale for one of its own decisive metrics would be refused at score time too.
    "Diagnostic, not refused" is about ENROLMENT; a structurally unusable leg still aborts
    (`test_a_degenerate_ANCHOR_leg_aborts_the_build`). `anndata` is a genuinely wider
    non-competition profile whose members all define a usable scale on this fixture."""
    baseline_pred, real, _sub = counts_bundle_inputs
    man = _build(tmp_path, real, baseline_pred, cfg=_cfg(metrics="anndata"))
    assert man["rule_digest"] is None
    assert any("profile" in m for m in man["rule_mismatches"]), man["rule_mismatches"]


@pytest.mark.parametrize("kw,needle", [({"n_splits": 1}, "n_splits"),
                                       ({"base_seed": 123}, "base_seed")])
def test_a_foreign_anchor_parameter_produces_a_diagnostic_bundle(
        counts_bundle_inputs, tmp_path, kw, needle):
    baseline_pred, real, _sub = counts_bundle_inputs
    man = _build(tmp_path, real, baseline_pred, **kw)
    assert man["rule_digest"] is None
    assert any(needle in m for m in man["rule_mismatches"])


def test_cache_strict_off_produces_a_diagnostic_bundle(counts_bundle_inputs, tmp_path):
    baseline_pred, real, _sub = counts_bundle_inputs
    man = _build(tmp_path, real, baseline_pred, cfg=_cfg(cache_strict=False))
    assert man["rule_digest"] is None
    assert any("cache_strict" in m for m in man["rule_mismatches"])


def test_the_baseline_leg_flip_does_not_make_the_bundle_diagnostic(
        counts_bundle_inputs, tmp_path):
    """⚠️ The regression this test exists for: `cache.config_hash` RETAINS
    `allow_fractional_counts`, and the baseline leg sets it True. Hashing the flipped config
    would return "not the competition" for every bundle ever built, and nothing else in the
    suite would notice -- the bundle still builds, it is just silently diagnostic."""
    baseline_pred, real, _sub = counts_bundle_inputs
    man = _build(tmp_path, real, baseline_pred)
    assert man["rule_digest"] == competition.competition_digest(), man["rule_mismatches"]


def test_a_non_empty_outdir_is_refused_unless_forced(counts_bundle_inputs, tmp_path):
    baseline_pred, real, _sub = counts_bundle_inputs
    _build(tmp_path, real, baseline_pred)
    with pytest.raises(ValueError, match="not empty"):
        _build(tmp_path, real, baseline_pred)
    _build(tmp_path, real, baseline_pred, force=True)          # no raise


@pytest.mark.parametrize("field,needle", [
    ("real_fingerprint", "real_fingerprint"),
    ("semantic_identity", "semantic identity"),
    ("config_hash", "config_hash"),
])
def test_ALL_THREE_legs_are_cross_checked(counts_bundle_inputs, tmp_path,
                                          monkeypatch, field, needle):
    """The leg the old three-sidecar design never checked at all. Each of the three is
    corrupted independently, because a loop that checks only the first two passes a test that
    corrupts only the first.

    ⚠️ The third comparison is `config_hash` on BOTH sides via `cache.config_hash` -- never
    `anchor_meta['config_hash']` against `run_meta['config_digest']`, which are different
    functions over different inputs and unequal for every real artifact."""
    from cell_eval2 import real_bundle

    baseline_pred, real, _sub = counts_bundle_inputs
    orig = real_bundle._anchor_leg
    monkeypatch.setattr(real_bundle, "_anchor_leg",
                        lambda *a, **k: (lambda t: (t[0], t[1], {**t[2], field: "deadbeef"}))(orig(*a, **k)))
    with pytest.raises(ValueError, match=needle):
        _build(tmp_path, real, baseline_pred)


def test_the_config_leg_check_is_hash_vs_hash_not_hash_vs_digest(
        counts_bundle_inputs, tmp_path):
    """The regression that would make the third leg a dead feature: if the producer compared
    the anchor's `config_hash` against the baseline leg's `config_digest`, the two would never
    be equal and EVERY bundle would fail to build. A clean build is the assertion."""
    baseline_pred, real, _sub = counts_bundle_inputs
    man = _build(tmp_path, real, baseline_pred)                     # must not raise
    meta = json.loads((tmp_path / "b" / "anchor_meta.json").read_text())
    base = json.loads((tmp_path / "b" / BASELINE_META).read_text())
    assert meta["config_hash"] != base["config_digest"], (
        "the two stamps happen to be equal here, so this test cannot detect the confusion")
    assert man["rule_digest"] == competition.competition_digest()


def test_membership_reaches_the_rule_check_as_the_EXACT_six(
        counts_bundle_inputs, tmp_path):
    """The anchor frame the producer emits is sorted ALPHABETICALLY while `PROFILES` is in
    catalog-insertion order, so an ordered comparison is False for every real artifact and
    would mark every bundle diagnostic.

    ⚠️ Asserting `got == sorted(got)` and `got != competition_members()` is NOT enough: the
    frame has ten rows and the rule has six, so the inequality is guaranteed however the check
    behaves. What has to be pinned is the set that actually reached the rule check -- the
    manifest's `members` -- and that it is exactly the canonical six.
    """
    baseline_pred, real, _sub = counts_bundle_inputs
    man = _build(tmp_path, real, baseline_pred)
    got = pl.read_parquet(tmp_path / "b" / "anchor_agg.parquet")["metric"].to_list()
    assert got == sorted(got) and len(got) == 10          # the producer's real order + width
    assert man["members"] == sorted(competition.competition_members())
    assert man["rule_digest"] == competition.competition_digest()


def test_a_frame_that_disagrees_with_its_SIDECAR_aborts_the_build(
        counts_bundle_inputs, tmp_path, monkeypatch):
    """⚠️ Dropping a frame row alone is NOT a membership test any more: the sidecar/frame gate
    catches it first and raises, so a bundle never exists to inspect. That gate is the point --
    a manifest publishing one membership while the scoring uses another is worse than a
    diagnostic bundle -- so this asserts the ABORT."""
    from cell_eval2 import real_bundle

    baseline_pred, real, _sub = counts_bundle_inputs
    orig = real_bundle._anchor_leg
    monkeypatch.setattr(
        real_bundle, "_anchor_leg",
        lambda *a, **k: (lambda t: (t[0].filter(pl.col("metric") != "de_wilcoxon_sig_jaccard"),
                                    t[1], t[2]))(orig(*a, **k)))
    with pytest.raises(ValueError, match="membership"):
        _build(tmp_path, real, baseline_pred)


def test_a_SELF_CONSISTENT_narrowed_membership_goes_diagnostic(
        counts_bundle_inputs, tmp_path, monkeypatch):
    """The membership rule check, exercised on an artifact that is internally consistent: drop
    the member from the frame AND the sidecar, so the build gets past the sidecar/frame gate
    and the rule check is the thing that decides. Without this, a membership check that is
    never actually consulted passes every test in the file."""
    from cell_eval2 import real_bundle

    baseline_pred, real, _sub = counts_bundle_inputs
    orig = real_bundle._anchor_leg

    def _short(*a, **k):
        frame, splits, meta = orig(*a, **k)
        frame = frame.filter(pl.col("metric") != "de_wilcoxon_sig_jaccard")
        meta = {**meta, "metric_names": [m for m in meta["metric_names"]
                                         if m != "de_wilcoxon_sig_jaccard"]}
        return frame, splits, meta

    monkeypatch.setattr(real_bundle, "_anchor_leg", _short)
    man = _build(tmp_path, real, baseline_pred)
    assert man["rule_digest"] is None
    assert any("members" in m for m in man["rule_mismatches"]), man["rule_mismatches"]


def test_a_DUPLICATE_sidecar_name_aborts_the_build(counts_bundle_inputs, tmp_path,
                                                   monkeypatch):
    """⚠️ Set equality would miss this: a sidecar naming one metric twice has the same SET as
    a unique frame, so it publishes and is then refused later by `validate_anchor`. Compared
    as sorted lists, matching `anchor.py:674`."""
    from cell_eval2 import real_bundle

    baseline_pred, real, _sub = counts_bundle_inputs
    orig = real_bundle._anchor_leg
    monkeypatch.setattr(
        real_bundle, "_anchor_leg",
        lambda *a, **k: (lambda t: (t[0], t[1],
                                    {**t[2], "metric_names": list(t[2]["metric_names"])
                                     + [t[2]["metric_names"][0]]}))(orig(*a, **k)))
    with pytest.raises(ValueError, match="membership"):
        _build(tmp_path, real, baseline_pred)


@pytest.mark.parametrize("field,bad,needle", [
    ("base_seed", 123, "base_seed"),
    ("n_splits", 1, "n_splits"),
    ("derived_seeds", [9, 9, 9, 9, 9], "derived_seeds"),
    ("bulk_target_sum", 1e6, "bulk_target_sum"),
    # ⚠️ `config_hash` is deliberately NOT here. The build-time leg check compares the
    # anchor's RAW stamp against this run's raw hash and RAISES on a difference, so corrupting
    # it produces a fatal error rather than a diagnostic manifest --
    # `test_ALL_THREE_legs_are_cross_checked` covers it there.
])
def test_the_rule_state_comes_from_the_ANCHOR_not_the_arguments(
        counts_bundle_inputs, tmp_path, monkeypatch, field, bad, needle):
    """⚠️ The anchor leg may be satisfied from the content-addressed cache, so the arguments
    `prep-real-bundle` passed are a REQUEST. Corrupting the returned sidecar while leaving the
    arguments correct must still produce a diagnostic bundle -- if it does not, a stale cache
    entry can receive the current competition digest."""
    from cell_eval2 import real_bundle

    baseline_pred, real, _sub = counts_bundle_inputs
    orig = real_bundle._anchor_leg
    monkeypatch.setattr(real_bundle, "_anchor_leg",
                        lambda *a, **k: (lambda t: (t[0], t[1], {**t[2], field: bad}))(orig(*a, **k)))
    man = _build(tmp_path, real, baseline_pred)          # arguments stay base_seed=0, n_splits=5
    assert man["rule_digest"] is None
    assert any(needle in m for m in man["rule_mismatches"]), man["rule_mismatches"]
    # ...and the manifest copies the ANCHOR's value, not the argument's.
    if field in ("base_seed", "n_splits", "derived_seeds", "bulk_target_sum"):
        assert man[field] == bad


def test_a_foreign_ESTIMATOR_in_the_frame_goes_diagnostic(
        counts_bundle_inputs, tmp_path, monkeypatch):
    """The frozen rule pins each member's estimator; the frame carries it per row. A member
    switched between split-half and the full-real gate answers a different question under the
    same column name."""
    from cell_eval2 import real_bundle
    from cell_eval2.anchor import SPLIT_HALF_RAW

    baseline_pred, real, _sub = counts_bundle_inputs
    orig = real_bundle._anchor_leg

    def _swap(*a, **k):
        frame, splits, meta = orig(*a, **k)
        return (frame.with_columns(
            pl.when(pl.col("metric") == "de_wilcoxon_lfc_nmae")
              .then(pl.lit(SPLIT_HALF_RAW)).otherwise(pl.col("estimator")).alias("estimator")),
            splits, meta)

    monkeypatch.setattr(real_bundle, "_anchor_leg", _swap)
    man = _build(tmp_path, real, baseline_pred)
    assert man["rule_digest"] is None
    assert any("estimator" in m for m in man["rule_mismatches"]), man["rule_mismatches"]


def test_a_SUPPLIED_real_side_DE_table_is_REFUSED(counts_bundle_inputs, tmp_path):
    """codex checkpoint-2 P0. `_baseline_leg` takes `de_real`; `compute_replicate_anchor` has no
    such parameter and recomputes its own full-real DE for the `full_gate_raw` estimator. So a
    supplied table would gate and normalize `de_wilcoxon_lfc_nmae`'s 0 end from one table and its
    1 end from another -- and NOTHING would report it: the anchor's semantic identity does not
    cover a supplied table and the manifest's `de_real_fingerprint` records only the baseline
    leg's, so every gate in this file would still pass. Refused at the door instead."""
    baseline_pred, real, _sub = counts_bundle_inputs
    de = pl.DataFrame([{"target": t, "feature": f, "log2_fold_change": 3.0, "p_adj": 0.001}
                       for t in ("GENE1", "GENE2") for f in ("g0", "g1")])
    # ⚠️ Matched on the REFUSAL's own wording, not on the metric name. This table has two gated
    # genes per target, below `min_gate_size=10`, so WITHOUT the guard the baseline leg's own
    # degeneracy error also names `de_wilcoxon_lfc_nmae` -- and a `match="de_wilcoxon_lfc_nmae"`
    # assertion would have been satisfied by the wrong exception (codex checkpoint-2 round 2).
    with pytest.raises(ValueError, match="SUPPLIED real-side DE table"):
        _build(tmp_path, real, baseline_pred, de_real=de)
    assert not os.path.exists(tmp_path / "b")          # nothing was written


def test_the_supplied_DE_refusal_happens_BEFORE_any_work(counts_bundle_inputs, tmp_path,
                                                          monkeypatch):
    """Independent of the exception text, and of what any later stage happens to raise: poison
    `_baseline_leg` so entering it is itself the failure. An absent output directory does NOT
    prove this on its own -- the baseline leg can enter and raise before anything is published
    (codex checkpoint-2 round 3)."""
    from cell_eval2 import real_bundle

    baseline_pred, real, _sub = counts_bundle_inputs

    def _poison(*a, **k):
        raise AssertionError("the baseline leg ran despite a supplied real-side DE table")

    monkeypatch.setattr(real_bundle, "_baseline_leg", _poison)
    with pytest.raises(ValueError, match="SUPPLIED real-side DE table"):
        _build(tmp_path, real, baseline_pred,
               de_real=pl.DataFrame([{"target": "GENE1", "feature": "g0",
                                      "log2_fold_change": 3.0, "p_adj": 0.001}]))


@pytest.mark.parametrize("force", [False, True])
def test_an_outdir_that_is_a_FILE_is_refused_before_any_work(counts_bundle_inputs, tmp_path,
                                                             monkeypatch, force):
    """Copilot, PR #290. `os.path.isdir` is False for a regular file, so BOTH the preflight and
    the publish-time recheck used to skip it and the failure surfaced as a bare
    `NotADirectoryError` out of `os.rename` (measured) -- after both legs had run. `--force`
    must NOT open this door either: forcing means "replace that bundle", not "delete that
    file". `_baseline_leg` is poisoned so entering it is itself the failure."""
    from cell_eval2 import real_bundle

    baseline_pred, real, _sub = counts_bundle_inputs
    squatter = tmp_path / "b"
    squatter.write_text("not a bundle")

    def _poison(*a, **k):
        raise AssertionError("the baseline leg ran against a non-directory outdir")

    monkeypatch.setattr(real_bundle, "_baseline_leg", _poison)
    with pytest.raises(ValueError, match="not a directory"):
        _build(tmp_path, real, baseline_pred, force=force)
    assert squatter.read_text() == "not a bundle"          # and it was not touched


def test_the_bundle_subcommand_offers_no_de_real_FLAG():
    """The other half of the refusal: a flag that only ever errors is worse than no flag. Both
    OTHER subcommands that take one must keep it, or "exactly one subparser lost it" is a claim
    this test does not actually make (codex checkpoint-2 round 2)."""
    import argparse

    from cell_eval2.cli import _build_parser

    sub = [a for a in _build_parser()._actions
           if isinstance(a, argparse._SubParsersAction)][0]

    def opts(name):
        return {o for act in sub.choices[name]._actions for o in act.option_strings}

    assert "--de-real" not in opts("prep-real-bundle")
    for keeps in ("run", "baseline"):
        assert "--de-real" in opts(keeps), keeps


def test_read_real_bundle_round_trips(counts_bundle_inputs, tmp_path):
    baseline_pred, real, _sub = counts_bundle_inputs
    man = _build(tmp_path, real, baseline_pred)
    b = read_real_bundle(str(tmp_path / "b"))
    assert b.manifest == man
    assert os.path.exists(b.baseline_agg)
    assert b.baseline_meta["source"] == "test-bundle-r1"


def test_a_directory_without_a_manifest_is_refused(tmp_path):
    (tmp_path / "empty").mkdir()
    with pytest.raises(ValueError, match="manifest.json"):
        read_real_bundle(str(tmp_path / "empty"))


@pytest.mark.parametrize("drop", ["real_fingerprint", "anchor_digest", "estimators",
                                  "members", "profile", "created_utc", "rule_mismatches"])
def test_a_manifest_missing_ANY_group_is_refused(counts_bundle_inputs, tmp_path,
                                                 drop):
    """Shape validation has to be exhaustive because every consumer reads the manifest with
    `.get()`, and `.get()` returns None on BOTH sides of a comparison when a field is absent
    -- which compares equal. Built by the producer, then edited, so the test exercises the
    real field names rather than the ones the checker happens to expect."""
    baseline_pred, real, _sub = counts_bundle_inputs
    _build(tmp_path, real, baseline_pred)
    path = tmp_path / "b" / MANIFEST
    man = json.loads(path.read_text())
    del man[drop]
    path.write_text(json.dumps(man))
    with pytest.raises(ValueError, match=drop):
        read_real_bundle(str(tmp_path / "b"))


def test_a_foreign_manifest_version_is_refused(counts_bundle_inputs, tmp_path):
    baseline_pred, real, _sub = counts_bundle_inputs
    _build(tmp_path, real, baseline_pred)
    path = tmp_path / "b" / MANIFEST
    man = json.loads(path.read_text())
    man["manifest_version"] = 999
    path.write_text(json.dumps(man))
    with pytest.raises(ValueError, match="manifest_version"):
        read_real_bundle(str(tmp_path / "b"))


def test_the_bundle_holds_EXACTLY_the_declared_file_set(counts_bundle_inputs, tmp_path):
    """The contract is these files and no others."""
    baseline_pred, real, _sub = counts_bundle_inputs
    _build(tmp_path, real, baseline_pred)
    assert sorted(os.listdir(tmp_path / "b")) == sorted([
        MANIFEST, BASELINE_AGG, BASELINE_META, "config.yaml",
        "anchor_agg.parquet", "anchor_splits.parquet", "anchor_meta.json"])


def test_the_baseline_leg_clears_outdir_and_cache_pred(counts_bundle_inputs, tmp_path,
                                                       monkeypatch):
    """⚠️ The file-set test above CANNOT cover this: `_cfg()` leaves both fields at their
    default `None`, so deleting both clears is invisible to it. Hand the builder NON-NULL
    sentinels and inspect the config `compute_metrics` actually received.

    `outdir` matters because the CLI maps `-o` onto `config.outdir`, so a leg that kept it
    writes `run_params.yaml` (`run.py:1163`) into the bundle BEFORE the gates have finished.
    `cache_pred` matters because two structurally identical baseline arms share a
    metadata-only fingerprint under a non-strict config, so a shared pred cache can serve one
    bundle's numbers to another."""
    from dataclasses import replace

    from cell_eval2 import real_bundle

    seen = {}
    orig = real_bundle.compute_metrics

    def _spy(pred_, real_, *, config, **kw):
        seen["cfg"] = config
        return orig(pred_, real_, config=config, **kw)

    monkeypatch.setattr(real_bundle, "compute_metrics", _spy)
    baseline_pred, real, _sub = counts_bundle_inputs
    leaky = replace(_cfg(), outdir=str(tmp_path / "leak"), cache_pred=str(tmp_path / "cp"))
    build_real_bundle(real, baseline_pred, config=leaky, outdir=str(tmp_path / "b"),
                      bundle_id="test-bundle-r1")
    assert seen["cfg"].outdir is None, "the baseline leg kept outdir"
    assert seen["cfg"].cache_pred is None, "the baseline leg kept cache_pred"
    assert seen["cfg"].allow_fractional_counts is True     # the one flip it MUST keep
    assert seen["cfg"].cache_real == leaky.cache_real       # ...and the one it must not touch
    assert not (tmp_path / "leak").exists(), "run_params.yaml leaked out of the leg"


@pytest.mark.parametrize("gate,mutate,needle", [
    ("anchor control source", "control_source_effective", "control_source_effective"),
])
def test_the_control_source_gate_aborts_the_build(counts_bundle_inputs, tmp_path,
                                                  monkeypatch, gate, mutate, needle):
    """Spec acceptance §7.14. A shared-control anchor would inflate the top of the scale, so
    the build ABORTS rather than producing a diagnostic bundle."""
    from cell_eval2 import real_bundle

    baseline_pred, real, _sub = counts_bundle_inputs
    orig = real_bundle._anchor_leg
    monkeypatch.setattr(real_bundle, "_anchor_leg",
                        lambda *a, **k: (lambda t: (t[0], t[1], {**t[2], mutate: "real"}))(orig(*a, **k)))
    with pytest.raises(ValueError, match=needle):
        _build(tmp_path, real, baseline_pred)


def test_a_degenerate_BASELINE_leg_aborts_the_build(counts_bundle_inputs, tmp_path,
                                                    monkeypatch):
    """Spec acceptance §7.14. `score_metrics` would refuse the artifact anyway; failing at
    build costs seconds instead of a campaign."""
    from cell_eval2 import real_bundle

    baseline_pred, real, _sub = counts_bundle_inputs
    monkeypatch.setattr(real_bundle, "_degenerate_metrics",
                        lambda agg, **kw: [{"metric": "pds_cosine", "value": 1.0,
                                            "reason": "forced", "decisive": True}])
    with pytest.raises(ValueError, match="degenerate"):
        _build(tmp_path, real, baseline_pred)


def test_a_degenerate_ANCHOR_leg_aborts_the_build(counts_bundle_inputs, tmp_path,
                                                  monkeypatch):
    """Spec acceptance §7.14, the other end. `_replicate_entries` raises for a DECISIVE member
    with no headroom, and every vcc2026 member is decisive -- so collapsing one member's
    replicate onto its baseline must stop the build."""
    from cell_eval2 import real_bundle

    baseline_pred, real, _sub = counts_bundle_inputs
    orig = real_bundle._anchor_leg

    def _flat(*a, **k):
        frame, splits, meta = orig(*a, **k)
        return (frame.with_columns(
            pl.when(pl.col("metric") == "pds_cosine").then(pl.lit(-1e9))
              .otherwise(pl.col("replicate")).alias("replicate")), splits, meta)

    monkeypatch.setattr(real_bundle, "_anchor_leg", _flat)
    with pytest.raises(ValueError, match="no usable replicate scale"):
        _build(tmp_path, real, baseline_pred)


def test_the_publish_paths(counts_bundle_inputs, tmp_path, monkeypatch):
    """⚠️ The rollback logic existed only in a scratchpad. These are its permanent tests --
    without them the whole move-aside/restore block could be replaced by `rmtree` + `rename`
    and the suite would stay green.

    Four states: the final rename fails and the OLD bundle comes back; restoration ALSO fails
    and the backup survives with its path named; a target that appears mid-build is refused
    without `--force`; and an initially-EMPTY directory is accepted."""
    import os
    import shutil

    from cell_eval2 import real_bundle

    baseline_pred, real, _sub = counts_bundle_inputs
    out = tmp_path / "b"

    # (a) an initially-empty directory is fine -- it was allowed at preflight, so refusing it
    #     at publish would fail a build that was never in doubt.
    out.mkdir()
    _build(tmp_path, real, baseline_pred)
    assert (out / MANIFEST).exists()

    # ⚠️ BOTH failure cases target the same destination, `out`, so the fake must distinguish
    # them by CALL COUNT, not by destination. A fake that fails every `dst == out` rename fails
    # the restoration too, which lands in case (c) and makes case (b) assert the wrong
    # exception type. And do NOT monkeypatch `shutil.rmtree` to protect the backup: keeping it
    # is the behaviour under test, so stubbing the deletion makes the survival assertion
    # vacuous -- the implementation keeps it by clearing its own local before raising.
    first = json.loads((out / MANIFEST).read_text())
    real_rename = os.rename

    def _fail_n_times(n):
        state = {"left": n}

        def _fake(src, dst):
            if str(dst) == str(out) and state["left"] > 0:
                state["left"] -= 1
                raise OSError("forced")
            return real_rename(src, dst)

        return _fake

    # (b) ONLY the stage -> out rename fails; the backup -> out restoration then succeeds,
    #     so the previous bundle comes back and the original OSError propagates.
    monkeypatch.setattr(os, "rename", _fail_n_times(1))
    with pytest.raises(OSError):
        _build(tmp_path, real, baseline_pred, force=True)
    monkeypatch.undo()
    assert json.loads((out / MANIFEST).read_text()) == first, "the old bundle was not restored"
    assert not [p for p in os.listdir(tmp_path) if p.startswith(".real-bundle-old-")], (
        "a successful restoration must leave no backup behind")

    # (c) the restoration fails as well -> the backup SURVIVES and the error names its path.
    monkeypatch.setattr(os, "rename", _fail_n_times(2))
    with pytest.raises(ValueError, match="intact at") as excinfo:
        _build(tmp_path, real, baseline_pred, force=True)
    monkeypatch.undo()
    kept = [p for p in os.listdir(tmp_path) if p.startswith(".real-bundle-old-")]
    assert len(kept) == 1, f"the backup was destroyed even though restoration failed: {kept}"
    surviving = tmp_path / kept[0]
    # The PATH IN THE MESSAGE must be the one that actually survived -- a message naming a
    # stale or already-deleted path leaves the bundle unrecoverable in practice.
    assert str(surviving) in str(excinfo.value)
    assert json.loads((surviving / MANIFEST).read_text()) == first
    shutil.rmtree(surviving)                     # this test owns the cleanup, not the code

    # (d) a NON-EMPTY target appearing mid-build is refused without force
    fresh = tmp_path / "c"
    orig_leg = real_bundle._anchor_leg

    def _appear(*a, **k):
        fresh.mkdir(exist_ok=True)
        (fresh / "squatter.txt").write_text("x")
        return orig_leg(*a, **k)

    monkeypatch.setattr(real_bundle, "_anchor_leg", _appear)
    with pytest.raises(ValueError, match="became non-empty"):
        build_real_bundle(real, baseline_pred, config=_cfg(), outdir=str(fresh),
                          bundle_id="c")
    # ...and the squatter is INTACT. Refusing while having already destroyed what was there
    # would be the same loss the refusal exists to prevent.
    assert (fresh / "squatter.txt").read_text() == "x"


def test_the_CLI_builds_a_bundle_and_PRINTS_its_classification(
        counts_bundle_inputs, tmp_path, capsys):
    """⚠️ Everything else in this file calls `build_real_bundle` directly, so the subparser,
    the argument mapping, the dispatch and the printed classification could all be absent and
    the suite would stay green. The print is load-bearing by design (spec §4.4): it is what
    makes a miscomputed rule check visible on the first build rather than after weeks of
    silently un-enrolled submissions."""
    from cell_eval2.cli import main

    baseline_pred, real, _sub = counts_bundle_inputs
    rp, pp = tmp_path / "r.h5ad", tmp_path / "p.h5ad"
    real.write_h5ad(rp)
    baseline_pred.write_h5ad(pp)
    main(["prep-real-bundle", "--preset", "vcc2026", "--real", str(rp), "--baseline", str(pp),
          "-o", str(tmp_path / "b"), "--id", "vcc2026-CCL_x-r1"])
    assert "competition bundle" in capsys.readouterr().out
    assert json.loads((tmp_path / "b" / MANIFEST).read_text())["real_bundle_id"] \
        == "vcc2026-CCL_x-r1"

    # `anndata` rather than `full`: see test_a_non_competition_profile_produces_a_DIAGNOSTIC_bundle
    main(["prep-real-bundle", "--preset", "vcc2026", "--profile", "anndata", "--real", str(rp),
          "--baseline", str(pp), "-o", str(tmp_path / "d")])
    out = capsys.readouterr().out
    assert "DIAGNOSTIC bundle" in out and "profile" in out
    # ...and --id defaults to the basename of -o.
    assert json.loads((tmp_path / "d" / MANIFEST).read_text())["real_bundle_id"] == "d"
