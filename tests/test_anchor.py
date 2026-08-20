import math

import numpy as np
import polars as pl
import pytest

from cell_eval2 import EvalConfig
from cell_eval2.catalog import resolve_metrics

# The derived-metric-safe profile config. `synthetic_counts_pair` + metrics="anndata" raises
# on 7 of 40 split seeds (measured), so every profile-level test here uses the fixture with a
# real effect, on the same terms tests/test_run_derived_shape.py already uses it.
PROFILE_KW = dict(metrics="anndata", pert_col="target", input_type="lognorm",
                  validate_input=False)


def _resolved(**kw):
    from cell_eval2.run import _resolve_config
    cfg = _resolve_config(EvalConfig(**{**PROFILE_KW, **kw}), {})
    available, _ = resolve_metrics(cfg.metrics, version=cfg.version)
    return cfg, list(available)


def test_score_one_split_returns_an_aggregate_and_a_cohort_count(synthetic_pair_with_effect):
    from cell_eval2.anchor import _score_one_split

    _pred, real = synthetic_pair_with_effect
    cfg, available = _resolved()
    agg, counts = _score_one_split(real, cfg, seed=0, metrics=available)

    # SUBSET, not equality. `aggregate_metrics` groups the tidy frame, so a selected metric
    # that emitted NO rows is simply absent. Task 3 is what turns that into a loud failure;
    # asserting equality HERE would make this test fail on thin data for the wrong reason.
    assert set(agg["metric"].to_list()) <= set(available)
    # ...but a SUBSET assertion alone is satisfied by the EMPTY set, so it would pass on an
    # aggregate that produced nothing at all. Pin one stable member end to end: present,
    # finite, and with a positive cohort count.
    row = agg.filter(pl.col("metric") == "expr_mae")
    assert row.height == 1, f"expr_mae missing from the aggregate: {agg['metric'].to_list()}"
    assert math.isfinite(float(row["mean"][0]))
    got = dict(zip(counts["metric"].to_list(), counts["n_perturbations"].to_list()))
    assert got.get("expr_mae", 0) > 0, f"expr_mae scored no perturbations: {got}"
    assert set(counts.columns) == {"metric", "n_perturbations"}
    assert counts.schema["n_perturbations"] == pl.Int64


def test_aggregate_drops_a_metric_that_emitted_no_rows(synthetic_counts_pair):
    """Pins the upstream behaviour Task 3 must defend against: `_build_counts` draws every
    perturbation from the SAME Poisson, so nothing is differentially expressed, de_lfc_nmae's
    real-side gate is empty for every target, it emits no tidy rows, and `aggregate_metrics`
    returns no row for it. If this ever starts failing, upstream began padding absent metrics
    and Task 3's guard can be revisited.

    Explicit metric list, NOT a profile: this fixture has no signal, so a profile carrying
    `expr_mse_unbiased_capped_norm` would raise here on some seeds (measured: 7/40)."""
    from cell_eval2.anchor import _score_one_split
    from cell_eval2.run import _resolve_config

    _pred, real = synthetic_counts_pair
    cfg = _resolve_config(
        EvalConfig(metrics=["de_wilcoxon_lfc_nmae", "de_wilcoxon_overlap"],
                   pert_col="target"), {})
    available, _ = resolve_metrics(cfg.metrics, version=cfg.version)
    agg, _counts = _score_one_split(real, cfg, seed=0, metrics=list(available))

    assert "de_wilcoxon_overlap" in agg["metric"].to_list()
    assert "de_wilcoxon_lfc_nmae" not in agg["metric"].to_list()


def test_score_one_split_forces_independent_controls(synthetic_pair_with_effect, tmp_path,
                                                     monkeypatch):
    """control_source MUST be 'pred' whatever the caller set: under 'real' both halves'
    log2FCs come from half_a's control, sharing the noise between the two quantities whose
    agreement is being measured. Also pins the overrides that stop the inner run writing over
    the caller's artifacts. Spec 6.2."""
    import cell_eval2.anchor as anchor_mod

    _pred, real = synthetic_pair_with_effect
    seen = {}
    inner = anchor_mod.compute_metrics

    def spy(pred_ad, real_ad, *, config, **kwargs):
        seen["cfg"] = config
        return inner(pred_ad, real_ad, config=config, **kwargs)

    monkeypatch.setattr(anchor_mod, "compute_metrics", spy)
    cfg, available = _resolved(control_source="real", outdir=str(tmp_path / "caller-out"),
                               cache_real=str(tmp_path / "cr"))
    anchor_mod._score_one_split(real, cfg, seed=0, metrics=available)

    got = seen["cfg"]
    assert got.control_source == "pred"
    assert got.outdir is None
    assert got.cache_real is None and got.cache_pred is None
    # and the CALLER's config is untouched -- `replace` must not mutate in place
    assert cfg.control_source == "real"


def test_score_one_split_is_seed_reproducible(synthetic_pair_with_effect):
    from cell_eval2.anchor import _score_one_split

    _pred, real = synthetic_pair_with_effect
    cfg, available = _resolved()
    a, _ = _score_one_split(real, cfg, seed=3, metrics=available)
    b, _ = _score_one_split(real, cfg, seed=3, metrics=available)
    assert a.sort("metric").equals(b.sort("metric"))


def test_score_one_split_differs_across_seeds(synthetic_pair_with_effect):
    """Different seeds repartition the same cells, so the aggregate must move. If this passes
    trivially the seed is not reaching _disjoint_halves."""
    from cell_eval2.anchor import _score_one_split

    _pred, real = synthetic_pair_with_effect
    cfg, available = _resolved()
    a, _ = _score_one_split(real, cfg, seed=0, metrics=available)
    b, _ = _score_one_split(real, cfg, seed=11, metrics=available)
    assert not a.sort("metric").equals(b.sort("metric"))


# The five seeds base_seed=0 actually derives, verified against the shipped numpy on this
# branch. PINNED LITERALLY on purpose: a test asserting only "deterministic and distinct"
# passes under an inline `base_seed + i` rule too, which is precisely the refactor
# SEED_DERIVATION exists to make impossible -- and which would silently move every shipped
# anchor. If numpy ever changes SeedSequence, this test must fail loudly rather than let a
# competition anchor drift.
DERIVED_SEEDS_0 = [2968811710, 3677149159, 745650761, 2884920346, 2642120001]


def test_derive_seeds_matches_the_pinned_literal_list():
    from cell_eval2.anchor import _derive_seeds

    assert _derive_seeds(0, 5) == DERIVED_SEEDS_0
    # a prefix, not a fresh draw: n_splits=1 must be the FIRST of the five, so a k=1 probe
    # and split 0 of a k=5 run are the same split
    assert _derive_seeds(0, 1) == DERIVED_SEEDS_0[:1]
    assert _derive_seeds(0, 2) == DERIVED_SEEDS_0[:2]
    assert _derive_seeds(1, 5) != DERIVED_SEEDS_0          # base seed matters
    assert all(isinstance(s, int) for s in _derive_seeds(0, 5))


def test_derive_seeds_rejects_zero_splits():
    from cell_eval2.anchor import _derive_seeds

    with pytest.raises(ValueError, match="n_splits"):
        _derive_seeds(0, 0)


def test_anchor_one_split_equals_the_core(synthetic_pair_with_effect):
    """n_splits=1 must reduce EXACTLY to a single _score_one_split at the FIRST DERIVED
    seed -- not to seed=base_seed. Pins the derivation rule against a later refactor."""
    from cell_eval2.anchor import _score_one_split, compute_replicate_anchor

    _pred, real = synthetic_pair_with_effect
    cfg, available = _resolved()
    direct, _ = _score_one_split(real, cfg, seed=DERIVED_SEEDS_0[0], metrics=available)

    _splits, anchor = compute_replicate_anchor(real, config=EvalConfig(**PROFILE_KW),
                                               base_seed=0, n_splits=1)
    d = dict(zip(direct["metric"].to_list(), direct["mean"].to_list()))
    a = dict(zip(anchor["metric"].to_list(), anchor["replicate"].to_list()))
    assert a == pytest.approx(d)


def test_anchor_is_the_mean_of_the_split_aggregates(synthetic_pair_with_effect):
    """Mean of the five AGGREGATES, not a pool of per-perturbation rows: the derived
    ratio_of_sums member has no per-perturbation column, so there is nothing to pool."""
    from cell_eval2.anchor import compute_replicate_anchor

    _pred, real = synthetic_pair_with_effect
    splits, anchor = compute_replicate_anchor(real, config=EvalConfig(**PROFILE_KW),
                                              base_seed=0, n_splits=5)

    assert splits["split_index"].n_unique() == 5
    assert sorted(set(splits["seed"].to_list())) == sorted(DERIVED_SEEDS_0)
    by_metric = splits.group_by("metric").agg(pl.col("value").mean().alias("m"))
    want = dict(zip(by_metric["metric"].to_list(), by_metric["m"].to_list()))
    got = dict(zip(anchor["metric"].to_list(), anchor["replicate"].to_list()))
    assert got == pytest.approx(want)


def test_anchor_spread_columns_are_recomputed_from_the_splits(synthetic_pair_with_effect):
    """Not `min <= mean <= max`, which holds for ANY three numbers drawn from the data --
    that ordering assertion passes even if `replicate_sd` is a constant. Recompute all four
    statistics from `splits` and compare."""
    from cell_eval2.anchor import SPLIT_HALF_RAW, compute_replicate_anchor

    _pred, real = synthetic_pair_with_effect
    splits, anchor = compute_replicate_anchor(real, config=EvalConfig(**PROFILE_KW),
                                              base_seed=0, n_splits=5)

    for metric, sd, lo, hi in zip(anchor["metric"].to_list(),
                                  anchor["replicate_sd"].to_list(),
                                  anchor["replicate_min"].to_list(),
                                  anchor["replicate_max"].to_list()):
        vals = np.asarray(splits.filter(pl.col("metric") == metric)["value"].to_list(),
                          dtype=float)
        assert len(vals) == 5, f"{metric} has {len(vals)} split rows, expected 5"
        assert sd == pytest.approx(float(vals.std(ddof=0)))   # POPULATION sd, ddof=0
        assert lo == pytest.approx(float(vals.min()))
        assert hi == pytest.approx(float(vals.max()))
    # a fixture whose splits are all identical would make the sd check vacuous
    assert max(anchor["replicate_sd"].to_list()) > 0.0
    assert set(anchor["estimator"].to_list()) == {SPLIT_HALF_RAW}


def test_anchor_covers_every_expected_metric(synthetic_pair_with_effect):
    """One row per name the run OUTPUTS -- not per name that happened to aggregate."""
    from cell_eval2.anchor import compute_replicate_anchor
    from cell_eval2.run import _resolve_config, metric_output_names

    _pred, real = synthetic_pair_with_effect
    cfg = EvalConfig(**PROFILE_KW)
    _splits, anchor = compute_replicate_anchor(real, config=cfg, base_seed=0, n_splits=2)
    assert anchor["metric"].to_list() == sorted(metric_output_names(_resolve_config(cfg, {})))
    # the derived member has no per-perturbation column at all -> its cohort is null, and
    # that must be a NULL rather than a 0 that would read as "scored nothing"
    row = anchor.filter(pl.col("metric") == "expr_mse_unbiased_capped_norm")
    assert row.height == 1
    assert row["n_perturbations_min"][0] is None


def test_anchor_raises_when_a_selected_metric_produced_nothing(synthetic_pair_with_effect,
                                                               monkeypatch):
    """A scored member must never silently leave the anchor.

    Driven by a STUB rather than by a thin fixture. Two reasons: on real data the metric
    that vanishes is de_lfc_nmae, and after Task 4 that name raises from a DIFFERENT guard
    (`_lfc_nmae_raw`), so a fixture-driven test would stop exercising THIS guard without
    failing; and the only fixture that reliably empties a DE gate is the signal-free one,
    which raises seed-dependently for unrelated reasons. The stub drops one expected metric
    from the aggregate, which is exactly the upstream behaviour pinned in Task 2."""
    import cell_eval2.anchor as anchor_mod

    _pred, real = synthetic_pair_with_effect
    inner = anchor_mod._score_one_split

    def drop_one(real_ad, cfg, seed, metrics):
        agg, counts = inner(real_ad, cfg, seed, metrics)
        return agg.filter(pl.col("metric") != "expr_mae"), counts

    monkeypatch.setattr(anchor_mod, "_score_one_split", drop_one)
    with pytest.raises(ValueError, match="no usable value.*expr_mae"):
        anchor_mod.compute_replicate_anchor(real, config=EvalConfig(**PROFILE_KW),
                                            base_seed=0, n_splits=1)


# device and DE backend are PINNED, not left on "auto". The cohort assertions below are exact
# integers at a gate boundary (uniform 3 vs member 4), and "auto" resolves to gpudge on a GPU
# host and pdex on a CPU one -- engines whose DE numbers already differ by ~1e-5 (the F2.2
# rationale in run.py's cache key). Unpinned, this contract would be green on one node and a
# coin flip on another: the same three-environments trap that reddened CI both ways on #264.
# `pdex` is present in CI -- it arrives with the dev group's `cell-eval==0.7.2` -- and it is
# what `auto` already resolves to on a CPU host, so pinning it changes nothing except the
# host-dependence.
NMAE_KW = dict(metrics=["de_wilcoxon_lfc_nmae"], pert_col="target",
               device="cpu", de={"backend": "pdex"})


def test_graded_fixture_has_signal_and_a_half_actually_loses_a_perturbation(
        graded_counts_real):
    """Guards the guard, twice over.

    (1) If this fixture ever stops producing significant genes, every lfc_nmae test below
        would pass vacuously on an empty reference.
    (2) If the uniform core's cohort ever EQUALS the member's here, the spec-6.3 assertion
        below stops discriminating between the right estimator and the wrong one -- it would
        pass with either, which is worse than no test.
    """
    from cell_eval2.anchor import _score_one_split
    from cell_eval2.lfc_nmae_ref import compute_lfc_nmae_reference
    from cell_eval2.run import _resolve_config, compute_metrics

    cfg_in = EvalConfig(**NMAE_KW)
    cfg = _resolve_config(cfg_in, {})
    seed = DERIVED_SEEDS_0[0]

    _res, ref = compute_lfc_nmae_reference(graded_counts_real, config=cfg_in, seed=seed)
    row = ref.filter(pl.col("statistic") == "mean")
    assert row["nmae_ref_raw"][0] is not None
    assert row["n_perturbations"][0] == 4

    tidy = compute_metrics(graded_counts_real, graded_counts_real, config=cfg_in)
    member_n = tidy.filter((pl.col("metric") == "de_wilcoxon_lfc_nmae")
                           & pl.col("value").is_not_null()
                           & pl.col("value").is_not_nan()).height
    assert member_n == 4

    _agg, counts = _score_one_split(graded_counts_real, cfg, seed,
                                    ["de_wilcoxon_lfc_nmae"])
    uniform = dict(zip(counts["metric"].to_list(),
                       counts["n_perturbations"].to_list()))["de_wilcoxon_lfc_nmae"]
    assert uniform < member_n, (
        f"the uniform split-half core scored {uniform} perturbations and the member scored "
        f"{member_n}; with these equal, the cohort test below cannot fail"
    )


def test_lfc_nmae_anchor_uses_the_full_gate_estimator(graded_counts_real):
    from cell_eval2.anchor import FULL_GATE_RAW, SPLIT_HALF_RAW, compute_replicate_anchor

    # PINNED, like NMAE_KW -- an unpinned backend makes this host-dependent
    _splits, anchor = compute_replicate_anchor(
        graded_counts_real,
        config=EvalConfig(**{**NMAE_KW,
                             "metrics": ["de_wilcoxon_lfc_nmae", "de_wilcoxon_overlap"]}),
        base_seed=0, n_splits=2)
    got = dict(zip(anchor["metric"].to_list(), anchor["estimator"].to_list()))
    assert got["de_wilcoxon_lfc_nmae"] == FULL_GATE_RAW
    assert got["de_wilcoxon_overlap"] == SPLIT_HALF_RAW


def test_lfc_nmae_anchor_cohort_equals_the_MEMBERS_own_cohort(graded_counts_real):
    """Spec 6.3. Against the MEMBER's tidy cohort on the same data -- not against the
    reference's self-reported count, which would agree with itself even if the two diverged.
    The companion guard above proves the uniform core scores FEWER here, so this fails if the
    substitution is ever dropped."""
    from cell_eval2.anchor import compute_replicate_anchor
    from cell_eval2.run import compute_metrics

    cfg_in = EvalConfig(**NMAE_KW)
    tidy = compute_metrics(graded_counts_real, graded_counts_real, config=cfg_in)
    member_n = tidy.filter((pl.col("metric") == "de_wilcoxon_lfc_nmae")
                           & pl.col("value").is_not_null()
                           & pl.col("value").is_not_nan()).height

    _splits, anchor = compute_replicate_anchor(graded_counts_real, config=cfg_in,
                                               base_seed=0, n_splits=2)
    row = anchor.filter(pl.col("metric") == "de_wilcoxon_lfc_nmae")
    assert row["n_perturbations_min"][0] == member_n
    assert row["n_perturbations_max"][0] == member_n


def test_lfc_nmae_row_is_INSERTED_even_when_the_aggregate_has_no_such_row(
        graded_counts_real, monkeypatch):
    """The regression this task's shape exists for. `agg.with_columns(when(...))` silently
    no-ops when the metric emitted no tidy rows -- exactly the case that motivates the
    substitution -- and the scored member vanishes from the anchor.

    Stubbed rather than fixture-driven: on THIS fixture the member does aggregate, so a
    plain end-to-end call would pass with either implementation. Dropping the row from the
    aggregate reproduces the empty-gate case deterministically."""
    import cell_eval2.anchor as anchor_mod

    inner = anchor_mod._score_one_split

    def drop_nmae(real_ad, cfg, seed, metrics):
        agg, counts = inner(real_ad, cfg, seed, metrics)
        return (agg.filter(pl.col("metric") != "de_wilcoxon_lfc_nmae"),
                counts.filter(pl.col("metric") != "de_wilcoxon_lfc_nmae"))

    monkeypatch.setattr(anchor_mod, "_score_one_split", drop_nmae)
    _splits, anchor = anchor_mod.compute_replicate_anchor(
        graded_counts_real, config=EvalConfig(**NMAE_KW), base_seed=0, n_splits=1)
    assert anchor["metric"].to_list() == ["de_wilcoxon_lfc_nmae"]
    assert anchor["replicate"][0] > 0.0
    assert anchor["n_perturbations_min"][0] == 4


def test_lfc_nmae_anchor_equals_nmae_ref_raw(graded_counts_real):
    """The substituted value must be nmae_ref_RAW, never the sqrt(2)-corrected column. A
    sqrt(2) here would move every submission's score on this member by 17-23%."""
    from cell_eval2.anchor import compute_replicate_anchor
    from cell_eval2.lfc_nmae_ref import compute_lfc_nmae_reference

    cfg = EvalConfig(**NMAE_KW)
    _res, ref = compute_lfc_nmae_reference(graded_counts_real, config=cfg,
                                           seed=DERIVED_SEEDS_0[0])
    mean_row = ref.filter(pl.col("statistic") == "mean")
    want = mean_row["nmae_ref_raw"][0]
    # The sqrt(2)-corrected sibling, renamed in Task 9. The point of reading it is that
    # the two are DISTINGUISHABLE, so the assertion above can fail.
    corrected = mean_row["nmae_ref_sqrt2"][0]

    _splits, anchor = compute_replicate_anchor(graded_counts_real, config=cfg,
                                               base_seed=0, n_splits=1)
    got = anchor.filter(pl.col("metric") == "de_wilcoxon_lfc_nmae")["replicate"][0]
    assert got == pytest.approx(want)
    assert abs(want - corrected) > 1e-6     # so the assertion above CAN fail


def test_full_real_de_is_computed_once_and_the_SAME_TABLE_reaches_every_split(
        graded_counts_real, monkeypatch):
    """The full-real DE table is seed-invariant, so it must be computed ONCE and reused.

    Counting `anchor._compute_de_side` alone is NOT sufficient and was the first draft's
    hole: `lfc_nmae_ref` imports `_compute_de_side` into its OWN module namespace
    (lfc_nmae_ref.py:53), so a bug where `de_real` never reaches the reference -- and each
    split recomputes the full table inside it -- leaves the anchor's counter at 1. Assert
    three things instead: one call through the anchor's binding, the identical OBJECT handed
    to every reference call, and 2 (not 3) half-table calls per split through the
    reference's own binding."""
    import cell_eval2.anchor as anchor_mod
    import cell_eval2.lfc_nmae_ref as ref_mod

    anchor_calls, ref_calls, seen_de_real = [], [], []
    anchor_inner, ref_inner = anchor_mod._compute_de_side, ref_mod._compute_de_side
    ref_fn = anchor_mod.compute_lfc_nmae_reference

    monkeypatch.setattr(anchor_mod, "_compute_de_side",
                        lambda *a, **k: (anchor_calls.append(1), anchor_inner(*a, **k))[1])
    monkeypatch.setattr(ref_mod, "_compute_de_side",
                        lambda *a, **k: (ref_calls.append(1), ref_inner(*a, **k))[1])

    def ref_spy(*a, **k):
        seen_de_real.append(k.get("de_real"))
        return ref_fn(*a, **k)

    monkeypatch.setattr(anchor_mod, "compute_lfc_nmae_reference", ref_spy)

    n_splits = 3
    anchor_mod.compute_replicate_anchor(graded_counts_real, config=EvalConfig(**NMAE_KW),
                                        base_seed=0, n_splits=n_splits)

    assert len(anchor_calls) == 1, f"full-real DE computed {len(anchor_calls)} times"
    assert len(seen_de_real) == n_splits
    assert seen_de_real[0] is not None, "de_real was never passed to the reference"
    assert all(t is seen_de_real[0] for t in seen_de_real), (
        "each split handed the reference a DIFFERENT full-real table object"
    )
    # 2 per split (half_a, half_b). 3 per split means `de_real` was ignored and the full
    # table was rebuilt inside the reference -- which the counter above cannot see.
    assert len(ref_calls) == 2 * n_splits, (
        f"{len(ref_calls)} half-table DE calls, expected {2 * n_splits}"
    )


def test_empty_lfc_nmae_reference_raises(synthetic_counts_pair):
    """On the signal-free fixture the reference is empty. That must RAISE, not produce a
    null anchor: a null top end silently removes a scored member from the scale."""
    from cell_eval2.anchor import compute_replicate_anchor

    _pred, real = synthetic_counts_pair
    with pytest.raises(ValueError, match="scored no perturbation"):
        compute_replicate_anchor(real, config=EvalConfig(**NMAE_KW), base_seed=0,
                                 n_splits=1)


def test_deseq2_relabelling_still_lands_the_substitution(graded_counts_real, monkeypatch):
    """Under de.backend="deseq2" `_effective_de_spec` relabels the emitted name to
    `de_deseq2_lfc_nmae`. Deriving the substitution's target list from the RESOLVED canonical
    names instead of the OUTPUT names would write the value under `de_wilcoxon_lfc_nmae`, a
    key nothing reads, and the emitted row would keep the uniform core's value. Asserted on
    `_lfc_nmae_names` directly so it needs no deseq2 runtime."""
    from cell_eval2.anchor import _lfc_nmae_names

    assert _lfc_nmae_names(["de_deseq2_lfc_nmae", "de_wilcoxon_overlap"]) == \
        ["de_deseq2_lfc_nmae"]
    assert _lfc_nmae_names(["de_wilcoxon_lfc_nmae"]) == ["de_wilcoxon_lfc_nmae"]
    assert _lfc_nmae_names(["de_wilcoxon_overlap"]) == []
