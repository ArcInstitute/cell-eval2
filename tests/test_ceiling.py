import math
from dataclasses import replace

import polars as pl
import pytest

from cell_eval2 import EvalConfig, compute_ceiling
from cell_eval2.catalog import CATALOG, resolve_metrics
from cell_eval2.ceiling import SB_METRICS, _disjoint_halves, _spearman_brown
from cell_eval2.run import metric_output_names


def test_sb_metrics_only_the_verified_set():
    """SB_METRICS is the hand-curated verified list: no AUC, error, count, or
    v2-native chance-corrected/directional metrics leak in."""
    excluded_names = (
        "de_wilcoxon_pr_auc",
        "de_wilcoxon_roc_auc",
        "expr_mae",
        "expr_mse",
        "expr_mse_unbiased",
        "expr_mse_unbiased_capped",
        "expr_distance_unbiased",
        "expr_mse_unbiased_capped_norm",
        "delta_mae",
        "de_wilcoxon_nsig_counts_real",
        "de_wilcoxon_overlap_adjusted",
        "de_wilcoxon_sig_mcc",
        "de_wilcoxon_lfc_spearman_pos",
        "de_deseq2_overlap",
        # v2-native direction metrics (#187): never validated under doubling, and
        # _universe is unbounded above, which SB's bounded-reliability rules out
        "de_wilcoxon_direction_precision",
        "de_wilcoxon_direction_sensitivity",
        "de_wilcoxon_direction_sensitivity_universe",
    )
    for excluded in excluded_names:
        assert excluded not in SB_METRICS
    # ...and each of those names must still BE a metric. Asserting that a name is absent
    # from SB_METRICS passes trivially once the name no longer exists, so a rename (e.g.
    # #195 renaming the three #187 direction metrics) would silently retire the exclusion
    # it was written to pin. Fail loudly instead, so the rename has to restate the policy.
    gone = [n for n in excluded_names if n not in CATALOG]
    assert not gone, (
        f"the SB exclusion list names metrics that no longer exist: {gone}. If they were "
        "renamed, update this list AND the SB_METRICS comment block in ceiling.py, and "
        "decide explicitly whether the new names are SB exclusions too (they are not "
        "excluded automatically)."
    )
    # The set is pinned EXACTLY, not by sampling a few members. SB_METRICS is a verified
    # list - a metric joins it only once someone has checked that cell_eval2 computes the
    # same quantity the validated cell-eval implementation did - so a metric appearing in
    # it is the thing under test. Spot-checking a handful would let a new arrival inherit
    # a ceiling nobody validated. Changing this literal is the point: it forces the claim
    # to be made deliberately.
    assert SB_METRICS == frozenset(
        {
            "delta_pearson",
            "pds_l1",
            "pds_l2",
            "pds_cosine",
            "de_wilcoxon_overlap",
            "de_wilcoxon_overlap_top50",
            "de_wilcoxon_overlap_top100",
            "de_wilcoxon_overlap_top200",
            "de_wilcoxon_overlap_top500",
            "de_wilcoxon_precision",
            "de_wilcoxon_precision_top50",
            "de_wilcoxon_precision_top100",
            "de_wilcoxon_precision_top200",
            "de_wilcoxon_precision_top500",
            "de_wilcoxon_nsig_spearman",
            "de_wilcoxon_lfc_spearman",
            "de_wilcoxon_direction_match",
            "de_wilcoxon_sig_recall",
        }
    )
    # every listed name is a real catalog metric
    available, _ = resolve_metrics(sorted(SB_METRICS))
    assert set(available) == set(SB_METRICS)


def test_spearman_brown_maps_and_nans():
    agg = pl.DataFrame(
        {"metric": ["delta_pearson", "pds_l1"], "mean": [0.5, 1.0 / 3.0]}
    )
    out = _spearman_brown(agg, ["delta_pearson", "pds_l1", "expr_mae"])
    d = dict(zip(out["metric"].to_list(), out["ceiling"].to_list()))
    assert d["delta_pearson"] == pytest.approx(2.0 / 3.0, abs=1e-9)  # 2*.5/1.5
    assert d["pds_l1"] == pytest.approx(0.5, abs=1e-9)  # 2*(1/3)/(4/3)
    assert math.isnan(d["expr_mae"])  # not an SB metric -> NaN
    # coverage: every requested metric appears exactly once
    assert out["metric"].to_list() == ["delta_pearson", "pds_l1", "expr_mae"]


def test_spearman_brown_nans_nonpositive_reliability():
    """SB (2r/(1+r)) stops being a correction at r <= 0: the formula has its pole at
    r = -1 (division by zero), and just short of it returns nonsense magnitudes
    (r = -0.9 -> -18). A non-positive split-half reliability => no defensible
    ceiling => NaN, and never a raise."""
    agg = pl.DataFrame(
        {
            "metric": [
                "delta_pearson",  # r < 0
                "de_wilcoxon_lfc_spearman",  # r == -1 (would ZeroDivisionError)
                "de_wilcoxon_nsig_spearman",  # r == 0 (boundary: guard is v > 0)
                "pds_l1",  # r < 0
            ],
            "mean": [-0.5, -1.0, 0.0, -0.9],
        }
    )
    out = _spearman_brown(agg, agg["metric"].to_list())
    d = dict(zip(out["metric"].to_list(), out["ceiling"].to_list()))
    for m, v in d.items():
        assert math.isnan(v), f"{m}: expected NaN for non-positive reliability, got {v}"


def test_disjoint_halves_share_no_cells(synthetic_counts_pair):
    _pred, real = synthetic_counts_pair
    a, b = _disjoint_halves(real, pert_col="target", control="non-targeting", seed=0)
    # disjoint: no cell (by name) in both halves
    assert set(a.obs_names).isdisjoint(set(b.obs_names))
    # both halves carry the same perturbations, including the control
    assert set(a.obs["target"].astype(str)) == set(b.obs["target"].astype(str))
    assert "non-targeting" in set(a.obs["target"].astype(str))


def test_compute_ceiling_end_to_end(synthetic_counts_pair):
    """compute_ceiling: only the verified reliability metrics get a ceiling; the
    aggregate covers every selected metric with NaN for the rest."""
    _pred, real = synthetic_counts_pair
    results, agg = compute_ceiling(real, config=EvalConfig(metrics="anndata"), seed=0)

    # only verified SB metrics are computed on the self-split
    computed = set(results["metric"].unique())
    assert computed  # non-empty
    assert computed.issubset(SB_METRICS)
    assert "delta_pearson" in computed

    d = dict(zip(agg["metric"].to_list(), agg["ceiling"].to_list()))
    # Post-guard invariant, asserted TWO-SIDED: an SB ceiling is either NaN (the
    # non-positive-reliability case) or in (0, 1] - SB doubling of an r in (0, 1]
    # can neither exceed 1 nor come out negative. A one-sided `<= 1.0` would let a
    # blown-up near-pole value (r = -0.9 -> -18.0) pass silently.
    for m, v in d.items():
        if m in SB_METRICS:
            assert math.isnan(v) or 0.0 < v <= 1.0 + 1e-9, f"{m}: {v}"
    # error metrics carry no ceiling -> NaN
    assert math.isnan(d["expr_mae"])
    assert math.isnan(d["expr_mse"])
    # coverage: aggregate lists exactly the selected profile's metrics
    available, _ = resolve_metrics("anndata")
    assert set(agg["metric"].to_list()) == set(available)


def test_compute_ceiling_matches_sb_metrics_under_v1_names(synthetic_counts_pair):
    """Every metric in SB_METRICS is corrected regardless of the spelling the config
    emits. Under version="v1" the run labels output with v1 names (pearson_delta,
    discrimination_score_l1), so a canonical-name-only lookup would silently return
    an all-NaN ceiling. SB membership is resolved on the canonical identity instead.
    """
    _pred, real = synthetic_counts_pair
    _results, agg = compute_ceiling(
        real, config=EvalConfig(metrics="anndata", version="v1"), seed=0
    )
    d = dict(zip(agg["metric"].to_list(), agg["ceiling"].to_list()))

    # the ceiling is labeled like the main run's output (v1), not canonically
    for v1_name, canonical in (
        ("pearson_delta", "delta_pearson"),
        ("discrimination_score_l1", "pds_l1"),
        ("mae", "expr_mae"),
    ):
        assert v1_name in d, f"expected v1 output name {v1_name!r}"
        assert canonical not in d, f"canonical {canonical!r} leaked into v1 output"

    # the regression itself: SB metrics are still corrected under the v1 spelling
    for v1_name in ("pearson_delta", "discrimination_score_l1"):
        v = d[v1_name]
        assert not math.isnan(v), f"{v1_name}: SB metric went NaN under v1 names"
        assert 0.0 < v <= 1.0 + 1e-9, f"{v1_name}: {v}"

    # exclusions still hold under the v1 spelling
    assert math.isnan(d["mae"])
    assert math.isnan(d["mse"])


def test_compute_ceiling_labels_metrics_like_the_run(synthetic_counts_pair):
    """ceiling_agg must carry the names the run OUTPUTS, one row per emitted metric.

    Under the deseq2 backend a DE metric and its explicitly-selected de_deseq2_*
    sibling collapse to the ONE name the run emits (run.metric_output_names). Labeling
    the ceiling with its own copy of that rule dropped the collapse and put a duplicate
    row in ceiling_agg, so its metric column no longer matched the run's name-for-name.
    """
    _pred, real = synthetic_counts_pair
    cfg = EvalConfig(metrics=["de_wilcoxon_overlap", "de_deseq2_overlap", "delta_pearson"])
    cfg = replace(cfg, de=replace(cfg.de, backend="deseq2"))
    _results, agg = compute_ceiling(real, config=cfg, seed=0)

    names = agg["metric"].to_list()
    assert names == metric_output_names(cfg)
    assert len(names) == len(set(names)), f"duplicate metric rows in ceiling_agg: {names}"
    # deseq2 DE names are unverified -> uncorrected; the continuous metric still is
    d = dict(zip(names, agg["ceiling"].to_list()))
    assert math.isnan(d["de_deseq2_overlap"])
    assert not math.isnan(d["delta_pearson"])


def test_compute_ceiling_seed_reproducible(synthetic_counts_pair):
    _pred, real = synthetic_counts_pair
    _, a1 = compute_ceiling(real, config=EvalConfig(metrics="anndata"), seed=3)
    _, a2 = compute_ceiling(real, config=EvalConfig(metrics="anndata"), seed=3)
    v1 = a1.sort("metric")["ceiling"].to_numpy()
    v2 = a2.sort("metric")["ceiling"].to_numpy()
    assert ((v1 == v2) | ((v1 != v1) & (v2 != v2))).all()  # equal, NaN==NaN


def test_compute_ceiling_forces_independent_controls(
    synthetic_counts_pair, monkeypatch, tmp_path
):
    """The inner half-split run must use control_source="pred" whatever the caller set.
    Under "real" the pred side's DE reference comes from the real side, so scoring
    half_b against half_a computes BOTH halves' log2FCs against half_a's control -
    shared sampling noise between the two quantities whose agreement is measured, which
    inflates the ceiling. Also pins the overrides that stop the inner run writing over
    the caller's artifacts."""
    _pred, real = synthetic_counts_pair
    seen = {}

    import cell_eval2.ceiling as ceiling_mod

    real_compute = ceiling_mod.compute_metrics

    def spy(pred_ad, real_ad, *, config, **kwargs):
        seen["cfg"] = config
        return real_compute(pred_ad, real_ad, config=config, **kwargs)

    monkeypatch.setattr(ceiling_mod, "compute_metrics", spy)

    # caller explicitly asks for the v2 default that would share one control, and sets an
    # outdir the inner run must not inherit (it would overwrite run_params.yaml there)
    compute_ceiling(
        real,
        config=EvalConfig(
            metrics="anndata", control_source="real", outdir=str(tmp_path / "caller-out")
        ),
        seed=0,
    )

    cfg = seen["cfg"]
    assert cfg.control_source == "pred"  # overridden, not inherited
    assert cfg.outdir is None  # cannot clobber the caller's run_params.yaml
    assert cfg.cache_real is None and cfg.cache_pred is None
