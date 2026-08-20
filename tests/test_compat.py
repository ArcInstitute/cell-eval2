import warnings

import polars as pl
import pytest

from cell_eval2.compat import MetricsEvaluator, score_agg_metrics
from cell_eval2.compat.utils import split_anndata_on_celltype


def test_compute_returns_describe_shaped_agg(synthetic_pair, tmp_path):
    pred, real = synthetic_pair
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        ev = MetricsEvaluator(adata_pred=pred, adata_real=real, pert_col="target",
                              control_pert="non-targeting", outdir=str(tmp_path),
                              skip_de=True)
        results, agg = ev.compute(profile="vcc")
    # VCC reads: agg.filter(statistic=='mean').select('mae').item()
    assert "statistic" in agg.columns and "mae" in agg.columns
    val = agg.filter(pl.col("statistic") == "mean").select("mae").item()
    assert val >= 0
    assert "mae" in results.columns and "perturbation" in results.columns


def test_compute_writes_prefixed_csvs(synthetic_pair, tmp_path):
    pred, real = synthetic_pair
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        ev = MetricsEvaluator(adata_pred=pred, adata_real=real, pert_col="target",
                              control_pert="non-targeting", outdir=str(tmp_path),
                              prefix="ctA", skip_de=True)
        ev.compute(profile="vcc")
    assert (tmp_path / "ctA_results.csv").exists()
    assert (tmp_path / "ctA_agg_results.csv").exists()


def test_deprecation_warning(synthetic_pair, tmp_path):
    pred, real = synthetic_pair
    with pytest.warns(DeprecationWarning):
        MetricsEvaluator(adata_pred=pred, adata_real=real, pert_col="target",
                         control_pert="non-targeting", outdir=str(tmp_path))


def test_score_agg_metrics_zero_metric():
    # mae is best_value="zero" -> from_baseline = 1 - user/base
    user = pl.DataFrame({"statistic": ["mean"], "mae": [0.5]})
    base = pl.DataFrame({"statistic": ["mean"], "mae": [1.0]})
    out = score_agg_metrics(results_user=user, results_base=base)
    assert out.filter(pl.col("metric") == "mae")["from_baseline"].item() == pytest.approx(0.5)
    assert out.filter(pl.col("metric") == "avg_score")["from_baseline"].item() == pytest.approx(0.5)


def test_score_agg_metrics_missing_statistic_raises():
    user = pl.DataFrame({"statistic": ["mean"], "mae": [0.5]})
    base = pl.DataFrame({"statistic": ["mean"], "mae": [1.0]})
    with pytest.raises(ValueError, match="comparison_statistic"):
        score_agg_metrics(results_user=user, results_base=base, comparison_statistic="median")


def test_split_anndata_on_celltype(synthetic_pair):
    pred, _ = synthetic_pair
    pred.obs["celltype"] = (["X"] * (pred.n_obs // 2)
                            + ["Y"] * (pred.n_obs - pred.n_obs // 2))
    parts = split_anndata_on_celltype(pred, "celltype")
    assert set(parts) == {"X", "Y"}


def test_compat_runs_discrimination_under_legacy(synthetic_pair):
    # compat must force the legacy preset so VCC scores stay bit-parity with
    # cell-eval. With the legacy preset, discrimination_score_l1 runs and its
    # values match a direct legacy-config call.
    pred, real = synthetic_pair
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        ev = MetricsEvaluator(adata_pred=pred, adata_real=real, pert_col="target",
                              control_pert="non-targeting", skip_de=True)
        results, _ = ev.compute(profile="vcc", write_csv=False)
    assert "discrimination_score_l1" in results.columns

    from cell_eval2 import EvalConfig, compute_metrics
    cfg = EvalConfig.from_dict({**EvalConfig.legacy().to_dict(),
                                "metrics": ["discrimination_score_l1"]})
    df = compute_metrics(pred, real, config=cfg)
    direct = {row["perturbation"]: row["value"] for row in df.iter_rows(named=True)}
    compat_vals = {row["perturbation"]: row["discrimination_score_l1"]
                   for row in results.iter_rows(named=True)
                   if row["perturbation"] != "non-targeting"}
    assert compat_vals == pytest.approx(direct)


def _de_pair():
    real = pl.DataFrame({"target": ["GENE1", "GENE1", "GENE2", "GENE2", "GENE3", "GENE3"],
                         "feature": ["g1", "g2", "g3", "g4", "g5", "g6"],
                         "log2_fold_change": [3.0, 2.0, 1.0, 5.0, 2.0, 4.0],
                         "fdr": [0.001, 0.001, 0.001, 0.001, 0.001, 0.001]})
    pred = pl.DataFrame({"target": ["GENE1", "GENE1", "GENE2", "GENE2", "GENE3", "GENE3"],
                         "feature": ["g1", "g9", "g4", "g3", "g5", "g6"],
                         "log2_fold_change": [4.0, 2.0, 5.0, 1.0, 2.0, 4.0],
                         "fdr": [0.001, 0.001, 0.001, 0.001, 0.001, 0.001]})
    return pred, real


def test_compat_skip_de_drops_de_metrics(synthetic_pair):
    pred_ad, real_ad = synthetic_pair
    ev = MetricsEvaluator(pred_ad, real_ad, skip_de=True)
    results, _ = ev.compute(profile="vcc", write_csv=False)
    assert "overlap_at_N" not in results.columns  # DE dropped, no tables needed


def test_compat_threads_de_tables(synthetic_pair):
    pred_ad, real_ad = synthetic_pair
    pred_de, real_de = _de_pair()  # fdr column -> aliased to p_adj on load
    ev = MetricsEvaluator(pred_ad, real_ad, de_pred=pred_de, de_real=real_de)
    results, _ = ev.compute(profile="vcc", write_csv=False)
    assert "overlap_at_N" in results.columns


def test_compat_computes_de_via_pdex(synthetic_pair):
    import warnings
    from cell_eval2.compat import MetricsEvaluator
    pred, real = synthetic_pair   # lognorm; compat forces v1 (lognorm)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        ev = MetricsEvaluator(pred, real, pert_col="target",
                              control_pert="non-targeting", allow_discrete=False)
        results, _agg = ev.compute(profile="de")
    assert results.height > 0   # DE metrics computed, not raised


def test_score_agg_metrics_zero_baseline_no_crash():
    # mae best_value="zero"; base==0 is degenerate -> score clamped to 0.0, no ZeroDivisionError
    user = pl.DataFrame({"statistic": ["mean"], "mae": [0.5]})
    base = pl.DataFrame({"statistic": ["mean"], "mae": [0.0]})
    out = score_agg_metrics(results_user=user, results_base=base)
    assert out.filter(pl.col("metric") == "mae")["from_baseline"].item() == 0.0


def test_score_agg_metrics_one_baseline_no_crash():
    # discrimination_score_l1 best_value="one"; base==1 is degenerate -> 0.0, no ZeroDivisionError
    user = pl.DataFrame({"statistic": ["mean"], "discrimination_score_l1": [0.5]})
    base = pl.DataFrame({"statistic": ["mean"], "discrimination_score_l1": [1.0]})
    out = score_agg_metrics(results_user=user, results_base=base)
    assert out.filter(pl.col("metric") == "discrimination_score_l1")["from_baseline"].item() == 0.0


def test_score_agg_metrics_one_metric_branch():
    # finding 12: the _norm_by_one scoring branch. (0.8-0.6)/(1-0.6) = 0.5
    user = pl.DataFrame({"statistic": ["mean"], "discrimination_score_l1": [0.8]})
    base = pl.DataFrame({"statistic": ["mean"], "discrimination_score_l1": [0.6]})
    out = score_agg_metrics(results_user=user, results_base=base)
    assert out.filter(pl.col("metric") == "discrimination_score_l1")["from_baseline"].item() == pytest.approx(0.5)


def test_score_agg_metrics_base_missing_statistic_raises():
    # finding 9: base lacks the requested comparison_statistic row -> clear ValueError, not OutOfBoundsError
    user = pl.DataFrame({"statistic": ["mean", "std"], "mae": [0.5, 0.1]})
    base = pl.DataFrame({"statistic": ["mean"], "mae": [1.0]})
    with pytest.raises(ValueError, match="baseline"):
        score_agg_metrics(results_user=user, results_base=base, comparison_statistic="std")


def test_compat_skip_metrics_accepts_v1_and_v2_names(synthetic_pair):
    # finding 8: compat emits v1 labels; skip_metrics must work with v2 canonical names too
    pred, real = synthetic_pair
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        ev1 = MetricsEvaluator(pred, real, skip_de=True)
        r_v2, _ = ev1.compute(profile="vcc", skip_metrics=["expr_mae"], write_csv=False)  # v2 name
        ev2 = MetricsEvaluator(pred, real, skip_de=True)
        r_v1, _ = ev2.compute(profile="vcc", skip_metrics=["mae"], write_csv=False)        # v1 name
    assert "mae" not in r_v2.columns
    assert "mae" not in r_v1.columns


def test_compat_de_values_match_direct_v1(synthetic_pair):
    from cell_eval2 import EvalConfig, compute_metrics
    pred_ad, real_ad = synthetic_pair
    pred_de, real_de = _de_pair()  # GENE1/GENE2/GENE3 (aligned in Task 4)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        ev = MetricsEvaluator(pred_ad, real_ad, de_pred=pred_de, de_real=real_de)
        results, _ = ev.compute(profile="vcc", write_csv=False)
    assert "overlap_at_N" in results.columns  # v1 label emitted by compat

    cfg = EvalConfig.from_dict({**EvalConfig.v1().to_dict(), "metrics": ["overlap_at_N"]})
    df = compute_metrics(pred_ad, real_ad, config=cfg, de_pred=pred_de, de_real=real_de)
    direct = {row["perturbation"]: row["value"] for row in df.iter_rows(named=True)}
    compat_vals = {row["perturbation"]: row["overlap_at_N"]
                   for row in results.iter_rows(named=True)
                   if row["overlap_at_N"] is not None}
    assert compat_vals == pytest.approx(direct)


def test_compat_allow_discrete_threaded_to_config(synthetic_pair):
    """allow_discrete must reach the v1 config (was silently dropped). On lognorm-like
    input, forcing counts treatment normalizes the matrix and changes metric values,
    so allow_discrete=True and =False must produce different results."""
    pred, real = synthetic_pair
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        r_auto, _ = MetricsEvaluator(pred, real, skip_de=True,
                                     allow_discrete=False).compute(profile="vcc", write_csv=False)
        r_counts, _ = MetricsEvaluator(pred, real, skip_de=True,
                                       allow_discrete=True).compute(profile="vcc", write_csv=False)
    mae_auto = {r["perturbation"]: r["mae"] for r in r_auto.iter_rows(named=True)}
    mae_counts = {r["perturbation"]: r["mae"] for r in r_counts.iter_rows(named=True)}
    assert mae_auto.keys() == mae_counts.keys()
    # at least one perturbation's mae differs materially -> allow_discrete had an effect
    assert any(abs(mae_auto[k] - mae_counts[k]) > 1e-6 for k in mae_auto)


def test_compute_empty_results_no_crash(synthetic_pair, tmp_path):
    """profile='de' + skip_de=True skips every metric -> empty wide table; the agg
    describe() must not raise on the zero-column frame."""
    pred, real = synthetic_pair
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        ev = MetricsEvaluator(pred, real, skip_de=True, outdir=str(tmp_path))
        results, agg = ev.compute(profile="de", write_csv=False)
    assert results.is_empty() or results.height == 0
    # describe-shaped agg with no metric columns: statistic rows present, no crash
    assert "statistic" in agg.columns
    assert "mean" in agg["statistic"].to_list()


def test_score_agg_metrics_row_order_partitions_zero_then_one():
    import polars as pl
    from cell_eval2.compat import score_agg_metrics
    # Columns deliberately interleave a zero-metric (expr_mae->mae) and one-metrics
    # (delta_pearson->pearson_delta, pds_l1->discrimination_score_l1).
    user = pl.DataFrame({
        "statistic": ["mean"], "mae": [0.5],
        "pearson_delta": [0.5], "mse": [0.5], "discrimination_score_l1": [0.5],
    })
    base = pl.DataFrame({
        "statistic": ["mean"], "mae": [1.0],
        "pearson_delta": [0.0], "mse": [1.0], "discrimination_score_l1": [0.0],
    })
    out = score_agg_metrics(user, base)
    order = out["metric"].to_list()
    # zero-metrics (mae, mse) first, then one-metrics (pearson_delta, disc_l1), then avg.
    assert order == ["mae", "mse", "pearson_delta", "discrimination_score_l1", "avg_score"]


def test_score_agg_metrics_nan_value_clamped_to_zero():
    import polars as pl
    from cell_eval2.compat import score_agg_metrics
    # A NaN agg value entering scoring must clamp to 0.0 (reference _score.py isfinite).
    user = pl.DataFrame({"statistic": ["mean"], "mae": [float("nan")]})
    base = pl.DataFrame({"statistic": ["mean"], "mae": [1.0]})
    out = score_agg_metrics(user, base)
    val = out.filter(pl.col("metric") == "mae")["from_baseline"].item()
    assert val == 0.0


def test_score_agg_metrics_null_value_clamped_to_zero():
    import polars as pl
    from cell_eval2.compat import score_agg_metrics
    # A null (None) agg cell — e.g. a blank cell in a user-supplied CSV — must not
    # crash _norm_by_*; it falls back to 0.0 like NaN.
    user = pl.DataFrame(
        {"statistic": ["mean"], "mae": [None]},
        schema={"statistic": pl.Utf8, "mae": pl.Float64},
    )
    base = pl.DataFrame({"statistic": ["mean"], "mae": [1.0]})
    out = score_agg_metrics(user, base)
    val = out.filter(pl.col("metric") == "mae")["from_baseline"].item()
    assert val == 0.0


def test_score_agg_metrics_declines_every_v2_native_metric():
    """score_agg_metrics reproduces upstream cell-eval, so it must score exactly the metrics
    a v1 run can emit -- and it knows only two normalizations, anchor-0 and anchor-1.

    This is a REGRESSION pin, not a nicety. The selection predicate used to read the derived
    `best_value` token, which is a property of a MUTABLE catalog: enrolling the 20 directional
    metrics in the native scorer flipped their token "none" -> "one" and silently made them
    scorable HERE, with this file byte-frozen. Measured at the time: a v2-shaped aggregate
    moved avg_score 0.5 -> 1.0068, with the anchorless metrics normalized by `(u-b)/(1-b)` --
    a formula for an anchor they do not have -- and their [-2, 2] clamp skipped entirely.
    """
    import polars as pl
    from cell_eval2.catalog import CATALOG
    from cell_eval2.compat import score_agg_metrics

    v2_native = sorted(n for n, s in CATALOG.items() if not s.v1_available)
    assert v2_native, "no v2-native metrics registered -- this test would be vacuous"
    cols = ["expr_mae"] + v2_native
    user = pl.DataFrame({"statistic": ["mean"], **{c: [0.5] for c in cols}})
    base = pl.DataFrame({"statistic": ["mean"], **{c: [1.0] for c in cols}})

    out = score_agg_metrics(user, base)
    scored = set(out["metric"]) - {"avg_score"}
    assert scored == {"expr_mae"}, f"compat scored v2-native metrics: {sorted(scored - {'expr_mae'})}"
    # and the aggregate is the v1 metric alone, not diluted by the declined ones
    assert out.filter(pl.col("metric") == "avg_score")["from_baseline"].item() == 0.5


def test_score_agg_metrics_still_scores_the_anchorless_free_v1_set():
    """The other half: declining v2-native metrics must not decline anything v1 emits. Every
    v1-available scored metric is anchored, so all of them remain scorable here."""
    import polars as pl
    from cell_eval2.catalog import CATALOG
    from cell_eval2.compat import score_agg_metrics

    expected = sorted(n for n, s in CATALOG.items() if s.v1_available and s.scoring.scored)
    assert all(CATALOG[n].scoring.anchor is not None for n in expected), \
        "an anchorless metric became v1-available; compat has no formula for it"
    user = pl.DataFrame({"statistic": ["mean"], **{c: [0.5] for c in expected}})
    base = pl.DataFrame({"statistic": ["mean"], **{c: [1.0] for c in expected}})
    out = score_agg_metrics(user, base)
    assert sorted(set(out["metric"]) - {"avg_score"}) == expected


# --- #252: a guide-level panel through the shim ------------------------------------------------

def _guide_level(pred, real):
    """Relabel both sides' perturbations to construct/guide IDs (`SYMBOL-N`), which is the panel
    shape #248 is about: the labels no longer match any measured gene, so exclusion resolves
    nothing. var_names are untouched, so `GENE1..3` are still present as FEATURES."""
    pred, real = pred.copy(), real.copy()
    for a in (pred, real):
        a.obs["target"] = [p if p == "non-targeting" else f"{p}-1"
                           for p in a.obs["target"].astype(str)]
    return pred, real


def test_compat_guide_level_panel_raises_without_either_knob(synthetic_pair):
    """#252's starting point: since #248 pds_* RAISES when exclusion is on and nothing resolves.
    Correct behaviour -- scoring on would silently exclude nothing -- but through the shim there
    was no way to do either the resolving or the opting out."""
    pred, real = _guide_level(*synthetic_pair)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        ev = MetricsEvaluator(pred, real, skip_de=True, pert_col="target",
                              control_pert="non-targeting")
        with pytest.raises(ValueError):
            ev.compute(profile="vcc", write_csv=False)


def test_compat_target_gene_map_reaches_the_same_result_as_the_native_path(synthetic_pair):
    """#252's acceptance: a guide-level panel through the shim reaches the same result the native
    EvalConfig path does with the same map."""
    from cell_eval2 import EvalConfig, compute_metrics

    pred, real = _guide_level(*synthetic_pair)
    gmap = {f"GENE{i}-1": f"GENE{i}" for i in (1, 2, 3)}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        results, _ = MetricsEvaluator(
            pred, real, skip_de=True, pert_col="target", control_pert="non-targeting",
            target_gene_map=gmap,
        ).compute(profile="vcc", write_csv=False)
    assert "discrimination_score_l1" in results.columns

    cfg = EvalConfig.from_dict({**EvalConfig.legacy().to_dict(),
                                "metrics": ["discrimination_score_l1"],
                                "target_gene_map": gmap})
    direct = {r["perturbation"]: r["value"]
              for r in compute_metrics(pred, real, config=cfg).iter_rows(named=True)}
    got = {r["perturbation"]: r["discrimination_score_l1"]
           for r in results.iter_rows(named=True) if r["perturbation"] != "non-targeting"}
    assert got == pytest.approx(direct)


def test_compat_exclude_target_gene_false_is_the_deliberate_opt_out(synthetic_pair):
    """The other half: a panel whose labels genuinely cannot be mapped must be able to say so
    explicitly rather than be blocked. Scores, where it previously raised."""
    pred, real = _guide_level(*synthetic_pair)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        off, _ = MetricsEvaluator(
            pred, real, skip_de=True, pert_col="target", control_pert="non-targeting",
            exclude_target_gene=False,
        ).compute(profile="vcc", write_csv=False)
    assert "discrimination_score_l1" in off.columns
    assert off["discrimination_score_l1"].null_count() == 0


def test_compat_both_knobs_reach_the_EvalConfig(synthetic_pair, monkeypatch):
    """Pinned at the WIRING seam, not on metric values.

    Measured first, and it is why: `discrimination_score_l1` is a RANK over the panel's
    perturbations, and on this 3-perturbation fixture excluding one of 40 genes moves no rank --
    map+exclusion-on and exclusion-off both give {1.0, 0.667, 0.667}. So a value comparison cannot
    discriminate here, and asserting one would be a test that passes for the wrong reason. What
    #252 is actually about is that the shim can REACH the knobs; what they then do to a value is
    the native path's business, covered by #248's own tests.
    """
    import cell_eval2.compat as compat_mod

    pred, real = _guide_level(*synthetic_pair)
    seen = []
    real_cm = compat_mod.compute_metrics
    monkeypatch.setattr(compat_mod, "compute_metrics",
                        lambda p, r, **kw: seen.append(kw["config"]) or real_cm(p, r, **kw))
    gmap = {f"GENE{i}-1": f"GENE{i}" for i in (1, 2, 3)}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        MetricsEvaluator(pred, real, skip_de=True, pert_col="target",
                         control_pert="non-targeting", target_gene_map=gmap,
                         exclude_target_gene=False).compute(profile="vcc", write_csv=False)
    assert len(seen) == 1
    cfg = seen[0]
    assert cfg.target_gene_map == gmap
    assert cfg.discrimination.exclude_target_gene is False
    # The map is COPIED, so a caller mutating theirs afterwards cannot change a scored config.
    gmap["GENE1-1"] = "ELSEWHERE"
    assert cfg.target_gene_map["GENE1-1"] == "GENE1"


def test_compat_defaults_leave_the_v1_config_untouched(synthetic_pair):
    """The scope note on #252: compat is the v1 byte-identity path. Both new arguments default to
    None, so a gene-level run must be byte-identical to one that never passes them."""
    pred, real = synthetic_pair
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        base, _ = MetricsEvaluator(pred, real, skip_de=True).compute(profile="vcc",
                                                                    write_csv=False)
        same, _ = MetricsEvaluator(pred, real, skip_de=True, target_gene_map=None,
                                   exclude_target_gene=None).compute(profile="vcc",
                                                                     write_csv=False)
    assert base.equals(same)
