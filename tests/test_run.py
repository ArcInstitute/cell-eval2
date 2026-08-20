import math
import polars as pl
import pytest

from cell_eval2 import EvalConfig, compute_metrics
from cell_eval2.cache import CacheStore
from cell_eval2.config import DEParams
from cell_eval2.run import _prepare_de_cached, aggregate_metrics


def test_compute_metrics_tidy_long(synthetic_pair):
    pred, real = synthetic_pair
    df = compute_metrics(pred, real, metrics=["mae"],
                         pert_col="target", control="non-targeting",
                         input_type="lognorm")
    assert df.columns == ["perturbation", "metric", "value"]
    assert set(df["metric"].unique()) == {"expr_mae"}
    assert set(df["perturbation"].unique()) == {"GENE1", "GENE2", "GENE3"}


def test_compute_metrics_writes_run_params(synthetic_pair, tmp_path):
    pred, real = synthetic_pair
    compute_metrics(
        pred, real,
        config=EvalConfig(metrics=["mae"], outdir=str(tmp_path), input_type="lognorm"),
    )
    assert (tmp_path / "run_params.yaml").exists()


def test_aggregate_metrics_mean(synthetic_pair):
    pred, real = synthetic_pair
    df = compute_metrics(pred, real, metrics=["mae"], input_type="lognorm")
    agg = aggregate_metrics(df)
    # `agg` records WHICH statistic the `mean` column holds per metric (issue #195). Since
    # #231 every shipped entry says "mean", but the column is emitted regardless: `agg` is
    # still a real per-metric field and `aggregate_metrics` still honours a median declared
    # on one. The column is named `mean` whatever the statistic -- renaming it would break
    # compat, score.py and every published artifact.
    assert set(agg.columns) == {"metric", "mean", "agg"}
    assert agg.filter(pl.col("metric") == "expr_mae")["mean"].item() >= 0


def test_aggregate_metrics_skips_nan():
    # One degenerate pert (NaN) must not poison the metric's aggregate mean.
    df = pl.DataFrame(
        {
            "perturbation": ["p1", "p2", "p3"],
            "metric": ["de_wilcoxon_roc_auc"] * 3,
            "value": [0.8, 0.6, float("nan")],
        },
        schema={"perturbation": pl.Utf8, "metric": pl.Utf8, "value": pl.Float64},
    )
    agg = aggregate_metrics(df)
    mean = agg.filter(pl.col("metric") == "de_wilcoxon_roc_auc")["mean"].item()
    assert mean == pytest.approx(0.7)  # mean of [0.8, 0.6], NaN skipped


def test_aggregate_metrics_all_nan_stays_nan():
    # A metric whose every pert is NaN aggregates to NaN (not null), so downstream
    # arithmetic in score_agg_metrics is unaffected and it matches reference .describe().
    df = pl.DataFrame(
        {
            "perturbation": ["p1", "p2"],
            "metric": ["de_wilcoxon_pr_auc"] * 2,
            "value": [float("nan"), float("nan")],
        },
        schema={"perturbation": pl.Utf8, "metric": pl.Utf8, "value": pl.Float64},
    )
    agg = aggregate_metrics(df)
    val = agg.filter(pl.col("metric") == "de_wilcoxon_pr_auc")["mean"].item()
    assert val is not None and math.isnan(val)


def test_compute_metrics_unknown_override_raises(synthetic_pair):
    pred, real = synthetic_pair
    with pytest.raises(ValueError, match="unknown compute_metrics override"):
        compute_metrics(pred, real, metrics=["mae"], pert_col="target",
                        control="non-targeting", input_type="lognorm", bogus_kwarg=123)


def test_discrimination_dispatch_runs_under_legacy(synthetic_pair):
    pred, real = synthetic_pair
    cfg = EvalConfig.legacy()
    cfg = EvalConfig.from_dict({
        **cfg.to_dict(), "metrics": ["discrimination_score_l1"], "input_type": "lognorm",
    })
    df = compute_metrics(pred, real, config=cfg)
    assert set(df["metric"].unique()) == {"discrimination_score_l1"}
    assert set(df["perturbation"].unique()) == {"GENE1", "GENE2", "GENE3"}
    # PDS scores live in (1/n, 1]; never negative.
    assert df["value"].min() > 0.0
    assert df["value"].max() <= 1.0 + 1e-12


def test_mae_dispatch_unchanged(synthetic_pair):
    # mae's signature lacks control_source/genes/etc.; signature-filtering must
    # still call it correctly and produce the same values as a direct call.
    pred, real = synthetic_pair
    from cell_eval2.metrics import mae
    direct = mae(pred=pred, real=real, pert_col="target", control="non-targeting")
    df = compute_metrics(pred, real, metrics=["mae"], pert_col="target",
                         control="non-targeting", input_type="lognorm")
    got = {row["perturbation"]: row["value"]
           for row in df.iter_rows(named=True) if row["metric"] == "expr_mae"}
    assert got == pytest.approx(direct)


def test_discrimination_variants_differ(synthetic_pair):
    # l1 and cosine are distinct distances -> generally distinct scores; both run.
    pred, real = synthetic_pair
    df = compute_metrics(
        pred, real,
        metrics=["discrimination_score_l1", "discrimination_score_cosine"],
        input_type="lognorm",
    )
    assert set(df["metric"].unique()) == {"pds_l1", "pds_cosine"}


def test_anndata_profile_emits_new_metrics(synthetic_pair):
    pred, real = synthetic_pair
    # default config: version=v2, control_source="real"; declare lognorm for the fixture
    df = compute_metrics(pred, real, metrics="anndata", input_type="lognorm")
    metrics = set(df["metric"].unique())
    assert {"delta_pearson", "expr_mse", "delta_mae", "delta_mse"} <= metrics
    for m in ("delta_pearson", "expr_mse", "delta_mae", "delta_mse"):
        vals = df.filter(pl.col("metric") == m)["value"].to_list()
        assert set(df.filter(pl.col("metric") == m)["perturbation"]) == {"GENE1", "GENE2", "GENE3"}
        assert all(math.isfinite(v) for v in vals)  # finite under control_source="real" (v2)


def test_delta_metrics_emit_v1_names_under_v1(synthetic_pair):
    pred, real = synthetic_pair
    cfg = EvalConfig.v1()
    cfg = EvalConfig.from_dict({
        **cfg.to_dict(),
        "metrics": ["pearson_delta", "mse", "mae_delta", "mse_delta"],
        "input_type": "lognorm",
    })
    df = compute_metrics(pred, real, config=cfg)
    assert set(df["metric"].unique()) == {"pearson_delta", "mse", "mae_delta", "mse_delta"}
    assert set(df["perturbation"].unique()) == {"GENE1", "GENE2", "GENE3"}


def test_dict_overrides_for_nested_dataclasses_are_coerced(synthetic_pair):
    # EvalConfig.from_dict accepts dict-valued `discrimination`/`filter`; the
    # compute_metrics override path must coerce them too, not leave a raw dict
    # that later cfg.discrimination.<attr> access chokes on (regression).
    pred, real = synthetic_pair
    df = compute_metrics(
        pred, real,
        metrics=["mae", "discrimination_score_l1"],
        discrimination={"rank_denominator": "n"},
        filter={"filter_gene_min_cpm_cell": None},
        input_type="lognorm",
    )
    assert set(df["metric"].unique()) == {"expr_mae", "pds_l1"}


def test_resolve_config_coerces_nested_dicts_on_config_object():
    # EvalConfig(filter=/discrimination=/de={...}) built directly (not via from_dict)
    # must be coerced so later cfg.filter.filter_gene_min_cpm_cell /
    # cfg.discrimination.* / cfg.de.* work.
    from cell_eval2.config import DEParams, DiscriminationParams, FilterParams
    from cell_eval2.run import _resolve_config
    cfg = EvalConfig(filter={"filter_gene_min_cpm_cell": 5.0}, discrimination={"distance": "l2"},
                     de={"nan_lfc_policy": "keep"})
    out = _resolve_config(cfg, {})
    assert isinstance(out.filter, FilterParams) and out.filter.filter_gene_min_cpm_cell == 5.0
    assert isinstance(out.discrimination, DiscriminationParams) and out.discrimination.distance == "l2"
    assert isinstance(out.de, DEParams) and out.de.nan_lfc_policy == "keep"


def _de_pair():
    # targets must match the synthetic_pair perturbations {GENE1, GENE2, GENE3}
    real = pl.DataFrame({"target": ["GENE1", "GENE1", "GENE2", "GENE2", "GENE3", "GENE3"],
                         "feature": ["g1", "g2", "g3", "g4", "g5", "g6"],
                         "log2_fold_change": [3.0, 2.0, 1.0, 5.0, 2.0, 4.0],
                         "p_adj": [0.001, 0.001, 0.001, 0.001, 0.001, 0.001]})
    pred = pl.DataFrame({"target": ["GENE1", "GENE1", "GENE2", "GENE2", "GENE3", "GENE3"],
                         "feature": ["g1", "g9", "g4", "g3", "g5", "g6"],
                         "log2_fold_change": [4.0, 2.0, 5.0, 1.0, 2.0, 4.0],
                         "p_adj": [0.001, 0.001, 0.001, 0.001, 0.001, 0.001]})
    return pred, real


def test_compute_metrics_de_only(synthetic_pair):
    pred_ad, real_ad = synthetic_pair
    pred_de, real_de = _de_pair()
    cfg = EvalConfig(metrics=["overlap_at_N"], de={"nan_lfc_policy": "keep"},
                     input_type="lognorm")
    df = compute_metrics(pred_ad, real_ad, config=cfg, de_pred=pred_de, de_real=real_de)
    assert set(df["metric"].unique()) == {"de_wilcoxon_overlap"}
    assert df.filter(pl.col("perturbation") == "GENE1")["value"][0] == pytest.approx(0.5)


def test_de_metric_without_tables_computes(synthetic_pair):
    pred_ad, real_ad = synthetic_pair
    cfg = EvalConfig(metrics=["overlap_at_N"], input_type="lognorm", de={"backend": "scanpy"})
    df = compute_metrics(pred_ad, real_ad, config=cfg)
    assert "de_wilcoxon_overlap" in set(df["metric"])


def test_de_compute_validates_backed_input_in_cache_mode(tmp_path, synthetic_counts_pair):
    # A path (backed) input whose declared input_type is wrong must still be validated
    # before its DE table is computed on a cache miss — the cache no longer makes X
    # "unused" once DE is computed from it (Copilot #1). Counts data declared lognorm.
    pred, real = synthetic_counts_pair
    pp, rp = tmp_path / "p.h5ad", tmp_path / "r.h5ad"
    pred.write_h5ad(pp)
    real.write_h5ad(rp)
    cfg = EvalConfig(metrics=["de_wilcoxon_overlap"], input_type="lognorm",
                     de={"backend": "scanpy"}, control_source="pred",
                     cache_real=str(tmp_path / "cr"), cache_pred=str(tmp_path / "cp"))
    with pytest.raises(ValueError, match="lognorm"):
        compute_metrics(str(pp), str(rp), config=cfg)


def test_de_metrics_computed_when_tables_absent(synthetic_counts_pair):
    from cell_eval2.run import compute_metrics
    pred, real = synthetic_counts_pair
    df = compute_metrics(pred, real, metrics=["de_wilcoxon_overlap"],
                         input_type="counts", de={"backend": "scanpy"})
    assert "de_wilcoxon_overlap" in set(df["metric"])
    assert df.height > 0


def test_de_profile_emits_all_new_metrics(synthetic_counts_pair):
    from cell_eval2.compat import score_agg_metrics
    pred_ad, real_ad = synthetic_counts_pair
    perts = ["GENE1", "GENE2", "GENE3"]
    feats = perts

    def de(seed):
        rows = [
            (target, feature, float((i + seed) % 5 - 2), 0.001 if i % 2 == 0 else 0.5)
            for target in perts
            for i, feature in enumerate(feats)
        ]
        return pl.DataFrame(
            rows, schema=["target", "feature", "log2_fold_change", "p_adj"], orient="row"
        )

    new_metrics = {
        "de_wilcoxon_nsig_spearman",
        "de_wilcoxon_lfc_spearman",
        "de_wilcoxon_direction_match",
        "de_wilcoxon_model_direction_match",
        "de_wilcoxon_sig_recall",
        "de_wilcoxon_pr_auc",
        "de_wilcoxon_roc_auc",
        "de_wilcoxon_nsig_counts_real",
        "de_wilcoxon_nsig_counts_pred",
        "de_wilcoxon_sig_jaccard",
    }
    cfg = EvalConfig(metrics="de", input_type="counts")
    df = compute_metrics(pred_ad, real_ad, config=cfg, de_pred=de(1), de_real=de(0))
    metrics = set(df["metric"].to_list())
    assert new_metrics <= metrics

    cfg_full = EvalConfig(metrics="full", input_type="counts")
    df_full = compute_metrics(pred_ad, real_ad, config=cfg_full, de_pred=de(1), de_real=de(0))
    assert new_metrics <= set(df_full["metric"].to_list())

    # #19: all 8 top-k overlap/precision variants must be emitted too (they declare
    # profiles=("full","de") but were dropped from PROFILES["de"]/["full"]). The
    # top-k cap gracefully when k exceeds the gene count, so each still emits a value.
    de_metrics = set(df["metric"].to_list())
    topk = {f"de_wilcoxon_{m}_top{k}"
            for m in ("overlap", "precision") for k in (50, 100, 200, 500)}
    assert topk <= de_metrics

    agg = aggregate_metrics(df)
    assert "de_wilcoxon_nsig_counts_real" in agg["metric"].to_list()
    wide = pl.DataFrame({
        "statistic": ["mean"],
        **{row["metric"]: [row["mean"]] for row in agg.iter_rows(named=True)},
    })
    scored = score_agg_metrics(wide, wide)
    scored_metrics = set(scored["metric"].to_list())
    assert "de_wilcoxon_nsig_counts_real" not in scored_metrics
    assert "de_wilcoxon_nsig_counts_pred" not in scored_metrics

    cfg_v1 = EvalConfig(metrics="de", input_type="counts", version="v1")
    df_v1 = compute_metrics(pred_ad, real_ad, config=cfg_v1, de_pred=de(1), de_real=de(0))
    assert "de_spearman_sig" in df_v1["metric"].to_list()
    assert "de_nsig_counts_real" in df_v1["metric"].to_list()


def test_de_computed_under_control_source_pred(synthetic_counts_pair):
    # control_source="pred" exercises the non-combined branch (_pred_de_input returns
    # pred as-is); the default v2 path (control_source="real", combined real-control
    # view) is exercised by the test above.
    from cell_eval2.run import compute_metrics
    pred, real = synthetic_counts_pair
    df = compute_metrics(pred, real, metrics=["de_wilcoxon_overlap"],
                         input_type="counts", control_source="pred",
                         de={"backend": "scanpy"})
    assert df.height > 0


def test_de_anndata_perturbation_mismatch_raises(synthetic_pair):
    # DE targets {GENE1, GENE2} omit GENE3 present in the anndata -> strict mismatch error
    pred_ad, real_ad = synthetic_pair
    real = pl.DataFrame({"target": ["GENE1", "GENE2"], "feature": ["g1", "g2"],
                         "log2_fold_change": [3.0, 2.0], "p_adj": [0.001, 0.001]})
    pred = pl.DataFrame({"target": ["GENE1", "GENE2"], "feature": ["g1", "g2"],
                         "log2_fold_change": [4.0, 2.0], "p_adj": [0.001, 0.001]})
    cfg = EvalConfig(metrics=["overlap_at_N"], de={"nan_lfc_policy": "keep"},
                     input_type="lognorm")
    with pytest.raises(ValueError, match="GENE3"):
        compute_metrics(pred_ad, real_ad, config=cfg, de_pred=pred, de_real=real)


def test_nan_lfc_mask_makes_nan_row_inert(synthetic_pair):
    # Under "mask", a NaN-LFC row gets p_adj=1 -> dropped by the significance gate, so the
    # result is identical to that row simply not existing. Proves the policy applies e2e.
    pred_ad, real_ad = synthetic_pair
    real = pl.DataFrame({"target": ["GENE1", "GENE1", "GENE2", "GENE2", "GENE3", "GENE3"],
                         "feature": ["g1", "g2", "g3", "g4", "g5", "g6"],
                         "log2_fold_change": [3.0, 2.0, 1.0, 5.0, 2.0, 4.0],
                         "p_adj": [0.001] * 6})
    base = {"target": ["GENE1", "GENE1", "GENE2", "GENE2", "GENE3", "GENE3"],
            "feature": ["g1", "g9", "g4", "g3", "g5", "g6"],
            "log2_fold_change": [4.0, 2.0, 5.0, 1.0, 2.0, 4.0],
            "p_adj": [0.001] * 6}
    pred_with_nan = pl.DataFrame({
        "target": base["target"] + ["GENE1"],
        "feature": base["feature"] + ["gX"],
        "log2_fold_change": base["log2_fold_change"] + [float("nan")],
        "p_adj": base["p_adj"] + [0.001],
    })
    pred_without = pl.DataFrame(base)
    cfg = EvalConfig(metrics=["overlap_at_N"], de={"nan_lfc_policy": "mask"},
                     input_type="lognorm")
    a = compute_metrics(pred_ad, real_ad, config=cfg,
                        de_pred=pred_with_nan, de_real=real).sort(["perturbation", "metric"])
    b = compute_metrics(pred_ad, real_ad, config=cfg,
                        de_pred=pred_without, de_real=real).sort(["perturbation", "metric"])
    assert a.equals(b)


def test_sides_processed_sequentially(synthetic_pair, monkeypatch):
    import cell_eval2.run as run
    pred, real = synthetic_pair
    order = []
    orig = run._materialize
    monkeypatch.setattr(run, "_materialize",
                        lambda src: (order.append(id(src)), orig(src))[1])
    compute_metrics(pred, real, metrics=["mae"], input_type="lognorm")
    # pred is materialized first now: pred pseudobulk runs before the DE section (to feed the
    # scale-limit gate its per-cell max), and real_bulks is computed last. Sides are still handled
    # one at a time (peak RAM unchanged) — only the materialize order flipped from the prior
    # real-first sequencing.
    assert order == [id(pred), id(real)]


def test_compute_metrics_accepts_pathlib_path(tmp_path, synthetic_pair):
    from pathlib import Path
    pred, real = synthetic_pair
    pp, rp = tmp_path / "p.h5ad", tmp_path / "r.h5ad"
    pred.write_h5ad(pp)
    real.write_h5ad(rp)
    df = compute_metrics(Path(pp), Path(rp), metrics=["mae"], control="non-targeting",
                         input_type="lognorm")
    assert set(df["metric"].unique()) == {"expr_mae"}


def test_compute_metrics_closes_backed_handles(tmp_path, synthetic_pair, monkeypatch):
    import cell_eval2.run as run
    pred, real = synthetic_pair
    pp, rp = tmp_path / "p.h5ad", tmp_path / "r.h5ad"
    pred.write_h5ad(pp)
    real.write_h5ad(rp)
    closed = []
    monkeypatch.setattr(run, "_close_backed", lambda ad, src: closed.append(src))
    compute_metrics(str(pp), str(rp), metrics=["mae"], control="non-targeting",
                    input_type="lognorm")
    assert str(pp) in closed and str(rp) in closed  # both path-opened handles get closed


def test_compute_metrics_closes_backed_handles_on_error(tmp_path, synthetic_pair, monkeypatch):
    import numpy as np
    import cell_eval2.run as run
    pred, real = synthetic_pair
    bad = pred.copy()
    bad.X = np.rint(bad.X) + 1.0  # all-integer but declared lognorm -> validation error mid-run
    pp, rp = tmp_path / "p.h5ad", tmp_path / "r.h5ad"
    bad.write_h5ad(pp)
    real.write_h5ad(rp)
    closed = []
    monkeypatch.setattr(run, "_close_backed", lambda ad, src: closed.append(src))
    with pytest.raises(ValueError, match="lognorm"):
        compute_metrics(str(pp), str(rp), metrics=["mae"], control="non-targeting",
                        input_type="lognorm")
    assert str(pp) in closed and str(rp) in closed  # finally closes handles despite the error


def test_auc_floor_threaded_v1_vs_v2(synthetic_pair, synthetic_counts_pair):
    # Same DE tables drive both runs; only the version-scoped auc_pval_floor differs
    # (v1=replace_zero, v2=min_nonzero). GENE1's pred has an exact-0 real-sig gene whose
    # ranking flips between strategies -> roc_auc 0.75 (v1) vs 1.0 (v2). GENE2/GENE3 keep
    # the DE target set aligned to the fixtures' perturbations {GENE1,GENE2,GENE3}.
    import math
    real = pl.DataFrame({
        "target": ["GENE1"] * 4 + ["GENE2"] * 2 + ["GENE3"] * 2,
        "feature": ["f1", "f2", "f3", "f4", "g1", "g2", "h1", "h2"],
        "log2_fold_change": [3.0, 2.0, 1.0, 1.0, 2.0, 1.0, 2.0, 1.0],
        "p_adj": [0.001, 0.001, 0.9, 0.9, 0.001, 0.9, 0.001, 0.9]})
    pred = pl.DataFrame({
        "target": ["GENE1"] * 4 + ["GENE2"] * 2 + ["GENE3"] * 2,
        "feature": ["f1", "f2", "f3", "f4", "g1", "g2", "h1", "h2"],
        "log2_fold_change": [3.0, 2.0, 1.0, 1.0, 2.0, 1.0, 2.0, 1.0],
        "p_adj": [0.0, 1e-300, 1e-50, 0.9, 0.001, 0.9, 0.001, 0.9]})

    pred_ad_v1, real_ad_v1 = synthetic_pair          # lognorm; v1 skips input validation
    cfg_v1 = EvalConfig.v1()
    cfg_v1.metrics = ["roc_auc"]
    df_v1 = compute_metrics(pred_ad_v1, real_ad_v1, config=cfg_v1, de_pred=pred, de_real=real)
    v1 = df_v1.filter((pl.col("perturbation") == "GENE1") &
                      (pl.col("metric").is_in(["roc_auc", "de_wilcoxon_roc_auc"])))["value"][0]

    pred_ad_v2, real_ad_v2 = synthetic_counts_pair   # counts; v2 default
    cfg_v2 = EvalConfig.v2()
    cfg_v2.metrics = ["roc_auc"]
    df_v2 = compute_metrics(pred_ad_v2, real_ad_v2, config=cfg_v2, de_pred=pred, de_real=real)
    v2 = df_v2.filter((pl.col("perturbation") == "GENE1") &
                      (pl.col("metric").is_in(["roc_auc", "de_wilcoxon_roc_auc"])))["value"][0]

    assert math.isclose(v1, 0.75, abs_tol=1e-9)   # replace_zero
    assert math.isclose(v2, 1.0, abs_tol=1e-9)    # min_nonzero


def test_dispatch_de_metrics_produces_tidy_rows():
    # dispatch_de_metrics over a PreparedDE returns tidy rows for de-kind names and skips
    # non-de names (CPU-only; no backend/GPU). The refactor is behavior-preserving — the
    # existing _run_metrics DE tests above are the regression guard.
    from cell_eval2 import de as de_mod
    from cell_eval2.config import EvalConfig
    from cell_eval2.run import dispatch_de_metrics

    perts = ["A", "B"]
    feats = ["g0", "g1", "g2", "g3"]

    def _tbl(seed):
        rows = [
            (t, f, float((i + seed) % 5 - 2),
             0.001 if i % 2 == 0 else 0.5,      # p_value
             0.002 if i % 2 == 0 else 0.6)      # p_adj
            for t in perts for i, f in enumerate(feats)
        ]
        return pl.DataFrame(
            rows, schema=["target", "feature", "log2_fold_change", "p_value", "p_adj"],
            orient="row",
        )

    prepared = de_mod.prepare_de(_tbl(1), _tbl(0), control="non-targeting")
    cfg = EvalConfig(metrics="de", input_type="counts")
    names = ["de_wilcoxon_nsig_counts_real", "de_wilcoxon_sig_recall"]
    rows = dispatch_de_metrics(names, prepared, cfg)
    assert rows
    assert {r["metric"] for r in rows} == set(names)
    assert all(isinstance(r["value"], float) for r in rows)
    # non-de names are skipped (use the resolved catalog key "expr_mae", not the "mae" alias)
    assert dispatch_de_metrics(["expr_mae"], prepared, cfg) == []


def test_use_gpu_pseudobulk_predicate():
    from cell_eval2.run import _use_gpu_pseudobulk

    assert _use_gpu_pseudobulk("cuda", "counts", 1e6) is True
    assert _use_gpu_pseudobulk("cuda", "counts", 1e4) is True
    assert _use_gpu_pseudobulk("cpu", "counts", 1e6) is False      # not cuda
    assert _use_gpu_pseudobulk("cuda", "lognorm", 1e6) is False    # not counts
    assert _use_gpu_pseudobulk("cuda", "counts", None) is False    # median target_sum
    assert _use_gpu_pseudobulk("cuda", "counts", 1e6, "lognorm") is True
    assert _use_gpu_pseudobulk("cuda", "counts", 1e6, "bogus") is False  # accumulator-unsupported norm


def test_side_bulks_routes_to_gpu_helper_on_cuda(monkeypatch):
    # When the device resolves to cuda + counts + numeric target_sum, _side_bulks must
    # call inmem_pseudobulk (run it on the cpu accumulator here so no GPU is needed),
    # and the result must equal the CPU to_normalization+pseudobulk reference.
    import anndata as ad
    import numpy as np
    import pandas as pd
    import scipy.sparse as sp

    import cell_eval2.run as run
    from cell_eval2 import norm as _norm
    from cell_eval2.config import EvalConfig
    from cell_eval2.prep import pseudobulk

    rng = np.random.default_rng(21)
    n, g = 200, 12
    X = sp.csr_matrix(rng.poisson(0.8, size=(n, g)).astype(np.float32))
    obs = pd.DataFrame({"target_gene": rng.choice(["non-targeting", "A", "B"], size=n)})
    adata = ad.AnnData(X=X, obs=obs, var=pd.DataFrame(index=[f"g{i}" for i in range(g)]))

    cfg = run._resolve_config(
        EvalConfig(input_type="counts", target_sum=1e6, pert_col="target_gene"), {})
    monkeypatch.setattr(run, "resolve_device", lambda d: "cuda")
    calls = {"n": 0}
    real_inmem = run.inmem_pseudobulk

    def spy(adata, **kw):
        calls["n"] += 1
        return real_inmem(adata, **{**kw, "device": "cpu"})  # force cpu so no GPU is used

    monkeypatch.setattr(run, "inmem_pseudobulk", spy)

    out = run._side_bulks(adata, fp=None, store=None, norms=["lognorm"], cfg=cfg, side="real")
    assert calls["n"] == 1
    ref_perts, ref_means = pseudobulk(
        _norm.to_normalization(adata, "counts", "lognorm", target_sum=1e6), "target_gene"
    )
    assert list(out["lognorm"][0]) == list(ref_perts)
    assert np.allclose(out["lognorm"][1], ref_means, rtol=1e-4, atol=1e-6)


def test_side_bulks_stays_cpu_off_cuda(monkeypatch):
    # device='cpu' must NOT call the GPU helper (CPU path, fp64 unchanged).
    import anndata as ad
    import numpy as np
    import pandas as pd
    import scipy.sparse as sp

    import cell_eval2.run as run
    from cell_eval2.config import EvalConfig

    rng = np.random.default_rng(22)
    n, g = 150, 10
    X = sp.csr_matrix(rng.poisson(0.7, size=(n, g)).astype(np.float32))
    obs = pd.DataFrame({"target_gene": rng.choice(["non-targeting", "A"], size=n)})
    adata = ad.AnnData(X=X, obs=obs, var=pd.DataFrame(index=[f"g{i}" for i in range(g)]))

    cfg = run._resolve_config(EvalConfig(input_type="counts", target_sum=1e6, device="cpu", pert_col="target_gene"), {})
    monkeypatch.setattr(run, "resolve_device", lambda d: "cpu")
    called = {"n": 0}
    monkeypatch.setattr(run, "inmem_pseudobulk",
                        lambda *a, **k: called.__setitem__("n", called["n"] + 1))
    out = run._side_bulks(adata, fp=None, store=None, norms=["lognorm"], cfg=cfg, side="real")
    assert called["n"] == 0
    assert out["lognorm"][1].dtype == np.float64  # CPU path keeps fp64


def test_side_bulks_gpu_matches_cpu_when_cuda_available():
    # On a real CUDA GPU, the gated _side_bulks (fp32) must match the CPU fp64 path
    # within fp32 tolerance. Skips on GPU-free nodes.
    import anndata as ad
    import numpy as np
    import pandas as pd
    import pytest
    import scipy.sparse as sp

    import cell_eval2.run as run
    from cell_eval2.config import EvalConfig
    from cell_eval2.gpu import resolve_device

    if resolve_device("auto") != "cuda":
        pytest.skip("no usable CUDA GPU")

    rng = np.random.default_rng(31)
    n, g = 400, 20
    X = sp.csr_matrix(rng.poisson(0.9, size=(n, g)).astype(np.float32))
    obs = pd.DataFrame({"target_gene": rng.choice(["non-targeting", "A", "B", "C"], size=n)})
    adata = ad.AnnData(X=X, obs=obs, var=pd.DataFrame(index=[f"g{i}" for i in range(g)]))

    cpu_cfg = run._resolve_config(EvalConfig(input_type="counts", target_sum=1e6, device="cpu", pert_col="target_gene"), {})
    gpu_cfg = run._resolve_config(
        EvalConfig(input_type="counts", target_sum=1e6, device="cuda", pert_col="target_gene"), {})
    cpu = run._side_bulks(adata, fp=None, store=None, norms=["lognorm"], cfg=cpu_cfg, side="real")
    gpu = run._side_bulks(adata, fp=None, store=None, norms=["lognorm"], cfg=gpu_cfg, side="real")
    assert list(cpu["lognorm"][0]) == list(gpu["lognorm"][0])
    assert np.allclose(cpu["lognorm"][1], gpu["lognorm"][1], rtol=1e-4, atol=1e-6)


# --- Lever 1 reorder: pred scale-limit runs after pred pseudobulk, must still fire pre-DE ---
def test_reorder_scale_limit_still_rejects_over_budget(synthetic_pair):
    # The pred scale-limit now runs after pred pseudobulk (not up front). It must still reject an
    # over-budget pred. Real is kept within budget so the rejection is specifically the PRED gate.
    import numpy as np
    pred, real = synthetic_pair
    pred, real = pred.copy(), real.copy()
    real.X = np.rint(np.asarray(real.X, dtype=np.float64))       # integer counts, small per-cell totals
    Xp = np.rint(np.asarray(pred.X, dtype=np.float64))
    Xp[0, 0] = 2_000_000.0                                       # pred cell 0 total over the 1e6 cap
    pred.X = Xp
    with pytest.raises(ValueError, match="max_counts_per_cell"):
        compute_metrics(pred, real, config=EvalConfig(
            metrics=["mae"], pert_col="target", control="non-targeting",
            input_type="counts", max_counts_per_cell=1_000_000.0))


def test_reorder_preserves_metric_values(synthetic_pair):
    # Guard: the reorder is output-invariant (the full suite proves exact values; this pins the
    # anndata path end-to-end still produces the expected metric + perturbations).
    pred, real = synthetic_pair
    df = compute_metrics(pred, real, metrics=["mae"], pert_col="target",
                         control="non-targeting", input_type="lognorm")
    assert df.filter(pl.col("metric") == "expr_mae")["value"].to_numpy().min() >= 0
    assert set(df["perturbation"].unique()) == {"GENE1", "GENE2", "GENE3"}


def _raw_de():
    return pl.DataFrame({
        "target": ["A", "A"], "feature": ["g_big", "g_small"],
        "log2_fold_change": [2.0, 0.3], "p_value": [0.001, 0.001], "p_adj": [0.01, 0.01],
    })


def test_prepare_de_cached_applies_floor_no_store():
    cfg = EvalConfig(control="ctrl", de=DEParams(min_abs_log2fc=1.0))
    # supplied=False models the COMPUTED-table path (non-strict, p_adj-blind fingerprint), which is
    # what the floor-awareness below exercises; the supplied=True (strict) path is covered in test_cache.
    prep = _prepare_de_cached(_raw_de(), _raw_de(), cfg=cfg,
                              real_store=None, pred_store=None,
                              de_real_supplied=False, de_pred_supplied=False)
    d = dict(zip(prep.real_df["feature"].to_list(), prep.real_df["p_adj"].to_list()))
    assert d["g_small"] == 1.0 and d["g_big"] == 0.01
    assert set(prep.real_rank["A"].drop_nulls().to_list()) == {"g_big"}


def test_prepare_de_cached_cache_is_floor_aware(tmp_path):
    # Cold populate with floor=1.0, warm re-read returns the SAME floored rank, then
    # floor=0.0 must NOT serve the stale floored rank. The non-strict fingerprint is
    # p_adj-blind (identical for both floors), so the min_abs_log2fc params entry is what
    # invalidates the cache.
    # supplied=False: exercise the COMPUTED-table path, whose non-strict fingerprint is p_adj-blind
    # so the min_abs_log2fc params entry (not the fingerprint) is what invalidates the cache.
    cfg_hi = EvalConfig(control="ctrl", de=DEParams(min_abs_log2fc=1.0))
    r = CacheStore(str(tmp_path / "real"))
    p = CacheStore(str(tmp_path / "pred"))
    supplied = dict(de_real_supplied=False, de_pred_supplied=False)
    cold = _prepare_de_cached(_raw_de(), _raw_de(), cfg=cfg_hi, real_store=r, pred_store=p, **supplied)
    assert set(cold.real_rank["A"].drop_nulls().to_list()) == {"g_big"}

    r2 = CacheStore(str(tmp_path / "real"))
    p2 = CacheStore(str(tmp_path / "pred"))
    warm = _prepare_de_cached(_raw_de(), _raw_de(), cfg=cfg_hi, real_store=r2, pred_store=p2, **supplied)
    assert set(warm.real_rank["A"].drop_nulls().to_list()) == {"g_big"}  # warm hit, same result

    cfg_lo = EvalConfig(control="ctrl", de=DEParams(min_abs_log2fc=0.0))
    nofloor = _prepare_de_cached(_raw_de(), _raw_de(), cfg=cfg_lo, real_store=r2, pred_store=p2, **supplied)
    assert set(nofloor.real_rank["A"].drop_nulls().to_list()) == {"g_big", "g_small"}  # no stale
