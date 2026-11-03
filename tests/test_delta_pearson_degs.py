from dataclasses import replace

import numpy as np
import polars as pl
import pytest

from cell_eval2.catalog import resolve_metrics
from cell_eval2.config import DEParams, EvalConfig
from cell_eval2.metrics.delta_degs import MIN_GENES, pearson_delta_degs
from cell_eval2.run import compute_metrics

#: Twelve off-target reference DEGs: above the ten-gene floor, with room to drop some.
OFF = [f"g{i}" for i in range(12)]
GENES = ["target", *OFF, "ns"]       # column 0 is the target, 1-12 are OFF, 13 is not DE
REAL = np.arange(1.0, 13.0)
PRED = np.array([1, 4, 2, 5, 3, 7, 6, 9, 8, 12, 10, 11.])


def _de(genes, *, label="guide", p_adj=None):
    return pl.DataFrame({
        "target": [label] * len(genes), "feature": genes,
        "p_adj": p_adj if p_adj is not None else [0.001] * len(genes),
        "log2_fold_change": [1.0] * len(genes),
    })


def _inputs():
    return dict(
        genes=GENES, de_real=_de(GENES, p_adj=[0.001] * 13 + [0.8]),
        pred_bulk=(np.array(["ctrl", "guide"]),
                   np.array([np.zeros(14), [-100, *PRED, -20.]])),
        real_bulk=(np.array(["guide", "ctrl"]),
                   np.array([[100, *REAL, 20.], np.zeros(14)])),
        control="ctrl", target_gene_map={"guide": "target"},
    )


def test_reference_selection_excludes_target_and_ignores_nonsignificant_genes():
    args = _inputs()
    expected = np.corrcoef(PRED, REAL)[0, 1]
    assert pearson_delta_degs(**args)["guide"] == pytest.approx(expected)
    # Neither the on-target knockdown nor a non-DE prediction may change the metric.
    args["pred_bulk"][1][1, [0, 13]] = 1e6
    assert pearson_delta_degs(**args)["guide"] == pytest.approx(expected)


def test_control_source_and_explicit_nongenetic_mapping():
    args = _inputs()
    args["pred_bulk"][1][0, 1:13] = PRED - REAL   # the pred control turns the pred delta into REAL
    assert pearson_delta_degs(**args)["guide"] == pytest.approx(1)
    assert pearson_delta_degs(**args, control_source="real")["guide"] != pytest.approx(1)
    args["target_gene_map"] = {"*": None}
    assert pearson_delta_degs(**args)["guide"] < 0


def test_threshold_and_lfc_floor_match_de_selection():
    args = _inputs()
    args["pred_bulk"][1][1, 1:13] = REAL
    args["pred_bulk"][1][1, 12] = -50.0           # g11 alone breaks the correlation
    assert pearson_delta_degs(**args)["guide"] < 0.99
    args["de_real"] = args["de_real"].with_columns(
        pl.when(pl.col("feature") == "g11").then(0.05).otherwise(pl.col("p_adj")).alias("p_adj")
    )
    # A p_adj equal to the threshold is not significant, as in the DE metrics.
    assert pearson_delta_degs(**args)["guide"] == pytest.approx(1)
    assert np.isnan(pearson_delta_degs(**args, min_abs_log2fc=2)["guide"])


def test_ten_reference_genes_are_scored_and_nine_are_not():
    assert MIN_GENES == 10
    args = _inputs()
    for n_sig, scored in ((MIN_GENES, True), (MIN_GENES - 1, False)):
        p_adj = [0.001] + [0.001 if i < n_sig else 0.8 for i in range(len(OFF))] + [0.8]
        args["de_real"] = _de(GENES, p_adj=p_adj)
        assert bool(np.isfinite(pearson_delta_degs(**args)["guide"])) == scored, n_sig


@pytest.mark.parametrize("case", ["too_few_genes", "constant_real", "nonfinite_real"])
def test_unscoreable_reference_is_nan(case):
    args = _inputs()
    if case == "too_few_genes":
        args["de_real"] = _de(GENES, p_adj=[0.01, 0.01] + [0.8] * 12)
    else:
        args["real_bulk"][1][0, 1:13] = 1 if case == "constant_real" else np.nan
    assert np.isnan(pearson_delta_degs(**args)["guide"])


@pytest.mark.parametrize("value", [0, np.nan])
def test_undefined_prediction_is_penalized(value):
    args = _inputs()
    args["pred_bulk"][1][1, 1:13] = value
    assert pearson_delta_degs(**args)["guide"] == -1


def test_missing_reference_population_is_not_silently_dropped():
    args = _inputs()
    args["de_real"] = _de(GENES, label="other")
    with pytest.raises(ValueError, match="missing perturbations"):
        pearson_delta_degs(**args)


def test_a_table_that_names_no_measured_gene_raises():
    # Ensembl IDs against a symbol panel, say: every row would otherwise read NaN.
    args = _inputs()
    args["de_real"] = _de([f"ENSG{i:011d}" for i in range(len(GENES))])
    with pytest.raises(ValueError, match="no feature of the real DE table is a measured gene"):
        pearson_delta_degs(**args)


def test_gene_order_is_aligned_by_name():
    args = _inputs()
    expected = pearson_delta_degs(**args)
    order = np.random.default_rng(0).permutation(len(GENES))
    args["genes"] = np.asarray(GENES)[order]
    for side in ("pred_bulk", "real_bulk"):
        labels, means = args[side]
        args[side] = labels, means[:, order]
    assert pearson_delta_degs(**args) == pytest.approx(expected)


def _table_for_pair(real):
    return pl.concat([_de(real.var_names.to_list(), label=p)
                      for p in sorted(set(real.obs["target"]) - {"non-targeting"})])


def _cfg(**kwargs):
    return EvalConfig(metrics=["delta_pearson_degs"], device="cpu", input_type="lognorm",
                      target_sum=None, de=DEParams(backend="pdex"), **kwargs)


@pytest.mark.parametrize("control_source", ["pred", "real"])
def test_inmemory_matches_manual_deltas_and_uses_only_real_de(
    synthetic_pair, monkeypatch, control_source,
):
    from cell_eval2 import run
    pred, real = synthetic_pair
    table = _table_for_pair(real)
    calls = []

    def compute_real(adata, *, side, **kwargs):
        assert side == "real"
        calls.append(side)
        return table

    monkeypatch.setattr(run, "_compute_de_side", compute_real)
    cfg = _cfg(control_source=control_source)
    result = compute_metrics(pred, real, config=cfg)
    assert calls == ["real"]
    ctrl = "non-targeting"
    def means(ad, label):
        return np.asarray(ad.X[ad.obs.target == label]).mean(axis=0)
    for p, _, value in result.iter_rows():
        mask = real.var_names != p
        dp = means(pred, p) - means(pred if control_source == "pred" else real, ctrl)
        dr = means(real, p) - means(real, ctrl)
        assert value == pytest.approx(np.corrcoef(dp[mask], dr[mask])[0, 1])
    calls.clear()
    supplied = compute_metrics(pred, real, config=cfg, de_real=table)
    assert not calls
    assert supplied.equals(result)


def test_supplied_reference_changes_result_cache(synthetic_pair, tmp_path, monkeypatch):
    from cell_eval2 import run
    pred, real = synthetic_pair
    cfg = _cfg(cache_pred=str(tmp_path / "cache"))
    table = _table_for_pair(real)
    a = compute_metrics(pred, real, config=cfg, de_real=table)
    # Fifteen of the forty genes stay significant: above the floor, and a different gene set.
    changed = table.with_columns(pl.when(pl.col("feature").is_in([f"g{i}" for i in range(15)]))
                                 .then(0.001).otherwise(1.0).alias("p_adj"))
    b = compute_metrics(pred, real, config=cfg, de_real=changed)
    assert not np.allclose(a["value"], b["value"])
    monkeypatch.setattr(run, "dispatch_anndata_metrics", lambda *a, **k: pytest.fail("cache miss"))
    assert compute_metrics(pred, real, config=cfg, de_real=changed).equals(b)


def test_counts_use_same_comparator_as_delta_pearson(synthetic_counts_pair):
    from cell_eval2.run import _side_bulks
    pred, real = synthetic_counts_pair
    cfg = replace(_cfg(), input_type="counts", target_sum=1e6)
    table = _table_for_pair(real)
    kwargs = dict(fp=None, store=None, norms=["bulk_lognorm"], cfg=cfg)
    expected = pearson_delta_degs(
        pred_bulk=_side_bulks(pred, side="pred", **kwargs)["bulk_lognorm"],
        real_bulk=_side_bulks(real, side="real", **kwargs)["bulk_lognorm"],
        genes=real.var_names, de_real=table, control_source=cfg.control_source,
    )
    got = compute_metrics(pred, real, config=cfg, de_real=table)
    assert dict(got.select("perturbation", "value").iter_rows()) == pytest.approx(expected)


def _structured_table(real):
    """A reference DE table on which each selection setting below changes the selected genes.

    Columns 0-2 are the three target genes, then g0, g1, ... (`conftest._var_names`).
    """
    rows = []
    for p in sorted(set(real.obs["target"]) - {"non-targeting"}):
        for j, g in enumerate(real.var_names):
            p_adj = 0.001 if j < 23 else 0.03 if j < 33 else 0.9   # g20-g29: 0.01 <= p < 0.05
            lfc = 0.2 if 3 <= j < 8 else 1.0                        # g0-g4: below a 0.5 floor
            if 33 <= j < 37:                                        # g30-g33: NaN fold change
                p_adj, lfc = 0.001, float("nan")
            rows.append((p, g, p_adj, lfc))
    return pl.DataFrame(rows, schema=["target", "feature", "p_adj", "log2_fold_change"],
                        orient="row")


@pytest.mark.parametrize("change", [
    {"de": {"p_adj_threshold": 0.01}},
    {"de": {"min_abs_log2fc": 0.5}},
    {"de": {"nan_lfc_policy": "keep"}},
    {"target_gene_map": {"GENE1": "g0", "GENE2": "GENE2", "GENE3": "GENE3"}},
], ids=["p_adj_threshold", "min_abs_log2fc", "nan_lfc_policy", "target_gene_map"])
def test_the_run_forwards_every_selection_setting(synthetic_pair, change):
    from cell_eval2.run import _side_bulks
    pred, real = synthetic_pair
    table = _structured_table(real)
    base = _cfg()
    cfg = replace(base, de=replace(base.de, **change.get("de", {})),
                  target_gene_map=change.get("target_gene_map"))

    def scored(c):
        return dict(compute_metrics(pred, real, config=c, de_real=table)
                    .select("perturbation", "value").iter_rows())

    kwargs = dict(fp=None, store=None, norms=["lognorm"], cfg=cfg)
    expected = pearson_delta_degs(
        pred_bulk=_side_bulks(pred, side="pred", **kwargs)["lognorm"],
        real_bulk=_side_bulks(real, side="real", **kwargs)["lognorm"],
        genes=real.var_names, de_real=table, control=cfg.control,
        control_source=cfg.control_source, target_gene_map=cfg.target_gene_map,
        p_adj_threshold=cfg.de.p_adj_threshold, min_abs_log2fc=cfg.de.min_abs_log2fc,
        nan_lfc_policy=cfg.de.nan_lfc_policy,
    )
    got = scored(cfg)
    assert got == pytest.approx(expected)
    # The setting moves the value on this table, so a run that dropped it would fail here.
    assert got != pytest.approx(scored(base))


def test_the_shard_stream_path_refuses_the_metric_before_reading_an_archive(monkeypatch):
    import cell_eval2.scale as scale

    def no_reads(*a, **k):
        raise AssertionError("read an archive before refusing the metric")

    monkeypatch.setattr(scale, "_shard_effective_input_type", no_reads)
    with pytest.raises(NotImplementedError, match="does not supply"):
        scale.score_streaming("pred.shad", "real.shad", config=_cfg())


def test_metric_is_opt_in_and_v2_only():
    assert "delta_pearson_degs" not in resolve_metrics("full", version="v2")[0]
    with pytest.raises(ValueError, match="v1"):
        resolve_metrics(["delta_pearson_degs"], version="v1")


def test_combined_selection_ignores_predicted_de_calls(synthetic_pair):
    pred, real = synthetic_pair
    table = _table_for_pair(real)
    cfg = _cfg()
    expected = compute_metrics(pred, real, config=cfg, de_real=table)
    cfg = replace(cfg, metrics=["delta_pearson_degs", "de_wilcoxon_sig_jaccard"])
    for p in (0.001, 1.0):
        predicted_de = table.with_columns(pl.lit(p).alias("p_adj"))
        combined = compute_metrics(pred, real, config=cfg, de_real=table, de_pred=predicted_de)
        assert combined.filter(pl.col("metric") == "delta_pearson_degs").equals(expected)


@pytest.mark.parametrize("control_source", ["pred", "real"])
def test_partitioned_equals_whole_prediction(synthetic_pair, tmp_path, monkeypatch, control_source):
    from cell_eval2 import partition_inmem
    pred, real = synthetic_pair
    table = _table_for_pair(real)
    cfg = _cfg(control_source=control_source)
    cache = str(tmp_path / "ref")
    # Real bundle I/O and scoring are exercised; only the GPU-only DE engine is stubbed.
    monkeypatch.setattr(partition_inmem, "_resolve_backend", lambda b: "gpudge")
    monkeypatch.setattr(partition_inmem, "compute_de", lambda *a, **k: table)
    partition_inmem.build_reference(real, config=cfg, cache_dir=cache,
                                    control_format="h5ad", comparator="lognorm")
    if control_source == "pred":
        pred_path = tmp_path / "pred.h5ad"
        pred.write_h5ad(pred_path)
        partition_inmem.build_pred_control_reference(
            str(pred_path), config=cfg, cache_dir=cache, control=cfg.control,
            input_type="lognorm", comparator="lognorm")
    monkeypatch.setattr(partition_inmem, "compute_de", lambda *a, **k: pytest.fail("pred DE"))
    expected = compute_metrics(pred, real, config=cfg, de_real=table).sort("perturbation")
    parts = []
    for label in expected["perturbation"]:
        piece = pred[pred.obs.target == label].copy()
        parts.append(partition_inmem.score_piece(piece, cache, config=cfg, comparator="lognorm"))
    got = pl.concat(parts).sort("perturbation")
    assert got["perturbation"].equals(expected["perturbation"])
    np.testing.assert_allclose(got["value"], expected["value"])
