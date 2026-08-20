import json

import polars as pl

from cell_eval2 import partition


def _write_partial(d, pid, rows, perts, ref="R", cfg="C"):
    pl.DataFrame(rows, schema={"perturbation": pl.Utf8, "metric": pl.Utf8, "value": pl.Float64}
                 ).write_parquet(d / f"{pid}.parquet")
    (d / f"{pid}.json").write_text(json.dumps(
        {"subset_id": pid, "real_ref_fingerprint": ref, "config_hash": cfg,
         "comparator": "lognorm",
         "perturbations": perts}))


def test_nsig_spearman_reconstructed_across_pieces(tmp_path):
    # Two disjoint pieces; counts chosen so the global Spearman is a clean value.
    _write_partial(tmp_path, "p0",
        [{"perturbation": "A", "metric": "de_wilcoxon_nsig_counts_real", "value": 10.0},
         {"perturbation": "A", "metric": "de_wilcoxon_nsig_counts_pred", "value": 8.0}],
        ["A"])
    _write_partial(tmp_path, "p1",
        [{"perturbation": "B", "metric": "de_wilcoxon_nsig_counts_real", "value": 20.0},
         {"perturbation": "B", "metric": "de_wilcoxon_nsig_counts_pred", "value": 25.0}],
        ["B"])
    full, agg = partition.aggregate_partials(
        str(tmp_path), reference_universe=["A", "B"], reduce_nsig_spearman=True)
    sp = full.filter(pl.col("metric") == "de_wilcoxon_nsig_spearman")
    assert sorted(sp["perturbation"].to_list()) == ["A", "B"]   # broadcast to all
    assert sp["value"].to_list() == [1.0, 1.0]                  # (10,8),(20,25) both ascending -> +1


def test_coverage_gap_raises(tmp_path):
    _write_partial(tmp_path, "p0",
        [{"perturbation": "A", "metric": "de_wilcoxon_nsig_counts_real", "value": 1.0}], ["A"])
    import pytest
    with pytest.raises(ValueError, match="coverage"):
        partition.aggregate_partials(str(tmp_path), reference_universe=["A", "B"])


def test_nsig_spearman_undefined_fills_worst_value(tmp_path):
    # Only ONE target has a real-significant gene -> Spearman over a single point is undefined
    # (NaN). The v2 no-droppable-NaN policy fills it with the catalog worst_value (-1.0),
    # matching compute_metrics' run._fill_no_drop. Regression: previously left NaN, which
    # diverged from whole-prediction scoring (exposed by the h5ad-manifest streamed-vs-whole parity test).
    _write_partial(tmp_path, "p0",
        [{"perturbation": "C", "metric": "de_wilcoxon_nsig_counts_real", "value": 1.0},
         {"perturbation": "C", "metric": "de_wilcoxon_nsig_counts_pred", "value": 0.0}],
        ["C"])
    full, _agg = partition.aggregate_partials(
        str(tmp_path), reference_universe=["C"], reduce_nsig_spearman=True)
    sp = full.filter(pl.col("metric") == "de_wilcoxon_nsig_spearman")
    assert sp["value"].to_list() == [-1.0]   # worst_value fill, NOT NaN


def test_inject_nsig_spearman_v1_names():
    # v1-named partial rows: de_nsig_counts_{real,pred}, target metric de_spearman_sig
    full = pl.DataFrame(
        {
            "perturbation": ["A", "B", "A", "B"],
            "metric": ["de_nsig_counts_real", "de_nsig_counts_real",
                       "de_nsig_counts_pred", "de_nsig_counts_pred"],
            "value": [3.0, 5.0, 2.0, 6.0],
        },
        schema={"perturbation": pl.Utf8, "metric": pl.Utf8, "value": pl.Float64},
    )
    out = partition._inject_nsig_spearman(
        full, metric_name="de_spearman_sig",
        real_metric="de_nsig_counts_real", pred_metric="de_nsig_counts_pred")
    got = out.filter(pl.col("metric") == "de_spearman_sig")
    assert got.height == 2                       # broadcast to both perts
    assert abs(got["value"][0] - 1.0) < 1e-9     # perfectly rank-correlated


def test_aggregate_partials_forwards_v1_nsig_names(tmp_path):
    # two partials, v1-named counts rows, one pert each, matching real/pred keys
    for sid, pert, nr, npd in (("p0", "A", 3.0, 2.0), ("p1", "B", 5.0, 6.0)):
        _write_partial(tmp_path, sid,
            [{"perturbation": pert, "metric": "de_nsig_counts_real", "value": nr},
             {"perturbation": pert, "metric": "de_nsig_counts_pred", "value": npd}],
            [pert])
    full, _agg = partition.aggregate_partials(
        str(tmp_path), reduce_nsig_spearman=True,
        nsig_spearman_metric="de_spearman_sig",
        nsig_real_metric="de_nsig_counts_real", nsig_pred_metric="de_nsig_counts_pred")
    assert full.filter(pl.col("metric") == "de_spearman_sig").height == 2
