"""AUC rankings must survive tiny p-values and changes in streaming batch size.

`preserve` casts p-values to float64 and floors only at the smallest positive float64. v2's
default stays `min_nonzero`; partitioned scoring cannot take a whole-table minimum, so
`partition_inmem._require_partition_config` normalizes it to `preserve`. On a Float64 table the two
differ only in what an exact zero becomes, and the zero tie that follows is pinned below.
"""

from dataclasses import replace

import numpy as np
import polars as pl
import pytest
from sklearn.metrics import average_precision_score, roc_auc_score

from cell_eval2 import partition_inmem
from cell_eval2 import run
from cell_eval2.config import EvalConfig
from cell_eval2.de import prepare_de
from cell_eval2.metrics.de import de_pr_auc, de_roc_auc
from cell_eval2.run import dispatch_de_metrics


def test_pr_auc_of_a_perfect_ranking_is_exactly_one():
    """#414: sklearn's average precision sums recall steps in floating point and can land one
    ULP above 1 on a perfect ranking; the metric is defined on [0, 1]."""
    import polars as pl
    from cell_eval2.de import prepare_de
    from cell_eval2.metrics.de import de_pr_auc
    k, n = 18, 20
    genes = [f"g{i:02d}" for i in range(k + n)]
    real = pl.DataFrame({"target": ["P"] * (k + n), "feature": genes,
                         "log2_fold_change": [1.0] * (k + n),
                         "p_adj": [1e-4] * k + [0.9] * n})
    pred = real.with_columns(pl.Series(
        "p_adj", [1e-12 * (i + 1) for i in range(k)] + [0.5 + 0.01 * i for i in range(n)]))
    assert de_pr_auc(prepare_de(pred, real, control="non-targeting"))["P"] == 1.0


def test_pr_auc_clip_is_scoped_to_pr_auc_metrics():
    """#414: only PR AUC selections can serve an unclipped warm result."""
    assert run._pr_auc_used(["de_wilcoxon_pr_auc"]) is True
    assert run._pr_auc_used(["expr_mae", "de_wilcoxon_roc_auc"]) is False


def test_pr_auc_clip_changes_only_the_opted_in_result_digest():
    """#414: invalidate unclipped results while preserving every non-PR-AUC warm key."""
    cfg = EvalConfig.v2()
    kw = dict(de_backend_used=True, comparator="bulk_lognorm")
    clipped = run._result_config_digest(cfg, **kw, pr_auc_clip_used=True)
    unchanged = run._result_config_digest(cfg, **kw, pr_auc_clip_used=False)
    assert clipped != unchanged
    assert unchanged == run._result_config_digest(cfg, **kw)


def _tables(pvalues, labels, target="P", dtype=pl.Float64):
    real = pl.DataFrame(
        {
            "target": [target] * len(labels),
            "feature": [f"g{i}" for i in range(len(labels))],
            "log2_fold_change": [1.0] * len(labels),
            "p_adj": [0.001 if label else 0.9 for label in labels],
        }
    )
    pred = real.with_columns(pl.Series("p_adj", pvalues, dtype=dtype))
    return real, pred


@pytest.mark.parametrize(
    "metric,reference",
    [
        (de_pr_auc, average_precision_score),
        (de_roc_auc, roc_auc_score),
    ],
)
@pytest.mark.parametrize(
    "pvalues",
    [
        [0.0, 1e-300, 1e-200, 0.5],
        [1e-200, 1e-100, 1e-50, 0.5],
        [0.0, 0.0, 0.0, 0.0],
        [np.nextafter(0.0, 1.0), 1e-320, 0.0, 0.5],
        # A zero on a significant gene TIES with the smallest positive float64 on a
        # non-significant one: that is the floor itself.
        [0.0, np.nextafter(0.0, 1.0), 0.5, 0.9],
        [float("nan"), None, 1e-100, 0.5],
    ],
)
# A Float32 table is cast to float64 BEFORE the floor: clipping in float32 would round the floor
# to 0 and score every zero as an infinite -log10.
@pytest.mark.parametrize("dtype", [pl.Float64, pl.Float32])
def test_preserve_auc_matches_float64_reference(metric, reference, pvalues, dtype):
    labels = [1, 0, 1, 0]
    real, pred = _tables(pvalues, labels, dtype=dtype)
    prepared = prepare_de(pred, real, control="ctrl")
    as_stored = np.asarray(pvalues, dtype=np.float64 if dtype == pl.Float64 else np.float32)
    clean = np.nan_to_num(as_stored.astype(np.float64), nan=1.0)
    scores = -np.log10(np.maximum(clean, np.nextafter(0.0, 1.0)))
    value = metric(prepared, auc_pval_floor="preserve")["P"]
    assert value == pytest.approx(reference(labels, scores), abs=1e-15)


@pytest.mark.parametrize("floor", ["min_nonzero", "preserve"])
def test_missing_prediction_is_scored_as_nonsignificant(floor):
    real, pred = _tables([0.1, 0.5, 1e-200, 0.9], [1, 0, 1, 0])
    prepared = prepare_de(pred.filter(pl.col("feature") != "g0"), real, control="ctrl")
    expected = average_precision_score(
        [1, 0, 1, 0], [0.0, -np.log10(0.5), 200.0, -np.log10(0.9)]
    )
    assert de_pr_auc(prepared, auc_pval_floor=floor)["P"] == pytest.approx(expected)


def test_partitioned_auc_matches_whole_preserve_and_sklearn(monkeypatch):
    monkeypatch.setattr(partition_inmem, "_resolve_backend", lambda _: "gpudge")
    cfg = EvalConfig.v2()
    streaming = partition_inmem._require_partition_config(cfg)
    whole_cfg = replace(cfg, de=replace(cfg.de, auc_pval_floor="preserve"))
    names = ["de_wilcoxon_pr_auc", "de_wilcoxon_roc_auc"]
    # Ten positives per population satisfy the reference-only scoring gate.
    labels = [1, 0, 1, 0] * 5
    real1, pred1 = _tables([0.0, 1e-300, 1e-200, 0.5] * 5, labels, "P1")
    real2, pred2 = _tables([1e-100, 1e-150, 1e-200, 0.9] * 5, labels, "P2")
    whole = prepare_de(
        pl.concat([pred1, pred2]), pl.concat([real1, real2]), control="ctrl"
    )
    whole_rows = dispatch_de_metrics(names, whole, whole_cfg)
    pieces = []
    for real, pred in [(real1, pred1), (real2, pred2)]:
        prepared = prepare_de(pred, real, control="ctrl")
        rows = dispatch_de_metrics(names, prepared, streaming)
        scores = -np.log10(np.maximum(pred["p_adj"].to_numpy(), np.nextafter(0.0, 1.0)))
        expected = {
            names[0]: average_precision_score(labels, scores),
            names[1]: roc_auc_score(labels, scores),
        }
        for row in rows:
            assert row["value"] == pytest.approx(expected[row["metric"]], abs=1e-15)
        pieces.extend(rows)

    def key(row):
        return row["perturbation"], row["metric"]

    assert sorted(pieces, key=key) == sorted(whole_rows, key=key)


def test_zero_tie_separates_partitioned_preserve_from_whole_table_min_nonzero(monkeypatch):
    """The documented difference, pinned. Whole-table `min_nonzero` gives an exact-zero pred
    p-value the table's smallest nonzero one -- 1e-300, which is in P1 -- so in P1 the zeros
    TIE with it, while partitioned `preserve` ranks them strictly above. P2 has zeros too, but
    not the table minimum, so the two paths agree on it exactly."""
    monkeypatch.setattr(partition_inmem, "_resolve_backend", lambda _: "gpudge")
    cfg = EvalConfig.v2()
    assert cfg.de.auc_pval_floor == "min_nonzero"
    streaming = partition_inmem._require_partition_config(cfg)
    pr, roc = names = ["de_wilcoxon_pr_auc", "de_wilcoxon_roc_auc"]
    # Zeros sit on real-significant genes, 1e-300 on non-significant ones.
    labels = [1, 0, 1, 0] * 5
    p1 = [0.0, 1e-300, 1e-5, 0.9] * 5
    real1, pred1 = _tables(p1, labels, "P1")
    real2, pred2 = _tables([0.0, 1e-150, 1e-200, 0.9] * 5, labels, "P2")
    whole = prepare_de(
        pl.concat([pred1, pred2]), pl.concat([real1, real2]), control="ctrl"
    )
    whole_rows = {(r["perturbation"], r["metric"]): r["value"]
                  for r in dispatch_de_metrics(names, whole, cfg)}
    pieces = {}
    for real, pred in [(real1, pred1), (real2, pred2)]:
        prepared = prepare_de(pred, real, control="ctrl")
        for r in dispatch_de_metrics(names, prepared, streaming):
            pieces[(r["perturbation"], r["metric"])] = r["value"]

    assert pieces[("P2", pr)] == whole_rows[("P2", pr)]
    assert pieces[("P2", roc)] == whole_rows[("P2", roc)]
    tied = -np.log10(np.where(np.asarray(p1) == 0.0, 1e-300, p1))
    strict = -np.log10(np.maximum(p1, np.nextafter(0.0, 1.0)))
    assert whole_rows[("P1", roc)] == pytest.approx(0.625)
    assert pieces[("P1", roc)] == pytest.approx(0.75)
    assert whole_rows[("P1", pr)] == pytest.approx(average_precision_score(labels, tied))
    assert pieces[("P1", pr)] == pytest.approx(average_precision_score(labels, strict))
    assert pieces[("P1", pr)] > whole_rows[("P1", pr)]


def test_v2_default_normalizes_to_preserve_without_float32_underflow(monkeypatch):
    monkeypatch.setattr(partition_inmem, "_resolve_backend", lambda _: "gpudge")
    cfg = EvalConfig.v2()
    assert cfg.de.auc_pval_floor == "min_nonzero"
    de = partition_inmem._require_partition_config(cfg).de
    assert de.auc_pval_floor == "preserve"
    # A real-significant gene at 1e-100 and a non-significant one at 1e-20.
    real, pred = _tables([1e-100, 1e-20, 0.5, 0.9], [1, 0, 1, 0])
    prepared = prepare_de(pred, real, control="ctrl")
    normalized = de_roc_auc(prepared, auc_pval_floor=de.auc_pval_floor,
                            auc_pval_floor_value=de.auc_pval_floor_value)["P"]
    assert normalized == pytest.approx(0.75)    # 1e-100 ranks above 1e-20
    # The pre-0.19.0 target: float32 turns 1e-100 into 0, then 1e-10, below 1e-20.
    legacy = de_roc_auc(prepared, auc_pval_floor="replace_zero", auc_pval_floor_value=1e-10)["P"]
    assert legacy == pytest.approx(0.5)
