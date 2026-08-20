import functools
import inspect

import numpy as np
import polars as pl
import pytest

cell_eval = pytest.importorskip("cell_eval")  # test-only golden source
pl.enable_string_cache()  # cell-eval relies on a global string cache


def test_mae_matches_cell_eval(synthetic_pair):
    pred, real = synthetic_pair

    from cell_eval import PerturbationAnndataPair
    from cell_eval.metrics import mae as ce_mae

    pair = PerturbationAnndataPair(
        real=real, pred=pred, pert_col="target", control_pert="non-targeting"
    )
    golden = ce_mae(pair)  # {pert: mae}

    from cell_eval2.metrics import mae
    ours = mae(pred=pred, real=real, pert_col="target", control="non-targeting")

    assert set(ours) == {str(k) for k in golden}
    for pert, gold in golden.items():
        assert ours[str(pert)] == pytest.approx(gold, rel=1e-6, abs=1e-9)


def test_mse_matches_cell_eval(synthetic_pair):
    pred, real = synthetic_pair

    from cell_eval import PerturbationAnndataPair
    from cell_eval.metrics import mse as ce_mse

    pair = PerturbationAnndataPair(
        real=real, pred=pred, pert_col="target", control_pert="non-targeting"
    )
    golden = ce_mse(pair)  # {pert: mse}

    from cell_eval2.metrics import mse
    ours = mse(pred=pred, real=real, pert_col="target", control="non-targeting")

    assert set(ours) == {str(k) for k in golden}
    for pert, gold in golden.items():
        assert ours[str(pert)] == pytest.approx(gold, rel=1e-6, abs=1e-9)


@pytest.mark.parametrize("name", ["mae_delta", "mse_delta", "pearson_delta"])
def test_delta_metrics_match_cell_eval(synthetic_pair, name):
    import math

    pred, real = synthetic_pair

    from cell_eval import PerturbationAnndataPair
    import cell_eval.metrics as ce_metrics

    pair = PerturbationAnndataPair(
        real=real, pred=pred, pert_col="target", control_pert="non-targeting"
    )
    golden = getattr(ce_metrics, name)(pair)  # {pert: value}

    import cell_eval2.metrics as c2_metrics
    # control_source="pred" reproduces upstream's within-realm delta (pred uses pred's
    # control, real uses real's), matching the discrimination parity convention.
    ours = getattr(c2_metrics, name)(
        pred=pred, real=real, pert_col="target", control="non-targeting",
        control_source="pred",
    )

    assert set(ours) == {str(k) for k in golden}
    for pert, gold in golden.items():
        a = ours[str(pert)]
        assert (math.isnan(a) and math.isnan(gold)) or a == pytest.approx(
            gold, rel=1e-6, abs=1e-9
        )


@pytest.mark.parametrize("metric", ["l1", "l2", "cosine"])
def test_discrimination_matches_cell_eval(synthetic_pair, metric):
    pred, real = synthetic_pair

    from cell_eval import PerturbationAnndataPair
    from cell_eval.metrics import discrimination_score as ce_disc

    pair = PerturbationAnndataPair(
        real=real, pred=pred, pert_col="target", control_pert="non-targeting"
    )
    golden = ce_disc(pair, metric=metric, exclude_target_gene=True)  # {pert: score}

    from cell_eval2.metrics import discrimination_score
    # Legacy preset: rank denominator n, predicted control, exclude target gene, and the
    # legacy argsort tie rule. `tie_policy` is stated EXPLICITLY rather than left to the
    # default, which is v2's "midrank" (issue #282): this fixture happens to have no tied
    # distances, so the two policies agree on it and the omission would not fail -- which
    # is exactly why it has to be written down. This test asserts upstream parity, and
    # upstream's rule is "position".
    # `exclusion_scope` is stated for the SAME reason, and here it is not hypothetical: the
    # v2 default is "panel" (#343 -- the whole panel's target genes leave the ranked feature
    # space) while upstream removes only the row's own, and on this fixture the two disagree
    # outright (0.667 vs 0.333 on the l2 arm). Upstream's rule is "row".
    ours = discrimination_score(
        pred=pred, real=real, pert_col="target", control="non-targeting",
        distance=metric, rank_denominator="n", control_source="pred",
        exclude_target_gene=True, tie_policy="position", exclusion_scope="row",
    )

    assert set(ours) == {str(k) for k in golden}
    for pert, gold in golden.items():
        assert ours[str(pert)] == pytest.approx(gold, rel=1e-6, abs=1e-9)


def _parity_pair():
    real = pl.DataFrame({
        "target": ["A", "A", "A", "B", "B", "C"],
        "feature": ["g1", "g2", "g3", "g4", "g5", "g6"],
        "log2_fold_change": [3.0, -2.0, 1.5, 5.0, 0.4, 2.0],
        "p_value": [1e-4, 1e-3, 1e-2, 1e-5, 0.3, 1e-3],
        "fdr": [1e-3, 1e-3, 0.2, 1e-3, 0.4, 1e-3],
    })
    pred = pl.DataFrame({
        "target": ["A", "A", "A", "B", "B", "C"],
        "feature": ["g1", "g9", "g3", "g4", "g5", "g7"],
        "log2_fold_change": [4.0, 2.0, 1.0, 5.0, 0.4, 2.0],
        "p_value": [1e-4, 1e-4, 1e-2, 1e-5, 0.3, 1e-3],
        "fdr": [1e-3, 1e-3, 0.2, 1e-3, 0.4, 1e-3],
    })
    return pred, real


@pytest.mark.parametrize("metric", ["overlap", "precision"])
@pytest.mark.parametrize("k", [None, 50, 100, 200, 500])
def test_de_overlap_parity_vs_upstream(metric, k):
    from cell_eval._types import initialize_de_comparison, DESortBy
    from cell_eval.metrics._de import de_overlap_metric
    from cell_eval2.de import prepare_de
    from cell_eval2.metrics.de import de_overlap

    pred, real = _parity_pair()
    expected = de_overlap_metric(
        initialize_de_comparison(real, pred), k=k, metric=metric,
        fdr_threshold=0.05, sort_by=DESortBy.ABS_LOG2_FOLD_CHANGE,
    )
    prep = prepare_de(pred, real, control="non-targeting",
                      sort_by="abs_log2_fold_change", p_adj_threshold=0.05,
                      nan_lfc_policy="keep")  # v1 behavior matches upstream
    got = de_overlap(prep, k=k, metric=metric)
    assert set(got) == set(expected)
    for pert in expected:
        assert got[pert] == pytest.approx(expected[pert]), f"{metric} k={k} pert={pert}"


def test_de_metrics_parity_vs_cell_eval():
    import math
    from cell_eval._types import initialize_de_comparison
    from cell_eval.metrics._de import (
        DEDirectionMatch,
        DENsigCounts,
        DESigGenesRecall,
        DESpearmanLFC,
        DESpearmanSignificant,
        compute_pr_auc,
        compute_roc_auc,
    )
    from cell_eval2.de import prepare_de
    from cell_eval2.metrics.de import (
        de_direction_match,
        de_lfc_spearman,
        de_nsig_counts,
        de_nsig_spearman,
        de_pr_auc,
        de_roc_auc,
        de_sig_recall,
    )

    rng = np.random.default_rng(0)
    targets = ["G1", "G2", "G3"]
    feats = [f"F{i}" for i in range(20)]

    def make(shift):
        rows = []
        for target in targets:
            for feature in feats:
                lfc = float(rng.normal(shift, 1.0))
                padj = float(rng.uniform(0, 1))
                rows.append((target, feature, lfc, padj))
        return pl.DataFrame(
            rows, schema=["target", "feature", "log2_fold_change", "p_adj"], orient="row"
        )

    real_c2 = make(0.5)
    pred_c2 = make(0.3)

    def to_upstream(df):
        return df.with_columns(
            pl.col("p_adj").alias("fdr"),
            pl.col("p_adj").alias("p_value"),
        ).drop("p_adj")

    real_up, pred_up = to_upstream(real_c2), to_upstream(pred_c2)

    prep = prepare_de(pred_c2, real_c2, control="non-targeting", p_adj_threshold=0.05)
    cmp = initialize_de_comparison(real=real_up, pred=pred_up, fdr_col="fdr")

    up_counts = DENsigCounts(0.05)(cmp)
    c2_real = de_nsig_counts(prep, side="real")
    c2_pred = de_nsig_counts(prep, side="pred")
    for pert in targets:
        assert c2_real[pert] == float(up_counts[pert]["real"])
        assert c2_pred[pert] == float(up_counts[pert]["pred"])

    up_nsig_sp = DESpearmanSignificant(0.05)(cmp)
    c2_nsig_sp = next(iter(de_nsig_spearman(prep).values()))
    assert math.isclose(c2_nsig_sp, up_nsig_sp, rel_tol=1e-6, abs_tol=1e-9)

    parity_pairs = [
        (de_sig_recall, DESigGenesRecall(0.05)),
        (de_direction_match, DEDirectionMatch(0.05)),
        (de_lfc_spearman, DESpearmanLFC(0.05)),
    ]
    # pos/neg LFC-Spearman parity only when the installed cell-eval's DESpearmanLFC supports
    # `lfc_direction` (present on the gpu branch pinned upstream; omitted by older releases like the
    # one in this venv). The pos/neg unit tests in test_de.py cover correctness regardless.
    if "lfc_direction" in inspect.signature(DESpearmanLFC).parameters:
        parity_pairs += [
            (functools.partial(de_lfc_spearman, lfc_direction="pos"),
             DESpearmanLFC(0.05, lfc_direction="pos")),
            (functools.partial(de_lfc_spearman, lfc_direction="neg"),
             DESpearmanLFC(0.05, lfc_direction="neg")),
        ]
    for c2_fn, up_obj in parity_pairs:
        c2 = c2_fn(prep)
        up = up_obj(cmp)
        for pert in up:
            assert math.isclose(c2[pert], up[pert], rel_tol=1e-6, abs_tol=1e-9) or (
                math.isnan(c2[pert]) and math.isnan(up[pert])
            )

    for c2_fn, up_fn in [(de_pr_auc, compute_pr_auc), (de_roc_auc, compute_roc_auc)]:
        c2, up = c2_fn(prep), up_fn(cmp)
        for pert in targets:
            a, b = c2[pert], up[pert]
            assert (math.isnan(a) and math.isnan(b)) or math.isclose(
                a, b, rel_tol=1e-6, abs_tol=1e-9
            )


def test_v1_de_compute_matches_pdex_cell_eval_convention(synthetic_pair):
    # cell_eval computes DE via pdex with geometric_mean=True (default), epsilon=0.0
    # (default), is_log1p=True for lognorm input (cell_eval/_evaluator._build_pdex_kwargs).
    # Our v1 compute_de computes the geometric+eps=0 LFC itself; assert it reproduces
    # pdex's native LFC at the project rel=1e-6 parity bar.
    from pdex import pdex
    from cell_eval2.de_compute import compute_de

    _pred, real = synthetic_pair  # lognorm fixture
    ours = compute_de(real, backend="pdex", groupby="target", reference="non-targeting",
                      mean_calc="geometric", epsilon=0.0, input_type="lognorm",
                      filter_gene_min_cpm_cell=None).sort(["target", "feature"])
    ref = pdex(real, groupby="target", mode="ref", reference="non-targeting",
               geometric_mean=True, epsilon=0.0, is_log1p=True, threads=1
               ).select(["target", "feature", "log2_fold_change"]).sort(["target", "feature"])
    j = ours.select(["target", "feature", "log2_fold_change"]).join(
        ref, on=["target", "feature"], suffix="_ref")
    assert j.height > 0
    a = j["log2_fold_change"].to_numpy()
    b = j["log2_fold_change_ref"].to_numpy()
    finite = np.isfinite(a) & np.isfinite(b)
    assert finite.any()
    assert np.allclose(a[finite], b[finite], rtol=1e-6, atol=1e-9)


def test_de_auc_parity_vs_cell_eval_divergent_regime():
    # The clean metric-parity proof of the v1 path: cev2's replace_zero reproduces
    # cell-eval's compute_pr_auc/compute_roc_auc bit-for-bit IN THE DIVERGENT REGIME
    # (exact 0 + sub-1e-10 pred p-values), which the uniform(0,1) parity test never reaches.
    import math
    from cell_eval._types import initialize_de_comparison
    from cell_eval.metrics._de import compute_pr_auc, compute_roc_auc
    from cell_eval2.de import prepare_de
    from cell_eval2.metrics.de import de_pr_auc, de_roc_auc

    real = pl.DataFrame({
        "target": ["G1", "G1", "G1", "G2", "G2", "G2"],
        "feature": ["a", "b", "c", "a", "b", "c"],
        "log2_fold_change": [3.0, 2.0, 1.0, 3.0, 2.0, 1.0],
        "p_adj": [0.001, 0.9, 0.9, 0.001, 0.001, 0.9],
    })
    pred = pl.DataFrame({
        "target": ["G1", "G1", "G1", "G2", "G2", "G2"],
        "feature": ["a", "b", "c", "a", "b", "c"],
        "log2_fold_change": [3.0, 2.0, 1.0, 3.0, 2.0, 1.0],
        "p_adj": [0.0, 1e-50, 0.9, 1e-300, 0.0, 0.5],
    })

    prep = prepare_de(pred, real, control="non-targeting", p_adj_threshold=0.05)

    def to_upstream(df):
        return df.with_columns(pl.col("p_adj").alias("fdr"),
                               pl.col("p_adj").alias("p_value")).drop("p_adj")
    cmp = initialize_de_comparison(real=to_upstream(real), pred=to_upstream(pred), fdr_col="fdr")

    for c2_fn, up_fn in [(de_pr_auc, compute_pr_auc), (de_roc_auc, compute_roc_auc)]:
        c2 = c2_fn(prep, auc_pval_floor="replace_zero", auc_pval_floor_value=1e-10)
        up = up_fn(cmp)
        for pert in ("G1", "G2"):
            a, b = c2[pert], up[pert]
            assert (math.isnan(a) and math.isnan(b)) or math.isclose(
                a, b, rel_tol=1e-6, abs_tol=1e-9
            ), f"{c2_fn.__name__}[{pert}]: cev2={a} cell-eval={b}"
