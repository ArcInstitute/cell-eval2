import numpy as np
import pytest

from _helpers import _dispatch_cfg
from cell_eval2.metrics.delta import (
    distance_unbiased,
    mae,
    mse,
    mse_unbiased,
    mse_unbiased_capped,
    pearson_delta,
)
from cell_eval2.moments import GroupMoments
from cell_eval2.prep import pseudobulk


def test_mse_accepts_one_bulk_one_anndata(synthetic_pair):
    # Hybrid input: a supplied real_bulk + pred AnnData must work. Before the
    # _resolve_bulks fix (Gemini impl review) this raised, because a single missing
    # bulk forced recomputing both and required both AnnData.
    pred, real = synthetic_pair
    real_bulk = pseudobulk(real, "target")
    out = mse(pred=pred, real_bulk=real_bulk, pert_col="target", control="non-targeting")
    assert set(out) == {"GENE1", "GENE2", "GENE3"}
    # Equivalent to the all-AnnData call (same real_bulk, just supplied vs computed).
    both = mse(pred=pred, real=real, pert_col="target", control="non-targeting")
    assert out == both


def test_delta_supplied_bulk_is_reused_not_recomputed(synthetic_pair):
    # A supplied bulk is reused verbatim; only the missing side is built from AnnData.
    pred, real = synthetic_pair
    pred_bulk = pseudobulk(pred, "target")
    # Sentinel that is NOT valid to pseudobulk-recompute (no AnnData passed for pred):
    out = pearson_delta(pred_bulk=pred_bulk, real=real, pert_col="target",
                        control="non-targeting", control_source="pred")
    assert set(out) == {"GENE1", "GENE2", "GENE3"}
    assert all(np.isfinite(v) for v in out.values())


def test_mae_accepts_one_bulk_one_anndata(synthetic_pair):
    # F6.1: mae must resolve bulks per-side (like mse/pearson_delta), not all-or-nothing. A supplied
    # real_bulk + pred AnnData must work; before the fix it raised (the missing pred_bulk forced
    # recomputing BOTH sides and required a real AnnData that was not passed).
    pred, real = synthetic_pair
    real_bulk = pseudobulk(real, "target")
    out = mae(pred=pred, real_bulk=real_bulk, pert_col="target", control="non-targeting")
    assert set(out) == {"GENE1", "GENE2", "GENE3"}
    both = mae(pred=pred, real=real, pert_col="target", control="non-targeting")
    assert out == both


def test_mae_supplied_bulk_reused_not_overwritten(synthetic_pair):
    # F6.1: a supplied pred_bulk must be reused verbatim, not silently overwritten by
    # pseudobulk(pred) when the OTHER side is given as AnnData. Feed a custom pred_bulk that differs
    # from pred's own pseudobulk; the result must reflect it (the all-recompute path would ignore it).
    pred, real = synthetic_pair
    perts, means = pseudobulk(pred, "target")
    custom = (perts, means + 1.0)      # different means -> different MAE than pred's own pseudobulk
    out_custom = mae(pred=pred, real=real, pred_bulk=custom, pert_col="target",
                     control="non-targeting")
    out_recompute = mae(pred=pred, real=real, pert_col="target", control="non-targeting")
    assert out_custom != out_recompute, "supplied pred_bulk was overwritten by pseudobulk(pred) (F6.1)"


def _fixture(jk_pred, jk_real):
    perts = np.array(["non-targeting", "A"])
    pb = np.array([[1.0, 2.0], [3.0, 5.0]])
    rb = np.array([[1.0, 2.0], [4.0, 7.0]])
    mp = GroupMoments(
        perts=perts,
        counts=np.array([4.0, 4.0]),
        sumsq=np.array([30.0, 140.0]),
        jk=jk_pred,
    )
    mr = GroupMoments(
        perts=perts,
        counts=np.array([4.0, 4.0]),
        sumsq=np.array([30.0, 270.0]),
        jk=jk_real,
    )
    return (perts, pb), (perts, rb), mp, mr


# MEASURED against the real funcs: e = 5.0, G = 2, PRED_TRACE_CAP_K = 1.0. C_pred=0.9 > C_real=0.6
# so min() BINDS -- rev 2 used (0.4, 0.6), where capped and uncapped are both exactly 2.0 and an
# uncapped implementation passes the capped case (finding h).
JK_PRED, JK_REAL = np.array([0.0, 0.9]), np.array([0.0, 0.6])


@pytest.mark.parametrize(
    "func,expected",
    [
        (mse_unbiased, 1.75),  # (5 - 0.9 - 0.6) / 2
        (mse_unbiased_capped, 1.9),  # (5 - min(0.9, 1.0 * 0.6) - 0.6) / 2
    ],
)
def test_the_unbiased_numerators_use_jk_under_bulk_lognorm(func, expected, monkeypatch):
    """EXACT formula from the stored jk, not merely 'different from lognorm'. An
    implementation applying any arbitrary comparator-dependent offset passes an inequality
    assertion; only the formula pins it. Both constants are measured, and they DIFFER from each
    other -- which is the part rev 2's fixture could not express.

    ⚠️ #348's budget is DISABLED here. This fixture scores ONE perturbation, and an across-
    perturbation spread is not merely noisy at n=1, it does not exist -- so #348 withholds the
    whole correction (`r = 0`, the conservative direction rather than a bypass) and the capped
    value would read `(5 - 0 - 0.6)/2 = 2.2`. What is under test here is the jk formula and #247's
    cap; #348's factor has its own file (`tests/test_pred_correction_bound_348.py`), including the
    one-row policy.
    """
    monkeypatch.setattr("cell_eval2.metrics.delta._across_pert_budget",
                        lambda *_a, **_k: float("inf"))
    pb, rb, mp, mr = _fixture(JK_PRED, JK_REAL)
    assert float(((pb[1][1] - rb[1][1]) ** 2).sum()) == 5.0
    got = func(
        pred_bulk=pb,
        real_bulk=rb,
        pred_moments=mp,
        real_moments=mr,
        control="non-targeting",
        comparator="bulk_lognorm",
        driver="t",
    )
    assert got["A"] == pytest.approx(expected)


def test_the_cap_binds_so_capped_and_uncapped_disagree():
    """#247's cap is the one thing a shared formula cannot express, and the ONLY guard against
    an implementation that threads `comparator` into `mse_unbiased` and forgets
    `mse_unbiased_capped`. Stated as an inequality on top of the two exact values above, so a
    future fixture edit that accidentally makes the cap inert fails HERE with a clear reason
    rather than making two other tests silently redundant."""
    pb, rb, mp, mr = _fixture(JK_PRED, JK_REAL)
    kw = dict(
        pred_bulk=pb,
        real_bulk=rb,
        pred_moments=mp,
        real_moments=mr,
        control="non-targeting",
        comparator="bulk_lognorm",
        driver="t",
    )
    assert mse_unbiased_capped(**kw)["A"] != pytest.approx(mse_unbiased(**kw)["A"])


def test_distance_unbiased_uses_jk_under_bulk_lognorm():
    """The THIRD function. Rev 1 tested only mse_unbiased, so leaving capped and distance on
    the analytic path passed. 16.575 == (34 - 0.6 - 0.25) / 2, measured."""
    _, rb, _, mr = _fixture(None, np.array([0.25, 0.6]))
    assert float(((rb[1][1] - rb[1][0]) ** 2).sum()) == 34.0
    got = distance_unbiased(
        real_bulk=rb,
        real_moments=mr,
        control="non-targeting",
        comparator="bulk_lognorm",
        driver="t",
    )
    assert got["A"] == pytest.approx(16.575)


def test_distance_unbiased_is_invariant_to_the_pred_control_correction():
    """P0-d. `distance_unbiased` accepts `pred_bulk`/`pred_moments` for dispatch compatibility
    and DELIBERATELY NEVER READS them (`delta.py:409-413`) -- measured bit-identical. The
    pred-side correction still has to EXIST for the numerators (`_aligned_pair` computes
    `pred_tn` over all pred labels at `delta.py:241`, before the control is dropped), so the
    two facts are easy to conflate. Pin the invariance so a well-meaning 'align to the
    prediction' change fails here."""
    pb, rb, mp, mr = _fixture(JK_PRED, np.array([0.25, 0.6]))
    bare = distance_unbiased(
        real_bulk=rb,
        real_moments=mr,
        control="non-targeting",
        comparator="bulk_lognorm",
        driver="t",
    )
    withpred = distance_unbiased(
        pred_bulk=pb,
        real_bulk=rb,
        pred_moments=mp,
        real_moments=mr,
        control="non-targeting",
        comparator="bulk_lognorm",
        driver="t",
    )
    assert withpred == bare


@pytest.mark.parametrize("func", [mse_unbiased, mse_unbiased_capped, distance_unbiased])
def test_the_unbiased_numerators_raise_without_a_comparator(func):
    """All THREE (finding h): rev 2 omitted distance_unbiased, whose `comparator` is the one
    most easily forgotten because it reaches the correction through `_real_rows` rather than
    `_aligned_pair`. One body covers all three -- distance_unbiased accepts the pred kwargs and
    ignores them (verified above)."""
    pb, rb, mp, mr = _fixture(JK_PRED, JK_REAL)
    with pytest.raises(TypeError):
        func(
            pred_bulk=pb,
            real_bulk=rb,
            pred_moments=mp,
            real_moments=mr,
            control="non-targeting",
            driver="t",
        )


def test_the_dispatcher_supplies_the_lognorm_fallback_comparator():
    """The orchestrator gate, and the assertion that P0-a is implemented. Every test above
    calls the metric func DIRECTLY, so omitting "comparator" from run.py's `available` dict
    (run.py:298-322) passes all of them and then fails at runtime deep inside dispatch.

    After Task 7 every anndata metric declares the comparator token, so the effective key and
    run comparator are necessarily equal. This test retains the fallback half of the gate: the
    dispatcher must supply "lognorm" and the jk-less moments must take the analytic branch.
    The Task 7 mirror in test_expr_comparator_move.py proves the bulk-lognorm/jk half.
    """
    from cell_eval2.run import dispatch_anndata_metrics

    pb, rb, mp, mr = _fixture(None, None)
    # ⚠️ Issue #172: this metric drops each perturbation's own gene and RAISES when no target
    # resolves, so the dispatch needs a gene named 'A'. Appending an ALL-ZERO third gene makes
    # the target resolve while leaving the expected value EXACTLY 1.9166666666666667 -- a zero
    # column adds 0 to the squared distance, 0 to `sumsq` and 0 to `||mu||^2` (so `tr Sigma` is
    # unchanged), and it is the column the exclusion then removes, taking the divisor back to 2.
    # The direct-call tests above pass no `genes` and keep the 2-gene panel untouched.
    pb = (pb[0], np.hstack([pb[1], np.zeros((pb[1].shape[0], 1))]))
    rb = (rb[0], np.hstack([rb[1], np.zeros((rb[1].shape[0], 1))]))
    rows = dispatch_anndata_metrics(
        ["expr_mse_unbiased"],
        {"lognorm": pb},
        {"lognorm": rb},
        np.array(["g0", "g1", "A"]),
        _dispatch_cfg(),
        comparator="lognorm",
        pred_moments={"lognorm": mp},
        real_moments={"lognorm": mr},
        driver="test",
    )
    assert {r["perturbation"]: r["value"] for r in rows}["A"] == pytest.approx(
        1.9166666666666667
    )
