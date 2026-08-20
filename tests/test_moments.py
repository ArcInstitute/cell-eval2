import anndata as ad
import numpy as np
import pandas as pd
import pytest
from scipy.sparse import csr_matrix

from cell_eval2.moments import (GroupMoments, trace_over_n_for, trace_sigma,
                                unbiased_sq_dist)
from cell_eval2.prep import pseudobulk, pseudobulk_with_moments


def test_trace_sigma_matches_var_ddof1_sum():
    rng = np.random.default_rng(0)
    X = rng.normal(3.0, 1.5, size=(40, 25))
    means = X.mean(axis=0)[None, :]
    counts = np.array([40.0])
    sumsq = np.array([float(np.sum(X * X))])
    got = trace_sigma(counts, sumsq, means)
    assert got.shape == (1,)
    assert got[0] == pytest.approx(float(X.var(axis=0, ddof=1).sum()), rel=1e-10)


def test_trace_sigma_nan_below_two_cells():
    means = np.zeros((2, 3))
    got = trace_sigma(np.array([1.0, 0.0]), np.array([5.0, 0.0]), means)
    assert np.isnan(got).all()


def test_trace_over_n_for_aligns_by_label_and_tolerates_extra_groups():
    # moments span 3 groups; the bulk carries only 2, in a different order.
    perts_all = np.array(["A", "B", "ctrl"])
    counts = np.array([10.0, 20.0, 30.0])
    sumsq = np.array([100.0, 400.0, 900.0])
    m = GroupMoments(perts=perts_all, counts=counts, sumsq=sumsq)
    bulk_perts = np.array(["B", "A"])
    bulk_means = np.zeros((2, 4))  # ||mu||^2 = 0 -> trace = sumsq/(n-1)
    got = trace_over_n_for(m, bulk_perts, bulk_means)
    assert got[0] == pytest.approx(400.0 / 19.0 / 20.0)
    assert got[1] == pytest.approx(100.0 / 9.0 / 10.0)


def test_trace_over_n_for_raises_when_moments_were_restricted():
    m = GroupMoments(perts=np.array(["A"]), counts=np.array([5.0]), sumsq=np.array([1.0]))
    with pytest.raises(ValueError, match="never be restricted"):
        trace_over_n_for(m, np.array(["A", "B"]), np.zeros((2, 3)))


def test_group_moments_rejects_mismatched_lengths():
    with pytest.raises(ValueError, match="same length"):
        GroupMoments(perts=np.array(["A", "B"]), counts=np.array([1.0]),
                     sumsq=np.array([1.0, 2.0]))


def test_trace_over_n_for_zero_count_RAISES():
    """#227, the inversion of `test_trace_over_n_for_zero_count_is_zero_without_a_numpy_warning`.

    n == 0 was never the #219 case: an empty group has no sample mean either, so its bulk row is
    meaningless and a zero correction was a fallback for a state that cannot occur rather than a
    modelled quantity. It does not arise from `pseudobulk_with_moments` (groups come from
    observed labels), so reaching here means a corrupt cache, a malformed stream, or a hand-built
    `GroupMoments` like this one.

    Still pinned under `errstate(all="raise")`: the validation must reach its own error without
    tripping a numpy divide/invalid signal on the way (`counts == 0` used to be the operand of a
    NaN/0 division this branch existed to avoid)."""
    m = GroupMoments(perts=np.array(["A"]), counts=np.array([0.0]), sumsq=np.array([0.0]))
    with np.errstate(all="raise"):
        with pytest.raises(ValueError, match="finite whole number"):
            trace_over_n_for(m, np.array(["A"]), np.zeros((1, 3)))


@pytest.mark.parametrize("label,count", [("nan", np.nan), ("neg", -3.0), ("frac", 1.5),
                                         ("inf", np.inf), ("zero", 0.0)])
def test_trace_over_n_for_malformed_counts_RAISE(label, count):
    """#227, the inversion of `test_trace_over_n_for_absorbs_malformed_counts_silently`.

    Each of these used to take the `counts < 2` branch and yield a 0.0 correction, silently --
    malformed moments became a plausible finite number instead of an error. Parametrized one per
    value so a fix that catches only some of them cannot pass on the strength of the others.

    `errstate(all="raise")` is kept from the original for the reason it was added: a NaN count is
    an OPERAND of the comparisons that classify it, which is where an invalid-value signal would
    come from if numpy ever reinstated one."""
    m = GroupMoments(perts=np.array([label]), counts=np.array([count]),
                     sumsq=np.array([0.0]))
    with np.errstate(all="raise"):
        with pytest.raises(ValueError, match="finite whole number"):
            trace_over_n_for(m, np.array([label]), np.zeros((1, 3)))


def test_trace_over_n_for_error_names_the_offending_perturbations():
    """The error has to name the perturbation the CALLER asked for, not a row index of a moments
    artifact it never sees -- the same reason the missing-label check a few lines above quotes
    the label. Sample capped at 5, so the sixth is elided rather than printed."""
    labels = [f"P{i}" for i in range(7)]
    m = GroupMoments(perts=np.array(labels), counts=np.zeros(7), sumsq=np.zeros(7))
    with pytest.raises(ValueError) as exc:
        trace_over_n_for(m, np.array(labels), np.zeros((7, 3)))
    msg = str(exc.value)
    assert "7 of 7 perturbations" in msg
    assert "'P0'" in msg and "'P4'" in msg      # first five are quoted
    assert "'P5'" not in msg and "'P6'" not in msg   # the cap holds
    assert "..." in msg
    # The casts before `!r` are load-bearing and the positive assertions above do NOT discriminate
    # -- `"'P0'"` is a substring of `np.str_('P0')` too. Under NumPy 2 a scalar reprs as its
    # constructor call, so dropping either cast leaks `np.str_(...)` / `np.float64(...)` into the
    # message while every assertion above still passes. Asserting the delimiter CONTEXT is what no
    # scalar repr can satisfy, and it does not depend on how a given NumPy words its reprs.
    assert ">= 1: 'P0': 0.0," in msg, "label and count must render bare: \"'P0': 0.0\""
    assert ", 'P4': 0.0," in msg
    assert "np.str_" not in msg and "np.float64" not in msg


def test_the_count_check_is_clean_under_errstate_all_raise():
    """A masked variant was proposed in review (PR #302) on the theory that comparing and rounding
    NaN/+-inf can raise `FloatingPointError` under `np.errstate(all="raise")`. Refuted by
    measurement and pinned here: numpy's comparisons use non-signaling predicates and `rint` of
    NaN/inf is exact, so the check raises the designed `ValueError` and nothing else."""
    m = GroupMoments(perts=np.array(["A", "B", "C"]),
                     counts=np.array([np.nan, np.inf, 4.0]),
                     sumsq=np.array([1.0, 1.0, 40.0]))
    with np.errstate(all="raise"):
        with pytest.raises(ValueError, match="finite whole number"):
            trace_over_n_for(m, np.array(["A", "B", "C"]), np.zeros((3, 2)))
        # and the all-good path must not trip it either
        ok = GroupMoments(perts=np.array(["A"]), counts=np.array([4.0]), sumsq=np.array([40.0]))
        assert np.isfinite(trace_over_n_for(ok, np.array(["A"]), np.zeros((1, 2)))).all()


def test_trace_over_n_for_validates_only_the_ALIGNED_rows():
    """The moments span every group and are never restricted, so a malformed count on a group the
    caller did not ask about must not fail the call. Only the rows actually looked up are
    validated -- which is also what makes the error message able to name them."""
    m = GroupMoments(perts=np.array(["A", "B"]), counts=np.array([4.0, 0.0]),
                     sumsq=np.array([40.0, 0.0]))
    got = trace_over_n_for(m, np.array(["A"]), np.array([[2.0, 1.0]]))
    assert np.isfinite(got).all()
    with pytest.raises(ValueError, match="finite whole number"):
        trace_over_n_for(m, np.array(["A", "B"]), np.zeros((2, 2)))


def test_trace_over_n_for_single_cell_group_subtracts_nothing():
    """n == 1 is the #219 case proper: one cell has no sample covariance, so the correction
    is zero and the perturbation keeps a finite (deliberately un-corrected) value rather than
    going NaN and being dropped from the aggregate."""
    m = GroupMoments(perts=np.array(["A", "B"]), counts=np.array([1.0, 4.0]),
                     sumsq=np.array([9.0, 40.0]))
    means = np.array([[3.0, 0.0], [2.0, 1.0]])
    got = trace_over_n_for(m, np.array(["A", "B"]), means)
    assert got[0] == 0.0                      # n == 1 -> subtract nothing
    assert got[1] > 0.0 and np.isfinite(got[1])


def test_unbiased_sq_dist_is_summed_not_averaged():
    """#202's m_p is summed over genes; only expr_mse applies 1/G, at its own call site."""
    pred = np.array([[1.0, 2.0, 2.0]])
    real = np.zeros((1, 3))
    got = unbiased_sq_dist(pred, real, np.array([0.0]), np.array([0.0]))
    assert got[0] == pytest.approx(1.0 + 4.0 + 4.0)          # summed, not /3


def test_unbiased_sq_dist_subtracts_both_trace_terms_and_propagates_nan():
    pred, real = np.array([[3.0, 4.0]]), np.zeros((1, 2))
    got = unbiased_sq_dist(pred, real, np.array([5.0]), np.array([2.0]))
    assert got[0] == pytest.approx(25.0 - 5.0 - 2.0)
    nan_out = unbiased_sq_dist(pred, real, np.array([np.nan]), np.array([2.0]))
    assert np.isnan(nan_out[0])


def _toy(sparse=False):
    rng = np.random.default_rng(7)
    labels = np.repeat(["ctrl", "A", "B"], 12)
    X = rng.lognormal(0.0, 1.0, size=(labels.size, 9))
    X[X < 0.5] = 0.0  # make sparsity real
    Xm = csr_matrix(X) if sparse else X
    obs = pd.DataFrame({"target": labels}, index=[str(i) for i in range(labels.size)])
    return ad.AnnData(X=Xm, obs=obs), X, labels


@pytest.mark.parametrize("sparse", [False, True])
def test_pseudobulk_with_moments_matches_var_ddof1_sum(sparse):
    adata, X, labels = _toy(sparse)
    perts, means, m = pseudobulk_with_moments(adata, "target")
    ref_perts, ref_means = pseudobulk(adata, "target")
    assert list(perts) == list(ref_perts)
    np.testing.assert_allclose(means, ref_means, rtol=0, atol=0)
    assert list(m.perts) == list(perts)
    for i, p in enumerate(perts):
        sub = X[labels == p]
        assert m.counts[i] == pytest.approx(float(sub.shape[0]))
        expect = float(sub.var(axis=0, ddof=1).sum())
        got = trace_sigma(m.counts[i:i + 1], m.sumsq[i:i + 1], means[i:i + 1])[0]
        assert got == pytest.approx(expect, rel=1e-10)


def test_pseudobulk_with_moments_sparse_dense_identical():
    dense, _, _ = _toy(False)
    sparse, _, _ = _toy(True)
    _, _, md = pseudobulk_with_moments(dense, "target")
    _, _, ms = pseudobulk_with_moments(sparse, "target")
    np.testing.assert_allclose(md.counts, ms.counts, rtol=0, atol=0)
    np.testing.assert_allclose(md.sumsq, ms.sumsq, rtol=1e-12, atol=0)


@pytest.mark.parametrize("n_cells", [2, 50, 300])
def test_fp32_means_perturb_the_trace_only_negligibly(n_cells):
    """`sumsq` is always fp64 but the accumulator's means are fp32
    (`gpu.bulk.GroupedMeanAccumulator.finalize` casts), so `sumsq - n*||mu||^2` -- a
    near-cancellation -- runs in mixed precision on `inmem_pseudobulk` (either device) and the
    GPU streaming path. This is a REGRESSION TRIPWIRE on one fixture, not a universal bound.

    Built in `log1p(CPM)`, the space the metric actually consumes, and measured on the two
    quantities that matter: the trace's own relative deviation, and `|d tr|/(n*G)` -- the
    absolute amount that reaches `expr_mse_unbiased` from one side. Small `n` is the adverse
    axis (the error scales ~1/(n-1)), NOT pred/real being drawn from the same population,
    which is worst-conditioned for the distance rather than for the trace.

    Measured on this fixture (G=2000): |d tr|/(n*G) = 3.3e-07 / 1.2e-09 / 1.1e-10 at
    n = 2 / 50 / 300, against a correction term `tr/n/G` of 2.37 / 0.094 / 0.016 -- i.e. the
    fp32 mean perturbs the subtracted correction by at most 1.5e-07 of its own size. The
    bounds below carry ~7x and ~9x margin over those. Do NOT tighten them below the measured
    values. If this FAILS, the fp32 mean has become material -- see `trace_sigma`'s docstring
    for the exact fix (GroupMoments carrying its own fp64 ||mu||^2)."""
    from cell_eval2 import norm as _norm

    n_groups, n_genes = 6, 2000
    rng = np.random.default_rng(4)
    labels = np.repeat([f"P{i}" for i in range(n_groups)], n_cells)
    lam = np.random.default_rng(999).gamma(0.4, 3.0, size=n_genes)
    X = rng.poisson(lam, size=(labels.size, n_genes)).astype(np.float32)
    obs = pd.DataFrame({"target": labels}, index=[str(i) for i in range(labels.size)])
    var = pd.DataFrame(index=[f"g{i}" for i in range(n_genes)])
    lognorm = _norm.to_normalization(
        ad.AnnData(X=csr_matrix(X), obs=obs, var=var), "counts", "lognorm", target_sum=1e6)

    _, means64, mom = pseudobulk_with_moments(lognorm, "target")
    exact = trace_sigma(mom.counts, mom.sumsq, means64)
    via_fp32 = trace_sigma(mom.counts, mom.sumsq, means64.astype(np.float32))
    assert np.all(exact > 0)  # a real panel's trace is positive; the deviation is pure noise

    delta = np.abs(via_fp32 - exact)
    assert (delta / exact).max() < 1e-6, (delta / exact).max()
    reaching_the_metric = delta / (mom.counts * n_genes)
    assert reaching_the_metric.max() < 3e-6, reaching_the_metric.max()


def test_sumsq_is_correct_on_a_noncanonical_csr():
    """A CSR may legally hold duplicate coordinates, and tocsr() on an existing CSR returns
    it UNCHANGED. .mean() sums duplicates (so means are right); squaring them separately is
    NOT -- (1+2)^2 = 9, not 1^2 + 2^2 = 5. Verified by hand-building the duplicate."""
    dup = csr_matrix((np.array([1.0, 2.0, 4.0]), np.array([0, 0, 1]), np.array([0, 3])),
                     shape=(1, 2))
    assert dup.has_canonical_format is False and dup.tocsr() is dup   # the trap, pinned
    obs = pd.DataFrame({"target": ["A"]}, index=["0"])
    adata = ad.AnnData(X=dup, obs=obs)
    _, means, m = pseudobulk_with_moments(adata, "target")
    np.testing.assert_allclose(means[0], [3.0, 4.0])          # duplicates summed
    assert m.sumsq[0] == pytest.approx(3.0**2 + 4.0**2)       # NOT 1 + 4 + 16 = 21
