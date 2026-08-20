# tests/test_jackknife.py
import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from cell_eval2.moments import GroupMoments, correction_for, jackknife_correction
from cell_eval2.prep import (
    bulk_lognorm_means,
    pseudobulk_bulk_lognorm,
    pseudobulk_bulk_lognorm_with_moments,
)
from cell_eval2.streaming_bulk import _streaming_jackknife, inmem_pseudobulk

TS = 1_000_000.0


def _brute_force_C(Y: np.ndarray, ts: float) -> float:
    """Spec §3.6, written the slow obvious way: recompute the whole bulk n times."""
    n = Y.shape[0]
    if n < 2:
        return 0.0
    V = np.empty((n, Y.shape[1]), dtype=np.float64)
    for i in range(n):
        P_minus = Y[np.delete(np.arange(n), i)].sum(axis=0)
        tot = P_minus.sum()
        V[i] = np.log1p(ts * P_minus / tot) if tot > 0 else 0.0
    return float(((n - 1) / n) * ((V - V.mean(axis=0)) ** 2).sum())


def _panel(seed=0, per_group=15, g=25, groups=3):
    """per_group * groups rows and EXACTLY that many codes -- rev 1 built 40 rows and 39 codes."""
    rng = np.random.default_rng(seed)
    n = per_group * groups
    Y = rng.poisson(3.0, size=(n, g)).astype(np.float64)
    Y[:, 0] += 200                      # one high-expresser -> the log regime is exercised
    codes = np.repeat(np.arange(groups), per_group).astype(np.intp)
    assert codes.size == Y.shape[0]
    return sp.csr_matrix(Y), codes


@pytest.mark.parametrize("ts", [1e4, TS])
def test_jackknife_matches_brute_force_delete_one(ts):
    """Two target sums, each against its OWN brute-force oracle. A single TS with a
    monotonicity check would pass any arbitrary monotone rescaling of the right answer."""
    X, codes = _panel()
    got = jackknife_correction(X, codes, 3, ts)
    Y = X.toarray()
    for p in range(3):
        assert got[p] == pytest.approx(_brute_force_C(Y[codes == p], ts), rel=1e-10)


def test_jackknife_is_exactly_zero_and_never_negative_for_a_tile():
    """A tile is one row repeated: dropping any cell leaves the composition unchanged, so C is
    EXACTLY zero. MEASURED: the unshifted subtractive `s2 - s1**2/n` form returns
    -6.821210263296962e-13 here, which is why the clamp exists; the shifted reduction the
    implementation actually uses returns exactly 0.0. The clamp must be to zero, not to an
    epsilon.

    ⚠️ Asserted against the LITERAL 0.0, deliberately NOT against `_brute_force_C`: the oracle
    is not exact here either (it measures 9.466330862652142e-29), so `got == _brute_force_C(...)`
    would fail on a correct implementation. Exact oracle agreement is asserted on the
    `[[4,0],[0,0]]` fixture below, where the two DO agree bit for bit."""
    row = np.arange(1, 13, dtype=np.float64)
    X = sp.csr_matrix(np.tile(row, (7, 1)))
    got = jackknife_correction(X, np.zeros(7, dtype=np.intp), 1, TS)
    assert got[0] == 0.0


def test_jackknife_is_never_negative_on_a_real_panel():
    X, codes = _panel(seed=9)
    assert np.all(jackknife_correction(X, codes, 3, TS) >= 0.0)


def test_jackknife_returns_zero_below_two_cells():
    """Mirrors trace_over_n_for's n<2 policy: subtract nothing rather than NaN. (Rev 2 also
    called this "the only guard against r_i == 0"; it is not -- `_loo_bulk` handles r == 0
    explicitly for every n, and the `[[4,0],[0,0]]` test below is what pins that. Round 3, P2.)"""
    X = sp.csr_matrix(np.array([[3.0, 0.0, 5.0]]))
    assert jackknife_correction(X, np.zeros(1, dtype=np.intp), 1, TS)[0] == 0.0


def test_jackknife_returns_zero_for_an_empty_group():
    X, codes = _panel(groups=2)
    assert jackknife_correction(X, codes, 3, TS)[2] == 0.0


def test_a_group_whose_counts_sit_in_one_cell_matches_the_oracle_exactly():
    """Deleting the ONLY nonzero cell leaves an all-zero remainder -> a zero LOO bulk (the
    existing all-zero-bulk contract, prep.py:98). Deleting a zero cell leaves the bulk
    unchanged. The two differ, so C is POSITIVE. Rev 1's resident kernel returned 0 here and
    its streaming kernel returned NaN; both were wrong.

    Rev 2 asserted only `isfinite and > 0`, which ANY positive constant passes -- including a
    kernel that stumbled onto the right sign for the wrong reason. MEASURED: the kernel and the
    oracle agree BIT FOR BIT at 47.71708990205765, so both are asserted. This is the fixture
    that pins `_loo_bulk`'s `r_i == 0` policy, and it is the one edge case where the shared
    policy is observable, which is why Task 4 pushes it through every driver too."""
    Y = np.array([[4.0, 0.0], [0.0, 0.0]])
    got = jackknife_correction(sp.csr_matrix(Y), np.zeros(2, dtype=np.intp), 1, TS)[0]
    assert got == _brute_force_C(Y, TS)
    assert got == pytest.approx(47.71708990205765, rel=1e-12)


def test_jackknife_rejects_a_code_vector_of_the_wrong_length():
    X, _ = _panel()
    with pytest.raises(ValueError, match="group_codes"):
        jackknife_correction(X, np.zeros(3, dtype=np.intp), 3, TS)


def test_jackknife_accepts_unsorted_group_codes():
    """`prep` builds `codes` by scattering group ids back onto ORIGINAL row positions, so the
    vector reaching this kernel is not sorted. The argsort/searchsorted bounds pair is what
    makes that safe; a version that assumed contiguity would pass every fixture above (all of
    which use np.repeat) and be wrong on every real adata."""
    Y = np.arange(1, 25, dtype=np.float64).reshape(6, 4)
    codes = np.array([1, 0, 1, 0, 1, 0], dtype=np.intp)
    got = jackknife_correction(sp.csr_matrix(Y), codes, 2, TS)
    for p in (0, 1):
        assert got[p] == pytest.approx(_brute_force_C(Y[codes == p], TS), rel=1e-10)


def test_the_chunk_size_does_not_change_the_answer():
    """The dense temporary is bounded by re-entering `_loo_bulk` per block and accumulating
    s1/s2 across blocks. An off-by-one or a reassignment instead of `+=` shows up only here.

    It is also the tripwire for the shifted reduction, and the test that CAUGHT rev 5's
    numerical defect. MEASURED on this panel: the unshifted `s2 - s1²/n` form differs between
    `chunk=512` and `chunk=7` by a relative 5.551e-11 -- 555x the asserted tolerance -- while
    the shifted form differs by exactly 0.0. So a regression that drops the `- b[None, :]`
    shift fails here, at n=40, long before the n=4000 regime where it costs six significant
    digits (rel err 1.539e-06). Do NOT loosen this rtol; it is the whole point.

    VERIFIED BY MUTATION: deleting the shift from `_loo_bulk` reddens exactly this test.
    It does NOT catch a shift applied to only SOME kernels -- see
    `test_the_streaming_kernel_agrees_with_the_resident_one_BIT_FOR_BIT` for that half."""
    X, codes = _panel(seed=5, per_group=40, g=12, groups=2)
    np.testing.assert_allclose(jackknife_correction(X, codes, 2, TS, chunk=512),
                               jackknife_correction(X, codes, 2, TS, chunk=7), rtol=1e-13)


def test_jackknife_accepts_a_dense_matrix_and_a_duplicate_bearing_csr():
    """Two contracts no rev asserted.

    DENSE: `_side_bulks` hands `pseudobulk_bulk_lognorm_with_moments` whatever `adata.X` is,
    and a dense ndarray is legal input. `X_csr.tocsr()` is an AttributeError there (measured),
    which is why Task 1 converts rather than Task 2 papering over it (finding m / N1).

    DUPLICATES: unlike `_grouped_sumsq` -- which squares duplicate coordinates separately and
    therefore MUST canonicalize -- this kernel only ever sums them (`.sum(axis=0)`,
    `.toarray()`), so it is exact on a duplicate-bearing CSR with no `sum_duplicates()` call.
    MEASURED against the oracle. Asserted so nobody adds a redundant guard, or a wrong one."""
    dense = np.array([[4.0, 1.0], [2.0, 3.0], [1.0, 1.0]])
    got = jackknife_correction(dense, np.zeros(3, dtype=np.intp), 1, TS)
    assert got[0] == pytest.approx(_brute_force_C(dense, TS), rel=1e-10)

    dup = sp.csr_matrix((np.array([1.0, 2.0, 5.0]), np.array([0, 0, 1]),
                         np.array([0, 2, 3])), shape=(2, 2))
    assert not dup.has_canonical_format
    before = (dup.data.copy(), dup.indices.copy(), dup.indptr.copy())
    got = jackknife_correction(dup, np.zeros(2, dtype=np.intp), 1, TS)
    assert got[0] == pytest.approx(_brute_force_C(dup.toarray(), TS), rel=1e-9)
    # ⚠️ And it must not have CANONICALIZED the caller's matrix. `X.tocsr()` on an already-CSR
    # input returns the SAME OBJECT, so an implementation that "defensively" called
    # sum_duplicates() would mutate the caller in place and still pass the value assertion above
    # (round 4, P2). This is the assertion inmem_pseudobulk:262-268 refuses to need.
    assert not dup.has_canonical_format
    for got_arr, want_arr in zip((dup.data, dup.indices, dup.indptr), before):
        np.testing.assert_array_equal(got_arr, want_arr)


def _moments(jk):
    perts = np.array(["a", "b", "c"])
    return GroupMoments(perts=perts, counts=np.array([10.0, 10.0, 10.0]),
                        sumsq=np.array([60.0, 260.0, 130.0]), jk=jk), perts


def test_correction_for_reorders_and_subsets_by_label():
    """Rev 1 requested labels in stored order, so `return moments.jk` -- ignoring perts
    entirely -- passed. Ask for a REVERSED SUBSET."""
    m, _ = _moments(np.array([0.5, 0.25, 0.125]))
    means = np.zeros((2, 4))
    got = correction_for(m, np.array(["c", "a"]), means, comparator="bulk_lognorm")
    np.testing.assert_allclose(got, [0.125, 0.5])


def test_correction_for_raises_on_a_label_absent_from_the_moments():
    m, _ = _moments(np.array([0.5, 0.25, 0.125]))
    with pytest.raises(ValueError, match="absent"):
        correction_for(m, np.array(["a", "zz"]), np.zeros((2, 4)), comparator="bulk_lognorm")


def test_correction_for_lognorm_reads_the_analytic_trace():
    """The analytic branch is tested with jk=None so it cannot accidentally be reading jk."""
    m, perts = _moments(None)
    means = np.array([[1.0, 2.0], [3.0, 4.0], [2.0, 2.0]])
    from cell_eval2.moments import trace_over_n_for
    np.testing.assert_allclose(correction_for(m, perts, means, comparator="lognorm"),
                               trace_over_n_for(m, perts, means))


def test_correction_for_raises_when_bulk_lognorm_has_no_jk():
    m, perts = _moments(None)
    with pytest.raises(ValueError, match="bulk_lognorm.*jackknife"):
        correction_for(m, perts, np.zeros((3, 2)), comparator="bulk_lognorm")


def test_correction_for_raises_when_a_jk_bearing_artifact_is_read_as_lognorm():
    """A GroupMoments carrying jk was built in the group-sum space; its counts/sumsq describe
    counts space, so tr Sigma/n over it is a plausible wrong number. Refuse both directions."""
    m, perts = _moments(np.array([0.5, 0.25, 0.125]))
    with pytest.raises(ValueError, match="lognorm.*jackknife"):
        correction_for(m, perts, np.zeros((3, 2)), comparator="lognorm")


def test_correction_for_rejects_an_unknown_comparator():
    m, perts = _moments(None)
    with pytest.raises(ValueError, match="comparator"):
        correction_for(m, perts, np.zeros((3, 2)), comparator="counts")


def test_jk_length_is_validated():
    with pytest.raises(ValueError, match="same length"):
        GroupMoments(perts=np.array(["a", "b"]), counts=np.array([1.0, 2.0]),
                     sumsq=np.array([1.0, 2.0]), jk=np.array([1.0]))


def _adata(seed=1, per_group=10, g=18):
    rng = np.random.default_rng(seed)
    n = per_group * 3
    X = rng.poisson(4.0, size=(n, g)).astype(np.float64)
    X[:, 0] += 150
    obs = pd.DataFrame({"pert": np.repeat(["ctrl", "A", "B"], per_group)})
    return ad.AnnData(X=sp.csr_matrix(X), obs=obs)


def test_inmem_reference_matches_the_brute_force_correction():
    a = _adata()
    perts, _, mom = pseudobulk_bulk_lognorm_with_moments(a, "pert", bulk_target_sum=TS)
    Y = a.X.toarray()
    for i, p in enumerate(perts):
        rows = a.obs["pert"].to_numpy() == str(p)
        assert mom.jk[i] == pytest.approx(_brute_force_C(Y[rows], TS), rel=1e-10)


def test_inmem_reference_bulk_is_the_bulk_lognorm_of_the_group_sum():
    a = _adata()
    perts, means, _ = pseudobulk_bulk_lognorm_with_moments(a, "pert", bulk_target_sum=TS)
    Y = a.X.toarray()
    for i, p in enumerate(perts):
        rows = a.obs["pert"].to_numpy() == str(p)
        np.testing.assert_allclose(
            means[i], bulk_lognorm_means(Y[rows].sum(axis=0)[None, :], TS)[0], rtol=1e-12)


def _adata_over_the_fp32_boundary():
    """A group whose per-gene sum crosses 2**24, where an input-dtype reduction and a wide one
    STOP AGREEING. MEASURED on fp32 `[[16777216, 1], [1, 1]]`: reducing in the input dtype (what
    `_grouped_sums` did before #271) returns 16777216, reducing wide returns 16777217, and the
    resulting bulk differs by 3.53e-10 at the shipped TS = 5e4 (6.35e-09 at the 1e6 default #268
    retired -- the delta scales with TS, so it is meaningless without one). `_grouped_sums` now returns the wide answer for both
    dtypes, which is why this fixture's job is to make a dtype-INVARIANCE claim falsifiable.

    ⚠️ This fixture is the whole point of the test below. `_adata()`'s sums are ~1e2-1e4, where
    the two reductions are bit-identical (measured: zero difference), so parametrizing `_adata`
    over dtype tests NOTHING about precision -- round 4, P1."""
    X = np.array([[16777216.0, 1.0], [1.0, 1.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32)
    obs = pd.DataFrame({"pert": ["A", "A", "ctrl", "ctrl"]})
    # explicit var index: without one AnnData emits ImplicitModificationWarning ("Transforming
    # to str index"), and this repo's suite runs with warnings visible.
    return ad.AnnData(X=sp.csr_matrix(X), obs=obs,
                      var=pd.DataFrame(index=["g0", "g1"]))


@pytest.mark.parametrize("adata_factory,dtype", [
    (_adata, np.float64), (_adata, np.float32),
    (_adata_over_the_fp32_boundary, np.float32),          # the boundary case
])
def test_inmem_reference_bulk_is_BIT_IDENTICAL_to_the_non_moments_function(adata_factory, dtype):
    """The invariant the "fp64 reference" claim was standing in for. `pseudobulk_bulk_lognorm`
    is the shipped bulk; this function must return the SAME one, not a more-precise one, or a
    moments run and a non-moments run over identical input disagree about the pseudobulk.

    Bit equality (`assert_array_equal`, not allclose) is the point: an implementation that
    "helpfully" widened the reduction on THIS path only would pass a tolerance-based check, and on
    the third fixture it moves the bulk by 3.53e-10 at this file's TS -- small, but a different
    number than the one the non-moments path ships. #271 widened the reduction both paths share, so they still move
    together and this stays a bit-equality assertion rather than being retired."""
    a = adata_factory()
    a.X = a.X.astype(dtype)
    want_perts, want_means = pseudobulk_bulk_lognorm(a, "pert", bulk_target_sum=TS)
    got_perts, got_means, _ = pseudobulk_bulk_lognorm_with_moments(a, "pert",
                                                                   bulk_target_sum=TS)
    np.testing.assert_array_equal(got_perts, want_perts)
    np.testing.assert_array_equal(got_means, want_means)


def test_inmem_reference_bulk_sums_to_target_exactly():
    """Spec §6 test 1: the representability property, asserted directly."""
    _, means, _ = pseudobulk_bulk_lognorm_with_moments(_adata(), "pert", bulk_target_sum=TS)
    np.testing.assert_allclose(np.expm1(means).sum(axis=1), TS, rtol=1e-9)


def test_inmem_reference_counts_and_sumsq_stay_in_COUNTS_space():
    """These two fields are not read under bulk_lognorm, so nothing else would catch them
    being zeroed or written in the bulk space -- and the npz artifact stores them. Oracle
    them directly against the raw counts."""
    a = _adata(seed=7)
    perts, _, mom = pseudobulk_bulk_lognorm_with_moments(a, "pert", bulk_target_sum=TS)
    Y = a.X.toarray()
    for i, p in enumerate(perts):
        rows = a.obs["pert"].to_numpy() == str(p)
        assert mom.counts[i] == float(rows.sum())
        assert mom.sumsq[i] == pytest.approx(float((Y[rows] ** 2).sum()), rel=1e-12)


def test_inmem_reference_moments_carry_labels_aligned_with_the_bulk():
    perts, _, mom = pseudobulk_bulk_lognorm_with_moments(_adata(), "pert", bulk_target_sum=TS)
    np.testing.assert_array_equal(np.asarray(perts, dtype=str),
                                  np.asarray(mom.perts, dtype=str))


def test_inmem_reference_accepts_a_DENSE_adata():
    """A dense `adata.X` is legal input and reaches here through `_side_bulks`. Rev 2's snippet
    would NameError on it (`sp` is not imported in prep.py) and Task 1's kernel would
    AttributeError on it; both are fixed, and this is the test that says so from production's
    side rather than the kernel's."""
    a = _adata(seed=11)
    dense = ad.AnnData(X=a.X.toarray(), obs=a.obs.copy())
    p_s, m_s, mom_s = pseudobulk_bulk_lognorm_with_moments(a, "pert", bulk_target_sum=TS)
    p_d, m_d, mom_d = pseudobulk_bulk_lognorm_with_moments(dense, "pert", bulk_target_sum=TS)
    np.testing.assert_array_equal(p_s, p_d)
    np.testing.assert_allclose(m_s, m_d, rtol=1e-12)
    np.testing.assert_allclose(mom_s.jk, mom_d.jk, rtol=1e-12)


def test_inmem_driver_jk_matches_the_prep_reference():
    a = _adata(seed=4, per_group=20, g=22)
    _, _, ref = pseudobulk_bulk_lognorm_with_moments(a, "pert", bulk_target_sum=TS)
    _, mom = inmem_pseudobulk(a, pert_col="pert", norms=["bulk_lognorm"], target_sum=None,
                              with_moments=True, bulk_target_sum=TS)
    np.testing.assert_allclose(mom["bulk_lognorm"].jk, ref.jk, rtol=1e-10)


def test_inmem_driver_carries_counts_space_sumsq_under_bulk_lognorm():
    """The keying bug: acc aliases bulk_lognorm to 'counts', so sumsq['bulk_lognorm'] does
    not exist. Asserting the VALUE (not just that it is finite) also pins that the alias maps
    to the raw-count accumulator rather than to whatever key happens to be present."""
    a = _adata(seed=8)
    _, _, ref = pseudobulk_bulk_lognorm_with_moments(a, "pert", bulk_target_sum=TS)
    _, mom = inmem_pseudobulk(a, pert_col="pert", norms=["bulk_lognorm"], target_sum=None,
                              with_moments=True, bulk_target_sum=TS)
    np.testing.assert_allclose(mom["bulk_lognorm"].sumsq, ref.sumsq, rtol=1e-9)
    np.testing.assert_allclose(mom["bulk_lognorm"].counts, ref.counts)


def test_inmem_driver_leaves_jk_none_on_lognorm():
    _, mom = inmem_pseudobulk(_adata(seed=5), pert_col="pert", norms=["lognorm"],
                              target_sum=1e4, with_moments=True, bulk_target_sum=TS)
    assert mom["lognorm"].jk is None


def test_inmem_driver_serves_both_normalizations_at_once():
    """The MIXED case is the NORMAL case after PR1: a profile carries lognorm WITH moments
    beside bulk_lognorm WITH moments, and only the latter gets a jk."""
    _, mom = inmem_pseudobulk(_adata(seed=6), pert_col="pert",
                              norms=["lognorm", "bulk_lognorm"], target_sum=1e4,
                              with_moments=True, bulk_target_sum=TS)
    assert mom["lognorm"].jk is None
    assert mom["bulk_lognorm"].jk is not None


def _one_cell_holds_everything():
    """The r_i == 0 fixture from Task 1, as an AnnData: one group of two cells, one of which
    holds every count. C = 47.71708990205765 exactly (Task 1 measured it against the oracle)."""
    obs = pd.DataFrame({"pert": ["A", "A", "B", "B", "B"]})
    X = np.array([[4.0, 0.0], [0.0, 0.0], [3.0, 1.0], [2.0, 2.0], [1.0, 4.0]])
    return ad.AnnData(X=sp.csr_matrix(X), obs=obs), 47.71708990205765


def test_inmem_driver_reproduces_the_r_zero_edge_case_exactly():
    """Finding (g): the `_loo_bulk` r_i == 0 policy is a SHARED policy, so every driver must
    reproduce it, not just the resident kernel. Rev 2 asserted only finite-and-positive on the
    resident path and ran the fixture nowhere else -- so a streaming kernel that returned NaN
    (rev 1's did) or 0 (rev 1's other one did) passed. Task 8 pushes this same fixture through
    the shard, cell and GPU routes; this is the in-memory leg."""
    a, want = _one_cell_holds_everything()
    _, mom = inmem_pseudobulk(a, pert_col="pert", norms=["bulk_lognorm"], target_sum=None,
                              with_moments=True, bulk_target_sum=TS)
    i = int(np.flatnonzero(np.asarray(mom["bulk_lognorm"].perts, dtype=str) == "A")[0])
    assert mom["bulk_lognorm"].jk[i] == pytest.approx(want, rel=1e-12)


@pytest.mark.parametrize("n,g", [(40, 12), (1000, 12), (1000, 200)])
def test_the_streaming_kernel_agrees_with_the_resident_one_BIT_FOR_BIT(n, g):
    """The three kernels each reduce their own s1/s2, so nothing else pins that they share the
    `_loo_bulk` SHIFT rather than merely the edge policy.

    This test exists because Task 4 shipped `_streaming_jackknife` and
    `GroupedMeanAccumulator.jackknife` calling an unshifted `_loo_bulk` while the resident
    kernel was shifted, and every cross-kernel assertion in this plan passed anyway: they run
    at n=10-20 with rtol=1e-8, where the divergence is 1e-11 or smaller. MEASURED divergence of
    that shipped state against the resident kernel:

        n=20   1.347e-11      n=200   7.293e-11      n=1000  3.759e-08
        n=40   4.277e-11                             n=4000  1.539e-06

    So the assertion is BIT EQUALITY (`assert_array_equal`, not allclose) at n large enough to
    separate them, which is only achievable because the shift now lives inside `_loo_bulk`
    itself. A tolerance here -- any tolerance -- re-admits the divergence.

    What this test does and does NOT pin, VERIFIED BY MUTATION:
      * Reproduce Task 4's shipped state (shift in `jackknife_correction` only) and ALL
        THREE parametrizations go red, while every other test in this module stays green.
      * Delete the shift from `_loo_bulk` outright and this test stays GREEN -- both
        kernels lose it together, so they still agree with each other. That mutation is
        caught by `test_the_chunk_size_does_not_change_the_answer` instead.
    The two are complementary: the chunk test pins that the shift EXISTS, this one pins
    that it is applied UNIFORMLY across kernels. Neither alone covers both."""
    rng = np.random.default_rng(5)
    Y = rng.poisson(3.0, size=(n, g)).astype(np.float64)
    Y[:, 0] += 200
    X = sp.csr_matrix(Y)
    codes = np.zeros(n, dtype=np.intp)

    resident = jackknife_correction(X, codes, 1, TS)
    streaming = _streaming_jackknife(lambda: iter([(X, codes)]),
                                     Y.sum(axis=0)[None, :],
                                     np.array([float(n)]), TS)
    np.testing.assert_array_equal(streaming, resident)


def test_the_streaming_kernel_returns_zero_for_an_all_zero_group():
    """`jackknife_correction` skips an empty group with an explicit `tot <= 0` continue; the
    streaming kernels have no such guard, so folding the shift into `_loo_bulk` put a 0/0 in
    their path. The `xp.where(S > 0, S, 1.0)` guard is what keeps this 0.0 instead of NaN."""
    Y = np.zeros((3, 4))
    X = sp.csr_matrix(Y)
    codes = np.zeros(3, dtype=np.intp)
    got = _streaming_jackknife(lambda: iter([(X, codes)]),
                               Y.sum(axis=0)[None, :], np.array([3.0]), TS)
    assert got[0] == 0.0
    assert jackknife_correction(X, codes, 1, TS)[0] == 0.0




# --- #271: the group-sum reduction is WIDE, so the bulk and its own jackknife agree -----------
#
# HISTORY, because the shape of this block is the history. #271 was implemented on the chunk-2
# branch, REVERTED (`ee0e6c9`) when the fix was measured to move the three official val bundles'
# FRACTIONAL baseline leg, and characterized instead: three tests were left asserting that the
# defect was PRESENT. It is now fixed, so those three are inverted below -- each keeps the
# measurement it was written to hold -- and the implementation's own tests are restored with them.
# `_grouped_sums_NARROW` is a pinned copy of the pre-fix reduction, so "what this change moved"
# is asserted against the behaviour that actually shipped rather than inferred.


def _grouped_sums_NARROW(X, order, bounds, n_groups):
    """The PRE-FIX reduction, pinned verbatim: reduce in the input dtype, cast only the result.

    Kept so the tests that record what #271 moved compare against the code that shipped, not
    against an fp64-cast INPUT (which conflates "the reduction widened" with "the caller stored
    a different matrix"). If this ever needs updating, that is a sign a test is using it to
    check current behaviour, which is not what it is for."""
    out = np.zeros((n_groups, X.shape[1]), dtype=np.float64)
    for g in range(n_groups):
        rows = order[bounds[g]:bounds[g + 1]]
        if rows.size == 0:
            continue
        out[g] = np.asarray(X[rows].sum(axis=0), dtype=np.float64).ravel()
    return out


# --- the three characterization tests, INVERTED (#271 is fixed) --------------------------------

def test_grouped_sums_reduces_WIDE_and_the_two_halves_AGREE():
    """Was `test_grouped_sums_reduces_in_the_INPUT_dtype_and_that_is_the_open_defect`.

    `_grouped_sums` used to reduce in whatever dtype `X` carried, while
    `moments.jackknife_correction` casts `.data` to fp64 first -- so the `bulk_lognorm` bulk and
    the correction subtracted from it could come from different `P_p`: one metric, two group sums.
    Both halves now reduce wide, so for ONE input the answer no longer depends on the dtype the
    caller happened to store it in, and the two halves see one `P_p`. ⚠️ That is the removal of a
    systematic narrow-vs-wide mismatch, NOT redistribution invariance and not order independence:
    floating addition is not associative, so two different cell-level layouts with the same
    mathematical total can still reduce differently."""
    from cell_eval2.prep import _group_row_index, _grouped_sums

    dense = np.array([[16777216.0, 1.0], [1.0, 1.0]], dtype=np.float32)
    X32 = sp.csr_matrix(dense)
    X64 = sp.csr_matrix(dense.astype(np.float64))
    order, bounds = _group_row_index(np.array(["p", "p"]))[1:]

    # the bulk's own reduction: dtype-INVARIANT now, and it keeps the +1 at 2**24
    got32 = _grouped_sums(X32, order, bounds, 1)
    got64 = _grouped_sums(X64, order, bounds, 1)
    assert got32[0, 0] == 16777217.0
    assert got64[0, 0] == 16777217.0
    assert got32[0, 0] == got64[0, 0]
    # and the pre-fix behaviour this replaced, so the fixture is known to discriminate
    assert _grouped_sums_NARROW(X32, order, bounds, 1)[0, 0] == 16777216.0, (
        "the fixture no longer crosses the boundary, so this test proves nothing")

    # the correction subtracted from that bulk: dtype-invariant all along, because
    # `jackknife_correction` casts `.data` to fp64 before reducing. The ASYMMETRY was the defect.
    codes = np.zeros(2, dtype=np.int64)
    jk32 = jackknife_correction(X32, codes, 1, 50_000.0)
    jk64 = jackknife_correction(X64, codes, 1, 50_000.0)
    assert jk32[0] == jk64[0], (
        "the jackknife always reduced wide; it was _grouped_sums that did not")


def test_a_bulk_delta_is_meaningless_without_its_bulk_target_sum():
    """A guard on the DOCUMENTATION, because the number went stale once already.

    #271's issue text and several docstrings quoted the 2**24 fixture's bulk move as 6.35e-09.
    That is the same fixture at `bulk_target_sum = 1e6` -- the default #268 RETIRED on 2026-08-11
    in favour of 5e4, where the same fixture reads 3.53e-10, eighteen times smaller. `log1p`'s
    linear-to-log knee moves with TS, so the divergence a group-sum gap produces scales with it.

    Both values are pinned here so a future TS change cannot leave a stale figure standing in
    prose that nothing executes."""
    from cell_eval2.prep import _group_row_index, _grouped_sums, bulk_lognorm_means

    X = sp.csr_matrix(np.array([[16777216.0, 1.0], [1.0, 1.0]], dtype=np.float32))
    order, bounds = _group_row_index(np.array(["p", "p"]))[1:]
    wide = _grouped_sums(X, order, bounds, 1)
    narrow = _grouped_sums_NARROW(X, order, bounds, 1)
    assert (wide[0, 0], narrow[0, 0]) == (16777217.0, 16777216.0)

    for ts, want in ((50_000.0, 3.531662e-10), (1_000_000.0, 6.348612e-09)):
        got = np.abs(bulk_lognorm_means(wide, ts) - bulk_lognorm_means(narrow, ts)).max()
        assert got == pytest.approx(want, rel=1e-4), f"TS={ts:g}: {got:.6e} != {want:.6e}"


def test_the_resident_and_streaming_drivers_AGREE_over_the_fp32_boundary():
    """Was `test_the_resident_and_streaming_drivers_DISAGREE_over_the_fp32_boundary`.

    `streaming_bulk._streaming_pseudobulk_cpu` accumulates into an fp64 array from fp64-cast data
    and `gpu.bulk.GroupedMeanAccumulator` does the same, so the RESIDENT path was the only
    reduction left in the input dtype -- the same submission COULD score differently depending on
    which driver ran it. MEASURED before the fix: the bulks the metrics read diverged by 3.53e-10.
    They are now bit-identical, and the pre-fix gap is asserted here as the counterfactual so the
    number that motivated the change stays in the tree.

    Bit equality is the point: 3.53e-10 is inside any sane rtol, which is why the original
    divergence went unseen."""
    import anndata as ad_mod
    import pandas as pd_mod

    from cell_eval2.prep import _group_row_index, bulk_lognorm_means, pseudobulk_bulk_lognorm
    from cell_eval2.streaming_bulk import _streaming_pseudobulk_cpu

    X = np.array([[16777216.0, 1.0], [1.0, 1.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32)
    labels = np.array(["A", "A", "ctrl", "ctrl"])
    adata = ad_mod.AnnData(X=sp.csr_matrix(X), obs=pd_mod.DataFrame({"target": labels}),
                           var=pd_mod.DataFrame(index=["g0", "g1"]))
    _, resident = pseudobulk_bulk_lognorm(adata, "target", bulk_target_sum=50_000.0)
    out = _streaming_pseudobulk_cpu([(adata.X, labels)], np.unique(labels), 2, adata.n_vars,
                                    ["bulk_lognorm"], None, bulk_target_sum=50_000.0)
    np.testing.assert_array_equal(resident, out["bulk_lognorm"][1])

    # the counterfactual: the resident bulk the pre-fix reduction produced, against the streaming
    # one it disagreed with. 3.53e-10 is the measurement recorded on #271.
    perts, order, bounds = _group_row_index(labels)
    pre_fix = bulk_lognorm_means(_grouped_sums_NARROW(adata.X, order, bounds, perts.size), 50_000.0)
    gap = np.abs(pre_fix - out["bulk_lognorm"][1]).max()
    assert gap == pytest.approx(3.53e-10, rel=0.2), (
        f"the pre-fix divergence changed size: {gap:.3g} -- re-measure before editing this")


def test_the_fix_MOVES_the_stored_FRACTIONAL_baseline_arm():
    """Was `test_the_reduction_divergence_reaches_the_FRACTIONAL_baseline_arm` -- THE measurement
    that stopped this fix shipping the first time, kept because the fix is what costs it.

    The exactness argument -- non-negative INTEGER counts below 2**24 reduce identically in fp32
    and fp64 -- holds for the reference panels and for split-half anchors. It does NOT hold for the
    BASELINE arm: `real_bundle._baseline_leg` sets `allow_fractional_counts=True` because a
    baseline is a mean, and `baseline.py` emits `.astype(np.float32)`. Fractional fp32 CAN round from
    the first addition, so the divergence appears far below the boundary.

    MEASURED on all three official contexts' `context_mean` arms AS STORED -- the archives the
    official bundles were built from, 138,400 cells x 18,533 genes, 301 groups, 95.3-95.6% of
    stored values fractional, largest per-gene group sum 2.9e6 (5.7-7.6x INSIDE 2**24): group sums
    move by up to 0.265 and bulks by up to 5.7e-06 (A 0.2029/5.58e-06, B 0.2646/5.65e-06,
    C 0.0684/5.73e-06). Rebuilding the arm through the builders below instead of reading the stored
    archive reads smaller (0.0809/0.1246/0.0884 and 3.9-4.5e-06). So an artifact stored from a
    coarse-float arm whose group sums MOVED has to be regenerated, and the CONSERVATIVE regenerate
    set is "a fractional arm at any depth, or an integer-valued one past the dtype's exactness
    boundary" -- conservative because an exactly representable fraction reduces identically. An
    integral arm below that boundary is bit-identical either way, which the final assertion here
    pins.

    Built here through the SAME builders the bundle uses -- `_profile_from_adata` then
    `_prediction_from_adata(emit="dispersed")` -- rather than through a hand-rolled fractional
    matrix, so the two decisive facts are asserted rather than asserted-in-prose: the arm those
    builders emit is fractional float32, and this reduction is on its path. Fixture scale
    reproduces the real magnitude (0.09 / 4.8e-06 vs 0.08-0.12 / 3.9-4.5e-06)."""
    import anndata as ad_mod
    import pandas as pd_mod

    from cell_eval2.baseline import _prediction_from_adata, _profile_from_adata
    from cell_eval2.prep import _group_row_index, _grouped_sums, bulk_lognorm_means

    rng = np.random.default_rng(0)
    n_per, n_genes = 600, 40
    labels = np.array(["non-targeting"] * n_per + ["g0"] * n_per + ["g1"] * n_per)
    counts = rng.poisson(30.0, size=(labels.size, n_genes)).astype(np.float32)
    ref = ad_mod.AnnData(
        X=sp.csr_matrix(counts),
        obs=pd_mod.DataFrame({"target_gene": labels},
                             index=[f"c{i}" for i in range(labels.size)]),
        var=pd_mod.DataFrame(index=[f"g{j}" for j in range(n_genes)]))
    profile = _profile_from_adata(ref, pert_col="target_gene", control="non-targeting",
                                 exclude_target_gene=True)
    arm = _prediction_from_adata(ref, profile, pert_col="target_gene", control="non-targeting",
                                emit="dispersed", seed=0)

    X = arm.X.tocsr() if sp.issparse(arm.X) else sp.csr_matrix(arm.X)
    assert X.dtype == np.float32, "the stored arm is float32 -- if that changes, re-measure"
    assert not np.allclose(X.data, np.rint(X.data)), (
        "the arm must be FRACTIONAL; if a baseline stops being a mean this test is measuring "
        "something else")

    perts, order, bounds = _group_row_index(arm.obs["target_gene"].to_numpy().astype(str))
    now = _grouped_sums(X, order, bounds, perts.size)               # shipped: reduces wide
    pre_fix = _grouped_sums_NARROW(X, order, bounds, perts.size)    # what the stored arm used
    assert now.max() < 2 ** 24 / 100, (
        f"the point is that this moves FAR below the boundary; max group sum {now.max():.0f}")
    assert np.abs(now - pre_fix).max() > 1e-3
    assert np.abs(bulk_lognorm_means(now, 50_000.0)
                  - bulk_lognorm_means(pre_fix, 50_000.0)).max() > 1e-7

    # and the same arm made INTEGRAL does not move -- the contrast that makes the fractional
    # regime the operative one rather than the 2**24 boundary
    whole = sp.csr_matrix((np.rint(X.data), X.indices, X.indptr), shape=X.shape).astype(np.float32)
    np.testing.assert_array_equal(_grouped_sums(whole, order, bounds, perts.size),
                                  _grouped_sums_NARROW(whole, order, bounds, perts.size))


# --- the implementation's own tests, restored with the fix -------------------------------------

def test_grouped_sums_reduces_WIDE_across_every_LAYOUT():
    """Every layout that can actually reach the loop reduces wide, not just CSR."""
    from cell_eval2.prep import _group_row_index, _grouped_sums

    X32 = np.array([[16777216.0, 1.0], [1.0, 1.0]], dtype=np.float32)
    labels = np.array(["p", "p"])
    perts, order, bounds = _group_row_index(labels)
    # `coo_array` is the one that exercises the no-`indices`/`indptr` fallback -- an earlier
    # version of this test passed `coo_matrix(...).tocsr()`, which is CSR twice over and proved
    # nothing about that branch (codex review). `coo_matrix` itself is NOT subscriptable on scipy
    # 1.18, so `X[rows]` raises before the fallback is reached; that is pre-existing and shared
    # with `_grouped_means`.
    layouts = [X32, np.asmatrix(X32), sp.csr_matrix(X32), sp.csc_matrix(X32), sp.csr_array(X32),
               sp.csc_array(X32)]
    # `coo_array` row-indexing landed in scipy 1.17; the package floor is scipy>=1.11, where
    # `coo_array[rows]` raises like `coo_matrix` does. Gate on the CAPABILITY rather than on a
    # version so this neither fails on an older scipy nor silently stops covering the fallback
    # on a newer one (codex review round 2).
    try:
        sp.coo_array(X32)[np.asarray([0])]
    except TypeError:
        pass
    else:
        layouts.append(sp.coo_array(X32))   # no indices/indptr -> exercises the astype fallback
    for X in layouts:
        got = _grouped_sums(X, order, bounds, perts.size)
        assert got[0, 0] == 16777217.0, f"{type(X).__name__} reduced in the input dtype: {got}"
    # `coo_matrix` is not subscriptable on any supported scipy, so `X[rows]` raises before the
    # fallback is reached. Pre-existing and shared with `_grouped_means`.
    with pytest.raises(TypeError):
        _grouped_sums(sp.coo_matrix(X32), order, bounds, perts.size)


def test_grouped_sums_fp32_and_fp64_agree_on_the_boundary_fixture():
    """The other half of the same statement, at the fixture the rest of this file uses."""
    from cell_eval2.prep import _group_row_index, _grouped_sums

    a = _adata_over_the_fp32_boundary()
    labels = a.obs["pert"].to_numpy().astype(str)
    perts, order, bounds = _group_row_index(labels)
    s32 = _grouped_sums(a.X.astype(np.float32), order, bounds, perts.size)
    s64 = _grouped_sums(a.X.astype(np.float64), order, bounds, perts.size)
    np.testing.assert_array_equal(s32, s64)


def test_the_bulk_and_its_jackknife_see_the_SAME_group_sums_over_the_boundary():
    """End to end on the fixture built for it: one metric, two halves, one `P_p`. The jackknife
    side is reduced the way `jackknife_correction` reduces it -- `.data` cast to fp64 first."""
    from cell_eval2.prep import _group_row_index, _grouped_sums

    a = _adata_over_the_fp32_boundary()          # fp32, per-gene group sum 16,777,217
    labels = a.obs["pert"].to_numpy().astype(str)
    perts, order, bounds = _group_row_index(labels)
    bulk_side = _grouped_sums(a.X, order, bounds, perts.size)
    Xd = sp.csr_matrix((a.X.data.astype(np.float64), a.X.indices, a.X.indptr), shape=a.X.shape)
    jk_side = _grouped_sums(Xd, order, bounds, perts.size)
    np.testing.assert_array_equal(bulk_side, jk_side)
    # the pre-fix state, so this fixture is known to have discriminated the two halves
    assert not np.array_equal(_grouped_sums_NARROW(a.X, order, bounds, perts.size), jk_side)


def test_the_wide_reduction_moves_NOTHING_below_the_boundary():
    """The guarantee that bounds what this change moved: for non-negative integer counts every
    partial sum is bounded by the total, so an fp32 reduction is EXACT while the group sum stays
    under 2**24. MEASURED on all three official contexts' REAL arms (138,400 cells x 18,533 genes
    each, integer counts): largest per-gene group sum 2.2-2.9e6, 5.7-7.6x inside the boundary, and
    the wide and narrow reductions are bit-identical over the whole matrix -- max|delta| exactly
    0.000000 on all three. An earlier 110,500-cell panel read 1,474,940, 11.4x inside."""
    rng = np.random.default_rng(3)
    from cell_eval2.prep import _group_row_index, _grouped_sums

    X = rng.poisson(40.0, size=(600, 50)).astype(np.float64)   # group sums ~ 1e4, far under 2**24
    labels = np.array(["A", "B", "C"] * 200)
    perts, order, bounds = _group_row_index(labels)
    assert _grouped_sums(X, order, bounds, perts.size).max() < 2 ** 24
    np.testing.assert_array_equal(
        _grouped_sums(sp.csr_matrix(X.astype(np.float32)), order, bounds, perts.size),
        _grouped_sums(sp.csr_matrix(X), order, bounds, perts.size),
    )
    # and against the pre-fix reduction on the same fp32 matrix: nothing moved here either
    np.testing.assert_array_equal(
        _grouped_sums(sp.csr_matrix(X.astype(np.float32)), order, bounds, perts.size),
        _grouped_sums_NARROW(sp.csr_matrix(X.astype(np.float32)), order, bounds, perts.size),
    )


def test_the_exactness_guarantee_is_INTEGER_only_and_fractional_input_DOES_move():
    """The honest limit on "#271 moves nothing at realistic depth".

    The guarantee is that every partial sum of non-negative INTEGERS is bounded by the total, so
    an fp32 reduction is exact below 2**24. Fractional values have no such property: they CAN round
    from the first addition, far below the boundary (not always -- an exactly representable fraction
    reduces exactly; what is gone is the guarantee). MEASURED at 2.7e-06 relative on a 4,000-cell
    group whose sums are ~6.2e3 -- 2,700x inside 2**24.

    ⚠️ `configs/vcc2026.yaml` pins `allow_fractional_counts: false` with `validate_input: true`,
    but that does NOT make the competition path integer-only end to end: the bundle's own baseline
    leg sets `allow_fractional_counts=True` because a baseline is a mean. That is the gap an
    earlier certification of this fix missed -- see
    `test_the_fix_MOVES_the_stored_FRACTIONAL_baseline_arm`."""
    from cell_eval2.prep import _group_row_index, _grouped_sums

    rng = np.random.default_rng(0)
    labels = np.array(["A"] * 4000)
    perts, order, bounds = _group_row_index(labels)

    frac = sp.csr_matrix((rng.random((4000, 30)) * 3.0).astype(np.float32))
    assert not np.allclose(frac.data, np.rint(frac.data)), "fixture must be fractional"
    new = _grouped_sums(frac, order, bounds, perts.size)
    old = _grouped_sums_NARROW(frac, order, bounds, perts.size)
    assert new.max() < 2 ** 24 / 1000, "fixture must sit FAR below the 2**24 boundary"
    assert np.abs(new - old).max() > 0.0, (
        "fractional fp32 input must be sensitive to the reduction dtype; if this is 0 the "
        "fixture stopped exercising the case the docstring's caveat is about")

    # the same shape, made integral: bit-identical, which is the contrast that makes the
    # integer guarantee meaningful rather than a hope
    whole = sp.csr_matrix(np.rint(frac.toarray() * 100).astype(np.float32))
    np.testing.assert_array_equal(_grouped_sums(whole, order, bounds, perts.size),
                                  _grouped_sums_NARROW(whole, order, bounds, perts.size))


def test_a_BACKED_sparse_dataset_still_works_and_reduces_wide(tmp_path):
    """The regression the widen branch nearly introduced.

    An anndata BACKED `X` is a `_CSRDataset`, which `scipy.sparse.issparse` reports FALSE -- but
    its `X[rows]` IS a csr_matrix. An earlier draft of the widen branch decided sparse-vs-dense from
    `X` and cast with `np.asanyarray`, which raised "setting an array element with a sequence" on
    exactly this input; main tolerated it by accident, summing before any cast.

    ⚠️ What this pins is that backed input is SUPPORTED and reduces wide -- NOT the placement of the
    `issparse` call. Once the generic branch became `sub.astype(np.float64)`, deciding from `X`
    stopped crashing (a csr_matrix has `.astype`), so hoisting the predicate is now only a wasted
    index copy per group and this test cannot see it -- verified by mutation (codex round 2).
    `_grouped_sums` documents that placement as frugality, not correctness.

    ⚠️ `_grouped_means` hoists AND uses `np.asarray`, so it still raises that ValueError for backed
    input -- measured on main. Not fixed here: that helper drives `pseudobulk`/`lognorm` and the v1
    parity gate pins its arithmetic."""
    import anndata as ad_mod
    import pandas as pd_mod

    from cell_eval2.prep import _group_row_index, _grouped_sums

    X = sp.csr_matrix(np.array([[16777216.0, 1.0], [1.0, 1.0]], dtype=np.float32))
    a = ad_mod.AnnData(X=X, obs=pd_mod.DataFrame({"p": ["p", "p"]}, index=["c0", "c1"]),
                       var=pd_mod.DataFrame(index=["g0", "g1"]))
    f = tmp_path / "backed.h5ad"
    a.write_h5ad(f)
    backed = ad_mod.read_h5ad(f, backed="r")
    assert not sp.issparse(backed.X), (
        "the fixture must be a backed dataset -- if anndata makes it report as sparse, this test "
        "no longer covers the hoisting bug")
    assert sp.issparse(backed.X[np.array([0, 1])]), "but its SLICE must be sparse"

    order, bounds = _group_row_index(np.array(["p", "p"]))[1:]
    got = _grouped_sums(backed.X, order, bounds, 1)
    assert got[0, 0] == 16777217.0, f"backed input did not reduce wide: {got}"


# --- exposure: the VALID submissions whose numbers this change moved (codex review) ------------

def test_a_COMPETITION_VALID_submission_can_cross_2_24():
    """The `vcc2026` preset caps per-CELL totals at 1e6, not GROUP totals, and there are 500 cells
    per perturbation -- so a per-gene group sum may legally reach 5e8, thirty times past 2**24.

    This fixture is deliberately built to PASS both gates rather than to be realistic: if a future
    change makes validation reject it, that closes the exposure and this test should be rewritten
    rather than deleted. The three official contexts measured here sit 5.7-7.6x inside the boundary
    (an earlier 110,500-cell panel measured 11.4x)."""
    import anndata as ad_mod

    from cell_eval2 import norm
    from cell_eval2.prep import _group_row_index, _grouped_sums, bulk_lognorm_means

    X = np.tile(np.array([[999999.0, 1.0]], dtype=np.float32), (17, 1))
    a = ad_mod.AnnData(sp.csr_matrix(X), obs=pd.DataFrame({"t": ["p"] * 17}),
                       var=pd.DataFrame(index=["g0", "g1"]))
    norm.validate_input_type(a, "counts", allow_fractional=False)   # must not raise
    norm.check_scale_limit(a, "counts", 1_000_000.0)                # must not raise

    perts, order, bounds = _group_row_index(np.array(["p"] * 17))
    new = _grouped_sums(a.X, order, bounds, 1)
    old = _grouped_sums_NARROW(a.X, order, bounds, 1)
    assert new[0, 0] > 2 ** 24, "fixture must cross the boundary to exercise the exposure"
    assert new[0, 0] != old[0, 0], (
        "a valid submission above 2**24 must be the case where the reduction dtype shows; if "
        "this is equal the fixture stopped exercising the documented exposure")
    delta = np.abs(bulk_lognorm_means(new, 50_000.0) - bulk_lognorm_means(old, 50_000.0)).max()
    assert delta > 0.0


def test_validate_input_type_now_REJECTS_the_near_integer_route():
    """The second route, and the reason the 2**24 boundary was never the operative limit for it:
    `norm._is_all_integer` used to compare with `np.allclose` at its DEFAULT rtol=1e-5, so
    1000.001 was accepted as counts with allow_fractional_counts=False -- and such a matrix
    diverges across the reduction dtype far BELOW the boundary, where the exactness argument
    offers no protection at all.

    This was pinned here as a documented property and explicitly NOT endorsed: "tightening it
    would change which submissions are ACCEPTED, which is a larger competition-facing decision
    than the reduction dtype". Alex took that decision (2026-08-18, before the 0.15.0 freeze):
    `norm._INT_ATOL` makes the tolerance ABSOLUTE (`rtol=0, atol=1e-6`), so the gate now refuses
    THIS matrix. Precisely that and no more: what closed is the VALUE-SCALED branch of the route --
    a deviation the old relative rule hid because the value was large. `vcc2026`'s published rule
    ("non-negative, integral ... rejected rather than scored") is still a PROXY here, not enforced
    literally: a deviation inside 1e-6 is accepted by design.

    So this test proves BOTH halves, and it needs both. The arithmetic exposure is unchanged and
    still real: reached here through `allow_fractional=True`, which the tiled baseline leg depends
    on -- one of three remaining ways in, the others being a sub-1e-6 deviation and the partitioned
    or direct-shard drivers, which never call `validate_input_type` at all (see
    `prep._grouped_sums`' exposure list). What changed is that THIS submission can no longer get
    there through the strict gate. Deleting the divergence half would leave the repo with no record
    of why an absolute tolerance was worth a behaviour change so close to the freeze."""
    import anndata as ad_mod

    from cell_eval2 import norm
    from cell_eval2.prep import _group_row_index, _grouped_sums, bulk_lognorm_means

    # TWO genes with DIFFERENT values. With one gene -- or with all genes equal -- the bulk is
    # `log1p(bulk_target_sum)` whatever the sum is, because `bulk_lognorm_means` divides by the
    # row total; the first cut of this test used one gene and so proved a group-sum difference
    # that could not reach a score (codex review round 2).
    Y = np.tile(np.array([[1000.001, 7.0]], dtype=np.float32), (4000, 1))
    b = ad_mod.AnnData(sp.csr_matrix(Y), obs=pd.DataFrame({"t": ["p"] * 4000}),
                       var=pd.DataFrame(index=["g0", "g1"]))

    # The gate, which is what closes the route. float32 stores 1000.001 as 1000.0009765625, i.e.
    # 9.8e-04 off the integer -- inside the old relative tolerance of 1.0e-02 at this magnitude
    # and three orders outside the absolute one.
    with pytest.raises(ValueError, match="fractional"):
        norm.validate_input_type(b, "counts", allow_fractional=False)
    norm.validate_input_type(b, "counts", allow_fractional=True)    # the documented bypass
    norm.check_scale_limit(b, "counts", 1_000_000.0)                # and inside the per-cell cap

    # The exposure the gate now closes. It is arithmetic, not policy, so it does not go away:
    # this is what a matrix admitted by the OLD gate did to `_grouped_sums`.
    perts, order, bounds = _group_row_index(np.array(["p"] * 4000))
    new = _grouped_sums(b.X, order, bounds, 1)
    old = _grouped_sums_NARROW(b.X, order, bounds, 1)
    assert new.max() < 2 ** 24, "the point is that this diverges BELOW the boundary"
    assert abs(new[0, 0] - old[0, 0]) > 1.0, (
        f"expected a whole-count-scale gap below 2**24; got {abs(new[0, 0] - old[0, 0])}")
    delta = np.abs(bulk_lognorm_means(new, 50_000.0) - bulk_lognorm_means(old, 50_000.0)).max()
    assert delta > 0.0, "a group-sum gap that cannot move the bulk is not a scoring exposure"


def test_a_NARROW_dtype_moves_far_below_fp32s_boundary():
    """The third route (codex review round 2). `2**24` is float32's integer-exactness limit, not a
    universal: it is `2**(nmant + 1)`, which is 2,048 for float16 and 2**53 for float64. Nothing in
    the pipeline rejects float16 counts -- `validate_input_type` checks sign and integrality, not
    width -- so a float16 matrix diverges four orders of magnitude below where fp32 would.

    Not a realistic submission format. Pinned because the exactness argument in `_grouped_sums`'
    docstring is dtype-conditional and was read as universal in its first draft."""
    import anndata as ad_mod

    from cell_eval2 import norm
    from cell_eval2.prep import _group_row_index, _grouped_sums, bulk_lognorm_means

    for dtype, limit in ((np.float16, 2 ** 11), (np.float32, 2 ** 24), (np.float64, 2 ** 53)):
        assert 2 ** (np.finfo(dtype).nmant + 1) == limit

    X = np.array([[2048.0, 1.0], [1.0, 1.0]], dtype=np.float16)
    a = ad_mod.AnnData(X, obs=pd.DataFrame({"t": ["p", "p"]}),
                       var=pd.DataFrame(index=["g0", "g1"]))
    norm.validate_input_type(a, "counts", allow_fractional=False)   # neither gate looks at width
    norm.check_scale_limit(a, "counts", 1_000_000.0)

    perts, order, bounds = _group_row_index(np.array(["p", "p"]))
    new = _grouped_sums(X, order, bounds, 1)
    old = _grouped_sums_NARROW(X, order, bounds, 1)
    assert new.max() < 2 ** 24, "the whole point is that this is far below FLOAT32's boundary"
    assert new[0, 0] == 2049.0 and old[0, 0] == 2048.0
    delta = np.abs(bulk_lognorm_means(new, 50_000.0) - bulk_lognorm_means(old, 50_000.0)).max()
    assert delta > 1e-4, f"expected a ~4.8e-04 bulk move; got {delta:.3g}"


# --- the four inversions the widen guard exists to avoid ---------------------------------------

def test_INTEGER_input_keeps_its_own_exact_reduction():
    """#271 widens FLOATING dtypes only, and that asymmetry is the fix, not an oversight.

    numpy reduces integers in a wide integer accumulator, which is exact up to that accumulator's
    RANGE (past it, it wraps rather than saturates -- see the `_grouped_sums` docstring); float64
    stops representing consecutive integers above 2**53. So casting an integer matrix to fp64
    BEFORE the reduction is a regression, and the first cut of #271 -- which upcast
    unconditionally -- had exactly that inversion (codex review round 3).

    MEASURED on int64 `[[2**53 + 1, 7], [1, 0]]`: reducing then casting gives the exact 2**53 + 2;
    casting then reducing gives 2**53, and the bulk moves by 1.8e-15. Unreachable under `vcc2026`
    (500 cells x a 1e6 per-cell cap is 5e8, far under 2**53), which is why it is fixed rather than
    filed."""
    from cell_eval2.prep import _group_row_index, _grouped_sums, bulk_lognorm_means

    order, bounds = _group_row_index(np.array(["p", "p"]))[1:]
    X = np.array([[2 ** 53 + 1, 7], [1, 0]], dtype=np.int64)
    got = _grouped_sums(X, order, bounds, 1)
    assert got[0, 0] == float(2 ** 53 + 2), (
        f"int64 lost its exact reduction: {got[0, 0]!r} != {2**53 + 2}")
    # uint64 separately and ABOVE 2**63, so it exercises the unsigned accumulator rather than a
    # value int64 could also have held (codex review round 4).
    Xu = np.array([[2 ** 63 + 1, 7], [1024, 0]], dtype=np.uint64)
    got_u = _grouped_sums(Xu, order, bounds, 1)
    assert got_u[0, 0] == float(2 ** 63 + 1025), (
        f"uint64 lost its exact reduction: {got_u[0, 0]!r}")

    # the counterfactual, so the assertion above is anchored to a real difference
    upcast_first = np.asarray(X, dtype=np.float64)[order].sum(axis=0)
    assert upcast_first[0] == float(2 ** 53), "the fixture must still discriminate the two orders"
    delta = np.abs(bulk_lognorm_means(_grouped_sums(X, order, bounds, 1), 50_000.0)
                   - bulk_lognorm_means(upcast_first.reshape(1, -1), 50_000.0)).max()
    assert delta > 0.0, "and the difference must be able to reach the bulk"


def test_small_integer_input_is_unaffected_either_way():
    """The ordinary case: well under 2**53, both orders agree, so the asymmetry above costs
    nothing on any realistic integer matrix."""
    from cell_eval2.prep import _group_row_index, _grouped_sums

    rng = np.random.default_rng(5)
    X = rng.integers(0, 5000, size=(300, 40)).astype(np.int64)
    labels = np.array(["A", "B", "C"] * 100)
    perts, order, bounds = _group_row_index(labels)
    np.testing.assert_array_equal(
        _grouped_sums(X, order, bounds, perts.size),
        _grouped_sums(X.astype(np.float64), order, bounds, perts.size),
    )


def test_LONGDOUBLE_is_not_downcast_and_big_endian_float64_is_not_copied():
    """The other half of the widen guard (codex review round 4).

    `dtype != np.float64` also catches longdouble and DOWNCASTS it -- the same inversion as the
    integer one, in the other direction. MEASURED with u = 2**-53 on longdouble
    `[[1 + u, 6u], [u, 0]]`: reducing natively then casting gives nextafter(1, inf), casting to
    fp64 first gives 1.0. It also needlessly copies a big-endian float64, whose eps IS float64's.

    Comparing eps says what is meant -- "coarser than float64" -- and gets all four right."""
    from cell_eval2.prep import _group_row_index, _grouped_sums

    eps64 = float(np.finfo(np.float64).eps)

    def widen(dt):
        """The production predicate, evaluated the way `_grouped_sums` evaluates it."""
        return (np.issubdtype(np.dtype(dt), np.floating)
                and float(np.finfo(dt).eps) > eps64)

    assert widen(np.float16) and widen(np.float32)
    assert not widen(np.float64) and not widen(">f8") and not widen("<f8")
    assert not widen(np.int64) and not widen(np.uint64) and not widen(bool)
    assert not widen(np.complex64), "non-floating must short-circuit before finfo"

    order, bounds = _group_row_index(np.array(["p", "p"]))[1:]
    # ⚠️ Branch rather than `pytest.skip`. A runtime skip here would be a GATE, and
    # `internal:tests/gated_modules.toml` would then have to classify "is longdouble wider than float64"
    # as an environment condition for every tracked environment -- for a test that in fact has
    # something to assert on BOTH kinds of platform. So assert the guard's verdict either way and
    # the precision claim only where longdouble can carry it.
    assert not widen(np.longdouble), "longdouble must never be widened, whatever its width here"
    u = np.longdouble(2.0) ** -53
    X = np.array([[np.longdouble(1) + u, 6 * u], [u, np.longdouble(0)]], dtype=np.longdouble)
    got = _grouped_sums(X, order, bounds, 1)
    if float(np.finfo(np.longdouble).eps) >= eps64:
        # longdouble IS float64 on this platform (e.g. aarch64/Windows): `1 + u` already rounded
        # to 1.0 on the way into the array, so there is no extra precision left to preserve and
        # the two orders necessarily agree. The guard assertion above is the whole claim here.
        assert got[0, 0] == 1.0
        return
    assert got[0, 0] == np.nextafter(1.0, np.inf), (
        f"longdouble was downcast before the reduction: {got[0, 0]!r}")
    # the counterfactual the previous guard produced
    assert np.asarray(X, dtype=np.float64)[order].sum(axis=0)[0] == 1.0



def test_a_MASKED_matrix_STILL_splits_the_two_halves():
    """A RESIDUAL, characterized rather than fixed -- and the honest limit on "the two halves see
    one P_p".

    `_grouped_sums` preserves a mask (the test below pins that). `jackknife_correction` does
    `csr_matrix(X)` on dense input, which STRIPS it and sums the hidden values back in. So for a
    MASKED matrix the bulk and its own correction still come from different group sums -- exactly
    the asymmetry #271 is about, for a reason that is not the reduction dtype.

    Asserted by DRIVING `jackknife_correction`, not by reproducing its cast: the masked input and
    the same matrix with the hidden value written in must give the SAME correction, and both must
    differ from the mask-honouring (hidden -> 0) matrix. Reproducing one source line instead would
    stay green if the jackknife ever started honouring masks, and would then falsely report the
    residual as surviving (codex round 2). MEASURED both ways: with `np.ma.filled(X, 0)` spliced
    into `jackknife_correction`, the masked reading moves to 30.19090768371814 and this test fails,
    which is the discrimination proof.

    MEASURED on masked float32 `[[10, 1], [--, 1]]` hiding 900: the bulk's group sum is [10, 2]
    while the jackknife reads [910, 2] and returns 4.815789351767213 -- bit-identical to the
    hidden-filled matrix, against 30.19090768371814 for the zero-filled one. PRE-EXISTING:
    `origin/main`'s narrow reduction also went through `MaskedArray.sum`, so it split the same way.
    Out of scope for a dtype fix -- closing it means deciding what a masked X MEANS to the
    comparator (honour it in both halves, or reject it at validation), a semantics call.

    Pinned so the invariant in `_grouped_sums`' docstring cannot be read as universal."""
    from cell_eval2.prep import _group_row_index, _grouped_sums

    base = np.array([[10.0, 1.0], [900.0, 1.0]], dtype=np.float32)
    M = np.ma.masked_array(base.copy(), mask=[[False, False], [True, False]])
    order, bounds = _group_row_index(np.array(["p", "p"]))[1:]

    # the bulk half HONOURS the mask
    np.testing.assert_array_equal(_grouped_sums(M, order, bounds, 1), np.array([[10.0, 2.0]]))

    # the correction half does NOT -- it equals the hidden-value matrix, not the masked one
    codes = np.zeros(2, dtype=np.intp)
    jk_masked = jackknife_correction(M, codes, 1, 50_000.0)[0]
    jk_hidden = jackknife_correction(base, codes, 1, 50_000.0)[0]
    jk_honoured = jackknife_correction(
        np.array([[10.0, 1.0], [0.0, 1.0]], dtype=np.float32), codes, 1, 50_000.0)[0]
    assert jk_masked == jk_hidden, (
        "if these now differ the jackknife has started honouring masks -- delete this test and "
        "drop the caveat from _grouped_sums' docstring, docs/metrics.md and the CHANGELOG entry")
    assert jk_masked != jk_honoured, (
        "the fixture must make the mask matter to the correction, or this proves nothing")
    assert jk_masked == pytest.approx(4.815789351767213)
    assert jk_honoured == pytest.approx(30.19090768371814)


def test_a_MASKED_coarse_float_keeps_its_mask():
    """#271's third inversion (codex review round 5).

    `np.asarray(sub, dtype=np.float64)` STRIPS an ndarray subclass, and AnnData accepts a masked
    X. MEASURED on masked float32 `[[10, 1], [--, 1]]` whose hidden value is 900: `asarray` sums
    the masked cell back in and gives [910, 2] where the masked reduction gives [10, 2] -- a bulk
    move of 4.32, from a mask the caller set deliberately. `asanyarray` preserves the subclass.

    float64 masked input never reaches that branch (`widen` is False for it), which is why only a
    COARSE dtype was affected -- and why this test parametrizes over both."""
    from cell_eval2.prep import _group_row_index, _grouped_sums

    order, bounds = _group_row_index(np.array(["p", "p"]))[1:]
    for dtype, coarse in ((np.float32, True), (np.float64, False)):
        M = np.ma.masked_array(np.array([[10.0, 1.0], [900.0, 1.0]], dtype=dtype),
                               mask=[[False, False], [True, False]])
        got = _grouped_sums(M, order, bounds, 1)
        assert got[0, 0] == 10.0, (
            f"{np.dtype(dtype).name} (widened={coarse}) lost its mask: {got[0]} -- the masked "
            "900 was summed back in")
        assert got[0, 1] == 2.0


def test_BIG_ENDIAN_float64_is_not_widened_IN_PRODUCTION():
    """An earlier version of this assertion re-evaluated a LOCAL copy of the predicate, so a
    production-only regression on `>f8` left it green -- no big-endian array ever reached
    `_grouped_sums` (codex review round 6). This observes the real conversion instead.

    `>f8` is the case `dtype != np.float64` got wrong: same eps as float64, so widening it buys
    nothing and costs a full copy of every group.

    The instrument is an ndarray SUBCLASS that records its own `astype` calls, rather than a
    monkeypatched module attribute. `prep.np` IS the numpy module, so patching an attribute on it
    is global and numpy's own internals call the same functions -- a bare call count there measures
    numpy, not this function (the first cut of this test read 2 calls that had nothing to do with
    `_grouped_sums`). A subclass sees only what is done to THIS array.

    ⚠️ It records `.astype` calls on the array itself, which is how the production cast is currently
    spelled -- NOT every possible conversion. A future branch using `np.asarray`/`np.asanyarray`
    would leave this green; the masked-mask test is what covers that direction."""
    from cell_eval2.prep import _group_row_index, _grouped_sums

    class _RecordingArray(np.ndarray):
        """Records the dtypes `_grouped_sums` casts this array (or its slices) to."""

        def astype(self, dtype, *a, **kw):
            type(self).seen.append(np.dtype(dtype))
            return super().astype(dtype, *a, **kw)

    def _make(values, dtype):
        arr = np.array(values, dtype=dtype).view(_RecordingArray)
        type(arr).seen = []
        return arr

    order, bounds = _group_row_index(np.array(["p", "p"]))[1:]

    X_be = _make([[3.0, 4.0], [5.0, 6.0]], np.dtype(">f8"))
    got_be = _grouped_sums(X_be, order, bounds, 1)
    assert type(X_be).seen == [], (
        f"big-endian float64 was converted -- it is already float64 and needs no copy: "
        f"{type(X_be).seen}")
    np.testing.assert_array_equal(np.asarray(got_be), np.array([[8.0, 10.0]]))

    # the contrast: a genuinely coarse dtype IS converted, so the assertion above is known to
    # observe this function rather than to be vacuous
    X32 = _make([[3.0, 4.0], [5.0, 6.0]], np.float32)
    _grouped_sums(X32, order, bounds, 1)
    assert np.dtype(np.float64) in type(X32).seen, (
        f"the coarse dtype was never widened, so the assertion above proves nothing: "
        f"{type(X32).seen}")
