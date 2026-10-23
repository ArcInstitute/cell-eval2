import contextlib
import logging

import numpy as np
import pytest
import scipy.sparse as sp

from _cupy_like import INVALID_VALUE, CupyLike, DeviceArray, FakeCudaError
from cell_eval2 import gpu
from cell_eval2.gpu import resolve_device
from cell_eval2.gpu.bulk import GroupedMeanAccumulator

_HAS_GPU = resolve_device("auto") == "cuda"
_cuda = pytest.param("cuda", marks=pytest.mark.skipif(not _HAS_GPU, reason="no usable CUDA GPU"))


def _reference(X, grp, P, target_sum, norms):
    # numpy sum-then-divide; lognorm = mean(log1p(CPM)), NO expm1.
    Xd = X.toarray().astype(np.float64)
    n, g = Xd.shape
    out = {}
    for norm in norms:
        ref = np.zeros((P, g))
        cnt = np.zeros(P)
        for r in range(n):
            row = Xd[r].copy()
            if norm != "counts":
                lib = row.sum() or 1.0
                row = row * (target_sum / lib)
            if norm == "lognorm":
                row = np.log1p(row)
            ref[grp[r]] += row
            cnt[grp[r]] += 1
        ref /= np.maximum(cnt, 1)[:, None]
        out[norm] = ref
    return out


@pytest.mark.parametrize("device", ["cpu", _cuda])
def test_grouped_mean_matches_reference(device):
    rng = np.random.default_rng(4)
    n, g, P = 200, 12, 4
    X = sp.csr_matrix(rng.poisson(0.6, size=(n, g)).astype(np.float32))
    grp = rng.integers(0, P, size=n)
    norms = ["counts", "normalized", "lognorm"]
    acc = GroupedMeanAccumulator(P, g, normalizations=norms, target_sum=1e6, device=device)
    acc.update(X, grp)
    out = acc.finalize()
    ref = _reference(X, grp, P, 1e6, norms)
    for norm in norms:
        assert out[norm][1].dtype == np.float32
        assert np.allclose(out[norm][1], ref[norm], rtol=1e-4, atol=1e-6)


@pytest.mark.parametrize("device", ["cpu", _cuda])
def test_accumulates_across_shards(device):
    # two update() calls must equal one over the concatenation (shard streaming).
    rng = np.random.default_rng(7)
    g, P = 10, 5
    X1 = sp.csr_matrix(rng.poisson(0.5, size=(80, g)).astype(np.float32))
    grp1 = rng.integers(0, P, size=80)
    X2 = sp.csr_matrix(rng.poisson(0.5, size=(120, g)).astype(np.float32))
    grp2 = rng.integers(0, P, size=120)
    norms = ["counts", "normalized", "lognorm"]
    acc = GroupedMeanAccumulator(P, g, normalizations=norms, target_sum=1e6, device=device)
    acc.update(X1, grp1)
    acc.update(X2, grp2)
    out = acc.finalize()
    Xall = sp.vstack([X1, X2]).tocsr()
    grpall = np.concatenate([grp1, grp2])
    ref = _reference(Xall, grpall, P, 1e6, norms)
    for norm in norms:
        assert np.allclose(out[norm][1], ref[norm], rtol=1e-4, atol=1e-6)


def test_lognorm_has_no_expm1():
    # lognorm = mean(log1p(CPM)); an erroneous trailing expm1 would inflate values by
    # orders of magnitude. A single-cell group's lognorm must equal log1p(CPM).
    g, P = 4, 1
    X = sp.csr_matrix(np.array([[0.0, 10.0, 0.0, 5.0]], dtype=np.float32))
    acc = GroupedMeanAccumulator(P, g, normalizations=["lognorm"], target_sum=1e6, device="cpu")
    acc.update(X, np.array([0]))
    out = acc.finalize()
    lib = 15.0
    expected = np.log1p(np.array([0.0, 10.0, 0.0, 5.0]) * (1e6 / lib))
    assert np.allclose(out["lognorm"][1][0], expected, rtol=1e-5)
    # raw CPM would be ~6.6e5/3.3e5; log-space stays well under 20.
    assert out["lognorm"][1].max() < 20.0


def test_empty_group_is_zero_not_nan():
    # a group with no cells -> mean 0 (denominator floored to 1), never NaN.
    g, P = 3, 4
    X = sp.csr_matrix(np.array([[1.0, 2.0, 3.0]], dtype=np.float32))
    acc = GroupedMeanAccumulator(P, g, normalizations=["counts"], target_sum=1e6, device="cpu")
    acc.update(X, np.array([1]))  # only group 1 populated
    out = acc.finalize()
    assert np.isfinite(out["counts"][1]).all()
    assert np.allclose(out["counts"][1][0], 0.0)
    assert np.allclose(out["counts"][1][2], 0.0)


def test_invalid_normalization_raises():
    with pytest.raises(ValueError, match="normalization"):
        GroupedMeanAccumulator(2, 3, normalizations=["bogus"], target_sum=1e6, device="cpu")


# --- max_row_total: reuse the accumulator's per-cell totals for the scale-limit gate ---
@pytest.mark.parametrize("device", ["cpu", _cuda])
def test_max_row_total_matches_row_sums(device):
    rng = np.random.default_rng(11)
    n, g, P = 150, 10, 4
    X = sp.csr_matrix(rng.poisson(0.7, size=(n, g)).astype(np.float32))
    grp = rng.integers(0, P, size=n)
    acc = GroupedMeanAccumulator(P, g, normalizations=["lognorm"], target_sum=1e6, device=device)
    acc.update(X, grp)
    expected = float(np.asarray(X.sum(axis=1)).ravel().max())
    assert acc.max_row_total == pytest.approx(expected)


@pytest.mark.parametrize("device", ["cpu", _cuda])
def test_max_row_total_none_for_counts_only(device):
    X = sp.csr_matrix(np.array([[1, 2], [3, 4]], dtype=np.float32))
    acc = GroupedMeanAccumulator(2, 2, normalizations=["counts"], target_sum=1e6, device=device)
    acc.update(X, np.array([0, 1]))
    assert acc.max_row_total is None


@pytest.mark.parametrize("device", ["cpu", _cuda])
def test_max_row_total_accumulates_across_shards_and_empty_rows(device):
    # empty + trailing-empty rows contribute total 0 (never the max); the global max is the
    # busiest row across both shards.
    X1 = sp.csr_matrix(np.array([[0, 0, 0], [5, 1, 0]], dtype=np.float32))   # row totals 0, 6
    X2 = sp.csr_matrix(np.array([[2, 9, 0], [0, 0, 0]], dtype=np.float32))   # row totals 11, 0
    acc = GroupedMeanAccumulator(2, 3, normalizations=["normalized"], target_sum=1e6, device=device)
    acc.update(X1, np.array([0, 1]))
    acc.update(X2, np.array([0, 1]))
    assert acc.max_row_total == pytest.approx(11.0)


@pytest.mark.parametrize("device", ["cpu", _cuda])
def test_max_row_total_propagates_nan_like_np_max(device):
    # Byte-identical to the current float(np.max(_row_totals)) path: a NaN row total must
    # propagate to the accumulated max regardless of block order. Distinguishes np.maximum
    # (propagates) from Python max (order-dependent: max(9.0, nan) -> 9.0 would lose the NaN).
    # NaN is only reachable via allow_fractional_counts + a NaN-bearing submission.
    X1 = sp.csr_matrix(np.array([[np.nan, 1.0], [2.0, 3.0]], dtype=np.float64))  # row totals nan, 5
    X2 = sp.csr_matrix(np.array([[4.0, 5.0]], dtype=np.float64))                 # finite block AFTER nan
    acc = GroupedMeanAccumulator(2, 2, normalizations=["normalized"], target_sum=1e6, device=device)
    acc.update(X1, np.array([0, 1]))
    acc.update(X2, np.array([0]))
    assert np.isnan(acc.max_row_total)


# --- Lever 2A: per-cell library size via a device row-isolated segment-sum ---
@pytest.mark.parametrize("device", ["cpu", _cuda])
def test_libs_bincount_matches_scipy_rowsum(device):
    # The per-cell library size (recovered through the normalized mean of a one-cell group --
    # that group mean IS that cell's CPM) must equal scipy Xr.sum(axis=1) byte-for-byte on
    # integer counts. Locks the contract that the device segment-sum must not regress.
    rng = np.random.default_rng(19)
    n, g = 64, 20
    X = sp.csr_matrix(rng.poisson(0.8, size=(n, g)).astype(np.float32))
    grp = np.arange(n)  # one cell per group -> mean == that cell's CPM
    ts = 1e6
    acc = GroupedMeanAccumulator(n, g, normalizations=["normalized"], target_sum=ts, device=device)
    acc.update(X, grp)
    out = acc.finalize()["normalized"][1]  # [n, g] fp32

    Xd = X.toarray().astype(np.float64)
    libs = np.asarray(X.sum(axis=1, dtype=np.float64)).ravel()
    libs_nz = np.where(libs == 0, 1.0, libs)
    expected = (Xd * (ts / libs_nz)[:, None]).astype(np.float32)
    np.testing.assert_array_equal(out, expected)  # byte-identical (integer counts -> exact)
    assert acc.max_row_total == float(libs.max())  # exact for integer counts


@pytest.mark.parametrize("device", ["cpu", _cuda])
def test_libs_segment_sum_isolates_nan_rows(device):
    # A NaN in one row must not leak into other rows' library sizes -- this is what rules out a
    # cumsum-diff prefix sum (which would make every later row's lib nan-nan=NaN). A row-isolated
    # segment-sum (bincount) confines the NaN to row 0; row 1 (lib 5.0) is untouched.
    X = sp.csr_matrix(np.array([[np.nan, 1.0], [2.0, 3.0]], dtype=np.float64))  # rows: nan, 5
    acc = GroupedMeanAccumulator(2, 2, normalizations=["normalized"], target_sum=1e6, device=device)
    acc.update(X, np.array([0, 1]))
    out = acc.finalize()["normalized"][1]
    assert np.isnan(out[0]).all()          # row 0 poisoned by its own NaN
    assert np.isfinite(out[1]).all()       # row 1 untouched by row 0's NaN
    np.testing.assert_array_equal(out[1], (np.array([2.0, 3.0]) * (1e6 / 5.0)).astype(np.float32))
    assert np.isnan(acc.max_row_total)     # NaN still propagates to the running max


# --- #133: byte-bounded pinned H2D sub-chunking (cupy 2 GiB single-alloc ceiling) ---
def _rand_csr(seed, n_cells=400, n_genes=50, density=0.3):
    rng = np.random.default_rng(seed)
    X = (rng.random((n_cells, n_genes)) < density) * rng.poisson(3, (n_cells, n_genes))
    return sp.csr_matrix(X.astype(np.float32))


def _finalize_all(acc):
    fin = acc.finalize()
    return {n: fin[n][1] for n in acc.norms}


def test_update_row_subchunking_is_bit_identical_cpu():
    # update() with a tiny byte budget (forced sub-chunking) must equal a huge budget,
    # bit-for-bit: the grouped-mean accumulation is row-additive and contiguous row
    # sub-chunks preserve the row-major add order (numpy add.at is in-order).
    from cell_eval2.gpu import bulk
    X = _rand_csr(1)
    groups = np.random.default_rng(2).integers(0, 4, size=X.shape[0]).astype(np.intp)
    norms = ["counts", "normalized", "lognorm"]

    big = GroupedMeanAccumulator(4, X.shape[1], normalizations=norms, target_sum=1e4, device="cpu")
    big.update(X, groups)
    want = _finalize_all(big)

    saved = bulk._PINNED_H2D_BUDGET_BYTES
    try:
        bulk._PINNED_H2D_BUDGET_BYTES = 64  # tiny -> many few-row sub-chunks
        small = GroupedMeanAccumulator(4, X.shape[1], normalizations=norms, target_sum=1e4, device="cpu")
        small.update(X, groups)
        got = _finalize_all(small)
    finally:
        bulk._PINNED_H2D_BUDGET_BYTES = saved

    for n in norms:
        np.testing.assert_array_equal(got[n], want[n])          # bit-identical
    assert small.max_row_total == big.max_row_total              # running max unaffected


def test_row_chunks_respect_byte_budget():
    # Every emitted chunk's nnz*max(itemsize) is within the budget; chunks tile all rows
    # contiguously; a lone over-budget row is emitted alone (can't be split).
    from cell_eval2.gpu import bulk
    X = _rand_csr(5, n_cells=500, n_genes=40, density=0.5)
    acc = GroupedMeanAccumulator(1, X.shape[1], normalizations=["counts"], target_sum=1e4, device="cpu")
    saved = bulk._PINNED_H2D_BUDGET_BYTES
    try:
        bulk._PINNED_H2D_BUDGET_BYTES = 2000
        itemsize = max(X.data.itemsize, X.indices.itemsize)
        chunks = list(acc._row_chunks(X))
        assert chunks[0][0] == 0 and chunks[-1][1] == X.shape[0]
        prev = 0
        for start, stop in chunks:
            assert start == prev and stop > start           # contiguous + progressing
            prev = stop
            nnz = int(X.indptr[stop] - X.indptr[start])
            assert nnz * itemsize <= bulk._PINNED_H2D_BUDGET_BYTES or (stop - start) == 1
    finally:
        bulk._PINNED_H2D_BUDGET_BYTES = saved


def test_row_chunks_empty_block():
    # a zero-row block yields no chunks (update() becomes a no-op, matching prior behavior).
    acc = GroupedMeanAccumulator(1, 4, normalizations=["counts"], target_sum=1e4, device="cpu")
    empty = sp.csr_matrix((0, 4), dtype=np.float32)
    assert list(acc._row_chunks(empty)) == []


def test_update_single_large_group_subchunks_like_control_pool():
    # The primary real-world #133 trigger is the control pool: ONE huge group passed as a single
    # whole-group block (streaming's transfer). Sub-chunking that block by bytes must stay
    # bit-identical and count every cell -- mirror it with one group split into many sub-chunks.
    from cell_eval2.gpu import bulk
    X = _rand_csr(9, n_cells=300, n_genes=40)
    groups = np.zeros(X.shape[0], dtype=np.intp)          # all one group (the "control pool")
    norms = ["counts", "normalized", "lognorm"]

    big = GroupedMeanAccumulator(1, X.shape[1], normalizations=norms, target_sum=1e4, device="cpu")
    big.update(X, groups)
    want = _finalize_all(big)

    saved = bulk._PINNED_H2D_BUDGET_BYTES
    try:
        bulk._PINNED_H2D_BUDGET_BYTES = 128                # force many sub-chunks
        acc = GroupedMeanAccumulator(1, X.shape[1], normalizations=norms, target_sum=1e4, device="cpu")
        assert len(list(acc._row_chunks(X))) > 1           # chunking actually engaged
        acc.update(X, groups)
        got = _finalize_all(acc)
    finally:
        bulk._PINNED_H2D_BUDGET_BYTES = saved

    for n in norms:
        np.testing.assert_array_equal(got[n], want[n])     # bit-identical across many sub-chunks
    assert acc._counts[0] == X.shape[0]                    # every cell counted


# --- #162: cupy bincount RAISES on an empty index array where numpy returns zeros ---
def _host(a):
    return np.asarray(a.get() if hasattr(a, "get") else a)


@pytest.mark.parametrize("device", ["cpu", _cuda])
def test_bincount_shim_empty_returns_numpy_zeros(device):
    # Direct cover of the shim. On an EMPTY index array numpy returns zeros(minlength) but cupy
    # raises `zero-size array to reduction operation CUPY_CUB_MAX which has no identity` -- it
    # takes idx.max() to size the output BEFORE applying minlength, so `minlength=` rescues
    # neither call form (measured on cupy 14.1.1). Both live forms are covered: unweighted
    # (the per-cell counts) and weighted (the per-cell library sizes).
    from cell_eval2.gpu import xp_for
    from cell_eval2.gpu.bulk import _bincount
    xp = xp_for(device)
    empty_idx = xp.zeros(0, dtype=xp.intp)
    unweighted = _bincount(xp, empty_idx, minlength=5)
    weighted = _bincount(xp, empty_idx, minlength=7, weights=xp.zeros(0, dtype=xp.float64))
    np.testing.assert_array_equal(_host(unweighted), np.zeros(5))
    np.testing.assert_array_equal(_host(weighted), np.zeros(7))
    assert _host(unweighted).dtype == np.dtype(np.intp)  # numpy's own unweighted dtype
    assert _host(weighted).dtype == np.float64  # the dtype every NON-empty weighted call gives


@pytest.mark.parametrize("device", ["cpu", _cuda])
def test_bincount_shim_leaves_the_nonempty_path_alone(device):
    # The shim must be a pure pass-through whenever the index array is non-empty -- it is on the
    # hot path (one weighted bincount over every nonzero of every block), so it must not change
    # a single value, in either call form.
    from cell_eval2.gpu import xp_for
    from cell_eval2.gpu.bulk import _bincount
    xp = xp_for(device)
    idx = xp.asarray(np.array([0, 2, 2, 3], dtype=np.intp))
    w = xp.asarray(np.array([1.5, 2.0, 3.0, 4.25], dtype=np.float64))
    for kwargs in ({}, {"weights": w}):
        got = _bincount(xp, idx, minlength=6, **kwargs)
        ref = xp.bincount(idx, minlength=6, **kwargs)
        np.testing.assert_array_equal(_host(got), _host(ref))
        assert _host(got).dtype == _host(ref).dtype


@pytest.mark.parametrize("device", ["cpu", _cuda])
def test_all_zero_matrix_scores_instead_of_raising(device):
    # The real-world #162 trigger: 1 of 339 submitted 2025 VCC predictions was an entirely empty
    # matrix -- well-formed, with cells and groups, but nnz == 0. `row_of_nnz` is then empty at
    # the weighted site and the cupy run ABORTED mid-scoring with a CUB internal error, while the
    # identical input scored fine on CPU -- so on CPU this test passes with or without the fix, and
    # only the [cuda] parametrization discriminates. (The synthetic-backend tests at the bottom of
    # this module are what pin the guard in CPU-only CI; see #183.)
    # An all-zero prediction must score: every mean 0.0, every cell still counted, and a
    # well-defined 0.0 max row total. Also locks the CPU semantics this fix has to match.
    n, g, P = 6, 4, 3
    X = sp.csr_matrix((n, g), dtype=np.float32)
    assert X.nnz == 0 and X.shape == (n, g)
    grp = np.array([0, 0, 1, 1, 1, 2], dtype=np.intp)
    norms = ["counts", "normalized", "lognorm", "bulk_lognorm"]
    acc = GroupedMeanAccumulator(P, g, normalizations=norms, target_sum=1e6, device=device)
    acc.update(X, grp)
    out = acc.finalize()
    for norm in norms:
        assert np.isfinite(out[norm][1]).all(), norm      # not NaN from a 0/0 or a floored lib
        np.testing.assert_array_equal(out[norm][1], np.zeros((P, g), dtype=np.float32))
    np.testing.assert_array_equal(_host(acc._counts), np.array([2.0, 3.0, 1.0]))
    assert acc.max_row_total == 0.0                        # max of an all-zero matrix, not None


@pytest.mark.parametrize("device", ["cpu", _cuda])
def test_accumulate_block_zero_cells_changes_nothing(device):
    # The sibling site (`_bincount(xp, group, ...)`) needs a zero-CELL block, which `_row_chunks`
    # never emits ("a single row always proceeds"), so it is reachable only by calling
    # `_accumulate_block` directly -- guarded anyway (#162), because cupy raises identically there
    # and `_accumulate_block` has no zero-cell early return of its own. A zero-cell block must be
    # an exact no-op on the accumulators, not a raise.
    g, P = 4, 3
    acc = GroupedMeanAccumulator(P, g, normalizations=["counts", "normalized"],
                                 target_sum=1e6, device=device)
    acc.update(sp.csr_matrix(np.array([[1, 0, 2, 0]], dtype=np.float32)), np.array([1]))
    before = _finalize_all(acc)
    acc._accumulate_block(sp.csr_matrix((0, g), dtype=np.float32), np.zeros(0, dtype=np.intp))
    after = _finalize_all(acc)
    for norm in before:
        np.testing.assert_array_equal(after[norm], before[norm])
    np.testing.assert_array_equal(_host(acc._counts), np.array([0.0, 1.0, 0.0]))


@pytest.mark.parametrize("device", ["cpu", _cuda])
def test_zero_cell_block_alone_leaves_max_row_total_unset(device):
    # Same site as above but on a FRESH accumulator, so nothing pre-seeds `max_row_total`: the
    # test above starts from a real row totalling 3.0 and would report that same 3.0 whether or not
    # the empty block wrongly contributed. A zero-cell block computes a length-0 `libs`, so `if
    # libs.size`
    # is False and the running max must stay None -- there is no row whose total it could be.
    acc = GroupedMeanAccumulator(3, 4, normalizations=["counts", "normalized"],
                                 target_sum=1e6, device=device)
    assert acc.max_row_total is None
    acc._accumulate_block(sp.csr_matrix((0, 4), dtype=np.float32), np.zeros(0, dtype=np.intp))
    assert acc.max_row_total is None
    np.testing.assert_array_equal(_host(acc._counts), np.zeros(3))


# --- #183: the two call sites, pinned in CPU-only CI ---
# The behaviour tests above discriminate only on CUDA; on CPU they pass with or without the fix,
# because numpy never had the bug. So reverting either `_bincount(...)` back to `xp.bincount(...)`
# is invisible to CI, which is exactly the coverage gap #183 is about. CPU CI still cannot run the
# real cupy failure -- but it can run the BEHAVIOUR, by making numpy raise on an empty index array
# exactly as cupy does. That pins both guards on every push. These are CPU-only by design: patching
# `np` says nothing about cupy, and the [cuda] cases above already cover the real backend.
@pytest.fixture
def cupy_like_bincount(monkeypatch):
    """Make `np.bincount` raise on an empty index array, as cupy 14.1.1 does (measured, #162)."""
    real = np.bincount

    def fake(x, weights=None, minlength=0):
        if np.asarray(x).size == 0:
            raise ValueError("zero-size array to reduction operation CUPY_CUB_MAX "
                             "which has no identity")
        return real(x, weights=weights, minlength=minlength)

    monkeypatch.setattr(np, "bincount", fake)


def test_cupy_like_bincount_fixture_bites(cupy_like_bincount):
    # The two tests below are worthless if the patch does not reach the module under test, and
    # what makes it reach is that `xp_for("cpu")` returns the numpy MODULE itself -- so go through
    # `xp`, not `np`, and assert that identity: were `xp_for` ever to return a wrapper or a shim,
    # patching `np` would silently stop pinning anything and these tests would pass regardless.
    from cell_eval2.gpu import xp_for
    xp = xp_for("cpu")
    assert xp is np
    with pytest.raises(ValueError, match="CUPY_CUB_MAX"):
        xp.bincount(np.zeros(0, dtype=np.intp), minlength=3)
    np.testing.assert_array_equal(xp.bincount(np.array([0, 2], dtype=np.intp), minlength=4),
                                  np.array([1, 0, 1, 0]))


def test_weighted_site_never_bincounts_an_empty_index(cupy_like_bincount):
    # The reachable site: nnz == 0 with cells present makes `row_of_nnz` empty. Under the fix
    # `xp.bincount` is never reached with it, so this scores; before the fix it raises here on CPU
    # too, which is the whole point -- CI can now see the GPU-only crash.
    acc = GroupedMeanAccumulator(3, 4, normalizations=["counts", "normalized", "lognorm"],
                                 target_sum=1e6, device="cpu")
    acc.update(sp.csr_matrix((6, 4), dtype=np.float32), np.array([0, 0, 1, 1, 1, 2]))
    out = acc.finalize()
    for norm in ("counts", "normalized", "lognorm"):
        np.testing.assert_array_equal(out[norm][1], np.zeros((3, 4), dtype=np.float32))
    assert acc.max_row_total == 0.0


def test_unweighted_site_never_bincounts_an_empty_index(cupy_like_bincount):
    # The sibling site: a zero-CELL block makes `group` empty. Unreachable through `update` (see
    # above), so it is called directly -- but the guard is what keeps it from raising, and without
    # this test nothing in CI would notice if it were removed.
    acc = GroupedMeanAccumulator(3, 4, normalizations=["counts"], target_sum=1e6, device="cpu")
    acc._accumulate_block(sp.csr_matrix((0, 4), dtype=np.float32), np.zeros(0, dtype=np.intp))
    np.testing.assert_array_equal(_host(acc._counts), np.zeros(3))


# --- #404: pinned H2D staging fails on a fragmented host; the copy falls back to pageable ---
# Linux 6.11 pins N pages with one physically contiguous N*8-byte kernel allocation (mm/gup.c,
# 53ba78de064b; removed again in 6.12 by 94efde1d1539). On a fragmented host that fails even
# with plenty free, CUDA reports cudaErrorInvalidValue, and cupy re-raises anything but
# cudaErrorMemoryAllocation (core.pyx:3072-3074). CPU CI drives the fallback with `CupyLike`.
@pytest.fixture
def pin_failures(monkeypatch):
    """`gpu._to_device` treats `FakeCudaError(INVALID_VALUE)` as a pin failure; latch reset."""
    monkeypatch.setattr(gpu, "_pin_failure_type", lambda: (FakeCudaError, INVALID_VALUE))
    monkeypatch.setattr(gpu, "_pinned_h2d_failed_nbytes", None)


def _pin_warnings(caplog):
    return [r for r in caplog.records if r.name == "cell_eval2.gpu"]


def test_to_device_is_numpy_asarray_on_cpu():
    a = np.arange(6, dtype=np.float32)
    assert gpu._to_device(np, a) is a                        # CPU path: no copy, no try


@pytest.mark.parametrize("layout", ["C", "F", "strided", "strided_big_endian"])
def test_to_device_falls_back_to_pageable_when_pinning_fails(pin_failures, caplog, layout):
    base = np.arange(24, dtype=np.float32).reshape(4, 6)
    a = {"C": base, "F": np.asfortranarray(base), "strided": base[:, ::2],
         # neither C nor F, with F-like strides: cupy picks C from the input before its byte-order
         # cast, and casting first would yield an F array
         "strided_big_endian": np.arange(24, dtype=">f8").reshape(4, 6).T[::2]}[layout]
    xp = CupyLike()
    with caplog.at_level(logging.WARNING, logger="cell_eval2.gpu"):
        out = gpu._to_device(xp, a)        # FakeStream fails it if the source dies before the wait
    assert isinstance(out, DeviceArray)
    assert out.dtype == a.dtype.newbyteorder("<") and out.shape == a.shape   # little-endian, as cupy
    np.testing.assert_array_equal(out, a)
    # cupy.asarray's order='K' rule (core.pyx:2961-2965): F only for an F-not-C source
    assert out.flags.f_contiguous == (layout == "F")
    assert out.flags.c_contiguous == (layout != "F")
    assert xp.syncs == 1                                      # the copy finished before return
    assert gpu._pinned_h2d_failed_nbytes == a.nbytes
    assert len(_pin_warnings(caplog)) == 1
    assert f"{a.nbytes:,}" in caplog.text and "cell_eval2#404" in caplog.text


def test_to_device_latches_pageable_after_the_first_failure(pin_failures, caplog):
    xp = CupyLike()
    with caplog.at_level(logging.WARNING, logger="cell_eval2.gpu"):
        gpu._to_device(xp, np.zeros(8, dtype=np.float32))
        out = gpu._to_device(xp, np.arange(5, dtype=np.int32))
    np.testing.assert_array_equal(out, np.arange(5, dtype=np.int32))
    assert xp.asarray_calls == 1                 # the second copy never tried to pin
    assert len(_pin_warnings(caplog)) == 1       # warned once per process
    assert gpu._pinned_h2d_failed_nbytes == 32   # the size that failed, not the latest


def test_to_device_converts_a_list_as_cupy_does(pin_failures):
    out = gpu._to_device(CupyLike(), [3, 1, 2])                  # cupy.asarray takes lists too
    assert out.dtype == np.asarray([3, 1, 2]).dtype
    np.testing.assert_array_equal(out, [3, 1, 2])


@pytest.mark.parametrize("status", [2, 700])   # cudaErrorMemoryAllocation, cudaErrorIllegalAddress
def test_to_device_reraises_any_other_cuda_error(pin_failures, status):
    with pytest.raises(FakeCudaError):
        gpu._to_device(CupyLike(status=status), np.zeros(4, dtype=np.float32))
    assert gpu._pinned_h2d_failed_nbytes is None


def test_to_device_reraises_when_cupy_is_absent(monkeypatch):
    monkeypatch.setattr(gpu, "_pin_failure_type", lambda: None)
    monkeypatch.setattr(gpu, "_pinned_h2d_failed_nbytes", None)
    with pytest.raises(FakeCudaError):
        gpu._to_device(CupyLike(), np.zeros(4, dtype=np.float32))


def test_to_device_propagates_a_failing_pageable_copy(pin_failures):
    # An InvalidValue from the copy itself (a bad argument) recurs in the pageable copy. It must
    # surface, never be swallowed into a silently wrong array. One from a call the retry does not
    # repeat (cupy's pointer query, say) would be absorbed; `_to_device`'s docstring says so.
    with pytest.raises(FakeCudaError):
        gpu._to_device(CupyLike(broken_set=True), np.zeros(4, dtype=np.float32))


def _accumulator_outputs(X, groups, norms, *, xp=None):
    acc = GroupedMeanAccumulator(4, X.shape[1], normalizations=norms, target_sum=1e4,
                                 device="cpu", with_moments=True)
    if xp is not None:
        acc._xp = xp      # every host->device copy now meets a pin that fails
    acc.update(X, groups)
    acc.jackknife(lambda: iter([(X, groups)]), per_gene=True)   # reaches the :341 site
    fin = acc.finalize()
    return ({n: fin[n][1] for n in norms}, acc.moments(), acc.jackknife_by_norm(),
            acc.jackknife_gene_by_norm(), acc.max_row_total)


def test_pageable_fallback_is_bit_identical_to_the_pinned_path_cpu(pin_failures, monkeypatch):
    from cell_eval2.gpu import bulk
    monkeypatch.setattr(bulk, "_PINNED_H2D_BUDGET_BYTES", 256)   # many sub-chunks, many copies
    X = _rand_csr(11)
    groups = np.random.default_rng(12).integers(0, 4, size=X.shape[0]).astype(np.intp)
    norms = ["counts", "normalized", "lognorm", "bulk_lognorm"]

    want = _accumulator_outputs(X, groups, norms)
    xp = CupyLike()
    got = _accumulator_outputs(X, groups, norms, xp=xp)

    # One failed pin, then pageable. A site that bypassed _to_device would raise FakeCudaError.
    assert xp.asarray_calls == 1
    assert xp.syncs > 4
    for n in norms:
        np.testing.assert_array_equal(got[0][n], want[0][n])
        for g_arr, w_arr in zip(got[1][n], want[1][n]):
            np.testing.assert_array_equal(g_arr, w_arr)
    np.testing.assert_array_equal(got[2]["bulk_lognorm"], want[2]["bulk_lognorm"])
    np.testing.assert_array_equal(got[3]["bulk_lognorm"], want[3]["bulk_lognorm"])
    assert got[4] == want[4]


@contextlib.contextmanager
def _pinned_alloc_fails():
    """cupy's pinned allocator raises cudaErrorInvalidValue, as a fragmented 6.11 host does."""
    import cupy
    from cupy_backends.cuda.api import runtime

    calls = []

    def fail(size):
        calls.append(size)
        raise runtime.CUDARuntimeError(gpu._CUDA_ERROR_INVALID_VALUE)

    cupy.cuda.set_pinned_memory_allocator(fail)
    try:
        yield calls
    finally:
        # what `import cupy` installs (cupy/__init__.py:907); cupy has no getter to restore from
        cupy.cuda.set_pinned_memory_allocator(cupy.get_default_pinned_memory_pool().malloc)


@pytest.mark.parametrize("device", [_cuda])
def test_pinned_staging_failure_falls_back_on_cuda(device, monkeypatch, caplog):
    import cupy
    monkeypatch.setattr(gpu, "_pinned_h2d_failed_nbytes", None)
    X = _rand_csr(21, n_cells=2000, n_genes=300)
    groups = np.random.default_rng(22).integers(0, 4, size=X.shape[0]).astype(np.intp)
    norms = ["counts", "normalized", "lognorm"]

    def run():
        acc = GroupedMeanAccumulator(4, X.shape[1], normalizations=norms, target_sum=1e4,
                                     device=device)
        acc.update(X, groups)
        return acc.finalize(), acc.max_row_total

    want, want_max = run()                                    # the ordinary pinned path
    with _pinned_alloc_fails() as calls, \
            caplog.at_level(logging.WARNING, logger="cell_eval2.gpu"):
        got, got_max = run()
    assert len(calls) == 1                                    # one pin attempt, then pageable
    assert len(_pin_warnings(caplog)) == 1 and "cell_eval2#404" in caplog.text
    np.testing.assert_array_equal(got["counts"][1], want["counts"][1])   # integer sums: exact
    for n in ("normalized", "lognorm"):
        # cupy's add.at scatters with fp64 atomics, whose order is not fixed between runs
        np.testing.assert_allclose(got[n][1], want[n][1], rtol=1e-6, atol=0)
    assert got_max == want_max
    assert int(cupy.arange(10).sum()) == 45                   # the context still works


@pytest.mark.parametrize("device", [_cuda])
def test_cuda_invalid_value_from_cudahostalloc_is_not_sticky(device):
    # #404 depends on cudaErrorInvalidValue from cudaHostAlloc leaving the context usable. Drive
    # the REAL driver into that error (flags it does not define), check that `_to_device` would
    # classify it as a pin failure, then do pinned staging, kernels and a CUB reduction.
    import cupy
    from cupy_backends.cuda.api import runtime
    with pytest.raises(runtime.CUDARuntimeError) as err:
        runtime.hostAlloc(1 << 20, 0xFFFFFFFF)
    assert str(err.value).startswith("cudaErrorInvalidValue")   # the driver's own name for it
    err_type, status = gpu._pin_failure_type()
    assert isinstance(err.value, err_type) and err.value.status == status
    n = 1 << 20
    x = cupy.asarray(np.arange(n, dtype=np.int64))
    assert int(x.sum()) == n * (n - 1) // 2 and int(x.max()) == n - 1


def test_pinned_budget_is_a_power_of_two_at_most_256_mib():
    # #404: cupy's pool rounds each pin up to a power of two, and on Linux 6.11 a pin of B bytes
    # needs a B/512-byte page array. 256 MiB keeps that at 512 KiB (order 7), gpudge's slab size
    # (gpudge_arc#159). A budget that is not a power of two would be rounded past itself.
    from cell_eval2.gpu import bulk
    budget = bulk._PINNED_H2D_BUDGET_BYTES
    assert budget & (budget - 1) == 0
    assert budget <= 1 << 28
