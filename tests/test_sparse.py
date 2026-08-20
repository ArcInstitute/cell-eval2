"""Unit tests for cell_eval2._sparse (CSR construction for the row-store streaming path).

The load-bearing assertions here are the BIT-IDENTITY ones. This is a pure speed/memory change:
the DE path's input must not move by a single ulp, or it stops being one. See the three invariants
in _sparse.py.
"""
import numpy as np
import pytest
import scipy.sparse as sp

from cell_eval2 import _sparse as sps
from cell_eval2.rowstore import scaled_log1p


def _sparse_block(n=200, g=64, density=0.07, seed=0):
    """Tahoe-like sparse uint16 counts (real Tahoe is 5.9-8.7% nonzero)."""
    rng = np.random.default_rng(seed)
    x = (rng.random((n, g)) < density) * rng.integers(1, 50, size=(n, g))
    return np.ascontiguousarray(x, dtype=np.uint16)


def test_csr_index_dtype_boundary():
    # INVARIANT 2: int32 iff it fits, else int64 -- for BOTH indices and indptr.
    assert sps.csr_index_dtype(0) is np.int32
    assert sps.csr_index_dtype(2**31 - 1) is np.int32
    assert sps.csr_index_dtype(2**31) is np.int64
    assert sps.csr_index_dtype(2**31 + 1) is np.int64


def test_extract_nnz_index_dtypes_match():
    # A mismatched-dtype CSR is the PR #70 landmine: scipy's X.min() upcast int32 indices to int64
    # and blew up a ~288 GiB temporary. A CCL_2-scale batch sits at 79% of the int32 ceiling.
    raw = _sparse_block()
    vals, indices, indptr = sps.extract_nnz(raw)
    assert indices.dtype == indptr.dtype
    assert vals.dtype == raw.dtype          # values keep the source dtype -- the scan does no math


def test_extract_nnz_structure_matches_scipy():
    raw = _sparse_block()
    vals, indices, indptr = sps.extract_nnz(raw)
    ref = sp.csr_matrix(raw)
    ref.sort_indices()
    np.testing.assert_array_equal(indptr.astype(np.int64), ref.indptr.astype(np.int64))
    np.testing.assert_array_equal(indices.astype(np.int64), ref.indices.astype(np.int64))
    np.testing.assert_array_equal(vals, ref.data)


def test_row_library_f32_bit_identical_to_dense_sum():
    # INVARIANT 3: the pass is chunked by ROWS to bound the float32 temp (a whole-batch temp would
    # be ~18 GiB on a real batch). A reduction along axis=1 happens WITHIN a row, so partitioning
    # rows cannot reorder any row's summation -- bit-identical by construction.
    raw = _sparse_block(n=5000, g=97, density=0.2, seed=3)
    ref = np.ascontiguousarray(raw, dtype=np.float32).sum(axis=1)
    np.testing.assert_array_equal(sps.row_library_f32(raw), ref)


def test_scaled_log1p_csr_is_bit_identical_to_scaled_log1p():
    """THE headline contract: the DE path's input does not move at all."""
    raw = _sparse_block(n=400, g=128, density=0.09, seed=1)
    got = sps.scaled_log1p_csr(raw, 1e4)
    ref = scaled_log1p(raw, 1e4)
    assert sp.issparse(got) and got.format == "csr" and got.dtype == np.float32
    np.testing.assert_array_equal(np.asarray(got.todense()), ref)      # max abs delta == 0


def test_scaled_log1p_csr_all_zero_row():
    # all-zero row -> library clamped to 1 -> log1p(0) == 0, and it stays STRUCTURALLY zero.
    raw = np.array([[3, 0, 7], [0, 0, 0], [0, 5, 0]], dtype=np.uint16)
    got = sps.scaled_log1p_csr(raw, 1e4)
    np.testing.assert_array_equal(np.asarray(got.todense()), scaled_log1p(raw, 1e4))
    assert got.getrow(1).nnz == 0


def test_to_csr_f32_exact():
    # output_space="raw": a pure dtype cast, no arithmetic at all.
    raw = _sparse_block(n=120, g=32, density=0.15, seed=2)
    got = sps.to_csr_f32(raw)
    assert got.dtype == np.float32 and got.format == "csr"
    np.testing.assert_array_equal(
        np.asarray(got.todense()), np.ascontiguousarray(raw, dtype=np.float32))


@pytest.mark.skipif(not sps.HAS_NUMBA, reason="numba not installed (CI runs the scipy fallback)")
def test_numba_path_equals_scipy_fallback_exactly(monkeypatch):
    """INVARIANT 1: numba SCANS, numpy COMPUTES -- so the two paths agree exactly.

    Letting numba do the log1p instead was measured at 4.768e-07 of drift; keeping every float op
    in numpy is precisely why this assertion can be `array_equal` and not `allclose`.
    """
    raw = _sparse_block(n=300, g=64, density=0.08, seed=4)
    numba_out = sps.scaled_log1p_csr(raw, 1e4)
    monkeypatch.setattr(sps, "HAS_NUMBA", False)          # force the scipy fallback
    scipy_out = sps.scaled_log1p_csr(raw, 1e4)
    np.testing.assert_array_equal(scipy_out.data, numba_out.data)
    np.testing.assert_array_equal(scipy_out.indices.astype(np.int64),
                                  numba_out.indices.astype(np.int64))
    np.testing.assert_array_equal(scipy_out.indptr.astype(np.int64),
                                  numba_out.indptr.astype(np.int64))


def test_estimate_density(tmp_path):
    raw = _sparse_block(n=2000, g=50, density=0.10, seed=5)
    p = tmp_path / "x.dat"
    raw.tofile(p)
    d = sps.estimate_density(p, dtype="uint16", n_rows=2000, n_genes=50, sample=500)
    assert abs(d - (np.count_nonzero(raw) / raw.size)) < 0.03      # sampled -> approximate
    assert 0.0 <= d <= 1.0


def test_estimate_density_empty_artifact(tmp_path):
    # An empty artifact is a legitimate DATA condition, not a caller error -> 0.0, no ZeroDivision.
    p = tmp_path / "e.dat"
    np.zeros((0, 4), dtype=np.uint16).tofile(p)
    assert sps.estimate_density(p, dtype="uint16", n_rows=0, n_genes=4) == 0.0


def test_estimate_density_rejects_nonpositive_sample(tmp_path):
    """A non-positive sample is a CALLER error -- fail loudly (Gemini, PR #105).

    Unguarded, sample=0 raised ZeroDivisionError (empty block -> blk.size == 0) and sample<0 raised
    a cryptic numpy "negative dimensions" ValueError. Returning 0.0 instead would silently claim
    "fully sparse" on zero evidence, so we raise -- matching the guard-the-branch style the codebase
    uses elsewhere (de_compute._to_linear, compute_de's boundary validation).
    """
    p = tmp_path / "x.dat"
    _sparse_block(n=50, g=8).tofile(p)
    for bad in (0, -3):
        with pytest.raises(ValueError, match="sample must be positive"):
            sps.estimate_density(p, dtype="uint16", n_rows=50, n_genes=8, sample=bad)


def test_sparse_density_max_below_break_even():
    # CSR costs 8 B/nnz (float32 data + a 4-byte index) against dense's 4 B/elem, so the memory
    # break-even is 50%. The gate must sit below it or an unusually dense store would REGRESS.
    assert 0.0 < sps.SPARSE_DENSITY_MAX < 0.5


# --------------------------------------------------------------------------------------------------
# The FUSED row-library scan. row_library_f32 was measured at 993.2 s = 36.7% of a clean 2,703 s real-
# Tahoe wall -- the single biggest function in the scorer, bigger than all of gpudge -- while the numba
# scan that already visits every one of the same elements cost 64.6 s. These tests pin the contract that
# lets the library ride along inside that scan for free.
# --------------------------------------------------------------------------------------------------


def _big_library_block():
    """uint16 rows whose libraries blow past 2**24 -- where the fused sum's exactness proof dies.

    float32 holds every integer exactly up to 2**24 (24-bit mantissa). BELOW that bound, every partial
    sum of NON-NEGATIVE integers is itself a non-negative integer <= the row total, so it is exactly
    representable, NOTHING rounds, and the summation ORDER is unobservable -- numpy's float32 pairwise
    sum and an exact int64 sum land on the identical float32. AT/ABOVE the bound that argument collapses
    and the two orders really do disagree. Here max library = 79,915,528, ~4.8x over 2**24 = 16,777,216.

    /!\\ THE VALUES MUST VARY. A block of IDENTICAL values (e.g. np.full(..., 60_000)) is DEGENERATE:
    equal terms make pairwise and sequential accumulation round the SAME way, so they agree even far
    above the bound (measured: 0/8 rows differ) -- and a test built on one would happily "confirm" the
    guard while being structurally incapable of detecting its removal. Random values in
    [40_000, 65_535) diverge on 8/32 rows. Seeded, so this is deterministic, not a coin flip.
    """
    rng = np.random.default_rng(7)
    return rng.integers(40_000, 65_535, size=(32, 1500), dtype=np.uint64).astype(np.uint16)


@pytest.mark.skipif(not sps.HAS_NUMBA, reason="numba not installed (CI runs the scipy fallback)")
def test_fused_library_is_bit_identical_to_row_library_f32():
    """THE fused-path contract: one scan yields the SAME libraries the separate numpy pass computed."""
    raw = _sparse_block(n=1000, g=257, density=0.07, seed=11)
    _, _, _, lib = sps.extract_nnz_and_library(raw)
    assert lib is not None and lib.dtype == np.float32
    np.testing.assert_array_equal(lib, sps.row_library_f32(raw))       # max abs delta == 0


@pytest.mark.skipif(not sps.HAS_NUMBA, reason="numba not installed")
def test_extract_nnz_and_library_returns_the_same_csr_as_extract_nnz():
    """Fusing the sum must not perturb the CSR structure -- INVARIANT 2 still holds."""
    raw = _sparse_block(n=300, g=64, density=0.08, seed=12)
    vals, indices, indptr, _ = sps.extract_nnz_and_library(raw)
    e_vals, e_indices, e_indptr = sps.extract_nnz(raw)
    np.testing.assert_array_equal(vals, e_vals)
    np.testing.assert_array_equal(indices.astype(np.int64), e_indices.astype(np.int64))
    np.testing.assert_array_equal(indptr.astype(np.int64), e_indptr.astype(np.int64))
    assert indices.dtype == indptr.dtype                               # INVARIANT 2
    assert vals.dtype == raw.dtype                                     # the scan still does no math


@pytest.mark.skipif(not sps.HAS_NUMBA, reason="numba not installed")
def test_fused_library_declines_above_the_float32_exact_bound():
    """The 2**24 guard is LOAD-BEARING, not decorative -- so prove BOTH halves of it here.

    (a) an UNGUARDED fused sum would genuinely disagree with numpy's pairwise sum on this block, and
    (b) the shipped code therefore declines the fast path (returns None) so the caller falls back.
    Delete the guard and (a) is what starts silently corrupting libraries.
    """
    raw = _big_library_block()
    unguarded = raw.sum(axis=1, dtype=np.int64).astype(np.float32)     # what the kernel computes
    assert not np.array_equal(unguarded, sps.row_library_f32(raw))     # (a) they REALLY diverge
    _, _, _, lib = sps.extract_nnz_and_library(raw)
    assert lib is None                                                 # (b) so we decline it


@pytest.mark.skipif(not sps.HAS_NUMBA, reason="numba not installed")
def test_fused_library_declines_on_non_unsigned_dtype():
    """NON-NEGATIVITY is the crux: it is what makes every partial sum <= the row total.

    A signed store could have partial sums that exceed the total (cancellation), and a float store is
    not exact under reordering at all. Only unsigned integers get the fused path.
    """
    raw = _sparse_block(n=64, g=32, density=0.1, seed=13).astype(np.float32)
    _, _, _, lib = sps.extract_nnz_and_library(raw)
    assert lib is None
    assert sps._fusable_dtype(np.uint8) and sps._fusable_dtype(np.uint16)
    assert sps._fusable_dtype(np.uint32)
    assert not sps._fusable_dtype(np.float32)
    assert not sps._fusable_dtype(np.int32)


def test_fused_library_declines_on_uint64_because_int64_cannot_hold_it():
    """uint64 must NOT take the fused path: int64 cannot represent it losslessly (Gemini, PR #106).

    The kernel's exactness rests on an int64 accumulator, and int64 tops out at 2**63-1 while uint64
    reaches 2**64-1. The tempting patch -- cast each value with `np.int64(v)` -- makes it WORSE: a uint64
    above 2**63 wraps to a NEGATIVE int64, and a negative library then sails straight through the
    `< 2**24` guard and corrupts silently. Excluding the dtype instead makes INVARIANT 1's "int64 is
    exact" claim true BY CONSTRUCTION rather than by luck about the input's magnitude. It costs nothing:
    a real .dat is uint16 counts or float32 lognorm, never uint64.

    (Empirically numba does compile the uint64 kernel today and gets the right answer for small values.
    That is precisely the kind of accident that stops being true without warning.)
    """
    assert not sps._fusable_dtype(np.uint64)
    assert np.iinfo(np.uint32).max <= np.iinfo(np.int64).max      # the boundary we are gating on
    assert np.iinfo(np.uint64).max > np.iinfo(np.int64).max

    raw = _sparse_block(n=64, g=32, density=0.12, seed=15).astype(np.uint64)
    _, _, _, lib = sps.extract_nnz_and_library(raw)
    assert lib is None                                            # declined -> numpy fallback
    np.testing.assert_array_equal(                                # ...and still bit-identical
        np.asarray(sps.scaled_log1p_csr(raw, 1e4).todense()), scaled_log1p(raw, 1e4))


def test_scaled_log1p_csr_bit_identical_above_the_exact_bound():
    """The fallback must be INVISIBLE: the headline contract holds on BOTH sides of the 2**24 guard.

    Not skipped on numba: CI (no numba) covers the scipy path, GPU1142 covers the fused one.
    """
    raw = _big_library_block()
    np.testing.assert_array_equal(
        np.asarray(sps.scaled_log1p_csr(raw, 1e4).todense()), scaled_log1p(raw, 1e4))


def test_scaled_log1p_csr_bit_identical_on_float32_store():
    """output_space='lognorm' .dat stores are float32 -> the non-fusable branch, still bit-identical."""
    raw = _sparse_block(n=200, g=64, density=0.12, seed=14).astype(np.float32)
    np.testing.assert_array_equal(
        np.asarray(sps.scaled_log1p_csr(raw, 1e4).todense()), scaled_log1p(raw, 1e4))


# --------------------------------------------------------------------------------------------------
# Density sampling. The cost here is SEEKS, not bytes: 1,000 scattered rows of an 18,151-gene store is
# ~1,000 separate 36 KB reads, and on wekafs that measured 103.2 s across one 5-context run's 10 sources
# = 3.8% of wall -- for a read the old docstring called "negligible".
# --------------------------------------------------------------------------------------------------


def test_estimate_density_reads_slabs_not_scattered_rows(tmp_path, monkeypatch):
    """Pin the SEEK COUNT -- that is the property that regresses if someone "simplifies" the sampler
    back to fancy indexing, and it is invisible to every other test here (they all check the value)."""
    raw = _sparse_block(n=4000, g=16, density=0.10, seed=21)
    p = tmp_path / "x.dat"
    raw.tofile(p)

    reads = []

    class CountingMemmap(np.memmap):
        """A real np.memmap SUBCLASS, so every ndarray behaviour the code under test relies on
        (dtype, shape, __array__, `del mm`) still works. Swapping in a bare proxy object that only
        implements __getitem__ would test a different object than production uses."""

        def __getitem__(self, key):
            reads.append(key)
            return super().__getitem__(key)

    monkeypatch.setattr(np, "memmap", CountingMemmap)      # _sparse calls np.memmap(...)
    sps.estimate_density(p, dtype="uint16", n_rows=4000, n_genes=16, sample=1000, slab=50)

    assert len(reads) <= 1000 // 50 + 1                    # ~20 reads, NOT ~1000
    assert all(isinstance(k, slice) for k in reads)        # contiguous slabs, not index arrays


def test_estimate_density_slabs_are_spread_not_one_block(tmp_path):
    """A row store is perturbation-CONTIGUOUS, so one big contiguous read would sample a single
    perturbation and lie. Plant the nonzeros in the back half only: a single front-anchored block reports
    ~0.0 and a single back-anchored one ~1.0, while spread slabs must land near the true 0.5.

    slab=10 -> 100 slabs, so the sampling s.d. is ~0.05 and the +/-0.15 window is ~3 s.d. (and the rng is
    seeded, so it is deterministic). A wider slab would leave too few slabs and make the window marginal.
    """
    raw = np.zeros((4000, 16), dtype=np.uint16)
    raw[2000:] = 7                                         # true density = 0.5
    p = tmp_path / "half.dat"
    raw.tofile(p)
    d = sps.estimate_density(p, dtype="uint16", n_rows=4000, n_genes=16, sample=1000, slab=10)
    assert 0.35 < d < 0.65


def test_estimate_density_still_classifies_both_regimes(tmp_path):
    """The only decision this feeds is `density < SPARSE_DENSITY_MAX`. Real Tahoe samples 5.9-8.7%, the
    CCL_2 simulator ~35%, and a hypothetical dense store above the gate must fall back to the dense path."""
    for density, want_sparse in ((0.07, True), (0.35, True), (0.60, False)):
        raw = _sparse_block(n=3000, g=32, density=density, seed=22)
        p = tmp_path / f"d{int(density * 100)}.dat"
        raw.tofile(p)
        got = sps.estimate_density(p, dtype="uint16", n_rows=3000, n_genes=32)
        assert abs(got - np.count_nonzero(raw) / raw.size) < 0.03
        assert (got < sps.SPARSE_DENSITY_MAX) is want_sparse


def test_estimate_density_is_deterministic(tmp_path):
    raw = _sparse_block(n=2000, g=24, density=0.09, seed=23)
    p = tmp_path / "x.dat"
    raw.tofile(p)
    kw = {"dtype": "uint16", "n_rows": 2000, "n_genes": 24}
    assert sps.estimate_density(p, **kw) == sps.estimate_density(p, **kw)


def test_estimate_density_rejects_nonpositive_slab(tmp_path):
    p = tmp_path / "x.dat"
    _sparse_block(n=50, g=8).tofile(p)
    for bad in (0, -4):
        with pytest.raises(ValueError, match="slab must be positive"):
            sps.estimate_density(p, dtype="uint16", n_rows=50, n_genes=8, slab=bad)


def test_estimate_density_handles_store_smaller_than_one_slab(tmp_path):
    """n_rows < slab must not trip rng.choice's `size > population` or read past the end of the store."""
    raw = _sparse_block(n=7, g=8, density=0.2, seed=24)
    p = tmp_path / "tiny.dat"
    raw.tofile(p)
    d = sps.estimate_density(p, dtype="uint16", n_rows=7, n_genes=8, sample=1000, slab=64)
    assert 0.0 <= d <= 1.0
