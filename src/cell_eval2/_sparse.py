"""Sparse (CSR) construction for the row-store streaming path.

Real single-cell data is overwhelmingly zeros -- Tahoe is 5.9-8.7% nonzero -- yet the row store's
.dat is DENSE on disk, so the scorer used to materialize every batch as dense float32 and spend
~94% of its arithmetic on zeros. Measured on real Tahoe, the streaming path is ~61% data conversion
(scaled_log1p 34.0%, _to_linear 26.7%) and only 4.8% disk: it is not I/O bound, it is bound by
arithmetic on zeros.

Building the slice as CSR instead makes _to_linear ~9.5x cheaper and the pseudobulk ~13x cheaper,
and -- just as importantly -- hands gpudge the format it actually wants. gpudge's numba kernel AND
its pinned-buffer H2D fast path are BOTH gated on `issparse(X) and X.format == "csr"`, so a dense X
silently falls back to scipy slicing with an unpinned copy (measured 6.9x slower on chunk extract).
We were disqualifying ourselves from an optimization that was already there.

INVARIANT 1 -- numba may SCAN and may accumulate an EXACT INTEGER SUM; numpy owns every FLOAT op.
The kernels locate nonzeros, gather their raw values, and (for unsigned-integer input only) total each
row. They do NO float arithmetic. That distinction is not stylistic -- it is what keeps the DE path
BIT-IDENTICAL (measured max abs delta = 0) -- and its two halves fail for different reasons:

  * log1p in a kernel is BANNED. numba's log1p is not numpy's log1p: a variant that moved the log1p
    into the kernel measured 4.768e-07 of drift. No argument recovers bit-identity there.
  * An integer row sum in a kernel is ALLOWED, but only under a guard. Accumulated in int64 it is
    EXACT, and an exact result cannot depend on the summation order or on who computed it. It equals
    numpy's float32 PAIRWISE sum precisely when that pairwise sum is itself exact -- i.e. when no
    partial sum rounds. For NON-NEGATIVE integers every partial sum is <= the row total, and float32
    represents every integer below 2**24 exactly, so a library under 2**24 rounds NOWHERE and the two
    agree bit-for-bit. At/above 2**24 they genuinely diverge (measured: 54/256 rows on random dense
    uint16). Hence the fused path is gated on BOTH an unsigned dtype (`_fusable_dtype`) and
    `library < _EXACT_F32_INT_MAX`, and falls back to `row_library_f32` otherwise. Real scRNA-seq
    libraries are ~1e3-1e5, so production always takes the fast path -- the guard exists so that
    correctness does not DEPEND on that remaining true.

Every remaining float op (row scaling, log1p) still runs in numpy on the compact nnz array -- same
implementation, same inputs, same order as the dense path. Do not move float math into the kernels.

INVARIANT 2 -- CSR index dtypes must MATCH. `indices` and `indptr` both get int32 iff nnz < 2**31,
else both int64. A mismatched-dtype CSR is the PR #70 landmine: scipy's X.min() upcast int32
indices to int64 and blew up a ~288 GiB temporary. A CCL_2-scale batch at 35% density reaches
~1.69e9 nnz -- 79% of the int32 ceiling -- so this is live, not theoretical.

INVARIANT 3 -- bounded temps. Row libraries are computed in ROW CHUNKS. Converting a whole batch to
float32 just to sum it would cost ~18 GiB and forfeit most of the memory win this module exists to
deliver.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp

try:                                    # numba is NOT a declared dependency of cell_eval2 -- it
    import numba                        # arrives only via the side-installed gpudge. CI has none,
    HAS_NUMBA = True                    # so CI exercises the scipy fallback in extract_nnz. That
except ImportError:                     # fallback is still a win (measured 1.57x on the host
    HAS_NUMBA = False                   # transform vs dense); numba just takes it to 3.51x.

#: Build CSR below this density. CSR costs 8 B/nnz (float32 data + a 4-byte index) against dense's
#: 4 B/elem, so the MEMORY break-even is 50%; 0.40 keeps a margin, so an unusually dense store
#: degrades to the dense path rather than regressing. Real scRNA-seq is 5-15% (Tahoe ~7%).
SPARSE_DENSITY_MAX = 0.40

#: Row-chunk budget for the library-size pass (INVARIANT 3): bounded, independent of batch size.
_LIB_CHUNK_BYTES = 256 << 20

#: Rows per contiguous slab in the density sample. 32 rows of an 18,151-gene uint16 store is a ~1.2 MB
#: sequential read -- ONE seek instead of 32. See estimate_density.
_DENSITY_SLAB = 32

#: float32 holds every integer below 2**24 exactly (24-bit mantissa). This is the precise boundary at
#: which the fused int64 row sum stops being provably equal to numpy's float32 pairwise sum. See
#: INVARIANT 1 -- it is a hard runtime gate, not a comment.
_EXACT_F32_INT_MAX = 2**24


def csr_index_dtype(nnz: int) -> type:
    """int32 iff nnz fits, else int64 -- for BOTH indices and indptr (INVARIANT 2)."""
    return np.int32 if int(nnz) < 2**31 else np.int64


def _fusable_dtype(dtype) -> bool:
    """May the row library be accumulated inside the scan kernel for this dtype?

    Unsigned integers of at most 4 bytes -- uint8 / uint16 / uint32. Two independent conditions, and
    both are load-bearing:

    * UNSIGNED. Non-negativity is what makes every partial sum <= the row total, which is what lets a
      total under 2**24 guarantee that nothing rounds (INVARIANT 1). A SIGNED store could have partial
      sums that exceed the total via cancellation, and a FLOAT store is not exact under reordering at
      all. Both keep numpy's pairwise sum.
    * AT MOST 4 BYTES. The accumulator is int64, which represents uint32 (max 4.29e9) and below
      losslessly but NOT uint64 (max 1.84e19 > int64's 9.22e18). Casting instead of excluding would be
      worse than useless: a uint64 above 2**63 wraps to a NEGATIVE int64, and a negative library sails
      straight through the `< _EXACT_F32_INT_MAX` guard and corrupts SILENTLY (Gemini, PR #106). This
      gate makes "the int64 accumulation is exact" true BY CONSTRUCTION rather than by luck about how
      big the input happens to be. It costs nothing: a real `.dat` is uint16 counts or float32 lognorm.
    """
    dt = np.dtype(dtype)
    return bool(np.issubdtype(dt, np.unsignedinteger) and dt.itemsize <= 4)


def row_library_f32(raw: np.ndarray) -> np.ndarray:
    """Per-row library size in float32 -- byte-identical to ``scaled_log1p``'s ``out.sum(axis=1)``.

    THE FALLBACK, not the primary path. ``extract_nnz_and_library`` normally produces these libraries
    for free inside the scan that is already locating the nonzeros; this runs only when that fused path
    declines (no numba / non-unsigned dtype / a library at or above 2**24). It is kept because it is the
    only implementation CI ever executes -- numba is not a declared dependency -- and because it is the
    reference the fused kernel is tested against.

    Computed in ROW chunks so the float32 temp stays bounded (INVARIANT 3). The reduction runs along
    axis=1, *within* each row, so partitioning rows cannot change any row's summation order -- this
    is bit-identical by construction, not by luck (verified: 0/30,000 rows differ on real Tahoe).

    It is also, measured, ~33x slower per byte than the kernel it now backs up (1.4 GB/s vs 46.6 GB/s
    on a 4 GiB Tahoe-shaped block): it materializes a float32 copy of every chunk purely in order to
    sum it -- read uint16 2 B, write f32 4 B, read f32 4 B. That is why it stopped being the main path.
    """
    n, g = raw.shape
    chunk = max(1, _LIB_CHUNK_BYTES // max(1, int(g) * 4))
    out = np.empty(n, dtype=np.float32)
    for i in range(0, n, chunk):
        blk = raw[i:i + chunk]
        out[i:i + chunk] = np.ascontiguousarray(blk, dtype=np.float32).sum(axis=1)
    return out


if HAS_NUMBA:

    @numba.njit(parallel=True)
    def _count_nnz(raw):                        # SCAN ONLY -- no arithmetic (INVARIANT 1)
        n, g = raw.shape
        counts = np.zeros(n, dtype=np.int64)
        for i in numba.prange(n):
            c = 0
            for j in range(g):
                if raw[i, j] != 0:
                    c += 1
            counts[i] = c
        return counts

    @numba.njit(parallel=True)
    def _count_nnz_sum(raw):                    # SCAN + EXACT int64 total (INVARIANT 1) -- no float math
        n, g = raw.shape
        counts = np.zeros(n, dtype=np.int64)
        sums = np.zeros(n, dtype=np.int64)
        for i in numba.prange(n):
            c = 0
            s = np.int64(0)
            for j in range(g):
                v = raw[i, j]
                if v != 0:
                    c += 1
                    s += v                      # zeros add 0, so skipping them changes nothing
            counts[i] = c
            sums[i] = s
        return counts, sums

    @numba.njit(parallel=True)
    def _fill_nnz(raw, indptr64, vals, cols):   # SCAN ONLY -- gathers, never computes
        n, g = raw.shape
        for i in numba.prange(n):
            p = indptr64[i]
            for j in range(g):
                v = raw[i, j]
                if v != 0:
                    vals[p] = v
                    cols[p] = j
                    p += 1


def _csr_from_counts(raw: np.ndarray, counts: np.ndarray):
    """``(values, indices, indptr)`` given a per-row nonzero count. INVARIANT 2: ONE index dtype."""
    indptr64 = np.zeros(raw.shape[0] + 1, dtype=np.int64)
    np.cumsum(counts, out=indptr64[1:])
    nnz = int(indptr64[-1])
    idt = csr_index_dtype(nnz)
    vals = np.empty(nnz, dtype=raw.dtype)
    cols = np.empty(nnz, dtype=idt)
    _fill_nnz(raw, indptr64, vals, cols)
    return vals, cols, indptr64.astype(idt, copy=False)


def extract_nnz(raw: np.ndarray):
    """``(values, indices, indptr)`` of a dense 2-D block's nonzeros. SCAN ONLY -- no arithmetic.

    ``values`` keep ``raw``'s dtype: the caller owns every float op (INVARIANT 1). ``indices`` and
    ``indptr`` share one dtype (INVARIANT 2). Columns are visited in ascending order, so indices are
    sorted within each row -- canonical CSR, which is what gpudge's numba kernel requires.
    """
    if not HAS_NUMBA:
        m = sp.csr_matrix(raw)          # scipy picks ONE consistent dtype for indices + indptr
        m.sort_indices()
        return m.data, m.indices, m.indptr
    return _csr_from_counts(raw, _count_nnz(raw))


def extract_nnz_and_library(raw: np.ndarray):
    """``(values, indices, indptr, library)`` -- the CSR structure AND the row libraries from ONE scan.

    ``library`` is float32 and BIT-IDENTICAL to ``row_library_f32(raw)``, or ``None`` when the fused
    path declines (no numba / non-unsigned dtype / a library at or above ``_EXACT_F32_INT_MAX``), in
    which case the caller MUST fall back to ``row_library_f32``. INVARIANT 1 explains why those three
    conditions are exactly the ones that can break bit-identity.

    WHY THIS EXISTS. On a clean, GPU-exclusive profile of real Tahoe, ``row_library_f32`` was the single
    most expensive function in the whole scorer -- 993.2 s, 36.7 % of a 2,703 s wall, more than all of
    gpudge -- while the numba scan that already visits every one of the same elements cost 64.6 s for
    TWO passes. The library pass was paying ~5x memory traffic, single-threaded, purely to add up numbers
    the scan was looking at anyway. Folding the sum into the scan costs +10.5 % of one scan (measured)
    and deletes the pass outright.
    """
    if not HAS_NUMBA or not _fusable_dtype(raw.dtype):
        return (*extract_nnz(raw), None)
    counts, sums = _count_nnz_sum(raw)
    csr = _csr_from_counts(raw, counts)
    if sums.size and int(sums.max()) >= _EXACT_F32_INT_MAX:
        return (*csr, None)             # a float32 sum would round here -> order becomes observable
    return (*csr, sums.astype(np.float32))      # exact int64 -> ONE correctly-rounded cast


def scaled_log1p_csr(raw: np.ndarray, target_sum: float) -> sp.csr_matrix:
    """CSR float32 of ``log1p(raw * target_sum / library)``.

    BIT-IDENTICAL to ``rowstore.scaled_log1p(raw, target_sum)``: the kernel only scans and totals
    (exactly), and numpy does every float op on the compact nnz array (INVARIANT 1). Zeros stay
    structural because ``log1p(0) == 0`` exactly.

    ONE pass over the dense block now yields both the nonzeros and the libraries; ``row_library_f32``
    runs only when ``extract_nnz_and_library`` declines the fused path.
    """
    vals, indices, indptr, lib = extract_nnz_and_library(raw)
    if lib is None:                                        # no numba / non-unsigned / library >= 2**24
        lib = row_library_f32(raw)
    lib = np.where(lib > 0, lib, np.float32(1.0))          # all-zero row -> library 1, as in dense
    data = vals.astype(np.float32)                         # numpy owns everything from here down
    data *= np.repeat(np.float32(target_sum) / lib, np.diff(indptr))
    np.log1p(data, out=data)
    return sp.csr_matrix((data, indices, indptr), shape=raw.shape)


def to_csr_f32(raw: np.ndarray) -> sp.csr_matrix:
    """CSR float32 of an already-normalized block (``output_space='raw'``).

    A pure dtype cast -- no arithmetic -- so exactly equal to
    ``np.ascontiguousarray(raw, dtype=np.float32)``.
    """
    vals, indices, indptr = extract_nnz(raw)
    return sp.csr_matrix((vals.astype(np.float32), indices, indptr), shape=raw.shape)


def estimate_density(dat_path, *, dtype, n_rows: int, n_genes: int, sample: int = 1000,
                     slab: int = _DENSITY_SLAB, seed: int = 0) -> float:
    """Nonzero fraction of the dense ``.dat``, from ~``sample`` rows read as random CONTIGUOUS SLABS.

    Sampled once per source. A row store is homogeneous enough that ~1000 rows place it firmly on one
    side of ``SPARSE_DENSITY_MAX`` (real Tahoe samples at 5.9-8.7%, the CCL_2 simulator at ~35%).

    SLABS, NOT SCATTERED ROWS -- the cost here is SEEKS, not bytes. Fancy-indexing 1000 random rows makes
    each row its own read (``n_genes * itemsize`` = 36 KB on Tahoe), so ~1000 seeks per source and ~10,000
    for a 5-context run. On wekafs that measured **103.2 s = 3.8% of wall** -- for a read this docstring
    used to call "negligible". Reading the same ~1000 rows as ~31 slabs of 32 CONTIGUOUS rows moves the
    same bytes in ~31 round trips.

    The slabs must be SPREAD, which is why this is not simply one big contiguous read: a row store is
    perturbation-CONTIGUOUS, so a single block would sit inside one perturbation and give a biased
    answer. Many small slabs at random offsets sample many perturbations.
    """
    n_rows, n_genes = int(n_rows), int(n_genes)
    if int(sample) <= 0:                                   # CALLER error: sample=0 used to crash with
        raise ValueError(                                  # ZeroDivisionError (empty block), sample<0
            f"sample must be positive, got {sample!r}")    # with a cryptic numpy ValueError. Failing
    if int(slab) <= 0:                                     # loudly beats returning 0.0, which would
        raise ValueError(                                  # silently claim "fully sparse" on zero
            f"slab must be positive, got {slab!r}")        # evidence and flip the path (Gemini #105).
    if n_rows <= 0 or n_genes <= 0:                        # empty artifact: a legitimate DATA
        return 0.0                                         # condition, not an error -> 0.0
    mm = np.memmap(dat_path, mode="r", dtype=dtype, shape=(n_rows, n_genes))
    width = min(int(slab), int(sample), n_rows)            # never overshoot the store or the budget
    n_slabs = max(1, min(int(sample) // width, n_rows - width + 1))
    starts = np.sort(np.random.default_rng(seed).choice(
        n_rows - width + 1, size=n_slabs, replace=False))
    nnz = total = 0
    for s in starts:
        blk = np.asarray(mm[int(s):int(s) + width])        # ONE contiguous read per slab
        nnz += int(np.count_nonzero(blk))
        total += int(blk.size)
    del mm
    return float(nnz) / float(total)
