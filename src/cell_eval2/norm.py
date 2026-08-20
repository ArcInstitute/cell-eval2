from __future__ import annotations

import logging

import anndata as ad
import numpy as np
from scipy.sparse import issparse

logger = logging.getLogger(__name__)

INPUT_TYPES = ("counts", "lognorm")
NORMALIZATIONS = ("counts", "normalized", "lognorm", "bulk_lognorm")

#: The token a catalog entry declares when it wants "whatever this version's expression
#: comparator is". Never an accumulator key -- `resolve_comparator` turns it into one.
EXPR_COMPARATOR = "expr_comparator"


def resolve_comparator(*, version: str, pred_input_type: str, real_input_type: str) -> str:
    """The concrete space an ``EXPR_COMPARATOR`` metric is computed in, for this run.

    ``bulk_lognorm`` (issue #264) needs raw counts on BOTH sides:

    * ``v1`` never moves -- it reproduces cell-eval/VCC, pinned by ``tests/test_v1_gate.py``.
    * Counts are unrecoverable from lognorm input (``to_normalization`` says exactly that for
      ``target="counts"``), so a lognorm side cannot supply ``P_g = sum_c y_cg``.
    * A counts-real / lognorm-pred run is a SUPPORTED path (``partition_inmem.py:210``).
      Resolving per side would put the two bulks in different spaces and the metric would
      compare incompatible quantities silently, so an asymmetric run falls back for BOTH
      sides to the only space they share.

    The fallback is self-consistent rather than a patch: ``lognorm`` is a per-cell mean and
    ``moments.trace_over_n_for`` is exact for a per-cell mean. The old comparator and the old
    analytic correction were always right together.

    ⚠️ This is a RUN-level decision. Every caller that has only one side (``precompute_cache``,
    the standalone reference builders) must be GIVEN the resolved value, never resolve its
    own -- see Task 7.

    ⚠️ **An ASYMMETRIC v2 run warns (issue #288).** The fallback is correct, but it is not free:
    the two comparators disagree by +0.8 to +1.4 on ``expr_mse_unbiased_capped_norm`` for the
    SAME predicted expression, declared two ways -- and the gap is largest where the prediction
    is best, because it is the ``lognorm`` concavity deficit (#258/#260) rather than a measure of
    accuracy. That is worth a line in the log for a run that did not ask for it. v1 does not
    warn: it never had the group-sum comparator to fall off, so there is no surprise to report.
    Not reachable under ``vcc2026``, which pins ``input_type: counts`` and
    ``autodetect_input_type: false`` inside the frozen rule.
    """
    for name, value in (("pred_input_type", pred_input_type),
                        ("real_input_type", real_input_type)):
        if value not in INPUT_TYPES:
            raise ValueError(f"{name} must be one of {INPUT_TYPES}, got {value!r}")
    if version == "v2" and pred_input_type == "counts" and real_input_type == "counts":
        return "bulk_lognorm"
    if version == "v2" and pred_input_type != real_input_type:
        logger.warning(
            "expression comparator: pred is %r and real is %r, so this v2 run falls back from "
            "the group-sum 'bulk_lognorm' comparator to the per-cell 'lognorm' one for BOTH "
            "sides -- counts cannot be recovered from log-normalized input, so the two sides "
            "have no group-sum space in common. The fallback is self-consistent, but its "
            "numbers are NOT comparable with a counts/counts run: measured at +0.8 to +1.4 on "
            "expr_mse_unbiased_capped_norm for the same predicted expression declared the two "
            # The remedy has to be stated as a DATA requirement, not a declaration one (Copilot
            # review of PR #302): "declare both sides as counts" reads as a config flag that buys
            # back `bulk_lognorm`, which would invite mis-declaring a log-normalized matrix as
            # counts. That is caught under `vcc2026` -- `validate_input_type` rejects
            # non-integer data declared counts -- but NOT with `allow_fractional_counts=True` or
            # on v1, where it would silently score nonsense.
            "ways, largest when the prediction is most accurate (issue #288). To use "
            "'bulk_lognorm', BOTH sides must actually BE counts -- supply raw counts for the "
            "log-normalized side; re-declaring it does not reconstruct them.",
            pred_input_type, real_input_type,
        )
    return "lognorm"


def _n_elements(X) -> int:
    return int(X.shape[0]) * int(X.shape[1])


# Sparse formats whose ``.data`` is a flat 1-D array of exactly the stored values AND that are
# canonical (one entry per coordinate) for this pipeline's inputs, so a reduction over ``.data``
# equals scipy's X.min()/X.max(). csr/csc from cellstream + the constructor qualify. Excluded (fall back
# to scipy's own min/max, Gemini PR #70): coo -- commonly carries duplicate coordinates that X.min()/
# X.max() SUM before reducing, so a raw .data reduction could diverge (coo is not used at scale here);
# dok/lil -- values live in a dict / object-array of Python lists; bsr/dia -- multi-dimensional .data.
_FLAT_DATA_FORMATS = ("csr", "csc")


def _sparse_extremum(X, reduce, fold) -> float:
    # scipy's X.min()/X.max() run sum_duplicates() first, which reconciles indptr and indices
    # to a single index dtype -- upcasting int32 column indices to int64 (a FULL-nnz temporary,
    # ~288 GiB at 36e9 nonzeros) whenever nnz > 2**31 forces an int64 indptr against int32
    # indices (the 5.5M-cell counts submission's exact layout: cellstream sets .data/.indices/
    # .indptr directly, so the CSR carries mismatched index dtypes). Reduce over the stored
    # .data instead: a 1-D streaming reduction that never touches the index arrays, so no dtype
    # reconciliation and no upcast. Byte-identical to X.min()/X.max() for a canonical matrix (one
    # stored value per coordinate -- the only inputs the pipeline processes) and strictly more
    # robust: scipy's X.min() RAISES on a non-canonical mismatched-dtype CSR. Called only for
    # _FLAT_DATA_FORMATS.
    nnz = int(X.nnz)
    if nnz == 0:  # all implicit zeros (caller guarantees n_elements > 0)
        return 0.0
    m = float(reduce(X.data[:nnz]))  # [:nnz] view guards an over-allocated/unpruned buffer
    if nnz != _n_elements(X):  # implicit zeros present -> fold in 0 (scalar builtin; no numpy temp)
        m = fold(m, 0.0)
    return m


def _min_value(X) -> float:
    if _n_elements(X) == 0:
        return 0.0
    if issparse(X):
        if getattr(X, "format", None) in _FLAT_DATA_FORMATS:
            return _sparse_extremum(X, np.min, min)  # reduce over .data, no index-dtype upcast
        return float(X.min())  # exotic formats (dok/lil/bsr/dia): defer to scipy (never seen at scale)
    return float(np.asarray(X).min())


def _max_value(X) -> float:
    if _n_elements(X) == 0:
        return 0.0
    if issparse(X):
        if getattr(X, "format", None) in _FLAT_DATA_FORMATS:
            return _sparse_extremum(X, np.max, max)
        return float(X.max())
    return float(np.asarray(X).max())


_INT_CHUNK = 1 << 22  # ~4M elements/chunk: chunk-sized temporaries only, never full-matrix

#: Absolute tolerance on the integrality test, in COUNTS -- paired with ``rtol=0`` at both
#: ``np.allclose`` sites below, and the reason they take an explicit tolerance at all.
#:
#: ``np.allclose``'s DEFAULT tolerance is RELATIVE (``atol + rtol*|b|``, ``rtol=1e-5``), so the
#: deviation it accepts as "still an integer" scales with the value: +-0.001 at 100, +-0.01 at
#: 1,000, and +-0.5 at 50,000 -- every possible fractional part. Above 50,000 the tolerance exceeds
#: a half and EVERY value passes. Nothing else bounded it: ``check_scale_limit`` caps the per-CELL
#: total at 1e6 under ``vcc2026``, so a single entry at 50,000+ is legal. MEASURED before this was
#: fixed (numpy 2.5.2): ``5000.5`` rejected, ``50000.5`` and ``999999.5`` ACCEPTED as counts with
#: ``allow_fractional=False``. ``docs/vcc2026_metrics/vcc2026-metrics.md`` section 0 -- the published
#: metric specification -- says a submission "must be raw, untransformed counts: non-negative,
#: integral, ... A matrix failing any of these is rejected rather than scored". This does not make
#: the gate that rule; it closes the VALUE-SCALED hole in the proxy for it, leaving an
#: integral-to-1e-6 test on the paths that reach this function (``partition_inmem.score_piece`` and
#: the direct shard drivers do not -- see ``check_scale_limit``'s docstring). Stated that way on
#: purpose: "now enforces the published rule" would be a third overclaim in the same comment.
#:
#: NOT exact equality, deliberately. float32 represents every integer exactly to ``2**24`` =
#: 16,777,216, sixteen times the per-cell cap, so ``(x == rint(x)).all()`` is defensible on the
#: width argument alone -- but tightening a validation gate is the one change here that can turn a
#: previously-accepted submission into a hard refusal, and this keeps a noise margin for an
#: otherwise-integral matrix carrying dust from an upstream float64 transform.
#:
#: ⚠️ What that margin is worth depends on the input's DTYPE, and the honest reading is
#: float64-shaped. At 1e6 this is ~8,590 float64 ULP but 1.6e-5 of a float32 ULP (float32's spacing
#: at 1e6 is 0.0625). MEASURED, the exact float32 crossover: a one-ULP-off value is still accepted
#: below 16 (spacing 9.54e-07 in [8, 16)) and rejected from 16 up (spacing 1.91e-06), so on FLOAT32
#: input this is exact equality **at 16 and above** -- not "about 10", and the boundary is a binade
#: edge rather than a round number. That is the intended strictness -- at that width a value which is
#: not the integer is not the integer -- but the margin then buys nothing for float32 ARITHMETIC
#: dust: MEASURED, ``expm1(log1p(x))`` in float32 over Poisson(50) counts deviates up to 1.5e-5 from
#: the integers it round-trips, which the old relative rule accepted and this rejects. A submission
#: reconstructed that way has to be rounded before it is counts.
#:
#: ⚠️ So this enforces "integral to 1e-6", not literal integrality, and the accepted dust is NOT
#: rounded -- it reaches the metrics as stored. MEASURED: float64 ``100.0000005`` and float32
#: ``10.000000953674316`` both pass, as does ``0.9999995`` -- the test is distance to the NEAREST
#: integer, not the size of a fractional part. A 400-cell group sum therefore carries at most ~4e-4
#: of a count from accepted dust, and above 16 a float32 matrix cannot hold such a value at all.
#: RULED (2026-08-19, Alex): KEEP the tolerance. Exact equality is defensible and was put to him
#: explicitly -- ``codex-review`` recommended it independently as the specification-correct rule,
#: since it would reject every dust example above -- and the noise margin is the deliberate choice
#: over it,
#: so ``1e-6`` is a decision rather than an oversight. Do not "tighten" this to ``==`` without
#: re-opening that decision; the tests below pin the margin on purpose.
#:
#: ⚠️ **The bound is 1e-6 AS REPRESENTABLE IN THE INPUT'S DTYPE, which is not 1e-6 for float16.**
#: ``_int_atol`` rounds it to the input's dtype EXPLICITLY (see there for why the code does this
#: rather than leaving it to numpy's promotion rules), and MEASURED on numpy 2.5.2
#: ``np.float16(1e-6)`` is ``1.0132789611816406e-06`` -- so float16 input gets that as its effective
#: tolerance and a stored ``1.0132789611816406e-06`` is ACCEPTED while ``1.0728836059570312e-06``
#: is not. Left in the input's dtype deliberately: promoting the
#: comparison would add a chunk-sized float64 temporary to the one path that exists to avoid
#: full-matrix temporaries, for a dtype ``tests/test_jackknife.py`` calls "not a realistic
#: submission format" (``prep._grouped_sums``' exposure list documents it; that phrase is the
#: test's).
#:
#: ⚠️ **This is NOT a pure tightening: in the neighbourhood of ZERO it LOOSENS by 100x.** The old
#: tolerance was ``atol + rtol*|rint(v)|``, and ``rint(v) == 0`` kills the relative term -- so near
#: zero the old rule was ``atol=1e-8`` alone, tighter than this. MEASURED, the verdict flips in both
#: directions: a value of ``5e-07`` was NOT integral and now is. The factor is 100x in float32 and
#: float64; in float16 it is unbounded, because ``np.float16(1e-8)`` UNDERFLOWS TO 0.0 -- the old
#: rule admitted no near-zero dust at all at that width. Two consequences, both real and both pinned
#: by tests:
#:
#:   * a counts matrix whose only non-integrality is sub-microcount dust near zero is newly ACCEPTED;
#:   * the lognorm-mislabel raise (below) can therefore newly FIRE, on an all-dust matrix that used
#:     to read as "not all-integer" -- so "a stricter predicate makes that gate fire less" is true
#:     only away from zero.
#:
#: Accepted rather than special-cased. A uniform absolute tolerance is the coherent rule the
#: relative one was not, and the direction it moves near zero is 1e-6 of a count on a value whose
#: nearest integer is 0 -- which is what "dust" means. Reinstating ``1e-8`` below some threshold
#: would buy monotonicity with a second, unmeasured constant.
_INT_ATOL = 1e-6


def _int_atol(dtype):
    """``_INT_ATOL`` as the input's OWN dtype represents it -- from the code, not from a promotion
    rule.

    ``numpy.isclose`` passes an ``int``/``float``/``complex`` tolerance through un-arrayed, so under
    NEP 50 it stays a WEAK scalar and the comparison already happens in the array's dtype with
    ``atol`` already rounded to it. This states that as POLICY instead of inheriting it, because
    ``pyproject`` declares ``numpy>=1.26`` and that range spans the NEP 50 switch, so the documented
    "1e-6 as representable in the input's dtype" was otherwise a property of whichever numpy resolved
    rather than of this function (Copilot, PR #341).

    ⚠️ It pins an outcome the two promotion regimes already SHARE -- it does not repair a measured
    divergence, and saying so would be a third overclaim in this file. Under numpy 1.26 the tolerance
    stays in-width because the LEGACY UFUNC PROMOTION of ``atol + rtol*abs(y)`` value-minimizes
    ``1e-6`` to the array's (or complex component's) width, so float16/32/64/longdouble and
    complex64/128 land on the same tolerance widths as under NEP 50. ⚠️ NOT because of
    ``result_type(y, 1.)``: that expression fixes ``y``'s dtype and never sees ``atol`` -- a wrong
    mechanism for the right conclusion, in two earlier drafts of this paragraph (codex review).
    ``NPY_PROMOTION_STATE`` is ignored after numpy 2.2, so the 1.26 side is read from its source
    rather than measured here.

    What the cast buys is that a future promotion change cannot move the FLOATING-dtype tolerance
    VALUE silently. It is not a freeze on ``rint``/``isclose`` behaviour, and it does not cover bool
    or complex, which stay on the uncast path by design. VERIFIED a no-op on numpy 2.5.2: identical
    verdicts for float16, float32, float64 and longdouble, including at float16's 17-vs-18-subnormal
    boundary.

    Floating ONLY, and the guard is load-bearing: ``np.bool_(1e-6)`` is ``True``, so casting there
    would INVENT a tolerance rather than pin one. ``bool`` and ``complex`` reach the two callers too
    and keep the weak-scalar path, unchanged.

    What those two actually do, since "complex compares on its magnitude" -- an earlier draft here --
    is misleading (Copilot). Neither integrality caller takes ``abs(x)`` before ``rint`` (``abs``
    appears elsewhere in this module, in ``guess_is_lognorm``): ``np.rint`` is applied COMPONENTWISE
    to a complex array, and ``np.allclose``'s own kernel compares ``abs(x - rint(x))``, the MODULUS OF
    THE DIFFERENCE -- not of the value. MEASURED: ``3+0.5j`` and ``3.5+0j`` both read as non-integral
    (deviation 5.0e-01 either way) while ``3+1e-07j`` reads as integral, so a fractional IMAGINARY
    part is caught. bool is simpler: its values equal their rounded targets exactly, so the tolerance
    is immaterial.

    ⚠️ The two are NOT in the same position, and a further draft got that wrong by calling both
    "unsupported" (codex review). The published rule is VALUE-based, not dtype-based, so a bool 0/1
    matrix is legitimately integral counts and ``validate_input_type`` ACCEPTS it -- it merely misses
    the integer fast path, because ``np.issubdtype(np.bool_, np.integer)`` is False. COMPLEX is the
    one outside the real-count contract, and it remains a documented unclosed exposure:
    ``prep._grouped_sums``' docstring records that its imaginary part is discarded with a
    ``ComplexWarning`` a caller can ignore, and that closing it belongs in ``validate_input_type``
    rather than there. This change does not close it.
    """
    # `np.dtype(...)` coerces, so a raw type class or a string works too. Both of this function's
    # callers pass a real `np.dtype`, so this is not for them -- it is because a private helper taking
    # a "dtype" and then raising `AttributeError: type object 'numpy.float32' has no attribute 'type'`
    # on `np.float32` is a trap for the next caller (Gemini, PR #341). MEASURED: `np.float32` and
    # `"float32"` both raised before this line.
    dtype = np.dtype(dtype)
    return dtype.type(_INT_ATOL) if np.issubdtype(dtype, np.floating) else _INT_ATOL


def _all_integer_1d(vals) -> bool:
    # AND of np.allclose over fixed-size chunks == np.allclose over the whole (per-element
    # tolerance, no cross-element interaction), but streamed and short-circuiting on the first
    # non-integer chunk.
    atol = _int_atol(vals.dtype)
    for start in range(0, vals.size, _INT_CHUNK):
        chunk = vals[start:start + _INT_CHUNK]
        if not np.allclose(chunk, np.rint(chunk), rtol=0, atol=atol):
            return False
    return True


def _is_all_integer(X) -> bool:
    # Sparse implicit zeros are integers, so only the stored values matter.
    if issparse(X):
        data = X.data
        if data.size == 0:
            return True
        return bool(np.issubdtype(data.dtype, np.integer)) or _all_integer_1d(data)
    arr = np.asarray(X)
    if arr.size == 0:
        return True
    if np.issubdtype(arr.dtype, np.integer):
        return True
    if arr.ndim != 2:
        return _all_integer_1d(arr.reshape(-1))
    # 2D dense: chunk along rows; a row-slice of an ndarray is a view (no full copy).
    rows_per_chunk = max(1, _INT_CHUNK // max(1, arr.shape[1]))
    atol = _int_atol(arr.dtype)
    for start in range(0, arr.shape[0], rows_per_chunk):
        block = arr[start:start + rows_per_chunk]
        if not np.allclose(block, np.rint(block), rtol=0, atol=atol):
            return False
    return True


def _csr_row_block(X, start, stop, *, data=None):
    """Row-block ``X[start:stop]`` as a check-free CSR of ``X``'s class.

    ``X`` is CSR; a contiguous row-slice of a valid CSR is itself valid (same ``indices``/``data``,
    ``indptr`` shifted by a constant). scipy's CSR constructor re-runs an O(nnz) ``check_format``
    even on ``copy=False`` construction from raw arrays, and its ``prune()`` step passes each array
    through ``_prune_array``, which COPIES any view satisfying ``size < base.size // 2`` (integer
    floor -- 4 of 9 does NOT qualify). Under roughly uniform nnz-per-row that holds for every block
    of a matrix several chunks tall, which is the regime this exists for; a block hoarding a
    disproportionate share of the nonzeros can still sit over the line. So ``copy=False`` does not
    avoid the copy. Build an empty of the correct shape (cheap) and assign the parent's ``data`` and
    ``indices`` *views* directly instead -- ``indptr`` is a fresh O(rows) array either way, since
    shifting it by the block offset allocates. Callers only READ the block, so the views add no host
    memory.

    Contract: the same logical block, with a bit-identical ``sum(axis=1)`` and identical ``.data``
    dtype and values. It is NOT byte-identical in the index arrays -- it PRESERVES the parent's
    ``indices``/``indptr`` dtypes where the raw constructor reconciles them (on a csr_array with
    int32 ``indices`` + int64 ``indptr``, the raw constructor upcasts ``indices`` to int64). That
    divergence is the point: avoiding exactly that upcast is why the row reductions chunk at all.

    ``data`` replaces the block's ``X.data[lo:hi]`` slice with a transform of it (e.g. ``np.expm1``)
    and must be 1-D of the same length.

    The empty block is built with NO ``dtype=``: the dtype comes from assigning ``.data``, and
    ``.dtype`` is a property derived from it. Passing ``dtype=data.dtype`` looks natural and is
    wrong -- ``np.expm1`` of ``bool``/``int8``/``uint8`` data returns **float16** (numpy 2.4.6), and
    scipy 1.18.0's empty-CSR constructor REJECTS ``dtype=float16`` while the raw-array constructor
    this replaces accepts it. That spelling would raise where the old code worked.

    Lives here (a leaf module: numpy/scipy/anndata only) and is re-exported by ``streaming_bulk``,
    which is where PR #73 first added it for ``inmem_pseudobulk``.
    """
    lo, hi = int(X.indptr[start]), int(X.indptr[stop])
    if data is None:
        data = X.data[lo:hi]
    else:
        data = np.asarray(data)
        if data.ndim != 1:
            raise ValueError(f"data override must be 1-D, got ndim={data.ndim}")
        if data.shape[0] != hi - lo:
            raise ValueError(
                f"data override has length {data.shape[0]}, expected length {hi - lo} "
                f"for rows [{start}, {stop})"
            )
    Xb = X.__class__((stop - start, X.shape[1]))  # empty, correct shape; dtype follows .data
    Xb.data = data
    Xb.indices = X.indices[lo:hi]
    Xb.indptr = X.indptr[start:stop + 1] - X.indptr[start]
    return Xb


_ROW_TOTALS_ROW_CHUNK = 100_000  # rows/block: bounds the per-block float64 upcast temporary


def _row_totals(X) -> np.ndarray:
    if issparse(X):
        # scipy's X.sum(axis=1) upcasts the ENTIRE data array to a wide dtype -- a full-nnz-sized
        # temporary (~288 GB at billions of nonzeros, e.g. a 5.5M-cell counts submission -> peaks
        # the in-memory scale-limit check near the host-RAM ceiling). For CSR, sum row-blocks
        # instead: each block's .sum(axis=1) upcasts only that block's data (bounded), delegating to
        # scipy for correctness (empty and trailing-empty rows included). Non-CSR formats (rare,
        # typically smaller) keep the direct sum.
        if getattr(X, "format", None) == "csr":
            n = int(X.shape[0])
            totals = np.empty(n, dtype=np.float64)
            for start in range(0, n, _ROW_TOTALS_ROW_CHUNK):
                stop = min(start + _ROW_TOTALS_ROW_CHUNK, n)
                # check-free block VIEW: scipy's constructor prunes (copies) data + indices for
                # every block satisfying size < base.size // 2, which is what makes a validation
                # pass cost up to two full copies of the matrix.
                block = _csr_row_block(X, start, stop)
                totals[start:stop] = np.asarray(block.sum(axis=1)).ravel()
            return totals
        return np.asarray(X.sum(axis=1)).ravel()
    return np.asarray(X).sum(axis=1)


def _expm1_row_totals(X) -> np.ndarray:
    # expm1(0) == 0, so only stored entries contribute; keep the matrix sparse.
    if issparse(X):
        # Non-CSR -> CSR first. tocsr() CANONICALIZES: it sums duplicate coordinates (expm1 is not
        # additive -- expm1(a) + expm1(b) != expm1(a + b) -- so applying expm1 to a raw non-canonical
        # coo .data before summing would be wrong) and yields a flat numeric .data (lil stores an
        # object-array of Python lists, dok has no .data -- both would raise). Non-CSR inputs are rare
        # and small here (cellstream yields CSR), so the conversion copy is fine. (Gemini PR #71.)
        if getattr(X, "format", None) != "csr":
            X = X.tocsr()
        # A CSR input is assumed CANONICAL (one entry per coordinate) -- true for every pipeline input
        # (real count/lognorm matrices have one value per (cell, gene); cellstream decodes canonical),
        # matching _min_value/_max_value (PR #70) and the pre-existing behavior. We deliberately do NOT
        # sum_duplicates()/read has_canonical_format on the FULL matrix to canonicalize a hypothetical
        # non-canonical CSR: on the mismatched-dtype CSR (int32 indices + int64 indptr) that call
        # upcasts the indices int32->int64 -- the ~288 GiB full-nnz temporary this function exists to
        # avoid (verified). Canonicalizing a non-canonical CSR would require per-block sum_duplicates.
        # Sum row-blocks: per block, expm1 ONLY that block's .data (a bounded float64 temporary) and
        # build the block CSR from `data`/`indices` VIEWS and a shifted `indptr`. The full-matrix path --
        # X.__class__((np.expm1(X.data), X.indices, X.indptr), shape) -- allocates a full-nnz float64
        # expm1 temporary AND, once nnz > 2**31 forces an int64 indptr against int32 column indices,
        # the constructor upcasts those indices int32->int64 (a second full-nnz temporary, ~2x ~288
        # GiB at 36e9 nonzeros -> over the host-RAM ceiling for a 5.5M-cell lognorm submission). Same
        # chunking as _row_totals; each block's nnz < 2**31 keeps the block index dtype int32 (no
        # upcast), and scipy handles empty / trailing-empty rows.
        indptr = X.indptr
        n = int(X.shape[0])
        totals = np.empty(n, dtype=np.float64)
        for start in range(0, n, _ROW_TOTALS_ROW_CHUNK):
            stop = min(start + _ROW_TOTALS_ROW_CHUNK, n)
            lo, hi = int(indptr[start]), int(indptr[stop])
            # expm1 ONLY this block's data (a bounded temporary); indices stays a view of the
            # parent (indptr is a fresh O(rows) array either way -- the offset shift allocates).
            block = _csr_row_block(X, start, stop, data=np.expm1(X.data[lo:hi]))
            totals[start:stop] = np.asarray(block.sum(axis=1)).ravel()
        return totals
    # dense: accumulate row totals in row-chunks; the expm1 temporary is chunk-sized only.
    arr = np.asarray(X)
    n = arr.shape[0]
    totals = np.empty(n, dtype=np.float64)
    rows_per_chunk = max(1, _INT_CHUNK // max(1, arr.shape[1]))
    for start in range(0, n, rows_per_chunk):
        block = arr[start:start + rows_per_chunk]
        totals[start:start + block.shape[0]] = np.expm1(block).sum(axis=1)
    return totals


def guess_is_lognorm(
    adata: ad.AnnData, *, n_cells: int = 500, epsilon: float = 1e-6, seed: int = 0
) -> bool:
    """Port of cell-eval utils.guess_is_lognorm.

    Sample up to n_cells, sum each across genes, and classify as lognorm if any
    cell total has a fractional part greater than epsilon.
    """
    X = adata.X
    n = int(adata.n_obs)
    if n == 0:
        return False
    if n <= n_cells:
        idx = np.arange(n)
    else:
        idx = np.sort(np.random.default_rng(seed).choice(n, size=n_cells, replace=False))
    sub = X[idx]
    sums = np.asarray(sub.sum(axis=1)).ravel() if issparse(sub) else np.asarray(sub).sum(axis=1)
    frac = np.abs(sums - np.rint(sums))
    return bool(np.any(frac > epsilon))


def resolve_input_type(
    adata: ad.AnnData, *, declared: str, version: str, allow_discrete: bool, autodetect: bool = False
) -> str:
    """Resolve the effective matrix convention for DE and normalization paths.

    v1 reproduces cell-eval's auto-detection: integer counts (and not allow_discrete)
    -> "counts"; fractional -> "lognorm". v2 trusts the declared type. ``autodetect``
    forces the v1-style auto-detection regardless of version (an opt-in so a fractional
    v2 submission is detected as "lognorm" instead of rejected). allow_discrete forces
    "counts" within the auto-detect path.
    """
    if version != "v1" and not autodetect:
        return declared
    if allow_discrete:
        return "counts"
    return "lognorm" if guess_is_lognorm(adata) else "counts"


def validate_input_type(adata: ad.AnnData, input_type: str, *, allow_fractional: bool = False) -> None:
    """Raise if the data is inconsistent with the declared input type.

    Sparse matrices are inspected without densification (memory-aware). ``allow_fractional``
    permits fractional values under ``input_type='counts'`` (e.g. a constructed average/null
    predictor); the negative-value and lognorm-mislabel checks are unaffected.
    """
    if input_type not in INPUT_TYPES:
        raise ValueError(f"input_type must be one of {INPUT_TYPES}, got {input_type!r}")
    X = adata.X
    if _min_value(X) < 0:
        raise ValueError("data contains negative values; expected non-negative")
    all_integer = _is_all_integer(X)
    if input_type == "counts" and not all_integer and not allow_fractional:
        raise ValueError("declared input_type='counts' but values are fractional")
    if input_type == "lognorm" and all_integer and _max_value(X) > 0:
        raise ValueError(
            "declared input_type='lognorm' but values are all-integer "
            "(likely mislabeled raw counts)"
        )


class ScaleLimitError(ValueError):
    """Let callers recognise a scale-limit rejection without string matching.

    This subclasses ``ValueError`` so every existing ``except ValueError`` and
    ``pytest.raises(ValueError)`` continues to work unchanged.
    """


#: Relative slack allowed on the RECONSTRUCTED per-cell total, in ULPs of the stored matrix's
#: own dtype (issue #287). Applies to the ``lognorm`` branch ONLY, where the total is recovered
#: by ``expm1`` of values that were stored after a ``log1p`` -- a round trip whose error the
#: counts branch simply does not have.
#:
#: MEASURED, not chosen: over an adversarial sweep of this library's own
#: ``to_normalization(counts -> lognorm)`` output -- caps 1e3..1e9, G from 2 to 18,533, uniform /
#: heavy-tailed / one-gene-spike compositions -- the worst relative excess of
#: ``_expm1_row_totals`` over the exact target was **8.05 ULP**, and it does NOT grow with G (the
#: dominant term is storing ``log1p(v)`` in float32, whose relative error in ``v`` is
#: ``~log1p(v) * eps``, not the row summation). 16 leaves a 2x margin over that ceiling and reads
#: as a precision budget rather than a fitted constant.
#:
#: ⚠️ **That domain is caps 1e3..1e9, and ``EvalConfig`` accepts any positive finite cap.** The
#: signed excess OSCILLATES with rounding phase -- it is not monotone in the cap -- but its
#: worst-case ENVELOPE is O(log1p(cap)), so far above the measured range a budget fixed in ULPs
#: stops covering the envelope. MEASURED: an exact-cap one-gene float32 normalization reads +6.4
#: ULP at cap 1e9 and +3.3 at 1e12 (both accepted), **+16.7 at cap 79101186206443.31, which is REJECTED**,
#: then -0.1 at 1e15 (accepted again -- the oscillation), and +32.8 at 4.31e31. So the first
#: rejection OBSERVED IN THE SWEEP is around 1e14, not somewhere astronomical. The cap
#: literal cannot be rounded: 7.9e13 reads only +1.9 ULP and is accepted, because the
#: excess oscillates (codex review rounds 2-4). That is the gate
#: failing closed on an input outside the domain its tolerance was measured on -- the recoverable
#: direction, and irrelevant to ``vcc2026``'s 1e6 -- but a cap-conditioned bound would need its own
#: measurement before it could replace this one.
#:
#: ⚠️ What this costs, stated plainly: at the v2 cap of 1e6 one extra COUNT is 8.39 ULP, i.e. the
#: same size as the roundoff itself. A breach of a few counts at that cap is therefore not
#: resolvable from float32 lognorm input by ANY tolerance -- the information is gone before this
#: function sees it. The gate goes on rejecting what it exists to reject (over-budget by orders
#: of magnitude, and the per-value ``log1p`` overflow guard above remains in place, now with the
#: same tolerance); it just no longer
#: rejects the library's own exact-cap output, which is what #287 measured it doing.
_SCALE_LIMIT_TOL_ULP = 16.0


def _scale_limit_rtol(X) -> float:
    """Relative tolerance for the lognorm comparisons, from ``X``'s OWN dtype, NEVER looser than
    float32's.

    Tied to the stored precision rather than fixed: a lognorm matrix held in float64 has ~1e-9 of
    the round-trip error a float32 one has, and giving it float32's slack would loosen the gate by
    eight orders for no reason.

    ⚠️ The ``min`` is load-bearing and the tolerance is CAPPED, not merely scaled (codex review).
    16 ULP was measured on float32/float64; applied to a narrower dtype it is not a small number.
    float16's eps is 9.77e-04, so 16 ULP of it is a **1.6% relative** budget -- a reconstructed
    total of 1010.0 would be accepted against a cap of 1000. That is arguably the honest precision
    of float16 lognorm input, but a gate should fail CLOSED on an input outside the domain its
    tolerance was measured on, not open. Capping at float32's budget makes a float16 matrix
    normalized exactly to the cap get REJECTED rather than silently granted a percent of slack;
    float16 lognorm is not a supported input, and rejection is the recoverable direction.

    Non-float dtypes fall back to float32 -- ``validate_input_type`` already rejects an all-integer
    matrix declared ``lognorm``, so that is a floor for exotic inputs rather than a live path.
    Precisions WIDER than float64 (longdouble) get float64's tolerance because
    ``_expm1_row_totals`` and ``_max_value`` reduce through float64 anyway, so the extra precision
    is not preserved end to end and claiming it would be false.
    """
    dtype = getattr(getattr(X, "dtype", None), "type", None)
    if dtype is None or not np.issubdtype(dtype, np.floating):
        dtype = np.float32
    eps = float(np.finfo(dtype).eps)
    # never looser than float32's budget (narrow dtypes), never tighter than float64's (wide ones).
    # Kept as two steps rather than one `np.clip` (Gemini review of PR #302 proposed collapsing
    # them): the two bounds exist for OPPOSITE reasons, each spelled out above, and this guard
    # already shipped three inverted versions during review -- a single clip line is where the
    # reason for each direction goes to die. It would also return an `np.float64` where the
    # annotation says `float`.
    eps = min(eps, float(np.finfo(np.float32).eps))
    eps = max(eps, float(np.finfo(np.float64).eps))
    return _SCALE_LIMIT_TOL_ULP * eps


def check_scale_limit(
    adata: ad.AnnData, input_type: str, max_counts_per_cell: float,
    *, precomputed_row_total_max: float | None = None,
) -> None:
    """Reject submissions whose per-cell totals exceed max_counts_per_cell.

    For lognorm input the check is overflow-safe: a value exceeding
    log1p(max_counts_per_cell) is rejected before expm1 is ever evaluated.
    Sparse matrices are inspected without densification (memory-aware).

    ``precomputed_row_total_max`` (counts only): the max raw per-cell total already
    computed elsewhere (the pseudobulk accumulator's libs_host, via
    run._check_scale_limit_once). When supplied, the counts path reuses it instead of a
    second full-matrix _row_totals pass. Ignored for lognorm (a counts concept; lognorm
    keeps its expm1 overflow-safe path).

    ⚠️ The LOGNORM comparison carries a relative tolerance (``_SCALE_LIMIT_TOL_ULP``, issue
    #287); the counts one does NOT and must not. A counts row is a sum of raw stored values with
    no round trip in it, and for non-negative integers every partial sum stays below the cap, so
    that reduction is exact and a tolerance there would only weaken the gate.

    ⚠️ Every comparison here is ``>``, so a NaN answers ``False`` and would pass a gate it never
    cleared. NaN is therefore rejected explicitly rather than by comparison (Copilot review of
    PR #302). What each non-finite value does when the matrix is INSPECTED, all measured -- with a
    finite ``precomputed_row_total_max`` none of the three is seen at all, because that shortcut
    skips ``_row_totals`` and the matrix is never read (trusted-producer contract; no current
    producer can reach it, since the GPU accumulator propagates NaN into its own max):

    * **NaN** passed BOTH branches before this check. Now rejected here.
    * **+inf** already raised in both, because ``+inf > budget`` is True. Untouched.
    * **-inf** passes both and still does: the counts row total is ``-inf`` so the *max* ignores
      it, and ``expm1(-inf)`` is a finite ``-1``. Deliberately NOT handled here --
      ``validate_input_type`` rejects negatives in every mode ("data contains negative values"),
      including ``allow_fractional=True`` and ``input_type="lognorm"``, so a magnitude budget is
      the wrong place to re-derive a sign rule. ⚠️ "Every mode" is true of the MODE and not of every
      MATRIX: ``_min_value`` reduces with ``np.min``, so a matrix carrying BOTH a NaN and a negative
      has a NaN minimum, ``NaN < 0`` is False, and that sign check does not fire -- which is the
      NaN half of this same gate, above. Pinned by
      ``tests/test_norm.py::test_a_NaN_POISONS_the_sign_check_so_negative_dust_survives_it``.

    Where the NaN hole was open: ``run.full`` calls ``_validate_input_once`` BEFORE
    ``_check_scale_limit_once`` and ``_is_all_integer`` is False on a NaN matrix, so the ordinary
    ``compute_metrics`` path already rejected it via ``allow_fractional_counts=False``. It stayed
    open for ``version="v1"`` (which skips that validation), for a fractional-allowed leg, and for
    ``input_type="lognorm"`` -- whose validation DOES test integrality, but in the opposite
    direction: it rejects an all-integer matrix as mislabeled counts, and a NaN matrix reads as
    "not all-integer", which is the verdict lognorm wants. ``partition_inmem.score_piece``
    materializes a pred piece with neither gate -- see ``score_piece``; that file is not
    this chunk's to change.
    """
    X = adata.X
    if _n_elements(X) == 0:
        return
    budget = max_counts_per_cell
    rtol = 0.0   # bound here, not only in the lognorm branch: the row-total message below reports
                 # it and must not RE-DERIVE it from `budget / max_counts_per_cell - 1`, which is 0
                 # at a subnormal cap and can overflow at an extreme one (codex review)
    if input_type == "lognorm":
        # `expm1` of values stored after a `log1p` cannot land back on the exact total, so a row
        # this library itself normalized TO the cap comes back a hair above it. BOTH lognorm
        # comparisons get that round trip's own precision, sized from the stored dtype (#287):
        # the per-VALUE guard sees it too, because a single gene carrying the whole budget stores
        # a log1p that rounds up past the float64 cap (measured at cap=1e3, where float32
        # log1p(1000) lands above np.log1p(1000)). One `rtol` for both -- the stored value and the
        # reconstructed total carry the same relative float32 error.
        #
        # ⚠️ The two applications are in DIFFERENT SPACES, and that is deliberate (Copilot review of
        # PR #302, which read it as a units bug). The per-VALUE guard budgets the error of storing
        # a LOG value in float32, which is relative to that log value, so `cap * (1 + rtol)` is its
        # natural form. The row-total guard budgets the error of reconstructing a whole row with
        # `expm1` and summing, which is relative on the COUNTS scale. Translated to counts, the
        # per-value slack is therefore `cap`x looser: MEASURED at the v2 cap, 2.635e-05 relative
        # against the row guard's 1.907e-06, a ratio of 13.816 == log1p(1e6).
        #
        # It cannot leak, because the per-value guard is an early/overflow check and the ROW TOTAL
        # is what binds on magnitude: a single gene stored just under the per-value bound is
        # rejected by the row-total guard (measured; pinned in test_norm.py). And the tightening
        # Copilot proposed -- `log1p(max_counts_per_cell * (1 + rtol))`, i.e. rtol on the counts
        # scale -- is END-TO-END MORE PERMISSIVE in the extreme case it cites as the risk: at a cap
        # of 1.7e308 the current bound admits a value the row-total guard then rejects, while the
        # proposed bound admits one that passes BOTH. Neither form makes `expm1` overflow there.
        rtol = _scale_limit_rtol(X)
        cap = np.log1p(max_counts_per_cell)
        mx = _max_value(X)
        if np.isnan(mx):
            raise ScaleLimitError(
                "lognorm matrix contains NaN; reject (a NaN compares False against every "
                "bound, so it would pass this gate without clearing it)"
            )
        if mx > cap * (1.0 + rtol):
            # Names the tolerance for the same reason the row-total message below does (Copilot
            # review of PR #302): both comparisons carry `rtol`, so a message quoting only the
            # untolerated cap makes a boundary rejection look like an arithmetic error.
            #
            # It reports the RELATIVE excess and `rtol` rather than the tolerated bound, because
            # `rtol` is dtype-scaled and printing two nearly-equal numbers is useless at every
            # fixed precision: on float64 input the bound differs from the cap in the 14th
            # decimal, so any readable format renders "exceeds 13.815512 ... up to 13.815512".
            #
            # The excess is measured against the CAP the sentence names, not against the tolerated
            # bound (codex review): measured against the bound it prints ~0 just past the
            # threshold, which contradicts its own "exceeds <cap> by" prose. Against the cap, a
            # marginal rejection prints ~rtol and the clause that follows says why that was not
            # enough.
            # `errstate` because the ratio is only a diagnostic: at a degenerate
            # `max_counts_per_cell` (subnormal, so `cap` underflows to ~0) it overflows, and a
            # RAISE must not also emit a NumPy warning on its way out. `inf relative` is then the
            # honest thing to print.
            with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
                excess = float(mx / cap - 1.0)
            # `.9g` and the RAW cap alongside its log, for the same reason as the row-total message
            # (Copilot review): `.6f` rendered a subnormal cap's log as `0.000000`, and the operator
            # configured `max_counts_per_cell`, not `log1p` of it.
            raise ScaleLimitError(
                f"lognorm value {mx:.9g} exceeds "
                f"log1p(max_counts_per_cell={max_counts_per_cell:.9g})={cap:.9g} by "
                f"{excess:.3g} relative -- more than the "
                f"{_SCALE_LIMIT_TOL_ULP:.0f} ULP allowed for the log1p round trip "
                f"(rtol={rtol:.3g}); reject (would over-budget / overflow expm1)"
            )
        totals_max = float(np.max(_expm1_row_totals(X)))
        budget = max_counts_per_cell * (1.0 + rtol)
    elif precomputed_row_total_max is not None:
        totals_max = float(precomputed_row_total_max)  # reuse the accumulator's libs max (no _row_totals pass)
    else:
        totals_max = float(np.max(_row_totals(X)))
    if np.isnan(totals_max):
        raise ScaleLimitError(
            f"per-cell total is NaN for {input_type} input; reject (a NaN compares False "
            f"against max_counts_per_cell={max_counts_per_cell:.9g}, so it would pass this "
            f"gate without clearing it)"
        )
    if totals_max > budget:
        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            excess = float(totals_max / max_counts_per_cell - 1.0)
        raise ScaleLimitError(
            # `.9g` rather than `.1f` on every configured quantity (Copilot review of PR #302,
            # raised twice): `.1f` renders a subnormal cap as `0.0` and a huge one without its
            # exponent, so a message written to make pathological input diagnosable was hiding the
            # input. At ordinary scales it reads the same or better -- 1000000, 1000024.9.
            f"per-cell total {totals_max:.9g} exceeds "
            f"max_counts_per_cell={max_counts_per_cell:.9g}"
            # `rtol` alongside the bound for the same reason as the per-value message above: on
            # float64 input `budget` and the cap render identically at any readable precision.
            # Excess against `max_counts_per_cell` -- the value the sentence names -- and `rtol`
            # read from the variable rather than re-derived from `budget` (codex review).
            + (f" by {excess:.3g} relative (allowing "
               f"{_SCALE_LIMIT_TOL_ULP:.0f} ULP for the lognorm round trip, rtol="
               f"{rtol:.3g}, i.e. up to {budget:.9g})"
               if input_type == "lognorm" else "")
        )


def _median_library_size(X) -> float:
    """Median per-cell total over cells whose total is ``> 0`` (the "nnz median").

    This is cell_eval2's CHOSEN rule, matching what scanpy's dense branch implements
    (``_compute_nnz_median``) -- not a rule scanpy states. Its public docs say only "median of
    total counts for observations" and never say whether zero-total observations are excluded;
    the exclusion is an implementation detail of that one branch. Its CSR branch instead takes
    ``np.median`` over ALL cells, so
    today the same control pool yields a different target depending on whether its matrix is
    CSR or dense. We pick one rule explicitly: a format-dependent normalization target is the
    same class of hidden dependence as the mem_budget-dependent one #155 is about. Reuses
    ``_row_totals`` (chunked; no full-nnz dtype upcast).
    """
    totals = _row_totals(X)
    positive = totals[totals > 0]
    if positive.size == 0:
        raise ValueError(
            "cannot resolve target_sum=None: every control cell has a total of 0, so the "
            "median library size is undefined"
        )
    return float(np.median(positive))


def resolve_target_sum(control_ad: ad.AnnData, *, input_type: str, target_sum) -> float | None:
    """Resolve ``target_sum=None`` to ONE number for the whole run (#155).

    ``target_sum=None`` means "normalize to the median library size of the matrix you were
    handed", so the two halves of one LFC ratio -- and successive batches/pieces of one run --
    can each get a different target. Normalizing to ``T`` makes every group mean ``T * f``, so
    splitting one ratio across ``T_target`` and ``T_ref`` adds ``log2(T_target / T_ref)`` to
    every log2FC. Resolving once, against the real control pool, removes that term and makes the
    result ``mem_budget``-independent by construction.

    - numeric ``target_sum`` -> returned UNCHANGED (no ``float()`` cast), with ZERO I/O, so no
      existing config object or ``config_hash`` shifts.
    - ``None`` + ``"counts"`` -> ``_median_library_size(control_ad.X)``.
    - ``None`` + ``"lognorm"`` -> ``None``. There is no library size to take a median of, and
      the IN-MEMORY consumers ignore ``target_sum`` on lognorm input (``_to_linear`` takes the
      ``expm1`` branch; ``to_normalization(lognorm -> normalized)`` likewise). The STREAMING
      consumers do NOT: ``streaming_bulk.py:128`` computes ``target_sum / libs`` and
      ``compute_de_streaming`` maps ``None`` to gpudge's ``"median"`` with no ``input_type``
      parameter at all (``de_compute.py:667``). ``scale.score_streaming`` therefore treats a
      still-``None`` target as a hard error rather than an inert one (Task 6).

    ``input_type`` must be the EFFECTIVE type of ``control_ad`` (``run._effective_input_type``),
    not the declared ``cfg.input_type``: v1 permits a declared ``lognorm`` config over data that
    is actually counts, and resolving on the declared type would return ``None`` while
    ``to_normalization`` went on deriving per-batch medians from those counts.
    """
    if target_sum is not None:
        return target_sum
    if input_type not in INPUT_TYPES:
        raise ValueError(f"input_type must be one of {INPUT_TYPES}, got {input_type!r}")
    if input_type == "lognorm":
        return None
    if control_ad.n_obs == 0:
        raise ValueError(
            "cannot resolve target_sum=None: no control cells to take a median library size "
            "over. target_sum=None normalizes to the real control pool's median (#155)."
        )
    median = _median_library_size(control_ad.X)
    if not np.isfinite(median) or median <= 0:
        raise ValueError(
            f"cannot resolve target_sum=None: the control pool's median library size is "
            f"{median!r}, which is not a usable normalization target"
        )
    return median


def to_normalization(
    adata: ad.AnnData, input_type: str, target: str, target_sum=None
) -> ad.AnnData:
    """Convert to the target normalization, raising on irrecoverable conversions."""
    if target not in NORMALIZATIONS:
        raise ValueError(f"target must be one of {NORMALIZATIONS}, got {target!r}")
    if target == "bulk_lognorm":
        raise ValueError(
            "bulk_lognorm is a PSEUDOBULK-level normalization -- log1p of the CPM of the "
            "group sum -- so it is not defined on a single cell and cannot be produced by "
            "to_normalization. Build it through prep.pseudobulk_bulk_lognorm or the "
            "accumulator paths in streaming_bulk/gpu.bulk (issue #264)."
        )
    if target == "lognorm":
        if input_type == "lognorm":
            return adata
        import scanpy as sc  # lazy: heavy import only needed for counts -> lognorm

        out = adata.copy()
        sc.pp.normalize_total(out, target_sum=target_sum)
        sc.pp.log1p(out)
        return out
    if target == "normalized":
        out = adata.copy()
        if input_type == "counts":
            import scanpy as sc  # lazy: heavy import only needed for counts -> normalized

            sc.pp.normalize_total(out, target_sum=target_sum)
        else:  # lognorm -> normalized (expm1; not necessarily equal-sum per cell)
            # `out` is already a deep copy (adata.copy()), so out.X is private: mutate it
            # in place rather than allocating a second full copy.
            if issparse(out.X):
                if not np.issubdtype(out.X.dtype, np.floating):
                    # int sparse declared lognorm: rebuild float .data so the in-place
                    # expm1 below can't fail the int cast. Share indices/indptr for csr/csc;
                    # fall back to astype for formats without them (e.g. coo/dia).
                    if hasattr(out.X, "indices") and hasattr(out.X, "indptr"):
                        out.X = out.X.__class__(
                            (out.X.data.astype(float), out.X.indices, out.X.indptr),
                            shape=out.X.shape,
                        )
                    else:
                        out.X = out.X.astype(float)
                np.expm1(out.X.data, out=out.X.data)
            else:
                Xd = np.asarray(out.X)
                if not np.issubdtype(Xd.dtype, np.floating):
                    Xd = Xd.astype(float)
                np.expm1(Xd, out=Xd)
                out.X = Xd
        return out
    # target == "counts"
    if input_type == "counts":
        return adata
    raise ValueError("cannot recover counts from lognorm input (irreversible)")
