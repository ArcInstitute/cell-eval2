from __future__ import annotations

import anndata as ad
import numpy as np
from scipy.sparse import issparse

from .moments import GroupMoments, jackknife_correction


def _group_row_index(labels: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Group rows by label in one pass, replacing the O(P*N) per-label mask scan.

    Returns ``(uniq, order, bounds)``: ``uniq`` is the sorted unique labels [P]; group g's
    original row indices are ``order[bounds[g]:bounds[g + 1]]`` in their original (stable)
    order. The stable argsort preserves within-group row order, so a per-group ``.mean()``
    over these indices is bit-identical to the previous ``X[labels == p].mean()``.
    """
    uniq, codes = np.unique(labels, return_inverse=True)
    codes = np.ravel(codes)  # defensive across numpy return_inverse shape changes
    order = np.argsort(codes, kind="stable")
    counts = np.bincount(codes, minlength=uniq.size)
    bounds = np.concatenate(([0], np.cumsum(counts))).astype(np.intp)
    return uniq, order, bounds


def _grouped_means(X, order, bounds, n_groups, *, log_space: bool = False) -> np.ndarray:
    """Per-group column means over the rows grouped by ``order``/``bounds``.

    Bit-identical to looping ``X[group_rows].mean(axis=0)`` per group (dense or sparse);
    ``log_space=True`` computes the geometric mean ``expm1(mean(log1p(X)))``. Groups by
    per-group integer fancy-index (no full-matrix reorder copy -- memory-frugal at VCC
    scale). Returns a [n_groups, n_genes] float64 matrix.
    """
    sparse = issparse(X)
    out = np.zeros((n_groups, X.shape[1]), dtype=np.float64)
    for g in range(n_groups):
        rows = order[bounds[g]:bounds[g + 1]]
        if rows.size == 0:
            continue
        sub = X[rows]
        if log_space:
            if sparse:
                if sub.dtype == np.float64:
                    # X[rows] is a fresh copy, so log1p in place on its data is safe.
                    np.log1p(sub.data, out=sub.data)
                    out[g] = np.expm1(np.asarray(sub.mean(axis=0)).ravel())
                elif hasattr(sub, "indices") and hasattr(sub, "indptr"):
                    # csr/csc: share indices/indptr, copy+cast only .data (log1p mutates it).
                    m = sub.__class__((sub.data.astype(np.float64), sub.indices, sub.indptr),
                                      shape=sub.shape)
                    np.log1p(m.data, out=m.data)
                    out[g] = np.expm1(np.asarray(m.mean(axis=0)).ravel())
                else:  # other sparse formats (e.g. coo): astype copies the whole structure
                    m = sub.astype(np.float64)
                    np.log1p(m.data, out=m.data)
                    out[g] = np.expm1(np.asarray(m.mean(axis=0)).ravel())
            else:
                out[g] = np.expm1(np.log1p(np.asarray(sub, dtype=np.float64)).mean(axis=0))
        elif sparse:
            if sub.dtype == np.float64:
                out[g] = np.asarray(sub.mean(axis=0)).ravel()  # already float64: no copy
            elif hasattr(sub, "indices") and hasattr(sub, "indptr"):
                m = sub.__class__((sub.data.astype(np.float64), sub.indices, sub.indptr),
                                  shape=sub.shape)
                out[g] = np.asarray(m.mean(axis=0)).ravel()
            else:  # other sparse formats (e.g. coo) lack indices/indptr
                out[g] = np.asarray(sub.astype(np.float64).mean(axis=0)).ravel()
        else:
            out[g] = np.asarray(sub, dtype=np.float64).mean(axis=0)
    return out


def _grouped_sums(X, order, bounds, n_groups) -> np.ndarray:
    """Per-group column sums of raw counts over the rows grouped by ``order``/``bounds``.

    Equal to looping ``X[group_rows].sum(axis=0)`` per group (dense or sparse) EXCEPT that a
    floating dtype genuinely COARSER than float64 is widened BEFORE the reduction -- see below;
    no divide. Groups by per-group integer fancy-index (no full-matrix reorder copy --
    memory-frugal at VCC scale), mirroring ``_grouped_means``'s ``order``/``bounds`` contract
    (both come from ``_group_row_index``). Returns a [n_groups, n_genes] float64 matrix. Used by
    the ``bulk_lognorm`` comparator (#264) and by the deseq2 backend's pseudobulk (summed raw
    counts per replicate).

    **REDUCE WIDE -- ONE policy, every caller (issue #271, fixed 2026-08-18).** This used to
    reduce in whatever dtype ``X`` carried and cast only the RESULT, which made an fp32 matrix
    disagree with itself and with the other drivers:

    * ``moments.jackknife_correction`` casts ``.data`` to fp64 before summing, so the
      ``bulk_lognorm`` bulk and the correction subtracted from it came from different ``P_p``.
      ⚠️ That ASYMMETRY was the defect, not either half: the jackknife already reduced wide.
    * ``streaming_bulk._streaming_pseudobulk_cpu`` accumulates into an fp64 array from fp64-cast
      data and ``gpu.bulk.GroupedMeanAccumulator`` does the same, so this -- the RESIDENT path --
      was the only reduction left in the input dtype. MEASURED on the ``2**24`` fixture: the
      resident and streaming bulks diverged by 3.53e-10 on fp32 input and agreed exactly on
      fp64. Widening CONVERGES them onto the answer two of three drivers already gave; it does
      not invent a convention.

    ⚠️ **"The drivers agree" is a statement about a COARSE FLOAT, not a general one**, and three
    limits are worth stating rather than being read into it:

    * the GPU accumulator still downcasts its FINAL means to fp32 for the host artifact
      (``gpu.bulk.finalize``) -- the eval store's precision, pre-existing, and separate from the
      reduction dtype;
    * summation ORDER still differs between drivers (``cell_source`` documents this), so
      "bit-identical" holds on the fixtures measured here, not by construction;
    * **LONGDOUBLE input still diverges between drivers where longdouble is WIDER than float64**
      (x86-64 Linux; on a platform where the two are the same type there is nothing to diverge),
      deliberately. The guard below leaves it native, so the resident path sums in longdouble and
      rounds once while the jackknife, the CPU streamer and the GPU accumulator each round to fp64
      first. MEASURED on the ``u = 2**-53``
      fixture: resident 1.0000000000000002 against 1.0. That is the price of "reduce wide" rather
      than "cast to fp64" -- fp64 is the narrower side there -- and closing it means picking ONE
      supported accumulator width across every driver, which is its own decision, not a rider here.

    ⚠️ **A MASKED matrix is a second residual, and the asymmetry #271 is about survives it.** This
    function preserves a mask (see the widen branch below); ``jackknife_correction`` does
    ``csr_matrix(X)`` on dense input, which STRIPS it. MEASURED on masked float32
    ``[[10, 1], [--, 1]]`` hiding 900: the bulk reduces ``[10, 2]`` while the jackknife's own group
    sum is ``[910, 2]``. Pre-existing -- main split the same way -- out of scope for a dtype fix,
    and characterized in
    ``tests/test_jackknife.py::test_a_MASKED_matrix_STILL_splits_the_two_halves``.

    The policy is deliberately "reduce as wide as the input, minimum fp64", NOT "cast everything
    to fp64": for some dtypes fp64 is the NARROWER side (see the widen guard below).

    ⚠️ **Both callers of the comparator move together**, so ``pseudobulk_bulk_lognorm`` and
    ``pseudobulk_bulk_lognorm_with_moments`` stay bit-identical by construction
    (``tests/test_jackknife.py::test_inmem_reference_bulk_is_BIT_IDENTICAL_to_the_non_moments_function``);
    that test rejects widening ONE of them, which is a different change from widening the
    shared reduction.

    ⚠️ **deseq2 moves too, and that is the ruling, not a side effect (Alex 2026-08-18).**
    ``deseq2_de._pseudobulk`` is the third caller. It was left narrow by #264 PR2 because "whether
    deseq2's numbers move must be deliberate"; the decision taken here is that it moves with one
    shared policy rather than carrying a ``widen=`` flag a future caller can forget -- see
    ``deseq2_de._pseudobulk`` for the three reasons and the one-line shape of the other choice.

    ### Where a floating reduction stops being exact

    For non-negative INTEGER counts held in a float, nothing moves below the dtype's
    consecutive-integer limit, and that is a guarantee rather than a measurement: every partial
    sum of non-negative integers is non-decreasing and bounded by the total, so if the total is
    representable every partial sum is, and the whole reduction is exact.

    ⚠️ That limit is ``2**(nmant + 1)``, inclusive, and it is the DTYPE's number, not a universal:
    **2,048 for float16**, **16,777,216 = ``2**24`` for float32**, ``2**53`` for float64. Write the
    boundary as the dtype's, never as a constant. Just past it an fp32 group sum loses 1 count:
    MEASURED on ``[[16777216, 1], [1, 1]]`` the narrow reduction returns 16,777,216 against the
    exact 16,777,217, and the bulk moves **3.53e-10** at the shipped ``bulk_target_sum = 5e4``.
    ⚠️ The figure quoted on the issue and in several docstrings is 6.35e-09, which is the SAME
    fixture at ``TS = 1e6`` -- the default #268 retired on 2026-08-11. Verified both:
    3.531662e-10 at 5e4, 6.348612e-09 at 1e6. The gap scales with TS because ``log1p``'s knee
    moves with it, so quote a bulk delta WITH its target sum. A 400-cell group at ~1e5 counts/gene is
    already there (sum ~4e7); the same group at 1e3 and 1e4 counts/gene is exact. Reachable
    because validation bounds each CELL (``max_counts_per_cell``, integrality, non-negativity)
    and never the accumulated per-gene GROUP total -- see the exposure routes below.

    **FRACTIONAL input has no such guarantee at all: it CAN round from the first addition**, far
    below any boundary. (Not "always" -- exactly representable fractions like 0.5 + 0.5 reduce
    identically; what fractionality removes is the guarantee, not the possibility of exactness.)
    MEASURED at 2.7e-06 relative on a 4,000-cell group whose sums are ~6.2e3, i.e. 2,700x INSIDE
    fp32's limit.

    ### What this change moved, stated plainly

    ⚠️ **What was MEASURED to move is the fractional baseline arms -- but the set of inputs that CAN
    move is larger**, and the exposure routes below enumerate it: an integer-valued float whose group
    sums cross its own dtype's boundary moves too (fp32 above ``2**24``, float16 above 2,048). "What
    moved" below means "what was measured on the shipped artifacts", not "the only thing that can".

    The stored baseline arms are fractional, so they are in that second regime:
    ``real_bundle._baseline_leg`` sets ``allow_fractional_counts=True`` because a baseline is a
    MEAN, and ``baseline.py`` emits ``.astype(np.float32)``. MEASURED on the three official
    competition contexts' ``context_mean`` arms **as stored** -- the archives the official bundles
    were built from, read back rather than rebuilt -- 138,400 cells x 18,533 genes, 301 groups,
    **95.3-95.6% of stored values fractional**::

        context   max group sum   x inside 2**24   max |d sum|   max |d bulk|
        A             2,430,157            6.90x      0.202881       5.58e-06
        B             2,917,567            5.75x      0.264648       5.65e-06
        C             2,207,197            7.60x      0.068359       5.73e-06

    Every one of them 5.7-7.6x INSIDE fp32's boundary, which the integer argument above never
    covered. (``max |d bulk|`` is over all 18,533 genes; a bundle build applies its own CPM gene
    filter first, which drops columns and so shifts each row total slightly. The group-sum column
    is filter-independent for any gene that survives.) Rebuilding the arm through the library's own
    ``_profile_from_adata``/``_prediction_from_adata`` instead of reading the stored archive reads
    smaller -- 0.0809/0.1246/0.0884 counts and 3.9-4.5e-06 -- so quote the stored figures for what
    the artifacts cost and the rebuild figures only for that path.

    That measurement is why an earlier implementation of this fix was REVERTED (``ee0e6c9``): it
    invalidates the baseline leg of the three official ``#276`` val bundles. Those bundles were
    since orphaned for an unrelated reason -- they carry ``rule_digest`` ``992cc849...`` while the
    runtime rule has moved, and ``score.py`` raises on that mismatch -- so the cost is sunk rather
    than pending. **Any stored artifact whose values were MEASURED to move must be regenerated**;
    the cache/identity terms that enforce that are deliberately coarser than the measurement, and
    the two must not be confused. `run._GROUPED_SUM_REDUCTION_SEMANTICS` invalidates by PATH -- any
    `bulk_lognorm` or deseq2 run -- because a key is computed before any value is read; but what
    actually MOVED is fractional coarse-float input. An integer-count artifact below the dtype's
    boundary is bit-identical across this change (measured: the three official real arms, max|delta|
    exactly 0) and is merely recomputed once.

    ⚠️ **The REAL arm of those same three contexts does NOT move -- max|delta| is 0.000000 on all
    three, group sums bit-identical.** Measured the same way, on the same archives: real counts are
    0.0000 fractional against the baseline arm's 0.953-0.956, and their largest group sums are the
    same 2.2-2.9e6. So the split is clean -- an INTEGER-count arm below the boundary is untouched,
    and it is specifically the FRACTIONAL baseline leg that moves. That is also why the earlier
    certification of this fix passed: it was run against an integer-count stand-in for the baseline.

    ⚠️ **``_grouped_means`` does NOT follow this policy, and that is out of #271's scope.** It
    widens unconditionally (cast, then reduce), so it carries the three inversions the guard below
    exists to avoid. MEASURED on the fixtures the tests use: a masked fp32 ``[[10, 1], [--, 1]]``
    whose hidden value is 900 gives a mean of ``[455, 1]`` -- the mask stripped by ``np.asarray``
    and the hidden cell averaged back in, 45x off -- where ``_grouped_sums`` returns ``[10, 2]``;
    int64 ``[[2**53 + 1, 7], [1, 0]]`` gives a mean of ``4503599627370496`` against the exact
    ``...497``; and longdouble input is downcast. Left alone deliberately: it drives ``pseudobulk``,
    ``lognorm`` and the geometric-mean path, and its fp64 sum-then-divide ORDER is what the v1
    parity gate pins, so that is its own change with its own blast radius -- not a rider on this
    one.

    ### Exposure: which VALID submissions this moves

    Under ``vcc2026``'s own gates AS THEY STOOD WHEN #271 LANDED, by three routes, none of which
    #271 closed -- they are the inputs whose numbers differ across it. Route 2 has since been
    closed by a different change; the tense in each entry says which is which:

    1. **The preset caps per-CELL totals, not GROUP totals.** ``max_counts_per_cell = 1e6`` over
       500 cells per perturbation lets a per-gene group sum legally reach 5e8, 30x past fp32's
       ``2**24``. MEASURED: 17 cells of ``[999999, 1]`` pass both ``validate_input_type`` and
       ``check_scale_limit(1e6)``, sum to 16,999,983, and their bulk moves 2.8e-09.
    2. **``validate_input_type`` did not actually require integers -- the VALUE-SCALED half of
       this was closed 2026-08-18.**
       ``norm._is_all_integer`` compared with ``np.allclose`` at its default ``rtol=1e-5``, so
       1000.001 passed as counts with ``allow_fractional_counts=False``, and such a matrix diverges
       well below the boundary. MEASURED: 4,000 fp32 values of 1000.001 sum to 4,000,000.25 narrow
       against 4,000,003.90625 wide -- a gap of 3.66. ``norm._INT_ATOL`` now makes that tolerance
       ABSOLUTE (``rtol=0, atol=1e-6``) and the gate refuses THAT matrix. The arithmetic is unchanged
       and the route stays listed, because it is still reachable three ways: with
       ``allow_fractional_counts=True``, which every tiled baseline leg sets; with a deviation inside
       1e-6, which the gate accepts by design; and through ``partition_inmem.score_piece`` or the
       direct shard drivers, which never call ``validate_input_type`` at all.
    3. **A NARROW dtype moves far below fp32's boundary.** float16 counts are exact only to 2,048
       and nothing rejects them -- ``validate_input_type`` checks sign and integrality, not width.
       MEASURED: a dense float16 ``[[2048, 1], [1, 1]]`` passes both gates, sums to 2,049 wide
       against 2,048 narrow, and its bulk moves **4.8e-04** -- five orders larger than route 1,
       from a group sum four orders smaller.

    ⚠️ No route made the OLD behaviour safe. For exactly those submissions the resident and
    streaming drivers disagreed with each other, so the score already depended on which driver
    ran it. This picks the answer the other two drivers were already computing rather than
    preserving a number that was never single-valued.

    ⚠️ Integer, bool and complex input is NOT widened and reduces in numpy's own accumulator for
    that dtype, exactly as before. Integers are exact there up to the accumulator's RANGE and then
    WRAP rather than saturate -- an int64 group summing past ``INT64_MAX`` yields a negative total
    and a zero bulk. That is pre-existing and unreachable under any sane ``max_counts_per_cell``,
    but "exact for anything the dtype holds" is false, and the final cast to float64 is still lossy
    above ``2**53`` for integer sums. Complex input likewise reduces natively and then has its
    imaginary part discarded by the float64 cast -- NOT silently: numpy emits
    ``ComplexWarning: Casting complex values to real discards the imaginary part``, and this repo's
    suite runs with warnings visible. ⚠️ It does not RAISE either, which a hosted reviewer asserted
    on PR #330: MEASURED on numpy 2.5.2, ``_grouped_sums`` on complex64
    ``[[3+4j, 1], [1+2j, 1]]`` returns ``[4., 2.]`` with that warning, and bare
    ``np.asarray(1+2j, dtype=np.float64)`` returns ``1.0`` the same way. So the exposure is a
    warning a caller can ignore, not an exception -- closing it belongs in
    ``norm.validate_input_type``, not here."""
    out = np.zeros((n_groups, X.shape[1]), dtype=np.float64)
    # ⚠️ Widen ONLY a floating dtype that is genuinely COARSER than float64. Three ways to get this
    # wrong, all three of which shipped for one commit each on the reverted branch (codex review
    # rounds 3, 4 and 5) and are pinned by tests:
    #
    #   * casting an INTEGER matrix to fp64 first is a REGRESSION. numpy reduces integers in a wide
    #     integer accumulator; fp64 stops representing consecutive integers above 2**53. MEASURED
    #     on int64 `[[2**53 + 1, 7], [1, 0]]`: reduce-then-cast gives the exact 2**53 + 2,
    #     cast-then-reduce gives 2**53, and the bulk moves 1.8e-15.
    #   * `dtype != np.float64` also catches LONGDOUBLE, which it then DOWNCASTS -- the same
    #     inversion in the other direction. MEASURED on longdouble `[[1 + u, 6u], [u, 0]]` with
    #     u = 2**-53: reduce-then-cast gives nextafter(1, inf), cast-then-reduce gives 1.0.
    #     It also copies a big-endian float64 (`>f8 != np.float64`) for nothing.
    #   * `np.asarray` on the dense branch STRIPS an ndarray subclass, and AnnData accepts a masked
    #     X -- see the `asanyarray` comment below.
    #
    # Comparing eps says what is actually meant -- "this dtype cannot hold what float64 can" --
    # and gets all four cases right by construction.
    widen = (np.issubdtype(X.dtype, np.floating)
             and float(np.finfo(X.dtype).eps) > float(np.finfo(np.float64).eps))
    for g in range(n_groups):
        rows = order[bounds[g]:bounds[g + 1]]
        if rows.size == 0:
            continue
        sub = X[rows]
        if widen:
            # Same frugal shapes as `_grouped_means`: csr/csc share indices/indptr and copy only
            # `.data`; anything else (dense, or a sparse format without them) goes through astype.
            # ⚠️ The sharing is already what this spelling gets -- `copy=False` is the scipy
            # constructor's DEFAULT for the `(data, indices, indptr)` form, VERIFIED with
            # `np.shares_memory` on scipy 1.18 for csr_matrix, csr_array, csc_matrix and csc_array
            # (all True; `copy=True` is what breaks it). Passing `copy=False` explicitly would be a
            # no-op -- a hosted reviewer suggested it as an optimization on PR #330, and the
            # optimization is the code below.
            #
            # ⚠️ `issparse(sub)`, NOT a hoisted `issparse(X)`, and the reason is now FRUGALITY
            # rather than correctness. An anndata BACKED `_CSRDataset` is not itself sparse but its
            # `X[rows]` IS a csr_matrix; deciding from `X` would send it to the generic branch
            # below, which handles it correctly (`csr_matrix.astype` exists) but copies indices and
            # indptr as well as the data. An earlier draft used `np.asanyarray` there, where the
            # same hoist RAISED "setting an array element with a sequence" -- that crash is gone
            # with the cast, so what is left is one wasted copy per group on backed input.
            # (`_grouped_means` hoists AND uses `np.asarray`, so it still raises that ValueError
            # for backed input -- measured on main. Out of scope; see the note above.)
            if issparse(sub) and hasattr(sub, "indices") and hasattr(sub, "indptr"):
                sub = sub.__class__((sub.data.astype(np.float64), sub.indices, sub.indptr),
                                    shape=sub.shape)
            else:
                # `sub.astype`, never `np.asarray(sub, dtype=...)`. `asarray` strips an ndarray
                # SUBCLASS, and AnnData accepts a masked X: on a masked float32 fixture
                # `[[10, 1], [--, 1]]` whose hidden value is 900, `asarray` sums the masked cell
                # back in -- [910, 2] against the masked [10, 2], moving the bulk by 4.32 (codex
                # review round 5). The object's own `astype` preserves what it is: a MaskedArray
                # keeps its mask, an `np.matrix` its orientation, any other duck array its type --
                # and a lazy array stays lazy, where `asanyarray` would materialize the whole
                # cells-by-genes group before reducing it. float64 input never reaches here at all
                # (`widen` is False), which is why only a coarse dtype was ever affected.
                sub = sub.astype(np.float64)
        out[g] = np.asarray(sub.sum(axis=0), dtype=np.float64).ravel()
    return out


def bulk_lognorm_means(sums: np.ndarray, bulk_target_sum: float) -> np.ndarray:
    """``log1p(TS * P / sum_g P)`` per group, for a ``[P, G]`` matrix of COUNT SUMS.

    Issue #264. Unlike ``lognorm`` -- ``mean_c log1p(CPM(cell))`` -- this normalizes the SUM
    and takes ``log1p`` once, so ``sum_g expm1(row) == TS`` exactly and a point-mass
    prediction is representable. The LEGACY ``lognorm`` comparator pins every point mass to
    a shell no real group is on, which is what makes THAT one a dispersion functional
    (spec 1); this function is the replacement, not the thing being described.

    An all-zero group returns zeros rather than raising: it has no composition to express,
    and every driver already tolerates empty groups.
    """
    sums = np.asarray(sums, dtype=np.float64)
    if sums.ndim != 2:
        raise ValueError(f"sums must be [P, G]; got {sums.shape}")
    if not np.isfinite(bulk_target_sum) or bulk_target_sum <= 0:
        raise ValueError(f"bulk_target_sum must be positive and finite, got {bulk_target_sum!r}")
    totals = sums.sum(axis=1)
    scale = np.zeros_like(totals)
    ok = totals > 0
    scale[ok] = bulk_target_sum / totals[ok]
    return np.log1p(sums * scale[:, None])


def pseudobulk_bulk_lognorm(adata, pert_col: str, *, bulk_target_sum: float):
    """``(perts, means)`` in the ``bulk_lognorm`` space, from a COUNTS ``adata``.

    The fp64 CPU reference the accumulator paths are checked against (Task 4).
    """
    labels = adata.obs[pert_col].to_numpy().astype(str)
    perts, order, bounds = _group_row_index(labels)
    sums = _grouped_sums(adata.X, order, bounds, perts.size)
    return perts, bulk_lognorm_means(sums, bulk_target_sum)


def pseudobulk_bulk_lognorm_with_moments(adata, pert_col: str, *, bulk_target_sum: float):
    """``(perts, means, GroupMoments)`` in the ``bulk_lognorm`` space, from a COUNTS ``adata``.

    The RESIDENT reference every driver is asserted against (spec §6 test 7): cells are resident,
    so the second pass is local and there is no streaming subtlety to get wrong.

    ``means`` is BIT-IDENTICAL to ``pseudobulk_bulk_lognorm``'s -- same ``_grouped_sums``, same
    ``bulk_lognorm_means`` -- which is the invariant that matters.

    ⚠️ **The two halves of the metric see ONE ``P_p`` (issue #271, fixed 2026-08-18).**
    ``_grouped_sums`` now reduces WIDE -- a floating dtype coarser than float64 is widened before
    the reduction -- and ``jackknife_correction`` has always cast ``.data`` to fp64 before
    reducing, so for any ORDINARY UNMASKED input -- integer, or a float of any width up to fp64 --
    the group mean and its leave-one-out correction are built from the same group sums, whatever dtype
    the caller stored that matrix in. ⚠️ That is the removal of a systematic NARROW-versus-WIDE
    mismatch for one input reduced two ways, NOT redistribution invariance and not order
    independence: floating addition is not associative, so two different cell-level layouts with the
    same mathematical total can still reduce to different values (in fp64, and in fp32 after
    widening, ``[1e16, 1, 1]`` and ``[1e16, 2, 0]`` differ by one fp64 ULP). Two
    residuals remain, both documented on ``_grouped_sums``: longdouble input (the guard leaves it
    native, so the resident path keeps precision the jackknife rounds away) and a MASKED matrix
    (this side honours the mask, ``jackknife_correction``'s ``csr_matrix(X)`` strips it). Before the fix they were not, on fp32 input, in two
    regimes: INTEGER counts diverged by 1 count once a per-gene group sum crossed ``2**24`` (a
    400-cell group at ~1e5 counts/gene; bulk 3.53e-10 at ``TS = 5e4``, 6.35e-09 at the retired
    1e6), and FRACTIONAL input -- which the stored
    baseline arms are, a baseline being a mean emitted ``.astype(np.float32)`` -- can round from the
    first addition, diverging up to 0.265 counts (bulk 5.7e-06) on the three official contexts as
    STORED, 5.7-7.6x INSIDE that boundary. ``_grouped_sums``' docstring carries the full measurement, the
    exposure routes, and what the change moved.

    They can still differ on INTEGER input above ``2**53``, where ``_grouped_sums`` reduces
    natively (exactly) and the jackknife casts to fp64 first. That is out of reach under any
    realistic per-cell cap, is the pre-existing behaviour rather than something #271 introduced,
    and is the reason the policy is "reduce wide" rather than "cast everything to fp64".

    ``counts``/``sumsq`` stay in COUNTS space. Neither is read when the comparator is
    ``bulk_lognorm`` -- ``correction_for`` reads ``jk`` -- but the npz artifact stores them, so
    they must mean something definite rather than whatever the last writer left.

    ``codes`` is scattered back onto ORIGINAL row positions, so it is NOT sorted; the kernel's
    argsort/searchsorted pair is what makes that safe (tested in Task 1).
    """
    labels = adata.obs[pert_col].to_numpy().astype(str)
    perts, order, bounds = _group_row_index(labels)
    n_groups = perts.size
    codes = np.empty(adata.n_obs, dtype=np.intp)
    for p in range(n_groups):
        codes[order[bounds[p]:bounds[p + 1]]] = p
    return perts, bulk_lognorm_means(_grouped_sums(adata.X, order, bounds, n_groups),
                                     bulk_target_sum), GroupMoments(
        perts=perts, counts=np.diff(bounds).astype(np.float64),
        sumsq=_grouped_sumsq(adata.X, order, bounds, n_groups),
        jk=jackknife_correction(adata.X, codes, n_groups, bulk_target_sum))


def _grouped_sumsq(X, order, bounds, n_groups) -> np.ndarray:
    """Per-group ``Σᵢ ‖xᵢ‖²`` over the rows grouped by ``order``/``bounds``.

    Sums the squares of every entry of the group's rows. On sparse input this runs over
    nonzeros only -- exact, because a zero contributes 0 to the sum of squares.

    ``sum_duplicates()`` is REQUIRED, not defensive. A CSR matrix may legally hold duplicate
    coordinates, and ``tocsr()`` on an already-CSR matrix returns it UNCHANGED -- duplicates
    intact. ``.mean()`` sums duplicates and is therefore correct, but squaring them separately
    gives a^2 + b^2 where the truth is (a + b)^2. Measured on a one-row matrix holding
    (1.0, 2.0) at the same coordinate: mean 3.0 (right), naive sum-of-squares 14.0, correct 18.0.
    ``X[rows]`` fancy-indexing already returns a fresh copy, so the in-place canonicalization
    cannot touch the caller's matrix.

    Returns [n_groups] float64.
    """
    sparse = issparse(X)
    out = np.zeros(n_groups, dtype=np.float64)
    for g in range(n_groups):
        rows = order[bounds[g]:bounds[g + 1]]
        if rows.size == 0:
            continue
        sub = X[rows]
        if sparse:
            sub = sub.tocsr()
            sub.sum_duplicates()          # see the docstring -- correctness, not hygiene
            d = np.asarray(sub.data, dtype=np.float64)
        else:
            d = np.asarray(sub, dtype=np.float64).ravel()
        out[g] = float(np.dot(d, d))
    return out


def pseudobulk_with_moments(
    adata: ad.AnnData, pert_col: str
) -> tuple[np.ndarray, np.ndarray, GroupMoments]:
    """``pseudobulk`` plus the per-group second moments (issue #198).

    Returns ``(perts, means, moments)``. ``perts``/``means`` are bit-identical to
    ``pseudobulk(adata, pert_col)`` -- the same ``_group_row_index`` and ``_grouped_means``
    run underneath. ``moments`` spans ALL groups, control included, and must never be
    restricted (see ``moments.GroupMoments``).

    The squares are taken in whatever space ``adata.X`` is already in, matching the means:
    for a lognorm-normalized input that is ``log1p(CPM)``.
    """
    labels = adata.obs[pert_col].to_numpy().astype(str)
    perts, order, bounds = _group_row_index(labels)
    means = _grouped_means(adata.X, order, bounds, perts.size, log_space=False)
    counts = np.diff(bounds).astype(np.float64)
    sumsq = _grouped_sumsq(adata.X, order, bounds, perts.size)
    return perts, means, GroupMoments(perts=perts, counts=counts, sumsq=sumsq)


def pseudobulk(adata: ad.AnnData, pert_col: str) -> tuple[np.ndarray, np.ndarray]:
    """Per-perturbation mean expression.

    Returns (perts, means) where `perts` is the sorted unique perturbation labels
    (shape [P]) and `means` is the per-perturbation mean matrix (shape [P, n_genes]).
    """
    labels = adata.obs[pert_col].to_numpy().astype(str)
    perts, order, bounds = _group_row_index(labels)
    means = _grouped_means(adata.X, order, bounds, perts.size, log_space=False)
    return perts, means


def delta(
    means: np.ndarray, perts: np.ndarray, control: str
) -> tuple[np.ndarray, np.ndarray]:
    """Signed per-perturbation effect: each non-control perturbation's mean minus
    the control mean (located by the `control` label in `perts`).

    Args:
        means: per-perturbation mean matrix [P, n_features] (e.g. from pseudobulk).
        perts: perturbation labels aligned to `means` rows [P].
        control: the control perturbation label; must appear in `perts`.

    Returns:
        (perts_nc, effects): the non-control labels (input order preserved) and the
        float64 effect matrix [P-1, n_features] = means[non-control] - control mean.
    """
    perts = np.asarray(perts).astype(str)
    means = np.asarray(means, dtype=np.float64)
    if means.shape[0] != perts.shape[0]:
        raise ValueError(
            f"delta: means has {means.shape[0]} rows but perts has {perts.shape[0]} "
            "labels; rows must align 1:1 with perturbation labels"
        )
    hits = np.flatnonzero(perts == control)
    if hits.size == 0:
        raise ValueError(
            f"control perturbation {control!r} not found in perts: {perts.tolist()}"
        )
    ctrl = means[hits[0]]
    mask = perts != control
    return perts[mask], means[mask] - ctrl
