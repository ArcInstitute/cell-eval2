"""GPU (cupy) grouped-mean pseudobulk accumulator — mirrors ``streaming_bulk``.

:class:`GroupedMeanAccumulator` streams cell blocks (CSR) and accumulates, per group
(perturbation), the Σ for each requested normalization plus per-group cell counts; then
:meth:`finalize` divides to arithmetic means. The values match
``streaming_bulk.streaming_pseudobulk`` (sum-then-divide):

* ``counts``     — mean of raw counts
* ``normalized`` — mean of CPM (``row * target_sum / libsize``)
* ``lognorm``    — mean of ``log1p(CPM)`` — **NO trailing expm1**. The lognorm
  accumulator holds ``Σ log1p(CPM)``; the reference pipeline takes the *arithmetic* mean
  of those log-values directly (an expm1 here would diverge by orders of magnitude).

``device="cuda"`` accumulates on cupy (``xp.add.at`` over CSR coordinates, fp64
accumulators cast to fp32 at finalize); ``device="cpu"`` runs the identical accumulation
on numpy. The work is over nonzeros (CSR coordinates), never a dense ``n_cells*n_genes``.
"""

from __future__ import annotations

import numpy as np

from . import resolve_device, xp_for
from ..moments import DEFAULT_BULK_TARGET_SUM

_ALLOWED = ("counts", "normalized", "lognorm", "bulk_lognorm")

# cupy 14.1.1 fails any single pinned (page-locked) host allocation > 2 GiB (2**31 B) with
# cudaErrorInvalidValue (issue #133). ``update`` pins each block's ``Xr.data`` AND ``Xr.indices``
# per H2D transfer, so a block whose ``nnz * itemsize`` exceeds that ceiling crashes. We cap the
# pinned footprint per transfer by sub-chunking a block's rows; the grouped-mean accumulation is
# row-additive, so contiguous-row sub-chunks are arithmetically exact.
_PINNED_H2D_BUDGET_BYTES = 1 << 30  # 1 GiB, comfortably under the 2 GiB ceiling


def _add_at(xp, dst, idx, val):
    """``dst[idx] += val`` with duplicate accumulation, on numpy or cupy.

    ``idx`` may be an index array (1-D ``dst``) or a tuple of index arrays (N-D ``dst``).
    ``cupy.add.at`` mirrors ``numpy.add.at`` (and supersedes the deprecated
    ``cupyx.scatter_add``), so a single ``xp.add.at`` covers both backends.
    """
    xp.add.at(dst, idx, val)


def _bincount(xp, idx, *, minlength, weights=None):
    """This module's two ``xp.bincount`` calls, given numpy's empty-input answer on cupy too (#162).

    NOT a general ``bincount`` compatibility shim: it is internal to the two call sites below
    and its empty branch does no validation, so ``idx`` must be a 1-D integral array on
    ``xp``, ``weights`` (when given) a matching 1-D array, and ``minlength`` a non-negative
    ``int``. An invalid argument that numpy would reject can return zeros here instead.

    On an EMPTY ``idx`` numpy returns ``zeros(minlength)``; **cupy raises**
    ``ValueError: zero-size array to reduction operation CUPY_CUB_MAX which has no
    identity`` — its implementation takes ``idx.max()`` to size the output *before*
    ``minlength`` is applied, so passing ``minlength`` rescues neither call form (measured
    on cupy 14.1.1 / H100, weighted and unweighted alike). A real 2025 VCC submission hit
    this: an entirely empty (``nnz == 0``) prediction matrix makes ``row_of_nnz`` empty at
    the weighted site, so the GPU run aborted with a CUB internal error while the same input
    scored fine on CPU. CI is CPU-only, so it cannot run the cupy failure itself; what it can
    run is the behaviour, which ``test_gpu_bulk.py`` pins with a numpy that raises the same
    way (issue #183).

    Returning ``zeros(minlength)`` keeps the length both call sites already assume, so the
    ``if libs.size`` guard and the ``libs == 0 -> 1.0`` floor below stay meaningful rather
    than becoming dead. The weighted empty case returns fp64 — the dtype every *non-empty*
    weighted call gives; numpy's own *empty* weighted call returns int64 instead, an
    inconsistency in numpy rather than a contract worth reproducing. The choice reaches
    ``libs`` and its floored copy but changes no result: with ``nnz == 0`` and cells present
    ``libs`` is a length-``n_cells`` zero vector either way, ``float(libs.max())`` is ``0.0``
    either way, both floor to numeric ones, and the division converges to the same fp64
    quotient — which is then indexed by an empty ``row_of_nnz``, so no value survives into
    ``data_cpm``.
    """
    if idx.size == 0:
        dtype = xp.float64 if weights is not None else xp.intp
        return xp.zeros(minlength, dtype=dtype)
    if weights is None:
        return xp.bincount(idx, minlength=minlength)
    return xp.bincount(idx, weights=weights, minlength=minlength)


class GroupedMeanAccumulator:
    """Streaming per-group column means for one or more normalizations, on a ``device``."""

    def __init__(self, n_groups, n_genes, *, normalizations, target_sum, device="auto",
                 with_moments=False, bulk_target_sum: float = DEFAULT_BULK_TARGET_SUM):
        norms = list(normalizations)
        bad = [n for n in norms if n not in _ALLOWED]
        if bad:
            raise ValueError(f"unknown normalization(s) {bad}; choose from {_ALLOWED}")
        self.n_groups = int(n_groups)
        self.n_genes = int(n_genes)
        self.norms = norms
        needs_ts = any(n in ("normalized", "lognorm") for n in norms)
        if needs_ts and target_sum is None:
            raise ValueError(
                f"normalization(s) {sorted(set(norms) & {'normalized', 'lognorm'})} require a "
                "numeric target_sum; got None"
            )
        self.target_sum = float(target_sum) if target_sum is not None else None
        # Validated here as well as in prep.bulk_lognorm_means: this class is constructible
        # directly, outside EvalConfig's validation, and a non-finite or non-positive target
        # would produce silent NaN/inf bulks rather than an error (Copilot, #265).
        if "bulk_lognorm" in norms and (
                isinstance(bulk_target_sum, bool)
                or not np.isfinite(bulk_target_sum)
                or bulk_target_sum <= 0):
            raise ValueError(
                f"bulk_target_sum must be a positive finite float, got {bulk_target_sum!r}"
            )
        self.bulk_target_sum = float(bulk_target_sum)
        self.device = resolve_device(device)
        xp = xp_for(self.device)
        self._xp = xp
        acc_norms = [n for n in norms if n != "bulk_lognorm"]
        if "bulk_lognorm" in norms and "counts" not in acc_norms:
            acc_norms.append("counts")
        # fp64 accumulators persist on the device across update() calls (shards).
        self._acc = {
            n: xp.zeros((self.n_groups, self.n_genes), dtype=xp.float64)
            for n in acc_norms
        }
        self._counts = xp.zeros(self.n_groups, dtype=xp.float64)
        # Σᵢ‖xᵢ‖² per group, per normalization (issue #198). None unless requested: it is one
        # extra scatter-add over EVERY nonzero per block, so it is never paid unasked.
        self._sumsq = (
            {n: xp.zeros(self.n_groups, dtype=xp.float64) for n in acc_norms}
            if with_moments else None
        )
        self._jk_by_norm = None
        self._max_row_total = None  # max raw per-cell total; set only when libs are computed

    def update(self, x_csr, group_idx):
        """Accumulate a block: ``x_csr`` [n_cells, n_genes] CSR, ``group_idx`` [n_cells] in [0, n_groups).

        Sub-chunks the block's rows so no single pinned H2D transfer exceeds
        ``_PINNED_H2D_BUDGET_BYTES`` (cupy's 2 GiB single-alloc ceiling, issue #133). The
        grouped-mean accumulation is row-additive, so contiguous-row sub-chunks are exact.
        """
        Xr = x_csr.tocsr()
        if Xr.shape[1] != self.n_genes:
            raise ValueError(f"x_csr has {Xr.shape[1]} genes; expected {self.n_genes}")
        group_idx = np.asarray(group_idx)
        if group_idx.shape[0] != Xr.shape[0]:
            raise ValueError(f"group_idx length {group_idx.shape[0]} != n_cells {Xr.shape[0]}")
        # Slice only when a block must actually be split. A full-span single chunk (the common,
        # under-budget case) passes Xr straight through -- no copy, identical to the pre-fix path.
        # When splitting is needed, use the check-free CSR row-block VIEW (_csr_row_block), NOT
        # scipy's Xr[start:stop] __getitem__: that deep-copies data/indices and is pathologically
        # slow on large CSR (streaming_bulk Lever 2B: ~91-253 s vs ~1 s per 100k-row block), and
        # would also negate inmem_pseudobulk's view-based row-blocking. (#133 review.)
        from ..streaming_bulk import _csr_row_block
        n_cells = Xr.shape[0]
        for start, stop in self._row_chunks(Xr):
            if start == 0 and stop == n_cells:
                self._accumulate_block(Xr, group_idx)
            else:
                self._accumulate_block(_csr_row_block(Xr, start, stop), group_idx[start:stop])

    def _row_chunks(self, Xr):
        """Yield contiguous ``(start, stop)`` row ranges whose pinned H2D footprint
        (``nnz * max(data.itemsize, indices.itemsize)``) stays within the byte budget. A lone
        row that already exceeds the budget is emitted alone (never split); in practice a single
        row is ``n_genes * itemsize`` bytes, far under the budget, so this only bites in tests."""
        n_cells = Xr.shape[0]
        if n_cells == 0:
            return
        indptr = Xr.indptr
        itemsize = max(Xr.data.itemsize, Xr.indices.itemsize)
        max_nnz = max(1, _PINNED_H2D_BUDGET_BYTES // itemsize)
        start = 0
        while start < n_cells:
            stop = int(np.searchsorted(indptr, int(indptr[start]) + max_nnz, side="right")) - 1
            if stop <= start:
                stop = start + 1  # a single row always proceeds (cannot be split)
            yield start, stop
            start = stop

    def _accumulate_block(self, x_csr, group_idx):
        """Accumulate one already-byte-bounded row sub-block into the persistent accumulators."""
        xp = self._xp
        Xr = x_csr
        n_cells = Xr.shape[0]
        group = xp.asarray(group_idx).astype(xp.intp)  # cast on device (cheaper than host-side)

        # per-cell counts (every cell, regardless of nonzeros). `_bincount` (not `xp.bincount`)
        # because cupy raises on an empty `group` -- a zero-CELL block. `_row_chunks` guarantees
        # `stop > start`, so this site is not reachable through `update()`; the guard is here
        # because `_accumulate_block` itself has no `n_cells == 0` early return and the sibling
        # site below IS reachable (issue #162).
        self._counts += _bincount(xp, group, minlength=self.n_groups).astype(xp.float64)

        # Transfer in native precision (fp32 data, int32 indices) then cast on device: halves
        # the PCIe volume and offloads the cast. fp32->fp64 and int32->int64 are exact.
        data = xp.asarray(Xr.data).astype(xp.float64)
        indices = xp.asarray(Xr.indices).astype(xp.intp)
        indptr = xp.asarray(Xr.indptr).astype(xp.intp)
        # destination group for each nonzero (shared by every normalization's scatter)
        row_of_nnz = xp.repeat(xp.arange(n_cells), xp.diff(indptr))
        grp_of_nnz = group[row_of_nnz]

        if "counts" in self._acc:
            _add_at(xp, self._acc["counts"], (grp_of_nnz, indices), data)
            if self._sumsq is not None:
                _add_at(xp, self._sumsq["counts"], grp_of_nnz, data * data)
        if "normalized" in self._acc or "lognorm" in self._acc:
            # Per-cell library size via a row-isolated segment-sum over the already-transferred
            # `data` (bincount weighted by the per-nonzero row index) -- computed on the device,
            # so no host scipy Xr.sum(axis=1) and no libs H2D copy back. Lever 2A: that host row-sum
            # was the single largest phase of pred_pseudobulk at scale (~49%, 87 s over 56 blocks
            # on the 36e9-nnz CCL_2 pred); a weighted bincount over the resident `data` is ~11x
            # faster on an H100 (measured), and the feared GPU atomic contention does not
            # materialize for the sorted, high-repetition row indices here. bincount sums each
            # row's nonzeros into its OWN bin, so for integer counts (an fp64 sum of ints < 2^53 is
            # order-independent) it is byte-identical to scipy Xr.sum(axis=1, dtype=float64), and,
            # unlike a cumsum-diff prefix sum, it keeps a NaN/inf confined to its own row -- the
            # per-row semantics the CPM/lognorm scatter and the scale-limit gate depend on.
            # `_bincount`, not `xp.bincount`: with `nnz == 0` (an all-zero input block, which a
            # real submission produced) `row_of_nnz` is empty and cupy raises before `libs` is
            # even bound, so the `if libs.size` guard below could never run (issue #162).
            libs = _bincount(xp, row_of_nnz, weights=data, minlength=n_cells)
            if libs.size:
                # Reuse these per-cell totals for the scale-limit gate (run._check_scale_limit_once):
                # track the running max, taken BEFORE the ==0->1.0 floor. float(libs.max()) pulls a
                # scalar off-device; numpy and cupy max both propagate NaN. np.maximum propagates NaN
                # in BOTH operand orders, so this stays byte-identical to the gate's
                # float(np.max(_row_totals(X))) path regardless of block order (Lever 1 contract),
                # even for a NaN-bearing allow_fractional submission.
                block_max = float(libs.max())
                self._max_row_total = (
                    block_max if self._max_row_total is None
                    else float(np.maximum(block_max, self._max_row_total))
                )
            # Floor empty rows to 1.0 branchlessly: xp.where is a fused elementwise select on the
            # device (no boolean-mask setitem, which on cupy syncs + allocs mask-index arrays).
            # Byte-identical -- leaves nonzero and NaN entries untouched (NaN == 0 is False).
            libs = xp.where(libs == 0, libs.dtype.type(1.0), libs)
            data_cpm = data * (self.target_sum / libs)[row_of_nnz]
            if "normalized" in self._acc:
                _add_at(xp, self._acc["normalized"], (grp_of_nnz, indices), data_cpm)
                if self._sumsq is not None:
                    _add_at(xp, self._sumsq["normalized"], grp_of_nnz, data_cpm * data_cpm)
            if "lognorm" in self._acc:
                # Σ log1p(CPM); finalize takes the arithmetic mean — NO expm1.
                data_log = xp.log1p(data_cpm)
                _add_at(xp, self._acc["lognorm"], (grp_of_nnz, indices), data_log)
                if self._sumsq is not None:
                    _add_at(xp, self._sumsq["lognorm"], grp_of_nnz, data_log * data_log)

    def finalize(self):
        """``{norm: (group_idx[n_groups], means[n_groups, n_genes] fp32 host array)}``."""
        denom = self._counts.copy()
        denom[denom == 0] = 1.0  # empty group -> mean 0, never NaN
        idx = np.arange(self.n_groups)
        out = {}
        for n in self.norms:
            if n == "bulk_lognorm":
                sums = self._acc["counts"]
                totals = sums.sum(axis=1)
                scale = self._xp.zeros_like(totals)
                ok = totals > 0
                scale[ok] = self.bulk_target_sum / totals[ok]
                # Transform on-device so the [P, G] matrix crosses to host only once.
                means = self._xp.log1p(sums * scale[:, None])
            else:
                means = self._acc[n] / denom[:, None]  # NO expm1 on lognorm
            host = means.get() if hasattr(means, "get") else np.asarray(means)
            out[n] = (idx, host.astype(np.float32))
        return out

    def moments(self):
        """``{norm: (counts[n_groups], sumsq[n_groups])}`` as fp64 HOST arrays, or ``None``
        when the accumulator was built with ``with_moments=False``.

        Deliberately separate from :meth:`finalize`, whose ``{norm: (idx, means)}`` shape is
        positionally indexed by ``streaming_bulk._streaming_pseudobulk_gpu`` and
        ``inmem_pseudobulk`` -- widening it would break both. ``counts`` is the SAME vector
        for every normalization (cells per group is a property of the grouping, not the
        space); it is copied per entry so a caller cannot alias it.
        """
        if self._sumsq is None:
            return None

        from ..streaming_bulk import _moment_key

        def _host(a):
            return np.asarray(a.get() if hasattr(a, "get") else a, dtype=np.float64)

        counts = _host(self._counts)
        return {n: (counts.copy(), _host(self._sumsq[_moment_key(n)])) for n in self.norms}

    def jackknife(self, make_blocks, *, chunk=512):
        """Run the bulk-lognorm delete-one second pass over fresh ``(X, codes)`` blocks."""
        if "bulk_lognorm" not in self.norms:
            self._jk_by_norm = {}
            return
        if self._sumsq is None:
            raise ValueError("bulk_lognorm jackknife requires with_moments=True")

        from ..moments import _loo_bulk

        xp = self._xp
        sums = self._acc["counts"]
        totals = sums.sum(axis=1)
        s1 = xp.zeros_like(sums)
        s2 = xp.zeros_like(sums)
        seen = 0
        for X, codes in make_blocks():
            codes = np.asarray(codes, dtype=np.intp)
            seen += int(codes.size)
            Xr = X.tocsr()
            Xd = Xr.__class__((Xr.data.astype(np.float64), Xr.indices, Xr.indptr),
                              shape=Xr.shape)
            for g in np.unique(codes):
                rows = np.flatnonzero(codes == g)
                for s in range(0, rows.size, chunk):
                    blk = Xd[rows[s:s + chunk]]
                    Y = xp.asarray(blk.toarray()).astype(xp.float64)
                    lib = Y.sum(axis=1)
                    V = _loo_bulk(sums[g], Y, totals[g] - lib,
                                  self.bulk_target_sum, xp=xp)
                    s1[g] += V.sum(axis=0)
                    s2[g] += xp.einsum("ij,ij->j", V, V)
        # Same tripwire as `streaming_bulk._streaming_jackknife` (Gemini, PR #269): a
        # consumed or re-derived block factory yields nothing and every C_p comes back 0.0,
        # a valid-looking "no correction". ⚠️ `_counts` is a DEVICE array on the cuda path
        # (`xp.zeros` at :86), so `np.asarray` on it raises rather than converting -- go
        # through the same `.get()`-or-`asarray` hop this method already uses for `out`
        # below (`_host` at :242 is local to `moments()`). One sync, once, after the pass.
        _c = self._counts
        want = int((_c.get() if hasattr(_c, "get") else np.asarray(_c)).sum())
        if seen != want:
            raise ValueError(
                f"the jackknife second pass saw {seen} cells where the first pass counted "
                f"{want}. make_blocks() must return a FRESH iterator over the SAME cells on "
                "every call; a consumed or re-derived one would silently return no "
                "correction at all."
            )
        n = self._counts.astype(xp.float64)
        out = xp.zeros(self.n_groups, dtype=xp.float64)
        ok = n >= 2
        out[ok] = xp.maximum(
            ((n[ok] - 1) / n[ok])
            * (s2[ok] - s1[ok] ** 2 / n[ok][:, None]).sum(axis=1),
            xp.float64(0.0),
        )
        host = out.get() if hasattr(out, "get") else np.asarray(out)
        self._jk_by_norm = {"bulk_lognorm": np.asarray(host, dtype=np.float64)}

    def jackknife_by_norm(self):
        """``{norm: jk[n_groups] fp64}`` or ``None`` when no second pass was run.

        Deliberately NOT folded into :meth:`moments`, whose 2-tuple is positionally consumed at
        four call sites (two in ``streaming_bulk``, two in ``test_moments_drivers``) and
        element-wise zipped at a fifth. Same reasoning that kept ``moments()`` out of
        ``finalize()``.
        """
        return self._jk_by_norm

    @property
    def max_row_total(self):
        """Max raw per-cell total (row-sum) seen across update()s, or None if libs were
        never computed (a counts-only accumulator). For a counts input this equals
        max(X.sum(axis=1)); the ``libs==0 -> 1.0`` tweak runs after this and cannot change
        the max. Lets the scale-limit gate reuse work instead of a second _row_totals pass."""
        return self._max_row_total
