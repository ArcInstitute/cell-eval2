"""Memory-bounded per-perturbation pseudobulk over packed ``.shad`` shards.

Produces the exact ``{normalization: (perts_sorted, means[P, G])}`` shape that
``prep.pseudobulk`` returns (sum-then-divide), so the existing anndata-metric
dispatch consumes it unchanged. One shard is resident at a time.
"""

from __future__ import annotations

import itertools

import numpy as np
import scipy.sparse as sp

from .moments import DEFAULT_BULK_TARGET_SUM, GroupMoments, _loo_bulk
from .norm import _csr_row_block  # noqa: F401  -- re-export: PR #73's tests and gpu/bulk import it here
from .prep import bulk_lognorm_means
from .stream import iter_blocks, read_reference_block, shad_metadata

_INMEM_BLOCK_ROWS = 100_000  # rows per GPU-transfer block; bounds device memory per update()


def _moment_key(norm: str) -> str:
    """bulk_lognorm is DERIVED from the counts accumulator (streaming_bulk.py:116), so its
    counts/sumsq live in the 'counts' slot. Reading sumsq['bulk_lognorm'] is a KeyError."""
    return "counts" if norm == "bulk_lognorm" else norm


def _streaming_jackknife(make_blocks, sums, counts, bulk_target_sum, *, chunk=512):
    """Second pass: Σᵢ v_ig and Σᵢ v_ig² per group, then C_p = (n-1)/n · Σ_g (s2 - s1²/n).

    ``make_blocks`` is a ZERO-ARGUMENT callable yielding ``(X, codes)`` -- the same convention
    as ``GroupedMeanAccumulator.update``, so no caller re-derives codes from labels twice and
    ``inmem_pseudobulk``, which has only ``codes[start:stop]``, can use it unchanged. It must
    return a FRESH iterator on every call; a consumed generator silently yields nothing and
    every ``C_p`` comes back 0.0, which is a valid-looking "no correction".

    ``sums`` is the first pass's raw per-group count matrix ``[P, G]`` and ``counts`` its
    per-group cell counts, so ``r_i = S_p - lib_i`` is available before this pass starts.
    Uses the SAME ``_loo_bulk`` edge policy and the same fp64 cast as the resident kernel; a
    divergence there is invisible until a real archive hits an ``r_i == 0`` group.
    """
    P_groups, G = sums.shape
    tot = sums.sum(axis=1)
    s1 = np.zeros((P_groups, G), dtype=np.float64)
    s2 = np.zeros((P_groups, G), dtype=np.float64)
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
                lib = np.asarray(blk.sum(axis=1), dtype=np.float64).ravel()
                V = _loo_bulk(sums[g], blk.toarray(), tot[g] - lib, bulk_target_sum)
                s1[g] += V.sum(axis=0)
                s2[g] += np.einsum("ij,ij->j", V, V)
    # The docstring above warns that a CONSUMED generator yields nothing and every C_p comes
    # back 0.0 -- a valid-looking "no correction". That warning was the only defense; this is
    # the tripwire (Gemini, PR #269). One int per block against an O(n*G) dense loop.
    want = int(counts.sum())
    if seen != want:
        raise ValueError(
            f"the jackknife second pass saw {seen} cells where the first pass counted {want}. "
            "make_blocks() must return a FRESH iterator over the SAME cells on every call; a "
            "consumed or re-derived one would silently return no correction at all."
        )
    n = counts.astype(np.float64)
    out = np.zeros(P_groups, dtype=np.float64)
    ok = n >= 2
    out[ok] = np.maximum(((n[ok] - 1) / n[ok])
                         * (s2[ok] - s1[ok] ** 2 / n[ok][:, None]).sum(axis=1), 0.0)
    return out


def streaming_pseudobulk(path, *, pert_col, norms, target_sum, noise=None, device="cpu",
                         with_median_umi=False, with_moments=False,
                         bulk_target_sum: float = DEFAULT_BULK_TARGET_SUM):
    """Per-perturbation pseudobulk over a packed ``.shad`` archive.

    Returns ``{norm: (perts, means)}``. Optional extras are APPENDED in a fixed order, so
    the four cases are::

        with_median_umi=False, with_moments=False -> out
        with_median_umi=True,  with_moments=False -> (out, median_umi)
        with_median_umi=False, with_moments=True  -> (out, moments)
        with_median_umi=True,  with_moments=True  -> (out, median_umi, moments)

    ``moments`` is ``{norm: GroupMoments}`` spanning ALL groups including the control
    (issue #198); it must never be restricted to a perturbation subset.
    """
    norms = list(norms)
    meta = shad_metadata(path)
    if pert_col != meta.group_by:
        # the archive is sharded BY group_by; streaming pseudobulk reads obs[pert_col] per
        # shard and maps it against perts (= unique group_by), so a mismatch silently
        # mis-maps labels. Streaming pseudobulk is only valid on the grouping column.
        raise ValueError(
            f"pert_col {pert_col!r} != archive group_by {meta.group_by!r}; streaming "
            "pseudobulk requires scoring on the archive's grouping column"
        )
    perts = np.sort(meta.perts)
    P, G = perts.size, meta.n_vars
    def make_label_blocks():
        blocks = iter_blocks(path, pert_col=pert_col)
        # iter_blocks EXCLUDES the archive's reference (control) shard, but ``perts`` (from the
        # full obs) INCLUDES the control label. Rebuild this exact prepend on every pass so
        # noise_blocks sees the same shard_idx sequence.
        ref_block = read_reference_block(path, pert_col=pert_col)
        if ref_block is not None:
            blocks = itertools.chain((ref_block,), blocks)
        if noise is not None:
            from .noise import noise_blocks

            blocks = noise_blocks(blocks, **noise)
        return blocks

    def make_blocks():
        for X, labels in make_label_blocks():
            yield X, np.searchsorted(perts, labels).astype(np.intp)

    blocks = make_label_blocks()
    # Optional: tee per-cell library sizes off the (already noise-wrapped) blocks so the
    # median UMI/cell falls out of THIS pass with no second read (Task 4). Wraps blocks
    # before the CPU/GPU branch so both paths get it.
    libs_sink: list[np.ndarray] = []
    if with_median_umi:
        blocks = _tee_library_sizes(blocks, libs_sink)

    # GPU path: feed the (already noise-wrapped) blocks to the cupy grouped-mean
    # accumulator. resolve_device("auto") picks cuda iff cupy+GPU, else cpu (this CPU
    # block). Same {norm: (perts, means)} shape; GPU means are fp32.
    from .gpu import resolve_device

    if resolve_device(device) == "cuda":
        result = _streaming_pseudobulk_gpu(blocks, perts, P, G, norms, target_sum,
                                           with_moments=with_moments,
                                           bulk_target_sum=bulk_target_sum,
                                           make_blocks=make_blocks)
    else:
        result = _streaming_pseudobulk_cpu(blocks, perts, P, G, norms, target_sum,
                                           with_moments=with_moments,
                                           bulk_target_sum=bulk_target_sum,
                                           make_blocks=make_blocks)
    out, moments = result if with_moments else (result, None)
    extras = []
    if with_median_umi:
        all_libs = np.concatenate(libs_sink) if libs_sink else np.zeros(0)
        extras.append(float(np.median(all_libs)) if all_libs.size else 0.0)
    if with_moments:
        extras.append(moments)
    return (out, *extras) if extras else out


def _tee_library_sizes(blocks, sink):
    """Yield ``(X_csr, labels)`` unchanged while appending each block's per-cell library
    sizes (row sums of the streamed X) to ``sink``, so the caller can compute the median
    UMI/cell in the same pass without a second read."""
    for X, labels in blocks:
        Xr = X.tocsr()
        sink.append(np.asarray(Xr.sum(axis=1), dtype=np.float64).ravel())
        yield Xr, labels


def _streaming_pseudobulk_cpu(blocks, perts, P, G, norms, target_sum, with_moments=False,
                              bulk_target_sum: float = DEFAULT_BULK_TARGET_SUM, make_blocks=None):
    """CPU grouped-mean over ``blocks`` (fp64 sum-then-divide). Returns ``{norm: (perts, means)}``,
    or ``({norm: (perts, means)}, {norm: GroupMoments})`` when ``with_moments``.

    One shard resident at a time; works over nnz (not n_cells*n_genes) via _scatter_rows.
    """
    # bulk_lognorm is DERIVED from the counts accumulator at finalize (issue #264): it reads
    # the group SUM, so it needs no per-cell divide, no per-cell log1p and no target_sum.
    want_bulk = "bulk_lognorm" in norms
    acc_norms = [n for n in norms if n != "bulk_lognorm"]
    if want_bulk and "counts" not in acc_norms:
        acc_norms.append("counts")
    counts = np.zeros(P, dtype=np.int64)
    acc = {n: np.zeros((P, G), dtype=np.float64) for n in acc_norms}
    # Σᵢ‖xᵢ‖² per group, per normalization (issue #198) -- one extra scatter-add over the
    # SAME nonzeros, so it is never paid unless asked for.
    sumsq = {n: np.zeros(P, dtype=np.float64) for n in acc_norms} if with_moments else None
    for X, labels in blocks:
        # perts is sorted -> searchsorted maps each label to its row index, fully vectorized (5M-safe)
        codes = np.searchsorted(perts, labels).astype(np.intp)
        Xr = X.tocsr()
        # float64 view sharing indices/indptr (only the data array is new) -- frugal cast
        Xc = Xr.__class__((Xr.data.astype(np.float64), Xr.indices, Xr.indptr), shape=Xr.shape)
        np.add.at(counts, codes, 1)
        # row index per nonzero -- identical for Xc/Xn/Xl (they share indptr); compute ONCE per
        # block, then the destination group per nonzero, reused for every normalization's scatter.
        row_of_nnz = np.repeat(np.arange(Xc.shape[0]), np.diff(Xc.indptr))
        grp_of_nnz = codes[row_of_nnz]
        if "counts" in acc:
            _scatter_rows(acc["counts"], grp_of_nnz, Xc)
            if sumsq is not None:
                np.add.at(sumsq["counts"], grp_of_nnz, Xc.data * Xc.data)
        if "normalized" in acc or "lognorm" in acc:
            libs = np.asarray(Xc.sum(axis=1)).ravel()
            libs[libs == 0] = 1.0
            # CPM row-scale by a per-nonzero factor; share indices/indptr (data-only new)
            Xn = Xc.__class__(
                (Xc.data * (target_sum / libs)[row_of_nnz], Xc.indices, Xc.indptr),
                shape=Xc.shape,
            )
            if "normalized" in acc:
                _scatter_rows(acc["normalized"], grp_of_nnz, Xn)
                if sumsq is not None:
                    np.add.at(sumsq["normalized"], grp_of_nnz, Xn.data * Xn.data)
            if "lognorm" in acc:
                # share indices/indptr; only the data array is new (log1p) -- matches prep._grouped_means
                Xl = Xn.__class__((np.log1p(Xn.data), Xn.indices, Xn.indptr), shape=Xn.shape)
                _scatter_rows(acc["lognorm"], grp_of_nnz, Xl)
                if sumsq is not None:
                    np.add.at(sumsq["lognorm"], grp_of_nnz, Xl.data * Xl.data)
    out = {}
    denom = counts.astype(np.float64)
    denom[denom == 0] = 1.0
    if want_bulk:
        out["bulk_lognorm"] = (
            perts.copy(), bulk_lognorm_means(acc["counts"], bulk_target_sum)
        )
    for n in norms:
        if n == "bulk_lognorm":
            continue
        # All three normalizations are sum-then-divide arithmetic means, matching
        # prep.pseudobulk(to_normalization(...)). For "lognorm" the per-cell values are
        # already log1p(normalized) (the accumulator holds Σ log1p), and the reference
        # pipeline (prep.pseudobulk, log_space=False on a lognorm-X) takes the *arithmetic*
        # mean of those log-values with NO trailing expm1 -- so we deliberately do not apply
        # expm1 here (doing so would diverge from compute_metrics by orders of magnitude).
        means = acc[n] / denom[:, None]
        out[n] = (perts.copy(), means)
    if not with_moments:
        return out
    jk = None
    if "bulk_lognorm" in norms:
        if make_blocks is None:
            raise ValueError("bulk_lognorm moments require a fresh second-pass block factory")
        jk = _streaming_jackknife(make_blocks, acc["counts"], counts, bulk_target_sum)
    moments = {
        n: GroupMoments(perts=perts.copy(), counts=counts.astype(np.float64),
                        sumsq=sumsq[_moment_key(n)],
                        jk=jk if n == "bulk_lognorm" else None)
        for n in norms
    }
    return out, moments


def _streaming_pseudobulk_gpu(blocks, perts, P, G, norms, target_sum, with_moments=False,
                              bulk_target_sum: float = DEFAULT_BULK_TARGET_SUM, make_blocks=None):
    """GPU grouped-mean over ``blocks`` via :class:`gpu.bulk.GroupedMeanAccumulator`.

    ``perts`` is the sorted unique group label set; each block's labels map to group codes
    by ``searchsorted`` (same convention as the CPU path). Returns the
    ``{norm: (perts, means)}`` shape with fp32 means (the accumulator finalizes to fp32),
    or ``(bulks, {norm: GroupMoments})`` when ``with_moments``.
    """
    from .gpu.bulk import GroupedMeanAccumulator

    acc = GroupedMeanAccumulator(
        P, G, normalizations=norms, target_sum=target_sum, device="cuda",
        with_moments=with_moments, bulk_target_sum=bulk_target_sum,
    )
    for X, labels in blocks:
        codes = np.searchsorted(perts, labels).astype(np.intp)
        acc.update(X.tocsr(), codes)
    if with_moments and "bulk_lognorm" in norms:
        if make_blocks is None:
            raise ValueError("bulk_lognorm moments require a fresh second-pass block factory")
        acc.jackknife(make_blocks)
    finalized = acc.finalize()  # {norm: (group_idx, means_fp32)}; group_idx aligns to perts order
    out = {n: (perts.copy(), finalized[n][1]) for n in norms}
    if not with_moments:
        return out
    raw, jk_by_norm = acc.moments(), (acc.jackknife_by_norm() or {})
    return out, {n: GroupMoments(perts=perts.copy(), counts=raw[n][0], sumsq=raw[n][1],
                                 jk=jk_by_norm.get(n))
                 for n in norms}


def _scatter_rows(dst, grp_of_nnz, Xcsr):
    """``dst[group] += each nonzero of Xcsr``, grouped -- on CSR coordinates, no dense temporary.

    Works over nnz, not ``n_cells * n_genes`` (a shard would densify to many GB).
    ``grp_of_nnz`` is the destination group for each nonzero (``codes[row_of_nnz]``);
    precomputed once per block since Xc/Xn/Xl share indptr.
    """
    np.add.at(dst, (grp_of_nnz, Xcsr.indices), Xcsr.data)


def inmem_pseudobulk(adata, *, pert_col, norms, target_sum, device="cpu",
                     block_rows=_INMEM_BLOCK_ROWS, with_moments=False,
                     bulk_target_sum: float = DEFAULT_BULK_TARGET_SUM):
    """Grouped-mean pseudobulk over row-blocks of a resident ``adata.X`` via
    :class:`gpu.bulk.GroupedMeanAccumulator`.

    Returns the same ``{normalization: (perts_sorted, means)}`` shape as
    ``prep.pseudobulk(norm.to_normalization(...))`` (sum-then-divide arithmetic means;
    ``lognorm`` = mean(log1p(CPM)), NO expm1), computed WITHOUT the full normalize
    transient -- the accumulator works over nonzeros per block. Assumes COUNTS input
    (the accumulator computes CPM from raw counts). ``target_sum`` must be numeric for the
    per-cell normalizations; a ``bulk_lognorm``-only call needs no per-cell target and
    accepts ``None`` (it scales the GROUP SUM by ``bulk_target_sum`` instead).
    ``means`` are float32 (the accumulator finalizes fp32 on cpu and cuda alike),
    matching the streaming GPU path. ``device='cuda'`` runs the accumulation on cupy.

    ``with_moments=True`` returns ``(bulks, {norm: GroupMoments})`` -- per-group counts and
    Σ‖x‖² over ALL groups including the control (issue #198).
    """
    from .gpu.bulk import GroupedMeanAccumulator

    norms = list(norms)
    # A duplicate-bearing CSR would make Σ‖x‖² wrong: squaring duplicates separately gives
    # a² + b² where the truth is (a + b)². Raise rather than canonicalize -- sum_duplicates()
    # would mutate the caller's matrix, and a defensive copy is untenable at CCL scale
    # (tens of billions of nonzeros). `has_canonical_format` is a cached flag: measured
    # 0.0000 s on a 7.9M-nnz canonical matrix, and it does not mutate (scipy 1.18).
    # Guarded by with_moments so no existing run changes behaviour. anndata writes canonical
    # CSR, so this should never fire on a file-loaded input.
    #
    # Do NOT narrow this to `X.format == "csr"`. Copilot (PR #209) suggested exactly that, on the
    # grounds that non-CSR input is converted per-block and the conversion canonicalizes. MEASURED,
    # and it is false for CSC: `csc.tocsr()` on a duplicate-bearing CSC returns a matrix that is
    # still NOT canonical -- data stayed [1., 2., 4., 5.] instead of collapsing to [3., 9.] -- so a
    # CSR-only guard would let a duplicate CSC through and silently produce the wrong Σ‖x‖².
    # The other half of that suggestion (some sparse types lack `has_canonical_format`, so this can
    # AttributeError) is true of lil/dok/dia in the abstract but UNREACHABLE here: anndata's
    # `coerce_array` rejects everything except CSR and CSC, and both expose the flag. Pinned by
    # tests/test_moments_drivers.py::test_duplicate_bearing_csc_is_rejected_not_silently_wrong.
    if with_moments and sp.issparse(adata.X) and not adata.X.has_canonical_format:
        raise ValueError(
            "per-group moments require a canonical sparse matrix: this one holds duplicate "
            "coordinates, which would make Σ‖x‖² (and expr_mse_unbiased, "
            "expr_mse_unbiased_capped, and expr_distance_unbiased) wrong. Call "
            "X.sum_duplicates() on the input first."
        )
    labels = adata.obs[pert_col].to_numpy().astype(str)
    perts = np.unique(labels)  # np.unique already returns sorted unique
    P, G = perts.size, int(adata.n_vars)
    acc = GroupedMeanAccumulator(
        P, G, normalizations=norms, target_sum=target_sum, device=device,
        with_moments=with_moments, bulk_target_sum=bulk_target_sum,
    )
    X = adata.X
    n = int(X.shape[0])
    # labels/perts are static across blocks -> map to group codes ONCE, slice per block.
    codes = np.searchsorted(perts, labels).astype(np.intp)
    # Fast per-block CSR row-slice via direct indptr/data/indices views. scipy's X[start:stop]
    # __getitem__ is pathologically slow on a many-billion-nonzero matrix: ~91-253 s PER 100k-row
    # block at 5.5M cells / 3.6e10 nnz (vs ~1 s here, structurally identical) -- it made in-memory
    # pseudobulk take HOURS at that scale. acc.update only READS Xb, so copy=False views are safe
    # and add no host memory. Non-CSR inputs keep the generic slice+convert.
    is_csr = sp.issparse(X) and X.format == "csr"
    def make_blocks():
        for start in range(0, n, block_rows):
            stop = min(start + block_rows, n)
            if is_csr:
                # Check-free slice: scipy's CSR constructor runs an O(nnz) check_format
                # validation per block. The parent is canonical + load-validated.
                Xb = _csr_row_block(X, start, stop)
            else:
                Xb = X[start:stop]
                Xb = Xb.tocsr() if sp.issparse(Xb) else sp.csr_matrix(Xb)
            yield Xb, codes[start:stop]

    for Xb, block_codes in make_blocks():
        acc.update(Xb, block_codes)
    if with_moments and "bulk_lognorm" in norms:
        acc.jackknife(make_blocks)
    finalized = acc.finalize()  # {norm: (group_idx aligned to perts order, means_fp32)}
    # Reuse the per-cell totals the accumulator already computed (libs_host) for the counts
    # scale-limit gate: stash the max on the adata (same convention as run.py's
    # _validated_inputs / _validated_scale_limits memos). None when libs weren't computed
    # (a counts-only norm set) -> the gate falls back to its own _row_totals pass.
    if acc.max_row_total is not None:
        try:
            adata._precomputed_row_total_max = float(acc.max_row_total)
        except (AttributeError, ValueError, TypeError):
            pass  # locked-down / view object -> skip the stash (gate falls back)
    out = {norm: (perts.copy(), finalized[norm][1]) for norm in norms}
    if not with_moments:
        return out
    raw, jk_by_norm = acc.moments(), (acc.jackknife_by_norm() or {})
    return out, {norm: GroupMoments(perts=perts.copy(), counts=raw[norm][0],
                                    sumsq=raw[norm][1], jk=jk_by_norm.get(norm))
                 for norm in norms}
