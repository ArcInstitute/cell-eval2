"""GPU (cupy) discrimination ranking — the CPU full-matrix path with ``xp`` for numpy.

This mirrors :func:`cell_eval2.distances.pairwise_full` /
:func:`cell_eval2.distances.correct_excluded_gene` and the ranking stage of
:func:`cell_eval2.metrics.discrimination.discrimination_score`, but with the heavy
linear algebra expressed through an array module ``xp`` so the *same* kernel runs on
numpy (CPU, for testing) or cupy (CUDA).

Public entry point :func:`discrimination_ranks` resolves the ``device`` knob:

* ``cpu`` delegates to the CPU reference (:func:`discrimination_score`) — exact parity,
  zero duplication.
* ``cuda`` runs :func:`_discrimination_ranks_xp` on cupy.

The kernel streams pred effects in ``pert_chunk`` rows, so peak memory is one
``[pert_chunk, n_real]`` distance block (plus, for l1, a gene-tiled
``[pert_chunk, n_real, gene_tile]`` temporary — l1 has no matmul identity).

Ties: resolved by ``tie_policy``, the SAME rule the CPU reference applies, so the two
paths agree on a tied row rather than each inheriting its own sort's ordering.

⚠️ This module previously claimed "pseudobulk effect values are continuous floats, so
exact ties don't occur and the GPU/CPU tie-ordering cannot diverge". **That premise is
false** (issue #282): under ``cosine`` a zero-norm predicted effect -- a submission
pasting the reference control cells for a target -- makes every distance in that row
exactly ``1.0``. Since cupy's sort is not numpy's introsort, the old
``argsort(argsort(...))`` could have resolved such a row differently on each device and
returned different per-target scores for one input.

Neither policy resolves a tie by a device-specific sort now, but by **two different
mechanisms**, and the distinction matters when reading the code:

* ``"midrank"`` is computed **arithmetically** (comparisons + reductions), so no sort
  ordering enters the answer at all and the kernel stays fully on device.
* ``"position"`` is inherently a sort ordering, so it is computed **on the host with
  numpy** even under cupy -- upstream cell-eval is numpy, and that is what this branch
  exists to reproduce. It costs a ``[B, n_real]`` transfer on the legacy path.

⚠️ **This buys identical tie ORDERING for a bit-identical distance block. It does not
make the whole result device-independent, and nothing here could.** The distance block
itself is still computed with ``xp``, and fp32-GPU vs fp64-CPU pseudobulk already differs
upstream of this function -- which is why ``_result_config_digest`` puts the DEVICE in the
result-cache key. Rounding can therefore still decide whether two *mathematically* equal
distances compare equal, and the rank follows. What is fixed is that once the block is
fixed, the ranking of it is not a function of which sort implementation ran. The #282 case
itself is exact on both devices: a zero-norm operand is assigned distance ``1.0`` by
construction, not by arithmetic that could round.

⚠️ Neither policy has been measured on a real GPU against the old behaviour; the
divergence the old code permitted is a property of the algorithms, not an observation.
"""

from __future__ import annotations

import numpy as np

from ..distances import (
    panel_reduced,
    resolve_exclusion_columns,
    resolve_panel_columns,
)
from ..prep import delta
from . import resolve_device, xp_for

# Memory budget for the l1 gene-tiled temporary [pert_chunk, n_real, gene_tile]. l1 has
# no matmul collapse, so the per-element |pred - real| must be materialized in tiles.
_L1_TILE_BYTES = 2 * 1024**3  # 2 GiB


def discrimination_ranks(
    real_bulk,
    pred_bulk,
    *,
    genes,
    metric,
    exclude_target_gene,
    exclusion_scope,
    rank_denominator,
    tie_policy,
    pert_chunk,
    device,
    control,
    control_source,
    perts=None,
    target_gene_map=None,
):
    """Per-perturbation discrimination scores, computed on ``device``.

    Returns the same ``{pert: score}`` mapping as the ranking stage of
    :func:`cell_eval2.metrics.discrimination.discrimination_score`. ``real_bulk`` /
    ``pred_bulk`` are ``(labels, means)`` tuples (as ``prep.pseudobulk`` returns).
    ``perts`` is reserved: when given, it is validated against the non-control labels
    that ``delta`` derives from ``real_bulk`` (otherwise it is derived).

    ``target_gene_map`` is the ``{perturbation: gene}`` exclusion override (issue #248);
    it must reach BOTH the cpu delegation and the xp kernel, since this is the path that
    runs by default on a CUDA box.
    """
    resolved = resolve_device(device)
    if resolved == "cpu":
        # Exact CPU reference — no duplicated kernel for the fallback path.
        from ..metrics.discrimination import discrimination_score

        return discrimination_score(
            pred_bulk=pred_bulk,
            real_bulk=real_bulk,
            control=control,
            distance=metric,
            rank_denominator=rank_denominator,
            tie_policy=tie_policy,
            exclude_target_gene=exclude_target_gene,
            exclusion_scope=exclusion_scope,
            control_source=control_source,
            genes=genes,
            target_gene_map=target_gene_map,
        )
    # Free cupy's caching pool before the discrimination cuBLAS ops (einsum/matmul allocate a cuBLAS
    # handle + workspace via cudaMalloc, OUTSIDE cupy's pool): a preceding in-memory pseudobulk can
    # leave VRAM parked in the pool -> CUBLAS_STATUS_ALLOC_FAILED at CCL_2 scale. No-op without a
    # GPU; never raises (F10.1, same pool-residue family as the gpudge boundaries).
    from . import _release_gpu_pool  # sibling gpu module (no heavy de_compute import — Gemini #119)
    _release_gpu_pool()
    return _discrimination_ranks_xp(
        xp_for(resolved),
        real_bulk,
        pred_bulk,
        genes=genes,
        metric=metric,
        exclude_target_gene=exclude_target_gene,
        exclusion_scope=exclusion_scope,
        rank_denominator=rank_denominator,
        tie_policy=tie_policy,
        pert_chunk=pert_chunk,
        control=control,
        control_source=control_source,
        perts=perts,
        target_gene_map=target_gene_map,
    )


def _discrimination_ranks_xp(
    xp,
    real_bulk,
    pred_bulk,
    *,
    genes,
    metric,
    exclude_target_gene,
    exclusion_scope,
    rank_denominator,
    tie_policy,
    pert_chunk,
    control,
    control_source,
    perts=None,
    target_gene_map=None,
):
    """xp-generic discrimination ranking (``xp`` = numpy or cupy).

    Computes the delta() effects and validation on the host (numpy), moves the effects
    to ``xp``, then streams pred rows in ``pert_chunk`` blocks: per block it builds the
    ``[B, n_real]`` distance matrix and ranks each row. Mirrors ``discrimination_score``
    exactly; the parity tests pin it.

    ⚠️ Where the drop-gene correction happens depends on ``exclusion_scope`` (#343). Under
    ``"row"`` it is applied per row inside the chunk loop, as it always was. Under ``"panel"``
    the feature space is reduced ONCE on the host, before ``real_eff`` is transferred, and no
    per-row correction runs at all -- ``do_exclude`` is cleared for exactly that reason, so
    the chunk loop below sees a narrower matrix and is otherwise unchanged.
    """
    if rank_denominator not in ("n", "n-1"):
        raise ValueError(f"rank_denominator must be 'n' or 'n-1', got {rank_denominator!r}")
    if tie_policy not in ("midrank", "position"):
        raise ValueError(f"tie_policy must be 'midrank' or 'position', got {tie_policy!r}")
    # Mirrors the CPU guard. Without it any value but "panel" falls into the legacy `elif`
    # below and returns a plausible ROW-scope score for a typo -- the one failure mode this
    # kernel must not have, since it is the branch that runs by default on a CUDA box.
    if exclusion_scope not in ("row", "panel"):
        raise ValueError(f"exclusion_scope must be 'row' or 'panel', got {exclusion_scope!r}")
    if control_source not in ("pred", "real"):
        raise ValueError(f"control_source must be 'pred' or 'real', got {control_source!r}")
    if exclude_target_gene and genes is None:
        raise ValueError(
            "genes (the var index) is required when exclude_target_gene=True"
        )

    real_perts = np.asarray(real_bulk[0]).astype(str)
    pred_perts = np.asarray(pred_bulk[0]).astype(str)
    real_means = np.asarray(real_bulk[1], dtype=np.float64)
    pred_means = np.asarray(pred_bulk[1], dtype=np.float64)
    if real_means.shape[0] != real_perts.shape[0] or pred_means.shape[0] != pred_perts.shape[0]:
        raise ValueError(
            "bulk perts/means row mismatch: "
            f"real ({real_perts.shape[0]} perts vs {real_means.shape[0]} rows), "
            f"pred ({pred_perts.shape[0]} perts vs {pred_means.shape[0]} rows)"
        )
    if real_means.shape[1] != pred_means.shape[1]:
        raise ValueError(
            f"pred/real feature dimension mismatch: pred {pred_means.shape[1]} != "
            f"real {real_means.shape[1]}"
        )

    # Real-side effects (always against the real control); pred-side per control_source.
    perts_arg = perts
    perts, real_eff = delta(real_means, real_perts, control)
    if control_source == "pred":
        pred_keys, pred_eff = delta(pred_means, pred_perts, control)
    else:  # "real": measure predicted effect against the real control
        ctrl_hits = np.flatnonzero(real_perts == control)
        if ctrl_hits.size == 0:
            raise ValueError(f"control {control!r} not found in real perturbations")
        real_ctrl = real_means[ctrl_hits[0]]
        mask = pred_perts != control
        pred_keys, pred_eff = pred_perts[mask], pred_means[mask] - real_ctrl

    # Generalize: pred keys must be a SUBSET of real keys (equality is the whole-prediction
    # case). Each pred effect is ranked against ALL real effects; the denominator is the
    # full real perturbation count. Mirrors discrimination_score's subset handling exactly.
    real_keys = np.asarray(perts).astype(str)
    pred_keys = np.asarray(pred_keys).astype(str)
    real_pos = {p: j for j, p in enumerate(real_keys)}
    missing = [p for p in pred_keys if p not in real_pos]
    if missing:
        raise ValueError(
            "pred non-control perturbations must be a subset of real; "
            f"labels missing from real: {missing[:10]}"
        )
    if perts_arg is not None and not np.array_equal(
        np.asarray(perts_arg).astype(str), real_keys
    ):
        raise ValueError(
            "supplied `perts` does not match the non-control labels derived from real_bulk"
        )

    do_exclude = exclude_target_gene and genes is not None
    exclusion_cols: dict[int, int] = {}
    if genes is not None:
        genes = np.asarray(genes).astype(str)
        if exclude_target_gene and genes.shape[0] != real_eff.shape[1]:
            raise ValueError(
                f"genes length ({genes.shape[0]}) does not match the pseudobulk feature "
                f"dimension ({real_eff.shape[1]}); cannot exclude target genes"
            )
        if do_exclude and exclusion_scope == "panel":
            # issue #343, the GPU twin of `metrics.discrimination`'s panel branch. The
            # reduction happens HOST-side, before `real_eff` is transferred, so the chunked
            # kernel below runs unchanged on a narrower matrix and no per-row correction is
            # applied at all -- `do_exclude` is cleared for exactly that reason. Columns come
            # from `real_keys` (the whole panel) so a sharded `pred_keys` is scored in the
            # same feature space as the whole, matching the CPU path row for row.
            panel_cols = resolve_panel_columns(real_keys, genes,
                                               target_gene_map=target_gene_map)
            pred_eff = panel_reduced(pred_eff, panel_cols)
            real_eff = panel_reduced(real_eff, panel_cols)
            do_exclude = False
        elif do_exclude:
            if np.unique(genes).size != genes.shape[0]:
                raise ValueError(
                    "duplicate gene names in `genes` are not supported when "
                    "exclude_target_gene=True: the full-matrix path needs a unique "
                    "gene->column mapping (deduplicate the var index, e.g. "
                    "AnnData.var_names_make_unique())"
                )
            # SAME resolver as the CPU path (issue #248): construct-ID labels resolve
            # through target_gene_map, and zero resolution raises instead of scoring
            # with nothing excluded. This branch is what runs by default on a CUDA box,
            # so a lookup written independently here is exactly how the bug survived.
            # gate_labels=real_keys for the same reason as the CPU path: pred_keys may be a
            # shard, real_keys is always the whole panel, and the raise is about the panel.
            exclusion_cols = resolve_exclusion_columns(
                pred_keys, genes, target_gene_map=target_gene_map,
                gate_labels=real_keys,
            )

    n = real_keys.size                          # denominator basis (D = n or n-1)
    D = n if rank_denominator == "n" else n - 1
    n_pred = pred_keys.size                      # loop/stream basis — pred rows only

    real_x = xp.asarray(real_eff)  # [n_real, G] resident on device
    n_real, n_genes = real_x.shape
    # l2/cosine reuse the real row-norms^2; l1 does not need them.
    real_norm_squares = None
    if metric in ("l2", "euclidean", "cosine"):
        real_norm_squares = xp.einsum("jg,jg->j", real_x, real_x)

    # Each pred row's matching real column, precomputed once on the host and transferred
    # in a single H2D copy (never built per-row inside the loop below).
    match_cols = np.asarray([real_pos[p] for p in pred_keys], dtype=np.int64)
    match_cols_xp = xp.asarray(match_cols)

    out: dict[str, float] = {}
    for start in range(0, n_pred, pert_chunk):
        stop = min(start + pert_chunk, n_pred)
        pc = xp.asarray(pred_eff[start:stop])  # [B, G]
        block, sim, pred_norm_squares = _distance_block(
            xp, pc, real_x, real_norm_squares, metric, n_real, n_genes
        )
        # Gather (local row, gene col) for the perts whose target gene is in the panel, then
        # correct all of them in ONE vectorized pass (no per-row Python loop / kernel launches).
        rows = []
        cols = []
        for li in range(stop - start):
            col = exclusion_cols.get(start + li)  # keyed by GLOBAL pred row index
            if col is not None:
                rows.append(li)
                cols.append(col)
        if rows:
            _correct_chunk(
                xp, block, pc, real_x, metric, xp.asarray(rows), xp.asarray(cols),
                sim=sim, pred_norm_squares=pred_norm_squares,
                real_norm_squares=real_norm_squares,
            )
        block_match = match_cols_xp[start:stop]  # each row's own real column (subset-safe)
        self_ranks = _match_ranks_xp(xp, block, block_match, tie_policy)
        host_ranks = self_ranks.get() if hasattr(self_ranks, "get") else np.asarray(self_ranks)
        for li in range(stop - start):
            # float(), NOT int(): a mid-rank is half-integral on an even tied block, and
            # int() would truncate 12.5 -> 12 and silently reintroduce a rank bias (#282).
            rank = float(host_ranks[li])
            out[str(pred_keys[start + li])] = 1.0 if D <= 0 else 1.0 - rank / D
    return out


def _match_ranks_xp(xp, block, match_cols, tie_policy):
    """0-based rank of each row's own real column within that row, vectorized over a block.

    The xp twin of :func:`cell_eval2.metrics.discrimination._match_rank`; the two must
    agree element-for-element, which ``test_gpu_distances`` pins on tied rows.

    ``"midrank"`` is ``n_less + (n_equal - 1) / 2`` -- computed arithmetically, so no sort
    ordering enters the result and cupy's sort cannot diverge from numpy's introsort on a
    tied row (issue #282). It also replaces the double ``argsort`` with two comparisons +
    two reductions, dropping the per-block cost from O(n log n) to O(n).

    ``"position"`` keeps the legacy ``argsort(argsort(...))`` for v1 parity, but runs it on
    the HOST with numpy -- see the branch comment and the module docstring's scope note.
    """
    if tie_policy == "position":
        # Ranked on the HOST with numpy, deliberately, even when `xp` is cupy. This branch
        # exists only to reproduce upstream cell-eval, and upstream is numpy: cupy's sort is
        # a different algorithm and need not resolve a tied row the same way, so ranking on
        # device would make the legacy answer DEVICE-DEPENDENT -- the exact divergence the
        # old "exact ties don't occur" premise excused. The transfer costs a [B, n_real]
        # copy on a path that is not the competition path; correctness over throughput.
        host_block = block.get() if hasattr(block, "get") else np.asarray(block)
        host_match = (match_cols.get() if hasattr(match_cols, "get")
                      else np.asarray(match_cols))
        order = np.argsort(np.argsort(host_block, axis=1), axis=1)
        return order[np.arange(host_block.shape[0]), host_match]
    b = xp.arange(block.shape[0])
    match = block[b, match_cols][:, None]                      # [B, 1]
    n_less = (block < match).sum(axis=1)
    n_equal = (block == match).sum(axis=1)
    # A NaN match distance is equal to nothing (n_equal == 0) and less than nothing, so it
    # would read rank -0.5. NaN sorts LAST, so it belongs in the trailing NaN block: n_less
    # is the finite count and n_equal the NaN count. Mirrors the CPU branch exactly.
    #
    # Computed UNCONDITIONALLY, never under `if bool(nan_match.any())` (Copilot + Gemini,
    # both independently, on the #282 PR). Converting a cupy array to a Python bool forces
    # a device->host transfer and synchronizes the pipeline on EVERY block, including the
    # overwhelmingly common one with no NaN at all -- which would have made this branch's
    # "stays on device" claim false in the one place it matters. The branchless form costs
    # one extra O(B x n_real) pass over a block already scanned twice, on a path that is
    # defensive rather than hot (a NaN distance needs a NaN pseudobulk), and it keeps ONE
    # code path for both array modules rather than a numpy-fast/cupy-slow split -- a split
    # is what let the CPU and GPU tie orderings diverge in the first place.
    n_nan = xp.isnan(block).sum(axis=1)
    nan_match = xp.isnan(match[:, 0])
    n_less = xp.where(nan_match, block.shape[1] - n_nan, n_less)
    n_equal = xp.where(nan_match, n_nan, n_equal)
    return n_less + (n_equal - 1) / 2.0


def _distance_block(xp, pc, real_x, real_norm_squares, metric, n_real, n_genes):
    """``[B, n_real]`` distances from each pred row in ``pc`` to every ``real_x`` row.

    Returns ``(block, sim, pred_norm_squares)``; ``sim`` / ``pred_norm_squares`` are the
    cosine drop-gene reuse parts (None for l1/l2).
    """
    sim = pred_norm_squares = None
    if metric in ("l2", "euclidean"):
        pns = xp.einsum("ig,ig->i", pc, pc)
        d2 = pns[:, None] + real_norm_squares[None, :] - 2.0 * (pc @ real_x.T)
        block = xp.sqrt(xp.maximum(d2, 0.0))
    elif metric == "cosine":
        sim = pc @ real_x.T
        pred_norm_squares = xp.einsum("ig,ig->i", pc, pc)
        block = _cosine_from_parts(xp, sim, pred_norm_squares, real_norm_squares)
    elif metric in ("l1", "manhattan", "cityblock"):
        B = pc.shape[0]
        gene_tile = min(max(1, int(_L1_TILE_BYTES // (8 * max(B * n_real, 1)))), n_genes)
        block = xp.zeros((B, n_real), dtype=xp.float64)
        # Reuse one [B, n_real, gene_tile] buffer across tiles instead of allocating a fresh
        # broadcast temp per iteration (zero per-iter alloc churn in the cupy memory pool).
        temp = xp.empty((B, n_real, gene_tile), dtype=xp.float64)
        for s in range(0, n_genes, gene_tile):
            e = min(s + gene_tile, n_genes)
            out_view = temp[:, :, : e - s]  # last tile may be narrower than gene_tile
            xp.subtract(pc[:, None, s:e], real_x[None, :, s:e], out=out_view)
            xp.abs(out_view, out=out_view)
            block += out_view.sum(axis=2)
    else:
        raise ValueError(f"unsupported distance metric {metric!r}; use l1, l2, or cosine")
    return block, sim, pred_norm_squares


def _cosine_from_parts(xp, sim, pred_norm_squares, real_norm_squares):
    """xp mirror of :func:`cell_eval2.distances.cosine_distance_from_parts`."""
    denom = xp.sqrt(pred_norm_squares)[:, None] * xp.sqrt(real_norm_squares)[None, :]
    # xp.where (not boolean-mask assignment) keeps this fully element-wise -> no device sync
    # or mask-index allocation on GPU. denom==0 (zero-norm operand) -> sim 0 -> distance 1.
    denom_safe = xp.where(denom != 0, denom, 1.0)
    out = xp.where(denom != 0, sim / denom_safe, 0.0)
    return xp.clip(1.0 - out, 0.0, 2.0)


def _correct_chunk(
    xp, block, pc, real_x, metric, rows, cols, *,
    sim, pred_norm_squares, real_norm_squares,
):
    """In-place vectorized drop-gene correction for all chunk rows with a target gene in panel.

    ``rows`` (local pred-row indices) and ``cols`` (each row's dropped gene column) are 1-D xp
    index arrays of equal length V. Corrects ``block[rows]`` to the distance with each row's
    gene removed — the xp mirror of :func:`cell_eval2.distances.correct_excluded_gene` applied
    to V rows at once (one set of kernels instead of V), keeping each row independent so it
    equals the per-row result exactly.
    """
    a = pc[rows, cols]            # (V,)        pred value at each row's dropped gene
    b = real_x[:, cols].T         # (V, n_real) real values at each row's dropped gene
    if metric in ("l1", "manhattan", "cityblock"):
        block[rows] = xp.maximum(block[rows] - xp.abs(b - a[:, None]), 0.0)
    elif metric in ("l2", "euclidean"):
        d2 = block[rows] ** 2 - (b - a[:, None]) ** 2
        block[rows] = xp.sqrt(xp.maximum(d2, 0.0))
    elif metric == "cosine":
        dot = sim[rows] - a[:, None] * b
        npd = xp.sqrt(xp.maximum(pred_norm_squares[rows] - a * a, 0.0))[:, None]  # (V, 1)
        nre = xp.sqrt(xp.maximum(real_norm_squares[None, :] - b * b, 0.0))        # (V, n_real)
        denom = npd * nre
        denom_safe = xp.where(denom != 0, denom, 1.0)
        sim_row = xp.where(denom != 0, dot / denom_safe, 0.0)
        block[rows] = xp.clip(1.0 - sim_row, 0.0, 2.0)
    else:
        raise ValueError(f"unsupported distance metric {metric!r}; use l1, l2, or cosine")
