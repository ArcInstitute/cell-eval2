from __future__ import annotations

import functools
import importlib.util
import logging
import os

import anndata as ad
import numpy as np
import polars as pl
from scipy.sparse import issparse
from scipy.stats import false_discovery_control

from .de import normalize_de_schema
from .gpu import _release_gpu_pool
from .prep import _group_row_index, _grouped_means

logger = logging.getLogger(__name__)


@functools.cache
def _notice_scanpy_ignores_threads() -> None:
    """One-time notice: the scanpy DE backend has no thread knob (process-level, cached)."""
    logger.info(
        "scanpy DE backend runs the Wilcoxon test single-threaded and ignores num_threads; "
        "install pdex or gpudge for parallel DE."
    )

# deseq2 is opt-in and NEVER auto-selected (see _resolve_backend's auto tuple). Its CPU numpy
# backend runs without a GPU, so _available needs only the module (no GPU branch like gpudge).
_BACKEND_MODULE = {"gpudge": "gpudge", "pdex": "pdex", "scanpy": "scanpy", "deseq2": "deseq2_gpu"}


def _available(backend: str) -> bool:
    if backend not in _BACKEND_MODULE:
        return False
    if importlib.util.find_spec(_BACKEND_MODULE[backend]) is None:
        return False
    if backend == "gpudge":  # GPU-only: needs torch + a CUDA device
        try:  # a broken/partial torch (or cuda) install -> "unavailable"; what _resolve_backend
              # does with that is TIERED (raise on a CUDA host, warn+fall back on a CPU one)
            if importlib.util.find_spec("torch") is None:
                return False
            import torch
            return bool(torch.cuda.is_available())
        except Exception:
            return False
    return True


# Warnings emitted at most once per process: _resolve_backend is called repeatedly per run
# (run._cache_backend, partition_inmem._require_partition_config, ...), and a per-call warning
# would bury the signal. Keyed by the resolved backend, so a run that later loses pdex still
# gets its own scanpy line.
_AUTO_WARNED: set[str] = set()


def _reset_auto_backend_warnings() -> None:
    """Test hook: clear the once-per-process 'auto' warning guard."""
    _AUTO_WARNED.clear()


def _cuda_device_present() -> bool:
    """Best-effort: is a CUDA device visible on this host, INDEPENDENT of gpudge?

    ``_available("gpudge")`` returns False for three different reasons -- module missing, torch
    missing, no CUDA device -- so it cannot distinguish "this host has no GPU" (where falling
    back to a CPU DE backend is the only sensible behavior) from "this host HAS a GPU but the
    GPU backend is unusable" (where the fallback silently changes every DE number).

    Probes cupy first, reusing ``gpu.resolve_device`` (which already handles the
    cupy-imports-but-no-driver case), then torch. UNDETECTABLE -> False: this gates a hard
    error, so a spurious raise on a host we cannot classify is worse than a missed one.
    """
    from . import gpu as _gpu

    try:
        if _gpu.resolve_device("auto") == "cuda":
            return True
    except Exception:  # noqa: BLE001 - detection must never fail a run
        pass
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:  # noqa: BLE001 - torch absent/broken -> cannot detect -> treat as CPU
        return False


def _resolve_backend(backend: str) -> str:
    if backend == "auto":
        if _available("gpudge"):
            return "gpudge"
        # A GPU host without a usable gpudge is the dangerous case: the caller believes they are
        # on the GPU engine, and a silent CPU fallback changes every DE number with no signal.
        # Fail loud there. On a CPU-only host there is no better backend to pick, so failing
        # would make the library unusable out of the box -- warn instead (ultrareview 2026-07-25).
        if _cuda_device_present():
            raise RuntimeError(
                "de.backend='auto' found a CUDA device but the gpudge DE backend is "
                "unavailable (module missing, or torch/CUDA unusable). Falling back to a CPU "
                "backend would silently change every DE number, so this is an error rather "
                "than a fallback. Install it with `uv pip install --torch-backend=auto -e "
                "'.[gpudge]'` -- its own extra, because it pulls torch and the `gpu` extra "
                "(cupy + nvCOMP) deliberately does not. ⚠️ `--torch-backend=auto` is not "
                "decoration: the torch build has to match THIS host's CUDA, and a default-index "
                "torch can be unusable on a driver where cupy still sees the device, which "
                "reproduces this same error. (Needs a recent uv -- verified on 0.7.12; an older "
                "one answers `unexpected argument '--torch-backend' found`.) With pip there is no "
                "equivalent, so pass the index for your own CUDA explicitly, e.g. "
                "`--extra-index-url https://download.pytorch.org/whl/cu126` for a 12.6 driver. "
                "When available, CLI `run` artifacts record gpudge's "
                "installed-distribution version and provenance under run_meta.json -> "
                "`environment` -> `gpudge`. "
                "Otherwise set de.backend explicitly to 'pdex' or 'scanpy' to accept the CPU "
                "engine."
            )
        for cand in ("pdex", "scanpy"):
            if _available(cand):
                if cand not in _AUTO_WARNED:
                    _AUTO_WARNED.add(cand)
                    if cand == "scanpy":
                        logger.warning(
                            "de.backend='auto': no CUDA device, so the gpudge GPU backend was "
                            "skipped, AND pdex is not installed -- falling back to scanpy, "
                            "which is substantially slower. Install pdex for a faster CPU "
                            "backend. DE numbers differ between engines."
                        )
                    else:
                        logger.warning(
                            "de.backend='auto': no CUDA device, so the gpudge GPU backend was "
                            "skipped -- using pdex. DE numbers differ between engines."
                        )
                return cand
        raise RuntimeError(
            "no DE backend available; install one of gpudge (GPU) / pdex / scanpy"
        )
    if backend not in _BACKEND_MODULE:  # clear error before _available's KeyError (Copilot #3)
        raise ValueError(
            f"unknown DE backend {backend!r}; choose from {tuple(_BACKEND_MODULE)} or 'auto'"
        )
    if not _available(backend):
        raise RuntimeError(
            f"DE backend {backend!r} is not available (module missing or, for gpudge, no CUDA device)"
        )
    return backend


def _gpudge_supports_inmem_external_ref() -> bool:
    """gpudge_arc #67: in-memory ``de(adata=, reference=<AnnData>)``. Pre-/post-#67 builds
    both report version 0.3.1, so detect by capability, not version string."""
    try:
        from gpudge import _refpool
    except Exception:
        return False
    return hasattr(_refpool, "inmem_external_ref_de")


def _log1p_view(adata_linear: ad.AnnData) -> ad.AnnData:
    """log1p of an already-linear (CPM / library-normalized) AnnData. Returns a new
    AnnData; does not mutate the input. Matches sc.pp.log1p semantics (sets uns['log1p'])."""
    import scanpy as sc

    return sc.pp.log1p(adata_linear, copy=True)  # copy=True: one copy, idiomatic scanpy


def _cpm_log1p(adata: ad.AnnData) -> ad.AnnData:
    """counts -> CPM (target_sum=1e6) + log1p. Computes the single CPM via _to_linear, then
    log1p in place on that (already-copied) matrix -- ONE copy, byte-identical to the prior
    in-line version (same ops, same order). Kept on the public surface because tests use it
    as the scanpy reference oracle and as a counts->lognorm fixture helper. The compute path
    does NOT call this -- it derives the log view from the single _to_linear CPM (via
    _log1p_view) to avoid normalizing twice."""
    import scanpy as sc

    out = _to_linear(adata, "counts")
    sc.pp.log1p(out)
    return out


#: Element budget for `_ref_cpm_from_cells`' dense path: one float64 block of this many values,
#: ~150 MB, which is small enough to hold beside the matrix and large enough that the BLAS call is
#: not dominated by per-chunk overhead. A module constant rather than a literal so a test can shrink
#: it and actually exercise the chunk loop (a fixture narrow enough to force chunking at the real
#: value would need millions of cells).
_REF_CPM_DENSE_ELEMENTS = 20_000_000

#: gpudge reads either spelling as its 1-vs-rest sentinel rather than as a group label
#: (`gpudge.de` docstring: ``ALL_OTHERS`` is ``"__all_others__"``, with ``"all_others"`` kept as the
#: pre-v0.1 alias). Rejected at `compute_de`'s boundary -- see the check there for why.
_GPUDGE_ALL_OTHERS_SPELLINGS = ("__all_others__", "all_others")


def _to_linear(adata: ad.AnnData, input_type: str, target_sum=1e6) -> ad.AnnData:
    """Linear library-normalized space for the LFC means.

    counts  -> normalize_total(target_sum=target_sum); target_sum=None means median
               (v1), 1e6 means CPM (v2).
    lognorm -> expm1 (back to linear; the user's target_sum approximately cancels in the LFC
    ratio -- exactly at epsilon=0, near-exact for epsilon=1e-9 above the epsilon floor).
    """
    import scanpy as sc

    # Guard the branch: _to_linear is imported directly (tests), and an unknown input_type
    # would silently take the lognorm/expm1 path (Copilot on PR #10). compute_de also
    # validates at its boundary (the gpudge counts path skips _to_linear).
    if input_type not in ("counts", "lognorm"):
        raise ValueError(f"input_type must be 'counts' or 'lognorm', got {input_type!r}")
    out = adata.copy()
    if input_type == "counts":
        out.X = out.X.astype(float) if issparse(out.X) else np.asarray(out.X, dtype=float)
        sc.pp.normalize_total(out, target_sum=target_sum)
    else:  # lognorm -> linear
        # Preserve float32 precision instead of upcasting to float64. The streaming/row-store
        # path feeds float32 (scaled_log1p); the fp64 upcast there is pure waste -- the input is
        # already float32-quantized, so expm1 in float32 loses only ~1 float32-ulp vs fp64, while
        # the upcast is a ~2x-memory fp64 temp + fp64 expm1 (52.9% of the streaming profile, PR
        # #102). ONLY float32 input changes; float64/int stay float64 -- bit-identical, so every
        # existing caller/test is untouched. Gated on the cell-eval-0.7.6 parity harness
        # against a real row-store forward eval.
        X = out.X
        dt = np.float32 if X.dtype == np.float32 else np.float64
        if issparse(X):
            X = X.tocsr().astype(dt, copy=True)  # canonical CSR + fresh float .data for in-place expm1 (Gemini PR #10/#103)
            np.expm1(X.data, out=X.data)
            out.X = X
        else:
            Xf = np.asarray(X, dtype=dt)  # float32 in -> no-op view (no fp64 upcast temp)
            out.X = np.expm1(Xf)
    return out


def _gpudge_counts_norm(target_sum):
    """Map a counts ``target_sum`` to gpudge's ``(cpm_normalize, normalize_target_sum)`` knobs.

    Mirrors ``compute_de_streaming``'s target_sum branch exactly so the in-mem native-normalize
    path (issue #142) and the streaming path resolve gpudge's normalization identically:
    ``1e6 -> CPM``, ``None -> "median"``, any other finite target -> that library size.
    """
    if target_sum == 1e6:
        return True, None
    if target_sum is None:
        return False, "median"
    # Fail loud on a bad target instead of coercing it (bool -> 1.0, NaN/inf/<=0) into gpudge's
    # on-GPU normalize, where it would divide by a garbage library size (Copilot review). The
    # CPM-filter gate only validates target_sum when a cpm filter is active, so the native path
    # needs its own check. Mirrors the "positive, finite" cpm-gate message.
    if isinstance(target_sum, bool) or not np.isfinite(target_sum) or target_sum <= 0:
        raise ValueError(
            "native_gpu_normalize counts target_sum must be positive, finite (or None for "
            f"median); got {target_sum!r}"
        )
    return False, float(target_sum)


def _group_means_linear(adata_linear, groupby, mean_calc) -> dict[str, np.ndarray]:
    """Per-(group, gene) mean in linear space.

    arithmetic: mean(X);  geometric: expm1(mean(log1p(X))). Delegates to the shared
    prep grouped-reduction helper (bit-identical to the previous per-mask loop).
    """
    # Guard the branch point itself: compute_lfc_table/_group_means_linear are imported
    # directly (tests), so an unknown mean_calc must not silently fall through to
    # arithmetic (Copilot on PR #10). compute_de also validates at its boundary (gpudge
    # path never reaches here).
    if mean_calc not in ("arithmetic", "geometric"):
        raise ValueError(f"mean_calc must be 'arithmetic' or 'geometric', got {mean_calc!r}")
    labels = adata_linear.obs[groupby].to_numpy().astype(str)
    uniq, order, bounds = _group_row_index(labels)
    m = _grouped_means(adata_linear.X, order, bounds, uniq.size,
                       log_space=(mean_calc == "geometric"))
    return {g: m[i] for i, g in enumerate(uniq)}


def _clipped_log2fc(
    target_mean: np.ndarray,
    ref_mean: np.ndarray,
    *,
    epsilon: float,
    clip_value: float | None = None,
) -> np.ndarray:
    """log2 fold change from aligned linear per-group means.

    clip_value is None -> log2((mt + epsilon) / (ref + epsilon))  (v2 epsilon path).
    else -> pdex 0.1.27 _fold_change zero-mean semantics on the linear ratio (no epsilon):
    both-zero -> 1, ref-zero -> clip_value, tgt-zero -> 1/clip_value."""
    mt = target_mean
    ref = ref_mean
    # Zero means produce expected inf/NaN on the clip_value=None + epsilon=0 path (the
    # deliberate v2/test inf-keeping case); suppress numpy's divide/invalid RuntimeWarnings
    # (values unchanged -- errstate only gates warning emission; mirrors pdex's _fold_change).
    with np.errstate(divide="ignore", invalid="ignore"):
        if clip_value is None:
            return np.log2((mt + epsilon) / (ref + epsilon))
        both0 = (mt == 0) & (ref == 0)
        ref0 = (ref == 0) & (mt != 0)
        tgt0 = (mt == 0) & (ref != 0)
        safe_ref = np.where(ref == 0, 1.0, ref)
        ratio = mt / safe_ref
        ratio = np.where(
            both0,
            1.0,
            np.where(ref0, clip_value, np.where(tgt0, 1.0 / clip_value, ratio)),
        )
        return np.log2(ratio)


def _lfc_from_means(
    means: dict[str, np.ndarray],
    genes: np.ndarray,
    *,
    reference: str,
    epsilon: float,
    clip_value: float | None = None,
) -> pl.DataFrame:
    """Assemble the canonical log2FC table from precomputed per-group linear means:
    log2((mean_target + epsilon) / (mean_ref + epsilon)) for every non-reference target."""
    if reference not in means:  # clear error instead of a cryptic KeyError (Copilot on PR #10)
        raise ValueError(f"reference group {reference!r} not found")
    ref = means[reference]
    frames = []
    for g in means:
        if g == reference:
            continue
        lfc = _clipped_log2fc(means[g], ref, epsilon=epsilon, clip_value=clip_value)
        frames.append(pl.DataFrame(
            {"target": g, "feature": genes, "log2_fold_change": lfc}))
    if not frames:  # only the reference group present -> typed-empty (avoids pl.concat([]) crash)
        return pl.DataFrame(schema={"target": pl.Utf8, "feature": pl.Utf8,
                                    "log2_fold_change": pl.Float64})
    return pl.concat(frames, how="vertical")


def compute_lfc_table(
    adata_linear, *, groupby, reference, mean_calc, epsilon, clip_value=None
) -> pl.DataFrame:
    """cell_eval2's canonical log2FC for every non-reference target:
    log2((mean_target + epsilon) / (mean_ref + epsilon)) in linear space."""
    means = _group_means_linear(adata_linear, groupby, mean_calc)
    genes = np.asarray(adata_linear.var_names, dtype=str)
    return _lfc_from_means(
        means, genes, reference=reference, epsilon=epsilon, clip_value=clip_value
    )


def _ref_cpm_from_cells(X, *, n_genes: int) -> np.ndarray:
    """Per-gene arithmetic mean, over these cells, of the TRUE per-cell CPM `x_ig / L_i * 1e6`.

    This is gpudge's `filter_gene_min_cpm_cell` gate quantity in EVERY normalization mode, which
    is what makes it the one definition cell_eval2 can hold on to:

      * gpudge normalizing to a target `T` -- it compares `arith_ref * 1e6/T`, and every cell was
        scaled to `T`, so `mean_i(x_norm) * 1e6/T == mean_i(x_raw/L_i * 1e6)`;
      * gpudge not normalizing -- it builds a separate per-cell-CPM accumulator for the gate
        (`_need_scaled_extra` -> `other_ref_acc`), which is this quantity directly.

    ⚠️ INVARIANT under any per-cell rescaling of `X`, because `x_ig/L_i` is. So it does not matter
    whether `X` holds raw counts, CPM, a `target_sum`-normalized matrix, or expm1'd lognorm values
    -- all four give the same vector. That is the property the caller needs: cell_eval2 holds a
    different one of those four on each gpudge sub-path.

    ⚠️ "Invariant" is the exact statement for the ARITHMETIC: `x_ig/L_i` is unchanged by any
    POSITIVE per-cell rescaling. The floating results of two such matrices can still differ in the
    last bits, which is why the frame route -- bit-exact against gpudge's own accumulator -- is kept
    for the competition rather than replaced by this.

    Computed as `w @ X` with `w_i = 1e6/(n * L_i)` rather than by materialising a rescaled copy: the
    reference pool is ~18k cells x ~18k genes on the official panels, so a float64 copy would be a
    ~2.7 GB transient for a vector. SPARSE input costs at most one O(nnz) copy of `.data` (none when
    it is already float64), never the matrix; DENSE input is walked in bounded CELL chunks, because
    `np.asarray(X, dtype=float)` on a float32 panel is exactly the whole-matrix transient this
    avoids. Row sums accumulate in float64 on both branches -- summing ~18k float32 values in
    float32 loses ~1e-4 relative, which is enough to move a gene sitting on the threshold.
    """
    n_cells = int(X.shape[0])
    if n_cells == 0:
        raise ValueError(
            "the reference cell pool is empty; the reference-only CPM gate (#351) has no cells to "
            "derive a per-gene CPM from"
        )
    if issparse(X):
        # Widen `.data` (O(nnz), never the matrix) BEFORE the reduction rather than casting its
        # result, so the row sums are float64 by construction instead of by scipy's choice of
        # accumulator. Measured on scipy 1.18: a float32 CSR's `.sum(axis=1)` already agreed with
        # the float64 answer on a case where the DENSE float32 reduction lost 12 counts -- but that
        # is an implementation detail of the build, not a contract, and gpudge widens for the same
        # reason (`_csr_dense.py`). One O(nnz) copy also serves the matvec, which would otherwise
        # upcast `.data` itself.
        Xw = X if X.dtype == np.float64 else X.astype(np.float64)
        L = np.asarray(Xw.sum(axis=1), dtype=np.float64).ravel()
        L[L <= 0.0] = 1.0      # a zero-library cell contributes 0 to every gene either way
        out = np.asarray(Xw.T @ ((1e6 / L) / n_cells), dtype=np.float64).ravel()
    else:
        # Cell chunks bounded by an ELEMENT budget, so the float64 block is the same size whatever
        # the panel's width. `max(1, ...)` twice: a zero-gene matrix must not divide by zero, and a
        # panel wider than the whole budget still advances one cell at a time.
        rows = max(1, int(_REF_CPM_DENSE_ELEMENTS // max(1, int(X.shape[1]))))
        out = np.zeros(int(X.shape[1]), dtype=np.float64)
        for lo in range(0, n_cells, rows):
            block = np.asarray(X[lo:lo + rows], dtype=np.float64)
            L = block.sum(axis=1)
            L[L <= 0.0] = 1.0
            out += ((1e6 / L) / n_cells) @ block
    if out.size != n_genes:    # a shape slip here would silently gate the wrong genes
        raise ValueError(f"reference CPM vector has {out.size} entries, expected {n_genes}")
    return out


def _gpudge_gate_plan(threshold, *, mean_calc, target_sum, gpudge_normalized, ref_cells,
                      var_names, where: str):
    """`(threshold, cpm_factor, ref_cpm, keep_gpudge_gate)` for the reference-only CPM gate (#351).

    cell_eval2 owns this gate even on the gpudge path, because gpudge's own
    `filter_gene_min_cpm_cell` keeps a (target, gene) cell when the TARGET group's mean CPM clears
    the threshold OR the reference's does (`gpudge._filter.combined_keep_mask`: "AND each active
    filter's (target OR ref) mask"). That OR is the #351 leak, and it is upstream of
    `_apply_cpm_filter` -- `compute_de` returns from its gpudge branch before the CPU gate runs --
    so the CPU clause alone would move no gpudge number.

    TWO ROUTES, and which one is taken decides whether gpudge's own gate may stay on:

    * `ref_cpm is None` (the FRAME route): `ref_mean * cpm_factor` is the per-cell-CPM mean the gate
      wants, and it is gpudge's OWN accumulator rather than merely equal to it. That needs
      `mean_calc='arithmetic'` (else `ref_mean` is the GEOMETRIC control mean -- gpudge does emit
      that, but its gate compares an ARITHMETIC reference mean it accumulates separately and does
      NOT return, so the emitted column is the wrong statistic) AND `gpudge_normalized` -- gpudge
      itself scaled every cell
      to a known finite `target_sum`, so its gate computes `arith_ref * 1e6/T` from the very array
      it returns. The competition takes this route with `cpm_factor` exactly 1.0.
      ⚠️ NOT enough that SOME normalization gave the cells a uniform library. When `_to_linear`
      pre-normalized counts on the CPU, `ref_mean * 1e6/T` is only ALGEBRAICALLY equal to the
      per-cell CPM: gpudge downcasts the staged values to float32, so the staged row totals need
      not be `T` bit-for-bit, and its gate reads a separate per-cell accumulator anyway. Those go
      to the matrix route, which compares the quantity this docstring names.
    * otherwise (the MATRIX route): `ref_cpm` comes from `_ref_cpm_from_cells` on the reference
      cells, which is normalization-invariant and therefore right on every remaining sub-path --
      lognorm, CPU-pre-normalized counts, an unknown library (gpudge's internal median), and a
      geometric `mean_calc` alike.

    `keep_gpudge_gate` is the fourth return, and the caller MUST pass
    `filter_gene_min_cpm_cell=None` to gpudge when it is False. It is True only on the FRAME route
    AND when GPUDGE ITSELF normalized to the finite target: only then does gpudge's gate literally
    compute `arith_ref * 1e6/T` from the array it returns, which is what makes the nesting exact
    (`ref > t` implies `tmean > t or ref > t`, so its pre-filter removes nothing this gate would
    keep) while letting it keep its gene-chunk pruning. In every other case gpudge compares a
    quantity that agrees with this one mathematically but not bit-for-bit -- a separate
    per-cell-CPM accumulator when cell_eval2 pre-normalized, an arithmetic mean when this gate is
    reading a geometric frame -- so a gene within float noise of the threshold could be dropped by
    its OR for one target and kept here for another. That is a target-DEPENDENT surviving set, the
    one thing #351 exists to rule out, so gpudge's gate is muted rather than trusted.

    `ref_cells` is None when the reference matrix is not in hand (`compute_de_streaming`: the
    reference shard lives inside the archive), and then the matrix route is unavailable -- an
    active gate that cannot take the frame route RAISES there rather than gate on the wrong
    quantity. No shipped preset reaches it: `v1` is the only one with `target_sum=None` or a
    geometric `mean_calc` and it sets `filter_gene_min_cpm_cell=None`, so its gate is inert.
    """
    if threshold is None:
        return None, 1.0, None, False
    if float(threshold) < 0.0:
        # A negative threshold is the documented explicit keep-all (`FilterParams.__post_init__`,
        # `_apply_cpm_filter`'s `threshold < 0` branch). It must not pick a route, and above all
        # must not RAISE on a path where no route is available: keeping every gene needs neither a
        # reference vector nor a rescale.
        return float(threshold), 1.0, None, False
    # `np.isfinite` raises TypeError on a non-numeric, so reject one explicitly and by name.
    # ⚠️ Deliberately a RAISE, not a coerce-to-None (Gemini round 1 suggested a try/except that
    # falls back to `None`): a `target_sum` that is neither a number nor None is a misconfiguration,
    # and silently re-routing it to the matrix route -- or to the refusal -- would hide it behind an
    # unrelated message. Same idiom and same reasoning as `compute_de_streaming_cell`'s own
    # target_sum check. Unreachable through `EvalConfig`, which validates the field.
    if target_sum is not None and (isinstance(target_sum, (bool, np.bool_))
                                   or not isinstance(target_sum, (int, float, np.integer,
                                                                  np.floating))):
        raise TypeError(
            f"{where}: target_sum must be a number or None for the reference-only CPM gate "
            f"(#351); got {target_sum!r} of type {type(target_sum).__name__}"
        )
    frame_route = (mean_calc == "arithmetic" and gpudge_normalized and target_sum is not None
                   and np.isfinite(target_sum) and target_sum > 0.0)
    if frame_route:
        return float(threshold), 1e6 / float(target_sum), None, bool(gpudge_normalized)
    if ref_cells is None:
        raise NotImplementedError(
            f"{where}: filter_gene_min_cpm_cell={threshold!r} cannot be resolved on this path. "
            f"The gate must be decided by the REFERENCE group alone (#351), which needs either "
            f"gpudge's own normalization target (mean_calc='arithmetic' plus a finite target_sum "
            f"-- got mean_calc={mean_calc!r}, target_sum={target_sum!r}) or the reference cells "
            f"themselves, and an archive-internal reference shard supplies neither. Pass "
            f"target_sum=1e6 with mean_calc='arithmetic' (the v2 defaults), hand the control pool "
            f"in as an AnnData `reference=` (streaming Mode 2, whose cells this can read), or set "
            f"filter_gene_min_cpm_cell=None (the v1 default)."
        )
    return float(threshold), 1.0, _ref_cpm_from_cells(ref_cells, n_genes=len(var_names)), False


def _finalize_gpudge_de(df, *, epsilon, clip_value, fdr_scope,
                        cpm_threshold=None, cpm_factor: float = 1.0, ref_cpm=None,
                        var_names=None):
    """Shared gpudge DE post-processing: v1 zero-mean LFC clip (clip_value not None) from
    gpudge's own target_mean/ref_mean, then the reference-only CPM gate (#351) and its
    per-target BH, then canonical schema, then global BH (fdr_scope='global'). Reused by the
    in-memory (compute_de) and both streaming (compute_de_streaming,
    compute_de_streaming_cell) gpudge paths so the post-processing has one source of truth.

    `cpm_threshold`/`cpm_factor`/`ref_cpm` come from `_gpudge_gate_plan` (threshold None = no
    gate); `ref_cpm` is that function's matrix route and needs `var_names` to align with. The gate
    order -- filter, then BH per target, then the fdr_scope='global' pool -- is the CPU path's
    order exactly (`_apply_cpm_filter` -> `normalize_de_schema` -> `_global_bh`), so the two
    backends agree on what a gated table means and not merely on which rows survive."""
    if clip_value is not None:
        # v1: route gpudge's native LFC through the SAME zero-mean clip as the CPU
        # backends. gpudge emits log2((mt+0)/(ref+0)) -> +/-inf/NaN on zero means;
        # recompute from gpudge's own Float64 target_mean/ref_mean (the exact arrays
        # it used, identical formula) so finite genes are reproduced bit-for-bit and
        # the zero-mean genes get pdex's clip. v2 (clip_value=None) keeps gpudge native.
        if "target_mean" not in df.columns or "ref_mean" not in df.columns:
            raise ValueError(
                "gpudge output is missing target_mean/ref_mean; cannot apply the v1 "
                "clip_value")
        df = df.with_columns(
            pl.Series(
                "log2_fold_change",
                _clipped_log2fc(
                    df["target_mean"].to_numpy(), df["ref_mean"].to_numpy(),
                    epsilon=epsilon, clip_value=clip_value,
                ),
            )
        )
    if cpm_threshold is not None and cpm_threshold >= 0.0:
        # A negative threshold is the explicit keep-all, same as _apply_cpm_filter's branch.
        # Either route decides on ONE value per gene, so it drops the same genes from EVERY target
        # -- the point of #351 -- and re-adjusts each side's own p_adj.
        if ref_cpm is not None:
            if var_names is None or len(var_names) != len(ref_cpm):
                raise ValueError(
                    f"ref_cpm has {len(ref_cpm)} entries but var_names has "
                    f"{None if var_names is None else len(var_names)}; the CPM gate cannot align "
                    "them"
                )
            axis = np.asarray(var_names, dtype=str)
            axis_list = axis.tolist()          # hoisted: both the guard and the keep-set use it
            # An axis that does not match the frame's own features would gate EVERY row away and
            # look like "the threshold was too high". Refuse instead: this is the silent-wrong-genes
            # failure mode the gate exists to remove. Reduce to the UNIQUE features first (Copilot +
            # Gemini round 1): the question is about the gene axis, so the check does not need one
            # row per (target, gene) -- on a 3M-row official frame that is ~18k values instead.
            unknown = df.select(pl.col("feature").unique()).filter(
                ~pl.col("feature").is_in(axis_list))
            if unknown.height:
                raise ValueError(
                    f"the CPM gate's gene axis does not cover the DE frame: "
                    f"{unknown.height} feature(s) are absent from the "
                    f"{axis.size}-gene reference axis, e.g. {unknown['feature'][0]!r}. The "
                    "reference cells and the DE table must share one gene axis."
                )
            kept = axis[np.asarray(ref_cpm) > cpm_threshold]
            df = df.filter(pl.col("feature").is_in(kept.tolist()))
        else:
            if "ref_mean" not in df.columns:
                raise ValueError(
                    "gpudge output is missing ref_mean; cannot apply the reference-only CPM gate "
                    f"(filter_gene_min_cpm_cell={cpm_threshold!r}). Present: {df.columns}"
                )
            df = df.filter(pl.col("ref_mean") * cpm_factor > cpm_threshold)
        df = _bh_per_target(df)
    df = normalize_de_schema(df, name="gpudge")
    # Global BH AFTER schema standardization so _global_bh always sees canonical
    # 'p_value'/'p_adj' regardless of the backend's raw column names (Gemini PR #16).
    if fdr_scope == "global":
        df = _global_bh(df)
    return df


def _resolve_cpm_filter(
    filter_gene_min_cpm_cell: float | None, *, input_type: str, resolved_backend: str
) -> float | None:
    """Effective CPM gene filter (None = disabled). Applies on counts always; on lognorm
    ONLY for gpudge, whose per-cell-CPM gate is normalization-invariant (recovers true CPM
    per cell) and reproduces pdex's pooled-bulk kept-gene set exactly. CPU backends'
    _apply_cpm_filter is a scale-dependent per-cell-mean CPM that would NOT match upstream
    on lognorm, so it is skipped there."""
    if filter_gene_min_cpm_cell is None:
        return None
    if input_type == "counts" or resolved_backend == "gpudge":
        return filter_gene_min_cpm_cell
    logger.info(
        "filter_gene_min_cpm_cell requires counts input for the %r backend; skipping the "
        "CPM gate on input_type=%r", resolved_backend, input_type,
    )
    return None


def compute_de(
    adata: ad.AnnData,
    *,
    backend: str,
    groupby: str,
    reference: "str | ad.AnnData",
    control_group: str | None = None,
    mean_calc: str,
    epsilon: float,
    input_type: str,
    target_sum: float | None = 1e6,
    clip_value: float | None = None,
    filter_gene_min_cpm_cell: float | None,
    fdr_scope: str = "per_pert",
    threads: int = -1,
    replicate_col: str | None = None,
    device: str = "cpu",
    native_gpu_normalize: bool = False,
) -> pl.DataFrame:
    """Compute a canonical-schema DE table via the selected backend.

    Owns backend resolution ('auto'), the non-counts filter skip, and output
    normalization. For CPU backends, cell_eval2 owns the LFC while engines supply
    rank-based p-values.

    ``control_group`` is meaningful ONLY on the gpudge in-memory external-ref path (an
    AnnData ``reference``): the caller may pass the FULL predictions as ``adata`` — including
    the control group — to avoid a subset-copy that transiently ~2x host RAM and OOMs at ~5M
    cells (run._pred_de_input). gpudge then ranks EVERY group in ``adata`` (incl. the control)
    against the external reference pool; the control-vs-refpool rows are spurious and are
    dropped BEFORE post-processing so a 'global' FDR pool matches the concat path exactly (the
    concat path never has a control target — the control is the reference group). ``None`` (or
    the string-reference path, where the control is the reference group and never a target)
    keeps every group."""
    resolved = _resolve_backend(backend)
    if resolved == "deseq2":
        # deseq2 owns its own LFC + p-values (NB-GLM); it ignores mean_calc/epsilon/clip_value and
        # runs its own validation + native per-contrast padj. It never reaches the CPU/gpudge paths.
        from .deseq2_de import run_deseq2_de
        if isinstance(reference, ad.AnnData):
            raise ValueError(
                "the deseq2 backend does not support an AnnData reference "
                "(external-ref DE is gpudge-only); use a string control label"
            )
        if fdr_scope != "per_pert":
            logger.warning(
                "the deseq2 backend computes its own padj (Cook + independent filtering) and "
                "ignores fdr_scope=%r", fdr_scope,
            )
        use_gpu = str(device).startswith("cuda")
        return run_deseq2_de(adata, pert_col=groupby, control=reference,
                             replicate_col=replicate_col, input_type=input_type, use_gpu=use_gpu)
    ext_ref = not isinstance(reference, str)  # AnnData reference -> in-memory external control pool
    if ext_ref:
        # #155: with an external reference the two halves of one LFC ratio are two SEPARATE
        # matrices. On the DEFAULT branch below they are normalized by two independent
        # _to_linear calls, so target_sum=None resolves a different median for each and every
        # log2FC shifts by log2(T_target/T_ref). Excluded: native_gpu_normalize, where gpudge
        # takes ONE union median over reference + all target cells (_refpool.py:536-546) and the
        # ratio is intact. In-tree callers resolve target_sum before reaching here
        # (norm.resolve_target_sum); this guard is for direct/public-API callers. Placed ahead
        # of the capability check so the actionable error wins over a gpudge-build error.
        if input_type == "counts" and target_sum is None and not native_gpu_normalize:
            raise ValueError(
                "external-reference DE with target_sum=None normalizes the target block and "
                "the control pool to two DIFFERENT medians, shifting every log2FC by "
                "log2(T_target/T_ref) (#155). Resolve target_sum to a number first: "
                "cell_eval2.norm.resolve_target_sum(control_ad, input_type='counts', "
                "target_sum=None) returns the control pool's median library size. "
                "(native_gpu_normalize=True is exempt: gpudge takes one union median there.)"
            )
        # gpudge_arc #67: rank a SEPARATE control AnnData (no target/reference concat). gpudge-only.
        if resolved != "gpudge":
            raise ValueError(
                "an AnnData reference= (in-memory external control pool) requires the gpudge "
                f"backend; got resolved backend {resolved!r} (pdex/scanpy must concatenate)"
            )
        if not _gpudge_supports_inmem_external_ref():
            raise RuntimeError(
                "in-memory external-reference DE (control_source='real') requires a gpudge build "
                "that supports an in-memory AnnData reference pool; upgrade gpudge"
            )
        if not reference.var_names.equals(adata.var_names):  # order-strict, idiomatic (Gemini)
            raise ValueError(
                "reference AnnData var_names must equal adata var_names (order-strict) for external-ref DE"
            )
        if reference.n_obs == 0:  # empty pool -> cryptic gpudge/CUDA failure; fail clearly here (Gemini)
            raise ValueError("reference AnnData must contain at least one cell (external control pool)")
        groups = set(adata.obs[groupby].astype(str).unique())
        if not groups:
            raise ValueError(f"no groups in obs[{groupby!r}] to compute DE for")
    else:
        groups = set(adata.obs[groupby].astype(str).unique())
        if reference in _GPUDGE_ALL_OTHERS_SPELLINGS:
            # gpudge reads these two strings as its ALL_OTHERS sentinel, NOT as a group label, and
            # then emits a rest-of-panel `ref_mean` that differs PER TARGET. The reference-only
            # gate (#351) rests on `ref_mean` being one value per gene, so a group literally named
            # `__all_others__` would slip past the membership check below and quietly restore a
            # target-dependent kept set. cell_eval2 has never intended 1-vs-rest DE -- it always
            # compares against a named control -- so reject the spelling outright rather than
            # carry a mode nothing asks for. Scoped to the backends that reach here -- gpudge and
            # the CPU pair; `deseq2` returns above and never consults gpudge, so the spelling is an
            # ordinary group label there.
            raise ValueError(
                f"reference group {reference!r} is gpudge's ALL_OTHERS sentinel spelling, which "
                f"it interprets as 1-vs-rest rather than as a group label -- and 1-vs-rest gives "
                f"each target its OWN reference mean, which the reference-only CPM gate (#351) "
                f"cannot decide from. Rename the group."
            )
        if reference not in groups:  # clear error before compute_lfc_table's KeyError (Gemini)
            raise ValueError(
                f"reference group {reference!r} not found in obs[{groupby!r}]"
            )
        if not (groups - {reference}):  # Copilot #4
            raise ValueError(
                f"no non-reference groups in obs[{groupby!r}]; DE needs at least one group "
                f"besides the reference {reference!r}"
            )
    # Validate the string conventions at the public boundary: a typo would otherwise
    # silently misroute (e.g. unknown input_type -> lognorm/expm1 path, unknown mean_calc
    # -> arithmetic), producing wrong numbers with no error (Copilot on PR #10).
    if input_type not in ("counts", "lognorm"):
        raise ValueError(f"input_type must be 'counts' or 'lognorm', got {input_type!r}")
    if mean_calc not in ("arithmetic", "geometric"):
        raise ValueError(f"mean_calc must be 'arithmetic' or 'geometric', got {mean_calc!r}")
    if fdr_scope not in ("global", "per_pert"):
        raise ValueError(f"fdr_scope must be 'global' or 'per_pert', got {fdr_scope!r}")
    is_counts = input_type == "counts"
    eff_filter = _resolve_cpm_filter(
        filter_gene_min_cpm_cell, input_type=input_type, resolved_backend=resolved
    )

    if resolved == "gpudge":  # native: gpudge computes the matching LFC on-GPU (validated, step 4)
        # gpudge normalizes on-GPU via (cpm_normalize, normalize_target_sum) -- CPM(1e6),
        # an explicit target, or "median" (see `_de_gpudge`/`compute_de_streaming`). Two ways
        # to feed it counts: (a) native_gpu_normalize -> hand gpudge RAW counts + the knobs
        # (no CPU copy; issue #142); (b) default -> pre-normalize on CPU via _to_linear and
        # pass cpm_normalize per target_sum==1e6. Lognorm always takes (b) (gpudge can't invert
        # log1p). Byte-identical between (a) and (b) at 1e6 (both native CPM); at other targets
        # (a) differs from scanpy normalize_total by ~1e-8 float-order (rank/DE bit-exact).
        native_cpm = is_counts and target_sum == 1e6
        if native_gpu_normalize and is_counts:
            # Lever 1 (issue #142): hand gpudge RAW counts + its own normalize knobs so it
            # CPM-normalizes the batch AND the external reference on-GPU (the same
            # `_refpool.inmem_external_ref_de` core the streaming path runs), instead of a CPU
            # `_to_linear` pre-normalization. Mapping mirrors `compute_de_streaming` exactly.
            # Opt-in: only `score_cellstream` sets it, so the shared in-mem `compute_metrics`
            # path (native_gpu_normalize=False) stays byte-identical. gpudge on-GPU CPM vs
            # scanpy normalize_total differ by float summation order -> continuous ~1e-8,
            # rank/DE bit-exact.
            g_cpm, g_norm_target = _gpudge_counts_norm(target_sum)
            gpudge_adata, gpudge_ref = adata, reference
        else:
            # Default / lognorm / CPU-parity path: the reference is on the SAME scale as adata
            # (run._pred_de_input converts the real control to the pred's effective input_type),
            # so normalize it the same way: native CPM (counts + target_sum=1e6) -> pass both raw
            # + cpm_normalize=True; else _to_linear both (honors target_sum / does the lognorm
            # expm1 gpudge can't invert). gpudge's cpm-cell gate is normalization-invariant, so
            # eff_filter passes through unchanged.
            g_cpm, g_norm_target = native_cpm, None
            gpudge_adata = adata if native_cpm else _to_linear(adata, input_type, target_sum)
            if ext_ref:
                gpudge_ref = reference if native_cpm else _to_linear(reference, input_type, target_sum)
            else:
                gpudge_ref = reference
        # The reference-only gate (#351), planned once the matrices are resolved because its
        # matrix route reads them. `ref_cells` is the block whose cells produce the frame's
        # `ref_mean`: the external pool, or the reference group's rows of what gpudge is handed.
        _lin = gpudge_adata
        if ext_ref:
            _ref_cells = gpudge_ref.X
        else:
            _mask = _lin.obs[groupby].to_numpy().astype(str) == str(reference)
            _ref_cells = _lin.X[_mask]
        # `gpudge_normalized`: gpudge itself scaled the cells, which is what makes the frame's
        # `ref_mean` its own gate's array. CPU-pre-normalized input goes to the matrix route even
        # though its libraries are nominally uniform -- see `_gpudge_gate_plan`.
        _gpudge_normalized = bool(g_cpm) or g_norm_target is not None
        gate_threshold, gate_factor, gate_ref_cpm, _keep_gpudge_gate = _gpudge_gate_plan(
            eff_filter, mean_calc=mean_calc, target_sum=target_sum,
            gpudge_normalized=_gpudge_normalized,
            ref_cells=_ref_cells, var_names=_lin.var_names, where="compute_de(gpudge)")
        # ⚠️ gpudge's OWN gate stays on ONLY when gpudge did the normalizing; see
        # `_gpudge_gate_plan` for why anything else has to mute it (a bit-level disagreement at the
        # threshold would restore a target-DEPENDENT surviving set).
        df = _de_gpudge(gpudge_adata, groupby=groupby, reference=gpudge_ref,
                        mean_calc=mean_calc, epsilon=epsilon, cpm_normalize=g_cpm,
                        normalize_target_sum=g_norm_target,
                        filter_gene_min_cpm_cell=(eff_filter if _keep_gpudge_gate else None))
        if ext_ref and control_group is not None:
            # Drop the control group's spurious control-vs-refpool rows BEFORE finalize so the
            # 'global' BH pool (fdr_scope='global') is identical to the concat path (see the
            # compute_de docstring). No-op if adata carries no control group.
            df = df.filter(pl.col("target") != control_group)
        return _finalize_gpudge_de(df, epsilon=epsilon, clip_value=clip_value, fdr_scope=fdr_scope,
                                   cpm_threshold=gate_threshold, cpm_factor=gate_factor,
                                   ref_cpm=gate_ref_cpm, var_names=_lin.var_names)

    # CPU backends: cell_eval2 owns the LFC; the engine supplies only MWU p-values.
    # Normalize ONCE: _to_linear gives the single CPM (counts) / expm1 (lognorm) matrix;
    # the log-space view the MWU engines need is derived from it (counts) or is the
    # original lognorm input -- no second normalize_total (PR #10 deferred optimization).
    linear = _to_linear(adata, input_type, target_sum)
    genes = np.asarray(linear.var_names, dtype=str)
    grp_labels = linear.obs[groupby].to_numpy().astype(str)
    grp_uniq, grp_order, grp_bounds = _group_row_index(grp_labels)
    _primary = _grouped_means(linear.X, grp_order, grp_bounds, grp_uniq.size,
                              log_space=(mean_calc == "geometric"))
    lfc_means = {g: _primary[i] for i, g in enumerate(grp_uniq)}
    lfc_df = _lfc_from_means(
        lfc_means, genes, reference=reference, epsilon=epsilon, clip_value=clip_value
    )

    # Log-space input for the MWU engine; the engine may mutate it in place (no second
    # copy inside the engine). counts -> log1p of the single CPM (a fresh, disposable
    # buffer). lognorm -> the original log-space input: scanpy writes uns so it needs a
    # private copy, but pdex does not mutate (verified PR #8) so it reads adata directly.
    if is_counts:
        log_adata = _log1p_view(linear)
    else:
        log_adata = adata.copy() if resolved == "scanpy" else adata
    if resolved == "scanpy":
        if threads is not None and threads > 1:
            _notice_scanpy_ignores_threads()  # scanpy ignores num_threads (finding #40)
        pvals = _de_scanpy_pvalues(log_adata, groupby=groupby, reference=reference)
    else:  # pdex
        pvals = _de_pdex_pvalues(log_adata, groupby=groupby, reference=reference,
                                 threads=threads)
    df = lfc_df.join(pvals, on=["target", "feature"], how="inner")

    if eff_filter is not None:  # pdex/scanpy: gate on the shared scanpy CPM + recompute BH
        # Reuse the LFC means for the gate when they ARE the arithmetic CPM means
        # (mean_calc='arithmetic'); for geometric, the gate computes its own arithmetic
        # means from the same `linear` matrix (no second normalize_total either way).
        if mean_calc == "arithmetic":
            gate_means = lfc_means  # the LFC means ARE the arithmetic CPM means
        else:
            # geometric LFC: arithmetic gate means from the SAME group index (one pass;
            # no second argsort/label-scan, no _group_means_linear re-call).
            _arith = _grouped_means(linear.X, grp_order, grp_bounds, grp_uniq.size,
                                    log_space=False)
            gate_means = {g: _arith[i] for i, g in enumerate(grp_uniq)}
        # filter_gene_min_cpm_cell is a TRUE CPM (per 1e6) threshold, but `linear` is normalized to
        # target_sum, so gate on means * (1e6/eff_target). eff_target is target_sum, or -- for
        # target_sum=None (median normalization) -- the median library, recovered exactly as the
        # common positive row-sum of `linear` (normalize_total scales every nonzero cell to it).
        # target_sum==1e6 gives factor 1.0 (unchanged), matching the v2 preset bit-for-bit (F4.1).
        if target_sum is None:
            _row_sums = np.asarray(linear.X.sum(axis=1)).ravel()
            _pos = _row_sums[_row_sums > 0]
            eff_target = float(np.median(_pos)) if _pos.size else 1e6
        else:
            eff_target = float(target_sum)
        # eff_target divides the CPM rescale; a direct compute_de call (bypassing config validation)
        # with target_sum <= 0 or non-finite must fail loud, not ZeroDivisionError / propagate NaN.
        if not (np.isfinite(eff_target) and eff_target > 0.0):
            raise ValueError(
                f"CPM gate requires a positive, finite normalization target; got {eff_target}"
            )
        df = _apply_cpm_filter(df, linear, groupby=groupby, reference=reference,
                               threshold=float(eff_filter), arith_means=gate_means,
                               cpm_factor=1e6 / eff_target)
    df = normalize_de_schema(df, name=resolved)
    # Global BH AFTER schema standardization so _global_bh always sees canonical
    # 'p_value'/'p_adj' regardless of the backend's raw column names (Gemini PR #16).
    if fdr_scope == "global":
        df = _global_bh(df)
    return df


def _de_scanpy_pvalues(log_adata, *, groupby, reference) -> pl.DataFrame:
    """MWU p-values from scanpy on PRE-LOG-NORMALIZED input (rank-based, convention-
    independent). LFC is computed by cell_eval2, not scanpy's geometric logfoldchanges.
    Operates IN PLACE on log_adata (rank_genes_groups writes uns), so the caller must pass a
    disposable buffer -- compute_de does (counts: a fresh _log1p_view; lognorm: adata.copy())."""
    import scanpy as sc

    sc.tl.rank_genes_groups(log_adata, groupby=groupby, reference=reference,
                            method="wilcoxon", tie_correct=True, n_genes=log_adata.n_vars)
    res = log_adata.uns["rank_genes_groups"]
    groups = list(res["names"].dtype.names)
    frames = [
        pl.DataFrame({
            "target": g,
            "feature": np.asarray(res["names"][g], dtype=str),
            "p_value": np.asarray(res["pvals"][g], dtype=float),
            "p_adj": np.asarray(res["pvals_adj"][g], dtype=float),
        })
        for g in groups
    ]
    return pl.concat(frames, how="vertical")


def _resolve_threads(threads: int) -> int:
    """pdex rejects threads<=0 / out-of-range; -1 -> all available CPUs (clamped)."""
    if threads is not None and threads > 0:
        return int(threads)
    n = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else (os.cpu_count() or 1)
    return max(1, n)


def _de_pdex_pvalues(log_adata, *, groupby, reference, threads) -> pl.DataFrame:
    """MWU p-values from pdex on PRE-LOG-NORMALIZED input (is_log1p=True); pdex's own
    LFC/mean are discarded -- cell_eval2 computes the LFC. pdex does not mutate its
    input (verified PR #8), so no copy is taken."""
    from pdex import pdex

    frame = pdex(
        log_adata, groupby=groupby, mode="ref", reference=reference,
        geometric_mean=True, epsilon=0.0,
        is_log1p=True, threads=_resolve_threads(threads),
    )
    return frame.select([
        "target", "feature",
        pl.col("p_value"),
        pl.col("fdr").alias("p_adj"),
    ])


def compute_de_streaming(
    shard_archive, *, backend, reference, groupby, mean_calc, epsilon,
    target_sum, clip_value, fdr_scope, filter_gene_min_cpm_cell,
    normalize_target_sum=None, input_type: str = "counts",
) -> pl.DataFrame:
    """gpudge DE streamed directly off a cellstream ``.shad`` archive (memory-bounded).

    The streaming DE path is gpudge-only. ``reference``: None -> the archive's own
    reference shard (Mode 1); a label str -> validated against the reference shard; an
    AnnData -> external control pool (Mode 2, all archive shards are targets). The .shad
    stores raw counts, so normalization is CPM (target_sum==1e6 -> cpm_normalize=True) or
    an explicit ``normalize_target_sum`` (e.g. a precomputed median). Post-processing is the
    SAME shared finalizer as the in-memory gpudge path.

    ``input_type`` (#182): the raw-counts contract this docstring has always STATED is now
    ENFORCED at the function that states it, rather than only at a caller. It previously took no
    ``input_type`` argument at all, so a lognorm archive was silently library-size-normalized --
    ``normalize_target_sum`` handed straight to gpudge, or ``cpm_normalize=True`` -- and the DE
    numbers that came back were plausible and meaningless. Callers pass the side's EFFECTIVE type
    (#266: a declared-counts config over lognorm data is exactly the case that slipped through);
    the ``"counts"`` default keeps every existing call site's meaning."""
    if input_type != "counts":
        raise NotImplementedError(
            f"compute_de_streaming requires a RAW-COUNTS archive; got input_type="
            f"{input_type!r}. It has no expm1/_to_linear step: it hands gpudge a library-size "
            "normalization target (or cpm_normalize=True), so already-log1p'd values would be "
            "rescaled as if they were counts and the DE numbers would be plausible and "
            "meaningless (#182/#266). Re-write the archive from raw counts, or score it through "
            "the in-memory path (compute_metrics/run), which handles lognorm input."
        )
    if backend not in ("auto", "gpudge"):
        raise ValueError(
            f"streaming DE requires the gpudge backend (GPU); got backend={backend!r}. "
            "pdex/scanpy are in-memory only."
        )
    _resolve_backend("gpudge")  # validates gpudge importable + a CUDA device (clear error otherwise)
    if mean_calc not in ("arithmetic", "geometric"):
        raise ValueError(f"mean_calc must be 'arithmetic' or 'geometric', got {mean_calc!r}")
    if fdr_scope not in ("global", "per_pert"):
        raise ValueError(f"fdr_scope must be 'global' or 'per_pert', got {fdr_scope!r}")
    # Counts archive -> CPM(1e6) via cpm_normalize, OR an explicit library-size target.
    if normalize_target_sum is not None:
        cpm_normalize, norm_target = False, float(normalize_target_sum)
    elif target_sum == 1e6:
        cpm_normalize, norm_target = True, None
    elif target_sum is None:
        # median mode without a precomputed value -> let gpudge compute the median (extra pass)
        cpm_normalize, norm_target = False, "median"
    else:
        cpm_normalize, norm_target = False, float(target_sum)
    # The reference-only gate (#351). The matrix route is available in Mode 2 ONLY -- an AnnData
    # external control pool, whose cells cell_eval2 is holding -- and not in Mode 1, where the
    # reference shard lives inside the archive and there is nothing here to derive a vector from.
    # The library size gpudge normalized to is the explicit `normalize_target_sum` when the caller
    # precomputed one (`target_sum` is None then), else `target_sum` itself; gpudge always
    # normalizes on this path, so `gpudge_normalized=True`.
    _stream_ref_is_adata = not isinstance(reference, str) and reference is not None
    gate_threshold, gate_factor, gate_ref_cpm, _keep_gpudge_gate = _gpudge_gate_plan(
        filter_gene_min_cpm_cell, mean_calc=mean_calc,
        target_sum=(float(normalize_target_sum) if normalize_target_sum is not None
                    else target_sum),
        gpudge_normalized=True,
        ref_cells=(reference.X if _stream_ref_is_adata else None),
        var_names=(np.asarray(reference.var_names, dtype=str) if _stream_ref_is_adata else ()),
        where="compute_de_streaming")

    import gpudge

    _release_gpu_pool()  # clean GPU so gpudge's chunk-sizer sees full free VRAM (gpudge_arc#76)
    try:
        df = gpudge.de(
            shard_archive=shard_archive, reference=reference, groupby=groupby,
            mean_calc=mean_calc, epsilon=epsilon, cpm_normalize=cpm_normalize,
            normalize_target_sum=norm_target,
            # muted unless the plan says gpudge's own gate nests exactly
            filter_gene_min_cpm_cell=(filter_gene_min_cpm_cell if _keep_gpudge_gate else None),
        )
    finally:
        # Same VRAM handback as the in-memory path: return cupy's pool to the driver so a later
        # same-process GPU phase isn't starved (gpudge_arc#76). In `finally` for the error path.
        _release_gpu_pool()
    return _finalize_gpudge_de(df, epsilon=epsilon, clip_value=clip_value, fdr_scope=fdr_scope,
                               cpm_threshold=gate_threshold, cpm_factor=gate_factor,
                               ref_cpm=gate_ref_cpm,
                               var_names=(np.asarray(reference.var_names, dtype=str)
                                          if _stream_ref_is_adata else None))


def _gpudge_supports_refpool_core() -> bool:
    """gpudge exposes the shared reference-pool DE core + the CSR helpers cell-layout
    streaming drives directly. Detected by capability, MIRRORING the existing
    ``_gpudge_supports_inmem_external_ref`` idiom (import the private module, then
    ``hasattr``) — not a version string — so a swap to a future PUBLIC gpudge cell-source
    API is a one-line change here."""
    try:
        from gpudge import _csr_dense, _refpool
    except Exception:
        return False
    return (hasattr(_refpool, "refpool_de_core")
            and hasattr(_csr_dense, "ensure_csr")
            and hasattr(_csr_dense, "csr_row_sums"))


def compute_de_streaming_cell(
    *, ref_X, group_iter_factory, targets, var_names, n_genes,
    backend, mean_calc, epsilon, target_sum, clip_value, fdr_scope,
    filter_gene_min_cpm_cell,
) -> pl.DataFrame:
    """gpudge DE streamed group-by-group off a cell-layout archive (memory-bounded).

    Drives gpudge's shared ``refpool_de_core`` — the exact core its own shard-streaming
    (``stream_de``) and in-memory external-ref (``inmem_external_ref_de``) paths use — with
    a cell-store-backed ``target_source``. Bit-identical to those paths by construction.
    ``ref_X`` is the control pool CSR; ``group_iter_factory()`` yields ``(g, X_csr)`` per
    non-control target group (from ``cell_source.iter_cell_groups``). Counts input only, any
    finite ``target_sum > 0`` (CPM to that library size; 1e6 = v2, 1e4 = cell-eval-0.7.6);
    ``target_sum=None`` (v1 median) is deferred (needs a median pre-pass). lognorm input is
    rejected upstream by ``score_streaming_cell`` (this fn has no expm1/_to_linear).
    Post-processing is the SAME ``_finalize_gpudge_de`` as the other gpudge paths."""
    if backend not in ("auto", "gpudge"):
        raise ValueError(
            f"streaming DE requires the gpudge backend (GPU); got backend={backend!r}. "
            "pdex/scanpy are in-memory only; deseq2 streaming is deferred (#125)."
        )
    if mean_calc not in ("arithmetic", "geometric"):
        raise ValueError(f"mean_calc must be 'arithmetic' or 'geometric', got {mean_calc!r}")
    if fdr_scope not in ("global", "per_pert"):
        raise ValueError(f"fdr_scope must be 'global' or 'per_pert', got {fdr_scope!r}")
    if target_sum is None:
        raise NotImplementedError(
            "cell-layout streaming DE with target_sum=None (v1 median normalization) is "
            "deferred: it needs a median pre-pass over all groups. Use target_sum=1e6 (v2 "
            "CPM); v1/median streaming is a fast-follow (mirrors the deseq2-streaming #125)."
        )
    # Reject bool (True/False/np.bool_ are int-like -> would slip through as 1.0/0.0) and any
    # non-numeric (a config string like "1e4" would make np.isfinite raise TypeError); the
    # isinstance check short-circuits before np.isfinite. Allow python AND numpy scalars
    # (np.float64/np.int64 are common) -- Gemini PR #131. (EvalConfig also validates target_sum
    # at construction; this hardens the direct-call boundary.)
    if (isinstance(target_sum, (bool, np.bool_))
            or not isinstance(target_sum, (int, float, np.integer, np.floating))
            or not (np.isfinite(target_sum) and target_sum > 0)):
        raise ValueError(
            f"cell-layout streaming DE target_sum must be a finite float > 0 (the counts "
            f"library-size/CPM target: 1e6 = v2 CPM, 1e4 = cell-eval-0.7.6); got {target_sum!r}"
        )
    _resolve_backend("gpudge")  # validates gpudge importable + a CUDA device (clear error)
    if not _gpudge_supports_refpool_core():
        raise RuntimeError(
            "cell-layout streaming DE requires a gpudge build exposing the reference-pool "
            "core (gpudge._refpool.refpool_de_core + _csr_dense helpers); upgrade gpudge. "
            "The `scale` extra does NOT supply gpudge -- it installs the archive reader."
        )
    import torch

    from gpudge._csr_dense import csr_row_sums, ensure_csr
    from gpudge._refpool import refpool_de_core

    ref_csr = ensure_csr(ref_X, name="reference.X")
    # target_sum is validated finite > 0 above; refpool_de_core takes the ALREADY-RESOLVED
    # target_sum, so pass it explicitly rather than re-deriving it from gpudge's cpm_normalize
    # convention. For target_sum=1e6 this is byte-identical to Stage-1 (whose gpudge call
    # resolves cpm_normalize=True to 1e6 before refpool_de_core); other finite targets (e.g.
    # 1e4 = cell-eval-0.7.6) normalize to that library size. Drops the gpudge._normalize private
    # dependency (Copilot PR #127).
    resolved_ts = float(target_sum)  # counts CPM target; any finite > 0 (1e6 = v2, 1e4 = 0.7.6)
    # The reference-only gate (#351). refpool_de_core normalizes to the resolved target, so an
    # arithmetic mean_calc takes the frame route (target_sum=None was rejected above); a geometric
    # one takes the matrix route off `ref_X`, the very control pool whose cells produce `ref_mean`.
    gate_threshold, gate_factor, gate_ref_cpm, _keep_gpudge_gate = _gpudge_gate_plan(
        filter_gene_min_cpm_cell, mean_calc=mean_calc, target_sum=resolved_ts,
        gpudge_normalized=True,
        ref_cells=ref_csr,                                  # already validated above
        var_names=np.asarray(var_names, dtype=str), where="compute_de_streaming_cell")

    def target_source(need_row_sums):
        # Per group: rank its cells vs the resident reference. rows index into this group's
        # own CSR (arange); Ls uses gpudge's OWN csr_row_sums so the CPM scaling matches the
        # shard/in-mem paths byte-for-byte.
        for g, X in group_iter_factory():
            X = ensure_csr(X, name="target.X")
            rows = np.arange(X.shape[0], dtype=np.int64)
            Ls = csr_row_sums(X) if need_row_sums else None
            yield g, X, rows, Ls

    _release_gpu_pool()  # clean GPU so gpudge's chunk-sizer sees full free VRAM (gpudge_arc#76)
    try:
        df = refpool_de_core(
            ref_X=ref_csr, target_source=target_source, targets=np.asarray(targets),
            n_genes=int(n_genes), var_names=np.asarray(var_names, dtype=str),
            device=torch.device("cuda"), mean_calc=mean_calc, epsilon=epsilon,
            gpu_gene_chunk_size=None, oom_recovery=True, target_sum=resolved_ts,
            output_columns=None, filter_gene_min_mean_value=None,
            filter_gene_min_total_value=None,
            # gpudge normalizes here, so its gate stays on unless the matrix route took over
            filter_gene_min_cpm_cell=(filter_gene_min_cpm_cell if _keep_gpudge_gate else None),
            filter_gene_min_cpm_bulk=None, keep_genes_arr=None, warn_noncount=True,
        )
    finally:
        _release_gpu_pool()  # hand VRAM back to the driver (finally covers the error path)
    return _finalize_gpudge_de(df, epsilon=epsilon, clip_value=clip_value,
                               fdr_scope=fdr_scope, cpm_threshold=gate_threshold,
                               cpm_factor=gate_factor, ref_cpm=gate_ref_cpm,
                               var_names=np.asarray(var_names, dtype=str))


def _de_gpudge(adata, *, groupby, reference, mean_calc, epsilon, cpm_normalize,
               filter_gene_min_cpm_cell, normalize_target_sum=None) -> pl.DataFrame:
    import gpudge

    # The caller (compute_de) owns the normalization decision and passes gpudge's knobs:
    # cpm_normalize=True (CPM 1e6), normalize_target_sum=<N|"median"> (gpudge normalizes RAW
    # counts on-GPU -- issue #142's native path), or neither (the caller pre-normalized via
    # _to_linear and passes cpm_normalize=False). gpudge's filter_gene_min_cpm_cell gate
    # recovers true per-cell CPM regardless (it divides by the per-cell sum), so it stays
    # correct on raw / median- / CPM-normalized input (#21).
    #
    # Free the pool BEFORE de() so gpudge's auto chunk-sizer sees the full GPU. It budgets its
    # gene-chunk from `cudaMemGetInfo` free VRAM, but a prior GPU phase (pred pseudobulk) leaves
    # tens of GB parked in cupy's caching pool -> the sizer reads a starved "free", picks a tiny
    # chunk, and de_pred runs ~10-20x slower (measured on CCL_2: de_pred 1607 s at the starved
    # pick vs ~148 s auto / ~76 s at chunk 4608 on a clean GPU; gpudge_arc#76). Handing the pool
    # back to the driver first restores the large-chunk pick.
    _release_gpu_pool()
    try:
        return gpudge.de(
            adata, groupby=groupby, reference=reference,
            mean_calc=mean_calc, epsilon=epsilon, cpm_normalize=cpm_normalize,
            normalize_target_sum=normalize_target_sum,
            filter_gene_min_cpm_cell=filter_gene_min_cpm_cell,
        )
    finally:
        # gpudge's in-mem sizer leaves ~all of VRAM parked in cupy's pool; hand it back to the
        # driver so a later same-process GPU phase (discrimination's cuBLAS) can allocate
        # (gpudge_arc#76). In `finally` so a gpudge OOM/error still releases the pool rather than
        # leaving VRAM clogged. Cheap: only returns already-free cached blocks, never live arrays.
        _release_gpu_pool()


def _global_bh(df: pl.DataFrame) -> pl.DataFrame:
    """Recompute p_adj as a SINGLE Benjamini-Hochberg pool over ALL rows' p_value.

    NaN-safe; matches pdex 0.1.27's global FDR. Used for fdr_scope='global'.
    """
    if df.is_empty():
        return df
    p = df["p_value"].to_numpy().astype(float)
    adj = np.full(p.shape, np.nan)
    valid = ~np.isnan(p)
    if valid.any():
        adj[valid] = false_discovery_control(p[valid], method="bh")
    return df.with_columns(pl.Series("p_adj", adj))


def _bh_per_target(df: pl.DataFrame) -> pl.DataFrame:
    """Recompute BH-FDR within each target over `df`'s OWN p_values, row order preserved.

    Every gene gate owes this: BH's step-up reads `i/m` over the rows actually tested, so
    dropping rows changes each target's multiple-testing universe `m` and the incoming `p_adj`
    stops describing the table it sits in. gpudge says the same of its own output ("p_adj values
    depend on which genes pass the filter"). It is also what makes the reference-only gate (#351)
    bite on the PREDICTION side: the gate runs per side, so removing an arm's boosted block
    (p ~ 1e-20) collapses the step-up and its remaining marginal calls stop clearing alpha too.

    NaN p_values (degenerate/constant genes from wilcoxon) are held out of the BH set and mapped
    back to NaN p_adj — scipy's false_discovery_control raises on NaN, so passing them through
    would crash the (v2-default) filter path (Gemini re-review).

    Row order is PRESERVED (a scatter back to the original positions, not a group-wise concat),
    so a gated table keeps the order the engine emitted and a caller can compare frames without
    sorting first. Equivalent to scipy per target either way: `false_discovery_control` is
    applied to each target's own non-NaN p_values, nothing crosses a target boundary.
    """
    if df.is_empty():
        return df
    p = df["p_value"].to_numpy().astype(float)
    adj = np.full(p.shape, np.nan)
    # Group by target WITHOUT reordering the frame: sort the row indices by a compact integer
    # code (categorical physical), walk the equal-target runs, scatter each BH result back to
    # the rows it came from. `to_numpy()` on a Utf8 column is an object array, whose argsort
    # costs a Python compare per element -- the codes make it an integer sort.
    # `fill_null(-1)` keeps a NULL target as ONE group, which is what `group_by` did -- without
    # it the physical column carries nulls, every null-vs-null compare is false, and each null row
    # would become its own BH family of size 1. (`normalize_de_schema` warns on such nulls; this
    # only refuses to change their meaning.)
    codes = (df["target"].cast(pl.Categorical).to_physical()
             .cast(pl.Int64).fill_null(-1).to_numpy())
    order = np.argsort(codes, kind="stable")
    sorted_codes = codes[order]
    run_start = np.flatnonzero(np.r_[True, sorted_codes[1:] != sorted_codes[:-1]])
    for lo, hi in zip(run_start, np.r_[run_start[1:], order.size]):
        idx = order[lo:hi]
        pv = p[idx]
        valid = ~np.isnan(pv)
        if valid.any():
            adj[idx[valid]] = false_discovery_control(pv[valid], method="bh")
    return df.with_columns(pl.Series("p_adj", adj))


def _apply_cpm_filter(df: pl.DataFrame, linear, *, groupby, reference, threshold,
                      arith_means=None, cpm_factor: float = 1.0) -> pl.DataFrame:
    """Keep (target,gene) rows whose gene clears `threshold` in the REFERENCE group's mean CPM
    (negative threshold = keep-all), then recompute BH-FDR per target. `linear` is the
    shared normalized AnnData; per-group means are the arithmetic mean of it -- reused from the
    LFC computation via `arith_means` when the LFC also used the arithmetic mean, else computed
    here from `linear`. `linear` is normalized to the LFC target_sum, so the means are scaled to
    TRUE CPM (per 1e6) by `cpm_factor` (=1e6/target_sum) before the threshold compare, keeping the
    kept-gene set target_sum-independent (F4.1).

    ⚠️ The REFERENCE group ALONE decides the kept set (#351), so it is one property of the data,
    identical for every target. It used to be `target-group OR reference-group`, and that OR made a
    row's mere PRESENCE a disclosure of its log2FC's sign: for a gene at or below the threshold in
    the control, the row (t, g) existed only when g rose above the threshold in t, so
    `tmean > threshold >= ref_mean`, hence `(tmean+eps)/(ref_mean+eps) > 1` and log2FC > 0.
    (NOT `log2FC > log2(threshold/ref_mean)`, which #351 states and which the pseudocount breaks:
    `(threshold+eps)/(ref_mean+eps) <= threshold/ref_mean` for `ref_mean <= threshold`, and the
    right-hand side is undefined at `ref_mean = 0`. Strict positivity is the claim that matters and
    the one the measurement confirms.) Measured on the
    three official val panels: P(real log2FC > 0) = 1.000000 over 26,373 / 33,969 / 26,839 such
    rows, and a submission that pasted control cells with counts added to exactly those genes --
    reading no perturbation-specific information at all -- took +0.3722 / +0.5059 / +0.3651 of
    `de_wilcoxon_direction_fidelity_yield_raw`'s `from_baseline` and +0.1057 of OVERALL `avg_score`.
    Under the reference-only gate those arms return to the honest control-paste floor. The cost is
    real and is not hidden: the 0.88%-1.16% of reference rows only a perturbed group detected are
    genuine up-regulations, and they leave the scoreable set.

    ⚠️ The per-target BH recomputation below is LOAD-BEARING for that, not bookkeeping. Dropping
    rows shrinks each target's multiple-testing universe, and it is applied to EACH SIDE's own
    table, so the prediction's `p_adj` is re-adjusted too: removing an attack's boosted block
    (p ~ 1e-20) collapses BH's step-up and its remaining marginal calls stop being significant as
    well. That is why the arm's `n_pred` goes to 0 rather than to ~4."""
    if df.is_empty():  # no rows -> nothing to filter; avoids an empty pl.concat (Gemini #1)
        return df
    means = arith_means if arith_means is not None else _group_means_linear(linear, groupby, "arithmetic")
    genes = np.asarray(linear.var_names, dtype=str)
    ref = means[reference]
    if threshold < 0.0:
        keep_rows = df                                   # explicit keep-all
    else:
        ref_cpm = ref * cpm_factor                       # target_sum-normalized -> true CPM (F4.1)
        # ONE gene set for every target (#351), so no per-target loop and no join: a plain
        # membership filter is the same set of surviving rows the (target, feature) inner join
        # produced, since `kept` no longer varies by target and `kept` is a subset of
        # `linear.var_names` either way.
        kept = genes[ref_cpm > threshold]
        keep_rows = df.filter(pl.col("feature").is_in(kept.tolist()))
    return _bh_per_target(keep_rows)
