# src/cell_eval2/cell_source.py
"""Streaming, memory-bounded source over a cell-layout ``cellstream.cell`` archive (#117).

The cell-layout counterpart to ``stream.py`` + ``streaming_bulk.py``: metadata, a
control-pool reference read, per-group blocks, and per-perturbation pseudobulk. DE
lives in ``de_compute.compute_de_streaming_cell`` (gpudge). This module imports the
``cellstream.cell`` API ONLY through ``_cell_archive`` (the sole-importer shim) and never imports
gpudge. Reuses the frozen ``streaming_bulk`` accumulators verbatim; its per-group means
match the materialize ``prep.pseudobulk`` to ~1e-9 relative (float summation-order between
the nnz scatter-add and the dense mean — NOT bit-identity; the same gate the shard
streaming path uses, ``test_streaming_bulk.py`` rtol=1e-9). Rank/set metrics downstream are
exact; continuous metrics inherit ~1e-8 (see ``tests/test_cell_source.py::_assert_parity``).
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp

from ._cell_archive import cell_group_spans, cell_reference_row_count
from .moments import DEFAULT_BULK_TARGET_SUM
from ._threads import resolve_gather_threads


@dataclass(frozen=True)
class CellMeta:
    n_obs: int
    n_vars: int
    var_names: np.ndarray
    group_by: str
    perts: np.ndarray  # unique group labels (INCLUDING the control)


def cell_metadata(store) -> CellMeta:
    group_by = store.manifest.get("group_by")
    if group_by is None:
        raise ValueError(
            "cell archive was not written with a group_by key (not target-aware); "
            "streaming scoring requires a grouped archive"
        )
    var_names = np.asarray(store.var.index.values, dtype=str)
    perts = np.unique(np.asarray(store.group_labels(), dtype=str))
    return CellMeta(int(store.n_obs), int(store.n_vars), var_names, str(group_by), perts)


def cell_fingerprint(store) -> str:
    """Stable content identity of a cell archive for partition-aggregate safety.

    Structural fallback identical to ``stream.shad_fingerprint`` (a sha256 of shape +
    group_by + perts + genes), but the PREFERRED key differs by layout — deliberately, and
    the two layouts are never aggregated together. ``shad_fingerprint`` prefers the shard
    manifest's ``writer_fingerprint`` when it is a non-empty STRING (a content id there);
    the CELL manifest's ``writer_fingerprint`` is ALWAYS a provenance dict
    (``{cellstream_version, writer_kind}`` -- ``shardad_version`` in archives written before
    the rename; both are dicts, so both take the same branch), never a string, so probing it
    would be dead code.
    Instead cell prefers ``payload_sha256`` — the cell format's actual payload digest —
    when present (written only with ``payload_checksum=True``; default writes set it to
    None, so the structural hash is the usual path). Consequently two cell archives with the
    same schema but different payload collide on the structural hash (same documented limit
    as shad's structural fallback); use ``payload_checksum=True`` for payload-level identity."""
    sha = store.manifest.get("payload_sha256")
    if isinstance(sha, str) and sha:
        return f"cell:{sha}:{store.manifest.get('group_by')}"
    m = cell_metadata(store)
    h = hashlib.sha256()
    for part in (m.n_obs, m.n_vars, m.group_by, *sorted(m.perts.tolist())):
        h.update(str(part).encode())
        h.update(b"\x00")
    for gene in m.var_names:
        h.update(str(gene).encode())
        h.update(b"\x00")
    return "cell:" + h.hexdigest()


def validate_cell_pair(pred_meta: CellMeta, real_meta: CellMeta, *, pert_col: str,
                       control: str) -> None:
    """Structural compatibility of a (pred, real) cell-archive pair, mirroring
    io.validate_pair on cell metadata (no payload decode): identical gene axis (count +
    names/order), matching grouping column, identical perturbation sets, and the control
    present. Raises ValueError on mismatch — same error class as compute_metrics."""
    if pred_meta.n_vars != real_meta.n_vars:
        raise ValueError(
            f"gene dimension mismatch: pred {pred_meta.n_vars} != real {real_meta.n_vars}"
        )
    if not np.array_equal(pred_meta.var_names, real_meta.var_names):
        raise ValueError("gene names/order differ between pred and real")
    for name, m in (("pred", pred_meta), ("real", real_meta)):
        if m.group_by != pert_col:
            raise ValueError(
                f"{name} archive group_by {m.group_by!r} != pert_col {pert_col!r}"
            )
    if not np.array_equal(np.sort(pred_meta.perts), np.sort(real_meta.perts)):
        only_pred = sorted(set(pred_meta.perts) - set(real_meta.perts))
        only_real = sorted(set(real_meta.perts) - set(pred_meta.perts))
        raise ValueError(
            f"perturbation sets differ: {pred_meta.perts.size} pred vs "
            f"{real_meta.perts.size} real; pred-only={only_pred[:20]}, "
            f"real-only={only_real[:20]}"
        )
    if control not in set(real_meta.perts):
        raise ValueError(
            f"control perturbation {control!r} not found: {sorted(real_meta.perts)}"
        )


def _to_f32_csr(X) -> sp.csr_matrix:
    """CSR float32 view; only the data array is recast (indices/indptr shared) — the
    frugal cast used across the streaming paths (stream.iter_blocks / streaming_bulk)."""
    X = X.tocsr() if sp.issparse(X) else sp.csr_matrix(X)
    if X.dtype != np.float32:
        X = X.__class__((X.data.astype(np.float32), X.indices, X.indptr), shape=X.shape)
    return X


def _group_row_counts(store) -> dict[str, int]:
    """``{label: n_rows}`` from the archive's own group records -- the per-group input to the
    gather-thread ramp (#149). Exact by construction; see ``_cell_archive.cell_group_spans``
    for why the obs-derived alternative is not equivalent. ``{}`` when unavailable, which makes
    callers fall back to the conservative small-read thread default."""
    return {label: stop - start for label, (start, stop) in cell_group_spans(store).items()}


def cell_reference(store, *, gather_threads: int = -1) -> sp.csr_matrix:
    """The control/non-targeting pool (stored reference-first from row 0), as float32 CSR.

    The gather-thread ramp is sized by the REFERENCE POOL's own row count, not by
    ``store.n_obs``: an upper bound would resolve a 40-row reference inside a million-row
    archive to the full cap and recreate exactly the small-read over-threading the ramp exists
    to prevent (#149). ``read_reference`` takes no row ids, so the count comes from the archive's
    group records via ``_cell_archive.cell_reference_row_count`` (``None`` -> conservative
    default). In production this is the large-read pattern anyway (217,760 rows / 1.107 G nnz on
    the real archive, still scaling at 12 threads).
    """
    ref = store.read_reference(
        n_threads=resolve_gather_threads(cell_reference_row_count(store), gather_threads))
    if ref is None:
        raise ValueError(
            "cell archive has no reference (control) pool; write it with "
            "reference=<control mask/label(s)> to score DE against the control"
        )
    return _to_f32_csr(ref)


def cell_group_blocks(store, *, exclude=None, gather_threads: int = -1):
    """Yield ``(X_csr float32, labels[str])`` per group label. One group resident at a
    time. ``exclude`` (str or iterable) drops matching labels (e.g. the control for the DE
    target stream). With ``exclude=None`` every cell is visited exactly once (the control
    is a normal group in a cell archive), so no separate reference block is prepended
    (unlike the shard path, whose ``iter_blocks`` excludes the reference shard).

    ``gather_threads`` is resolved PER GROUP against that group's row count (#149): group sizes
    span three orders of magnitude in a real archive (median 727 rows, control pool 217,760),
    and a flat thread count is measurably wrong at both ends.
    """
    if exclude is None:
        skip = set()
    elif isinstance(exclude, str):
        skip = {exclude}
    else:
        skip = set(map(str, exclude))
    sizes = _group_row_counts(store)
    for label in store.group_labels():
        label = str(label)
        if label in skip:
            continue
        X = _to_f32_csr(store.read_group(
            label, n_threads=resolve_gather_threads(sizes.get(label), gather_threads)))
        labels = np.full(X.shape[0], label)  # numpy infers a '<U' str array from the value
        yield X, labels


def iter_cell_groups(store, labels, *, gather_threads: int = -1):
    """Yield ``(g, X_csr float32)`` for ``g, label in enumerate(labels)`` — the DE
    ``target_source`` feedstock. gpudge-free: ``de_compute`` adds per-row library sizes
    (with gpudge's own ``csr_row_sums``, for byte-parity) and wraps this into the
    ``refpool_de_core`` target source. ``gather_threads`` is resolved per group against that
    group's row count (#149).
    """
    sizes = _group_row_counts(store)
    for g, label in enumerate(labels):
        label = str(label)
        yield g, _to_f32_csr(store.read_group(
            label, n_threads=resolve_gather_threads(sizes.get(label), gather_threads)))


def cell_pseudobulk(store, *, pert_col, norms, target_sum, noise=None, device="cpu",
                    with_median_umi=False, gather_threads: int = -1, with_moments=False,
                    bulk_target_sum: float = DEFAULT_BULK_TARGET_SUM):
    """Memory-bounded per-perturbation pseudobulk over a cell archive. Reuses the frozen
    ``streaming_bulk`` accumulators verbatim (per-group means match the materialize path to
    ~1e-9 relative — see the module docstring), so the only new code is sourcing blocks from
    the cell store instead of shard shards.

    Returns ``{norm: (perts, means)}``. Optional extras are APPENDED in a fixed order, so
    the four cases are::

        with_median_umi=False, with_moments=False -> out
        with_median_umi=True,  with_moments=False -> (out, median_umi)
        with_median_umi=False, with_moments=True  -> (out, moments)
        with_median_umi=True,  with_moments=True  -> (out, median_umi, moments)

    ``moments`` is ``{norm: GroupMoments}`` spanning ALL groups including the control
    (issue #198); it must never be restricted to a perturbation subset.

    Unlike the shard path, cell groups already INCLUDE the control (it is a normal group),
    so all groups are streamed via ``cell_group_blocks`` with no separate reference block.
    """
    from .gpu import resolve_device
    from .streaming_bulk import (_streaming_pseudobulk_cpu, _streaming_pseudobulk_gpu,
                                 _tee_library_sizes)
    norms = list(norms)
    meta = cell_metadata(store)
    if pert_col != meta.group_by:
        # the archive is grouped BY group_by; a mismatch would mis-map labels (mirrors
        # streaming_bulk.streaming_pseudobulk's guard).
        raise ValueError(
            f"pert_col {pert_col!r} != archive group_by {meta.group_by!r}; streaming "
            "pseudobulk requires scoring on the archive's grouping column"
        )
    if target_sum is None and any(n in ("normalized", "lognorm") for n in norms):
        # v1 median normalization (target_sum=None) would fail inside the accumulator (None
        # used in the CPM row-scale); raise a clear deferred-feature error instead (mirrors
        # the DE deferral in compute_de_streaming_cell). 'counts' needs no target_sum.
        raise NotImplementedError(
            "cell-layout streaming pseudobulk with target_sum=None (v1 median normalization) "
            "is deferred: the 'normalized'/'lognorm' accumulators require a numeric target_sum. "
            "Use target_sum=1e6 (v2 CPM)."
        )
    perts = np.sort(meta.perts)
    P, G = perts.size, meta.n_vars
    def make_label_blocks():
        # ALL groups including control; cell archives have no separate reference block.
        blocks = cell_group_blocks(store, gather_threads=gather_threads)
        if noise is not None:
            from .noise import noise_blocks

            blocks = noise_blocks(blocks, **noise)
        return blocks

    def make_blocks():
        for X, labels in make_label_blocks():
            yield X, np.searchsorted(perts, labels).astype(np.intp)

    blocks = make_label_blocks()
    libs_sink: list[np.ndarray] = []
    if with_median_umi:
        blocks = _tee_library_sizes(blocks, libs_sink)
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
