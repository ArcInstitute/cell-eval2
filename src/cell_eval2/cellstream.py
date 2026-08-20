"""Out-of-core scoring of cell-layout ``.shad`` archives on the ``partition_inmem`` engine.

``CellBatchSource`` adapts a ``cellstream.cell`` archive to the ``partition_inmem.PertBatchSource``
protocol (perturbation-complete AnnData batches of stored values), and ``score_cellstream``
orchestrates one context through ``_build_reference_streaming_core`` / ``score_piece``. All
normalization (counts CPM / lognorm expm1) happens downstream per ``cfg.input_type`` -- this
module adds none. See internal:docs/superpowers/specs/2026-07-21-cellstream-scoring-unification-design.md.
"""
from __future__ import annotations

import logging
from collections.abc import Iterator
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from . import norm as _norm
from ._cell_archive import open_cell_store
from .cell_source import cell_fingerprint
from ._threads import resolve_gather_threads
from .h5ad_manifest import plan_pert_batches

if TYPE_CHECKING:
    import anndata as ad

logger = logging.getLogger(__name__)


def _sanitize_cell_adata(adata):
    """Cast pandas string-extension (StringArray/ArrowStringArray) indices and columns in
    ``obs``/``var`` to numpy ``object`` dtype in place, and return ``adata``.

    cellstream reads ``obs``/``var`` from parquet, which yields pandas ``string`` (nullable
    StringArray) dtypes -- notably the ``var`` index (gene names). anndata's h5ad writer (used by
    the partition reference cache -- ``real_control.h5ad`` / ``pred_control.h5ad``,
    ``partition_inmem.py``) refuses to write these (``allow_write_nullable_strings=False``).
    h5ad-sourced AnnData carry plain ``object`` strings, so this makes cell-sourced batches match
    and keeps the shared downstream partition/gpudge path unchanged.
    """
    for names_attr in ("obs_names", "var_names"):
        idx = getattr(adata, names_attr)
        if pd.api.types.is_extension_array_dtype(idx.dtype):
            setattr(adata, names_attr, idx.astype(object))
    for df in (adata.obs, adata.var):
        for col in df.columns:
            dtype = df[col].dtype
            # Only the nullable/Arrow string extension arrays are the h5ad-write problem; leave
            # categoricals alone (anndata writes them fine, and is_string_dtype can be True for a
            # string-category dtype on some pandas versions) -- Gemini review.
            if (pd.api.types.is_extension_array_dtype(dtype)
                    and pd.api.types.is_string_dtype(dtype)
                    and not isinstance(dtype, pd.CategoricalDtype)):
                df[col] = df[col].astype(object)
    return adata


class CellBatchSource:
    """A ``PertBatchSource`` over one context of a ``layout='cell'`` cellstream archive.

    ``context=None`` (default; the only mode used in this work) treats the whole archive as one
    context. Reads stored values verbatim -- normalization is applied downstream.
    """

    def __init__(self, store_or_path, *, pert_col, control, context=None, context_col="context",
                 gather_threads=-1):
        self.store = open_cell_store(store_or_path)
        self.pert_col = str(pert_col)
        self.control = str(control)
        self.context = context
        # EvalConfig.gather_threads, stored VERBATIM (-1 = auto), NOT int()-coerced: coercion
        # would turn True/1.5 into 1, a new silent-serial path. It is resolved per gather --
        # against that gather's row count -- and resolve_gather_threads raises on a bad value.
        self.gather_threads = gather_threads
        self.stream_tag = f"{cell_fingerprint(self.store)}|ctx:{context}"
        obs = self.store.obs
        self._labels = obs[self.pert_col].to_numpy().astype(str)
        if context is None:
            self._rows_all = np.arange(int(self.store.n_obs), dtype=np.int64)
        else:
            ctx = obs[context_col].to_numpy().astype(str)
            self._rows_all = np.flatnonzero(ctx == str(context)).astype(np.int64)

    def read_control_block(self) -> "ad.AnnData":
        rows = self._rows_all[self._labels[self._rows_all] == self.control]
        if rows.size == 0:
            raise ValueError(
                f"no control rows for control={self.control!r} in pert_col={self.pert_col!r}; "
                f"available groups: {sorted(set(self._labels[self._rows_all].tolist()))}"
            )
        rows = np.sort(rows)
        return _sanitize_cell_adata(self.store.gather_rows_adata(
            rows, n_threads=resolve_gather_threads(rows.size, self.gather_threads)))

    def iter_pert_batches(self, mem_budget) -> "Iterator[tuple[list[str], ad.AnnData]]":
        ctx_rows = self._rows_all
        if ctx_rows.size == 0:      # empty context slice -> yield nothing (multi-context future)
            return
        ctx_labels = self._labels[ctx_rows]
        order = np.argsort(ctx_labels, kind="stable")
        sorted_labels = ctx_labels[order]
        # sorted_labels is already sorted -> find group boundaries in O(N) instead of the redundant
        # internal sort np.unique would do on this (up to millions-of-cells) array (Gemini review).
        change = np.concatenate(([True], sorted_labels[1:] != sorted_labels[:-1]))
        uniq = sorted_labels[change]
        first = np.flatnonzero(change)
        counts = np.diff(np.concatenate([first, [sorted_labels.size]]))
        rows_by = {u: ctx_rows[order[first[k]:first[k] + counts[k]]] for k, u in enumerate(uniq)}
        counts_by = dict(zip(uniq.tolist(), counts.tolist()))
        ctrl_n = int(counts_by.get(self.control, 0))
        sizes = [(str(u), int(c)) for u, c in zip(uniq.tolist(), counts.tolist())
                 if u != self.control]
        # Derive itemsize from the archive's materialized X dtype instead of hardcoding f32, so the
        # batch memory budget stays accurate for float64 archives (Copilot review; mirrors
        # h5ad_manifest.iter_h5ad_pert_batches deriving it from backed.X.dtype).
        x_dtype = getattr(self.store, "_x_out_dtype", None) or \
            self.store.manifest.get("value_dtype_on_disk", "float32")
        itemsize = np.dtype(x_dtype).itemsize
        for batch_perts in plan_pert_batches(sizes, n_genes=int(self.store.n_vars),
                                             itemsize=itemsize, control_cells=ctrl_n,
                                             mem_budget=mem_budget):
            rows = np.sort(np.concatenate([rows_by[p] for p in batch_perts]))
            yield batch_perts, _sanitize_cell_adata(self.store.gather_rows_adata(
                rows, n_threads=resolve_gather_threads(rows.size, self.gather_threads)))

    def close(self):
        self.store.close()


def _control_label(store, cfg) -> str:
    """The control/reference label. Prefer the archive's uniform ``control_value`` obs column
    (these archives carry it), else fall back to ``cfg.control``.

    #177: whichever path supplies it, the label is then CHECKED for membership in
    ``obs[cfg.pert_col]``. The fallback used to be silent, so a preset's label
    (``cell-eval-0.7.6`` ships ``control: non-targeting``) could be adopted for a dataset that
    labels its controls anything else, and the mistake surfaced downstream as empty or wrong
    metrics, or as a confusing error out of the DE path -- never at the point the wrong decision
    was taken. One membership check covers all three ways to get there: the column is absent,
    the column is present but non-uniform, or ``cfg.control`` is simply wrong.
    """
    obs = store.obs
    source = f"config.control={cfg.control!r}"
    label = str(cfg.control)
    if "control_value" in obs.columns:
        # uniform check in O(N) without sorting / string-materializing the whole column (Gemini)
        vals = obs["control_value"]
        if not vals.empty and vals.nunique(dropna=False) == 1:   # robust O(N) uniform check (Gemini)
            label = str(vals.iloc[0])
            source = "the archive's uniform control_value column"
        else:
            # Present but non-uniform: cellstream scores ONE context, so a control_value that
            # varies across rows means either a multi-context archive being scored as one or a
            # malformed write. Falling back is still the right resolution -- the membership check
            # below is the real guard -- but doing it silently hides a genuinely anomalous
            # archive, and this state cannot occur for anything a conforming writer produced.
            logger.warning(
                "cellstream: the archive's control_value column is present but NOT uniform "
                "(%d distinct values); falling back to %s. cellstream scores a single context, "
                "so a varying control_value suggests a multi-context or malformed archive.",
                int(vals.nunique(dropna=False)), source,
            )
    else:
        logger.info("cellstream: no control_value column in the archive; using %s", source)

    if cfg.pert_col not in obs.columns:
        raise ValueError(
            f"perturbation column {cfg.pert_col!r} missing from the archive's obs; "
            f"got {sorted(map(str, obs.columns))}"
        )
    present = obs[cfg.pert_col].astype(str)
    if not (present == label).any():
        available = sorted(set(present.unique()))
        raise ValueError(
            f"the resolved control label {label!r} (from {source}) does not appear in "
            f"obs[{cfg.pert_col!r}], so scoring would run against a control that does not "
            f"exist -- the reference block would be empty and the failure would surface "
            f"downstream instead of here (#177). {len(available)} label(s) present, e.g. "
            f"{available[:8]}."
        )
    return label


def cell_archive_input_type(archive, *, config=None, strict: bool = False,
                            peek_rows: int = 2000, allow_discrete: bool | None = None) -> str:
    """The expression space a cell-layout ``cellstream.cell`` archive's stored values are in.

    Public entry point for #179. A downstream caller imported ``_resolve_input_type_cell`` to do
    exactly
    this and then compared the answer against its own declared ``input_type`` itself; freezing an
    underscore-private symbol is a promise not expressed in the code, and the caller's real
    question was the AGREEMENT check rather than the peek. So this exposes the outcome:

    * ``strict=False`` (default) -- return the resolved space, ``"counts"`` or ``"lognorm"``.
    * ``strict=True`` -- additionally RAISE when it disagrees with ``config.input_type``.

    ``archive`` may be a path or an already-open cell store; a path is opened and closed here, so
    a caller needs neither ``open_cell_store`` nor any other private import. ``config`` defaults
    to ``EvalConfig()``; only ``input_type``, ``version``, ``allow_discrete`` and
    ``gather_threads`` are read.

    ⚠️ The peek always autodetects (``autodetect=True``), independent of
    ``config.autodetect_input_type``. That is deliberate for every cell/shard driver -- the
    accumulators take no ``input_type`` argument, so a declaration cannot be trusted here -- and
    it is why ``strict=True`` is a genuine check rather than a tautology: the resolved value comes
    from the DATA, and ``config.input_type`` is the claim being tested against it.

    ``allow_discrete=False`` makes the answer independent of ``config.allow_discrete``, which
    otherwise short-circuits the resolver to "counts" without inspecting a value -- pass it when the
    answer is feeding a safety decision rather than an ordinary read. ``None`` (default) uses the
    config's own value.

    ``strict`` is the CONSUMER's policy, not the drivers'. ``score_cellstream`` deliberately
    adopts the peeked value over the declaration (``piece_cfg`` below), because it scores whatever
    space the archive is in. A caller who wants a mismatch to be an error asks for it here.
    """
    from .config import EvalConfig
    cfg = EvalConfig() if config is None else config
    # strict=True is a SAFETY question, so it must not inherit the allow_discrete bypass:
    # `resolve_input_type` answers "counts" the moment that flag is set, without inspecting a value,
    # which would make the advertised agreement check pass for a genuine lognorm archive
    # (codex-review round 2). An explicit allow_discrete= still wins.
    if strict and allow_discrete is None:
        allow_discrete = False
    store, opened = (archive, False)
    if not hasattr(archive, "gather_rows_adata"):
        store, opened = open_cell_store(archive), True
    try:
        resolved = _resolve_input_type_cell(store, cfg, peek_rows=peek_rows,
                                            allow_discrete=allow_discrete)
    finally:
        if opened:
            store.close()
    if strict and resolved != cfg.input_type:
        raise ValueError(
            f"the archive's stored values resolve to input_type={resolved!r}, but the config "
            f"declares {cfg.input_type!r}. The cell-layout accumulators take no input_type "
            "argument and treat stored values according to the resolved space, so scoring on "
            "would silently normalize in the wrong one. Re-declare input_type, or write the "
            "archive in the declared space."
        )
    return resolved


def _resolve_input_type_cell(store, cfg, *, peek_rows: int = 2000,
                             allow_discrete: bool | None = None) -> str:
    """Peek-autodetect counts vs lognorm from a row block of the archive (mirrors
    ``h5ad_manifest._resolve_input_type_h5ad``). The downstream ``_build_reference_streaming_core`` /
    ``score_piece`` validation (``_validate_input_once``) is the guard against a mis-detected type.

    Kept as the private name #179 records an external consumer for; ``cell_archive_input_type``
    above is the supported entry point, and takes a path as well as an open store.

    ``allow_discrete=None`` uses ``cfg.allow_discrete`` -- the behaviour every existing caller
    wants, because for them this is an ORDINARY type resolution and ``allow_discrete`` is a
    legitimate policy for reading an ambiguous integer matrix. A SAFETY GATE must pass ``False``:
    ``resolve_input_type`` returns "counts" the moment the flag is set, without looking at a single
    value (its ``if allow_discrete: return "counts"`` short-circuit), so honouring it would let a
    genuinely lognorm archive declare its way through a raw-counts guard (codex-review). Splitting
    the two here rather than changing the resolver keeps ``score_cellstream`` -- which legitimately
    handles lognorm and threads the resolved type downstream -- unaffected.
    """
    from . import norm as _norm
    stop = min(int(store.n_obs), int(peek_rows))
    peek = store.gather_rows_adata(
        np.arange(stop, dtype=np.int64),
        n_threads=resolve_gather_threads(stop, cfg.gather_threads))
    return _norm.resolve_input_type(
        peek, declared=cfg.input_type, version=cfg.version,
        allow_discrete=cfg.allow_discrete if allow_discrete is None else allow_discrete,
        autodetect=True,
    )


def _uniform_obs(store, col, default) -> str:
    """The single value of ``obs[col]`` if the archive is uniform in it (these archives are
    single-dataset/context); else ``default``. Used only to tag the output frame."""
    obs = store.obs
    if col in obs.columns:
        vals = obs[col]
        if not vals.empty and vals.nunique(dropna=False) == 1:   # robust O(N) uniform check (Gemini)
            return str(vals.iloc[0])
    return default


def score_cellstream(pred, real, *, config=None, mem_budget, outdir=None):
    """Score a pair of cell-layout ``.shad`` archives (single context) on the ``partition_inmem``
    engine. Handles counts and lognorm (via the shared in-memory ``compute_de`` + ``_side_bulks``).
    Returns a ``ScoreResult`` (per_pert / per_context / overall). GPU-only (gpudge backend).
    """
    import os
    import tempfile
    from dataclasses import replace

    import polars as pl

    from .partition import aggregate_partials
    from .partition_inmem import (
        _RefBundle,
        _build_pred_control_reference_core,
        _build_reference_streaming_core,
        _require_partition_config,
        score_piece,
    )
    from .run import _resolve_config
    from .h5ad_manifest import _assemble_score_result, _nsig_names

    # Fail fast + normalize the AUC floor (spec §5.3): the core builders each guard internally,
    # but validating once up front raises before any archive I/O when the config is ineligible
    # (e.g. fdr_scope!='per_pert', or no gpudge backend -- score_cellstream is GPU-only).
    cfg = _require_partition_config(_resolve_config(config, {}))
    if cfg.pert_col == "target":
        cfg = replace(cfg, pert_col="perturbation")

    real_store = open_cell_store(real)
    try:
        control = _control_label(real_store, cfg)
        real_itype = _resolve_input_type_cell(real_store, cfg)
        dataset_tag = _uniform_obs(real_store, "dataset", "cellstream")
        context_tag = _uniform_obs(real_store, "context", "all")
    finally:
        real_store.close()
    pred_store = open_cell_store(pred)
    try:
        pred_itype = _resolve_input_type_cell(pred_store, cfg)
    finally:
        pred_store.close()
    comparator = _norm.resolve_comparator(
        version=cfg.version,
        pred_input_type=pred_itype,
        real_input_type=real_itype,
    )

    nsig_spearman, nsig_real, nsig_pred = _nsig_names(cfg)
    with tempfile.TemporaryDirectory() as ref_dir, tempfile.TemporaryDirectory() as parts_dir:
        real_src = CellBatchSource(real, pert_col=cfg.pert_col, control=control,
                                   gather_threads=cfg.gather_threads)
        try:
            _build_reference_streaming_core(real_src, config=cfg, cache_dir=ref_dir,
                                            mem_budget=mem_budget, input_type=real_itype,
                                            native_gpu_normalize=True,
                                            comparator=comparator)
        finally:
            real_src.close()

        if cfg.control_source == "pred":
            pc_src = CellBatchSource(pred, pert_col=cfg.pert_col, control=control,
                                     gather_threads=cfg.gather_threads)
            try:
                _build_pred_control_reference_core(pc_src, config=cfg, cache_dir=ref_dir,
                                                   input_type=pred_itype,
                                                   comparator=comparator)
            finally:
                pc_src.close()

        piece_cfg = replace(cfg, control=control, input_type=pred_itype)
        # One bundle per context: the reference artifacts are written above and never change
        # during this loop, so re-reading them per batch was pure waste (#153). Built BEFORE the
        # source is opened -- constructing it after would leak pred_src if it raised, since the
        # try/finally that closes the source has not been entered yet.
        bundle = _RefBundle(ref_dir, piece_cfg)
        pred_src = CellBatchSource(pred, pert_col=cfg.pert_col, control=control,
                                   gather_threads=cfg.gather_threads)
        try:
            for k, (batch_perts, batch_ad) in enumerate(pred_src.iter_pert_batches(mem_budget)):
                score_piece(batch_ad, ref_dir, config=piece_cfg,
                            piece_id=f"ctx_{k}", partial_out=parts_dir, bundle=bundle,
                            comparator=comparator,
                            # counts -> gpudge normalizes on-GPU (skips the CPU _to_linear copy,
                            # issue #142); lognorm falls through to _to_linear inside compute_de.
                            native_gpu_normalize=True)
        finally:
            pred_src.close()
            # Release the cached control pool / DE table before aggregation. A Python local is
            # not scoped to the block, so without this the bundle stays live through
            # aggregate_partials -- at a large control pool that is GBs held for no reason (#153).
            del bundle

        full, _agg = aggregate_partials(
            parts_dir, reduce_nsig_spearman=True, nsig_spearman_metric=nsig_spearman,
            nsig_real_metric=nsig_real, nsig_pred_metric=nsig_pred)

    full = full.with_columns(
        pl.lit(dataset_tag).alias("dataset"),
        # int sentinel: manifest orchestrators set an int panel_id, so keep the ScoreResult schema
        # type-consistent (avoids a Utf8-vs-Int64 Polars error if outputs are concatenated) -- Gemini
        pl.lit(-1, dtype=pl.Int64).alias("panel_id"),
        pl.lit(context_tag).alias("context"),
    )
    result = _assemble_score_result([full])
    if outdir is not None:
        os.makedirs(outdir, exist_ok=True)
        result.per_pert.write_parquet(os.path.join(outdir, "per_pert.parquet"))
        result.per_context.write_parquet(os.path.join(outdir, "per_context.parquet"))
        result.overall.write_parquet(os.path.join(outdir, "overall.parquet"))
    return result
