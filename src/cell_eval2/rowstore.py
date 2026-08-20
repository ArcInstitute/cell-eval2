"""Score row-store `.dat` forward-eval predictions with cell_eval2.

A *row store* is the sharded on-disk format a forward-eval writer emits: a ``staging_dir``
holding ``plan.json`` plus one ``artifact_<NNNNN>/`` per (dataset, panel, context) with
``real_X.dat`` / ``pred_X.dat`` (``np.memmap``, raw counts), ``obs.csv``, and
``var_names.npy``. This module reads that format directly and streams it into the SP2
partitioned scorer (``partition_inmem`` / ``partition``) -- never materializing a whole
context -- applying ``scaled_log1p`` so the X matches what upstream cell-eval was fed.

NOTE: ``plan.json`` records ABSOLUTE paths from the generating machine; they are treated as
stale. Every file is resolved relative to the caller's ``staging_dir``.
"""
from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Iterator
from dataclasses import dataclass

import anndata as ad
import numpy as np
import pandas as pd
import polars as pl

from ._sparse import SPARSE_DENSITY_MAX, estimate_density, scaled_log1p_csr, to_csr_f32
from . import norm as _norm
from .config import EvalConfig
from .norm import resolve_target_sum


def scaled_log1p(x: np.ndarray, target_sum: float) -> np.ndarray:
    """``log1p(x * target_sum / library_size)`` per row, in float32.

    Every step is float32 -- the input cast, the library-size scaling and the ``log1p`` -- not
    float64 with a narrowing cast at the end. That is the contract
    ``output_space='scaled_log1p'`` carries, and it is what makes the scored X reproducible
    bit-for-bit against a producer that computed it the same way.
    """
    out = np.ascontiguousarray(x, dtype=np.float32)
    library = out.sum(axis=1, keepdims=True)
    library = np.where(library > 0, library, 1.0)
    out = out * (target_sum / library)
    np.log1p(out, out=out)
    return out


@dataclass(frozen=True)
class RowStoreArtifact:
    artifact_id: str
    dataset: str
    panel_id: int
    context: str
    control_value: str
    n_rows: int
    n_genes: int
    dtype: str
    real_path: str
    pred_path: str
    obs_path: str
    var_names_path: str


def read_rowstore_plan(staging_dir) -> list[RowStoreArtifact]:
    """Parse ``<staging_dir>/plan.json`` into artifacts.

    plan.json is read for METADATA only; every file path is resolved as
    ``<staging_dir>/<artifact_id>/<fixed filename>`` -- ``real_X.dat``, ``pred_X.dat``,
    ``obs.csv``, ``var_names.npy`` -- ignoring plan.json's stale absolute paths.
    """
    staging_dir = os.fspath(staging_dir)
    plan_path = os.path.join(staging_dir, "plan.json")
    if not os.path.isfile(plan_path):
        raise FileNotFoundError(f"row-store plan.json not found: {plan_path!r}")
    with open(plan_path, encoding="utf-8") as fh:
        payload = json.load(fh)
    out: list[RowStoreArtifact] = []
    for row in payload["artifacts"]:
        aid = str(row["artifact_id"])
        adir = os.path.join(staging_dir, aid)
        art = RowStoreArtifact(
            artifact_id=aid, dataset=str(row["dataset"]), panel_id=int(row["panel_id"]),
            context=str(row["context"]), control_value=str(row["control_value"]),
            n_rows=int(row["n_rows"]), n_genes=int(row["n_genes"]), dtype=str(row["dtype"]),
            real_path=os.path.join(adir, "real_X.dat"),
            pred_path=os.path.join(adir, "pred_X.dat"),
            obs_path=os.path.join(adir, "obs.csv"),
            var_names_path=os.path.join(adir, "var_names.npy"),
        )
        for p in (art.real_path, art.pred_path, art.obs_path, art.var_names_path):
            if not os.path.isfile(p):
                raise FileNotFoundError(f"row-store artifact {aid} missing file: {p!r}")
        vn = np.load(art.var_names_path, allow_pickle=True)
        if vn.shape[0] != art.n_genes:
            raise ValueError(
                f"artifact {aid}: var_names has {vn.shape[0]} entries, plan n_genes={art.n_genes}")
        out.append(art)
    if not out:
        raise ValueError(f"row-store plan {plan_path!r} has no artifacts")
    return out


class RowStoreBatchSource:
    """A ``PertBatchSource`` over one row-store artifact side ('real' or 'pred').

    Reads ``obs.csv``'s perturbation column once, stable-argsorts it into per-perturbation
    row-index runs (mirroring ``h5ad_manifest.iter_h5ad_pert_batches``), and slices the ``uint16``
    memmap on demand -- applying ``scaled_log1p`` (``output_space='scaled_log1p'``) or a plain
    float32 view (``'raw'``). obs carries only ``pert_col`` (the sole column downstream metrics
    read).
    """

    def __init__(self, artifact: RowStoreArtifact, *, side: str, pert_col: str, control,
                 output_space: str = "scaled_log1p", target_sum: float = 1e4):
        if side not in ("real", "pred"):
            raise ValueError(f"side must be 'real' or 'pred', got {side!r}")
        if output_space not in ("scaled_log1p", "raw"):
            raise ValueError(f"output_space must be 'scaled_log1p' or 'raw', got {output_space!r}")
        self.artifact = artifact
        self.side = side
        self.pert_col = pert_col
        self.control = str(control)
        self.output_space = output_space
        self.target_sum = float(target_sum)
        self.n_genes = int(artifact.n_genes)
        self._dat_path = artifact.real_path if side == "real" else artifact.pred_path
        # deterministic per-source tag for the reference-bundle fingerprint (see the core's
        # PertBatchSource contract in partition_inmem.py) -- the .dat file backing this side.
        self.stream_tag = self._dat_path
        # The .dat is dense on disk, but real single-cell data is ~94% zeros (Tahoe 5.9-8.7%), so
        # build CSR batches unless this store is unusually dense. CSR costs 8 B/nnz vs 4 B/elem
        # dense -> the memory break-even is 50%; SPARSE_DENSITY_MAX=0.40 keeps a margin so a dense
        # store degrades to the old dense path instead of regressing. Sampled ONCE per source.
        self.density = estimate_density(
            self._dat_path, dtype=artifact.dtype, n_rows=artifact.n_rows, n_genes=self.n_genes)
        self.use_sparse = self.density < SPARSE_DENSITY_MAX
        self._var_names = np.load(artifact.var_names_path, allow_pickle=True).astype(str)
        labels = (pl.read_csv(artifact.obs_path, columns=["perturbation"])["perturbation"]
                  .to_numpy().astype(str))
        if labels.shape[0] != artifact.n_rows:
            raise ValueError(
                f"artifact {artifact.artifact_id}: obs has {labels.shape[0]} rows, "
                f"plan n_rows={artifact.n_rows}")
        self._labels = labels
        order = np.argsort(labels, kind="stable")   # stable -> ascending original-row order/group
        uniq, first, counts = np.unique(labels[order], return_index=True, return_counts=True)
        self._idx_by_label = {u: order[first[k]:first[k] + counts[k]] for k, u in enumerate(uniq)}
        # np.unique returns ascending uniq -> deterministic non-control order (matches h5ad path)
        self._counts = [(str(u), int(c)) for u, c in zip(uniq, counts)]

    def var_names(self) -> list[str]:
        return self._var_names.tolist()

    def _make_ad(self, rows: np.ndarray) -> "ad.AnnData":
        rows = np.sort(np.asarray(rows))            # ascending -> efficient memmap read
        mm = np.memmap(self._dat_path, mode="r", dtype=self.artifact.dtype,
                       shape=(self.artifact.n_rows, self.n_genes))
        raw = np.asarray(mm[rows])                  # materialize the slice (detaches from memmap)
        del mm
        if self.output_space == "scaled_log1p":
            x = (scaled_log1p_csr(raw, self.target_sum) if self.use_sparse
                 else scaled_log1p(raw, self.target_sum))
        else:
            x = (to_csr_f32(raw) if self.use_sparse
                 else np.ascontiguousarray(raw, dtype=np.float32))
        del raw                      # drop the dense slice before AnnData takes ownership of X
        obs = pd.DataFrame({self.pert_col: self._labels[rows]})
        var = pd.DataFrame(index=pd.Index(self._var_names, name="gene"))
        return ad.AnnData(X=x, obs=obs, var=var)

    def read_control_block(self) -> "ad.AnnData":
        if self.control not in self._idx_by_label:
            raise ValueError(
                f"no control rows for {self.control!r} in {self._dat_path!r}")
        return self._make_ad(self._idx_by_label[self.control])

    def iter_pert_batches(self, mem_budget) -> "Iterator[tuple[list[str], ad.AnnData]]":
        from .h5ad_manifest import plan_pert_batches
        control_cells = dict(self._counts).get(self.control, 0)
        sizes = [(u, c) for u, c in self._counts if u != self.control]
        # resident batch is float32 after scaled_log1p -> itemsize=4 (matches the h5ad path)
        batches = plan_pert_batches(
            sizes, n_genes=self.n_genes, itemsize=4, control_cells=control_cells,
            mem_budget=mem_budget)
        for batch_perts in batches:
            rows = np.concatenate([self._idx_by_label[p] for p in batch_perts])
            yield batch_perts, self._make_ad(rows)


def _resolve_artifact_target_sum(art, cfg) -> float:
    """The one normalization target for one row-store artifact (#155).

    ``float(cfg.target_sum) if cfg.target_sum else 1e4`` was a truthiness test, so ``None``
    silently became ``1e4`` for the decode while ``cfg.target_sum`` stayed ``None`` for the DE
    call downstream. Each context is an independent scoring unit with its own control, so the
    median is resolved per artifact, from the real side's RAW control rows -- the store is raw
    counts on disk regardless of the requested ``output_space``.
    """
    if cfg.target_sum is not None:
        return float(cfg.target_sum)
    probe = RowStoreBatchSource(art, side="real", pert_col=cfg.pert_col,
                                control=art.control_value, output_space="raw", target_sum=1.0)
    return resolve_target_sum(probe.read_control_block(), input_type="counts", target_sum=None)


def score_rowstore(staging_dir, *, config=None, mem_budget, output_space="scaled_log1p",
                   outdir=None):
    """Score every (dataset, panel, context) artifact in a ROW STORE out-of-core.

    Reads the row-store staging dir directly (``read_rowstore_plan`` + ``RowStoreBatchSource``)
    and drives the SP2 streaming split (real reference + pred-control reference + per-batch
    ``score_piece`` + ``aggregate_partials``) -- the same machinery ``score_h5ad_manifest`` uses
    for .h5ad pairs. ``output_space='scaled_log1p'`` (the row-store convention) fixes
    ``input_type='lognorm'``; no autodetect/peek is needed (we produced the space).

    ``config`` defaults to the ``cell-eval-0.7.6`` preset (upstream cell-eval 0.7.6 reproduction).
    Returns per-perturbation / per-context / overall tidy frames; optionally CSV-dumps to ``outdir``.
    """
    from dataclasses import replace as _replace

    from .partition import aggregate_partials
    from .partition_inmem import (_RefBundle, _build_pred_control_reference_core,
                                  _build_reference_streaming_core, score_piece)
    from .h5ad_manifest import _assemble_score_result, _nsig_names

    cfg = config if config is not None else EvalConfig.from_preset("cell-eval-0.7.6")
    if cfg.pert_col == "target":                       # row-store obs uses "perturbation"
        cfg = _replace(cfg, pert_col="perturbation")
    if output_space == "scaled_log1p":
        itype = "lognorm"
    elif output_space == "raw":
        itype = cfg.input_type
    else:
        raise ValueError(f"output_space must be 'scaled_log1p' or 'raw', got {output_space!r}")

    arts = read_rowstore_plan(staging_dir)
    per_pert_frames = []
    for art in arts:
        art_target_sum = _resolve_artifact_target_sum(art, cfg)
        art_cfg = _replace(cfg, target_sum=art_target_sum)
        comparator = _norm.resolve_comparator(
            version=art_cfg.version, pred_input_type=itype, real_input_type=itype,
        )
        real_src = RowStoreBatchSource(art, side="real", pert_col=cfg.pert_col,
                                       control=art.control_value, output_space=output_space,
                                       target_sum=art_target_sum)
        pred_src = RowStoreBatchSource(art, side="pred", pert_col=cfg.pert_col,
                                       control=art.control_value, output_space=output_space,
                                       target_sum=art_target_sum)
        with tempfile.TemporaryDirectory() as ref_dir, tempfile.TemporaryDirectory() as parts_dir:
            _build_reference_streaming_core(real_src, config=art_cfg, cache_dir=ref_dir,
                                            mem_budget=mem_budget, input_type=itype,
                                            comparator=comparator)
            if cfg.control_source == "pred":
                _build_pred_control_reference_core(pred_src, config=art_cfg, cache_dir=ref_dir,
                                                   input_type=itype, comparator=comparator)
            k = 0
            # art_cfg, NOT cfg: it carries this artifact's resolved target_sum (#155), and the
            # bundle's config_hash check inside score_piece compares against exactly this object.
            piece_cfg = _replace(art_cfg, control=art.control_value, input_type=itype)
            bundle = _RefBundle(ref_dir, piece_cfg)   # one per artifact; ref_dir is stable (#153)
            try:
                for batch_perts, batch_ad in pred_src.iter_pert_batches(mem_budget):
                    score_piece(batch_ad, ref_dir, config=piece_cfg,
                                piece_id=f"{art.context}_{k}", partial_out=parts_dir,
                                bundle=bundle, comparator=comparator)
                    k += 1
            finally:
                # Drop this artifact's cached control pool / DE table before aggregation AND
                # before the NEXT artifact's reference build. A Python local outlives the `with`
                # block it was assigned in, so without this two contexts' control pools are
                # resident at once -- a real OOM risk at scale (#153).
                del bundle
            nsig_metric, nsig_real, nsig_pred = _nsig_names(art_cfg)
            full, _agg = aggregate_partials(
                parts_dir, reduce_nsig_spearman=True, nsig_spearman_metric=nsig_metric,
                nsig_real_metric=nsig_real, nsig_pred_metric=nsig_pred)
        per_pert_frames.append(full.with_columns(
            pl.lit(art.dataset).alias("dataset"),
            pl.lit(art.panel_id).alias("panel_id"),
            pl.lit(art.context).alias("context"),
        ))

    res = _assemble_score_result(per_pert_frames)
    if outdir is not None:
        os.makedirs(outdir, exist_ok=True)
        res.per_pert.write_csv(os.path.join(outdir, "per_pert.csv"))
        res.per_context.write_csv(os.path.join(outdir, "per_context.csv"))
        res.overall.write_csv(os.path.join(outdir, "overall.csv"))
    return res
