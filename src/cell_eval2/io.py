from __future__ import annotations

import os

import anndata as ad
import numpy as np


def load_anndata(x: ad.AnnData | str | os.PathLike, *, backed: bool = False) -> ad.AnnData:
    """Return an AnnData, reading from a path if a str/PathLike is given.

    A cell-layout cellstream archive (any extension) is auto-detected by manifest
    and materialized in-memory via the ``_cell_archive`` shim; a plain h5ad path
    uses anndata. `backed=True` applies only to the h5ad path — cell archives
    always materialize here (out-of-core cell scoring is the streaming path, not
    this loader). Already-loaded AnnData objects pass through unchanged.
    """
    if isinstance(x, (str, os.PathLike)):
        from ._cell_archive import cell_layout, materialize_cell
        if cell_layout(x):
            return materialize_cell(x)
        return ad.read_h5ad(x, backed="r") if backed else ad.read_h5ad(x)
    return x


def validate_gene_axis(pred_genes, real_genes) -> None:
    """Raise if two gene axes differ in count or in names/order.

    Extracted from ``validate_pair`` so the streaming entry points enforce exactly what the
    in-memory path enforces, and so the comparison is unit-testable on a CPU host --
    ``score_piece`` itself needs a gpudge-resolvable config to reach (ultrareview 2026-07-25).
    """
    pred_genes = np.asarray(pred_genes, dtype=str)
    real_genes = np.asarray(real_genes, dtype=str)
    if pred_genes.size != real_genes.size:
        raise ValueError(
            f"gene dimension mismatch: pred {pred_genes.size} != real {real_genes.size}"
        )
    if not np.array_equal(pred_genes, real_genes):
        raise ValueError("gene names/order differ between pred and real")


def validate_pair(
    pred: ad.AnnData, real: ad.AnnData, *, pert_col: str, control: str
) -> None:
    """Validate structural compatibility of a (pred, real) AnnData pair."""
    validate_gene_axis(pred.var.index.values, real.var.index.values)
    for name, adata in (("pred", pred), ("real", real)):
        if pert_col not in adata.obs.columns:
            raise ValueError(f"perturbation column '{pert_col}' missing from {name}.obs")
    pred_perts = np.unique(pred.obs[pert_col].to_numpy().astype(str))
    real_perts = np.unique(real.obs[pert_col].to_numpy().astype(str))
    if not np.array_equal(pred_perts, real_perts):
        only_pred = sorted(set(pred_perts) - set(real_perts))
        only_real = sorted(set(real_perts) - set(pred_perts))
        raise ValueError(
            f"perturbation sets differ: {len(pred_perts)} pred vs {len(real_perts)} real; "
            f"pred-only ({len(only_pred)})={only_pred[:20]}, "
            f"real-only ({len(only_real)})={only_real[:20]}"
        )
    if control not in real_perts:
        raise ValueError(f"control perturbation '{control}' not found in data: {real_perts}")
