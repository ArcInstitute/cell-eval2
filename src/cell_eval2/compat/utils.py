from __future__ import annotations

import anndata as ad


def split_anndata_on_celltype(adata: ad.AnnData, celltype_col: str) -> dict[str, ad.AnnData]:
    """Split an AnnData into one object per cell-type label."""
    if celltype_col not in adata.obs.columns:
        raise ValueError(f"celltype column '{celltype_col}' not found in adata.obs")
    return {
        str(ct): adata[adata.obs[celltype_col] == ct].copy()
        for ct in adata.obs[celltype_col].unique()
    }
