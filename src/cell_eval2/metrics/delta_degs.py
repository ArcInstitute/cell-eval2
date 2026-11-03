"""Expression-delta correlation on reference-significant, off-target genes."""

from __future__ import annotations

import logging

import numpy as np
import polars as pl

from ..de import prep_de_side
from ..real_gate import DEFAULT_MIN_GATE_SIZE
from ..safety import safe_pearson
from .delta import _delta_eval, _exclusion_cols

logger = logging.getLogger(__name__)

#: The fewest reference-significant, off-target genes a perturbation is scored on: the default
#: floor of the gated metrics (`real_gate.DEFAULT_MIN_GATE_SIZE`, issue #391). A Pearson
#: correlation over 2 genes is +/-1 whatever the prediction, and over 3 its null SD is ~0.7.
MIN_GENES = DEFAULT_MIN_GATE_SIZE


def pearson_delta_degs(
    *, pred_bulk, real_bulk, genes, de_real,
    control="non-targeting", control_source="pred", target_gene_map=None,
    p_adj_threshold=0.05, min_abs_log2fc=0.0, nan_lfc_policy="mask",
):
    """Pearson correlation of deltas restricted to the reference's significant genes.

    Bulks must already be in the run's resolved expression comparator. Selection uses
    only real DE, with the usual significance/LFC thresholds and target-gene exclusion.
    Fewer than ``MIN_GENES`` (10) eligible genes or a constant/nonfinite reference delta is
    unscorable (NaN). A constant/nonfinite prediction on a scoreable reference is charged -1.
    """
    if de_real is None:
        raise ValueError("delta_pearson_degs requires a real DE table")
    real_de, de_labels = prep_de_side(
        de_real, name="real", sort_by="p_adj", nan_lfc_policy=nan_lfc_policy,
        min_abs_log2fc=min_abs_log2fc,
    )
    labels = [str(p) for p in pred_bulk[0] if str(p) != control]
    missing = sorted(set(labels) - set(de_labels))
    if missing:
        raise ValueError(f"delta_pearson_degs: real DE table missing perturbations {missing}")
    genes = np.asarray(genes, dtype=str)
    # A table that names genes differently from the panel (Ensembl IDs against symbols, say)
    # would leave every row below the floor and the metric NaN with only a warning. Raise
    # instead, like the zero-resolve target-gene check (#248).
    if not np.isin(genes, real_de["feature"].drop_nulls().unique().to_list()).any():
        raise ValueError(
            "delta_pearson_degs: no feature of the real DE table is a measured gene, so no "
            "perturbation can be scored; the DE table and the expression panel must name "
            "genes the same way"
        )
    excluded = _exclusion_cols(
        labels, genes, target_gene_map=target_gene_map,
        gate_labels=[str(p) for p in real_bulk[0] if str(p) != control],
        who="delta_pearson_degs", n_genes=real_bulk[1].shape[1],
    )
    significant = {
        label: set(features)
        for label, features in real_de.filter(pl.col("p_adj") < p_adj_threshold)
        .group_by("target").agg(pl.col("feature")).iter_rows()
    }
    # Reuse the ordinary delta metric's control lookup and label alignment. Keeping
    # both deltas together also permits a different reference-only gene mask per row.
    deltas = _delta_eval(
        pred_bulk, real_bulk, lambda pred, real: (pred, real),
        control=control, control_source=control_source,
    )
    result = {}
    unscorable = failed = 0
    for i, label in enumerate(labels):
        mask = np.isin(genes, list(significant.get(label, ())))
        if i in excluded:
            mask[excluded[i]] = False
        pred, real = (delta[mask] for delta in deltas[label])
        if real.size < MIN_GENES or not np.isfinite(real).all() or np.ptp(real) == 0:
            result[label] = float("nan")
            unscorable += 1
        elif not np.isfinite(pred).all() or np.ptp(pred) == 0:
            result[label] = -1.0
            failed += 1
        else:
            result[label] = float(safe_pearson(pred, real))
    if unscorable or failed:
        logger.warning(
            "delta_pearson_degs: %d reference-unscorable perturbations (NaN: fewer than %d "
            "significant off-target genes, or a constant/nonfinite real delta); "
            "%d undefined predictions on scoreable references charged -1",
            unscorable, MIN_GENES, failed,
        )
    return result
