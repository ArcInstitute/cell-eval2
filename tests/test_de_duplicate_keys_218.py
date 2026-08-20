"""Issue #218 -- the ONE seam that refuses a duplicated `(target, feature)` key, and the
compatibility break it creates.

`prep_de_side` raising is a BREAKING change for a caller currently feeding duplicated tables.
This pins the concrete route by which an ordinary AnnData reaches that state, and shows what
the run did before: a plausible number, on the 2025 competition profile's own DE metric, with
no error anywhere.
"""
from __future__ import annotations

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from cell_eval2 import EvalConfig, compute_metrics
from cell_eval2.config import DEParams

G, N = 12, 60


def _panel(duplicate_gene: bool):
    """A duplicated `var_names` entry is the concrete route: `io.validate_gene_axis` checks
    the two axes for equal length and identical order but NOT for uniqueness, and the DE
    adapters carry `var_names` straight into `feature`."""
    rng = np.random.default_rng(0)
    genes = [f"g{i}" for i in range(G)]
    if duplicate_gene:
        genes[3] = "g0"
    x = rng.poisson(20, size=(3 * N, G)).astype(np.float32)
    x[N:2 * N, 5] *= 6
    obs = pd.DataFrame({"target": ["non-targeting"] * N + ["P1"] * N + ["P2"] * N},
                       index=[f"c{i}" for i in range(3 * N)])
    var = pd.DataFrame(index=genes)
    real = ad.AnnData(sp.csr_matrix(x), obs=obs.copy(), var=var.copy())
    pred = ad.AnnData(sp.csr_matrix(x.copy()), obs=obs.copy(), var=var.copy())
    return pred, real


def _run(pred, real, *, control_source):
    # de_wilcoxon_overlap ALONE: the `vcc` profile's DE metric, and one of the metrics #218
    # measured returning a silently wrong value. Selecting a direction metric instead would
    # raise on polars' validate="1:1" and prove nothing about this seam.
    cfg = EvalConfig(metrics=["de_wilcoxon_overlap"], pert_col="target",
                     control="non-targeting", device="cpu", control_source=control_source,
                     de=DEParams(backend="pdex"))
    return compute_metrics(pred, real, config=cfg)


def test_duplicate_var_names_reach_the_de_table_and_are_refused():
    pred, real = _panel(duplicate_gene=True)
    with pytest.raises(ValueError, match=r"duplicated \(target, feature\) key"):
        _run(pred, real, control_source="pred")


def test_the_same_panel_scores_normally_without_the_duplicate():
    """So the raise is attributable to the duplicated key rather than to anything else about
    the fixture."""
    pred, real = _panel(duplicate_gene=False)
    frame = _run(pred, real, control_source="pred")
    assert frame.height > 0


def test_computed_de_under_control_source_real_already_failed_earlier():
    """Scope of the break, stated narrowly. On THIS route -- DE computed on the CPU/pdex
    backend under `control_source="real"` -- the control-pool assembly reindexes on the gene
    axis and pandas rejects the duplicate before DE runs at all, on this branch and on `main`
    alike, so nothing that worked now fails.

    ⚠️ That is a claim about this route only, not about `control_source="real"` in general: a
    SUPPLIED pred DE table, and gpudge's external-reference path, do not go through this
    concat. The newly refused configuration this test does pin is `control_source="pred"`
    with a duplicated gene axis, above, which previously returned a plausible number on a
    profile with no metric to raise."""
    pred, real = _panel(duplicate_gene=True)
    with pytest.raises(ValueError, match="duplicate labels"):
        _run(pred, real, control_source="real")
