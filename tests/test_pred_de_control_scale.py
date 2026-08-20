"""control_source='real' must put the substituted real control on the SAME scale as the
predictions before they are combined for pred-side DE. If predictions are log-norm, the real
controls (raw counts) must be log-normed too — otherwise the combined matrix mixes scales
(log-norm pred + raw-counts control), distorting DE / tripping the scale-limit gate."""
import dataclasses

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

from cell_eval2 import EvalConfig
from cell_eval2.run import _pred_de_input


def _lognorm_pred_counts_real():
    genes = [f"g{i}" for i in range(4)]
    labels = ["g1", "g1", "g2", "g2", "non-targeting", "non-targeting"]
    obs = pd.DataFrame({"target_gene": labels})
    var = pd.DataFrame(index=genes)
    # pred: fractional (log-norm scale, small) -> v1 auto-detects 'lognorm'
    pred_X = np.array([[0.5, 1.2, 0.0, 2.1], [0.6, 1.0, 0.1, 2.0],
                       [1.5, 0.2, 0.3, 0.0], [1.4, 0.3, 0.2, 0.1],
                       [0.9, 0.9, 0.9, 0.9], [0.8, 1.0, 1.0, 0.7]], dtype=np.float32)
    # real: integer counts, large enough that a lognorm mis-detection would trip the scale gate
    real_X = np.array([[10, 50, 0, 80], [12, 45, 1, 75],
                       [60, 5, 9, 0], [55, 7, 8, 2],
                       [40, 60, 30, 90], [38, 62, 28, 88]], dtype=np.float32)
    pred = ad.AnnData(X=sp.csr_matrix(pred_X), obs=obs.copy(), var=var.copy())
    real = ad.AnnData(X=sp.csr_matrix(real_X), obs=obs.copy(), var=var.copy())
    return pred, real


def test_pred_de_input_matches_real_control_to_pred_lognorm_scale(monkeypatch):
    import cell_eval2.run as run
    monkeypatch.setattr(run, "_use_inmem_external_ref", lambda cfg: False)  # exercise the concat path
    pred, real = _lognorm_pred_counts_real()
    cfg = dataclasses.replace(EvalConfig.v1(), pert_col="target_gene",
                              control="non-targeting", control_source="real")
    combined, ref = _pred_de_input(pred, real, cfg=cfg)
    assert ref is None
    X = combined.X.toarray() if sp.issparse(combined.X) else np.asarray(combined.X)
    ctrl_mask = combined.obs["target_gene"].astype(str).to_numpy() == "non-targeting"
    ctrl_rows = X[ctrl_mask]
    # the substituted real control must now be log-norm (fractional + small), matching the pred,
    # NOT the raw integer counts it started as.
    assert ctrl_rows.size > 0
    assert not np.allclose(ctrl_rows, np.rint(ctrl_rows))   # log-normed -> fractional
    assert float(X.max()) < 20.0                            # whole combined matrix on lognorm scale


# ---- external-ref (gpudge #67) vs concat (CPU) structure: control_source='real' ----

def _counts_pred_real_v2():
    genes = [f"g{i}" for i in range(4)]
    labels = ["P1", "P1", "P2", "P2", "non-targeting", "non-targeting"]
    obs = pd.DataFrame({"target_gene": labels})
    var = pd.DataFrame(index=genes)
    rng = np.random.default_rng(1)
    pred = ad.AnnData(X=sp.csr_matrix(rng.integers(0, 50, size=(6, 4)).astype(np.float32)),
                      obs=obs.copy(), var=var.copy())
    real = ad.AnnData(X=sp.csr_matrix(rng.integers(0, 50, size=(6, 4)).astype(np.float32)),
                      obs=obs.copy(), var=var.copy())
    cfg = dataclasses.replace(EvalConfig.v2(), pert_col="target_gene",
                              control="non-targeting", control_source="real")
    return pred, real, cfg


def test_pred_de_input_returns_pieces_for_gpudge_external_ref(monkeypatch):
    import cell_eval2.run as run
    monkeypatch.setattr(run, "_use_inmem_external_ref", lambda cfg: True)
    pred, real, cfg = _counts_pred_real_v2()
    tgt, ref = run._pred_de_input(pred, real, cfg=cfg)
    assert ref is not None
    # Option A: the DE targets are the FULL pred (control group INCLUDED) — passed to gpudge as-is
    # with NO subset/copy. gpudge ranks every group vs the pool; compute_de drops the control
    # group's spurious rows (control_group=cfg.control), so the control is NOT excluded here.
    assert cfg.control in set(tgt.obs[cfg.pert_col].astype(str))          # targets: full pred incl control
    assert tgt is pred                                                    # the exact object — no subset, no copy
    assert set(ref.obs[cfg.pert_col].astype(str)) == {cfg.control}        # reference: control only
    # Neither may be a VIEW: a view pins the full parent alive and makes gpudge re-materialize X on
    # access -> ~2x host RAM -> OOM at ~5M cells (CCL_2). The full pred is a real AnnData (not a
    # view/copy); real_ctrl is a .copy() of the control slice. Regression guard.
    assert not tgt.is_view and not ref.is_view


def test_pred_de_input_concats_for_cpu_backend(monkeypatch):
    import cell_eval2.run as run
    monkeypatch.setattr(run, "_use_inmem_external_ref", lambda cfg: False)
    pred, real, cfg = _counts_pred_real_v2()
    combined, ref = run._pred_de_input(pred, real, cfg=cfg)
    assert ref is None
    assert cfg.control in set(combined.obs[cfg.pert_col].astype(str))     # concat keeps control group
