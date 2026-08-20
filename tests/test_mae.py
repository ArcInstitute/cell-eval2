import numpy as np
import pytest

from cell_eval2.metrics import mae
from cell_eval2.prep import pseudobulk


def test_mae_excludes_control_and_values(synthetic_pair):
    pred, real = synthetic_pair
    out = mae(pred=pred, real=real, pert_col="target", control="non-targeting")
    assert "non-targeting" not in out
    assert set(out) == {"GENE1", "GENE2", "GENE3"}
    assert all(v >= 0 for v in out.values())


def test_mae_hybrid_equivalence(synthetic_pair):
    pred, real = synthetic_pair
    from_anndata = mae(pred=pred, real=real, pert_col="target", control="non-targeting")
    from_bulk = mae(pred_bulk=pseudobulk(pred, "target"),
                    real_bulk=pseudobulk(real, "target"),
                    pert_col="target", control="non-targeting")
    assert from_anndata == from_bulk


def test_mae_requires_inputs():
    with pytest.raises(ValueError, match="provide"):
        mae(pert_col="target", control="non-targeting")


def test_mae_missing_pert_raises():
    pred_bulk = (np.array(["non-targeting", "GENE1"]), np.array([[0.0, 0.0], [1.0, 1.0]]))
    real_bulk = (np.array(["non-targeting", "GENE2"]), np.array([[0.0, 0.0], [2.0, 2.0]]))
    with pytest.raises(ValueError, match="missing from real_bulk"):
        mae(pred_bulk=pred_bulk, real_bulk=real_bulk,
            pert_col="target", control="non-targeting")


def test_mae_ignores_extra_real_perts():
    pred_bulk = (np.array(["non-targeting", "GENE1"]),
                 np.array([[0.0, 0.0], [1.0, 1.0]]))
    real_bulk = (np.array(["non-targeting", "GENE1", "GENE2"]),
                 np.array([[0.0, 0.0], [1.0, 1.0], [5.0, 5.0]]))
    out = mae(pred_bulk=pred_bulk, real_bulk=real_bulk,
              pert_col="target", control="non-targeting")
    assert set(out) == {"GENE1"}          # real-only GENE2 ignored; control excluded
    assert out["GENE1"] == 0.0
