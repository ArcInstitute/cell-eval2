import pytest

from cell_eval2.io import load_anndata, validate_pair


def test_load_passthrough(synthetic_pair):
    pred, _ = synthetic_pair
    assert load_anndata(pred) is pred


def test_validate_pair_ok(synthetic_pair):
    pred, real = synthetic_pair
    validate_pair(pred, real, pert_col="target", control="non-targeting")


def test_validate_pair_gene_mismatch(synthetic_pair):
    pred, real = synthetic_pair
    pred2 = pred[:, ::-1].copy()  # reversed gene order
    with pytest.raises(ValueError, match="gene"):
        validate_pair(pred2, real, pert_col="target", control="non-targeting")


def test_validate_pair_missing_control(synthetic_pair):
    pred, real = synthetic_pair
    with pytest.raises(ValueError, match="control"):
        validate_pair(pred, real, pert_col="target", control="absent-ctrl")


def test_load_anndata_backed_from_path(tmp_path, synthetic_pair):
    pred, _ = synthetic_pair
    p = tmp_path / "pred.h5ad"
    pred.write_h5ad(p)
    backed = load_anndata(str(p), backed=True)
    assert backed.isbacked
    assert backed.n_obs == pred.n_obs
    assert list(backed.var.index) == list(pred.var.index)  # metadata available


def test_load_anndata_inmemory_passthrough(synthetic_pair):
    pred, _ = synthetic_pair
    assert load_anndata(pred, backed=True) is pred  # objects pass through unchanged


def test_validate_pair_pert_mismatch_message_is_truncated():
    import numpy as np
    import anndata as ad
    import pandas as pd
    import pytest
    from cell_eval2.io import validate_pair

    genes = ["g0", "g1"]

    def _ad(perts):
        n = len(perts)
        return ad.AnnData(
            X=np.zeros((n, 2), dtype=np.float32),
            obs=pd.DataFrame({"perturbation": perts}, index=[f"c{i}" for i in range(n)]),
            var=pd.DataFrame(index=genes),
        )

    pred = _ad(["ctrl", "A", "B", "EXTRA_PRED"])
    real = _ad(["ctrl", "A", "B", "ONLY_REAL"])
    with pytest.raises(ValueError) as exc:
        validate_pair(pred, real, pert_col="perturbation", control="ctrl")
    msg = str(exc.value)
    assert "perturbation sets differ" in msg
    assert "EXTRA_PRED" in msg and "ONLY_REAL" in msg     # symmetric diff reported
    assert "pred-only" in msg and "real-only" in msg      # diff format, not raw arrays
