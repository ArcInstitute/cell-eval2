import anndata as ad
import numpy as np
import pandas as pd
import pytest
from scipy.sparse import csr_matrix, issparse

from cell_eval2.prep import delta, pseudobulk


def test_pseudobulk_means_sorted():
    X = np.array([[0.0, 2.0], [2.0, 4.0], [10.0, 10.0]], dtype=np.float64)
    obs = pd.DataFrame({"target": ["A", "A", "B"]}, index=["c0", "c1", "c2"])
    var = pd.DataFrame(index=["g0", "g1"])
    perts, means = pseudobulk(ad.AnnData(X=X, obs=obs, var=var), "target")
    assert list(perts) == ["A", "B"]               # sorted
    assert np.allclose(means[0], [1.0, 3.0])        # mean of A's cells
    assert np.allclose(means[1], [10.0, 10.0])      # mean of B's cell


def test_pseudobulk_sparse_matches_dense():
    X = np.array([[0.0, 2.0], [2.0, 4.0], [10.0, 10.0]], dtype=np.float64)
    obs = pd.DataFrame({"target": ["A", "A", "B"]}, index=["c0", "c1", "c2"])
    var = pd.DataFrame(index=["g0", "g1"])
    d_perts, d_means = pseudobulk(ad.AnnData(X=X, obs=obs, var=var), "target")
    s_perts, s_means = pseudobulk(ad.AnnData(X=csr_matrix(X), obs=obs, var=var), "target")
    assert list(d_perts) == list(s_perts)
    assert np.allclose(d_means, s_means)


def test_delta_subtracts_control_and_drops_it():
    means = np.array([[3.0, 0.0],    # A
                      [0.0, 4.0],    # B
                      [1.0, 1.0]],   # ctrl
                     dtype=np.float64)
    perts = np.array(["A", "B", "ctrl"])
    out_perts, eff = delta(means, perts, "ctrl")
    assert list(out_perts) == ["A", "B"]                 # control row dropped
    assert np.allclose(eff, [[2.0, -1.0], [-1.0, 3.0]])  # mean - ctrl
    assert eff.dtype == np.float64


def test_delta_missing_control_raises():
    means = np.array([[1.0], [2.0]], dtype=np.float64)
    perts = np.array(["A", "B"])
    with pytest.raises(ValueError, match="control"):
        delta(means, perts, "ctrl")


def test_delta_shape_mismatch_raises():
    means = np.array([[1.0], [2.0], [3.0]], dtype=np.float64)  # 3 rows
    perts = np.array(["A", "ctrl"])                            # 2 labels
    with pytest.raises(ValueError, match="align 1:1"):
        delta(means, perts, "ctrl")


def _ref_pseudobulk(adata, pert_col):
    """Pre-change reference implementation, inlined, to assert bit-identity."""
    X = adata.X
    sparse = issparse(X)
    if not sparse:
        X = np.asarray(X, dtype=np.float64)
    labels = adata.obs[pert_col].to_numpy().astype(str)
    perts = np.unique(labels)
    means = np.zeros((perts.size, X.shape[1]), dtype=np.float64)
    for i, p in enumerate(perts):
        group = X[labels == p]
        means[i] = (np.asarray(group.astype(np.float64).mean(axis=0)).ravel()
                    if sparse else group.mean(axis=0))
    return perts, means


def _rand_adata(n, g, p, dense, seed):
    rng = np.random.default_rng(seed)
    labels = np.array([f"g{rng.integers(p)}" for _ in range(n)]).astype(str)
    base = np.abs(rng.standard_normal((n, g)))
    X = base.astype(np.float64) if dense else csr_matrix(base * (rng.random((n, g)) < 0.3))
    return ad.AnnData(X=X, obs=pd.DataFrame({"target": labels}))


@pytest.mark.parametrize("dense", [True, False])
def test_pseudobulk_bit_identical_to_reference(dense):
    # The vectorized grouping must be EXACTLY equal to the old per-mask loop (protects
    # the bit-exact VCC reproduction), not merely allclose.
    adata = _rand_adata(3000, 60, 80, dense, seed=7)
    perts, means = pseudobulk(adata, "target")
    rperts, rmeans = _ref_pseudobulk(adata, "target")
    assert np.array_equal(perts, rperts)
    np.testing.assert_array_equal(means, rmeans)


def test_pseudobulk_int_sparse_counts_bit_identical():
    rng = np.random.default_rng(11)
    X = csr_matrix((rng.integers(0, 5, size=(2000, 40))).astype(np.int64))
    labels = np.array([f"g{rng.integers(35)}" for _ in range(2000)]).astype(str)
    adata = ad.AnnData(X=X, obs=pd.DataFrame({"target": labels}))
    _, means = pseudobulk(adata, "target")
    _, rmeans = _ref_pseudobulk(adata, "target")
    np.testing.assert_array_equal(means, rmeans)
