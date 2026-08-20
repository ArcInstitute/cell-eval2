import numpy as np
import pytest
from sklearn.metrics import pairwise_distances

from cell_eval2.distances import (
    correct_excluded_gene,
    cosine_distance_from_parts,
    pairwise_full,
    pairwise_to_vector,
)


def _ref(M, v, metric):
    return pairwise_distances(M, np.asarray(v).reshape(1, -1), metric=metric).flatten()


@pytest.mark.parametrize("metric", ["l1", "l2", "cosine"])
def test_matches_sklearn(metric):
    rng = np.random.default_rng(0)
    M = rng.normal(size=(6, 5))
    M[2] = 0.0  # a zero-norm row
    v = rng.normal(size=5)
    got = pairwise_to_vector(M, v, metric)
    assert np.allclose(got, _ref(M, v, metric), rtol=1e-12, atol=1e-12)


def test_cosine_zero_norm_vector_is_distance_one():
    M = np.array([[1.0, 2.0, 3.0], [0.0, 0.0, 0.0], [-1.0, 0.5, 2.0]])
    z = np.zeros(3)
    got = pairwise_to_vector(M, z, "cosine")
    assert np.allclose(got, [1.0, 1.0, 1.0])
    assert np.allclose(got, _ref(M, z, "cosine"))


def test_unknown_metric_raises():
    with pytest.raises(ValueError, match="metric"):
        pairwise_to_vector(np.zeros((2, 2)), np.zeros(2), "chebyshev")


@pytest.mark.parametrize("metric", ["l1", "l2", "cosine"])
def test_pairwise_full_matches_sklearn(metric):
    rng = np.random.default_rng(1)
    A = rng.normal(size=(7, 5))
    A[3] = 0.0  # zero-norm pred row
    B = rng.normal(size=(6, 5))
    B[2] = 0.0  # zero-norm real row
    got = pairwise_full(A, B, metric)
    exp = pairwise_distances(A, B, metric=metric)
    assert got.shape == (7, 6)
    assert np.allclose(got, exp, rtol=1e-10, atol=1e-10)


def test_cosine_distance_from_parts_matches_pairwise_full():
    # single-source-of-truth: the helper that discrimination_score's cosine+exclude
    # path uses to build the base distance must reproduce pairwise_full exactly.
    rng = np.random.default_rng(33)
    G = 6
    pred = rng.normal(size=(9, G))
    pred[2] = 0.0  # zero-norm row
    real = rng.normal(size=(7, G))
    real[4] = 0.0  # zero-norm row
    sim = pred @ real.T
    pnsq = np.einsum("ig,ig->i", pred, pred)
    rnsq = np.einsum("jg,jg->j", real, real)
    got = cosine_distance_from_parts(sim, pnsq, rnsq)
    exp = pairwise_full(pred, real, "cosine")
    assert np.allclose(got, exp, rtol=1e-12, atol=1e-12)


def test_pairwise_full_unknown_metric_raises():
    with pytest.raises(ValueError, match="metric"):
        pairwise_full(np.zeros((2, 3)), np.zeros((2, 3)), "chebyshev")


@pytest.mark.parametrize("metric", ["l1", "l2", "cosine"])
def test_correct_excluded_gene_equals_column_drop(metric):
    rng = np.random.default_rng(2)
    P, G = 5, 4
    pred = rng.normal(size=(P, G))
    real = rng.normal(size=(P, G))
    col = 1
    full = pairwise_full(pred, real, metric)
    for i in range(P):
        correct_excluded_gene(full, pred, real, metric, i, col)
    keep = np.delete(np.arange(G), col)
    exp = pairwise_full(pred[:, keep], real[:, keep], metric)
    assert np.allclose(full, exp, rtol=1e-9, atol=1e-9)


def test_correct_excluded_gene_cosine_fast_path_matches_slow():
    # precomputed real_norm_squares + pred_norm_squares + sim (the O(P) caller
    # path) must equal the self-contained recompute path exactly.
    rng = np.random.default_rng(12)
    P, G, col = 6, 5, 2
    pred = rng.normal(size=(P, G))
    real = rng.normal(size=(P, G))
    slow = pairwise_full(pred, real, "cosine")
    fast = slow.copy()
    rns = np.einsum("jg,jg->j", real, real)
    pns = np.einsum("ig,ig->i", pred, pred)
    sim = pred @ real.T
    for i in range(P):
        correct_excluded_gene(slow, pred, real, "cosine", i, col)
        correct_excluded_gene(
            fast, pred, real, "cosine", i, col,
            real_norm_squares=rns, pred_norm_squares=pns, sim=sim,
        )
    assert np.allclose(fast, slow, rtol=1e-12, atol=1e-12)


def test_correct_excluded_gene_l1_non_negative():
    # contract: corrected l1 distances stay >= 0 (the clip guards summation roundoff
    # on near-identical rows). The engineered pair is identical on all kept genes, so
    # its reduced l1 is exactly 0 -- never negative.
    rng = np.random.default_rng(21)
    P, G, col = 8, 6, 3
    pred = rng.normal(size=(P, G))
    real = rng.normal(size=(P, G))
    real[0] = pred[0]
    real[0, col] += 0.5  # identical to pred[0] except the dropped gene
    full = pairwise_full(pred, real, "l1")
    for i in range(P):
        correct_excluded_gene(full, pred, real, "l1", i, col)
    assert (full >= 0.0).all()
    assert full[0, 0] == pytest.approx(0.0, abs=1e-12)


def test_correct_excluded_gene_unknown_metric_raises():
    d = np.zeros((2, 2))
    with pytest.raises(ValueError, match="metric"):
        correct_excluded_gene(d, np.zeros((2, 3)), np.zeros((2, 3)),
                              "chebyshev", 0, 0)
