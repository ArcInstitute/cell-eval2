import numpy as np
import pytest
import scipy.sparse as sp

from cell_eval2 import noise


def _blocks():
    rng = np.random.default_rng(7)
    X = sp.csr_matrix(rng.poisson(1.0, size=(10, 5)).astype(np.float32))
    yield X, np.array(["A"] * 5 + ["B"] * 5)


def test_level_zero_is_identity():
    ((x0, l0),) = list(_blocks())
    ((x1, l1),) = list(noise.noise_blocks(_blocks(), kind="gaussian", level=0.0, seed=0))
    np.testing.assert_array_equal(x0.todense(), x1.todense())
    np.testing.assert_array_equal(l0, l1)


def test_deterministic_same_seed():
    a = [x.todense() for x, _ in noise.noise_blocks(_blocks(), kind="gaussian", level=0.5, seed=3)]
    b = [x.todense() for x, _ in noise.noise_blocks(_blocks(), kind="gaussian", level=0.5, seed=3)]
    for u, v in zip(a, b):
        np.testing.assert_array_equal(u, v)


def test_different_seed_differs():
    a = [x.todense() for x, _ in noise.noise_blocks(_blocks(), kind="gaussian", level=0.5, seed=3)]
    b = [x.todense() for x, _ in noise.noise_blocks(_blocks(), kind="gaussian", level=0.5, seed=4)]
    assert any(not np.array_equal(u, v) for u, v in zip(a, b))


def test_downsample_reduces_counts():
    ((x,),) = [[x for x, _ in noise.noise_blocks(_blocks(), kind="downsample", level=0.5, seed=1)]]
    ((orig,),) = [[x for x, _ in _blocks()]]
    assert x.sum() <= orig.sum()


def test_downsample_rejects_fractional():
    rng = np.random.default_rng(0)
    X = sp.csr_matrix((rng.random(size=(4, 3)) + 0.1).astype(np.float32))

    def gen():
        yield X, np.array(["A"] * 4)

    with pytest.raises(ValueError, match="integer count data"):
        list(noise.noise_blocks(gen(), kind="downsample", level=0.5, seed=0))


def test_bad_kind_raises():
    with pytest.raises(ValueError, match="kind must be one of"):
        list(noise.noise_blocks(_blocks(), kind="nope", level=0.5, seed=0))


def test_negative_level_raises():
    with pytest.raises(ValueError, match="level must be >= 0"):
        list(noise.noise_blocks(_blocks(), kind="gaussian", level=-1.0, seed=0))


def test_noise_hook_level_zero_identity_in_pseudobulk(tmp_path):
    pytest.importorskip("cellstream")
    import anndata as ad
    import pandas as pd
    from cellstream import write_sharded

    from cell_eval2 import streaming_bulk

    rng = np.random.default_rng(9)
    X = sp.csr_matrix(rng.poisson(0.5, size=(40, 6)).astype(np.float32))
    obs = pd.DataFrame({"target": np.repeat(["non-targeting", "A", "B", "C"], 10)})
    adata = ad.AnnData(X=X, obs=obs, var=pd.DataFrame(index=[f"g{i}" for i in range(6)]))
    path = str(tmp_path / "f.shad")
    write_sharded(adata, path, group_by="target")

    plain = streaming_bulk.streaming_pseudobulk(
        path, pert_col="target", norms=["counts", "lognorm"], target_sum=1e6
    )
    noised = streaming_bulk.streaming_pseudobulk(
        path, pert_col="target", norms=["counts", "lognorm"], target_sum=1e6,
        noise={"kind": "gaussian", "level": 0.0, "seed": 0},
    )
    for n in ("counts", "lognorm"):
        np.testing.assert_array_equal(plain[n][0], noised[n][0])
        np.testing.assert_array_equal(plain[n][1], noised[n][1])


def test_noise_block_matches_noise_blocks():
    """noise_block (per-block helper) reproduces noise_blocks' per-shard output exactly, and
    level-0 is an identity passthrough (same object)."""
    blocks = list(_blocks())
    via_stream = [x for x, _ in noise.noise_blocks(iter(blocks), kind="downsample", level=0.5, seed=7)]
    via_block = [noise.noise_block(x, kind="downsample", level=0.5, seed=7, shard_idx=i)
                 for i, (x, _) in enumerate(blocks)]
    for a, b in zip(via_stream, via_block):
        np.testing.assert_array_equal(a.todense(), b.todense())
    x0 = blocks[0][0]
    assert noise.noise_block(x0, kind="gaussian", level=0.0, seed=1, shard_idx=0) is x0


def test_downsample_does_not_mutate_input():
    """eliminate_zeros() on the downsample path must not corrupt the caller's matrix
    (regression: it mutated shared indices/indptr in place -- PR #56 review)."""
    import numpy as np
    import scipy.sparse as sp

    from cell_eval2 import noise

    X = sp.csr_matrix(np.array([[5, 0, 3], [0, 2, 0]], dtype=np.float32))
    before = (X.data.copy(), X.indices.copy(), X.indptr.copy())
    list(noise.noise_blocks(iter([(X, np.array(["A", "B"]))]), kind="downsample", level=0.9, seed=1))
    np.testing.assert_array_equal(X.data, before[0])
    np.testing.assert_array_equal(X.indices, before[1])
    np.testing.assert_array_equal(X.indptr, before[2])


def test_gaussian_noise_int_dtype_rounds_and_clips_no_overflow():
    # F7.1: gaussian noise on an integer count dtype must round (not truncate toward zero via
    # .astype) and clip to the dtype max (not silently wrap on overflow). Replicate the code's RNG
    # (seed ^ shard_idx) to assert the exact rounded+clipped output; a value near uint16 max would
    # wrap to a tiny int under the buggy .astype truncation.
    data = np.array([65500, 3, 100, 40000], dtype=np.uint16)  # 65500*~1.001 overflows uint16 -> clip
    X = sp.csr_matrix((data, np.array([0, 1, 2, 3]), np.array([0, 4])), shape=(1, 4))
    seed, shard, level = 7, 0, 1.0
    out = noise.noise_block(X, kind="gaussian", level=level, seed=seed, shard_idx=shard)
    rng = np.random.default_rng(np.uint64(seed) ^ np.uint64(shard))
    factors = np.exp(rng.normal(0.0, level, size=data.shape))
    info = np.iinfo(np.uint16)
    expected = np.clip(np.rint(data * factors), info.min, info.max).astype(np.uint16)
    np.testing.assert_array_equal(out.data, expected)
    assert out.data.dtype == np.uint16
    assert (out.data <= info.max).all()      # no wraparound
