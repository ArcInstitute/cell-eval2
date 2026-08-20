import itertools

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

pytest.importorskip("cellstream")

from cell_eval2 import norm as _norm  # noqa: E402
from cell_eval2 import prep, streaming_bulk  # noqa: E402


def _write(tmp_path, seed=1, n=60, g=8, k=6, reference=None):
    import anndata as ad
    import pandas as pd
    import scipy.sparse as sp
    from cellstream import write_sharded

    rng = np.random.default_rng(seed)
    X = sp.csr_matrix(rng.poisson(0.5, size=(n, g)).astype(np.float32))
    labels = ["non-targeting", "A", "B", "C", "D", "E"][:k]
    obs = pd.DataFrame({"target": np.repeat(labels, n // k)})
    adata = ad.AnnData(X=X, obs=obs, var=pd.DataFrame(index=[f"g{i}" for i in range(g)]))
    path = str(tmp_path / "f.shad")
    write_sharded(adata, path, group_by="target", reference=reference)
    return path, adata


def _inmemory_ref(adata, target):
    # The streaming pseudobulk accumulates in float64; scanpy's normalize_total runs in the
    # X dtype (float32 here), which alone introduces ~1e-7 rounding. Cast the reference to
    # float64 so the comparison isolates the *algorithm* (sum-then-divide arithmetic mean,
    # incl. lognorm = arithmetic mean of log1p values) at machine precision, not float32 noise.
    ad64 = adata.copy()
    ad64.X = ad64.X.astype(np.float64)
    return prep.pseudobulk(_norm.to_normalization(ad64, "counts", target, target_sum=1e6), "target")


def test_streaming_pseudobulk_matches_inmemory(tmp_path):
    path, adata = _write(tmp_path)
    for target in ("counts", "normalized", "lognorm"):
        s_perts, s_means = streaming_bulk.streaming_pseudobulk(
            path, pert_col="target", norms=[target], target_sum=1e6
        )[target]
        ref = _inmemory_ref(adata, target)
        order = np.argsort(s_perts)
        np.testing.assert_array_equal(s_perts[order], ref[0])
        np.testing.assert_allclose(s_means[order], ref[1], rtol=1e-9, atol=1e-9)


def test_streaming_pseudobulk_includes_reference_shard_control(tmp_path):
    # Archive written WITH a separate reference shard (reference="non-targeting"): iter_blocks
    # deliberately EXCLUDES that shard, so streaming_pseudobulk must read it back (via
    # read_reference_block) or the control row is silently all-zeros -- which corrupts every
    # delta/discrimination metric when pred != real. Regression test for that zero-control bug.
    path, adata = _write(tmp_path, reference="non-targeting")
    for target in ("counts", "normalized", "lognorm"):
        s_perts, s_means = streaming_bulk.streaming_pseudobulk(
            path, pert_col="target", norms=[target], target_sum=1e6
        )[target]
        ref = _inmemory_ref(adata, target)
        order = np.argsort(s_perts)
        np.testing.assert_array_equal(s_perts[order], ref[0])
        np.testing.assert_allclose(s_means[order], ref[1], rtol=1e-9, atol=1e-9)
        ctrl_i = int(np.flatnonzero(s_perts == "non-targeting")[0])
        assert np.any(s_means[ctrl_i] != 0.0)  # the bug made this control row all-zeros


def test_streaming_pseudobulk_multi_norm_in_one_pass(tmp_path):
    path, adata = _write(tmp_path, seed=4)
    out = streaming_bulk.streaming_pseudobulk(
        path, pert_col="target", norms=["counts", "normalized", "lognorm"], target_sum=1e6
    )
    assert set(out) == {"counts", "normalized", "lognorm"}
    for target in out:
        s_perts, s_means = out[target]
        ref = _inmemory_ref(adata, target)
        order = np.argsort(s_perts)
        np.testing.assert_array_equal(s_perts[order], ref[0])
        np.testing.assert_allclose(s_means[order], ref[1], rtol=1e-9, atol=1e-9)


def test_streaming_pseudobulk_rejects_mismatched_pert_col(tmp_path):
    """streaming_pseudobulk must reject a pert_col that differs from the archive's group_by
    (it would silently mis-map labels -- PR #56 review)."""
    import anndata as ad
    import numpy as np
    import pandas as pd
    import pytest
    import scipy.sparse as sp
    from cellstream import write_sharded

    from cell_eval2 import streaming_bulk

    adata = ad.AnnData(
        X=sp.csr_matrix(np.ones((10, 3), dtype=np.float32)),
        obs=pd.DataFrame({"target": np.repeat(["non-targeting", "A"], 5)}),
        var=pd.DataFrame(index=["g0", "g1", "g2"]),
    )
    path = str(tmp_path / "f.shad")
    write_sharded(adata, path, group_by="target")
    with pytest.raises(ValueError, match="group_by"):
        streaming_bulk.streaming_pseudobulk(path, pert_col="WRONG", norms=["counts"], target_sum=1e6)


def test_streaming_pseudobulk_median_umi(tmp_path):
    # with_median_umi=True returns (bulks, median) where median is the exact median per-cell
    # library size over all streamed cells (no-reference archive -> every cell is streamed).
    path, adata = _write(tmp_path)
    expected = float(np.median(np.asarray(adata.X.sum(axis=1)).ravel()))
    bulks, med = streaming_bulk.streaming_pseudobulk(
        path, pert_col="target", norms=["lognorm"], target_sum=1e6, device="cpu",
        with_median_umi=True,
    )
    assert "lognorm" in bulks
    assert abs(med - expected) <= 1e-6 * abs(expected)


def test_streaming_pseudobulk_default_returns_dict_only(tmp_path):
    # Default (with_median_umi=False) keeps the existing {norm: (perts, means)} return shape.
    path, _adata = _write(tmp_path)
    out = streaming_bulk.streaming_pseudobulk(path, pert_col="target", norms=["counts"],
                                              target_sum=1e6, device="cpu")
    assert isinstance(out, dict) and "counts" in out


def test_inmem_pseudobulk_matches_cpu_reference():
    # inmem_pseudobulk (accumulator over row-blocks) == prep.pseudobulk(to_normalization(...)).
    import anndata as ad
    import numpy as np
    import pandas as pd
    import scipy.sparse as sp

    from cell_eval2 import norm as _norm
    from cell_eval2.prep import pseudobulk
    from cell_eval2.streaming_bulk import inmem_pseudobulk

    rng = np.random.default_rng(11)
    n, g = 300, 16
    X = sp.csr_matrix(rng.poisson(0.8, size=(n, g)).astype(np.float32))
    obs = pd.DataFrame({"target": rng.choice(["non-targeting", "A", "B", "C"], size=n)})
    adata = ad.AnnData(X=X, obs=obs, var=pd.DataFrame(index=[f"g{i}" for i in range(g)]))

    ref_perts, ref_means = pseudobulk(
        _norm.to_normalization(adata, "counts", "lognorm", target_sum=1e6), "target"
    )
    out = inmem_pseudobulk(adata, pert_col="target", norms=["lognorm"],
                           target_sum=1e6, device="cpu", block_rows=64)
    perts, means = out["lognorm"]
    assert list(perts) == list(ref_perts)
    assert means.dtype == np.float32
    assert np.allclose(means, ref_means, rtol=1e-4, atol=1e-6)


def test_inmem_pseudobulk_block_size_invariant():
    # The result must not depend on block_rows (one block == many blocks).
    import anndata as ad
    import numpy as np
    import pandas as pd
    import scipy.sparse as sp

    from cell_eval2.streaming_bulk import inmem_pseudobulk

    rng = np.random.default_rng(12)
    n, g = 257, 10
    X = sp.csr_matrix(rng.poisson(0.7, size=(n, g)).astype(np.float32))
    obs = pd.DataFrame({"target": rng.choice(["non-targeting", "A", "B"], size=n)})
    adata = ad.AnnData(X=X, obs=obs, var=pd.DataFrame(index=[f"g{i}" for i in range(g)]))

    one = inmem_pseudobulk(adata, pert_col="target", norms=["counts", "lognorm"],
                           target_sum=1e6, device="cpu", block_rows=10_000)
    many = inmem_pseudobulk(adata, pert_col="target", norms=["counts", "lognorm"],
                            target_sum=1e6, device="cpu", block_rows=32)
    for norm in ("counts", "lognorm"):
        assert np.allclose(one[norm][1], many[norm][1], rtol=1e-5, atol=1e-7)


def test_inmem_pseudobulk_handles_dense_X():
    # A dense X block must be accepted (converted to CSR per block).
    import anndata as ad
    import numpy as np
    import pandas as pd

    from cell_eval2.streaming_bulk import inmem_pseudobulk

    rng = np.random.default_rng(13)
    n, g = 120, 8
    X = rng.poisson(0.9, size=(n, g)).astype(np.float32)  # dense ndarray
    obs = pd.DataFrame({"target": rng.choice(["non-targeting", "A"], size=n)})
    adata = ad.AnnData(X=X, obs=obs, var=pd.DataFrame(index=[f"g{i}" for i in range(g)]))
    out = inmem_pseudobulk(adata, pert_col="target", norms=["lognorm"],
                           target_sum=1e6, device="cpu", block_rows=50)
    assert out["lognorm"][1].shape == (2, g)
    assert np.isfinite(out["lognorm"][1]).all()


def test_inmem_pseudobulk_stashes_max_row_total():
    # The accumulator computes per-cell libs anyway; inmem_pseudobulk stashes their max on the
    # adata so the scale-limit gate can skip its own _row_totals pass. Equals max(X.sum(axis=1)).
    import anndata as ad
    import pandas as pd
    import scipy.sparse as sp
    from cell_eval2.streaming_bulk import inmem_pseudobulk
    rng = np.random.default_rng(3)
    X = sp.csr_matrix(rng.poisson(0.8, size=(120, 8)).astype(np.float32))
    labels = np.array(["a", "b"] * 60)
    a = ad.AnnData(X=X, obs=pd.DataFrame({"pert": labels}, index=[f"c{i}" for i in range(X.shape[0])]),
                   var=pd.DataFrame(index=[f"g{j}" for j in range(X.shape[1])]))
    inmem_pseudobulk(a, pert_col="pert", norms=["lognorm"], target_sum=1e6, device="cpu")
    assert a._precomputed_row_total_max == pytest.approx(float(np.asarray(X.sum(axis=1)).ravel().max()))


def test_inmem_pseudobulk_no_stash_for_counts_only():
    # counts-only norm set -> accumulator never computes libs -> no stash -> gate falls back.
    import anndata as ad
    import pandas as pd
    import scipy.sparse as sp
    from cell_eval2.streaming_bulk import inmem_pseudobulk
    X = sp.csr_matrix(np.array([[1, 2], [3, 4]], dtype=np.float32))
    a = ad.AnnData(X=X, obs=pd.DataFrame({"pert": ["a", "b"]}, index=["c0", "c1"]),
                   var=pd.DataFrame(index=["g0", "g1"]))
    inmem_pseudobulk(a, pert_col="pert", norms=["counts"], target_sum=1e6, device="cpu")
    assert getattr(a, "_precomputed_row_total_max", None) is None


@pytest.mark.parametrize("cls_name", ["csr_matrix", "csr_array"])
def test_csr_row_block_matches_raw_constructor(cls_name):
    # _csr_row_block builds each block WITHOUT the CSR constructor's O(nnz) check_format
    # validation; it must match the old X.__class__((data,indices,indptr)) construction in
    # .data and in the row sums.
    #
    # NOT byte-identical in the index arrays: the helper preserves the parent's dtypes where
    # the raw constructor reconciles them (a csr_array with int32 indices + int64 indptr comes
    # back int64/int64 from the raw path). The assertions below use np.array_equal, which is
    # dtype-blind, so they do not catch that. test_norm.py's
    # test_csr_row_block_preserves_parent_index_dtypes pins both sides.
    #
    # The real resident pred X is a csr_matrix (int64 indptr / int32 indices), so cover that
    # class explicitly plus csr_array for future-proofing.
    import scipy.sparse as sp
    cls = getattr(sp, cls_name)
    from cell_eval2.streaming_bulk import _csr_row_block
    rng = np.random.default_rng(3)
    n, g = 400, 30
    dense = ((rng.random((n, g)) < 0.3) * rng.integers(1, 50, (n, g))).astype(np.uint16)
    X = cls(dense)
    X.indptr = X.indptr.astype(np.int64)   # mimic cellstream's int64 indptr / int32 indices split
    X.indices = X.indices.astype(np.int32)
    for start, stop in [(100, 300), (0, n), (50, 51)]:
        lo, hi = int(X.indptr[start]), int(X.indptr[stop])
        ref = X.__class__(
            (X.data[lo:hi], X.indices[lo:hi], X.indptr[start:stop + 1] - X.indptr[start]),
            shape=(stop - start, X.shape[1]), copy=False,
        )
        blk = _csr_row_block(X, start, stop)
        assert type(blk) is type(ref)
        assert blk.shape == ref.shape and blk.nnz == ref.nnz
        assert np.array_equal(blk.data, ref.data)
        assert np.array_equal(blk.indices, ref.indices)
        assert np.array_equal(blk.indptr, ref.indptr)
        assert np.array_equal(
            np.asarray(blk.tocsr().sum(axis=1, dtype=np.float64)).ravel(),
            np.asarray(ref.sum(axis=1, dtype=np.float64)).ravel(),
        )


@pytest.mark.parametrize("cls_name", ["csr_matrix", "csr_array"])
def test_csr_row_block_zero_nnz(cls_name):
    # A row-block with no nonzeros (lo == hi) must build cleanly and sum to all zeros.
    import scipy.sparse as sp
    cls = getattr(sp, cls_name)
    from cell_eval2.streaming_bulk import _csr_row_block
    X = cls(np.zeros((100, 5), dtype=np.uint16))
    X.indptr = X.indptr.astype(np.int64)
    X.indices = X.indices.astype(np.int32)
    blk = _csr_row_block(X, 10, 60)
    assert blk.shape == (50, 5) and blk.nnz == 0
    assert not np.asarray(blk.tocsr().sum(axis=1, dtype=np.float64)).any()


def test_shard_streaming_jk_matches_the_prep_reference(tmp_path):
    from cell_eval2.prep import pseudobulk_bulk_lognorm_with_moments

    path, adata = _write(tmp_path, n=60, g=8, k=6, reference="non-targeting")
    a64 = adata.copy()
    a64.X = a64.X.astype(np.float64)
    ref_perts, _, ref = pseudobulk_bulk_lognorm_with_moments(a64, "target",
                                                             bulk_target_sum=1e6)
    _, mom = streaming_bulk.streaming_pseudobulk(
        path, pert_col="target", norms=["bulk_lognorm"], target_sum=None,
        with_moments=True, bulk_target_sum=1e6)
    order = np.argsort(mom["bulk_lognorm"].perts)
    np.testing.assert_array_equal(np.asarray(mom["bulk_lognorm"].perts)[order], ref_perts)
    np.testing.assert_allclose(mom["bulk_lognorm"].jk[order], ref.jk, rtol=1e-8)


def test_the_jackknife_pass_sees_the_same_blocks_as_the_bulk_pass(tmp_path):
    """The two-pass trap. noise_blocks seeds per shard_idx from enumerate (noise.py:62), so a
    second pass rebuilt WITHOUT re-prepending the reference block shifts every index and
    noises different cells -- a correction for data that was never scored, with no error.

    ⚠️ `reference=` is REQUIRED here. `_write`'s default writes no separate reference block
    (test_streaming_bulk.py:10), and with no block to prepend the bug cannot fire -- rev 1's
    version of this test was green against the bug it existed for.

    A same-seed rerun is not enough either (both runs wrong identically). The oracle is the
    in-memory reference over the SAME noised cells: noise_block is a pure function of
    (X, kind, level, seed, shard_idx)."""
    from cell_eval2.noise import noise_block
    from cell_eval2.prep import pseudobulk_bulk_lognorm_with_moments
    from cell_eval2.stream import iter_blocks, read_reference_block

    path, _ = _write(tmp_path, n=60, g=8, k=6, reference="non-targeting")
    assert read_reference_block(path, pert_col="target") is not None
    noise = {"kind": "downsample", "level": 0.5, "seed": 7}
    _, mom = streaming_bulk.streaming_pseudobulk(
        path, pert_col="target", norms=["bulk_lognorm"], target_sum=None,
        with_moments=True, noise=noise, bulk_target_sum=1e6)

    blocks = itertools.chain((read_reference_block(path, pert_col="target"),),
                             iter_blocks(path, pert_col="target"))
    Xs, labs = [], []
    for i, (X, lab) in enumerate(blocks):
        Xs.append(noise_block(X, shard_idx=i, **noise))
        labs.append(np.asarray(lab, dtype=str))
    noised = ad.AnnData(X=sp.vstack(Xs).tocsr().astype(np.float64),
                        obs=pd.DataFrame({"target": np.concatenate(labs)}))
    _, _, ref = pseudobulk_bulk_lognorm_with_moments(noised, "target", bulk_target_sum=1e6)
    order = np.argsort(mom["bulk_lognorm"].perts)
    np.testing.assert_allclose(mom["bulk_lognorm"].jk[order], ref.jk, rtol=1e-8)
    assert np.all(ref.jk > 0.0)          # the fixture actually exercises the correction


def test_cell_layout_streaming_jk_matches_the_prep_reference(tmp_path):
    """The SIXTH driver. cell_pseudobulk imports the guard function-locally, so deleting the
    helper breaks it at CALL time, invisible to import checks and to ruff."""
    from cellstream.cell import write_cell_archive   # NOT cell_eval2._cell_archive: that
    # module exposes READERS only (open_cell_store at :79); the writer lives in cellstream,
    # which is how tests/test_cellstream.py:23 and tests/test_cell_source.py:34 import it.
    from cell_eval2._cell_archive import open_cell_store
    from cell_eval2.cell_source import cell_pseudobulk
    from cell_eval2.prep import pseudobulk_bulk_lognorm_with_moments

    _, adata = _write(tmp_path, n=60, g=8, k=6)
    a64 = adata.copy()
    a64.X = a64.X.astype(np.float64)
    cpath = str(tmp_path / "c.csad")
    write_cell_archive(a64, cpath, group_by="target", codec="zstd")
    ref_perts, _, ref = pseudobulk_bulk_lognorm_with_moments(a64, "target",
                                                             bulk_target_sum=1e6)
    _, mom = cell_pseudobulk(open_cell_store(cpath), pert_col="target",
                             norms=["bulk_lognorm"], target_sum=None, with_moments=True,
                             bulk_target_sum=1e6)
    order = np.argsort(mom["bulk_lognorm"].perts)
    np.testing.assert_array_equal(np.asarray(mom["bulk_lognorm"].perts)[order], ref_perts)
    np.testing.assert_allclose(mom["bulk_lognorm"].jk[order], ref.jk, rtol=1e-8)
