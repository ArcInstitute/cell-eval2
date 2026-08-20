import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

pytest.importorskip("cellstream")
import anndata as ad  # noqa: E402

from cell_eval2 import stream  # noqa: E402


def _write_fixture(tmp_path):
    from cellstream import write_sharded

    rng = np.random.default_rng(0)
    X = sp.csr_matrix((rng.poisson(0.3, size=(40, 6))).astype(np.float32))
    obs = pd.DataFrame({"target": np.repeat(["non-targeting", "A", "B", "C"], 10)})
    var = pd.DataFrame(index=[f"g{i}" for i in range(6)])
    adata = ad.AnnData(X=X, obs=obs, var=var)
    out = str(tmp_path / "fix.shad")
    write_sharded(adata, out, group_by="target")  # packed SHPK by default
    return out, adata


def test_is_shad_and_metadata(tmp_path):
    path, adata = _write_fixture(tmp_path)
    assert stream.is_shad(path)
    meta = stream.shad_metadata(path)
    assert meta.n_obs == 40 and meta.n_vars == 6 and meta.group_by == "target"
    assert set(meta.perts) == {"non-targeting", "A", "B", "C"}


def test_is_shad_false_for_non_shad(tmp_path):
    assert not stream.is_shad(str(tmp_path / "does-not-exist"))
    plain = tmp_path / "plain.txt"
    plain.write_text("hello")
    assert not stream.is_shad(str(plain))


def test_shad_fingerprint_is_stable_string(tmp_path):
    path, _ = _write_fixture(tmp_path)
    fp1 = stream.shad_fingerprint(path)
    fp2 = stream.shad_fingerprint(path)
    assert isinstance(fp1, str) and fp1.startswith("shad:")
    assert fp1 == fp2


def test_iter_blocks_reconstructs_matrix(tmp_path):
    path, adata = _write_fixture(tmp_path)
    rows_X, rows_lab = [], []
    for x, lab in stream.iter_blocks(path, pert_col="target"):
        assert sp.issparse(x) and x.dtype == np.float32
        rows_X.append(x)
        rows_lab.append(lab)
    # per-group totals must match regardless of shard/row ordering
    got = {}
    for x, lab in zip(rows_X, rows_lab):
        for g in np.unique(lab):
            got[g] = got.get(g, 0) + np.asarray(x[lab == g].sum(axis=0)).ravel()
    want = {}
    labels = adata.obs["target"].to_numpy().astype(str)
    for g in np.unique(labels):
        want[g] = np.asarray(adata.X[labels == g].sum(axis=0)).ravel()
    for g in want:
        np.testing.assert_allclose(got[g], want[g], rtol=0, atol=0)


def test_shad_fingerprint_distinguishes_gene_sets(tmp_path):
    """Two archives with identical shape/perts but different gene panels must NOT share a
    structural fingerprint (else a cache hit returns the wrong genes -- PR #56 review)."""
    import anndata as ad
    import numpy as np
    import pandas as pd
    import scipy.sparse as sp
    from cellstream import write_sharded

    from cell_eval2 import stream

    rng = np.random.default_rng(0)
    X = sp.csr_matrix(rng.poisson(0.5, size=(20, 4)).astype(np.float32))
    obs = pd.DataFrame({"target": np.repeat(["non-targeting", "A"], 10)})
    a = ad.AnnData(X=X.copy(), obs=obs.copy(), var=pd.DataFrame(index=["g0", "g1", "g2", "g3"]))
    b = ad.AnnData(X=X.copy(), obs=obs.copy(), var=pd.DataFrame(index=["g0", "g1", "g2", "gX"]))
    pa, pb = str(tmp_path / "a.shad"), str(tmp_path / "b.shad")
    write_sharded(a, pa, group_by="target")
    write_sharded(b, pb, group_by="target")
    assert stream.shad_fingerprint(pa) != stream.shad_fingerprint(pb)
