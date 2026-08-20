"""Cell-layout cellstream archive input (#117): detection, materialize, and score parity.

Stage 1 (minimal materialize) is the correctness oracle: scoring a cell-layout
archive must equal scoring the identical data as h5ad. Uses the zstd codec for
fixtures because reading pfordelta archives needs the optional ``pyfastpfor``.
"""
import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

PERTS = ["non-targeting", "GENE_A", "GENE_B", "GENE_C"]


def _synth(seed, n_per=40, n_genes=60):
    rng = np.random.default_rng(seed)
    X = sp.csr_matrix(rng.poisson(0.5, size=(n_per * len(PERTS), n_genes)).astype(np.float32))
    obs = pd.DataFrame({"target": np.repeat(PERTS, n_per)},
                       index=[f"cell{i}" for i in range(X.shape[0])])
    genes = PERTS[1:] + [f"g{j}" for j in range(len(PERTS) - 1, n_genes)]
    var = pd.DataFrame(index=genes)
    return ad.AnnData(X=X, obs=obs, var=var)


def _dense(x):
    """Dense ndarray from a sparse or dense matrix (materializers may return either)."""
    return x.toarray() if sp.issparse(x) else np.asarray(x)


@pytest.fixture
def paired_inputs(tmp_path):
    """(pred, real) each written both as .h5ad and as a cell-layout .shad.

    Needs cellstream to write the archives, so it skips (not errors) when cellstream is
    absent. The cellstream-missing / packed-non-cell error-path tests below do NOT use
    this fixture, so they still run in a no-cellstream env (e.g. CI) — where they matter.
    """
    pytest.importorskip("cellstream")
    from cellstream.cell import write_cell_archive
    out = {}
    for side, seed in (("pred", 1), ("real", 2)):
        adata = _synth(seed)
        h5 = tmp_path / f"{side}.h5ad"
        adata.write_h5ad(h5)
        shad = tmp_path / f"{side}.shad"
        ref = (adata.obs["target"] == "non-targeting").to_numpy()
        write_cell_archive(adata, shad, group_by="target", reference=ref,
                           codec="zstd", overwrite=True)
        out[side] = (h5, shad)
    return out


def test_cell_layout_detects_cell_archive(paired_inputs, tmp_path):
    from cell_eval2._cell_archive import cell_layout
    h5, shad = paired_inputs["pred"]
    assert cell_layout(shad) is True
    assert cell_layout(h5) is False           # a plain h5ad is not a packed cell archive
    assert cell_layout(tmp_path / "nope.shad") is False   # missing file


def test_load_anndata_materializes_cell_layout(paired_inputs):
    from cell_eval2.io import load_anndata
    h5, shad = paired_inputs["pred"]
    a_h5 = load_anndata(h5)
    a_cell = load_anndata(shad)
    assert a_cell.shape == a_h5.shape
    assert list(a_cell.var_names) == list(a_h5.var_names)
    # the cell archive stores cells reference-first; align on obs_names before comparing X
    order = a_h5.obs_names.get_indexer(a_cell.obs_names)
    assert (order >= 0).all()
    np.testing.assert_array_equal(_dense(a_h5.X[order]), _dense(a_cell.X))


def test_score_cell_layout_matches_h5ad(paired_inputs):
    from cell_eval2.config import EvalConfig
    from cell_eval2.run import compute_metrics
    cfg = EvalConfig()  # v2 defaults; DE backend "auto" -> CPU (pdex/scanpy) in a no-GPU env
    (hp, cp), (hr, cr) = paired_inputs["pred"], paired_inputs["real"]
    df_h5 = compute_metrics(hp, hr, config=cfg).sort(["perturbation", "metric"])
    df_cell = compute_metrics(cp, cr, config=cfg).sort(["perturbation", "metric"])
    # results are permutation-invariant aggregations, so cell storage order must not
    # matter: identical keys, and values compared with np.testing.assert_array_equal,
    # which treats NaN == NaN as equal by default (unlike np.array_equal, which needs
    # equal_nan=True) — verified empirically for this numpy.
    assert df_h5["perturbation"].to_list() == df_cell["perturbation"].to_list()
    assert df_h5["metric"].to_list() == df_cell["metric"].to_list()
    # The three moment-consuming expression metrics (#198) are not bit-identical across cell
    # storage order. They subtract tr(Sigma_hat)/n from a plug-in squared distance -- a
    # difference of two large, nearly-equal quantities -- and its Sigma-hat term comes from a
    # sumsq accumulator that sums nonzeros in cell order, so reordering the cells changes the
    # fp64 rounding and the cancellation amplifies it. Measured on the GPU node: max relative
    # difference 1.7e-14 (max absolute 6.7e-15) over 3 perturbations. rtol below carries ~60x
    # margin over that. Every OTHER metric stays an EXACT comparison -- do not widen this to
    # the whole array, and do not tighten rtol below the measured value.
    metrics = np.asarray(df_h5["metric"].to_list())
    v_h5, v_cell = df_h5["value"].to_numpy(), df_cell["value"].to_numpy()
    unbiased = np.isin(metrics, (
        "expr_mse_unbiased", "expr_mse_unbiased_capped", "expr_distance_unbiased",
    ))
    np.testing.assert_array_equal(v_h5[~unbiased], v_cell[~unbiased])
    np.testing.assert_allclose(v_h5[unbiased], v_cell[unbiased], rtol=1e-12, atol=0)


def test_cli_run_accepts_cell_layout(paired_inputs, tmp_path):
    from cell_eval2.cli import main
    (_, cp), (_, cr) = paired_inputs["pred"], paired_inputs["real"]
    out = tmp_path / "out"
    # `minimal`, not `anndata`: see test_cli_run_with_cache_dirs (#257, null panel).
    main(["run", "-ap", str(cp), "-ar", str(cr), "--profile", "minimal", "-o", str(out)])
    assert (out / "results.csv").exists()


def test_cell_layout_missing_cellstream_raises_clear_error(tmp_path, monkeypatch):
    """A cell (SHPK) archive fed without cellstream installed must fail with a clear
    'install cellstream' error, not a cryptic HDF5 fallthrough. Non-SHPK files still
    return False (so an h5ad loads normally even without cellstream)."""
    import sys

    from cell_eval2._cell_archive import cell_layout

    monkeypatch.setitem(sys.modules, "cellstream.packed.reader", None)  # simulate cellstream absent

    shpk = tmp_path / "fake.shad"
    shpk.write_bytes(b"SHPK" + b"\x00" * 128)
    with pytest.raises(ImportError, match="cellstream"):
        cell_layout(shpk)

    plain = tmp_path / "plain.h5ad"
    plain.write_bytes(b"\x89HDF\r\n\x1a\n" + b"\x00" * 128)
    assert cell_layout(plain) is False


def test_cell_layout_packed_non_cell_raises(tmp_path, monkeypatch):
    """A packed archive whose manifest layout isn't 'cell' (e.g. shard-layout) must
    fail clearly, not fall through to anndata. Runs without a real cellstream via a
    fake packed-reader module."""
    import sys
    import types

    from cell_eval2._cell_archive import cell_layout

    fake = types.ModuleType("cellstream.packed.reader")

    class _FakePacked:
        def __init__(self, _p):
            pass

        @property
        def manifest(self):
            return {"layout": "shard"}

    fake.PackedArchive = _FakePacked
    # inject the full package path so `from cellstream.packed.reader import ...` resolves
    # even when cellstream isn't installed (e.g. CI), where the parent packages are absent
    monkeypatch.setitem(sys.modules, "cellstream", types.ModuleType("cellstream"))
    monkeypatch.setitem(sys.modules, "cellstream.packed", types.ModuleType("cellstream.packed"))
    monkeypatch.setitem(sys.modules, "cellstream.packed.reader", fake)

    f = tmp_path / "shard.shad"
    f.write_bytes(b"SHPK" + b"\x00" * 64)
    with pytest.raises(ValueError, match="layout"):
        cell_layout(f)


def test_open_cell_store_requires_gather_rows_adata(tmp_path, monkeypatch):
    """A CellStore lacking cell-layout streaming (no gather_rows_adata) must fail loudly at
    open time with a clear message — not a cryptic AttributeError deep in the batch loop —
    and must close the store it opened before raising. Runs without a real cellstream via an
    injected fake ``cellstream.cell``."""
    import sys
    import types

    from cell_eval2._cell_archive import open_cell_store

    fake_cell = types.ModuleType("cellstream.cell")
    opened = []

    class _OldStore:  # open/close but no gather_rows_adata -- the capability this gate is for
        def __init__(self, _p):
            self.closed = False
            opened.append(self)

        def close(self):
            self.closed = True

    fake_cell.open_cell = lambda p: _OldStore(p)
    monkeypatch.setitem(sys.modules, "cellstream", types.ModuleType("cellstream"))
    monkeypatch.setitem(sys.modules, "cellstream.cell", fake_cell)

    with pytest.raises(ImportError, match="gather_rows_adata"):
        open_cell_store(tmp_path / "old.shad")

    assert len(opened) == 1  # opened exactly once
    assert opened[0].closed is True  # best-effort close ran before raising


def test_open_cell_store_accepts_capable_store(tmp_path, monkeypatch):
    """A CellStore with gather_rows_adata(..., n_threads=) is returned as-is."""
    import sys
    import types

    from cell_eval2._cell_archive import open_cell_store

    fake_cell = types.ModuleType("cellstream.cell")

    class _NewStore:
        def __init__(self, _p):
            pass

        def gather_rows_adata(self, row_ids, n_threads=1):  # both capabilities present
            raise NotImplementedError

        def close(self):  # real CellStore is closable — keep the fake faithful
            pass

    fake_cell.open_cell = lambda p: _NewStore(p)
    monkeypatch.setitem(sys.modules, "cellstream", types.ModuleType("cellstream"))
    monkeypatch.setitem(sys.modules, "cellstream.cell", fake_cell)

    store = open_cell_store(tmp_path / "new.shad")
    assert hasattr(store, "gather_rows_adata")


def test_open_cell_store_requires_n_threads_support(tmp_path, monkeypatch):
    """A store with gather_rows_adata present but WITHOUT n_threads must fail loudly at
    open time — the gather sites call it with n_threads= and have no fallback, so otherwise it
    surfaces as a cryptic TypeError on the first batch (#149)."""
    import sys
    import types

    from cell_eval2._cell_archive import open_cell_store

    fake_cell = types.ModuleType("cellstream.cell")
    opened = []

    class _V070Store:  # legacy shardad-shaped store: gather_rows_adata exists, no n_threads
        def __init__(self, _p):
            self.closed = False
            opened.append(self)

        def gather_rows_adata(self, row_ids):
            raise NotImplementedError

        def close(self):
            self.closed = True

    fake_cell.open_cell = lambda p: _V070Store(p)
    monkeypatch.setitem(sys.modules, "cellstream", types.ModuleType("cellstream"))
    monkeypatch.setitem(sys.modules, "cellstream.cell", fake_cell)

    with pytest.raises(ImportError, match="n_threads"):
        open_cell_store(tmp_path / "v070.shad")

    assert len(opened) == 1
    assert opened[0].closed is True     # best-effort close ran before raising


def test_cell_group_spans_and_reference_count(tmp_path):
    """The adapters must return the archive's REAL spans -- they feed the row-count ramp, and
    a wrong count silently mis-sizes every decode (#149)."""
    pytest.importorskip("cellstream")
    import anndata as ad
    import numpy as np
    import pandas as pd
    import scipy.sparse as sp
    from cellstream.cell import write_cell_archive

    from cell_eval2._cell_archive import (cell_group_spans, cell_reference_row_count,
                                          open_cell_store)

    perts = ["ctrl", "A", "B"]
    n_per = {"ctrl": 50, "A": 20, "B": 30}
    labels = np.concatenate([np.repeat(p, n_per[p]) for p in perts])
    n = labels.size
    adata = ad.AnnData(
        X=sp.csr_matrix(np.ones((n, 4), dtype=np.float32)),
        obs=pd.DataFrame({"target": labels}, index=[f"c{i}" for i in range(n)]),
        var=pd.DataFrame(index=[f"g{j}" for j in range(4)]),
    )
    shad = tmp_path / "spans.shad"
    write_cell_archive(adata, shad, group_by="target",
                       reference=(adata.obs["target"] == "ctrl").to_numpy(),
                       codec="zstd", overwrite=True)

    store = open_cell_store(shad)
    try:
        spans = cell_group_spans(store)
        assert set(spans) == set(perts)
        for label, (start, stop) in spans.items():
            assert stop - start == n_per[label]
            assert store.read_group(label).shape[0] == stop - start   # spans match reality
        assert cell_reference_row_count(store) == n_per["ctrl"]
        assert cell_reference_row_count(store) == store.read_reference().shape[0]
    finally:
        store.close()


def test_cell_group_spans_degrade_to_unknown():
    """A store without the cellstream internal (a future refactor, or a non-CellStore) must yield
    'unknown' rather than raising -- callers then use the conservative small-read default."""
    from cell_eval2._cell_archive import cell_group_spans, cell_reference_row_count

    class _NoGroups:
        pass

    assert cell_group_spans(_NoGroups()) == {}
    assert cell_reference_row_count(_NoGroups()) is None


def test_cell_reference_row_count_none_without_reference():
    """An archive written with no reference pool -> None (cell_reference raises its own clear
    error downstream; the adapter must not invent a count)."""
    from cell_eval2._cell_archive import cell_reference_row_count

    class _NoRef:
        def _load_groups(self):
            return {"reference": None, "groups": [{"label": "A", "start": 0, "stop": 5}]}

    assert cell_reference_row_count(_NoRef()) is None


def test_cell_reference_row_count_multi_label():
    """cellstream stores 'reference' as a single label OR a list; the pool is [0, max stop)."""
    from cell_eval2._cell_archive import cell_reference_row_count

    class _MultiRef:
        def _load_groups(self):
            return {"reference": ["A", "B"],
                    "groups": [{"label": "A", "start": 0, "stop": 5},
                               {"label": "B", "start": 5, "stop": 12},
                               {"label": "C", "start": 12, "stop": 20}]}

    assert cell_reference_row_count(_MultiRef()) == 12


def test_cell_group_spans_propagates_loader_errors():
    """A corrupt groups record must NOT be laundered into 'unknown sizes' -- that would score a
    broken archive silently. Only a MISSING accessor is a legitimate fallback."""
    from cell_eval2._cell_archive import cell_group_spans, cell_reference_row_count

    class _Corrupt:
        def _load_groups(self):
            raise ValueError("corrupt groups member")

    for fn in (cell_group_spans, cell_reference_row_count):
        with pytest.raises(ValueError, match="corrupt"):
            fn(_Corrupt())


def test_cell_reference_row_count_raises_on_unknown_label():
    """Mirror read_reference exactly: a reference label with no group record is a KeyError
    there, so it must not silently become a smaller count here."""
    from cell_eval2._cell_archive import cell_reference_row_count

    class _Dangling:
        def _load_groups(self):
            return {"reference": ["A", "GHOST"],
                    "groups": [{"label": "A", "start": 0, "stop": 5}]}

    with pytest.raises(KeyError):
        cell_reference_row_count(_Dangling())


def test_cell_reference_row_count_empty_reference_list_raises():
    """`reference: []` -> read_reference does max() over an empty sequence -> ValueError. The
    sizing adapter must mirror that, not silently return None for a read that will crash."""
    from cell_eval2._cell_archive import cell_reference_row_count

    class _EmptyRefList:
        def _load_groups(self):
            return {"reference": [], "groups": [{"label": "A", "start": 0, "stop": 5}]}

    with pytest.raises(ValueError):
        cell_reference_row_count(_EmptyRefList())


def test_adapters_surface_a_none_record():
    """A `_load_groups()` that returns None is malformed (the real one always returns a dict);
    it must surface as an error, not be laundered into 'unknown'."""
    from cell_eval2._cell_archive import cell_group_spans, cell_reference_row_count

    class _NoneRecord:
        def _load_groups(self):
            return None

    for fn in (cell_group_spans, cell_reference_row_count):
        with pytest.raises((TypeError, AttributeError)):
            fn(_NoneRecord())


def test_open_cell_store_rejects_uninspectable_gather(tmp_path, monkeypatch):
    """If inspect.signature() cannot introspect gather_rows_adata (it raises TypeError/ValueError
    for some C-backed callables), the capability gate must fail loud AND closed with the upgrade
    message — not leak the opened store and error cryptically mid-batch (Checkpoint-2 codex)."""
    import sys
    import types

    from cell_eval2._cell_archive import open_cell_store

    class _Uninspectable:
        @property
        def __signature__(self):
            raise ValueError("no signature")     # makes inspect.signature() raise

        def __call__(self, row_ids, n_threads=1):
            raise NotImplementedError

    opened = []

    class _WeirdStore:
        def __init__(self, _p):
            self.closed = False
            self.gather_rows_adata = _Uninspectable()   # present but not introspectable
            opened.append(self)

        def close(self):
            self.closed = True

    fake_cell = types.ModuleType("cellstream.cell")
    fake_cell.open_cell = lambda p: _WeirdStore(p)
    monkeypatch.setitem(sys.modules, "cellstream", types.ModuleType("cellstream"))
    monkeypatch.setitem(sys.modules, "cellstream.cell", fake_cell)

    with pytest.raises(ImportError, match="n_threads"):
        open_cell_store(tmp_path / "weird.shad")

    assert len(opened) == 1
    assert opened[0].closed is True     # closed before raising, no leak


def test_open_cell_store_accepts_kwargs_forwarding_gather(tmp_path, monkeypatch):
    """A gather_rows_adata that forwards **kwargs DOES support n_threads and must be accepted.
    The bind-probe accepts it; the old literal 'n_threads' parameter-name check would have
    wrongly rejected it (Checkpoint-2 codex)."""
    import sys
    import types

    from cell_eval2._cell_archive import open_cell_store

    class _KwStore:
        def __init__(self, _p):
            pass

        def gather_rows_adata(self, row_ids, **kwargs):   # supports n_threads via **kwargs
            raise NotImplementedError

        def close(self):
            pass

    fake_cell = types.ModuleType("cellstream.cell")
    fake_cell.open_cell = lambda p: _KwStore(p)
    monkeypatch.setitem(sys.modules, "cellstream", types.ModuleType("cellstream"))
    monkeypatch.setitem(sys.modules, "cellstream.cell", fake_cell)

    store = open_cell_store(tmp_path / "kw.shad")
    assert hasattr(store, "gather_rows_adata")            # accepted, returned as-is
