"""A missing perturbation label in a REAL cellstream archive (cell layout and packed shard) is
rejected with a ValueError at the entry points that read one.

The archives are written by cellstream with two missing labels. cellstream stores them as a
missing value in obs (and as a group named "nan" in the groups record, which is why the check
reads obs). The fake-store tests for the same entry points are in test_missing_pert_labels.py.
"""
import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

pytest.importorskip("cellstream")

from _helpers import assert_missing_labels_error, full_minus_moments, make_labels  # noqa: E402
from cell_eval2.config import EvalConfig  # noqa: E402
from cell_eval2.h5ad_manifest import MemBudget  # noqa: E402

_BIG = MemBudget(host_bytes=10**12, gpu_bytes=10**12)
N_PER = 10
PERTS = ["control", "GENE_A", "GENE_B", "GENE_C"]
N_CELLS = N_PER * len(PERTS)
MISSING_AT = [N_PER + 2, 2 * N_PER + 3]       # one GENE_A cell, one GENE_B cell


def _adata(kind, *, bad):
    labels = np.repeat(PERTS, N_PER).tolist()
    index = [f"c{i}" for i in range(N_CELLS)]
    if bad:
        col = make_labels(kind, labels, MISSING_AT, index=index)
    else:
        col = pd.Series(labels, index=index)
    rng = np.random.default_rng(0)
    X = sp.csr_matrix(rng.poisson(1.0, size=(N_CELLS, 8)).astype(np.float32))
    genes = PERTS[1:] + [f"g{j}" for j in range(5)]
    return ad.AnnData(X=X, obs=pd.DataFrame({"perturbation": col}, index=index),
                      var=pd.DataFrame(index=genes))


def _write_cell(tmp_path, name, kind, *, bad):
    from cellstream.cell import write_cell_archive

    adata = _adata(kind, bad=bad)
    reference = np.repeat(PERTS, N_PER) == "control"
    path = tmp_path / f"{name}.shad"
    write_cell_archive(adata, path, group_by="perturbation", reference=reference,
                       codec="zstd", overwrite=True)
    return path


def _write_packed(tmp_path, name, kind, *, bad):
    from cellstream import write_sharded

    path = tmp_path / f"{name}.shad"
    write_sharded(_adata(kind, bad=bad), path, group_by="perturbation", reference="control",
                  overwrite=True)
    return path


KINDS = ("object_none", "string_na", "categorical")
SIDES = [("pred", "real"), ("real", "pred")]


@pytest.fixture(params=KINDS)
def kind(request):
    return request.param


def _pair(writer, tmp_path, kind, bad):
    return (writer(tmp_path, "pred", kind, bad=bad == "pred"),
            writer(tmp_path, "real", kind, bad=bad == "real"))


def _config(**kw):
    base = dict(pert_col="perturbation", control="control", device="cpu", target_sum=1e4)
    base.update(kw)
    return EvalConfig(**base)


def test_the_written_archives_hold_the_missing_labels(tmp_path, kind):
    """The precondition of every test below: cellstream keeps the missing labels in obs."""
    from cell_eval2._cell_archive import open_cell_store

    store = open_cell_store(_write_cell(tmp_path, "cell", kind, bad=True))
    try:
        assert int(store.obs["perturbation"].isna().sum()) == 2
    finally:
        store.close()


def test_cellbatchsource_rejects_a_real_archive(tmp_path, kind):
    from cell_eval2.cellstream import CellBatchSource

    path = _write_cell(tmp_path, "real", kind, bad=True)
    with pytest.raises(ValueError) as exc:
        CellBatchSource(path, pert_col="perturbation", control="control")
    assert_missing_labels_error(exc, n_missing=2, n_cells=N_CELLS, column="perturbation",
                                side="cell archive")


@pytest.mark.parametrize("bad, other", SIDES)
@pytest.mark.parametrize("metrics", [["cell_cov_ccc"], ["expr_mae"]],
                         ids=["cell_stream_fast_path", "in_memory_materialize"])
def test_compute_metrics_rejects_cell_archives(tmp_path, kind, bad, other, metrics):
    from cell_eval2 import compute_metrics

    pred, real = _pair(_write_cell, tmp_path, kind, bad)
    with pytest.raises(ValueError) as exc:
        compute_metrics(str(pred), str(real), config=_config(metrics=metrics))
    assert_missing_labels_error(exc, n_missing=2, n_cells=N_CELLS, column="perturbation",
                                side=bad, other_side=other)


@pytest.mark.parametrize("bad, other", SIDES)
def test_score_cellstream_rejects_cell_archives(tmp_path, monkeypatch, kind, bad, other):
    import cell_eval2.partition_inmem as pinmem
    from cell_eval2.cellstream import score_cellstream

    pred, real = _pair(_write_cell, tmp_path, kind, bad)
    built = []
    monkeypatch.setattr(pinmem, "_require_partition_config", lambda cfg: cfg)
    monkeypatch.setattr(pinmem, "_build_reference_streaming_core",
                        lambda *a, **k: built.append(1))
    cfg = _config(metrics=full_minus_moments())
    with pytest.raises(ValueError) as exc:
        score_cellstream(str(pred), str(real), config=cfg, mem_budget=_BIG)
    assert_missing_labels_error(exc, n_missing=2, n_cells=N_CELLS, column="perturbation",
                                side=bad, other_side=other)
    assert built == []


@pytest.mark.parametrize("bad, other", SIDES)
def test_score_streaming_cell_rejects_cell_archives(tmp_path, kind, bad, other):
    from cell_eval2.scale import score_streaming_cell

    pred, real = _pair(_write_cell, tmp_path, kind, bad)
    with pytest.raises(ValueError) as exc:
        score_streaming_cell(str(pred), str(real), config=_config(metrics=["expr_mae"]))
    assert_missing_labels_error(exc, n_missing=2, n_cells=N_CELLS, column="perturbation",
                                side=bad, other_side=other)


def test_build_reference_rejects_a_packed_archive(tmp_path, monkeypatch, kind):
    """`build_reference(<.shad>)` rejects a packed archive with a missing label: the check
    after the load raises, before any DE."""
    from cell_eval2 import partition_inmem
    from test_target_sum_resolution import N_GENES, _part_cfg, _stub_gpudge

    seen = {}
    _stub_gpudge(monkeypatch, seen, n_genes=N_GENES)
    real = _write_packed(tmp_path, "real", kind, bad=True)
    with pytest.raises(ValueError, match=r"2 of 40 cells in .* have a missing \(null\)") as exc:
        partition_inmem.build_reference(
            str(real), config=_part_cfg(pert_col="perturbation", control="control"),
            cache_dir=str(tmp_path / "ref"), control_format="h5ad", comparator="bulk_lognorm")
    assert "'perturbation'" in str(exc.value)
    assert not seen.get("calls")


def test_build_reference_rejects_a_packed_archive_with_a_string_fingerprint(
        tmp_path, monkeypatch):
    """A stubbed string writer fingerprint does not stop the check after the load from
    rejecting the missing label of a packed archive."""
    import cell_eval2.partition_inmem as pinmem
    from test_target_sum_resolution import N_GENES, _part_cfg, _stub_gpudge

    seen = {}
    _stub_gpudge(monkeypatch, seen, n_genes=N_GENES)
    real = _write_packed(tmp_path, "real", "object_none", bad=True)
    monkeypatch.setattr(pinmem, "shad_fingerprint", lambda path: "shad:fixed")
    with pytest.raises(ValueError) as exc:
        pinmem.build_reference(
            str(real), config=_part_cfg(pert_col="perturbation", control="control"),
            cache_dir=str(tmp_path / "ref"), control_format="h5ad", comparator="bulk_lognorm")
    assert_missing_labels_error(exc, n_missing=2, n_cells=N_CELLS, column="perturbation",
                                side="real")
    assert not seen.get("calls")


def _write_packed_group(tmp_path, name, *, null_in):
    """A packed archive whose group_by column ("grp") differs from pert_col ("perturbation");
    `null_in` names the column that holds the two missing labels."""
    from cellstream import write_sharded

    adata = _adata("object_none", bad=False)
    adata.obs["grp"] = adata.obs["perturbation"].copy()
    if null_in:
        adata.obs[null_in] = make_labels("object_none", adata.obs[null_in].tolist(), MISSING_AT,
                                         index=adata.obs.index)
    path = tmp_path / f"{name}.shad"
    write_sharded(adata, path, group_by="grp", reference="control", overwrite=True)
    return path


def test_build_reference_checks_pert_col_not_group_by(tmp_path, monkeypatch):
    """A null in the archive's group_by column is no concern when pert_col is complete."""
    from cell_eval2 import partition_inmem
    from test_target_sum_resolution import _part_cfg, _stub_gpudge

    seen = {}
    _stub_gpudge(monkeypatch, seen, n_genes=5)
    real = _write_packed_group(tmp_path, "real", null_in="grp")
    partition_inmem.build_reference(
        str(real), config=_part_cfg(pert_col="perturbation", control="control"),
        cache_dir=str(tmp_path / "ref"), control_format="h5ad", comparator="bulk_lognorm")


def test_build_reference_rejects_a_null_in_pert_col_when_group_by_differs(tmp_path, monkeypatch):
    from cell_eval2 import partition_inmem
    from test_target_sum_resolution import N_GENES, _part_cfg, _stub_gpudge

    seen = {}
    _stub_gpudge(monkeypatch, seen, n_genes=N_GENES)
    real = _write_packed_group(tmp_path, "real", null_in="perturbation")
    with pytest.raises(ValueError) as exc:
        partition_inmem.build_reference(
            str(real), config=_part_cfg(pert_col="perturbation", control="control"),
            cache_dir=str(tmp_path / "ref"), control_format="h5ad", comparator="bulk_lognorm")
    assert_missing_labels_error(exc, n_missing=2, n_cells=N_CELLS, column="perturbation",
                                side="real")
    assert not seen.get("calls")


