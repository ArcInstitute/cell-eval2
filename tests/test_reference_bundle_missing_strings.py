"""`build_reference` writes `real_control.h5ad` when the control cells carry a string column with
missing values, in obs or var (the h5ad writer raises "Can't implicitly convert non-string
objects to strings" for a string column that is missing on every row).
"""
import json

import anndata as ad
import numpy as np
import pandas as pd
import pytest

from cell_eval2 import partition_inmem

from test_target_sum_resolution import N_GENES, _counts_pair, _part_cfg, _stub_gpudge


def _string_column(kind, values, index):
    if kind == "string":
        return pd.Series(pd.array(values, dtype="string"), index=index)
    return pd.Series(values, index=index, dtype=object)


def _build(real, cache_dir):
    return partition_inmem.build_reference(
        real, config=_part_cfg(), cache_dir=str(cache_dir), control_format="h5ad",
        comparator="bulk_lognorm")


@pytest.mark.parametrize("kind", ["object", "string"])
@pytest.mark.parametrize("axis", ["obs", "var"])
@pytest.mark.parametrize("missing", ["all", "partial"])
def test_build_reference_writes_control_h5ad_with_missing_strings(
        tmp_path, monkeypatch, kind, axis, missing):
    _stub_gpudge(monkeypatch, {}, n_genes=N_GENES)
    _, real = _counts_pair()
    _build(real, tmp_path / "clean")

    frame = getattr(real, axis)
    values = [None if missing == "all" or i % 3 == 0 else f"v{i}" for i in range(len(frame))]
    frame["note"] = _string_column(kind, values, frame.index)

    manifest = _build(real, tmp_path / "bundle")

    control = ad.read_h5ad(tmp_path / "bundle" / "real_control.h5ad")
    is_control = (real.obs["target"] == "non-targeting").to_numpy()
    rows = np.flatnonzero(is_control) if axis == "obs" else np.arange(real.n_vars)
    assert getattr(control, axis)["note"].astype(str).tolist() == [
        "" if values[i] is None else values[i] for i in rows]
    assert set(control.obs["target"].astype(str)) == {"non-targeting"}
    # Nothing but the control file's added column differs from a bundle built without the column.
    clean_control = ad.read_h5ad(tmp_path / "clean" / "real_control.h5ad")
    np.testing.assert_array_equal(np.asarray(control.X), np.asarray(clean_control.X))
    assert manifest == json.loads((tmp_path / "clean" / "reference.json").read_text())
    name = "real_pseudobulk_bulk_lognorm.npz"
    with np.load(tmp_path / "clean" / name) as a, np.load(tmp_path / "bundle" / name) as b:
        assert sorted(a.files) == sorted(b.files)
        for key in a.files:
            np.testing.assert_array_equal(a[key], b[key])
    assert (tmp_path / "clean" / "real_de.parquet").read_bytes() == (
        tmp_path / "bundle" / "real_de.parquet").read_bytes()
    assert [None if pd.isna(v) else v for v in frame["note"]] == values


def test_build_reference_h5ad_path_input_still_writes_control_h5ad(tmp_path, monkeypatch):
    """A file stores a missing string as a categorical NaN, which the rule leaves unchanged, so
    a path input cannot carry the failing column."""
    _stub_gpudge(monkeypatch, {}, n_genes=N_GENES)
    monkeypatch.setattr(partition_inmem, "is_shad", lambda path: False)
    _, real = _counts_pair()
    real.obs["batch"] = pd.Series(
        [None if i % 3 == 0 else f"b{i}" for i in range(real.n_obs)],
        index=real.obs.index, dtype=object)
    real.write_h5ad(tmp_path / "real.h5ad")

    _build(str(tmp_path / "real.h5ad"), tmp_path / "bundle")

    control = ad.read_h5ad(tmp_path / "bundle" / "real_control.h5ad")
    assert control.n_obs == int((real.obs["target"] == "non-targeting").sum())
    expected = [None if pd.isna(v) else v
                for v in real.obs.loc[real.obs["target"] == "non-targeting", "batch"]]
    assert [None if pd.isna(v) else v for v in control.obs["batch"]] == expected


def test_build_reference_h5ad_writes_a_missing_cell_name(tmp_path, monkeypatch):
    """The index is written as given: a nullable-string `obs_names` with a missing name writes on
    pandas 3, and the name reads back missing. anndata on pandas 2 refuses any nullable-string
    index, with or without the missing name."""
    _stub_gpudge(monkeypatch, {}, n_genes=N_GENES)
    _, real = _counts_pair()
    names = [f"cell{i}" for i in range(real.n_obs)]
    names[3] = None
    real.obs_names = pd.Index(pd.array(names, dtype="string"))

    if int(pd.__version__.split(".")[0]) < 3:
        with pytest.raises(RuntimeError, match="allow_write_nullable_strings"):
            _build(real, tmp_path)
        return
    _build(real, tmp_path)

    control = ad.read_h5ad(tmp_path / "real_control.h5ad")
    assert pd.isna(control.obs_names[3])
    assert control.obs_names[:3].tolist() == names[:3]


def _round_trip_input(real):
    """Categorical obs/var index and a clean nullable `string` obs column with unique values."""
    real.obs_names = pd.CategoricalIndex([str(x) for x in real.obs_names])
    real.var_names = pd.CategoricalIndex([str(x) for x in real.var_names])
    real.obs["note"] = pd.Series(
        pd.array([f"v{i}" for i in range(real.n_obs)], dtype="string"), index=real.obs.index)
    return real


def _assert_round_trip(back, expected_note):
    assert isinstance(back.obs_names.dtype, pd.CategoricalDtype)
    assert isinstance(back.var_names.dtype, pd.CategoricalDtype)
    # A clean nullable `string` column reads back as the default string dtype of the installed
    # pandas: `str` on pandas 3, `object` on pandas 2.
    assert back.obs["note"].dtype == pd.Series(["a"]).dtype
    assert back.obs["note"].tolist() == expected_note


def test_build_reference_h5ad_keeps_a_categorical_index_and_reads_a_clean_string_column(
        tmp_path, monkeypatch):
    _stub_gpudge(monkeypatch, {}, n_genes=N_GENES)
    _, real = _counts_pair()
    _round_trip_input(real)

    _build(real, tmp_path)

    control = ad.read_h5ad(tmp_path / "real_control.h5ad")
    rows = np.flatnonzero((real.obs["target"] == "non-targeting").to_numpy())
    _assert_round_trip(control, [f"v{i}" for i in rows])
