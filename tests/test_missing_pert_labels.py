"""A missing perturbation label is rejected with a ValueError at every entry point that reads the
perturbation column, before any label is cast to text and before any scoring step runs.

CPU only, no cellstream: archives are fakes, DE is the gpudge stub, and each heavy step that must
not run is replaced by a recorder that the test asserts was never called. The check itself is
`cell_eval2.io.require_pert_labels`; each test asserts its message ("2 of N cells in <side> ..."),
the column name and the side, so a bare column-name match cannot pass for another error.

Label kinds reachable per input: in-memory AnnData takes all five; an `.h5ad` file stores a
categorical (its writer turns strings into one); an archive store hands back str, `string` or
categorical; a row store reads a blank `obs.csv` field as null.
"""
import contextlib

import anndata as ad
import pandas as pd
import polars as pl
import pytest

from _helpers import MISSING_LABEL_KINDS, assert_missing_labels_error, make_labels, set_labels
from cell_eval2.config import EvalConfig
from test_cellstream import _BIG, _FakeCellStore
from test_target_sum_resolution import N_GENES, _cfg_median, _counts_pair, _part_cfg, _stub_gpudge

ARCHIVE_KINDS = ("string_na", "str", "categorical")
SIDES = [("pred", "real"), ("real", "pred")]


def _raises(*, n_missing, n_cells, column, side, other_side=None):
    """`pytest.raises(ValueError)` plus the shared message assertion, as one context manager."""
    @contextlib.contextmanager
    def ctx():
        with pytest.raises(ValueError) as exc:
            yield
        assert_missing_labels_error(exc, n_missing=n_missing, n_cells=n_cells, column=column,
                                    side=side, other_side=other_side)
    return ctx()


class _Recorder:
    """A callable that counts its calls; the test asserts `calls == 0` for a step that must not run."""

    def __init__(self):
        self.calls = 0

    def __call__(self, *args, **kwargs):
        self.calls += 1


def _de_calls(seen):
    return len(seen.get("calls", []))


# ---------------------------------------------------------------- in-memory scoring ----------

@pytest.mark.parametrize("kind", MISSING_LABEL_KINDS)
@pytest.mark.parametrize("bad, other", SIDES)
def test_compute_metrics_in_memory_rejects(kind, bad, other):
    from cell_eval2 import compute_metrics

    pred, real = _counts_pair()
    n = pred.n_obs
    set_labels(pred if bad == "pred" else real, "target", kind, [3, n - 2])
    with _raises(n_missing=2, n_cells=n, column="target", side=bad, other_side=other):
        compute_metrics(pred, real, config=_cfg_median(target_sum=1e4))


@pytest.mark.parametrize("kind", MISSING_LABEL_KINDS)
@pytest.mark.parametrize("side, other", SIDES)
def test_precompute_cache_rejects(tmp_path, kind, side, other):
    from cell_eval2 import precompute_cache

    pred, real = _counts_pair()
    adata = real if side == "real" else pred
    set_labels(adata, "target", kind, [3, adata.n_obs - 2])
    cfg = _cfg_median(target_sum=1e4, cache_real=str(tmp_path / "r"), cache_pred=str(tmp_path / "p"))
    with _raises(n_missing=2, n_cells=adata.n_obs, column="target", side=side, other_side=other):
        precompute_cache(adata, side=side, config=cfg)


# ---------------------------------------------------------------- cell-layout archives --------

def _fake_store(kind, *, bad=True, n_per=100):
    store = _FakeCellStore(n_per=n_per)
    if bad:
        labels = store.obs["perturbation"].tolist()
        store.obs["perturbation"] = make_labels(kind, labels, [n_per + 10, 2 * n_per + 20],
                                                index=store.obs.index)
    return store


@pytest.mark.parametrize("kind", ARCHIVE_KINDS)
@pytest.mark.parametrize("bad, other", SIDES)
def test_compute_metrics_cell_stream_fast_path_rejects(monkeypatch, kind, bad, other):
    import cell_eval2._cell_archive as archive
    import cell_eval2.run as run
    from cell_eval2 import compute_metrics

    stores = {"pred.shad": _fake_store(kind, bad=bad == "pred"),
              "real.shad": _fake_store(kind, bad=bad == "real")}
    monkeypatch.setattr(archive, "open_cell_store", lambda p: stores[p])
    monkeypatch.setattr(run, "_cell_stream_eligible", lambda *a, **k: True)
    dispatch = _Recorder()
    monkeypatch.setattr(run, "dispatch_cell_metrics", dispatch)
    cfg = EvalConfig(pert_col="perturbation", control="control", metrics=["cell_cov_ccc"],
                     device="cpu")
    with _raises(n_missing=2, n_cells=300, column="perturbation", side=bad, other_side=other):
        compute_metrics("pred.shad", "real.shad", config=cfg)
    assert dispatch.calls == 0
    assert [s.close_count for s in stores.values()] == [1, 1]


def _score_cellstream_stubs(monkeypatch, stores):
    import cell_eval2.cellstream as cs
    import cell_eval2.norm as _norm
    import cell_eval2.partition_inmem as pinmem

    monkeypatch.setattr(cs, "open_cell_store", lambda p: stores[p])
    monkeypatch.setattr(_norm, "resolve_input_type", lambda *a, **k: "counts")
    monkeypatch.setattr(pinmem, "_require_partition_config", lambda cfg: cfg)
    steps = {"reference": _Recorder(), "pred_control": _Recorder(), "source": _Recorder()}
    monkeypatch.setattr(pinmem, "_build_reference_streaming_core", steps["reference"])
    monkeypatch.setattr(pinmem, "_build_pred_control_reference_core", steps["pred_control"])
    monkeypatch.setattr(cs, "CellBatchSource", steps["source"])
    return steps


@pytest.mark.parametrize("kind", ARCHIVE_KINDS)
@pytest.mark.parametrize("bad, other", SIDES)
def test_score_cellstream_rejects(monkeypatch, kind, bad, other):
    import cell_eval2.cellstream as cs
    from _helpers import full_minus_moments

    stores = {"pred.shad": _fake_store(kind, bad=bad == "pred"),
              "real.shad": _fake_store(kind, bad=bad == "real")}
    steps = _score_cellstream_stubs(monkeypatch, stores)
    cfg = EvalConfig(pert_col="perturbation", control="control", metrics=full_minus_moments())
    with _raises(n_missing=2, n_cells=300, column="perturbation", side=bad, other_side=other):
        cs.score_cellstream("pred.shad", "real.shad", config=cfg, mem_budget=_BIG)
    assert [s.calls for s in steps.values()] == [0, 0, 0]
    assert [stores["pred.shad"].close_count, stores["real.shad"].close_count] == [
        1 if bad == "pred" else 0, 1]


@pytest.mark.parametrize("kind", ARCHIVE_KINDS)
def test_cellbatchsource_rejects(monkeypatch, kind):
    import cell_eval2.cellstream as cs

    store = _fake_store(kind)
    monkeypatch.setattr(cs, "open_cell_store", lambda p: store)
    with _raises(n_missing=2, n_cells=300, column="perturbation", side="cell archive"):
        cs.CellBatchSource("x.shad", pert_col="perturbation", control="control")
    assert store.close_count == 1


@pytest.mark.parametrize("kind", ARCHIVE_KINDS)
def test_cellbatchsource_checks_labels_before_fingerprinting(monkeypatch, kind):
    import cell_eval2.cellstream as cs

    def boom(store):
        raise RuntimeError("fingerprint reached")

    store = _fake_store(kind)
    monkeypatch.setattr(cs, "open_cell_store", lambda p: store)
    monkeypatch.setattr(cs, "cell_fingerprint", boom)
    with _raises(n_missing=2, n_cells=300, column="perturbation", side="cell archive"):
        cs.CellBatchSource("x.shad", pert_col="perturbation", control="control")


def test_cellbatchsource_closes_its_store_when_construction_fails(monkeypatch):
    import cell_eval2.cellstream as cs

    store = _FakeCellStore()
    monkeypatch.setattr(cs, "open_cell_store", lambda p: store)
    with pytest.raises(KeyError):
        cs.CellBatchSource("x.shad", pert_col="absent", control="control")
    assert store.close_count == 1


@pytest.mark.parametrize("kind", ARCHIVE_KINDS)
@pytest.mark.parametrize("bad, other", SIDES)
def test_score_streaming_cell_rejects_each_store_as_it_is_opened(monkeypatch, kind, bad, other):
    import cell_eval2.cell_source as cell_source
    import cell_eval2.scale as scale
    from test_cell_source import _FakeGroupStore

    def store(is_bad):
        s = _FakeGroupStore()
        if is_bad:
            s.obs["target"] = make_labels(kind, s.obs["target"].tolist(), [5, 60],
                                          index=s.obs.index)
        return s

    stores = {"pred.shad": store(bad == "pred"), "real.shad": store(bad == "real")}
    opened = []

    def _open(path):
        opened.append(path)
        return stores[path]

    monkeypatch.setattr(scale, "open_cell_store", _open)
    validate = _Recorder()
    monkeypatch.setattr(cell_source, "validate_cell_pair", validate)
    cfg = EvalConfig(target_sum=1e4, device="cpu", metrics=["expr_mae"])
    with _raises(n_missing=2, n_cells=100, column="target", side=bad, other_side=other):
        scale.score_streaming_cell("pred.shad", "real.shad", config=cfg)
    assert validate.calls == 0
    # The pred check runs before the real store is opened.
    assert opened == (["pred.shad"] if bad == "pred" else ["pred.shad", "real.shad"])


def test_score_streaming_cell_keeps_the_group_by_mismatch_error(monkeypatch):
    """A pert_col that is not an obs column is still reported as the group_by mismatch."""
    import cell_eval2.scale as scale
    from test_cell_source import _FakeGroupStore

    stores = {"pred.shad": _FakeGroupStore(), "real.shad": _FakeGroupStore()}
    monkeypatch.setattr(scale, "open_cell_store", lambda path: stores[path])
    cfg = EvalConfig(pert_col="not_a_column", target_sum=1e4, device="cpu", metrics=["expr_mae"])
    with pytest.raises(ValueError, match="group_by 'target' != pert_col 'not_a_column'"):
        scale.score_streaming_cell("pred.shad", "real.shad", config=cfg)


# ---------------------------------------------------------------- .h5ad manifest ---------------

def _manifest_with_missing(tmp_path, *, bad_side):
    """A manifest of h5ad pairs where `bad_side`'s file has two missing labels (categorical, the
    only kind a file keeps). Returns (manifest path, the corrupted file's path, n_cells)."""
    from test_h5ad_manifest import _make_synth_artifact

    from cell_eval2 import h5ad_manifest

    man = _make_synth_artifact(str(tmp_path), contexts=["ctxA"], perts=["A", "B", "C"])
    art = h5ad_manifest.read_manifest(man)[0]
    path = art.real_abs if bad_side == "real" else art.pred_abs
    adata = ad.read_h5ad(path)
    set_labels(adata, "perturbation", "categorical", [3, adata.n_obs - 2])
    adata.write_h5ad(path)
    return man, path, adata.n_obs


def test_read_group_block_rejects(tmp_path):
    from cell_eval2.h5ad_manifest import read_group_block

    _, path, n = _manifest_with_missing(tmp_path, bad_side="real")
    with _raises(n_missing=2, n_cells=n, column="perturbation", side=path):
        read_group_block(path, pert_col="perturbation", labels={"non-targeting"})


def test_iter_h5ad_pert_batches_rejects_on_the_first_pull(tmp_path):
    from cell_eval2.h5ad_manifest import iter_h5ad_pert_batches

    _, path, n = _manifest_with_missing(tmp_path, bad_side="pred")
    gen = iter_h5ad_pert_batches(path, pert_col="perturbation", control="non-targeting",
                                 mem_budget=_BIG)
    with _raises(n_missing=2, n_cells=n, column="perturbation", side=path):
        next(gen)


def test_build_reference_streaming_rejects_before_any_de(tmp_path, monkeypatch):
    from cell_eval2 import partition_inmem

    seen = {}
    _stub_gpudge(monkeypatch, seen, n_genes=N_GENES)
    _, path, n = _manifest_with_missing(tmp_path / "m", bad_side="real")
    with _raises(n_missing=2, n_cells=n, column="perturbation", side=path):
        partition_inmem.build_reference_streaming(
            path, config=_part_cfg(pert_col="perturbation"), cache_dir=str(tmp_path / "ref"),
            control="non-targeting", mem_budget=_BIG, comparator="bulk_lognorm")
    assert _de_calls(seen) == 0


def test_build_pred_control_reference_rejects(tmp_path, monkeypatch):
    from cell_eval2 import partition_inmem

    seen = {}
    _stub_gpudge(monkeypatch, seen, n_genes=N_GENES)
    _, path, n = _manifest_with_missing(tmp_path / "m", bad_side="pred")
    with _raises(n_missing=2, n_cells=n, column="perturbation", side=path):
        partition_inmem.build_pred_control_reference(
            path, config=_part_cfg(pert_col="perturbation", target_sum=1e4), cache_dir=str(tmp_path / "ref"),
            control="non-targeting", comparator="bulk_lognorm")
    assert _de_calls(seen) == 0


@pytest.mark.parametrize("bad, other", SIDES)
def test_score_h5ad_manifest_rejects_before_scoring_a_batch(tmp_path, monkeypatch, bad, other):
    from cell_eval2 import h5ad_manifest, partition_inmem

    seen = {}
    _stub_gpudge(monkeypatch, seen, n_genes=N_GENES)
    monkeypatch.setattr(partition_inmem, "_require_partition_config", lambda cfg: cfg)
    piece = _Recorder()
    monkeypatch.setattr(partition_inmem, "score_piece", piece)
    reference = _Recorder()
    if bad == "pred":     # the real reference is built first; stub it so the pred file is reached
        monkeypatch.setattr(partition_inmem, "build_reference_streaming", reference)
    man, path, n = _manifest_with_missing(tmp_path, bad_side=bad)
    other_path = path.replace(f"adata_{bad}", f"adata_{other}")
    with pytest.raises(ValueError) as exc:
        h5ad_manifest.score_h5ad_manifest(man, config=_part_cfg(), mem_budget=_BIG)
    msg = assert_missing_labels_error(exc, n_missing=2, n_cells=n, column="perturbation",
                                      side=path)
    assert other_path not in msg
    assert piece.calls == 0 and _de_calls(seen) == 0
    assert reference.calls == (1 if bad == "pred" else 0)


# ---------------------------------------------------------------- row store --------------------

def _rowstore_with_blank_labels(tmp_path, blank_at):
    """A one-context row store whose obs.csv has a blank perturbation field at `blank_at`."""
    from test_rowstore import _one_ctx_staging

    from cell_eval2 import rowstore as rs

    staging, ctrl, labels, _, _ = _one_ctx_staging(tmp_path)
    art = rs.read_rowstore_plan(staging)[0]
    obs = pd.read_csv(art.obs_path)
    obs.loc[list(blank_at), "perturbation"] = None
    obs.to_csv(art.obs_path, index=False)
    return staging, art, ctrl, labels.size


@pytest.mark.parametrize("side", ["real", "pred"])
def test_rowstore_source_rejects_a_blank_label(tmp_path, side):
    from cell_eval2 import rowstore as rs

    _, art, ctrl, n = _rowstore_with_blank_labels(tmp_path, [3, 30])
    with _raises(n_missing=2, n_cells=n, column="perturbation",
                 side=f"{side} row store (artifact {art.artifact_id})"):
        rs.RowStoreBatchSource(art, side=side, pert_col="perturbation", control=ctrl)


def test_score_rowstore_rejects_before_any_reference_is_built(tmp_path, monkeypatch):
    from cell_eval2 import partition_inmem
    from cell_eval2 import rowstore as rs

    staging, art, ctrl, n = _rowstore_with_blank_labels(tmp_path, [3, 30])
    steps = [_Recorder(), _Recorder(), _Recorder()]
    monkeypatch.setattr(partition_inmem, "_build_reference_streaming_core", steps[0])
    monkeypatch.setattr(partition_inmem, "_build_pred_control_reference_core", steps[1])
    monkeypatch.setattr(partition_inmem, "score_piece", steps[2])
    cfg = EvalConfig(pert_col="perturbation", target_sum=1e4, metrics=["expr_mae"], device="cpu")
    with _raises(n_missing=2, n_cells=n, column="perturbation",
                 side=f"real row store (artifact {art.artifact_id})"):
        rs.score_rowstore(staging, config=cfg, mem_budget=_BIG)
    assert [s.calls for s in steps] == [0, 0, 0]


def test_rowstore_source_keeps_an_empty_string_label(tmp_path):
    """polars writes "" as a quoted field and reads it back as "": a label, not a null."""
    from test_rowstore import _one_ctx_staging

    from cell_eval2 import rowstore as rs

    staging, ctrl, labels, _, _ = _one_ctx_staging(tmp_path)
    art = rs.read_rowstore_plan(staging)[0]
    obs = pl.read_csv(art.obs_path)
    new = pl.Series("perturbation", [("" if i in (3, 30) else v)
                                     for i, v in enumerate(labels.tolist())])
    obs.with_columns(new).write_csv(art.obs_path)
    raw = pl.read_csv(art.obs_path, columns=["perturbation"])["perturbation"]
    assert raw.null_count() == 0 and (raw == "").sum() == 2

    src = rs.RowStoreBatchSource(art, side="real", pert_col="perturbation", control=ctrl)
    assert "" in src._idx_by_label


# ---------------------------------------------------------------- partition_inmem --------------

@pytest.mark.parametrize("kind", MISSING_LABEL_KINDS)
def test_build_reference_rejects_before_any_de(tmp_path, monkeypatch, kind):
    from cell_eval2 import partition_inmem

    seen = {}
    _stub_gpudge(monkeypatch, seen, n_genes=N_GENES)
    _, real = _counts_pair()
    set_labels(real, "target", kind, [3, real.n_obs - 2])
    with _raises(n_missing=2, n_cells=real.n_obs, column="target", side="real"):
        partition_inmem.build_reference(real, config=_part_cfg(), cache_dir=str(tmp_path),
                                        control_format="h5ad", comparator="bulk_lognorm")
    assert _de_calls(seen) == 0
    assert not (tmp_path / "reference.json").exists()


def test_build_reference_h5ad_path_rejects(tmp_path, monkeypatch):
    from cell_eval2 import partition_inmem

    seen = {}
    _stub_gpudge(monkeypatch, seen, n_genes=N_GENES)
    monkeypatch.setattr(partition_inmem, "is_shad", lambda path: False)
    _, real = _counts_pair()
    set_labels(real, "target", "categorical", [3, real.n_obs - 2])
    real.write_h5ad(tmp_path / "real.h5ad")
    with _raises(n_missing=2, n_cells=real.n_obs, column="target", side="real"):
        partition_inmem.build_reference(str(tmp_path / "real.h5ad"), config=_part_cfg(),
                                        cache_dir=str(tmp_path / "ref"), control_format="h5ad",
                                        comparator="bulk_lognorm")
    assert _de_calls(seen) == 0


@pytest.mark.parametrize("kind", MISSING_LABEL_KINDS)
def test_score_piece_rejects(tmp_path, monkeypatch, kind):
    from cell_eval2 import partition_inmem

    seen = {}
    _stub_gpudge(monkeypatch, seen, n_genes=N_GENES)
    pred, real = _counts_pair()
    partition_inmem.build_reference(real, config=_part_cfg(target_sum=1e4),
                                    cache_dir=str(tmp_path), control_format="h5ad",
                                    comparator="bulk_lognorm")
    piece = pred[pred.obs["target"] != "non-targeting"].copy()
    set_labels(piece, "target", kind, [2, piece.n_obs - 2])
    de_before = _de_calls(seen)
    with _raises(n_missing=2, n_cells=piece.n_obs, column="target", side="pred piece"):
        partition_inmem.score_piece(piece, str(tmp_path), config=_part_cfg(target_sum=1e4),
                                    piece_id="p0", comparator="bulk_lognorm")
    assert _de_calls(seen) == de_before


# ---------------------------------------------------------------- baseline ---------------------

def _baseline_cfg(**kw):
    base = dict(metrics=["expr_mae"], pert_col="target", control="non-targeting",
                input_type="lognorm", validate_input=False)
    base.update(kw)
    return EvalConfig(**base)


@pytest.mark.parametrize("kind", MISSING_LABEL_KINDS)
def test_build_generic_baseline_rejects(synthetic_pair, tmp_path, kind):
    from cell_eval2.baseline import build_generic_baseline

    _, real = synthetic_pair
    set_labels(real, "target", kind, [30, 60])
    out = tmp_path / "pred.h5ad"
    with _raises(n_missing=2, n_cells=real.n_obs, column="target", side="real"):
        build_generic_baseline(real, config=_baseline_cfg(), save_pred=str(out), emit="tile")
    assert not out.exists()


def test_generic_response_profile_rejects(synthetic_pair):
    from cell_eval2.baseline import generic_response_profile

    _, real = synthetic_pair
    set_labels(real, "target", "object_none", [30, 60])
    with _raises(n_missing=2, n_cells=real.n_obs, column="target", side="real"):
        generic_response_profile(real, pert_col="target", control="non-targeting")


@pytest.mark.parametrize("builder", ["build_baseline_prediction", "build_constant_prediction"])
def test_baseline_prediction_builders_reject_a_template(synthetic_pair, builder):
    from cell_eval2.baseline import (build_baseline_prediction, build_constant_prediction,
                                     generic_response_profile)

    _, real = synthetic_pair
    profile = generic_response_profile(real, pert_col="target", control="non-targeting")
    template = real.copy()
    set_labels(template, "target", "string_na", [30, 60])
    with _raises(n_missing=2, n_cells=real.n_obs, column="target", side="template"):
        if builder == "build_baseline_prediction":
            build_baseline_prediction(profile, template, pert_col="target",
                                      control="non-targeting", emit="tile")
        else:
            build_constant_prediction(profile, template, pert_col="target",
                                      control="non-targeting")


@pytest.mark.parametrize("bad, other", SIDES)
def test_lock_matrix_space_rejects(synthetic_pair, bad, other):
    from cell_eval2.baseline import lock_matrix_space

    pred, real = synthetic_pair
    set_labels(pred if bad == "pred" else real, "target", "object_nan", [30, 60])
    with _raises(n_missing=2, n_cells=real.n_obs, column="target", side=bad, other_side=other):
        lock_matrix_space(real, pred, config=_baseline_cfg())


@pytest.mark.parametrize("bad, other", SIDES)
def test_build_run_meta_rejects(synthetic_pair, bad, other):
    from cell_eval2.baseline import build_run_meta

    pred, real = synthetic_pair
    set_labels(pred if bad == "pred" else real, "target", "str", [30, 60])
    with _raises(n_missing=2, n_cells=real.n_obs, column="target", side=bad, other_side=other):
        build_run_meta(_baseline_cfg(), real, pred)


# ---------------------------------------------------------------- reference estimators --------

@pytest.mark.parametrize("kind", MISSING_LABEL_KINDS)
def test_compute_ceiling_rejects_instead_of_dropping_the_cells(kind):
    from cell_eval2.ceiling import compute_ceiling

    _, real = _counts_pair()
    set_labels(real, "target", kind, [3, real.n_obs - 2])
    with _raises(n_missing=2, n_cells=real.n_obs, column="target", side="real"):
        compute_ceiling(real, config=_cfg_median(target_sum=1e4))


@pytest.mark.parametrize("kind", MISSING_LABEL_KINDS)
def test_compute_lfc_nmae_reference_rejects(kind):
    from cell_eval2.lfc_nmae_ref import compute_lfc_nmae_reference

    _, real = _counts_pair()
    set_labels(real, "target", kind, [3, real.n_obs - 2])
    with _raises(n_missing=2, n_cells=real.n_obs, column="target", side="real"):
        compute_lfc_nmae_reference(real, config=_cfg_median(target_sum=1e4))


@pytest.mark.parametrize("kind", MISSING_LABEL_KINDS)
def test_compute_replicate_anchor_rejects(kind):
    from cell_eval2.anchor import compute_replicate_anchor

    _, real = _counts_pair()
    set_labels(real, "target", kind, [3, real.n_obs - 2])
    with _raises(n_missing=2, n_cells=real.n_obs, column="target", side="real"):
        compute_replicate_anchor(real, config=_cfg_median(target_sum=1e4))


@pytest.mark.parametrize("kind", MISSING_LABEL_KINDS)
def test_cached_anchor_rejects_before_fingerprinting(tmp_path, kind):
    from cell_eval2.anchor import cached_anchor
    from cell_eval2.cache import CacheStore

    _, real = _counts_pair()
    set_labels(real, "target", kind, [3, real.n_obs - 2])
    with _raises(n_missing=2, n_cells=real.n_obs, column="target", side="real"):
        cached_anchor(real, _cfg_median(target_sum=1e4), store=CacheStore(str(tmp_path)))
