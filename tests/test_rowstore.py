"""Tests for the row-store .dat reader (cell_eval2.rowstore)."""
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from cell_eval2 import MemBudget
from cell_eval2 import rowstore as rs
from cell_eval2.config import EvalConfig


def _write_synthetic_rowstore(root: Path, artifacts: list[dict]) -> Path:
    """Write a minimal-but-valid row store under <root>/staging and return it.

    Each artifact dict: dataset, context, control_value, labels (n,), var_names (G,),
    real (n,G uint16), pred (n,G uint16). plan.json records DELIBERATELY BOGUS absolute
    paths so tests prove the reader re-bases on the passed staging_dir (stale-path immunity).
    """
    staging = root / "staging"
    plan = {"staging_dir": "/bogus/generating/machine/staging", "artifacts": []}
    for i, a in enumerate(artifacts):
        aid = f"artifact_{i:05d}"
        adir = staging / aid
        adir.mkdir(parents=True)
        real = np.ascontiguousarray(a["real"], dtype=np.uint16)
        pred = np.ascontiguousarray(a["pred"], dtype=np.uint16)
        n, g = real.shape
        real.tofile(adir / "real_X.dat")
        pred.tofile(adir / "pred_X.dat")
        pd.DataFrame({
            "dataset": a["dataset"], "context": a["context"],
            "perturbation": list(a["labels"]), "control_value": a["control_value"],
            "file_idx": 0, "cell_idx": list(range(n)),
        }).to_csv(adir / "obs.csv", index=False)
        np.save(adir / "var_names.npy", np.asarray(a["var_names"], dtype=np.str_))
        plan["artifacts"].append({
            "artifact_id": aid, "dataset": a["dataset"], "dataset_slug": a["dataset"],
            "panel_id": 0, "context": a["context"], "context_slug": a["context"],
            "control_value": a["control_value"], "gene_ids": list(range(g)),
            "var_names": list(a["var_names"]), "n_rows": n, "n_genes": g, "dtype": "uint16",
            "real_path": f"/bogus/{aid}/real_X.dat", "pred_path": f"/bogus/{aid}/pred_X.dat",
            "written_path": f"/bogus/{aid}/written.dat", "obs_path": f"/bogus/{aid}/obs.csv",
            "var_names_path": f"/bogus/{aid}/var_names.npy",
        })
    (staging / "plan.json").write_text(json.dumps(plan))
    return staging


def test_scaled_log1p_matches_reference():
    # verbatim replica: out=f32; lib=row sum (0->1); out*=target/lib; log1p in place.
    x = np.array([[1, 2, 3], [0, 0, 0], [10, 0, 5]], dtype=np.uint16)
    out = rs.scaled_log1p(x, 1e4)
    ref = np.ascontiguousarray(x, dtype=np.float32)
    lib = ref.sum(axis=1, keepdims=True)
    lib = np.where(lib > 0, lib, 1.0)
    ref = ref * (1e4 / lib)
    np.log1p(ref, out=ref)
    assert out.dtype == np.float32
    np.testing.assert_array_equal(out, ref)          # bit-identical
    assert np.all(out[1] == 0.0)                      # all-zero row -> lib=1 -> log1p(0)=0


def test_read_rowstore_plan_rebases_paths_ignoring_stale(tmp_path):
    staging = _write_synthetic_rowstore(tmp_path, [dict(
        dataset="tahoe", context="C1", control_value="[('DMSO', 0.0, 'uM')]",
        labels=["[('DMSO', 0.0, 'uM')]"] * 4 + ["P1"] * 3,
        var_names=["g0", "g1"], real=np.ones((7, 2)), pred=np.ones((7, 2)),
    )])
    arts = rs.read_rowstore_plan(staging)
    assert len(arts) == 1
    a = arts[0]
    assert a.dataset == "tahoe" and a.context == "C1" and a.panel_id == 0
    assert a.control_value == "[('DMSO', 0.0, 'uM')]"
    assert a.n_rows == 7 and a.n_genes == 2 and a.dtype == "uint16"
    # paths re-based on the PASSED staging dir, NOT the bogus plan.json absolute paths
    assert a.real_path == str(staging / "artifact_00000" / "real_X.dat")
    assert a.obs_path == str(staging / "artifact_00000" / "obs.csv")
    assert "/bogus/" not in a.real_path
    assert Path(a.real_path).is_file() and Path(a.var_names_path).is_file()


def test_read_rowstore_plan_missing_dir_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        rs.read_rowstore_plan(tmp_path / "nope")


def _one_ctx_staging(tmp_path, seed=0):
    rng = np.random.default_rng(seed)
    ctrl = "[('DMSO', 0.0, 'uM')]"
    labels = np.array([ctrl] * 20 + ["P1"] * 8 + ["P2"] * 11 + ["P3"] * 6)
    n, g = labels.size, 12
    real = rng.integers(0, 40, size=(n, g), dtype=np.uint16)
    pred = rng.integers(0, 40, size=(n, g), dtype=np.uint16)
    staging = _write_synthetic_rowstore(tmp_path, [dict(
        dataset="tahoe", context="C1", control_value=ctrl, labels=labels,
        var_names=[f"g{j}" for j in range(g)], real=real, pred=pred)])
    return staging, ctrl, labels, real, pred


def test_rowstore_source_control_block(tmp_path):
    staging, ctrl, labels, real, _ = _one_ctx_staging(tmp_path)
    art = rs.read_rowstore_plan(staging)[0]
    src = rs.RowStoreBatchSource(art, side="real", pert_col="perturbation", control=ctrl)
    cb = src.read_control_block()
    assert cb.n_obs == 20 and cb.n_vars == 12
    assert set(cb.obs["perturbation"].astype(str)) == {ctrl}
    # X == scaled_log1p of the control rows (control rows are the first 20)
    np.testing.assert_array_equal(np.asarray(cb.X), rs.scaled_log1p(real[:20], 1e4))


def test_rowstore_source_pert_batches_cover_noncontrol(tmp_path):
    staging, ctrl, labels, _, _ = _one_ctx_staging(tmp_path)
    art = rs.read_rowstore_plan(staging)[0]
    src = rs.RowStoreBatchSource(art, side="pred", pert_col="perturbation", control=ctrl)
    mem = MemBudget(host_bytes=10**9, gpu_bytes=10**9)
    seen_perts, seen_rows = [], 0
    for batch_perts, batch_ad in src.iter_pert_batches(mem):
        assert ctrl not in batch_perts                 # never the control
        assert set(batch_ad.obs["perturbation"].astype(str)) == set(batch_perts)
        seen_perts += batch_perts
        seen_rows += batch_ad.n_obs
    assert sorted(seen_perts) == ["P1", "P2", "P3"]     # all non-control perts, once each
    assert seen_rows == 8 + 11 + 6


def test_rowstore_source_batches_split_under_tight_budget(tmp_path):
    staging, ctrl, _, _, _ = _one_ctx_staging(tmp_path)
    art = rs.read_rowstore_plan(staging)[0]
    src = rs.RowStoreBatchSource(art, side="pred", pert_col="perturbation", control=ctrl)
    # plan_pert_batches charges the control block (20 cells) to EVERY batch and applies a
    # safety=3.0 factor: per_cell = n_genes*itemsize*safety = 12*4*3 = 144 B. Size the budget
    # to fit control (20) + ~12 pert cells/batch -> the 8+11+6 non-control perts split into
    # >=2 batches (but each single pert still fits, so no ValueError).
    per_cell = 12 * 4 * 3
    tight = MemBudget(host_bytes=(20 + 12) * per_cell, gpu_bytes=(20 + 12) * per_cell)
    batches = list(src.iter_pert_batches(tight))
    assert len(batches) >= 2


def test_resolve_artifact_target_sum_is_the_control_median_not_1e4(tmp_path):
    """#155: `float(cfg.target_sum) if cfg.target_sum else 1e4` was a truthiness test, so None
    silently decoded at 1e4 while cfg.target_sum stayed None for the DE call downstream -- one
    config with two meanings inside one entry point."""
    from dataclasses import replace

    from cell_eval2 import rowstore as rs
    from cell_eval2.config import EvalConfig
    from cell_eval2.norm import resolve_target_sum

    staging, _ctrl, _labels, _real, _pred = _one_ctx_staging(tmp_path)   # 5-tuple, not a path
    art = rs.read_rowstore_plan(staging)[0]
    cfg = replace(EvalConfig.from_preset("cell-eval-0.7.6"), pert_col="perturbation",
                  target_sum=None)

    resolved = rs._resolve_artifact_target_sum(art, cfg)

    raw = rs.RowStoreBatchSource(art, side="real", pert_col=cfg.pert_col,
                                 control=art.control_value, output_space="raw", target_sum=1.0)
    expected = resolve_target_sum(raw.read_control_block(), input_type="counts",
                                  target_sum=None)
    assert resolved == expected
    assert resolved != 1e4, "None must no longer silently decode at 1e4"


def test_resolve_artifact_target_sum_passes_a_numeric_config_through(tmp_path):
    from dataclasses import replace

    from cell_eval2 import rowstore as rs
    from cell_eval2.config import EvalConfig

    staging, _ctrl, _labels, _real, _pred = _one_ctx_staging(tmp_path)   # 5-tuple, not a path
    art = rs.read_rowstore_plan(staging)[0]
    cfg = replace(EvalConfig.from_preset("cell-eval-0.7.6"), pert_col="perturbation",
                  target_sum=1e4)
    assert rs._resolve_artifact_target_sum(art, cfg) == 1e4


def _no_gpu():
    try:
        import cupy
        return cupy.cuda.runtime.getDeviceCount() == 0
    except Exception:
        return True


def test_score_rowstore_api_exported():
    # Task 6 deliverable: the row-store scorer + plan reader are public cell_eval2 API.
    from cell_eval2 import read_rowstore_plan, score_rowstore
    assert callable(score_rowstore) and callable(read_rowstore_plan)


@pytest.mark.skipif(not _no_gpu(), reason="CPU-only: asserts the gpudge-gate NotImplementedError")
def test_score_rowstore_requires_gpudge_on_cpu(tmp_path):
    # The SP2 partitioned path is gpudge-only (in-memory external-ref DE). On a no-GPU node the
    # DE backend resolves to pdex, so score_rowstore must surface the gate cleanly -- this also
    # exercises the CPU plumbing up to it: read_rowstore_plan + RowStoreBatchSource + core entry.
    rng = np.random.default_rng(1)
    ctrl = "[('DMSO', 0.0, 'uM')]"
    labels = np.array([ctrl] * 10 + ["P1"] * 6 + ["P2"] * 7)
    n, g = labels.size, 8
    staging = _write_synthetic_rowstore(tmp_path, [dict(
        dataset="tahoe", context="C1", control_value=ctrl, labels=labels,
        var_names=[f"g{j}" for j in range(g)],
        real=rng.integers(0, 40, size=(n, g), dtype=np.uint16),
        pred=rng.integers(0, 40, size=(n, g), dtype=np.uint16))])
    cfg = replace(EvalConfig.from_preset("cell-eval-0.7.6"), pert_col="perturbation")
    mem = MemBudget(host_bytes=10**9, gpu_bytes=10**9)
    with pytest.raises(NotImplementedError, match="gpudge"):
        rs.score_rowstore(staging, config=cfg, mem_budget=mem)


# --- sparse (CSR) path -------------------------------------------------------------------------
# The fixtures above use rng.integers(0, 40, ...) -> ~97.5% DENSE, so they select the dense path
# under the 40% gate. That is deliberate: they keep proving the dense path is unregressed. But it
# means the CSR path needs its OWN fixture, or it gets zero coverage.


def _sparse_ctx_staging(tmp_path, density=0.07, seed=0):
    """Like _one_ctx_staging but REALISTICALLY SPARSE (~7%, i.e. Tahoe-like) -> picks the CSR path."""
    rng = np.random.default_rng(seed)
    ctrl = "[('DMSO', 0.0, 'uM')]"
    labels = np.array([ctrl] * 20 + ["P1"] * 8 + ["P2"] * 11 + ["P3"] * 6)
    n, g = labels.size, 40

    def _blk():
        m = (rng.random((n, g)) < density) * rng.integers(1, 60, size=(n, g))
        return np.ascontiguousarray(m, dtype=np.uint16)

    real, pred = _blk(), _blk()
    staging = _write_synthetic_rowstore(tmp_path, [dict(
        dataset="tahoe", context="C1", control_value=ctrl, labels=labels,
        var_names=[f"g{j}" for j in range(g)], real=real, pred=pred)])
    return staging, ctrl, real, pred


def test_rowstore_source_selects_sparse_on_sparse_store(tmp_path):
    staging, ctrl, _, _ = _sparse_ctx_staging(tmp_path)
    art = rs.read_rowstore_plan(staging)[0]
    src = rs.RowStoreBatchSource(art, side="real", pert_col="perturbation", control=ctrl)
    assert src.density < 0.40 and src.use_sparse is True


def test_rowstore_source_selects_dense_on_dense_store(tmp_path):
    staging, ctrl, _, _, _ = _one_ctx_staging(tmp_path)          # ~97.5% dense
    art = rs.read_rowstore_plan(staging)[0]
    src = rs.RowStoreBatchSource(art, side="real", pert_col="perturbation", control=ctrl)
    assert src.density > 0.40 and src.use_sparse is False


def test_rowstore_control_block_sparse_is_bit_identical_to_dense(tmp_path):
    """The whole point: CSR changes the container, never the numbers."""
    import scipy.sparse as sp
    staging, ctrl, real, _ = _sparse_ctx_staging(tmp_path)
    art = rs.read_rowstore_plan(staging)[0]
    src = rs.RowStoreBatchSource(art, side="real", pert_col="perturbation", control=ctrl)
    cb = src.read_control_block()
    assert sp.issparse(cb.X) and cb.X.format == "csr"
    np.testing.assert_array_equal(                                # max abs delta == 0
        np.asarray(cb.X.todense()), rs.scaled_log1p(real[:20], 1e4))


def test_rowstore_raw_output_space_sparse_is_exact(tmp_path):
    import scipy.sparse as sp
    staging, ctrl, real, _ = _sparse_ctx_staging(tmp_path)
    art = rs.read_rowstore_plan(staging)[0]
    src = rs.RowStoreBatchSource(art, side="real", pert_col="perturbation", control=ctrl,
                                 output_space="raw")
    cb = src.read_control_block()
    assert sp.issparse(cb.X) and cb.X.format == "csr" and cb.X.dtype == np.float32
    np.testing.assert_array_equal(
        np.asarray(cb.X.todense()), np.ascontiguousarray(real[:20], dtype=np.float32))


def test_rowstore_sparse_batches_cover_noncontrol(tmp_path):
    """iter_pert_batches is unaffected by sparsity -- plan_pert_batches is deliberately untouched."""
    staging, ctrl, _, _ = _sparse_ctx_staging(tmp_path)
    art = rs.read_rowstore_plan(staging)[0]
    src = rs.RowStoreBatchSource(art, side="pred", pert_col="perturbation", control=ctrl)
    mem = MemBudget(host_bytes=10**9, gpu_bytes=10**9)
    seen, rows = [], 0
    for batch_perts, batch_ad in src.iter_pert_batches(mem):
        assert ctrl not in batch_perts
        seen += batch_perts
        rows += batch_ad.n_obs
    assert sorted(seen) == ["P1", "P2", "P3"] and rows == 8 + 11 + 6


@pytest.mark.skipif(not _no_gpu(), reason="CPU-only: stops at the gpudge gate on purpose")
def test_score_rowstore_threads_the_resolved_target_into_both_sources(tmp_path, monkeypatch):
    """#155 WIRING, not just the helper.

    ``test_resolve_artifact_target_sum_*`` above call ``_resolve_artifact_target_sum``
    directly, so they stay green even if the per-artifact threading in ``score_rowstore`` is
    reverted to the old module-level ``float(cfg.target_sum) if cfg.target_sum else 1e4`` --
    which is precisely the truthiness bug #155 is about. This pins the wiring.

    Runs on CPU because ``score_rowstore`` has no early backend gate: the gpudge gate fires
    inside ``_build_reference_streaming_core``, which is AFTER the artifact loop has resolved
    the target and constructed both ``RowStoreBatchSource`` instances. The
    ``NotImplementedError`` is therefore the stopping point, and everything under test has
    already happened by then.
    """
    from dataclasses import replace

    from cell_eval2.norm import resolve_target_sum

    staging, _ctrl, _labels, _real, _pred = _one_ctx_staging(tmp_path)
    cfg = replace(EvalConfig.from_preset("cell-eval-0.7.6"), pert_col="perturbation",
                  target_sum=None)

    art = rs.read_rowstore_plan(staging)[0]
    probe = rs.RowStoreBatchSource(art, side="real", pert_col=cfg.pert_col,
                                   control=art.control_value, output_space="raw",
                                   target_sum=1.0)
    expected = resolve_target_sum(probe.read_control_block(), input_type="counts",
                                  target_sum=None)
    assert expected != 1e4, "fixture must not coincide with the old 1e4 fallback"

    seen = []
    real_cls = rs.RowStoreBatchSource

    def _spy(art_, **kw):
        seen.append(kw)
        return real_cls(art_, **kw)

    monkeypatch.setattr(rs, "RowStoreBatchSource", _spy)

    mem = MemBudget(host_bytes=10**9, gpu_bytes=10**9)
    with pytest.raises(NotImplementedError, match="gpudge"):
        rs.score_rowstore(staging, config=cfg, mem_budget=mem, output_space="scaled_log1p")

    # The resolver's own probe is output_space='raw' @ target_sum=1.0 by construction; the two
    # sources that actually decode carry the REQUESTED space, which is how they are told apart.
    threaded = [kw for kw in seen if kw.get("output_space") == "scaled_log1p"]
    assert {kw["side"] for kw in threaded} == {"real", "pred"}, \
        f"expected one real and one pred source, saw {[kw.get('side') for kw in threaded]}"
    for kw in threaded:
        assert kw["target_sum"] == expected, (
            f"{kw['side']} source decodes at {kw['target_sum']!r}, expected the resolved "
            f"median {expected!r} (1e4 would be the pre-#155 truthiness fallback)"
        )


def test_score_rowstore_threads_the_resolved_target_to_the_BUNDLE_and_every_score_piece(
        tmp_path, monkeypatch):
    """#185's test gap 2. The test above stops at the gpudge gate inside
    ``_build_reference_streaming_core`` -- so it proves the resolved target reaches both row-store
    sources, but NOT that ``art_cfg`` reaches the ``_RefBundle`` and every ``score_piece`` call,
    which is exactly what the #155-into-#176 merge had to get right and what #185's guard change
    touches.

    Stubs the two core builders, the bundle, the scorer and the aggregator, so this runs on CPU
    with no GPU and no gpudge: everything under test is orchestration.
    """
    from cell_eval2 import partition, partition_inmem as pinmem
    from cell_eval2.norm import resolve_target_sum

    staging, _ctrl, _labels, _real, _pred = _one_ctx_staging(tmp_path)
    cfg = replace(EvalConfig.from_preset("cell-eval-0.7.6"), pert_col="perturbation",
                  target_sum=None)
    art = rs.read_rowstore_plan(staging)[0]
    probe = rs.RowStoreBatchSource(art, side="real", pert_col=cfg.pert_col,
                                   control=art.control_value, output_space="raw", target_sum=1.0)
    expected = resolve_target_sum(probe.read_control_block(), input_type="counts", target_sum=None)
    assert expected != 1e4, "fixture must not coincide with the old 1e4 fallback"

    monkeypatch.setattr(pinmem, "_require_partition_config", lambda c: c)
    monkeypatch.setattr(pinmem, "_build_reference_streaming_core", lambda *a, **k: None)
    monkeypatch.setattr(pinmem, "_build_pred_control_reference_core", lambda *a, **k: None)

    bundle_cfgs, piece_cfgs, piece_bundles = [], [], []

    class _FakeBundle:
        def __init__(self, cache_dir, cfg_):
            bundle_cfgs.append(cfg_)

    monkeypatch.setattr(pinmem, "_RefBundle", _FakeBundle)
    monkeypatch.setattr(pinmem, "score_piece",
                        lambda *a, **k: (piece_cfgs.append(k["config"]),
                                         piece_bundles.append(k["bundle"]), None)[-1])

    class _Done(Exception):
        pass

    monkeypatch.setattr(partition, "aggregate_partials",
                        lambda *a, **k: (_ for _ in ()).throw(_Done()))

    mem = MemBudget(host_bytes=10**9, gpu_bytes=10**9)
    with pytest.raises(_Done):
        rs.score_rowstore(staging, config=cfg, mem_budget=mem, output_space="scaled_log1p")

    assert len(bundle_cfgs) == 1, f"one bundle per artifact, got {len(bundle_cfgs)}"
    assert bundle_cfgs[0].target_sum == expected, (
        f"the bundle was built at target_sum={bundle_cfgs[0].target_sum!r}, not the artifact's "
        f"resolved {expected!r}")
    assert piece_cfgs, "no score_piece call was made"
    for c in piece_cfgs:
        assert c.target_sum == expected, (
            f"a piece was scored at target_sum={c.target_sum!r}, not {expected!r}")
    # ...and it is the SAME bundle object for every piece (#153), not one per batch.
    assert all(b is piece_bundles[0] for b in piece_bundles) and piece_bundles[0] is not None
