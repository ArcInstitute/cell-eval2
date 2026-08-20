"""GPU acceptance tests for one resolved median across partitioned scoring (#155).

The real control pool is deliberately three times deeper than the perturbed cells, so deriving
a fresh median per piece or real batch changes the normalization target enough to be visible.
These tests require real gpudge/CUDA execution and skip on CPU-only hosts.
"""

from dataclasses import replace

import anndata as ad
import numpy as np
import pandas as pd
import polars as pl
import pytest

from cell_eval2 import compute_metrics, partition, partition_inmem
from cell_eval2.config import EvalConfig
from _helpers import full_minus_moments, resolved_comparator


def _no_gpu():
    try:
        import cupy

        return cupy.cuda.runtime.getDeviceCount() == 0
    except Exception:
        return True


pytestmark = pytest.mark.skipif(_no_gpu(), reason="needs a CUDA GPU (gpudge DE both sides)")

PERTS = ["A", "B", "C", "D", "E", "F"]


def _cfg(*, device="auto"):
    v2 = EvalConfig.v2()
    # Fixed AUC floor on BOTH sides so pr_auc/roc_auc are partition-invariant. target_sum=None
    # is the behavior under acceptance; on this counts fixture it resolves from the real control.
    return replace(
        v2,
        pert_col="target_gene",
        target_sum=None,
        device=device,
        # Moment-consuming expression metrics are unavailable on the partitioned driver (#198).
        metrics=full_minus_moments(),
        de=replace(v2.de, auc_pval_floor="replace_zero", auc_pval_floor_value=1e-10),
    )


def _data(ctrl_scale=3.0):
    perts = ["non-targeting"] * 80 + sum(([g] * 50 for g in PERTS), [])
    genes = PERTS + [f"g{i}" for i in range(len(PERTS), 16)]

    def mk(seed, scale_control):
        r = np.random.default_rng(seed)
        X = r.poisson(3, size=(len(perts), 16)).astype(np.float32)
        if scale_control != 1.0:
            control = np.asarray(perts) == "non-targeting"
            X[control] *= scale_control
        return ad.AnnData(
            X=X,
            obs={"target_gene": perts},
            var=pd.DataFrame({"gene": genes}, index=genes),
        )

    return mk(0, ctrl_scale), mk(100, 1.0)  # real, pred


def _tidy_map(df):
    return {(r["perturbation"], r["metric"]): r["value"] for r in df.iter_rows(named=True)}


def _assert_tidy_close(a, b):
    assert set(a) == set(b), f"key mismatch: {set(a) ^ set(b)}"
    for key, av in a.items():
        bv = b[key]
        if av != av:  # NaN
            assert bv != bv, f"{key}: left NaN vs {bv}"
        else:
            assert bv == pytest.approx(av, rel=1e-6, abs=1e-9), \
                f"{key}: left {av} vs right {bv}"


def _score_split(root, *, real, pred, cfg, split):
    cache = str(root / "ref")
    partials = str(root / "partials")
    comparator = resolved_comparator(cfg)
    partition_inmem.build_reference(
        real, config=cfg, cache_dir=cache, control_format="h5ad", comparator=comparator)
    for group in split:
        mask = pred.obs["target_gene"].isin(group).values
        piece = pred[mask].copy()
        partition_inmem.score_piece(
            piece,
            cache,
            config=cfg,
            piece_id="_".join(group),
            partial_out=partials,
            comparator=comparator,
        )
    universe = ["A", "B", "C", "D", "E", "F"]
    full, _ = partition.aggregate_partials(str(root / "partials"),
                                           reference_universe=universe,
                                           reduce_nsig_spearman=True)
    return full


def test_piece_split_does_not_change_the_scores(tmp_path):
    cfg = _cfg()
    real, pred = _data()
    two = _score_split(
        tmp_path / "two",
        real=real,
        pred=pred,
        cfg=cfg,
        split=[["A", "B", "C"], ["D", "E", "F"]],
    )
    three = _score_split(
        tmp_path / "three",
        real=real,
        pred=pred,
        cfg=cfg,
        split=[["A", "B"], ["C", "D"], ["E", "F"]],
    )
    _assert_tidy_close(_tidy_map(two), _tidy_map(three))


def test_mem_budget_does_not_change_the_scores(tmp_path):
    from cell_eval2.h5ad_manifest import H5adBatchSource, MemBudget

    cfg = _cfg()
    real, _pred = _data()
    real_h5ad = str(tmp_path / "real.h5ad")
    real.write_h5ad(real_h5ad)
    source = H5adBatchSource(
        real_h5ad, pert_col=cfg.pert_col, control="non-targeting")

    # float32 counts, 16 genes, safety=3.0: 192 bytes/cell. The resident control has 80
    # cells. A 55-cell perturbation capacity forces 6 one-pert batches; 155 forces 2 batches
    # of three. Assert the layouts before scoring so this cannot pass with equivalent plans.
    per_cell = 16 * np.dtype(real.X.dtype).itemsize * 3
    one_at_a_time = MemBudget(
        host_bytes=(80 + 55) * per_cell, gpu_bytes=(80 + 55) * per_cell)
    three_at_a_time = MemBudget(
        host_bytes=(80 + 155) * per_cell, gpu_bytes=(80 + 155) * per_cell)
    layout_one = [group for group, _ in source.iter_pert_batches(one_at_a_time)]
    layout_three = [group for group, _ in source.iter_pert_batches(three_at_a_time)]
    assert layout_one == [["A"], ["B"], ["C"], ["D"], ["E"], ["F"]]
    assert layout_three == [["A", "B", "C"], ["D", "E", "F"]]

    one_dir = str(tmp_path / "one")
    three_dir = str(tmp_path / "three")
    comparator = resolved_comparator(cfg)
    partition_inmem._build_reference_streaming_core(
        source, config=cfg, cache_dir=one_dir, mem_budget=one_at_a_time,
        comparator=comparator)
    partition_inmem._build_reference_streaming_core(
        source, config=cfg, cache_dir=three_dir, mem_budget=three_at_a_time,
        comparator=comparator)

    from pathlib import Path

    one_files = sorted(p.name for p in Path(one_dir).glob("real_pseudobulk_*.npz"))
    three_files = sorted(p.name for p in Path(three_dir).glob("real_pseudobulk_*.npz"))
    assert one_files == three_files
    assert one_files, "expected at least one real pseudobulk artifact"
    for name in one_files:
        with np.load(str(Path(one_dir) / name)) as a, np.load(str(Path(three_dir) / name)) as b:
            np.testing.assert_array_equal(a["perts"], b["perts"])
            np.testing.assert_allclose(a["means"], b["means"], rtol=1e-6, atol=1e-9)

    a_de = pl.read_parquet(str(Path(one_dir) / "real_de.parquet")).sort(["target", "feature"])
    b_de = pl.read_parquet(str(Path(three_dir) / "real_de.parquet")).sort(["target", "feature"])
    assert a_de.equals(b_de)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_partitioned_equals_whole_under_median(tmp_path, device):
    """The #155 acceptance claim, with ONE variable: partitioning must not change the scores.

    Both sides run on the SAME device, so the only thing that differs is whether the pred side
    was scored whole or in pieces. Measured on an H100: **198/198 metrics bit-identical**, for
    cpu-vs-cpu and cuda-vs-cuda alike -- so this asserts EXACT equality, not a tolerance.

    An earlier revision compared whole(device='cpu') against partitioned(device='cuda'), which
    conflated the invariance under test with the orthogonal fp64->fp32 device switch and could
    therefore only ever be asserted approximately. That device difference is real and is now
    measured on its own, below.
    """
    cfg = _cfg(device=device)
    real, pred = _data()
    whole = compute_metrics(pred, real, config=cfg)
    split = _score_split(
        tmp_path / "partitioned",
        real=real,
        pred=pred,
        cfg=cfg,
        split=[["A", "B", "C"], ["D", "E", "F"]],
    )
    w, s = _tidy_map(whole), _tidy_map(split)
    assert set(w) == set(s), f"key mismatch: {set(w) ^ set(s)}"
    for key, wv in w.items():
        sv = s[key]
        if wv != wv:  # NaN
            assert sv != sv, f"{key}: whole NaN vs partitioned {sv}"
        else:
            assert sv == wv, \
                f"{key}: whole {wv!r} vs partitioned {sv!r} -- expected BIT-IDENTICAL"


def test_cpu_gpu_device_difference_under_median_is_bounded(tmp_path):
    """Bound the fp64-CPU vs fp32-GPU difference that median resolution newly admits.

    ``run._use_gpu_pseudobulk`` gates on ``target_sum is not None`` (run.py:304), so resolving
    ``None`` to a number moves these runs from the fp64 CPU accumulator onto the fp32 GPU one.
    Spec section 5 accepted that deliberately; this measures it rather than assuming it.

    Orthogonal to #155 -- partitioning itself is bit-identical (the test above) -- so it is
    asserted separately and with its own, honest bound.

    MEASURED on an H100 over 198 metrics: 168 bit-identical; worst |rel| 1.51e-05 and worst
    |abs| 5.65e-07, both on ('B', 'delta_pearson') = 0.0375, where a near-zero value inflates
    the relative figure while the absolute one stays tiny. Spec section 5's pre-measurement
    "~1e-7" was a per-element pseudobulk estimate, not a metric-level bound. The bounds below
    sit ~1 order of magnitude above the measurement, so a genuine regression still trips them.
    """
    real, pred = _data()
    cpu = _tidy_map(compute_metrics(pred, real, config=_cfg(device="cpu")))
    gpu = _tidy_map(compute_metrics(pred, real, config=_cfg(device="cuda")))
    assert set(cpu) == set(gpu), f"key mismatch: {set(cpu) ^ set(gpu)}"

    worst_abs, worst_rel, worst_key = 0.0, 0.0, None
    for key, cv in cpu.items():
        gv = gpu[key]
        # NaN must be SYMMETRIC, and checked before any arithmetic. A one-sided NaN would
        # otherwise make `d` NaN, and NaN fails every comparison: `d > worst_abs` is False and
        # `max(worst_rel, nan)` returns worst_rel, so a finite-to-NaN regression on the GPU
        # would leave both bounds at 0.0 and this test would PASS. Verified.
        if cv != cv or gv != gv:
            assert cv != cv and gv != gv, f"{key}: NaN on one side only -- cpu {cv}, gpu {gv}"
            continue
        assert np.isfinite(cv) and np.isfinite(gv), f"{key}: non-finite -- cpu {cv}, gpu {gv}"
        d = abs(cv - gv)
        if d > worst_abs:
            worst_abs, worst_key = d, key
        if cv != 0.0:
            worst_rel = max(worst_rel, d / abs(cv))
    assert worst_abs <= 1e-5, f"device |abs| drift {worst_abs:.3e} at {worst_key} exceeds 1e-5"
    assert worst_rel <= 1e-4, f"device |rel| drift {worst_rel:.3e} exceeds 1e-4"


def test_gpu_pseudobulk_matches_cpu_under_a_resolved_median(tmp_path):
    """The same device difference one layer down, at the pseudobulk itself rather than at the
    metrics -- this is where spec section 5's ~1e-7 estimate actually lives."""
    from cell_eval2 import run
    from cell_eval2.norm import resolve_target_sum

    real, _pred = _data()
    control = real[real.obs["target_gene"] == "non-targeting"].copy()
    target_sum = resolve_target_sum(control, input_type="counts", target_sum=None)

    def _bulk(device):
        return run._side_bulks(
            real,
            fp=None,
            store=None,
            norms=["lognorm"],
            cfg=replace(_cfg(device=device), target_sum=target_sum),
            side="real",
            effective_input_type="counts",
        )["lognorm"]

    cpu_bulk, gpu_bulk = _bulk("cpu"), _bulk("cuda")
    np.testing.assert_array_equal(cpu_bulk[0], gpu_bulk[0])
    np.testing.assert_allclose(cpu_bulk[1], gpu_bulk[1], rtol=1e-6, atol=1e-9)
