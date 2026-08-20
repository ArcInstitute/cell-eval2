import math
import os
import re
import numpy as np
import anndata as ad
import pandas as pd
import pytest

from cell_eval2 import h5ad_manifest
from _helpers import full_minus_moments, resolved_comparator


def _no_gpu():
    try:
        import cupy
        return cupy.cuda.runtime.getDeviceCount() == 0
    except Exception:
        return True


# GPU-dependent tests (anything driving gpudge DE via build_reference_streaming / score_piece /
# score_h5ad_manifest / compute_metrics) SKIP on a no-GPU node, matching the repo convention
# (tests/test_partition_inmem_reference.py). Decorate every such test with @pytestmark_gpu.
# They are VERIFIED on a slurm GPU node (this dev node has no GPU). Tasks 1-4's tests need no GPU.
pytestmark_gpu = pytest.mark.skipif(_no_gpu(), reason="needs a CUDA GPU (gpudge DE)")


def _write_context_h5ad(path_dir, *, perts, control, n_cells_per, n_genes, seed, log1p=False):
    """Write adata_real.h5ad + adata_pred.h5ad for one context: perts contiguous
    (name-sorted), control block LAST, raw integer counts (or log1p)."""
    os.makedirs(path_dir, exist_ok=True)
    perts = list(perts)
    rng = np.random.default_rng(seed)
    labels, blocks_real, blocks_pred = [], [], []
    for p in sorted(perts) + [control]:            # perts name-sorted, control last
        n = n_cells_per
        labels += [p] * n
        blocks_real.append(rng.poisson(3.0, size=(n, n_genes)).astype(np.float32))
        blocks_pred.append(rng.poisson(3.0, size=(n, n_genes)).astype(np.float32))
    Xr, Xp = np.vstack(blocks_real), np.vstack(blocks_pred)
    if log1p:
        Xr, Xp = np.log1p(Xr), np.log1p(Xp)
    obs = pd.DataFrame({
        "dataset": "CCL_1", "context": os.path.basename(path_dir),
        "perturbation": labels, "control_value": control,
    })
    genes = perts + [f"g{i}" for i in range(len(perts), n_genes)]
    var = pd.DataFrame(index=pd.Index(genes, name="gene"))
    np.save(os.path.join(os.path.dirname(path_dir), "var_names.npy"), np.asarray(genes))
    ad.AnnData(X=Xr, obs=obs.copy(), var=var.copy()).write_h5ad(os.path.join(path_dir, "adata_real.h5ad"))
    ad.AnnData(X=Xp, obs=obs.copy(), var=var.copy()).write_h5ad(os.path.join(path_dir, "adata_pred.h5ad"))
    return len(labels), n_genes


def _make_synth_artifact(root, *, contexts, perts, control="non-targeting",
                         n_cells_per=40, n_genes=12, log1p=False):
    """Write a manifest.csv + one h5ad pair per context under `root`. Returns manifest path."""
    rows = []
    for i, ctx in enumerate(contexts):
        cdir = os.path.join(root, "CCL_1", "panel_0", ctx)
        n_cells, ng = _write_context_h5ad(
            cdir, perts=perts, control=control, n_cells_per=n_cells_per,
            n_genes=n_genes, seed=100 + i, log1p=log1p)
        rows.append({
            "dataset": "CCL_1", "panel_id": 0, "context": ctx, "control_value": control,
            "path_real": os.path.relpath(os.path.join(cdir, "adata_real.h5ad"), root),
            "path_pred": os.path.relpath(os.path.join(cdir, "adata_pred.h5ad"), root),
            "var_names_path": os.path.relpath(os.path.join(root, "CCL_1", "panel_0", "var_names.npy"), root),
            "n_cells": n_cells, "n_genes": ng,
        })
    man = os.path.join(root, "manifest.csv")
    pd.DataFrame(rows).to_csv(man, index=False)
    return man


def test_read_manifest_resolves_paths(tmp_path):
    man = _make_synth_artifact(str(tmp_path), contexts=["ctxA", "ctxB"],
                               perts=["GENE1", "GENE2", "GENE3"])
    arts = h5ad_manifest.read_manifest(man)
    assert len(arts) == 2
    a = arts[0]
    assert a.dataset == "CCL_1" and a.control_value == "non-targeting"
    assert os.path.isfile(a.real_abs) and os.path.isfile(a.pred_abs)
    # accepts a results_dir too
    assert len(h5ad_manifest.read_manifest(str(tmp_path))) == 2


def test_resolve_input_type_counts_vs_log1p(tmp_path):
    from cell_eval2.config import EvalConfig
    cfg = EvalConfig.v2()
    man_c = _make_synth_artifact(str(tmp_path / "c"), contexts=["x"], perts=["A", "B"], log1p=False)
    man_l = _make_synth_artifact(str(tmp_path / "l"), contexts=["x"], perts=["A", "B"], log1p=True)
    art_c = h5ad_manifest.read_manifest(man_c)[0]
    art_l = h5ad_manifest.read_manifest(man_l)[0]
    assert h5ad_manifest._resolve_input_type_h5ad(art_c.real_abs, cfg=cfg) == "counts"
    assert h5ad_manifest._resolve_input_type_h5ad(art_l.real_abs, cfg=cfg) == "lognorm"


def test_plan_pert_batches_respects_budget_and_keeps_perts_whole():
    sizes = [("A", 100), ("B", 100), ("C", 100), ("D", 100)]
    ng, itemsize, ctrl = 10, 4, 50
    # bytes/cell = 10*4=40; safety 3 -> 120 bytes/cell. budget picks ~2 perts/batch:
    # allow (batch+ctrl) up to ~ (250 cells) -> 200 pert cells + 50 ctrl
    mb = h5ad_manifest.MemBudget(host_bytes=250 * 120, gpu_bytes=10**12)
    batches = h5ad_manifest.plan_pert_batches(sizes, n_genes=ng, itemsize=itemsize,
                                       control_cells=ctrl, mem_budget=mb, safety=3.0)
    assert [p for b in batches for p in b] == ["A", "B", "C", "D"]   # full coverage, in order
    assert all(len(b) >= 1 for b in batches)
    # each batch's pert-cells + ctrl within budget
    for b in batches:
        cells = sum(dict(sizes)[p] for p in b) + ctrl
        assert cells * ng * itemsize * 3.0 <= mb.host_bytes + 1e-9


def test_plan_pert_batches_errors_when_single_pert_too_big():
    mb = h5ad_manifest.MemBudget(host_bytes=10, gpu_bytes=10)
    with pytest.raises(ValueError, match="exceeds"):
        h5ad_manifest.plan_pert_batches([("A", 100)], n_genes=10, itemsize=4,
                                 control_cells=0, mem_budget=mb, safety=1.0)


def test_plan_pert_batches_error_names_a_budget_that_would_work():
    """#178: the message said the budget was too small but not what would be big enough, so the
    first encounter cost a debugging cycle and the second a bisect -- and because of #146 the
    workable band is narrow, so each trial costs a GPU allocation. The suggested value must
    actually plan."""
    sizes, ng, itemsize, ctrl, safety = [("A", 100)], 10, 4, 50, 3.0
    per_cell = ng * itemsize * safety
    mb = h5ad_manifest.MemBudget(host_bytes=int(60 * per_cell), gpu_bytes=10**12)
    with pytest.raises(ValueError) as ei:
        h5ad_manifest.plan_pert_batches(sizes, n_genes=ng, itemsize=itemsize, control_cells=ctrl,
                                 mem_budget=mb, safety=safety)
    msg = str(ei.value)
    assert "host_bytes" in msg and "is the binding limit" in msg, msg   # names the binding knob
    assert "#146" in msg, msg                                          # and the upper bound
    m = re.search(r"Try host_bytes AND gpu_bytes >= [\d.]+ GiB \((\d+) bytes\)", msg)
    assert m, msg
    suggested = int(m.group(1))
    assert suggested == (ctrl + 100) * per_cell
    # The suggestion is not decorative: at that budget the plan succeeds.
    ok = h5ad_manifest.plan_pert_batches(
        sizes, n_genes=ng, itemsize=itemsize, control_cells=ctrl,
        mem_budget=h5ad_manifest.MemBudget(host_bytes=suggested, gpu_bytes=10**12), safety=safety)
    assert ok == [["A"]]


def test_the_suggested_budget_covers_BOTH_limits_not_just_the_binding_one():
    """codex-review: the usable budget is min(host, gpu), so a suggestion naming only the current
    minimum fails again whenever the OTHER limit is also below `need`. The old test pinned the
    non-binding budget at 10**12, so it could not see this."""
    sizes, ng, itemsize, ctrl, safety = [("A", 100), ("B", 100)], 10, 4, 50, 3.0
    per_cell = ng * itemsize * safety
    tight = int(60 * per_cell)
    mb = h5ad_manifest.MemBudget(host_bytes=tight, gpu_bytes=tight)        # BOTH short, equally
    with pytest.raises(ValueError) as ei:
        h5ad_manifest.plan_pert_batches(sizes, n_genes=ng, itemsize=itemsize, control_cells=ctrl,
                                 mem_budget=mb, safety=safety)
    m = re.search(r"Try host_bytes AND gpu_bytes >= [\d.]+ GiB \((\d+) bytes\)",
                  str(ei.value))
    assert m, str(ei.value)
    suggested = int(m.group(1))
    # Raising ONLY the named binding limit must still fail -- which is the whole point.
    with pytest.raises(ValueError):
        h5ad_manifest.plan_pert_batches(
            sizes, n_genes=ng, itemsize=itemsize, control_cells=ctrl,
            mem_budget=h5ad_manifest.MemBudget(host_bytes=suggested, gpu_bytes=tight), safety=safety)
    # Raising BOTH, as the message says, plans.
    ok = h5ad_manifest.plan_pert_batches(
        sizes, n_genes=ng, itemsize=itemsize, control_cells=ctrl,
        mem_budget=h5ad_manifest.MemBudget(host_bytes=suggested, gpu_bytes=suggested), safety=safety)
    assert [p for b in ok for p in b] == ["A", "B"]


def test_the_suggested_budget_rounds_a_fractional_requirement_UP():
    """`int(need)` truncates, which suggests a value a byte short of sufficient. A non-integer
    safety factor is the way to produce one."""
    sizes, ng, itemsize, ctrl, safety = [("A", 3)], 7, 4, 5, 1.3   # per_cell = 36.4 B
    per_cell = ng * itemsize * safety
    need = (ctrl + 3) * per_cell
    assert need != int(need), "fixture must make the requirement fractional"
    with pytest.raises(ValueError) as ei:
        h5ad_manifest.plan_pert_batches(sizes, n_genes=ng, itemsize=itemsize, control_cells=ctrl,
                                 mem_budget=h5ad_manifest.MemBudget(host_bytes=10, gpu_bytes=10),
                                 safety=safety)
    m = re.search(r">= [\d.]+ GiB \((\d+) bytes\)", str(ei.value))
    assert m and int(m.group(1)) == math.ceil(need), (int(m.group(1)) if m else None, need)
    ok = h5ad_manifest.plan_pert_batches(
        sizes, n_genes=ng, itemsize=itemsize, control_cells=ctrl,
        mem_budget=h5ad_manifest.MemBudget(host_bytes=int(m.group(1)), gpu_bytes=int(m.group(1))),
        safety=safety)
    assert ok == [["A"]]


def test_plan_pert_batches_says_when_the_CONTROL_alone_does_not_fit():
    """#178's other diagnosis. With no headroom every perturbation trips the same raise, and the
    old message blamed whichever one came first -- so a caller who added that perturbation's
    footprint to the budget still had too little."""
    with pytest.raises(ValueError, match="resident control pool alone"):
        h5ad_manifest.plan_pert_batches([("A", 1)], n_genes=10, itemsize=4, control_cells=1000,
                                 mem_budget=h5ad_manifest.MemBudget(host_bytes=1000, gpu_bytes=10**12),
                                 safety=1.0)


def test_plan_pert_batches_error_names_gpu_bytes_when_that_is_the_binding_limit():
    """Which side binds decides which knob to raise; naming the minimum alone sends the caller
    to the wrong one."""
    with pytest.raises(ValueError, match=r"gpu_bytes=.*is the binding limit"):
        h5ad_manifest.plan_pert_batches([("A", 100)], n_genes=10, itemsize=4, control_cells=0,
                                 mem_budget=h5ad_manifest.MemBudget(host_bytes=10**12, gpu_bytes=100),
                                 safety=1.0)


def test_plan_pert_batches_uses_gpu_bound_when_tighter():
    sizes = [("A", 100), ("B", 100), ("C", 100), ("D", 100)]
    ng, itemsize, ctrl = 10, 4, 50
    # gpu is the strictly tighter bound (host huge); ~2 perts/batch under the gpu cap
    mb = h5ad_manifest.MemBudget(host_bytes=10**12, gpu_bytes=250 * 120)
    batches = h5ad_manifest.plan_pert_batches(sizes, n_genes=ng, itemsize=itemsize,
                                       control_cells=ctrl, mem_budget=mb, safety=3.0)
    assert [p for b in batches for p in b] == ["A", "B", "C", "D"]   # full coverage, in order
    for b in batches:                                                # gpu bound respected
        cells = sum(dict(sizes)[p] for p in b) + ctrl
        assert cells * ng * itemsize * 3.0 <= mb.gpu_bytes + 1e-9
    assert max(len(b) for b in batches) <= 2   # gpu strictly tighter than host -> it binds


def test_iter_h5ad_pert_batches_covers_all_noncontrol_perts(tmp_path):
    man = _make_synth_artifact(str(tmp_path), contexts=["x"],
                               perts=["A", "B", "C", "D"], n_cells_per=30, n_genes=8)
    art = h5ad_manifest.read_manifest(man)[0]
    mb = h5ad_manifest.MemBudget(host_bytes=2 * 30 * 8 * 4 * 3, gpu_bytes=10**12)  # ~2 perts/batch
    seen_perts, total_cells = [], 0
    for batch_perts, batch_ad in h5ad_manifest.iter_h5ad_pert_batches(
            art.pred_abs, pert_col="perturbation", control="non-targeting", mem_budget=mb):
        assert set(batch_ad.obs["perturbation"]) == set(batch_perts)
        assert "non-targeting" not in set(batch_ad.obs["perturbation"])   # control excluded
        seen_perts += batch_perts
        total_cells += batch_ad.n_obs
    assert sorted(seen_perts) == ["A", "B", "C", "D"]     # full coverage, no dup
    assert total_cells == 4 * 30


def test_read_group_block_reads_control(tmp_path):
    man = _make_synth_artifact(str(tmp_path), contexts=["x"], perts=["A", "B"], n_cells_per=25)
    art = h5ad_manifest.read_manifest(man)[0]
    ctrl = h5ad_manifest.read_group_block(art.real_abs, pert_col="perturbation", labels={"non-targeting"})
    assert ctrl.n_obs == 25 and set(ctrl.obs["perturbation"]) == {"non-targeting"}


def test_read_group_block_raises_on_no_match(tmp_path):
    man = _make_synth_artifact(str(tmp_path), contexts=["x"], perts=["A", "B"], n_cells_per=20)
    art = h5ad_manifest.read_manifest(man)[0]
    with pytest.raises(ValueError):
        h5ad_manifest.read_group_block(art.real_abs, pert_col="perturbation", labels={"NOSUCHPERT"})


def test_h5ad_batch_source_matches_helpers(tmp_path):
    # H5adBatchSource must reproduce read_group_block + iter_h5ad_pert_batches exactly.
    from cell_eval2 import MemBudget
    from cell_eval2.h5ad_manifest import H5adBatchSource
    man = _make_synth_artifact(str(tmp_path), contexts=["x"], perts=["A", "B", "C"],
                               n_cells_per=30, n_genes=8)
    art = h5ad_manifest.read_manifest(man)[0]
    src = H5adBatchSource(art.real_abs, pert_col="perturbation", control="non-targeting")
    assert src.control == "non-targeting" and src.stream_tag == art.real_abs
    cb = src.read_control_block()
    assert set(cb.obs["perturbation"].astype(str)) == {"non-targeting"}
    mem = MemBudget(host_bytes=10**9, gpu_bytes=10**9)
    perts = []
    for bp, bad in src.iter_pert_batches(mem):
        assert "non-targeting" not in bp
        assert set(bad.obs["perturbation"].astype(str)) == set(bp)
        perts += bp
    assert "non-targeting" not in perts and len(perts) == len(set(perts))
    assert sorted(perts) == ["A", "B", "C"]


def test_assemble_score_result_nan_skip_and_means():
    import polars as pl
    from cell_eval2.h5ad_manifest import _assemble_score_result
    frame = pl.DataFrame({
        "dataset": ["d", "d", "d", "d"], "panel_id": [0, 0, 0, 0],
        "context": ["c", "c", "c", "c"], "perturbation": ["P1", "P1", "P2", "P2"],
        "metric": ["mae", "x", "mae", "x"], "value": [0.2, float("nan"), 0.4, 1.0],
    })
    res = _assemble_score_result([frame])
    pc = {(r["context"], r["metric"]): r["value"] for r in res.per_context.iter_rows(named=True)}
    assert abs(pc[("c", "mae")] - 0.3) < 1e-12           # mean(0.2,0.4)
    assert abs(pc[("c", "x")] - 1.0) < 1e-12             # NaN skipped -> mean(1.0)
    ov = {r["metric"]: r["value"] for r in res.overall.iter_rows(named=True)}
    assert abs(ov["mae"] - 0.3) < 1e-12


@pytestmark_gpu
def test_build_reference_streaming_matches_materialized(tmp_path):
    """The streaming reference bundle produces the same metrics as the materialized
    build_reference: same real DE table (per target) and same real pseudobulk means."""
    import numpy as np
    import polars as pl
    from cell_eval2 import h5ad_manifest
    from cell_eval2.config import EvalConfig
    from cell_eval2.partition_inmem import build_reference, build_reference_streaming

    man = _make_synth_artifact(str(tmp_path), contexts=["x"],
                               perts=["A", "B", "C"], n_cells_per=60, n_genes=10)
    art = h5ad_manifest.read_manifest(man)[0]
    cfg = EvalConfig.v2()  # metrics defaults to "full"
    cfg.pert_col = "perturbation"  # manifest/synthetic column; default is "target"

    # materialized reference over the whole real context (fits in RAM here)
    import anndata as ad
    whole = ad.read_h5ad(art.real_abs)
    ref_dir = str(tmp_path / "ref_whole")
    comparator = resolved_comparator(
        cfg, pred_input_type="counts", real_input_type="counts")
    build_reference(
        whole, config=cfg, cache_dir=ref_dir, control_format="h5ad",
        comparator=comparator)

    # streaming reference, batched
    stream_dir = str(tmp_path / "ref_stream")
    mb = h5ad_manifest.MemBudget(host_bytes=60 * 10 * 4 * 3 * 2, gpu_bytes=10**12)  # ~1-2 perts/batch
    build_reference_streaming(art.real_abs, config=cfg, cache_dir=stream_dir,
                              control="non-targeting", mem_budget=mb, input_type="counts",
                              comparator=comparator)

    # real DE tables equal (sort for order-independence)
    de_a = pl.read_parquet(os.path.join(ref_dir, "real_de.parquet")).sort(["target", "feature"])
    de_b = pl.read_parquet(os.path.join(stream_dir, "real_de.parquet")).sort(["target", "feature"])
    assert de_a.select(["target", "feature", "log2_fold_change", "p_adj"]).equals(
        de_b.select(["target", "feature", "log2_fold_change", "p_adj"]))

    # real pseudobulk means equal per perturbation. ⚠️ The artifact is named for the
    # RESOLVED comparator, not for a fixed normalization: this test already resolves it
    # above, and since #264 PR2 a v2 counts run writes `real_pseudobulk_bulk_lognorm.npz`
    # while the lognorm fallback writes `..._lognorm.npz`. Hard-coding either one makes the
    # test a FileNotFoundError in the other regime -- which is exactly how it was failing.
    art_npz = f"real_pseudobulk_{comparator}.npz"
    with np.load(os.path.join(ref_dir, art_npz)) as za, \
         np.load(os.path.join(stream_dir, art_npz)) as zb:
        ma = dict(zip(za["perts"].tolist(), za["means"]))
        mb_ = dict(zip(zb["perts"].tolist(), zb["means"]))
        assert set(ma) == set(mb_)
        for p in ma:
            np.testing.assert_allclose(ma[p], mb_[p], rtol=1e-5, atol=1e-6)


@pytestmark_gpu
def test_score_h5ad_manifest_end_to_end(tmp_path):
    import polars as pl
    from cell_eval2 import h5ad_manifest
    from cell_eval2.config import EvalConfig

    man = _make_synth_artifact(str(tmp_path), contexts=["ctxA", "ctxB"],
                               perts=["A", "B", "C"], n_cells_per=50, n_genes=10)
    cfg = EvalConfig.v2()  # metrics defaults to "full"
    # Moment-consuming expression metrics are unavailable on this driver (#198).
    cfg.metrics = full_minus_moments()
    mb = h5ad_manifest.MemBudget(host_bytes=50 * 10 * 4 * 3 * 2, gpu_bytes=10**12)
    res = h5ad_manifest.score_h5ad_manifest(man, config=cfg, mem_budget=mb)

    # per_pert: every non-control pert of every context, with dataset/panel_id/context columns
    assert set(res.per_pert.columns) == {"dataset", "panel_id", "context", "perturbation",
                                         "metric", "value"}
    assert set(res.per_pert["context"].unique()) == {"ctxA", "ctxB"}
    assert set(res.per_pert["perturbation"].unique()) == {"A", "B", "C"}
    # PDS + nsig_spearman present (global metrics), within context
    metrics = set(res.per_pert["metric"].unique())
    assert "pds_l1" in metrics and "de_wilcoxon_nsig_spearman" in metrics
    # per_context = MetricSpec.agg over perts (per dataset/panel_id/context unit) with the `agg`
    # column #233 added; overall = mean over units, and deliberately carries no `agg`.
    assert set(res.per_context.columns) == {"dataset", "panel_id", "context", "metric", "value",
                                            "agg"}
    assert res.per_context.columns[-1] == "agg", "appended last -- a released consumer reads these frames"
    assert set(res.overall.columns) == {"metric", "value"}
    # overall equals unweighted mean of per_context across contexts (per metric)
    chk = (res.per_context.group_by("metric").agg(pl.col("value").mean().alias("v"))
           .sort("metric"))
    got = res.overall.sort("metric").join(chk, on="metric")
    import numpy as np
    m = got.filter(pl.col("value").is_not_nan() & pl.col("v").is_not_nan())
    np.testing.assert_allclose(m["value"].to_numpy(), m["v"].to_numpy(), rtol=1e-6)


@pytestmark_gpu
def test_h5ad_manifest_scorer_distinguishes_colliding_context_across_panels(tmp_path):
    """Fix 1 regression: the manifest's unique scoring unit is (dataset, panel_id, context), not
    context alone. Build a manifest with TWO units that share the SAME context string but
    differ in panel_id, and assert per_pert keeps them as distinct rows (no pooling), and
    that overall averages unweighted over the 2 units."""
    import numpy as np
    import polars as pl
    from cell_eval2 import h5ad_manifest
    from cell_eval2.config import EvalConfig

    root = str(tmp_path)
    shared_ctx = "shared_ctx"
    rows = []
    for panel_id in (0, 1):
        cdir = os.path.join(root, "CCL_1", f"panel_{panel_id}", shared_ctx)
        n_cells, ng = _write_context_h5ad(
            cdir, perts=["A", "B"], control="non-targeting",
            n_cells_per=30, n_genes=8, seed=200 + panel_id)
        rows.append({
            "dataset": "CCL_1", "panel_id": panel_id, "context": shared_ctx,
            "control_value": "non-targeting",
            "path_real": os.path.relpath(os.path.join(cdir, "adata_real.h5ad"), root),
            "path_pred": os.path.relpath(os.path.join(cdir, "adata_pred.h5ad"), root),
            "var_names_path": os.path.relpath(
                os.path.join(root, "CCL_1", f"panel_{panel_id}", "var_names.npy"), root),
            "n_cells": n_cells, "n_genes": ng,
        })
    man = os.path.join(root, "manifest.csv")
    pd.DataFrame(rows).to_csv(man, index=False)

    cfg = EvalConfig.v2()  # metrics defaults to "full"
    # Moment-consuming expression metrics are unavailable on this driver (#198).
    cfg.metrics = full_minus_moments()
    mb = h5ad_manifest.MemBudget(host_bytes=30 * 8 * 4 * 3 * 2, gpu_bytes=10**12)
    res = h5ad_manifest.score_h5ad_manifest(man, config=cfg, mem_budget=mb)

    # both panel_0 and panel_1 units are present under the shared context string -- distinct
    # (dataset, panel_id) rows, not pooled into one.
    assert (res.per_pert.filter(pl.col("context") == shared_ctx)
            .select(["dataset", "panel_id"]).unique().height == 2)
    # per_context keeps the two units separate: 2 rows per metric under the shared context.
    counts = (res.per_context.filter(pl.col("context") == shared_ctx)
              .group_by("metric").agg(pl.len().alias("n")))
    assert (counts["n"] == 2).all()
    # overall = unweighted mean over all (dataset, panel_id, context) units in per_context.
    chk = (res.per_context.group_by("metric").agg(pl.col("value").mean().alias("v"))
           .sort("metric"))
    got = res.overall.sort("metric").join(chk, on="metric")
    m = got.filter(pl.col("value").is_not_nan() & pl.col("v").is_not_nan())
    np.testing.assert_allclose(m["value"].to_numpy(), m["v"].to_numpy(), rtol=1e-6)


@pytestmark_gpu
def test_h5ad_manifest_scorer_matches_whole_context_compute_metrics(tmp_path):
    """Acceptance linchpin: for one context, the streamed per_pert metrics must equal
    whole-context v2 compute_metrics for every (perturbation, metric), since the partition
    split is bit-identical under full coverage."""
    import anndata as ad
    import numpy as np
    from cell_eval2 import h5ad_manifest
    from cell_eval2.config import EvalConfig
    from cell_eval2.run import compute_metrics

    man = _make_synth_artifact(str(tmp_path), contexts=["only"],
                               perts=["A", "B", "C", "D"], n_cells_per=80, n_genes=12)
    art = h5ad_manifest.read_manifest(man)[0]
    cfg = EvalConfig.v2()  # metrics defaults to "full"
    # Both sides use the same reduced list: moment-consuming expression metrics are
    # unavailable on the partitioned driver (#198).
    cfg.metrics = full_minus_moments()
    cfg.pert_col = "perturbation"  # manifest/synthetic column; default is "target"
    # align whole-context AUC floor to the partitioned path's fixed floor
    # (_require_partition_config), so pr_auc/roc_auc match bit-for-bit. DEParams is frozen, so
    # replace() the de field (cfg itself stays mutable).
    from dataclasses import replace
    cfg.de = replace(cfg.de, auc_pval_floor="replace_zero", auc_pval_floor_value=1e-10)

    # whole-context reference: compute_metrics over the full pred/real context AnnData
    pred = ad.read_h5ad(art.pred_abs)
    real = ad.read_h5ad(art.real_abs)
    whole = compute_metrics(pred, real, config=cfg)   # tidy (perturbation, metric, value)

    # streamed, forced to multiple batches (small budget)
    mb = h5ad_manifest.MemBudget(host_bytes=80 * 12 * 4 * 3 * 2, gpu_bytes=10**12)
    res = h5ad_manifest.score_h5ad_manifest(man, config=cfg, mem_budget=mb)
    streamed = res.per_pert.drop(["dataset", "panel_id", "context"])

    key = ["perturbation", "metric"]
    j = (whole.rename({"value": "whole"}).join(
         streamed.rename({"value": "stream"}), on=key, how="inner")).sort(key)
    # every (pert, metric) present on both sides
    assert j.height == whole.height == streamed.height
    w, s = j["whole"].to_numpy(), j["stream"].to_numpy()
    both_nan = np.isnan(w) & np.isnan(s)
    np.testing.assert_allclose(w[~both_nan], s[~both_nan], rtol=1e-5, atol=1e-6)


@pytestmark_gpu
def test_h5ad_manifest_scorer_log1p_input_smoke(tmp_path):
    """A scaled_log1p artifact scores end-to-end: input_type resolves to 'lognorm',
    the CPM filter is skipped, and every context yields per-pert metrics (spec: accept
    both counts and log1p)."""
    from cell_eval2 import h5ad_manifest
    from cell_eval2.config import EvalConfig

    man = _make_synth_artifact(str(tmp_path), contexts=["x"], perts=["A", "B", "C"],
                               n_cells_per=60, n_genes=10, log1p=True)
    cfg = EvalConfig.v2()  # metrics defaults to "full"
    # Moment-consuming expression metrics are unavailable on this driver (#198).
    cfg.metrics = full_minus_moments()
    art = h5ad_manifest.read_manifest(man)[0]
    # directly verifies the docstring's claim that input_type resolves to 'lognorm'
    assert h5ad_manifest._resolve_input_type_h5ad(art.real_abs, cfg=cfg) == "lognorm"
    mb = h5ad_manifest.MemBudget(host_bytes=60 * 10 * 4 * 3 * 2, gpu_bytes=10**12)
    res = h5ad_manifest.score_h5ad_manifest(man, config=cfg, mem_budget=mb)
    assert set(res.per_pert["perturbation"].unique()) == {"A", "B", "C"}
    assert "pds_l1" in set(res.per_pert["metric"].unique())
    assert res.overall.height > 0


@pytestmark_gpu
def test_run_h5ad_manifest_cli_writes_outputs(tmp_path):
    """The CLI runner parses arguments, invokes score_h5ad_manifest, and writes overall.csv."""
    import subprocess
    import sys
    man = _make_synth_artifact(str(tmp_path), contexts=["x"], perts=["A", "B"], n_cells_per=40, n_genes=8)
    out = str(tmp_path / "out")
    # Resolve the CLI script absolutely from the repo root (tests/ -> repo root), so the test is
    # independent of the process cwd (pytest may run from outside the repo, e.g. over ssh).
    # ⚠️ Spelled as ONE literal, not joined segment-by-segment, so the publication gate that
    # forbids a shipping file naming an unshipped path can actually SEE it.
    cli = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "tools/h5ad_manifest/run_h5ad_manifest.py")
    # ⚠️ THE TOOL DOES NOT TRAVEL. `tools/**` is DROPped from the public cut, so in the public
    # tree this `cli` does not exist and `subprocess.run` would return 2 -- a shipped test failing
    # on every public GPU host, invisible to CPU CI because the module is GPU-gated. Guarded the
    # same way tests/test_score.py and tests/test_baseline_emission.py guard their tool imports:
    # it runs here and skips there. Found by the codex review of the dangling-path rule, not by a
    # suite run -- a GPU-gated test cannot fail on a CPU runner.
    if not os.path.isfile(cli):
        pytest.skip("run_h5ad_manifest.py is an internal tool and does not ship to the public cut")
    r = subprocess.run(
        [sys.executable, cli, "--manifest", man,
         "--host-bytes", str(40 * 8 * 4 * 3 * 4), "--gpu-bytes", str(10**12), "--outdir", out],
        capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert os.path.isfile(os.path.join(out, "overall.csv"))


@pytestmark_gpu
def test_score_h5ad_manifest_pred_control(tmp_path):
    from dataclasses import replace

    from cell_eval2.config import EvalConfig
    man = _make_synth_artifact(str(tmp_path), contexts=["ctxA"],
                               perts=["GENE1", "GENE2", "GENE3"], log1p=True)
    # v1: the profile string lets resolve_metrics filter v2-native metrics silently; an explicit list would raise (#198).
    cfg = EvalConfig.from_preset("cell-eval-0.7.6")     # v1/pred/clip, pert_col="target"
    # synth artifacts use obs["perturbation"]; align the config's pert_col.
    cfg = replace(cfg, pert_col="perturbation")
    res = h5ad_manifest.score_h5ad_manifest(
        man, config=cfg, mem_budget=h5ad_manifest.MemBudget(host_bytes=1 << 34, gpu_bytes=1 << 34))
    # v1 output name for nsig spearman is de_spearman_sig; it must be present per-pert.
    metrics = set(res.per_pert["metric"].unique())
    assert "de_spearman_sig" in metrics
    assert set(res.per_pert["perturbation"].unique()) == {"GENE1", "GENE2", "GENE3"}


# --- #233: _assemble_score_result must honour MetricSpec.agg -----------------------------------

def _pp(rows):
    """per-pert frame in _assemble_score_result's input shape."""
    import polars as pl
    return pl.DataFrame(
        {"dataset": [r[0] for r in rows], "panel_id": [r[1] for r in rows],
         "context": [r[2] for r in rows], "perturbation": [r[3] for r in rows],
         "metric": [r[4] for r in rows], "value": [r[5] for r in rows]},
        schema={"dataset": pl.Utf8, "panel_id": pl.Int64, "context": pl.Utf8,
                "perturbation": pl.Utf8, "metric": pl.Utf8, "value": pl.Float64},
    )


def test_assemble_score_result_honours_a_median_agg_at_the_perturbation_level(monkeypatch):
    """#233: the reduction was an unconditional mean, and two docstrings claimed it "matches
    run.aggregate_metrics". No shipped metric declares median since #231, so this is exercised
    the same way run.py's own median branch is -- by injecting the statistic onto a metric
    (tests/test_metric_aggregation.py::test_median_agg_is_honoured_by_both_aggregators).

    The values are chosen so mean and median differ a lot: [1, 2, 60] -> mean 21, median 2.
    """
    from dataclasses import replace as _replace

    import cell_eval2.run as run_mod
    monkeypatch.setitem(run_mod.CATALOG, "expr_mae",
                        _replace(run_mod.CATALOG["expr_mae"], agg="median"))
    df = _pp([("D", 0, "ctxA", p, "expr_mae", v)
              for p, v in (("A", 1.0), ("B", 2.0), ("C", 60.0))]
             + [("D", 0, "ctxA", p, "pds_cosine", v)
                for p, v in (("A", 1.0), ("B", 2.0), ("C", 60.0))])
    res = h5ad_manifest._assemble_score_result([df])
    got = {r["metric"]: (r["value"], r["agg"])
           for r in res.per_context.iter_rows(named=True)}
    assert got["expr_mae"] == (pytest.approx(2.0), "median")      # the declared statistic
    assert got["pds_cosine"] == (pytest.approx(21.0), "mean")     # untouched neighbour
    # And it agrees with the library aggregator on the same values -- the claim the docstring made.
    from cell_eval2.run import aggregate_metrics
    lib = {r["metric"]: r["mean"] for r in aggregate_metrics(
        df.select(["perturbation", "metric", "value"])).iter_rows(named=True)}
    assert lib["expr_mae"] == pytest.approx(got["expr_mae"][0])
    assert lib["pds_cosine"] == pytest.approx(got["pds_cosine"][0])


def test_assemble_score_result_overall_is_a_MEAN_over_contexts_even_for_a_median_metric(
        monkeypatch):
    """Decision 1 of #233, pinned. `overall` reduces over CONTEXTS -- a small designed set, not
    the heavy-tailed per-perturbation population MetricSpec.agg is about -- so it stays an
    unweighted mean. A median here would also make the overall value depend on how a manifest
    happened to be partitioned."""
    from dataclasses import replace as _replace

    import cell_eval2.run as run_mod
    monkeypatch.setitem(run_mod.CATALOG, "expr_mae",
                        _replace(run_mod.CATALOG["expr_mae"], agg="median"))
    # Three contexts whose per-context medians are 1, 2 and 60: mean 21, median 2.
    rows = []
    for ctx, vals in (("c1", (0.0, 1.0, 1.0)), ("c2", (2.0, 2.0, 3.0)), ("c3", (59.0, 60.0, 61.0))):
        rows += [("D", 0, ctx, f"p{i}", "expr_mae", v) for i, v in enumerate(vals)]
    res = h5ad_manifest._assemble_score_result([_pp(rows)])
    pc = sorted(r["value"] for r in res.per_context.iter_rows(named=True))
    assert pc == [pytest.approx(1.0), pytest.approx(2.0), pytest.approx(60.0)]   # medians
    assert res.overall["value"].to_list() == [pytest.approx(21.0)]               # MEAN of those
    assert "agg" not in res.overall.columns


def test_assemble_score_result_refuses_a_derived_metrics_per_perturbation_rows():
    """A ratio_of_sums member cannot be reduced from per-perturbation values, so a row bearing
    its name must raise rather than be silently meaned -- the same class of defect #233 reports.
    Unreachable through the shipped drivers (#270: they never emit it), which is exactly when a
    guard is cheap."""
    with pytest.raises(ValueError, match="derived metric"):
        h5ad_manifest._assemble_score_result([_pp([
            ("D", 0, "ctxA", "A", "expr_mse_unbiased_capped_norm", 0.5)])])
