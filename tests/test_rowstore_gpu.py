"""GPU linchpin: full-profile (cell-eval-0.7.6 preset) score_rowstore == whole compute_metrics
on a synthetic row store. Drives gpudge DE both sides -> needs a CUDA GPU; SKIPS on CPU."""
import json
from dataclasses import replace

import anndata as ad
import numpy as np
import pandas as pd
import polars as pl
import pytest

from cell_eval2 import MemBudget, compute_metrics, score_rowstore
from cell_eval2 import rowstore as rs
from cell_eval2.config import EvalConfig


def _no_gpu():
    try:
        import cupy
        return cupy.cuda.runtime.getDeviceCount() == 0
    except Exception:
        return True


pytestmark = pytest.mark.skipif(_no_gpu(), reason="needs a CUDA GPU (gpudge DE both sides)")


def _cfg():
    # cell-eval-0.7.6 preset, pert_col aligned to the synthetic row-store obs column.
    # exclude_target_gene=False deliberately (#275): both fixtures label perturbations A-D
    # against genes g0-gN, so nothing resolves and #248's gate refuses to score at all. These
    # are parity tests -- they need the guard not to fire, and both sides of each comparison
    # share this config, so the equality asserted is unaffected. `replace` on the preset's OWN
    # DiscriminationParams, never a fresh one: the preset carries distance='l1',
    # rank_denominator='n' and tie_policy='position', which a bare DiscriminationParams
    # would silently reset to the v2 defaults.
    cfg = replace(EvalConfig.from_preset("cell-eval-0.7.6"), pert_col="perturbation")
    return replace(cfg, discrimination=replace(cfg.discrimination, exclude_target_gene=False))


def _write_rowstore(root, *, ctrl, labels, var_names, real, pred):
    staging = root / "staging"
    adir = staging / "artifact_00000"
    adir.mkdir(parents=True)
    np.ascontiguousarray(real, np.uint16).tofile(adir / "real_X.dat")
    np.ascontiguousarray(pred, np.uint16).tofile(adir / "pred_X.dat")
    pd.DataFrame({"dataset": "tahoe", "context": "C1", "perturbation": list(labels),
                  "control_value": ctrl, "file_idx": 0,
                  "cell_idx": list(range(len(labels)))}).to_csv(adir / "obs.csv", index=False)
    np.save(adir / "var_names.npy", np.asarray(var_names, dtype=np.str_))
    (staging / "plan.json").write_text(json.dumps({"staging_dir": "/bogus", "artifacts": [{
        "artifact_id": "artifact_00000", "dataset": "tahoe", "dataset_slug": "tahoe",
        "panel_id": 0, "context": "C1", "context_slug": "C1", "control_value": ctrl,
        "gene_ids": list(range(len(var_names))), "var_names": list(var_names),
        "n_rows": len(labels), "n_genes": len(var_names), "dtype": "uint16",
        "real_path": "/b/real_X.dat", "pred_path": "/b/pred_X.dat", "written_path": "/b/w.dat",
        "obs_path": "/b/obs.csv", "var_names_path": "/b/var_names.npy"}]}))
    return staging


def test_score_rowstore_full_preset_matches_whole(tmp_path):
    rng = np.random.default_rng(3)
    ctrl = "[('DMSO', 0.0, 'uM')]"
    labels = np.array([ctrl] * 80 + sum(([g] * 50 for g in ["A", "B", "C", "D"]), []))
    n, g = labels.size, 24
    real = rng.integers(0, 80, size=(n, g), dtype=np.uint16)
    pred = rng.integers(0, 80, size=(n, g), dtype=np.uint16)
    var_names = [f"g{j}" for j in range(g)]
    staging = _write_rowstore(tmp_path, ctrl=ctrl, labels=labels, var_names=var_names,
                              real=real, pred=pred)

    cfg = _cfg()
    mem = MemBudget(host_bytes=10**9, gpu_bytes=10**9)
    res = score_rowstore(staging, config=cfg, mem_budget=mem)

    xr, xp = rs.scaled_log1p(real, 1e4), rs.scaled_log1p(pred, 1e4)
    obs = pd.DataFrame({"perturbation": labels})
    var = pd.DataFrame(index=pd.Index(var_names, name="gene"))
    ref = compute_metrics(ad.AnnData(X=xp, obs=obs.copy(), var=var.copy()),
                          ad.AnnData(X=xr, obs=obs.copy(), var=var.copy()),
                          config=replace(cfg, control=ctrl))

    got = {(r["perturbation"], r["metric"]): r["value"] for r in res.per_pert.iter_rows(named=True)}
    want = {(r["perturbation"], r["metric"]): r["value"] for r in ref.iter_rows(named=True)}
    assert set(got) == set(want), f"key mismatch: {set(got) ^ set(want)}"
    for k, wv in want.items():
        gv = got[k]
        if wv != wv:
            assert gv != gv, f"{k}: whole NaN vs {gv}"
        else:
            assert abs(wv - gv) <= 1e-9 + 1e-6 * abs(wv), f"{k}: whole {wv} vs rowstore {gv}"


def test_score_rowstore_sparse_matches_whole(tmp_path):
    """Partition-invariance on a REALISTICALLY SPARSE store.

    The fixture above is ~98.8% dense (rng.integers(0, 80)), so it selects the DENSE path and never
    exercises CSR. This one is sparse, so score_rowstore takes the CSR path while the in-memory
    reference stays dense -- making it the end-to-end proof that CSR changes the container and not
    the numbers, through gpudge, under the full preset.
    """
    rng = np.random.default_rng(7)
    ctrl = "[('DMSO', 0.0, 'uM')]"
    labels = np.array([ctrl] * 80 + sum(([g] * 50 for g in ["A", "B", "C", "D"]), []))
    n, g = labels.size, 32

    def _blk():
        m = (rng.random((n, g)) < 0.15) * rng.integers(1, 80, size=(n, g))
        return np.ascontiguousarray(m, dtype=np.uint16)

    real, pred = _blk(), _blk()
    var_names = [f"g{j}" for j in range(g)]
    staging = _write_rowstore(tmp_path, ctrl=ctrl, labels=labels, var_names=var_names,
                              real=real, pred=pred)

    # Guard the premise: if the fixture ever drifts dense, this test silently stops testing CSR.
    art = rs.read_rowstore_plan(staging)[0]
    src = rs.RowStoreBatchSource(art, side="real", pert_col="perturbation", control=ctrl)
    assert src.use_sparse, f"fixture must select the CSR path (density={src.density:.3f})"

    cfg = _cfg()
    mem = MemBudget(host_bytes=10**9, gpu_bytes=10**9)
    res = score_rowstore(staging, config=cfg, mem_budget=mem)

    xr, xp = rs.scaled_log1p(real, 1e4), rs.scaled_log1p(pred, 1e4)     # DENSE in-memory reference
    obs = pd.DataFrame({"perturbation": labels})
    var = pd.DataFrame(index=pd.Index(var_names, name="gene"))
    ref = compute_metrics(ad.AnnData(X=xp, obs=obs.copy(), var=var.copy()),
                          ad.AnnData(X=xr, obs=obs.copy(), var=var.copy()),
                          config=replace(cfg, control=ctrl))

    got = {(r["perturbation"], r["metric"]): r["value"] for r in res.per_pert.iter_rows(named=True)}
    want = {(r["perturbation"], r["metric"]): r["value"] for r in ref.iter_rows(named=True)}
    assert set(got) == set(want), f"key mismatch: {set(got) ^ set(want)}"
    for k, wv in want.items():
        gv = got[k]
        if wv != wv:
            assert gv != gv, f"{k}: whole NaN vs {gv}"
        else:
            assert abs(wv - gv) <= 1e-9 + 1e-6 * abs(wv), f"{k}: whole {wv} vs rowstore {gv}"


def test_target_gene_exclusion_is_live_through_the_rowstore(tmp_path):
    """`exclude_target_gene=True` survives `score_rowstore` and MOVES the numbers.

    The two parity tests above turn exclusion OFF (#275) because their A-D labels resolve to
    no gene in the g0-gN panel and #248's gate refuses to score at all. That is the honest fix
    for a parity assertion, but on its own it would leave the rowstore driver with no
    end-to-end proof that exclusion still reaches `resolve_exclusion_columns` at all -- the
    exact gap codex flagged reviewing #275.

    Supply the `target_gene_map` arm of #248's remedy instead, and pin both halves: it scores
    (so the map is genuinely consumed) and it disagrees with the exclusion-off run (so an
    exclusion that quietly became a no-op fails here). `real` and `pred` are independent draws,
    so pds is not saturated and a re-ranking is visible.
    """
    rng = np.random.default_rng(23)
    ctrl = "[('DMSO', 0.0, 'uM')]"
    labels = np.array([ctrl] * 80 + sum(([g] * 50 for g in ["A", "B", "C", "D"]), []))
    n, g = labels.size, 24
    real = rng.integers(0, 80, size=(n, g), dtype=np.uint16)
    pred = rng.integers(0, 80, size=(n, g), dtype=np.uint16)
    var_names = [f"g{j}" for j in range(g)]
    staging = _write_rowstore(tmp_path, ctrl=ctrl, labels=labels, var_names=var_names,
                              real=real, pred=pred)

    mem = MemBudget(host_bytes=10**9, gpu_bytes=10**9)
    off = _cfg()                                            # exclusion off (the #275 knob)
    on = replace(off, target_gene_map={p: f"g{i}" for i, p in enumerate("ABCD")},
                 discrimination=replace(off.discrimination, exclude_target_gene=True))

    res_on = score_rowstore(staging, config=on, mem_budget=mem).per_pert
    res_off = score_rowstore(staging, config=off, mem_budget=mem).per_pert

    # Compare ONLY the discrimination rows. The v1 preset emits 29 metric columns, most of
    # them gpudge DE; letting any of those count would mean run-to-run jitter from a second
    # GPU execution could satisfy `moved > 0` on its own and the assertion would pass without
    # exclusion having done anything -- a test that cannot fail. `target_gene_map` is expected
    # to move the discrimination rows and nothing else, so that is what gets asserted.
    _pds = pl.col("metric").str.starts_with("discrimination_score_")
    pds_on, pds_off = res_on.filter(_pds), res_off.filter(_pds)
    assert pds_on.height, (
        "the preset emitted no discrimination metric, so the comparison below would be "
        f"vacuous; emitted: {sorted(res_on['metric'].unique())}"
    )
    # JOIN on the keys rather than subtracting the two value columns positionally: equal
    # heights do NOT imply equal key sets, and a positional diff would silently compare
    # different perturbations against each other. The inner-join height equalling BOTH inputs
    # is the key-set assertion -- and it also catches a fan-out, which matters because
    # per_pert is really keyed on (dataset, panel_id, context, perturbation, metric) and this
    # fixture is single-artifact. Project first so the join does not carry dataset_off /
    # panel_id_off / context_off into every failure message.
    _keep = ["perturbation", "metric", "value"]
    j = pds_on.select(_keep).join(pds_off.select(_keep), on=["perturbation", "metric"],
                                  how="inner", suffix="_off")
    assert j.height == pds_on.height == pds_off.height, (
        "exclusion changed which discrimination rows are emitted"
    )
    # `Series.is_finite().all()` IGNORES nulls -- pl.Series([1.0, None]).is_finite().all() is
    # True on polars 1.43.2 -- so a null would sail through a finiteness check. Assert the
    # absence of nulls separately, then finiteness. A one-sided NaN must FAIL loudly rather
    # than count as either movement or agreement: pds ranks are finite here (four
    # perturbations, ties give finite positional ranks, a zero-norm cosine operand yields
    # distance 1.0), so a non-finite value means something else broke.
    assert j["value"].null_count() == 0 and j["value_off"].null_count() == 0, (
        f"null discrimination values: {j.filter(pl.col('value').is_null() | pl.col('value_off').is_null())}"
    )
    assert j["value"].is_finite().all() and j["value_off"].is_finite().all(), (
        f"discrimination values must be finite on both arms; got "
        f"{j.filter(~(pl.col('value').is_finite() & pl.col('value_off').is_finite()))}"
    )
    moved = ((j["value"] - j["value_off"]).abs() > 1e-12).sum()
    assert moved > 0, (
        "exclude_target_gene=True produced discrimination values identical to "
        "exclude_target_gene=False through score_rowstore; the exclusion is a no-op on the "
        "rowstore driver"
    )
