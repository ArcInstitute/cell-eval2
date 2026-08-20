"""SP2 pred-control reference + score_piece pred mode. The pseudobulk/write test is CPU-safe
(no DE); the score_piece test needs gpudge DE and skips on a CPU node."""
import os
from dataclasses import replace

import anndata as ad
import numpy as np
import pytest

from cell_eval2 import partition_inmem
from cell_eval2.config import EvalConfig
from _helpers import resolved_comparator


def _no_gpu():
    try:
        import cupy
        return cupy.cuda.runtime.getDeviceCount() == 0
    except Exception:
        return True


def _preset():
    # cell-eval-0.7.6: v1/pred/clip. pert_col defaults to "target"; use it as-is.
    # v1: the profile string lets resolve_metrics filter v2-native metrics silently; an explicit list would raise (#198).
    # exclude_target_gene=False deliberately (#275): this fixture's labels are A/B/C, and the
    # panel they are matched against is var_names "0".."11" -- NOT "g0".."g11". `var={"gene":
    # [...]}` puts those labels in a var COLUMN and leaves AnnData to mint a default RangeIndex,
    # so the measured gene names are the stringified integers. Either way nothing resolves and
    # #248's gate refuses to score at all. `replace` on
    # the preset's OWN DiscriminationParams, never a fresh one -- the preset carries
    # distance='l1', rank_denominator='n' and tie_policy='position', and a bare
    # DiscriminationParams(exclude_target_gene=False) would silently reset all three to v2.
    cfg = EvalConfig.from_preset("cell-eval-0.7.6")
    return replace(cfg, discrimination=replace(cfg.discrimination, exclude_target_gene=False))


def _pred_h5ad(tmp_path):
    rng = np.random.default_rng(7)
    labels = ["non-targeting"] * 50 + sum(([g] * 40 for g in ["A", "B", "C"]), [])
    # lognorm input (preset input_type=lognorm): log1p of counts.
    X = np.log1p(rng.poisson(3, size=(len(labels), 12)).astype(np.float32))
    a = ad.AnnData(X=X, obs={"target": labels}, var={"gene": [f"g{i}" for i in range(12)]})
    p = str(tmp_path / "pred.h5ad")
    a.write_h5ad(p)
    return p


def test_build_pred_control_reference_writes_control_and_pseudobulk(tmp_path, monkeypatch):
    # build_pred_control_reference does no DE (only pseudobulk + control-cell write), so the
    # gpudge-backend gate in _require_partition_config is bypassed here to run CPU-only.
    monkeypatch.setattr(partition_inmem, "_resolve_backend", lambda b: "gpudge")
    cfg = _preset()
    cache = str(tmp_path / "ref")
    os.makedirs(cache, exist_ok=True)
    partition_inmem.build_pred_control_reference(
        _pred_h5ad(tmp_path), config=cfg, cache_dir=cache,
        control="non-targeting", input_type="lognorm",
        comparator=resolved_comparator(
            cfg, pred_input_type="lognorm", real_input_type="lognorm"))
    assert os.path.isfile(os.path.join(cache, "pred_control.h5ad"))
    ctrl = ad.read_h5ad(os.path.join(cache, "pred_control.h5ad"))
    assert set(ctrl.obs["target"].astype(str)) == {"non-targeting"}
    # at least one pred_pseudobulk_<norm>.npz exists, each holding exactly the control row
    npzs = [f for f in os.listdir(cache) if f.startswith("pred_pseudobulk_")]
    assert npzs
    with np.load(os.path.join(cache, npzs[0])) as z:
        assert list(z["perts"].astype(str)) == ["non-targeting"]
        assert z["means"].shape[0] == 1


@pytest.mark.skipif(_no_gpu(), reason="needs a CUDA GPU (gpudge DE)")
def test_score_piece_pred_mode_emits_expr_and_de(tmp_path):
    from cell_eval2.h5ad_manifest import MemBudget

    cfg = _preset()
    pred = _pred_h5ad(tmp_path)
    # a matching real h5ad (same schema) for the real reference
    rng = np.random.default_rng(11)
    labels = ["non-targeting"] * 50 + sum(([g] * 40 for g in ["A", "B", "C"]), [])
    Xr = np.log1p(rng.poisson(3, size=(len(labels), 12)).astype(np.float32))
    real_p = str(tmp_path / "real.h5ad")
    ad.AnnData(X=Xr, obs={"target": labels},
               var={"gene": [f"g{i}" for i in range(12)]}).write_h5ad(real_p)

    cache = str(tmp_path / "ref")
    mb = MemBudget(host_bytes=1 << 34, gpu_bytes=1 << 34)
    comparator = resolved_comparator(
        cfg, pred_input_type="lognorm", real_input_type="lognorm")
    partition_inmem.build_reference_streaming(
        real_p, config=cfg, cache_dir=cache, control="non-targeting", mem_budget=mb,
        input_type="lognorm", comparator=comparator)
    partition_inmem.build_pred_control_reference(
        pred, config=cfg, cache_dir=cache, control="non-targeting", input_type="lognorm",
        comparator=comparator)

    piece = ad.read_h5ad(pred)
    piece = piece[piece.obs["target"].isin(["A", "B"]).values].copy()  # NO controls
    df = partition_inmem.score_piece(
        piece, cache, config=cfg, piece_id="ab", comparator=comparator)
    assert df.height > 0
    assert set(df["perturbation"].unique()) == {"A", "B"}
    # both an expression (delta) metric and a DE metric are emitted under pred mode
    metrics = set(df["metric"].unique())
    assert any(m for m in metrics if "mae" in m)          # a delta/expression metric
    assert any(m.startswith("de_") for m in metrics)      # a DE metric
