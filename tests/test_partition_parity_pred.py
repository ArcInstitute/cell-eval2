"""Linchpin acceptance test (SP2): partitioned pred-control scoring == whole-prediction scoring
under the cell-eval-0.7.6 preset (v1 / control_source='pred' / auc_pval_floor='clip'). Both
sides run gpudge DE, so this needs a CUDA GPU and SKIPS on a CPU node."""
from dataclasses import replace

import anndata as ad
import numpy as np
import pytest

from cell_eval2 import compute_metrics, partition, partition_inmem
from cell_eval2.config import EvalConfig
from _helpers import resolved_comparator


def _no_gpu():
    try:
        import cupy
        return cupy.cuda.runtime.getDeviceCount() == 0
    except Exception:
        return True


pytestmark = pytest.mark.skipif(_no_gpu(), reason="needs a CUDA GPU (gpudge DE both sides)")


def _cfg():
    # cell-eval-0.7.6 preset, pert_col aligned to the synthetic obs column.
    # v1: the profile string lets resolve_metrics filter v2-native metrics silently; an explicit list would raise (#198).
    # exclude_target_gene=False deliberately (#275): this fixture's labels are A-F, and the panel
    # they are matched against is var_names "0".."15" -- NOT "g0".."g15". `var={"gene": [...]}`
    # puts those labels in a var COLUMN and leaves AnnData to mint a default RangeIndex, so the
    # measured gene names are the stringified integers. Either way nothing resolves and #248's
    # gate refuses to score at all. This test only needs
    # the guard not to fire -- both sides of the parity comparison share this config, so the
    # equality it asserts is unaffected. `replace` on the preset's OWN DiscriminationParams,
    # never a fresh one: the preset carries distance='l1', rank_denominator='n' and
    # tie_policy='position', which a bare DiscriminationParams would silently reset to v2.
    cfg = replace(EvalConfig.from_preset("cell-eval-0.7.6"), pert_col="target_gene")
    return replace(cfg, discrimination=replace(cfg.discrimination, exclude_target_gene=False))


def _data(tmp_path):
    perts = ["non-targeting"] * 80 + sum(([g] * 50 for g in ["A", "B", "C", "D", "E", "F"]), [])

    def mk(seed):
        r = np.random.default_rng(seed)
        X = np.log1p(r.poisson(3, size=(len(perts), 16)).astype(np.float32))  # lognorm (preset)
        return ad.AnnData(X=X, obs={"target_gene": perts},
                          var={"gene": [f"g{i}" for i in range(16)]})

    real, pred = mk(0), mk(100)
    pred_p = str(tmp_path / "pred.h5ad")
    pred.write_h5ad(pred_p)                     # build_pred_control_reference needs an h5ad path
    return real, pred, pred_p


def _tidy_map(df):
    return {(r["perturbation"], r["metric"]): r["value"] for r in df.iter_rows(named=True)}


def test_partitioned_pred_equals_whole(tmp_path):
    cfg = _cfg()
    comparator = resolved_comparator(cfg)
    real, pred, pred_p = _data(tmp_path)
    whole = compute_metrics(pred, real, config=cfg)

    cache = str(tmp_path / "ref")
    partition_inmem.build_reference(
        real, config=cfg, cache_dir=cache, control_format="h5ad", comparator=comparator)
    partition_inmem.build_pred_control_reference(
        pred_p, config=cfg, cache_dir=cache, control="non-targeting", comparator=comparator)
    for group in (["A", "B", "C"], ["D", "E", "F"]):
        mask = pred.obs["target_gene"].isin(group).values
        piece = pred[mask].copy()
        partition_inmem.score_piece(piece, cache, config=cfg, piece_id="_".join(group),
                                    partial_out=str(tmp_path / "partials"),
                                    comparator=comparator)
    universe = ["A", "B", "C", "D", "E", "F"]
    # v1 nsig names (preset is version=v1)
    full, _ = partition.aggregate_partials(
        str(tmp_path / "partials"), reference_universe=universe, reduce_nsig_spearman=True,
        nsig_spearman_metric="de_spearman_sig",
        nsig_real_metric="de_nsig_counts_real", nsig_pred_metric="de_nsig_counts_pred")

    w, p = _tidy_map(whole), _tidy_map(full)
    assert set(w) == set(p), f"key mismatch: {set(w) ^ set(p)}"
    for k, wv in w.items():
        pv = p[k]
        if wv != wv:  # NaN
            assert pv != pv, f"{k}: whole NaN vs {pv}"
        else:
            assert abs(wv - pv) <= 1e-9 + 1e-6 * abs(wv), f"{k}: whole {wv} vs partitioned {pv}"
