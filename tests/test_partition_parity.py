"""Linchpin acceptance test: partitioned in-memory scoring == whole-prediction scoring.

Scores a small dataset whole (``compute_metrics`` v2) vs partitioned (``build_reference`` +
N disjoint ``score_piece`` + ``aggregate_partials``) and asserts the tidy frames match for
every metric. BOTH sides run the gpudge DE backend (partitioned scoring requires it), so this
test needs a real CUDA GPU and SKIPS on a CPU-only node.

A CCL-scale parity run (a 6-way split of a real archive) is done manually on a GPU node via
slurm (>50 GB RAM rule), NOT in CI — see internal:tools/partition/README.md. This CI-sized test uses
small synthetic data.
"""

from dataclasses import replace

import anndata as ad
import numpy as np
import pandas as pd
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


def _cfg():
    v2 = EvalConfig.v2()
    # Fixed AUC floor on BOTH sides so pr_auc/roc_auc are bit-identical between whole-prediction
    # and partitioned scoring (PR #83 review). Device left at auto -> cuda so both sides use gpudge.
    # Moment-consuming expression metrics are unavailable on the partitioned driver (#198).
    return replace(v2, pert_col="target_gene", metrics=full_minus_moments(),
                   de=replace(v2.de, auc_pval_floor="replace_zero", auc_pval_floor_value=1e-10))


def _data():
    perts = ["non-targeting"] * 80 + sum(([g] * 50 for g in ["A", "B", "C", "D", "E", "F"]), [])
    genes = ["A", "B", "C", "D", "E", "F"] + [f"g{i}" for i in range(6, 16)]

    def mk(seed):
        r = np.random.default_rng(seed)
        return ad.AnnData(
            X=r.poisson(3, size=(len(perts), 16)).astype(np.float32),
            obs={"target_gene": perts},
            var=pd.DataFrame({"gene": genes}, index=genes),
        )

    return mk(0), mk(100)  # real, pred


def _tidy_map(df):
    return {(r["perturbation"], r["metric"]): r["value"] for r in df.iter_rows(named=True)}


def test_partitioned_equals_whole(tmp_path):
    cfg = _cfg()
    comparator = resolved_comparator(cfg)
    real, pred = _data()
    whole = compute_metrics(pred, real, config=cfg)

    partition_inmem.build_reference(real, config=cfg, cache_dir=str(tmp_path / "ref"),
                                    control_format="h5ad", comparator=comparator)
    for group in (["A", "B", "C"], ["D", "E", "F"]):  # two disjoint pieces, control-free
        mask = pred.obs["target_gene"].isin(group).values
        piece = pred[mask].copy()
        partition_inmem.score_piece(piece, str(tmp_path / "ref"), config=cfg,
                                    piece_id="_".join(group),
                                    partial_out=str(tmp_path / "partials"),
                                    comparator=comparator)
    universe = ["A", "B", "C", "D", "E", "F"]
    full, _ = partition.aggregate_partials(str(tmp_path / "partials"),
                                           reference_universe=universe,
                                           reduce_nsig_spearman=True)

    w, p = _tidy_map(whole), _tidy_map(full)
    assert set(w) == set(p), f"key mismatch: {set(w) ^ set(p)}"
    for k, wv in w.items():
        pv = p[k]
        if wv != wv:  # NaN
            assert pv != pv, f"{k}: whole NaN vs {pv}"
        else:
            assert abs(wv - pv) <= 1e-9 + 1e-6 * abs(wv), f"{k}: whole {wv} vs partitioned {pv}"
