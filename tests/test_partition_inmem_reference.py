import json
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


pytestmark_gpu = pytest.mark.skipif(_no_gpu(), reason="needs a CUDA GPU (gpudge DE)")


def _toy_real(n_per=40, n_genes=12, seed=0):
    rng = np.random.default_rng(seed)
    perts = ["non-targeting"] * n_per + sum(([g] * n_per for g in ["A", "B", "C"]), [])
    X = rng.poisson(3, size=(len(perts), n_genes)).astype(np.float32)
    obs = {"target_gene": perts}
    return ad.AnnData(X=X, obs=obs, var={"gene": [f"g{i}" for i in range(n_genes)]})


def test_config_gate_rejects_v1(tmp_path):
    cfg = EvalConfig.v1()
    with pytest.raises(NotImplementedError):
        partition_inmem.build_reference(
            _toy_real(), config=cfg, cache_dir=str(tmp_path),
            comparator=resolved_comparator(cfg))


@pytestmark_gpu
def test_build_reference_writes_bundle(tmp_path):
    cfg = replace(EvalConfig.v2(), pert_col="target_gene")  # replace() keeps nested dataclasses
    manifest = partition_inmem.build_reference(
        _toy_real(), config=cfg, cache_dir=str(tmp_path), control_format="h5ad",
        comparator=resolved_comparator(cfg))
    assert set(manifest["perturbation_universe"]) == {"A", "B", "C"}
    assert (tmp_path / "reference.json").exists()
    assert (tmp_path / "real_de.parquet").exists()
    assert (tmp_path / "real_control.h5ad").exists()
    assert any(p.name.startswith("real_pseudobulk_") for p in tmp_path.iterdir())
    on_disk = json.loads((tmp_path / "reference.json").read_text())
    assert on_disk["control_format"] == "h5ad"
