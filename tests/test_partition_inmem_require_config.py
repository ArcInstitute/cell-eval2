"""CPU-only unit tests for the relaxed partition config gate (SP2). _resolve_backend is
monkeypatched to 'gpudge' so these run on the CPU dev node (the real gate needs a CUDA
device to resolve backend='auto' to gpudge)."""
from dataclasses import replace

import pytest

from cell_eval2 import partition_inmem
from cell_eval2.config import EvalConfig


@pytest.fixture(autouse=True)
def _force_gpudge(monkeypatch):
    monkeypatch.setattr(partition_inmem, "_resolve_backend", lambda b: "gpudge")


def _preset():
    # the shipped cell-eval-0.7.6 preset: v1 + pred control + clip floor + per_pert fdr
    return EvalConfig.from_preset("cell-eval-0.7.6")


def test_accepts_v1_pred_clip_preset():
    cfg = partition_inmem._require_partition_config(_preset())
    assert cfg.version == "v1"
    assert cfg.control_source == "pred"
    assert cfg.de.auc_pval_floor == "clip"          # clip preserved (partition-invariant)
    assert cfg.de.auc_pval_floor_value == 1e-10


def test_accepts_v2_real_still_normalizes_min_nonzero():
    v2 = EvalConfig.v2()                              # default auc_pval_floor = min_nonzero
    assert v2.de.auc_pval_floor == "min_nonzero"
    cfg = partition_inmem._require_partition_config(v2)
    assert cfg.control_source == "real"
    assert cfg.de.auc_pval_floor == "replace_zero"   # normalized (NOT partition-invariant)
    assert cfg.de.auc_pval_floor_value == 1e-10


def test_preserves_replace_zero():
    v1 = EvalConfig.v1()                              # v1 default = replace_zero, but fdr=global
    v1 = replace(v1, de=replace(v1.de, fdr_scope="per_pert"))
    cfg = partition_inmem._require_partition_config(v1)
    assert cfg.de.auc_pval_floor == "replace_zero"


def test_still_rejects_global_fdr():
    v1 = EvalConfig.v1()                              # v1 default fdr_scope = global
    with pytest.raises(NotImplementedError, match="fdr_scope"):
        partition_inmem._require_partition_config(v1)
