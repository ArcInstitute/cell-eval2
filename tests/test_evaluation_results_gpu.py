"""Exercise #412's document contract on real GPU scoring paths (design §9 and §12)."""
from dataclasses import replace
import json
from pathlib import Path

import anndata as ad
import jsonschema
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from _helpers import full_minus_moments
from cell_eval2.cellstream import score_cellstream
from cell_eval2.config import EvalConfig
from cell_eval2.evaluation_results import FILENAME, build_document, load_document
from cell_eval2.h5ad_manifest import MemBudget
from cell_eval2.run import _compute_metrics_with_facts, metric_output_names
from cell_eval2.settings import facts_checker, settings_to_config


PERTS = ["control", "GENE_A", "GENE_B", "GENE_C"]
_BIG = MemBudget(host_bytes=10**12, gpu_bytes=10**12)


def _skip_if_no_gpu():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("score_cellstream requires a CUDA device (gpudge external-reference DE)")
    pytest.importorskip("gpudge")


def _write_cell(tmp_path, name, seed, *, n_per=30, n_genes=50):
    """Write a minimal grouped, reference-first archive for the #412 GPU gate."""
    pytest.importorskip("cellstream")
    from cellstream.cell import write_cell_archive

    rng = np.random.default_rng(seed)
    n = n_per * len(PERTS)
    X = sp.csr_matrix(rng.poisson(0.5, size=(n, n_genes)).astype(np.float32))
    obs = pd.DataFrame(
        {"perturbation": np.repeat(PERTS, n_per), "context": np.repeat("ctxA", n),
         "control_value": np.repeat("control", n), "dataset": np.repeat("dsX", n)},
        index=[f"{name}_c{i}" for i in range(n)],
    )
    var = pd.DataFrame(index=PERTS[1:] + [f"g{j}" for j in range(len(PERTS) - 1, n_genes)])
    adata = ad.AnnData(X=X, obs=obs, var=var)
    shad = tmp_path / f"{name}.shad"
    ref = (obs["perturbation"] == "control").to_numpy()
    write_cell_archive(adata, shad, group_by="perturbation", reference=ref,
                       codec="zstd", overwrite=True)
    return shad


def _pinned(cfg):
    """Pin the AUC floor and per-perturbation FDR for both #412 scoring paths."""
    return replace(cfg, de=replace(cfg.de, auc_pval_floor="replace_zero",
                                   auc_pval_floor_value=1e-10, fdr_scope="per_pert"))


def _config():
    return _pinned(EvalConfig(device="cuda", pert_col="perturbation", control="control",
                              input_type="counts", target_sum=1e6,
                              metrics=full_minus_moments()))


def _validate(doc):
    """Check the saved schema independently of the #412 runtime document validator."""
    schema = json.loads((Path(__file__).parent /
                         "fixtures/arcbench/evaluation_results_v1.schema.json").read_text())
    jsonschema.Draft202012Validator(schema).validate(doc)


def test_score_cellstream_writes_a_valid_document_gpu(tmp_path):
    _skip_if_no_gpu()
    pred = _write_cell(tmp_path, "pred", 1)
    real = _write_cell(tmp_path, "real", 2)
    o = tmp_path / "streaming"
    score_cellstream(pred, real, config=_config(), mem_budget=_BIG, outdir=o, write_de=True)
    doc = load_document(o / FILENAME)
    _validate(doc)


def test_both_paths_agree_on_keys_statistics_and_axes_gpu(tmp_path):
    _skip_if_no_gpu()
    pred = _write_cell(tmp_path, "pred", 1)
    real = _write_cell(tmp_path, "real", 2)
    cfg = _config()
    o = tmp_path / "streaming"
    score_cellstream(pred, real, config=cfg, mem_budget=_BIG, outdir=o)
    a = load_document(o / FILENAME)
    df, facts = _compute_metrics_with_facts(pred, real, config=cfg)
    b = build_document(df, facts, metrics=metric_output_names(cfg))

    assert sorted(a["metrics"]) == sorted(b["metrics"])
    for k in a["metrics"]:
        assert set(a["metrics"][k]["statistics"]) == set(b["metrics"][k]["statistics"])
        assert ("axis" in a["metrics"][k]) == ("axis" in b["metrics"][k])
    assert a["axes"] == b["axes"]
    assert a["suite"]["settings"]["scoring_path"] == "streaming"
    assert b["suite"]["settings"]["scoring_path"] == "in_memory"


def test_a_streaming_document_replays_through_score_cellstream_gpu(tmp_path):
    _skip_if_no_gpu()
    pred = _write_cell(tmp_path, "pred", 1)
    real = _write_cell(tmp_path, "real", 2)
    first = tmp_path / "first"
    replay = tmp_path / "replay"
    score_cellstream(pred, real, config=_config(), mem_budget=_BIG, outdir=first)
    a = load_document(first / FILENAME)
    score_cellstream(pred, real, config=settings_to_config(a), check_facts=facts_checker(a),
                     mem_budget=_BIG, outdir=replay)
    b = load_document(replay / FILENAME)
    assert b["suite"]["settings"] == a["suite"]["settings"]
    assert sorted(b["metrics"]) == sorted(a["metrics"])
