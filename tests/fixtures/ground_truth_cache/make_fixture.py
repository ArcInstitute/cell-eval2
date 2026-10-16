#!/usr/bin/env python
"""Regenerate the frozen ground-truth-cache fixture archives (the cache's Phase 0 contract kit).

Writes five small cellstream cell-layout archives from one deterministic synthetic
dataset (seeded numpy) and records their identities in ``fixture_manifest.json``:

  real.shad             ground truth, payload_checksum=True  (the cache-eligible archive)
  real_nochecksum.shad  identical cells, payload_checksum=False (must be REFUSED when caching)
  real_onevalue.shad    one count changed, payload_checksum=True (must yield a different fingerprint)
  pred_a.shad           "good" prediction (resampled from the real means), payload_checksum=False
  pred_b.shad           "null" prediction (control means for every perturbation), payload_checksum=True

The dataset is tiny on purpose (320 cells x 150 genes, 6 perturbations + control) so the
archives stay small and evaluate in seconds. Perturbation labels equal gene names (TG01..TG06)
so CellEval2's target-gene resolution works without a target_gene_map.

Cellstream writes are byte-deterministic except for the manifest's ``created_at`` (second
resolution), so a regeneration reproduces every ``payload_sha256`` but not necessarily the
file SHA-256. The frozen fixture is the committed bytes; ``fixture_manifest.json`` records both.

Requires: cellstream>=0.10.0, anndata, numpy, scipy, pandas.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata as md
import json
import platform
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import gtc_reference as gtc  # noqa: E402

SEED = 20260916
N_GENES = 150
N_CTRL = 80
N_PER_PERT = 40
PERTS = ["TG01", "TG02", "TG03", "TG04", "TG05", "TG06"]
CONTROL = "non-targeting"
PERT_COL = "perturbation"
CONTEXT = "ctxA"
DATASET = "gtc_fixture"


def _gene_names() -> list[str]:
    # First six genes are the perturbation targets; the rest are anonymous.
    return PERTS + [f"GENE_{i:04d}" for i in range(1, N_GENES - len(PERTS) + 1)]


def _base_means(rng: np.random.Generator) -> np.ndarray:
    # Log-normal gene means, library size ~ 3k counts / cell.
    mu = rng.lognormal(mean=0.0, sigma=1.2, size=N_GENES)
    return mu / mu.sum() * 3000.0


def _effects(rng: np.random.Generator, genes: list[str]) -> dict[str, np.ndarray]:
    """Multiplicative per-perturbation effect vectors: own target knocked down x0.1, plus 8
    random DE genes at x0.5 or x2.0; everything else 1.0."""
    eff = {}
    for p in PERTS:
        f = np.ones(N_GENES)
        f[genes.index(p)] = 0.1
        idx = rng.choice(np.arange(len(PERTS), N_GENES), size=8, replace=False)
        f[idx] = rng.choice([0.5, 2.0], size=8)
        eff[p] = f
    return eff


def _sample(rng: np.random.Generator, means: np.ndarray, n: int) -> np.ndarray:
    # Negative binomial via gamma-Poisson (dispersion 0.3) -> integer counts.
    shape = 1.0 / 0.3
    lam = rng.gamma(shape, means / shape, size=(n, means.size))
    return rng.poisson(lam)


def _adata(rng: np.random.Generator, means: np.ndarray, effects: dict[str, np.ndarray],
           genes: list[str], *, null_model: bool = False) -> ad.AnnData:
    blocks, labels = [], []
    blocks.append(_sample(rng, means, N_CTRL))
    labels += [CONTROL] * N_CTRL
    for p in PERTS:
        m = means if null_model else means * effects[p]
        blocks.append(_sample(rng, m, N_PER_PERT))
        labels += [p] * N_PER_PERT
    X = np.vstack(blocks).astype(np.float32)
    # Shuffle rows so the archive writer's group ordering is exercised.
    perm = rng.permutation(X.shape[0])
    X, labels = X[perm], [labels[i] for i in perm]
    obs = pd.DataFrame({
        PERT_COL: labels,
        "control_value": CONTROL,
        "context": CONTEXT,
        "dataset": DATASET,
    }, index=[f"cell_{i:04d}" for i in range(X.shape[0])])
    var = pd.DataFrame(index=pd.Index(genes, name=None))
    return ad.AnnData(X=sp.csr_matrix(X), obs=obs, var=var)


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _describe(path: Path, *, role: str, payload_checksum: bool) -> dict:
    from cellstream.cell import open_cell
    s = open_cell(str(path))
    try:
        m = s.manifest
        var_names = [str(v) for v in s.var.index.tolist()]
        pert_labels = [str(v) for v in s.obs[PERT_COL].tolist()]
        rec = {
            "file": path.name,
            "role": role,
            "size_bytes": path.stat().st_size,
            "file_sha256": _sha256_file(path),
            "written_with_payload_checksum": payload_checksum,
            "archive_manifest": {
                "schema_version": m.get("schema_version"),
                "layout": m.get("layout"),
                "codec": m.get("codec"),
                "group_by": m.get("group_by"),
                "reference": m.get("reference"),
                "n_obs_total": m.get("n_obs_total"),
                "n_vars": m.get("n_vars"),
                "total_nnz": m.get("total_nnz"),
                "value_dtype_on_disk": m.get("value_dtype_on_disk"),
                "payload_sha256": m.get("payload_sha256"),
                "writer_fingerprint": m.get("writer_fingerprint"),
            },
            "pert_col": PERT_COL,
            "var_names": var_names,
            "pert_labels_storage_order": pert_labels,
            "feature_axis_sha256": gtc.sequence_sha256(var_names),
            "perturbation_axis_sha256": gtc.sequence_sha256(pert_labels),
        }
        sha = m.get("payload_sha256")
        if isinstance(sha, str) and sha:
            doc = gtc.ground_truth_fingerprint_document(
                payload_sha256=sha, var_names=var_names, pert_labels=pert_labels, pert_col=PERT_COL)
            rec["ground_truth_fingerprint"] = doc
            rec["ground_truth_fingerprint_sha256"] = gtc.document_sha256(doc)
            rec["cache_eligible"] = True
        else:
            rec["ground_truth_fingerprint"] = None
            rec["ground_truth_fingerprint_sha256"] = None
            rec["cache_eligible"] = False
            rec["reason"] = "manifest.payload_sha256 is null (written with payload_checksum=False)"
        return rec
    finally:
        s.close()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--outdir", default=str(HERE / "archives"))
    ap.add_argument("--manifest", default=str(HERE / "fixture_manifest.json"))
    args = ap.parse_args()
    from cellstream.cell import write_cell_archive

    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    genes = _gene_names()
    means = _base_means(rng)
    effects = _effects(rng, genes)

    real = _adata(np.random.default_rng(SEED + 1), means, effects, genes)
    pred_a = _adata(np.random.default_rng(SEED + 2), means, effects, genes)
    pred_b = _adata(np.random.default_rng(SEED + 3), means, effects, genes, null_model=True)
    # One expression value changed: first non-zero entry of the first perturbed cell (+1 count).
    real_onevalue = real.copy()
    X = real_onevalue.X.tolil()
    row = int(np.flatnonzero(real_onevalue.obs[PERT_COL].to_numpy() != CONTROL)[0])
    col = int(X.rows[row][0])
    X[row, col] = X[row, col] + 1.0
    real_onevalue.X = X.tocsr()

    plan = [
        ("real.shad", real, True, "ground truth"),
        ("real_nochecksum.shad", real, False,
         "ground truth without payload checksum (must be refused when caching)"),
        ("real_onevalue.shad", real_onevalue, True,
         f"ground truth with one count changed (input row {row}, col {col})"),
        ("pred_a.shad", pred_a, False, "prediction A: resampled from the real means"),
        ("pred_b.shad", pred_b, True, "prediction B: null model (control means for every perturbation)"),
    ]
    records = []
    for name, adata, checksum, role in plan:
        path = out / name
        write_cell_archive(adata, str(path), group_by=PERT_COL, reference=CONTROL,
                           payload_checksum=checksum, overwrite=True, tmp_dir=str(out))
        lock = path.with_name(path.name + ".lock")
        if lock.exists():  # cellstream's advisory write lock; not part of the fixture
            lock.unlink()
        records.append(_describe(path, role=role, payload_checksum=checksum))
        print(f"wrote {path.name}: {records[-1]['size_bytes']} bytes "
              f"payload_sha256={records[-1]['archive_manifest']['payload_sha256']}")

    manifest = {
        "fixture": "ground_truth_cache",
        "spec": "GROUND-TRUTH-CACHE.md Phase 0 frozen cross-repository fixture",
        "generator": "tests/fixtures/ground_truth_cache/make_fixture.py",
        "seed": SEED,
        "design": {
            "n_genes": N_GENES, "n_control_cells": N_CTRL, "n_cells_per_perturbation": N_PER_PERT,
            "perturbations": PERTS, "control": CONTROL, "pert_col": PERT_COL,
            "context": CONTEXT, "dataset": DATASET,
            "values": "raw integer counts (gamma-Poisson, dispersion 0.3)",
        },
        "writer": {
            "cellstream": md.version("cellstream"),
            "python": platform.python_version(),
            "numpy": np.__version__,
            "note": ("file bytes are deterministic except manifest.created_at (second resolution); "
                     "payload_sha256 is deterministic"),
        },
        "archives": records,
    }
    Path(args.manifest).write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {args.manifest}")


if __name__ == "__main__":
    main()
