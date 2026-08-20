# Filtering DEGs by effect size — the `min_abs_log2fc` floor

A focused, **runnable** how-to for the post-FDR minimum-|log2 fold-change| floor (added in
#109), driven from the **`cell-eval-0.7.6`**
preset. It shows how to score at **two cutoffs (0.25 and 0.5) in a single run** while computing the
(expensive) differential expression **once**.

This builds on the main [cell_eval2 tutorial](../tutorial.md) — see there for install, the example
data, and the core `compute_metrics` API. Run Python with the project venv (`.venv/bin/python` or
`uv run --no-sync python`), from the repo root.

---

## 1. What the floor does

By default a gene counts as a differentially-expressed gene (DEG) for a perturbation when its
BH-adjusted p-value clears the significance gate, `p_adj < p_adj_threshold` (default `0.05`). The
new knob adds a **conjunctive effect-size condition**:

```
gene is a DEG  ⇔  p_adj < p_adj_threshold  AND  |log2_fold_change| >= min_abs_log2fc
```

- **Parameter:** `DEParams.min_abs_log2fc` (YAML key `de.min_abs_log2fc`). Default **`0.0` = off**
  (a true no-op — every preset stays bit-identical until you set it).
- **Post-FDR.** `p_adj` (the BH adjustment) is already final; the floor is a pure *membership*
  filter layered on top. It never re-pools or re-runs FDR — so it does **not** change the raw DE
  table, only which of its genes count as significant.
- **Applied symmetrically** to the real and predicted DE, at one chokepoint, so **every**
  DE-membership metric honors it: nsig counts, overlap/precision (+ top-k), recall, both direction
  match variants, LFC-spearman, PR/ROC-AUC, and the chance-corrected MCC family.
- **Universe-preserving.** Below-floor genes are reclassified as *tested-but-not-significant*
  (their `p_adj` is masked to `1.0`); they stay in the tested-gene universe, so the denominators of
  the chance-corrected metrics are unchanged.
- **Inclusive:** a gene with `|log2fc|` exactly equal to the floor is **kept** (the mask is a
  strict `<`, parallel to the strict `p_adj < threshold` gate).
- Validated: `min_abs_log2fc` must be a **finite float `>= 0`** (NaN/inf/negative raise).

The floor is on the **log2** scale: `0.25 ≈ 1.19×`, `0.5 ≈ 1.41×`, `1.0 = 2×` fold change.

---

## 2. Setup: the example data + a prediction

We reuse the main tutorial's committed 5 MB reference (`docs/data/H1-VCC-2025-training.h5ad`, 600
cells × 1000 genes of raw counts; perturbation label in `obs["target_gene"]`, control
`non-targeting`) and its toy `pred_good` prediction (a teaching prop, not a real model):

```python
import anndata as ad
import numpy as np
import scipy.sparse as sp

real = ad.read_h5ad("docs/data/H1-VCC-2025-training.h5ad")

# pred_good: real per-perturbation mean profile + mild multiplicative noise (from the main tutorial)
X = real.X.toarray()
labels = real.obs["target_gene"].to_numpy()
rng = np.random.default_rng(1)
Xg = np.zeros_like(X)
for g in np.unique(labels):
    m = labels == g
    mean = X[m].mean(axis=0, keepdims=True)
    Xg[m] = np.rint(mean * rng.lognormal(0.0, 0.35, size=(m.sum(), X.shape[1]))).clip(min=0)
pred = ad.AnnData(sp.csr_matrix(Xg.astype(np.float32)), obs=real.obs.copy(), var=real.var.copy())
```

---

## 3. Start from the `cell-eval-0.7.6` preset and add the floor

`cell-eval-0.7.6` reproduces a *modern* upstream (cell-eval 0.7.6 + pdex 0.2.6 — the upstream
forward-eval recipe): predicted control, geometric mean, keep-NaN LFCs, 5-CPM gene filter, per-pert
FDR, and the v1 metric names. The floor is orthogonal to all of that — you just set
`de.min_abs_log2fc` on top of the preset.

Build a floored config from the preset with `dataclasses.replace` (it re-runs `DEParams`
validation, and doesn't mutate the shared `base`):

```python
from dataclasses import replace
from cell_eval2 import EvalConfig

base = EvalConfig.from_preset("cell-eval-0.7.6")     # de.min_abs_log2fc defaults to 0.0 (off)
cfg_025 = replace(base, de=replace(base.de, min_abs_log2fc=0.25))
cfg_050 = replace(base, de=replace(base.de, min_abs_log2fc=0.50))
```

> The preset's DE backend is `auto` (→ gpudge on GPU). On a CPU box pin it: add
> `backend="pdex"` to the inner `replace(base.de, ...)`. pdex is **not** part of the base
> install — `pip install pdex` (dev environments get it transitively via `cell-eval`).

---

## 4. One run, two cutoffs — computing DE **once**

The floor is applied *after* the DE table is computed, so the raw DE table is **identical** for
every cutoff. Point `cache_real` / `cache_pred` at directories and the DE table is computed on the
first cutoff and **reused** for the rest — only the cheap floor-mask + rank + metrics recompute per
cutoff. (Under the hood the cache stores both `de_wilcoxon_table.parquet` — the floor-independent
DE table — and `de_wilcoxon_rank.parquet`, which is floor-aware.)

```python
import os, tempfile, time
from cell_eval2 import compute_metrics, aggregate_metrics

results = {}
with tempfile.TemporaryDirectory() as d:              # in real use: a PERSISTENT dir you keep
    cache_real, cache_pred = os.path.join(d, "real"), os.path.join(d, "pred")
    for floor in (0.25, 0.50):
        cfg = replace(base, de=replace(base.de, backend="pdex", min_abs_log2fc=floor),
                      cache_real=cache_real, cache_pred=cache_pred)
        t = time.time()
        df = compute_metrics(pred, real, config=cfg, metrics="de",
                             pert_col="target_gene", control="non-targeting")
        results[floor] = aggregate_metrics(df)
        print(f"floor={floor}: {time.time() - t:.2f}s")
# floor=0.25: ~7.0s   (cold — computes + caches the DE table)
# floor=0.5:  ~0.14s  (warm — DE table reused; ~50x faster)
```

The first cutoff pays the DE cost once; the second is near-instant. On a real (large) dataset with a
GPU backend this is the difference between running gpudge once vs. once per cutoff.

*(Prefer to be explicit? You can also precompute the DE tables yourself and pass them via
`compute_metrics(..., de_real=<frame-or-path>, de_pred=<frame-or-path>)` — the caching route above
is just the ergonomic version of the same "compute once, reuse" idea.)*

---

## 5. Reading the results — the floor sharpens the DEG set

Raising the floor removes small-effect significant genes, so the DEG counts shrink and the
downstream DE metrics move. On this toy prediction (illustrative — 5 perturbations, a teaching prop):

```python
import polars as pl

def pick(agg, metric):
    return agg.filter(pl.col("metric") == metric)["mean"].item()

for floor in (0.25, 0.50):
    agg = results[floor]
    print(f"floor={floor:>4}: nsig_real={pick(agg,'de_nsig_counts_real'):.2f}  "
          f"overlap_at_N={pick(agg,'overlap_at_N'):.3f}  "
          f"precision_at_N={pick(agg,'precision_at_N'):.3f}")
```

| `min_abs_log2fc` | mean DEGs / pert (`de_nsig_counts_real`) | `overlap_at_N` | `precision_at_N` |
|---|---|---|---|
| `0.0` (preset default, no floor) | 9.40 | 0.267 | 0.081 |
| `0.25` | 5.40 | 0.425 | 0.109 |
| `0.50` | 1.00 | 0.267 | 0.300 |

The headline is the **monotonic drop in the DEG count** (9.4 → 5.4 → 1.0 genes/perturbation) as the
floor tightens — exactly the "keep only genes with a real effect size" behavior. Precision rises as
the surviving DEGs are the higher-confidence, larger-effect ones. (Recall the toy scale: with only
5 perturbations these downstream numbers are noisy; on real data they move smoothly.)

Reduce a run to one number per metric with `aggregate_metrics` (a NaN-skipping mean over
perturbations), as above. `de_nsig_counts_real` / `_pred` are diagnostic counts (`best_value="none"`)
— they are exactly the DEG-set sizes the floor controls.

---

## 6. The command line

Set the floor directly on the CLI with the generic **`--set KEY.PATH=VALUE`** override — no
per-cutoff YAML needed. Write the `cell-eval-0.7.6` preset to a YAML once (there's no `--preset`
flag; `--version` only selects v1/v2), then `--set de.min_abs_log2fc=…` per run:

```bash
# one-time: materialize the preset as a base config
python -c "from cell_eval2 import EvalConfig; EvalConfig.from_preset('cell-eval-0.7.6').to_yaml('cell-eval-0.7.6.yaml')"

# score at the 0.25 floor on CPU (swap 0.25 -> 0.5 for the second cutoff):
cell-eval2 run -ap pred.h5ad -ar real.h5ad --config cell-eval-0.7.6.yaml \
    --set de.min_abs_log2fc=0.25 --set de.backend=pdex \
    --pert-col target_gene --control non-targeting \
    --cache-real cache/real --cache-pred cache/pred -o out_lfc025/
# -> out_lfc025/results.csv ; reuse the SAME --cache-* dirs for the 0.5 run to compute DE once.
```

`--set` reaches **any** config field by dotted path (nested `de.*` / `discrimination.*` /
`filter.*` and top-level fields alike), repeatable, applied on top of `--config` and the explicit
flags. `VALUE` is parsed as YAML (`0.25`→float, `null`→None, `true`→bool), and it goes through the
same validation as the config — a bad `--set de.min_abs_log2fc=-1` is rejected with a clear error.

Prefer a fully-committed config file instead? Bake the floor into the YAML and pass just `--config`:

```python
cfg = replace(EvalConfig.from_preset("cell-eval-0.7.6"),
              metrics="de", pert_col="target_gene", control="non-targeting",
              de=replace(base.de, backend="pdex", min_abs_log2fc=0.25))
cfg.to_yaml("cell-eval-0.7.6-lfc025.yaml")   # de.min_abs_log2fc: 0.25 lands in the `de:` block
# cell-eval2 run -ap pred.h5ad -ar real.h5ad --config cell-eval-0.7.6-lfc025.yaml -o out_lfc025/
```

---

## 7. Notes & edge cases

- **Post-FDR, not pre-FDR.** The floor never re-pools p-values — BH runs once at DE-compute time on
  the full gene set; the floor only re-labels significance. Two cutoffs share the same `p_adj`.
- **Non-finite / null LFCs are never floored.** `|NaN|`, `|inf|`, and null `log2_fold_change` all
  fail the strict `<`, so the floor leaves them untouched; their significance stays governed by
  `nan_lfc_policy` (which, note, masks only NaN — `inf`/`null` are warned about, not masked).
  `inf` (a maximal effect) is correctly retained.
- **Works with any preset.** `cell-eval-0.7.6` uses `nan_lfc_policy="keep"`; v2 uses `"mask"`. The
  floor composes with either — set `de.min_abs_log2fc` and it applies on top.
- **Cache invalidation is floor-aware.** Changing the floor produces a different rank-cache key (via
  the `min_abs_log2fc` cache-params entry), so you never get a stale floored result — while the
  floor-independent DE table is still reused.

## Links

- [Main tutorial](../tutorial.md) · [metric catalog](../tutorial.md#appendix-a-metric-catalog)
- Design + rationale: PR #109
