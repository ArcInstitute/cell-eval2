# cell_eval2 tutorial

A guided, mostly-runnable tour of how to use **cell_eval2** to evaluate perturbation-prediction
models at single-cell resolution. It starts from the simplest possible call and builds up to
GPU acceleration and large-scale streaming. Every section says which public API it exercises.

The runnable examples use a small, committed real dataset (`docs/data/H1-VCC-2025-training.h5ad`,
~5 MB) — a subset of the **public** [Virtual Cell Challenge 2025](https://huggingface.co/datasets/arcinstitute/VCC_train)
training data. Sections that can only run on a GPU or on very large data (CPU vs GPU,
large-scale streaming) are shown as commands plus **reported headline numbers** from Arc's
internal benchmark runs.

> Throughout, run Python with the project's virtualenv: `.venv/bin/python` (or
> `uv run --no-sync python`).
> `python` is not assumed to be on your `PATH`.

---

## 0. Setup & example data

Clone the repo, create a virtualenv, and install the package. **The base install is CPU-only
and pulls nothing private.** Two things below require more than it: the DE examples explicitly
select the `pdex` engine, which the base install does not include (`pip install pdex` — see
§4), and §8's streaming examples need the `scale` extra. Run every command **from the repo
root** — the tutorial's relative data paths (e.g. `docs/data/…`) assume it.

```bash
git clone https://github.com/ArcInstitute/cell-eval2.git
cd cell-eval2

python -m venv .venv && source .venv/bin/activate     # or: uv venv

pip install -e .            # base: CPU, all public dependencies
pip install -e '.[gpu]'     # + cupy (GPU acceleration)
pip install -e '.[scale]'   # + cellstream (.shad streaming) — public on PyPI
```

> The `scale` extra installs [`cellstream`](https://github.com/ArcInstitute/cellstream) from PyPI,
> so it is an ordinary install for everyone. It is optional only because §8's streaming section is
> the one part of this tutorial that needs it. ⚠️ Prebuilt wheels cover **CPython 3.11 and 3.12 on
> glibc x86-64 Linux**; every other interpreter and platform (3.13+, macOS, Windows, ARM, PyPy)
> falls back to the source distribution, which needs a Rust toolchain to build.

`pip install -e .` installs from the current directory (the clone you just `cd`'d into) in
editable mode — it does not fetch `cell_eval2` from anywhere else; only its PyPI dependencies
are downloaded.

Load the example reference. It holds real perturbed + control cells: 600 cells × 1000 genes of
raw integer counts, with the perturbation label in `obs["target_gene"]` and the control cells
labeled `non-targeting`.

```python
import anndata as ad

real = ad.read_h5ad("docs/data/H1-VCC-2025-training.h5ad")
print(real.shape)                          # (600, 1000)
print(real.obs["target_gene"].value_counts())
# non-targeting + 5 perturbations (TMSB4X, STAT1, MED12, TET1, SRC), ~100 cells each.
```

`pert_col="target_gene"` and `control="non-targeting"` describe this dataset; you pass them to
every call below.

**A prediction to score.** cell_eval2 compares a *predicted* AnnData against the *real* one. We
don't have a trained model here, so we fabricate two clearly-illustrative toy predictions from the
real data (these are teaching props, **not** a real model):

- `pred_good` — each cell's expression is the real per-perturbation mean profile plus mild
  multiplicative noise, so it stays correlated with the truth.
- `pred_baseline` — every cell gets the *global* mean profile (the classic "mean baseline"),
  identical across all perturbations.

```python
import numpy as np
import scipy.sparse as sp

def build_predictions(real, seed=1):
    X = real.X.toarray()                       # ndarray of counts
    labels = real.obs["target_gene"].to_numpy()
    rng = np.random.default_rng(seed)

    # pred_good: real per-perturbation mean profile + mild multiplicative noise
    Xg = np.zeros_like(X)
    for g in np.unique(labels):
        m = labels == g
        mean = X[m].mean(axis=0, keepdims=True)
        Xg[m] = np.rint(mean * rng.lognormal(0.0, 0.35, size=(m.sum(), X.shape[1]))).clip(min=0)

    # pred_baseline: the global mean profile broadcast to every cell (ignores perturbation)
    Xb = np.rint(np.repeat(X.mean(axis=0, keepdims=True), X.shape[0], axis=0)).clip(min=0)

    def mk(mat):
        return ad.AnnData(sp.csr_matrix(mat.astype(np.float32)),
                          obs=real.obs.copy(), var=real.var.copy())
    return mk(Xg), mk(Xb)

pred_good, pred_baseline = build_predictions(real)
```

A prediction is just an `AnnData` with the same cells and genes as the reference. You can score
it in memory (as below) or write it to `.h5ad` and score it from disk — `compute_metrics` accepts
either an `AnnData` or a path. The `.h5ad` is exactly the file format you would submit:

```python
pred_good.write_h5ad("pred_good.h5ad")          # the prediction file you'd submit
# then score straight from the path — equivalent to passing the in-memory object:
# compute_metrics("pred_good.h5ad", real, metrics=["expr_mae", "pds_l1"],
#                 pert_col="target_gene", control="non-targeting")
```

---

## 1. Your first evaluation, and reading the output

`compute_metrics(pred, real, ...)` is the main entry point. Give it the two AnnData objects, the
metrics you want, and the perturbation/control labels:

```python
from cell_eval2 import compute_metrics, aggregate_metrics

df = compute_metrics(pred_good, real,
                     metrics=["expr_mae", "pds_l1"],
                     pert_col="target_gene", control="non-targeting")
print(df)
```

The result is a **tidy (long) polars DataFrame** with one row per `(perturbation, metric)`:

```
┌──────────────┬─────────┬──────────┐
│ perturbation ┆ metric  ┆ value    │
│ ---          ┆ ---     ┆ ---      │
│ MED12        ┆ expr_mae┆ 0.0389…  │
│ MED12        ┆ pds_l1  ┆ 1.0      │
│ …            ┆ …       ┆ …        │
└──────────────┴─────────┴──────────┘
```

Reduce it to one number per metric with `aggregate_metrics` (a NaN-skipping mean over
perturbations):

```python
print(aggregate_metrics(df))          # columns: (metric, mean)
# expr_mae ≈ 0.04   pds_l1 = 1.0
```

**Which direction is "good"?** Each metric carries a `scoring` policy. `scoring.direction` is
`"lower"` (e.g. `expr_mae`, a distance), `"higher"` (e.g. `pds_l1`, a rank score), or `None`
for a diagnostic count with no intrinsic best. `scoring.scored` says whether it enters
`avg_score`, and `scoring.anchor` is the value a perfect submission attains — `None` when the
metric is unbounded and has no such value. See the metric catalog in Appendix A below.

> The older `spec.best_value` token still exists and still returns `"zero"` / `"one"` /
> `"none"`, but it is **deprecated and lossy**: it cannot distinguish an anchorless scored
> metric from an anchored one, and both report `"one"`. Read `scoring` instead.

**Good vs baseline — why discrimination matters.** Score the baseline the same way:

```python
print(aggregate_metrics(compute_metrics(pred_baseline, real,
                                        metrics=["expr_mae", "pds_l1"],
                                        pert_col="target_gene", control="non-targeting")))
# expr_mae ≈ 0.06   pds_l1 ≈ 0.55
```

Notice the lesson: the baseline's per-cell error (`expr_mae`) looks *almost as good* as
`pred_good`'s — a mean profile is a decent guess for any single cell. But its `pds_l1` is ≈0.55
(≈chance): because it predicts an identical profile for every perturbation, it **cannot tell
perturbations apart**. `pred_good` scores `pds_l1 = 1.0` — perfect discrimination on these five
well-separated perturbations. Perturbation discrimination is exactly the failure mode a
mean-baseline hides and `pds_*` exposes.

---

## 2. Choosing metrics: all, or a subset

`metrics=` accepts either a **profile name** (a curated group) or an **explicit list** of metric
names.

```python
from cell_eval2 import EvalConfig
from cell_eval2.config import DEParams

# All implemented metrics. 'full' includes DE, so pick a DE backend (see §4); on CPU use pdex.
cfg_full = EvalConfig(metrics="full", pert_col="target_gene", control="non-targeting",
                      de=DEParams(backend="pdex"))
df_full = compute_metrics(pred_good, real, config=cfg_full)

# A curated subset by profile (the 13 expression/discrimination metrics — no DE needed):
df_anndata = compute_metrics(pred_good, real, metrics="anndata",
                             pert_col="target_gene", control="non-targeting")
print(sorted(df_anndata["metric"].unique()))
```

The shipped profiles:

| profile | what it is |
|---|---|
| `full` | every implemented metric (expression + discrimination + DE) |
| `minimal` | a small mixed set (MAE, PDS-l1, pearson-delta, MSE, DE overlap/precision, nsig counts) |
| `vcc` | the 2025 Virtual Cell Challenge scored set: `expr_mae`, `pds_l1`, `de_wilcoxon_overlap` |
| `vcc2026` | the 2026 Virtual Cell Challenge scored set — a *replacement* for `vcc`, not an extension. Six scored: `pds_cosine`, `expr_mse_unbiased_capped_norm`, `de_wilcoxon_lfc_nmae`, `de_wilcoxon_direction_fidelity_yield_raw`, `de_wilcoxon_direction_reach_raw`, `de_wilcoxon_sig_jaccard`; plus four related expression diagnostics — two of them the derived metric's numerator and denominator, two audit-only — so ten entries |
| `anndata` | the 13 expression/discrimination metrics (no DE) |
| `de` | all differential-expression metrics |
| `pds` | just `pds_l1` |

Or pass an explicit list. Canonical (v2) names, v1 names, and aliases all resolve to the same
metric and are de-duplicated:

```python
compute_metrics(pred_good, real, metrics=["expr_mae", "pds_l1"],
                pert_col="target_gene", control="non-targeting")               # canonical
compute_metrics(pred_good, real, metrics=["mae", "discrimination_score_l1"],
                pert_col="target_gene", control="non-targeting")               # v1 names — same two metrics
```

Two metrics (`edistance_pearson`/`pearson_edistance`, `clustering_agreement`) are recognized but
not yet implemented: if you request them explicitly they are accepted and skipped with a warning.

---

### 2a. Scoring *direction* rather than set membership

Most DE metrics ask which genes a model called. The **direction metrics** ask a different
question: of the calls it made, how many point the right way — and how deep does its ranking stay
pointing the right way? Significant-gene sets replicate at Jaccard ≈ 0.26–0.39 on real data, while
the *direction* of those calls replicates at 0.89–0.99, so direction is the part that reproduces.

Three metrics, all in the `full` and `de` profiles and none in `vcc`, `vcc2026` or `minimal` — so
they enter the `full`/`de` `avg_score` but never a competition score:

```python
# continues from §2 — these are DE metrics, so pick a DE backend (see §4); on CPU use pdex
cfg_dir = EvalConfig(
    metrics=[
        "de_wilcoxon_direction_precision",             # [0,1], higher better
        "de_wilcoxon_direction_sensitivity",           # [0,1], higher better
        "de_wilcoxon_direction_sensitivity_universe",  # unbounded — see below
    ],
    pert_col="target_gene", control="non-targeting",
    de=DEParams(backend="pdex"),
)
df_dir = compute_metrics(pred_good, real, config=cfg_dir)
print(df_dir.pivot(on="metric", index="perturbation", values="value"))
```

`metrics="de"` and `metrics="full"` include all three already. If you have DE tables computed
elsewhere, pass them as `de_pred=` / `de_real=` (§4) — that skips the backend's *computation*, but
`de.backend` still decides the metric-family name and the cache provenance, so keep it consistent
with whatever produced the tables or a DESeq2 table will be reported under `de_wilcoxon_*`. Under
`EvalConfig(de=DEParams(backend="deseq2"))` they are emitted as `de_deseq2_direction_*`.

**`direction_precision`** — of the genes the model called significant, the fraction whose log₂FC
sign the reference agrees with. The reference's *own* significance is deliberately ignored, which
is what makes it discriminate: conditioning on both sides calling a gene forces the answer to ≈1.0.

**`direction_sensitivity`** — rank the reference-significant genes by the model's evidence, walk
down that list, and find the deepest point at which directional purity still holds at
`P0 = 1 − α/2` (0.975 at the default α = 0.05). Report that depth as a fraction of how many genes
the reference confidently called. `P0` is derived, not a knob.

**`direction_sensitivity_universe`** — the same, ranked over every shared gene instead. ⚠️ This one
is **unbounded above** and is known to invert against a generic-response baseline. It is scored
anyway — it has a direction, and that is the enrolment rule — but with `anchor=None`, so it
normalizes against its own baseline and is clamped to `[-2, 2]`. The clamp bounds how far the
inversion can move an aggregate; it does not correct it. Prefer the bounded variant unless you
specifically want the unrestricted curve.

**What counts as "no direction".** A log₂FC that is null, `NaN`, or exactly `0` carries no
direction (`±inf` does — it just means one condition is zero and the other is not; the sign
still says which way). The rule is asymmetric,
and deliberately so:

| situation | treatment |
|---|---|
| reference can't adjudicate — gene absent from the real table, or its log₂FC undefined | pair **excluded** |
| model declined to commit — its log₂FC undefined while calling the gene significant | counts as a **miss** |

A model can therefore never improve its score by declining to answer, which is the same principle
that makes every metric fill a failed perturbation with its worst value rather than dropping it.

**Two gotchas.** For the two *sensitivity* variants, perturbations where the reference called
nothing significant are undefined and take the worst value (`0`) under v2 — `direction_precision`
is unaffected, since it ignores reference significance by design and can still score 1.0 there — on real data that is 13–24% of targets, so the aggregate
moves. Note all three are scored and enter the `full`/`de` `avg_score`; none is in `vcc` or
`vcc2026`, so neither competition ranking is affected. They are v2-native, so `score_agg_metrics`
(the upstream-compat scorer) declines all three with a warning — use `score_metrics`.
And duplicate `(target, feature)`
rows now raise rather than being silently double-counted, because a duplicate inflates the
adjudicated sensitivity past its own upper bound of 1.

There is also an older `de_wilcoxon_model_direction_match`, kept unchanged for continuity with
v0.3.0. It computes the same quantity but scores a both-zero or both-`NaN` pair as *agreement*.
Prefer `de_wilcoxon_direction_precision` for new work.

---

### 2b. Scoring against the generic-response baseline

A raw metric is easier to interpret beside a generic-response comparator. The three commands
below use the `pred_good.h5ad` written in §0 and the same committed reference, perturbation column,
control label, and CPU DE backend used above:

```bash
cell-eval2 baseline -ar docs/data/H1-VCC-2025-training.h5ad -o baseline/ \
    --profile vcc --pert-col target_gene --control non-targeting \
    --input-type counts --set de.backend=pdex

cell-eval2 run -ap pred_good.h5ad -ar docs/data/H1-VCC-2025-training.h5ad -o user/ \
    --profile vcc --pert-col target_gene --control non-targeting \
    --input-type counts --set de.backend=pdex

cell-eval2 score --user-agg user/agg_results.csv \
    --baseline-agg baseline/baseline_agg.csv -o user/from_baseline.csv
```

This is an **oracle comparator, not a floor a submission could reach**: its average response is
computed from the evaluated real perturbations, which a real submission cannot access. The scored
CSV reports each metric's margin over that comparator plus `avg_score`; §6a of
[`metrics.md`](metrics.md) explains the interpretation and provenance checks.

---

### 2c. How high can any model score? The data ceiling

§2b gives a metric a comparator at the trivial end. The **data ceiling** gives it one at the other
end: an estimate of how high *any* model could score given the sampling noise in the real data
itself. Nothing about a prediction enters it — the ceiling is a property of the real data alone.

The measurement is a split-half reliability. Each perturbation's cells (and the control's) are
shuffled and split into two **disjoint** halves of `floor(n/2)` cells; one half is scored as if it
were a prediction of the other. Under the parallel-halves approximation their differences are
treated as sampling variation, so their agreement is what the metric can resolve at *half* depth. The
[Spearman–Brown](https://en.wikipedia.org/wiki/Spearman%E2%80%93Brown_prediction_formula)
correction `r' = 2r/(1+r)` maps that back to the depth of the two halves combined — the run's own
depth, give or take the one cell an odd count drops:

```python
# continues from §2 — the ceiling reads only `real`; `pred_good` plays no role.
# These are DE metrics, so pick a DE backend (see §4); on CPU use pdex.
from cell_eval2 import compute_ceiling

cfg_ceil = EvalConfig(metrics="vcc", pert_col="target_gene", control="non-targeting",
                      input_type="counts", de=DEParams(backend="pdex"))
results_ceil, agg_ceil = compute_ceiling(real, config=cfg_ceil, seed=0)
print(agg_ceil)
# metric                ceiling
# expr_mae                  NaN     <- error metrics get no ceiling (see below)
# pds_l1                  0.947
# de_wilcoxon_overlap     0.298
```

`results_ceil` is the tidy per-perturbation self-split frame; `agg_ceil` is `(metric, ceiling)` —
one row per emitted metric, spelled the way *this config* spells it, so it lines up name-for-name
with a matching run's `agg_results.csv` and many-to-one onto its long-form `results.csv` (under
`--version v1` that spelling is `discrimination_score_l1`, not `pds_l1`).

Read that output as: on this toy dataset, ~50 cells per half cannot reproduce their own DE calls,
so `de_wilcoxon_overlap`'s estimated reproducibility at this depth is only ≈0.30, while `pds_l1` is
nearly saturated at ≈0.95. **A low ceiling is a statement about the data's depth, not about the
model.** Comparing a submission's 0.25 overlap against a ceiling of 0.30 tells a very different
story than comparing it against 1.0. It is an estimate, not a proved bound — measured on one
random split, and a model that denoises the reference better than half of it does could score
above it.

From the CLI, `--ceiling` rides along with a normal run, and `-ap/--adata-pred` becomes optional —
omit it to compute *only* the ceiling:

```bash
# ceiling only: no prediction, so no results.csv / agg_results.csv / run_meta.json
cell-eval2 run -ar docs/data/H1-VCC-2025-training.h5ad --ceiling -o ceil/ \
    --profile vcc --pert-col target_gene --control non-targeting \
    --input-type counts --set de.backend=pdex
# -> writes ceil/ceiling_results.csv + ceil/ceiling_agg.csv

# or alongside the usual scoring, into the same outdir
cell-eval2 run -ap pred_good.h5ad -ar docs/data/H1-VCC-2025-training.h5ad --ceiling -o out/
```

**Only verified metrics get a number.** The correction is applied to a hand-maintained list of 18
reliability metrics (`cell_eval2.ceiling.SB_METRICS`), each checked 1:1 against the validated
cell-eval implementation the ceiling was justified on. The other 36 metrics the `full` profile
emits under a wilcoxon backend — the AUC metrics; the error metrics, including
`expr_mse_unbiased_capped_norm` and `de_wilcoxon_lfc_nmae`; the four expression diagnostics;
significant-gene counts; the four
rank/significance chance-corrected metrics;
the three #187 direction metrics; the eleven #195 chance-corrected direction metrics;
`model_direction_match`; the signed LFC-Spearman pair; and `sig_jaccard` — are reported as
`NaN` rather than given an unvalidated ceiling. `NaN` here means "no defensible ceiling", not
"zero". Under `de.backend="deseq2"` every DE metric is renamed `de_deseq2_*` and joins that list,
leaving a ceiling only on `delta_pearson` and the three PDS variants.

Four things worth knowing before you quote a ceiling:

- **The inner run forces `control_source="pred"`**, whatever your config says, so each half uses its
  own control cells. Under `"real"` both halves' log₂FCs would be computed against the *same*
  control and share its sampling noise — which biases the ceiling upward.
- **A non-positive split-half reliability yields `NaN`**, not a negative ceiling: `2r/(1+r)` is a
  correction only for `r > 0`. A split that gives no positive evidence of repeatability has no
  defensible ceiling.
- **Cells are dropped, so the perturbation set can differ from the main run.** An odd cell out is
  discarded (both halves must be the same depth), and a perturbation with fewer than 2 cells cannot
  be split at all — it is dropped from both halves, with a warning. A *control* that cannot be
  split is fatal, not a warning.
- **It roughly doubles the wall time of a DE-bearing run.** The real matrix is loaded unbacked and
  split into two copies, and — when the selected SB metrics include DE ones — DE is recomputed on
  each half. On the CLI the ceiling runs after the main scoring and reloads the real data, so peak
  memory is about the larger phase, not the sum; from the Python API your own AnnData stays
  resident and they do add up.

§6b of [`metrics.md`](metrics.md) gives the formula, the assumptions it rests on, and the full
exclusion table.

---

## 3. Evaluation standards: v1, v1.5, v2

cell_eval2 ships these standards as presets on `EvalConfig`:

- **v2** — the native default (`EvalConfig()` is identical to `EvalConfig.v2()`). Arc's corrected
  standard: real control, rank denominator `n−1`, cosine PDS, mask NaN log-fold-changes, CPM
  normalization (`target_sum=1e6`), gene filter on.
- **v1** — `EvalConfig.v1()` (alias `legacy()`). Reproduces upstream cell-eval / VCC
  **byte-for-byte** and emits the v1 metric names: predicted control, rank denominator `n`, l1,
  keep NaN-LFCs, median normalization, global FDR, LFC clip 20.
- **`cell-eval-0.7.6`** — `EvalConfig.from_preset("cell-eval-0.7.6")` (alias `cell_eval_0_7_6`).
  Reproduces a *modern* upstream — cell-eval 0.7.6 + pdex 0.2.6 (the `gpu` branch) —
  per-perturbation metrics. It is a v1/v2 **hybrid** (v1 predicted-control / geometric mean /
  rank `n` + v2 5-CPM filter / per-pert FDR / epsilon `1e-9` + a `clip` AUC floor) that emits the
  v1 metric names. Use it to match a forward-eval scored by upstream cell-eval.

```python
from cell_eval2 import EvalConfig, FilterParams
from cell_eval2.config import DEParams   # NOTE: DEParams lives in cell_eval2.config (FilterParams is top-level)

cfg_v2 = EvalConfig()
cfg_v1 = EvalConfig.v1()
assert cfg_v2 == EvalConfig.v2()
```

**"v1.5" is not a shipped preset** — it is any *custom* standard you assemble from individual
knobs. A useful one is "v2 numerics, but with v1's three statistical knobs" (predicted control,
gene filter off, global FDR). Build it with the `FilterParams` / `DEParams` dataclasses (not raw
dicts) so it round-trips cleanly through YAML:

```python
cfg_v1_5 = EvalConfig(
    control_source="pred",                                 # v1 knob
    filter=FilterParams(filter_gene_min_cpm_cell=None),    # v1 knob: gene filter off
    de=DEParams(fdr_scope="global"),                       # v1 knob
)

# Save / load any config as YAML:
cfg_v1_5.to_yaml("v1_5_config.yaml")
cfg_roundtrip = EvalConfig.from_yaml("v1_5_config.yaml")
assert cfg_roundtrip == cfg_v1_5
```

**`version` is a naming + provenance toggle, not a numeric master switch.** Selecting the `v1`
preset sets both the numeric conventions *and* the output labels; setting `version` alone only
changes which metric names appear in the output (and v1's input-type auto-guess). The numeric
behavior comes from the preset/knobs, not from `version`. (Relatedly: which PDS distance runs is
chosen by the metric *name* — `pds_l1` / `pds_l2` / `pds_cosine` — not by
`discrimination.distance`.)

---

## 4. Differential-expression inputs

The DE-containing profiles (`vcc`, `vcc2026`, `minimal`, `de`, `full`) need a per-perturbation
differential-expression table. You can either let `compute_metrics` compute it internally, or pass
precomputed tables via `de_pred=` / `de_real=` (paths or frames) to skip and share the work across
runs.

The DE backend is chosen by `EvalConfig.de.backend`. The default `auto` returns **gpudge** (the GPU
engine; needs torch + CUDA) when it is available. Otherwise it is **tiered**: on a host with a
visible CUDA device it **raises** rather than silently switching engines, because the DE numbers
would change with no signal; on a host with no CUDA device it warns once and uses **pdex**, or
**scanpy** if pdex is missing (scanpy is substantially slower). pdex and scanpy are the CPU
engines; **scanpy** is a hard dependency and is always present, while **pdex** is not part of
the base install — `pip install pdex` to get the faster CPU engine. (It does arrive
transitively with the dev dependency group, via `cell-eval`, so a dev environment already has
it.) On a CPU box, pin the backend so it runs without a GPU and without the warning:

```python
from cell_eval2 import EvalConfig, compute_metrics, aggregate_metrics
from cell_eval2.config import DEParams

cfg_de = EvalConfig(metrics="de", pert_col="target_gene", control="non-targeting",
                    de=DEParams(backend="pdex"))
df_de = compute_metrics(pred_good, real, config=cfg_de)
print(aggregate_metrics(df_de))
# de_wilcoxon_overlap, de_wilcoxon_precision, de_wilcoxon_roc_auc, … (the full `de` profile)
```

On this 600-cell subset the CPU DE runs in a few seconds.

**Another backend: `deseq2`.** An opt-in, non-default `deseq2` backend computes DE with a
pseudobulk **negative-binomial GLM** (via the `deseq2_gpu` engine, an **Arc-internal package**,
so it is not installable outside Arc) instead of a Wilcoxon rank test. It owns its own
LFC + p-values and reports them under a dedicated `de_deseq2_*` metric family.

---

## 5. CPU vs GPU

Device selection is the `EvalConfig.device` knob — `"auto"` (default), `"cuda"`, or `"cpu"`:

- `"auto"` uses the GPU when cupy is installed *and* a GPU is present, otherwise CPU.
- `"cuda"` forces the GPU and raises if the `gpu` extra isn't installed.
- `"cpu"` forces the CPU.

```python
cfg_cpu = EvalConfig(metrics="anndata", device="cpu",
                     pert_col="target_gene", control="non-targeting")
df_cpu = compute_metrics(pred_good, real, config=cfg_cpu)
```

The GPU path accelerates pseudobulk aggregation, the PDS/discrimination ranks, and DE (via
gpudge). Two things to note:

- **`device` is a config/YAML setting — there is no `--device` CLI flag** (see §7). The DE
  `backend` (§4) is a *separate* knob from `device`.
- The speedup only matters at scale. On large real datasets (aliased dataset **CCL_1**:
  1.2 M cells, ~5,100 perturbations), a full 8-metric run *including*
  discrimination takes **≈503 s (8.4 min) on a GPU (H100) vs ≈4.9 h on CPU** — the O(P²×G)
  discrimination wall alone collapses from **≈282 min to 16.4 s (≈1000×)**, at lower peak host
  memory (25.4 GiB GPU vs 42.2 GiB CPU).

---

## 6. Caching real & predictions

Scoring many predictions against one shared real reference? Point `cache_real` / `cache_pred` at
folders and the real side is computed once and reused; only the prediction side is recomputed per
prediction. `cache_strict=True` content-hashes the matrix instead of the default metadata
fingerprint; corrupt/stale entries self-heal.

```python
import tempfile, os
# A temp dir keeps this demo self-cleaning; in real use, point cache_real/cache_pred at a
# PERSISTENT directory you keep across runs — that persistence is the whole value of caching.
with tempfile.TemporaryDirectory() as d:
    cfg = EvalConfig(metrics="anndata", pert_col="target_gene", control="non-targeting",
                     cache_real=os.path.join(d, "real"), cache_pred=os.path.join(d, "pred_good"))
    # First run computes + populates both caches:
    df1 = compute_metrics(pred_good, real, config=cfg)
    # Score a second prediction reusing the SAME real cache (only pred recomputed):
    cfg_b = EvalConfig(metrics="anndata", pert_col="target_gene", control="non-targeting",
                       cache_real=os.path.join(d, "real"), cache_pred=os.path.join(d, "pred_baseline"))
    df2 = compute_metrics(pred_baseline, real, config=cfg_b)
    print("real cache reused across predictions:", os.listdir(os.path.join(d, "real")))
```

To warm one side ahead of time (loading only that side), use `precompute_cache`:

```python
from cell_eval2 import precompute_cache
# precompute_cache(real, side="real", config=cfg)   # or the `prep-cache` CLI in §7
```

Passing **paths** (rather than in-memory AnnData) keeps peak memory to one matrix at a time.

---

## 7. The command line

`cell-eval2` mirrors the Python API. Given `pred_good.h5ad` and `real.h5ad` on disk:

```bash
cell-eval2 run -ap pred_good.h5ad -ar real.h5ad \
    --profile vcc --pert-col target_gene --control non-targeting -o out/
# -> writes out/results.csv, out/agg_results.csv, out/run_meta.json (and out/run_params.yaml)

cell-eval2 prep-cache --side real --adata real.h5ad --cache-real cache/real \
    --pert-col target_gene --control non-targeting
```

A few things that trip people up:

- Metric selection is **`--profile`** (a profile name) — there is **no `--metrics`** flag for
  explicit lists on the CLI.
- **Any config field without a dedicated flag** — nested `de.*` / `discrimination.*` / `filter.*`
  and top-level fields like `device` / `target_sum` — is reachable via the generic, repeatable
  **`--set KEY.PATH=VALUE`** (e.g. `--set device=cpu --set de.min_abs_log2fc=0.25`). `VALUE` is
  parsed as YAML and validated exactly like the config; `--set` applies on top of `--config` and
  the explicit flags. (There is still no dedicated `--device` flag — use `--set device=…` or a
  `--config` YAML.)
- **`--version`** selects the v1/v2 conventions — it is not a "print program version" flag.
- Output is `results.csv` (plus `agg_results.csv` and `run_meta.json`) in the `--outdir`
  (default `./cell-eval2-outdir`). `--ceiling` adds `ceiling_results.csv` + `ceiling_agg.csv`
  (§2c), and is the one case where **`-ap/--adata-pred` may be omitted** — a ceiling-only run
  writes the two ceiling files and nothing else. `--de-pred`/`--de-real` and
  `--cache-real`/`--cache-pred` do not apply to a ceiling-only run and are warned about rather
  than silently dropped.

---

## 8. Scaling up: streaming and partitioned scoring

For a prediction that fits in memory, `compute_metrics` (whole-object, in-RAM) is the fastest path.
For a prediction too large to hold in memory, use the **streaming** path, which reads a packed
`cellstream` `.shad` archive shard-by-shard and never materializes the full matrix. A `.shad`
stores raw counts, sharded by the perturbation column. (`cellstream` is the packer the `scale`
extra installs from PyPI — see §0. It is the public successor to the Arc-internal `shardad` this
path was built against; the `SHPK` container is unchanged, so archives written by either package
read here.)

```python
from cell_eval2 import EvalConfig
from cell_eval2.scale import score_streaming

cfg = EvalConfig(metrics="anndata", pert_col="target_gene", control="non-targeting")
df = score_streaming("pred.shad", "real.shad", config=cfg)   # memory-bounded
```

**Optional — build a tiny `.shad` and run it (requires `pip install -e '.[scale]'`).** DE-at-scale
needs a GPU (gpudge), so the CPU streaming demo uses the `anndata` profile:

```python
import tempfile, os
from cellstream import write_sharded
from cell_eval2 import EvalConfig
from cell_eval2.scale import score_streaming

with tempfile.TemporaryDirectory() as d:
    real_shad = os.path.join(d, "real.shad")
    pred_shad = os.path.join(d, "pred_good.shad")
    write_sharded(real, real_shad, group_by="target_gene")
    write_sharded(pred_good, pred_shad, group_by="target_gene")
    cfg = EvalConfig(metrics="anndata", pert_col="target_gene", control="non-targeting")
    df_stream = score_streaming(pred_shad, real_shad, config=cfg)   # CPU; DE-at-scale is GPU-only
```

**Partitioned perturbations.** Split the perturbation universe into `fraction` disjoint subsets and
score each on its own run/node against the *same* real reference, writing a partial each time; then
recombine:

```python
from cell_eval2.scale import score_streaming
from cell_eval2.partition import aggregate_partials

# on each worker i in [0, N):
score_streaming("pred.shad", "real.shad", config=cfg,
                fraction=N, index=i, partial_out="parts/")

# once all partials exist:
full_tidy, aggregate = aggregate_partials("parts/")
```

`aggregate_partials` refuses to combine partials computed against different references or configs,
or with overlapping `(perturbation, metric)` rows. (The `score_streaming` caching is driven by
`config.cache_real`; there is no separate `cache_real` argument.)

> ⚠️ **Do not mix partials written by different cell_eval2 releases — discard and rebuild the
> whole directory.** A partial sidecar records the reference and config hashes and the
> perturbations it covers, but **not** which metrics were in the profile, and
> `aggregate_partials` applies the **current** catalog when it reduces them. So partials that
> straddle a change in profile MEMBERSHIP concatenate into a plausible aggregate over an
> incomplete cohort, with no error. This bites `vcc2026` directories spanning v0.7.x and
> v0.8.0, where the profile swapped its two direction members for their `_raw` siblings.
> Tracked in #246. A sibling **in-RAM partitioned**
path — `cell_eval2.partition_inmem.build_reference` / `score_piece` — scores disjoint in-memory
`.h5ad` pieces against one shared reference (v2-only, gpudge DE).

Neither path has a `cell-eval2` subcommand — drive both from the Python API above.

**How this holds up at scale** (Arc's internal benchmark runs, aliased datasets):

- CPU streaming stays memory-bounded: **peak host RSS 54.6 GiB at 5.54 M cells** (CCL_2) —
  far under a 512 GB budget — with throughput ~1,074–1,538 cells/s.
- On a single GCP A100-40GB, three 300-perturbation sets score in **256.3 s** (≈4.3 min of
  compute). Billable VM uptime is **555 s ≈ 9 min** — roughly half of it one-time provisioning,
  which a pre-baked image removes. Peak GPU **38–39 GiB** — the device-wide `memory.used` field
  sampled from `nvidia-smi`, i.e. memory allocated from the driver (caching allocators, the CUDA
  context, any co-resident process) rather than the live working set, so it does **not** imply a
  40 GiB minimum (H100 baseline: 33.9 s / 300-pert).

---

## 9. Drop-in parity with cell-eval / VCC

To reproduce upstream cell-eval or a VCC submission's numbers byte-for-byte, score under the **v1**
preset — it also emits the v1 metric names:

```python
# compute_metrics accepts keyword overrides on top of `config`, so pass the v1 preset directly:
df_v1 = compute_metrics(pred_good, real, config=EvalConfig.v1(),
                        metrics="anndata", pert_col="target_gene")
print(sorted(df_v1["metric"].unique()))
# v1 labels: mae, mae_delta, mse, mse_delta, pearson_delta,
#            discrimination_score_l1, discrimination_score_l2, discrimination_score_cosine
```

For existing cell-eval call sites, the deprecated `cell_eval2.compat.MetricsEvaluator` shim wraps
this (it forces the v1 preset with `de.backend="pdex"`).

---

## Appendix A: metric catalog

Canonical (v2) name, inherited v1 name, the profiles it belongs to, the derived `best_value`
token (`one` = scored, higher is better; `zero` = scored, lower is better; `none` = **not
scored**), and its kind. The token is an enrolment label, not a direction: `none` can cover both
"no direction" (the four `de_*_nsig_counts_*` entries and the four expression diagnostics) and
"has one but is deliberately not enrolled" — `expr_mse_unbiased_norm` occupied the latter until
it was enrolled, and #257 then removed it outright. `docs/metrics.md` §7 splits `dir` / `anchor` / `scored` into their own
columns and is the precise version. Metrics with no v1 name (—) are v2-native: they have no upstream cell-eval counterpart
and are never emitted under `version="v1"`. That is a statement about v1 availability, not about
enrolment — most of them are scored.

> ⚠️ This table omits the **eleven chance-corrected direction metrics** (`de_*_direction_fidelity`,
> `…_coverage`, `…_yield`, `…_reach`, and their `_raw`/`_unbounded` variants) added by issue #195,
> the `de_deseq2_*` mirrors, and the four unscored expression diagnostics of #257/#264
> (`expr_mse_unbiased`, `expr_mse_unbiased_capped`, `expr_distance_unbiased`,
> `expr_real_mass_ratio`). The complete 48-row catalog with ranges and anchors is
> [`docs/metrics.md`](metrics.md) §7.

| metric (v2) | v1 name | profiles | best | kind |
|---|---|---|---|---|
| `expr_mae` | `mae` | full, minimal, vcc, anndata | zero | anndata |
| `pds_l1` | `discrimination_score_l1` | full, minimal, vcc, anndata, pds | one | anndata |
| `pds_l2` | `discrimination_score_l2` | full, anndata | one | anndata |
| `pds_cosine` | `discrimination_score_cosine` | full, anndata, vcc2026 | one | anndata |
| `delta_pearson` | `pearson_delta` | full, minimal, anndata | one | anndata |
| `expr_mse` | `mse` | full, minimal, anndata | zero | anndata |
| `expr_mse_unbiased_capped_norm` | — | full, anndata, vcc2026 | zero | anndata |
| `delta_mse` | `mse_delta` | full, anndata | zero | anndata |
| `delta_mae` | `mae_delta` | full, anndata | zero | anndata |
| `de_wilcoxon_overlap` | `overlap_at_N` | full, minimal, de, vcc | one | de |
| `de_wilcoxon_overlap_top50` | `overlap_at_50` | full, de | one | de |
| `de_wilcoxon_overlap_top100` | `overlap_at_100` | full, de | one | de |
| `de_wilcoxon_overlap_top200` | `overlap_at_200` | full, de | one | de |
| `de_wilcoxon_overlap_top500` | `overlap_at_500` | full, de | one | de |
| `de_wilcoxon_precision` | `precision_at_N` | full, minimal, de | one | de |
| `de_wilcoxon_precision_top50` | `precision_at_50` | full, de | one | de |
| `de_wilcoxon_precision_top100` | `precision_at_100` | full, de | one | de |
| `de_wilcoxon_precision_top200` | `precision_at_200` | full, de | one | de |
| `de_wilcoxon_precision_top500` | `precision_at_500` | full, de | one | de |
| `de_wilcoxon_nsig_counts_real` | `de_nsig_counts_real` | full, minimal, de | none | de |
| `de_wilcoxon_nsig_counts_pred` | `de_nsig_counts_pred` | full, minimal, de | none | de |
| `de_wilcoxon_nsig_spearman` | `de_spearman_sig` | full, de | one | de |
| `de_wilcoxon_sig_recall` | `de_sig_genes_recall` | full, de | one | de |
| `de_wilcoxon_direction_match` | `de_direction_match` | full, de | one | de |
| `de_wilcoxon_model_direction_match` | `de_model_direction_match` | full, de | one | de |
| `de_wilcoxon_direction_precision` | — | full, de | one | de |
| `de_wilcoxon_direction_sensitivity` | — | full, de | one | de |
| `de_wilcoxon_direction_sensitivity_universe` | — | full, de | one | de |
| `de_wilcoxon_lfc_spearman` | `de_spearman_lfc_sig` | full, de | one | de |
| `de_wilcoxon_lfc_nmae` | — | full, de, vcc2026 | zero | de |
| `de_wilcoxon_pr_auc` | `pr_auc` | full, de | one | de |
| `de_wilcoxon_roc_auc` | `roc_auc` | full, de | one | de |
| `de_wilcoxon_overlap_adjusted` | — | full, de | one | de |
| `de_wilcoxon_precision_adjusted` | — | full, de | one | de |
| `de_wilcoxon_sig_recall_adjusted` | — | full, de | one | de |
| `de_wilcoxon_sig_mcc` | — | full, de | one | de |
| `de_wilcoxon_sig_jaccard` | — | full, de, vcc2026 | one | de |

## Appendix B: key `EvalConfig` fields

| field | default | meaning |
|---|---|---|
| `metrics` | `"full"` | profile name or explicit list of metric names |
| `pert_col` | `"target"` | obs column with the perturbation label (this dataset uses `target_gene`) |
| `control` | `"non-targeting"` | control label within `pert_col` |
| `version` | `"v2"` | output metric-name convention + provenance stamp |
| `control_source` | `"real"` | which side supplies the control (v1 uses `"pred"`) |
| `input_type` | `"counts"` | `"counts"` or `"lognorm"` |
| `target_sum` | `1e6` | normalization target (`None` = median normalize, v1) |
| `device` | `"auto"` | `"auto"` / `"cuda"` / `"cpu"` (no CLI flag) |
| `filter` | `FilterParams()` | gene filter (`filter_gene_min_cpm_cell`, default 5.0; v1 = off) |
| `de` | `DEParams()` | DE settings incl. `backend`, `fdr_scope`, `nan_lfc_policy`, … |
| `cache_real` / `cache_pred` | `None` | cache folders for the real / prediction side |

## Links

- [README](../README.md) — install, status, and API overview
- [cell-eval](https://github.com/ArcInstitute/cell-eval) — the upstream suite cell_eval2 reimplements

[`cellstream`](https://github.com/ArcInstitute/cellstream) — the `.shad` archive format read by
§8 — is **public on PyPI**, installed by the `scale` extra (see §0). It supersedes the
Arc-internal `shardad` this path was originally built against.
