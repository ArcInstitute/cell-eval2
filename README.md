# cell_eval2

Evaluation of perturbation-prediction models at single-cell resolution: a GPU-accelerated 
streamlined reimplementation of [`cell-eval`](https://github.com/ArcInstitute/cell-eval).

- **[Tutorial](docs/tutorial.md)** — a runnable tour: metric selection, caching, CPU/GPU, streaming.
- **[Metrics reference](docs/metrics.md)** — the definition of every metric in the catalog.
- **[DE tables](docs/de_tables.md)** — what "unthresholded" means, the DE columns and row order,
  and the `score_cellstream(write_de=True)` output contract.
- **[Ground-truth DE cache](docs/ground_truth_cache.md)** — the content-addressed, immutable cache
  of the real-DE table on the `score_cellstream` path: identity, on-disk format, policies, CLI.
- **[vcc2026 metric specification](docs/vcc2026_metrics/)** — what the 2026 competition scores, and
  how. Start here if you are entering: the
  [full document](docs/vcc2026_metrics/vcc2026-metrics.pdf) carries the reasoning, the
  [abridged edition](docs/vcc2026_metrics/vcc2026-metrics-brief.pdf) is definitions and measured
  values only.
- **[v1 parity notes](docs/parity/v1-parity-divergences.md)** — every accepted behavioural
  difference from upstream `cell-eval`, and why each is left alone.

## Install

Python >= 3.11. The base install is **CPU-only** and needs no GPU, no CUDA and nothing private:

```bash
pip install cell-eval2
cell-eval2 --help
```

| extra | adds | when you need it |
|---|---|---|
| *(none)* | — | everything except GPU DE and out-of-core input |
| `[gpu]` | `cupy-cuda12x`, `nvidia-nvcomp-cu12<6` | CUDA pseudobulk / distance / rank kernels |
| `[gpudge]` | [`gpudge`](https://github.com/ArcInstitute/gpudge) | GPU differential expression |
| `[scale]` | [`cellstream`](https://github.com/ArcInstitute/cellstream) | scoring `.csad` archives out of core |

Combine them: `pip install 'cell-eval2[gpu,gpudge,scale]'`.

### With and without a GPU

- **No GPU** — everything works except GPU DE. Expression, PDS and delta metrics run on NumPy; DE
  runs on a CPU engine (`pdex`, or `scanpy` if pdex is absent).
- **With a GPU** — add `[gpu]` for the kernels and `[gpudge]` for DE. `de.backend="auto"` then
  selects `gpudge`.
- ⚠️ `auto` is **tiered, not a silent fallback**. On a host where a CUDA device is visible but no
  GPU DE engine is installed it **raises**: quietly computing DE on a CPU engine there would move
  every DE number with no signal. With no CUDA device it falls back and warns once. Set
  `de.backend` explicitly (`"pdex"`, `"scanpy"`) to pin an engine and silence it.
- ⚠️ **`[gpudge]` pulls `torch`, and the wheel must match your driver.** A default-index torch can
  be unusable on a driver where cupy works fine. Install with
  `uv pip install --torch-backend=auto 'cell-eval2[gpudge]'` (uv picks the build for the detected
  driver), or name the index yourself:
  `pip install 'cell-eval2[gpudge]' --extra-index-url https://download.pytorch.org/whl/cu126`.

### Optional dependencies

Both are public Arc packages on PyPI. Both are optional; neither is needed for the CPU path.

- **[`cellstream`](https://github.com/ArcInstitute/cellstream)** (`[scale]`, `>=0.9.1`) — the
  `SHPK` archive format and its readers, for scoring datasets too large to hold in memory.
  cell_eval2 reads its **cell layout** (`.csad`, one row group per perturbation).
  ⚠️ Prebuilt wheels cover **CPython 3.11 and 3.12 only**; on 3.13 the resolver falls through to
  the source distribution and needs a Rust toolchain, so prefer 3.11/3.12 for this extra.
- **[`gpudge`](https://github.com/ArcInstitute/gpudge)** (`[gpudge]`, `>=0.9.0`) — the GPU
  differential-expression engine. It computes fold changes and p-values on-GPU, and is what
  `de.backend="auto"` selects when a CUDA device is present. Kept separate from `[gpu]` because it
  pulls `torch`, which most users of the kernels do not want.

## Quickstart

```python
from cell_eval2 import compute_metrics, EvalConfig

# expression / PDS metrics need only the (pred, real) pair
df = compute_metrics("pred.h5ad", "real.h5ad", metrics=["expr_mae", "pds_cosine"])

# a metric profile
cfg = EvalConfig(metrics="de", pert_col="target_gene", control="non-targeting",
                 input_type="counts")
df = compute_metrics("pred.h5ad", "real.h5ad", config=cfg)

# the competition: the preset carries every vcc2026 parameter
from dataclasses import replace
cfg = replace(EvalConfig.from_preset("vcc2026"), pert_col="target_gene")
df = compute_metrics("pred.h5ad", "real.h5ad", config=cfg)
```

```bash
# a metric profile
cell-eval2 run -ap pred.h5ad -ar real.h5ad \
  --profile de --pert-col target_gene --control non-targeting --input-type counts -o out/

# the competition: --preset carries every vcc2026 parameter, not just the metric list
cell-eval2 run -ap pred.h5ad -ar real.h5ad --preset vcc2026 --pert-col target_gene -o out/
```

`--preset` (`vcc2026`, `v1`, `v2`, `cell-eval-0.7.6`) and `--config` set the config *base*;
explicit flags and `--set KEY.PATH=VALUE` override it.

## Metrics

Four classes, normally selected by profile rather than one at a time:

- **Expression error** (`expr_*`) — how far the predicted profile sits from the measured one,
  including the sampling-noise-corrected forms the 2026 competition scores.
- **Perturbation discrimination** (`pds_*`) — is a predicted profile closer to *its own* measured
  profile than to any other perturbation's? A rank, so it measures separability rather than
  distance. `pds_l1` / `pds_l2` / `pds_cosine` differ only in the distance used.
- **Signed-effect geometry** (`delta_*`) — direction and magnitude of the predicted effect relative
  to control.
- **Differential expression** (`de_*`, the largest family) — computed from per-gene rank tests:
  overlap, precision and recall, rank and fold-change correlation, direction agreement,
  significant-set agreement, and chance-corrected variants.

| profile | members | scored |
|---|---|---|
| `full` | 54 | 48 |
| `de` | 41 | 39 |
| `anndata` | 13 | 9 |
| `minimal` | 8 | 6 |
| `vcc2026` | 10 | 6 |
| `vcc` | 3 | 3 |
| `pds` | 1 | 1 |

Unscored members are **diagnostics** — reported but not enrolled in a score, usually because they
are a scored metric's numerator, denominator or count.

Every metric carries a canonical **v2** name and an inherited **v1** name. `EvalConfig.version`
(default `v2`) selects which appears in the output; it is a **naming toggle and provenance stamp,
not a switch over the numerics**. `cell_eval2.compat` forces the full v1 standard (rank denominator
*n*, predicted control, keep NaN log-fold-changes) for byte-for-byte parity with upstream
`cell-eval`.

## The 2026 competition (`vcc2026`)

Six scored members, each rescaled so that **0 is the mean-response baseline and 1 is a replicate of
the reference measurement**:

| member | measures |
|---|---|
| `pds_cosine` | separability of predicted profiles |
| `expr_mse_unbiased_capped_norm` | size of the expression error |
| `de_wilcoxon_direction_fidelity_yield_raw` | whether predicted directions are right |
| `de_wilcoxon_direction_reach_raw` | how deep the directions stay right |
| `de_wilcoxon_sig_jaccard` | agreement of responding-gene sets |
| `de_wilcoxon_lfc_nmae` | accuracy of predicted fold changes |

**[`docs/vcc2026_metrics/`](docs/vcc2026_metrics/)** has the exact definitions, every parameter
value, the submission requirements, and the measured baselines and replicate anchors. The profile's
other four members are unscored expression diagnostics.

## Out-of-core scoring

For datasets too large to hold in memory, cell_eval2 scores directly from
[`cellstream`](https://github.com/ArcInstitute/cellstream) **cell-layout** archives (`.csad`, one
row group per perturbation) without ever materializing the full matrix.

Needs **`[scale]` and `[gpudge]`**, plus a CUDA device: it runs the shared GPU DE, so it refuses a
config without a gpudge-resolvable backend. It also requires `de.fdr_scope="per_pert"` — the
default — because a global FDR pool cannot be split across batches.

```python
from cell_eval2 import score_cellstream, MemBudget
from cell_eval2.config import EvalConfig

res = score_cellstream(
    "pred.csad", "real.csad",
    config=EvalConfig(metrics="minimal", device="cuda", pert_col="target_gene"),  # not "full"
    mem_budget=MemBudget(host_bytes=64 * 2**30, gpu_bytes=40 * 2**30),
)
print(res.overall)     # per_pert / per_context / overall frames
```

- Supports every metric of the `"full"` profile except the ones that need per-group moments,
  which its reference bundles do not carry: `expr_mse_unbiased`, `expr_mse_unbiased_capped`, `expr_distance_unbiased`,
  and `expr_mse_unbiased_capped_norm`, which is derived from two of them. A selection that
  resolves to any of them — including the default `"full"`, `"anndata"` and `"vcc2026"` — raises
  `NotImplementedError`. The selection must also include `de_wilcoxon_nsig_counts_real` and
  `de_wilcoxon_nsig_counts_pred`: the driver always reconstructs `de_wilcoxon_nsig_spearman` from
  them, so a selection without them (the `"vcc"` and `"pds"` profiles, or `["expr_mae"]`) raises
  `ValueError` after the whole DE run. `"de"` and `"minimal"` qualify. DE runs on gpudge only, so
  the `de_deseq2_*` metrics are unavailable.
- At matched settings, results equal in-memory `compute_metrics`: bit-exact on rank and DE
  metrics and within `1e-7` on continuous ones. One default differs: the `min_nonzero` AUC p-value
  floor does not split across batches, so this driver computes `preserve` instead (float64,
  floored only at the smallest positive float64). On gpudge's Float64 tables the two differ only
  in what an exact-zero predicted p-value becomes: `min_nonzero` gives it the table's smallest
  positive p-value, tying the two wherever they share a perturbation, and `preserve` the smallest
  positive float64 ([`docs/metrics.md`](docs/metrics.md) §4.4). Before 0.19.0 this driver used
  `replace_zero` at `1e-10`, which could rank a p-value of `1e-100` below `1e-20`; set
  `de.auc_pval_floor="replace_zero"` and `de.auc_pval_floor_value=1e-10` to reproduce those
  numbers.
- Handles counts and lognorm across the whole config matrix; `input_type` is auto-detected from
  the archive.
- Also scores the cell metrics (`cell_cov_ccc`, `cell_cov_corr`, `cell_cov_eig_ratio`), selected
  by name next to a selection the first bullet allows (no moments metric, both nsig-count
  metrics), in a separate pass after the partitioned loop (a cell-only selection belongs to
  `compute_metrics`). These need both sides to resolve to raw
  counts, and on a CUDA device `cell_cov_ccc` needs up to `21·G²` bytes (G = gene count) within
  `mem_budget.gpu_bytes`; both are checked before any DE runs. Two of the pass's own checks also
  run before any DE, when some perturbation has at least 50 cells on both sides: for
  `cell_cov_corr` and `cell_cov_eig_ratio`, the target-gene resolution over the whole panel (a
  construct-ID panel needs `target_gene_map`); and, with `validate_input` on and a version other
  than v1, the value check on the predicted cells the pass would sample for the first such
  perturbation. With those settings the pass still value-checks every block of cells it reads.
  With `allow_discrete`, an archive resolves to counts without reading a value; the value checks
  still run. The pass logs one `cell pass:` line with its wall time and GPU peaks.
- Materializes perturbation-complete batches under an explicit memory budget.
- The control label comes from a uniform `obs.control_value` column when the archive has one,
  otherwise from `EvalConfig.control` — so set it explicitly if your archive lacks that column.
- Pass `outdir=...` to also write the `ScoreResult` frames there (`per_pert.parquet`,
  `per_context.parquet`, `overall.parquet`) and, for v2, `evaluation_results.json`.
  Add **`write_de=True`** to additionally emit the raw
  gene-level DE tables `de_real.parquet` + `de_pred.parquet` — the unthresholded tables the DE
  metrics consume, the same artifact the in-memory path writes under `--write-degenes`, here
  written through a pinned zstd-3 writer with bounded memory and published together with the three
  score files and the evaluation document when present (rollback on error). Off by default;
  requires `outdir` and at least one DE metric
  in the selection. See [`docs/de_tables.md`](docs/de_tables.md) for the exact contract.

  ```python
  score_cellstream(
      "pred.csad", "real.csad",
      config=EvalConfig(metrics="de", device="cuda", pert_col="target_gene"),
      mem_budget=MemBudget(host_bytes=64 * 2**30, gpu_bytes=40 * 2**30),
      outdir="results/", write_de=True,   # -> results/de_{real,pred}.parquet
  )
  ```
- Pass **`ground_truth_cache=<root>`** to compute the real (ground-truth) DE table once and reuse it:
  later runs against the same real archive and real-DE settings — whatever the prediction — read
  it instead of recomputing it. The real archive must carry a `payload_sha256` checksum. With
  `write_de=True` the outputs gain `ground_truth_de_cache.json`, recording the lookup. See
  [`docs/ground_truth_cache.md`](docs/ground_truth_cache.md).

## Evaluation results (`evaluation_results.json`)

`cell-eval2 run` writes `evaluation_results.json` beside `agg_results.csv` when scoring a
prediction. `score_cellstream` writes it when given `outdir`: after the score frames normally,
or staged and published with the other outputs under `write_de=True` (rollback on error).
Runs with `version="v1"` or `de.backend="deseq2"` write no document; neither does the CLI's
real-data-only mode.

The document identifies each metric by its key and `version` within the `cell_eval2` suite,
and carries its headline statistic, display name and description, resolved `suite.settings`,
statistics and, for metrics with a perturbation axis, per-perturbation values.

`cell-eval2 run --settings-from evaluation_results.json` reruns a call from its document;
it requires `-ap`, is mutually exclusive with `--config` and `--preset`, and refuses `--profile`
because the metrics come from the document. Flags or `--set` may change only output, cache,
thread and `allow_fractional_counts` settings; any other change is refused. The run is also
refused if a recorded fact differs from what the replay resolves: perturbation column, control
label, effective input types, comparator, DE engine or scoring path. A streaming document is
refused by the CLI because its `scoring_path` differs; replay it through
`score_cellstream(config=settings_to_config(doc), check_facts=facts_checker(doc), ...)`, using
`settings_to_config` and `facts_checker` exported from `cell_eval2`.

## Caching

`cell_eval2` has two independent caches.

**The general artifact cache** (`CacheStore`). Point at two cache folders to reuse the real side
across many predictions:

```python
cfg = EvalConfig(metrics="de", cache_real="cache/real", cache_pred="cache/pred_A")
df = compute_metrics("pred_A.h5ad", "real.h5ad", config=cfg)
```

A cached value is used only when its fingerprint and parameters match, otherwise it is recomputed;
stale or corrupt entries self-heal. `cache_strict=True` content-hashes the matrix instead of using
the default metadata fingerprint. Passing **paths** rather than in-memory objects keeps peak memory
to one matrix at a time.

**The ground-truth DE cache** (`score_cellstream(..., ground_truth_cache=<root>)`, see
*Out-of-core scoring*). A content-addressed, immutable store of the unthresholded real-DE table,
keyed by the real archive's content checksum plus every setting that can change the table. Every
hit is re-validated, and a corrupt entry is fatal (`GroundTruthCacheCorruptError`) — never
recomputed over, because a false hit on ground truth is the one failure it must not produce.
`cache_real` plays no part in it. See [`docs/ground_truth_cache.md`](docs/ground_truth_cache.md).

## License

MIT — see [LICENSE](./LICENSE).
