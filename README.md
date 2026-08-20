# cell_eval2

A clean, streamlined, GPU-accelerated reimplementation of
[`cell-eval`](https://github.com/ArcInstitute/cell-eval) — Arc Institute's
suite for evaluating perturbation-prediction models at single-cell resolution.

> **New here?** See the [tutorial](docs/tutorial.md) for a guided, runnable tour of the
> usage patterns (metric selection, v1/v2 standards, caching, CPU/GPU, streaming, and more),
> and the [metrics reference](docs/metrics.md) for the math and formula of every metric.
>
> **Coming from `cell-eval`?**
> [v1 ↔ cell-eval parity divergences](docs/parity/v1-parity-divergences.md) lists every accepted
> behavioural difference between this suite's v1 mode and upstream cell-eval 0.6.6, and why each
> one is deliberately left alone.

## Goals

- **Clean & streamlined** — a focused, well-typed core with a minimal,
  predictable public API.
- **GPU-accelerated** — push the heavy numerics (pseudobulk aggregation,
  pairwise distances, differential expression, rank/overlap metrics) onto the
  GPU while keeping a CPU fallback.
- **API-compatible** — match the upstream `cell-eval` API and CLI as closely as
  practical so existing workflows port over with minimal friction.

## Install

Requires Python ≥ 3.11. The base install is **CPU-only** and pulls nothing private:

```bash
git clone https://github.com/ArcInstitute/cell-eval2.git
cd cell-eval2

python -m venv .venv && source .venv/bin/activate
pip install -e .
cell-eval2 --help
```

Releases are **git tags**, not PyPI packages — `git checkout <tag>` after cloning to pin one;
a bare clone tracks `main`. `pip install -e '.[gpu]'` adds the CUDA pseudobulk/rank kernels
(cupy + nvCOMP); the `gpudge` GPU **DE** backend is a separate extra, because it pulls `torch`
and most users of the kernels do not want it. Its torch build has to match **your** host's CUDA —
a default-index torch can be unusable on a driver where cupy still works — so install it with
`uv pip install --torch-backend=auto -e '.[gpudge]'`, where `auto` selects the build for the
detected driver. That flag needs a recent uv — verified on 0.7.12; an older one answers
`unexpected argument '--torch-backend' found`. pip has no equivalent, so name the index for your
own CUDA:
`pip install -e '.[gpudge]' --extra-index-url https://download.pytorch.org/whl/cu126` for a 12.6
driver, `.../whl/cu124` for 12.4, and so on. Both extras are opt-in and the CPU quickstart above
needs neither.

> ℹ️ **CI installs with `uv pip install -e . --group dev` and invokes `.venv/bin/…` directly,
> rather than `uv sync` / `uv run`.** This used to be a hard prohibition: `scale` pointed at a
> private Arc git repository, and locking the project *resolved* that URL and failed outright for
> anyone without access, even when they were not installing the extra. **That reason is gone** —
> `scale` is now the public `cellstream` on PyPI. What remains is a preference, not a ban:
> `uv sync` writes a lock and prunes the environment to match it, which is not what you want in a
> venv you are also installing GPU wheels into by hand. It does **not** pull the optional extras
> (`--extra` / `--all-extras` are required, and no default-extras are declared), so if you prefer
> `uv sync`, it works. Pass `--no-sync` if you want `uv run` without the sync.

Dev tooling (pytest, ruff, …) lives in a PEP 735 `[dependency-groups]`, not in a pip extra.
CI installs it with `uv pip install -e . --group dev` and then runs
`.venv/bin/pytest tests/ -q -ra`.

CI runs on CPU runners with base + dev only, so a large slice of the suite never executes
here — including anything needing a real CUDA device, a real `.shad` archive, the private DE
engines, or Arc-only validation artifacts. The in-repo pseudobulk/rank kernels and much of the
streaming control flow do still run, on their NumPy backend and against in-process fakes; the
CuPy kernels and the real `cellstream` / `gpudge` paths are covered out of band, on a GPU host,
before a release tag. `-ra` prints every skip and its reason, so the log says what never ran
rather than reporting it as a single number under a green badge.

## Status

The catalog (`cell_eval2.catalog.CATALOG`) carries **95 metrics, 87 of them
scored**, across four families: centroid error (`expr_*`), the **Perturbation
Discrimination Score** (`pds_l1`/`pds_l2`/`pds_cosine`), signed-effect geometry
(`delta_*`), and a large **differential-expression** family (`de_*` — 82
members: overlap/precision/recall, rank and LFC correlation, direction,
significance agreement, and chance-corrected variants), over the `wilcoxon` and
opt-in `deseq2` DE backends. Seven profiles select subsets — `full` (54), `de`
(41), `anndata` (13), `minimal` (8), `vcc` (3, the 2025 competition set),
`vcc2026` (10 — 6 scored, the 2026 competition set, plus 4 related expression
diagnostics, two of which are the derived metric's numerator and denominator)
and `pds` (1). Eight members are unscored
diagnostics: the four `de_*_nsig_counts_{real,pred}` entries and the four
per-perturbation expression diagnostics (`expr_mse_unbiased`,
`expr_mse_unbiased_capped`, `expr_distance_unbiased`, `expr_real_mass_ratio`).
Scoring policy (anchor, direction, penalty, clamps) is declared per-metric on
`Scoring`; see [`docs/metrics.md`](docs/metrics.md) for the math of every metric
and its enrolment — including §1.2's **comparator**, the space the 13 anndata
metrics are compared in, which issue #264 moved to a group-sum bulk for v2
counts runs.

Each metric carries a canonical **v2** name (the default) and an inherited
**v1** name. `EvalConfig.version` (`v1`/`v2`, default `v2`) and the `--version`
CLI flag select which names appear in the output — `version` is a **naming
toggle + provenance stamp**, not a master switch over the numerics. Two shipped
presets record each evaluation standard: native `EvalConfig()` is **v2** (rank
denominator n−1, real control, cosine, mask NaN-LFC); `cell_eval2.compat` forces
**v1** (rank denominator n, predicted control, keep NaN-LFC) and emits the
inherited v1 names, so VCC/State stay **byte-for-byte** bit-parity with
cell-eval. (`legacy()`/`corrected()` remain as aliases of `v1()`/`v2()`.) The v1
PDS and DE overlap/precision are differential-tested against upstream cell-eval.
Two metrics remain deferred — `edistance_pearson` and `clustering_agreement`
(`catalog.KNOWN_DEFERRED`, issue #23).

**Which PDS distance runs is chosen by the metric variant name**
(`pds_l1` / `pds_l2` / `pds_cosine`), not by the preset — e.g. the `vcc` profile
hardcodes `pds_l1`, so it runs l1 under either preset (the presets change the
rank denominator and control source, not the distance). The `v1`/`v2` `distance`
field records each preset's canonical distance and round-trips in YAML.

```python
from cell_eval2 import compute_metrics, EvalConfig

# Native defaults = v2 (canonical names; n−1 + real control). Expression / PDS
# metrics need only the (pred, real) AnnData pair:
df = compute_metrics(adata_pred, adata_real, metrics=["expr_mae", "pds_l1"],
                     pert_col="target", control="non-targeting")
# metric labels: expr_mae, pds_l1

# cell-eval / VCC drop-in: the v1 preset gives bit-parity numerics AND v1 names.
# DE-containing profiles (vcc/vcc2026/minimal/de/full) also require precomputed DE tables.
df_v1 = compute_metrics(adata_pred, adata_real, de_pred=de_pred, de_real=de_real,
                        config=EvalConfig.from_dict({**EvalConfig.v1().to_dict(),
                                                     "metrics": "vcc"}))
# metric labels: mae, discrimination_score_l1, overlap_at_N
```

## DE backends

The DE tables consumed by the DE metrics are produced by a selectable backend
(`de.backend`). The default is `auto`, which returns `gpudge` (GPU) when it is
available. When it is not, `auto` is **tiered rather than an unconditional
fallback**: on a host where a CUDA device is visible it **raises**, because
silently computing DE on a CPU engine there would change every DE number with no
signal; on a host with no CUDA device it falls back to `pdex` — or to `scanpy`,
which is substantially slower, if pdex is missing — warning once per process. All
three are rank-test (Wilcoxon) backends: `pdex` and `scanpy` supply p-values while
cell_eval2 computes the LFC, whereas `gpudge` computes both itself on-GPU. Set
`de.backend` explicitly (`"pdex"` / `"scanpy"`) to pin a CPU engine and silence
the warning.

**`deseq2` (opt-in, non-default).** A fourth backend computes DE with a
negative-binomial GLM via the `deseq2_gpu`
engine — an **Arc-internal package**, so this backend is opt-in and not installable outside
Arc; every other backend and metric works without it. Unlike the
rank backends, **deseq2 owns its own LFC and p-values** (like gpudge), so
`mean_calc`/`epsilon`/`clip_value` do not apply and `fdr_scope` is ignored (it uses
DESeq2's native per-contrast padj: Cook's flagging + independent filtering; a warning
is emitted if `fdr_scope` is non-default).

- **`auto` never selects `deseq2`** — every existing preset stays byte-identical; it
  runs only when `de.backend="deseq2"` is set explicitly.
- **Requires `input_type="counts"`** (raw counts; fails loud on `lognorm`).
- **Requires `de.replicate_col`** — the `.obs` column defining pseudobulk replicates
  (the NTC-guide column). Cells are summed per `(pert_col, replicate_col)` into
  pseudobulk samples: the control must have **≥2 replicate levels** (e.g. multiple NTC
  guides), while each perturbation may be a single sample (n=1). Each perturbation is
  tested against the replicated control (control = reference level).
- **`control_source`** works as elsewhere: `"real"` (v2 default) tests the predictions
  against the real NTC-guide control (predictions need no NTC guides of their own);
  `"pred"` requires the prediction to carry its own control cells.
- **`device`**: CPU (numpy) by default; set `device="cuda"` to use deseq2_gpu's batched
  JAX `fit_contrasts` GPU path (pseudobulk keeps the fitted matrix tiny, so CPU is
  adequate at cell_eval2 scale).

`.obs` layout (both sides), e.g. `target_gene` + `guide`:

| target_gene   | guide         |
|---------------|---------------|
| non-targeting | NTC_1 … NTC_k |
| GENE_A        | GENE_A_g1     |
| GENE_B        | GENE_B_g1     |

```python
from cell_eval2 import EvalConfig, compute_metrics
from cell_eval2.config import DEParams

cfg = EvalConfig(
    pert_col="target_gene", control="non-targeting", input_type="counts",
    control_source="real",
    de=DEParams(backend="deseq2", replicate_col="guide"),
)
compute_metrics("pred.h5ad", "real.h5ad", config=cfg)
```

CLI (nested `de.*` via the generic `--set` escape hatch):

```bash
cell-eval2 run -ap pred.h5ad -ar real.h5ad \
  --pert-col target_gene --control non-targeting --input-type counts \
  --set de.backend=deseq2 --set de.replicate_col=guide -o out/
```

## Caching (opt-in)

Point `compute_metrics` at two cache folders to skip recomputation on re-runs. The
real side is computed once and reused across many predictions:

```python
from cell_eval2 import EvalConfig, compute_metrics, precompute_cache

cfg = EvalConfig(metrics="vcc", cache_real="cache/real", cache_pred="cache/pred_A")

# (optional) build the real cache once, loading only the real side:
precompute_cache("real.h5ad", side="real", config=cfg)

# every prediction reuses the real artifacts; only the pred side is computed:
df = compute_metrics("pred_A.h5ad", "real.h5ad", config=cfg)
```

A folder holds fixed-name artifacts plus a `manifest.json`; a cached value is used
only when its fingerprint and params match, otherwise it is recomputed (corrupted or
stale entries self-heal). `cache_strict=True` content-hashes the matrix instead of the
default metadata-only fingerprint. Passing **paths** (not in-memory AnnData) keeps peak
memory to one full matrix at a time. CLI: `--cache-real/--cache-pred/--cache-strict` on
`run`, and `cell-eval2 prep-cache --side real --adata real.h5ad --cache-real cache/real`.

## Large-scale (`.shad`) scoring

For datasets too large to hold in memory, cell_eval2 has a CPU, memory-bounded
**streaming** path that reads packed `cellstream` `.shad` archives (the `SHPK` format)
shard-by-shard, computes per-perturbation pseudobulk without ever materializing the full
matrix, scores the **anndata** metrics against a shared real reference, and can split the
work across independent runs.

[`cellstream`](https://github.com/ArcInstitute/cellstream) is **public on PyPI**, so the `scale`
extra is an ordinary install. It is the successor to the Arc-internal `shardad` this path was
built against; the container format is unchanged (`SHPK`), so archives written by either package
read here.

Install the optional extra (`cellstream ≥ 0.9.1` — the code needs
`gather_rows_adata` and its `n_threads` keyword, both present from that release):

> ⚠️ Prebuilt wheels cover **CPython 3.11 and 3.12 only**. On 3.13 the resolver falls through to
> the source distribution and needs a Rust toolchain to build it, so prefer 3.11/3.12 — which is
> also exactly what CI tests.

```bash
uv pip install -e '.[scale]'      # or: pip install -e '.[scale]'
```

```python
from cell_eval2 import EvalConfig
from cell_eval2.scale import score_streaming

cfg = EvalConfig(metrics="anndata", pert_col="target_gene", control="non-targeting")

# score a predicted .shad against a real .shad (defaults are v2):
df = score_streaming("pred.shad", "real.shad", config=cfg)
```

**Partition + aggregate.** Split the perturbations into `fraction` subsets, score each
subset on its own node/run against the *same* real reference, write a partial, then
recombine:

```python
from cell_eval2.scale import score_streaming
from cell_eval2.partition import aggregate_partials

# on each worker i in [0, N): pred subset scored against the full real reference
score_streaming("pred.shad", "real.shad", config=cfg,
                fraction=N, index=i, partial_out="parts/")

# once all partials exist, recombine (refuses to mix different references/configs):
full, per_metric = aggregate_partials("parts/")
```

The real reference is streamed at most once and is cached when `cfg.cache_real` is set
(streaming artifacts are namespaced `stream_pseudobulk_*`, so they never collide with
`compute_metrics`' own cache in the same dir). Each partial carries a fingerprint of the
real archive and the config; `aggregate_partials` refuses to combine partials computed
against different references/configs or with overlapping `(perturbation, metric)` rows.

Those three steps — cache the reference once, score each shard independently, recombine — are
the whole streaming contract. There is no `cell-eval2` subcommand for them: drive them from the
Python API above.

DE metrics **are** supported on this path: they run through `gpudge`, which needs a CUDA
device, so a DE-free metric set is what runs on CPU. The remaining `NotImplementedError`s here
are about the *input type*, not the metric kind — `target_sum=None` on a non-counts archive.

### Cell-layout archives with full DE (`score_cellstream`)

For a pair of **cell-layout** `cellstream.cell` archives (`layout='cell'`, one row-group per
perturbation), `score_cellstream` scores out-of-core on the same `partition_inmem` engine as
in-memory `compute_metrics` — materializing perturbation-complete batches under a memory budget
and running the shared GPU DE (`gpudge`) + pseudobulk. Unlike the shard-layout `score_streaming`
above, it computes the **full metric set (including DE)** and handles **counts and lognorm** with
the entire config matrix (v1/v2, `control_source` real/pred, any `target_sum`), so it directly
scores `scaled_log1p` cell-stream archives. It is **GPU-only** (needs a gpudge backend +
`fdr_scope="per_pert"`) and single-context.

```python
from cell_eval2 import score_cellstream, MemBudget
from cell_eval2.config import EvalConfig

res = score_cellstream(
    "pred.shad", "real.shad",
    config=EvalConfig(device="cuda", pert_col="perturbation"),  # control from obs.control_value
    mem_budget=MemBudget(host_bytes=64 * 2**30, gpu_bytes=40 * 2**30),
)
print(res.overall)   # ScoreResult: per_pert / per_context / overall polars frames
```

Results are bit-exact (rank/DE) / within `1e-7` (continuous) of `compute_metrics` on the same
cells. `input_type` (counts vs lognorm) is auto-detected from the archive. The control label is
taken from a **uniform `obs.control_value`** column if the archive has one (forward-eval archives do);
otherwise it falls back to `EvalConfig.control` (default `"non-targeting"`), so set
`config=EvalConfig(..., control=...)` explicitly when your archive lacks that column.

## License

MIT — see [LICENSE](./LICENSE).
