# v1 ↔ cell-eval 0.6.6 parity divergences (accepted)

> **Decision (2026-06-27): these are ACCEPTED divergences, not bugs — do NOT "fix" them.**
>
> - cell_eval2's **v1** mode exists for *verification only* and is effectively **deprecating**.
> - v1 reproduces the VCC **v0_6_6** leaderboard scoring **bit-identically (Δ=0)** today
>   (see the project's VCC reproduction recipe).
> - Every divergence below is **latent** — unreachable on real perturbation data; it surfaces
>   only on adversarial / pathological inputs that the production VCC path never produces.
> - Aligning v1's *algorithm text* to the reference would therefore risk perturbing a
>   reproduction that already works, in exchange for edge-cases that never arise.
>
> Tracking issue: #53 &nbsp;•&nbsp; v2 (the kept version) is unaffected by these.

## Reference oracle

The reference is **cell-eval 0.6.6 + pdex 0.1.27** (the versions the bit-exact VCC v0_6_6
reproduction recipe pins; run from a local reproduction venv — not committed here).
`guess_is_lognorm` was verified **byte-identical** across cell-eval 0.6.6, 0.6.8, and 0.7.2,
so the divergences below are independent of which 0.6.x/0.7.x reference is used.

All `cell_eval2` line references are at the commit that introduced this doc; treat them as
"as of writing" pointers, not guarantees.

---

## The three numeric divergences

### 1. `guess_is_lognorm` — per-row-sum sample vs per-element scan

| | cell_eval2 (v1) | cell-eval 0.6.6 (`cell_eval/utils.py`) |
|---|---|---|
| What is tested | fractional part of the **per-cell row sum** | fractional part of **every element** (`np.modf(X.data)`) |
| Coverage | a **500-cell sample** | all stored values |
| `epsilon` | `1e-6` | `1e-3` |
| Range validation | none | `max_threshold=15.0` log1p-range check, gated by `validate=not allow_discrete` |

- cell_eval2: `src/cell_eval2/norm.py:92-111`
- **Why latent:** genuine log-normalized submissions have fractional row sums (sums of hundreds–thousands
  of fractional log values are essentially never integer), and genuine integer counts have integer row
  sums — so both criteria agree on real data. Misclassification needs an adversarial input whose every
  sampled cell's gene values cancel to a near-integer row total, which real screens don't produce.

### 2. input-type precedence — `allow_discrete` first vs lognorm-guess first

- cell_eval2 `resolve_input_type` (`src/cell_eval2/norm.py:114-129`) checks `allow_discrete` **before**
  the lognorm guess.
- cell-eval 0.6.6 `_convert_to_normlog` (`cell_eval/_evaluator.py:192`) **always** evaluates
  `guess_is_lognorm` first; `allow_discrete` only (a) sets `validate=not allow_discrete` inside the guess
  and (b) decides whether *integer* data is left discrete vs `normalize_total`+`log1p`'d.
- **Why latent:** the precedence only flips the outcome for the `allow_discrete=True` **and**
  genuinely-lognorm combination, which the VCC path does not hit (`allow_discrete=False`).

> **Not a divergence:** counts→`normalize_total` target. 0.6.6 normalizes counts to the **median**
> library size (`normalize_total` with no `target_sum`). cell_eval2 **v1 already matches** this —
> `_VERSION_CONVENTIONS["v1"]["target_sum"] = None` (`src/cell_eval2/config.py`). The `1e6`/CPM target
> is the **v2** default only.

### 3. Float32 DE ranking / significance — cev2 ranks in Float64, reference in Float32

- cell-eval 0.6.6 casts **all** DE numerics (`fold_change, p_value, fdr, log2_fold_change,
  abs_log2_fold_change`) to **Float32** in `DEResults.__post_init__` (`cell_eval/_types/_de.py:114-115`)
  before `get_top_genes` (ordinal rank), `filter_to_significant`/`get_significant_genes`, and the AUC
  label/score; pdex 0.1.27 computes DE from a **float32** matrix (`pdex/_single_cell.py:639,641`).
- cell_eval2 **v1** ranks (`src/cell_eval2/de.py:88-105`, `_rank_matrix`) and filters significance
  (`src/cell_eval2/metrics/de.py:96,221,259`) in **Float64**. The asymmetry: cev2 already does the
  `cast(Float32).cast(Float64)` round-trip on the AUC `replace_zero` path
  (`src/cell_eval2/metrics/de.py:281`) but **not** on ranking/significance.
- **Why latent:** the real side is loaded from a Float32-origin DE CSV, so it ranks identically; only the
  natively-computed pred side diverges, and only on **closely-spaced LFC ties** — which were Δ=0 on the 5
  VCC spot-checks. Tiny clean parity fixtures (values exact in Float32) never exercise it.

---

## Related known items (not addressed here)

- **`version`-knob decoupling (correctness/UX footgun, parity-independent).** `EvalConfig(version='v1')`
  constructed directly, or CLI `--version v1` layered on the default config, yields **v2 numerics under
  v1 labels** and skips the input-type validation, because `version` does not re-derive the numeric knobs
  (`src/cell_eval2/config.py` has no `__post_init__` reconciliation). The self-consistent v1 entry points
  are `EvalConfig.v1()` / `for_version` / the shipped `configs/v1.yaml`. Tracked separately.
- **compat `.describe()` NaN propagation.** The compat scoring path (`src/cell_eval2/compat/__init__.py`)
  shares the same latent NaN-propagation that the native aggregator had — but it is the deprecating v1
  surface, so it is documented rather than changed.

## v2 NaN-aggregation (fixed)

Separate from the v1 divergences above, the **v2 native** aggregator `aggregate_metrics`
(`src/cell_eval2/run.py`) previously used `pl.col('value').mean()`, which **propagates NaN** — one
degenerate perturbation (a single-class AUC pert emitting NaN at `metrics/de.py:296,301`, routine on real
screens) nulled out a whole metric's aggregate. This was **fixed** (NaN-skipping
`drop_nans().mean().fill_null(nan)`) in the PR that introduced this doc: skip NaN when valid perts exist,
keep NaN for an all-NaN metric so downstream `score_agg_metrics` arithmetic is unaffected. This matches
the reference's NaN-skipping `.describe()` semantics. v2 is the kept version, so this is a real fix, not
an accepted divergence.
