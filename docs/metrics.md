# cell_eval2 metrics reference

This document defines **every metric** in `cell_eval2`, with its mathematical formula,
range, and behavior. It complements the usage-oriented [`docs/tutorial.md`](tutorial.md):
the tutorial shows how to *run* the metrics; this document explains what they *compute*.

> **Math rendering.** Formulas use LaTeX and render natively on GitHub (and most Markdown
> viewers that support math). If you are reading the raw source, inline math is delimited by
> `$…$` and display math is written as `math` code fences.

- [1. Notation and shared preliminaries](#1-notation-and-shared-preliminaries)
- [2. Expression metrics (pseudobulk)](#2-expression-metrics-pseudobulk)
- [3. Perturbation Discrimination Score (PDS)](#3-perturbation-discrimination-score-pds)
- [4. Differential-expression (DE) metrics](#4-differential-expression-de-metrics)
- [5. Chance-corrected DE agreement metrics](#5-chance-corrected-de-agreement-metrics)
- [6. From per-metric means to a score](#6-from-per-metric-means-to-a-score)
- [6a. The generic-response baseline](#6a-the-generic-response-baseline)
- [6b. The data ceiling](#6b-the-data-ceiling)
- [7. Metric catalog summary](#7-metric-catalog-summary)

---

## 1. Notation and shared preliminaries

A run compares a **predicted** and a **real** `AnnData` over the same cells, genes, and
perturbation labels.

| symbol | meaning |
|---|---|
| $c$ | a cell; $x_c \in \mathbb{R}^{G}$ its expression vector over $G$ genes |
| $p$ | a perturbation label (in `obs[pert_col]`, default `target`) |
| $\mathrm{ctrl}$ | the control label (default `non-targeting`) |
| $\mathcal{C}_p$ | the set of cells with label $p$ |
| $P$ | the number of non-control perturbations |
| $\hat{\cdot}$ | a hatted quantity is from the **prediction**; unhatted is **real** |

With one exception, a metric is computed **per perturbation** and then averaged over
perturbations (the control is never scored); those aggregates are NaN-skipping means.
`expr_mse_unbiased_capped_norm` (§2.3) is derived only at panel aggregation time as a ratio of
sums and has no per-perturbation column.

### 1.1 Pseudobulk

Most metrics operate on **pseudobulk** profiles — the per-perturbation mean expression:

```math
\bar{x}_p = \frac{1}{|\mathcal{C}_p|}\sum_{c \in \mathcal{C}_p} x_c \in \mathbb{R}^{G}.
```

### 1.2 Normalization

Every per-cell step works in **log-normalized** space, and so did the expression metrics until
#264 moved them to the group-sum comparator below. (DE is the exception in the other
direction: its fold changes are computed on the normalized but *non-logged* values — §1.4.) Each cell is library-size normalized
to a common total $s$ and then $\log(1+\cdot)$ transformed:

```math
x_{c,g} \longmapsto \log\!\Big(1 + s\cdot \frac{x_{c,g}}{\sum_{g'} x_{c,g'}}\Big).
```

The target sum $s$ is the `target_sum` knob: **v2** uses $s = 10^{6}$ (counts-per-million);
**v1** uses $s = \text{median library size}$. DE fold changes are computed on the normalized
(non-logged) space — see §1.4.

#### The comparator: which space the expression metrics are computed in (v2, issue #264)

The per-cell recipe above averages $\log(1+\cdot)$ over cells, so the resulting "pseudobulk"
is a **dispersion functional**, not a mean: by Jensen's inequality it sits below the log of
the mean, by an amount set by how variable the cells are. Two submissions reporting the same
biology through differently dispersed cells land in different places, and no zero-dispersion
prediction can reproduce a real profile at all (#258, #260). Issue #264 replaces the space
the **13 anndata metrics** are compared in — every `expr_*`, `pds_*` and `delta_*` — with a
**group-sum** bulk:

```math
b_{p,g} = \log\!\Big(1 + \mathrm{TS}\cdot\frac{P_{p,g}}{\sum_{g'} P_{p,g'}}\Big),
\qquad P_{p,g}=\sum_{c\in\mathcal{C}_p} y_{c,g},
```

with $\mathrm{TS} =$ `bulk_target_sum` (default $5\times10^{4}$ as of #268; see §2.3 below). Summing
counts first and transforming once makes the statistic a function of the group's total, which
is what "the perturbation's expression profile" was always meant to mean.

**How big the property was, and that #264 removes it exactly (#258, #260, #261).** Under `lognorm`,
$\sum_g \operatorname{expm1}(b_{p,g})$ falls short of the target sum, and the shortfall *is* the
within-group dispersion — by Jensen, strictly short whenever the cells of a group differ in
*normalized* composition, and exactly zero only in the degenerate case where they do not. So a
zero-dispersion prediction is pinned to a shell no real group is
on, and no rescaling reaches it (scaling $x$ cancels out of $\mathrm{TS}\cdot x/\sum x$ entirely).
The deficit grows with `target_sum`, so the knob sets the size of the penalty on under-dispersed
submissions:

Each comparator is normalized by its **own** knob — `lognorm` by `target_sum`, `bulk_lognorm` by
`bulk_target_sum` — so the second column names a different field per row:

| comparator | its normalization knob | $\sum_g \operatorname{expm1}(b_{p,g})\,/\,\text{target}$ | deficit |
|---|---:|---:|---:|
| `lognorm` | `target_sum` $=10^{4}$ | 0.855 | 14.5% |
| `lognorm` | `target_sum` $=2.5\times10^{4}$ | 0.795 | 20.5% |
| `lognorm` | `target_sum` $=10^{5}$ | 0.701 | 29.9% |
| `lognorm` | `target_sum` $=10^{6}$ (the v2 default) | 0.572 | **42.8%** |
| **`bulk_lognorm`** | `bulk_target_sum`, any $\mathrm{TS}$ | **1.000000** | **0, algebraically** |

(The `lognorm` rows are the measurement from #261 on a 6-line, 400-cell/perturbation, 18,533-gene
panel; the deficit is smaller on shallower data — 4–5% on the 100-cell `docs/data` fixture — but the
`bulk_lognorm` row is exact for every panel, being $\sum_g \operatorname{expm1}\bigl(\log(1 +
\mathrm{TS}\,P_{p,g}/\sum_{g'} P_{p,g'})\bigr) = \sum_g \mathrm{TS}\,P_{p,g}/\sum_{g'} P_{p,g'} =
\mathrm{TS}$ identically — for every group carrying positive mass, and up to
floating-point roundoff. An all-zero group has no composition to express and
`prep.bulk_lognorm_means` returns zeros for it rather than raising, so its ratio is 0, not 1.)

The oracle test makes the consequence concrete: hand the metric each group's **own** real profile,
emitted with no dispersion. Under `lognorm` at the v2 `target_sum` that scores $\lVert
\text{tile}-\text{real}\rVert^2 = 5.6$ per perturbation on the `docs/data` fixture — a large error
for a perfect prediction. Under `bulk_lognorm` it scores $2\times10^{-27}$, i.e. machine zero. **The
range constraint #258/#260 describe is a property of the `lognorm` comparator, not of what v2 counts/counts
runs score.** `expr_real_mass_ratio` reports this ratio per perturbation on whatever data is actually
being scored (1.0 by construction under `bulk_lognorm`; the deficit under the fallback), so a
submitter never has to infer it.

⚠️ **That is a statement about $b_p$, not about every metric built on it (#278).** Take two
submissions with equal per-$(p,g)$ column sums — the same counts, redistributed across the cells of
each group. Their $b_p$ are algebraically identical, and bit-identical wherever the group-sum arrays
themselves are — and floating addition is not associative, so two *different cell-level layouts* with
the same mathematical total can still reduce to different values: in fp64 (and in fp32 after
widening) `[1e16, 1, 1]` and `[1e16, 2, 0]` differ by `2.0`, one fp64 ULP at that magnitude. What
#271 removed is narrower and is the part that was a defect: **the systematic narrow-versus-wide
mismatch** — the same matrix no longer reduces differently depending on the dtype it was stored in,
the two halves of one metric no longer disagree, and the resident driver no longer disagrees with the
streaming and GPU ones about the accumulation width. It does not make the pipeline
order-independent, and it does not touch input quantization or the GPU's fp32 output. ⚠️ **The
mismatch it did remove was real before #271** (fixed 2026-08-18):
`prep._grouped_sums` used to reduce in the *input* dtype while `moments.jackknife_correction` casts
to fp64 first, so on fp32 input the bulk and its own correction could come from different group
totals — and the resident driver could disagree with the streaming and GPU ones, which already
accumulate in fp64 (measured 3.53e-10 in bulk space). Every group-sum reduction now widens a
floating dtype coarser than fp64 before reducing, so all three drivers **accumulate a coarse float
in at least fp64** and the two halves of the metric see one $P_p$. Read that as the coarse-float
statement it is: output precision still differs (the GPU accumulator hands back fp32 means) and so
does summation order, and `longdouble` input still diverges between drivers **where longdouble is
wider than float64** (x86-64 Linux; not where the two are the same type) by design — fp64 is the
*narrower* side there, so the guard leaves it native. A **masked** matrix is a second residual: the
group sum honours the mask while `jackknife_correction`'s own `csr_matrix(X)` strips it.

⚠️ **The exactness boundary, recorded because it is the reason this mattered.** For
non-negative *integer* counts a float reduction is exact while the group sum stays under the dtype's
consecutive-integer limit $2^{\text{nmant}+1}$ — **2,048 for float16, $2^{24} = 16{,}777{,}216$
for float32, $2^{53}$ for float64** — and diverges by 1 count just past it (a 400-cell group at
~1e5 counts/gene; the bulk moves **3.53e-10** at the shipped `bulk_target_sum` of 5e4 — the
`6.35e-09` this was previously quoted as is the same fixture at the 1e6 target sum #268 retired,
so a bulk delta only means something alongside its TS). The three 138,400-cell official contexts measured here
sit $5.7$–$7.6\times$ inside float32's (an earlier 110,500-cell panel measured $11.4\times$). Validation bounds each *cell* (`max_counts_per_cell`,
integrality, non-negativity) and never the accumulated per-gene *group* total, so a large honest
submission can cross it.

⚠️ **Fractional input has no such guarantee at all: it can round from the first addition**, far below
that boundary. (Not *always* — exactly representable fractions reduce identically; what fractionality
removes is the guarantee.) So the inputs that CAN move are the fractional ones at any depth *plus* the
integer-valued floating ones past their own dtype's boundary; and the CONSERVATIVE regenerate set is both groups --
conservative because an exactly representable fraction reduces identically. What was MEASURED to move
on the shipped artifacts is the fractional group, because the **baseline arms are fractional** (a baseline is a mean, emitted as
float32). Measured on all three official contexts' arms **as stored** — 138,400 cells, 301 groups,
95.3–95.6% of stored values fractional — their group sums moved by up to **0.265 counts** and their
bulks by up to **5.7e-06** across this fix, at $5.7$–$7.6\times$ inside $2^{24}$
(A 0.2029/5.58e-06, B 0.2646/5.65e-06, C 0.0684/5.73e-06). ⚠️ **The REAL arm of those same three
contexts does not move at all** — 0 % of its stored values are fractional and its group sums are
bit-identical across the fix, at the same $2.2$–$2.9\times10^6$ magnitudes. The split is clean: the
integer real leg is untouched and it is the fractional baseline leg that moves. **Any stored baseline or bundle leg built
from a coarse-float arm whose group sums actually moved must be regenerated** — that is a FRACTIONAL
arm at any depth, or an integer-valued one whose per-gene group sums cross the dtype's exactness
boundary. (The
cache-invalidation terms are deliberately coarser than that — they key on the code path, since a key
is computed before any value is read — so an integer-count artifact below the boundary is merely
recomputed once, not moved.) Of the six scored `vcc2026` members, **two** read
these bulks — `pds_cosine` and `expr_mse_unbiased_capped_norm` — while the four `de_wilcoxon_*`
members read DE tables computed from cells and do not move. (Their `de_deseq2_*` siblings do, since
that backend pseudobulks through the same helper; it is opt-in and cannot form an enrolled bundle.) An earlier implementation was reverted for exactly that
reason and re-landed once the three #276 bundles it would have invalidated were already orphaned by
an unrelated rule change. `competition_digest()` does not move across this fix — the competition
*rule* is unchanged — so the rule digest does not separate the two eras. Within the library the
separation is carried by a code-semantics term in the pseudobulk, DE-table, anchor, partial and
result-cache keys (`run._GROUPED_SUM_REDUCTION_SEMANTICS`), which is what stops a pre-fix cached
bulk or score being served, and it makes a **MIXED** pair fail: a current submission against a
pre-fix anchor or reference bundle no longer validates, because the term is in the anchor's semantic
identity.

⚠️ **A FULLY pre-fix pair still scores, and that is by design, not by omission.** Anchor and
real-bundle validation is *peer-to-peer* — `expect_from_run_meta` reads the submission's own recorded
identity and `validate_anchor` compares the two artifacts with each other, never with current
semantics — so two artifacts both built before this change agree and pass. Inside the 0.13.0 window
`cell_eval2_version` separated nothing: both sides stamped `0.13.0`, so #271 joined #172, #320, #321
and #322 as a value-moving change that a same-version pair cannot see. ⚠️ **0.14.0 closes that
window going forward** — the stamp moves — and an ENROLLED competition bundle built at 0.13.0 is
refused either way, on its stale `rule_digest`. What survives below is the loose and diagnostic
paths. Three
consequences worth naming: loose `--baseline-agg` pairing is unguarded even for a mixed pair
(`baseline.config_digest` carries no metric-semantics term at all — the documented `#314` gap,
deferred by Alex on 2026-08-17, of which #271 is the **fourth** live instance and the first whose
moving object is the pseudobulk rather than a metric's definition); an **all-legacy** partial
directory with no semantics sidecar is accepted with a warning by #246's compatibility ruling; and a
pre-fix `lfc_nmae_ref` reference computed with the deseq2 backend is accepted (diagnostic
`from_reference` only — `avg_score` is computed before that column is added). 0.14.0 is that
release bump: regenerate rather than mix.

⚠️ **Regenerating "the baseline" is not enough — regenerate BOTH SIDES.** On the `--baseline-agg`
path the *submission's* own `agg_results.csv` + `run_meta.json` can be the pre-fix artifact; rebuild
the baseline, keep a moving pre-fix user aggregate, and `cli` accepts the pair without a word,
because `baseline.config_digest` carries no semantics term and both sides stamp the same version. A supplied
`--anchor`/`--anchor-cache` artifact is *not* the same case — a mixed-era anchor fails loudly on its
semantic identity, and only a fully pre-fix pair slips through — but the remedy is the same.

⚠️ **And regenerate any exported DE tables.** `--de-real`/`--de-pred` skip DE computation entirely, so
the semantics term never runs: it protects tables this build COMPUTES or serves from its own cache,
not a parquet you hand it. A pre-fix `de_pred.parquet` from the **deseq2** backend reused under
current code launders pre-#271 values into a freshly stamped `agg_results.csv`. Either omit those
flags or regenerate the tables first. Where its diagnostic column matters, the same holds for
`lfc_nmae_ref_agg.csv`. **If either side of a comparison may have moved, regenerate every artifact in
it.**

Which metrics inherit that invariance:

| | metrics | why |
|---|---|---|
| **carry it** | `expr_mae`, `expr_mse`, all `pds_*`, all `delta_*` | read the submission only through $b_p$ |
| **carry it** | `expr_distance_unbiased`, `expr_real_mass_ratio` | real-side only; do not read the submission at all |
| **do not carry it** | `expr_mse_unbiased`, `expr_mse_unbiased_capped`, `expr_mse_unbiased_capped_norm` | additionally subtract $C_p$ |

$C_p$ is a sampling-noise correction, and it reads the **within-group cell layout** that the group
total discards. That is by design, not a leak: a submission's cells are treated as exchangeable
samples of a predicted cell population exactly as the reference's are of a real one, so both
pseudobulk means carry sampling error and both are debiased.

"Do not carry the invariant" is not the same as "always differ" — two layouts can still score
identically, either because both exceed #247's cap (§2.3) or because the relayout leaves $C_p$
algebraically unchanged, as a permutation of a group's rows does (up to floating-point reduction
order, which `moments.py` documents as ULP-level). See §2.3 for what $C_p$ is and which direction
it moves the score.

Which space a run uses is a **run-level** decision, resolved once by
`norm.resolve_comparator` and stamped in the run metadata:

| version | inputs | comparator |
|---|---|---|
| v2 | counts on **both** sides | `bulk_lognorm` |
| v2 | any other combination — including counts/lognorm and lognorm/counts | `lognorm` (per-cell, above) |
| v1 | any | `lognorm` — v1 never moves; it reproduces upstream cell-eval |

Counts are unrecoverable from log-normalized input, so a lognorm side cannot supply
$P_{p,g}$; and an asymmetric run (counts real, lognorm pred — a supported path) falls back on
**both** sides rather than compare two different spaces silently. The fallback is
self-consistent rather than a patch: `lognorm` is a per-cell mean, and the analytic sampling
correction of §2.3 is exact for a per-cell mean. ⚠️ Numbers do not carry across the two
comparators — see §2.3 and §6a.

#### How far apart the two comparators are, on the member that lives in them (#288)

"Numbers do not carry across" is worth a size. The same *predicted expression*, submitted twice —
once as counts, once as that submission's own $\log(1+\mathrm{CPM})$ of those counts — is scored by
the two comparators as follows on `expr_mse_unbiased_capped_norm` (lower is better; `vcc2026` scores
it as $1-u$):

| per-gene multiplicative noise in the prediction | counts/counts → `bulk_lognorm` | lognorm/counts → `lognorm` | gap |
|---:|---:|---:|---:|
| 0.00 (the exact per-perturbation mean, tiled) | $-0.233$ | $1.135$ | $+1.368$ |
| 0.02 | $-0.105$ | $1.247$ | $+1.352$ |
| 0.05 | $0.543$ | $1.751$ | $+1.208$ |
| 0.10 | $3.004$ | $3.812$ | $+0.808$ |

The gap is **largest where the prediction is best**, because it is the concavity deficit above
(#258, #260) and not a measure of accuracy: under `lognorm` the real side's bulk sits below the log
of its mean by the within-group dispersion, while a tiled prediction's does not, and the two are
pushed apart by an amount that has nothing to do with the biology. The other five `vcc2026` members
are bit-identical across the same swap — the channel is specific to the pseudobulk-magnitude member.

⚠️ **Not reachable on the competition path.** Getting the pred side typed `lognorm` against a counts
reference requires `autodetect_input_type=True`; `configs/vcc2026.yaml` pins `input_type: counts` and
`autodetect_input_type: false`, and both are inside the frozen rule digest. This is a library
property, and the reason `resolve_comparator` now logs a **warning** when a v2 run falls back off
`bulk_lognorm` because of an asymmetric declaration, rather than falling back silently.

### 1.3 Effect (delta)

The **effect** of a perturbation is its pseudobulk minus a control pseudobulk:

```math
\delta_p = \bar{x}_p - \bar{x}_{\mathrm{ctrl}}.
```

The real side always subtracts the real control. For the prediction, `control_source`
selects which control is subtracted: `"pred"` uses the predicted control $\hat{\bar{x}}_{\mathrm{ctrl}}$
(within-realm; **v1** default), `"real"` uses the real control $\bar{x}_{\mathrm{ctrl}}$ (**v2** default):

```math
\hat{\delta}_p = \hat{\bar{x}}_p - \hat{\bar{x}}_{\mathrm{ctrl}}\ \text{(pred)}\qquad\text{or}\qquad \hat{\delta}_p = \hat{\bar{x}}_p - \bar{x}_{\mathrm{ctrl}}\ \text{(real)}.
```

### 1.4 The differential-expression (DE) table

DE metrics consume a per-perturbation Wilcoxon DE table with, for each gene $g$ and
perturbation $p$, a **log₂ fold change** and a **BH-adjusted p-value** $p^{\mathrm{adj}}_{p,g}$:

```math
\mathrm{lfc}_{p,g} = \log_2\!\frac{m^{\text{target}}_{p,g} + \epsilon}{m^{\text{ref}}_{p,g} + \epsilon},
```

where $m$ is the **arithmetic** mean (v2) or **geometric** mean (v1) of normalized expression,
and $\epsilon$ is a small floor ($10^{-9}$ in v2, $0$ in v1). A gene is **significant** for
perturbation $p$ when

```math
p^{\mathrm{adj}}_{p,g} < T \qquad (T = \texttt{p\_adj\_threshold},\ \text{default } 0.05).
```

Let $S_p = \lbrace g : p^{\mathrm{adj}}_{p,g} \lt T \rbrace$ denote the significant-gene **set** for $p$
(so $S_p^{\text{real}}$ and $S_p^{\text{pred}}$ are the real/pred sets). Where a metric needs a
**ranked** list, genes are ordered by `sort_by` (default $|\mathrm{lfc}|$, descending).

### 1.5 Versions, direction, and degenerate handling

- **v1 vs v2.** Every metric has a canonical **v2** name and, where an upstream `cell-eval`
  equivalent exists, a **v1** alias (used both as an accepted input spelling and as the v1
  output label). v1 reproduces upstream conventions (median `target_sum`, geometric LFC,
  `control_source="pred"`, no gene filter); v2 is Arc's native standard.
- **Scoring policy (`scoring`).** `direction` (↓/↑/·) is the metric's mathematics; `anchor` is
  where perfection sits on its own scale (blank = no constant anchor); `scored` is policy —
  whether it enters `avg_score`. Splitting them is what let enrolment stop being a property of
  the mathematics: the old `best_value` token could not say "higher is better" without also
  claiming an anchor. Every metric with a direction is scored today, but the rule is still
  `scored` ⇒ has a direction, **not an equivalence** — the policy can express
  "directional but not enrolled", and `expr_mse_unbiased_norm` occupied exactly that state
  before it was enrolled and later removed by #257. Eight diagnostics have no direction and
  cannot be scored: the four `de_*_nsig_counts_*` entries and the four per-perturbation
  expression diagnostics in §2.3 (the derived metric's two components, its uncapped audit
  sibling, and `expr_real_mass_ratio`).
- **No droppable NaN (v2).** A degenerate perturbation (e.g. no significant genes, a
  single-class label, a zero-variance correlation) must never silently vanish from the mean.
  In **v2**, such cases map to the metric's **worst** value so they count as a penalty; **v1**
  keeps the upstream omit/NaN behavior for parity. The worst value is noted per metric below.

---

## 2. Expression metrics (pseudobulk)

These compare the predicted and real pseudobulk per perturbation, over all $G$ genes, in
whichever space §1.2's comparator resolved to — the **group-sum** `bulk_lognorm` bulk on a v2
counts run, the per-cell **log-normalized** mean otherwise. `kind = anndata`.

### 2.1 `expr_mae` (v1: `mae`)

Mean absolute error between predicted and real pseudobulk:

```math
\mathrm{expr\_mae}_p = \frac{1}{G}\sum_{g=1}^{G} \big| \hat{\bar{x}}_{p,g} - \bar{x}_{p,g} \big|.
```

Range $[0,\infty)$; **best $=0$**. A core scored metric (`vcc`, `minimal`, `full`).

### 2.2 `expr_mse` (v1: `mse`)

Mean squared error between predicted and real pseudobulk:

```math
\mathrm{expr\_mse}_p = \frac{1}{G}\sum_{g=1}^{G} \big( \hat{\bar{x}}_{p,g} - \bar{x}_{p,g} \big)^2.
```

Range $[0,\infty)$; **best $=0$**.

### 2.3 Unbiased expression diagnostics and derived ratio (v2 only)

Issue #257 replaced `expr_mse_unbiased_norm` with three auditable per-perturbation components
and one derived panel metric. Let $\hat\mu_p$ and $\mu_p$ be predicted and real pseudobulk in
the run's comparator space (§1.2), and let $C_p$ be the **estimated sampling contribution** of
that group — the part of a squared distance that would be there even if the two sides agreed
perfectly in expectation. Every squared distance below is summed over the **scored gene set**
$\mathcal{G}_p$ and then divided by $|\mathcal{G}_p|$.

```math
\mathrm{expr\_mse\_unbiased}_p = \left(\lVert\hat\mu_p-\mu_p\rVert^2_{\mathcal{G}_p} - C^{\mathrm{pred}}_p - C^{\mathrm{real}}_p\right) / |\mathcal{G}_p|
```

```math
\mathrm{expr\_mse\_unbiased\_capped}_p = \left(\lVert\hat\mu_p-\mu_p\rVert^2_{\mathcal{G}_p} - r\,\min\!\left(C^{\mathrm{pred}}_p,\;k\,C^{\mathrm{real}}_p\right) - C^{\mathrm{real}}_p\right) / |\mathcal{G}_p|,\qquad k=\texttt{PRED\_TRACE\_CAP\_K}=1,\quad r=\min\!\left(1,\;\frac{B^{\mathrm{pred}}}{\sum_q w_q\min(C^{\mathrm{pred}}_q,\,k\,C^{\mathrm{real}}_q)}\right),\quad w_q=1/|\mathcal{G}_q|
```

```math
\mathrm{expr\_distance\_unbiased}_p = \left(\lVert\mu_p-\mu_{\mathrm{ctrl}}\rVert^2_{\mathcal{G}_p} - C^{\mathrm{real}}_p - C^{\mathrm{real}}_{\mathrm{ctrl}}\right) / |\mathcal{G}_p|
```

**The scored gene set excludes the perturbed gene's own transcript**
(#172, ruled 2026-08-17):
$\mathcal{G}_p = \mathcal{G}\setminus\{g_p\}$ where $g_p$ is the gene perturbation $p$ targets,
resolved through the same `target_gene_map`-aware lookup `pds_*` and the eleven chance-corrected
DE metrics use (`distances.resolve_exclusion_columns`), so one run has one notion of "the target
gene". A perturbation whose label does not resolve keeps the whole panel, and a panel where
**nothing** resolves raises rather than silently excluding nothing — that is #248's failure mode,
and both legs carry the same gate. See §4 for the DE half of the same ruling.

⚠️ **The corrections $C$ are subtracted whole — this is the one approximation in the change, and
it is measured.** $C_p$ *is* exactly decomposable over genes (under `bulk_lognorm` it is
$\frac{n-1}{n}\sum_g q_g$, verified additive to $2\times10^{-16}$), but the cached moments
artifact stores only the scalar sum, so $q_{g_p}$ cannot be recovered from it; getting it exactly
needs either a target-aware moments artifact or the full $[P,G]$ matrix. Measured on the real side
of all three official val contexts, against the same cached moments the official bundles were
built from: the target gene's share of its own $C_p$ is a median **0.0055–0.0083%** against
$1/G = 0.0054\%$ — an *ordinary* gene for the variance, even though it is the single
largest-moving gene for the *distance* in 57–66% of perturbations. Leaving $C$ whole therefore
understates $\sum_p \mathrm{den}_p$ by **0.0125% / 0.0108% / 0.0070%** on contexts A / B / C,
against the **10.21% / 11.30% / 6.07%** of the metric's range the exclusion itself removes.

The **1.0 anchor holds exactly**: a submission emitting the real control's own cells has
$\hat\mu_p = \mu_{\mathrm{ctrl}}$ and $C^{\mathrm{pred}}_p = C^{\mathrm{real}}_{\mathrm{ctrl}}$,
so numerator and denominator carry the same correction terms and cancel wherever the #247 cap
does not bind — the same condition as before #172.

⚠️ **Since #348 that condition has a second half** — and so does the *perfection* fixed point at 0:
a truth-matched SAMPLED prediction on a binding panel retains $(1-r)\,C^{\mathrm{pred}}$ in
expectation, so "expected numerator 0" is conditional too. Both are exact where the panel does not
bind, which is where an accurate submission sits (the replicate anchor's budget is
$1.54/1.38/1.58\times$ its claim on the official val A/B/C panels, measured at the anchor's own
half depth — `anchor._score_one_split` scores one half against the other, so its $C^{\mathrm{real}}$
is a half's too), but that is a property of the panel rather than a guarantee — it is the
second residual of §5.1 seen from the honest side. For the no-skill point the cancellation also
needs $r = 1$, i.e. the panel's claimed correction
$\sum_q w_q\min(C^{\mathrm{pred}}_q, k\,C^{\mathrm{real}}_q)$ must not exceed its own
across-perturbation budget $B^{\mathrm{pred}}$. An arm emitting an *independent* draw per
perturbation is close to that boundary and pays only the residual (measured on official val A: the
honest control-paste arm's budget is $0.974\times$ its claim, so it pays 2.6% of the member's
range — there is deliberately no tolerance factor, because any slack above 1 is a **rebate** a
submitter can farm, measured at $+10.37\%$ per unit injected for a $1.1\times$ variance ceiling).
An arm reusing **one** emitted control cell block for every perturbation pays everything: its
across-perturbation budget is exactly zero, so the whole correction is withheld and it reads
$1 + C^{\mathrm{pred}}/\sum_p\mathrm{den}_p$. That arm's correction is real but perfectly
*common-mode*, and common-mode error is not identifiable from **bias** in a single submission — no
estimator can credit one without crediting the other, which is why #348 credits neither. It costs
no reachable score: such an arm already reads at or above the baseline end of the scale (member
$\approx 1.0$ against a measured baseline of 0.9858, lower being better), so its `from_baseline`
is 0 before the change and 0 after.

⚠️ **A lever does open, and it is bounded rather than absent.** Subtracting $C^{\mathrm{pred}}_p$
whole while the target gene's plug-in error has been removed means a submission is still
credited for variance it generates in a coordinate that no longer costs it anything. Before
#172, scattering counts in that column also moved $\hat\mu_{p,g_p}$ and so raised the error the
numerator charged. The counterexample reproduces: on a **two-gene** panel, a prediction
differing from the no-skill arm only in its target column reads $1.2756$ before and
$\mathbf{-0.5228}$ after.

What contains it is that the exploit's two requirements pull against each other. Concentrating
$q_{g_p}^{\mathrm{pred}}$ on the target gene means scattering that column, which drives
$C^{\mathrm{pred}}_p$ past $k\,C^{\mathrm{real}}_p$ — where the #247 cap pins the subtracted term
and further concentration buys nothing at all. Staying *under* the cap instead requires being
under-dispersed off-target, which this metric already punishes hard (§2.3's `mse_unbiased_capped`
notes, #278). Measured on a $G=4{,}000$ / 400-cell panel calibrated to the official contexts on
both ratios that scale the effect — target gene 4.6% of $\sum D$ (val 3.2–5.5%) and corrections
87% of $\sum D$ (val 46–55%) — across adversaries spanning the whole trade-off, including fully
flattened off-target cells carrying 100% of $q^{\mathrm{pred}}$ on the target gene: the best any
of them does against the gene-corrected estimator is **0.013 of the range, with the sign against
the adversary** (this form reads *worse* for it, 6.1296 vs 6.1167). The two-gene blow-up needs
one gene to be half the library, which no real panel is.

The trade is therefore a bounded, measured, wrong-signed residual against a **target-aware
moments artifact** — $O(P)$ extra floats, but accumulation would need `target_gene_map`, and
every warm cache plus all three jackknife kernels (resident, streaming, GPU) would move. If that
artifact is ever built, exclude the gene from all three corrections and apply the #247 cap
*after* the exclusion.

All three columns are **unscored diagnostics** in gene-averaged expression units. Those units
are panel-dependent, are not comparable across datasets, and — since #264 — are not comparable
across comparators either: a value in `bulk_lognorm` space cannot be read against a published
pre-#264 `lognorm` one.

- `expr_mse_unbiased` is the pre-#247 estimator, restored bit-for-bit **over the whole panel**;
  since #172 it shares the capped sibling's gene set, so the bit-identity is reachable by asking
  for the whole panel (`genes=None`) rather than by default. It subtracts the prediction's full
  sampling correction and exists both to restore that diagnostic and to make the capped numerator
  auditable — and an audit over a different gene set audits nothing, which is why it excludes too.
  It must not be scored: a submitter can report the same mean through more dispersed cells,
  enlarge the subtracted term, and lower the value for free.
- `expr_mse_unbiased_capped` is the numerator used by the scored metric, and its correction is
  bounded twice. The #247 cap says a submission may never claim a larger sampling correction than
  the reference itself earns. **#348 adds a second bound, on the panel TOTAL: nor a larger one in
  aggregate than its own across-perturbation spread allows.** The budget is
  $B^{\mathrm{pred}} = \sum_g \sum_{p \in R_g} w_p (\hat\mu_{p,g} - \bar{\hat\mu}^{w}_{\cdot,g})^2$
  over the whole predicted panel wherever a driver supplies it (which is what makes the value
  independent of how a run is partitioned) and over the perturbations the call scores otherwise,
  with $w_p = 1/|\mathcal{G}_p|$, $R_g$ the rows that *score* gene $g$
  and the centring weighted over that same set. Each row's own target gene is left out — the same
  coordinate #172 drops from the distance, and for the same reason: that cell is free, so a budget
  reading it could be bought at no cost. The claim it is compared against carries the same weights,
  so both sides are in the units the metric reports; in raw gene units a panel with *partial* target
  resolution could arbitrage the two divisors. Writing
  $\hat\mu_p = \mu_p^{\mathrm{true}} + \varepsilon_p$ with $\varepsilon$ independent across
  perturbations, $\mathbb{E}[B_g] = B_g(\mu^{\mathrm{true}}) + \sum_{p \in R_g} w_p\sigma_{pg}^2 -
  (\sum_{p \in R_g} w_p^2\sigma_{pg}^2)/(\sum_{p \in R_g} w_p)$, so it *tracks* the correction the
  submission is owed while sitting deliberately **below** it — at equal weights, by $1/n_g$, i.e.
  0.33% at 300 perturbations. ⚠️ It is a **conservative budget, not an unbiased upper bound**, and
  the two cannot both be had: the missing degree-of-freedom factor is exactly the rebate a variance
  form pays out. When the claim exceeds it, **every row is scaled by the same
  $r < 1$**: proportional, because predicted cells per perturbation are not constrained by the
  rules and a single per-row ceiling would clip the high-variance rows first. A centred *sum* of
  squares and no multiplier, both deliberately — it is exactly the quantity the numerator charges
  for the same variation, so buying budget is break-even at worst, whereas a per-row variance
  ceiling ($\mathrm{ddof}=1$, times $P$ rows) or any tolerance factor pays the submitter a rebate.
  ⚠️ It does **not** close the channel for a submission whose *genuine* spread already exceeds its
  claim: signal cannot authenticate sampling error, so that arm is in the non-binding regime and can
  still pin its aggregates. #348 narrows the exploit to near-flat submissions — where it was
  measured, and where the live one sat — rather than removing the class. Only the correction is
  bounded; the diagnostic value remains signed.
- `expr_distance_unbiased` is the denominator: a sampling-corrected squared distance from each
  real perturbation to the real control — **unbiased** where $C$ is (the analytic `lognorm`
  branch), and only bias-corrected under `bulk_lognorm`, whose jackknife is itself measured
  0.32% high at the shipped $\mathrm{TS}=5\times10^{4}$ (2.06% at the retired $10^{6}$). It reads the real side only, so it is
  **submission-independent** and identical for every submission on a panel. It is routinely
  negative, and that is correct: measured negative rates are 7.7% on `CCL_2`, 2.2% on
  `H1_CGS`, and 0% on VCC Test (⚠️ pre-#264 `lognorm` rates; how often this crosses zero is set
  by how large $C$ is against the real effect, and #268 measures only 25% of the denominator
  surviving its own correction at $\mathrm{TS}=10^6$). A negative row says that this isotropic
  estimator cannot resolve the mean shift at that depth; it does not establish that the
  perturbation is null, and rows must not be filtered on this diagnostic. **#172's exclusion
  enlarges that population**, because the gene it removes is the largest-moving one: on the
  official val contexts it removes 5.5% / 5.0% / 3.2% of the raw summed distance and takes the
  count of negative rows from 0 / 0 / 1 to 0 / 3 / 8 of 300. The derived metric aggregates as
  `ratio_of_sums`, so a negative row is a smaller contribution to one sum rather than a
  divide-by-near-zero.

#### What $C_p$ is, and what it costs (issue #264)

$C_p$ follows the comparator, and only the comparator: the metric formulas above, the #247
cap, the $n<2 \Rightarrow 0$ policy and the derived ratio are identical either way.

- Under **`lognorm`** (v1, and any v2 run that is not counts-on-both-sides) the pseudobulk is
  a per-cell mean, so its sampling contribution is the analytic
  $C_p = \operatorname{tr}\hat\Sigma_p / n_p$ — one pass, $O(\mathrm{nnz})$, essentially free.
- Under **`bulk_lognorm`** it is not. $b_p$ is a nonlinear function of the group *sum*, and
  dropping one cell moves the denominator $\sum_{g'}P_{p,g'}$ and therefore **every** gene —
  including genes where that cell had no counts. So $C_p$ is a **delete-1 jackknife**, with
  $r_i = S_p - \mathrm{lib}_i$ and $v_{ig} = \log(1 + \mathrm{TS}(P_{p,g}-y_{ig})/r_i)$:

  ```math
  C_p = \frac{n-1}{n}\sum_g\sum_i \left(v_{ig}-\bar v_g\right)^2 .
  ```

  This is **two-pass and $O(n\cdot G)$ dense**, where every other moment is $O(\mathrm{nnz})$.
  Benchmarked at 500 cells/construct and 18,533 genes: **~222 ms per group, ~67 s per
  300-construct archive** — roughly doubling an expression-only scoring run, and negligible
  against the DE family. Cells are visited in blocks, and the measured peak Python allocation
  at that shape is **349 MiB** at the default block of 512 rows (89 MiB at 32 rows, which was
  also faster on the box measured; the block size does not change the answer — measured
  bit-identical on most shapes and 1 ULP apart at worst, three orders inside the `rtol=1e-13`
  the chunk-invariance test asserts). The stored artifact does **not** grow: $C_p$ is one
  scalar per group, so a moments bundle stays $O(P)$.

⚠️ **$C_p$ is biased upward under `bulk_lognorm`, by an amount `bulk_target_sum` controls.**
Measured by the split-half identity $\mathbb{E}\lVert b_A-b_B\rVert^2 = C_A + C_B$ on a
6-line, 400-cell/construct, ~20k-UMI, 18,533-gene panel: **0.19%** at $\mathrm{TS}=2\times10^3$
rising to **2.06%** at $10^{6}$. The shipped default is $5\times10^{4}$, which reads **0.32%**
(~0.07% net of a +0.25% complementary-subset artifact) and so meets the ~0.1% the estimator
needs. $10^{6}$ was the shipped value until **#268** and is the one point on the sweep where
the metric breaks: the split-half ceiling goes negative on 6 of 6 lines, the "predict the
control" anchor reads 1.073 instead of 1.0, and only ~25% of the denominator survives its own
correction, amplifying every error ~4x.

#### `expr_real_mass_ratio` — reading the comparator off the data being scored

One more unscored diagnostic, real side only (#264, and #260/#261 which asked for it):

```math
\mathrm{expr\_real\_mass\_ratio}_p = \frac{\sum_g \left(e^{\mu_{p,g}}-1\right)}{\text{mass target}},
\qquad \text{mass target} = \begin{cases}\texttt{bulk\_target\_sum} & \texttt{bulk\_lognorm}\\ \texttt{target\_sum} & \texttt{lognorm}\end{cases}
```

It inverts the comparator's own transform and asks how much of the target mass the profile
actually carries. Under `bulk_lognorm` it is **1.0 by construction** for any group with
positive mass — which makes it a tripwire — and **0** for an all-zero group, which
`bulk_lognorm_means` maps to a zero bulk by policy rather than to NaN. Under the `lognorm`
fallback it is the **concavity deficit** that makes that space a dispersion functional rather
than a mean: measured **0.8199** on a 200-construct control group at a target of 28,118, i.e.
18% of the mass lost to Jensen's inequality. It is `NaN` only when the per-cell target is
unresolvable — `target_sum=null` on lognorm-effective input, where there is no library-size
median to take; an explicitly configured numeric `target_sum` resolves and is used — because
a ratio against a guessed denominator is worse than no number.

The scored metric is derived after the per-perturbation work:

```math
\mathrm{expr\_mse\_unbiased\_capped\_norm} = \frac{\sum_p \mathrm{expr\_mse\_unbiased\_capped}_p}{\sum_p \mathrm{expr\_distance\_unbiased}_p}.
```

`expr_mse_unbiased_capped_norm` has **no per-perturbation column at all**. It exists only in
the aggregate frames. In the wide aggregate frame only `mean` is populated; `count`,
`null_count`, `std`, `min`, `max`, and `median` are `NaN`. Negative component rows
remain in both sums. A non-positive panel sum
$\sum_p\mathrm{expr\_distance\_unbiased}_p$ raises because the panel has no measurable
aggregate signal.

This ratio of sums weights perturbations by estimated real effect size. Equivalently, it is a
weighted mean of the unreported per-perturbation ratios with
$\mathrm{expr\_distance\_unbiased}_p$ as the weight. Emitting those ratios would be misleading:
their ordinary mean would not equal the shipped panel score, while any subset score can be
re-derived correctly from the two component columns. The statistic is a ratio estimator, so it
is consistent rather than exactly unbiased at finite sample size.

The population anchors require no fitted constants:

- A no-skill prediction emitting the control has expected numerator equal to expected
  denominator, so the no-skill point is **1.0 whatever the reference panel's depth**.
- A perfect prediction has expected numerator $0$, so perfection is anchored at **0.0**.
- The first statement is not invariance to the submission. The 1.0 point holds while
  $C^{\mathrm{pred}}_p \le k\,C^{\mathrm{real}}_p$, i.e. wherever #247's
  cap does not bind. A submission that breaches it — **fewer cells or more dispersed ones** —
  reads above 1, because the cap refuses a correction the reference does not earn. Measured
  no-skill on the synthetic anchor panel at $n^{\mathrm{real}}=200/500/2000$: 8.11/3.79/1.68
  at a tenth of the depth, and 3.34/1.91/1.22 at the same depth with twice the dispersion.
  That is the cap working as intended. ⚠️ Those levels are pre-#264 `lognorm` measurements.
- ⚠️ **The bullet above is a statement in expectation over an emission model, not about a single
  submission (#278).** It reads the cap condition off
  $\mathbb{E}[\cdot] = \lVert\Delta\rVert^2 + \max(0, C^{\mathrm{pred}}_p - k\,C^{\mathrm{real}}_p)$,
  where the $C$ there are *true* variance traces. That decomposition is general — it needs
  independence, not linearity, so it survives the move to a nonlinear bulk statistic. It is an
  oracle expression: what the shipped metric subtracts is the **estimate**
  $\min(\hat C^{\mathrm{pred}}, k\,\hat C^{\mathrm{real}})$, whose expectation is not the $\min$ of
  the expectations. Estimate and estimand come apart pointwise:
  - **Dispersion.** Under `bulk_lognorm`, $b_p$ reads the group *total* only. Hold every
    per-$(p,g)$ sum fixed and redistribute counts across the group's cells: the realized
    $\lVert\hat\mu-\mu\rVert^2$ cannot change, while the delete-1 jackknife
    $\hat C^{\mathrm{pred}}_p$ **can** (a permutation of a group's rows does not move it; the
    layouts in #278 do). Where it does, the reported metric value moves entirely through the
    subtraction, and is **non-increasing** in
    $\min(\hat C^{\mathrm{pred}}_p, k\,\hat C^{\mathrm{real}}_p)$ — strictly decreasing below the cap,
    exactly constant at or above it. Lower is better, so an **under-dispersed** submission forfeits
    the correction and reads **worse**, the opposite sign to the bullet above, with no gradient past
    saturation. What is specific to `bulk_lognorm` is that fixed raw group totals *automatically*
    fix $b_p$; under `lognorm` fixed totals did not guarantee a fixed per-cell-mean pseudobulk.
    **The under-dispersion penalty is intended** (Alex, 2026-08-15): a group of bit-identical cells
    is not a cell population, and the metric does not grant it a correction it has not earned.
  - **Depth.** "Fewer cells" is a tendency under a matched i.i.d. emission model, not a property of
    the realized cell count: the binding condition is $\hat C^{\mathrm{pred}}_p > k\,\hat C^{\mathrm{real}}_p$,
    and cell count alone does not determine $\hat C^{\mathrm{pred}}_p$. A single cell carrying the
    whole group total scores identically to many proportional cells carrying it, and the
    $n<2 \Rightarrow 0$ policy hands that submission $\hat C^{\mathrm{pred}}_p = 0$ — no correction
    rather than a capped one.
  - **What was open, and is now bounded.** $\hat C^{\mathrm{pred}}$ is an *empirical* estimator
    that assumes the group's cells are exchangeable draws, and nothing enforces that: a degenerate
    layout inflates the estimate without inflating the realized error. That was **#294**, parked
    for after the competition on the reading that #247's cap bounded the damage — #278 measured
    $\hat C^{\mathrm{pred}}/G$ of 0.0168 and 45.02 giving bit-identical scores (at the retired
    $\mathrm{TS}=10^{6}$, so treat those levels accordingly), and the absence of a gradient past
    saturation was taken as the bound.
    ⚠️ **#348 measured what that reading missed**: a flat deduction is harmless only if the
    plug-in distance *contains* comparable own-noise. Saturating the cap and pinning the aggregate
    are independent moves, and doing both makes the deduction pure gain — and since the derived
    member's denominator is fixed by the real data, a constant gain is a fixed *fraction* of its
    range. Measured on the official `-r2` val panels: pinning the per-(p, g) sums of an honest
    control-paste arm and changing nothing else moved `expr_mse_unbiased_capped_norm` from 0.0000
    to **0.9031** `from_baseline`, and a dev-leaderboard submission took **+0.1389 of a 0.2295
    OVERALL** through it. Hence the factor $r$ in the numerator above. #294's own preferred direction
    — estimate $C^{\mathrm{pred}}$ from a sampling model at the submitted depth rather than from
    the jackknife — does not close it: a pinned arm needs no manipulated estimate, since it submits
    the reference's own cell count at the reference's own depth and a nominal estimate lands where
    the cap already is.
- Both anchors are statements about $\mathbb{E}[\hat C]$, so they inherit whatever bias $\hat C$
  carries. Under `bulk_lognorm` at the shipped $\mathrm{TS}=5\times10^{4}$ that bias is the measured
  0.32% above. At the retired $10^{6}$ it was 2.06% — enough to put a split-half *replicate*
  below 0, i.e. scoring better than perfect (#268), which is why the default moved. The
  anchors are exact for the **oracle** form in the true variance traces, not for the shipped
  estimator. Two gaps, not one: the jackknife's own bias above, and — for the **capped** member —
  the clip, since $\mathbb{E}[\min(\hat C^{\mathrm{pred}}, k\,\hat C^{\mathrm{real}})] \neq
  \min(\mathbb{E}\hat C^{\mathrm{pred}}, k\,\mathbb{E}\hat C^{\mathrm{real}})$, so even
  individually unbiased corrections would leave the capped anchor inexact.

Historically, `expr_mse_unbiased_norm` debiased its numerator on both sides but divided each
perturbation by a plug-in real-effect denominator. A no-skill submission therefore read
$1-\mathrm{noise}_p/D_p$, not 1.0: measured panel aggregates were 0.7643 on VCC Test, 0.2386
on `CCL_2`, and 0.2754 on `H1_CGS`. Because that gap was set by reference depth,
`low-random_high-1_v1` credited a do-nothing submission 0.24–0.76 of the full random-to-paste
range. `expr_mse_unbiased_norm` is removed with no alias; a historical column cannot bind to
the new definition.

`expr_mse_unbiased_capped_norm` is scored in `full`, `anndata`, and `vcc2026` with
`Scoring(scored=True, direction="lower", anchor=0.0, penalty="boxcox", clamp_low=0.0, clamp_high=1.0)`.
The value is signed, so `clamp_high=1.0` remains load-bearing; it clamps the score, never the
reported aggregate value. (`clamp_low` became `0.0` in #276 part C — this is the one scored
`vcc2026` member clamped on **both** ends; the other five keep `clamp_high=None`. Not to be
confused with the frozen `_v9` scale's `−6.0` floor, which is a scale constant, not catalog
scoring.) The three components require group moments and are unavailable on
drivers whose reference bundle does not carry them.

### 2.4 `delta_mae` (v1: `mae_delta`) and `delta_mse` (v1: `mse_delta`)

The same errors, but on the **effect** (delta) vectors of §1.3 instead of raw pseudobulk — so
they score how well the predicted *change from control* matches the real change:

```math
\mathrm{delta\_mae}_p = \frac{1}{G}\sum_g \big| \hat{\delta}_{p,g} - \delta_{p,g} \big|,\qquad \mathrm{delta\_mse}_p = \frac{1}{G}\sum_g \big( \hat{\delta}_{p,g} - \delta_{p,g} \big)^2.
```

Range $[0,\infty)$; **best $=0$**.

⚠️ **Under the v2 default these are ALIASES of §2.1/§2.2, not an additional signal (#189).** The
sentence above — "how well the predicted *change from control* matches the real change" — implies
an independence that the v2 default does not have. The control cancels in any *difference*-based
error:

```math
\hat{\delta}_{p} - \delta_{p} \;=\; (\hat{x}_p - c) - (x_p - c) \;=\; \hat{x}_p - x_p .
```

`_delta_eval` subtracts the real control from the real side always, and from the predicted side
when `control_source="real"` — which **is** the v2 default (`configs/v2.yaml`). Both sides subtract
the same vector, so

$$\texttt{delta\_mae} \equiv \texttt{expr\_mae}, \qquad \texttt{delta\_mse} \equiv \texttt{expr\_mse}$$

per perturbation — the same quantity, not a second signal. (Equal to roundoff rather than
bit-for-bit: $(\hat{x}-c)-(x-c)$ and $\hat{x}-x$ are different evaluation orders. On one measured
sweep of 600 lognormal-bulk pairs, 91 differed, by 1–2 ULP — evidence that the difference is real
and small on ordinary data, not a bound. The bound the tests assert is the *forward* error of the
cancellation, which scales with the operands, since a result-relative one is unbounded near zero.)
Under
**v1** (`control_source="pred"`) they do differ substantively, but only by the constant vector
$\hat{c}-c$, which is the same for every perturbation — so even there the pair carries one
perturbation-varying signal plus a fixed offset, not two.

**What to do with that.** Do not put both in one equal-weight aggregate: it silently double-weights
that one error. Nothing shipped does — `vcc`/`vcc2026` score neither, and the generalist set uses
`mae` only — but the `full` profile scores all four, and a reader assembling a metric set from this
document would reasonably include both. The metrics are not wrong and are kept: removing them would
lose the v1 variant, which is *not* redundant, and break the catalog count assertions and every
stored baseline carrying those columns.

By contrast `delta_pearson` (§2.5) is **not** redundant with any `expr_*` metric — correlation is
not translation-invariant, so subtracting the control genuinely changes it (it removes the dominant
shared baseline-expression component). Only the *difference*-based errors collapse.

### 2.5 `delta_pearson` (v1: `pearson_delta`)

Pearson correlation between the predicted and real effect vectors, across genes:

```math
\mathrm{delta\_pearson}_p = \frac{\sum_g \big(\hat{\delta}_{p,g}-\overline{\hat{\delta}_p}\big)\big(\delta_{p,g}-\overline{\delta_p}\big)}{\sqrt{\sum_g \big(\hat{\delta}_{p,g}-\overline{\hat{\delta}_p}\big)^2} \sqrt{\sum_g \big(\delta_{p,g}-\overline{\delta_p}\big)^2}}.
```

Range $[-1,1]$; **best $=1$**. It measures the *shape* of the effect (correlation), not its
magnitude. **Degenerate:** if either effect vector has zero variance (e.g. a constant / mean-
baseline prediction $\Rightarrow \hat{\delta}_p \equiv 0$), the correlation is undefined; in
**v2** this maps to the worst value $-1$ (issue #92), in v1 it is NaN.

---

## 3. Perturbation Discrimination Score (PDS)

`pds_l1` (v1 `discrimination_score_l1`), `pds_l2` (`discrimination_score_l2`),
`pds_cosine` (`discrimination_score_cosine`). `kind = anndata`.

PDS is a **rank-based retrieval** score. It asks: *is the predicted effect for perturbation
$p$ closer to the real effect of $p$ than to the real effect of other perturbations?* This
rewards predictions that are **distinguishable across perturbations**, which plain per-gene
error does not.

For each non-control perturbation $p$, compute the distance from the predicted effect
$\hat{\delta}_p$ to **every** real effect $\delta_q$, then find the rank of the true match
$\delta_p$ when those distances are sorted ascending (rank $0$ = closest):

```math
\mathrm{rank}_p = \underbrace{\big|\{\, q : d(\hat{\delta}_p, \delta_q) \lt d(\hat{\delta}_p, \delta_p) \,\}\big|}_{\text{strictly closer}} \;+\; \frac{\big|\{\, q : d(\hat{\delta}_p, \delta_q) = d(\hat{\delta}_p, \delta_p) \,\}\big| - 1}{2},
```

i.e. **strictly-closer count plus the mid-rank of the tied block** (the tied count
includes the true match itself, hence the $-1$). With no ties the second term vanishes and
this is the plain strictly-closer count.

```math
\mathrm{PDS}_p = 1 - \frac{\mathrm{rank}_p}{D},\qquad D = \begin{cases} P & \texttt{rank\_denominator="n"}\\ P-1 & \texttt{rank\_denominator="n-1"} \ (\text{v2 default})\end{cases}
```

where $P$ is the number of non-control perturbations. $\mathrm{rank}_p = 0$ (the true match is
nearest) gives $\mathrm{PDS}_p = 1$ (perfect). Range $[0,1]$; **best $=1$**; a random guess
scores $\approx 0.5$.

**Distance** $d(a,b)$ selects the variant:

```math
d_{\ell_1}(a,b) = \sum_g |a_g - b_g|,\qquad d_{\ell_2}(a,b) = \sqrt{\sum_g (a_g - b_g)^2},\qquad d_{\cos}(a,b) = 1 - \frac{a\cdot b}{\lVert a\rVert\,\lVert b\rVert}.
```

For cosine, a zero-norm operand yields similarity $0$ (distance $1$), and the distance is
clipped to $[0,2]$.

**Ties (`tie_policy`, issue #282).**
Equidistant competitors share the **mid-rank** of the block they form — the $\tfrac{n-1}{2}$
term above — under the v2 default `tie_policy="midrank"`. This matters because an entire row
can tie exactly: under **cosine** a zero-norm predicted effect (a submission pasting the
reference control cells for a target) is at distance exactly $1$ from *every* real effect. Such
a target then scores exactly $0.5$, the no-information point.

⚠️ `tie_policy="position"` is the legacy rule (the value the **`v1` preset** carries, for
upstream `cell-eval` parity — note that is the *preset*, not the `version` field: an
`EvalConfig` built directly with `version="v1"` keeps the dataclass defaults here exactly as
it does for `rank_denominator` and `control_source`) and it is **not a neutral tie-break**. It resolves a tie to the target's index in the
sorted perturbation array — i.e. its **alphabetical position** — so an all-tied row reads
$1 - \text{index}/D$: $1.0$ for an early-alphabet target, $0.0$ for a late one. A *fully*
constant prediction self-corrects (the ranks are a bijection, so the panel mean is $0.5$
regardless); a *partially* constant one does not, which is what made this a live scoring
defect rather than a curiosity. Under `midrank` neither the per-target value nor the panel
mean depends on target names.

**Target-gene exclusion.** With `exclude_target_gene=True` (the default) target-gene columns are
dropped from both vectors before the distance is computed, so a prediction cannot trivially win
by reproducing the knocked-down gene itself. `exclusion_scope` says WHICH columns.

`"panel"` (the v2 default, #343) drops **every** panel target gene from the ranked feature space
once, up front, so all $n^2$ distances are computed on one fixed set of genes.

`"row"` is the legacy rule — pinned by the `v1` and `cell-eval-0.7.6` presets for upstream
cell-eval parity — and drops only perturbation $p$'s own column, from $p$'s comparison against
*every* reference perturbation. ⚠️ That is asymmetric and it is scoreable: reference perturbation
$q$'s own knockdown stays visible in $\delta_q$, so a submission that spikes the panel's *other*
target genes is anti-correlated with every off-diagonal competitor while its own comparison —
where its gene *is* dropped — sits at cosine $0$, and the self-match wins on no information
beyond the published target list. Measured on the three official validation contexts:
`pds_cosine` 0.798 / 0.757 / 0.761 against baselines of 0.530 / 0.528 / 0.510, i.e. **+0.57 /
+0.49 / +0.51 of member score for a submission carrying no biology**. Under `"panel"` those arms
measure exactly 0.500, a perfect submission still measures 1.000, and a partial one moves by at
most 0.01 of member score. `"row"` also deflates the DIAGONAL's reference-side norm alone, which
is worth ~+0.017 raw `pds_cosine` to any submission with a non-zero delta.

`pds_l1` is a core scored metric (`vcc`, `minimal`, `full`, `anndata`, `pds`); `pds_l2` is
`full`/`anndata`; `pds_cosine` is `full`/`anndata` and, as the 2026 competition's
discrimination metric, `vcc2026` (§6).

---

## 4. Differential-expression (DE) metrics

All are `kind = de` and consume the DE table of §1.4. Names are prefixed `de_wilcoxon_`.

**Target-gene exclusion is split across this family, but the split no longer runs through the
competition six** (#172, **ruled
2026-08-17**). The discrimination family has excluded the perturbed gene's own column since v1
(§3); on the DE side the eleven chance-corrected direction metrics exclude it too — that is what
`resolve_target_genes` and `_ontarget_excluded_frame` are for (#195, #248) — and #172 added
`de_wilcoxon_sig_jaccard` and `de_wilcoxon_lfc_nmae`, the last two scored `vcc2026` DE members
that still passed it through. With `expr_mse_unbiased_capped_norm`'s two legs (§2.3) **all six
scored `vcc2026` members now exclude.**

**Every OTHER DE metric still passes it straight through, deliberately.** The table below
measures `de_wilcoxon_overlap`, `de_wilcoxon_precision` and `de_wilcoxon_sig_recall` moving too,
by $-0.019$, $-0.030$ and $-0.019$ — they are **not scored by `vcc2026`**, and the ruling is
scoped to the competition six, so they are out of scope rather than overlooked. The same holds
for the `de_sig_agreement` family and `de_nsig_counts_*`.

Measured on the official val A panel, honest half-data arm against the
`context_mean` baseline, by stripping `feature == target` from both DE tables (FDR is computed
upstream, so this does not re-pool the multiple-testing correction):

| metric | own gene in | own gene out | Δ raw | scaled score |
|---|---:|---:|---:|---|
| `de_wilcoxon_direction_fidelity_yield_raw` *(scored)* | 0.457326 | 0.457326 | $0$ | — already excluded |
| `de_wilcoxon_direction_reach_raw` *(scored)* † | 0.996721 | 0.996721 | $0$ | — already excluded |
| `de_wilcoxon_sig_jaccard` *(scored)* | 0.400139 | 0.381338 | $-0.0188$ | $0.9188 \to 0.8717$ ($-0.0471$) |
| `de_wilcoxon_lfc_nmae` *(scored)* | 0.194301 | 0.191268 | $-0.0030$ | $1.2614 \to 1.2661$ ($+0.0047$) |
| `de_wilcoxon_precision` | 0.606486 | 0.576153 | $-0.0303$ | — not scored, **out of scope** |
| `de_wilcoxon_overlap` | 0.424565 | 0.405968 | $-0.0186$ | — not scored, **out of scope** |
| `de_wilcoxon_sig_recall` | 0.424803 | 0.406224 | $-0.0186$ | — not scored, **out of scope** |

† measured at the old `direction_reach_raw` purity floor $P_0 = 0.975$ (§4.3). The Δ is $0$
because the metric already excludes the gene, which the floor does not affect; the *level*
$0.996721$ is from that run and has not been re-measured at $P_0 = 0.9$.

The gene is not rare and it is not marginal: it appears in **299 of 300** targets' reference
rows and is reference-**significant in all 299**. The baseline arm barely moves when it is
removed ($0.033382 \to 0.033286$ on `sig_jaccard`), because a generic-response baseline does
not specifically nail the knocked-down gene — so the *scaled* score does move: an honest arm
loses $0.047$ of `sig_jaccard`'s span and gains $0.005$ on `lfc_nmae`, a net
$\approx -0.007$ on the six-member `avg_score`.

⚠️ **Read the scaled column with two caveats.** The span is measured against the **frozen val A
bundle**, whose $0$ end is the `context_mean` baseline and whose $1$ end is the *measured*
replicate (§6c-bis), **not** the constant anchor: for `sig_jaccard` that span is
$0.4325 - 0.0334 = 0.399$, so a raw shift divides by $0.399$ rather than by $1$ and is ~2.5×
larger in score terms than it looks. And the **replicate anchor was not recomputed** — a
replicate also nails the knocked-down gene, so excluding it would lower that end too and offset
part of the loss. The honest reading is "$-0.047$ of span against a fixed anchor", not "$-0.047$
of a submission's final score".

That is the measurement #172 asks for as
its first step, and it is what the ruling rests on. **It has now been made**, for the two scored
DE members above and for both legs of `expr_mse_unbiased_capped_norm`.

Two things about the ruling worth keeping, because both were live questions in the issue:

- **v1 parity was not a constraint.** #172's opening called it "the hard constraint"; v1 is
  obsolete and does not bind v2 decisions. It would not have bound here in any case —
  `sig_jaccard` and `lfc_nmae` are both `v1_name=None, v1_available=False`, and neither is in the
  `vcc` (2025) profile, whose only DE member is `de_wilcoxon_overlap`, out of scope above. The
  2025 leaderboard is therefore unchanged by this.
- **The exclusion is only worth as much as the label→gene resolution.** Both metrics route it
  through `de.require_resolution` / `de.exclude_on_target`, so guide-level labels
  (`'ABCA1-1'` vs feature `'ABCA1'`) with no `EvalConfig.target_gene_map` **raise** rather than
  quietly excluding nothing and keeping the pre-#172 meaning. That is #248's failure mode, where
  it let a trivially-gameable submission win. A *partial* resolution does not raise — the harm is
  continuous and no threshold is principled (ruled 2026-08-16 for `pds_*`) — so the rows actually
  removed are logged instead, the DE-side analogue of `baseline_meta.json`'s `n_excluded`.

**What the change does to persisted artifacts.** Three members changing meaning inside one
version is the case an identity digest cannot see. Where `cell_eval2_version` is not keyed at
all — the result cache — it obviously cannot help; where it *is* keyed, as in
`anchor_cache_params`, it is keyed but insufficient, because #172 lands *within* `0.13.0` and
because the stamp resolves through installed distribution metadata, which in an editable tree
need not be the tree under test.

Closed here:

- **result cache** — `run._result_config_digest` gains an `ontarget_exclusion_semantics` term,
  gated on `run._ontarget_exclusion_used(names)` so a run #172 cannot move keeps its warm cache.
- **anchor** — `anchor.anchor_semantic_params` gains the same term, and also gains
  `target_gene_map`, which previously entered only when a `pds_*` or a DE metric was selected;
  neither predicate fires for an expression-only anchor, whose two `expr_*` legs now resolve
  through that map. The subset feeds both the cache key and `validate_anchor`.

- **partition sidecar** — `partition.result_semantics` (#246, landed in #307 while this change was
  in review) gains `ontarget_exclusion_semantics` as the third sibling of its two existing
  "meaning changed" counters, with the schema bumped 2 → 3. ⚠️ None of the four fields
  `aggregate_partials` compares can see #172 — `real_ref_fingerprint`, `config_hash`,
  `comparator`, `metrics` all describe what was *asked for* — so two shards straddling the change
  agree on every one of them. This closed
  #315.

Deferred and filed:

- #314 — `baseline.config_digest` has no
  semantics term, so a warm pre-#172 baseline digests identically; #248 has the same gap, so does
  the `direction_reach_raw` purity floor (§4.3), and so does **#271** (`prep._grouped_sums` reduces
  wide, moving the `bulk_lognorm` pseudobulk itself for coarse-float input) — **four** live
  instances. ⚠️ #271 is a different shape from the other three: they change what a metric *means*,
  it changes the pseudobulk every expression/PDS member *reads*, so a pre-fix fractional baseline
  paired with post-fix submissions is the silently-wrong-margins failure with nothing in the digest
  to see it. The floor is the
  sharpest case for the shared registry, since it *is* a single value and needs no per-issue
  constant at all. The competition `--real-bundle` path is unaffected: it pairs through the
  anchor's semantic identity, which carries the floor.
  Deferred rather than patched because `baseline.py` carries no semantics counters at all: unlike
  the partition payload there is no slot to fill, and the fix is one shared registry every surface
  reads rather than a per-issue constant repeated again.
- #319 — a standalone
  `lfc_nmae_ref_agg.csv` is accepted on its columns alone, so a pre-#172 reference passes silently.
  It differs in **cohort** as well as level (the gate shrinks by one per resolved target and
  `min_gate_size` is judged after exclusion), which makes the two frames not row-aligned. A second
  accepted stale instance since #271: a reference computed with the **deseq2** backend pseudobulks
  through `prep._grouped_sums`, so a pre-#271 one is likewise accepted on its columns alone. Feeds
  `from_reference` only, never the enrolled `avg_score` — `score.py` computes the average before
  that column is added.

⚠️ One boundary worth stating precisely, since #172 narrows it: `partition._check_result_semantics`
only **warns** for a directory in which *no* sidecar declares semantics (#246's deliberate case 3 —
refusing would break every warm partial directory). So the one directory that can straddle #172
undetected is `pre-#307 main` + `this branch pre-merge`, both of which predate the payload. Once
this lands every build declares, and any mix of a declaring and a legacy shard raises instead.
Tightening case 3 further would be a change to #246's ruling rather than to #172, and it was **ruled to stay as #246 set it** (Alex, 2026-08-17) — warn, do not raise. Rejecting all-legacy directories outright breaks every warm partial directory; rejecting them only when the metric set includes an excluding member still breaks them for the competition profile. Pinned by `test_an_ALL_legacy_partial_directory_still_only_WARNS`.

⚠️ **`competition_digest()` is a fifth surface with the same gap, and `rule_version = 2`
is how it was closed for this change.** `competition_payload` freezes each member's scoring
*policy* — direction, anchor, penalty, clamps, `agg`, `derived`, resolved normalization,
`worst_value`, estimator — and nothing about which gene set the member computes over, so it does
not move for #172. Two real bundles built either side of this change would therefore carry the
same `rule_digest` while their frozen replicate anchors were computed over different gene sets,
which is exactly the comparability the digest is supposed to certify. The lever is
`rule_version` ("bump deliberately"), and moving it invalidates every already-built bundle by
design — so it was **deferred to one bump for the whole set** rather than taken per PR (Alex,
2026-08-17), and **pulled in the `0.14.0` release wave** (#317): `rule_version` 1 → 2 for
exactly three semantics changes — #172's target-gene exclusion, `direction_reach_raw`'s
calibrated purity floor, and #271's wide pseudobulk reduction — with the three #276 val bundles
rebuilt as `-r2` against it in the same wave. The debt list against `rule_version = 2` is empty,
and it lives in the code at the literal itself so whoever bumps next can see what the version
means. Nothing *fails* if a future semantics change forgets to join it, which is why the lever is
written down redundantly — at the literal, in the digest test's comment, here, and in the release
notes.

⚠️ **That paragraph is the `rule_version = 2` history, and 2 is no longer current.** `0.15.0`
pulled the lever again — `rule_version` 2 → 3 for three further semantics changes the digest
cannot see: #343's panel-scope target-gene exclusion on `pds_cosine`, #348's bound on
`expr_mse_unbiased_capped`'s prediction-side correction, and #351's reference-only DE gene gate.
All three landed under ONE bump because no bundle had yet been built at version 3, and the debt
list against version 3 is likewise empty. `competition_digest()` is `fb5aa56b…`, so this release
REQUIRES the three #276 val bundles to be rebuilt as `-r3` against it: a `-r2` bundle carries
`f32f0f9c…` and `score.py` refuses it rather than scoring it.

The scale registry is the fourth identity surface and it *was* moved: `low-random_high-1_v6` is
retired and `low-random_high-1_v7` was minted (and `_v8`, `_v9`, `_v10` after it), on the rule that
a change to what a keyed metric
*means* mints a version even when every field is identical (§6d) — three of its six keyed
members changed here. (`_v7` was in turn retired in the same release wave by the
`direction_reach_raw` purity-floor change, then `_v9` for #271's reduce-wide;
**`low-random_high-1_v10`** is what ships.)

One consequence to know about `de_wilcoxon_sig_jaccard`: exclusion can turn a non-empty union
into an empty one, which scores 1.0 by the $J(\varnothing,\varnothing)=1$ convention (§4.5). A
perturbation whose *only* reference-significant gene was its own target now has an empty
reference set, so a submission calling nothing significant for it reads 1.0 where it previously
read 0.0. It needs $|R_p| = 1$ before exclusion, so it is rare, and it is real-side — see §6.0 for
how the six members handle a perturbation with no recoverable signal.

### 4.1 Significant-gene counts

**`de_wilcoxon_nsig_counts_real`** (v1 `de_nsig_counts_real`) and **`..._pred`** — the number
of significant genes on each side:

```math
\mathrm{nsig\_real}_p = |S_p^{\text{real}}|,\qquad \mathrm{nsig\_pred}_p = |S_p^{\text{pred}}|.
```

Purely **diagnostic** (`scored=False`, and genuinely direction-less: neither more nor fewer
significant genes is "better").

**`de_wilcoxon_nsig_spearman`** (v1 `de_spearman_sig`) — one global Spearman rank correlation
between the per-perturbation real and predicted significant-gene counts, over all
perturbations with $\ge 1$ real-significant gene, broadcast to every perturbation:

```math
\mathrm{nsig\_spearman} = \rho_{\text{Spearman}}\big(\{|S_p^{\text{real}}|\}_p,\ \{|S_p^{\text{pred}}|\}_p\big).
```

Range $[-1,1]$; **best $=1$**. **Degenerate** (undefined correlation, e.g. zero-variance
counts): **v2** worst $-1$ (issue #92), v1 NaN.

### 4.2 Top-gene overlap and precision

Let $R_p^{(k)}$ and $P_p^{(k)}$ be the **top-$k$** real and predicted genes ranked by
$|\mathrm{lfc}|$ (restricted to significant genes).

**`de_wilcoxon_overlap`** (v1 `overlap_at_N`) — recall-style overlap, sized by the **real**
significant set. With $k' = \min(k, m_r)$ and $m_r = |S_p^{\text{real}}|$ (for the un-capped
`overlap` metric, $k' = m_r$):

```math
\mathrm{overlap}_p@k = \frac{\big|\,R_p^{(k')} \cap P_p^{(k')}\,\big|}{k'}.
```

**`de_wilcoxon_precision`** (v1 `precision_at_N`) — precision-style, sized by the **pred**
significant set: with $k' = \min(k, m_p)$ and $m_p = |S_p^{\text{pred}}|$,

```math
\mathrm{precision}_p@k = \frac{\big|\,R_p^{(k')} \cap P_p^{(k')}\,\big|}{k'}.
```

Both range $[0,1]$; **best $=1$**. Each comes in an un-capped form ($k=N$, all significant
genes) plus fixed cutoffs $k \in \lbrace 50,100,200,500 \rbrace$ (`..._top50` … `..._top500`).
`de_wilcoxon_overlap` (the un-capped overlap) is one of the three **`vcc`** competition
metrics.

> These raw overlap/precision scores are inflated by chance when significant sets are large
> relative to the gene panel — see §5 for the chance-corrected versions.

### 4.3 Significant-gene recall, direction, and LFC correlation

These are keyed off the **real** significant set $S_p^{\text{real}}$, except for the explicitly
model-conditioned `..._model_direction_match`, whose denominator is $S_p^{\text{pred}}$.

**`de_wilcoxon_sig_recall`** (v1 `de_sig_genes_recall`) — fraction of real-significant genes
that are also significant in the prediction:

```math
\mathrm{sig\_recall}_p = \frac{\big|\,S_p^{\text{real}} \cap S_p^{\text{pred}}\,\big|}{\big|\,S_p^{\text{real}}\,\big|}.
```

Range $[0,1]$; **best $=1$**. **v2** worst $=0$ for perturbations with no real-significant
genes (issue #89); v1 omits them.

**`de_wilcoxon_direction_match`** (v1 `de_direction_match`) — among real-significant genes that
are also present in the prediction ($\mathcal{G}^{\text{pred}}$), the fraction whose predicted
log₂FC has the **same sign** as the real one:

```math
\mathrm{direction\_match}_p = \frac{1}{|S_p^{\text{real}} \cap \mathcal{G}^{\text{pred}}|}\sum_{g \in S_p^{\text{real}} \cap \mathcal{G}^{\text{pred}}} \mathbf{1}\!\big[\operatorname{sign}(\widehat{\mathrm{lfc}}_{p,g}) = \operatorname{sign}(\mathrm{lfc}_{p,g})\big].
```

Range $[0,1]$; **best $=1$**. **v2** worst $=0$ (issue #89); v1 omits zero-real-sig perts.

**`de_wilcoxon_model_direction_match`** (v1 `de_model_direction_match`) — the reverse
conditioning: among model-significant genes that are also present in the real DE table
($\mathcal{G}^{\text{real}}$), the fraction whose real and predicted log₂FC signs agree:

```math
\mathrm{model\_direction\_match}_p = \frac{1}{|S_p^{\text{pred}} \cap \mathcal{G}^{\text{real}}|}\sum_{g \in S_p^{\text{pred}} \cap \mathcal{G}^{\text{real}}} \mathbf{1}\!\big[\operatorname{sign}(\widehat{\mathrm{lfc}}_{p,g}) = \operatorname{sign}(\mathrm{lfc}_{p,g})\big].
```

Real significance is deliberately ignored: every model DEG contributes a sign check whenever
the gene is present in the real table. Range $[0,1]$; **best $=1$**. **v2** worst $=0$ for
perturbations with no model-significant genes; v1 omits them.

> **Superseded by `de_wilcoxon_direction_precision`.** `model_direction_match` compares
> `sign(lfc)` with a plain `==`, so a gene whose log₂FC is exactly $0$ — or `NaN` — on
> *both* sides scores as agreement. It is retained **unchanged** for continuity with
> v0.3.0; new work should prefer `de_wilcoxon_direction_precision`, which states the rule
> explicitly. (The other sibling, `de_wilcoxon_direction_match`, is pinned byte-for-byte
> against upstream `cell-eval` and cannot change either.)

**`de_wilcoxon_direction_precision`** (v2-native) — the same conditioning as
`model_direction_match`, with an explicit rule for undefined directions. A direction is
undefined when a log₂FC is null, `NaN`, or exactly $0$; $\pm\infty$ **is** a direction.

- **The reference cannot adjudicate** — gene absent from the real table, or real log₂FC
  undefined → the pair is **excluded** from numerator and denominator.
- **The model declined to commit** — pred log₂FC undefined while claiming significance →
  the pair **stays in the denominator** and counts as a **miss**.

The asymmetry is deliberate: it is the no-droppable-NaN principle (issue #89) at gene
granularity, so a model can never improve its score by declining to answer. Excluding on
the reference side is safe because the reference is not the adversary. ⚠️ This diverges
from `dge_robust`, which uses **symmetric exclusion** — correct there, because its τ arm
compares two replicates of one method with no adversary, and counting undefined as a miss
would confound "no direction" with "wrong direction".

**Measured, not assumed:** on the H1_CGS across-replicate arm the two rules differ by
**exactly zero** — 0 disagreeing pairs out of 987,394 model-significant, reference-adjudicable
pairs across wilcoxon, memento and deseq2, and identical per-target medians to four decimals. The
case the two rules disagree about — a gene the reference can adjudicate but the model called
significant without committing to a direction — was **not observed in those 987,394 pairs**, which
is unsurprising since a method that calls a gene significant rarely gives it a zero or `NaN` fold
change. This is one arm, not a general result. The rules remain *conceptually*
different, and a predictor built to exploit the difference would separate them — which is why
cell_eval2 keeps the ungameable one — but published numbers from the two repos are comparable on
data of this kind.

Duplicate `(target, feature)` rows raise rather than being silently fanned out by the join.

Range $[0,1]$; **best $=1$**, worst $=0$.

**`de_wilcoxon_direction_sensitivity`** (v2-native) — how deep the prediction's ranking
stays directionally pure, relative to what the reference confidently adjudicated:

```math
P(k) = \frac{\#\{\text{matching among the first } k \text{ adjudicable pairs}\}}{k}, \qquad
k^*(t) = \max\{k : P(k) \ge P_0\}, \qquad
\mathrm{sensitivity}_t = \frac{k^*(t)}{N_{\mathrm{conf}}(t)}.
```

Depth $k$ counts **adjudicable pairs**, not ranked rows: a pair whose reference log₂FC is
null, NaN or exactly zero carries no directional evidence and is skipped by both the
numerator and $k$ itself. (Before #204
$k$ counted ranked rows while the denominator counted adjudicable pairs, which is what made
the full-universe variants ~50× larger — see the correction note below.)

Genes are ranked within a target by the **prediction's**
`(Float64(p_adj) asc, Float64(p_value) asc, |log₂FC| desc, feature asc)` — the two p-keys
are Float64-normalised before sorting, so `Decimal` p-values differing only beyond ~15
significant digits collide and the later keys decide their order. `p_adj` is primary deliberately:
`nan_lfc_policy="mask"` and the `min_abs_log2fc` floor both rewrite `p_adj` without
touching `p_value`, so a `p_value`-primary key would stop the significant set being a
prefix of the ranking. `p_value` is not a required column — when absent it drops out of the
key and a warning is logged; because it only ever breaks ties *within* a `p_adj` plateau,
its absence cannot move a gene across the significance boundary.

**$P_0$ is not the same constant for both metrics in this family.**

- `de_wilcoxon_direction_sensitivity` (v0.5.0) keeps $P_0 = 1 - \alpha/2$, **derived**: a false
  discovery is a true null, so its sign in an independent repeat is a coin flip, and an
  FDR-$\alpha$ set sustains a purity of $1 - \alpha/2$. At $\alpha = 0.05$, $P_0 = 0.975$.
- The `corrected=False` spellings use `direction.REACH_PURITY_FLOOR` $= 0.9$: a **calibrated
  constant, chosen not derived**, and deliberately independent of $\alpha$ (Alex, 2026-08-17).
  `_register_de_family` runs for **both** DE backends, so there are eight registered reach
  names — `de_{wilcoxon,deseq2}_direction_reach{,_unbounded}{,_raw}` — of which the four
  `_raw` ones read the floor. `de_wilcoxon_direction_reach_raw` is the `vcc2026` scored member;
  the `deseq2` spelling is one metric behind a second backend and can only ever reach a
  *diagnostic* column.
- The four `corrected=True` spellings keep their own $q + (1-\alpha)(1-q)$ majority-sign null:
  their **threshold and arithmetic are untouched**. Their *cache identity* does change — the
  `_reach_floor_used` predicate is keyed on function identity, so it fires for all eight names
  on purpose (deliberate over-invalidation, documented at the definition).

The derivation above is sound but computes the purity a *perfect* independent repeat attains
against an $\alpha$-FDR reference — the metric's **ceiling**. Using it as the pass mark left no
margin: measured on the three official val lines, a real split-half replicate's per-gene
directional accuracy over the reference-significant adjudicable set is $0.9561/0.9391/0.9488$
(mean over targets), *below* $0.975$ on every line. The first depth tolerating one sign error is
$\lceil 1/(1-P_0)\rceil$ — $40$ at $0.975$, $10$ at $0.9$ — so $46$–$61\%$ of the $300$-target val
cohort previously had no depth at which a single error was survivable, against $12$–$30\%$ now;
and at a fixed uniform $95\%$ accuracy the cohort mean varied $13.5\times$ across
$N_{\mathrm{conf}}$ strata at $0.975$ against $1.3\times$ at $0.9$. The change does **not** move
the chance floor beyond $+0.0017$ at worst ($0.042694/0.065849/0.080890 \to
0.042694/0.067557/0.082166$; A exactly unmoved) and does not saturate the
anchor.

⚠️ The two metrics therefore no longer agree on a fixture with no target-gene row, an identity
that used to hold and that `tests/test_direction_reach.py` now pins as *broken on purpose*.
`direction_sensitivity` is one of the three v0.5.0 metrics whose values must not move, and it is
diagnostic-only. The `corrected=True` reach variant likewise keeps its own $1-\alpha$
majority-sign derivation; it is not a `vcc2026` member.

$N_{\mathrm{conf}}(t)$ counts reference-significant genes across the **whole real table**,
not only those the prediction covers, so a model cannot shrink its own **budget** by
omitting genes.
$k^* = 0$ when purity never reaches $P_0$ — a computed zero. Targets where the reference
adjudicated nothing ($N_{\mathrm{conf}} = 0$) are undefined and take the worst value $0$
under v2; ⚠️ that fraction has been measured at 13–24% on real data, so it moves the
aggregate materially.

⚠️ **That invariance is denominator-only, and the score is not omission-proof**
(#291). This sentence used to
read "so a model cannot raise its score by omitting genes"; the premise is true and
measured, the conclusion does not follow. $N_{\mathrm{conf}}$ comes from the real table
alone, but the **numerator** is built on the pred/real *inner* join, so a
$(t, g)$ pair the real table carries and the pred table does not is absent from the
ranking pool entirely — it neither advances the depth nor counts as a miss — while still
counting in $N_{\mathrm{conf}}$. Because $k^*$ takes the *deepest* qualifying depth,
deleting **head-ranked misses** is worth a discontinuous jump: on a one-target toy with 80
adjudicable pairs whose 3 top-ranked are wrong, deleting exactly those 3 rows moves
`direction_reach_raw` from $0.0$ to $0.9625$ at the old $P_0 = 0.975$. ⚠️ **The block has to be
bigger at $P_0 = 0.9$ and the jump is smaller**: $k^* = 0$ needs $m > (1-P_0)N$, so it takes $9$
head misses of $80$ rather than $3$, and deleting them moves $0.0 \to 0.8875$ (re-measured). The
exploit is ~$4\times$ harder to reach, not closed. Three qualifications, all measured:

- **Omission is a *targeted* edit, not a monotone gain.** A miss also *buys* depth, so
  deleting **tail** misses lowers the score. It pays only where the deleted misses hold
  $\mathrm{misses}(k) > (1-P_0)k$ for every $k$ — i.e. at the head.
- **Deletion is not the strongest available edit.** Keeping the rows and setting their
  predicted $p_{\mathrm{adj}}$ to $1$ — declining to call them — scores $0.975$ on the same
  toy, *above* deleting them, because the adjudicated pool is filtered on the **reference's**
  significance but *ranked* by the **prediction's**, so a declined call is demoted to the
  tail rather than removed. The left-join repair proposed on the issue re-admits an omitted
  pair with a null predicted $p_{\mathrm{adj}}$, which `_rank_p` sends to that same tail, and
  therefore reproduces $0.975$ — scoring the omission **higher** than the shipped inner join.
  Any real fix has to decide where a non-answer *ranks*.
- **Coverage used to differ on honest runs, and no longer does on the h5ad path.**
  `filter_gene_min_cpm_cell` (5.0 under the competition preset) is applied to each side's own
  table. Under `rule_version` 2 it kept a gene when the target group's mean CPM cleared the
  threshold **or** the control's did, so a gene the prediction expressed below the cutoff left the
  prediction's DE table while staying in the reference's: on the official val A panel the
  `context_mean` baseline arm omitted **1,754 of 102,786** reference-significant pairs (1.706%)
  across 159 of 300 targets, and an honest half-data arm **199** (0.194%) across 68 — counted after
  removing each resolved target's own gene, the same population `_reference_stats` builds
  $N_{\mathrm{conf}}$ from (before that exclusion the denominator read $103{,}085$; both numerators
  unchanged). Under `rule_version` 3 the **control alone** decides the kept set (#351), and
  `control_source="real"` gives both sides the same control, so the two kept sets are identical and
  the omission is **0 of 100,771** on that same baseline arm — same post-exclusion population, with
  the gate taking 2,015 pairs out of the reference's confident budget (0 on the #351 attack arm, was
  1,097; 0 on an honest 200-cell arm, was 240). On the ordinary h5ad path the invariant above is therefore
  no longer denominator-only — there is nothing for the inner join to drop. It stays
  denominator-only for a **supplied** `--de-pred` table and for `control_source="pred"` (the
  replicate anchor's splits, where each half carries its own control), which is what the diagnostic
  still guards. The eleven target-excluding direction metrics now log this at WARNING when one of them
  runs — emitted once per prepared DE object by `direction._warn_coverage_once`, which carries its
  own memo so that reading `_reference_stats` for a diagnostic cannot consume it and silence a
  later metric. A DE run that selects none of the eleven, or only the three v0.5.0 direction
  metrics (which read the unfiltered frame), stays quiet. Anchoring the
  pool on the real side and ranking non-answers first closes the toy exploit but moves
  `direction_reach_raw` on those honest arms too (measured at $P_0 = 0.975$: baseline
  $0.0423 \to 0.0286$, honest arm
  $0.9967 \to 0.9881$ — against the frozen val A bundle that is $-0.011$ of the member's span,
  $-0.002$ on `avg_score`, with the replicate anchor left un-recomputed), which is a release
  decision rather than a metric one.

Both variants sit only in the `full` and `de` profiles, so neither reaches the `vcc`
competition score; both are `scored=True`, so both enter the `full`/`de` `avg_score`. They
differ in their **anchor**, not their enrolment: `direction_sensitivity` is bounded by $1$
and normalizes against it, while `direction_sensitivity_universe` has no constant anchor
(see below) and normalizes against its own baseline instead.

**`de_wilcoxon_direction_sensitivity_universe`** — identical, except the curve is ranked
over the **whole shared gene universe** rather than the reference-adjudicated set. ⚠️
**Unbounded above:** $k^*$ is no longer capped by $N_{\mathrm{conf}}$, so the ratio exceeds
$1$ for 19.2% of targets on a 1,029-perturbation panel, and its generic-response baseline
*beats* a real technical replicate — the metric inverts. ⚠️ It is nonetheless **scored**, like
every other DE metric with a direction; what its unboundedness buys it is `anchor=None` — there is no
constant perfection point, so its score is $u/b - 1$ rather than a fraction of the distance to
perfection. That score is clamped to $[-2, 2]$ like every anchorless metric (§6.1), so it cannot
dominate the aggregate, but read its contribution with the inversion in mind. It reaches no v1
output (`v1_name=None`), so only the `full`/`de` aggregate sees it.

⚠️ **Corrected by issue #204.** The
"roughly 74% of targets" this section previously reported was measured while prefix *depth*
counted ranked **rows** rather than adjudicable **pairs**: `_purity_curve` used
`in_denom.cum_count()`, and polars' `cum_count()` counts non-null entries rather than `True`
ones, so a pair the reference cannot adjudicate (real log₂FC null, NaN or exactly zero) was
excluded from the purity denominator yet still extended the prefix for free. Those pairs
concentrate at the *head* of the ranking — a gene the reference never detected still draws a
small predicted delta, and a zero-variance predicted control makes it maximally significant —
so 97.9% of the median $k^*$ prefix carried no directional evidence at all. Depth now counts
adjudicable pairs. Measured effect on the full-universe variants (median):
`direction_sensitivity_universe` 12.015 → 0.2393, `direction_reach_unbounded` and
`direction_reach_unbounded_raw` 12.181 → 0.2424 — all ~50×. The adjudicated variants,
including the scored `direction_reach`, were **unchanged on all six reference lines
measured**: not one reference-significant gene there was unadjudicable, so those curves
carried no padding.
⚠️ That is an empirical result, not a structural guarantee. Nothing forbids a
reference-significant row from carrying a log₂FC of exactly zero or null (or, under
`nan_lfc_policy="keep"`, NaN), and an externally supplied real table can carry anything —
such a row, once joined, no longer advances the adjudicated depth and can therefore shorten
$k^*$ (*can*, not *must*: it may be absent from the prediction, or there may simply be no
qualifying curve row at or after it — note purity is **not** monotone and $k^*$ is the
*deepest* qualifying depth, so a dip that later recovers still counts). The adjudicated
variants can therefore move on a dataset of that shape, consistently with the corrected
semantics.

**Precision is a point on the purity curve.** Because the ranking is `p_adj`-primary and the
significant set is by definition $\{p_{\mathrm{adj}} < \alpha\}$, that set is exactly the
top-$k$ prefix — including for masked, floored and user-supplied tables whose `p_adj` is not
BH of their `p_value`. ✅ It is now a prefix **by construction**
(#207): the purity curve's sort
leads with a *native* $p_{\mathrm{adj}}^{\text{pred}} < \alpha$ boolean before the Float64
ranking key. Previously the two could disagree — the ranking key is Float64-normalised while
the significance filter compares natively, and polars compares a `Decimal` column natively
against an *integer* or `Decimal` threshold (only a *float* threshold casts) — so a non-float
`p_adj_threshold` over a `Decimal` `p_adj` split the two representations and the significant
set could stop being a prefix. That was **pre-existing**, broken before
#204 as well, and reachable only
with an externally supplied `Decimal` table (no DE producer in this repo emits one) *and* two
p-values colliding in Float64 across the boundary *and* a later key component ordering the
non-significant one first; built end to end, `direction_precision` read $1.0$ where purity at
the boundary read $0.0$. The new key is **inert** wherever the two representations already
agreed — for any faithfully-castable `p_adj` it is a monotone function of the next key — so
it moves results only for the input that was wrong. Verified bit-identical on the official
val A DE tables. So `direction_precision`
is purity evaluated at
the **end of that prefix** on the **full-universe** curve — since #204, at
$k = |S^{\text{pred}}_p \cap \{\text{adjudicable}\}|$ rather than at
$k = |S^{\text{pred}}_p|$, the two differing exactly when the model calls a gene the
reference cannot adjudicate. Purity itself is untouched by that change, so the identity is
re-indexed rather than weakened. The equality of *values*
additionally requires a non-zero denominator, and the two degenerate perturbations miss it
differently: one with **no model-significant gene at all** has no $k = 0$ curve row to
compare against, whereas one whose significant set is non-empty but **wholly
reference-unadjudicable** does have a row, carrying purity `null`. Raw precision omits both,
and the v2 no-drop fill reports each as $0$. The identity does *not* hold on the adjudicated
curve at all, where the top-$k$ is a different gene set.

**`de_wilcoxon_lfc_spearman`** (v1 `de_spearman_lfc_sig`) — Spearman correlation of the log₂FC
values over the real-significant genes (predicted LFC filled with $0$ where absent):

```math
\mathrm{lfc\_spearman}_p = \rho_{\text{Spearman}}\big(\{\mathrm{lfc}_{p,g}\}_{g\in S_p^{\text{real}}},\ \{\widehat{\mathrm{lfc}}_{p,g}\}_{g\in S_p^{\text{real}}}\big).
```

Range $[-1,1]$; **best $=1$**. **v2** worst $=-1$ for degenerate/undefined cases (issue #89);
v1 omits/NaN.

**`de_wilcoxon_lfc_spearman_pos`** (v1 `de_spearman_pos_lfc_sig`) and
**`de_wilcoxon_lfc_spearman_neg`** (v1 `de_spearman_neg_lfc_sig`) — the same correlation
restricted to the real-significant genes whose **real** log₂FC is positive (`pos`, up-regulated)
or negative (`neg`, down-regulated):

```math
\mathrm{lfc\_spearman}^{+}_p = \rho_{\text{Spearman}}\big(\{\mathrm{lfc}_{p,g}\}_{g\in S_p^{\text{real},+}},\ \{\widehat{\mathrm{lfc}}_{p,g}\}_{g\in S_p^{\text{real},+}}\big),\qquad S_p^{\text{real},+} = \lbrace g \in S_p^{\text{real}} : \mathrm{lfc}_{p,g} \gt 0 \rbrace,
```

and analogously $S_p^{\text{real},-}=\lbrace g\in S_p^{\text{real}} : \mathrm{lfc}_{p,g} \lt 0\rbrace$ for
`neg`. They diagnose **directional asymmetry** — a model that recovers induced genes well but
repressed genes poorly (or vice versa) — which the combined `lfc_spearman` can mask. Range
$[-1,1]$; **best $=1$**; **v2** worst $=-1$ for degenerate/undefined cases (a perturbation with no
qualifying up-/down-regulated significant gene); v1 omits/NaN.

**`de_wilcoxon_lfc_nmae`** (v2-native, no v1 alias; issue
#208) — the **magnitude** counterpart of
the three correlations above. They say whether the ordering is right; this says whether a two-fold
change is reported as two-fold. Over the real-significant gate $S_p^{\text{real}}$ **less the
perturbed gene's own row** (#172, §4):

```math
\mathrm{lfc\_nmae}_p = \frac{\mathrm{mean}_{g\in S_p^{\text{real}}\setminus\{g_p\}}\big|\widehat{\mathrm{lfc}}_{p,g} - \mathrm{lfc}_{p,g}\big|}{\mathrm{mean}_{g\in S_p^{\text{real}}\setminus\{g_p\}}\big|\mathrm{lfc}_{p,g}\big|}.
```

The gate supplies **both** the numerator and the denominator here, so removing that row removes it
from each — unlike `de_wilcoxon_sig_jaccard`, whose loss is one-sided, the effect is a ratio of two
shrinking means and is small and signed either way. The gate *size* shrinks by one per resolved
target too, so a perturbation sitting at exactly `min_gate_size` before exclusion falls below it
after; that is real-side like every other rule here, so the omitted set stays identical for every
submission.

Range $[0,\infty)$; **best $=0$**; lower is better. The normalization is what the `n` denotes and
it is load-bearing: **a prediction whose log₂FC is exactly zero scores exactly $1.0$ on every
dataset**, by construction — at $\widehat{\mathrm{lfc}}=0$ the numerator *is* the denominator. That
fixes one end of the scale without any reference to the evaluation data, and it makes $2.0$ mean
"predicting backwards", exactly twice as bad as silence.

⚠️ **The anchor is exact in log₂FC space, not in submission space**
(#286). This paragraph used to say "a
submission predicting no change scores exactly $1.0$", and that step does not survive
`control_source="real"` — the v2 default and what `vcc2026` scores. There a predicted
perturbation's log₂FC is computed against the **real** control's own cells, not the prediction's,
so a submission that broadcasts the exact unrounded true control mean to every cell is still not
compared against itself: §1.4's DE mean is a per-cell normalize-then-average, which *need not
agree with* normalizing the mean — the dispersion-functional asymmetry §1.2 records as
deliberately *not* covered by #264 on the DE side. Such a submission can therefore land **near**
$1.0$ rather than at it, and can land below it for nothing:

| panel | perturbations | exact-control-mean submission | dispersed sibling |
|---|---:|---|---|
| committed fixture (`control_source="real"`) | 5 | 0.9397 – 1.0397, **3 of 5 below 1.0** | — |
| committed fixture (`control_source="pred"`) | 5 | $1.0$ on all five, exactly | — |
| official val A / B / C | 272 / 229 / 218 returned, of 300 | 1.0058 / 1.0047 / 1.0097 | 0.9976 / 0.9987 / 0.9988 |

The fixture's 6% is per-perturbation noise on 100 control cells at a library-size CV of $0.3373$;
averaging over the 272 / 229 / 218 perturbations this member *returns* — the rest fail the
real-side gate below, so they are never averaged at all — shrinks the *aggregate* to under
$1\%$, but not the per-perturbation deviation. A dispersed context-mean arm reaches $0.9909$,
i.e. $0.91\%$ below the no-skill point — though that arm is an **oracle** comparator rather
than a floor a submission could reach (§6a), so it bounds the metric's triviality and is not
a score a model collects for free. Reproduced on both CPU backends (pdex and scanpy agree to 3 decimals; the
GPU backend was not run).

⚠️ **What decides it is depth–composition covariance, and depth heterogeneity alone is
insufficient** — #286 and §1.2 both name the latter. With $\pi_c = x_c / L_c$ the per-cell
composition,

```math
\mathrm{CPM}\big(\textstyle\mathrm{mean}_c\, x_c\big) - \mathrm{mean}_c\, \mathrm{CPM}(x_c)
  \;=\; 10^{6}\,\frac{\mathrm{Cov}_c\big(L_c,\ \pi_c\big)}{\mathbb{E}[L]}.
```

Depth spread is *necessary* — no spread, no covariance — and nowhere near sufficient: a panel whose
cells differ $10\times$ in depth while sharing **one** composition has a discrepancy of exactly
zero, because every cell's CPM vector is literally the same vector. Measured on two synthetic
panels carrying the same depth multiset, the same composition multiset and the same library-size CV
($0.8182$), differing only in which depth each composition is *paired* with — so their real-side DE
gate and the real log₂FCs — hence the `nmae` denominator — are the same, checked by running the
production DE path on both panels: the anchor reads exactly $1.00000000$ at zero covariance, and
$1.2010$ / $1.1936$ for the two pairings with it. `tests/test_lfc_nmae_anchor_286.py` pins all three
values and the matching itself.

**Scope:** the exact-zero claim is about the perturbations the metric *returns* — the gate rules
below drop an empty gate, one below `min_gate_size`, or a zero *or non-finite* denominator,
and those score nothing rather than $1.0$. The enrolled `avg_score` normalizes against a *measured* baseline and
never reads the $1.0$; what reads it is `low-random_high-1_v10`'s `base` for this member (§6d) —
and a requested `--scale` column carries its **own** `avg_score`, which does read every scale
base. That base is a policy constant — the same situation §6d already records for
`expr_mse_unbiased_capped_norm`. The competition rule pins `control_source` to `real` for the scored leg and `pred` for the anchor leg,
and the invariant is exact only on the latter.

**The gate, the gate size and the denominator are computed from the real side alone**, before the
prediction is joined, and three rules follow from that ordering:

- **Omission is real-side-only, hence identical for every submission.** A perturbation is omitted
  when its gate is empty, smaller than `min_gate_size` (default $10$ — a ratio over a handful of
  just-over-threshold genes is noise), or when $\mathrm{mean}|\mathrm{lfc}_{p,g}|$ is $0$ or
  non-finite. Each omission
  is logged at WARNING with its count and reason. Because the gate never depends on the prediction,
  no submission can influence its own omission set — which is what makes `worst_value=None` safe
  here. `nmae` is unbounded above, so there is no finite worst value to fill with and a constant
  would be an invented number.
- **A non-finite predicted log₂FC is *filled* with $0$, exactly like an absent gene — never
  masked.** #208 §5.2 proposed masking; this implementation deliberately diverges. Masking would
  hand the submission control of its own gate size (emit NaNs until a badly-scoring perturbation
  drops below `min_gate_size` and vanishes from its own aggregate), and a model emitting `inf`
  everywhere would leave an empty numerator over a non-empty denominator and score $0.0$ —
  *perfect*. Filled, that model scores exactly $1.0$, the same as silence. The substitution is
  logged at WARNING.
- **Real-side** non-finite log₂FCs *do* leave the gate. That is the other direction and it is a
  property of the evaluation data, identical for every submission.

Duplicate `(target, feature)` rows raise on either side rather than being silently aggregated: a
duplicate changes the gate size *and* the denominator with no other signal.

**Scored with `scoring.ERROR_LINEAR`, not `ERROR`.** Below the baseline the score is a straight
line $\max(-6,\,1-r)$ rather than the four centroid error metrics' Box–Cox tail; at and above
the baseline the two are the same $1-r$. See §6.1 for why this family alone was moved.

`de_deseq2_lfc_nmae` is registered under the deseq2 namespace with the same function, but — like
every `de_deseq2_*` entry — carries `profiles=()` and is reached only through the backend relabel,
never selected by a profile. It moves with `de_wilcoxon_lfc_nmae` on scoring policy — one metric,
two DE backends. ⚠️ That is not inert: `run._effective_de_spec` (and `metric_output_names`, which
mirrors it) relabels the *wilcoxon* name to it under `de.backend="deseq2"`, so a run whose
**resolved metric set contains the wilcoxon sibling** scores this entry instead of it, and its
below-baseline scores moved with the policy exactly as the wilcoxon sibling's did. That includes
`vcc2026` — `_guard_deseq2_metric_selection` permits the profile's wilcoxon names under that
backend — so a DIAGNOSTIC `vcc2026` per-metric column and its `avg_score` move too. The one thing
the DESeq2 spelling cannot reach is an **enrolled** official competition score. `de.backend` is
excluded from `rule_config_hash`, so the config hash does *not* flag it; what does is the
relabelled **membership** — four wilcoxon members missing and four DESeq2 members extra — which is
sufficient on its own to make the bundle diagnostic. The per-member estimator check reports the
same relabelling independently.

### 4.4 Significance-recovery AUC

**`de_wilcoxon_pr_auc`** (v1 `pr_auc`) and **`de_wilcoxon_roc_auc`** (v1 `roc_auc`) — treat
recovering the real-significant genes as a per-perturbation retrieval problem. For each gene,
the **label** is $\mathbf{1}[g \in S_p^{\text{real}}]$ and the **score** is
$-\log_{10}(p^{\mathrm{adj},\text{pred}}_{p,g})$ (with a small floor on the p-value so zeros do
not blow up):

- `pr_auc` is the **average precision** (area under the precision–recall curve),
- `roc_auc` is the **area under the ROC curve** (true-positive rate vs false-positive rate).

Both range $[0,1]$; **best $=1$** (ROC $0.5$ = chance). A perturbation with a **single-class**
label (all genes significant, or none) has an undefined AUC: **v2** maps it to the worst value
$0$ (issue #89); v1 emits NaN.

### 4.5 Symmetric significance agreement (Jaccard)

**`de_wilcoxon_sig_jaccard`** (v2 only) — Jaccard index of the two significance sets.

```math
\mathrm{sig\_jaccard}_p = \frac{\big|\,\tilde S_p^{\text{real}} \cap \tilde S_p^{\text{pred}}\,\big|}{\big|\,\tilde S_p^{\text{real}} \cup \tilde S_p^{\text{pred}}\,\big|} = \frac{\mathrm{TP}}{a + b - \mathrm{TP}},
\qquad \tilde S_p^{\bullet} = S_p^{\bullet}\setminus\{g_p\}
```

**Both** sets drop the perturbed gene's own row
(#172, §4). The symmetry is required
rather than cosmetic: dropping the pair from the reference alone would leave a predicted
on-target call in $b$ as a pure penalty, and dropping it from the prediction alone would leave it
in $a$ as a guaranteed miss. Here $a$, $b$, and $\mathrm{TP}$ are the quantities in §5's
$2\times2$ table, counted over unique `(target, feature)` pairs and after that exclusion; see the
duplicate-row qualification below. The score is
symmetric, unlike the recall side (`sig_recall`, denominator $a$) and the precision side
(`precision_adjusted`'s $\mathrm{PPV}$ term, denominator $b$): it penalizes both missed real
DEGs and spurious predicted ones. Its range is $[0,1]$; **best $=1$**.

An **empty union** ($a=b=0$, neither side calls anything significant) is the only undefined
case and returns **1.0**, the set convention $J(\varnothing,\varnothing)=1$ — both sides agree
there is no response. This is a real regime: **on four real CCL contexts, 4.5%–17.8% of
perturbations have no significant genes in the reference**. On the 24-submission VCC campaign
the convention never fires: $a=0$ and $b=0$ each occur 0/2400 times. The consequence is worth
stating plainly: a model that calls nothing significant anywhere collects $1.0$ on every
non-responsive perturbation, so read this metric next to `de_wilcoxon_nsig_counts_pred`.

⚠️ **#172's exclusion enlarges that regime slightly.** A perturbation whose *only*
reference-significant gene was its own target now has $\tilde S_p^{\text{real}} = \varnothing$, so
a submission calling nothing significant for it reads $1.0$ where it previously read $0.0$. The
condition needs $|S_p^{\text{real}}| = 1$ before exclusion and is real-side, hence identical for
every submission; on val A the reference calls its own gene significant for 299 of 300 targets
against a mean of ~343 significant genes per target, so it is rare. It is a genuine consequence of
combining two documented rules, not an artifact.

Because the union grows with the predicted set, `sig_jaccard` is **not** gameable by flooding
significance — the failure mode §5 raises against the naive adjusted-Rand form. Measured over
five submissions of the 24-submission VCC campaign, one of them scores `sig_recall` $=0.96$ —
it recovers almost every real DEG — while its `sig_jaccard` is $0.16$ and its `sig_mcc` is
$0.02$, i.e. chance: it calls so many genes significant that recovering the real ones costs it
nothing. The other four sit at `sig_recall` $\in [0.38, 0.65]$ with `sig_jaccard`
$\in [0.18, 0.22]$. A one-sided denominator cannot separate those two situations; this one can.

It is still **chance-inflated** in the other direction, as every raw set-overlap score is: when
both sets are large relative to the panel, two random lists already share many genes.
On a well-formed DE table — one row per `(target, feature)` on each side — `sig_jaccard` and
`de_wilcoxon_sig_mcc` (§5) are the uncorrected and chance-corrected views of exactly the same
$2\times2$ table. They diverge only on a malformed table carrying duplicate rows:
`sig_jaccard` reads the de-duplicated set (which keeps it inside $[0,1]$), while the
`de_sig_agreement` family still counts rows.

---

## 5. Chance-corrected DE agreement metrics

The raw overlap/precision/recall scores (§4.2–4.3) are inflated by chance: when the
significant sets are large relative to the gene panel, two *random* gene lists already share
many genes. These four metrics (issue #14) correct for chance using **correlation-family
measures over a per-perturbation $2\times 2$ contingency table**, so they read $0$ at chance,
$1$ at perfect agreement, and negative for worse-than-chance. All are `full`/`de` only, so
neither competition ranking is affected, and all are v2-native (no v1 alias) — which means they are
**not emitted under `version="v1"`**: there is no upstream cell-eval counterpart for them to be
byte-compatible with.

For a perturbation over a gene universe of size $G$ (the union of tested genes), classify every
gene by membership in the real set and the predicted set:

|  | pred positive | pred negative |
|---|---|---|
| **real positive** | $\mathrm{TP}$ | $\mathrm{FN}$ |
| **real negative** | $\mathrm{FP}$ | $\mathrm{TN}$ |

with real positives $a = \mathrm{TP}+\mathrm{FN}$, pred positives $b = \mathrm{TP}+\mathrm{FP}$,
and $\mathrm{TN} = G - a - b + \mathrm{TP}$. The measures:

```math
\text{Informedness} = \frac{\mathrm{TP}}{a} + \frac{\mathrm{TN}}{G-a} - 1 = \mathrm{TPR} + \mathrm{TNR} - 1 \quad(\text{recall side}),
```

```math
\text{Markedness} = \frac{\mathrm{TP}}{b} + \frac{\mathrm{TN}}{G-b} - 1 = \mathrm{PPV} + \mathrm{NPV} - 1 \quad(\text{precision side}),
```

```math
\mathrm{MCC} = \frac{\mathrm{TP}\cdot \mathrm{TN} - \mathrm{FP}\cdot \mathrm{FN}}{\sqrt{a\,b\,(G-a)(G-b)}} = \operatorname{sign}(\mathrm{TP}\cdot \mathrm{TN} - \mathrm{FP}\cdot \mathrm{FN})\cdot\sqrt{\text{Informedness}\cdot\text{Markedness}}.
```

MCC is the $\phi$ coefficient (Pearson correlation of the two binary membership vectors). All
three lie in $[-1,1]$: $0$ = chance (independence), $1$ = perfect, $-1$ = perfectly anti-aligned.

| metric (v2 name) | $2\times2$ table | measure |
|---|---|---|
| `de_wilcoxon_sig_recall_adjusted` | significance membership ($p^{\mathrm{adj}} \lt T$) | Informedness |
| `de_wilcoxon_precision_adjusted` | significance membership | Markedness |
| `de_wilcoxon_sig_mcc` | significance membership | MCC |
| `de_wilcoxon_overlap_adjusted` | top-$m_r$ **ranked** membership (real vs pred top-$m_r$) | MCC |

**best $=1$** for all four. A degenerate table (any of $a,b,G-a,G-b$ equal to $0$ — e.g. no
significant genes, or a prediction that floods every gene as significant) maps to the worst
value $-1$ for every perturbation, so it is penalized rather than dropped.

Why not the naive adjusted-Rand form $(\text{obs}-\mathbb{E})/(\min(a,b)-\mathbb{E})$? It is
**gameable by flooding significance** — marking most genes significant drives the observed
overlap to $\min(a,b)$ for free, giving a spurious $1.0$. The full $2\times2$ correlation
measures count the false positives that flooding creates (via the true-negative cell) and
treat flooding as what it is: uninformative ($\approx 0$).

---

## 6. From per-metric means to a score

Both scorers turn per-metric aggregate means into a single `avg_score` by normalizing each
metric against a **baseline** run (e.g. a mean-baseline submission). They do not cover the same
metrics — see the note below the formula. For a user value $u$ and baseline value $b$:

```math
s = \frac{\text{gap}}{D},\quad
\text{gap} = \begin{cases} u - b & \text{higher is better}\\ b - u & \text{lower is better}\end{cases},\quad
D = \begin{cases} a - b \text{ or } b - a & \text{anchor } a \text{ known}\\ |b| & \text{no anchor, signed}\\ b & \text{no anchor, non-negative}\end{cases}
```

`best_value` has been replaced by a per-metric `Scoring` policy — `scored` (enrolment),
`direction`, `anchor`, `penalty`, `clamp_low`, `clamp_high`. Separating them is what let
enrolment stop being a property of the mathematics: under one token, saying a metric was
higher-is-better also claimed an anchor for it. Enrolment implies a direction; the converse is
not guaranteed by the policy, though no catalog entry occupies that state today —
`expr_mse_unbiased_norm` held it alone until it was enrolled and was later removed by #257.

⚠️ **The two scorers do not cover the same metrics.** Only the anchored branch of $D$ above is
implemented by `compat.score_agg_metrics` — it reproduces upstream cell-eval, so it scores the
**scored subset of what a v1 run can emit** (27 of the 29 emitted columns; the two
`de_*_nsig_counts_*` diagnostics are emitted but not scored) and declines everything v2-native,
all of which is either anchorless or has no upstream counterpart. The anchorless branches
($|b|$ and $b$) and the clamps below belong to `cell_eval2.score_metrics` alone.

**Anchorless metrics are clamped to $[-2, 2]$.** With an anchor $a$, the score is a fraction
of the distance from baseline to perfection and cannot exceed $1$. Without one, $D = b$ and
the score is $u/b - 1$, which nothing bounds: a near-zero baseline turns a single metric into
an arbitrarily large term (at $u/b = 100$ the raw score is $99$, more than the entire
achievable range of every other metric combined). The twelve anchorless scored metrics
therefore carry `clamp_low=-2.0, clamp_high=2.0`. For the ten with range $[0,\infty)$ the
floor is inert — $u \ge 0$ already gives $s \ge -1$; for `direction_yield`, whose baseline is
signed and centred near zero, both bounds bind.

**Four bounded metrics are deliberately unfloored.** A $[0, 1]$ metric scored against
an anchor cannot produce an unbounded term — $(u-b)/(a-b)$ is bounded below by $-b/(a-b)$ for
every valid $u$ — so the flat clip at $0$ protected nothing the metric could do, only a
*missing* value. The four bounded `vcc2026` members therefore carry `clamp_low=None` plus
`metric_min=0.0`: no clip, and a missing/NaN/overflowing value scores exactly what the worst
possible submission scores rather than $-\infty$. The policy object is
`scoring.BOUNDED_UNFLOORED`; the shared `BOUNDED` singleton that 60+ other entries use —
including every metric in the frozen 2025 `vcc` profile — is untouched and still clips at $0$.

In **`score_agg_metrics`** each score is clamped to $\ge 0$, and `avg_score` is their mean
(metrics that are unscored *or* not `v1_available` are skipped). Intuition: a submission that **equals the baseline**
scores $0$; a **perfect** submission ($0$ error, or $1$ for a higher-is-better metric) scores
$1$. The $\ge 0$ clamp is this scorer's, not a property of scoring in general — §6.1's
`score_metrics` deliberately goes negative for the error class.

The competition (`vcc` profile) scores exactly three metrics: `expr_mae`, `pds_l1`, and
`de_wilcoxon_overlap`.

The **2026** competition (`vcc2026` profile) scores a different six: `pds_cosine`,
`expr_mse_unbiased_capped_norm`, `de_wilcoxon_lfc_nmae`,
`de_wilcoxon_direction_fidelity_yield_raw`, `de_wilcoxon_direction_reach_raw`, and
`de_wilcoxon_sig_jaccard`. It **replaces** rather than extends `vcc` — a discrimination metric
under cosine instead of L1, a sampling-bias-corrected squared error instead of MAE, and
symmetric set agreement instead of rank overlap — and `vcc` itself is untouched, so the 2025
score is unchanged.

The profile contains ten catalog entries because the scored derived expression metric needs
two of them as its numerator and denominator — `expr_mse_unbiased_capped` and
`expr_distance_unbiased` — and two more ride along for auditability: `expr_mse_unbiased` (the
uncapped sibling, #257) and `expr_real_mass_ratio` (#264).
Those four appear in the result and aggregate frames for auditability but do not enter
`avg_score`, so the profile is six scored members and four diagnostics. Facts worth knowing
before reading that score:

- **Five scored members and all four diagnostics aggregate by mean; the sixth scored member
  is a ratio of sums.** `expr_mse_unbiased_capped_norm` is derived only in the aggregate frames
  as described in §2.3. `avg_score` remains a plain mean over the six scored metrics. Issue #229
  moved `direction_fidelity_yield` and `direction_reach` — the pair this profile scored at the
  time — off the DOR's median; issue #231 (v0.8.0) moved the remaining eighteen median entries.
  Thus every non-derived catalog metric says `agg="mean"`, while the derived entry says
  `agg="ratio_of_sums"`. The live median aggregation branch remains available to out-of-tree
  entries, and `aggregate_metrics_wide` still publishes a median statistic where applicable.
- **The direction pair is the RAW one as of v0.8.0** (issue #231). v0.5.0–v0.7.0 scored the
  chance-corrected `direction_fidelity_yield` / `direction_reach` here; the profile now takes
  their `_raw` siblings, and the corrected pair stays in `full`/`de`. The corrected
  `fidelity_yield`'s no-skill point is neither zero nor stable: measured over six CCL lines ×
  4 arms on two panels, its baseline runs $-0.6086$ to $-0.8948$ and moves $0.05$–$0.10$
  between panels, while `direction_fidelity_yield_raw` sits at $0.4863$–$0.5148$ across all
  twelve line × baseline cells — empirically the theoretical random point — and moves
  $\le 0.012$ between panels. Both raw entries already carried `anchor=1`, `clamp_low=0`,
  `penalty="none"` and `scored=True`, identical to the pair they replace, so `avg_score`'s
  formal range is unchanged. Known consequence, accepted: `vcc2026` no longer charges a
  submission for predicting each gene's habitual direction, and `direction_reach_raw` carries
  the anti-gaming load alone — an abstaining predictor drives bare `fidelity_raw` to $0.9999$
  but is held to $0.005$–$0.05$ on `reach_raw` (measured at $P_0 = 0.975$ and not re-run at
  $0.9$; the relevant evidence that it still holds is that the chance floor barely moved — val A
  exactly unmoved, B and C up by $0.0017$ and $0.0013$).
  ⚠️ **`reach_raw`'s own no-skill point is $\approx c/N_{\mathrm{conf}}$, not a constant**
  (#279). A match on the FIRST
  adjudicable pair in the prediction's own ranking is *sufficient* for $k^* \ge 1$ — one coin
  flip, independent of depth — so a no-skill submission scores $\approx 1/N_{\mathrm{conf}}$
  about half the time at *every* depth. ⚠️ It is **not necessary**, and #279's "iff" is wrong:
  purity recovers, so a leading miss followed by 9 matches gives
  $\mathrm{purity}(10) = 9/10 = 0.9 \ge P_0$ and $k^*$ reaches $10$ (built end to end, that
  fixture returns $1.0$). The empirical claim survives because the no-skill mean is dominated by
  the $k^* = 1$ event, not by recovery: at $P_0 = 0.9$ the **earliest** recovery opportunity is
  depth 10, which qualifies on at least 9 of the first 10 calls — $11/1024 = 1.1\%$ for a coin
  flip — and the resulting $10/N_{\mathrm{conf}}$ is small against the budget. ⚠️ That is the
  probability the depth-10 *prefix* qualifies, not of recovery in general: 8 of 10 fails at
  depth 10 and can still qualify at depth 20 on 18 of 20, so it is a **lower bound** on total
  recovery. At $0.975$ the corresponding first opportunity is depth 40 on at least 39 of 40,
  $41/2^{40} \approx 3.7\times10^{-11}$ — eight orders of magnitude rarer. That is why the
  measured chance floor **barely** moved when $P_0$ went $0.975 \to 0.9$ — val A exactly
  unmoved, B and C up by $0.0017$ and $0.0013$.
  Measured at 0.13.0, 200 replicates per cell:
  $\mathbb{E}[\texttt{reach\_raw}]$ runs $0.5100$ at $N_{\mathrm{conf}}=1$ down to $0.0019$ at
  $500$, with $P(\texttt{reach\_raw} > 0) = 0.385$–$0.565$ at every $N_{\mathrm{conf}}$, and
  the fitted $c$ is $\approx 0.96$ no-skill against $\approx 1.90$ for a submission that ranks
  one *habitual-direction* gene first. **The calibration half of that issue is closed by
  #276 part C**: scoring is against a *measured* baseline, and on official val A the baseline
  arm reads $0.042314$ against an estimated no-skill $0.96 \times \overline{1/N_{\mathrm{conf}}}
  = 0.96 \times 0.039343 = 0.0378$ — i.e. the baseline already sits at the no-skill point,
  which the scale's old constant `base = 0.0` did not. **The exploit half survives**: the
  habitual-direction head estimates at $0.0748$, $+0.038$ of the member's span above that
  baseline (val A's span for this member is $0.9044 - 0.0423$, the frozen replicate anchor over
  the frozen baseline), **$+0.006$ on `avg_score`** for no target-specific skill. ⚠️ That
  $0.0748$ is an *estimate* from the fitted $c/N_{\mathrm{conf}}$ law applied to val A's own
  $N_{\mathrm{conf}}$ distribution, not a direct simulation on the panel. What bounds it is panel
  construction, not the metric — val A's smallest $N_{\mathrm{conf}}$ is $4$ and no target
  falls in the $N_{\mathrm{conf}} \le 3$ band where the head flip is worth $0.29$–$0.51$;
  $12.3\%$ of its targets sit at $4 \le N_{\mathrm{conf}} \le 10$, median $46$. A
  `min_n_conf` gate or a $k^* \ge 2$ rule would remove it, and both move scored numbers.
- **Five of the six members carry downside risk.** `de_wilcoxon_lfc_nmae` can score in
  $[-6, 1]$ *on the baseline scale* — on the replicate scale its top end is the geometry's,
  $1 + a/(b-a)$, which is $1.565$ on official val A. It was the only member that could go
  negative when its shape was chosen, which is why it, and only it, was moved off the Box–Cox
  tail onto a straight line to the same $-6$ (§6.1): its shape *was* the average's entire
  downside. ⚠️ It no longer is — the clip-at-0 removal below gives four more members a live
  sub-zero range.

  `expr_mse_unbiased_capped_norm` is bounded to $[0, 1]$ on both scales (#276 part C, Alex
  2026-08-13) and is the one member that clamps at *either* end. The other four —
  `pds_cosine`, `de_wilcoxon_direction_fidelity_yield_raw`, `de_wilcoxon_direction_reach_raw`,
  `de_wilcoxon_sig_jaccard` — are **unfloored**: they keep `clamp_high=None` and may exceed 1
  on the replicate scale, and since the clip-at-0 removal they may also go below 0. Their floor
  is not a constant but the score their own structural worst value ($u = 0$) earns, $-b/(a-b)$:

  | member | val A | val B | val C |
  |---|---:|---:|---:|
  | `pds_cosine` | $-1.17$ | $-1.26$ | $-1.22$ |
  | `de_wilcoxon_direction_fidelity_yield_raw` | $-1.58$ | $-2.68$ | $-1.85$ |
  | `de_wilcoxon_direction_reach_raw` | $-0.05$ | $-0.12$ | $-0.12$ |
  | `de_wilcoxon_sig_jaccard` | $-0.08$ | $-0.07$ | $-0.10$ |

  (Measured on the three official val bundles, replicate scale. ⚠️ The `-r1` set predates this
  release wave, and two of the four changed what they compute, so their $b$, $r$ and hence these
  floors shift when the #317 wave rebuilds — but **not both for the same reason**:
  `sig_jaccard` moved in #172, whereas `direction_reach_raw` already excluded its target gene
  before #172 (Δ = 0 in that section's own table) and moves here because of the purity-floor
  change. The
  mechanism does not: $-b/(a-b)$ is deepest where the comparator sits nearest the replicate.)

  The six-member `avg_score` therefore **no longer floors at a constant** $-6/6 = -1.0$: it
  floors at $-1.48$, $-1.69$ and $-1.55$ on those three arms. `ERROR_LINEAR`'s declared
  $-6$ still contributes exactly $-1.0$ of that; the rest is the four unfloored members, and it
  moves with the panel. ⚠️ `DEFAULT_PENALTY_CAP = 6.0` was chosen to produce the old constant
  and no longer carries that meaning either — since `ERROR_LINEAR` declares its floor outright,
  the cap now governs only the `ERROR` class and a call-time `penalty="boxcox"` override.
  The depth is set by how far the *replicate* clears the comparator —
  `fidelity_yield_raw`'s comparator is pinned at chance ($b \approx 0.5$), so its floor scales
  as $-0.5/(r - 0.5)$ and deepens sharply as $r \to$ chance.
- **All six scored members are decisive** (§6.1), so a degenerate baseline for any of them
  fails loud. The four unscored expression diagnostics are not decisive; that is inert because
  every baseline/scoring consumer skips an unscored metric before asking about decisiveness.
- **Only one of the ten aggregate entries has a data ceiling.** `ceiling.SB_METRICS` (§6b)
  covers `pds_cosine` and none of the other nine, so a `--ceiling` run reports `NaN` for 9/10
  of this profile.
  `NaN` there means "no defensible ceiling", not "zero".
- ⚠️ **Under `version="v1"` the profile collapses to `pds_cosine` alone.** Nine members are
  v2-native, and a not-v1-available name arriving via a *profile* is filtered silently (§1.5),
  so `--profile vcc2026 --version v1` scores one metric and still calls itself `vcc2026`. Use
  `vcc2026` with the v2 default; v1 exists to reproduce upstream cell-eval, and a 2026 profile
  has nothing there to be compatible with.

### 6.0 What the six members do with a perturbation that has no recoverable signal

#238 asked whether the six `vcc2026`
members should agree about a perturbation the reference could not adjudicate — one where
$N_{\mathrm{conf}} = 0$, i.e. no reference-significant budget *after target-gene exclusion*.
(Not "nothing is knowable": reference log₂FC **directions** still exist outside the significant
set, which is exactly what the loud model's $0.473684$ below is graded against.) **They do not
agree, and the ruling
(2026-08-15) is that they do not have to.** Each convention is right for its own metric; what
was missing was a single place saying what they are. This is that place.

Measured end to end on a synthetic panel under the competition preset, reading the
**per-perturbation** frame so an omission is visible as an absent row rather than as a value.
Two null perturbations, identical in the reference and differing only in what the model did:
`quiet` makes no significant calls, `loud` makes many.

| member | reference has signal | null, model **quiet** | null, model **loud** | convention |
|---|---:|---:|---:|---|
| `pds_cosine` | 1.000000 | **0.500000** | **0.000000** | midrank tie — a zero predicted delta ties every competitor (#282) |
| `expr_mse_unbiased_capped_norm` | — | — | — | **no per-perturbation value exists**; it is a ratio of sums formed in the aggregate frame (#257) |
| `de_wilcoxon_lfc_nmae` | 0.037491 | **omitted** | **omitted** | empty gate, and `min_gate_size = 10` besides; real-side only, so the omission set is identical for every submission |
| `de_wilcoxon_direction_fidelity_yield_raw` | 0.994962 | **NaN** | **0.473684** | $k/\max(n_{\mathrm{pred}}, N_{\mathrm{conf}})$; both zero ⇒ $0/0$ ⇒ NaN, dropped from the mean |
| `de_wilcoxon_direction_reach_raw` | 1.000000 | **NaN** | **NaN** | $N_{\mathrm{conf}} = 0$ ⇒ the pool is empty ⇒ NaN, dropped from the mean |
| `de_wilcoxon_sig_jaccard` | 0.989975 | **1.000000** | **0.000000** | empty union ⇒ $1.0$ (#215) — agreement on "nothing happened" *is* agreement |

Four different answers to one input: a **midrank 0.5**, a **maximum**, a **NaN that vanishes
from the aggregate**, and an **omission decided before the prediction is read**. Two of them are
flatly opposed — on the quiet model `sig_jaccard` awards its maximum while
`fidelity_yield_raw` awards nothing at all.

Three things follow that are easy to get wrong:

- ⚠️ **The 31.9% figure on #238 was measured on a metric this profile does not score.**
  `direction_fidelity_yield` — the *chance-corrected* member — returns $1.0$ when
  $N_{\mathrm{conf}} = 0$ and the model also called nothing, and on the arm measured there that
  branch covered 31.9% of a weak tertile. `vcc2026` scores `direction_fidelity_yield_raw`,
  which returns **NaN** on exactly that cell. Check which member of a raw/corrected pair a
  result names before carrying its number across.
- **NaN is not a free win, and it is not a penalty either.** With `agg="mean"` a NaN
  perturbation leaves the aggregate's denominator, so the arm is scored on the remaining
  non-NaN perturbations. That is a *smaller* cohort, not a better score.
- **Being loud on a null perturbation is not free either.** $0.473684$ is a coin flip against
  noise, and it lands just below `low-random_high-1_v10`'s $0.5$ base for that member (§6d) —
  so the loud model buys a slightly negative contribution where the quiet one buys none.

`de_wilcoxon_sig_jaccard`'s maximum is the one that most looks like a free win, and
#243 closed the question by measuring
it: on the official val A bundle the set-size axis is worth 0.2% of the span, and padding a real
submission is *punished* — a skilled arm loses 0.26 of the span for ten extra calls per target.

⚠️ **#172 widened who lands in the
"null" columns**, for two members and by exactly one gene. `sig_jaccard` and `lfc_nmae` now
exclude the perturbed gene's own row (§4), so a perturbation whose *only* reference-significant
gene was its own target reaches those two as an empty set — the `1.000000` and the omission above
— where before #172 it had a budget of one. That is the same convention applied to a slightly
larger population, not a new convention; it needs $|S_p^{\text{real}}| = 1$, and it is decided
real-side, so the population is identical for every submission. The four already-excluding members
were always read on the post-exclusion budget, which is why this row of the table did not move for
them.

### 6.1 Penalizing error metrics below baseline (`score_metrics`)

`score_agg_metrics` above clamps every score to $\ge 0$, so a submission *worse* than the
baseline ties one that matches it, at $0$ — no downside risk. `cell_eval2.score_metrics` is a
sibling scorer that is **bit-identical** to `score_agg_metrics` on the metrics they both
score — the v1-available, anchored ones — for every submission at or below baseline **and no
degenerate baseline**, but replaces the flat clamp for the unbounded
error metrics (`direction = lower`: `expr_mae`, `expr_mse`,
`expr_mse_unbiased_capped_norm`, `delta_mae`, `delta_mse`) with a capped Box–Cox penalty. With
$r = u/b$, exponent $p = 2$, cap $C = 6$ (both config knobs; $C$ was 10 through v0.12.0 —
#276 part C retuned it so the six-member `vcc2026` average floors at exactly $-1$):

```math
\text{score} = \begin{cases} 1 - r & r \le 1 \\[2mm] \max\!\left(-C,\; -\dfrac{r^{p}-1}{p}\right) & r > 1 \end{cases}
```

The tail is $C^1$ at $r = 1$ (slope $-1$ on both sides), strictly decreasing, and saturates at
$-C$ for $r \ge \sqrt{1+2C} = \sqrt{13} \approx 3.606$. A non-finite or missing *user* value
(NaN/inf MAE) takes the cap $-C$ (no-droppable-NaN, §1.5).

**`de_*_lfc_nmae` takes a straight line to the same floor** (`scoring.ERROR_LINEAR`; Alex,
2026-08-17). Same anchor, same direction, same $-6$ floor — the shape between them is
$\text{score} = \max(-6,\, 1 - r)$ with no penalty term, so it reaches the floor at $r = 7$
where the quadratic tail had already saturated at $r \approx 3.606$. Submissions between those
two ratios are ranked rather than tied at the floor.

Why this member and not the four that keep `ERROR`: it is the only `vcc2026` member with a live
sub-zero range — the other five clip at $0$ — so its shape alone decided the whole downside of
the competition average, and a quadratic made that average a near-binary test of *escaping the
floor* rather than a ranking. `expr_mae`, `expr_mse`, `delta_mae` and `delta_mse` keep the
Box–Cox tail, and `expr_mae` is in the frozen 2025 `vcc` profile, whose published scores must
not move.

It is also the shape the frozen `low-random_high-1_v10` scale (§6d) already used for this metric,
so catalog and scale now agree on SHAPE and on the perfection anchor ($0.0$ for both). What
differs is the base — the constant $1.0$ there, a measured baseline here — and the clamps:
$[-1.0,\, 1.0]$ there, $[-6.0,\, \infty)$ here.

Measured on the three official val bundles, the discriminating band widens from
$\mathrm{nmae} \le 2.67 / 2.39 / 2.63$ to $\le 4.83 / 4.20 / 4.74$; last year's 339-submission
field has a median $\mathrm{nmae}$ of $22.9$ against a baseline of $0.95$, so it saturates under
either shape. ⚠️ Those six boundaries are $a + \sqrt{13}(b-a)$ and $a + 7(b-a)$ over the
`vcc2026-val{A,B,C}-r1` replicate/baseline pairs as those artifacts stand — built **before**
#172 removed the perturbed gene's own row
from this metric, so a rebuilt bundle's $a$ and $b$ differ and these exact numbers will not
reproduce. What does not move with the rebuild is the ratio the boundaries sit at, $r = \sqrt{13}$
and $r = 7$; quote those instead if the bundle generation is in doubt. A degenerate *baseline* — one whose
denominator $D$ is not a **finite positive** number, or whose $b$ is itself non-finite —
  **fails loud** for every scored metric that a v1 run can emit or that the `vcc` or `vcc2026`
  profile scores, where a wrong number decides a ranking and scoring against an undefined
  denominator is worse than stopping. Unscored metrics are skipped before this predicate is
  asked. This includes $b = 1$ on an anchor-1 metric, which previously scored every submission
  exactly $0$, silently. The bounded (anchor-1) metrics otherwise keep the $\ge 0$ clamp, except
  the four unfloored `vcc2026` members, whose missing-value sentinel is `metric_min` instead.

For any other scored metric — a large class (see `catalog.is_decisive`) — a degenerate baseline
logs a warning and that one metric is excluded from `avg_score`, rather than aborting the run —
so the aggregate is a mean over a different metric set and is not comparable with a run where it
scored. Every scored `vcc2026` member is decisive, so none takes this branch. If that leaves
*nothing* scoreable it raises, rather than reporting the fallback $0$. `de_*_direction_yield` is why: it is signed and
centred at zero *by construction* — it evaluates to exactly $0$ when the model calls nothing
($n_{\mathrm{pred}} = 0$), so a baseline that calls nothing for a
majority of perturbations lands on exactly $0$ legitimately. Before it was enrolled this could
not arise; letting it abort the other 86 metrics is the wrong trade.
`score_agg_metrics` keeps its own clip-at-0 arithmetic for upstream parity, and scores only the
**v1-emitted scored subset**.

---

## 6a. The generic-response baseline

A raw metric value is uninterpretable without a comparator. `cell-eval2 baseline` builds
the predictor that supplies one: a model handed the **average perturbation response** and
nothing target-specific.

    cell-eval2 baseline -ar real.h5ad -o baseline/ --profile full
    cell-eval2 run -ap pred.h5ad -ar real.h5ad -o user/ --profile full
    cell-eval2 score --user-agg user/agg_results.csv \
                     --baseline-agg baseline/baseline_agg.csv

**It is an oracle comparator, not a floor a submission could reach.** The profile is
averaged from the *evaluated* real perturbations, so it is transductive: a real submission
cannot know that oracle profile. The emission itself is executable — normalization is public
and participants received the control pool — so the oracle property comes from the profile,
not a private construction. What it bounds is the *triviality* of a metric: even a predictor
handed the true average response, with no target-specific information whatsoever, scores X,
so anything at or below X is not evidence of target-specific skill. Do not read it as "the
score a model gets for free".

For each gene, the profile is the mean per-perturbation pseudobulk over all non-control
perturbations, **omitting the perturbation that targets that gene** (`--no-exclude-target-gene`
turns the omission off). Averaged in the reference's own matrix space; equal weight per
perturbation, not per cell.

By default (`--emit dispersed`), each non-control group resamples the reference's control
cells with replacement and scales each sampled cell gene-wise by
`profile / control_pseudobulk`. In **counts** space, conditional on that fixed control pool,
the expected group mean equals the profile exactly on genes with control support before
float32 rounding. This does not make the lognorm group mean equal the lognorm profile: the
measured cost is +2.8% (0.012568 versus 0.012226 for the profile compared directly without
cells) — ⚠️ all three figures on **pre-#247 `expr_mse_unbiased`**, in squared expression units.
The current `expr_mse_unbiased` restores that diagnostic bit-for-bit, so the comparison is on
its shipped units again. It does not transfer quantitatively to
`expr_mse_unbiased_capped_norm`, whose cap can change the numerator and whose panel value is a
ratio of sums against the debiased real-side denominator; only the qualitative claim survives.
`--seed` controls the resampling (default 0). `--emit tile` selects the legacy constant-cell,
known-biased arm solely to reproduce pre-fix numbers; it is never selected implicitly. Both
values are recorded in `baseline_meta.json`.

⚠️ **Which arm `expr_mse` prefers INVERTED in #264, and this is the change most likely to
surprise.** Measured on a frozen 85.6%-zero over-dispersed fixture, on `expr_mse`
specifically — the mechanism is a property of the comparator and so applies to the metrics that
read the submission only through $b_p$ (`expr_mae`, `expr_mse`, `pds_*`, `delta_*`), but only this
metric was measured on this fixture. ⚠️ It does **not** carry to the unbiased family unchanged:
those subtract $C_p$, which reads the cell layout and moves the *other* way on the tiled arm — see
§1.2 and §2.3, and #278 for the measurement:

| comparator | tiled `expr_mse` | dispersed `expr_mse` | preferred |
|---|---:|---:|---|
| `lognorm` (pre-#264) | 40.6× the dispersed arm | 1× | **dispersed**, by 40.6× |
| `bulk_lognorm`, $\mathrm{TS}=5\times10^{4}$ (shipped) | 0.8010 | 5.9414 | **tiled**, by ~7.4× |
| `bulk_lognorm`, $\mathrm{TS}=10^{6}$ (retired) | 1.0348 | 12.3205 | **tiled**, by ~11.9× |

That is the point of the change, not a regression. Under `lognorm` the comparator is a
dispersion functional (§1.2), so a tiled arm was penalized for a property the metric never
claimed to measure — the real control itself scored 13× when tiled (#258, #260). Under
`bulk_lognorm` the bulk comes from the group sum, which a tiled arm reproduces exactly, so the
pathology is gone. **The cost, stated so nobody has to rediscover it: a degenerate
zero-dispersion submission is no longer penalized by `expr_mse`.** That penalty was an
artifact, but it was load-bearing in practice (#234, #259). **DECIDED (Alex, 2026-08-11): that
is accepted, and no replacement guard is being added.** `expr_mse` never claimed to measure
cell-to-cell variation, and the one metric family that would — a cell-to-cell variance metric —
was measured at a +2.0% ceiling against the mean's +14.5% and rejected on that basis. Anything
that needs to grade emission realism has to make that case on its own evidence, not inherit it
from a comparator artifact. `--emit dispersed` remains the default, because it
is the arm that is unbiased in counts space and the one every other metric family is
calibrated against.

**Coverage.** The baseline is scored as an ordinary submission through the unmodified
metric layer, so it covers every metric the configured DE backend can produce — including
metrics added later, for free. The one carve-out is `de.backend="deseq2"`, which
`cell-eval2 baseline` **rejects**: the profile is a mean, so the prediction's pseudobulk is
fractional, and fractional input to a negative-binomial GLM is not statistically meaningful.

**Which comparator this is.** Because it is scored as an ordinary submission, it is graded
on **its own** significant gene set wherever a metric conditions on the prediction — which is
`de_wilcoxon_direction_precision` and `de_wilcoxon_model_direction_match`, but *not*
`de_wilcoxon_direction_match` (conditioned on the reference-significant set) or
`de_wilcoxon_direction_sensitivity` (a reference-adjudicated denominator). Where it applies,
that is the *self-selected* comparator, which is **weaker** than grading the baseline on the model's gene set: a generic
profile's most significant genes are the strongly co-regulated ones that move the same way
under nearly every perturbation, so it self-selects into the easy part of gene space.
Precision is also a point on the purity curve — since #204 at depth
`k = |S ∩ adjudicable|` rather than `k = |S|` — so the two numbers are purity at different
depths. Published margins must say which comparator they used.

**Nothing about `control_source` is overridden.** The prediction's control rows carry the
real control's own cells, so `control_source="pred"` and `"real"` give the same numbers and
the baseline is convention-neutral. (Filling those rows with the profile instead — as a
naive constant predictor would — makes the predicted delta identically zero and the DE fully
tied; that is a bug, not a floor.)

**The matrix space is locked.** Under v1, and under v2 with `autodetect_input_type`, the
pipeline re-detects each side's convention independently, and the reference and the
fractional profile can land in different spaces in either direction. `cell-eval2 baseline`
resolves both sides, reconciles them when it can, and **fails loud** when it cannot
(`allow_discrete_effective` and `input_type_*_effective` in `baseline_meta.json` record the
outcome). Dispersed emission is defined only for a counts-effective reference; when the
requested config resolves the reference as lognorm, the builder rejects it and names
`--emit tile` as the legacy arm that still applies.

**Pair the right two runs.** `run` writes `run_meta.json` beside `agg_results.csv`, and
`score` compares it against `baseline_meta.json`: the code version, the scoring-config digest,
the resolved DE backend and device (`de.backend="auto"` picks a different engine per host, and
the device selects fp32-GPU vs fp64-CPU means), the reference's fingerprint — metadata-level
(shape, dtype, `var` index, per-cell labels) unless both runs used `--cache-strict`, and `score`
says which level it compared — both sides' effective input types, and the fingerprint of any *supplied* DE table. The check is fail-closed —
a missing field is a mismatch — and it reports **not verified** rather than verifying something
weaker when `run_meta.json` is absent. `--allow-config-mismatch` downgrades any of those to a
warning.
`score` also re-checks the degenerate-baseline gate on the `--comparison-statistic` it was
given, since the build-time gate validates `mean`.

**`n_excluded` is worth reading.** The target-gene omission matches perturbation labels
against gene names by exact string equality. When labels are guide IDs it matches nothing
and silently does nothing; `baseline_meta.json`'s `n_excluded` is the proof either way.

**A degenerate baseline fails loud.** Before returning, every scored metric's aggregate is
checked: a non-finite value, or one whose denominator `D` is not a finite positive number, is rejected —
`b <= 0` against an anchor of 0, `b >= 1` against an anchor of 1, `b == 0` for a signed
anchorless metric. `--allow-degenerate-baseline` writes the artifact anyway and records the
offenders in the stamp, each tagged `decisive`. What that buys depends on which metric is
  degenerate: `score_metrics` still **refuses** an artifact degenerate on a metric v1 can emit or
  that `vcc`/`vcc2026` scores, so for those the waiver is diagnostic-only; an artifact degenerate
  only on other scored metrics is **scored with those metrics dropped** from `avg_score` (and
  refused if that leaves nothing).

**Scale.** Dispersed prediction mirrors the template's sparsity: a sparse template produces
CSR float32, a dense template produces a dense float32 array, and the dense
`n_obs × n_genes` construction intermediate is gone for sparse references. On the VCC
reference, measured peak memory fell from **102 GiB** for tile to **47 GiB** for dispersed —
but read that pair narrowly: **both figures were measured through a separate offline
builder**, which emits group-major rows and no control block.
`build_generic_baseline` mirrors the template instead, so it additionally `vstack`s the
control rows and applies a row permutation, each an O(nnz) copy; **its own peak on VCC is
unmeasured and is higher than 47 GiB.** Nor is this a general sparsity escape: at roughly
48% density, float32 CSR plus int32 indices costs about as much as dense float32, and
compressed archive size says nothing about RAM. `build_generic_baseline` still materializes
both sides and calls the in-memory scorer, so CCL_2 scale remains **unvalidated** and
O(nnz)-limited. The dense-specific VCC ceiling is removed for sparse references; the scale
ceiling is not gone.

Issue #191's motivation — a scalable dispersed path — remains valid, but its proposed
mechanism must not be implemented as written. Injecting one broadcast pseudobulk vector at
the scoring seam would permanently bake in normalization-before-averaging's order-of-
operations defect and make `tr(Sigma-hat_pred) = 0` structural. The previously reported
**65.2 GiB** peak and **3–7 min** runtime on a 170,846 × 18,080 reference were measured on
the old tiled build, not the current dispersed default; the old full validation run (both
arms, a submission, scoring) took **9–17 min**.

**What the target-gene omission actually buys — measured.** Both arms were run on a real
reference. Read *both* of the numbers below together, because either one alone misleads:

- At the profile level the omission removes a real leak: the ungated on-target direction call
  drops from **64/100** to **47/100**, and all **17** flips remove a correct call while none
  introduce one — off a median profile shift of just −0.92 %.
- In the aggregate metrics that shows up at the **3.9e−6 – 1.2e−3** level. Not because the effect is
  small per perturbation: for `direction_precision` 97 of 100 perturbations move, 45 down and 52
  up, and the net is 0.87 % of the gross. It is **cancellation**, so the aggregate cannot be used
  to infer how much of the leak was scored.

So the omission is justified by the leak, not by the metric diff. A reader who sees only the
aggregate table will conclude the flag is ceremony; it is not.

---

## 6b. The data ceiling

§6a supplies a comparator at the trivial end. The **data ceiling** supplies one at the other end:
an *estimate* of how high a model could score given the sampling noise in the real data itself. It
is computed from the real data alone — no prediction enters it. It is an empirical
reproducibility estimate, **not a proved upper bound**: it is measured on one random split, and a
model that denoises the reference better than a half of it does can in principle score above it.
Read it as "what this metric can resolve at this depth", not as a hard maximum.

    cell-eval2 run -ar real.h5ad --ceiling -o ceil/ --profile full
    # -> ceil/ceiling_results.csv (per-perturbation self-split), ceil/ceiling_agg.csv (metric, ceiling)

Equivalently `compute_ceiling(real, *, config=None, seed=0) -> (results, agg_ceiling)`.

**Method.** For each perturbation (and the control), shuffle its cells with `seed` and split them
into two **disjoint** halves of $\lfloor n/2 \rfloor$ cells — the leftover cell of an odd $n$ is
discarded so both halves sit at the same depth, and a perturbation with $n < 2$ is dropped from
both halves with a warning (if the *control* is the one that cannot be split, that is a fatal
`ValueError`: with no control cells in a half there is nothing to compute an effect against).
Half B is then scored *as a prediction of* half A through the ordinary `compute_metrics` path, so
normalization, DE and device match the main run. The resulting per-metric mean $r$ is a split-half
reliability at **half** depth. Spearman–Brown maps it to the combined depth of the two halves,
$2\lfloor n/2 \rfloor$ — the run's full depth for even $n$, one cell short of it for odd $n$:

```math
r' = \frac{2r}{1+r}
```

This is the $k = 2$ case of the Spearman–Brown prediction formula, $r'_k = kr/(1 + (k-1)r)$: it
says what reliability a test of $k$ times the length would have, under the assumption that the
added measurements are **parallel** — same true score, same error variance, independent errors.
Cells drawn from one perturbation approximate that, which is what licenses transferring the
formula from test length to cell count; it is an approximation to the extent the cells are *not*
exchangeable (donor, batch or cell-state structure inside a perturbation correlates the halves).
Note also that the classical derivation is about a reliability *correlation*: for the ranking and
set-overlap metrics in the list below, applying it is an empirical extrapolation of the same
"agreement rises with depth" shape rather than a theorem.

**The correction is applied only for $r > 0$.** Outside that range it stops being a correction:
$2r/(1+r)$ is undefined at $r = -1$ (and $r \to -1^{+}$ sends it to $-\infty$), and for any
$r \le 0$ its output has no reliability interpretation to extrapolate. The sign-unbounded
reliability metrics (`delta_pearson`, the two Spearman metrics) can land there on a small or
degenerate context. Non-positive reliability means the split gives **no positive evidence of
repeatability** — the halves agree no better than chance, or disagree systematically — so there is
no defensible ceiling. It is reported as `NaN`: never a negative "ceiling" below every achievable
score, and never a raise.

**Which metrics get a ceiling.** An explicit, hand-maintained list of 18 reliability metrics
(`cell_eval2.ceiling.SB_METRICS`), each verified 1:1 against the validated cell-eval
implementation the correction was justified on — same computation, not merely the same name:
`delta_pearson`; `pds_l1`/`pds_l2`/`pds_cosine`; `de_wilcoxon_overlap` and its four `_top{k}`
variants; `de_wilcoxon_precision` and its four `_top{k}` variants; `de_wilcoxon_nsig_spearman`;
`de_wilcoxon_lfc_spearman`; `de_wilcoxon_direction_match`; `de_wilcoxon_sig_recall`. Every other
metric the run *emits* is `NaN` (a selected-but-unimplemented metric such as
`edistance_pearson` is dropped by `resolve_metrics` and appears in neither frame).
Deliberately excluded, and not to be added without re-verifying and re-validating:

| excluded | why |
|---|---|
| `de_wilcoxon_pr_auc`, `de_wilcoxon_roc_auc` | cell_eval2's pred p-value floor (`min_nonzero`, or `replace_zero`) differs from cell-eval's clip-to-$10^{-10}$, so the AUC is a different number than the validated one |
| error metrics (`expr_mae`/`expr_mse`/`expr_mse_unbiased`/`expr_mse_unbiased_capped`/`expr_distance_unbiased`/`expr_mse_unbiased_capped_norm`/`delta_mae`/`delta_mse`/`de_*_lfc_nmae`) and `de_*_nsig_counts_*` | not bounded reliabilities; SB does not apply |
| the v2-native chance-corrected metrics (§5), the #187 direction metrics (§4.3), and the set metric (§4.5): `overlap_adjusted`, `precision_adjusted`, `sig_recall_adjusted`, `sig_mcc`, `direction_precision`, `direction_sensitivity`, `direction_sensitivity_universe`, `sig_jaccard` | no cell-eval equivalent, never validated under doubling; `direction_sensitivity_universe` is unbounded above, which SB's bounded-reliability assumption rules out outright |
| `de_wilcoxon_model_direction_match`, `de_wilcoxon_lfc_spearman_pos`, `de_wilcoxon_lfc_spearman_neg` | these *do* carry v1 aliases, but they were not part of the validated set the correction was justified on — excluded for the same "not verified 1:1" reason, not for being v2-native |
| every `de_deseq2_*` name | the deseq2 backend's DE metrics are unverified against the validated implementation |

Together those rows cover every metric the `full` profile emits under a rank/wilcoxon DE backend:
53 metrics, 18 corrected and 35 `NaN`. Under `de.backend="deseq2"` all 41 DE metrics are relabeled
`de_deseq2_*` and drop out, leaving **4 corrected** (`delta_pearson`, `pds_l1`/`pds_l2`/`pds_cosine`)
and 49 `NaN`.

**`control_source` is forced to `"pred"` on the inner run**, whatever the caller's config says.
Under `"real"` the pred side takes its DE reference from the real side, so scoring half B against
half A computes *both* halves' log₂FCs against half A's control: their sampling noise is shared
rather than independent, which correlates the two quantities whose agreement is being measured and
biases reliability upward. Measured on a small self-split, `lfc_spearman` went
$0.54 \to 0.74$ and `nsig_spearman` went from `NaN` (no defensible ceiling) to a confident-looking
$0.94$ — a shared control can manufacture reliability where there is none and defeat the $r > 0$
guard. The inner run also clears `outdir`/`cache_real`/`cache_pred` so it cannot overwrite the
caller's `run_params.yaml` or a prebuilt cache.

**Convention caveat.** The correction was validated on cell-eval numbers, which correspond to
cell_eval2's *v1-equivalent* conventions (`control_source="pred"`, PDS `rank_denominator="n"`,
`nan_lfc_policy="keep"`, `min_abs_log2fc=0`). Under v2 defaults the same algorithms run with
different conventions, so absolute values differ, and the extrapolation is applied to them on the
same empirical footing rather than a separately validated one — those conventions preserve the
"agreement rises with depth" shape SB is being used for, but were not themselves checked against
cell-eval numbers. But note the
consequence of the `control_source` override: against a main run that uses
`control_source="real"`, the ceiling is not measured under identical conventions. That is the
intended trade-off — a biased estimator is worse than a convention mismatch, and an
upward-biased ceiling understates every score measured against it.

**Cost.** The ceiling phase loads the real matrix unbacked and holds it alongside two `.copy()`
halves whose rows sum to (about) another full matrix, so its own input-matrix footprint is roughly
$2\times$ the real matrix. When the selected SB metrics include DE ones, DE is computed twice —
once per half — and, in a combined run whose main phase computes its own DE rather than consuming
supplied tables, that is on top of the main run's. In the CLI the two phases are *sequential*
(`compute_metrics` closes path-backed inputs before `compute_ceiling` opens the real data again),
so a combined `run --ceiling` peaks at about the larger of the two phases rather than their sum;
passing in-memory AnnData objects to the Python API keeps the caller's copy resident and the two
do add up. Wall time for a combined run over a DE-bearing profile often approaches $2\times$ the
plain run; it does not when the main phase is cheap (cached or supplied DE tables) or when few SB
metrics are selected.

**Reading it.** `ceiling_agg.csv` carries one row per emitted metric, spelled the way the run
emits it (v1 aliases under `--version v1`, `de_deseq2_*` under that backend), so its `metric`
column lines up name-for-name with `agg_results.csv` and joins many-to-one onto the long-form
`results.csv`. A ceiling well below 1 is a statement about the depth of the reference, not about
any model: on the tutorial's 600-cell toy reference, `de_wilcoxon_overlap` ceilings at
$\approx 0.30$ while `pds_l1` reaches $\approx 0.95$. Tutorial walkthrough: §2c of
[`tutorial.md`](tutorial.md).

---

## 6c. The `de_lfc_nmae` replicate reference and the scaled score

`de_lfc_nmae` is a valid ranking instrument on its own, but its **level** is a property of the
evaluation data rather than of the model: the same submission reads better on deeper data. §6b's
ceiling answers the same question for the bounded reliability metrics; this is the error-metric
analogue, and it is deliberately a *separate* code path rather than an extension of `ceiling.py`
(see "Why not the ceiling" below).

**The reference.** `cell-eval2 run -ar real.h5ad --lfc-nmae-ref [--lfc-nmae-ref-seed N]`, or
`compute_lfc_nmae_reference(real, *, config=None, seed=0, de_real=None)`, splits the real cells
into two disjoint halves — the same `ceiling._disjoint_halves` split, control group included, so
each half necessarily uses its own control cells — computes a DE table on each, and measures how
well one half's log₂FCs reproduce the other's:

```math
\mathrm{nmae\_ref\_raw}(p) = \frac{\mathrm{mean}_{g\in S_p^{\text{real}}}\big|\mathrm{lfc}_A(g) - \mathrm{lfc}_B(g)\big|}{\mathrm{mean}_{g\in S_p^{\text{real}}}\big|\mathrm{lfc}_{\text{real}}(g)\big|}, \qquad \mathrm{nmae\_ref}(p) = \frac{\mathrm{nmae\_ref\_raw}(p)}{\sqrt{2}}.
```

**The gate and the denominator come from the FULL real table**, not from either half — only the
numerator's two vectors come from the halves. That is why this needs *three* DE tables and is not
a shape `compute_ceiling` (which scores half B against half A and never sees the full-depth table)
can express.

**The $\sqrt{2}$.** With equal independent halves $\mathrm{Var}(e_{\text{half}}) =
2\,\mathrm{Var}(e_{\text{full}})$. The quantity wanted is $\mathrm{Var}(Y-X) =
2\,\mathrm{Var}(e_{\text{full}})$ and the measurable one is $\mathrm{Var}(A-B) =
4\,\mathrm{Var}(e_{\text{full}})$, so the ratio of standard deviations is $\sqrt{2}$, and it
carries to $E|\cdot|$ under any common scale family. Those assumptions are approximate for
heteroskedastic or heavily zero-inflated genes — which is exactly why **both** values are emitted:
the correction stays inspectable rather than folded away.

**Precondition.** Every perturbation must have **at least 2 cells**; the reference raises
otherwise. `_disjoint_halves` drops a group at $n=1$, and such a target would sit in the full-real
gate while being absent from both halves — so the reference would omit it while the member had
already averaged it in, and the two aggregates would be means over different target sets. Null
`pert_col` labels are rejected for the same reason (the splitter cannot see them, but the DE
backends stringify them into a real target). A gated target missing from a half *after* DE also
raises: the v2 default `filter.filter_gene_min_cpm_cell = 5.0` can remove every gene for a target
whose cells survived the split.

Writes `lfc_nmae_ref.csv` (per-perturbation: `perturbation`, `nmae_ref_raw`, `nmae_ref_sqrt2`, `n_gate`)
and `lfc_nmae_ref_agg.csv` (one `mean` row plus `n_perturbations`).

**Cost:** three extra DE passes — two when `--de-real` is supplied, which the reference *does*
consume (`compute_metrics` owns the main run's real DE table internally and does not return it).

**The scaled score.** `cell-eval2 score --lfc-nmae-ref lfc_nmae_ref_agg.csv`, or
`score_metrics(..., lfc_nmae_ref=...)`, adds a `from_reference` column:

```math
\text{from\_reference} = \frac{1 - \overline{\mathrm{nmae}}}{1 - \overline{\mathrm{nmae\_ref\_raw}}}.
```

**The denominator is the RAW reference** (#276 part B). It used to be the
sqrt(2)-corrected column, which was renamed `nmae_ref_sqrt2` in the same change so no
reader gets new arithmetic under an unchanged name. Both columns are still emitted, but
with Spearman-Brown out of the scoring design there is no depth correction anywhere in
the scheme, and leaving one here would disagree with the replicate anchor by 17–23% at
the score level for the same metric on the same data.

**Aggregate first, divide once** — the division is over the two *means*, never per perturbation
(the per-perturbation form is dominated by near-zero denominators). $0$ is *attained by* an
all-zero predicted LFC table — not uniquely, since a uniform $c\times$ real prediction reads
$|c-1|$, so $c=2$ lands there too — and $1$ is as good as re-running the experiment; **values above 1 are attainable and are reported, not
clipped** — beating a noisy replicate is a result. The column is populated only on the
`de_*_lfc_nmae` rows and is **deliberately not enrolled in `avg_score`**, so supplying a reference
cannot change any existing score. Passing a reference requires `comparison_statistic="mean"` and
raises otherwise, since the reference frame carries only a mean.

**⚠️ The two normalizations have different zeros, and that is not a defect.** `from_baseline` is
$1 - u/b$ against whatever baseline was published; against a baseline whose `nmae` is $1.0$ exactly
that is exactly $1-\mathrm{nmae}$, but against the baselines actually deployed it is
not — a generic-response baseline (§6a) reads $\mathrm{nmae}\approx 0.96$, so an all-zero
predicted-LFC table scores about $-0.04$ on `from_baseline` while scoring exactly $0$ on
`from_reference`. ⚠️ Both statements are about the predicted **log₂FC table**, not about a
submission that emits the control: under `control_source="real"` those are different objects
(§4.3, #286).
Both are correct; they answer different questions — "how does this compare to the baseline we
published" versus "where does this sit between silence and a repeat of the experiment". Never read
them as the same quantity.

**Degenerate and empty references.** If $\overline{\mathrm{nmae\_ref\_raw}} \ge 1$ the denominator is
non-positive; `score` reports the **unrescaled** $1-\mathrm{nmae}$ with a warning naming the source
and the measured value, rather than dividing by it and silently inverting the ranking. If nothing
cleared the gate the aggregate carries a null `nmae_ref_raw` with `n_perturbations = 0`; `from_reference`
is left null, a warning is issued, and every other metric is scored normally. A *malformed*
reference (a missing column, no single `mean` row, a NaN/inf/negative value, or a null contradicted
by a non-zero count) is a caller error and raises.

**Why not the ceiling.** `ceiling.py` corrects half depth to full depth with Spearman-Brown
$2r/(1+r)$, a **reliability** correction for bounded correlation metrics, and `SB_METRICS`
deliberately excludes the error metrics — `de_*_lfc_nmae` among them. An error metric needs the
$\sqrt{2}$ above, not that.

---

## 6c-bis. The replicate anchor — the 1.0 end of the 0=baseline / 1=replicate scale

Issue #276 defines a scale where **0 is the baseline and 1 is a replicate**, both measured per
dataset. The 0 end already existed (`from_baseline`). The **replicate anchor** is the 1 end: a
per-dataset artifact covering *every* metric the run emits.

    cell-eval2 run -ar real.h5ad --anchor --anchor-splits 5 --cache-real CACHE -o out/

Writes `anchor_agg.parquet` (one row per metric: `replicate`, `replicate_sd` / `_min` / `_max`,
`n_perturbations_min` / `_max`, `estimator`), `anchor_splits.parquet` (one row per split × metric)
and `anchor_meta.json`.

**What it measures.** The real data is split into disjoint halves per perturbation, `half_b` is
scored against `half_a` through the ordinary `compute_metrics` path, and the per-split
*aggregates* are averaged over five seeded splits. `control_source` is **forced to `"pred"`**
inside the split core and is not optional: under `"real"` both halves' log2FCs would be computed
against one shared control, correlating the two quantities whose agreement is being measured and
biasing the anchor upward — which under this scheme biases every submission's score *downward*,
uniformly, with nothing in the output that looks wrong.

**It is RAW.** No Spearman-Brown, no $\sqrt{2}$, no depth correction of any kind — anywhere in
this scheme. `ceiling.py` is untouched and is legacy for this purpose.

**Seeds.** One `base_seed` is pinned; the five split seeds are derived by
`numpy.random.SeedSequence(base_seed).generate_state(n_splits)` and **stamped literally** in the
sidecar, so a later refactor of the rule cannot silently move a shipped anchor.

**`de_*_lfc_nmae` uses a different estimator, and that is a correctness requirement.** That
member omits a perturbation whose real-side significance gate holds fewer than `min_gate_size`
genes, and a half calls far fewer genes significant than the full data — so the uniform
split-half core would average over a **21–35 % smaller cohort** than the member it normalizes
(measured on six internal lines, `CCL_1`..`CCL_6`). `compute_lfc_nmae_reference` takes its gate
and denominator from the **full real** table and only the numerator's two vectors from the halves,
so its cohort equals the member's exactly. Its `estimator` column therefore reads `full_gate_raw`
where every other metric reads `split_half_raw`, and validation **enforces** that pairing: an
anchor claiming `split_half_raw` for an lfc_nmae member is rejected.

**Resolution order: supplied → cached → raise.** `score` never recomputes an anchor.

    cell-eval2 score --user-agg user/agg_results.csv --baseline-agg base/baseline_agg.csv \
        --anchor out/            # or: --anchor-cache CACHE

Both doors run the **same** validation: exact dtypes, exactly one row per expected metric (no
duplicates, no extras, no missing, never empty), a finite `replicate`, the estimator pairing
above, and agreement on `real_fingerprint` (the **strict content** hash — the metadata hash is
stamped as provenance and is never the gate), `semantic_identity` and `cell_eval2_version`. The
expectations come from the **user run's** `run_meta.json`, never from the anchor's own sidecar,
which would validate an artifact against itself. A user run without `--cache-strict` has only a
metadata fingerprint and is **refused**.

**The scored column.** `from_replicate` is policy-applied: it is
`score_one(u, b, policy)` with `policy.anchor` set to the metric's measured replicate. Thus

```math
\frac{u-b}{r-b}
```

is only the unclamped linear core. The column inherits the metric's own clamps, penalty shape
(the Box–Cox tail, or `de_*_lfc_nmae`'s linear ramp — §6.1) and `is_degenerate` rule from the
catalog. There is deliberately no second policy table: one
`Scoring` per metric serves both scales.

It is also **policy-frozen**. Score-time overrides (`--penalty-cap`, `clamp_*`, `penalty`, or
`overrides=`) apply to `from_baseline` only and never to `from_replicate`, exactly as they never
move a frozen scale. ("Apply to", not "move": a *standalone* `--penalty-cap` is inert under the
shipped `vcc2026` policies as of the `ERROR_LINEAR` change — `de_*_lfc_nmae` has no penalty and
`expr_mse_unbiased_capped_norm` clips its tail away — so on that profile it moves neither column.
It becomes live again only under a policy that actually carries a Box–Cox tail on that member, and
the route to one is `overrides={"de_wilcoxon_lfc_nmae": ERROR}`, not a global setting: there is no
`--penalty` CLI flag, and `score_metrics(..., penalty="boxcox")` over a whole `vcc2026` frame
raises on the first higher-is-better member. Under such a per-metric policy the declared `-6` floor
still catches a non-finite value the cap would otherwise have set.) A degenerate replicate scale — no headroom over the baseline, or a non-finite end
— raises for a **DECISIVE** metric, which includes every `vcc2026` member, and warns-and-omits
any other metric. It no longer silently nulls the row.

**Enrolment depends on the artifact.** With `score --real-bundle` on a **competition** bundle,
`from_replicate` at `avg_score` is the competition score and `from_baseline` at `avg_score` goes
null. A null is a visible break; a plausible no-longer-official number under the label every
consumer already reads is not. With a plain `--anchor`, or a **diagnostic** bundle,
`from_replicate` stays diagnostic and both averages stand.

Two measured facts matter when interpreting the column. First, a perfect submission scores
well above 1, not 1.0: the anchor is a half-depth split-half estimate, an easier bar than a
full-depth replicate. It measured approximately 1.4–1.6 on the internal 478–544-construct
parent panel and approximately 1.59 on the repository's synthetic fixture. Second, the members
are unequally weighted in practice: `de_wilcoxon_sig_jaccard` reaches approximately 3.0 where
`expr_mse_unbiased_capped_norm` is capped at 1.0.

The frame also carries `anchor_source` (`supplied`/`cached`/`real_bundle`) and `anchor_digest`.
On a bundle run it additionally carries `real_bundle_id` and `real_bundle_digest`, so the score
file identifies both the anchor and the bundle that produced it.

**Anchor stability is per-metric and per-dataset.** `replicate_sd` / `_min` / `_max` are stamped
rather than thresholded: on one internal line in six the significance-family anchors move by
+23 %, +12 %, +19 % and −36 % under a 1.41× depth step while its expression anchors stay healthy.
No threshold is pre-registered.

---

## 6c-ter. The real bundle — both scale ends, and the identity that binds them

`cell-eval2 prep-real-bundle` writes exactly seven files: `manifest.json`,
`baseline_agg.csv`, `baseline_meta.json`, `config.yaml`, `anchor_agg.parquet`,
`anchor_splits.parquet`, and `anchor_meta.json`. It writes aggregates and metadata only — no
matrices and no other file. Every file except the manifest and `config.yaml` is byte-format
identical to what `run`, `baseline`, and `run --anchor` already write, so `read_anchor` opens a
bundle unchanged.

Both scale legs are computed from one config. The supplied baseline arm is scored as the
prediction to make the 0 end; a five-split replicate anchor is computed from the real data to
make the 1 end. Three build-time gates must all pass: the legs agree on the real data's
**strict content fingerprint**, on the anchor's **semantic identity**, and on
`cache.config_hash`. The last comparison is hash against hash, never against
`baseline.config_digest`, which is host-dependent. A degenerate leg on either end aborts the
build.

⚠️ Because `score` runs the same `(base, replicate)` builder, a bundle built under a **wider
profile** aborts if any decisive metric in that profile has no usable replicate scale. The
build fails in seconds rather than the campaign failing later.

The manifest stamps one of three rule states, decided once at build time:

- `rule_digest` equal to this build's `competition_digest()` means a **competition** bundle;
  its `from_replicate` average is enrolled.
- `rule_digest: null` means a **diagnostic** bundle. `rule_mismatches` lists every reason, and
  the CLI prints those reasons on the first build.
- A present but different `rule_digest` means the bundle was built under a competition rule
  that has since moved. `score` refuses it.

`score --real-bundle` compares nine submission peers from the manifest with the submission's
own `run_meta.json`: `cell_eval2_version`, `config_digest`, `comparator`,
`source_fingerprint`, `source_fingerprint_strict`, `resolved_device`,
`resolved_de_backend`, `input_type_real_effective`, and
`de_real_fingerprint`. Any disagreement is fatal and names every mismatched field; there is no
waiver. `--allow-config-mismatch` alongside `--real-bundle` is a usage error, not an escape
hatch. `--real-bundle` is mutually exclusive with `--baseline-agg`, `--baseline-meta`,
`--anchor`, and `--anchor-cache` because the bundle supplies both scale ends.

Two **prediction-side** fields are treated differently from the nine peers, because the
prediction is the one side a baseline/submission pairing is *expected* to differ on. They are
not handled the same way as each other, and the difference is where each one lives:

- `input_type_pred_effective` is **in the manifest, and never compared** (#192). It is copied
  from the baseline leg exactly as a peer is, and `read_real_bundle` still requires it — only
  the comparison went. The baseline arm is a fractional mean of counts that the matrix-space
  lock pulls back to `counts`, while a submission is commonly log-normalized, and both sides
  are converted into the same metric space before any metric is computed. Under the frozen
  competition rule it cannot differ at all: `vcc2026` sets `version: v2` and
  `autodetect_input_type: false`, so the effective type is the declared one on both sides and
  is already covered by `config_digest`.
- `de_pred_fingerprint` is **NOT in the manifest**, and is checked **one-sidedly on the
  submission** (#291): `run_meta.json` must carry it, and it must be null. It records *that* a
  pred-side DE table was supplied with `--de-pred`, not what the prediction contains, so a
  null value is every ordinary submission and a non-null one is a DE table that was not
  derived from the submitted cells. It is deliberately not a peer: the bundle side is null by
  construction — `_baseline_leg` never passes a `de_pred` — so there is nothing to compare
  against, and requiring it in the manifest would make every bundle built before the check
  unreadable. (The bundle's `baseline_meta.json` carries it, as it carries the whole baseline
  `run_meta`; the *manifest* does not.)

`--de-pred` is also the isolator the metric campaigns are built on, and those arms want the
bundle's own scale. **`score --diagnostic-supplied-de-pred`** is the one opt-in on this gate,
and it buys the run rather than the label: the submission is scored, and it is **not
enrolled**. The result gets exactly the treatment a diagnostic *bundle*'s does —
`from_replicate` is reported, `from_baseline` keeps its `avg_score`, and a warning names the
waiver — so there is one meaning of "scored against a bundle but not enrolled" rather than
two. It waives nothing else: a peer mismatch is still fatal alongside it, and so is a
*missing* `de_pred_fingerprint`, since the opt-in is scoped to a table that was actually
supplied. It is a usage error without `--real-bundle`, which has no enrolment to downgrade.

⚠️ The downgrade lives in the scoring call, not in the artifact. The bundle is a competition
bundle and its manifest still carries the real `rule_digest`, so a reader of `scored.csv`
tells an enrolled run from a diagnostic one by `avg_score`'s `from_baseline` being **null**
(enrolled) or **populated** (not).

⚠️ The manifest is **provenance, not a checksum**. Nothing re-verifies the bundle's own files
at score time, so editing a cell moves every score with no trace. The threat model is accidental
mismatch, not tampering.

Use `--preset vcc2026`, backed by the packaged `configs/vcc2026.yaml`. It follows v2's
conventions with exactly two deltas: the `vcc2026` metric profile and `cache_strict: true`. The
latter is load-bearing because the anchor gate is the strict content hash. A complete build and
score pair, using dataset aliases only, is:

```bash
cell-eval2 prep-real-bundle --preset vcc2026 \
    --real CCL_2.real.csad --baseline CCL_2.context_mean.csad \
    --id vcc2026-CCL_2-r1 -o bundles/vcc2026-CCL_2-r1
cell-eval2 score --user-agg sub/agg_results.csv --real-bundle bundles/vcc2026-CCL_2-r1
```

---

## 6d. Scales — scoring against constant reference points

`from_baseline` (§6.1) and `from_reference` (§6c) both measure against something *measured*.
A **scale** measures against two *constants*, so a value reads without the comparator in hand.

    cell-eval2 score --user-agg user/agg_results.csv --scale low-random_high-1_v10

Each `--scale` adds one column named for the scale; `from_baseline` and its `avg_score` are
untouched. `--scale` is repeatable, and `--baseline-agg` becomes optional when one is given —
a scale carries its own reference points, so it needs no baseline artifact at all, nothing to
regenerate when `cell_eval2_version` changes, and no way to go degenerate.

**`low-random_high-1_v10`** — 0 at the random minimum, 1 at real input (the real count matrix
pasted as the prediction). Covers the six scored `vcc2026` metrics; the profile's four
unscored expression diagnostics are not scale entries:

| metric | base (0) | direction | anchor (1) | penalty | clamp_high | clamp_low |
|---|---:|---|---:|---|---:|---:|
| `expr_mse_unbiased_capped_norm` | 1.0 | lower | 0.0 | none | 1.0 | −6.0 |
| `de_wilcoxon_lfc_nmae` | 1.0 | lower | 0.0 | none | 1.0 | −1.0 |
| `pds_cosine` | 0.5 | higher | 1.0 | none | 1.0 | −1.0 |
| `de_wilcoxon_direction_fidelity_yield_raw` | 0.5 | higher | 1.0 | none | 1.0 | −1.0 |
| `de_wilcoxon_direction_reach_raw` | 0.0 | higher | 1.0 | none | 1.0 | 0.0 |
| `de_wilcoxon_sig_jaccard` | 0.0 | higher | 1.0 | none | 1.0 | 0.0 |

A pasted real matrix scores **1.0 on all six**; the random point scores **0.0 on all six**.
`avg_score` under this scale spans $[-1.5, 1]$. Real replicate arms land at **0.8028**
(`CCL_2`, a technical split-half) and **0.6539** (`H1_CGS`, two biological replicates) —
⚠️ both measured under `_v2`, i.e. the pre-#264 `lognorm` comparator. A replicate is exactly
the arm the comparator move affects most (#268 measured its `expr_mse_unbiased_capped_norm`
going *negative* at the retired `bulk_target_sum=1e6`, which `clamp_high=1.0` then read as a
full 1.0 on that metric; the shipped 5e4 restores a positive ceiling), so re-measure before
quoting either number for a v2 counts run.

For `expr_mse_unbiased_capped_norm`, 1.0 is the no-skill point whatever the **reference**
panel's depth because both panel sums are debiased. It is not invariant to the submission: in the
matched-i.i.d. no-skill experiment, thinning raised the expected value above 1, because #247's cap
refuses a correction the reference does not earn. Where the realized value does exceed 1 the scale
maps it below 0 before applying its −6.0 floor. (Only a tendency, and only for depth: the binding
condition is the cap inequality rather than the cell count, and the *dispersion* half has the
opposite sign under `bulk_lognorm` at fixed group totals — see §2.3 and #278.) ⚠️ The scale's base is a **policy constant** — it does not move with the panel —
while the metric's own no-skill point is exact only for an unbiased $C$, so a control-emitting
submission scores slightly below 0 on this scale rather than exactly 0. How far below tracks
the jackknife's bias. Measured on the panel that characterises it (#268): at the retired
$\mathrm{TS}=10^{6}$ the anchor read **1.0727**, i.e. the base shipped 7.3% wrong; at the
shipped $5\times10^{4}$ it reads **1.0249**, a 2.9× smaller drift.

⚠️ **`expr_mse_unbiased_capped_norm` can overshoot 0 on a paste.** A paste is the same sample,
not an independent draw, so the signed numerator is negative rather than exactly zero;
`clamp_high=1.0` absorbs that score overshoot. Historically, `expr_mse_unbiased_norm` had the
same signed-paste issue: its numerator collapsed to
$-2\,\mathrm{tr}\hat\Sigma_{\text{real}}/n_{\text{real}}$, and its aggregate was measured at
$-1.4532$ and $-1.3935$ at 500 cells per label. Those are historical
`expr_mse_unbiased_norm` values, not values of the new ratio-of-sums metric.

⚠️ **`sig_jaccard`'s 0 is chosen, not derived.** Its analytic chance level is
$E[J]\approx\frac{ab/G}{a+b-ab/G}$ — 0.0062 at replicate-sized predictions, 0.0121–0.0124 at
the generic baseline's set sizes — so the theoretical minimum credits ~0.006–0.012 of free
chance overlap in exchange for a stable anchor.

**A shipped scale is immutable.** Any change to any field mints a new `_v<n>`; a name is never
redefined, so an old column can never bind to a new definition. **A change to what a keyed
metric *means* mints one too, even when every field is identical** — otherwise the name would
silently rebind to a new definition, which is the exact failure the rule exists to prevent.
Immutability does not make an unevaluable scale permanent. The chain so far:

Each row names what **retired** that version — i.e. the change that minted its successor, not
the change that minted the row's own name. (`_v5` was minted by #268 and retired by #282, so it
appears on the `_v4` row as the retiring cause and on the `_v5` row as the thing retired.)

| scale | retired by |
|---|---|
| `low-random_high-1_v1` | #257 removed `expr_mse_unbiased_norm`, the metric it keyed — an unconstructible scale, checked at import |
| `low-random_high-1_v2` | #264 moved `pds_cosine` to the `bulk_lognorm` comparator |
| `low-random_high-1_v3` | #264 moved `expr_mse_unbiased_capped_norm` to it as well |
| `low-random_high-1_v4` | #268 moved `bulk_target_sum` 1e6 → 5e4, shifting every scored value |
| `low-random_high-1_v5` | #282 changed `pds_cosine`'s tie handling — same table again, new meaning |
| `low-random_high-1_v6` | #172 excluded each perturbation's own target gene from three of the six keyed members — same table again, new meaning |
| `low-random_high-1_v7` | `direction_reach_raw`'s purity floor moved `1 − α/2` → `REACH_PURITY_FLOOR = 0.9` — same table again, new meaning |
| `low-random_high-1_v8` | #271 made `prep._grouped_sums` reduce WIDE, moving the values keyed for `pds_cosine` and `expr_mse_unbiased_capped_norm` — same table again, new *pseudobulk* |
| `low-random_high-1_v9` | #343 removed EVERY panel target gene from `pds_cosine`'s feature space, and #348 bounded `expr_mse_unbiased_capped`'s prediction-side correction by the submission's own across-perturbation spread — two keyed members, both new meanings. ⚠️ #343 shipped without a mint, so `_v9` had already begun to span two definitions; `_v10` pays that debt |
| **`low-random_high-1_v10`** | — **current, not retired**; same table as `_v2`…`_v9` |

A digest test enforces the current registry.

## 7. Metric catalog summary

`dir`: ↓ = lower is better, ↑ = higher is better, · = no direction.
`anchor`: the value a perfect submission attains (blank = none defined).
`scored`: ✓ = enters `avg_score`.
`worst (v2)`: value a degenerate perturbation maps to in v2 (— = not applicable / cannot drop).
⚠️ This column is **not** simply `MetricSpec.worst_value`: the four §5 chance-corrected metrics
carry `worst_value=None` because the $-1$ is applied *inside the metric*, and
`de_wilcoxon_sig_jaccard` carries `worst_value=None` because it returns $1.0$ for an empty
union inside the metric rather than deferring to the v2 dispatch. Do not regenerate this
column from the catalog alone — see the note below the table.
Profiles: **v**=vcc, **2026**=vcc2026, **m**=minimal, **f**=full, **a**=anndata, **d**=de, **p**=pds.

| metric (v2) | v1 alias | range | dir | anchor | scored | worst (v2) | profiles |
|---|---|---|---|---|---|---|---|
| `expr_mae` | `mae` | $[0,\infty)$ | ↓ | $0$ | ✓ | — | v m f a |
| `expr_mse` | `mse` | $[0,\infty)$ | ↓ | $0$ | ✓ | — | m f a |
| `expr_mse_unbiased` | — | $(-\infty,\infty)$ | · | | | — | f a 2026 |
| `expr_mse_unbiased_capped` | — | $(-\infty,\infty)$ | · | | | — | f a 2026 |
| `expr_distance_unbiased` | — | $(-\infty,\infty)$ | · | | | — | f a 2026 |
| `expr_real_mass_ratio` | — | $[0,\infty)$ | · | | | — | f a 2026 |
| `expr_mse_unbiased_capped_norm` | — | $(-\infty,\infty)$ | ↓ | $0$ | ✓ | — | f a 2026 |
| `delta_mae` | `mae_delta` | $[0,\infty)$ | ↓ | $0$ | ✓ | — | f a |
| `delta_mse` | `mse_delta` | $[0,\infty)$ | ↓ | $0$ | ✓ | — | f a |
| `delta_pearson` | `pearson_delta` | $[-1,1]$ | ↑ | $1$ | ✓ | $-1$ | m f a |
| `pds_l1` | `discrimination_score_l1` | $[0,1]$ | ↑ | $1$ | ✓ | — | v m f a p |
| `pds_l2` | `discrimination_score_l2` | $[0,1]$ | ↑ | $1$ | ✓ | — | f a |
| `pds_cosine` | `discrimination_score_cosine` | $[0,1]$ | ↑ | $1$ | ✓ | — | f a 2026 |
| `de_wilcoxon_overlap` | `overlap_at_N` | $[0,1]$ | ↑ | $1$ | ✓ | — | v m f d |
| `de_wilcoxon_overlap_top{50,100,200,500}` | `overlap_at_{k}` | $[0,1]$ | ↑ | $1$ | ✓ | — | f d |
| `de_wilcoxon_precision` | `precision_at_N` | $[0,1]$ | ↑ | $1$ | ✓ | — | m f d |
| `de_wilcoxon_precision_top{50,100,200,500}` | `precision_at_{k}` | $[0,1]$ | ↑ | $1$ | ✓ | — | f d |
| `de_wilcoxon_nsig_counts_real` | `de_nsig_counts_real` | $[0,G]$ | · | | | — | m f d |
| `de_wilcoxon_nsig_counts_pred` | `de_nsig_counts_pred` | $[0,G]$ | · | | | — | m f d |
| `de_wilcoxon_nsig_spearman` | `de_spearman_sig` | $[-1,1]$ | ↑ | $1$ | ✓ | $-1$ | f d |
| `de_wilcoxon_sig_recall` | `de_sig_genes_recall` | $[0,1]$ | ↑ | $1$ | ✓ | $0$ | f d |
| `de_wilcoxon_direction_match` | `de_direction_match` | $[0,1]$ | ↑ | $1$ | ✓ | $0$ | f d |
| `de_wilcoxon_model_direction_match` | `de_model_direction_match` | $[0,1]$ | ↑ | $1$ | ✓ | $0$ | f d |
| `de_wilcoxon_direction_precision` | — | $[0,1]$ | ↑ | $1$ | ✓ | $0$ | f d |
| `de_wilcoxon_direction_sensitivity` | — | $[0,1]$ | ↑ | $1$ | ✓ | $0$ | f d |
| `de_wilcoxon_direction_sensitivity_universe` | — | $[0,\infty)$ | ↑ | | ✓ | $0$ | f d |
| `de_wilcoxon_lfc_spearman` | `de_spearman_lfc_sig` | $[-1,1]$ | ↑ | $1$ | ✓ | $-1$ | f d |
| `de_wilcoxon_lfc_spearman_pos` | `de_spearman_pos_lfc_sig` | $[-1,1]$ | ↑ | $1$ | ✓ | $-1$ | f d |
| `de_wilcoxon_lfc_spearman_neg` | `de_spearman_neg_lfc_sig` | $[-1,1]$ | ↑ | $1$ | ✓ | $-1$ | f d |
| `de_wilcoxon_lfc_nmae` | — | $[0,\infty)$ | ↓ | $0$ | ✓ | — | f d 2026 |
| `de_wilcoxon_pr_auc` | `pr_auc` | $[0,1]$ | ↑ | $1$ | ✓ | $0$ | f d |
| `de_wilcoxon_roc_auc` | `roc_auc` | $[0,1]$ | ↑ | $1$ | ✓ | $0$ | f d |
| `de_wilcoxon_sig_jaccard` | — | $[0,1]$ | ↑ | $1$ | ✓ | — | f d 2026 |
| `de_wilcoxon_overlap_adjusted` | — | $[-1,1]$ | ↑ | $1$ | ✓ | $-1$ | f d |
| `de_wilcoxon_precision_adjusted` | — | $[-1,1]$ | ↑ | $1$ | ✓ | $-1$ | f d |
| `de_wilcoxon_sig_recall_adjusted` | — | $[-1,1]$ | ↑ | $1$ | ✓ | $-1$ | f d |
| `de_wilcoxon_sig_mcc` | — | $[-1,1]$ | ↑ | $1$ | ✓ | $-1$ | f d |
| `de_wilcoxon_direction_fidelity` | — | $[-q/d,\ (1-q)/d]$ | ↑ | $1$ | ✓ | — | f d |
| `de_wilcoxon_direction_fidelity_raw` | — | $[0,1]$ | ↑ | $1$ | ✓ | — | f d |
| `de_wilcoxon_direction_coverage` | — | $[0,\infty)$ | ↑ | | ✓ | — | f d |
| `de_wilcoxon_direction_yield` | — | unbounded both ways | ↑ | | ✓ | — | f d |
| `de_wilcoxon_direction_yield_raw` | — | $[0,\infty)$ | ↑ | | ✓ | — | f d |
| `de_wilcoxon_direction_fidelity_yield` | — | $[0,1]$ | ↑ | $1$ | ✓ | — | f d |
| `de_wilcoxon_direction_fidelity_yield_raw` | — | $[0,1]$ | ↑ | $1$ | ✓ | — | f d 2026 |
| `de_wilcoxon_direction_reach` | — | $[0,1]$ | ↑ | $1$ | ✓ | — | f d |
| `de_wilcoxon_direction_reach_raw` | — | $[0,1]$ | ↑ | $1$ | ✓ | — | f d 2026 |
| `de_wilcoxon_direction_reach_unbounded` | — | $[0,\infty)$ | ↑ | | ✓ | — | f d |
| `de_wilcoxon_direction_reach_unbounded_raw` | — | $[0,\infty)$ | ↑ | | ✓ | — | f d |

> The eleven `direction_*` chance-corrected rows (§4.3) were added by issue #195 without a
> table entry. `de_wilcoxon_direction_yield` is the
> only catalog metric whose baseline may legitimately be negative, so its denominator is
> $|b|$ rather than $b$.

> The chance-corrected metrics (§5) map degenerate tables to $-1$ inside the metric itself
> (always, both versions), while `sig_jaccard` resolves its own only-undefined case (an empty
> union) to $1.0$ inside the metric. The other v2 worst values are applied by the v2 dispatch
> (issue \#89 / \#92) while v1 keeps the upstream omit/NaN behavior.

---

*Sources: `src/cell_eval2/metrics/{delta,de,discrimination}.py`, `distances.py`, `prep.py`,
`norm.py`, `catalog.py`, `compat/`.*
