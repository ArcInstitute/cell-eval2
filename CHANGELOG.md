# Changelog

All notable changes to cell_eval2. Releases before 0.0.4 (`v0.0.2`, `v0.0.3`) predate this file.

## [0.16.0] — 2026-08-19

**The release that can be published.** Nothing here changes a scored number: `rule_version`
stays 3 and `competition.competition_digest()` is bit-identical to what the three official `-r3`
bundles stamp (verified, not assumed). What changes is that the distribution is installable by a
public reader at all — #358 replaced `[scale]`'s private `git+ssh` reference with
`cellstream>=0.9.1` and removed `allow-direct-references`, and PyPI rejects any distribution whose
`requires_dist` carries a direct reference, so every prior version was unpublishable regardless of
repository visibility.

⚠️ **This bump invalidates the `-r3` bundles, and only the bump does.** `cell_eval2_version` is
the first entry in `real_bundle.SUBMISSION_PEERS`, compared exactly and fail-closed, with no
waiver on the bundle path — so a run at 0.16.0 cannot be scored against a bundle stamping
`0.15.0`. The three official val bundles are rebuilt at 0.16.0 after this tag; the rebuild is a
re-stamp rather than a re-measurement, which is what makes it verifiable.

⚠️ **Reinstall before building anything whose provenance matters.** An editable install freezes
the version `importlib.metadata` reports at install time, so a build run from a bumped checkout
without reinstalling stamps artifacts `0.15.0`. See the note in `__init__.py`.


### Added

- **`docs/vcc2026_metrics/vcc2026-metrics-brief.{md,pdf}` — an abridged reference edition of the
  metric specification.** Definitions, parameters and measured values only, with the motivation,
  derivations and worked reasoning removed: 12 pages against the full document's 19. It is a
  companion, not a replacement, and says so in its own header — where the two disagree the full
  document carries the argument and the brief carries the number. `build_brief.sh` and
  `main_brief.tex` build it through the same pandoc → pdflatex pipeline and the same
  `preamble.tex` as the full document, so the two typeset alike.

- **Both documents are re-measured on the three official `-r3` bundles** (`0.15.0`,
  `rule_version` 3), replacing the estimate block that section 0 carried while the rebuild was
  pending. The values are read from the panel arms scored against those bundles rather than
  derived from `b` and `r` by hand, and both controls hold: the baseline arm reproduces 0 on every
  member and the reference scored against itself reproduces each bundle's stamped perfect score.

  The headline change is `pds_cosine`: its baseline is now **exactly 0.500** on all three
  contexts, not 0.510–0.530. With #343 removing all 300 panel target genes the mean-response arm
  carries nothing that distinguishes one perturbation from another, so every distance row ties and
  every perturbation takes the mid-rank — the baseline IS the no-information point rather than
  sitting above it, the control-pasting arm scores 0 rather than −0.07 to −0.02, and the scale is
  symmetric. `expr_mse_unbiased_capped_norm` did not move at all, baseline or anchor, which is
  what #348 predicted. All four DE members' spans widened.

- **Notation fix in both metric documents: the expression error's correction cap is `\rho`, not
  `r`.** `r` was doing double duty — the replicate anchor in `s = (u-b)/(r-b)`, which every member
  uses, and section 2's cap `r = min(1, B / \sum_q ...)`, which bounds the sampling correction a
  submission may claim. The formulas were right and no number moves, but a reader implementing
  section 2 against the specification had to infer from context which `r` a given line meant. The
  cap is now `\rho` throughout; the replicate keeps `r`. The collision came from the source, where
  `metrics/delta.py` also writes `r = min(1, ...)` — but the code has no competing `r`, so only the
  documents needed the distinct symbol (Copilot, #368).

### Changed

- **BREAKING (packaging): `[scale]` installs the public `cellstream>=0.9.1`, not the private
  `shardad` (#358).** `scale` was a PEP 440 direct reference —
  `shardad @ git+ssh://git@github.com/ArcInstitute/shardad` — which is why `[tool.hatch.metadata]
  allow-direct-references` existed, and it is now gone with it. That removal is what makes
  cell_eval2 **publishable at all**: PyPI rejects a distribution whose `requires_dist` carries a
  direct reference, extras included, so every version of this package was unpublishable while
  `scale` named a git URL, independent of repository visibility. `release.yml` guards the class.

  The migration itself is a pure namespace swap — every module path and symbol maps 1:1 onto
  cellstream, verified against the installed wheel — so no scored value, cache key, digest,
  anchor or bundle changes. `rule_version` stays 3 and the `-r3` bundles are unaffected.

  ⚠️ **The floor is `>=0.9.1`, and the `.1` is load-bearing on Python 3.13.** cellstream 0.9.1
  declares `Requires-Python: >=3.11,<3.13`; 0.9.0 does not, is not yanked, and on 3.13 installs,
  imports and reports its Rust extension present before failing non-deterministically. Under
  `>=0.9.0` a 3.13 resolver excludes 0.9.1 by its own cap and silently takes 0.9.0, which
  *satisfies* the spec — so the guardrail has to be the floor. Under `>=0.9.1` there is no
  candidate and the install fails loudly, which is the wanted outcome. 54 of the 55 payload files
  in the two cp312 wheels hash identically (only `_version.py` differs), so every measurement
  taken against 0.9.0 carries over. cell_eval2's own `requires-python` stays `>=3.11`: the cap
  belongs to the dependency that has it, not to every consumer of this package.

  ⚠️ `scale` takes **bare** `cellstream`. The cell-layout reader's default `pfordelta` codec needs
  cellstream's optional `[cell]` extra (`pyfastpfor`), and its GPU `.shad` decode needs `[gpu]`;
  neither is pulled in. `.[scale,gpu]` supplies cupy and nvCOMP from cell_eval2's own `gpu` extra
  instead, deliberately unpinned against cellstream's conservative `nvidia-nvcomp-cu12>=4,<5`.

  ⚠️ Residual, upstream: `gpudge[streaming]` pins `cellstream>=0.9.0`, one minor below this floor,
  so it remains a second door to the 3.13 hazard for anyone who installs it directly. That belongs
  in gpudge, not here.

### Added

- **`[gpudge]` optional dependency — `gpudge>=0.9.0` (#361).** gpudge is the GPU DE engine
  `de.backend="auto"` selects whenever a CUDA device is visible, and it computes four of the six
  scored `vcc2026` members; until now it was reachable only from a private checkout, so a public
  reader on a GPU host hit `auto`'s deliberate hard raise with no supported way to satisfy it.
  It is its **own** extra rather than part of `gpu`: `gpu` is cupy + nvCOMP for the pseudobulk and
  rank kernels, while gpudge pulls `torch>=2.5`, and folding it in would put torch in front of
  everyone who wants only the kernels.

  Both halves of the blocker recorded in the 0.3.0 notes are gone — gpudge 0.9.0 published to PyPI
  2026-08-19 as a pure-Python `py3-none-any` wheel declaring `Requires-Python: >=3.11`, so the
  "gpudge needs >=3.12 while cell_eval2 supports >=3.11" mismatch no longer exists.

  ⚠️ Installing the extra **raises shared dependency floors above cell_eval2's own** —
  `anndata>=0.12`, `numpy>=2`, `polars>=1.38`, `pyarrow>=20`, `scipy>=1.17` — and adds
  `hdf5plugin>=6` and `torch>=2.5`, which cell_eval2 does not otherwise require. That cost is why
  it is opt-in rather than a base dependency.

  ⚠️ **Scope note, resolved by #358 in this same `[Unreleased]` window.** As written, this entry
  said the shard-streaming path — `score_streaming` → `gpudge.de(shard_archive=...)` — needed
  gpudge's own `streaming` extra, which pins `cellstream>=0.9.0`, while `scale` installed
  `shardad`, so `.[scale,gpu,gpudge]` resolved and then failed at import on that one path. #358
  makes `scale` install `cellstream>=0.9.1`, and `gpudge[streaming]`'s SOLE dependency at 0.9.0 is
  `cellstream>=0.9.0` — which `scale` now satisfies transitively and strictly. So
  `.[scale,gpu,gpudge]` covers that path too, with no `gpudge[streaming]` needed. `streaming-gpu`
  is a separate matter: it names `cellstream[gpu]`, and `scale` takes bare `cellstream`.

## [0.15.0] — 2026-08-19

The version was parked at `0.15.0.dev0` until the release commit — an editable install freezes the
distribution version `importlib.metadata` reports at INSTALL time and every later run, baseline,
anchor and real-bundle build copies that value into its metadata's `cell_eval2_version`, so bumping
before the freeze would have labelled artifacts
`0.15.0` while the tree was still moving; and reverting to the tagged `0.14.0` instead would have
let a submission built from this tree clear `check_submission`'s version peer against the three
official `-r2` bundles that stamp it.

**BREAKING: `pds_cosine` moves, the competition rule moves with it, and every real bundle,
baseline and anchor built before this release is invalidated.** That was one change with one
reason when it was written; #348 and #351 joined the same `rule_version` 3 wave afterwards, so the
release carries THREE semantics changes under ONE bump and ONE rebuild.

`discrimination.exclusion_scope` is new, and defaults to `"panel"`: under
`exclude_target_gene=True` the ranked feature space now has EVERY panel target gene removed,
once, before any distance is computed. The previous rule — now `exclusion_scope="row"`, kept for
upstream cell-eval parity and pinned by the `v1` and `cell-eval-0.7.6` presets — removed only the
prediction row's own target gene, and removed it from that row's comparison against *every*
reference perturbation. So in cell `(i, j)` of the distance matrix, reference perturbation `j`'s
own knockdown stayed fully visible.

That asymmetry was scoreable. A submission that spikes the panel's OTHER target genes is
anti-correlated with every off-diagonal competitor, while its own diagonal — where its gene *is*
dropped — sits at cosine 0, so the self-match wins by construction on no information beyond the
target list every participant is given. Measured on the three official val contexts:
`pds_cosine` **0.7982 / 0.7570 / 0.7614** against baselines 0.5304 / 0.5284 / 0.5102, i.e.
**+0.57 / +0.49 / +0.51 of member score for a submission carrying no biology at all**. Under
`"panel"` the same arms measure **0.5000** on all three — exactly the control-paste floor.

The fix costs the legitimate end almost nothing: a perfect submission still scores `pds_cosine`
1.0000, and a partial one (a generic response mixed with a fraction of the true target-specific
direction) moves by **at most 0.01 of member score**. It also retires a smaller standing gift —
under `"row"` only the DIAGONAL's reference-side norm lost its knockdown, so every submission with
a non-zero delta was ranked against a systematically deflated self-distance.

### Added
- **`run_meta.json` and every bundle's `baseline_meta.json` now record WHAT COMPUTED THE
  NUMBERS.** One new nested `environment` key beside the `resolved_device` /
  `resolved_de_backend` already stamped, holding `{"version", "provenance"}` for the RESOLVED DE
  backend plus `polars`, `numpy`, `scipy`, `anndata` and `scanpy` (`cell_eval2` itself is already
  covered by `cell_eval2_version`).

  The artifacts named the engine and nothing named its VERSION, while `gpudge` drives four of the
  six scored `vcc2026` members. MEASURED 2026-08-18: the `-r2` official bundles were built against
  gpudge `3a71cc5` — `v0.7.0-4`, an untagged dev commit four past the tag — recorded nowhere at
  all. So "reproduce the official bundles" was unanswerable from the artifacts, and "install gpudge
  v0.7.0" was the *wrong* instruction, that tag predating the Mann-Whitney tie-accumulator change.
  The numeric stack is the other half: `pyproject.toml` declares lower bounds only
  (`polars>=1.0`, `numpy>=1.26`, `scipy>=1.11`, `anndata>=0.10`, `scanpy>=1.10`), releases are git
  tags, and nothing enforces `uv.lock` at install time.

  **`provenance` is the point, not decoration.** `importlib.metadata.version` reads INSTALL-TIME
  metadata, so an editable install reports whatever its tree declared when it was linked and goes
  on reporting it as that tree moves — which is exactly how `0.7.0` came to stand for `v0.7.0-4`. A
  bare version string would therefore have recorded a plausible falsehood. It is classified from
  PEP 610's `direct_url.json` into `release` (from an index) / `git`·`hg`·`svn`·`bzr`·`vcs` (a
  checkout, the VCS read rather than assumed) / `local-editable` / `local` / `archive`.
  ⚠️ **The classification only — never the URL or the path.** The editable case holds an absolute
  path into the build host's home directory and the VCS case a private repository URL, and this
  block ships inside every bundle; a test asserts the absence over the whole record, because a
  value that exists only at runtime is invisible to the publish sweeps.

  ⚠️ **It stops at `baseline_meta.json` and does NOT reach `manifest.json`** — the honest
  limitation. The manifest is built from `SUBMISSION_PEERS + MANIFEST_RECORDED_ONLY`, and
  `read_real_bundle` requires every name in both tuples to be PRESENT in a manifest it reads, so
  adding it to either would make all three frozen official val bundles unreadable outright (#291,
  measured). The bundle DIRECTORY records the environment; the manifest does not.

  Nothing compares the field, so it can never refuse a legitimate pairing: `check_submission`
  iterates `SUBMISSION_PEERS` and `cli._check_baseline_config` a fixed tuple, and extra keys are
  inert on both. It cannot move a digest — `config_digest` takes the CONFIG, not the meta — and
  `competition_digest()` / `scales_digest()` are unchanged by construction, verified against
  whatever this cycle's values are (they are `fb5aa56b…` and `c49271e6…` as this lands — the first
  moved by the `rule_version` 2 → 3 entry below, the second by #348's `_v10` mint; **this** change
  moves neither, and touches
  `rule_version` not at all: it records provenance, it changes no metric semantics). Best-effort throughout,
  mirroring `anchor._version()`'s "never lose an anchor to provenance": a missing package is
  recorded as `null`, an unreadable `direct_url.json` keeps the version and nulls the class, one
  failing distribution nulls only its own entry, and nothing in the block can raise. CI installs no
  gpudge, so its entry is legitimately `null` there. It adds MEASURED 7.2 ms per call (six
  `importlib.metadata` lookups), once per run and once per bundle leg, never in a per-perturbation
  loop — the "must not change what `run` costs" contract is about the matrices, and this reads no
  `X`.

### Changed
- **BREAKING (competition rule): `rule_version` 2 → 3, and the debt outstanding against it is
  empty** (#343). `competition_digest()` moves `80558072…` → `fb5aa56b…`. The three official
  `-r2` val bundles, and any test bundle, must be rebuilt against this version; a bundle stamped
  `rule_version = 2` enrols with neither it nor them.
- **BREAKING (competition rule, same `rule_version` 3 wave): the DE gene gate keeps a gene on the
  REFERENCE group's mean CPM alone** (#351). `filter_gene_min_cpm_cell` (5.0 under the competition
  preset) used to keep a `(target, gene)` row when the TARGET group's mean CPM cleared the threshold
  **or** the control's did. For a gene at or below the threshold in the control, the row `(t, g)`
  then existed **only** when the gene had risen above the threshold in `t`, so
  `tmean > threshold ≥ ref_mean` and therefore `log2FC = log2((tmean + ε)/(ref_mean + ε)) > 0` —
  a row's mere PRESENCE in the reference disclosed the sign the submission was being asked to
  predict. (#351 writes the stronger bound `log2FC > log2(threshold/ref_mean)`; the pseudocount
  breaks it — `(threshold + ε)/(ref_mean + ε) ≤ threshold/ref_mean` whenever `ref_mean ≤ threshold`,
  and the right-hand side is undefined at `ref_mean = 0`. Strict positivity is the true statement
  and the measured one.) Measured
  on the official `-r2` val panels, on-target rows excluded exactly as the members exclude them:
  **`P(real log2FC > 0) = 1.000000`** over **26,373 / 33,969 / 26,839** such rows (0.88%–1.16% of
  the reference table, 88–113 per perturbation), the bound holding row by row and the minimum
  `target_mean` sitting exactly on the gate's 5.0 boundary.
  That was scoreable with **no perturbation-specific information at all**: take 400 real control
  cells, add counts to every gene whose CONTROL mean CPM lies in [4, 5] until its group mean is 4×
  the control's, and submit that one block unchanged as the prediction for all 300 perturbations.
  `de_wilcoxon_direction_fidelity_yield_raw` measured **0.689661 / 0.764160 / 0.688485** against
  baselines 0.505647 / 0.522736 / 0.509365 — **+0.3722 / +0.5059 / +0.3651 `from_baseline`**,
  `from_replicate` +0.576 / +1.237 / +0.652, and **+0.1057 of OVERALL `avg_score`** averaged over
  the three contexts, i.e. 76% of the leading dev-leaderboard submission's whole 0.2295 on val B
  from an arm containing no biology. Amplitude did not matter (4× and 8× read +0.3722 and +0.3665 —
  only the SIGN is scored, and the gate had already fixed it), and unlike #343 it **stacked** on a
  submission carrying real signal rather than competing with it: a 25%-real-cells arm went
  0.046054 → **0.757105** raw with the same transform bolted on.
  Under version 3 those arms return to the honest control-paste floor: raw **0.001249 / 0.001124 /
  0.000272** with **`n_pred` 76.4 → 0.0**, val A's OVERALL `avg_score` +0.0685 → −0.2852 (the
  honest control-paste's own value to four decimals), and the boost's contribution to the
  signal-carrying arm −0.00006 of raw value. The smaller channels from the same leak shrink with it
  **on this arm**: `de_wilcoxon_direction_reach_raw` +0.0236 → +0.0047 (val A), +0.0060 → −0.0115
  (val B), +0.0070 → −0.0037 (val C) (#352).
  ⚠️ **#352 is CLOSED BY RULING, not because its residuals closed** (Alex, 2026-08-19) — the
  distinction is the difference between "fixed" and "measured, accepted, written down". A
  DIFFERENT arm, one that injects library mass rather than boosting genes, keeps most of
  `reach_raw` after this change: `from_baseline` **+0.130 / +0.169 / +0.144**, i.e. 85–89 % of the
  pre-fix channel surviving. The bridge to the overall score runs through `from_replicate`
  **+0.133 / +0.178 / ~+0.155**, which over six members is this member's contribution to
  `avg_score`: **+0.022 / +0.030 / +0.026**, from an arm containing no biology. ⚠️ val C's figure
  is an OVERestimate — its `reach_raw` replicate anchor was not remeasured after the fix, and
  anchors move upward. CPM normalization predicts a small DOWN shift on every gene that arm never touched, and
  real effects are down-skewed, so the head of that ranking clears the 0.9 purity floor for free —
  97–98 % of the surviving prefix is a predicted-down call on an UNBOOSTED gene and 0.0 % of it is
  a boosted one. No filter closes that: it is `reach_raw`'s no-skill point (#279), and closing it
  needs the direction family's per-target chance correction (#229/#259), **which is not happening
  for this competition**. `de_wilcoxon_lfc_nmae`'s filter route is likewise only partly closed
  (62 % / 35 % / 1 % of the channel survives on val A / B / C; only C is genuinely closed). This
  change IS a full fix for `de_wilcoxon_direction_fidelity_yield_raw`, the member that lands.
  ⚠️ **`competition_digest()` does NOT move**, and that is deliberate rather than an oversight: the
  payload freezes each member's scoring POLICY and this is DE semantics, so `rule_version` is the
  only lever — and version 3 had no bundle built against it yet, so #351 joins it for ONE bump and
  ONE rebuild wave by the same argument #348 used. Land it after a v3 bundle exists and two bundles
  would carry the same `rule_digest` over different DE semantics.
  ⚠️ **The gate that leaked runs inside gpudge, so this is not the one-clause change it looks
  like.** `gpudge._filter.combined_keep_mask` ANDs each active filter's "(target OR ref)" mask, and
  `compute_de` returns from its gpudge branch BEFORE `_apply_cpm_filter` — the official bundles are
  gpudge/cuda runs, so changing the CPU clause alone would have moved no official number.
  cell_eval2 now takes the reference-only decision itself, in `_finalize_gpudge_de`, by one of the
  two routes `_gpudge_gate_plan` picks between (described below). **gpudge is unchanged and
  unpinned**; its own gate is forwarded only where it nests exactly and muted otherwise. The CPU
  backends keep their own clause in `_apply_cpm_filter`, and both paths now share one
  `_bh_per_target`.
  ⚠️ **The per-side BH recomputation is the mechanism, not bookkeeping.** The gate is applied to
  each side's own table, so the PREDICTION's `p_adj` is re-adjusted too: removing the attack's
  boosted block (p ≈ 1e-20) collapses BH's step-up and its remaining marginal drift calls stop
  clearing α as well. That is why `n_pred` goes to 0 rather than to ~4.
  **The scale ends move, in the direction that helps.** Baselines move by at most **0.0039** on any
  of the four `de_wilcoxon_*` members of any context, while **every replicate anchor moves UP and
  every span widens** — 2–5% on val A, **22–45% on val B**, 3–6% on val C — so the gate had been
  spending part of the members' resolution on direction-selected rows. **The cost is real and is
  not hidden**: the 0.88%–1.16% of reference rows only a perturbed group detected are genuine
  up-regulations and they leave the scoreable set, and `de_wilcoxon_lfc_nmae`'s gate cohort drops
  263 → 261 (A), 214 → 211 (B), 209 → 209 (C) — a real-side decision, so the omitted set stays
  identical for every submission.
  **The frozen 2025 `vcc` profile cannot move**: v1 sets `filter_gene_min_cpm_cell = None`, so its
  gate is inert. The gpudge gate has **two routes**, and which one is taken decides whether gpudge's
  own gate may stay on. The FRAME route — `ref_mean × 1e6/target_sum`, bit-exact, no matrix pass —
  is valid when `mean_calc='arithmetic'` **and gpudge itself normalized the cells** to a known
  finite `target_sum` — only then does its gate compute `arith_ref × 1e6/T` from the very array it
  returns, which is what makes the compare bit-exact AND the two sets nest exactly, so gpudge's own
  gate is forwarded and keeps its gene-chunk pruning. That is the competition, at `cpm_factor`
  exactly 1.0. A merely *nominally* uniform library is not enough: when `_to_linear` pre-normalized
  counts on the CPU, `ref_mean × 1e6/target_sum` is only algebraically the per-cell CPM (gpudge
  downcasts the staged values to float32, so the staged row totals need not be `target_sum`
  bit-for-bit, and its gate reads a separate accumulator regardless). So CPU-pre-normalized counts,
  lognorm and a geometric `mean_calc` all take the MATRIX route: the reference group's arithmetic mean of the true per-cell CPM, computed by
  cell_eval2 from the reference cells. That quantity is **invariant under any per-cell rescaling**,
  so it is right on raw counts, CPM, a non-1e6 `target_sum`, median normalization and expm1'd
  lognorm alike — which matters because `_to_linear` does **not** apply `target_sum` to lognorm
  input (it only applies `expm1`), so `1e6/target_sum` would be the wrong factor there. (Invariance
  is exact for the arithmetic — `x_ig/L_i` does not move under any *positive* per-cell rescaling —
  while two such matrices can still differ in the last floating bits, which is why the frame route
  is kept for the competition rather than replaced by this one.) The shipped `cell-eval-0.7.6`
  preset (lognorm, geometric, `target_sum=1e4`, filter 5.0) therefore **keeps its gate**, on the
  matrix route. A **negative** threshold stays the documented explicit keep-all and picks no route
  at all. ⚠️ One combination RAISES: an active gate on `compute_de_streaming` in Mode 1, where the
  reference shard lives inside the archive and there are no reference cells to derive the vector
  from. Mode 2 — an AnnData external control pool — *is* served, by the matrix route off those very
  cells. **No shipped preset reaches the refusal** — a parametrized test asserts that over all four —
  but a custom configuration can: there is no dedicated CLI flag for the gate, yet a generic
  `--set`/`--config` that pairs an active threshold with a geometric `mean_calc` or
  `target_sum=null` on the shard-streaming path lands on it, and gets this error rather than a wrong
  number. ⚠️ `reference` may no longer be spelled `"__all_others__"` or `"all_others"`: gpudge
  reads either as its 1-vs-rest sentinel and then emits a per-target `ref_mean`, which no
  reference-only gate can be decided from.

  ⚠️ **STALE CACHES ARE NOT INVALIDATED BY THIS CHANGE, AND THAT IS A HAZARD FOR THE REBUILD WAVE.**
  **Both** DE-table caches, the result fingerprint, the anchor's semantic identity and the
  reference-bundle semantics all key on the configured `filter_gene_min_cpm_cell` **value** and on
  nothing that says what the gate MEANS — `run.py`'s in-memory `params` and `_DE_RESULT_SEMANTICS`,
  `scale.py`'s separate shard-streaming DE key, `anchor.py`'s `_SEMANTIC_FILTER_FIELDS`, and
  `partition_inmem.py`'s semantics list. A warm cache written before this change therefore still
  hits, and serves OR-gated rows into a run that believes it is gated on the control alone —
  whatever version that cache was labelled. A cache already stamped `rule_version` 3, built for #343
  or #348 before #351 joined the version, is exactly as suspect as a version-2 one; the three
  official `-r2` frozen real caches are included. Closing it needs a **new** gate-semantics
  term modelled on `run._GROUPED_SUM_REDUCTION_SEMANTICS` and threaded through all five surfaces
  (bumping that existing counter would not do it — it does not reach the gpudge DE-table keys). It is
  deliberately **not** added here. **Until it is: build every bundle and score every submission
  against a COLD cache** — a fresh `--cache-real` / `--cache-pred` directory, or the pre-existing
  ones deleted. Any run that may have read a pre-fix cache entry is untrustworthy as rule_version 3
  output, whatever it stamps.
  **Free consequence:** under `control_source="real"` both sides gate on the same `ref_mean`, so
  the kept sets are identical and #291's pred-coverage omission goes to **0** — 1,754 of 102,786 →
  **0 of 100,771** on the val A baseline arm (same post-exclusion population; the gate takes 2,015
  pairs out of the reference's confident budget), 1,097 → 0 on the attack arm, 240 → 0 on an honest
  one.
  `de._warn_pred_gene_coverage` stops firing on ordinary h5ad runs and #291's "a model cannot raise
  its score by omitting genes" stops being denominator-only on that path; it stays a live
  diagnostic for supplied `--de-pred` tables and for `control_source="pred"` (the anchor's splits,
  where each half carries its own control).
- **Also shipping here, from #342 (#327), which landed on `main` after `v0.14.0` was tagged and
  bumped nothing**: `reach_purity_floor` is now a resolved parameter that
  `competition_digest()` serializes, closing that leg of #317's registered debt by the digest
  rather than by the version lever. It moved the digest `f32f0f9c…` → `80558072…` on its own and
  **no scored number moves with it**. ⚠️ It also recorded the expectation that the rebuilt `-r3`
  bundles come out numerically identical to `-r2` on every member. **That expectation is dead, and
  so is its first correction.** #343 moved `pds_cosine`, and then #348 and #351 joined the same
  wave — so FIVE of the six members are expected to move, and only
  `expr_mse_unbiased_capped_norm` may come out inert. Even that one is inert only if the replicate
  anchor's #348 budget stays non-binding (1.54/1.38/1.58× its claim on val A/B/C), not because
  #348 spares the member by construction. Verify against the MEASURED `-r2` → `-r3` table, never
  against an expectation of equality.
- **BREAKING (competition rule, same `rule_version` 3 wave): `expr_mse_unbiased_capped` bounds
  the prediction's TOTAL sampling correction by the submission's OWN across-perturbation spread**
  (#348, the measured half of #294). The subtracted term is now
  `r · min(jk_pred, PRED_TRACE_CAP_K · jk_real)` with
  `r = min(1, B_pred / Σ_q w_q·min(jk_pred_q, k·jk_real_q))`, where `B_pred` is the submission's
  across-perturbation **centred sum of squares** over the whole predicted panel when a driver
  supplies it (which is what makes the value independent of how a run is partitioned), otherwise
  over the rows the call scores — weighted per row by
  `w_p = 1/|G_p|` on both sides, each row's own target gene left out, and the per-gene centring taken
  over the rows that score that gene.
  #247's cap alone bounds the correction at a CONSTANT `k · jk_real` and SATURATES, so a submission
  whose per-cell scatter cancels in the pseudobulk collected that constant against a plug-in
  distance containing none of it — the delete-1 cell jackknife measures WITHIN-set dispersion,
  which equals the submitted pseudobulk's own error only for exchangeable cells, and nothing
  enforced exchangeability. MEASURED on the official `-r2` val panels: pinning the per-(p, g) sums
  of an honest control-paste arm — same profile, same 400 cells, same depth, nothing else changed,
  no flood and no target-gene manipulation — moved `expr_mse_unbiased_capped_norm` from **0.0000 to
  0.9031** `from_baseline`, and a dev-leaderboard submission took **+0.1389 of a 0.2295 OVERALL**
  through it, its single largest member. Under the bound those arms return to ~1.0, the "predicted
  the control" value.
  Because biological signal only ADDS, `B_pred` tracks the correction a submission is owed, derived
  from the submission itself — no threshold and no model of the submitter's emission. ⚠️ It is a
  **conservative budget, not an unbiased upper bound**: `E[B_g] = B_g(μ_true) + Σ_p w_p σ²_pg −
  (Σ_p w_p² σ²_pg)/(Σ_p w_p)`, which at equal weights sits `1/n_g` (0.33% at P=300) below what the
  panel is strictly owed. The two cannot both be had — the missing degree-of-freedom factor is
  exactly the rebate below. ⚠️ **Both oracle fixed points now carry the condition `r = 1`**, the
  perfection point at 0 as well as the no-skill point at 1: a truth-matched sampled prediction on a
  binding panel retains `(1−r)·C_pred` in expectation.
  - **Both sides are formed in SCORE units**, weighted by each row's own `1/|G_p|` divisor. Since
    #172 that divisor is `G − 1` where the label resolves and `G` where it does not, so a panel with
    partial resolution has two exchange rates between raw gene units and the units the metric
    reports — and in raw units a submission could buy budget on its cheap `G` rows and spend the
    unlocked correction on its expensive `G − 1` rows. Worth `G/(G−1)` per unit: 1.00005 on the
    official 18,533-gene panels, 1.5 on a three-gene one, and reproduced by the review on a
    guide-level construction. No official number moves (every target resolves, so the weight is a
    constant and cancels).
  - **A centred SUM of squares, allocated proportionally, and with no tolerance multiplier — all
    three deliberate.** The sum is exactly what the numerator CHARGES for across-perturbation
    variation, so buying budget is break-even at worst; a per-row variance ceiling (`ddof=1`, times
    `P` rows) or any multiplier `t` pays the submitter a **rebate** of `t·P/(P−1) − 1` per unit
    injected. MEASURED on a P=300 panel with the cross term projected out: a `1.1 × Σ_g Var_p`
    ceiling rebates **+10.37%**, `1.0 ×` still rebates +0.33%, and this form measures −0.003%, i.e.
    exactly break-even (found by the cross-provider review; regression
    `test_injecting_spread_can_never_pay_for_itself`). Proportional allocation is what keeps it
    neutral to uneven cells per perturbation, which the competition rules permit.
  - ⚠️ **What it does NOT close — two residuals, both disclosed rather than claimed away.** The
    budget contains the submission's biological signal, and signal cannot authenticate sampling
    error. (1) **Non-binding regime:** a submission whose GENUINE across-perturbation spread already
    exceeds its claim has `r = 1` and can still pin its aggregates — that is where the replicate
    anchor sits (budget 1.54/1.38/1.58× claim on official val A/B/C, measured at the anchor's own
    half depth, since `anchor._score_one_split` scores one half against the other), so an ACCURATE
    submission is not protected by this bound at all. (2) **Truth-aligned overscaling inside the binding regime:** for a prediction `α·d` against
    a real centred component `d`, the plug-in error is `(α−1)²‖d‖²` while the budget is `α²‖d‖²`, so
    the difference keeps improving past `α = 1` until the budget crosses the claim. It requires
    `B_pred(at truth) < claim`; the official panels are 1.37–2.05× outside that, and the algebra says
    truth is globally optimal whenever it fails, but that is a property of the panel rather than of
    the bound. The candidate fix — also cap by the reference's centred sum of squares,
    `r = min(1, B_pred/claim, B_real/claim)` — closes the direction and is now MEASURED on all
    three contexts and DECLINED. It is inert for every FULL-PANEL submission and for the anchor:
    `B_real/claim_max` = 2.047/1.715/2.050 over the reference's non-control rows, which bounds any
    submission's claim through #247's per-row cap (`_row_weights` gives a row `1/G` or `1/(G−1)`
    from its own resolution alone, independent of the row-set size, so the all-reference total
    bounds every subset's), and 1.54/1.38/1.58 for the anchor at its own depth. ⚠️ A genuinely
    PARTIAL panel is not certified by those ratios: `B_real` is then formed over the same subset and
    a biologically narrow slice has less spread, so the numerator moves too. And on the panels where it
    would bite it is expected to withhold correction from the ANCHOR at least as much — halving the
    depth doubles the sampling half of the budget and leaves the genuine spread alone — i.e. risk
    the scale inversion this design exists to avoid. What the measurement is good for instead is a
    real-side-only PANEL check at bundle-build time, which cannot touch the scale. So this narrows the channel to near-flat
    submissions — where it was measured, and where the live exploit sat — rather than removing the
    class. Closing the class needs both ends of the scale on a matched emission, or a submission
    format that forces a stochastic emission.
  - ⚠️ **The stored ends of the scale do NOT move, unlike #343's.** Measured: the baseline arm's
    400 cells per group are identical, so its `jk_pred` is exactly 0 and there is no correction to
    scale; the replicate anchor's budget is 1.54/1.38/1.58× its claim on val A/B/C at its own half
    depth (real across-perturbation biology), so `r = 1`. Submissions' scores move; baseline and
    anchor VALUES do not.
    ⚠️ **That does not make the `-r3` rebuild a re-stamp, and an earlier version of this entry said
    it did.** The `-r2` anchor ARTIFACT cannot be reused at all: `_ONTARGET_EXCLUSION_SEMANTICS`
    1 → 2 moved `anchor_semantic_identity` (`566a0f3e…` → `1be2fea1…`) and `validate_anchor`
    refuses it — verified by running `score --anchor` against an `-r2` bundle at this version. So the
    anchor leg is RECOMPUTED for `-r3` (5 splits × 3 contexts of GPU work); #348 is merely expected
    to leave its numbers where they were, while #343 moves its `pds_cosine` outright.
  - ⚠️ **The correction is BOUNDED, not dropped or re-estimated, and that is load-bearing.**
    Dropping it and bootstrapping the submitted cells were both measured, and both INVERT the
    scale (anchor 0.0282 → 1.9031 and ~1.86, against a baseline of 0.9858): the prediction-side
    correction is what makes this member emission-invariant, and the baseline is a smooth
    zero-dispersion mean while the anchor is real cells with full emission noise. #294's own
    preferred direction — a nominal estimate at the submitted depth — does not close the channel
    either: a pinned arm submits the reference's own cell count at the reference's own depth, so a
    nominal estimate lands where the cap already is.
  - ⚠️ **What it costs an honest arm.** With no slack, an arm whose budget lands just under its
    claim forfeits the difference: measured 2.6% of the member's range for the honest
    control-paste. An arm with ZERO across-perturbation spread forfeits everything, and one such
    construction is honest — reusing ONE emitted control cell block for every perturbation, whose
    noise is real but perfectly common-mode. Common-mode error is not identifiable from BIAS in a
    single submission, so no estimator can credit it without also crediting systematic error. No
    reachable score moves: such arms read ~1.0 against a measured baseline of 0.9858, so their
    `from_baseline` is 0 before and 0 after. The consequence is that #172's exact-1.0 anchor
    identity now has a second precondition (`r = 1`), so that test asserts it with the bound
    disabled AND asserts the bound's exact effect separately.
  - **`dispatch_anndata_metrics` gains `pred_bulks_full`**, and `scale.py`'s two streaming drivers
    pass the pre-`_restrict` dict. This is the one term whose value depends on which OTHER predicted
    perturbations are present, so a partial run (`--subset` / `--fraction`) would otherwise make the
    member partition-dependent — and a subset is not merely a noisier estimate, since a
    biologically narrow slice genuinely has less spread. With the full bulk every partial forms the
    same `r`, so concatenated fractions match a whole-panel run bit for bit
    (`test_a_partial_run_matches_the_whole_panel_when_the_full_bulk_is_supplied`). Where no fuller
    panel exists — a submission that genuinely does not cover the real panel — the subset is used
    and the run WARNS. Signature-filtered, so only this metric receives it.
  - `PRED_CORRECTION_BUDGET_FLAG_RATIO = 0.7` adds a log-only WARNING at `r < 0.7`. Measured
    separation: honest arms 0.974–0.996, the two pinned arms 0.006 and 0.000. It changes no number.
  - **The scale registry mints `low-random_high-1_v10` and retires `_v9`.** A shipped scale is
    immutable and "what a keyed metric MEANS changed" is the case the rule names explicitly, so this
    is the fourth such mint — and the first to cover two changes in one wave, exactly as
    `rule_version = 3` does. ⚠️ It also pays an outstanding debt: **#343 changed `pds_cosine`'s
    meaning and shipped in `1c05408` with no mint**, so `_v9` had already begun to span two
    definitions of a keyed member. Table byte-identical to `_v2`…`_v9`'s for the fifth mint running
    (proved by byte-comparison, and the digest test now ties its literal to both predecessors'
    shipped digests); `scales_digest()` moves
    `8542ae14…` → `c49271e6…`.
  - ⚠️ **A one-perturbation cohort now gets NO prediction-side correction** (`B = 0`, so `r = 0`),
    where before it got `min(jk_pred, k·jk_real)`. That is the conservative direction: at n = 1 there
    is no across-perturbation spread to estimate, and the alternative — skipping the bound — handed
    a one-target panel or a one-row streaming subset the pre-#348 formula.
  - `_ONTARGET_EXCLUSION_SEMANTICS` 1 → 2, retiring warm result-cache entries for runs that
    request `expr_mse_unbiased_capped`. Neither the version string nor `competition_digest()`
    covers this — the result cache keys on (inputs + config) with `cell_eval2_version`
    deliberately absent, so a pre-#348 run at the same version reproduces the key exactly and its
    now-wrong value would be served in preference to recomputing. The same counter is what
    `anchor`'s semantic params and `partition`'s result semantics read, so those caches move with
    it. It over-invalidates the funcs in the gated set that #348 does not move (including the
    uncapped `expr_mse_unbiased` audit column, which is untouched because #348 is part of the CAP);
    that is the same trade the counter already documents, and the precedent is
    `_PDS_EXCLUSION_SEMANTICS` 2 → 3 at #343.
- **BREAKING: `pds_l1` / `pds_l2` / `pds_cosine` values move under `exclude_target_gene=True`**
  for every profile except `v1` and `cell-eval-0.7.6`, which pin `exclusion_scope: row` and are
  byte-identical to `0.14.0`.
- `anchor`'s `_SEMANTIC_DISCRIMINATION_FIELDS` gains `exclusion_scope`, so an anchor frozen under
  one scope cannot be enrolled against a run scored under the other.
- **Fixed, pre-existing: `tie_policy` was missing from the same tuple.** It is consumed at
  dispatch and it moves a scored member — on a fully tied row, which is what a control-pasting
  submission produces under cosine (the #282 case), `pds_cosine` reads `{0.5, 0.5}` under
  `"midrank"` and the target's ALPHABETICAL index under `"position"` — yet an anchor frozen under
  one policy carried the same semantic identity as a run scored under the other. Found by the
  cross-provider review of #343 and fixed in the same wave, where it costs nothing: `rule_version`
  3 invalidates every artifact regardless. `tests/test_anchor_artifact.py`'s mutation matrix now
  covers both knobs (verified to fail against the pre-fix tuple).
- `_PDS_EXCLUSION_SEMANTICS` 2 → 3, retiring pre-`0.15.0` warm result-cache entries for
  exclusion-enabled `pds_*` runs. Runs with no `pds_*` metric, or with exclusion off, keep their
  caches: `exclusion_scope` is dropped from the config digest where it cannot move a number.

### Added
- `DiscriminationParams.exclusion_scope` (`"panel"` | `"row"`), and
  `distances.resolve_panel_columns` / `distances.panel_reduced` — the panel-scope counterparts of
  `resolve_exclusion_columns`, resolving through it so the two scopes cannot drift on what "this
  perturbation's target gene" means. `panel_reduced` raises rather than returning a degenerate
  0.5 when the panel covers every measured gene.
### Also shipped in 0.15.0, unreleased on `main` when this section was cut

These landed after `v0.14.0` was tagged and bumped nothing, so 0.15.0 is the release that
carries them. Their entries are reproduced verbatim.

#### `fix(norm)!` — the counts integrality gate is an ABSOLUTE tolerance (#341)

- **The counts integrality gate is an ABSOLUTE tolerance, so a submission reaching
  `validate_input_type` can no longer sit more than a NOMINAL `1e-6` from an integer.**
  `norm._is_all_integer` tested integrality with `np.allclose` at numpy's
  DEFAULT tolerance, which is RELATIVE (`atol + rtol*|b|`, `rtol=1e-5`) — so the deviation
  accepted as "still an integer" scaled with the value: ±0.001 at 100, ±0.01 at 1,000, ±0.5 at
  50,000, and above 50,000 every value whatsoever. MEASURED before the fix: `5000.5` was rejected
  and `50000.5` and `999999.5` were **accepted** as counts under `allow_fractional_counts: false`.
  Nothing else bounded it — `check_scale_limit` caps the per-CELL total at `1e6`, so a single
  entry at 50,000+ is legal. Both call sites now pass `rtol=0` with `norm._int_atol(dtype)`, the
  nominal `1e-6` of `norm._INT_ATOL` as the input's own dtype represents it.

  `docs/vcc2026_metrics/vcc2026-metrics.md` §0 — the published metric specification — already says
  values "must be raw, untransformed counts: non-negative, integral, and with no cell total above
  $10^{6}$. A matrix failing any of these is rejected rather than scored." The document does not
  move. ⚠️ Nor does the code now *enforce* that sentence: it closes the value-scaled hole in the
  **proxy** for it, leaving an integral-to-`1e-6` test on the paths that reach
  `validate_input_type`. Both qualifiers are load-bearing and both are spelled out below.

  **Not exact equality — ruled, not defaulted.** Exact equality was put to Alex explicitly, with
  `codex-review` recommending it independently as the specification-correct rule; he ruled on
  2026-08-19 to keep the tolerance, so `1e-6` is the decision rather than an oversight. The reasoning
  behind it: float32 holds every integer exactly to `2**24` = 16,777,216, sixteen times the per-cell
  cap, so `(x == rint(x)).all()` would be defensible on the width argument alone — but tightening a
  validation gate is the one change that can turn a
  previously-accepted submission into a hard refusal, and `1e-6` keeps a noise margin for an
  otherwise-integral matrix carrying dust from an upstream float64 transform. ⚠️ That margin is
  float64-shaped: at `1e6` it is ~8,590 float64 ULP but 1.6e-5 of a float32 ULP. ⚠️ And the bound is
  `1e-6` **as the input's dtype represents it**, which for float16 is `1.0132789611816406e-06` — 1.3%
  wider. `norm._int_atol` performs that rounding explicitly rather than leaving it to numpy's
  promotion rules, because `pyproject` declares `numpy>=1.26`, a range spanning the NEP 50 switch.
  ⚠️ It pins an outcome both promotion regimes already share rather than repairing a divergence:
  under 1.26 the tolerance stays in-width because the legacy ufunc promotion of `atol + rtol*abs(y)`
  value-minimizes `1e-6` to the array's width — *not* because of `result_type(y, 1.)`, which fixes
  `y`'s dtype and never sees `atol`. So what the cast buys is that a future promotion change cannot
  move the floating-dtype tolerance *value* silently; it freezes nothing about `rint`/`isclose`, and
  bool and complex stay on the uncast path by design. Verified a no-op on numpy 2.5.2
  across float16/32/64 and longdouble, including at float16's 17-vs-18-subnormal
  boundary. `bool` and `complex` reach the same loops and are deliberately NOT
  cast (`np.bool_(1e-6)` is `True`). MEASURED, the
  exact float32 crossover is a binade edge: a one-ULP-off value is still accepted below 16 (spacing
  9.54e-07 in `[8, 16)`) and rejected from 16 up (spacing 1.91e-06). So on **float32** input this
  is exact equality **at 16 and above** — and what the gate enforces is "integral to `1e-6`", not
  literal integrality: float64 `100.0000005` and float32 `10.000000953674316` both pass, and the
  accepted dust is not rounded before scoring.

  **Which inputs change verdict, and which do not.**
  - Rejected now, accepted before: a matrix with every value inside the old *relative* tolerance
    of an integer and at least one outside `1e-6` of it — `1000.001`, `50000.5`, `999999.5`.
  - Unchanged: every exactly-integer matrix at any dtype, integer dtypes (never scanned), the
    `1e6` cap value itself, and float64 dust up to `1e-6` around a NONZERO integer (around zero it
    moves — see below). Also unchanged, and both pre-existing: a
    NaN matrix reads as not-all-integer under both rules so `counts` refuses it as "fractional", and
    a `+inf` matrix reads as ALL-INTEGER under both (`np.allclose(inf, rint(inf))` is True) so this
    gate accepts it — `check_scale_limit` is what refuses it, on a per-cell total of `inf`.
  - Not reachable through the loosening below on a **NaN-free** matrix: NEGATIVE dust.
    `validate_input_type` tests `_min_value(X) < 0` before it computes `_is_all_integer`, so `-5e-07`
    is refused on non-negativity and never reaches the tolerance. That is ORDER, not tolerance, and
    it is pinned by a test because the loosening is what makes the interaction worth stating.
    ⚠️ The NaN-free qualifier is load-bearing and the hole is pre-existing: `_min_value` reduces with
    `np.min`, so one NaN makes the minimum NaN, `NaN < 0` is False, and the sign check does not fire
    — a matrix holding both NaN and `-5e-07` is accepted under `allow_fractional=True` and under
    `lognorm`. `check_scale_limit` is what refuses it. Also pinned, so the ordering claim above is
    not read as unconditional.
  - ⚠️ **Accepted now, rejected before — this is NOT a pure tightening.** `np.allclose`'s tolerance
    is `atol + rtol*|b|` with `b = rint(v)`, and `rint(v) == 0` kills the relative term, so in the
    neighbourhood of ZERO the old rule was `atol=1e-8` alone — 100× *tighter* than the new absolute
    one. MEASURED: a value of `5e-07` was not integral and now is. So a counts matrix whose only
    non-integrality is sub-microcount dust near zero is newly **accepted**. Kept rather than
    special-cased: a uniform absolute tolerance is the coherent rule the relative one was not, and
    `1e-6` of a count on a value whose nearest integer is 0 is what "dust" means — reinstating
    `1e-8` below a threshold would buy monotonicity with a second, unmeasured constant.
  - `allow_fractional_counts=True` still bypasses the gate entirely. Every tiled baseline leg
    depends on it (`real_bundle._baseline_leg`, `baseline.build_generic_baseline`), because a mean
    profile is fractional by construction, and bundle builds are untouched.
  - `validate_input_type`'s OTHER raise — `input_type='lognorm'` on all-integer values — fires
    LESS *away from zero*, since a stricter predicate returns `False` more often; and near zero it
    can newly **fire**, because an all-dust matrix now reads as all-integer with a positive max,
    which is exactly what that raise tests for (MEASURED on a `5e-07` matrix). Both directions are
    inert on one representative lognorm fixture, which is a measurement rather than a proof over
    all lognorm data: a lognorm value is `log1p` of a CPM-scale quantity, so under the `1e6` cap it
    cannot exceed a mathematical `log1p(1e6)` = 13.8155 — storage rounds, and float32 stores that as
    13.815511703491211, just *above* the float64 value — so the bound that matters is the nearest
    integer, 14 under either storage, where the old tolerance was at most 1.4001e-04; and either rule
    needs EVERY value that close to an integer. MEASURED on the 500 × 2,000 CPM-normalized Poisson
    matrix the test itself builds: of its 777,297 nonzero entries, **0** are within either tolerance
    of an integer — only the exact zeros qualify, under both rules.

  ⚠️ **No `rule_version` bump, and no bundle rebuild ON THIS ACCOUNT.** This changes which inputs are
  ACCEPTED, not what any accepted input scores: every previously-valid integer matrix scores
  bit-identically. And `competition_digest()` does not move — **measured against this branch's base**,
  `80558072…` both with and without the change — because this changes no config FIELD, only what the
  code does with one. ⚠️ Not because the digest is blind to validation: it embeds `rule_config_hash`,
  which deliberately retains `allow_fractional_counts`, `validate_input` and `max_counts_per_cell`.
  An earlier draft of this entry gave that as the reason and it was wrong (codex review).
  `scales_digest()` is likewise unchanged at `8542ae14…`.

  ⚠️ **The `-r3` bundle rebuild is still mandatory, and #342 is why, not this.** #342 added
  `reach_purity_floor` to `competition_payload()`, moving `competition_digest()` `f32f0f9c…` →
  `80558072…`, so the three official `-r2` bundles already fail `score.py`'s rule gate. Read
  "no bundle rebuild" here as "this entry adds no reason for one", not as "none is needed".

  ⚠️ **Three limits this does NOT close, all pre-existing and all RULED rather than overlooked**
  (each raised by codex-review, each put to Alex, each declined 2026-08-19):
  1. **The partitioned/streaming prediction path never reached this gate in the first place.**
     `partition_inmem.score_piece` materializes a pred piece with neither `validate_input_type` nor
     `check_scale_limit` — a gap `check_scale_limit`'s own docstring already records — and the
     direct shard/cell drivers classify from a 2,000-row peek at fractional ROW TOTALS
     (`guess_is_lognorm`), not elementwise. A fractional piece can still be scored through those
     drivers. Closing it means newly validating every partitioned submission, which is a change of
     a different size than a tolerance and is not smuggled in here.
  2. **A results-cache hit is not re-validated.** A backed (path) input skips the up-front
     validation loop and a warm result-cache hit returns before the deferred sites run, so a
     submission cached before this change can still be served from that cache.
     **RULED (2026-08-19, Alex): no cache term.** Neither `CACHE_FORMAT_VERSION` nor a scoped
     `_*_SEMANTICS` key term — the mechanism codex-review recommended, and the one
     `_PDS_EXCLUSION_SEMANTICS` was minted for — because usage to date has been light and the existing
     caches carry nothing worth protecting. ⚠️ The rationale is that ruling, NOT a claim that cache
     keys only carry value-affecting semantics: `validate_input`, `allow_fractional_counts` and
     `max_counts_per_cell` are already explicit key terms precisely because a cache hit skips the gate
     (#161), and an earlier draft of this entry argued the opposite (codex review). Adding one is a
     contained change if the calculus ever shifts: it would touch the result, pseudobulk and DE keys
     in `run.py` only, and provably NOT `partition.result_semantics` or
     `partition_inmem._bundle_semantics`, so existing bundles stay readable. Until then: re-run
     WITHOUT `--cache-pred` (it is a path, not a toggle) for a checked answer.
  3. **`noise.py`'s downsample guard has the same relative-tolerance defect.** It refuses input with
     "requires integer count data (fractional/normalized input would be silently truncated)" after an
     `np.allclose` at the DEFAULT tolerance, so MEASURED, a `50000.5` matrix passes it and is then
     truncated to `50000` by `astype(np.int64)` — the exact silent truncation the message exists to
     prevent. **RULED (2026-08-19, Alex): leave it.** It is off every scored path — `noise=` is a
     Python-API/tools-only parameter with no CLI flag and a `None` default, reached only from
     `tools/scale/run_scale.py` and `tools/rowstore/simulate_rowstore.py` — and it is expected to be
     retired with the rowstore format. A fix would also have to round rather than truncate the dust it
     accepts, which moves numbers on that path; not worth it for a vehicle on the way out.

## [0.14.0] — 2026-08-18

**BREAKING on four axes, and every baseline, anchor and real bundle built before this release is
invalidated.** `score` compares `cell_eval2_version` exactly, so nothing stamped `0.13.0` pairs
with a `0.14.0` submission — and this release moves the competition rule and the scale registry as
well, so a `0.13.0` bundle would not enrol even if the version string were ignored:

1. **The competition rule.** `rule_version` 1 → 2, and `competition_digest()` moves
   `1b93878b…` → `f32f0f9c…` (#317). Version 2 means the three metric-semantics changes the digest
   structurally cannot see: #172's target-gene exclusion, `de_wilcoxon_direction_reach_raw`'s
   calibrated purity floor, and #271's wide pseudobulk reduction. The three official #276 val
   bundles are rebuilt against it in the same wave (`-r2`).
2. **The scored values.** Five changes moved scored `vcc2026` numbers since `v0.13.0` — #316
   (#172's exclusion), #321 (`de_*_lfc_nmae` on a straight line below the baseline), #320 (the
   flat clip at 0 removed from the four bounded members), #322 (the calibrated reach purity
   floor) and #330 (#271: `_grouped_sums` reduces WIDE, which moves `pds_cosine` and
   `expr_mse_unbiased_capped_norm`). A submission's `avg_score` moves. #318 ships here too but is
   a documentation correction to the `lfc_nmae` anchor — identical executable code, no number
   moved.
3. **The scale registry.** `low-random_high-1_v8` is retired and `low-random_high-1_v9` minted —
   a "the numbers under a shipped name moved" mint for #271, claiming no ordinal (`scales.py`
   says why: `_v6`/`_v7`/`_v8` are the "what a keyed metric MEANS changed" sequence and this is
   not one of those). The table is
   byte-identical to `_v8`'s and no shipped score moved with the rename, so the only thing moving
   `scales_digest()` (`22b3d6b1…` → `8542ae14…`) is the name itself.
4. **The public API and packaging.** The out-of-core h5ad-manifest entry point is now
   `score_h5ad_manifest` -- a BREAKING rename, spelled out with its two renamed modules in the
   Removed section below, with **no deprecation alias** (#326); and the `deseq2`
   optional-dependency extra is gone, so `pip install -e '.[deseq2]'` warns that the extra does
   not exist and installs the base package WITHOUT `deseq2_gpu` (#328).
   The DESeq2 backend itself — every `de_deseq2_*` metric and `de.backend="deseq2"` — is
   untouched.

### Changed
- **BREAKING (competition rule): `rule_version` 1 → 2, and the debt outstanding against it is now
  empty** (#317). `competition_digest()` moves
  `1b93878bf645e7ed63d67382b6eb6c60e0a7c6aba3898ec3fbdb183ca3a55e16` →
  `f32f0f9c43c8a9faf448bd9d8445eb6929fbc3ed4612ff806332235e5c2c4cc2`, by hand — which is the only
  way it can move for a change to what a member MEANS. The payload freezes each member's RESOLVED
  scoring policy (direction, anchor, penalty, clamps, `agg`, the derived pair, normalization,
  `worst_value`, estimator) and nothing about which gene set the member computes over, or which
  constant its arithmetic thresholds on, or which pseudobulk that arithmetic reads. Three such
  changes were owed and are paid here: #172's target-gene exclusion on three scored members,
  `de_wilcoxon_direction_reach_raw`'s purity floor at the calibrated 0.9 rather than the derived
  `1 - alpha/2` = 0.975, and #271's wide `_grouped_sums` reduction. The two other digest
  moves inside the 0.13.0 gap — `ERROR_LINEAR` and the clip-at-0 removal — were POLICY-field
  changes that moved the digest on their own and owed no bump; they are not what version 2
  records. **Every real bundle built before this release is invalidated by design**: `score.py`
  refuses a stale `rule_digest` outright, and a bundle carrying none is diagnostic rather than
  enrolled. The debt list lives at the literal in `competition.py` so whoever bumps next can see
  what the version means; this bump empties it and keeps the instruction to add to it, never
  replace it.
- **BREAKING (public API): the out-of-core h5ad-manifest entry point is now
  `score_h5ad_manifest`, and two modules are renamed** — `cell_eval2.h5ad_manifest` (the
  manifest.csv + dense `.h5ad`-pair reader) and `cell_eval2.rowstore` (the `plan.json` +
  `artifact_<NNNNN>/` memmap reader). Modules are named after the artifact FORMAT they read
  rather than after the pipeline that writes it, so nothing in the PUBLISHED tree names an
  unreleased internal producer (internal plan/spec/audit documents and `tools/` keep their
  historical wording; none of them ship). **No deprecation alias** — an alias would reintroduce exactly the name
  this change removes, the same reasoning that removed `expr_mse_unbiased_norm` outright in
  #257. `tools/` and the four `tests/test_*` modules follow the same two names.

  `MemBudget`, `ScoreResult`, `read_rowstore_plan` and `score_rowstore` are unchanged and keep
  their top-level re-exports, so `from cell_eval2 import score_rowstore, MemBudget` is
  unaffected; `score_h5ad_manifest` is the only public symbol whose spelling moved. There is no
  CLI surface — no flag or subcommand changed. Three reader diagnostics are relabelled (the
  manifest-not-found and empty-manifest errors, and the pred/real output-space mismatch
  `ValueError`); no test asserts on their text.

  **Value-neutral by construction, and verified.** `competition_digest()` hashes the rule's
  field VALUES and `scales_digest()` the scale payload; neither reads a module name, and
  `source_fingerprint` fingerprints input DATA despite its name. Both digests are bit-identical
  to the pre-change tree (`1b93878b…` / `22b3d6b1…`), so no real bundle is invalidated by this
  change on its own.

  Historical entries below now spell these symbols by their current names. What each entry
  describes is unchanged — only the label moved.

- **BREAKING (competition scoring): the flat clip at 0 is removed from the four bounded
  `vcc2026` members** — `pds_cosine`, `de_wilcoxon_direction_fidelity_yield_raw`,
  `de_wilcoxon_direction_reach_raw` and `de_wilcoxon_sig_jaccard` (and their `de_deseq2_*`
  siblings, which must not change policy because the DE backend changed). A submission below
  the comparator now scores its true negative value instead of `0.0`, on both the baseline and
  the replicate scale.

  A `[0, 1]` metric scored against an anchor cannot produce an unbounded term — `(u-b)/(a-b)`
  is bounded below by `-b/(a-b)` for every valid `u` — so the clip was truncating real signal
  rather than protecting the average. Measured on the three official val bundles (replicate
  scale) the truncated depth ran to `-1.17/-1.26/-1.22` (`pds_cosine`),
  `-1.58/-2.68/-1.85` (`direction_fidelity_yield_raw`), `-0.05/-0.12/-0.12`
  (`direction_reach_raw`) and `-0.08/-0.07/-0.10` (`sig_jaccard`).

  ⚠️ Those are measured on the `-r1` bundles, which predate this whole release wave: two of
  the four members changed what they compute, so their `b` and `r` — and therefore these floors
  — shift once the #317 wave rebuilds. ⚠️ Not for the same reason, and #316 is not the cause for
  both: `sig_jaccard` moved in **#172/#316** (it began excluding each perturbation's own target
  gene), while `direction_reach_raw` **already excluded** its target gene before #172 — its
  §5 table there shows Δ = 0, "already excluded" — and moves here because of the **purity-floor**
  change instead. The mechanism and its
  sign do not: `-b/(a-b)` is deepest where the comparator sits nearest the replicate, and
  `fidelity_yield_raw`'s comparator is pinned at chance by construction either way.

  ⚠️ **The six-member `avg_score` no longer floors at a constant `-1.0`.** It floors at
  `-1.48` / `-1.69` / `-1.55` on val A/B/C. `DEFAULT_PENALTY_CAP = 6.0` was tuned in #276
  part C to produce the old constant; it keeps its value but no longer has that meaning.

  ⚠️ **Every real bundle built under the previous rule must be rebuilt.** The competition rule
  digest moves `eed507d2…` → `1b93878b…`, and `score_metrics` **raises** rather than scoring a
  submission against a bundle built under a different rule (verified end to end against the
  real `vcc2026-valA-r1`).

  Note the contrast with #172, whose note now sits directly above this constant in
  `tests/test_competition_rule.py`: that change moved VALUES and left the digest standing,
  because the digest freezes policy rather than what a member computes. This one moves a
  policy field, so it is caught rather than owed — it does NOT go on `competition.py`'s
  `rule_version` debt list. What the two share is the consequence: the three #276 val bundles
  are rebuilt together at the #317 release wave, not per-PR (Alex, 2026-08-17).

  Scope is verified, not asserted: 80 of 87 scored catalog entries are bit-identical over a
  408-case sweep, the frozen scale's scores are unchanged (`_v7` at the time of that sweep; this
  release ships `_v9`), and the
  frozen v1 path (`compat.score_agg_metrics`) is untouched — it carries its own hard-coded
  `max(0.0, …)` and never reads `clamp_low`.

- `Scoring` gains `metric_min`, the metric's structural worst value. A scored policy may now
  be genuinely unfloored (`clamp_low=None` with `penalty="none"`) provided it declares one:
  the sentinel a missing / non-finite / overflowing value takes is then the score that worst
  value earns, so `avg_score` still cannot be dragged to `-inf`. `metric_min` requires a
  non-None anchor (the anchorless normalization has no second end to bound the sentinel) and
  must sit on the worse side of it. `scales.scale_payload` and `competition.competition_payload`
  both serialize it; `scales_digest()` moves `2a9a51e1…` → `a22b98e2…` as a **schema-only**
  bump on top of #172's `_v6` → `_v7` rename — no shipped scale is unfloored and none of their
  numbers moved (`test_the_metric_min_bump_moved_no_shipped_score` replays the pre-change
  values bit-exactly via `float.hex()`). No mint **for this change**: unlike the `_v7` rename,
  nothing about what a member means moved with it. ⚠️ `a22b98e2…` is not what this release
  ships: the purity-floor change below mints `_v8` on top and #271 then mints `_v9`, so the
  shipped digest is `8542ae14…`.


- **BREAKING (competition rule): three scored `vcc2026` members stopped scoring each
  perturbation's own target gene**
  (#172,
  #316) —
  `de_wilcoxon_sig_jaccard`, `de_wilcoxon_lfc_nmae` and both legs of
  `expr_mse_unbiased_capped_norm`. It landed while `main` still stamped 0.13.0, as did the
  `ERROR_LINEAR` move, the clip-at-0 removal and the purity floor; **0.14.0 is the release that
  carries all four**. Two of them — this one and the purity floor — are what `rule_version` 1 → 2
  pays (#317); `ERROR_LINEAR` and the clip-at-0 removal moved `competition_digest()` on their own,
  so they owed no bump and are not what version 2 records.
  ⚠️ For anything BUILT inside that window the version stamp is still not what separates a pre-
  from a post-change artifact — every such pair is an all-`0.13.0` pair — so the value-keyed
  semantics terms remain what does. Full treatment in
  `docs/metrics.md` §2.3 and §4.3
  and `tests/test_target_gene_exclusion_172.py`; it was also the first entry on the
  `rule_version` outstanding-debt list in `competition.py`, which this release pays and empties.
- **BREAKING (competition rule): `de_*_lfc_nmae` scores on a straight line below the baseline
  instead of a Box–Cox tail.** The policy moves from `scoring.ERROR` to the new
  `scoring.ERROR_LINEAR`: same `direction="lower"`, same `anchor=0.0`, same `-6.0` floor, and
  bit-identical at and above the baseline — `penalty="boxcox"` becomes `penalty="none"` and the
  floor is declared as `clamp_low=-6.0` rather than derived from `DEFAULT_PENALTY_CAP`. Below
  the baseline the score is `max(-6, 1 - r)`, which reaches the floor at `r = 7` where the
  quadratic saturated at `r = sqrt(13) ≈ 3.606` (Alex, 2026-08-17).

  `de_wilcoxon_lfc_nmae` is the only `vcc2026` member with a live sub-zero range — the other
  five clip at 0 — so its shape alone set the entire downside of the six-member competition
  average, and the quadratic made that average a near-binary test of escaping the floor rather
  than a ranking. On the three official val bundles the discriminating band widens from
  `nmae ≤ 2.67 / 2.39 / 2.63` to `≤ 4.83 / 4.20 / 4.74`. The six-member average floored at exactly
  `-1.0` when this landed — the property #276 part C retuned `DEFAULT_PENALTY_CAP` to 6 for,
  carried by `ERROR_LINEAR`'s declared `clamp_low` instead of by the cap. ⚠️ **That no longer
  holds**: the entry above removes the clip from four of the five members whose 0-floor this
  argument rests on, so the six-member floor is now data-dependent. `ERROR_LINEAR` still
  contributes exactly `-6/6` of it.

  ⚠️ **`competition_digest()` moves from `992cc849…` to
  `eed507d239387fcfa06160275ef699876fb4535400f7674bf42e715fe6323a81`, so the three official
  `vcc2026-val{A,B,C}-r1` bundles are refused by `score` and must be rebuilt before they can
  score anything again.** Their stored baseline and anchor values are unaffected — nothing here
  recomputes a metric, only how one of them becomes a score — but `rule_digest` is compared
  exactly, and a bundle built under the old rule is not comparable with a submission scored
  under the new one. **This change does not rebuild them** (Alex, 2026-08-17): that is a
  separate operational step, `prep-real-bundle` at ~11 min per context.

  ⚠️ **`ERROR` itself is unchanged**, so `expr_mae`, `expr_mse`, `delta_mae` and `delta_mse`
  keep the Box–Cox tail; `expr_mae` is in the frozen 2025 `vcc` profile, whose published scores
  must not move. `compat.score_agg_metrics` carries its own clip-at-0 and never read either
  policy, so the v1 parity path is untouched. The frozen scale (`_v9` as shipped) is
  unchanged too — it already scored this metric linearly, at its own `-1.0` floor and against
  its own constant base.

  ⚠️ **The breaking scope is wider than the official competition score.** Every `de_*_lfc_nmae`
  score in the interval `1 < r < 7` moves — `from_baseline` and `from_replicate` alike, on `full`,
  `de` and `vcc2026`, on both DE backends. Outside that interval nothing moves: at `r <= 1` the
  two policies are bit-identical, at `r >= 7` both read the shared `-6` floor, and a non-finite
  value took `-6` before and takes it now. That is why last year's field, at a median `r` of ~24,
  saturates under either shape. `de_deseq2_lfc_nmae` moves with its wilcoxon
  sibling (one metric behind two backends): its own `profiles=()` means no profile selects it
  directly, but `run._effective_de_spec` relabels the wilcoxon name to it under
  `de.backend="deseq2"`, and `_guard_deseq2_metric_selection` permits `vcc2026` under that
  backend, so a **diagnostic** `vcc2026` column and its `avg_score` move as well. The only thing
  the DESeq2 spelling cannot reach is an **enrolled** official competition score — enrolment
  needs a competition bundle, and the relabelled membership is what makes such a bundle
  diagnostic in the first place.
- **The test suite no longer reads this repository's git history.**
  `tests/test_expr_mse_unbiased_ratio.py`'s #257 characterization used to reconstruct
  `delta.mse_unbiased` as it stood at `810215c~1` by shelling out to `git show`, which made it
  depend on one object in one `.git`: it raised — rather than skipped — in any full clone that
  could not resolve that revision, and it needed `fetch-depth: 0` in CI. Both source files are
  now vendored verbatim under `tests/fixtures/pre_247/`, with each file's SHA-256 verified on
  every run, because a drifted characterization fixture would keep passing while comparing
  against something else. The characterization now runs (rather than skipping) in a shallow
  clone and in an unpacked source distribution that is not a git repository at all, and CI
  checks out shallow again.

- **BREAKING — the `corrected=False` reach spellings threshold the purity curve at a calibrated
  `REACH_PURITY_FLOOR = 0.9` instead of the derived `1 - alpha/2 = 0.975`.** `_register_de_family`
  runs for both DE backends, so this is four of the eight registered names —
  `de_{wilcoxon,deseq2}_direction_reach{,_unbounded}_raw`. The four `corrected=True` spellings
  keep their own `q + (1-alpha)(1-q)` null: their threshold and arithmetic are untouched, though
  their cache identity moves with the rest (the predicate is func-identity, deliberately).
  `de_wilcoxon_direction_reach_raw` is a `vcc2026` scored member, so **the three official #276
  competition bundles are stale** and must be rebuilt against this release; previously published
  scores are not comparable. (A *diagnostic* bundle carries `rule_digest = None` and is not
  gated on the rule at all.) (⚠️ Not the *cause* of their staleness as of 2026-08-17: the `ERROR_LINEAR`
  change above already moved `competition_digest()` on its own, so the three #276 val bundles
  were orphaned before this landed. This adds to what the rebuild has to cover, not to whether
  one is needed.)

  ⚠️ **This landed with no version bump of its own** (Alex, 2026-08-18) — *within* `0.13.0`,
  exactly as #172 did; 0.14.0 is the release that carries it. So among artifacts built inside
  that window `cell_eval2_version` — a `SUBMISSION_PEERS` field, compared
  **bundle-against-submission**, never against the running library (`real_bundle.py`) —
  contributes NOTHING: every such pair is an all-`0.13.0` pair and the stamp is equal on both
  sides by construction. That was deliberate and it cost nothing, because the version stamp was
  never the gate. It reads *installed distribution metadata*, which an editable install freezes
  at install time, so a source tree can run new code while reporting an old version; that is
  precisely why the four bindings below key on the floor's VALUE instead. Precedent and
  reasoning are already written down at `baseline.config_digest`: "`cell_eval2_version` is not
  the fallback ... note that #172 lands WITHIN 0.13.0, so the stamp does not move for it at
  all."

  What actually separates a pre- from a post-change artifact, verified in code rather than
  assumed:

  ⚠️ **It does not, in this release.** `ERROR_LINEAR` above moved `competition_digest()`, and
  `score.py`'s rule gate compares the bundle's frozen `rule_digest` against the *runtime* one and
  raises unconditionally on a mismatch — the three #276 bundles carry `992cc849…` against a
  runtime `1b93878b…` (moved by `ERROR_LINEAR` and again by the clip-at-0 removal). So
  historical competition bundles are **refused** by current tooling, not
  merely paired more strictly, and re-scoring them requires the old tooling. (This paragraph said
  the opposite before #321 landed, when it was true.)

  The gap that pairing check did **not** close, and that this release fixes: nothing in the
  result cache key, the partial-result sidecar or the anchor's semantic identity could see the
  purity floor, so a warm artifact computed under the old floor would be served, re-stamped with
  the current version, and pass a bundle's peer check as a first-class submission. All three now
  carry it, following the mechanism the repo already uses for exactly this
  (`_ONTARGET_EXCLUSION_SEMANTICS` at #172, `_PDS_EXCLUSION_SEMANTICS` at #248):

  - `run._result_config_digest` gains a **scoped** `reach_purity_floor` term — a run selecting no
    `direction_reach*` metric keeps its warm cache.
  - `partition.result_semantics` gains it **unconditionally**, `_PARTIAL_SEMANTICS_SCHEMA` 3 → 4,
    so a partial directory straddling the change is refused rather than reduced.
  - `anchor.anchor_semantic_params` gains it when a `direction_reach*` metric is selected, so a
    frozen anchor cannot be reused across the change. The predicate is func-identity, so it also
    fires for the *corrected* variants, which do not read the floor — deliberate
    over-invalidation, in the same direction as `_ontarget_exclusion_used`: a spurious term costs
    one recompute, a missing one is a false hit.

  ⚠️ **One surface is deliberately NOT bound: the loose `--baseline-agg` pairing path.**
  `baseline.config_digest` covers what a run *asked for* and nothing about what a metric
  *means*, so an old-floor baseline and a new-floor submission can pair when both are stamped
  by the same release — which was every case while all artifacts stamped `0.13.0`, and remains
  the case for any two built inside one release.
  That is the known gap **#314**, already ruled deferred (Alex, 2026-08-17) on the grounds that a
  per-issue constant there would be the third copy of one idea and #314 proposes the shared
  registry all of them should read. This change is registered as its **third live instance**
  alongside #172 and #248 (#271 later became the fourth), and is the sharpest case for that
  registry since the floor *is* a
  single value. The `--real-bundle` path is unaffected: it goes through the anchor's semantic
  identity, which now carries the floor.

  Carried as the **value**, not a hand-bumped counter, for two reasons: here the semantics *are*
  one number, so a future retune moves every key with nothing to remember; and it is immune to
  the stale-`__version__` trap, since `cell_eval2.__version__` reads *installed distribution
  metadata* that an editable install froze at install time — a source tree can run new code while
  reporting an old version. DE-table and rank caches are keyed separately and are **not**
  invalidated, so a re-run recomputes metrics over cached DE rather than re-running DE.

  The derivation the old constant came from is sound but computes the purity a *perfect*
  independent repeat attains against an alpha-FDR reference — the metric's own **ceiling** —
  and using it as the pass mark left zero margin. Measured on the three official val lines, a
  real split-half replicate's per-gene directional accuracy is 0.9561 / 0.9391 / 0.9488 (mean
  over targets), *below* the old threshold on every line. Consequences of the old value, all
  measured: the first depth tolerating one sign error was 40, so 46–61% of the 300-target
  cohort had no depth at which one error was survivable (12–30% now); and at a fixed uniform
  95% per-gene accuracy the cohort mean varied 13.5× across `N_conf` strata (1.3× now), i.e.
  identical quality was graded by the reference's budget size.

  Effect on scoring, per val line A/B/C: `from_replicate` of a 90%-accurate submission moves
  0.24/0.43/0.30 → 0.75/0.91/0.76. The baseline arm gains only +0.004…+0.009 while the
  replicate gains +0.039…+0.050, so the span *widens* 4–5%; a perfect submission's member
  contribution eases from 1.097/1.360/1.118 to 1.038/1.277/1.068. The chance floor does not
  move materially — 0.042694/0.065849/0.080890 → 0.042694/0.067557/0.082166, i.e. val A
  exactly unmoved and B/C up by 0.0017/0.0013, which is 0.24%/0.15% of those lines' own spans —
  and the anchor does not saturate
  (0.965/0.802/0.943). The cost is a 16–32% relative gain for a marginal-sign exploit, measured
  against an arm that also holds the reference's own |log2FC| as its ranking key — an upper
  bound. Full measurement:
  `docs/validation/2026-08-17-direction-reach-purity-threshold.md`.

  Two deliberate non-changes. `de_direction_sensitivity` keeps `1 - alpha/2`: it is one of the
  three v0.5.0 metrics whose values must not move, and it is diagnostic-only — so the
  raw-reach == sensitivity identity is now **broken on purpose**, and pinned as such. The
  `corrected=True` reach variant keeps its own `1 - alpha` majority-sign derivation and is not
  a `vcc2026` member.

  Side effect on #291: the
  head-miss omission exploit needs `m > (1 - P0) * N` misses to bite, so on the 80-pair toy it
  now takes 9 rather than 3, and the jump it buys falls from 0.0 → 0.9625 to 0.0 → 0.8875. The
  defect is ~4× harder to reach; it is **not** closed.

  **The scale registry mints `low-random_high-1_v8` and retires `_v7`** — and `_v8` is in turn
  retired by #271 later in this same cycle, so the release ships `_v9`. The
  `de_wilcoxon_direction_reach_raw` *entry* is correctly unchanged — `base = 0.0` is the
  metric's minimum, `anchor = 1.0` its perfect value, and neither depends on the floor (the
  no-skill point is dominated by the `k* = 1` event, which the floor barely touches --
  measured, it moves by 0.0 / +0.0017 / +0.0013 on val A/B/C). That is exactly why the
  NAME has to move: `scales_digest()` covers numbers, there are none to move, and the registry's
  own rule is that a change to what a keyed metric *means* mints a new `_v<n>` — the `_v6` →
  `_v7` precedents (#282's `pds_cosine` tie handling and #172's target-gene exclusion, both with
  byte-identical tables). Without it a published `_v7` column would silently span two
  definitions of a scored member. `scales_digest()` moves because the name does.

  **`competition_digest()` does not move for this, and that is the standing ruling, not an
  oversight.** `competition_payload` freezes each member's scoring *policy* and nothing about
  what the member computes; this change leaves every policy field standing — direction, anchor
  `1.0`, `penalty none`, `agg`, estimator, and the clamps, which #320 independently moved to
  `clamp_low = None` with `metric_min = 0.0` in this same release. The purity floor is invisible
  to the payload either way. #172 hit the identical gap and was
  ruled the same way (Alex, 2026-08-17: no `rule_version` bump until all changes are in — one
  bump for the whole set at the release wave that rebuilds the three #276 val bundles, tracked
  as #317). **That wave is this release**: the change went onto the outstanding-debt list at the
  `rule_version` literal in `competition.py`, and 0.14.0's bump to `rule_version = 2` pays it
  alongside #172 and #271, emptying the list.

- **BREAKING (the pseudobulk two scored members read, on coarse-float input):
  `prep._grouped_sums` reduces WIDE**
  (#271). It evaluated
  `X[rows].sum(axis=0)` *before* casting, so the reduction ran in the input dtype while
  `moments.jackknife_correction` casts `.data` to fp64 first — the `bulk_lognorm` bulk and the
  correction subtracted from it came from different group sums. ⚠️ The **asymmetry** was the
  defect, not either half: the jackknife already reduced wide, and so do
  `streaming_bulk._streaming_pseudobulk_cpu` and `gpu.bulk.GroupedMeanAccumulator`. The resident
  path was the last narrow one, so **the same submission could score differently depending on which
  driver ran it** (measured 3.53e-10 in bulk space). A floating dtype coarser than float64 is now
  widened before the reduction and all three drivers accumulate a coarse float in at least fp64.
  ⚠️ That is the coarse-float statement, not a general one: the GPU accumulator still hands back
  fp32 means, summation order still differs between drivers, `longdouble` input still diverges by
  design **where longdouble is wider than float64** (fp64 is the *narrower* side there; on a platform
  where the two are the same type there is nothing to diverge), and a **masked** matrix still splits
  the two halves —
  the group sum honours the mask, `jackknife_correction`'s `csr_matrix(X)` strips it. All four are
  pre-existing and documented on `_grouped_sums`.

  "Reduce wide", not "cast to fp64": fp64 is the *narrower* side for integer input above `2^53` and
  for longdouble, so the guard widens only a float whose `eps` exceeds float64's. Integer, bool,
  complex, longdouble and big-endian float64 input reduce exactly as before.

  **Both `bulk_lognorm` callers and the deseq2 backend's pseudobulk move together — one policy, no
  per-caller flag.** #264 PR2 left this narrow partly because widening moves `deseq2_de._pseudobulk`
  too; that is now a deliberate ruling (Alex, 2026-08-18) rather than a deferral, on the grounds
  that the alternative is a `widen=` keyword on a helper shared by two families, and a shared
  helper's per-caller precondition is the thing that gets forgotten. For every dtype the guard
  touches, the wide reduction is the more accurate sum of exactly the values stored, so deseq2 can
  only move toward its own exact pseudobulk; and **no ENROLLED official bundle can be built with that
  opt-in backend** (a cached or exported `de_deseq2_*` table CAN be frozen, and its cache key now
  carries the semantics term so it is recomputed rather than served). `tests/test_deseq2_backend.py` pins the choice.

  **Measured, in two regimes.** *Integer* counts held in a float are exact until a per-gene group
  sum crosses the dtype's consecutive-integer limit `2^(nmant+1)` — 2,048 for float16,
  `2^24` for float32, `2^53` for float64 — then lose 1 count (the bulk moves `3.53e-10` at the shipped
  `bulk_target_sum` of 5e4; a 400-cell group at ~1e5 counts/gene is already there). *Fractional* input can round from the first addition (an exactly representable fraction does not),
  far below that boundary — and **the stored baseline arms are fractional** (a baseline is a mean,
  emitted as float32). On all three official contexts' `context_mean` arms as stored (138,400 cells
  × 18,533 genes, 301 groups, 95.3–95.6 % of values fractional) the group sums move by up to
  **0.265 counts** and the bulks by up to **5.7e-06**, at 5.7–7.6× *inside* `2^24`:
  A `0.2029/5.58e-06`, B `0.2646/5.65e-06`, C `0.0684/5.73e-06`.

  ⚠️ **The REAL arm of those same three contexts does not move at all** — 0 % fractional values,
  group sums bit-identical, at the same 2.2–2.9e6 magnitudes. So the split is clean: the integer
  real leg is untouched, the fractional baseline leg moves. (That is also why this fix's earlier
  certification passed — it was verified against an integer-count stand-in for the baseline arm.)

  ⚠️ **Every stored baseline, anchor or bundle leg whose group sums actually moved must be
  regenerated** — a FRACTIONAL coarse-float arm at any depth, or an integer-valued floating one whose
  per-gene group sums cross that dtype's exactness boundary (fp32 above `2^24`, float16 above 2,048).
  Treat that as a conservative regenerate set rather than a proof each one moved — an exactly
  representable fraction reduces identically. ⚠️ **And regenerate BOTH SIDES of a comparison**: on
  `--baseline-agg` the *submission's* own `agg_results.csv` + `run_meta.json` can be the pre-fix
  artifact, and rebuilding only the baseline leaves a pair `cli` accepts without a word. ⚠️ **And
  exported DE tables**: `--de-real`/`--de-pred` skip DE computation entirely, so the semantics term
  never runs — it protects tables this build COMPUTES or serves from its own cache, not a parquet
  handed to it. A pre-fix **deseq2** `de_pred.parquet` reused under current code launders pre-#271
  values into a freshly stamped `agg_results.csv`. Omit those flags or regenerate the tables first.
  (The cache and identity terms below invalidate more coarsely than the measurement, by code path
  rather than by measured movement — a key is computed before any value is read — so an integer-count
  artifact below the boundary is recomputed once rather than moved.)

  This exact fix was implemented and **reverted** (`ee0e6c9`) in the chunk-2 wave
  because it invalidates the baseline leg of the three official #276 val bundles; it re-lands now
  that those three are already orphaned by the `rule_digest` move (`992cc849…` vs a runtime
  `1b93878b…`, which `score_metrics` raises on), so the cost is sunk rather than pending. No bundle
  is rebuilt here.

  **Which members move: two of the six scored `vcc2026` ones**, not all six — `pds_cosine` and
  `expr_mse_unbiased_capped_norm` (both legs) read a `bulk_lognorm` bulk, while the four
  `de_wilcoxon_*` members read DE tables computed from cells and are untouched. Their `de_deseq2_*`
  siblings do move, since that backend pseudobulks through the same helper, but it is opt-in and
  cannot form an enrolled official bundle. The unscored `expr_*` diagnostics move with the two.

  ⚠️ **`competition_digest()` does NOT move for this change** — `1b93878b…` before and after —
  because the competition *rule* is unchanged; what moved is the pseudobulk those members read.
  So it went onto the outstanding-debt list at the `rule_version` literal in `competition.py`,
  and 0.14.0's `rule_version` 1 → 2 pays it in the same release (the digest moves there, by
  hand, to `f32f0f9c…`).

  **Inside the library the two eras are separated by a code-semantics term**, the device
  `_PDS_EXCLUSION_SEMANTICS` (#248) and `_ONTARGET_EXCLUSION_SEMANTICS` (#172) were minted for:
  `run._GROUPED_SUM_REDUCTION_SEMANTICS` now enters the `pseudobulk_bulk_lognorm[_moments]` artifact
  key, the deseq2 DE-table key, the anchor identity, `partition.result_semantics` (schema 4 → 5) and
  the result-cache digest — scoped, so a `lognorm` run on a non-deseq2 backend keeps its warm cache.
  Without it a pre-fix run at the SAME version reproduces every key exactly and its cached bulk or
  score is served in preference to recomputing. The in-mem reference bundle's semantic subset
  carries it too, so a pre-fix bundle is refused with "rebuild the bundle" rather than stacked with
  this build's pseudobulks. Together these make a **MIXED** pair fail — a current submission against a
  pre-fix anchor or reference bundle no longer validates.

  ⚠️ **A FULLY pre-fix pair still scores, by design.** Anchor and real-bundle validation is
  *peer-to-peer*: `expect_from_run_meta` reads the submission's own recorded identity and
  `validate_anchor` compares the two artifacts with each other, never with current semantics. So two
  artifacts both built before this change agree and pass. Inside the 0.13.0 window
  `cell_eval2_version` separated nothing — both sides stamped `0.13.0`, so #271 joined #172, #320,
  #321 and #322 as a value-moving change a same-version pair cannot see. ⚠️ **0.14.0 closes that
  window going forward**: the stamp moves, and an ENROLLED competition bundle built at 0.13.0 is
  refused either way, because `score_metrics` compares the bundle's `rule_digest` against the
  runtime rule and raises on a stale one. What survives is the loose and diagnostic paths.
  Three named consequences: loose
  `--baseline-agg` pairing is unguarded even for a mixed pair, since `baseline.config_digest` carries
  no metric-semantics term at all (the documented **#314** gap, deferred by Alex on 2026-08-17, of
  which #271 is the **fourth** live instance and the first whose moving object is the pseudobulk
  rather than a metric's definition); an **all-legacy** partial directory with no semantics sidecar is
  accepted with a warning under #246's compatibility ruling; and a pre-fix `lfc_nmae_ref` reference
  computed with the deseq2 backend is accepted, moving the diagnostic `from_reference` column only
  (`avg_score` is computed before that column is added; #319 owns that identity work).

  **The scale registry mints `low-random_high-1_v9` and retires `_v8`** (Alex, 2026-08-18). The
  registry's rule is "mint when the numbers under a shipped name move even though the table does
  not" — the `_v5` (`bulk_target_sum`) and `_v8` (purity floor) precedents — and `_v8` keys both
  moving members, so a `_v8`-headed column would have spanned two *group-sum* eras. One rung lower
  than the three mints before it: what moved is not a member's policy, nor even its arithmetic, but
  the pseudobulk that arithmetic reads. The table is **byte-identical** to `_v8`'s — proved, not
  asserted: `test_the_v9_table_is_BYTE_IDENTICAL_to_v8s_minus_the_name` compares every field
  `scale_payload` serializes, so the only thing moving `scales_digest()` (`22b3d6b1…` →
  `8542ae14…`) is the rename. ⚠️ **Neither `_v7` nor `_v8` ever shipped in a release** — `v0.13.0`
  ships `_v6`, and both were minted inside this same unreleased cycle — so retiring `_v8` orphans no
  column **defined by a tagged release**. (Tags cannot prove nobody scored from an untagged
  checkout; verified across every tag: `v0.13.0` → `_v6`, `v0.11.0` → `_v5`, `v0.10.0` → `_v4`, and
  no tag contains `_v7` or `_v8`.)

  Three arguments for an exception were drafted and are recorded **refuted** in `scales.py` rather
  than deleted, two of them by tests in this change: the competition profile does *not* reject the
  input that moves (`norm._is_all_integer` compares with `np.allclose` at `rtol=1e-5` and nothing
  checks dtype width — three submissions that pass both numeric input gates and move are pinned); a
  scale is *not* only for baseline-free runs (`--scale` works alongside `--baseline-agg` and
  `--real-bundle`, so a fractional prediction can take the column); and the official arms being
  bit-identical says nothing about submissions.

  The three characterization tests that asserted the defect was present are inverted, and the
  reverted implementation's own tests — including the four inversions its review found (integer
  downcast, longdouble downcast, masked-array mask stripping, needless big-endian copy) — land with
  it. `docs/metrics.md` records the `2^24` threshold as the issue's third acceptance item asked.

### Added
- `cell-eval2 run --write-degenes` — also writes the computed DE tables to the
  output directory as `de_real.parquet` and `de_pred.parquet` (the full
  per-`(target, feature)` tables, before ranking/thresholding). Off by default,
  and a no-op for a profile with no DE metrics (and in real-data-only mode, which
  warns); through the Python API, `write_de=True` requires `EvalConfig(outdir=...)`
  and raises rather than defaulting to a directory in the caller's CWD.
  A side supplied with `--de-real`/`--de-pred` is written as given; a
  computed side is taken from its own DE cache when that side is cached and
  recomputed otherwise (`--cache-pred` alone is enough to cache results, so a run
  without `--cache-real` recomputes the real side). The tables are written on a
  warm results-cache hit too, without re-running the metrics the results cache
  exists to skip.

  **Contributed by [@cachris1](https://github.com/cachris1) (Chris Carpenter) in
  #197 — thank you.** Landed from
  `fix/197-write-degenes`, which carries two review fixes on top of the original: the flag no
  longer discards a warm results cache (it wrote the tables *instead of* serving the hit, so
  every metric was recomputed), and the two tables are both loaded before either is written and
  the loaded frames rebound, so a `--de-real`/`--de-pred` path naming an emitted file is neither
  clobbered nor silently swapped into the metrics.

  ⚠️ Nothing here can move a score: `write_de` is a `compute_metrics` keyword, not an
  `EvalConfig` field, so it enters no `config_hash`, no `result_fingerprint`, and no
  `competition_digest`. With the flag off — the default, and what `--preset vcc2026` scoring
  uses — the code path is byte-for-byte the pre-existing one.

### Removed
- **BREAKING (packaging): the `deseq2` optional-dependency extra is gone.**
  `pip install -e '.[deseq2]'` no longer installs the engine: pip warns that the extra does not
  exist and installs the base package without it. The DESeq2 DE backend itself is untouched —
  every `de_deseq2_*` metric, `de.backend="deseq2"` and the whole opt-in path behave exactly as
  before — but the engine it needs, `deseq2_gpu`, is an Arc-internal package that is not
  published, so the extra could never resolve for anyone outside Arc. Install the engine
  directly alongside cell_eval2 instead; the backend's tests `importorskip` it and skip cleanly
  where it is absent. `scale` (shardad) and `gpu` (cupy + nvcomp) are unaffected, and
  `[tool.hatch.metadata] allow-direct-references = true` stays for `scale`'s git URL.

## [0.13.0] — 2026-08-13

**BREAKING: `score` compares `cell_eval2_version` exactly, so every baseline, anchor, and
real-bundle artifact must be regenerated at 0.13.0.** An artifact stamped by 0.12.0 cannot be
paired with a 0.13.0 submission.

### Added
- **The real bundle is the competition artifact.** `cell-eval2 prep-real-bundle` builds the
  baseline and replicate-anchor scale ends from one config and binds them with `manifest.json`;
  `score --real-bundle` checks the submission against that identity. The manifest stamps one of
  three rule states: the current `rule_digest` is a competition bundle and enrolls
  `from_replicate`; `rule_digest: null` is diagnostic and carries every reason in
  `rule_mismatches`; a present but stale digest is refused.
- The packaged `configs/vcc2026.yaml` preset and `--preset` on every subcommand that accepts
  config flags. It declares every `EvalConfig` field in full and differs from `v2.yaml` in
  exactly two: `metrics: vcc2026`, and `cache_strict: true` — the latter load-bearing, because
  the anchor gate is the strict CONTENT hash and a run without it cannot be scored against a
  bundle at all. `tests/test_vcc2026_preset.py` fails if a third field ever differs.
- `cell_eval2.competition`: the frozen competition rule, its derived JSON payload, and its
  SHA-256 `competition_digest()`, pinned by `tests/test_competition_rule.py`.
- `cli._build_parser()`, so parser construction can be exercised without running the CLI.

### Changed
- **BREAKING: `from_replicate` is now policy-applied and policy-frozen.** It uses each metric's
  catalog `Scoring` policy with the measured replicate as its anchor, including clamps, the
  Box–Cox tail, and degeneracy handling; score-time policy overrides still move
  `from_baseline` but do not move this frozen scale. With a competition bundle,
  `from_replicate` replaces `from_baseline` at `avg_score`, and the old cell is null. A plain
  anchor or diagnostic bundle remains unenrolled.
- **BREAKING: `DEFAULT_PENALTY_CAP` is 6, down from 10.** Every below-baseline score for an
  unbounded error metric moves, and the Box–Cox tail saturates at $\sqrt{13}\approx3.606$
  instead of $\sqrt{21}\approx4.58$. Two paths are provably unaffected:
  `compat.score_agg_metrics` is penalty-free, so the 2025 bit-exact reproduction cannot see the
  cap; and every shipped scale entry uses `penalty="none"`, while `scales.py` refuses a Box–Cox
  entry with an unpinned cap, so `tests/test_scales.py`'s frozen digest is untouched.
- `expr_mse_unbiased_capped_norm` gains `clamp_low=0.0`, making it $[0,1]$ on both scales. It is
  the only `vcc2026` member that clamps at the top; the other five may exceed 1 on the replicate
  scale.
- `RESERVED_COLUMNS` now includes `from_replicate`, `anchor_source`, `anchor_digest`,
  `real_bundle_id`, and `real_bundle_digest`.
- **`pert_col` is excluded from the competition rule hash** (Alex, 2026-08-13), joining `device`,
  `pert_chunk` and `de.backend` in `competition._RULE_EXCLUDED`. Found by the real-data
  checkpoint: the deliverable panel labels its perturbation column `perturbation` while the
  preset declares `target`, and that single field was the ONLY difference between the
  competition preset and the production run — so every official bundle came out diagnostic with
  the profile, seeds, `bulk_target_sum`, `control_source`, estimators and members all matching.
  `pert_col` names *where* the labels live, not what they mean; renaming an obs column cannot
  move a number. `control` and `target_gene_map` deliberately stay **inside** the rule, and the
  submission↔bundle gate is unaffected because `baseline.config_digest` still carries `pert_col`
  and `check_submission` still compares it.
- `competition_payload()` now records **four things the frozen digest could not previously see**,
  each a way the rule could widen with the hash unmoved:
  - `rule_excluded` — the exclusion set, `de.backend` included. `config_hash` normalizes each
    excluded field to the preset's own value, so hashing the preset is a no-op for every
    exclusion; measured, adding `pert_col` left the digest bit-identical.
  - `require_cache_strict` — `cache_strict` is checked outside the hash, so a hardcoded check
    was invisible to the digest. It is now a module constant that both the payload and
    `is_competition_rule` read.
  - `profile_resolved` — all ten resolved `vcc2026` names. `members` is the six *scored* ones,
    so a diagnostic joining or leaving the profile changed what `is_competition_rule` demands
    while leaving `members` untouched.
  - `members[*].derived_components` — the resolved `normalization` and `worst_value` of each
    derived member's numerator and denominator. `expr_mse_unbiased_capped_norm` is computed from
    two `scored=False` entries the payload never otherwise reaches; the first changes the space
    both sums are taken in and the second fills a missing perturbation before they are summed.
    Their `agg` is deliberately absent — `run._derived_value` sums the components'
    per-perturbation values directly and never reads it.

  `is_competition_rule` now *reads* `profile_resolved` out of the payload rather than
  recomputing it, and its `cache_strict` check reads the same `REQUIRE_CACHE_STRICT` constant
  the payload serializes — so in both cases the frozen value is load-bearing rather than
  descriptive. Likewise `de.backend`'s exclusion is driven from a `_RULE_EXCLUDED_NESTED` tuple the
  payload serializes — a hand-written `"de.backend"` string would have left that one exclusion
  in the very blind spot this field closes.

### Fixed
- `anchor.control_source_effective` is now **derived** from `anchor._inner_config` instead of
  stamped as a literal. It joined `_REQUIRED_META`, so both supplied and cached anchor doors
  refuse an artifact that omits it, and `validate_anchor` refuses any value except `"pred"`.
  A shared control correlates the two halves whose agreement the anchor measures and was
  measured 0.5–2.3% optimistic on `lfc_nmae`.
- `score._reference_column` is the shared core behind every scale column and
  `from_replicate`. It also closes a fail-open case: a reference that names a metric with no
  output row now fails instead of silently averaging over fewer members.

### Refused, deliberately
- **`prep-real-bundle` does not accept a supplied real-side DE table** (`--de-real` is not a flag
  on that subcommand, and `build_real_bundle(de_real=...)` raises). The baseline leg would score
  against the supplied table while `compute_replicate_anchor` recomputes its own for the
  `full_gate_raw` estimator, so `de_wilcoxon_lfc_nmae`'s 0 end and 1 end would be gated and
  normalized by different tables — and nothing could report it: the anchor's semantic identity
  does not cover a supplied table, and the manifest's `de_real_fingerprint` records only the
  baseline leg's. Threading it into the anchor means threading it into the anchor's cache key and
  semantic identity too; until that exists, refusing is the only honest answer.

### Known behaviour
⚠️ A real bundle built under a **wider profile** aborts if any decisive metric in that profile
has no usable replicate scale because its baseline sits on the wrong side of the measured
replicate. `score` would refuse that bundle anyway, so the build fails fast rather than the
campaign failing late.

### Migration
- Regenerate every baseline, anchor, and bundle at 0.13.0.
- Build competition bundles with `cell-eval2 prep-real-bundle --preset vcc2026`, and score with
  `cell-eval2 score --real-bundle`. Read the competition number from `from_replicate` at
  `avg_score`, not `from_baseline`.
- In every editable development environment, rerun
  `uv pip install -e ".[gpu]" --group dev`. An editable install records the version at install
  time and otherwise continues stamping 0.12.0 into newly produced artifacts.

### Regeneration note
Any current report produced by the default-resolving tools changes below baseline and should be
regenerated: `tools/metricval/cap_rules.py:129`, `tools/metricval/scores.py:237`, and
`tools/baselineval/rescore_234_comparison.py:42-49`. The frozen historical study
`tools/vccval/generalist_rescore/rescore_generalist.py` passes an explicit
`penalty_cap=10.0`, so its numbers stand.

## [0.12.0] — 2026-08-12

**BREAKING: `pds_*` values move on tied distances, the shipped scale is renamed, and every
baseline artifact must be regenerated.** `cell_eval2_version` is compared exactly by `score`,
so a 0.11.0 baseline does not pair with a 0.12.0 run.

⚠️ **This version also carries the #276 replicate anchor (#284), which landed on `main`
without a changelog entry or a version bump** — `main` sat at `0.11.0` after it, so the bump
below is the first one to cover it. That work is substantial and breaking in its own right
(a new `anchor.py`, new `score`/CLI surface, `cache` and `catalog` changes); it is **not**
described here because this entry's author did not write it. **Whoever releases 0.12.0 should
get an entry from #284's author before tagging**, or the anchor ships undocumented.

### Fixed
- **`pds_cosine` scored a zero-effect target by its ALPHABETICAL POSITION** (#282). Three
  facts composed into a live scoring defect on a `vcc2026` member:
  1. under cosine a **zero-norm predicted effect** is at distance exactly `1.0` from *every*
     real effect, so the whole row ties (measured: one distinct float64 value across the row);
  2. the rank came from `np.argsort`, which returns the **identity permutation** for an
     all-equal row — deterministic, not unstable, for n = 5…1000;
  3. `prep.pseudobulk` returns **sorted** labels, so that position is the target's
     **alphabetical index**, identical for every submission and predictable from the target
     list alone.

  Net: `PDS_p = 1 - index/(P-1)` — up to `1.0` for an early-alphabet target and `0.0` for a
  late one, instead of the `0.5` a no-information prediction earns. A *fully* constant
  submission self-corrected (the ranks are a bijection, so the panel mean is `0.5`); a
  *partially* constant one did not. Measured end-to-end on a 26-target synthetic panel:
  pasting the reference control for the first six targets alphabetically read **0.6123**, for
  the last six **0.3477** — a **0.265 spread from target names alone**, worth **0.088 on
  `avg_score`**. The reachable trigger is bit-exact equality (pasting the control cells), not
  an approximate no-change prediction, which carries sampling noise and does not tie.

  Fixed by a `discrimination.tie_policy` knob, **`"midrank"`** in v2: every member of a tied
  block takes the average of the positions the block spans, so an all-tied row scores exactly
  `0.5` per target. **Bit-identical wherever distances are distinct** (a tied block of size 1
  has mid-rank == position), and O(n) rather than the O(n log n) sort it replaces. **The `v1`
  and `cell-eval-0.7.6` presets keep `"position"`** for upstream `cell-eval` parity, in both
  `_VERSION_CONVENTIONS` and the preset YAMLs.

  ⚠️ Read that as *the preset*, not *the version*. `EvalConfig(version="v1")` constructed
  directly still carries the dataclass defaults — `tie_policy="midrank"`, and equally
  `rank_denominator="n-1"` and `control_source="real"`. That is the pre-existing "`version` is
  naming/provenance, conventions come from the preset" architecture; this knob behaves exactly
  like `rank_denominator` and adds no new asymmetry. Use `EvalConfig.from_preset("v1")`.
- **The GPU kernel's tie justification was false.** `gpu/distances.py` claimed "pseudobulk
  effect values are continuous floats, so exact ties don't occur and the GPU/CPU tie-ordering
  cannot diverge". Exact ties do occur, and cupy's sort is not numpy's introsort, so the old
  `argsort(argsort(...))` could have returned different per-target scores per device for one
  input. Neither policy resolves a tie by a device-specific sort now, by **two different
  mechanisms**: `"midrank"` is arithmetic and stays on device, while `"position"` —
  inherently a sort ordering — is computed **on the host with numpy** even under cupy, since
  reproducing upstream `cell-eval` (numpy) is the only reason that branch exists. It costs a
  `[B, n_real]` transfer on the legacy path.

  ⚠️ **Scope: this buys identical tie *ordering* for a bit-identical distance block, not a
  device-independent result, and nothing could.** The block is still computed with `xp`, and
  fp32-GPU vs fp64-CPU pseudobulk already differs upstream — which is why
  `_result_config_digest` puts the **device** in the result-cache key. Rounding can still
  decide whether two mathematically equal distances compare equal. The #282 case itself is
  exact on both devices: a zero-norm operand is *assigned* `1.0` by construction, not
  computed. ⚠️ Whether the old code actually diverged on a real GPU was never measured; this
  is a property of the algorithms, not an observation.
- **`docs/metrics.md` §3 documented a free-win exploit** (#281). Its rank was a bare
  strictly-closer count, which on an all-tied row gives rank 0 ⇒ `PDS = 1.0`. The formula now
  carries the mid-rank term, so spec and code agree on ties as well as on distinct distances.

### Changed
- **`low-random_high-1_v6` ships; `_v5` is retired.** The table is byte-identical — the name
  changes because what `pds_cosine` *means* changed, which is the case the immutability rule
  names explicitly and the first time it has fired. `scales_digest()` would not have caught
  it, the same blind spot `_v5` was minted for.
- `pds_cosine`'s scale base of `0.5` is now exact **per target** rather than only on a panel
  average over a fully-degenerate submission. Same number, sounder derivation.

### Added
- `tests/test_preset_yaml_matches_conventions.py` — the preset YAMLs and
  `_VERSION_CONVENTIONS` were two sources of truth with nothing tying them together, and
  `from_dict` backfills anything a YAML omits from the **dataclass** defaults (the v2 values),
  so a missing v1 override silently gave v1 the v2 behaviour. Not hypothetical: this fix hit
  it. Mutation-checked — stripping the knob back out fails 3 tests.

### Note
`tests/test_discrimination.py::test_exclude_target_gene_drops_named_column` moves `1.0` →
`0.75` for target A. That fixture is **genuinely tied** (pred A's effect is L1 = 8 from both
real effects), so the old `1.0` was the free win the legacy rule handed to whichever target
sorted first. The pre-#282 number is now pinned under `tie_policy="position"` in the same test.
`test_cosine_zero_norm_pred_effect_is_finite` built exactly the zero-effect case and asserted
only `isfinite`, so it passed on the defect for its whole life; the value is pinned now.

## [0.11.0] — 2026-08-12

**BREAKING: regenerate every baseline artifact again.** `bulk_target_sum` is part of
`config_hash`, so every cache key and every baseline pairing moves, and the shipped scale is
renamed. Artifacts measured against `0.10.0` do not carry over.

### Changed
- **`bulk_target_sum` 1e6 → 5e4** (#268). 1e6 was the only value on the 2e3..1e6 sweep where
  the expression comparator breaks. Measured on the characterisation panel:

  | `TS` | jackknife bias | split-half ceiling | `control_mean` (must be 1.0) | denominator surviving |
  |---|---:|---:|---:|---:|
  | **50,000** (shipped) | **0.32%** | **+0.0401** | **1.0249** | **47.3%** |
  | 1,000,000 (retired) | 2.06% | −0.0900 | 1.0727 | 25.1% |

  At 1e6 the split-half ceiling is negative on 6 of 6 lines — a replicate scoring *better than
  perfect* — and the "predict the control" anchor, which `scales.py` ships as a hard
  `base=1.0`, sits 7.3% off. At 5e4 the jackknife bias is ~0.07% net of a +0.25%
  complementary-subset artifact, meeting the ~0.1% the estimator needs. 1e6 also lost on the
  axis it was originally chosen for: effective gene count peaks at 3e5 and is lower at 1e6
  (8,485) than at 1e5 (8,856).
- **`low-random_high-1_v5` ships; `_v4` is retired.** The table is byte-identical — the name
  changes because the values scored under it do. ⚠️ A scale name is a **label, not a
  certification**: `score` applies a requested scale without checking run identity, so an
  older 1e6 aggregate still gets the `v5` heading. Fail-closed provenance is #276's work.
- `configs/v2.yaml` now declares `bulk_target_sum` explicitly. The preset YAMLs backfill
  omitted fields from the dataclass defaults, so reading `v2.yaml` did not previously show the
  knob that sets the expression comparator's scale.
- One shared `moments.DEFAULT_BULK_TARGET_SUM`. `EvalConfig` and all six low-level bulk entry
  points (`streaming_bulk` ×4, `cell_source`, `gpu.bulk`) read it, so a future move cannot
  leave some signatures on the old value.

### Note
The tile/dispersed inversion accepted in 0.10.0 is **reduced but not removed**: on the frozen
fixture, dispersed/tile falls from 11.9× to **7.4×**. Tile remains the preferred arm.

## [0.10.0] — 2026-08-11

**BREAKING, and the headline: every anndata metric moves to a new comparator space, so EVERY
BASELINE ARTIFACT MUST BE REGENERATED.** `score` compares `cell_eval2_version` exactly, which
invalidates measured baselines on its own; the comparator move invalidates them again on the
numbers. Read the #264 entry below before regenerating anything.

### Added
- `expr_mse_unbiased` — the pre-#247 metric restored bit-for-bit, unscored diagnostic.
- `expr_mse_unbiased_capped` — the same with #247's cap; the numerator of the scored metric.
- `expr_distance_unbiased` — the sampling-corrected real-perturbation-to-real-control
  distance (unbiased where the correction is: the analytic `lognorm` branch; only
  bias-corrected under `bulk_lognorm`, see #264 below). Read only from the reference, so it
  is identical for every submission on a panel, and it is routinely negative where a
  perturbation's mean shift is unresolvable at that depth.
- **Scales** (#255): a registry of named, frozen reference points that score each metric
  against constant endpoints instead of a measured baseline. `score --scale NAME` adds one
  column per scale and leaves `from_baseline` untouched; `--baseline-agg` is now optional
  when `--scale` is given. Released scale: `low-random_high-1_v4`, where 0 is the random
  minimum and 1 is real input for the six scored `vcc2026` metrics. A pasted real matrix
  scores 1.0 on all six; the random point scores 0.0 on all six. `_v1`, `_v2` and `_v3` were
  each retired within this same unreleased cycle and are not released registry entries: `_v1`
  when #257 removed the metric it keyed, then `_v2` and `_v3` in turn as #264 moved
  `pds_cosine` and then `expr_mse_unbiased_capped_norm` to the new comparator. `_v4` carries
  the same table as `_v2` — a shipped name is never redefined, and a change in what a keyed
  metric MEANS mints a new version just as a changed field does.

### Changed
- **BREAKING** (#264): v2 counts/counts runs now evaluate **all 13 anndata metrics** — every
  `expr_*`, `pds_*` and `delta_*` — in the group-sum `bulk_lognorm` comparator
  `log1p(bulk_target_sum · P_g / Σ_g P_g)`; v1 and runs with an already-log-normalized side
  use the legacy per-cell `lognorm` fallback. None is catalogued on `lognorm` any more. The
  effective comparator is a run-level decision, stamped, and pairing artifacts from different
  comparator spaces is rejected. **Every baseline artifact must be regenerated.**
  - The sampling correction follows the comparator: the analytic `tr Σ̂/n` under `lognorm`,
    and a **delete-1 jackknife** under `bulk_lognorm`, because the group-sum bulk's sampling
    variance is not a single-pass sufficient statistic. It is two-pass and `O(n·G)` dense
    (~222 ms/group at 500 cells × 18,533 genes); the stored artifact stays `O(P)`.
  - ⚠️ **The tiled/dispersed preference inverts.** Under `lognorm` a tiled baseline arm scored
    40.6× a dispersed one on `expr_mse`; under `bulk_lognorm` the tiled arm wins by ~11.9×.
    That is the fix — the old comparator was a dispersion functional no zero-dispersion
    prediction could match (#258, #260) — and its cost is that `expr_mse` no longer penalizes
    a degenerate zero-dispersion submission.
  - ⚠️ The jackknife is **upward-biased**, by an amount `bulk_target_sum` controls: measured
    0.19% at 2e3 rising to **2.06%** at the shipped default 1e6, where the split-half ceiling
    also goes negative on 6 of 6 real lines and "predict the control" reads 1.073 (#268).
  - New unscored diagnostic `expr_real_mass_ratio`: `Σ_g expm1(bulk) / target` — 1.0 by
    construction under `bulk_lognorm` for any positive-mass group (0 for an all-zero one,
    which the bulk maps to a zero profile by policy), and the concavity deficit (measured
    0.8199) under the fallback.
- **BREAKING** `expr_mse_unbiased_norm` is removed, with no alias. Its declared 1.0 no-skill
  anchor was not where it sat: the numerator was debiased on both sides while the denominator
  was a plug-in, so a control-emitting submission read `1 - noise/D` — measured 0.7643 on VCC
  Test, 0.2386 on `CCL_2`, 0.2754 on `H1_CGS`. It is replaced by
  `expr_mse_unbiased_capped_norm`, whose denominator is debiased and whose aggregate is a
  ratio of sums, putting the no-skill point at 1.0 on every panel whatever the reference's
  depth. This is not invariance to submission depth: emitting thinner than the reference reads
  above 1 because #247's cap refuses a correction the reference does not earn (#257). ⚠️ That
  1.0 is exact for the estimator, not for the estimate — it inherits whatever bias the
  correction carries, and under `bulk_lognorm` at the shipped `bulk_target_sum=1e6` the same
  panel that measures the jackknife's 2.06% bias reads 1.073 there (#268).
- **BREAKING** `expr_mse_unbiased_capped_norm` has **no per-perturbation column**. Its value
  exists only in the aggregate frames.
- **BREAKING** `low-random_high-1_v1` is retired and `low-random_high-1_v4` ships in its
  place (`_v1` forced: `build_scale` validates at import that every key names a catalog
  metric, and #257 removed the one it keyed; `_v2`/`_v3` because #264 moved the metrics they
  key into a different space).
- **BREAKING** every baseline must be regenerated — and for several independent reasons, so
  this is not a restatement of any one entry above: #257 changed which metrics exist, #264
  changed the space they are computed in, and `score` compares `cell_eval2_version` exactly, so
  a release bump alone would invalidate them regardless.
- `is_decisive` now requires the metric to be scored. This also demotes the v1-emitted but
  unscored `de_wilcoxon_nsig_counts_real` and `de_wilcoxon_nsig_counts_pred`; the demotion is
  inert because every consumer skips unscored metrics before asking about decisiveness.
- **The six scored `vcc2026` metrics are now decisive** (#255, the `vcc2026` half of #222): a degenerate
  baseline for any of the six now raises instead of warning and dropping the metric from
  `avg_score`. Previously only `pds_cosine` failed loud, so a degenerate baseline could turn
  a six-metric competition average into a five-metric one with nothing recording it. The
  earlier objection — that `expr_mse_unbiased_norm` reaches `base <= 0` legitimately — was
  measured and does not hold for the deployed baseline (169.16 / 197.05 against a threshold
  of 0, 0% of perturbations at or below).

## [0.9.0] — 2026-08-06

**With thanks to Jeremy Sullivan ([@sullivanj91](https://github.com/sullivanj91)), who found,
quantified and fixed the `pds_*` target-gene exclusion bug below (#248, #250) — the most
consequential correctness finding in this release.**

⚠️ **BREAKING for every baseline** carrying `expr_mse_unbiased_norm` — it is scored in `full`,
`anndata` and `vcc2026`. Regenerate. As at 0.8.0, `cell-eval2 score` only aborts on a
mismatched pair when **both** `baseline_meta.json` and `run_meta.json` are present; with either
missing it logs `pairing NOT verified` and scores anyway.

⚠️ **`pds_*` scores change on GUIDE-LEVEL panels** (labels like `ADNP-1` rather than bare gene
symbols) — see *Fixed*, below. Gene-level panels are unchanged. Any `pds_*` number previously
reported on a guide-level panel is too high and should be reissued.

### Fixed

- **`exclude_target_gene` silently no-oped on guide-level labels** (#248, PR #250 — reported,
  measured and fixed by Jeremy Sullivan). `discrimination_score` resolved each perturbation's
  own gene column by looking the **perturbation label** up in the gene index. On guide-level
  panels the labels are construct IDs, the lookup missed, and the exclusion was skipped with **no
  warning and no error** — in both the CPU and the GPU implementation, which are written
  separately, the GPU one being the default on a CUDA box.

  **This was a ranking inversion, not a bias.** Measured on a 200-construct / 200-gene panel:
  roughly half of `pds_cosine`'s above-chance margin was the on-target gene, and a submission
  predicting **only** the knockdown it was told about — no downstream biology whatsoever —
  scored **0.95–0.98**, against **0.72–0.83** for real submissions. The optimal strategy for the
  metric was to predict *less* biology. `pds_cosine` is one of the six `vcc2026` metrics.

  The fix introduces one shared `distances.resolve_exclusion_columns` used by both kernels, with
  resolution order mirroring `de.resolve_target_genes` (`target_gene_map` first, then the raw
  label), so `pds_*` and the eleven chance-corrected DE metrics now agree on what "the target
  gene" means inside a single run. **Zero resolution raises** rather than scoring with nothing
  excluded; partial resolution logs at INFO. `cfg.target_gene_map` is threaded through both
  dispatch branches, so `scale.py` and `partition_inmem.py` inherit it.

  Two consequences worth knowing:
  - **A guide-level panel now needs `EvalConfig.target_gene_map`**, or it raises. That is
    deliberate: the alternative is the silent wrong number this release removes. Pass
    `exclude_target_gene=False` to score without exclusion on purpose. ⚠️ The v1
    `compat.MetricsEvaluator` shim can supply neither — tracked as #252.
  - **Result caches are invalidated for runs that could have been affected** (a `pds_*` metric
    requested with `exclude_target_gene` on). Deliberately conservative: it also invalidates
    gene-level panels that were already correct, because whether a given panel was affected
    cannot be decided from the cache key alone.

  Follow-ups: #252 (compat shim has no escape hatch), #253 (a third copy of the same lookup in
  `baseline._generic_profile`, which warns rather than being silent).

### Renamed

- **`expr_mse_unbiased` → `expr_mse_unbiased_norm`.** The metric no longer returns an MSE — it
  returns a dimensionless ratio (see *Changed*, below). **No alias is registered, and that is
  deliberate:** an alias would let a 0.8.x baseline column bind silently to a metric with a
  different definition, which is exactly the failure the rename exists to prevent. A 0.8.x
  baseline now fails to match by name rather than matching wrongly. Historical CHANGELOG
  entries, plans and specs keep the old name — they describe the old metric.

### Changed

- **`expr_mse_unbiased_norm` caps the prediction's sampling correction at the real side's** (#247).
  The correction is now `min(tr Σ̂_pred/n_pred, k · tr Σ̂_real/n_real)` with **k = 1**
  (`metrics.delta.PRED_TRACE_CAP_K`) — *a submission may never claim a larger sampling
  correction than the reference itself earns.*

  The estimator was unbiased only under honest iid emission, which nothing verifies: reporting
  the same predicted mean through more dispersed cells enlarged the submission's own subtracted
  term and lowered the metric for free. The only bound was `mean(m²)`, a property of the panel's
  expression scale — measured at ~2000× the honest pred-side term on an accurate panel, enough
  to drive a good submission to ≤ 0 and tie `clamp_high=1.0`, i.e. to score identically to a
  perfect predictor on 1/6 of `avg_score_vcc2026`. The new bound,
  `u ≥ expr_mse − 2·tr Σ̂_real/n_real`, does not depend on the submitter at all.

  **Cost — cell-count invariance, in half its range.** A prediction carrying the reference's
  dispersion through *fewer* cells now reads worse than the same prediction emitted at the
  reference's depth, by exactly `SD²(1/n_pred − 1/n_real)`. #198's criterion 4 holds only while
  the prediction's term is at or under the reference's. Accepted deliberately: `n_pred` is
  submitter-chosen and `validate_cell_pair` checks the perturbation *sets* but nothing about
  per-group cell counts.

  **Measured impact — near zero on real submissions.** Over the 330 scored VCC Test submissions
  the median submission's own term is 0.045× the real side's (models emit *under*-dispersed
  cells), the cap fires on 0.0% of the median submission's perturbations and moves its mean
  value by +0.000000; it bites 16 of 330. On genuine replicates the honest ratio spans only
  0.953–1.068 over 2,201 perturbation-arms across two real datasets, so the cap fires on ~half
  the perturbations there by symmetry, at negligible magnitude.

  **Not affected:** the reported value stays signed and unclamped (only the *correction* is
  bounded); `moments.trace_over_n_for` and `moments.unbiased_sq_dist` still return the honest
  uncapped quantities, so anything calling them directly inherits the old behaviour.

- **`expr_mse_unbiased_norm` is normalized by the real effect size.** It now divides by
  `‖μ_real,p − μ_real,ctrl‖²` — the perturbation's own real response, against the **real**
  control always — so it is a **dimensionless ratio** rather than squared expression units.
  Numerator and denominator both carried a `1/G`, which cancels.

  **`1.0` is the no-skill point.** A submission emitting the control unchanged has a numerator
  equal to its denominator and reads ≈1 on every perturbation, so a raw value is interpretable
  without a baseline: below 1 is skill, 0 is perfection, and the value stays signed.
  `anchor=0.0` / `direction="lower"` / `clamp_high=1.0` are unchanged and still correct.

  **Calibration changes completely.** Over the 330 scored VCC Test submissions the range is now
  **0.94 (best) / 783 (median) / 14,771 (worst)** where the pre-normalization range on the same
  axis was 0.59–3.17. Read against 1.0 that is the finding the old scale hid: the best
  submission is only level with predicting the control on the expression axis. Mostly a scale
  effect — the median submission's error is 2.97 in squared expression units against real
  effects of 0.0053 — so it says those submissions are not calibrated to the reference's
  expression scale.

  ⚠️ **Two properties travel together.** `agg="mean"` over a ratio gives weak perturbations the
  most leverage (the denominator spans 151× on VCC Test, 15–19× on the replicate arms), and the
  plug-in denominator *flatters* exactly those, since it carries the reference's own sampling
  noise (`d/(d+floor)`). Correcting the denominator would be worse: it is signed and the weakest
  real perturbations sit **at** that floor (min `d_p`/floor 0.93 on both replicate arms), so it
  would flip the ratio's sign there. That self-flooring is also why there is no epsilon or
  effect-size gate.

  A real perturbation mean exactly equal to the real control mean, or a real control missing
  from `real_bulk`, now **raises** — both are reference defects, and a NaN would silently drop
  the perturbation (`worst_value=None`).

  Measured for orientation, a genuine replicate scored as a submission: **−0.030** median on the
  CCL technical split-half arm but **+0.848** on the CGS *biological* replicate arm.

## [0.8.0] — 2026-08-06

⚠️ **BREAKING for every baseline.** Two independent changes move whole-cohort metric
numbers, and a third (#241, below) changes what the generic-response baseline predicts.
**Regenerate every baseline** built with 0.7.x or earlier.

⚠️ **The mismatch check is not a safety net you can rely on.** `cell-eval2 score` aborts on a
mismatched pair only when BOTH `baseline_meta.json` and `run_meta.json` are present. If either
is missing it logs `pairing NOT verified` and **scores anyway** — so a metadata-less 0.7
aggregate is still scoreable, and its margins will be silently wrong. Regenerate; do not rely
on being stopped.

### Changed

- **One aggregation statistic for the whole catalog** (#231). The 18 remaining
  `agg="median"` entries — nine chance-corrected direction suffixes × the
  `wilcoxon`/`deseq2` families — now aggregate by **mean**, joining the two that moved in
  #229. Two statistics meant a metric's whole-cohort number answered a different question
  depending on which suffix you read, and any profile average built as a plain mean over
  metrics silently mixed them.
  - **Per-perturbation values do NOT change**; only the 18 whole-cohort aggregates do. A
    warm result cache stays valid (it stores the per-perturbation tidy frame), and the
    frozen v1 byte-identity gate cannot move: all 18 entries are v2-native
    (`v1_name is None`), so `compat.score_agg_metrics` never sees them. Both facts are now
    asserted rather than argued —
    `test_scoring_catalog.py::test_the_formerly_median_entries_are_all_v1_unavailable` and
    `test_baseline_build.py::test_the_agg_mapping_does_NOT_reach_the_RESULT_cache_key`.
  - `agg` remains a real per-entry field with **no default** on `add()`, and
    `run.aggregate_metrics` keeps its generic `median` branch (exercised by a monkeypatched
    test). Changing the statistic stays a catalog edit, not a source edit.
- **`vcc2026` scores the RAW direction pair** (#231):
  `de_wilcoxon_direction_fidelity_yield` → `de_wilcoxon_direction_fidelity_yield_raw` and
  `de_wilcoxon_direction_reach` → `de_wilcoxon_direction_reach_raw`. The chance-corrected
  pair stays in `full`/`de`; the profile still has six members.
  - **Why.** The corrected `fidelity_yield`'s no-skill point is neither zero nor stable.
    Measured over six CCL lines × 4 arms on two panels (replicate = split-half arms,
    baseline = control_mean/context_mean, mean aggregation): its baseline runs **−0.6086 to
    −0.8948** and moves 0.05–0.10 between panels, while `direction_fidelity_yield_raw` sits
    at **0.4863–0.5148** across all twelve line × baseline cells — empirically the
    theoretical random point — and moves ≤0.012 between panels. `direction_reach_raw`
    scores replicate 0.9118 against baseline 0.0251.
  - Both raw entries already carried `anchor=1.0`, `clamp_low=0.0`, `penalty="none"`,
    `scored=True`, `worst=None` — identical to the pair they replace — so `vcc2026`'s
    formal `avg_score` range is unchanged and no scoring policy moved.
  - **Known consequence, accepted:** the 2026 score no longer charges a submission for
    predicting each gene's habitual direction. `direction_reach_raw` carries the anti-gaming
    load alone — an abstaining predictor drives bare `fidelity_raw` to 0.9999 but is held to
    0.005–0.05 on `reach_raw`.
- **The baseline config digest records each resolved metric's aggregation statistic**
  (#231). `baseline.config_digest` gains an ordered `metric_agg` term beside `metrics`.
  Resolved metric NAMES cannot see a change of statistic — the 18 entries kept their names
  and their `full`/`de` membership — so without it a 0.7 baseline and a 0.8 run would digest
  identically while their numbers answer different questions. This makes the mismatch a
  loud failure (`cell-eval2 score` aborts unless `--allow-config-mismatch`) instead of a
  silent one, and does not depend on the runtime version stamp, which resolves through
  installed distribution metadata. ⚠️ This fires only when both metadata files are present —
  without them the pairing is reported as NOT VERIFIED and scoring continues.
  - Deliberately **not** mirrored into `run._result_config_digest`: that keys the
    per-perturbation tidy frame, to which `agg` is applied only afterwards, so adding it
    there would buy nothing but a spurious cache miss.
  - ⚠️ This is a deliberate digest **schema bump**: a baseline stamped by 0.7.x no longer
    matches a 0.8 run even for metrics whose own semantics did not move. That is the
    intended behaviour — every 0.7 baseline is invalid under this release anyway.
- **`tools/metricval/strata.py` and `scores.py` re-aggregate by the RUN's own statistic.**
  Both now read `run_replicate/metric_aggregation.csv` (new `strata.agg_map`) instead of the
  installed catalog, so 0.8 tooling analysing a 0.7 run no longer silently restates its
  medians as means. A missing, unreadable, malformed, duplicated, invalid-valued or
  incomplete mapping **fails**; disagreement with the installed catalog **warns** and the
  run wins, since for 0.8 tooling on a 0.7 run disagreement is the expected case.
  `scores.py`'s `DEFAULT_METRICS` is now derived from `resolve_metrics("vcc2026")` rather
  than hand-listed, so it cannot drift from the profile again.

### Fixed

- **The generic-response baseline emits dispersed cells, not one tiled profile** (#234,
  PR #241). `build_generic_baseline` tiled its profile with `np.tile`, making every
  predicted cell identical. Measured on `adata_Test`: 96% of `expr_mse` was construction
  artifact (0.391759 tile vs 0.012568 dispersed), and identical cells forced
  `tr(Σ̂_pred) = 0`, maximally powering the pred-side Wilcoxon (9,701 significant genes per
  perturbation against a real 2,739; 414 dispersed). The default now draws `n_p` control
  cells with replacement per non-control perturbation and scales each gene-wise by
  `r = profile / control_pseudobulk`. `emit="tile"` is kept as an explicitly-selected
  non-default, verified bit-for-bit identical to the pre-fix aggregate. **This is breaking
  on its own**: on the 330 scored submissions, those clearing the comparator go 7 → 1 on
  `expr_mse_unbiased` and 266 → 325 on `sig_jaccard`.
- `tools/metricval/README.md` said the wide agg CSV has six rows; it has seven — `median`
  was appended to `_WIDE_STATISTICS` in #230.

### Added

- `tools/metricval` CPM-gate sweep tooling and its CCL_2 split-half validation report
  (#235): `cpm_deg_counts.py`, `cpm_gene_universe.py`, `cpm_sweep_report.py`,
  `cpm_sweep.sbatch`, and `docs/validation/2026-08-04-cpm-cutoff-sweep-ccl2.md`.

### Upgrading

- Rebuild every baseline. `cell-eval2 baseline` output from 0.7.x or earlier is rejected by
  `score` **when its `baseline_meta.json` and the run's `run_meta.json` are both present**;
  without them the pair is reported as unverified and scored anyway (see BREAKING, above).
- ⚠️ **Discard and rebuild `vcc2026` partial directories** rather than mixing 0.7 and 0.8
  partials. Partition/streaming partial sidecars record reference and config hashes but no
  metric MEMBERSHIP, and `partition.aggregate_partials` applies the CURRENT catalog when it
  reduces them — so 0.7 partials carrying `direction_fidelity_yield`/`direction_reach` rows
  and 0.8 partials carrying the `_raw` rows will concatenate into a plausible aggregate over
  an incomplete cohort, with no error. `full` and `de` partials are unaffected (both pairs
  are members before and after), and change 1 does not affect *complete* partials at all,
  since it moves no per-perturbation value. Giving the sidecars a result-semantics digest is
  tracked in #246.

## [0.7.0] — 2026-08-05

### Added
- `vcc2026` profile — the six metrics scored for the **2026** Virtual Cell Challenge:
  `pds_cosine`, `expr_mse_unbiased`, `de_wilcoxon_lfc_nmae`,
  `de_wilcoxon_direction_fidelity_yield`, `de_wilcoxon_direction_reach`, and
  `de_wilcoxon_sig_jaccard`. It **replaces** `vcc` rather than extending it — `pds_cosine` for
  `pds_l1`, `expr_mse_unbiased` for `expr_mae`, `sig_jaccard` for `overlap` — and `vcc` is
  untouched, so the 2025 competition score is unchanged. This is the set
  `tools/metricval` characterized on real data. Every member already existed, so the profile is
  purely a `MetricSpec.profiles` tag (`PROFILES` is derived from those tags); the `de_deseq2_*`
  siblings stay untagged and are reached only by the backend relabel in
  `run.dispatch_de_metrics`. `expr_mse_unbiased` carries `needs_moments=True`, so a `vcc2026`
  run pulls the per-group moments artifact. Five properties of the set, all in §6 of
  [`docs/metrics.md`](docs/metrics.md):
  - **All six aggregate by MEAN.** `direction_fidelity_yield` and `direction_reach` were
    moved off the direction family's median for this reason (see Changed, below): `avg_score`
    is itself a plain mean over metrics, so a six-metric competition average should not
    average across two statistics. `full`/`de` still mix (9 median members each).
  - **The error class holds 84.6% of the achievable range.** `expr_mse_unbiased` and
    `de_wilcoxon_lfc_nmae` each score in `[-10, 1]` against the other four's `[0, 1]`, so
    `avg_score` spans `[-3.33, 1]`.
  - ⚠️ **Five of the six are not `is_decisive`**, so a degenerate baseline for any of them is
    excluded from `avg_score` with a warning rather than aborting — silently changing a
    six-metric average's denominator. Accepted for now (the exclusion fires only on a genuinely
    degenerate baseline); issue #222 tracks making decisiveness profile-aware.
  - **Only `pds_cosine` has a data ceiling.** The other five are absent from
    `ceiling.SB_METRICS`, so a `--ceiling` run reports `NaN` for five of the six.
  - ⚠️ **Under `version="v1"` the profile collapses to `pds_cosine` alone** — a profile-supplied
    name that is not v1-available is filtered silently, so `--profile vcc2026 --version v1`
    scores one metric under the vcc2026 label. Pinned in `tests/test_v1_gate.py`.
- **`tools/metricval`** — an offline runbook that measures two empirical reference points per
  metric on real data, so a raw metric value can be read against both a triviality comparator
  and an achievability reference: `baseline` (the generic-response context-mean predictor — an
  ORACLE, since its profile is averaged from the evaluated data, so it bounds a metric's
  *triviality* rather than being a reachable floor) and `replicate` (one experimental half
  scored as an ordinary submission against the other — an empirical reference point, not a
  proved upper bound). Two arms reuse `dge_robust`'s frozen split-half manifests: `CCL_2`
  `splithalf_v5` partition 1 (a seeded disjoint split of one experiment) and `H1_CGS`
  `repro_across_n500` (two genuine experimental replicates); the pair is descriptive, not
  causal — the arms differ in cell type, screen, cohort and selection as well as in
  split-vs-replicate. Includes materialization, DEG-strength strata, cohort-aware scoring and
  the comparator report. **`tools/` is not packaged and there are no `src/` changes**, so
  nothing here affects library scoring. (#225)
- **`tools/metricval/cap_rules.py`** — measures two *proposed* per-perturbation rules (a
  `[-1, 1]` clamp for `direction_fidelity_yield`, a `k * b0` cap for the two error-class
  members) against the same two arms, applied before aggregation rather than to the
  already-aggregated value as `scoring.score_one`'s clamps are. ⚠️ **Analysis only — no rule
  here is applied by library scoring**; issue #229 tracks whether any of them should be. (#230)

### Changed
- ⚠️ **BREAKING for `full` / `de` baselines.** `de_wilcoxon_direction_fidelity_yield` and
  `de_wilcoxon_direction_reach` now aggregate over perturbations by **mean** instead of by the
  direction family's median (`MetricSpec.agg`). They are the two members of that family the
  `vcc2026` profile scores, and `avg_score` is a plain unweighted mean over metrics — a
  six-metric competition average should not average across two aggregation statistics. `agg`
  is a property of the metric and not of the profile, so this moves them in `full` and `de`
  too; every baseline containing either metric must be regenerated. The `de_deseq2_*` siblings
  move with them — `_register_de_family` passes one `agg` to each pair, and a metric must not
  change its aggregation statistic because the DE backend changed — so four catalog entries
  move in total and the catalog split goes 22/69 median/mean to 18/73. The other nine members
  of the family keep the median in both backends. Measured on both `tools/metricval`
  split-half arms, on the nDEG > 50 cohort:
  `direction_reach` 0.9197 → 0.8952 (CCL_2) and 0.9253 → 0.8984 (H1_CGS),
  `direction_fidelity_yield` 0.8686 → 0.8412 and 0.8513 → 0.8423 — every score moves by under
  0.03 and both datasets agree on the direction of every move. Issue #229 carries the two
  follow-ups: a per-perturbation clamp for the chance-corrected members (`fidelity_yield`'s
  whole-cohort mean is dominated by a −20 floor its median never saw), and the same
  re-assessment for the other nine.

  **Migration.** Rebuild affected baselines with `cell-eval2 baseline` under 0.7.0 against the
  same reference, profile, config, host and DE inputs. Note the practical scope is wider than
  the two metrics: `score` compares `cell_eval2_version` **exactly** among its pairing fields
  (`cli.py`), so a baseline stamped `0.6.0` is rejected outright when paired with a 0.7.0 run —
  for `vcc` or `pds` as much as for `full`/`de`. That guard is unchanged in this release and is
  deliberate; `--allow-config-mismatch` downgrades it to a warning, and a baseline carrying no
  metadata at all is only warned about. So every 0.6.0-stamped baseline needs a rebuild (or a
  deliberate override), while only four catalog entries change their aggregation statistic —
  and separately, `expr_mse_unbiased` values change for thin groups, as described under Fixed.
- `aggregate_metrics_wide` emits a **`median` row** alongside `mean`, so a metric's
  non-scoring statistic is still published rather than discarded — and `score
  --comparison-statistic median` now works for every metric. Appended to `_WIDE_STATISTICS`
  rather than inserted, so the existing row order of every published `agg_results.csv` and
  `baseline_agg.csv` is unchanged. The `mean` row still holds whatever `MetricSpec.agg`
  declares; for a median-aggregated metric the two rows are therefore equal.

### Fixed
- **#219 — `expr_mse_unbiased` no longer drops a perturbation solely because `n < 2`.**
  Where a group has fewer than 2 cells the sampling correction is now **zero** instead of NaN,
  so with finite inputs the perturbation keeps a finite value and stays in the aggregate.
  `moments.trace_sigma` still reports the covariance as undefined at `n < 2` — the change is in
  `moments.trace_over_n_for`, which returns the correction to SUBTRACT and now deliberately
  subtracts zero when there is no estimate to be had. Zero is a fallback, not the estimator's
  answer: the true expected term at `n = 1` is `tr Σ`, so this leaves the summed primitive
  upward-biased by `tr Σ` and the metric, which carries the `1/G`, by `tr Σ/G`. (A NaN arriving
  in the *input* still propagates and is still dropped.)

  It mattered because NaN meant *dropped*: `worst_value=None` skips the no-drop fill and
  `aggregate_metrics` drops NaN before the mean, and `n_pred` is chosen by the submitter
  (predicted group sizes need not match the reference, only the perturbation set does). Measured
  on the old behaviour: thinning one badly-predicted perturbation to a single cell moved the
  aggregate `0.49733 → 0.00012` — ~4000× better for a strictly worse submission — while
  `expr_mse` correctly worsened `0.50060 → 0.75930`.

  Subtracting zero inverts that into a self-calibrating penalty *in expectation*: with one cell
  the estimator keeps the full sampling inflation `tr Σ / n = tr Σ` instead of removing it.
  Measured at `SD=0.7`, `G=2000`: the honest `n=300` **metric value** is 0.9978 and the thinned
  `n=1` value is 1.4863 — a +0.488 expected penalty against a predicted `SD² = 0.490`. (Values,
  not scores — `clamp_high=1.0` makes a score of 1.4863 impossible.) No invented constant and no
  policy call about whether thin groups are legitimate; a lucky single draw can still land close,
  so the penalty holds on average rather than per realization.

  **Scope.** This closes the `n < 2` door only. The estimator stays unbiased for every `n ≥ 2`,
  so nothing bias-based replaces the dropped-perturbation route there — but it does **not** make
  the metric robust to a submission that misreports its dispersion, which is a separate channel —
  bounded under input validation but real, and quantified in §2.3. A weak resampling lottery
  also remains at small `n` (~7% on the best of 400 draws at `n=2`).

  **Values change** for any perturbation with fewer than 2 cells on either side: previously NaN
  and omitted from the aggregate, now — with finite inputs — finite and included. Runs whose
  panels have no thin groups are unaffected — the five leaderboard submissions measured for this
  change all scored 100/100 perturbations either way.

## [0.6.0] — 2026-08-03

### Added
- `de_wilcoxon_sig_jaccard` and `de_deseq2_sig_jaccard`: per-perturbation Jaccard index of the
  real and predicted significant-gene sets, `|R ∩ P| / |R ∪ P|`. Symmetric — it penalizes both
  missed real DEGs and spurious predicted ones — where `de_*_sig_recall` divides by `|R|` alone;
  it is the chance-UNcorrected companion of `de_*_sig_mcc`, over the same 2x2 table on a
  well-formed DE table; `sig_jaccard` de-duplicates `(target, feature)` where the
  `de_sig_agreement` family counts rows, so they differ only if duplicate rows are present. Scored
  (`BOUNDED`: higher is better, anchor 1). The wilcoxon entry is enrolled in `full`/`de` only,
  so the `vcc` competition score is unchanged; the deseq2 sibling has literal `profiles=()` and
  is reached only by the backend relabel in `run.dispatch_de_metrics`. Both are v2-native — no
  v1 alias, and not emitted under `version="v1"`. Significance is read as a SET of unique
  `(target, feature)` pairs, so a duplicated DE row cannot inflate the intersection and push the
  ratio outside `[0, 1]`. An empty union (neither side calls anything significant) returns 1.0
  by the set convention J(∅,∅)=1; that regime is reachable on real data, so see §4.5 of
  [`docs/metrics.md`](docs/metrics.md) for the measured rate and what it means for a model
  that predicts nothing.
- `expr_mse_unbiased`: `expr_mse` with the exact additive sampling term `tr Σ̂/n` subtracted from
  both sides (issue #198). **Scored** — `Scoring(scored=True, direction="lower", anchor=0.0,
  penalty="boxcox", clamp_low=None, clamp_high=1.0)`: the `ERROR` policy the other centroid error
  metrics carry, plus an explicit upper clamp. Enrolled in `full` and `anndata` only, so the `vcc`
  profile and the competition ranking are unchanged. Signed by construction — negative *metric*
  values are correct and are never clipped; the clamp applies to the **score**.

  The upper clamp is not cosmetic. For a signed metric the anchor stops being an upper bound:
  `s = 1 - u/b` exceeds 1 for every `u < 0`, which is reachable by construction. Measured on the
  unclamped policy: `b=0.05, u=-0.5` scores **11.0**; the baseline side is worse, with
  `b=1e-4, u=-1e-3` at **11.0** and `b=1e-6` at **1001.0**. The degenerate-baseline guard does not
  catch it — it fires at `b <= 0`, and a tiny *positive* baseline is non-degenerate and explodes
  anyway. `clamp_high=1.0` restores the catalog-wide invariant "an anchored metric cannot score
  above 1" and costs no real discrimination (at or below 0 a submission is indistinguishable
  from perfect). It bounds the **score**, not the incentive: it caps what a negative `u` is
  worth at 1.0 rather than 1001.0, but reaching `u < 0` still pays in full, and the subtracted
  `tr Σ̂_pred/n_pred` is computed from the cells the submission emits. Unbiasedness assumes those
  are an honest iid sample; cell_eval2 assumes it and does not verify it (§2.3). A degenerate
  baseline stays **non-decisive**: v2-native and outside `vcc`, so it warns and excludes rather
  than aborting — which matters more here, since an estimator centred on 0 can legitimately
  produce a baseline at or below 0.

  ⚠️ **Known limitations that ship with this enrolment** (both documented in §2.3 of
  [`docs/metrics.md`](docs/metrics.md)); a third, #219, is **fixed** — see *Fixed* above:
  - **The pred-side correction is submission-controlled.** `tr Σ̂_pred/n_pred` is computed from
    the cells the submission emits, so the metric is only meaningful when those are an honest
    iid sample of what the model predicts: reporting the same predicted mean through a more
    dispersed set of cells lowers the metric for free, and any negative value clamps to a score
    of 1.0. It is **bounded under input validation** — `validate_input_type` requires
    non-negative values (`EvalConfig.validate_input=False` skips it, and then the bound does not
    hold): non-negativity alone caps the per-gene correction at `m²` for `n ≥ 2` at a fixed
    per-gene mean `m`, and putting all mass on one cell attains that ceiling (though that
    construction must still clear `check_scale_limit`). On the metric's scale (after the `1/G`)
    the cap is `mean(m²)`. Measured at a fixed predicted mean (`G=2000`, gamma panel, `mean(m²) = 0.4593`,
    `expr_mse` identical at 25.0000 in both rows): the correction moves 0.0015 → 0.4593 and `u`
    only 24.9985 → 24.5407 — not the −9975 an unvalidated construction gives. Generally
    `u ≥ expr_mse − mean(m²) − C_real` — a lower bound, not an equality, since attaining it also
    needs the extremizer to clear `check_scale_limit`. A large error is safe only *relative to*
    `mean(m²)`, which tracks the panel's expression scale. Real submissions produce
    `expr_mse_unbiased` **metric values** of 0.59–3.17 (values, not scores); whether any of them
    is inside the reachable set has not been measured. Sound for internal evaluation; do not rely
    on it against an adversarial submitter, and read it next to `expr_mse`.
  - **#220 — the fp32 compromise in `moments.trace_sigma`** (~3e-7 at n=2) was accepted on the
    reasoning that it "cannot move a conclusion drawn from this diagnostic". Scored, `|Δu|/b`
    is unbounded as the baseline shrinks, and the exposure is driver-dependent (the `prep` and
    CPU-streaming paths use fp64 means; the accumulator and GPU paths do not).
- Per-group moments seam: every pseudobulk driver can now accumulate per-group cell counts and
  `Σᵢ‖xᵢ‖²` (one fp64 scalar per group, over nonzeros only) and surface them as `GroupMoments`.
  Opt-in per run; moments span all groups including the control and are never restricted. Cached
  under separate `*_moments_*` keys, so existing caches are untouched.
- The partitioned in-memory scorer (`score_piece`) raises `NotImplementedError` if a
  moments-consuming metric is requested, rather than silently returning the biased value. This
  reaches all three public entry points that run through it — `cellstream.score_cellstream`,
  `h5ad_manifest.score_h5ad_manifest` and `rowstore.score_rowstore` — and since `EvalConfig()`
  and `EvalConfig.v2()` default to `metrics="full"`, a DEFAULT config raises on those drivers.
  Score `full` minus `expr_mse_unbiased` there (the manifest CLI runner already does).
- **Data ceiling** — `compute_ceiling(real, *, config=None, seed=0)` and `cell-eval2 run
  --ceiling [--ceiling-seed N]` estimate a per-metric data ceiling from the real data alone.
  Each perturbation's cells are split into two disjoint `floor(n/2)` halves; the verified
  reliability metrics (`SB_METRICS`) are scored on that self-split (reusing `compute_metrics`
  under the caller's config, except that `control_source` is forced to `"pred"` so each half
  uses its own control cells — under `"real"` both halves' log₂FCs would be computed against
  the *same* control, sharing its sampling noise and inflating the ceiling upward — and
  `outdir`/`cache_real`/`cache_pred` are cleared so the inner run cannot overwrite the main
  run's `run_params.yaml` or prebuilt caches); and each metric's per-context mean is mapped
  from half depth to the combined split depth `2*floor(n/2)` via the Spearman-Brown correction
  `r' = 2r/(1+r)`, applied only for `r > 0`.
  Non-positive reliability and every non-verified metric are reported as `NaN`. Writes
  `ceiling_results.csv` (per-perturbation self-split) + `ceiling_agg.csv` (per-metric ceiling).
  `-ap/--adata-pred` is **optional** when `--ceiling` is passed: since the ceiling reads only
  the real data, `cell-eval2 run -ar real.h5ad --ceiling` computes just the ceiling and writes
  none of the prediction-side artifacts — no `results.csv`, `agg_results.csv` or
  `run_meta.json` (omitting the prediction *without* `--ceiling` is a usage error). In that
  mode `--de-pred`/`--de-real` and `--cache-real`/`--cache-pred` cannot apply — the ceiling
  computes any DE it needs on its own halves and runs with caching disabled — so each is
  warned about rather than silently dropped.
  Documented in §2c of [`docs/tutorial.md`](docs/tutorial.md) (walkthrough) and §6b of
  [`docs/metrics.md`](docs/metrics.md) (formula, the assumptions it rests on, and the exclusion
  table accounting for every uncorrected metric of the `full` profile).
  **Contributed by [@LeonHafner](https://github.com/LeonHafner) (Leon Hafner), from an idea
  proposed by [@beabevi](https://github.com/beabevi) (Beatrice Bevilacqua) — thank you both.**
- `cell-eval2 baseline` — builds the generic-response baseline (one average
  perturbation response, self-target gene omitted) and scores it as an ordinary
  submission, giving every metric the configured DE backend can produce an oracle
  comparator. The `deseq2` backend is rejected (a fractional pseudobulk is not valid
  NB-GLM input). `cell-eval2 score` wires the existing `score_metrics` to the CLI and
  checks that the two sides are actually comparable, and `run` now also writes
  `agg_results.csv` and `run_meta.json` (resolved DE backend, device, reference
  fingerprint and effective input types).
- `aggregate_metrics_wide` — the native `statistic`-indexed aggregate `score_metrics`
  consumes. Previously only the deprecated compat evaluator produced that shape.
- **Eleven chance-corrected direction metrics** (issue #195) — **22 new entries**, since
  `_register_de_family` runs for both `wilcoxon` and `deseq2`. Seven in the *fidelity*
  family (`de_*_direction_fidelity`, `…_fidelity_raw`, `…_coverage`, `…_yield`,
  `…_yield_raw`, `…_fidelity_yield`, `…_fidelity_yield_raw`) and four in the *reach*
  family (`…_reach`, `…_reach_raw`, `…_reach_unbounded`, `…_reach_unbounded_raw`).
  They correct the v0.5.0 direction metrics for the reference's own majority-sign rate
  `q`: a model that predicts the majority sign everywhere earns ≈`q` uncorrected but
  ≈0 after correction. `de_*_direction_fidelity_yield` is
  `min(1, n_pred/N_conf) · fidelity` — the **capped-coverage** form, so padding the call
  set beyond the reference's confident budget cannot buy anything.
  **All eleven are scored** — see the `Scoring` entry under *Changed*; `…_yield` and
  `…_reach_unbounded` are unbounded above by design and are clamped rather than excluded.
  All eleven **aggregate by median** rather than mean, and carry `worst_value=None` so
  the no-droppable-NaN fill never rewrites a genuinely unscoreable target.
  The three v0.5.0 direction metrics are unchanged and still emitted; deprecating them is
  issue #196.
- `EvalConfig.target_gene_map` — an explicit `{target: feature}` override for target-gene
  resolution, authoritative where supplied and deliberately not re-checked against the
  feature index. It enters the result-cache digest only when set, so warm caches survive.
- `metric_aggregation.csv` — written by `cell-eval2 run` beside `agg_results.csv`, mapping
  each metric to the statistic (`mean`/`median`) that its `mean` row actually holds. It is
  a sidecar rather than a row because the wide frame must stay strictly numeric: a string
  row would coerce every metric column to text on a CSV round-trip.
- **`de_wilcoxon_lfc_nmae` / `de_deseq2_lfc_nmae`** (#208) — a per-gene log-fold-change
  *accuracy* member for the `de_*` block:
  `mean|lfc_pred - lfc_real| / mean|lfc_real|` over the real-significant gate, lower is
  better. The rank members say whether the ordering is right; this says whether a two-fold
  change is reported as two-fold. The normalization is load-bearing: **a submission
  predicting no change scores exactly 1.0 on every dataset**, by construction, which anchors
  the scale without any reference to the evaluation data. Enrolled in `full` and `de`
  (scored, `direction="lower"`, `anchor=0`, Box-Cox tail); **not** in `vcc`, whose membership
  would also flip `is_decisive`. The gate, the gate size and the denominator come from the
  real side alone, so omission — an empty gate, fewer than `min_gate_size=10` gated genes, or
  a zero denominator — is identical for every submission and is logged by reason.
  **A non-finite predicted log2FC is FILLED with 0.0 like an absent gene, never masked**
  (deliberately against #208 section 5.2): masking would let a submission shrink its own gate,
  and a model emitting `inf` everywhere would score a perfect 0.0. Filled, it scores exactly
  1.0 — the same as silence.
- **The `de_lfc_nmae` replicate reference and its scaled score** (#208) —
  `cell-eval2 run --lfc-nmae-ref [--lfc-nmae-ref-seed N]` and the public
  `compute_lfc_nmae_reference(real, *, config=None, seed=0, de_real=None)` estimate, from the
  real data alone, how well one disjoint half of the cells reproduces the other's log2FCs:
  `mean|lfc_A - lfc_B| / mean|lfc_real|`, corrected by `sqrt(2)` from half depth to full
  depth. **The gate and the denominator come from the FULL real table** — only the numerator's
  two vectors come from the halves — so it needs three DE tables and is not a shape
  `compute_ceiling` can express, and it is a separate module rather than an extension of
  `ceiling.py` (Spearman-Brown is a *reliability* correction for bounded metrics; an error
  metric needs the `sqrt(2)`). Both the raw and the corrected value are emitted so the
  correction stays inspectable. Requires **every perturbation to have at least 2 cells** and
  raises otherwise. Writes `lfc_nmae_ref.csv` + `lfc_nmae_ref_agg.csv`; costs three extra DE
  passes, two when `--de-real` is supplied (which the reference does consume, unlike
  `--ceiling`). Feed the `_agg` file to `cell-eval2 score --lfc-nmae-ref` (or
  `score_metrics(..., lfc_nmae_ref=...)`) for a `from_reference` column:
  `(1 - mean nmae) / (1 - mean nmae_ref)` — 0 is predicting no change, 1 is as good as
  re-running the experiment, and values above 1 are reported rather than clipped. It is
  computed on the two *means*, populated only on the `de_*_lfc_nmae` rows, and
  **deliberately not enrolled in `avg_score`**, so supplying a reference cannot change any
  existing score. A degenerate reference (`mean nmae_ref >= 1`) reports the unrescaled
  `1 - nmae` with a warning rather than dividing by a non-positive denominator and inverting
  the ranking; an empty one leaves the column null and keeps scoring everything else.
  Documented in section 4.3 and the new section 6c of
  [`docs/metrics.md`](docs/metrics.md).

### BREAKING
- **`full` and `de` gain a scored metric, so their `avg_score` moves, and existing `full`/`de`
  baseline artifacts must be regenerated** (#208). `score` compares the user and baseline
  aggregates column-for-column, so scoring a post-#208 run against a pre-#208 baseline raises
  `user/base columns do not match`. Rebuild the baseline with `cell-eval2 baseline` under the
  same profile. **`vcc` and every v1 output are unchanged** — the member is not in the `vcc`
  profile and is `v1_available=False` (no cell-eval equivalent), so neither the competition
  score nor `compat.score_agg_metrics` is touched.

### Changed
- **Purity-curve depth counts adjudicable pairs, not ranked rows** (#204). `_purity_curve`
  computed depth as `pl.col("in_denom").cum_count()`. polars' `cum_count()` counts *non-null*
  entries rather than `True` ones, and `_defined()` ends in `.fill_null(False)`, so `in_denom`
  is never null and that expression was simply the 1-based ranked row position: a pair the
  reference cannot adjudicate (real log₂FC null, NaN or exactly zero) was excluded from the
  purity denominator yet still extended the prefix for free. Such pairs are not scattered —
  a gene the reference never detected still draws a small predicted delta, and where the
  prediction's control cells are all zero every predicted cell equals the same constant, so
  it takes zero within-group variance, maximal predicted significance, and a place at the
  *front* of the ranking. Measured on a 1,029-perturbation panel: 97.9% of the median `k*`
  prefix carried no directional evidence.
  ⚠️ **Three metric values change, by ~50× (median):** `de_*_direction_reach_unbounded` and
  `de_*_direction_reach_unbounded_raw` 12.181 → 0.2424, and
  `de_*_direction_sensitivity_universe` 12.015 → 0.2393. All three remain unbounded above
  (now exceeding 1 for 19.2% of targets rather than ~74%).
  ⚠️ **Amended by #203, which landed after this entry was written.** It said these three are
  `best_value="none"` diagnostics "so no scored profile moves". They are **scored** now —
  enrolment is "does this metric have a direction?" — so while the `vcc` competition ranking
  is still untouched (none of the three is in that profile), the `full`/`de` `avg_score` is
  now **exposed** to the change and can move with them. Not necessarily: an anchorless score
  is `(u−b)/b`, so a proportional shift in both user and baseline leaves it unchanged, and a
  contribution already at a clamp bound stays there. The `[-2, 2]` clamp bounds any resulting
  shift. The validation above measured the raw metrics, not `avg_score`. The `universe="adjudicated"` metrics — including the
  **scored** `de_*_direction_reach` — were **unchanged on all six reference lines measured**:
  not one of their reference-significant genes was unadjudicable (0 of 326,832 / 177,289 /
  419,959 / 339,572 / 205,840 / 244,205). ⚠️ That is empirical, not structural. The
  adjudicated universe filters on reference *significance*, which does not imply
  adjudicability — nothing forbids a significant row whose real log₂FC is exactly zero or
  null (or NaN under `nan_lfc_policy="keep"`), and an externally supplied real table can
  carry anything. Such a row no longer advances the depth and can therefore shorten `k*`, so
  the adjudicated variants **can** move on a dataset of that shape, consistently with the
  corrected semantics.
  Verified end to end: re-scoring three cell lines × two predictors through the full GPU
  pipeline moved exactly **3 of 47** metrics in all six runs — the three above — with the
  other 44 identical to `<1e-12`. Predictor discrimination on the three
  (|pointmass − sampled_control| / mean) rose from 0.01–0.21% to 1.30–6.53%: before the fix
  they were nearly blind to the model, because the padding is a property of the reference.
  The `precision == purity` identity is **re-indexed, not withdrawn**: purity is untouched,
  so precision still equals purity at the end of the `S_pred` prefix, now at
  `k = |S_pred ∩ adjudicable|`. The `de_direction_reach` docstring's claim that row-counting
  depth was deliberate *because* it made the identity exact is withdrawn — no test had ever
  exercised it (all four identity fixtures passed under both depth definitions), and one now
  does. One **pre-existing** exception, which already broke the identity before this change
  and is not introduced by it: a non-float `p_adj_threshold` (integer or `Decimal`) over a
  `Decimal` `p_adj` column compares natively while the ranking key is Float64-normalised, so
  the significant set can stop being a prefix of the ranking (the non-float threshold removes
  the guarantee; a break also needs a cross-boundary Float64 collision and adverse
  tie-breaking) — see `_rank_p`'s docstring; it
  needs its own issue, since fixing it changes the sort contract.
- **`MetricSpec.best_value` is replaced by a per-metric `Scoring` policy** (`scored`,
  `direction`, `anchor`, `penalty`, `penalty_exponent`, `penalty_cap`, `clamp_low`,
  `clamp_high`, `allow_negative_baseline`). The single `best_value` token conflated three
  unrelated facts — the normalization anchor, enrolment in `avg_score`, and "there is no
  constant anchor" — so a metric could not be recorded as higher-is-better without also
  claiming an anchor. `best_value` survives as a **deprecated, derived, read-only property**
  for out-of-tree consumers, but nothing under `src/` reads it and it is **lossy**: an
  anchorless scored metric reports `"one"`, indistinguishable from an anchored one. Read
  `spec.scoring` instead.
- **Every metric with a direction is now scored**; the scored set goes **62 → 82**. Enrolment
  is exactly "does this metric have a direction?", asserted catalog-wide as an equivalence
  rather than a count. The four directionless `de_*_nsig_counts_{real,pred}` entries are the
  only ones that stay diagnostic. *(Amended twice by later work in this same release:
  `de_*_sig_jaccard` and `de_*_lfc_nmae` add four scored family entries, and enrolling
  `expr_mse_unbiased` adds one more, taking the scored set 82 → **87** of 91. The four
  directionless entries remain the only unscored ones, so the catalog-wide relation is an
  equivalence again — but it is asserted as the implication `scored` ⇒ has a direction, since
  `Scoring` still expresses "directional but not enrolled" and `expr_mse_unbiased` occupied
  that state for most of this release.)* None of the newly enrolled metrics is in the `vcc`
  profile, so **the competition ranking is unchanged**; they enter the `full`/`de`/`anndata`
  `avg_score` only.
- **A degenerate baseline no longer aborts the whole scoring run for a non-decisive metric.**
  `score_metrics` still fails loud for every metric that is `v1_available` or in the `vcc`
  profile (`catalog.is_decisive`) — the ones where a wrong number decides a ranking. For a
  **v2-native, `full`/`de`-only scored** metric it warns and excludes that metric from
  `avg_score` (so the aggregate is not comparable with a run where it scored); if that leaves
  nothing scoreable it raises rather than reporting the fallback `avg_score = 0.0`. The
  `cell-eval2 score` precheck uses the same predicate, so it aborts early only on a decisive
  offender. The `baseline` writer is unchanged and still refuses to write ANY degenerate
  aggregate without `--allow-degenerate-baseline`; it now records `decisive` per offender so
  the stamp says how each one will be HANDLED at scoring time. Whether the artifact is
  scoreable additionally requires at least one metric to survive.
  `de_*_direction_yield` is the motivating case: it is signed and centred at zero by
  construction, returns exactly `0.0` when the model calls nothing, and aggregates by median,
  so an exactly-zero baseline is reachable from a legitimate baseline run.
- **The twelve anchorless scored metrics are clamped to `[-2, 2]`.** Without an anchor the
  score is `u/b − 1`, which nothing bounds — at `u/b = 100` the raw score is 99, more than the
  entire achievable range of every other metric combined. For the ten with range `[0, ∞)` the
  floor is inert; on `de_*_direction_yield`, whose baseline is signed and centred near zero,
  both bounds bind.
- **`v1_available` is derived from `v1_name`** instead of hand-flagged: a metric with no
  upstream cell-eval name is v2-native and is never offered under `version="v1"`. This removes
  7 v2-native wilcoxon metrics and the 28 `de_deseq2_*` mirrors from v1 output, which were
  reaching it only because the flag was never passed. ⚠️ **`compat.score_agg_metrics` now
  declines every metric that is not `v1_available`**, where it previously scored some of them.
  It reproduces upstream cell-eval and implements only the anchor-0 and anchor-1
  normalizations, so scoring an anchorless metric there produced a confidently wrong number.
  Behaviour on v1-shaped aggregates is unchanged — verified bit-identical over 1,000 randomized
  frames (28,000/28,000 rows).
- **`version="v1"` with `de.backend="deseq2"` now raises** at config construction. v1
  reproduces upstream cell-eval, which has no deseq2 backend, and the combination silently
  bypassed the version gate: `resolve_metrics` blocked the `de_deseq2_*` names, then
  `_effective_de_spec` relabelled the surviving `de_wilcoxon_*` selections into that same
  family afterwards.
- **The eleven chance-corrected direction metrics exclude each target's own gene**, and
  they **fail loud when no target can be resolved to a measured gene**. Resolution is
  decided once per dataset against the unsliced truth DE table (so the check cannot become
  shard-local on the partitioned or streaming drivers) and matches the target label against
  the feature index exactly — no guide-suffix stripping, which would mis-strip a gene
  legitimately ending in `-2`.
  ⚠️ **This can turn a previously-succeeding `full`/`de` profile run into an error** on a
  dataset whose perturbation labels are not gene symbols present in `var_names` (e.g.
  construct IDs like `GENEX-1` against a feature `GENEX`). That is deliberate — the
  alternative is excluding nothing and returning a plausible wrong number. Supply
  `EvalConfig.target_gene_map` to override. The `vcc` profile is unaffected, and a
  *partial* resolution rate is logged, never raised on.
- `resolve_metrics` takes a keyword-only `version=` (default `"v2"`). Under `"v1"` the
  eleven are excluded: **silently** when they arrive via a profile, and with a `ValueError`
  when a caller names one explicitly — `compat` expands a profile into an explicit list
  before dispatch, so raising on explicit names alone would break every ordinary v1 run.
- `aggregate_metrics` gains an `agg` column recording which statistic each row's `mean`
  column holds. The column stays named `mean` for every metric, so compat, `score.py` and
  every published artifact are unaffected.

## [0.5.0] — 2026-07-29

### Added
- **Direction metrics (#187)** — three v2-native DE metrics scoring the *direction* of a call
  rather than set membership: `de_wilcoxon_direction_precision` (fraction of
  model-significant genes whose sign the reference agrees with),
  `de_wilcoxon_direction_sensitivity` (how deep the prediction's ranking stays pure, over
  the reference-adjudicated set), and `de_wilcoxon_direction_sensitivity_universe` (the
  same over the whole shared universe; **unbounded above**, diagnostic only *— as shipped
  at v0.5.0; #203 scores it with `anchor=None` and a `[-2, 2]` clamp*). All three
  are `full`/`de` only and gain automatic `de_deseq2_*` siblings. Additive schema change
  → minor version bump.
- Undefined directions (log₂FC null, `NaN`, or exactly $0$; $\pm\infty$ *is* a direction)
  follow an explicit asymmetric rule: the reference failing to adjudicate excludes the
  pair, the model failing to commit counts as a miss. This deliberately diverges from
  `dge_robust`'s symmetric exclusion — see `docs/metrics.md`.
- The direction metrics reject duplicate `(target, feature)` rows rather than silently
  double-counting them.
- Known limitation: the direction ranking is **Float64-normalised**, so two `Decimal`
  p-values differing only beyond ~15 significant digits collide and the later key
  components decide their order. Every DE producer here emits float64 p-values, and the
  significance boundary (and therefore the precision/purity identity) is unaffected.
  ⚠️ *Corrected under `[Unreleased]` (#204 review): the last sentence is wrong for a
  non-float `p_adj_threshold`. polars compares a `Decimal` column natively against an
  integer or `Decimal` scalar and only casts to Float64 against a float one, so a non-float
  threshold splits the significance filter from the ranking key, so the significant set can
  stop being a prefix. Left as written above for the historical record.*
- **Validated against the published H1_CGS across-replicate table** (355-target cohort,
  α = 0.05, no effect-size filter, per-target medians). Every published figure reproduces
  at its published precision; worst deviation **0.0044** on the 2-decimal table and
  **0.0005** on the 3-decimal one, against a ±0.01 acceptance criterion. The asymmetric
  vs. symmetric undefined-direction rules were **measured to differ by exactly zero** on
  that arm — 0 disagreeing pairs out of 987,394 — so the pre-registered ~0.15% estimate was an
  overestimate. That is one arm, not a general result: the two conventions agreed *there*.
- `de_wilcoxon_model_direction_match` is **unchanged**. It scores a both-zero or both-NaN
  pair as agreement (a both-null pair compares null and is ignored by the mean);
  `de_wilcoxon_direction_precision` is its corrected-semantics successor, and both remain
  available.

### Changed
- **Audit corrections to the published scale results (#174).** The 2026-07-25/26 sweep found
  overstated RAM and GPU-capacity claims in `docs/scale/RESULTS_gcp_a100_CCL_2.md` and the
  benchmark summary; those comparisons and conclusions are corrected or retracted in place (the
  38.2–39.4 GiB GPU measurement itself stands; the capacity/headroom and chunk-size conclusions
  drawn from it do not), and the consolidated
  triage map for issues #155–#171 now lives at `docs/audits/2026-07-26-sweep-triage.md`.
- **`target_sum=None` now means one target per run, not one per matrix (#155).** BREAKING for
  median-normalization runs. `sc.pp.normalize_total(target_sum=None)` normalizes to the median
  library size of *the matrix it is handed*, and cell_eval2 handed different matrices to
  different calls inside one comparison. Normalizing to `T` makes every group mean `T·f`, so
  splitting one ratio across `T_target` and `T_ref` adds `log2(T_target/T_ref)` to every log2FC
  (+2.0 for a 2000-vs-500 counts/cell pair); partitioned runs had the same shape per batch and
  per piece, which made scores depend on `mem_budget`. `target_sum=None` now resolves ONCE per
  run to the real control pool's nonzero-total median (`norm.resolve_target_sum`), and
  `reference.json` records it as `normalize_target_sum` — which consumers both adopt and
  *verify*, closing a pre-existing hole where a bundle built at one target could be scored at
  another without `aggregate_partials` noticing. **Every existing reference bundle must be
  rebuilt**, including numeric-target and lognorm ones: the key is mandatory for a
  `target_sum=None` consumer and no pre-#155 bundle has it. Both shipped presets are unaffected
  on their own data: v2 pins `target_sum=1e6`, and v1 on genuinely lognorm input ignores
  `target_sum` (v1 over raw counts does resolve, since v1 auto-detects). Numbers change for
  `tools/vccval/configs/v2_median_geom.yaml`; `tools/vccval/configs/v1_5.yaml` in **both** its
  continuous metrics and its DE (it uses `mean_calc: geometric` and `epsilon: 1e-9`, neither
  invariant to the shared target); row-store runs that previously decoded `None` at `1e4`;
  `.shad` streaming DE runs that previously used gpudge's union median; and v1
  declared-`lognorm`-over-counts partitioned runs, which were `expm1`-ing raw counts because
  `partition_inmem` passed the declared type to DE while the pseudobulk used the effective one.
  All must be re-run, not compared across this change. Secondary effects: the resolved value
  enters the cache key (invalidating `target_sum=None` entries), and `_use_gpu_pseudobulk` now
  admits median runs, moving them from the fp64 CPU accumulator to the fp32 GPU one. **Measured
  on an H100** over 198 metrics: partitioned scoring is **bit-identical** to whole-prediction
  scoring on the same device, and the CPU-vs-GPU difference is at most `1.51e-05` relative /
  `5.65e-07` absolute at the metric level (worst case `delta_pearson` ≈ 0.0375, where a
  near-zero value inflates the relative figure) and within `rtol=1e-6` at the pseudobulk level.
- **The median rule is now explicit and format-independent (#155).** scanpy 1.12.3 takes
  `np.median` over all cells in its CSR branch but the nonzero-total median in its dense
  branch, so the same control pool resolved differently depending on its matrix format.
  cell_eval2 adopts the nonzero-total median for both. To be exact about provenance: scanpy's
  public docs say only "median of total counts for observations" and never state whether
  zero-total observations are excluded, so this is cell_eval2's chosen rule matching scanpy's
  *dense implementation*, not a rule scanpy documents.
- **Two per-batch defects removed from the cell-stream scoring path (#153).** The phase-profiling
  campaign refuted this issue's original attribution — the per-perturbation loops in `metrics/de.py`
  are **0.14 %** of the ctx-XL wall — and located the time in two per-batch defects instead.
  (1) `check_scale_limit` → `_row_totals` built a scipy CSR per 100k-row block, and scipy's
  constructor prunes a block view into a **copy** despite `copy=False` whenever
  `size < base.size // 2`, so a validation pass copied the qualifying blocks' `data` and
  `indices` — up to the whole matrix, twice. On ctx-M that single frame held 87.9 % of
  `ref_build:unattributed`; on ctx-XL the phase was 67.7 % of the wall. `_row_totals` and
  `_expm1_row_totals` now use the check-free `_csr_row_block` view PR #73 added for
  `inmem_pseudobulk` — whose own follow-up note named `norm._row_totals` and was never actioned.
  The helper moves to `norm.py` (the leaf module) and is re-exported from `streaming_bulk`.
  (2) `score_piece` re-read `reference.json`, every pseudobulk `.npz`, `real_de.parquet` and the
  whole control pool on **every** batch; the control read alone was 34–60 % of the ctx-M wall. A
  lazy `_RefBundle` now reads each artifact at most once per context, and the three drivers
  (`score_cellstream`, `score_h5ad_manifest`, `score_rowstore`) build one per context.
  Both changes are value-neutral: the row-total reduction is bitwise identical, and `score_piece`
  without a bundle behaves exactly as before.

### Fixed
- **`rowstore.score_rowstore` no longer decodes `target_sum=None` as `1e4` (#155).**
  `float(cfg.target_sum) if cfg.target_sum else 1e4` was a truthiness test, so `None` silently
  became `1e4` for the decode while `cfg.target_sum` stayed `None` for the DE call downstream —
  one config with two meanings inside one entry point.
- **`scale.score_streaming` no longer crashes on counts `target_sum=None` (#155)** — it raised
  `TypeError` from `float(None)` inside the pseudobulk accumulator whenever an anndata metric
  was requested. Its counts DE half already worked (gpudge resolved a union median) and keeps
  working; it now uses the archive's control-pool median instead. A **lognorm** archive with
  `target_sum=None` now raises an actionable `NotImplementedError` for **any** requested metric,
  DE included: `compute_de_streaming` takes no `input_type` and maps `None` to gpudge's
  `"median"` unconditionally, so a DE-only lognorm run was not "unaffected" — it was silently
  library-size-normalizing already-`log1p`'d values. For **median resolution and anndata-only
  scoring**, an archive with no designated reference shard falls back to the configured control
  group — DE still requires a designated reference shard, since the streaming DE path passes
  `reference=None` and lets gpudge read the archive's own. An unknown control label now raises
  that same actionable `ValueError` rather than propagating `read_group`'s bare `KeyError`.
- **Partitioned DE now uses the effective input type (#155).** `partition_inmem` passed
  `cfg.input_type` to `compute_de` while `_side_bulks` used the auto-detected type, so a v1
  config declaring `lognorm` over raw counts had DE `expm1` those counts.
- **`compute_de` fails loud on default-branch external-reference DE with counts +
  `target_sum=None` (#155)** instead of silently returning shifted log2FCs, and
  `precompute_cache` refuses counts `target_sum=None` rather than writing cache entries no
  resolved run can hit (lognorm, where `target_sum` is inert, is still accepted).
  `native_gpu_normalize=True` is deliberately exempt from the DE guard: gpudge resolves one
  union median across reference and targets there, so that path had no within-call shift.

## [0.4.0] — 2026-07-27

### Added
- **`gather_threads`: cell-layout reads now decode in parallel (#149).** shardad ≥ v0.7.1 exposes
  `n_threads` on the cell gather path, defaulting to `1`; cell_eval2 never passed it, so every
  cell-layout read decoded single-threaded. A new `EvalConfig.gather_threads` (default `-1` =
  auto) is threaded into all six gather sites (`cellstream.py` ×3, `cell_source.py` ×3) through
  a row-count-aware resolver: `-1` resolves to the process's CPU-affinity allowance
  (`len(os.sched_getaffinity(0))`, **not** `os.cpu_count()` — 208 reported vs a 12–16 cgroup
  allowance on the reporting nodes) as a cap, then `min(cap, ceil(n_rows / 96))`, because
  measured decode speedup saturates at 8 threads for small per-group reads (and regresses at 12)
  while large batch/reference reads keep scaling to ~10×. Numerics are unchanged (parallel decode
  is byte-identical to serial) and `gather_threads` is excluded from the cache key, exactly as
  `num_threads` is. The `[scale]` extra repins shardad `v0.7.0` → `v0.7.1`, and `open_cell_store`
  now fails loudly on an older install rather than raising a cryptic `TypeError` mid-batch.

### Changed
- **Unprofiled re-measurement of the `score_cellstream` phase shares (#157).** The published
  CPU-lognorm-pseudobulk share was read out of a cProfile run at a different memory budget. Measured
  clean, it is **11.23 %** of wall, not 7 % — an Amdahl ceiling of **1.127×**. The design decision it
  informed (not building a GPU lognorm-pseudobulk accumulator) is unchanged, now recorded as a
  maintainer judgement rather than a number-forced verdict. The same run identifies a larger
  untouched lever: **`_to_linear` at 21.73 %** (ceiling 1.278×), which lognorm input cannot avoid
  because `native_gpu_normalize` takes a counts-only branch. Ships the instrument
  (`tools/cellstream_perf.py`, `tools/cellstream_perf_validate.py`) and its artifacts; three claims
  in `docs/perf/2026-07-21-cellstream-perf.md` are retracted.
- CI's ruff rule set is pinned so a future ruff default expansion cannot break the lint job (#147).
### Fixed
- **BREAKING: `de.backend="auto"` now fails on a GPU host without gpudge (ultrareview
  2026-07-25).** `auto` walked `gpudge → pdex → scanpy`, and since scanpy is a hard dependency it
  always succeeded — so a GPU host without gpudge silently produced scanpy DE numbers, different
  from every published cell_eval2 result. `auto` now raises when a CUDA device is visible but
  gpudge is unusable, and warns (once per process) when it falls back on a host with no GPU,
  including a distinct warning when pdex is missing and it lands on the much slower scanpy.
  gpudge remains **undeclared** in `pyproject.toml` for now: it requires Python ≥3.12 while
  cell_eval2 supports ≥3.11, so a `[gpudge]` extra could not resolve on a supported interpreter.
  A pinned extra was expected to follow once gpudge published a `>=3.11` release and cell_eval2
  validated against it (gpudge_arc#95; at the time, the floor change was on upstream `HEAD` but in
  no tag). ⚠️ **Historical.** Both halves of that sentence have since moved — the installed engine
  is 0.8.0 with `Requires-Python: >=3.11` — so this paragraph is a record of the 0.3.0 decision and
  NOT current install guidance. For what `auto` needs today, read the error it raises.
- **The `de_<method>_rank` cache no longer serves a stale rank matrix.** Its key carried only the
  four rank knobs plus a value-blind table fingerprint (row count + column names +
  target/feature value-counts — no log2FC, no p-values). Unlike every other artifact key, it
  carried nothing about *how* the table it keys was generated: the DE-table key enumerates ~20
  such knobs and the pseudobulk key ten, while the rank key is keyed on the derived table alone.
  Changing any DE-content knob (`de.mean_calc`, `control_source`, `target_sum`, the resolved
  backend, …) therefore recomputed the DE table correctly and then served the previous run's
  ranks, making every rank-derived `de_*` metric silently wrong while `de_wilcoxon_nsig_*` in the
  same frame stayed correct. The key now hashes table content.
  **Cache invalidation:** `de_*_rank` entries written for a *computed* DE table under the default
  `cache_strict=False` are invalidated — that is the bug being fixed. Entries from supplied DE
  tables, and from `cache_strict=True` runs, were already fingerprinted strictly and stay warm.
  **Result-cache entries from runs that computed at least one DE table are also invalidated**
  (precisely: DE metrics requested with at least one side not supplied), because
  the result cache is consulted before the rank cache, so a result already poisoned by the old key
  would otherwise still be served. Runs with no DE metrics (e.g. `metrics=["mae"]`) or with both DE
  tables supplied keep their cache.
- **The streaming entry points now validate the (pred, real) gene axis.** `score_streaming`,
  `score_cellstream` and `score_h5ad_manifest` adopted the real side's `var_names` for both
  sides without ever comparing pred's, so a pred archive with the same gene count in a different
  order was scored gene-position-wise and returned plausible finite numbers. They now raise
  `ValueError("gene names/order differ between pred and real")` — the same error the in-memory
  path and `score_streaming_cell` already raised. Note that `score_rowstore` is unaffected: a
  row-store artifact carries a single gene axis shared by both sides, so a mis-ordered
  `pred_X.dat` remains undetectable on that route.

## [0.3.0] — 2026-07-23

### Added
- **Model-conditioned DE direction match** — `de_wilcoxon_model_direction_match` (v1 alias
  `de_model_direction_match`) reverses the existing direction metric's conditioning: it selects
  model-significant genes and measures whether their predicted and real log₂FC signs agree.
  The metric is included in the `full` and `de` profiles and mirrors to
  `de_deseq2_model_direction_match` under the DESeq2 backend.

## [0.2.1] — 2026-07-22

### Fixed
- **`[scale]` extra now installs a shardad that can run `score_cellstream`.** v0.2.0 pinned
  `shardad @ v0.6.2`, which lacks `CellStore.gather_rows_adata` — the method `score_cellstream`'s
  `CellBatchSource` requires (no fallback) — so a clean `pip install cell_eval2[scale]` raised
  `AttributeError: 'CellStore' object has no attribute 'gather_rows_adata'` on the first batch.
  Repinned to `shardad @ v0.7.0`, which adds `gather_rows_adata`; its cell/packed/read code is
  byte-identical to the tree the #130 GPU parity run validated on, so behavior is unchanged. (#140)
- **Clear error when shardad is too old for cell-layout streaming.** `open_cell_store` now raises
  an `ImportError` naming the missing `gather_rows_adata` (added in shardad 0.7.0) instead of
  letting a cryptic `AttributeError` surface deep in the batch loop, so a bring-your-own older
  shardad fails loudly at open time. (#140)

## [0.2.0] — 2026-07-22

### Added
- **`score_cellstream(pred, real, *, config, mem_budget, outdir=None)`** — out-of-core scorer for
  a pair of cell-layout `shardad.cell` (`.shad`) archives, built on the `partition_inmem` engine
  via a new `CellBatchSource` (a `PertBatchSource` that materializes perturbation-complete
  `AnnData` batches from the archive). Unlike the counts-only `scale.score_streaming_cell`, it
  supports **counts and lognorm** with the full option matrix (v1/v2, `control_source`
  real/pred, any `target_sum`) by reusing the shared in-memory `compute_de` + `_side_bulks`, so
  it directly unblocks `scaled_log1p` cell-stream archives. Returns a `ScoreResult`
  (`per_pert`/`per_context`/`overall`). Single-context (multi-context deferred); GPU-only
  (requires a gpudge-resolvable backend + `fdr_scope="per_pert"`). Structural parity with
  `compute_metrics` validated on GPU — rank/DE metrics bit-exact, continuous within
  `rtol/atol=1e-7` — across counts@1e6, counts@1e4, lognorm@1e4, and the v1 `control_source=pred`
  (`cell-eval-0.7.6`) config. All existing scorers are unchanged. (#130)

## [0.1.1] — 2026-07-21

### Fixed
- **GPU pseudobulk crash on dense contexts** — `GroupedMeanAccumulator.update` pinned a whole
  block's `Xr.data` and `Xr.indices` per host-to-device copy, so any block whose nonzeros
  exceed cupy 14.1.1's 2 GiB single-pinned-allocation ceiling aborted with
  `cudaErrorInvalidValue` — hitting both the streaming path (the whole-group control pool, a
  single 210–290K-cell group → 3.8–5.2 GiB) and the in-mem path (the fixed 100k-row block at
  ~4,800 nnz/cell → ~1.9 GiB). `update` now byte-bounds each pinned transfer to ≤ 1 GiB by
  sub-chunking the block's rows; full-span (common) blocks pass through with no copy, and
  oversized blocks split via the check-free `_csr_row_block` view. The grouped-mean
  accumulation is row-additive, so sub-chunking is bit-identical (CPU proof + GPU parity). One
  fix covers both the `_streaming_pseudobulk_gpu` and `inmem_pseudobulk` callers. (#133)
  Independently diagnosed first by **@nuoliu** in #108 (2026-07-17), which characterized cupy's
  2 GiB single-transfer ceiling to the byte and derived the ~5,368 nnz/cell density threshold at
  which the fixed 100k-row block overflows; this entry's byte-bounded fix supersedes that PR.

## [0.1.0] — 2026-07-20

### Added
- **Cell-layout shardad archive input** — `compute_metrics` and the `run` CLI accept a
  cell-layout shardad archive (`shardad.cell`) as pred/real input, auto-detected by manifest
  (any extension, e.g. `.shad`/`.csa`) and materialized via `shardad.read_h5ad`. The reader
  side of the forward-eval→cell_eval2 "C2" seam; out-of-core streaming is a follow-up. (#117)
- **Cell-layout streaming scorer (`scale.score_streaming_cell`)** — out-of-core,
  memory-bounded scoring for a pair of cell-layout `shardad.cell` archives, mirroring the
  shard-layout `score_streaming`: per-group pseudobulk streamed via the existing
  `streaming_bulk` accumulators (`cell_source.cell_pseudobulk`), and DE streamed through
  gpudge's shared reference-pool core (`gpudge._refpool.refpool_de_core`) over a
  cell-store-backed target source (`de_compute.compute_de_streaming_cell`). DE is
  gpudge-only; v1 median-normalization streaming and the `deseq2` backend are deferred
  (mirrors #125). Numerically equivalent to the Stage-1 materialize path (`compute_metrics`
  on the same archives) — rank/DE metrics bit-exact, continuous metrics ~1e-8 relative
  (floating-point summation-order noise vs the materialize path's dense reduction); full GPU
  DE parity validated on an H100. (#117)
- **Non-1e6 `target_sum` for counts in cell-layout streaming DE** —
  `compute_de_streaming_cell` accepts any finite `target_sum > 0` for counts input (e.g.
  `1e4` for the `cell-eval-0.7.6` preset), not just v2's `1e6`; `refpool_de_core` already
  CPM-normalizes to an arbitrary target, so this is a guard relaxation, not new numerics.
  `target_sum=None` (v1 median) stays deferred. GPU-validated: counts at `target_sum=1e4`
  streams == materialize (bit-exact rank/DE). (#129, #131)

### Changed
- **`score_streaming_cell` is counts-only** — an up-front `input_type == "counts"` guard
  rejects lognorm/`scaled_log1p` input loudly (both its gpudge DE and its anndata pseudobulk
  assume raw counts and have no `expm1`/`_to_linear`), preventing silent mis-scoring once the
  DE `target_sum` gate was relaxed; also closes a pre-existing `lognorm`+`1e6` latent hole.
  lognorm cell-layout streaming is deferred (#129/#130). (#131)

## [0.0.4] — 2026-07-19

### Added
- **`deseq2` DE backend** (opt-in, non-default) — a pseudobulk negative-binomial GLM via the
  private `deseq2_gpu` engine (control replicated by NTC guides; unreplicated perturbations).
  `backend="auto"` never selects it, so every existing preset stays bit-identical. (#120)
- **`de_deseq2_*` metric family** — when `de.backend="deseq2"`, the DE metrics are emitted
  under method-correct `de_deseq2_*` names (the same metric set as `de_wilcoxon_*`, relabeled
  by the backend) so a `deseq2` run's output is self-describing and never mistaken for a
  Wilcoxon rank result. Rank backends keep `de_wilcoxon_*`. `DEParams.method` now records the
  `deseq2` provenance. (#121)
- Runnable **deseq2 tutorial** (`docs/tutorials/deseq2.md`) and a cluster GPU-path validation
  runner (`tools/deseq2_gpu_validate.sbatch`); the `device="cuda"` JAX path is validated to
  match the CPU path within a small fp32-vs-fp64 tolerance on an H100. (#121)

### Changed
- `__version__` is now read from the installed package metadata (`importlib.metadata`)
  so it is single-sourced from `pyproject.toml` and can't drift out of sync. (#123)
