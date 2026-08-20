"""The frozen competition rule: which metrics, under which policy, from which anchor.

Split from `scales.py` on purpose. A scale's reference points are CONSTANTS, so a scale can be
frozen whole. The competition's reference points are MEASURED per dataset, so what the repo can
freeze is the RULE -- the members, their resolved policies, and how the anchor is produced --
while the per-dataset values belong to the real bundle that carries them (#276 part C).

Everything here is DERIVED, never hand-listed: the members come from the catalog profile, the
estimators and seeds from `anchor`, the resolved knobs from `scoring`'s defaults. A digest over
a hand-maintained copy would freeze the copy, not the behaviour.

`is_competition_rule` is what makes the rule load-bearing. It runs ONCE, in
`prep-real-bundle`, against the artifact that came BACK (never the arguments that were asked
for), and its answer is stamped into the manifest as `rule_digest` and PRINTED. It is
deliberately not consulted at score time: a check evaluated per submission can be wrong in the
silent direction (always "not the competition"), which is exactly how #276 part C-1's two dead
features survived their own unit tests.
"""
from __future__ import annotations

import hashlib
import json

from .catalog import CATALOG, PROFILES, resolve_metrics
from .scoring import DEFAULT_PENALTY_CAP, DEFAULT_PENALTY_EXPONENT

# `anchor` is imported INSIDE the functions below. Not for a cycle that exists today --
# `score.py` already imports `anchor` at module scope and `cell_eval2/__init__` imports both
# `run` and `score`, so importing this module already pulls that graph in. The reason is
# narrower: keep `competition` itself dependency-light so a future consumer can read the
# constants without the DE stack, and keep digest evaluation strictly RUNTIME -- never at
# module-import time, where the deferred import would not save it.

#: The catalog profile the competition scores.
COMPETITION_PROFILE = "vcc2026"

#: The ONE pinned base seed; the five split seeds derive from it by `SEED_DERIVATION`.
BASE_SEED = 0

#: k = 5, ruled 2026-08-13 independently of the seed-spread measurement.
N_SPLITS = 5

#: #268: at the retired 1e6 the split-half ceiling is NEGATIVE on 6/6 real lines, so an anchor
#: built at that value is not merely different -- it is unusable.
BULK_TARGET_SUM = 50_000.0

#: The anchor's `replicate` is a mean of five aggregates, so the submission side must be read
#: at the mean too; anything else compares a median against a mean and calls it a score.
COMPARISON_STATISTIC = "mean"

#: The scored side's convention -- what the preset pins and `config_digest` compares.
CONTROL_SOURCE_SCORED = "real"

#: The anchor's convention: per-half controls, forced inside the producer. A shared control
#: correlates the two halves whose agreement is measured (0.5-2.3% optimistic on lfc_nmae).
CONTROL_SOURCE_ANCHOR = "pred"

#: `cache_strict` is checked OUTSIDE the config hash, because `cache.config_hash` skips it. A
#: constant rather than a literal in `is_competition_rule` so the payload can record it and the
#: check can read it: a hardcoded `if not config.cache_strict` would be invisible to the frozen
#: digest, so relaxing the requirement would widen the rule without moving the hash -- the same
#: blind spot the exclusion set had (codex checkpoint-2 P1).
REQUIRE_CACHE_STRICT = True


def competition_members() -> tuple[str, ...]:
    """The scored members, in catalog order. The four `vcc2026` diagnostics are excluded:
    `scored=False` means "not part of an average", and the anchor stamps them anyway."""
    return tuple(m for m in PROFILES[COMPETITION_PROFILE] if CATALOG[m].scoring.scored)


def derived_seeds() -> tuple[int, ...]:
    """The literal split seeds this rule produces."""
    from .anchor import _derive_seeds

    return tuple(int(s) for s in _derive_seeds(BASE_SEED, N_SPLITS))


def competition_comparator() -> str:
    """The expression space the competition's `expr_comparator` metrics resolve to.

    Statically resolvable because the preset DECLARES `input_type` on both sides and leaves
    `autodetect_input_type` off, so `_effective_input_type` never samples a matrix. For the
    counts-based preset this is `bulk_lognorm`.
    """
    from . import norm
    from .config import EvalConfig

    it = EvalConfig.from_preset(COMPETITION_PROFILE).input_type
    return norm.resolve_comparator(version="v2", pred_input_type=it, real_input_type=it)


#: Fields the rule deliberately excludes and `cache.config_hash` does NOT. Normalized to the
#: preset's values before hashing so an explicit `device="cpu"` that resolves identically to the
#: preset's "auto", a different `pert_chunk`, an explicitly-named DE backend, or a dataset whose
#: perturbation labels live in a differently-named obs column does not turn a competition bundle
#: diagnostic.
#:
#: ⚠️ `pert_col` was added 2026-08-13 (Alex) on a MEASUREMENT, not on principle: the deliverable
#: panel labels its perturbation column `perturbation` while the preset declares `target`, and
#: that one field was the ONLY difference between the competition preset and the real production
#: run -- so every official bundle came out diagnostic, with everything else (profile, seeds,
#: bulk_target_sum, control_source, estimators, members) matching. `pert_col` names WHERE the
#: labels live, not what they mean: given the same dataset, renaming the obs column cannot move a
#: number. `control` and `target_gene_map` are deliberately NOT here -- the first selects which
#: cells are the reference and the second re-maps genes, and both move every number.
#:
#: Excluding it here does not weaken the SUBMISSION<->BUNDLE gate, which is the comparison
#: `pert_col` is genuinely signal for: `baseline.config_digest` still carries it, and
#: `real_bundle.check_submission` still compares that digest peer-to-peer, so a submission whose
#: perturbation column differs from its bundle's is refused exactly as before.
_RULE_EXCLUDED = ("device", "pert_chunk", "pert_col")

#: The same idea for NESTED fields, as `(section, field)` pairs. Separate from `_RULE_EXCLUDED`
#: only because `dataclasses.replace` cannot reach into a nested dataclass by keyword -- but
#: DATA rather than an inline `de=replace(...)`, so `competition_payload` records the exclusion
#: that is actually performed. A hand-written `"de.backend"` string in the payload would be
#: documentation: deleting the normalization would leave both the string and the digest
#: unchanged, which is exactly the blind spot the payload field exists to close (codex
#: checkpoint-2 round 2 P1).
_RULE_EXCLUDED_NESTED = (("de", "backend"),)

#: Every excluded field in one dotted-name list -- what `competition_payload` freezes.
def _excluded_names() -> list[str]:
    return [*_RULE_EXCLUDED, *(f"{sec}.{field}" for sec, field in _RULE_EXCLUDED_NESTED)]


def rule_config_hash(config) -> str:
    """`cache.config_hash` over the config with `_RULE_EXCLUDED` + `_RULE_EXCLUDED_NESTED`
    normalized.

    Everything else `cache.config_hash` already handles correctly: it SKIPS `metrics`,
    `cache_strict`, `outdir`, `num_threads` and `gather_threads` (the first two are checked
    separately by `is_competition_rule`, the rest move no number), and RETAINS every field that
    does move one -- `bulk_target_sum`, `control_source`, `input_type`, `version`, and
    `allow_fractional_counts`.
    """
    from dataclasses import replace

    from .cache import config_hash
    from .config import EvalConfig

    p = EvalConfig.from_preset(COMPETITION_PROFILE)
    norm_cfg = replace(config, **{f: getattr(p, f) for f in _RULE_EXCLUDED})
    for section, field in _RULE_EXCLUDED_NESTED:
        norm_cfg = replace(norm_cfg, **{section: replace(
            getattr(norm_cfg, section), **{field: getattr(getattr(p, section), field)})})
    return config_hash(norm_cfg.to_dict())


def competition_payload() -> dict:
    """Every field that changes a competition number, as plain JSON-able data.

    ⚠️ NOT strict JSON since the four bounded members were unfloored: their
    ``effective_clamp_low()`` is ``-inf``, which ``json.dumps`` emits as the non-standard
    ``-Infinity`` token (its ``allow_nan`` defaults to True). Accepted deliberately rather
    than encoded around, because this payload is never written anywhere: its only two
    consumers are ``competition_digest`` -- which hashes it in-process -- and
    ``is_competition_rule``, which compares it in memory. A bundle stores the resulting hex
    ``rule_digest``, never the payload, so no artifact and no non-Python reader ever sees the
    token. Encoding infinities as strings would be more portable and would move the digest a
    third time in one change for no behavioural gain. If a consumer ever DOES serialize this
    for something other than hashing, encode there or change it here and re-pin -- and note
    `test_the_payload_is_not_strict_json` pins the current state so that becomes a deliberate
    decision rather than a surprise (Copilot round 5, declined with reasons).

    Knobs are RESOLVED, not copied: a catalog `penalty_cap=None` scores against
    `DEFAULT_PENALTY_CAP`, so serializing the None would leave the digest blind to exactly the
    retune #276 part C performed. `agg`/`derived`/`normalization` are here because they change
    what the metric IS -- they move `u`, `b` and `r` alike -- and `baseline.py`'s digest
    already serializes the same facts. Prose is excluded so a comment can be improved without
    moving the digest.

    Deliberately ABSENT: the run config's device, `pert_chunk` and `de.backend`. They can move
    a number, but they are per-RUN provenance, not the rule -- `_check_baseline_config` already
    compares `resolved_device`, `resolved_de_backend` and `config_digest` between the artifacts
    that must agree, and a per-run value in a static digest is a value CI cannot check.
    """
    from .anchor import FULL_GATE_RAW, SEED_DERIVATION, SPLIT_HALF_RAW, _lfc_nmae_names
    from .metrics.direction import REACH_PURITY_FLOOR
    from .config import EvalConfig
    from .run import effective_normalization

    members = competition_members()
    comparator = competition_comparator()
    lfc = set(_lfc_nmae_names(list(members)))
    out = {}
    for name in members:
        spec = CATALOG[name]
        s = spec.scoring
        out[name] = {
            "direction": s.direction,
            "anchor": s.anchor,
            "penalty": s.penalty,
            "penalty_exponent": (DEFAULT_PENALTY_EXPONENT if s.penalty_exponent is None
                                 else s.penalty_exponent),
            "penalty_cap": (DEFAULT_PENALTY_CAP if s.penalty_cap is None else s.penalty_cap),
            "clamp_low": s.effective_clamp_low(),
            "clamp_high": s.clamp_high,
            # The unusable-submission sentinel for an UNFLOORED member: with
            # `clamp_low=None` and `penalty="none"` the effective floor above serializes as
            # `-Infinity`, which says the score is unclipped but not what a MISSING value
            # scores. That is `metric_min`, and it moves a competition number, so the digest
            # must see it -- serializing only the two clamps would leave the rule blind to
            # the one field that decides what a NaN member contributes to `avg_score`.
            "metric_min": s.metric_min,
            "allow_negative_baseline": s.allow_negative_baseline,
            "agg": spec.agg,
            "derived": (None if spec.derived is None else
                        {"numerator": spec.derived.numerator,
                         "denominator": spec.derived.denominator}),
            # RESOLVED, not declared: `expr_comparator` is an intent, not a space. Under the
            # preset both expr members resolve to `bulk_lognorm`. Freezing the declaration
            # would repeat #264 PR2 exactly -- a comparator move that changes every number
            # while the catalog string sits still, and `baseline.config_digest` already
            # serializes the resolved value for the same reason.
            "normalization": effective_normalization(spec, comparator),
            "worst_value": spec.worst_value,
            # ⚠️ The COMPONENTS of a derived member, resolved. The parent's own `agg`,
            # `derived` pair and `normalization` above do not cover them: the numerator and
            # denominator are separate catalog entries (both `scored=False`, so
            # `competition_members()` never reaches them), and moving a component's `agg` or
            # its resolved normalization moves the parent's `u`, `b` and `r` alike while
            # leaving every field above bit-identical. Nothing in the catalog forces a
            # component's normalization to agree with its parent's (codex checkpoint-2 P1).
            #
            # ⚠️ NOT the components' `agg`. `run._derived_value` sums the two components'
            # PER-PERTURBATION values directly and never consults either one's aggregation
            # policy, so a component `agg` cannot move the parent -- freezing it would only
            # churn the digest for a diagnostic-only change (codex checkpoint-2 round 2 P2).
            # `normalization` and `worst_value` DO move it: the first changes the space both
            # sums are taken in, the second fills a missing perturbation before they are summed.
            "derived_components": (None if spec.derived is None else {
                c: {"normalization": effective_normalization(CATALOG[c], comparator),
                    "worst_value": CATALOG[c].worst_value}
                for c in (spec.derived.numerator, spec.derived.denominator)}),
            "estimator": FULL_GATE_RAW if name in lfc else SPLIT_HALF_RAW,
        }
    return {
        # `rule_version` is the lever for the changes this digest CANNOT see, and 0.14.0 pulled
        # it (#317). Everything above freezes each member's scoring POLICY -- direction, anchor,
        # penalty, clamps, agg, the derived pair, resolved normalization, worst_value, estimator
        # -- and almost nothing about the member's IMPLEMENTATION: not which gene set it
        # computes over (#172). The reach purity floor used to sit on that list and no longer
        # does: #327 made it a resolved parameter and `reach_purity_floor` below freezes it, so
        # that leg of #317 is now closed by the digest rather than by the version lever. So a
        # change to what a member MEANS leaves this digest standing still, and two real bundles
        # built either side of such a change would carry the same `rule_digest` while their
        # frozen replicate anchors were computed under different metric SEMANTICS. `score.py`
        # then enrols them as mutually comparable, which is the one thing `rule_digest` is
        # supposed to certify. Bumping the version by hand is what breaks that false pairing --
        # and it invalidates every already-built bundle by design, which is why it happens once
        # per wave and not once per PR.
        #
        # `rule_version = 2` (0.14.0, 2026-08-18) means exactly three semantics changes, all
        # invisible to the fields above:
        #   * #172 -- `de_wilcoxon_sig_jaccard`, `de_wilcoxon_lfc_nmae` and both legs of
        #     `expr_mse_unbiased_capped_norm` stopped scoring each perturbation's own target gene.
        #   * the purity floor -- `de_wilcoxon_direction_reach_raw` thresholds its purity curve
        #     at the calibrated `direction.REACH_PURITY_FLOOR` (0.9) instead of the derived
        #     `1 - alpha/2` (0.975).
        #   * #271 -- `prep._grouped_sums` reduces WIDE (a floating dtype coarser than float64 is
        #     widened before the reduction) instead of reducing in the input dtype. This one is a
        #     rung LOWER than the two above: it is not a policy field and not even a member's
        #     arithmetic, it is the PSEUDOBULK that arithmetic reads. ⚠️ TWO of the six scored
        #     members, not all six: `pds_cosine` and `expr_mse_unbiased_capped_norm` (both legs)
        #     read a `bulk_lognorm` bulk, while the four `de_wilcoxon_*` members read DE tables
        #     computed from CELLS and are untouched -- their `de_deseq2_*` siblings do move, since
        #     that backend pseudobulks through the same helper, but it is opt-in and can never
        #     form an enrolled official bundle. MEASURED: the stored FRACTIONAL baseline arms move
        #     up to 0.265 counts / 5.7e-06 in bulk space on the three official contexts (on the
        #     archives as stored); integer counts below float32's 2**24 do not move at all.
        # The three #276 val bundles are rebuilt against this version in the same wave;
        # anything stamped 0.13.0 pairs with neither it nor them.
        #
        # `rule_version = 3` (shipped in 0.15.0; landed 2026-08-19) means exactly THREE further
        # semantics changes, again invisible to the fields above. All three land in the SAME wave
        # and none HAD had a bundle built against it when they joined, which is why the second and
        # third did not open a version 4 -- the version already covers them. ⚠️ Past tense on
        # purpose: the three `-r3` val bundles were built at 0.15.0, so version 3 is CLOSED. A
        # fourth semantics change cannot join it; see the ⚠️ on `rule_version` below.
        #   * #343 -- `pds_cosine` ranks in the feature space with EVERY panel target gene
        #     removed (`discrimination.exclusion_scope = "panel"`), where rule_version 2 removed
        #     only the prediction row's own and so left each reference perturbation's knockdown
        #     visible in the off-diagonal cells of its distance matrix. That asymmetry was a
        #     scoreable channel: a submission spiking the panel's OTHER targets measured
        #     pds_cosine 0.7982 / 0.7570 / 0.7614 on the three official contexts against
        #     baselines 0.5304 / 0.5284 / 0.5102 -- +0.57 / +0.49 / +0.51 of member score on no
        #     information beyond the target list. Under version 3 those arms measure 0.5000,
        #     the control-paste floor, a perfect submission still measures 1.0000, and a partial
        #     one moves by at most 0.01 of member score.
        #   * #348 -- `expr_mse_unbiased_capped` bounds the prediction's TOTAL sampling correction
        #     by the submission's OWN across-perturbation centred sum of squares as well as by the
        #     reference's per-row cap (`r * min(jk_pred, k * jk_real)` with
        #     `r = min(1, B_pred / sum_q w_q min(jk_pred_q, k * jk_real_q))` with `w_q = 1/|G_q|`
        #     on both sides, `delta.py`). Version 2
        #     bounded it only by the reference, and that bound SATURATES, so a submission whose
        #     per-cell scatter cancels in the pseudobulk collected a constant `k * jk_real`
        #     against a plug-in distance containing none of it -- the measured half of #294.
        #     MEASURED on the official val panels: pinning the per-(p, g) sums of an honest
        #     control-paste, changing nothing else, moved that member from 0.0000 to 0.9031
        #     `from_baseline`, and a dev-leaderboard submission took +0.1389 of a 0.2295 OVERALL
        #     through it. Under version 3 those arms return to ~1.0, the "predicted the control"
        #     value. ⚠️ Unlike #343 this one does NOT move the stored ends of the scale: the
        #     baseline arm's cells are identical so its `jk_pred` is exactly 0 and no bound can
        #     bind, and the replicate anchor carries real across-perturbation biology. MEASURED
        #     at the anchor's own half depth, two ways: per row, `Var_across_pert` 46.64 against a
        #     median `jk_pred` of 29.34 (ratio 1.59); and as the quantity the bound ACTUALLY forms,
        #     the weighted total `B_pred/claim` = 1.54 / 1.38 / 1.58 on val A / B / C over the
        #     anchor's five derived seeds -- so `r = 1` and the value is unchanged. Submissions'
        #     scores move; baseline and anchor VALUES do not. ⚠️ The anchor ARTIFACT is still not
        #     reusable across this bump: `_ONTARGET_EXCLUSION_SEMANTICS` 1 -> 2 moved
        #     `anchor_semantic_identity` and `validate_anchor` refuses an `-r2` anchor here, so the
        #     `-r3` build RECOMPUTES it (and #343 moves its `pds_cosine` regardless).
        #   * #351 -- the DE gene gate keeps a (target, gene) row on the REFERENCE group's mean
        #     CPM alone, where rule_version 2 kept it when the TARGET group's cleared the
        #     threshold OR the reference's did. Under the OR, a gene at or below the threshold in
        #     the control entered perturbation `t`'s rows only when it ROSE above the threshold in
        #     `t`, so `tmean > threshold >= ref_mean` and log2FC > 0 -- a row's mere PRESENCE
        #     disclosed its sign. (#351 states the stronger bound
        #     `log2FC > log2(threshold/ref_mean)`; the pseudocount breaks it, since
        #     `(threshold+eps)/(ref_mean+eps) <= threshold/ref_mean` whenever
        #     `ref_mean <= threshold`. Strict positivity is what is true and what was measured.)
        #     MEASURED on the official val panels: P(real log2FC > 0) = 1.000000 over
        #     26,373 / 33,969 / 26,839 such rows (0.88%-1.16% of the reference table, 88-113 per
        #     perturbation), and a submission that pasted control cells with counts added to
        #     exactly those genes -- reading NO perturbation-specific information, the same block
        #     submitted for all 300 targets -- measured `de_wilcoxon_direction_fidelity_yield_raw`
        #     0.689661 / 0.764160 / 0.688485 against baselines 0.505647 / 0.522736 / 0.509365,
        #     i.e. +0.3722 / +0.5059 / +0.3651 `from_baseline` and +0.1057 of OVERALL `avg_score`.
        #     It also stacked on a submission carrying real signal (+0.711 of raw value on a
        #     25%-real-cells arm) rather than competing with it. Under version 3 those arms return
        #     to the honest control-paste floor: raw 0.001249 / 0.001124 / 0.000272 with `n_pred`
        #     0.0, and the boost's contribution to the signal-carrying arm goes to -0.00006.
        #     ⚠️ Unlike #348 this one DOES move the stored ends of the scale, in the direction that
        #     helps: baselines move by at most 0.0039 on any of the four `de_wilcoxon_*` members
        #     of any context, while EVERY replicate anchor moves UP and every span widens (2-5% on
        #     val A, 22-45% on val B, 3-6% on val C) -- the gate was spending part of the members'
        #     resolution on direction-selected rows. ⚠️ It is also the one entry here that is not
        #     purely a cell_eval2 change in effect: the gate that leaked runs INSIDE gpudge
        #     (`_filter.combined_keep_mask` ANDs each active filter's "(target OR ref)" mask), and
        #     `compute_de` returns from its gpudge branch before `_apply_cpm_filter` ever runs, so
        #     the reference-only decision is now taken by cell_eval2 on gpudge's returned frame
        #     (`_finalize_gpudge_de` <- `_gpudge_gate_plan`). gpudge is unchanged and unpinned. On
        #     the competition path -- and ONLY where gpudge itself normalized the cells -- the gate
        #     compares the frame's own `ref_mean`, which IS the array gpudge's gate scales, so it is
        #     bit-exact and gpudge's gate stays on (the reference-only set nests inside the OR set).
        #     Everywhere else -- lognorm, CPU-pre-normalized counts, a geometric `mean_calc`, an
        #     unknown library -- the gate compares a per-cell-CPM reference vector cell_eval2 derives
        #     from the reference cells, and gpudge's gate is MUTED: the two agree mathematically but
        #     not bit-for-bit, and a boundary gene dropped by the OR for one target and kept here for
        #     another would restore exactly the target-dependence this entry removes.
        #     ⚠️ CACHES ARE NOT INVALIDATED BY THIS. Nothing in EITHER DE-table cache key (`run.py`'s
        #     in-memory `params`, `scale.py`'s separate shard-streaming key), the result fingerprint,
        #     the anchor's semantic identity or the reference-bundle semantics records what the gate
        #     MEANS -- only the configured threshold VALUE -- so ANY warm cache written before this
        #     change still hits and serves OR-gated rows, whatever version it was labelled: a cache
        #     already stamped version 3 (built for #343/#348, before #351 joined) is just as suspect
        #     as a version-2 one. That includes the three official `-r2` frozen real caches. Closing it needs a NEW gate-semantics term
        #     modelled on `run._GROUPED_SUM_REDUCTION_SEMANTICS` and threaded through all five of
        #     those surfaces -- bumping that existing counter would NOT do it, since it never reaches
        #     the gpudge DE-table keys. Deliberately not in this change; until it lands, the rebuild
        #     wave MUST run against a cold cache.
        # The three #276 val bundles must be rebuilt against this version; anything stamped
        # rule_version 2 pairs with neither it nor them.
        #
        # ⚠️ `competition_digest()` cannot structurally SEE any of the three, so none of them moves
        # it on its own account. It moved ONCE, when #343 bumped `rule_version` 2 -> 3 (`80558072...`
        # -> `fb5aa56b...`), and then stayed fixed as #348 and #351 joined: the number below stays 3.
        # That was the whole reason a third entry was allowed to join -- no bundle had yet been
        # built against version 3, so nothing in the field carried a digest that would be
        # falsified.
        #
        # ⚠️ THAT WINDOW IS NOW SHUT, and this is the sentence to read before proposing a
        # "small" metric change. The three official `-r3` val bundles were built at 0.15.0
        # against `fb5aa56b...`, so bundles in the field DO now carry a version-3 digest. The
        # next semantics change costs a bump to `rule_version = 4` PLUS a full rebuild of every
        # val AND test bundle. Nothing gets folded in for free any more.
        #
        # The cost of getting this wrong is asymmetric and worth stating: build a v3 bundle first
        # and land #351 second, and the two bundles carry the SAME `rule_digest` while their frozen
        # baselines and anchors were computed under different DE semantics -- `score.py` would
        # enrol them as mutually comparable, exactly the false pairing the version lever exists to
        # break.
        #
        # Debt outstanding against `rule_version = 3` -- landed since this was last bumped and
        # therefore NOT covered by the digest a bundle carries today:
        #   * (none -- #348 and #351 are in the list above, not here: both landed BEFORE any
        #     bundle was built against version 3, so the version already covers them.)
        # Add to this list, do not replace it: the bump is once for the whole set, so the list is
        # what tells the person doing it what the new version means. Empty it when you bump.
        #
        # ⚠️ A POLICY change does not belong on that list and owes no bump -- it moves
        # `competition_digest()` on its own. Two landed inside the 0.13.0 gap and are recorded in
        # the CHANGELOG rather than here: `ERROR_LINEAR` (`de_*_lfc_nmae` off the Box-Cox tail
        # below the baseline; `penalty`/`clamp_low` ARE serialized above) and the clip-at-0
        # removal on the four bounded members (`clamp_low` 0.0 -> None with `metric_min`).
        # ⚠️ In particular, do NOT read the purity floor's entry above as "that member's
        # policy row was untouched": the clip-at-0 removal changed
        # `de_wilcoxon_direction_reach_raw`'s own `clamp_low` in the same release. The two are
        # independent -- one is a scoring policy this payload serializes, the other is metric
        # arithmetic it cannot see.
        "rule_version": 3,          # bump deliberately; the digest moves with it
        "profile": COMPETITION_PROFILE,
        "members": out,
        # The reduction is an unweighted arithmetic mean over the members in FRAME order,
        # which is what `_reference_column` actually sums. That order is NOT profile order:
        # `aggregate_metrics_wide` sorts metric columns BY NAME and `score_metrics` preserves
        # that within its lower-then-higher partitions. Freezing profile order here would
        # freeze an order nothing uses.
        "member_order_in_frame": sorted(m for m in members
                                        if CATALOG[m].scoring.direction == "lower")
                                 + sorted(m for m in members
                                          if CATALOG[m].scoring.direction == "higher"),
        "reducer": "unweighted_arithmetic_mean",
        "comparator": comparator,
        # The REQUESTED scoring config, by `rule_config_hash` -- `cache.config_hash` with the
        # fields spec §5.7 excludes (plus `pert_col`) normalized away. The same function the anchor
        # already stamps as `config_hash` (anchor.py:500) underneath, so the two are
        # comparable. Host-independent by construction, which is what lets a static repo
        # constant reproduce it.
        #
        # ⚠️ NOT `baseline.config_digest`, which is what `run_meta.json` stamps under the name
        # `config_digest`. That one RESOLVES machine-resolvable spellings -- `device="auto"` to
        # the actual device, the DE backend when a DE metric is requested -- and folds in the
        # effective comparator and a supplied-DE fingerprint. It is therefore HOST-DEPENDENT
        # and cannot be reproduced by a repo constant at all: two hosts legitimately differ.
        # Comparing a `config_hash` against a `config_digest` is false for every real run.
        # `config_digest` is still the right SUBMISSION<->BUNDLE gate, and the bundle compares
        # it peer-to-peer (both sides stamped by the same host), which is the one comparison
        # for which it is valid.
        "config_hash": rule_config_hash(EvalConfig.from_preset(COMPETITION_PROFILE)),
        # The EXCLUSION SET itself, because `config_hash` alone cannot see it: the hash above
        # normalizes each excluded field to the preset's own value, so hashing the PRESET is a
        # no-op for every exclusion and the digest does not move when one is added or removed.
        # Measured when `pert_col` was excluded (2026-08-13): the digest was bit-identical
        # before and after, i.e. a change to WHICH runs count as the competition was invisible
        # to the frozen digest. Recording the list makes that change loud -- an accidental
        # exclusion of `control` or `target_gene_map`, both of which move every number, now
        # fails `test_the_digest_is_frozen` instead of silently widening the rule.
        #
        # DERIVED from both exclusion tuples, never hand-listed: `rule_config_hash` drives the
        # nested `de.backend` exclusion from `_RULE_EXCLUDED_NESTED`, so deleting that entry
        # moves this list and therefore the digest. A literal string here would have been
        # documentation, and documentation cannot close a blind spot.
        "rule_excluded": _excluded_names(),
        # Checked OUTSIDE the hash (see `is_competition_rule`), so it needs recording for the
        # same reason as the exclusions: dropping the requirement would widen the rule silently.
        "require_cache_strict": REQUIRE_CACHE_STRICT,
        # The profile RESOLVED, all ten names. `members` above is the six SCORED ones, so a
        # diagnostic joining or leaving `vcc2026` changes what `is_competition_rule` demands
        # -- it compares the full resolved list -- while leaving `members` untouched.
        "profile_resolved": sorted(
            resolve_metrics(COMPETITION_PROFILE,
                            version=EvalConfig.from_preset(COMPETITION_PROFILE).version)[0]),
        "base_seed": BASE_SEED,
        "n_splits": N_SPLITS,
        "seed_derivation": SEED_DERIVATION,
        "derived_seeds": list(derived_seeds()),
        "bulk_target_sum": BULK_TARGET_SUM,
        # `de_wilcoxon_direction_reach_raw` thresholds its purity curve here. It is a scored
        # member and this is the one number its arithmetic turns on, so leaving it out made the
        # rule digest blind to a change that moves the member -- #317's registered debt, and the
        # reason #322 had to be paid for with a `rule_version` bump instead of a digest move.
        # #327 made it a function parameter, which is what lets a static digest see it at all:
        # the RESOLVED default is what every catalog-reachable spelling computes, because
        # `_register_de_family` registers `de_direction_reach` with no override and nothing in
        # `EvalConfig` or the CLI can supply one. Serializing the resolved value therefore
        # freezes what the member actually does, not what it was asked to do.
        "reach_purity_floor": REACH_PURITY_FLOOR,
        "comparison_statistic": COMPARISON_STATISTIC,
        "control_source": {"scored": CONTROL_SOURCE_SCORED,
                           "anchor": CONTROL_SOURCE_ANCHOR},
    }


def competition_digest() -> str:
    """SHA-256 over the rule. Pinned by `tests/test_competition_rule.py`."""
    blob = json.dumps(competition_payload(), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode()).hexdigest()


def is_competition_rule(config, *, members, anchor_meta, estimators) -> list[str]:
    """Every way this production run is NOT the competition rule. Empty means it is.

    Called ONCE, by `prep-real-bundle`, on the config the user asked for -- never on the
    baseline leg's internally-flipped copy. `cache.config_hash` RETAINS
    `allow_fractional_counts`, which the baseline leg sets True, so hashing the flipped config
    would return "not the competition" for every bundle ever built.

    ⚠️ `anchor_meta` and `estimators` come from the artifact the producer RECEIVED, never from
    the arguments it passed. The anchor leg may be satisfied from the content-addressed cache,
    so `base_seed=0, n_splits=5` on the call is a request; only the returned sidecar and the
    returned frame's `estimator` column are observations. Reading the request instead would let
    a stale or hand-edited cache entry receive the current competition digest.

    Two config fields are checked outside the hash because `cache.config_hash` skips them:
    `metrics` (so a `full` run would otherwise hash identically to a `vcc2026` one) and
    `cache_strict` (load-bearing -- without it the artifacts cannot be scored against a bundle
    and the two legs cannot be cross-checked by content).
    """
    _MISSING = object()
    p = competition_payload()
    out = []
    if rule_config_hash(config) != p["config_hash"]:
        out.append(
            "the run config is not the competition preset's (diff it against "
            f"`--preset {COMPETITION_PROFILE}`); its config_hash differs"
        )
    # SORTED, not the resolution order. `resolve_metrics` preserves the caller's order for an
    # explicit list, so an ordered comparison would mark `metrics: [<the ten names, reversed>]`
    # diagnostic even though it resolves to the same profile and the frame order is normalized
    # elsewhere -- which contradicts this function's own promise that an explicit equivalent
    # list counts as the competition profile.
    # `want` comes from the PAYLOAD, not from a second `resolve_metrics` call. Recomputing it
    # here let the frozen `profile_resolved` and the gate drift apart: an edit to one would be
    # invisible to the other, so a change to which profiles qualify could leave the digest
    # unmoved (codex checkpoint-2 round 2 P2).
    got_profile = sorted(resolve_metrics(config.metrics, version=config.version)[0])
    want_profile = p["profile_resolved"]
    if got_profile != want_profile:
        out.append(f"the metric profile is {config.metrics!r}, the competition scores "
                   f"{COMPETITION_PROFILE!r}")
    if REQUIRE_CACHE_STRICT and not config.cache_strict:
        out.append("cache_strict is off; a competition artifact must carry the strict content "
                   "fingerprint or it cannot be scored against a bundle at all")
    # Membership as a SET. The producer sorts the anchor frame alphabetically
    # (`anchor.py`'s `.sort("metric")`) while `PROFILES` follows catalog insertion order, so an
    # ORDERED comparison is False for every real artifact and would mark every bundle
    # diagnostic -- C-1's dead feature, exactly.
    if set(members) != set(competition_members()):
        missing = sorted(set(competition_members()) - set(members))
        extra = sorted(set(members) - set(competition_members()))
        out.append(f"the scored members are not the competition's six (missing {missing}, "
                   f"extra {extra})")
    # ...every field the ANCHOR stamped, against the frozen rule.
    #
    # ⚠️ `config_hash` is deliberately NOT in this loop. `anchor.build_meta` stamps the RAW
    # `cache.config_hash(cfg.to_dict())` (anchor.py:500), while `p["config_hash"]` is the
    # NORMALIZED `rule_config_hash`. Comparing them would undo this function's own exclusion:
    # a run with `device="cpu"` produces a raw anchor stamp that differs from the normalized
    # frozen hash, so an allowed per-run override would silently mark the bundle diagnostic
    # -- and a unit test whose sidecar helper stamps the normalized hash regardless of the
    # config under test cannot see it. The two comparisons that DO cover this are separate and
    # each is like-for-like:
    #   * `prep-real-bundle`'s leg check: the anchor's RAW stamp vs
    #     `cache.config_hash(config.to_dict())` -- "this anchor was built under THIS run's
    #     config", which is what catches a foreign cached anchor;
    #   * the `rule_config_hash(config)` comparison above: "this run's config IS the
    #     competition's", with the ruled per-run fields normalized away.
    # Together they are transitive, and neither can reject a legal override.
    for field, want in (("base_seed", BASE_SEED), ("n_splits", N_SPLITS),
                        ("seed_derivation", p["seed_derivation"]),
                        ("bulk_target_sum", BULK_TARGET_SUM),
                        ("control_source_effective", CONTROL_SOURCE_ANCHOR)):
        got = anchor_meta.get(field, _MISSING)
        if got is _MISSING:
            out.append(f"the anchor stamps no {field}")
        elif field in ("base_seed", "n_splits") and int(got) != want:
            out.append(f"the anchor's {field}={got!r}, the competition rule says {want!r}")
        elif field == "bulk_target_sum" and float(got) != want:
            out.append(f"the anchor's {field}={got!r}, the competition rule says {want!r}")
        elif field in ("seed_derivation", "control_source_effective") and got != want:
            out.append(f"the anchor's {field}={got!r}, the competition rule says {want!r}")
    got_seeds = anchor_meta.get("derived_seeds")
    if got_seeds is None or [int(s) for s in got_seeds] != list(derived_seeds()):
        out.append(f"the anchor's derived_seeds={got_seeds!r}, the rule derives "
                   f"{list(derived_seeds())!r}")
    # ...and each member's ESTIMATOR, read off the returned frame. A member silently switched
    # between the split-half and full-real-gated estimator answers a different question with
    # the same column name.
    want_est = {m: p["members"][m]["estimator"] for m in competition_members()}
    if bad := {m: estimators.get(m) for m in want_est if estimators.get(m) != want_est[m]}:
        out.append(f"the anchor's estimator differs for {bad!r}; the rule says "
                   f"{ {m: want_est[m] for m in bad} !r}")
    return out
