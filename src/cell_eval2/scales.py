"""Named, frozen SCALES: fixed reference points that score against constant endpoints.

A scale answers "where does this submission sit between two fixed points" rather than "how
does it compare to the baseline we published". It supplies, per metric, a constant zero
point ``base`` plus a full :class:`~cell_eval2.scoring.Scoring` policy. Nothing here is new
arithmetic -- ``scoring.score_one`` already computes ``(u - b) / (a - b)``; a scale just
hands it a constant ``b`` instead of a measured baseline.

Three properties follow from ``base`` being a constant rather than a measurement, and they
are the whole reason the registry exists:

* **No artifact.** A scale needs no baseline build, so there is nothing to regenerate when
  ``cell_eval2_version`` changes -- ``score`` compares that field exactly, which invalidates
  every measured baseline against artifacts stamped by another release. (A same-generation
  artifact PAIR is internally consistent, and the version peer alone would still score it. Note
  that a COMPETITION bundle has a second, independent gate: `score.py` compares its frozen
  `rule_digest` against the runtime `competition_digest()` and raises, so it does not survive a
  release that moves the competition rule regardless of how its peers pair. A DIAGNOSTIC bundle
  carries `rule_digest = None` and the gate skips it -- `score.py`'s guard is
  `rule is not None and rule != want`.)
* **It cannot go degenerate.** ``is_degenerate`` is checked at import for every entry, so
  the failure that drops a metric out of a measured ``avg_score`` cannot reach a scale.
* **It reads absolutely.** ``low-random_high-1_v10`` puts 0 at the random minimum and 1 at
  real input, so a value means something without the comparator in hand.

**A shipped scale is IMMUTABLE.** Any change to any field mints a new ``_v<n>``; a name is
never redefined. That is the ``expr_mse_unbiased`` -> ``expr_mse_unbiased_norm`` lesson made
structural: that rename shipped with no alias on purpose, precisely so an old column could
not silently bind to a metric with a different definition. ``tests/test_scales.py`` pins a
digest of this registry, so editing a shipped table fails CI rather than quietly moving
published numbers.

**Immutable means never REDEFINED, not never RETIRED** (Alex, 2026-08-08).
``low-random_high-1_v1`` was retired in #257 rather than edited: it keyed
``expr_mse_unbiased_norm``, and ``build_scale`` validates at import that every key names a
catalog metric, so removing that metric made ``_v1`` unconstructible -- and because
``__init__`` imports ``score`` which imports this module, that is an unimportable PACKAGE,
not a stale number. Keeping a deprecated stub metric alive purely to satisfy a scale would
have been worse. A scale naming a metric that no longer exists cannot be evaluated, so
publishing it would be a lie. ``_v2`` replaced that metric with
``expr_mse_unbiased_capped_norm``; it was then retired in #264 because ``pds_cosine`` moved
to the new expression comparator. ``_v3`` carried the same table, then retired because #264
also moved ``expr_mse_unbiased_capped_norm`` to that comparator. ``_v4`` carried the unchanged
table without redefining the meaning of a shipped name.

⚠️ ``_v5`` (#268) is the FIRST version minted for a **config-parameter** change rather than a
metric-definition one, and it does not follow from the "any change to any field" rule above:
no field moved -- ``bulk_target_sum`` went 1e6 -> 5e4, which shifts the scored values without
touching this table, and ``scales_digest()`` would not have noticed the SHIFT (it sees the
table and the name, and neither had moved). Minted deliberately (Alex, 2026-08-11) so the name at
least CHANGES when the numbers under it do -- which is also what then moves the digest.

``_v6`` (#282) was the "what a keyed metric MEANS changed" case the rule names explicitly, and
the first one: ``pds_cosine``'s tie handling moved from the legacy argsort position --
which resolved an all-tied row to the target's ALPHABETICAL index -- to the mid-rank, so a
zero-effect target now scores 0.5 rather than 1.0 or 0.0 depending on its name. The table is
byte-identical to ``_v5``'s, so ``scales_digest()`` would not have noticed the change of MEANING
either -- only the rename it forced; the
base of 0.5 for ``pds_cosine`` is in fact MORE true under the new definition, since it is now
the value a no-information prediction actually attains per target rather than only on average
across a fully-degenerate panel.

``_v7`` (#172) is the SECOND "what a keyed metric MEANS changed" mint, and the widest: three
of the six keyed members -- ``de_wilcoxon_sig_jaccard``, ``de_wilcoxon_lfc_nmae`` and
``expr_mse_unbiased_capped_norm`` -- stopped scoring each perturbation's OWN target gene, so
every one of them is computed over a different gene set than it was under ``_v6``. The table
is byte-identical to ``_v6``'s for the third time running, so ``scales_digest()`` would not have
noticed the change in MEANING at all -- what moves the digest is the RENAME itself, since
``scale_payload`` carries ``name``. That is exactly why the mint IS the signal here rather than a
label on one: the numbers under the heading move
(the removed on-target gift measured 10.21%/11.30%/6.07% of ``expr_mse_unbiased_capped_norm``'s
[0, 1] range on the three #276 val bundles, and 0.02 raw on ``de_wilcoxon_sig_jaccard``) while
nothing in this file does. All three bases survive
the redefinition rather than merely being carried over, and TWO of the three for the same
structural reason -- the excluded gene leaves a numerator and a denominator together, so the
ratio's fixed point does not move. ``expr_mse_unbiased_capped_norm``'s 1.0 is PRESERVED, which
is the claim to make and not "exact": the caveats below it already say the shipped estimator
misses that anchor by ~0.32% and that #247's cap makes even an unbiased correction miss it, and
#172 neither adds to nor removes from that -- both legs of the ratio drop the same gene, so
whatever exactness the anchor had it still has. (A numerator-only exclusion would NOT preserve
it -- see ``tests/test_target_gene_exclusion_172.py``.) ``de_wilcoxon_lfc_nmae``'s 1.0 is its
no-skill point, ``mean|lfc_pred - lfc_real| / mean|lfc_real|`` at ``lfc_pred = 0``, where the
numerator IS the denominator; one gate feeds both, so dropping a row drops it from both and the
identity survives on the smaller gate. ``de_wilcoxon_sig_jaccard``'s 0.0 remains the metric's
lower BOUND on any gene set whatever; it stops being ATTAINABLE for a perturbation whose only
reference-significant gene was its own target, since the union is then empty and the
``J(0,0) = 1`` convention returns 1.0 (``docs/metrics.md`` section 4). A bound is what a scale
base needs, so 0.0 stands.

``_v8`` is the THIRD such mint and it retires ``_v7``: ``de_wilcoxon_direction_reach_raw``'s
purity floor moved from the derived ``1 - alpha/2`` (0.975) to the calibrated
``direction.REACH_PURITY_FLOOR`` (0.9), so every value keyed under that name moves while this
table does not -- byte-identical for the FOURTH mint running. Its own entry is correctly
UNCHANGED: ``base = 0.0`` is the metric's minimum and ``anchor = 1.0`` its perfect value, and
neither depends on the floor (the no-skill point is DOMINATED by the ``k* = 1`` event, which
the floor barely touches -- measured, a random-sign predictor reads 0.042694/0.065849/0.080890 on the three
#276 val lines at 0.975 and 0.042694/0.067557/0.082166 at 0.9: A exactly unmoved, B and C up
by 0.0017 and 0.0013). ⚠️ That is exactly why the NAME has to move.
``scales_digest()`` is computed over ``scale_payload``, which carries the table AND the ``name``:
no number moved here, so the RENAME is the only thing that moves the digest -- and without it a
published ``_v7`` column would silently span two definitions of a scored member.

``_v9`` retires ``_v8`` (**Alex, 2026-08-18**): #271 made ``prep._grouped_sums`` reduce WIDE -- a
floating dtype coarser than float64 is widened before the reduction -- which moves values keyed here
for two members, ``pds_cosine`` and ``expr_mse_unbiased_capped_norm``, while this table does not
move. Byte-identical to ``_v8``'s, as every mint since ``_v2`` has been. No ordinal is claimed for it
on purpose: ``_v6``/``_v7``/``_v8`` are the "what a keyed metric MEANS changed" sequence and this is
NOT one of those, while ``_v5`` and this one are both "the numbers moved and the table did not"
without forming a tidy series. Precisely the ``_v5``/``_v8`` case: no number moved, so the RENAME is
the only thing moving ``scales_digest()`` (``22b3d6b1...`` -> ``8542ae14...``), and without it a
published ``_v8`` column would silently span two GROUP-SUM eras -- one rung lower than the three mints before it, which each changed what a keyed
member MEANS, because this one changed the PSEUDOBULK that meaning is computed from.

``_v10`` retires ``_v9`` (2026-08-19), and it is the FOURTH "what a keyed metric MEANS changed"
mint -- the case the rule names explicitly -- covering TWO changes in one wave, exactly as
``rule_version = 3`` does:

* #343 -- ``pds_cosine`` now ranks with EVERY panel target gene removed from the feature space
  rather than only the prediction row's own. ⚠️ **That change shipped in ``1c05408`` without a
  mint, so ``_v9`` had already begun to span two definitions of a keyed member; this pays that
  debt rather than opening a fifth name for it.**
* #348 -- ``expr_mse_unbiased_capped``'s prediction-side correction is bounded by the
  submission's own across-perturbation centred sum of squares, so every capped value where that
  bound binds moves. ⚠️ It also makes this entry's ``base`` CONDITIONAL: see the note on the
  ``expr_mse_unbiased_capped_norm`` row below.

Byte-identical to ``_v9``'s table, as every mint since ``_v2`` has been; what moved is what two of
the keyed members mean, which is precisely what a byte-identical digest cannot see.

⚠️ Three arguments for an exception were drafted and are recorded REFUTED rather than deleted, since
each is the kind that gets re-derived -- and two of them are refuted by tests in the same change:

1. *"The competition profile rejects the input that moves."* FALSE **as of the ``_v9`` mint** -- read
   the whole item before quoting it: one third of it has since been repaired (Copilot asked for this
   flag up front rather than only in the ⚠️ below). ``configs/vcc2026.yaml`` does pin
   ``allow_fractional_counts: false`` with ``validate_input: true``, but ``norm._is_all_integer``
   compared with ``np.allclose`` at its default ``rtol=1e-5``, and nothing checks dtype WIDTH.
   ``tests/test_jackknife.py`` pinned three submissions that passed both numeric input gates
   (``validate_input_type`` and ``check_scale_limit``, not a whole ``vcc2026`` run) and move:
   integer-valued fp32 above ``2**24``, near-integer fp32 (1000.001) below it, and integer-valued
   float16 above ``2**11``. ⚠️ **Kept as the record of why ``_v9`` was minted, and one third of it
   has since been repaired**: ``norm._INT_ATOL`` (2026-08-18) made the integrality tolerance
   ABSOLUTE, so the near-integer submission is now rejected. The refutation stands on the other
   two -- integer-valued fp32 above ``2**24`` and float16 above ``2**11`` both still pass every
   gate ``vcc2026`` applies, because nothing checks dtype WIDTH.
2. *"A scale is only for baseline-free runs, and a baseline arm is not scored through one."* FALSE
   twice over -- ``score`` applies a requested scale AFTER either scoring mode and ``cli`` permits
   ``--scale`` alongside ``--baseline-agg`` or ``--real-bundle``, so a fractional PREDICTION can
   receive the column whatever else is in the frame.
3. *"The official arms are bit-identical, so nothing moves."* True of those arms
   (``max|delta| = 0.000000``, measured on the stored archives) and says nothing about submissions.

⚠️ The paragraph immediately below -- a scale name is a LABEL, never a certification -- is why a mint
buys DOCUMENTATION rather than a guarantee. That is a reason to weigh the cost, not a reason to skip
it: the cost is one rename and its references, and the alternative is a name that means two things.

⚠️ **A scale name is a LABEL, not a certification.** ``score`` applies a requested scale
without checking run identity, so a column headed ``low-random_high-1_v10`` does NOT prove the
aggregate under it was produced at ``bulk_target_sum=5e4`` -- an older 1e6 aggregate, or a run
with an explicitly different target, still gets that heading. No scale has ever certified
this; fail-closed provenance is #276's preset bundle, and that is where it belongs.

**Do not generalize this into a rule** that every config knob
mints a scale version -- ``target_sum`` and ``filter_gene_min_cpm_cell`` move scored values
too and have never done so. The general fix is run identity carried by the #276 preset
bundle; when that ships, this reason expires.

Layering: this module imports ``catalog`` and ``scoring`` and is imported by ``score``. It
must never be imported BY ``catalog`` or ``scoring`` -- ``scoring`` is deliberately
catalog-free so the arithmetic engine stays pure.
"""
from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType

from .catalog import CATALOG, _NAME_TO_CANONICAL
from .scoring import Scoring, is_degenerate

#: Column names ``score_metrics`` owns. A scale's name becomes a column name, so a scale
#: spelled like one of these would overwrite it instead of adding to the frame. The anchor
#: columns joined in #276 part B; the two bundle columns and the enrolment in part C.
RESERVED_COLUMNS = frozenset({"metric", "from_baseline", "from_reference",
                              "from_replicate", "anchor_source", "anchor_digest",
                              "real_bundle_id", "real_bundle_digest"})


@dataclass(frozen=True)
class ScaleEntry:
    """One metric's place in a scale: where zero sits, and how the score is shaped.

    ``base`` plays the role a measured baseline plays in ``score_metrics`` -- it is the
    ``b`` in ``score_one(user, b, scoring)``. ``scoring`` carries everything else: the
    anchor (the 1 end), the direction, the penalty shape and both clamps.
    """

    base: float
    scoring: Scoring


@dataclass(frozen=True)
class Scale:
    """A named, frozen set of per-metric reference points.

    ``entries`` is keyed by CANONICAL metric name and is a read-only mapping, so a scale
    cannot be mutated after ``build_scale`` validated it.
    """

    name: str
    description: str
    entries: Mapping[str, ScaleEntry]


def build_scale(name: str, description: str,
                entries: Mapping[str, ScaleEntry]) -> Scale:
    """Canonicalize and validate one scale. Raises rather than shipping a broken table.

    Every check here runs at IMPORT time for the shipped registry, which is the point: a
    scale that cannot define a scale for one of its metrics must fail the build, not the
    run. ``is_degenerate`` is the load-bearing one -- it structurally enforces "``base``
    sits strictly on the worse side of the anchor", and it is why a scale can never suffer
    the degenerate-baseline failure a measured baseline can.
    """
    if name in RESERVED_COLUMNS:
        raise ValueError(
            f"scale name {name!r} is reserved: a scale's name becomes a column name in the "
            f"scored frame, and {sorted(RESERVED_COLUMNS)} are columns score_metrics owns. "
            "Pick another name."
        )
    if not entries:
        raise ValueError(f"scale {name!r} must name at least one metric; got an empty table")
    resolved: dict[str, ScaleEntry] = {}
    seen: dict[str, str] = {}
    for key, entry in entries.items():
        if not isinstance(entry, ScaleEntry):
            raise TypeError(
                f"scale {name!r} entry for {key!r} must be a ScaleEntry, got "
                f"{type(entry).__name__}"
            )
        canonical = _NAME_TO_CANONICAL.get(key, key)
        if canonical not in CATALOG:
            raise ValueError(
                f"scale {name!r} names an unknown metric {key!r}. Keys must name a catalog "
                "metric in any accepted spelling (canonical, v1 alias, or other alias)."
            )
        if canonical in resolved:
            raise ValueError(
                f"scale {name!r} names metric {canonical!r} twice, as {seen[canonical]!r} "
                f"and {key!r}; pass exactly one key per metric"
            )
        # A frozen scale must not depend on a value the digest cannot see. `score_one`
        # resolves a None `penalty_exponent`/`penalty_cap` from `scoring.DEFAULT_*`, which are
        # module constants -- so a Box-Cox scale that left them None would move its own scores
        # whenever those constants were retuned, while `scales_digest()` stayed put and the
        # freeze test kept passing. The linear scale shipped today never reaches this, which is
        # exactly why the guard belongs here now rather than after someone adds a boxcox one.
        if entry.scoring.penalty == "boxcox" and (
            entry.scoring.penalty_exponent is None or entry.scoring.penalty_cap is None
        ):
            raise ValueError(
                f"scale {name!r} entry {canonical!r} uses penalty='boxcox' but leaves "
                f"penalty_exponent={entry.scoring.penalty_exponent!r} and/or "
                f"penalty_cap={entry.scoring.penalty_cap!r} unset. score_one would resolve "
                "those from scoring.DEFAULT_PENALTY_EXPONENT/DEFAULT_PENALTY_CAP, so the "
                "scale's numbers would follow a module constant that scales_digest() does "
                "not cover. Set both explicitly on a frozen scale."
            )
        if not entry.scoring.scored:
            raise ValueError(
                f"scale {name!r} entry {canonical!r} has scored=False. A scale exists to "
                "produce a score, so every entry must carry scored=True."
            )
        if is_degenerate(entry.base, entry.scoring):
            raise ValueError(
                f"scale {name!r} entry {canonical!r} is degenerate: base={entry.base!r} "
                f"against anchor={entry.scoring.anchor!r}, direction="
                f"{entry.scoring.direction!r} does not define a usable scale. Either the "
                "denominator is not finite and positive -- the base must sit strictly on the "
                "worse side of the anchor -- or, for an UNFLOORED entry, the score its "
                f"metric_min ({entry.scoring.metric_min!r}) earns is not representable. "
                "No shipped scale is unfloored, so the first is the likely one."
            )
        resolved[canonical] = entry
        seen[canonical] = key
    return Scale(name=name, description=description,
                 entries=MappingProxyType(dict(resolved)))


def _higher(base: float, clamp_low: float) -> ScaleEntry:
    """A higher-is-better member: anchor 1, linear, floored at ``clamp_low``."""
    return ScaleEntry(
        base=base,
        scoring=Scoring(scored=True, direction="higher", anchor=1.0,
                        penalty="none", clamp_low=clamp_low, clamp_high=1.0),
    )


def _lower(base: float, clamp_low: float) -> ScaleEntry:
    """A lower-is-better member: anchor 0, linear, floored at ``clamp_low``."""
    return ScaleEntry(
        base=base,
        scoring=Scoring(scored=True, direction="lower", anchor=0.0,
                        penalty="none", clamp_low=clamp_low, clamp_high=1.0),
    )


_LOW_RANDOM_HIGH_1_V10 = build_scale(
    "low-random_high-1_v10",
    "0 = the random minimum, 1 = real input (the real count matrix pasted as the "
    "prediction). Covers the six SCORED members of the vcc2026 profile, each computed "
    "with every perturbation's own target gene excluded (#172); its four unscored "
    "expression diagnostics are not scale entries.",
    {
        # --- lower is better -----------------------------------------------------------
        # base 1.0: a submission emitting the control unchanged reads 1.0, and as of #257 that
        # is a PROPERTY on any panel whatever the REFERENCE's depth rather than a declaration.
        # ⚠️ CONDITIONAL as of #348 (`_v10`): the property needs the prediction-side correction to
        # survive #348's budget, i.e. `r = 1`. An arm emitting an INDEPENDENT draw of the control's
        # cells per perturbation is at r ~ 0.97 and reads 1.0 to within that; an arm reusing ONE
        # emitted cell block for every perturbation has zero across-perturbation spread, so r = 0
        # and it reads `1 + C_pred/sum_p den_p`. Its correction is real but perfectly common-mode,
        # and common-mode error is not identifiable from BIAS in one submission -- see
        # `metrics/delta.py::_numerator`. Neither arm's SCALED score moves: both sit at or above
        # this base, i.e. at the 0 end for a member where lower is better
        # -- the denominator is debiased and the aggregate is a ratio of sums. ⚠️ Exact for the
        # ORACLE form in the true variance traces, not for the shipped estimator: it inherits the
        # correction's bias, and under #264's bulk_lognorm that correction is a jackknife whose
        # bias depends on `bulk_target_sum`. The capped member also CLIPS, and
        # E[min(C_pred_hat, k C_real_hat)] != min(E[C_pred_hat], k * E[C_real_hat]), so even an
        # unbiased correction would leave that anchor inexact.
        # At the retired 1e6 it measured 2.06% high and this anchor read 1.073 -- i.e. the base
        # shipped 7.3% wrong -- which is why #268 moved the shipped value to 5e4, where
        # `moments.jackknife_correction`'s sweep puts it at 0.32% (~0.07% net of the +0.25%
        # complementary-subset artifact). The scale's base is a policy constant either way --
        # it does not move with the panel; what 5e4 buys is that it is CLOSER to true. On the
        # panel #268 characterises, this anchor reads 1.0249 at 5e4 against 1.0727 at 1e6 --
        # a 2.9x smaller drift, not a fix: a control-emitting submission still scores slightly
        # below 0 here rather than exactly 0. (A submission whose own correction exceeds k times
        # the reference's reads above 1 in the matched-iid no-skill ORACLE calculation, i.e.
        # scores below 0: #247's cap refuses a correction the reference does not earn. ⚠️ "fewer
        # cells OR more dispersed ones" used to be given as the trigger. That is an expectation
        # under a matched-iid emission model, and #278 showed the dispersion half inverting
        # pointwise under `bulk_lognorm`; pointwise the inequality decides only whether the cap
        # BINDS, not which side of 1 the value lands.) Under v1's metric it read 0.7643
        # on VCC Test and 0.2386 on CCL_2. clamp_high=1.0 is LOAD-BEARING here and only here:
        # the metric is signed, so a paste overshoots and the clamp is what makes it read 1.0.
        # clamp_low=-6 is a policy call (Alex, 2026-08-06), carried over unchanged.
        # As of #172 both legs are summed over G-1 genes rather than G (the perturbation's own
        # target gene is dropped from the plug-in distance and from the divisor; the jackknife
        # correction C is left whole -- see `metrics/delta._drop_on_target`). The ORACLE fixed
        # point is preserved, and so is the SOURCE of the estimator drift described above,
        # because the drop is symmetric: numerator and denominator lose the same gene's term.
        # Dropping it from the numerator alone would move the fixed point, which is what makes
        # the symmetry load-bearing.
        # ⚠️ The drift's MAGNITUDE is not preserved, and an earlier draft of this comment wrongly
        # said it was (codex round 3). Writing the numerator as `D + eps` and the common removed
        # term as `T`, the ratio goes `1 + eps/D` -> `1 + eps/(D - T)`: same fixed point, drift
        # scaled by `D/(D - T)`. `T` is measured -- the target gene is 5.5%/5.0%/3.2% of the raw
        # summed distance on val A/B/C (`metrics/delta.distance_unbiased`) -- so to first order
        # the drift AMPLIFIES by about +5.8%/+5.3%/+3.3% relative, e.g. a 2.49%-high anchor
        # reading about 2.63% high on an A-like panel. First-order only: partial resolution also
        # reweights the per-row terms, and this has not been remeasured end to end.
        "expr_mse_unbiased_capped_norm": _lower(1.0, -6.0),
        # base 1.0: the `n` normalization makes a prediction whose log2FC is exactly zero
        # read 1.0 -- at lfc_hat = 0 the numerator IS the denominator -- so this end of the
        # scale needs no reference to the evaluation data. With the -1.0 floor the linear
        # score 1 - u reaches it at exactly u = 2.0, the "predicting backwards, twice as bad
        # as silence" point `docs/metrics.md` §4.3 names. Unlike `pds_cosine`'s, that floor
        # is a POLICY choice rather than a derived one: nmae is unbounded above, so nothing
        # forces a clip at 2.0.
        # ⚠️ Exact in log2FC SPACE, not in SUBMISSION space (#286). vcc2026 scores under
        # `control_source="real"` (`competition.CONTROL_SOURCE_SCORED`), so a predicted
        # perturbation's log2FC is taken against the REAL control's own cells: a submission
        # broadcasting the exact unrounded control mean is still not compared against itself.
        # What decides the gap is depth-COMPOSITION COVARIANCE, not depth spread alone -- a
        # panel with 10x depth spread at a single composition has a discrepancy of exactly
        # zero. Mechanism and the identity: `docs/metrics.md` §4.3, pinned by
        # `tests/test_lfc_nmae_anchor_286.py`.
        # Measured on the three official val bundles, an exact-control-mean submission reads
        # 1.0058 / 1.0047 / 1.0097 over 272 / 229 / 218 RETURNED perturbations (of 300 panel
        # targets; the rest fail the real-side gate). All three means came out ABOVE the base,
        # by +4.5 / +1.8 / +4.5 standard errors, so that submission scores slightly BELOW 0
        # here rather than exactly 0 -- the same situation `expr_mse_unbiased_capped_norm`
        # records above. Per perturbation the spread is far wider, 0.8769-1.4501 across the
        # three panels.
        # ⚠️ A penalty for THAT arm is not a penalty in general. It costs the submission on
        # these three aggregates and on both synthetic pairings in the test above, but the
        # DISPERSED context-mean arm gains instead -- `docs/metrics.md` §4.3 records one at
        # 0.9909 -- and that gain needs no target-specific skill, since accidental
        # depth-composition covariance can point the offset the right way on its own. ⚠️ That
        # arm is an ORACLE comparator, not a floor a submission could reach (§6a): it bounds
        # the metric's triviality rather than naming a score a model collects for free.
        # The ENROLLED `from_replicate` competition score never reads this base: it takes its
        # 0 end from a MEASURED baseline (`score._replicate_entries`). Only an explicitly
        # requested `--scale` column reads it, and on that enrolled path it changes neither
        # `from_baseline` nor its `avg_score`. (Off it, a scale CAN restore a row the baseline
        # pass declined, with a null `from_baseline` -- the numeric avg_score is unaffected
        # because nulls are excluded.) #286 was ruled docs-only on that basis
        # (Alex, 2026-08-17).
        # ⚠️ #172 does not disturb any of the above, and the reason is structural rather than
        # empirical: ONE gate supplies both halves of the ratio, so dropping each target's own
        # gene drops it from `mean|lfc_hat - lfc|` and from `mean|lfc|` together and the
        # `lfc_hat = 0` identity holds on the smaller gate exactly as it held on the full one.
        # What #172 DOES change is the cohort: the gate shrinks by one per resolved target and
        # `min_gate_size` is judged afterwards, so a perturbation sitting at the threshold is
        # omitted. ⚠️ The 1.0058/1.0047/1.0097 readings above, and the 272/229/218 returned
        # counts, were measured BEFORE that -- both the values and the cohort sizes can move a
        # little; the MECHANISM they illustrate is what carries over, not the digits.
        # ⚠️ Do not confuse this base with `lfc_nmae_ref` (#208): that is the ANCHOR leg, a
        # measured replicate reference, and it does not reach this base at all. An earlier
        # draft of this comment attributed the 1.0 to it, which was wrong.
        "de_wilcoxon_lfc_nmae": _lower(1.0, -1.0),
        # --- higher is better ----------------------------------------------------------
        # base 0.5: PDS is 1 - rank/D, so a uniform rank lands at 0.5. The -1 floor is
        # DERIVED, not chosen: the raw metric bottoms out at 0, which maps to exactly -1.
        # As of #282 this base is exact PER TARGET, not merely on a panel average: a
        # zero-effect target ties the whole cosine row and the mid-rank puts it at 0.5.
        # Under _v5 and earlier it landed on its ALPHABETICAL index instead (1.0 for the
        # first target, 0.0 for the last), which averaged to 0.5 only when EVERY target
        # was degenerate -- so the base was right for a fully-degenerate submission and
        # wrong for every partially-degenerate one. Same number, sounder now.
        "pds_cosine": _higher(0.5, -1.0),
        # base 0.5: a coin-flip sign. Measured 0.4863-0.5148 over 12 line x baseline cells.
        "de_wilcoxon_direction_fidelity_yield_raw": _higher(0.5, -1.0),
        # base 0.0: random purity 0.5 essentially never sustains the purity floor past the
        # head, so k* <= 1 and reach is ~0.5/N_conf -- measured 0.0017-0.011 on VCC Test and
        # 0.043-0.081 on the three val lines. The 0 floor is forced, not chosen: base IS the
        # metric's minimum, so the scaled value cannot go below 0.
        #
        # This entry did NOT move when `REACH_PURITY_FLOOR` went 0.975 -> 0.9, and that is a
        # measurement, not an assumption: the no-skill mean is dominated by the k* = 1 event
        # (one coin flip on the first ranked pair), which no purity floor touches. Measured on
        # the val lines, a random-sign predictor reads 0.042694/0.065849/0.080890 at 0.975 and
        # 0.042694/0.067557/0.082166 at 0.9 -- A exactly unmoved, B +0.0017, C +0.0013, so the
        # k*=1 event DOMINATES the no-skill point rather than wholly determining it. `base` is the metric's minimum either way, so
        # `scales_digest()` -- which covers the table and the name, not comments -- is
        # unchanged by this reasoning; what moves it is the mint the floor change forced.
        "de_wilcoxon_direction_reach_raw": _higher(0.0, 0.0),
        # base 0.0 is Alex's call (2026-08-06): the theoretical minimum. The ANALYTIC chance
        # level is neither 0 nor constant -- E[J] ~ (ab/G)/(a+b-ab/G), which is 0.0062 at
        # replicate-sized predictions and 0.0121-0.0124 at the generic baseline's set sizes
        # -- so 0 buys a stable anchor at the cost of crediting ~0.006-0.012 of free chance
        # overlap. Recorded here so the choice stays visible rather than looking derived.
        # 0 remains the theoretical minimum under #172's exclusion -- an empty intersection is
        # still reachable on the gene set minus one member -- so this base is unchanged in
        # substance, not merely carried over.
        "de_wilcoxon_sig_jaccard": _higher(0.0, 0.0),
    },
)

#: The registry. Read-only: a caller must not be able to add a scale at runtime and have it
#: look as frozen as a shipped one.
def _registry(*scales: Scale) -> Mapping[str, Scale]:
    """Build the registry, refusing two scales that share a name.

    A dict comprehension would silently keep the LAST of a duplicate pair, which is precisely
    the "a name is never redefined" guarantee this module claims -- broken quietly, and in the
    one place a reader would not think to check. Cannot fire with a single shipped scale; it
    exists so that adding the second one cannot introduce the failure.
    """
    out: dict[str, Scale] = {}
    for s in scales:
        if s.name in out:
            raise ValueError(
                f"two scales are both named {s.name!r}. A shipped scale is immutable and a "
                "name is never redefined -- mint a new _v<n> instead."
            )
        out[s.name] = s
    return MappingProxyType(out)


SCALES: Mapping[str, Scale] = _registry(_LOW_RANDOM_HIGH_1_V10)


def resolve_scales(names: str | Sequence[str] | None) -> list[Scale]:
    """Resolve requested scale name(s) to :class:`Scale` objects, preserving order.

    ``None`` yields ``[]``, which is what makes "no scale requested" expressible without a
    sentinel. A bare string is accepted so the common single-scale call needs no list.

    An unknown name raises rather than being ignored: a silently dropped scale is a missing
    column with nothing in the output saying so. A repeated name raises too -- it would
    otherwise produce two identical columns competing for one header.
    """
    if names is None:
        return []
    if isinstance(names, str):
        names = [names]
    out: list[Scale] = []
    seen: set[str] = set()
    for name in names:
        if name not in SCALES:
            raise ValueError(f"unknown scale {name!r}; known: {sorted(SCALES)}")
        if name in seen:
            raise ValueError(f"scale {name!r} requested twice; pass each scale once")
        seen.add(name)
        out.append(SCALES[name])
    return out


def scale_payload(scale: Scale) -> dict:
    """The frozen content of one scale, as plain JSON-able data.

    Two kinds of thing, and the distinction matters for what ``scales_digest`` can see: the
    scale's IDENTITY (``name``) and every field that changes a NUMBER. ``description`` is prose
    and is deliberately excluded so wording can be improved without minting a version.

    ⚠️ ``name`` being in here is what makes a MINT visible to the digest. Four mints running have
    carried a byte-identical table (see this module's history), so the rename is the ONLY thing
    that moved the digest each time -- which is precisely why the registry versions the name.
    """
    return {
        "name": scale.name,
        "entries": {
            metric: {
                "base": entry.base,
                "scored": entry.scoring.scored,
                "direction": entry.scoring.direction,
                "anchor": entry.scoring.anchor,
                "penalty": entry.scoring.penalty,
                "penalty_exponent": entry.scoring.penalty_exponent,
                "penalty_cap": entry.scoring.penalty_cap,
                "clamp_low": entry.scoring.clamp_low,
                "clamp_high": entry.scoring.clamp_high,
                # `None` on every shipped entry -- no frozen scale is unfloored -- but it
                # changes what a missing value scores wherever it IS set, so it belongs here.
                "metric_min": entry.scoring.metric_min,
                "allow_negative_baseline": entry.scoring.allow_negative_baseline,
            }
            for metric, entry in sorted(scale.entries.items())
        },
    }


def scales_digest() -> str:
    """SHA-256 over the registry's IDENTITY plus its score-affecting content -- i.e. exactly what
    :func:`scale_payload` serializes, names included. Pinned by a test (Task 2).

    ⚠️ Not "numeric content" alone: a mint that leaves every number alone still moves this digest,
    through the name. What it CANNOT see is a change in what a keyed member MEANS -- that is the
    gap the mint history exists to close by hand.
    """
    payload = [scale_payload(SCALES[name]) for name in sorted(SCALES)]
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode()).hexdigest()
