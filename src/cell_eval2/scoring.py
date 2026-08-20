"""Per-metric scoring policy: what a metric's best value is, and whether it is scored.

Separates three facts that ``MetricSpec.best_value`` used to carry in one token: the
normalization anchor (mathematics), enrolment in ``avg_score`` (policy), and the absence
of a constant anchor (a missing formula). See
``internal:docs/superpowers/specs/2026-08-01-metric-scoring-policy-design.md``.

Deliberately free of any catalog import so the engine stays pure and both ``score.py``
and ``catalog.py`` can depend on it.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

DEFAULT_PENALTY_EXPONENT = 2.0   # p
DEFAULT_PENALTY_CAP = 6.0        # C. Was 10.0 through v0.12.0; #276 part C retuned it to 6
                                 # so the six-member vcc2026 average floored at exactly -1.0.
                                 # ⚠️ HISTORICAL as of 2026-08-17: that floor is now carried by
                                 # ERROR_LINEAR's own clamp_low, and no default vcc2026 score
                                 # DEPENDS NUMERICALLY on C -- score_one still resolves it and
                                 # competition_payload still records it; it is inert, not
                                 # unread. What C still governs is the ERROR class
                                 # (expr_mae -- in the frozen 2025 `vcc` profile -- expr_mse,
                                 # delta_mae, delta_mse), a call-time penalty="boxcox"
                                 # override, and the resolved knobs competition_payload
                                 # records. Retuning it moves those, not the vcc2026 default.


@dataclass(frozen=True)
class Scoring:
    """How one metric is turned into a baseline-relative score.

    ``scored`` is policy and carries no mathematical content; ``direction`` and ``anchor``
    are intrinsic to the metric and are recorded even when it is not scored.
    ``anchor`` is the value that SCORES 1.0 -- the metric's perfect value on the baseline
    scale (0.0 for errors, 1.0 for bounded rank metrics), and the measured replicate on the
    replicate scale (#276 part C). ``None`` when perfection is undefined or data-dependent,
    which selects the ratio-to-baseline denominator (spec 2).
    """

    scored: bool = False
    direction: Literal["lower", "higher"] | None = None
    anchor: float | None = None
    penalty: Literal["none", "boxcox"] = "none"
    penalty_exponent: float | None = None     # None -> caller's default
    penalty_cap: float | None = None          # None -> caller's default
    clamp_low: float | None = 0.0             # None + boxcox -> -penalty_cap; None + none -> -inf
    clamp_high: float | None = None
    allow_negative_baseline: bool = False     # anchorless only; D = |b| instead of b (spec 2.2)
    metric_min: float | None = None           # the metric's structural worst value (spec 2.4b)

    def __post_init__(self) -> None:
        # This loop runs FIRST: the >0 and clamp-ordering guards below would otherwise fire
        # on a NaN/inf input and raise a message that does not mention finiteness, which the
        # tests assert on.
        for fname in ("anchor", "clamp_low", "clamp_high", "penalty_exponent",
                      "penalty_cap", "metric_min"):
            v = getattr(self, fname)
            if v is not None and not math.isfinite(v):
                raise ValueError(f"{fname} must be finite when set, got {v!r}")
        if self.direction not in (None, "lower", "higher"):
            raise ValueError(f"direction must be 'lower', 'higher' or None, got {self.direction!r}")
        if self.penalty not in ("none", "boxcox"):
            raise ValueError(f"penalty must be 'none' or 'boxcox', got {self.penalty!r}")
        if self.penalty_exponent is not None and self.penalty_exponent <= 0:
            raise ValueError(f"penalty_exponent must be > 0, got {self.penalty_exponent!r}")
        if self.penalty_cap is not None and self.penalty_cap <= 0:
            raise ValueError(f"penalty_cap must be > 0, got {self.penalty_cap!r}")
        # Compare the EFFECTIVE floor, not the declared one: `clamp_low=None` + boxcox
        # derives the floor from penalty_cap (spec 2.4), so checking only the declared
        # fields would accept an effective window of [-10, -20].
        eff_low = self.effective_clamp_low()
        if self.clamp_high is not None and eff_low > self.clamp_high:
            raise ValueError(
                f"clamp_low ({eff_low}, effective) must not exceed clamp_high ({self.clamp_high})"
            )
        # The Box-Cox tail is defined on a ratio, and only the lower/anchored class has one
        # (spec 2.3) -- for higher/anchored `1-s` is (a-u)/(a-b), and for anchorless it is not
        # a ratio at all.
        if self.penalty == "boxcox" and (self.direction != "lower" or self.anchor is None):
            raise ValueError(
                "penalty='boxcox' requires direction='lower' and a non-None anchor "
                f"(got direction={self.direction!r}, anchor={self.anchor!r})"
            )
        # `metric_min` needs an ANCHOR, and NOT because an anchor bounds the sentinel -- it
        # does not, which is why `is_degenerate` checks the sentinel at runtime. The reason is
        # that `metric_min` means "the worse end of this metric's range", and only an anchored
        # policy has a stated perfect end for it to be worse THAN: the rule below, and the
        # sentinel's meaning, are both undefined without one.
        if self.metric_min is not None and self.anchor is None:
            raise ValueError(
                "metric_min requires a non-None anchor: without one the score is normalized "
                "by the baseline alone and the sentinel is not bounded (spec 2.4b)"
            )
        # `metric_min` is the metric's WORST attainable value, so it must sit on the worse
        # side of perfection. A wrong-side value would put the unusable-submission sentinel
        # ABOVE the anchor -- i.e. a NaN scoring better than a perfect prediction.
        # STRICT, not `<=`/`>=`. Equality is the pathological case, not a harmless edge: the
        # sentinel is the score `metric_min` earns, so `metric_min == anchor` makes a MISSING
        # or NaN submission score exactly 1.0 -- indistinguishable from a perfect prediction,
        # which is the worst answer this field can give. Verified before tightening: under the
        # old `>` check, `Scoring(direction="higher", anchor=1.0, clamp_low=None,
        # metric_min=1.0)` was accepted and `score_one(None, 0.4, ...)` returned 1.0.
        # (Copilot round 4.) A metric whose worst value IS its perfect value is constant, and
        # nothing about scoring a constant is meaningful anyway.
        if self.metric_min is not None and self.anchor is not None:
            if self.direction == "higher" and self.metric_min >= self.anchor:
                raise ValueError(
                    f"metric_min ({self.metric_min}) must be strictly below anchor "
                    f"({self.anchor}) for a higher-is-better metric; at equality a missing "
                    "value would score as a perfect prediction"
                )
            if self.direction == "lower" and self.metric_min <= self.anchor:
                raise ValueError(
                    f"metric_min ({self.metric_min}) must be strictly above anchor "
                    f"({self.anchor}) for a lower-is-better metric; at equality a missing "
                    "value would score as a perfect prediction"
                )
        # With an anchor the baseline's side is already checkable, so the flag is meaningless.
        if self.allow_negative_baseline and self.anchor is not None:
            raise ValueError("allow_negative_baseline requires anchor=None (spec 2.2)")
        if self.scored:
            if self.direction is None:
                raise ValueError("a scored metric must declare a direction")
            # A non-finite value takes clamp_low (spec 2.4); an infinite floor would let one
            # NaN submission drag avg_score to -inf. Disabling clamping is a CALL-TIME
            # override only, never a catalog state.
            if (self.clamp_low is None and self.penalty != "boxcox"
                    and self.metric_min is None):
                raise ValueError(
                    "a scored metric needs a finite clamp_low (set it, use penalty='boxcox' "
                    "so penalty_cap supplies it, or declare metric_min so the metric's own "
                    "structural worst value supplies the score an unusable submission takes)"
                )

    def effective_clamp_low(self, *, cap: float | None = None,
                            penalty: str | None = None) -> float:
        """The floor actually applied, resolving ``clamp_low=None`` (spec 2.4).

        ``cap`` and ``penalty`` are the values that will ACTUALLY be used at the call,
        already resolved by spec 3.2's four-level order; ``None`` means "self-resolve",
        which is what ``__post_init__`` wants. Re-deriving them from ``self`` here would
        resolve them a SECOND time with the opposite precedence -- a global ``penalty_cap``
        would then reach the tail but not the floor.
        """
        if self.clamp_low is not None:
            return float(self.clamp_low)
        pen = self.penalty if penalty is None else penalty
        # The cap supplies a floor only for boxcox, so this must follow the RESOLVED
        # penalty: overriding penalty="none" takes the floor away with it.
        if pen == "boxcox":
            eff = cap if cap is not None else (
                DEFAULT_PENALTY_CAP if self.penalty_cap is None else self.penalty_cap)
            return -float(eff)
        return float("-inf")


# The classes that exist today, plus the diagnostic default.
ERROR = Scoring(scored=True, direction="lower", anchor=0.0, penalty="boxcox", clamp_low=None)
#: ``ERROR`` with the Box-Cox tail straightened into a line (Alex, 2026-08-17). Above the
#: baseline nothing moves -- both are the same anchored ``1 - r`` -- and the floor is the same
#: -6.0. What changes is the shape between them: the quadratic tail saturated at
#: ``r = sqrt(1 + 2C) = 3.606`` and the line reaches the floor at ``r = 7``, so a submission
#: between those two ratios is now ranked instead of pinned at the floor. It is also the
#: shape ``scales.low-random_high-1_v10`` already scores this metric with, so the two now agree
#: on SHAPE and on the perfection ANCHOR (0.0 for both). What differs is the base -- a constant
#: 1.0 there, a measured baseline here -- and the clamps: [-1.0, 1.0] there, [-6.0, inf) here.
#:
#: The floor is WRITTEN OUT rather than derived from ``DEFAULT_PENALTY_CAP``. The number is
#: carried over from #276 part C and its reason survives the change -- 6 is what floors the
#: six-member ``vcc2026`` average at exactly -1.0 -- but a policy that no longer has a penalty
#: must not track a penalty knob: a later retune of ``C`` would otherwise move this floor
#: silently, and under the SHIPPED policies no ``vcc2026`` member's score depends on ``C`` any
#: more (``expr_mse_unbiased_capped_norm`` keeps ``penalty="boxcox"`` but its declared
#: ``clamp_low=0.0`` clips before the tail can show through). Two caveats on that sentence.
#: ``competition_payload`` still RECORDS the resolved ``penalty_cap`` per member, so the rule
#: digest does move with ``C`` -- provenance, not arithmetic. And a CALL-TIME
#: ``penalty="boxcox"`` re-enables the tail on this policy rather than being refused (lower +
#: a non-None anchor is all the precondition asks), which makes ``C`` live again for the
#: finite branch while the declared ``clamp_low`` still catches a non-finite one.
#:
#: Carried by the ``de_*_lfc_nmae`` family ONLY. ``ERROR`` itself is unchanged and still
#: carries ``expr_mae``/``expr_mse``/``delta_mae``/``delta_mse``; ``expr_mae`` is in the frozen
#: 2025 ``vcc`` profile, whose scores must not move.
ERROR_LINEAR = Scoring(scored=True, direction="lower", anchor=0.0,
                       penalty="none", clamp_low=-6.0)
BOUNDED = Scoring(scored=True, direction="higher", anchor=1.0)
#: `BOUNDED` with the flat clip-at-0 removed. For a [0, 1] metric the score `(u - b)/(a - b)`
#: is already bounded BELOW by `-b/(a - b)` for every valid `u`, so the floor buys no
#: protection from the metric -- only from a MISSING value, which `metric_min` now covers
#: exactly. Carried by the four bounded `vcc2026` members, where the clip was truncating real
#: below-comparator signal: measured on the three official val bundles the truncated depth ran
#: to -2.68 (`direction_fidelity_yield_raw`, val B) on the replicate scale. (Measured on the
#: PRE-#172 `-r1` bundles; the depths shift when #317's wave rebuilds them, the mechanism does
#: not.)
#: NOT a drop-in for every bounded metric -- `BOUNDED` is shared by 60+ catalog entries,
#: including the frozen 2025 `vcc` profile, whose scores must not move.
#:
#: ⚠️ WHAT THIS TRADES. The clip at 0 caught ANY out-of-range value; `metric_min` catches only
#: the declared one. A `u` BELOW `metric_min` now scores unclamped -- at b=0.5, anchor=0.9 a
#: hypothetical u = -1.0 scores -3.75 where the clip returned 0.0. That is sound only because
#: all four members are structurally [0, 1]: `pds_cosine` is `1 - rank/D` with a 0-based
#: midrank and `rank_denominator="n-1"`, `direction_fidelity_yield_raw` is
#: `k/max(n_pred, N_conf)`, `direction_reach_raw` is `k*/N_conf`, `sig_jaccard` is
#: `|R n P|/|R u P|` -- and `aggregate_metrics` means over per-perturbation values in that same
#: range. A negative aggregate is therefore a bug UPSTREAM of scoring, and this policy makes it
#: loud in `avg_score` rather than absorbing it into a 0. Deliberate: absorbing it is what let
#: a wrong number decide a ranking silently. Do not carry this policy on a metric whose range
#: you have not checked.
BOUNDED_UNFLOORED = Scoring(scored=True, direction="higher", anchor=1.0,
                            clamp_low=None, metric_min=0.0)
DIAG = Scoring(scored=False)


def denominator(base: float, policy: Scoring) -> float:
    """``D`` from spec 2 -- the scale the baseline-relative gap is measured in.

    With an anchor the baseline's sign is checkable (it must lie on the known side of
    perfection), so ``D`` is signed and a wrong-side baseline yields ``D <= 0`` and fails
    loud. Without an anchor there is no such check, and "no anchor" does not by itself
    license a negative baseline: only a policy that also sets ``allow_negative_baseline``
    gets ``D = |b|``, which keeps the score increasing in ``user`` for every ``base != 0``
    instead of inverting the ordering. For every other anchorless policy ``D = b``, so a
    negative baseline stays degenerate (spec 2.2).
    """
    if policy.anchor is None:
        # "No anchor" does not by itself license a negative baseline: of the 12 anchorless
        # catalog entries only direction_yield is signed, and abs() on the other ten would
        # silently score corrupt input (spec 2.2).
        return abs(float(base)) if policy.allow_negative_baseline else float(base)
    if policy.direction == "higher":
        return float(policy.anchor) - float(base)
    return float(base) - float(policy.anchor)


def is_degenerate(base: float | None, policy: Scoring, *,
                  clamp_low: float | None = None,
                  clamp_high: float | None = None,
                  penalty_exponent: float | None = None,
                  penalty_cap: float | None = None,
                  penalty: str | None = None) -> bool:
    """True when ``base`` is missing/non-finite, or ``D`` is not a FINITE POSITIVE number.

    Finiteness is a condition on ``D``, not only on ``base``: ``anchor - base`` can overflow
    for two finite values, and an infinite denominator divides the gap to a plausible ``0.0``.

    Reproduces ``base <= 0`` for an error metric (anchor 0 -> D = base) EXACTLY, and
    deliberately EXTENDS the bounded check from today's ``base == 1`` to ``base >= 1``
    (anchor 1 -> D = 1 - base): a baseline at or past perfection is corrupt either way
    (spec 6). Also covers the anchorless cases -- ``base == 0`` when signed, ``base <= 0``
    when non-negative.

    The keyword arguments are the CALL-TIME overrides that will be handed to ``score_one``,
    and they matter because degeneracy is no longer a property of the denominator alone: an
    UNFLOORED policy also needs a representable sentinel, and whether it is unfloored depends
    on the effective ``clamp_low``/``penalty``. Omit them and the policy self-resolves, which
    is right for every caller that also calls ``score_one`` without overrides. ``score_metrics``
    passes what it is about to pass (codex round 2, finding 1). They are VALIDATED here too,
    by the same code that validates them in ``score_one`` -- see ``_resolve_call``.
    """
    # BEFORE the base checks, exactly as `score_one` validates before doing anything: a call
    # with an invalid override is a bad call whatever the base is, and short-circuiting on
    # `base is None` would let it return a boolean instead of raising.
    p, cap, pen, low, _high = _resolve_call(
        policy, penalty_exponent=penalty_exponent, penalty_cap=penalty_cap,
        clamp_low=clamp_low, clamp_high=clamp_high, penalty=penalty)
    if base is None:
        return True
    b = float(base)
    if not _isfinite(b):
        return True
    d = denominator(b, policy)
    # A FINITE base can still produce a non-finite D: `anchor - base` overflows for a large
    # anchor and an opposite-signed base. `D = inf` passes `<= 0` and then divides the gap to
    # a plausible 0.0 -- a wrong number with no signal, which is the one outcome this
    # predicate exists to prevent. Checked before the sign tests so it covers both classes.
    if not _isfinite(d):
        return True
    # For the SIGNED anchorless case the test is == 0: abs() is never negative, so `<= 0`
    # would carry a dead branch that reads as if it guarded something (spec 6).
    if policy.anchor is None and policy.allow_negative_baseline:
        return d == 0.0
    if d <= 0.0:
        return True
    # An UNFLOORED policy has no finite floor to absorb an overflow, so `D > 0` is not
    # sufficient on its own: `metric_min` is independent of both ends, and a distant enough
    # one divided by a small enough `D` is `-inf`. One such member carries `avg_score` to
    # `-inf` -- the failure the finite-floor rule exists to prevent, reached through the
    # DENOMINATOR rather than through the policy. Rejecting here routes it into the existing
    # decisive-raise / warn-and-exclude machinery instead of into a silent `-inf`.
    # A FLOORED policy is untouched: its clamp catches the overflow, which is what it is for,
    # and `effective_clamp_low` is finite so `score_one` never consults `metric_min` at all.
    if not _isfinite(low):
        sentinel = _sentinel_score(b, policy, p=p, cap=cap, pen=pen)
        if sentinel is not None and not _isfinite(sentinel):
            return True
    return False


def _resolve_call(policy: Scoring, *, penalty_exponent, penalty_cap, clamp_low, clamp_high,
                  penalty) -> tuple[float, float, str, float, float]:
    """Validate the call-time overrides and resolve every knob -> (p, cap, pen, low, high).

    Spec 3.1a's validation and spec 3.2's resolution order -- call-time argument, then policy
    field, then built-in default -- in ONE place, shared by ``is_degenerate`` and
    ``score_one``.

    Sharing is not cosmetic. The predicate depends on the effective floor and penalty (an
    unfloored policy's sentinel has to be representable), so a predicate that self-resolves
    while the scorer honours an override disagrees in both directions -- rejecting a scale a
    call-time ``clamp_low=0.0`` makes safe, accepting one a ``clamp_low=-inf`` makes unusable
    (codex round 2). And sharing RESOLUTION alone was still not enough: the predicate then
    ran the scoring arithmetic on knobs the scorer REFUSES -- with ``penalty="boxcox",
    penalty_exponent=0.0`` the tail computes ``-(r**0 - 1)/0`` and ``is_degenerate`` raised a
    bare ``ZeroDivisionError`` where ``score_one`` raises a ``ValueError`` naming the bad
    argument; ``penalty="garbage"`` and a NaN ``clamp_low`` likewise returned a boolean
    instead of raising (codex round 3).
    """
    # Only the clamp-disabling infinities are allowed; NaN or a wrong-signed infinity would
    # poison avg_score exactly as a bad catalog value would.
    if clamp_low is not None and (clamp_low != clamp_low or clamp_low == float("inf")):
        raise ValueError(f"clamp_low override must be finite or -inf, got {clamp_low!r}")
    if clamp_high is not None and (clamp_high != clamp_high or clamp_high == float("-inf")):
        raise ValueError(f"clamp_high override must be finite or +inf, got {clamp_high!r}")
    # `inf`/`nan` slip past a bare `<= 0` guard (that is the pre-existing hole in
    # score_metrics); an infinite cap yields an infinite floor, a NaN cap a NaN one.
    for _n, _v in (("penalty_exponent", penalty_exponent), ("penalty_cap", penalty_cap)):
        if _v is not None and (not _isfinite(float(_v)) or float(_v) <= 0):
            raise ValueError(f"{_n} override must be finite and > 0, got {_v!r}")

    def _pick(arg, field, default):
        if arg is not None:
            return arg
        return default if field is None else field

    p = float(_pick(penalty_exponent, policy.penalty_exponent, DEFAULT_PENALTY_EXPONENT))
    cap = float(_pick(penalty_cap, policy.penalty_cap, DEFAULT_PENALTY_CAP))
    pen = policy.penalty if penalty is None else penalty
    # A call-time `penalty` must satisfy the SAME precondition __post_init__ enforces, and an
    # unrecognised string must raise rather than fall through to linear behaviour: silently
    # ignoring an override the caller asked for is worse than refusing it.
    if pen not in ("none", "boxcox"):
        raise ValueError(f"penalty override must be 'none' or 'boxcox', got {pen!r}")
    if pen == "boxcox" and (policy.direction != "lower" or policy.anchor is None):
        raise ValueError(
            "penalty='boxcox' requires direction='lower' and a non-None anchor "
            f"(policy has direction={policy.direction!r}, anchor={policy.anchor!r})"
        )

    # Pass the RESOLVED cap and penalty, never the policy's own -- see effective_clamp_low.
    low = (policy.effective_clamp_low(cap=cap, penalty=pen)
           if clamp_low is None else float(clamp_low))
    # Overriding `penalty` away from boxcox removes the floor the cap was supplying, which
    # would leave a SCORED metric unfloored -- exactly what spec 3.1 forbids in the catalog,
    # and one NaN submission would then drag avg_score to -inf. Refuse instead of inheriting
    # -inf silently; the caller can say what floor they want.
    if (policy.scored and clamp_low is None and not _isfinite(low)
            and policy.metric_min is None):
        raise ValueError(
            f"penalty={pen!r} leaves this scored policy without a finite floor: its "
            "clamp_low is None, only penalty='boxcox' derives one from penalty_cap, and it "
            "declares no metric_min. Pass clamp_low as well (clamp_low=0.0 reproduces the "
            "frozen clip-at-0)."
        )
    if clamp_high is not None:
        high = float(clamp_high)
    else:
        high = float("inf") if policy.clamp_high is None else float(policy.clamp_high)
    # Order the EFFECTIVE pair, not the two supplied arguments: an override checked only
    # against its sibling lets score_one(..., BOUNDED, clamp_high=-1.0) resolve to
    # low=0.0, high=-1.0 and return -1.0, below its own floor.
    if low > high:
        raise ValueError(f"effective clamp_low {low!r} exceeds effective clamp_high {high!r}")
    return p, cap, pen, low, high


def _raw_score(u: float, b: float, policy: Scoring, *, p: float, cap: float,
               pen: str) -> float:
    """The unclamped score: spec 2's linear core plus the Box-Cox tail. No clamping.

    ONE definition, module level, called by ``score_one`` and by ``_sentinel_score``. It was
    briefly two -- a nested closure for the score and a linear-only helper for the sentinel
    -- which agreed by coincidence rather than by construction and left the "cannot drift"
    claim unimplemented (codex round 2, finding 2). Notably the tail MATTERS here: a boxcox
    policy whose clamping is disabled at call time still saturates at ``-cap``, so a
    linear-only sentinel would call that scale degenerate when the scorer handles it fine.
    """
    # Two evaluation orders, not one: float division does not reassociate, and each
    # anchored class must match its frozen kernel bit-for-bit (spec 2). `(b-u)/b` drifts
    # from `1 - u/b` by 1 ulp on 20 of 56 sampled pairs, which breaks the v1 parity gate.
    r = None
    if policy.anchor is not None and policy.direction == "lower":
        r = (u - policy.anchor) / (b - policy.anchor)   # anchor 0 -> exactly u/b
        s = 1.0 - r                                     # == compat._norm_by_zero
    elif policy.anchor is not None:
        s = (u - b) / (policy.anchor - b)               # anchor 1 -> == compat._norm_by_one
    else:
        gap = (u - b) if policy.direction == "higher" else (b - u)
        s = gap / denominator(b, policy)

    # Both the catalog path (__post_init__) and the call-time path guarantee that boxcox
    # implies lower+anchored, so `r is not None` here; the guard is kept as a structural one
    # -- the tail is defined only where a ratio was actually carried.
    if s < 0.0 and pen == "boxcox" and r is not None:
        # Uses the CARRIED r, never 1 - s. On the shipped knobs (p=2, C=6) reconstruction
        # happens to agree bitwise -- the tail saturates at r >= sqrt(13) and 1-(1-r) == r
        # for every r in [1, 2^53] -- but r = 1-s holds algebraically only for this class,
        # and a per-metric penalty_cap large enough to reach r > 2^53 is expressible in the
        # catalog, where the two DO differ (spec 2.3).
        # The cap saturates the TAIL (spec 2.3's `max(-C, ...)`), and is applied here rather
        # than left to the clamp. The two coincide only while clamp_low == -cap, which is
        # ERROR's default -- with any lower floor (a catalog `clamp_low=-100`, or a global
        # `clamp_low=float("-inf")`) relying on the clamp lets the tail run away: u/b = 100
        # would score -4999.5 instead of -6, and one such submission drags avg_score with it.
        # `-(r**p - 1)/p`, NOT `-expm1(p*log(r))/p`. The expm1 form is the numerically stable
        # one and a reviewer will propose it again -- it is mathematically identical and does
        # avoid cancellation as p -> 0 (measured at p=1e-12: -0.405453 vs the true
        # -0.4054651). It is declined because it MOVES SHIPPED NUMBERS: at the shipped p=2.0
        # the two forms differ on 116,870 of 200,000 random r in [1, 5), max |delta| 3.6e-15.
        # Buying accuracy in a regime nothing uses (penalty_exponent defaults to 2.0) by
        # perturbing every penalty score is the wrong trade for a scorer whose output is a
        # competition ranking. Revisit only alongside a decision to re-baseline.
        #
        # The `max` argument order is also deliberate, though it is moot here:
        # `max(-cap, x)` returns -cap when x is NaN, `max(x, -cap)` returns NaN. x cannot be
        # NaN in this branch (r > 1 and p > 0, both finite; overflow is caught below), but if
        # it ever could, -cap is the answer that matches the frozen kernel -- NaN would fall
        # through to the clamp and return `low`, which is only the same value while
        # clamp_low == -cap.
        try:
            s = max(-cap, -(r ** p - 1.0) / p)
        except OverflowError:
            s = -cap               # far past the cap, exactly as the frozen kernel returned
    return s


def _sentinel_score(base: float, policy: Scoring, *, p: float, cap: float,
                    pen: str) -> float | None:
    """The score ``policy.metric_min`` earns against ``base``, or ``None`` if undeclared.

    This is what an UNFLOORED policy gives a missing/non-finite submission, and it is NOT
    guaranteed finite. An earlier revision claimed it was, arguing that two distinct floats
    differ by at least one ulp of the larger so the ratio is bounded by 2^54. That argument
    is WRONG: it bounds ``a - b`` from below but says nothing about ``metric_min - b``, which
    is a third, independent value. Codex counterexample, verified --
    ``direction="higher"``, ``anchor=5e-324``, ``base=-5e-324``,
    ``metric_min=-sys.float_info.max`` is accepted by ``Scoring`` AND by the sign checks in
    ``is_degenerate``, and its sentinel is ``-inf``. Hence the check, in both callers.

    Goes through ``_raw_score``, the SAME function ``score_one`` uses, with the SAME
    resolved knobs -- so the tail is applied wherever the scorer would apply it and the two
    cannot disagree about which scales are usable.
    """
    if policy.metric_min is None:
        return None
    return _raw_score(float(policy.metric_min), float(base), policy, p=p, cap=cap, pen=pen)


def score_one(
    user: float | None,
    base: float,
    policy: Scoring,
    *,
    penalty_exponent: float | None = None,
    penalty_cap: float | None = None,
    clamp_low: float | None = None,
    clamp_high: float | None = None,
    penalty: str | None = None,
) -> float:
    """One metric's baseline-relative score (spec 2). ``base`` must be non-degenerate.

    ``clamp_low`` / ``clamp_high`` / ``penalty`` are CALL-TIME overrides of the policy's own
    values. They are arguments rather than a rebuilt ``Scoring`` on purpose:
    ``dataclasses.replace()`` re-runs ``__post_init__``, so
    ``replace(policy, clamp_low=float("-inf"))`` would be rejected by the finite-floor rule --
    verified, it raises. The catalog object keeps its invariant; the override lives in the
    call (spec 3.2).

    A non-finite or missing ``user`` is a degenerate *model* output and takes ``clamp_low``:
    one rule, resolved per policy, with no special case (spec 2.4). Three of the resolutions,
    as examples rather than an enumeration -- any policy may declare its own floor -- are
    ``0.0`` for the bounded class, ``-penalty_cap`` for the Box-Cox error class
    (``clamp_low=None``, so the cap supplies the floor), and the DECLARED ``-6.0`` for
    ``ERROR_LINEAR``. Note the last does not follow the cap: under a call-time
    ``penalty="boxcox", penalty_cap=2`` a bad finite value saturates at ``-2`` while a
    non-finite one still takes ``-6``.
    """
    # Validation (spec 3.1a) and resolution (spec 3.2) both live in `_resolve_call`, which
    # `is_degenerate` also uses -- so the predicate cannot accept a call this function
    # refuses, nor run the arithmetic on a knob this function would have rejected.
    p, cap, pen, low, high = _resolve_call(
        policy, penalty_exponent=penalty_exponent, penalty_cap=penalty_cap,
        clamp_low=clamp_low, clamp_high=clamp_high, penalty=penalty)

    b = float(base)

    def _core(u: float) -> float:
        """This call's unclamped score, with the resolved knobs bound. A thin binding of the
        module-level `_raw_score` -- the arithmetic and every comment on it live there, so
        the sentinel path and the scoring path cannot drift apart."""
        return _raw_score(u, b, policy, p=p, cap=cap, pen=pen)

    def _unusable() -> float:
        """The score an UNUSABLE submission takes: missing, non-finite, or one whose
        normalization overflowed (spec 2.4).

        Normally that is ``clamp_low`` -- unchanged, and bit-identical for every floored
        policy. A policy that is deliberately UNFLOORED (``clamp_low=None`` with
        ``penalty="none"``) has an effective floor of ``-inf``, and returning it would let
        ONE missing metric drag ``avg_score`` to ``-inf``. Such a policy must declare
        ``metric_min`` (``__post_init__`` enforces it), and the sentinel is then the score
        the metric's own structural worst value earns -- exactly the floor the unfloored
        policy has, computed rather than declared because it depends on ``base`` and on the
        anchor in play (the replicate scale moves the anchor, #276 part C).

        A FINITE `low` short-circuits, so every floored policy keeps today's answer exactly
        and the `metric_min` branch is reached only where there is no floor -- including when
        the caller disabled clamping with `clamp_low=float("-inf")` on a policy that declares
        `metric_min`. That is deliberate: such a caller asked for no CLIP, not for a poisoned
        average.
        """
        if _isfinite(low) or policy.metric_min is None:
            return _clip(float("-inf"), low, high)
        sentinel = _core(float(policy.metric_min))
        if not _isfinite(sentinel):
            # `is_degenerate` rejects this scale, and every in-tree caller checks it first --
            # but `score_one` is public and does not re-check. RAISE rather than return
            # `-inf` from the one branch whose entire purpose is to avoid `-inf`: a caller
            # who skipped the guard gets a message naming the three values, not a poisoned
            # average they will find weeks later.
            raise ValueError(
                f"unusable scale for an unfloored policy: metric_min={policy.metric_min!r} "
                f"against base={base!r} and anchor={policy.anchor!r} gives a "
                f"non-finite sentinel ({sentinel!r}), so a missing value has no "
                "representable score. Check is_degenerate() before calling score_one()."
            )
        # ⚠️ A DEGENERATE base now raises `ZeroDivisionError` for a non-finite `user` as it
        # already did for a finite one, where this branch previously returned `low` without
        # dividing. Consistent rather than silently different, and it reaches only the
        # unfloored policies. It is not a live hazard because every scoring caller rejects or
        # excludes a degenerate base BEFORE `score_one` -- `score.score_metrics`,
        # `scales.build_scale` (at import) and `tools/metricval` all call `is_degenerate`
        # first. Note this is NOT the same claim as "every unfloored entry is decisive":
        # the three `de_deseq2_*` siblings are deliberately NOT decisive, so for them a
        # degenerate base warns and excludes where the four wilcoxon/pds entries raise.
        # Decisiveness picks WHICH of those two, not whether the guard runs.
        return _clip(sentinel, low, high)

    if user is None or not _isfinite(float(user)):
        return _unusable()

    s = _core(float(user))
    # A non-finite COMPUTED score takes the same sentinel as a non-finite `user`, not just a
    # non-finite `user`: a finite u can overflow the normalization (b just under 1,
    # u = float_info.max -> +inf), and today's isfinite fallback turns that into 0.0.
    # Clipping +inf into [0, inf) would return inf and poison avg_score (spec 2.4).
    if not _isfinite(s):
        return _unusable()
    return _clip(s, low, high)


def _isfinite(x: float) -> bool:
    """Thin delegate, kept as a named function so the call sites read the same.

    Was a hand-rolled `x == x and x not in (inf, -inf)`. Verified equivalent to
    `math.isfinite` over the edge cases this engine actually sees -- +-0.0, subnormals,
    +-inf, NaN, 2**53, +-1e308 -- and over 2,000 random values, with zero mismatches, and
    measured 4.15x faster over 200k calls. Every call site passes an already-`float()`-cast
    value, so `math.isfinite`'s stricter typing is not a behaviour change here."""
    return math.isfinite(x)


def _clip(x: float, low: float, high: float) -> float:
    """NaN-safe, and returns the BOUND on equality so a -0.0 cannot survive where
    today's ``max(0.0, score)`` yields +0.0."""
    if not _isfinite(x):
        return low                      # NaN fails every comparison; never let it through
    if x <= low:
        return low
    return high if x >= high else x
