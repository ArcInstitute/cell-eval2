import math
import sys

import pytest

from cell_eval2.scoring import BOUNDED, DIAG, ERROR, Scoring


def test_presets_match_todays_two_scored_classes():
    assert ERROR.scored and ERROR.direction == "lower" and ERROR.anchor == 0.0
    assert ERROR.penalty == "boxcox" and ERROR.clamp_low is None
    assert BOUNDED.scored and BOUNDED.direction == "higher" and BOUNDED.anchor == 1.0
    assert BOUNDED.penalty == "none" and BOUNDED.clamp_low == 0.0
    assert DIAG.scored is False and DIAG.direction is None and DIAG.anchor is None


def test_scored_requires_a_direction():
    with pytest.raises(ValueError, match="direction"):
        Scoring(scored=True, direction=None, anchor=1.0)


def test_scored_requires_a_finite_effective_clamp_low():
    # penalty="none" + clamp_low=None means no floor -> one NaN submission would drag
    # avg_score to -inf.
    with pytest.raises(ValueError, match="clamp_low"):
        Scoring(scored=True, direction="higher", anchor=1.0, penalty="none", clamp_low=None)


def test_boxcox_supplies_clamp_low_from_penalty_cap():
    # ERROR carries clamp_low=None but penalty="boxcox", so the floor comes from the cap.
    assert ERROR.effective_clamp_low(cap=10.0) == -10.0
    assert BOUNDED.effective_clamp_low(cap=10.0) == 0.0
    # A pinned catalog cap must NOT win over the cap the caller resolved (spec 3.2): the
    # tail and the floor have to see the same number, or a global penalty_cap reaches one
    # and not the other.
    pinned = Scoring(scored=True, direction="lower", anchor=0.0, penalty="boxcox",
                     clamp_low=None, penalty_cap=8.0)
    assert pinned.effective_clamp_low() == -8.0            # self-resolved
    assert pinned.effective_clamp_low(cap=3.0) == -3.0     # caller-resolved wins
    # The floor follows the RESOLVED penalty, not the catalog's: the cap only supplies a
    # floor for boxcox.
    assert ERROR.effective_clamp_low(cap=10.0, penalty="none") == float("-inf")


def test_clamp_low_must_not_exceed_clamp_high():
    with pytest.raises(ValueError, match="clamp_low"):
        Scoring(scored=True, direction="higher", anchor=1.0, clamp_low=1.0, clamp_high=0.0)


def test_clamp_ordering_is_checked_on_the_EFFECTIVE_floor():
    # clamp_low=None + boxcox derives the floor from penalty_cap (-10), so comparing only
    # the DECLARED fields would let this through with an effective window of [-10, -20].
    with pytest.raises(ValueError, match="clamp_low|clamp_high"):
        Scoring(scored=True, direction="lower", anchor=0.0, penalty="boxcox",
                clamp_low=None, penalty_cap=10.0, clamp_high=-20.0)


@pytest.mark.parametrize("kw", [{"penalty_exponent": 0.0}, {"penalty_cap": -1.0}])
def test_non_positive_knobs_rejected(kw):
    with pytest.raises(ValueError):
        Scoring(scored=True, direction="lower", anchor=0.0, **kw)


@pytest.mark.parametrize("kw", [
    {"direction": "higher", "anchor": 1.0, "penalty": "boxcox"},   # wrong direction
    {"direction": "lower", "anchor": None, "penalty": "boxcox"},   # no anchor -> no ratio
])
def test_boxcox_restricted_to_the_lower_anchored_class(kw):
    with pytest.raises(ValueError, match="boxcox"):
        Scoring(scored=False, **kw)


@pytest.mark.parametrize("field", ["anchor", "clamp_low", "clamp_high",
                                   "penalty_exponent", "penalty_cap"])
@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_non_finite_floats_rejected(field, bad):
    with pytest.raises(ValueError, match="finite"):
        Scoring(scored=False, direction="higher", **{field: bad})


def test_allow_negative_baseline_requires_no_anchor():
    with pytest.raises(ValueError, match="allow_negative_baseline"):
        Scoring(scored=False, direction="higher", anchor=1.0, allow_negative_baseline=True)


def test_diagnostic_may_record_direction_without_being_scored():
    # The whole point of splitting the token: a policy CAN say "higher is better" without
    # claiming enrolment. Deliberately a hypothetical policy rather than a catalog entry, so
    # it cannot go stale -- and it has now outlived one occupant. `expr_mse_unbiased_norm` (#198)
    # held this state alone (directional: lower, anchor 0; deliberately unscored) until it was
    # enrolled, at which point NO catalog entry occupies it. That is precisely why the
    # mechanism is pinned here on a directly-built policy: a catalog-sourced test would have
    # been deleted along with the last occupant, taking the guarantee with it.
    # `test_scoring_catalog` asserts the occupant set is empty; this test pins the mechanism.
    s = Scoring(scored=False, direction="higher", anchor=None)
    assert s.direction == "higher" and s.anchor is None and s.scored is False


def test_default_penalty_cap_is_six():
    """#276 part C retuned C from 10 to 6 so the six-member competition average floored at
    exactly -1.0.

    ⚠️ That reason is HISTORICAL as of 2026-08-17. `de_wilcoxon_lfc_nmae` moved to
    `ERROR_LINEAR`, whose declared `clamp_low=-6.0` now carries that floor, and no default
    `vcc2026` score DEPENDS NUMERICALLY on C -- `score_one` still resolves it and
    `competition_payload` still records it, so it is inert rather than unread. Retuning it would
    NOT move the competition number any more. Still pinned as its own assertion, for what C does still govern: the `ERROR` class
    (`expr_mae`, in the frozen 2025 `vcc` profile, plus `expr_mse`/`delta_mae`/`delta_mse`), a
    call-time `penalty="boxcox"` override, and the resolved knobs `competition_payload` records
    into the rule digest. Every other cap test derives from the constant, so a silent retune
    would move those with nothing failing.
    """
    from cell_eval2.scoring import DEFAULT_PENALTY_CAP

    assert DEFAULT_PENALTY_CAP == 6.0


def test_mse_capped_norm_is_bounded_to_the_unit_interval():
    """Alex 2026-08-13: this member -- and ONLY this member -- clamps to [0, 1]."""
    from cell_eval2.catalog import CATALOG

    sc = CATALOG["expr_mse_unbiased_capped_norm"].scoring
    assert (sc.clamp_low, sc.clamp_high) == (0.0, 1.0)
    for other in ("pds_cosine", "de_wilcoxon_lfc_nmae",
                  "de_wilcoxon_direction_fidelity_yield_raw",
                  "de_wilcoxon_direction_reach_raw", "de_wilcoxon_sig_jaccard"):
        assert CATALOG[other].scoring.clamp_high is None, (
            f"{other} must stay unclamped above 1: the ruling was MSE only"
        )


def test_an_explicit_penalty_cap_still_overrides_the_default():
    """The retune must move the DEFAULT, not the override layer."""
    from dataclasses import replace

    from cell_eval2.scoring import ERROR, score_one

    assert score_one(100.0, 1.0, ERROR) == -6.0                              # resolved
    assert score_one(100.0, 1.0, replace(ERROR, penalty_cap=10.0)) == -10.0  # catalog-level
    assert score_one(100.0, 1.0, ERROR, penalty_cap=10.0) == -10.0           # call-time


# --- the unfloored class (clip-at-0 removal for the four bounded vcc2026 members) ---------

def test_an_unfloored_scored_policy_needs_a_metric_min():
    """`clamp_low=None` + `penalty='none'` means no floor. That is now EXPRESSIBLE, but only
    with a declared structural worst value -- otherwise one missing metric drags avg_score to
    -inf, which is the whole reason the finite-floor rule existed."""
    with pytest.raises(ValueError, match="clamp_low"):
        Scoring(scored=True, direction="higher", anchor=1.0, penalty="none", clamp_low=None)
    ok = Scoring(scored=True, direction="higher", anchor=1.0, penalty="none",
                 clamp_low=None, metric_min=0.0)
    assert ok.effective_clamp_low() == float("-inf")     # genuinely unclipped


def test_metric_min_must_sit_on_the_worse_side_of_the_anchor():
    """A wrong-side value would put the unusable-submission sentinel ABOVE the anchor, i.e.
    a missing metric scoring better than a perfect prediction."""
    with pytest.raises(ValueError, match="metric_min"):
        Scoring(scored=True, direction="higher", anchor=1.0, clamp_low=None, metric_min=1.5)
    with pytest.raises(ValueError, match="metric_min"):
        Scoring(scored=True, direction="lower", anchor=0.0, clamp_low=0.0, metric_min=-0.5)
    with pytest.raises(ValueError, match="metric_min"):     # non-finite like every other knob
        Scoring(scored=True, direction="higher", anchor=1.0, clamp_low=0.0,
                metric_min=float("nan"))
    # EQUALITY is refused too, and it is the case that actually bites: the sentinel is the
    # score `metric_min` earns, so `metric_min == anchor` makes a missing value score exactly
    # 1.0 -- a perfect prediction. Verified against the pre-fix `>` check, which accepted this
    # and returned 1.0 from `score_one(None, 0.4, ...)`. (Copilot round 4.)
    with pytest.raises(ValueError, match="strictly below anchor"):
        Scoring(scored=True, direction="higher", anchor=1.0, penalty="none", clamp_low=None,
                metric_min=1.0)
    with pytest.raises(ValueError, match="strictly above anchor"):
        Scoring(scored=True, direction="lower", anchor=0.0, penalty="none", clamp_low=-6.0,
                metric_min=0.0)


def test_an_unfloored_score_is_not_clipped_at_zero():
    """The point of the change. `(u - b)/(a - b)` at the metric's worst value is -b/(a-b),
    and that number now reaches avg_score instead of 0.0."""
    from cell_eval2.scoring import BOUNDED, BOUNDED_UNFLOORED, score_one

    assert score_one(0.0, 0.5, BOUNDED) == 0.0                     # unchanged, still clipped
    assert score_one(0.0, 0.5, BOUNDED_UNFLOORED) == -1.0
    assert score_one(0.0, 0.8, BOUNDED_UNFLOORED) == pytest.approx(-4.0)
    # and the two agree everywhere at or above the comparator, which is most of the range
    for u in (0.5, 0.6, 0.75, 1.0):
        assert score_one(u, 0.5, BOUNDED) == score_one(u, 0.5, BOUNDED_UNFLOORED)


@pytest.mark.parametrize("bad", [None, float("nan"), float("inf"), float("-inf")])
def test_an_unusable_submission_takes_the_metric_min_score(bad):
    """NOT -inf, and not 0.0 either: a missing metric scores exactly what the worst possible
    submission scores. That keeps avg_score finite without reintroducing the clip."""
    from cell_eval2.scoring import BOUNDED_UNFLOORED, score_one

    assert score_one(bad, 0.5, BOUNDED_UNFLOORED) == score_one(0.0, 0.5, BOUNDED_UNFLOORED)
    assert score_one(bad, 0.5, BOUNDED_UNFLOORED) == -1.0


def test_the_unusable_sentinel_is_finite_on_every_non_degenerate_scale():
    """The guarantee the runtime guard actually provides: whatever `is_degenerate` calls
    USABLE has a finite sentinel. Not a property of the arithmetic -- an earlier revision
    claimed `(m - b)/(a - b)` was bounded by ~2^54 because two distinct floats differ by at
    least one ulp of the larger, and that is FALSE: it bounds `a - b` and says nothing about
    `metric_min - b`. See `test_a_scale_whose_sentinel_overflows_is_degenerate` for the
    counterexamples. This sweeps the worst scales the type admits and checks the pairing:
    usable implies finite."""
    from cell_eval2.scoring import is_degenerate, score_one

    extremes = [1e-320, 5e-324, 1e-300, 1e-8, 0.5, 1.0, 1e8, 1e300, 1.7e308]
    checked = 0
    for b in extremes:
        for a in extremes:
            pol = Scoring(scored=True, direction="higher", anchor=a,
                          clamp_low=None, metric_min=0.0)
            if is_degenerate(b, pol):
                continue
            for bad in (None, float("nan"), float("inf")):
                got = score_one(bad, b, pol)
                assert math.isfinite(got), f"sentinel blew up at base={b!r} anchor={a!r}"
            checked += 1
    assert checked >= 20, f"the sweep only exercised {checked} usable scales"


@pytest.mark.parametrize("direction,anchor,base,mmin", [
    # Codex round 1, finding 1 -- both verified to return -inf on the first implementation.
    # The bug was a wrong lemma: `a - b >= ulp(max(|a|,|b|))` bounds the DENOMINATOR, but
    # `metric_min` is a third, independent value and `metric_min - b` is unbounded.
    ("higher", 5e-324, -5e-324, -sys.float_info.max),
    ("lower", 0.0, 5e-324, sys.float_info.max),
])
def test_a_scale_whose_sentinel_overflows_is_degenerate(direction, anchor, base, mmin):
    """An unfloored policy has no clamp left to absorb the overflow, so `-inf` would reach
    `avg_score` -- the exact failure the finite-floor rule exists to prevent, arriving through
    the denominator instead of through the policy. `is_degenerate` rejects it up front, and
    `score_one` refuses rather than returning `-inf` for a caller who skipped that check."""
    from cell_eval2.scoring import is_degenerate, score_one

    pol = Scoring(scored=True, direction=direction, anchor=anchor, penalty="none",
                  clamp_low=None, metric_min=mmin)
    assert is_degenerate(base, pol) is True
    with pytest.raises(ValueError, match="non-finite sentinel"):
        score_one(None, base, pol)


def test_is_degenerate_honours_the_knobs_score_one_will_be_called_with():
    """Codex round 2, finding 1. Degeneracy stopped being a property of the DENOMINATOR
    alone once an unfloored policy also needed a representable sentinel -- and whether a
    policy is unfloored depends on the effective `clamp_low`/`penalty`, which are call-time
    arguments. A predicate that self-resolves while the scorer honours an override disagrees
    in BOTH directions, so the knobs travel with the question.

    All three cases below were verified against the round-1 implementation, where the
    predicate self-resolved unconditionally."""
    from cell_eval2.scoring import is_degenerate, score_one

    # (a) an overflowing sentinel that a call-time FLOOR makes harmless
    hi = Scoring(scored=True, direction="higher", anchor=5e-324, clamp_low=None,
                 metric_min=-sys.float_info.max)
    assert is_degenerate(-5e-324, hi) is True                       # self-resolved: unfloored
    assert is_degenerate(-5e-324, hi, clamp_low=0.0) is False        # ...but not with a floor
    assert score_one(None, -5e-324, hi, clamp_low=0.0) == 0.0

    # (b) an overflowing sentinel that a call-time BOX-COX TAIL makes harmless. This is why
    # the sentinel has to go through the tail-aware core: the tail saturates at -cap.
    lo = Scoring(scored=True, direction="lower", anchor=0.0, penalty="none", clamp_low=None,
                 metric_min=sys.float_info.max)
    assert is_degenerate(5e-324, lo) is True
    assert is_degenerate(5e-324, lo, penalty="boxcox") is False
    assert score_one(None, 5e-324, lo, penalty="boxcox") == -6.0

    # (c) the other direction: a FLOORED policy that a call-time clamp_low=-inf unfloors.
    # Round 1 called this non-degenerate and then let score_one raise inside score_metrics.
    fl = Scoring(scored=True, direction="higher", anchor=5e-324, clamp_low=0.0,
                 metric_min=-sys.float_info.max)
    assert is_degenerate(-5e-324, fl) is False
    assert is_degenerate(-5e-324, fl, clamp_low=float("-inf")) is True


def test_the_predicate_and_the_scorer_resolve_one_call_identically():
    """The invariant behind the test above, stated over the shipped catalog rather than over
    a constructed pair: for every scored entry and a spread of override combinations,
    `is_degenerate` False must imply `score_one` returns a finite number.

    ONE documented exception, and the sweep asserts it rather than skipping it: a caller may
    pass `clamp_low=float("-inf")` to disable clamping outright, and for a policy with no
    `metric_min` there is then nothing to fall back to -- a missing value scores `-inf`
    because that is precisely what was asked for. That predates this change. Where the policy
    DOES declare a `metric_min` the same override must still yield a finite score, which is
    the half this change is responsible for."""
    import itertools

    from cell_eval2.catalog import CATALOG
    from cell_eval2.scoring import is_degenerate, score_one

    combos = [{}, {"clamp_low": 0.0}, {"clamp_low": float("-inf")}, {"penalty_cap": 3.0}]
    checked = 0
    for name, spec in CATALOG.items():
        if not spec.scoring.scored:
            continue
        for base, kw in itertools.product((0.02, 0.5, 0.97, 1.5), combos):
            # No `except ValueError` here on purpose: every combination below is LEGAL for
            # every scored policy, so a raise is a regression, not a case to skip. Codex
            # round 3 confirmed a replay encounters none.
            if is_degenerate(base, spec.scoring, **kw):
                continue
            try:
                got = score_one(None, base, spec.scoring, **kw)
            except ValueError:
                raise AssertionError(
                    f"{name}: is_degenerate said usable with {kw}, score_one refused"
                ) from None
            if kw.get("clamp_low") == float("-inf") and spec.scoring.metric_min is None:
                assert got == float("-inf"), (
                    f"{name}: clamping was explicitly disabled and there is no metric_min, "
                    f"so a missing value has no floor; got {got!r}"
                )
            else:
                assert math.isfinite(got), f"{name} with {kw} at base={base} scored {got!r}"
            checked += 1
    assert checked > 500, f"the sweep only exercised {checked} combinations"


def test_a_floored_policy_is_untouched_by_the_sentinel_check():
    """The new `is_degenerate` branch must not widen for a policy that has a real floor: its
    clamp catches the overflow, and `score_one` never consults `metric_min` at all."""
    from cell_eval2.scoring import is_degenerate, score_one

    floored = Scoring(scored=True, direction="higher", anchor=5e-324, clamp_low=0.0,
                      metric_min=-sys.float_info.max)
    assert is_degenerate(-5e-324, floored) is False
    assert score_one(None, -5e-324, floored) == 0.0


def test_metric_min_requires_an_anchor():
    """`metric_min` means "the worse end of this metric's range", so it needs a stated
    perfect end to be worse THAN -- both the worse-side rule and the sentinel's meaning are
    undefined without one. NOT because an anchor bounds the sentinel: it does not, which is
    why `is_degenerate` checks representability at runtime for anchored policies too."""
    with pytest.raises(ValueError, match="metric_min requires a non-None anchor"):
        Scoring(scored=True, direction="higher", anchor=None,
                clamp_low=-2.0, clamp_high=2.0, metric_min=0.0)


@pytest.mark.parametrize("kw,match", [
    ({"penalty": "garbage"}, "penalty override must be"),
    ({"clamp_low": float("nan")}, "clamp_low override must be"),
    ({"clamp_low": float("inf")}, "clamp_low override must be"),
    ({"clamp_high": float("-inf")}, "clamp_high override must be"),
    ({"penalty_cap": 0.0}, "penalty_cap override must be"),
    ({"penalty_exponent": float("nan")}, "penalty_exponent override must be"),
])
def test_is_degenerate_refuses_exactly_what_score_one_refuses(kw, match):
    """Codex round 3, finding 1. Sharing RESOLUTION was not enough -- the predicate then ran
    the scoring arithmetic on knobs the scorer refuses. With
    `penalty="boxcox", penalty_exponent=0.0` the tail computes `-(r**0 - 1)/0` and
    `is_degenerate` raised a bare `ZeroDivisionError`; `penalty="garbage"` and a NaN
    `clamp_low` returned a boolean instead of raising. Both now go through `_resolve_call`.
    """
    from cell_eval2.scoring import BOUNDED, is_degenerate, score_one

    with pytest.raises(ValueError, match=match):
        is_degenerate(0.5, BOUNDED, **kw)
    with pytest.raises(ValueError, match=match):
        score_one(0.25, 0.5, BOUNDED, **kw)


def test_the_predicate_validates_before_it_short_circuits_on_the_base():
    """A call with an invalid override is a bad call whatever the base is. Validating after
    the `base is None` check would let it return a boolean instead of raising."""
    from cell_eval2.scoring import BOUNDED, is_degenerate

    assert is_degenerate(None, BOUNDED) is True
    with pytest.raises(ValueError, match="penalty override must be"):
        is_degenerate(None, BOUNDED, penalty="garbage")


def test_the_zero_exponent_boxcox_case_raises_identically_in_both():
    """The exact case codex round 3 reported, pinned end to end."""
    from cell_eval2.scoring import is_degenerate, score_one

    pol = Scoring(scored=True, direction="lower", anchor=0.0, penalty="none",
                  clamp_low=None, metric_min=sys.float_info.max)
    kw = {"clamp_low": float("-inf"), "penalty": "boxcox", "penalty_exponent": 0.0}
    with pytest.raises(ValueError, match="penalty_exponent override must be finite"):
        is_degenerate(5e-324, pol, **kw)
    with pytest.raises(ValueError, match="penalty_exponent override must be finite"):
        score_one(None, 5e-324, pol, **kw)
