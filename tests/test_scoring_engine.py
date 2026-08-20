import math
import sys

import numpy as np
import pytest

from cell_eval2.scoring import (
    BOUNDED,
    DEFAULT_PENALTY_CAP,
    DEFAULT_PENALTY_EXPONENT,
    ERROR,
    Scoring,
    denominator,
    is_degenerate,
    score_one,
)

P, C = DEFAULT_PENALTY_EXPONENT, DEFAULT_PENALTY_CAP


def _score(u, b, policy):
    return score_one(u, b, policy, penalty_exponent=P, penalty_cap=C)


def _reference_error(u, b, p=P, c=C):
    """score.py::_penalized_zero, written out so the engine is checked against
    arithmetic rather than against the implementation it replaces."""
    if u is None or not np.isfinite(u):
        return -c
    s = 1.0 - (u / b)
    if s >= 0.0:
        return s
    r = u / b
    return max(-c, -(r ** p - 1.0) / p)


def _reference_bounded(u, b):
    """score.py::_norm_by_one followed by the non-finite fallback and the >=0 clamp."""
    s = float("nan") if b == 1 else (u - b) / (1.0 - b)
    if not np.isfinite(s):
        s = 0.0
    return max(0.0, s)


@pytest.mark.parametrize("b", [0.05, 0.2, 0.5, 0.9])
@pytest.mark.parametrize("u", [0.0, 0.01, 0.2, 0.5, 1.0, 3.0])
def test_error_class_is_bit_identical_to_the_reference(u, b):
    assert _score(u, b, ERROR) == _reference_error(u, b)


@pytest.mark.parametrize("b", [0.0, 0.2, 0.5, 0.9])
@pytest.mark.parametrize("u", [0.0, 0.3, 0.5, 0.95, 1.0])
def test_bounded_class_is_bit_identical_to_the_reference(u, b):
    assert _score(u, b, BOUNDED) == _reference_bounded(u, b)


def test_boxcox_tail_in_s_matches_the_r_form():
    b = 0.2
    for u in (0.3, 0.4, 0.8, 1.5):
        r = u / b
        expected = max(-C, -(r ** P - 1.0) / P)
        assert _score(u, b, ERROR) == expected      # necessary, but see the next test


def test_the_tail_uses_a_CARRIED_r_not_one_reconstructed_from_s():
    """Acceptance criterion 2. The previous test cannot distinguish the two: on the
    shipped knobs (p=2, C=6) the tail saturates at r >= sqrt(13), and 1-(1-r) == r
    bitwise for every r in [1, 2^53], so a reconstructing implementation passes it.
    This one uses the regime where they actually differ."""
    u, b, p, cap = float(2 ** 53 + 2), 1.0, 1.0, 1e17
    r = u / b
    assert 1.0 - (1.0 - r) != r          # precondition: fail loudly if this stops biting
    carried = max(-cap, -(r ** p - 1.0) / p)
    reconstructed = max(-cap, -((1.0 - (1.0 - r)) ** p - 1.0) / p)
    assert carried != reconstructed      # the two are genuinely separable here
    assert score_one(u, b, ERROR, penalty_exponent=p, penalty_cap=cap) == carried


def test_the_cap_saturates_the_TAIL_not_only_the_FLOOR():
    """Spec 2.3 applies `max(-C, ...)` inside the tail; spec 2.4's clip is a SEPARATE step.
    They coincide only while clamp_low == -cap, which is ERROR's default -- so every test
    that uses the default floor passes either way. With a lower floor the difference is the
    whole score: relying on the clip alone gives -(100**2-1)/2 = -4999.5 instead of -6, and
    one such submission drags avg_score with it (checkpoint-2 codex finding 1)."""
    assert score_one(100.0, 1.0, ERROR, clamp_low=float("-inf")) == -C
    assert score_one(100.0, 1.0, ERROR, clamp_low=-100.0) == -C
    # a catalog-expressible floor below the cap, i.e. not only the call-time route
    deep = Scoring(scored=True, direction="lower", anchor=0.0, penalty="boxcox",
                   clamp_low=-100.0, penalty_cap=10.0)
    assert score_one(100.0, 1.0, deep) == -10.0
    # ...and the overflow path takes the cap too, as the frozen kernel did
    assert score_one(1e300, 1.0, ERROR, clamp_low=float("-inf")) == -C


def test_a_non_finite_denominator_is_degenerate_not_a_plausible_zero():
    """`anchor - base` overflows for a large anchor and an opposite-signed base, both of
    them finite and both accepted by __post_init__. D = inf passes a bare `<= 0` test and
    then divides the gap to 0.0 -- a wrong number (the true value is 0.5) with no signal
    (checkpoint-2 codex finding 4)."""
    huge = Scoring(scored=True, direction="higher", anchor=sys.float_info.max)
    assert denominator(-sys.float_info.max, huge) == math.inf
    assert is_degenerate(-sys.float_info.max, huge) is True


def test_non_finite_user_takes_clamp_low_in_both_classes():
    assert _score(float("nan"), 0.2, ERROR) == -C
    assert _score(None, 0.2, ERROR) == -C
    assert _score(float("nan"), 0.5, BOUNDED) == 0.0


# --- anchorless (spec 2.2) ---------------------------------------------------
ANCHORLESS = Scoring(scored=True, direction="higher", anchor=None, clamp_low=-5.0,
                     clamp_high=5.0, allow_negative_baseline=True)   # the direction_yield shape
NONNEG = Scoring(scored=True, direction="higher", anchor=None, clamp_low=-5.0, clamp_high=5.0)


def test_anchorless_uses_abs_baseline_and_stays_monotone_for_negative_base():
    # direction_yield: higher is better, baseline legitimately negative. With D = b
    # (no abs) these would come out sign-flipped.
    assert _score(0.0, -0.5, ANCHORLESS) == pytest.approx(1.0)
    assert _score(-0.25, -0.5, ANCHORLESS) == pytest.approx(0.5)
    assert _score(-0.5, -0.5, ANCHORLESS) == pytest.approx(0.0)
    assert _score(-1.0, -0.5, ANCHORLESS) == pytest.approx(-1.0)


def test_anchorless_positive_base_reduces_to_ratio_minus_one():
    assert _score(1.0, 0.5, ANCHORLESS) == pytest.approx(1.0)   # u/b - 1


def test_anchorless_lower_matches_anchor_zero_when_base_positive():
    lower = Scoring(scored=True, direction="lower", anchor=None, clamp_low=-5.0)
    anchored = Scoring(scored=True, direction="lower", anchor=0.0, clamp_low=-5.0)
    for u in (0.05, 0.1, 0.2):
        assert _score(u, 0.2, lower) == pytest.approx(_score(u, 0.2, anchored))


# --- clamps ------------------------------------------------------------------
def test_clamp_high_truncates_above():
    assert _score(10.0, 0.5, ANCHORLESS) == 5.0


def test_clamping_is_disabled_by_CALL_TIME_ARGUMENTS_not_a_rebuilt_policy():
    # dataclasses.replace(policy, clamp_low=-inf) would be REJECTED by the finite-floor rule
    # (replace re-runs __post_init__ -- verified). So the override is an argument.
    assert score_one(10.0, 0.5, ANCHORLESS, penalty_exponent=P, penalty_cap=C,
                     clamp_low=float("-inf"), clamp_high=float("inf")) == pytest.approx(19.0)


def test_replace_on_a_scored_policy_rejects_an_infinite_floor():
    from dataclasses import replace
    with pytest.raises(ValueError, match="finite"):
        replace(BOUNDED, clamp_low=float("-inf"))


def test_finite_user_that_overflows_the_normalization_takes_clamp_low():
    # Today: _norm_by_one -> inf -> isfinite fallback -> 0.0. A naive clip would return inf.
    b, u = math.nextafter(1.0, 0.0), sys.float_info.max
    assert _score(u, b, BOUNDED) == 0.0


# --- _clip's two named edge semantics (acceptance criterion 5c) ------------------
def test_clip_is_nan_safe_and_normalizes_negative_zero():
    """Neither is reachable through the plain paths above, and both are specifically
    required: NaN fails every comparison, so a `low if x < low else ...` clip returns the
    NaN; and a `<` (rather than `<=`) lower test lets -0.0 through where today's
    max(0.0, score) yields +0.0."""
    from cell_eval2.scoring import _clip

    assert _clip(float("nan"), 0.0, float("inf")) == 0.0
    assert math.copysign(1.0, _clip(-0.0, 0.0, float("inf"))) == 1.0
    assert _clip(float("inf"), 0.0, 5.0) == 0.0        # non-finite -> the FLOOR, not the ceiling


# --- call-time override validation (spec 3.1a) ----------------------------------
def test_call_time_penalty_override_is_validated_not_silently_ignored():
    # boxcox on a higher/anchored policy is not merely inapplicable -- asking for it and
    # getting linear behaviour back is a silent wrong answer.
    with pytest.raises(ValueError, match="boxcox"):
        score_one(0.8, 0.5, BOUNDED, penalty="boxcox")
    with pytest.raises(ValueError, match="penalty"):
        score_one(0.8, 0.5, BOUNDED, penalty="quadratic")


@pytest.mark.parametrize("kw", [{"penalty_cap": float("inf")}, {"penalty_cap": float("nan")},
                                {"penalty_exponent": float("inf")}, {"penalty_cap": 0.0}])
def test_call_time_penalty_knobs_must_be_finite_and_positive(kw):
    # `inf`/`nan` slip past a bare `<= 0` guard; an infinite cap gives an infinite floor.
    with pytest.raises(ValueError, match="penalty_"):
        score_one(2.0, 1.0, ERROR, **kw)


def test_effective_clamp_bounds_are_ordered_not_just_the_supplied_ones():
    # low resolves from BOUNDED's catalog 0.0, high from the override -> [0.0, -1.0].
    # Checking only the two supplied arguments would return -1.0, below the floor.
    with pytest.raises(ValueError, match="clamp"):
        score_one(0.8, 0.5, BOUNDED, clamp_high=-1.0)


# --- spec 3.2 precedence, ALL FIVE knobs (criterion 9) --------------------------
# The catalog values are deliberately unusual, so a global argument that fails to
# override one is visible in the NUMBER rather than only in a flag. u/b = 3 -> s = -2,
# and boxcox with p=2 gives -(3**2 - 1)/2 = -4.
_PINNED = Scoring(scored=True, direction="lower", anchor=0.0, penalty="boxcox",
                  clamp_low=-8.0, penalty_exponent=2.0, penalty_cap=8.0, clamp_high=0.25)
# clamp_low=None so penalty_cap is observed through the FLOOR it derives. (Since the cap
# also saturates the tail it is observable with an explicit floor too -- see
# test_the_cap_saturates_the_TAIL_not_only_the_FLOOR -- but this row is specifically about
# the derived-floor route, which is the one spec 2.4's default table describes.)
_PINNED_CAP_FLOOR = Scoring(scored=True, direction="lower", anchor=0.0, penalty="boxcox",
                            clamp_low=None, penalty_exponent=2.0, penalty_cap=8.0)


@pytest.mark.parametrize("policy,u,kwargs,expected", [
    (_PINNED,           3.0, {},                        -4.0),   # catalog throughout
    (_PINNED,           3.0, {"penalty_exponent": 1.0}, -2.0),   # -(3-1)/1
    (_PINNED,           3.0, {"penalty": "none"},       -2.0),   # linear s, tail skipped
    (_PINNED,           3.0, {"clamp_low": -3.0},       -3.0),   # beats the catalog -8.0
    (_PINNED,           3.0, {"clamp_high": -4.5},      -4.5),   # beats the catalog 0.25
    (_PINNED_CAP_FLOOR, 5.0, {},                        -8.0),   # -(25-1)/2 = -12, floored
    (_PINNED_CAP_FLOOR, 5.0, {"penalty_cap": 3.0},      -3.0),   # global cap moves the FLOOR
])
def test_every_knob_resolves_global_argument_over_catalog(policy, u, kwargs, expected):
    """The last row is the one that bites: `penalty_cap` feeds the floor via
    effective_clamp_low, so a version that re-derives the cap from the policy there
    resolves it twice with opposite precedence and returns -8.0."""
    assert score_one(u, 1.0, policy, **kwargs) == pytest.approx(expected)


def test_penalty_none_override_uses_the_RESOLVED_penalty_for_the_floor():
    with pytest.raises(ValueError, match="finite floor"):
        score_one(100.0, 1.0, ERROR, penalty="none")
    # ...and with a floor supplied it is linear and clipped, i.e. the frozen behaviour.
    assert score_one(100.0, 1.0, ERROR, penalty="none", clamp_low=0.0) == 0.0
    assert score_one(0.25, 1.0, ERROR, penalty="none", clamp_low=0.0) == 0.75


def test_unscored_policy_may_stay_unfloored_under_a_penalty_override():
    # The finite-floor rule protects avg_score, so it binds only on scored policies.
    diag = Scoring(scored=False, direction="lower", anchor=0.0, penalty="boxcox",
                   clamp_low=None)
    assert score_one(100.0, 1.0, diag, penalty="none") == -99.0


def test_nonnegative_anchorless_rejects_a_negative_baseline():
    assert is_degenerate(-0.5, NONNEG) is True
    assert is_degenerate(-0.5, ANCHORLESS) is False


# --- degeneracy (spec 6) -----------------------------------------------------
@pytest.mark.parametrize(
    "base,policy,expected",
    [
        (0.0, ERROR, True), (-0.1, ERROR, True), (0.2, ERROR, False),
        (1.0, BOUNDED, True), (1.5, BOUNDED, True), (0.5, BOUNDED, False),
        (0.0, ANCHORLESS, True), (-0.5, ANCHORLESS, False), (0.5, ANCHORLESS, False),
        (None, ERROR, True), (float("nan"), BOUNDED, True), (math.inf, ERROR, True),
    ],
)
def test_degenerate_predicate(base, policy, expected):
    assert is_degenerate(base, policy) is expected


def test_denominator_matches_the_spec_table():
    assert denominator(0.2, ERROR) == pytest.approx(0.2)        # b - anchor
    assert denominator(0.5, BOUNDED) == pytest.approx(0.5)      # anchor - b
    assert denominator(-0.5, ANCHORLESS) == pytest.approx(0.5)  # |b|
