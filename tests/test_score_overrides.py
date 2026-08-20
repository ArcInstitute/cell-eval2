import polars as pl
import pytest

from cell_eval2.score import score_metrics
from cell_eval2.scoring import BOUNDED, Scoring


def _frames(user_val, base_val, metric="pds_l1"):
    user = pl.DataFrame({"statistic": ["mean"], metric: [user_val]})
    base = pl.DataFrame({"statistic": ["mean"], metric: [base_val]})
    return user, base


def test_degenerate_bounded_baseline_now_raises():
    # Was: _norm_by_one(u, 1) -> NaN -> silently clipped to 0.0 for every submission.
    user, base = _frames(0.8, 1.0)
    with pytest.raises(ValueError, match="degenerate baseline"):
        score_metrics(user, base)


def test_degenerate_error_baseline_still_raises():
    user, base = _frames(0.1, 0.0, metric="expr_mae")
    with pytest.raises(ValueError, match="degenerate baseline"):
        score_metrics(user, base)


def test_global_clamp_override_disables_the_zero_clip():
    user, base = _frames(0.3, 0.5)          # below baseline -> normally clipped to 0.0
    out = score_metrics(user, base, clamp_low=float("-inf"), clamp_high=float("inf"))
    got = out.filter(pl.col("metric") == "pds_l1")["from_baseline"][0]
    assert got == pytest.approx((0.3 - 0.5) / (1.0 - 0.5))


def test_clamp_high_override_truncates_above():
    user, base = _frames(0.9, 0.5)          # (0.9-0.5)/(1-0.5) = 0.8
    out = score_metrics(user, base, clamp_high=0.5)
    assert out.filter(pl.col("metric") == "pds_l1")["from_baseline"][0] == pytest.approx(0.5)


def test_per_metric_override_beats_the_global_one_THROUGH_THE_ALIAS_PATH():
    """Criterion 9. The column carries the v1 name while the override key is canonical, so
    the lookup must try BOTH `name` and `spec.name` -- and so must the has-override test
    that suppresses the globals. A `name in overrides` check alone passes a same-name test
    and fails this one; that is exactly the regression codex round 2 flagged."""
    user, base = _frames(0.9, 0.5, metric="discrimination_score_l1")
    out = score_metrics(user, base, clamp_high=0.5,
                        overrides={"pds_l1": Scoring(scored=True, direction="higher",
                                                     anchor=1.0, clamp_high=0.2)})
    got = out.filter(pl.col("metric") == "discrimination_score_l1")["from_baseline"][0]
    assert got == pytest.approx(0.2)          # the per-metric 0.2, not the global 0.5


def test_a_per_metric_override_suppresses_the_global_penalty_knobs_too():
    """A wholesale replacement must not inherit globals for exponent/cap either (spec 3.2).
    The override pins cap=1.0, so a non-finite user takes -1.0, not the global -4.0."""
    user, base = _frames(float("nan"), 0.5, metric="expr_mae")
    out = score_metrics(user, base, penalty_cap=4.0,
                        overrides={"expr_mae": Scoring(scored=True, direction="lower",
                                                       anchor=0.0, penalty="boxcox",
                                                       clamp_low=None, penalty_cap=1.0)})
    assert out.filter(pl.col("metric") == "expr_mae")["from_baseline"][0] == -1.0


@pytest.mark.parametrize("knob,global_kw,expected", [
    # The override pins every knob, so each global below must be IGNORED. Each row is
    # chosen so the global would move the NUMBER if it leaked through, not merely raise.
    # Override: lower/anchor-0, boxcox, p=2, cap=8, clamp_low=-8.0, clamp_high=0.25.
    # With u/b = 3: s = -2, tail = max(-8, -(3**2-1)/2) = -4.
    ("penalty_exponent", {"penalty_exponent": 1.0}, -4.0),   # would give -(3-1)/1 = -2
    ("penalty_cap", {"penalty_cap": 3.0}, -4.0),             # would saturate the tail at -3
    ("penalty", {"penalty": "none"}, -4.0),                  # would skip the tail -> -2
    ("clamp_low", {"clamp_low": -3.0}, -4.0),                # would floor at -3
    ("clamp_high", {"clamp_high": -4.5}, -4.0),              # would truncate to -4.5
])
def test_a_per_metric_override_suppresses_EVERY_global_knob(knob, global_kw, expected):
    """Criterion 9 says all five knobs, not just the clamps. A wholesale replacement that
    still inherited one global would invert the documented precedence for that knob only --
    invisible unless each is exercised on its own with an observably different number.

    The floor is EXPLICIT (-8.0) rather than derived: with clamp_low=None a leaked
    penalty="none" would raise (no finite floor) instead of returning a different number,
    so that row would pass for the wrong reason. penalty_cap stays observable anyway,
    because the cap saturates the tail independently of the floor."""
    policy = Scoring(scored=True, direction="lower", anchor=0.0, penalty="boxcox",
                     clamp_low=-8.0, penalty_exponent=2.0, penalty_cap=8.0, clamp_high=0.25)
    user, base = _frames(3.0, 1.0, metric="expr_mae")
    out = score_metrics(user, base, overrides={"expr_mae": policy}, **global_kw)
    got = out.filter(pl.col("metric") == "expr_mae")["from_baseline"][0]
    assert got == pytest.approx(expected), f"the global {knob} leaked through the override"


def test_an_override_key_works_in_EITHER_spelling_against_EITHER_column():
    """The key and the column must not have to agree. Matching raw key against raw column
    makes the same override honoured or ignored depending on which spelling the aggregate
    frame carries -- a v1 key silently did nothing against a canonical column
    (checkpoint-2 codex finding 3)."""
    policy = Scoring(scored=True, direction="higher", anchor=1.0, clamp_high=0.2)
    for column in ("pds_l1", "discrimination_score_l1"):
        for key in ("pds_l1", "discrimination_score_l1"):
            user, base = _frames(0.9, 0.5, metric=column)
            out = score_metrics(user, base, overrides={key: policy})
            got = out.filter(pl.col("metric") == column)["from_baseline"][0]
            assert got == pytest.approx(0.2), f"key={key!r} column={column!r}"


def test_an_unknown_override_key_raises_rather_than_being_ignored():
    # A silently dropped override is a wrong number with nothing in the output saying so.
    user, base = _frames(0.9, 0.5)
    with pytest.raises(ValueError, match="unknown metric in overrides"):
        score_metrics(user, base, overrides={"pds_11": BOUNDED})   # typo for pds_l1


def test_two_synonymous_override_keys_raise():
    # Otherwise dict order silently decides which of the two policies wins.
    user, base = _frames(0.9, 0.5)
    with pytest.raises(ValueError, match="twice"):
        score_metrics(user, base,
                      overrides={"pds_l1": BOUNDED, "discrimination_score_l1": BOUNDED})


def test_a_global_penalty_knob_beats_the_catalog():
    # ERROR leaves penalty_cap None, so the global wins and a NaN user takes -4.0.
    user, base = _frames(float("nan"), 0.5, metric="expr_mae")
    out = score_metrics(user, base, penalty_cap=4.0)
    assert out.filter(pl.col("metric") == "expr_mae")["from_baseline"][0] == -4.0


def test_unscored_metric_is_still_skipped():
    user, base = _frames(3.0, 2.0, metric="de_wilcoxon_nsig_counts_real")
    out = score_metrics(user, base)
    assert out["metric"].to_list() == ["avg_score"]


# --- the degenerate raise, at the FRAME level (criterion 6) ----------------------
def test_bounded_baseline_PAST_perfection_also_raises():
    """The deliberate extension of spec 6: today b > 1 is scored (negative, then clipped
    to 0), not just b == 1. Predicate-level coverage alone would not catch a scorer that
    forgets to call is_degenerate for this class."""
    user, base = _frames(0.8, 1.5)
    with pytest.raises(ValueError, match="degenerate baseline"):
        score_metrics(user, base)


@pytest.mark.parametrize("base_val,signed,raises", [
    (0.0, True, True),      # signed anchorless: D == 0
    (-0.5, True, False),    # signed anchorless: D = |b| = 0.5, fine
    (-0.5, False, True),    # non-negative anchorless: D = b <= 0
])
def test_anchorless_degeneracy_reaches_score_metrics_through_an_override(base_val, signed, raises):
    """Exercises the two anchorless degeneracy classes end-to-end through `overrides`.

    (The docstring here used to say no catalog metric is anchorless AND scored. That stopped
    being true when the 20 directional metrics were enrolled -- there are twelve. The
    override route is still the clean way to reach BOTH classes with fixed values.)"""
    policy = Scoring(scored=True, direction="higher", anchor=None, clamp_low=-5.0,
                     allow_negative_baseline=signed)
    user, base = _frames(1.0, base_val)
    if raises:
        with pytest.raises(ValueError, match="degenerate baseline"):
            score_metrics(user, base, overrides={"pds_l1": policy})
    else:
        score_metrics(user, base, overrides={"pds_l1": policy})


def test_a_degenerate_v2_native_baseline_skips_that_metric_instead_of_aborting():
    """Fail-loud is narrowed to metrics where a wrong number decides something.

    `de_*_direction_yield` is the motivating case and it is not a corner: it is signed and
    centred at zero BY CONSTRUCTION, returns exactly 0.0 when `n_pred == 0` (pinned by
    test_direction_fidelity), and aggregates by MEDIAN -- so a baseline that calls nothing
    for a majority of perturbations lands on exactly 0.0 legitimately. `is_degenerate`
    rejects that, and before enrolment the metric was unscored so it could never fire. Taking
    every other scored metric down with it is the wrong trade. (`expr_mse_unbiased_norm`
    (#198, removed by #257) was the second such case, in `full`/`anndata` rather than
    `full`/`de`.)
    """
    user = pl.DataFrame({"statistic": ["mean"], "expr_mae": [0.5],
                         "de_wilcoxon_direction_yield": [0.3]})
    base = pl.DataFrame({"statistic": ["mean"], "expr_mae": [1.0],
                         "de_wilcoxon_direction_yield": [0.0]})
    out = score_metrics(user, base)
    assert set(out["metric"]) == {"expr_mae", "avg_score"}
    # and the aggregate is the surviving metric alone, not diluted by a placeholder
    assert out.filter(pl.col("metric") == "avg_score")["from_baseline"].item() == 0.5


def test_a_degenerate_baseline_on_a_decisive_metric_still_aborts():
    """The other half. Everything v1 can emit keeps the hard failure -- scoring every
    submission against an undefined denominator is worse than stopping."""
    for metric, bad_base in (("expr_mae", 0.0), ("pds_l1", 1.0), ("de_wilcoxon_overlap", 1.0)):
        user, base = _frames(0.5, bad_base, metric=metric)
        with pytest.raises(ValueError, match="degenerate baseline"):
            score_metrics(user, base)


def test_the_vcc_half_of_the_decisive_predicate_is_load_bearing(monkeypatch):
    """Every vcc metric happens to be v1-available today, so deleting the `vcc` disjunct would
    break nothing the catalog can currently demonstrate. Build the metric that exposes it:
    `de_wilcoxon_direction_yield` is v2-native and outside both competition profiles, so its
    legitimate zero baseline remains skippable. Installing it in `vcc` must make the same
    baseline fail loud. Asserting `is_decisive` alone would still pass if the scorer stopped
    consulting it and read `spec.v1_available` directly."""
    from dataclasses import replace as dc_replace

    from cell_eval2.catalog import CATALOG, is_decisive

    name = "de_wilcoxon_direction_yield"
    v2_native = CATALOG[name]
    assert v2_native.v1_available is False and is_decisive(v2_native) is False

    in_vcc = dc_replace(v2_native, profiles=("full", "de", "vcc"))
    assert in_vcc.v1_available is False          # still v2-native...
    assert is_decisive(in_vcc) is True           # ...but decisive, via the vcc disjunct alone

    user, base = _frames(0.3, 0.0, metric=name)
    # baseline: as a plain v2-native metric it is skipped, so the run has nothing left
    with pytest.raises(ValueError, match="nothing left to score"):
        score_metrics(user, base)
    # in vcc, the SAME degenerate baseline must be refused as a degenerate baseline instead
    monkeypatch.setitem(CATALOG, name, in_vcc)
    with pytest.raises(ValueError, match="degenerate baseline"):
        score_metrics(user, base)


def test_skipping_every_metric_raises_rather_than_reporting_avg_score_zero():
    """The empty-set hazard, one level down from baseline.py's. If every scoreable column is
    dropped for a degenerate baseline, the `if scores else 0.0` fallback would report
    avg_score = 0.0 -- a number that reads as "equals the baseline" rather than "nothing was
    scored"."""
    user = pl.DataFrame({"statistic": ["mean"], "de_wilcoxon_direction_yield": [0.3]})
    base = pl.DataFrame({"statistic": ["mean"], "de_wilcoxon_direction_yield": [0.0]})
    with pytest.raises(ValueError, match="nothing left to score"):
        score_metrics(user, base)


def test_the_skip_warning_says_avg_score_composition_changed(caplog):
    """The warning is the only signal that avg_score now means something different, so assert
    it rather than trusting it exists."""
    user = pl.DataFrame({"statistic": ["mean"], "expr_mae": [0.5],
                         "de_wilcoxon_direction_yield": [0.3]})
    base = pl.DataFrame({"statistic": ["mean"], "expr_mae": [1.0],
                         "de_wilcoxon_direction_yield": [0.0]})
    with caplog.at_level("WARNING"):
        score_metrics(user, base)
    hits = [r.message for r in caplog.records if "direction_yield" in r.message]
    assert hits, "the skipped metric must be named in a warning"
    assert any("EXCLUDING it from avg_score" in m and "not directly comparable" in m
               for m in hits)
