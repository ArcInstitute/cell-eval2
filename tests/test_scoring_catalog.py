import pytest

from cell_eval2.catalog import CATALOG, MetricSpec
from cell_eval2.scoring import BOUNDED, Scoring

DIRECTIONLESS = {"de_wilcoxon_nsig_counts_real", "de_wilcoxon_nsig_counts_pred",
                 "de_deseq2_nsig_counts_real", "de_deseq2_nsig_counts_pred"}


def test_every_spec_has_a_scoring_policy():
    assert len(CATALOG) == 95          # 13 anndata literals + 41 per DE family x 2
    assert all(isinstance(s.scoring, Scoring) for s in CATALOG.values())


# The DIRECTIONLESS entries, named. #198 briefly made these a STRICT subset of the unscored
# ones (`expr_mse_unbiased_norm` was unscored WITH a direction); enrolling it restored the
# identity, and #257 replaced it with three components that are unscored AND directionless,
# pinned separately below -- so the two sets coincide again. Asserted in both directions
# rather than assumed, since that is exactly the equivalence #203 warned was a description of
# the catalog and never its design. Pinning the complement rather than the
# 87 scored ones keeps the golden short while still fixing BOTH sides exactly: a count plus a
# spot check would pass if a scored higher-is-better metric swapped places with an unscored one.
_UNSCORED = {
    f"de_{m}_{s}"
    for m in ("wilcoxon", "deseq2")
    for s in ("nsig_counts_real", "nsig_counts_pred")
}
_COMPONENT_DIAGNOSTICS = {
    "expr_mse_unbiased", "expr_mse_unbiased_capped", "expr_distance_unbiased",
}
_ANNDATA_DIAGNOSTICS = _COMPONENT_DIAGNOSTICS | {"expr_real_mass_ratio"}


def test_enrolment_implies_a_direction_but_not_conversely():
    """Every scored metric has a direction. Two states occupied today, both pinned by name
    below so neither can silently become the other:

      * scored, with a direction        -- 87 entries
      * unscored because DIRECTIONLESS  -- the four `de_*_nsig_counts_*`, the three
                                           diagnostic components of the derived metric, and
                                           the real-mass diagnostic

    The third state -- directional but unscored BY DECISION -- is currently EMPTY, and that
    emptiness is asserted rather than left implicit. The mechanism that makes it expressible is
    pinned separately by `test_diagnostic_may_record_direction_without_being_scored`, which
    builds the policy directly instead of relying on a catalog entry to occupy it.

    So the relation is once again an equivalence over the catalog -- but only coincidentally,
    which is why the two directions are asserted separately. #203 landed with exactly this
    shape and asserted the equivalence as if it were the design; it was not. Under
    `best_value`, "unscored" and "unbounded" and "directionless" were one token, so the 20
    directional-but-unscored metrics could not be enrolled without also claiming an anchor
    they do not have. Splitting the axes is what let enrolment stop being a property of the
    mathematics, and re-collapsing them here would undo that.
    """
    scored = {n for n, s in CATALOG.items() if s.scoring.scored}
    assert len(scored) == 87
    # `_UNSCORED` is the four DE directionless entries: it is reused below against a
    # DE-only `former_none` set, where a fifth member would be wrong.
    assert set(CATALOG) - scored == _UNSCORED | _ANNDATA_DIAGNOSTICS
    # scored => has a direction. The converse holds today but is NOT guaranteed by the type,
    # so the empty third state is asserted explicitly: a future directional-but-unscored
    # entry must update this test deliberately rather than slip in under a passing count.
    for name, spec in CATALOG.items():
        if spec.scoring.scored:
            assert spec.scoring.direction is not None, name
    assert {n for n, s in CATALOG.items()
            if s.scoring.direction is None} == _UNSCORED | _ANNDATA_DIAGNOSTICS
    assert {n for n, s in CATALOG.items()
            if s.scoring.direction is not None and not s.scoring.scored} == set()
    lower = {n for n, s in CATALOG.items() if s.scoring.scored and s.scoring.direction == "lower"}
    # #208: de_lfc_nmae is the first non-centroid lower-is-better member -- a normalized
    # per-gene LFC error, anchored at 0 like the four centroid error metrics. #198's
    # expr_mse_unbiased_capped_norm is the signed member of the class.
    assert lower == {"expr_mae", "expr_mse", "delta_mae", "delta_mse",
                     "expr_mse_unbiased_capped_norm",
                     "de_wilcoxon_lfc_nmae", "de_deseq2_lfc_nmae"}


# Scored entries whose METRIC VALUE can fall on the far side of its own anchor, so that the
# anchor does not bound the score. Only `expr_mse_unbiased_capped_norm` qualifies: it is a
# bias-corrected estimator of a non-negative quantity and is negative about half the time when
# the truth is near zero. (#247's cap biases it upward where it binds, so it is not strictly
# unbiased any more -- but negatives stay reachable, which is all this set asserts.) Every other anchored entry is non-negative (the errors) or lives in
# [-1, 1] with anchor 1 (the bounded ones), so its anchor really is a ceiling.
_SIGNED_PAST_ANCHOR = {"expr_mse_unbiased_capped_norm"}


def test_every_scored_metric_is_bounded_above():
    """No scored metric may contribute an unbounded term to avg_score. Unclamped, a near-zero
    baseline turns one metric into an arbitrarily large one: at u/b = 100 the raw anchorless
    score is 99, which moves a profile's avg_score by more than the entire achievable range of
    every other metric in it combined. (Stated without a metric count on purpose: the count
    changes every time a metric is enrolled, and it went stale here once already.)

    There are TWO doors to that failure, and the anchorless one is only the first:

      * anchorless          -- the score is u/b - 1, which nothing bounds; needs `clamp_high`.
      * anchored but SIGNED -- `1 - u/b` exceeds 1 for every u past the anchor. #198's
        `expr_mse_unbiased_capped_norm` is the one such entry, and its predecessor
        walked through this test
        while it read `anchor is None`: the check passed VACUOUSLY on it. Enrolment is what
        made the old docstring's "an anchored metric's score cannot exceed 1 by construction"
        false, so the rule is stated over the property that actually matters -- can the value
        pass its own anchor -- rather than over the presence of an anchor.
    """
    for name, spec in CATALOG.items():
        if not spec.scoring.scored:
            continue
        if spec.scoring.anchor is None or name in _SIGNED_PAST_ANCHOR:
            assert spec.scoring.clamp_high is not None, f"{name} is unbounded above"


def test_the_signed_anchored_clamp_actually_truncates():
    """The companion to `test_the_clamps_actually_truncate_the_runaway_cases`, for the anchored
    door. Pinning `clamp_high is not None` is not enough -- assert the NUMBER, in the regime
    that motivated the cap. All three were measured on the unclamped policy first."""
    from cell_eval2.scoring import is_degenerate, score_one

    sc = CATALOG["expr_mse_unbiased_capped_norm"].scoring
    assert score_one(-0.05, 0.05, sc) == 1.0        # raw 1 - u/b = 2.0
    assert score_one(-0.5, 0.05, sc) == 1.0         # raw 11.0, from the USER side
    assert score_one(-1e-3, 1e-4, sc) == 1.0        # raw 11.0, from the BASELINE side
    assert score_one(-1e-3, 1e-6, sc) == 1.0        # raw 1001.0
    # ...and the tiny-but-positive baseline that drove the third case is NOT caught by the
    # degenerate guard, which is why the clamp has to carry it.
    assert is_degenerate(1e-6, sc) is False
    # The interior is untouched: the linear region still runs from 1 at perfection to 0 at the
    # baseline.
    assert score_one(0.025, 0.05, sc) == 0.5
    # BELOW the baseline the tail no longer shows through: #276 part C floors this member at
    # 0.0, so r = 4 reads 0.0 where it read -7.5 (cap 10) and would read -6.0 (cap 6)
    # unfloored. This member is [0, 1] on both scales; every other boxcox metric keeps the
    # graded tail.
    assert score_one(0.2, 0.05, sc) == 0.0
    assert score_one(1e6, 0.05, sc) == 0.0


@pytest.mark.parametrize("method", ["wilcoxon", "deseq2"])
def test_the_lfc_nmae_family_is_linear_below_the_baseline(method):
    """`de_*_lfc_nmae` carries `ERROR_LINEAR`, not `ERROR` (Alex, 2026-08-17): the same -6.0
    floor, reached along a straight line rather than a Box-Cox quadratic.

    Asserted as NUMBERS in the regime that decided it, not as a field comparison. The four
    discriminating rows are r = 2, 3, 4 and 6: the tail read -1.5 and -4.0 at the first two and
    had already saturated at -6.0 by r = 4. The rest are pinned BECAUSE they did not move --
    at and above the baseline, at the floor, and on an unusable submission the two policies
    agree exactly, and that is the half a field comparison would not have caught.

    b = 1.0 so `r` is `u` and every expectation is exact in binary floating point; the
    replicate-scaled case below is the one that needs a tolerance.
    """
    from cell_eval2.scoring import score_one

    sc = CATALOG[f"de_{method}_lfc_nmae"].scoring
    assert (sc.penalty, sc.clamp_low, sc.clamp_high) == ("none", -6.0, None)
    # At and above the baseline: identical to `ERROR`, and to the frozen clip-at-0 kernel.
    assert score_one(0.0, 1.0, sc) == 1.0            # perfection
    assert score_one(0.5, 1.0, sc) == 0.5
    assert score_one(1.0, 1.0, sc) == 0.0            # the baseline itself
    # Below it: a line, where the tail was a parabola. -1.5 and -4.0 were the old readings.
    assert score_one(2.0, 1.0, sc) == -1.0
    assert score_one(3.0, 1.0, sc) == -2.0
    # The band that the shape change actually buys: the tail saturated at r = sqrt(13) = 3.606
    # and everything past it tied at the floor. The line ranks out to r = 7.
    assert score_one(4.0, 1.0, sc) == -3.0           # was -6.0 (saturated)
    assert score_one(6.0, 1.0, sc) == -5.0           # was -6.0 (saturated)
    # ...and the floor still binds, so one runaway submission cannot drag avg_score.
    assert score_one(7.0, 1.0, sc) == -6.0
    assert score_one(1e6, 1.0, sc) == -6.0
    # An unusable submission takes the floor, exactly as it did under the cap.
    assert score_one(None, 1.0, sc) == -6.0
    assert score_one(float("nan"), 1.0, sc) == -6.0
    assert score_one(float("inf"), 1.0, sc) == -6.0


def test_the_lfc_nmae_shape_survives_the_replicate_anchor():
    """The competition score moves the anchor off 0 onto the measured replicate (#276 part C,
    `score._replicate_entries`), so the shape has to be asserted THERE too -- `replace()`
    keeps `penalty` and `clamp_low`, and it is the anchored form that ships.

    Numbers are val A's frozen bundle geometry: replicate 0.361329, baseline 1.000421.
    """
    from dataclasses import replace

    from cell_eval2.scoring import score_one

    a, b = 0.361329, 1.000421
    sc = replace(CATALOG["de_wilcoxon_lfc_nmae"].scoring, anchor=a)
    assert score_one(0.0, b, sc) == pytest.approx(1.5654, abs=5e-5)   # a perfect submission
    assert score_one(a, b, sc) == pytest.approx(1.0)                  # the replicate anchor
    assert score_one(b, b, sc) == 0.0                                 # triviality
    assert score_one(2.0, b, sc) == pytest.approx(-1.5641, abs=5e-5)  # was -2.787 under the tail
    assert score_one(3.0, b, sc) == pytest.approx(-3.1288, abs=5e-5)  # was -6.0 (saturated)
    assert score_one(4.835, b, sc) == pytest.approx(-6.0, abs=1e-3)   # the floor, reached at r=7
    assert score_one(22.9, b, sc) == -6.0                             # last year's field median


def test_a_call_time_boxcox_override_still_reaches_the_lfc_policy():
    """`ERROR_LINEAR` keeps `direction="lower"` and a non-None anchor, so it still SATISFIES
    `score_one`'s boxcox precondition: a call-time `penalty="boxcox"` restores the old tail
    rather than raising. Pinned deliberately -- a reader could reasonably expect
    `penalty="none"` to close that door, and it does not.

    ⚠️ The cap and the floor come apart under that override, and that is pre-existing behaviour
    rather than something this change introduced: `penalty_cap` saturates the TAIL while
    `clamp_low` floors the RESULT, so with `cap=2` a bad finite value reads -2 while a
    non-finite one still takes the declared -6. The same separation is what
    `effective_clamp_low` documents for every policy with an explicit floor.
    """
    from cell_eval2.scoring import score_one

    sc = CATALOG["de_wilcoxon_lfc_nmae"].scoring
    assert score_one(2.0, 1.0, sc) == -1.0                            # the shipped line
    assert score_one(2.0, 1.0, sc, penalty="boxcox") == -1.5          # the old tail, restored
    assert score_one(20.0, 1.0, sc, penalty="boxcox") == -6.0         # saturating at the cap
    assert score_one(20.0, 1.0, sc, penalty="boxcox", penalty_cap=2.0) == -2.0
    # ...while the DECLARED floor, not the cap, still catches an unusable submission.
    assert score_one(float("nan"), 1.0, sc, penalty="boxcox", penalty_cap=2.0) == -6.0
    # And `penalty_cap` alone -- no `penalty` override -- is inert, because there is no tail.
    assert score_one(20.0, 1.0, sc, penalty_cap=1.0) == -6.0
    assert score_one(2.0, 1.0, sc, penalty_cap=1.0) == -1.0


def test_the_four_centroid_error_metrics_keep_the_box_cox_tail():
    """The companion half of the move: `ERROR` itself did NOT change. `expr_mae` is in the
    frozen 2025 `vcc` profile, so straightening its tail would move a published leaderboard."""
    from cell_eval2.scoring import ERROR, score_one

    for name in ("expr_mae", "expr_mse", "delta_mae", "delta_mse"):
        sc = CATALOG[name].scoring
        assert sc == ERROR, name
        assert sc.penalty == "boxcox", name
        assert score_one(2.0, 1.0, sc) == -1.5, name     # the quadratic, not the line
        assert score_one(4.0, 1.0, sc) == -6.0, name     # saturated well before r = 7


@pytest.mark.parametrize("suffix", [
    "direction_coverage", "direction_yield_raw", "direction_reach_unbounded",
    "direction_reach_unbounded_raw", "direction_sensitivity_universe", "direction_yield",
])
@pytest.mark.parametrize("method", ["wilcoxon", "deseq2"])
def test_the_anchorless_clamps_are_pinned(method, suffix):
    sc = CATALOG[f"de_{method}_{suffix}"].scoring
    assert (sc.clamp_low, sc.clamp_high) == (-2.0, 2.0)


def test_the_clamps_actually_truncate_the_runaway_cases():
    """Pinning the fields is not enough -- assert the NUMBER, in the regime that motivated
    the cap. Both were measured before the clamp: 99.0 and 4999.0."""
    from cell_eval2.scoring import score_one

    open_ = CATALOG["de_wilcoxon_direction_coverage"].scoring
    assert score_one(2.0, 0.02, open_) == 2.0            # raw u/b - 1 = 99
    assert score_one(0.0, 0.02, open_) == -1.0           # the metric's own range floors it
    signed = CATALOG["de_wilcoxon_direction_yield"].scoring
    assert score_one(0.5, 0.0001, signed) == 2.0         # raw (u-b)/|b| = 4999
    assert score_one(-1.0, 0.01, signed) == -2.0         # the floor binds on this one


def test_a_directly_built_spec_is_not_silently_v1_available():
    """The derivation is STRUCTURAL, not a convention `_register_de_family` follows. It was
    the latter first, and that is not enough: `MetricSpec`'s own default was True, so any
    registration outside the DE helper -- the nine anndata literals, or any future metric --
    would be offered under v1 with no name to emit it under. A catalog-wide pin cannot catch
    that, because it only sees entries that already exist.
    """
    from cell_eval2.catalog import MetricSpec

    kw = dict(func=lambda: None, scoring=BOUNDED, profiles=(), kind="de",
              normalization=None, agg="mean")
    assert MetricSpec(name="v2_native", **kw).v1_available is False
    assert MetricSpec(name="has_v1", v1_name="mae", **kw).v1_available is True
    # "" is not a usable spelling either -- _build_name_index skips it, so deriving on
    # `is not None` would claim availability under a label nothing can resolve
    assert MetricSpec(name="empty_v1", v1_name="", **kw).v1_available is False
    with pytest.raises(ValueError, match="non-empty v1_name"):
        MetricSpec(name="empty_explicit", v1_name="", v1_available=True, **kw)
    # an explicit False is still honoured, so a named metric can be withheld from v1
    assert MetricSpec(name="withheld", v1_name="mae", v1_available=False, **kw).v1_available is False
    # ...but claiming availability with no name to emit it under is refused
    with pytest.raises(ValueError, match="non-empty v1_name"):
        MetricSpec(name="nameless", v1_available=True, **kw)

    for name, spec in CATALOG.items():
        assert spec.v1_available == bool(spec.v1_name), name


def test_the_newly_enrolled_metrics_reach_no_v1_output():
    """All 20 are v2-native, which is what makes this enrolment safe; assert it rather than
    assume it.

    The hazard was concrete. compat.score_agg_metrics used to select metrics by the DERIVED
    best_value token, so enrolling these flipped it "none" -> "one" and compat began scoring
    them -- with that file byte-frozen, and using an anchor-1 formula they have no anchor for.
    compat now reads `v1_available` + `scoring` directly, but this stays the load-bearing
    property: nothing v1 can emit may change scoring class."""
    newly = {n for n, s in CATALOG.items()
             if s.scoring.scored and s.scoring.anchor is None} | {
             f"de_{m}_{s}" for m in ("wilcoxon", "deseq2")
             for s in ("direction_fidelity", "direction_fidelity_raw",
                       "direction_fidelity_yield_raw", "direction_reach_raw")}
    assert newly, "the anchorless scored set must not be empty"
    for n in newly:
        assert not CATALOG[n].v1_available, f"{n} would leak into v1 scoring"
        assert CATALOG[n].v1_name is None, n


def test_best_value_property_reproduces_todays_token():
    for name, spec in CATALOG.items():
        if not spec.scoring.scored:
            assert spec.best_value == "none", name
        elif spec.scoring.direction == "lower":
            assert spec.best_value == "zero", name
        else:
            assert spec.best_value == "one", name


def test_directionless_metrics_keep_direction_none():
    for name in DIRECTIONLESS:
        assert CATALOG[name].scoring.direction is None, name


def test_previously_none_metrics_now_record_their_direction():
    """The 24 entries that carried best_value="none" (12 suffixes x 2 methods): 20 are
    higher-is-better (and now scored), 4 are genuinely directionless (and cannot be).
    Recording the direction is what made the split between those two groups expressible at
    all. Counts are ENTRIES throughout -- the two directionless suffixes are 4 entries."""
    former_none = {f"de_{m}_{s}" for m in ("wilcoxon", "deseq2")
                   for s in ("nsig_counts_real", "nsig_counts_pred",
                             "direction_sensitivity_universe", "direction_fidelity",
                             "direction_fidelity_raw", "direction_coverage",
                             "direction_yield", "direction_yield_raw",
                             "direction_fidelity_yield_raw", "direction_reach_raw",
                             "direction_reach_unbounded", "direction_reach_unbounded_raw")}
    assert len(former_none) == 24
    with_direction = {n for n in former_none if CATALOG[n].scoring.direction == "higher"}
    assert len(with_direction) == 20
    assert "de_wilcoxon_direction_yield" in with_direction
    assert "de_wilcoxon_direction_reach_unbounded" in with_direction
    assert {n for n in former_none if CATALOG[n].scoring.direction is None} == _UNSCORED


@pytest.mark.parametrize("name,anchor", [
    ("de_wilcoxon_direction_fidelity_raw", 1.0),
    ("de_wilcoxon_direction_fidelity", 1.0),
    ("de_wilcoxon_direction_fidelity_yield_raw", 1.0),
    ("de_wilcoxon_direction_reach_raw", 1.0),
    ("de_wilcoxon_direction_coverage", None),
    ("de_wilcoxon_direction_yield", None),
    ("de_wilcoxon_direction_yield_raw", None),
    ("de_wilcoxon_direction_reach_unbounded", None),
    ("de_wilcoxon_direction_reach_unbounded_raw", None),
    ("de_wilcoxon_direction_sensitivity_universe", None),
])
def test_anchors_follow_the_spec_ranges(name, anchor):
    assert CATALOG[name].scoring.anchor == anchor


def test_only_direction_yield_allows_a_negative_baseline():
    """spec 2.2: of the twelve anchorless entries only direction_yield is genuinely
    signed. abs() on the other ten would silently score corrupt input, so this is the
    assertion that keeps the flag from spreading."""
    signed = {n for n, s in CATALOG.items() if s.scoring.allow_negative_baseline}
    assert signed == {"de_wilcoxon_direction_yield", "de_deseq2_direction_yield"}


def test_agg_is_required():
    with pytest.raises(TypeError):
        MetricSpec(name="x", func=lambda: {}, scoring=Scoring(),
                   profiles=(), kind="de", normalization=None)


def test_the_de_family_helper_also_requires_agg():
    """Criterion 11 names `add()` as well as MetricSpec: it is the funnel every future DE
    metric goes through, so a default there would preserve the exact failure this change
    prevents. The helper is a closure, so parse the source -- with `ast`, not string
    matching: `agg: Literal[...] = "mean"` contains no literal `agg=` and would sail past
    a substring check, and a signature wrapped over two lines would fail one."""
    import ast
    import inspect
    import textwrap

    from cell_eval2 import catalog as cat

    tree = ast.parse(textwrap.dedent(inspect.getsource(cat._register_de_family)))
    add = next(n for n in ast.walk(tree)
               if isinstance(n, ast.FunctionDef) and n.name == "add")
    names = [a.arg for a in add.args.kwonlyargs]
    assert "agg" in names, "add() must accept agg"
    assert add.args.kw_defaults[names.index("agg")] is None, "agg must have NO default"


# The nine chance-corrected direction suffixes that carried `agg="median"` through v0.7.0.
# Both spellings of each are pinned below: `_register_de_family` passes a single `agg` to
# each wilcoxon/deseq2 sibling pair, so a metric cannot change its aggregation statistic
# because the DE backend changed, and a "fix" that special-cased one family must not pass.
_FORMERLY_MEDIAN = tuple(
    f"de_{m}_{s}"
    for m in ("wilcoxon", "deseq2")
    for s in ("direction_fidelity", "direction_fidelity_raw", "direction_coverage",
              "direction_yield", "direction_yield_raw", "direction_fidelity_yield_raw",
              "direction_reach_raw", "direction_reach_unbounded",
              "direction_reach_unbounded_raw")
)


def test_the_catalog_has_mean_rows_and_one_ratio_of_sums():
    """Issue #231 keeps every per-perturbation entry at MEAN; #257 adds one derived ratio.

    The set equality is the invariant. The explicit "nothing is median" line follows it so
    that a regression NAMES the offenders instead of printing two whole set literals -- the
    likely regression is a new entry copy-pasted from a pre-0.8 direction metric, and the
    failure should say which one.

    `agg` remains a real field with a live `median` branch in `run.aggregate_metrics` (see
    `test_metric_aggregation.py`); the invariant is "every SHIPPED entry says mean", not
    "the median is unimplemented".
    """
    assert {s.agg for s in CATALOG.values()} == {"mean", "ratio_of_sums"}
    ratios = sorted(n for n, s in CATALOG.items() if s.agg == "ratio_of_sums")
    assert ratios == ["expr_mse_unbiased_capped_norm"]
    medians = sorted(n for n, s in CATALOG.items() if s.agg == "median")
    assert medians == [], f"these entries still aggregate by median: {medians}"


def test_the_formerly_median_entries_are_all_v1_unavailable():
    """The frozen v1 byte-identity gate cannot move under #231.

    `compat.score_agg_metrics` scores a metric only if it resolves to a v1 spelling, so an
    entry with `v1_name is None` / `v1_available is False` is invisible to it. All 18 entries
    whose statistic changed are v2-native, which is WHY switching them cannot perturb the
    frozen v1 comparison -- asserted here rather than left as prose in the plan.
    """
    assert len(_FORMERLY_MEDIAN) == 18                     # 9 suffixes x 2 families
    for name in _FORMERLY_MEDIAN:
        spec = CATALOG[name]
        assert spec.v1_name is None, name
        assert spec.v1_available is False, name


def test_lfc_nmae_is_not_in_the_2025_vcc_profile():
    """#208 ships the member OUTSIDE `vcc` deliberately. `vcc` is the 2025 competition score,
    while `vcc2026` is the 2026 competition score and makes its members decisive. The WILCOXON
    metric is therefore decisive through `vcc2026`; its deseq2 sibling stays non-decisive
    because it is a relabel-only implementation with profiles=(). The asymmetry is intended
    and asserted directly, not inferred from a shared family registration.

    Asserted on the observable consequence, not just on the tuple: `is_decisive` is the
    thing that would actually change behaviour, and a test that only read `profiles` would
    keep passing if the predicate itself were rewired.
    """
    from cell_eval2.catalog import CATALOG, is_decisive
    wilcoxon = CATALOG["de_wilcoxon_lfc_nmae"]
    deseq2 = CATALOG["de_deseq2_lfc_nmae"]
    for name, spec in (("de_wilcoxon_lfc_nmae", wilcoxon),
                       ("de_deseq2_lfc_nmae", deseq2)):
        assert "vcc" not in spec.profiles, name
        assert spec.scoring.direction == "lower"
        assert spec.scoring.anchor == 0.0
        assert spec.worst_value is None      # unbounded above -> nothing to fill with
    assert "vcc2026" in wilcoxon.profiles
    assert is_decisive(wilcoxon)
    assert deseq2.profiles == ()   # relabel-only; never profile-selected
    assert not is_decisive(deseq2)


def test_full_and_de_profiles_gain_the_scored_member():
    """#208 spec 4.5: this IS a change to those two profiles' avg_score, and their baseline
    artifacts must be regenerated. Pinned as a fact so it cannot be quietly reverted, and so
    the CHANGELOG's BREAKING note stays true."""
    from cell_eval2.catalog import CATALOG, resolve_metrics
    for profile in ("full", "de"):
        available, _ = resolve_metrics(profile)
        assert "de_wilcoxon_lfc_nmae" in available, profile
    assert CATALOG["de_wilcoxon_lfc_nmae"].scoring.scored
    vcc, _ = resolve_metrics("vcc")
    assert "de_wilcoxon_lfc_nmae" not in vcc          # vcc is NOT affected


def test_every_scored_vcc2026_member_is_decisive():
    """A competition metric whose baseline is degenerate must stop the run, not silently
    shrink avg_score's denominator (#255; the vcc2026 half of #222).

    Before this, only pds_cosine was decisive -- and only incidentally, because it happens
    to carry a v1 name. The other five were warn-and-dropped, turning a six-metric
    competition average into a five-metric one with nothing in the output recording it.
    """
    from cell_eval2.catalog import CATALOG, PROFILES, is_decisive

    scored = {n for n in PROFILES["vcc2026"] if CATALOG[n].scoring.scored}
    unscored = set(PROFILES["vcc2026"]) - scored
    assert unscored == _ANNDATA_DIAGNOSTICS
    for name in scored:
        assert is_decisive(CATALOG[name]), name
    for name in unscored:
        assert not is_decisive(CATALOG[name]), name


def test_no_deseq2_sibling_became_decisive():
    """vcc2026 is a WILCOXON profile: its members carry the profile, their deseq2 siblings
    carry profiles=(). The asymmetry is intended and is asserted so a future change to
    _register_de_family cannot widen the raise silently."""
    from cell_eval2.catalog import CATALOG, is_decisive

    for name in ("de_deseq2_lfc_nmae", "de_deseq2_sig_jaccard",
                 "de_deseq2_direction_reach_raw",
                 "de_deseq2_direction_fidelity_yield_raw"):
        assert not is_decisive(CATALOG[name]), name


# Each metric at ITS OWN degenerate boundary -- D == 0 for that metric's class. Testing only
# one member would leave the other five's raise unexercised.
@pytest.mark.parametrize("metric,bad_base", [
    ("expr_mse_unbiased_capped_norm", 0.0),                 # lower/anchor-0 -> D = base
    ("de_wilcoxon_lfc_nmae", 0.0),
    ("pds_cosine", 1.0),                                    # higher/anchor-1 -> D = 1 - base
    ("de_wilcoxon_direction_fidelity_yield_raw", 1.0),
    ("de_wilcoxon_direction_reach_raw", 1.0),
    ("de_wilcoxon_sig_jaccard", 1.0),
])
def test_a_degenerate_baseline_on_any_vcc2026_member_raises(metric, bad_base):
    """Every one of the six now stops the run. Before #255 only pds_cosine did, so a
    degenerate baseline on any other member silently made avg_score a five-metric mean."""
    import polars as pl
    import pytest as _pytest

    from cell_eval2.catalog import PROFILES
    from cell_eval2.score import score_metrics

    metrics = list(PROFILES["vcc2026"])
    # Healthy elsewhere: 0.5 is non-degenerate for both classes (D = 0.5 either way).
    user = {m: 0.5 for m in metrics}
    base = {m: 0.5 for m in metrics}
    base[metric] = bad_base

    def agg(d):
        return pl.DataFrame({"statistic": ["mean"], **{m: [d[m]] for m in metrics}})

    with _pytest.raises(ValueError, match="degenerate baseline"):
        score_metrics(agg(user), agg(base))


def test_components_only_request_scores_avg_over_expr_mae_alone():
    """#257: diagnostic components must not implicitly enrol their scored derivative."""
    import polars as pl

    from cell_eval2.run import aggregate_metrics_wide
    from cell_eval2.score import score_metrics

    request = ["expr_mae", "expr_mse_unbiased_capped", "expr_distance_unbiased"]
    derived = "expr_mse_unbiased_capped_norm"
    honest_avg = 0.5                         # expr_mae: 1 - user/base = 1 - 1/2
    wrong_derived_score = 0.0                # equal user/base derived ratios
    wrong_two_metric_avg = (honest_avg + wrong_derived_score) / 2.0
    assert honest_avg != wrong_two_metric_avg, "fixture cannot distinguish the wrong average"

    def tidy(expr_mae, numerator, denominator):
        return pl.DataFrame(
            {"perturbation": ["p1", "p1", "p1"],
             "metric": request,
             "value": [expr_mae, numerator, denominator]},
            schema={"perturbation": pl.String, "metric": pl.String, "value": pl.Float64},
        )

    user = aggregate_metrics_wide(tidy(1.0, 1.0, 4.0), metrics=request)
    base = aggregate_metrics_wide(tidy(2.0, 2.0, 8.0), metrics=request)
    assert derived not in user.columns, (
        f"components-only request injected scored column {derived} into {user.columns}"
    )

    scored = score_metrics(user, base)
    rows = dict(zip(scored["metric"].to_list(), scored["from_baseline"].to_list()))
    assert set(rows) == {"expr_mae", "avg_score"}, f"unexpected scored rows: {rows}"
    assert rows["avg_score"] == pytest.approx(honest_avg), (
        f"avg_score was {rows['avg_score']}; injected {derived} would produce the wrong "
        f"two-metric mean {wrong_two_metric_avg}, while expr_mae alone is {honest_avg}"
    )


# --- the clip-at-0 removal (four bounded `vcc2026` members) -------------------------------

#: The complete set. Named rather than derived: the whole point of the change is that it is
#: SCOPED, and a predicate would silently absorb a fifth entry someone unfloored by accident.
_UNFLOORED = {
    "pds_cosine",
    "de_wilcoxon_direction_fidelity_yield_raw",
    "de_deseq2_direction_fidelity_yield_raw",
    "de_wilcoxon_direction_reach_raw",
    "de_deseq2_direction_reach_raw",
    "de_wilcoxon_sig_jaccard",
    "de_deseq2_sig_jaccard",
}


def test_exactly_these_entries_are_unfloored():
    """`clamp_low=None` on a `penalty='none'` policy means no clip at 0. It is correct for
    the four bounded `vcc2026` members (plus the three deseq2 siblings, which must not change
    policy because the DE backend changed) and for nothing else -- 60+ other entries share
    the `BOUNDED` singleton, including the frozen 2025 `vcc` profile."""
    got = {n for n, s in CATALOG.items()
           if s.scoring.scored and s.scoring.clamp_low is None
           and s.scoring.penalty == "none"}
    assert got == _UNFLOORED


def test_every_unfloored_entry_declares_its_structural_worst_value():
    """Without `metric_min` the policy would not be constructible at all -- this asserts the
    VALUE, which `Scoring` cannot check: all seven are [0, 1] metrics, so 0.0 is the worst."""
    for name in _UNFLOORED:
        sc = CATALOG[name].scoring
        assert sc.metric_min == 0.0, name
        assert sc.anchor == 1.0 and sc.direction == "higher", name


def test_the_unfloored_entries_are_not_uniformly_decisive():
    """Pinning a fact a code comment got WRONG (codex round 1, finding 3). Four of the seven
    are decisive; the three `de_deseq2_*` siblings deliberately are not, because they are
    v2-native and outside `vcc`. What protects `score_one`'s non-degenerate-base precondition
    is therefore that every scoring CALLER checks `is_degenerate` first -- decisiveness only
    picks whether that check raises or warns-and-excludes."""
    from cell_eval2.catalog import is_decisive

    decisive = {n for n in _UNFLOORED if is_decisive(CATALOG[n])}
    assert decisive == {
        "pds_cosine",
        "de_wilcoxon_direction_fidelity_yield_raw",
        "de_wilcoxon_direction_reach_raw",
        "de_wilcoxon_sig_jaccard",
    }
    assert _UNFLOORED - decisive == {
        "de_deseq2_direction_fidelity_yield_raw",
        "de_deseq2_direction_reach_raw",
        "de_deseq2_sig_jaccard",
    }


def test_the_shared_BOUNDED_singleton_still_clips_at_zero():
    """The regression that would be silent and expensive: editing `BOUNDED` in place instead
    of minting `BOUNDED_UNFLOORED` would have moved every rank metric in `full`/`de` AND the
    frozen 2025 `vcc` competition score."""
    from cell_eval2.scoring import BOUNDED, score_one

    assert BOUNDED.clamp_low == 0.0 and BOUNDED.metric_min is None
    assert score_one(0.0, 0.5, BOUNDED) == 0.0
    still_floored = [n for n in ("pds_l1", "de_wilcoxon_overlap", "de_wilcoxon_direction_match",
                                 "de_wilcoxon_direction_fidelity_yield", "de_wilcoxon_sig_mcc",
                                 "de_wilcoxon_direction_reach")]
    for n in still_floored:
        assert CATALOG[n].scoring.clamp_low == 0.0, f"{n} lost its floor"


def test_the_frozen_v1_path_is_unaffected_by_the_clip_removal():
    """`pds_cosine` is v1-available, so unflooring it could have reached the frozen parity
    path. It does not: `compat.score_agg_metrics` carries its own hard-coded `max(0.0, ...)`
    and never reads `clamp_low`."""
    import polars as pl

    from cell_eval2.compat import score_agg_metrics

    user = pl.DataFrame({"statistic": ["mean"], "discrimination_score_cosine": [0.10]})
    base = pl.DataFrame({"statistic": ["mean"], "discrimination_score_cosine": [0.50]})
    out = score_agg_metrics(user, base)
    got = dict(zip(out["metric"].to_list(), out["from_baseline"].to_list()))
    assert got["discrimination_score_cosine"] == 0.0     # clipped, raw would be -0.8
    assert got["avg_score"] == 0.0
    # ...and the v2 engine on the same numbers now does NOT clip. That contrast is the point.
    from cell_eval2.scoring import score_one
    assert score_one(0.10, 0.50, CATALOG["pds_cosine"].scoring) == pytest.approx(-0.8)
