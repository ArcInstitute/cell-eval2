import polars as pl
import pytest

from cell_eval2.score import score_metrics

NAME = "low-random_high-1_v10"
M = ["expr_mse_unbiased_capped_norm", "de_wilcoxon_lfc_nmae", "pds_cosine",
     "de_wilcoxon_direction_fidelity_yield_raw", "de_wilcoxon_direction_reach_raw",
     "de_wilcoxon_sig_jaccard"]
RANDOM = {"expr_mse_unbiased_capped_norm": 1.0, "de_wilcoxon_lfc_nmae": 1.0, "pds_cosine": 0.5,
          "de_wilcoxon_direction_fidelity_yield_raw": 0.5,
          "de_wilcoxon_direction_reach_raw": 0.0, "de_wilcoxon_sig_jaccard": 0.0}
# A paste: every metric is at its declared anchor.
PASTE = {"expr_mse_unbiased_capped_norm": 0.0, "de_wilcoxon_lfc_nmae": 0.0, "pds_cosine": 1.0,
         "de_wilcoxon_direction_fidelity_yield_raw": 1.0,
         "de_wilcoxon_direction_reach_raw": 1.0, "de_wilcoxon_sig_jaccard": 1.0}
# ⚠️ `de_wilcoxon_direction_reach_raw`'s 0.996343 was measured at the pre-`REACH_PURITY_FLOOR` floor
# 0.975. These are FIXTURE inputs to the scale arithmetic, not assertions about the metric, so
# a stale level changes nothing under test -- but do not cite them as current metric values.
REPLICATE_CCL = {"expr_mse_unbiased_capped_norm": -1.5, "de_wilcoxon_lfc_nmae": 0.297730,
                 "pds_cosine": 0.874433,
                 "de_wilcoxon_direction_fidelity_yield_raw": 0.918961,
                 "de_wilcoxon_direction_reach_raw": 0.996343,
                 "de_wilcoxon_sig_jaccard": 0.531676}
REPLICATE_CGS = {"expr_mse_unbiased_capped_norm": 8.0, "de_wilcoxon_lfc_nmae": 0.290712,
                 "pds_cosine": 0.876433,
                 "de_wilcoxon_direction_fidelity_yield_raw": 0.909091,
                 "de_wilcoxon_direction_reach_raw": 0.971933,
                 "de_wilcoxon_sig_jaccard": 0.520372}


def _agg(values, metrics=None):
    metrics = M if metrics is None else metrics
    return pl.DataFrame({"statistic": ["mean"], **{m: [values[m]] for m in metrics}})


def _col(values, base=None, **kw):
    """Score `values` against `base` (default: the random point) and return the scale cells."""
    df = score_metrics(_agg(values), _agg(RANDOM if base is None else base),
                       scale=NAME, **kw)
    return dict(zip(df["metric"].to_list(), df[NAME].to_list()))


def test_paste_scores_exactly_one_everywhere():
    c = _col(PASTE)
    for m in M:
        assert c[m] == pytest.approx(1.0), m
    assert c["expr_mse_unbiased_capped_norm"] == pytest.approx(1.0)
    assert c["avg_score"] == pytest.approx(1.0)


def test_the_random_point_scores_exactly_zero_everywhere():
    c = _col(RANDOM)
    for m in M:
        assert c[m] == pytest.approx(0.0), m
    assert c["expr_mse_unbiased_capped_norm"] == pytest.approx(0.0)
    assert c["avg_score"] == pytest.approx(0.0)


def test_clamp_high_caps_the_signed_metric_overshoot():
    """A raw -1.5 maps to 1 - (-1.5) = 2.5. clamp_high must cap it at 1.0."""
    assert _col(REPLICATE_CCL)["expr_mse_unbiased_capped_norm"] == pytest.approx(1.0)


def test_replicate_arm_reproduces_the_recorded_numbers():
    c = _col(REPLICATE_CCL)
    assert c["pds_cosine"] == pytest.approx(0.7489, abs=1e-4)
    assert c["de_wilcoxon_lfc_nmae"] == pytest.approx(0.7023, abs=1e-4)
    assert c["de_wilcoxon_direction_fidelity_yield_raw"] == pytest.approx(0.8379, abs=1e-4)
    assert c["de_wilcoxon_direction_reach_raw"] == pytest.approx(0.9963, abs=1e-4)
    assert c["de_wilcoxon_sig_jaccard"] == pytest.approx(0.5317, abs=1e-4)
    assert c["expr_mse_unbiased_capped_norm"] == pytest.approx(1.0)   # 2.5 raw, capped
    assert c["avg_score"] == pytest.approx(0.8028461666666667)


def test_biological_replicate_arm_reproduces_the_recorded_numbers():
    """The other five are still H1_CGS's recorded numbers -- two genuine BIOLOGICAL
    replicates, so a real between-replicate difference survives the sampling correction. The
    derived metric's input is NOT a recorded number: #257 replaced it with the boundary case
    that exercises the floor, raw 8.0 -> -7.0 -> clamped to -6.0. Its recorded H1_CGS value
    will be re-measured when the baselines are regenerated."""
    c = _col(REPLICATE_CGS)
    assert c["expr_mse_unbiased_capped_norm"] == pytest.approx(-6.0)
    assert c["de_wilcoxon_lfc_nmae"] == pytest.approx(0.7093, abs=1e-4)
    assert c["pds_cosine"] == pytest.approx(0.7529, abs=1e-4)
    assert c["de_wilcoxon_direction_fidelity_yield_raw"] == pytest.approx(0.8182, abs=1e-4)
    assert c["de_wilcoxon_direction_reach_raw"] == pytest.approx(0.9719, abs=1e-4)
    assert c["de_wilcoxon_sig_jaccard"] == pytest.approx(0.5204, abs=1e-4)
    assert c["avg_score"] == pytest.approx(-0.37122649999999996)


# Each `raw` is chosen so the UNCLAMPED score lands strictly BELOW the floor -- otherwise the
# assertion passes even with clamping removed. For the four higher-is-better metrics the raw
# domain minimum (0.0) maps exactly ONTO the floor, so a below-domain value is used instead:
# pds_cosine at -0.5 gives (-0.5-0.5)/0.5 = -2, which reads -1 only because the clamp binds.
@pytest.mark.parametrize("metric,raw,floor,unclamped", [
    ("expr_mse_unbiased_capped_norm", 1e9, -6.0, -1e9 + 1),
    ("de_wilcoxon_lfc_nmae", 1e9, -1.0, -1e9 + 1),
    ("pds_cosine", -0.5, -1.0, -2.0),
    ("de_wilcoxon_direction_fidelity_yield_raw", -0.5, -1.0, -2.0),
    ("de_wilcoxon_direction_reach_raw", -0.5, 0.0, -0.5),
    ("de_wilcoxon_sig_jaccard", -0.5, 0.0, -0.5),
])
def test_each_metric_floors_at_its_own_clamp_low(metric, raw, floor, unclamped):
    assert unclamped < floor, "the fixture must put the unclamped score below the floor"
    assert _col(dict(RANDOM, **{metric: raw}))[metric] == pytest.approx(floor)


def test_the_higher_metrics_reach_their_floor_exactly_at_the_domain_minimum():
    """Separate from the clamp test above, deliberately: at raw 0 the four higher-is-better
    metrics land ON their floor by arithmetic, not by clamping. Asserting both keeps the two
    facts from being mistaken for one."""
    for metric, floor in (("pds_cosine", -1.0),
                          ("de_wilcoxon_direction_fidelity_yield_raw", -1.0),
                          ("de_wilcoxon_direction_reach_raw", 0.0),
                          ("de_wilcoxon_sig_jaccard", 0.0)):
        assert _col(dict(RANDOM, **{metric: 0.0}))[metric] == pytest.approx(floor), metric


def test_worst_reachable_avg_score_is_minus_one_point_five():
    worst = {"expr_mse_unbiased_capped_norm": 1e9, "de_wilcoxon_lfc_nmae": 1e9, "pds_cosine": 0.0,
             "de_wilcoxon_direction_fidelity_yield_raw": 0.0,
             "de_wilcoxon_direction_reach_raw": 0.0, "de_wilcoxon_sig_jaccard": 0.0}
    assert _col(worst)["avg_score"] == pytest.approx(-1.5)


def test_no_scale_leaves_the_frame_shape_untouched():
    df = score_metrics(_agg(REPLICATE_CCL), _agg(RANDOM))
    assert df.columns == ["metric", "from_baseline"]


def test_a_scale_only_adds_a_column():
    plain = score_metrics(_agg(REPLICATE_CCL), _agg(RANDOM))
    scaled = score_metrics(_agg(REPLICATE_CCL), _agg(RANDOM), scale=NAME)
    assert scaled.columns == ["metric", "from_baseline", NAME]
    assert scaled["metric"].to_list() == plain["metric"].to_list()
    assert scaled["from_baseline"].to_list() == pytest.approx(
        plain["from_baseline"].to_list())


def test_a_metric_the_scale_does_not_name_is_null_in_its_column():
    metrics = M + ["expr_mae"]
    vals = dict(REPLICATE_CCL, expr_mae=0.5)
    base = dict(RANDOM, expr_mae=1.0)
    df = score_metrics(_agg(vals, metrics), _agg(base, metrics), scale=NAME)
    c = dict(zip(df["metric"].to_list(), df[NAME].to_list()))
    assert c["expr_mae"] is None
    # ...and it must not move the scale's own average.
    assert c["avg_score"] == pytest.approx(0.8028461666666667)


def test_a_metric_the_scale_names_but_the_agg_lacks_raises():
    partial = [m for m in M if m != "pds_cosine"]
    with pytest.raises(ValueError, match="pds_cosine"):
        score_metrics(_agg(REPLICATE_CCL, partial), _agg(RANDOM, partial), scale=NAME)


def test_a_scale_named_metric_dropped_by_the_baseline_pass_still_gets_a_row():
    """VERIFIED REACHABLE (codex checkpoint-1, P0): an override with scored=False makes the
    baseline loop `continue` before appending a row, while the metric is still present in the
    aggregate -- so _scale_column's absent-from-aggregate check passes and the scale would
    silently average five metrics under a six-metric name.

    The scale's base is a constant that cannot be degenerate, so a baseline-side decision must
    never remove a scale's cell. The row is restored with a null from_baseline.
    """
    from cell_eval2.scoring import Scoring

    df = score_metrics(_agg(REPLICATE_CCL), _agg(RANDOM), scale=NAME,
                       overrides={"pds_cosine": Scoring(scored=False)})
    rows = df["metric"].to_list()
    assert "pds_cosine" in rows
    by_metric = dict(zip(rows, df[NAME].to_list()))
    assert by_metric["pds_cosine"] == pytest.approx(0.7489, abs=1e-4)
    # from_baseline is null there -- the baseline pass really did decline to score it...
    assert dict(zip(rows, df["from_baseline"].to_list()))["pds_cosine"] is None
    # ...and the scale's own average still covers all six.
    assert by_metric["avg_score"] == pytest.approx(0.8028461666666667)
    assert rows[-1] == "avg_score"


def test_unknown_scale_name_raises():
    with pytest.raises(ValueError, match="unknown scale"):
        score_metrics(_agg(RANDOM), _agg(RANDOM), scale="nope_v1")


def test_the_same_scale_requested_twice_raises():
    with pytest.raises(ValueError, match="twice"):
        score_metrics(_agg(RANDOM), _agg(RANDOM), scale=[NAME, NAME])


# (override, raw expr_mse_unbiased_capped_norm, honest cell, cell IF the global leaked).
# Every row must have honest != leaked, or it proves nothing. Two earlier versions of this
# test did not: the first passed `penalty_cap` against a `penalty="none"` policy where the cap
# is inert, and the second paired `clamp_high` with a raw value that floors at -6 regardless
# (codex checkpoint-2 P2, rounds 1 and 2). The `assert honest != leaked` below is what stops
# a third such row being added silently.
@pytest.mark.parametrize("override,raw,honest,leaked", [
    ({"clamp_low": 0.0}, 1e9, -6.0, 0.0),      # would floor the scale at 0, not its own -6
    ({"clamp_high": 0.25}, 0.0, 1.0, 0.25),    # needs a raw that SCORES 1.0, or -6 hides it
])
def test_global_overrides_do_not_reach_the_scale_column(override, raw, honest, leaked):
    """The five global knobs belong to BASELINE scoring. A scale carries its own frozen
    policy, and a CLI flag that could move it would make scales_digest() a lie."""
    assert honest != pytest.approx(leaked), "fixture cannot detect a leak"
    vals = dict(RANDOM, expr_mse_unbiased_capped_norm=raw)
    assert _col(vals)["expr_mse_unbiased_capped_norm"] == pytest.approx(honest)
    got = _col(vals, **override)["expr_mse_unbiased_capped_norm"]
    assert got == pytest.approx(honest), f"global {override} reached the scale column"


def test_a_global_boxcox_penalty_does_not_reach_a_scale_only_run():
    """`penalty="boxcox"` cannot be tested through the baseline-backed helper above: the four
    higher-is-better members reject it, so the baseline pass raises before the scale column
    exists. SCALE-ONLY mode has no baseline pass, so the global is merely unused there — which
    is the case worth pinning (codex checkpoint-2 round 2b).

    Leaked, it would replace the scale's linear tail: r = 1e9 saturates the Box-Cox tail at
    -penalty_cap = -2.0, against the scale's own -6.0 floor.
    """
    vals = dict(RANDOM, expr_mse_unbiased_capped_norm=1e9)
    df = score_metrics(_agg(vals), scale=NAME, penalty="boxcox", penalty_cap=2.0)
    got = dict(zip(df["metric"].to_list(), df[NAME].to_list()))
    assert got["expr_mse_unbiased_capped_norm"] == pytest.approx(-6.0)
    assert got["expr_mse_unbiased_capped_norm"] != pytest.approx(-2.0), "the global leaked"


def test_a_duplicate_column_for_a_scale_metric_raises():
    """Two columns resolving to one canonical metric would weight it twice in the scale's
    avg_score and take whichever came last. Measured before the fix: avg_score moved
    0.7333 -> 0.5714, with both rows carrying the ALIAS column's value (codex
    checkpoint-2 P1)."""
    user = _agg(REPLICATE_CCL).with_columns(
        pl.lit(0.6).alias("discrimination_score_cosine"))
    base = _agg(RANDOM).with_columns(
        pl.lit(0.5).alias("discrimination_score_cosine"))
    with pytest.raises(ValueError, match="two columns that both name"):
        score_metrics(user, base, scale=NAME)


def test_a_duplicate_column_the_scale_does_not_name_is_tolerated():
    """The raise above is scoped to the scale's own metrics: a malformed column elsewhere in
    a wide profile is not this function's business, and raising on it would break callers the
    scale has no stake in."""
    metrics = M + ["expr_mae", "mae"]
    vals = dict(REPLICATE_CCL, expr_mae=0.4, mae=0.4)
    base = dict(RANDOM, expr_mae=0.8, mae=0.8)
    df = score_metrics(_agg(vals, metrics), _agg(base, metrics), scale=NAME)
    got = dict(zip(df["metric"].to_list(), df[NAME].to_list()))
    assert got["avg_score"] == pytest.approx(0.8028461666666667)


def test_scale_only_with_a_mean_reference_populates_both_columns():
    ref = pl.DataFrame({"statistic": ["mean"], "nmae_ref_raw": [0.2287],
                        "n_perturbations": [254]})
    df = score_metrics(_agg(REPLICATE_CCL), scale=NAME, lfc_nmae_ref=ref)
    assert df.columns == ["metric", "from_reference", NAME]
    by_metric = dict(zip(df["metric"].to_list(), df["from_reference"].to_list()))
    # (1 - 0.297730) / (1 - 0.2287) = 0.91045...
    assert by_metric["de_wilcoxon_lfc_nmae"] == pytest.approx(0.9105, abs=1e-4)
    assert by_metric["pds_cosine"] is None


def test_scale_only_omits_the_from_baseline_column():
    df = score_metrics(_agg(REPLICATE_CCL), scale=NAME)
    assert df.columns == ["metric", NAME]


def test_scale_only_scores_the_same_numbers_as_with_a_baseline():
    with_base = score_metrics(_agg(REPLICATE_CCL), _agg(RANDOM), scale=NAME)
    alone = score_metrics(_agg(REPLICATE_CCL), scale=NAME)
    assert alone["metric"].to_list() == with_base["metric"].to_list()
    assert alone[NAME].to_list() == pytest.approx(with_base[NAME].to_list())


def test_scale_only_row_order_is_lower_then_higher_then_avg():
    """Matches the baseline path's convention so the two modes read the same, and follows
    the scale's own declaration order within each direction group."""
    rows = score_metrics(_agg(REPLICATE_CCL), scale=NAME)["metric"].to_list()
    assert rows[:2] == ["expr_mse_unbiased_capped_norm", "de_wilcoxon_lfc_nmae"]
    assert rows[2:-1] == ["pds_cosine", "de_wilcoxon_direction_fidelity_yield_raw",
                          "de_wilcoxon_direction_reach_raw", "de_wilcoxon_sig_jaccard"]
    assert rows[-1] == "avg_score"


def test_neither_baseline_nor_scale_raises():
    with pytest.raises(ValueError, match="nothing to score against"):
        score_metrics(_agg(REPLICATE_CCL))


def test_scale_only_still_refuses_a_non_mean_statistic_with_a_reference():
    """The lfc_nmae_ref guard is about a RATIO OF TWO MEANS, not about the baseline, so it
    must fire in scale-only mode too. Left inside the baseline branch this call would
    silently compute (1 - user_std) / (1 - ref_mean) and label it a scaled score."""
    ref = pl.DataFrame({"statistic": ["mean"], "nmae_ref_raw": [0.2287],
                        "n_perturbations": [254]})
    user = pl.DataFrame({"statistic": ["mean", "std"],
                         **{m: [REPLICATE_CCL[m], 0.1] for m in M}})
    with pytest.raises(ValueError, match="comparison_statistic"):
        score_metrics(user, scale=NAME, comparison_statistic="std", lfc_nmae_ref=ref)
