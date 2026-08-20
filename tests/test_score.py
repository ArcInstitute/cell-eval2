import polars as pl
import pytest

from cell_eval2.compat import score_agg_metrics
from cell_eval2.score import score_metrics
from cell_eval2.scoring import DEFAULT_PENALTY_CAP as C
from cell_eval2.scoring import DEFAULT_PENALTY_EXPONENT as P
from cell_eval2.scoring import ERROR, score_one


def _pz(user, base):
    """The engine standing in for the deleted score._penalized_zero: the ERROR class on
    the shipped knobs IS that function, so the curve/C1/monotonicity/cap assertions below
    are exactly the properties the migration must preserve."""
    return score_one(user, base, ERROR, penalty_exponent=P, penalty_cap=C)

# (r, expected s) with base fixed at 1.0 so user == r. Machine-checked anchors.
_CURVE = [
    (0.0, 1.0), (0.5, 0.5), (1.0, 0.0),
    (1.1, -0.105), (1.5, -0.625), (2.0, -1.5),
    (3.0, -4.0), (4.0, -6.0),
    (5.0, -6.0), (10.0, -6.0), (100.0, -6.0),
]


@pytest.mark.parametrize("r,expected", _CURVE)
def test_penalized_zero_curve(r, expected):
    assert _pz(r, 1.0) == pytest.approx(expected)


def test_penalized_zero_positive_region_bit_identical():
    # r <= 1 must return the SAME fp expression as compat._norm_by_zero: 1 - user/base
    for user, base in [(0.3, 1.2), (0.0, 2.0), (1.0, 1.0), (0.999, 1.0)]:
        assert _pz(user, base) == 1.0 - (user / base)


def test_penalized_zero_c1_at_r1():
    # both one-sided slopes -> -1 at r = 1
    h = 1e-6
    left = (_pz(1.0, 1.0) - _pz(1.0 - h, 1.0)) / h
    right = (_pz(1.0 + h, 1.0) - _pz(1.0, 1.0)) / h
    assert abs(left - (-1.0)) < 1e-4
    assert abs(right - (-1.0)) < 1e-4


def test_penalized_zero_strictly_decreasing_then_capped():
    rs = [i / 100.0 for i in range(1, 361)]  # (0, 3.60], just below sqrt(13)
    ss = [_pz(r, 1.0) for r in rs]
    assert all(a > b for a, b in zip(ss, ss[1:]))          # strictly decreasing
    assert _pz(3.61, 1.0) == -C                            # capped just past sqrt(13)
    assert _pz(1000.0, 1.0) == -C


def test_penalized_zero_cap_exact():
    # 1e300 also verifies the OverflowError guard: (1e300)**2 overflows Python
    # floats, but the value is far past the cap, so it must return -C, not raise.
    for r in (5.0, 10.0, 100.0, 1e300):
        assert _pz(r, 1.0) == -C                            # exact, not approx


def test_penalized_zero_nonfinite_user_takes_cap():
    for bad in (float("inf"), float("nan"), None):
        assert _pz(bad, 1.0) == -C


# ---- score_metrics: happy path, penalty, ordering, bit-identity ----

def test_score_metrics_zero_metric_below_baseline_matches_clip():
    user = pl.DataFrame({"statistic": ["mean"], "mae": [0.5]})
    base = pl.DataFrame({"statistic": ["mean"], "mae": [1.0]})
    out = score_metrics(user, base)
    assert out.filter(pl.col("metric") == "mae")["from_baseline"].item() == pytest.approx(0.5)
    assert out.filter(pl.col("metric") == "avg_score")["from_baseline"].item() == pytest.approx(0.5)


def test_score_metrics_zero_metric_above_baseline_is_penalized():
    # user = 2 * base -> r = 2 -> penalty -1.5 (frozen would clip to 0.0)
    user = pl.DataFrame({"statistic": ["mean"], "mae": [2.0]})
    base = pl.DataFrame({"statistic": ["mean"], "mae": [1.0]})
    out = score_metrics(user, base)
    assert out.filter(pl.col("metric") == "mae")["from_baseline"].item() == pytest.approx(-1.5)


def test_score_metrics_nonfinite_user_takes_cap():
    user = pl.DataFrame({"statistic": ["mean"], "mae": [float("nan")]})
    base = pl.DataFrame({"statistic": ["mean"], "mae": [1.0]})
    out = score_metrics(user, base)
    assert out.filter(pl.col("metric") == "mae")["from_baseline"].item() == -C


def test_score_metrics_bounded_metric_still_clips_at_zero():
    # discrimination_score_l1 is best_value="one"; user < base -> clipped to 0.0
    user = pl.DataFrame({"statistic": ["mean"], "discrimination_score_l1": [0.3]})
    base = pl.DataFrame({"statistic": ["mean"], "discrimination_score_l1": [0.6]})
    out = score_metrics(user, base)
    assert out.filter(pl.col("metric") == "discrimination_score_l1")["from_baseline"].item() == 0.0


def test_score_metrics_row_order_zero_then_one_then_avg():
    user = pl.DataFrame({"statistic": ["mean"], "mae": [0.5], "pearson_delta": [0.5],
                         "mse": [0.5], "discrimination_score_l1": [0.5]})
    base = pl.DataFrame({"statistic": ["mean"], "mae": [1.0], "pearson_delta": [0.0],
                         "mse": [1.0], "discrimination_score_l1": [0.0]})
    out = score_metrics(user, base)
    assert out["metric"].to_list() == ["mae", "mse", "pearson_delta",
                                       "discrimination_score_l1", "avg_score"]


def _bits(x):
    """The float's 64-bit pattern. `==` cannot separate -0.0 from +0.0, so a test that
    calls itself bit-identical while comparing with `==` is claiming more than it checks --
    and the sign of zero is exactly what the clip's `<=` boundary decides."""
    import struct
    return struct.pack("<d", float(x))


def _assert_frame_bit_identical(a, b):
    assert a.columns == b.columns          # #208: the helper never checked SHAPE
    assert a["metric"].to_list() == b["metric"].to_list()
    assert a["from_baseline"].to_list() == b["from_baseline"].to_list()  # exact ==
    assert [_bits(v) for v in a["from_baseline"]] == [_bits(v) for v in b["from_baseline"]]


def test_frame_bit_identity_helper_separates_signed_zero():
    """The helper above is only worth having if it fails where `==` passes."""
    a = pl.DataFrame({"metric": ["m"], "from_baseline": [0.0]})
    b = pl.DataFrame({"metric": ["m"], "from_baseline": [-0.0]})
    assert a["from_baseline"].to_list() == b["from_baseline"].to_list()   # `==` cannot tell
    with pytest.raises(AssertionError):
        _assert_frame_bit_identical(a, b)


def test_signed_zero_at_the_clip_boundary_matches_the_frozen_scorer():
    """u = -0.0, b = 0.0 on an anchor-1 metric: the RAW score is (-0.0-0.0)/(1.0-0.0) =
    -0.0, so the frozen max(0.0, .) and the engine's `x <= low -> low` are both being asked
    to normalize a genuine negative zero. u == b would not test this -- its raw score is
    +0.0 already, and the case would pass even if the scoring path leaked -0.0."""
    user = pl.DataFrame({"statistic": ["mean"], "discrimination_score_l1": [-0.0]})
    base = pl.DataFrame({"statistic": ["mean"], "discrimination_score_l1": [0.0]})
    assert _bits((-0.0 - 0.0) / (1.0 - 0.0)) == _bits(-0.0)   # the raw score really is -0.0
    out = score_metrics(user, base)
    _assert_frame_bit_identical(out, score_agg_metrics(user, base))
    assert _bits(out["from_baseline"][0]) == _bits(0.0)       # ...and both normalize it


def test_score_metrics_bit_identical_when_no_penalty_engages():
    # Every zero-metric at/below baseline, arbitrary one-metrics -> identical to frozen.
    user = pl.DataFrame({"statistic": ["mean"], "mae": [0.5], "mse": [1.0],
                         "pearson_delta": [0.9], "discrimination_score_l1": [0.2]})
    base = pl.DataFrame({"statistic": ["mean"], "mae": [1.0], "mse": [1.0],
                         "pearson_delta": [0.1], "discrimination_score_l1": [0.6]})
    _assert_frame_bit_identical(score_metrics(user, base), score_agg_metrics(user, base))


def test_score_metrics_bit_identical_one_metric_above_baseline():
    # one-metrics are never penalized: even user > base stays identical to frozen.
    user = pl.DataFrame({"statistic": ["mean"], "discrimination_score_l1": [0.9]})
    base = pl.DataFrame({"statistic": ["mean"], "discrimination_score_l1": [0.6]})
    _assert_frame_bit_identical(score_metrics(user, base), score_agg_metrics(user, base))


# ---- degenerate baseline fail-loud + input-validation parity ----

def test_score_metrics_degenerate_zero_baseline_raises():
    # base <= 0 for mae (best_value="zero") -> fail loud, naming the metric.
    # Negative is as invalid as zero for an error metric (MAE/MSE are >= 0).
    for bad_base in (0.0, -0.5):
        user = pl.DataFrame({"statistic": ["mean"], "mae": [0.5]})
        base = pl.DataFrame({"statistic": ["mean"], "mae": [bad_base]})
        with pytest.raises(ValueError, match="mae"):
            score_metrics(user, base)


def test_score_metrics_nonfinite_baseline_raises():
    user = pl.DataFrame({"statistic": ["mean"], "mae": [0.5]})
    base = pl.DataFrame({"statistic": ["mean"], "mae": [float("inf")]})
    with pytest.raises(ValueError, match="mae"):
        score_metrics(user, base)


def test_bounded_degenerate_baseline_now_fails_loud():
    """base == 1 on an anchor-1 metric used to make _norm_by_one return NaN, which was
    clipped to 0.0 -- every submission scoring exactly 0 forever, silently. score_metrics
    now raises; compat.score_agg_metrics keeps the frozen 0.0 (see test_compat.py)."""
    user = pl.DataFrame({"statistic": ["mean"], "pds_l1": [0.8]})
    base = pl.DataFrame({"statistic": ["mean"], "pds_l1": [1.0]})
    with pytest.raises(ValueError, match="degenerate baseline"):
        score_metrics(user, base)


def test_score_metrics_column_mismatch_raises():
    user = pl.DataFrame({"statistic": ["mean"], "mae": [0.5]})
    base = pl.DataFrame({"statistic": ["mean"], "mse": [1.0]})
    with pytest.raises(ValueError, match="columns"):
        score_metrics(user, base)


def test_score_metrics_missing_statistic_raises():
    user = pl.DataFrame({"statistic": ["mean"], "mae": [0.5]})
    base = pl.DataFrame({"statistic": ["mean"], "mae": [1.0]})
    with pytest.raises(ValueError, match="comparison_statistic"):
        score_metrics(user, base, comparison_statistic="median")


def test_score_metrics_base_missing_statistic_raises():
    user = pl.DataFrame({"statistic": ["mean", "std"], "mae": [0.5, 0.1]})
    base = pl.DataFrame({"statistic": ["mean"], "mae": [1.0]})
    with pytest.raises(ValueError, match="baseline"):
        score_metrics(user, base, comparison_statistic="std")


# ---- public export ----

def test_score_metrics_is_public_export():
    import cell_eval2
    assert "score_metrics" in cell_eval2.__all__
    assert cell_eval2.score_metrics is score_metrics


# ---- compare_vcc scorer selection ----

def test_compare_vcc_select_scorer(monkeypatch):
    # `tools/vccval/` is internal and does NOT travel to the public tree, so the import is
    # guarded rather than assumed: here it resolves and the mapping is asserted; where the tool
    # is absent the test skips instead of failing.
    # `exc_type` is passed EXPLICITLY, not left to the default: pytest 8.x skips on any
    # ImportError and pytest 9 narrowed that to ModuleNotFoundError, and `pytest>=8.2` is what
    # this repo pins. With it pinned here, a vccval that exists and raises anything OTHER than
    # ModuleNotFoundError fails loudly on every supported pytest instead of skipping on some of
    # them. What this does NOT separate is a missing TRANSITIVE dependency, which arrives as a
    # ModuleNotFoundError of its own and therefore still skips --
    # accepted deliberately: the alternative is a bespoke `exc.name` check, and
    # internal:tools/gate_manifest.py's scanner only recognises `pytest.importorskip`, so a hand-rolled
    # guard would make this gate invisible to the coverage manifest.
    monkeypatch.syspath_prepend("tools")
    _select_scorer = pytest.importorskip(
        "vccval.compare_vcc", exc_type=ModuleNotFoundError)._select_scorer
    from cell_eval2.compat import score_agg_metrics as clip_scorer
    assert _select_scorer("clip") is clip_scorer
    assert _select_scorer("penalty") is score_metrics
    with pytest.raises(ValueError):
        _select_scorer("bogus")


# ---- config-knob validation + path-like inputs ----

def test_score_metrics_invalid_penalty_params_raise():
    user = pl.DataFrame({"statistic": ["mean"], "mae": [0.5]})
    base = pl.DataFrame({"statistic": ["mean"], "mae": [1.0]})
    with pytest.raises(ValueError, match="penalty_exponent"):
        score_metrics(user, base, penalty_exponent=0.0)
    with pytest.raises(ValueError, match="penalty_cap"):
        score_metrics(user, base, penalty_cap=0.0)


def test_score_metrics_accepts_pathlike(tmp_path):
    up, bp = tmp_path / "user.csv", tmp_path / "base.csv"
    pl.DataFrame({"statistic": ["mean"], "mae": [2.0]}).write_csv(up)
    pl.DataFrame({"statistic": ["mean"], "mae": [1.0]}).write_csv(bp)
    out = score_metrics(up, bp)   # pathlib.Path inputs (not str) -> r=2 -> penalty -1.5
    assert out.filter(pl.col("metric") == "mae")["from_baseline"].item() == pytest.approx(-1.5)


def test_scaled_score_is_one_minus_nmae_over_one_minus_ref(tmp_path):
    """score = (1 - mean nmae) / (1 - mean nmae_ref_RAW). Aggregate first, divide once.

    #276 part B: the denominator is the RAW reference, not the sqrt(2)-corrected one --
    the ref frame below carries both, so a regression to the old column is visible as a
    value change rather than as a missing key."""
    import polars as pl
    from cell_eval2.score import score_metrics

    user = pl.DataFrame({"statistic": ["mean"], "de_wilcoxon_lfc_nmae": [0.68]})
    base = pl.DataFrame({"statistic": ["mean"], "de_wilcoxon_lfc_nmae": [0.96]})
    ref = pl.DataFrame({"statistic": ["mean"], "nmae_ref_raw": [0.6], "nmae_ref_sqrt2": [0.4],
                        "n_perturbations": [100]})
    out = score_metrics(user, base, lfc_nmae_ref=ref)
    row = out.filter(pl.col("metric") == "de_wilcoxon_lfc_nmae").row(0, named=True)
    assert row["from_reference"] == pytest.approx((1 - 0.68) / (1 - 0.6), abs=1e-12)
    # and DISTINGUISHABLE from the old arithmetic, so this assertion can fail
    assert row["from_reference"] != pytest.approx((1 - 0.68) / (1 - 0.4), abs=1e-12)


def test_scaled_score_is_zero_when_mean_nmae_is_one():
    """Mean nmae == 1 must map to exactly 0 on the scaled scale -- the property the rescaling
    exists to give. An all-zero predicted-LFC table is ONE construction that attains it, not
    the only one: a uniform c x real prediction reads |c - 1|, so c = 2 gets there as well
    (test_de_lfc_nmae.py::test_uniform_scaling_is_linear_in_abs_c_minus_one). It is NOT 0 on
    from_baseline, which measures against the deployed baseline instead; the two have
    different zeros. And a submission that emits the control need not reach nmae == 1 at
    all (#286)."""
    import polars as pl
    from cell_eval2.score import score_metrics

    user = pl.DataFrame({"statistic": ["mean"], "de_wilcoxon_lfc_nmae": [1.0]})
    base = pl.DataFrame({"statistic": ["mean"], "de_wilcoxon_lfc_nmae": [0.96]})
    ref = pl.DataFrame({"statistic": ["mean"], "nmae_ref_raw": [0.6], "nmae_ref_sqrt2": [0.4],
                        "n_perturbations": [100]})
    out = score_metrics(user, base, lfc_nmae_ref=ref)
    row = out.filter(pl.col("metric") == "de_wilcoxon_lfc_nmae").row(0, named=True)
    assert row["from_reference"] == 0.0
    assert row["from_baseline"] != 0.0        # the two zeros are genuinely different


def test_scaled_score_above_one_is_reported_not_clipped():
    """Beating a noisy replicate is attainable and is a result, not an error."""
    import polars as pl
    from cell_eval2.score import score_metrics

    user = pl.DataFrame({"statistic": ["mean"], "de_wilcoxon_lfc_nmae": [0.1]})
    base = pl.DataFrame({"statistic": ["mean"], "de_wilcoxon_lfc_nmae": [0.96]})
    ref = pl.DataFrame({"statistic": ["mean"], "nmae_ref_raw": [0.99], "nmae_ref_sqrt2": [0.7],
                        "n_perturbations": [100]})
    out = score_metrics(user, base, lfc_nmae_ref=ref)
    row = out.filter(pl.col("metric") == "de_wilcoxon_lfc_nmae").row(0, named=True)
    assert row["from_reference"] == pytest.approx(0.9 / 0.01, abs=1e-12)
    assert row["from_reference"] > 1.0


def test_degenerate_reference_reports_unrescaled_and_warns(caplog, tmp_path):
    """mean nmae_ref_raw >= 1 -> 1 - nmae_ref_raw is not positive. Report the unrescaled value,
    never divide by it -- a negative denominator would INVERT the ranking silently."""
    import polars as pl
    from cell_eval2.score import score_metrics

    user = pl.DataFrame({"statistic": ["mean"], "de_wilcoxon_lfc_nmae": [0.68]})
    base = pl.DataFrame({"statistic": ["mean"], "de_wilcoxon_lfc_nmae": [0.96]})
    ref = pl.DataFrame({"statistic": ["mean"], "nmae_ref_raw": [1.5], "nmae_ref_sqrt2": [1.06],
                        "n_perturbations": [100]})
    # Passed as a PATH, so the test also exercises capture-before-read: the source label
    # must be taken before `pl.read_csv` discards it.
    path = tmp_path / "ref_agg.csv"
    ref.write_csv(path)
    with caplog.at_level("WARNING"):
        out = score_metrics(user, base, lfc_nmae_ref=str(path))
    row = out.filter(pl.col("metric") == "de_wilcoxon_lfc_nmae").row(0, named=True)
    assert row["from_reference"] == pytest.approx(1 - 0.68, abs=1e-12)
    # The rendered message, not merely `caplog.records` -- it must name the SOURCE and the
    # measured value, which is what makes the warning actionable in a multi-context run.
    rendered = " ".join(r.getMessage() for r in caplog.records)
    assert "degenerate" in rendered and "1.5" in rendered and "ref_agg.csv" in rendered


def test_passing_a_reference_does_not_change_avg_score():
    """avg_score keeps averaging from_baseline over the enrolled metrics, so PASSING A
    REFERENCE cannot change any score.

    That is the narrow claim and it is the only one this test proves. Registering the
    member at all DOES move avg_score for `full` and `de` and does require their baselines
    to be regenerated (spec 4.5) -- a catalog-composition change this test cannot see. An
    earlier draft named this test as though it proved the broader claim.
    """
    import polars as pl
    from cell_eval2.score import score_metrics

    user = pl.DataFrame({"statistic": ["mean"], "de_wilcoxon_lfc_nmae": [0.68],
                         "de_wilcoxon_overlap": [0.4]})
    base = pl.DataFrame({"statistic": ["mean"], "de_wilcoxon_lfc_nmae": [0.96],
                         "de_wilcoxon_overlap": [0.2]})
    ref = pl.DataFrame({"statistic": ["mean"], "nmae_ref_raw": [0.6], "nmae_ref_sqrt2": [0.4],
                        "n_perturbations": [100]})
    without = score_metrics(user, base)
    with_ref = score_metrics(user, base, lfc_nmae_ref=ref)
    a = without.filter(pl.col("metric") == "avg_score")["from_baseline"][0]
    b = with_ref.filter(pl.col("metric") == "avg_score")["from_baseline"][0]
    assert a == b


def test_no_reference_means_no_extra_column():
    """Without a reference the frame is EXACTLY as it is today. tests/test_cli_baseline.py
    asserts `df.columns == ["metric", "from_baseline"]` on the score CLI's output, and
    score_metrics' frame shape is pinned against the frozen compat.score_agg_metrics -- an
    unconditional third column breaks both."""
    import polars as pl
    from cell_eval2.score import score_metrics

    user = pl.DataFrame({"statistic": ["mean"], "de_wilcoxon_lfc_nmae": [0.68]})
    base = pl.DataFrame({"statistic": ["mean"], "de_wilcoxon_lfc_nmae": [0.96]})
    out = score_metrics(user, base)
    assert out.columns == ["metric", "from_baseline"]


def test_non_mean_comparison_statistic_with_a_reference_raises():
    """The scaled score is a ratio of two MEANS. The user side comes from whichever
    statistic the caller selected while the reference frame only carries a mean, so
    comparison_statistic="std" would compute (1 - user_std)/(1 - ref_mean) and label it a
    rescaled score. Refuse rather than emit a number that is not what it says."""
    import polars as pl
    from cell_eval2.score import score_metrics

    user = pl.DataFrame({"statistic": ["mean", "std"], "de_wilcoxon_lfc_nmae": [0.68, 0.2]})
    base = pl.DataFrame({"statistic": ["mean", "std"], "de_wilcoxon_lfc_nmae": [0.96, 0.3]})
    ref = pl.DataFrame({"statistic": ["mean"], "nmae_ref_raw": [0.6], "nmae_ref_sqrt2": [0.4],
                        "n_perturbations": [100]})
    with pytest.raises(ValueError, match="mean"):
        score_metrics(user, base, comparison_statistic="std", lfc_nmae_ref=ref)
    score_metrics(user, base, comparison_statistic="std")     # fine without a reference


def test_empty_reference_leaves_from_reference_null_and_does_not_raise(caplog, tmp_path):
    """compute_lfc_nmae_reference returns a null nmae_ref_raw when nothing cleared the gate
    (spec 4.4). That is a data outcome, not a caller error: leave the column null, warn, and
    keep scoring every other metric."""
    import polars as pl
    from cell_eval2.score import score_metrics

    user = pl.DataFrame({"statistic": ["mean"], "de_wilcoxon_lfc_nmae": [0.68],
                         "de_wilcoxon_overlap": [0.4]})
    base = pl.DataFrame({"statistic": ["mean"], "de_wilcoxon_lfc_nmae": [0.96],
                         "de_wilcoxon_overlap": [0.2]})
    ref = pl.DataFrame({"statistic": ["mean"], "nmae_ref_raw": [None], "nmae_ref_sqrt2": [None],
                        "n_perturbations": [0]},
                       schema={"statistic": pl.Utf8, "nmae_ref_raw": pl.Float64,
                               "nmae_ref_sqrt2": pl.Float64, "n_perturbations": pl.Int64})
    path = tmp_path / "empty_ref_agg.csv"      # a PATH, so the source label is exercised
    ref.write_csv(path)
    with caplog.at_level("WARNING"):
        out = score_metrics(user, base, lfc_nmae_ref=str(path))
    assert out["from_reference"].null_count() == out.height
    assert out.filter(pl.col("metric") == "avg_score").height == 1     # still scored
    rendered = " ".join(r.getMessage() for r in caplog.records)
    assert "scored no perturbations" in rendered and "empty_ref_agg.csv" in rendered


@pytest.mark.parametrize("ref,match", [
    ({"statistic": ["mean"], "nmae_ref_raw": [float("nan")], "n_perturbations": [5]},
     "non-finite"),
    ({"statistic": ["mean"], "nmae_ref_raw": [float("inf")], "n_perturbations": [5]},
     "non-finite"),
    ({"statistic": ["mean"], "nmae_ref_raw": [-0.1], "n_perturbations": [5]}, "negative"),
    ({"statistic": ["mean"], "nmae_ref_raw": [None], "n_perturbations": [5]},
     "null nmae_ref_raw but n_perturbations"),
    ({"statistic": ["mean"], "nmae_ref_raw": [None], "n_perturbations": [None]},
     "non-integer n_perturbations"),
    ({"statistic": ["mean"], "nmae_ref_raw": [0.4], "n_perturbations": [0]},
     "n_perturbations"),
    ({"statistic": ["mean"], "nmae_ref_raw": [0.4], "n_perturbations": [-1]},
     "n_perturbations"),
])
def test_reference_value_domain_is_enforced(ref, match):
    """'Empty' is defined as NULL with n_perturbations == 0 -- nothing else. A NaN, an inf,
    a negative mean, or a null contradicted by a non-zero count are all corrupt input and
    must raise rather than quietly degrade to "no scaled score"."""
    import polars as pl
    from cell_eval2.score import score_metrics

    user = pl.DataFrame({"statistic": ["mean"], "de_wilcoxon_lfc_nmae": [0.68]})
    base = pl.DataFrame({"statistic": ["mean"], "de_wilcoxon_lfc_nmae": [0.96]})
    with pytest.raises(ValueError, match=match):
        score_metrics(user, base, lfc_nmae_ref=pl.DataFrame(ref))


@pytest.mark.parametrize("bad,match", [
    ({"statistic": ["mean"], "wrong": [0.4]}, "missing column"),
    # carries only the CORRECTED column -> the raw-column guard must reject it
    ({"statistic": ["mean"], "nmae_ref_sqrt2": [0.4]}, "missing column"),
    ({"nmae_ref_sqrt2": [0.4], "n_perturbations": [5]}, "missing column"),  # no statistic
    ({"statistic": ["median"], "nmae_ref_raw": [0.4], "n_perturbations": [5]}, "exactly one"),
    ({"statistic": ["mean", "mean"], "nmae_ref_raw": [0.4, 0.5],
      "n_perturbations": [5, 5]}, "exactly one"),
])
def test_malformed_reference_raises(bad, match):
    """A malformed reference IS a caller error and must raise, unlike an empty one. A
    missing `statistic` in particular must raise on its OWN terms, not surface as an
    incidental polars column error."""
    import polars as pl
    from cell_eval2.score import score_metrics

    user = pl.DataFrame({"statistic": ["mean"], "de_wilcoxon_lfc_nmae": [0.68]})
    base = pl.DataFrame({"statistic": ["mean"], "de_wilcoxon_lfc_nmae": [0.96]})
    with pytest.raises(ValueError, match=match):
        score_metrics(user, base, lfc_nmae_ref=pl.DataFrame(bad))


def test_lfc_nmae_metric_set_lives_in_the_catalog():
    """It moved out of score.py so `anchor.py` can read it without closing an
    anchor <-> score import cycle (Task 10 makes score import anchor). score.py keeps
    re-exporting the SAME object, so no existing reader moves."""
    from cell_eval2 import catalog, score

    assert catalog._LFC_NMAE_METRICS == ("de_wilcoxon_lfc_nmae", "de_deseq2_lfc_nmae")
    assert score._LFC_NMAE_METRICS is catalog._LFC_NMAE_METRICS
    # every member must be a real catalog entry, or the substitution silently matches nothing
    for name in catalog._LFC_NMAE_METRICS:
        assert name in catalog.CATALOG, f"{name!r} is not a catalog entry"


def test_from_reference_divides_by_the_RAW_reference():
    """(1 - nmae) / (1 - nmae_ref_raw). Dividing by the sqrt(2) column instead reads 17-23%
    lower (measured across six real cell lines)."""
    import polars as pl

    from cell_eval2.score import _from_reference_column

    ref = pl.DataFrame({"statistic": ["mean"], "nmae_ref_raw": [0.40],
                        "nmae_ref_sqrt2": [0.40 / 2 ** 0.5], "n_perturbations": [7]})
    out = _from_reference_column(
        ["de_wilcoxon_lfc_nmae"], [0.50], ["de_wilcoxon_lfc_nmae"], ref)
    assert out.to_list()[0] == pytest.approx((1 - 0.50) / (1 - 0.40))
    # and is DISTINGUISHABLE from the old arithmetic, so this can fail
    assert out.to_list()[0] != pytest.approx((1 - 0.50) / (1 - 0.40 / 2 ** 0.5))


def test_from_reference_rejects_a_frame_missing_the_raw_column():
    import polars as pl

    from cell_eval2.score import _from_reference_column

    ref = pl.DataFrame({"statistic": ["mean"], "nmae_ref_sqrt2": [0.28],
                        "n_perturbations": [7]})
    with pytest.raises(ValueError, match="nmae_ref_raw"):
        _from_reference_column(["de_wilcoxon_lfc_nmae"], [0.5],
                               ["de_wilcoxon_lfc_nmae"], ref)
