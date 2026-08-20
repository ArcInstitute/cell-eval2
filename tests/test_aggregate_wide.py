import math

import polars as pl
import pytest

from cell_eval2 import EvalConfig, aggregate_metrics_wide, compute_metrics, score_metrics
from cell_eval2.run import aggregate_metrics, metric_output_names

_STATS = ["count", "null_count", "mean", "std", "min", "max", "median"]


def _tidy(rows):
    return pl.DataFrame(
        {"perturbation": [r[0] for r in rows],
         "metric": [r[1] for r in rows],
         "value": [r[2] for r in rows]},
        schema={"perturbation": pl.Utf8, "metric": pl.Utf8, "value": pl.Float64},
    )


def test_shape_and_statistic_rows():
    wide = aggregate_metrics_wide(_tidy([
        ("A", "expr_mae", 1.0), ("B", "expr_mae", 3.0),
        ("A", "pds_cosine", 0.5), ("B", "pds_cosine", 0.7),
    ]))
    assert wide["statistic"].to_list() == _STATS
    # metric columns sorted ascending; 'statistic' first
    assert wide.columns == ["statistic", "expr_mae", "pds_cosine"]
    row = {s: v for s, v in zip(wide["statistic"], wide["expr_mae"])}
    assert row["count"] == 2.0
    assert row["null_count"] == 0.0
    assert row["mean"] == 2.0
    assert row["std"] == pytest.approx(math.sqrt(2.0))   # sample std of [1, 3]
    assert row["min"] == 1.0
    assert row["max"] == 3.0


def test_mean_skips_nan_and_DIFFERS_from_describe():
    """Discriminating: assert the NaN-skipping value AND that describe() would differ.

    describe() propagates NaN, which would null out the whole metric and then silently
    clip every best_value='one' submission to 0.0 in score_metrics. A test that only
    asserted 'mean == 2.0' would still pass if someone later swapped in describe() on a
    frame that happened to contain no NaN, so the divergence is asserted directly.
    """
    tidy = _tidy([("A", "expr_mae", 1.0), ("B", "expr_mae", 3.0),
                  ("C", "expr_mae", float("nan"))])
    wide = aggregate_metrics_wide(tidy)
    mean = dict(zip(wide["statistic"], wide["expr_mae"]))["mean"]
    assert mean == 2.0                      # NaN-skipping: (1+3)/2

    # the rejected implementation, computed here so the test discriminates
    describe_mean = (
        tidy.pivot(index="perturbation", on="metric", values="value")
        .drop("perturbation").describe()
        .filter(pl.col("statistic") == "mean")["expr_mae"][0]
    )
    assert math.isnan(describe_mean)        # would have poisoned the metric
    assert mean != describe_mean


def test_matches_aggregate_metrics_mean():
    """The long and wide aggregates must agree on 'mean' — they are the same statistic."""
    tidy = _tidy([("A", "expr_mae", 1.0), ("B", "expr_mae", float("nan")),
                  ("C", "expr_mae", 5.0), ("A", "pds_cosine", 0.25)])
    long = aggregate_metrics(tidy)
    wide = aggregate_metrics_wide(tidy)
    wide_mean = {m: dict(zip(wide["statistic"], wide[m]))["mean"]
                 for m in wide.columns if m != "statistic"}
    for r in long.iter_rows(named=True):
        assert wide_mean[r["metric"]] == pytest.approx(r["mean"], nan_ok=True)


def test_all_nan_metric_aggregates_to_nan_not_null():
    wide = aggregate_metrics_wide(_tidy([("A", "expr_mae", float("nan")),
                                        ("B", "expr_mae", float("nan"))]))
    mean = dict(zip(wide["statistic"], wide["expr_mae"]))["mean"]
    assert mean is not None and math.isnan(mean)


def test_std_is_NAN_when_undefined_not_zero():
    """polars .std() is the SAMPLE std: undefined for one observation and for none alike.
    Reporting 0.0 (as the frozen internal:tools/vccval/make_baseline.py:90 does) is a false claim of
    zero spread, and score_metrics accepts comparison_statistic='std'."""
    one = aggregate_metrics_wide(_tidy([("A", "expr_mae", 1.0)]))
    assert math.isnan(dict(zip(one["statistic"], one["expr_mae"]))["std"])
    # genuinely-zero spread still reports 0.0, so the NaN is not a blanket
    flat = aggregate_metrics_wide(_tidy([("A", "expr_mae", 2.0), ("B", "expr_mae", 2.0)]))
    assert dict(zip(flat["statistic"], flat["expr_mae"]))["std"] == 0.0


def test_absent_expected_metric_is_materialized_as_nan():
    """A metric that emitted no tidy rows must appear as a NaN column, not vanish.

    If it vanished, score_metrics would reject the whole run for mismatched columns
    instead of reporting one undefined comparator. Under v1 this is reachable: the
    no-droppable-NaN fill is version-gated (run.py:207-211), so a metric can contribute
    nothing at all.
    """
    wide = aggregate_metrics_wide(
        _tidy([("A", "expr_mae", 1.0)]), metrics=["expr_mae", "delta_pearson"]
    )
    assert wide.columns == ["statistic", "delta_pearson", "expr_mae"]
    col = dict(zip(wide["statistic"], wide["delta_pearson"]))
    assert col["count"] == 0.0
    assert math.isnan(col["mean"])


def test_expected_metrics_do_not_drop_observed_ones():
    """A metric present in the frame but absent from `metrics` is a caller bug; the union
    keeps it visible rather than silently dropping data."""
    wide = aggregate_metrics_wide(_tidy([("A", "pds_cosine", 0.5)]), metrics=["expr_mae"])
    assert wide.columns == ["statistic", "expr_mae", "pds_cosine"]


def test_empty_metric_set_raises():
    """A statistic-only frame scores to a vacuous avg_score = 0.0 -- silently 'perfect'.
    Fail loud instead."""
    with pytest.raises(ValueError, match="no metrics"):
        aggregate_metrics_wide(_tidy([]))
    with pytest.raises(ValueError, match="no metrics"):
        aggregate_metrics_wide(_tidy([]), metrics=[])


def test_metric_output_names_tracks_the_version():
    v2 = EvalConfig(metrics=["expr_mae", "delta_pearson"], version="v2")
    v1 = EvalConfig(metrics=["expr_mae", "delta_pearson"], version="v1")
    assert metric_output_names(v2) == ["expr_mae", "delta_pearson"]
    assert metric_output_names(v1) == ["mae", "pearson_delta"]


def test_metric_output_names_mirrors_the_deseq2_relabel():
    """dispatch_de_metrics (run.py:250-275) relabels de_wilcoxon_* -> de_deseq2_* under the
    deseq2 backend and dedupes explicitly-selected siblings. If this helper read CATALOG
    directly, aggregate_metrics_wide would materialize a phantom all-NaN de_wilcoxon_*
    column beside the real de_deseq2_* one, and the degeneracy gate would then reject it."""
    from dataclasses import replace
    base = EvalConfig(metrics=["expr_mae", "de_wilcoxon_overlap"])
    assert metric_output_names(base) == ["expr_mae", "de_wilcoxon_overlap"]
    ds = replace(base, de=replace(base.de, backend="deseq2"))
    assert metric_output_names(ds) == ["expr_mae", "de_deseq2_overlap"]
    # sibling collapse: both spellings selected -> ONE emitted name, order preserved
    both = replace(EvalConfig(metrics=["de_wilcoxon_overlap", "de_deseq2_overlap"]),
                   de=replace(base.de, backend="deseq2"))
    assert metric_output_names(both) == ["de_deseq2_overlap"]


def test_score_metrics_accepts_it_on_both_sides(synthetic_pair):
    """End of the seam: the wide agg is a valid score_metrics input, and a frame scored
    against itself gives 0.0 everywhere (1 - u/u = 0 and (u-u)/(1-u) = 0)."""
    pred, real = synthetic_pair
    cfg = EvalConfig(metrics=["expr_mae", "delta_pearson"], pert_col="target",
                     control="non-targeting", input_type="lognorm", validate_input=False)
    tidy = compute_metrics(pred, real, config=cfg)
    agg = aggregate_metrics_wide(tidy, metrics=metric_output_names(cfg))
    # expr_mae is best_value="zero"; score_metrics fails loud on base <= 0, so the
    # fixture must have a strictly positive MAE. Assert that precondition explicitly.
    assert dict(zip(agg["statistic"], agg["expr_mae"]))["mean"] > 0
    out = score_metrics(agg, agg)
    assert out.filter(pl.col("metric") == "avg_score")["from_baseline"][0] == pytest.approx(0.0)


# --- #239: the cohort the aggregate was actually computed over -------------------------------

def test_metric_cohorts_counts_the_values_the_statistic_used():
    from cell_eval2.run import metric_cohorts
    coh = metric_cohorts(_tidy([
        ("A", "expr_mae", 1.0), ("B", "expr_mae", float("nan")), ("C", "expr_mae", 3.0),
    ]))
    row = coh.filter(pl.col("metric") == "expr_mae").to_dicts()[0]
    assert row == {"metric": "expr_mae", "n_used": 2, "n_rows": 3, "n_nan": 1, "n_null": 0,
                   "derived": False}


def test_metric_cohorts_is_NOT_the_wide_frames_count_row():
    """#239's actual gap, and why 'agg_results.csv already carries a count' does not close it.

    `count`/`null_count` describe the RAW series on purpose, and polars nulls are not NaNs. So a
    metric that returns NaN for a perturbation it could not score reports count=3, null_count=0
    while the mean was taken over 1 value. The two agree only for a metric that OMITS the row.
    """
    from cell_eval2.run import metric_cohorts
    df = _tidy([
        ("A", "expr_mae", 1.0), ("B", "expr_mae", float("nan")), ("C", "expr_mae", float("nan")),
        ("A", "pds_cosine", 0.5),                       # this metric omitted B and C instead
    ])
    wide = aggregate_metrics_wide(df)
    coh = metric_cohorts(df)
    counts = {s: dict(zip(wide.columns[1:], vals))
              for s, *vals in wide.rows()}
    used = dict(zip(coh["metric"].to_list(), coh["n_used"].to_list()))

    assert counts["count"]["expr_mae"] == 3.0 and used["expr_mae"] == 1      # NaN-emitting: differ
    assert counts["null_count"]["expr_mae"] == 0.0                          # nulls != NaNs
    assert counts["count"]["pds_cosine"] == 1.0 and used["pds_cosine"] == 1  # row-omitting: agree
    # And n_used is the cohort the reported mean was over, not merely "some smaller number".
    assert counts["mean"]["expr_mae"] == 1.0


def test_metric_cohorts_covers_exactly_the_wide_frames_metric_columns():
    """A sidecar describing a different metric set from the file it annotates is worse than
    none, so the two share one name resolution (`_wide_metric_names`)."""
    from cell_eval2.run import metric_cohorts
    df = _tidy([("A", "expr_mae", 1.0)])
    for metrics in (None, ["expr_mae", "pds_cosine"]):
        wide = aggregate_metrics_wide(df, metrics=metrics)
        coh = metric_cohorts(df, metrics=metrics)
        assert coh["metric"].to_list() == list(wide.columns[1:])


def test_metric_cohorts_flags_a_derived_metric_rather_than_reporting_zero_as_a_cohort():
    """`expr_mse_unbiased_capped_norm` is agg='ratio_of_sums' and emits NO per-perturbation
    rows, so 0 means 'not applicable'. Kept in the frame (the metric set must match the wide
    frame's columns) but marked, so a reader cannot mistake it for 'nothing scored'."""
    from cell_eval2.run import metric_cohorts
    df = _tidy([("A", "expr_mse_unbiased_capped", 0.2), ("A", "expr_mse_unbiased", 0.4)])
    coh = metric_cohorts(df, metrics=["expr_mse_unbiased_capped_norm",
                                      "expr_mse_unbiased_capped", "expr_mse_unbiased"])
    derived = coh.filter(pl.col("metric") == "expr_mse_unbiased_capped_norm").to_dicts()[0]
    assert derived["derived"] is True
    assert derived["n_used"] == 0 and derived["n_rows"] == 0
    assert not any(coh.filter(pl.col("metric") == m)["derived"][0]
                   for m in ("expr_mse_unbiased_capped", "expr_mse_unbiased"))


def test_metric_cohorts_empty_metric_set_raises_like_the_wide_frame():
    from cell_eval2.run import metric_cohorts
    with pytest.raises(ValueError, match="no metrics to aggregate"):
        metric_cohorts(_tidy([]))


# --- #277: peak host memory is reported as a number ---------------------------------------------

def test_peak_host_rss_bytes_returns_a_plausible_number():
    """#277 item 3. Plausibility, not a value: ru_maxrss is a process high-water mark, so the only
    safe assertions are that it is positive and that the KiB->bytes conversion happened (a
    scoring process is above 16 MiB and below 10 TiB)."""
    from cell_eval2.run import peak_host_rss_bytes
    rss = peak_host_rss_bytes()
    assert rss is not None, "resource.getrusage should be available on this platform"
    assert 16 * 2**20 < rss < 10 * 2**40, rss


def test_compute_metrics_logs_the_peak_host_rss(synthetic_pair, caplog):
    """The point of #277 item 3 is that the number appears in every run's log, so a future
    increase shows up as a number rather than as a truncated leaderboard."""
    import logging

    from cell_eval2 import EvalConfig, compute_metrics
    pred, real = synthetic_pair
    cfg = EvalConfig.from_dict({**EvalConfig.v2().to_dict(), "metrics": ["expr_mae"],
                                "device": "cpu", "validate_input": False})
    with caplog.at_level(logging.INFO, logger="cell_eval2.run"):
        compute_metrics(pred, real, config=cfg)
    assert "peak host RSS at end of compute_metrics attempt" in caplog.text
    assert "GiB" in caplog.text and "#277" in caplog.text


def test_the_peak_rss_line_appears_on_a_WARM_RESULT_CACHE_HIT_too(synthetic_pair, tmp_path,
                                                                   caplog):
    """codex-review: `_run_metrics` returns EARLY on a result-cache hit, so logging at its end
    missed exactly the runs a memory report is cheapest on -- and made "every in-memory scoring
    run" false. The log now sits in `compute_metrics`, in a finally around `_run_metrics`."""
    import logging

    from cell_eval2 import EvalConfig, compute_metrics
    pred, real = synthetic_pair
    cfg = EvalConfig.from_dict({**EvalConfig.v2().to_dict(), "metrics": ["expr_mae"],
                                "device": "cpu", "validate_input": False,
                                "cache_pred": str(tmp_path / "p"),
                                "cache_real": str(tmp_path / "r")})
    compute_metrics(pred, real, config=cfg)              # cold: writes the result cache
    caplog.clear()
    with caplog.at_level(logging.INFO, logger="cell_eval2.run"):
        compute_metrics(pred, real, config=cfg)          # warm: returns from the cache
    assert "peak host RSS at end of compute_metrics attempt" in caplog.text, \
        "a result-cache hit must still report peak memory"


def test_the_rss_reporter_never_fails_a_run(synthetic_pair, monkeypatch, caplog):
    """Provenance must never be able to break scoring. Both halves degrade: an unreadable
    ru_maxrss returns None and logs nothing, and an object with no .n_obs still logs the number."""
    import logging

    from cell_eval2 import run as run_mod

    monkeypatch.setattr(run_mod, "peak_host_rss_bytes", lambda: None)
    run_mod._log_peak_host_rss(object(), object())          # must not raise

    monkeypatch.setattr(run_mod, "peak_host_rss_bytes", lambda: 3 * 2**30)
    with caplog.at_level(logging.INFO, logger="cell_eval2.run"):
        run_mod._log_peak_host_rss(object(), object())      # shape unreadable -> "?"
    assert "3.00 GiB (?)" in caplog.text


def test_metric_cohorts_separates_nulls_from_nans_in_the_same_column():
    """The case no existing cohort fixture covered: nulls AND NaNs in one metric's column.

    ⚠️ Every earlier fixture had `n_null == 0`, so the null/NaN interaction was untested in both
    directions -- which mattered while `n_nan` was derived as `raw.len() - nn.len()` and silently
    depended on `drop_nans()` preserving nulls. The group_by form (round 3) no longer depends on
    that: `is_nan()` yields null for a null entry and `.sum()` skips nulls, so it counts true NaNs
    directly. The end-to-end row below is the check that survived both implementations, which is
    why it is written against the OUTPUT rather than against either arithmetic.

    The two are reported separately on purpose: a polars null and a NaN are different states, and
    collapsing them would hide which one a metric is actually producing.
    """
    from cell_eval2.run import metric_cohorts

    df = pl.DataFrame(
        [("A", "expr_mae", 1.0), ("B", "expr_mae", float("nan")), ("C", "expr_mae", None),
         ("D", "expr_mae", 4.0), ("E", "expr_mae", float("nan"))],
        schema=["perturbation", "metric", "value"], orient="row",
    )
    row = metric_cohorts(df).row(by_predicate=pl.col("metric") == "expr_mae", named=True)
    assert row == {"metric": "expr_mae", "n_used": 2, "n_rows": 5, "n_nan": 2, "n_null": 1,
                   "derived": False}

    # ...and the polars semantics the group_by form relies on, stated directly, so a failure here
    # names the cause rather than leaving someone to infer it from a wrong n_nan: `is_nan()` must
    # yield NULL (not False) for a null entry, and `.sum()` must skip it.
    v = df["value"]
    assert v.is_nan().null_count() == v.null_count(), "is_nan() must yield null for a null entry"
    assert v.is_nan().sum() == 2, "sum() over is_nan() must count true NaNs only, skipping nulls"
