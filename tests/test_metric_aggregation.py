import polars as pl
import pytest

from cell_eval2.catalog import CATALOG
from cell_eval2.run import aggregate_metrics, aggregate_metrics_wide


def _tidy_pairs(pairs):
    return pl.DataFrame(
        {"perturbation": [f"p{i}" for i in range(len(pairs))],
         "metric": [m for m, _ in pairs],
         "value": [float(v) for _, v in pairs]},
        schema={"perturbation": pl.String, "metric": pl.String, "value": pl.Float64},
    )


def test_every_nonderived_catalog_entry_aggregates_by_mean():
    """No per-perturbation family is "median by design" any more (#231).

    `test_scoring_catalog.py::test_the_catalog_has_exactly_one_aggregation_statistic` is the
    canonical statement of the aggregation-policy half. It is restated here because this is the file a
    reader lands in when `aggregate_metrics` misbehaves, and the v1-availability invariant
    below has to iterate the catalog anyway.
    """
    for name, spec in CATALOG.items():
        if spec.derived is None:
            assert spec.agg == "mean", name
        else:
            assert spec.agg == "ratio_of_sums", name
        # v1-availability is DERIVED from v1_name now: a metric with an upstream cell-eval
        # equivalent is offered under version="v1", a v2-native one never is. Asserting a
        # bare `spec.v1_available` pinned the old state, in which seven v2-native metrics
        # were v1-available purely because nobody passed the flag.
        assert spec.v1_available == (spec.v1_name is not None), name


def test_aggregate_metrics_emits_an_agg_column():
    df = _tidy_pairs([("expr_mae", 1.0), ("expr_mae", 3.0)])
    out = aggregate_metrics(df)
    assert out["agg"].to_list() == ["mean"]
    assert out["mean"].to_list() == [pytest.approx(2.0)]


def test_wide_frame_stays_strictly_numeric(monkeypatch):
    """Spec 4: a string row would coerce every metric column to text and score.py would
    silently stop receiving numbers after one CSV round-trip."""
    df = _tidy_pairs([("expr_mae", 1.0), ("expr_mae", 3.0)])
    wide = aggregate_metrics_wide(df, metrics=["expr_mae"])
    assert wide["expr_mae"].dtype == pl.Float64
    assert set(wide.columns) == {"statistic", "expr_mae"}


def test_median_agg_is_honoured_by_both_aggregators(monkeypatch):
    from dataclasses import replace
    import cell_eval2.run as run_mod
    spec = replace(CATALOG["expr_mae"], agg="median")
    monkeypatch.setitem(run_mod.CATALOG, "expr_mae", spec)
    df = _tidy_pairs([("expr_mae", 1.0), ("expr_mae", 2.0), ("expr_mae", 60.0)])
    assert aggregate_metrics(df)["mean"].to_list() == [pytest.approx(2.0)]
    wide = aggregate_metrics_wide(df, metrics=["expr_mae"])
    row = wide.filter(pl.col("statistic") == "mean")
    assert row["expr_mae"].to_list() == [pytest.approx(2.0)]


def test_wide_csv_survives_a_round_trip_with_numeric_columns(tmp_path):
    df = _tidy_pairs([("expr_mae", 1.0), ("expr_mae", 3.0)])
    wide = aggregate_metrics_wide(df, metrics=["expr_mae"])
    path = tmp_path / "results.csv"
    wide.write_csv(path)
    back = pl.read_csv(path)
    assert back["expr_mae"].dtype == pl.Float64


def test_vcc2026_uses_mean_components_and_one_ratio_of_sums():
    """#229: `avg_score` is a plain unweighted mean over metrics, so a competition profile
    whose members answer two different questions is exactly what this removes.

    Resolved THROUGH THE PROFILE rather than against a name list: enrolling a
    median-aggregated metric in `vcc2026` later must fail here, not silently reintroduce the
    mix this was written to end.
    """
    from cell_eval2.catalog import resolve_metrics
    names, missing = resolve_metrics("vcc2026")
    assert not missing
    derived = {n for n in names if CATALOG[n].derived is not None}
    assert derived == {"expr_mse_unbiased_capped_norm"}
    assert {CATALOG[n].agg for n in names if n not in derived} == {"mean"}
    assert {CATALOG[n].agg for n in derived} == {"ratio_of_sums"}


def test_the_direction_family_is_moved_WHOLESALE():
    """#231 finished what #229 started: the family is uniform, not split by profile.

    `de_wilcoxon_direction_fidelity` was the sharpest case against the split. It carries the
    identical `(raw - q)/d` chance correction as `direction_fidelity_yield` and the same
    [-20, 1] range, yet through v0.7.0 it kept the median purely because `vcc2026` did not
    score it -- a metric's own statistic depending on a profile membership it does not have.

    Enumerated from the FUNCTION MODULE rather than a frozen name list, so a direction metric
    added later with `agg="median"` fails here instead of quietly reinstating the split.
    """
    # `func` is a bare function for most entries and a functools.partial for the ones that
    # bind `universe`/`corrected`; unwrap one level so both spellings resolve to the module.
    family = {n: s for n, s in CATALOG.items()
              if getattr(getattr(s.func, "func", s.func), "__module__", "")
              == "cell_eval2.metrics.direction"}
    assert len(family) == 28                     # 14 suffixes x 2 backend families
    for name, spec in family.items():
        assert spec.agg == "mean", name


def test_aggregation_is_backend_invariant():
    """A metric must not answer a different question because the DE backend changed.

    `_register_de_family` passes ONE `agg` to each wilcoxon/deseq2 pair, so this holds by
    construction today -- which is exactly why it needs a test: the deseq2 siblings are not
    in `vcc2026`, so a profile-driven change looks like it should move only the wilcoxon
    entry, and a hand-written special case would be an easy and silent regression.
    """
    pairs = [(n, n.replace("de_wilcoxon_", "de_deseq2_"))
             for n in CATALOG if n.startswith("de_wilcoxon_")]
    assert pairs
    for wilcoxon, deseq2 in pairs:
        assert CATALOG[wilcoxon].agg == CATALOG[deseq2].agg, wilcoxon


def test_wide_frame_publishes_the_median_beside_the_mean():
    """Skewed on purpose (mean 4.0 vs median 2.0) -- an accidental alias would pass on a
    symmetric sample."""
    wide = aggregate_metrics_wide(_tidy_pairs([("expr_mae", 1.0), ("expr_mae", 2.0),
                                               ("expr_mae", 9.0)]))
    row = dict(zip(wide["statistic"], wide["expr_mae"]))
    assert row["mean"] == pytest.approx(4.0)
    assert row["median"] == pytest.approx(2.0)


def test_the_median_row_is_unconditional_not_a_second_agg_lookup(monkeypatch):
    """For a metric that SCORES on the median both rows are equal: `mean` holds whatever
    `MetricSpec.agg` declares, and the new row is the median regardless. Asserting the pair
    is what distinguishes "publishes the median" from "publishes the other statistic"."""
    from dataclasses import replace
    import cell_eval2.run as run_mod
    monkeypatch.setitem(run_mod.CATALOG, "expr_mae",
                        replace(CATALOG["expr_mae"], agg="median"))
    wide = aggregate_metrics_wide(_tidy_pairs([("expr_mae", 1.0), ("expr_mae", 2.0),
                                               ("expr_mae", 9.0)]), metrics=["expr_mae"])
    row = dict(zip(wide["statistic"], wide["expr_mae"]))
    assert row["mean"] == pytest.approx(2.0)
    assert row["median"] == pytest.approx(2.0)


NUM, DEN, DERIVED = ("expr_mse_unbiased_capped", "expr_distance_unbiased",
                     "expr_mse_unbiased_capped_norm")


@pytest.fixture(autouse=True)
def _register_derived_if_absent(monkeypatch):
    """Make this module runnable BEFORE Task 6 registers the real entries.

    Task 5 lands the machinery; Task 6 lands the catalog. Without this the tests below are red
    between the two, which is indistinguishable from a broken implementation (codex,
    checkpoint 1). Once Task 6 lands the catalog already has them and this is a no-op.
    """
    from cell_eval2.catalog import CATALOG, DerivedAgg, MetricSpec
    from cell_eval2.scoring import DIAG, Scoring

    if DERIVED in CATALOG:
        return
    patched = dict(CATALOG)
    for name in (NUM, DEN):
        patched[name] = MetricSpec(
            name=name, func=lambda **_: {}, scoring=DIAG, agg="mean", profiles=("full",),
            kind="anndata", normalization="lognorm", needs_moments=True)
    patched[DERIVED] = MetricSpec(
        name=DERIVED, func=None, agg="ratio_of_sums",
        derived=DerivedAgg(numerator=NUM, denominator=DEN),
        scoring=Scoring(scored=True, direction="lower", anchor=0.0, penalty="boxcox",
                        clamp_low=None, clamp_high=1.0),
        profiles=("full",), kind="anndata", normalization="lognorm")
    monkeypatch.setattr("cell_eval2.run.CATALOG", patched)


def _tidy(rows):
    return pl.DataFrame(rows, schema={"perturbation": pl.String, "metric": pl.String,
                                      "value": pl.Float64}, orient="row")


def _metric_names(aggregator, out):
    return out["metric"].to_list() if aggregator is aggregate_metrics else out.columns[1:]


def _derived_mean(aggregator, out):
    if aggregator is aggregate_metrics:
        return out.filter(pl.col("metric") == DERIVED)["mean"][0]
    return out.filter(pl.col("statistic") == "mean")[DERIVED][0]


@pytest.mark.parametrize("aggregator", [aggregate_metrics, aggregate_metrics_wide],
                         ids=["tidy", "wide"])
def test_components_only_selection_does_not_inject_the_derived_metric(aggregator):
    df = _tidy([("p1", NUM, 2.0), ("p1", DEN, 8.0)])
    wrong_injected_value = 0.25
    assert 2.0 / 8.0 == wrong_injected_value, "fixture no longer builds the wrong value"

    out = aggregator(df, metrics=[NUM, DEN])
    names = _metric_names(aggregator, out)
    assert DERIVED not in names, (
        f"components-only selection injected {DERIVED}={wrong_injected_value}; "
        f"aggregate names were {names}"
    )


@pytest.mark.parametrize("aggregator", [aggregate_metrics, aggregate_metrics_wide],
                         ids=["tidy", "wide"])
def test_selecting_the_derived_metric_builds_it_from_the_same_frame(aggregator):
    df = _tidy([("p1", NUM, 2.0), ("p1", DEN, 8.0)])
    expected = 0.25
    assert 2.0 / 8.0 == expected, "fixture no longer has the expected ratio"

    out = aggregator(df, metrics=[NUM, DEN, DERIVED])
    names = _metric_names(aggregator, out)
    assert DERIVED in names, f"selected {DERIVED} vanished; aggregate names were {names}"
    got = _derived_mean(aggregator, out)
    assert got == pytest.approx(expected), f"selected {DERIVED} was {got}, expected {expected}"


@pytest.mark.parametrize("aggregator", [aggregate_metrics, aggregate_metrics_wide],
                         ids=["tidy", "wide"])
def test_unknown_selection_keeps_injecting_the_buildable_derived_metric(aggregator):
    df = _tidy([("p1", NUM, 3.0), ("p1", DEN, 12.0)])
    expected = 0.25
    assert 3.0 / 12.0 == expected, "fixture no longer has the expected legacy value"

    out = aggregator(df, metrics=None)
    names = _metric_names(aggregator, out)
    assert DERIVED in names, f"metrics=None stopped legacy injection; names were {names}"
    got = _derived_mean(aggregator, out)
    assert got == pytest.approx(expected), f"legacy derived value was {got}, expected {expected}"


@pytest.mark.parametrize("aggregator", [aggregate_metrics, aggregate_metrics_wide],
                         ids=["tidy", "wide"])
@pytest.mark.parametrize("empty,other", [(NUM, DEN), (DEN, NUM)],
                         ids=["numerator_empty", "denominator_empty"])
def test_selected_derived_metric_raises_when_a_component_has_no_rows(
        aggregator, empty, other):
    df = _tidy([("p1", other, 4.0), ("p2", other, 6.0)])
    assert df.filter(pl.col("metric") == empty).height == 0, "empty side gained a row"
    assert df.filter(pl.col("metric") == other).height == 2, "other-side count changed"

    with pytest.raises(ValueError) as excinfo:
        aggregator(df, metrics=[NUM, DEN, DERIVED])
    message = str(excinfo.value)
    assert DERIVED in message, f"error did not name the requested metric: {message}"
    assert f"component {empty} is empty" in message, f"error did not name empty side: {message}"
    assert f"{other} has 2 row(s)" in message, f"error lost the other-side row count: {message}"


@pytest.mark.parametrize("aggregator", [aggregate_metrics, aggregate_metrics_wide],
                         ids=["tidy", "wide"])
def test_a_derived_alias_in_the_selection_is_compared_canonically(
        aggregator, monkeypatch):
    import cell_eval2.run as run_mod

    alias = "derived_ratio_alias"
    monkeypatch.setitem(run_mod._NAME_TO_CANONICAL, alias, DERIVED)
    df = _tidy([("p1", NUM, 5.0), ("p1", DEN, 10.0)])
    expected = 0.5
    assert 5.0 / 10.0 == expected, "fixture no longer has the expected alias-selected value"

    out = aggregator(df, metrics=[NUM, DEN, alias])
    emitted_name = DERIVED if aggregator is aggregate_metrics else alias
    got = (out.filter(pl.col("metric") == emitted_name)["mean"][0]
           if aggregator is aggregate_metrics else
           out.filter(pl.col("statistic") == "mean")[emitted_name][0])
    assert got == pytest.approx(expected), (
        f"selection alias {alias!r} did not select {DERIVED}; got {got}, expected {expected}"
    )


def test_the_derived_metric_is_the_ratio_of_sums_not_the_mean_of_ratios():
    # Chosen so the two disagree by a wide margin: p2's denominator is near zero, which is
    # exactly the case a mean of ratios blows up on.
    df = _tidy([("p1", NUM, 2.0), ("p2", NUM, 0.10),
                ("p1", DEN, 8.0), ("p2", DEN, 0.01)])
    ratio_of_sums = (2.0 + 0.10) / (8.0 + 0.01)      # 0.2622
    mean_of_ratios = (2.0 / 8.0 + 0.10 / 0.01) / 2   # 5.125
    assert abs(ratio_of_sums - mean_of_ratios) > 1.0, "fixture cannot discriminate"
    got = aggregate_metrics(df).filter(pl.col("metric") == DERIVED)["mean"][0]
    assert got == pytest.approx(ratio_of_sums), f"got {got}, mean-of-ratios is {mean_of_ratios}"


def test_a_negative_denominator_contributes_without_raising():
    df = _tidy([("p1", NUM, 1.0), ("p2", NUM, 1.0),
                ("p1", DEN, 5.0), ("p2", DEN, -1.0)])
    got = aggregate_metrics(df).filter(pl.col("metric") == DERIVED)["mean"][0]
    assert got == pytest.approx(2.0 / 4.0)


def test_a_non_positive_denominator_SUM_raises():
    df = _tidy([("p1", NUM, 1.0), ("p1", DEN, 2.0), ("p2", NUM, 1.0), ("p2", DEN, -2.0)])
    with pytest.raises(ValueError, match="sum of expr_distance_unbiased"):
        aggregate_metrics(df)


def test_a_perturbation_is_used_only_when_BOTH_sides_are_finite():
    # Summing the two columns independently would keep p2's denominator while dropping its
    # numerator, biasing the ratio down. The pairing is what prevents that.
    df = _tidy([("p1", NUM, 3.0), ("p2", NUM, float("nan")),
                ("p1", DEN, 6.0), ("p2", DEN, 100.0)])
    got = aggregate_metrics(df).filter(pl.col("metric") == DERIVED)["mean"][0]
    unpaired = 3.0 / 106.0
    assert abs(0.5 - unpaired) > 0.4, "fixture cannot discriminate"
    assert got == pytest.approx(0.5), f"got {got}; the unpaired answer would be {unpaired}"


def test_both_components_present_but_no_finite_pair_RAISES_rather_than_vanishing():
    # Distinguished from "a component is absent", which legitimately yields no row: here the
    # metric can be asked for and cannot be answered, and a scored metric quietly leaving the
    # aggregate is the #250 failure mode (checkpoint-2 review, #257).
    df = _tidy([("p1", NUM, float("nan")), ("p1", DEN, float("nan"))])
    with pytest.raises(ValueError, match="no perturbation has a finite value on BOTH sides"):
        aggregate_metrics(df)


def test_finite_sums_whose_quotient_overflows_RAISE():
    # Both sums are finite and the denominator is positive; the DIVISION overflows.
    df = _tidy([("p1", NUM, 1e308), ("p1", DEN, 1e-308)])
    with pytest.raises(ValueError, match="overflow in the division itself"):
        aggregate_metrics(df)


def test_the_derived_metric_is_absent_when_a_component_is():
    df = _tidy([("p1", NUM, 3.0)])
    assert DERIVED not in aggregate_metrics(df)["metric"].to_list()


@pytest.mark.parametrize("bad", [float("inf"), float("-inf")])
@pytest.mark.parametrize("side", [NUM, DEN], ids=["in_numerator", "in_denominator"])
def test_an_infinite_value_is_excluded_like_a_NaN(bad, side):
    # is_not_nan() lets +/-inf through; an infinite denominator sum passes `> 0` and yields
    # 0.0 or NaN silently. Parametrized over BOTH columns: filtering only the denominator
    # (`n.is_not_nan() & d.is_finite()`) would otherwise pass (codex round 2).
    rows = [("p1", NUM, 3.0), ("p2", NUM, 1.0), ("p1", DEN, 6.0), ("p2", DEN, 2.0)]
    rows = [(p, m, bad if (p == "p2" and m == side) else v) for p, m, v in rows]
    got = aggregate_metrics(_tidy(rows)).filter(pl.col("metric") == DERIVED)["mean"][0]
    assert got == pytest.approx(0.5), f"got {got}; p2 should have been excluded entirely"


def test_finite_inputs_that_overflow_when_summed_raise():
    # The guard is separate from the is_finite filter: every input is finite, the SUM is not.
    big = 1e308
    df = _tidy([("p1", NUM, 1.0), ("p2", NUM, 1.0),
                ("p1", DEN, big), ("p2", DEN, big)])
    with pytest.raises(ValueError, match="overflowed"):
        aggregate_metrics(df)


@pytest.mark.parametrize("aggregator", [aggregate_metrics, aggregate_metrics_wide],
                         ids=["tidy", "wide"])
def test_a_derived_metric_appearing_as_per_perturbation_rows_raises(aggregator):
    # It would otherwise be group-by'd AND appended, giving two rows for one metric. Both
    # aggregators must reject it -- covering only the tidy one lets the wide guard be omitted.
    df = _tidy([("p1", NUM, 2.0), ("p1", DEN, 8.0), ("p1", DERIVED, 0.25)])
    with pytest.raises(ValueError, match="appear as per-perturbation rows"):
        aggregator(df)


@pytest.mark.parametrize("aggregator", [aggregate_metrics, aggregate_metrics_wide],
                         ids=["tidy", "wide"])
def test_a_derived_metric_under_an_ALIAS_is_rejected_too(aggregator, monkeypatch):
    # The guard's alias arm used to be a tautology -- it canonicalized the CATALOG key, which
    # is already canonical, instead of the observed name. The one shipped derived metric has
    # no alias, so only a patched one can show it (both PR bots, round 3).
    from dataclasses import replace

    import cell_eval2.run as run_mod
    from cell_eval2.catalog import CATALOG as REAL

    alias = "expr_mse_unbiased_norm_legacy"
    spec = run_mod.CATALOG[DERIVED] if DERIVED in run_mod.CATALOG else REAL[DERIVED]
    monkeypatch.setattr("cell_eval2.run._NAME_TO_CANONICAL",
                        {**run_mod._NAME_TO_CANONICAL, alias: DERIVED})
    monkeypatch.setitem(run_mod.CATALOG, DERIVED, replace(spec, aliases=(alias,)))
    df = _tidy([("p1", NUM, 2.0), ("p1", DEN, 8.0), ("p1", alias, 0.25)])
    assert alias not in {n for n in run_mod.CATALOG}, "the alias must not be a catalog KEY"
    with pytest.raises(ValueError, match="appear as per-perturbation rows"):
        aggregator(df)


def test_the_components_must_cover_the_same_perturbations():
    df = _tidy([("p1", NUM, 2.0), ("p2", NUM, 1.0), ("p1", DEN, 8.0)])
    with pytest.raises(ValueError, match="cover different perturbations"):
        aggregate_metrics(df)


def test_the_derived_aggregate_recomputes_correctly_on_a_subset():
    # Spec §6: with no per-perturbation column, a subset's score is re-derived from the
    # component columns. Restricting the frame must give the subset's ratio of sums, NOT a
    # rescaling of the whole-panel number.
    full = _tidy([("p1", NUM, 2.0), ("p2", NUM, 1.0), ("p3", NUM, 6.0),
                  ("p1", DEN, 8.0), ("p2", DEN, 2.0), ("p3", DEN, 10.0)])
    whole = aggregate_metrics(full).filter(pl.col("metric") == DERIVED)["mean"][0]
    subset = aggregate_metrics(full.filter(pl.col("perturbation") != "p3")).filter(
        pl.col("metric") == DERIVED)["mean"][0]
    assert whole == pytest.approx(9.0 / 20.0)
    assert subset == pytest.approx(3.0 / 10.0)
    assert abs(whole - subset) > 0.1, "fixture cannot discriminate"


def test_the_wide_frame_carries_only_a_mean_for_the_derived_metric():
    df = _tidy([("p1", NUM, 2.0), ("p1", DEN, 8.0)])
    wide = aggregate_metrics_wide(df)
    col = dict(zip(wide["statistic"].to_list(), wide[DERIVED].to_list()))
    assert col["mean"] == pytest.approx(0.25)
    for stat in ("count", "null_count", "std", "min", "max", "median"):
        assert col[stat] != col[stat], f"{stat} should be NaN for a derived metric, got {col[stat]}"
