"""Issue #207 -- `precision == purity` must not depend on how polars represents a comparison.

`_rank_p` casts the ranking key to Float64 (load-bearing: `is_nan()` raises on a Decimal
column). The significance filter compares the NATIVE column, and polars picks that
comparison's representation from the THRESHOLD's type -- so a non-float threshold (integer
or Decimal) over a Decimal `p_adj` put the filter and the ranking on different
representations and `{p_adj < alpha}` could stop being a prefix of the ranking.

The fixture below is that shape end to end, and it is deliberately fragile: two p-values
that collide in Float64 across the boundary, plus a later key component that orders the
NON-significant one first. Before the fix `de_direction_precision` read 1.0 while purity at
the boundary read 0.0.
"""
from __future__ import annotations

from decimal import Decimal

import polars as pl
import pytest

from cell_eval2.de import prepare_de
from cell_eval2.metrics.direction import (
    _direction_frame,
    _purity_curve,
    _rank_p,
    de_direction_precision,
)

DEC = pl.Decimal(30, 21)
JUST_UNDER = Decimal("0.999999999999999999998")   # < 1 natively; 1.0 once cast to Float64
EXACTLY_ONE = Decimal("1.000000000000000000000")  # not < 1; also 1.0 in Float64


def _frame(features, lfcs, p_adjs):
    return pl.DataFrame({
        "target": pl.Series(["G1"] * len(features), dtype=pl.String),
        "feature": pl.Series(features, dtype=pl.String),
        "log2_fold_change": pl.Series(lfcs, dtype=pl.Float64),
        "p_adj": pl.Series(p_adjs, dtype=DEC),
    })


def _fixture():
    # A is significant natively and MATCHES the reference.
    # B is not significant, but carries the larger |lfc| so it wins the Float64 tiebreak and
    # would sort FIRST without a native-significance key -- and it is a MISS.
    pred = _frame(["A", "B"], [1.0, -5.0], [JUST_UNDER, EXACTLY_ONE])
    real = _frame(["A", "B"], [1.0, 1.0], [Decimal("0.001"), Decimal("0.001")])
    return prepare_de(pred, real, control="non-targeting", p_adj_threshold=1,
                      nan_lfc_policy="keep")


def test_the_fixture_really_is_the_broken_shape():
    """Guard the guard. If polars ever stops splitting these representations the test below
    would pass for the wrong reason, so assert the split itself."""
    pred = _frame(["A", "B"], [1.0, -5.0], [JUST_UNDER, EXACTLY_ONE])
    assert pred.select(_rank_p("p_adj").alias("r"))["r"].to_list() == [1.0, 1.0]
    assert pred.select(pl.col("p_adj") < 1)["p_adj"].to_list() == [True, False]
    assert pred.select(pl.col("p_adj") < 1.0)["p_adj"].to_list() == [False, False]


def test_significant_set_is_a_prefix_under_a_non_float_threshold():
    prepared = _fixture()
    curve = _purity_curve(_direction_frame(prepared), universe="all", alpha=1)
    # k = |S_pred INTERSECT adjudicable| = 1, and the row at that depth must be A.
    assert curve.filter(pl.col("k") == 1)["purity"][0] == pytest.approx(1.0)


def test_precision_equals_purity_at_the_significant_prefix():
    prepared = _fixture()
    curve = _purity_curve(_direction_frame(prepared), universe="all", alpha=1)
    precision = de_direction_precision(prepared)["G1"]
    assert precision == pytest.approx(1.0)
    assert curve.filter(pl.col("k") == 1)["purity"][0] == pytest.approx(precision)


def test_the_leading_key_is_what_repairs_it():
    """Reconstruct the pre-#207 sort and show it still inverts, so the assertions above are
    attributable to the new key rather than to some other property of the fixture."""
    prepared = _fixture()
    frame = _direction_frame(prepared)
    old = (
        frame.sort(["target", "rank_p_adj", "rank_p_value", "abs_lfc_pred", "feature"],
                   descending=[False, False, False, True, False])
        .with_columns(n_denom=pl.col("in_denom").cast(pl.Int64).cum_sum().over("target"),
                      n_match=pl.col("match").cast(pl.Int64).cum_sum().over("target"))
    )
    assert old["feature"].to_list() == ["B", "A"]              # non-significant ranks first
    assert (old["n_match"][0] / old["n_denom"][0]) == 0.0      # purity(1) = 0.0
    assert de_direction_precision(prepared)["G1"] == 1.0       # ...against precision 1.0


def test_a_float_threshold_orders_identically_with_and_without_the_key():
    """The key is INERT wherever the two representations already agreed -- which is every
    configuration this repo's own DE producers can reach."""
    pred = pl.DataFrame({
        "target": pl.Series(["G1"] * 6, dtype=pl.String),
        "feature": pl.Series(list("ABCDEF"), dtype=pl.String),
        "log2_fold_change": pl.Series([1.0, -2.0, 3.0, -4.0, 5.0, -6.0], dtype=pl.Float64),
        "p_adj": pl.Series([0.001, 0.04, 0.049, 0.051, 0.5, 1.0], dtype=pl.Float64),
    })
    real = pl.DataFrame({
        "target": pl.Series(["G1"] * 6, dtype=pl.String),
        "feature": pl.Series(list("ABCDEF"), dtype=pl.String),
        "log2_fold_change": pl.Series([1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype=pl.Float64),
        "p_adj": pl.Series([0.001] * 6, dtype=pl.Float64),
    })
    prepared = prepare_de(pred, real, control="non-targeting", p_adj_threshold=0.05,
                          nan_lfc_policy="keep")
    frame = _direction_frame(prepared)
    keys = ["target", "rank_p_adj", "rank_p_value", "abs_lfc_pred", "feature"]
    without = frame.sort(keys, descending=[False, False, False, True, False])["feature"]
    with_key = (
        frame.with_columns(_sig=(pl.col("p_adj_pred") < 0.05).fill_null(False))
        .sort(["target", "_sig"] + keys[1:],
              descending=[False, True, False, False, True, False])["feature"]
    )
    assert without.to_list() == with_key.to_list()
