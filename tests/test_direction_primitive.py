import logging

import polars as pl
import pytest

from cell_eval2.de import (
    apply_nan_policy,
    assemble_prepared_de,
    normalize_de_schema,
    prepare_de,
    rank_de_side,
)


def _prep(real: pl.DataFrame, pred: pl.DataFrame, *, threshold: float = 0.05):
    """Toy PreparedDE. nan_lfc_policy='keep' so NaN-LFC rows survive the significance
    gate — under the v2 'mask' default they are forced to p_adj=1 and the pred-side NaN
    cases below become unreachable."""
    return prepare_de(pred, real, control="non-targeting", p_adj_threshold=threshold,
                      nan_lfc_policy="keep")


def _prep_unvalidated(real: pl.DataFrame, pred: pl.DataFrame, *, threshold: float = 0.05):
    """A PreparedDE assembled WITHOUT `prep_de_side`, so #218's key check never runs.

    `assemble_prepared_de` is a public constructor, so a caller holding frames already can
    enter it without `prep_de_side` and the per-metric guards below stay reachable — #218 made
    `prep_de_side` the one seam that REFUSES a duplicated key, not the only thing standing
    between a malformed table and a wrong number. (The slicing drivers are NOT that caller:
    they go through `prepare_de` and so do get the check.)"""
    def side(df, name):
        out = apply_nan_policy(normalize_de_schema(df, name=name), name=name,
                               nan_lfc_policy="keep")
        return out, sorted(out["target"].unique().to_list())

    real_df, real_perts = side(real, "real")
    pred_df, pred_perts = side(pred, "pred")
    return assemble_prepared_de(
        rank_de_side(real_df, sort_by="abs_log2_fold_change", p_adj_threshold=threshold),
        real_perts,
        rank_de_side(pred_df, sort_by="abs_log2_fold_change", p_adj_threshold=threshold),
        pred_perts,
        control="non-targeting", sort_by="abs_log2_fold_change",
        p_adj_threshold=threshold, real_df=real_df, pred_df=pred_df,
    )


def test_defined_treats_inf_as_a_direction_but_not_zero_or_nan():
    from cell_eval2.metrics.direction import _defined
    df = pl.DataFrame({"x": [1.0, -1.0, 0.0, float("nan"), float("inf"), float("-inf"), None]})
    assert df.select(_defined("x").alias("d"))["d"].to_list() == [
        True, True, False, False, True, True, False,
    ]


def test_rank_p_sends_null_and_nan_last():
    from cell_eval2.metrics.direction import _rank_p
    df = pl.DataFrame({"p": [0.5, None, 0.1, float("nan")], "f": list("ABCD")})
    ordered = df.with_columns(r=_rank_p("p")).sort("r")["f"].to_list()
    assert ordered[:2] == ["C", "A"]          # real p-values first, ascending
    assert set(ordered[2:]) == {"B", "D"}     # null and NaN both pushed to the tail


def test_direction_frame_applies_the_asymmetric_rule():
    from cell_eval2.metrics.direction import _direction_frame
    # A: adjudicable + committed, signs agree -> in_denom, match
    # B: real lfc == 0     (cannot adjudicate)-> NOT in_denom, no match
    # C: pred lfc == 0     (declined)         -> in_denom, no match
    # D: real lfc == NaN   (cannot adjudicate)-> NOT in_denom, no match
    # E: pred lfc == NaN   (declined)         -> in_denom, no match
    real = pl.DataFrame({
        "target": ["G1"] * 5, "feature": list("ABCDE"),
        "log2_fold_change": [2.0, 0.0, 3.0, float("nan"), 4.0], "p_adj": [0.5] * 5,
    })
    pred = pl.DataFrame({
        "target": ["G1"] * 5, "feature": list("ABCDE"),
        "log2_fold_change": [1.0, 5.0, 0.0, 6.0, float("nan")], "p_adj": [0.001] * 5,
    })
    got = _direction_frame(_prep(real, pred)).sort("feature")
    assert got["in_denom"].to_list() == [True, False, True, False, True]
    assert got["match"].to_list() == [True, False, False, False, False]


def _duplicated_pair():
    real = pl.DataFrame({
        "target": ["G1"], "feature": ["A"], "log2_fold_change": [1.0], "p_adj": [0.001],
    })
    pred = pl.DataFrame({
        "target": ["G1", "G1"], "feature": ["A", "A"],
        "log2_fold_change": [1.0, 1.0], "p_adj": [0.001, 0.001],
    })
    return real, pred


def test_prep_de_side_rejects_duplicate_target_feature_rows():
    """#218's single seam: `prepare_de` refuses the malformed table outright, so every
    metric is correct at once rather than three of them disagreeing about it."""
    real, pred = _duplicated_pair()
    with pytest.raises(ValueError, match=r"duplicated \(target, feature\) key"):
        _prep(real, pred)


def test_direction_frame_rejects_duplicate_target_feature_rows():
    from cell_eval2.metrics.direction import _direction_frame
    real, pred = _duplicated_pair()
    # Unguarded this fans out 1x2 and inflates k* past N_conf, so the "bounded by 1"
    # adjudicated variant would return 2.0. Fail loudly instead. Assert the SPECIFIC
    # exception -- a bare `Exception` would be satisfied by any unrelated error.
    #
    # Entered through `_prep_unvalidated` since #218: `prep_de_side` now raises first on
    # this input, which would make the assertion below pass for the wrong reason and leave
    # `validate="1:1"` -- the guard that protects the HAND-ASSEMBLED path -- untested.
    # (The slicing drivers are not that path: they call `prepare_de` and do get the check.)
    with pytest.raises(pl.exceptions.ComputeError, match="1:1"):
        _direction_frame(_prep_unvalidated(real, pred))


def test_full_ranking_key_is_honoured_at_every_level():
    """The four-part key must be tested AS a key, and the fixture must be mutation-proof
    against deleting ANY of its four components.

    Rows are supplied in feature order A,B,C,D,F,E, which is deliberate:
      A,B  equal p_adj; A has the lower p_value but the SMALLER |lfc| -> A must win.
           Deleting the p_value tiebreak orders B,A.
      C,D  equal p_adj and p_value; C (the EARLIER feature) is the NaN one and D is
           defined -> D must win. Putting NaN on the LATER feature would let a fixture
           pass with the |lfc| key deleted entirely, since the feature tiebreak would
           produce the same order by accident.
      F,E  tie on every numeric key and carry the LOWEST p_value in the table (0.0001)
           while carrying the HIGHEST p_adj -> they must still sort last, which is what
           makes p_adj primary observable. They are supplied F-then-E so that E must win
           on feature ascending; supplying them E-then-F would let a fixture pass with
           the feature key deleted, since a stable sort preserves input order.
    Correct order is therefore A,B,D,C,E,F. Match/miss alternates along that order, so any
    reordering changes the purity sequence. Asserted through purity because the curve
    exposes no feature column.

    Mutation-tested: deleting p_adj, deleting the p_value tiebreak, deleting the |lfc| key,
    deleting the feature key, and failing to remap the NaN magnitude ALL change this
    sequence. An earlier version of this fixture caught only the last four -- with p_adj
    removed, its p_value groups happened to reproduce the same order.
    """
    from cell_eval2.metrics.direction import _direction_frame, _purity_curve
    feats = ["A", "B", "C", "D", "F", "E"]
    real = pl.DataFrame({
        "target": ["G1"] * 6, "feature": feats,
        # sorted A,B,D,C,E,F -> match, miss, match, miss(pred NaN), miss, match
        "log2_fold_change": [1.0, -1.0, 1.0, 1.0, 1.0, -1.0], "p_adj": [0.5] * 6,
    })
    pred = pl.DataFrame({
        "target": ["G1"] * 6, "feature": feats,
        "log2_fold_change": [1.0, 9.0, float("nan"), 5.0, 4.0, 4.0],
        "p_adj":   [0.01,  0.01,  0.02, 0.02, 0.03,   0.03],
        "p_value": [0.002, 0.003, 0.02, 0.02, 0.0001, 0.0001],
    })
    curve = _purity_curve(_direction_frame(_prep(real, pred)), universe="all", alpha=0.05)
    # order A,B,D,C,E,F -> matches 1,1,2,2,2,3 over denominators 1..6
    assert curve["purity"].to_list() == pytest.approx([1.0, 0.5, 2 / 3, 0.5, 0.4, 0.5])


def test_direction_frame_is_memoized_on_the_prepared_object():
    from cell_eval2.metrics.direction import _direction_frame
    real = pl.DataFrame({
        "target": ["G1"], "feature": ["A"], "log2_fold_change": [1.0], "p_adj": [0.5],
    })
    pred = pl.DataFrame({
        "target": ["G1"], "feature": ["A"], "log2_fold_change": [1.0], "p_adj": [0.001],
    })
    prep = _prep(real, pred)
    assert _direction_frame(prep) is _direction_frame(prep)


def test_purity_curve_depth_counts_adjudicable_pairs_not_ranked_rows():
    """Issue #204: an unadjudicable pair must NOT advance the depth.

    Before the fix `k` was `in_denom.cum_count()`, which counts NON-NULL rather than True
    entries -- and `_defined()` ends in `.fill_null(False)`, so it was simply the ranked
    row position and B below took k=2. On real data that padding was 97.9% of the median
    k* prefix. `k` and `n_denom` are now the same count by construction, which is what
    the second assertion pins.
    """
    from cell_eval2.metrics.direction import _direction_frame, _purity_curve
    # Ranked A, B, C. B is reference-unadjudicable (real lfc 0), so it neither advances
    # the depth nor enters n_denom. A and C match.
    real = pl.DataFrame({
        "target": ["G1"] * 3, "feature": list("ABC"),
        "log2_fold_change": [1.0, 0.0, 1.0], "p_adj": [0.5] * 3,
    })
    pred = pl.DataFrame({
        "target": ["G1"] * 3, "feature": list("ABC"),
        "log2_fold_change": [1.0, 1.0, 1.0], "p_adj": [0.01, 0.02, 0.03],
        "p_value": [0.01, 0.02, 0.03],
    })
    curve = _purity_curve(_direction_frame(_prep(real, pred)), universe="all", alpha=0.05)
    assert curve["k"].to_list() == [1, 1, 2]
    assert curve["k"].to_list() == curve["n_denom"].to_list()
    assert curve["n_denom"].to_list() == [1, 1, 2]
    assert curve["n_match"].to_list() == [1, 1, 2]
    assert curve["purity"].to_list() == [1.0, 1.0, 1.0]


def test_purity_curve_adjudicated_universe_keeps_only_reference_significant_genes():
    from cell_eval2.metrics.direction import _direction_frame, _purity_curve
    real = pl.DataFrame({
        "target": ["G1"] * 3, "feature": list("ABC"),
        "log2_fold_change": [1.0, 1.0, 1.0], "p_adj": [0.001, 0.5, 0.001],
    })
    pred = pl.DataFrame({
        "target": ["G1"] * 3, "feature": list("ABC"),
        "log2_fold_change": [1.0, 1.0, 1.0], "p_adj": [0.01, 0.02, 0.03],
        "p_value": [0.01, 0.02, 0.03],
    })
    frame = _direction_frame(_prep(real, pred))
    assert _purity_curve(frame, universe="all", alpha=0.05).height == 3
    adj = _purity_curve(frame, universe="adjudicated", alpha=0.05)
    assert adj.height == 2                       # B (real p_adj=0.5) dropped
    assert adj["k"].to_list() == [1, 2]


def test_purity_curve_purity_is_null_while_no_defined_pair_seen():
    from cell_eval2.metrics.direction import _direction_frame, _purity_curve
    real = pl.DataFrame({
        "target": ["G1"] * 2, "feature": list("AB"),
        "log2_fold_change": [0.0, 1.0], "p_adj": [0.5, 0.5],
    })
    pred = pl.DataFrame({
        "target": ["G1"] * 2, "feature": list("AB"),
        "log2_fold_change": [1.0, 1.0], "p_adj": [0.01, 0.02],
        "p_value": [0.01, 0.02],
    })
    curve = _purity_curve(_direction_frame(_prep(real, pred)), universe="all", alpha=0.05)
    assert curve["purity"].to_list() == [None, 1.0]


def test_purity_curve_rejects_an_unknown_universe():
    from cell_eval2.metrics.direction import _direction_frame, _purity_curve
    real = pl.DataFrame({
        "target": ["G1"], "feature": ["A"], "log2_fold_change": [1.0], "p_adj": [0.001],
    })
    pred = pl.DataFrame({
        "target": ["G1"], "feature": ["A"], "log2_fold_change": [1.0], "p_adj": [0.001],
    })
    with pytest.raises(ValueError, match="universe"):
        _purity_curve(_direction_frame(_prep(real, pred)), universe="everything", alpha=0.05)


def test_k_star_takes_the_deepest_k_not_the_first_dip():
    from cell_eval2.metrics.direction import _k_star
    curve = pl.DataFrame({
        "target": ["G1"] * 4, "k": [1, 2, 3, 4], "purity": [1.0, 0.5, 0.666, 0.75],
    })
    assert _k_star(curve, p0=0.6) == {"G1": 4}     # deepest, past the dip at k=2
    assert _k_star(curve, p0=0.9) == {"G1": 1}


def test_k_star_omits_a_target_that_never_reaches_p0():
    from cell_eval2.metrics.direction import _k_star
    curve = pl.DataFrame({"target": ["G1", "G1"], "k": [1, 2], "purity": [0.4, 0.5]})
    assert _k_star(curve, p0=0.975) == {}


def test_non_numeric_p_value_column_is_treated_as_absent(caplog):
    """An all-blank 'p_value' column in a user CSV arrives as String, because p_value is
    not in REQUIRED_COLS and load_de_table only pins target/feature dtypes. is_nan() raises
    InvalidOperationError on a non-numeric dtype, so before the guard this crashed all
    three metrics outright. An unusable column must degrade to the p_adj-only ranking."""
    from cell_eval2.metrics.direction import _direction_frame
    real = pl.DataFrame({
        "target": ["G1"] * 2, "feature": list("AB"),
        "log2_fold_change": [1.0, 1.0], "p_adj": [0.5, 0.5],
    })
    pred = pl.DataFrame({
        "target": ["G1"] * 2, "feature": list("AB"),
        "log2_fold_change": [1.0, 1.0], "p_adj": [0.02, 0.01],
        "p_value": [None, None],
    }, schema_overrides={"p_value": pl.String})
    prep = _prep(real, pred)
    with caplog.at_level(logging.WARNING, logger="cell_eval2.metrics.direction"):
        frame = _direction_frame(prep)
    assert "non-numeric" in caplog.text
    # falls back to the p_adj ordering: B (0.01) ahead of A (0.02)
    assert frame.sort("rank_p_adj")["feature"].to_list() == ["B", "A"]


def test_numeric_all_null_p_value_column_is_still_used():
    """The guard rejects a non-numeric DTYPE, not null VALUES. A Float64 p_value that
    happens to be entirely null is usable -- _rank_p maps its nulls to +inf."""
    from cell_eval2.metrics.direction import _direction_frame
    real = pl.DataFrame({
        "target": ["G1"] * 2, "feature": list("AB"),
        "log2_fold_change": [1.0, 1.0], "p_adj": [0.5, 0.5],
    })
    pred = pl.DataFrame({
        "target": ["G1"] * 2, "feature": list("AB"),
        "log2_fold_change": [1.0, 1.0], "p_adj": [0.02, 0.01],
        "p_value": [None, None],
    }, schema_overrides={"p_value": pl.Float64})
    frame = _direction_frame(_prep(real, pred))
    assert frame["rank_p_value"].to_list() == [float("inf"), float("inf")]


def test_decimal_p_columns_do_not_raise():
    """polars raises InvalidOperationError on is_nan() for a Decimal column, and
    Decimal.is_numeric() is True so the dtype guard admits it. A parquet DE table can
    carry Decimal p-columns. This covers BOTH the optional p_value and the REQUIRED
    p_adj, which reaches _rank_p with no upstream screen of its own."""
    import decimal
    from cell_eval2.metrics.direction import _direction_frame

    def _table(decimal_col: str) -> pl.DataFrame:
        cols = {"target": ["G1"] * 2, "feature": list("AB"),
                "log2_fold_change": [1.0, 2.0]}
        overrides = {decimal_col: pl.Decimal(10, 6)}
        decimals = [decimal.Decimal("0.001"), decimal.Decimal("0.002")]
        if decimal_col == "p_adj":
            cols["p_adj"] = decimals
        else:
            cols["p_adj"] = [0.001, 0.002]
            cols["p_value"] = decimals
        return pl.DataFrame(cols, schema_overrides=overrides)

    for col in ("p_value", "p_adj"):
        table = _table(col)
        frame = _direction_frame(_prep(table, table)).sort("feature")
        assert frame.height == 2, col
        # Assert BEHAVIOUR, not dtype: a dtype assertion would lock in the Float64
        # normalisation as a contract rather than a documented limitation, and the
        # p_value case would pass even if a Decimal p_value were treated as absent.
        assert frame["rank_p_adj"].to_list() == [0.001, 0.002], col
        expected_pv = [0.001, 0.002] if col == "p_value" else [0.0, 0.0]
        assert frame["rank_p_value"].to_list() == expected_pv, col


def test_high_precision_decimal_ranking_is_float64_normalised():
    """Pins the DOCUMENTED limitation in _rank_p so it stays a decision, not an accident.

    Two Decimal p_values differing only beyond Float64's ~15 significant digits collide
    once cast, so the later key components decide the order. Here that costs the ranking:
    sensitivity reads 0.0 where exact-Decimal ordering would give 0.5. If this ever starts
    returning 0.5, someone has made the ranking dtype-aware -- update the docstring, the
    CHANGELOG note and this test together."""
    import decimal
    from cell_eval2.metrics.direction import de_direction_sensitivity
    real = pl.DataFrame({
        "target": ["G1"] * 2, "feature": list("AB"),
        "log2_fold_change": [1.0, 1.0], "p_adj": [0.001, 0.001],
    })
    pred = pl.DataFrame({
        "target": ["G1"] * 2, "feature": list("AB"),
        "log2_fold_change": [-1.0, 1.0],          # A mismatches, B matches
        "p_adj": [0.01, 0.01],                    # tied, so p_value decides
        "p_value": [decimal.Decimal("0.001000000000000000002"),
                    decimal.Decimal("0.001000000000000000001")],
    }, schema_overrides={"p_value": pl.Decimal(30, 21)})
    got = de_direction_sensitivity(_prep(real, pred), universe="adjudicated")
    assert got == {"G1": 0.0}


def test_direction_frame_falls_back_and_warns_exactly_once_without_p_value(caplog):
    from cell_eval2.metrics.direction import _direction_frame
    real = pl.DataFrame({
        "target": ["G1"] * 2, "feature": list("AB"),
        "log2_fold_change": [1.0, 1.0], "p_adj": [0.5, 0.5],
    })
    pred = pl.DataFrame({
        "target": ["G1"] * 2, "feature": list("AB"),
        "log2_fold_change": [1.0, 1.0], "p_adj": [0.02, 0.01],
    })
    prep = _prep(real, pred)
    with caplog.at_level(logging.WARNING, logger="cell_eval2.metrics.direction"):
        _direction_frame(prep)
        _direction_frame(prep)      # memoized -> must NOT warn again
    assert len([r for r in caplog.records if "p_value" in r.message]) == 1
    # p_adj still orders correctly: B (0.01) ahead of A (0.02)
    assert _direction_frame(prep).sort("rank_p_adj")["feature"].to_list() == ["B", "A"]
