import polars as pl
import pytest

from cell_eval2.de import prepare_de


def _prep(real, pred, *, threshold=0.05, policy="keep", floor=0.0):
    return prepare_de(pred, real, control="non-targeting", p_adj_threshold=threshold,
                      nan_lfc_policy=policy, min_abs_log2fc=floor)


def test_precision_uses_the_model_significant_denominator():
    from cell_eval2.metrics.direction import de_direction_precision
    # Model-sig {B, C}; real B agrees, C disagrees -> 0.5. A is real-significant but
    # model-non-significant and must not enter the denominator.
    real = pl.DataFrame({
        "target": ["G1"] * 3, "feature": list("ABC"),
        "log2_fold_change": [3.0, -2.0, 4.0], "p_adj": [0.001, 0.5, 0.5],
    })
    pred = pl.DataFrame({
        "target": ["G1"] * 3, "feature": list("ABC"),
        "log2_fold_change": [-1.0, -4.0, -2.0], "p_adj": [0.5, 0.001, 0.001],
    })
    assert de_direction_precision(_prep(real, pred))["G1"] == 0.5


# --- spec section 8.2: one PUBLIC test per undefined-direction quadrant ----------------

@pytest.mark.parametrize(
    "real_lfc, pred_lfc, expected, why",
    [
        (0.0,          5.0,          1.0, "real zero -> excluded, denominator 1"),
        (float("nan"), 5.0,          1.0, "real NaN  -> excluded, denominator 1"),
        (3.0,          0.0,          0.5, "pred zero -> miss, denominator 2"),
        (3.0,          float("nan"), 0.5, "pred NaN  -> miss, denominator 2"),
    ],
)
def test_precision_undefined_direction_quadrants(real_lfc, pred_lfc, expected, why):
    from cell_eval2.metrics.direction import de_direction_precision
    # Gene A is always a clean match. Gene B carries the quadrant under test. Both are
    # model-significant, so B leaves the denominator only when the REFERENCE cannot
    # adjudicate it. Exactly one side is undefined in each row, so reference-exclusion
    # can never mask the pred-side rule.
    real = pl.DataFrame({
        "target": ["G1"] * 2, "feature": list("AB"),
        "log2_fold_change": [1.0, real_lfc], "p_adj": [0.5, 0.5],
    })
    pred = pl.DataFrame({
        "target": ["G1"] * 2, "feature": list("AB"),
        "log2_fold_change": [2.0, pred_lfc], "p_adj": [0.001, 0.001],
    })
    assert de_direction_precision(_prep(real, pred))["G1"] == expected, why


def test_precision_excludes_a_model_significant_gene_absent_from_real():
    from cell_eval2.metrics.direction import de_direction_precision
    # B is model-significant but the reference never tested it -> excluded, not a miss.
    real = pl.DataFrame({
        "target": ["G1"], "feature": ["A"], "log2_fold_change": [1.0], "p_adj": [0.5],
    })
    pred = pl.DataFrame({
        "target": ["G1"] * 2, "feature": list("AB"),
        "log2_fold_change": [2.0, -9.0], "p_adj": [0.001, 0.001],
    })
    assert de_direction_precision(_prep(real, pred))["G1"] == 1.0


def test_precision_omits_a_target_with_an_empty_denominator():
    from cell_eval2.metrics.direction import de_direction_precision
    real = pl.DataFrame({
        "target": ["G1"], "feature": ["A"], "log2_fold_change": [2.0], "p_adj": [0.001],
    })
    pred = pl.DataFrame({
        "target": ["G1"], "feature": ["A"], "log2_fold_change": [2.0], "p_adj": [0.5],
    })
    assert de_direction_precision(_prep(real, pred)) == {}


# --- the identity, including the three cases that broke the first draft ---------------

def _identity_holds(prep) -> None:
    """precision == purity at the END of the S_pred prefix, exactly.

    RE-INDEXED for issue #204. Depth now counts ADJUDICABLE pairs, so that boundary sits
    at k = |S_pred INTERSECT adjudicable| rather than at k = |S_pred|. The two coincide
    whenever S_pred carries no unadjudicable pair, which is true of every fixture here
    except test_identity_is_reindexed_...; that one is the reason this helper had to
    change, and the only one that can tell the two indices apart.

    Unambiguous despite k now repeating: an unadjudicable row leaves n_denom AND n_match
    untouched (it can never be a match), so every row sharing a k value shares its purity.
    """
    from cell_eval2.metrics.direction import (
        _direction_frame, _purity_curve, de_direction_precision,
    )
    precision = de_direction_precision(prep)
    frame = _direction_frame(prep)
    curve = _purity_curve(frame, universe="all", alpha=prep.p_adj_threshold)
    boundaries = (
        frame.filter(pl.col("p_adj_pred") < prep.p_adj_threshold)
        .group_by("target")
        .agg(k=pl.col("in_denom").cast(pl.Int64).sum())
    )
    assert boundaries.height, "fixture must have at least one model-significant gene"
    # A target whose S_pred is ENTIRELY unadjudicable has no precision at all (n_denom == 0,
    # so `de_direction_precision` omits it) and no curve row above k = 0 to compare against:
    # the identity is undefined there, not violated. That case is pinned separately by
    # test_precision_omits_a_target_whose_significant_genes_are_all_unadjudicable. Drop such
    # targets rather than KeyError on `precision[target]` under the misleading
    # "at least one model-significant gene" message -- and re-assert non-emptiness, so the
    # helper can never end up checking nothing.
    boundaries = boundaries.filter(pl.col("k") > 0)
    assert boundaries.height, (
        "every model-significant gene in this fixture is reference-unadjudicable, so no "
        "target has a precision value and the identity is not exercised at all"
    )
    for target, k in zip(boundaries["target"].to_list(), boundaries["k"].to_list(),
                         strict=True):
        at_k = curve.filter((pl.col("target") == target) & (pl.col("k") == k))["purity"][0]
        assert abs(precision[target] - at_k) == 0.0, target


def _discriminates(prep) -> None:
    """Assert the fixture would actually FAIL under the rejected p_value-primary key.

    Without this, a fixture that merely *contains* a mask or a floor proves nothing: two of
    the three originally written for this spec had the same number of matches in both
    candidate prefixes, so both orderings gave 0.5 and the tests passed under the very
    implementation they were written to reject. Recomputing the rejected ordering inline
    keeps that failure mode from coming back silently.
    """
    from cell_eval2.metrics.direction import _direction_frame, de_direction_precision
    precision = de_direction_precision(prep)
    frame = _direction_frame(prep)
    checked = 0
    for target, want in precision.items():
        rows = frame.filter(pl.col("target") == target)
        k = rows.filter(pl.col("p_adj_pred") < prep.p_adj_threshold).height
        # The REJECTED ordering, reconstructed in FULL: p_value, then |lfc| desc, then
        # feature. Sorting on p_value alone stops reproducing it the moment a fixture has
        # tied p-values, which would make this helper quietly stop checking anything.
        top = rows.sort(["rank_p_value", "abs_lfc_pred", "feature"],
                        descending=[False, True, False]).head(k)
        n_denom = top["in_denom"].cast(pl.Int64).sum()
        wrong = (top["match"].cast(pl.Int64).sum() / n_denom) if n_denom else None
        assert wrong != want, (
            f"{target}: p_value-primary ordering also yields {want}; this fixture does not "
            f"discriminate and would pass under the rejected implementation"
        )
        checked += 1
    assert checked, "no target had a scoreable precision -- this helper passed vacuously"


def test_identity_on_clean_bh_consistent_input():
    """precision == purity(k=|S_pred|) on the FULL-UNIVERSE curve, exactly. This only
    holds if the curve's denominator is n_denom rather than k -- check that before
    loosening anything."""
    real = pl.DataFrame({
        "target": ["G1"] * 4 + ["G2"] * 4, "feature": list("ABCD") * 2,
        "log2_fold_change": [1.0, 4.0, 2.0, -5.0, -1.0, -2.0, 3.0, 1.0],
        "p_adj": [0.5] * 8,
    })
    pred = pl.DataFrame({
        "target": ["G1"] * 4 + ["G2"] * 4, "feature": list("ABCD") * 2,
        "log2_fold_change": [3.0, -2.0, 1.0, -1.0, -3.0, 2.0, 1.0, -1.0],
        "p_adj": [0.001, 0.01, 0.5, 0.9, 0.002, 0.02, 0.6, 0.8],
        "p_value": [0.0001, 0.002, 0.3, 0.8, 0.0003, 0.004, 0.4, 0.7],
    })
    _identity_holds(_prep(real, pred))


def test_identity_survives_the_nan_mask():
    """nan_lfc_policy='mask' (the v2 DEFAULT) forces p_adj=1 on a NaN-LFC row while
    leaving its small p_value alone. A p_value-primary ranking key put that row inside the
    top-|S_pred| and broke the identity; p_adj-primary does not."""
    real = pl.DataFrame({
        "target": ["G1"] * 4, "feature": list("ABCD"),
        "log2_fold_change": [1.0, 1.0, 1.0, 1.0], "p_adj": [0.5] * 4,
    })
    pred = pl.DataFrame({
        "target": ["G1"] * 4, "feature": list("ABCD"),
        "log2_fold_change": [3.0, 2.0, float("nan"), 2.0],   # C is NaN -> masked to p_adj=1
        "p_adj": [0.001, 0.002, 0.003, 0.004],
        "p_value": [0.0001, 0.0002, 0.0003, 0.0004],
    })
    prep = _prep(real, pred, policy="mask")
    _identity_holds(prep)
    _discriminates(prep)


def test_identity_survives_the_lfc_floor():
    """apply_lfc_floor forces p_adj=1 below the floor, again without touching p_value.

    real A is -1.0 so that A is a MISS. With A a match the two candidate prefixes
    ({B,C} correct, {A,B} rejected) both contain exactly one match and both score 0.5,
    and the test passes under the implementation it exists to reject -- verified."""
    real = pl.DataFrame({
        "target": ["G1"] * 3, "feature": list("ABC"),
        "log2_fold_change": [-1.0, -1.0, 1.0], "p_adj": [0.5] * 3,
    })
    pred = pl.DataFrame({
        "target": ["G1"] * 3, "feature": list("ABC"),
        "log2_fold_change": [0.05, 2.0, 3.0],       # A is below the floor
        "p_adj": [0.001, 0.01, 0.02],
        "p_value": [0.001, 0.01, 0.02],
    })
    prep = _prep(real, pred, policy="mask", floor=0.5)
    _identity_holds(prep)
    _discriminates(prep)                 # p_adj-primary 0.5 vs p_value-primary 0.0


def test_identity_survives_a_supplied_table_whose_p_adj_is_not_bh_of_p_value():
    """A user CSV can carry any p_adj at all. p_adj-primary ranking makes the identity
    independent of whether the two columns are mutually consistent.

    real A is -1.0 for the same reason as the floor fixture above."""
    real = pl.DataFrame({
        "target": ["G1"] * 3, "feature": list("ABC"),
        "log2_fold_change": [-1.0, -1.0, 1.0], "p_adj": [0.5] * 3,
    })
    pred = pl.DataFrame({
        "target": ["G1"] * 3, "feature": list("ABC"),
        "log2_fold_change": [2.0, 2.0, 3.0],
        "p_adj":   [0.9,   0.01, 0.02],     # deliberately anti-monotone in p_value
        "p_value": [0.001, 0.01, 0.02],
    })
    prep = _prep(real, pred)
    _identity_holds(prep)
    _discriminates(prep)                 # p_adj-primary 0.5 vs p_value-primary 0.0


def test_identity_is_reindexed_when_s_pred_contains_an_unadjudicable_pair():
    """Issue #204: the identity holds at k = |S_pred INTERSECT adjudicable|, not |S_pred|.

    None of the four fixtures above can see the difference -- every reference row in them
    carries a direction, so the two indices coincide and all four passed under BOTH depth
    definitions (verified against `in_denom.cum_sum()` before the fix landed). That is why
    the docstring's claim that row-counting depth was load-bearing for the identity went
    unchallenged for so long: nothing tested it.

    Here B is model-significant but reference-unadjudicable (real lfc 0), so it sits INSIDE
    S_pred: |S_pred| = 3 while the boundary is k = 2. A matches, C is a genuine miss.
    """
    real = pl.DataFrame({
        "target": ["G1"] * 4, "feature": list("ABCD"),
        "log2_fold_change": [1.0, 0.0, 1.0, 1.0], "p_adj": [0.5] * 4,
    })
    pred = pl.DataFrame({
        "target": ["G1"] * 4, "feature": list("ABCD"),
        "log2_fold_change": [1.0, 1.0, -1.0, 1.0],
        "p_adj": [0.001, 0.002, 0.003, 0.5],       # D is not model-significant
        "p_value": [0.001, 0.002, 0.003, 0.5],
    })
    prep = _prep(real, pred)
    _identity_holds(prep)

    # ...and pin that the fixture DISCRIMINATES, for the same reason `_discriminates`
    # exists: under the pre-#204 row-counting depth the identity's lookup lands on an
    # EARLIER prefix and returns the wrong purity. Recomputed inline so a revert fails
    # here loudly rather than passing vacuously.
    from cell_eval2.metrics.direction import _direction_frame, de_direction_precision
    assert de_direction_precision(prep)["G1"] == 0.5
    old_depth = (
        _direction_frame(prep)
        .sort(["target", "rank_p_adj", "rank_p_value", "abs_lfc_pred", "feature"],
              descending=[False, False, False, True, False])
        .with_columns(
            k_rows=pl.col("in_denom").cum_count().over("target").cast(pl.Int64),
            n_denom=pl.col("in_denom").cast(pl.Int64).cum_sum().over("target"),
            n_match=pl.col("match").cast(pl.Int64).cum_sum().over("target"),
        )
        .with_columns(purity=pl.col("n_match") / pl.col("n_denom"))
    )
    assert old_depth.filter(pl.col("k_rows") == 2)["purity"][0] == 1.0, (
        "fixture does not discriminate: row-counting depth yields the same purity at the "
        "boundary, so this test would pass under the behaviour it exists to reject"
    )


def test_precision_omits_a_target_whose_significant_genes_are_all_unadjudicable():
    """The second empty-denominator route from spec section 5.2: S_pred is NON-empty, but
    every member has an undefined real direction. Distinct from the S_pred-empty case, and
    the one that would regress silently."""
    from cell_eval2.metrics.direction import de_direction_precision
    real = pl.DataFrame({
        "target": ["G1"], "feature": ["A"], "log2_fold_change": [0.0], "p_adj": [0.5],
    })
    pred = pl.DataFrame({
        "target": ["G1"], "feature": ["A"], "log2_fold_change": [2.0], "p_adj": [0.001],
    })
    assert de_direction_precision(_prep(real, pred)) == {}


def test_model_direction_match_is_untouched_and_now_disagrees():
    """The frozen metric must keep its old semantics. Both fixtures are cases where polars
    scores an undefined direction as agreement (0 == 0, NaN == NaN); the new metric
    excludes them. If a future edit 'unifies' the two, this fails loudly."""
    from cell_eval2.metrics.de import de_model_direction_match
    from cell_eval2.metrics.direction import de_direction_precision

    zero = _prep(
        pl.DataFrame({
            "target": ["G1"] * 2, "feature": list("AB"),
            "log2_fold_change": [-1.0, 0.0], "p_adj": [0.5, 0.5],
        }),
        pl.DataFrame({
            "target": ["G1"] * 2, "feature": list("AB"),
            "log2_fold_change": [2.0, 0.0], "p_adj": [0.001, 0.001],
        }),
    )
    assert de_model_direction_match(zero)["G1"] == 0.5   # 0 == 0 scored as agreement
    assert de_direction_precision(zero)["G1"] == 0.0     # B excluded; A is a real miss

    nan = _prep(
        pl.DataFrame({
            "target": ["G1"] * 2, "feature": list("AB"),
            "log2_fold_change": [float("nan"), -3.0], "p_adj": [0.5, 0.5],
        }),
        pl.DataFrame({
            "target": ["G1"] * 2, "feature": list("AB"),
            "log2_fold_change": [float("nan"), 2.0], "p_adj": [0.001, 0.001],
        }),
    )
    assert de_model_direction_match(nan)["G1"] == 0.5    # NaN == NaN scored as agreement
    assert de_direction_precision(nan)["G1"] == 0.0      # A excluded; B is a real miss
