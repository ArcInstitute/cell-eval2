import math

import polars as pl
import pytest

from cell_eval2.catalog import CATALOG, PROFILES, resolve_metrics
from cell_eval2.config import EvalConfig
from cell_eval2.de import (
    apply_nan_policy,
    assemble_prepared_de,
    normalize_de_schema,
    prepare_de,
    rank_de_side,
)
from cell_eval2.metrics.de import de_nsig_counts, de_sig_jaccard
from cell_eval2.run import dispatch_de_metrics
from cell_eval2.scoring import BOUNDED, BOUNDED_UNFLOORED


METRIC = "de_wilcoxon_sig_jaccard"


def _toy(real, pred, threshold=0.05):
    return prepare_de(pred, real, control="non-targeting", p_adj_threshold=threshold)


def _unvalidated(real, pred, threshold=0.05):
    """A PreparedDE assembled WITHOUT `prep_de_side`, so #218's duplicate-key check never
    runs — the direct-assembly path, where the metric's own guards still matter. (Not the
    slicing drivers: those call `prepare_de` and do get the check.)"""
    def side(df, name):
        out = apply_nan_policy(normalize_de_schema(df, name=name), name=name,
                               nan_lfc_policy="mask")
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


def _framed(target, all_features, sig_features, *, p_adj=None, on_target=True):
    """One-target DE frame with every tested gene represented exactly once.

    ⚠️ `on_target=True` (the default) appends ONE extra row whose feature IS the target, held
    SIGNIFICANT on whichever side it is built for. Two things follow, and both are the point
    (issue #172):

    * the target now RESOLVES against the real feature index, so `de_sig_jaccard` runs with a
      non-empty `TargetResolution` instead of tripping the zero-resolve gate;
    * every closed form below is stated over `all_features` ALONE, so each of these tests fails
      unless the on-target row is excluded from BOTH sides. It is a significant hit on both, so
      forgetting to exclude it would add 1 to the intersection AND to the union.

    `on_target=False` builds the pre-#172 shape, for the tests that assert the gate fires.
    """
    features = list(all_features)
    p_adj_col = ([0.001 if f in sig_features else 0.9 for f in features]
                 if p_adj is None else list(p_adj))
    if on_target:
        features = features + [target]
        p_adj_col = p_adj_col + [0.001]
    return pl.DataFrame({
        "target": [target] * len(features),
        "feature": features,
        "log2_fold_change": [2.0] * len(features),
        "p_adj": p_adj_col,
    })


def test_closed_form():
    # real a=4, pred b=3, TP=2, union=5 -> 2/5
    genes = [f"g{i}" for i in range(6)]
    real = _framed("P", genes, {"g0", "g1", "g2", "g3"})
    pred = _framed("P", genes, {"g2", "g3", "g4"})
    assert de_sig_jaccard(_toy(real, pred))["P"] == 0.4


def test_perfect_nonempty_sets():
    genes = ["g0", "g1", "g2"]
    prepared = _toy(_framed("P", genes, {"g0", "g2"}),
                    _framed("P", genes, {"g0", "g2"}))
    assert de_sig_jaccard(prepared)["P"] == 1.0


def test_disjoint_nonempty_sets():
    genes = ["g0", "g1", "g2"]
    prepared = _toy(_framed("P", genes, {"g0"}), _framed("P", genes, {"g1", "g2"}))
    assert de_sig_jaccard(prepared)["P"] == 0.0


def test_empty_union_uses_set_convention():
    genes = ["g0", "g1", "g2"]
    prepared = _toy(_framed("P", genes, set()), _framed("P", genes, set()))
    assert de_sig_jaccard(prepared)["P"] == 1.0


@pytest.mark.parametrize(("real_sig", "pred_sig"), [
    (set(), {"g0"}),
    ({"g0"}, set()),
])
def test_exactly_one_side_empty(real_sig, pred_sig):
    genes = ["g0", "g1", "g2"]
    prepared = _toy(_framed("P", genes, real_sig), _framed("P", genes, pred_sig))
    assert de_sig_jaccard(prepared)["P"] == 0.0


def test_symmetric_when_real_and_pred_are_swapped():
    genes = [f"g{i}" for i in range(6)]
    left = _framed("P", genes, {"g0", "g1", "g2", "g3"})
    right = _framed("P", genes, {"g2", "g3", "g4"})
    forward = de_sig_jaccard(_toy(left, right))["P"]
    reverse = de_sig_jaccard(_toy(right, left))["P"]
    assert forward == reverse


def test_against_python_set_intersection_and_union():
    genes = [f"g{i}" for i in range(6)]
    real_sets = {"P1": {"g0", "g1", "g2"}, "P2": {"g3"}, "P3": set()}
    pred_sets = {"P1": {"g1", "g2", "g4"}, "P2": {"g0", "g3"}, "P3": set()}
    real = pl.concat([_framed(p, genes, sig) for p, sig in real_sets.items()])
    pred = pl.concat([_framed(p, genes, sig) for p, sig in pred_sets.items()])
    prepared = _toy(real, pred)

    a = de_nsig_counts(prepared, side="real")
    b = de_nsig_counts(prepared, side="pred")
    observed = de_sig_jaccard(prepared)
    for p in prepared.perturbations:
        # `de_nsig_counts` is NOT one of the six scored `vcc2026` members, so #172's exclusion
        # deliberately does not reach it: it counts the on-target row `_framed` appends, and
        # `de_sig_jaccard` below does not. The +1 IS the assertion that the exclusion is
        # metric-scoped rather than applied to the prepared frames.
        assert a[p] == len(real_sets[p]) + 1
        assert b[p] == len(pred_sets[p]) + 1
        # Both operands are plain Python sets, independent of the metric's counting/join seam.
        inter = len(real_sets[p] & pred_sets[p])
        union = len(real_sets[p] | pred_sets[p])
        expected = 1.0 if union == 0 else inter / union
        assert observed[p] == expected


def test_prepare_de_refuses_a_duplicated_key_before_any_metric_sees_it():
    """#218's seam. A duplicated `(target, feature)` row used to reach the metrics, where
    it got three different answers; `prep_de_side` now refuses it for all of them."""
    genes = ["g0", "g1", "g2"]
    real = _framed("P", genes, {"g0", "g1"})
    pred = _framed("P", genes, {"g0"})
    pred_g0 = pred.filter(pl.col("feature") == "g0")
    with pytest.raises(ValueError, match=r"duplicated \(target, feature\) key"):
        _toy(real, pl.concat([pred, pred_g0]))


def test_duplicate_multiplicity_is_set_deduplicated_and_bounded():
    """`de_sig_jaccard`'s own `.unique()` is KEPT after #218 and still tested.

    It is reachable on the HAND-ASSEMBLED path, which does not enter `prep_de_side`, so
    removing it would trade a metric that stays inside `[0, 1]` on a malformed input for one
    that does not. (Not the slicing drivers -- they call `prepare_de` and do get the check.) The frames are therefore assembled directly
    here rather than through `prepare_de`, which now raises on this input."""
    genes = ["g0", "g1", "g2"]

    real = _framed("P", genes, {"g0", "g1"})
    pred = _framed("P", genes, {"g0"})
    pred_g0 = pred.filter(pl.col("feature") == "g0")
    asymmetric = de_sig_jaccard(_unvalidated(real, pl.concat([pred, pred_g0])))["P"]
    assert asymmetric == 0.5
    assert 0.0 <= asymmetric <= 1.0

    real = _framed("P", genes, {"g0", "g1"})
    pred = _framed("P", genes, {"g0", "g2"})
    real_g0 = real.filter(pl.col("feature") == "g0")
    pred_g0 = pred.filter(pl.col("feature") == "g0")
    symmetric = de_sig_jaccard(_unvalidated(
        pl.concat([real, real_g0, real_g0]),
        pl.concat([pred, pred_g0, pred_g0]),
    ))["P"]
    assert symmetric == pytest.approx(1 / 3)
    assert 0.0 <= symmetric <= 1.0


def test_no_perturbation_is_dropped_and_all_values_are_bounded():
    genes = ["g0", "g1", "g2"]
    real = pl.concat([
        _framed("P1", genes, {"g0", "g1"}),
        _framed("P2", genes, set()),
        _framed("P3", genes, {"g2"}),
    ])
    pred = pl.concat([
        _framed("P1", genes, {"g1"}),
        _framed("P2", genes, set()),
        _framed("P3", genes, {"g0"}),
    ])
    prepared = _toy(real, pred)
    out = de_sig_jaccard(prepared)
    assert set(out) == set(prepared.perturbations)
    assert out["P2"] == 1.0
    assert all(math.isfinite(value) and 0.0 <= value <= 1.0 for value in out.values())


def test_threshold_changes_membership_and_value():
    genes = ["g0", "g1", "g2"]
    real = _framed("P", genes, set(), p_adj=[0.01, 0.08, 0.9])
    pred = _framed("P", genes, set(), p_adj=[0.01, 0.9, 0.08])
    strict = de_sig_jaccard(_toy(real, pred, threshold=0.05))["P"]
    relaxed = de_sig_jaccard(_toy(real, pred, threshold=0.1))["P"]
    assert strict == 1.0
    assert relaxed == pytest.approx(1 / 3)
    assert strict != relaxed


def test_threshold_boundary_is_exclusive():
    genes = ["g0", "g1", "g2"]
    real = _framed("P", genes, set(), p_adj=[0.05, 0.9, 0.9])
    pred = _framed("P", genes, set(), p_adj=[0.001, 0.9, 0.9])
    assert de_sig_jaccard(_toy(real, pred, threshold=0.05))["P"] == 0.0


def test_null_and_nan_p_adj_rows_are_not_significant():
    genes = ["g0", "g1", "g2", "g3", "g4"]
    real = _framed("P", genes, set(), p_adj=[0.001, None, float("nan"), 0.9, 0.001])
    pred = _framed("P", genes, set(), p_adj=[0.001, float("nan"), None, 0.001, 0.9])
    value = de_sig_jaccard(_toy(real, pred))["P"]
    assert value == pytest.approx(1 / 3)
    assert 0.0 <= value <= 1.0


def test_standalone_forwards_min_abs_log2fc_into_preparation():
    genes = ["g0", "g1"]
    real = _framed("P", genes, {"g0"})
    pred = _framed("P", genes, {"g0", "g1"}).with_columns(
        pl.when(pl.col("feature") == "g0")
        .then(0.25)
        .otherwise(pl.col("log2_fold_change"))
        .alias("log2_fold_change")
    )
    unfiltered = de_sig_jaccard(de_pred=pred, de_real=real)["P"]
    filtered = de_sig_jaccard(
        de_pred=pred,
        de_real=real,
        min_abs_log2fc=1.0,
    )["P"]
    assert unfiltered == 0.5
    assert filtered == 0.0
    assert filtered != unfiltered


def test_standalone_raw_frames_match_prepared_path():
    genes = [f"g{i}" for i in range(6)]
    real = _framed("P", genes, {"g0", "g1", "g2", "g3"})
    pred = _framed("P", genes, {"g2", "g3", "g4"})
    prepared = de_sig_jaccard(_toy(real, pred))
    standalone = de_sig_jaccard(
        de_pred=pred,
        de_real=real,
        control="non-targeting",
        p_adj_threshold=0.05,
    )
    assert standalone == prepared


def test_catalog_wiring_and_deseq2_sibling():
    assert METRIC in PROFILES["full"]
    assert METRIC in PROFILES["de"]
    assert METRIC in PROFILES["vcc2026"]
    assert METRIC not in PROFILES["vcc"]
    assert METRIC not in PROFILES["minimal"]

    spec = CATALOG[METRIC]
    assert spec.func is de_sig_jaccard
    # `BOUNDED_UNFLOORED`, not `BOUNDED`: a `vcc2026` member, so its clip at 0 was removed.
    # Everything else about the policy is `BOUNDED`'s -- asserted field-by-field so this stays
    # a wiring test rather than an identity check against whichever preset it happens to use.
    assert spec.scoring == BOUNDED_UNFLOORED
    import dataclasses

    assert (spec.scoring.direction, spec.scoring.anchor) == ("higher", 1.0)
    assert spec.scoring.clamp_low is None and spec.scoring.metric_min == 0.0
    differing = {f.name for f in dataclasses.fields(BOUNDED)
                 if getattr(BOUNDED, f.name) != getattr(spec.scoring, f.name)}
    assert differing == {"clamp_low", "metric_min"}, (
        "the unfloored policy must differ from BOUNDED in the floor and nothing else"
    )
    assert spec.agg == "mean"
    assert spec.worst_value is None
    assert spec.v1_name is None
    assert spec.v1_available is False
    with pytest.raises(ValueError, match="not available under version='v1'"):
        resolve_metrics([METRIC], version="v1")

    sibling = CATALOG["de_deseq2_sig_jaccard"]
    assert sibling.func is de_sig_jaccard
    # The sibling moves with it: a metric must not change policy because the DE backend did.
    assert sibling.scoring == BOUNDED_UNFLOORED
    assert sibling.agg == "mean"
    assert sibling.worst_value is None
    assert sibling.profiles == ()
    assert sibling.v1_name is None
    assert sibling.v1_available is False


def test_dispatch_emits_one_row_per_perturbation():
    genes = ["g0", "g1", "g2"]
    real = pl.concat([
        _framed("P1", genes, {"g0", "g1"}),
        _framed("P2", genes, set()),
    ])
    pred = pl.concat([
        _framed("P1", genes, {"g1"}),
        _framed("P2", genes, set()),
    ])
    prepared = _toy(real, pred)
    rows = dispatch_de_metrics([METRIC], prepared, EvalConfig(metrics=[METRIC]))
    metric_rows = [row for row in rows if row["metric"] == METRIC]
    assert len(metric_rows) == len(prepared.perturbations)
    assert {row["perturbation"] for row in metric_rows} == set(prepared.perturbations)
    assert all(math.isfinite(row["value"]) for row in metric_rows)
