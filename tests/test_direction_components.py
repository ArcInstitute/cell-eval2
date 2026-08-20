import polars as pl
import pytest

from cell_eval2.de import TargetResolution, prepare_de
from cell_eval2.metrics.direction import (
    _components,
    _direction_frame,
    _ontarget_excluded_frame,
    _reference_stats,
)


def _tbl(rows):
    """(target, feature, lfc, p_adj) rows -> DE table."""
    return pl.DataFrame(
        {
            "target": [r[0] for r in rows],
            "feature": [r[1] for r in rows],
            "log2_fold_change": [float(r[2]) for r in rows],
            "p_adj": [float(r[3]) for r in rows],
        },
        schema={"target": pl.String, "feature": pl.String,
                "log2_fold_change": pl.Float64, "p_adj": pl.Float64},
    )


def _resolution(rows, mode):
    """mode='self' (default) -> every target maps to its own label; None -> derive."""
    if mode is None:
        return None
    if mode != "self":
        return mode
    targets = sorted({r[0] for r in rows})
    return TargetResolution({t: t for t in targets}, len(targets))


def _prep(pred_rows, real_rows, *, resolution="self"):
    """Default to a SELF-MAP resolution -- the H1_CGS shape.

    A fixture whose target label is not also a feature (e.g. target 'A', features 'B','C')
    resolves ZERO targets and RAISES at the gate before any metric runs. The self-map is
    both the fix and the realistic case: every target names a measured gene, while its own
    gene has already been removed from its own rows.
    """
    return prepare_de(_tbl(pred_rows), _tbl(real_rows), control="non-targeting",
                      p_adj_threshold=0.05,
                      target_resolution=_resolution(real_rows, resolution))


def test_excluded_frame_drops_the_resolved_on_target_row():
    """Self-map resolution {A: A, B: B} removes BOTH on-target rows: (A, A) and (B, B)."""
    rows = [("A", "A", 1.0, 0.01), ("A", "B", 1.0, 0.01), ("B", "B", 1.0, 0.01)]
    prep = _prep(rows, rows)
    full = _direction_frame(prep)
    excl = _ontarget_excluded_frame(prep)
    assert full.height == 3
    assert excl.height == 1
    assert excl.select(["target", "feature"]).rows() == [("A", "B")]


def test_excluded_frame_uses_the_RESOLVED_feature_not_the_raw_label():
    """Spec 2.7c: `filter(feature != target)` would ignore target_gene_map entirely."""
    rows = [("GENEX-1", "GENEX", 1.0, 0.01), ("GENEX-1", "OTHER", 1.0, 0.01)]
    prep = _prep(rows, rows,
                 resolution=TargetResolution({"GENEX-1": "GENEX"}, 1))
    excl = _ontarget_excluded_frame(prep)
    assert excl["feature"].to_list() == ["OTHER"]


def test_unresolved_target_excludes_nothing():
    rows = [("GENEX-1", "GENEX", 1.0, 0.01)]
    prep = _prep(rows, rows, resolution=TargetResolution({}, 1))
    assert _ontarget_excluded_frame(prep).height == 1


def test_the_two_frames_are_separately_memoized():
    """Spec 3/8: the eleven read the excluded frame, the v0.5.0 three read the
    unfiltered one; neither cache may be served to the other's callers."""
    rows = [("A", "A", 1.0, 0.01), ("A", "B", 1.0, 0.01)]
    prep = _prep(rows, rows)
    assert _direction_frame(prep).height == 2
    assert _ontarget_excluded_frame(prep).height == 1
    assert _direction_frame(prep).height == 2  # not clobbered by the second memo


def test_reference_stats_seeds_from_perturbations_not_surviving_rows():
    """A target whose ONLY significant gene is its own target gene must produce an
    explicit N_conf = 0 record, not vanish (spec 3)."""
    real = [("A", "A", 1.0, 0.01), ("A", "B", 1.0, 0.90), ("B", "B", 1.0, 0.01)]
    prep = _prep(real, real)
    stats = _reference_stats(prep).sort("target")
    assert stats["target"].to_list() == ["A", "B"]
    assert stats["n_conf"].to_list() == [0, 0]  # A's only sig gene was its own; B's too


def test_q_is_the_majority_sign_rate_and_d_has_a_floor():
    real = [("A", "B", 1.0, 0.01), ("A", "C", 1.0, 0.01),
            ("A", "D", -1.0, 0.01), ("A", "E", 1.0, 0.20)]
    prep = _prep(real, real)
    stats = _reference_stats(prep)
    assert stats["n_conf"].to_list() == [3]
    assert stats["q"].to_list() == [pytest.approx(2 / 3)]
    assert stats["d"].to_list() == [pytest.approx(1 / 3)]


def test_q_is_null_when_no_significant_gene_carries_a_direction():
    """N_conf > 0 with q undefined: significance alone counts for N_conf, but a gene
    with no direction cannot vote on q (spec 2.1)."""
    real = [("A", "B", 0.0, 0.01), ("A", "C", 0.0, 0.01)]
    prep = _prep(real, real)
    stats = _reference_stats(prep)
    assert stats["n_conf"].to_list() == [2]
    assert stats["q"].to_list() == [None]
    assert stats["d"].to_list() == [None]


def test_d_floor_engages_at_q_equals_one():
    real = [("A", "B", 1.0, 0.01), ("A", "C", 1.0, 0.01)]
    prep = _prep(real, real)
    assert _reference_stats(prep)["d"].to_list() == [pytest.approx(0.05)]


def test_components_counts_the_scored_set_and_matches():
    pred = [("A", "B", 1.0, 0.01), ("A", "C", -1.0, 0.01), ("A", "D", 1.0, 0.90)]
    real = [("A", "B", 1.0, 0.01), ("A", "C", 1.0, 0.01), ("A", "D", 1.0, 0.01)]
    prep = _prep(pred, real)
    comp = _components(prep)
    assert comp["n_pred"].to_list() == [2]   # B and C are pred-significant + adjudicable
    assert comp["k"].to_list() == [1]        # only B's sign agrees
    assert comp["n_conf"].to_list() == [3]


def test_components_gives_every_perturbation_a_row():
    pred = [("A", "B", 1.0, 0.90), ("B", "A", 1.0, 0.01)]
    real = [("A", "B", 1.0, 0.01), ("B", "A", 1.0, 0.01)]
    prep = _prep(pred, real)
    comp = _components(prep).sort("target")
    assert comp["target"].to_list() == ["A", "B"]
    assert comp["n_pred"].to_list() == [0, 1]
