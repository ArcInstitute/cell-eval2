"""The reference-only gates of `cell_eval2.real_gate` (issue #391), as a unit.

Spec: internal:docs/internal/superpowers/specs/2026-10-08-real-side-nan-not-scored-design.md §4-§5.
"""
import numpy as np
import pytest
from _helpers import GATE_GENES, GATE_NS, gate_block, gate_lfc, gate_prep, gate_sig

from cell_eval2.real_gate import (DEFAULT_MIN_GATE_SIZE, GateResult, RealGate,
                                  constant_delta_prediction, constant_prediction,
                                  count_corr_decision, evaluate_de_gate, evaluate_delta_gate,
                                  outside_summary, prediction_counts_constant)


def test_the_default_floor_is_ten():
    assert DEFAULT_MIN_GATE_SIZE == 10
    assert RealGate("sig_set").min_gate_size == 10


@pytest.mark.parametrize("kwargs, match", [
    ({"kind": "nope"}, "kind must be one of"),
    ({"kind": "sig_set", "min_gate_size": 0}, "int >= 1"),
    ({"kind": "sig_set", "min_gate_size": True}, "int >= 1"),
    ({"kind": "sig_set", "min_gate_size": None}, "int >= 1"),
    ({"kind": "sig_set", "min_gate_size": 2.0}, "int >= 1"),
    ({"kind": "delta_var"}, "takes no floor"),
    ({"kind": "sig_set", "lfc_direction": "pos"}, "only for a 'rank_corr'"),
    ({"kind": "rank_corr", "lfc_direction": "up"}, "lfc_direction must be one of"),
])
def test_an_ill_formed_gate_is_refused(kwargs, match):
    with pytest.raises(ValueError, match=match):
        RealGate(**kwargs)


def test_the_payload_is_by_value():
    assert RealGate("rank_corr", lfc_direction="pos").payload() == ["rank_corr", 10, "pos"]
    assert RealGate("sig_set", min_gate_size=3).payload() == ["sig_set", 3, "all"]
    assert RealGate("delta_var", min_gate_size=None).payload() == ["delta_var", None, "all"]


# ---- evaluate_de_gate ------------------------------------------------------------------------

def test_sig_set_scores_at_the_floor_and_not_below_it():
    prep = gate_prep(
        [gate_block("A", gate_lfc(1), gate_sig(9)), gate_block("B", gate_lfc(2), gate_sig(10)),
         gate_block("C", gate_lfc(3), gate_sig(0))],
        [gate_block("A", gate_lfc(1), gate_sig(9)), gate_block("B", gate_lfc(2), gate_sig(10)),
         gate_block("C", gate_lfc(3), gate_sig(0))])
    res = evaluate_de_gate(prep, RealGate("sig_set"))
    assert res.inside == {"B"}
    assert res.outside == {"A": "fewer than 10 reference-significant genes",
                           "C": "no reference-significant gene"}


def test_the_floor_is_a_parameter():
    prep = gate_prep([gate_block("A", gate_lfc(1), gate_sig(3))],
                     [gate_block("A", gate_lfc(1), gate_sig(3))])
    assert evaluate_de_gate(prep, RealGate("sig_set")).inside == frozenset()
    assert evaluate_de_gate(prep, RealGate("sig_set", min_gate_size=3)).inside == {"A"}


def test_two_class_needs_a_non_significant_gene():
    prep = gate_prep([gate_block("A", gate_lfc(1), gate_sig(60)),
                      gate_block("B", gate_lfc(2), gate_sig(12))],
                     [gate_block("A", gate_lfc(1), gate_sig(60)),
                      gate_block("B", gate_lfc(2), gate_sig(12))])
    res = evaluate_de_gate(prep, RealGate("two_class"))
    assert res.inside == {"B"}
    assert res.outside == {"A": "every labelled gene is reference-significant (single class)"}
    assert evaluate_de_gate(prep, RealGate("sig_set")).inside == {"A", "B"}


def test_two_class_counts_a_nan_p_adj_as_a_negative_like_de_auc():
    """`_de_auc` labels `p_adj < T` and drops only a NULL label, so a NaN p_adj is a negative.
    A target whose only non-significant rows carry NaN p_adj is therefore two-class."""
    padj = gate_sig(12)[:12] + [float("nan")] * 48
    prep = gate_prep([gate_block("A", gate_lfc(1), padj)],
                     [gate_block("A", gate_lfc(1), gate_sig(12))])
    assert evaluate_de_gate(prep, RealGate("two_class")).inside == {"A"}


def test_sig_lfc_does_not_count_a_null_reference_log2fc():
    """`direction_match` compares signs, and a null reference sign compares null and is ignored by
    the mean -- so a significant gene with a null log2FC is not one of its points (codex, round 1)."""
    lfc = gate_lfc(1)[:9] + [None] * 3 + gate_lfc(1)[12:]
    prep = gate_prep([gate_block("A", lfc, gate_sig(12))], [gate_block("A", lfc, gate_sig(12))])
    assert evaluate_de_gate(prep, RealGate("sig_set")).inside == {"A"}
    res = evaluate_de_gate(prep, RealGate("sig_lfc"))
    assert res.outside == {"A": "fewer than 10 reference-significant genes with a log2FC"}


def test_sig_universe_counts_a_null_p_adj_row_where_two_class_does_not():
    """Ten significant genes plus one with a null p_adj. `_de_auc` leaves that gene unlabelled, so
    the AUC has a single class; the chance-corrected tables count it in `_tested_universe`, so they
    are defined -- a matching prediction scores 1 there (codex, round 1)."""
    padj = [1e-4] * 10 + [None]
    genes = GATE_GENES[:11]
    prep = gate_prep([gate_block("A", gate_lfc(1, 11), padj, genes=genes)],
                     [gate_block("A", gate_lfc(1, 11), padj, genes=genes)])
    assert evaluate_de_gate(prep, RealGate("two_class")).outside == {
        "A": "every labelled gene is reference-significant (single class)"}
    assert evaluate_de_gate(prep, RealGate("sig_universe")).inside == {"A"}
    every = gate_prep([gate_block("B", gate_lfc(2, 11), [1e-4] * 11, genes=genes)],
                      [gate_block("B", gate_lfc(2, 11), [1e-4] * 11, genes=genes)])
    assert evaluate_de_gate(every, RealGate("sig_universe")).outside == {
        "B": "every reference row is significant"}


def test_a_null_feature_is_a_point_only_where_the_metric_can_reach_it():
    """`de_overlap_adjusted` takes its reference set from `real_rank[...].drop_nulls()`, and the
    inner-join metrics never match a null feature, so a null-feature row counts for neither --
    while `de_sig_agreement` and `sig_recall` count it (codex, checkpoint 1 round 2)."""
    import polars as pl

    from cell_eval2.de import prepare_de

    genes = GATE_GENES[:10] + [None]
    real = pl.DataFrame({"target": ["A"] * 11, "feature": genes,
                         "log2_fold_change": gate_lfc(1, 11), "p_adj": [1e-4] * 11},
                        schema={"target": pl.String, "feature": pl.String,
                                "log2_fold_change": pl.Float64, "p_adj": pl.Float64})
    prep = prepare_de(real, real, control="non-targeting")
    # eleven significant rows, ten of them named: a ranked set of 10 below a universe of 11
    assert evaluate_de_gate(prep, RealGate("ranked_universe")).inside == {"A"}
    assert evaluate_de_gate(prep, RealGate("sig_universe")).outside == {
        "A": "every reference row is significant"}
    keep = pl.col("feature").is_null() | (pl.col("feature") != "g09")   # 9 named + the null
    nine = prepare_de(real.filter(keep), real.filter(keep), control="non-targeting")
    for kind in ("sig_lfc", "adjudicable", "lfc_present"):
        assert evaluate_de_gate(nine, RealGate(kind)).inside == frozenset(), kind
    assert evaluate_de_gate(nine, RealGate("sig_set")).inside == {"A"}   # recall counts it


def test_rank_corr_drops_a_null_reference_log2fc():
    """polars' correlation drops a pair with a null, so ten equal log2FCs plus a null are a
    CONSTANT reference, not two distinct values (codex, round 1)."""
    lfc = [2.0] * 10 + [None] + [0.1] * 49
    prep = gate_prep([gate_block("A", lfc, gate_sig(11))], [gate_block("A", lfc, gate_sig(11))])
    assert evaluate_de_gate(prep, RealGate("rank_corr")).outside == {
        "A": "constant reference log2FC over the gate"}


@pytest.mark.parametrize("kind, lfc, reason", [
    ("adjudicable", [0.0] * 60, "fewer than 10 reference genes with a log2FC direction"),
    ("lfc_present", [None] * 60, "fewer than 10 reference genes with a log2FC"),
])
def test_a_reference_with_no_usable_log2fc_is_outside(kind, lfc, reason):
    """`direction_precision` can adjudicate only a reference gene with a direction, and
    `model_direction_match` ignores a null comparison -- so a reference without such genes is
    not the prediction's fault (codex, round 1)."""
    prep = gate_prep([gate_block("A", lfc, gate_sig(12))],
                     [gate_block("A", gate_lfc(1), gate_sig(12))])
    assert evaluate_de_gate(prep, RealGate(kind)).outside == {"A": reason}
    ok = gate_prep([gate_block("B", gate_lfc(2), gate_sig(12))],
                   [gate_block("B", gate_lfc(2), gate_sig(12))])
    assert evaluate_de_gate(ok, RealGate(kind)).inside == {"B"}


def test_rank_corr_counts_the_direction_subset_and_needs_two_distinct_values():
    up12 = [1.0 + i for i in range(12)] + [-1.0] * 3 + [0.1] * 45   # 12 up, 3 down significant
    flat = [2.0] * 15 + [0.1] * 45                                    # 15 significant, all equal
    prep = gate_prep([gate_block("A", up12, gate_sig(15)), gate_block("B", flat, gate_sig(15))],
                     [gate_block("A", up12, gate_sig(15)), gate_block("B", flat, gate_sig(15))])
    both = evaluate_de_gate(prep, RealGate("rank_corr"))
    assert both.inside == {"A"}
    assert both.outside == {"B": "constant reference log2FC over the gate"}
    assert evaluate_de_gate(prep, RealGate("rank_corr", lfc_direction="pos")).inside == {"A"}
    neg = evaluate_de_gate(prep, RealGate("rank_corr", lfc_direction="neg"))
    assert neg.inside == frozenset()
    assert neg.outside["A"] == "fewer than 10 neg-direction reference-significant genes"


def test_count_corr_is_global():
    gate = RealGate("count_corr")
    assert count_corr_decision([11] * 3, gate=gate).startswith("fewer than 10 targets")
    assert count_corr_decision([11] * 12, gate=gate) == (
        "every such target has the same reference-significant gene count")
    assert count_corr_decision(list(range(1, 13)), gate=gate) is None
    with pytest.raises(ValueError, match="needs a 'count_corr' gate"):
        count_corr_decision([1, 2], gate=RealGate("sig_set"))


def test_count_corr_admits_every_perturbation_or_none():
    blocks = [gate_block(f"T{i:02d}", gate_lfc(i), gate_sig(1 + i)) for i in range(12)]
    prep = gate_prep(blocks, blocks)
    assert evaluate_de_gate(prep, RealGate("count_corr")).inside == set(prep.perturbations)
    few = gate_prep(blocks[:5], blocks[:5])
    res = evaluate_de_gate(few, RealGate("count_corr"))
    assert res.inside == frozenset() and set(res.outside) == set(few.perturbations)


def test_the_gate_never_reads_the_prediction():
    real = [gate_block(t, gate_lfc(i), gate_sig(5 + 3 * i)) for i, t in enumerate("ABCDEF")]
    honest = [gate_block(t, gate_lfc(i), gate_sig(5 + 3 * i)) for i, t in enumerate("ABCDEF")]
    silent = [gate_block(t, [0.0] * 60, gate_sig(0)) for t in "ABCDEF"]
    for gate in (RealGate("sig_set"), RealGate("two_class"), RealGate("rank_corr"),
                 RealGate("rank_corr", lfc_direction="pos"), RealGate("count_corr")):
        assert evaluate_de_gate(gate_prep(real, honest), gate) == \
            evaluate_de_gate(gate_prep(real, silent), gate), gate


def test_the_gate_does_not_even_touch_the_prediction():
    """Output invariance is not enough: every DE kind must decide without reading `pred_df` at all
    (Copilot, PR #401). A prediction table that raises on any access proves it."""
    import dataclasses

    class _Untouchable:
        def __getattr__(self, name):
            raise AssertionError(f"the gate read the prediction (.{name})")

    blocks = [gate_block(f"T{i:02d}", gate_lfc(i), gate_sig(1 + i)) for i in range(12)]
    prep = dataclasses.replace(gate_prep(blocks, blocks), pred_df=_Untouchable())
    for kind in ("sig_set", "sig_lfc", "two_class", "sig_universe", "ranked_universe",
                 "rank_corr", "count_corr", "adjudicable", "lfc_present"):
        evaluate_de_gate(prep, RealGate(kind))


def test_a_delta_var_gate_is_refused_by_the_de_evaluator():
    prep = gate_prep([gate_block("A", gate_lfc(1), gate_sig(12))],
                     [gate_block("A", gate_lfc(1), gate_sig(12))])
    with pytest.raises(ValueError, match="evaluate_delta_gate"):
        evaluate_de_gate(prep, RealGate("delta_var", min_gate_size=None))


def test_the_result_is_memoized_per_gate():
    prep = gate_prep([gate_block("A", gate_lfc(1), gate_sig(12))],
                     [gate_block("A", gate_lfc(1), gate_sig(12))])
    first = evaluate_de_gate(prep, RealGate("sig_set"))
    assert evaluate_de_gate(prep, RealGate("sig_set")) is first
    assert evaluate_de_gate(prep, RealGate("sig_set", min_gate_size=20)) is not first


# ---- constant_prediction --------------------------------------------------------------------

def test_a_flat_or_missing_prediction_is_constant_over_the_gate():
    real = [gate_block("FLAT", gate_lfc(1), gate_sig(12)),
            gate_block("MISS", gate_lfc(2), gate_sig(12)),
            gate_block("OK", gate_lfc(3), gate_sig(12))]
    pred = [gate_block("FLAT", [0.7] * 60, gate_sig(12)),
            gate_block("MISS", gate_lfc(2)[12:], [GATE_NS] * 48, genes=GATE_GENES[12:]),
            gate_block("OK", gate_lfc(3), gate_sig(12))]
    prep = gate_prep(real, pred)
    gate = RealGate("rank_corr")
    inside = evaluate_de_gate(prep, gate).inside
    assert inside == {"FLAT", "MISS", "OK"}
    assert constant_prediction(prep, gate, inside) == {"FLAT", "MISS"}
    # only perturbations the reference admitted are ever checked
    assert constant_prediction(prep, gate, {"OK"}) == frozenset()
    # a kind with no constancy check
    assert constant_prediction(prep, RealGate("sig_set"), inside) == frozenset()


def test_constancy_ignores_rows_with_a_null_reference_log2fc():
    """A prediction varying only where the REFERENCE log2FC is null is constant where the
    correlation is computed, since polars drops those pairs (codex, round 1)."""
    real_lfc = gate_lfc(5)[:11] + [None] * 2 + gate_lfc(5)[13:]
    pred_lfc = [0.5] * 11 + [3.0, -3.0] + [0.5] * 47
    prep = gate_prep([gate_block("A", real_lfc, gate_sig(13))],
                     [gate_block("A", pred_lfc, gate_sig(13))])
    gate = RealGate("rank_corr")
    assert constant_prediction(prep, gate, evaluate_de_gate(prep, gate).inside) == {"A"}


def test_constant_prediction_counts():
    assert prediction_counts_constant([0, 0, 0])
    assert prediction_counts_constant([3.0, 3])
    assert not prediction_counts_constant([0, 1])


# ---- evaluate_delta_gate ---------------------------------------------------------------------

def _bulk(rows):
    return (np.array(["non-targeting", "A"]), np.array(rows, dtype=float))


@pytest.mark.parametrize("rows, reason", [
    ([[1.0, 2.0, 3.0], [2.0, 3.0, 4.0]], "constant reference delta"),
    ([[1.0, 2.0], [np.nan, 3.0]], "non-finite reference delta"),
    ([[1.0], [5.0]], "fewer than 2 genes"),
])
def test_a_reference_delta_that_cannot_carry_a_correlation_is_outside(rows, reason):
    res = evaluate_delta_gate(_bulk(rows), control="non-targeting", perts=["A"])
    assert res.outside == {"A": reason}


def test_a_usable_reference_delta_is_inside():
    res = evaluate_delta_gate(_bulk([[1.0, 2.0, 3.0], [2.0, 2.0, 5.0]]),
                              control="non-targeting", perts=["A", "Z"])
    assert res.inside == {"A"}
    assert res.outside == {"Z": "absent from the reference"}


def test_the_delta_gate_needs_the_control():
    with pytest.raises(ValueError, match="control 'ctrl' not found"):
        evaluate_delta_gate(_bulk([[1.0, 2.0], [2.0, 3.0]]), control="ctrl", perts=["A"])


# ---- constant_delta_prediction ---------------------------------------------------------------

_DPERTS = np.array(["non-targeting", "A", "B"])
_DPRED = np.array([[1.0, 1.0], [3.0, 3.0], [1.0, 2.0]])
_DREAL = np.array([[0.0, 1.0], [5.0, 2.0], [1.0, 3.0]])


@pytest.mark.parametrize("control_source, constant", [
    ("pred", {"A"}),     # A - predicted control = [2, 2]; B = [0, 1]
    ("real", {"B"}),     # A - reference control = [3, 2]; B = [1, 1]
])
def test_a_constant_predicted_delta_is_built_against_the_metrics_own_control(control_source,
                                                                             constant):
    got = constant_delta_prediction((_DPERTS, _DPRED), real_bulk=(_DPERTS, _DREAL),
                                    control="non-targeting", control_source=control_source,
                                    inside={"A", "B"})
    assert got == constant


def test_the_delta_constancy_check_only_narrows_the_admitted_set():
    got = constant_delta_prediction((_DPERTS, _DPRED), real_bulk=(_DPERTS, _DREAL),
                                    control="non-targeting", control_source="pred",
                                    inside={"B", "Z"})       # A is constant, but not admitted
    assert got == frozenset()


def test_a_constant_de_check_refuses_a_delta_gate():
    with pytest.raises(ValueError, match="use constant_delta_prediction"):
        constant_prediction(None, RealGate("delta_var", min_gate_size=None), {"A"})


def test_outside_summary_orders_by_frequency():
    res = GateResult(frozenset(), {"A": "x", "B": "y", "C": "y"})
    assert outside_summary(res) == "2 x y; 1 x x"
