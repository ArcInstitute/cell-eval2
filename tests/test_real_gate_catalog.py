"""The shipped gates (issue #391, spec §3): which entries are gated, and what they now emit.

Spec: internal:docs/internal/superpowers/specs/2026-10-08-real-side-nan-not-scored-design.md.
"""
import math

import pytest
from _helpers import GATE_GENES, GATE_NS, gate_block, gate_lfc, gate_prep, gate_sig, gate_v2

from cell_eval2.catalog import CATALOG

_GATED = {
    # name: (gate payload, worst_value)
    # group A -- only the reference can make it undefined
    "de_wilcoxon_sig_recall": (["sig_set", 10, "all"], None),
    "de_wilcoxon_pr_auc": (["two_class", 10, "all"], None),
    "de_wilcoxon_roc_auc": (["two_class", 10, "all"], None),
    "de_wilcoxon_direction_sensitivity": (["sig_set", 10, "all"], None),
    "de_wilcoxon_direction_sensitivity_universe": (["sig_set", 10, "all"], None),
    # group B -- either side can
    "de_wilcoxon_lfc_spearman": (["rank_corr", 10, "all"], -1.0),
    "de_wilcoxon_lfc_spearman_pos": (["rank_corr", 10, "pos"], -1.0),
    "de_wilcoxon_lfc_spearman_neg": (["rank_corr", 10, "neg"], -1.0),
    "de_wilcoxon_direction_match": (["sig_lfc", 10, "all"], 0.0),
    "de_wilcoxon_nsig_spearman": (["count_corr", 10, "all"], -1.0),
    "delta_pearson": (["delta_var", None, "all"], -1.0),
    # group C -- #14, in-function -1
    "de_wilcoxon_sig_recall_adjusted": (["sig_universe", 10, "all"], None),
    "de_wilcoxon_sig_mcc": (["sig_universe", 10, "all"], None),
    "de_wilcoxon_overlap_adjusted": (["ranked_universe", 10, "all"], None),
    # prediction-conditioned -- a prediction calling nothing is still charged 0.0, but a
    # reference with nothing to compare against is gated (codex, checkpoint 1)
    "de_wilcoxon_direction_precision": (["adjudicable", 10, "all"], 0.0),
    "de_wilcoxon_model_direction_match": (["lfc_present", 10, "all"], 0.0),
}


def test_the_gated_entries():
    for name, (payload, worst) in _GATED.items():
        spec = CATALOG[name]
        assert spec.real_gate is not None and spec.real_gate.payload() == payload, name
        assert spec.worst_value == worst, name
        if name.startswith("de_wilcoxon_"):
            twin = CATALOG[name.replace("de_wilcoxon_", "de_deseq2_", 1)]
            assert twin.real_gate == spec.real_gate and twin.worst_value == worst, name


def test_no_other_entry_carries_a_gate_or_a_worst_value():
    """Every worst value now sits behind a gate: a prediction-side charge is never applied to a
    perturbation the reference could not score."""
    have = set(_GATED)
    have |= {n.replace("de_wilcoxon_", "de_deseq2_", 1) for n in have
             if n.startswith("de_wilcoxon_")}
    for name, spec in CATALOG.items():
        if name not in have:
            assert spec.real_gate is None and spec.worst_value is None, name


def test_no_competition_member_is_gated():
    for name, spec in CATALOG.items():
        if "vcc" in spec.profiles or "vcc2026" in spec.profiles:
            assert spec.real_gate is None and spec.worst_value is None, name


# ---- end to end through the shipped catalog --------------------------------------------------

def _panel():
    real, pred = [], []
    real.append(gate_block("EMPTY", gate_lfc(1), gate_sig(0)))
    pred.append(gate_block("EMPTY", gate_lfc(1), gate_sig(12)))
    real.append(gate_block("SMALL", gate_lfc(2), gate_sig(5)))
    pred.append(gate_block("SMALL", gate_lfc(2), gate_sig(5)))
    real.append(gate_block("GOOD", gate_lfc(3), gate_sig(12)))
    pred.append(gate_block("GOOD", gate_lfc(3), gate_sig(12)))
    real.append(gate_block("FLAT", gate_lfc(4), gate_sig(12)))
    pred.append(gate_block("FLAT", [0.7] * 60, gate_sig(12)))
    real.append(gate_block("SILENT", gate_lfc(5), gate_sig(12)))
    pred.append(gate_block("SILENT", gate_lfc(5), [GATE_NS] * 60))
    return gate_prep(real, pred)


# The direction subsets of a 12-gene gate hold fewer than 10 genes each, so `_pos`/`_neg` are
# outside on every target here; `tests/test_real_gate.py` covers their subset arithmetic. The two
# prediction-conditioned metrics gate on every reference gene (60 per target here), so EMPTY and
# SMALL are inside for them, and they are left to the tests below.
_ALL_DIRECTION = sorted(n for n in _GATED if n.startswith("de_wilcoxon_")
                        and n not in ("de_wilcoxon_nsig_spearman",
                                      "de_wilcoxon_lfc_spearman_pos",
                                      "de_wilcoxon_lfc_spearman_neg",
                                      "de_wilcoxon_direction_precision",
                                      "de_wilcoxon_model_direction_match"))


def test_rectangular_and_nan_only_where_the_reference_cannot_score():
    prep = _panel()
    out = gate_v2(_ALL_DIRECTION, prep)
    for name in _ALL_DIRECTION:
        vals = out[name]
        assert set(vals) == set(prep.perturbations), name
        assert math.isnan(vals["EMPTY"]) and math.isnan(vals["SMALL"]), name
        assert math.isfinite(vals["GOOD"]), name


def test_a_flat_or_silent_prediction_is_charged_inside_the_gate():
    out = gate_v2(["de_wilcoxon_lfc_spearman", "de_wilcoxon_sig_mcc",
                   "de_wilcoxon_overlap_adjusted", "de_wilcoxon_sig_recall"], _panel())
    assert out["de_wilcoxon_lfc_spearman"]["FLAT"] == -1.0
    # calling nothing significant is NOT a NaN cause for the LFC correlation (#391, measured)
    assert out["de_wilcoxon_lfc_spearman"]["SILENT"] == pytest.approx(1.0)
    assert out["de_wilcoxon_sig_mcc"]["SILENT"] == -1.0            # prediction-side 0/0
    assert out["de_wilcoxon_overlap_adjusted"]["SILENT"] == -1.0
    assert out["de_wilcoxon_sig_recall"]["SILENT"] == 0.0


def test_the_14_metrics_no_longer_charge_an_empty_reference():
    out = gate_v2(["de_wilcoxon_sig_recall_adjusted", "de_wilcoxon_sig_mcc",
                   "de_wilcoxon_overlap_adjusted"], _panel())
    for name, vals in out.items():
        assert math.isnan(vals["EMPTY"]), name                    # was -1.0 in-function


def test_direction_match_charges_a_prediction_covering_none_of_the_gate():
    prep = gate_prep([gate_block("A", gate_lfc(1), gate_sig(12))],
                     [gate_block("A", gate_lfc(1)[12:], [GATE_NS] * 48, genes=GATE_GENES[12:])])
    assert gate_v2(["de_wilcoxon_direction_match"], prep)["de_wilcoxon_direction_match"]["A"] == 0.0


def _nsig_panel(real_counts, pred_counts):
    real = [gate_block(f"T{i:02d}", gate_lfc(i), gate_sig(k)) for i, k in enumerate(real_counts)]
    pred = [gate_block(f"T{i:02d}", gate_lfc(i), gate_sig(k)) for i, k in enumerate(pred_counts)]
    return gate_prep(real, pred)


@pytest.mark.parametrize("real_counts, pred_counts, expect", [
    ([0] * 12, list(range(12)), "nan"),                     # was a free 1.0 for everyone
    ([5] * 12, list(range(12)), "nan"),                     # constant reference counts
    (list(range(1, 6)), list(range(5)), "nan"),             # fewer than 10 targets
    (list(range(1, 13)), [0] * 12, -1.0),                   # constant prediction: charged
    (list(range(1, 13)), list(range(1, 13)), 1.0),          # perfect
])
def test_nsig_spearman_is_gated_globally(real_counts, pred_counts, expect):
    vals = gate_v2(["de_wilcoxon_nsig_spearman"], _nsig_panel(real_counts, pred_counts))[
        "de_wilcoxon_nsig_spearman"]
    assert len({str(v) for v in vals.values()}) == 1         # still one broadcast value
    v = next(iter(vals.values()))
    assert math.isnan(v) if expect == "nan" else v == pytest.approx(expect)


def test_a_panel_no_target_of_which_clears_the_floor_scores_without_crashing():
    """Review Focus 1: a weak panel where a gated metric is unscoreable everywhere. Its column is
    NaN on the submission AND on any baseline built against the same reference, and scoring
    excludes it with a warning (spec §10.4) instead of refusing the run."""
    import polars as pl

    from cell_eval2.run import aggregate_metrics_wide
    from cell_eval2.score import score_metrics

    prep = gate_prep([gate_block(t, gate_lfc(i), gate_sig(4)) for i, t in enumerate("ABC")],
                     [gate_block(t, gate_lfc(i), gate_sig(4)) for i, t in enumerate("ABC")])
    names = ["de_wilcoxon_lfc_spearman", "de_wilcoxon_precision"]

    def frame(p):
        rows = [{"perturbation": q, "metric": m, "value": v}
                for m, vals in gate_v2(names, p).items() for q, v in vals.items()]
        return aggregate_metrics_wide(pl.DataFrame(rows), metrics=names)

    out = score_metrics(frame(prep), frame(prep).with_columns(
        pl.when(pl.col("statistic") == "mean").then(0.1).otherwise(pl.col("de_wilcoxon_precision"))
        .alias("de_wilcoxon_precision")))
    assert "de_wilcoxon_lfc_spearman" not in set(out["metric"])
    assert "de_wilcoxon_precision" in set(out["metric"])


def test_the_prediction_conditioned_pair_is_charged_for_calling_nothing_not_for_the_reference():
    """`direction_precision` and `model_direction_match` read the PREDICTION's calls: a silent
    prediction is charged 0.0 wherever the reference has genes to compare against, and only a
    reference with no usable log2FC is outside their gates."""
    out = gate_v2(["de_wilcoxon_direction_precision", "de_wilcoxon_model_direction_match"],
                  _panel())
    for name, vals in out.items():
        assert vals["SILENT"] == 0.0, name               # no call: charged
        assert vals["EMPTY"] == pytest.approx(1.0), name  # the prediction's calls agree
    zero = gate_prep([gate_block("Z", [0.0] * 60, gate_sig(12))],
                     [gate_block("Z", gate_lfc(9), gate_sig(12))])
    vals = gate_v2(["de_wilcoxon_direction_precision"], zero)["de_wilcoxon_direction_precision"]
    assert math.isnan(vals["Z"])                         # nothing adjudicable: not scored
