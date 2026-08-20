import math

import polars as pl
import pytest
from cell_eval2.catalog import CATALOG, PROFILES, resolve_metrics
from cell_eval2.config import EvalConfig
from cell_eval2.de import prepare_de
from cell_eval2.metrics.de import (
    _informedness,
    _markedness,
    _mcc,
    de_nsig_counts,
    de_overlap_adjusted,
    de_sig_agreement,
)
from cell_eval2.run import dispatch_de_metrics

_NEW = [
    "de_wilcoxon_overlap_adjusted",
    "de_wilcoxon_precision_adjusted",
    "de_wilcoxon_sig_recall_adjusted",
    "de_wilcoxon_sig_mcc",
]


def _toy(real, pred, threshold=0.05):
    return prepare_de(pred, real, control="non-targeting", p_adj_threshold=threshold)


def _framed(target, all_features, sig_features):
    """One-target DE frame: every gene in all_features gets a row; sig genes get p_adj<T."""
    return pl.DataFrame({
        "target": [target] * len(all_features),
        "feature": list(all_features),
        "log2_fold_change": [2.0] * len(all_features),
        "p_adj": [0.001 if f in sig_features else 0.9 for f in all_features],
    })


def test_closed_form_asymmetric():
    # G=10, real-sig a=4, pred-sig b=2, TP=2 -> FP=0, FN=2, TN=6
    tp, a, b, g = 2, 4, 2, 10
    assert math.isclose(_informedness(tp, a, b, g), 0.5, rel_tol=1e-12)      # 2/4 - 0/6
    assert math.isclose(_markedness(tp, a, b, g), 0.75, rel_tol=1e-12)       # 2/2 - 2/8
    assert math.isclose(_mcc(tp, a, b, g), 12 / math.sqrt(384), rel_tol=1e-12)


def test_mcc_is_geometric_mean_of_the_factors():
    tp, a, b, g = 2, 4, 2, 10
    inf, mk = _informedness(tp, a, b, g), _markedness(tp, a, b, g)
    assert math.isclose(_mcc(tp, a, b, g), math.copysign(math.sqrt(abs(inf * mk)), inf), rel_tol=1e-12)


def test_chance_prediction_is_zero():
    # TP == a*b/G exactly -> independence -> 0 for all three
    tp, a, b, g = 1, 10, 10, 100  # a*b/G = 1
    assert math.isclose(_informedness(tp, a, b, g), 0.0, abs_tol=1e-12)
    assert math.isclose(_markedness(tp, a, b, g), 0.0, abs_tol=1e-12)
    assert math.isclose(_mcc(tp, a, b, g), 0.0, abs_tol=1e-12)


def test_perfect_prediction_is_one():
    tp, a, b, g = 5, 5, 5, 100
    for fn in (_informedness, _markedness, _mcc):
        assert math.isclose(fn(tp, a, b, g), 1.0, rel_tol=1e-12)


def test_worse_than_chance_is_negative():
    tp, a, b, g = 0, 10, 10, 100  # below the chance level a*b/G=1
    assert _informedness(tp, a, b, g) < 0
    assert _markedness(tp, a, b, g) < 0
    assert _mcc(tp, a, b, g) < 0


# Degenerate conditions differ per measure (informedness: a in {0,G};
# markedness: b in {0,G}; mcc: any of a,b,G-a,G-b == 0) -> parameterize separately.
@pytest.mark.parametrize("tp,a,b,g", [
    (0, 0, 3, 10),   # a == 0
    (3, 3, 3, 3),    # a == G (and b == G)
    (0, 3, 0, 10),   # b == 0
    (0, 0, 0, 0),    # empty universe
])
def test_mcc_degenerate_maps_to_minus_one(tp, a, b, g):
    assert _mcc(tp, a, b, g) == -1.0


@pytest.mark.parametrize("tp,a,b,g", [
    (0, 0, 3, 10),   # a == 0
    (3, 3, 3, 3),    # a == G
    (0, 0, 0, 0),    # empty universe
])
def test_informedness_degenerate_maps_to_minus_one(tp, a, b, g):
    assert _informedness(tp, a, b, g) == -1.0


@pytest.mark.parametrize("tp,a,b,g", [
    (3, 3, 3, 3),    # b == G
    (0, 3, 0, 10),   # b == 0
    (0, 0, 0, 0),    # empty universe
])
def test_markedness_degenerate_maps_to_minus_one(tp, a, b, g):
    assert _markedness(tp, a, b, g) == -1.0


def test_flood_of_significance_scores_near_chance_not_one():
    # G=100, 5 real-sig, pred floods 90 sig (captures all 5): raw recall would be 1.0
    tp, a, b, g = 5, 5, 90, 100
    assert _informedness(tp, a, b, g) < 0.2
    assert _markedness(tp, a, b, g) < 0.2
    assert _mcc(tp, a, b, g) < 0.2


def test_sig_agreement_known_small_table():
    # G=10 tested; real-sig a={g0..g3}=4, pred-sig b={g0,g1}=2, TP=2 -> matches closed form
    genes = [f"g{i}" for i in range(10)]
    real = _framed("P", genes, {"g0", "g1", "g2", "g3"})
    pred = _framed("P", genes, {"g0", "g1"})
    prep = _toy(real, pred)
    assert math.isclose(de_sig_agreement(prep, measure="informedness")["P"], 0.5, rel_tol=1e-9)
    assert math.isclose(de_sig_agreement(prep, measure="markedness")["P"], 0.75, rel_tol=1e-9)
    assert math.isclose(de_sig_agreement(prep, measure="mcc")["P"], 12 / math.sqrt(384), rel_tol=1e-9)


def test_sig_agreement_flood_regression_guard():
    # pred floods 90/100 significant, captures all 5 real -> raw recall 1.0; corrected ~ chance
    genes = [f"g{i}" for i in range(100)]
    real = _framed("P", genes, {f"g{i}" for i in range(5)})
    pred = _framed("P", genes, {f"g{i}" for i in range(90)})  # includes g0..g4
    prep = _toy(real, pred)
    for measure in ("informedness", "markedness", "mcc"):
        assert de_sig_agreement(prep, measure=measure)["P"] < 0.2


def test_sig_agreement_every_pert_present_and_finite():
    # P2 has zero real-sig genes -> must still get a finite value (-1), not be dropped
    genes = ["g0", "g1", "g2"]
    real = pl.concat([_framed("P1", genes, {"g0", "g1"}), _framed("P2", genes, set())])
    pred = pl.concat([_framed("P1", genes, {"g0"}), _framed("P2", genes, {"g0"})])
    prep = _toy(real, pred)
    out = de_sig_agreement(prep, measure="mcc")
    assert set(out) == {"P1", "P2"}
    assert out["P2"] == -1.0
    assert all(math.isfinite(v) and -1.0 <= v <= 1.0 for v in out.values())


def test_sig_agreement_bad_measure_raises():
    genes = ["g0", "g1"]
    prep = _toy(_framed("P", genes, {"g0"}), _framed("P", genes, {"g0"}))
    with pytest.raises(ValueError, match="measure"):
        de_sig_agreement(prep, measure="bogus")


def test_overlap_adjusted_perfect_top_list():
    # real top-2 == pred top-2 (by abs LFC) over a 10-gene universe -> MCC 1.0
    genes = [f"g{i}" for i in range(10)]
    real = _framed("P", genes, {"g0", "g1"}).with_columns(
        pl.when(pl.col("feature").is_in(["g0", "g1"])).then(5.0).otherwise(0.0).alias("log2_fold_change"))
    pred = real
    prep = _toy(real, pred)
    assert math.isclose(de_overlap_adjusted(prep)["P"], 1.0, rel_tol=1e-9)


def test_overlap_adjusted_flood_is_not_gamed():
    # pred floods 55 genes sig but ranks the NON-real ones highest by |LFC|, so its
    # top-5 (= m_r) is disjoint from the real top-5 -> ~chance, not gamed to 1.0.
    genes = [f"g{i}" for i in range(100)]
    real_sig = {f"g{i}" for i in range(5)}
    real = _framed("P", genes, real_sig).with_columns(
        pl.when(pl.col("feature").is_in(list(real_sig))).then(9.0).otherwise(0.0).alias("log2_fold_change"))
    pred_sig = real_sig | {f"g{i}" for i in range(50, 100)}
    pred = _framed("P", genes, pred_sig).with_columns(
        pl.when(pl.col("feature").is_in([f"g{i}" for i in range(50, 100)])).then(9.0)   # rank highest
        .when(pl.col("feature").is_in(list(real_sig))).then(1.0)
        .otherwise(0.0).alias("log2_fold_change"))
    prep = _toy(real, pred)
    assert de_overlap_adjusted(prep)["P"] < 0.5


def test_overlap_adjusted_every_pert_present_and_bounded():
    genes = ["g0", "g1", "g2"]
    real = pl.concat([_framed("P1", genes, {"g0", "g1"}), _framed("P2", genes, set())])
    pred = pl.concat([_framed("P1", genes, {"g0"}), _framed("P2", genes, {"g0"})])
    prep = _toy(real, pred)
    out = de_overlap_adjusted(prep)
    assert set(out) == {"P1", "P2"}
    assert out["P2"] == -1.0                       # zero real-sig -> a=0 -> -1
    assert all(math.isfinite(v) and -1.0 <= v <= 1.0 for v in out.values())


def test_new_metrics_in_full_and_de_profiles():
    for name in _NEW:
        assert name in CATALOG
        assert CATALOG[name].scoring.scored
        assert CATALOG[name].scoring.direction == "higher"
        assert CATALOG[name].kind == "de"
        assert CATALOG[name].profiles == ("full", "de")
        assert name in PROFILES["full"] and name in PROFILES["de"]


def test_new_metrics_not_in_scored_profiles():
    for prof in ("vcc", "minimal"):
        avail, _ = resolve_metrics(prof)
        assert not any(n in avail for n in _NEW)


def test_dispatch_emits_one_finite_row_per_pert():
    genes = ["g0", "g1", "g2"]
    real = pl.concat([_framed("P1", genes, {"g0", "g1"}), _framed("P2", genes, set())])
    pred = pl.concat([_framed("P1", genes, {"g0"}), _framed("P2", genes, {"g0"})])
    prep = _toy(real, pred)
    rows = dispatch_de_metrics(_NEW, prep, EvalConfig(metrics=_NEW))
    for name in _NEW:
        vals = [r["value"] for r in rows if r["metric"] == name]
        assert len(vals) == 2                                   # one row per pert, none dropped
        assert all(math.isfinite(v) and -1.0 <= v <= 1.0 for v in vals)


def _floor_side_frame():
    # One target: big-effect + small-effect significant genes, plus a background of
    # tested-but-nonsignificant genes so the universe G is well defined.
    return pl.DataFrame({
        "target": ["A", "A", "A", "A"],
        "feature": ["g_big", "g_small", "bg1", "bg2"],
        "log2_fold_change": [2.0, 0.3, 0.05, 0.02],
        "p_value": [0.001, 0.001, 0.9, 0.9],
        "p_adj": [0.01, 0.01, 0.9, 0.9],
    })


def test_standalone_metric_honors_floor():
    # Without a floor both g_big and g_small are significant -> nsig real count 2.
    base = de_nsig_counts(side="real", de_pred=_floor_side_frame(),
                          de_real=_floor_side_frame(), control="ctrl")
    assert base["A"] == 2.0
    # With floor 1.0, g_small (|lfc|=0.3) drops -> nsig real count 1.
    floored = de_nsig_counts(side="real", de_pred=_floor_side_frame(),
                             de_real=_floor_side_frame(), control="ctrl", min_abs_log2fc=1.0)
    assert floored["A"] == 1.0


def test_standalone_floor_preserves_universe_mcc_finite():
    # Universe G (union of tested genes) is unchanged by the floor, so MCC stays finite.
    out = de_sig_agreement(measure="mcc", de_pred=_floor_side_frame(),
                           de_real=_floor_side_frame(), control="ctrl", min_abs_log2fc=1.0)
    assert out["A"] == out["A"]          # not NaN
    assert -1.0 <= out["A"] <= 1.0
