import math

import polars as pl
from cell_eval2.catalog import CATALOG
from cell_eval2.config import EvalConfig
from cell_eval2.de import prepare_de
from cell_eval2.run import _fill_no_drop, dispatch_de_metrics

_WORST = {
    "de_wilcoxon_sig_recall": 0.0,
    "de_wilcoxon_direction_match": 0.0,
    "de_wilcoxon_model_direction_match": 0.0,
    "de_wilcoxon_lfc_spearman": -1.0,
    "de_wilcoxon_pr_auc": 0.0,
    "de_wilcoxon_roc_auc": 0.0,
}


def test_metricspec_worst_value_set_on_degenerate_metrics():
    for name, worst in _WORST.items():
        assert CATALOG[name].worst_value == worst


# Metrics outside the #89 five that also legitimately carry worst_value (added by #92).
_ALSO_WORST = {"delta_pearson", "de_wilcoxon_nsig_spearman",
               "de_wilcoxon_lfc_spearman_pos", "de_wilcoxon_lfc_spearman_neg",
               "de_wilcoxon_direction_precision", "de_wilcoxon_direction_sensitivity",
               "de_wilcoxon_direction_sensitivity_universe"}


def test_metricspec_worst_value_none_elsewhere():
    have = set(_WORST) | _ALSO_WORST
    # the de_deseq2_* family mirrors its de_wilcoxon_* siblings' worst_value (same funcs)
    have |= {n.replace("de_wilcoxon_", "de_deseq2_", 1) for n in have if n.startswith("de_wilcoxon_")}
    for name, spec in CATALOG.items():
        if name not in have:
            assert spec.worst_value is None, name


def test_fill_no_drop_missing_and_nan_map_to_worst():
    perts = ["P1", "P2", "P3", "P4"]
    result = {"P1": 0.7, "P2": float("nan"), "P4": None}   # P3 missing, P2 NaN, P4 None
    out = _fill_no_drop(result, perts, worst_value=0.0)
    assert out == {"P1": 0.7, "P2": 0.0, "P3": 0.0, "P4": 0.0}
    assert all(math.isfinite(v) for v in out.values())


def test_fill_no_drop_drops_out_of_scope_keys():
    out = _fill_no_drop({"P1": 0.5, "X": 0.9}, ["P1"], worst_value=-1.0)
    assert out == {"P1": 0.5}


def _nd_prep():
    """P1: non-degenerate for all no-drop metrics.
    P2: zero real/model-sig genes (degenerate direction/recall/lfc; single-class AUC)."""
    genes = ["g0", "g1", "g2", "g3"]
    real = pl.concat([
        pl.DataFrame({"target": ["P1"] * 4, "feature": genes,
                      "log2_fold_change": [3.0, 2.0, 1.0, 0.2],
                      "p_adj": [0.001, 0.001, 0.001, 0.9]}),
        pl.DataFrame({"target": ["P2"] * 4, "feature": genes,
                      "log2_fold_change": [0.5, 0.5, 0.5, 0.5],
                      "p_adj": [0.9, 0.9, 0.9, 0.9]}),          # zero real-sig
    ])
    pred = pl.concat([
        pl.DataFrame({"target": ["P1"] * 4, "feature": genes,
                      "log2_fold_change": [3.0, 2.0, 1.0, 0.2],
                      "p_adj": [0.001, 0.001, 0.9, 0.9]}),
        pl.DataFrame({"target": ["P2"] * 4, "feature": genes,
                      "log2_fold_change": [0.5, 0.5, 0.5, 0.5],
                      "p_adj": [0.9, 0.9, 0.9, 0.9]}),
    ])
    return prepare_de(pred, real, control="non-targeting", p_adj_threshold=0.05)


_NO_DROP = list(_WORST)


def test_v2_dispatch_fills_every_pert_finite_and_worst_for_degenerate():
    prep = _nd_prep()
    rows = dispatch_de_metrics(_NO_DROP, prep, EvalConfig(metrics=_NO_DROP, version="v2"))
    for name in _NO_DROP:
        vals = {r["perturbation"]: r["value"] for r in rows if r["metric"] == name}
        assert set(vals) == {"P1", "P2"}, name                      # no pert dropped
        assert all(math.isfinite(v) for v in vals.values()), name   # no NaN
        assert vals["P2"] == CATALOG[name].worst_value, name        # degenerate -> worst


def test_v1_dispatch_keeps_omit_and_nan():
    prep = _nd_prep()
    rows = dispatch_de_metrics(_NO_DROP, prep, EvalConfig(metrics=_NO_DROP, version="v1"))
    # recall / direction / lfc: P2 omitted entirely under v1 (inner-join drop)
    for name in ("de_wilcoxon_sig_recall", "de_wilcoxon_direction_match",
                 "de_wilcoxon_model_direction_match",
                 "de_wilcoxon_lfc_spearman"):
        v1 = CATALOG[name].v1_name
        perts = sorted(r["perturbation"] for r in rows if r["metric"] == v1)
        assert perts == ["P1"], name                                # P2 dropped in v1
    # AUC: P2 present but NaN under v1 (single-class label)
    for name in ("de_wilcoxon_pr_auc", "de_wilcoxon_roc_auc"):
        v1 = CATALOG[name].v1_name
        vals = {r["perturbation"]: r["value"] for r in rows if r["metric"] == v1}
        assert math.isnan(vals["P2"]), name
