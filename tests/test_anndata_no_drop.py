import math

import numpy as np
import polars as pl
from cell_eval2.catalog import CATALOG
from cell_eval2.config import EvalConfig
from cell_eval2.de import prepare_de
from cell_eval2.run import dispatch_anndata_metrics, dispatch_de_metrics

_NEW_WORST = {"delta_pearson": -1.0, "de_wilcoxon_nsig_spearman": -1.0,
              "de_wilcoxon_lfc_spearman_pos": -1.0, "de_wilcoxon_lfc_spearman_neg": -1.0,
              "de_wilcoxon_direction_precision": 0.0,
              "de_wilcoxon_direction_sensitivity": 0.0,
              "de_wilcoxon_direction_sensitivity_universe": 0.0}
# from #89 (must remain set):
_PRIOR_WORST = {
    "de_wilcoxon_sig_recall": 0.0, "de_wilcoxon_direction_match": 0.0,
    "de_wilcoxon_model_direction_match": 0.0,
    "de_wilcoxon_lfc_spearman": -1.0, "de_wilcoxon_pr_auc": 0.0, "de_wilcoxon_roc_auc": 0.0,
}


def test_new_worst_values_set():
    for name, worst in _NEW_WORST.items():
        assert CATALOG[name].worst_value == worst


def test_prior_worst_values_unchanged():
    for name, worst in _PRIOR_WORST.items():
        assert CATALOG[name].worst_value == worst


def test_worst_value_none_elsewhere():
    have = {**_NEW_WORST, **_PRIOR_WORST}
    # the de_deseq2_* family mirrors its de_wilcoxon_* siblings' worst_value (same funcs)
    have.update({n.replace("de_wilcoxon_", "de_deseq2_", 1): w
                 for n, w in list(have.items()) if n.startswith("de_wilcoxon_")})
    for name, spec in CATALOG.items():
        if name not in have:
            assert spec.worst_value is None, name


def _delta_bulks():
    """3 groups incl. control. P1's PRED delta is constant (pred_pert = pred_ctrl + 5)
    -> zero variance -> safe_pearson NaN -> degenerate. P2 is non-degenerate."""
    perts = np.array(["non-targeting", "P1", "P2"])
    pred_means = np.array([[1.0, 2.0, 3.0], [6.0, 7.0, 8.0], [2.0, 4.0, 6.0]])  # P1 Δ=[5,5,5]
    real_means = np.array([[1.0, 2.0, 3.0], [2.0, 4.0, 6.0], [3.0, 6.0, 9.0]])
    # numpy array (not list) to match the real caller (run.py: np.asarray(var.index, dtype=str))
    genes = np.asarray(["g0", "g1", "g2"], dtype=str)
    return {"lognorm": (perts, pred_means)}, {"lognorm": (perts, real_means)}, genes


def test_v2_anndata_dispatch_fills_delta_pearson_nan_to_worst():
    pred_bulks, real_bulks, genes = _delta_bulks()
    cfg = EvalConfig(metrics=["delta_pearson"], version="v2", device="cpu")
    rows = dispatch_anndata_metrics(
        ["delta_pearson"], pred_bulks, real_bulks, genes, cfg, comparator="lognorm")
    vals = {r["perturbation"]: r["value"] for r in rows if r["metric"] == "delta_pearson"}
    assert set(vals) == {"P1", "P2"}                       # control excluded, none dropped
    assert all(math.isfinite(v) for v in vals.values())    # no NaN
    assert vals["P1"] == -1.0                              # constant Δpred -> worst


def test_v1_anndata_dispatch_keeps_delta_pearson_nan():
    pred_bulks, real_bulks, genes = _delta_bulks()
    cfg = EvalConfig(metrics=["delta_pearson"], version="v1", device="cpu")
    rows = dispatch_anndata_metrics(
        ["delta_pearson"], pred_bulks, real_bulks, genes, cfg, comparator="lognorm")
    v1 = CATALOG["delta_pearson"].v1_name                  # "pearson_delta"
    vals = {r["perturbation"]: r["value"] for r in rows if r["metric"] == v1}
    assert math.isnan(vals["P1"])                          # v1 keeps the NaN


def _nsig_prep():
    """Two perts with equal real-sig (2) and equal pred-sig (2) counts -> zero-variance
    count vectors -> global Spearman NaN (broadcast to all perts)."""
    genes = ["g0", "g1", "g2", "g3"]

    def frame(t, sig):
        return pl.DataFrame({"target": [t] * 4, "feature": genes,
                             "log2_fold_change": [2.0] * 4,
                             "p_adj": [0.001 if f in sig else 0.9 for f in genes]})
    real = pl.concat([frame("P1", {"g0", "g1"}), frame("P2", {"g0", "g1"})])
    pred = pl.concat([frame("P1", {"g0", "g1"}), frame("P2", {"g2", "g3"})])
    return prepare_de(pred, real, control="non-targeting", p_adj_threshold=0.05)


def test_v2_de_dispatch_fills_nsig_spearman_nan_to_worst():
    prep = _nsig_prep()
    names = ["de_wilcoxon_nsig_spearman"]
    rows = dispatch_de_metrics(names, prep, EvalConfig(metrics=names, version="v2"))
    vals = {r["perturbation"]: r["value"] for r in rows if r["metric"] == "de_wilcoxon_nsig_spearman"}
    assert set(vals) == {"P1", "P2"}
    assert all(math.isfinite(v) for v in vals.values())
    assert all(v == -1.0 for v in vals.values())           # degenerate global corr -> worst


def test_v1_de_dispatch_keeps_nsig_spearman_nan():
    prep = _nsig_prep()
    names = ["de_wilcoxon_nsig_spearman"]
    rows = dispatch_de_metrics(names, prep, EvalConfig(metrics=names, version="v1"))
    v1 = CATALOG["de_wilcoxon_nsig_spearman"].v1_name       # "de_spearman_sig"
    vals = {r["perturbation"]: r["value"] for r in rows if r["metric"] == v1}
    assert all(math.isnan(v) for v in vals.values())
