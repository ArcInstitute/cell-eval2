"""The v2 dispatch applies a metric's reference-only gate (issue #391).

These tests install gates on catalog entries with `monkeypatch`, so they exercise the MECHANISM
independently of which entries ship gated; `tests/test_real_gate_catalog.py` covers the shipped
assignments. Spec: internal:docs/internal/superpowers/specs/2026-10-08-real-side-nan-not-scored-design.md.
"""
import dataclasses
import logging
import math

import numpy as np
import pytest
from _helpers import GATE_GENES, gate_block, gate_lfc, gate_prep, gate_sig, gate_v2

from cell_eval2.catalog import CATALOG, DerivedAgg, MetricSpec
from cell_eval2.config import EvalConfig
from cell_eval2.metrics.de import de_lfc_spearman, de_sig_recall
from cell_eval2.real_gate import RealGate
from cell_eval2.run import _fill_no_drop, dispatch_anndata_metrics, dispatch_de_metrics
from cell_eval2.scoring import BOUNDED


# ---- MetricSpec ------------------------------------------------------------------------------

def test_a_derived_metric_cannot_carry_a_gate():
    with pytest.raises(ValueError, match="must leave real_gate=None"):
        MetricSpec(name="x", func=None, scoring=BOUNDED, profiles=(), kind="anndata",
                   normalization=None, agg="ratio_of_sums",
                   derived=DerivedAgg(numerator="a", denominator="b"),
                   real_gate=RealGate("delta_var", min_gate_size=None))


@pytest.mark.parametrize("kind, gate", [
    ("de", RealGate("delta_var", min_gate_size=None)),
    ("anndata", RealGate("sig_set")),
])
def test_a_gate_must_match_the_metric_kind(kind, gate):
    with pytest.raises(ValueError, match="cannot carry"):
        MetricSpec(name="x", func=de_sig_recall, scoring=BOUNDED, profiles=(), kind=kind,
                   normalization=None, agg="mean", real_gate=gate)


def test_a_correlation_gate_needs_a_worst_value():
    with pytest.raises(ValueError, match="worst_value must be set"):
        MetricSpec(name="x", func=de_lfc_spearman, scoring=BOUNDED, profiles=(), kind="de",
                   normalization=None, agg="mean", real_gate=RealGate("rank_corr"))


# ---- _fill_no_drop ---------------------------------------------------------------------------

def test_outside_the_gate_is_nan_even_for_a_finite_value():
    out = _fill_no_drop({"A": 0.4, "B": 0.9}, ["A", "B"], -1.0, inside={"A"})
    assert out["A"] == 0.4 and math.isnan(out["B"])


def test_inside_the_gate_nan_omitted_and_constant_take_the_worst_value():
    out = _fill_no_drop({"A": float("nan"), "C": 0.9}, ["A", "B", "C"], -1.0,
                        inside={"A", "B", "C"}, constant={"C"})
    assert out == {"A": -1.0, "B": -1.0, "C": -1.0}


@pytest.mark.parametrize("bad", [float("inf"), float("-inf")])
def test_an_infinite_value_inside_the_gate_is_undefined_too(bad):
    """Inside a gate the contract is a FINITE value: an infinity takes the worst value, and raises
    where there is none (Copilot, PR #401)."""
    assert _fill_no_drop({"A": bad}, ["A"], -1.0, inside={"A"}) == {"A": -1.0}
    with pytest.raises(RuntimeError, match="declares no worst_value"):
        _fill_no_drop({"A": bad}, ["A"], None, inside={"A"}, metric="m")


def test_an_undefined_value_inside_a_gate_with_no_worst_value_raises():
    with pytest.raises(RuntimeError, match="declares no worst_value"):
        _fill_no_drop({"A": float("nan")}, ["A"], None, inside={"A"}, metric="m")


def test_the_ungated_fill_is_unchanged():
    out = _fill_no_drop({"P1": 0.7, "P2": float("nan"), "P4": None},
                        ["P1", "P2", "P3", "P4"], 0.0)
    assert out == {"P1": 0.7, "P2": 0.0, "P3": 0.0, "P4": 0.0}


# ---- dispatch_de_metrics ---------------------------------------------------------------------

def _gate(monkeypatch, name, gate, worst):
    for n in (name, name.replace("de_wilcoxon_", "de_deseq2_", 1)):
        if n in CATALOG:
            monkeypatch.setitem(CATALOG, n, dataclasses.replace(CATALOG[n], real_gate=gate,
                                                                worst_value=worst))


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
    return gate_prep(real, pred)


def test_a_gated_de_metric_is_nan_outside_and_charged_inside(monkeypatch):
    _gate(monkeypatch, "de_wilcoxon_lfc_spearman", RealGate("rank_corr"), -1.0)
    vals = gate_v2(["de_wilcoxon_lfc_spearman"], _panel())["de_wilcoxon_lfc_spearman"]
    assert set(vals) == {"EMPTY", "SMALL", "GOOD", "FLAT"}       # rectangular
    assert math.isnan(vals["EMPTY"]) and math.isnan(vals["SMALL"])
    assert vals["GOOD"] == pytest.approx(1.0)
    assert vals["FLAT"] == -1.0


def test_the_constancy_check_overrides_a_finite_spurious_correlation(monkeypatch):
    """polars 1.43.2 returns ~2e-16, not NaN, for a constant column at a 49-gene gate (2.0.0
    returns NaN), so the ungated fill was never reached and a flat prediction scored ~0 instead
    of -1 (spec §5.2). Pinned whichever polars is installed: the real function must give -1, and
    so must one that returns the spurious finite value outright."""
    _gate(monkeypatch, "de_wilcoxon_lfc_spearman", RealGate("rank_corr"), -1.0)
    prep = gate_prep([gate_block("A", gate_lfc(7), gate_sig(49))],
                     [gate_block("A", [0.0] * 60, gate_sig(49))])
    assert gate_v2(["de_wilcoxon_lfc_spearman"], prep)["de_wilcoxon_lfc_spearman"]["A"] == -1.0
    monkeypatch.setitem(CATALOG, "de_wilcoxon_lfc_spearman", dataclasses.replace(
        CATALOG["de_wilcoxon_lfc_spearman"], func=lambda prepared: {"A": 2e-16}))
    assert gate_v2(["de_wilcoxon_lfc_spearman"], prep)["de_wilcoxon_lfc_spearman"]["A"] == -1.0


def test_a_gate_with_no_worst_value_passes_inside_values_through(monkeypatch):
    _gate(monkeypatch, "de_wilcoxon_sig_recall", RealGate("sig_set"), None)
    vals = gate_v2(["de_wilcoxon_sig_recall"], _panel())["de_wilcoxon_sig_recall"]
    assert math.isnan(vals["EMPTY"]) and math.isnan(vals["SMALL"])
    assert vals["GOOD"] == 1.0 and vals["FLAT"] == 1.0


def test_v1_ignores_the_gate(monkeypatch):
    _gate(monkeypatch, "de_wilcoxon_sig_recall", RealGate("sig_set"), None)
    rows = dispatch_de_metrics(["de_wilcoxon_sig_recall"], _panel(),
                               EvalConfig(version="v1", device="cpu"))
    vals = {r["perturbation"]: r["value"] for r in rows}
    assert "EMPTY" not in vals                       # omitted, exactly as upstream
    assert vals["SMALL"] == 1.0                      # below the floor, still scored under v1


def test_the_gate_is_logged_by_reason(monkeypatch, caplog):
    _gate(monkeypatch, "de_wilcoxon_lfc_spearman", RealGate("rank_corr"), -1.0)
    with caplog.at_level(logging.WARNING, logger="cell_eval2.run"):
        gate_v2(["de_wilcoxon_lfc_spearman"], _panel())
    text = caplog.text
    assert "2 of 4 perturbation(s) not scored" in text
    assert "1 x fewer than 10 reference-significant genes" in text
    assert "same for every submission" in text
    assert "1 perturbation(s) charged the worst value" in text


def test_the_deseq2_relabel_carries_the_gate(monkeypatch):
    from cell_eval2.config import DEParams

    _gate(monkeypatch, "de_wilcoxon_lfc_spearman", RealGate("rank_corr"), -1.0)
    out = gate_v2(["de_wilcoxon_lfc_spearman"], _panel(), de=DEParams(backend="deseq2"))
    vals = out["de_deseq2_lfc_spearman"]
    assert math.isnan(vals["SMALL"]) and vals["FLAT"] == -1.0


# ---- dispatch_anndata_metrics ----------------------------------------------------------------

def test_delta_pearson_gate(monkeypatch):
    spec = CATALOG["delta_pearson"]
    monkeypatch.setitem(CATALOG, "delta_pearson", dataclasses.replace(
        spec, real_gate=RealGate("delta_var", min_gate_size=None), worst_value=-1.0))
    perts = np.array(["non-targeting", "A", "B"])
    ctrl = [1.0, 2.0, 3.0, 4.0]
    real = np.array([ctrl, [6.0, 7.0, 8.0, 9.0], [2.0, 1.0, 5.0, 3.0]])  # A: constant Δreal
    pred = np.array([ctrl, [3.0, 1.0, 4.0, 1.0], [6.0, 7.0, 8.0, 9.0]])  # B: constant Δpred
    rows = dispatch_anndata_metrics(
        ["delta_pearson"], {"lognorm": (perts, pred)}, {"lognorm": (perts, real)},
        np.asarray(GATE_GENES[:4], dtype=str), EvalConfig(version="v2", device="cpu"),
        comparator="lognorm")
    vals = {r["perturbation"]: r["value"] for r in rows}
    assert math.isnan(vals["A"])                  # the reference delta is constant
    assert vals["B"] == -1.0                      # the prediction delta is constant


def test_a_constant_predicted_delta_is_charged_whatever_scipy_returns(monkeypatch):
    """For a two-gene input scipy 1.14.1's `pearsonr` returns 0.0 for a constant vector (its
    two-point shortcut, sign(dx) * sign(dy)); 1.11.4 and 1.18.0 return NaN. The charge must not
    depend on which is installed, so this emulates 1.14.1 (codex, checkpoint 2)."""
    import cell_eval2.metrics.delta as delta_mod

    def scipy_1_14_1(x, y):
        x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
        return float(np.sign(x[1] - x[0]) * np.sign(y[1] - y[0]))

    monkeypatch.setattr(delta_mod, "safe_pearson", scipy_1_14_1)
    perts = np.array(["non-targeting", "A", "B"])
    real = np.array([[1.0, 2.0], [3.0, 7.0], [2.0, 1.0]])
    pred = np.array([[1.0, 2.0], [4.0, 5.0], [3.0, 1.0]])   # A: constant Δpred; B: [2, -1]
    rows = dispatch_anndata_metrics(
        ["delta_pearson"], {"lognorm": (perts, pred)}, {"lognorm": (perts, real)},
        np.asarray(GATE_GENES[:2], dtype=str), EvalConfig(version="v2", device="cpu"),
        comparator="lognorm")
    vals = {r["perturbation"]: r["value"] for r in rows}
    assert vals["A"] == -1.0                      # emulated scipy said 0.0
    assert vals["B"] == 1.0                       # an ordinary two-point correlation is kept
