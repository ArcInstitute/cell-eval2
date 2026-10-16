"""The partition reducer gates `nsig_spearman` exactly as the whole-prediction dispatch does
(issue #391). Spec: internal:docs/internal/superpowers/specs/2026-10-08-real-side-nan-not-scored-design.md §7.
"""
import dataclasses
import math

import polars as pl
import pytest
from _helpers import gate_block, gate_lfc, gate_prep, gate_sig, gate_v2

from cell_eval2.catalog import CATALOG
from cell_eval2.partition import _inject_nsig_spearman
from cell_eval2.real_gate import RealGate

_NAME = "de_wilcoxon_nsig_spearman"


@pytest.fixture
def gated(monkeypatch):
    monkeypatch.setitem(CATALOG, _NAME, dataclasses.replace(
        CATALOG[_NAME], real_gate=RealGate("count_corr"), worst_value=-1.0))


def _panel(real_counts, pred_counts):
    real = [gate_block(f"T{i:02d}", gate_lfc(i), gate_sig(k)) for i, k in enumerate(real_counts)]
    pred = [gate_block(f"T{i:02d}", gate_lfc(i), gate_sig(k)) for i, k in enumerate(pred_counts)]
    return gate_prep(real, pred)


def _reduced(prep):
    """nsig_spearman as the partition path rebuilds it from the per-perturbation counts."""
    whole = gate_v2([_NAME, "de_wilcoxon_nsig_counts_real", "de_wilcoxon_nsig_counts_pred"], prep)
    rows = [{"perturbation": p, "metric": m, "value": v}
            for m in ("de_wilcoxon_nsig_counts_real", "de_wilcoxon_nsig_counts_pred")
            for p, v in whole[m].items()]
    full = _inject_nsig_spearman(pl.DataFrame(rows), metric_name=_NAME)
    got = {r["perturbation"]: r["value"]
           for r in full.filter(pl.col("metric") == _NAME).iter_rows(named=True)}
    return got, whole[_NAME]


@pytest.mark.parametrize("real_counts, pred_counts, expect", [
    ([0] * 12, list(range(12)), "nan"),                     # no target has a reference DEG
    ([5] * 12, list(range(12)), "nan"),                     # constant reference counts
    (list(range(1, 6)), list(range(5)), "nan"),             # fewer than 10 such targets
    (list(range(1, 13)), [0] * 12, -1.0),                   # constant prediction: charged
    (list(range(1, 13)), [3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5, 8], "finite"),
])
def test_the_reducer_equals_the_whole_prediction_value(gated, real_counts, pred_counts, expect):
    got, want = _reduced(_panel(real_counts, pred_counts))
    assert set(got) == set(want)
    for p in want:
        if expect == "nan":
            assert math.isnan(got[p]) and math.isnan(want[p]), p
        else:
            assert got[p] == pytest.approx(want[p]), p
            if expect == -1.0:
                assert got[p] == -1.0, p


def test_ungated_the_reducer_keeps_its_old_behaviour(monkeypatch):
    """With no gate the reducer still reads 1.0 when no target has a reference DEG -- the free
    win #391 removes only through the gate, so the mechanism must not depend on the catalog."""
    monkeypatch.setitem(CATALOG, _NAME, dataclasses.replace(CATALOG[_NAME], real_gate=None))
    got, _ = _reduced(_panel([0] * 12, list(range(12))))
    assert set(got.values()) == {1.0}
