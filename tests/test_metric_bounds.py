import pytest

from cell_eval2.catalog_text import metric_text
from cell_eval2.evaluation_results import build_document
from cell_eval2.run import _compute_metrics_with_facts, metric_output_names
from _edge_fixtures import SCENARIOS, config, scenario

PROFILES = ("full", "vcc2026", "cell")


@pytest.fixture(scope="module")
def docs():
    out = {}
    for profile in PROFILES:
        for name in SCENARIOS:
            pred, real, de = scenario(name)
            cfg = config(profile, de)
            df, facts = _compute_metrics_with_facts(pred, real, config=cfg)
            out[profile, name] = build_document(df, facts, metrics=metric_output_names(cfg))
    return out


def test_every_value_and_headline_statistic_lies_within_its_declared_bounds(docs):
    bad = []
    for key, doc in docs.items():
        for m, e in doc["metrics"].items():
            lo, hi = e["headline"].get("min"), e["headline"].get("max")
            stats = {s: v for s, v in e["statistics"].items() if s not in ("count", "null_count", "std")}
            for where, v in [*(("value", v) for v in e.get("values", [])), *stats.items()]:
                if v is not None and ((lo is not None and v < lo) or (hi is not None and v > hi)):
                    bad.append((key, m, where, v))
    assert bad == []


def test_every_declared_bound_is_exercised(docs):
    bounded = {m for d in docs.values() for m in d["metrics"]
               if metric_text(m).min is not None or metric_text(m).max is not None}
    seen = {m for d in docs.values() for m, e in d["metrics"].items()      # count/null_count are
            if any(v is not None for v in (*e.get("values", []),          # always numbers, so only
                                            e["statistics"][e["headline"]["statistic"]]))}  # data counts
    assert len(bounded) == 52
    assert bounded - seen == set()       # nsig_spearman's gate of 10 is met by `twelve_perts` only


@pytest.mark.parametrize("pred_counts", [
    list(range(10, 22)),                                      # perfect
    list(range(21, 9, -1)),                                   # anti
    [15] * 12,                                                # constant: charged the worst value
    [10, 10, 12, 12, 14, 14, 16, 16, 18, 18, 20, 20],         # ties
])
def test_nsig_spearman_stays_within_its_bounds(pred_counts):
    from _helpers import gate_block, gate_lfc, gate_prep, gate_sig, gate_v2
    targets = [f"T{i:02d}" for i in range(12)]
    real = [gate_block(t, gate_lfc(i), gate_sig(k)) for i, (t, k) in enumerate(zip(targets, range(10, 22)))]
    pred = [gate_block(t, gate_lfc(i), gate_sig(k)) for i, (t, k) in enumerate(zip(targets, pred_counts))]
    out = gate_v2(["de_wilcoxon_nsig_spearman"], gate_prep(real, pred))["de_wilcoxon_nsig_spearman"]
    lo, hi = metric_text("de_wilcoxon_nsig_spearman").min, metric_text("de_wilcoxon_nsig_spearman").max
    assert len(set(out.values())) == 1 and lo <= next(iter(out.values())) <= hi
