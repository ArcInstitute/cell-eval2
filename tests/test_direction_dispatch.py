import polars as pl

from cell_eval2.catalog import CATALOG, PROFILES, deseq2_metric_name
from cell_eval2.config import EvalConfig
from cell_eval2.de import prepare_de
from cell_eval2.run import dispatch_de_metrics

_NAMES = [
    "de_wilcoxon_direction_precision",
    "de_wilcoxon_direction_sensitivity",
    "de_wilcoxon_direction_sensitivity_universe",
]


def test_catalog_entries_registered_with_the_agreed_metadata():
    expected_scored = {
        "de_wilcoxon_direction_precision": True,
        "de_wilcoxon_direction_sensitivity": True,
        # Now scored too. It is unbounded above and inverts against the generic baseline,
        # which is why it carries anchor=None -- but that is a statement about its
        # mathematics, and enrolment is a separate field. It reaches no v1 output
        # (v1_name=None -> not v1_available), so only the full/de avg_score sees it.
        "de_wilcoxon_direction_sensitivity_universe": True,
    }
    for name in _NAMES:
        spec = CATALOG[name]
        assert spec.kind == "de"
        assert spec.normalization is None
        assert spec.v1_name is None            # v2-native
        assert spec.worst_value == 0.0
        assert tuple(spec.profiles) == ("full", "de")
        assert spec.scoring.scored is expected_scored[name]
        assert spec.scoring.direction == "higher"   # all three, scored or not


def test_metrics_reach_no_competition_profile():
    """Renamed from `test_metrics_are_diagnostic_only`: all three are scored now, so the old
    name contradicted the body. What the body actually asserts is profile membership -- they
    enter the full/de avg_score and never the vcc competition score."""
    for name in _NAMES:
        assert name in PROFILES["full"] and name in PROFILES["de"]
        assert name not in PROFILES["vcc"]
        assert name not in PROFILES["minimal"]


def test_deseq2_siblings_exist():
    for name in _NAMES:
        sibling = name.replace("de_wilcoxon_", "de_deseq2_", 1)
        assert deseq2_metric_name(name) == sibling
        assert sibling in CATALOG


def test_dispatch_fills_omitted_targets_with_the_worst_value_under_v2():
    # G2 has neither a reference-significant nor a model-significant gene, so every metric
    # omits it; v2 no-drop must fill 0.0 rather than let it vanish from the aggregate.
    real = pl.DataFrame({
        "target": ["G1", "G1", "G2"], "feature": ["A", "B", "A"],
        "log2_fold_change": [1.0, 1.0, 1.0], "p_adj": [0.001, 0.001, 0.9],
    })
    pred = pl.DataFrame({
        "target": ["G1", "G1", "G2"], "feature": ["A", "B", "A"],
        "log2_fold_change": [1.0, 1.0, 1.0], "p_adj": [0.001, 0.001, 0.9],
        "p_value": [0.0001, 0.0002, 0.9],
    })
    prep = prepare_de(pred, real, control="non-targeting", p_adj_threshold=0.05)
    rows = dispatch_de_metrics(_NAMES, prep, EvalConfig())
    got = {(r["metric"], r["perturbation"]): r["value"] for r in rows}
    for name in _NAMES:
        assert got[(name, "G2")] == 0.0
        assert got[(name, "G1")] == 1.0
