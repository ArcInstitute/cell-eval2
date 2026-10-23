from cell_eval2.catalog import CATALOG
from cell_eval2.catalog_text import (AXES, FACETS, FAMILIES, METRIC_TEXT, STATISTICS,
                                     de_method, metric_text, text_key)

_K = (50, 100, 200, 500)
FAMILY_MEMBERS = {   # design §7, verbatim
    "expr": {"expr_mae", "expr_mse", "expr_mse_unbiased", "expr_mse_unbiased_capped",
             "expr_distance_unbiased", "expr_real_mass_ratio", "expr_mse_unbiased_capped_norm"},
    "delta": {"delta_mae", "delta_mse", "delta_pearson"},
    "pds": {"pds_l1", "pds_l2", "pds_cosine"},
    "overlap": {"overlap", "precision", *(f"overlap_top{k}" for k in _K),
                *(f"precision_top{k}" for k in _K)},
    "significance": {"nsig_counts_real", "nsig_counts_pred", "nsig_spearman", "sig_recall",
                     "sig_jaccard", "pr_auc", "roc_auc"},
    "lfc": {"lfc_spearman", "lfc_spearman_pos", "lfc_spearman_neg", "lfc_nmae"},
    "direction": {"model_direction_match", "direction_match", "direction_precision",
                  "direction_sensitivity", "direction_sensitivity_universe",
                  "direction_fidelity", "direction_fidelity_raw", "direction_coverage",
                  "direction_yield", "direction_yield_raw", "direction_fidelity_yield",
                  "direction_fidelity_yield_raw", "direction_reach", "direction_reach_raw",
                  "direction_reach_unbounded", "direction_reach_unbounded_raw"},
    "agreement": {"overlap_adjusted", "precision_adjusted", "sig_recall_adjusted", "sig_mcc"},
    "covariance": {"cell_cov_corr", "cell_cov_eig_ratio", "cell_cov_ccc"},
}


def test_every_catalog_entry_has_one_row_and_every_row_an_entry():
    assert {text_key(n) for n in CATALOG} == set(METRIC_TEXT)
    assert len(METRIC_TEXT) == 57


def test_both_de_families_share_their_row():
    for n in CATALOG:
        if n.startswith("de_wilcoxon_"):
            assert metric_text(n) is metric_text(n.replace("de_wilcoxon_", "de_deseq2_", 1))


def test_families_are_the_design_table():
    got = {f: {k for k, t in METRIC_TEXT.items() if t.family == f} for f in FAMILIES}
    assert got == FAMILY_MEMBERS


def test_text_is_one_plain_sentence_and_non_empty():
    for key, t in METRIC_TEXT.items():
        assert t.display_name and t.display_name == t.display_name.strip(), key
        assert t.description.endswith(".") and ". " not in t.description[:-1], key
        assert "`" not in t.display_name + t.description, key


def test_bounds_follow_the_draft():
    def b(k):
        return METRIC_TEXT[k].min, METRIC_TEXT[k].max
    assert b("pr_auc") == (0.0, 1.0) and b("sig_mcc") == (-1.0, 1.0)
    assert b("direction_fidelity") == b("direction_fidelity_yield") == (-20.0, 1.0)
    assert b("lfc_nmae") == b("cell_cov_eig_ratio") == (0.0, None)
    assert b("direction_yield") == b("expr_mse_unbiased_capped_norm") == (None, None)
    for key, t in METRIC_TEXT.items():
        if t.min is not None and t.max is not None:
            assert t.min < t.max, key


def test_de_method_only_on_de_metrics():
    for n, spec in CATALOG.items():
        assert (de_method(n) is None) == (spec.kind != "de"), n
    assert de_method("de_wilcoxon_overlap") == "wilcoxon"
    assert de_method("de_deseq2_overlap") == "deseq2"


def test_declarations():
    assert {s: d["combine"] for s, d in STATISTICS.items()} == {
        "count": "sum", "null_count": "sum", "mean": "mean", "std": "none", "min": "none",
        "max": "none", "median": "none", "value": "mean", "ratio_of_sums": "mean"}
    assert list(AXES) == ["perturbation"] and list(FACETS) == ["family", "kind", "de_method"]
    for d in (*STATISTICS.values(), *AXES.values(), *FACETS.values()):
        assert d["display_name"] and d["description"].endswith(".")
        assert "`" not in d["display_name"] + d["description"]
