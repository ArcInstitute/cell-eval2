import polars as pl
import pytest

from cell_eval2.catalog import CATALOG, _NAME_TO_CANONICAL, resolve_metrics
from cell_eval2.run import compute_metrics
from cell_eval2.scoring import ERROR


def test_catalog_keyed_by_canonical_with_v1name():
    assert "expr_mae" in CATALOG and CATALOG["expr_mae"].v1_name == "mae"
    assert "pds_l1" in CATALOG and CATALOG["pds_l1"].v1_name == "discrimination_score_l1"
    assert "de_wilcoxon_overlap" in CATALOG and CATALOG["de_wilcoxon_overlap"].v1_name == "overlap_at_N"
    assert "de_wilcoxon_overlap_top50" in CATALOG
    assert CATALOG["de_wilcoxon_overlap_top50"].v1_name == "overlap_at_50"
    assert "mae" not in CATALOG  # rekeyed


def test_reverse_map_no_collisions_and_maps_both_spellings():
    for spec in CATALOG.values():
        assert _NAME_TO_CANONICAL[spec.name] == spec.name
        if spec.v1_name:
            assert _NAME_TO_CANONICAL[spec.v1_name] == spec.name


def test_resolve_accepts_v1_and_v2_spellings():
    assert resolve_metrics(["mae"]) == (["expr_mae"], [])
    assert resolve_metrics(["expr_mae"]) == (["expr_mae"], [])
    assert resolve_metrics(["overlap_at_N"]) == (["de_wilcoxon_overlap"], [])


def test_version_toggle_same_value_different_label(synthetic_pair):
    pred, real = synthetic_pair
    v2 = compute_metrics(pred, real, metrics=["mae"], version="v2",
                         pert_col="target", control="non-targeting",
                         input_type="lognorm")
    v1 = compute_metrics(pred, real, metrics=["mae"], version="v1",
                         pert_col="target", control="non-targeting",
                         input_type="lognorm")
    assert set(v2["metric"].unique()) == {"expr_mae"}   # v2 (native default) = canonical
    assert set(v1["metric"].unique()) == {"mae"}        # v1 = inherited
    v2v = {r["perturbation"]: r["value"] for r in v2.iter_rows(named=True)}
    v1v = {r["perturbation"]: r["value"] for r in v1.iter_rows(named=True)}
    assert v2v == pytest.approx(v1v)                    # identical values, only the label differs


def test_score_agg_resolves_v1_named_columns():
    from cell_eval2.compat import score_agg_metrics
    user = pl.DataFrame({"statistic": ["mean"], "mae": [0.5]})
    base = pl.DataFrame({"statistic": ["mean"], "mae": [1.0]})
    out = score_agg_metrics(user, base)  # "mae" must resolve to expr_mae's best_value
    assert out.filter(pl.col("metric") == "mae")["from_baseline"].item() == pytest.approx(0.5)


def test_resolve_dedups_v1_and_v2_spellings_of_same_metric():
    # Both spellings of one metric must collapse to a single canonical entry,
    # so compute_metrics doesn't run it twice / double the rows.
    assert resolve_metrics(["mae", "expr_mae"]) == (["expr_mae"], [])


def test_new_de_metric_name_roundtrip():
    pairs = {
        "de_spearman_sig": "de_wilcoxon_nsig_spearman",
        "de_spearman_lfc_sig": "de_wilcoxon_lfc_spearman",
        "de_spearman_pos_lfc_sig": "de_wilcoxon_lfc_spearman_pos",
        "de_spearman_neg_lfc_sig": "de_wilcoxon_lfc_spearman_neg",
        "de_direction_match": "de_wilcoxon_direction_match",
        "de_model_direction_match": "de_wilcoxon_model_direction_match",
        "de_sig_genes_recall": "de_wilcoxon_sig_recall",
        "pr_auc": "de_wilcoxon_pr_auc",
        "roc_auc": "de_wilcoxon_roc_auc",
        "de_nsig_counts_real": "de_wilcoxon_nsig_counts_real",
        "de_nsig_counts_pred": "de_wilcoxon_nsig_counts_pred",
    }
    for v1, v2 in pairs.items():
        avail, missing = resolve_metrics([v1])
        assert avail == [v2] and not missing
        avail2, _ = resolve_metrics([v2, v1])
        assert avail2 == [v2]


def test_build_name_index_skips_empty_v1_name():
    from cell_eval2.catalog import MetricSpec, _build_name_index
    specs = {
        "a": MetricSpec(name="a", v1_name="", func=lambda: {}, scoring=ERROR, agg="mean",
                        profiles=(), kind="anndata", normalization="lognorm"),
    }
    assert _build_name_index(specs) == {"a": "a"}  # empty v1_name is skipped, not indexed


def test_build_name_index_raises_on_collision():
    # A spelling that maps to two different canonical metrics must raise, not
    # silently shadow one — this guards the catalog against accidental dup names.
    from cell_eval2.catalog import MetricSpec, _build_name_index
    specs = {
        "a": MetricSpec(name="a", v1_name="shared", func=lambda: {}, scoring=ERROR,
                        agg="mean", profiles=(), kind="anndata", normalization="lognorm"),
        "b": MetricSpec(name="b", v1_name="shared", func=lambda: {}, scoring=ERROR,
                        agg="mean", profiles=(), kind="anndata", normalization="lognorm"),
    }
    with pytest.raises(ValueError, match="duplicate metric spelling"):
        _build_name_index(specs)
