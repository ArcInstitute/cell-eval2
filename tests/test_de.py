import polars as pl
import pytest
from cell_eval2.de import (
    PreparedDE,
    apply_lfc_floor,
    apply_nan_policy,
    assemble_prepared_de,
    load_de_table,
    normalize_de_schema,
    prepare_de,
    prep_de_side,
    rank_de_side,
)
from cell_eval2.metrics.de import de_overlap


def _df():
    return pl.DataFrame({"target": ["A"], "feature": ["g1"],
                         "log2_fold_change": [1.0], "p_value": [0.01], "p_adj": [0.02]})


def test_load_from_polars_is_identity():
    df = _df()
    assert load_de_table(df).equals(df)


def test_load_from_csv(tmp_path):
    p = tmp_path / "de.csv"
    _df().write_csv(str(p))
    assert load_de_table(str(p)).shape == (1, 5)


def test_load_from_parquet(tmp_path):
    p = tmp_path / "de.parquet"
    _df().write_parquet(str(p))
    assert load_de_table(str(p)).shape == (1, 5)


def test_load_bad_extension_raises(tmp_path):
    p = tmp_path / "de.txt"
    p.write_text("x")
    with pytest.raises(ValueError, match="extension"):
        load_de_table(str(p))


def test_fdr_aliased_to_p_adj():
    df = pl.DataFrame({"target": ["A"], "feature": ["g"], "log2_fold_change": [2.0],
                       "p_value": [0.01], "fdr": [0.03]})
    out = normalize_de_schema(df, name="real")
    assert "p_adj" in out.columns and out["p_adj"][0] == 0.03


def test_abs_lfc_always_derived_overwrites():
    df = pl.DataFrame({"target": ["A"], "feature": ["g"], "log2_fold_change": [-2.0],
                       "p_adj": [0.01], "abs_log2_fold_change": [999.0]})
    out = normalize_de_schema(df, name="real")
    assert out["abs_log2_fold_change"][0] == 2.0  # recomputed, not 999.0


def test_missing_required_column_raises():
    df = pl.DataFrame({"target": ["A"], "feature": ["g"], "p_adj": [0.01]})  # no log2_fold_change
    with pytest.raises(ValueError, match="missing required"):
        normalize_de_schema(df, name="pred")


def test_nulls_in_required_columns_warn(caplog):
    import logging
    df = pl.DataFrame({"target": ["A", None], "feature": ["g", "h"],
                       "log2_fold_change": [1.0, 2.0], "p_adj": [0.01, None]})
    with caplog.at_level(logging.WARNING):
        normalize_de_schema(df, name="real")
    assert any("nulls in required columns" in r.message for r in caplog.records)


def _nan_df():
    return pl.DataFrame({
        "target": ["A", "A"], "feature": ["g1", "g2"],
        "log2_fold_change": [float("nan"), 1.0], "p_adj": [0.001, 0.001],
    })


def test_nan_policy_keep_leaves_p_adj():
    out = apply_nan_policy(_nan_df(), name="real", nan_lfc_policy="keep")
    assert out["p_adj"].to_list() == [0.001, 0.001]


def test_nan_policy_mask_sets_p_adj_one_on_nan_rows():
    out = apply_nan_policy(_nan_df(), name="real", nan_lfc_policy="mask")
    # row 0 (NaN LFC) masked to 1.0; row 1 untouched
    assert out["p_adj"].to_list() == [1.0, 0.001]


def test_nan_policy_mask_warns_on_incoherent(caplog):
    import logging
    with caplog.at_level(logging.WARNING):
        apply_nan_policy(_nan_df(), name="real", nan_lfc_policy="mask")
    assert any("p_adj<1" in r.message for r in caplog.records)


def test_nan_policy_invalid_raises():
    with pytest.raises(ValueError, match="nan_lfc_policy"):
        apply_nan_policy(_nan_df(), name="real", nan_lfc_policy="bogus")


def _pair():
    # targets A,B share gene set; ranking by abs_log2_fold_change (desc)
    real = pl.DataFrame({
        "target": ["A", "A", "B", "B"], "feature": ["g1", "g2", "g3", "g4"],
        "log2_fold_change": [3.0, 2.0, 1.0, 5.0], "p_adj": [0.001, 0.001, 0.2, 0.001],
    })
    pred = pl.DataFrame({
        "target": ["A", "A", "B", "B"], "feature": ["g1", "g9", "g4", "g3"],
        "log2_fold_change": [4.0, 2.0, 5.0, 1.0], "p_adj": [0.001, 0.001, 0.001, 0.2],
    })
    return pred, real


def _toy_prepared(real, pred, threshold=0.05):
    return prepare_de(pred, real, control="non-targeting", p_adj_threshold=threshold)


def test_prepare_de_strict_alignment_raises():
    pred, real = _pair()
    pred2 = pred.with_columns(pl.col("target").str.replace("B", "C"))
    with pytest.raises(ValueError, match="target sets differ"):
        prepare_de(pred2, real, control="non-targeting")


def test_prepare_de_control_as_target_raises():
    pred, real = _pair()
    with pytest.raises(ValueError, match="control"):
        prepare_de(pred, real, control="A")


def test_prepare_de_perts_sorted_shared():
    pred, real = _pair()
    prep = prepare_de(pred, real, control="non-targeting")
    assert prep.perturbations == ["A", "B"]
    assert isinstance(prep, PreparedDE)


def test_prepared_de_carries_normalized_frames():
    real = pl.DataFrame({
        "target": ["GENE1", "GENE1", "GENE2"],
        "feature": ["A", "B", "A"],
        "log2_fold_change": [3.0, -2.0, 1.0],
        "p_adj": [0.001, 0.2, 0.001],
    })
    pred = pl.DataFrame({
        "target": ["GENE1", "GENE1", "GENE2"],
        "feature": ["A", "B", "A"],
        "log2_fold_change": [2.5, -1.0, 0.5],
        "p_adj": [0.001, 0.3, 0.04],
    })
    prep = prepare_de(pred, real, control="non-targeting", p_adj_threshold=0.05)
    # Full frames are present, carry p_adj + log2_fold_change for ALL genes (not just sig).
    assert {"target", "feature", "log2_fold_change", "p_adj"} <= set(prep.real_df.columns)
    assert prep.real_df.height == 3 and prep.pred_df.height == 3
    assert prep.p_adj_threshold == 0.05


def test_overlap_at_N_recall_denominator():
    pred, real = _pair()
    prep = prepare_de(pred, real, control="non-targeting", nan_lfc_policy="keep")
    got = de_overlap(prep, k=None, metric="overlap")
    # A: real sig {g1,g2}, pred sig {g1,g9}; overlap {g1} / 2 = 0.5
    # B: real sig {g4} (g3 not sig), pred sig {g4} (g3 not sig); {g4}/1 = 1.0
    assert got["A"] == pytest.approx(0.5)
    assert got["B"] == pytest.approx(1.0)


def test_overlap_topk_caps():
    pred, real = _pair()
    prep = prepare_de(pred, real, control="non-targeting", nan_lfc_policy="keep")
    got = de_overlap(prep, k=1, metric="overlap")
    # A top1 real=g1 (lfc 3), pred=g1 (lfc 4) -> overlap 1.0
    assert got["A"] == pytest.approx(1.0)


def test_no_significant_genes_all_zero():
    df = pl.DataFrame({"target": ["A", "B"], "feature": ["g", "g"],
                       "log2_fold_change": [1.0, 1.0], "p_adj": [0.9, 0.9]})
    prep = prepare_de(df, df, control="non-targeting", nan_lfc_policy="keep")
    got = de_overlap(prep, k=None, metric="overlap")
    assert got == {"A": 0.0, "B": 0.0}


def test_de_overlap_raw_frames_path():
    pred, real = _pair()
    got = de_overlap(de_pred=pred, de_real=real, k=None, metric="overlap",
                     nan_lfc_policy="keep", control="non-targeting")
    assert got["A"] == pytest.approx(0.5)


def test_nsig_counts_real_and_pred():
    from cell_eval2.metrics.de import de_nsig_counts
    real = pl.DataFrame({
        "target": ["G1", "G1", "G2"],
        "feature": ["A", "B", "A"],
        "log2_fold_change": [3.0, 2.0, 1.0],
        "p_adj": [0.001, 0.001, 0.2],
    })
    pred = pl.DataFrame({
        "target": ["G1", "G1", "G2"],
        "feature": ["A", "B", "A"],
        "log2_fold_change": [3.0, 2.0, 1.0],
        "p_adj": [0.001, 0.3, 0.001],
    })
    prep = _toy_prepared(real, pred)
    assert de_nsig_counts(prep, side="real") == {"G1": 2.0, "G2": 0.0}
    assert de_nsig_counts(prep, side="pred") == {"G1": 1.0, "G2": 1.0}


def test_nsig_spearman_broadcasts_global_scalar():
    from cell_eval2.metrics.de import de_nsig_spearman
    # real sig counts per target: G1=2, G2=1, G3=0 (only real-sig targets correlate)
    real = pl.DataFrame({
        "target": ["G1", "G1", "G2", "G3"],
        "feature": ["A", "B", "A", "A"],
        "log2_fold_change": [3.0, 2.0, 1.0, 1.0],
        "p_adj": [0.001, 0.001, 0.001, 0.9],
    })
    pred = pl.DataFrame({
        "target": ["G1", "G1", "G2", "G3"],
        "feature": ["A", "B", "A", "A"],
        "log2_fold_change": [3.0, 2.0, 1.0, 1.0],
        "p_adj": [0.001, 0.001, 0.9, 0.9],
    })
    prep = _toy_prepared(real, pred)
    out = de_nsig_spearman(prep)
    # Same value broadcast to every perturbation.
    assert set(out.keys()) == set(prep.perturbations)
    assert len({round(v, 12) for v in out.values()}) == 1


def test_nsig_spearman_empty_returns_one():
    from cell_eval2.metrics.de import de_nsig_spearman
    real = pl.DataFrame({
        "target": ["G1"],
        "feature": ["A"],
        "log2_fold_change": [3.0],
        "p_adj": [0.9],
    })
    pred = pl.DataFrame({
        "target": ["G1"],
        "feature": ["A"],
        "log2_fold_change": [3.0],
        "p_adj": [0.9],
    })
    prep = _toy_prepared(real, pred)
    assert all(v == 1.0 for v in de_nsig_spearman(prep).values())


def test_sig_recall_intersection_over_real():
    from cell_eval2.metrics.de import de_sig_recall
    # G1 real-sig {A,B}, pred-sig {A,C} -> recall 1/2; G2 real-sig {} -> omitted
    real = pl.DataFrame({
        "target": ["G1", "G1", "G2"],
        "feature": ["A", "B", "A"],
        "log2_fold_change": [3.0, 2.0, 1.0],
        "p_adj": [0.001, 0.001, 0.9],
    })
    pred = pl.DataFrame({
        "target": ["G1", "G1", "G2"],
        "feature": ["A", "C", "A"],
        "log2_fold_change": [3.0, 2.0, 1.0],
        "p_adj": [0.001, 0.001, 0.001],
    })
    out = de_sig_recall(_toy_prepared(real, pred))
    assert out["G1"] == 0.5
    assert "G2" not in out  # zero real-sig genes -> omitted


def test_direction_match_sign_agreement():
    from cell_eval2.metrics.de import de_direction_match
    # G1 real-sig {A:+, B:-}; pred A:+ (match), B:+ (mismatch) -> 0.5
    real = pl.DataFrame({
        "target": ["G1", "G1"],
        "feature": ["A", "B"],
        "log2_fold_change": [3.0, -2.0],
        "p_adj": [0.001, 0.001],
    })
    pred = pl.DataFrame({
        "target": ["G1", "G1"],
        "feature": ["A", "B"],
        "log2_fold_change": [1.0, 4.0],
        "p_adj": [0.5, 0.5],
    })
    out = de_direction_match(_toy_prepared(real, pred))
    assert out["G1"] == 0.5


def test_de_model_direction_match_uses_model_significant_set():
    from cell_eval2.metrics.de import de_direction_match, de_model_direction_match
    # Model-sig {B:-, C:-}; real B:- (match), C:+ (mismatch) -> 0.5.
    # A is the only real-significant gene, but is model-nonsignificant and must not enter
    # this reverse metric's denominator.
    real = pl.DataFrame({
        "target": ["G1", "G1", "G1"],
        "feature": ["A", "B", "C"],
        "log2_fold_change": [3.0, -2.0, 4.0],
        "p_adj": [0.001, 0.5, 0.5],
    })
    pred = pl.DataFrame({
        "target": ["G1", "G1", "G1"],
        "feature": ["A", "B", "C"],
        "log2_fold_change": [-1.0, -4.0, -2.0],
        "p_adj": [0.5, 0.001, 0.001],
    })
    prep = _toy_prepared(real, pred)
    assert de_model_direction_match(prep)["G1"] == 0.5
    assert de_direction_match(prep)["G1"] == 0.0


def test_de_model_direction_match_omits_target_with_no_model_degs():
    from cell_eval2.metrics.de import de_model_direction_match
    real = pl.DataFrame({
        "target": ["G1"], "feature": ["A"],
        "log2_fold_change": [2.0], "p_adj": [0.001],
    })
    pred = pl.DataFrame({
        "target": ["G1"], "feature": ["A"],
        "log2_fold_change": [2.0], "p_adj": [0.5],
    })
    assert de_model_direction_match(_toy_prepared(real, pred)) == {}


def test_lfc_spearman_per_target():
    import math
    from cell_eval2.metrics.de import de_lfc_spearman
    # G1 real-sig 3 genes; pred LFCs same rank order -> spearman 1.0
    real = pl.DataFrame({
        "target": ["G1", "G1", "G1"],
        "feature": ["A", "B", "C"],
        "log2_fold_change": [3.0, 2.0, 1.0],
        "p_adj": [0.001, 0.001, 0.001],
    })
    pred = pl.DataFrame({
        "target": ["G1", "G1", "G1"],
        "feature": ["A", "B", "C"],
        "log2_fold_change": [6.0, 4.0, 2.0],
        "p_adj": [0.5, 0.5, 0.5],
    })
    out = de_lfc_spearman(_toy_prepared(real, pred))
    assert math.isclose(out["G1"], 1.0, rel_tol=1e-9)


def test_lfc_spearman_single_gene_is_nan():
    import math
    from cell_eval2.metrics.de import de_lfc_spearman
    real = pl.DataFrame({
        "target": ["G1"],
        "feature": ["A"],
        "log2_fold_change": [3.0],
        "p_adj": [0.001],
    })
    pred = pl.DataFrame({
        "target": ["G1"],
        "feature": ["A"],
        "log2_fold_change": [2.0],
        "p_adj": [0.5],
    })
    assert math.isnan(de_lfc_spearman(_toy_prepared(real, pred))["G1"])


def test_lfc_spearman_pos_restricts_to_positive_real_lfc():
    import math
    from cell_eval2.metrics.de import de_lfc_spearman
    # G1: real-sig genes with real LFC>0 (A,B), <0 (C), and NaN (D). pos should correlate
    # only over {A,B} -> 1.0. C is excluded by sign; D (NaN LFC) is excluded because the
    # default nan_lfc_policy='mask' forces p_adj=1 on NaN-LFC rows so they never reach the
    # significance gate — so the `> 0` filter never sees a NaN.
    real = pl.DataFrame({
        "target": ["G1", "G1", "G1", "G1"],
        "feature": ["A", "B", "C", "D"],
        "log2_fold_change": [3.0, 2.0, -4.0, float("nan")],
        "p_adj": [0.001, 0.001, 0.001, 0.001],
    })
    pred = pl.DataFrame({
        "target": ["G1", "G1", "G1", "G1"],
        "feature": ["A", "B", "C", "D"],
        "log2_fold_change": [6.0, 4.0, 5.0, 1.0],   # C,D preds are positive: would perturb an all-corr
        "p_adj": [0.5, 0.5, 0.5, 0.5],
    })
    out = de_lfc_spearman(_toy_prepared(real, pred), lfc_direction="pos")
    assert math.isclose(out["G1"], 1.0, rel_tol=1e-9)


def test_lfc_spearman_neg_restricts_to_negative_real_lfc():
    import math
    from cell_eval2.metrics.de import de_lfc_spearman
    # Only C,D have real LFC<0; over {C,D} pred preserves rank order -> 1.0.
    real = pl.DataFrame({
        "target": ["G1", "G1", "G1", "G1"],
        "feature": ["A", "B", "C", "D"],
        "log2_fold_change": [3.0, 2.0, -1.0, -2.0],
        "p_adj": [0.001, 0.001, 0.001, 0.001],
    })
    pred = pl.DataFrame({
        "target": ["G1", "G1", "G1", "G1"],
        "feature": ["A", "B", "C", "D"],
        "log2_fold_change": [6.0, 4.0, -3.0, -8.0],
        "p_adj": [0.5, 0.5, 0.5, 0.5],
    })
    out = de_lfc_spearman(_toy_prepared(real, pred), lfc_direction="neg")
    assert math.isclose(out["G1"], 1.0, rel_tol=1e-9)


def test_lfc_spearman_pos_all_negative_real_is_empty():
    # No positive-real-sig genes -> target omitted entirely (matches upstream + drives v2 worst-fill).
    from cell_eval2.metrics.de import de_lfc_spearman
    real = pl.DataFrame({
        "target": ["G1", "G1"], "feature": ["A", "B"],
        "log2_fold_change": [-3.0, -2.0], "p_adj": [0.001, 0.001],
    })
    pred = pl.DataFrame({
        "target": ["G1", "G1"], "feature": ["A", "B"],
        "log2_fold_change": [1.0, 2.0], "p_adj": [0.5, 0.5],
    })
    assert de_lfc_spearman(_toy_prepared(real, pred), lfc_direction="pos") == {}


def test_lfc_spearman_invalid_direction_raises():
    import pytest
    from cell_eval2.metrics.de import de_lfc_spearman
    real = pl.DataFrame({"target": ["G1"], "feature": ["A"],
                         "log2_fold_change": [3.0], "p_adj": [0.001]})
    pred = pl.DataFrame({"target": ["G1"], "feature": ["A"],
                         "log2_fold_change": [2.0], "p_adj": [0.5]})
    with pytest.raises(ValueError, match="lfc_direction"):
        de_lfc_spearman(_toy_prepared(real, pred), lfc_direction="up")


def test_pr_roc_auc_perfect_and_degenerate():
    import math
    from cell_eval2.metrics.de import de_pr_auc, de_roc_auc
    # G1: A,B real-sig (label 1), C,D not (label 0); pred ranks sig genes ahead -> AUC 1.0
    real = pl.DataFrame({
        "target": ["G1"] * 4,
        "feature": ["A", "B", "C", "D"],
        "log2_fold_change": [3.0, 2.0, 1.0, 1.0],
        "p_adj": [0.001, 0.001, 0.9, 0.9],
    })
    pred = pl.DataFrame({
        "target": ["G1"] * 4,
        "feature": ["A", "B", "C", "D"],
        "log2_fold_change": [3.0, 2.0, 1.0, 1.0],
        "p_adj": [0.001, 0.01, 0.9, 0.95],
    })
    prep = _toy_prepared(real, pred)
    assert math.isclose(de_roc_auc(prep)["G1"], 1.0, rel_tol=1e-9)
    assert math.isclose(de_pr_auc(prep)["G1"], 1.0, rel_tol=1e-9)
    # All-one-class label -> NaN
    real2 = pl.DataFrame({
        "target": ["G2", "G2"],
        "feature": ["A", "B"],
        "log2_fold_change": [3.0, 2.0],
        "p_adj": [0.001, 0.001],
    })
    pred2 = pl.DataFrame({
        "target": ["G2", "G2"],
        "feature": ["A", "B"],
        "log2_fold_change": [3.0, 2.0],
        "p_adj": [0.001, 0.5],
    })
    assert math.isnan(de_roc_auc(_toy_prepared(real2, pred2))["G2"])


@pytest.mark.parametrize("metric", ["overlap", "precision"])
def test_k_zero_is_zero_not_full(metric):
    # k==0 means "top-0 genes" -> 0.0; must NOT fall back to the full-list size.
    pred, real = _pair()
    prep = prepare_de(pred, real, control="non-targeting", nan_lfc_policy="keep")
    got = de_overlap(prep, k=0, metric=metric)
    assert got == {"A": 0.0, "B": 0.0}


def test_negative_k_raises():
    pred, real = _pair()
    prep = prepare_de(pred, real, control="non-targeting", nan_lfc_policy="keep")
    with pytest.raises(ValueError, match="non-negative"):
        de_overlap(prep, k=-1, metric="overlap")


def test_prepare_de_sort_by_missing_column_raises():
    # _pair() tables have no p_value column; sort_by='p_value' must fail fast.
    pred, real = _pair()
    with pytest.raises(ValueError, match="p_value"):
        prepare_de(pred, real, control="non-targeting", sort_by="p_value",
                   nan_lfc_policy="keep")


def test_prepare_de_null_target_raises():
    bad = pl.DataFrame({"target": ["A", None], "feature": ["g1", "g2"],
                        "log2_fold_change": [1.0, 2.0], "p_adj": [0.01, 0.01]})
    good = pl.DataFrame({"target": ["A", "A"], "feature": ["g1", "g2"],
                         "log2_fold_change": [1.0, 2.0], "p_adj": [0.01, 0.01]})
    with pytest.raises(ValueError, match="null values in 'target'"):
        prepare_de(bad, good, control="non-targeting", nan_lfc_policy="keep")


def test_perturbation_named_rank_does_not_collide():
    # A target literally named "rank" must not collide with the (now sentinel) index.
    df = pl.DataFrame({"target": ["rank", "rank", "A"], "feature": ["g1", "g2", "g3"],
                       "log2_fold_change": [3.0, 2.0, 1.0], "p_adj": [0.001, 0.001, 0.001]})
    prep = prepare_de(df, df, control="non-targeting", nan_lfc_policy="keep")
    # rank matrix holds only perturbation columns (sentinel index dropped)
    assert set(prep.real_rank.columns) == {"rank", "A"}
    got = de_overlap(prep, k=None, metric="overlap")
    assert got["rank"] == pytest.approx(1.0)  # identical tables -> full overlap
    assert got["A"] == pytest.approx(1.0)


def _toy_de(targets=("GENE1", "GENE2")):
    rows = []
    for t in targets:
        rows += [{"target": t, "feature": "g0", "log2_fold_change": 3.0, "p_adj": 0.001},
                 {"target": t, "feature": "g1", "log2_fold_change": -2.0, "p_adj": 0.01},
                 {"target": t, "feature": "g2", "log2_fold_change": 0.1, "p_adj": 0.9}]
    return pl.DataFrame(rows)


def test_prep_de_side_returns_df_and_perts():
    df, perts = prep_de_side(_toy_de(), name="real", sort_by="abs_log2_fold_change",
                             nan_lfc_policy="mask")
    assert perts == ["GENE1", "GENE2"]
    assert "abs_log2_fold_change" in df.columns


def test_rank_de_side_filters_and_ranks():
    df, _ = prep_de_side(_toy_de(), name="real", sort_by="abs_log2_fold_change",
                         nan_lfc_policy="mask")
    rank = rank_de_side(df, sort_by="abs_log2_fold_change", p_adj_threshold=0.05)
    assert set(rank.columns) == {"GENE1", "GENE2"}      # only significant genes ranked
    assert rank.height == 2                              # g0, g1 pass; g2 (p_adj=0.9) drops


def test_assemble_prepared_de_cross_check():
    df_r, perts_r = prep_de_side(_toy_de(), name="real", sort_by="abs_log2_fold_change",
                                 nan_lfc_policy="mask")
    rank_r = rank_de_side(df_r, sort_by="abs_log2_fold_change", p_adj_threshold=0.05)
    with pytest.raises(ValueError, match="target sets differ"):
        assemble_prepared_de(rank_r, perts_r, rank_r, ["GENE1"], control="non-targeting",
                             sort_by="abs_log2_fold_change", p_adj_threshold=0.05,
                             real_df=rank_r, pred_df=rank_r)
    with pytest.raises(ValueError, match="must not appear as a DE target"):
        assemble_prepared_de(rank_r, perts_r, rank_r, perts_r, control="GENE1",
                             sort_by="abs_log2_fold_change", p_adj_threshold=0.05,
                             real_df=rank_r, pred_df=rank_r)


def test_auc_metrics_handle_nan_pred_p_adj():
    # A NaN pred p_adj (degenerate genes from the CPM-gate BH recompute) must not crash the
    # AUC metrics: fill_nan -> treated as non-significant (score 0), no ValueError from sklearn.
    import math

    from cell_eval2.metrics.de import de_pr_auc, de_roc_auc

    real = pl.DataFrame({"target": ["G1"] * 4, "feature": ["A", "B", "C", "D"],
                         "log2_fold_change": [3.0, 2.0, 1.0, 1.0],
                         "p_adj": [0.001, 0.001, 0.9, 0.9]})
    pred = pl.DataFrame({"target": ["G1"] * 4, "feature": ["A", "B", "C", "D"],
                         "log2_fold_change": [3.0, 2.0, 1.0, 1.0],
                         "p_adj": [float("nan"), 0.01, 0.9, 0.95]})
    prep = _toy_prepared(real, pred)
    for fn in (de_pr_auc, de_roc_auc):
        val = fn(prep)["G1"]
        assert isinstance(val, float) and not math.isinf(val)  # finite or NaN, never a crash


def _auc_divergent_prepared():
    # GENE1 exercises the floor regime: f1 is a real-significant gene whose pred p_adj
    # is EXACTLY 0 (most significant); f2 real-sig with sub-1e-10 nonzero; f3 a
    # real-NONsig gene with a sub-1e-10 nonzero pred p (ranked high by replace_zero);
    # f4 real-nonsig, normal pred p. GENE2/GENE3 are strategy-invariant (mixed labels,
    # no zeros/sub-1e-10) so per-pert AUC stays valid.
    real = pl.DataFrame({
        "target": ["GENE1"] * 4 + ["GENE2"] * 2 + ["GENE3"] * 2,
        "feature": ["f1", "f2", "f3", "f4", "g1", "g2", "h1", "h2"],
        "log2_fold_change": [3.0, 2.0, 1.0, 1.0, 2.0, 1.0, 2.0, 1.0],
        "p_adj": [0.001, 0.001, 0.9, 0.9, 0.001, 0.9, 0.001, 0.9],
    })
    pred = pl.DataFrame({
        "target": ["GENE1"] * 4 + ["GENE2"] * 2 + ["GENE3"] * 2,
        "feature": ["f1", "f2", "f3", "f4", "g1", "g2", "h1", "h2"],
        "log2_fold_change": [3.0, 2.0, 1.0, 1.0, 2.0, 1.0, 2.0, 1.0],
        "p_adj": [0.0, 1e-300, 1e-50, 0.9, 0.001, 0.9, 0.001, 0.9],
    })
    return _toy_prepared(real, pred)


def test_auc_min_nonzero_is_order_consistent():
    import math
    from cell_eval2.metrics.de import de_roc_auc
    prep = _auc_divergent_prepared()
    # replace_zero: p=0 gene f1 -> 1e-10 (score 10), but f3 (nonsig) keeps 1e-50
    # (score 50) -> f1 ranked BELOW f3 -> ROC 0.75 (order-inverted).
    rz = de_roc_auc(prep, auc_pval_floor="replace_zero", auc_pval_floor_value=1e-10)["GENE1"]
    # min_nonzero: f1 (p=0) -> smallest nonzero (1e-300) -> top score -> perfect ROC 1.0.
    mn = de_roc_auc(prep, auc_pval_floor="min_nonzero")["GENE1"]
    assert math.isclose(rz, 0.75, abs_tol=1e-9)
    assert math.isclose(mn, 1.0, abs_tol=1e-9)
    assert not math.isclose(rz, mn, abs_tol=1e-6)


def test_auc_clip_reproduces_pre_change_behavior():
    import math
    from cell_eval2.metrics.de import de_roc_auc
    prep = _auc_divergent_prepared()
    # clip(1e-10, 1.0): every p <= 1e-10 (0, 1e-300, 1e-50) -> 1e-10 (score 10) ties
    # f1,f2,f3 -> ROC 0.75. This is exactly the old hard-coded behavior.
    clipped = de_roc_auc(prep, auc_pval_floor="clip", auc_pval_floor_value=1e-10)["GENE1"]
    assert math.isclose(clipped, 0.75, abs_tol=1e-9)


def test_auc_default_strategy_is_min_nonzero():
    import math
    from cell_eval2.metrics.de import de_roc_auc
    prep = _auc_divergent_prepared()
    # No explicit strategy -> min_nonzero (library v2 default).
    assert math.isclose(de_roc_auc(prep)["GENE1"], 1.0, abs_tol=1e-9)


def test_auc_nan_guard_holds_under_all_strategies():
    import math
    from cell_eval2.metrics.de import de_pr_auc, de_roc_auc
    real = pl.DataFrame({"target": ["G1"] * 4, "feature": ["A", "B", "C", "D"],
                         "log2_fold_change": [3.0, 2.0, 1.0, 1.0],
                         "p_adj": [0.001, 0.001, 0.9, 0.9]})
    pred = pl.DataFrame({"target": ["G1"] * 4, "feature": ["A", "B", "C", "D"],
                         "log2_fold_change": [3.0, 2.0, 1.0, 1.0],
                         "p_adj": [float("nan"), 0.0, 0.9, 0.95]})
    prep = _toy_prepared(real, pred)
    for strat in ("clip", "replace_zero", "min_nonzero"):
        for fn in (de_pr_auc, de_roc_auc):
            val = fn(prep, auc_pval_floor=strat)["G1"]
            assert isinstance(val, float) and not math.isinf(val)  # finite or NaN, never a crash


def test_auc_direct_call_validates_floor_args():
    # de_pr_auc/de_roc_auc are public and bypass DEParams validation, so _de_auc must
    # guard its own args (an out-of-range value -> -log10(0)=inf would silently corrupt AUC).
    from cell_eval2.metrics.de import de_pr_auc
    prep = _auc_divergent_prepared()
    with pytest.raises(ValueError, match="auc_pval_floor_value"):
        de_pr_auc(prep, auc_pval_floor_value=0.0)
    with pytest.raises(ValueError, match="auc_pval_floor_value"):
        de_pr_auc(prep, auc_pval_floor_value=2.0)
    with pytest.raises(ValueError, match="auc_pval_floor"):
        de_pr_auc(prep, auc_pval_floor="bogus")


def _auc_global_floor_prepared():
    # GENE1's exact-0 pred p_adj sits on a real-NONsig gene (f3). GENE1's own
    # smallest nonzero is 1e-4 (f1); the GLOBAL smallest nonzero (1e-200) lives in
    # GENE2 (g1). The GLOBAL floor turns f3 (0 -> 1e-200, huge score) into a top-rank
    # non-sig gene -> GENE1 ROC 0.5. A per-pert floor (GENE1's own 1e-4) would leave
    # f3 below the sig genes -> ROC 0.625.
    real = pl.DataFrame({
        "target": ["GENE1"] * 4 + ["GENE2"] * 2,
        "feature": ["f1", "f2", "f3", "f4", "g1", "g2"],
        "log2_fold_change": [3.0, 2.0, 1.0, 1.0, 2.0, 1.0],
        "p_adj": [0.001, 0.001, 0.9, 0.9, 0.001, 0.9],
    })
    pred = pl.DataFrame({
        "target": ["GENE1"] * 4 + ["GENE2"] * 2,
        "feature": ["f1", "f2", "f3", "f4", "g1", "g2"],
        "log2_fold_change": [3.0, 2.0, 1.0, 1.0, 2.0, 1.0],
        "p_adj": [1e-4, 1e-3, 0.0, 0.5, 1e-200, 0.9],
    })
    return _toy_prepared(real, pred)


def test_auc_min_nonzero_floor_is_global_not_per_pert():
    import math
    from cell_eval2.metrics.de import de_roc_auc
    prep = _auc_global_floor_prepared()
    res = de_roc_auc(prep, auc_pval_floor="min_nonzero")
    # GENE1's exact-0 floored with the GLOBAL min (1e-200 from GENE2), not GENE1's
    # own 1e-4: global -> 0.5; a per-pert floor would give 0.625.
    assert math.isclose(res["GENE1"], 0.5, abs_tol=1e-9)
    # GENE2 has no zeros -> unaffected, perfect separation.
    assert math.isclose(res["GENE2"], 1.0, abs_tol=1e-9)


def _floor_frame():
    # abs_log2_fold_change present as if normalize_de_schema already ran.
    return pl.DataFrame({
        "target": ["A", "A", "A"],
        "feature": ["g_big", "g_small", "g_ns"],
        "log2_fold_change": [2.0, 0.3, 5.0],
        "p_value": [0.001, 0.001, 0.9],
        "p_adj": [0.01, 0.01, 0.9],
        "abs_log2_fold_change": [2.0, 0.3, 5.0],
    })


def _raw_floor_frame():
    # No abs col: as loaded before normalize_de_schema.
    return pl.DataFrame({
        "target": ["A", "A"], "feature": ["g_big", "g_small"],
        "log2_fold_change": [2.0, 0.3], "p_value": [0.001, 0.001], "p_adj": [0.01, 0.01],
    })


def test_apply_lfc_floor_masks_small_effect():
    out = apply_lfc_floor(_floor_frame(), name="x", min_abs_log2fc=1.0)
    d = dict(zip(out["feature"].to_list(), out["p_adj"].to_list()))
    assert d["g_big"] == 0.01     # |lfc|=2.0 >= 1.0 -> kept
    assert d["g_small"] == 1.0    # |lfc|=0.3 < 1.0 -> masked out of significance
    assert d["g_ns"] == 0.9       # |lfc|=5.0 >= 1.0 -> untouched


def test_apply_lfc_floor_noop_at_zero():
    frame = _floor_frame()
    assert apply_lfc_floor(frame, name="x", min_abs_log2fc=0.0).equals(frame)


def test_apply_lfc_floor_boundary_is_inclusive():
    out = apply_lfc_floor(_floor_frame(), name="x", min_abs_log2fc=0.3)
    d = dict(zip(out["feature"].to_list(), out["p_adj"].to_list()))
    assert d["g_small"] == 0.01   # |lfc| == floor -> kept (strict <)


def test_apply_lfc_floor_ignores_nonfinite_lfc():
    frame = pl.DataFrame({
        "target": ["A", "A", "A"], "feature": ["g_nan", "g_inf", "g_null"],
        "log2_fold_change": [float("nan"), float("inf"), None],
        "p_value": [0.001, 0.001, 0.001], "p_adj": [0.01, 0.01, 0.01],
        "abs_log2_fold_change": [float("nan"), float("inf"), None],
    })
    out = apply_lfc_floor(frame, name="x", min_abs_log2fc=1.0)
    d = dict(zip(out["feature"].to_list(), out["p_adj"].to_list()))
    assert d["g_nan"] == 0.01     # NaN < floor is not true -> untouched
    assert d["g_inf"] == 0.01     # inf < floor is false -> max effect kept
    assert d["g_null"] == 0.01    # null < floor is null -> fill_null(False) -> kept (not nulled)


def test_apply_lfc_floor_rejects_bad_values():
    # Direct callers bypass DEParams validation, so the chokepoint rejects negative/non-finite.
    frame = _floor_frame()
    for bad in (-0.5, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="min_abs_log2fc"):
            apply_lfc_floor(frame, name="x", min_abs_log2fc=bad)


def test_prep_de_side_applies_floor():
    df, perts = prep_de_side(_raw_floor_frame(), name="real",
                             sort_by="abs_log2_fold_change",
                             nan_lfc_policy="mask", min_abs_log2fc=1.0)
    d = dict(zip(df["feature"].to_list(), df["p_adj"].to_list()))
    assert d["g_small"] == 1.0 and d["g_big"] == 0.01


def test_prepare_de_floor_excludes_from_rank_and_keeps_universe():
    prep = prepare_de(_raw_floor_frame(), _raw_floor_frame(),
                      control="ctrl", min_abs_log2fc=1.0)
    surviving = set(prep.real_rank["A"].drop_nulls().to_list())
    assert surviving == {"g_big"}          # floored gene not in the significant rank set
    assert prep.real_df.height == 2        # universe preserved (row still present)
    assert prep.pred_rank["A"].drop_nulls().to_list() == ["g_big"]  # symmetric on pred


def test_prepare_de_floor_zero_is_noop():
    base = prepare_de(_raw_floor_frame(), _raw_floor_frame(), control="ctrl")
    floored0 = prepare_de(_raw_floor_frame(), _raw_floor_frame(), control="ctrl",
                          min_abs_log2fc=0.0)
    assert base.real_df.equals(floored0.real_df)
    assert base.real_rank.equals(floored0.real_rank)
