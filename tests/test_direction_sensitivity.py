import polars as pl
import pytest

from cell_eval2.de import prepare_de


def _prep(real, pred, *, threshold=0.05):
    return prepare_de(pred, real, control="non-targeting", p_adj_threshold=threshold,
                      nan_lfc_policy="keep")


def _one_conf_all_matching():
    """N_conf = 1 (only A is reference-significant) but three shared genes, all with
    agreeing directions -- so the full-universe curve stays pure to k=3."""
    real = pl.DataFrame({
        "target": ["G1"] * 3, "feature": list("ABC"),
        "log2_fold_change": [1.0, 1.0, 1.0], "p_adj": [0.001, 0.5, 0.5],
    })
    pred = pl.DataFrame({
        "target": ["G1"] * 3, "feature": list("ABC"),
        "log2_fold_change": [1.0, 1.0, 1.0], "p_adj": [0.001, 0.5, 0.6],
        "p_value": [0.0005, 0.1, 0.2],
    })
    return _prep(real, pred)


def test_adjudicated_sensitivity_is_bounded_by_one():
    from cell_eval2.metrics.direction import de_direction_sensitivity
    assert de_direction_sensitivity(_one_conf_all_matching(), universe="adjudicated")["G1"] == 1.0


def test_universe_sensitivity_is_unbounded_above():
    from cell_eval2.metrics.direction import de_direction_sensitivity
    # k* = 3 over the whole shared universe, N_conf = 1 -> 3.0. A specified property of
    # this variant (Notion section 1.2), not a bug -- which is why it carries anchor=None
    # (formerly best_value='none'): there is no constant value a perfect submission attains.
    assert de_direction_sensitivity(_one_conf_all_matching(), universe="all")["G1"] == 3.0


def test_n_conf_counts_the_whole_real_table_not_the_join():
    from cell_eval2.metrics.direction import de_direction_sensitivity
    # D is reference-significant but absent from pred, so it can never be ranked. It must
    # still inflate N_conf to 2 -- a model cannot raise its score by omitting genes.
    real = pl.DataFrame({
        "target": ["G1"] * 2, "feature": ["A", "D"],
        "log2_fold_change": [1.0, 1.0], "p_adj": [0.001, 0.001],
    })
    pred = pl.DataFrame({
        "target": ["G1"], "feature": ["A"],
        "log2_fold_change": [1.0], "p_adj": [0.001], "p_value": [0.0005],
    })
    got = de_direction_sensitivity(prepared=None, de_pred=pred, de_real=real,
                                   control="non-targeting", nan_lfc_policy="keep",
                                   universe="adjudicated")
    assert got["G1"] == 0.5      # k* = 1, N_conf = 2


def test_sensitivity_is_zero_when_the_prediction_covers_no_adjudicated_gene():
    from cell_eval2.metrics.direction import de_direction_sensitivity
    # A(t) is empty (the one reference-significant gene is absent from pred) while
    # N_conf = 1. Spec section 5.2 requires a COMPUTED 0.0, not an omission.
    real = pl.DataFrame({
        "target": ["G1"] * 2, "feature": ["A", "D"],
        "log2_fold_change": [1.0, 1.0], "p_adj": [0.5, 0.001],
    })
    pred = pl.DataFrame({
        "target": ["G1"], "feature": ["A"],
        "log2_fold_change": [1.0], "p_adj": [0.5], "p_value": [0.5],
    })
    got = de_direction_sensitivity(prepared=None, de_pred=pred, de_real=real,
                                   control="non-targeting", nan_lfc_policy="keep",
                                   universe="adjudicated")
    assert got == {"G1": 0.0}


def test_sensitivity_is_zero_when_purity_never_reaches_p0():
    from cell_eval2.metrics.direction import de_direction_sensitivity
    real = pl.DataFrame({
        "target": ["G1"] * 2, "feature": list("AB"),
        "log2_fold_change": [1.0, 1.0], "p_adj": [0.001, 0.001],
    })
    pred = pl.DataFrame({
        "target": ["G1"] * 2, "feature": list("AB"),
        "log2_fold_change": [-1.0, -1.0], "p_adj": [0.001, 0.001],
        "p_value": [0.0001, 0.0002],
    })
    assert de_direction_sensitivity(_prep(real, pred), universe="adjudicated")["G1"] == 0.0


def test_p0_is_derived_from_alpha_not_hard_coded():
    """Every other fixture uses alpha=0.05 with purity of only 0 or 1, so a hard-coded
    P0=0.975 would pass them all. Here purity runs 0, .5, 2/3, .75, .8 over five ranked
    genes: at alpha=0.40 (P0=0.80) the deepest qualifying k is 5; at alpha=0.05
    (P0=0.975) nothing qualifies. Real p_adj sits below BOTH thresholds so N_conf and the
    adjudicated universe are identical in the two runs and only P0 moves."""
    from cell_eval2.metrics.direction import de_direction_sensitivity
    real = pl.DataFrame({
        "target": ["G1"] * 5, "feature": list("ABCDE"),
        "log2_fold_change": [1.0] * 5, "p_adj": [0.001] * 5,
    })
    pred = pl.DataFrame({
        "target": ["G1"] * 5, "feature": list("ABCDE"),
        "log2_fold_change": [-1.0, 1.0, 1.0, 1.0, 1.0],   # first ranked gene mismatches
        "p_adj": [0.001, 0.002, 0.003, 0.004, 0.005],
        "p_value": [0.001, 0.002, 0.003, 0.004, 0.005],
    })
    loose = de_direction_sensitivity(_prep(real, pred, threshold=0.40), universe="adjudicated")
    tight = de_direction_sensitivity(_prep(real, pred, threshold=0.05), universe="adjudicated")
    assert loose["G1"] == 1.0     # k* = 5 of N_conf = 5
    assert tight["G1"] == 0.0     # purity peaks at 0.8 < 0.975


def test_sensitivity_omits_a_target_with_no_reference_significant_genes():
    from cell_eval2.metrics.direction import de_direction_sensitivity
    real = pl.DataFrame({
        "target": ["G1"], "feature": ["A"], "log2_fold_change": [1.0], "p_adj": [0.5],
    })
    pred = pl.DataFrame({
        "target": ["G1"], "feature": ["A"], "log2_fold_change": [1.0], "p_adj": [0.001],
        "p_value": [0.0001],
    })
    assert de_direction_sensitivity(_prep(real, pred), universe="adjudicated") == {}


def test_sensitivity_rejects_an_unknown_universe():
    from cell_eval2.metrics.direction import de_direction_sensitivity
    with pytest.raises(ValueError, match="universe"):
        de_direction_sensitivity(_one_conf_all_matching(), universe="everything")


def test_an_unadjudicable_reference_significant_gene_shortens_the_adjudicated_depth():
    """Issue #204: `universe='adjudicated'` filters on reference SIGNIFICANCE, which does
    not imply adjudicability -- so the adjudicated variants CAN move on the right dataset.

    This is the shape the #204 measurement did not contain: across all six reference lines
    every reference-significant gene carried a direction (0 unadjudicable of 326,832 /
    177,289 / 419,959 / 339,572 / 205,840 / 244,205), so `direction_sensitivity` and the
    scored `direction_reach` were unchanged there. That is empirical, not structural, and
    this fixture pins the behaviour when it does not hold: B is reference-significant with
    an exactly-zero log2FC, so it enters N_conf but not the depth.

    Under the pre-#204 row-counting depth both A and B advanced k, giving k* = 2 and 1.0.
    """
    from cell_eval2.metrics.direction import de_direction_sensitivity
    real = pl.DataFrame({
        "target": ["G1"] * 2, "feature": list("AB"),
        "log2_fold_change": [1.0, 0.0],       # B: significant but carries NO direction
        "p_adj": [0.001, 0.001],
    })
    pred = pl.DataFrame({
        "target": ["G1"] * 2, "feature": list("AB"),
        "log2_fold_change": [1.0, 1.0], "p_adj": [0.001, 0.002],
        "p_value": [0.0001, 0.0002],
    })
    prep = _prep(real, pred)
    # N_conf = 2 (significance alone), but only A can be adjudicated -> k* = 1.
    assert de_direction_sensitivity(prep, universe="adjudicated")["G1"] == 0.5
