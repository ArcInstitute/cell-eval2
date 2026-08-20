"""de_lfc_nmae: per-perturbation normalized MAE of log2 fold changes over the
real-significant gate. Issue #208.

Every assertion here is EXACT. The metric is a ratio of two means over one gene
set, so each failure mode below has a closed form -- there is nothing to
approximate and nothing that needs real cells.
"""
import math

import polars as pl
import pytest
from cell_eval2.metrics.de import de_lfc_nmae

# 12 genes so the default min_gate_size=10 gate is comfortably cleared.
_LFC = [3.0, -2.0, 1.5, -1.0, 2.5, -3.5, 0.5, -0.5, 4.0, -4.0, 1.0, -1.5]
_MEAN_ABS = sum(abs(x) for x in _LFC) / len(_LFC)     # 2.0416666666666665


# The on-target row `_real` appends: significant, and carrying a log2FC no prediction below
# comes anywhere near. Issue #172 excludes it from the gate, so every closed form in this file
# is stated over `_LFC` alone -- and a regression that stopped excluding it would move each of
# them by an unmistakable margin rather than by a rounding error.
_ON_TARGET_LFC = 100.0


def _real(lfc=None, p_adj=None, target="A", on_target=True):
    """One-target real DE frame over `g0..g{n-1}`.

    ⚠️ `on_target=True` (the default) appends ONE extra SIGNIFICANT row whose feature IS the
    target. It makes the target RESOLVE against the real feature index -- without it
    `de_lfc_nmae` trips #172's zero-resolve gate -- and it is what every assertion in this file
    relies on being EXCLUDED. `on_target=False` builds the pre-#172 shape, for the tests that
    assert the gate fires.
    """
    lfc = _LFC if lfc is None else lfc
    n = len(lfc)
    features = [f"g{i}" for i in range(n)]
    lfc_col = list(lfc)
    p_adj_col = ([0.001] * n) if p_adj is None else list(p_adj)
    if on_target:
        features = features + [target]
        lfc_col = lfc_col + [_ON_TARGET_LFC]
        p_adj_col = p_adj_col + [0.001]
    return pl.DataFrame({
        "target": [target] * len(features),
        "feature": features,
        "log2_fold_change": lfc_col,
        "p_value": [0.001] * len(features),
        "p_adj": p_adj_col,
    })


def _pred(lfc, target="A", features=None, on_target=True):
    """The prediction. Its on-target row carries a DELIBERATELY WRONG value: the gate is
    real-side, so what protects each closed form is the REAL frame's exclusion, and a
    prediction that is wildly wrong exactly where the reference row is excluded must not be
    charged for it."""
    n = len(lfc)
    feats = [f"g{i}" for i in range(n)] if features is None else list(features)
    lfc_col = list(lfc)
    if on_target:
        feats = feats + [target]
        lfc_col = lfc_col + [-_ON_TARGET_LFC]
    return pl.DataFrame({
        "target": [target] * len(feats),
        "feature": feats,
        "log2_fold_change": lfc_col,
        "p_value": [0.5] * len(feats),
        "p_adj": [0.5] * len(feats),
    })


def _score(pred_df, real_df=None, **kw):
    return de_lfc_nmae(de_pred=pred_df, de_real=real_df if real_df is not None else _real(), **kw)


def test_no_change_prediction_scores_exactly_one():
    """The defining property: the denominator IS the numerator at lfc_pred = 0, so an
    ALL-ZERO predicted-LFC table scores 1.0 on every perturbation the metric RETURNS, by
    construction (an empty gate, one under min_gate_size, or a zero-or-non-finite
    denominator is omitted rather than scored 1.0).

    That is a statement about the LFC TABLE, not about a submission that emits the control:
    under `control_source="real"` the two differ (#286)."""
    assert _score(_pred([0.0] * len(_LFC)))["A"] == 1.0


def test_exact_prediction_scores_exactly_zero():
    assert _score(_pred(list(_LFC)))["A"] == 0.0


@pytest.mark.parametrize("c", [0.0, 0.5, 1.5, 2.0, -1.0])
def test_uniform_scaling_is_linear_in_abs_c_minus_one(c):
    """mean|c*x - x| / mean|x| = |c - 1| exactly, for every c."""
    got = _score(_pred([c * x for x in _LFC]))["A"]
    assert got == pytest.approx(abs(c - 1.0), abs=1e-12)


@pytest.mark.parametrize("delta", [0.25, -0.75, 3.0])
def test_constant_offset_is_abs_delta_over_mean_abs_real(delta):
    """A SYSTEMATICALLY BIASED rather than noisy predictor -- spec section 7.4 names this
    as untested upstream. mean|x + d - x| / mean|x| = |d| / mean|x|, exactly."""
    got = _score(_pred([x + delta for x in _LFC]))["A"]
    assert got == pytest.approx(abs(delta) / _MEAN_ABS, abs=1e-12)


def test_negated_prediction_scores_exactly_two():
    """An anti-predictor: mean|-x - x| / mean|x| = 2. Also pins that the metric is
    unbounded above and that predicting backwards is exactly twice as bad as silence."""
    assert _score(_pred([-x for x in _LFC]))["A"] == pytest.approx(2.0, abs=1e-12)


def test_within_perturbation_gene_permutation_is_finite_and_worse_than_exact():
    """WRONG GENES, right magnitudes -- the other failure mode spec section 7.4 names.
    No closed form; pin that it is finite, positive, and separated from an exact
    prediction, which is what the member has to be able to say."""
    rotated = _LFC[1:] + _LFC[:1]
    got = _score(_pred(rotated))["A"]
    assert math.isfinite(got)
    assert got > 0.0


def test_two_perturbations_are_scored_independently():
    real = pl.concat([_real(target="A"), _real(target="B")])
    pred = pl.concat([_pred([0.0] * len(_LFC), target="A"),
                      _pred(list(_LFC), target="B")])
    got = de_lfc_nmae(de_pred=pred, de_real=real)
    assert got["A"] == 1.0
    assert got["B"] == 0.0


# ---- gate, omission and masking -------------------------------------------------------

def test_gate_excludes_non_significant_real_genes():
    """Only real p_adj < T enters the gate. Make 6 of 12 non-significant and give the
    prediction a wrong value on exactly those: the score must not move."""
    p_adj = [0.001] * 6 + [0.9] * 6
    pred = _pred(_LFC[:6] + [99.0] * 6)
    got = de_lfc_nmae(de_pred=pred, de_real=_real(p_adj=p_adj), min_gate_size=6)
    assert got["A"] == 0.0


def test_gate_below_min_gate_size_omits_the_perturbation():
    p_adj = [0.001] * 9 + [0.9] * 3          # gate of 9
    got = de_lfc_nmae(de_pred=_pred([0.0] * len(_LFC)), de_real=_real(p_adj=p_adj))
    assert "A" not in got


def test_gate_exactly_min_gate_size_is_scored():
    """The boundary is inclusive -- `< min_gate_size` omits, so == is kept."""
    p_adj = [0.001] * 10 + [0.9] * 2         # gate of 10
    got = de_lfc_nmae(de_pred=_pred([0.0] * len(_LFC)), de_real=_real(p_adj=p_adj))
    assert got["A"] == 1.0


def test_empty_gate_omits_the_perturbation():
    got = de_lfc_nmae(de_pred=_pred([0.0] * len(_LFC)),
                      de_real=_real(p_adj=[0.9] * len(_LFC)))
    assert got == {}


def test_null_predicted_lfc_is_filled_with_zero():
    """A gene absent from the prediction is treated as predicted no-change -- the same
    convention de_lfc_spearman uses. Dropping HALF the genes must still score exactly 1.0,
    because filling with 0 makes the numerator equal the denominator on those genes too."""
    pred = _pred([0.0] * 6, features=[f"g{i}" for i in range(6)])
    got = de_lfc_nmae(de_pred=pred, de_real=_real())
    assert got["A"] == 1.0


def test_non_finite_predicted_lfc_is_filled_with_zero_not_masked():
    """A non-finite prediction is treated as no-change, exactly like a null -- so the gene
    STAYS in the gate and contributes |0 - lfc_real|. Deliberately against #208 5.2, whose
    'mask' rule hands the submission control of its own gate size (see the two adversarial
    tests below)."""
    pred_lfc = [float("inf")] + list(_LFC[1:])
    got = de_lfc_nmae(de_pred=_pred(pred_lfc), de_real=_real())["A"]
    expected = abs(_LFC[0]) / len(_LFC) / _MEAN_ABS      # only gene 0 contributes
    assert got == pytest.approx(expected, abs=1e-12)


def test_all_non_finite_prediction_scores_exactly_one_never_zero():
    """THE adversarial case. Under the rejected masking rule this scored 0.0 -- an empty
    numerator over a non-empty denominator, i.e. a PERFECT submission. Filled, a model that
    emits nothing usable scores exactly 1.0, the same as silence."""
    for bad in (float("inf"), float("-inf"), float("nan")):
        got = de_lfc_nmae(de_pred=_pred([bad] * len(_LFC)), de_real=_real())["A"]
        assert got == 1.0, bad


def test_a_submission_cannot_shrink_its_own_gate():
    """The direct guard on the non-gameable-omission claim. Exactly min_gate_size real-gated
    genes and all but one prediction non-finite: the perturbation must still be SCORED, and
    the scored set must be identical to a clean submission's."""
    # 10 gated genes + the on-target row, which #172 excludes -> a gate of exactly 10. The
    # `.head(10)` this used to apply would now drop the on-target row and trip the
    # zero-resolve gate instead of testing what it is here to test.
    real10 = _real(lfc=_LFC[:10], p_adj=[0.001] * 10)
    poisoned = de_lfc_nmae(de_pred=_pred([float("nan")] * 9 + [0.0]),
                           de_real=real10, min_gate_size=10)
    clean = de_lfc_nmae(de_pred=_pred([0.0] * 10),
                        de_real=real10, min_gate_size=10)
    assert set(poisoned) == set(clean) == {"A"}


def test_real_side_non_finite_lfc_leaves_the_gate():
    """Real-side is the OTHER direction and is allowed to change the gate -- it is a
    property of the evaluation data, identical for every submission."""
    real = _real(lfc=[float("inf")] + list(_LFC[1:]))
    got = de_lfc_nmae(de_pred=_pred([0.0] * len(_LFC)), de_real=real, min_gate_size=11)
    assert got["A"] == 1.0        # 11 genes gated, all predicted no-change


def test_zero_denominator_omits_rather_than_returning_inf():
    """Every gated real LFC is 0 -> mean|real| == 0. Must omit, never divide."""
    real = _real(lfc=[0.0] * len(_LFC))
    got = de_lfc_nmae(de_pred=_pred([1.0] * len(_LFC)), de_real=real)
    assert "A" not in got


def test_duplicate_real_feature_raises():
    real = _real()
    dup = pl.concat([real, real.head(1)])
    with pytest.raises(ValueError, match="duplicate"):
        de_lfc_nmae(de_pred=_pred(list(_LFC)), de_real=dup)


def test_duplicate_pred_feature_raises():
    pred = _pred(list(_LFC))
    dup = pl.concat([pred, pred.head(1)])
    with pytest.raises(ValueError, match="duplicate"):
        de_lfc_nmae(de_pred=dup, de_real=_real())


def test_min_gate_size_below_one_raises():
    with pytest.raises(ValueError, match="min_gate_size"):
        de_lfc_nmae(de_pred=_pred(list(_LFC)), de_real=_real(), min_gate_size=0)


def test_omission_is_reported(caplog):
    """Two perturbations, ONE omitted and ONE scored, so the assertion cannot be satisfied
    by everything being dropped -- and the message must name the count."""
    real = pl.concat([_real(p_adj=[0.001] * 9 + [0.9] * 3, target="A"),   # gate 9 -> omitted
                      _real(target="B")])                                 # gate 12 -> scored
    pred = pl.concat([_pred([0.0] * len(_LFC), target="A"),
                      _pred([0.0] * len(_LFC), target="B")])
    with caplog.at_level("WARNING", logger="cell_eval2.metrics.de"):
        got = de_lfc_nmae(de_pred=pred, de_real=real)
    assert set(got) == {"B"}
    rendered = " ".join(r.getMessage() for r in caplog.records)
    assert "omitted 1 perturbation" in rendered


def test_empty_gate_omission_is_reported_by_reason(caplog):
    """The empty-gate case is the one a count taken from the group_by output CANNOT see --
    a target with nothing significant produces no group row at all. Two perturbations, one
    with an empty gate and one scored, so the assertion cannot pass by everything being
    omitted, and the reason must be named."""
    real = pl.concat([_real(p_adj=[0.9] * len(_LFC), target="A"),   # nothing significant
                      _real(target="B")])
    pred = pl.concat([_pred([0.0] * len(_LFC), target="A"),
                      _pred([0.0] * len(_LFC), target="B")])
    with caplog.at_level("WARNING", logger="cell_eval2.metrics.de"):
        got = de_lfc_nmae(de_pred=pred, de_real=real)
    assert set(got) == {"B"}
    rendered = " ".join(r.getMessage() for r in caplog.records)
    assert "an empty gate" in rendered and "omitted 1 perturbation" in rendered


def test_non_finite_substitution_is_reported(caplog):
    with caplog.at_level("WARNING", logger="cell_eval2.metrics.de"):
        de_lfc_nmae(de_pred=_pred([float("inf")] + list(_LFC[1:])), de_real=_real())
    rendered = " ".join(r.getMessage() for r in caplog.records)
    assert "non-finite" in rendered
