import math

import polars as pl
import pytest

from cell_eval2.de import prepare_de
from cell_eval2.metrics.direction import (
    de_direction_coverage,
    de_direction_fidelity,
    de_direction_fidelity_raw,
    de_direction_fidelity_yield,
    de_direction_fidelity_yield_raw,
    de_direction_yield,
    de_direction_yield_raw,
)


def _tbl(rows):
    return pl.DataFrame(
        {
            "target": [r[0] for r in rows],
            "feature": [r[1] for r in rows],
            "log2_fold_change": [float(r[2]) for r in rows],
            "p_adj": [float(r[3]) for r in rows],
        },
        schema={
            "target": pl.String,
            "feature": pl.String,
            "log2_fold_change": pl.Float64,
            "p_adj": pl.Float64,
        },
    )


def _resolution(rows, mode):
    """mode='self' (default) -> every target maps to its own label; None -> derive."""
    from cell_eval2.de import TargetResolution

    if mode is None:
        return None
    if mode != "self":
        return mode
    targets = sorted({r[0] for r in rows})
    return TargetResolution({t: t for t in targets}, len(targets))


def _prep(pred_rows, real_rows, *, resolution="self"):
    """Default to a SELF-MAP resolution -- the H1_CGS shape; see the plan's fixture-trap
    note. Without it a fixture whose target label is not also a feature resolves zero
    targets and RAISES at the gate before any metric runs."""
    return prepare_de(
        _tbl(pred_rows),
        _tbl(real_rows),
        control="non-targeting",
        p_adj_threshold=0.05,
        target_resolution=_resolution(real_rows, resolution),
    )


SIG, NS = 0.01, 0.90


def test_ordinary_case_matches_the_closed_forms():
    """N_conf = 4 (B,C,D,E all real-significant), q = 3/4, d = 1/4.
    Pred calls B,C significant: n_pred = 2, k = 1 (B agrees, C does not)."""
    pred = [
        ("A", "B", 1.0, SIG),
        ("A", "C", -1.0, SIG),
        ("A", "D", 1.0, NS),
        ("A", "E", 1.0, NS),
    ]
    real = [
        ("A", "B", 1.0, SIG),
        ("A", "C", 1.0, SIG),
        ("A", "D", 1.0, SIG),
        ("A", "E", -1.0, SIG),
    ]
    p = _prep(pred, real)
    q, d, n_conf, n_pred, k = 0.75, 0.25, 4, 2, 1

    assert de_direction_fidelity_raw(p)["A"] == pytest.approx(k / n_pred)
    assert de_direction_fidelity(p)["A"] == pytest.approx((k / n_pred - q) / d)
    assert de_direction_coverage(p)["A"] == pytest.approx(n_pred / n_conf)
    assert de_direction_yield_raw(p)["A"] == pytest.approx(k / n_conf)
    assert de_direction_yield(p)["A"] == pytest.approx(
        (k - q * n_pred) / (n_conf * d)
    )
    assert de_direction_fidelity_yield_raw(p)["A"] == pytest.approx(
        k / max(n_pred, n_conf)
    )
    assert de_direction_fidelity_yield(p)["A"] == pytest.approx(
        min(1.0, n_pred / n_conf) * ((k / n_pred - q) / d)
    )


def test_fidelity_yield_is_the_capped_coverage_form_not_a_min():
    """Spec 2.3: `min(fidelity, yield)` flips branch once F < 0, handing an
    under-calling model the UNPENALISED F -- rewarding it for calling less."""
    pred = [("A", "B", -1.0, SIG), ("A", "C", 1.0, NS), ("A", "D", 1.0, NS)]
    real = [("A", "B", 1.0, SIG), ("A", "C", 1.0, SIG), ("A", "D", 1.0, SIG)]
    p = _prep(pred, real)
    fid = de_direction_fidelity(p)["A"]
    cov = de_direction_coverage(p)["A"]
    fy = de_direction_fidelity_yield(p)["A"]
    # N_conf = 3 (B,C,D all real-significant and all UP -> q = 1, d = 0.05);
    # n_pred = 1 (only B is pred-significant), k = 0 -> fidelity = (0 - 1)/0.05 = -20,
    # coverage = 1/3, so the capped form gives (1/3)(-20) = -20/3.
    assert fid == pytest.approx(-20.0)
    assert cov == pytest.approx(1 / 3)
    assert fy == pytest.approx(min(1.0, cov) * fid)  # = -20/3
    assert fy == pytest.approx(-20 / 3)
    # The literal min(fidelity, yield) = min(-20, -20/3) = -20: it hands back the
    # UNPENALISED F, i.e. it stops scaling by coverage once F goes negative. spec 2.3
    # writes the capped form as max(F, r*F) in that branch for exactly this reason.
    assert min(fid, cov * fid) == pytest.approx(-20.0)
    assert fy != pytest.approx(min(fid, cov * fid))  # the WRONG form
    assert fy > fid  # capped is the less extreme one


@pytest.mark.parametrize(
    "metric,expected",
    [
        (de_direction_fidelity, 1.0),
        (de_direction_fidelity_yield, 1.0),
        (de_direction_fidelity_raw, math.nan),
        (de_direction_fidelity_yield_raw, math.nan),
        (de_direction_coverage, math.nan),
        (de_direction_yield, math.nan),
        (de_direction_yield_raw, math.nan),
    ],
)
def test_matrix_nconf0_npred0(metric, expected):
    """Spec 5 column 1: the model also called nothing -> agreement."""
    pred = [("A", "B", 1.0, NS)]
    real = [("A", "B", 1.0, NS)]
    got = metric(_prep(pred, real))["A"]
    assert math.isnan(got) if math.isnan(expected) else got == pytest.approx(expected)


@pytest.mark.parametrize(
    "metric,expected",
    [
        (de_direction_fidelity, 0.0),
        (de_direction_fidelity_yield, 0.0),
        (de_direction_fidelity_raw, 1.0),
        (de_direction_fidelity_yield_raw, 1.0),
        (de_direction_coverage, math.nan),
        (de_direction_yield, math.nan),
        (de_direction_yield_raw, math.nan),
    ],
)
def test_matrix_nconf0_npred_positive(metric, expected):
    """Spec 5 column 2: the model claimed a budget that did not exist -> 0.
    The RAW metrics stay measurements and report k/n_pred = 1/1."""
    pred = [("A", "B", 1.0, SIG)]
    real = [("A", "B", 1.0, NS)]
    got = metric(_prep(pred, real))["A"]
    assert math.isnan(got) if math.isnan(expected) else got == pytest.approx(expected)


@pytest.mark.parametrize(
    "metric,expected",
    [
        (de_direction_fidelity, math.nan),
        (de_direction_fidelity_yield, 0.0),
        (de_direction_fidelity_raw, math.nan),
        (de_direction_fidelity_yield_raw, 0.0),
        (de_direction_coverage, 0.0),
        (de_direction_yield, 0.0),
        (de_direction_yield_raw, 0.0),
    ],
)
def test_matrix_nconf_positive_q_defined_npred0(metric, expected):
    """Spec 5 column 3."""
    pred = [("A", "B", 1.0, NS)]
    real = [("A", "B", 1.0, SIG)]
    got = metric(_prep(pred, real))["A"]
    assert math.isnan(got) if math.isnan(expected) else got == pytest.approx(expected)


@pytest.mark.parametrize(
    "metric,expected",
    [
        (de_direction_fidelity, math.nan),
        (de_direction_fidelity_yield, math.nan),
        (de_direction_fidelity_raw, math.nan),
        (de_direction_fidelity_yield_raw, 0.0),
        (de_direction_coverage, 0.0),
        (de_direction_yield, math.nan),
        (de_direction_yield_raw, 0.0),
    ],
)
def test_matrix_nconf_positive_q_undefined_npred0(metric, expected):
    """Spec 5 column 4 -- the cell round 3 found overlapping. q-undefined beats
    n_pred = 0, so fidelity_yield is NaN here and 0.0 in column 3."""
    pred = [("A", "B", 1.0, NS)]
    real = [("A", "B", 0.0, SIG)]  # significant but NO direction -> q undefined
    got = metric(_prep(pred, real))["A"]
    assert math.isnan(got) if math.isnan(expected) else got == pytest.approx(expected)


@pytest.mark.parametrize(
    "metric,expected",
    [
        (de_direction_fidelity, math.nan),
        (de_direction_fidelity_yield, math.nan),
        (de_direction_yield, math.nan),
    ],
)
def test_matrix_nconf_positive_q_undefined_npred_positive_corrected(metric, expected):
    """Spec 5 column 5: every CORRECTED metric is NaN without q."""
    pred = [("A", "B", 1.0, SIG), ("A", "C", 1.0, SIG)]
    real = [("A", "B", 0.0, SIG), ("A", "C", 1.0, NS)]
    got = metric(_prep(pred, real))["A"]
    assert math.isnan(got)


def test_matrix_nconf_positive_q_undefined_npred_positive_raw():
    """Spec 5 column 5, the RAW half -- the four uncorrected metrics stay measurements
    when q is undefined, because none of them reads q.

    The corrected parametrization above pins only that the three corrected metrics are
    NaN, which a bug that NaN-ed the WHOLE column would also satisfy. Same fixture:
    N_conf = 1 (B is reference-significant; C is not), q undefined (B has no direction),
    n_pred = 1 and k = 1 (C is the only pred-significant AND adjudicable gene, and its
    sign agrees), so all four raw metrics are 1.0.
    """
    pred = [("A", "B", 1.0, SIG), ("A", "C", 1.0, SIG)]
    real = [("A", "B", 0.0, SIG), ("A", "C", 1.0, NS)]
    p = _prep(pred, real)
    assert de_direction_fidelity_raw(p)["A"] == pytest.approx(1.0)
    assert de_direction_fidelity_yield_raw(p)["A"] == pytest.approx(1.0)
    assert de_direction_coverage(p)["A"] == pytest.approx(1.0)
    assert de_direction_yield_raw(p)["A"] == pytest.approx(1.0)


def test_nconf0_precedence_is_not_overridden_by_q_undefined():
    """Spec 5 precedence step 1. N_conf = 0 ALWAYS implies q undefined, so an
    unqualified 'q-undefined wins' would turn 1.0 into NaN and delete the agreement
    convention entirely."""
    pred = [("A", "B", 1.0, NS)]
    real = [("A", "B", 1.0, NS)]
    assert de_direction_fidelity(_prep(pred, real))["A"] == pytest.approx(1.0)


def test_every_perturbation_gets_a_row():
    pred = [("A", "B", 1.0, SIG), ("B", "A", 1.0, NS)]
    real = [("A", "B", 1.0, SIG), ("B", "A", 1.0, SIG)]
    p = _prep(pred, real)
    for metric in (
        de_direction_fidelity,
        de_direction_fidelity_raw,
        de_direction_coverage,
        de_direction_yield,
        de_direction_yield_raw,
        de_direction_fidelity_yield,
        de_direction_fidelity_yield_raw,
    ):
        assert set(metric(p)) == {"A", "B"}, metric.__name__


def test_fidelity_raw_equals_v0_5_0_precision_when_no_feature_equals_its_target():
    """Spec 8: isolates the exclusion away, turning spec 7 into a checkable claim."""
    from cell_eval2.metrics.direction import de_direction_precision

    pred = [("A", "B", 1.0, SIG), ("A", "C", -1.0, SIG)]
    real = [("A", "B", 1.0, SIG), ("A", "C", 1.0, SIG)]
    p = _prep(pred, real)
    assert de_direction_fidelity_raw(p)["A"] == pytest.approx(
        de_direction_precision(p)["A"]
    )


def test_target_gene_exclusion_changes_fidelity_raw():
    """The converse: a CONSTRUCTED fixture where removing the target gene moves a
    numerator. Without construction old and new can both be 1.0 and the test passes
    vacuously (spec 8)."""
    from cell_eval2.metrics.direction import de_direction_precision

    pred = [("A", "A", 1.0, SIG), ("A", "B", -1.0, SIG)]
    real = [("A", "A", 1.0, SIG), ("A", "B", 1.0, SIG)]
    p = _prep(pred, real)
    assert de_direction_precision(p)["A"] == pytest.approx(0.5)  # A matches, B does not
    assert de_direction_fidelity_raw(p)["A"] == pytest.approx(0.0)  # A removed
