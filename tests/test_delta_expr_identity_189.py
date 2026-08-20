"""#189: under the v2 default `control_source="real"`, delta_mae and expr_mae are the same
quantity, as are delta_mse and expr_mse -- algebraically, and equal to roundoff per perturbation.

Docs-only was the ruling (`docs/metrics.md` §2.4): the metrics are not wrong, and removing them
would lose the v1 variant, which is not redundant. But the claim in the document is a numerical
one, so it is pinned here rather than left as prose. If `_delta_eval`'s control handling ever
changes, this goes red and §2.4 has to be rewritten with it.

"Equal", not "bit-identical". `(p - c) - (r - c)` and `p - r` are different evaluation orders. On
ONE measured sweep -- 600 (seed, perturbation) pairs of lognormal bulks -- 91 differed, by 1-2 ULP;
read that as evidence the difference is real and small on ordinary data, not as a bound.

⚠️ Five successive attempts to assert a TIGHT bound here were all wrong, each in a different way,
and each passed its own suite. They are enumerated in the docstring of
`test_delta_error_equals_the_expr_error_TO_ROUNDOFF` below, along with why the sixth version stops
trying: the claim this file exists to pin is "same quantity", and every defect that would falsify
it moves the value by O(1). A loose relative tolerance plus a separate test showing that a REAL
defect is gross is a stronger construction than a tight bound nobody can get right.

A new file rather than an addition to an existing one: `metrics/delta.py`'s error metrics have no
dedicated unit-test file today, and the shared files that touch them (`test_catalog.py`,
`test_run.py`, `test_bulk_lognorm.py`) are about other things.
"""

import anndata as ad
import numpy as np
import pandas as pd
import pytest

from cell_eval2.metrics.delta import mae, mae_delta, mse, mse_delta

PERTS = ("A", "B", "C")


def _pair(seed=0, G=200, n=30):
    """A real and a predicted AnnData whose CONTROLS also differ -- otherwise the identity
    would hold trivially and the test would not discriminate."""
    rng = np.random.default_rng(seed)
    obs, rows = [], []
    for p in ("non-targeting", *PERTS):
        for _ in range(n):
            obs.append(p)
            rows.append(rng.lognormal(0.0, 1.0, G))
    X = np.array(rows, dtype=np.float64)
    frame = pd.DataFrame({"target": obs}, index=[f"c{i}" for i in range(len(obs))])
    var = pd.DataFrame(index=[f"g{i}" for i in range(G)])
    real = ad.AnnData(X=X.copy(), obs=frame.copy(), var=var.copy())
    pred = ad.AnnData(X=X * 1.3 + 0.2, obs=frame.copy(), var=var.copy())
    return pred, real


@pytest.mark.parametrize("plain,delta", [(mae, mae_delta), (mse, mse_delta)])
def test_delta_error_equals_the_expr_error_TO_ROUNDOFF(plain, delta):
    """The control cancels in any difference-based error: (p - c) - (r - c) = p - r.

    ⚠️ **This deliberately asserts a RELATIVE tolerance, not a tight ULP bound.** Five versions of
    a tight bound were written and all five were wrong, each in a different way:

      1. `==` -- held only because `_pair(seed=0)` happens to cancel exactly;
      2. 8 ULP RELATIVE TO THE RESULT -- unbounded near zero, and never exercised on that fixture;
      3. a forward-error bound POOLED across perturbations;
      4. the same bound UNDERFLOWING to 0.0 on subnormal operands, and WRAPPING on integer ones;
      5. the same bound omitting the rounding of the two `np.mean` reductions themselves --
         counterexample at G=72 gives a gap of 2.0817e-16 against a bound of 1.9737e-16.

    Each patch was correct about the term it added and wrong about a term it still omitted, which
    is the signal to stop: a *complete* forward-error analysis of two nested reductions is not what
    this test is for. What it exists to pin is the claim in `docs/metrics.md` §2.4 -- that these
    are the same quantity -- and any defect that would falsify that claim (a dropped subtraction, a
    control read from the wrong side, a changed `control_source` default) moves the value by O(1),
    not by tens of ULP. A 1e-12 relative tolerance with an absolute floor catches all of those and
    cannot be broken by a reassociation. Non-bit-identity is pinned separately, by a deterministic
    literal, in the test below.

    The ABSOLUTE floor carries the near-zero case and has to be calibrated to the domain, not set
    to something symbolic. `abs=1e-300` did nothing: valid one-gene bulks
    `p = nextafter(0.1), r = 0.1, c = 0.25` give MAE 1.39e-17 against 2.78e-17 -- a factor of two
    apart, pure reassociation, and a 1e-12 RELATIVE tolerance rejects it (codex review round 6,
    refuting the "cannot be broken by a reassociation" claim this docstring used to make). These
    are expression errors in log space, ordinarily 1e-3..1e1; anything under 1e-12 absolute is
    indistinguishable from zero at that scale, while a real defect measures ~0.8 RELATIVE, nine
    orders above the relative tolerance. Loose against noise, tight against defects.
    """
    for seed in range(40):
        r = np.random.default_rng(seed)
        perts = np.array(["ctrl", "A", "B"])
        pm, rm = r.lognormal(0, 1, (3, 40)), r.lognormal(0, 1, (3, 40))
        a = plain(pred_bulk=(perts, pm), real_bulk=(perts, rm), control="ctrl")
        d = delta(pred_bulk=(perts, pm), real_bulk=(perts, rm), control="ctrl",
                  control_source="real")
        for p in ("A", "B"):
            assert a[p] == pytest.approx(d[p], rel=1e-9, abs=1e-12), (
                f"seed {seed} {p}: {a[p]!r} vs {d[p]!r} -- far too far apart to be reassociation")


def test_a_REAL_defect_would_be_caught_by_that_tolerance():
    """The other half of choosing a loose bound: show it still discriminates.

    `control_source="pred"` is the v1 behaviour and the nearest plausible defect -- one side
    subtracting the wrong control. It moves the value by O(1), i.e. ~12 orders outside the 1e-12
    tolerance above, so nothing is lost by not chasing a 10-ULP bound."""
    pred, real = _pair()
    plain = mae(pred, real, pert_col="target")
    wrong = mae_delta(pred, real, pert_col="target", control_source="pred")
    for p in PERTS:
        rel = abs(plain[p] - wrong[p]) / abs(plain[p])
        assert rel > 1e-3, f"{p}: a wrong-control defect must be gross, not subtle (rel={rel:.3g})"


@pytest.mark.parametrize("plain,delta", [(mse, mse_delta), (mae, mae_delta)])
def test_the_identity_is_NOT_bit_exact_and_this_pins_that(plain, delta):
    """The counterexample, as a ONE-GENE deterministic literal.

    One gene deliberately: with more, the two results are a SUM over genes, and a legitimate
    NumPy reduction reassociation can make them coincide -- the three-gene literal this replaces
    became bit-equal under the summation order `(g0 + g2) + g1` (codex review round 3). With a
    single gene there is no reduction, so the difference is a property of the arithmetic and not
    of the reducer.

    p = 8.388, r = 2.576, c = 1.853:  (p - r) = 5.811999999999999 while
    (p - c) - (r - c) = 5.812. Ordinary values; the gap is the last ulp."""
    perts = np.array(["ctrl", "A"])
    pm = np.array([[1.853], [8.388]])
    rm = np.array([[1.853], [2.576]])
    a = plain(pred_bulk=(perts, pm), real_bulk=(perts, rm), control="ctrl")["A"]
    d = delta(pred_bulk=(perts, pm), real_bulk=(perts, rm), control="ctrl",
              control_source="real")["A"]
    assert a == pytest.approx(d, rel=1e-12), "they must still agree to roundoff"
    assert a != d, (
        "if these ever become bit-equal the evaluation order changed and docs/metrics.md 2.4 "
        f"should be revisited: {a!r} vs {d!r}")


@pytest.mark.parametrize("plain,delta", [(mae, mae_delta), (mse, mse_delta)])
def test_the_v1_variant_is_NOT_redundant(plain, delta):
    """The other half of the ruling, and the reason the catalog entries stay. Under v1
    (`control_source="pred"`) the pair genuinely differs -- which is what "docs-only, do not
    remove" rests on. The fixture's predicted control differs from the real one, so the offset
    is nonzero."""
    pred, real = _pair()
    got_plain = plain(pred, real, pert_col="target")
    got_v1 = delta(pred, real, pert_col="target", control_source="pred")
    assert max(abs(got_plain[p] - got_v1[p]) for p in PERTS) > 1e-6


def test_the_v1_difference_is_a_CONSTANT_offset_not_a_second_signal():
    """'Not redundant' is still weaker than 'independent'. Under v1 the delta error differs from
    the expression error by the fixed vector (pred_ctrl - real_ctrl), the same for every
    perturbation -- so the pair carries one perturbation-varying signal plus an offset, exactly
    as §2.4 says. Shown on MSE, where the identity is algebraic:

        delta_mse_p(v1) = mse_p - (2/G) <d_p, k> + (1/G)||k||^2,   k = pred_ctrl - real_ctrl

    so `delta_mse_p(v1) - mse_p + (2/G)<d_p, k>` is the SAME constant for every p.
    """
    pred, real = _pair()
    from cell_eval2.prep import pseudobulk

    pred_perts, pred_means = pseudobulk(pred, "target")
    real_perts, real_means = pseudobulk(real, "target")
    idx_p = {str(q): i for i, q in enumerate(pred_perts)}
    idx_r = {str(q): i for i, q in enumerate(real_perts)}
    k = pred_means[idx_p["non-targeting"]] - real_means[idx_r["non-targeting"]]
    G = pred_means.shape[1]

    plain_mse = mse(pred, real, pert_col="target")
    v1_mse = mse_delta(pred, real, pert_col="target", control_source="pred")
    residuals = []
    for p in PERTS:
        d = pred_means[idx_p[p]] - real_means[idx_r[p]]
        residuals.append(v1_mse[p] - plain_mse[p] + (2.0 / G) * float(d @ k))
    assert np.allclose(residuals, residuals[0], rtol=1e-9, atol=0)
    assert residuals[0] == pytest.approx(float(k @ k) / G, rel=1e-9)


@pytest.mark.parametrize("plain,delta", [(mae, mae_delta), (mse, mse_delta)])
def test_a_NEAR_ZERO_cancellation_is_inside_the_tolerance(plain, delta):
    """The counterexample that killed `rel=1e-12, abs=1e-300` (codex review round 6).

    `p = nextafter(0.1), r = 0.1, c = 0.25` is a valid one-gene bulk whose two evaluation orders
    give 1.39e-17 and 2.78e-17 -- a factor of TWO apart in relative terms, and entirely
    reassociation. A relative-only tolerance rejects it; the absolute floor is what carries it,
    which is why that floor has to sit at the metric's own scale rather than at a symbolic
    1e-300."""
    perts = np.array(["ctrl", "A"])
    pred_m = np.array([[0.25], [np.nextafter(0.1, np.inf)]])
    real_m = np.array([[0.25], [0.1]])
    a = plain(pred_bulk=(perts, pred_m), real_bulk=(perts, real_m), control="ctrl")["A"]
    d = delta(pred_bulk=(perts, pred_m), real_bulk=(perts, real_m), control="ctrl",
              control_source="real")["A"]
    assert a != d, "fixture must still exercise the reassociation it was built for"
    assert abs(a - d) / max(abs(a), abs(d)) > 1e-12, (
        "and the gap must still be large in RELATIVE terms -- that is the whole point")
    assert a == pytest.approx(d, rel=1e-9, abs=1e-12)
