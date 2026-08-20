"""Issue #348: the prediction-side correction is bounded by the submission's OWN
across-perturbation spread, so a pinned aggregate cannot collect a correction it has not earned.

The measured defect, on the official `-r2` val bundles at the v0.14.0 pin: an arm that predicts
"no effect, emit the control profile" scores `expr_mse_unbiased_capped_norm` 0.0000
`from_baseline` when its 400 cells are an i.i.d. draw and **0.9031** when the same profile is
emitted with its per-(p, g) sums PINNED -- nothing else changed, no flood, no target-gene
manipulation. A live dev-leaderboard submission took +0.1389 of a 0.2295 OVERALL through it.
The mechanism: `pred_tn` is a delete-1 cell jackknife, i.e. WITHIN-set dispersion, and that
equals the submitted pseudobulk's own error only for exchangeable cells.

The fixture below reproduces that pair EXACTLY rather than approximately -- the honest arm's
emission noise is built orthogonal to each perturbation's true delta and zero on its own target
gene, so the two arms' values are algebraically comparable to the last bit. See
`test_the_pinned_arm_beat_the_honest_one_by_exactly_its_claimed_correction`.
"""
import logging

import numpy as np
import pytest

from cell_eval2.metrics.delta import (PRED_CORRECTION_BUDGET_FLAG_RATIO, _across_pert_budget,
                                      _row_weights, mse_unbiased, mse_unbiased_capped)
from cell_eval2.moments import GroupMoments

CONTROL = "ctrl"

#: The correction both arms CLAIM. Below `JK_REAL` so #247's cap never binds and the only thing
#: under test is #348's bound.
C_E = 0.05
JK_REAL = 0.12


def _panel(n_perts=8, n_genes=32, seed=11):
    """Real side only: bulks plus jackknife moments. No cells -- the metric reads neither."""
    rng = np.random.default_rng(seed)
    genes = np.array([f"g{i}" for i in range(n_genes)])
    perts = np.array([str(genes[i]) for i in range(n_perts)] + [CONTROL])
    mu_ctrl = rng.uniform(0.5, 4.0, size=n_genes)
    mu_real = np.tile(mu_ctrl, (n_perts + 1, 1))
    for p in range(n_perts):
        mu_real[p] += rng.normal(0.0, 0.2, size=n_genes)
        mu_real[p, p] -= 3.0                       # the knockdown, and the largest mover
    mu_real[n_perts] = mu_ctrl
    real_mom = GroupMoments(perts=perts, counts=np.full(n_perts + 1, 500.0),
                            sumsq=np.zeros(n_perts + 1), jk=np.full(n_perts + 1, JK_REAL))
    return perts, genes, mu_ctrl, mu_real, real_mom


def _w(n_scored, n_genes, cols):
    """The row weights the metric forms internally (`1/|G_p|`), so a direct helper call compares
    the same quantity the metric does."""
    return _row_weights(n_scored, n_genes, cols)


def _pinned_arm(mu_ctrl, n_rows):
    """"Emit the control profile", with every perturbation's aggregate IDENTICAL.

    Zero across-perturbation spread, a non-zero claimed jackknife: cells that scatter inside
    each group and still land on a bit-identical group total. That is the exploit's signature,
    and it is also what an exactly-reused control cell block looks like -- see the module
    docstring of `tests/test_target_gene_exclusion_172.py`'s anchor test for why the two are
    indistinguishable to any estimator reading one submission.
    """
    return np.tile(mu_ctrl, (n_rows, 1))


def _emitted_arm(mu_ctrl, mu_real, n_perts, seed=3):
    """The same profile, EMITTED: each perturbation carries its own sampling noise.

    Built so the comparison against `_pinned_arm` is exact rather than statistical:

    * the noise is ZERO on each row's own target gene, so `_drop_on_target` removes the same
      quantity from both arms;
    * it is ORTHOGONAL to that row's true delta, so the plug-in distance is exactly
      ``||delta_p||^2 + C_E`` with no cross term;
    * every row's noise has squared norm exactly ``C_E``, which is therefore exactly the
      correction the arm has EARNED and exactly what it declares as its jackknife.
    """
    rng = np.random.default_rng(seed)
    n_genes = mu_ctrl.size
    e = rng.normal(0.0, 1.0, size=(n_perts, n_genes))
    e -= e.mean(axis=0)                            # no common-mode component to argue about
    for p in range(n_perts):
        e[p, p] = 0.0                              # the own-target column carries no noise
        d = mu_real[p] - mu_ctrl
        d = d.copy()
        d[p] = 0.0                                 # ... so orthogonalize on the same subspace
        e[p] -= d * (d @ e[p]) / (d @ d)
        e[p] *= np.sqrt(C_E / (e[p] @ e[p]))       # squared norm exactly C_E
    arm = np.tile(mu_ctrl, (n_perts + 1, 1))
    arm[:n_perts] += e
    return arm


def _pred_mom(perts, jk):
    return GroupMoments(perts=perts, counts=np.full(perts.size, 500.0),
                        sumsq=np.zeros(perts.size), jk=np.full(perts.size, jk))


def _values(perts, arm, pred_mom, mu_real, real_mom, genes, fn=mse_unbiased_capped):
    return fn(pred_bulk=(perts, arm), pred_moments=pred_mom,
              real_bulk=(perts, mu_real), real_moments=real_mom,
              comparator="bulk_lognorm", pert_col="perturbation", control=CONTROL, genes=genes)


def _unbounded(monkeypatch):
    """Disable ONLY #348's bound, leaving #247's cap in place -- the pre-#348 metric.

    Patches the estimator rather than `PRED_VAR_ACROSS_TOL_K`: an infinite TOLERANCE times a
    zero spread is NaN, and a zero spread is exactly the case under test here.
    """
    monkeypatch.setattr("cell_eval2.metrics.delta._across_pert_budget",
                        lambda *_a, **_k: float("inf"))


# --------------------------------------------------------------------------------------------
# the headline pair
# --------------------------------------------------------------------------------------------

def test_the_pinned_arm_beat_the_honest_one_by_exactly_its_claimed_correction(monkeypatch):
    """Both arms predict the SAME profile. One earned its correction, one did not.

    Pre-#348 the pinned arm reads LOWER (better) by exactly `C_E` on every perturbation -- the
    whole of the correction it claims and none of which is in its plug-in distance. That is the
    defect, in one number, with no adversarial content beyond pinning the aggregate.
    """
    perts, genes, mu_ctrl, mu_real, real_mom = _panel()
    n_perts = perts.size - 1
    pinned = _pinned_arm(mu_ctrl, perts.size)
    emitted = _emitted_arm(mu_ctrl, mu_real, n_perts)
    mom = _pred_mom(perts, C_E)

    with monkeypatch.context() as m:
        _unbounded(m)
        old_pinned = _values(perts, pinned, mom, mu_real, real_mom, genes)
        old_emitted = _values(perts, emitted, mom, mu_real, real_mom, genes)
    per_gene = C_E / (genes.size - 1)              # the metric is gene-AVERAGED over G - 1
    for p in old_pinned:
        assert old_pinned[p] < old_emitted[p], f"{p}: the pinned arm did not win pre-#348"
        assert old_emitted[p] - old_pinned[p] == pytest.approx(per_gene, rel=1e-12)

    new_pinned = _values(perts, pinned, mom, mu_real, real_mom, genes)
    new_emitted = _values(perts, emitted, mom, mu_real, real_mom, genes)

    # The honest arm's own budget covers all but the rounding of its claim: the noise block's rows
    # each have squared norm exactly C_E, but CENTRING and the own-target row drop take a little
    # off the sum, and with no tolerance multiplier (any slack is a rebate -- see the module
    # constant) that little is forfeited. Quantified here rather than hidden, and it bounds the
    # honest arm's movement below.
    excl = {k: k for k in range(n_perts)}
    w = _w(n_perts, genes.size, excl)
    slack = 1.0 - (_across_pert_budget(emitted[:n_perts], excl, w) / float((w * C_E).sum()))
    assert 0.0 <= slack < 0.05, f"fixture: unrepresentative budget slack {slack}"

    for p in new_pinned:
        # the pinned arm forfeits exactly the correction it claimed ...
        assert new_pinned[p] - old_pinned[p] == pytest.approx(per_gene, rel=1e-12)
        # ... which lands it, to the last bit, on what an HONEST emission of the same profile is
        # worth. That is the whole point of the fix in one assertion.
        assert new_pinned[p] == pytest.approx(old_emitted[p], rel=1e-12), (
            f"{p}: the pinned arm is not worth what the honest arm it copies is worth"
        )
        # and the honest arm moves by no more than its own forfeited slack
        assert abs(new_emitted[p] - old_emitted[p]) <= slack * per_gene * (1 + 1e-9)


def test_a_pinned_aggregate_forfeits_the_correction_and_nothing_more(monkeypatch):
    """The bound withholds exactly `min(jk_pred, k*jk_real)` and does not touch the distance."""
    perts, genes, mu_ctrl, mu_real, real_mom = _panel()
    pinned = _pinned_arm(mu_ctrl, perts.size)
    mom = _pred_mom(perts, C_E)
    with monkeypatch.context() as m:
        _unbounded(m)
        old = _values(perts, pinned, mom, mu_real, real_mom, genes)
    new = _values(perts, pinned, mom, mu_real, real_mom, genes)
    # zero spread -> zero bound -> the whole claimed correction is withheld, per row
    for p in old:
        assert new[p] - old[p] == pytest.approx(C_E / (genes.size - 1), rel=1e-12)
    # and with no correction left to withhold there is nothing further to lose
    zero_jk = _values(perts, pinned, _pred_mom(perts, 0.0), mu_real, real_mom, genes)
    assert new == zero_jk


# --------------------------------------------------------------------------------------------
# the two ends of the scoring scale, which the fix must not move
# --------------------------------------------------------------------------------------------

def test_the_baseline_leg_cannot_be_reached_by_the_bound(monkeypatch):
    """The published baseline arm's 400 cells per group are IDENTICAL, so its jackknife is
    exactly 0 (measured min 0, max 0 on `A.context_mean.csad`) and there is no correction for
    any bound to withhold. Modelled here as jk_pred = 0 against a zero-spread arm -- the
    worst case for the bound and still bit-identical."""
    perts, genes, mu_ctrl, mu_real, real_mom = _panel()
    arm = _pinned_arm(mu_ctrl, perts.size)
    mom = _pred_mom(perts, 0.0)
    with monkeypatch.context() as m:
        _unbounded(m)
        old = _values(perts, arm, mom, mu_real, real_mom, genes)
    assert _values(perts, arm, mom, mu_real, real_mom, genes) == old


def test_a_replicate_anchor_keeps_its_own_jackknife(monkeypatch):
    """Real data carries real biology across perturbations, so `Var_across_pert` DOMINATES the
    within-group jackknife and the `min` keeps selecting the arm's own honest correction.
    MEASURED on the official val A anchor at its own 200-cell split size: `Var_across_pert`
    46.64 against a median `pred_tn` of 29.34, ratio 1.59 -- a PER-ROW proxy; the quantity the
    bound actually forms is the weighted total, measured 1.54 / 1.38 / 1.58 on val A / B / C over
    the anchor's five derived seeds. Modelled here by predicting the real profiles themselves."""
    perts, genes, mu_ctrl, mu_real, real_mom = _panel()
    arm = mu_real.copy()                            # a perfect submission: full real spread
    mom = _pred_mom(perts, C_E)
    n_perts = perts.size - 1
    excl = {k: k for k in range(n_perts)}
    assert _across_pert_budget(arm[:n_perts], excl, _w(n_perts, genes.size, excl)) > C_E, "fixture: the bound would bind"
    with monkeypatch.context() as m:
        _unbounded(m)
        old = _values(perts, arm, mom, mu_real, real_mom, genes)
    assert _values(perts, arm, mom, mu_real, real_mom, genes) == old


def test_an_arm_whose_budget_covers_its_claim_is_bit_identical(monkeypatch):
    """The non-binding regime, exactly. An arm claiming less than its own across-perturbation
    centred sum of squares keeps every bit of its correction: #348 is a CEILING on the total, not
    a rescaling of everyone. MEASURED, at each arm's own depth: the replicate anchor's budget is
    1.54 / 1.38 / 1.58x its claim on the official val A / B / C panels and the honest
    control-paste's is 0.974x on val A -- the anchor is in this regime, and it is why the published
    ends of the scale do not move."""
    perts, genes, mu_ctrl, mu_real, real_mom = _panel()
    n_perts = perts.size - 1
    emitted = _emitted_arm(mu_ctrl, mu_real, n_perts)
    excl = {k: k for k in range(n_perts)}
    w = _w(n_perts, genes.size, excl)
    budget = _across_pert_budget(emitted[:n_perts], excl, w)
    jk = budget / float(w.sum()) / 2.0             # claim = budget / 2, comfortably inside
    assert jk < JK_REAL, "fixture: #247's cap must not be the thing under test"
    mom = _pred_mom(perts, jk)
    with monkeypatch.context() as m:
        _unbounded(m)
        old = _values(perts, emitted, mom, mu_real, real_mom, genes)
    assert _values(perts, emitted, mom, mu_real, real_mom, genes) == old


def test_injecting_spread_can_never_pay_for_itself():
    """The property that forced the budget form, and the regression that keeps it.

    Injecting centred variation into a scored gene COSTS its squared norm in the plug-in distance
    and BUYS at most the same amount of correction, so the summed numerator can never go down.
    MEASURED on a P=300 panel: a `1.1 * sum_g Var_p(ddof=1)` ceiling rebates **+10.371%** of
    everything injected (matching `t*P/(P-1) - 1` = 0.10368), a `1.0 *` one still rebates +0.325%
    through `P/(P-1)` alone, and this form measures -0.003%. A rebate is a guaranteed-profit
    channel, which is why there is no tolerance factor and why the budget is a centred SUM.

    The injected vector is projected orthogonal to BOTH the all-ones vector and the gene's true
    delta, which is what makes the accounting exact: without the second projection the arm also
    becomes slightly more ACCURATE, the first-order term swamps the effect under test at small
    amplitudes, and the test passes or fails on the draw. (Measured: the same sweep reads
    -1.1e-4 with only the centring, and -2.2e-16 with both.)
    """
    perts, genes, mu_ctrl, mu_real, real_mom = _panel(n_perts=10, n_genes=40)
    n_perts = perts.size - 1
    pinned = _pinned_arm(mu_ctrl, perts.size)

    bound_fired = []

    def sweep(col, jk):
        mom = _pred_mom(perts, jk)
        base = sum(_values(perts, pinned, mom, mu_real, real_mom, genes).values())
        # The rows this gene is scored on: if it is some perturbation's own target, that row is
        # dropped from BOTH the distance and the budget, so the injection is built on the retained
        # rows and leaves the free cell alone (its inertness is
        # `test_the_own_target_gene_cannot_inflate_the_bound`). Projecting over all rows instead
        # would force compensating variation into the free cell and measure accuracy, not rebate.
        rows = [r for r in range(n_perts) if str(genes[col]) != str(perts[r])]
        d = mu_real[rows, col] - mu_ctrl[col]
        basis, _ = np.linalg.qr(np.stack([np.ones(len(rows)), d]).T)
        for amp in (0.02, 0.1, 0.5, 2.0, 8.0):
            w = np.linspace(-1.0, 1.0, len(rows)) ** 3        # deterministic, not symmetric
            w = w - basis @ (basis.T @ w)
            w *= amp / np.sqrt((w ** 2).mean())
            assert abs(w.sum()) < 1e-9 and abs(d @ w) < 1e-9, "fixture: the projection failed"
            arm = pinned.copy()
            arm[rows, col] += w
            got = sum(_values(perts, arm, mom, mu_real, real_mom, genes).values())
            assert got >= base - 1e-12, (
                f"gene {col}, amplitude {amp}: injecting {float((w ** 2).sum()):.6g} of centred "
                f"spread LOWERED the summed numerator by {base - got:.6g} -- that is a rebate, "
                f"i.e. guaranteed profit for manufacturing across-perturbation spread"
            )
            # ... and the deduction actually taken is `min(budget, claim)` in score units. Without
            # this the test also passes with #348 REMOVED, since a constant deduction plus a
            # non-negative injected error is non-decreasing either way (codex review round 3).
            no_corr = sum(_values(perts, arm, _pred_mom(perts, 0.0), mu_real, real_mom,
                                  genes).values())
            row_w = _row_weights(n_perts, genes.size, {k: k for k in range(n_perts)})
            budget = _across_pert_budget(arm[:n_perts],
                                         {k: k for k in range(n_perts)}, row_w)
            claim = float((row_w * min(jk, JK_REAL)).sum())
            want = min(budget, claim)
            if budget < claim:
                bound_fired.append((col, amp))
            assert (no_corr - got) == pytest.approx(want, rel=1e-9), (
                f"gene {col}, amplitude {amp}: the deduction taken was {no_corr - got}, not "
                f"min(budget {budget}, claim {claim}) = {want} -- an unbounded implementation "
                f"would always have deducted {claim}"
            )

    # an ordinary gene (nobody's target) with the claim under #247's cap and at it, and a gene
    # that IS a row's own target -- where the budget drops one row and the distance drops the
    # same one, so the inequality has to survive the mismatch
    for col, jk in ((30, JK_REAL * 0.9), (31, JK_REAL * 5.0),
                    (3, JK_REAL * 0.9), (0, JK_REAL * 5.0)):
        sweep(col, jk)
    # the sweep must actually cross the boundary, or the deduction assertion above is vacuous:
    # at large amplitudes the budget clears the claim and `r` saturates at 1, which is the correct
    # behaviour and also the regime where an unbounded implementation agrees with this one
    assert bound_fired, "no amplitude landed in the BINDING regime; the sweep proves nothing"


# --------------------------------------------------------------------------------------------
# the channel that must stay closed
# --------------------------------------------------------------------------------------------

def test_the_own_target_gene_cannot_inflate_the_bound():
    """#172 removes each row's own target-gene coordinate from the DISTANCE, which makes that
    cell free: anything can go in it at no cost. If the bound's variance saw it, a pinned arm
    would scatter its own target column across perturbations and buy the whole correction back
    for nothing. Both the bound and the returned values must be blind to it."""
    perts, genes, mu_ctrl, mu_real, real_mom = _panel()
    n_perts = perts.size - 1
    pinned = _pinned_arm(mu_ctrl, perts.size)
    mom = _pred_mom(perts, C_E)
    base = _values(perts, pinned, mom, mu_real, real_mom, genes)

    scattered = pinned.copy()
    for p in range(n_perts):
        scattered[p, p] += 50.0 * (1 + p)          # wild, and entirely inside the free column
    excl = {k: k for k in range(n_perts)}
    # the channel is REAL -- unexcluded, that scatter is worth thousands of times the whole
    # correction the bound exists to withhold ...
    assert _across_pert_budget(scattered[:n_perts], {}, _w(n_perts, genes.size, {})) > 1e3 * C_E
    # ... and excluded, it is worth nothing: the pinned arm's own zero, up to rounding
    assert _across_pert_budget(scattered[:n_perts], excl, _w(n_perts, genes.size, excl)) == pytest.approx(
        _across_pert_budget(pinned[:n_perts], excl, _w(n_perts, genes.size, excl)), abs=1e-8), (
        "the free column moved Var_across_pert, so the bound is manipulable"
    )
    # The returned values move only in their last digits, and not because of the bound:
    # `_drop_on_target` corrects an already-summed distance, so a 50.0 term entering the sum and
    # then leaving it costs a few digits to cancellation. That route is #172's, deliberate and
    # documented there; what matters here is that the bound saw nothing.
    got = _values(perts, scattered, mom, mu_real, real_mom, genes)
    for k, v in base.items():
        assert got[k] == pytest.approx(v, rel=1e-9)


def test_across_pert_budget_is_a_leave_the_target_row_out_centred_sum_of_squares():
    """The helper's arithmetic, against the definition. A centred SUM of squares, NOT a variance:
    the sum is what the numerator charges for the same variation, and that identity is what makes
    the budget unprofitable to buy. Covers a gene that is the target of more than one perturbation
    (a guide-level panel) and one that is nobody's."""
    rng = np.random.default_rng(0)
    x = rng.normal(5.0, 1.0, size=(12, 7))
    cols = {0: 3, 5: 3, 7: 1}
    ones = np.ones(x.shape[0])                     # unit weights -> the plain centred SS
    want = 0.0
    for g in range(x.shape[1]):
        rows = [i for i in range(x.shape[0]) if cols.get(i) != g]
        if len(rows) >= 2:
            col = x[rows, g]
            want += float(((col - col.mean()) ** 2).sum())
    assert _across_pert_budget(x, cols, ones) == pytest.approx(want, rel=1e-12)
    assert _across_pert_budget(x, {}, ones) == pytest.approx(
        float(((x - x.mean(axis=0)) ** 2).sum()), rel=1e-12)
    # a WEIGHTED call is the same sum with w_p folded in, per gene
    wts = np.linspace(0.5, 2.0, x.shape[0])
    want_w = 0.0
    for g in range(x.shape[1]):
        rows = [i for i in range(x.shape[0]) if cols.get(i) != g]
        if len(rows) >= 2:
            col, ww = x[rows, g], wts[rows]
            want_w += float((ww * (col - (ww * col).sum() / ww.sum()) ** 2).sum())
    assert _across_pert_budget(x, cols, wts) == pytest.approx(want_w, rel=1e-12)
    # a shift of 1e6 must not eat it: this is a log-normalized pseudobulk, and the naive
    # sum(x^2) - sum(x)^2/n form is what loses here
    assert _across_pert_budget(x + 1e6, cols, ones) == pytest.approx(want, rel=1e-8)
    # one row is no information at all -> a budget of exactly 0, i.e. no correction. NOT a bypass.
    assert _across_pert_budget(x[:1], {}, ones[:1]) == 0.0


# --------------------------------------------------------------------------------------------
# scope: what the bound must NOT touch, and the degenerate shapes
# --------------------------------------------------------------------------------------------

def test_the_uncapped_audit_column_is_untouched(monkeypatch):
    """`expr_mse_unbiased` is the pre-#247 column kept so the capped one is auditable. #348 is
    part of the CAP, so the uncapped value must not move even on the pinned arm."""
    perts, genes, mu_ctrl, mu_real, real_mom = _panel()
    pinned = _pinned_arm(mu_ctrl, perts.size)
    mom = _pred_mom(perts, C_E)
    with monkeypatch.context() as m:
        _unbounded(m)
        old = _values(perts, pinned, mom, mu_real, real_mom, genes, fn=mse_unbiased)
    assert _values(perts, pinned, mom, mu_real, real_mom, genes, fn=mse_unbiased) == old


def test_a_single_scored_perturbation_forfeits_the_correction_rather_than_bypassing(monkeypatch):
    """One perturbation is not a noisy estimate of an across-perturbation spread -- there is none.

    So the budget is exactly 0 and the whole correction is withheld: the conservative direction,
    and NOT a bypass. An earlier draft guarded on `keep.size >= 2` and skipped the bound entirely,
    which handed a one-target panel (or a one-row streaming subset) the pre-#348 formula -- the
    exploitable one (codex review round 2). It must still be a NUMBER, not a NaN, which is what
    `ddof`-style arithmetic at n=1 would produce.
    """
    perts, genes, mu_ctrl, mu_real, real_mom = _panel()
    one = np.array([str(genes[0]), CONTROL])
    arm = np.stack([mu_ctrl, mu_ctrl])
    mom = GroupMoments(perts=one, counts=np.full(2, 500.0), sumsq=np.zeros(2),
                       jk=np.full(2, C_E))
    kw = dict(real_bulk=(perts, mu_real), real_moments=real_mom, comparator="bulk_lognorm",
              pert_col="perturbation", control=CONTROL, genes=genes)
    got = mse_unbiased_capped(pred_bulk=(one, arm), pred_moments=mom, **kw)
    assert set(got) == {str(genes[0])}
    value = next(iter(got.values()))
    assert np.isfinite(value)

    # exactly what an arm with NO correction to claim reads -- i.e. the correction is fully withheld
    zero = mse_unbiased_capped(pred_bulk=(one, arm),
                               pred_moments=GroupMoments(perts=one, counts=np.full(2, 500.0),
                                                         sumsq=np.zeros(2), jk=np.zeros(2)), **kw)
    assert got == zero
    # ... and it is NOT the pre-#348 value, which is what a bypass would have returned
    with monkeypatch.context() as m:
        _unbounded(m)
        assert mse_unbiased_capped(pred_bulk=(one, arm), pred_moments=mom, **kw) != got


def test_the_flag_warns_on_a_pinned_arm_and_stays_quiet_on_an_emission(caplog):
    """The ratio is #348's flagging gate, logged rather than scored: honest arms measure
    0.977-0.996 and the two arms that pin their aggregate 0.006 and 0.000."""
    perts, genes, mu_ctrl, mu_real, real_mom = _panel()
    n_perts = perts.size - 1
    mom = _pred_mom(perts, C_E)

    with caplog.at_level(logging.WARNING, logger="cell_eval2.metrics.delta"):
        _values(perts, _pinned_arm(mu_ctrl, perts.size), mom, mu_real, real_mom, genes)
    assert any("across-perturbation spread" in r.message for r in caplog.records), \
        "the pinned arm did not raise the flag"
    assert str(PRED_CORRECTION_BUDGET_FLAG_RATIO) in caplog.text

    caplog.clear()
    emitted = _emitted_arm(mu_ctrl, mu_real, n_perts)
    excl = {k: k for k in range(n_perts)}
    w = _w(n_perts, genes.size, excl)
    assert (_across_pert_budget(emitted[:n_perts], excl, w) / float((w * C_E).sum())
            > PRED_CORRECTION_BUDGET_FLAG_RATIO)
    with caplog.at_level(logging.WARNING, logger="cell_eval2.metrics.delta"):
        _values(perts, emitted, mom, mu_real, real_mom, genes)
    assert not [r for r in caplog.records if "across-perturbation spread" in r.message], \
        "an honest emission raised the flag"


def test_the_bound_is_covered_by_the_result_cache_lever():
    """A warm pre-#348 entry must not be served in preference to recomputing.

    The result cache keys on (inputs + config) and `cell_eval2_version` is deliberately absent
    from that key, so a pre-#348 run at the SAME version reproduces the key exactly and its now
    known-wrong value would win. `run._ONTARGET_EXCLUSION_SEMANTICS` 1 -> 2 is what retires those
    entries. The digest WIRING is proven in `tests/test_target_gene_exclusion_172.py`
    (`expr_mse_unbiased_capped` moves the key, `expr_mae` does not); what is asserted here is the
    LINK -- that this metric is inside the gated set, so the bump actually reaches it.
    """
    from cell_eval2.run import _ONTARGET_EXCLUSION_SEMANTICS, _ontarget_exclusion_used
    assert _ontarget_exclusion_used(["expr_mse_unbiased_capped"]) is True
    assert _ONTARGET_EXCLUSION_SEMANTICS >= 2, (
        "#348 moves every capped value where the new bound binds, so the counter must be at "
        "least 2; at 1 a warm pre-#348 result is served instead of being recomputed"
    )


def test_a_partial_run_matches_the_whole_panel_when_the_full_bulk_is_supplied():
    """PARTITION-INDEPENDENCE, which is why `pred_bulk_full` exists.

    #348's budget is the one term whose value depends on which OTHER predicted perturbations are
    present, and `scale.py`'s two streaming drivers restrict the pred bulks to the rows a partial
    emits. Handed the unrestricted bulk, every partial forms the SAME ratio, so scoring a panel in
    pieces and scoring it whole agree bit for bit -- and a subset is not merely a noisier estimate,
    since a biologically narrow slice genuinely has less spread (codex review).
    """
    perts, genes, mu_ctrl, mu_real, real_mom = _panel(n_perts=8)
    n_perts = perts.size - 1
    # an arm that BINDS, so the ratio is actually in play: pinned off-target with a real claim
    arm = _pinned_arm(mu_ctrl, perts.size)
    arm[:n_perts] += np.linspace(0.0, 0.4, n_perts)[:, None] * 0.5
    mom = _pred_mom(perts, JK_REAL * 0.9)
    kw = dict(real_bulk=(perts, mu_real), real_moments=real_mom, comparator="bulk_lognorm",
              pert_col="perturbation", control=CONTROL, genes=genes)
    whole = mse_unbiased_capped(pred_bulk=(perts, arm), pred_moments=mom, **kw)

    pieces = {}
    for lo, hi in ((0, 3), (3, 5), (5, n_perts)):
        idx = list(range(lo, hi)) + [n_perts]                 # the control travels with each piece
        part = mse_unbiased_capped(pred_bulk=(perts[idx], arm[idx]), pred_moments=mom,
                                   pred_bulk_full=(perts, arm), **kw)
        pieces.update(part)
    assert pieces == whole, "concatenated partials disagree with the whole-panel run"

    # ... and WITHOUT the full bulk they do disagree, which is what the warning is for
    naive = {}
    for lo, hi in ((0, 3), (3, 5), (5, n_perts)):
        idx = list(range(lo, hi)) + [n_perts]
        naive.update(mse_unbiased_capped(pred_bulk=(perts[idx], arm[idx]), pred_moments=mom, **kw))
    assert naive != whole, "fixture: the budget did not bind, so this proves nothing"


def test_a_partial_pred_panel_says_so(caplog):
    """The bound is the one term whose value depends on which OTHER predicted perturbations are
    present, so a run scoring a SUBSET estimates it from that subset. The real panel is always
    handed over whole, so this is detectable here -- and it must not be silent, because those
    values are then not bit-comparable with a whole-panel run's."""
    perts, genes, mu_ctrl, mu_real, real_mom = _panel()
    half = np.array([str(genes[0]), str(genes[1]), str(genes[2]), CONTROL])
    arm = np.stack([mu_real[0], mu_real[1], mu_real[2], mu_ctrl])
    mom = _pred_mom(half, C_E)
    kw = dict(real_bulk=(perts, mu_real), real_moments=real_mom, comparator="bulk_lognorm",
              pert_col="perturbation", control=CONTROL, genes=genes)

    with caplog.at_level(logging.WARNING, logger="cell_eval2.metrics.delta"):
        got = mse_unbiased_capped(pred_bulk=(half, arm), pred_moments=mom, **kw)
    assert set(got) == {str(genes[0]), str(genes[1]), str(genes[2])}
    assert any("formed from that SUBSET" in r.message for r in caplog.records), \
        "a partial pred panel was scored without saying so"

    # the whole panel says nothing
    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="cell_eval2.metrics.delta"):
        mse_unbiased_capped(pred_bulk=(perts, mu_real.copy()),
                            pred_moments=_pred_mom(perts, C_E), **kw)
    assert not [r for r in caplog.records if "SUBSET" in r.message]


def test_partial_target_resolution_forms_the_ratio_in_SCORE_units():
    """The second rebate the review found, and why budget and claim are weighted by `1/|G_p|`.

    Since #172 the per-row divisor is `G - 1` where the label resolves to a measured gene and `G`
    where it does not, so a panel with PARTIAL resolution has two exchange rates between raw gene
    units and the score units the metric reports. Formed in RAW units, a submission could buy
    budget on its cheap `G`-divisor rows and spend the unlocked correction on its expensive
    `G - 1` rows -- `G/(G-1)` per unit, which is 1.00005 on the official panels' 18,533 genes but
    1.5 on a three-gene one, and the cross-provider review reproduced it on this shape.

    The two candidate ratios differ here, so this pins WHICH one the metric used rather than
    inferring it: the raw-unit ratio would leave a different value in every row.
    """
    n_genes = 4
    genes = np.array([f"g{i}" for i in range(n_genes)])
    # two rows resolve (and SHARE a target gene, the guide-level case), two do not
    # DISTINCT guide labels mapped to the same gene: `real_index` and the result dict are keyed by
    # label, so two rows literally named "g0" would collapse to one and the test would read the same
    # output row twice (codex review round 3). This is the production guide-level route.
    perts = np.array(["g0-1", "g0-2", "no_such_gene_a", "no_such_gene_b", CONTROL])
    rng = np.random.default_rng(2)
    mu_ctrl = rng.uniform(0.5, 4.0, size=n_genes)
    mu_real = np.tile(mu_ctrl, (perts.size, 1)) + rng.normal(0.0, 0.3, size=(perts.size, n_genes))
    mu_real[-1] = mu_ctrl
    real_mom = GroupMoments(perts=perts, counts=np.full(perts.size, 500.0),
                            sumsq=np.zeros(perts.size), jk=np.full(perts.size, JK_REAL))
    jk = JK_REAL * 0.9
    mom = _pred_mom(perts, jk)
    arm = np.tile(mu_ctrl, (perts.size, 1))
    arm[:4] += np.array([[0.03], [-0.01], [0.02], [-0.04]]) * np.ones(n_genes)
    kw = dict(real_bulk=(perts, mu_real), real_moments=real_mom, comparator="bulk_lognorm",
              pert_col="perturbation", control=CONTROL, genes=genes,
              target_gene_map={"g0-1": "g0", "g0-2": "g0"})

    excl = {0: 0, 1: 0}                            # rows 0 and 1 both target g0; rows 2, 3 do not
    w = _row_weights(4, n_genes, excl)
    assert sorted(set(w.tolist())) == [1 / n_genes, 1 / (n_genes - 1)], "fixture: one divisor only"
    ones = np.ones(4)
    r_score = _across_pert_budget(arm[:4], excl, w) / float((w * jk).sum())
    r_raw = _across_pert_budget(arm[:4], excl, ones) / (4 * jk)
    assert r_score < 1.0 and abs(r_score - r_raw) > 1e-6, "fixture: the two ratios do not differ"

    got = mse_unbiased_capped(pred_bulk=(perts, arm), pred_moments=mom, **kw)
    with_zero = mse_unbiased_capped(pred_bulk=(perts, arm),
                                    pred_moments=_pred_mom(perts, 0.0), **kw)
    for k, p in enumerate(perts[:4]):
        p = str(p)
        # value = (distance - r*term)/G_p, and `with_zero` is the same distance with no term at all
        deduction = (with_zero[p] - got[p]) * (1.0 / w[k])
        assert deduction == pytest.approx(r_score * jk, rel=1e-9), (
            f"{p}: the deduction implies r = {deduction / jk}, not the score-unit "
            f"{r_score} (the raw-unit ratio is {r_raw})"
        )
