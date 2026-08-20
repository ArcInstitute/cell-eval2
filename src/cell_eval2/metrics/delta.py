"""Expression-error and delta metrics in the run's declared expression space.

For v2 counts/counts runs, **every** metric here uses the resolved ``bulk_lognorm``
comparator (issue #264; PR1 moved ``pds_*``/``delta_*``/``expr_real_mass_ratio``, PR2 the
six remaining ``expr_*``, so all 13 anndata metrics are on it and none is catalogued on
``lognorm``)::

    log1p(bulk_target_sum * group_gene_count_sum / group_total_count)

When either side is already log-normalized, and for every v1 run, they fall back together to
the legacy per-cell space ``mean_c log1p(target_sum * C_cg / L_c)`` -- one run-level
decision, taken by ``norm.resolve_comparator`` and stamped in the run metadata, never per
metric and never per side.

⚠️ The sampling correction follows the comparator: ``moments.correction_for`` returns the
analytic ``tr Sigma-hat/n`` under ``lognorm`` and the delete-1 jackknife under
``bulk_lognorm``. The formulas in the docstrings below are written with the analytic term
because that is where their measured history was taken; read ``C`` for it under the
group-sum comparator, where it is a jackknife with its own measured upward bias (#268).
"""

from __future__ import annotations

import logging

import anndata as ad
import numpy as np

from ..distances import resolve_exclusion_columns
from ..moments import correction_for, unbiased_sq_dist
from ..prep import pseudobulk
from ..safety import safe_mae, safe_mse, safe_pearson

logger = logging.getLogger(__name__)

#: `expr_mse_unbiased_capped` subtracts at most
#: ``PRED_TRACE_CAP_K * tr Sigma_hat_real/n_real`` on the
#: prediction's side (issue #247). k=1 is Alex's call, 2026-08-06, and it reads as a policy
#: rather than a tuning constant: **a submission may never claim a larger sampling correction
#: than the reference itself earns.** Do not treat this as a knob to loosen -- raising it
#: reopens exactly the headroom the cap exists to close, and every value above ~1.07 was
#: measured to be indistinguishable from no cap at all on honest replicates.
PRED_TRACE_CAP_K = 1.0

#: `expr_mse_unbiased_capped` bounds the prediction's TOTAL sampling correction by the
#: submission's own across-perturbation centred sum of squares (issue #348, the measured half of
#: #294), allocated across rows in proportion to what each row claims. Below this ratio of budget
#: to claim the run WARNS -- it is log-only and changes no number.
#:
#: ⚠️ There is deliberately NO tolerance multiplier on the budget, and this constant is not one.
#: Any slack above 1.0 is a REBATE, not a safety margin: injecting centred variation `z` into an
#: ordinary scored gene costs `||z||^2` in the plug-in distance and raises the budget by exactly
#: `||z||^2`, so a multiplier `t` pays the submitter back `t - 1` per unit injected. MEASURED on a
#: P=300 synthetic panel with the cross term projected out: a scalar `1.1 * Var_across_pert`
#: (ddof=1) form rebates **+10.37%** of everything injected, matching `t*P/(P-1) - 1` to four
#: decimals, and even `t = 1.0` with `ddof=1` rebates +0.33% through the `P/(P-1)` factor alone.
#: The budget-and-proportional-allocation form below measures **-0.003%**, i.e. exactly
#: break-even. (Codex review, 2026-08-19; `test_injecting_spread_can_never_pay_for_itself`.)
#:
#: The price of having no slack is that an honest arm whose budget lands just under its claim
#: forfeits the difference: MEASURED, official val A, the honest control-paste arm has a budget
#: 0.974x its claim and so pays 2.6% of the member's range. That is accepted rather than tuned
#: away, because tuning it away is the rebate above -- and it costs no reachable score, since such
#: an arm already reads at or above the baseline end of the scale.
PRED_CORRECTION_BUDGET_FLAG_RATIO = 0.7


def _exclusion_cols(labels, genes, *, target_gene_map, gate_labels, who, n_genes):
    """``{row index -> that row's own gene column}``, or ``{}`` when ``genes`` is unavailable.

    Issue #172: the two legs of ``expr_mse_unbiased_capped_norm`` drop each perturbation's own
    gene from their gene sum, like ``pds_*`` has since v1 and the eleven chance-corrected DE
    metrics since #195. Resolution goes through :func:`distances.resolve_exclusion_columns`, the
    ONE definition shared with both discrimination kernels, so a run has one notion of "the
    target gene" -- and so a zero-resolve panel RAISES there instead of excluding nothing and
    returning a plausible wrong number (#248).

    ⚠️ ``genes=None`` is the one route that silently excludes nothing, so it WARNS. Every
    production driver passes the var index (``run._run_metrics`` builds it from
    ``pred_ad.var.index`` and hands it to ``dispatch_anndata_metrics``, which the streaming and
    partitioned drivers call too), so this fires only for a direct call -- a test, or a caller
    holding bulks without their gene labels. It is a warning rather than a raise because the
    argument is genuinely optional in the signature and a direct caller may not have it.
    """
    if genes is None:
        logger.warning(
            "%s: no gene labels supplied (genes=None), so the perturbed gene's own column is "
            "NOT excluded and the value keeps its pre-#172 meaning. Every production driver "
            "passes the var index; a direct caller must pass `genes=` to get the scored "
            "definition.", who,
        )
        return {}
    # ⚠️ NO duplicate-gene check here, deliberately, and this is the one place a reader will
    # expect one: a duplicated gene label WAS the bug Copilot found on PR #316 (the exclusion
    # dropped the LAST column with that label instead of the target's -- 34.0 where 1.0 was
    # correct, silently). The check lives in `resolve_exclusion_columns` instead, which is the
    # first thing the call below does. Keeping a copy here as well was measured and dropped
    # (Gemini, PR #316): it cost 8.6 ms at G = 20000 -- 48x the saving this file's
    # `_drop_on_target` comment declines a vectorization over -- while the resolver's check is
    # free, because it compares the `gene_pos` dict it has to build anyway. The resolver's
    # message names this metric through `who` below, so nothing is lost by not repeating it.
    #
    # The OTHER half of `resolve_exclusion_columns`' input contract -- "unique, and matching the
    # feature dimension". The resolver backstops uniqueness; it cannot check the dimension,
    # because it never sees the means. Unchecked, a short `genes` makes the gene->column mapping
    # cover only a PREFIX of the coordinates: a target whose own gene sits past the end of
    # `genes` resolves to nothing, the global gate still passes on the targets that did resolve,
    # and that row silently keeps its own target coordinate in the sum -- a warned-but-wrong
    # value rather than an error (codex round 3). Production drivers derive `genes` and the means
    # from the same axis so they cannot trip it; a direct caller can.
    if len(genes) != n_genes:
        raise ValueError(
            f"{who}: `genes` has {len(genes)} label(s) but the aligned means carry {n_genes} "
            "coordinate(s). Excluding each perturbation's own gene (issue #172) needs a label "
            "per column: a mismatch maps only a prefix of the coordinates, so a target whose "
            "gene lies outside the labelled range would silently keep its own gene in the sum. "
            "Pass the var index that these means were built from."
        )
    return resolve_exclusion_columns(
        labels, genes, target_gene_map=target_gene_map, gate_labels=gate_labels,
        # These two keep the zero-resolve error honest for a family that has no
        # `exclude_target_gene` knob: exclusion is part of what the metric IS here, so there is
        # nothing to unset and the only remedy is the map.
        who=f"{who} excludes each perturbation's own gene (issue #172)",
        escape=("which is the only way to score this panel: exclusion is part of the metric "
                "and cannot be switched off."),
    )


def _row_weights(n_scored, n_genes, cols):
    """``1 / |G_p|`` per scored row: the divisor `_per_gene` will apply to that row.

    #348's budget and claim are compared to each other, and the quantity the metric actually sums
    is ``(distance_p - correction_p) / |G_p|``. Since #172 that divisor is ``G - 1`` on a row whose
    target resolves and ``G`` on one whose does not, so a panel with PARTIAL resolution has two
    different exchange rates between raw gene units and score units. Forming the comparison in raw
    units would then let a submission buy budget on its cheap (``G``) rows and spend the unlocked
    correction on its expensive (``G - 1``) rows -- a rebate of up to ``G/(G-1)`` per unit, which is
    1.00005 on the official panels' 18,533 genes but 1.5 on a three-gene one, and it was
    reproduced on a guide-level construction (codex review round 2). Weighting both sides by the
    row's own divisor removes the exchange rate entirely.

    ⚠️ A row whose scored gene set is EMPTY (``G == 1`` and that gene is its own target) has
    ``n_row == 0`` and gets weight 0: `_per_gene` gives it NaN, so it must contribute to neither
    side. That is why the ``where=`` guard is not removable -- ``n_row`` is NOT guaranteed positive
    (Gemini's suggestion on PR #353 assumed it was, while keeping the guard; the guard is what
    makes it safe).
    """
    n_row = np.full(int(n_scored), float(n_genes), dtype=np.float64)
    for k in cols:
        n_row[k] = n_genes - 1
    # scalar numerator: `np.ones_like(n_row)` would allocate a second array for nothing (Gemini,
    # PR #353)
    return np.divide(1.0, n_row, out=np.zeros_like(n_row), where=n_row > 0)


def _across_pert_budget(pred_sel, cols, weights):
    """``sum_g sum_p w_p (mu_pred[p, g] - wmean_p mu_pred[., g])^2`` over the SCORED rows, each
    row's own target gene left out: the TOTAL sampling correction this submission is entitled to,
    in the SCORE units `_per_gene` will report (see `_row_weights`).

    Why this quantity (issue #348). Writing ``mu_hat_p = mu_p_true + eps_p`` with ``eps``
    independent across perturbations, the per-gene term over the retained set ``R_g`` has
    expectation

        E[B_g] = B_g(mu_true) + sum_{p in R_g} w_p s2_pg
                              - (sum_{p in R_g} w_p^2 s2_pg) / (sum_{p in R_g} w_p)

    with ``s2_pg = Var(mu_hat[p, g])``. Biological signal can only ADD, so it tracks the correction
    the submission is owed, derived from the submission itself, with no model of its emission and
    nothing a cell layout can inflate.

    ⚠️ **It is a deliberately CONSERVATIVE budget, NOT an unbiased upper bound, and the two cannot
    both be had.** The third term above is the shortfall: at equal weights it collapses to
    ``(1/n_g) * sum_p w_p s2_pg``, i.e. the same degree-of-freedom factor whose "correction"
    (dividing by ``n_g - 1``, turning the sum into a variance) is exactly the rebate below --
    restoring it hands a submitter ``n_g/(n_g - 1)`` per unit of manufactured spread. So a flat
    homoscedastic panel is budgeted ``1/n_g`` under what it is strictly owed -- 0.33% at the
    competition's 300 perturbations -- and that shortfall is accepted, because break-even against
    manufactured spread is the property worth having. Earlier drafts of this docstring claimed the
    strict inequality, then claimed the equal-weight factor as exact; both were wrong (codex review
    rounds 2 and 3).

    ⚠️ **A centred SUM of squares, not a variance, and no multiplier -- this is what makes the
    bound unprofitable to attack.** It is exactly the quantity the numerator CHARGES for
    across-perturbation variation: injecting centred variation into an ordinary scored gene raises
    the plug-in distance and this budget by the same amount, so buying budget is break-even at
    best. Any per-row variance form (``var(ddof=1)`` summed over genes, times P rows) overpays by
    ``P/(P-1)``, and any tolerance multiplier overpays by itself -- see
    ``PRED_CORRECTION_BUDGET_FLAG_RATIO`` for the measured rebate rates.

    ⚠️ **The exclusion is load-bearing, and leaving it out reopens #172 into this bound.**
    ``_drop_on_target`` removes each row's own target-gene coordinate from the DISTANCE, so that
    coordinate is free: a submitter can put anything in cell ``(p, g_p)`` and pay nothing. If this
    budget saw it, a pinned submission could scatter its own target column across perturbations to
    buy back the whole correction the bound exists to withhold, at a cost of exactly zero. Dropping
    the same rows keeps budget and charge on the same coordinates, so the inequality stays in the
    safe direction. It costs an honest submission one gene's share of the sum -- a median
    0.0055-0.0083% by the measurement ``_drop_on_target`` quotes.
    (``tests/test_target_gene_exclusion_172.py`` fails without it: the #172 adversary arm and the
    plain predict-the-control arm stop being the same submission.)

    Computed in the shifted two-pass form, because a log-normalized pseudobulk always has means
    large against its spread and the naive ``sum(x^2) - sum(x)^2/n`` loses digits there. ``cnt``
    is per gene because a gene is the target of at most one perturbation on a gene-level panel but
    of several on a guide-level one, and a gene left with fewer than two retained rows carries no
    estimate at all rather than a divide error -- so a one-perturbation panel yields a budget of
    exactly 0, i.e. no correction, which is the conservative direction and not a bypass.
    """
    n_rows, n_genes = pred_sel.shape
    w = np.asarray(weights, dtype=np.float64)
    dev = pred_sel - pred_sel.mean(axis=0)
    s1 = np.einsum("ij,i->j", dev, w)
    s2 = np.einsum("ij,ij,i->j", dev, dev, w)
    wsum = np.full(n_genes, float(w.sum()), dtype=np.float64)
    cnt = np.full(n_genes, float(n_rows), dtype=np.float64)
    for k, col in cols.items():
        v = dev[k, col]
        s1[col] -= w[k] * v
        s2[col] -= w[k] * v * v
        wsum[col] -= w[k]
        cnt[col] -= 1.0
    ok = (cnt >= 2.0) & (wsum > 0.0)
    if not ok.any():
        return 0.0
    # sum_p w_p (x - wmean_retained)^2 = s2 - s1^2/wsum, with the shift cancelling exactly.
    ss_g = s2[ok] - s1[ok] ** 2 / wsum[ok]
    # A rounding-negative gene is a zero, not a credit against another gene's.
    return float(np.maximum(ss_g, 0.0).sum())


def _whole_panel_budget(pred_bulk_full, pred_moments, real_tn, real_index, *, control,
                        comparator, genes, target_gene_map, who, n_genes, n_scored):
    """``(budget, claim)`` over the WHOLE pred panel, or ``None`` when it adds nothing.

    #348's budget is the one term in this metric whose value depends on which OTHER predicted
    perturbations are present. ``scale.py``'s two streaming drivers restrict the pred bulks to the
    perturbations a partial run emits (``_restrict``, so ``expr_distance_unbiased`` does not emit
    the whole panel into every partial and collide in ``partition.aggregate_partials``), which
    would make this member's value depend on the partitioning. They therefore also hand over the
    UNRESTRICTED bulk, and the ratio is formed from it: identical for every partial of a panel, so
    concatenated fractions match a whole-panel run exactly.

    Returns ``None`` when the full panel scores no more rows than this call does -- the
    whole-panel case, where recomputing would only repeat the work, and the genuinely
    partial-coverage case, where there is no fuller panel to be had and ``_numerator`` says so.

    A label absent from the real panel is SKIPPED rather than raised on: ``_aligned_pair`` already
    raises for the rows being scored, and this is a diagnostic input to a bound, not a new gate.
    """
    f_perts, f_means = pred_bulk_full
    f_keep, f_rows = [], []
    for i, pert in enumerate(f_perts):
        pert = str(pert)
        if pert == control or pert not in real_index:
            continue
        f_keep.append(i)
        f_rows.append(real_index[pert])
    if len(f_keep) <= n_scored:
        return None
    f_keep = np.asarray(f_keep, dtype=np.intp)
    f_rows = np.asarray(f_rows, dtype=np.intp)
    f_means = np.asarray(f_means, dtype=np.float64)
    f_tn = correction_for(pred_moments, f_perts, f_means, comparator=comparator)
    f_term = np.minimum(f_tn[f_keep], PRED_TRACE_CAP_K * real_tn[f_rows])
    # `genes is None` already WARNED once for the scored rows in `_numerator`; do not warn twice
    # for the same call, and an unresolvable panel excludes nothing on either row set.
    f_excl = {} if genes is None else _exclusion_cols(
        [str(f_perts[i]) for i in f_keep], genes, target_gene_map=target_gene_map,
        gate_labels=[p for p in real_index if p != control], who=who, n_genes=n_genes,
    )
    f_w = _row_weights(f_keep.size, n_genes, f_excl)
    return (_across_pert_budget(f_means[f_keep], f_excl, f_w), float((f_term * f_w).sum()))


def _drop_on_target(summed, a_means, a_rows, b_means, b_rows, n_genes, cols):
    """Remove each row's own target-gene coordinate from an already-summed squared distance.

    Returns ``(summed_without_that_gene, genes_summed_per_row)``. Issue #172, and the ROUTE
    matters as much as the result:

    ``summed`` arrives from :func:`moments.unbiased_sq_dist` as
    ``sum_g (a_g - b_g)^2 - C_a - C_b``, so subtracting ``(a_g* - b_g*)^2`` for one gene ``g*``
    leaves that gene out of the PLUG-IN distance while both sampling corrections stay whole.
    That is the same shape as ``metrics.discrimination.correct_excluded_gene`` -- correct an
    already-computed distance rather than materialize an ``[n_perts, G-1]`` copy -- and it is
    also the same DEFINITION: exclusion removes a coordinate from the distance and leaves the
    panel-wide normalization (which ``mu`` is built from) untouched.

    ⚠️ **Do NOT vectorize the loop below into fancy indexing.** It looks like an obvious win and
    it is not: ``out[ks] -= (a_means[a_rows[ks], cs] - b_means[b_rows[ks], cs]) ** 2`` is NOT
    bit-identical to the scalar loop. A ``np.float64`` scalar ``x ** 2`` goes through libm
    ``pow``; the array path multiplies, and the two disagree by 1 ULP on ~0.09% of operands
    (1756 of 2e6 measured on numpy 2.5.2). No array form escapes it -- ``x ** 2``, ``x * x`` and
    ``np.square`` all agree with each other and all differ from the scalar path. On a
    20000-perturbation panel that moved 9 perturbation values, i.e. it would silently change a
    competition metric's last bit. What it buys, measured at G = 20000: **0.18 ms** at the
    competition's 300 perturbations and 13 ms at 20000, against a metric whose cost is
    pseudobulk over millions of cells. Declined on that basis (Gemini, PR #316); if it is ever
    revisited, the gate is bit-identity on a real panel, not the microbenchmark.

    ⚠️ **The corrections are deliberately NOT gene-corrected, and this is the one approximation
    in #172's expression half.** ``C_p`` is exactly decomposable over genes -- under
    ``bulk_lognorm`` it is ``((n-1)/n) * sum_g q_g`` (``moments.jackknife_correction``), verified
    additive to 2e-16 -- but the cached artifact stores only the SCALAR sum, so ``q_{g*}`` is not
    recoverable from it. Recovering it exactly needs either a target-aware moments artifact
    (O(P) extra floats, invalidating every warm cache) or the full ``[P, G]`` per-gene matrix.
    MEASURED, real side of all three official val contexts, against the same cached moments the
    official bundles were built from: the target gene's share of its own ``C_p`` is a median
    0.0055-0.0083% against ``1/G = 0.0054%`` -- i.e. an ORDINARY gene for the variance, even
    though it is the single largest-moving gene for the DISTANCE in 57-66% of perturbations.
    Leaving the corrections whole therefore understates ``sum_p den_p`` by **0.0125% / 0.0108% /
    0.0070%** on contexts A / B / C, against the **10.21% / 11.30% / 6.07%** of the metric's
    range that the exclusion itself removes -- roughly a thousandth of the effect being fixed.

    **The 1.0 anchor holds.** A submission emitting the real control's own cells has
    ``mu_pred = mu_ctrl`` and ``C_pred = C_ctrl``, so numerator and denominator carry the SAME
    correction terms and cancel wherever the #247 cap does not bind -- the same condition as
    before #172.

    ⚠️ **A LEVER DOES OPEN, and it is bounded rather than absent.** An earlier version of this
    note claimed none did, on the grounds that the exploitable range of the subtracted term is
    ``min(C_pred, k C_real,p)`` either way. That argument is wrong: it compares the BOUND while
    ignoring what exploiting it costs. Before #172, scattering counts in the target gene's own
    column to inflate ``C_pred`` also moved ``mu_pred`` at that gene and so raised the plug-in
    error the numerator charged for it. After #172 that gene's plug-in error is gone, and the
    variance it generates is still subtracted. Codex found the counterexample and it reproduces:
    on a TWO-gene panel a prediction differing from the no-skill arm only in its target column
    reads 1.2756 before and **-0.5228** after.

    What contains it is that the two conditions the exploit needs pull against each other.
    Concentrating ``q_pred`` on the target gene requires scattering that column, which raises
    ``C_pred`` past ``k C_real,p`` -- where #247's cap pins the subtracted term and further
    concentration buys exactly nothing. Staying UNDER the cap instead requires being
    under-dispersed off-target, which this metric already punishes hard (#278). MEASURED on a
    G=4,000 / 400-cell panel calibrated to the official contexts on both ratios that scale the
    effect -- target gene 4.6% of ``sum D`` (val 3.2-5.5%) and corrections 87% of ``sum D``
    (val 46-55%) -- over adversaries spanning the whole trade-off, including fully flattened
    off-target cells with 100% of ``q_pred`` on the target gene: the best any of them does
    against the gene-corrected estimator is **0.013 of the range, and the sign is against the
    adversary** (this form reads WORSE for it, 6.1296 vs 6.1167). The two-gene blow-up needs one
    gene to be half the library, which no real panel is.

    So the trade is: a bounded, measured, wrong-signed residual here, against a target-aware
    moments artifact -- ``[P]`` extra floats, but accumulation would need ``target_gene_map`` and
    every warm cache and all three jackknife kernels would move. If that artifact is ever built,
    exclude the gene from all three corrections and apply the #247 cap AFTER the exclusion.
    """
    n_row = np.full(a_rows.size, float(n_genes), dtype=np.float64)
    if not cols:
        return summed, n_row
    out = np.array(summed, dtype=np.float64, copy=True)
    for k, col in cols.items():
        out[k] -= (a_means[a_rows[k], col] - b_means[b_rows[k], col]) ** 2
        n_row[k] = n_genes - 1
    return out, n_row


def _per_gene(summed, n_row):
    """``summed / n_row``, with an exhausted gene set giving NaN rather than a divide error.

    ``n_row == 0`` needs G == 1 AND that single gene being the perturbation's own -- so the
    scored gene set is empty and there is no value, which is exactly ``safe_mse``'s
    empty-input contract (``NaN``, not a ZeroDivisionError or an inf)."""
    return np.divide(summed, n_row, out=np.full(np.shape(summed), np.nan, dtype=np.float64),
                     where=n_row > 0)


def mae(
    pred: ad.AnnData | None = None,
    real: ad.AnnData | None = None,
    *,
    pred_bulk: tuple[np.ndarray, np.ndarray] | None = None,
    real_bulk: tuple[np.ndarray, np.ndarray] | None = None,
    pert_col: str = "target",
    control: str = "non-targeting",
) -> dict[str, float]:
    """Mean absolute error between pred and real pseudobulk, per perturbation.

    Hybrid input: pass raw AnnData (`pred`, `real`) or precomputed pseudobulk
    tuples (`pred_bulk`, `real_bulk`) as returned by `prep.pseudobulk`.
    The control perturbation is excluded from the result. Every predicted
    perturbation must appear in `real_bulk`; extra real-only perturbations
    (if any) are ignored.
    """
    # Resolve each side independently (F6.1): a supplied bulk is reused as-is and only the missing
    # side needs its AnnData -- so a hybrid call (one bulk + the other's AnnData) works and a
    # supplied bulk is never silently overwritten. Matches mse/mae_delta/mse_delta/pearson_delta.
    pred_bulk, real_bulk = _resolve_bulks(pred, real, pred_bulk, real_bulk, pert_col)

    pred_perts, pred_means = pred_bulk
    real_perts, real_means = real_bulk
    real_index = {str(p): i for i, p in enumerate(real_perts)}

    out: dict[str, float] = {}
    for i, p in enumerate(pred_perts):
        p = str(p)
        if p == control:
            continue
        if p not in real_index:
            raise ValueError(
                f"perturbation {p!r} present in pred_bulk but missing from real_bulk; "
                "every predicted perturbation must appear in real_bulk"
            )
        out[p] = safe_mae(pred_means[i], real_means[real_index[p]])
    return out


def _resolve_real_bulk(real, real_bulk, pert_col):
    """The real-side half of `_resolve_bulks`, so a real-only metric never touches pred.

    `_resolve_bulks` is refactored to call this for its second side; there stays exactly one
    implementation.
    """
    if real_bulk is None:
        if real is None:
            raise ValueError("provide real AnnData or a real_bulk tuple")
        real_bulk = pseudobulk(real, pert_col)
    return real_bulk


def _resolve_bulks(pred, real, pred_bulk, real_bulk, pert_col):
    """Return (pred_bulk, real_bulk), computing pseudobulk from AnnData for whichever
    side's bulk is absent. Each side is resolved independently: a supplied bulk is reused
    as-is (never recomputed), and only the missing side requires its AnnData (Gemini
    impl review — supports hybrid input and avoids redundant pseudobulk)."""
    if pred_bulk is None:
        if pred is None:
            raise ValueError("provide pred AnnData or a pred_bulk tuple")
        pred_bulk = pseudobulk(pred, pert_col)
    real_bulk = _resolve_real_bulk(real, real_bulk, pert_col)
    return pred_bulk, real_bulk


def real_mass_ratio(
    pred=None,
    real=None,
    *,
    pred_bulk=None,
    real_bulk=None,
    mass_target: float | None = None,
    pert_col: str = "target",
    control: str = "non-targeting",
    **_,
) -> dict[str, float]:
    """``sum_g expm1(real bulk_p) / mass_target``, per real perturbation. DIAGNOSTIC.

    1.0 by construction under ``bulk_lognorm`` for any group with POSITIVE MASS, and 0 for an
    all-zero group -- ``prep.bulk_lognorm_means`` maps that to a zero bulk by policy rather
    than to NaN, so the ratio follows. Under the ``lognorm`` fallback it is the concavity
    deficit that makes that comparator a dispersion functional -- 0.8199 measured on a
    200-construct control group at a target of 28,118 (issues #264, #260, #261).

    ``mass_target`` is COMPARATOR-DEPENDENT and supplied by the dispatcher:
    ``cfg.bulk_target_sum`` under ``bulk_lognorm``, the resolved per-cell ``target_sum`` under
    ``lognorm``. It is ``None`` only where the per-cell target is genuinely unresolvable --
    ``target_sum=None`` on lognorm-effective input, where ``resolve_target_sum`` has no
    library-size median to take. An explicitly configured numeric ``target_sum`` resolves and
    is used, lognorm input or not. With ``None`` the metric emits NaN: a ratio against a
    guessed denominator would be worse than no number.

    REAL SIDE ONLY, like ``expr_distance_unbiased``: it describes the reference panel, so it
    is identical for every submission and cacheable once per panel. ``pred`` and ``pred_bulk``
    are accepted for dispatch compatibility and deliberately never read.
    """
    real_bulk = _resolve_real_bulk(real, real_bulk, pert_col)
    perts, means = real_bulk
    if mass_target is None:
        return {str(p): float("nan") for p in perts if str(p) != control}
    mass = np.expm1(np.asarray(means, dtype=np.float64)).sum(axis=1) / float(mass_target)
    return {str(p): float(m) for p, m in zip(perts, mass) if str(p) != control}


def _refuse_adata_under_bulk_lognorm(comparator, who, **sides):
    """Refuse AnnData input on the group-sum comparator (Copilot, PR #269).

    `_resolve_bulks` computes `prep.pseudobulk`, a plain arithmetic mean of `adata.X`. Under
    `bulk_lognorm` the means must be `log1p(TS * P_g / sum_g P_g)` -- a different space, and
    `pseudobulk` has no `bulk_target_sum` to build it with. The correction does not catch the
    mistake: `correction_for` returns `jk` without consulting `means` on this branch, so the
    metric would subtract a log-space jackknife from a raw-mean-space distance and return a
    plausible number. RAISE instead.

    Nothing in `src/` or `tools/` reaches this -- every driver passes precomputed bulks
    (`run.py:275`) -- so this closes a public-API path, and changes no run. The hybrid
    AnnData form stays fully supported under `lognorm`, where `pseudobulk` IS the comparator.
    """
    missing = [name for name, bulk in sides.items() if bulk is None]
    if comparator == "bulk_lognorm" and missing:
        raise ValueError(
            f"{who}: comparator='bulk_lognorm' needs precomputed bulks, but {missing} "
            "was/were not supplied. Computing them here would use prep.pseudobulk -- a plain "
            "mean of adata.X -- while the correction is a jackknife over log1p(TS * P / S), "
            "so the two sides would be in different spaces and the result would be silently "
            "wrong rather than an error. Pass the bulks from a "
            "*_with_moments/pseudobulk driver, or use comparator='lognorm', where "
            "prep.pseudobulk IS the comparator."
        )


def mse(
    pred: ad.AnnData | None = None,
    real: ad.AnnData | None = None,
    *,
    pred_bulk: tuple[np.ndarray, np.ndarray] | None = None,
    real_bulk: tuple[np.ndarray, np.ndarray] | None = None,
    pert_col: str = "target",
    control: str = "non-targeting",
) -> dict[str, float]:
    """Mean squared error between pred and real pseudobulk, per perturbation.

    Sibling of `mae` (expr_mae): absolute expression error, no delta. Hybrid input,
    control excluded, every predicted perturbation must appear in real_bulk.
    """
    pred_bulk, real_bulk = _resolve_bulks(pred, real, pred_bulk, real_bulk, pert_col)
    pred_perts, pred_means = pred_bulk
    real_perts, real_means = real_bulk
    real_index = {str(p): i for i, p in enumerate(real_perts)}

    out: dict[str, float] = {}
    for i, p in enumerate(pred_perts):
        p = str(p)
        if p == control:
            continue
        if p not in real_index:
            raise ValueError(
                f"perturbation {p!r} present in pred_bulk but missing from real_bulk; "
                "every predicted perturbation must appear in real_bulk"
            )
        out[p] = safe_mse(pred_means[i], real_means[real_index[p]])
    return out


def _real_rows(real_bulk, real_moments, control, who, *, comparator, driver=None):
    """Every non-control perturbation in `real_bulk`, from the real side ALONE.

    ⚠️ Both the row set and the values come from the reference. Nothing here reads
    `pred_bulk`, `pred_moments` or predicted expression, which is what makes
    `expr_distance_unbiased` cacheable once per panel and impossible for a submission to
    steer. Two earlier drafts each broke one half of that and are worth knowing about:

    * Draft 1 routed this through `_aligned_pair`, taking the moments from the PRED side. The
      test meant to catch it varied only predicted VALUES, so it passed (codex round 1).
    * Draft 2 kept the values real-only but scoped the ROW SET to the predicted labels. That
      is subtler and worse: on shard streaming, which validates only the gene axis
      (`scale.py:116-123`), a submission omitting a real perturbation makes BOTH components
      omit it, so `_derived_value`'s label-set check sees two equal sets and cannot detect the
      omission -- the submission picks its own denominator cohort (codex round 3).

    Row-count consistency with the numerator is achieved in the DISPATCH instead: `scale.py`
    runs every metric against the FULL `real_bulks` and then filters the EMITTED ROWS to the
    partition's `chosen` cohort. Restricting `real_bulks` itself was tried and reverted -- it
    silently re-ranked `pds_*`, which ranks each predicted effect against the whole real panel
    and takes its denominator from the full real count (`discrimination.py:114-150`). `chosen`
    comes from the REAL universe via `select_subset`, never from the submission, so the row
    filter stays submitter-independent.

    Returns `(rows, real_means, real_tn, labels, ci)` or `None` when nothing survives.
    """
    if real_moments is None:
        raise ValueError(
            f"{who} requires per-group moments; the real side supplied none. "
            f"Driver: {driver or 'unknown (moments were not routed to this metric)'}. "
            "That driver must run its pseudobulk with with_moments=True, or the metric must "
            "not be requested on it. This never falls back to the biased value."
        )
    real_perts, real_means = real_bulk
    labels = [str(p) for p in real_perts]
    if control not in labels:
        raise ValueError(
            f"{who} measures ||mu_real_p - mu_real_control||^2 and the control {control!r} is "
            f"absent from real_bulk; it carries {len(labels)} groups. The real control is "
            "required -- there is no fallback, and the pred control is not a substitute "
            "because this quantity must not depend on the submission."
        )
    ci = labels.index(control)
    rows = np.asarray([i for i, p in enumerate(labels) if p != control], dtype=np.intp)
    if rows.size == 0:
        return None
    real_tn = correction_for(real_moments, real_perts, real_means, comparator=comparator)
    return rows, np.asarray(real_means, dtype=np.float64), real_tn, labels, ci


def _aligned_pair(
    pred_bulk,
    real_bulk,
    pred_moments,
    real_moments,
    driver,
    control,
    who,
    *,
    comparator,
):
    """Shared prologue for the two NUMERATORS: align pred rows onto real rows, drop control.

    Returns `(keep, rows, pred_means, real_means, pred_tn, real_tn, real_index, pred_perts)`
    -- EIGHT items; `pred_perts` is last and is what both numerators key their output dict by.
    `keep`/`rows` are intp arrays. Returns `None` when there is nothing to score.
    """
    if pred_moments is None or real_moments is None:
        missing = " and ".join(
            n for n, v in (("pred", pred_moments), ("real", real_moments)) if v is None
        )
        raise ValueError(
            f"{who} requires per-group moments; the {missing} side(s) supplied none. "
            f"Driver: {driver or 'unknown (moments were not routed to this metric)'}. "
            "That driver must run its pseudobulk with with_moments=True, or the metric must "
            "not be requested on it. This never falls back to the biased value."
        )
    pred_perts, pred_means = pred_bulk
    real_perts, real_means = real_bulk
    real_index = {str(p): i for i, p in enumerate(real_perts)}
    pred_tn = correction_for(pred_moments, pred_perts, pred_means, comparator=comparator)
    real_tn = correction_for(real_moments, real_perts, real_means, comparator=comparator)
    keep, rows = [], []
    for i, p in enumerate(pred_perts):
        p = str(p)
        if p == control:
            continue
        if p not in real_index:
            raise ValueError(
                f"perturbation {p!r} present in pred_bulk but missing from real_bulk; "
                "every predicted perturbation must appear in real_bulk"
            )
        keep.append(i)
        rows.append(real_index[p])
    if not keep:
        return None
    return (np.asarray(keep, dtype=np.intp), np.asarray(rows, dtype=np.intp),
            np.asarray(pred_means, dtype=np.float64),
            np.asarray(real_means, dtype=np.float64),
            pred_tn, real_tn, real_index, pred_perts)


def _numerator(
    pred_bulk,
    real_bulk,
    pred_moments,
    real_moments,
    driver,
    control,
    *,
    cap,
    comparator,
    genes=None,
    target_gene_map=None,
    pred_bulk_full=None,
):
    """`expr_mse_unbiased` (cap=False) or `expr_mse_unbiased_capped` (cap=True).

    Gene-averaged PER ROW, not by one shared `/G`: since #172 the divisor is `G - 1` for every
    perturbation whose label resolves to a measured gene and `G` for the rest, which is what
    `_drop_on_target` returns alongside the summed distance. Both public wrappers document the
    exclusion; this line said "/G" and would have misled a reader of the internal contract
    (Copilot, PR #316).
    """
    who = "expr_mse_unbiased_capped" if cap else "expr_mse_unbiased"
    got = _aligned_pair(
        pred_bulk,
        real_bulk,
        pred_moments,
        real_moments,
        driver,
        control,
        who,
        comparator=comparator,
    )
    if got is None:
        return {}
    keep, rows, pred_means, real_means, pred_tn, real_tn, real_index, pred_perts = got
    n_genes = pred_means.shape[1]
    if n_genes == 0:  # matches safe_mse's empty-input contract (NaN, not a divide error)
        return {str(pred_perts[i]): float("nan") for i in keep}
    # Issue #172. Resolved against the labels this call SCORES (`keep`, pred-side and possibly a
    # shard) while the zero-resolve raise is judged on the REAL panel, which every driver hands
    # over whole -- otherwise a shard holding only unresolved targets would hard-fail a panel
    # that scores fine as a whole (`distances.resolve_exclusion_columns`). The control is dropped
    # from the gate set: it has no target gene and never resolves, so leaving it in would report
    # a spurious map gap.
    excl = _exclusion_cols(
        [str(pred_perts[i]) for i in keep], genes, target_gene_map=target_gene_map,
        gate_labels=[p for p in real_index if p != control], who=who, n_genes=n_genes,
    )
    # One fancy-index copy, reused by the #348 bound below and by `unbiased_sq_dist`. The
    # `[P, G]` copy is what the call already made; hoisting it does not add a second one.
    pred_sel = pred_means[keep]
    pred_term = pred_tn[keep]
    full = None
    if cap and pred_bulk_full is not None:
        full = _whole_panel_budget(
            pred_bulk_full, pred_moments, real_tn, real_index, control=control,
            comparator=comparator, genes=genes, target_gene_map=target_gene_map, who=who,
            n_genes=n_genes, n_scored=int(keep.size),
        )
    if cap:
        # THE GUARD (issue #247). Row-aligned, so it must happen after the reindex above.
        # Only the CORRECTION is bounded; the value stays signed and uncapped.
        pred_term = np.minimum(pred_term, PRED_TRACE_CAP_K * real_tn[rows])
        fired = int(np.count_nonzero(pred_term < pred_tn[keep]))
        if fired:
            logger.info(
                "expr_mse_unbiased_capped: capped the prediction's sampling correction "
                "(comparator=%s) at %.3gx the real side's on %d of %d perturbations "
                "(issue #247)", comparator,
                PRED_TRACE_CAP_K, fired, len(keep),
            )
        # THE SECOND BOUND (issue #348, the measured half of #294). `pred_tn` is a delete-1
        # cell jackknife under `bulk_lognorm`, i.e. WITHIN-set dispersion, and that equals the
        # submitted pseudobulk's own error only for exchangeable cells. A submission whose
        # per-cell scatter CANCELS in the aggregate has almost none of that error in the plug-in
        # distance and was still handed the full subtraction -- measured worth 0.72-0.90 of this
        # member's [0, 1] range on the official val panels, reachable with no adversarial content
        # at all (pin the per-(p, g) sums; nothing else) and taken by a live dev-leaderboard
        # submission for +0.1389 of a 0.2295 OVERALL.
        #
        # #247's cap alone cannot close it, and the reason is the shape of the cap: it SATURATES,
        # so above saturation the deduction is a CONSTANT `k * jk_real` with no gradient -- which
        # #294 read as the bound and #348 measured as the exploit. A constant deduction against a
        # denominator fixed by the real data is a fixed FRACTION of the member's range.
        #
        # What is bounded here is the TOTAL: a submission may claim at most its own
        # across-perturbation centred sum of squares (`_across_pert_budget`, a CONSERVATIVE proxy
        # for `sum_p trVar(mu_hat_p)` -- biological signal only adds, but the estimator also falls
        # `1/n_g` short of what a flat panel is strictly owed, so it is deliberately not an upper
        # bound; the helper's docstring has the measurement and why the two cannot both be had),
        # and when it claims more,
        # every row is scaled down in proportion to what it claimed. Two properties follow, and
        # both are the reason for this form rather than a per-row `min` against a variance:
        #   * ATTACKING IT IS NOT PROFITABLE. Budget and charge are the same quantity on the same
        #     coordinates, so injecting across-perturbation variation is break-even at best
        #     (measured -0.003%); a variance-with-tolerance form rebates +10.37%.
        #   * IT IS NEUTRAL TO UNEVEN CELL COUNTS. Predicted cells per perturbation are NOT
        #     constrained by the competition rules, so rows are not homoscedastic in general and a
        #     single scalar ceiling would clip the high-variance rows first. Proportional
        #     allocation binds all rows or none -- the test is purely `budget < claim`.
        #
        # ⚠️ WHAT THIS DOES NOT CLOSE -- two residuals, both disclosed in #348 rather than claimed
        # away. The budget contains the submission's biological SIGNAL, and signal cannot
        # authenticate sampling error:
        #   1. NON-BINDING REGIME. A submission whose genuine across-perturbation spread already
        #      exceeds its total claim has `r = 1`, and can still pin its aggregates and collect the
        #      capped deduction. That is where the replicate anchor sits, so an ACCURATE submission
        #      is not protected by this bound at all. MEASURED at the anchor's OWN depth, on its own
        #      five derived seeds -- `_score_one_split` scores `half_b` against `half_a`, so BOTH of
        #      its sides are `floor(n/2)`-cell halves and its `jk_real` is half_a's, ~2x the cached
        #      full-depth one -- the anchor's budget is **1.54 / 1.38 / 1.58x its claim** on the
        #      official val A / B / C panels. ⚠️ An earlier version of this note said ~3.2x; that
        #      divided the anchor's HALF-depth budget by the FULL-depth claim and is not a ratio of
        #      two commensurable things. The conclusion is unchanged, the margin is half what was
        #      written. What is protected is the near-flat end -- where the channel was measured, and
        #      where the live exploit sat. Buying into the non-binding regime is not a route: it
        #      costs the charge and unlocks at most `k * jk_real` per row.
        #   2. TRUTH-ALIGNED OVERSCALING, inside the binding regime. For a prediction `alpha * d`
        #      against a real centred component `d`, the plug-in error is `(alpha-1)^2 ||d||^2`
        #      while the budget is `alpha^2 ||d||^2`, so their difference `(1 - 2 alpha) ||d||^2`
        #      keeps improving past `alpha = 1` -- amplifying a correct answer beyond the truth pays
        #      until the budget crosses the claim. Reproduced through the production pipeline
        #      (codex review round 2). It requires `B_pred(at the truth) < claim`, and the algebra
        #      says truth is globally optimal whenever that fails: `N(1) - N(alpha_max) =
        #      -(sqrt(claim) - sqrt(B))^2 <= 0`. The official panels are 1.37-2.05x OUTSIDE the
        #      regime (the span of every measured cell, anchor half depth included), which is why
        #      the measured arms do not show it, but that is a property of the panel, not of the
        #      bound.
        #      The candidate fix -- also cap by the REFERENCE's centred sum of squares,
        #      `r = min(1, B_pred/claim, B_real/claim)` -- closes the direction and is INERT on all
        #      three official contexts for every FULL-PANEL submission and for the measured anchor,
        #      and it is MEASURED AND NOT TAKEN (#348, done 2026-08-19). Two things settle it:
        #        * For a SUBMISSION the comparison needs no arm at all. A submission's real side is
        #          always the reference panel and #247 caps each row against it, so over whatever
        #          row set `S` the call uses, `claim_S <= k sum_{q in S} w_q jk_real_q`. Taking `S`
        #          = ALL non-control REFERENCE rows gives `claim_max`, which bounds every possible
        #          `S` -- including `_whole_panel_budget`'s `f_keep`, which can be LARGER than the
        #          rows this call scores and applies the same per-row `min`. Measured on that
        #          reference row set (300/300, own target gene out): `B_real/claim_max` =
        #          2.047 / 1.715 / 2.050 on val A / B / C, so no submission covering the panel
        #          reaches the term. For the anchor, at its own depth, 1.54 / 1.38 / 1.58 (above).
        #          ⚠️ A submission covering only PART of the panel forms budget and claim from
        #          that subset, and a biologically narrow slice genuinely has less spread, so this
        #          is a whole-panel statement -- the same caveat the partial-panel warning below
        #          carries.
        #        * On the panels where it could help, the anchor's EXPECTED margin is no larger --
        #          which is why the measurement above is the argument, and why the honest form of
        #          the conclusion is "measured on the three official panels", not "proved".
        #          ⚠️ What follows is an EXPECTATION-level heuristic under idealised iid emission,
        #          NOT a theorem. `_across_pert_budget`'s identity (see its docstring) is per gene
        #          `E[B_g] = B_g(mu_true) + V_g - (sum_{p in R_g} w_p^2 s2_pg)/(sum_{p in R_g} w_p)`
        #          with `V_g = sum_{p in R_g} w_p s2_pg`, so at equal weights, in exact arithmetic,
        #          and summing only over the genes this code KEEPS
        #          (`H = {g : cnt_g >= 2 and wsum_g > 0}`; the implementation also clamps each
        #          `ss_g` at 0, a roundoff floor):
        #          `E[B] = B(mu_true) + sum_{g in H} (1 - 1/|R_g|) V_g`.
        #          The factor is PER GENE, so collapsing it to one number means taking the
        #          variance-weighted average `a = sum_H (1 - 1/|R_g|) V_g / sum_H V_g`. `|R_g|` is
        #          `P` or `P - 1` on a GENE-level panel (verified on the official contexts -- 300
        #          rows resolve to 300 DISTINCT target columns), so there `a` is 0.99666-0.99667 at
        #          `P = 300`; on a GUIDE-level panel `m_g` rows target gene `g`, `cnt` drops once
        #          per pair, `|R_g| = P - m_g`, and `a` CAN fall further -- only the
        #          duplicated-target genes' factors move, so whether the panel-wide `a` follows
        #          depends on how much `V_g` weight those genes carry.
        #          ONLY IF `claim ~= sum_H V_g` does that give `E[B]/claim ~= a + B_true/claim` --
        #          and that antecedent is the load-bearing assumption, NOT something the code
        #          provides: see gap (ii). Where it holds, the term reaches an arm only when the
        #          arm's GENUINE spread is `(1 - a)` of its own sampling variance -- a few tenths of
        #          a percent on the official distinct-target panels, potentially more on a
        #          guide-level one.
        #          Halving the depth doubles the sampling half and leaves `B_true` alone, so the
        #          ANCHOR's expected margin is smaller wherever `B_true > 0` and equal at
        #          `B_true = 0` -- never larger, but not strictly ordered.
        #          THE THREE GAPS. (i) It is `E[B]`, not `B`; the realized value is one draw,
        #          concentrated only because the sum runs over ~18.5k genes. (ii) NOTHING bounds
        #          `claim` by `sum_H V_g`. `sum_H V_g` is the RETAINED-coordinate, gated trace --
        #          each row's own target column left out (#172) and the ungated genes dropped --
        #          while `claim`'s jackknife is the WHOLE-panel trace, and the `min` gives only
        #          `claim <= sum_q w_q jk_pred_q`, a biased estimate of it. So an arm whose scatter
        #          sits only in its own target column has `B = 0` with `claim > 0`. That gap is not
        #          a flaw: it is exactly the binding case this bound exists to catch, and
        #          `_across_pert_budget`'s docstring explains why dropping that coordinate from the
        #          BUDGET (here, via the shared `excl` map) and from the PLUG-IN CHARGE (via
        #          `_drop_on_target`, which consumes the same map independently) -- but NOT from
        #          `claim` -- is what keeps the comparison in the safe direction.
        #          Under the same antecedent `claim ~= sum_H V_g` the identity gives
        #          `E[B_pred]/claim ~= a + B_true/claim`, so this bound's binding regime and
        #          residual 2's coincide: both need `B_true/claim < 1 - a`.
        #          (iii) `floor(n/2)`, the nonlinear jackknife and
        #          `min(jk_b, jk_a)` mean the anchor's depth halving is not an exact doubling.
        #      What the measurement is good for instead is a PANEL check: `B_real` and `claim_max`
        #      are real-side-only and cacheable, so a bundle build can certify the regime is
        #      unreachable for its panel without any metric touching the scale.
        # The CLASS is closed only by putting both ends of the scale on a matched emission, or by
        # constraining the submission format so the emission is stochastic -- both out of scope
        # here and recorded in #348.
        # Both sides in SCORE units -- see `_row_weights` for the rebate that raw units open.
        row_w = _row_weights(keep.size, n_genes, excl)
        claim = float((pred_term * row_w).sum())
        if claim > 0.0:
            # ⚠️ PARTIAL PRED PANELS. This is the one term whose value depends on which OTHER
            # predicted perturbations are present, so a run scoring a SUBSET of the pred panel
            # forms budget and claim from that subset -- and a subset is not merely a noisier
            # estimate, since a biologically narrow slice genuinely has less spread. `pred_bulk`
            # carries the whole pred panel wherever a driver can supply it, which restores exact
            # partition-independence; where it cannot, the subset is used and the run says so.
            # The REAL panel is always handed over whole, which is what makes this detectable.
            n_real_perts = len(real_index) - (1 if control in real_index else 0)
            if full is not None:
                budget, claim = full
            elif keep.size < n_real_perts:
                logger.warning(
                    "expr_mse_unbiased_capped: scoring %d of the real panel's %d perturbations "
                    "and no whole-panel pred bulk was supplied, so #348's correction budget is "
                    "formed from that SUBSET. Values are then not comparable with a whole-panel "
                    "run's. Pass the unrestricted pred bulk, or score the whole panel "
                    "(issue #348).", keep.size, n_real_perts,
                )
                budget = _across_pert_budget(pred_sel, excl, row_w)
            else:
                budget = _across_pert_budget(pred_sel, excl, row_w)
            ratio = budget / claim
            if ratio < 1.0:
                pred_term = pred_term * ratio
                logger.info(
                    "expr_mse_unbiased_capped: scaled the prediction's sampling correction "
                    "(comparator=%s) to %.4gx on all %d perturbations -- it claimed %.4g against "
                    "an across-perturbation budget of %.4g (issue #348)",
                    comparator, ratio, len(keep), claim, budget,
                )
            if ratio < PRED_CORRECTION_BUDGET_FLAG_RATIO:
                logger.warning(
                    "expr_mse_unbiased_capped: this prediction's across-perturbation spread is "
                    "%.4g against a claimed sampling correction of %.4g (ratio %.4g < %.3g). Its "
                    "per-cell scatter largely CANCELS in the pseudobulk, which is not what an "
                    "i.i.d. sample of a predicted cell population looks like -- honest arms "
                    "measure ~1.0 and the two arms that pin their aggregates measured 0.006 and "
                    "0.000. The correction is bounded (issue #348), so the score is not inflated "
                    "by it, but the submission is worth a look.",
                    budget, claim, ratio, PRED_CORRECTION_BUDGET_FLAG_RATIO,
                )
    summed = unbiased_sq_dist(pred_sel, real_means[rows], pred_term, real_tn[rows])
    summed, n_row = _drop_on_target(summed, pred_means, keep, real_means, rows, n_genes, excl)
    vals = _per_gene(summed, n_row)
    return {str(pred_perts[i]): float(v) for i, v in zip(keep, vals)}


def mse_unbiased(
    pred: ad.AnnData | None = None,
    real: ad.AnnData | None = None,
    *,
    comparator: str,
    pred_bulk: tuple[np.ndarray, np.ndarray] | None = None,
    real_bulk: tuple[np.ndarray, np.ndarray] | None = None,
    pred_moments=None,
    real_moments=None,
    driver: str | None = None,
    pert_col: str = "target",
    control: str = "non-targeting",
    genes=None,
    target_gene_map: dict[str, str] | None = None,
) -> dict[str, float]:
    """Sampling-bias-corrected `expr_mse`, averaged over the gene set it SCORES. UNCAPPED — a
    DIAGNOSTIC only.

        G_p = G - {g_p}   (issue #172: the perturbation's own gene, when its label resolves)

        expr_mse_unbiased_p = ( ||mu_pred - mu_real,p||^2_{G_p}
                                - C_pred - C_real,p ) / |G_p|

    ⚠️ BANNER (2026-08-10, #264 PR2). EVERY MEASUREMENT QUOTED BELOW WAS MADE UNDER THE
    PRE-#264 PER-CELL `lognorm` COMPARATOR. It is kept as history, not as a current reading.
    There, `mu` was a per-cell mean of `log1p(CPM)` and the subtracted `C` was the analytic
    `tr Sigma-hat/n`. Under the shipped v2 comparator `mu` is `log1p(TS * P_g / sum_g P_g)`,
    built from the group SUM, and `C` is a delete-1 jackknife (`moments.correction_for`).
    Different estimator, different space: nothing below transfers quantitatively — the values,
    their scale, and even the emission ordering they imply have all moved (a tiled arm now
    scores BETTER than a dispersed one, `tests/test_baseline_emission.py
    ::test_the_comparator_move_INVERTS_which_emission_model_expr_mse_prefers`). The history
    survives deletion because published pre-#264 columns are still readable and need a key
    (spec §7); which comparator produced a given run is stamped in its metadata.

    This is the pre-#247 metric restored bit-for-bit -- that revision computed
    `unbiased_sq_dist(...) / n_genes` with the uncapped `pred_tn` (`810215c~1`,
    `delta.py:176`). Verified over 1,744 perturbation-arms with matched moments and 657 with
    independent ones (`pred_tn/real_tn` spanning -3.78 to 5.52e3, `counts` of 0 and 1):
    bit-identical, max abs diff 0.0. Exact for a structural reason -- `trace_sigma`,
    `trace_over_n_for` and `unbiased_sq_dist` are AST-identical to that revision, and #219's
    `n < 2 -> 0.0` fix (`8b9e85d`) is an ANCESTOR of `810215c` rather than a divergence.

    ⚠️ That reproduction claim is scoped to `comparator="lognorm"` (#264 PR2). This function
    now calls `moments.correction_for`, not `trace_over_n_for` directly, and under `"lognorm"`
    that IS `trace_over_n_for` verbatim -- so the bit-identity above still holds on exactly the
    runs it was measured on. Under `"bulk_lognorm"` the subtracted term is the delete-1
    jackknife instead, which is a DIFFERENT estimator in a different space; nothing about the
    pre-#247 comparison carries over to it.

    ⚠️ Requires PRECOMPUTED bulks under `bulk_lognorm` -- see `_refuse_adata_under_bulk_lognorm`.

    ⚠️ **The bit-identity claim above is now scoped to the WHOLE-PANEL gene set (issue #172).**
    Since the 2026-08-17 ruling this shares `_numerator`'s on-target exclusion with the capped
    sibling, so on a panel where targets resolve it sums `G - 1` genes and divides by `G - 1`.
    That is deliberate: this exists to make the capped numerator AUDITABLE, and an audit whose
    gene set differs from the audited quantity's audits nothing. The pre-#247 reproduction is
    still reachable -- it is what this function returns with `genes=None`, which is also the
    shape every measurement above was taken in.

    ⚠️ NEVER SCORE THIS. The uncapped correction is the submitter-controlled lever #247
    closed: reporting the same predicted mean through more dispersed cells enlarges the
    subtracted term and lowers the value for free. It is also in gene-averaged expression
    units, hence panel-dependent and not comparable across datasets. It exists so the
    numerator of `expr_mse_unbiased_capped_norm` is auditable, and so the pre-#247 column is
    readable again.
    """
    _refuse_adata_under_bulk_lognorm(comparator, "expr_mse_unbiased",
                                     pred_bulk=pred_bulk, real_bulk=real_bulk)
    pred_bulk, real_bulk = _resolve_bulks(pred, real, pred_bulk, real_bulk, pert_col)
    return _numerator(pred_bulk, real_bulk, pred_moments, real_moments, driver, control,
                      cap=False, comparator=comparator, genes=genes,
                      target_gene_map=target_gene_map)


def mse_unbiased_capped(
    pred: ad.AnnData | None = None,
    real: ad.AnnData | None = None,
    *,
    comparator: str,
    pred_bulk: tuple[np.ndarray, np.ndarray] | None = None,
    real_bulk: tuple[np.ndarray, np.ndarray] | None = None,
    pred_moments=None,
    real_moments=None,
    driver: str | None = None,
    pert_col: str = "target",
    control: str = "non-targeting",
    genes=None,
    target_gene_map: dict[str, str] | None = None,
    pred_bulk_full: tuple[np.ndarray, np.ndarray] | None = None,
) -> dict[str, float]:
    """`expr_mse_unbiased` with the prediction's correction BOUNDED twice -- per row by the
    reference's own (#247) and in TOTAL by the submission's across-perturbation spread (#348) --
    averaged over the gene set it SCORES.

        G_p = G - {g_p}   (issue #172: the perturbation's own gene, when its label resolves)

        expr_mse_unbiased_capped_p = ( ||mu_pred - mu_real,p||^2_{G_p}
                                       - r * min(tr Sigma_pred/n_pred,
                                                 k * tr Sigma_real,p/n_real)
                                       - tr Sigma_real,p/n_real ) / |G_p|

        r = min(1, B_pred / sum_q w_q min(tr Sigma_pred/n_pred, k tr Sigma_real,q/n_real))

        B_pred = sum_g sum_{p in R_g} w_p (mu_pred[p,g] - wmean_{R_g} mu_pred[.,g])^2

    -- the submission's across-perturbation centred sum of squares over the WHOLE pred panel
    wherever a driver supplies it (`_whole_panel_budget`, which is what makes the value
    partition-independent) and over the rows this call scores otherwise, WEIGHTED
    per row by `w_p = 1/|G_p|` on both sides so the comparison is in the units the metric reports,
    with `R_g` the rows that SCORE gene `g` (i.e. gene `g`'s own perturbation dropped, exactly as
    `_drop_on_target` drops it from the distance) and the centring weighted over that same set.
    See `_across_pert_budget` and `_row_weights`.

    ⚠️ **Both oracle fixed points now carry the condition `r = 1`**, not just the no-skill one. A
    truth-matched SAMPLED prediction on a binding panel retains `(1 - r) * C_pred` in expectation,
    so "a perfect prediction has expected numerator 0" holds where the panel does not bind -- which
    is where an accurate submission sits (the replicate anchor's budget is 1.54 / 1.38 / 1.58x its
    claim on the official val A / B / C panels, measured at the anchor's own half depth), but
    it is a property of the panel rather than a guarantee. That is the second residual recorded
    below, seen from the honest side.

    The factor `r` is issue #348, and it is what closes the OPEN paragraph at the bottom of this
    docstring rather than leaving it to the cap. Read it there first: #247's cap bounds the
    correction at a CONSTANT `k * C_real,p` and SATURATES, and a submission whose per-cell scatter
    cancels in the aggregate collects that constant against a plug-in distance containing none of
    it. MEASURED on the official val panels, that was worth **0.72-0.90 of this member's [0, 1]
    range** and needed no adversarial content: pinning the per-(p, g) sums of an honest
    control-paste -- same profile, same 400 cells, same depth -- moved it from 0.0000 to 0.9031
    `from_baseline`, and a live dev-leaderboard submission took +0.1389 of a 0.2295 OVERALL.

    What is bounded is the TOTAL. `V_pred` is the submission's own across-perturbation centred sum
    of squares (`_across_pert_budget`), a CONSERVATIVE proxy for `sum_p trVar(mu_hat_p)` --
    biological signal only ADDS, but the estimator is also `1/n_g` short of what a flat panel is
    strictly owed, so it is deliberately NOT an upper bound; when the panel claims more than that,
    EVERY row is scaled by
    `V_pred / claim` -- proportional, so uneven cells per perturbation (which the competition rules
    permit) are not clipped high-row-first, and there is no tolerance multiplier because any slack
    is a rebate the submitter can farm (see `PRED_CORRECTION_BUDGET_FLAG_RATIO`). Under it the
    gamed arms return to the correct ~1.0 "predicted the control" value while the baseline
    (`pred_tn` exactly 0) and the replicate anchor (budget 1.54 / 1.38 / 1.58x its claim on val
    A / B / C, at its own depth -- real biology across perturbations) are measured UNCHANGED. That last property is the whole reason the correction is
    BOUNDED rather than dropped or re-estimated: see `_numerator` for the two candidates measured
    to invert the scale, and for what this form does NOT close.

    ⚠️ **EXCLUDES the perturbed gene's own column (issue #172, ruled 2026-08-17)**, so the sum
    runs over `G - 1` genes and the divisor is `G - 1` on every perturbation whose label resolves
    to a measured gene. `expr_distance_unbiased` -- the DENOMINATOR of
    `expr_mse_unbiased_capped_norm` -- drops the same gene, which is what keeps "1.0 = predicted
    the control" exact. See `_drop_on_target` for the mechanism, for the one approximation it
    makes (the sampling corrections stay whole) and for what that approximation was measured to
    cost.

    MEASURED, all three official val contexts, from the cached moments the official bundles were
    built from: the adversary #172 describes -- predict the control everywhere, predict the truth
    at your own target gene -- was taking **10.21% / 11.30% / 6.07%** of this member's [0, 1]
    range for free. One gene in 18,533 is worth that much because it is the single largest-moving
    gene in 57-66% of perturbations AND because the jackknife consumes 46-55% of the raw squared
    distance, roughly halving the denominator the gene's share is read against; an a-priori `1/G`
    estimate is wrong by three orders of magnitude. The other direction, algebraic from the same
    numbers: exclusion removes that term from the DENOMINATOR too, so an honest submission's raw
    value rises by `1/(1 - gift)` = x1.11 / x1.13 / x1.06 (worse -- lower is better), and both
    ends of the scaled score move with it.

    The cap (issue #247, `PRED_TRACE_CAP_K`) replaces an unverified honesty assumption with a
    bound the submitter cannot move. It is a one-sided truncation, so this is bias-corrected
    rather than unbiased wherever it binds -- and where it binds, a NO-SKILL submission reads
    ABOVE the 1.0 no-skill point of `expr_mse_unbiased_capped_norm` rather than at it:
    `E[.] = ||Delta||^2 + max(0, tr Sigma_pred/n_pred - k tr Sigma_real,p/n_real)`.

    So the 1.0 point holds for exactly one condition -- `tr Sigma_pred/n_pred <= k tr
    Sigma_real,p/n_real`, i.e. wherever the cap does not bind. That is NOT the same as
    "emitted at the reference's depth": the quantity is a correction, so MORE DISPERSED cells
    breach it at an identical cell count just as fewer cells do. Measured on the synthetic
    anchor panel, no-skill, at n_real = 200 / 500 / 2000 --
    n_pred = n_real/10: 8.11 / 3.79 / 1.68; same n_pred with 2x the dispersion:
    3.34 / 1.91 / 1.22. That is the cap doing its job (the submission is claiming a correction
    the reference does not earn), not a broken anchor. This is the NUMERATOR of
    `expr_mse_unbiased_capped_norm`; it is unscored on its own because it is in gene-averaged
    expression units and therefore panel-dependent.

    ⚠️ Those anchor-panel numbers are PRE-#264, measured under the per-cell `lognorm`
    comparator -- see the banner on `mse_unbiased`. Under `bulk_lognorm` the bounded quantity is
    a jackknife, whose own bias depends on `bulk_target_sum` (#268) -- and a cap on a biased
    estimator inherits that bias.

    ⚠️⚠️ This banner USED to say "the cap's ALGEBRA is comparator-agnostic, so the mechanism
    above is unchanged; the levels are not". Everything above is stated IN EXPECTATION, from
    `E||mu_hat - mu||^2 = ||Delta||^2 + C_pred + C_real` with TRUE variance traces -- a
    decomposition that needs independence, not linearity, so it does survive the move to a
    nonlinear bulk statistic. It is an ORACLE expression: what the metric subtracts is the
    ESTIMATE `min(C_pred_hat, k C_real_hat)`, whose expectation is not the min of the
    expectations. #278 measured estimate and estimand coming apart pointwise:

      * DISPERSION. Under `bulk_lognorm` the pseudobulk reads the group TOTAL only. Hold every
        per-(p, g) sum fixed and redistribute counts across a group's cells: the realized
        `||mu_hat - mu||^2` CANNOT move, while the delete-1 jackknife CAN (a permutation of a
        group's rows leaves it algebraically unchanged, up to reduction order; #278's layouts
        do move it). Where it does, the reported metric
        value is NON-INCREASING in `min(C_pred_hat, k C_real_hat)` -- strictly decreasing below
        the cap, exactly constant at or above it -- so, lower being better, an UNDER-dispersed
        submission reads WORSE, the opposite sign to the paragraph above, with no gradient past
        saturation.
        What is specific to `bulk_lognorm` is that fixed raw group totals AUTOMATICALLY fix the
        pseudobulk; under `lognorm` fixed totals did not guarantee a fixed per-cell mean.
        The under-dispersion penalty is INTENDED (Alex, 2026-08-15) -- a group of bit-identical
        cells is not a cell population.
      * DEPTH. "Fewer cells" is a tendency under a matched i.i.d. emission model, not a property
        of the realized cell count: the binding condition is `C_pred_hat > k C_real_hat` -- an
        inequality on the ESTIMATES -- and cell count alone does not determine `C_pred_hat`. One
        cell carrying the whole group total scores the same as many proportional cells carrying
        it, and the `n < 2 => 0` policy gives that submission `C_pred_hat = 0` -- no correction
        at all rather than a capped one.
      * WAS OPEN, NOW BOUNDED. `C_pred_hat` is an EMPIRICAL estimator assuming the group's cells
        are exchangeable draws, and nothing enforces that, so a degenerate layout inflates the
        estimate without inflating the realized error. That was #294, parked until after the
        competition on the reading that #247's cap bounded the damage -- "a cliff, not a slope".
        #278 measured `C_pred_hat/G` of 0.0168 and 45.02 scoring bit-identically (at the retired
        `bulk_target_sum` of 1e6, so treat those levels accordingly), and the absence of a
        gradient past saturation was taken as the bound.
        ⚠️ That reading missed the other half, and #348 measured it: a flat deduction is only
        harmless if the plug-in distance CONTAINS comparable own-noise. Saturating the cap and
        pinning the aggregate are INDEPENDENT moves, and doing both makes the deduction pure
        gain -- and because the member's denominator is fixed by the real data, a constant gain
        is a fixed FRACTION of its range. Hence the third `min` term above. #294's own preferred
        direction -- estimate `C_pred` from a sampling model at the submitted depth instead of
        from the jackknife -- does not close this: it removes the manipulability of the estimate,
        but a pinned arm needs no manipulated estimate, since it submits the reference's own cell
        count at the reference's own depth and a nominal estimate lands where the cap already is.
    """
    _refuse_adata_under_bulk_lognorm(comparator, "expr_mse_unbiased_capped",
                                     pred_bulk=pred_bulk, real_bulk=real_bulk)
    pred_bulk, real_bulk = _resolve_bulks(pred, real, pred_bulk, real_bulk, pert_col)
    return _numerator(pred_bulk, real_bulk, pred_moments, real_moments, driver, control,
                      cap=True, comparator=comparator, genes=genes,
                      target_gene_map=target_gene_map, pred_bulk_full=pred_bulk_full)


def distance_unbiased(
    pred: ad.AnnData | None = None,
    real: ad.AnnData | None = None,
    *,
    comparator: str,
    pred_bulk: tuple[np.ndarray, np.ndarray] | None = None,
    real_bulk: tuple[np.ndarray, np.ndarray] | None = None,
    pred_moments=None,
    real_moments=None,
    driver: str | None = None,
    pert_col: str = "target",
    control: str = "non-targeting",
    genes=None,
    target_gene_map: dict[str, str] | None = None,
) -> dict[str, float]:
    """Sampling-corrected squared distance from each real perturbation to the real control,
    averaged over the gene set it SCORES -- UNBIASED where `C` is, which is the analytic
    `lognorm` branch; see the warning below for the `bulk_lognorm` jackknife.

        G_p = G - {g_p}   (issue #172: the perturbation's own gene, when its label resolves)

        expr_distance_unbiased_p = ( ||mu_real,p - mu_real,ctrl||^2_{G_p}
                                     - tr Sigma_real,p/n_real
                                     - tr Sigma_real,ctrl/N_ctrl ) / |G_p|

    The DENOMINATOR of `expr_mse_unbiased_capped_norm`: "how much real effect there was to
    get right". `E[.] = ||mu_real,p - mu_real,ctrl||^2 / G` with no dependence on either
    side's cell count -- which is what puts that metric's no-skill point at 1.0 on every
    panel.

    ⚠️ THAT EXPECTATION IS EXACT ONLY WHERE `C` IS. Under `lognorm`, `tr Sigma-hat/n` is an
    exact unbiased estimator of a per-cell mean's sampling term and the argument is airtight.
    Under `bulk_lognorm` the subtracted term is a delete-1 jackknife, which carries the usual
    Efron-Stein upward bias -- MEASURED at 0.32% at the shipped `bulk_target_sum=5e4`, rising
    to 2.06% at the retired 1e6 (#268). So the 1.0 point is exact for the ESTIMATOR and
    approximate for the ESTIMATE: at 1e6, on the real panel that measures the bias, "predict
    the control" read 1.073 rather than 1.0 -- the drift that motivated the move to 5e4. The
    structural claim -- that the denominator does not carry a
    depth-dependent noise term the way the plug-in it replaced did -- holds either way. The plug-in it replaces carried
    `tr Sigma_p/n_real + tr Sigma_ctrl/N_ctrl`, so a no-skill submission read `1 - noise/D`:
    measured 0.7643 on VCC Test, 0.2386 on `CCL_2`, 0.2754 on `H1_CGS` (issue #257). That 1.0
    is a property of the DENOMINATOR and holds whatever the reference's depth; the numerator's
    #247 cap can still push a no-skill submission above it -- see `mse_unbiased_capped`.

    ⚠️ **EXCLUDES the perturbed gene's own column (issue #172, ruled 2026-08-17)**, matching
    `mse_unbiased_capped` gene-for-gene -- so the sum above runs over `G - 1` genes and the `/ G`
    in it is `G - 1` on every perturbation whose label resolves, `G` on the rest (the divisor is
    per row, not one shared constant). Both legs of the derived metric must drop the same gene
    or the "1.0 = predicted the control" normalization stops holding -- with only the numerator
    excluding, a submission that predicted the control exactly would read below 1.0 by the
    on-target term alone. Exclusion also makes this quantity SMALLER, and by the largest-moving
    gene: measured on the official val contexts it removes 5.5% / 5.0% / 3.2% of the raw summed
    distance, and it pushes 0 / 3 / 8 of 300 perturbations' per-perturbation values below zero
    (from 0 / 0 / 1). That is the same "routinely negative and correct" property as below, in a
    slightly larger population; the metric aggregates as `ratio_of_sums`, so a negative row is a
    smaller contribution to one sum rather than a divide-by-near-zero.

    ⚠️ SUBMISSION-INDEPENDENT. It reads only the real side, so it is identical for every
    submission on a panel and is cacheable once per panel. It is in the results frame because
    the derived metric's arithmetic has to be auditable, not because it grades anything.

    ⚠️ ROUTINELY NEGATIVE, and that is CORRECT. It is an unbiased estimator of a
    non-negative quantity, so it must go below zero when the truth is near zero -- otherwise
    it would be biased upward. A negative value means the perturbation's mean shift is not
    resolvable at this depth by this isotropic statistic. Measured: 7.7% of `CCL_2`, 2.2% of
    `H1_CGS`, 0% of VCC Test. It does NOT mean the perturbation is null -- there is no
    calibrated null here, the comparison is isotropic while the noise is not, and the
    crossing perturbations are not DE-silent (up to 7 and 9 significant genes at FDR<0.05).
    Do not filter perturbations on it.

    ⚠️ Every rate and level above is PRE-#264, measured under the per-cell `lognorm`
    comparator -- see the banner on `mse_unbiased`. The 1.0 argument itself is comparator-free
    (it needs only `E[C] = ` the sampling term the distance carries), but the negative-row
    rates are not: how often this crosses zero is set by how large `C` is relative to the real
    effect, and #268 measured only 25% of the denominator surviving its own correction at the
    retired `bulk_target_sum=1e6` -- the shipped 5e4 leaves far more of it standing, so the
    rates move again. Re-measure them before quoting them for a v2 counts run.
    """
    # ⚠️ `pred`, `pred_bulk` and `pred_moments` are accepted for dispatch compatibility
    # (run.py builds kwargs from the signature) and are DELIBERATELY NEVER READ. Do not
    # "helpfully" align to the prediction here -- see `_real_rows`. `driver` reaches the
    # missing-moments error and NOTHING else: naming the driver is what makes that error
    # actionable, and it is the same context `_aligned_pair` already reports (Copilot, #262).
    _refuse_adata_under_bulk_lognorm(comparator, "expr_distance_unbiased",
                                     real_bulk=real_bulk)
    real_bulk = _resolve_real_bulk(real, real_bulk, pert_col)
    got = _real_rows(
        real_bulk,
        real_moments,
        control,
        "expr_distance_unbiased",
        comparator=comparator,
        driver=driver,
    )
    if got is None:
        return {}
    rows, real_means, real_tn, labels, ci = got
    n_genes = real_means.shape[1]
    if n_genes == 0:
        return {labels[i]: float("nan") for i in rows}
    # Issue #172, the DENOMINATOR leg. Both legs of `expr_mse_unbiased_capped_norm` must drop
    # the same gene or "1.0 = predicted the control" stops holding. Real-side rows only, so the
    # gate set IS the scored set -- there is no shard/panel distinction to make here.
    excl = _exclusion_cols([labels[i] for i in rows], genes,
                           target_gene_map=target_gene_map, gate_labels=None,
                           who="expr_distance_unbiased", n_genes=n_genes)
    ctrl_means = np.broadcast_to(real_means[ci], real_means[rows].shape)
    ctrl_tn = np.full(rows.shape, real_tn[ci], dtype=np.float64)
    summed = unbiased_sq_dist(real_means[rows], ctrl_means, real_tn[rows], ctrl_tn)
    # `ctrl_means` is a broadcast VIEW of one row, so the on-target correction reads the control
    # through its own index `ci` rather than through a per-row index into the view.
    summed, n_row = _drop_on_target(summed, real_means, rows, real_means,
                                    np.full(rows.shape, ci, dtype=np.intp), n_genes, excl)
    vals = _per_gene(summed, n_row)
    return {labels[i]: float(v) for i, v in zip(rows, vals)}




def _delta_eval(pred_bulk, real_bulk, reduce, *, control, control_source):
    """Apply `reduce(Δpred, Δreal)` per non-control perturbation.

    Δreal = real_pert − real_ctrl (always the real control). Δpred = pred_pert − ctrl,
    where ctrl is the pred control (`control_source="pred"`, upstream within-realm) or
    the real control (`control_source="real"`). Control excluded; every predicted
    perturbation must appear in real_bulk.
    """
    if control_source not in ("pred", "real"):
        raise ValueError(f"control_source must be 'pred' or 'real', got {control_source!r}")
    pred_perts = np.asarray(pred_bulk[0]).astype(str)
    real_perts = np.asarray(real_bulk[0]).astype(str)
    pred_means = np.asarray(pred_bulk[1], dtype=np.float64)
    real_means = np.asarray(real_bulk[1], dtype=np.float64)
    real_index = {p: i for i, p in enumerate(real_perts)}

    def control_mean(perts, means, side):
        hits = np.flatnonzero(perts == control)
        if hits.size == 0:
            raise ValueError(f"control {control!r} not found in {side} perturbations")
        return means[hits[0]]

    real_ctrl = control_mean(real_perts, real_means, "real")
    pred_ctrl = (
        control_mean(pred_perts, pred_means, "pred")
        if control_source == "pred"
        else real_ctrl
    )

    out: dict[str, float] = {}
    for i, p in enumerate(pred_perts):
        if p == control:
            continue
        if p not in real_index:
            raise ValueError(
                f"perturbation {p!r} present in pred_bulk but missing from real_bulk; "
                "every predicted perturbation must appear in real_bulk"
            )
        d_pred = pred_means[i] - pred_ctrl
        d_real = real_means[real_index[p]] - real_ctrl
        out[p] = reduce(d_pred, d_real)
    return out


def mae_delta(
    pred: ad.AnnData | None = None,
    real: ad.AnnData | None = None,
    *,
    pred_bulk: tuple[np.ndarray, np.ndarray] | None = None,
    real_bulk: tuple[np.ndarray, np.ndarray] | None = None,
    pert_col: str = "target",
    control: str = "non-targeting",
    control_source: str = "pred",
) -> dict[str, float]:
    """MAE between the predicted and real perturbation-control deltas, per perturbation."""
    pred_bulk, real_bulk = _resolve_bulks(pred, real, pred_bulk, real_bulk, pert_col)
    return _delta_eval(pred_bulk, real_bulk, safe_mae, control=control, control_source=control_source)


def mse_delta(
    pred: ad.AnnData | None = None,
    real: ad.AnnData | None = None,
    *,
    pred_bulk: tuple[np.ndarray, np.ndarray] | None = None,
    real_bulk: tuple[np.ndarray, np.ndarray] | None = None,
    pert_col: str = "target",
    control: str = "non-targeting",
    control_source: str = "pred",
) -> dict[str, float]:
    """MSE between the predicted and real perturbation-control deltas, per perturbation."""
    pred_bulk, real_bulk = _resolve_bulks(pred, real, pred_bulk, real_bulk, pert_col)
    return _delta_eval(pred_bulk, real_bulk, safe_mse, control=control, control_source=control_source)


def pearson_delta(
    pred: ad.AnnData | None = None,
    real: ad.AnnData | None = None,
    *,
    pred_bulk: tuple[np.ndarray, np.ndarray] | None = None,
    real_bulk: tuple[np.ndarray, np.ndarray] | None = None,
    pert_col: str = "target",
    control: str = "non-targeting",
    control_source: str = "pred",
) -> dict[str, float]:
    """Pearson correlation between the predicted and real perturbation-control deltas."""
    pred_bulk, real_bulk = _resolve_bulks(pred, real, pred_bulk, real_bulk, pert_col)
    return _delta_eval(pred_bulk, real_bulk, safe_pearson, control=control, control_source=control_source)
