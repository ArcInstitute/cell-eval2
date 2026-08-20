from __future__ import annotations

import anndata as ad
import numpy as np

from ..moments import trace_over_n_for, unbiased_sq_dist
from ..prep import pseudobulk
from ..safety import safe_mae, safe_mse, safe_pearson


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


def _resolve_bulks(pred, real, pred_bulk, real_bulk, pert_col):
    """Return (pred_bulk, real_bulk), computing pseudobulk from AnnData for whichever
    side's bulk is absent. Each side is resolved independently: a supplied bulk is reused
    as-is (never recomputed), and only the missing side requires its AnnData (Gemini
    impl review — supports hybrid input and avoids redundant pseudobulk)."""
    if pred_bulk is None:
        if pred is None:
            raise ValueError("provide pred AnnData or a pred_bulk tuple")
        pred_bulk = pseudobulk(pred, pert_col)
    if real_bulk is None:
        if real is None:
            raise ValueError("provide real AnnData or a real_bulk tuple")
        real_bulk = pseudobulk(real, pert_col)
    return pred_bulk, real_bulk


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


def mse_unbiased(
    pred: ad.AnnData | None = None,
    real: ad.AnnData | None = None,
    *,
    pred_bulk: tuple[np.ndarray, np.ndarray] | None = None,
    real_bulk: tuple[np.ndarray, np.ndarray] | None = None,
    pred_moments=None,
    real_moments=None,
    driver: str | None = None,
    pert_col: str = "target",
    control: str = "non-targeting",
) -> dict[str, float]:
    """Sampling-bias-corrected `expr_mse`, per perturbation (issue #198).

    `expr_mse` is the plug-in estimator of the squared distance between two POPULATION
    means, each estimated from a finite sample, so it carries an additive inflation that
    depends only on sampling:

        E[expr_mse] = (1/G) * ( ||mu_pred - mu_real||^2
                                + tr Sigma_pred/n_pred + tr Sigma_real/n_real )

    Subtracting it is exact, not a shrinkage. `expr_mse` uses `safe_mse` (a mean over genes),
    so the whole correction carries the same 1/G -- which is why the primitive returns the
    SUMMED quantity and the division happens here.

    The result is SIGNED: an unbiased estimator of a non-negative quantity is negative
    roughly half the time when the truth is near zero. That is correct behaviour. It is also
    why the catalog entry -- which is now SCORED, with `direction="lower"` and `anchor=0.0` --
    carries an explicit `clamp_high`: for a signed metric the anchor does not bound the score
    from above. See the catalog comment on that entry.

    `pred_moments`/`real_moments` are `GroupMoments` spanning ALL groups, control included.
    They are required: a driver that cannot supply them must make this raise rather than
    silently return the biased value.
    """
    pred_bulk, real_bulk = _resolve_bulks(pred, real, pred_bulk, real_bulk, pert_col)
    if pred_moments is None or real_moments is None:
        missing = " and ".join(
            n for n, v in (("pred", pred_moments), ("real", real_moments)) if v is None
        )
        raise ValueError(
            f"expr_mse_unbiased requires per-group moments; the {missing} side(s) supplied "
            f"none. Driver: {driver or 'unknown (moments were not routed to this metric)'}. "
            "That driver must run its pseudobulk with with_moments=True, or the metric must "
            "not be requested on it. This never falls back to the biased value."
        )
    pred_perts, pred_means = pred_bulk
    real_perts, real_means = real_bulk
    real_index = {str(p): i for i, p in enumerate(real_perts)}
    # tr Sigma_hat/n aligned to each side's OWN bulk rows; the moments may span more groups.
    pred_tn = trace_over_n_for(pred_moments, pred_perts, pred_means)
    real_tn = trace_over_n_for(real_moments, real_perts, real_means)

    # Reindex the real side onto the pred row order, then one vectorized primitive call.
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
        return {}
    keep = np.asarray(keep, dtype=np.intp)
    rows = np.asarray(rows, dtype=np.intp)
    pred_means = np.asarray(pred_means, dtype=np.float64)
    real_means = np.asarray(real_means, dtype=np.float64)
    n_genes = pred_means.shape[1]
    if n_genes == 0:  # matches safe_mse's empty-input contract (NaN, not a divide error)
        return {str(pred_perts[i]): float("nan") for i in keep}
    # unbiased_sq_dist returns the SUMMED quantity; safe_mse is a mean over genes, so /G here.
    vals = unbiased_sq_dist(pred_means[keep], real_means[rows],
                            pred_tn[keep], real_tn[rows]) / n_genes
    return {str(pred_perts[i]): float(v) for i, v in zip(keep, vals)}


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
