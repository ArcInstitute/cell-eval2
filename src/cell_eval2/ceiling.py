"""Data ceiling via a disjoint self-split + Spearman-Brown correction.

The ceiling estimates how high any model could score on a metric given the noise
inherent in the real data. It is an empirical reproducibility estimate rather than
a proved upper bound - it is measured on ONE random split, and a model that
denoises the reference better than a half of it does can in principle score above
it. It is computed from the REAL data only: each
perturbation's cells (and the control's) are split into two *disjoint* halves of
``floor(n/2)`` cells (no cell in both; the leftover cell of an odd ``n`` is
dropped; a perturbation with < 2 cells is dropped, and a control that cannot be
split is a fatal error), one half is treated as "real" and the other as "prediction", the
verified reliability metrics are scored on that self-split (reusing
:func:`cell_eval2.run.compute_metrics` with the caller's config, so
normalization / DE / device match the main run - except ``control_source``, see
Config below), and each metric's
per-context mean is mapped from half depth to the COMBINED split depth
``2 * floor(n/2)`` - the run's full depth for even ``n``, one cell short for odd
``n`` - with the analytical Spearman-Brown correction ``r' = 2r/(1+r)`` (applied
only for ``r > 0``; see :func:`_spearman_brown`). The classical formula assumes
parallel measurements with independent errors; for the ranking and set-overlap
metrics in :data:`SB_METRICS` it is an empirical extrapolation of the same
"agreement rises with depth" shape rather than a theorem.

Cost. The real matrix is loaded unbacked and held alongside two ``.copy()`` halves
whose rows sum to about another full matrix, so this phase's input footprint is
roughly ``2x`` the real matrix. When the selected SB metrics include DE ones, DE is
computed once per half. Via the CLI the phases are SEQUENTIAL - ``compute_metrics``
closes path-backed inputs before the ceiling reopens the real data - so a combined
``run --ceiling`` peaks at about the larger phase rather than their sum; a caller
passing in-memory AnnData keeps its own copy resident and the two do add up. Wall
time for a DE-bearing combined run often approaches ``2x`` the plain run.

Scope of the correction. Spearman-Brown is applied ONLY to the metrics in
:data:`SB_METRICS` - a hand-maintained list (deliberately NOT derived from
``MetricSpec.scoring``) of the reliability metrics whose cell_eval2
implementation was verified to compute the same quantity as the validated
cell-eval metric on which the ceiling was empirically justified. Every other metric
the run EMITS is reported as ``NaN`` (no defensible ceiling); a selected metric that
is recognized but not implemented is dropped by ``resolve_metrics`` and appears in
neither frame.

Config. The ceiling inherits the caller's config - normalization, DE backend,
device and metric selection all follow the main run - with one deliberate
exception: ``control_source`` is forced to ``"pred"``. Under ``"real"`` the pred
side's DE reference comes from the real side, so scoring ``pred=half_b`` against
``real=half_a`` computes BOTH halves' log2FCs against ``half_a``'s control. That
shares the control's sampling noise between the two quantities whose agreement is
being measured and biases the ceiling upward. Independent controls per
half are what make them replicates, so the override is not optional; see
:func:`compute_ceiling`.

Version caveat. The correction was validated on the cell-eval numbers, which
correspond to cell_eval2's *v1-equivalent* conventions (``control_source="pred"``,
PDS ``rank_denominator="n"``, ``nan_lfc_policy="keep"``, ``min_abs_log2fc=0``).
Under v2 defaults the SB metrics use the *same algorithm* but different
conventions (rank denominator, significance pre-filters, normalization target), so
their absolute values differ. The extrapolation is applied to them on the same
empirical footing rather than a separately validated one: those conventions
preserve the "agreement rises with depth" shape SB is used for, but were not
themselves checked against cell-eval numbers. Note the consequence of the ``control_source`` override: against a main
run that uses ``control_source="real"``, the ceiling is not measured under
identical conventions. That is the intended trade-off - a biased estimator is worse
than a convention mismatch, and an upward-biased ceiling understates every score
measured against it.
"""

from __future__ import annotations

import logging
import math
import os
from dataclasses import replace

import anndata as ad
import numpy as np
import polars as pl

from .catalog import _NAME_TO_CANONICAL, resolve_metrics
from .config import EvalConfig
from .io import load_anndata
from .run import (
    _TIDY_SCHEMA,
    _resolve_config,
    aggregate_metrics,
    compute_metrics,
    metric_output_names,
)

logger = logging.getLogger(__name__)

# Reliability metrics that receive the Spearman-Brown ceiling correction. Names are
# the wilcoxon-family canonical v2 names. Verified 1:1 against the validated
# cell-eval implementations (same computation, not just the same name).
#
# Deliberately EXCLUDED (do not add without re-verifying + re-validating):
#   * de_wilcoxon_pr_auc / de_wilcoxon_roc_auc - cell_eval2's pred p-value floor
#     (min_nonzero/replace_zero) differs from cell-eval's clip-to-1e-10, so the AUC
#     is a genuinely different number than the one we validated.
#   * error metrics (expr_mae/mse, expr_mse_unbiased_capped_norm, delta_mae/mse) and counts
#     (de_*_nsig_counts_*). expr_mse_unbiased_capped_norm is additionally signed, so it is not a
#     bounded reliability in the first place.
#   * v2-native chance-corrected metrics (overlap_adjusted, precision_adjusted,
#     sig_recall_adjusted, sig_mcc) - no cell-eval equivalent, never validated.
#   * model_direction_match and the signed pair lfc_spearman_pos/neg - these DO carry
#     v1 aliases, but they were not part of the validated set, so they are excluded
#     for the same "not verified 1:1" reason, not for being v2-native.
#   * the #187 direction metrics (direction_precision, direction_sensitivity,
#     direction_sensitivity_universe) - v2-native, no cell-eval equivalent, never
#     validated under doubling. Note direction_sensitivity_universe is documented
#     UNBOUNDED ABOVE, which Spearman-Brown's bounded-reliability assumption rules
#     out outright.
#   * deseq2-backend DE names (de_deseq2_*) - not verified; under that backend the
#     DE metrics are left uncorrected (NaN).
SB_METRICS: frozenset[str] = frozenset(
    {
        "delta_pearson",
        "pds_l1",
        "pds_l2",
        "pds_cosine",
        "de_wilcoxon_overlap",
        "de_wilcoxon_overlap_top50",
        "de_wilcoxon_overlap_top100",
        "de_wilcoxon_overlap_top200",
        "de_wilcoxon_overlap_top500",
        "de_wilcoxon_precision",
        "de_wilcoxon_precision_top50",
        "de_wilcoxon_precision_top100",
        "de_wilcoxon_precision_top200",
        "de_wilcoxon_precision_top500",
        "de_wilcoxon_nsig_spearman",
        "de_wilcoxon_lfc_spearman",
        "de_wilcoxon_direction_match",
        "de_wilcoxon_sig_recall",
    }
)


def _disjoint_halves(
    real: ad.AnnData, pert_col: str, control: str, seed: int
) -> tuple[ad.AnnData, ad.AnnData]:
    """Split ``real`` into two *disjoint* halves of ``floor(n/2)`` cells per perturbation.

    Each perturbation's cells (including the control's) are shuffled and split
    without replacement into two halves of ``floor(n/2)`` cells - so no cell
    appears in both. When ``n`` is odd the one leftover cell is discarded (both
    halves must be the same depth for the doubling to hold). Perturbations with
    < 2 cells cannot be split and are dropped from both halves. The resulting
    half depth is corrected back to the combined depth of the two halves,
    ``2 * floor(n/2)``, by the Spearman-Brown doubling in :func:`compute_ceiling`.
    """
    rng = np.random.default_rng(seed)
    a_idx: list[np.ndarray] = []
    b_idx: list[np.ndarray] = []
    dropped = 0
    for _pert, idx in real.obs.groupby(pert_col, observed=True).indices.items():
        perm = rng.permutation(np.asarray(idx))
        h = perm.size // 2
        if h < 1:
            dropped += 1  # < 2 cells: cannot form two disjoint halves
            continue
        a_idx.append(perm[:h])
        b_idx.append(perm[h : 2 * h])
    if not a_idx:
        raise ValueError("no perturbation has >= 2 cells to split for the ceiling")
    if dropped:
        logger.warning(
            "Ceiling: dropped %d perturbation(s) with < 2 cells (cannot be split); "
            "the ceiling mean is over the remaining %d perturbation(s), a different "
            "set than the main run's aggregate.",
            dropped,
            len(a_idx),
        )
    half_a = real[np.concatenate(a_idx)].copy()
    half_b = real[np.concatenate(b_idx)].copy()
    if control not in set(half_a.obs[pert_col].astype(str)):
        raise ValueError(
            f"control {control!r} has < 2 cells; cannot compute a disjoint-split ceiling"
        )
    return half_a, half_b


def _spearman_brown(agg_mean: pl.DataFrame, all_metrics: list[str]) -> pl.DataFrame:
    """Map per-metric half-depth means to combined-depth ceilings via Spearman-Brown.

    ``agg_mean`` is the :func:`aggregate_metrics` output (columns ``metric``,
    ``mean``) over the computed SB metrics. Returns a ``(metric, ceiling)`` frame
    covering EVERY metric in ``all_metrics``: ``2r/(1+r)`` for the computed SB
    metrics and ``NaN`` for every other metric, so the ceiling's coverage matches
    the main run.

    ``all_metrics`` holds the names as the run OUTPUTS them, which depend on the
    config (``version="v1"`` emits ``pearson_delta``, v2 emits ``delta_pearson``;
    the deseq2 backend emits ``de_deseq2_*``). Membership in :data:`SB_METRICS` is
    therefore tested on the *canonical* name via ``_NAME_TO_CANONICAL``, so a
    listed metric is corrected under every spelling instead of silently falling
    through to ``NaN`` because the label differed.

    ``2r/(1+r)`` is a reliability correction only for ``r > 0``. At ``r <= 0`` it
    stops being one: the formula's pole is at ``r == -1`` (division by zero), just
    short of it the output is nonsense (``r = -0.9 -> -18``), and nowhere in that
    range is there a reliability to extrapolate. The sign-unbounded SB metrics
    (``delta_pearson`` and the two Spearman metrics) can land there on a small or
    degenerate context. A non-positive split-half reliability is no positive
    evidence of repeatability, i.e. there is no defensible ceiling, so those are
    reported as ``NaN`` (never a negative "ceiling" worse than any achievable
    score, never a raise).
    """
    means = dict(zip(agg_mean["metric"].to_list(), agg_mean["mean"].to_list()))
    ceilings: list[float] = []
    for m in all_metrics:
        v = means.get(m)
        canon = _NAME_TO_CANONICAL.get(m, m)  # v1/deseq2 spelling -> canonical identity
        if canon in SB_METRICS and v is not None and math.isfinite(v) and v > 0.0:
            ceilings.append(2.0 * v / (1.0 + v))
        else:
            ceilings.append(float("nan"))
    return pl.DataFrame(
        {"metric": all_metrics, "ceiling": ceilings},
        schema={"metric": pl.Utf8, "ceiling": pl.Float64},
    )


def compute_ceiling(
    real: ad.AnnData | str | os.PathLike,
    *,
    config: EvalConfig | None = None,
    seed: int = 0,
    **overrides,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Estimate a per-metric data ceiling from the real data alone.

    See the module docstring for the method. ``real`` is an AnnData or a path;
    ``pred`` plays no role (the ceiling is a property of the real data). Returns
    ``(results, agg_ceiling)``:

      * ``results`` - tidy per-perturbation self-split scores ``(perturbation,
        metric, value)``, for the verified reliability metrics only.
      * ``agg_ceiling`` - ``(metric, ceiling)``: the Spearman-Brown-corrected value
        per verified metric, ``NaN`` for every other selected metric.

    The caller's config is inherited except for four overrides on the inner
    half-split run: ``cache_real``/``cache_pred`` and ``outdir`` are cleared so it
    cannot overwrite the caller's cache or ``run_params.yaml``, and
    ``control_source`` is forced to ``"pred"`` so each half uses its OWN control
    cells - under ``"real"`` both halves would share one control, inflating the
    ceiling. Neither writes anything: the two frames are returned, not persisted.
    """
    cfg = _resolve_config(config, overrides)
    real_ad = load_anndata(real, backed=False)  # need it in memory to split

    available, _missing = resolve_metrics(cfg.metrics, version=cfg.version)
    deseq2 = getattr(cfg.de, "backend", None) == "deseq2"
    # Only the verified SB metrics the caller selected; under the deseq2 backend the
    # DE metrics are renamed de_deseq2_* (unverified) -> leave them out (they become NaN).
    sb_run = [
        m
        for m in available
        if m in SB_METRICS and not (deseq2 and m.startswith("de_wilcoxon_"))
    ]

    half_a, half_b = _disjoint_halves(real_ad, cfg.pert_col, cfg.control, seed)
    if sb_run:
        # Four deliberate overrides on the half-split run. The first two stop the inner
        # run from writing over artifacts that belong to the CALLER's run; the third is
        # required for the estimate to be valid at all.
        #
        # cache_real/cache_pred: the halves can never hit the caller's cache anyway (the
        #   fingerprint includes n_obs + the per-cell label assignment), and leaving them
        #   set would let CacheStore.put overwrite the manifest with the half-data
        #   artifacts AND os.remove the caller's full-data files - destroying a prebuilt
        #   cache.
        # outdir: compute_metrics writes run_params.yaml into cfg.outdir unconditionally,
        #   so an inherited outdir makes this inner run clobber the main run's provenance
        #   with the ceiling's own config (metrics narrowed to sb_run, caching disabled).
        #   Nothing is lost by dropping it - callers write the ceiling frames themselves.
        # control_source: MUST be "pred" here, whatever the caller uses. Under "real" the
        #   pred side's DE takes its reference from the real side, so with pred=half_b and
        #   real=half_a BOTH halves' log2FCs are computed against half_a's control. Their
        #   sampling noise is then shared rather than independent, which correlates the two
        #   quantities whose agreement is being measured and biases split-half reliability
        #   upward. Measured on a small self-split: lfc_spearman 0.54 -> 0.74, and
        #   nsig_spearman went from NaN (no defensible ceiling) to a confident-looking 0.94,
        #   i.e. the shared control can manufacture reliability where there is none and
        #   defeat the r > 0 guard. Under "pred" each half uses its own control cells, which
        #   is what makes them independent replicates - and the convention Spearman-Brown
        #   was validated against.
        results = compute_metrics(
            half_b,
            half_a,
            config=replace(
                cfg,
                metrics=sb_run,
                cache_real=None,
                cache_pred=None,
                outdir=None,
                control_source="pred",
            ),
        )
    else:
        results = pl.DataFrame(schema=_TIDY_SCHEMA)

    # Label the ceiling with the names the run OUTPUTS - one row per emitted metric, so
    # ceiling_agg lines up name-for-name with the main run's aggregate and many-to-one
    # onto its long-form results (under version="v1" that is `pearson_delta`, not the
    # canonical `delta_pearson`). metric_output_names is the run's own rule, which also
    # collapses a metric and an explicitly-selected deseq2 sibling to the ONE name the
    # run emits - hand-rolling it here would put a duplicate row in ceiling_agg.
    agg_ceiling = _spearman_brown(
        aggregate_metrics(results, metrics=sb_run), metric_output_names(cfg)
    )
    return results, agg_ceiling
