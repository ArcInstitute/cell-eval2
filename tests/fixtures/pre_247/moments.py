"""Per-group second moments and the primitives built on them.

The sufficient statistic for the sampling-bias correction in issue #198 is one fp64 scalar
per group, ``sumsq[p] = Σᵢ ‖xᵢ‖²``, alongside the per-group cell count every pseudobulk
driver already accumulates. From those two::

    tr Σ̂_p = ( sumsq[p] − counts[p]·‖means[p]‖² ) / ( counts[p] − 1 )

Only the TRACE is ever needed, never the covariance matrix, so memory is O(P), not O(P·G).

This module lives at the top level rather than under ``metrics/`` because ``prep``,
``streaming_bulk``, ``gpu.bulk`` and ``cache`` all need :class:`GroupMoments` and none of
them may import from ``metrics/``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class GroupMoments:
    """Per-group cell counts and Σ‖x‖², in one normalization's own space.

    ``perts`` carries the labels so alignment against a pseudobulk is CHECKABLE rather than
    assumed. Moments span ALL groups, including the control, and must never be restricted to
    a chosen perturbation subset: the control's trace is needed after the control row has
    been dropped from the bulk (see the spec's §4.6 invariant).
    """

    perts: np.ndarray   # [P] labels, same order as the bulk this was produced with
    counts: np.ndarray  # [P] fp64, cells per group
    sumsq: np.ndarray   # [P] fp64, Σᵢ ‖xᵢ‖² in this normalization's space

    def __post_init__(self) -> None:
        perts = np.asarray(self.perts)
        counts = np.asarray(self.counts, dtype=np.float64)
        sumsq = np.asarray(self.sumsq, dtype=np.float64)
        if perts.ndim != 1 or counts.ndim != 1 or sumsq.ndim != 1:
            raise ValueError(
                f"GroupMoments fields must be 1-D; got perts {perts.shape}, "
                f"counts {counts.shape}, sumsq {sumsq.shape}"
            )
        if not (perts.shape[0] == counts.shape[0] == sumsq.shape[0]):
            raise ValueError(
                f"GroupMoments fields must have the same length; got perts {perts.shape[0]}, "
                f"counts {counts.shape[0]}, sumsq {sumsq.shape[0]}"
            )
        object.__setattr__(self, "perts", perts)
        object.__setattr__(self, "counts", counts)
        object.__setattr__(self, "sumsq", sumsq)


def trace_sigma(counts: np.ndarray, sumsq: np.ndarray, means: np.ndarray) -> np.ndarray:
    """``tr Σ̂_p`` per group, for arrays already aligned row-for-row.

    NaN where ``counts < 2`` (the sample covariance is undefined) rather than an inf or a
    divide-by-zero warning; real panels do contain single-cell perturbations.

    MIXED PRECISION, measured and accepted. ``sumsq`` is always fp64, but ``means`` may be
    fp32: ``gpu.bulk.GroupedMeanAccumulator.finalize`` casts to fp32, so the accumulator
    paths (``inmem_pseudobulk`` on either device, and the GPU streaming path) feed an
    fp32-rounded mean into ``sumsq - n‖μ‖²``, which is a near-cancellation. The ``prep`` and
    CPU-streaming paths feed fp64 means and are unaffected.

    Measured in ``log1p(CPM)`` on Poisson panels (6 groups, G=2000). The adverse axis is
    SMALL ``n`` -- the deviation scales roughly as ``1/(n-1)`` -- not how close pred and real
    are, which is worst-conditioned for the distance rather than for the trace::

        n            2         50        300
        |Δtr|/(nG)   3.3e-07   1.2e-09   1.1e-10     <- what reaches the metric, per side
        tr/(nG)      2.37      0.094     0.016       <- the correction term being subtracted

    So the fp32 mean perturbs the subtracted correction by at most ~1.5e-07 of its own size,
    and contributes at most ~3e-07 in absolute metric units at n=2 (far less at realistic n),
    against per-perturbation values that run 1e-03..1e-02 on those panels. It cannot move a
    conclusion drawn from this diagnostic.
    ``tests/test_moments.py::test_fp32_means_perturb_the_trace_only_negligibly`` pins both
    bounds as a regression tripwire. Exactness would need :class:`GroupMoments` to carry its
    own fp64 ``‖μ‖²`` rather than recomputing it from the bulk -- a dataclass AND
    cache-artifact change, worth folding into #202 if that family ever needs it.
    """
    counts = np.asarray(counts, dtype=np.float64)
    sumsq = np.asarray(sumsq, dtype=np.float64)
    means = np.asarray(means, dtype=np.float64)
    if means.ndim != 2 or means.shape[0] != counts.shape[0]:
        raise ValueError(
            f"means must be [P, G] aligned to counts [P]; got means {means.shape}, "
            f"counts {counts.shape}"
        )
    out = np.full(counts.shape, np.nan, dtype=np.float64)
    ok = counts >= 2
    if ok.any():
        sq_norm = np.einsum("ij,ij->i", means[ok], means[ok])
        out[ok] = (sumsq[ok] - counts[ok] * sq_norm) / (counts[ok] - 1.0)
    return out


def trace_over_n_for(moments: GroupMoments, perts, means: np.ndarray) -> np.ndarray:
    """``tr Σ̂_p / n_p`` for each label in ``perts``, looked up in ``moments`` by label.

    ``moments`` may span MORE groups than ``perts`` -- it always carries the control and is
    never restricted -- but every label in ``perts`` must be present in it. A missing label
    means some driver restricted the moments, which is the invariant this raises on.

    **``counts < 2`` yields 0.0, not NaN (issue #219).** ``trace_sigma`` is right to call the
    covariance UNDEFINED there; this function returns the CORRECTION TO SUBTRACT, and when
    there is no estimate available it deliberately subtracts ZERO. That is a fallback, not the
    estimator's answer: the true expected term at n=1 is ``tr Σ``, so subtracting nothing
    leaves this SUMMED primitive upward-biased by ``tr Σ`` and the metric, which carries the
    ``1/G``, upward-biased by ``tr Σ / G``. The distinction is the whole fix, which is why the
    policy lives here and the primitive stays honest.

    ``counts == 0`` takes the same branch and is NOT distinguished from ``counts == 1``,
    though a group with no cells has no sample mean either. It does not arise from
    ``prep.pseudobulk_with_moments`` (groups come from observed labels, so
    ``counts = np.diff(bounds) >= 1``), but it can arrive from a corrupt cache, a malformed
    stream, or a directly constructed :class:`GroupMoments`. The same branch also absorbs
    negative, fractional-below-two and NaN counts. All of it is SILENT -- malformed moments
    become a plausible finite number rather than an error. Making that raise is a change to
    the moments contract, deliberately not folded in here (issue #227).

    Returning NaN made the perturbation vanish: ``expr_mse_unbiased`` carries
    ``worst_value=None``, so ``run.py``'s no-drop fill skips it and ``aggregate_metrics``
    drops NaN before the mean. Since ``n_pred`` is chosen by the SUBMITTER -- predicted group
    sizes need not match the reference, only the perturbation set does -- a submission could
    emit a single cell for exactly the perturbations it predicted worst and have them leave
    its own aggregate. Measured on the old behaviour: thinning one badly-predicted group to
    one cell moved the aggregate 0.49733 -> 0.00012, a ~4000x improvement for DEGRADING the
    submission, while ``expr_mse`` correctly worsened 0.50060 -> 0.75930.

    Subtracting zero instead makes that a PENALTY IN EXPECTATION rather than a reward, and a
    self-calibrating one: with a single cell the estimator keeps the full sampling inflation
    ``E‖x̄ − μ‖² = tr Σ / n = tr Σ`` instead of removing it. Measured at ``SD = 0.7``,
    ``G = 2000``: the honest n=300 mean METRIC VALUE is 0.9978 and the thinned n=1 value is
    1.4863 -- a +0.488 expected penalty against a predicted ``SD² = 0.490``. (Values, not
    scores; ``clamp_high=1.0`` makes a score of 1.4863 impossible.) No invented constant, and
    no policy call about whether thin groups are legitimate: they are, they just cannot claim a
    correction they did not earn. A lucky single draw can still land near the reference -- the
    penalty holds on average, not per realization.

    SCOPE -- this closes the ``n < 2`` door, not the metric. The estimator stays UNBIASED for
    every ``n >= 2`` (its mean is flat at 0.996-0.998 from n=2 to n=300), so nothing
    bias-based replaces the dropped-perturbation route at that door. It does NOT make
    ``expr_mse_unbiased`` robust in general: ``tr Σ̂_pred/n_pred`` is still computed from the
    cells the submission emits, so reporting the same predicted mean through a more dispersed
    set of cells lowers the metric for free. That channel is separate from this one, and it is
    BOUNDED under input validation -- ``validate_input_type`` requires non-negative values (and
    ``EvalConfig.validate_input=False`` skips it, in which case the bound does not hold).
    NON-NEGATIVITY ALONE caps the per-gene correction at m² for every n >= 2 at a fixed
    per-gene mean m; at n < 2 it is the zero fallback above, not m². Putting all mass on one
    cell attains that ceiling, though that construction must still clear the rest of
    validation (``check_scale_limit`` may reject it); the ceiling holds either way. This
    function
    returns the SUMMED quantity, so the bound here is ``G*mean(m²)``; after ``mse_unbiased``'s
    1/G it is ``mean(m²)`` on the metric scale. Measured at a fixed predicted mean (G=2000,
    gamma panel, mean(m²)=0.4593): the correction goes 0.0015 -> 0.4593 and u only 24.9985 ->
    24.5407, where an unvalidated 2 cells at ±100 would give -9975. Generally
    ``u >= expr_mse - mean(m²) - C_real`` -- a LOWER BOUND, not an equality, since attaining it
    also needs the extremizer to clear ``check_scale_limit``. A large error is safe only
    RELATIVE to ``mean(m²)``. See ``docs/metrics.md`` §2.3. Unbiasedness assumes the emitted cells are an
    honest iid sample; cell_eval2 assumes it and does not verify it.

    A weak lottery also remains at small n, since the VARIANCE grows as n shrinks --
    best-of-400 draws improves 0.9929 at n=300 to 0.9216 at n=2, ~7% -- and it costs 400
    resamples against a reference the submitter cannot see. A minimum-n gate on the REAL side
    (the ``de_lfc_nmae`` pattern) would NOT close it: it keeps omission submission-independent
    but leaves ``n_pred``, which drives the lottery, unconstrained.
    """
    index = {str(p): i for i, p in enumerate(np.asarray(moments.perts))}
    rows = np.empty(len(perts), dtype=np.intp)
    for k, p in enumerate(perts):
        try:
            rows[k] = index[str(p)]
        except KeyError:
            raise ValueError(
                f"perturbation {str(p)!r} is present in the pseudobulk but absent from its "
                "moments; moments must span all groups and must never be restricted"
            ) from None
    counts = np.asarray(moments.counts, dtype=np.float64)[rows]
    sumsq = np.asarray(moments.sumsq, dtype=np.float64)[rows]
    tr = trace_sigma(counts, sumsq, means)
    # Divide only where the trace is defined; elsewhere subtract NOTHING (#219, above).
    # The fill is 0.0 rather than NaN: thinness alone no longer makes a group vanish -- it
    # forfeits the correction and is penalized in expectation (not on every draw).
    # `counts == 0` takes the same branch: it would otherwise be NaN/0, which happens to be
    # NaN without a warning on current numpy, but relying on that is fragile.
    out = np.zeros(tr.shape, dtype=np.float64)
    ok = counts >= 2
    out[ok] = tr[ok] / counts[ok]
    return out


def unbiased_sq_dist(pred_means, real_means, pred_trace_over_n, real_trace_over_n):
    """Unbiased ``‖μ_pred − μ_real‖²``, SUMMED over genes, per aligned row.

    The plug-in squared distance minus both sides' ``tr Σ̂/n`` terms. Summed, not
    gene-averaged, because #202's magnitude ``m_p`` is defined summed over genes; ``expr_mse``
    applies its own ``1/G`` at the call site (:func:`metrics.delta.mse_unbiased`).

    All four arguments must be row-aligned: ``pred_means``/``real_means`` are [n, G] and the
    two trace terms are [n]. NaN in either trace term still propagates, but ``n < 2`` is no
    longer such a case: :func:`trace_over_n_for` returns 0.0 there rather than NaN (issue
    #219), so a thin group now reaches the metric with an uncorrected -- hence upward-biased --
    finite value instead of vanishing from the aggregate.
    """
    pred_means = np.asarray(pred_means, dtype=np.float64)
    real_means = np.asarray(real_means, dtype=np.float64)
    if pred_means.shape != real_means.shape:
        raise ValueError(
            f"pred_means {pred_means.shape} and real_means {real_means.shape} must match"
        )
    d = pred_means - real_means
    return (np.einsum("ij,ij->i", d, d)
            - np.asarray(pred_trace_over_n, dtype=np.float64)
            - np.asarray(real_trace_over_n, dtype=np.float64))
