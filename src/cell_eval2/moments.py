"""Per-group second moments and the primitives built on them.

The sufficient statistic for the sampling-bias correction in issue #198 is one fp64 scalar
per group, ``sumsq[p] = Σᵢ ‖xᵢ‖²``, alongside the per-group cell count every pseudobulk
driver already accumulates. From those two::

    tr Σ̂_p = ( sumsq[p] − counts[p]·‖means[p]‖² ) / ( counts[p] − 1 )

Only the TRACE is ever needed, never the covariance matrix, so memory is O(P), not O(P·G).

⚠️ THAT IS THE ``lognorm`` HALF. Issue #264's ``bulk_lognorm`` comparator is a function of
the group SUM, whose sampling variance no single-pass sufficient statistic can express, so
it carries a **second** per-group scalar — ``jk``, the delete-1 jackknife of
:func:`jackknife_correction`, which costs a dense second pass. On that path ``sumsq``/
``counts`` are still accumulated but stay in **raw counts** space rather than the
comparator's, so they are NOT interchangeable with the bulk means and ``tr Σ̂`` over them is
not a correction for anything. :func:`correction_for` is the only supported way to ask for
"the correction": it dispatches on the comparator and refuses each artifact in the other's
slot rather than falling back. The artifact stays O(P) either way.

This module lives at the top level rather than under ``metrics/`` because ``prep``,
``streaming_bulk``, ``gpu.bulk`` and ``cache`` all need :class:`GroupMoments` and none of
them may import from ``metrics/``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.sparse import csr_matrix, issparse

#: The shipped `bulk_target_sum` (#268). ONE definition: `EvalConfig.bulk_target_sum` and every
#: low-level bulk entry point that takes the argument read it from here, so a future move
#: cannot leave some signatures on the old value -- which is exactly what #268's review found
#: after the first pass changed only `EvalConfig` (six signatures were still defaulting to the
#: retired 1e6, so a DIRECT caller silently got the old normalization while every production
#: driver got the new one).
#:
#: This module owns it because `jackknife_correction` below owns the measurement that sets it:
#: the bias sweep is the reason the value is 5e4 and not something else. `moments` imports
#: nothing from the package, so anything may import this without a cycle.
DEFAULT_BULK_TARGET_SUM: float = 50_000.0


@dataclass(frozen=True)
class GroupMoments:
    """Per-group cell counts, Σ‖x‖², and (on the group-sum path) the jackknife.

    ⚠️ NOT all in one space: under ``lognorm`` ``sumsq`` is in the comparator's own space
    and IS the correction's sufficient statistic; under ``bulk_lognorm`` it stays in raw
    counts, the correction is ``jk``, and mixing the two is what ``correction_for``
    refuses in both directions.

    ``perts`` carries the labels so alignment against a pseudobulk is CHECKABLE rather than
    assumed. Moments span ALL groups, including the control, and must never be restricted to
    a chosen perturbation subset: the control's trace is needed after the control row has
    been dropped from the bulk (see the spec's §4.6 invariant).
    """

    perts: np.ndarray   # [P] labels, same order as the bulk this was produced with
    counts: np.ndarray  # [P] fp64, cells per group
    sumsq: np.ndarray   # [P] fp64, Σᵢ ‖xᵢ‖² -- in the per-cell normalization's space under
                        # `lognorm`, and in RAW COUNTS space on the `bulk_lognorm` path,
                        # where the correction is `jk` and this is carried for diagnostics
    jk: np.ndarray | None = None   # [P] fp64 delete-1 jackknife C_p, bulk_lognorm only; None elsewhere

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
        if self.jk is not None:
            jk = np.asarray(self.jk, dtype=np.float64)
            if jk.ndim != 1 or jk.shape[0] != perts.shape[0]:
                raise ValueError(
                    f"GroupMoments fields must have the same length; got jk {jk.shape}, "
                    f"perts {perts.shape}"
                )
            object.__setattr__(self, "jk", jk)


def _loo_bulk(P, Y_block, r, bulk_target_sum, *, xp=np):
    """``v_ig - b_g`` for a block of cells: the leave-one-out bulk's DEVIATION from the
    full-sample bulk ``b_g = log1p(TS * P_g / S_p)``, with the ONE shared edge policy.

    ``r_i == 0`` means cell i holds every count in the group; the leave-one-out remainder is
    all zero, and the existing all-zero-bulk contract (``prep.bulk_lognorm_means``) maps that
    to a zero bulk rather than a NaN. Negative ``r_i`` is impossible for counts and is a
    corrupt input, not a case to smooth over. Note the masking happens BEFORE the shift, so an
    ``r_i == 0`` row contributes ``0 - b_g``: its ``v`` really is zero, and the variance is
    taken over the actual values.

    WARNING -- returns the DEVIATION, not ``v`` itself, and that is load-bearing. Every caller
    feeds this straight into a variance, and variance is invariant under ``v -> v - b``, so the
    shift is exact. It is not cosmetic: without it the ``s2 - s1**2/n`` reduction cancels
    roughly ``log10(n**2 * v_bar**2 / var)`` digits -- MEASURED at 4.4 digits at n=10 rising to
    9.7 at n=4000, costing six significant digits there and making the answer depend on the
    block size.

    Folding it in HERE rather than at each call site is deliberate. The resident kernel
    (:func:`jackknife_correction`), the shard/cell-streaming kernel
    (``streaming_bulk._streaming_jackknife``) and the GPU kernel
    (``gpu.bulk.GroupedMeanAccumulator.jackknife``) each reduce their own ``s1``/``s2``, so a
    shift applied at only some of them is a silent per-driver divergence -- measured at
    **1.539e-06** between the unshifted streaming kernel and the resident one at n=4000, a
    divergence that still passes the 1e-8 cross-kernel assertion those tests use.

    ``xp`` is the array module: pass ``self._xp`` (from ``gpu.xp_for(device)``) on the GPU path
    so ``V`` stays on device. Defaults to numpy for every host caller.
    """
    if bool(xp.any(r < 0)):
        raise ValueError("negative leave-one-out remainder; input is not counts")
    safe = xp.where(r > 0, r, 1.0)[:, None]
    V = xp.log1p(bulk_target_sum * (P[None, :] - Y_block) / safe)
    # S == 0 means the whole group is empty: P is all zeros, so b is 0 and the masked V is 0.
    # The xp.where avoids a 0/0 -> NaN that would poison an all-zero group. The resident kernel
    # skips those with an explicit `tot <= 0` continue; the two streaming kernels do not, so
    # the guard has to live here.
    S = P.sum()
    b = xp.log1p(bulk_target_sum * P / xp.where(S > 0, S, 1.0))
    return xp.where((r > 0)[:, None], V, 0.0) - b[None, :]


def jackknife_correction(X, group_codes, n_groups: int,
                         bulk_target_sum: float, *, chunk: int = 512) -> np.ndarray:
    """Delete-1 jackknife ``C_p`` for the ``bulk_lognorm`` comparator (issue #264 §3.6).

        r_i  = S_p - lib_i          S_p = Σ_g P_p,g,  lib_i = Σ_g y_ig
        v_ig = log1p( TS * (P_p,g - y_ig) / r_i )
        C_p  = (n-1)/n * Σ_g Σ_i ( v_ig - v̄_g )²

    SUMMED OVER GENES, interchangeable with ``trace_over_n_for``'s return.

    ``O(n·G)`` DENSE, not ``O(nnz)``: dropping cell ``i`` moves ``S_p``, so a gene where that
    cell has no counts still changes. Cells are visited in ``chunk``-row blocks, so the working
    set scales as ``chunk × G`` -- but the CONSTANT is not 1. :func:`_loo_bulk` holds several
    ``chunk × G`` fp64 temporaries live at once (the dense block, the numerator, the ``log1p``,
    the masked ``where``, the shift), and MEASURED at 500 cells × 18,533 genes, ~20k UMI/cell,
    the peak Python allocation is **349 MiB at the default ``chunk=512``** against a naive
    ``chunk × G × 8B`` of 72 MiB -- **4.8×**. Budget from the measured figure, not the bound.

    That peak is the knob to turn if memory is tight, and it is free to turn: the shifted
    reduction is chunk-invariant to ~1 ULP (see below), so the block size is a cost knob and
    not an answer knob. Measured on that same group -- 349 MiB / 375 ms at ``chunk=512``, 157 MiB / 161 ms
    at 128, **89 MiB / 116 ms at 32**. Smaller blocks are both smaller and faster here because
    one temporary at ``chunk=32`` is 4.7 MB and stays in cache while ``chunk=512``'s is 76 MB
    and does not. The default is left at 512 pending a measurement on a second box.

    Two different benchmarks, so do not read them as one run: spec §4.1 measured **~222 ms per
    group / ~67 s per 300-construct archive** on the real panel, and the memory figures above
    come from a synthetic panel of the same shape on a different host (which timed the same
    default block at 375 ms). Either way the correction roughly doubles an expression-only
    scoring run and is negligible next to DE.

    ``n < 2`` returns 0.0, mirroring ``trace_over_n_for``'s policy of subtracting nothing
    rather than returning NaN. The reduction is ``s2 - s1²/n`` over :func:`_loo_bulk`'s
    return, which is ALREADY the deviation from the full-sample bulk -- see its docstring for
    why that shift is exact and why it lives there rather than at each kernel. Shifted, this
    lands at machine epsilon and is chunk-invariant to **~1 ULP** -- MEASURED bit-identical
    across ``chunk`` 7…1024 at (n=500, G=113) and (n=300, G=997), and 2.6e-16 relative at
    (n=400, G=251), which is three orders inside the ``rtol=1e-13`` that
    ``test_the_chunk_size_does_not_change_the_answer`` asserts. Block summation reassociates,
    so exact equality is a common outcome and not a guarantee; the UNSHIFTED form was 5.551e-11
    off at n=40 and 1.539e-06 at n=4000, which is what that test exists to catch. The result is
    still clamped at zero because a correction is a variance and a negative one would be
    subtracted as a bonus.

    ⚠️ **``C`` IS BIASED UPWARD, and how much depends on ``bulk_target_sum`` (issue #268).**
    The delete-1 jackknife carries the usual Efron-Stein upward bias, and ``TS`` sets where
    ``log1p``'s linear→log knee sits relative to expression: the larger ``TS`` is against the
    panel's own depth, the more near-empty genes are treated as log-fold-changes, and those are
    exactly where the bias lives. MEASURED on a 6-line, 400-cell/construct, ~20k-UMI,
    18,533-gene panel by the split-half identity ``E‖b_A − b_B‖² = C_A + C_B`` (uncapped -- the
    #247 ``min`` hides two thirds of it, reading +0.75% where the uncapped form reads +2.06%):

        TS      2e3    5e3    1e4    2e4    5e4    1e5    3e5    1e6
        bias   0.19%  0.23%  0.24%  0.26%  0.32%  0.42%  0.81%  2.06%

    Those readouts include a +0.25% inflation from the halves being complementary subsets of
    one group, so subtract it for the estimator's own bias.

    **The shipped default is ``TS = 5e4`` (#268, 2026-08-11).** It reads 0.32% here, i.e.
    ~0.07% once the +0.25% artifact is removed, which meets spec §2's "``C`` right to ~0.1%".
    ``TS = 1e6`` was the shipped value until then and is **the one value on this sweep where
    the metric breaks**: 2.06% (~1.81% net), the split-half ceiling negative on 6 of 6 lines,
    and the "predict the control" anchor at 1.073 where it must read 1.0 -- against a
    ``scales.py`` base that ships as a hard 1.0. Only ~25% of the denominator survives its own
    correction there, which amplifies every error ~4x. 1e6 did not win on the axis spec §3.2
    chose it for either: effective gene count peaks at 3e5 and is lower at 1e6 (8,485) than at
    1e5 (8,856).

    ``X`` may be sparse OR a dense ndarray, like every other ``prep`` grouped helper: a dense
    ``adata.X`` is legal input and reaches here through ``_side_bulks``. Duplicate coordinates
    need NO ``sum_duplicates()`` -- both reductions used here (``.sum``, ``.toarray``) collapse
    them, unlike ``_grouped_sumsq``, which squares them separately and therefore must.
    """
    X = X.tocsr() if issparse(X) else csr_matrix(X)
    codes = np.asarray(group_codes, dtype=np.intp)
    if codes.size != X.shape[0]:
        raise ValueError(
            f"group_codes has {codes.size} entries for {X.shape[0]} rows")
    out = np.zeros(n_groups, dtype=np.float64)
    order = np.argsort(codes, kind="stable")
    bounds = np.searchsorted(codes[order], np.arange(n_groups + 1))
    for p in range(n_groups):
        rows = order[bounds[p]:bounds[p + 1]]
        n = rows.size
        if n < 2:
            continue
        Xp = X[rows]
        Xd = Xp.__class__((Xp.data.astype(np.float64), Xp.indices, Xp.indptr), shape=Xp.shape)
        P = np.asarray(Xd.sum(axis=0), dtype=np.float64).ravel()
        lib = np.asarray(Xd.sum(axis=1), dtype=np.float64).ravel()
        tot = float(P.sum())
        if tot <= 0.0:
            continue
        G = Xd.shape[1]
        s1 = np.zeros(G, dtype=np.float64)
        s2 = np.zeros(G, dtype=np.float64)
        for s in range(0, n, chunk):
            V = _loo_bulk(P, Xd[s:s + chunk].toarray(), tot - lib[s:s + chunk],
                          bulk_target_sum)
            s1 += V.sum(axis=0)
            s2 += np.einsum("ij,ij->j", V, V)   # no V*V temporary
        out[p] = max(((n - 1) / n) * float((s2 - s1 ** 2 / n).sum()), 0.0)
    return out


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


def _require_cell_counts(counts: np.ndarray, perts) -> None:
    """Raise unless every aligned count is a finite, positive whole number of cells (#227).

    ``n == 1`` PASSES -- that is the #219 case, and :func:`trace_over_n_for` answers it with a
    zero correction. What this rejects is everything the ``counts < 2`` branch used to absorb
    alongside it: ``0`` (no cells, hence no sample mean either), negative, fractional and
    non-finite. Those are not states a cell count can be in, so they mean the moments disagree
    with the pseudobulk they were produced with, and the old behaviour turned that disagreement
    into a plausible finite number.

    Checked on the ALIGNED counts, so the error names the perturbation the caller asked for
    rather than a row of a moments artifact it never sees. The sample is capped at 5: a
    CCL-scale panel would otherwise build a full-length list to slice off the front.

    BLAST RADIUS, checked rather than assumed. Every producer in the tree emits whole counts >= 1:
    ``prep`` uses ``np.diff(bounds)`` over observed labels, ``streaming_bulk`` an int64
    ``np.add.at`` bincount, ``gpu.bulk`` an ``xp.bincount``. The full suite surfaced exactly ONE
    caller that did not -- ``tests/test_expr_mse_unbiased_ratio.py``'s pre-#247 characterization
    fixture, which built ``pred_counts`` as ``real_counts * uniform(0.05, 3.0)`` with a literal
    ``0.0``. Both were incidental to what that test characterizes and the fixture was corrected;
    it is recorded here because "no producer emits these" is the claim this validation rests on,
    and it was worth one counterexample to find out.

    Reached only under the ``lognorm`` comparator: ``correction_for`` returns ``moments.jk``
    under ``bulk_lognorm`` and never calls :func:`trace_over_n_for`, so the competition path does
    not pass through here at all.
    """
    # Comparing and rounding the NON-finite entries here is safe, and masking them out first was
    # proposed and REFUTED (Gemini review of PR #302, which expected a `FloatingPointError` under
    # `np.errstate(all="raise")`). Measured on numpy 2.4.6: `isfinite`, `>= 1.0`, `rint` and
    # `== rint` all run clean on NaN/+-inf under that errstate, because numpy's comparisons use
    # non-signaling predicates and `rint` of NaN/inf is exact. Pinned in test_moments.py.
    bad = ~(np.isfinite(counts) & (counts >= 1.0) & (counts == np.rint(counts)))
    if not bad.any():
        return
    labels = np.asarray(perts)
    idx = np.flatnonzero(bad)
    # Both `str(...)` and `float(...)` are load-bearing, though each reads redundant (Gemini review
    # of PR #302 proposed dropping the `str`). These are NumPy scalars, and under NumPy 2 a scalar
    # reprs as its constructor call: `repr(np.str_("P0"))` is `np.str_('P0')` and
    # `repr(np.float64(0.0))` is `np.float64(0.0)`. Without both casts the sample reads
    # `np.str_('P0'): np.float64(0.0)` instead of `'P0': 0.0`. The count side was leaking exactly
    # that until this round. Pinned in test_moments.py by asserting the delimiter CONTEXT
    # (`">= 1: 'P0':"`), which no scalar repr can satisfy -- a bare `"'P0'" in msg` cannot, being
    # a substring of `np.str_('P0')` too.
    sample = ", ".join(f"{str(labels[i])!r}: {float(counts[i])!r}" for i in idx[:5])
    raise ValueError(
        f"{idx.size} of {counts.size} perturbations carry a cell count that is not a finite "
        f"whole number >= 1: {sample}"
        f"{', ...' if idx.size > 5 else ''}. A count of 0 means the group has no cells and so "
        "no sample mean either; negative, fractional and non-finite counts cannot occur at "
        "all. Either way these moments disagree with the pseudobulk they were produced with -- "
        "a corrupt cache, a malformed stream, or a hand-built GroupMoments. Returning a zero "
        "correction here would turn that disagreement into a plausible finite number (issue "
        "#227). n == 1 is legal and subtracts nothing (issue #219)."
    )


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

    **Everything OTHER than ``n >= 1`` now RAISES (issue #227).** ``counts == 0`` is not the
    #219 case and never was: a group with no cells has no sample mean either, so its bulk row
    is meaningless and a zero correction would be a fallback for a state that cannot occur
    rather than a modelled quantity. Negative, fractional and non-finite counts are not states
    that can occur at all -- each one means the moments disagree with the pseudobulk they were
    produced with. Until #227 all four took the ``counts < 2`` branch alongside ``n == 1`` and
    became a plausible finite number, silently; they are now rejected in the same spirit as the
    missing-label check a few lines above, which already treats a moments/pseudobulk mismatch as
    an invariant violation rather than something to paper over.

    None of them arises from ``prep.pseudobulk_with_moments`` (groups come from observed labels,
    so ``counts = np.diff(bounds) >= 1``); they arrive from a corrupt cache, a malformed stream,
    or a directly constructed :class:`GroupMoments`. ``trace_sigma``'s NaN-at-``n < 2`` contract
    is UNCHANGED -- it is right to call the covariance undefined, and the validation lives here
    for the same reason the ``n < 2`` policy does: the primitive stays honest and the consumer
    decides what it is allowed to accept.

    ⚠️ EVERY NUMBER IN THE REST OF THIS DOCSTRING IS PRE-#247, in squared-expression
    units and measured before the cap. That metric is restored as ``expr_mse_unbiased``.
    The behaviour these numbers justify -- subtract zero rather than NaN at ``n < 2`` -- is
    unchanged and still right, and the magnitudes once more describe that restored metric.

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
    returns the SUMMED quantity, so the bound here is ``G*mean(m²)``; dividing by G puts it at
    ``mean(m²)``, which is the restored metric's scale. ``metrics.delta.mse_unbiased`` applies
    that 1/G at the call site once more. Measured at a fixed predicted mean (G=2000,
    gamma panel, mean(m²)=0.4593): the correction goes 0.0015 -> 0.4593 and u only 24.9985 ->
    24.5407, where an unvalidated 2 cells at ±100 would give -9975. Generally
    ``u >= expr_mse - mean(m²) - C_real`` -- a LOWER BOUND, not an equality, since attaining it
    also needs the extremizer to clear ``check_scale_limit``. A large error is safe only
    RELATIVE to ``mean(m²)``. See ``docs/metrics.md`` §2.3.

    ⚠️ **That paragraph describes the world before issue #247, and this primitive is still that
    world.** The dispersion channel is now bounded at the CALL SITE, not here:
    ``metrics.delta.mse_unbiased_capped`` caps the prediction's term at
    ``PRED_TRACE_CAP_K * tr Σ̂_real/n_real`` (k=1), so the reachable subtraction is
    ``2 * C_real`` rather than ``mean(m²)``. The policy lives there for the same reason the
    ``n < 2`` policy lives in *this* function rather than in ``trace_sigma``: the primitive
    returns the honest quantity and the metric decides what it is allowed to claim. Anything
    calling ``trace_over_n_for`` directly gets the UNCAPPED term and inherits the old bound.
    Unbiasedness still assumes an honest iid emission; cell_eval2 does not verify it, and no
    longer relies on it for this channel.

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
    _require_cell_counts(counts, perts)
    tr = trace_sigma(counts, sumsq, means)
    # Divide only where the trace is defined; elsewhere subtract NOTHING (#219, above).
    # The fill is 0.0 rather than NaN: thinness alone no longer makes a group vanish -- it
    # forfeits the correction and is penalized in expectation (not on every draw).
    # After #227 the only row that can take the zero branch is `counts == 1`; every other
    # sub-2 value raised above.
    out = np.zeros(tr.shape, dtype=np.float64)
    ok = counts >= 2
    out[ok] = tr[ok] / counts[ok]
    return out


def correction_for(moments: GroupMoments, perts, means: np.ndarray, *,
                   comparator: str) -> np.ndarray:
    """The sampling correction to subtract, in whichever space this run is scoring.

    ``comparator`` is REQUIRED and never defaulted. Both mismatches raise: a ``bulk_lognorm``
    run over jk-less moments would silently get ``tr Σ̂/n``, which is not the variance of a
    group-sum statistic; and a ``lognorm`` run over jk-bearing moments is reading an artifact
    built in the wrong space. Neither is a number worth producing.
    """
    if comparator not in ("lognorm", "bulk_lognorm"):
        raise ValueError(
            f"comparator must be 'lognorm' or 'bulk_lognorm', got {comparator!r}")
    if comparator == "lognorm":
        if moments.jk is not None:
            raise ValueError(
                "comparator is 'lognorm' but these GroupMoments carry a jackknife, so they "
                "were built in the bulk_lognorm space; their counts/sumsq do not describe "
                "this comparator."
            )
        return trace_over_n_for(moments, perts, means)
    if moments.jk is None:
        raise ValueError(
            "comparator is 'bulk_lognorm' but these GroupMoments carry no jackknife (jk is "
            "None). The driver that produced them did not run the second pass; it must, or "
            "the metric must not be requested on it. This never falls back to tr Sigma/n."
        )
    index = {str(p): i for i, p in enumerate(moments.perts)}
    try:
        rows = np.asarray([index[str(p)] for p in perts], dtype=np.intp)
    except KeyError as exc:
        raise ValueError(
            f"label {exc.args[0]!r} is absent from these GroupMoments; some driver restricted "
            "them. Moments span ALL groups including the control."
        ) from None
    return moments.jk[rows]


def unbiased_sq_dist(pred_means, real_means, pred_trace_over_n, real_trace_over_n):
    """Unbiased ``‖μ_pred − μ_real‖²``, SUMMED over genes, per aligned row.

    The plug-in squared distance minus both sides' ``tr Σ̂/n`` terms. Summed, not
    gene-averaged, because #202's magnitude ``m_p`` is defined summed over genes.

    ⚠️ The metric call sites apply ``1/G`` to this SUMMED primitive.
    :func:`metrics.delta.mse_unbiased` and :func:`metrics.delta.mse_unbiased_capped` thereby
    return the historical gene-averaged squared-expression scale.

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
