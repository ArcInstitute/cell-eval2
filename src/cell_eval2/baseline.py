"""Generic-response baseline: one average perturbation response, emitted for every
perturbation, scored as an ordinary submission.

A raw metric value is uninterpretable without a comparator. This module builds the
predictor that supplies one -- a model handed the average perturbation response and
nothing target-specific. It is an ORACLE comparator, not a floor a submission could
reach: the profile is averaged from the evaluated real perturbations, so it is
transductive and no real submission has access to that data. What it bounds is a
metric's TRIVIALITY -- a score at or below it is not evidence of target-specific skill.
Designs of record:
``internal:docs/superpowers/specs/2026-07-29-generic-response-baseline-design.md`` for the profile
and ``internal:docs/superpowers/specs/2026-08-05-baseline-dispersed-emission-design.md`` for its
emission as cells.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import warnings
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from importlib.metadata import Distribution, PackageNotFoundError
from typing import TYPE_CHECKING, Literal

import numpy as np
import polars as pl
import scipy.sparse as sp

if TYPE_CHECKING:
    from anndata import AnnData

from . import norm
from .cache import fingerprint_adata, fingerprint_de_table
from .catalog import CATALOG, _NAME_TO_CANONICAL, derived_policy, is_decisive, resolve_metrics
from .config import EvalConfig
from .de import load_de_table
from .distances import resolve_exclusion_columns
from .io import load_anndata
from .prep import pseudobulk
from .scoring import denominator, is_degenerate
from .run import _close_backed, effective_normalization   # build_run_meta's backed
# provenance reads _close_backed; effective_normalization joins catalog declaration to run policy
from .run import (
    _cache_backend,
    _cache_device,
    _effective_input_type,
    aggregate_metrics_wide,
    compute_metrics,
    metric_output_names,
)

logger = logging.getLogger(__name__)


def _version_lazy() -> str:
    from . import __version__  # lazy: baseline <- run <- __init__ ordering
    return __version__


def _materialize_reference(source):
    """Load a reference IN MEMORY. Never backed -- see the note at the top of Task 2:
    ``prep._grouped_means`` decides sparsity once with ``issparse(X)``, which is False for a
    backed h5ad CSR matrix, and ``pseudobulk`` then raises ``ValueError: setting an array
    element with a sequence``. The builder needs the whole matrix anyway, and a non-backed
    read closes the file, so there is no handle to leak and no try/finally to get wrong.

    A CALLER-supplied backed object is materialized into a local copy WITHOUT disturbing
    their handle. ``AnnData.to_memory()`` is the obvious route and the wrong one: on some
    anndata versions it CLOSES the source's backing file (measured -- it does not on
    anndata 0.13.2, it does on the version pinned for py3.11 in CI), which is a surprising
    side effect on an object the caller still owns. Re-reading the same path cannot do
    that. The fallback is only for a backed object with no usable filename, or a VIEW,
    whose path would re-read MORE cells than the caller handed us.
    """
    adata = load_anndata(source)                    # backed=False
    if not getattr(adata, "isbacked", False):
        return adata
    filename = getattr(adata, "filename", None)
    if filename is not None and not getattr(adata, "is_view", False):
        return load_anndata(str(filename))
    return adata.to_memory()


@dataclass(frozen=True)
class GenericProfile:
    """The average perturbation response, plus the provenance needed to trust it.

    ``n_excluded`` is the count of ``(perturbation, gene)`` terms the self-target
    exclusion actually dropped. It is reported rather than merely logged because the
    exclusion is a SILENT NO-OP when perturbation labels are guide IDs instead of gene
    symbols: nothing matches, and a stamped ``n_excluded == 0`` beside
    ``exclude_target_gene == True`` is the only after-the-fact proof of that.
    """

    values: np.ndarray
    genes: np.ndarray
    n_perturbations: int
    exclude_target_gene: bool
    n_excluded: int


def generic_response_profile(
    real,
    *,
    pert_col: str,
    control: str,
    exclude_target_gene: bool = True,
    target_gene_map: dict[str, str] | None = None,
) -> GenericProfile:
    """Mean per-perturbation pseudobulk of ``real``, over non-control perturbations.

    With ``exclude_target_gene=True`` (the default) gene ``g``'s mean omits the
    perturbation that targets ``g``::

        profile[g] = mean over { p : p != control, label(p) != g } of pseudobulk[p][g]

    The exclusion is per GENE, not per perturbation, so the result is still ONE constant
    vector -- which is what keeps the whole baseline cheap. Its numerical footprint is
    O(1/N) and will not visibly move expr_mae; what it protects is narrow and
    high-salience -- without it, the perturbation targeting gene g gets a leaked correct
    direction call on the single most important gene in its own comparison.

    The matrix is averaged AS LOADED, with no normalization conversion, so the profile is
    numerically in the reference's own space. That is NOT sufficient on its own: under v1
    (and under v2 with ``autodetect_input_type``) the pipeline re-detects each side's
    convention independently, and the two sides can land in DIFFERENT spaces in either
    direction. ``lock_matrix_space`` closes that gap -- see section 3.0 of the design.

    Equal weight per perturbation, not per cell: ``pseudobulk`` reduces to a
    per-perturbation mean first, so a perturbation with many cells cannot dominate.

    ``target_gene_map`` is ``EvalConfig.target_gene_map``: the construct-ID escape hatch
    (``'ADNP-1'`` -> ``'ADNP'``) that decides what "this perturbation's target gene" means
    for ``pds_*`` and the chance-corrected DE metrics. It has to reach here too, or the
    baseline disagrees with the metrics it is the reference point for (#253, #285).
    """
    return _profile_from_adata(
        _materialize_reference(real), pert_col=pert_col, control=control,
        exclude_target_gene=exclude_target_gene, target_gene_map=target_gene_map,
    )


def _profile_from_adata(
    adata, *, pert_col: str, control: str, exclude_target_gene: bool,
    target_gene_map: dict[str, str] | None = None,
) -> GenericProfile:
    """``generic_response_profile`` on an already-open handle (the orchestrator owns one)."""
    if pert_col not in adata.obs.columns:
        raise ValueError(
            f"perturbation column {pert_col!r} missing from adata.obs; "
            f"present: {list(adata.obs.columns)}"
        )
    # Validate the control label from obs FIRST: it is a cheap metadata read, and a
    # mistyped label would otherwise cost a full grouped-mean pass over the reference
    # before failing.
    labels_all = adata.obs[pert_col].to_numpy().astype(str)
    if control not in set(labels_all.tolist()):
        raise ValueError(
            f"control {control!r} not found in {pert_col!r}; "
            f"present: {sorted(set(labels_all.tolist()))[:10]}"
        )

    genes = np.asarray(adata.var.index.values).astype(str)
    perts, means = pseudobulk(adata, pert_col)
    perts = np.asarray(perts).astype(str)

    keep = perts != control
    labels = perts[keep]
    bulk = np.asarray(means, dtype=np.float64)[keep]
    n = int(bulk.shape[0])
    if n == 0:
        raise ValueError(
            f"no non-control perturbation left after excluding {control!r}; "
            "a generic-response profile needs at least one"
        )

    total = bulk.sum(axis=0)
    count = np.full(genes.size, float(n))
    n_excluded = 0

    if exclude_target_gene:
        if n < 2:
            raise ValueError(
                f"exclude_target_gene=True needs at least 2 non-control perturbations "
                f"(got {n}): dropping the only contribution to a targeted gene would "
                "leave its mean undefined. Pass exclude_target_gene=False to average "
                "the single perturbation as-is."
            )
        # A duplicated var index makes 'the target gene' ambiguous.
        if len(set(genes.tolist())) != genes.size:
            raise ValueError(
                "exclude_target_gene=True requires a unique var index; "
                f"{genes.size - len(set(genes.tolist()))} duplicate gene name(s) found"
            )
        # #253/#285: THE SHARED DEFINITION, not a third hand-rolled copy. This block used to
        # be `gene_pos.get(label)` against the raw label alone, which ignored
        # `target_gene_map` -- so on a construct-ID panel every lookup missed, `n_excluded`
        # stayed 0, and the profile silently equalled the plain mean. That is the #248 bug at
        # its third call site, and this is the 0 END of the competition scale rather than a
        # metric, which is what makes a silent no-op here worth a raise.
        #
        # `gate_labels` is left at its default: `labels` here IS the whole panel (every
        # non-control perturbation of the reference), so the gate set and the resolution set
        # are the same thing. The shard-vs-panel distinction the parameter exists for belongs
        # to the partitioned metric drivers, not to a builder that owns the whole reference.
        #
        # ⚠️ The helper's zero-resolve RAISE comes with it, replacing this block's warning.
        # Behaviour-changing: a caller asking for exclusion on a construct-ID panel with no
        # map used to get a warning and a plain-mean baseline, and now gets an error naming
        # both escape hatches (supply the map, or pass exclude_target_gene=False). #285's
        # suggested fix, and the only coherent option once the resolver is shared -- catching
        # its ValueError to re-WARN would share the code while re-splitting the behaviour,
        # which is the divergence #248 exists to close.
        #
        # Catching it to RE-CONTEXTUALIZE is a different thing, and is what happens below:
        # the behaviour is identical (it still raises, on the same condition, with the same
        # two remedies), only the CONSEQUENCE sentence is wrong here. The resolver's message
        # is written for the discrimination metrics -- "the ranked vector", "inflates the
        # discrimination score" -- and this caller computes neither (Copilot, PR #298). The
        # original is prepended to rather than reworded, and chained, so the diagnosis and the
        # remedies stay defined in exactly one place.
        try:
            cols = resolve_exclusion_columns(labels, genes, target_gene_map=target_gene_map)
        except ValueError as e:
            raise ValueError(
                "the generic-response baseline cannot exclude target genes. The consequence "
                "here is not a discrimination score: the profile would equal the PLAIN MEAN, "
                "leaving each perturbation's own on-target knockdown in for the very gene it "
                "perturbs most strongly -- and this profile is the 0 END of the competition "
                f"scale. {e}"
            ) from e
        for i, j in cols.items():
            total[j] -= bulk[i, j]
            count[j] -= 1.0
        n_excluded = len(cols)

        # NEWLY REACHABLE through the map, and silent without this guard: `count[j]` counts
        # the perturbations still contributing to gene j, and a gene targeted by EVERY
        # perturbation loses them all -- 0/0, a NaN in the middle of the profile (measured).
        # The raw-label lookup could not reach it in practice because distinct labels were
        # distinct genes; a map is many-to-one, so `{'g0-1': 'g0', 'g0-2': 'g0'}` on a
        # two-perturbation panel is enough. Same reason as the `n < 2` raise above, one
        # granularity down: dropping every contribution leaves that gene's mean undefined.
        if (empty := np.flatnonzero(count <= 0)).size:
            raise ValueError(
                f"exclude_target_gene=True leaves no perturbation contributing to "
                f"{empty.size} gene(s) -- every one of the {n} non-control perturbations "
                f"targets them, so their generic-response mean is undefined. Sample: "
                f"{genes[empty][:5].tolist()}. Score a panel with a perturbation that does "
                "not target them, or pass exclude_target_gene=False."
            )

    return GenericProfile(
        values=total / count,
        genes=genes,
        n_perturbations=n,
        exclude_target_gene=exclude_target_gene,
        n_excluded=n_excluded,
    )


def _emission_scale(profile_values, control_pseudobulk, genes) -> tuple[np.ndarray, dict]:
    """Return the gene-wise emission scale and JSON-safe profile diagnostics."""
    profile_values = np.asarray(profile_values, dtype=np.float64).ravel()
    control_pseudobulk = np.asarray(control_pseudobulk, dtype=np.float64).ravel()
    genes = np.asarray(genes).astype(str).ravel()
    if (profile_values.size != control_pseudobulk.size
            or profile_values.size != genes.size):
        raise ValueError(
            "profile, control pseudobulk and genes must have the same length "
            f"({profile_values.size}, {control_pseudobulk.size}, {genes.size})"
        )

    control_zero = control_pseudobulk <= 0
    # The safe denominator is required even though np.where selects the zero arm only
    # afterwards: numpy evaluates both division operands before applying that selection.
    denominator = np.where(control_zero, 1.0, control_pseudobulk)
    scale = np.where(control_zero, 0.0, profile_values / denominator)
    profile_zero_control_positive = (profile_values == 0) & ~control_zero
    # This is an L1 MASS fraction: mixed signs in a general reference must not cancel.
    # On the measured counts reference the profile is a mean of non-negative counts, so
    # abs() is a no-op and section 3.1's recorded 1.55e-07 is unchanged.
    profile_l1_mass = float(np.abs(profile_values).sum())
    positive = scale[scale > 0]
    names = genes[control_zero].tolist()
    diagnostics = {
        "n_genes_control_zero": int(control_zero.sum()),
        "genes_control_zero": names,
        "profile_mass_unreachable": (
            float(np.abs(profile_values[control_zero]).sum() / profile_l1_mass)
            if profile_l1_mass != 0 else 0.0
        ),
        "n_genes_profile_zero_control_positive": int(
            profile_zero_control_positive.sum()
        ),
        "r_max": float(scale.max()) if scale.size else 0.0,
        "r_median_nonzero": float(np.median(positive)) if positive.size else 0.0,
    }
    return scale, diagnostics


def _emit_scaled_resample(
    control_X, group_sizes, scale, rng
) -> tuple[sp.csr_matrix, np.ndarray, dict]:
    """Resample control-pool row positions by group and scale their genes."""
    control_X = control_X.tocsr() if sp.issparse(control_X) else sp.csr_matrix(control_X)
    if control_X.shape[0] == 0:
        raise ValueError("cannot emit cells from an empty control pool")
    scale = np.asarray(scale).ravel()
    if scale.size != control_X.shape[1]:
        raise ValueError(
            f"scale has {scale.size} genes but the control pool has "
            f"{control_X.shape[1]} columns"
        )

    sizes = []
    for value in group_sizes:
        size = int(value)
        if size != value or size < 0:
            raise ValueError(f"group sizes must be non-negative integers, got {value!r}")
        sizes.append(size)
    draws = [rng.integers(0, control_X.shape[0], size=size) for size in sizes]
    source = (np.concatenate(draws).astype(np.intp, copy=False) if draws
              else np.empty(0, dtype=np.intp))
    out = control_X[source].copy()
    # scale stays float64. Casting it to float32 first and multiplying in place was suggested
    # (Gemini, PR #241) to halve the gathered temporary; REJECTED on measurement. A float64
    # scale rounds ONCE, and the float32 result is then the correctly-rounded exact product
    # (verified ulp-exact on 200k entries at the reference's r_max of 46.85); rounding scale
    # first adds a second rounding, ~1.7x the error, and moved 55,936 of those 200,000
    # entries. It would also break the bit-reproducibility of the artifacts already written
    # by internal:tools/baselineval/make_baselines.py, which used exactly this arithmetic.
    out.data = (out.data * scale[out.indices]).astype(np.float32)
    stored_before = int(out.nnz)
    out.eliminate_zeros()
    row_totals = np.asarray(out.sum(axis=1)).ravel()
    diagnostics = {
        "max_row_total": float(row_totals.max()) if row_totals.size else 0.0,
        "n_explicit_zeros_removed": stored_before - int(out.nnz),
        "n_rows": int(out.shape[0]),
    }
    return out, source, diagnostics


def build_baseline_prediction(
    profile: GenericProfile,
    template,
    *,
    pert_col: str,
    control: str,
    emit: Literal["dispersed", "tile"] = "dispersed",
    seed: int = 0,
) -> AnnData:
    """Build a generic-response prediction shaped exactly like ``template``.

    ``emit="dispersed"`` is COUNTS-ONLY. With a lognorm-effective template, scaling
    log-space values by a ratio of log-space means is not a fold change, and no later
    per-cell CPM normalization repairs it. This is a DOCUMENTED PRECONDITION, not an
    enforced one: the guard lives in ``build_generic_baseline``, because effective input
    type is a function of ``EvalConfig`` and this constructor receives no config.

    In counts space, conditional on the fixed control pool, the construction has
    ``E[group counts mean] = r * ctrl = profile`` exactly on every reachable gene before
    float32 rounding. It does NOT imply that the group's lognorm mean equals the lognorm
    profile; the measured cost is +2.8% (0.012568 versus 0.012226 for the profile compared
    directly without cells) -- ⚠️ **all three figures are pre-#247 ``expr_mse_unbiased``**, in
    squared expression units. The RATIO does not transfer either: normalization happens per
    perturbation and then aggregates, and ``mean(N_a/D_p) / mean(N_b/D_p)`` is not
    ``mean(N_a)/mean(N_b)``. Only the QUALITATIVE claim survives -- emitting through cells
    costs something, and it is small. Re-measure before quoting a number against an
    ``expr_mse_unbiased_capped`` column. Calibrating ``r`` against the
    post-normalization target was considered and rejected: it fits the comparator to the
    metric and breaks the counts-space semantics used by this module. The emission is not
    unexecutable by a submission: normalization is public and the control pool was handed
    out. The profile itself remains oracle information.

    Scaling is multiplicative because counts must remain non-negative. An additive shift
    needs clipping, which biases the mean it was meant to hit. With-replacement draws are
    iid from the empirical transformed-control distribution conditional on the pool. Thus
    ``tr(Sigma-hat_pred) / n_pred`` is the right ``expr_mse_unbiased_capped`` correction:
    duplicates are legal and no finite-population correction is needed. Drawing without
    replacement would require one and the metric would over-subtract.

    A direct caller who supplies its own ``cache_pred`` can collide across emission modes
    and seeds: non-strict ``fingerprint_adata`` never reads ``X``.

    Both arms retain the template contract: same shape, whole ``obs`` including its index,
    same ``var``, and control rows copied from the template and cast to float32.
    """
    return _prediction_from_adata(
        _materialize_reference(template), profile, pert_col=pert_col, control=control,
        emit=emit, seed=seed,
    )


def build_constant_prediction(
    profile: GenericProfile, template, *, pert_col: str, control: str
):
    """Build the legacy ``emit="tile"`` prediction.

    This arm is known-biased. Per-cell normalization followed by averaging is not the
    normalization of a tiled mean: the measured +0.464 mean-per-gene gap is 96% of tile's
    ``expr_mse`` (0.3936 versus 0.0126 for dispersed). Identical predicted cells also
    force the pred-side covariance to zero, yielding 9,701 significant genes per
    perturbation versus 414 for dispersed and 2,739 in the reference. Use
    ``build_baseline_prediction(..., emit="dispersed")`` for the supported construction;
    tile exists only to reproduce pre-fix numbers.

    Same perturbations, same cells per perturbation, same ``var``, same ``obs`` -- because
    cell counts drive the DE p-values, so a baseline with fabricated counts would be a
    comparator for a different experiment. The WHOLE ``obs`` is copied, not just
    ``pert_col``: the deseq2 backend requires ``de.replicate_col`` in the prediction's obs
    (``deseq2_de.py:26-32``), and the real-control concat retains only columns common to
    both sides, so a pert_col-only copy would drop it and DE would fail.

    The CONTROL rows carry ``template``'s own control cells, NOT the profile. Filling them
    with the profile makes the prediction's control equally constant, so under
    ``control_source="pred"`` the predicted delta is identically zero and the DE is fully
    tied (measured: direction_precision 0.0, delta_pearson -1.0). With the real control's
    cells the two ``control_source`` settings agree, which is what lets the baseline stay
    convention-neutral -- no forced override, no v1-vs-v2 wrinkle. It is also the honest
    predictor: the generic-response hypothesis is about PERTURBATION responses and says
    nothing about the control.

    ``X`` remains the old materialized dense float32 ``np.tile`` array, deliberately not
    ``np.broadcast_to``. This exact representation is retained only to reproduce #225's
    reference points and the pre-fix campaign; dispersed emission mirrors template
    sparsity and avoids this dense construction intermediate for sparse references.

    float32 is also why the control rows are ``template.X`` CAST, not ``template.X``: on a
    float64 reference the two ``control_source`` settings then agree to float32 round-off
    (measured: DE metrics exactly, delta_pearson to 6.5e-9) rather than bit-for-bit.
    Promoting to the template's dtype would buy that exactness by doubling the largest
    artifact in the design -- see section 4.2.
    """
    # The message has to be honest about the counts-only precondition: this warning is the
    # ONE thing a lognorm-reference caller of this function reads, and pointing them at
    # emit="dispersed" unqualified sends them to a construction that is undefined for their
    # data -- build_generic_baseline would refuse it outright (Copilot, PR #241, suppressed).
    warnings.warn(
        "build_constant_prediction is deprecated because tiled emission is known-biased; "
        "use build_baseline_prediction(..., emit='dispersed') for a COUNTS reference. "
        "Dispersed emission is defined for counts only, so on a lognorm-effective reference "
        "emit='tile' remains the applicable arm -- keep using it there, deliberately, rather "
        "than as the default",
        DeprecationWarning,
        stacklevel=2,
    )
    return build_baseline_prediction(
        profile, template, pert_col=pert_col, control=control, emit="tile"
    )


def _prediction_from_adata(
    adata,
    profile: GenericProfile,
    *,
    pert_col: str,
    control: str,
    emit: Literal["dispersed", "tile"],
    seed: int,
):
    """``build_baseline_prediction`` on an already-materialized reference."""
    import anndata as ad

    if pert_col not in adata.obs.columns:
        raise ValueError(
            f"perturbation column {pert_col!r} missing from template obs; "
            f"present: {list(adata.obs.columns)}"
        )
    genes = np.asarray(adata.var.index.values).astype(str)
    if genes.size != profile.genes.size or not np.array_equal(genes, profile.genes):
        raise ValueError(
            "template var index does not match the profile's genes "
            f"({genes.size} vs {profile.genes.size} entries); the profile must be built "
            "from the same reference used as the template"
        )
    labels = adata.obs[pert_col].to_numpy().astype(str)
    ctrl_mask = labels == control
    if not ctrl_mask.any():
        raise ValueError(
            f"control {control!r} not found in template {pert_col!r}: the prediction's "
            "control rows must carry the real control's own cells"
        )

    if emit not in ("dispersed", "tile"):
        raise ValueError(f"emit must be 'dispersed' or 'tile', got {emit!r}")
    # in-memory by construction (_materialize_reference), so a plain slice suffices; an
    # in-memory sparse X yields a SparseCSRMatrixView, for which scipy.issparse IS True.
    ctrl_X = adata[ctrl_mask].X

    if emit == "tile":
        # This is intentionally the pre-#234 implementation byte for byte: np.tile creates
        # dense float32 output and sparse control rows are densified before assignment.
        X = np.tile(np.asarray(profile.values, dtype=np.float32), (adata.n_obs, 1))
        dense_ctrl = ctrl_X.toarray() if sp.issparse(ctrl_X) else np.asarray(ctrl_X)
        X[ctrl_mask] = dense_ctrl.astype(np.float32, copy=False)
        return ad.AnnData(
            X=X, obs=adata.obs.copy(), var=adata.var.copy(),
            uns={"baseline_emission": {"emit": "tile"}},
        )

    # The float64 recast is LOAD-BEARING, not redundant, and it was challenged as such
    # (Gemini, PR #241: "scipy's mean(axis=0) already accumulates in float64"). It does not:
    # measured on scipy 1.18.0, csr.mean(axis=0) on a float32 matrix returns float32 and
    # accumulates in float32. Over 40,000 values near 1e3 -- the regime of VCC's 38,176-cell
    # control pool -- that costs 5.8e-02 of absolute error against 6.8e-13 with the recast.
    # ctrl_pb is the DENOMINATOR of r, so the error would propagate into every emitted cell.
    if sp.issparse(ctrl_X):
        ctrl_pb = np.asarray(ctrl_X.astype(np.float64).mean(axis=0)).ravel()
    else:
        ctrl_pb = np.asarray(ctrl_X, dtype=np.float64).mean(axis=0)
    scale, scale_diag = _emission_scale(profile.values, ctrl_pb, genes)

    non_control = sorted(set(labels[~ctrl_mask].tolist()))
    group_positions = [np.flatnonzero(labels == label) for label in non_control]
    group_sizes = [int(rows.size) for rows in group_positions]
    emitted, source, kernel_diag = _emit_scaled_resample(
        ctrl_X, group_sizes, scale, np.random.default_rng(seed)
    )
    del source  # the library mirrors obs exactly and deliberately exposes no source_cell
    non_control_positions = (np.concatenate(group_positions) if group_positions
                             else np.empty(0, dtype=np.intp))
    control_positions = np.flatnonzero(ctrl_mask)

    if sp.issparse(adata.X):
        # copy=False: vstack only READS this block, and the row permutation below builds a
        # fresh matrix, so the prediction never aliases the template (Gemini, PR #241)
        ctrl_csr = ctrl_X.tocsr().astype(np.float32, copy=False)
        blocks = sp.vstack([emitted, ctrl_csr], format="csr")
        block_positions = np.concatenate([non_control_positions, control_positions])
        order = np.empty(adata.n_obs, dtype=np.intp)
        order[block_positions] = np.arange(adata.n_obs, dtype=np.intp)
        X = blocks[order].tocsr()
    else:
        X = np.empty((adata.n_obs, adata.n_vars), dtype=np.float32)
        X[non_control_positions] = emitted.toarray()
        X[control_positions] = np.asarray(ctrl_X).astype(np.float32, copy=False)

    # Summed over the ACTUAL prediction, not reconstructed as max(kernel_max, control_max).
    # That reconstruction was suggested (Copilot, PR #241) and is cheaper, but it does not
    # agree: measured on a dense template, 2 of 4 seeds differed (95.06866455078125 vs
    # 95.06865692138672) because the kernel sums a CSR row over its nonzeros in index order
    # while a dense row is summed pairwise over all n_genes. This number is quoted in the
    # scale-gate rejection message beside max_counts_per_cell, so it must be the value the
    # gate itself would measure on this matrix; one pass before a scoring run that traverses
    # it many times is the right trade.
    full_totals = np.asarray(X.sum(axis=1)).ravel()
    diagnostics = {
        "emit": "dispersed",
        "seed": int(seed),
        **scale_diag,
        **kernel_diag,
        "max_scaled_noncontrol_row_total": kernel_diag["max_row_total"],
        "max_row_total_full_prediction": (
            float(full_totals.max()) if full_totals.size else 0.0
        ),
    }
    return ad.AnnData(
        X=X, obs=adata.obs.copy(), var=adata.var.copy(),
        uns={"baseline_emission": diagnostics},
    )


def lock_matrix_space(real, pred, *, config: EvalConfig, de_pred=None, de_real=None) -> EvalConfig:
    """Make the reference and the baseline prediction resolve to the SAME matrix space.

    ``norm.resolve_input_type`` resolves each side INDEPENDENTLY, and under v1 (or v2 with
    ``autodetect_input_type``) it ignores the declared type and calls ``guess_is_lognorm``.
    The reference may be integer counts while a mean-scaled emission is fractional, so the
    prediction can detect as ``lognorm`` against a ``counts`` reference and every expression
    metric would silently compare two different spaces. ``allow_fractional_counts`` does
    not help: it gates ``validate_input_type``, not detection.

    This CHECKS the invariant rather than inferring it, because ``guess_is_lognorm`` tests
    fractional ROW TOTALS, not fractional entries: a fractional profile whose row total is
    an integer resolves to ``counts``, and it can do so against a ``lognorm`` reference --
    the same corruption in the opposite direction, which an inferred rule would leave in
    place (design 3.0, with the measured counterexample).

    * both sides already agree -- change nothing. Automatically true under v2 without
      ``autodetect_input_type``, so no inert flag is ever set and ``run_params.yaml`` never
      misdescribes the run.
    * they disagree and REAL resolves to ``counts`` -- set ``allow_discrete=True``, then
      re-resolve and require agreement.
    * they disagree and REAL resolves to ``lognorm`` -- raise. ``allow_discrete`` can only
      pull a side toward ``counts``, so the only reconciliation would re-interpret the
      REFERENCE, which is not the baseline's to redefine.

    ``allow_discrete`` stays IN the config digest; what makes forcing it safe is that
    ``config_digest`` is taken over the REQUESTED config, before this lock (design 3.0/6).
    """
    return _lock_from_adata(_materialize_reference(real), pred, config,
                            de_pred=de_pred, de_real=de_real)


def _lock_from_adata(adata, pred, config: EvalConfig, *, de_pred=None, de_real=None) -> EvalConfig:
    """``lock_matrix_space`` on an already-materialized reference.

    Check the matrices the SELECTED path will re-detect -- no more, no fewer.
    ``run._pred_de_input`` re-resolves SUBSETS independently, but only when a DE metric is
    computed with ``control_source="real"``: the CPU DE path takes the prediction's
    non-control cells (``run.py:533``) and the real control pool (``run.py:524``) and
    resolves each on its own (``run.py:545-546``), converting one to the other's scale if
    they disagree. ``target_sum=None`` separately re-resolves the real control pool
    (``run.py:765-773``). The prediction's emitted non-control rows can resolve differently
    from the full prediction (which mixes in copied control cells), so a two-matrix check
    can pass and then die inside DE at ``norm.py:441`` (*cannot recover counts from lognorm
    input (irreversible)*).

    Checking them UNCONDITIONALLY is worse than checking too few: a
    ``metrics=["expr_mae"]`` run with a numeric ``target_sum`` re-detects neither subset, so
    a subset disagreement nothing will ever look at would fail a run that was correct.
    """
    ctrl = adata.obs[config.pert_col].to_numpy().astype(str) == config.control
    pred_ctrl = pred.obs[config.pert_col].to_numpy().astype(str) == config.control
    from .run import _use_inmem_external_ref

    names = resolve_metrics(config.metrics, version=config.version)[0]
    # EXACT predicate, not _de_backend_used: _pred_de_input runs only when the PREDICTION's
    # DE is computed (run.py:834), and even then the non-control subset is taken only by the
    # CPU backends -- a resolved gpudge backend passes the FULL prediction as its DE target
    # (_use_inmem_external_ref, run.py:472-519).
    substitutes_control = (any(CATALOG[n].kind == "de" for n in names)
                           and de_pred is None
                           and config.control_source == "real")
    parts = [("real", adata, "real"), ("pred", pred, "pred")]
    if substitutes_control or config.target_sum is None:
        parts.append(("real_control_pool", adata[ctrl], "real"))
    if substitutes_control and not _use_inmem_external_ref(config):
        parts.append(("pred_non_control", pred[~pred_ctrl], "pred"))

    def _spaces(cfg):
        return {name: _effective_input_type(mat, cfg, side=side)
                for name, mat, side in parts}

    spaces = _spaces(config)
    if len(set(spaces.values())) == 1:
        return config
    real_eff, pred_eff = spaces["real"], spaces["pred"]
    if real_eff != "counts":
        raise ValueError(
            f"cannot lock the baseline's matrix space: {spaces}. allow_discrete can only "
            "force a side to 'counts', so reconciling this would re-interpret the "
            "REFERENCE. Under v1 the declared input_type is ignored entirely "
            "(norm.py:256-263), so the escapes there are allow_discrete=True -- only if "
            "the reference really is counts -- or moving to version='v2' with the correct "
            "declaration; under v2 the escape is autodetect_input_type=False."
        )
    forced = replace(config, allow_discrete=True)
    locked = _spaces(forced)
    if len(set(locked.values())) != 1:  # not reachable today; a guard, not a prediction
        raise ValueError(
            f"failed to lock the baseline's matrix space: allow_discrete=True still "
            f"leaves {locked}."
        )
    logger.warning(
        "baseline: the reference resolves to 'counts' but the fractional prediction "
        "resolves to %r; forcing allow_discrete=True so both sides are read in the same "
        "space. "
        "Recorded in the baseline stamp as allow_discrete_effective. The config digest is "
        "taken over the REQUESTED config, so this forcing does not make the baseline "
        "mismatch an ordinary run.",
        pred_eff,
    )
    return forced


@dataclass(frozen=True)
class BaselineResult:
    """Everything a baseline run produces: the tidy per-perturbation values, the wide
    aggregate ``score_metrics`` consumes, the profile, and the provenance stamp."""

    results: pl.DataFrame
    agg: pl.DataFrame
    profile: GenericProfile
    meta: dict


# Config fields excluded from the digest.
#   * outdir / cache_* / num_threads / gather_threads -- genuinely performance-only.
#   * allow_fractional_counts -- a validation ALLOWANCE, not a scoring semantic: it only
#     permits fractional values under input_type="counts" and changes no metric's math.
# device and pert_chunk are deliberately NOT exempt: run.py:398-403 keys the pseudobulk
# cache on the resolved device because fp32 (GPU) and fp64 (CPU) means differ, and
# pert_chunk governs GPU reduction blocking. control_source is not exempt either -- nothing
# forces it any more, so it must match like any other scoring knob. NOR is allow_discrete:
# it is value-affecting, and what makes it safe to force is that build_generic_baseline
# digests the REQUESTED config (below), so the forced value never reaches this function.
DIGEST_EXEMPT_FIELDS = frozenset({
    "outdir", "cache_real", "cache_pred", "cache_strict",
    "num_threads", "gather_threads",
    "allow_fractional_counts",
})


def _baseline_policy_dict(names, *, comparator: str) -> dict:
    """The per-metric policy the digest must cover, beyond the config's own fields.

    ``metric_normalization`` is load-bearing and was added in #264 PR2. The run-level
    ``comparator`` token alone does NOT identify the space each metric was computed in: PR1
    stamped ``comparator="bulk_lognorm"`` while the six remaining ``expr_*`` entries still
    declared ``normalization="lognorm"`` in the catalog, so a PR1 baseline and a PR2 run --
    same config, same comparator, and the same ``cell_eval2_version`` inside one unreleased
    cycle -- produced identical digests while computing different numbers, and ``cli.py``'s
    pairing check would have accepted the pair and published the margins. Hashing the
    RESOLVED per-metric normalization closes that: an artifact from either side of the move
    now fails the pairing check by digest instead of scoring.
    """
    d = {}
    d["metric_agg"] = [[n, CATALOG[n].agg] for n in names]
    d["metric_derived"] = derived_policy(names)
    d["metric_normalization"] = [[n, effective_normalization(CATALOG[n], comparator)]
                                 for n in names]
    return d


def config_digest(config: EvalConfig, *, comparator: str, de_real=None) -> str:
    """sha256 over the scoring-relevant config, for detecting a baseline/submission
    mismatch that ``score_metrics``'s column check cannot see.

    The column check catches a v1-vs-v2 mix (the versions emit different metric names). It
    cannot see a convention mismatch behind identical names -- a baseline built at
    ``de.p_adj_threshold=0.01`` scored against submissions at 0.05 yields
    identically-shaped frames and silently wrong margins. That is what this is for.

    CALL IT ON THE REQUESTED CONFIG, never the effective one. The baseline's deliberate
    deviations (allow_fractional_counts, allow_discrete, cache_pred) must not reach the
    digest, or every baseline mismatches every ordinary run by construction; but exempting
    the value-affecting ones is unsound in the other direction, since a baseline requested
    with allow_discrete=False and a user run requested with True would then digest
    identically while interpreting matrices in different spaces. Digesting what was ASKED
    FOR resolves both.

    ``metrics`` is normalized to its RESOLVED canonical list so a run using the profile
    name and a run using the equivalent explicit list agree -- otherwise the mismatch
    check would fire on two runs that are in fact identical. For the same reason the
    MACHINE-RESOLVABLE spellings are canonicalized to what they resolve to, exactly as
    ``run._result_config_digest`` does for the result-cache key (``run.py:660-676``):
    ``device="auto"`` and an explicit ``device="cpu"`` on one host produce identical
    numbers and must not mismatch, while two hosts that genuinely resolve differently
    still must. The DE backend is resolved only when a DE metric is actually requested
    with at least one side computed -- resolving it otherwise would make a minimal install
    raise for nothing (``run.py:663-670``).

    ``metric_agg`` records each resolved metric's AGGREGATION STATISTIC alongside its name
    (#231). Names alone cannot see a change of statistic: the 18 direction entries that moved
    from ``median`` to ``mean`` in v0.8.0 kept their names and their ``full``/``de``
    membership, so a v0.7 baseline and a v0.8 run would digest identically while their
    whole-cohort numbers answer different questions -- and the margins ``score_metrics``
    computes from them would be silently wrong. This is what makes an aggregation change a
    LOUD failure (``cli.py`` raises unless ``--allow-config-mismatch``) rather than a quiet
    one. The runtime version stamp is not a substitute: ``cell_eval2_version`` resolves
    through the INSTALLED distribution metadata, which in a dev tree need not be the tree
    under test.

    ⚠️ **KNOWN GAP -- no metric-SEMANTICS term (#314).** Everything above digests what the run
    ASKED FOR; nothing digests what a metric MEANS. A baseline built before a metric's
    definition changed, scored against submissions computed after it, is precisely the failure
    the second paragraph describes -- identically-shaped frames, silently wrong margins -- and
    it passes this digest. FIVE live instances: #172 (three scored ``vcc2026`` members stopped
    scoring each perturbation's own target gene), #248 (the same for ``pds_*``), the
    ``direction_reach_raw`` purity floor moving from ``1 - alpha/2`` to
    ``direction.REACH_PURITY_FLOOR``, **#271** (``prep._grouped_sums`` reduces WIDE, so the
    ``bulk_lognorm`` pseudobulk itself moved for coarse-float input), and **#348**
    (``expr_mse_unbiased_capped`` gained the ``r`` factor on the prediction's correction, and
    added no config field -- verified, ``config.py`` is untouched by that commit, where its
    same-wave sibling #343 DID add ``discrimination.exclusion_scope`` and so IS caught here).
    ⚠️ #348 is benign FOR THE MEASURED OFFICIAL BASELINES, and the SCOPE is the point -- read it
    before reusing any baseline across the bump. ``metrics.delta._numerator`` gates the whole #348
    block on ``if claim > 0.0``, so a baseline whose cells are IDENTICAL within a group has
    ``jk_pred`` -- and hence ``claim`` -- at 0, the block is SKIPPED, and its
    ``expr_mse_unbiased_capped_norm`` does not move. MEASURED so on the three official #276 val
    baselines, where a pre-#348 baseline paired with a post-#348 submission therefore still yields
    correct margins.
    ⚠️ **That does NOT generalize to any baseline.** ``build_generic_baseline`` defaults to
    ``emit="dispersed"``, whose groups hold RESAMPLED, heterogeneous cells -- so ``jk_pred`` and
    ``claim`` can be POSITIVE, ``r`` can bind, and #348 CAN move such a baseline's value with
    nothing here to see it. Nor is the identical-cell zero structural:
    ``moments.jackknife_correction`` reaches it through a floating-point cancellation behind a zero
    floor and declines to promise exact equality. So a dispersed -- or any unmeasured -- baseline is
    NOT safe to reuse across #348 until its claim is measured non-binding.
    ⚠️ What makes an instance DANGEROUS is the baseline's OWN value moving. Do NOT read the
    #343/#348 pair as a rule that the dangerous ones are the caught ones: that holds for that pair
    by coincidence, and #271 immediately below is the counterexample in this very list -- it moves a
    fractional baseline's value and adds no config field either. The list is what has to be
    consulted; there is no shortcut. ⚠️ #271 is also a DIFFERENT SHAPE
    from the other four: they change what a metric MEANS, it changes the pseudobulk every
    expression/PDS member reads -- so a pre-#271 fractional baseline scored against post-#271
    submissions is the second-paragraph failure with nothing in this digest to see it. It is listed
    rather than patched for the reason the next paragraph gives, and its own constant
    (``run._GROUPED_SUM_REDUCTION_SEMANTICS``) is already carried by the other three surfaces
    (result cache, ``partition.result_semantics``, ``anchor.anchor_semantic_params``), so #314's
    registry has one more entry to absorb rather than a new idea. The third is the sharpest case for #314's shared registry:
    the floor IS a single value, so a registry entry reading it needs no per-issue constant at
    all -- ``run._result_config_digest``, ``partition.result_semantics`` and
    ``anchor.anchor_semantic_params`` each already key on the value directly, and this is the
    one surface of the four that does not.
    Deferred rather than patched (Alex, 2026-08-17) because a per-issue constant here would be
    the third copy of one idea: ``run._result_config_digest`` and ``anchor``'s
    ``anchor_semantic_params`` each already carry their own ``ontarget_exclusion_semantics``,
    and #314 proposes the shared registry all three should read instead.
    ``cell_eval2_version`` is not the fallback -- see the paragraph above, and note that #172
    lands WITHIN 0.13.0, so the stamp does not move for it at all.

    Deliberately NOT mirrored into ``run._result_config_digest``. That digest keys the RESULT
    cache, whose artifact is the per-perturbation tidy frame (``run.py``); ``agg`` is applied
    only afterwards, ``cache.config_hash`` strips ``metrics`` outright, and the resolved names
    reach the result key through the fingerprint instead. Adding ``agg`` there would buy a
    pure spurious cache miss. The two digests share policy only for inputs that move
    per-perturbation numbers, and they are never compared to each other.
    """
    d = {k: v for k, v in config.to_dict().items() if k not in DIGEST_EXEMPT_FIELDS}
    d["comparator"] = comparator
    if config.target_gene_map is None:
        # Drop the field when inert, EXACTLY as run._result_config_digest does (#195).
        # to_dict() picks up every new EvalConfig field, so leaving it in changes this
        # digest for a default config that scores identically -- and unlike the result
        # cache, a stale value here is not a silent recompute but a hard failure:
        # cli.py:170 raises SystemExit("baseline/user mismatch -- the margins would be
        # meaningless") unless --allow-config-mismatch is passed. Every baseline stamped
        # before #195 would have rejected every run after it. The two digests must agree
        # on this policy; they diverged because #195 originally fixed only the run path.
        d.pop("target_gene_map", None)
    names = resolve_metrics(config.metrics, version=config.version)[0]
    d["metrics"] = names
    # Ordered, and parallel to `names` -- a list of pairs rather than a dict so the payload
    # cannot depend on mapping-iteration order (`json.dumps(sort_keys=True)` would sort a
    # dict, but the pairs also keep the two fields obviously in step).
    d.update(_baseline_policy_dict(names, comparator=comparator))
    d["device"] = _cache_device(config)
    d["de"]["backend"] = (_cache_backend(config)
                          if _de_backend_used(config, names, de_real)
                          else config.de.backend)
    payload = json.dumps(d, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode()).hexdigest()


#: The numeric stack every metric value passes through, recorded beside the DE engine in
#: ``build_run_meta``'s ``environment`` block. ``pyproject.toml`` declares LOWER BOUNDS only
#: (``polars>=1.0``, ``numpy>=1.26``, ``scipy>=1.11``, ``anndata>=0.10``, ``scanpy>=1.10``),
#: releases are git tags, and nothing enforces ``uv.lock`` at install time -- so two installs
#: that satisfy the same declaration can compute different numbers with nothing in any
#: artifact to say which one ran (#338's premise). ``cell_eval2`` itself is deliberately
#: absent: ``cell_eval2_version`` already carries it, four fields up.
_ENVIRONMENT_PEERS = ("polars", "numpy", "scipy", "anndata", "scanpy")

#: PEP 610's named version-control systems. A CLOSED set on purpose: `_distribution_provenance`
#: reads `vcs` out of a file on disk, and the whole point of the token is that a consumer can
#: trust it, so an unrecognised value becomes the generic `vcs` rather than reaching the record.
_VCS = ("git", "hg", "svn", "bzr")


def _distribution_provenance(dist: Distribution) -> str | None:
    """How an installed distribution GOT here, as one fixed token -- never its URL or path.

    ⚠️ The token is the point, not decoration. ``importlib.metadata.version`` reads
    INSTALL-TIME metadata, so an editable install reports whatever its tree declared when it
    was linked and goes on reporting it as that tree moves. MEASURED 2026-08-18: the ``-r2``
    official bundles were computed by gpudge ``3a71cc5`` (``v0.7.0-4``, four commits past the
    tag) while ``version("gpudge")`` said ``0.7.0``. A bare version string would therefore
    record a plausible FALSEHOOD in exactly our situation; ``local-editable`` beside it says
    "this is an install-time claim about a mutable tree, do not read it as a revision".

    ⚠️ NEVER RECORD THE URL OR THE PATH, only the classification. PEP 610's
    ``direct_url.json`` holds an absolute ``file://`` path into the build host's own home
    directory for the editable case, and a PRIVATE ``ssh://git@github.com/...`` URL for the VCS
    one -- and this block reaches ``baseline_meta.json`` inside every shipped bundle, so the
    path would publish an internal filesystem layout and the URL a private repository. (Neither
    literal is written out here. ``src/**`` ships, and the path in particular is a
    ``check_publish_set.SWEEP_TOKENS`` entry, so quoting it would have made this very file a
    publish blocker -- it did, in this change's first commit.) The token answers the only
    question a consumer has -- "can I trust this version string?" -- and leaks neither.
    ``test_environment_provenance.py`` asserts the absence directly, because a value that
    exists only at runtime is invisible to the publish sweeps.

    The tokens map onto PEP 610's three mutually exclusive sub-keys: no ``direct_url.json``
    -> ``release`` (resolved from an index); ``vcs_info`` -> that checkout's VCS, ``git`` in
    every case this repository has; ``dir_info`` with ``editable`` -> ``local-editable``; any
    other LOCAL source, unpacked directory or built wheel alike -> ``local``; a remote artifact
    fetched by direct URL -> ``archive``. ``archive`` is not padding: without it a direct-URL
    install would fall into ``release``, the token a reader trusts MOST, which is the one
    mislabelling that actually costs something.

    The VCS is READ rather than assumed. Answering ``git`` for a mercurial or subversion
    checkout would be a small lie of exactly the kind this whole field exists to prevent, and
    ``_VCS`` bounds the answer to PEP 610's four named systems so an unrecognised value becomes
    the generic ``vcs`` instead of an arbitrary string from a file on disk.

    ``file:`` is matched CASE-FOLDED because a URI scheme is case-insensitive (RFC 3986 3.1):
    a ``FILE:///...`` direct URL is a local install and must not be reported as a remote
    ``archive``. Only the comparison is folded -- the URL itself is never recorded.

    ⚠️ A ``direct_url.json`` that is not a JSON OBJECT (a list, a bare ``null``, a number) yields
    ``None`` -- UNCLASSIFIED -- and deliberately not ``release``, which Gemini's suggested patch
    proposed (PR #347). ``release`` means "resolved from an index", and the mere PRESENCE of this
    file is positive evidence that it was not: that is the whole basis for the ``release`` branch
    two lines up. Answering ``release`` for an unreadable payload would put the token a reader
    trusts MOST on the case we know LEAST about -- the one mislabelling this docstring already
    says actually costs something. ``None`` says "there is direct-URL provenance here and it
    could not be read", which is both true and useful.
    """
    raw = dist.read_text("direct_url.json")
    if raw is None:
        return "release"
    info = json.loads(raw)
    if not isinstance(info, dict):
        return None
    vcs_info = info.get("vcs_info")
    if isinstance(vcs_info, dict):
        vcs = vcs_info.get("vcs")
        return vcs if vcs in _VCS else "vcs"
    dir_info = info.get("dir_info")
    if isinstance(dir_info, dict) and dir_info.get("editable"):
        return "local-editable"
    return "local" if str(info.get("url", "")).lower().startswith("file:") else "archive"


def _package_provenance(name: str) -> dict | None:
    """``{"version", "provenance"}`` for one installed distribution, or ``None`` when absent.

    ``None`` is a legitimate answer rather than a failure: CI installs no gpudge, so the DE
    backend's entry is null there, and a minimal install carries no scanpy. A malformed or
    unreadable ``direct_url.json`` nulls the CLASSIFICATION only -- the version is still worth
    recording, and losing it to a broken sidecar would be the provenance tail wagging the dog.

    THE TWO HALVES ARE INDEPENDENT, and each is guarded on its own. ``dist.version`` reads
    ``METADATA`` and can itself raise on a corrupt or unreadable one (Gemini, PR #347), so a
    package can legitimately record a provenance with no version, or a version with no
    provenance. Nulling BOTH because one failed would throw away information this record exists
    to carry -- and letting either escape would make the caller's guard the only thing standing
    between a damaged install and a lost run.
    """
    try:
        dist = Distribution.from_name(name)
    except PackageNotFoundError:
        return None
    try:
        version = dist.version
    except Exception:
        version = None
    try:
        provenance = _distribution_provenance(dist)
    except Exception:
        provenance = None
    return {"version": version, "provenance": provenance}


def _environment_record(resolved_de_backend: str | None) -> dict:
    """WHAT COMPUTED THE NUMBERS: the resolved DE engine plus the numeric stack, each as
    ``{"version", "provenance"}``.

    Nested under one new top-level key on purpose. A single key cannot collide with a
    compared field name -- ``check_submission`` iterates ``SUBMISSION_PEERS`` and
    ``cli._check_baseline_config`` a fixed tuple, so extra keys are inert on both paths -- and
    the nesting reads unmistakably as provenance rather than as something to gate on.

    ⚠️ It does NOT reach ``manifest.json``, and that is deliberate. The manifest copies
    ``**{f: baseline_meta.get(f) for f in SUBMISSION_PEERS + MANIFEST_RECORDED_ONLY}``
    (``real_bundle.py:399``) and ``read_real_bundle`` requires EVERY name in both tuples to be
    present in the manifest it reads (``real_bundle.py:509-519``), so adding this to either
    tuple would make all three frozen official val bundles unreadable outright -- measured,
    #291. It does reach ``baseline_meta.json``, a file inside every bundle
    (``real_bundle.py:226`` builds it, :438 writes it). The bundle directory records the
    environment; the manifest does not. That is this record's honest limitation.

    BEST-EFFORT THROUGHOUT: nothing here may raise, mirroring ``anchor._version()``'s "never
    lose an anchor to provenance". A failure yields a partial or empty block, never a lost run.
    """
    record: dict = {}
    try:
        targets = {p: p for p in _ENVIRONMENT_PEERS}
        if resolved_de_backend is not None:
            # Lazy, same doctrine as `_cache_backend` -- and MEASURED, because on an
            # EXPLICIT-backend run `_cache_backend` returns without importing `de_compute` at
            # all, so this can be the cold import (Codex, checkpoint 2). It costs 0.5 ms: the
            # `scipy.stats` that module imports is ALREADY in `sys.modules` by the time
            # `cell_eval2.baseline` finishes importing, so nothing heavy is paid here.
            from .de_compute import _BACKEND_MODULE
            # KEYED BY THE BACKEND TOKEN so the entry cross-references `resolved_de_backend`
            # by name; only `deseq2` differs from its distribution (`deseq2_gpu`), and only
            # `scanpy` can collide with a peer of the same name -- same distribution, same
            # value, one entry.
            #
            # ⚠️ A TOKEN THIS MAP DOES NOT KNOW IS DROPPED, never used as its own fallback
            # distribution name. `EvalConfig.de.backend` is a `Literal` and `_cache_backend`
            # returns either that or `_resolve_backend`'s output, so an arbitrary string cannot
            # reach here through the public API -- but "never records a path" has to hold on
            # the FUNCTION, not on validation somewhere up the stack, and a fallback of
            # `.get(tok, tok)` would put a caller-supplied string straight into a shipped
            # artifact as a key. A backend absent from this map is unusable anyway:
            # `de_compute._available` returns False for it, so it can never have run.
            if resolved_de_backend in _BACKEND_MODULE:
                targets[resolved_de_backend] = _BACKEND_MODULE[resolved_de_backend]
            else:
                logger.debug("run_meta: DE backend %r is not in _BACKEND_MODULE; its version "
                             "is not recorded.", resolved_de_backend)
        for key, dist_name in targets.items():
            # PER-TARGET, so one distribution with unreadable metadata cannot truncate the key
            # set and leave a reader guessing whether the rest was absent or merely unrecorded.
            try:
                record[key] = _package_provenance(dist_name)
            except Exception:
                record[key] = None
    except Exception:            # never lose a run to provenance
        logger.debug("run_meta: environment block incomplete", exc_info=True)
    return record


def build_run_meta(config: EvalConfig, real, pred, *, de_real=None, de_pred=None) -> dict:
    """The resolved-identity record an ordinary ``run`` writes, so ``score`` can compare it
    against the baseline stamp. Same fields, same names, same resolution helpers -- keeping
    it here rather than in cli.py is what stops the two records drifting.

    IT MUST NOT CHANGE WHAT ``run`` COSTS. ``compute_metrics`` opens path inputs BACKED
    (``run.py:723-732``) and can return from a result-cache hit without ever reading ``X``
    (``run.py:816``). Materializing and content-hashing both sides for a provenance file
    would destroy both properties on the hottest path in the repo. So this reads BACKED and
    never materializes: ``fingerprint_adata`` is metadata-only unless ``cache_strict``
    (shape, dtype, var index, per-cell labels -- it deliberately never reads ``X``,
    ``cache.py:88-100``), and ``_effective_input_type`` costs at most a 500-row sample
    (``norm.py:226-245``), or nothing under v2 without ``autodetect_input_type``. The one
    exception is a cell-layout cellstream archive, which ``load_anndata`` materializes even
    with ``backed=True`` (``io.py:18-22``) -- there is no backed reader for that format, so
    this pass costs the same as the scoring read does.
    ``source_fingerprint_strict`` is recorded so the two sides never compare a metadata
    hash against a content hash.

    ⚠️ The contract above is about the MATRICES -- never materialize ``X``, never content-hash a
    side -- and the ``environment`` block does add I/O that is not of that kind: six
    ``importlib.metadata`` distribution lookups, each reading at most ``METADATA`` and
    ``direct_url.json``. MEASURED on the reference venv 2026-08-19: **7.2 ms** for the whole
    block, ~0.8 ms per distribution, once per call -- and this function is called exactly once
    per run (``cli.py:542``) and once per bundle leg (``real_bundle.py:226``), never inside a
    per-perturbation loop. Against the two backed h5ad opens and the up-to-500-row
    ``_effective_input_type`` sample this same function already performs, that is noise. It is
    written down rather than left to be discovered, because "must not change what ``run``
    costs" read literally forbids any new I/O at all (Codex, checkpoint 2), and the exemption
    is deliberate and bounded rather than an oversight.

    Returns the dict; the CALLER writes it only after the aggregate it describes has been
    written successfully. Publishing first would leave new metadata beside a STALE
    ``agg_results.csv`` in a reused output directory when the run then fails, and ``score``
    would certify the stale aggregate.
    """
    names, _ = resolve_metrics(config.metrics, version=config.version)
    meta = {
        "cell_eval2_version": _version_lazy(),
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source": str(real) if isinstance(real, (str, os.PathLike)) else "<in-memory AnnData>",
        "resolved_device": _cache_device(config),
    }
    # Resolved into a local so the `environment` block can record the SAME backend rather than
    # resolve it twice, and resolved HERE -- at the exact point in the sequence the dict literal
    # used to reach it. Neither neighbour is pure: `_cache_device("auto")` probes the GPU driver
    # through `resolve_device`, and `_cache_backend` on `backend="auto"` probes for an installed
    # engine and CAN raise (`resolve_device("auto")` cannot -- every driver failure returns
    # "cpu", `gpu/__init__.py:50-58`). So this position is not load-bearing for which error a
    # broken host reports; it keeps `created_utc` stamped BEFORE the engine probe rather than
    # after it, and it costs nothing to leave the sequence exactly as it was. (Codex, checkpoint
    # 2, twice over: the first version of this hoist claimed "everything above it is pure",
    # which was wrong, and the second claimed both neighbours could raise, which was also wrong.)
    resolved_de_backend = (_cache_backend(config)
                           if _de_backend_used(config, names, de_real)
                           else None)
    meta.update({
        "resolved_de_backend": resolved_de_backend,
        # WHAT COMPUTED THE NUMBERS. `resolved_de_backend` names the engine and nothing named
        # its VERSION: gpudge drives four of the six scored `vcc2026` members, and the `-r2`
        # bundles record the string 'gpudge' with no version anywhere in `manifest.json`,
        # `baseline_meta.json` or `anchor_meta.json`. See `_environment_record` for why it is
        # nested, why it stops short of the manifest, and why the version alone would lie.
        "environment": _environment_record(resolved_de_backend),
        "de_real_fingerprint": (fingerprint_de_table(load_de_table(de_real), strict=True)
                                if de_real is not None else None),
        "de_pred_fingerprint": (fingerprint_de_table(load_de_table(de_pred), strict=True)
                                if de_pred is not None else None),
    })
    effective_types = {}
    for side, source, fp_field, ty_field in (
        ("real", real, "source_fingerprint", "input_type_real_effective"),
        ("pred", pred, None, "input_type_pred_effective"),
    ):
        adata = load_anndata(source, backed=isinstance(source, (str, os.PathLike)))
        try:
            if config.pert_col not in adata.obs.columns:
                raise ValueError(
                    f"perturbation column {config.pert_col!r} missing from the {side} side's "
                    f"obs; present: {list(adata.obs.columns)}"
                )   # else fingerprint_adata raises a bare KeyError from cache.py:95
            if fp_field:
                meta[fp_field] = fingerprint_adata(adata, pert_col=config.pert_col,
                                                   strict=config.cache_strict)
            effective_types[side] = _effective_input_type(adata, config, side=side)
            meta[ty_field] = effective_types[side]
            if side == "real":
                # #276: what `score.expect_from_run_meta` reads to check a supplied or
                # cached anchor against THIS run rather than against the anchor's own
                # sidecar. Computed INSIDE the real-side iteration, while that side's
                # backed handle is open: the loop rebinds `adata` on the next pass and
                # closes it in the `finally`, so after the loop `adata` is the PREDICTION
                # and may already be closed.
                #
                # The import MUST stay in the function body -- this is a real cycle, not a
                # style preference. `lfc_nmae_ref.py:49` does `from .baseline import
                # _materialize_reference` at module scope, so a module-level
                # `from .anchor import semantic_identity` here closes
                # `baseline -> anchor -> lfc_nmae_ref -> baseline`. Do not let a tidy-up or
                # an import linter hoist it.
                #
                # Nothing here forces materialization: `anchor_semantic_params` calls
                # `_effective_input_type` (already called on this same backed handle, one
                # line up), plus `resolve_comparator` and `_is_discrimination`, both pure.
                # The never-materialize-X contract above is preserved -- the STRICT
                # fingerprint is not computed here, only the semantic identity.
                from .anchor import AnchorBackendUnresolved, semantic_identity
                try:
                    # BOTH into locals first, then ONE update: assigning them sequentially
                    # inside the `try` would let a failure in the second leave only the
                    # first, breaking the "omit both" contract `expect_from_run_meta`
                    # relies on.
                    _anchor_sem = semantic_identity(config, adata, list(names))
                    _anchor_names = metric_output_names(config)
                except AnchorBackendUnresolved as exc:
                    # OMITTED, not fatal, and this is the point of the try. The anchor's
                    # identity includes the RESOLVED DE backend, because the anchor always
                    # computes its own DE. An ORDINARY run need not: `--de-pred P
                    # --de-real R` supplies both tables, and `_de_backend_used` above is
                    # written precisely so such a run never resolves `backend="auto"` --
                    # which would demand an installed engine and raise on a CUDA host
                    # without gpudge. Computing the anchor identity unconditionally would
                    # reintroduce that failure on a supported path, for a field the run
                    # will never use.
                    #
                    # Omitting keeps `expect_from_run_meta` FAIL-CLOSED: a later
                    # `score --anchor` against this run refuses by name rather than
                    # scoring against an unverified top end.
                    logger.debug(
                        "run_meta: anchor identity not recorded (%s). `score --anchor` "
                        "will refuse against this run; ordinary scoring is unaffected.",
                        exc,
                    )
                else:
                    meta.update({"anchor_semantic_identity": _anchor_sem,
                                 "anchor_metric_names": _anchor_names})
        finally:
            _close_backed(adata, source)
    comparator = norm.resolve_comparator(
        version=config.version,
        pred_input_type=effective_types["pred"],
        real_input_type=effective_types["real"],
    )
    meta["comparator"] = comparator
    # The comparator token alone does not say what SPACE each metric was computed in: PR1
    # stamped `bulk_lognorm` while six `expr_*` entries were still declared `lognorm`
    # (#264 PR2). The digest covers this, but a digest only answers "same or different" --
    # two PR1 artifacts agree with each other. Stamping the map makes an artifact say what
    # it is, so a consumer can check it against current semantics rather than against a
    # peer. Ordered pairs, parallel to `metrics`, for the same reason the digest uses them.
    meta["metric_normalization"] = [
        [n, effective_normalization(CATALOG[n], comparator)]
        for n in resolve_metrics(config.metrics, version=config.version)[0]
    ]
    meta["config_digest"] = config_digest(
        config, comparator=comparator, de_real=de_real,
    )
    meta["source_fingerprint_strict"] = config.cache_strict
    return meta


def write_json_meta(meta: dict, path: str) -> None:
    """``allow_nan=False``: ``json.dump`` emits a bare ``NaN`` token by default, which is not
    valid JSON. Non-finite values are already recorded as ``None``, so this only fires if
    some other non-finite reached the record -- and then it should."""
    with open(path, "w") as fh:
        json.dump(meta, fh, indent=2, sort_keys=True, allow_nan=False)


def _de_backend_used(config: EvalConfig, names, de_real=None) -> bool:
    """Whether the DE engine identity is part of the identity a BASELINE PAIRING compares:
    a DE metric is requested AND the REAL side's table is COMPUTED rather than supplied.

    The real side is the one the two runs share, so it is the only side whose engine can
    make them incomparable. Keying on "either side computed" (``run._result_config_digest``'s
    predicate, run.py:663-670) breaks a supported pairing: the baseline's prediction is
    synthetic, so its DE is ALWAYS computed, while an ordinary ``run --de-pred P --de-real R``
    computes neither -- the two would record ``pdex`` vs ``null`` and be rejected for a
    difference on the PREDICTION side. That is the same reason ``de_pred_fingerprint`` is
    recorded but never compared (design section 6): the prediction side is expected to differ.

    This is never MORE eager than run.py's predicate (real-computed implies either-computed),
    so it still never resolves ``auto`` for a run that needs no engine -- which would depend
    on a backend being installed, and would raise on a CUDA host without gpudge.
    """
    if not any(CATALOG[n].kind == "de" for n in names):
        return False
    return de_real is None


def baseline_config(config: EvalConfig) -> EvalConfig:
    """The config a baseline run must actually use.

    ``control_source`` is NOT touched. An earlier draft forced it to "real"; the real
    control CELLS in the prediction's control rows (``_prediction_from_adata``) make both
    settings produce identical numbers, so forcing would only change the v1 estimand and
    force a digest exemption, for nothing.

    ``allow_fractional_counts=True`` -- the profile is a MEAN, hence fractional in any
    space. The flag only *permits* fractional counts, so it is inert on lognorm input and
    needs no input_type branch here. It is a validation allowance, not a scoring semantic,
    so it is digest-exempt.

    ``cache_pred=None`` -- non-strict ``fingerprint_adata`` (``cache.py:88-100``;
    ``cache_strict=False`` is the DEFAULT) hashes shape, dtype, var index and per-cell
    labels, and deliberately never reads X. Predictions from two seeds, two emit modes or
    the two ``exclude_target_gene`` arms therefore fingerprint identically for the same
    reference even though their values differ. A warm pred cache could silently return
    one construction's pseudobulk and DE for another. ``cache_real`` is untouched and
    SHOULD be reused: the real side is genuinely identical across the baseline and every
    submission, and it is the expensive one.

    ``de.backend="deseq2"`` is REJECTED: the profile is a mean, so the prediction's
    pseudobulk is fractional, and fractional input to a negative-binomial GLM is not
    statistically meaningful (``deseq2_de.py:26-32`` requires input_type="counts" for the
    same reason). Failing loud beats emitting a number nobody should use.
    """
    _reject_unsupported(config)
    changes: dict = {}
    if not config.allow_fractional_counts:
        changes["allow_fractional_counts"] = True
    if config.cache_pred is not None:
        logger.warning(
            "baseline: disabling cache_pred (%r). Seeds, emit modes and both "
            "exclude_target_gene arms share a non-strict fingerprint, so a warm pred "
            "cache could silently return another construction's results. cache_real is "
            "kept.",
            config.cache_pred,
        )
        changes["cache_pred"] = None
    return replace(config, **changes) if changes else config


def _reject_unsupported(config: EvalConfig) -> None:
    """Configs the baseline cannot serve. Called first, before any I/O, so the failure
    costs nothing."""
    if config.de.backend == "deseq2":
        raise ValueError(
            "the generic-response baseline does not support de.backend='deseq2': the "
            "profile is a MEAN, so the prediction's pseudobulk is fractional, and "
            "fractional input to a negative-binomial GLM is not statistically "
            "meaningful. Build the baseline with a wilcoxon backend "
            "(gpudge/pdex/scanpy)."
        )


def _degenerate_metrics(agg: pl.DataFrame, *, statistic: str = "mean") -> list[dict]:
    """Scoreable metrics whose baseline aggregate is unusable (design 7.1).

    Each entry carries ``decisive`` (``catalog.is_decisive``), which says what will happen to
    the artifact LATER, at scoring time: a decisive offender makes ``score_metrics`` refuse it
    outright, while any other scored metric is dropped from ``avg_score`` with a warning. Every
    ``vcc2026`` member is decisive as of #255, so all six are refused at scoring time. This does
    NOT govern the build gate -- ``build_generic_baseline`` refuses to WRITE an
    artifact with any offender unless ``allow_degenerate=True``, whichever kind it is, because
    writing a baseline is a deliberate act and a surprising one is worth stopping on.

    ``statistic`` is a parameter because ``score --comparison-statistic`` accepts ANY row in
    the frame, and a ``std`` row is NaN wherever the sample std is undefined (one
    perturbation, or none) -- so a baseline whose ``mean`` row is perfectly healthy can still
    be unscoreable on the row actually being consumed. The build-time gate checks ``mean``;
    ``score`` re-checks whatever it was actually asked for. (Before spec 6 that case was
    worse than unscoreable: it was scored, silently, as 0.0 for every submission.)

    A non-finite value is recorded as ``None`` so the stamp stays valid JSON: Python's
    ``json.dump`` emits a bare ``NaN`` token by default, which strict readers reject.
    """
    row = agg.filter(pl.col(agg.columns[0]) == statistic).drop(agg.columns[0])
    if row.height == 0:
        raise ValueError(f"aggregate has no {statistic!r} row to validate")
    out: list[dict] = []
    for name, val in zip(row.columns, row.row(0)):
        spec = CATALOG.get(_NAME_TO_CANONICAL.get(name, name))
        if spec is None or not spec.scoring.scored:
            continue          # unscored by score_metrics -> a degenerate value is inert
        if not is_degenerate(val, spec.scoring):
            continue
        if val is None or not np.isfinite(val):
            reason = "non-finite"
            val = None
        else:
            # Report the ACTUAL D rather than restating the rule. "denominator <= 0" is
            # wrong for the overflow case (D = inf), and "not finite and > 0" reads as
            # "not finite" for the far commoner D <= 0 case; the number itself is
            # unambiguous for both and says which one happened.
            reason = (f"denominator {denominator(float(val), spec.scoring)!r} is not a "
                      f"finite positive number (anchor={spec.scoring.anchor!r}, "
                      f"direction={spec.scoring.direction!r})")
        out.append({"metric": name, "statistic": statistic,
                    "value": None if val is None else float(val),
                    "direction": spec.scoring.direction,
                    "decisive": is_decisive(spec),
                    "reason": reason})
    return out


def _degenerate_message(offenders, *, allow_degenerate: bool) -> str:
    detail = "; ".join(f"{d['metric']}={d['value']!r} ({d['reason']})" for d in offenders)
    decisive = [d["metric"] for d in offenders if d.get("decisive", True)]
    skippable = [d["metric"] for d in offenders if not d.get("decisive", True)]
    msg = (f"degenerate baseline aggregate for {len(offenders)} scoreable metric(s): {detail}. "
           f"Drop them from the metric profile or investigate the degeneracy.")
    if decisive:
        # A decisive offender is fatal at scoring time regardless of what else is here, so
        # this branch must not be softened by the skippable one below.
        msg += (f" score_metrics REFUSES an artifact degenerate on {sorted(decisive)} "
                "(spec 6) -- these decide a ranking.")
    if skippable and not decisive:
        msg += (f" {sorted(skippable)} would be EXCLUDED from avg_score with a warning rather "
                "than refused, so the artifact stays scoreable on the remaining metrics "
                "(and score_metrics raises if none remain).")
    elif skippable:
        msg += (f" {sorted(skippable)} would additionally be excluded from avg_score, but the "
                "refusal above applies first.")
    if not allow_degenerate:
        return msg + (" Pass allow_degenerate=True (--allow-degenerate-baseline) to write it "
                      "anyway.")
    return msg + " Recorded in the stamp (allow_degenerate=True)."


def _require_scoreable(agg: pl.DataFrame) -> None:
    """At least one column must actually contribute to ``avg_score``.

    ``aggregate_metrics_wide`` only refuses a frame with NO metric columns, and
    ``_degenerate_metrics`` skips every unscored column as inert. A profile of purely
    diagnostic metrics (e.g. only ``de_wilcoxon_nsig_counts_real``) therefore sails through
    both and reaches ``score_metrics``, which scores nothing and falls back to
    ``avg_score = 0.0`` -- a number that reads like a result. Same hazard as the empty-set
    raise, one level up.
    """
    cols = [c for c in agg.columns if c != agg.columns[0]]
    scoreable = [c for c in cols
                 if (spec := CATALOG.get(_NAME_TO_CANONICAL.get(c, c))) is not None
                 and spec.scoring.scored]
    if not scoreable:
        raise ValueError(
            f"no scoreable metric in the baseline aggregate (columns: {cols}). Every one is "
            "scored=False, so score_metrics would score nothing and report a vacuous "
            "avg_score of 0.0. Select at least one scoreable metric."
        )


def build_generic_baseline(
    real,
    *,
    config: EvalConfig,
    exclude_target_gene: bool = True,
    emit: Literal["dispersed", "tile"] = "dispersed",
    seed: int = 0,
    de_real=None,
    save_pred=None,
    allow_degenerate: bool = False,
) -> BaselineResult:
    """Build the generic-response prediction and score it as an ORDINARY submission.

    No metric is special-cased anywhere: that is what makes the comparator general, and
    what makes it cover metrics added later for free. The prediction's own significant
    gene set is its own, so the direction metrics get the SELF-SELECTED precision
    semantics -- a weaker comparator than grading the baseline on the model's gene set,
    which a two-way ``(pred, real)`` scoring call cannot express. See section 2 of the
    design.

    ``de_real`` is forwarded to ``compute_metrics``: the real side is a genuine input and
    its DE is the expensive half. There is no ``de_pred`` -- the prediction is synthetic,
    so its DE must be computed from it (design section 8).
    """
    _reject_unsupported(config)              # cheapest possible rejection, before any I/O
    # Load a supplied DE table ONCE: the same frame goes to compute_metrics and to the
    # fingerprint, so the stamp cannot describe a different file than the one that scored.
    de_real = load_de_table(de_real) if de_real is not None else None
    real_ad = _materialize_reference(real)

    # _profile_from_adata validates pert_col AND the control label from obs, so everything
    # after this point can assume both. Fingerprinting first would turn a mistyped pert_col
    # into a bare KeyError from cache.py:95.
    # ⚠️ `config.target_gene_map` -- the field was already in this builder's `config_digest`
    # (it is dropped there only when None), so before #253/#285 two baselines differing ONLY
    # in the map digested differently and came out numerically identical. Threading it makes
    # the digest describe the artifact again.
    profile = _profile_from_adata(
        real_ad, pert_col=config.pert_col, control=config.control,
        exclude_target_gene=exclude_target_gene,
        target_gene_map=config.target_gene_map,
    )
    # Effective input type belongs to the REQUESTED config. Locking or baseline_config
    # first would let builder-specific changes influence a precondition on the reference.
    if (emit == "dispersed"
            and _effective_input_type(real_ad, config, side="real") == "lognorm"):
        raise ValueError(
            "emit=\"dispersed\" is defined for counts input only; the reference resolves "
            "to lognorm under the requested EvalConfig. Select emit=\"tile\" for the "
            "legacy arm that still applies to lognorm input."
        )
    if emit == "tile":
        logger.warning(
            "baseline: emit='tile' is known-biased and exists only to reproduce pre-fix "
            "numbers; emit='dispersed' is the supported default."
        )
    pred = _prediction_from_adata(
        real_ad, profile, pert_col=config.pert_col, control=config.control,
        emit=emit, seed=seed,
    )
    effective = baseline_config(_lock_from_adata(real_ad, pred, config, de_real=de_real))
    # Same semantics as build_run_meta, or the two records would compare a metadata hash
    # against a content hash and mismatch by construction.
    source_fp = fingerprint_adata(real_ad, pert_col=config.pert_col,
                                  strict=effective.cache_strict)

    if emit == "dispersed":
        emission = pred.uns["baseline_emission"]
        try:
            results = compute_metrics(pred, real_ad, config=effective, de_real=de_real)
        except norm.ScaleLimitError as e:
            # The ordinary v2 scale gate runs inside compute_metrics. A rejected build
            # never gets a stamp, so carry the construction diagnostics in the exception;
            # retaining the original text and chaining preserves the actual failure.
            #
            # ...but ONLY when the PREDICTION is the side that violated the cap.
            # compute_metrics gates the REAL side too (run.py:851, `side == "real"`), so a
            # ScaleLimitError is not necessarily about this construction, and bolting
            # "dispersed-emission diagnostics" onto a reference-side rejection would point
            # the reader at the wrong matrix (Copilot, PR #241, from its suppressed bucket --
            # narrowing the catch to a TYPE at checkpoint 2 was not enough, the SIDE matters
            # too). The test is the counts rule the construction actually guarantees; if the
            # prediction is inside the cap, re-raise untouched rather than guess.
            if emission["max_row_total_full_prediction"] <= effective.max_counts_per_cell:
                raise
            raise norm.ScaleLimitError(
                f"baseline compute_metrics failed: {e}. Dispersed-emission diagnostics: "
                f"r_max={emission['r_max']}, "
                "max_scaled_noncontrol_row_total="
                f"{emission['max_scaled_noncontrol_row_total']}, "
                "max_row_total_full_prediction="
                f"{emission['max_row_total_full_prediction']}, "
                f"configured max_counts_per_cell={effective.max_counts_per_cell}."
            ) from e
    else:
        results = compute_metrics(pred, real_ad, config=effective, de_real=de_real)
    names = metric_output_names(effective)
    agg = aggregate_metrics_wide(results, metrics=names)

    _require_scoreable(agg)
    degenerate = _degenerate_metrics(agg)
    if degenerate:
        if not allow_degenerate:
            raise ValueError(_degenerate_message(degenerate, allow_degenerate=False))
        logger.warning("baseline: %s",
                       _degenerate_message(degenerate, allow_degenerate=True))

    # AFTER the gate: nothing the caller asked for is written for a rejected baseline.
    # (compute_metrics writes its own run_params.yaml when outdir is set; that is a record
    # of the attempt, not an artifact anyone scores.)
    if save_pred is not None:
        pred.write_h5ad(save_pred)

    resolved = resolve_metrics(effective.metrics, version=effective.version)[0]
    effective_types = {
        "real": _effective_input_type(real_ad, effective, side="real"),
        "pred": _effective_input_type(pred, effective, side="pred"),
    }
    comparator = norm.resolve_comparator(
        version=effective.version,
        pred_input_type=effective_types["pred"],
        real_input_type=effective_types["real"],
    )
    meta = {
        "cell_eval2_version": _version_lazy(),
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source": str(real) if isinstance(real, (str, os.PathLike)) else "<in-memory AnnData>",
        "source_fingerprint": source_fp,
        "source_fingerprint_strict": effective.cache_strict,
        "pert_col": effective.pert_col,
        "control": effective.control,
        "n_perturbations": profile.n_perturbations,
        "n_genes": int(profile.genes.size),
        "exclude_target_gene": profile.exclude_target_gene,
        "n_excluded": profile.n_excluded,
        "emit": emit,
        "seed": int(seed),
        "baseline_emission": pred.uns["baseline_emission"],
        "control_source_requested": config.control_source,
        "control_source_effective": effective.control_source,
        # The digest keys on the REQUESTED config, so these record what actually ran and
        # are what `score` compares against the user run's run_meta.json.
        "input_type_real_effective": effective_types["real"],
        "input_type_pred_effective": effective_types["pred"],
        "comparator": comparator,
        # See build_run_meta: the token does not identify the per-metric space. #264 PR2.
        # ⚠️ Keyed by CANONICAL name: `names` here is metric_output_names(), which is v1
        # aliases under v1 -- `CATALOG["mae"]` is a KeyError. Resolve, as every other
        # catalog lookup outside the output frame does.
        "metric_normalization": [
            [c, effective_normalization(CATALOG[c], comparator)]
            for c in (_NAME_TO_CANONICAL.get(n, n) for n in names)
            if c in CATALOG
        ],
        "allow_discrete_effective": effective.allow_discrete,
        # RESOLVED, not requested: de.backend="auto" picks a different engine per host
        # ("DE numbers differ between engines"), and device selects fp32-GPU vs fp64-CPU
        # means. Two runs with an identical EvalConfig are not necessarily comparable.
        "resolved_device": _cache_device(effective),
        "resolved_de_backend": (_cache_backend(effective)
                                if _de_backend_used(effective, resolved, de_real)
                                else None),
        "de_real_supplied": de_real is not None,
        # A SUPPLIED table is value-affecting and invisible to the config -- the de_real
        # test proves two tables change the aggregate -- so its identity is a fingerprint,
        # not a boolean. None when the table was computed from the reference.
        "de_real_fingerprint": (fingerprint_de_table(de_real, strict=True)
                                if de_real is not None else None),
        # recorded, never compared: always None here, a hash for `run --de-pred`
        "de_pred_fingerprint": None,   # the prediction is synthetic; --de-pred is rejected
        "degenerate_metrics": degenerate,
        "metrics": names,
        "config_requested": config.to_dict(),
        # post-lock/post-forcing, but PRE target_sum resolution -- run.py:765-773 resolves
        # target_sum=None inside compute_metrics and run.py:905 writes that fully-resolved
        # config to run_params.yaml, which is therefore the authority. Do not overwrite it.
        "config": effective.to_dict(),
        "config_digest": config_digest(
            config, comparator=comparator, de_real=de_real,
        ),
    }
    return BaselineResult(results=results, agg=agg, profile=profile, meta=meta)
