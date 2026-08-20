"""Opt-in ``deseq2`` DE backend: a negative-binomial GLM via the deseq2_gpu engine.

Pseudobulks raw counts by ``(pert_col, replicate_col)`` (control = replicated NTC guides,
each perturbation typically n=1), builds one contrast per perturbation (control as the
reference level), fits via deseq2_gpu (looped ``fit_nb_glm``+``results`` on CPU, batched
``fit_contrasts`` on GPU), and converts to cell_eval2's canonical DE schema. deseq2 owns its
own LFC + p-values (like gpudge), so ``mean_calc``/``epsilon``/``clip_value`` do not apply and
``fdr_scope`` is ignored (native per-contrast padj: Cook's flagging + independent filtering).
See ``internal:docs/superpowers/specs/2026-07-19-deseq2-backend-design.md``.
"""
from __future__ import annotations

import logging

import numpy as np
import polars as pl

from .de import normalize_de_schema
from .prep import _group_row_index, _grouped_sums

logger = logging.getLogger(__name__)

_SEP = "\x1f"  # unit separator: safe composite key joiner (won't occur in labels)


def _validate(adata, *, pert_col, control, replicate_col, input_type) -> None:
    """Fail loud on any layout the deseq2 backend cannot handle (raw-count/replicate model)."""
    if input_type != "counts":
        raise ValueError(
            f"the deseq2 backend requires input_type='counts' (raw counts); got {input_type!r}"
        )
    if replicate_col is None:
        raise ValueError("the deseq2 backend requires de.replicate_col to be set")
    for name, col in (("pert_col", pert_col), ("replicate_col", replicate_col)):
        if col not in adata.obs.columns:
            raise ValueError(
                f"obs is missing the {name} column {col!r} (needed by the deseq2 backend)"
            )
    labels = adata.obs[pert_col].astype(str).to_numpy()
    if control not in set(labels):
        raise ValueError(f"control {control!r} not found in obs[{pert_col!r}]")
    ctrl_reps = adata.obs[replicate_col].astype(str).to_numpy()[labels == control]
    n_ctrl_rep = len(set(ctrl_reps.tolist()))
    if n_ctrl_rep < 2:
        raise ValueError(
            f"the deseq2 backend needs >= 2 control replicate levels in obs[{replicate_col!r}] "
            f"(e.g. multiple NTC guides); found {n_ctrl_rep}"
        )


def _pseudobulk(adata, *, pert_col, control, replicate_col):
    """Sum raw counts per ``(pert_col, replicate_col)``. Returns
    (counts_gxs[float64 genes x samples], conditions[list], replicates[list], genes[np.ndarray]).

    ``control`` is accepted for call-site symmetry with the rest of the adapter; the pseudobulk
    grouping itself is control-agnostic (every distinct (condition, replicate) is one sample).

    ⚠️ **This path moves with #271's shared reduce-wide policy, deliberately (Alex 2026-08-18).**
    #264 PR2 left ``prep._grouped_sums`` reducing in the input dtype partly because widening it
    moves deseq2's numbers too, and "whether deseq2 moves must be deliberate". It is: the
    alternative was a ``widen=`` keyword on a helper shared by two families, and a shared helper's
    per-caller precondition is the thing that gets forgotten. Three reasons this side does not opt
    out:

    * for every dtype the widen guard actually touches -- a float COARSER than float64 -- the wide
      reduction is the more accurate sum of exactly the values stored, so this path can only move
      TOWARD its own exact pseudobulk. The dtypes where fp64 would be the narrower side (integer
      above ``2**53``, longdouble) are excluded by that guard, here as everywhere else;
    * the groups here are per ``(condition, replicate)``, which OFTEN makes them smaller than the
      per-perturbation groups the comparator sums -- but that is a tendency, not a bound: only the
      CONTROL is required to have two replicate levels, so a treatment with one has a group identical
      to its perturbation group, and even a small group can cross ``2**24`` at enough depth. So this
      is EXPECTED to be a no-op on realistic counts, and ``_validate`` already requires
      ``input_type='counts'`` rather than a normalized space -- neither of which is an exactness
      guarantee;
    * this backend is opt-in, non-default and behind a private dep, and **no ENROLLED competition
      artifact can be built with it** -- ``de_deseq2_*`` members cannot form an official bundle. ⚠️
      Three different things, kept apart: nothing ENROLLED is orphaned; a CACHED table IS invalidated,
      deliberately, since ``run``'s deseq2 DE-table key carries
      ``_GROUPED_SUM_REDUCTION_SEMANTICS`` so a warm pre-#271 table is recomputed rather than served;
      and an EXPORTED table handed back through ``--de-pred``/``--de-real`` is protected by NEITHER,
      because those flags skip DE computation entirely -- which is why the regeneration guidance in
      ``docs/metrics.md`` names those tables explicitly.

    Opting out would mean giving ``_grouped_sums`` a ``widen=`` keyword and passing ``False`` on
    the call below; ``tests/test_deseq2_backend.py`` pins the choice that WAS made, so making that
    change fails a test rather than passing silently.

    ⚠️ It does NOT follow that the deseq2 RESULTS move only microscopically: the sums here feed
    ``poscounts`` size factors and an NB GLM fit, so a 1-count change can in principle flip a
    borderline p-value and therefore a DE gene set. The claim is about the direction and the
    reachability of the input change, not about the smoothness of what consumes it."""
    # Vectorized composite (condition, replicate) key via pandas str concat (Cython) -- no
    # Python per-cell loop, avoiding the O(n_obs) list-comp bottleneck + memory spike at VCC scale.
    composite = (
        adata.obs[pert_col].astype(str) + _SEP + adata.obs[replicate_col].astype(str)
    ).to_numpy(dtype=str)
    uniq, order, bounds = _group_row_index(composite)
    sums = _grouped_sums(adata.X, order, bounds, uniq.size)  # [n_samples, n_genes]
    conditions, replicates = [], []
    for key in uniq:
        c, r = key.split(_SEP, 1)
        conditions.append(c)
        replicates.append(r)
    genes = np.asarray(adata.var_names, dtype=str)
    return sums.T.astype(np.float64), conditions, replicates, genes


def _size_factors(counts_gxs) -> np.ndarray:
    """poscounts size factors over all pseudobulk samples (genes x samples); strictly positive."""
    from deseq2_gpu import estimate_size_factors

    return np.asarray(estimate_size_factors(counts_gxs, method="poscounts"), dtype=np.float64)


def _contrasts(counts_gxs, conditions, *, control, size_factors):
    """One (pert_name, sub_counts, design, sf_sub) per non-control condition.
    design = [intercept=1, condition] with control=0, perturbation=1 (reference = control)."""
    conditions = np.asarray(conditions, dtype=object)
    ctrl_cols = np.flatnonzero(conditions == control)
    out = []
    for g in sorted(set(conditions.tolist()) - {control}):
        g_cols = np.flatnonzero(conditions == g)
        cols = np.concatenate([ctrl_cols, g_cols])
        sub = np.ascontiguousarray(counts_gxs[:, cols])
        cond_ind = np.concatenate([np.zeros(len(ctrl_cols)), np.ones(len(g_cols))])
        design = np.column_stack([np.ones_like(cond_ind), cond_ind]).astype(np.float64)
        out.append((g, sub, design, np.asarray(size_factors)[cols]))
    return out


_CONTRAST_VEC = np.array([0.0, 1.0])  # test the condition coefficient (control=reference)


def _fit_one(sub, design, sf):
    """CPU single-contrast fit -> results DataFrame (rows aligned to sub's gene order)."""
    from deseq2_gpu import fit_nb_glm, results

    fit = fit_nb_glm(sub, design, size_factors=sf, tool="deseq2", backend="numpy")
    return results(fit, _CONTRAST_VEC)


def _fit_all_gpu(contrasts):
    """GPU batched fit -> list of results DataFrames aligned to `contrasts` order."""
    from deseq2_gpu import fit_contrasts, results

    trios = [(sub, design, sf) for (_g, sub, design, sf) in contrasts]
    fits = fit_contrasts(trios, tool="deseq2", backend="jax")
    return [results(f, _CONTRAST_VEC) for f in fits]


def run_deseq2_de(adata, *, pert_col, control, replicate_col, input_type, use_gpu=False) -> pl.DataFrame:
    """Compute a canonical-schema DE table via deseq2_gpu (NB-GLM). deseq2 owns the LFC + p-values.

    results() indexes its output positionally for ndarray input (verified: wald_test synthesizes
    ``gene_i`` names in row order), so ``res`` rows align 1:1 with ``genes`` (var_names order) and a
    positional assignment is exact -- the parity test (test_run_deseq2_de_matches_direct_fit) is the
    gate. If a future engine ever reordered/indexed the frame by name, switch to a gene-name join."""
    _validate(adata, pert_col=pert_col, control=control,
              replicate_col=replicate_col, input_type=input_type)
    counts, conditions, _replicates, genes = _pseudobulk(
        adata, pert_col=pert_col, control=control, replicate_col=replicate_col)
    sf = _size_factors(counts)
    contrasts = _contrasts(counts, conditions, control=control, size_factors=sf)
    res_list = _fit_all_gpu(contrasts) if use_gpu else [_fit_one(sub, d, s) for (_g, sub, d, s) in contrasts]
    frames = []
    for (g, _sub, _d, _s), res in zip(contrasts, res_list):
        frames.append(pl.DataFrame({
            "target": g,
            "feature": genes,                                   # positional: res rows == gene order
            "log2_fold_change": np.asarray(res["log2FoldChange"], dtype=float),
            "p_value": np.asarray(res["pvalue"], dtype=float),
            "p_adj": np.asarray(res["padj"], dtype=float),
        }))
    df = pl.concat(frames, how="vertical") if frames else pl.DataFrame(
        schema={"target": pl.Utf8, "feature": pl.Utf8, "log2_fold_change": pl.Float64,
                "p_value": pl.Float64, "p_adj": pl.Float64})
    return normalize_de_schema(df, name="deseq2")
