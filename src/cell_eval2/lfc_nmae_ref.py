"""Split-half replicate reference for ``de_lfc_nmae`` (issue #208).

``de_lfc_nmae`` is a valid ranking instrument on its own, but its LEVEL is a property of
the evaluation data rather than of the model: the same submission reads better on deeper
data. The reference fixes both ends of the scale so scores are comparable across datasets
and averageable across cell lines -- 0 is ATTAINED BY an all-zero predicted-LFC table (it is
not the only way to reach it: nmae is |c-1| for a uniform c x real prediction, so c = 2 lands
there too), 1 is as good as re-running the experiment. Both ends are properties of the
TARGET, shared by every submission.
⚠️ The 0 end is exact in log2FC SPACE. Under ``control_source="real"`` a submission that emits
the control need not produce an all-zero LFC table, so it need not land exactly on 0
(#286, ``docs/metrics.md`` 4.3).

    nmae_ref_raw(p) = mean_g |lfc_A(g) - lfc_B(g)| / mean_g |lfc_real(g)|
    nmae_ref_sqrt2(p) = nmae_ref_raw(p) / sqrt(2)

Both are emitted, but as of #276 part B the SCORED quantity is the RAW one:
`score`'s ``from_reference`` divides by ``nmae_ref_raw``, and no depth correction is
applied anywhere in that scheme. ``nmae_ref_sqrt2`` stays emitted so the previous
number remains auditable.

The sqrt(2) maps the half-depth agreement onto the full-depth replicate
error it stands in for: with equal independent halves ``Var(e_half) = 2 Var(e_full)``, the
quantity wanted is ``Var(Y - X) = 2 Var(e_full)`` and the measurable one is
``Var(A - B) = 4 Var(e_full)``, so the ratio of standard deviations is sqrt(2) and it
carries to ``E|.|`` under any common scale family. Those assumptions are approximate for
heteroskedastic or heavily zero-inflated genes, which is exactly why the RAW value is
reported alongside -- the correction stays inspectable rather than folded away.

**Not** an extension of :mod:`cell_eval2.ceiling`, for two independent reasons.

1. ``ceiling.py`` corrects half depth to full depth with Spearman-Brown ``2r/(1+r)``, a
   RELIABILITY correction for bounded correlation metrics; ``SB_METRICS`` deliberately
   excludes error metrics. An error metric needs the sqrt(2) above, not that.
2. ``compute_ceiling`` scores ``half_b`` against ``half_a`` and never sees the full-depth
   table. This reference gates and normalizes on ALL the real cells while the numerator's
   two vectors come from the halves, so it needs THREE DE tables. That is not a shape
   ``compute_ceiling`` can express.

Control independence is STRUCTURAL here, not an override. ``compute_ceiling`` must force
``control_source="pred"`` because it scores one half against the other, and under
``"real"`` both halves' log2FCs would be computed against one shared control -- which it
measured inflating ``lfc_spearman`` 0.54 -> 0.74. This module never scores one half
against the other: each of the three DE tables is computed from its own AnnData against
the string control label, so each half necessarily uses its own control cells.
``_disjoint_halves`` splits the control group along with every perturbation, which is what
makes that true, and :func:`compute_lfc_nmae_reference` asserts it rather than assuming it.
"""
from __future__ import annotations

import logging
import math
import os

import anndata as ad
import numpy as np
import polars as pl

from .baseline import _materialize_reference
from .ceiling import _disjoint_halves
from .config import EvalConfig
from .de import (
    TargetResolution,
    exclude_on_target,
    prep_de_side,
    resolve_target_genes,
)
from .run import _compute_de_side, _resolve_config, _resolve_target_sum_from_control

logger = logging.getLogger(__name__)

_SQRT2 = math.sqrt(2.0)

_REF_SCHEMA = {
    "perturbation": pl.Utf8,
    "nmae_ref_raw": pl.Float64,
    "nmae_ref_sqrt2": pl.Float64,
    "n_gate": pl.Int64,
}

_AGG_SCHEMA = {
    "statistic": pl.Utf8,
    "nmae_ref_raw": pl.Float64,
    "nmae_ref_sqrt2": pl.Float64,
    "n_perturbations": pl.Int64,
}


def _empty_agg() -> pl.DataFrame:
    """The aggregate for "nothing was scoreable" -- a null ``nmae_ref_raw``, NOT a NaN mean.

    ``score.py`` reads this and leaves ``from_reference`` null with a warning; it does not
    raise. A missing scaled number must never take the rest of the run down with it, and a
    NaN would propagate into the division instead of being detectable (spec 4.4).
    """
    return pl.DataFrame(
        {"statistic": ["mean"], "nmae_ref_raw": [None], "nmae_ref_sqrt2": [None],
         "n_perturbations": [0]},
        schema=_AGG_SCHEMA,
    )


def _nmae_ref_from_tables(
    de_full: pl.DataFrame,
    de_a: pl.DataFrame,
    de_b: pl.DataFrame,
    *,
    p_adj_threshold: float,
    min_gate_size: int,
    target_resolution=None,
) -> pl.DataFrame:
    """The pure arithmetic, over three canonical-schema DE tables. No cells involved.

    The gate and the denominator come from ``de_full``; only the numerator's two vectors
    come from the halves. A gene the half's DE did not report is a 0 LFC on that side --
    the same convention ``de_lfc_nmae`` uses for a null predicted LFC.

    ⚠️ ``target_resolution`` carries issue #172's on-target exclusion, and it is not optional in
    spirit. This function builds ``de_lfc_nmae``'s gate a SECOND time rather than calling the
    metric, so the member's exclusion does not reach it for free -- and ``anchor.py`` then
    SUBSTITUTES this reference for the member's own split value, making it the 1.0 end of that
    member's scaled score. Two gates over different gene sets would answer different questions:
    not just a different raw level, but a different COHORT, since the gate shrinks by one per
    resolved target while ``min_gate_size`` is judged on the result. A target with nine
    off-target significant genes plus its own row is scored by an unexcluded reference and
    omitted by the member.

    It must be the DATASET-GLOBAL resolution, resolved from the UNSLICED full-real table, for
    the same reason ``de.resolve_target_genes`` insists on that. ``None`` means "exclude
    nothing" and is for a direct caller testing the arithmetic alone; every production caller
    passes one, and ``compute_lfc_nmae_reference`` builds it.
    """
    if min_gate_size < 1:
        raise ValueError(f"min_gate_size must be >= 1, got {min_gate_size!r}")
    # ALL THREE tables, before any filter or join. A duplicate in a HALF multiplies the join
    # and silently changes n_gate, the numerator and the denominator -- the same failure the
    # member guards against with its join-height check, which this function does not have.
    for _label, _df in (("full real", de_full), ("half A", de_a), ("half B", de_b)):
        _dup = _df.group_by("target", "feature").len().filter(pl.col("len") > 1)
        if _dup.height:
            raise ValueError(
                f"the {_label} DE table has {_dup.height} duplicate (target, feature) "
                f"row(s), e.g. {_dup.row(0)[:2]}; de_lfc_nmae's reference cannot aggregate "
                "them silently."
            )
    # Issue #172, and it must happen HERE -- inside the gate construction, before the halves are
    # joined and before `min_gate_size` is applied -- so the reference's gate is the member's
    # gate gene-for-gene. Applying it later would leave `n_gate` and the cohort disagreeing.
    gate = exclude_on_target(
        de_full.filter(pl.col("p_adj") < p_adj_threshold)
        .filter(pl.col("log2_fold_change").is_finite())
        .select("target", "feature", "log2_fold_change"),
        target_resolution if target_resolution is not None else TargetResolution({}, 0),
    )
    if gate.height == 0:
        # Same count/reason/example shape as the per-reason warnings below, so a reader sees
        # one format whether the gate was partially or entirely empty.
        _all = sorted(de_full["target"].unique().to_list())
        logger.warning(
            "lfc_nmae reference: omitted %d target(s) for an empty gate at p_adj < %s "
            "(e.g. %s) -- every target, so the reference is empty",
            len(_all), p_adj_threshold, _all[0] if _all else "<none>",
        )
        return pl.DataFrame(schema=_REF_SCHEMA)
    # Every target in the gate MUST be in both halves. The <= 1 cell precondition upstream
    # guarantees every target's CELLS survive the split -- it does NOT guarantee a DE ROW,
    # because the v2 default filter_gene_min_cpm_cell=5.0 can remove every gene for a
    # surviving target. So this is a genuine second enforcement point, not a restatement of
    # the first, and it raises rather than degrading to a warning: a target the reference
    # cannot measure while the member can is exactly the mismatch both checks exist to
    # prevent. A target missing here would be left-joined to 0.0 on BOTH sides, giving
    # |A - B| = 0 and nmae_ref_raw = 0: a PERFECT replicate for a target never measured, and
    # then a denominator of 1 in the scaled score. The member would still have averaged it
    # in, so the two means would be over different panels and `score` sees only the scalars.
    absent = sorted(set(gate["target"].unique().to_list())
                    - (set(de_a["target"].unique().to_list())
                       & set(de_b["target"].unique().to_list())))
    if absent:
        raise ValueError(
            f"{len(absent)} gated target(s) are missing from a split half (e.g. "
            f"{absent[0]!r}), so the reference cannot measure them while the member can -- "
            "their aggregates would be means over different target sets. TWO causes, both "
            "real: (1) the group had <= 1 cell, which compute_lfc_nmae_reference refuses up "
            "front, so this is not it unless that check was bypassed; (2) the DE backend "
            "emitted no row for the target on a half -- the v2 default "
            "filter.filter_gene_min_cpm_cell=5.0 inner-joins the DE frame and can remove "
            "every gene for a target whose CELLS survived the split. For (2), lower or "
            "disable the CPM gate, or drop the target from the real data."
        )
    merged = (
        gate.join(de_a.select("target", "feature", "log2_fold_change"),
                  on=["target", "feature"], suffix="_a", how="left")
        .join(de_b.select("target", "feature", "log2_fold_change"),
              on=["target", "feature"], suffix="_b", how="left")
    )
    # Each join applies its OWN explicit suffix, so both columns exist under the names
    # given above -- verified against this repo's polars, not assumed. Do NOT add a
    # fallback to "log2_fold_change_right": a fallback that fires for BOTH sides would
    # silently alias _a to _b, making every |A - B| zero and every reference 0.0, which
    # looks like a perfect replicate rather than a bug.
    # Half-side null AND non-finite both become 0.0 -- "this half's DE said nothing about
    # this gene". NEVER row-filter on a half's validity: that would shrink n_gate and change
    # the denominator, so the reference would be measured over a different gene set than the
    # member and the ratio in spec 4.1 would not compare like with like. `prep_de_side` does
    # not strip +/-inf (it only forces p_adj=1 on NaN rows), so this path is reachable.
    # The real-side filter stayed on the GATE above, where it belongs.
    merged = merged.with_columns(
        pl.when(pl.col("log2_fold_change_a").is_finite())
        .then(pl.col("log2_fold_change_a")).otherwise(0.0).alias("_a"),
        pl.when(pl.col("log2_fold_change_b").is_finite())
        .then(pl.col("log2_fold_change_b")).otherwise(0.0).alias("_b"),
    )
    agg = merged.group_by("target").agg(
        num=(pl.col("_a") - pl.col("_b")).abs().mean(),
        den=pl.col("log2_fold_change").abs().mean(),
        n_gate=pl.len(),
    )
    rows, small, degenerate = [], [], []
    for target, num, den, n_gate in agg.iter_rows():
        if n_gate < min_gate_size:
            small.append(target)
            continue
        if den is None or not math.isfinite(den) or den == 0.0:
            degenerate.append(target)
            continue
        raw = float(num) / float(den)
        rows.append({"perturbation": target, "nmae_ref_raw": raw,
                     "nmae_ref_sqrt2": raw / _SQRT2, "n_gate": int(n_gate)})
    # Counted from the COMPLETE real target set, not from `agg`: a target whose gate is
    # empty produces no group row at all, so counting from the aggregate would silently miss
    # exactly the case most worth reporting. Warned per reason, and warned even when other
    # targets survived -- a partial omission is the one a reader will not notice.
    empty = sorted(set(de_full["target"].unique().to_list())
                   - set(agg["target"].to_list()))
    for reason, names in (("an empty gate", empty),
                          (f"a gate below min_gate_size={min_gate_size}", small),
                          ("a zero or non-finite denominator", degenerate)):
        if names:
            logger.warning(
                "lfc_nmae reference: omitted %d target(s) for %s (e.g. %s)",
                len(names), reason, sorted(names)[0],
            )
    if not rows:
        return pl.DataFrame(schema=_REF_SCHEMA)
    return pl.DataFrame(rows, schema=_REF_SCHEMA).sort("perturbation")


def _assert_disjoint_controls(half_a: ad.AnnData, half_b: ad.AnnData, *,
                              pert_col: str, control: str) -> None:
    """The property the whole reference rests on. Asserted, not assumed: with a SHARED
    control the two halves' log2FCs share their control's sampling noise, which correlates
    the two quantities whose agreement is being measured -- ``ceiling.py`` measured that
    turning a 0.54 split-half reliability into 0.74 and manufacturing a 0.94 out of noise.
    """
    a = set(half_a.obs_names[half_a.obs[pert_col].astype(str) == control])
    b = set(half_b.obs_names[half_b.obs[pert_col].astype(str) == control])
    if not a or not b:
        raise ValueError(
            f"control {control!r} is missing from one half ({len(a)} / {len(b)} cells); "
            "cannot build a split-half reference"
        )
    shared = a & b
    if shared:
        raise ValueError(
            f"the two halves share {len(shared)} control cell(s); the reference requires "
            "independent controls per half or it measures shared noise as agreement"
        )


def _warn_if_degenerate(mean_raw: float) -> None:
    """Warn when the reference leaves no usable headroom, thresholding on the RAW mean.

    On the RAW mean because that is the quantity `score._from_reference_column` divides by
    (#276 part B): `from_reference = (1 - nmae) / (1 - nmae_ref_raw)`. The old gate
    thresholded the sqrt(2)-corrected mean, and `mean_raw = mean_sqrt2 * sqrt(2)`, so it
    fired LATE -- at `mean_raw = 1.2` the division is already inverted while `mean_sqrt2` is
    only 0.849 and the old rule said nothing at all. Renaming the column without moving the
    threshold would leave the producer warning about a quantity nobody scores.

    Extracted rather than left inline so the boundary is testable without running three DE
    passes to reach it.
    """
    if mean_raw >= 1.0:
        logger.warning(
            "lfc_nmae reference: mean nmae_ref_raw = %.4f >= 1, so 1 - nmae_ref_raw is not "
            "positive and the data carries no usable signal for this member here. `score` "
            "will report the UNRESCALED value rather than divide by a non-positive number.",
            mean_raw,
        )


def compute_lfc_nmae_reference(
    real: ad.AnnData | str | os.PathLike,
    *,
    config: EvalConfig | None = None,
    seed: int = 0,
    de_real: pl.DataFrame | str | os.PathLike | None = None,
    min_gate_size: int = 10,
    **overrides,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Estimate ``nmae_ref_raw`` from the real data alone.

    Returns ``(results, agg)``:

      * ``results`` -- per-perturbation ``(perturbation, nmae_ref_raw, nmae_ref_sqrt2, n_gate)``
        over the perturbations that cleared the gate.
      * ``agg`` -- ``(statistic, nmae_ref_raw, nmae_ref_sqrt2, n_perturbations)``, one ``mean``
        row. **Aggregate first, divide once**: the scaled score in :mod:`cell_eval2.score`
        divides by THIS mean, never by a per-perturbation ratio. Upstream measured the
        per-perturbation form reaching 448 with a cross-line spread of 2.25, against 0.013
        for the ratio of means.

    ``de_real`` lets a caller pass a full-real DE table it already computed, so a run that
    has one pays only the two half-data DE passes; without it the cost is THREE. Neither
    frame is persisted -- callers write them.

    RAISES if any perturbation in ``real`` has <= 1 cell (spec 3.2.1).
    """
    if min_gate_size < 1:
        raise ValueError(f"min_gate_size must be >= 1, got {min_gate_size!r}")
    cfg = _resolve_config(config, overrides)

    # Defined UP FRONT: the supplied-`de_real` validation below needs `_prep` before the
    # split happens, and a `def` after first use would be an UnboundLocalError.
    def _prep(src):
        # Same nan-policy and effect-size floor the METRIC's PreparedDE applies, or the
        # reference and the member would be gated on different gene sets and their ratio
        # would not be a ratio of comparable quantities. Also accepts a PATH, which is what
        # `--de-real` hands us.
        #
        # A header-only CSV is MALFORMED, not empty. `load_de_table` overrides only the
        # `target`/`feature` dtypes, so it loads `log2_fold_change` as a string and
        # `normalize_de_schema`'s `.abs()` raises. Let it. A correctly typed zero-row frame
        # can normalize, and the absent-target warning below makes its empty coverage loud.
        out, _perts = prep_de_side(src, name="real", sort_by=cfg.de.sort_by,
                                   nan_lfc_policy=cfg.de.nan_lfc_policy,
                                   min_abs_log2fc=cfg.de.min_abs_log2fc)
        return out

    def _de(adata):
        return _compute_de_side(adata, cfg=cfg, fp=None, store=None, side="real")

    # `load_anndata` passes an ALREADY-LOADED AnnData through unchanged, so a caller that
    # hands us a backed object still has one here and `_disjoint_halves`' `.copy()` would
    # operate on a backed view. Materialize via baseline's helper, NOT `to_memory()`: on the
    # anndata version pinned for py3.11 in CI, `to_memory()` CLOSES the source's backing
    # file, which is a surprising side effect on an object the caller still owns.
    # `_materialize_reference` re-reads the same path instead and documents the measurement.
    real_ad = _materialize_reference(real)
    # On its own terms, not as an incidental pandas KeyError three lines down -- the message
    # a user needs here is "which column did you mean", and `groupby` cannot give it.
    if cfg.pert_col not in real_ad.obs.columns:
        raise ValueError(
            f"pert_col {cfg.pert_col!r} is not a column of the real data's obs; present "
            f"columns: {list(real_ad.obs.columns)}"
        )
    # THE PRECONDITION (spec 3.2.1), checked before any DE is computed, so a bad input never
    # pays the THREE DE passes. It does not make a bad input free: `_materialize_reference`
    # above has already loaded the matrix, which is the dominant cost on a large h5ad.
    # Moving the obs preflight behind a backed handle is filed as #216.
    # _disjoint_halves drops a group iff n // 2 < 1, i.e. at exactly n == 1, and a
    # target dropped there would be absent from both half DE tables while still sitting in
    # the full-real gate -- so the reference would omit it while the member had already
    # averaged it in, and `score` (which sees only the two scalars) could neither detect nor
    # correct the mismatch. Refusing the input makes the two target sets identical BY
    # CONSTRUCTION. Alex, 2026-08-02.
    # Count with THE SPLITTER'S OWN grouping, not `astype(str).value_counts()`. The two
    # disagree: `_disjoint_halves` groups the raw pandas values, so labels `1` and `"1"` are
    # two one-cell groups it would DROP, while a stringified count sees one group of two and
    # waves them through. Calling the same `groupby(..., observed=True).indices` makes the
    # precheck and the split the same partition, mixed dtypes included -- the same "one
    # definition, not two" reason `catalog.is_decisive` exists. NULLS are NOT included by
    # either (both default to dropna=True), which is why they are rejected separately above.
    # NULLS FIRST: `groupby(...).indices` defaults to dropna=True, so a null-labelled cell
    # is invisible to BOTH this precheck and `_disjoint_halves` -- while the DE path
    # stringifies labels (prep.py, deseq2_de.py), turning them into a real target named
    # "nan". That target would then sit in the full-real gate with no group in either half:
    # exactly the mismatch the precondition exists to prevent, entering through a door the
    # precondition cannot see.
    if real_ad.obs[cfg.pert_col].isna().any():
        n_null = int(real_ad.obs[cfg.pert_col].isna().sum())
        raise ValueError(
            f"{n_null} cell(s) have a null {cfg.pert_col!r}; the lfc_nmae reference cannot "
            "split a group it cannot see, and the DE backends stringify the label into a "
            "target regardless. Drop or relabel those cells first."
        )
    groups = real_ad.obs.groupby(cfg.pert_col, observed=True).indices
    thin = sorted(str(k) for k, idx in groups.items() if len(idx) <= 1)
    if thin:
        raise ValueError(
            f"{len(thin)} perturbation(s) in the real data have <= 1 cell "
            f"(e.g. {thin[0]!r}); the lfc_nmae reference needs at least 2 per group so "
            "every target survives into both halves of the split. Filter them out of the "
            "real data first -- omitting them here would silently make the member's and "
            "the reference's aggregates means over different target sets."
        )
    labels = {str(k) for k in groups}
    if cfg.control not in labels:
        raise ValueError(
            f"control {cfg.control!r} is absent from {cfg.pert_col!r}; present labels "
            f"include {sorted(labels)[:5]}"
        )
    # A SUPPLIED de_real is the other door into a target-set mismatch: it can name a target
    # with no cells in `real` at all, which the member would score (its gate comes from this
    # very table) while the reference cannot. Normalized through `prep_de_side` FIRST -- the
    # argument is a DE SOURCE (path or frame), the same surface every other entry point
    # takes, so indexing it as a frame would crash on the documented `--de-real <path>` fast
    # path; and its `target` column may be numeric, which must be compared stringified
    # against the obs labels rather than raw. Validated BEFORE the all-control return below,
    # or a stray table would slip through whenever the input happened to be all control.
    if de_real is not None:
        de_real = _prep(de_real)          # ONE argument -- `_prep` hard-codes name="real"
        stray = sorted(set(str(t) for t in de_real["target"].unique().to_list())
                       - (labels - {cfg.control}))
        if stray:
            raise ValueError(
                f"the supplied de_real names {len(stray)} target(s) with no cells in the "
                f"real data (e.g. {stray[0]!r}); it must be the DE table OF this dataset."
            )
        absent_from_de = sorted((labels - {cfg.control})
                                - set(str(t) for t in de_real["target"].unique().to_list()))
        if absent_from_de:
            logger.warning(
                "the supplied de_real has no row for %d of the %d non-control target(s) in "
                "the real data (e.g. %s); the reference is measured over the %d it does name. "
                "A stale or partial table is the usual cause -- the reference cannot gate a "
                "target its full-real table never mentions.",
                len(absent_from_de), len(labels) - 1, absent_from_de[0],
                len(labels) - 1 - len(absent_from_de),
            )
    # Classified from the FULL input, before the split -- a post-split check cannot tell
    # "there were never any targets" from "the split lost them", and the second is a bug.
    if labels == {cfg.control}:
        logger.warning(
            "lfc_nmae reference: the real data has no non-control group; returning an "
            "empty reference rather than a NaN mean",
        )
        return pl.DataFrame(schema=_REF_SCHEMA), _empty_agg()
    # All three DE tables share ONE normalization target, resolved from the real control
    # pool exactly as `_run_metrics` does (#155). Without this each call resolves its own:
    # `target_sum=None` means "the median of whichever matrix normalize_total was handed",
    # and on real data the full table and the two halves picked 54264 / 54250 / 54290
    # against the control pool's 54758.
    #
    # MEASURED, because the obvious severity model is wrong: that divergence moves
    # `log2_fold_change` by 6.6e-11, NOT by the log2(T_x/T_ref) ~ 0.013 a naive reading
    # predicts. Within one DE call the perturbed group and the control group are normalized
    # with the SAME target, so it cancels in the fold change; only log1p's nonlinearity
    # survives. Re-running the real-data reference before and after this line changed
    # `nmae_ref_raw` by 0.0 to 15 significant figures.
    #
    # It is kept anyway, and not as tidiness. The cancellation is a property of every DE
    # call here comparing two groups drawn from ONE matrix -- which is exactly what #155
    # broke when a pred side was compared against the real control pool under
    # `control_source="real"`. One resolved config cannot be relied on to stay harmless;
    # three unresolved ones are a latent version of the bug #155 fixed.
    #
    # Placed AFTER the all-control return, so a degenerate input does not pay for it, and
    # before the first `_de(...)`, which is the only consumer.
    cfg = _resolve_target_sum_from_control(cfg, real_ad)
    half_a, half_b = _disjoint_halves(real_ad, cfg.pert_col, cfg.control, seed)
    _assert_disjoint_controls(half_a, half_b, pert_col=cfg.pert_col, control=cfg.control)

    # `de_real` was already normalized during validation above, so `_prep` runs once on it,
    # not twice -- `prep_de_side` is not idempotent-free (it re-applies the LFC floor).
    prepped_full = de_real if de_real is not None else _prep(_de(real_ad))
    # Issue #172. Resolved from the UNSLICED full-real table and its FULL target list -- never
    # from a half, whose feature index is a strict subset and would make the resolution depend
    # on the split (`de.resolve_target_genes`).
    #
    # ⚠️ NO zero-resolve raise here, deliberately, and it is not a hole. The gate belongs to the
    # METRIC (`metrics.de._exclude_own_gene`), and a panel where nothing resolves cannot produce
    # a mismatched pair: the reference would exclude nothing, but the member raises before it
    # can be scaled by it, so the divergence is unreachable rather than silent. Under PARTIAL
    # resolution both sides consume the same `resolve_target_genes` result through the same map,
    # so they exclude the same (target, feature) pairs by construction. Adding a second raise
    # would only hard-fail a standalone `--lfc-nmae-ref` build on a panel whose member could
    # never be scored anyway.
    resolution = resolve_target_genes(
        prepped_full,
        sorted(prepped_full["target"].unique().to_list()),
        target_gene_map=cfg.target_gene_map,
    )
    results = _nmae_ref_from_tables(
        prepped_full, _prep(_de(half_a)), _prep(_de(half_b)),
        p_adj_threshold=cfg.de.p_adj_threshold, min_gate_size=min_gate_size,
        target_resolution=resolution,
    )
    if results.height == 0:
        logger.warning(
            "lfc_nmae reference: no perturbation cleared the gate (min_gate_size=%d); "
            "returning an empty result rather than a NaN mean", min_gate_size,
        )
        return results, _empty_agg()
    mean_raw = float(np.mean(results["nmae_ref_raw"].to_numpy()))
    mean_sqrt2 = float(np.mean(results["nmae_ref_sqrt2"].to_numpy()))
    _warn_if_degenerate(mean_raw)
    agg = pl.DataFrame(
        {"statistic": ["mean"], "nmae_ref_raw": [mean_raw], "nmae_ref_sqrt2": [mean_sqrt2],
         "n_perturbations": [results.height]},
        schema=_AGG_SCHEMA,
    )
    return results, agg
