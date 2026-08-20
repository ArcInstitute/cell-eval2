from __future__ import annotations

import logging
import math

from typing import Literal

import numpy as np
import polars as pl
from sklearn.metrics import auc, average_precision_score, roc_curve

from ..de import PreparedDE, exclude_on_target, prepare_de, require_resolution

logger = logging.getLogger(__name__)


def _exclude_own_gene(df: pl.DataFrame, prepared: PreparedDE, *,
                      who: str, side: str) -> pl.DataFrame:
    """Drop each resolved target's own gene from ``df``, gated and counted. Issue #172.

    Two scored ``vcc2026`` members call this -- ``de_sig_jaccard`` (both sides) and
    ``de_lfc_nmae`` (the real-side gate). Before #172 they were the DE family's last scored
    members passing the perturbed gene's own row straight through, while the discrimination
    family had excluded it since v1 and the eleven chance-corrected direction metrics since
    #195. Afterwards all six scored members exclude.

    ⚠️ **The gate is not optional and is the point of this wrapper.** Exclusion is worth
    exactly as much as the label->gene resolution: on a construct-ID panel (``'ADNP-1'`` vs
    feature ``'ADNP'``) with no ``target_gene_map`` NOTHING resolves, the anti-join removes
    nothing, and the metric silently keeps its pre-#172 meaning. That is issue #248's failure
    mode -- there it let a trivially-gameable submission win -- so :func:`de.require_resolution`
    raises rather than returning a plausible wrong number. The competition profile reaches the
    same gate through the direction family too; this closes the ``de``-profile run that selects
    only these two.

    ⚠️ The PARTIAL case does not raise (ruled 2026-08-16 for ``pds_*``, and the same reasoning
    holds here: the harm is continuous and no threshold is principled), so the ROWS REMOVED are
    reported instead -- the DE-side analogue of ``baseline_meta.json``'s ``n_excluded``. The
    line carries BOTH counts (rows removed, and targets resolved) precisely because either one
    alone is ambiguous: zero rows removed at full resolution is legitimate, and is what H1_CGS
    does -- ``pooled_universe`` drops each target's own gene from its own universe, so every
    target resolves and none excludes anything (:class:`de.TargetResolution`). Only the pair
    distinguishes that from a partly-unresolved panel.
    """
    require_resolution(prepared.perturbations, prepared.target_resolution)
    out = exclude_on_target(df, prepared.target_resolution)
    res = prepared.target_resolution
    logger.info(
        "%s: excluded the perturbed gene's own row from the %s side -- %d row(s) removed of "
        "%d, over %d/%d targets resolved to a measured gene (issue #172).",
        who, side, df.height - out.height, df.height, res.n_resolved,
        res.n_targets or len(prepared.perturbations),
    )
    return out


def _ensure_prepared(prepared, de_pred, de_real, *, control, sort_by,
                     p_adj_threshold, nan_lfc_policy, min_abs_log2fc: float = 0.0) -> PreparedDE:
    """Return the given PreparedDE, or build one from raw de_pred/de_real frames
    (standalone path, mirroring de_overlap). The dispatch path always passes `prepared`."""
    if prepared is not None:
        return prepared
    if de_pred is None or de_real is None:
        raise ValueError("provide a PreparedDE or both de_pred and de_real")
    return prepare_de(de_pred, de_real, control=control, sort_by=sort_by,
                      p_adj_threshold=p_adj_threshold, nan_lfc_policy=nan_lfc_policy,
                      min_abs_log2fc=min_abs_log2fc)


def _to_float_nan(x) -> float:
    """polars-null correlation/value -> NaN so the dispatch float(value) never sees None."""
    return float("nan") if x is None else float(x)


def _informedness(tp: int, a: int, b: int, g: int) -> float:
    """Recall-side chance correction -- Youden's J = TPR + TNR - 1 = TPR - FPR, in
    [-1, 1] (computed in the reduced TPR - FPR form). Degenerate (a <= 0 or a >= g)
    -> worst value -1.0. The >=/<= bounds (vs ==) keep the primitive safe if dirty
    input ever makes a set size exceed the unique universe (a > g)."""
    if a <= 0 or a >= g:
        return -1.0
    return tp / a - (b - tp) / (g - a)


def _markedness(tp: int, a: int, b: int, g: int) -> float:
    """Precision-side chance correction -- PPV + NPV - 1 = PPV - FOR, in [-1, 1]
    (reduced PPV - FOR form). Degenerate (b <= 0 or b >= g) -> worst value -1.0."""
    if b <= 0 or b >= g:
        return -1.0
    return tp / b - (a - tp) / (g - b)


def _mcc(tp: int, a: int, b: int, g: int) -> float:
    """Matthews correlation (== phi coefficient) over the 2x2 table, in [-1, 1].
    == sign*sqrt(informedness*markedness); the numerator (TP*TN - FP*FN) reduces
    exactly to (TP*g - a*b). Degenerate (any of a, b, g-a, g-b <= 0) -> -1.0."""
    if a <= 0 or a >= g or b <= 0 or b >= g:
        return -1.0
    return (tp * g - a * b) / ((a * b * (g - a) * (g - b)) ** 0.5)


_MEASURES = {"mcc": _mcc, "markedness": _markedness, "informedness": _informedness}


def _count_map(frame: pl.DataFrame) -> dict[str, int]:
    """{target: count} from a two-column (target, len) group_by('target').len() frame.
    Column-name access (not positional) so it is robust to group_by output ordering."""
    return dict(zip(frame["target"].to_list(), frame["len"].to_list(), strict=True))


def _tested_universe(prepared: PreparedDE) -> dict[str, int]:
    """Per-target gene-universe size G = union of tested genes across real & pred.
    Best-effort memoized on the (transient, in-process) PreparedDE instance: all
    four chance-corrected metrics call this on the same object within one run and
    the union is a heavy concat+unique over the full DE tables. The map cannot go
    stale (real_df/pred_df are immutable for a given instance). The defensive
    try/except on the assignment matches the repo's memoization idiom
    (run.py::_validate_input_once / _check_scale_limit_once) so it degrades to
    recompute if PreparedDE is ever frozen or a read-only view/mock."""
    cached = getattr(prepared, "_tested_universe_cache", None)
    if cached is not None:
        return cached
    tested = (
        pl.concat([
            prepared.real_df.select("target", "feature"),
            prepared.pred_df.select("target", "feature"),
        ])
        .unique()
        .group_by("target").len()
    )
    g_map = _count_map(tested)
    try:
        prepared._tested_universe_cache = g_map
    except (AttributeError, ValueError, TypeError):  # read-only view / locked-down object -> skip memo
        pass
    return g_map


def de_sig_agreement(
    prepared: PreparedDE | None = None,
    *,
    measure: str,
    de_pred=None,
    de_real=None,
    control: str = "non-targeting",
    sort_by: str = "abs_log2_fold_change",
    p_adj_threshold: float = 0.05,
    nan_lfc_policy: str = "mask",
    min_abs_log2fc: float = 0.0,
) -> dict[str, float]:
    """Chance-corrected agreement over the per-pert significance-membership 2x2 table.
    `measure` selects informedness (recall side), markedness (precision side), or mcc
    (balanced). Every perturbation gets a finite value in [-1, 1]; degenerate -> -1."""
    if measure not in _MEASURES:
        raise ValueError(f"measure must be one of {sorted(_MEASURES)}, got {measure!r}")
    prepared = _ensure_prepared(prepared, de_pred, de_real, control=control, sort_by=sort_by,
                                p_adj_threshold=p_adj_threshold, nan_lfc_policy=nan_lfc_policy,
                                min_abs_log2fc=min_abs_log2fc)
    T = prepared.p_adj_threshold
    # Only target/feature are needed for the counts + intersection; selecting them
    # here narrows the join and avoids materializing the wide suffixed columns.
    real_sig = prepared.real_df.filter(pl.col("p_adj") < T).select("target", "feature")
    pred_sig = prepared.pred_df.filter(pl.col("p_adj") < T).select("target", "feature")
    a_map = _count_map(real_sig.group_by("target").len())
    b_map = _count_map(pred_sig.group_by("target").len())
    inter = real_sig.join(pred_sig, on=["target", "feature"], how="inner").group_by("target").len()
    tp_map = _count_map(inter)
    g_map = _tested_universe(prepared)
    fn = _MEASURES[measure]
    return {
        p: fn(tp_map.get(p, 0), a_map.get(p, 0), b_map.get(p, 0), g_map.get(p, 0))
        for p in prepared.perturbations
    }


def de_sig_jaccard(
    prepared: PreparedDE | None = None,
    *,
    de_pred=None,
    de_real=None,
    control: str = "non-targeting",
    sort_by: str = "abs_log2_fold_change",
    p_adj_threshold: float = 0.05,
    nan_lfc_policy: str = "mask",
    min_abs_log2fc: float = 0.0,
) -> dict[str, float]:
    """Per-perturbation Jaccard |R ∩ P| / |R ∪ P| over significance membership
    (p_adj < T). Unlike real-conditioned `de_sig_recall` (denominator |R|) and the
    pred-conditioned side of `de_model_direction_match` (denominator |P|), it is symmetric
    and penalizes both missed real DEGs and spurious predicted ones. Range [0, 1], best 1;
    an empty union returns 1.0. This v2-native metric has no upstream cell-eval counterpart
    and is the chance-uncorrected companion of `de_sig_agreement(measure="mcc")` (catalog:
    `de_wilcoxon_sig_mcc` / `de_deseq2_sig_mcc`), reading the same 2x2 table on a well-formed
    DE table; `de_sig_jaccard` de-duplicates `(target, feature)` where the `de_sig_agreement`
    family counts rows, so they differ only if duplicate rows are present.

    ⚠️ **EXCLUDES the perturbed gene's own row, from BOTH sides (issue #172, ruled
    2026-08-17).** Knocking a gene down and then reporting that gene as differentially
    expressed is the experiment's premise, not a prediction, and this metric was one of the
    last two scored `vcc2026` DE members still counting it. `de_sig_agreement`'s family and
    the other set members deliberately do NOT exclude: they are not scored by `vcc2026`, and
    the ruling is scoped to the competition six.

    Measured (val A, honest half-data arm vs the `context_mean` baseline, `docs/metrics.md`
    §4): the gene appears in 299 of 300 targets' reference rows and is reference-significant
    in all 299, so the raw value falls 0.400139 -> 0.381338 while the baseline barely moves
    (0.033382 -> 0.033286) -- an honest arm loses 0.047 of this member's span. Removing a
    guaranteed hit is exactly why the drop is one-sided.

    ⚠️ Exclusion can turn a NON-EMPTY union into an empty one, which this function scores
    1.0. That is the documented `J(empty, empty) = 1` convention meeting a new population:
    a perturbation whose ONLY reference-significant gene was its own target now has an empty
    reference set, so a submission calling nothing significant for it scores 1.0 where it
    previously scored 0.0. The condition is real-side and rare -- it needs `|R| == 1` before
    exclusion -- but it is a genuine consequence rather than an artifact, and `docs/metrics.md`
    §6.0 is where the six members' no-signal behaviour is compared.
    """
    prepared = _ensure_prepared(prepared, de_pred, de_real, control=control, sort_by=sort_by,
                                p_adj_threshold=p_adj_threshold, nan_lfc_policy=nan_lfc_policy,
                                min_abs_log2fc=min_abs_log2fc)
    T = prepared.p_adj_threshold
    # Jaccard is defined on sets, so unique significant (target, feature) pairs are the
    # definition, not a workaround for dirty input. Uniqueness makes tp <= min(a, b)
    # structural, hence union >= max(a, b) >= 0 and every ratio lies in [0, 1]. It also
    # leaves union == 0 with exactly its documented meaning: a == b == 0, rather than a
    # many-to-many join artifact. de_sig_agreement, de_sig_recall and de_nsig_counts share
    # this un-deduped seam but are deliberately unchanged here: moving four-plus shipped
    # metrics' values belongs in a separate change.
    # The on-target anti-join is applied to BOTH sides (issue #172). Symmetry is required, not
    # cosmetic: dropping the pair from the reference alone would leave a predicted on-target
    # call in |P| as a pure penalty, and dropping it from the prediction alone would leave it
    # in |R| as a guaranteed miss. Applied AFTER `.unique()`, where the frame is already the
    # set the Jaccard is defined on, so the anti-join is over one row per (target, feature).
    real_sig = _exclude_own_gene(
        prepared.real_df.filter(pl.col("p_adj") < T).select("target", "feature").unique(),
        prepared, who="de_sig_jaccard", side="real",
    )
    pred_sig = _exclude_own_gene(
        prepared.pred_df.filter(pl.col("p_adj") < T).select("target", "feature").unique(),
        prepared, who="de_sig_jaccard", side="pred",
    )
    a_map = _count_map(real_sig.group_by("target").len())
    b_map = _count_map(pred_sig.group_by("target").len())
    inter = real_sig.join(pred_sig, on=["target", "feature"], how="inner").group_by("target").len()
    tp_map = _count_map(inter)

    out: dict[str, float] = {}
    for p in prepared.perturbations:
        tp = tp_map.get(p, 0)
        union = a_map.get(p, 0) + b_map.get(p, 0) - tp
        # |R ∪ P| == 0 <=> a == b == 0: neither side called anything significant. Returns
        # 1.0, the set convention J(empty, empty) = 1 -- both sides agree there is no
        # response. Alex's call, 2026-08-02. Every OTHER perturbation has union > 0, so
        # this function returns a finite value in [0, 1] for every perturbation and needs
        # no `worst_value` fill in the catalog (see the four #14 metrics, same pattern).
        out[p] = 1.0 if union == 0 else tp / union
    return out


def de_overlap(
    prepared: PreparedDE | None = None,
    *,
    de_pred=None,
    de_real=None,
    k: int | None,
    metric: str = "overlap",
    sort_by: str = "abs_log2_fold_change",
    p_adj_threshold: float = 0.05,
    nan_lfc_policy: str = "mask",
    min_abs_log2fc: float = 0.0,
    control: str = "non-targeting",
) -> dict[str, float]:
    """Top-k DE-gene overlap (recall-style) or precision, per perturbation.

    Mirrors cell-eval `de_overlap_metric`/`compute_overlap`. Pass a `PreparedDE`
    (the dispatch path) or raw `de_pred`/`de_real` frames (standalone). The prep
    args are used only when building from raw frames.
    """
    if metric not in ("overlap", "precision"):
        raise ValueError(f"metric must be 'overlap' or 'precision', got {metric!r}")
    if k is not None and k < 0:
        raise ValueError(f"k must be None or a non-negative int, got {k!r}")
    prepared = _ensure_prepared(prepared, de_pred, de_real, control=control, sort_by=sort_by,
                                p_adj_threshold=p_adj_threshold,
                                nan_lfc_policy=nan_lfc_policy,
                                min_abs_log2fc=min_abs_log2fc)

    real_rank, pred_rank = prepared.real_rank, prepared.pred_rank
    if real_rank.is_empty() or pred_rank.is_empty():
        return {p: 0.0 for p in prepared.perturbations}

    real_cols, pred_cols = set(real_rank.columns), set(pred_rank.columns)
    out: dict[str, float] = {}
    for pert in prepared.perturbations:
        if pert not in real_cols or pert not in pred_cols:
            out[pert] = 0.0
            continue
        real_genes = real_rank[pert].drop_nulls().to_numpy()
        pred_genes = pred_rank[pert].drop_nulls().to_numpy()
        if metric == "overlap":
            k_eff = min(real_genes.size if k is None else k, real_genes.size)
        else:  # precision
            k_eff = min(pred_genes.size if k is None else k, pred_genes.size)
        if k_eff == 0:
            out[pert] = 0.0
        else:
            inter = np.intersect1d(real_genes[:k_eff], pred_genes[:k_eff]).size
            out[pert] = float(inter / k_eff)
    return out


def de_overlap_adjusted(
    prepared: PreparedDE | None = None,
    *,
    de_pred=None,
    de_real=None,
    control: str = "non-targeting",
    sort_by: str = "abs_log2_fold_change",
    p_adj_threshold: float = 0.05,
    nan_lfc_policy: str = "mask",
    min_abs_log2fc: float = 0.0,
) -> dict[str, float]:
    """Chance-corrected top-`m_r` overlap: MCC over the ranked-membership 2x2 table
    (real's top-`m_r` vs pred's top-`m_r` by `sort_by`). Bounded [-1, 1]; every pert
    gets a value; degenerate -> -1. Mirrors the `de_overlap(k=None, 'overlap')` slice."""
    prepared = _ensure_prepared(prepared, de_pred, de_real, control=control, sort_by=sort_by,
                                p_adj_threshold=p_adj_threshold, nan_lfc_policy=nan_lfc_policy,
                                min_abs_log2fc=min_abs_log2fc)
    real_rank, pred_rank = prepared.real_rank, prepared.pred_rank
    g_map = _tested_universe(prepared)
    real_cols = set() if real_rank.is_empty() else set(real_rank.columns)
    pred_cols = set() if pred_rank.is_empty() else set(pred_rank.columns)
    out: dict[str, float] = {}
    for pert in prepared.perturbations:
        real_genes = real_rank[pert].drop_nulls().to_numpy() if pert in real_cols else np.empty(0, dtype=object)
        pred_genes = pred_rank[pert].drop_nulls().to_numpy() if pert in pred_cols else np.empty(0, dtype=object)
        m_r = real_genes.size
        pred_top = pred_genes[:m_r]
        tp = int(np.intersect1d(real_genes, pred_top).size)
        out[pert] = _mcc(tp, m_r, int(pred_top.size), g_map.get(pert, 0))
    return out


def de_nsig_counts(
    prepared: PreparedDE | None = None,
    *,
    side: str,
    de_pred=None,
    de_real=None,
    control: str = "non-targeting",
    sort_by: str = "abs_log2_fold_change",
    p_adj_threshold: float = 0.05,
    nan_lfc_policy: str = "mask",
    min_abs_log2fc: float = 0.0,
) -> dict[str, float]:
    """Per-perturbation count of significant genes (p_adj < T) on `side` ('real'|'pred').
    Diagnostic (``scored=False``, and genuinely ``direction=None`` -- neither more nor fewer
    significant genes is better): every perturbation present (0 if none)."""
    if side not in ("real", "pred"):
        raise ValueError(f"side must be 'real' or 'pred', got {side!r}")
    prepared = _ensure_prepared(prepared, de_pred, de_real, control=control,
                                sort_by=sort_by, p_adj_threshold=p_adj_threshold,
                                nan_lfc_policy=nan_lfc_policy,
                                min_abs_log2fc=min_abs_log2fc)
    df = prepared.real_df if side == "real" else prepared.pred_df
    counts = df.filter(pl.col("p_adj") < prepared.p_adj_threshold).group_by("target").len()
    count_map = dict(zip(counts["target"].to_list(), counts["len"].to_list(), strict=True))
    return {p: float(count_map.get(p, 0)) for p in prepared.perturbations}


def de_nsig_spearman(
    prepared: PreparedDE | None = None,
    *,
    de_pred=None,
    de_real=None,
    control: str = "non-targeting",
    sort_by: str = "abs_log2_fold_change",
    p_adj_threshold: float = 0.05,
    nan_lfc_policy: str = "mask",
    min_abs_log2fc: float = 0.0,
) -> dict[str, float]:
    """Global Spearman correlation between per-perturbation significant-gene counts
    (real vs pred), broadcast to every perturbation. Matches cell-eval DESpearmanSignificant."""
    prepared = _ensure_prepared(prepared, de_pred, de_real, control=control,
                                sort_by=sort_by, p_adj_threshold=p_adj_threshold,
                                nan_lfc_policy=nan_lfc_policy,
                                min_abs_log2fc=min_abs_log2fc)
    T = prepared.p_adj_threshold
    # Correlate counts over targets with >=1 real-significant gene (LEFT join from
    # filt_real), exactly as upstream DESpearmanSignificant -- NOT over all perturbations.
    # Correlating over all perts changes the value (e.g. -1.0 -> -0.95 in a forced case);
    # test_de_metrics_parity_vs_cell_eval pins the upstream behavior. The single scalar is
    # broadcast to every perturbation below, so the OUTPUT is still rectangular.
    filt_real = prepared.real_df.filter(pl.col("p_adj") < T).group_by("target").len()
    filt_pred = prepared.pred_df.filter(pl.col("p_adj") < T).group_by("target").len()
    merged = filt_real.join(filt_pred, on="target", how="left",
                            suffix="_pred", coalesce=True).fill_null(0)
    if merged.height == 0:
        value = 1.0
    else:
        corr = merged.select(
            pl.corr(pl.col("len"), pl.col("len_pred"), method="spearman")
        ).to_numpy().ravel()[0]
        value = _to_float_nan(corr)
    return {p: value for p in prepared.perturbations}


def de_sig_recall(
    prepared: PreparedDE | None = None,
    *,
    de_pred=None,
    de_real=None,
    control: str = "non-targeting",
    sort_by: str = "abs_log2_fold_change",
    p_adj_threshold: float = 0.05,
    nan_lfc_policy: str = "mask",
    min_abs_log2fc: float = 0.0,
) -> dict[str, float]:
    """Per-perturbation recall of real significant genes also significant in pred
    (|real_sig ∩ pred_sig| / |real_sig|). Perts with no real-sig genes are omitted.
    Matches cell-eval DESigGenesRecall."""
    prepared = _ensure_prepared(prepared, de_pred, de_real, control=control,
                                sort_by=sort_by, p_adj_threshold=p_adj_threshold,
                                nan_lfc_policy=nan_lfc_policy,
                                min_abs_log2fc=min_abs_log2fc)
    T = prepared.p_adj_threshold
    filt_real = prepared.real_df.filter(pl.col("p_adj") < T)
    filt_pred = prepared.pred_df.filter(pl.col("p_adj") < T)
    recall_frame = (
        filt_real.join(filt_pred, on=["target", "feature"], how="inner", coalesce=True)
        .group_by("target").len()
        .join(filt_real.group_by("target").len(), on="target", how="full",
              suffix="_expected", coalesce=True)
        .fill_null(0)
        .with_columns(recall=pl.col("len") / pl.col("len_expected"))
        .select(["target", "recall"])
    )
    return {row[0]: _to_float_nan(row[1]) for row in recall_frame.iter_rows()}


def de_direction_match(
    prepared: PreparedDE | None = None,
    *,
    de_pred=None,
    de_real=None,
    control: str = "non-targeting",
    sort_by: str = "abs_log2_fold_change",
    p_adj_threshold: float = 0.05,
    nan_lfc_policy: str = "mask",
    min_abs_log2fc: float = 0.0,
) -> dict[str, float]:
    """Per-perturbation fraction of real-significant genes whose log2FC sign agrees
    with pred (inner-join real-sig to pred). Matches cell-eval DEDirectionMatch."""
    prepared = _ensure_prepared(prepared, de_pred, de_real, control=control,
                                sort_by=sort_by, p_adj_threshold=p_adj_threshold,
                                nan_lfc_policy=nan_lfc_policy,
                                min_abs_log2fc=min_abs_log2fc)
    T = prepared.p_adj_threshold
    merged = (
        prepared.real_df.filter(pl.col("p_adj") < T)
        .join(prepared.pred_df, on=["target", "feature"], suffix="_pred", how="inner")
    )
    # No fill on the sign equality, matching upstream DEDirectionMatch exactly: a NaN pred LFC
    # gives sign(NaN)==sign(real) -> False (counted as a mismatch); a genuine null -> null
    # (ignored by pl.mean), same as upstream. fill_null(False) would diverge on the null case.
    agg = (
        merged.with_columns(
            direction_match=(
                pl.col("log2_fold_change").sign() == pl.col("log2_fold_change_pred").sign()
            )
        )
        .group_by("target").agg(pl.mean("direction_match"))
    )
    # Targets with zero real-significant genes are omitted (inner-join drops them), matching
    # upstream DEDirectionMatch. Emitting them as NaN would diverge from upstream AND poison
    # aggregate_metrics' mean (polars propagates NaN). Same rationale as de_sig_recall.
    return {row[0]: _to_float_nan(row[1]) for row in agg.iter_rows()}


def de_model_direction_match(
    prepared: PreparedDE | None = None,
    *,
    de_pred=None,
    de_real=None,
    control: str = "non-targeting",
    sort_by: str = "abs_log2_fold_change",
    p_adj_threshold: float = 0.05,
    nan_lfc_policy: str = "mask",
    min_abs_log2fc: float = 0.0,
) -> dict[str, float]:
    """Per-perturbation fraction of model-significant genes whose log2FC sign agrees
    with real (inner-join pred-sig to real).

    This is the model-DE-conditioned reverse of :func:`de_direction_match`: predicted
    significance selects the denominator, while real significance is deliberately ignored.
    """
    prepared = _ensure_prepared(prepared, de_pred, de_real, control=control,
                                sort_by=sort_by, p_adj_threshold=p_adj_threshold,
                                nan_lfc_policy=nan_lfc_policy,
                                min_abs_log2fc=min_abs_log2fc)
    T = prepared.p_adj_threshold
    merged = (
        prepared.pred_df.filter(pl.col("p_adj") < T)
        .select("target", "feature", "log2_fold_change")
        .join(
            prepared.real_df.select("target", "feature", "log2_fold_change"),
            on=["target", "feature"], suffix="_real", how="inner",
        )
    )
    # Mirror de_direction_match's null/NaN semantics. A NaN real LFC compares unequal and
    # counts as a mismatch; a genuine null comparison is ignored by pl.mean.
    agg = (
        merged.with_columns(
            direction_match=(
                pl.col("log2_fold_change").sign()
                == pl.col("log2_fold_change_real").sign()
            )
        )
        .group_by("target").agg(pl.mean("direction_match"))
    )
    # Targets with zero model-significant genes are omitted here. v2 dispatch fills them
    # with the catalog's worst value (0); v1 retains the raw omitted shape.
    return {row[0]: _to_float_nan(row[1]) for row in agg.iter_rows()}


def de_lfc_spearman(
    prepared: PreparedDE | None = None,
    *,
    de_pred=None,
    de_real=None,
    control: str = "non-targeting",
    sort_by: str = "abs_log2_fold_change",
    p_adj_threshold: float = 0.05,
    nan_lfc_policy: str = "mask",
    min_abs_log2fc: float = 0.0,
    lfc_direction: Literal["all", "pos", "neg"] = "all",
) -> dict[str, float]:
    """Per-perturbation Spearman correlation of log2FCs over real-significant genes
    (real-sig left-join pred, pred LFC null->0), optionally restricted by the sign of the
    REAL log2FC: `lfc_direction` 'all' (default) | 'pos' (real LFC>0) | 'neg' (real LFC<0).
    Matches cell-eval DESpearmanLFC(lfc_direction=...)."""
    if lfc_direction not in ("all", "pos", "neg"):
        raise ValueError(
            f"lfc_direction must be 'all', 'pos', or 'neg', got {lfc_direction!r}"
        )
    prepared = _ensure_prepared(prepared, de_pred, de_real, control=control,
                                sort_by=sort_by, p_adj_threshold=p_adj_threshold,
                                nan_lfc_policy=nan_lfc_policy,
                                min_abs_log2fc=min_abs_log2fc)
    T = prepared.p_adj_threshold
    real_sig = prepared.real_df.filter(pl.col("p_adj") < T)
    # Restrict to up-/down-regulated real-significant genes by the REAL LFC sign, mirroring
    # upstream DESpearmanLFC's `match lfc_direction`. NaN note: polars `NaN > 0` is True, but
    # under v2 (nan_lfc_policy='mask') NaN-LFC rows are already dropped by the significance gate
    # above, so `> 0` never sees a NaN; under v1 (keep) upstream's polars `> 0` includes them
    # and plain `> 0` reproduces that (byte-parity) — do NOT add an is_nan() guard here.
    if lfc_direction == "pos":
        real_sig = real_sig.filter(pl.col("log2_fold_change") > 0)
    elif lfc_direction == "neg":
        real_sig = real_sig.filter(pl.col("log2_fold_change") < 0)
    # Project to just the columns the correlation needs BEFORE the join, so we don't
    # materialize the other suffixed pred columns (p_adj_pred, abs_log2_fold_change_pred, ...).
    merged = (
        real_sig.select("target", "feature", "log2_fold_change")
        .join(
            prepared.pred_df.select("target", "feature", "log2_fold_change"),
            on=["target", "feature"], suffix="_pred", how="left",
        )
        # fill_null only, matching upstream DESpearmanLFC (also polars fill_null): a NaN pred
        # LFC stays NaN and polars spearman handles it without crashing (unlike sklearn), so
        # no fill_nan is needed; adding one would diverge from upstream.
        .with_columns(pl.col("log2_fold_change_pred").fill_null(0.0))
    )
    agg = merged.group_by("target").agg(
        pl.corr(
            pl.col("log2_fold_change").cast(pl.Float64),
            pl.col("log2_fold_change_pred").cast(pl.Float64),
            method="spearman",
        ).alias("spearman_corr")
    )
    # Targets with zero qualifying real-significant genes are omitted (no rows after the
    # real-sig filter), matching upstream DESpearmanLFC; see de_direction_match for why NaN-
    # filling is avoided.
    return {row[0]: _to_float_nan(row[1]) for row in agg.iter_rows()}


def de_lfc_nmae(
    prepared: PreparedDE | None = None,
    *,
    de_pred=None,
    de_real=None,
    control: str = "non-targeting",
    sort_by: str = "abs_log2_fold_change",
    p_adj_threshold: float = 0.05,
    nan_lfc_policy: str = "mask",
    min_abs_log2fc: float = 0.0,
    min_gate_size: int = 10,
) -> dict[str, float]:
    """Per-perturbation normalized MAE of log2 fold changes over real-significant genes
    (issue #208): ``mean|lfc_pred - lfc_real| / mean|lfc_real|``, lower is better.

    At ``lfc_pred = 0`` the numerator IS the denominator, so a prediction whose LOG2FC is
    exactly zero scores exactly 1.0 on every dataset. That is what fixes one end of the
    scale without any reference to the evaluation data. (On every perturbation the metric
    RETURNS, that is: the gate rules below drop a perturbation whose gate is empty, smaller
    than ``min_gate_size``, or whose denominator is zero or non-finite -- those score nothing
    rather than 1.0. All three are real-side decisions, so the surviving set is the same for every
    submission.)

    ⚠️ Issue #286 -- that anchor is exact in LFC SPACE, and the step from it to "a
    submission predicting no change scores exactly 1.0", which this docstring used to make,
    does not hold under ``control_source="real"`` (the v2 default, and what ``vcc2026``
    scores). A predicted perturbation's LFC is then computed against the REAL control's own
    cells rather than the prediction's, so even a submission broadcasting the exact,
    unrounded true control mean to every cell is not compared against itself: §1.4's DE mean
    is a per-cell normalize-then-average, which NEED NOT agree with normalizing the mean --
    the dispersion-functional asymmetry #264 fixed for the ``expr_*``/``pds_*`` comparator and
    deliberately did NOT apply to DE. Such a submission can therefore land NEAR 1.0 rather
    than AT it, and can land below it for free.

    ⚠️ What decides whether the two functionals agree is depth-COMPOSITION COVARIANCE, and
    #286 and §1.2 both say "heterogeneous library sizes" instead. With ``pi_c = x_c / L_c``
    the per-cell composition,

        CPM(mean_c x_c) - mean_c CPM(x_c) = 1e6 * Cov_c(L_c, pi_c) / E[L].

    Depth spread is NECESSARY (no spread, no covariance) and nowhere near SUFFICIENT: a panel
    whose cells differ 10x in depth while sharing ONE composition has a discrepancy of exactly
    zero, because every cell's CPM vector is literally the same vector. Measured on two panels
    carrying the same depth multiset, the same composition multiset and the same library-size
    CV (0.8182) -- differing only in which depth each composition is PAIRED with, so the
    real-side DE gate and the real log2FCs -- hence the nmae denominator -- are the same, checked
    by running the production DE path on both: the anchor reads exactly 1.00000000 at zero
    covariance, and 1.2010 / 1.1936 for the two pairings with it
    (``tests/test_lfc_nmae_anchor_286.py`` pins all three).

    Measured, both directions:

    * committed fixture (100 control cells, library-size CV 0.3373, 5 perturbations,
      ``p_adj_threshold=1.0`` to widen the gate): the exact-control-mean submission reads
      0.9397-1.0397, THREE of five below 1.0. Under ``control_source="pred"`` all five read
      exactly 1.0, confirming the mechanism rather than the arithmetic. Reproduced on both
      CPU backends -- pdex and scanpy agree to 3 decimals; the GPU backend was not run, so
      "backend-agnostic" is a claim about those two.
    * official val panels A/B/C (300 panel targets each, of which this member RETURNS
      272 / 229 / 218 -- the rest fail the real-side gate; competition preset): the same
      submission class reads 1.0058 / 1.0047 / 1.0097 -- within 0.97%, and ABOVE the frozen
      base, i.e. on the penalized side -- while its dispersed sibling reads 0.9976 / 0.9987 / 0.9988 and a dispersed
      context-mean arm reaches 0.9909, i.e. up to 0.91% BELOW the no-skill point. That arm
      is an ORACLE comparator, not a floor a submission could reach (`docs/metrics.md` 6a):
      it bounds the metric's triviality, and is not a score a model collects for free.
      Averaging over the returned perturbations is what shrinks the fixture's 6%; the
      per-perturbation deviation does not shrink.

    Consequence, and it is narrow: the enrolled ``avg_score`` normalizes against a MEASURED
    baseline, so nothing there reads the 1.0. What reads it is `low-random_high-1_v10`'s
    ``base`` for this member -- and a requested ``--scale`` column carries its OWN
    ``avg_score``, which does read every scale base. That base is a policy constant -- the
    same situation §6d already records for
    ``expr_mse_unbiased_capped_norm``, whose own no-skill point drifts 2.5% on the shipped
    comparator. The competition rule pins ``control_source`` as ``real`` for the scored leg
    and ``pred`` for the anchor leg, and the invariant is exact only on the latter.

    ⚠️ **EXCLUDES the perturbed gene's own row from the gate (issue #172, ruled 2026-08-17).**
    The knocked-down gene's own log2FC is the experiment's premise rather than a prediction, and
    this was one of the last two scored `vcc2026` DE members still scoring it. Because the gate
    supplies BOTH the numerator and the denominator here, removing the row removes it from
    ``mean|lfc_pred - lfc_real|`` and from ``mean|lfc_real|`` together -- so unlike
    `de_sig_jaccard`, whose loss is one-sided, the effect is a ratio of two shrinking means and
    is small and signed either way: measured on val A (honest half-data arm) the raw value falls
    0.194301 -> 0.191268, which is +0.005 on the SCALED score because lower is better
    (`docs/metrics.md` §4).

    The gate SIZE shrinks by one for every resolved target, which interacts with
    ``min_gate_size``: a perturbation sitting at exactly the threshold before exclusion falls
    below it after and is omitted. That is a real-side decision like every other rule here, so
    the omitted set stays identical for every submission.

    Three deliberate divergences from :func:`de_lfc_spearman`, which shares this gate:

    * A **non-finite** predicted LFC is filled with 0.0, exactly like a null -- and
      DELIBERATELY AGAINST #208 5.2, which says to mask it. 5.2's concern is that one
      +/-inf would dominate the mean; filling answers that just as completely, and masking
      does not survive the paragraph below. A model that emits nothing usable for a gene
      predicted no change, whether it said `null` or `inf`.
    * ``min_gate_size`` (default 10) omits a perturbation whose gate is smaller. A ratio
      over a handful of just-over-threshold genes is noise. ``de_lfc_spearman`` omits only
      on an EMPTY gate; this is a stricter rule and therefore an explicit parameter.
    * A zero or non-finite denominator omits the perturbation rather than returning
      ``inf`` -- the code's own warning names both (``de.py`` rejects
      ``den is None or not math.isfinite(den) or den == 0.0``).

    Omission is NOT gameable, and that is a CONSTRAINT on the code above rather than an
    observation about it. The gate, the gate SIZE and the denominator are all computed from
    ``real_df`` before ``pred_df`` is joined, so which perturbations are dropped is a
    property of the evaluation data and is identical for every submission. Masking a
    non-finite prediction would break this two ways: a submission could emit NaNs until a
    perturbation fell below ``min_gate_size`` and vanished from its own aggregate, and an
    all-``inf`` model would leave an EMPTY numerator over a non-empty denominator and score
    0.0 -- perfect. Filled, it scores exactly 1.0, the same as silence.

    Non-gameable omission is also what makes ``worst_value=None`` safe in the catalog:
    ``nmae`` is unbounded above, so there is no finite worst value to fill an omitted
    perturbation with, and a constant would be an invented number.
    """
    if min_gate_size < 1:
        raise ValueError(f"min_gate_size must be >= 1, got {min_gate_size!r}")
    prepared = _ensure_prepared(prepared, de_pred, de_real, control=control,
                                sort_by=sort_by, p_adj_threshold=p_adj_threshold,
                                nan_lfc_policy=nan_lfc_policy,
                                min_abs_log2fc=min_abs_log2fc)
    T = prepared.p_adj_threshold
    # Issue #172: the perturbed gene's own row leaves the GATE, so it counts in neither the
    # numerator nor the denominator nor `n_gate`. A real-side decision like the two filters
    # above, so which perturbations are dropped stays a property of the evaluation data and
    # the non-gameability argument below survives it unchanged.
    real_sig = _exclude_own_gene(
        prepared.real_df.filter(pl.col("p_adj") < T)
        # Real-side non-finite LFCs leave the GATE, before the prediction is joined. A real
        # NaN has no target value to be right about. This is a real-side decision, so it is
        # identical for every submission and the guarantee in the docstring survives it.
        .filter(pl.col("log2_fold_change").is_finite())
        .select("target", "feature", "log2_fold_change"),
        prepared, who="de_lfc_nmae", side="real (gate)",
    )
    # Duplicate (target, feature) rows would change BOTH the gate size and the denominator
    # with no other signal, so they raise rather than being silently aggregated. Checked on
    # the gate rather than the whole table -- and therefore AFTER the on-target anti-join,
    # for the same reason: rows outside the gate cannot affect this metric.
    dup = real_sig.group_by("target", "feature").len().filter(pl.col("len") > 1)
    if dup.height:
        raise ValueError(
            f"real DE table has {dup.height} duplicate (target, feature) row(s) inside the "
            f"significance gate, e.g. {dup.row(0)[:2]}; de_lfc_nmae cannot aggregate them "
            "silently -- the duplicate changes the gate size and the denominator."
        )
    merged = real_sig.join(
        prepared.pred_df.select("target", "feature", "log2_fold_change"),
        on=["target", "feature"], suffix="_pred", how="left",
    )
    # A LEFT join can only grow beyond the left frame if the RIGHT side has duplicate keys.
    if merged.height != real_sig.height:
        raise ValueError(
            f"pred DE table has duplicate (target, feature) rows: the gate has "
            f"{real_sig.height} rows but the join produced {merged.height}; de_lfc_nmae "
            "cannot aggregate them silently."
        )
    # Counted BEFORE the substitution, and only for values that were actually present:
    # `is_finite()` is null for a null, so `is_not_null() & ~is_finite()` counts NaN/inf
    # without counting the ordinary absent-gene case.
    n_nonfinite = int(merged.select(
        (pl.col("log2_fold_change_pred").is_not_null()
         & ~pl.col("log2_fold_change_pred").is_finite()).sum()
    ).item())
    # ONE expression for null AND non-finite -> 0.0. `when(null)` takes the otherwise
    # branch, so nulls are covered without a separate fill_null. NEVER replace this with a
    # filter: dropping the row would shrink n_gate below, which is the whole gameability
    # hole (see the docstring).
    merged = merged.with_columns(
        pl.when(pl.col("log2_fold_change_pred").is_finite())
        .then(pl.col("log2_fold_change_pred"))
        .otherwise(0.0)
        .alias("log2_fold_change_pred")
    )
    agg = merged.group_by("target").agg(
        num=(pl.col("log2_fold_change_pred") - pl.col("log2_fold_change")).abs().mean(),
        den=pl.col("log2_fold_change").abs().mean(),
        n_gate=pl.len(),
    )
    # From the COMPLETE real target set, not from `agg`: a target whose gate is empty
    # produces no group row at all, so counting from the aggregate misses exactly the case
    # most worth reporting -- the perturbation nothing was significant for.
    empty: list[str] = sorted(set(prepared.real_df["target"].unique().to_list())
                              - set(agg["target"].to_list()))
    out: dict[str, float] = {}
    small: list[str] = []
    degenerate: list[str] = []
    for target, num, den, n_gate in agg.iter_rows():
        if n_gate < min_gate_size:
            small.append(target)
            continue
        if den is None or not math.isfinite(den) or den == 0.0:
            degenerate.append(target)
            continue
        out[target] = float(num) / float(den)
    # Substituting a value must never be invisible, even though it is the honest reading.
    if n_nonfinite:
        logger.warning(
            "de_lfc_nmae: %d non-finite predicted log2FC value(s) treated as no-change "
            "(0.0). They are NOT removed from the gate -- the gate is real-side only.",
            n_nonfinite,
        )
    # Silently dropping the weakest perturbations flatters every submission equally, so the
    # count is reported by REASON rather than inferred from a shorter result.
    for reason, names in (("an empty gate", empty),
                          (f"fewer than {min_gate_size} gated genes", small),
                          ("a zero or non-finite mean|real log2FC|", degenerate)):
        if names:
            logger.warning(
                "de_lfc_nmae: omitted %d perturbation(s) for %s (e.g. %s). The gate is "
                "real-side only, so this set is the same for every submission.",
                len(names), reason, names[0],
            )
    return out


def _de_auc(prepared: PreparedDE, method: str, *,
            auc_pval_floor: str = "min_nonzero",
            auc_pval_floor_value: float = 1e-10) -> dict[str, float]:
    """Per-perturbation PR/ROC AUC for recovering real-significant genes. Label =
    (real p_adj < T). Pred score = -log10(floor(pred p_adj)), where the floor is:
      - "clip":         clip(value, 1.0)         -- ties zeros + sub-value (legacy cev2)
      - "replace_zero": replace 0 -> value       -- only exact zeros (cell-eval exact)
      - "min_nonzero":  replace 0 -> smallest nonzero pred p (order-consistent; v2 default)
    NaN AUC when a target has no rows or single-class labels."""
    if auc_pval_floor not in ("clip", "replace_zero", "min_nonzero"):
        raise ValueError(f"auc_pval_floor must be one of "
                         f"('clip','replace_zero','min_nonzero'), got {auc_pval_floor!r}")
    if not (0.0 < auc_pval_floor_value <= 1.0):
        raise ValueError(f"auc_pval_floor_value must be in (0, 1], "
                         f"got {auc_pval_floor_value!r}")
    T = prepared.p_adj_threshold
    labeled_real = (
        prepared.real_df
        .with_columns((pl.col("p_adj") < T).cast(pl.Float32).alias("label"))
        .select(["target", "feature", "label"])
    )
    # fill_nan BEFORE fill_null: polars fill_null does NOT replace float NaN, and a NaN p_adj
    # (degenerate genes from the CPM-gate BH recompute / wilcoxon) would propagate through
    # the floor + -log10 to a NaN score and crash sklearn. Treat a NaN/absent pred p_adj as
    # non-significant (p_adj=1 -> score 0).
    merged = (
        labeled_real.join(
            prepared.pred_df.select(["target", "feature", "p_adj"]),
            on=["target", "feature"], how="left", coalesce=True,
        )
        .drop_nulls(["label"])
        .with_columns(pl.col("p_adj").fill_nan(1.0).fill_null(1.0).alias("p_adj"))
    )
    fv = auc_pval_floor_value
    if auc_pval_floor == "clip":
        q = pl.col("p_adj").clip(fv, 1.0)
    elif auc_pval_floor == "replace_zero":
        # cell-eval stores DE numerics as Float32 (DEResults), so pred p < ~1.4e-45 underflow
        # to 0 BEFORE its replace(0, 1e-10). Reproduce that round-trip, then replace 0 -> value,
        # so v1 matches cell-eval's compute_generic_auc bit-for-bit (incl. the underflow tie).
        q = pl.col("p_adj").cast(pl.Float32).cast(pl.Float64).replace(0.0, fv)
    else:  # min_nonzero
        # in-engine min over the single p_adj column (no full-frame materialization);
        # .item() is None when every scored p_adj is 0 -> fall back to fv.
        nz = merged.select(pl.col("p_adj").filter(pl.col("p_adj") > 0).min()).item()
        q = pl.col("p_adj").replace(0.0, fv if nz is None else nz)
    merged = merged.with_columns((-q.log10()).alias("nlp"))
    # Partition once (O(rows)) rather than filtering the full frame per perturbation
    # (O(perts x rows)) -- matters on real data with many perturbations. Result-identical to
    # upstream compute_generic_auc's per-pert filter. as_dict keys are 1-tuples in polars >=1.
    out: dict[str, float] = {}
    partitioned = merged.partition_by("target", as_dict=True)
    for pert in prepared.perturbations:
        sub = partitioned.get((pert,))
        if sub is None or sub.height == 0:
            out[pert] = float("nan")
            continue
        labels = sub["label"].to_numpy()
        scores = sub["nlp"].to_numpy()
        if not (0 < labels.sum() < len(labels)):
            out[pert] = float("nan")
            continue
        if method == "pr":
            out[pert] = float(average_precision_score(labels, scores))
        else:
            fpr, tpr, _ = roc_curve(labels, scores)
            out[pert] = float(auc(fpr, tpr))
    return out


def de_pr_auc(
    prepared: PreparedDE | None = None,
    *,
    de_pred=None,
    de_real=None,
    control: str = "non-targeting",
    sort_by: str = "abs_log2_fold_change",
    p_adj_threshold: float = 0.05,
    nan_lfc_policy: str = "mask",
    min_abs_log2fc: float = 0.0,
    auc_pval_floor: str = "min_nonzero",
    auc_pval_floor_value: float = 1e-10,
) -> dict[str, float]:
    """Per-perturbation precision-recall AUC for significant-gene recovery."""
    prepared = _ensure_prepared(prepared, de_pred, de_real, control=control,
                                sort_by=sort_by, p_adj_threshold=p_adj_threshold,
                                nan_lfc_policy=nan_lfc_policy,
                                min_abs_log2fc=min_abs_log2fc)
    return _de_auc(prepared, "pr", auc_pval_floor=auc_pval_floor,
                   auc_pval_floor_value=auc_pval_floor_value)


def de_roc_auc(
    prepared: PreparedDE | None = None,
    *,
    de_pred=None,
    de_real=None,
    control: str = "non-targeting",
    sort_by: str = "abs_log2_fold_change",
    p_adj_threshold: float = 0.05,
    nan_lfc_policy: str = "mask",
    min_abs_log2fc: float = 0.0,
    auc_pval_floor: str = "min_nonzero",
    auc_pval_floor_value: float = 1e-10,
) -> dict[str, float]:
    """Per-perturbation ROC AUC for significant-gene recovery."""
    prepared = _ensure_prepared(prepared, de_pred, de_real, control=control,
                                sort_by=sort_by, p_adj_threshold=p_adj_threshold,
                                nan_lfc_policy=nan_lfc_policy,
                                min_abs_log2fc=min_abs_log2fc)
    return _de_auc(prepared, "roc", auc_pval_floor=auc_pval_floor,
                   auc_pval_floor_value=auc_pval_floor_value)
