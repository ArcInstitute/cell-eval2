"""Reference-only scoreability gates (issue #391).

A gate decides, from the REFERENCE alone, which perturbations a metric can score. It never reads
the prediction, so the set it admits is identical for every submission scored against the same
reference, whatever order the dispatch evaluates it in, and omission through it cannot be gamed
-- the argument `de_lfc_nmae` makes for its own gate
(`internal:docs/internal/superpowers/specs/2026-08-02-de-lfc-nmae-design.md` §2.3).

The v2 dispatch (`run.dispatch_de_metrics`, `run.dispatch_anndata_metrics`) and the partition
reducer (`partition._inject_nsig_spearman`) are the only consumers. A perturbation outside the
gate is emitted as NaN and leaves the aggregate; inside it, a prediction that makes the metric
undefined is charged the metric's `worst_value`. v1 never reaches this module.

Design: `internal:docs/internal/superpowers/specs/2026-10-08-real-side-nan-not-scored-design.md`.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, Sequence

import numpy as np
import polars as pl

if TYPE_CHECKING:
    from .de import PreparedDE

#: The floor every gated metric shares unless its catalog entry overrides it (ruled 2026-10-08).
#: Deliberately NOT `de_lfc_nmae`'s `min_gate_size`: that is a competition member with a different
#: gate (target gene excluded, non-finite reference LFCs dropped), so a change here must not move
#: it. And deliberately not an `EvalConfig` field: any new config field enters `rule_config_hash`
#: and with it `competition_digest`.
DEFAULT_MIN_GATE_SIZE = 10

GateKind = Literal["sig_set", "sig_lfc", "two_class", "sig_universe", "ranked_universe",
                   "rank_corr", "count_corr", "adjudicable", "lfc_present", "delta_var"]
_KINDS: tuple[str, ...] = ("sig_set", "sig_lfc", "two_class", "sig_universe", "ranked_universe",
                           "rank_corr", "count_corr", "adjudicable", "lfc_present", "delta_var")
#: The kinds whose metric reaches the reference through an inner join on (target, feature): a
#: row whose feature is null can never match there, so it is not one of their points.
_NAMED_ONLY: frozenset[str] = frozenset({"sig_lfc", "ranked_universe", "adjudicable",
                                         "lfc_present"})
_DIRECTIONS: tuple[str, ...] = ("all", "pos", "neg")
#: The kinds whose metric is a correlation the PREDICTION can make undefined by being constant --
#: and for which the backend does not reliably return NaN, so the dispatch checks it: polars'
#: Spearman (spec §5.2), and scipy's Pearson, which for a constant two-gene input returns 0.0 at
#: 1.14.1 (measured; 1.11.4 and 1.18.0 return NaN).
CONSTANCY_CHECKED: frozenset[str] = frozenset({"rank_corr", "count_corr", "delta_var"})


@dataclass(frozen=True)
class RealGate:
    """Which reference-side condition a metric needs before it can score a perturbation.

    Each kind mirrors the reference-side population its metrics actually compute over -- which is
    why there are several, and why two that look alike are not interchangeable:

    * ``sig_set`` -- at least ``min_gate_size`` reference-significant genes.
    * ``sig_lfc`` -- the same, counting only significant genes whose reference log2FC is not null:
      a null sign compares null and is ignored by the mean, so it is not a point.
    * ``two_class`` -- ``sig_set``, plus at least one LABELLED gene that is not significant. A
      gene is labelled when its ``p_adj`` is not null (`metrics.de._de_auc`'s rule; a NaN
      ``p_adj`` is a negative).
    * ``sig_universe`` -- ``sig_set``, plus at least one reference row that is not significant,
      null ``p_adj`` included: `metrics.de._tested_universe` counts every row, so the reference
      side of the chance-corrected 2x2 tables is degenerate only when EVERY row is significant.
    * ``ranked_universe`` -- the same over `de_overlap_adjusted`'s own reference set, which comes
      from ``real_rank[...].drop_nulls()`` and so leaves out a row whose feature is null: at least
      ``min_gate_size`` significant NAMED genes, fewer than the reference rows.
    * ``rank_corr`` -- over the ``lfc_direction`` subset of the reference-significant genes (by
      the sign of the REFERENCE log2FC) whose reference log2FC is not null -- polars' correlation
      drops a pair with a null: at least ``min_gate_size`` genes and at least two distinct
      reference log2FC values.
    * ``count_corr`` -- GLOBAL: at least ``min_gate_size`` targets carry a reference-significant
      gene, and their significant-gene counts take at least two distinct values. Every
      perturbation is inside, or none is.
    * ``adjudicable`` -- at least ``min_gate_size`` reference genes, of any significance, whose
      log2FC carries a direction (not null, NaN or zero): the only genes
      `metrics.direction.de_direction_precision` can adjudicate. Never binds on a real panel; it
      exists so a reference that adjudicates nothing is not charged to the prediction.
    * ``lfc_present`` -- at least ``min_gate_size`` reference genes whose log2FC is not null:
      `metrics.de.de_model_direction_match` ignores a null comparison. Never binds on a real
      panel either.
    * ``delta_var`` -- the reference delta (perturbation minus control) has at least two genes,
      all finite, taking at least two distinct values. No floor: its points are every measured
      gene, not a reference selection.

    ``sig_lfc``, ``ranked_universe``, ``adjudicable`` and ``lfc_present`` count NAMED genes only:
    their metrics meet the reference through an inner join on ``(target, feature)``, where a null
    feature never matches.

    A declarative record rather than a callable, so it serializes BY VALUE into the result-cache
    key and the partition semantics payload (`payload`).
    """

    kind: GateKind
    min_gate_size: int | None = DEFAULT_MIN_GATE_SIZE
    lfc_direction: Literal["all", "pos", "neg"] = "all"

    def __post_init__(self) -> None:
        if self.kind not in _KINDS:
            raise ValueError(f"RealGate.kind must be one of {_KINDS}, got {self.kind!r}")
        if self.kind == "delta_var":
            if self.min_gate_size is not None:
                raise ValueError(
                    "a 'delta_var' gate takes no floor (min_gate_size=None): its points are every "
                    f"measured gene, not a reference selection; got {self.min_gate_size!r}"
                )
        elif (isinstance(self.min_gate_size, bool) or not isinstance(self.min_gate_size, int)
              or self.min_gate_size < 1):
            raise ValueError(
                f"RealGate({self.kind!r}).min_gate_size must be an int >= 1, "
                f"got {self.min_gate_size!r}"
            )
        if self.lfc_direction not in _DIRECTIONS:
            raise ValueError(
                f"RealGate.lfc_direction must be one of {_DIRECTIONS}, got {self.lfc_direction!r}"
            )
        if self.kind != "rank_corr" and self.lfc_direction != "all":
            raise ValueError(
                f"lfc_direction={self.lfc_direction!r} is meaningful only for a 'rank_corr' gate, "
                f"not {self.kind!r}"
            )

    def payload(self) -> list:
        """Stable, JSON-able identity for cache keys and the partition semantics payload."""
        return [self.kind, self.min_gate_size, self.lfc_direction]


@dataclass(frozen=True)
class GateResult:
    """``inside``: the perturbations the reference can score. ``outside``: every other in-scope
    perturbation, mapped to the reason it is not scored (for the log)."""

    inside: frozenset[str]
    outside: dict[str, str]


def _reference_sig(prepared: "PreparedDE") -> pl.DataFrame:
    """The reference-significant rows, as every gated DE metric selects them: ``p_adj < T``."""
    return prepared.real_df.filter(pl.col("p_adj") < prepared.p_adj_threshold)


def _directional(df: pl.DataFrame, lfc_direction: str) -> pl.DataFrame:
    """Restrict to up-/down-regulated rows by the REFERENCE log2FC sign, with exactly the
    expressions `metrics.de.de_lfc_spearman` uses (a bare `> 0` / `< 0`, no NaN guard)."""
    if lfc_direction == "pos":
        return df.filter(pl.col("log2_fold_change") > 0)
    if lfc_direction == "neg":
        return df.filter(pl.col("log2_fold_change") < 0)
    return df


def _counts(frame: pl.DataFrame, **aggs: pl.Expr) -> dict[str, dict[str, int]]:
    """{target: {name: value}} from a group_by('target') aggregation."""
    if frame.height == 0:
        return {}
    out = frame.group_by("target").agg(**aggs)
    return {row["target"]: {name: row[name] for name in aggs}
            for row in out.iter_rows(named=True)}


def count_corr_decision(real_counts: Sequence[float], *, gate: RealGate) -> str | None:
    """`nsig_spearman`'s global gate over the targets that carry a reference-significant gene.

    ``real_counts`` are those targets' counts (the caller has already dropped the zeros, the
    upstream `DESpearmanSignificant` population). Returns the reason the metric cannot be
    scored, or None when it can. Shared by the whole-prediction dispatch and the partition
    reducer, so the two cannot drift.
    """
    if gate.kind != "count_corr":
        raise ValueError(f"count_corr_decision needs a 'count_corr' gate, got {gate.kind!r}")
    values = [float(v) for v in real_counts]
    if len(values) < gate.min_gate_size:
        return (f"fewer than {gate.min_gate_size} targets carry a reference-significant gene "
                f"({len(values)})")
    if len(set(values)) < 2:
        return "every such target has the same reference-significant gene count"
    return None


def prediction_counts_constant(pred_counts: Sequence[float]) -> bool:
    """True when the prediction's counts over the gated targets take a single value."""
    return len({float(v) for v in pred_counts}) <= 1


def _nsig_frames(prepared: "PreparedDE") -> tuple[list[float], list[float]]:
    """(reference counts, prediction counts) over the targets with a reference-significant
    gene, built exactly as `metrics.de.de_nsig_spearman` builds them (LEFT join, missing -> 0)."""
    T = prepared.p_adj_threshold
    filt_real = prepared.real_df.filter(pl.col("p_adj") < T).group_by("target").len()
    filt_pred = prepared.pred_df.filter(pl.col("p_adj") < T).group_by("target").len()
    merged = filt_real.join(filt_pred, on="target", how="left",
                            suffix="_pred", coalesce=True).fill_null(0)
    return merged["len"].to_list(), merged["len_pred"].to_list()


def evaluate_de_gate(prepared: "PreparedDE", gate: RealGate) -> GateResult:
    """The perturbations a DE metric with ``gate`` can score, from ``prepared.real_df`` ALONE.

    Never reads ``prepared.pred_df``: that is the whole contract, pinned by a test that swaps the
    prediction and asserts an identical result. Seeded from ``prepared.perturbations``, so a
    target with no reference rows at all is reported (empty gate) rather than silently absent.
    Memoized on the PreparedDE by the gate's payload, like the direction family's memos --
    sound because the result depends only on fields of that object and on the gate.
    """
    if gate.kind == "delta_var":
        raise ValueError("a 'delta_var' gate reads pseudobulks; use evaluate_delta_gate")
    cache = getattr(prepared, "_real_gate_cache", None)
    key = tuple(gate.payload())
    if cache is not None and key in cache:
        return cache[key]
    result = _evaluate_de_gate(prepared, gate)
    if cache is None:
        cache = {}
        try:
            prepared._real_gate_cache = cache
        except (AttributeError, ValueError, TypeError):   # read-only view -> skip the memo
            return result
    cache[key] = result
    return result


def _evaluate_de_gate(prepared: "PreparedDE", gate: RealGate) -> GateResult:
    perts = list(prepared.perturbations)
    m = gate.min_gate_size
    if gate.kind == "count_corr":
        # The reference half of `_nsig_frames`, built WITHOUT touching the prediction.
        real_counts = (prepared.real_df.filter(pl.col("p_adj") < prepared.p_adj_threshold)
                       .group_by("target").len()["len"].to_list())
        reason = count_corr_decision(real_counts, gate=gate)
        if reason is None:
            return GateResult(frozenset(perts), {})
        return GateResult(frozenset(), {p: reason for p in perts})
    named = pl.col("feature").is_not_null()
    sig = _reference_sig(prepared)
    if gate.kind in _NAMED_ONLY:
        sig = sig.filter(named)
    inside: set[str] = set()
    outside: dict[str, str] = {}
    if gate.kind in ("adjudicable", "lfc_present"):
        lfc = pl.col("log2_fold_change")
        usable = (lfc.is_not_null() & ~lfc.is_nan() & (lfc != 0) if gate.kind == "adjudicable"
                  else lfc.is_not_null())
        what = ("reference genes with a log2FC direction" if gate.kind == "adjudicable"
                else "reference genes with a log2FC")
        stats = _counts(prepared.real_df.filter(usable & named).select("target"), n=pl.len())
        for p in perts:
            k = stats.get(p, {"n": 0})["n"]
            if k < m:
                outside[p] = f"fewer than {m} {what}"
            else:
                inside.add(p)
        return GateResult(frozenset(inside), outside)
    if gate.kind == "rank_corr":
        stats = _counts(
            _directional(sig, gate.lfc_direction)
            .filter(pl.col("log2_fold_change").is_not_null())
            .select("target", "log2_fold_change"),
            n=pl.len(), distinct=pl.col("log2_fold_change").cast(pl.Float64).n_unique(),
        )
        subset = "" if gate.lfc_direction == "all" else f" {gate.lfc_direction}-direction"
        for p in perts:
            s = stats.get(p, {"n": 0, "distinct": 0})
            if s["n"] == 0:
                outside[p] = f"no{subset} reference-significant gene"
            elif s["n"] < m:
                outside[p] = f"fewer than {m}{subset} reference-significant genes"
            elif s["distinct"] < 2:
                outside[p] = "constant reference log2FC over the gate"
            else:
                inside.add(p)
        return GateResult(frozenset(inside), outside)
    if gate.kind == "sig_lfc":
        sig = sig.filter(pl.col("log2_fold_change").is_not_null())
    n_sig = _counts(sig.select("target"), n=pl.len())
    n_rows: dict[str, dict[str, int]] = {}
    if gate.kind == "two_class":
        # `_de_auc` labels every row whose p_adj is not null; a NaN p_adj is a negative.
        n_rows = _counts(prepared.real_df.filter(pl.col("p_adj").is_not_null()).select("target"),
                         n=pl.len())
    elif gate.kind in ("sig_universe", "ranked_universe"):
        # `_tested_universe` counts every (target, feature) row, null p_adj included.
        n_rows = _counts(prepared.real_df.select("target"), n=pl.len())
    one, many = (("reference-significant gene with a log2FC",
                  "reference-significant genes with a log2FC") if gate.kind == "sig_lfc"
                 else ("reference-significant gene", "reference-significant genes"))
    for p in perts:
        k = n_sig.get(p, {"n": 0})["n"]
        if k == 0:
            outside[p] = f"no {one}"
        elif k < m:
            outside[p] = f"fewer than {m} {many}"
        elif (gate.kind in ("two_class", "sig_universe", "ranked_universe")
              and n_rows.get(p, {"n": 0})["n"] - k < 1):
            outside[p] = ("every labelled gene is reference-significant (single class)"
                          if gate.kind == "two_class"
                          else "every reference row is significant")
        else:
            inside.add(p)
    return GateResult(frozenset(inside), outside)


def constant_prediction(prepared: "PreparedDE", gate: RealGate, inside) -> frozenset[str]:
    """The perturbations INSIDE the gate whose prediction is constant over it (spec §5.2).

    For a ``rank_corr`` gate: the predicted log2FCs over the gated genes, null -> 0.0 exactly as
    `de_lfc_spearman` fills them, take one distinct value (polars' ``n_unique`` counts all NaNs as
    one value and does not split 0.0 from -0.0, which is what a rank statistic sees). For
    ``count_corr``: the predicted counts over the gated targets take one value, which makes every
    perturbation constant at once. ``delta_var`` reads pseudobulks, so it has its own function,
    `constant_delta_prediction`. Any other kind has no such check and returns an empty set.

    This and `constant_delta_prediction` are the only gate-side functions that read the
    PREDICTION, and they only ever narrow a set the reference already admitted.
    """
    if gate.kind == "delta_var":
        raise ValueError("a 'delta_var' gate reads pseudobulks; use constant_delta_prediction")
    inside = frozenset(inside)
    if gate.kind not in CONSTANCY_CHECKED or not inside:
        return frozenset()
    if gate.kind == "count_corr":
        _, pred_counts = _nsig_frames(prepared)
        return inside if prediction_counts_constant(pred_counts) else frozenset()
    # The correlation's own population: a null REFERENCE log2FC drops the pair, so a prediction
    # varying only on such rows is still constant where it is scored.
    real_sig = _directional(_reference_sig(prepared), gate.lfc_direction).filter(
        pl.col("log2_fold_change").is_not_null())
    merged = (
        real_sig.filter(pl.col("target").is_in(sorted(inside)))
        .select("target", "feature")
        .join(prepared.pred_df.select("target", "feature", "log2_fold_change"),
              on=["target", "feature"], how="left")
        .with_columns(pl.col("log2_fold_change").cast(pl.Float64).fill_null(0.0))
    )
    stats = _counts(merged, distinct=pl.col("log2_fold_change").n_unique())
    return frozenset(t for t, s in stats.items() if s["distinct"] <= 1)


def evaluate_delta_gate(real_bulk, *, control: str, perts: Sequence[str]) -> GateResult:
    """The `delta_var` gate: which of ``perts`` have a usable reference delta.

    ``real_bulk`` is the ``(perturbations, means)`` pair `run.dispatch_anndata_metrics` reads.
    The delta is computed exactly as `metrics.delta._delta_eval` computes the real side:
    ``real_means[p] - real_means[control]``.
    """
    real_perts = np.asarray(real_bulk[0]).astype(str)
    real_means = np.asarray(real_bulk[1], dtype=np.float64)
    index = {p: i for i, p in enumerate(real_perts)}
    if control not in index:
        raise ValueError(f"control {control!r} not found in real perturbations")
    ctrl = real_means[index[control]]
    inside: set[str] = set()
    outside: dict[str, str] = {}
    for p in perts:
        if p not in index:
            outside[p] = "absent from the reference"
            continue
        delta = real_means[index[p]] - ctrl
        if delta.size < 2:
            outside[p] = "fewer than 2 genes"
        elif not np.all(np.isfinite(delta)):
            outside[p] = "non-finite reference delta"
        elif np.all(delta == delta[0]):
            outside[p] = "constant reference delta"
        else:
            inside.add(p)
    return GateResult(frozenset(inside), outside)


def constant_delta_prediction(pred_bulk, *, real_bulk, control: str, control_source: str,
                              inside) -> frozenset[str]:
    """The perturbations INSIDE a ``delta_var`` gate whose PREDICTED delta is constant.

    Pearson is undefined for a constant input, and scipy does not reliably say so: for a
    two-gene input 1.14.1 returns 0.0 where 1.11.4 and 1.18.0 return NaN (measured), so the
    worst-value fill would not be reached. Exact equality, as for the Spearman metrics (spec
    §5.2); the near-constant case stays out of scope (§8). The delta is built as
    `metrics.delta._delta_eval` builds the prediction side: ``pred_means[p] - ctrl``, where
    ``ctrl`` is the PREDICTED control under ``control_source="pred"`` and the reference one under
    ``"real"``. A perturbation the prediction does not hold is left to the fill.
    """
    inside = frozenset(inside)
    if not inside:
        return frozenset()
    if control_source not in ("pred", "real"):
        raise ValueError(f"control_source must be 'pred' or 'real', got {control_source!r}")
    pred_perts = np.asarray(pred_bulk[0]).astype(str)
    pred_means = np.asarray(pred_bulk[1], dtype=np.float64)
    side_perts, side_means, side = ((pred_perts, pred_means, "pred") if control_source == "pred"
                                    else (np.asarray(real_bulk[0]).astype(str),
                                          np.asarray(real_bulk[1], dtype=np.float64), "real"))
    hits = np.flatnonzero(side_perts == control)
    if hits.size == 0:
        raise ValueError(f"control {control!r} not found in {side} perturbations")
    ctrl = side_means[hits[0]]
    index = {p: i for i, p in enumerate(pred_perts)}
    out: set[str] = set()
    for p in inside:
        if p not in index:
            continue
        delta = pred_means[index[p]] - ctrl
        if np.all(delta == delta[0]):
            out.add(p)
    return frozenset(out)


def outside_summary(result: GateResult) -> str:
    """'n x reason; ...' for the WARNING the dispatch logs, most frequent reason first."""
    tally: dict[str, int] = {}
    for reason in result.outside.values():
        tally[reason] = tally.get(reason, 0) + 1
    return "; ".join(f"{n} x {reason}" for reason, n in
                     sorted(tally.items(), key=lambda kv: (-kv[1], kv[0])))
