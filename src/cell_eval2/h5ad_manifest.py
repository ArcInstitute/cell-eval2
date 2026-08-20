"""Out-of-core scoring of manifest-indexed forward-eval predictions.

Reads the artifact directly (a manifest.csv indexing one dense .h5ad pair per
(dataset, panel, context)) and scores it WITHOUT materializing a whole context:
per context, build an out-of-core real reference then score pred perturbation-complete
batches via ``partition_inmem.score_piece``, combined by ``partition.aggregate_partials``.
Reports per-context metrics and an unweighted mean over contexts.
"""

from __future__ import annotations

import csv
import math
import os
import tempfile
from collections.abc import Iterator
from dataclasses import dataclass

import anndata as ad
import numpy as np
import polars as pl

from . import norm as _norm
from .config import EvalConfig
from .run import _close_backed


@dataclass(frozen=True)
class MemBudget:
    host_bytes: int
    gpu_bytes: int


@dataclass(frozen=True)
class Artifact:
    dataset: str
    panel_id: int
    context: str
    control_value: str
    path_real: str
    path_pred: str
    var_names_path: str
    n_cells: int
    n_genes: int
    real_abs: str
    pred_abs: str


def read_manifest(manifest) -> list[Artifact]:
    manifest = os.fspath(manifest)
    if os.path.isdir(manifest):
        manifest = os.path.join(manifest, "manifest.csv")
    if not os.path.isfile(manifest):
        raise FileNotFoundError(f"h5ad manifest not found: {manifest!r}")
    root = os.path.dirname(os.path.abspath(manifest))
    out: list[Artifact] = []
    with open(manifest, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            out.append(Artifact(
                dataset=str(row["dataset"]), panel_id=int(row["panel_id"]),
                context=str(row["context"]), control_value=str(row["control_value"]),
                path_real=str(row["path_real"]), path_pred=str(row["path_pred"]),
                var_names_path=str(row["var_names_path"]),
                n_cells=int(row["n_cells"]), n_genes=int(row["n_genes"]),
                real_abs=os.path.join(root, str(row["path_real"])),
                pred_abs=os.path.join(root, str(row["path_pred"])),
            ))
    if not out:
        raise ValueError(f"h5ad manifest {manifest!r} has no rows")
    return out


def _resolve_input_type_h5ad(real_abs, *, cfg, peek_rows: int = 2000) -> str:
    """Detect 'counts' vs 'lognorm' from a peek of the real h5ad, reusing v1's
    integer-vs-fractional test. The producer writes both sides in the same output_space, so
    the real side is authoritative."""
    backed = ad.read_h5ad(real_abs, backed="r")
    try:
        stop = min(int(backed.n_obs), peek_rows)
        peek = backed[0:stop].to_memory()
    finally:
        _close_backed(backed, real_abs)
    # autodetect=True forces the integer->counts / fractional->lognorm rule regardless of version.
    # allow_discrete=False, NOT cfg.allow_discrete: this answer feeds the pred/real equality guard in
    # score_h5ad_manifest, and `resolve_input_type` returns "counts" the moment that flag is set
    # without inspecting a value -- so honouring it would let a counts/lognorm pair resolve as
    # counts/counts and walk through the guard that exists to refuse it (codex-review round 2).
    return _norm.resolve_input_type(
        peek, declared=cfg.input_type, version=cfg.version,
        allow_discrete=False, autodetect=True,
    )


def _gib(n_bytes: float) -> str:
    return f"{float(n_bytes) / 2**30:.2f} GiB"


def plan_pert_batches(pert_sizes, *, n_genes, itemsize, control_cells, mem_budget,
                      safety=3.0) -> list[list[str]]:
    """Greedy pack consecutive perturbations into batches within the memory budget.
    A perturbation is never split; the resident control is charged to every batch.

    #178: the raise reports the footprint it just computed and a budget that would work.
    Callers thread this value by hand because of #146 -- too small fails to plan, too large
    blows cupy's 2 GiB pinned-allocation ceiling -- so every trial costs a GPU allocation, and
    the planner already holds the number that turns that search into a single read.
    """
    host_b, gpu_b = int(mem_budget.host_bytes), int(mem_budget.gpu_bytes)
    budget = min(host_b, gpu_b)
    # WHICH side binds decides which knob to raise. Reporting only the minimum sends a caller to
    # raise gpu_bytes when host_bytes is the short one, which changes nothing and costs a cycle.
    binding = "host_bytes" if host_b <= gpu_b else "gpu_bytes"
    per_cell = float(n_genes) * float(itemsize) * float(safety)
    ctrl_bytes = control_cells * per_cell
    cap_cells = (budget - ctrl_bytes) / per_cell if per_cell > 0 else 0
    batches, cur, cur_cells = [], [], 0
    for name, n in pert_sizes:
        if n > cap_cells:
            need = (int(control_cells) + int(n)) * per_cell
            headroom = budget - ctrl_bytes
            # "The control alone does not fit" is a different diagnosis from "this one
            # perturbation is too big", and the old message blamed the perturbation for both.
            # When headroom <= 0 EVERY perturbation raises here, and a caller who added this
            # perturbation's own footprint to the budget would still not have enough.
            lead = (
                f"the resident control pool alone ({control_cells} cells, "
                f"{_gib(ctrl_bytes)}) does not fit the memory budget"
                if headroom <= 0 else
                f"perturbation {name!r} ({n} cells, {_gib(n * per_cell)}) plus the resident "
                f"control ({control_cells} cells, {_gib(ctrl_bytes)}) exceeds the memory budget"
            )
            raise ValueError(
                f"{lead}: {binding}={_gib(budget)} is the binding limit "
                f"(host_bytes={_gib(host_b)}, gpu_bytes={_gib(gpu_b)}), leaving "
                f"{_gib(max(headroom, 0.0))} for perturbation cells at {n_genes} genes x "
                f"{itemsize} bytes x {safety} safety = {per_cell:.0f} B/cell. "
                # BOTH budgets, not just the binding one: the usable budget is min(host, gpu), so
                # raising only the current minimum fails again the moment the OTHER one is also
                # below `need` (codex-review). ceil, not int: truncating a fractional requirement
                # suggests a value a byte short of sufficient.
                f"Try host_bytes AND gpu_bytes >= {_gib(need)} ({math.ceil(need)} bytes) -- the "
                f"usable budget is min(host_bytes, gpu_bytes), so raising only {binding} fails "
                "again if the other is also short. "
                "See #146 for the UPPER bound: raising it until planning succeeds can walk "
                "into cupy's pinned-allocation ceiling on the largest control pools."
            )
        if cur and cur_cells + n > cap_cells:
            batches.append(cur)
            cur, cur_cells = [], 0
        cur.append(name)
        cur_cells += n
    if cur:
        batches.append(cur)
    return batches


def read_group_block(h5ad_abs, *, pert_col, labels) -> ad.AnnData:
    """In-memory AnnData of rows whose obs[pert_col] is in `labels`."""
    backed = ad.read_h5ad(h5ad_abs, backed="r")
    try:
        obs_labels = backed.obs[pert_col].to_numpy().astype(str)
        idx = np.flatnonzero(np.isin(obs_labels, list(map(str, labels))))
        if idx.size == 0:
            raise ValueError(f"no rows for labels {sorted(labels)!r} in {h5ad_abs!r}")
        return backed[idx].to_memory()
    finally:
        _close_backed(backed, h5ad_abs)


def iter_h5ad_pert_batches(h5ad_abs, *, pert_col, control,
                          mem_budget) -> Iterator[tuple[list[str], "ad.AnnData"]]:
    """Yield (batch_perts, batch_adata) of non-control perturbations, perturbation-complete,
    sized to `mem_budget`. Rows for these artifacts are perturbation-contiguous, but this
    reads by explicit membership so it is order-robust.

    The obs labels are read and grouped into per-perturbation row indices ONCE (a single
    O(N log N) argsort), and the backed file is held open across the whole iteration; each
    batch then slices by its pre-computed indices. This avoids re-opening the file and
    re-parsing the full obs column for every batch (the previous per-batch read_group_block
    call was O(batches * N) disk I/O + string parsing -- a bottleneck at scale)."""
    backed = ad.read_h5ad(h5ad_abs, backed="r")
    try:
        obs_labels = backed.obs[pert_col].to_numpy().astype(str)
        n_genes = int(backed.n_vars)
        itemsize = np.dtype(getattr(backed.X, "dtype", np.float32)).itemsize
        # One O(N log N) pass: sort once, then each label's rows are a contiguous run of the
        # sort order -> per-pert counts (for planning) AND per-pert row indices (for slicing)
        # without an O(N*M) per-label scan. Stable sort keeps each label's rows in ascending
        # original-row order.
        order = np.argsort(obs_labels, kind="stable")
        uniq, first, counts = np.unique(obs_labels[order], return_index=True, return_counts=True)
        idx_by_label = {u: order[first[k]:first[k] + counts[k]] for k, u in enumerate(uniq)}
        ctrl = str(control)
        control_cells = int(dict(zip(uniq, counts)).get(ctrl, 0))
        # np.unique returns ascending-sorted uniq; keep non-control perts in that order.
        sizes = [(u, int(counts[k])) for k, u in enumerate(uniq) if u != ctrl]
        batches = plan_pert_batches(
            sizes, n_genes=n_genes, itemsize=itemsize, control_cells=control_cells,
            mem_budget=mem_budget)
        for batch_perts in batches:
            rows = np.concatenate([idx_by_label[p] for p in batch_perts])
            rows.sort()   # ascending row order -> efficient backed read (rows are contiguous)
            yield batch_perts, backed[rows].to_memory()
    finally:
        _close_backed(backed, h5ad_abs)


class H5adBatchSource:
    """``PertBatchSource`` over a manifest ``.h5ad`` side -- wraps ``read_group_block`` and
    ``iter_h5ad_pert_batches`` so the streaming reference-builders are format-agnostic. The
    row-store analogue is ``rowstore.RowStoreBatchSource``; both satisfy the
    ``partition_inmem.PertBatchSource`` contract (``control`` / ``stream_tag`` /
    ``read_control_block`` / ``iter_pert_batches``)."""

    def __init__(self, h5ad_path, *, pert_col, control):
        self.h5ad_path = os.fspath(h5ad_path)
        self.pert_col = pert_col
        self.control = str(control)
        self.stream_tag = self.h5ad_path        # fingerprint tag (byte-identical to the old path)

    def read_control_block(self) -> "ad.AnnData":
        return read_group_block(self.h5ad_path, pert_col=self.pert_col, labels={self.control})

    def iter_pert_batches(self, mem_budget):
        return iter_h5ad_pert_batches(
            self.h5ad_path, pert_col=self.pert_col, control=self.control, mem_budget=mem_budget)


@dataclass
class ScoreResult:
    # per_pert: one row per (dataset, panel_id, context, perturbation, metric); the scoring
    # unit is the full (dataset, panel_id, context) triple, NOT context alone -- a manifest can
    # reuse the same context string across different panels/datasets, and dataset+panel_id keep
    # those rows distinct instead of silently pooling them.
    per_pert: pl.DataFrame
    # per_context: one row per (dataset, panel_id, context, metric) = MetricSpec.agg over
    # perturbations within that (dataset, panel_id, context) unit (NaN-skip; matches
    # run.aggregate_metrics, which #233 found this claim had stopped being true of). Carries an
    # `agg` column (appended last) naming the statistic that produced each row.
    per_context: pl.DataFrame
    # overall: one row per metric = unweighted mean over every (dataset, panel_id, context) unit
    # in per_context (NaN-skip). ALWAYS a mean, whatever the metric's per-perturbation statistic
    # -- see _assemble_score_result for why, and why it therefore carries no `agg` column.
    overall: pl.DataFrame


def _nsig_names(cfg):
    """The emitted nsig metric names for ``cfg.version`` (v1 uses the v1_name aliases).
    Under v1 the partials carry ``de_nsig_counts_{real,pred}`` / ``de_spearman_sig``, not the
    v2 ``de_wilcoxon_*`` names, so the nsig_spearman reduction must key on these."""
    from .catalog import CATALOG

    def out(key):
        spec = CATALOG[key]
        return spec.v1_name if (cfg.version == "v1" and spec.v1_name) else spec.name

    return (out("de_wilcoxon_nsig_spearman"),
            out("de_wilcoxon_nsig_counts_real"),
            out("de_wilcoxon_nsig_counts_pred"))


def _assemble_score_result(per_pert_frames) -> ScoreResult:
    """Concat per-pert frames -> per-context (``MetricSpec.agg`` over perts, NaN-skip) ->
    overall (unweighted mean over contexts). Shared by ``score_h5ad_manifest`` and
    ``rowstore.score_rowstore``. The scoring unit is the full (dataset, panel_id, context)
    triple -- a manifest can reuse a context string across panels/datasets, so all three columns
    key the aggregation. ``fill_null(nan)``: ``drop_nans().mean()`` on an all-NaN group returns
    null in Polars; fill_null makes an all-NaN metric aggregate to NaN (not null), matching
    ``run.aggregate_metrics``.

    **#233.** ``per_context`` used an unconditional mean and so violated the generic
    ``MetricSpec.agg`` contract that ``run.aggregate_metrics`` and ``run.aggregate_metrics_wide``
    both honour, while two docstrings here claimed it "matches run.aggregate_metrics". No SHIPPED
    metric declares ``median`` any more (#231), so this moves NO number today -- the issue's
    headline "18 median metrics are averaged" is stale -- but the statistic is meant to stay a
    catalog edit rather than a source edit, and this path is the whole manifest-artifact
    integration surface.
    The shape below deliberately mirrors ``aggregate_metrics``: compute both statistics in one
    pass and select with a ``when/otherwise`` on an ``agg`` column, so the two implementations
    stay recognizably the same.

    **The two decisions #233 says it needs.**

    1. ``overall`` keeps the unconditional MEAN, and only ``per_context`` resolves the statistic.
       ``MetricSpec.agg`` describes how to reduce the heavy-tailed PER-PERTURBATION population
       -- a median is there to stop one runaway perturbation deciding the metric. ``overall``
       reduces over CONTEXTS, which are units of a designed comparison and few in number, and
       ``run.py`` has no equivalent level, so there is no precedent to copy and no tail to
       defend against. Reducing contexts by median would additionally make the overall value
       depend on how a manifest happened to be partitioned.
    2. ``per_context`` GAINS an ``agg`` column, so the artifact records which statistic produced
       each row instead of implying it. It is a tidy frame, so unlike ``run.py``'s strictly
       numeric wide frame it can carry a string column and needs no separate sidecar --
       ``score_h5ad_manifest``'s ``per_context.csv`` is self-describing. APPENDED last, never
       inserted, for the same reason ``_WIDE_STATISTICS`` appends ``median``: a released
       downstream orchestrator consumes these frames and a positional reader must keep
       working. ``overall``
       deliberately does NOT get the column: its value is always a context mean, so an ``agg``
       there naming the perturbation-level statistic would read as a claim about the row.

    A ``ratio_of_sums`` (derived) metric cannot be reduced from per-perturbation values at all,
    and ``_reject_derived_rows`` is called so such a row raises here rather than being silently
    meaned -- which is the same class of defect #233 reports. Unreachable today: the three
    partition-backed drivers never emit the derived member (#270).
    """
    from .run import _reject_derived_rows, metric_agg

    per_pert = pl.concat(per_pert_frames, how="vertical").select(
        ["dataset", "panel_id", "context", "perturbation", "metric", "value"])
    _reject_derived_rows(per_pert)
    both = (per_pert.group_by(["dataset", "panel_id", "context", "metric"])
            .agg(_mean=pl.col("value").drop_nans().mean().fill_null(float("nan")),
                 _median=pl.col("value").drop_nans().median().fill_null(float("nan"))))
    # Resolve `agg` by joining a per-UNIQUE-metric lookup rather than calling `metric_agg` once per
    # ROW (Gemini, PR #307 round 2). Measured 24.4 ms -> 7.9 ms, **3.1x**, on 200k rows, producing a
    # byte-identical frame. Note this is a different proposal from round 1's `.map_elements()`,
    # which measured 1.58x SLOWER and which polars itself warns against -- the win here is the
    # vectorized join, not avoiding the round-trip.
    #
    # `how="left"` (never "inner"): the map is built from this frame's own uniques so nothing can
    # miss, and a left join cannot silently DROP a context if that ever stops being true.
    # `maintain_order="left"` is passed even though the join preserved order without it in
    # testing -- `per_context` is consumed by a released orchestrator, so row order is a guarantee to
    # specify, not a behaviour to observe.
    uniq = both["metric"].unique().to_list()
    agg_map = pl.DataFrame({"metric": uniq, "agg": [metric_agg(m) for m in uniq]},
                           schema={"metric": pl.String, "agg": pl.String})
    per_context = (
        both.join(agg_map, on="metric", how="left", maintain_order="left")
        .with_columns(value=pl.when(pl.col("agg") == "median")
                      .then(pl.col("_median")).otherwise(pl.col("_mean")))
        .select(["dataset", "panel_id", "context", "metric", "value", "agg"])
    )
    overall = (per_context.group_by("metric")
               .agg(pl.col("value").drop_nans().mean().fill_null(float("nan")).alias("value")))
    return ScoreResult(per_pert=per_pert, per_context=per_context, overall=overall)


def score_h5ad_manifest(manifest, *, config=None, mem_budget, outdir=None) -> ScoreResult:
    """Score every (dataset, panel, context) artifact in an h5ad manifest out-of-core.

    Per (dataset, panel_id, context) unit: build an out-of-core real reference
    (``build_reference_streaming``), then score perturbation-complete pred batches
    (``iter_h5ad_pert_batches`` + ``score_piece``) into per-unit partials, combined with
    ``partition.aggregate_partials``. Ties Tasks 1-5 together; returns per-perturbation,
    per-context (mean over perturbations within a (dataset, panel_id, context) unit), and
    overall (unweighted mean over all units) tidy Polars frames. If ``outdir`` is given,
    each frame is also written as a CSV there.

    If ``config`` is ``None``, defaults to ``EvalConfig.v2()`` with ``target_sum=1e4``
    (the artifact format's own convention, matching the CLI's ``--target-sum`` default) rather than v2's
    usual 1e6; a caller-supplied ``config`` is used unchanged.
    """
    from dataclasses import replace as _replace

    from .partition import aggregate_partials
    from .partition_inmem import (_RefBundle, build_pred_control_reference,
                                  build_reference_streaming, score_piece)

    cfg = config if config is not None else _replace(EvalConfig.v2(), target_sum=1e4)
    # These h5ads use obs["perturbation"]; the default config's pert_col is "target",
    # which would find no perturbations. Default it to that column (via replace, not
    # mutation, so a caller-supplied config object is never modified in place). Doing this
    # once here means build_reference_streaming, score_piece, and iter_h5ad_pert_batches all
    # inherit the corrected column below.
    if cfg.pert_col == "target":
        cfg = _replace(cfg, pert_col="perturbation")

    arts = read_manifest(manifest)
    per_pert_frames = []
    for art in arts:
        itype = _resolve_input_type_h5ad(art.real_abs, cfg=cfg)
        # `itype` is peeked from the REAL h5ad and then used for both sides, including the
        # pred builder and piece_cfg. That predates #264 and rests on the producer writing both
        # sides in one output_space (see `_resolve_input_type_h5ad`) -- but #264 makes it feed
        # the COMPARATOR too, where a violation would silently put real and pred in different
        # expression spaces rather than merely mislabelling the pred input. So the invariant
        # is now ASSERTED instead of assumed: peek the pred side and fail loudly if the two
        # disagree. Resolving them separately was considered and rejected -- it would
        # legitimize a divergent pair the producer does not write, and the two bulks would then
        # be incomparable whatever comparator was chosen (Copilot, #265).
        pred_itype = _resolve_input_type_h5ad(art.pred_abs, cfg=cfg)
        if pred_itype != itype:
            raise ValueError(
                f"manifest artifact {art.pred_abs!r} peeks as {pred_itype!r} but its real "
                f"side {art.real_abs!r} peeks as {itype!r}. The producer writes both sides "
                "in one "
                "output_space, so this pair is malformed: scoring it would compare bulks "
                "computed in different expression spaces. Re-export the artifact."
            )
        comparator = _norm.resolve_comparator(
            version=cfg.version, pred_input_type=pred_itype, real_input_type=itype,
        )
        with tempfile.TemporaryDirectory() as ref_dir, tempfile.TemporaryDirectory() as parts_dir:
            build_reference_streaming(
                art.real_abs, config=cfg, cache_dir=ref_dir,
                control=art.control_value, mem_budget=mem_budget, input_type=itype,
                comparator=comparator)
            if cfg.control_source == "pred":
                # pred side scored within-realm: build the pred-control reference (pred control
                # cells as the gpudge external-ref pool + pred-control pseudobulk per norm) that
                # score_piece reads under control_source='pred'. Real side already built above.
                build_pred_control_reference(
                    art.pred_abs, config=cfg, cache_dir=ref_dir,
                    control=art.control_value, input_type=itype, comparator=comparator)
            k = 0
            piece_cfg = _replace(cfg, control=art.control_value, input_type=itype)
            bundle = _RefBundle(ref_dir, piece_cfg)   # one per artifact; ref_dir is stable (#153)
            try:
                for batch_perts, batch_ad in iter_h5ad_pert_batches(
                        art.pred_abs, pert_col=cfg.pert_col, control=art.control_value,
                        mem_budget=mem_budget):
                    score_piece(batch_ad, ref_dir, config=piece_cfg,
                                piece_id=f"{art.context}_{k}", partial_out=parts_dir,
                                bundle=bundle, comparator=comparator)
                    k += 1
            finally:
                # Drop this artifact's cached control pool / DE table before aggregation AND
                # before the NEXT artifact's reference build. A Python local outlives the `with`
                # block it was assigned in, so without this two contexts' control pools are
                # resident at once -- a real OOM risk at scale (#153).
                del bundle
            nsig_metric, nsig_real, nsig_pred = _nsig_names(cfg)
            full, _agg = aggregate_partials(
                parts_dir, reduce_nsig_spearman=True,
                nsig_spearman_metric=nsig_metric,
                nsig_real_metric=nsig_real, nsig_pred_metric=nsig_pred)
        per_pert_frames.append(full.with_columns(
            pl.lit(art.dataset).alias("dataset"),
            pl.lit(art.panel_id).alias("panel_id"),
            pl.lit(art.context).alias("context"),
        ))

    # The manifest's unique scoring unit is (dataset, panel_id, context); _assemble_score_result keys
    # the per-context/overall aggregation on all three (not context alone) -- a manifest can span
    # datasets/panels that share a context string.
    res = _assemble_score_result(per_pert_frames)
    if outdir is not None:
        os.makedirs(outdir, exist_ok=True)
        res.per_pert.write_csv(os.path.join(outdir, "per_pert.csv"))
        res.per_context.write_csv(os.path.join(outdir, "per_context.csv"))
        res.overall.write_csv(os.path.join(outdir, "overall.csv"))
    return res
