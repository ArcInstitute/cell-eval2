"""Transitional cell-eval-compatible API. Deprecated; migrate to cell_eval2 native API."""

from __future__ import annotations

import logging
import os
import warnings
from dataclasses import replace

import anndata as ad
import numpy as np
import polars as pl

from ..catalog import CATALOG, _NAME_TO_CANONICAL, resolve_metrics
from ..config import EvalConfig
from ..run import compute_metrics

logger = logging.getLogger(__name__)


def _to_wide(tidy: pl.DataFrame) -> pl.DataFrame:
    if tidy.is_empty():
        return pl.DataFrame({"perturbation": []})
    return tidy.pivot(index="perturbation", on="metric", values="value")


class MetricsEvaluator:
    """Drop-in shim for cell_eval.MetricsEvaluator (deprecated).

    ``target_gene_map`` and ``exclude_target_gene`` are cell_eval2 additions rather than part of
    the upstream signature (#252). They exist because #248 left the shim with no correct answer
    for a guide-level panel: since #248 ``pds_*`` RAISES when exclusion is on and no perturbation
    label resolves to a measured gene -- the construct-ID-vs-symbol mismatch -- and before it, such
    a run silently excluded nothing and returned a wrong PDS. The native ``EvalConfig`` path has
    had both knobs all along; only the shim had neither, so a guide-level run through it could do
    neither thing correctly. ``metric_configs``, its only other plausible configuration channel,
    is accepted and dropped (see ``compute``).

    Both default to ``None`` = exactly the previous behaviour, so v1 byte-parity for gene-level
    panels cannot move (the parity gate covers that).
    """

    def __init__(
        self,
        adata_pred: ad.AnnData | str,
        adata_real: ad.AnnData | str,
        de_pred=None,
        de_real=None,
        control_pert: str = "non-targeting",
        pert_col: str = "target",
        num_threads: int = -1,
        outdir: str = "./cell-eval2-outdir",
        allow_discrete: bool = False,
        prefix: str | None = None,
        pdex_kwargs: dict | None = None,
        skip_de: bool = False,
        target_gene_map: dict[str, str] | None = None,
        exclude_target_gene: bool | None = None,
    ) -> None:
        warnings.warn(
            "cell_eval2.compat.MetricsEvaluator is transitional; "
            "migrate to cell_eval2.compute_metrics.",
            DeprecationWarning,
            stacklevel=2,
        )
        self.adata_pred = adata_pred
        self.adata_real = adata_real
        self.de_pred = de_pred
        self.de_real = de_real
        self.pdex_kwargs = pdex_kwargs
        self.skip_de = skip_de
        self.control_pert = control_pert
        self.pert_col = pert_col
        self.num_threads = num_threads
        self.outdir = outdir
        self.allow_discrete = allow_discrete
        self.prefix = prefix
        self.target_gene_map = target_gene_map
        self.exclude_target_gene = exclude_target_gene

    def compute(
        self,
        profile: str = "full",
        metric_configs: dict | None = None,
        skip_metrics: list[str] | None = None,
        basename: str = "results.csv",
        write_csv: bool = True,
        break_on_error: bool = False,
    ) -> tuple[pl.DataFrame, pl.DataFrame]:
        if metric_configs:
            logger.warning(
                "metric_configs are ignored in this unit (per-metric config not yet wired): %s. "
                "The two settings a guide-level panel actually needs are constructor arguments "
                "instead: MetricsEvaluator(..., target_gene_map=..., exclude_target_gene=...) "
                "(#252).",
                metric_configs,
            )
        if break_on_error:
            logger.warning("break_on_error is ignored in this unit")
        input_type = "counts" if self.allow_discrete else "lognorm"
        # Force the v1 config (cell-eval / VCC bit-parity:
        # PDS l1, rank denominator n, predicted control). Native callers get the
        # corrected defaults; the compat layer never does.
        cfg = replace(
            EvalConfig.v1(),
            metrics=profile,
            pert_col=self.pert_col,
            control=self.control_pert,
            input_type=input_type,
            allow_discrete=self.allow_discrete,
            num_threads=self.num_threads,
        )
        # #252: both default to None = today's config exactly, so a gene-level v1 run is
        # byte-identical to before. `exclude_target_gene` lives on the nested discrimination
        # params, `target_gene_map` at the top level -- and BOTH are needed, not either: the map
        # is what makes guide labels resolve, and the flag is the deliberate opt-out for a panel
        # whose labels cannot resolve at all.
        if self.target_gene_map is not None:
            cfg = replace(cfg, target_gene_map=dict(self.target_gene_map))
        if self.exclude_target_gene is not None:
            cfg = replace(cfg, discrimination=replace(
                cfg.discrimination, exclude_target_gene=bool(self.exclude_target_gene)))
        de_over = {"backend": "pdex"}
        if self.pdex_kwargs:
            unmapped = {}
            for k, v in self.pdex_kwargs.items():
                if k == "geometric_mean":
                    de_over["mean_calc"] = "geometric" if v else "arithmetic"
                elif k == "epsilon":
                    de_over["epsilon"] = v
                elif k in ("is_log1p", "threads"):
                    pass  # is_log1p derives from input_type; threads from num_threads
                else:
                    unmapped[k] = v
            if unmapped:
                logger.warning("pdex_kwargs not mapped to a backend param: %s", unmapped)
        cfg = replace(cfg, de=replace(cfg.de, **de_over))
        names, _ = resolve_metrics(profile, version=cfg.version)
        if self.skip_de:
            names = [n for n in names if CATALOG[n].kind != "de"]
        cfg = replace(cfg, metrics=names)
        tidy = compute_metrics(
            self.adata_pred, self.adata_real, config=cfg,
            de_pred=self.de_pred, de_real=self.de_real,
        )
        if skip_metrics:
            # compat emits v1 labels; resolve any spelling (v1/v2/alias) to its v1 output
            # label so skip_metrics works regardless of which name the caller passed.
            skip_labels = set()
            for m in skip_metrics:
                spec = CATALOG.get(_NAME_TO_CANONICAL.get(m, m))
                skip_labels.add((spec.v1_name or spec.name) if spec is not None else m)
            tidy = tidy.filter(~pl.col("metric").is_in(list(skip_labels)))

        results = _to_wide(tidy)
        # describe() raises on a zero-column frame; when every metric is skipped the wide
        # table is just the empty "perturbation" column. Emit a describe-shaped agg with no
        # metric columns (the standard statistic-label rows, derived version-robustly) so
        # downstream code that filters on `statistic` gets the expected rows, not 0 rows.
        body = results.drop("perturbation") if "perturbation" in results.columns else results
        agg = body.describe() if body.width else pl.DataFrame({"_": [0.0]}).describe().select("statistic")

        if write_csv:
            os.makedirs(self.outdir, exist_ok=True)
            pfx = f"{self.prefix.replace('/', '-')}_" if self.prefix else ""
            base = basename.replace("/", "-")
            results.write_csv(os.path.join(self.outdir, f"{pfx}{base}"))
            agg.write_csv(os.path.join(self.outdir, f"{pfx}agg_{base}"))

        return results, agg


def _norm_by_zero(user: float, base: float) -> float:
    if base == 0:                 # degenerate baseline; revives the isfinite guard below
        return float("nan")
    return 1.0 - (user / base)


def _norm_by_one(user: float, base: float) -> float:
    if base == 1:                 # degenerate baseline; revives the isfinite guard below
        return float("nan")
    return (user - base) / (1.0 - base)


def score_agg_metrics(
    results_user: pl.DataFrame | str,
    results_base: pl.DataFrame | str,
    output: str | None = None,
    comparison_statistic: str = "mean",
) -> pl.DataFrame:
    """Score user aggregate metrics against a baseline (cell-eval-compatible)."""
    if isinstance(results_user, str):
        results_user = pl.read_csv(results_user)
    if isinstance(results_base, str):
        results_base = pl.read_csv(results_base)
    if results_user.columns != results_base.columns:
        raise ValueError("user/base columns do not match")
    if "statistic" not in results_user.columns:
        raise ValueError("missing 'statistic' column in agg results")
    available = results_user["statistic"].to_list()
    if comparison_statistic not in available:
        raise ValueError(
            f"comparison_statistic {comparison_statistic!r} not found in agg results; "
            f"available: {available}"
        )
    base_available = results_base["statistic"].to_list()
    if comparison_statistic not in base_available:
        raise ValueError(
            f"comparison_statistic {comparison_statistic!r} not found in baseline agg results; "
            f"available: {base_available}"
        )

    u_row = results_user.filter(pl.col("statistic") == comparison_statistic).drop("statistic")
    b_row = results_base.filter(pl.col("statistic") == comparison_statistic).drop("statistic")
    metric_names = u_row.columns
    u_vals = u_row.row(0)
    b_vals = b_row.row(0)

    # Reference cell-eval (_score.py) emits all norm-by-zero metrics, then all
    # norm-by-one, then avg_score; reproduce that row order. Per-metric values and
    # the avg (order-invariant mean) are unchanged.
    metrics_zero, scores_zero = [], []
    metrics_one, scores_one = [], []
    for name, uv, bv in zip(metric_names, u_vals, b_vals):
        spec = CATALOG.get(_NAME_TO_CANONICAL.get(name, name))
        # Reads the POLICY, never the derived ``best_value`` token. Freezing this FILE never
        # froze this FUNCTION: ``best_value`` is derived from a MUTABLE catalog, so enrolling a
        # metric in the native scorer silently made it scorable here, with a zero-line diff.
        # Measured: a v2-shaped aggregate moved avg_score 0.5 -> 1.0068 that way.
        #
        # ``v1_available`` is part of the predicate on purpose, and it is the half that closes
        # the hole. This scorer reproduces upstream cell-eval, so it must score the SCORED
        # SUBSET of what a v1 run can emit (27 of the 29 columns -- the two nsig_counts
        # diagnostics are emitted but never scored) -- and it only knows two normalizations, anchor-0 and
        # anchor-1. An anchorless metric has NEITHER; scoring one here would apply
        # ``_norm_by_one`` to a quantity with no anchor at 1 and skip its clamp, i.e. return a
        # confidently wrong number rather than decline. Native ``score_metrics`` owns those.
        if spec is None or not spec.v1_available or not spec.scoring.scored:
            logger.warning("metric %r not scored (unknown, v2-native, or scored=False)", name)
            continue
        if spec.scoring.direction == "lower":
            score = _norm_by_zero(uv, bv) if (uv is not None and bv is not None) else float("nan")
            bucket_m, bucket_s = metrics_zero, scores_zero
        else:
            score = _norm_by_one(uv, bv) if (uv is not None and bv is not None) else float("nan")
            bucket_m, bucket_s = metrics_one, scores_one
        if not np.isfinite(score):
            score = 0.0
        bucket_m.append(name)
        bucket_s.append(max(0.0, score))

    metrics = metrics_zero + metrics_one
    scores = scores_zero + scores_one
    avg = float(np.mean(scores)) if scores else 0.0
    out = pl.DataFrame({"metric": metrics + ["avg_score"], "from_baseline": scores + [avg]})
    if output is not None:
        out.write_csv(output)
    return out
