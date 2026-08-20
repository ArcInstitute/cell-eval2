"""Replicate anchor: the 1.0 end of #276's 0=baseline/1=replicate scale.

A RAW split-half replicate, measured on the real data alone. Deliberately NOT
Spearman-Brown corrected and not depth-corrected in any other way -- the correction
`ceiling.py` applies is a bounded-reliability extrapolation that does not hold for five of
the six scored `vcc2026` members, and as of 2026-08-12 no depth correction is applied
anywhere in this scheme.

ONE shared split-and-score core, so the split semantics never get a second
implementation. `control_source` is forced to "pred" inside that core and is not
optional: under "real" the pred side's DE reference comes from the real side, so scoring
half_b against half_a computes BOTH halves' log2FCs against half_a's control. That shares
the control's sampling noise between the two quantities whose agreement is being measured
and biases the anchor upward -- and under this scheme an upward-biased anchor biases every
submission's score DOWNWARD, uniformly, with nothing in the output that looks wrong.
Measured for de_lfc_nmae on a real panel: shared controls read 0.5-2.3% optimistic,
consistently, on all 12 comparisons. `ceiling.py` measured the same mistake turning a 0.54
split-half reliability into 0.74.
"""
from __future__ import annotations

import json
import logging
import math
import os
from dataclasses import dataclass, replace
from importlib.metadata import PackageNotFoundError, version as _pkg_version

import anndata as ad
import numpy as np
import polars as pl

from .cache import _hash_obj, config_hash, fingerprint_adata
from .catalog import _LFC_NMAE_METRICS, _NAME_TO_CANONICAL, resolve_metrics
from .ceiling import _disjoint_halves
from .config import EvalConfig
from .io import load_anndata
from .lfc_nmae_ref import _assert_disjoint_controls, compute_lfc_nmae_reference
from .run import (
    _cache_backend,
    _cache_device,
    _compute_de_side,
    _resolve_config,
    _resolve_target_sum_from_control,
    aggregate_metrics,
    compute_metrics,
    metric_output_names,
)

logger = logging.getLogger(__name__)

#: The producer that made a given metric's anchor. Stamped per metric in the artifact so
#: "which estimator, raw or otherwise" is a property of the FILE rather than of the
#: invocation -- the same reason the anchor source is stamped rather than flagged.
SPLIT_HALF_RAW = "split_half_raw"

#: `de_lfc_nmae`'s anchor comes from a DIFFERENT estimator, and this is not a
#: refinement -- it is a correctness requirement. `de_lfc_nmae` omits a perturbation whose
#: real-side significance gate holds fewer than `min_gate_size` genes. A HALF calls far
#: fewer genes significant than the full data (measured: the median gate is 1.83-3.36x
#: smaller), so the uniform core averages over 65-79% of the perturbations the MEMBER
#: averages over. `compute_lfc_nmae_reference` takes its gate and denominator from the FULL
#: real table and only the numerator's two vectors from the halves, so its cohort equals
#: the member's EXACTLY (verified on six real cell lines). Anchoring a metric on a
#: different population than it is measured on is the mismatch `score` cannot detect --
#: it sees only the two scalars.
FULL_GATE_RAW = "full_gate_raw"

#: How the five split seeds come off the ONE base seed the preset pins. Named and stamped
#: rather than left implicit: an inline `base_seed + i` is the kind of thing a later
#: refactor changes without anything noticing, and it would silently move every shipped
#: anchor. The DERIVED seeds are stamped literally beside this string, so the artifact is
#: self-contained even if this rule is later replaced.
SEED_DERIVATION = "numpy.random.SeedSequence(base_seed).generate_state(n_splits)"

_ANCHOR_SCHEMA = {
    "metric": pl.Utf8, "replicate": pl.Float64, "replicate_sd": pl.Float64,
    "replicate_min": pl.Float64, "replicate_max": pl.Float64,
    "n_perturbations_min": pl.Int64, "n_perturbations_max": pl.Int64,
    "estimator": pl.Utf8,
}
_SPLITS_SCHEMA = {
    "split_index": pl.Int64, "seed": pl.Int64, "metric": pl.Utf8,
    "value": pl.Float64, "n_perturbations": pl.Int64,
}


def _inner_config(cfg: EvalConfig) -> EvalConfig:
    """The config ONE split actually runs under.

    Four overrides, carrying ceiling.py's rationale. `control_source="pred"` is required for
    the estimate to be valid at all (module docstring): under a shared control both halves'
    log2FCs are computed against the same cells, correlating the two quantities whose
    agreement is being measured -- measured optimistic by 0.5-2.3% on `lfc_nmae`, in the same
    direction on all 12 comparisons. The other three stop this inner run from writing over
    artifacts that belong to the CALLER's run: `compute_metrics` writes `run_params.yaml`
    into `cfg.outdir` unconditionally, and `CacheStore.put` would overwrite the caller's
    manifest with half-data artifacts and `os.remove` its full-data files.

    A function rather than an inline `replace` so `build_meta` can STAMP what it returns
    instead of restating it. A literal stamp survives the deletion of the thing it describes.
    """
    return replace(cfg, control_source="pred", cache_real=None, cache_pred=None, outdir=None)


def _score_one_split(
    real_ad: ad.AnnData, cfg: EvalConfig, seed: int, metrics: list[str]
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Score half_b against half_a for ONE seed. Returns (agg, counts).

    ``agg`` is `aggregate_metrics`' frame (metric, mean, agg). ``counts`` is
    (metric, n_perturbations) over the rows that produced a usable value -- the
    POST-GATE cohort, which is the count worth recording: `_disjoint_halves` drops a
    group iff ``n // 2 < 1``, and ``perm.size`` does not depend on the permutation, so
    ITS drop set is seed-invariant and stamping it would be a constant.
    """
    half_a, half_b = _disjoint_halves(real_ad, cfg.pert_col, cfg.control, seed)
    _assert_disjoint_controls(half_a, half_b, pert_col=cfg.pert_col, control=cfg.control)
    inner = _inner_config(cfg)
    tidy = compute_metrics(half_b, half_a, config=inner)
    agg = aggregate_metrics(tidy, metrics=metrics)
    counts = (
        # Projected before the filter/group_by (Gemini, PR #284): `tidy` is the
        # per-perturbation frame and carries columns this count does not read.
        tidy.select(["metric", "value"])
        .filter(pl.col("value").is_not_null() & pl.col("value").is_not_nan())
        .group_by("metric")
        .len()
        .rename({"len": "n_perturbations"})
        .with_columns(pl.col("n_perturbations").cast(pl.Int64))
    )
    return agg, counts


def _derive_seeds(base_seed: int, n_splits: int) -> list[int]:
    """The five split seeds from the one pinned base seed. See SEED_DERIVATION."""
    if n_splits < 1:
        raise ValueError(f"n_splits must be >= 1, got {n_splits!r}")
    return [int(s) for s in
            np.random.SeedSequence(int(base_seed)).generate_state(n_splits)]


def _lfc_nmae_names(names: list[str]) -> list[str]:
    """OUTPUT names whose CANONICAL identity is an lfc_nmae member.

    Fed `metric_output_names(cfg)`, never the resolved canonical list: under
    `de.backend="deseq2"` the emitted name is `de_deseq2_lfc_nmae` while the resolved name
    is still `de_wilcoxon_lfc_nmae`, so a canonical-derived list would write the substituted
    value under a key no frame carries and leave the emitted row on the uniform core.
    """
    return [n for n in names if _NAME_TO_CANONICAL.get(n, n) in _LFC_NMAE_METRICS]


def _lfc_nmae_raw(real_ad, cfg, seed, de_full) -> tuple[float, int]:
    """`nmae_ref_raw` at one seed. RAW, never the sqrt(2)-corrected column -- that is a depth
    correction this scheme does not apply anywhere. Substituting it would move every
    submission's score on this member by 17-23% (measured across six real cell lines).

    An empty reference RAISES here even though `score._from_reference_column` treats the
    same shape as a survivable data outcome. The difference is what the number is FOR: a
    null `from_reference` leaves one diagnostic column blank, whereas a null anchor leaves
    the competition scale with no top end for a scored member.
    """
    _res, ref = compute_lfc_nmae_reference(real_ad, config=cfg, seed=seed, de_real=de_full)
    row = ref.filter(pl.col("statistic") == "mean")
    raw, n = row["nmae_ref_raw"][0], row["n_perturbations"][0]
    if raw is None or not math.isfinite(float(raw)):
        raise ValueError(
            f"the lfc_nmae reference scored no perturbation at seed {seed} "
            f"(n_perturbations={n!r}), so this dataset has no replicate anchor for a "
            "SCORED member. Usual cause: too few genes clear the real-side significance "
            "gate at min_gate_size. Refusing rather than emitting an anchor that cannot "
            "be divided by."
        )
    return float(raw), int(n)


def compute_replicate_anchor(
    real: ad.AnnData | str | os.PathLike,
    *,
    config: EvalConfig | None = None,
    base_seed: int = 0,
    n_splits: int = 5,
    **overrides,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Estimate a per-metric replicate anchor from the real data alone.

    Returns ``(splits, anchor)``. ``splits`` is the long per-(split, metric) frame;
    ``anchor`` is one row per metric carrying the mean of the five per-split AGGREGATES.

    Mean of the AGGREGATES, deliberately, not a pool of per-perturbation rows: the
    derived `expr_mse_unbiased_capped_norm` is ``agg="ratio_of_sums"`` with no
    per-perturbation column at all, so pooling has nothing to pool; and the five splits
    are re-splits of the SAME cells, so pooling would treat them as five times the data
    and understate the spread -- the opposite of why five splits are taken.
    """
    cfg = _resolve_config(config, overrides)
    # REFUSED, not silently handled. `_score_one_split` hands half_b to `compute_metrics` as
    # the PRED side. Under `autodetect_input_type` (pred side only, run.py:149) or under v1
    # (which auto-detects BOTH sides), a half can classify counts-vs-lognorm independently of
    # the full matrix -- so the comparator the artifact stamps need not be the comparator the
    # splits ran under, and a comparator decides which normalization every expr_* member is
    # computed in. Neither config is a competition config (`vcc2026` is v2 without
    # autodetect), so refusing costs nothing and removes a whole class of silent mismatch.
    if cfg.autodetect_input_type or cfg.version == "v1":
        raise ValueError(
            "the replicate anchor does not support "
            f"{'autodetect_input_type' if cfg.autodetect_input_type else 'version=v1'}: "
            "each split hands a HALF to the pred side, which re-types independently there, "
            "so the effective input type -- and therefore the comparator every expr_* "
            "member is computed in -- would not be a property of the dataset. Declare "
            "input_type explicitly under v2."
        )
    real_ad = load_anndata(real, backed=False)  # need it in memory to split
    available, _missing = resolve_metrics(cfg.metrics, version=cfg.version)
    metrics = list(available)
    # The EXPECTED OUTPUT names, per the run's own rule -- NOT the resolved canonical list.
    # Under de.backend="deseq2" `_effective_de_spec` relabels de_wilcoxon_* to de_deseq2_*,
    # so the two lists disagree exactly where the lfc_nmae substitution has to land (Task 4).
    # Every frame below is built from THIS list rather than from whatever the aggregate
    # happened to contain -- see the `absent` guard.
    expected = metric_output_names(cfg)
    seeds = _derive_seeds(base_seed, n_splits)

    nmae_names = _lfc_nmae_names(expected)       # OUTPUT names -- see _lfc_nmae_names
    de_full = None
    if nmae_names:
        # ONE full-real DE table, reused by every split. It is seed-invariant: the gate,
        # the denominator, n_gate and every drop rule in `_nmae_ref_from_tables` come from
        # THIS table alone, and the single place a half is consulted (`absent`) RAISES
        # rather than dropping. Resolve the normalization target from the control pool
        # first, exactly as compute_lfc_nmae_reference does internally (#155), so the
        # table we hand it is the one it would have built.
        nmae_cfg = _resolve_target_sum_from_control(cfg, real_ad)
        de_full = _compute_de_side(real_ad, cfg=nmae_cfg, fp=None, store=None, side="real")

    rows: list[pl.DataFrame] = []
    for i, seed in enumerate(seeds):
        agg, counts = _score_one_split(real_ad, cfg, seed, metrics)
        vals = dict(zip(agg["metric"].to_list(), agg["mean"].to_list()))
        ns = dict(zip(counts["metric"].to_list(), counts["n_perturbations"].to_list()))
        if nmae_names:
            raw, n_ref = _lfc_nmae_raw(real_ad, cfg, seed, de_full)
            for m in nmae_names:
                # INSERT into the dict, NOT `agg.with_columns(...)`. Verified by dry-run:
                # when de_lfc_nmae emits no tidy rows the metric is ABSENT from `agg`, and
                # `with_columns` cannot add a row -- the substitution would silently no-op
                # and the member would vanish from the anchor.
                vals[m] = raw
                ns[m] = n_ref
        # MEASURED, not assumed: `aggregate_metrics` groups the tidy frame, so a selected
        # metric that emitted no rows is absent from `agg` entirely rather than present as
        # NaN. Building the split frame FROM `agg` would let a scored member disappear from
        # the anchor with nothing recording it -- #222's exact shape, one layer up. Build
        # from `expected` and refuse instead.
        absent = [m for m in expected
                  if vals.get(m) is None or not math.isfinite(float(vals[m]))]
        if absent:
            raise ValueError(
                f"split {i} (seed {seed}) produced no usable value for {len(absent)} "
                f"selected metric(s): {absent}. The anchor must cover every selected "
                "metric: a member missing from the anchor cannot be scored on this scale, "
                "and silently dropping it would change the competition average's "
                "denominator with nothing in the artifact saying so."
            )
        rows.append(pl.DataFrame(
            {
                "split_index": [i] * len(expected),
                "seed": [seed] * len(expected),
                "metric": expected,
                "value": [float(vals[m]) for m in expected],
                "n_perturbations": [None if ns.get(m) is None else int(ns[m])
                                    for m in expected],
            },
            schema=_SPLITS_SCHEMA,
        ))
    splits = pl.concat(rows).sort("split_index", "metric")
    # NOT projected before this group_by (Gemini, PR #284, medium). Measured: `splits` is
    # n_output_metrics x n_splits rows -- 50 for `vcc2026` at k=5, 270 for the widest
    # profile -- over 5 narrow columns, 3 of which the aggregation reads. There is nothing
    # to save, and the suggested rewrite dropped the estimator/select/sort chain below.
    anchor = (
        splits.group_by("metric")
        .agg(
            replicate=pl.col("value").mean(),
            replicate_sd=pl.col("value").std(ddof=0),
            replicate_min=pl.col("value").min(),
            replicate_max=pl.col("value").max(),
            n_perturbations_min=pl.col("n_perturbations").min(),
            n_perturbations_max=pl.col("n_perturbations").max(),
        )
        .with_columns(
            estimator=pl.when(pl.col("metric").is_in(nmae_names))
            .then(pl.lit(FULL_GATE_RAW, dtype=pl.Utf8))
            .otherwise(pl.lit(SPLIT_HALF_RAW, dtype=pl.Utf8))
        )
        .select(list(_ANCHOR_SCHEMA))
        .sort("metric")
    )
    return splits, anchor


ANCHOR_AGG = "anchor_agg.parquet"
ANCHOR_SPLITS = "anchor_splits.parquet"
ANCHOR_META = "anchor_meta.json"

#: `control_source` is deliberately ABSENT: the producer forces "pred" (module docstring),
#: so the anchor's value cannot depend on what the caller asked for. Keying on it would miss
#: the cache and reject a valid artifact for a difference that provably cannot move a number.
#: Both values are stamped in the meta instead, which is what lets part C enforce estimand
#: alignment without B guessing at the convention.
_SEMANTIC_FIELDS = (
    "version", "input_type", "target_sum", "bulk_target_sum", "allow_discrete",
    "autodetect_input_type", "allow_fractional_counts", "max_counts_per_cell",
    "validate_input", "pert_col", "control",
)
#: `target_gene_map` has THREE consumers, and gating it on any one alone is wrong: the
#: discrimination dispatch (run.py:296), `_prepare_de_cached`'s `resolve_target_genes`
#: (run.py:633), which feeds the target-excluding DE metrics that `vcc2026` scores, and as of
#: #172 `metrics.delta`'s `_exclusion_cols`, which feeds the two EXPRESSION legs of
#: `expr_mse_unbiased_capped_norm`. #248 is what happens when the map is absent: guide-level
#: labels match no gene, the exclusion silently no-ops, and a trivially-gameable submission
#: wins. Included whenever ANY of the three families is selected.
#: `distance` is NOT here: run.py binds each pds_* variant's distance through
#: functools.partial, so the dispatcher never reads the field (config.py:46-49). The others
#: ARE consumed at dispatch and move `pds_cosine`, a scored member.
#: `exclusion_scope` (#343) is here for that reason and is NOT optional bookkeeping: an anchor
#: frozen under "row" and a run scored under "panel" differ by up to +0.27 of `pds_cosine` on a
#: content-free submission, and without this field the two would carry the SAME semantic
#: identity and be enrolled as comparable.
#: `tie_policy` is here for the same reason and its omission was a PRE-EXISTING hole, found by
#: the cross-provider review of #343 and fixed in the same wave because this is the moment it
#: costs nothing (rule_version 3 invalidates every artifact anyway). It is consumed at dispatch
#: and it moves a scored member: on a fully tied row -- what a control-pasting submission
#: produces under cosine, the #282 case -- `pds_cosine` reads {0.5, 0.5} under "midrank" and
#: {1.0, 0.0} under "position", i.e. the target's ALPHABETICAL index. Without this field an
#: anchor frozen under one policy enrols against a run scored under the other.
_SEMANTIC_DISCRIMINATION_FIELDS = ("rank_denominator", "exclude_target_gene",
                                   "exclusion_scope", "tie_policy", "embed_key")
_SEMANTIC_DE_FIELDS = (
    "method", "mean_calc", "epsilon", "p_adj_threshold", "sort_by",
    "nan_lfc_policy", "min_abs_log2fc", "clip_value", "fdr_scope", "auc_pval_floor",
    "auc_pval_floor_value",
)
_SEMANTIC_FILTER_FIELDS = ("filter_gene_min_cpm_cell",)

#: Sidecar fields an anchor must carry to be checkable at all. `read_anchor` requires them
#: on the SUPPLIED side (a caller error -> raise); Task 8's `_bundle_from_obj` requires the
#: same tuple on the CACHED side, where their absence is a corrupt entry -> a miss and a
#: recompute. Getting that boundary wrong turns a damaged cache file into an aborted run.
_REQUIRED_META = ("real_fingerprint", "semantic_identity", "cell_eval2_version",
                  "metric_names", "control_source_effective")


def _version() -> str:
    try:
        return _pkg_version("cell_eval2")
    except PackageNotFoundError:        # never lose an anchor to provenance
        return "unknown"


class AnchorBackendUnresolved(Exception):
    """The DE backend an anchor's identity depends on could not be resolved.

    Raised only from `anchor_semantic_params`, and only for the resolution of
    `de.backend="auto"` -- which needs an installed engine and raises on a CUDA host without
    gpudge (`de_compute._resolve_backend`). It exists so `baseline.build_run_meta` can
    tolerate exactly this one failure while still letting a genuine programming error in the
    identity computation surface: an ORDINARY run that supplies both DE tables needs no
    engine, and `baseline._de_backend_used` is written precisely so such a run never resolves
    the backend. The anchor's identity DOES depend on it (an anchor always computes its own
    DE), so a run that cannot resolve it simply has no anchor identity to record.
    """


def anchor_semantic_params(cfg: EvalConfig, real_ad, names) -> dict:
    """The NAMED subset of config an anchor's value actually depends on.

    Written once and used in BOTH places that ask "are these two anchors the same": the
    artifact's `semantic_identity` gate (Task 5) and the cache key (Task 8). Two separate
    lists would drift, and the failure mode of a cache key narrower than the validation gate
    is a false hit that ships another configuration's anchor.

    Deliberately NOT `cache.config_hash`: that digests the whole config minus five fields,
    so it moves on knobs that cannot move an anchor. #181's design calls for a named subset;
    this is it. `config_hash` is still stamped beside it as provenance.

    Takes `real_ad` and the resolved `names` because two of the dependencies are not
    readable off the config alone:

    * the EFFECTIVE input type and the resolved COMPARATOR. `compute_metrics` resolves both
      from the data (run.py:981, run.py:1004) and the comparator decides which normalization
      every `expr_*` metric is computed in -- which is what #264 moved and #268 retuned. The
      declared `cfg.input_type` is not a substitute under v1 or `autodetect_input_type`.
      Both halves of an anchor split come from the real side, so ONE effective type answers
      for both.
    * whether a DE ENGINE runs at all. The DE fields and the resolved backend enter the key
      ONLY when a DE metric is selected -- mirroring `_result_config_digest`'s
      `de_backend_used` predicate (run.py:880-890) and `baseline._de_backend_used`. Making
      them unconditional would (a) reject an expression-only anchor after a DE-threshold
      change that provably cannot move it and (b) make an expression-only `backend="auto"`
      run resolve a backend, which RAISES on a CUDA host without gpudge and in a minimal
      install. The anchor always COMPUTES its DE (it never accepts a supplied table), so the
      predicate here is simply "a DE metric is selected".

    Add a field here whenever a new config knob can change a selected metric's value.
    Leaving one out is a false cache hit; a spurious one only costs a recompute.
    """
    from .catalog import CATALOG
    from .norm import resolve_comparator
    from .metrics.direction import REACH_PURITY_FLOOR
    from .run import (_effective_input_type, _GROUPED_SUM_REDUCTION_SEMANTICS,
                      _grouped_sum_reduction_used, _is_discrimination,
                      _ONTARGET_EXCLUSION_SEMANTICS, _ontarget_exclusion_used,
                      _reach_floor_used)

    params = {f: getattr(cfg, f) for f in _SEMANTIC_FIELDS}
    # BOTH sides' effective types, with the SAME roles the split run uses: `_score_one_split`
    # passes half_b as the PRED side, and `_effective_autodetect` (run.py:149) applies
    # `autodetect_input_type` to the pred side ONLY.
    #
    # These are resolved on `real_ad`, not on the halves -- which is exact for every config
    # this producer ACCEPTS, because `compute_replicate_anchor` refuses the two configs where
    # a half could re-type independently of the full matrix (see the guard there). Without
    # that refusal the stamped comparator could be one the run never used, and a comparator
    # decides which normalization every expr_* member is computed in.
    eff_real = _effective_input_type(real_ad, cfg, side="real")
    eff_pred = _effective_input_type(real_ad, cfg, side="pred")
    params["input_type_effective_real"] = eff_real
    params["input_type_effective_pred"] = eff_pred
    params["comparator"] = resolve_comparator(version=cfg.version,
                                              pred_input_type=eff_pred,
                                              real_input_type=eff_real)
    is_pds = any(_is_discrimination(CATALOG[n].func) for n in names)
    is_de = any(CATALOG[n].kind == "de" for n in names)
    # #172 gave the EXPRESSION legs a third consumer of the map, and it is neither of the two
    # the block below was written for: `mse_unbiased_capped` and `distance_unbiased` resolve
    # each perturbation's own gene through `cfg.target_gene_map` too. Reusing `run`'s own
    # predicate rather than restating the metric list here is the point -- the two must agree,
    # and the docstring above says why a narrower key than the validation gate is unsafe.
    is_ontarget = _ontarget_exclusion_used(names)
    # Same shape as `is_ontarget`, for the same reason: the anchor is a FROZEN artifact, and a
    # bundle built before the purity floor moved carries a replicate computed under a different
    # rule for a scored member. Nothing else in this dict can see that -- the floor is a module
    # constant, not a config knob, so `_SEMANTIC_FIELDS` and `config_hash` are both blind to it.
    # By VALUE, so a future retune moves the identity with nothing to remember.
    if _reach_floor_used(names):
        params["reach_purity_floor"] = REACH_PURITY_FLOOR
    # #271, and the same argument as `reach_purity_floor` directly above: the anchor is a FROZEN
    # artifact, and one built before `prep._grouped_sums` began reducing wide carries a replicate
    # computed from differently-rounded group sums. Nothing else in this dict can see it -- the
    # reduction dtype is not a config knob, so `_SEMANTIC_FIELDS` and `config_hash` are both blind,
    # and `params["comparator"]` above records which comparator was resolved rather than what a
    # group sum under it MEANS. Conditional on the same predicate the result cache uses.
    # ⚠️ `cfg.de.backend` is the RAW spelling and can be "auto" -- not the resolved backend, which
    # the result-cache path gets from `_cache_backend`. That is sufficient here for one reason only:
    # `auto` NEVER selects deseq2 (`de_compute._resolve_backend` excludes it, pinned by
    # `test_auto_never_resolves_deseq2`), so "auto" cannot hide a deseq2 pseudobulk. If that
    # invariant is ever relaxed, resolve the backend here instead of reading the raw field.
    if _grouped_sum_reduction_used(comparator=params["comparator"],
                                   de_backend=(cfg.de.backend if is_de else None)):
        params["grouped_sum_reduction_semantics"] = _GROUPED_SUM_REDUCTION_SEMANTICS
    # CONDITIONAL, on the same principle as the DE block below: a knob no selected metric
    # reads cannot move this anchor, and keying on it rejects a legitimate artifact.
    if is_pds:
        params.update({f"discrimination.{f}": getattr(cfg.discrimination, f)
                       for f in _SEMANTIC_DISCRIMINATION_FIELDS})
        # GPU PDS reduction blocking (run.py:290). `baseline.DIGEST_EXEMPT_FIELDS`
        # (baseline.py:636) deliberately does NOT exempt it from scoring identity; it is
        # gated here rather than unconditional because the discrimination dispatch is its
        # only consumer.
        params["pert_chunk"] = cfg.pert_chunk
    if is_pds or is_de or is_ontarget:
        # ALL THREE families, not just pds_*: `_prepare_de_cached` resolves target genes
        # through this map at run.py:633, which the target-excluding direction members of
        # `vcc2026` then consume, and as of #172 `metrics.delta` resolves through it for the
        # two expression legs of `expr_mse_unbiased_capped_norm`. Gating it on pds_* alone
        # would let a map change move a scored DE member with the cache key standing still;
        # gating it on pds_*/DE alone leaves the same hole for an EXPRESSION-only anchor,
        # which is the one selection where neither of the first two predicates fires.
        params["target_gene_map"] = cfg.target_gene_map
    if is_ontarget:
        # A CODE-semantics term, not a config knob -- the one case the "add a field whenever a
        # new config knob can change a value" rule above does not reach. `anchor_cache_params`
        # stamps `cell_eval2_version`, but that is not a substitute twice over: it resolves
        # through the INSTALLED distribution metadata (stale in an editable dev tree), and
        # #172 lands WITHIN 0.13.0, so a warm anchor built before it carries the same version
        # string as a run after it. Without this term that anchor is a false hit whose value
        # was computed over a gene set the run no longer scores. Mirrors
        # `run._result_config_digest`'s term of the same name; bump BOTH together if the
        # exclusion's meaning ever moves again.
        params["ontarget_exclusion_semantics"] = _ONTARGET_EXCLUSION_SEMANTICS
    # RESOLVED device, not the declared one: "auto" is cuda on a GPU node and cpu elsewhere,
    # and fp32-GPU vs fp64-CPU pseudobulk moves every pseudobulk-based metric. This is the
    # F2.1 rationale already in the pseudobulk cache key. CONSEQUENCE, stated so it is a
    # decision and not a surprise: an anchor built on a GPU node will not validate when
    # scoring on a CPU node. That is the rule the existing caches already enforce; the
    # alternative is silently mixing two engines' numbers inside one competition scale.
    params["device"] = _cache_device(cfg)
    if is_de:
        params.update({f"de.{f}": getattr(cfg.de, f) for f in _SEMANTIC_DE_FIELDS})
        try:
            params["de.backend_resolved"] = _cache_backend(cfg)
        except Exception as exc:        # noqa: BLE001 -- re-raised as a NAMED failure
            # Narrowed to a dedicated type so `build_run_meta` can tolerate this one
            # environment failure without swallowing a programming error. `_resolve_backend`
            # raises RuntimeError today; the wrapper does not depend on that staying true.
            raise AnchorBackendUnresolved(
                f"the anchor's identity needs the resolved DE backend, and "
                f"de.backend={cfg.de.backend!r} could not be resolved: {exc}"
            ) from exc
        # `filter_gene_min_cpm_cell` belongs HERE, not in the unconditional block: every
        # consumer in the tree is a DE path (run.py:762, de_compute.py, partition_inmem,
        # scale.py's DE call) -- no pseudobulk or expression metric reads it. Keying an
        # expression-only anchor on it would reject a valid artifact after a change that
        # provably cannot move a single one of its numbers.
        params.update({f"filter.{f}": getattr(cfg.filter, f)
                       for f in _SEMANTIC_FILTER_FIELDS})
        if cfg.de.backend == "deseq2":
            # Both only for deseq2, mirroring run.py:804-806: the pseudobulk replicate
            # grouping AND the CPU(numpy)/GPU(jax) fit each change the DE table. The fit
            # predicate is the RAW device prefix, NOT _cache_device -- "auto" fits on CPU
            # while _cache_device maps it to cuda.
            params["de.replicate_col"] = cfg.de.replicate_col
            params["de.deseq2_gpu_fit"] = str(cfg.device).startswith("cuda")
    return params


def semantic_identity(cfg: EvalConfig, real_ad, names) -> str:
    """Digest of `anchor_semantic_params`. The gate both anchor doors run."""
    return _hash_obj("anchor_semantic",
                     json.dumps(anchor_semantic_params(cfg, real_ad, names),
                                sort_keys=True, default=str))


def build_meta(*, real_ad, cfg, names, base_seed, n_splits, seeds, metrics) -> dict:
    """The sidecar that makes an anchor self-identifying.

    Without it the supplied-artifact path is a hole: `lfc_nmae_ref` today accepts ANY frame
    carrying the right three columns and validates nothing about provenance, so another
    dataset's reference returns a plausible number. Stamping the fingerprint means "cached"
    is *we found one whose fingerprint matched* and "supplied" is *you handed us one and we
    check the same fields* -- one guard, both doors.

    `real_fingerprint` is the STRICT content hash and it is the only fingerprint the gate
    ever compares -- the spec's field name and the spec's strength, unchanged.

    An earlier draft stamped both strengths and let the gate pick whichever the peer
    carried. That was wrong, and wrong in the worst direction: `build_run_meta` computes
    `source_fingerprint` at `strict=cfg.cache_strict` (baseline.py:793), which is FALSE by
    default, so the "compare like with like" rule would have made the metadata hash the
    DEFAULT gate -- and two datasets with identical shape, dtype, gene names and per-cell
    labels but different `X` would validate as the same anchor. That is exactly the hole
    `strict=True` was introduced to close. The metadata hash is kept as auxiliary
    provenance, never as a gate; the strength requirement is enforced at the `score`
    boundary instead (Task 10: a user run without a strict fingerprint cannot score against
    an anchor, and is told to re-run with `--cache-strict`).
    """
    return {
        # THE gate. strict: hashes X. The default hashes structure and the per-cell label
        # assignment but NOT X.
        "real_fingerprint": fingerprint_adata(real_ad, pert_col=cfg.pert_col, strict=True),
        # auxiliary provenance ONLY -- never compared. Recorded so a mismatch can be
        # diagnosed ("same structure, different X") rather than merely reported.
        "real_fingerprint_meta": fingerprint_adata(real_ad, pert_col=cfg.pert_col,
                                                   strict=False),
        "semantic_identity": semantic_identity(cfg, real_ad, names),
        # readable beside the digest, so a mismatch can be diffed rather than guessed at
        "semantic_params": anchor_semantic_params(cfg, real_ad, names),
        "config_hash": config_hash(cfg.to_dict()),        # provenance only, never the gate
        # BOTH, per spec 4.2. The producer forces "pred"; ordinary v2 scoring defaults to
        # "real". Stamping only the caller's config would record a convention that did not
        # produce the number, and part C cannot enforce estimand alignment against a value
        # that was never used.
        "control_source_requested": cfg.control_source,
        "control_source_effective": _inner_config(cfg).control_source,
        "base_seed": int(base_seed),
        "seed_derivation": SEED_DERIVATION,
        "derived_seeds": [int(s) for s in seeds],
        "n_splits": int(n_splits),
        "cell_eval2_version": _version(),
        "bulk_target_sum": float(cfg.bulk_target_sum),
        "metric_names": list(metrics),
    }


def write_anchor(outdir, splits: pl.DataFrame, anchor: pl.DataFrame, *, meta: dict) -> str:
    """Persist the anchor, the splits and the sidecar. Returns the sidecar path."""
    os.makedirs(outdir, exist_ok=True)
    anchor.write_parquet(os.path.join(outdir, ANCHOR_AGG))
    splits.write_parquet(os.path.join(outdir, ANCHOR_SPLITS))
    path = os.path.join(outdir, ANCHOR_META)
    with open(path, "w") as fh:
        json.dump(meta, fh, indent=1, sort_keys=True, default=str)
    return path


def read_anchor(source) -> tuple[pl.DataFrame, pl.DataFrame, dict]:
    """Load (anchor, splits, meta) from a sidecar path or its directory.

    LOAD-TIME checks only, and the distinction matters (Copilot, PR #284): this verifies the
    sidecar is an object carrying `_REQUIRED_META`, and that the anchor frame has the
    expected COLUMNS. It does NOT check dtypes, row uniqueness, metric coverage, finiteness
    or the estimator pairing -- those are `validate_anchor`'s, run through `resolve_anchor`
    against a specific run's `AnchorExpect`, which this function has no access to. A caller
    that reads an anchor and skips `resolve_anchor` has NOT validated it.
    """
    path = str(source)
    outdir = os.path.dirname(path) if path.endswith(".json") else path
    meta_path = os.path.join(outdir, ANCHOR_META)
    if not os.path.exists(meta_path):
        raise ValueError(
            f"no {ANCHOR_META} at {outdir!r}; an anchor directory must carry its sidecar "
            "-- a bare parquet has no provenance to check."
        )
    with open(meta_path) as fh:
        meta = json.load(fh)
    # The sidecar must be a DICT carrying the gate's fields, checked HERE. `validate_anchor`
    # reaches for `meta.get(...)`, so a sidecar holding `[]` or `"x"` would die with an
    # AttributeError naming the validator instead of the documented source-naming
    # ValueError -- and the supplied door is the one a competitor hands us a file for.
    # Task 8 covers the same shape on the CACHED side, where it is a miss rather than a raise.
    if not isinstance(meta, dict):
        raise ValueError(
            f"anchor sidecar ({meta_path}) holds a {type(meta).__name__}, not an object; "
            "pass the anchor_meta.json written by `run --anchor`."
        )
    if missing_meta := [f for f in _REQUIRED_META if f not in meta]:
        raise ValueError(
            f"anchor sidecar ({meta_path}) is missing {missing_meta}; it cannot be checked "
            "against this run. Rebuild the anchor."
        )
    frame = pl.read_parquet(os.path.join(outdir, ANCHOR_AGG))
    splits = pl.read_parquet(os.path.join(outdir, ANCHOR_SPLITS))
    if missing := set(_ANCHOR_SCHEMA) - set(frame.columns):
        raise ValueError(
            f"anchor ({outdir}) is missing column(s) {sorted(missing)}; got "
            f"{frame.columns}. Pass the directory written by `run --anchor`."
        )
    return frame, splits, meta


def anchor_digest(frame: pl.DataFrame, meta: dict) -> str:
    """A digest over the anchor's VALUES plus its identity, stamped into the scored frame.

    Over the values, not only the meta: two anchors of the same dataset under the same
    config differ only in their numbers, and "which anchor produced this score" has to be
    answerable from the score file alone.
    """
    payload = frame.sort("metric").select(["metric", "replicate", "estimator"]).to_dicts()
    return _hash_obj("anchor", json.dumps(payload, sort_keys=True),
                     meta.get("real_fingerprint"), meta.get("semantic_identity"),
                     meta.get("cell_eval2_version"))


@dataclass(frozen=True)
class AnchorExpect:
    """What THIS run needs an anchor to be. Built in exactly two places: the producer (from
    the real data it just read) and `score` (from the user run's `run_meta.json`).

    `fingerprint` is ALWAYS the strict content hash. There is deliberately no strength knob:
    a knob makes the weaker comparison reachable, and because `build_run_meta` is
    metadata-only by default it would have made the weaker comparison the DEFAULT. `score`
    enforces the strength at its own boundary instead, by refusing a user run that has no
    strict fingerprint.

    One object rather than five loose kwargs, because five kwargs is how the first draft
    ended up with `score` passing none of them and a test handing the anchor its own
    metadata back as its own expectation.
    """
    fingerprint: str
    semantic_identity: str
    version: str
    metrics: tuple[str, ...]


def validate_anchor(frame, meta, expect: AnchorExpect, *, source) -> None:
    """The ONE guard both doors run. Source named in every message.

    The identity gate is `semantic_identity` -- a NAMED subset (see
    `anchor_semantic_params`) -- and not the repo's broad `config_hash`, which moves on
    fields that cannot change a number and would reject a legitimate anchor. `config_hash`
    stays in the meta as provenance.
    """
    # `real_fingerprint` is the STRICT content hash and the only fingerprint compared.
    # `real_fingerprint_meta` sits in the sidecar as provenance and is never a fallback:
    # degrading to it would accept a different X under identical structure.
    for field, want in (("real_fingerprint", expect.fingerprint),
                        ("semantic_identity", expect.semantic_identity),
                        ("cell_eval2_version", expect.version)):
        got = meta.get(field)
        if got != want:
            raise ValueError(
                f"anchor ({source}) has {field}={got!r} but this run needs {want!r}. An "
                "anchor is a property of ONE dataset scored under ONE configuration -- "
                "using another's returns a plausible number that is not comparable."
                + ("" if got is not None else
                   f" (the sidecar carries no {field!r}; rebuild the anchor.)")
            )
    # The anchor's estimand is not negotiable (spec 5.5): the producer forces per-half
    # controls, and an artifact reporting anything else was produced by a build whose forcing
    # was removed, or hand-edited. Either way the top of the competition scale would be
    # optimistic with no other signal, so refuse rather than warn.
    effective = meta.get("control_source_effective")
    if effective != "pred":
        raise ValueError(
            f"the {source} anchor reports control_source_effective={effective!r}, but a "
            "replicate anchor is only valid under per-half controls ('pred'): a shared "
            "control correlates the two halves whose agreement it measures (measured "
            "optimistic by 0.5-2.3% on lfc_nmae). Rebuild it with `run --anchor`."
        )
    # EXACT schema, not a subset: a Float32 `replicate` reads back a different number than
    # the Float64 that was written, and a silently-widened Int32 cohort count would compare
    # unequal to the member's. Checked before any value is read.
    if dict(frame.schema) != dict(_ANCHOR_SCHEMA):
        missing = sorted(set(_ANCHOR_SCHEMA) - set(frame.columns))
        if missing:
            raise ValueError(
                f"anchor ({source}) is missing column(s) {missing}; got {frame.columns}"
            )
        bad = {c: (str(frame.schema[c]), str(t)) for c, t in _ANCHOR_SCHEMA.items()
               if frame.schema.get(c) != t}
        raise ValueError(
            f"anchor ({source}) has the wrong dtype for {sorted(bad)}: got/expected "
            f"{bad}. Rebuild it rather than casting -- a cast hides which producer wrote it."
        )
    if frame.height == 0:
        raise ValueError(
            f"anchor ({source}) is empty. An empty anchor is not 'nothing to scale' -- it "
            "is a producer that failed, and scoring against it would silently drop every "
            "member from the scale."
        )
    names = frame["metric"].to_list()
    if len(set(names)) != len(names):
        dupes = sorted({n for n in names if names.count(n) > 1})
        raise ValueError(
            f"anchor ({source}) has duplicate metric row(s): {dupes}. One row per metric -- "
            "with two, which one scales the submission depends on frame order."
        )
    want = set(expect.metrics)
    if missing := sorted(want - set(names)):
        raise ValueError(
            f"anchor ({source}) is missing {len(missing)} expected metric(s): {missing}. "
            "This run scores them, so an anchor without them has no top end for those "
            "members."
        )
    if extra := sorted(set(names) - want):
        raise ValueError(
            f"anchor ({source}) carries {len(extra)} unexpected metric(s): {extra}. It was "
            "built for a different metric selection, so its other rows were measured under "
            "a selection this run does not use."
        )
    # The sidecar's own metric list must agree with the frame. They are written together, so
    # a disagreement means one of the two files was replaced -- and `metric_names` is what
    # the CACHE key was built from, so a frame that outgrew it would be served forever.
    if sorted(meta.get("metric_names") or []) != sorted(names):
        raise ValueError(
            f"anchor ({source}) has metric_names={sorted(meta.get('metric_names') or [])} "
            f"but its frame carries {sorted(names)}; the sidecar and the parquet disagree, "
            "so one of them was replaced. Rebuild the anchor."
        )
    bad = [m for m, v in zip(names, frame["replicate"].to_list())
           if v is None or not math.isfinite(float(v))]
    if bad:
        raise ValueError(
            f"anchor ({source}) has a non-finite replicate for {len(bad)} metric(s) "
            f"(e.g. {bad[0]!r}); it would propagate into the division rather than be "
            "detectable. Rebuild the anchor."
        )
    # The ESTIMATOR column is the artifact's record of the settled decision (spec 4.2), so
    # it is checked for MEANING and not only for dtype. An anchor claiming `split_half_raw`
    # for an lfc_nmae member asserts it was built by the estimator spec 3.2 rules out -- a
    # 21-35% cohort mismatch on real data that `score`, seeing two scalars, cannot detect.
    known = {SPLIT_HALF_RAW, FULL_GATE_RAW}
    nmae = set(_lfc_nmae_names(names))
    for metric, est in zip(names, frame["estimator"].to_list()):
        if est not in known:
            raise ValueError(
                f"anchor ({source}) has an unknown estimator {est!r} for {metric!r}; "
                f"expected one of {sorted(known)}. It was not written by this producer."
            )
        wanted = FULL_GATE_RAW if metric in nmae else SPLIT_HALF_RAW
        if est != wanted:
            raise ValueError(
                f"anchor ({source}) records estimator={est!r} for {metric!r}, but this "
                f"metric's anchor must come from {wanted!r}. "
                + ("An lfc_nmae anchor built by the uniform split-half core averages over "
                   "a 21-35% smaller cohort than the member it normalizes (spec 3.2)."
                   if metric in nmae else
                   "Only the lfc_nmae members use the full-real-gated estimator.")
            )


def _as_bundle(src) -> tuple[pl.DataFrame, dict]:
    """A door's argument as (frame, meta), whichever shape it arrived in.

    The supplied door is a path or directory; the CACHED door is the three-part value
    ``(anchor, splits, meta)`` that `cached_anchor` returns, because
    `CacheStore.get_or_compute` returns the cached object rather than a directory.
    Normalizing here is what keeps ONE validation for both -- a second code path for the
    in-memory shape is how the cached door ends up trusted rather than checked.
    """
    if isinstance(src, tuple):
        if len(src) != 3:
            raise ValueError(
                f"an in-memory anchor must be the 3-tuple (anchor, splits, meta) that "
                f"`cached_anchor`/`read_anchor` return; got a {len(src)}-tuple"
            )
        frame, _splits, meta = src
        return frame, meta
    frame, _splits, meta = read_anchor(src)
    return frame, meta


def resolve_anchor(expect: AnchorExpect, *, supplied=None, cached=None):
    """Supplied wins, else cached, else RAISE. Never recompute at score time.

    Each door is a path/directory OR the three-part in-memory bundle; see `_as_bundle`.

    Recomputing here is expensive AND it would derive the anchor from whatever data is at
    hand, which is precisely the plausible-wrong-number shape this scheme closes. The
    caller stamps the returned source label into the scored frame, so which door was used
    is a property of the ARTIFACT rather than of the invocation -- strictly more
    informative than a mode flag recording what the caller intended.
    """
    for label, src in (("supplied", supplied), ("cached", cached)):
        if src is None:
            continue
        frame, meta = _as_bundle(src)
        validate_anchor(frame, meta, expect,
                        source=str(src) if not isinstance(src, tuple) else "<cache>")
        return frame, meta, label
    raise ValueError(
        "no anchor available: none was supplied and none is cached for this dataset. "
        "Build one with `cell-eval2 run --anchor` first -- scoring will not silently "
        "recompute it."
    )


ANCHOR_CACHE_KEY = "replicate_anchor"


class _BadBundle(Exception):
    """A cached value that parsed as JSON but is not a usable anchor bundle."""


def anchor_store(cfg):
    """The anchor's cache, or None. The anchor is a property of the REAL side, so it lives in
    the real-side store beside that dataset's pseudobulk and DE artifacts. No `cache_real`
    means no anchor cache -- a missing cache, not an error."""
    if cfg.cache_real is None:
        return None
    from .cache import CacheStore
    return CacheStore(cfg.cache_real)


def anchor_cache_params(cfg, real_ad, names, *, base_seed, n_splits, metrics) -> dict:
    """The anchor's cache key params: the full production-semantic digest.

    `CacheStore` trusts exact `(fingerprint, params)` equality and knows nothing else, and an
    anchor's value depends on far more than its production parameters. The semantic half is
    `anchor_semantic_params` -- the SAME subset `validate_anchor` gates on, so a cache hit and
    a validation pass can never disagree about what "the same anchor" means.

    `validate_input` arrives through that subset (#161): here the artifact IS a score, so a
    permissive run's anchor must never be served to a run that asked for the guard.
    """
    params = dict(anchor_semantic_params(cfg, real_ad, names))
    params.update({
        "base_seed": int(base_seed),
        "n_splits": int(n_splits),
        "seed_derivation": SEED_DERIVATION,
        "metric_names": list(metrics),
        "cell_eval2_version": _version(),
    })
    return params


def _bundle_to_obj(splits, anchor, meta) -> dict:
    return {"anchor": anchor.to_dicts(), "splits": splits.to_dicts(), "meta": meta}


def _bundle_from_obj(obj) -> tuple[pl.DataFrame, pl.DataFrame, dict]:
    """Decode a cached bundle, raising `_BadBundle` on anything that is not one.

    Rebuilt against the PINNED schemas, so the round-trip cannot widen an Int64 cohort count
    or narrow a Float64 replicate -- `validate_anchor` checks dtypes exactly, and a dtype
    that only survives on the write path would make the cached door reject its own artifact.

    `_REQUIRED_META` is the same tuple `read_anchor` enforces on the supplied side. Here its
    absence is a corrupt cache entry -- a MISS and a recompute -- rather than a caller error.
    Getting that boundary wrong turns a damaged cache file into an aborted scoring run.
    """
    if not isinstance(obj, dict) or not {"anchor", "splits", "meta"} <= set(obj):
        got = sorted(obj) if isinstance(obj, dict) else type(obj).__name__
        raise _BadBundle(f"not an anchor bundle: {got}")
    meta = obj["meta"]
    # `meta` must be a DICT with the fields the gate reads. A bundle whose frames decode but
    # whose meta is a list escapes the check above and then crashes inside validation with a
    # TypeError, pointing at the validator rather than at the cache.
    if not isinstance(meta, dict) or any(f not in meta for f in _REQUIRED_META):
        raise _BadBundle(
            f"bundle meta is {type(meta).__name__} missing "
            f"{[f for f in _REQUIRED_META if not isinstance(meta, dict) or f not in meta]}"
        )
    try:
        anchor = pl.DataFrame(obj["anchor"], schema=_ANCHOR_SCHEMA)
        splits = pl.DataFrame(obj["splits"], schema=_SPLITS_SCHEMA)
    except Exception as exc:                       # noqa: BLE001 -- any decode failure
        raise _BadBundle(str(exc)) from exc
    return anchor, splits, meta


def cached_anchor(real, cfg, *, store, base_seed=0, n_splits=5):
    """The cached anchor for this dataset, computing it once on a miss.

    Calls `CacheStore.get_or_compute` -- a params dict alone caches nothing, which is what
    the first draft of this plan shipped. `get_or_compute` returns the VALUE, so the cached
    representation round-trips the aggregate, the splits and the meta as one object.

    A parseable-but-malformed entry is a MISS, not a crash. `CacheStore.get`'s read-failure
    catch (cache.py:383) spans only the JSON load, so a valid JSON object that is not a
    bundle would otherwise abort a scoring run on a corrupt cache file. A cache is an
    optimization; a corrupt one must never be fatal.
    """
    real_ad = load_anndata(real, backed=False)
    names, _missing = resolve_metrics(cfg.metrics, version=cfg.version)
    metrics = metric_output_names(cfg)
    fp = fingerprint_adata(real_ad, pert_col=cfg.pert_col, strict=True)
    params = anchor_cache_params(cfg, real_ad, list(names), base_seed=base_seed,
                                 n_splits=n_splits, metrics=metrics)

    def compute():
        # Through the MODULE namespace, so the cold/warm test's
        # `monkeypatch.setattr(anchor_mod, "compute_replicate_anchor", ...)` intercepts it.
        # A module-local binding here would make that counting test pass vacuously.
        splits, anchor = globals()["compute_replicate_anchor"](
            real_ad, config=cfg, base_seed=base_seed, n_splits=n_splits)
        meta = build_meta(real_ad=real_ad, cfg=cfg, names=list(names),
                          base_seed=base_seed, n_splits=n_splits,
                          seeds=_derive_seeds(base_seed, n_splits), metrics=metrics)
        return _bundle_to_obj(splits, anchor, meta)

    obj = store.get_or_compute(ANCHOR_CACHE_KEY, fingerprint=fp, params=params,
                               kind="json", compute=compute)
    try:
        return _bundle_from_obj(obj)
    except _BadBundle as exc:
        logger.warning(
            "anchor cache entry at %s is unusable (%s); recomputing and replacing it.",
            store.root, exc,
        )
        fresh = compute()
        store.put(ANCHOR_CACHE_KEY, fresh, fingerprint=fp, params=params, kind="json")
        return _bundle_from_obj(fresh)
