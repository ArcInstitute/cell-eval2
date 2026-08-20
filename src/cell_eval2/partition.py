"""Subset selection, partial-result output, and exact partial aggregation.

A scale run scores a *subset* of perturbations against a shared real reference and
writes a partial (parquet + json sidecar). ``aggregate_partials`` recombines the
partials from independent runs into the full tidy frame plus the v2 NaN-aware
aggregate -- refusing to combine partials computed against different real references
or configs, or with overlapping ``(perturbation, metric)`` rows.

#172's exclusion reaches the sidecar through ``result_semantics``' third counter,
``ontarget_exclusion_semantics``, and NOT through any of the four cross-partial fields
``aggregate_partials`` compares (``real_ref_fingerprint``, ``config_hash``, ``comparator``,
``metrics``) -- all four describe what was ASKED FOR rather than what a metric MEANS, so two
shards straddling that change agree on every one of them. Without the counter they would merge
into one frame in which some perturbations were scored over a gene set the rest were not, with
nothing in the output saying so. That is #246's failure mode with a different cause, which is
why the fix is a term in #246's payload rather than a fifth cross-partial field.
"""

from __future__ import annotations

import glob
import json
import logging
import os

import polars as pl

from .catalog import CATALOG
from .run import aggregate_metrics

logger = logging.getLogger(__name__)


def select_subset(perts, *, subset=None, fraction=None, index=None):
    perts = [str(p) for p in perts]
    if subset is not None:
        wanted = set(map(str, subset))
        return [p for p in perts if p in wanted]
    if fraction is not None:
        if index is None or not (0 <= index < fraction):
            raise ValueError(f"index must be in [0,{fraction}); got {index}")
        ordered = sorted(perts)
        return [p for i, p in enumerate(ordered) if i % fraction == index]
    return perts


#: Bump when the RESULT SEMANTICS payload below gains, loses or re-spells a term. It is part of
#: the payload, so a bump makes every partial written before it compare unequal to one written
#: after -- which is the point: a schema change is itself a semantics change.
#: 2 = `metric_worst_value`, `metric_kind` and `metric_needs_moments` added (codex-review). The
#: comment above is a rule, and round 1 added a term without honouring it.
#: 3 = `ontarget_exclusion_semantics` added (#172), the third sibling of the two counters below.
#: 4 = `reach_purity_floor` added -- `de_direction_reach`'s raw form moved from the derived
#:     `1 - alpha/2` to the calibrated `direction.REACH_PURITY_FLOOR`. Not a counter: the value
#:     itself, so a future retune moves the payload with no bump to remember.
_PARTIAL_SEMANTICS_SCHEMA = 5      # 2->3 #172, 3->4 the purity floor, 4->5 #271

#: The sidecar key. Named for what it is rather than for the digest, because `aggregate_partials`
#: reports the differing PAYLOADS on a mismatch -- a bare hash would say only "these disagree".
PARTIAL_SEMANTICS_KEY = "result_semantics"


def _semantics_diff(a: dict, b: dict) -> list[str]:
    """The term names on which two semantics payloads disagree, so a mismatch message says WHAT
    changed rather than only that something did."""
    return sorted(k for k in set(a) | set(b) if a.get(k) != b.get(k))


def _check_result_semantics(out_dir, semantics: dict, legacy: list[str], *, n_sidecars: int,
                            declared: list[tuple] = ()):
    """#246's cross-partial semantics guard, with the graded legacy policy the issue asks to make
    explicit. Three cases, and they are deliberately not one rule:

    1. Every sidecar declares and they DISAGREE -> raise. This is the defect: partials computed
       under two different meanings of the same metric name.
    2. Some declare and some do not -> raise. A mix is exactly the straddle #246 describes (old
       partials predate the key), and it cannot be verified: the legacy ones assert nothing, so
       "no disagreement observed" would be a statement about the schema, not about the numbers.
    3. NONE declares -> WARN and proceed. An all-legacy directory predates the key entirely, and
       there is no evidence its partials disagree -- one code version most likely wrote all of
       them. Refusing would break every warm partial directory to protect against a mix that, by
       construction, is not present. This is the same lenient/strict split the ``metrics`` key
       already uses two blocks below, for the same reason. ⚠️ #271 is a live instance of what case 3
       tolerates: an all-legacy directory holding ``bulk_lognorm`` results computed under the NARROW
       group-sum reduction is aggregated with a warning by this rule, since it declares nothing to
       disagree with. Accepted under #246's ruling rather than reopened here.

    ⚠️ AND, before any of the three: each declared payload is checked against what THIS code would
    produce for that sidecar's own (metrics, comparator). Cross-sidecar agreement alone is NOT
    enough, and the first version of this guard shipped that hole -- if every sidecar declares the
    same OLD payload they agree with each other, the guard passes, and ``aggregate_metrics`` then
    reduces them with the CURRENT catalog. An all-``mean`` set could be median-reduced after a
    catalog change with no error: exactly the silent failure #246 is about, moved one step. Found by
    ``codex-review``, not by a test.
    """
    for name, metrics, comparator, sem in declared:
        # ⚠️ NEVER skip. The first version of this loop did `continue` when `metrics` was absent, and
        # that failed open: one stale sidecar -- or a whole directory of them -- was accepted without
        # any check (codex-review round 2, which also caught that my own cross-sidecar tests
        # manufactured exactly that hybrid to reach the other branch). `metrics` PREDATES
        # `result_semantics`, so no genuine writer emits semantics without it; the combination means
        # a hand-edited or corrupt sidecar.
        if not isinstance(sem, dict) or not isinstance(metrics, list):
            raise ValueError(
                f"partial sidecar {name!r} in {out_dir!r} carries {PARTIAL_SEMANTICS_KEY!r} but "
                f"metrics={metrics!r} and semantics of type {type(sem).__name__} -- a sidecar that "
                "declares result semantics must also declare the metric list they describe "
                "(`metrics` predates this key, so no writer produces one without the other). "
                "Refusing to aggregate: without the list there is nothing to validate the payload "
                "against, and skipping the check would accept a stale payload silently (#246)."
            )
        want = result_semantics(metrics, comparator=comparator)
        if sem != want:
            raise ValueError(
                f"partial sidecar {name!r} in {out_dir!r} declares result semantics this build "
                f"does not produce: it disagrees on {_semantics_diff(sem, want)}. The partial was "
                "computed under a different meaning of its own metric names, so reducing it with "
                "the current catalog would produce a plausible aggregate over incompatible values "
                "-- and cross-sidecar agreement cannot see that, because every sidecar in a stale "
                "directory agrees with every other (#246). Discard the directory and rebuild the "
                "partials with this version."
            )
    if semantics and legacy:
        raise ValueError(
            f"partial sidecars in {out_dir!r} MIX declared and undeclared result semantics "
            f"({len(legacy)} of {n_sidecars} carry no {PARTIAL_SEMANTICS_KEY!r}: "
            f"{legacy[:5]}); refusing to aggregate. A partial written before this key existed "
            "cannot be verified against one written after, and a partial straddling a change in "
            "what a metric MEANS -- rather than in which metrics were selected -- otherwise "
            "concatenates into a plausible aggregate with no error (#246). Discard the directory "
            "and rebuild the partials with one version."
        )
    if len(semantics) > 1:
        (fa, a), (fb, b) = list(semantics.values())[:2]
        raise ValueError(
            f"partial sidecars in {out_dir!r} differ in RESULT SEMANTICS: {fa!r} and {fb!r} "
            f"disagree on {_semantics_diff(a, b)} (of {len(semantics)} distinct payloads across "
            f"{n_sidecars} sidecars); refusing to aggregate. These partials were computed under "
            "different meanings of the same metric names, so reducing them together would "
            "produce a plausible aggregate over incompatible values (#246). Discard the "
            "directory and rebuild the partials with one version."
        )
    if not semantics:
        logger.warning(
            "no partial sidecar in %r declares %r, so the result-semantics guard (#246) cannot "
            "run: a partial straddling a change in what a metric MEANS would reduce silently. "
            "These partials predate the key; rebuild them to get the guard.",
            out_dir, PARTIAL_SEMANTICS_KEY,
        )


def result_semantics(names, *, comparator: str) -> dict:
    """What the partial's numbers MEAN, for the cross-partial guard (#246).

    ``aggregate_partials`` already refuses partials that differ in reference, config or metric
    SELECTION. What it could not see is a partial straddling a change in what a selected metric
    *means*: it validated what was recorded and then applied the CURRENT catalog to reduce
    whatever it found, so such a mix concatenated into a plausible aggregate with exit 0. #263 --
    the selection half -- was already closed by the ``metrics`` key and the check below; this is
    the half that survived.

    Modelled on ``baseline._baseline_policy_dict``, which #231 gave an ordered ``metric_agg``
    term for the same reason, plus the two semantics counters ``run._result_config_digest``
    already keys the result cache on:

    * ``metric_agg`` -- which statistic reduces each metric. Ordered pairs, not a dict, so
      mapping-iteration order cannot move the payload.
    * ``metric_derived`` -- a derived metric's identity is its name PLUS the two components it
      divides (``catalog.derived_policy``); a component swap is invisible to the name alone.
    * ``metric_normalization`` -- the RESOLVED per-metric space. The run-level ``comparator``
      token alone does not identify it: #264 PR1 stamped ``comparator="bulk_lognorm"`` while six
      ``expr_*`` entries still declared ``lognorm``, so two runs agreed on the comparator while
      computing different numbers.
    * ``de_rank_semantics`` / ``pds_exclusion_semantics`` / ``ontarget_exclusion_semantics`` --
      the repo's "the meaning of this family changed" counters (#248 is why the second exists,
      #172 the third). Included UNCONDITIONALLY here, unlike in the result cache where they are
      scoped so nothing that could not have been affected loses a warm entry. A partial is a
      transient scale-run intermediate, so there is no warm-cache cost to pay for the simpler,
      stricter rule.
    * ``reach_purity_floor`` -- the same idea for `de_direction_reach`'s raw form, but carried
      as the VALUE rather than a counter, since here the semantics are a single number.

    Returns the payload rather than a digest so a mismatch can name the terms that differ.
    """
    from .catalog import CATALOG, _NAME_TO_CANONICAL, derived_policy
    from .metrics.direction import REACH_PURITY_FLOOR
    from .run import (_DE_RESULT_SEMANTICS, _GROUPED_SUM_REDUCTION_SEMANTICS,
                      _ONTARGET_EXCLUSION_SEMANTICS, _PDS_EXCLUSION_SEMANTICS,
                      effective_normalization, metric_agg)

    ordered = sorted(set(names))
    specs = {n: CATALOG.get(_NAME_TO_CANONICAL.get(n, n)) for n in ordered}
    return {
        "schema": _PARTIAL_SEMANTICS_SCHEMA,
        "metric_agg": [[n, metric_agg(n)] for n in ordered],
        "metric_derived": derived_policy(ordered),
        # `worst_value` is the v2 no-droppable-NaN fill (#89): `run._fill_no_drop` (run.py:444)
        # REPLACES a NaN per-perturbation value with it, so it moves emitted values, not just their
        # reduction. Missing from the first version of this payload -- codex-review.
        "metric_worst_value": [[n, (specs[n].worst_value if specs[n] is not None else None)]
                              for n in ordered],
        # `kind` routes a metric to the anndata or DE dispatch; `needs_moments` selects a DIFFERENT
        # cache artifact (pseudobulk_moments_* vs pseudobulk_*) and a different computation. Both
        # move what was computed while leaving agg/normalization/worst_value untouched
        # (codex-review).
        "metric_kind": [[n, (specs[n].kind if specs[n] is not None else None)] for n in ordered],
        "metric_needs_moments": [[n, bool(getattr(specs[n], "needs_moments", False))
                                  if specs[n] is not None else None] for n in ordered],
        "metric_normalization": [
            [n, (effective_normalization(specs[n], comparator) if specs[n] is not None else None)]
            for n in ordered
        ],
        "de_rank_semantics": _DE_RESULT_SEMANTICS,
        "pds_exclusion_semantics": _PDS_EXCLUSION_SEMANTICS,
        # #172: three scored `vcc2026` members stopped scoring each perturbation's own target
        # gene. Unconditional for the same reason as its two siblings above -- and note that
        # NEITHER `real_ref_fingerprint` nor `config_hash` nor `comparator` nor `metrics` can see
        # it, since all four describe what was asked for rather than what a metric means.
        "ontarget_exclusion_semantics": _ONTARGET_EXCLUSION_SEMANTICS,
        # The raw `direction_reach` purity floor. Unconditional like its three siblings, and by
        # VALUE rather than a counter: a partial directory straddling the 0.975 -> 0.9 move
        # reduces two different metrics under one name, and nothing else in this payload sees it.
        "reach_purity_floor": REACH_PURITY_FLOOR,
        # #271: `prep._grouped_sums` reduces WIDE, so the `bulk_lognorm` pseudobulk every
        # expression/PDS member reads moved. Unconditional like the three counters above -- a
        # partial is a transient scale-run intermediate, so there is no warm-cache cost to the
        # simpler rule -- and note that a partial directory straddling this change mixes pieces
        # whose bulks were rounded differently under one metric name. `comparator` in this payload
        # cannot see it: that records which comparator was asked for, not what its group sum means.
        "grouped_sum_reduction_semantics": _GROUPED_SUM_REDUCTION_SEMANTICS,
    }


def write_partial(df, out_dir, *, subset_id, meta) -> str:
    if subset_id != os.path.basename(subset_id) or subset_id in ("", ".", ".."):
        raise ValueError(f"subset_id must be a bare name, got {subset_id!r}")
    os.makedirs(out_dir, exist_ok=True)
    pq = os.path.join(out_dir, f"{subset_id}.parquet")
    df.write_parquet(pq)
    with open(os.path.join(out_dir, f"{subset_id}.json"), "w", encoding="utf-8") as fh:
        json.dump({"subset_id": subset_id, **meta}, fh, indent=2, sort_keys=True)
    return pq


def aggregate_partials(out_dir, *, reference_universe=None, reduce_nsig_spearman=False,
                       nsig_spearman_metric="de_wilcoxon_nsig_spearman",
                       nsig_real_metric="de_wilcoxon_nsig_counts_real",
                       nsig_pred_metric="de_wilcoxon_nsig_counts_pred"):
    sidecars = sorted(glob.glob(os.path.join(out_dir, "*.json")))
    if not sidecars:
        raise ValueError(f"no partial sidecars (*.json) found in {out_dir!r}")
    refs, cfgs, comparators, frames = set(), set(), set(), []
    metric_selections = set()
    all_have_metrics = True
    semantics, legacy_semantics, declared_semantics = {}, [], []
    for sc in sidecars:
        with open(sc, encoding="utf-8") as fh:
            meta = json.load(fh)
        ref, cfg_hash = meta.get("real_ref_fingerprint"), meta.get("config_hash")
        comparator = meta.get("comparator")
        if ref is None or cfg_hash is None or comparator is None:
            raise ValueError(
                f"partial sidecar {sc!r} is missing 'real_ref_fingerprint', 'config_hash', "
                "or 'comparator'; "
                "refusing to aggregate (the cross-partial safety guard would be bypassed)"
            )
        refs.add(ref)
        cfgs.add(cfg_hash)
        comparators.add(comparator)
        metrics = meta.get("metrics")
        if metrics is None:
            all_have_metrics = False
        else:
            metric_selections.add(tuple(sorted(metrics)))
        sem = meta.get(PARTIAL_SEMANTICS_KEY)
        if sem is None:
            legacy_semantics.append(os.path.basename(sc))
        else:
            semantics[json.dumps(sem, sort_keys=True)] = (os.path.basename(sc), sem)
            declared_semantics.append((os.path.basename(sc), metrics, comparator, sem))
        frames.append(pl.read_parquet(sc[:-5] + ".parquet"))
    if len(refs) > 1 or len(cfgs) > 1:
        raise ValueError(
            f"partials differ in reference/config (refs={refs}, configs={cfgs}); "
            "refusing to aggregate"
        )
    if len(comparators) > 1:
        raise ValueError(
            f"partials differ in comparator (comparators={comparators}); refusing to aggregate"
        )
    # Unconditional, NOT gated on `all_have_metrics`: two sidecars that declare DIFFERENT
    # selections are incompatible whatever a third one that predates the key does or does not
    # say. Only the decision to PASS a selection downstream needs every sidecar to agree
    # (Gemini, PR #262).
    if len(metric_selections) > 1:
        raise ValueError(
            f"partials differ in metric selections (metrics={sorted(metric_selections)}); "
            "refusing to aggregate"
        )
    # #246, LAST of the cross-partial guards on purpose. Each check above is more specific than
    # the next -- reference, then config, then comparator, then metric selection -- and each has a
    # message written for its own failure. A metric-selection difference also shows up as a
    # semantics difference (different names -> different payload), so running this first would
    # replace a precise message with a generic one.
    _check_result_semantics(out_dir, semantics, legacy_semantics, n_sidecars=len(sidecars),
                            declared=declared_semantics)
    full = pl.concat(frames, how="vertical")
    dup = full.group_by(["perturbation", "metric"]).len().filter(pl.col("len") > 1)
    if dup.height:
        raise ValueError(
            f"duplicate (perturbation,metric) across partials: {dup.head().to_dicts()}"
        )
    if reference_universe is not None:
        seen = set(full["perturbation"].unique().to_list())
        want = set(map(str, reference_universe))
        if seen != want:
            missing, extra = want - seen, seen - want
            raise ValueError(
                f"partition coverage mismatch: missing={sorted(missing)[:10]} "
                f"extra={sorted(extra)[:10]}"
            )
    if reduce_nsig_spearman:
        full = _inject_nsig_spearman(full, metric_name=nsig_spearman_metric,
                                     real_metric=nsig_real_metric, pred_metric=nsig_pred_metric)
    if all_have_metrics:
        return full, aggregate_metrics(full, metrics=next(iter(metric_selections)))
    return full, aggregate_metrics(full)


def _inject_nsig_spearman(full, *, metric_name,
                          real_metric="de_wilcoxon_nsig_counts_real",
                          pred_metric="de_wilcoxon_nsig_counts_pred"):
    """Reconstruct the cross-perturbation ``nsig_spearman`` scalar from the per-perturbation
    significant-gene counts every partial carries, then broadcast it to all perturbations.
    Matches ``metrics/de.py::de_nsig_spearman``: correlate over targets with a real count > 0
    (pred count filled 0 where absent). Any pre-existing ``metric_name`` rows are replaced."""
    real = full.filter(pl.col("metric") == real_metric).select(
        [pl.col("perturbation"), pl.col("value").alias("n_real")])
    pred = full.filter(pl.col("metric") == pred_metric).select(
        [pl.col("perturbation"), pl.col("value").alias("n_pred")])
    if real.height == 0:
        raise ValueError(
            f"{real_metric!r} rows absent from partials; cannot reconstruct {metric_name}")
    # A silently-missing pred count metric would let the left join fill every n_pred with 0.0,
    # masking the omission and yielding a bogus Spearman (PR #83 review).
    if pred.height == 0:
        raise ValueError(
            f"{pred_metric!r} rows absent from partials; cannot reconstruct {metric_name}")
    merged = real.join(pred, on="perturbation", how="left").with_columns(
        pl.col("n_pred").fill_null(0.0)).filter(pl.col("n_real") > 0)
    if merged.height == 0:
        value = 1.0
    else:
        corr = merged.select(
            pl.corr(pl.col("n_real"), pl.col("n_pred"), method="spearman")
        ).to_numpy().ravel()[0]
        value = float(corr) if corr is not None and corr == corr else float("nan")
    # Match compute_metrics' v2 no-droppable-NaN policy (run._fill_no_drop): an undefined
    # Spearman (NaN -- e.g. fewer than 2 targets with a real-significant gene) is filled with the
    # metric's catalog worst_value, so partitioned/streamed output equals whole-prediction
    # scoring. The partition path is v2-only, where that fill always applies.
    if value != value:  # NaN
        _spec = CATALOG.get(metric_name)
        if _spec is not None and _spec.worst_value is not None:
            value = float(_spec.worst_value)
    all_perts = full["perturbation"].unique().to_list()
    add = pl.DataFrame(
        {"perturbation": all_perts, "metric": [metric_name] * len(all_perts),
         "value": [value] * len(all_perts)},
        schema={"perturbation": pl.Utf8, "metric": pl.Utf8, "value": pl.Float64})
    return pl.concat([full.filter(pl.col("metric") != metric_name), add], how="vertical")
