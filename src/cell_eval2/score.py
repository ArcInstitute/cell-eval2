"""Native baseline-relative scoring with a penalty tail for unbounded error metrics.

`compat.score_agg_metrics` reproduces upstream cell-eval's clip-at-0 scoring
bit-for-bit and is FROZEN for parity. This module is its sibling: identical for
every submission at or below baseline, but for the unbounded error metrics
(``direction="lower"``: expr_mae/expr_mse/expr_mse_unbiased_capped_norm/delta_mae/delta_mse, plus
``de_*_lfc_nmae``) it replaces the flat clip with a graded penalty so a model
worse than the null baseline scores below 0.

That penalty comes in two shapes, and which one a metric takes is a per-metric policy, not a
property of this module. expr_mae/expr_mse/delta_mae/delta_mse carry ``scoring.ERROR``'s capped
Box-Cox tail; the ``de_*_lfc_nmae`` family carries ``scoring.ERROR_LINEAR`` -- the same -6.0
floor reached along a straight line instead of a quadratic (Alex, 2026-08-17); and
``expr_mse_unbiased_capped_norm`` carries neither preset but its own policy, whose declared
``clamp_low=0.0`` clips before its Box-Cox tail can show through. See the ``ERROR_LINEAR``
comment for why the lfc family was moved and why ``ERROR`` was not.

``expr_mse_unbiased_capped_norm`` is the one member that is also SIGNED, so its anchor does not
bound the score from above and its policy carries an explicit ``clamp_high``; see the
catalog comment on that entry.

The arithmetic lives in ``scoring.py``; this module resolves each column to its
``MetricSpec.scoring`` policy, applies the call-time overrides in spec 3.2's order, and
assembles the frame. The engine deliberately keeps one branch per anchored class rather
than one unified expression: float division does not reassociate, and each branch must
match its frozen ``compat`` kernel bit-for-bit.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
import logging
import math
import os

import numpy as np
import polars as pl

from . import competition
from .anchor import AnchorExpect, anchor_digest, resolve_anchor
from .catalog import CATALOG, _LFC_NMAE_METRICS, _NAME_TO_CANONICAL, is_decisive
from .scales import Scale, resolve_scales
from .scoring import Scoring, is_degenerate, score_one

logger = logging.getLogger(__name__)


def _canonical_overrides(overrides: dict[str, Scoring] | None) -> dict[str, Scoring]:
    """Resolve every ``overrides`` key to its canonical metric name (spec 3.2).

    A key may be written in any accepted spelling, exactly like an aggregate column, and the
    two must not have to agree: matching the raw key against the raw column name makes the
    SAME override honoured or ignored depending on which spelling the frame happens to
    carry -- ``{"discrimination_score_l1": policy}`` works against a v1-named column and
    silently does nothing against a ``pds_l1`` one.

    An unknown key raises rather than being ignored: a silently dropped override is a wrong
    number with nothing in the output saying so, which is the same failure class the
    degenerate-baseline raise exists to prevent. Two synonymous keys for one metric raise
    too -- dict order would otherwise decide which policy wins.
    """
    if not overrides:
        return {}
    resolved: dict[str, Scoring] = {}
    seen: dict[str, str] = {}
    for key, policy in overrides.items():
        # Validate the VALUE at the boundary too: a wrong type would otherwise surface as an
        # AttributeError on `policy.scored` deep in the scoring loop, pointing at the loop
        # rather than at the caller's dict.
        if not isinstance(policy, Scoring):
            raise TypeError(
                f"overrides[{key!r}] must be a Scoring, got {type(policy).__name__}"
            )
        canonical = _NAME_TO_CANONICAL.get(key, key)
        if canonical not in CATALOG:
            # List every ACCEPTED spelling, not just the canonical ones: the sentence above
            # promises aliases work, so a caller who mistyped `discrimination_score_l1`
            # would search a canonical-only list, fail to find it, and conclude the alias
            # was not supported after all.
            raise ValueError(
                f"unknown metric in overrides: {key!r}. Keys must name a catalog metric "
                f"in any accepted spelling (canonical, v1 alias, or other alias); "
                f"known: {sorted(_NAME_TO_CANONICAL)}"
            )
        if canonical in resolved:
            raise ValueError(
                f"overrides names metric {canonical!r} twice, as {seen[canonical]!r} and "
                f"{key!r}; pass exactly one key per metric"
            )
        resolved[canonical] = policy
        seen[canonical] = key
    return resolved


def _from_reference_column(row_names, u_vals, metric_names, lfc_nmae_ref):
    """The `from_reference` column: (1 - nmae) / (1 - nmae_ref_raw), null everywhere else.

    Deliberately NOT enrolled in avg_score. avg_score keeps averaging `from_baseline` over
    the enrolled metrics, so passing a reference cannot move any existing profile's score.

    The two columns have DIFFERENT ZEROS and that is not a defect. `from_baseline` is
    1 - u/b against whatever baseline was published (a generic-response baseline reads
    nmae ~ 0.96), while this is measured against silence and a replicate: an ALL-ZERO
    predicted-LFC table is exactly 0 here and about -0.04 there. They answer different
    questions. ⚠️ "all-zero predicted LFC", not "a no-change submission" -- under
    `control_source="real"` a submission emitting the control need not produce that table
    (#286, `docs/metrics.md` §4.3).
    """
    n = len(row_names)
    # Captured BEFORE the read, or the warnings below cannot name where the reference came
    # from -- which is the whole point of naming it when a run has several contexts.
    source = str(lfc_nmae_ref) if isinstance(lfc_nmae_ref, (str, os.PathLike)) else "<frame>"
    if isinstance(lfc_nmae_ref, (str, os.PathLike)):
        lfc_nmae_ref = pl.read_csv(lfc_nmae_ref)
    # MALFORMED is a caller error and raises. EMPTY -- a NULL nmae_ref_raw with
    # n_perturbations == 0, the one shape `_empty_agg()` produces -- is a data outcome and
    # must not raise: it leaves this column null and lets every other metric score (spec
    # 4.4). Validate the SCHEMA explicitly rather than by attribute access, so a two-column
    # frame does not sail through and a missing `statistic` raises on its own terms instead
    # of surfacing as an incidental polars error.
    required = {"statistic", "nmae_ref_raw", "n_perturbations"}
    if missing_cols := required - set(lfc_nmae_ref.columns):
        raise ValueError(
            f"lfc_nmae_ref ({source}) is missing column(s) {sorted(missing_cols)}; got "
            f"{lfc_nmae_ref.columns}. Pass the lfc_nmae_ref_agg.csv written by "
            "`run --lfc-nmae-ref`."
        )
    mean_row = lfc_nmae_ref.filter(pl.col("statistic") == "mean")
    if mean_row.height != 1:
        raise ValueError(
            f"lfc_nmae_ref ({source}) must have exactly one 'mean' row, found "
            f"{mean_row.height}"
        )
    ref, n_perts = mean_row["nmae_ref_raw"][0], mean_row["n_perturbations"][0]
    # `if n_perts:` would accept None and "" as if they were zero. The contract says
    # EXACTLY zero, so validate the count itself before comparing.
    if not isinstance(n_perts, (int, np.integer)) or isinstance(n_perts, bool):
        raise ValueError(
            f"lfc_nmae_ref ({source}) has a non-integer n_perturbations: {n_perts!r}"
        )
    if ref is None:
        # A null WITH a non-zero count is inconsistent, not empty.
        if n_perts != 0:
            raise ValueError(
                f"lfc_nmae_ref ({source}) has a null nmae_ref_raw but n_perturbations="
                f"{n_perts!r}; a null is only meaningful when nothing was scoreable."
            )
        logger.warning(
            "lfc_nmae reference %s scored no perturbations: leaving from_reference null. "
            "Every other metric is scored normally.", source,
        )
        return pl.Series("from_reference", [None] * n, dtype=pl.Float64)
    # The contract is an EQUIVALENCE: null <-> n_perturbations == 0. The null side is
    # enforced above; this is the other direction. A finite mean over "no perturbations" is
    # self-contradictory, and a negative count is corrupt however it is spelled.
    if n_perts <= 0:
        raise ValueError(
            f"lfc_nmae_ref ({source}) has a non-null nmae_ref_raw ({ref!r}) but "
            f"n_perturbations={n_perts!r}; a mean over no perturbations is not a number."
        )
    ref = float(ref)
    # NaN/inf are NOT "empty" -- empty is defined as null. A negative mean of non-negative
    # ratios is impossible. Both are corrupt input, so both raise rather than degrade.
    if not math.isfinite(ref):
        raise ValueError(f"lfc_nmae_ref ({source}) has a non-finite nmae_ref_raw: {ref!r}")
    if ref < 0.0:
        raise ValueError(
            f"lfc_nmae_ref ({source}) has a negative nmae_ref_raw ({ref!r}); it is a mean of "
            "non-negative ratios and cannot be below zero."
        )
    by_name = dict(zip(metric_names, u_vals))
    vals: list[float | None] = []
    for name in row_names:
        canon = _NAME_TO_CANONICAL.get(name, name)
        if canon not in _LFC_NMAE_METRICS:
            vals.append(None)
            continue
        u = by_name.get(name)
        if u is None or not math.isfinite(float(u)):
            vals.append(None)
            continue
        numerator = 1.0 - float(u)
        denominator = 1.0 - ref
        if denominator <= 0.0:
            # A non-positive denominator would INVERT the ranking silently. Report the
            # unrescaled numerator, which is still a valid "how far from silence" number.
            logger.warning(
                "lfc_nmae reference %s is degenerate (mean nmae_ref_raw = %.4f >= 1): "
                "reporting the UNRESCALED 1 - nmae for %r rather than dividing by a "
                "non-positive denominator, which would invert the ranking. The value is "
                "not comparable across cell lines.", source, ref, name,
            )
            vals.append(numerator)
            continue
        vals.append(numerator / denominator)
    return pl.Series("from_reference", vals, dtype=pl.Float64)


def _reference_column(row_names, u_vals, metric_names, entries, *, column, label):
    """Score every metric a reference table NAMES, and average over exactly those.

    The shared core behind a frozen scale's column and `from_replicate` (#276 part C). The
    two differ only in where `(base, Scoring)` comes from -- a frozen registry, or a measured
    baseline plus a measured anchor -- and both need identical membership rules, because both
    print a name on a column whose `avg_score` is a mean over a set the reader cannot see.

    `label` names the reference in every error message; `column` is the Series name.

    Rows the table does not name are null, exactly as `from_reference` leaves every row but
    its two. A metric the table NAMES but the aggregate lacks RAISES -- the opposite of the
    null-for-unnamed rule, deliberately: an unnamed metric going null is harmless, but a named
    one going missing would quietly redefine this column's `avg_score` over fewer metrics.
    """
    # TWO columns resolving to one metric -- e.g. `pds_cosine` beside its v1 spelling
    # `discrimination_score_cosine` -- must raise, and the check is restricted to the metrics
    # THIS reference names so a malformed column elsewhere in a wide profile is not this
    # function's business. Measured on the collision it prevents: the reference scored BOTH
    # rows, both took the LAST column's value rather than the canonical one, and the duplicate
    # entered the mean twice, moving avg_score 0.7333 -> 0.5714. Silent double-weighting of
    # one metric is exactly the denominator change the missing-metric raise below exists to
    # stop, arriving from the other direction.
    by_canon: dict[str, object] = {}
    spelt: dict[str, str] = {}
    for name, u in zip(metric_names, u_vals):
        canon = _NAME_TO_CANONICAL.get(name, name)
        if canon in by_canon and canon in entries:
            raise ValueError(
                f"{label}: the aggregate carries two columns that both name {canon!r} -- "
                f"{spelt[canon]!r} and {name!r}. Scoring it would weight that metric twice "
                f"in {column}'s avg_score and take whichever column came last. Emit one "
                "column per metric."
            )
        by_canon[canon] = u
        spelt[canon] = name
    if missing := sorted(m for m in entries if m not in by_canon):
        raise ValueError(
            f"{label} names metric(s) {missing} that are absent from the aggregate being "
            f"scored (it carries {sorted(by_canon)}). Scoring only part of it would redefine "
            f"{column}'s avg_score over fewer metrics while still reporting that column; "
            "score a profile that covers the reference, or drop it."
        )
    # Presence in the AGGREGATE is not presence in the OUTPUT. A metric the baseline pass
    # declined to score (`overrides={m: Scoring(scored=False)}`, or a non-decisive metric
    # dropped for a degenerate baseline) still has an aggregate COLUMN but no output ROW, so
    # the check above passes and the loop below would quietly average the survivors -- a
    # five-member competition score under a six-member anchor, with nothing to see. The scale
    # path avoids this by RESTORING such rows before calling in (score_metrics); this raise is
    # the backstop for any caller that does not.
    rows = [_NAME_TO_CANONICAL.get(r, r) for r in row_names]
    if absent := sorted(m for m in entries if m not in rows):
        raise ValueError(
            f"{label} names metric(s) {absent} that have no output row: they are in the "
            f"aggregate but the baseline pass did not score them, so {column}'s avg_score "
            "would silently cover fewer metrics than the reference names."
        )
    if not row_names or row_names[-1] != "avg_score":
        raise ValueError(
            f"{label}: the last row must be 'avg_score' (got "
            f"{row_names[-1] if row_names else None!r}); {column}'s average is written there "
            "by position."
        )
    vals: list[float | None] = []
    scored: list[float] = []
    for name in row_names:
        canon = _NAME_TO_CANONICAL.get(name, name)
        entry = entries.get(canon)
        if entry is None:                      # an unnamed metric, and the avg_score row
            vals.append(None)
            continue
        s = score_one(by_canon[canon], entry.base, entry.scoring)
        vals.append(s)
        scored.append(s)
    if not scored:
        raise ValueError(f"{label} scored nothing: none of its metrics reached a row")
    vals[-1] = float(np.mean(scored))
    return pl.Series(column, vals, dtype=pl.Float64)


def _scale_column(row_names, u_vals, metric_names, scale: Scale) -> pl.Series:
    """One scale's column: ``score_one`` against the scale's constant base, per metric.

    Deliberately NOT the ``from_baseline`` loop. Three differences make sharing it wrong:
    the base comes from the scale rather than the baseline frame, the policy comes from the
    scale rather than the catalog, and the global penalty/clamp arguments must NOT reach it
    -- those knobs belong to baseline scoring, and letting ``--penalty-cap`` move a frozen
    scale would make the digest in ``tests/test_scales.py`` a lie.

    Everything else is `_reference_column`, which `from_replicate` shares (#276 part C).
    """
    return _reference_column(row_names, u_vals, metric_names, scale.entries,
                             column=scale.name, label=f"scale {scale.name!r}")


def _cached_bundle(descriptor):
    """The cached anchor, or None. NEVER computes.

    `score` has no AnnData, so it cannot fill the cache; a miss must reach
    `resolve_anchor`'s "no anchor available" raise rather than a producer (spec 4.4). The
    descriptor is `run_meta.json`'s `anchor_cache` block -- the exact
    `(key, fingerprint, params, kind)` quadruple `CacheStore.get` needs, none of which can
    be derived from a root plus an `AnchorExpect`.
    """
    from .anchor import _BadBundle, _bundle_from_obj
    from .cache import MISS, CacheStore
    if not descriptor:
        return None
    # Named, not a KeyError with a stack trace (Copilot, PR #284). The descriptor comes from
    # `run_meta.json`, so it can be hand-edited or written by an older version -- a truncated
    # one is bad INPUT and must say which key is missing.
    if missing := [k for k in ("root", "key", "fingerprint", "params") if k not in descriptor]:
        raise ValueError(
            f"the anchor cache descriptor is missing {missing}; it must carry the exact "
            "(root, key, fingerprint, params) coordinates `CacheStore.get` needs. It is "
            "written by `run --anchor --cache-real`; rebuild it rather than editing "
            "run_meta.json by hand."
        )
    obj = CacheStore(descriptor["root"]).get(
        descriptor["key"], fingerprint=descriptor["fingerprint"],
        params=descriptor["params"], kind=descriptor.get("kind", "json"))
    if obj is MISS:
        return None
    try:
        return _bundle_from_obj(obj)
    except _BadBundle as exc:
        # A damaged cache must not abort a run that could still use a supplied anchor -- but
        # it must be DIAGNOSABLE (Copilot, PR #284). Returning None silently turns cache
        # corruption into the generic "no anchor available", which sends the reader looking
        # for a missing file rather than a broken one. `score` cannot repair it the way
        # `cached_anchor` does (it has no AnnData to recompute from), so warn and move on.
        logger.warning(
            "the cached anchor at %s is unusable (%s); ignoring it. `score` cannot recompute "
            "an anchor -- rebuild it with `run --anchor --cache-real`, or pass one with "
            "--anchor.", descriptor.get("root"), exc,
        )
        return None


def expect_from_run_meta(meta: dict) -> AnchorExpect:
    """This run's anchor expectations, read from its own ``run_meta.json``.

    FAIL-CLOSED on a missing key, with the key named -- `_check_baseline_config`
    (cli.py:120) states exactly this rule, and `.get()` would let an empty JSON object pass
    as fully verified.

    The gate is the STRICT content hash. `build_run_meta` computes `source_fingerprint` at
    ``strict=cfg.cache_strict``, FALSE by default (deliberately, so `run` never materializes
    X on the hot path), so a default run carries the metadata hash -- under which two
    datasets with identical shape, dtype, gene names and per-cell labels but different `X`
    are indistinguishable. Refuse and name the flag rather than silently weakening the gate.
    """
    required = ("source_fingerprint", "source_fingerprint_strict", "cell_eval2_version",
                "anchor_semantic_identity", "anchor_metric_names")
    for field in required:
        if field not in meta:
            raise ValueError(
                f"run_meta.json is missing {field!r}, so this run's anchor expectations "
                "cannot be built and a supplied or cached anchor could only be validated "
                "against itself. The anchor fields are recorded on a best-effort basis: "
                "they are omitted when the run's DE backend cannot be resolved, which "
                "happens for a run that supplies both DE tables and needs no engine. "
                "Re-run `cell-eval2 run` on a host where that backend is available to "
                "regenerate them."
            )
    if not meta["source_fingerprint_strict"]:
        raise ValueError(
            "this run's source_fingerprint is metadata-only (source_fingerprint_strict is "
            "false), and the anchor gate is the strict CONTENT hash: two datasets with "
            "identical shape, dtype, gene names and per-cell labels but different X would "
            "validate as the same anchor. Re-run both sides with `--cache-strict`."
        )
    return AnchorExpect(fingerprint=meta["source_fingerprint"],
                        semantic_identity=meta["anchor_semantic_identity"],
                        version=meta["cell_eval2_version"],
                        metrics=tuple(meta["anchor_metric_names"]))


def _replicate_entries(base_by_name, anchor_frame):
    """The replicate scale as `(base, Scoring)` pairs: 0 = baseline, 1 = replicate.

    #276 comment 1 fixes the scale; part C makes it POLICY-APPLIED. `score_one` with
    `policy.anchor` set to the measured replicate has `(u - b) / (r - b)` as its unclamped
    linear core for both directions, so the replicate scale needs no arithmetic of its own and
    inherits the clamps, the penalty SHAPE (a Box-Cox tail, or `de_*_lfc_nmae`'s straight line)
    and `is_degenerate` from the catalog. There is
    deliberately no second policy table: one `Scoring` per metric serves both scales.

    Membership is the anchor's rows INTERSECTED with the scored catalog metrics: the anchor
    covers all ten `vcc2026` names, and the four diagnostics (`scored=False`) are not part of
    any average.

    Both maps are keyed CANONICALLY. The anchor and the aggregate may legitimately spell one
    metric differently (`pds_cosine` vs `discrimination_score_cosine`), so a raw-string lookup
    would miss a baseline that is present and silently omit the member.
    """
    from .scales import ScaleEntry

    base_by_canon = {_NAME_TO_CANONICAL.get(k, k): v for k, v in base_by_name.items()}
    entries: dict[str, ScaleEntry] = {}
    seen: dict[str, str] = {}
    for name, rep in zip(anchor_frame["metric"].to_list(),
                         anchor_frame["replicate"].to_list()):
        canon = _NAME_TO_CANONICAL.get(name, name)
        # Two spellings of one metric in ONE anchor is a corrupt artifact, not a preference:
        # a dict would take whichever came last, silently choosing a replicate value.
        if canon in seen:
            raise ValueError(
                f"the anchor names metric {canon!r} twice, as {seen[canon]!r} and {name!r}; "
                "one row per metric -- otherwise the replicate value is whichever row came "
                "last."
            )
        seen[canon] = name
        spec = CATALOG.get(canon)
        if spec is None:
            logger.warning("the anchor names metric %r, which is not in the catalog; "
                           "leaving from_replicate null for it", name)
            continue
        if not spec.scoring.scored:
            continue                      # a diagnostic: stamped in the anchor, never averaged
        b, problem, policy = base_by_canon.get(canon), None, None
        if b is None or rep is None:
            problem = f"baseline={b!r}, replicate={rep!r}: one of the two ends is missing"
        elif not (math.isfinite(float(b)) and math.isfinite(float(rep))):
            problem = f"baseline={b!r}, replicate={rep!r}: an end is not finite"
        if problem is None:
            # `allow_negative_baseline` is an ANCHORLESS-only flag (scoring.py:74-76): with an
            # anchor the baseline's side is checkable, so the flag is meaningless and
            # `__post_init__` refuses the pair. Clearing it is the correct resolution -- the
            # alternative, catching the raise, would silently drop `direction_yield` from every
            # anchored `full` run.
            try:
                policy = replace(spec.scoring, anchor=float(rep), allow_negative_baseline=False)
            except ValueError as exc:
                # `replace` re-runs `__post_init__`, which can reject the PAIR rather than
                # either end on its own -- `metric_min` must sit on the worse side of the
                # anchor, so a malformed anchor carrying a negative replicate for a [0, 1]
                # metric raises here. `anchor.py` validates finiteness only, so a supplied or
                # in-memory frame can reach this. Route it through the curated handling below
                # instead of letting a raw `Scoring` message escape a function whose whole
                # job is to say "this metric has no usable replicate scale" (codex round 1).
                problem = f"replicate={rep!r} is not a usable anchor for this policy: {exc}"
        if problem is None and is_degenerate(float(b), policy):
            problem = (f"the anchor ({float(rep):.6g}) leaves no usable headroom over the "
                       f"baseline ({float(b):.6g})")
        if problem is not None:
            msg = f"metric {canon!r} has no usable replicate scale: {problem}"
            if is_decisive(spec):
                raise ValueError(
                    f"{msg}. Every vcc2026 member is decisive, so scoring every submission "
                    "against an undefined scale is worse than stopping -- rebuild the anchor "
                    "rather than reporting the column."
                )
            logger.warning("%s; leaving from_replicate null for it and scoring the rest.", msg)
            continue
        entries[canon] = ScaleEntry(base=float(b), scoring=policy)
    return entries


def _insert_metric_rows(out: pl.DataFrame, canonical: list[str]) -> pl.DataFrame:
    """Add rows for metrics the baseline pass did not score, keeping the frame's contract.

    Every existing column is null-filled for the new rows -- `from_baseline` genuinely has no
    value for them, and a reference column that names them fills its own cell afterwards.
    Placement follows the frame's convention (lower-is-better, then higher-is-better) and the
    `avg_score` row stays LAST, which `_reference_column` requires and asserts.

    A metric already present is skipped rather than duplicated, so a scale and an anchor
    asking for the same restored row produce ONE row.

    ⚠️ It NORMALIZES even when there is nothing to add, which is why the caller invokes it
    unconditionally. A requested scale restores its own dropped rows earlier, on the
    metric/score lists (`score.py:594-618`), by APPENDING -- so a frame can arrive here already
    carrying the row but in an order `member_order_in_frame` does not describe. Sorting is a
    no-op on any frame that never needed restoring, because `aggregate_metrics_wide` already
    yields each group in name order.
    """
    present = {_NAME_TO_CANONICAL.get(m, m) for m in out["metric"].to_list()}
    add = [c for c in dict.fromkeys(canonical) if c not in present]
    lower = sorted(c for c in add if CATALOG[c].scoring.direction == "lower")
    higher = sorted(c for c in add if CATALOG[c].scoring.direction != "lower")
    new = pl.DataFrame(
        {"metric": lower + higher,
         **{c: [None] * len(add) for c in out.columns if c != "metric"}},
        schema={c: out.schema[c] for c in out.columns},
    )
    # WITHIN each group, and SORTED within it. `[body, new, tail]` would yield
    # (existing lower, existing higher, new lower, new higher) -- a restored lower-is-better
    # metric landing after the higher-is-better block. Appending within the group is not
    # enough either: `competition_payload`'s `member_order_in_frame` freezes each group as
    # SORTED, and `aggregate_metrics_wide` sorts metric columns by name, so a restored metric
    # appended after the existing ones would put the frame in an order the frozen rule does
    # not describe.
    # ⚠️ `map_elements` deliberately, and both hosted reviewers flagged it on PR #290 (Copilot
    # and Gemini independently, both suggesting a precomputed `is_in` set). Kept for two
    # reasons. This frame is ONE ROW PER METRIC -- at most ~15 in the widest profile -- so the
    # per-row Python callback is not a hot path in any measurable sense. And the callback RAISES
    # `KeyError` on a metric the catalog does not know, where an `is_in` set would silently
    # classify it as higher-is-better and place its row in the wrong direction group; a loud
    # failure is the better half of that trade in a function whose whole job is row placement.
    is_lower = pl.col("metric").map_elements(
        lambda m: CATALOG[_NAME_TO_CANONICAL.get(m, m)].scoring.direction == "lower",
        return_dtype=pl.Boolean)
    body = out.filter(pl.col("metric") != "avg_score")
    tail = out.filter(pl.col("metric") == "avg_score")
    lo = pl.concat([body.filter(is_lower), new.filter(is_lower)], how="vertical") \
           .sort("metric")
    hi = pl.concat([body.filter(~is_lower), new.filter(~is_lower)], how="vertical") \
           .sort("metric")
    return pl.concat([lo, hi, tail], how="vertical")


def score_metrics(
    results_user: pl.DataFrame | str | os.PathLike,
    results_base: pl.DataFrame | str | os.PathLike | None = None,
    output: str | os.PathLike | None = None,
    comparison_statistic: str = "mean",
    *,
    penalty_exponent: float | None = None,   # was DEFAULT_PENALTY_EXPONENT
    penalty_cap: float | None = None,        # was DEFAULT_PENALTY_CAP
    clamp_low: float | None = None,
    clamp_high: float | None = None,
    penalty: str | None = None,
    overrides: dict[str, Scoring] | None = None,
    lfc_nmae_ref: pl.DataFrame | str | os.PathLike | None = None,
    scale: str | Sequence[str] | None = None,
    anchor: str | os.PathLike | None = None,
    anchor_cache: dict | None = None,
    anchor_expect: "AnchorExpect | None" = None,
    real_bundle: str | os.PathLike | None = None,
    user_meta: dict | None = None,
    diagnostic_supplied_de_pred: bool = False,
) -> pl.DataFrame:
    """Score user aggregate metrics against a baseline, penalizing unbounded error
    metrics that exceed the baseline instead of clipping them to 0.

    Same ``{metric, from_baseline}`` + ``avg_score`` frame shape and row order
    (lower-is-better metrics, then higher-is-better ones, then ``avg_score``) as
    ``compat.score_agg_metrics``, and IDENTICAL to it -- under the catalog defaults, with
    no overrides -- whenever no error metric exceeds its baseline AND no baseline is
    degenerate. The override arguments below exist precisely to change the numbers, so any
    of them may break that identity deliberately.

    A degenerate baseline -- one whose denominator ``D`` is not a FINITE positive number, or
    whose ``base`` is missing/non-finite -- FAILS LOUD for every DECISIVE metric (spec 6;
    ``catalog.is_decisive``: anything v1 can emit or that ``vcc`` or ``vcc2026`` scores), including
    ``base >= 1`` on an anchor-1 metric, which used to score every submission exactly 0,
    silently. Any other scored metric warns and is excluded from ``avg_score`` instead. Every
    ``vcc2026`` member now reaches the raise, so a degenerate baseline cannot silently change
    that competition average's denominator. If the exclusions leave NOTHING scoreable it raises,
    rather than returning the
    fallback ``avg_score = 0.0``, which reads as "equals the baseline".

    ``penalty_exponent`` / ``penalty_cap`` / ``clamp_low`` / ``clamp_high`` / ``penalty``
    are global overrides; ``None`` means "not supplied at this level", which is what makes
    spec 3.2's order expressible -- with a VALUE default, "caller passed 2.0" and "caller
    passed nothing" would be the same call. ``overrides={name: Scoring}`` replaces one
    metric's policy WHOLESALE and therefore suppresses all five globals for it.

    ``scale`` names one or more frozen scales (``cell_eval2.scales.SCALES``) and appends one
    column per scale, each holding ``score_one`` against that scale's CONSTANT base rather
    than against ``results_base``. It never changes ``from_baseline`` or its ``avg_score``:
    a scale only ever adds a column. The five global knobs above are NOT applied to a scale
    column -- a scale carries its own frozen policy, and letting a CLI flag move it would
    make the registry's digest meaningless.

    ``diagnostic_supplied_de_pred`` permits a submission whose pred-side DE table was
    SUPPLIED (``run --de-pred``) to be scored against a bundle, and TAKES ITS ENROLMENT for
    doing so (#291): the result gets exactly the treatment a diagnostic bundle's does --
    ``from_replicate`` is reported, ``from_baseline`` keeps its ``avg_score``, and no
    competition average is claimed. It exists because ``--de-pred`` is the isolator the
    metric campaigns are built on, and those arms want the bundle's scale; it is not an
    escape hatch from the peer comparison, which has none.
    """
    # #276 part C. FIRST, before the "nothing to score against" guard below: a bundle-only
    # call arrives with results_base=None and no scale, which that guard rejects. A bundle
    # supplies BOTH ends plus the identity that binds them, so it is exclusive with the loose
    # arguments rather than layered over them.
    rb_manifest, rb_waivers = None, []
    # REFUSED rather than ignored, like `anchor_expect` above: the flag only has meaning
    # against a bundle's submission gate, and a caller who passed it on the loose path is
    # asking for a downgrade that path never applies. Silently accepting it would let a
    # harness believe it had marked a run diagnostic when nothing did.
    if diagnostic_supplied_de_pred and real_bundle is None:
        # PARAMETER names, not CLI flags: this is the library API, and the CLI never reaches
        # it -- `cli.py`'s score dispatch raises its own SystemExit naming `--real-bundle`
        # before `diagnostic_supplied_de_pred` is ever put into the kwargs (Copilot round 2,
        # suppressed). Same convention as the `results_base`/`anchor`/`anchor_cache` refusal
        # below.
        raise ValueError(
            "diagnostic_supplied_de_pred applies only to real_bundle: it downgrades a bundle "
            "ENROLMENT, and scoring against results_base/anchor has no enrolment to "
            "downgrade. Drop it; a supplied pred-side DE table needs no permission there."
        )
    if real_bundle is not None:
        from .real_bundle import check_submission, manifest_digest, read_real_bundle

        if results_base is not None or anchor is not None or anchor_cache is not None:
            raise ValueError(
                "real_bundle supplies both ends of the scale (the baseline aggregate and the "
                "replicate anchor), so it cannot be combined with results_base, anchor or "
                "anchor_cache. Pass the bundle alone."
            )
        # ⚠️ `anchor_expect` too, and this one is a GATE BYPASS rather than a redundancy. The
        # expectation is what ties the anchor to THIS submission (`anchor_semantic_identity`,
        # `anchor_metric_names`, the strict fingerprint); a caller who supplies one built from
        # the bundle's own metadata would have the artifact validate against itself, which
        # passes for any artifact whatsoever. Refusing is better than silently overriding:
        # a caller who passed it deserves to know it was ignored.
        if anchor_expect is not None:
            raise ValueError(
                "real_bundle builds anchor_expect from the submission's own run_meta, so a "
                "caller-supplied anchor_expect cannot be honoured: an expectation derived "
                "from anywhere but the submission would let the bundle validate against "
                "itself. Drop anchor_expect, or use the --anchor path."
            )
        if user_meta is None:
            raise ValueError(
                "scoring against a real bundle requires user_meta -- the submission's "
                "run_meta.json. Enrolment is an affirmative claim, and without the "
                "submission's own identity the bundle could only be checked against itself, "
                "which passes for any artifact whatsoever."
            )
        bundle = read_real_bundle(real_bundle)
        rb_manifest = bundle.manifest
        # #291. A non-empty return is a waiver TAKEN, and it disables enrolment below -- the
        # opt-in buys the run, not the label.
        rb_waivers = check_submission(
            rb_manifest, user_meta,
            diagnostic_supplied_de_pred=diagnostic_supplied_de_pred)
        results_base = bundle.baseline_agg
        anchor = bundle.root
        anchor_expect = expect_from_run_meta(user_meta)      # unconditional: see above

    # ⚠️ SCOPED TO THE ANCHOR, and evaluated HERE rather than inside the anchor block below.
    # It fires for ANY anchor (supplied, cached or from a bundle) because the reason is a
    # property of the anchor itself: `replicate` is a mean over the five split aggregates.
    # Beside the `lfc_nmae_ref` guard it would refuse `median` for every ordinary baseline
    # run, anchored or not -- so it stays anchor-scoped. But it cannot wait for the anchor
    # block: `aggregate_metrics_wide` gives a DERIVED metric a value at `mean` only and NaN
    # elsewhere, so a vcc2026 pair read at the median raises `degenerate baseline` in the
    # BASELINE pass first (measured), and a guard placed later never runs.
    if (anchor is not None or anchor_cache is not None) \
            and comparison_statistic != competition.COMPARISON_STATISTIC:
        raise ValueError(
            f"an anchor requires comparison_statistic="
            f"{competition.COMPARISON_STATISTIC!r} (got {comparison_statistic!r}): the "
            "anchor's `replicate` is a mean over the split aggregates, so any other "
            "statistic divides one quantity's gap by another's span. Drop the anchor or "
            "read the mean row."
        )

    # `<= 0` alone lets inf and NaN through: an infinite cap becomes an infinite floor and a
    # NaN cap a NaN one, either of which poisons avg_score. The hole is pre-existing; spec 3.1
    # already forbids the same values in the catalog, so closing it here makes the two agree.
    for _name, _val in (("penalty_exponent", penalty_exponent), ("penalty_cap", penalty_cap)):
        if _val is not None and (not math.isfinite(_val) or _val <= 0):
            raise ValueError(f"{_name} must be finite and > 0, got {_val!r}")
    # Before reading the frames, so a mistyped key fails on its own terms rather than
    # depending on whether the aggregate happens to contain that column.
    overrides = _canonical_overrides(overrides)
    scales = resolve_scales(scale)
    if results_base is None and not scales:
        raise ValueError(
            "score_metrics has nothing to score against: pass results_base (a baseline "
            "aggregate) or scale= (a named scale, which carries its own constant reference "
            "points and needs no baseline)."
        )
    if results_base is not None:
        if isinstance(results_user, (str, os.PathLike)):
            results_user = pl.read_csv(results_user)
        if isinstance(results_base, (str, os.PathLike)):
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
        # The scaled score is a ratio of two MEANS (spec 4.1). `u_vals` below comes from the
        # SELECTED statistic while the reference frame only carries a mean, so any other choice
        # would compute e.g. (1 - user_std)/(1 - ref_mean) and label it a rescaled score.
        if lfc_nmae_ref is not None and comparison_statistic != "mean":
            raise ValueError(
                f"lfc_nmae_ref requires comparison_statistic='mean' (spec 4.1 is a ratio of "
                f"means); got {comparison_statistic!r}. Drop the reference or use the mean."
            )

        u_row = results_user.filter(pl.col("statistic") == comparison_statistic).drop("statistic")
        b_row = results_base.filter(pl.col("statistic") == comparison_statistic).drop("statistic")
        metric_names = u_row.columns
        u_vals = u_row.row(0)
        b_vals = b_row.row(0)

        metrics_zero, scores_zero = [], []
        metrics_one, scores_one = [], []
        skipped: list[str] = []
        for name, uv, bv in zip(metric_names, u_vals, b_vals):
            spec = CATALOG.get(_NAME_TO_CANONICAL.get(name, name))
            if spec is None:
                logger.warning("metric %r not scored (unknown)", name)
                continue
            # Precedence (spec 3.2), listed LEAST specific first -- catalog, then global, then
            # per-metric -- and the most specific one that applies WINS, so a per-metric
            # override beats a global argument, which beats the catalog policy. A
            # per-metric override REPLACES the policy wholesale (it is a validated Scoring), so
            # it also suppresses the globals for ALL FIVE knobs -- letting a global penalty_cap
            # leak into a metric whose entire policy was replaced is the same precedence
            # inversion one level down. The globals are passed to score_one as ARGUMENTS, never
            # via replace(), which re-runs __post_init__ and would reject an infinite floor.
            # Both sides are canonical here -- the keys were normalized on entry -- so one
            # lookup answers for every accepted spelling of the column AND of the key.
            policy = overrides.get(spec.name, spec.scoring)
            has_override = spec.name in overrides
            eff_low = None if has_override else clamp_low
            eff_high = None if has_override else clamp_high
            eff_pen = None if has_override else penalty
            eff_exp = None if has_override else penalty_exponent
            eff_cap = None if has_override else penalty_cap
            if not policy.scored:
                logger.warning("metric %r not scored (scored=False)", name)
                continue
            # The SAME knobs `score_one` is called with below. `is_degenerate` self-resolves
            # when they are omitted, which would let the predicate and the scorer disagree
            # about an unfloored policy: a call-time `clamp_low=0.0` makes an overflowing
            # sentinel harmless, and a call-time `clamp_low=-inf` makes a floored one
            # unusable (codex round 2, finding 1).
            if is_degenerate(bv, policy, clamp_low=eff_low, clamp_high=eff_high,
                             penalty_exponent=eff_exp, penalty_cap=eff_cap,
                             penalty=eff_pen):
                # Fail loud only where a wrong number decides something. For a metric v1 can emit
                # or that either competition profile (`vcc` or `vcc2026`) scores, a degenerate
                # baseline is a corrupt input and scoring every submission against an undefined
                # denominator is worse than stopping. Other metrics can still reach the
                # warn-and-exclude path. One has a baseline that is legitimately degenerate:
                #   * `de_*_direction_yield` (full/de) is signed and centred at zero BY
                #     CONSTRUCTION -- it returns exactly 0.0 when n_pred == 0, so a baseline that
                #     calls nothing for most perturbations lands on exactly 0.0 legitimately.
                #     It aggregated by median through v0.7.0 and by mean since #231; the
                #     rationale is unchanged either way, because per-perturbation zeros
                #     mean-aggregate to zero just as they median-aggregate to zero.
                # `expr_mse_unbiased_capped_norm` does not reach this branch: the old objection used
                # 65.7% negative perturbations from an ACCURATE technical replicate, not the
                # deployed generic-response comparator. The latter measures 169.16 / 197.05, its
                # worst perturbation remains 17x above 0, and 0% land at or below the boundary.
                # It is therefore decisive through `vcc2026`, including when selected by
                # full/anndata. #222 remains open for a profile-aware predicate if a future,
                # stronger comparator makes that distinction necessary.
                msg = (
                    f"degenerate baseline for metric {name!r}: base={bv!r}. The baseline must "
                    f"define a finite positive scale (anchor={policy.anchor!r}, "
                    f"direction={policy.direction!r})"
                )
                if is_decisive(spec):
                    raise ValueError(
                        f"{msg}; fix the baseline file rather than scoring every submission "
                        "against an undefined denominator."
                    )
                logger.warning(
                    "%s; EXCLUDING it from avg_score (not decisive for v1, vcc, or vcc2026) "
                    "and scoring the "
                    "rest. avg_score is therefore a mean over a different metric set than a run "
                    "where this metric was scoreable, and the two are not directly comparable.",
                    msg,
                )
                skipped.append(name)
                continue
            score = score_one(uv, bv, policy,
                              penalty_exponent=eff_exp, penalty_cap=eff_cap,
                              clamp_low=eff_low, clamp_high=eff_high, penalty=eff_pen)
            if policy.direction == "lower":
                metrics_zero.append(name)
                scores_zero.append(score)
            else:
                metrics_one.append(name)
                scores_one.append(score)

        metrics = metrics_zero + metrics_one
        scores = scores_zero + scores_one
        # #276: the 0 end of the replicate scale, kept in scope for `_from_replicate_column`.
        # Every accepted spelling maps to itself here, exactly as `metric_names` does.
        base_by_name = dict(zip(metric_names, b_vals))
        # Every scoreable metric was skipped for a degenerate baseline -> there is no score. The
        # `if scores else 0.0` fallback below would report avg_score = 0.0, which reads as "equals
        # the baseline" rather than "nothing was scored" -- the same hazard baseline.py raises on
        # one level up for an all-diagnostic profile.
        if skipped and not scores:
            raise ValueError(
                f"nothing left to score: every scoreable metric had a degenerate baseline "
                f"({', '.join(sorted(skipped))}). avg_score would be a fallback 0.0, which reads "
                "as 'equals the baseline'; fix the baseline rather than reporting that number."
            )
        # Rows the BASELINE pass declined to score, but a requested scale names anyway. Their
        # from_baseline is null; their scale cell is populated. Ordered lower-then-higher to match
        # the surrounding convention, and appended within each group so existing row order for
        # already-scored metrics is untouched.
        if scales:
            present = {_NAME_TO_CANONICAL.get(m, m) for m in metrics}
            add_zero, add_one = [], []
            for sc in scales:
                for canon in sc.entries:
                    if canon in present:
                        continue
                    present.add(canon)
                    target = add_zero if CATALOG[canon].scoring.direction == "lower" else add_one
                    target.append(canon)
            if add_zero or add_one:
                logger.info(
                    "restoring %d row(s) the baseline pass did not score, because a requested "
                    "scale names them: %s. Their from_baseline is null; their scale column is "
                    "populated, since a scale's reference points are constants the baseline "
                    "cannot affect.", len(add_zero) + len(add_one), sorted(add_zero + add_one),
                )
            metrics_zero, scores_zero = metrics_zero + add_zero, scores_zero + [None] * len(add_zero)
            metrics_one, scores_one = metrics_one + add_one, scores_one + [None] * len(add_one)
            metrics = metrics_zero + metrics_one
            scores = scores_zero + scores_one
        real = [s for s in scores if s is not None]
        avg = float(np.mean(real)) if real else 0.0
        out = pl.DataFrame({"metric": metrics + ["avg_score"],
                            "from_baseline": scores + [avg]},
                           schema={"metric": pl.String, "from_baseline": pl.Float64})
    else:
        # Scale-only: the user frame is the only input, so read and validate just it.
        if isinstance(results_user, (str, os.PathLike)):
            results_user = pl.read_csv(results_user)
        if "statistic" not in results_user.columns:
            raise ValueError("missing 'statistic' column in agg results")
        available = results_user["statistic"].to_list()
        if comparison_statistic not in available:
            raise ValueError(
                f"comparison_statistic {comparison_statistic!r} not found in agg results; "
                f"available: {available}"
            )
        if lfc_nmae_ref is not None and comparison_statistic != "mean":
            raise ValueError(
                f"lfc_nmae_ref requires comparison_statistic='mean' (spec 4.1 is a ratio "
                f"of means); got {comparison_statistic!r}. Drop the reference or use the mean."
            )
        u_row = results_user.filter(
            pl.col("statistic") == comparison_statistic).drop("statistic")
        metric_names = u_row.columns
        u_vals = u_row.row(0)
        # Row order follows the baseline path's convention -- lower-is-better first, then
        # higher-is-better, then avg_score -- so the two modes read the same. Within each
        # group it follows the scale's own DECLARATION order, which is what the frozen table
        # reads top to bottom; iterating a set here would make row order vary between runs.
        seen, lower, higher = set(), [], []
        for sc in scales:
            for m in sc.entries:                      # dict preserves insertion order
                if m in seen:
                    continue
                seen.add(m)
                (lower if CATALOG[m].scoring.direction == "lower" else higher).append(m)
        out = pl.DataFrame({"metric": lower + higher + ["avg_score"]})
    # ONLY when a reference is supplied. Unconditionally adding the column changes the
    # frame's shape for every existing caller: tests/test_cli_baseline.py asserts
    # `df.columns == ["metric", "from_baseline"]`, and this frame is pinned against the
    # frozen compat.score_agg_metrics.
    if lfc_nmae_ref is not None:
        out = out.with_columns(
            _from_reference_column(out["metric"].to_list(), u_vals, metric_names,
                                   lfc_nmae_ref)
        )
    # #276 part B. ONLY when a door was given, for the same frame-shape reason as above.
    if anchor is not None or anchor_cache is not None:
        if results_base is None:
            raise ValueError(
                "an anchor needs BOTH ends of the scale, and scale-only scoring has no "
                "baseline frame: from_replicate is (u - b) / (r - b), so without a baseline "
                "the 0 end does not exist and the column would be undefined rather than "
                "merely null. Pass results_base."
            )
        if anchor_expect is None:
            raise ValueError(
                "score_metrics(anchor=/anchor_cache=) requires anchor_expect: an anchor is a "
                "property of ONE dataset under ONE configuration, and without this run's own "
                "expectations the artifact would only be validated against itself -- which "
                "passes for any artifact whatsoever. Build it with "
                "`expect_from_run_meta(run_meta)`."
            )
        # `_cached_bundle` is evaluated ONLY when there is no supplied anchor. "Supplied
        # wins" has to mean the cache is never TOUCHED: an inaccessible cache root, or a
        # descriptor pointing at a moved directory, must not abort scoring that had a
        # perfectly good artifact in hand.
        frame, ameta, source = resolve_anchor(
            anchor_expect, supplied=anchor,
            cached=_cached_bundle(anchor_cache) if anchor is None else None)
        digest = anchor_digest(frame, ameta)
        entries = _replicate_entries(base_by_name, frame)
        # Rows the BASELINE pass declined to score, but the anchor names anyway -- exactly the
        # restoration a requested scale gets. Without it a metric present in the aggregate but
        # absent from the output (an `overrides={m: Scoring(scored=False)}`, or a non-decisive
        # metric dropped for a degenerate baseline) would make the anchor's average cover
        # fewer members than the anchor names.
        present = {_NAME_TO_CANONICAL.get(m, m) for m in out["metric"].to_list()}
        if restore := [c for c in entries if c not in present]:
            logger.info("restoring %d row(s) the baseline pass did not score, because the "
                        "anchor names them: %s. Their from_baseline is null.",
                        len(restore), sorted(restore))
        # CALLED UNCONDITIONALLY, even with nothing to add. A requested SCALE restores its own
        # dropped rows earlier, on the metric/score lists before this frame was built
        # (score.py:594-618), and it appends rather than sorting -- so with a scale and a
        # row-removing override both in play the anchor would find the row already present,
        # skip restoration, and leave the frame in an order the frozen `member_order_in_frame`
        # does not describe. `_insert_metric_rows` normalizes both direction groups, and is a
        # no-op on any frame that never needed restoring.
        out = _insert_metric_rows(out, restore)
        out = out.with_columns(
            _reference_column(out["metric"].to_list(), u_vals, metric_names, entries,
                              column="from_replicate", label="the replicate anchor"),
            anchor_source=pl.lit(source, dtype=pl.Utf8),
            anchor_digest=pl.lit(digest, dtype=pl.Utf8),
        )
        if rb_manifest is not None:
            out = out.with_columns(
                anchor_source=pl.lit("real_bundle", dtype=pl.Utf8),
                real_bundle_id=pl.lit(rb_manifest["real_bundle_id"], dtype=pl.Utf8),
                real_bundle_digest=pl.lit(manifest_digest(rb_manifest), dtype=pl.Utf8),
            )
            rule = rb_manifest.get("rule_digest")
            want = competition.competition_digest()
            if rule is not None and rule != want:
                raise ValueError(
                    f"real bundle {rb_manifest.get('real_bundle_id')!r} was built under "
                    f"competition rule {rule!r}, but this build's rule is {want!r}. Its "
                    "numbers are not comparable with anything scored under the current rule; "
                    "rebuild the bundle."
                )
            overrides_supplied = any(v is not None for v in (
                penalty_exponent, penalty_cap, clamp_low, clamp_high, penalty)) or bool(overrides)
            if overrides_supplied:
                logger.warning(
                    "score-time policy overrides were supplied; they apply to from_baseline "
                    "but NOT to from_replicate, which is policy-frozen exactly as a frozen "
                    "scale is. (Applying is not the same as moving: a STANDALONE penalty_cap "
                    "is inert under the shipped vcc2026 policies, so on that profile it moves "
                    "neither column -- it takes a per-metric overrides= policy that carries a "
                    "Box-Cox tail to make it live again.)"
                )
            if rb_waivers:
                # #291. The bundle is a competition bundle and the SUBMISSION is not, so the
                # downgrade has to happen here rather than being read off the artifact: a
                # reader of the bundle sees a real `rule_digest` either way. Deliberately the
                # SAME downgrade a diagnostic bundle gets -- from_replicate is reported,
                # from_baseline keeps its avg_score -- rather than a new signal, so there is
                # one meaning of "scored against a bundle but not enrolled" and not two.
                logger.warning(
                    "real bundle %r scored a DIAGNOSTIC submission: %s. from_replicate is "
                    "reported and from_baseline keeps its avg_score; this number is NOT a "
                    "competition score.",
                    rb_manifest.get("real_bundle_id"), "; ".join(rb_waivers))
            if rule is None:
                logger.warning(
                    "real bundle %r is a DIAGNOSTIC bundle (profile=%r): %s. from_replicate is "
                    "reported and from_baseline keeps its avg_score.",
                    rb_manifest.get("real_bundle_id"), rb_manifest.get("profile"),
                    "; ".join(rb_manifest.get("rule_mismatches") or ["no reason recorded"]))
            # ⚠️ INDEPENDENT conditions, and the enrolment predicate is spelled out rather
            # than left to an if/elif/else fallthrough. The two reasons not to enrol are
            # unrelated -- the BUNDLE can be diagnostic and the SUBMISSION can be -- and a
            # chain would report only the first while hiding the other's reason from a reader
            # who has to act on it.
            if rule is not None and not rb_waivers:
                # #276 part C (Alex, 2026-08-13): `from_replicate` REPLACES `from_baseline` in
                # avg_score. Nulling the old cell rather than leaving both is the whole point
                # -- every consumer today reads `from_baseline` @ avg_score, and after this
                # release that cell would be a well-formed, plausible, no-longer-official
                # number under the label they already read. A null is a visible break; a wrong
                # number is not.
                out = out.with_columns(
                    pl.when(pl.col("metric") == "avg_score")
                      .then(pl.lit(None, dtype=pl.Float64))
                      .otherwise(pl.col("from_baseline"))
                      .alias("from_baseline")
                )
    for sc in scales:
        out = out.with_columns(
            _scale_column(out["metric"].to_list(), u_vals, metric_names, sc)
        )
    if output is not None:
        out.write_csv(output)
    return out
