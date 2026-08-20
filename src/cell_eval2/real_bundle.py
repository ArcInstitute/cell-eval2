"""The real bundle: one dataset's two scale ends, plus the identity that binds them.

A competition triple was three producers writing three sidecars, with the pairing re-derived
at score time from two of the three -- and the baseline<->anchor leg never checked at all
(#276 part C, spec 1.1). A real bundle makes all three agreements true ONCE, at production,
and records the result.

The bundle is a MANIFEST LAID OVER TODAY'S ARTIFACTS, not a new container: every file but
`manifest.json` and `config.yaml` is byte-format identical to what `run`, `baseline` and
`run --anchor` already write, so `read_anchor` opens a bundle unchanged and a human can
inspect one with the tools they already have.

The manifest is PROVENANCE, not a checksum (Alex, 2026-08-13). Nothing re-verifies the
bundle's own files at score time; an edited cell moves every score with no trace. The threat
model is accidental mismatch, not tampering.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import tempfile
from dataclasses import dataclass, replace

import polars as pl

from . import competition
from .anchor import (ANCHOR_META, anchor_digest, anchor_store, build_meta, cached_anchor,
                     compute_replicate_anchor, write_anchor, _derive_seeds)
from .baseline import build_run_meta, write_json_meta, _degenerate_metrics
from .cache import config_hash
from .io import load_anndata
from .run import aggregate_metrics_wide, compute_metrics, metric_output_names

logger = logging.getLogger(__name__)

MANIFEST = "manifest.json"
MANIFEST_VERSION = 1
BASELINE_AGG = "baseline_agg.csv"
BASELINE_META = "baseline_meta.json"
CONFIG_YAML = "config.yaml"

#: Copied verbatim from the baseline leg's `run_meta` -- exactly `_check_baseline_config`'s
#: comparison set. Both PRED-side fields are deliberately absent: the prediction side is
#: EXPECTED to differ between the baseline arm and a submission, so comparing it would reject
#: precisely the pairings the check exists to permit. `de_pred_fingerprint` is nonetheless
#: CHECKED, one-sidedly and against a constant rather than against a peer -- see
#: `check_submission` (#291). `input_type_pred_effective` (#192) is recorded in the manifest
#: and not compared at all; see `MANIFEST_RECORDED_ONLY`.
SUBMISSION_PEERS = (
    "cell_eval2_version", "config_digest", "comparator", "source_fingerprint",
    "source_fingerprint_strict", "resolved_device", "resolved_de_backend",
    "input_type_real_effective", "de_real_fingerprint",
)

#: Copied into the manifest exactly like a peer, and never compared. #192 removed
#: `input_type_pred_effective` from the comparison, not from the record: it stays because
#: every bundle ever built carries it, `read_real_bundle` requires it, and dropping it would
#: make a new bundle's provenance strictly poorer than the frozen ones' for no gain.
#:
#: ⚠️ Under the frozen competition rule this field cannot differ in the first place. The
#: `vcc2026` preset sets `version="v2"` and `autodetect_input_type=False`, and
#: `norm.resolve_input_type` then returns the DECLARED type for both sides unconditionally --
#: so its value is a constant already covered by `config_digest`, and removing it from the
#: comparison is provably inert for the three official val bundles (measured). What it
#: unblocks is a DIAGNOSTIC bundle built with `autodetect_input_type=True` or `version="v1"`,
#: where the pred side is re-typed per matrix and a log-normalized submission was refused
#: outright -- with no `--allow-config-mismatch` waiver available on the bundle path at all.
MANIFEST_RECORDED_ONLY = ("input_type_pred_effective",)


@dataclass(frozen=True)
class RealBundle:
    root: str
    manifest: dict
    baseline_agg: str
    baseline_meta: dict


def manifest_digest(manifest: dict) -> str:
    """A stable id for one bundle's manifest, stamped into the scored frame.

    `created_utc` is excluded so two builds of the same inputs produce the same id. Nothing
    ever COMPARES this value -- it is derived at score time from the manifest that was read,
    so it cannot become a gate that silently fails (spec 5.4).
    """
    payload = {k: v for k, v in manifest.items() if k != "created_utc"}
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(blob.encode()).hexdigest()


def check_submission(manifest: dict, user_meta: dict, *,
                     diagnostic_supplied_de_pred: bool = False) -> list[str]:
    """Refuse a submission that was not produced under this bundle's config. Fatal by design.

    Passing `--real-bundle` is an affirmative claim -- "this is a competition submission" --
    so a disagreement is an error, not a downgrade. There is deliberately no escape hatch:
    `--allow-config-mismatch` is a USAGE ERROR alongside a bundle rather than a waiver, so a
    harness cannot land in diagnostic mode by habit.

    The fields are exactly `_check_baseline_config`'s, each already justified in place. The
    manifest supplies the peer values that used to come from `baseline_meta.json`.

    Plus ONE non-peer check (#291): the prediction side's DE table must have been COMPUTED
    from the submitted cells, never supplied.

    `diagnostic_supplied_de_pred` is the ONE opt-in, and it is not a waiver in the
    `--allow-config-mismatch` sense: it does not make the run pass as a submission, it makes
    the run DIAGNOSTIC. RETURNS the waivers taken, so the caller cannot honour the opt-in
    and forget the downgrade -- `score_metrics` refuses to enrol a non-empty return.
    Returning it beats leaving the caller to re-derive it from `user_meta`, which is the
    shape that lets the two drift.
    """
    _MISSING = object()
    problems, waived = [], []
    for field in SUBMISSION_PEERS:
        want = manifest.get(field, _MISSING)
        got = user_meta.get(field, _MISSING)
        if want is _MISSING or got is _MISSING:
            # FAIL-CLOSED: a missing key is a mismatch, not a match. Comparing with `.get()`
            # would let two empty objects pass as fully verified (cli.py:120).
            problems.append(f"{field}: missing from "
                            + ("the bundle manifest" if want is _MISSING else "run_meta.json"))
        elif want != got:
            problems.append(f"{field}: bundle={want!r} submission={got!r}")

    # #291. `de_pred_fingerprint` records THAT a pred-side DE table was supplied (`--de-pred`),
    # not what the prediction contains: `build_run_meta` leaves it null whenever the table was
    # computed, which is every ordinary submission. So the gap the issue reports is real -- a
    # supplied pred-side table reaches scoring unremarked, and it can omit `(target, feature)`
    # pairs that `_direction_frame`'s inner join would otherwise score as misses, which moves
    # `direction_reach_raw` discontinuously (0.0 -> 0.9625 in the issue's repro, measured at
    # the pre-`REACH_PURITY_FLOOR` purity floor 0.975; at `REACH_PURITY_FLOOR = 0.9` the same repro needs 9
    # head misses rather than 3 and moves 0.0 -> 0.8875 -- smaller, still discontinuous).
    #
    # But it is checked ONE-SIDEDLY, against the constant `None`, rather than added to
    # SUBMISSION_PEERS. Two reasons, and the second is decisive:
    #
    #   1. There is nothing to compare against. The bundle side is null BY CONSTRUCTION --
    #      `_baseline_leg` never passes `de_pred`, and `build_real_bundle` refuses a supplied
    #      `de_real` outright -- so a peer comparison would be a comparison with a constant
    #      wearing a peer's clothes.
    #   2. ⚠️ Adding it to SUBMISSION_PEERS is NOT backward compatible and is not equivalent.
    #      `read_real_bundle` requires every peer to be PRESENT in the manifest, and no bundle
    #      built before this check carries the key. MEASURED: all three official val bundles
    #      stop being readable at all -- "manifest.json is missing ['de_pred_fingerprint']" --
    #      so the gate would not tighten enrolment, it would end it.
    #
    # FAIL-CLOSED on absence, like the loop above: `build_run_meta` has stamped this field
    # unconditionally since the field existed, and `cell_eval2_version` is a peer, so any
    # submission that can legitimately reach this line carries it.
    #
    # ⚠️ The opt-in is scoped to a SUPPLIED table, never to a MISSING key. `--de-pred` is
    # also the isolator the metric campaigns are built on (#238/#259/#279 and #291 itself),
    # and those arms are scored against a bundle for its scale -- measured: 23 of the 30 runs
    # in the c243-flood sweep. Refusing them outright would move that rig off the bundle
    # path; letting them enrol would stamp `real_bundle_id` on a number the gate exists to
    # disqualify. The opt-in keeps the rig and takes the enrolment.
    fingerprint = user_meta.get("de_pred_fingerprint", _MISSING)
    refused_a_supplied_table = False
    if fingerprint is _MISSING:
        problems.append("de_pred_fingerprint: missing from run_meta.json")
    elif fingerprint is not None:
        supplied = (
            f"de_pred_fingerprint: {fingerprint!r} -- the prediction's DE table was SUPPLIED "
            "(--de-pred) rather than computed from the submitted cells, so the scored "
            "differential expression is not derived from the submission"
        )
        if diagnostic_supplied_de_pred:
            waived.append(supplied)
        else:
            problems.append(supplied)
            refused_a_supplied_table = True

    if problems:
        raise ValueError(
            f"the submission does not match real bundle {manifest.get('real_bundle_id')!r} -- "
            "the scores would be meaningless: " + "; ".join(problems)
            + ". Rebuild the submission with `--preset vcc2026` against the same real data, "
              "or score against --baseline-agg/--anchor for a diagnostic number."
            # ⚠️ BOTH remedies are conditional, and neither is in the sentence above. A
            # `--de-pred` table is the cause of exactly one of the problems this loop can
            # report, so telling every refusal to rebuild without it is advice that fixes
            # nothing for a `config_digest` or `source_fingerprint` mismatch -- and naming the
            # opt-in unconditionally would read as an escape hatch from the peer comparison,
            # which it is not (Copilot, PR #298).
            # Phrased as "for the supplied table", not "this one is", because `problems` may
            # hold several and the clause addresses exactly one of them (Copilot round 2,
            # suppressed). It has to read correctly whether that is the only problem or one of
            # four.
            + (" For the supplied DE table specifically: rebuild without --de-pred, or score "
               "against this bundle for a DIAGNOSTIC number with "
               "--diagnostic-supplied-de-pred, which does NOT enrol the result."
               if refused_a_supplied_table else "")
        )
    return waived


def _baseline_leg(real, baseline_pred, *, config, de_real):
    """Score the supplied baseline arm against the real data: the 0 end.

    `allow_fractional_counts=True` is set HERE and nowhere else. The tiled arm is a mean and
    therefore fractional in any counts space; the flip is pred-side only (`run.py:166`) and is
    in `DIGEST_EXEMPT_FIELDS` (`baseline.py:645`), so it cannot move `config_digest` and the
    submission side never carries it. `build_generic_baseline` makes the identical deviation
    (`baseline.py:930`).

    Three further overrides on the internal copy, none of which reach `config_digest` either
    (all three are in `DIGEST_EXEMPT_FIELDS`):

    * `outdir=None` -- the CLI maps `-o` onto `config.outdir`, so leaving it set makes
      `compute_metrics` write `run_params.yaml` into the bundle directory (`run.py:1163`),
      BEFORE the build gates have finished. That is an undeclared file inside an artifact whose
      whole contract is "these files and no others", written at a point where the build may
      still raise.
    * `cache_pred=None` -- the baseline arm is the pred side, and under a diagnostic
      (non-`cache_strict`) config two structurally identical arms share a metadata-only
      fingerprint, so a shared pred cache can serve one bundle's numbers to another's.
    * `cache_real` is LEFT ALONE: it is where the anchor bundle lives, and it is content-keyed.

    ⚠️ The run_meta is built from the CALLER's config, not the internal copy, because it is
    what a submission is compared against.
    """
    meta = build_run_meta(config, real, baseline_pred, de_real=de_real)
    scoring_cfg = replace(config, allow_fractional_counts=True, outdir=None, cache_pred=None)
    df = compute_metrics(baseline_pred, real, config=scoring_cfg, de_real=de_real)
    agg = aggregate_metrics_wide(df, metrics=metric_output_names(config))
    return agg, meta


def _anchor_leg(real_ad, *, config, base_seed, n_splits, names, metrics):
    """The replicate anchor: the 1 end. Through the cache when one is configured, so a re-run
    after a failure in the other leg costs nothing."""
    store = anchor_store(config)
    if store is not None:
        return cached_anchor(real_ad, config, store=store, base_seed=base_seed,
                             n_splits=n_splits)
    splits, frame = compute_replicate_anchor(real_ad, config=config, base_seed=base_seed,
                                             n_splits=n_splits)
    meta = build_meta(real_ad=real_ad, cfg=config, names=list(names), base_seed=base_seed,
                      n_splits=n_splits, seeds=_derive_seeds(base_seed, n_splits),
                      metrics=metrics)
    return frame, splits, meta


def build_real_bundle(real, baseline_pred, *, config, outdir, bundle_id,
                      base_seed=0, n_splits=5, de_real=None, force=False) -> dict:
    """Compute both scale ends from one config and write the bundle. Returns the manifest.

    Cheap checks first: the pre-flight `build_run_meta` reads backed and never materializes X
    (`baseline.py:747-757`), so a bad config, a missing `pert_col` or an unresolvable DE
    backend fails in seconds rather than after the anchor's five full metric runs.
    """
    from .catalog import resolve_metrics

    # A path that exists and is NOT a directory can never become one. Caught at the door
    # because the alternative is a bare `NotADirectoryError` from `os.rename` (measured) AFTER
    # both legs have run -- ten minutes of compute to report a mistake visible in zero
    # (Copilot, PR #290). `--force` does not open this: forcing means "replace that bundle",
    # not "delete that file".
    if os.path.exists(outdir) and not os.path.isdir(outdir):
        raise ValueError(
            f"{outdir!r} exists and is not a directory, so a bundle cannot be written there. "
            "A real bundle is a DIRECTORY of aggregates and sidecars; --force replaces an "
            "existing bundle, it does not delete a file. Choose another path."
        )
    if os.path.isdir(outdir) and os.listdir(outdir) and not force:
        raise ValueError(
            f"{outdir!r} is not empty. Two bundles' files interleaved in one directory are "
            "unreadable, and the manifest would describe only half of them. Pass force=True "
            "(`--force`) to overwrite, or choose a new directory."
        )
    # ⚠️ REFUSED, not threaded through (codex checkpoint-2 P0). `_baseline_leg` would score the
    # 0 end against the SUPPLIED table while `compute_replicate_anchor` recomputes its own
    # full-real DE for the `full_gate_raw` estimator, so `de_wilcoxon_lfc_nmae`'s two ends would
    # be gated and normalized by DIFFERENT tables -- the exact cohort mismatch that estimator
    # exists to prevent. No gate here can see it: the anchor's semantic identity does not cover
    # a supplied table, and the manifest's `de_real_fingerprint` records only the baseline leg's,
    # so every check would pass and the enrolled number would be quietly wrong. Threading it into
    # the anchor means threading it into the anchor's cache key and semantic identity too; until
    # that exists, refusing is the only honest answer.
    if de_real is not None:
        raise ValueError(
            "a real bundle cannot be built from a SUPPLIED real-side DE table. The baseline leg "
            "would use it while the replicate anchor recomputes its own for the full-real gate, "
            "so de_wilcoxon_lfc_nmae's 0 end and 1 end would come from different DE tables and "
            "nothing downstream could detect it. Rebuild without --de-real and let both legs "
            "compute the same table."
        )
    names, _ = resolve_metrics(config.metrics, version=config.version)
    metrics = metric_output_names(config)

    agg, baseline_meta = _baseline_leg(real, baseline_pred, config=config, de_real=de_real)
    # DECISIVE degeneracy stops the build. `score_metrics` would refuse the artifact anyway,
    # so finding out now costs seconds instead of a campaign.
    if bad := [d for d in _degenerate_metrics(agg) if d.get("decisive", True)]:
        raise ValueError(
            "the baseline leg is degenerate on metric(s) that decide a ranking: "
            + "; ".join(f"{d['metric']}={d['value']!r} ({d['reason']})" for d in bad)
            + ". A bundle built on it could not be scored."
        )

    real_ad = load_anndata(real, backed=False)
    frame, splits, anchor_meta = _anchor_leg(real_ad, config=config, base_seed=base_seed,
                                             n_splits=n_splits, names=names, metrics=metrics)

    # --- the three legs must agree -------------------------------------------------------
    if anchor_meta.get("control_source_effective") != competition.CONTROL_SOURCE_ANCHOR:
        raise ValueError(
            "the anchor reports control_source_effective="
            f"{anchor_meta.get('control_source_effective')!r}; a replicate anchor is only "
            "valid under per-half controls ('pred')."
        )
    if baseline_meta.get("source_fingerprint_strict"):
        # Only comparable when the baseline leg hashed CONTENT: `build_run_meta` computes
        # `source_fingerprint` at `strict=cfg.cache_strict`, while the anchor's
        # `real_fingerprint` is ALWAYS strict. Comparing a metadata hash against a content
        # hash would fail for a reason that is not a mismatch.
        if anchor_meta.get("real_fingerprint") != baseline_meta.get("source_fingerprint"):
            raise ValueError(
                "the two legs were built from different real data: anchor real_fingerprint="
                f"{anchor_meta.get('real_fingerprint')!r}, baseline source_fingerprint="
                f"{baseline_meta.get('source_fingerprint')!r}."
            )
    else:
        logger.warning(
            "cache_strict is off, so the baseline leg carries a METADATA-only fingerprint and "
            "the two legs cannot be cross-checked by content. This bundle is diagnostic."
        )
    if anchor_meta.get("semantic_identity") != baseline_meta.get("anchor_semantic_identity"):
        raise ValueError(
            "the two legs disagree on the anchor's semantic identity: anchor="
            f"{anchor_meta.get('semantic_identity')!r}, baseline="
            f"{baseline_meta.get('anchor_semantic_identity')!r}. They were built under "
            "different scoring conventions."
        )
    # The sidecar's own membership must match the frame it describes, or the manifest would
    # publish one and the scoring would use the other.
    #
    # SORTED LISTS, not sets -- `anchor.py:674` compares the same way and for the same reason:
    # a sidecar naming one metric twice has the same SET as a unique frame, so it would
    # publish here and then be refused later by `validate_anchor`, at score time, on a
    # competitor's submission.
    if sorted(anchor_meta.get("metric_names") or []) != sorted(frame["metric"].to_list()):
        raise ValueError(
            "the anchor's sidecar and its frame disagree on membership: sidecar "
            f"{sorted(anchor_meta.get('metric_names') or [])}, frame "
            f"{sorted(frame['metric'].to_list())}."
        )
    # The THIRD leg, and the one the three-sidecar design never had. `cache.config_hash` on
    # BOTH sides -- the anchor stamps exactly this function (anchor.py:500), RAW, which is why
    # `is_competition_rule` does not also compare it against the normalized frozen hash.
    #
    # ⚠️ NEVER `baseline_meta["config_digest"]`. That is `baseline.config_digest`, which
    # resolves `device="auto"` and the DE backend and folds in the comparator; it is a
    # different function over different inputs and is unequal to a `config_hash` for every
    # real artifact. Comparing the two would make this check fire on every build -- the
    # membership-ordering defect of #276 part C-1 in a new costume, pointing the other way.
    want_hash = config_hash(config.to_dict())
    if anchor_meta.get("config_hash") != want_hash:
        raise ValueError(
            "the anchor was built under a different config than this bundle's: anchor "
            f"config_hash={anchor_meta.get('config_hash')!r}, this run's {want_hash!r}. A "
            "cached anchor from an earlier config cannot define this bundle's top end."
        )
    # Both ends usable for every decisive member. `_replicate_entries` raises for a decisive
    # metric with no headroom, which is the anchor-side counterpart of the baseline check
    # above -- and it exercises the exact code `score` will run.
    from .score import _replicate_entries

    # ⚠️ The wide aggregate has ONE ROW PER STATISTIC (`aggregate_metrics_wide`), so the row
    # must be selected by name. `agg[c][0]` would silently take whichever statistic sorted
    # first and score every submission against it.
    row = agg.filter(pl.col("statistic") == competition.COMPARISON_STATISTIC)
    if row.height != 1:
        raise ValueError(
            f"the baseline aggregate has no {competition.COMPARISON_STATISTIC!r} row "
            f"(it carries {agg['statistic'].to_list()}); the anchor's replicate is a mean, so "
            "the 0 end must be read at the mean too."
        )
    base_by_name = {c: row[c].item() for c in agg.columns if c != "statistic"}
    entries = _replicate_entries(base_by_name, frame)

    # ⚠️ EVERYTHING below reads the artifact that came BACK, never the arguments. The anchor
    # leg may be satisfied from the content-addressed cache, so `base_seed`/`n_splits` above
    # are a request; only `anchor_meta` and the frame's `estimator` column are observations.
    # Deriving the rule state from the request would let a stale or hand-edited cache entry
    # receive the current competition digest.
    estimators = dict(zip(frame["metric"].to_list(), frame["estimator"].to_list()))
    reasons = competition.is_competition_rule(
        config, members=tuple(entries), anchor_meta=anchor_meta, estimators=estimators)

    manifest = {
        "manifest_version": MANIFEST_VERSION,
        "real_bundle_id": bundle_id,
        "created_utc": baseline_meta["created_utc"],
        **{f: baseline_meta.get(f) for f in SUBMISSION_PEERS + MANIFEST_RECORDED_ONLY},
        "real_fingerprint": anchor_meta.get("real_fingerprint"),
        "anchor_semantic_identity": anchor_meta.get("semantic_identity"),
        "anchor_config_hash": anchor_meta.get("config_hash"),
        # From the SIDECAR (`build_meta` stamps `metric_names`), not from `metrics` -- a
        # cached anchor's membership is an observation, the request is not.
        "anchor_metric_names": list(anchor_meta.get("metric_names") or []),
        "anchor_digest": anchor_digest(frame, anchor_meta),
        # From the SIDECAR, not from the arguments -- same reason as above.
        "base_seed": anchor_meta.get("base_seed"),
        "n_splits": anchor_meta.get("n_splits"),
        "seed_derivation": anchor_meta.get("seed_derivation"),
        "derived_seeds": [int(s) for s in anchor_meta.get("derived_seeds", [])],
        "bulk_target_sum": anchor_meta.get("bulk_target_sum"),
        "control_source_effective": anchor_meta.get("control_source_effective"),
        "estimators": dict(sorted(estimators.items())),
        "profile": config.metrics,
        "members": sorted(entries),
        # The whole enrolment decision, made ONCE and stamped. `null` is a diagnostic bundle;
        # a value that is neither null nor the repo's current digest is a bundle built under a
        # rule that has since moved, which `score` refuses (spec 4.4).
        "rule_digest": None if reasons else competition.competition_digest(),
        "rule_mismatches": reasons,
    }

    # --- publish: staged, rollback-safe replacement ---------------------------------------
    # Staged in a sibling directory and moved into place, because nothing re-verifies a bundle
    # at score time (spec 2.2): a `--force` rebuild that died after replacing
    # anchor_agg.parquet but before rewriting manifest.json would leave a well-formed manifest
    # describing the PREVIOUS anchor's numbers, and `score` would enrol it without complaint.
    parent = os.path.dirname(os.path.abspath(outdir)) or "."
    os.makedirs(parent, exist_ok=True)
    stage = tempfile.mkdtemp(prefix=".real-bundle-", dir=parent)   # same filesystem -> rename
    backup = None
    try:
        write_anchor(stage, splits, frame, meta=anchor_meta)
        agg.write_csv(os.path.join(stage, BASELINE_AGG))
        # SCRUBBED: `build_run_meta` stamps the absolute input path, the official bundles are
        # distributed, and `source` is not in `_check_baseline_config`'s comparison set.
        write_json_meta({**baseline_meta, "source": bundle_id},
                        os.path.join(stage, BASELINE_META))
        config.to_yaml(os.path.join(stage, CONFIG_YAML))
        with open(os.path.join(stage, MANIFEST), "w") as fh:
            json.dump(manifest, fh, indent=2, sort_keys=True, allow_nan=False)
        if os.path.isdir(outdir):
            # RE-CHECKED here, not only at entry: the preflight ran before hours of compute,
            # and a directory that appeared in the meantime must not be destroyed by a
            # non-force build. `listdir` and not mere existence -- an EMPTY directory was
            # allowed at preflight, so refusing it here would fail a build that was fine.
            if os.listdir(outdir) and not force:
                raise ValueError(
                    f"{outdir!r} became non-empty while this bundle was being built. Refusing "
                    "to overwrite it; pass force=True (`--force`) or choose another directory."
                )
            # MOVED ASIDE, not deleted. `rmtree` then `rename` loses the previous bundle
            # outright if the rename fails.
            backup = tempfile.mkdtemp(prefix=".real-bundle-old-", dir=parent)
            os.rmdir(backup)                      # need the NAME, not the directory
            os.rename(outdir, backup)
        try:
            os.rename(stage, outdir)
        except OSError:
            if backup is not None:                # put the old bundle back before re-raising
                try:
                    os.rename(backup, outdir)
                except OSError:
                    # ⚠️ RESTORATION FAILED. The previous bundle now exists ONLY at `backup`,
                    # so it must survive this function -- and the `finally` below deletes
                    # whatever `backup` still names. Clearing the local FIRST is what keeps it,
                    # and the message has to carry the path or the bundle is unrecoverable in
                    # practice even though it is intact on disk.
                    kept, backup = backup, None
                    raise ValueError(
                        f"publishing {outdir!r} failed AND the previous bundle could not be "
                        f"restored. It is intact at {kept!r}; move it back by hand. The new "
                        "bundle was discarded."
                    ) from None
                backup = None
            raise
    finally:
        shutil.rmtree(stage, ignore_errors=True)  # no-op once the rename succeeded
        if backup is not None:                    # None when restoration kept it deliberately
            shutil.rmtree(backup, ignore_errors=True)
    return manifest


def read_real_bundle(root) -> RealBundle:
    """Load a bundle's manifest and baseline leg. The anchor is read by `read_anchor(root)`.

    Shape checks only, exactly as `read_anchor` does: this verifies the manifest is an object
    carrying the fields a consumer needs. It does NOT re-verify the artifacts against it --
    the manifest is provenance (module docstring).
    """
    path = os.path.join(str(root), MANIFEST)
    if not os.path.exists(path):
        raise ValueError(
            f"no {MANIFEST} at {str(root)!r}; a real bundle is a directory written by "
            "`cell-eval2 prep-real-bundle`. A bare anchor directory has no baseline leg and "
            "no identity to check a submission against."
        )
    with open(path) as fh:
        manifest = json.load(fh)
    if not isinstance(manifest, dict):
        raise ValueError(f"{path} holds a {type(manifest).__name__}, not an object.")
    # ALL THREE groups, not just the ones the gate happens to read. This is SHAPE validation,
    # which the provenance-only ruling (module docstring) does not reach: it never compares a
    # recorded value against the files. The reason it has to be exhaustive is that every
    # consumer reads the manifest with `.get()`, and `.get()` returns None on BOTH sides of a
    # comparison when a field is absent -- which compares EQUAL. A manifest missing its whole
    # anchor-identity group would otherwise pass every check in the gate.
    required = (
        "manifest_version", "real_bundle_id", "created_utc",              # header
        *SUBMISSION_PEERS, *MANIFEST_RECORDED_ONLY,                        # submission peers
        "real_fingerprint", "anchor_semantic_identity", "anchor_config_hash",
        "anchor_metric_names", "anchor_digest", "base_seed", "n_splits", "seed_derivation",
        "derived_seeds", "bulk_target_sum", "control_source_effective", "estimators",
        "profile", "members", "rule_digest", "rule_mismatches",            # rule state
    )
    if missing := [f for f in required if f not in manifest]:
        raise ValueError(
            f"{path} is missing {missing}; it cannot certify a submission. Rebuild the bundle "
            "with `cell-eval2 prep-real-bundle`."
        )
    if manifest["manifest_version"] != MANIFEST_VERSION:
        raise ValueError(
            f"{path} declares manifest_version={manifest['manifest_version']!r}; this build "
            f"reads version {MANIFEST_VERSION}. A manifest from another version may name the "
            "same fields with different meanings, so it is refused rather than guessed at."
        )
    agg = os.path.join(str(root), BASELINE_AGG)
    meta_path = os.path.join(str(root), BASELINE_META)
    for p in (agg, meta_path, os.path.join(str(root), ANCHOR_META)):
        if not os.path.exists(p):
            raise ValueError(f"the real bundle at {str(root)!r} is missing {os.path.basename(p)}")
    with open(meta_path) as fh:
        baseline_meta = json.load(fh)
    return RealBundle(root=str(root), manifest=manifest, baseline_agg=agg,
                      baseline_meta=baseline_meta)
