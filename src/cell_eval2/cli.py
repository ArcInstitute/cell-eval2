from __future__ import annotations

import argparse
import json
import logging
import os
from dataclasses import replace

import polars as pl
import yaml

from .baseline import _degenerate_metrics, build_run_meta, write_json_meta
from .ceiling import compute_ceiling
from .config import EvalConfig
from .run import (aggregate_metrics_wide, compute_metrics, metric_agg, metric_cohorts,
                  metric_output_names, precompute_cache)
from .score import score_metrics

logger = logging.getLogger(__name__)

# argparse dest (hyphens -> underscores) -> EvalConfig field; --profile maps to `metrics`.
_FLAG_TO_FIELD = {
    "profile": "metrics", "pert_col": "pert_col", "control": "control",
    "input_type": "input_type", "version": "version", "outdir": "outdir",
    "cache_real": "cache_real", "cache_pred": "cache_pred", "cache_strict": "cache_strict",
}


def _add_config_flags(p: argparse.ArgumentParser) -> None:
    p.add_argument("--config", default=None, help="EvalConfig YAML (base; explicit flags override)")
    p.add_argument("--preset", default=None,
                   help="shipped preset as the config base: 'vcc2026' (the competition), "
                        "'v1', 'v2', 'cell-eval-0.7.6'. Mutually exclusive with --config "
                        "(both are the BASE); explicit flags and --set override it.")
    p.add_argument("--profile", default=None, help="metric profile (default: full)")
    p.add_argument("--pert-col", default=None, help="perturbation column (default: target)")
    p.add_argument("--control", default=None, help="control label (default: non-targeting)")
    p.add_argument("--input-type", default=None, choices=["counts", "lognorm"],
                   help="input matrix type (default: lognorm)")
    p.add_argument("--version", default=None, choices=["v1", "v2"],
                   help="metric-name version for output labels (default: v2)")
    p.add_argument("--cache-real", default=None, help="cache folder for real-side artifacts")
    p.add_argument("--cache-pred", default=None, help="cache folder for pred-side artifacts")
    # BooleanOptionalAction (--cache-strict / --no-cache-strict) with default=None so an
    # explicit flag can both enable AND disable the config's value; absence -> no override.
    p.add_argument("--cache-strict", action=argparse.BooleanOptionalAction, default=None,
                   help="content-hash fingerprints (stronger, but reads X)")
    # Generic escape hatch for ANY config field (nested de.*/discrimination.*/filter.* +
    # less-common top-level fields) that has no dedicated flag. Repeatable; VALUE parsed as
    # YAML; applied AFTER --config and the explicit flags (highest precedence).
    p.add_argument("--set", action="append", default=None, dest="set_overrides",
                   metavar="KEY.PATH=VALUE",
                   help="override any config field by dotted path, repeatable "
                        "(e.g. --set de.min_abs_log2fc=0.25). VALUE is parsed as YAML; "
                        "applied after --config and the explicit flags.")


def _apply_set_overrides(cfg: EvalConfig, overrides: list[str]) -> EvalConfig:
    """Apply repeatable ``--set KEY.PATH=VALUE`` dotted overrides onto ``cfg``. VALUE is
    parsed with YAML semantics (``0.25``->float, ``null``->None, ``true``->bool), spliced into
    the config dict by dotted path, and the config is rebuilt via ``EvalConfig.from_dict`` so
    all dataclass validation runs. User-facing errors raise ``SystemExit`` (clean message)."""
    d = cfg.to_dict()
    for item in overrides:
        path, sep, raw = item.partition("=")
        path = path.strip()
        if not sep or not path:
            raise SystemExit(f"--set expects KEY.PATH=VALUE, got {item!r}")
        keys = path.split(".")
        cur = d
        for k in keys[:-1]:
            if not isinstance(cur, dict) or k not in cur:
                raise SystemExit(f"--set: unknown config path {path!r}")
            cur = cur[k]
        if not isinstance(cur, dict) or keys[-1] not in cur:
            raise SystemExit(f"--set: unknown config path {path!r}")
        # Require a leaf: overwriting a whole section (de/discrimination/filter) either drops its
        # sibling fields silently (dict value) or bypasses dataclass validation -> late crash
        # (scalar value). Force `de.<field>=...` instead.
        if isinstance(cur[keys[-1]], dict):
            raise SystemExit(
                f"--set: {path!r} is a config section, not a field; "
                f"set a field within it (e.g. {path}.<field>=...)"
            )
        try:
            cur[keys[-1]] = yaml.safe_load(raw)
        except yaml.YAMLError as e:
            raise SystemExit(f"--set: malformed YAML value in {item!r}: {e}") from None
    try:
        return EvalConfig.from_dict(d)
    except (ValueError, TypeError) as e:
        raise SystemExit(f"--set: invalid config value: {e}") from None


def _build_cfg(args) -> EvalConfig:
    cli_overrides = {field: getattr(args, flag)
                     for flag, field in _FLAG_TO_FIELD.items()
                     if hasattr(args, flag) and getattr(args, flag) is not None}
    if getattr(args, "preset", None) and args.config:
        raise SystemExit("--preset and --config are both config BASES; pass one. To adjust a "
                         "preset, use explicit flags or --set, which override it.")
    if getattr(args, "preset", None):
        try:
            cfg = EvalConfig.from_preset(args.preset)
        except ValueError as e:
            raise SystemExit(f"--preset: {e}") from None
    elif args.config:
        cfg = EvalConfig.from_yaml(args.config)
    else:
        cfg = EvalConfig()
    if cli_overrides:
        cfg = replace(cfg, **cli_overrides)
    set_overrides = getattr(args, "set_overrides", None)
    if set_overrides:
        cfg = _apply_set_overrides(cfg, set_overrides)
    return cfg


def _check_baseline_config(args) -> None:
    """Fail loud when the baseline and the user run used different SCORING configs.

    ``score_metrics`` already rejects mismatched metric COLUMNS, which catches a v1-vs-v2
    mix because the versions emit different names. It cannot see a convention mismatch
    behind identical names -- a baseline built at ``de.p_adj_threshold=0.01`` scored
    against submissions at 0.05 yields identically-shaped frames and silently wrong
    margins. That is what the digest is for.

    It compares the whole execution identity, because the digest alone cannot see most of
    it: the code version, the scoring-config digest (over the REQUESTED config on both
    sides), the RESOLVED device and DE backend, the reference's strict fingerprint, the REAL
    side's effective input type, and the fingerprint of a SUPPLIED real-side DE table. A
    baseline built from a different reference, on a different engine, or against a different
    supplied DE table is otherwise invisible. The two PRED-side fields are recorded and not
    compared -- see the comment at the loop.

    FAIL-CLOSED. A missing key is a mismatch, not a match -- comparing with ``.get()`` would
    let two empty JSON objects pass as "fully verified", and would make an absent
    ``resolved_de_backend`` equal a legitimate ``null``.

    There is deliberately NO fallback to ``run_params.yaml``. That file is written AFTER
    ``compute_metrics`` resolves ``target_sum=None`` to the real control pool's median
    (``run._resolve_target_sum_from_control``, called from ``run._run_metrics``), so
    digesting it would compare a resolved numeric target against the baseline's requested
    ``None`` and mismatch two runs that asked for the same thing. Nothing predating this PR writes ``agg_results.csv`` either, so no
    legacy artifact needs serving. Without ``run_meta.json`` the pairing is reported as NOT
    VERIFIED rather than verified more weakly.
    """
    _MISSING = object()
    base = _read_meta(_find_meta(args.baseline_meta, args.baseline_agg,
                                 "baseline_meta.json"))
    user = _read_meta(_find_meta(args.user_meta, args.user_agg, "run_meta.json"))
    log = logging.getLogger(__name__)
    if base is None or user is None:
        log.warning(
            "score: pairing NOT verified -- need baseline_meta.json beside --baseline-agg "
            "(or --baseline-meta) and run_meta.json beside --user-agg (or --user-meta). A "
            "baseline built under a different scoring config, engine, reference or supplied "
            "DE table yields silently wrong margins."
        )
        return

    if not (base.get("source_fingerprint_strict") or user.get("source_fingerprint_strict")):
        log.warning(
            "score: reference identity compared at metadata level only (shape, dtype, var "
            "index, per-cell labels) -- fingerprint_adata does not read X unless "
            "cache_strict is set (cache.py:88-100). Two references differing only in their "
            "VALUES would not be distinguished. Re-run both sides with --cache-strict to "
            "compare content."
        )
    base_comparator = base.get("comparator", _MISSING)
    user_comparator = user.get("comparator", _MISSING)
    if base_comparator is _MISSING or user_comparator is _MISSING:
        missing_from = []
        if base_comparator is _MISSING:
            missing_from.append("the baseline stamp")
        if user_comparator is _MISSING:
            missing_from.append("run_meta.json")
        raise SystemExit(
            "baseline/user comparator mismatch -- the margins would be meaningless: "
            f"comparator missing from {' and '.join(missing_from)}. Rebuild both artifacts "
            "with comparator-aware cell-eval."
        )
    if base_comparator != user_comparator:
        raise SystemExit(
            "baseline/user comparator mismatch -- the margins would be meaningless: "
            f"comparator baseline={base_comparator!r} user={user_comparator!r}. "
            "Rebuild the baseline in the same effective expression space."
        )
    problems = []
    # TWO pred-side fields are recorded and deliberately NOT compared, for the same reason.
    #
    # de_pred_fingerprint: the baseline's prediction is synthetic and --de-pred is rejected
    # for it, so its value is always null while an ordinary `run --de-pred` has a hash. They
    # can never match, and comparing them would reject exactly the legitimate pairings this
    # check exists to permit -- the prediction side is EXPECTED to differ, just as adata_pred
    # does.
    #
    # input_type_pred_effective (#192): the same argument, and it was not applied. It is the
    # resolved matrix convention of the PREDICTION, and the two sides genuinely differ in the
    # common case -- the baseline's prediction is a fractional mean of counts that section
    # 3.0's matrix-space lock pulls back to `counts`, while a VCC submission is usually
    # log-normalized. MEASURED on a real submission: eight of the nine fields matched and the
    # run aborted on the ninth. Both values were correct, and both sides are converted into
    # the same metric space before any metric is computed, so the margins were comparable.
    #
    # ⚠️ Dropping it rather than demoting it to a warning, because the flag is BLANKET:
    # --allow-config-mismatch was the only way past this field and it also downgrades
    # config_digest, source_fingerprint, resolved_de_backend, resolved_device and
    # de_real_fingerprint to warnings. The common case was forcing users to disarm the five
    # checks this function exists for. input_type_REAL_effective stays compared -- the real
    # side is the one the two runs genuinely share.
    for field in ("cell_eval2_version", "config_digest", "source_fingerprint",
                  "source_fingerprint_strict", "resolved_device", "resolved_de_backend",
                  "input_type_real_effective", "de_real_fingerprint"):
        b, u = base.get(field, _MISSING), user.get(field, _MISSING)
        if b is _MISSING or u is _MISSING:
            problems.append(f"{field}: missing from "
                            + ("the baseline stamp" if b is _MISSING else "run_meta.json"))
        elif b != u:
            problems.append(f"{field}: baseline={b!r} user={u!r}")
    if not problems:
        return
    msg = ("baseline/user mismatch -- the margins would be meaningless: "
           + "; ".join(problems)
           + ". Rebuild the baseline against the same reference, config, host and DE "
             "inputs, or pass --allow-config-mismatch if this is deliberate.")
    if args.allow_config_mismatch:
        log.warning("score: %s", msg)
        return
    raise SystemExit(msg)


def _find_meta(explicit, sibling_of, name):
    if explicit is not None:
        return explicit
    guess = os.path.join(os.path.dirname(os.path.abspath(sibling_of)), name)
    return guess if os.path.exists(guess) else None


def _read_meta(path):
    if path is None:
        return None
    with open(path) as fh:
        return json.load(fh)


def _check_baseline_statistic(args) -> None:
    """Re-run the design-7.1 degeneracy gate on the statistic actually being consumed.

    The build-time gate validates ``mean``, but ``--comparison-statistic`` accepts any row,
    and a ``std`` row is NaN wherever the sample std is undefined -- which used to re-create
    the silent zeroing on a baseline whose ``mean`` row is perfectly healthy.

    Mirrors ``score_metrics``'s own split rather than pre-empting it: a DECISIVE metric
    (``catalog.is_decisive`` -- anything v1 can emit or that ``vcc`` or ``vcc2026`` scores)
    aborts here, and
    ``--allow-degenerate-baseline`` does not cover it, because the flag governs WRITING a
    diagnostic artifact, not scoring one. Any other scored metric is reported and the run
    continues, so the scorer can drop it and score the rest.

    This function used to abort on EVERY offender, which made the scorer's graceful branch
    unreachable through ``cell-eval2 score`` -- the predicate has to live in one place
    (``catalog.is_decisive``) or the two drift apart exactly like that.
    """
    import polars as pl

    bad = _degenerate_metrics(pl.read_csv(args.baseline_agg),
                              statistic=args.comparison_statistic)
    if not bad:
        return
    # Decisive offenders FIRST. Announcing "the rest is scored" and then exiting on a
    # decisive offender in the same call would be a promise the next line breaks.
    decisive = [d for d in bad if d.get("decisive", True)]
    if decisive:
        raise SystemExit(
            f"degenerate baseline on --comparison-statistic {args.comparison_statistic!r}: "
            + "; ".join(f"{d['metric']}={d['value']!r} ({d['reason']})" for d in decisive)
            + ". score_metrics refuses a baseline that cannot define a scale for a metric "
              "that decides a ranking; --allow-degenerate-baseline does not cover scoring."
        )
    # Skippable-only: say nothing here. `score_metrics` warns per metric as it drops them,
    # and it is the only one of the two that knows whether ANYTHING survived -- if nothing
    # does it raises rather than scoring. Warning from both places would duplicate the text
    # and let this copy claim "the rest is scored" in the case where there is no rest.


def _build_parser() -> argparse.ArgumentParser:
    """The whole CLI surface, as a parser. Extracted from `main` (#276 part C) so config
    resolution can be tested by parsing an argv rather than by running a subcommand."""
    parser = argparse.ArgumentParser(prog="cell-eval2")
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="run metrics on a (pred, real) pair, or real-data-only "
                                     "estimates (--ceiling / --lfc-nmae-ref, no --adata-pred)")
    # `main` no longer holds this subparser after parser extraction. Keep it on the parsed
    # namespace so `.error` still prints run's usage and exits 2, as
    # tests/test_cli.py::test_cli_requires_pred_unless_ceiling pins.
    run.set_defaults(_run_parser=run)
    run.add_argument("-ap", "--adata-pred", required=False,
                     help="predicted h5ad. Optional in real-data-only mode: omit it when "
                          "passing --ceiling and/or --lfc-nmae-ref to compute just those "
                          "real-data estimates. That mode writes only their outputs, no "
                          "results.csv")
    run.add_argument("-ar", "--adata-real", required=True)
    run.add_argument("--de-pred", default=None,
                     help="DE table (CSV/parquet) for the predicted side")
    run.add_argument("--de-real", default=None,
                     help="DE table (CSV/parquet) for the real side")
    run.add_argument("-o", "--outdir", default=None,
                     help="output dir (default: ./cell-eval2-outdir)")
    run.add_argument("--ceiling", action="store_true",
                     help="additionally compute a real-data-only data ceiling "
                          "(disjoint self-split + Spearman-Brown) -> "
                          "ceiling_results.csv + ceiling_agg.csv. Recomputes DE on "
                          "each half, so any --de-real is ignored for the ceiling.")
    run.add_argument("--ceiling-seed", type=int, default=0,
                     help="random seed for the ceiling disjoint split [default: %(default)s]")
    run.add_argument("--lfc-nmae-ref", action="store_true",
                     help="additionally compute the de_lfc_nmae split-half replicate "
                          "reference from the real data, writing lfc_nmae_ref.csv + "
                          "lfc_nmae_ref_agg.csv. Costs three extra DE passes (two if "
                          "--de-real is supplied). Requires every perturbation to have at "
                          "least 2 cells. Feed the "
                          "_agg file to `score --lfc-nmae-ref` for the scaled score.")
    run.add_argument("--lfc-nmae-ref-seed", type=int, default=0,
                     help="random seed for the reference's disjoint split "
                          "[default: %(default)s]")
    run.add_argument("--anchor", action="store_true",
                     help="additionally compute the replicate anchor from the real data "
                          "(#276), writing anchor_agg.parquet + anchor_splits.parquet + "
                          "anchor_meta.json. Costs one full metric run per split.")
    run.add_argument("--anchor-base-seed", type=int, default=0,
                     help="base seed for the anchor's splits [default: %(default)s]")
    run.add_argument("--anchor-splits", type=int, default=5,
                     help="number of disjoint splits to average [default: %(default)s]")
    run.add_argument("--write-degenes", action="store_true",
                     help="also write the computed DE tables to the outdir as "
                          "de_real.parquet and de_pred.parquet")
    _add_config_flags(run)

    pc = sub.add_parser("prep-cache", help="precompute one side's cache (loads only that side)")
    pc.add_argument("--side", required=True, choices=["real", "pred"])
    pc.add_argument("--adata", required=True, help="h5ad path for the chosen side")
    pc.add_argument("--de", default=None, help="DE table (CSV/parquet) for the chosen side")
    pc.add_argument("--comparator", choices=["bulk_lognorm", "lognorm"], default=None,
                    help="run-scoped expression comparator for this one-sided cache warm")
    _add_config_flags(pc)

    pb = sub.add_parser("prep-real-bundle",
                        help="build a real bundle: the baseline and replicate-anchor ends of "
                             "one dataset's competition scale, plus the manifest that binds "
                             "them. Holds aggregates and metadata only -- NO matrices.")
    pb.add_argument("--real", required=True, help="real .csad/.h5ad (the 1 end is split from it)")
    pb.add_argument("--baseline", required=True,
                    help="the pre-built baseline arm, scored as the PREDICTION (the 0 end)")
    pb.add_argument("-o", "--outdir", required=True, help="bundle directory to write")
    pb.add_argument("--id", default=None, dest="bundle_id",
                    help="bundle identifier, stamped into every scored frame and used in "
                         "place of the input path in the bundle's metadata "
                         "[default: the basename of -o]")
    pb.add_argument("--anchor-base-seed", type=int, default=0,
                    help="base seed for the anchor's splits [default: %(default)s]")
    pb.add_argument("--anchor-splits", type=int, default=5,
                    help="number of disjoint splits to average [default: %(default)s]")
    pb.add_argument("--force", action="store_true",
                    help="overwrite a non-empty output directory")
    _add_config_flags(pb)

    bl = sub.add_parser("baseline",
                        help="build + score the generic-response baseline for a reference")
    bl.add_argument("-ar", "--adata-real", required=True)
    bl.add_argument("--de-real", default=None,
                    help="DE table (CSV/parquet) for the real side")
    bl.add_argument("-o", "--outdir", default=None,
                    help="output dir (default: ./cell-eval2-baseline)")
    # BooleanOptionalAction gives --exclude-target-gene / --no-exclude-target-gene.
    # NOT a config field: it describes how the PREDICTION was built, changes no metric's
    # behaviour, and has nothing to match against -- putting it in EvalConfig would imply
    # otherwise and would perturb the config digest of every ordinary run.
    bl.add_argument("--exclude-target-gene", action=argparse.BooleanOptionalAction,
                    default=True,
                    help="omit each perturbation's own target gene from the average "
                         "(default: on)")
    bl.add_argument("--emit", choices=("dispersed", "tile"), default="dispersed",
                    help="cell emission: dispersed (default) resamples and scales controls; "
                         "tile is the legacy, known-biased arm kept only to reproduce "
                         "pre-fix numbers")
    bl.add_argument("--seed", type=int, default=0,
                    help="random seed for dispersed emission [default: %(default)s]")
    bl.add_argument("--save-pred", default=None,
                    help="also write the baseline prediction as h5ad (large)")
    bl.add_argument("--allow-degenerate-baseline", action="store_true",
                    help="Write the baseline artifact even when a scored metric's aggregate "
                         "cannot define a scale (denominator <= 0 or non-finite). Whether the "
                         "artifact can then be scored depends on WHICH metric: score_metrics "
                         "refuses one degenerate on a metric v1 emits or that vcc/vcc2026 "
                         "scores, and drops any other scored metric from avg_score, raising "
                         "if that leaves "
                         "nothing scoreable.")
    _add_config_flags(bl)

    sc = sub.add_parser("score", help="score a user aggregate against a baseline aggregate")
    sc.add_argument("--user-agg", required=True, help="wide agg CSV from `run`")
    sc.add_argument("--baseline-agg", default=None,
                    help="wide agg CSV from `baseline`. Optional when --scale is given: a "
                         "scale carries its own constant reference points, so scoring "
                         "against one needs no baseline artifact at all.")
    sc.add_argument("--real-bundle", default=None, metavar="ROOT",
                    help="a real bundle from `prep-real-bundle`: BOTH ends of the scale plus "
                         "the identity that binds them. Supplies the baseline and the anchor, "
                         "so it is mutually exclusive with --baseline-agg/--baseline-meta/"
                         "--anchor/--anchor-cache. A submission that does not match the "
                         "bundle is REFUSED, including one whose pred-side DE table was "
                         "supplied (see --diagnostic-supplied-de-pred). With a competition "
                         "bundle, from_replicate takes avg_score and from_baseline's goes "
                         "null.")
    # #291. The one opt-in on the bundle gate, and it buys the RUN, not the label: a
    # supplied pred-side DE table is the isolator the metric campaigns are built on, and
    # those arms want the bundle's scale. It is not a waiver of the peer comparison, which
    # has none -- see `real_bundle.check_submission`.
    sc.add_argument("--diagnostic-supplied-de-pred", action="store_true",
                    help="permit a submission whose pred-side DE table was SUPPLIED "
                         "(`run --de-pred`) to be scored against --real-bundle, and take its "
                         "ENROLMENT for it: from_replicate is reported, from_baseline keeps "
                         "its avg_score, and the result is NOT a competition score. Only "
                         "valid alongside --real-bundle.")
    sc.add_argument("--scale", action="append", default=None, metavar="NAME",
                    help="also score against a named scale, adding one column per scale. "
                         "Repeatable. `low-random_high-1_v10` puts 0 at the random minimum "
                         "and 1 at real input for the vcc2026 metrics. Does NOT change "
                         "from_baseline or its avg_score.")
    sc.add_argument("--lfc-nmae-ref", default=None,
                    help="lfc_nmae_ref_agg.csv from `run --lfc-nmae-ref`. Adds a "
                         "from_reference column with the replicate-scaled de_lfc_nmae "
                         "score. Does NOT change avg_score.")
    sc.add_argument("--anchor", default=None,
                    help="anchor directory written by `run --anchor`. Supplied wins over "
                         "the cached anchor; with neither, scoring raises. Adds the "
                         "from_replicate column as a DIAGNOSTIC -- it never takes avg_score. "
                         "The competition score is `--real-bundle`, which supplies both ends "
                         "and checks the submission against them.")
    sc.add_argument("--anchor-cache", default=None, metavar="ROOT",
                    help="cache root holding a replicate anchor for this dataset (the "
                         "--cache-real of the run that built it). Used only when --anchor "
                         "is not given, and only when run_meta.json carries the anchor's "
                         "cache descriptor.")
    sc.add_argument("-o", "--output", default=None, help="write the scored CSV here")
    sc.add_argument("--comparison-statistic", default="mean")
    sc.add_argument("--penalty-exponent", type=float, default=None)
    sc.add_argument("--penalty-cap", type=float, default=None)
    sc.add_argument("--baseline-meta", default=None,
                    help="baseline_meta.json (default: beside --baseline-agg)")
    sc.add_argument("--user-meta", default=None,
                    help="run_meta.json from the user run (default: beside --user-agg)")
    sc.add_argument("--allow-config-mismatch", action="store_true",
                    help="downgrade a baseline/user config, engine or reference mismatch "
                         "to a warning")
    sc.add_argument("--allow-degenerate-baseline", action="store_true",
                    help="accepted for symmetry with `baseline`, but it does NOT waive "
                         "anything here: the flag applies to WRITING an artifact. A baseline "
                         "degenerate on a metric v1 emits or that vcc/vcc2026 scores is "
                         "refused; any other degenerate scored metric is handled with "
                         "those metrics dropped from avg_score, provided at least one "
                         "scoreable metric remains")

    return parser


def main(argv: list[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)

    if args.command == "run":
        cfg = _build_cfg(args)
        if cfg.outdir is None:
            cfg = replace(cfg, outdir="./cell-eval2-outdir")  # CLI always needs a write target
        # Ceiling-only mode: the ceiling is estimated from the real data alone, so no
        # prediction is needed. Without --ceiling there would be nothing left to
        # compute, which is a usage error (args._run_parser.error -> subcommand usage + exit 2).
        ceiling_only = args.adata_pred is None
        if ceiling_only:
            if not (args.ceiling or args.lfc_nmae_ref or args.anchor):
                args._run_parser.error(
                    "-ap/--adata-pred is required unless --ceiling, --lfc-nmae-ref "
                    "or --anchor is passed (real-data-only mode)")
            # These only feed the main (pred, real) scoring, which is skipped here. Warn
            # rather than fail, so a stray flag does not break an otherwise valid run,
            # but the user is not left believing it took effect.
            ignored = [("--de-pred", args.de_pred)]
            if not args.lfc_nmae_ref:
                # --lfc-nmae-ref DOES consume --de-real (it supplies the full-real gate and
                # denominator), so warning that it is ignored would be wrong.
                ignored.append(("--de-real", args.de_real))
            elif args.ceiling and args.de_real is not None:
                # BOTH estimates requested: the reference consumes --de-real, the ceiling
                # still recomputes DE on its own halves. Saying nothing would leave a user
                # believing it speeds up both.
                logger.warning(
                    "--de-real is used by --lfc-nmae-ref (it supplies the full-real gate "
                    "and denominator) but NOT by --ceiling, which always recomputes DE on "
                    "its own disjoint halves. It saves one of the reference's three DE "
                    "passes and none of the ceiling's."
                )
            # Name only the flag(s) actually requested: `run --ceiling` alone should not be
            # told what --lfc-nmae-ref does, and vice versa.
            # --anchor belongs here or an anchor-only run reaches `_on[0]` on an EMPTY
            # list and raises IndexError before anything runs.
            _on = [f for f, on in (("--ceiling", args.ceiling),
                                    ("--lfc-nmae-ref", args.lfc_nmae_ref),
                                    ("--anchor", args.anchor)) if on]
            active = (f"{' and '.join(_on)} compute their own DE on disjoint halves"
                      if len(_on) > 1 else
                      f"{_on[0]} computes its own DE on disjoint halves")
            for flag, value in ignored:
                if value is not None:
                    logger.warning("%s is ignored in real-data-only mode: it feeds only the "
                                   "skipped (pred, real) scoring. %s of the real data.",
                                   flag, active)
            # --cache-real is NO LONGER ignored under --anchor: the anchor BUNDLE is
            # cached there (the inner splits still are not -- `_score_one_split` clears
            # both cache fields). Leaving the blanket warning would have the CLI assert
            # something the --anchor path contradicts.
            for flag, value in (("--cache-real", cfg.cache_real),
                                ("--cache-pred", cfg.cache_pred)):
                if value is None:
                    continue
                if flag == "--cache-real" and args.anchor:
                    logger.info("--cache-real holds the replicate anchor bundle for this "
                                "dataset; the inner half-splits still run uncached.")
                    continue
                logger.warning("%s is ignored in real-data-only mode: the half-splits "
                               "scored here can never hit the cache, so it runs with "
                               "caching disabled.", flag)
            if args.write_degenes:
                logger.warning("--write-degenes is ignored in real-data-only mode: the DE "
                               "tables it writes are the main (pred, real) scoring's, which "
                               "is skipped here.")
        if not ceiling_only:
            # BUILT before the metrics (backed reads only -- see build_run_meta), so a bad
            # config fails before the expensive work. run_meta.json mirrors the baseline
            # stamp's RESOLVED identity -- without it `score` has nothing on the user side to
            # compare against, and the stamped backend/device/reference are decorative.
            run_meta = build_run_meta(cfg, args.adata_real, args.adata_pred,
                                      de_real=args.de_real, de_pred=args.de_pred)
            df = compute_metrics(args.adata_pred, args.adata_real, config=cfg,
                                 de_pred=args.de_pred, de_real=args.de_real,
                                 write_de=args.write_degenes)
            os.makedirs(cfg.outdir, exist_ok=True)
            out_path = os.path.join(cfg.outdir, "results.csv")
            df.write_csv(out_path)
            # Same EXPECTED name list the baseline uses, so score_metrics' ordered column
            # comparison matches by construction rather than by luck.
            wide = aggregate_metrics_wide(df, metrics=metric_output_names(cfg))
            wide.write_csv(os.path.join(cfg.outdir, "agg_results.csv"))
            # Sidecar so a consumer of agg_results.csv alone can tell which statistic each
            # column holds: the wide frame's row is named "mean" for every metric, but it
            # carries whatever that metric's agg declares -- a metric with agg="median"
            # would put its MEDIAN there. Every shipped entry says "mean" since #231, so the
            # sidecar is currently constant; it is still written, because a consumer must be
            # able to READ the statistic rather than assume the invariant held for the
            # release that produced the file. It cannot be a row in the wide frame -- a
            # string row would coerce every metric column to text and score.py:83 would
            # silently stop receiving numbers after one CSV round-trip.
            # #239 adds the COHORT to the same sidecar, which is where the issue asks for it: a
            # metric's gate can drop perturbations as a function of the config
            # (filter_gene_min_cpm_cell moved de_lfc_nmae's cohort 557 -> 254 on the measured
            # sweep), so a cross-config comparison is partly a cohort comparison and the
            # aggregate alone cannot say so. `n_used` is the number of values the reported
            # statistic was taken over -- NOT the wide frame's `count` row, which describes the
            # RAW series and so over-reports for any metric that emits NaN rather than omitting
            # the row. See run.metric_cohorts for the measurement that separates the two.
            # APPENDED, so `metric` and `agg` keep their positions and meanings -- but this file
            # goes from 2 columns to 7, so a reader that destructures a fixed-width row or asserts
            # an exact schema must adapt (codex-review). The two internal readers do neither:
            # internal:tools/metricval/strata.py checks `{"metric","agg"} <= columns` and
            # internal:tools/metricval/report.py zips the two by name.
            pl.DataFrame(
                {"metric": list(wide.columns[1:]),
                 "agg": [metric_agg(m) for m in wide.columns[1:]]},
                schema={"metric": pl.String, "agg": pl.String},
            ).join(
                metric_cohorts(df, metrics=metric_output_names(cfg)), on="metric", how="left",
            ).write_csv(os.path.join(cfg.outdir, "metric_aggregation.csv"))
            # ...and WRITTEN, descriptor-free, as soon as the aggregate it describes exists.
            # Two hazards, and the meta has to be written TWICE to close both. Publishing it
            # BEFORE the aggregate would leave new metadata beside a STALE agg_results.csv in
            # a reused outdir if the run then failed. But deferring it past the anchor block
            # (which must run first, because it mutates `run_meta` with the cache descriptor)
            # opens the mirror-image window: the new aggregate lands, the anchor raises, and
            # the PREVIOUS run's run_meta.json is still sitting beside it. Either way `score`
            # would certify an aggregate the metadata does not describe. So: write it here,
            # matching what has been produced so far, and rewrite it after the anchor.
            write_json_meta(run_meta, os.path.join(cfg.outdir, "run_meta.json"))
            print(out_path)
        # BEFORE the rewrite below: the anchor block mutates `run_meta` with the cache
        # descriptor, and `run_meta.json` is written at the end of the `not ceiling_only`
        # branch. Appending this after that write would leave the descriptor on the floor
        # and `score --anchor-cache` permanently unopenable.
        if args.anchor:
            from .anchor import (ANCHOR_CACHE_KEY, _derive_seeds, anchor_cache_params,
                                 anchor_store, build_meta, cached_anchor,
                                 compute_replicate_anchor, write_anchor)
            from .cache import fingerprint_adata
            from .catalog import resolve_metrics
            from .io import load_anndata

            real_ad = load_anndata(args.adata_real, backed=False)
            # Resolved ONCE, before the branch: both arms need `names`, and the descriptor
            # below needs it too.
            names, _ = resolve_metrics(cfg.metrics, version=cfg.version)
            metrics = metric_output_names(cfg)
            # THROUGH the cache when one is configured. Calling compute_replicate_anchor
            # directly here is what would leave the cache written by nothing and
            # `resolve_anchor`'s cached door unreachable.
            store = anchor_store(cfg)
            if store is not None:
                anchor_frame, splits, meta = cached_anchor(
                    real_ad, cfg, store=store, base_seed=args.anchor_base_seed,
                    n_splits=args.anchor_splits)
            else:
                splits, anchor_frame = compute_replicate_anchor(
                    real_ad, config=cfg, base_seed=args.anchor_base_seed,
                    n_splits=args.anchor_splits)
                meta = build_meta(real_ad=real_ad, cfg=cfg, names=list(names),
                                  base_seed=args.anchor_base_seed,
                                  n_splits=args.anchor_splits,
                                  seeds=_derive_seeds(args.anchor_base_seed,
                                                      args.anchor_splits),
                                  metrics=metrics)
            os.makedirs(cfg.outdir, exist_ok=True)
            # The directory is written EITHER way: it is the supplied door's input, the
            # human-readable artifact, and what a competition submission ships.
            print(write_anchor(cfg.outdir, splits, anchor_frame, meta=meta))
            # The cache descriptor, for `score`'s cached door. Only when a store exists AND
            # this run produced a run_meta (anchor-only mode writes none).
            if store is not None and not ceiling_only:
                run_meta["anchor_cache"] = {
                    "root": cfg.cache_real, "key": ANCHOR_CACHE_KEY, "kind": "json",
                    "fingerprint": fingerprint_adata(real_ad, pert_col=cfg.pert_col,
                                                     strict=True),
                    "params": anchor_cache_params(cfg, real_ad, list(names),
                                                  base_seed=args.anchor_base_seed,
                                                  n_splits=args.anchor_splits,
                                                  metrics=metrics),
                }
        if not ceiling_only and args.anchor:
            # REWRITE, now carrying the anchor cache descriptor the block above added. Only
            # when an anchor was actually requested -- otherwise the first write is already
            # complete and rewriting it would be a no-op that only widens the window in
            # which the file is absent.
            write_json_meta(run_meta, os.path.join(cfg.outdir, "run_meta.json"))
        if args.ceiling:
            cres, cagg = compute_ceiling(args.adata_real, config=cfg,
                                         seed=args.ceiling_seed)
            os.makedirs(cfg.outdir, exist_ok=True)
            cres.write_csv(os.path.join(cfg.outdir, "ceiling_results.csv"))
            cagg.write_csv(os.path.join(cfg.outdir, "ceiling_agg.csv"))
            print(os.path.join(cfg.outdir, "ceiling_agg.csv"))
        if args.lfc_nmae_ref:
            from .lfc_nmae_ref import compute_lfc_nmae_reference
            # THREE extra DE passes, not two, unless --de-real is supplied: compute_metrics
            # owns the main run's real DE table internally and does not return it, so there
            # is nothing for `run` to hand over. When the user DID supply --de-real, pass it
            # through -- that is the one case where the cost really is two half-data passes.
            # Threading compute_metrics' internal table out is a later optimization.
            rres, ragg = compute_lfc_nmae_reference(args.adata_real, config=cfg,
                                                    seed=args.lfc_nmae_ref_seed,
                                                    de_real=args.de_real)
            os.makedirs(cfg.outdir, exist_ok=True)
            rres.write_csv(os.path.join(cfg.outdir, "lfc_nmae_ref.csv"))
            ragg.write_csv(os.path.join(cfg.outdir, "lfc_nmae_ref_agg.csv"))
            print(os.path.join(cfg.outdir, "lfc_nmae_ref_agg.csv"))
    elif args.command == "prep-cache":
        cfg = _build_cfg(args)
        precompute_cache(args.adata, side=args.side, config=cfg, de=args.de,
                         comparator=args.comparator)
        root = cfg.cache_real if args.side == "real" else cfg.cache_pred
        print(root)
    elif args.command == "prep-real-bundle":
        from .real_bundle import build_real_bundle

        cfg = _build_cfg(args)
        bundle_id = args.bundle_id or os.path.basename(os.path.normpath(args.outdir))
        try:
            man = build_real_bundle(
                args.real, args.baseline, config=cfg, outdir=args.outdir,
                bundle_id=bundle_id, base_seed=args.anchor_base_seed,
                n_splits=args.anchor_splits, force=args.force)
        except ValueError as e:
            raise SystemExit(f"prep-real-bundle: {e}") from None
        # PRINTED, and this line is load-bearing. A miscomputed rule check would otherwise
        # un-enrol every submission scored against this bundle, silently and for weeks; here
        # it is visible on the first build.
        if man["rule_digest"] is None:
            print(f"wrote DIAGNOSTIC bundle {bundle_id!r} to {args.outdir} -- it will NOT "
                  f"take avg_score: {'; '.join(man['rule_mismatches'])}")
        else:
            print(f"wrote competition bundle {bundle_id!r} to {args.outdir}")
    elif args.command == "baseline":
        from .baseline import build_generic_baseline
        cfg = _build_cfg(args)
        if cfg.outdir is None:
            cfg = replace(cfg, outdir="./cell-eval2-baseline")
        os.makedirs(cfg.outdir, exist_ok=True)
        res = build_generic_baseline(
            args.adata_real, config=cfg,
            exclude_target_gene=args.exclude_target_gene,
            emit=args.emit,
            seed=args.seed,
            de_real=args.de_real,
            save_pred=args.save_pred,
            allow_degenerate=args.allow_degenerate_baseline,
        )
        agg_path = os.path.join(cfg.outdir, "baseline_agg.csv")
        res.agg.write_csv(agg_path)
        res.results.write_csv(os.path.join(cfg.outdir, "baseline_results.csv"))
        # allow_nan=False: json.dump emits a bare `NaN` token by default, which is not
        # valid JSON. _degenerate_metrics already records None for non-finite values, so
        # this only fires if some other non-finite reached the stamp -- and then it should.
        with open(os.path.join(cfg.outdir, "baseline_meta.json"), "w") as fh:
            json.dump(res.meta, fh, indent=2, sort_keys=True, allow_nan=False)
        # NOTE: run_params.yaml is NOT written here. compute_metrics already wrote it
        # from the config it actually ran, which is the only place target_sum=None has
        # been resolved to the real control pool's median
        # (run._resolve_target_sum_from_control). Rewriting it from res.meta["config"]
        # would silently replace the resolved value with `null`.
        print(agg_path)
    elif args.command == "score":
        kw = {}                                    # FIRST: the bundle branch writes into it
        if args.real_bundle is None and args.baseline_agg is None and not args.scale:
            raise SystemExit(
                "score needs --baseline-agg, --scale, --real-bundle, or a combination: with "
                "none of them there is nothing to score against."
            )
        if args.real_bundle is not None:
            for flag in ("baseline_agg", "baseline_meta", "anchor", "anchor_cache"):
                if getattr(args, flag) is not None:
                    raise SystemExit(
                        f"--real-bundle and --{flag.replace('_', '-')} are mutually "
                        "exclusive: a real bundle already supplies both ends of the scale.")
            if args.allow_config_mismatch:
                # REFUSED, not ignored. The hatch stays available for --baseline-agg; it does
                # not open on the competition path, and silently accepting a flag that does
                # nothing is the failure cli.py:643 already argues against.
                raise SystemExit(
                    "--allow-config-mismatch does not apply to --real-bundle: a submission "
                    "that does not match its bundle is refused, with no waiver. Score against "
                    "--baseline-agg/--anchor for a diagnostic number instead.")
            user_meta_path = _find_meta(args.user_meta, args.user_agg, "run_meta.json")
            if user_meta_path is None:
                raise SystemExit(
                    "--real-bundle needs run_meta.json beside --user-agg (or --user-meta): "
                    "without this run's own identity the bundle cannot certify it.")
            kw["real_bundle"] = args.real_bundle
            kw["user_meta"] = _read_meta(user_meta_path)
            kw["diagnostic_supplied_de_pred"] = args.diagnostic_supplied_de_pred
        elif args.diagnostic_supplied_de_pred:
            # REFUSED, not ignored -- the same argument as --allow-config-mismatch above,
            # pointing the other way. The flag downgrades a bundle ENROLMENT, and the loose
            # path has no enrolment to downgrade, so accepting it would let a harness believe
            # it had marked a run diagnostic when nothing did. `score_metrics` refuses it too;
            # this is here so the message names the flag rather than the parameter.
            raise SystemExit(
                "--diagnostic-supplied-de-pred applies only to --real-bundle: it downgrades a "
                "bundle enrolment, and --baseline-agg/--anchor has no enrolment to downgrade. "
                "Drop it; a supplied --de-pred table needs no permission on that path.")
        if args.baseline_agg is not None and args.real_bundle is None:
            # UNCHANGED for the loose path. The bundle path runs `check_submission` instead,
            # and the baseline degeneracy gate is already covered at bundle-build time.
            _check_baseline_config(args)
            _check_baseline_statistic(args)
        if args.penalty_exponent is not None:
            kw["penalty_exponent"] = args.penalty_exponent
        if args.penalty_cap is not None:
            kw["penalty_cap"] = args.penalty_cap
        # #276 part B. The expectations come from the USER RUN's run_meta.json -- the same
        # record `_check_baseline_config` reads -- never from the anchor's own sidecar,
        # which would validate an artifact against itself and pass for any artifact at all.
        if args.anchor is not None or args.anchor_cache is not None:
            from .score import expect_from_run_meta
            user_meta_path = _find_meta(args.user_meta, args.user_agg, "run_meta.json")
            if user_meta_path is None:
                raise SystemExit(
                    "--anchor/--anchor-cache needs run_meta.json beside --user-agg (or "
                    "--user-meta): an anchor is a property of ONE dataset under ONE "
                    "configuration, and without this run's own identity the artifact "
                    "cannot be verified. Scoring against an unverified top end is exactly "
                    "what this gate exists to prevent."
                )
            user_meta = _read_meta(user_meta_path)
            try:
                kw["anchor_expect"] = expect_from_run_meta(user_meta)
            except ValueError as exc:
                raise SystemExit(str(exc)) from exc
            kw["anchor"] = args.anchor
            # The DESCRIPTOR, not a bare root: `CacheStore.get` needs the exact
            # (key, fingerprint, params, kind) quadruple, none of which `score` can derive.
            # The flag supplies only the ROOT, so a moved cache directory still works while
            # the key material still comes from the run that built it.
            # SUPPLIED WINS: only demand a cache descriptor when the cached door is the
            # only one. With --anchor also given, a run_meta without the block is not an
            # error -- the supplied artifact is what will be used.
            if args.anchor_cache is not None and args.anchor is None:
                if not user_meta.get("anchor_cache"):
                    # REFUSE, do not fall through. Without this the cached door is simply
                    # not passed, `score_metrics`' `anchor is not None or anchor_cache is
                    # not None` guard is False, and the user who asked for --anchor-cache
                    # gets NO from_replicate column and NO error -- a silent no-op on the
                    # flag they passed. Only a `run --anchor --cache-real` writes the
                    # descriptor, because `params` needs base_seed/n_splits.
                    raise SystemExit(
                        f"--anchor-cache {args.anchor_cache!r} was given, but this run's "
                        f"run_meta.json ({user_meta_path}) carries no 'anchor_cache' "
                        "descriptor, so the cached door cannot be opened: `CacheStore.get` "
                        "needs the exact (key, fingerprint, params) triple, and only the "
                        "run that BUILT the anchor can record it. Re-run with "
                        "`run --anchor --cache-real <root>`, or pass the anchor directory "
                        "with --anchor instead."
                    )
                kw["anchor_cache"] = {**user_meta["anchor_cache"],
                                      "root": args.anchor_cache}
            elif args.anchor_cache is not None and user_meta.get("anchor_cache"):
                # Both flags given and the descriptor exists: pass it, but `score_metrics`
                # only reads it if the supplied door yields nothing.
                kw["anchor_cache"] = {**user_meta["anchor_cache"],
                                      "root": args.anchor_cache}
        out = score_metrics(args.user_agg, args.baseline_agg, args.output,
                            args.comparison_statistic, lfc_nmae_ref=args.lfc_nmae_ref,
                            scale=args.scale, **kw)
        if args.output:
            print(args.output)
        else:
            print(out)
