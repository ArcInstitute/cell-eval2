from __future__ import annotations

import inspect
import logging
import math
import os
import uuid
from collections.abc import Sequence
from dataclasses import fields, replace

import anndata as ad
import numpy as np
import polars as pl

from . import norm as _norm
from .cache import (MISS, CacheStore, config_hash, fingerprint_adata,
                    fingerprint_de_table, result_fingerprint)
from .catalog import CATALOG, _NAME_TO_CANONICAL, deseq2_metric_name, resolve_metrics
from .config import DEParams, DiscriminationParams, EvalConfig, FilterParams
from .de import assemble_prepared_de, prep_de_side, rank_de_side, resolve_target_genes
from .gpu import resolve_device
from .io import load_anndata, validate_pair
from .metrics.de import de_lfc_nmae, de_sig_jaccard
from .metrics.direction import de_direction_reach
from .metrics.delta import distance_unbiased, mse_unbiased, mse_unbiased_capped
from .metrics.discrimination import discrimination_score
from .prep import (pseudobulk, pseudobulk_bulk_lognorm,
                   pseudobulk_bulk_lognorm_with_moments, pseudobulk_with_moments)
from .streaming_bulk import inmem_pseudobulk

logger = logging.getLogger(__name__)

_TIDY_SCHEMA = {"perturbation": pl.Utf8, "metric": pl.Utf8, "value": pl.Float64}


def _resolve_config(config: EvalConfig | None, overrides: dict) -> EvalConfig:
    cfg = config if config is not None else EvalConfig()
    # Coerce nested dataclass fields that may arrive as raw dicts when a caller builds
    # EvalConfig(filter={...}) / (discrimination={...}) / (de={...}) directly rather than
    # via from_dict; otherwise later attribute access
    # (cfg.filter.filter_gene_min_cpm_cell, …) would fail.
    if isinstance(cfg.filter, dict):
        cfg = replace(cfg, filter=FilterParams(**cfg.filter))
    if isinstance(cfg.discrimination, dict):
        cfg = replace(cfg, discrimination=DiscriminationParams(**cfg.discrimination))
    if isinstance(cfg.de, dict):
        cfg = replace(cfg, de=DEParams(**cfg.de))
    overrides = dict(overrides)
    fmin = overrides.pop("filter_gene_min_cpm_cell", None)
    valid = {f.name for f in fields(EvalConfig)}
    unknown = set(overrides) - valid
    if unknown:
        raise ValueError(
            f"unknown compute_metrics override(s): {sorted(unknown)}; "
            f"valid keys: {sorted(valid)} or 'filter_gene_min_cpm_cell'"
        )
    # Coerce nested-dataclass overrides passed as plain dicts (mirrors
    # EvalConfig.from_dict) so e.g. compute_metrics(..., discrimination={"distance": "l2"})
    # works instead of leaving a dict that later attribute access would choke on.
    if isinstance(overrides.get("filter"), dict):
        overrides["filter"] = FilterParams(**overrides["filter"])
    if isinstance(overrides.get("discrimination"), dict):
        overrides["discrimination"] = DiscriminationParams(**overrides["discrimination"])
    if isinstance(overrides.get("de"), dict):
        overrides["de"] = DEParams(**overrides["de"])
    # UNCONDITIONAL `replace`, even with no overrides: it re-runs EvalConfig.__post_init__,
    # which is where the cross-field rules live (notably version='v1' + de.backend='deseq2').
    # EvalConfig is mutable and callers do assign fields, so
    # `cfg = EvalConfig.v1(); cfg.de = DEParams(backend='deseq2')` builds an invalid pair that
    # construction-time validation never sees. Revalidating at the one boundary every driver
    # passes through closes that without freezing the dataclass.
    cfg = replace(cfg, **overrides)
    if fmin is not None:
        cfg = replace(cfg, filter=replace(cfg.filter, filter_gene_min_cpm_cell=fmin))
    return cfg


def _resolve_target_sum_from_control(cfg, real_ad):
    """Resolve `target_sum=None` ONCE, against the real control pool, returning the
    (possibly updated) cfg. #155: `target_sum=None` means "the median of whichever matrix
    normalize_total was handed", so every matrix that reaches DE separately would pick its
    own target and every log2FC would shift by log2(T_x/T_ref). Callers that compute DE on
    more than one matrix (compute_metrics' two sides; lfc_nmae_ref's three tables) MUST
    resolve before the first DE call, not per call.

    No-op with ZERO extra I/O when `target_sum` is already numeric.
    """
    # EFFECTIVE input type, not cfg.input_type: v1 allows a declared lognorm over real
    # counts. Guarded on `is None`, so this is a no-op with ZERO extra I/O for every
    # numeric-target config.
    if cfg.target_sum is None:
        ctrl_mat = _materialize(real_ad[real_ad.obs[cfg.pert_col].astype(str) == cfg.control])
        resolved = _norm.resolve_target_sum(
            ctrl_mat, input_type=_effective_input_type(ctrl_mat, cfg, side="real"),
            target_sum=None)
        if resolved is not None:      # None only for lognorm input, where target_sum is inert
            logger.info("resolved target_sum=None to the real control pool's median library "
                        "size: %s", resolved)
            cfg = replace(cfg, target_sum=resolved)
    return cfg


def _validate_input_once(adata, input_type, *, allow_fractional) -> None:
    # Memo on the AnnData object itself. AnnData is NOT hashable (verified, anndata 0.12.16), so
    # WeakKeyDictionary / id()-dict keying is out; a private attribute works and dies with the
    # object (natural per-submission lifetime). try/except so a locked-down object degrades to
    # always-validate (correct, just unmemoized) rather than erroring. (Gemini PR #47.)
    seen = getattr(adata, "_validated_inputs", None)
    if seen is None:
        seen = set()
        try:
            adata._validated_inputs = seen
        except (AttributeError, ValueError, TypeError):  # read-only view / locked-down object -> skip memo
            pass
    key = (input_type, allow_fractional)
    if key in seen:
        return
    _norm.validate_input_type(adata, input_type, allow_fractional=allow_fractional)
    if allow_fractional and input_type == "counts":
        _warn_fractional_allowance_used(adata)
    seen.add(key)


def _warn_fractional_allowance_used(adata) -> None:
    """Say so when ``allow_fractional_counts`` actually let something through.

    ⚠️ The flag is the one DIGEST-EXEMPT bypass of the guard stopping a log1p submission under the
    frozen ``vcc2026`` rule (``validate_input=False`` disables it too, but that field is *not*
    exempt, so a pairing check can see it), and it is in ``baseline.DIGEST_EXEMPT_FIELDS`` -- so ``config_digest``,
    the field ``score``'s baseline/submission pairing check compares, CANNOT SEE IT. Measured on
    ``docs/data/H1-VCC-2025-training.h5ad``: the same log1p submission scored as counts moves every
    scored member (``sig_jaccard`` 0.811 -> 0.003, ``lfc_nmae`` 0.170 -> 1.927, ``pds_cosine``
    1.000 -> 0.450) with a BYTE-IDENTICAL ``config_digest``. The harm is self-inflicted -- every
    member gets worse, so it is a scoring-integrity gap and not an exploit -- but it was silent.

    RULED (2026-08-17, Alex): warn; keep the exemption. Removing the flag from
    ``DIGEST_EXEMPT_FIELDS`` would break EVERY baseline/submission pairing, because both baseline
    producers (``baseline.baseline_config``, ``real_bundle._baseline_leg``) flip it True on an
    internal copy and rely on the exemption so the artifact still pairs. The clean fix -- digest the
    REQUESTED config, as ``build_generic_baseline`` already does for ``allow_discrete`` -- moves
    ``config_digest`` and so invalidates existing baselines: a release decision, in files this chunk
    does not own.

    Fires only when the allowance was LOAD-BEARING, i.e. this matrix is fractional and
    ``validate_input_type`` would otherwise have raised. An inert flag says nothing, which is what
    keeps this from becoming a line people learn to ignore. ``_is_all_integer`` is the validator's
    OWN predicate (the same one ``validate_input_type``'s ``input_type == "counts"`` branch tests)
    rather than an equivalent of mine, deliberately: the warning must fire exactly when the raise
    would have, and a re-implementation could drift from it.

    Three scope limits, all real. The flag is pred-side only (``_val_allow_fractional``), so a True
    value here IS the pred side. Every caller gates on ``validate_input and version != "v1"``, so
    this is silent for ``validate_input=False`` and on v1 -- where nothing is validated at all, a
    wider hole that a warning on this path would not describe. And it cannot fire when the matrix is
    never classified: a BACKED (path) input skips the up-front validation loop, and a result-cache
    hit returns before the deferred sites run, so a warm run on path inputs reaches neither. Forcing
    a classification there would make a cache hit read ``X``, which is the one cost the results
    cache exists to avoid -- so that case gets the cheaper, scan-free
    ``_warn_fractional_allowance_unclassified`` instead (codex-review round 8, P1).
    """
    # Only reached on the permissive path, so the extra integer scan never touches the hot
    # strict path -- and on a baseline arm the matrix is a mean profile, which is cheap.
    if _norm._is_all_integer(adata.X):
        return
    # Worded to what is KNOWN AT EMISSION (codex-review round 8, P2). Two earlier claims were
    # false: this fires BEFORE `_check_scale_limit_once`, so the run may still refuse after it, and
    # "no rescaling" is wrong for a counts path, which normalizes to `target_sum`.
    logger.warning(
        "allow_fractional_counts=True was LOAD-BEARING here: the pred side has fractional values "
        "under declared input_type='counts', so it passed the input-type check ONLY because of the "
        "flag -- without it validation would have refused the run (\"declared input_type='counts' "
        "but values are fractional\"). If this is a BASELINE arm that is expected: a mean profile "
        "is fractional in any space. If this is a SUBMISSION and processing continues (later gates, "
        "including the scale limit, may still refuse it), these values will be INTERPRETED as "
        "counts. Note that config_digest cannot see this flag "
        "(baseline.DIGEST_EXEMPT_FIELDS), so `score --baseline-agg` pairing will NOT report a "
        "baseline/submission mismatch on it; the --real-bundle path IS protected "
        "(anchor._SEMANTIC_FIELDS compares it, via score.expect_from_run_meta)."
    )


def _warn_fractional_allowance_unclassified(cfg, pred_ad) -> None:
    """The scan-free half of the above, for the one path where classification never happens.

    A result-cache hit returns from ``_run_metrics`` before the deferred validation sites run, and a
    BACKED (path) input skips the up-front loop -- so a warm run on path inputs would otherwise be
    completely silent about a permissive flag (codex-review round 8, P1). Reporting the flag rather
    than its effect is deliberate: deciding whether the allowance was load-bearing needs an ``X``
    scan, and making a cache hit read ``X`` would spend exactly the cost the results cache exists to
    avoid.

    Gated on the pred side being backed, because an in-memory pred was already classified up front
    and would double-report. Same ``validate_input`` / v1 gates as every other site, so the scope
    limits the loud version documents hold here too.
    """
    if not (cfg.allow_fractional_counts and cfg.validate_input and cfg.version != "v1"):
        return
    if not getattr(pred_ad, "isbacked", False):
        return          # already classified up front; the loud warning covers it
    logger.warning(
        "allow_fractional_counts=True on a run served from the RESULTS CACHE with a backed (path) "
        "pred input, so whether the allowance was load-bearing was not re-checked -- that would "
        "require reading X, which is the cost the cache exists to avoid. The cached numbers were "
        "produced under this flag. config_digest cannot see it "
        "(baseline.DIGEST_EXEMPT_FIELDS), so `score --baseline-agg` pairing will NOT report a "
        "baseline/submission mismatch on it. Re-run WITHOUT --cache-pred (it is a path, not a "
        "toggle -- there is no --no-cache-pred) to get the checked answer."
    )


def _check_scale_limit_once(adata, input_type, max_counts_per_cell) -> None:
    # Same memo pattern for the scale-limit guard (its expm1 scan is inherently expensive, so
    # skipping repeats is a real CPU win, not just RAM). (Gemini PR #47 R3.)
    seen = getattr(adata, "_validated_scale_limits", None)
    if seen is None:
        seen = set()
        try:
            adata._validated_scale_limits = seen
        except (AttributeError, ValueError, TypeError):
            pass
    key = (input_type, max_counts_per_cell)
    if key in seen:
        return
    # Reuse the pseudobulk accumulator's per-cell max if it ran first and stashed it
    # (counts + GPU-accumulator path, run._run_metrics computes pred pseudobulk before this
    # gate); else check_scale_limit runs its own _row_totals pass.
    precomputed = getattr(adata, "_precomputed_row_total_max", None)
    _norm.check_scale_limit(adata, input_type, max_counts_per_cell,
                            precomputed_row_total_max=precomputed)
    seen.add(key)


def _materialize(source):
    """Full in-memory AnnData for a side's heavy pass: reads the whole file for a
    path, or materializes an already-loaded object (to memory if it was opened backed,
    so the heavy normalize/pseudobulk never runs in slow backed mode)."""
    if isinstance(source, (str, os.PathLike)):
        return load_anndata(source, backed=False)
    return source.to_memory() if getattr(source, "isbacked", False) else source


def _effective_autodetect(cfg, *, side: str) -> bool:
    """autodetect_input_type re-types the PRED submission only; the REAL side trusts the declared
    type (stays strict — Gemini #2). v1 still auto-detects BOTH sides via resolve_input_type's
    version branch, independent of this flag."""
    return cfg.autodetect_input_type and side == "pred"


def _effective_input_type(adata, cfg, *, side: str) -> str:
    """For v1 (or when autodetect_input_type is set, pred side only), auto-detect
    counts-vs-lognorm; otherwise use the declared type."""
    return _norm.resolve_input_type(
        adata, declared=cfg.input_type, version=cfg.version, allow_discrete=cfg.allow_discrete,
        autodetect=_effective_autodetect(cfg, side=side),
    )


def peak_host_rss_bytes() -> int | None:
    """This PROCESS's peak resident set size in bytes, or None where it cannot be read.

    ``ru_maxrss`` is a high-water mark for the whole process and is never reset, so it includes
    anything the caller allocated before this run. That makes it the right number for "did the
    scoring driver fit in the job's --mem" and the wrong number for "what did this call allocate";
    the log line below says so.

    Linux reports KiB, macOS bytes. ``resource`` is absent on Windows.
    """
    try:
        import resource
        import sys
        raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    except Exception:  # noqa: BLE001 - provenance only; never fail a scoring run over it
        return None
    return int(raw) if sys.platform == "darwin" else int(raw) * 1024


def _log_peak_host_rss(pred_ad, real_ad) -> None:
    """#277 item 3: report peak HOST memory as a number on every in-memory scoring run.

    #277 is "the envelope is undocumented and has no margin", not "there is a leak": the pre-v0.8.0
    code's tool-measured peak was 126.53 GiB against a 128 GiB budget -- 1.1% under -- and v0.10.0
    crosses it, so the largest VCC Test submissions are OOM-killed. The failure lands on exactly
    the submissions a leaderboard cares about, and at 128G it truncates the top of a re-scored
    board rather than erroring in a way that points at the cause.

    A logged number, NOT an assertion: an assertion needs an envelope, and what the envelope should
    be is the open question on the issue. This makes a future increase show up in every run's log.
    Deliberately no new EvalConfig field for a threshold either -- to_dict() feeds config_hash, so
    adding one would invalidate every warm result cache for a diagnostic.
    """
    rss = peak_host_rss_bytes()
    if rss is None:
        return
    shape = "?"
    try:
        shape = (f"pred {pred_ad.n_obs}x{pred_ad.n_vars}, real {real_ad.n_obs}x{real_ad.n_vars}")
    except Exception:  # noqa: BLE001
        pass
    logger.info(
        "peak host RSS at end of compute_metrics attempt: %.2f GiB (%s). Reported whether the run "
        "SUCCEEDED or raised -- it is in a finally, so an OOM-adjacent failure still reports the "
        "high-water mark it reached, which is the number #277 is about. Process high-water mark "
        "(ru_maxrss), so it "
        "covers this run AND anything the caller allocated before it -- the number to compare "
        "against a job's --mem, not a per-call allocation. #277: the in-memory driver has no "
        "documented host-memory envelope, and the pre-v0.8.0 peak on the largest VCC Test "
        "submissions was 126.53 GiB against 128 GiB.",
        rss / 2**30, shape,
    )


def _warn_mixed_library_scale(cfg, effective_types: dict[str, str]) -> None:
    """#193: warn when the two sides' effective input types differ under a NUMERIC ``target_sum``,
    because only the counts side is then moved and the two end up at different library scales.

    ``to_normalization`` returns a lognorm matrix UNCHANGED when the target is also lognorm
    (``norm.py``), so with ``target_sum=T``:

        reference (counts)   -> normalize_total(T) -> log1p  =  log1p(T * f_real)
        prediction (lognorm) -> untouched                    =  log1p(S_c * f_pred)

    where ``S_c`` is whatever library scale the prediction's author normalized cell ``c`` to.
    MEASURED on a real VCC Test submission (132,870 predicted cells;
    internal:docs/validation/2026-07-30-generic-response-baseline-real-data.md): at the v2 default
    ``target_sum=1e6`` the two sides sit **14.9x-25.2x apart**, median 18.4x. ``target_sum=None``
    resolves to the reference control pool's median and removes the bulk of it -- 0.82x-1.37x,
    median 1.00x -- but not exactly, because the submission is not normalized to one constant.

    The distortion is NOT a constant additive offset in log1p space: it is
    ``log1p(T*f) - log1p(S_c*f)``, gene-dependent and vanishing at zero expression. So it
    DISTORTS rather than shifts, and ``expr_mae`` / ``expr_mse`` / ``delta_mae`` / ``delta_mse``
    measure the distortion rather than the prediction. ``delta_pearson`` is largely insensitive
    and the DE metrics are computed from per-side ratios, which is why a run can look healthy on
    most of the profile while the error metrics are meaningless.

    A WARNING and not a raise, deliberately: a counts-real / lognorm-pred pair is an explicitly
    SUPPORTED path (#155 spec 8), pinned by
    ``test_mixed_counts_real_lognorm_pred_still_runs_and_is_not_forced_to_one_scale``. Rescaling
    the lognorm side instead (#193's option 2) would change numbers for every existing lognorm run
    and needs a version gate, so it is a release decision rather than a driver fix.

    Cannot fire under the frozen ``vcc2026`` rule, and that is measured rather than assumed: the
    rule pins ``autodetect_input_type: false``, so ``_effective_autodetect`` returns False for
    both sides and both effective types are the declared ``counts``. It takes v1, or
    ``autodetect_input_type=true``, to reach a mixed pair at all.
    """
    pred_eff, real_eff = effective_types["pred"], effective_types["real"]
    if pred_eff == real_eff or cfg.target_sum is None:
        return
    lognorm_side = "pred" if pred_eff == "lognorm" else "real"
    logger.warning(
        "the two sides' effective input types DIFFER (pred=%r, real=%r) and target_sum=%r is "
        "numeric, so only the counts side is normalized to it: the %s side keeps whatever library "
        "scale it already had. The two are then compared at different scales -- measured at "
        "14.9x-25.2x apart (median 18.4x) on a real VCC Test submission at target_sum=1e6 -- and "
        "the error metrics (expr_mae/expr_mse/delta_mae/delta_mse) measure that mismatch rather "
        "than the prediction. It distorts rather than shifts, so delta_pearson and the "
        "ratio-based DE metrics will still look healthy. Pass target_sum=None to normalize to the "
        "reference control pool's median instead (measured 0.82x-1.37x, median 1.00x), or supply "
        "both sides in the same space (#193).",
        pred_eff, real_eff, cfg.target_sum, lognorm_side,
    )


def _val_allow_fractional(cfg, *, side: str) -> bool:
    """allow_fractional_counts applies to the PRED side only; the REAL side is always validated
    strictly (it is ground-truth integer counts). Keeps a permissive predictor from relaxing the
    real-side check (Codex #1)."""
    return cfg.allow_fractional_counts and side == "pred"


def _close_backed(adata, source) -> None:
    """Close a backed AnnData file handle we opened for a path input, to avoid leaking file
    descriptors in long-running/batch callers. Only str/PathLike sources are ours, so a
    caller-supplied (in-memory or backed) AnnData object is never closed out from under them."""
    if isinstance(source, (str, os.PathLike)) and getattr(adata, "isbacked", False):
        try:
            adata.file.close()
        except Exception:  # noqa: BLE001 - best-effort cleanup; never affect the result
            pass


def _needed_normalizations(names: list[str], *, comparator: str) -> list[str]:
    return sorted({
        key for key in (effective_normalization(CATALOG[n], comparator) for n in names)
        if key is not None
    })


def effective_normalization(spec, comparator: str) -> str | None:
    """The accumulator key ``spec`` reads, given the run's resolved comparator.

    ``EXPR_COMPARATOR`` is a DECLARATION of intent; ``comparator`` is the POLICY, resolved
    once per run by ``norm.resolve_comparator``. Every other declared value is already an
    accumulator key and passes through. This is the only place where catalog declaration and
    run policy are joined: both ``_needed_normalizations`` (what is built) and
    ``dispatch_anndata_metrics`` (what is read) must pass through it. Diverging those paths is a
    ``KeyError`` at best and a silently wrong comparator at worst.
    """
    if spec.kind != "anndata" or spec.normalization is None:
        return None
    return comparator if spec.normalization == _norm.EXPR_COMPARATOR else spec.normalization


def _moment_normalizations(names, *, comparator: str) -> set[str]:
    """The normalization keys that must be built WITH per-group moments.

    Moments are a property of the (metric, normalization) pair, not of the run. The expression
    comparator resolves to either `lognorm` (analytic correction) or `bulk_lognorm` (jackknife),
    and only the metrics declaring `needs_moments` request an artifact for that key. One global
    boolean cannot express that.
    """
    return {
        k for k in (effective_normalization(CATALOG[n], comparator)
                    for n in names if n in CATALOG and CATALOG[n].needs_moments)
        if k is not None
    }


def _needs_moments(names) -> bool:
    """True if any resolved metric in ``names`` consumes per-group moments (issue #198).

    Drives BOTH whether the pseudobulk drivers are asked for moments and which cache
    artifact the run reads/writes -- so it must be computed from the resolved metric list,
    once, before any pseudobulk work starts.
    """
    return any(CATALOG[n].needs_moments for n in names if n in CATALOG)


def _reject_moments_metrics(names, driver: str) -> None:
    """Raise if a moments-consuming metric was requested on a driver that cannot supply them.

    Per-driver adoption is deliberate (#198 §4.3): a driver that does not carry moments must
    fail loudly rather than let the metric fall back to the biased value.
    """
    blocked = sorted(n for n in names if n in CATALOG and CATALOG[n].needs_moments)
    if blocked:
        raise NotImplementedError(
            f"metric(s) {blocked} require per-group moments, which the {driver} driver does "
            "not supply. Use the in-memory, shard-streaming, or cell-streaming scorer, or "
            "deselect these metrics."
        )


def dispatch_anndata_metrics(names, pred_bulks, real_bulks, genes, cfg,
                             *, comparator: str, pred_moments=None, real_moments=None,
                             driver="unspecified", pred_bulks_full=None) -> list[dict]:
    """Run the anndata-kind metrics in ``names`` over precomputed pseudobulk dicts.

    Pure helper lifted from ``_run_metrics`` so the streaming scale runner can reuse the
    exact same dispatch (per-metric output naming, signature-filtered kwargs, tidy rows)
    without re-touching the DE branch. ``de``-kind names are skipped here (DE stays in
    ``_run_metrics`` / Plan 2). ``pred_bulks`` / ``real_bulks`` are
    ``{normalization: (perts, means)}`` dicts; returns the tidy ``rows`` list.

    ``pred_moments``/``real_moments`` are the parallel ``{normalization: GroupMoments}``
    dicts (issue #198) or ``None``. They are NOT restricted alongside the bulks: they span
    every group including the control, which is what the delta_magnitude_* family (#202)
    will consume. A metric that needs them and gets ``None`` raises inside the metric func.

    ``pred_bulks_full`` is the pred bulks BEFORE any partial run restricted them, and defaults to
    ``pred_bulks``. Only `expr_mse_unbiased_capped` declares it: #348's correction budget is the
    one term in this unit whose value depends on which OTHER predicted perturbations are present,
    so a driver that restricts (``scale.py``'s two streaming paths) must hand the unrestricted dict
    over as well or that member becomes partition-dependent. Signature-filtered like
    ``target_gene_map``, so no other metric sees it.
    """
    from .gpu import resolve_device

    device = resolve_device(cfg.device)
    rows: list[dict] = []
    for name in names:
        spec = CATALOG[name]
        if spec.derived is not None:
            continue   # produced at aggregation time from its components; it has no func
        if spec.kind == "de":
            continue
        if spec.kind != "anndata":
            raise NotImplementedError(f"metric kind {spec.kind!r} not supported in this unit")
        out_name = spec.v1_name if (cfg.version == "v1" and spec.v1_name) else spec.name
        key = effective_normalization(spec, comparator)
        pred_bulk, real_bulk = pred_bulks[key], real_bulks[key]
        # Discrimination (pds_*) on cuda -> GPU full-matrix kernel; identical scores, big
        # speed win at scale. _disc binds the distance via functools.partial, so the
        # underlying func identity + the bound `distance` keyword identify these metrics.
        # embed_key (obsm-space PDS) is unsupported on the GPU path -> fall through to the
        # CPU func, which raises NotImplementedError as before.
        is_disc = _is_discrimination(spec.func)   # shared with the cache predicate
        if is_disc and device == "cuda" and cfg.discrimination.embed_key is None:
            from .gpu.distances import discrimination_ranks

            result = discrimination_ranks(
                real_bulk, pred_bulk, perts=None, genes=genes,
                metric=spec.func.keywords["distance"],
                exclude_target_gene=cfg.discrimination.exclude_target_gene,
                exclusion_scope=cfg.discrimination.exclusion_scope,
                rank_denominator=cfg.discrimination.rank_denominator,
                tie_policy=cfg.discrimination.tie_policy,
                pert_chunk=cfg.pert_chunk, device=device,
                control=cfg.control, control_source=cfg.control_source,
                # issue #248: without the map, guide-level labels ("ADNP-1") match no
                # gene and the exclusion silently no-ops. Same map the DE side resolves
                # through (`resolve_target_genes` below), so one run has ONE notion of
                # "the target gene" across pds_* and the chance-corrected DE metrics.
                target_gene_map=cfg.target_gene_map,
            )
        else:
            available = {
                "pred_bulk": pred_bulk, "real_bulk": real_bulk,
                "comparator": key,
                "pert_col": cfg.pert_col, "control": cfg.control,
                "control_source": cfg.control_source, "genes": genes,
                "rank_denominator": cfg.discrimination.rank_denominator,
                "tie_policy": cfg.discrimination.tie_policy,
                "exclude_target_gene": cfg.discrimination.exclude_target_gene,
                "exclusion_scope": cfg.discrimination.exclusion_scope,
                # issue #248, CPU twin of the GPU branch above. Signature-filtered below,
                # so only discrimination_score (which declares it) receives it.
                "target_gene_map": cfg.target_gene_map,
                "embed_key": cfg.discrimination.embed_key,
                # Comparator-space mass has a comparator-space denominator. Under the
                # fallback, cfg.target_sum is already resolved against the real control pool
                # (or remains None for already-lognorm input).
                "mass_target": (
                    cfg.bulk_target_sum if comparator == "bulk_lognorm" else cfg.target_sum
                ),
                # issue #348, and the reason it is a separate argument rather than read off
                # `pred_bulk`: a partial run hands over a RESTRICTED pred bulk, and the
                # correction budget must not depend on the partitioning. Signature-filtered,
                # so only `mse_unbiased_capped` (which declares it) receives it.
                "pred_bulk_full": (pred_bulks_full or pred_bulks)[key],
                # parallel moments, keyed by the SAME normalization as the bulks (#198)
                "pred_moments": (pred_moments or {}).get(key),
                "real_moments": (real_moments or {}).get(key),
                # so a metric that needs moments and gets none can NAME the driver that
                # failed to supply them, as the spec's error contract requires
                "driver": driver,
            }
            sig = inspect.signature(spec.func)
            kwargs = {k: v for k, v in available.items() if k in sig.parameters}
            result = spec.func(**kwargs)
        # v2 no-droppable-NaN (issue #92, mirrors #89): map a degenerate/NaN pert to the
        # metric's worst value so it penalizes rather than vanishing from the aggregate mean.
        # v1 keeps the upstream NaN behavior (parity); raw metric funcs stay unchanged.
        if cfg.version != "v1" and spec.worst_value is not None:
            perts = [p_str for p in pred_bulk[0] if (p_str := str(p)) != cfg.control]
            result = _fill_no_drop(result, perts, spec.worst_value)
        for pert, value in result.items():
            rows.append({"perturbation": pert, "metric": out_name, "value": float(value)})
    return rows


def _fill_no_drop(result: dict[str, float], perturbations, worst_value: float) -> dict[str, float]:
    """v2 no-droppable-NaN (issue #89): every in-scope pert gets a finite value; a pert
    that is missing from ``result``, mapped to ``None``, or present with a NaN value maps
    to the metric's worst value. ``result.get(p)`` folds the missing and ``None`` cases
    into a single lookup (both surface as ``None``, and both map to worst); ``val != val``
    is the import-free NaN test (NaN is the only value not equal to itself). Rebuilding
    from ``perturbations`` also drops any stray key not in scope. Never called on the
    v1/parity path, which keeps the upstream omit/NaN behavior."""
    out: dict[str, float] = {}
    for p in perturbations:
        val = result.get(p)
        out[p] = worst_value if (val is None or val != val) else val
    return out


def _effective_de_spec(name, backend):
    """The MetricSpec to emit for a resolved DE metric ``name`` under ``backend``: relabeled
    to the deseq2 family (``de_wilcoxon_<x>`` -> ``de_deseq2_<x>``) when ``backend=='deseq2'``,
    else the name's own spec. Same metric set the user selected; method-correct names out.
    Only called for DE-kind names (after the kind filter in ``dispatch_de_metrics``)."""
    return CATALOG[deseq2_metric_name(name)] if backend == "deseq2" else CATALOG[name]


def _guard_deseq2_metric_selection(names, backend) -> None:
    """A ``de_deseq2_*`` metric can only be computed from a deseq2 DE table. Reject the
    reverse mislabel (deseq2-named metric requested under a rank backend)."""
    if backend != "deseq2" and any(n.startswith("de_deseq2_") for n in names):
        raise ValueError(
            "de_deseq2_* metrics require de.backend='deseq2' "
            f"(got backend={backend!r}); select DE metrics normally and set the backend."
        )


def dispatch_de_metrics(names, prepared_de, cfg) -> list[dict]:
    """Run the de-kind metrics in ``names`` over a PreparedDE. Lifted from ``_run_metrics``
    so the streaming scale runner reuses the exact same dispatch (per-metric naming,
    signature-filtered kwargs, tidy rows). Non-de names are skipped. When
    ``cfg.de.backend == 'deseq2'`` the DE metrics are relabeled to the ``de_deseq2_*``
    family (provenance-correct names for the NB-GLM backend). The reverse-mislabel guard
    lives here — every DE-metric dispatch funnels through this one function — so an explicit
    ``de_deseq2_*`` selection under a rank backend is rejected on all such paths, not just
    ``compute_metrics``. NOTE: this does not make partitioned/manifest scoring deseq2-compatible
    — their downstream nsig reducers hard-code the ``de_wilcoxon_*`` names, so
    ``backend='deseq2'`` with partitioned scoring is unsupported (design spec, out-of-scope)."""
    _guard_deseq2_metric_selection(names, cfg.de.backend)
    rows: list[dict] = []
    seen: set[str] = set()
    for name in names:
        spec = CATALOG[name]
        if spec.derived is not None:
            continue   # produced at aggregation time from its components; it has no func
        if spec.kind != "de":
            continue
        spec = _effective_de_spec(name, cfg.de.backend)  # de_wilcoxon_* -> de_deseq2_* under deseq2
        # under deseq2, a metric and its wilcoxon sibling (e.g. de_wilcoxon_overlap +
        # de_deseq2_overlap, both explicitly selected) relabel to the same spec — emit once,
        # order-preservingly, so no duplicate (perturbation, metric) rows are produced.
        if spec.name in seen:
            continue
        seen.add(spec.name)
        out_name = spec.v1_name if (cfg.version == "v1" and spec.v1_name) else spec.name
        de_available = {
            "prepared": prepared_de,
            "auc_pval_floor": cfg.de.auc_pval_floor,
            "auc_pval_floor_value": cfg.de.auc_pval_floor_value,
        }
        sig = inspect.signature(spec.func)
        kwargs = {k: v for k, v in de_available.items() if k in sig.parameters}
        result = spec.func(**kwargs)
        # v2 no-droppable-NaN (issue #89): map omitted/NaN perts to the metric's worst
        # value so they penalize rather than vanish from the aggregate mean. v1 keeps the
        # upstream omit/NaN behavior (parity), so the raw funcs stay byte-for-byte unchanged.
        if cfg.version != "v1" and spec.worst_value is not None:
            result = _fill_no_drop(result, prepared_de.perturbations, spec.worst_value)
        for pert, value in result.items():
            rows.append({"perturbation": pert, "metric": out_name, "value": float(value)})
    return rows


def _use_gpu_pseudobulk(device: str, input_type: str, target_sum, target=None) -> bool:
    """GPU pseudobulk applies iff the device is CUDA, the matrix is raw counts (the
    accumulator computes CPM from counts), target_sum is numeric (the per-block
    accumulator cannot do the v1 median, target_sum=None), and -- when a target
    normalization is given -- it is one the accumulator supports, so an
    accumulator-unsupported norm falls back to the CPU path instead of raising.
    Otherwise the CPU to_normalization+pseudobulk path runs (CPU numbers unchanged, fp64)."""
    if target == "bulk_lognorm":
        # The accumulator sums counts and transforms at finalize; no per-cell CPM, so no
        # target_sum is needed and the v1-median exclusion does not apply (issue #264).
        return device == "cuda" and input_type == "counts"
    return (device == "cuda" and input_type == "counts" and target_sum is not None
            and (target is None or target in ("counts", "normalized", "lognorm")))


def _cache_device(cfg) -> str:
    """Device value for cache keys. Only "auto" is machine-dependent (resolves to cuda vs cpu);
    an explicit "cuda"/"cpu" is used VERBATIM so building a cache key never calls resolve_device
    for an already-concrete value -- resolve_device("cuda") raises in a no-cupy install, which
    must not make caching break an otherwise-runnable job (Copilot review, PR #114)."""
    return resolve_device(cfg.device) if cfg.device == "auto" else cfg.device


def _cache_backend(cfg) -> str:
    """DE backend for cache keys. Resolve only "auto" (machine-dependent gpudge/pdex/scanpy); an
    explicit backend is used VERBATIM (don't resolve -- or risk raising for -- a concrete backend
    just to build a key)."""
    if cfg.de.backend != "auto":
        return cfg.de.backend
    from .de_compute import _resolve_backend
    return _resolve_backend(cfg.de.backend)


def _de_supplied_strict(supplied: bool, cfg) -> bool:
    """Whether a DE table's fingerprint must be STRICT (full parquet content). SUPPLIED tables are
    always strict, regardless of cfg.cache_strict: their stats (p_adj/log2_fold_change/p_value) are
    external input that nothing else in the cache key characterizes, so a non-strict fingerprint
    (row count + column names + target/feature value-counts only) lets two supplied tables sharing
    (target, feature) structure but differing in stats collide on the result AND rank caches,
    serving a stale wrong result (ultrareview F9.1). A COMPUTED table needs no such treatment -- its
    content is already pinned by the adata fingerprint + DE config + resolved backend in the key."""
    return supplied or cfg.cache_strict


def _side_bulks(source, *, fp, store, norms, cfg, side, effective_input_type=None,
                moment_norms: set[str] | None = None):
    """{normalization: (perts, means)} for one side, or ``(bulks, {normalization:
    GroupMoments})`` when ``moment_norms`` is non-empty (issue #198). Materializes the full side at
    most once (only on a cache miss) and validates it then; with a store, each
    normalization's pseudobulk is loaded from / written to the L2 disk cache.
    `side` ('real'/'pred') scopes allow_fractional_counts to the pred side (Codex #1).

    Moments use a SEPARATE cache key (``pseudobulk_moments_{target}``) and a separate
    handler: cache.py keeps one manifest entry per key and put() GCs the superseded file, so
    distinguishing the two by ``params`` on a shared key would make moments and non-moments
    runs invalidate each other on every alternation.
    """
    norms = list(norms)
    moment_norms = set(moment_norms or ())
    unknown_moment_norms = moment_norms - set(norms)
    if unknown_moment_norms:
        raise ValueError(
            f"moment_norms must be a subset of norms; unknown {sorted(unknown_moment_norms)}"
        )
    state = {}

    # #155: partitioned callers resolve the effective input type ONCE per scoring unit and pass
    # it here. Without this, _side_bulks re-runs _effective_input_type on whatever matrix it was
    # handed -- once per BATCH inside _build_reference_streaming_core -- and for v1 that
    # autodetect is unconditional (norm.resolve_input_type:205), so rebinding cfg.input_type
    # cannot suppress it. Two batches of one real side could then autodetect differently, which
    # is the same mem_budget-dependence #155 is about. Default None = today's behaviour, so
    # run._run_metrics (one materialization per side, not split-dependent) is untouched.
    def _eff(adata):
        return effective_input_type or _effective_input_type(adata, cfg, side=side)

    def full():
        if "ad" not in state:
            adata = _materialize(source)
            # Validate path inputs AND already-loaded *backed* objects on materialize;
            # non-backed in-memory inputs were validated up front in compute_metrics.
            if isinstance(source, (str, os.PathLike)) or getattr(source, "isbacked", False):
                eff = _eff(adata)
                if cfg.validate_input and cfg.version != "v1":
                    _validate_input_once(adata, eff, allow_fractional=_val_allow_fractional(cfg, side=side))
                if cfg.validate_input:
                    _check_scale_limit_once(adata, eff, cfg.max_counts_per_cell)
            state["ad"] = adata
        return state["ad"]

    out = {}
    moments = {} if moment_norms else None
    for target in norms:
        target_with_moments = target in moment_norms

        def compute(target=target, target_with_moments=target_with_moments):
            adata = full()
            eff = _eff(adata)
            # GPU pseudobulk: accumulator over row-blocks of the resident X -- no full
            # normalize transient. Per-target call (norms=[target]) keeps the cache
            # get_or_compute structure untouched; for the single-norm anndata profile this
            # is one pass. Multi-norm batching is a deferred optimization.
            if _use_gpu_pseudobulk(resolve_device(cfg.device), eff, cfg.target_sum, target):
                if target_with_moments:
                    bulks, moms = inmem_pseudobulk(
                        adata, pert_col=cfg.pert_col, norms=[target],
                        target_sum=cfg.target_sum, device="cuda", with_moments=True,
                        bulk_target_sum=cfg.bulk_target_sum,
                    )
                    return bulks[target], moms[target]
                return inmem_pseudobulk(
                    adata, pert_col=cfg.pert_col, norms=[target],
                    target_sum=cfg.target_sum, device="cuda",
                    bulk_target_sum=cfg.bulk_target_sum,
                )[target]
            if target == "bulk_lognorm":
                # to_normalization cannot express this target (spec A1) -- it is a group-sum
                # transform. Go straight to the pseudobulk seam. This branch is the CPU
                # in-memory path's ONLY route to bulk_lognorm; it does not go through
                # inmem_pseudobulk, so the moments arm must be handled HERE.
                if target_with_moments:
                    perts, means, moms = pseudobulk_bulk_lognorm_with_moments(
                        adata, cfg.pert_col, bulk_target_sum=cfg.bulk_target_sum)
                    return (perts, means), moms
                return pseudobulk_bulk_lognorm(
                    adata, cfg.pert_col, bulk_target_sum=cfg.bulk_target_sum
                )
            normalized = _norm.to_normalization(adata, eff, target, target_sum=cfg.target_sum)
            if target_with_moments:
                perts, means, moms = pseudobulk_with_moments(normalized, cfg.pert_col)
                return (perts, means), moms
            return pseudobulk(normalized, cfg.pert_col)
        # max_counts_per_cell is the scale-limit gate: include it so lowering the limit
        # invalidates the cache and re-runs validation (a hit otherwise skips the check).
        # version: pseudobulk is now version-dependent via _effective_input_type
        # (v1 guesses counts/lognorm; v2 trusts declared) -> key on it (Copilot PR #16).
        params = {"pert_col": cfg.pert_col, "input_type": cfg.input_type, "target": target,
                  "version": cfg.version,
                  # fp32 (GPU accumulator) vs fp64 (CPU) means are chosen by the resolved
                  # device, so key on it -- else a cpu run (chosen for fp64 bit-exact refs)
                  # can be served fp32 GPU means from a shared cache (F2.1). Mirrors the
                  # streaming sibling scale._real_reference, which already does this.
                  "device": _cache_device(cfg),
                  "target_sum": cfg.target_sum,
                  "bulk_target_sum": cfg.bulk_target_sum,
                  "allow_discrete": cfg.allow_discrete,
                  # effective per-side flags: autodetect changes the input type on the pred
                  # side (value-affecting) and allow_fractional gates validation; keying on the
                  # per-side value keeps a permissive fill from being reused by a stricter run
                  # without spuriously invalidating the always-strict real side (Codex #2).
                  "autodetect_input_type": _effective_autodetect(cfg, side=side),
                  "allow_fractional_counts": _val_allow_fractional(cfg, side=side),
                  # #161: the MASTER validation switch, missed when its three neighbours
                  # above were added. Without it a validate_input=False run fills an
                  # artifact that is then served to a validate_input=True run, which never
                  # executes the guard it asked for.
                  "validate_input": cfg.validate_input,
                  "max_counts_per_cell": cfg.max_counts_per_cell}
        if target == "bulk_lognorm":
            # #271. This artifact IS the object that moved -- `pseudobulk_bulk_lognorm` reduces
            # through `prep._grouped_sums`, and nothing else in this params dict can see that its
            # reduction dtype changed (every other key describes what was ASKED FOR). Scoped to the one target that reaches that helper, so a
            # `lognorm` artifact (which goes through `_grouped_means`) keeps its warm entry.
            # Covers the moments sibling too: `jk` is computed from the same group sums.
            params["grouped_sum_reduction_semantics"] = _GROUPED_SUM_REDUCTION_SEMANTICS
        key = (f"pseudobulk_moments_{target}" if target_with_moments
               else f"pseudobulk_{target}")
        kind = "npz_moments" if target_with_moments else "npz"
        if store is None:
            value = compute()
        else:
            value = store.get_or_compute(key, fingerprint=fp, params=params, kind=kind,
                                         compute=compute)
        if target_with_moments:
            out[target], moments[target] = value
        else:
            out[target] = value
    return (out, moments) if moment_norms else out


def _prepare_de_cached(de_pred, de_real, *, cfg, real_store, pred_store,
                       de_real_supplied, de_pred_supplied):
    """PreparedDE with each side's rank matrix served from / written to its L2 cache.
    The cheap load+normalize (prep_de_side) always runs (needed for the target set and
    the fingerprint); only the rank pivot is cached."""
    key = f"de_{cfg.de.method}_rank"
    params = {"sort_by": cfg.de.sort_by, "p_adj_threshold": cfg.de.p_adj_threshold,
              "nan_lfc_policy": cfg.de.nan_lfc_policy,
              "min_abs_log2fc": cfg.de.min_abs_log2fc}

    def side(src, name, store, supplied):
        df, perts = prep_de_side(src, name=name, sort_by=cfg.de.sort_by,
                                 nan_lfc_policy=cfg.de.nan_lfc_policy,
                                 min_abs_log2fc=cfg.de.min_abs_log2fc)

        def compute():
            return rank_de_side(df, sort_by=cfg.de.sort_by, p_adj_threshold=cfg.de.p_adj_threshold)

        if store is None:
            return compute(), perts, df
        # ALWAYS strict for the rank artifact. Every OTHER artifact key carries the knobs that
        # determine how its artifact is GENERATED -- the pseudobulk key ten, the DE-table key ~20
        # -- while this one is keyed on a DERIVED table (the DE table) through a fingerprint that
        # is value-blind by default, plus 4 rank knobs, and carries nothing at all about how that
        # table was produced. So every DE-generation knob is invisible here: the table recomputes
        # correctly and the STALE RANK is served. Enumerating the content knobs would reopen the
        # hand-maintained-allowlist bug the audit named as systemic, so it hashes content instead
        # (ultrareview 2026-07-25). Unchanged elsewhere: the two _de_supplied_strict calls that
        # build compute_metrics' RESULT fingerprint, and precompute_cache's supplied-DE branch --
        # the other writer of this same rank artifact -- which already passes supplied=True
        # literally, so it is always strict and now hashes identically to this line.
        fp = fingerprint_de_table(df, strict=True)
        hit = store.get(key, fingerprint=fp, params=params, kind="parquet")
        if hit is not MISS:
            return hit, perts, df
        rank = compute()
        # A zero-column rank means no gene is significant for any target (degenerate DE
        # table). It is deliberately NOT cached: an empty pl.DataFrame is not
        # parquet-serializable, and _rank_matrix returns it via an early exit after a cheap
        # filter (before the expensive pivot), so recompute is ~free. Full re-runs are
        # short-circuited by the result cache anyway.
        if rank.width > 0:
            store.put(key, rank, fingerprint=fp, params=params, kind="parquet")
        return rank, perts, df

    real_rank, real_perts, real_df = side(de_real, "real", real_store, de_real_supplied)
    pred_rank, pred_perts, pred_df = side(de_pred, "pred", pred_store, de_pred_supplied)
    # real_df here is the WHOLE truth table -- this path never slices -- so resolving from
    # it is dataset-level by construction (spec 2.7b).
    resolution = resolve_target_genes(real_df, real_perts,
                                      target_gene_map=cfg.target_gene_map)
    return assemble_prepared_de(real_rank, real_perts, pred_rank, pred_perts,
                                control=cfg.control, sort_by=cfg.de.sort_by,
                                p_adj_threshold=cfg.de.p_adj_threshold,
                                real_df=real_df, pred_df=pred_df,
                                target_resolution=resolution)


def _use_inmem_external_ref(cfg) -> bool:
    """True iff the in-memory pred DE should pass the real control as a SEPARATE gpudge
    external reference (gpudge_arc #67) rather than concatenating it into the predictions.
    Only for control_source='real' + a resolved gpudge backend; CPU backends (pdex/scanpy)
    have no external-ref capability and must concat."""
    if cfg.control_source != "real":
        return False
    # Resolve ONLY "auto" -- the same doctrine as _cache_backend above, and for the same reason:
    # a warm DE-table hit is served by _compute_de_side's store without ever calling compute_de,
    # so an explicit backend that is not installed must not fail a run that can complete from
    # cache. (This PR's _DE_RESULT_SEMANTICS bump makes exactly that shape -- warm table, cold
    # result -- more likely.) An explicit backend picks the layout verbatim: only gpudge has the
    # external-reference capability, every other backend concats.
    if cfg.de.backend != "auto":
        return cfg.de.backend == "gpudge"
    # No try/except on the "auto" path. It is only reached from _pred_de_input, which _run_metrics
    # calls only when a DE metric is requested and the pred table must be computed, and 'auto' now
    # RAISES on a CUDA host without gpudge (ultrareview 2026-07-25). Swallowing that would defer
    # an actionable policy error past the CPU concat below -- documented as OOM-prone at CCL_2
    # scale -- so the user could get an OOM instead of the error telling them what to install.
    from .de_compute import _resolve_backend
    return _resolve_backend("auto") == "gpudge"


def _pred_de_input(pred_ad, real_ad, *, cfg):
    """The DE inputs for the PRED side, as ``(target_adata, reference_adata_or_None)``.

    - control_source != 'real': returns ``(pred_ad, None)`` unmaterialized (backed/in-memory
      as given); materialization is deferred to the DE-table cache-miss closure, so a warm
      cache hit never loads X (Gemini #1). The control is an in-adata group label.
    - control_source == 'real': substitutes the real control as the reference by combining
      pred's non-control cells with real's control cells (genes aligned by validate_pair).
      The real control is materialized and put on the pred's scale EAGERLY. Then:
      * gpudge backend (``_use_inmem_external_ref``): returns
        ``(pred_ad, real_ctrl)`` — the FULL predictions (NO subset/copy) as the DE targets +
        a SEPARATE real control pool passed to gpudge #67's in-memory external-ref DE, with NO
        target/reference concat. gpudge ranks every group (incl. the control) vs the pool and
        compute_de drops the control group's spurious rows (control_group=cfg.control). Passing
        the full pred — rather than a non-control subset copy — avoids the ~2x host-RAM
        transient (the subset copy on top of the still-held full pred) that OOM'd at ~5M cells
        (CCL_2), and the int64-index/super-linear host RAM that the concat OOM'd on.
      * CPU backends: returns ``(concat(pred_non_ctrl, real_ctrl), None)`` — control folded
        in as a group, as before. Under 'real' a warm pred-DE-cache hit still loads the pred
        non-control X; fully deferring this is a tracked optimization. The reused REAL side is
        unaffected — it goes through _compute_de_side directly and stays lazy."""
    if cfg.control_source != "real":
        return pred_ad, None  # backed/in-memory as given; materialized lazily in _compute_de_side
    external_ref = _use_inmem_external_ref(cfg)
    # .copy() the control pool so it is a standalone AnnData (a real csr_matrix for gpudge's
    # external reference / for the concat), not a view that would pin the full real_ad alive after
    # `del real_ad` in callers. Control is ~1.4% of cells, so it's cheap.
    real_ctrl = _materialize(real_ad[real_ad.obs[cfg.pert_col].astype(str) == cfg.control]).copy()
    if real_ctrl.n_obs == 0:
        raise ValueError(f"control {cfg.control!r} absent from the real side; cannot substitute")
    # DE targets. External-ref (gpudge #67): the FULL pred, passed as-is (NO subset/copy). The
    # control group rides along and its spurious control-vs-refpool rows are dropped by compute_de
    # (control_group=cfg.control). Materializing the non-control subset (98.6% of pred) as a real
    # matrix would transiently ~2x host RAM while the caller still holds the full pred_ad -> OOM at
    # ~5M cells (CCL_2); the full pred (a real AnnData from _load, not a view/copy) has no such
    # transient (221.5 GiB standalone proof). CPU concat path: fold the real control in as a group,
    # so materialize pred's non-control cells for the concat.
    pred_de_targets = (
        pred_ad if external_ref else
        _materialize(pred_ad[pred_ad.obs[cfg.pert_col].astype(str) != cfg.control])
    )
    # Scale invariant: the substituted real control must be on the SAME scale as the predictions
    # before they are combined (else the concat mixes e.g. a log-norm pred with raw-counts control,
    # distorting DE / tripping the scale-limit gate). If the effective input types differ, convert
    # the real control to the pred's scale (counts->lognorm via target_sum; lognorm->counts raises).
    # This conversion applies to BOTH the concat and the external-ref path (so the separate
    # reference pool is on the pred's scale, and compute_de normalizes both sides identically).
    # Detecting pred_eff from the full pred (external-ref) vs the non-control subset (concat) is
    # equivalent: the control cells are the same counts/lognorm convention as the targets.
    pred_eff = _effective_input_type(pred_de_targets, cfg, side="pred")
    real_eff = _effective_input_type(real_ctrl, cfg, side="real")
    # The combined pred-DE matrix is validated downstream as the PRED side (possibly permissive),
    # so validate the real-control slice strictly HERE before it is folded in (real side is always
    # strict — Codex #1/P1). No-op for the always-integer real control; gated like the other sites.
    if cfg.validate_input and cfg.version != "v1":
        _validate_input_once(real_ctrl, real_eff, allow_fractional=False)
    if real_eff != pred_eff:
        real_ctrl = _norm.to_normalization(real_ctrl, real_eff, pred_eff, target_sum=cfg.target_sum)
    if external_ref:
        return pred_de_targets, real_ctrl  # (full pred targets, separate real control pool) — gpudge #67
    return ad.concat([pred_de_targets, real_ctrl], join="inner", merge="first"), None  # CPU: concat


def _compute_de_side(adata, *, cfg, fp, store, side, reference_adata=None):
    from .de_compute import compute_de
    # reference_adata set -> gpudge in-memory external-ref (adata holds targets only, control is a
    # SEPARATE pool); the control-in-adata check applies only to the in-adata (str-label) path.
    if reference_adata is None and cfg.control not in set(adata.obs[cfg.pert_col].astype(str)):
        raise ValueError(
            f"control {cfg.control!r} not present in the side being DE-computed; "
            "DE is computed vs the control group"
        )

    def compute():
        # Materialize ONLY here, so a warm DE-table cache hit never loads X (Gemini #2).
        # (Exception: control_source='real' assembles the combined view eagerly in
        # _pred_de_input — see its docstring.) Validate the materialized matrix against
        # the declared input_type / scale here too: in cache mode the up-front check
        # skips backed inputs, and once DE is COMPUTED from X the matrix is no longer
        # "unused", so a mislabeled backed/path input must not slip through (Copilot #1).
        mat = _materialize(adata)
        eff = _effective_input_type(mat, cfg, side=side)
        if cfg.validate_input and cfg.version != "v1":
            _validate_input_once(mat, eff, allow_fractional=_val_allow_fractional(cfg, side=side))
        if cfg.validate_input:
            _check_scale_limit_once(mat, eff, cfg.max_counts_per_cell)
        return compute_de(
            mat, backend=cfg.de.backend, groupby=cfg.pert_col,
            reference=(reference_adata if reference_adata is not None else cfg.control),
            # external-ref only: mat is the FULL pred (incl. control); drop the control group's
            # spurious rows from the gpudge output (control_group). No-op on the string path.
            control_group=(cfg.control if reference_adata is not None else None),
            mean_calc=cfg.de.mean_calc, epsilon=cfg.de.epsilon, input_type=eff,
            target_sum=cfg.target_sum,
            clip_value=cfg.de.clip_value,
            fdr_scope=cfg.de.fdr_scope,
            filter_gene_min_cpm_cell=cfg.filter.filter_gene_min_cpm_cell, threads=cfg.num_threads,
            # deseq2 backend only: replicate_col defines the pseudobulk grouping; device selects
            # its CPU (numpy) vs GPU (jax fit_contrasts) fit. Inert for gpudge/pdex/scanpy.
            replicate_col=cfg.de.replicate_col, device=cfg.device,
        )

    if store is None or fp is None:
        return compute()
    # version: DE compute is now version-dependent via _effective_input_type (the v1
    # counts/lognorm guess) on top of the version-scoped values below -> key on it so a
    # version flip can't reuse a table computed under the other version (Copilot PR #16).
    # "auto" resolves to gpudge/pdex/scanpy by hardware, and those engines' DE numbers differ
    # (~1e-5), so key on the RESOLVED backend -- else a GPU node (gpudge) and a CPU node (pdex)
    # collide on a shared cache under the identical "auto" config and false-hit (F2.2).
    params = {"backend": _cache_backend(cfg), "mean_calc": cfg.de.mean_calc,
              "epsilon": cfg.de.epsilon, "method": cfg.de.method,
              "filter_gene_min_cpm_cell": cfg.filter.filter_gene_min_cpm_cell,
              "version": cfg.version,
              "target_sum": cfg.target_sum,
              "clip_value": cfg.de.clip_value,
              "fdr_scope": cfg.de.fdr_scope,
              "allow_discrete": cfg.allow_discrete,
              # effective per-side flags: autodetect changes the input type on the pred side
              # (value-affecting); allow_fractional + max_counts_per_cell gate validation/scale-checks
              # a cache hit would skip. Keying on the per-side value keeps a permissive fill from
              # being reused by a stricter run without spuriously invalidating the real side (Codex #2/P2).
              "autodetect_input_type": _effective_autodetect(cfg, side=side),
              "allow_fractional_counts": _val_allow_fractional(cfg, side=side),
              # #161: the MASTER validation switch, missed when its three neighbours above
              # were added. A permissive run's DE table must not be served to a run that
              # asked for the guard -- and once a competition score is derived from this
              # cache, that is a number where there should have been a refusal.
              "validate_input": cfg.validate_input,
              "max_counts_per_cell": cfg.max_counts_per_cell,
              # compute_de uses control as the reference group and pert_col as groupby, so both
              # change the DE table; key on them or a config change false-hits a stale table
              # (Gemini PR #35). pert_col is also in the adata fingerprint, but key explicitly.
              "control": cfg.control, "pert_col": cfg.pert_col,
              "input_type": cfg.input_type, "control_source": cfg.control_source}
    # deseq2 backend (never auto-selected, so backend=="deseq2" ⇔ deseq2 actually runs): its
    # pseudobulk replicate grouping AND its CPU(numpy,fp64)/GPU(jax) fit both change the DE table.
    # Key on both ONLY for deseq2, so (a) every other backend's existing cache key is unchanged (no
    # invalidation), (b) a non-deseq2 run that merely has replicate_col set doesn't needlessly split
    # the cache, and (c) a deseq2 CPU table can't false-hit a GPU-computed one. The GPU predicate
    # mirrors compute_de's use_gpu (raw device.startswith("cuda"); "auto"->CPU), NOT the resolved
    # _cache_device (which maps "auto"->cuda) -- so "auto" (CPU fit) and "cuda" (GPU fit) stay
    # distinct even on a GPU host (Copilot PR #120).
    if cfg.de.backend == "deseq2":
        params["replicate_col"] = cfg.de.replicate_col
        params["deseq2_gpu_fit"] = str(cfg.device).startswith("cuda")
        # #271: this backend's pseudobulk goes through `prep._grouped_sums`, whose reduction dtype
        # changed, so the table it fits moved. deseq2-only for the same reason the two keys above
        # are: no other backend builds a pseudobulk, so none of their caches is invalidated. The
        # RANK artifact needs no term -- it is keyed on a STRICT content hash of this table
        # (`fingerprint_de_table(df, strict=True)`), so it re-misses on its own once this moves.
        params["grouped_sum_reduction_semantics"] = _GROUPED_SUM_REDUCTION_SEMANTICS
    if reference_adata is not None:
        # gpudge in-memory external-ref (control_source='real'): the DE input is (targets, a
        # SEPARATE reference pool), NOT the concat. Add the mode + reference content to the key so
        # an external-ref table never false-hits a concat-built table. These keys are added ONLY on
        # the external-ref path, so the real-side and CPU-concat cache keys are unchanged (existing
        # populated caches keep hitting).
        params["de_input_mode"] = "external_ref"
        params["reference_fp"] = fingerprint_adata(reference_adata, pert_col=cfg.pert_col,
                                                   strict=cfg.cache_strict)
    return store.get_or_compute(f"de_{cfg.de.method}_table", fingerprint=fp,
                                params=params, kind="parquet", compute=compute)


def _materialize_de_sides(pred_ad, real_ad, *, cfg, de_real, de_pred, real_fp,
                          real_store, pred_store):
    """Return both DE tables, computing whichever side was not supplied.

    Factored out because _run_metrics needs it from TWO places -- the ordinary metric path
    and the results-cache hit that --write-degenes has to serve -- and the pred side is not
    a one-liner (external-ref mode, its own fingerprint, the deliberate release below).
    """
    if de_real is None:
        de_real = _compute_de_side(real_ad, cfg=cfg, fp=real_fp, store=real_store, side="real")
    if de_pred is None:
        # _pred_de_input slices backed inputs lazily (materializing only subsets, or
        # nothing in the non-'real' case); the fingerprint reads metadata only.
        # pred_ref is non-None only on the gpudge external-ref path (control_source='real'):
        # a SEPARATE control pool passed as the DE reference, no concat.
        pred_de_in, pred_ref = _pred_de_input(pred_ad, real_ad, cfg=cfg)
        pfp = fingerprint_adata(pred_de_in, pert_col=cfg.pert_col, strict=cfg.cache_strict) \
            if pred_store is not None else None
        de_pred = _compute_de_side(pred_de_in, cfg=cfg, fp=pfp, store=pred_store, side="pred",
                                   reference_adata=pred_ref)
        # On the external-ref path pred_de_in IS pred_ad itself (the full predictions, no
        # subset/copy — see _pred_de_input), so this `del` only drops the local alias; pred_ad
        # stays alive for pseudobulk (still held by the caller). No concat is allocated either
        # way; the concat path's pred_de_in is a fresh non-control subset freed here. Returning
        # frees them anyway; kept explicit so the release still reads as deliberate.
        del pred_de_in  # release the (possibly pred-sized) DE input before pseudobulk
        if pred_ref is not None:
            del pred_ref
    return de_real, de_pred


def _write_de_tables(de_real, de_pred, *, cfg):
    """Write both DE tables to the outdir as de_real.parquet / de_pred.parquet.

    These are the full per-(target, feature) tables (loaded or freshly computed), BEFORE
    ranking/thresholding -- the artifact --write-degenes asks for.

    RETURNS the loaded frames, and the caller MUST rebind de_real/de_pred to them. A
    supplied side is still a bare path here (the load_de_table rebinding above the result
    key only runs when pred_store is set), and these writes can land on a path the other
    side was supplied from -- so anything that re-reads the original arguments afterwards
    reads what this function just overwrote. Returning the snapshots is what keeps the
    scored tables and the emitted files the ones the caller actually asked for.

    The guarantee is ROLLBACK ON ERROR, not atomic visibility as a pair (Copilot, PR #292).
    Both tables are validated, then staged, then swapped in, with any pre-existing destination
    held aside and restored if publication does not complete -- so no failure of this function
    leaves a half-published pair behind. The two swaps are still sequential, so a reader in
    another process can catch one new file beside one old one even when nothing goes wrong, and
    a crash between them leaves exactly that on disk. Closing either needs a directory-level
    commit no filesystem here offers.
    """
    from .de import load_de_table, normalize_de_schema
    if cfg.outdir is None:  # compute_metrics rejects this up front; guards direct _run_metrics use
        raise ValueError("write_de=True requires config.outdir to be set")
    outdir = cfg.outdir
    # BOTH loaded before EITHER is written, for that same aliasing reason: writing
    # de_real.parquet first would otherwise clobber the file --de-pred names before it is
    # read.
    real_tbl, pred_tbl = load_de_table(de_real), load_de_table(de_pred)
    # An outdir that exists and is not a directory can never become one, and `os.makedirs`
    # reports it as a bare `FileExistsError: [Errno 17] File exists: <path>` (measured) with no
    # hint of what the caller did wrong. Same door-check `real_bundle.py` grew for the same
    # reason in #290 (Copilot, PR #292).
    if os.path.exists(outdir) and not os.path.isdir(outdir):
        raise ValueError(
            f"outdir {outdir!r} exists and is not a directory, so --write-degenes cannot write "
            "its DE tables there. Remove it or choose another output directory."
        )
    os.makedirs(outdir, exist_ok=True)
    # VALIDATED before anything is published (codex checkpoint-2 round 3 P2). `_prepare_de_cached`
    # raises on a schema-invalid table anyway, but it runs AFTER the write, so the run died with
    # two published files describing an input it had already rejected. This only moves the timing
    # of an error that was always going to be raised; it cannot change a successful run's metrics.
    # The normalized frames are DISCARDED on purpose -- the artifact is the table as supplied,
    # before normalization as well as before ranking/thresholding. The cost is that a null-column
    # warning is logged twice for a table that has one.
    for tbl, side in ((real_tbl, "real"), (pred_tbl, "pred")):
        normalize_de_schema(tbl, name=f"{side} (--write-degenes)")
    for name in ("de_real.parquet", "de_pred.parquet"):
        # A destination that is not a regular file cannot be published onto: os.replace would
        # move a DIRECTORY aside and then leave it as undeletable backup debris. Refuse at the
        # door, the way real_bundle.py refuses a non-directory outdir.
        dest = os.path.join(outdir, name)
        if os.path.exists(dest) and not os.path.isfile(dest):
            raise ValueError(
                f"{dest!r} exists and is not a regular file, so --write-degenes cannot publish "
                "there. Remove it or choose another output directory."
            )
    # STAGED, then published with os.replace, then ROLLED BACK as a pair on any failure
    # (codex checkpoint-2 P1, rounds 1-2). Writing straight to the destinations publishes them
    # one at a time, so a failure on the second leaves a NEW de_real.parquet beside a stale or
    # missing de_pred.parquet -- and, in the aliasing case this function exists for, that
    # half-write lands ON the file the other side was supplied from, destroying the caller's
    # own input. Staging alone is not enough: a failure of the second os.replace still leaves
    # the first destination replaced, so an existing destination is renamed aside first and
    # restored if publication does not complete. What this does NOT buy is atomic visibility as
    # a pair: the two renames are sequential, so a concurrent reader can catch one new file
    # beside one old one on the success path, and a crash (SIGKILL, power loss) between them
    # leaves that state on disk. Both need a directory-level commit no filesystem here offers.
    #
    # ⚠️ NOT tempfile.mkstemp: it creates the file 0600, and os.replace carries that mode onto
    # the PUBLISHED artifact -- measured 0o600 where a direct write_parquet gives 0o644. An
    # emitted table only its author can read is a regression on a shared filesystem, and it is
    # the same shape as the mkdtemp-0700 leak in the real-bundle publisher. A plain name in the
    # destination directory stays on the same filesystem (so the rename is atomic) and lets
    # polars create the file under the caller's own umask. The suffix is `cache._atomic_write`'s:
    # uuid4 because a pid is NOT unique across threads of one process, nor across containers
    # with namespaced pids sharing a volume; the pid is kept only to make debris attributable.
    uniq = f"{os.getpid()}.{uuid.uuid4().hex}"
    staged: list[tuple[str, str]] = []      # (tmp, dest) still needing cleanup
    backups: list[tuple[str, str]] = []     # (dest, bak) for destinations that existed
    published: list[str] = []

    def _rm(path):
        try:
            os.unlink(path)
        except OSError:      # never let cleanup mask the exception that triggered it
            pass

    try:
        for tbl, name in ((real_tbl, "de_real.parquet"), (pred_tbl, "de_pred.parquet")):
            tmp = os.path.join(outdir, f".{name}.tmp.{uniq}")
            staged.append((tmp, os.path.join(outdir, name)))
            tbl.write_parquet(tmp)
        try:
            for tmp, dest in list(staged):
                if os.path.exists(dest):
                    bak = f"{dest}.bak.{uniq}"
                    os.replace(dest, bak)
                    backups.append((dest, bak))
                os.replace(tmp, dest)
                # DROPPED from `staged` the moment it is published: the name now belongs to the
                # destination, and leaving it in the cleanup list risks unlinking whatever else
                # later occupies it (codex round 2 P2).
                staged.remove((tmp, dest))
                published.append(dest)
        except Exception as exc:
            # ⚠️ NOTHING on the rollback path may be best-effort. `_rm` is right for a temp
            # file, whose loss costs nothing, and WRONG here: a published table that cannot be
            # removed leaves the half-published pair this whole dance exists to prevent, and a
            # backup that cannot be restored leaves the destination missing with its only copy
            # under a generated name the cleanup below would then delete. Both are collected
            # and named in one error -- an intact file nobody can find is lost in practice
            # (real_bundle.py:373; codex checkpoint-2 rounds 3-4).
            #
            # ...but only where withdrawal is the ONLY way back. A destination that was backed
            # up is restored by its own `os.replace(bak, dest)` just below, which overwrites
            # whatever is there, so unlinking it first is redundant -- it merely opens a window
            # where the file is missing, and a failed unlink would be reported as a rollback
            # failure that did not happen (Gemini, PR #292). Only a destination created fresh by
            # this call has no backup to overwrite it, and there the unlink is load-bearing.
            backed_up = {dest for dest, _ in backups}
            stuck = []
            for dest in published:
                if dest in backed_up:
                    continue
                try:
                    os.unlink(dest)
                except OSError:
                    stuck.append(dest)
            unrestored = []
            for dest, bak in backups:
                try:
                    os.replace(bak, dest)
                except OSError:
                    unrestored.append((dest, bak))
            # NOTHING deletes a backup on this path, and nothing needs to (Copilot, PR #292 --
            # an earlier revision filtered `backups` here and claimed the filtering was what
            # preserved the survivors, which was false: this block always exits by raising, so
            # the success-path sweep below is unreachable from here). A restore that SUCCEEDED
            # consumed its own backup via the rename; one that failed is deliberately left on
            # disk and named in the error. If backup cleanup is ever moved into the `finally`,
            # `test_write_de_tables_reports_an_unrestorable_backup` is what will catch it.
            problems = []
            if stuck:
                problems.append("these were published and could not be removed: "
                                + ", ".join(repr(p) for p in stuck))
            if unrestored:
                # The two sub-cases leave the destination in DIFFERENT states, and a recovery
                # message that cannot tell them apart is worse than none: a backed-up
                # destination whose own publish failed is now missing, while one that published
                # before a later step failed still holds the newly written table (it is no
                # longer unlinked -- see above).
                pub = set(published)
                problems.append("the previous contents survive only as " + "; ".join(
                    f"{bak!r}, which belongs at {dest!r} ("
                    + ("that path currently holds the newly written table"
                       if dest in pub else "that path is now missing") + ")"
                    for dest, bak in unrestored))
            if problems:
                raise ValueError(
                    f"--write-degenes failed to publish into {outdir!r} and could not fully "
                    f"roll back -- {'; and '.join(problems)}. Resolve by hand before trusting "
                    "anything in that directory."
                ) from exc
            raise
        for _, bak in backups:
            _rm(bak)
    finally:
        for tmp, _ in staged:
            _rm(tmp)
    return real_tbl, pred_tbl


# Bumped when a change alters what an already-CACHED RESULT means for a run that folded the DE
# backend into its key (de_backend_used). v2 (2026-07-25): the de_<method>_rank key became
# content-hashed. Before that, a run could miss the result cache, be served a STALE rank via a
# colliding value-blind fingerprint, and then store that wrong output under its OWN correct
# config_hash -- so fixing the rank key alone still serves the poisoned result, because
# compute_metrics consults the result cache BEFORE calling _prepare_de_cached. Bumping this
# invalidates exactly those entries.
_DE_RESULT_SEMANTICS = 2

#: Bump when the MEANING of `pds_*` under `exclude_target_gene=True` changes. 3 = issue #343:
#: the default `exclusion_scope` is "panel" -- EVERY panel target gene leaves the ranked feature
#: space, where "row" dropped only the prediction row's own and left each reference
#: perturbation's knockdown visible off-diagonal. Every `pds_*` value under exclusion moves, so a
#: warm entry keyed on (inputs + config) at the same version would serve a pre-#343 score. 2 = issue #248:
#: the exclusion column is resolved through `target_gene_map` and a zero-resolve panel raises.
#: Before that, a guide-level panel silently excluded NOTHING and cached a number that is now
#: known wrong -- and the result-cache key is (inputs + config), which that run reproduces
#: exactly. Without this term a warm cache keeps serving the pre-#248 score, or keeps serving a
#: number where the new code would raise. Same device as `_DE_RESULT_SEMANTICS`, and scoped the
#: same way so nothing that could not have been affected loses its cache.
_PDS_EXCLUSION_SEMANTICS = 3

#: Bump when the MEANING of the DE/expression members that drop the perturbed gene's own
#: row changes. 2 = issue #348: `expr_mse_unbiased_capped`'s prediction-side sampling correction
#: is additionally bounded by the submission's own across-perturbation spread, computed with the
#: SAME per-row exclusion (`delta._across_pert_budget`) -- so #172's dropped coordinate now governs
#: the correction as well as the distance, and every capped value where the bound binds moves.
#: 1 = issue #172 (ruled 2026-08-17): `de_*_sig_jaccard`, `de_*_lfc_nmae` and both
#: legs of `expr_mse_unbiased_capped_norm` now exclude it, where before they summed it.
#:
#: ⚠️ At 2 this OVER-invalidates more than at 1: three of the five funcs in
#: `_ONTARGET_EXCLUDING_FUNCS` (`de_sig_jaccard`, `de_lfc_nmae`, `distance_unbiased`) are
#: untouched by #348, and `mse_unbiased` -- the uncapped audit column -- is untouched too even
#: though it shares `_numerator`, because #348 is part of the CAP
#: (`test_the_uncapped_audit_column_is_untouched`). A run requesting only those recomputes once
#: for nothing. That is the same trade this counter already documents below, taken rather than
#: minting a fifth counter and a fifth gate for one metric: the alternative was measured against
#: the precedent at `_PDS_EXCLUSION_SEMANTICS`, which went 2 -> 3 for #343, a different semantic
#: change to the same FAMILY. If a future change needs to distinguish them, split then.
#:
#: ⚠️ This is NOT optional bookkeeping, and the version string does not cover it. The result
#: cache keys on (inputs + config) and `cell_eval2_version` is deliberately absent from that key
#: -- neither `cache.py` nor `EvalConfig` carries it -- so a pre-#172 run at the SAME version
#: reproduces the key exactly and its cached value, now known wrong, would be served in
#: preference to recomputing. Exactly the hole `_PDS_EXCLUSION_SEMANTICS` was minted for at #248,
#: and scoped the same way so a run that could not have been affected keeps its cache.
_ONTARGET_EXCLUSION_SEMANTICS = 2

#: The five catalog funcs whose VALUE changed meaning in #172, keyed by function identity so one
#: entry covers both backend families (`de_wilcoxon_*` / `de_deseq2_*` share a func) and any
#: future variant. `mse_unbiased` is here even though it is never scored: it shares
#: `metrics.delta._numerator` with the capped leg, so its column moved too.
_ONTARGET_EXCLUDING_FUNCS = (de_sig_jaccard, de_lfc_nmae, mse_unbiased, mse_unbiased_capped,
                             distance_unbiased)


#: Bump when the MEANING of a grouped group SUM changes -- the `bulk_lognorm` pseudobulk every
#: expression/PDS member reads, and the deseq2 backend's per-replicate pseudobulk. 1 = issue #271
#: (2026-08-18): `prep._grouped_sums` widens a floating dtype COARSER than float64 before reducing,
#: where it used to reduce in the input dtype and cast only the result.
#:
#: ⚠️ The same hole its three siblings above were minted for, and neither the version string nor the
#: competition digest covers it. The result cache keys on (inputs + config) with `cell_eval2_version`
#: deliberately absent, and `competition.competition_digest()` does NOT move across #271 either --
#: the competition RULE is unchanged; what moved is the pseudobulk the members read. So a pre-#271
#: run at the SAME version would reproduce every key exactly WITHOUT this term, and its cached bulk,
#: moments, deseq2 DE table and final score would each be served in preference to recomputing.
#:
#: MEASURED on the three official contexts' stored fp32 baseline arms: group sums move by up to
#: 0.265 counts and bulks by up to 5.7e-06. Integer counts below float32's 2**24 do not move at all
#: -- but a cache key is computed before any value is read, so the scoping below is by PATH (which
#: comparator, which DE backend), never by what the data turns out to hold.
_GROUPED_SUM_REDUCTION_SEMANTICS = 1


def _grouped_sum_reduction_used(*, comparator, de_backend) -> bool:
    """Whether a run's numbers could depend on #271's group-sum reduction semantics.

    `prep._grouped_sums` has exactly two families of caller: `pseudobulk_bulk_lognorm` /
    `..._with_moments` (reached only when the resolved comparator is `bulk_lognorm`), and
    `deseq2_de._pseudobulk`. A `lognorm` run goes through `_grouped_means` instead and keeps its
    warm cache; so does any other DE backend, whose tables are computed from cells rather than
    from a pseudobulk.

    Scoped like `_ontarget_exclusion_used` and its siblings, and OVER-invalidating in the same
    direction: an integer-count submission below float32's `2**24` reduces identically either
    way, but that cannot be decided here -- the key is computed before the bulks are built, so
    the values are not in hand. A one-time recompute for runs that never moved is the right side
    to err on; the alternative is serving a bulk this build would not produce.
    """
    return comparator == "bulk_lognorm" or de_backend == "deseq2"


#: The purity floor `de_direction_reach`'s RAW form thresholds at. Keyed by VALUE, not by a
#: hand-bumped counter like the semantics terms above and below, because here the semantics ARE a
#: single number: a future retune moves this key with no one having to remember to bump
#: anything. (The other three cover code changes that no config value identifies.)
#:
#: ⚠️ The version string does not cover this, for the reason `_ONTARGET_EXCLUSION_SEMANTICS`
#: already gives -- and for a second one it does not: `cell_eval2.__version__` reads the
#: INSTALLED distribution metadata, which an editable install froze at install time, so a
#: source tree can run 0.9 code while reporting an older version. A term computed from the
#: constant itself cannot go stale that way.
_REACH_FLOOR_FUNCS = (de_direction_reach,)


def _func_is_one_of(func, targets) -> bool:
    """Whether a catalog entry's ``func`` is one of ``targets``, however it is wrapped.

    The catalog binds arguments with ``functools.partial``, so ``func.func`` may be the real
    callable -- and a bare ``getattr(f, "func", None) is target`` test is silently False for a
    plain function or a doubly-wrapped partial. Every caller here decides either DISPATCH or
    RESULT-CACHE INVALIDATION from the answer, and a False where True was meant is silent in
    both: a variant would dispatch to the wrong kernel, or keep a cache key that says its
    meaning did not change. Unwrapping the whole chain makes that robust to how a future entry
    is built.
    """
    seen = 0
    while func is not None and seen < 8:          # bounded: a cycle here would hang the run
        if any(func is t for t in targets):
            return True
        func = getattr(func, "func", None)
        seen += 1
    return False


def _is_discrimination(func) -> bool:
    """Whether a catalog entry's ``func`` is the discrimination metric, however it is wrapped.

    Two places MUST agree on this: the GPU dispatch below, and
    ``_discrimination_exclusion_used``, which decides result-cache invalidation. If they ever
    disagreed, a variant would dispatch to the discrimination kernel while keeping a cache key
    that says it did not (Copilot, PR #250).
    """
    return _func_is_one_of(func, (discrimination_score,))


def _ontarget_exclusion_used(names) -> bool:
    """Whether this run's result could depend on #172's target-gene exclusion semantics.

    True when any requested metric is one of the five funcs whose value changed meaning. Keyed
    on FUNCTION IDENTITY rather than a name prefix so both backend families
    (`de_wilcoxon_*` / `de_deseq2_*`, which share one func) and any future variant are covered
    without a second edit here.

    ⚠️ The derived member `expr_mse_unbiased_capped_norm` has no func of its own, and does not
    need one: `resolve_metrics` appends a derived entry's two components to the resolved list,
    and every profile claiming it must carry them
    (`test_every_shipped_derived_metric_has_its_components_in_every_profile_it_claims`), so
    `names` already holds both legs by the time this is asked.

    Like `_discrimination_exclusion_used` this OVER-invalidates: a panel whose targets resolve
    to nothing scored identically before and after (it now raises instead, which is the point),
    and one whose targets are not measured genes is unaffected either way. Both still take a new
    key and recompute once. Deciding otherwise needs the labels and the gene index, neither of
    which exists at digest time -- the key is computed before the bulks are built.
    """
    return any(_func_is_one_of(CATALOG[n].func, _ONTARGET_EXCLUDING_FUNCS)
               for n in names if n in CATALOG)


def _reach_floor_used(names) -> bool:
    """Whether this run's result could depend on `direction.REACH_PURITY_FLOOR`.

    Keyed on FUNCTION IDENTITY like `_ontarget_exclusion_used`, so both backend families share
    one entry here. Deliberately OVER-invalidating in one direction: all four `direction_reach*`
    variants bind the same func, and only the two RAW ones read the floor, so a run selecting
    only a corrected variant takes a new key it did not need. Narrowing it would mean reading
    the partial's bound `corrected` kwarg, which is exactly the kind of introspection
    `_func_is_one_of` exists to avoid depending on.
    """
    return any(_func_is_one_of(CATALOG[n].func, _REACH_FLOOR_FUNCS)
               for n in names if n in CATALOG)


def _discrimination_exclusion_used(cfg, names) -> bool:
    """Whether this run's result could depend on the #248 exclusion semantics.

    True when a `pds_*` metric is requested AND `exclude_target_gene` is on. Derived from the
    CATALOG rather than a name prefix, so a future discrimination variant is covered without a
    second edit here.

    ⚠️ **Deliberately CONSERVATIVE, and it over-invalidates.** A gene-level panel whose raw
    labels already matched the gene index scores identically before and after #248, yet still
    takes a new key and recomputes once. Narrowing it would mean comparing old raw-label
    resolution against new map-first resolution -- which needs the labels and the gene index,
    neither of which exists at digest time (the key is computed before the bulks are built).
    A one-time recompute for runs that were already correct is the right side to err on: the
    alternative is serving a score that silently excluded nothing.
    """
    if not cfg.discrimination.exclude_target_gene:
        return False
    return any(_is_discrimination(CATALOG[n].func) for n in names)


def _result_config_digest(
    cfg, *, de_backend_used: bool, comparator: str, pds_exclusion_used: bool = False,
    ontarget_exclusion_used: bool = False, reach_floor_used: bool = False,
) -> str:
    """config_hash for the result cache key. The DEVICE always enters the key (via _cache_device:
    "auto" is resolved to cuda/cpu, an explicit device is used verbatim) -- fp32 GPU vs fp64 CPU
    pseudobulk affects every pseudobulk-based metric. The DE BACKEND is resolved into the key ONLY
    when de_backend_used holds -- a DE metric is requested AND at least one side's table is
    computed rather than supplied. (A DE-TABLE CACHE HIT still counts: no engine process runs, but
    the backend still determined the cached table, so it must stay in the key.) Resolving it
    otherwise would (a) make a run that needs no DE
    engine (e.g. metrics=["mae"], or DE metrics with BOTH tables supplied) depend on a DE backend
    being installed -- raising in a minimal install -- and (b) needlessly split that cache across
    gpudge/pdex nodes whose results are identical (Copilot review, PR #114). When the engine does
    not enter the key (no DE metrics, or both tables supplied) the raw cfg value (e.g. "auto")
    stays in the key, which is machine-independent and so collision-free. to_dict() returns fresh nested dicts -- safe to mutate."""
    d = cfg.to_dict()
    d["comparator"] = comparator
    if cfg.target_gene_map is None:
        # Adding ANY field to EvalConfig invalidates every warm result cache, because
        # to_dict() -> config_hash picks it up. Drop it when inert so existing caches
        # survive and only runs that actually supply a map get a new key. Exactly the
        # replicate_col fix below (Copilot PR #120).
        d.pop("target_gene_map", None)
    d["device"] = _cache_device(cfg)
    resolved_de = _cache_backend(cfg) if de_backend_used else None
    if resolved_de == "deseq2":
        # deseq2 is the only backend replicate_col / the CPU-vs-GPU fit affect. Keep replicate_col
        # (to_dict carries it) and record the CPU(numpy)/GPU(jax) fit predicate -- mirrors
        # compute_de's use_gpu = raw device.startswith("cuda") (NOT the resolved _cache_device, which
        # maps "auto"->cuda), so a CPU result can't false-hit a GPU one. resolved_de reused (computed
        # once) for the backend key.
        d["de"]["backend"] = resolved_de
        d["de"]["deseq2_gpu_fit"] = str(cfg.device).startswith("cuda")
    else:
        # non-deseq2 (or DE engine not run): replicate_col is inert -> drop it so existing configs,
        # and non-deseq2 configs that merely set it, keep their prior result-cache key (to_dict now
        # always carries the field, which would otherwise invalidate every warm cache; Copilot PR #120).
        d["de"].pop("replicate_col", None)
        if de_backend_used:
            d["de"]["backend"] = resolved_de
    if de_backend_used:
        # Scoped by de_backend_used, i.e. DE metrics requested with at least one side COMPUTED
        # rather than supplied -- not "the engine executed", since a DE-table cache hit skips
        # execution but still keys on the backend. A metrics=["mae"] run, or one with BOTH DE
        # tables supplied, could never have been poisoned by the rank cache: it keeps its cache.
        d["de_rank_cache_semantics"] = _DE_RESULT_SEMANTICS
    if not pds_exclusion_used:
        # `exclusion_scope` (#343) is a DiscriminationParams field, so to_dict() carries it into
        # EVERY config_hash -- including runs with no pds_* metric, or with exclusion off, which
        # it cannot possibly move. Dropping it there is the same inert-field rule target_gene_map
        # and replicate_col already follow, and it is what keeps those warm caches alive. Note the
        # predicate is `pds_exclusion_used`, not `de_backend_used`: the two are independent, and
        # keying the pop on the DE term would drop the scope from exactly the pds runs that have
        # a DE metric alongside -- i.e. every competition run.
        d["discrimination"].pop("exclusion_scope", None)
    if pds_exclusion_used:
        # Scoped like the DE term above, though more coarsely: a run with no pds_* metric, or
        # with exclude_target_gene=False, could never have been poisoned by the #248 hole and
        # keeps its warm cache. Within exclusion-enabled pds runs it does NOT distinguish the
        # ones that were already correct -- see _discrimination_exclusion_used for why that
        # cannot be decided here. `discrimination.exclusion_scope` (#343) is left IN `d` on this
        # branch, so "row" and "panel" key apart; this counter retires pre-#343 entries of both.
        d["pds_exclusion_semantics"] = _PDS_EXCLUSION_SEMANTICS
    if ontarget_exclusion_used:
        # Scoped like the two terms above: a run requesting none of the five #172 funcs keeps
        # its warm cache. See _ontarget_exclusion_used for why this cannot be narrowed further.
        d["ontarget_exclusion_semantics"] = _ONTARGET_EXCLUSION_SEMANTICS
    if reach_floor_used:
        # Scoped like the three terms above: a run requesting no `direction_reach*` metric keeps
        # its warm cache. The VALUE, not a counter -- see `_REACH_FLOOR_FUNCS`.
        from .metrics.direction import REACH_PURITY_FLOOR
        d["reach_purity_floor"] = REACH_PURITY_FLOOR
    if _grouped_sum_reduction_used(comparator=comparator, de_backend=resolved_de):
        # #271. Scoped like the four terms above: a `lognorm` run on a non-deseq2 backend never
        # reaches `prep._grouped_sums` and keeps its warm cache. `d["comparator"]` already splits
        # the two comparators, but it says which one was ASKED for -- not what a group sum MEANS,
        # which is what changed. `resolved_de`, not `cfg.de.backend`: it is None when no DE engine
        # enters the key at all, and deseq2 is never auto-selected, so the two agree whenever a
        # deseq2 table could have been built.
        d["grouped_sum_reduction_semantics"] = _GROUPED_SUM_REDUCTION_SEMANTICS
    return config_hash(d)


def compute_metrics(
    pred: ad.AnnData | str | os.PathLike,
    real: ad.AnnData | str | os.PathLike,
    *,
    config: EvalConfig | None = None,
    de_pred=None,
    de_real=None,
    write_de=False,
    **overrides,
) -> pl.DataFrame:
    """Run the selected metrics and return a tidy-long DataFrame.

    Columns: (perturbation, metric, value). Explicit kwargs override `config`.

    Each metric receives only the kwargs its signature declares (via
    inspect.signature filtering), so heterogeneous metrics (mae vs PDS) share one
    dispatch. NOTE: `distance` is intentionally NOT in the dispatched kwargs — it
    is bound per-variant by functools.partial in the catalog; passing it here
    would silently override the partial.
    """
    cfg = _resolve_config(config, overrides)
    # Checked before anything is loaded, so a long run cannot do all its work and only then
    # find it has nowhere to put the tables. outdir=None means "the API writes nothing", so
    # defaulting it here would silently drop files into the CWD of whoever imported us --
    # exactly the kind of silent fallback this codebase keeps having to remove.
    if write_de and cfg.outdir is None:
        raise ValueError(
            "write_de=True requires an output directory, but config.outdir is None; "
            "set EvalConfig(outdir=...) (the `cell-eval2 run` CLI always sets one)"
        )
    real_store = CacheStore(cfg.cache_real) if cfg.cache_real else None
    pred_store = CacheStore(cfg.cache_pred) if cfg.cache_pred else None

    pred_ad = load_anndata(pred, backed=isinstance(pred, (str, os.PathLike)))
    real_ad = None  # bound before the try so the finally can't NameError if the real load raises
    try:
        real_ad = load_anndata(real, backed=isinstance(real, (str, os.PathLike)))
        try:
            return _run_metrics(pred_ad, real_ad, cfg=cfg, de_pred=de_pred, de_real=de_real,
                                real_store=real_store, pred_store=pred_store, write_de=write_de)
        finally:
            # #277 item 3, HERE and not at the end of _run_metrics: that function returns EARLY on
            # a result-cache hit (run.py:1388), so logging inside it missed exactly the runs a
            # memory report is cheapest on -- and made "every in-memory scoring run" false
            # (codex-review). In a finally, so a run that OOMs its way to an exception still
            # reports the high-water mark it reached, which is the number #277 is about.
            _log_peak_host_rss(pred_ad, real_ad)
    finally:  # close only the handles we opened for path inputs, even on error (caller objects untouched)
        _close_backed(pred_ad, pred)
        if real_ad is not None:  # None only if the real load itself raised -- pred still gets closed
            _close_backed(real_ad, real)


def _run_metrics(pred_ad, real_ad, *, cfg, de_pred, de_real, real_store, pred_store,
                 write_de=False):
    """Body of compute_metrics, split out so the backed file handles compute_metrics opened
    are always closed (its try/finally) even if computation raises."""
    # Captured before de_pred/de_real are rebound (load below, compute later): a non-None table
    # here is SUPPLIED by the caller, so it must be fingerprinted strictly (F9.1, _de_supplied_strict).
    de_real_supplied = de_real is not None
    de_pred_supplied = de_pred is not None
    validate_pair(pred_ad, real_ad, pert_col=cfg.pert_col, control=cfg.control)
    # Resolve each side's effective type exactly once. The comparator policy consumes BOTH
    # values; using cfg.input_type (or accidentally reading pred twice) breaks asymmetric runs.
    effective_types = {
        "pred": _effective_input_type(pred_ad, cfg, side="pred"),
        "real": _effective_input_type(real_ad, cfg, side="real"),
    }
    for side, adata in (("pred", pred_ad), ("real", real_ad)):  # in-memory: validate up front (cheap, backed-safe skip)
        if not getattr(adata, "isbacked", False):
            eff = effective_types[side]
            if cfg.validate_input and cfg.version != "v1":
                _validate_input_once(adata, eff, allow_fractional=_val_allow_fractional(cfg, side=side))
            # Scale-limit: REAL up front (cheap: control-only shard in the warm-cache scenario). PRED
            # is deferred to just after its pseudobulk (below) so the counts+GPU path reuses the
            # accumulator's per-cell max instead of a redundant _row_totals pass; negativity/type
            # fail-fast stays up front for both sides above.
            if cfg.validate_input and side == "real":
                _check_scale_limit_once(adata, eff, cfg.max_counts_per_cell)

    _warn_mixed_library_scale(cfg, effective_types)

    # Resolve before cache keys are built, so the resolved number enters them and flows to
    # DE, both sides' pseudobulk, and _pred_de_input's scale conversion.
    cfg = _resolve_target_sum_from_control(cfg, real_ad)

    names, missing = resolve_metrics(cfg.metrics, version=cfg.version)
    if missing:
        logger.warning("Skipping not-yet-implemented metrics: %s", missing)
    comparator = _norm.resolve_comparator(
        version=cfg.version,
        pred_input_type=effective_types["pred"],
        real_input_type=effective_types["real"],
    )

    # real_fp: needed for the real pseudobulk cache (real_store) or the result key (pred_store).
    # pred_fp: needed only when pred_store is set (pred pseudobulk cache or the result key) — so
    # a cache_real-only run never materializes/hashes pred (matters under cache_strict).
    real_fp = fingerprint_adata(real_ad, pert_col=cfg.pert_col, strict=cfg.cache_strict) \
        if (real_store or pred_store) else None
    pred_fp = fingerprint_adata(pred_ad, pert_col=cfg.pert_col, strict=cfg.cache_strict) \
        if pred_store else None

    result_fp = None
    if pred_store is not None:
        de_fps = []
        has_de = any(CATALOG[n].kind == "de" for n in names)
        if has_de:
            from .de import load_de_table
            # Fingerprint each SUPPLIED DE side independently so its content enters the
            # result key even when only one side is supplied (the other is computed):
            # otherwise two different supplied tables sharing the same adata fingerprints +
            # config collide on a stale cached result. A computed side is already captured
            # by real_fp/pred_fp + config_digest; a stable placeholder keeps de_fps positions
            # consistent.
            if de_real is not None:
                de_real = load_de_table(de_real)  # rebind so _prepare_de_cached reuses it (passthrough)
            if de_pred is not None:
                de_pred = load_de_table(de_pred)
            de_fps = [
                fingerprint_de_table(de_real, strict=_de_supplied_strict(de_real_supplied, cfg))
                if de_real is not None else "no-de-real",
                fingerprint_de_table(de_pred, strict=_de_supplied_strict(de_pred_supplied, cfg))
                if de_pred is not None else "no-de-pred",
            ]
        # de_backend_used: the backend affects results (and so must enter the key) only when a DE
        # metric is requested AND at least one side is computed, not supplied. This is a
        # REQUEST-SHAPE predicate, not "the engine executed" -- a DE-table cache hit skips
        # execution and is still True, because the backend produced the table being served.
        de_backend_used = has_de and (de_real is None or de_pred is None)
        result_fp = result_fingerprint(
            real_fp=real_fp, pred_fp=pred_fp, de_fps=de_fps,
            config_digest=_result_config_digest(
                cfg, de_backend_used=de_backend_used,
                comparator=comparator,
                pds_exclusion_used=_discrimination_exclusion_used(cfg, names),
                ontarget_exclusion_used=_ontarget_exclusion_used(names),
                reach_floor_used=_reach_floor_used(names),
            ),
            metric_names=names)
        cached = pred_store.get("results", fingerprint=result_fp, params={}, kind="parquet")
        if cached is not MISS:
            # --write-degenes asks for an extra ARTIFACT, not for different numbers, so it must
            # not cost the results hit: materialize + write the DE tables here and still serve
            # the cached results. A supplied side is used as given, and a computed one goes
            # through its own DE cache -- which HITS if that side is cached (the run behind
            # this result populated it) and recomputes otherwise, since --cache-pred alone is
            # enough to cache results and leaves the real side uncached. Bypassing the
            # short-circuit instead would re-run every metric, including the ones the results
            # cache exists to skip. (The rebinding is dead here -- we return `cached` -- but
            # keeps both call sites' contract identical; see _write_de_tables.)
            if write_de and has_de:
                de_real, de_pred = _materialize_de_sides(
                    pred_ad, real_ad, cfg=cfg, de_real=de_real, de_pred=de_pred,
                    real_fp=real_fp, real_store=real_store, pred_store=pred_store,
                )
                de_real, de_pred = _write_de_tables(de_real, de_pred, cfg=cfg)
            if cfg.outdir:
                os.makedirs(cfg.outdir, exist_ok=True)
                cfg.to_yaml(os.path.join(cfg.outdir, "run_params.yaml"))
            _warn_fractional_allowance_unclassified(cfg, pred_ad)
            return cached

    # Pred pseudobulk EARLY (before the DE section): on the counts + GPU-accumulator path
    # inmem_pseudobulk stashes the per-cell max on pred_ad, which the deferred pred scale-limit
    # gate just below then reuses -- eliminating a redundant full _row_totals pass. Placed AFTER
    # the result-cache early-return so a cache hit never computes it. When norms is empty (DE-only)
    # _side_bulks returns {} and sets no stash, so the gate falls back to its own full pass.
    names_norms = _needed_normalizations(names, comparator=comparator)
    moment_norms = _moment_normalizations(names, comparator=comparator)
    pred_side = _side_bulks(pred_ad, fp=pred_fp, store=pred_store, norms=names_norms, cfg=cfg,
                            side="pred", moment_norms=moment_norms,
                            effective_input_type=effective_types["pred"])
    pred_bulks, pred_moments = pred_side if moment_norms else (pred_side, None)
    if not getattr(pred_ad, "isbacked", False) and cfg.validate_input:
        _check_scale_limit_once(pred_ad, effective_types["pred"], cfg.max_counts_per_cell)

    prepared_de = None
    if any(CATALOG[n].kind == "de" for n in names):
        de_real, de_pred = _materialize_de_sides(
            pred_ad, real_ad, cfg=cfg, de_real=de_real, de_pred=de_pred,
            real_fp=real_fp, real_store=real_store, pred_store=pred_store,
        )
        # Written before _prepare_de_cached, which only consumes these tables -- and rebound
        # to the frames _write_de_tables loaded. Keeping the original arguments would leave
        # _prepare_de_cached re-reading a supplied PATH that these writes may have just
        # overwritten, scoring the run on the wrong table with correct files on disk.
        if write_de:
            de_real, de_pred = _write_de_tables(de_real, de_pred, cfg=cfg)
        prepared_de = _prepare_de_cached(
            de_pred, de_real, cfg=cfg, real_store=real_store, pred_store=pred_store,
            de_real_supplied=de_real_supplied, de_pred_supplied=de_pred_supplied,
        )
        ad_perts = set(map(str, pred_ad.obs[cfg.pert_col].unique())) - {cfg.control}
        de_perts = set(prepared_de.perturbations)
        if ad_perts != de_perts:
            only_ad = sorted(ad_perts - de_perts)
            only_de = sorted(de_perts - ad_perts)
            raise ValueError(
                "perturbation sets differ between anndata and DE tables "
                f"(anndata-only={only_ad}, DE-only={only_de}); DE targets must exactly "
                "match the anndata perturbations (excluding the control)"
            )

    genes = np.asarray(pred_ad.var.index.values, dtype=str)
    norms = names_norms  # computed above (before the DE section) for the early pred pseudobulk
    if not norms and real_store is None and pred_store is None:
        # DE-only run with NO cache: no normalization materializes a side, but a no-cache
        # run must validate inputs exactly as before — validate path/backed sides one at a
        # time (in-memory sides were validated up front). In cache mode this is skipped:
        # the anndata X is unused by DE metrics (only metadata), so materializing it would
        # defeat the cache's RAM/speed benefit (cache-hit semantics apply).
        for side, adata in (("real", real_ad), ("pred", pred_ad)):
            if getattr(adata, "isbacked", False):
                checked = _materialize(adata)  # to_memory() on the open handle; no re-read
                eff = effective_types[side]
                if cfg.validate_input and cfg.version != "v1":
                    _validate_input_once(checked, eff, allow_fractional=_val_allow_fractional(cfg, side=side))
                if cfg.validate_input:
                    _check_scale_limit_once(checked, eff, cfg.max_counts_per_cell)
                del checked

    # real_bulks last (pred_bulks was computed before the DE section so its per-cell max feeds the
    # pred scale-limit gate). Sides are still materialized one at a time — each _side_bulks/
    # _compute_de_side materializes transiently and releases, so peak host RAM is a single side's
    # working set (pred_bulks ~= de_pred are the two heavy phases, sequential), unchanged by the
    # reorder. Pass the already-open AnnData objects (backed for path inputs) so a miss materializes
    # via to_memory() on the open handle instead of re-reading the file (avoids a redundant
    # open + a TOCTOU window vs the earlier backed read used for metadata/fingerprint).
    real_side = _side_bulks(real_ad, fp=real_fp, store=real_store, norms=norms, cfg=cfg,
                            side="real", moment_norms=moment_norms,
                            effective_input_type=effective_types["real"])
    real_bulks, real_moments = real_side if moment_norms else (real_side, None)

    rows: list[dict] = []
    rows.extend(dispatch_de_metrics(names, prepared_de, cfg))
    rows.extend(dispatch_anndata_metrics(names, pred_bulks, real_bulks, genes, cfg,
                                         comparator=comparator,
                                         pred_moments=pred_moments,
                                         real_moments=real_moments,
                                         driver="compute_metrics (in-memory)"))

    df = pl.DataFrame(rows, schema=_TIDY_SCHEMA) if rows else pl.DataFrame(schema=_TIDY_SCHEMA)
    if pred_store is not None and result_fp is not None:
        pred_store.put("results", df, fingerprint=result_fp, params={}, kind="parquet")
    if cfg.outdir:
        os.makedirs(cfg.outdir, exist_ok=True)
        cfg.to_yaml(os.path.join(cfg.outdir, "run_params.yaml"))
    return df


def metric_agg(output_name: str) -> str:
    """The aggregate statistic for a metric's OUTPUT name ('mean' or 'median').

    The tidy frame carries output names (v1 aliases under v1), so resolve through
    `_NAME_TO_CANONICAL`. An unknown name falls back to 'mean' -- an observed-but-
    unexpected metric is a caller bug, and silently changing its statistic would be a
    worse failure than reporting the default.
    """
    spec = CATALOG.get(_NAME_TO_CANONICAL.get(output_name, output_name))
    return spec.agg if spec is not None else "mean"


def _derived_value(
    df: pl.DataFrame, spec, *, require_components: bool = False
) -> float | None:
    """`Σ numerator / Σ denominator` over perturbations where BOTH sides are finite.

    The pairing is load-bearing. Summing the two columns independently would include a
    perturbation's denominator while dropping its NaN numerator, which biases the ratio
    toward zero -- silently, and worst on exactly the panels with the most NaN.

    ⚠️ The two components must cover the SAME perturbations, and that is CHECKED rather than
    assumed. The numerators are emitted per predicted perturbation and
    `expr_distance_unbiased` per real one, so on a path that validates the two sets they agree
    -- and on one that does not, they disagree loudly. That is the point: shard streaming
    validates only the gene axis (`scale.py:116-123`), so a submission omitting a real
    perturbation shrinks the numerator alone and lands here. Inner-joining on the intersection
    instead would let it choose its own denominator by covering only the perturbations with
    the largest real effects -- a free lever of exactly the kind #247 closed on the numerator.

    Returns None when either component is absent from the frame and the caller does not know
    whether this metric was selected. When ``require_components`` is true, absence raises:
    a requested derived metric must not disappear from the aggregate silently.
    """
    num = df.filter(pl.col("metric") == spec.derived.numerator).select(
        "perturbation", n="value")
    den = df.filter(pl.col("metric") == spec.derived.denominator).select(
        "perturbation", d="value")
    if num.height == 0:
        if require_components:
            raise ValueError(
                f"{spec.name}: requested derived metric cannot be computed because component "
                f"{spec.derived.numerator} is empty; the other component "
                f"{spec.derived.denominator} has {den.height} row(s). A requested metric "
                "must not disappear from the aggregate."
            )
        return None
    if den.height == 0:
        if require_components:
            raise ValueError(
                f"{spec.name}: requested derived metric cannot be computed because component "
                f"{spec.derived.denominator} is empty; the other component "
                f"{spec.derived.numerator} has {num.height} row(s). A requested metric "
                "must not disappear from the aggregate."
            )
        return None
    n_labels = set(num["perturbation"].to_list())
    d_labels = set(den["perturbation"].to_list())
    if n_labels != d_labels:
        only_n, only_d = sorted(n_labels - d_labels), sorted(d_labels - n_labels)
        raise ValueError(
            f"{spec.name}: its components cover different perturbations -- "
            f"{len(only_n)} only in {spec.derived.numerator} ({only_n[:5]}), "
            f"{len(only_d)} only in {spec.derived.denominator} ({only_d[:5]}). Both are "
            "emitted over the in-scope cohort, so a mismatch means something restricted one "
            "side and not the other. Joining on the intersection instead would let a "
            "submission choose its own denominator."
        )
    # `is_finite` on both, NOT `is_not_nan`: the latter lets +/-inf through, and an infinite
    # denominator sum passes `d_sum > 0` to yield 0.0 or NaN silently (codex, checkpoint 1).
    pair = num.join(den, on="perturbation", how="inner").filter(
        pl.col("n").is_finite() & pl.col("d").is_finite()
    )
    if pair.height == 0:
        # NOT `return None`: that is reserved for components ABSENT from the frame, where a
        # profile simply does not carry them. Here both columns are present and every pair was
        # non-finite, so the metric CAN be asked for and cannot be answered -- returning None
        # would drop a scored metric out of the aggregate silently, which is the #250 failure
        # mode (a number quietly disappearing rather than stopping the run).
        raise ValueError(
            f"{spec.name}: both components are present over "
            f"{num.height} and {den.height} perturbation(s), but no perturbation has a finite "
            "value on BOTH sides, so the ratio of sums has nothing to sum. A derived metric "
            "that cannot be computed must not disappear from the aggregate."
        )
    n_sum, d_sum = float(pair["n"].sum()), float(pair["d"].sum())
    if not math.isfinite(n_sum) or not math.isfinite(d_sum):
        raise ValueError(
            f"{spec.name}: summing {pair.height} finite per-perturbation value(s) overflowed "
            f"to numerator={n_sum!r}, denominator={d_sum!r}. The inputs were finite, so this "
            "is an accumulation failure, not degenerate data."
        )
    if not d_sum > 0.0:
        raise ValueError(
            f"{spec.name}: the sum of {spec.derived.denominator} over "
            f"{pair.height} perturbation(s) is {d_sum!r}, which is not positive, so the "
            "ratio of sums is undefined or sign-flipped. That means this reference panel "
            "carries no measurable aggregate effect at this cell depth -- a property of the "
            "REFERENCE, not of the submission. Individual negative values are normal and "
            "expected; only the SUM going non-positive is a failure. Fix or replace the "
            "reference panel."
        )
    value = n_sum / d_sum
    # Both sums are finite and the denominator is positive, and the quotient can STILL
    # overflow (a huge numerator over a tiny denominator). An infinite metric value would
    # travel into scoring as a number.
    if not math.isfinite(value):
        raise ValueError(
            f"{spec.name}: numerator {n_sum!r} over denominator {d_sum!r} is {value!r}. Both "
            "sums are finite, so this is an overflow in the division itself, not degenerate "
            "data."
        )
    return value


def _reject_derived_rows(df: pl.DataFrame) -> None:
    """A derived metric must never appear in the tidy frame.

    It has no per-perturbation value, so a row bearing its name came from somewhere that does
    not know that. Left alone it would be group-by'd into a row AND appended as a derived row,
    returning a frame with two rows for one metric -- which a downstream lookup-by-name
    resolves arbitrarily (codex, checkpoint 1).
    """
    if not df.height:
        return
    # Canonicalize the OBSERVED names, not the catalog keys. The keys are already canonical,
    # so `_NAME_TO_CANONICAL.get(n, n)` returns `n` and the alias arm of this test was a
    # tautology -- a row carrying a derived metric under a v1 name or alias walked straight
    # through the guard. Nothing exercises that today (the one derived entry has no alias),
    # which is exactly what a vacuous guard looks like until it matters; #219 shipped one.
    # Found independently by both PR bots, round 3.
    observed = {_NAME_TO_CANONICAL.get(m, m) for m in df["metric"].unique()}
    bad = sorted(n for n, s in CATALOG.items() if s.derived is not None and n in observed)
    if bad:
        raise ValueError(
            f"derived metric(s) {bad} appear as per-perturbation rows in the tidy frame. A "
            "derived metric's value exists only in the aggregate -- its per-perturbation "
            "ratio is deliberately not emitted. Whatever produced these rows is computing it "
            "the wrong way."
        )


def aggregate_metrics(
    df: pl.DataFrame, *, metrics: Sequence[str] | None = None
) -> pl.DataFrame:
    """Native aggregate: one NaN-skipping value per metric over perturbations -- the mean
    or the median, per ``MetricSpec.agg`` (see below).

    The summary line said "mean" until #230: the body already described the per-metric
    statistic, so the two halves of this docstring disagreed. As of #231 "mean" would in
    fact be accurate for the SHIPPED catalog -- but this function implements the published
    generic format, not the current catalog, so it keeps describing the field it reads.

    ``drop_nans()`` before the statistic so one degenerate perturbation (e.g. a
    single-class AUC pert emitting NaN, metrics/de.py:296,301) cannot null out
    the whole metric's aggregate. Matches the reference's NaN-skipping
    ``.describe()`` semantics and the compat path. Per-pert NaN values are left
    untouched in the tidy frame; only the aggregate skips them. An all-NaN
    metric aggregates to NaN (not null) via ``fill_null`` so downstream
    arithmetic (e.g. ``compat.score_agg_metrics``) is unaffected, matching the
    reference ``.describe()`` (an all-NaN column's mean is NaN).

    The statistic is per-metric (``MetricSpec.agg``). Since #231 EVERY shipped entry
    declares ``mean``, so the median branch below is unreachable through the catalog as
    published -- it is kept because the statistic is meant to stay a catalog edit rather
    than a source edit, and it stays exercised by
    ``test_metric_aggregation.py::test_median_agg_is_honoured_by_both_aggregators``, which
    injects ``agg="median"`` onto a metric and asserts THIS function returns the median.
    (``test_the_median_row_is_unconditional_not_a_second_agg_lookup`` covers the wide
    aggregator's separate branch, not this one.) The output column keeps the name ``mean``
    whatever the statistic -- renaming it would break compat, score.py and every published
    artifact -- and the ``agg`` column records which statistic each row actually holds.
    ``aggregate_metrics_wide`` additionally publishes a ``median`` row for every metric, so
    the statistic a metric does not score on is still readable.

    When ``metrics`` is supplied, it is the resolved selection behind ``df``. Derived rows
    are emitted only when selected, and a selected derived metric whose components are absent
    raises instead of disappearing. Accepted aliases and v1 names are compared by canonical
    identity. ``metrics=None`` preserves the legacy request-unknown behaviour: every buildable
    derived metric is injected and an absent component simply omits it.
    """
    _reject_derived_rows(df)
    selected = (None if metrics is None else
                {_NAME_TO_CANONICAL.get(m, m) for m in metrics})
    both = df.group_by("metric").agg(
        _mean=pl.col("value").drop_nans().mean().fill_null(float("nan")),
        _median=pl.col("value").drop_nans().median().fill_null(float("nan")),
    )
    aggs = pl.Series("agg", [metric_agg(m) for m in both["metric"].to_list()],
                     dtype=pl.String)
    out = (
        both.with_columns(aggs)
        .with_columns(
            mean=pl.when(pl.col("agg") == "median")
            .then(pl.col("_median"))
            .otherwise(pl.col("_mean"))
        )
        .select(["metric", "mean", "agg"])
        .sort("metric")
    )
    extra = []
    for name, spec in CATALOG.items():
        if spec.derived is None:
            continue
        if selected is not None and name not in selected:
            continue
        value = _derived_value(df, spec, require_components=selected is not None)
        if value is not None:
            extra.append({"metric": name, "mean": value, "agg": "ratio_of_sums"})
    if extra:
        out = pl.concat([out, pl.DataFrame(extra, schema=out.schema)]).sort("metric")
    return out


#: ``median`` is APPENDED, never inserted. ``score_metrics`` looks a row up by name,
#: but the row ORDER is what every published ``agg_results.csv`` and
#: ``baseline_agg.csv`` already carries, and an out-of-tree consumer reading
#: positionally must keep working.
#:
#: The ``mean`` row keeps its name while holding whatever ``MetricSpec.agg`` declares
#: -- renaming it would break compat, score.py and every published artifact (see
#: ``aggregate_metrics``). Every shipped entry declares ``mean`` since #231, so in practice
#: the ``median`` row now always carries the statistic the metric does NOT score on. For a
#: metric declared ``agg="median"`` the two rows would be equal by construction. That is
#: the point either way: moving a metric between statistics must not delete the number it
#: moved away from (#229).
_WIDE_STATISTICS = ("count", "null_count", "mean", "std", "min", "max", "median")


def _wide_metric_names(df, metrics, *, caller: str) -> list[str]:
    """The metric columns ``aggregate_metrics_wide`` reports on, for this (df, metrics) pair.

    Factored out so ``metric_cohorts`` covers exactly the same set (#239). A sidecar that
    described a different metric set from the file it annotates would be worse than no sidecar,
    and two copies of this resolution would drift the first time either changed.
    """
    observed = df["metric"].unique().to_list() if df.height else []
    selected = (None if metrics is None else
                {_NAME_TO_CANONICAL.get(m, m) for m in metrics})
    derived_ready = [n for n, s in CATALOG.items()
                     if s.derived is not None
                     and (selected is None or n in selected)
                     and s.derived.numerator in observed
                     and s.derived.denominator in observed]
    names = sorted(set(observed) | set(metrics or []) |
                   (set(derived_ready) if selected is None else set()))
    if not names:
        raise ValueError(
            f"{caller}: no metrics to aggregate (empty tidy frame and no "
            "`metrics=`). A statistic-only frame scores to a vacuous avg_score of 0.0."
        )
    return names


def metric_cohorts(df: pl.DataFrame, *, metrics: list[str] | None = None) -> pl.DataFrame:
    """Per-metric COHORT SIZES for the aggregate ``aggregate_metrics_wide`` reports (#239).

    Columns: ``metric``, ``n_used``, ``n_rows``, ``n_nan``, ``n_null``, ``derived``.

    ``n_used`` is the number of per-perturbation values the reported statistic was actually
    taken over -- the raw series with NaNs and nulls removed, exactly what
    ``aggregate_metrics_wide`` reduces (``nn = raw.drop_nans()``, then a null-skipping
    ``mean``/``median``).

    ⚠️ This is NOT the wide frame's ``count`` row, and #239 is the reason the distinction
    matters. ``count``/``null_count`` describe the RAW series on purpose ("their job is to
    report what was there"), and polars nulls are not NaNs -- so a metric that returns NaN for
    a perturbation it could not score reports ``count = N`` and ``null_count = 0`` while the
    statistic averaged fewer than N values. MEASURED on the public VCC 2025 training data
    (docs/data/H1-VCC-2025-training.h5ad, half-data arm, vcc2026 preset): the two mechanisms
    split cleanly, and only one of them is already visible.

    * ``de_wilcoxon_lfc_nmae`` OMITS the row for a perturbation its gate drops -- 5 groups in,
      1 row out, ``count = 1``. For this metric ``count`` already IS the cohort, which is why
      the metric #239 is titled about looked satisfied.
    * ``de_wilcoxon_direction_reach_raw`` and ``de_wilcoxon_direction_fidelity_yield_raw``
      EMIT NaN instead -- 5 rows out, 3 of them NaN, ``count = 5``, ``null_count = 0``, mean
      over 2. #239 names ``direction_reach`` explicitly as having the same problem, and for
      this class the cohort was NOT recoverable from the aggregate at all.

    A derived metric (``agg="ratio_of_sums"``) has no per-perturbation rows by construction, so
    its counts are all 0 and ``derived`` is true -- 0 there means "not applicable", not "nothing
    scored". Flagged rather than omitted so the sidecar's metric set still matches the wide
    frame's columns.

    NOT included: the per-perturbation OMISSION REASONS #239 also asks for ("ideally"). They are
    logged by the metric implementations in ``metrics/de.py``, and capturing them means changing
    what those functions return -- a different owner's file and a wider change than a cohort
    column. Filed as the remaining half of #239 rather than reached for.
    """
    _reject_derived_rows(df)
    # Project first: only these two columns are ever read below, and the per-metric filter runs
    # once per name, so carrying `perturbation` (and anything else a caller attached) through each
    # of them is wasted memory. Measured 1.10x on a 188k-row tidy frame -- small, but the memory
    # point is the real one at 940k rows (Gemini, PR #307). AFTER `_reject_derived_rows`, which
    # reports the offending perturbations and therefore needs the full frame.
    df = df.select(["metric", "value"])
    names = _wide_metric_names(df, metrics, caller="metric_cohorts")
    # ONE vectorized pass instead of a filter per metric (Gemini, PR #307 round 3). Measured
    # 2.5x at 18.8k rows, 4.6x at 188k and 940k, and bit-identical on every edge case checked:
    # nulls and NaNs in one column, all-null, empty frame, and a requested metric absent from the
    # frame.
    #
    # ⚠️ I refused this idea in round 1 when Copilot raised it, on the grounds that a group_by
    # rewrite would put the derived-metric and absent-metric branches back in play. That was an
    # objection to a rewrite nobody proposed: those two branches stay exactly where they were, in
    # the loop below, and only the per-metric FILTER moves. Measure the implementation, not the
    # idea.
    #
    # `is_nan()` yields null for a null entry and `.sum()` skips nulls, so `n_nan` counts true NaNs
    # only and `n_used` is `non-null minus NaN`. This no longer depends on `drop_nans()` preserving
    # nulls, which the previous form did.
    stats: dict[str, tuple[int, int, int, int]] = {}
    if df.height:
        agg = df.group_by("metric").agg(
            n_rows=pl.len(),
            n_null=pl.col("value").null_count(),
            n_nan=pl.col("value").is_nan().sum(),
            n_used=pl.col("value").is_not_null().sum() - pl.col("value").is_nan().sum(),
        )
        stats = {r["metric"]: (r["n_used"], r["n_rows"], r["n_nan"], r["n_null"])
                 for r in agg.iter_rows(named=True)}
    rows = []
    for m in names:
        spec = CATALOG.get(_NAME_TO_CANONICAL.get(m, m))
        if spec is not None and spec.derived is not None:
            rows.append({"metric": m, "n_used": 0, "n_rows": 0, "n_nan": 0, "n_null": 0,
                         "derived": True})
            continue
        n_used, n_rows, n_nan, n_null = stats.get(m, (0, 0, 0, 0))
        rows.append({
            "metric": m,
            "n_used": int(n_used),
            "n_rows": int(n_rows),
            "n_nan": int(n_nan),
            "n_null": int(n_null),
            "derived": False,
        })
    return pl.DataFrame(rows, schema={"metric": pl.String, "n_used": pl.Int64,
                                      "n_rows": pl.Int64, "n_nan": pl.Int64,
                                      "n_null": pl.Int64, "derived": pl.Boolean})


def metric_output_names(config: EvalConfig) -> list[str]:
    """The metric names a run with ``config`` emits into the tidy frame, in profile order.

    v1 emits the inherited aliases (``mae``), v2 the canonical names (``expr_mae``). The
    rule is spelled out at run.py:175, run.py:275 and h5ad_manifest.py:209; it is factored here
    rather than copied a fourth time, because ``aggregate_metrics_wide(metrics=...)``
    needs exactly this list to materialize a metric that emitted no rows.

    It must mirror ``dispatch_de_metrics`` (run.py:250-275), not just the catalog: under
    ``de.backend="deseq2"`` a DE metric is relabeled to its ``de_deseq2_*`` sibling by
    ``_effective_de_spec``, and a metric plus an explicitly-selected sibling collapse to
    ONE emitted name via the ``seen`` set. Reading ``CATALOG[n]`` directly instead would
    hand ``aggregate_metrics_wide`` a phantom ``de_wilcoxon_*`` column (all NaN, then
    rejected by the degeneracy gate) alongside the real ``de_deseq2_*`` one.
    """
    names, _ = resolve_metrics(config.metrics, version=config.version)
    out, seen = [], set()
    for n in names:
        spec = CATALOG[n]
        if spec.kind == "de":
            spec = _effective_de_spec(n, config.de.backend)
            if spec.name in seen:
                continue
            seen.add(spec.name)
        out.append(spec.v1_name if (config.version == "v1" and spec.v1_name) else spec.name)
    return out


def aggregate_metrics_wide(
    df: pl.DataFrame, *, metrics: Sequence[str] | None = None
) -> pl.DataFrame:
    """Wide, ``statistic``-indexed aggregate -- the frame ``score.score_metrics`` consumes.

    One row per statistic in ``_WIDE_STATISTICS``, one column per metric, metric columns
    sorted by name. ``score_metrics`` compares ``results_user.columns ==
    results_base.columns`` as an ordered list, so a deterministic column order is
    load-bearing, not cosmetic.

    ``metrics`` is the EXPECTED output-name list (see ``metric_output_names``). Its spellings
    are canonicalized when deciding whether to emit a derived metric. When supplied, a
    derived column is emitted only if selected, and missing components raise instead of
    producing an all-NaN column. With ``metrics=None``, buildable derived metrics continue to
    be discovered from their components and an absent component continues to omit the column.
    Columns
    are the union of expected and observed: an expected metric that emitted no tidy rows
    becomes an all-NaN column instead of vanishing -- otherwise ``score_metrics`` would
    reject the whole run for mismatched columns rather than reporting one undefined
    comparator, and under v1 a metric CAN emit nothing (the no-droppable-NaN fill is
    version-gated at run.py:207-211). An observed-but-unexpected metric is kept rather
    than dropped: that combination is a caller bug, and silent data loss is the worse
    failure. An empty column set RAISES -- a statistic-only frame scores to a vacuous
    ``avg_score = 0.0``, which reads as a real result.

    Every statistic is computed on the NaN-DROPPED series, matching ``aggregate_metrics``
    (issue #92 / PR #52) and NOT ``DataFrame.describe()``, which propagates NaN. The
    difference is not cosmetic: a NaN aggregate is a degenerate baseline, and one
    degenerate perturbation would otherwise decide the whole column. ``score_metrics`` now
    refuses such a baseline for every DECISIVE metric (spec 6; ``catalog.is_decisive``), now
    including every ``vcc2026`` member, and drops any other scored one from ``avg_score``,
    rather than -- as it used to for the anchor-1 class -- silently scoring
    every submission 0 on that metric with no
    error raised. ``count`` and ``null_count`` describe the RAW series, since their job is
    to report what was there.

    An undefined statistic is NaN, never a stand-in value. That includes ``std``: polars
    ``.std()`` is the SAMPLE std, undefined for a single observation and for an empty
    series alike, and reporting ``0.0`` for either is a false claim of zero spread that
    ``comparison_statistic="std"`` would consume as fact.
    """
    _reject_derived_rows(df)
    names = _wide_metric_names(df, metrics, caller="aggregate_metrics_wide")
    empty = pl.Series("value", [], dtype=pl.Float64)
    cols: dict[str, list] = {"statistic": list(_WIDE_STATISTICS)}
    for m in names:
        spec = CATALOG.get(_NAME_TO_CANONICAL.get(m, m))
        if spec is not None and spec.derived is not None:
            # A derived metric has no per-perturbation rows, so every dispersion statistic
            # would describe an empty column. NaN says "not applicable" where 0 would be a
            # claim. Only `mean` is real -- and it is the ratio of sums, not a mean.
            # `selected is not None` before the name resolution moved into
            # _wide_metric_names; it was exactly `metrics is not None` by construction.
            value = _derived_value(df, spec, require_components=metrics is not None)
            cols[m] = [float("nan")] * len(_WIDE_STATISTICS)
            cols[m][_WIDE_STATISTICS.index("mean")] = (
                float("nan") if value is None else value)
            continue
        raw = df.filter(pl.col("metric") == m)["value"] if df.height else empty
        nn = raw.drop_nans()
        # Per-metric statistic (MetricSpec.agg). The ROW keeps the name "mean" -- the wide
        # frame must stay strictly numeric, so the metric->statistic mapping cannot live
        # here as a string row; it goes to the metric_aggregation.csv sidecar instead.
        # The median is computed ONCE and reused. For a metric whose declared `agg` IS the
        # median -- no shipped entry since #231, but the format still admits one -- the
        # `mean` row and the `median` row hold the same number, and calling
        # `nn.median()` twice both did the work twice and left that identity incidental
        # rather than structural (Copilot, #230). Reusing it is why
        # `test_the_median_row_is_unconditional_not_a_second_agg_lookup` cannot drift: the
        # two rows are now the same object, not two computations that happen to agree.
        med = nn.median()
        centre = med if metric_agg(m) == "median" else nn.mean()
        cols[m] = [
            float(raw.len()),
            float(raw.null_count()),
            *(float(v) if v is not None else float("nan")
              for v in (centre, nn.std(), nn.min(), nn.max(), med)),
        ]
    return pl.DataFrame(cols)


def precompute_cache(adata_or_path, *, side: str, config: EvalConfig, de=None,
                     comparator: str | None = None) -> None:
    """Build one side's cache by loading only that side. Writes pseudobulk for every
    normalization the config's metrics need (and the DE rank if `de` is given) to
    config.cache_real (side='real') or config.cache_pred (side='pred').

    ``target_sum=None`` on counts input is supported for ``side='real'`` and refused for
    ``side='pred'`` (#267). The asymmetry is not a policy choice: ``target_sum=None`` resolves
    to the REAL control pool's median library size (#155), and only the real side carries that
    pool. Warming the real side therefore keys its entries at the number a later
    ``compute_metrics`` run resolves to, so they are hit; a pred-side warm would have to invent
    a target and would orphan every entry it wrote.
    """
    if side not in ("real", "pred"):
        raise ValueError(f"side must be 'real' or 'pred', got {side!r}")
    cfg = _resolve_config(config, {})
    root = cfg.cache_real if side == "real" else cfg.cache_pred
    if root is None:
        raise ValueError(f"precompute_cache(side={side!r}) requires config.cache_{side} to be set")
    store = CacheStore(root)

    adata = load_anndata(adata_or_path, backed=isinstance(adata_or_path, (str, os.PathLike)))
    try:
        eff = _effective_input_type(adata, cfg, side=side)
        if cfg.target_sum is None and eff == "counts":
            # #267: the refusal was unconditional, and for side='real' it was too strong. The
            # real CONTROL POOL is inside the one side being loaded, so resolving here calls
            # the same helper on the same rows compute_metrics does (`run.py:75`) and lands on
            # the same number -- which is exactly what makes the resulting entries HITTABLE
            # rather than orphaned. The pert_col check has to move ABOVE this, because the
            # resolver indexes obs[pert_col] and would otherwise raise a bare KeyError first.
            if side != "real":
                raise NotImplementedError(
                    "precompute_cache does not support target_sum=None on counts input for "
                    "side='pred' (#155/#267): the pred side carries no real control pool to "
                    "anchor the median on, and any target it chose would key entries that a "
                    "resolved compute_metrics run could never hit. Pass a numeric target_sum, "
                    "warm the REAL side instead (which can resolve it), or let compute_metrics "
                    "build the cache."
                )
            if cfg.pert_col not in adata.obs.columns:
                raise ValueError(f"perturbation column {cfg.pert_col!r} missing from adata.obs")
            if not (adata.obs[cfg.pert_col].astype(str) == cfg.control).any():
                # resolve_target_sum's own "no control cells" message does not name the label,
                # and a wrong `control` is the likeliest cause on a one-sided warm.
                raise ValueError(
                    f"precompute_cache(side='real') cannot resolve target_sum=None: no rows "
                    f"with {cfg.pert_col}=={cfg.control!r} in this side. target_sum=None "
                    "normalizes to the real control pool's median library size (#155), so the "
                    "control label must be present. Fix `control`, or pass a numeric target_sum."
                )
            cfg = _resolve_target_sum_from_control(cfg, adata)
        if not getattr(adata, "isbacked", False):  # validate in-memory inputs up front, like
            if cfg.validate_input and cfg.version != "v1":
                _validate_input_once(adata, eff, allow_fractional=_val_allow_fractional(cfg, side=side))
            if cfg.validate_input:
                _check_scale_limit_once(adata, eff, cfg.max_counts_per_cell)  # validate in full()
        if cfg.pert_col not in adata.obs.columns:  # clear error before fingerprint/pseudobulk KeyError
            raise ValueError(f"perturbation column {cfg.pert_col!r} missing from adata.obs")
        fp = fingerprint_adata(adata, pert_col=cfg.pert_col, strict=cfg.cache_strict)
        names, _missing = resolve_metrics(cfg.metrics, version=cfg.version)
        if comparator is None:
            expr_names = [n for n in names if CATALOG[n].normalization == _norm.EXPR_COMPARATOR]
            if expr_names:
                raise ValueError(
                    "precompute_cache is one-sided and requires comparator= when expression "
                    f"comparator metrics are requested: {expr_names}"
                )
            comparator = "lognorm"  # inert while every selected metric declares a concrete key
        if comparator not in ("bulk_lognorm", "lognorm"):
            raise ValueError(
                "comparator must be 'bulk_lognorm' or 'lognorm', "
                f"got {comparator!r}"
            )
        if comparator == "bulk_lognorm":
            if cfg.version != "v2":
                raise ValueError(
                    "precompute_cache comparator='bulk_lognorm' requires version='v2', "
                    f"got version={cfg.version!r}"
                )
            if eff != "counts":
                raise ValueError(
                    "precompute_cache comparator='bulk_lognorm' requires the local side to "
                    f"have effective input type 'counts', got {eff!r} for side={side!r}"
                )
        norms = _needed_normalizations(names, comparator=comparator)
        moment_norms = _moment_normalizations(names, comparator=comparator)
        _side_bulks(adata, fp=fp, store=store, norms=norms, cfg=cfg, side=side,
                    moment_norms=moment_norms)  # reuse the open handle

        if de is not None:  # `de` is supplied by the caller -> strict fingerprint, matching the
            df, _perts = prep_de_side(de, name=side, sort_by=cfg.de.sort_by,  # reader in _prepare_de_cached (F9.1)
                                      nan_lfc_policy=cfg.de.nan_lfc_policy,
                                      min_abs_log2fc=cfg.de.min_abs_log2fc)
            rank = rank_de_side(df, sort_by=cfg.de.sort_by, p_adj_threshold=cfg.de.p_adj_threshold)
            if rank.width > 0:  # empty (no-significant-gene) ranks deliberately not cached — see _prepare_de_cached
                store.put(f"de_{cfg.de.method}_rank", rank,
                          fingerprint=fingerprint_de_table(df, strict=_de_supplied_strict(True, cfg)),
                          params={"sort_by": cfg.de.sort_by, "p_adj_threshold": cfg.de.p_adj_threshold,
                                  "nan_lfc_policy": cfg.de.nan_lfc_policy,
                                  "min_abs_log2fc": cfg.de.min_abs_log2fc},
                          kind="parquet")
    finally:
        _close_backed(adata, adata_or_path)  # close the handle we opened, even on error
