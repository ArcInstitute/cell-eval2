"""Streaming, memory-bounded anndata-metric scoring over packed ``.shad`` archives.

Composes the Plan-1 foundation: stream both sides' pseudobulk (the real reference
once, optionally cached via ``config.cache_real``), restrict the pred side to a
perturbation subset, run the existing **anndata** metric dispatch, and optionally
write a partial for later aggregation.

**DE is implemented here, on both layouts** -- ``score_streaming`` (shard) goes
through ``de_compute.compute_de_streaming`` and ``score_streaming_cell`` (cell
archive) through ``compute_de_streaming_cell``, both on the gpudge engine, which
needs a CUDA device; a DE-free metric set runs on CPU. An earlier revision of this
docstring said DE raised ``NotImplementedError`` here, which has not been true since
those two landed. Every ``NotImplementedError`` that remains is about the INPUT TYPE,
not the metric kind: ``target_sum=None`` on a non-counts archive in ``score_streaming``,
and any non-counts ``input_type`` at all in ``score_streaming_cell``.
"""

from __future__ import annotations

import logging
from dataclasses import replace

import numpy as np
import polars as pl

from . import norm as _norm
from . import cell_source
from ._cell_archive import open_cell_store
from .catalog import CATALOG, resolve_metrics
from .de import prepare_de, resolve_target_genes
from .de_compute import compute_de_streaming, compute_de_streaming_cell
from .partition import (PARTIAL_SEMANTICS_KEY, result_semantics, select_subset,
                        write_partial)
from .run import (_moment_normalizations, _needed_normalizations,
                  _resolve_config, dispatch_anndata_metrics, dispatch_de_metrics,
                  effective_normalization)
from .io import validate_gene_axis
from .stream import shad_fingerprint, shad_metadata, shad_var_names
from .streaming_bulk import streaming_pseudobulk

logger = logging.getLogger(__name__)

_TIDY_SCHEMA = {"perturbation": pl.Utf8, "metric": pl.Utf8, "value": pl.Float64}


def _shard_effective_input_type(path, cfg, *, side: str) -> str:
    """Resolve one shard archive's effective type from a small resident group."""
    from cellstream.read import ShardedArchive

    arch = ShardedArchive(path)
    meta = shad_metadata(path)
    stop = min(int(meta.n_obs), 2000)
    if stop == 0:
        raise ValueError(f"cannot resolve input type from empty archive {path!r}")
    # Slice BEFORE materialization. A perturbation group or reference shard can be millions of
    # rows; read_group/read_reference would turn a 2k-row type peek into an unbounded load.
    peek = arch[:stop].to_anndata()
    # Shard streaming must inspect BOTH archives even though the ordinary in-memory v2 policy
    # keeps real-side autodetection strict/off. The shard accumulator has no input_type argument
    # and always treats stored values as counts, so trusting a declaration here would turn a
    # lognorm real archive into plausible wrong bulk values instead of triggering the safety gate.
    #
    # ⚠️ allow_discrete=False UNCONDITIONALLY, and not cfg.allow_discrete. `resolve_input_type`
    # returns "counts" the moment allow_discrete is set, WITHOUT looking at a single value (its
    # `if allow_discrete: return "counts"` short-circuit), so passing the config's value let a
    # genuinely lognorm archive declare its way straight through the safety gate below and be
    # normalized as counts -- the exact failure the gate exists to stop, reachable by setting an
    # unrelated-sounding flag. Found by codex-review.
    # `allow_discrete` is a policy about how to READ an ambiguous integer matrix; it is not evidence
    # about what this archive contains, which is the only question here.
    return _norm.resolve_input_type(
        peek, declared=cfg.input_type, version=cfg.version,
        allow_discrete=False, autodetect=True,
    )


def score_streaming(pred, real, *, config, subset=None, fraction=None, index=None,
                    partial_out=None, noise=None) -> pl.DataFrame:
    cfg = _resolve_config(config, {})
    names, _missing = resolve_metrics(cfg.metrics, version=cfg.version)
    effective_types = {
        "pred": _shard_effective_input_type(pred, cfg, side="pred"),
        "real": _shard_effective_input_type(real, cfg, side="real"),
    }
    comparator = _norm.resolve_comparator(
        version=cfg.version,
        pred_input_type=effective_types["pred"],
        real_input_type=effective_types["real"],
    )
    # #266/#182: the DE family gets the same gate as the anndata family. It had none -- verified
    # on `main` before #264, which narrowed the hole for `kind="anndata"` and left the DE arm
    # exactly where it was. `compute_de_streaming` documents and assumes raw counts
    # (de_compute.py: "The .shad stores raw counts") and took no input_type argument at all, so a
    # lognorm archive with a NUMERIC target_sum returned plausible DE numbers computed from
    # re-normalized lognorm values rather than raising. (`target_sum=None` was already covered by
    # #155's guard above; a numeric target is exactly as wrong and slipped straight through.)
    #
    # Keyed on the EFFECTIVE type for both families, never the declared one -- a declared-counts
    # config over lognorm data is the case #266 names, and it is the only one that matters here:
    # `_shard_effective_input_type` hard-codes autodetect=True (scale.py:63), so this gate sees
    # the DATA even when the config's `autodetect_input_type` is off.
    #
    # ⚠️ CONSEQUENCE, larger than either issue's text and worth stating: the catalog holds exactly
    # two kinds (13 anndata, 82 de), and EVERY anndata entry's effective normalization is in the
    # unsafe set under either comparator -- verified against the catalog, not assumed. So with the
    # DE family added, this gate covers every metric that exists, and `score_streaming` is now
    # counts-only in practice, matching `score_streaming_cell`'s already-declared contract.
    #
    # The `comparator == "lognorm"` fallback is therefore unreachable FOR SCORING AND OUTPUT on this
    # driver -- NOT unreachable outright, which is what an earlier wording of this comment claimed
    # (codex-review round 7 corrected it). `resolve_comparator` runs ABOVE this gate, so on an
    # asymmetric declaration the fallback IS resolved, and since #302 it also WARNS -- both
    # observable from outside. What it never reaches is metric dispatch or partial emission. The
    # fallback is left in place because the comparator resolution is shared, not driver-specific.
    unsafe_names = [
        n for n in names
        if CATALOG[n].kind == "anndata"
        and effective_normalization(CATALOG[n], comparator)
        in {"bulk_lognorm", "lognorm", "normalized"}
    ]
    unsafe_de_names = [n for n in names if CATALOG[n].kind == "de"]
    non_counts = {s: t for s, t in effective_types.items() if t != "counts"}
    if (unsafe_names or unsafe_de_names) and non_counts:
        detail = []
        if unsafe_names:
            detail.append(
                "anndata metrics whose effective normalization the shard accumulator cannot "
                f"safely serve from stored non-counts values: {unsafe_names} (it treats stored "
                "values as counts and would normalize them again)"
            )
        if unsafe_de_names:
            detail.append(
                f"DE metrics: {unsafe_de_names} (compute_de_streaming hands gpudge a library-size "
                "normalization target and has no expm1 step, so log1p'd values would be rescaled "
                "as counts -- #182/#266)"
            )
        # Metric detail BEFORE the effective types, matching the pre-#266 message order:
        # tests/test_scale_runner.py:137 asserts `expr_mse.*pred=.*real=`, and that assertion is
        # about the message naming both the metric and both sides -- which it still does. An
        # incidental reordering is not worth breaking a passing check for.
        # #155's own NotImplementedError said the same thing for the target_sum=None case, from
        # BELOW the median resolution. Now that this gate runs first it subsumes that case for
        # every metric -- so carry #155's specific advice here rather than let the more useful
        # message become unreachable. Keeping the literal "target_sum" is deliberate: it names the
        # condition, and what it must NOT say is "pass a numeric target_sum", because a numeric
        # target rescales log1p values just as badly.
        target_note = (
            " This run also has target_sum=None, which on a non-counts archive has no library-size "
            "median to resolve (#155): the pseudobulk accumulator normalizes every cell to the "
            "target and the streaming DE path hands gpudge a normalization target, so already-"
            "log1p'd values would be rescaled either way. Re-write the archive from raw counts, or "
            "score it through the in-memory path."
            if cfg.target_sum is None else ""
        )
        raise ValueError(
            "score_streaming (.shad) requires raw counts on BOTH sides. Affected -- "
            + " Also: ".join(detail)
            + f"; effective input types are pred={effective_types['pred']!r}, "
            f"real={effective_types['real']!r}." + target_note
            + " ⚠️ If these values are fractional COUNTS rather "
            "than log-normalized ones, this gate cannot tell the two apart: "
            "`norm.guess_is_lognorm` classifies any matrix with a fractional per-cell total as "
            "lognorm, so a scaled or averaged counts matrix (e.g. a dispersed baseline arm) lands "
            "here too. Score that through the in-memory path, which takes the declaration via "
            "allow_fractional_counts."
        )

    # #155: resolve target_sum=None to ONE number before anything is computed, so the streaming
    # pseudobulk (which cannot take None: float(None) in gpu/bulk.py:56) and the DE path use the
    # same target, and that target means the same thing here as in the in-memory and partitioned
    # entry points. The real archive's reference shard IS the real control pool; it is already
    # read below for control_source='real'. gpudge would otherwise resolve its own union median
    # over reference + all targets -- correct, but a different number.
    if cfg.target_sum is None:
        from cellstream.read import ShardedArchive

        _arch = ShardedArchive(real)
        # read_reference() returns None when the archive was written WITHOUT a designated
        # reference (cellstream.read.ShardedArchive.read_reference) -- fall back to the control
        # group. read_group RAISES KeyError for an unknown label
        # (cellstream.read.ShardedArchive.read_group); it never returns None, so without this
        # except the caller gets a bare KeyError instead of the actionable message.
        # Cited by SYMBOL, not line number: the old `shardad/read.py:1144-1158` / `:1352` cites
        # were already stale against a renamed package and cannot be checked by grep.
        _ref_ad = _arch.read_reference()
        if _ref_ad is None:
            try:
                _ref_ad = _arch.read_group(cfg.control)
            except KeyError:
                _ref_ad = None
        if _ref_ad is None or _ref_ad.n_obs == 0:
            # A numeric target_sum is NOT a fix for a missing reference/control. It bypasses this
            # median-resolution step and nothing else: streaming DE still reads
            # the archive's own reference shard (scale.py:137 -> compute_de_streaming(
            # reference=None)) and the delta/discrimination metrics still look cfg.control up by
            # name (delta.py:116, discrimination.py:111), so the run fails downstream on exactly
            # the same missing thing. Say what actually fixes it. (Spec section 8.)
            raise ValueError(
                f"target_sum=None needs the real archive's control pool, but {real!r} has no "
                f"reference shard and no group {cfg.control!r}. Fix the control label, or write "
                "the archive with a designated reference. For raw-count input an explicit "
                "numeric target_sum bypasses only this median-resolution requirement -- DE and "
                "the control-relative metrics still need a valid reference/control."
            )
        # effective_types["real"], NOT a second _effective_input_type call: that helper applies the
        # IN-MEMORY policy, which under v2 trusts the declaration and would answer "counts" for a
        # lognorm archive -- yielding a meaningless median from log1p values. The shard-side type was
        # already resolved from the data at the top of this function (codex-review). Unreachable now
        # that the non-counts gate runs above, but wrong is wrong: this is the value that belongs
        # here, and the gate should not be the only thing standing between the two.
        cfg = replace(cfg, target_sum=_norm.resolve_target_sum(
            _ref_ad, input_type=effective_types["real"], target_sum=None))

    # Still None => effective-lognorm input, where the resolver correctly has no library size to
    # take a median of. That is NOT inert on this path -- BOTH halves misbehave, so this guard
    # covers every metric kind, not just anndata ones:
    #   - anndata: streaming_bulk.py:128 computes `target_sum / libs` and gpu/bulk.py:56 does
    #     float(target_sum) -> TypeError;
    #   - DE: compute_de_streaming takes no input_type at all and maps None -> gpudge "median"
    #     unconditionally (de_compute.py:667), so it would library-size-normalize values that are
    #     already log1p'd. Silently wrong, which is the worse failure of the two. Its own
    #     docstring states the contract it assumes: "The .shad stores raw counts"
    #     (de_compute.py:648).
    # An earlier revision let the DE-only case through as "unaffected". It is not unaffected; it
    # is unchecked.
    #
    # ⚠️ MOSTLY SUBSUMED, and kept. `target_sum` can only still be None here for effective-lognorm
    # input, and #266's gate above refuses that for every metric kind (the catalog has only two, and
    # both are covered) -- so this branch is unreachable for every NON-EMPTY current metric
    # selection. It IS still reachable when the resolved selection is EMPTY (`metrics=[]`, or a list
    # of only deferred names): both unsafe lists are then empty, the gate does not fire, and this is
    # the guard that stops a lognorm archive here. Round 2 of codex-review corrected my earlier
    # "unreachable" claim, which was wrong in exactly that case. Its advice also appears in the
    # gate's message, so neither wording is lost.
    if cfg.target_sum is None:
        raise NotImplementedError(
            # Keep the literal "target_sum" in the text: the Step 1 tests match on it, and it
            # names the condition. What it must NOT say is "pass a numeric target_sum" -- a
            # numeric target rescales log1p values just as badly (spec section 8).
            "streaming (.shad) scoring with target_sum=None requires a RAW-COUNTS archive: "
            "this one's effective input type is lognorm, so there is no library-size median to "
            "resolve (#155). The "
            "pseudobulk accumulator normalizes every cell to the target and the streaming DE "
            "path hands gpudge a normalization target, so already-log1p'd values would be "
            "rescaled either way. Re-write the archive from raw counts, or score it through "
            "the in-memory path."
        )

    de_names = [n for n in names if CATALOG[n].kind == "de"]
    anndata_names = [n for n in names if CATALOG[n].kind == "anndata"]

    # Real-reference identity is ALWAYS derived from the archive (never a placeholder), so the
    # aggregate_partials safety guard is meaningful even without a shared cache: two nodes scoring
    # against the same real .shad get the same fingerprint (safe to aggregate); a different real
    # archive trips the guard. Caching (cfg.cache_real) only changes whether we recompute.
    real_fp = shad_fingerprint(real)
    meta = shad_metadata(real)
    genes = meta.var_names
    # Fail fast on a mismatched gene axis, mirroring io.validate_pair (the in-memory oracle) and
    # score_streaming_cell's validate_cell_pair. Everything below adopts REAL's var order for
    # BOTH sides, so a same-count/different-order pred would be scored gene-POSITION-wise and
    # return plausible finite numbers (ultrareview 2026-07-25). shad_var_names, not
    # shad_metadata: the latter also decodes the full obs to build `perts`, which is pure waste
    # here and is not otherwise paid for the pred side.
    validate_gene_axis(shad_var_names(pred), genes)
    # Subset/fraction selects PRED targets; the real reference always stays the full archive.
    universe = sorted(p for p in meta.perts if p != cfg.control)
    chosen = set(select_subset(universe, subset=subset, fraction=fraction, index=index))

    rows: list[dict] = []

    # --- DE metrics: gpudge streaming (per-cell MWU); all DE metrics share one table/side ---
    if de_names:
        rows.extend(_score_streaming_de(pred, real, cfg=cfg, de_names=de_names,
                                        chosen=chosen, real_fp=real_fp,
                                        effective_types=effective_types))

    # --- anndata/discrimination metrics: comparator-resolved pseudobulk stream ---
    # (a complementary pass; the space is whatever `comparator` resolved to, normally
    # `bulk_lognorm` on a v2 counts run since #264)
    # The pred pseudobulk pass also yields the exact median UMI/cell (free — it touches every
    # cell), surfaced for a future gpudge normalize_target_sum='median' mode. For v2
    # (target_sum=1e6) gpudge uses cpm_normalize, so the median is recorded but NOT fed to the
    # DE call.
    median_umi = None
    norms = _needed_normalizations(anndata_names, comparator=comparator)
    if norms:
        moment_norms = _moment_normalizations(anndata_names, comparator=comparator)
        moment_norm_list = [n for n in norms if n in moment_norms]
        plain_norms = [n for n in norms if n not in moment_norms]

        # The public stream APIs intentionally keep one with_moments flag for all requested
        # norms. A profile needing moments on some norms and not others is therefore two
        # explicit passes -- the moment-bearing norms first, then the plain ones -- merged
        # back into one normalization-keyed dict. WHICH norm is which follows the resolved
        # comparator, not a fixed name: since #264 PR2 the moment-bearing one is normally
        # `bulk_lognorm` (a v2 counts run needs no `lognorm` pass at all), and it is
        # `lognorm` on the fallback. An earlier revision of this comment named them.
        real_bulks = {}
        real_moments = {} if moment_norms else None
        if moment_norm_list:
            built, built_moments = _real_reference(
                real, cfg=cfg, norms=moment_norm_list, real_fp=real_fp, with_moments=True,
            )
            real_bulks.update(built)
            real_moments.update(built_moments)
        if plain_norms:
            real_bulks.update(_real_reference(
                real, cfg=cfg, norms=plain_norms, real_fp=real_fp, with_moments=False,
            ))

        pred_bulks = {}
        pred_moments = {} if moment_norms else None
        if moment_norm_list:
            built, median_umi, built_moments = streaming_pseudobulk(
                pred, pert_col=cfg.pert_col, norms=moment_norm_list,
                target_sum=cfg.target_sum, noise=noise, device=cfg.device,
                with_median_umi=True, with_moments=True,
                bulk_target_sum=cfg.bulk_target_sum,
            )
            pred_bulks.update(built)
            pred_moments.update(built_moments)
        if plain_norms:
            built = streaming_pseudobulk(
                pred, pert_col=cfg.pert_col, norms=plain_norms,
                target_sum=cfg.target_sum, noise=noise, device=cfg.device,
                with_median_umi=not moment_norms, with_moments=False,
                bulk_target_sum=cfg.bulk_target_sum,
            )
            if moment_norms:
                pred_bulks.update(built)
            else:
                built, median_umi = built
                pred_bulks.update(built)
        # _restrict touches the BULKS ONLY. Moments span every group including the control
        # and must never be restricted -- the delta_magnitude_* family (#202) needs the
        # control's trace AFTER the control row has been dropped from the bulk.
        # issue #348: the correction budget in `expr_mse_unbiased_capped` is a
        # whole-PANEL quantity, so the unrestricted dict goes to the dispatcher
        # alongside the restricted one -- otherwise that member would depend on the
        # partitioning. Capture before `_restrict`; it is the same object on a
        # whole-panel run, where `_restrict` is a no-op.
        pred_bulks_full = pred_bulks
        pred_bulks = _restrict(pred_bulks, chosen | {cfg.control})
        # #257: `expr_distance_unbiased` reads only the REAL side, so on a partial run it
        # emits the WHOLE panel and `partition.aggregate_partials` rejects the repeated
        # (perturbation, metric) rows. Restrict the ROWS, not `real_bulks`: `pds_*` ranks each
        # predicted effect against the FULL real panel and takes its denominator from the full
        # real count (`discrimination.py:114-150`), so restricting the real bulk would silently
        # change every partial's PDS. `chosen` comes from the REAL universe via `select_subset`,
        # never from the submission, so this is submitter-independent -- and it mirrors what the
        # DE path already does with `chosen` (`_score_streaming_de` filters both tables).
        # It is a no-op on a whole-panel run, where `chosen` IS the universe.
        rows.extend(r for r in dispatch_anndata_metrics(
            anndata_names, pred_bulks, real_bulks, genes, cfg,
            comparator=comparator,
            pred_moments=pred_moments, real_moments=real_moments,
            pred_bulks_full=pred_bulks_full,
            driver="score_streaming (shard-stream)") if r["perturbation"] in chosen)
        logger.info("streaming pred median UMI/cell = %s", median_umi)

    df = pl.DataFrame(rows, schema=_TIDY_SCHEMA)
    if partial_out:
        from .cache import config_hash

        sid = f"frac{fraction}_idx{index}" if fraction is not None else "all"
        meta_out = {
            "real_ref_fingerprint": real_fp,
            "config_hash": config_hash(cfg.to_dict()),
            "comparator": comparator,
            "metrics": sorted(names),
            "perturbations": sorted(chosen),
            # #246: what the numbers MEAN, not just which metrics were selected.
            PARTIAL_SEMANTICS_KEY: result_semantics(names, comparator=comparator),
        }
        if median_umi is not None:
            meta_out["median_umi_pred"] = median_umi
        write_partial(df, partial_out, subset_id=sid, meta=meta_out)
    return df


def _score_streaming_de(pred, real, *, cfg, de_names, chosen, real_fp, effective_types=None):
    """Build pred/real gpudge DE tables off the .shad archives, restrict to ``chosen``,
    assemble, and dispatch the DE metrics. Real side uses Mode 1 (archive's own reference
    shard); the v2 pred side (control_source='real') uses Mode 2 (real's control as an
    external AnnData reference).

    ``effective_types`` (#182/#266): each side's resolved type, threaded to
    ``compute_de_streaming`` so the raw-counts contract is enforced at the function that states
    it and not only by the caller's gate. Defaults to counts/counts, which is what the gate in
    ``score_streaming`` has already guaranteed by the time this runs -- the argument makes the
    guarantee explicit rather than assumed, and it is what a future direct caller would need."""
    eff = {"pred": "counts", "real": "counts"} if effective_types is None else effective_types
    real_de = _real_de_table(real, cfg=cfg, real_fp=real_fp, input_type=eff["real"])
    if cfg.control_source == "real":
        from cellstream.read import ShardedArchive

        pred_ref = ShardedArchive(real).read_reference()  # real's non-targeting control pool
    else:  # control_source == "pred": pred archive's own reference shard (Mode 1)
        pred_ref = None
    pred_de = compute_de_streaming(
        pred, backend=cfg.de.backend, reference=pred_ref, groupby=cfg.pert_col,
        mean_calc=cfg.de.mean_calc, epsilon=cfg.de.epsilon, target_sum=cfg.target_sum,
        clip_value=cfg.de.clip_value, fdr_scope=cfg.de.fdr_scope,
        filter_gene_min_cpm_cell=cfg.filter.filter_gene_min_cpm_cell,
        input_type=eff["pred"],
    )
    chosen_l = list(chosen)
    # Resolve BEFORE slicing: the gate is a property of the DATASET (spec 2.7b). Resolving
    # after the filter would leave a single-target piece with only that target's universe
    # as its "global" index -- exactly the universe an H1_CGS-shaped dataset drops the gene
    # from -- so the piece would raise while the same data scored whole passes.
    resolution = resolve_target_genes(
        real_de, sorted(real_de["target"].unique().to_list()),
        target_gene_map=cfg.target_gene_map,
    )
    real_de = real_de.filter(pl.col("target").is_in(chosen_l))
    pred_de = pred_de.filter(pl.col("target").is_in(chosen_l))
    prepared = prepare_de(
        pred_de, real_de, control=cfg.control, sort_by=cfg.de.sort_by,
        p_adj_threshold=cfg.de.p_adj_threshold, nan_lfc_policy=cfg.de.nan_lfc_policy,
        min_abs_log2fc=cfg.de.min_abs_log2fc, target_resolution=resolution,
    )
    return dispatch_de_metrics(de_names, prepared, cfg)


def _real_de_table(real, *, cfg, real_fp, input_type: str = "counts"):
    """Real-side gpudge DE table (Mode 1, archive's own reference shard). Cached once under
    cfg.cache_real, namespaced by resolved device + DE config, so multi-subset pred runs reuse
    it without re-streaming the reference. The real DE table is the FULL target set; subsetting
    happens in the caller."""
    from .cache import MISS, CacheStore
    from .gpu import resolve_device

    def compute():
        return compute_de_streaming(
            real, backend=cfg.de.backend, reference=None, groupby=cfg.pert_col,
            mean_calc=cfg.de.mean_calc, epsilon=cfg.de.epsilon, target_sum=cfg.target_sum,
            clip_value=cfg.de.clip_value, fdr_scope=cfg.de.fdr_scope,
            filter_gene_min_cpm_cell=cfg.filter.filter_gene_min_cpm_cell,
            input_type=input_type,
        )

    if not cfg.cache_real:
        return compute()
    store = CacheStore(cfg.cache_real)
    params = {
        "pert_col": cfg.pert_col, "control": cfg.control, "method": cfg.de.method,
        "mean_calc": cfg.de.mean_calc, "epsilon": cfg.de.epsilon, "target_sum": cfg.target_sum,
        "clip_value": cfg.de.clip_value, "fdr_scope": cfg.de.fdr_scope,
        "filter_gene_min_cpm_cell": cfg.filter.filter_gene_min_cpm_cell,
        "device": resolve_device(cfg.device), "version": cfg.version,
    }
    hit = store.get(f"stream_de_{cfg.de.method}", fingerprint=real_fp, params=params, kind="parquet")
    if hit is not MISS:
        return hit
    df = compute()
    store.put(f"stream_de_{cfg.de.method}", df, fingerprint=real_fp, params=params, kind="parquet")
    return df


def _real_reference(real, *, cfg, norms, real_fp, with_moments=False):
    """Real-side pseudobulk per normalization, cached once under ``cfg.cache_real`` when set.

    Streams the real archive at most ONCE -- only for the normalizations missing from the
    cache. Streaming cache keys are namespaced (``stream_pseudobulk_*``) so they never collide
    with ``compute_metrics``' own ``pseudobulk_*`` artifacts if the same cache dir is reused.
    """
    from .cache import MISS, CacheStore
    from .gpu import resolve_device

    store = CacheStore(cfg.cache_real) if cfg.cache_real else None
    # Namespace the cache by resolved device: GPU means are fp32, CPU means fp64, so a
    # cpu run must never reuse a fp32 GPU-written reference (keeps device="cpu" exact).
    params = {
        "pert_col": cfg.pert_col, "target_sum": cfg.target_sum,
        "bulk_target_sum": cfg.bulk_target_sum,
        "device": resolve_device(cfg.device),
    }
    out, missing = {}, []
    moments = {} if with_moments else None
    key_prefix = "stream_pseudobulk_moments_" if with_moments else "stream_pseudobulk_"
    kind = "npz_moments" if with_moments else "npz"
    for n in norms:
        if store is not None:
            hit = store.get(f"{key_prefix}{n}", fingerprint=real_fp, params=params, kind=kind)
            if hit is not MISS:
                if with_moments:
                    out[n], moments[n] = hit
                else:
                    out[n] = hit
                continue
        missing.append(n)
    if missing:
        computed = streaming_pseudobulk(
            real, pert_col=cfg.pert_col, norms=missing, target_sum=cfg.target_sum,
            device=cfg.device, with_moments=with_moments,
            bulk_target_sum=cfg.bulk_target_sum,
        )  # single pass for all missing
        computed, computed_moments = computed if with_moments else (computed, None)
        for n in missing:
            out[n] = computed[n]
            if with_moments:
                moments[n] = computed_moments[n]
            if store is not None:
                value = (computed[n], computed_moments[n]) if with_moments else computed[n]
                store.put(f"{key_prefix}{n}", value, fingerprint=real_fp, params=params,
                          kind=kind)
    return (out, moments) if with_moments else out


def _restrict(bulks, chosen):
    chosen = list(chosen)
    out = {}
    for n, (perts, means) in bulks.items():
        keep = np.isin(perts, chosen)  # vectorized membership
        out[n] = (perts[keep], means[keep])
    return out


def score_streaming_cell(pred, real, *, config, subset=None, fraction=None, index=None,
                         partial_out=None, noise=None) -> pl.DataFrame:
    """Out-of-core scoring of a pair of cell-layout ``cellstream.cell`` archives (#117 Stage 2).

    The cell-layout counterpart of :func:`score_streaming`: streams both sides' pseudobulk
    per group (reusing the frozen accumulators) and streams gpudge DE via a cell target
    source, then runs the SAME layout-agnostic dispatch/prepare helpers. Matches the Stage-1
    materialize path (compute_metrics on the same archives, its correctness oracle):
    rank/set metrics (pds_*, de_*) bit-exact, continuous metrics (mae/mse/pearson) to ~1e-8
    relative (float summation-order; see tests/test_cell_source.py::_assert_parity). First
    supported pairing: BOTH sides cell-layout. gpudge DE needs a CUDA device; a DE-free
    metric set runs on CPU."""
    cfg = _resolve_config(config, {})
    if cfg.input_type != "counts":
        raise NotImplementedError(
            "score_streaming_cell (cell-layout streaming) is counts-only: both its gpudge DE "
            "and its anndata pseudobulk assume raw counts and have no expm1/_to_linear, so "
            f"lognorm/scaled_log1p input would be silently mis-normalized. Got input_type="
            f"{cfg.input_type!r}. lognorm streaming is deferred (see #129/#130); use the "
            "materialize path (compute_metrics/run) or write raw-count archives."
        )
    names, _missing = resolve_metrics(cfg.metrics, version=cfg.version)
    comparator = _norm.resolve_comparator(
        version=cfg.version, pred_input_type="counts", real_input_type="counts",
    )
    de_names = [n for n in names if CATALOG[n].kind == "de"]
    anndata_names = [n for n in names if CATALOG[n].kind == "anndata"]
    if cfg.cache_real:
        # Unlike shard-layout score_streaming, the cell path does not yet cache the real-side
        # streaming artifacts; warn rather than silently ignore cache_real (Copilot #127).
        logger.warning(
            "cache_real is set but not yet supported by score_streaming_cell (cell-layout "
            "streaming) — the real side is recomputed each call; a cell-layout cache is a "
            "deferred optimization."
        )

    pred_store = open_cell_store(pred)
    real_store = None
    try:
        real_store = open_cell_store(real)
        # #266/#182, the cell-layout half. The gate above keys on cfg.input_type -- the DECLARED
        # value -- so a config declaring counts over lognorm archives walked straight through it
        # and the values were silently mis-normalized, which is exactly what that gate's own
        # message says must not happen. Check the EFFECTIVE type of both archives too, using the
        # same peek the cellstream driver uses. Deliberately AFTER the declared check so a caller
        # who declared lognorm still gets the "counts-only, use the materialize path" message
        # rather than a data-shaped one, and after the stores are open so it costs one 2k-row peek
        # per side rather than a second open.
        from .cellstream import _resolve_input_type_cell
        # allow_discrete=False: this is a SAFETY classification, and honouring cfg.allow_discrete
        # would let `resolve_input_type` answer "counts" without inspecting a value (codex-review).
        eff_cell = {"pred": _resolve_input_type_cell(pred_store, cfg, allow_discrete=False),
                    "real": _resolve_input_type_cell(real_store, cfg, allow_discrete=False)}
        if any(t != "counts" for t in eff_cell.values()):
            raise NotImplementedError(
                "score_streaming_cell (cell-layout streaming) is counts-only, and these archives' "
                f"stored values are not counts: pred={eff_cell['pred']!r}, "
                f"real={eff_cell['real']!r} (the config DECLARED "
                f"input_type={cfg.input_type!r}). Both its gpudge DE and its anndata pseudobulk "
                "have no expm1/_to_linear step, so the values would be normalized as if they were "
                "counts. Use the materialize path (compute_metrics/run), or write raw-count "
                "archives. ⚠️ Fractional COUNTS land here too: `norm.guess_is_lognorm` classifies "
                "any matrix with a fractional per-cell total as lognorm and cannot tell a scaled "
                "or averaged counts matrix from a log-normalized one."
            )
        real_fp = cell_source.cell_fingerprint(real_store)
        meta = cell_source.cell_metadata(real_store)
        # Fail fast on a mismatched (pred, real) pair, mirroring io.validate_pair (the
        # materialize oracle's check) -> same error class as compute_metrics (Copilot #127).
        cell_source.validate_cell_pair(cell_source.cell_metadata(pred_store), meta,
                                       pert_col=cfg.pert_col, control=cfg.control)
        genes = meta.var_names
        universe = sorted(p for p in meta.perts if p != cfg.control)
        chosen = set(select_subset(universe, subset=subset, fraction=fraction, index=index))

        rows: list[dict] = []

        # --- DE metrics: gpudge streaming (per-cell MWU), both sides ---
        if de_names:
            rows.extend(_score_streaming_cell_de(
                pred_store, real_store, cfg=cfg, de_names=de_names, chosen=chosen))

        # --- anndata/discrimination metrics: per-group pseudobulk stream, both sides ---
        median_umi = None
        norms = _needed_normalizations(anndata_names, comparator=comparator)
        if norms:
            moment_norms = _moment_normalizations(anndata_names, comparator=comparator)
            moment_norm_list = [n for n in norms if n in moment_norms]
            plain_norms = [n for n in norms if n not in moment_norms]

            real_bulks = {}
            real_moments = {} if moment_norms else None
            if moment_norm_list:
                built, built_moments = cell_source.cell_pseudobulk(
                    real_store, pert_col=cfg.pert_col, norms=moment_norm_list,
                    target_sum=cfg.target_sum, device=cfg.device,
                    gather_threads=cfg.gather_threads, with_moments=True,
                    bulk_target_sum=cfg.bulk_target_sum,
                )
                real_bulks.update(built)
                real_moments.update(built_moments)
            if plain_norms:
                real_bulks.update(cell_source.cell_pseudobulk(
                    real_store, pert_col=cfg.pert_col, norms=plain_norms,
                    target_sum=cfg.target_sum, device=cfg.device,
                    gather_threads=cfg.gather_threads, with_moments=False,
                    bulk_target_sum=cfg.bulk_target_sum,
                ))

            pred_bulks = {}
            pred_moments = {} if moment_norms else None
            if moment_norm_list:
                built, median_umi, built_moments = cell_source.cell_pseudobulk(
                    pred_store, pert_col=cfg.pert_col, norms=moment_norm_list,
                    target_sum=cfg.target_sum, noise=noise, device=cfg.device,
                    with_median_umi=True, gather_threads=cfg.gather_threads,
                    with_moments=True, bulk_target_sum=cfg.bulk_target_sum,
                )
                pred_bulks.update(built)
                pred_moments.update(built_moments)
            if plain_norms:
                built = cell_source.cell_pseudobulk(
                    pred_store, pert_col=cfg.pert_col, norms=plain_norms,
                    target_sum=cfg.target_sum, noise=noise, device=cfg.device,
                    with_median_umi=not moment_norms, gather_threads=cfg.gather_threads,
                    with_moments=False, bulk_target_sum=cfg.bulk_target_sum,
                )
                if moment_norms:
                    pred_bulks.update(built)
                else:
                    built, median_umi = built
                    pred_bulks.update(built)
            # _restrict touches the BULKS ONLY. Moments span every group including the control
            # and must never be restricted -- the delta_magnitude_* family (#202) needs the
            # control's trace AFTER the control row has been dropped from the bulk.
            # issue #348: the correction budget in `expr_mse_unbiased_capped` is a
            # whole-PANEL quantity, so the unrestricted dict goes to the dispatcher
            # alongside the restricted one -- otherwise that member would depend on the
            # partitioning. Capture before `_restrict`; it is the same object on a
            # whole-panel run, where `_restrict` is a no-op.
            pred_bulks_full = pred_bulks
            pred_bulks = _restrict(pred_bulks, chosen | {cfg.control})
            # #257: `expr_distance_unbiased` reads only the REAL side, so on a partial run it
            # emits the WHOLE panel and `partition.aggregate_partials` rejects the repeated
            # (perturbation, metric) rows. Restrict the ROWS, not `real_bulks`: `pds_*` ranks each
            # predicted effect against the FULL real panel and takes its denominator from the full
            # real count (`discrimination.py:114-150`), so restricting the real bulk would silently
            # change every partial's PDS. `chosen` comes from the REAL universe via `select_subset`,
            # never from the submission, so this is submitter-independent -- and it mirrors what the
            # DE path already does with `chosen` (`_score_streaming_de` filters both tables).
            # It is a no-op on a whole-panel run, where `chosen` IS the universe.
            rows.extend(r for r in dispatch_anndata_metrics(
                anndata_names, pred_bulks, real_bulks, genes, cfg,
                comparator=comparator,
                pred_moments=pred_moments, real_moments=real_moments,
                pred_bulks_full=pred_bulks_full,
                driver="score_streaming_cell (cell-stream)") if r["perturbation"] in chosen)
            logger.info("cell-streaming pred median UMI/cell = %s", median_umi)

        df = pl.DataFrame(rows, schema=_TIDY_SCHEMA)
        if partial_out:
            from .cache import config_hash

            sid = f"frac{fraction}_idx{index}" if fraction is not None else "all"
            meta_out = {
                "real_ref_fingerprint": real_fp,
                "config_hash": config_hash(cfg.to_dict()),
                "comparator": comparator,
                "metrics": sorted(names),
                "perturbations": sorted(chosen),
                # #246: what the numbers MEAN, not just which metrics were selected.
                PARTIAL_SEMANTICS_KEY: result_semantics(names, comparator=comparator),
            }
            if median_umi is not None:
                meta_out["median_umi_pred"] = median_umi
            write_partial(df, partial_out, subset_id=sid, meta=meta_out)
        return df
    finally:
        pred_store.close()
        if real_store is not None:
            real_store.close()


def _score_streaming_cell_de(pred_store, real_store, *, cfg, de_names, chosen):
    """Build pred/real gpudge DE tables off the cell archives, restrict to ``chosen``,
    assemble, and dispatch the DE metrics. Mirrors :func:`_score_streaming_de`: the real
    side ranks vs its own reference pool (Mode 1); the v2 pred side
    (``control_source='real'``) ranks vs the REAL control pool (Mode 2). Both drive
    ``compute_de_streaming_cell`` (gpudge refpool core)."""
    meta = cell_source.cell_metadata(real_store)
    genes, n_genes = meta.var_names, int(meta.n_vars)

    def de_table(store, ref_X, store_meta=None):
        if store_meta is None:
            store_meta = cell_source.cell_metadata(store)
        targets = [p for p in np.sort(store_meta.perts) if p != cfg.control]
        return compute_de_streaming_cell(
            ref_X=ref_X,
            group_iter_factory=lambda: cell_source.iter_cell_groups(
                store, targets, gather_threads=cfg.gather_threads),
            targets=targets, var_names=genes, n_genes=n_genes,
            backend=cfg.de.backend, mean_calc=cfg.de.mean_calc, epsilon=cfg.de.epsilon,
            target_sum=cfg.target_sum, clip_value=cfg.de.clip_value,
            fdr_scope=cfg.de.fdr_scope,
            filter_gene_min_cpm_cell=cfg.filter.filter_gene_min_cpm_cell,
        )

    # Read real's control pool ONCE: it is real's Mode-1 reference AND (v2,
    # control_source='real') the pred side's Mode-2 external pool. refpool_de_core only
    # READS ref_X, so sharing the object across both calls is safe and avoids decoding the
    # (large) control pool twice; reuse the already-loaded real `meta` for real's de_table
    # (Gemini PR #127 plan review).
    real_ref = cell_source.cell_reference(real_store, gather_threads=cfg.gather_threads)
    real_de = de_table(real_store, real_ref, store_meta=meta)   # Mode 1
    if cfg.control_source == "real":
        pred_ref = real_ref                                     # real's control pool (Mode 2)
    else:
        pred_ref = cell_source.cell_reference(
            pred_store, gather_threads=cfg.gather_threads)       # pred's own reference (Mode 1)
    pred_de = de_table(pred_store, pred_ref)

    chosen_l = list(chosen)
    # Resolve BEFORE slicing: the gate is a property of the DATASET (spec 2.7b). Resolving
    # after the filter would leave a single-target piece with only that target's universe
    # as its "global" index -- exactly the universe an H1_CGS-shaped dataset drops the gene
    # from -- so the piece would raise while the same data scored whole passes.
    resolution = resolve_target_genes(
        real_de, sorted(real_de["target"].unique().to_list()),
        target_gene_map=cfg.target_gene_map,
    )
    real_de = real_de.filter(pl.col("target").is_in(chosen_l))
    pred_de = pred_de.filter(pl.col("target").is_in(chosen_l))
    prepared = prepare_de(
        pred_de, real_de, control=cfg.control, sort_by=cfg.de.sort_by,
        p_adj_threshold=cfg.de.p_adj_threshold, nan_lfc_policy=cfg.de.nan_lfc_policy,
        min_abs_log2fc=cfg.de.min_abs_log2fc, target_resolution=resolution,
    )
    return dispatch_de_metrics(de_names, prepared, cfg)
