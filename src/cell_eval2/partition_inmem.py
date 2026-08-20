"""In-memory partitioned scoring: build a portable cached real reference once, then
score disjoint pred pieces against it (never concatenating the pieces).

This module has three entry points: ``build_reference`` computes, once, every artifact
a pred piece needs -- the real pseudobulk per needed normalization, the full real DE
table, the real control cells (as a portable ``.shad`` or ``.h5ad`` file), and a
``reference.json`` manifest tying them together with a content fingerprint + config
hash. ``build_reference_streaming`` builds the same bundle out-of-core, streaming
perturbation-complete batches of a single real ``.h5ad`` context instead of
materializing the whole real matrix (used by ``h5ad_manifest.score_h5ad_manifest``).
``score_piece`` scores one in-memory pred piece (perturbed cells only, no
controls) against that cached bundle -- never loading the full real matrix -- and
optionally writes a partial for later aggregation (``partition.aggregate_partials``).

Config requires ``fdr_scope="per_pert"`` and a gpudge-resolvable DE backend; both
``version in {"v1","v2"}`` and ``control_source in {"real","pred"}`` are supported
(SP2) -- see ``_require_partition_config``. Under ``control_source="pred"`` the pred
side is scored against a pred-control reference (``build_pred_control_reference``):
the pred DE uses the pred's own control cells as the gpudge external-reference pool
and each piece's pred pseudobulk is augmented with the pred-control row. Any other
config raises ``NotImplementedError``; use ``compute_metrics`` (whole-prediction)
or the streaming path (``scale.py``) instead.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterator
from dataclasses import replace
from typing import TYPE_CHECKING, Protocol

import numpy as np
import polars as pl

if TYPE_CHECKING:
    import anndata as ad

from .cache import config_hash, fingerprint_adata
from . import norm as _norm
from .catalog import CATALOG, resolve_metrics
from .config import EvalConfig
from .de import prepare_de, resolve_target_genes
from .de_compute import _resolve_backend, compute_de
from .io import load_anndata, validate_gene_axis
from .norm import resolve_target_sum
from .partition import PARTIAL_SEMANTICS_KEY, result_semantics, write_partial
from .run import (_close_backed, _effective_input_type, _materialize, _needed_normalizations,
                  _reject_moments_metrics, _resolve_config, _side_bulks,
                  dispatch_anndata_metrics, dispatch_de_metrics)
from .stream import is_shad, shad_fingerprint

_CONTROL_FORMATS = ("shad", "h5ad")
_TIDY_SCHEMA = {"perturbation": pl.Utf8, "metric": pl.Utf8, "value": pl.Float64}


class PertBatchSource(Protocol):
    """A perturbation-batch source for the streaming reference builders. Implemented by
    ``h5ad_manifest.H5adBatchSource`` (h5ad) and ``rowstore.RowStoreBatchSource`` (row store),
    decoupling ``_build_reference_streaming_core`` / ``_build_pred_control_reference_core`` from
    any on-disk format.

    - ``control``: the per-context control label (overrides ``cfg.control`` in the core).
    - ``stream_tag``: a deterministic string identifying the backing file, used only in the
      real-reference bundle fingerprint (each context builds into a fresh temp dir, so this is a
      within-run cache key, not a cross-run guarantee).
    - ``read_control_block()``: in-memory ``AnnData`` of the control rows.
    - ``iter_pert_batches(mem_budget)``: perturbation-complete, budget-sized non-control batches.
    """
    control: str
    stream_tag: str
    def read_control_block(self) -> "ad.AnnData": ...  # noqa: E704
    def iter_pert_batches(self, mem_budget) -> "Iterator[tuple[list[str], ad.AnnData]]": ...  # noqa: E704


def _require_partition_config(cfg: EvalConfig) -> EvalConfig:
    """Validate ``cfg`` is partition-scoring-eligible; return it with the AUC floor
    normalized so partitioned ``pr_auc``/``roc_auc`` are partition-invariant.

    Eligible iff: ``de.fdr_scope="per_pert"`` (a "global" FDR pool cannot be split
    across independently-scored pieces) and the resolved DE backend is gpudge (the only
    backend with an in-memory external-reference DE mode -- gpudge_arc#67).
    ``version in {"v1","v2"}`` and ``control_source in {"real","pred"}`` are both
    supported (SP2). Any other config raises ``NotImplementedError`` pointing at
    ``compute_metrics`` (whole-prediction) or the streaming path (``scale.py``) instead.

    The returned cfg has the AUC floor normalized so partitioned ``pr_auc``/``roc_auc``
    stay partition-invariant: ``clip`` and ``replace_zero`` are elementwise (each piece's
    floor equals the whole-prediction floor) and pass through unchanged; ``min_nonzero``
    is NOT partition-invariant and is normalized to ``replace_zero@1e-10``.
    """
    if cfg.de.fdr_scope != "per_pert":
        raise NotImplementedError(
            f"partitioned in-memory scoring requires de.fdr_scope='per_pert' "
            f"(got {cfg.de.fdr_scope!r}); a 'global' FDR pool cannot be split across "
            "independently-scored pieces -- use compute_metrics (whole-prediction) instead"
        )
    try:
        resolved = _resolve_backend(cfg.de.backend)
    except Exception as e:
        raise NotImplementedError(
            f"partitioned in-memory scoring requires a gpudge-resolvable DE backend; "
            f"resolving de.backend={cfg.de.backend!r} raised {e!r}. Use compute_metrics "
            "(whole-prediction) or the streaming path (scale.py) instead."
        ) from e
    if resolved != "gpudge":
        raise NotImplementedError(
            f"partitioned in-memory scoring requires the gpudge DE backend (in-memory "
            f"external-reference DE); de.backend={cfg.de.backend!r} resolved "
            f"to {resolved!r}. Use compute_metrics (whole-prediction) or the streaming path "
            "(scale.py) instead."
        )
    # Partition-invariant AUC floor: 'clip' and 'replace_zero' are elementwise (each piece's
    # floor equals the whole-prediction floor), so they pass through unchanged. 'min_nonzero'
    # is NOT partition-invariant (each piece sees a different nonzero-p-value set), so it is
    # normalized to replace_zero@1e-10 -- exactly the pre-SP2 forced behavior (PR #83 review).
    if cfg.de.auc_pval_floor == "min_nonzero":
        return replace(cfg, de=replace(cfg.de, auc_pval_floor="replace_zero",
                                       auc_pval_floor_value=1e-10))
    return cfg


#: #181. The manifest key holding the semantic subset below.
BUNDLE_SEMANTICS_KEY = "semantic_fields"

#: The space `pred_control.h5ad` is in, written by `_build_pred_control_reference_core`. Under
#: `control_source="pred"` that artifact is the one `score_piece` hands `compute_de`, so the real
#: side's `effective_input_type` is the wrong thing to check the piece against (codex-review r3).
PRED_CONTROL_TYPE_KEY = "pred_control_effective_input_type"

#: #181: the DELIBERATE subset of config fields a consumer is verified against, beyond the
#: `normalize_target_sum` / `bulk_target_sum` / `comparator` checks that already exist. A blanket
#: `config_hash` comparison is NOT the answer and was rejected in #155's spec (§4.3): `h5ad_manifest.py`,
#: `rowstore.py` and `cellstream.py` all legitimately `_replace` `control` and/or
#: `input_type` between building a bundle and consuming it, because each context is an independent
#: scoring unit with its own control label. So the check has to be over a named subset.
#:
#: IN, because each one changes what the cached artifacts mean:
#:   de.mean_calc      arithmetic vs geometric group means -> a different real DE table
#:   de.epsilon        the LFC pseudocount -> a different real DE table
#:   de.clip_value     the LFC clip -> a different real DE table
#:   control_source    'real' vs 'pred' decides which control pool the pred DE is run against,
#:                     and whether pred_control.* artifacts are even required
#:   device            the RESOLVED device (auto -> cuda/cpu). fp32 GPU and fp64 CPU pseudobulk
#:                     means differ, and `run._side_bulks` keys its own cache on the resolved device
#:                     for exactly that reason (run.py) -- so a bundle whose pseudobulk was
#:                     accumulated in fp32 and a piece scored in fp64 are not comparable. Missed in
#:                     the first list (codex-review); NOT excluded by #181. Compared
#:                     UNCONDITIONALLY, deliberately -- see _check_bundle_semantics for why the
#:                     narrower "only when it could have mattered" gate was tried and reverted.
#:   filter.filter_gene_min_cpm_cell
#:                     the CPM gate. MISSED in the first version of this list (codex-review) and
#:                     the most consequential omission: it is passed into BOTH the cached real DE
#:                     (partition_inmem.py:616) and every per-piece pred DE (:1000), so a bundle
#:                     built at one cutoff and consumed at another compares two DE tables computed
#:                     over different gene universes. Measured on CCL_2: cutoff 0 INVERTS three of
#:                     four DE metrics against any nonzero cutoff.
#:
#: OUT, and each for its own reason rather than by omission:
#:   control           varies per context BY DESIGN (the three drivers rebind it); already
#:                     recorded separately in the manifest as provenance
#:   input_type        the bundle records `effective_input_type` for the REAL side, and
#:                     `score_piece` binds the PRED side's -- a counts-real / lognorm-pred run is
#:                     a supported path whose two sides legitimately disagree (#155 spec §8). A
#:                     naive equality check here would refuse it. Comparing real-against-real
#:                     needs the consumer to know the real side's type, which `score_piece` does
#:                     not; left as provenance, and #193's mixed-scale question is the right
#:                     place for it rather than this guard.
#:   pert_col          not a semantic of the cached NUMBERS; a wrong one fails loudly at the
#:                     piece's own obs lookup
#:   cache paths       performance only (and already in cache.config_hash's skip set)
#:   metric lists      a piece may legitimately score a subset; `aggregate_partials` owns that
#:                     comparison (#246), and putting it here would refuse a valid partial run
BUNDLE_SEMANTIC_FIELDS = ("de.mean_calc", "de.epsilon", "de.clip_value", "control_source",
                          "filter.filter_gene_min_cpm_cell", "device",
                          # #271: not a config path at all -- see `_bundle_semantics`. A bundle
                          # built before `prep._grouped_sums` began reducing wide holds pseudobulks
                          # rounded the other way, and every field above compares equal across that
                          # change. `_check_bundle_semantics` compares KEY SETS first, so adding it
                          # makes such a bundle fail loudly with "rebuild the bundle" -- which is
                          # the correct answer for a rebuildable cache whose numbers moved.
                          "grouped_sum_reduction_semantics")


def _bundle_semantics(cfg: EvalConfig) -> dict:
    """The recorded values of BUNDLE_SEMANTIC_FIELDS for ``cfg`` (#181).

    ``device`` is the RESOLVED device, not the raw field: "auto" means different things on a GPU and
    a CPU host, and it is the resolved value that decides fp32 vs fp64 accumulation.
    """
    from .run import _cache_device, _GROUPED_SUM_REDUCTION_SEMANTICS
    d = cfg.to_dict()
    out = {}
    for path in BUNDLE_SEMANTIC_FIELDS:
        if path == "device":
            out[path] = _cache_device(cfg)
            continue
        if path == "grouped_sum_reduction_semantics":
            # #271. A CODE-semantics counter, not a config field, which is why it cannot be looked
            # up in `cfg.to_dict()` -- the same reason `device` is special-cased above, one step
            # further: `device` is at least a knob, this is what a group sum MEANS. Unconditional
            # here, unlike in the result cache: this payload gates whether a whole reference bundle
            # may be consumed, and a bundle is a rebuildable cache, so the strict rule costs one
            # rebuild where the scoped one would need the bundle's own comparator in hand.
            out[path] = _GROUPED_SUM_REDUCTION_SEMANTICS
            continue
        node = d
        for part in path.split("."):
            node = node[part]
        out[path] = node
    return out


def _check_bundle_semantics(cache_dir, cfg: EvalConfig, manifest, *, caller: str) -> None:
    """Verify ``cfg`` against the bundle's recorded semantic subset (#181).

    Before this, exactly ONE field was verified -- ``normalize_target_sum`` -- and the same
    argument extended to every other field that changes what the cached artifacts mean, none of
    which was checked. ``aggregate_partials``' cross-partial guard cannot see it either: every
    partial records the same CALLER-derived ``config_hash``, so the guard observes agreement and
    passes while the artifacts sit on incompatible footings.

    A manifest without the key was written before this check existed. Refused rather than skipped,
    matching what ``_apply_bundle_target_sum`` does for a pre-#155 bundle: a reference bundle is a
    rebuildable cache (the three drivers build one per context into a temp dir), so "rebuild it" is
    a real remedy, and silently accepting an unverifiable bundle is the failure this closes.
    """
    recorded = manifest.get(BUNDLE_SEMANTICS_KEY)
    if recorded is None:
        raise ValueError(
            f"reference bundle in {cache_dir!r} has no {BUNDLE_SEMANTICS_KEY!r}; it was built "
            f"before {caller} verified the semantic subset {list(BUNDLE_SEMANTIC_FIELDS)} (#181), "
            "so its artifacts cannot be checked against this config. Rebuild the bundle "
            "(build_reference / build_reference_streaming) before consuming it."
        )
    mine = _bundle_semantics(cfg)
    # ⚠️ `device` is compared UNCONDITIONALLY, and that is a deliberate reversal.
    #
    # Round 3 of codex-review observed -- correctly -- that the device can only move a cached
    # artifact when the GPU accumulator actually ran, so comparing it for a DE-only bundle (no
    # pseudobulk at all) or a non-counts side (CPU path regardless) produced a false rejection. I
    # gated it on `manifest["norms"]` and the real side's type. Round 4 showed that gate opened a
    # WORSE hole: `reference.json["norms"]` describes the REAL pseudobulks, while
    # `_build_pred_control_reference_core` resolves its own selection and writes its own
    # (in that core, not here) -- so the two can differ. With a lognorm real side and a counts pred
    # control, the gate dropped `device` even though the cached pred-control pseudobulk could be
    # CUDA/fp32 while later pieces are CPU/fp64; the TYPES match, so `_check_control_space` passes,
    # and `_augment_pred_control` then stacks numerically incompatible rows.
    #
    # So: back to the conservative comparison. A false rejection is loud and its remedy is one
    # rebuild; a missed guard silently combines fp32 and fp64 artifacts. Doing this properly means
    # recording the pred-control's OWN norms and actual accumulation mode alongside its type and
    # comparing the two artifact sets independently -- reported as a follow-up rather than guessed
    # at here.
    # KEY SETS first. `recorded.get(k) != mine[k]` alone accepted a manifest that OMITTED a field
    # whenever the expected value happened to be None -- and `de.clip_value` IS None under v2, so a
    # manifest missing it compared equal and the field went unverified (codex-review). A recorded
    # key this build does not know is equally a version mismatch.
    if set(recorded) != set(mine):
        raise ValueError(
            f"reference bundle in {cache_dir!r} records semantic fields "
            f"{sorted(recorded)} but {caller} verifies {sorted(mine)} (#181). A field present in "
            "one and absent from the other cannot be compared -- an absent key would silently "
            "match any expected None. Rebuild the bundle with this version."
        )
    bad = {k: (recorded[k], mine[k]) for k in mine if recorded[k] != mine[k]}
    if bad:
        detail = "; ".join(f"{k}: bundle={b!r} vs this run={m!r}" for k, (b, m) in sorted(bad.items()))
        raise ValueError(
            f"reference bundle in {cache_dir!r} was built under different scoring semantics than "
            f"this {caller} call -- {detail}. The cached real DE table and control pool were "
            "produced under the bundle's values, so scoring a prediction against them would put "
            "the two sides on incompatible footings, and aggregate_partials cannot detect it "
            "(every partial records the same caller-derived config_hash). Rebuild the bundle with "
            "this config, or score with the bundle's."
        )


def _pred_control_space(cache_dir, manifest) -> str:
    """The recorded space of ``pred_control.*`` / ``pred_pseudobulk_*``, raising if it is absent.

    ⚠️ The key's PRESENCE is also a generation marker, and that is what makes this a guard rather
    than a lookup. ``_write_reference_bundle`` builds a FRESH manifest dict, so rebuilding the real
    bundle DROPS this key -- while nothing removes the old ``pred_pseudobulk_*.npz`` files. Without
    this check, an anndata-only ``control_source='pred'`` run after such a rebuild would have
    ``_augment_pred_control`` silently subtract a STALE pred control (codex-review round 5, which
    found it as the hole opened by making the type comparison DE-only).
    """
    recorded = manifest.get(PRED_CONTROL_TYPE_KEY)
    if recorded not in ("counts", "lognorm"):
        raise ValueError(
            f"control_source='pred' needs the pred-control reference in {cache_dir!r} to be current, "
            f"but the manifest records {PRED_CONTROL_TYPE_KEY}={recorded!r}. Rebuilding the real "
            "bundle rewrites reference.json and drops this key while leaving the old "
            "pred_pseudobulk_*.npz files in place -- so its absence means those artifacts belong to "
            "an earlier build and would be consumed as if they were current. Re-run "
            "build_pred_control_reference against this bundle."
        )
    return recorded


def _check_control_space(cache_dir, cfg: EvalConfig, manifest, *, piece_eff: str) -> None:
    """Refuse a piece whose space differs from the CONTROL ARTIFACT ``control_source`` selects.

    ⚠️ Which artifact that is depends on ``control_source``, and getting it wrong cuts both ways
    (codex-review round 3). Under ``"pred"`` the bundle hands ``compute_de`` ``pred_control.h5ad``,
    NOT the real control -- so comparing against the real side's ``effective_input_type`` would
    refuse a supported mixed pair whose pred control matches its pieces (``score_cellstream`` can
    produce exactly that, since it resolves both sides independently), while letting a STALE pred
    control in a different space through whenever the real side happened to match.

    The needed key must be PRESENT: a ``.get()`` returning None and skipping the comparison is the
    same fail-open that round 2's semantics loop shipped. A bundle predating the key is refused with
    a rebuild instruction, matching every other guard in this family.

    REFUSED, never converted. The conversion is the right end state but MOVES NUMBERS for every
    mixed partitioned run, so it is a release decision and belongs in its own issue; failing loud
    costs a matched run -- the only kind any in-tree driver produces -- nothing.
    """
    if cfg.control_source == "pred":
        what = "pred-control artifact (control_source='pred')"
        recorded = _pred_control_space(cache_dir, manifest)
    else:
        what = "real control (control_source='real')"
        recorded = manifest.get("effective_input_type")
        if recorded not in ("counts", "lognorm"):
            raise ValueError(
                f"reference bundle in {cache_dir!r} records effective_input_type={recorded!r}, so "
                f"the space of its {what} cannot be checked against this prediction piece (#181). "
                "Rebuild the bundle (build_reference / build_reference_streaming) before consuming "
                "it."
            )
    if recorded != piece_eff:
        raise NotImplementedError(
            f"the bundle's {what} is {recorded!r} but this prediction piece resolves to "
            f"{piece_eff!r}, and score_piece hands ONE input_type to compute_de for both the piece "
            "and that control -- so the control would be normalized as if it were in the "
            "prediction's space. The whole-prediction path (compute_metrics) converts the control "
            "first and handles this pair correctly; use it, or supply both sides in the same space. "
            "(#181/#193; the conversion is not done here because it would move numbers for every "
            "mixed partitioned run.)"
        )


def _bundle_identity_hash(cfg: EvalConfig) -> str:
    """``cache.config_hash`` MINUS ``target_sum`` -- the digest that decides whether a
    ``_RefBundle`` may serve a ``score_piece`` call (#185).

    ``target_sum`` is excluded rather than canonicalized because canonicalization cannot fix the
    case the issue reports. ``score_piece``'s guard ran BEFORE ``_bundle_target_sum``'s
    adopt-or-verify step, so it compared PRE-adoption configs -- config identity, not
    post-adoption equivalence -- and rejected two callers that mean exactly the same thing:

    * a bundle built with ``target_sum=None`` consumed by a call whose cfg already carries the
      manifest's resolved number, and the reverse. No representation change makes ``None`` equal
      to ``54611.0``, so only exclusion covers it.
    * ``1000000`` vs ``1000000.0`` vs ``np.float32(1e6)``, which hash differently.
      ``_apply_bundle_target_sum``'s closing comment exists to stop exactly this diverging
      downstream -- but the hash guard ran first, so the canonicalization never got the chance.

    Nothing is lost: the REAL protection -- a bundle built at one target consumed at another -- is
    ``_apply_bundle_target_sum``, which verifies against ``reference.json``'s
    ``normalize_target_sum``. That is the authoritative record of what the artifacts were
    normalized to, and it already raises naming both values. The hash was a weaker, second-hand
    proxy for it that also produced false rejections.

    ``bulk_target_sum`` is deliberately KEPT in the digest. It has no unresolved form, so it
    cannot hit the None-vs-number case, and ``_apply_bundle_bulk_target_sum`` verifies it only
    when the bundle's comparator is ``bulk_lognorm`` -- so for a ``lognorm`` bundle this digest is
    the only comparison there is. (Inert there, but a check that costs nothing.)
    """
    d = cfg.to_dict()
    d.pop("target_sum", None)
    return config_hash(d)


def _bundle_target_sum(cache_dir, cfg: EvalConfig, *, manifest=None) -> EvalConfig:
    """Bind ``cfg.target_sum`` to the reference bundle's resolved normalization target (#155).

    A consumer must NOT re-derive its own median -- that is exactly what made partitioned
    scores mem_budget-dependent. When the caller supplies a number it is VERIFIED against the
    bundle rather than trusted: a bundle built at one target and consumed at another puts the
    real and pred artifacts on two scales, and ``aggregate_partials``' cross-partial guard
    cannot see it (every partial records the same caller-derived ``config_hash``).
    """
    if manifest is not None:
        # Already read by the caller's _RefBundle (#153); ``cache_dir`` stays only for the error
        # messages. Callers that pass one have necessarily read the file, so the not-a-file
        # branch below cannot apply to them.
        return _apply_bundle_target_sum(cache_dir, cfg, manifest)
    ref_json = os.path.join(cache_dir, "reference.json")
    if not os.path.isfile(ref_json):
        if cfg.target_sum is not None:
            # build_pred_control_reference is a PUBLIC builder used standalone against an empty
            # cache dir today (tests/test_partition_inmem_pred_control.py:37). Requiring a real
            # bundle first would be an unannounced API break; only the None case genuinely needs
            # the manifest, because only it has nothing to resolve from.
            return cfg
        raise FileNotFoundError(
            f"target_sum=None needs the real reference bundle's resolved normalization target, "
            f"but {ref_json!r} does not exist; build the real reference first"
        )
    with open(ref_json, encoding="utf-8") as fh:
        manifest = json.load(fh)
    return _apply_bundle_target_sum(cache_dir, cfg, manifest)


def _apply_bundle_target_sum(cache_dir, cfg: EvalConfig, manifest) -> EvalConfig:
    """The adopt-or-verify rule itself, over an already-read manifest (see _bundle_target_sum)."""
    if "normalize_target_sum" not in manifest:
        raise ValueError(
            f"reference bundle in {cache_dir!r} has no 'normalize_target_sum'; it was built "
            "before #155 and its artifacts may have been normalized per batch. Rebuild the "
            "bundle (build_reference / build_reference_streaming) before consuming it."
        )
    bundle_ts = manifest["normalize_target_sum"]
    if cfg.target_sum is None:
        return replace(cfg, target_sum=bundle_ts)
    if bundle_ts is None or float(cfg.target_sum) != float(bundle_ts):
        raise ValueError(
            f"config target_sum={cfg.target_sum!r} disagrees with the reference bundle's "
            f"normalize_target_sum={bundle_ts!r} in {cache_dir!r}: the real and pred artifacts "
            "would be normalized to two different targets, which aggregate_partials cannot "
            "detect. Rebuild the bundle with this config, or score with the bundle's target."
        )
    # Equal but possibly differently represented (1000000 vs 1000000.0, np.float32 vs float):
    # canonicalize to the manifest's scalar so downstream config_hashes cannot diverge.
    return replace(cfg, target_sum=bundle_ts)


def _apply_bundle_bulk_target_sum(cache_dir, cfg: EvalConfig, manifest) -> EvalConfig:
    """Verify the bulk-lognorm target recorded by a reference bundle.

    ``bulk_target_sum`` has no unresolved/``None`` form, so the target-sum precedent's
    adopt-or-verify policy reduces to verify for this field. Equal values are still
    canonicalized to the manifest scalar so downstream config hashes stay stable.
    """
    if manifest.get("comparator") != "bulk_lognorm":
        return cfg
    if "bulk_target_sum" not in manifest:
        raise ValueError(
            f"reference bundle in {cache_dir!r} has comparator='bulk_lognorm' but no "
            "'bulk_target_sum'; rebuild it before consuming it"
        )
    bundle_ts = manifest["bulk_target_sum"]
    if float(cfg.bulk_target_sum) != float(bundle_ts):
        raise ValueError(
            f"config bulk_target_sum={cfg.bulk_target_sum!r} disagrees with the reference "
            f"bundle's bulk_target_sum={bundle_ts!r} in {cache_dir!r}: the real and pred "
            "artifacts would be normalized to two different bulk-lognorm targets. Rebuild "
            "the bundle with this config, or score with the bundle's target."
        )
    return replace(cfg, bulk_target_sum=bundle_ts)


def _one_sided_comparator(names, comparator, *, cfg: EvalConfig, caller: str,
                          fallback: str, effective_input_type: str | None = None) -> str:
    """Validate an explicit run comparator at a one-sided API boundary.

    A comparator is semantically required only when a selected catalog entry declares
    ``EXPR_COMPARATOR``. Concrete-normalization and DE-only callers remain source compatible;
    for them ``fallback`` is inert because no selected lookup consumes the policy.
    """
    expr_names = [n for n in names if CATALOG[n].normalization == _norm.EXPR_COMPARATOR]
    if comparator is None:
        if expr_names:
            raise ValueError(
                f"{caller} is one-sided and requires comparator= when expression comparator "
                f"metrics are requested: {expr_names}"
            )
        comparator = fallback
    if comparator not in ("bulk_lognorm", "lognorm"):
        raise ValueError(
            f"{caller} comparator must be 'bulk_lognorm' or 'lognorm', got {comparator!r}"
        )
    if comparator == "bulk_lognorm":
        if cfg.version != "v2":
            raise ValueError(
                f"{caller} comparator='bulk_lognorm' requires version='v2', got "
                f"version={cfg.version!r}"
            )
        if effective_input_type is not None and effective_input_type != "counts":
            raise ValueError(
                f"{caller} comparator='bulk_lognorm' requires a counts local side, but its "
                f"effective input type is {effective_input_type!r}"
            )
    return comparator


def _bundle_comparator(cache_dir, names, comparator, *, cfg: EvalConfig, manifest,
                       caller: str, effective_input_type: str | None = None) -> str:
    """Adopt or verify the run comparator recorded by a reference bundle."""
    if "comparator" not in manifest:
        raise ValueError(
            f"reference bundle in {cache_dir!r} has no 'comparator'; rebuild it before "
            f"calling {caller}"
        )
    recorded = manifest["comparator"]
    requested = _one_sided_comparator(
        names, comparator, cfg=cfg, caller=caller, fallback=recorded,
        effective_input_type=effective_input_type,
    )
    if requested != recorded:
        raise ValueError(
            f"{caller} comparator={requested!r} disagrees with reference bundle "
            f"comparator={recorded!r} in {cache_dir!r}"
        )
    return recorded


def _write_reference_bundle(cache_dir, *, cfg, bulks, de_df, control_ad,
                            real_ref_fingerprint, var_index, universe, control_format,
                            comparator):
    """Write the real-reference cache bundle (pseudobulk npz per norm, real DE parquet,
    control cells, reference.json). Shared by build_reference and build_reference_streaming."""
    for norm, (perts, means) in bulks.items():
        np.savez(
            os.path.join(cache_dir, f"real_pseudobulk_{norm}.npz"),
            perts=np.asarray(perts, dtype=str), means=np.asarray(means),
        )
    de_df.write_parquet(os.path.join(cache_dir, "real_de.parquet"))
    if control_format == "h5ad":
        control_ad.write_h5ad(os.path.join(cache_dir, "real_control.h5ad"))
    else:  # "shad"
        from cellstream import write_sharded

        write_sharded(control_ad, os.path.join(cache_dir, "real_control.shad"),
                     group_by=cfg.pert_col, reference=None, overwrite=True)
    manifest = {
        "real_ref_fingerprint": real_ref_fingerprint,
        "config_hash": config_hash(cfg.to_dict()),
        "perturbation_universe": universe,
        "var_index": var_index,
        "norms": list(bulks.keys()),
        "control_format": control_format,
        "control": cfg.control,
        # #155: the resolved normalization target. ALWAYS written (1e6 for v2, the resolved
        # median for a target_sum=None run, null for lognorm) so the bundle records the number
        # actually used and consumers are HANDED it instead of re-deriving a per-piece median.
        # float(): EvalConfig validates with math.isfinite, which accepts an np.float32 that
        # json.dump cannot serialize.
        "normalize_target_sum": (None if cfg.target_sum is None else float(cfg.target_sum)),
        # Bulk-lognorm is group-normalized independently of the per-cell target above. Bind
        # this second scale explicitly so a prediction piece cannot consume a real bulk built
        # in a different lognorm space.
        "bulk_target_sum": float(cfg.bulk_target_sum),
        # #155: the effective input type the bundle's REAL-side artifacts were actually built
        # on, which for a v1 declared-lognorm config over counts is NOT the declared
        # cfg.input_type. PROVENANCE ONLY -- written, never verified: score_piece binds the PRED
        # side's effective type, and a counts-real / lognorm-pred run is a supported path whose
        # two sides legitimately disagree (spec section 8). By this point cfg.input_type has
        # already been rebound to the effective type (Step 3d), so it is the right value to
        # record.
        "effective_input_type": cfg.input_type,
        # Run-scoped policy. Consumers verify this exact value rather than resolving again from
        # an individual prediction piece, whose autodetected type is not a run-level decision.
        "comparator": comparator,
        # #181: the named subset of fields that change what the artifacts MEAN, so a consumer
        # scoring against them can be refused instead of producing artifacts on incompatible
        # footings. See BUNDLE_SEMANTIC_FIELDS for why each one is in, and which are out.
        BUNDLE_SEMANTICS_KEY: _bundle_semantics(cfg),
    }
    with open(os.path.join(cache_dir, "reference.json"), "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2, sort_keys=True)
    return manifest


def build_reference(real, *, config, cache_dir, control_format="shad",
                    comparator=None) -> dict:
    """Compute + write the portable real-reference bundle to ``cache_dir``.

    Writes ``real_pseudobulk_<norm>.npz`` (one per normalization the configured
    metrics need), ``real_de.parquet`` (the full real DE table, every non-control
    perturbation vs ``cfg.control``), ``real_control.{shad|h5ad}`` (the real control
    cells, portable input for a later gpudge external-reference DE), and
    ``reference.json`` (the manifest tying them together). Returns the manifest dict.

    ``real`` may be an in-memory AnnData, an ``.h5ad`` path, or a packed ``.shad``
    path (detected via ``stream.is_shad``); the ``.shad`` path is read via
    ``ShardedArchive`` and its identity comes from ``shad_fingerprint`` rather than
    ``fingerprint_adata`` (streaming-safe, matches the ``.shad``-archive convention
    used elsewhere in the repo, e.g. ``scale.score_streaming``).
    """
    if control_format not in _CONTROL_FORMATS:
        raise ValueError(f"control_format must be one of {_CONTROL_FORMATS}, got {control_format!r}")
    cfg = _require_partition_config(_resolve_config(config, {}))
    os.makedirs(cache_dir, exist_ok=True)

    real_is_shad = isinstance(real, (str, os.PathLike)) and is_shad(real)
    if real_is_shad:
        from cellstream.read import ShardedArchive

        real_ref_fingerprint = shad_fingerprint(real)
        real_ad = ShardedArchive(real).to_anndata()
    else:
        real_ad = load_anndata(real, backed=isinstance(real, (str, os.PathLike)))
        real_ref_fingerprint = fingerprint_adata(real_ad, pert_col=cfg.pert_col, strict=cfg.cache_strict)

    try:
        # Materialize exactly once; reused below for pseudobulk (_side_bulks), the real DE
        # table (compute_de), and the control-cell slice -- none of the three mutate it in
        # place (compute_de/_to_linear always .copy() before transforming).
        real_mat = _materialize(real_ad)

        # #155: resolve target_sum=None ONCE, against the real control pool, before either the
        # pseudobulk or the DE table is computed -- so both are built on one target and the
        # number can be persisted for score_piece. Moving the control slice above compute_de
        # also makes a missing control fail before the DE work, not after it. EFFECTIVE input
        # type, not cfg.input_type (v1 allows a declared lognorm over real counts).
        control_ad = real_mat[real_mat.obs[cfg.pert_col].astype(str) == cfg.control].copy()
        if control_ad.n_obs == 0:
            raise ValueError(
                f"control {cfg.control!r} not present in real.obs[{cfg.pert_col!r}]; "
                "cannot build a reference bundle without control cells"
            )
        # Rebind BOTH fields to the effective type + resolved target, so the resolution, the
        # pseudobulk (_side_bulks, which already uses the effective type) and the DE below all
        # read the same decision. Without the input_type rebind, a v1 config declaring lognorm
        # over real counts resolves a median and then compute_de expm1's raw counts, because
        # partition_inmem passes cfg.input_type verbatim (:193, :289, :497).
        eff = _effective_input_type(control_ad, cfg, side="real")
        cfg = replace(cfg, input_type=eff,
                      target_sum=resolve_target_sum(control_ad, input_type=eff,
                                                    target_sum=cfg.target_sum))

        names, _missing = resolve_metrics(cfg.metrics, version=cfg.version)
        comparator = _one_sided_comparator(
            names, comparator, cfg=cfg, caller="build_reference", fallback="lognorm",
            effective_input_type=eff,
        )
        norms = _needed_normalizations(names, comparator=comparator)
        # effective_input_type=eff (Step 3a): _side_bulks does NOT read cfg.input_type, so the
        # rebind above reaches compute_de but not the pseudobulk without this.
        bulks = _side_bulks(real_mat, fp=None, store=None, norms=norms, cfg=cfg, side="real",
                            effective_input_type=eff)

        de_df = compute_de(
            real_mat, backend=cfg.de.backend, groupby=cfg.pert_col, reference=cfg.control,
            mean_calc=cfg.de.mean_calc, epsilon=cfg.de.epsilon, input_type=cfg.input_type,
            target_sum=cfg.target_sum, clip_value=cfg.de.clip_value, fdr_scope=cfg.de.fdr_scope,
            filter_gene_min_cpm_cell=cfg.filter.filter_gene_min_cpm_cell, threads=cfg.num_threads,
        )

        universe = sorted(set(real_mat.obs[cfg.pert_col].astype(str)) - {cfg.control})
        var_index = np.asarray(real_mat.var.index.values, dtype=str).tolist()
        return _write_reference_bundle(
            cache_dir, cfg=cfg, bulks=bulks, de_df=de_df, control_ad=control_ad,
            real_ref_fingerprint=real_ref_fingerprint, var_index=var_index,
            universe=universe, control_format=control_format, comparator=comparator)
    finally:
        if not real_is_shad:  # ShardedArchive-backed real_ad has no file handle of ours to close
            _close_backed(real_ad, real)


def build_reference_streaming(real_h5ad, *, config, cache_dir, control, mem_budget,
                              input_type=None, comparator=None) -> dict:
    """Out-of-core real reference for a single manifest ``.h5ad`` context. Thin wrapper over
    ``_build_reference_streaming_core`` with an ``H5adBatchSource`` -- unchanged public API used
    by ``h5ad_manifest.score_h5ad_manifest`` and the SP2 tests. ``control`` is the per-context label."""
    from .h5ad_manifest import H5adBatchSource

    cfg = _require_partition_config(_resolve_config(config, {}))
    source = H5adBatchSource(real_h5ad, pert_col=cfg.pert_col, control=control)
    return _build_reference_streaming_core(
        source, config=config, cache_dir=cache_dir, mem_budget=mem_budget,
        input_type=input_type, comparator=comparator)


def _build_reference_streaming_core(source: PertBatchSource, *, config, cache_dir, mem_budget,
                                    input_type=None, native_gpu_normalize=False,
                                    comparator=None) -> dict:
    """Out-of-core real reference for one context from a ``PertBatchSource``. Streams
    perturbation-complete batches (never materializing the whole real matrix), computing
    per-batch pseudobulk + per-batch DE (vs the resident control), concatenated into the same
    bundle ``build_reference`` writes. ``source.control`` is the per-context control label.

    Note: the written ``real_ref_fingerprint`` covers only the resident control block plus
    ``source.stream_tag`` (via ``fingerprint_adata`` + a ``|stream:<tag>`` suffix), not the whole
    real matrix -- cheap because it never materializes here. Safe because each context gets its
    own freshly-built bundle in a fresh temp dir (``score_h5ad_manifest`` /
    ``rowstore.score_rowstore`` create ``ref_dir``/``parts_dir`` per context) read only by
    that context's ``score_piece`` calls, so there is no cross-context cache reuse to guard."""
    from dataclasses import replace as _replace

    from .cache import fingerprint_adata
    from .norm import resolve_target_sum
    from .run import (_check_scale_limit_once, _effective_input_type, _needed_normalizations,
                      _side_bulks, _val_allow_fractional, _validate_input_once)

    cfg = _require_partition_config(_resolve_config(config, {}))
    cfg = _replace(cfg, control=source.control)     # per-context control from the source
    if input_type is not None:
        cfg = _replace(cfg, input_type=input_type)
    os.makedirs(cache_dir, exist_ok=True)

    def _validate_stream_slice(adata):
        # _side_bulks validates only path/backed sources; these materialized in-memory slices
        # bypass it, so validate here to mirror _side_bulks's path-input behavior (and the
        # up-front validation compute_metrics does). Gating matches _side_bulks exactly.
        if not cfg.validate_input:
            return
        eff = cfg.input_type          # #155: the resolved decision, not a per-slice autodetect
        if cfg.version != "v1":
            _validate_input_once(adata, eff, allow_fractional=_val_allow_fractional(cfg, side="real"))
        _check_scale_limit_once(adata, eff, cfg.max_counts_per_cell)

    control_ad = source.read_control_block()

    # #155: resolve BEFORE the batch loop -- and before any validation -- so the effective type,
    # every batch's pseudobulk, and every per-batch DE use ONE decision. Computed from the
    # control block that is already resident, so it costs no extra pass; fixed before blocking,
    # so it cannot depend on mem_budget. Rebind input_type too: the per-batch compute_de at :289
    # passes cfg.input_type verbatim. For v2 / autodetect=False this rebind is an IDENTITY
    # (_effective_input_type returns the declared type), so callers that pass input_type=
    # explicitly -- `score_h5ad_manifest`, `score_cellstream` and `score_rowstore`, each into
    # the reference builders -- keep their value; only v1 (which autodetects unconditionally)
    # is corrected, which is the point.
    _eff = _effective_input_type(control_ad, cfg, side="real")
    cfg = _replace(cfg, input_type=_eff,
                   target_sum=resolve_target_sum(control_ad, input_type=_eff,
                                                 target_sum=cfg.target_sum))

    names, _missing = resolve_metrics(cfg.metrics, version=cfg.version)
    comparator = _one_sided_comparator(
        names, comparator, cfg=cfg, caller="build_reference_streaming", fallback="lognorm",
        effective_input_type=_eff,
    )
    norms = _needed_normalizations(names, comparator=comparator)

    _validate_stream_slice(control_ad)

    # pseudobulk: control row + each batch's rows (batches are perturbation-complete -> concat)
    acc = {n: {"perts": [], "means": []} for n in norms}
    ctrl_bulks = _side_bulks(control_ad, fp=None, store=None, norms=norms, cfg=cfg, side="real",
                             effective_input_type=_eff)
    for n in norms:
        cp, cm = ctrl_bulks[n]
        acc[n]["perts"].append(np.asarray(cp, dtype=str))
        acc[n]["means"].append(np.asarray(cm))

    de_frames = []
    universe = []
    for batch_perts, batch_ad in source.iter_pert_batches(mem_budget):
        universe += list(batch_perts)
        _validate_stream_slice(batch_ad)
        b_bulks = _side_bulks(batch_ad, fp=None, store=None, norms=norms, cfg=cfg, side="real",
                              effective_input_type=_eff)
        for n in norms:
            bp, bm = b_bulks[n]
            acc[n]["perts"].append(np.asarray(bp, dtype=str))
            acc[n]["means"].append(np.asarray(bm))
        de_frames.append(compute_de(
            batch_ad, backend=cfg.de.backend, groupby=cfg.pert_col, reference=control_ad,
            control_group=cfg.control, mean_calc=cfg.de.mean_calc, epsilon=cfg.de.epsilon,
            input_type=cfg.input_type, target_sum=cfg.target_sum, clip_value=cfg.de.clip_value,
            fdr_scope=cfg.de.fdr_scope, filter_gene_min_cpm_cell=cfg.filter.filter_gene_min_cpm_cell,
            threads=cfg.num_threads, native_gpu_normalize=native_gpu_normalize,
        ))

    bulks = {}
    for n in norms:
        perts = np.concatenate(acc[n]["perts"])
        means = np.vstack(acc[n]["means"])
        order = np.argsort(perts, kind="stable")
        bulks[n] = (perts[order], means[order])
    de_df = pl.concat(de_frames, how="vertical")

    real_ref_fingerprint = fingerprint_adata(control_ad, pert_col=cfg.pert_col,
                                             strict=cfg.cache_strict) + f"|stream:{source.stream_tag}"
    var_index = np.asarray(control_ad.var.index.values, dtype=str).tolist()
    return _write_reference_bundle(
        cache_dir, cfg=cfg, bulks=bulks, de_df=de_df, control_ad=control_ad,
        real_ref_fingerprint=real_ref_fingerprint, var_index=var_index,
        universe=sorted(universe), control_format="h5ad", comparator=comparator)


def build_pred_control_reference(pred_h5ad, *, config, cache_dir, control,
                                 input_type=None, comparator=None) -> None:
    """Write the pred-control artifacts for ``control_source='pred'`` from a manifest ``.h5ad``.
    Thin wrapper over ``_build_pred_control_reference_core`` with an ``H5adBatchSource`` --
    unchanged public API used by ``h5ad_manifest.score_h5ad_manifest`` and the SP2 tests."""
    from .h5ad_manifest import H5adBatchSource

    cfg = _require_partition_config(_resolve_config(config, {}))
    source = H5adBatchSource(pred_h5ad, pert_col=cfg.pert_col, control=control)
    _build_pred_control_reference_core(
        source, config=config, cache_dir=cache_dir, input_type=input_type,
        comparator=comparator)


def _build_pred_control_reference_core(source: PertBatchSource, *, config, cache_dir,
                                       input_type=None, comparator=None) -> None:
    """Write the pred-control artifacts needed by ``score_piece`` under ``control_source='pred'``:
    ``pred_control.h5ad`` (the pred side's own control cells, a standalone AnnData used as the
    gpudge external-reference pool for pred DE) and ``pred_pseudobulk_<norm>.npz`` (a single
    control-row pseudobulk per normalization, injected into each piece's ``pred_bulks`` so the
    delta/discrimination metrics find the pred control). Mirrors the real-control machinery in
    ``_build_reference_streaming_core`` but reads only the (small) control block -- no whole-pred
    streaming needed. Written into the SAME ``cache_dir`` as the real reference bundle.

    ``source.control``/``input_type`` set cfg per-context (as ``_build_reference_streaming_core``)."""
    from dataclasses import replace as _replace

    from .run import (_check_scale_limit_once, _effective_input_type, _needed_normalizations,
                      _side_bulks, _val_allow_fractional, _validate_input_once)

    cfg = _require_partition_config(_resolve_config(config, {}))
    cfg = _replace(cfg, control=source.control)     # per-context control from the source
    if input_type is not None:
        cfg = _replace(cfg, input_type=input_type)
    # #155: use the number the REAL bundle resolved rather than resolving a second one from the
    # pred control -- a second target would reintroduce the split this change removes. This core
    # always runs after the real bundle exists, into the same cache_dir.
    cfg = _bundle_target_sum(cache_dir, cfg)
    os.makedirs(cache_dir, exist_ok=True)

    names, _missing = resolve_metrics(cfg.metrics, version=cfg.version)
    ref_json = os.path.join(cache_dir, "reference.json")
    manifest = None
    if os.path.isfile(ref_json):
        with open(ref_json, encoding="utf-8") as fh:
            manifest = json.load(fh)

    pred_ctrl = source.read_control_block()
    pred_eff = _effective_input_type(pred_ctrl, cfg, side="pred")
    cfg = _replace(cfg, input_type=pred_eff)
    if manifest is not None:
        comparator = _bundle_comparator(
            cache_dir, names, comparator, cfg=cfg, manifest=manifest,
            caller="build_pred_control_reference", effective_input_type=pred_eff,
        )
        cfg = _apply_bundle_bulk_target_sum(cache_dir, cfg, manifest)
        # #181. Only inside this branch: the `else` below is the standalone numeric-target
        # builder that runs against an EMPTY cache dir with no real bundle to verify against
        # (tests/test_partition_inmem_pred_control.py:37), and requiring a manifest there would be
        # the unannounced API break _bundle_target_sum already declines to make.
        _check_bundle_semantics(cache_dir, cfg, manifest,
                                caller="build_pred_control_reference")
    else:
        # Preserve the established standalone numeric-target builder. With no real bundle there
        # is nothing to verify, so an expression-comparator selection requires the explicit value.
        comparator = _one_sided_comparator(
            names, comparator, cfg=cfg, caller="build_pred_control_reference",
            fallback="lognorm", effective_input_type=pred_eff,
        )
    norms = _needed_normalizations(names, comparator=comparator)
    if cfg.validate_input:
        if cfg.version != "v1":
            _validate_input_once(pred_ctrl, pred_eff,
                                 allow_fractional=_val_allow_fractional(cfg, side="pred"))
        _check_scale_limit_once(pred_ctrl, pred_eff, cfg.max_counts_per_cell)

    ctrl_bulks = _side_bulks(pred_ctrl, fp=None, store=None, norms=norms, cfg=cfg, side="pred",
                             effective_input_type=pred_eff)
    for norm, (perts, means) in ctrl_bulks.items():
        np.savez(
            os.path.join(cache_dir, f"pred_pseudobulk_{norm}.npz"),
            perts=np.asarray(perts, dtype=str), means=np.asarray(means),
        )
    pred_ctrl.write_h5ad(os.path.join(cache_dir, "pred_control.h5ad"))
    # Record the space THIS artifact is in. Under control_source='pred' it -- not the real control --
    # is what score_piece hands compute_de, so the real side's `effective_input_type` says nothing
    # about it (codex-review round 3). Written into the existing manifest rather than a second file
    # so there is one record of what the bundle contains.
    ref_json = os.path.join(cache_dir, "reference.json")
    if os.path.isfile(ref_json):
        with open(ref_json, encoding="utf-8") as fh:
            man = json.load(fh)
        man[PRED_CONTROL_TYPE_KEY] = pred_eff
        with open(ref_json, "w", encoding="utf-8") as fh:
            json.dump(man, fh, indent=2, sort_keys=True)


def _augment_pred_control(pred_bulks, bundle, norms, *, control):
    """Prepend the cached pred-control pseudobulk row to each norm's (perts, means) so the
    delta/discrimination metrics can find the pred control under control_source='pred'."""
    cached = bundle.pred_control_bulks(norms)
    out = {}
    for norm in norms:
        cperts, cmeans = cached[norm]
        pperts, pmeans = pred_bulks[norm]
        pperts = np.asarray(pperts, dtype=str)
        if control in set(pperts):          # defensive: never double-add a control row
            out[norm] = (pperts, pmeans)
            continue
        out[norm] = (np.concatenate([cperts, pperts]),
                     np.vstack([np.asarray(cmeans), np.asarray(pmeans)]))
    return out


class _RefBundle:
    """The reference-bundle artifacts in ``cache_dir``, read at most once and reused across every
    ``score_piece`` call for one context.

    Every member is lazy: a config that dispatches no DE metrics never reads ``real_de.parquet``,
    and a ``control_source='real'`` run never looks for ``pred_control.h5ad``. Laziness is also what
    keeps the missing-artifact errors where they were -- raised from inside ``score_piece``, not
    early from a constructor.

    Safe to hold across a whole batch loop because nothing writes ``cache_dir`` during one: the
    reference builders finish before it, and ``score_piece`` writes its partials to a separate
    ``partial_out`` directory. The control AnnData it hands out is SHARED across pieces, so no
    consumer may mutate it -- ``compute_de`` either passes the reference straight through
    (``native_gpu_normalize`` + counts) or normalizes onto a fresh copy via ``_to_linear``.
    """

    def __init__(self, cache_dir, cfg):
        self.cache_dir = str(cache_dir)
        self._cfg = _require_partition_config(_resolve_config(cfg, {}))
        # #185: the bundle-IDENTITY digest, which deliberately omits `target_sum` -- see
        # _bundle_identity_hash. The name is kept because `score_piece`'s guard and
        # test_partition_inmem_refbundle both reference it.
        self.config_hash = _bundle_identity_hash(self._cfg)
        self._manifest = None
        self._real_bulks = {}
        self._real_de = None
        self._control_ad = None
        self._pred_control_bulks = {}

    @property
    def manifest(self):
        if self._manifest is None:
            with open(os.path.join(self.cache_dir, "reference.json"), encoding="utf-8") as fh:
                self._manifest = json.load(fh)
        return self._manifest

    def real_bulks(self, norms):
        for norm in norms:
            if norm not in self._real_bulks:
                with np.load(os.path.join(self.cache_dir, f"real_pseudobulk_{norm}.npz")) as z:
                    self._real_bulks[norm] = (z["perts"], z["means"])
        return {norm: self._real_bulks[norm] for norm in norms}

    @property
    def real_de(self):
        if self._real_de is None:
            self._real_de = pl.read_parquet(os.path.join(self.cache_dir, "real_de.parquet"))
        return self._real_de

    def target_resolution(self, cfg):
        """Dataset-level target-gene resolution, computed once from the UNSLICED real_de.

        Memoized on the bundle so every piece is handed the same resolution rather than
        re-deriving one from its own slice (spec 2.7b). h5ad_manifest, rowstore and cellstream
        reach the metrics through this bundle and so inherit it.
        """
        cached = getattr(self, "_target_resolution_cache", None)
        if cached is None:
            cached = resolve_target_genes(
                self.real_de, sorted(self.real_de["target"].unique().to_list()),
                target_gene_map=cfg.target_gene_map,
            )
            self._target_resolution_cache = cached
        return cached

    @property
    def control_ad(self):
        if self._control_ad is None:
            self._control_ad = self._load_control()
        return self._control_ad

    def _load_control(self):
        if self._cfg.control_source == "pred":
            # pred DE vs the PRED control pool (build_pred_control_reference), NOT the real control
            # -- mirrors run._pred_de_input's within-realm pred DE for control_source='pred'.
            path = os.path.join(self.cache_dir, "pred_control.h5ad")
            if not os.path.isfile(path):
                raise FileNotFoundError(
                    f"control_source='pred' needs a pred-control reference in {self.cache_dir!r}; "
                    "call build_pred_control_reference before score_piece"
                )
            return load_anndata(path)
        if self.manifest["control_format"] == "shad":
            from cellstream.read import ShardedArchive

            return ShardedArchive(os.path.join(self.cache_dir, "real_control.shad")).to_anndata()
        return load_anndata(os.path.join(self.cache_dir, "real_control.h5ad"))

    def pred_control_bulks(self, norms):
        for norm in norms:
            if norm in self._pred_control_bulks:
                continue
            path = os.path.join(self.cache_dir, f"pred_pseudobulk_{norm}.npz")
            if not os.path.isfile(path):
                raise FileNotFoundError(
                    f"control_source='pred' needs {path!r}; call build_pred_control_reference "
                    "before score_piece"
                )
            with np.load(path) as z:  # copy/detach the (tiny) control row from the closing NpzFile
                self._pred_control_bulks[norm] = (np.asarray(z["perts"], dtype=str),
                                                  z["means"].copy())
        return {norm: self._pred_control_bulks[norm] for norm in norms}


def score_piece(pred_piece, cache_dir, *, config, piece_id=None, partial_out=None,
                native_gpu_normalize=False, bundle=None, comparator=None) -> pl.DataFrame:
    """Score one in-memory pred piece (perturbed cells only, NO controls) against the
    cached reference bundle written by ``build_reference``, mirroring
    ``scale.score_streaming``'s dispatch pattern but reading cached real artifacts
    (pseudobulk ``.npz``, the real DE ``.parquet``, and the real control cells) instead
    of streaming a ``.shad`` archive.

    ``pred_piece`` may be an in-memory AnnData or an ``.h5ad`` path. Pred DE uses
    gpudge's in-memory external-reference mode (gpudge_arc#67): the piece's own cells
    are the DE targets and the cached real control pool (materialized from
    ``real_control.{shad|h5ad}``) is the SEPARATE reference -- no concatenation.
    ``de_wilcoxon_nsig_spearman`` is deliberately excluded from the dispatched DE
    metrics (``aggregate_partials`` reconstructs it from the per-piece nsig-count
    metrics once every piece is in).

    Under ``control_source="pred"`` the pred side is scored within-realm against a
    pred-control reference (``build_pred_control_reference``): pred DE uses the pred's
    own control cells (``pred_control.h5ad``) as the external-reference pool, and each
    piece's ``pred_bulks`` is augmented with the cached pred-control pseudobulk row so
    the delta/discrimination metrics find the pred control. Under ``"real"`` the pred
    side is scored against the cached real control (unchanged).

    Raises ``ValueError`` if the piece contains control cells (``cfg.control`` in
    ``obs[cfg.pert_col]``) or perturbations outside ``reference.json``'s
    ``perturbation_universe``. When ``partial_out`` is set, writes
    ``{piece_id}.parquet`` + ``{piece_id}.json`` (meta with ``perturbations``);
    ``piece_id`` defaults to the pred file stem for a path input and otherwise must
    be given explicitly (an in-memory AnnData has no file stem -- PR #83 review).

    ``bundle``: a ``_RefBundle`` for ``cache_dir``, so a driver scoring many pieces against one
    reference reads each cached artifact once instead of once per piece (#153). Omitted -> one is
    built for this call, which is exactly the previous behavior. A bundle built for a different
    ``cache_dir``, or one whose numerics digest differs, is a ``ValueError`` rather than a
    silent stale read. That digest (``_bundle_identity_hash``) excludes ``metrics``
    (``cache.config_hash``) and, since #185, ``target_sum`` -- so it is a numerics digest, not
    exact config identity, and ``target_sum`` is verified against the bundle's own manifest
    instead of by hash equality.
    """
    from dataclasses import replace as _replace

    from .run import _effective_input_type

    cfg = _require_partition_config(_resolve_config(config, {}))
    if bundle is None:
        bundle = _RefBundle(cache_dir, cfg)
    else:
        if os.path.realpath(bundle.cache_dir) != os.path.realpath(cache_dir):
            raise ValueError(
                f"bundle was built for cache_dir {bundle.cache_dir!r}, not {str(cache_dir)!r}"
            )
        if bundle.config_hash != _bundle_identity_hash(cfg):
            raise ValueError(
                "bundle was built for a different config than this score_piece call; build one "
                "bundle per (cache_dir, config). (target_sum is excluded from this comparison "
                "and is verified against the bundle manifest instead -- #185.)"
            )
    manifest = bundle.manifest
    names, _missing = resolve_metrics(cfg.metrics, version=cfg.version)
    comparator = _bundle_comparator(
        cache_dir, names, comparator, cfg=cfg, manifest=manifest, caller="score_piece",
    )
    # #155: adopt or verify the bundle's target, against reference.json's
    # `normalize_target_sum`. #185 removed `target_sum` from the identity guard above, so the
    # ORDER of these two steps no longer decides anything: the guard is now blind to the field
    # this step resolves, and this step compares against the manifest rather than against a
    # hash. Before that it was load-bearing and wrong -- the guard ran first, on PRE-adoption
    # configs, and rejected a bundle built at `target_sum=None` consumed by a call carrying the
    # manifest's own resolved number (and the reverse).
    # Hand it the manifest the bundle already holds so a per-piece driver does not re-read
    # reference.json once per batch, which is the whole point of the bundle (#153).
    cfg = _bundle_target_sum(cache_dir, cfg, manifest=manifest)
    cfg = _apply_bundle_bulk_target_sum(cache_dir, cfg, manifest)
    # #181: and every OTHER field that changes what the cached artifacts mean. Verified, never
    # adopted -- unlike the two targets above, there is no unresolved form to fill in, and
    # silently adopting a bundle's de.epsilon would mean scoring under a policy the caller did
    # not ask for.
    _check_bundle_semantics(cache_dir, cfg, manifest, caller="score_piece")

    piece_ad = load_anndata(pred_piece, backed=isinstance(pred_piece, (str, os.PathLike)))
    try:
        pert_values = piece_ad.obs[cfg.pert_col].astype(str)
        if cfg.control in set(pert_values):
            raise ValueError(
                f"pred piece must not contain control cells (found control {cfg.control!r} "
                f"in obs[{cfg.pert_col!r}]); score_piece scores perturbed cells only"
            )
        piece_perts = sorted(set(pert_values))
        universe = set(manifest["perturbation_universe"])
        unknown = set(piece_perts) - universe
        if unknown:
            raise ValueError(
                f"pred piece contains perturbation(s) outside the reference universe: "
                f"{sorted(unknown)}"
            )

        # The reference bundle's var_index IS the gene axis every cached artifact was built on,
        # and the pseudobulk below indexes into it positionally -- so a piece whose genes are a
        # permutation of the reference's would be scored gene-POSITION-wise and silently return
        # plausible numbers. Covers score_cellstream and score_h5ad_manifest, which both reach
        # this engine; score_rowstore shares one gene axis between sides by construction --
        # both `RowStoreBatchSource`s load the same `RowStoreArtifact.var_names_path` -- so
        # this can never fire there (ultrareview 2026-07-25).
        validate_gene_axis(piece_ad.var.index.values, manifest["var_index"])

        anndata_names = [n for n in names if CATALOG[n].kind == "anndata"]
        de_names = [n for n in names if CATALOG[n].kind == "de"]

        # Materialize exactly once; reused below for the pred pseudobulk (_side_bulks) and
        # the pred DE table (compute_de) -- neither mutates it in place (compute_de's
        # _to_linear/gpudge path always .copy()/normalizes onto a fresh matrix).
        piece_mat = _materialize(piece_ad)
        # #155: score_piece is a SEPARATE function from the two reference builders, so their
        # cfg.input_type rebind does not reach it -- and :497 below passes cfg.input_type
        # verbatim to compute_de while _side_bulks (:468) autodetects piece_mat independently.
        # A v1 config declaring lognorm over raw counts therefore had DE expm1 those counts
        # while the pseudobulk treated them as counts. Bind once, here, from the SAME matrix
        # both consumers use, so they cannot disagree. Anchored per piece, not per run: under
        # control_source='real' there is no pred control block and no pred bundle to anchor on
        # (spec section 8 records that residual).
        piece_eff = _effective_input_type(piece_mat, cfg, side="pred")
        # ⚠️ #181/#193's seam. codex-review round 2 found it is not merely unverified but ACTIVELY
        # WRONG: `piece_eff` is the PREDICTION's effective type, and it is handed to compute_de
        # together with the bundle's cached CONTROL -- so on a mixed pair the control gets the
        # prediction's space applied to it (expm1 on integer counts). `compute_metrics` does NOT
        # have this bug: it converts the control into the pred's scale FIRST (run.py:819-824)
        # precisely so compute_de normalizes both sides identically. #181's own text flags the
        # manifest field as "written, never verified".
        #
        # Checked HERE, before any metric dispatch or bundle-artifact read, so a missing or corrupt
        # artifact cannot replace the incompatibility error with its own (codex-review round 3).
        # ...but ONLY when a DE metric is selected. `compute_de` is the only consumer of a control
        # AnnData, so an anndata-only (or empty) selection never touches one -- and refusing such a
        # run for a control's space would be a false rejection whose message describes an operation
        # that will not happen (codex-review round 4). The pred-control PSEUDOBULK that the anndata
        # path does consume under control_source='pred' is a different artifact with its own space,
        # and it is guarded by the semantic-subset check below rather than by this one.
        if cfg.control_source == "pred":
            # Unconditional on metric kind: `_augment_pred_control` consumes pred_pseudobulk_* on the
            # ANNDATA path too, and a real-bundle rebuild orphans those files while dropping the key
            # that identifies them. Type EQUALITY stays DE-only below -- only compute_de consumes a
            # control AnnData -- but currency must be established either way (codex-review round 5).
            _pred_control_space(cache_dir, manifest)
        if de_names:
            _check_control_space(cache_dir, cfg, manifest, piece_eff=piece_eff)
        cfg = _replace(cfg, input_type=piece_eff)
        _one_sided_comparator(
            names, comparator, cfg=cfg, caller="score_piece", fallback=comparator,
            effective_input_type=piece_eff,
        )

        rows: list[dict] = []

        if anndata_names:
            # The partitioned driver's REAL side comes from a persisted _RefBundle whose
            # on-disk format carries no moments, so a moments metric cannot be served here.
            # Fail loud rather than fall back to the biased value (#198 §4.3).
            _reject_moments_metrics(anndata_names, driver="score_piece (partitioned in-memory)")
            norms = _needed_normalizations(anndata_names, comparator=comparator)
            genes = np.asarray(manifest["var_index"], dtype=str)
            real_bulks = bundle.real_bulks(norms)
            pred_bulks = _side_bulks(piece_mat, fp=None, store=None, norms=norms, cfg=cfg,
                                     side="pred", effective_input_type=piece_eff)
            if cfg.control_source == "pred":
                # A control-free piece has no pred control row; the delta/discrimination metrics
                # look up cfg.control INSIDE pred_bulks under control_source='pred'. Prepend the
                # cached pred-control pseudobulk row (build_pred_control_reference) per norm.
                pred_bulks = _augment_pred_control(pred_bulks, bundle, norms, control=cfg.control)
            rows.extend(dispatch_anndata_metrics(
                anndata_names, pred_bulks, real_bulks, genes, cfg,
                comparator=comparator,
            ))

        if de_names:
            real_de = bundle.real_de
            control_ad = bundle.control_ad

            pred_de = compute_de(
                piece_mat, backend=cfg.de.backend, groupby=cfg.pert_col, reference=control_ad,
                control_group=cfg.control, mean_calc=cfg.de.mean_calc, epsilon=cfg.de.epsilon,
                input_type=cfg.input_type, target_sum=cfg.target_sum, clip_value=cfg.de.clip_value,
                fdr_scope=cfg.de.fdr_scope, filter_gene_min_cpm_cell=cfg.filter.filter_gene_min_cpm_cell,
                threads=cfg.num_threads, native_gpu_normalize=native_gpu_normalize,
            )
            real_de = real_de.filter(pl.col("target").is_in(piece_perts))
            pred_de = pred_de.filter(pl.col("target").is_in(piece_perts))
            prepared = prepare_de(
                pred_de, real_de, control=cfg.control, sort_by=cfg.de.sort_by,
                p_adj_threshold=cfg.de.p_adj_threshold, nan_lfc_policy=cfg.de.nan_lfc_policy,
                min_abs_log2fc=cfg.de.min_abs_log2fc,
                target_resolution=bundle.target_resolution(cfg),
            )
            # Excluded here (not merely un-dispatched): aggregate_partials reconstructs it
            # from de_wilcoxon_nsig_counts_{real,pred} once every piece has been scored.
            de_names_scored = [n for n in de_names if n != "de_wilcoxon_nsig_spearman"]
            rows.extend(dispatch_de_metrics(de_names_scored, prepared, cfg))

        # Real-side-only diagnostics dispatch against the full reference panel, and PDS must
        # continue to rank against that full panel. Narrow the emitted rows, not real_bulks,
        # so every partial contains exactly its piece's perturbations without changing ranks.
        df = pl.DataFrame(rows, schema=_TIDY_SCHEMA).filter(
            pl.col("perturbation").is_in(piece_perts)
        )
        if partial_out:
            if piece_id is None:
                if isinstance(pred_piece, (str, os.PathLike)):
                    piece_id = os.path.splitext(os.path.basename(str(pred_piece)))[0]
                else:
                    raise ValueError(
                        "score_piece needs an explicit piece_id for in-memory pred pieces"
                    )
            write_partial(df, partial_out, subset_id=piece_id, meta={
                "real_ref_fingerprint": manifest["real_ref_fingerprint"],
                "config_hash": config_hash(cfg.to_dict()),
                "comparator": comparator,
                "metrics": sorted(names),
                "perturbations": sorted(piece_perts),
                # #246: what the numbers MEAN, not just which metrics were selected.
                PARTIAL_SEMANTICS_KEY: result_semantics(names, comparator=comparator),
            })
        return df
    finally:
        _close_backed(piece_ad, pred_piece)
