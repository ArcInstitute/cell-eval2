from __future__ import annotations

import math
import os
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from typing import Literal

import yaml

from .moments import DEFAULT_BULK_TARGET_SUM

# Single source of truth for the runtime-enforced Literal value sets. Each is reused by
# the matching __post_init__ validator so the annotation and the check cannot drift.
_VERSIONS = ("v1", "v2")
_DEVICES = ("auto", "cuda", "cpu")
_CONTROL_SOURCES = ("pred", "real")
_INPUT_TYPES = ("counts", "lognorm")
_DISTANCES = ("l1", "l2", "cosine")
_RANK_DENOMINATORS = ("n", "n-1")
_TIE_POLICIES = ("midrank", "position")
_EXCLUSION_SCOPES = ("row", "panel")
_DE_SORT_KEYS = ("abs_log2_fold_change", "log2_fold_change", "p_value", "p_adj")
_DE_METHODS = ("wilcoxon", "deseq2")
_DE_BACKENDS = ("auto", "gpudge", "pdex", "scanpy", "deseq2")
_MEAN_CALCS = ("arithmetic", "geometric")
_NAN_LFC_POLICIES = ("keep", "mask")
_FDR_SCOPES = ("global", "per_pert")
_AUC_PVAL_FLOORS = ("clip", "replace_zero", "min_nonzero")


@dataclass
class FilterParams:
    filter_gene_min_cpm_cell: float | None = 5.0   # v2 default; v1 overrides to None

    def __post_init__(self) -> None:
        # A negative threshold = keep-all (de_compute._apply_cpm_filter's threshold<0 branch), so any
        # finite value is allowed; only NaN/inf would silently corrupt the gate (F1.2).
        v = self.filter_gene_min_cpm_cell
        if v is not None and not math.isfinite(v):
            raise ValueError(f"filter_gene_min_cpm_cell must be a finite float or None, got {v!r}")


@dataclass
class DiscriminationParams:
    """Per-metric parameters for the discrimination score (PDS).

    `distance` records the preset's canonical distance and round-trips in YAML,
    but the named catalog variants (pds_l1/pds_l2/pds_cosine) bind
    their distance via functools.partial, so the dispatcher does NOT read this
    field. The other fields ARE consumed at dispatch time.
    """

    distance: Literal["l1", "l2", "cosine"] = "cosine"      # corrected default
    rank_denominator: Literal["n", "n-1"] = "n-1"           # corrected default
    exclude_target_gene: bool = True
    embed_key: str | None = None                            # obsm-space PDS (deferred)
    # How EQUIDISTANT competitors share a rank (issue #282). "midrank" (v2) gives every
    # member of a tied block the average of the positions the block spans, so an
    # all-tied row scores exactly 0.5 -- the no-information point -- for every target.
    # "position" (v1) is the legacy argsort-position rule, which resolves a tie to the
    # target's index in the SORTED perturbation array, i.e. its alphabetical position;
    # it is retained only for upstream cell-eval parity. See `discrimination_score`.
    tie_policy: Literal["midrank", "position"] = "midrank"   # corrected default
    # WHICH target genes `exclude_target_gene` removes, and from where (issue #343).
    # "panel" (v2) drops EVERY perturbation's target gene from the ranked feature space
    # once, up front, so every cell of the distance matrix is computed on one fixed set of
    # genes. "row" is the legacy rule: for prediction row i, only perturbation i's own
    # target gene is dropped -- from that row's comparison against EVERY reference
    # perturbation, so perturbation j's own knockdown stays visible in cell (i, j). That
    # asymmetry is a scoreable channel; see `discrimination_score`. Retained for upstream
    # cell-eval parity only, which is why v1 and cell-eval-0.7.6 pin it.
    # Inert when `exclude_target_gene` is False -- nothing is excluded under either scope.
    exclusion_scope: Literal["row", "panel"] = "panel"       # corrected default

    def __post_init__(self) -> None:
        if self.distance not in _DISTANCES:
            raise ValueError(f"distance must be one of {_DISTANCES}, got {self.distance!r}")
        if self.rank_denominator not in _RANK_DENOMINATORS:
            raise ValueError(
                f"rank_denominator must be one of {_RANK_DENOMINATORS}, got {self.rank_denominator!r}"
            )
        if self.tie_policy not in _TIE_POLICIES:
            raise ValueError(
                f"tie_policy must be one of {_TIE_POLICIES}, got {self.tie_policy!r}"
            )
        if self.exclusion_scope not in _EXCLUSION_SCOPES:
            raise ValueError(
                f"exclusion_scope must be one of {_EXCLUSION_SCOPES}, got {self.exclusion_scope!r}"
            )


@dataclass(frozen=True)
class DEParams:
    """Per-metric parameters for the DE-table metrics (overlap/precision, …)."""

    p_adj_threshold: float = 0.05  # significance gate, applied to the `p_adj` column
    min_abs_log2fc: float = 0.0  # post-FDR effect-size floor on |log2_fold_change|; 0.0 = off
    sort_by: Literal[
        "abs_log2_fold_change", "log2_fold_change", "p_value", "p_adj"
    ] = "abs_log2_fold_change"
    method: Literal["wilcoxon", "deseq2"] = "wilcoxon"  # provenance (DE test that produced the table)
    nan_lfc_policy: Literal["keep", "mask"] = "mask"  # v1=keep (cell-eval), v2=mask (force p_adj=1)
    backend: Literal["auto", "gpudge", "pdex", "scanpy", "deseq2"] = "auto"  # availability choice; NOT version-scoped
    replicate_col: str | None = None  # obs column defining pseudobulk replicates (deseq2 backend only)
    mean_calc: Literal["arithmetic", "geometric"] = "arithmetic"    # version-scoped (= v2)
    epsilon: float = 1e-9                                           # version-scoped (= v2)
    clip_value: float | None = None  # v1=20.0: pdex zero-mean fold_change clip; None=epsilon/inf (v2)
    fdr_scope: Literal["global", "per_pert"] = "per_pert"  # v1=global (pdex 0.1.27 pools all perts)
    auc_pval_floor: Literal["clip", "replace_zero", "min_nonzero"] = "min_nonzero"  # v1=replace_zero (cell-eval)
    auc_pval_floor_value: float = 1e-10  # floor for clip & replace_zero; ignored by min_nonzero

    def __post_init__(self) -> None:
        if self.sort_by not in _DE_SORT_KEYS:
            raise ValueError(f"sort_by must be one of {_DE_SORT_KEYS}, got {self.sort_by!r}")
        if not math.isfinite(self.min_abs_log2fc) or self.min_abs_log2fc < 0:
            raise ValueError(
                f"min_abs_log2fc must be a finite float >= 0, got {self.min_abs_log2fc!r}"
            )
        if self.method not in _DE_METHODS:
            raise ValueError(f"method must be one of {_DE_METHODS}, got {self.method!r}")
        if self.nan_lfc_policy not in _NAN_LFC_POLICIES:
            raise ValueError(
                f"nan_lfc_policy must be one of {_NAN_LFC_POLICIES}, got {self.nan_lfc_policy!r}"
            )
        if self.backend not in _DE_BACKENDS:
            raise ValueError(f"backend must be one of {_DE_BACKENDS}, got {self.backend!r}")
        # ``method`` is DERIVED provenance: it names the statistical DE test the backend runs,
        # so it always tracks ``backend`` (deseq2 -> "deseq2"; auto/rank -> "wilcoxon"). This is
        # what the DE cache keys (``de_<method>_{table,rank}``, ``stream_de_<method>``) and the
        # run_params stamp key on — leaving it at the bare "wilcoxon" default would let a deseq2
        # rank/table collide with a Wilcoxon one in a shared non-strict cache (the rank cache's
        # params carry no backend). Backend is authoritative, so an inconsistent explicit method
        # is canonicalized here rather than trusted. DEParams is frozen so this can't drift after
        # construction — post-construction mutation would bypass the derivation and reintroduce
        # the collision; ``dataclasses.replace`` (how every internal update is done) reruns it.
        object.__setattr__(self, "method", "deseq2" if self.backend == "deseq2" else "wilcoxon")
        if self.replicate_col is not None and not isinstance(self.replicate_col, str):
            raise ValueError(
                f"replicate_col must be a string or None, got {self.replicate_col!r}"
            )
        if self.mean_calc not in _MEAN_CALCS:
            raise ValueError(f"mean_calc must be one of {_MEAN_CALCS}, got {self.mean_calc!r}")
        if not math.isfinite(self.epsilon) or self.epsilon < 0:
            raise ValueError(f"epsilon must be a finite float >= 0, got {self.epsilon!r}")
        if self.clip_value is not None and (not math.isfinite(self.clip_value) or self.clip_value <= 1):
            raise ValueError(f"clip_value must be a finite float > 1 or None, got {self.clip_value!r}")
        if not math.isfinite(self.p_adj_threshold) or not 0.0 <= self.p_adj_threshold <= 1.0:
            raise ValueError(
                f"p_adj_threshold must be a finite float in [0, 1], got {self.p_adj_threshold!r}"
            )
        if self.fdr_scope not in _FDR_SCOPES:
            raise ValueError(f"fdr_scope must be one of {_FDR_SCOPES}, got {self.fdr_scope!r}")
        if self.auc_pval_floor not in _AUC_PVAL_FLOORS:
            raise ValueError(
                f"auc_pval_floor must be one of {_AUC_PVAL_FLOORS}, got {self.auc_pval_floor!r}"
            )
        if not (0.0 < self.auc_pval_floor_value <= 1.0):
            raise ValueError(
                f"auc_pval_floor_value must be in (0, 1], got {self.auc_pval_floor_value!r}"
            )


# Single source of truth for what each version's conventions are. Values are plain
# (not dataclass instances) so each preset build gets fresh nested dataclasses.
# Only fields that DIFFER between v1 and v2 are listed; fields identical across
# versions (e.g. exclude_target_gene, embed_key) are intentionally omitted and stay
# at their dataclass defaults — this omission is load-bearing for EvalConfig() == v2().
# ⚠️ This table, NOT configs/*.yaml, is what `EvalConfig.v1()` / `.v2()` build from, and
# `compat.MetricsEvaluator` goes through `EvalConfig.v1()`. A version-differing field added
# to the yaml presets alone therefore leaves v1 on the v2 dataclass default and silently
# breaks upstream parity — which is exactly what `exclusion_scope` (#343) would have done.
_VERSION_CONVENTIONS: dict[str, dict] = {
    "v1": {
        "control_source": "pred",
        "input_type": "lognorm",
        "target_sum": None,
        # cell-eval 0.6.6 has no scale-limit gate; v1 (cell-eval parity) relaxes it to 1e9 so
        # log-norm predictions whose expm1 pseudo-counts are large are not rejected. v2 keeps
        # the protective 1e6 default. log1p(1e9)=20.72 leaves a wide expm1-overflow margin.
        "max_counts_per_cell": 1_000_000_000.0,
        # tie_policy "position" is the upstream cell-eval argsort convention; v1 exists to
        # reproduce it byte-for-byte, so the #282 correction is deliberately NOT applied here.
        # exclusion_scope "row" is the upstream per-row target-gene exclusion; v2 removes the
        # whole panel (#343), so v1 must pin the legacy scope for the same byte-parity reason
        # as tie_policy above.
        "discrimination": {"distance": "l1", "rank_denominator": "n", "tie_policy": "position",
                           "exclusion_scope": "row"},
        "de": {
            "nan_lfc_policy": "keep",
            "mean_calc": "geometric",
            "epsilon": 0.0,
            "clip_value": 20.0,
            "fdr_scope": "global",
            "auc_pval_floor": "replace_zero",
        },
        "filter": {"filter_gene_min_cpm_cell": None},
    },
    "v2": {
        "control_source": "real",
        "input_type": "counts",
        "target_sum": 1_000_000.0,
        "max_counts_per_cell": 1_000_000.0,   # v2 keeps the protective scale-limit gate
        "discrimination": {"distance": "cosine", "rank_denominator": "n-1",
                           "tie_policy": "midrank"},
        "de": {"nan_lfc_policy": "mask", "mean_calc": "arithmetic", "epsilon": 1e-9},
        "filter": {"filter_gene_min_cpm_cell": 5.0},
    },
}


@dataclass
class EvalConfig:
    metrics: str | list[str] = "full"
    pert_col: str = "target"
    control: str = "non-targeting"
    version: Literal["v1", "v2"] = "v2"   # drives output naming + stamps provenance
    control_source: Literal["pred", "real"] = "real"        # corrected default
    input_type: Literal["counts", "lognorm"] = "counts"   # v2 default; v1 overrides to lognorm
    max_counts_per_cell: float = 1_000_000.0
    target_sum: float | None = 1_000_000.0  # normalize_total target; v1 overrides to None (median)
    allow_discrete: bool = False  # v1: skip the lognorm guess and treat raw counts as-is (cell-eval parity)
    # permit a fractional 'counts' predictor (e.g. a null/avg baseline). ⚠️ DIGEST-EXEMPT:
    # `config_digest` cannot see this flag, so `score --baseline-agg` pairing will not report a
    # mismatch on it (`--real-bundle` DOES -- `anchor._SEMANTIC_FIELDS` compares it). RULED
    # keep-the-exemption + warn; `run._warn_fractional_allowance_used` carries the measurement.
    allow_fractional_counts: bool = False
    autodetect_input_type: bool = False  # opt-in: cell-eval counts-vs-lognorm auto-detect regardless of version (likely deprecated)
    validate_input: bool = True  # opt-out: skip input-type + scale-limit validation entirely (trusted/benchmark runs)
    filter: FilterParams = field(default_factory=FilterParams)
    num_threads: int = -1
    device: Literal["auto", "cuda", "cpu"] = "auto"  # auto -> cuda iff cupy+GPU, else cpu
    pert_chunk: int = 512  # GPU discrimination: pred-effect rows streamed per distance block
    outdir: str | None = None
    cache_real: str | None = None
    cache_pred: str | None = None
    cache_strict: bool = False
    discrimination: DiscriminationParams = field(default_factory=DiscriminationParams)
    de: DEParams = field(default_factory=DEParams)
    # Decode threads for the cell-layout gather path (cellstream's n_threads, #149). -1 = auto:
    # the cap is the process's CPU-affinity allowance; a positive value is used as the cap
    # verbatim. Either way the value is row-count aware -- see _threads.resolve_gather_threads.
    # Deliberately SEPARATE from num_threads (which governs the gpudge DE backend) so decode
    # parallelism can be throttled under concurrent jobs without changing DE threading.
    # APPENDED LAST on purpose: EvalConfig is a plain (not kw_only) dataclass, so inserting a
    # field beside num_threads would shift device/pert_chunk/outdir for positional callers.
    gather_threads: int = -1
    # Explicit {target: feature} override for target-gene resolution (spec 2.1/2.7a).
    # Authoritative where supplied and NOT re-checked against the feature index, so it is
    # also the escape hatch for a single-target dataset whose own gene is absent.
    # On EvalConfig rather than the metric functions for two reasons: (1) to_dict() is
    # asdict and config_hash keeps every non-skipped field, so it enters the result-cache
    # digest automatically -- as a function argument two runs with different maps would
    # collide on one cache key; (2) it must be fixed for the whole dataset, since
    # resolution is computed once at PreparedDE construction (spec 2.7b).
    # APPENDED LAST on purpose -- see the gather_threads comment above.
    target_gene_map: dict[str, str] | None = None
    # The expression comparator's target sum (issue #264). NOT `target_sum`: that one is
    # per-cell and drives input normalization and the whole DE family; this one is per-GROUP
    # and drives `bulk_lognorm` alone, which is why a run carries both. A field rather than a
    # constant so it enters config_hash and can be swept by tools/metricval, and so a later
    # edit cannot move every published number with no digest change.
    #
    # 5e4 as of #268 (Alex, 2026-08-11), down from the 1e6 spec 3.2 originally fixed. The
    # sweep found 1e6 the ONLY value on 2e3..1e6 where the metric breaks: the jackknife's
    # bias jumps to 2.06% (0.19-0.32% strictly below 1e5, 0.42% AT it), only 25.1% of the
    # denominator survives its own correction so every error is amplified ~4x (47.3% at 5e4),
    # the split-half ceiling goes NEGATIVE on 6 of 6 lines (-0.0900, against +0.0401 at 5e4),
    # and the "predict the control" anchor -- which
    # `scales.py` ships as a hard base=1.0 -- drifts to 1.0727, against 1.0249 at 5e4. 1e6
    # did not even win on the
    # axis spec 3.2 chose it for: effective gene count peaks at 3e5 and is lower at 1e6
    # (8,485) than at 1e5 (8,856). Re-verified at v0.10.0: the ceiling is positive on 6/6
    # at 5e4. ⚠️ This is a v2-path knob in practice -- `norm.resolve_comparator` pins v1 to
    # the `lognorm` comparator, which never reads it.
    # APPENDED LAST on purpose -- see the gather_threads comment.
    bulk_target_sum: float = DEFAULT_BULK_TARGET_SUM

    def __post_init__(self) -> None:
        # Coerce the nested params FIRST. A caller may build `EvalConfig(de={...})` directly
        # rather than through `from_dict` -- `run._resolve_config` documents and supports that
        # -- and any cross-field validation below would then read `.backend` off a plain dict.
        # Normalizing here rather than only in the consumers means every field is a real
        # dataclass by the time anything looks at it, whatever route built the config.
        for _field, _cls in (("filter", FilterParams), ("discrimination", DiscriminationParams),
                             ("de", DEParams)):
            _val = getattr(self, _field)
            if isinstance(_val, dict):
                setattr(self, _field, _cls(**_val))
            elif _val is None:
                # An explicit `None` means "use the default", matching `from_dict`'s handling
                # of an explicit YAML null. Coerced rather than guarded at each use: a
                # `de=None` that survives construction just fails further away, and it failed
                # ASYMMETRICALLY -- the v1/deseq2 check reached `.backend` while a v2 config
                # short-circuited past it and looked fine.
                setattr(self, _field, _cls())
        if self.version not in _VERSIONS:
            raise ValueError(f"version must be one of {_VERSIONS}, got {self.version!r}")
        if self.control_source not in _CONTROL_SOURCES:
            raise ValueError(
                f"control_source must be one of {_CONTROL_SOURCES}, got {self.control_source!r}"
            )
        if self.input_type not in _INPUT_TYPES:
            raise ValueError(f"input_type must be one of {_INPUT_TYPES}, got {self.input_type!r}")
        if self.device not in _DEVICES:
            raise ValueError(f"device must be one of {_DEVICES}, got {self.device!r}")
        if self.pert_chunk <= 0:
            raise ValueError(f"pert_chunk must be a positive int, got {self.pert_chunk!r}")
        if self.target_sum is not None and (not math.isfinite(self.target_sum) or self.target_sum <= 0):
            raise ValueError(f"target_sum must be a finite float > 0 or None, got {self.target_sum!r}")
        if (isinstance(self.bulk_target_sum, bool)
                or not math.isfinite(self.bulk_target_sum)
                or self.bulk_target_sum <= 0):
            raise ValueError(
                f"bulk_target_sum must be a positive finite float, got {self.bulk_target_sum!r}"
            )
        if not math.isfinite(self.max_counts_per_cell) or self.max_counts_per_cell <= 0:
            raise ValueError(
                f"max_counts_per_cell must be a finite float > 0, got {self.max_counts_per_cell!r}"
            )
        if (not isinstance(self.num_threads, int) or isinstance(self.num_threads, bool)
                or self.num_threads == 0 or self.num_threads < -1):
            raise ValueError(f"num_threads must be -1 (all cores) or a positive int, got {self.num_threads!r}")
        if (not isinstance(self.gather_threads, int) or isinstance(self.gather_threads, bool)
                or self.gather_threads == 0 or self.gather_threads < -1):
            raise ValueError(
                f"gather_threads must be -1 (auto) or a positive int, got {self.gather_threads!r}"
            )
        # v1 exists to reproduce upstream cell-eval, which has no deseq2 backend, so the
        # combination has no referent. Rejected rather than filtered: `resolve_metrics`
        # gates the de_wilcoxon_* names, but `_effective_de_spec` then relabels the
        # survivors to the `de_deseq2_*` family, every member of which is v2-native --
        # measured, all 21 emitted names under `metrics="de"` were `v1_available=False`.
        # So the version gate was bypassed wholesale, and the alternative (re-filtering
        # after relabeling) would silently emit an almost-empty table instead.
        if self.version == "v1" and self.de.backend == "deseq2":
            raise ValueError(
                "version='v1' is incompatible with de.backend='deseq2': v1 reproduces "
                "upstream cell-eval, which has no deseq2 backend, and every de_deseq2_* "
                "metric is v2-native. Use version='v2' with the deseq2 backend, or keep "
                "version='v1' with a rank backend."
            )
        if self.cache_real is not None and self.cache_pred is not None:
            # realpath resolves symlinks + ./cache-vs-cache + trailing slashes; normcase makes it
            # case-insensitive on Windows/macOS (no-op on POSIX). Best-effort collision guard.
            _cr = os.path.normcase(os.path.realpath(self.cache_real))
            _cp = os.path.normcase(os.path.realpath(self.cache_pred))
            if _cr == _cp:
                raise ValueError(
                    "cache_real and cache_pred must differ "
                    "(fixed artifact names would collide between sides)"
                )

    @classmethod
    def for_version(cls, version: str) -> "EvalConfig":
        """Build a config from a versioned evaluation standard. v1 reproduces
        cell-eval/VCC (predicted control, rank denominator n, l1 PDS, keep NaN-LFC);
        v2 is Arc's improved default (real control, n-1, cosine, mask NaN-LFC).
        Convention fields stay independently overridable; only their defaults come
        from the table. Which PDS distance actually runs is still selected by the
        metric variant name (pds_l1/_l2/_cosine), not the recorded `distance`."""
        if version not in _VERSION_CONVENTIONS:
            raise ValueError(f"unknown version {version!r}; choose 'v1' or 'v2'")
        conv = _VERSION_CONVENTIONS[version]
        return cls(
            version=version,
            control_source=conv["control_source"],
            input_type=conv["input_type"],
            target_sum=conv["target_sum"],
            max_counts_per_cell=conv["max_counts_per_cell"],
            discrimination=DiscriminationParams(**conv["discrimination"]),
            de=DEParams(**conv["de"]),
            filter=FilterParams(**conv["filter"]),
        )

    @classmethod
    def v1(cls) -> "EvalConfig":
        return cls.for_version("v1")

    @classmethod
    def v2(cls) -> "EvalConfig":
        return cls.for_version("v2")

    @classmethod
    def legacy(cls) -> "EvalConfig":
        """Back-compat alias for v1()."""
        return cls.v1()

    @classmethod
    def corrected(cls) -> "EvalConfig":
        """Back-compat alias for v2()."""
        return cls.v2()

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping) -> "EvalConfig":
        if not isinstance(data, Mapping):
            raise ValueError(
                f"config must be a YAML mapping, got {type(data).__name__} "
                "(an empty or comment-only YAML file parses to None)"
            )
        data = dict(data)
        # Build the nested params and pass them THROUGH the constructor. They used to be
        # popped, the config built from what was left, and the params assigned afterwards --
        # which meant `__post_init__` validated a config whose `de`/`filter`/`discrimination`
        # were still the defaults. Any cross-field rule involving them (e.g. v1 vs the deseq2
        # backend) silently did not apply to `from_dict`, `from_yaml`, or the CLI's
        # `--set de.backend=...`, all of which land here.
        for _field, _cls in (("filter", FilterParams), ("discrimination", DiscriminationParams),
                             ("de", DEParams)):
            _val = data.get(_field)
            if isinstance(_val, dict):
                data[_field] = _cls(**_val)
            elif _val is None and _field in data:
                data.pop(_field)          # explicit null in YAML -> take the default
        return cls(**data)

    def to_yaml(self, path: str) -> None:
        with open(path, "w") as fh:
            yaml.safe_dump(self.to_dict(), fh, sort_keys=False)

    @classmethod
    def from_yaml(cls, path: str) -> "EvalConfig":
        with open(path) as fh:
            return cls.from_dict(yaml.safe_load(fh))

    @classmethod
    def from_preset(cls, name: str) -> "EvalConfig":
        """Load a shipped preset from its packaged YAML. Accepts 'v1'/'v2' and the
        back-compat aliases 'legacy'(→v1)/'corrected'(→v2)."""
        # Identity entries (v1→v1, v2→v2) are included so membership doubles as validation.
        alias = {
            "v1": "v1", "v2": "v2", "legacy": "v1", "corrected": "v2",
            "cell-eval-0.7.6": "cell-eval-0.7.6", "cell_eval_0_7_6": "cell-eval-0.7.6",
            "vcc2026": "vcc2026", "vcc-2026": "vcc2026", "vcc_2026": "vcc2026",
        }
        if name not in alias:
            raise ValueError(f"unknown preset {name!r}; choose from {sorted(alias)}")
        from importlib.resources import files

        text = files("cell_eval2").joinpath("configs", f"{alias[name]}.yaml").read_text(encoding="utf-8")
        return cls.from_dict(yaml.safe_load(text))
