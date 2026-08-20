from __future__ import annotations

import logging
import math
import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType

import polars as pl

logger = logging.getLogger(__name__)

REQUIRED_COLS = ("target", "feature", "log2_fold_change", "p_adj")
SORT_KEYS = ("abs_log2_fold_change", "log2_fold_change", "p_value", "p_adj")


@dataclass(frozen=True, init=False)
class TargetResolution:
    """Which targets name a gene this assay measured -- decided ONCE per dataset.

    `mapping` holds ONLY the targets that resolved: target -> the feature label of its
    own gene. A target that did not resolve is absent, and excludes nothing.

    Spec 2.1 separates two questions that a naive "the target must be locatable" rule
    conflates:

    RESOLUTION -- does this target's name correspond to a gene the assay measured? Asked
    against the GLOBAL feature index (the union over all targets of the UNSLICED truth DE
    table). This is what the fail-loud gate checks.

    EXCLUSION -- is that gene among this target's OWN rows, so there is a row to drop?
    That is data, not an error: a target can resolve and still exclude nothing. On H1_CGS
    every target resolves and NONE excludes anything, because `pooled_universe` drops each
    target's own gene from its own universe while leaving it in the assay.

    The resolution is genuinely read-only, not merely documented as such: `frozen=True`
    blocks field REASSIGNMENT but leaves a plain dict mutable, and this object is a field
    of PreparedDE on which the direction memos are identity-keyed (spec 2.7b) -- so a
    post-construction mutation would silently serve stale exclusions, reference stats and
    components with no error anywhere.

    ⚠️ The STORED field is `pairs`, a sorted tuple, and `mapping` is a read-only view over
    it. A `MappingProxyType` stored directly as a field would look right and then break
    `copy.deepcopy` and `dataclasses.asdict` with "cannot pickle 'mappingproxy' object" --
    `asdict(PreparedDE(...))` deep-copies every field. A tuple of string pairs pickles,
    deep-copies, compares and sorts deterministically.

    `n_features` and `unresolved` are recorded at RESOLUTION time and are therefore
    DATASET-GLOBAL. That matters for the error message: a partitioned run would otherwise
    report a global target count against a sliced feature count and a one-target sample,
    which reads as a contradiction.
    """
    pairs: tuple[tuple[str, str], ...]
    n_targets: int
    n_features: int | None
    unresolved: tuple[str, ...]

    def __init__(self, mapping=None, n_targets=0, n_features=None, unresolved=(),
                 *, pairs=None):
        # init=False on the dataclass + an explicit __init__ so callers keep writing
        # TargetResolution({"A": "A"}, 1) while the stored form is the picklable tuple.
        # object.__setattr__ because the dataclass is frozen.
        #
        # ⚠️ `pairs=` must be accepted even though nothing writes it by hand:
        # `dataclasses.replace` reconstructs via cls(**{every field}), so an __init__ that
        # took only `mapping` would raise "unexpected keyword argument 'pairs'". Accepting
        # it also makes the generated repr round-trippable.
        src = pairs if pairs is not None else (mapping or {})
        object.__setattr__(self, "pairs", tuple(sorted(dict(src).items())))
        object.__setattr__(self, "n_targets", int(n_targets))
        # None, not 0: an unrecorded global feature count is UNKNOWN, and reporting it as
        # zero puts a false measurement in the gate's error message.
        object.__setattr__(self, "n_features",
                           None if n_features is None else int(n_features))
        object.__setattr__(self, "unresolved", tuple(unresolved))

    @property
    def mapping(self) -> Mapping[str, str]:
        """Read-only {target: feature} view. Rebuilt per access -- called a handful of
        times per dispatch, so the cost is irrelevant next to the join it guards."""
        return MappingProxyType(dict(self.pairs))

    @property
    def n_resolved(self) -> int:
        return len(self.pairs)


def resolve_target_genes(
    real_df: pl.DataFrame,
    targets: list[str],
    *,
    target_gene_map: dict[str, str] | None = None,
) -> TargetResolution:
    """Resolve each target to its own gene's feature label. Spec 2.1.

    `real_df` MUST be the UNSLICED truth DE table and `targets` the FULL target list.
    Passing a sliced table makes the gate shard-local: a single-target piece has only
    that target's universe as its index, which on an H1_CGS-shaped dataset is exactly the
    universe that drops the gene -- so the piece raises while the same data scored whole
    passes, and whether scoring raises comes to depend on partition size (spec 2.7b).

    Resolution order, per target:
      1. `target_gene_map[target]` if present -- AUTHORITATIVE and deliberately NOT
         re-checked against the index (spec 2.1). Re-checking would make the map useless
         in the one case that needs it: a user mapping a target to its correctly-named but
         deliberately-absent gene. The cost is that a fully-mapped run always passes the
         gate, correct map or not -- an explicit override, like `validate_input=False`.
      2. exact match of the target label against the global feature index.

    ⚠️ **This function NEVER raises.** Resolution and validation are deliberately
    separate. Every DE run constructs a PreparedDE, including runs that select only the
    legacy metrics or run under v1, and those metrics have no target-gene semantics at
    all -- raising here would break them for any dataset whose targets are not also
    features. `tests/test_de.py:105` is exactly that shape (targets A/B, features g1..g4).

    The zero-resolve GATE belongs to the eleven chance-corrected metrics and is enforced
    at their entry, by `metrics.direction._require_resolution`. That placement also closes
    the converse hole: a caller who passes an explicitly EMPTY TargetResolution would
    otherwise bypass a construction-time gate entirely and get plausible numbers at zero
    resolution -- the exact silent-wrong-number the gate exists to prevent.

    A PARTIAL rate is ordinary biology or CPM filtering (`de_compute.py:861`) and is
    logged either way. No regex canonicalization: stripping `-\\d+$` is a CCL naming
    convention and mis-strips a gene legitimately ending in `-2`.
    """
    features = set(real_df["feature"].unique().to_list()) if real_df.height else set()
    mapping: dict[str, str] = {}
    for t in targets:
        if target_gene_map and t in target_gene_map:
            mapping[t] = target_gene_map[t]
        elif t in features:
            mapping[t] = t
    unresolved = tuple(sorted(t for t in targets if t not in mapping))
    if targets and unresolved:
        logger.info(
            "target-gene resolution: %d/%d targets resolved to a measured gene; the rest "
            "exclude nothing (ordinary when a target is not a measured gene, or the CPM "
            "filter dropped its row). Supply EvalConfig.target_gene_map to override.",
            len(mapping), len(targets),
        )
    # n_features and unresolved are captured HERE, where the table is the unsliced one,
    # so the gate's error message stays dataset-global on a partitioned run.
    return TargetResolution(mapping, len(targets), n_features=len(features),
                            unresolved=unresolved[:5])


def on_target_pairs(resolution: TargetResolution) -> pl.DataFrame:
    """The ``(target, feature)`` pairs to remove: each RESOLVED target's own gene.

    Lives HERE rather than in one metrics module because issue #172 gave it a second family
    of callers. It was written for the eleven chance-corrected direction metrics
    (``metrics.direction``); ``de_sig_jaccard`` and ``de_lfc_nmae`` now exclude too, and
    ``metrics.direction`` imports ``metrics.de``, so a helper shared by both cannot live in
    either without a cycle. ``de`` already owns :class:`TargetResolution`, so it owns this.
    """
    mapping = resolution.mapping
    return pl.DataFrame(
        {"target": list(mapping.keys()), "feature": list(mapping.values())},
        schema={"target": pl.String, "feature": pl.String},
    )


def exclude_on_target(df: pl.DataFrame, resolution: TargetResolution) -> pl.DataFrame:
    """Anti-join away each resolved target's own gene. Spec 2.7c.

    NOT ``filter(feature != target)``: that is the raw-label comparison spec 2.1 rejects,
    and it ignores ``target_gene_map`` completely -- the map would be accepted,
    fingerprinted, and then have no effect. A target that did not resolve drops nothing.
    """
    if not resolution.mapping:
        return df
    return df.join(on_target_pairs(resolution), on=["target", "feature"], how="anti")


def require_resolution(perturbations, resolution: TargetResolution) -> None:
    """The zero-resolve gate (spec 2.1/5). Raises when NO target resolved.

    ⚠️ Enforced at the ENTRY OF EACH EXCLUDING METRIC, and NOT in
    :func:`resolve_target_genes` or the :class:`PreparedDE` constructors. Two reasons, and
    both are failure modes rather than preferences:

    (1) Every DE run builds a PreparedDE, including runs selecting only the legacy metrics
    and every v1 run. Those metrics have no target-gene semantics, so a construction-time
    raise would break them for any dataset whose targets are not also features --
    ``tests/test_de.py:105`` is exactly that shape.

    (2) A construction-time gate is bypassable: a caller passing an explicitly EMPTY
    TargetResolution would skip it and get plausible numbers at zero resolution. Checking
    the RESOLUTION rather than the resolving validates the supplied one too.

    This does not re-resolve anything, so spec 2.7b's "pieces apply the resolution they are
    handed and never re-run the gate" still holds: ``n_resolved`` is the DATASET's count,
    computed once before slicing, so a single-target piece carrying it passes.

    ⚠️ Gate on the caller's ``perturbations``, NOT on ``resolution.n_targets``. PreparedDE's
    field defaults to ``TargetResolution({}, 0)``, so an object built without going through
    the constructors -- or one handed ``TargetResolution({}, 0)`` explicitly -- has
    ``n_targets == 0`` and would sail straight through an ``n_targets``-based check with zero
    resolution, which is the exact bypass this gate exists to close. ``perturbations`` is the
    authoritative target list on the object; ``n_targets`` is advisory metadata a caller can
    get wrong.
    """
    if not perturbations or resolution.n_resolved:
        return
    # Dataset-global figures, recorded at resolution time -- NOT recomputed from a sliced
    # real_df, which would make the message report a global target count against a piece's
    # features.
    sample = list(resolution.unresolved) or sorted(perturbations)[:5]
    # n_features is None when the resolution did not record one (e.g. a hand-built
    # TargetResolution). Say so rather than printing 0, which reads as a measurement.
    n_feat = ("an unrecorded number of" if resolution.n_features is None
              else str(resolution.n_features))
    raise ValueError(
        f"no target resolves to a gene in the reference feature index: 0 of "
        f"{resolution.n_targets or len(perturbations)} targets matched any of "
        f"{n_feat} features. This is the "
        f"construct-ID-vs-gene-symbol mismatch (e.g. target 'GENEX-1' vs feature "
        f"'GENEX'), which would otherwise exclude nothing and return a plausible "
        f"wrong number -- the worst failure mode available to these metrics. "
        f"Unresolved sample: {sample}. Supply "
        f"EvalConfig.target_gene_map={{target: feature}} to override; map entries are "
        f"authoritative and are not re-checked against the index."
    )


def load_de_table(src) -> pl.DataFrame:
    """Load a DE table from a polars/pandas DataFrame or a CSV/parquet path."""
    if isinstance(src, pl.DataFrame):
        return src
    try:
        import pandas as pd

        if isinstance(src, pd.DataFrame):
            return pl.from_pandas(src)
    except ImportError:
        pass
    if isinstance(src, (str, os.PathLike)):
        path = str(src)
        path_lower = path.lower()  # tolerate uppercase extensions (.CSV/.PARQUET)
        if path_lower.endswith((".parquet", ".pq")):
            return pl.read_parquet(path)
        if path_lower.endswith((".csv", ".csv.gz")):
            return pl.read_csv(path, schema_overrides={"target": pl.Utf8, "feature": pl.Utf8})
        raise ValueError(f"unsupported DE table extension: {path!r}; use .csv or .parquet")
    raise TypeError(f"unsupported DE table source type: {type(src)!r}")


def normalize_de_schema(df: pl.DataFrame, *, name: str) -> pl.DataFrame:
    """Apply the canonical schema: fdr->p_adj alias, derive abs LFC, check required cols."""
    if "p_adj" not in df.columns and "fdr" in df.columns:
        df = df.rename({"fdr": "p_adj"})
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(
            f"{name} DE table missing required columns {missing}; present: {df.columns}"
        )
    # Diagnostic (spec §3): warn on nulls in ANY required column — otherwise nulls in
    # target/feature/p_adj pass silently and get dropped implicitly downstream. Null
    # *handling* (coverage/joins) is deferred to PR #4; this only surfaces them.
    null_counts = {c: cnt for c in REQUIRED_COLS if (cnt := df[c].null_count())}
    if null_counts:
        logger.warning("%s DE table: nulls in required columns %s", name, null_counts)
    return df.with_columns(
        pl.col("target").cast(pl.Utf8),
        pl.col("feature").cast(pl.Utf8),
        pl.col("log2_fold_change").abs().alias("abs_log2_fold_change"),
    )


def apply_nan_policy(df: pl.DataFrame, *, name: str, nan_lfc_policy: str) -> pl.DataFrame:
    """Version-scoped non-finite handling. 'keep' = leave as-is (v1/cell-eval);
    'mask' = force p_adj=1 on NaN-LFC rows (v2) so they drop out of the significance gate."""
    if nan_lfc_policy not in ("keep", "mask"):
        raise ValueError(f"nan_lfc_policy must be 'keep' or 'mask', got {nan_lfc_policy!r}")
    lfc = pl.col("log2_fold_change")
    n_nan, n_inf, n_null = df.select(
        lfc.is_nan().sum().alias("nan"),
        lfc.is_infinite().sum().alias("inf"),
        lfc.is_null().sum().alias("null"),
    ).row(0)
    if n_nan or n_inf or n_null:
        logger.warning(
            "%s DE table: non-finite log2_fold_change — NaN=%d inf=%d null=%d",
            name, n_nan, n_inf, n_null,
        )
    if nan_lfc_policy == "mask":
        incoherent = df.filter(lfc.is_nan() & (pl.col("p_adj") < 1.0)).height
        if incoherent:
            logger.warning(
                "%s DE table: %d NaN-LFC rows had p_adj<1 (incoherent); masking to p_adj=1",
                name, incoherent,
            )
        df = df.with_columns(
            pl.when(lfc.is_nan()).then(1.0).otherwise(pl.col("p_adj")).alias("p_adj")
        )
    return df


def apply_lfc_floor(df: pl.DataFrame, *, name: str, min_abs_log2fc: float) -> pl.DataFrame:
    """Post-FDR effect-size floor. Force p_adj=1 on rows whose |log2_fold_change| is
    strictly below `min_abs_log2fc`, so they drop out of the significance gate — a
    membership filter layered on the already-finalized p_adj (BH is never re-pooled).
    Mirrors apply_nan_policy's mask. No-op at the default 0.0 (guarded), keeping every
    preset bit-identical. Non-finite/null LFC (NaN, inf, null) all fail the strict `<`, so
    the floor never masks them — they keep the p_adj they arrive with (NaN-LFC rows are
    separately masked to p_adj=1 by nan_lfc_policy='mask'; inf/null are only warned about,
    not masked). Consumes the abs_log2_fold_change column from normalize_de_schema.
    Validates min_abs_log2fc here too (not just in DEParams) so direct callers of
    prepare_de / the standalone metrics cannot silently pass a negative/non-finite floor."""
    if not math.isfinite(min_abs_log2fc) or min_abs_log2fc < 0:
        raise ValueError(f"min_abs_log2fc must be a finite float >= 0, got {min_abs_log2fc!r}")
    if min_abs_log2fc <= 0.0:
        return df
    # The below-count is only for the INFO log; skip the extra full-table scan when INFO is off.
    if logger.isEnabledFor(logging.INFO):
        below = df.filter(
            (pl.col("abs_log2_fold_change") < min_abs_log2fc) & (pl.col("p_adj") < 1.0)
        ).height
        if below:
            logger.info(
                "%s DE table: %d rows below |log2fc| floor %g -> masked to p_adj=1",
                name, below, min_abs_log2fc,
            )
    return df.with_columns(
        # .fill_null(False): a null abs_log2_fold_change makes the `<` comparison null; making
        # it explicitly False keeps null-LFC rows unmasked independent of polars' null-condition
        # semantics (empirically already the otherwise branch, but no longer relied on implicitly).
        pl.when((pl.col("abs_log2_fold_change") < min_abs_log2fc).fill_null(False))
        .then(1.0).otherwise(pl.col("p_adj")).alias("p_adj")
    )


def _rank_matrix(df: pl.DataFrame, *, sort_by: str, p_adj_threshold: float) -> pl.DataFrame:
    """Per-target significance-filtered genes, ordinal-ranked by sort_by, pivoted rank x target.
    Mirrors cell-eval DEResults.get_top_genes (descending for LFC keys). Uses a sentinel
    index name and drops it after sorting, so the result holds ONLY perturbation columns
    (de_overlap treats columns as the perturbation set) and cannot collide with a target
    literally named 'rank'."""
    descending = sort_by in ("log2_fold_change", "abs_log2_fold_change")
    filtered = df.filter(pl.col("p_adj") < p_adj_threshold)
    if filtered.height == 0:
        return pl.DataFrame()
    return (
        filtered.with_columns(
            (pl.struct(sort_by).rank("ordinal", descending=descending).over("target") - 1).alias("__rank__")
        )
        .pivot(index="__rank__", on="target", values="feature")
        .sort("__rank__")
        .drop("__rank__")
    )


@dataclass
class PreparedDE:
    real_rank: pl.DataFrame
    pred_rank: pl.DataFrame
    perturbations: list[str]
    sort_by: str
    p_adj_threshold: float
    real_df: pl.DataFrame
    pred_df: pl.DataFrame
    # Appended last (plain, non-kw_only dataclass -- inserting would shift positional
    # callers). Spec 2.7b: this is a FIELD, not something the metrics read from cfg at
    # call time, and that distinction is load-bearing. The direction memos are cached by
    # setattr on this object, whose lifetime is NOT bounded by one dispatch; identity-
    # keying is sound only for values that are fields of the keyed object. That is why
    # the alpha-dependent memos are safe (p_adj_threshold IS a field) and why a config
    # value read inside the function would not be.
    target_resolution: TargetResolution = field(default_factory=lambda: TargetResolution({}, 0))


def _reject_duplicate_keys(df: pl.DataFrame, *, name: str) -> None:
    """A repeated ``(target, feature)`` row is malformed input, not a data condition.
    Issue #218.

    ONE decision at ONE seam, deliberately, because the alternative is the state #218
    measured: the same duplicated table produced THREE different answers across the metric
    set. Fifteen metrics already refused it -- the fourteen direction metrics via polars'
    ``validate="1:1"`` and `de_lfc_nmae` via its own purpose-written raise -- `sig_jaccard`
    de-duplicated and stayed correct, and the rest returned silently wrong numbers, four of
    them OUTSIDE their documented range (`sig_recall` 1.3333 on a ``[0,1]`` metric) and
    several more plausibly in-range (`de_wilcoxon_overlap` 0.5 -> 0.3333, `pr_auc`
    0.5 -> 0.8933). The last group is the reason this raises rather than de-duplicating:
    de-duplication is a lie for `nsig_counts` (the count genuinely changes) and cannot be
    right for `de_lfc_nmae`, whose denominator is the gate size.

    Reachable because DE tables are a SUPPORTED user-supplied input (``--de-pred`` /
    ``--de-real``), so duplicates need not come from this repo's own DE backends -- and the
    fifteen raisers do not make it safe, because a metric set excluding all of them
    completes with wrong numbers.

    ⚠️ BREAKING for a caller currently feeding duplicated tables. That caller is getting
    wrong numbers today, which is why the trade is worth taking.

    ⚠️ This does NOT make the per-metric guards redundant, and they are kept. A PreparedDE
    can be hand-built without passing through here -- `assemble_prepared_de` takes the frames
    directly -- so `de_lfc_nmae`'s raise and the direction metrics' ``validate="1:1"`` remain
    the last line of defence for that path.

    ⚠️ The SLICING drivers are not that path, contrary to an earlier version of this note:
    `scale._score_streaming_de`, `scale._score_streaming_cell_de` and
    `partition_inmem.score_piece` all call `prepare_de`, so they DO get this check, once per
    piece. `run._prepare_de_cached` does not call `prepare_de` either, but it runs
    `prep_de_side` per side itself, so it gets the check too. HAND ASSEMBLY -- calling
    `assemble_prepared_de` with frames that never went through `prep_de_side` -- is the only
    route that skips it.
    """
    # Fast path first: one hash pass over the key columns, and the group_by that builds the
    # message is paid for only by a table that is actually malformed. Measured on a
    # 3,005,073-row real DE table: 0.26 s for the count against 0.59 s for the group_by.
    keys = df.select(["target", "feature"])
    if keys.n_unique() == df.height:
        return
    dup = (
        keys.group_by(["target", "feature"]).len().filter(pl.col("len") > 1)
        .sort(["target", "feature"])
    )
    t, f, n = dup.row(0)
    raise ValueError(
        f"{name} DE table has {dup.height} duplicated (target, feature) key(s), e.g. "
        f"({t!r}, {f!r}) appearing {n} times. A repeated key fans out on every join and "
        f"changes row counts, gate sizes and denominators with no other signal, so the "
        f"metrics cannot aggregate it consistently -- some raise, some de-duplicate and "
        f"some return silently wrong values (issue #218). De-duplicate the table before "
        f"supplying it."
    )


def _warn_pred_gene_coverage(real_df: pl.DataFrame, pred_df: pl.DataFrame, *,
                             p_adj_threshold: float,
                             target_resolution: "TargetResolution | None" = None) -> None:
    """Surface reference-SIGNIFICANT ``(target, feature)`` pairs the pred table omits.
    Issue #291, the symmetric counterpart of #213's real-side coverage warning.

    The direction family builds its ranking pool with an INNER join
    (`metrics/direction.py::_direction_frame`), so a pair present in the real table and
    absent from the pred table is not in the pool at all: it neither advances the purity
    depth nor counts as a miss, while still counting in ``N_conf``. The family's stated
    invariant -- "a model cannot raise its score by omitting genes" -- is therefore
    DENOMINATOR-ONLY, and this is the diagnostic that says how far the pool departed from
    the reference's own budget on this run.

    ⚠️ It went QUIET on the ordinary h5ad path with #351, and that is a result rather than a
    regression. ``filter_gene_min_cpm_cell`` (5.0 under the competition preset) is applied to
    EACH SIDE's own table, and under rule_version 2 it kept a gene when the TARGET group's mean
    CPM cleared the threshold OR the control's did -- a per-side, per-target decision, so a gene
    the prediction expressed below the cutoff left the prediction's DE table while remaining in
    the reference's. Measured then on the official val A panel, counted on the post-exclusion
    population below: the `context_mean` baseline arm omitted 1,754 of 102,786
    reference-significant pairs (1.706%), touching 159 of 300 targets; an honest half-data arm
    omitted 199 (0.194%), touching 68. (Before excluding on-target pairs the denominator read
    103,085; both numerators were unchanged, because both pred tables happened to carry all 299
    resolved on-target rows.)

    Under rule_version 3 the gate keeps a gene on the CONTROL group's mean CPM alone (#351), and
    ``control_source="real"`` gives both sides the SAME control -- so both gate on the same
    ``ref_mean``, the two kept sets are identical, and the prediction covers the reference's whole
    confident budget. Re-measured on val A through the patched gate, same post-exclusion population
    as above: the `context_mean` baseline arm omits **0 of 100,771** (the gate takes 2,015 pairs out
    of the reference's confident budget on the way), the #351 attack arm 0 (was 1,097), an honest
    200-cell arm 0 (was 240). So #291's "a model cannot raise its score by omitting genes" stops
    being DENOMINATOR-ONLY on that path and becomes a full invariant, because there is nothing left
    for the inner join to drop.

    ⚠️ Still live, and still worth the two anti-joins, wherever the two sides do NOT share a
    control: a SUPPLIED ``--de-pred`` table (computed elsewhere, against anything) and
    ``control_source="pred"`` (the replicate anchor's splits, where each half carries its own
    control and therefore its own ``ref_mean``). A non-zero count there is the diagnostic doing
    its job; a non-zero count on an ordinary shared-control h5ad run now means the gate is not
    behaving as #351 describes.

    ⚠️ ``target_resolution`` is not optional in spirit. The scored direction metrics remove
    each RESOLVED target's own gene from ``N_conf`` before counting it
    (``direction._reference_stats`` -> ``direction._exclude_on_target``), so counting them here
    would report a gap that moves neither scored member. It is anti-joined away by the same
    ``(target, feature)`` mapping the metrics use -- NOT by ``feature != target``, which
    ignores ``target_gene_map`` entirely and is the raw-label comparison spec 2.1 rejects.
    Defaulted to None only so a direct caller can ask the unresolved question deliberately.

    Real-side only in what it counts: the significant set comes from `real_df` before the
    prediction is consulted, so the number this reports cannot be influenced by the
    submission except through the omissions it is measuring.

    ⚠️ CALLED FROM `metrics.direction._warn_coverage_once`, not from a constructor, and that
    placement is the point rather than an accident. That helper carries its own memo on the
    PreparedDE and is invoked by `_fidelity_family` and `de_direction_reach` -- between them
    exactly the eleven target-excluding direction metrics this diagnostic describes.
    Emitting it from
    `assemble_prepared_de` instead, as an earlier version did, spent two anti-joins on every
    DE run and reported an on-target-excluded count to a `vcc`/`minimal`/v1 caller that had
    selected no direction metric at all, or to one that had selected only the three v0.5.0
    metrics -- which read the UNFILTERED frame and so do not use this population. (Both
    Copilot and codex flagged that seam independently.) One residual gap remains, cheap for
    a diagnostic: a warm RESULT-cache hit returns before any metric runs, so it says nothing.
    """
    sig = (
        real_df.select(["target", "feature", "p_adj"])
        .filter(pl.col("p_adj") < p_adj_threshold)
        .select(["target", "feature"])
    )
    mapping = target_resolution.mapping if target_resolution is not None else {}
    if mapping:
        sig = sig.join(
            pl.DataFrame({"target": list(mapping.keys()), "feature": list(mapping.values())},
                         schema={"target": pl.String, "feature": pl.String}),
            on=["target", "feature"], how="anti",
        )
    if sig.height == 0:
        return
    missing = sig.join(pred_df.select(["target", "feature"]),
                       on=["target", "feature"], how="anti")
    if missing.height == 0:
        return
    worst = (
        missing.group_by("target").len().sort(["len", "target"], descending=[True, False])
    )
    t, n = worst.row(0)
    logger.warning(
        "pred DE table omits %d of the %d reference-significant (target, feature) pairs "
        "(%.3f%%), across %d of %d target(s); worst is %r with %d. An omitted pair never "
        "enters the direction metrics' ranking pool -- it neither advances the purity depth "
        "nor counts as a miss -- while still counting in N_conf, so the eleven "
        "target-excluding direction metrics (including the two vcc2026 scores among them) "
        "can read differently from a table covering the reference's whole confident budget. "
        "The effect is NOT one-directional: omitting head-ranked misses can inflate "
        "direction_reach_raw, while omitting correct calls or absorbable tail misses can "
        "lower it. Counts are for THIS prepared DE slice (issue #291).",
        missing.height, sig.height, 100.0 * missing.height / sig.height,
        worst.height, real_df["target"].n_unique(), t, n,
    )


def prep_de_side(src, *, name: str, sort_by: str,
                 nan_lfc_policy: str, min_abs_log2fc: float = 0.0) -> tuple[pl.DataFrame, list[str]]:
    """Load + normalize + nan-policy + lfc-floor a single side's DE table. Returns the
    validated frame and the sorted unique target labels. The (expensive) rank pivot — and
    the p_adj_threshold filter it applies — is a separate step (`rank_de_side`) so it can
    be cached independently."""
    df = apply_nan_policy(normalize_de_schema(load_de_table(src), name=name),
                          name=name, nan_lfc_policy=nan_lfc_policy)
    df = apply_lfc_floor(df, name=name, min_abs_log2fc=min_abs_log2fc)
    if sort_by not in df.columns:
        raise ValueError(
            f"sort_by={sort_by!r} but {name} DE table has no {sort_by!r} column; "
            f"present: {df.columns}"
        )
    if df["target"].null_count():
        raise ValueError(f"{name} DE table has null values in 'target'; targets must be non-null")
    _reject_duplicate_keys(df, name=name)
    perts = sorted(df["target"].unique().to_list())
    return df, perts


def rank_de_side(df: pl.DataFrame, *, sort_by: str, p_adj_threshold: float) -> pl.DataFrame:
    return _rank_matrix(df, sort_by=sort_by, p_adj_threshold=p_adj_threshold)


def assemble_prepared_de(real_rank, real_perts, pred_rank, pred_perts, *,
                         control: str, sort_by: str, p_adj_threshold: float,
                         real_df: pl.DataFrame, pred_df: pl.DataFrame,
                         target_resolution: TargetResolution | None = None) -> PreparedDE:
    """Assemble a PreparedDE from already-prepped sides.

    THIS -- not `prepare_de` -- is the constructor the main `compute_metrics` path enters:
    `run._prepare_de_cached` calls it directly and never calls `prepare_de` (spec 2.7b). Both
    take `target_resolution` for that reason.

    `target_resolution=None` DERIVES it by exact matching against `real_df`, which is
    correct only when the caller handed over an UNSLICED table. The three slicing entry
    points -- `scale._score_streaming_de`, `scale._score_streaming_cell_de` and
    `partition_inmem.score_piece` -- must pass it explicitly; deriving there reproduces the
    shard-local gate this parameter exists to prevent. (They reach here through `prepare_de`
    rather than calling this directly; symbol names rather than line numbers because the
    previous version of this note cited three that had all moved.)
    """
    if real_perts != pred_perts:
        raise ValueError(
            f"DE target sets differ between real and pred (real={real_perts}, pred={pred_perts})"
        )
    if control in real_perts:
        raise ValueError(f"control {control!r} must not appear as a DE target")
    if target_resolution is None:
        target_resolution = resolve_target_genes(real_df, real_perts)
    return PreparedDE(real_rank=real_rank, pred_rank=pred_rank, perturbations=real_perts,
                      sort_by=sort_by, p_adj_threshold=p_adj_threshold,
                      real_df=real_df, pred_df=pred_df,
                      target_resolution=target_resolution)


def prepare_de(de_pred, de_real, *, control: str,
               sort_by: str = "abs_log2_fold_change", p_adj_threshold: float = 0.05,
               nan_lfc_policy: str = "mask", min_abs_log2fc: float = 0.0,
               target_resolution: TargetResolution | None = None) -> PreparedDE:
    if sort_by not in SORT_KEYS:
        raise ValueError(f"sort_by must be one of {SORT_KEYS}, got {sort_by!r}")
    real_df, real_perts = prep_de_side(de_real, name="real", sort_by=sort_by,
                                       nan_lfc_policy=nan_lfc_policy, min_abs_log2fc=min_abs_log2fc)
    pred_df, pred_perts = prep_de_side(de_pred, name="pred", sort_by=sort_by,
                                       nan_lfc_policy=nan_lfc_policy, min_abs_log2fc=min_abs_log2fc)
    return assemble_prepared_de(
        rank_de_side(real_df, sort_by=sort_by, p_adj_threshold=p_adj_threshold), real_perts,
        rank_de_side(pred_df, sort_by=sort_by, p_adj_threshold=p_adj_threshold), pred_perts,
        control=control, sort_by=sort_by, p_adj_threshold=p_adj_threshold,
        real_df=real_df, pred_df=pred_df,
        target_resolution=target_resolution,
    )
