from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Literal

import polars as pl

from ..de import (
    PreparedDE,
    _warn_pred_gene_coverage,
    exclude_on_target,
    on_target_pairs,
    require_resolution,
)
from .de import _ensure_prepared, _to_float_nan

logger = logging.getLogger(__name__)

_FRAME_CACHE_ATTR = "_direction_frame_cache"
_EXCLUDED_CACHE_ATTR = "_direction_excluded_cache"
_REFSTATS_CACHE_ATTR = "_direction_refstats_cache"
_COMPONENTS_CACHE_ATTR = "_direction_components_cache"
_COVERAGE_WARNED_ATTR = "_direction_coverage_warned"

#: Purity floor for the RAW form of `de_direction_reach` -- the `vcc2026` scored member.
#: CALIBRATED, not derived, and deliberately INDEPENDENT of alpha; the derivation it
#: replaced produced the metric's own ceiling. See `de_direction_reach`.
REACH_PURITY_FLOOR = 0.9


def _defined(col: str) -> pl.Expr:
    """A log2FC carries a direction iff it is non-null, non-NaN and non-zero.

    NOT `is_finite()`: sign(+-inf) is well-defined, so an infinite log2FC (a gene going
    from zero to non-zero expression) IS a direction. `.fill_null(False)` because every
    comparison against a null yields null, and an unknown direction is not a direction.
    """
    c = pl.col(col)
    return (c.is_not_null() & c.is_not_nan() & (c != 0)).fill_null(False)


def _rank_p(col: str) -> pl.Expr:
    """A p-column mapped for sorting: null/NaN -> +inf so it ranks LAST.

    polars sorts nulls FIRST ascending and treats NaN as the largest float, so the raw
    column would put a gene with no p-value at the head of the ranking despite it being
    non-significant. Verified behaviour, not a precaution.

    The Float64 cast is load-bearing, not tidiness: ``is_nan()`` raises
    InvalidOperationError on a Decimal column, which a parquet DE table can carry, and
    ``Decimal.is_numeric()`` is True so the caller's dtype guard admits it. The cast also
    covers the REQUIRED ``p_adj`` column, which reaches this helper too and which no guard
    upstream screens for Decimal. ``strict=False`` maps anything uncastable to null, which
    then falls through to the +inf branch rather than raising.

    DOCUMENTED LIMITATION -- the ranking is therefore **Float64-normalised**. Two Decimal
    p-values that differ only beyond Float64's ~15 significant digits collide here, and the
    later key components decide their order instead. Measured: with p_adj tied and Decimal
    p_values differing at the 19th digit, sensitivity reads 0.0 where exact-Decimal
    ordering would give 0.5. This is accepted rather than fixed because every DE producer
    in this repo (gpudge, deseq2, pdex) emits float64 p-values, so a Decimal column can
    only arrive from an externally supplied table (parquet, or a DataFrame handed in
    directly), and at any realistic scale (e.g. Decimal(10,6)) the conversion is
    order-preserving and injective over valid p-values -- not bit-exact, since most decimal
    fractions have no exact binary representation, but exactness is not what the ranking
    needs.

    ⚠️ CORRECTED (issue #204 review): this docstring used to claim the normalisation does
    NOT threaten the precision == purity identity, on the grounds that polars implicitly
    casts Decimal for the ``p_adj < alpha`` comparison too, so filter and ranking share one
    representation. That holds only when ``alpha`` is a FLOAT. Measured on polars 1.43.0
    against a Decimal(30,21) column whose two values collide in Float64:

        p < 1.0                     (float)   -> [False, False]   # cast to Float64
        p < 1                       (int)     -> [True,  False]   # native Decimal
        p < Decimal("1")            (Decimal) -> [True,  False]   # native Decimal

    So any NON-FLOAT threshold -- integer or Decimal, i.e. anything that preserves the
    Decimal comparison -- splits the significance filter from the Float64 ranking key, so
    ``{p_adj < alpha}`` CAN stop being a prefix of the ranking and the identity CAN break.
    The guarantee is what a non-float threshold removes; an actual break additionally needs
    two p-values that collide in Float64 across the boundary AND a later key component that
    orders the non-significant one first. Built end to end with an integer threshold and
    exactly that shape, precision reads 1.0 while purity at the boundary reads 0.0.

    This prefix violation was PRE-EXISTING and already broke the OLD identity -- the same
    construction reads 0.0 against row-counting depth. #204 changes neither the ranking,
    the significance membership, nor purity. It does move the lookup POSITION, though, from
    ``|S_pred|`` to ``|S_pred INTERSECT adjudicable|``, so a crafted input could satisfy one
    indexing and not the other; the claim is that the violation predates #204, not that
    every affected input behaves identically across it.

    ✅ FIXED by issue #207, in `_purity_curve` rather than here: its sort now LEADS with a
    native ``p_adj_pred < alpha`` boolean, so the significant set is a prefix by
    construction and no longer depends on the two sharing a representation. This helper is
    unchanged and the Float64 normalisation above still stands -- it decides the order
    WITHIN the significant block and within the non-significant block, where two Decimal
    p-values differing beyond ~15 significant digits still collide and fall through to the
    later key components. That residual is accepted for the reasons above; what #207
    removed is the boundary crossing, which is the part the identity rested on.
    """
    c = pl.col(col).cast(pl.Float64, strict=False)
    return pl.when(c.is_null() | c.is_nan().fill_null(True)).then(float("inf")).otherwise(c)


def _direction_frame(prepared: PreparedDE) -> pl.DataFrame:
    """One row per gene shared by the pred and real DE tables, carrying the ranking key
    and the two flags the asymmetric rule (spec section 3) needs:

    ``in_denom`` -- the REFERENCE can adjudicate. False (pair excluded entirely) when the
    real log2FC is null, NaN or exactly zero.
    ``match`` -- ``in_denom`` AND the model committed AND the signs agree. A model that
    declines to commit (pred log2FC null/NaN/zero) therefore stays in the denominator and
    counts as a miss; it is never silently dropped.

    Because a pair that is both adjudicable and committed has two well-defined +-1 signs,
    polars' ``NaN == NaN -> True`` can never be reached here by construction.

    Ranked by the PREDICTION's (p_adj, p_value, |log2FC| desc, feature). p_adj is primary
    and that is load-bearing: apply_nan_policy('mask') and apply_lfc_floor both rewrite
    p_adj to 1.0 without touching p_value, so a p_value-primary key would stop
    {p_adj < alpha} being a prefix of the ranking and would falsify the
    precision == purity identity -- which since issue #204 is indexed at
    k = |S_pred INTERSECT adjudicable| rather than at k = |S_pred|, purity itself being
    unchanged by that fix. See spec section 4.1.

    Memoized on the (transient, in-process) PreparedDE: the three v0.5.0 direction
    metrics call this on the same object within one dispatch and the join is heavy. The
    eleven chance-corrected metrics (issue #195) read `_ontarget_excluded_frame`, which
    is derived from this frame but cached SEPARATELY -- the two caches must never be
    served to each other's callers. The defensive try/except matches the repo idiom in
    metrics/de.py::_tested_universe.

    Two invariants the cache rests on, neither of which the type system enforces:
    (1) this frame is ALPHA-FREE -- alpha is applied later, by _purity_curve's filter and
    by precision's filter -- so the same cache is valid for every alpha, and the alpha
    filter must never be moved in here; (2) PreparedDE must not be MUTATED after
    construction. It is a plain, non-frozen dataclass, so keying on object identity
    assumes no-mutation as a contract rather than proving it; reassigning pred_df/real_df
    after a first call would return a stale frame. _tested_universe assumes the same.
    """
    cached = getattr(prepared, _FRAME_CACHE_ATTR, None)
    if cached is not None:
        return cached

    # p_value is optional AND untyped: it is not in REQUIRED_COLS, and load_de_table only
    # pins target/feature dtypes, so an all-blank column in a user CSV arrives as String.
    # is_nan() raises InvalidOperationError on a non-numeric dtype, so an unusable column
    # is treated as ABSENT rather than crashing all three metrics. Numeric-but-null values
    # are fine -- _rank_p maps them to +inf.
    _p_dtype = prepared.pred_df.schema.get("p_value")
    has_p_value = _p_dtype is not None and _p_dtype.is_numeric()
    if not has_p_value:
        logger.warning(
            "pred DE table has no usable 'p_value' column (%s); the direction metrics will "
            "rank by 'p_adj' alone. BH plateaus make that a coarser ordering WITHIN a "
            "plateau, so values may differ from a p_value-carrying run. The significance "
            "boundary, and therefore direction_precision, are unaffected.",
            "absent" if _p_dtype is None else f"non-numeric dtype {_p_dtype}",
        )
    pred_cols = ["target", "feature", "log2_fold_change", "p_adj"]
    if has_p_value:
        pred_cols.append("p_value")
    pred = prepared.pred_df.select(pred_cols).rename(
        {"log2_fold_change": "lfc_pred", "p_adj": "p_adj_pred"}
    )
    pred = pred.with_columns(
        rank_p_adj=_rank_p("p_adj_pred"),
        # A constant when p_value is absent: present in the sort key either way, but
        # unable to affect the order. Keeps the key one fixed shape.
        rank_p_value=(_rank_p("p_value") if has_p_value else pl.lit(0.0)),
    )
    if has_p_value:
        pred = pred.drop("p_value")
    real = prepared.real_df.select(["target", "feature", "log2_fold_change", "p_adj"]).rename(
        {"log2_fold_change": "lfc_real", "p_adj": "p_adj_real"}
    )
    # validate='1:1': prepare_de does not enforce key uniqueness, and a duplicated
    # (target, feature) fans out on the join -- one reference gene matched by two pred rows
    # gives k*=2 against N_conf=1, i.e. an "adjudicated" sensitivity of 2.0, above the
    # bound the metric advertises. Raise instead of silently double-counting.
    frame = (
        pred.join(real, on=["target", "feature"], how="inner", validate="1:1")
        .with_columns(
            in_denom=_defined("lfc_real"),
            _committed=_defined("lfc_pred"),
            # null/NaN magnitude -> -inf so it sorts LAST under `descending`; +-inf keeps
            # its magnitude and leads a tie, consistent with it being a real direction.
            abs_lfc_pred=pl.when(
                pl.col("lfc_pred").is_null() | pl.col("lfc_pred").is_nan().fill_null(True)
            ).then(float("-inf")).otherwise(pl.col("lfc_pred").abs()),
        )
        .with_columns(
            match=(
                pl.col("in_denom")
                & pl.col("_committed")
                & (pl.col("lfc_pred").sign() == pl.col("lfc_real").sign()).fill_null(False)
            )
        )
        .drop("_committed")
    )
    try:
        setattr(prepared, _FRAME_CACHE_ATTR, frame)
    except (AttributeError, ValueError, TypeError):  # frozen / read-only view -> skip memo
        pass
    return frame


def _memo(prepared: PreparedDE, attr: str, build):
    """Identity-keyed memo on the PreparedDE, matching _direction_frame's idiom.

    ⚠️ Identity-keying is sound ONLY for values that are FIELDS of the keyed object
    (spec 3). `p_adj_threshold` qualifies, and so does `target_resolution` -- which is
    exactly why the resolution is a PreparedDE field rather than something read from cfg
    at call time: the memo is a setattr that outlives the dispatch, so a config value
    would let one map's results be served to a caller passing another. Any future
    per-metric input that is neither a PreparedDE field nor a constant must enter the
    memo key explicitly.
    """
    cached = getattr(prepared, attr, None)
    if cached is not None:
        return cached
    value = build()
    try:
        setattr(prepared, attr, value)
    except (AttributeError, ValueError, TypeError):  # frozen / read-only view -> skip memo
        pass
    return value


def _warn_coverage_once(prepared: PreparedDE) -> None:
    """Issue #291's pred-coverage diagnostic, at most once per PreparedDE.

    ⚠️ Memoized on its OWN key, not folded into `_reference_stats`. That build has direct
    callers which are not metrics -- a diagnostic reading `n_conf`, or a test -- and emitting
    from inside it would let such a call consume the memo and SWALLOW the warning for a
    scored metric run later on the same object, which is worst if the diagnostic call
    happened to have logging suppressed. Keeping the build side-effect-free and gating the
    emission separately makes the warning a property of "a metric that reads this population
    ran", which is what it claims to be.

    Called from `_fidelity_family` (the seven readers) and `de_direction_reach` (the four
    reach variants) -- between them exactly the eleven target-excluding metrics, and none of
    the three v0.5.0 ones, which read the UNFILTERED frame and so do not use the
    on-target-excluded population this counts over.
    """
    _memo(prepared, _COVERAGE_WARNED_ATTR,
          lambda: (_warn_pred_gene_coverage(
              prepared.real_df, prepared.pred_df,
              p_adj_threshold=prepared.p_adj_threshold,
              target_resolution=prepared.target_resolution) or True))


def _require_resolution(prepared: PreparedDE) -> None:
    """The zero-resolve gate (spec 2.1/5) for the eleven chance-corrected metrics.

    ⚠️ The gate itself moved to `de.require_resolution` when issue #172 gave it a second
    family of callers (`de_sig_jaccard` and `de_lfc_nmae` exclude too). This stays as the
    PreparedDE-shaped entry point the eleven call, so the placement argument -- enforced at
    the entry of each EXCLUDING metric, never in a constructor -- is unchanged; see that
    function for why.
    """
    require_resolution(prepared.perturbations, prepared.target_resolution)


def _on_target_pairs(prepared: PreparedDE) -> pl.DataFrame:
    """The (target, feature) pairs to remove: each RESOLVED target's own gene."""
    return on_target_pairs(prepared.target_resolution)


def _exclude_on_target(df: pl.DataFrame, prepared: PreparedDE) -> pl.DataFrame:
    """Anti-join away each resolved target's own gene. Spec 2.7c.

    NOT `filter(feature != target)`: that is the raw-label comparison spec 2.1 rejects,
    and it ignores target_gene_map completely -- the map would be accepted, fingerprinted,
    and then have no effect. A target that did not resolve drops nothing.
    """
    return exclude_on_target(df, prepared.target_resolution)


def _ontarget_excluded_frame(prepared: PreparedDE) -> pl.DataFrame:
    """`_direction_frame` with each resolved target's own gene removed.

    ALL ELEVEN chance-corrected metrics read this. The UNFILTERED `_direction_frame` is
    read only by the three v0.5.0 metrics, whose values must not move (spec 1/3) --
    the two caches are separate and must never be served to each other's callers.
    """
    return _memo(prepared, _EXCLUDED_CACHE_ATTR,
                 lambda: _exclude_on_target(_direction_frame(prepared), prepared))


def _reference_stats(prepared: PreparedDE) -> pl.DataFrame:
    """Per-target `N_conf`, `q` and `d`, from `prepared.real_df` ALONE. Spec 2.1.

    Never touches the join: `_direction_frame` is an INNER join, so anything computed
    from it is prediction-dependent, and `q`/`N_conf` must not be -- that invariance is
    what stops a model shrinking its own BUDGET by omitting genes.

    ⚠️ DENOMINATOR-ONLY. Issue #291 is the correction, and it is a correction to the
    CONCLUSION, not to the premise: the invariance established here is real, and a model
    genuinely cannot shrink `N_conf`. It does not extend to the family's score, because
    the NUMERATOR is built on the inner join. A `(target, feature)` pair present in the
    real table and absent from the pred table is not in the ranking pool at all -- it
    neither advances `_purity_curve`'s depth nor counts as a miss -- while still counting
    in `N_conf` here. Since `k*` takes the DEEPEST qualifying depth, deleting misses that
    sit at the HEAD of the ranking is worth a discontinuous jump: measured on a one-target
    toy with 80 adjudicable pairs whose 9 top-ranked are wrong, deleting exactly those 9
    rows moves `direction_reach_raw` from 0.0 to 0.8875.

    ⚠️ The SIZE of the head-miss block that triggers this scales with the purity floor.
    `k* = 0` needs `(N - m)/N < P0`, i.e. `m > (1 - P0)N` -- 3 of 80 under the old
    `P0 = 0.975`, 9 of 80 under `REACH_PURITY_FLOOR = 0.9`. So moving the floor down made the
    exploit ~4x harder to reach and smaller when it lands (the jump was 0.0 -> 0.9625), but it
    did NOT close it, and it must not be recorded as having done so.

    ⚠️ Row ABSENCE is not the strongest form of that move, so a fix aimed only at absence
    would miss. On the same toy, setting those 9 rows' `p_adj_pred` to 1.0 -- declining to
    call them, with every row still present -- scores 0.975, ABOVE what deleting them
    gets. `universe='adjudicated'` filters the pool on the REFERENCE's significance but
    RANKS it by the PREDICTION's `p_adj`, so a declined call is merely demoted to the tail
    where it costs almost nothing. This is also why the real-anchored (left) join proposed
    on #291 is not the fix it looks like: a re-admitted pair carries a null `p_adj_pred`,
    which `_rank_p` sends to the tail, reproducing that same 0.975 and scoring the omission
    HIGHER than the shipped inner join does (measured; both numbers re-measured at
    `REACH_PURITY_FLOOR = 0.9`). `de._warn_pred_gene_coverage`
    reports the departure; closing it means deciding where a non-answer RANKS, which moves
    scored numbers on honest submissions too and is a release decision, not a metric one.

    `N_conf` and `q` have deliberately DIFFERENT populations. `N_conf` requires
    significance only (a budget count a model must not be able to shrink); `q`
    additionally requires a direction, because it is a rate over SIGNS and a gene with no
    direction cannot vote. Consequence: q can be undefined (null) while N_conf > 0.

    Seeded from `prepared.perturbations`, NOT from the surviving rows -- otherwise a
    target whose only significant gene WAS its own target gene disappears from the frame
    instead of producing an explicit `N_conf = 0` record, and spec 5's convention never
    fires for it.

    ALPHA-DEPENDENT (unlike `_direction_frame`): safe to identity-cache only because
    `p_adj_threshold` is a field of the same unmutated PreparedDE.
    """
    def build() -> pl.DataFrame:
        alpha = prepared.p_adj_threshold
        real = _exclude_on_target(
            prepared.real_df.select(["target", "feature", "log2_fold_change", "p_adj"]),
            prepared,
        )
        sig = real.filter(pl.col("p_adj") < alpha)
        n_conf = sig.group_by("target").len().rename({"len": "n_conf"})
        votes = (
            sig.with_columns(in_denom=_defined("log2_fold_change"))
            .filter(pl.col("in_denom"))
            .with_columns(pos=(pl.col("log2_fold_change") > 0).cast(pl.Int64))
        )
        qf = (
            votes.group_by("target")
            .agg(n_votes=pl.len(), n_pos=pl.col("pos").sum())
            .with_columns(
                q=pl.max_horizontal(
                    pl.col("n_pos"), pl.col("n_votes") - pl.col("n_pos")
                ) / pl.col("n_votes")
            )
            .select(["target", "q"])
        )
        base = pl.DataFrame({"target": list(prepared.perturbations)},
                            schema={"target": pl.String})
        return (
            base.join(n_conf, on="target", how="left")
            .join(qf, on="target", how="left")
            .with_columns(n_conf=pl.col("n_conf").fill_null(0).cast(pl.Int64))
            .with_columns(
                # d = max(1 - q, 0.05) -- but UNDEFINED where q is. max_horizontal skips
                # nulls, so without this guard an undefined q would silently yield 0.05.
                d=pl.when(pl.col("q").is_null())
                .then(None)
                .otherwise(pl.max_horizontal(1.0 - pl.col("q"), pl.lit(0.05)))
                .cast(pl.Float64)
            )
        )

    return _memo(prepared, _REFSTATS_CACHE_ATTR, build)


def _components(prepared: PreparedDE) -> pl.DataFrame:
    """`_reference_stats` plus the per-target scored-set size `n_pred` and match count `k`.

    Scored set = {g : p_adj_pred(g) < alpha} INTERSECT {in_denom}, with the target gene
    removed (spec 2.1). `k` counts matches within it. Every perturbation gets a row.
    """
    def build() -> pl.DataFrame:
        alpha = prepared.p_adj_threshold
        scored = _ontarget_excluded_frame(prepared).filter(
            (pl.col("p_adj_pred") < alpha) & pl.col("in_denom")
        )
        agg = scored.group_by("target").agg(
            n_pred=pl.len(), k=pl.col("match").cast(pl.Int64).sum()
        )
        return (
            _reference_stats(prepared)
            .join(agg, on="target", how="left")
            .with_columns(
                n_pred=pl.col("n_pred").fill_null(0).cast(pl.Int64),
                k=pl.col("k").fill_null(0).cast(pl.Int64),
            )
        )

    return _memo(prepared, _COMPONENTS_CACHE_ATTR, build)


def _purity_curve(
    frame: pl.DataFrame, *, universe: Literal["adjudicated", "all"], alpha: float
) -> pl.DataFrame:
    """Cumulative direction purity over a per-target ranked prefix.

    ``universe='adjudicated'`` ranks only reference-significant genes (bounding k* by
    N_conf); ``'all'`` ranks every shared gene (leaving k* unbounded above).

    ``k`` is the DEPTH: how many ADJUDICABLE pairs the prefix has examined. It is
    deliberately the same count as ``n_denom`` -- computed once and aliased so the two
    cannot drift -- because a pair the reference cannot adjudicate carries no directional
    evidence and must neither advance the depth nor enter the purity denominator.

    ⚠️ Until issue #204 the depth was ``pl.col("in_denom").cum_count()``, which counts
    NON-NULL entries rather than True ones; ``_defined()`` ends in ``.fill_null(False)``,
    so ``in_denom`` is never null and that expression was simply the 1-based ranked ROW
    position. Unadjudicable pairs therefore extended every prefix for free -- and they are
    not scattered: a gene the reference never detected has real log2FC exactly zero, yet
    still draws a small predicted delta whose within-group variance is zero when the
    prediction's control cells are all zero, giving it maximal predicted significance and
    sorting it to the FRONT of the ranking. Measured on a 1,029-perturbation panel, 97.9%
    of the median k* prefix carried no directional evidence, inflating
    direction_reach_unbounded, direction_reach_unbounded_raw and
    direction_sensitivity_universe by ~50x (median). ``universe='adjudicated'`` was NOT
    affected -- zero unadjudicable rows among the reference-significant genes of all six
    reference lines measured -- so the SCORED direction_reach and direction_sensitivity
    did not move.

    The precision == purity identity survives, RE-INDEXED: purity is untouched (the same
    n_match/n_denom at the same prefixes; only the depth axis is relabelled), so precision
    still equals purity at the end of the S_pred prefix -- now at
    k = |S_pred INTERSECT adjudicable| rather than at k = |S_pred|. See the identity tests
    in tests/test_direction_precision.py, one of which carries an unadjudicable row inside
    S_pred specifically to pin the re-indexed form.

    ⚠️ That prefix used to have one PRE-EXISTING exception, and issue #207 closes it: a
    NON-FLOAT ``p_adj_threshold`` (integer or Decimal) over a Decimal ``p_adj`` column put
    the significance filter and the Float64 ranking key on different representations, so
    the significant set COULD stop being a prefix and precision could read 1.0 where purity
    at the boundary read 0.0. The sort now leads with a NATIVE ``p_adj_pred < alpha``
    boolean, which makes the prefix structural rather than representational. It is inert
    wherever the two representations already agreed -- see the sort below -- so it moves
    results only for an externally supplied Decimal table under a non-float threshold,
    which is the input that was wrong.

    ⚠️ ``universe='adjudicated'`` is filtered on reference SIGNIFICANCE, which does not
    imply adjudicability: nothing forbids a significant row whose real log2FC is exactly
    zero or null (or NaN under nan_lfc_policy='keep'), and an externally supplied real
    table can carry anything. Such a row, once joined, no longer advances the depth and can
    therefore shorten k* -- can, not must: it may be absent from the prediction, or there may
    simply be no qualifying curve row at or after it. (NOT "it sits past where purity fell
    below P0": purity is not monotone and `_k_star` takes the DEEPEST qualifying k, so a dip
    that recovers still counts.) So the adjudicated variants CAN move on a dataset of that
    shape. They did not move on any of the six reference lines measured --
    0 unadjudicable rows among 326,832 / 177,289 / 419,959 / 339,572 / 205,840 / 244,205
    reference-significant genes -- which is an empirical result, not a guarantee.
    """
    if universe not in ("adjudicated", "all"):
        raise ValueError(f"universe must be 'adjudicated' or 'all', got {universe!r}")
    if universe == "adjudicated":
        frame = frame.filter(pl.col("p_adj_real") < alpha)
    return (
        frame.with_columns(
            # #207: the significant set as a NATIVE comparison, sorted FIRST, so
            # {p_adj_pred < alpha} is a prefix of the ranking BY CONSTRUCTION rather than
            # as a consequence of the two sharing a representation. `_rank_p` casts to
            # Float64 (it must -- is_nan() raises on Decimal), while this comparison keeps
            # whatever representation polars picks from the THRESHOLD's type, and those two
            # disagree for a non-float threshold over a Decimal column. Making the boolean
            # the leading key removes the dependency instead of papering over it.
            #
            # INERT wherever the two already agree, which is every configuration this repo's
            # own DE producers can reach (they all emit Float64 p-values): for such a p_adj
            # the Float64 cast orders faithfully, `_sig_pred` is exactly
            # `rank_p_adj < alpha`, a monotone function of the next key, so sorting on it
            # first cannot reorder anything. Checked exhaustively over the awkward values --
            # 0, alpha exactly, NaN, +-inf and null all satisfy the identity -- verified on
            # the official val A DE tables (every direction metric bit-identical), and by
            # the whole suite.
            #
            # ⚠️ One behaviour change on a degenerate input, deliberately not guarded: a
            # NON-NUMERIC `p_adj_pred` (e.g. an all-blank String column) makes this
            # comparison raise ComputeError, where `universe='all'` previously ranked it as
            # all-missing and returned a number. Every production path already raises on it
            # earlier -- `rank_de_side`/`_rank_matrix` filters `p_adj < threshold` on both
            # sides -- so only a hand-built PreparedDE reaches this, and there loud beats a
            # curve computed from no p-values at all. A dtype fallback would reintroduce
            # exactly the representational coupling this key exists to remove; `p_value`
            # gets that treatment in `_direction_frame` because it is OPTIONAL, and `p_adj`
            # is not.
            #
            # `.fill_null(False)`: a null p_adj yields a null comparison, and `filter`
            # drops a null row, so False is what keeps this key agreeing with the
            # significance filter `de_direction_precision` applies. NaN compares False for
            # the same reason and `_rank_p` sends both to +inf, so they stay last either way.
            _sig_pred=(pl.col("p_adj_pred") < alpha).fill_null(False),
        )
        .sort(
            ["target", "_sig_pred", "rank_p_adj", "rank_p_value", "abs_lfc_pred", "feature"],
            descending=[False, True, False, False, True, False],
        )
        .with_columns(
            n_denom=pl.col("in_denom").cast(pl.Int64).cum_sum().over("target"),
            n_match=pl.col("match").cast(pl.Int64).cum_sum().over("target"),
        )
        .with_columns(
            # Aliased, not recomputed: depth and the purity denominator are the same
            # count by construction (#204), and two separate expressions would let a
            # future edit silently reintroduce the row-counting depth.
            k=pl.col("n_denom"),
            purity=pl.when(pl.col("n_denom") > 0)
            .then(pl.col("n_match") / pl.col("n_denom"))
            .otherwise(None),
        )
        .select(["target", "k", "n_denom", "n_match", "purity"])
    )


def _k_star(curve: pl.DataFrame, *, p0: float | Mapping[str, float]) -> dict[str, int]:
    """Deepest prefix whose purity still reaches p0, per target.

    'Deepest' is literal: purity is not monotone, so a k past a dip counts if purity
    recovers. Targets that never reach p0 are absent from the result -- callers read that
    as k* = 0, a computed zero rather than a missing value.

    `p0` may be a SCALAR -- `REACH_PURITY_FLOOR` for raw reach, or `1 - alpha/2` for the
    v0.5.0 `de_direction_sensitivity`, shared by every target -- or a per-target MAPPING
    (the corrected threshold q + (1-alpha)(1-q), which varies with q).
    A target ABSENT from a mapping drops out entirely -- that is how an undefined q
    reaches the caller as "no threshold exists", which the caller must render as NaN
    rather than as the k* = 0 that a never-reached threshold means (spec 5).
    """
    if curve.height == 0:
        return {}
    # ⚠️ Dispatch on Mapping, NOT on `isinstance(p0, (int, float))`. numpy scalars are not
    # Python floats -- `isinstance(np.float32(0.5), float)` is False (only np.float64
    # subclasses it) -- so an int/float test sends a np.float32 threshold down the MAPPING
    # branch and dies with "'numpy.float32' object has no attribute 'keys'". That reaches
    # here from an ordinary `DEParams(p_adj_threshold=np.float32(0.05))`, because
    # `1.0 - np.float32(a)/2.0` stays np.float32 under NEP 50, and it would REGRESS the
    # v0.5.0 de_direction_sensitivity, which handled numpy thresholds before this change.
    if not isinstance(p0, Mapping):
        hit = curve.filter(pl.col("purity") >= float(p0))
    else:
        thresholds = pl.DataFrame(
            {"target": list(p0.keys()), "_p0": [float(v) for v in p0.values()]},
            schema={"target": pl.String, "_p0": pl.Float64},
        )
        hit = curve.join(thresholds, on="target", how="inner").filter(
            pl.col("purity") >= pl.col("_p0")
        )
    hit = hit.group_by("target").agg(pl.max("k").alias("k_star"))
    return dict(zip(hit["target"].to_list(), hit["k_star"].to_list(), strict=True))


def de_direction_precision(
    prepared: PreparedDE | None = None,
    *,
    de_pred=None,
    de_real=None,
    control: str = "non-targeting",
    sort_by: str = "abs_log2_fold_change",
    p_adj_threshold: float = 0.05,
    nan_lfc_policy: str = "mask",
    min_abs_log2fc: float = 0.0,
) -> dict[str, float]:
    """Per-perturbation fraction of model-significant genes whose log2FC sign the
    reference agrees with; the reference's own significance is deliberately ignored.

    Denominator: model-significant genes the reference can adjudicate. A gene absent from
    the real table, or whose real log2FC carries no direction, is EXCLUDED; a gene the
    model called significant without committing to a direction stays in and counts as a
    MISS. See spec section 3 -- the asymmetry is this repo's no-droppable-NaN principle
    (issue #89) at gene granularity, so a model can never gain by declining to answer.

    Note this denominator is the opposite of the rule the competition notes prescribe for
    a graded submission, which is correct here: the reference, not the model, decides what
    can be scored.

    Corrected-semantics successor to de_model_direction_match, which is retained unchanged
    -- it scores a both-ZERO or both-NaN pair as agreement (verified). A both-NULL pair is
    not agreement there: the comparison yields null and pl.mean ignores it.
    """
    prepared = _ensure_prepared(prepared, de_pred, de_real, control=control,
                                sort_by=sort_by, p_adj_threshold=p_adj_threshold,
                                nan_lfc_policy=nan_lfc_policy,
                                min_abs_log2fc=min_abs_log2fc)
    agg = (
        _direction_frame(prepared)
        .filter(pl.col("p_adj_pred") < prepared.p_adj_threshold)
        .group_by("target")
        .agg(
            n_denom=pl.col("in_denom").cast(pl.Int64).sum(),
            n_match=pl.col("match").cast(pl.Int64).sum(),
        )
        .filter(pl.col("n_denom") > 0)
        .with_columns(precision=pl.col("n_match") / pl.col("n_denom"))
        .select(["target", "precision"])
    )
    # Targets with no adjudicable model-significant gene are omitted; v2 dispatch fills
    # them with the catalog's worst value (0.0). Same convention as de_sig_recall.
    return {row[0]: _to_float_nan(row[1]) for row in agg.iter_rows()}


def de_direction_sensitivity(
    prepared: PreparedDE | None = None,
    *,
    universe: Literal["adjudicated", "all"] = "adjudicated",
    de_pred=None,
    de_real=None,
    control: str = "non-targeting",
    sort_by: str = "abs_log2_fold_change",
    p_adj_threshold: float = 0.05,
    nan_lfc_policy: str = "mask",
    min_abs_log2fc: float = 0.0,
) -> dict[str, float]:
    """Per-perturbation depth to which the prediction's ranking stays directionally pure,
    as a fraction of what the reference confidently adjudicated: ``k*(t) / N_conf(t)``.

    ``P0 = 1 - alpha/2`` is DERIVED, not configurable: a false discovery is a true null,
    so its sign in an independent repeat is a coin flip, and an FDR-alpha set therefore
    sustains a purity of 1 - alpha/2 (Notion section 1.3). At the project-standard
    alpha = 0.05 this is 0.975.

    ``universe='adjudicated'`` (the headline) ranks only reference-significant genes, so
    k* <= N_conf and the result is bounded by 1. ``universe='all'`` ranks the whole shared
    universe and is UNBOUNDED ABOVE -- measured exceeding 1 for 19.2% of targets on a
    1,029-perturbation panel, with a generic-response baseline that beats a real replicate.
    Its catalog entry states those as two separate facts: unboundedness is ``anchor=None``
    (there is no constant value a perfect submission attains), while enrolment is
    ``scored=True`` (it has a direction). The old ``best_value="none"`` was one token for
    both, so neither could be stated without the other -- which is why the metric was
    diagnostic-only until the two were split.

    ⚠️ It is scored DESPITE the inversion. The ``[-2, 2]`` clamp its anchorless policy carries
    bounds how far the inversion can move an aggregate; it does not correct it. Read this
    metric's contribution with that in mind.

    ⚠️ The ~74% figure this docstring previously carried was measured BEFORE issue #204:
    depth then counted ranked rows rather than adjudicable pairs, so the full-universe
    variant was ~50x (median) larger and exceeded 1 far more often. The metric still
    exceeds 1 -- a larger ranking pool genuinely can sustain purity past N_conf -- so the
    anchorless policy is unchanged.

    N_conf counts reference-significant genes across the whole real table, not just those
    the prediction covers. A target whose adjudicated genes the prediction does not carry
    AT ALL therefore scores a computed 0.0, not an omission.

    ⚠️ That is a statement about the DENOMINATOR, and issue #291 corrects the "so a model
    cannot raise its score by omitting genes" this docstring used to draw from it. The
    numerator's pool is the inner join at `_direction_frame`, so a PARTIAL omission --
    dropping some of a target's reference-significant pairs while keeping the rest --
    removes those pairs from the ranking without removing them from `N_conf`, and dropping
    head-ranked MISSES raises `k*` discontinuously. See `_reference_stats` for the measured
    figures and for why the obvious left-join repair scores the omission higher still.
    """
    prepared = _ensure_prepared(prepared, de_pred, de_real, control=control,
                                sort_by=sort_by, p_adj_threshold=p_adj_threshold,
                                nan_lfc_policy=nan_lfc_policy,
                                min_abs_log2fc=min_abs_log2fc)
    alpha = prepared.p_adj_threshold
    curve = _purity_curve(_direction_frame(prepared), universe=universe, alpha=alpha)
    k_star = _k_star(curve, p0=1.0 - alpha / 2.0)
    n_conf = prepared.real_df.filter(pl.col("p_adj") < alpha).group_by("target").len()
    # Targets with N_conf == 0 are absent from n_conf entirely and so are omitted here
    # (undefined -- the reference adjudicated nothing); v2 dispatch fills them with the
    # catalog's worst value.
    return {
        target: _to_float_nan(k_star.get(target, 0) / count)
        for target, count in zip(n_conf["target"].to_list(), n_conf["len"].to_list(),
                                 strict=True)
        if count > 0
    }


_NAN = float("nan")


def _safe_div(num: pl.Expr, den: pl.Expr) -> pl.Expr:
    """num/den as Float64, with a zero denominator giving NaN rather than +-inf.

    Explicit rather than relying on IEEE 0/0: every zero denominator in spec 5's matrix
    is a genuinely undefined quantity, and `k/0` with k > 0 must not become +inf.
    """
    return (
        pl.when(den == 0)
        .then(pl.lit(_NAN, dtype=pl.Float64))
        .otherwise(num.cast(pl.Float64) / den.cast(pl.Float64))
    )


def _fidelity_family(prepared: PreparedDE) -> pl.DataFrame:
    """All seven fidelity-family metrics, per target. Spec 2.2 + spec 5.

    ⚠️ Spec 5's degenerate matrix is implemented HERE and nowhere else. Reimplementing it
    per metric is how the columns drift apart, and several cells differ between metrics in
    ways that look like typos but are not (e.g. `fidelity_yield` is 0.0 when q is defined
    and NaN when it is not, at the same n_pred = 0).

    PRECEDENCE, in full order -- normative, and applied as written:
      1. N_conf == 0 -> the DOR's AGREEMENT convention, which OVERRIDES q-undefined for
         the corrected fidelity pair (1.0 if the model also called nothing, 0.0 if it
         claimed a budget that did not exist). N_conf == 0 always implies q undefined, so
         a bare "q-undefined wins" would turn every one of those cells into NaN and delete
         this convention entirely.
      2. Within N_conf > 0, q-undefined beats n_pred == 0: without q there is no chance
         correction to apply, whereas the n_pred == 0 convention presumes a well-defined
         correction the model declined to use.
      3. Otherwise the spec 2.2 formulas.

    Only `fidelity` and `fidelity_yield` need the explicit chain; the other five reduce to
    plain arithmetic once a zero denominator gives NaN. That reduction is verified against
    spec 5's table in tests/test_direction_fidelity.py -- do not take it on trust.
    """
    _require_resolution(prepared)  # spec 5's precondition, at the metrics that own it
    _warn_coverage_once(prepared)   # issue #291's diagnostic, at the metrics it describes
    comp = _components(prepared)
    n_conf, n_pred, k, q, d = (pl.col(c) for c in ("n_conf", "n_pred", "k", "q", "d"))
    zero_conf, q_undef, zero_pred = n_conf == 0, q.is_null(), n_pred == 0

    fidelity_raw = _safe_div(k, n_pred)
    coverage = _safe_div(n_pred, n_conf)
    # `corrected` is only READ where N_conf > 0 and q is defined; the guards below make
    # the null-arithmetic in the other branches unreachable.
    corrected = (fidelity_raw - q) / d

    fidelity = (
        pl.when(zero_conf)
        .then(pl.when(zero_pred).then(1.0).otherwise(0.0))
        .when(q_undef)
        .then(pl.lit(_NAN, dtype=pl.Float64))
        .when(zero_pred)
        .then(pl.lit(_NAN, dtype=pl.Float64))
        .otherwise(corrected)
        .cast(pl.Float64)
    )
    fidelity_yield = (
        pl.when(zero_conf)
        .then(pl.when(zero_pred).then(1.0).otherwise(0.0))
        .when(q_undef)
        .then(pl.lit(_NAN, dtype=pl.Float64))
        .when(zero_pred)
        .then(0.0)
        # min(1, coverage) * fidelity -- the CAPPED-COVERAGE form (spec 2.3), NOT
        # min(fidelity, yield), which flips branch once fidelity < 0 and rewards an
        # under-calling model for calling less.
        .otherwise(pl.min_horizontal(pl.lit(1.0), coverage) * corrected)
        .cast(pl.Float64)
    )

    return comp.with_columns(
        direction_fidelity=fidelity,
        direction_fidelity_raw=fidelity_raw,
        direction_coverage=coverage,
        direction_yield_raw=_safe_div(k, n_conf),
        # The EXPANDED form (spec 2.4): no division by n_pred, so it extends continuously
        # to n_pred = 0 where it evaluates to 0. Valid only for N_conf > 0.
        direction_yield=pl.when(zero_conf | q_undef)
        .then(pl.lit(_NAN, dtype=pl.Float64))
        .otherwise(
            (k.cast(pl.Float64) - q * n_pred.cast(pl.Float64))
            / (n_conf.cast(pl.Float64) * d)
        )
        .cast(pl.Float64),
        direction_fidelity_yield=fidelity_yield,
        direction_fidelity_yield_raw=_safe_div(k, pl.max_horizontal(n_pred, n_conf)),
    )


def _family_reader(column: str):
    """Bind one column of `_fidelity_family` as a public metric function."""

    def metric(
        prepared: PreparedDE | None = None,
        *,
        de_pred=None,
        de_real=None,
        control: str = "non-targeting",
        sort_by: str = "abs_log2_fold_change",
        p_adj_threshold: float = 0.05,
        nan_lfc_policy: str = "mask",
        min_abs_log2fc: float = 0.0,
    ) -> dict[str, float]:
        prepared = _ensure_prepared(
            prepared,
            de_pred,
            de_real,
            control=control,
            sort_by=sort_by,
            p_adj_threshold=p_adj_threshold,
            nan_lfc_policy=nan_lfc_policy,
            min_abs_log2fc=min_abs_log2fc,
        )
        frame = _fidelity_family(prepared)
        return {
            row[0]: _to_float_nan(row[1])
            for row in frame.select(["target", column]).iter_rows()
        }

    return metric


de_direction_fidelity = _family_reader("direction_fidelity")
de_direction_fidelity_raw = _family_reader("direction_fidelity_raw")
de_direction_coverage = _family_reader("direction_coverage")
de_direction_yield = _family_reader("direction_yield")
de_direction_yield_raw = _family_reader("direction_yield_raw")
de_direction_fidelity_yield = _family_reader("direction_fidelity_yield")
de_direction_fidelity_yield_raw = _family_reader("direction_fidelity_yield_raw")


_FAMILY_DOCS = {
    "de_direction_fidelity":
        "Chance-corrected directional accuracy over the reference-defined scored set: "
        "(k/n_pred - q)/d, where q is the reference's majority-sign rate and "
        "d = max(1-q, 0.05). Coverage-weighted companion: de_direction_fidelity_yield.",
    "de_direction_fidelity_raw":
        "Uncorrected k/n_pred. Same arithmetic as the v0.5.0 de_*_direction_precision, "
        "differing only by target-gene exclusion and the n_pred = 0 convention.",
    "de_direction_coverage":
        "n_pred/N_conf -- what fraction of the reference's confident budget the model "
        "called. NOT bounded above by 1, so it is scored with anchor=None and clamped.",
    "de_direction_yield":
        "(k - q*n_pred)/(N_conf*d) -- chance-corrected correct calls per unit of the "
        "reference's budget. Unbounded in BOTH directions, so it is scored with anchor=None (signed: D = |b|) and clamped.",
    "de_direction_yield_raw":
        "k/N_conf. Exceeds 1 whenever the model makes more correct directional calls "
        "than the reference's budget, so it is scored with anchor=None and clamped.",
    "de_direction_fidelity_yield":
        "min(1, n_pred/N_conf) * de_direction_fidelity -- the CAPPED-COVERAGE form, not "
        "min(fidelity, yield). Bounded above by 1, so it uses anchor=1.0 and "
        "denominator D = 1 - b.",
    "de_direction_fidelity_yield_raw":
        "k/max(n_pred, N_conf) -- the uncorrected companion to fidelity_yield. The "
        "max() denominator makes it exactly min(precision, recall) over the scored set: "
        "over-calling is charged through n_pred, under-calling through N_conf. ⚠️ Issue "
        "#259 called that under-calling penalty a defect and it was RULED INTENTIONAL "
        "(2026-08-15) -- a submission that declines to call collapses to ~0 because it "
        "has no recall, which is the answer this metric is supposed to give. The issue is "
        "PARKED with a candidate replacement recorded on it; do not soften this without "
        "reopening that ruling, and note the recorded alternative rewards silence. What "
        "#259 measured as an inversion -- a degenerate tiled arm at 0.508 against an "
        "honest no-skill arm at 0.0002 -- is #234's zero-within-group-variance defect "
        "arriving through this metric, not a property of the ratio.",
}
for _name, _doc in _FAMILY_DOCS.items():
    globals()[_name].__name__ = _name
    globals()[_name].__qualname__ = _name
    globals()[_name].__doc__ = _doc


def de_direction_reach(
    prepared: PreparedDE | None = None,
    *,
    universe: Literal["adjudicated", "all"] = "adjudicated",
    corrected: bool = True,
    de_pred=None,
    de_real=None,
    control: str = "non-targeting",
    sort_by: str = "abs_log2_fold_change",
    p_adj_threshold: float = 0.05,
    nan_lfc_policy: str = "mask",
    min_abs_log2fc: float = 0.0,
    purity_floor: float | None = None,
) -> dict[str, float]:
    """Depth to which the prediction's ranking stays directionally pure, as a fraction of
    the reference's confident budget: ``k*(t) / N_conf(t)``, with the target gene excluded
    from both the ranking pool and N_conf.

    ``corrected=True`` thresholds at ``P0 = q + (1-alpha)(1-q)`` -- the majority-sign-null
    threshold. Spec 2.5 derives it: the DOR defines the threshold on the CORRECTED curve
    ``P*(k) = (P(k) - q)/(1 - q)`` at ``1 - alpha``, which for ``q < 1`` is algebraically
    identical to thresholding the RAW curve at ``q + (1-alpha)(1-q)``. Implementing the
    raw form leaves `_purity_curve` untouched and never divides by ``1 - q``, so ``q -> 1``
    needs no floor (the DOR confirms reach applies none); at ``q = 1`` the threshold
    becomes 1.0, which is a stated limiting CONVENTION, not a consequence of the identity.

    ⚠️ ``P0 = 1 - alpha`` on the corrected curve is the majority-sign null -- conservative
    against a marginal-exploiting model, NOT a universal identity. Under the DOR's
    alternative null, which draws predicted and true signs independently with probability
    ``r`` of predicting the truth-side majority sign, agreement is ``(1-q) + r(2q-1)``, and
    at ``r = q`` the ideal corrected purity is 0.942, not 0.95.

    ``corrected=False`` thresholds at ``P0 = REACH_PURITY_FLOOR = 0.9``, or at
    ``purity_floor`` when one is given. Raw reach never reads ``q``.

    ``purity_floor`` exists so a sensitivity sweep can drive the SHIPPED metric (#327). The
    calibration behind ``0.9`` could not: ``internal:tools/reachval/purity_threshold_sweep.py`` had to
    import ``_purity_curve``, ``_k_star``, ``_ontarget_excluded_frame`` and
    ``_reference_stats`` and thread its own ``p0`` through them, because this entry point had
    no way in -- so the number that moved a scored member came from a hand-reassembled copy
    of the metric rather than from the metric.
    It is a FUNCTION argument only -- nothing in ``EvalConfig``, the CLI or the catalog can
    set it, so no scored run can reach it. ``competition_payload()`` freezes the resolved
    default, so the rule digest now sees what this member computes (#317).

    ⚠️ That constant is CALIBRATED (Alex, 2026-08-17), not derived, and it replaces the
    v0.5.0 rule ``P0 = 1 - alpha/2 = 0.975``. The old derivation -- a false discovery is a
    true null, so its sign in an independent repeat is a coin flip, and an FDR-alpha set
    therefore SUSTAINS a purity of ``1 - alpha/2`` -- is sound, but it computes the purity a
    PERFECT independent repeat attains against an alpha-FDR reference. That is the metric's
    own CEILING, and using it as the pass mark left zero margin. Measured on the three
    official val lines, a real split-half replicate's per-gene directional accuracy over the
    reference-significant adjudicable set is 0.9561 / 0.9391 / 0.9488 (mean over targets) --
    BELOW the old threshold on every line, so a real experimental replicate failed it.

    Three consequences of the old value, all measured
    (internal:docs/validation/2026-08-17-direction-reach-purity-threshold.md):

    * The first depth tolerating one sign error is ``ceil(1/(1-P0))`` = 40 at 0.975 and 10 at
      0.9, so 46-61% of the 300-target val cohort had NO depth at which a single error was
      survivable, against 12-30% now.
    * At a fixed, uniform 95% per-gene accuracy the cohort mean varied 13.5x across N_conf
      strata at 0.975 (0.756 at <= 10 significant genes, 0.056 above 500) and varies 1.3x at
      0.9 -- i.e. the old threshold graded identical quality by the reference's budget size.
    * `from_replicate` of a 90%-accurate submission moves 0.24/0.43/0.30 -> 0.75/0.91/0.76.

    What the change does NOT do: it barely moves the chance floor (a random-sign predictor
    reads 0.042694 / 0.065849 / 0.080890 at 0.975 and 0.042694 / 0.067557 / 0.082166 at 0.9 -- val A EXACTLY unmoved, B +0.0017, C +0.0013, i.e. NEARLY inert rather than
    unmoved: 0.24% of val B's own span), and it does not saturate the anchor (the replicate
    reads 0.965/0.802/0.943, held down PRIMARILY -- not exclusively -- by the small-budget
    targets a lower floor cannot help; the N_conf <= 10 stratum is 45/60/75% of the shortfall
    on A/B/C and every stratum sits below 1). The baseline arm
    gains only +0.004 to +0.009 while the replicate gains +0.039 to +0.050, so the score's
    span WIDENS by 4-5% rather than collapsing. The cost is a 16.0-31.5% relative gain for a
    marginal-sign exploit (31.5 / 21.3 / 16.0 on val A/B/C) -- measured against an arm that also holds the reference's own
    ``|log2FC|`` as its ranking key, so that figure is an upper bound.

    ⚠️ `de_direction_sensitivity` KEEPS ``1 - alpha/2``, so the two metrics no longer agree.
    That identity (raw reach == v0.5.0 sensitivity on a fixture with no target-gene row) held
    only because they shared a threshold; sensitivity is one of the three v0.5.0 metrics whose
    values must not move (spec 1/3), and it is diagnostic-only, so it is deliberately left
    alone. The ``corrected=True`` branch above keeps its own ``1 - alpha`` derivation for the
    same reason: it is a different null, and it is not a `vcc2026` member.

    ⚠️ Issue #279 -- ``reach_raw``'s no-skill point is ``~c/N_conf``, NOT a constant, and for
    a no-skill submission it is decided almost entirely at the HEAD of a ranking the
    submission controls. A first-pair match is SUFFICIENT for ``k* >= 1`` (purity 1/1 >= P0),
    one coin flip independent of depth, and it dominates the no-skill probability.

    ⚠️ It is NOT necessary, and #279's own "iff" is wrong. Purity can recover: a leading
    miss followed by 9 matches gives purity(10) = 9/10 = 0.9 >= P0, so k* reaches 10 -- built
    end to end, that fixture returns reach_raw = 1.0, not 0. The empirical claim survives
    because the no-skill mean is dominated by the ``k* = 1`` event (one coin flip), not by
    recovery: at P0 = 0.9 the EARLIEST recovery opportunity is depth 10, which qualifies on at
    least 9 of the first 10 calls -- 11/1024 = 1.1% for a coin flip -- and the resulting
    ``10/N_conf`` is small against the budget. ⚠️ That is the probability the depth-10 PREFIX
    qualifies, not of recovery in general: 8/10 fails at 10 and can still qualify at 20 on
    18/20, so it is a lower bound on total recovery. At 0.975 the corresponding first
    opportunity is depth 40 on at least 39 of 40, 41/2**40 = 3.7e-11 -- eight orders of
    magnitude rarer, which is why lowering the floor left the measured chance floor NEARLY
    unmoved: A exactly, B and C up by 0.0017 and 0.0013. The DERIVATION was overstated at
    either threshold.

    Measured at 0.13.0 over 200 replicates per cell: ``E[reach_raw]`` runs 0.5100 at
    ``N_conf = 1`` to 0.0019 at 500 with ``P(reach_raw > 0) = 0.385-0.565`` at EVERY N_conf,
    and the fitted ``c`` is ~0.96 for a coin-flip predictor against ~1.90 for one that ranks a
    single HABITUAL-DIRECTION gene first -- which §6a notes is close to free.

    The CALIBRATION half of that issue is closed by #276 part C: scoring is against a measured
    baseline, and on official val A the baseline arm reads 0.042314 against an ESTIMATED
    no-skill 0.0378, so the baseline already sits at the no-skill point. The EXPLOIT half
    survives at ~+0.038 of the member's span -- val A's span here is the frozen replicate
    anchor 0.904381 over that baseline, NOT [0, 1] -- i.e. **+0.006 on avg_score**. ⚠️ The
    0.0748 behind that is an estimate from the fitted c/N_conf law applied to val A's own
    N_conf distribution, not a direct simulation on the panel. What bounds it is the panel
    rather than the metric: val A's smallest N_conf is 4, so the N_conf <= 3 band where the
    head flip is worth 0.29-0.51 is empty. Removing it needs a ``min_n_conf`` gate or a
    ``k* >= 2`` rule; both move scored numbers.

    ``universe='adjudicated'`` ranks only reference-significant genes, so ``k* <= N_conf``
    and the result is bounded by 1. ``universe='all'`` ranks the whole shared universe and
    is UNBOUNDED ABOVE, so it is scored with ``anchor=None`` and clamped to ``[-2, 2]``
    rather than normalized against a constant perfection point.

    Depth counts ADJUDICABLE pairs, the same population purity's denominator counts
    (spec 2.6, as amended by issue #204): a pair the reference cannot adjudicate carries
    no directional evidence and neither extends the prefix nor affects purity.

    ⚠️ This is a CORRECTION. Depth previously counted ranked ROWS, and the docstring here
    claimed that was deliberate because it made ``precision == purity(k = |S_pred|)``
    exact. The identity was real but the justification did not hold: purity is unchanged
    by the fix, so the identity survives re-indexed to
    ``k = |S_pred INTERSECT adjudicable|``, while row-counting depth made the
    ``universe='all'`` variants ~98% reference sparsity. See `_purity_curve`.
    """
    prepared = _ensure_prepared(prepared, de_pred, de_real, control=control,
                                sort_by=sort_by, p_adj_threshold=p_adj_threshold,
                                nan_lfc_policy=nan_lfc_policy,
                                min_abs_log2fc=min_abs_log2fc)
    _require_resolution(prepared)   # spec 5's precondition, at the metrics that own it
    _warn_coverage_once(prepared)   # issue #291's diagnostic, at the metrics it describes
    alpha = prepared.p_adj_threshold
    # `p_adj_threshold` is caller-supplied and need not be a Python float: a Decimal makes
    # `1.0 - alpha` raise TypeError, and a numpy scalar propagates its dtype into the
    # threshold. Only the CORRECTED branch still does that arithmetic (raw reach reads the
    # constant `REACH_PURITY_FLOOR`), so the `float()` normalization now lives there. Everything downstream compares against a Float64 polars column, so a
    # float is the right type there; `float()` is exact and hence a no-op for an ordinary
    # float threshold. ⚠️ `alpha` itself stays as given, because `_purity_curve` passes it
    # to a polars filter -- and the v0.5.0 `de_direction_sensitivity` is deliberately NOT
    # given the same treatment: it raises identically on a Decimal on `origin/main`
    # (measured), so "fixing" it here would change a v0.5.0 metric this PR must leave
    # untouched. That belongs to #196.
    curve = _purity_curve(_ontarget_excluded_frame(prepared), universe=universe, alpha=alpha)
    stats = _reference_stats(prepared)

    if corrected:
        # Scoped to THIS branch, not hoisted: it is the only arithmetic on alpha left, and
        # hoisting it would make `test_raw_reach_does_no_arithmetic_on_alpha` pass whether or
        # not the raw path is alpha-free -- `float()` swallows the Decimal either way.
        alpha_f = float(alpha)
        p0: float | dict[str, float] = {
            t: q + (1.0 - alpha_f) * (1.0 - q)
            for t, q in zip(stats["target"].to_list(), stats["q"].to_list(), strict=True)
            if q is not None
        }
    else:
        # NOT alpha-derived any more (see REACH_PURITY_FLOOR): the raw branch does no
        # arithmetic on alpha at all, so a Decimal or numpy `p_adj_threshold` can no longer
        # raise here. (`_purity_curve` always received the ORIGINAL `alpha`, never `alpha_f`;
        # what changed is that this branch no longer computes anything from it. `alpha_f` is
        # deliberately not in scope.)
        # `purity_floor` overrides the calibrated default for CALIBRATION WORK ONLY (#327).
        # It is deliberately NOT reachable from `EvalConfig` or the CLI: a per-run knob that
        # moved a scored `vcc2026` member under an unchanged rule digest would let two
        # submissions scored at different floors compare as if they were comparable, and let a
        # bundle built at one floor score a submission computed at another with nothing
        # raising. `_register_de_family` registers this function with no override, so every
        # CATALOG-reachable spelling resolves to `REACH_PURITY_FLOOR` -- which is why the three
        # semantics surfaces (`run._result_config_digest`, `partition.result_semantics`,
        # `anchor.anchor_semantic_params`) can go on stamping the module default, and why
        # `_reach_floor_used`'s function-identity key stays sound.
        p0 = REACH_PURITY_FLOOR if purity_floor is None else float(purity_floor)
    k_star = _k_star(curve, p0=p0)

    out: dict[str, float] = {}
    for target, n_conf, q in zip(stats["target"].to_list(), stats["n_conf"].to_list(),
                                 stats["q"].to_list(), strict=True):
        # Spec 5: "never reached P0 -> a computed 0.0" is DOMAIN-RESTRICTED and must not
        # override the matrix. It is trivially true at N_conf = 0 (empty pool) and when q
        # is undefined (no threshold exists) -- both are NaN, not 0.
        if n_conf == 0 or (corrected and q is None):
            out[target] = _NAN
        else:
            out[target] = _to_float_nan(k_star.get(target, 0) / n_conf)
    return out
