from __future__ import annotations

import functools
from dataclasses import dataclass
from typing import Callable, Literal

from .metrics.delta import (
    distance_unbiased,
    mae,
    mae_delta,
    mse,
    mse_delta,
    mse_unbiased,
    mse_unbiased_capped,
    pearson_delta,
    real_mass_ratio,
)
from .metrics.de import (
    de_direction_match,
    de_lfc_nmae,
    de_lfc_spearman,
    de_nsig_counts,
    de_nsig_spearman,
    de_overlap,
    de_overlap_adjusted,
    de_pr_auc,
    de_roc_auc,
    de_sig_agreement,
    de_sig_jaccard,
    de_sig_recall,
    de_model_direction_match,
)
from .metrics.direction import (
    de_direction_coverage,
    de_direction_fidelity,
    de_direction_fidelity_raw,
    de_direction_fidelity_yield,
    de_direction_fidelity_yield_raw,
    de_direction_precision,
    de_direction_reach,
    de_direction_sensitivity,
    de_direction_yield,
    de_direction_yield_raw,
)
from .metrics.discrimination import discrimination_score
from .scoring import BOUNDED, BOUNDED_UNFLOORED, DIAG, ERROR, ERROR_LINEAR, Scoring


@dataclass(frozen=True)
class DerivedAgg:
    """A metric whose AGGREGATE is a ratio of two other metrics' per-perturbation sums.

    Both names are canonical catalog keys. The metric itself emits no per-perturbation
    rows: `Σ num / Σ den` is not the mean of anything the frame could carry, and a column
    whose rows do not average to its own aggregate is a trap that gets reintroduced
    downstream forever (see issue #233, where a downstream consumer averages whatever it finds).
    """

    numerator: str
    denominator: str


@dataclass(frozen=True)
class MetricSpec:
    name: str                                # canonical v2 label; CATALOG dict key
    func: Callable | None                    # None <=> derived; see `derived` below
    scoring: Scoring                         # anchor + direction + enrolment (see scoring.py)
    profiles: tuple[str, ...]
    kind: Literal["de", "anndata"]
    normalization: Literal[
        "counts", "normalized", "lognorm", "bulk_lognorm", "expr_comparator"
    ] | None
    agg: Literal["mean", "median", "ratio_of_sums"]   # REQUIRED: no silent default
    v1_name: str | None = None               # inherited byte-compat label (v1 output + input alias)
    aliases: tuple[str, ...] = ()            # any other accepted-on-input spellings
    worst_value: float | None = None         # v2 no-droppable-NaN target (issue #89); None = leave output as-is
    # False -> excluded from v1 output entirely (issue #195). Named `v1_available` rather
    # than `v2_only` because `_register_de_family` already binds `v2_only` as its family flag.
    #
    # `_register_de_family` DERIVES this from `v1_name` (a metric with no upstream cell-eval
    # name has nothing to be byte-compatible with). An earlier comment here argued the
    # opposite -- that deriving it would remove metrics v1 emits and break the compat gate.
    # It does remove some, and that turned out to be the correction, not the risk: what it
    # removes are v2-native metrics that were reaching v1 output only because nobody passed
    # the flag. The compat gate is unaffected because those metrics have no upstream
    # counterpart to be compared against; `test_v1_gate` pins the v1-emitted set directly.
    #
    # `None` (the default) means DERIVE it, which is what makes the invariant structural
    # rather than a convention `_register_de_family` happens to follow: a spec built anywhere
    # else with no v1 name is v2-native too, and cannot leak into v1 by omission. An explicit
    # `False` is still honoured, so a metric that HAS an upstream name can still be withheld
    # from v1; no entry needs that today. Explicit `True` with no v1 name is refused -- there
    # would be no name to emit it under.
    v1_available: bool | None = None
    # True -> this metric consumes per-group moments (counts + Σ‖x‖²) and the runner must ask
    # its pseudobulk driver for them (issue #198). Declared rather than derived from the
    # function signature: this boolean selects which cache artifact a run reads, and a wrong
    # derivation would change that silently.
    needs_moments: bool = False
    derived: DerivedAgg | None = None        # set <=> func is None

    def __post_init__(self) -> None:
        # Truthiness, not `is not None`: `_build_name_index` skips "" as an invalid spelling,
        # so an empty v1_name is no more emittable than a missing one and must not derive
        # availability from it.
        if self.v1_available is None:
            object.__setattr__(self, "v1_available", bool(self.v1_name))
        elif self.v1_available and not self.v1_name:
            raise ValueError(
                f"{self.name!r}: v1_available=True requires a non-empty v1_name -- there is "
                "no label to emit it under. Pass a v1_name, or leave v1_available unset to "
                "derive it."
            )
        # Exactly one of the two ways a metric produces a value. A spec with both would
        # have an unreachable `func`; a spec with neither is a metric that cannot run.
        if (self.func is None) == (self.derived is None):
            raise ValueError(
                f"metric {self.name!r}: exactly one of `func` and `derived` must be set; "
                f"got func={self.func!r}, derived={self.derived!r}"
            )
        # The statistic and the mechanism are two fields that must not drift apart:
        # `metric_agg` drives both aggregators and the metric_aggregation.csv sidecar, so a
        # derived metric declaring "mean" would be aggregated as a mean of rows it does not
        # have -- silently NaN rather than loudly wrong.
        if (self.agg == "ratio_of_sums") != (self.derived is not None):
            raise ValueError(
                f"metric {self.name!r}: agg='ratio_of_sums' and `derived` imply each other; "
                f"got agg={self.agg!r}, derived={self.derived!r}"
            )
        if self.derived is not None and self.needs_moments:
            raise ValueError(
                f"derived metric {self.name!r} must not set needs_moments: it never runs, so "
                "the flag would route a cache artifact to nothing. Its COMPONENTS declare it."
            )
        # Spec §4.4: the per-perturbation machinery must refuse a derived metric EXPLICITLY
        # rather than skip it and leave a field that reads as configured. `worst_value` drives
        # `run.py`'s no-drop fill, which iterates per-perturbation rows a derived metric does
        # not have -- so a finite value here would be silently inert, which is exactly the
        # vacuous-guard failure #219 shipped.
        if self.derived is not None and self.worst_value is not None:
            raise ValueError(
                f"derived metric {self.name!r} must leave worst_value=None: it emits no "
                "per-perturbation rows, so the no-drop fill has nothing to fill and any value "
                "here would be silently inert."
            )

    @property
    def best_value(self) -> Literal["zero", "one", "none"]:
        """DEPRECATED, derived, read-only. ``scoring`` is the authoritative fact; nothing in
        ``src/`` reads this any more (``compat.score_agg_metrics`` was the last consumer and
        now reads the policy directly). Kept only so out-of-tree consumers and the ``tools/``
        scripts keep working.

        It is LOSSY, which is why it was retired from the live path: three tokens cannot
        express the four states ``scoring`` distinguishes. In particular an anchorless SCORED
        metric reports ``"one"`` here, and a consumer that reads that as "anchor is 1" will
        normalize it against the wrong denominator. Read ``scoring.anchor`` instead."""
        if not self.scoring.scored:
            return "none"
        return "zero" if self.scoring.direction == "lower" else "one"


def is_decisive(spec: "MetricSpec") -> bool:
    """Whether a degenerate baseline for this metric must FAIL LOUD rather than be skipped.

    True for anything a v1 run can emit or that either competition profile (``vcc`` or
    ``vcc2026``) scores -- the metrics where a wrong number decides a ranking, and where
    scoring every submission against an undefined denominator is worse than stopping. False
    for any other metric, which is skipped with a warning instead of aborting the run (see
    ``score.score_metrics``).

    That second half is a RULE and the class it selects is LARGE -- every scored
    ``de_deseq2_*`` sibling (reachable only under ``de.backend="deseq2"``) and the v2-native
    ``de_wilcoxon_*`` entries outside ``vcc2026``. Do not read that list as exhaustive;
    derive the members from the predicate. (No count on purpose:
    it goes stale on every enrolment.) ``de_*_direction_yield`` has a baseline that is
    legitimately degenerate rather than corrupt: it is signed and centred at zero by
    construction, so an exactly-zero baseline is reachable from a legitimate baseline run.

    ONE definition. ``score_metrics`` and the ``score`` CLI precheck BRANCH on it -- two
    copies would drift, and the CLI's copy drifting is not hypothetical: it aborted on every
    degenerate metric, so the graceful path in the scorer was unreachable through
    ``cell-eval2 score``. ``baseline`` only ANNOTATES with it (each offender record carries
    ``decisive``, and ``_degenerate_message`` words itself accordingly); the build gate still
    refuses to write any degenerate aggregate without ``allow_degenerate``.

    ``vcc`` is subsumed by ``v1_available`` today; both are named so a future v2-native metric
    added to ``vcc`` still fails loud.

    ``vcc2026`` IS in this disjunction as of #255. It was deliberately excluded before, on
    the grounds that ``expr_mse_unbiased_norm`` was bias-corrected and centred near 0, so a
    ``base <= 0`` was "reachable from a legitimate baseline run" and adding the profile
    would make ``full``/``anndata`` runs abort on an outcome that is normal there.

    ``scored`` gates the whole disjunction as of #257. An UNSCORED metric is never compared
    against a baseline, so a degenerate baseline for it cannot decide a ranking and aborting
    a run over one is a false alarm. This became reachable when #257 put three unscored
    diagnostics into ``vcc2026`` -- a derived metric's components must sit in every profile
    it claims. ``test_the_scored_term_moves_exactly_the_unscored_gate_members_and_nothing_else``
    pins which metrics the term moves.

    Two metrics shipped before #257 move with it -- ``de_wilcoxon_nsig_counts_real`` and
    ``de_wilcoxon_nsig_counts_pred``, unscored but v1-emitted -- and the move is inert: every
    consumer skips unscored metrics before asking. ``score_metrics`` continues on
    ``not policy.scored``, and ``baseline._degenerate_metrics`` (whose output the ``score``
    CLI precheck reads) skips them before recording ``decisive``.

    MEASURED, that does not hold. The deployed generic-response baseline reads 169.16
    (``CCL_2``) and 197.05 (``H1_CGS``) on ``expr_mse_unbiased_norm``, against a degeneracy
    threshold of 0 -- its WORST single perturbation is still 17x above the threshold, and 0%
    of perturbations land at or below it. The 65.7%-negative figure the old argument relied
    on describes the metric on an ACCURATE PREDICTOR (a real technical replicate), not on
    the generic-response comparator, which carries no target-specific information at all.
    Those are different populations. See
    ``internal:docs/validation/2026-08-06-vcc2026-rescale-anchors.md``.

    The change is also right on the merits, not only on the measurement: when ``D <= 0`` the
    metric cannot be scored against that baseline in ANY profile, so failing loud in
    ``full`` is correct behaviour rather than collateral damage.

    RESIDUAL RISK, accepted: a future comparator better than generic-response could reach
    ``D <= 0`` and abort a ``full`` run. #222 stays open for the profile-aware
    ``is_decisive(spec, profile)`` resolution -- decisiveness is still a property of the
    (metric, profile) pair, and this predicate still reads only the spec.
    """
    return bool(spec.scoring.scored) and (
        bool(spec.v1_available)
        or "vcc" in spec.profiles
        or "vcc2026" in spec.profiles
    )


def _disc(distance: str) -> Callable:
    """Bind the distance for a named discrimination variant. The remaining params
    (rank_denominator, tie_policy, exclude_target_gene, exclusion_scope, control_source,
    genes, target_gene_map, embed_key) are supplied at dispatch time from EvalConfig — see run.compute_metrics."""
    return functools.partial(discrimination_score, distance=distance)


CATALOG: dict[str, MetricSpec] = {
    "expr_mae": MetricSpec(
        name="expr_mae", v1_name="mae", func=mae, scoring=ERROR, agg="mean",
        profiles=("full", "minimal", "vcc", "anndata"), kind="anndata",
        normalization="expr_comparator",
    ),
    "pds_l1": MetricSpec(
        name="pds_l1", v1_name="discrimination_score_l1", func=_disc("l1"), scoring=BOUNDED,
        agg="mean",
        profiles=("full", "minimal", "vcc", "anndata", "pds"), kind="anndata", normalization="expr_comparator",
    ),
    "pds_l2": MetricSpec(
        name="pds_l2", v1_name="discrimination_score_l2", func=_disc("l2"), scoring=BOUNDED,
        agg="mean",
        profiles=("full", "anndata"), kind="anndata", normalization="expr_comparator",
    ),
    # `BOUNDED_UNFLOORED`, not `BOUNDED`: one of the four bounded `vcc2026` members whose
    # clip-at-0 was removed. Its comparator sits at the metric's chance point (a uniform rank
    # is 0.5), so the clip was truncating a full unit of below-comparator range -- measured
    # -1.17/-1.26/-1.22 on the three official val bundles' replicate scale (the PRE-#172
    # `-r1` set -- `pds_cosine` itself is untouched by #172, but see the note in scoring.py).
    # ⚠️ The v1 path is UNAFFECTED: `compat.score_agg_metrics` carries its own hard-coded
    # `max(0.0, score)` and never reads `clamp_low`, so this metric's frozen v1 number stands.
    "pds_cosine": MetricSpec(
        name="pds_cosine", v1_name="discrimination_score_cosine", func=_disc("cosine"),
        scoring=BOUNDED_UNFLOORED, agg="mean",
        profiles=("full", "anndata", "vcc2026"), kind="anndata", normalization="expr_comparator",
    ),
    "delta_pearson": MetricSpec(
        name="delta_pearson", v1_name="pearson_delta", func=pearson_delta, scoring=BOUNDED,
        agg="mean",
        profiles=("full", "minimal", "anndata"), kind="anndata", normalization="expr_comparator",
        worst_value=-1.0,
    ),
    "expr_mse": MetricSpec(
        name="expr_mse", v1_name="mse", func=mse, scoring=ERROR, agg="mean",
        profiles=("full", "minimal", "anndata"), kind="anndata",
        normalization="expr_comparator",
    ),
    # The three-column split (issue #257). `expr_mse_unbiased_norm` declared a no-skill point
    # of 1.0 that it did not have: its numerator was debiased on both sides while its
    # denominator was a plug-in carrying tr Sigma_p/n_real + tr Sigma_ctrl/N_ctrl, so a
    # control-emitting submission read `1 - noise/D` -- measured 0.7643 on VCC Test, 0.2386 on
    # CCL_2, 0.2754 on H1_CGS. The gap is a property of the reference panel's depth, so
    # `low-random_high-1_v1` credited a do-nothing submission 0.24-0.76 of its full range.
    #
    # These three are DIAGNOSTICS and must never be scored: gene-averaged expression units are
    # panel-dependent (TODO #2's original complaint), and the uncapped value is the
    # submitter lever #247 closed. TWO of them -- the capped numerator and the distance --
    # are the derived metric's actual components; `expr_mse_unbiased` rides along so the cap
    # is auditable. They are in `vcc2026` because a derived metric's components
    # must sit in every profile it claims, which `is_decisive` now tolerates by requiring
    # `scored`.
    "expr_mse_unbiased": MetricSpec(
        name="expr_mse_unbiased", v1_name=None, func=mse_unbiased, agg="mean", scoring=DIAG,
        profiles=("full", "anndata", "vcc2026"), kind="anndata",
        normalization="expr_comparator",
        worst_value=None, needs_moments=True,
    ),
    "expr_mse_unbiased_capped": MetricSpec(
        name="expr_mse_unbiased_capped", v1_name=None, func=mse_unbiased_capped, agg="mean",
        scoring=DIAG, profiles=("full", "anndata", "vcc2026"), kind="anndata",
        normalization="expr_comparator", worst_value=None, needs_moments=True,
    ),
    "expr_distance_unbiased": MetricSpec(
        name="expr_distance_unbiased", v1_name=None, func=distance_unbiased, agg="mean",
        scoring=DIAG, profiles=("full", "anndata", "vcc2026"), kind="anndata",
        normalization="expr_comparator", worst_value=None, needs_moments=True,
    ),
    "expr_real_mass_ratio": MetricSpec(
        name="expr_real_mass_ratio", v1_name=None, func=real_mass_ratio, agg="mean",
        scoring=DIAG, profiles=("full", "anndata", "vcc2026"), kind="anndata",
        normalization="expr_comparator", worst_value=None,
    ),
    # The scored metric. It has NO per-perturbation column: `Σ num / Σ den` is not the mean of
    # anything, and a column whose rows do not average to its own aggregate is a trap (#233).
    # `anchor=0.0` and `direction="lower"` are unchanged -- 0 is still perfection -- and 1.0 is
    # now genuinely the no-skill point on every panel whatever the REFERENCE's depth, not a
    # declaration. ⚠️ As of #348 that point also needs `r = 1`, i.e. the panel's claimed
    # prediction-side correction not exceeding its own across-perturbation budget: an arm emitting
    # an INDEPENDENT draw per perturbation is at r ~ 0.97 and reads 1.0 to within that, while one
    # reusing ONE emitted cell block for every perturbation has B = 0, forfeits the correction and
    # reads above 1 (deliberately -- see `metrics/delta.py::_numerator`). Neither moves a SCALED
    # score: both already sit at or past the no-skill end. ⚠️ EXACT for the ORACLE form in the true variance traces, approximate for the
    # shipped estimator -- and for THIS member twice over, since the cap clips and
    # E[min(C_pred_hat, k C_real_hat)] != min(E[C_pred_hat], k * E[C_real_hat]), so even an unbiased
    # correction would leave the capped anchor inexact. On the bias itself: under
    # `bulk_lognorm` the subtracted correction is a jackknife with a MEASURED upward bias of
    # 0.32% at the shipped bulk_target_sum=5e4 (#268; 2.06% at the retired 1e6, where the same
    # panel's "predict the control" anchor read 1.073 instead of 1.0). It is NOT invariant to
    # the SUBMISSION either: a prediction
    # whose own correction exceeds k times the reference's -- fewer cells OR more dispersed
    # ones -- reads above 1, because #247's cap refuses a correction the reference does not
    # earn. ⚠️ "Above 1" is the matched-iid no-skill ORACLE calculation, not a pointwise rule.
    # Pointwise the cap inequality decides only whether the cap BINDS. #278 showed the
    # DISPERSION half inverting under `bulk_lognorm`: at fixed group totals the squared-error
    # term cannot move, so a MORE dispersed submission reads LOWER than it otherwise would
    # (until saturation) -- which need not put it on either side of 1. The trigger is also not
    # the cell count. See `metrics.delta.mse_unbiased_capped`, which carries the full correction.
    # `clamp_high=1.0` stays load-bearing: the value is signed, so a paste overshoots.
    # `clamp_low=0.0` since #276 part C (Alex, 2026-08-13): this is the ONE scored vcc2026
    # member that clamps to [0, 1], on BOTH scales. The metric is capped so that 1.0 =
    # predicting the control, so a value past the baseline is already no-skill and the graded
    # Box-Cox tail below it buys nothing a ranking uses. The other five members keep
    # clamp_high=None and may exceed 1 -- that asymmetry was asked about explicitly and ruled.
    "expr_mse_unbiased_capped_norm": MetricSpec(
        name="expr_mse_unbiased_capped_norm", v1_name=None, func=None, agg="ratio_of_sums",
        derived=DerivedAgg(numerator="expr_mse_unbiased_capped",
                           denominator="expr_distance_unbiased"),
        scoring=Scoring(scored=True, direction="lower", anchor=0.0,
                        penalty="boxcox", clamp_low=0.0, clamp_high=1.0),
        profiles=("full", "anndata", "vcc2026"), kind="anndata",
        normalization="expr_comparator",
        worst_value=None,
    ),
    # ⚠️ These two are ALIASES of `expr_mse`/`expr_mae` under the v2 default
    # `control_source="real"` (issue #189) -- the same quantity per perturbation, because the
    # control cancels in any difference-based error: (pred - c) - (real - c) = pred - real.
    # (Equal to roundoff, not bit-for-bit: the two evaluation orders differed by 1-2 ULP on 91 of
    # 600 pairs in one measured sweep -- evidence, not a bound. Reassociation, not a signal.)
    # `full` scores all four, which is fine for a diagnostic profile but means an equal-weight
    # aggregate over that set double-weights one quantity; do not build one. Under v1
    # (`control_source="pred"`) they DO differ -- by the constant vector (pred_ctrl - real_ctrl),
    # the same for every perturbation, so one varying signal plus a fixed offset rather than two
    # signals. Kept rather than removed: dropping them loses that v1 variant and breaks the
    # catalog count assertions and every stored baseline carrying these columns. `delta_pearson`
    # above is NOT redundant -- correlation is not translation-invariant, so subtracting the
    # control genuinely changes it. Pinned by `tests/test_delta_expr_identity_189.py`; stated in
    # `docs/metrics.md` §2.4.
    "delta_mse": MetricSpec(
        name="delta_mse", v1_name="mse_delta", func=mse_delta, scoring=ERROR, agg="mean",
        profiles=("full", "anndata"), kind="anndata", normalization="expr_comparator",
    ),
    "delta_mae": MetricSpec(
        name="delta_mae", v1_name="mae_delta", func=mae_delta, scoring=ERROR, agg="mean",
        profiles=("full", "anndata"), kind="anndata", normalization="expr_comparator",
    ),
}


def _de_overlap(k, metric):
    return functools.partial(de_overlap, k=k, metric=metric)


def _register_de_family(catalog: dict[str, MetricSpec], *, method: str, v2_only: bool) -> None:
    """Register the DE metric family under the ``de_<method>_*`` namespace.

    Called once for ``wilcoxon`` (the rank backends' family, with its inherited v1 names +
    profile tags) and once for ``deseq2`` (``v2_only=True`` -> ``v1_name=None`` and
    ``profiles=()``; reached only via the backend relabel in ``run.dispatch_de_metrics``,
    never auto-selected by a profile). Same metric functions either way -- the funcs consume
    a DE table and are method-agnostic. The chance-corrected agreement metrics (issue #14)
    are v2-native (``v1_name=None``) even in the wilcoxon family."""
    pre = f"de_{method}_"

    def add(suffix, *, v1, func, scoring, profiles, agg, worst=None, v1_avail=True):
        name = pre + suffix
        catalog[name] = MetricSpec(
            name=name, func=func, scoring=scoring, agg=agg,
            profiles=(() if v2_only else profiles), kind="de", normalization=None,
            v1_name=(None if v2_only else v1), worst_value=worst,
            # `None` -> MetricSpec derives it from the v1_name computed just above, so a
            # metric with no upstream cell-eval equivalent is never offered under
            # version="v1". Derived rather than hand-flagged because that is how the four
            # #14 chance-corrected metrics and the three v0.5.0 direction metrics ended up
            # v1-available: nobody passed the flag. `v1_avail=False` stays honoured, but it
            # is now redundant wherever v1 is already None.
            v1_available=(None if v1_avail else False),
        )

    for metric in ("overlap", "precision"):
        for k in (None, 50, 100, 200, 500):
            v1 = f"{metric}_at_{k if k else 'N'}"                 # inherited (v1)
            suffix = metric + (f"_top{k}" if k else "")           # de_<method>_overlap[_topK]
            if k is None:
                profiles = ("full", "minimal", "de") + (("vcc",) if metric == "overlap" else ())
            else:
                profiles = ("full", "de")
            add(suffix, v1=v1, func=_de_overlap(k, metric), scoring=BOUNDED, agg="mean",
                profiles=profiles)

    # Every directional metric here is SCORED; only the two nsig counts are not, because
    # they alone have no direction (neither more nor fewer significant genes is better).
    # Ranges are from the #195 design 2.2; (1-q)/d equals 1 whenever q <= 0.95, which is the
    # d-floor's intended regime.
    #
    # These are all v2-native (v1_name=None -> v1_available False), so enrolling them cannot
    # reach compat.score_agg_metrics or any v1 output; the frozen v1 path is unaffected.
    _NO_DIRECTION = DIAG                    # nsig counts: no better/worse, so nothing to score
    _UP_BOUNDED = BOUNDED                   # [0, 1] with anchor 1 -- the same policy as the
                                            # rank metrics, reached by a different route
    # `_UP_BOUNDED` with the clip-at-0 removed, for the `vcc2026` members only. A [0, 1]
    # metric cannot produce an unbounded score, so the floor only ever protected against a
    # MISSING value -- which `metric_min=0.0` now covers exactly. See `scoring.BOUNDED_UNFLOORED`.
    _UP_UNFLOORED = BOUNDED_UNFLOORED
    # No constant anchor: `D = b` and the score is u/b - 1, which the metric's own range does
    # not bound above the way an anchor does. Clamped to [-2, 2] (Alex, 2026-08-02) so one
    # near-zero baseline cannot dominate avg_score: at u/b = 100 the raw score is 99, which
    # would move a profile's aggregate by more than the entire range of every other metric.
    # (No metric count here on purpose -- it goes stale on every enrolment.)
    # NOTE the floor is inert for these five: u >= 0 and b > 0 give score >= -1 already.
    _UP_OPEN = Scoring(scored=True, direction="higher", anchor=None,
                       clamp_low=-2.0, clamp_high=2.0)
    # direction_yield is the ONLY signed metric: (k - q*n_pred)/(N_conf*d) is unbounded in
    # BOTH directions, so a negative baseline is data, not corruption (spec 2.2), and D = |b|.
    # Here BOTH clamps bind -- the score is genuinely two-sided, and this baseline is centred
    # near zero by construction, so a vanishing denominator is its expected neighbourhood
    # rather than an edge case.
    _UP_SIGNED = Scoring(scored=True, direction="higher", anchor=None,
                         allow_negative_baseline=True, clamp_low=-2.0, clamp_high=2.0)

    add("nsig_counts_real", v1="de_nsig_counts_real",
        func=functools.partial(de_nsig_counts, side="real"), scoring=_NO_DIRECTION, agg="mean",
        profiles=("full", "minimal", "de"))
    add("nsig_counts_pred", v1="de_nsig_counts_pred",
        func=functools.partial(de_nsig_counts, side="pred"), scoring=_NO_DIRECTION, agg="mean",
        profiles=("full", "minimal", "de"))
    add("nsig_spearman", v1="de_spearman_sig", func=de_nsig_spearman, scoring=BOUNDED,
        agg="mean", profiles=("full", "de"), worst=-1.0)
    add("sig_recall", v1="de_sig_genes_recall", func=de_sig_recall, scoring=BOUNDED,
        agg="mean", profiles=("full", "de"), worst=0.0)
    add("direction_match", v1="de_direction_match", func=de_direction_match, scoring=BOUNDED,
        agg="mean", profiles=("full", "de"), worst=0.0)
    add("model_direction_match", v1="de_model_direction_match",
        func=de_model_direction_match, scoring=BOUNDED, agg="mean",
        profiles=("full", "de"), worst=0.0)
    add("lfc_spearman", v1="de_spearman_lfc_sig", func=de_lfc_spearman, scoring=BOUNDED,
        agg="mean", profiles=("full", "de"), worst=-1.0)
    add("lfc_spearman_pos", v1="de_spearman_pos_lfc_sig",
        func=functools.partial(de_lfc_spearman, lfc_direction="pos"), scoring=BOUNDED,
        agg="mean", profiles=("full", "de"), worst=-1.0)
    add("lfc_spearman_neg", v1="de_spearman_neg_lfc_sig",
        func=functools.partial(de_lfc_spearman, lfc_direction="neg"), scoring=BOUNDED,
        agg="mean", profiles=("full", "de"), worst=-1.0)
    # Issue #208. The de_* block's only NUMBER-SCALE member: the rank members say whether
    # the ordering is right, this says whether a two-fold change is reported as two-fold.
    # `ERROR_LINEAR`, not `ERROR`: same direction, same anchor, same -6.0 floor, but the
    # Box-Cox tail below the baseline is REPLACED by a straight line (Alex, 2026-08-17).
    # `penalty="none"`, so there is no tail left to shape -- the floor is a declared
    # `clamp_low`, which is also why `penalty_cap` is now inert on this member. It WAS the only
    # `vcc2026` member with a live sub-zero range -- the other five clipped at 0 -- so it alone
    # decided the shape of the competition average's whole downside, and a quadratic tail made
    # that a near-binary test of saturating rather than a ranking. `expr_mae` and the three
    # other error metrics keep `ERROR`; `expr_mae` is in the frozen 2025 `vcc` profile.
    # ⚠️ PAST TENSE as of the clip-at-0 removal, which unfloored FOUR of those five
    # (`pds_cosine`, `direction_fidelity_yield_raw`, `direction_reach_raw`, `sig_jaccard`:
    # `clamp_low=None` with `metric_min=0.0`). This member is no longer alone in reaching
    # below zero, and the six-member average no longer floors at exactly -1. The reasoning
    # above is the rationale AT THE TIME and is kept for that; do not read it as current
    # policy.
    # ⚠️ This `add()` runs for BOTH families, so `de_deseq2_lfc_nmae` moves with it. That is
    # intended -- the two are one metric behind two DE backends -- and it is NOT inert: the
    # deseq2 family is `v2_only` (profiles=()), so no profile selects it, but
    # `run._effective_de_spec` relabels the wilcoxon name to it under `de.backend="deseq2"`, so
    # a run whose RESOLVED METRIC SET contains the wilcoxon sibling scores this entry instead --
    # and `_guard_deseq2_metric_selection` permits `vcc2026` under that backend too, so a
    # DIAGNOSTIC `vcc2026` number moves as well. What it cannot reach is an ENROLLED official
    # competition score: that needs a competition bundle, and `de.backend` is excluded from
    # `rule_config_hash`, so it is the relabelled MEMBERSHIP that makes such a bundle diagnostic
    # (four wilcoxon members missing, four deseq2 members extra). Sufficient on its own; the
    # per-member estimator check reports the same relabelling independently.
    # worst=None (the add() default) is REQUIRED, not incidental: nmae is unbounded above,
    # so there is no finite worst value to fill an omitted perturbation with. Omission is
    # decided by the REAL-side gate alone and is therefore identical for every submission.
    # NOT in `vcc` -- the 2025 competition profile -- so that leaderboard is unchanged. It IS
    # in `vcc2026`, whose members are decisive as of #255: a degenerate baseline is a hard
    # failure rather than a silent change to the competition average's denominator.
    add("lfc_nmae", v1=None, func=de_lfc_nmae, scoring=ERROR_LINEAR, agg="mean",
        profiles=("full", "de", "vcc2026"))
    add("pr_auc", v1="pr_auc", func=de_pr_auc, scoring=BOUNDED, agg="mean",
        profiles=("full", "de"), worst=0.0)
    add("roc_auc", v1="roc_auc", func=de_roc_auc, scoring=BOUNDED, agg="mean",
        profiles=("full", "de"), worst=0.0)
    # Chance-corrected agreement metrics (issue #14): correlation-family measures over a
    # per-perturbation 2x2 table, bounded [-1, 1] (0 = chance, 1 = perfect, -1 = worst).
    # full/de only (so the vcc competition score is unchanged) and v2-native (no cell-eval
    # equivalent -> v1_name=None, hence not v1_available). Scored: they have a direction.
    add("overlap_adjusted", v1=None, func=de_overlap_adjusted, scoring=BOUNDED, agg="mean",
        profiles=("full", "de"))
    add("precision_adjusted", v1=None,
        func=functools.partial(de_sig_agreement, measure="markedness"), scoring=BOUNDED,
        agg="mean", profiles=("full", "de"))
    add("sig_recall_adjusted", v1=None,
        func=functools.partial(de_sig_agreement, measure="informedness"), scoring=BOUNDED,
        agg="mean", profiles=("full", "de"))
    add("sig_mcc", v1=None,
        func=functools.partial(de_sig_agreement, measure="mcc"), scoring=BOUNDED,
        agg="mean", profiles=("full", "de"))
    # Direction metrics: they score the DIRECTION of a call rather than set membership,
    # with explicit undefined-direction semantics (reference cannot adjudicate ->
    # excluded; model declined to commit -> miss). v2-native; the older
    # model_direction_match is retained unchanged and scores a both-zero or both-NaN pair
    # as agreement (a both-null pair compares null and is ignored by the mean). Appended rather than grouped with model_direction_match so
    # the ordered GOLDEN_WILCOXON snapshot gains rows instead of renumbering.
    add("direction_precision", v1=None, func=de_direction_precision, scoring=BOUNDED,
        agg="mean", profiles=("full", "de"), worst=0.0)
    add("direction_sensitivity", v1=None,
        func=functools.partial(de_direction_sensitivity, universe="adjudicated"),
        scoring=BOUNDED, agg="mean", profiles=("full", "de"), worst=0.0)
    # `_UP_OPEN`: the full-universe variant is unbounded above (k* is not capped by N_conf),
    # so an anchor of 1 would be false. It IS scored (it has a direction) but anchorless, so
    # it normalizes against its own baseline and is clamped. compat.score_agg_metrics knows
    # only the anchor-0 and anchor-1 normalizations and declines anything v2-native, so it
    # skips this one with a warning rather than scoring it against a denominator it does not
    # have. See the note on `MetricSpec.best_value` -- the derived token cannot say this.
    add("direction_sensitivity_universe", v1=None,
        func=functools.partial(de_direction_sensitivity, universe="all"),
        scoring=_UP_OPEN, agg="mean", profiles=("full", "de"), worst=0.0)

    # Chance-corrected direction metrics (issue #195). Eleven suffixes x two methods.
    # APPENDED, not grouped with the v0.5.0 direction metrics above, so the ordered
    # GOLDEN_WILCOXON snapshot gains rows instead of renumbering.
    #
    # All eleven: worst=None (spec 5 -- _fill_no_drop must NOT rewrite these NaNs; a 0.0
    # fill would report "reached nothing" for a target that was never scoreable), and
    # v1_avail=False (v2-native; the gate lives in resolve_metrics).
    #
    # ALL eleven aggregate by MEAN, in BOTH families -- `add` passes one `agg` to each
    # wilcoxon/deseq2 sibling pair, so a metric cannot change its aggregation statistic
    # because the DE backend changed. Nine of them held agg="median" (the DOR's statistic)
    # through v0.7.0, the two then in `vcc2026` having moved to the mean in #229; issue #231
    # finished the job, and the catalog now has exactly ONE aggregation statistic. Two
    # statistics meant a whole-cohort number answered a different question depending on which
    # suffix you read, and a profile average built as a plain mean over metrics silently mixed
    # them. `agg` REMAINS a per-entry field with no default on `add`, and
    # `run.aggregate_metrics` still implements the median branch generically, so changing the
    # statistic stays a catalog edit rather than a source edit.
    # Change 1 does not move any per-perturbation value; it moves 18 whole-cohort numbers, so
    # every baseline predating v0.8.0 is invalid (see `baseline.config_digest`, which records
    # the metric -> agg mapping precisely so the mismatch fails loud).
    #
    # `scoring.scored` is TRUE for all eleven. 4c28dac 2.6 had split them SCORED/diagnostic
    # on boundedness, because the old token could not say "higher is better but not enrolled"
    # -- enrolment and boundedness had to share one spelling. With `anchor` recording
    # boundedness separately (spec 4.1) that constraint is gone, and for these eleven the
    # enrolment rule is simply "does this metric have a direction?", which all eleven do.
    # (Catalog-wide that rule now happens to hold in both directions, since enrolling
    # `expr_mse_unbiased_norm` (since replaced, #257) emptied the directional-but-unscored
    # state -- but only
    # coincidentally: `Scoring` still expresses it, and no DE metric was ever in that
    # position anyway. See `test_enrolment_implies_a_direction_but_not_conversely`.)
    # None of them is in the `vcc` profile, so enrolling them changed the full/de avg_score
    # only, never the 2025 competition score. TWO of them -- `direction_fidelity_yield_raw`
    # and `direction_reach_raw` -- ARE in `vcc2026`, so for that profile they are competition
    # metrics; see the `vcc2026` note above `is_decisive`.
    #
    # WHY the RAW pair and not the chance-CORRECTED one (issue #231). v0.5.0-v0.7.0 scored
    # `direction_fidelity_yield` / `direction_reach` in `vcc2026`; v0.8.0 scores their `_raw`
    # siblings. The corrected `fidelity_yield`'s no-skill point is neither zero nor stable.
    # ⚠️ Every `direction_reach_raw` number quoted in this block was measured at the old purity
    # floor `1 - alpha/2 = 0.975`, before `REACH_PURITY_FLOOR = 0.9`, and its levels are
    # therefore stale. The CORRECTED sibling's numbers are NOT: it keeps its own q-dependent
    # threshold `q + (1-alpha)(1-q)` and did not move with the floor -- so the two siblings did
    # NOT move together, and the block's comparison now spans two different rules. What survives
    # is the QUESTION it settles (which sibling `vcc2026` should score, decided on the corrected
    # one's unstable no-skill point), which is a property of the corrected metric alone.
    # Measured over six CCL lines x 4 arms on two panels (`noReplogle_100_100_100` and
    # `ndeg20_100_100_100`), each panel's own run; replicate = the split-half arms, baseline =
    # control_mean/context_mean, mean aggregation:
    #
    #     metric                         replicate       baseline             gap
    #     direction_reach_raw            0.9118          0.0251               0.8867
    #     direction_fidelity_yield_raw   0.8194          0.5015               0.3179
    #     direction_reach (corrected)    0.9007-0.9077   0.0144-0.0258        ~0.89
    #     direction_fidelity_yield       0.6560-0.7131   -0.6086 to -0.8948   --
    #
    # The corrected `fidelity_yield`'s baseline is NEGATIVE and panel-sensitive, so the
    # denominator `score_metrics` normalizes against depends on which panel built the
    # baseline. The raw one is pinned at 0.4863-0.5148 across all 12 line x baseline cells --
    # empirically the theoretical random point. The raw pair is panel-robust too: between the
    # two panels the raw aggregates move <=0.012 where the corrected ones move 0.05-0.10.
    # Both raw entries then carried anchor=1.0, clamp_low=0.0, penalty="none", scored=True,
    # worst=None -- identical to the corrected pair they replace -- so `vcc2026`'s formal
    # `avg_score` range did not change AT THAT TIME. ⚠️ The clip-at-0 removal has since
    # unfloored both (`clamp_low=None`, `metric_min=0.0`), so that last clause no longer
    # describes the shipped policy; the substitution argument it supports is unaffected,
    # since it turns on the raw pair's no-skill stability, not on their clamps. The corrected pair stays in `full`/`de`.
    # Known consequence, accepted: `vcc2026` no longer charges a submission for predicting
    # each gene's habitual direction. `direction_reach_raw` now carries the anti-gaming load
    # on its own -- an abstaining predictor drives bare `fidelity_raw` to 0.9999 but is held
    # to 0.005-0.05 on `reach_raw`.
    #
    # `_direction_kw` carries agg="mean"; passing an explicit agg= to any of these eleven is a
    # TypeError (duplicate keyword), so only the other 19 call sites name it. Since #231 there
    # is nothing left to override there, and `_direction_kw_2026` differs by `profiles` ALONE
    # -- which still has to be overridden inside the dict, not as a second keyword, because
    # the base dict already binds it.
    _direction_kw = dict(v1=None, profiles=("full", "de"), worst=None,
                         agg="mean", v1_avail=False)
    _direction_kw_2026 = {**_direction_kw, "profiles": ("full", "de", "vcc2026")}
    add("direction_fidelity", func=de_direction_fidelity, scoring=_UP_BOUNDED, **_direction_kw)
    add("direction_fidelity_raw", func=de_direction_fidelity_raw, scoring=_UP_BOUNDED,
        **_direction_kw)
    add("direction_coverage", func=de_direction_coverage, scoring=_UP_OPEN, **_direction_kw)
    add("direction_yield", func=de_direction_yield, scoring=_UP_SIGNED, **_direction_kw)
    add("direction_yield_raw", func=de_direction_yield_raw, scoring=_UP_OPEN, **_direction_kw)
    add("direction_fidelity_yield", func=de_direction_fidelity_yield, scoring=BOUNDED,
        **_direction_kw)
    # `_UP_UNFLOORED`, not `_UP_BOUNDED`: a `vcc2026` member, clip-at-0 removed. This is the
    # DEEPEST of the four -- its comparator is pinned at chance (0.5) by construction, so the
    # unclamped floor is -0.5/(r - 0.5) and scales as 1/(r - chance): -1.58/-2.68/-1.85 on the
    # three official arms. The corrected sibling above keeps `_UP_BOUNDED` (full/de only).
    add("direction_fidelity_yield_raw", func=de_direction_fidelity_yield_raw,
        scoring=_UP_UNFLOORED, **_direction_kw_2026)
    add("direction_reach", scoring=BOUNDED,
        func=functools.partial(de_direction_reach, universe="adjudicated", corrected=True),
        **_direction_kw)
    # `vcc2026` member, clip-at-0 removed. Barely binding in practice -- its comparator sits
    # near the metric's own floor (0.02-0.09), so the truncated depth was only -0.05/-0.12/-0.12.
    add("direction_reach_raw", scoring=_UP_UNFLOORED,
        func=functools.partial(de_direction_reach, universe="adjudicated", corrected=False),
        **_direction_kw_2026)
    add("direction_reach_unbounded", scoring=_UP_OPEN,
        func=functools.partial(de_direction_reach, universe="all", corrected=True),
        **_direction_kw)
    add("direction_reach_unbounded_raw", scoring=_UP_OPEN,
        func=functools.partial(de_direction_reach, universe="all", corrected=False),
        **_direction_kw)

    # Raw (chance-UNcorrected) symmetric agreement over the same significance-membership
    # 2x2 table as `sig_mcc`: |R ∩ P| / |R ∪ P| (identically so on a well-formed DE table;
    # `sig_jaccard` de-duplicates `(target, feature)` where the `sig_agreement` family counts
    # rows, so they differ only if duplicate rows are present). Appended last so the ordered
    # GOLDEN_WILCOXON snapshot gains a row rather than renumbering. `worst=None` because
    # the metric itself returns a finite value for EVERY perturbation (an empty union is
    # 1.0, not NaN), exactly like the four #14 chance-corrected entries -- a worst_value
    # here would be an unreachable branch.
    # `vcc2026` member, clip-at-0 removed (truncated depth -0.08/-0.07/-0.10 on the three
    # official arms -- its comparator sits near the metric's own floor).
    add("sig_jaccard", v1=None, func=de_sig_jaccard, scoring=_UP_UNFLOORED, agg="mean",
        profiles=("full", "de", "vcc2026"))


_register_de_family(CATALOG, method="wilcoxon", v2_only=False)
_register_de_family(CATALOG, method="deseq2", v2_only=True)

def _build_profiles(catalog: dict[str, MetricSpec]) -> dict[str, list[str]]:
    """Group catalog metrics by profile tag, in CATALOG insertion order.

    `MetricSpec.profiles` is the single source of truth for profile membership;
    deriving PROFILES from it (rather than hand-maintaining a parallel dict) keeps
    the two from drifting (issue #19). Every name a profile yields is therefore a
    real CATALOG entry; deferred reference metrics (issue #23) are simply absent
    until they declare their own `.profiles`.
    """
    profiles: dict[str, list[str]] = {}
    for name, spec in catalog.items():
        for profile in dict.fromkeys(spec.profiles):  # dedup own tags, order-preserving
            profiles.setdefault(profile, []).append(name)
    return profiles


PROFILES: dict[str, list[str]] = _build_profiles(CATALOG)


def _build_name_index(catalog: dict[str, MetricSpec]) -> dict[str, str]:
    """Map every accepted spelling (canonical name, v1_name, aliases) → canonical name.
    A spelling must map to exactly one canonical metric."""
    index: dict[str, str] = {}
    for spec in catalog.values():
        for spelling in (spec.name, spec.v1_name, *spec.aliases):
            if not spelling:        # skip None and "" (valid spellings are non-empty)
                continue
            if spelling in index and index[spelling] != spec.name:
                raise ValueError(
                    f"duplicate metric spelling {spelling!r}: maps to both "
                    f"{index[spelling]!r} and {spec.name!r}"
                )
            index[spelling] = spec.name
    return index


_NAME_TO_CANONICAL: dict[str, str] = _build_name_index(CATALOG)

# The metrics whose scaled score `lfc_nmae_ref` applies to. Named rather than derived: the
# reference measures a SPECIFIC quantity (a split-half replicate's normalized LFC MAE), so
# it is meaningful for exactly these two entries and for nothing else that happens to share
# their scoring policy.
#
# It lives here rather than in `score.py` because it is a catalog fact -- which catalog
# entries the lfc_nmae reference is meaningful for -- and because `anchor.py` needs it: an
# `anchor -> score` import would close a cycle once `score` imports `anchor` for
# `resolve_anchor`. `catalog.py` imports only `metrics.*` and `scoring`, so nothing can
# cycle back through it.
_LFC_NMAE_METRICS = ("de_wilcoxon_lfc_nmae", "de_deseq2_lfc_nmae")


def derived_policy(names) -> list[list[str]]:
    """Stable digest payload for every derived metric in ``names``.

    A derived metric's IDENTITY is its name plus the two components it divides. Hashing only
    the name would let a numerator or denominator swap reuse a cached result computed under
    the old definition -- the same class of failure as #250's cache serving pre-fix scores.
    Sorted, so request order never moves the digest.
    """
    out = []
    for n in sorted(set(names)):
        spec = CATALOG.get(_NAME_TO_CANONICAL.get(n, n))
        if spec is not None and spec.derived is not None:
            out.append([spec.name, spec.agg, spec.derived.numerator, spec.derived.denominator])
    return out


def needs_moments_transitively(name: str) -> bool:
    """Does ``name`` consume per-group moments, directly or through a derived component?

    ``needs_moments`` alone is not enough to exclude the moments family from a metric set.
    A derived ``ratio_of_sums`` entry cannot set the flag -- it has no per-perturbation
    column and its own func never touches moments -- but :func:`resolve_metrics` pulls in
    its ``numerator`` and ``denominator``, which do. MEASURED on the shipped filter:
    ``[m for m in PROFILES["full"] if not CATALOG[m].needs_moments]`` goes 51 in -> 53 out,
    with ``expr_mse_unbiased_capped`` and ``expr_distance_unbiased`` back; ``vcc2026`` goes
    7 -> 9. With this predicate they are 50 -> 50 and 6 -> 6.

    Callers are the two moments exclusions that exist because ``partition_inmem`` cannot
    supply moments -- ``tests/_helpers.full_minus_moments`` and the out-of-core manifest CLI
    runner -- both of which issue #272 deletes.
    """
    spec = CATALOG[name]
    if spec.needs_moments:
        return True
    if spec.derived is None:
        return False
    return any(CATALOG[component].needs_moments for component in (
        spec.derived.numerator, spec.derived.denominator,
    ))


def deseq2_metric_name(name: str) -> str:
    """Map a canonical DE metric name to its deseq2 sibling: ``de_wilcoxon_<x>`` ->
    ``de_deseq2_<x>`` (identity if already deseq2). Raises KeyError if the target is not a
    registered DE metric -- the backend relabel must never invent a name."""
    if name.startswith("de_deseq2_"):
        return name
    mapped = name.replace("de_wilcoxon_", "de_deseq2_", 1)
    if mapped not in CATALOG:
        raise KeyError(f"no deseq2 sibling for metric {name!r}")
    return mapped


# Reference metrics named in built-in profiles but not yet implemented (issue #23).
# An explicit-list name here is treated as deferred (-> `missing`), not a typo.
KNOWN_DEFERRED: frozenset[str] = frozenset(
    {"edistance_pearson", "pearson_edistance", "clustering_agreement"}
)


def resolve_metrics(
    metrics: str | list[str], *, version: str = "v2"
) -> tuple[list[str], list[str]]:
    """Resolve a profile name or explicit list into (available, missing) canonical names.

    Any accepted spelling (canonical, v1, or alias) is mapped to its canonical name.
    `available` are present in CATALOG. For a profile input `missing` is always empty
    (every profile name is a real CATALOG entry, post-#19). `missing` is only ever
    non-empty for an explicit list naming a known-deferred metric (issue #23); an
    explicit-list name that is neither in CATALOG nor known-deferred is a typo and
    raises ValueError (an unknown profile likewise raises).

    ``version='v1'`` additionally applies the v1 availability gate, and PROVENANCE
    matters: a metric that is not v1-available is filtered SILENTLY when it arrived via a
    PROFILE, and RAISES when the caller named it explicitly. The distinction is not
    cosmetic -- ``compat/__init__.py:108`` expands a profile into an explicit list and
    writes it back onto the config before calling ``compute_metrics``, so raising on
    explicit names alone would turn every ordinary v1 profile run into an error.

    The gate lives HERE, in the one shared resolution path, and not in dispatch: this
    function is called independently by run, compat, baseline, scale and partition_inmem,
    and ``metric_output_names`` resolves the profile independently of dispatch, after
    which ``aggregate_metrics_wide`` materializes expected-but-unobserved metrics as
    all-NaN columns -- so a dispatch-only gate leaves the tidy frame clean while the
    PUBLISHED wide CSV still carries the columns.
    """
    from_profile = isinstance(metrics, str)
    if from_profile:
        if metrics not in PROFILES:
            raise ValueError(f"unknown profile {metrics!r}; known: {sorted(PROFILES)}")
        names = PROFILES[metrics]
    else:
        names = list(metrics)
    # Dedup while preserving order: a caller mixing v1 and v2 spellings of the same
    # metric (e.g. ["mae", "expr_mae"]) must not run it twice / double the tidy rows.
    canonical = list(dict.fromkeys(_NAME_TO_CANONICAL.get(n, n) for n in names))
    # A derived metric is computed at AGGREGATION time from two other metrics' columns, and
    # the compute dispatch skips it (it has no func). Requesting it without its components
    # would therefore run nothing and produce no aggregate row at all -- silently. Close the
    # dependency here, in the one shared resolution path, so the request means what it says.
    # Profiles already carry the components (a catalog invariant,
    # `test_every_shipped_derived_metric_has_its_components_in_every_profile_it_claims`), so
    # this only ever fires for an explicit list, and dedup keeps the order stable (#257).
    for n in list(canonical):
        spec = CATALOG.get(n)
        if spec is not None and spec.derived is not None:
            for side in (spec.derived.numerator, spec.derived.denominator):
                if side not in canonical:
                    canonical.append(side)
    if version == "v1":
        blocked = [n for n in canonical
                   if n in CATALOG and not CATALOG[n].v1_available]
        if blocked and not from_profile:
            raise ValueError(
                f"metric(s) {sorted(blocked)} are not available under version='v1' "
                "(v2-native, issue #195). Drop them from the request or run with "
                "version='v2'."
            )
        canonical = [n for n in canonical if n not in set(blocked)]
    available = [n for n in canonical if n in CATALOG]
    missing = [n for n in canonical if n not in CATALOG]
    # Validate for any explicit (non-profile) input — list, tuple, set, generator —
    # not just `list`, mirroring the `else` branch above (which accepts any non-string
    # iterable). Otherwise a tuple/set of names could silently bypass the typo check.
    if not from_profile:
        unknown = [n for n in missing if n not in KNOWN_DEFERRED]
        if unknown:
            raise ValueError(
                f"unknown metric name(s): {unknown}; known: {sorted(CATALOG)}"
            )
    return available, missing
