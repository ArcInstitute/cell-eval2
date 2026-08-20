"""The competition's RULE is frozen in the repo; its per-dataset VALUES are not.

An anchor value is a property of one dataset, so a constant here would be a hash CI has no
data to recompute -- a test comparing a constant to itself. What CI *can* check is every
field that decides a number: the members, their resolved policies, and how the anchor is
produced. That is what this digest covers, in the `scales_digest()` spirit.
"""
import pytest

from cell_eval2 import competition
from cell_eval2.config import EvalConfig

# If this test failed, the question is NOT "what is the new hash" -- it is "did a competition
# number, or the RULE PROVENANCE this digest records, just move, and was that intended?".
# Update it only alongside a deliberate, changelogged scoring change.
#
# ⚠️ NEITHER DIRECTION OF THAT IMPLICATION IS TIGHT, and the two failures are opposite.
#
# A digest move need not mean a NUMBER moved. The payload serializes each member's RESOLVED
# knobs, so `penalty_cap` is recorded for a member whose policy has no penalty at all; retuning
# `DEFAULT_PENALTY_CAP` therefore moves this constant while every default `vcc2026` score stands
# still. That is deliberate -- the payload is a provenance SUPERSET of the arithmetic -- and it
# is the direction that produces a scary-looking diff over an inert change.
#
# ⚠️ THE CONVERSE DOES NOT HOLD, and issue #172 is the worked example. This digest covers the
# members and their resolved SCORING POLICIES -- direction, anchor, penalty, clamps, agg, the
# derived pair -- plus how the anchor is produced. It does NOT cover what a member COMPUTES.
# #172 moved scored numbers on three of the six members (the perturbed gene's own row left
# `de_wilcoxon_sig_jaccard`, `de_wilcoxon_lfc_nmae` and both legs of
# `expr_mse_unbiased_capped_norm`) and this digest was bit-identical before and after --
# correctly, since no policy field changed. A green run here is evidence about the RULE, never
# about the values.
#
# ⚠️ AND THAT GAP IS STRUCTURAL, not merely a scoping note. A bundle's `rule_digest` is what
# `score.py` checks to decide two submissions are mutually comparable, so two bundles built
# either side of a semantics change would pass that check while their frozen replicate anchors
# were computed over DIFFERENT GENE SETS. The only lever that closes it is a deliberate
# `rule_version` bump, which is why this constant is expected to move by hand from time to time:
# a move here can be the plan executing rather than an accident to revert to green.
#
# ⚠️ Moved 2026-08-18 by exactly that bump -- `rule_version` 1 -> 2 in the 0.14.0 release
# wave (#317), from `1b93878b...`. Version 2 means the three semantics changes this digest could
# not see: #172's target-gene exclusion, `de_wilcoxon_direction_reach_raw`'s calibrated purity
# floor (0.9, not the derived 0.975), and #271's wide pseudobulk reduction. Every artifact stamped 0.13.0 is invalidated by
# design; the three #276 val bundles are rebuilt as `-r2` in the same wave. The debt outstanding
# against `rule_version = 2` is listed in `competition.py` at the literal itself, and it is now
# empty. Two earlier moves within 0.13.0 -- `ERROR_LINEAR`, then the clip-at-0 removal on the
# four bounded members -- are recorded in the CHANGELOG rather than here: both were POLICY-field
# changes, which this digest catches on its own, so neither owed a bump and neither is what this
# constant records.
#
# ⚠️ Moved 2026-08-19 by #327, from `f32f0f9c...`, and this one is the OPPOSITE case to the
# bump above: the digest moved BY ITSELF because the payload gained a field it was previously
# blind to. `reach_purity_floor` is the constant `de_wilcoxon_direction_reach_raw` thresholds
# its purity curve on -- a scored member's one arithmetic knob, which the digest could not see
# and which is why #322's recalibration had to be paid for with the `rule_version` bump rather
# than with a digest move. `rule_version` STAYS 2: the lever exists for the changes this digest
# cannot see, and this change removes an entry from that list rather than adding one.
#
# ⚠️ NO SCORED NUMBER MOVES WITH IT. #327 made the floor a function argument whose default is
# the same calibrated 0.9; the digest moves because the payload records it now, not because the
# member computes anything different. The consequence to plan for is downstream: the three
# official `-r2` bundles carry the old `rule_digest` and `score.py` RAISES on the mismatch, so
# they must be rebuilt as `-r3` -- and `-r3` must come out numerically identical to `-r2` on
# every member, with only `rule_digest`, `cell_eval2_version`, the ids and the timestamp
# differing. That comparison is the release pass's verification of this change.
# ⚠️ SUPERSEDED, and then superseded again. #343 landed before `-r3` was built and moves
# `pds_cosine`; #348 then changed `expr_mse_unbiased_capped`'s scoring semantics and #351 moved all
# four DE members, both joining the same `rule_version` 3 wave. So FIVE of the six are expected to
# move and only
# `expr_mse_unbiased_capped_norm` may come out inert. Verify against the MEASURED `-r2` -> `-r3`
# table, NOT against an expectation of equality on any fixed count of members.
#
# ⚠️ Moved again 2026-08-19 by a second `rule_version` bump -- 2 -> 3, shipped in 0.15.0
# (#343), from `80558072...`. Version 3 now means THREE semantics changes this digest cannot see,
# and the digest literal below is UNCHANGED by the two that joined after the bump -- which is the
# point of the version lever, and is why they were allowed to join at all: no bundle had been built
# against version 3 yet, so no digest in the field was falsified. `competition.py` carries all
# three at the `rule_version` literal itself. The two that joined:
#   * #348 -- `expr_mse_unbiased_capped` bounds the prediction's total sampling correction by the
#     submission's OWN across-perturbation spread, not by the reference's per-row cap alone.
#   * #351 -- the DE gene gate keeps a gene on the REFERENCE group's mean CPM alone, where version
#     2 kept it when the target group's cleared the threshold OR the control's did. That OR made a
#     row's presence disclose its log2FC's sign (`P(real log2FC > 0) = 1.000000` over 26k-34k rows
#     per official context), worth +0.3722 / +0.5059 / +0.3651 of
#     `de_wilcoxon_direction_fidelity_yield_raw` `from_baseline` on a submission reading no
#     perturbation-specific information at all.
# The change the bump itself was for:
# `pds_cosine` now ranks in the feature space with EVERY panel target gene removed
# (`discrimination.exclusion_scope = "panel"`), where version 2 removed only the PREDICTION
# row's own and dropped it from that row's comparison against every reference perturbation --
# leaving perturbation q's own knockdown visible in cell (i, q). A submission spiking the
# panel's other targets scored `pds_cosine` 0.7982 / 0.7570 / 0.7614 on the three official
# contexts against baselines of 0.5304 / 0.5284 / 0.5102, i.e. +0.57 / +0.49 / +0.51 of member
# score on no information beyond the published target list; under version 3 those arms measure
# exactly 0.5000. Unlike #327 above, this one DOES move a scored number, which is why it takes
# the version lever and not a digest move alone. The debt outstanding against `rule_version = 3`
# is listed in `competition.py` at the literal itself, and it is empty.
FROZEN_DIGEST = "fb5aa56b74368a7bc5befc8ccca8ca02a2cf4c8c7ad59b156ab86c3ae0859762"

def test_the_six_members_are_the_scored_vcc2026_metrics():
    assert competition.competition_members() == (
        "pds_cosine",
        "expr_mse_unbiased_capped_norm",
        "de_wilcoxon_lfc_nmae",
        "de_wilcoxon_direction_fidelity_yield_raw",
        "de_wilcoxon_direction_reach_raw",
        "de_wilcoxon_sig_jaccard",
    )


def test_the_payload_carries_every_field_that_decides_a_number():
    p = competition.competition_payload()
    assert p["base_seed"] == 0 and p["n_splits"] == 5
    assert p["seed_derivation"].startswith("numpy.random.SeedSequence")
    # The LITERAL seeds, not just the algorithm string: a change to `_derive_seeds` moves every
    # shipped anchor while leaving the documented rule identical.
    assert p["derived_seeds"] == list(competition.derived_seeds())
    assert p["control_source"] == {"scored": "real", "anchor": "pred"}
    assert p["comparison_statistic"] == "mean"
    assert p["bulk_target_sum"] == 50_000.0
    mse = p["members"]["expr_mse_unbiased_capped_norm"]
    assert (mse["clamp_low"], mse["clamp_high"]) == (0.0, 1.0)
    assert mse["penalty_cap"] == 6.0            # RESOLVED, not the catalog's None
    # What the metric IS, not only how it is scored: `agg` and the derived pair change u, b
    # and r alike, and the baseline digest already serializes the same facts.
    assert mse["agg"] == "ratio_of_sums"
    assert mse["derived"] == {"numerator": "expr_mse_unbiased_capped",
                              "denominator": "expr_distance_unbiased"}
    # ⚠️ RESOLVED, not declared. Both expr members declare `expr_comparator`, which is an
    # intent rather than a space; under the preset it resolves to `bulk_lognorm` (measured).
    # Freezing the declaration would leave the digest blind to the #264 PR2 defect exactly:
    # a comparator move that changes every number while the catalog string sits still.
    assert mse["normalization"] == "bulk_lognorm"
    assert competition.competition_comparator() == "bulk_lognorm"
    assert p["members"]["de_wilcoxon_lfc_nmae"]["normalization"] is None      # not an expr metric
    assert p["members"]["de_wilcoxon_lfc_nmae"]["estimator"] == "full_gate_raw"
    # The one member with a live sub-zero range -- the other five clip at 0 -- so its SHAPE is
    # the whole downside of the competition average and is pinned here, not only hashed.
    # `penalty_cap` is still resolved to 6.0 in the payload and is inert under penalty "none";
    # the floor is `clamp_low`, which is what `effective_clamp_low()` now reports directly.
    lfc = p["members"]["de_wilcoxon_lfc_nmae"]
    assert lfc["penalty"] == "none"
    assert (lfc["clamp_low"], lfc["clamp_high"]) == (-6.0, None)
    assert p["members"]["pds_cosine"]["estimator"] == "split_half_raw"
    assert p["reducer"] == "unweighted_arithmetic_mean"
    assert p["member_order_in_frame"][-1] == "pds_cosine"     # sorted-higher block ends here


def test_the_digest_is_frozen():
    assert competition.competition_digest() == FROZEN_DIGEST


# `changes` is a DICT rather than a single (field, value) pair because two policy fields are
# now co-constrained: `metric_min` must sit on the worse side of `anchor`, so flipping
# `direction` on a member that carries one has to clear it in the SAME `replace()` --
# `Scoring.__post_init__` would otherwise raise before any digest was computed. Keeping the
# single-field form would have forced the `direction` case to be dropped, which is exactly
# the coverage this test exists to provide.
@pytest.mark.parametrize("member,changes,control", [
    ("expr_mse_unbiased_capped_norm", {"clamp_low": -3.0}, None),
    ("expr_mse_unbiased_capped_norm", {"clamp_high": 2.0}, None),
    ("expr_mse_unbiased_capped_norm", {"penalty_cap": 5.0}, None),
    ("expr_mse_unbiased_capped_norm", {"penalty_exponent": 3.0}, None),
    ("expr_mse_unbiased_capped_norm", {"anchor": 0.5}, None),
    ("expr_mse_unbiased_capped_norm", {"scored": False}, None),
    # ⚠️ `direction` and `penalty` need a member whose policy permits the flip.
    # `Scoring.__post_init__` requires penalty='boxcox' to pair with direction='lower' and a
    # non-None anchor, so flipping direction on the boxcox MSE member raises before any digest
    # is computed -- but flipping it on `pds_cosine` (penalty='none') is legal, and an earlier
    # draft wrongly called the whole field unreachable on that basis. `penalty` goes the other
    # way: 'none' is legal on MSE only AFTER Task 1 gives it a finite clamp_low.
    # `pds_cosine` is UNFLOORED since the clip removal, so a bare direction flip is illegal
    # twice over: a lower-is-better metric cannot have a worst value BELOW its anchor, and
    # clearing `metric_min` alone leaves a scored policy with no finite floor. The CONTROL
    # carries those two repairs WITHOUT the direction flip, so the comparison still isolates
    # `direction` -- comparing against the unmodified catalog would not.
    ("pds_cosine", {"direction": "lower", "metric_min": None, "clamp_low": 0.0},
     {"metric_min": None, "clamp_low": 0.0}),
    ("expr_mse_unbiased_capped_norm", {"penalty": "none"}, None),
    # The new field itself. It decides what a MISSING member contributes to avg_score on an
    # unfloored policy, so a digest blind to it would freeze the wrong rule.
    ("pds_cosine", {"metric_min": 0.25}, None),
])
def test_the_digest_moves_when_any_member_policy_moves(member, changes, control, monkeypatch):
    """Serialize a MODIFIED Scoring object -- never mutate the already-built payload, which
    would pass even if `competition_payload` omitted the field entirely.

    `control` is the state the mutation is measured AGAINST -- normally the unmodified
    catalog (`None`), but a field whose flip needs co-repairs to stay legal is compared
    against those repairs alone, so the digest move is attributable to the field under test
    and not to its escort.
    """
    from dataclasses import replace

    from cell_eval2.catalog import CATALOG

    def digest_with(overrides):
        if not overrides:
            return competition.competition_digest()
        spec = CATALOG[member]
        patched = dict(CATALOG)
        patched[member] = replace(spec, scoring=replace(spec.scoring, **overrides))
        monkeypatch.setattr(competition, "CATALOG", patched)
        try:
            return competition.competition_digest()
        finally:
            monkeypatch.undo()

    assert digest_with(changes) != digest_with(control)


def test_allow_negative_baseline_is_recorded_even_though_it_cannot_be_mutated():
    """The one policy field with no legal mutation: `Scoring.__post_init__` refuses
    `allow_negative_baseline=True` alongside a non-None anchor, and every member has one.
    A mutation test is therefore impossible, but the field must still be IN the payload --
    `FROZEN_DIGEST` is filled from the first green run, so a field omitted at that moment is
    frozen as omitted and would pass forever. Assert its presence directly."""
    p = competition.competition_payload()
    assert all(m["allow_negative_baseline"] is False for m in p["members"].values())


def test_the_REACH_PURITY_FLOOR_is_recorded_even_though_it_is_not_a_policy_field():
    """#327's half of #317. `de_wilcoxon_direction_reach_raw` thresholds its purity curve on
    `direction.REACH_PURITY_FLOOR`, and that number is not a Scoring field, not a config knob
    and not in the catalog -- so every mutation test in this file passed while the digest was
    blind to the one constant a scored member's arithmetic turns on. #322 moved it (0.975 ->
    0.9), the digest stood still, and the move had to be paid for with a `rule_version` bump.

    ⚠️ Asserted DIRECTLY, like `allow_negative_baseline` and `derived_components` above and
    for the same reason: `FROZEN_DIGEST` is filled from a green run, so a field omitted at
    that moment is frozen as omitted and passes forever after.
    """
    from cell_eval2.metrics import direction

    p = competition.competition_payload()
    assert p["reach_purity_floor"] == direction.REACH_PURITY_FLOOR
    assert p["reach_purity_floor"] == 0.9        # the shipped, calibrated value (#322)


def test_the_digest_moves_when_the_REACH_PURITY_FLOOR_moves(monkeypatch):
    """The wiring, and it is not the same claim as the presence test. `competition_payload`
    imports the constant INSIDE the function, so it re-reads the live value on every call --
    a module-level import, or a literal `0.9` in the payload, satisfies the presence
    assertion identically while freezing a number that no longer describes the member.

    Patched at its SOURCE module for exactly that reason; there is no name in `competition`
    to patch.
    """
    before = competition.competition_digest()
    monkeypatch.setattr("cell_eval2.metrics.direction.REACH_PURITY_FLOOR", 0.975)
    assert competition.competition_digest() != before
    monkeypatch.undo()
    assert competition.competition_digest() == before        # ...and nothing else moved


@pytest.mark.parametrize("const,new", [
    ("N_SPLITS", 7), ("BASE_SEED", 123), ("BULK_TARGET_SUM", 1e6),
    ("COMPARISON_STATISTIC", "median"), ("CONTROL_SOURCE_SCORED", "pred"),
    ("CONTROL_SOURCE_ANCHOR", "real"),
])
def test_the_digest_moves_when_the_production_rule_moves(const, new, monkeypatch):
    before = competition.competition_digest()
    monkeypatch.setattr(competition, const, new)
    assert competition.competition_digest() != before


@pytest.mark.parametrize("member,field,new", [
    # ⚠️ The member is not free: `MetricSpec.__post_init__` couples `agg='ratio_of_sums'` with
    # `derived`, forbids `worst_value` on a derived metric, and requires exactly one of `func`
    # and `derived`. So `replace(mse_spec, agg='median')` RAISES rather than moving the digest,
    # and the test would error instead of passing. `agg`, `worst_value` and `derived` are
    # therefore mutated on a NON-derived member; `normalization` is legal on either.
    ("pds_cosine", "agg", "median"),
    ("pds_cosine", "worst_value", 0.0),
    ("pds_cosine", "normalization", "lognorm"),
    ("expr_mse_unbiased_capped_norm", "normalization", "lognorm"),
])
def test_the_digest_moves_when_what_the_metric_IS_moves(member, field, new, monkeypatch):
    """Spec §5.7 group 5: `agg`, the derived pair and the resolved `normalization` change what
    the metric is -- they move u, b and r alike. Mutating the MetricSpec, never the payload."""
    from dataclasses import replace

    from cell_eval2.catalog import CATALOG

    before = competition.competition_digest()
    patched = dict(CATALOG)
    patched[member] = replace(CATALOG[member], **{field: new})
    monkeypatch.setattr(competition, "CATALOG", patched)
    assert competition.competition_digest() != before


def test_the_digest_moves_when_the_DERIVED_pair_moves(monkeypatch):
    """The one group-5 field with no legal single-field mutation on its own member: `derived`
    and `agg` imply each other, so the numerator is swapped in place instead."""
    from dataclasses import replace

    from cell_eval2.catalog import CATALOG

    before = competition.competition_digest()
    spec = CATALOG["expr_mse_unbiased_capped_norm"]
    patched = dict(CATALOG)
    patched["expr_mse_unbiased_capped_norm"] = replace(
        spec, derived=replace(spec.derived, numerator="expr_mse"))
    monkeypatch.setattr(competition, "CATALOG", patched)
    assert competition.competition_digest() != before


def test_the_digest_moves_when_the_ESTIMATOR_assignment_moves(monkeypatch):
    """Group 3's per-member estimator. `_lfc_nmae_names` decides which members use the
    full-real gate; a change there re-labels a member without touching any constant."""
    before = competition.competition_digest()
    monkeypatch.setattr("cell_eval2.anchor._lfc_nmae_names", lambda names: [])
    assert competition.competition_digest() != before


def test_the_digest_moves_when_the_SEED_DERIVATION_moves(monkeypatch):
    """Both halves of group 3: the documented rule string AND the literal seeds it yields.
    They are frozen separately because `_derive_seeds` can move every shipped anchor while
    the documented rule reads identically."""
    before = competition.competition_digest()
    monkeypatch.setattr("cell_eval2.anchor.SEED_DERIVATION", "something else")
    assert competition.competition_digest() != before
    monkeypatch.undo()
    monkeypatch.setattr("cell_eval2.anchor._derive_seeds", lambda b, n: list(range(n)))
    assert competition.competition_digest() != before


def _meta(**over):
    """An anchor sidecar shaped exactly as `anchor.build_meta` stamps one."""
    m = {"base_seed": competition.BASE_SEED, "n_splits": competition.N_SPLITS,
         "seed_derivation": competition.competition_payload()["seed_derivation"],
         "derived_seeds": list(competition.derived_seeds()),
         "bulk_target_sum": competition.BULK_TARGET_SUM,
         "control_source_effective": competition.CONTROL_SOURCE_ANCHOR}
    m.update(over)
    return m


def _estimators(**over):
    from cell_eval2.anchor import FULL_GATE_RAW, SPLIT_HALF_RAW

    e = {m: (FULL_GATE_RAW if "lfc_nmae" in m else SPLIT_HALF_RAW)
         for m in competition.competition_members()}
    e.update(over)
    return e


def _is(cfg=None, **over):
    kw = {"members": competition.competition_members(), "anchor_meta": _meta(),
          "estimators": _estimators()}
    kw.update(over)
    return competition.is_competition_rule(
        cfg if cfg is not None else EvalConfig.from_preset("vcc2026"), **kw)


def test_the_preset_IS_the_competition_rule():
    """Built from the shipped preset and a producer-shaped sidecar, not from hand-made values:
    this is the assertion that catches the preset and the rule drifting apart, which would make
    every official bundle silently diagnostic."""
    assert _is() == []


def test_membership_is_compared_as_a_SET_not_a_sequence():
    """The producer sorts the anchor frame alphabetically while PROFILES follows catalog
    insertion order, so an ordered comparison is False for EVERY real artifact -- the check
    would mark every real bundle diagnostic, and a unit test handed a pre-sorted profile-order
    tuple could never show it."""
    assert _is(members=tuple(sorted(competition.competition_members()))) == []


@pytest.mark.parametrize("over,needle", [
    ({"n_splits": 1}, "n_splits"),
    ({"base_seed": 123}, "base_seed"),
    ({"derived_seeds": [1, 2, 3, 4, 5]}, "derived_seeds"),
    ({"seed_derivation": "something else"}, "seed_derivation"),
    ({"bulk_target_sum": 1e6}, "bulk_target_sum"),
    ({"control_source_effective": "real"}, "control_source_effective"),
])
def test_a_foreign_ANCHOR_STAMP_is_named(over, needle):
    """Every one of these is read out of the artifact the producer returned, never out of the
    arguments it was called with -- a cached anchor can carry any of them."""
    out = _is(anchor_meta=_meta(**over))
    assert any(needle in m for m in out), out


def test_a_foreign_ESTIMATOR_is_named():
    from cell_eval2.anchor import SPLIT_HALF_RAW

    out = _is(estimators=_estimators(de_wilcoxon_lfc_nmae=SPLIT_HALF_RAW))
    assert any("estimator" in m for m in out), out


def test_a_narrowed_membership_is_named():
    out = _is(members=competition.competition_members()[:5])
    assert any("members" in m for m in out), out


@pytest.mark.parametrize("field,new,needle", [
    ("bulk_target_sum", 1e6, "config"),
    ("control_source", "pred", "config"),
    ("metrics", "full", "profile"),
    ("cache_strict", False, "cache_strict"),
    ("allow_fractional_counts", True, "config"),
])
def test_a_non_competition_config_field_is_named(field, new, needle):
    from dataclasses import replace

    cfg = replace(EvalConfig.from_preset("vcc2026"), **{field: new})
    out = _is(cfg)
    assert any(needle in m for m in out), out


@pytest.mark.parametrize("field,new", [("device", "cpu"), ("pert_chunk", 64),
                                       ("pert_col", "perturbation")])
def test_the_EXCLUDED_fields_do_NOT_make_a_bundle_diagnostic(field, new):
    """⚠️ Spec §5.7 rules device/pert_chunk/de.backend OUT of the rule, but
    `cache.config_hash` skips none of them, so hashing the config as-is would mark a bundle
    diagnostic for an explicit `device="cpu"` that resolves identically to the preset's
    "auto" on a CPU host. `rule_config_hash` normalizes exactly those, plus `pert_col`.

    ⚠️ `pert_col` is here on a MEASUREMENT: the deliverable panel labels its perturbation
    column `perturbation` while the preset declares `target`, and that ONE field was the only
    difference between the competition preset and the real production run -- so every official
    bundle came out diagnostic with everything else matching. It names WHERE the labels live,
    not what they mean, and `check_submission` still compares it through `config_digest`.

    ⚠️ This test is only meaningful because `is_competition_rule` does NOT compare the
    anchor's `config_hash` stamp: `build_meta` stamps the RAW hash (anchor.py:500), so a
    comparison against the normalized frozen hash would re-break exactly this case."""
    from dataclasses import replace

    assert _is(replace(EvalConfig.from_preset("vcc2026"), **{field: new})) == []


def _backend(name):
    """The preset with an explicitly-named rank DE backend -- the exclusion under test."""
    from dataclasses import replace

    cfg = EvalConfig.from_preset("vcc2026")
    return replace(cfg, de=replace(cfg.de, backend=name))


def test_the_digest_moves_when_a_DERIVED_COMPONENT_moves(monkeypatch):
    """⚠️ The parent's own fields do not cover its components. `expr_mse_unbiased_capped_norm`
    is computed from `expr_mse_unbiased_capped` / `expr_distance_unbiased`, both `scored=False`
    and therefore never reached by `competition_members()`. Moving a component's `agg` or its
    resolved normalization moves the parent's `u`, `b` and `r` alike while every field the
    payload records ABOUT THE PARENT stays bit-identical, and nothing in the catalog forces a
    component's normalization to agree with its parent's (codex checkpoint-2 P1)."""
    from dataclasses import replace

    from cell_eval2.catalog import CATALOG

    for component, field, new in (("expr_distance_unbiased", "normalization", "lognorm"),
                                  ("expr_mse_unbiased_capped", "worst_value", 0.0)):
        before = competition.competition_digest()
        patched = dict(CATALOG)
        patched[component] = replace(CATALOG[component], **{field: new})
        monkeypatch.setattr(competition, "CATALOG", patched)
        assert competition.competition_digest() != before, (component, field)
        monkeypatch.undo()


def test_the_derived_components_are_RECORDED_not_merely_mutable():
    """`FROZEN_DIGEST` is filled from a green run, so a field omitted at that moment is frozen
    as omitted and every mutation test above would still pass on the OTHER fields. Assert the
    block's presence and shape directly, and that a NON-derived member records None rather than
    an empty dict -- the two are different claims."""
    p = competition.competition_payload()
    comp = p["members"]["expr_mse_unbiased_capped_norm"]["derived_components"]
    assert set(comp) == {"expr_mse_unbiased_capped", "expr_distance_unbiased"}
    # `agg` is deliberately ABSENT: `run._derived_value` sums the components' per-perturbation
    # values directly and never reads either one's aggregation policy, so freezing it would
    # churn the digest for a change that cannot move a competition number.
    assert all(set(v) == {"normalization", "worst_value"} for v in comp.values())
    assert comp["expr_distance_unbiased"]["normalization"] == "bulk_lognorm"
    assert p["members"]["pds_cosine"]["derived_components"] is None


def test_the_digest_moves_when_the_CACHE_STRICT_REQUIREMENT_moves(monkeypatch):
    """`cache_strict` is checked outside the config hash, so a hardcoded `if not
    config.cache_strict` would be invisible to the digest: dropping it would widen the rule
    with nothing failing. The constant is recorded, and the check reads it -- so this test also
    proves the two are wired to each other rather than merely coexisting."""
    before = competition.competition_digest()
    monkeypatch.setattr(competition, "REQUIRE_CACHE_STRICT", False)
    assert competition.competition_digest() != before
    # ...and the CHECK follows the constant, which is what makes recording it meaningful.
    from dataclasses import replace

    assert _is(replace(EvalConfig.from_preset("vcc2026"), cache_strict=False)) == []


def test_the_digest_moves_when_the_RESOLVED_PROFILE_moves(monkeypatch):
    """`members` is the six SCORED metrics, so a DIAGNOSTIC joining or leaving `vcc2026` leaves
    it untouched -- while `is_competition_rule` compares the FULL resolved list, so which runs
    count as the competition really did change. `profile_resolved` is what makes that visible."""
    from cell_eval2.catalog import resolve_metrics as real

    before = competition.competition_digest()

    def wider(metrics, *, version):
        names, missing = real(metrics, version=version)
        return ([*names, "expr_mae"] if metrics == competition.COMPETITION_PROFILE
                else list(names)), missing

    monkeypatch.setattr(competition, "resolve_metrics", wider)
    assert competition.competition_digest() != before
    # `members` is unmoved, which is exactly why the new field was needed.
    assert competition.competition_payload()["members"].keys() == \
        {m: None for m in competition.competition_members()}.keys()


def test_the_gate_READS_the_frozen_profile_rather_than_recomputing_it(monkeypatch):
    """The wiring, asserted directly. `test_the_digest_moves_when_the_RESOLVED_PROFILE_moves`
    above patches the gate's `got` and the payload's `want` TOGETHER, so it stays green even if
    `is_competition_rule` reverts to a second `resolve_metrics` call (codex checkpoint-2 round
    3). Patching only the PAYLOAD separates them: a gate that recomputes ignores this and
    returns [], a gate that reads the payload names the profile."""
    real = competition.competition_payload

    def narrowed():
        p = real()
        return {**p, "profile_resolved": p["profile_resolved"][:-1]}

    monkeypatch.setattr(competition, "competition_payload", narrowed)
    out = _is()
    assert any("profile" in m for m in out), out


def test_the_digest_moves_when_the_EXCLUSION_SET_moves(monkeypatch):
    """The gap the `pert_col` exclusion exposed. `payload["config_hash"]` hashes the PRESET
    with each excluded field normalized to the preset's own value, so it is a no-op for every
    exclusion -- adding one was measured to leave the digest bit-identical, which made a change
    to WHICH runs count as the competition invisible to the frozen digest. The payload records
    the tuple for exactly this reason."""
    before = competition.competition_digest()
    monkeypatch.setattr(competition, "_RULE_EXCLUDED",
                        competition._RULE_EXCLUDED + ("control",))
    assert competition.competition_digest() != before
    monkeypatch.undo()
    # ...and the NESTED half. `de.backend` is normalized through `_RULE_EXCLUDED_NESTED` rather
    # than the flat tuple, so a payload that hand-listed the string would leave this exclusion
    # in the very blind spot the field exists to close: dropping the entry must move the digest,
    # and must also make `de.backend` start affecting the hash again.
    assert competition.rule_config_hash(_backend("pdex")) == competition.rule_config_hash(
        EvalConfig.from_preset("vcc2026"))
    monkeypatch.setattr(competition, "_RULE_EXCLUDED_NESTED", ())
    assert competition.competition_digest() != before
    assert competition.rule_config_hash(_backend("pdex")) != competition.rule_config_hash(
        EvalConfig.from_preset("vcc2026"))


@pytest.mark.parametrize("field,new", [("control", "ctrl"), ("target_gene_map", "/tmp/m.csv")])
def test_the_two_NEIGHBOURING_fields_still_DO(field, new):
    """The other half of the `pert_col` exclusion, and the reason it is narrow. `control`
    selects which cells are the reference and `target_gene_map` re-maps genes; both move every
    number, so both must stay INSIDE the rule. Without this a later "tidy-up" that swept them
    into `_RULE_EXCLUDED` alongside `pert_col` would pass every other test in this file."""
    from dataclasses import replace

    cfg = replace(EvalConfig.from_preset("vcc2026"), **{field: new})
    assert competition.rule_config_hash(cfg) != competition.rule_config_hash(
        EvalConfig.from_preset("vcc2026"))
    assert any("config" in m for m in _is(cfg))


def test_an_explicit_equivalent_metric_LIST_is_the_competition_profile():
    """`resolve_metrics` preserves the caller's order for an explicit list, so an ordered
    comparison would mark a reversed-but-identical list diagnostic. Frame order is normalized
    elsewhere and is frozen separately as `member_order_in_frame`."""
    from dataclasses import replace

    from cell_eval2.catalog import resolve_metrics

    names = list(resolve_metrics("vcc2026", version="v2")[0])
    cfg = replace(EvalConfig.from_preset("vcc2026"), metrics=list(reversed(names)))
    assert _is(cfg) == []


def test_a_RANK_de_backend_is_excluded_but_deseq2_is_NOT():
    """`de.backend` is excluded from the hash, and for the rank backends that is the whole
    story -- they change the engine, not the metric.

    ⚠️ `deseq2` is different and must NOT read as excluded: `metric_output_names` relabels the
    selected members to `de_deseq2_*` (run.py:1406) while `competition_members()` stays
    `de_wilcoxon_*`, and the two do not canonicalize to each other. A deseq2 bundle is
    therefore diagnostic -- caught by the MEMBERSHIP check rather than by the hash, which is
    why the exclusion is safe to keep."""
    from dataclasses import replace

    cfg = EvalConfig.from_preset("vcc2026")
    assert _is(replace(cfg, de=replace(cfg.de, backend="pdex"))) == []
    deseq2 = replace(cfg, de=replace(cfg.de, backend="deseq2"))
    out = _is(deseq2, members=tuple(m.replace("de_wilcoxon_", "de_deseq2_")
                                    for m in competition.competition_members()))
    assert any("members" in m for m in out), out


def test_allow_fractional_counts_is_part_of_the_rule_hash():
    """⚠️ `cache.config_hash` RETAINS this field, and `prep-real-bundle` flips it True for its
    baseline leg. Hashing the flipped config would make every bundle diagnostic, so the
    producer must hash the user-facing config. Pinned here because the failure is silent --
    the bundle still builds."""
    from dataclasses import replace

    cfg = EvalConfig.from_preset("vcc2026")
    assert competition.rule_config_hash(replace(cfg, allow_fractional_counts=True)) \
        != competition.rule_config_hash(cfg)


def test_the_payload_is_not_strict_json():
    """Pins a KNOWN wart so changing it is deliberate (Copilot round 5).

    Unflooring the four bounded members makes their `effective_clamp_low()` `-inf`, which
    `json.dumps` emits as the non-standard `-Infinity` token. That is fine here and only here:
    the payload is hashed in-process and compared in memory, and a bundle stores the resulting
    hex `rule_digest` rather than the payload, so nothing on disk and no non-Python reader ever
    sees it. This test exists so that if someone starts serializing the payload for real, they
    trip over the decision rather than shipping a file no strict parser can read.
    """
    import json

    from cell_eval2.scoring import _isfinite

    payload = competition.competition_payload()
    unfloored = {m for m, v in payload["members"].items() if not _isfinite(v["clamp_low"])}
    assert unfloored == {
        "pds_cosine",
        "de_wilcoxon_direction_fidelity_yield_raw",
        "de_wilcoxon_direction_reach_raw",
        "de_wilcoxon_sig_jaccard",
    }
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    assert "-Infinity" in blob
    with pytest.raises(ValueError):                      # a STRICT parser refuses it
        json.loads(blob, parse_constant=_reject)


def _reject(token):
    raise ValueError(f"non-standard JSON constant: {token}")
