import pytest

from cell_eval2.catalog import (
    CATALOG,
    PROFILES,
    DerivedAgg,
    MetricSpec,
    is_decisive,
    needs_moments_transitively,
    resolve_metrics,
)
from cell_eval2.norm import EXPR_COMPARATOR
from cell_eval2.scoring import DIAG, ERROR


def _spec(**kw):
    """A minimal valid spec; callers override exactly the field under test."""
    base = dict(name="x", func=lambda **_: {}, scoring=ERROR, agg="mean",
                profiles=("full",), kind="anndata", normalization="lognorm")
    base.update(kw)
    return MetricSpec(**base)


def test_mae_registered():
    spec = CATALOG["expr_mae"]
    assert spec.scoring.scored and spec.scoring.direction == "lower"
    assert spec.kind == "anndata"
    assert spec.normalization == EXPR_COMPARATOR


def test_discrimination_variants_registered():
    for name in ("pds_l1", "pds_l2", "pds_cosine"):
        spec = CATALOG[name]
        assert spec.scoring.scored and spec.scoring.direction == "higher"
        assert spec.scoring.anchor == 1.0      # higher is better, best = 1
        assert spec.kind == "anndata"
        assert spec.normalization == EXPR_COMPARATOR


def test_anndata_delta_metrics_registered():
    expected = {
        "delta_pearson": ("pearson_delta", "higher", EXPR_COMPARATOR),
        "expr_mse": ("mse", "lower", EXPR_COMPARATOR),
        "delta_mse": ("mse_delta", "lower", EXPR_COMPARATOR),
        "delta_mae": ("mae_delta", "lower", EXPR_COMPARATOR),
    }
    for name, (v1, direction, normalization) in expected.items():
        spec = CATALOG[name]
        assert spec.v1_name == v1
        assert spec.scoring.scored and spec.scoring.direction == direction
        assert spec.kind == "anndata"
        assert spec.normalization == normalization


def test_pr2_moves_every_anndata_metric_to_the_expression_comparator():
    for name in (
        "pds_l1", "pds_l2", "pds_cosine", "delta_pearson", "delta_mse", "delta_mae",
        "expr_mae", "expr_mse", "expr_mse_unbiased", "expr_mse_unbiased_capped",
        "expr_distance_unbiased", "expr_mse_unbiased_capped_norm",
    ):
        assert CATALOG[name].normalization == EXPR_COMPARATOR, name


def test_v1_still_resolves_every_moved_metric_to_lognorm():
    from cell_eval2 import norm, run as R

    comparator = norm.resolve_comparator(
        version="v1", pred_input_type="counts", real_input_type="counts",
    )
    assert comparator == "lognorm"

    for name in ("pds_l1", "delta_pearson"):
        assert R.effective_normalization(CATALOG[name], comparator) == "lognorm"


def test_anndata_delta_metrics_resolve_by_v1_and_canonical():
    for v1, canon in [("pearson_delta", "delta_pearson"), ("mse", "expr_mse"),
                      ("mse_delta", "delta_mse"), ("mae_delta", "delta_mae")]:
        a1, m1 = resolve_metrics([v1])
        assert a1 == [canon] and m1 == []
        a2, m2 = resolve_metrics([canon])
        assert a2 == [canon] and m2 == []


def test_anndata_delta_metrics_in_profiles():
    av_min, _ = resolve_metrics("minimal")
    assert {"delta_pearson", "expr_mse"} <= set(av_min)
    assert "delta_mse" not in av_min and "delta_mae" not in av_min
    av_full, _ = resolve_metrics("full")
    assert {"delta_pearson", "expr_mse", "delta_mse", "delta_mae"} <= set(av_full)
    av_ad, _ = resolve_metrics("anndata")
    assert {"delta_pearson", "expr_mse", "delta_mse", "delta_mae"} <= set(av_ad)


def test_the_three_component_metrics_are_never_scored():
    for name in ("expr_mse_unbiased", "expr_mse_unbiased_capped", "expr_distance_unbiased"):
        spec = CATALOG[name]
        assert spec.scoring.scored is False, (
            f"{name} is scored; it is in gene-averaged expression units (panel-dependent) "
            "and the uncapped one is the submitter lever #247 closed"
        )


def test_real_mass_ratio_is_a_comparator_space_diagnostic():
    spec = CATALOG["expr_real_mass_ratio"]
    assert spec.scoring == DIAG
    assert spec.agg == "mean"
    assert spec.kind == "anndata"
    assert spec.normalization == EXPR_COMPARATOR
    assert spec.profiles == ("full", "anndata", "vcc2026")


def test_the_removed_metric_has_no_alias_anywhere():
    # It must be NAMED here: this asserts the absence of a specific retired spelling, and a
    # test that cannot say which one asserts nothing (#257 removed it with no alias on
    # purpose -- the same policy that governed the expr_mse_unbiased -> _norm rename).
    from cell_eval2.catalog import _NAME_TO_CANONICAL
    assert "expr_mse_unbiased_norm" not in CATALOG
    assert "expr_mse_unbiased_norm" not in _NAME_TO_CANONICAL, (
        "an alias would let an old column bind silently to a metric with a different "
        "definition -- the same policy that governed the expr_mse_unbiased -> _norm rename"
    )


def test_asking_for_the_derived_metric_alone_resolves_its_components():
    # Dispatch skips a derived metric (it has no func) and the aggregators build it from its
    # components' columns, so a request naming it alone would compute nothing and emit no
    # aggregate row -- silently. Checkpoint-2 review, #257.
    available, missing = resolve_metrics(["expr_mse_unbiased_capped_norm"])
    assert missing == []
    assert available[0] == "expr_mse_unbiased_capped_norm", "the request's own order moved"
    assert set(available) == {"expr_mse_unbiased_capped_norm", "expr_mse_unbiased_capped",
                              "expr_distance_unbiased"}


def test_needs_moments_transitively_covers_derived_components_and_resolution():
    derived = "expr_mse_unbiased_capped_norm"
    assert CATALOG[derived].needs_moments is False
    assert needs_moments_transitively(derived) is True

    for name in (
        "expr_mse_unbiased", "expr_mse_unbiased_capped", "expr_distance_unbiased",
    ):
        assert needs_moments_transitively(name) is True
    assert needs_moments_transitively("expr_mse") is False

    direct_only = [m for m in PROFILES["full"] if not CATALOG[m].needs_moments]
    direct_only_resolved, _ = resolve_metrics(direct_only)
    assert [m for m in direct_only_resolved if CATALOG[m].needs_moments] == [
        "expr_mse_unbiased_capped", "expr_distance_unbiased",
    ]

    transitive = [m for m in PROFILES["full"] if not needs_moments_transitively(m)]
    transitive_resolved, _ = resolve_metrics(transitive)
    assert not [m for m in transitive_resolved if CATALOG[m].needs_moments]


def test_closing_the_derived_dependency_adds_nothing_to_a_plain_request():
    # The closure must fire ONLY for a derived metric; a plain list is untouched.
    available, _ = resolve_metrics(["expr_mae", "expr_mse"])
    assert available == ["expr_mae", "expr_mse"]


def test_the_derived_metric_is_scored_decisive_and_has_no_func():
    spec = CATALOG["expr_mse_unbiased_capped_norm"]
    assert spec.func is None
    assert spec.agg == "ratio_of_sums"
    assert spec.scoring.scored is True
    assert is_decisive(spec) is True
    assert spec.derived.numerator == "expr_mse_unbiased_capped"
    assert spec.derived.denominator == "expr_distance_unbiased"


def test_resolve_profile_splits_available_and_missing():
    available, missing = resolve_metrics("vcc")
    assert "expr_mae" in available
    assert "pds_l1" in available                    # now implemented
    assert "de_wilcoxon_overlap" in available        # now implemented
    assert missing == []


def test_resolve_explicit_list():
    available, missing = resolve_metrics(["mae"])
    assert available == ["expr_mae"] and missing == []


def test_resolve_unknown_profile_raises():
    with pytest.raises(ValueError, match="profile"):
        resolve_metrics("does-not-exist")


def test_de_overlap_variants_registered():
    for name in ["de_wilcoxon_overlap", "de_wilcoxon_overlap_top50", "de_wilcoxon_overlap_top100",
                 "de_wilcoxon_overlap_top200", "de_wilcoxon_overlap_top500",
                 "de_wilcoxon_precision", "de_wilcoxon_precision_top50", "de_wilcoxon_precision_top100",
                 "de_wilcoxon_precision_top200", "de_wilcoxon_precision_top500"]:
        assert name in CATALOG
        assert CATALOG[name].kind == "de"
        assert CATALOG[name].normalization is None
        assert CATALOG[name].scoring.scored
        assert CATALOG[name].scoring.direction == "higher"


def test_new_de_metric_variants_registered():
    # the scored direction, or None where the metric is diagnostic. `None` means
    # scored=False, NOT direction=None -- the nsig counts are the only entries with no
    # direction at all, and they are exactly the ones that stay out of avg_score.
    scored_direction = {
        "de_wilcoxon_nsig_spearman": "higher",
        "de_wilcoxon_lfc_spearman": "higher",
        "de_wilcoxon_lfc_spearman_pos": "higher",
        "de_wilcoxon_lfc_spearman_neg": "higher",
        "de_wilcoxon_direction_match": "higher",
        "de_wilcoxon_model_direction_match": "higher",
        "de_wilcoxon_sig_recall": "higher",
        "de_wilcoxon_pr_auc": "higher",
        "de_wilcoxon_roc_auc": "higher",
        "de_wilcoxon_nsig_counts_real": None,
        "de_wilcoxon_nsig_counts_pred": None,
        "de_wilcoxon_direction_precision": "higher",
        "de_wilcoxon_direction_sensitivity": "higher",
        "de_wilcoxon_direction_sensitivity_universe": "higher",
    }
    for name, direction in scored_direction.items():
        assert name in CATALOG
        spec = CATALOG[name]
        assert spec.kind == "de"
        assert spec.normalization is None
        if direction is None:
            assert not spec.scoring.scored, name
        else:
            assert spec.scoring.scored and spec.scoring.direction == direction, name


def test_vcc_profile_fully_resolved():
    available, missing = resolve_metrics("vcc")
    assert "de_wilcoxon_overlap" in available
    assert missing == []  # vcc = [expr_mae, pds_l1, de_wilcoxon_overlap], all implemented


def test_metricspec_has_v1name_and_aliases_defaults():
    spec = MetricSpec(name="expr_mae", func=lambda: {}, scoring=ERROR, agg="mean",
                      profiles=("full",), kind="anndata", normalization="lognorm")
    assert spec.v1_name is None
    assert spec.aliases == ()
    spec2 = MetricSpec(name="expr_mae", func=lambda: {}, scoring=ERROR, agg="mean",
                       profiles=("full",), kind="anndata", normalization="lognorm",
                       v1_name="mae", aliases=("MAE",))
    assert spec2.v1_name == "mae"
    assert spec2.aliases == ("MAE",)


_TOPK = [f"de_wilcoxon_{m}_top{k}"
         for m in ("overlap", "precision") for k in (50, 100, 200, 500)]


def test_de_profile_includes_topk():
    # #19: metrics="de" must resolve to all DE metrics, including the 8 top-k
    # variants. Before the fix PROFILES["de"] hand-listed only 10, silently
    # dropping overlap/precision_top{50,100,200,500}.
    available, missing = resolve_metrics("de")
    assert set(_TOPK) <= set(available)
    assert len(available) == 41          # 21 DE + 4 chance-corrected (#14) + 3 direction (v0.5.0) + 11 chance-corrected direction (#195) + 1 sig_jaccard + 1 lfc_nmae (#208)
    assert missing == []


def test_full_profile_includes_topk():
    available, missing = resolve_metrics("full")
    assert set(_TOPK) <= set(available)
    assert len(available) == 54          # 13 anndata + 41 DE (including 11 chance-corrected direction metrics, #195, + 1 sig_jaccard + 1 lfc_nmae (#208))
    assert missing == []


def test_profiles_single_source_of_truth():
    # PROFILES is derived from MetricSpec.profiles: a metric is in PROFILES[P]
    # iff its .profiles declares P. Both directions; lists are unique and name
    # only real CATALOG metrics.
    for profile, names in PROFILES.items():
        assert len(names) == len(set(names)), f"duplicate metrics in profile {profile}"
        for name in names:
            assert name in CATALOG, name            # clean failure, not a KeyError below
            assert profile in CATALOG[name].profiles, (name, profile)
    for name, spec in CATALOG.items():
        for profile in spec.profiles:
            assert name in PROFILES[profile], (name, profile)


def test_profiles_keys_match_declared_tags():
    tags = {p for spec in CATALOG.values() for p in spec.profiles}
    assert set(PROFILES) == tags


@pytest.mark.parametrize("profile,expected", [
    ("vcc", {"expr_mae", "pds_l1", "de_wilcoxon_overlap"}),
    ("vcc2026", {"pds_cosine", "expr_mse_unbiased", "expr_mse_unbiased_capped",
                 "expr_distance_unbiased", "expr_real_mass_ratio",
                 "expr_mse_unbiased_capped_norm",
                 "de_wilcoxon_lfc_nmae",
                 "de_wilcoxon_direction_fidelity_yield_raw",
                 "de_wilcoxon_direction_reach_raw",
                 "de_wilcoxon_sig_jaccard"}),
    ("minimal", {"expr_mae", "pds_l1", "de_wilcoxon_overlap", "de_wilcoxon_precision",
                 "de_wilcoxon_nsig_counts_real", "de_wilcoxon_nsig_counts_pred",
                 "delta_pearson", "expr_mse"}),
    ("anndata", {"expr_mae", "pds_l1", "pds_l2", "pds_cosine",
                 "delta_pearson", "expr_mse", "expr_mse_unbiased",
                 "expr_mse_unbiased_capped", "expr_distance_unbiased",
                 "expr_real_mass_ratio",
                 "expr_mse_unbiased_capped_norm", "delta_mse", "delta_mae"}),
    ("pds", {"pds_l1"}),
])
def test_profile_available_sets(profile, expected):
    # Locks each profile's available-set after adding the 4 anndata metrics
    # (delta_pearson/expr_mse join minimal+anndata; delta_mse/delta_mae join anndata),
    # plus #257's two derived components, its uncapped audit sibling and the derived metric,
    # and #264's fourth diagnostic `expr_real_mass_ratio` -- four unscored diagnostics in all,
    # which join full+anndata+vcc2026.
    # vcc2026 is pinned alongside vcc so that adding the 2026 profile is provably a
    # REPLACEMENT set and not an extension: vcc must still be exactly its three metrics.
    available, _ = resolve_metrics(profile)
    assert set(available) == expected


def test_every_SCORED_vcc2026_member_is_decisive():
    """Every SCORED vcc2026 member fails loud on a degenerate baseline, so the competition's
    six-metric average cannot silently shrink its denominator (#255). Narrowed by #257: the
    profile now also carries three UNSCORED diagnostics -- the derived metric's components --
    and an unscored metric is never compared against a baseline, so it cannot be decisive.
    The earlier objection conflated an accurate technical replicate (where the removed
    `expr_mse_unbiased_norm` was often negative) with the deployed generic-response baseline,
    whose values remain safely above zero."""
    from cell_eval2.catalog import is_decisive
    names, _ = resolve_metrics("vcc2026")
    scored = {n for n in names if CATALOG[n].scoring.scored}
    unscored = set(names) - scored
    assert unscored, "no unscored members: this test has degenerated back into the old one"
    assert {n for n in scored if is_decisive(CATALOG[n])} == scored
    assert not any(is_decisive(CATALOG[n]) for n in unscored), (
        f"unscored vcc2026 members are decisive: "
        f"{sorted(n for n in unscored if is_decisive(CATALOG[n]))}"
    )


def test_resolve_unknown_explicit_metric_raises():
    import pytest
    from cell_eval2.catalog import resolve_metrics
    with pytest.raises(ValueError, match="unknown metric"):
        resolve_metrics(["expr_mae", "delta_pearsn"])  # typo


def test_resolve_known_deferred_metric_is_missing_not_error():
    from cell_eval2.catalog import resolve_metrics
    available, missing = resolve_metrics(["expr_mae", "edistance_pearson"])
    assert "expr_mae" in available
    assert "edistance_pearson" in missing  # deferred, not a typo -> no raise


def test_resolve_all_typo_explicit_list_raises():
    import pytest
    from cell_eval2.catalog import resolve_metrics
    with pytest.raises(ValueError, match="unknown metric"):
        resolve_metrics(["nope1", "nope2"])


def test_resolve_unknown_metric_in_tuple_also_raises():
    # Non-list explicit input (tuple) must not bypass the typo check.
    import pytest
    from cell_eval2.catalog import resolve_metrics
    with pytest.raises(ValueError, match="unknown metric"):
        resolve_metrics(("expr_mae", "delta_pearsn"))


def test_a_derived_spec_needs_no_func():
    spec = _spec(func=None, agg="ratio_of_sums",
                 derived=DerivedAgg(numerator="a", denominator="b"))
    assert spec.func is None
    assert spec.derived == DerivedAgg(numerator="a", denominator="b")


def test_a_spec_may_not_carry_both_a_func_and_a_derivation():
    with pytest.raises(ValueError, match="exactly one of"):
        _spec(agg="ratio_of_sums", derived=DerivedAgg(numerator="a", denominator="b"))


def test_a_spec_with_neither_func_nor_derivation_is_refused():
    with pytest.raises(ValueError, match="exactly one of"):
        _spec(func=None)


def test_a_derived_spec_must_declare_ratio_of_sums():
    with pytest.raises(ValueError, match="agg='ratio_of_sums'"):
        _spec(func=None, agg="mean", derived=DerivedAgg(numerator="a", denominator="b"))


def test_ratio_of_sums_without_a_derivation_is_refused():
    with pytest.raises(ValueError, match="agg='ratio_of_sums'"):
        _spec(agg="ratio_of_sums")


def test_a_derived_spec_may_not_carry_a_worst_value():
    with pytest.raises(ValueError, match="worst_value"):
        _spec(func=None, agg="ratio_of_sums", worst_value=0.0,
              derived=DerivedAgg(numerator="a", denominator="b"))


def test_a_derived_spec_may_not_ask_for_moments():
    # Its components carry `needs_moments`; asking here would route a cache artifact to a
    # metric that never runs, which is silent waste rather than a visible error.
    with pytest.raises(ValueError, match="needs_moments"):
        _spec(func=None, agg="ratio_of_sums", needs_moments=True,
              derived=DerivedAgg(numerator="a", denominator="b"))


def test_every_shipped_derived_metric_has_its_components_in_every_profile_it_claims():
    # A derived metric in a profile whose components are absent cannot be computed on a run
    # of that profile -- it would silently vanish from the aggregate.
    offenders = []
    for name, spec in CATALOG.items():
        if spec.derived is None:
            continue
        for side in (spec.derived.numerator, spec.derived.denominator):
            comp = CATALOG[side]
            missing = set(spec.profiles) - set(comp.profiles)
            if missing:
                offenders.append((name, side, sorted(missing)))
    assert offenders == [], f"derived metrics whose components miss profiles: {offenders}"


def test_an_unscored_profile_member_is_not_decisive():
    # An unscored metric is never compared against a baseline, so a degenerate baseline for
    # it cannot decide a ranking -- aborting a run over one would be a false alarm.
    spec = _spec(scoring=DIAG, profiles=("full", "vcc2026"))
    assert spec.scoring.scored is False, "fixture must be unscored or this asserts nothing"
    assert is_decisive(spec) is False


#: The ONLY metrics the `scored` term is allowed to move. Two of them shipped before #257:
#: `de_wilcoxon_nsig_counts_real` / `_pred` are unscored but v1-emitted, so the old predicate
#: called them decisive. That demotion is INERT, measured at every call site: `score_metrics`
#: skips unscored metrics before it reaches `is_decisive`, `baseline._degenerate_metrics` does
#: the same before recording `decisive`, and the `score` CLI precheck reads that same list. No
#: consumer ever asks whether an unscored metric is decisive. #257 adds three unscored
#: component metrics, and #264 adds the real-mass diagnostic, as `vcc2026` members.
EXPECTED_MOVED_BY_SCORED_TERM = {
    "de_wilcoxon_nsig_counts_real", "de_wilcoxon_nsig_counts_pred",
    "expr_mse_unbiased", "expr_mse_unbiased_capped", "expr_distance_unbiased",
    "expr_real_mass_ratio",
}


def test_the_scored_term_moves_exactly_the_unscored_gate_members_and_nothing_else():
    # Recompute the OLD predicate and diff. Asserting "nothing moved" would go permanently red
    # the moment Task 6 lands -- the three new entries are unscored `vcc2026` members, which
    # is precisely the case this term exists to handle.
    moved = set()
    for name, spec in CATALOG.items():
        old = (bool(spec.v1_available)
               or "vcc" in spec.profiles
               or "vcc2026" in spec.profiles)
        if is_decisive(spec) != old:
            moved.add(name)
            assert spec.scoring.scored is False, (
                f"{name} moved but is SCORED -- the term must only ever demote unscored "
                "metrics, so this is a real regression"
            )
    # Before Task 6 the expected set is not yet in the catalog; compare against what exists.
    expected = EXPECTED_MOVED_BY_SCORED_TERM & set(CATALOG)
    assert moved == expected, (
        f"the `scored` term moved {sorted(moved)}; expected exactly {sorted(expected)}. "
        "Any other name here is a metric silently losing its fail-loud behaviour."
    )
