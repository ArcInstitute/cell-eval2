import polars as pl
import pytest

from cell_eval2 import partition


def _write_comparator_sidecar(root, subset_id, comparator):
    meta = {
        "real_ref_fingerprint": "rf",
        "config_hash": "cf",
        "perturbations": [subset_id],
    }
    if comparator is not None:
        meta["comparator"] = comparator
    partition.write_partial(
        pl.DataFrame({"perturbation": [subset_id], "metric": ["mae"], "value": [1.0]}),
        str(root), subset_id=subset_id, meta=meta,
    )


def test_fraction_index_partitions_perts():
    perts = [f"p{i}" for i in range(10)]
    parts = [partition.select_subset(perts, fraction=3, index=i) for i in range(3)]
    flat = sorted(x for s in parts for x in s)
    assert flat == sorted(perts)  # cover all, no overlap
    assert all(parts)  # none empty


def test_subset_filters_to_requested():
    perts = ["a", "b", "c", "d"]
    assert partition.select_subset(perts, subset=["b", "d", "zzz"]) == ["b", "d"]


def test_no_selector_returns_all():
    perts = ["a", "b", "c"]
    assert partition.select_subset(perts) == ["a", "b", "c"]


def test_bad_index_raises():
    with pytest.raises(ValueError, match="index must be in"):
        partition.select_subset(["a", "b"], fraction=2, index=5)


def test_write_and_aggregate_roundtrip(tmp_path):
    out = str(tmp_path)
    meta = {"real_ref_fingerprint": "rf", "config_hash": "cf",
            "comparator": "lognorm", "code_version": "t"}
    d1 = pl.DataFrame({"perturbation": ["A"], "metric": ["mae"], "value": [1.0]})
    d2 = pl.DataFrame({"perturbation": ["B"], "metric": ["mae"], "value": [3.0]})
    partition.write_partial(d1, out, subset_id="s0", meta={**meta, "perturbations": ["A"]})
    partition.write_partial(d2, out, subset_id="s1", meta={**meta, "perturbations": ["B"]})
    full, agg = partition.aggregate_partials(out)
    assert set(full["perturbation"]) == {"A", "B"}
    assert agg.filter(pl.col("metric") == "mae")["mean"][0] == 2.0


def test_aggregate_refuses_mixed_reference(tmp_path):
    out = str(tmp_path)
    partition.write_partial(
        pl.DataFrame({"perturbation": ["A"], "metric": ["mae"], "value": [1.0]}),
        out, subset_id="s0",
        meta={"real_ref_fingerprint": "rf", "config_hash": "cf", "comparator": "lognorm",
              "perturbations": ["A"]},
    )
    partition.write_partial(
        pl.DataFrame({"perturbation": ["B"], "metric": ["mae"], "value": [2.0]}),
        out, subset_id="s1",
        meta={"real_ref_fingerprint": "DIFFERENT", "config_hash": "cf",
              "comparator": "lognorm", "perturbations": ["B"]},
    )
    with pytest.raises(ValueError, match="differ"):
        partition.aggregate_partials(out)


def test_aggregate_refuses_duplicate_pert_metric(tmp_path):
    out = str(tmp_path)
    meta = {"real_ref_fingerprint": "rf", "config_hash": "cf", "comparator": "lognorm"}
    partition.write_partial(
        pl.DataFrame({"perturbation": ["A"], "metric": ["mae"], "value": [1.0]}),
        out, subset_id="s0", meta={**meta, "perturbations": ["A"]},
    )
    partition.write_partial(
        pl.DataFrame({"perturbation": ["A"], "metric": ["mae"], "value": [9.0]}),
        out, subset_id="s1", meta={**meta, "perturbations": ["A"]},
    )
    with pytest.raises(ValueError, match="duplicate"):
        partition.aggregate_partials(out)


def test_aggregate_empty_dir_raises(tmp_path):
    with pytest.raises(ValueError, match="no partial sidecars"):
        partition.aggregate_partials(str(tmp_path))


def test_write_partial_rejects_path_subset_id(tmp_path):
    with pytest.raises(ValueError, match="bare name"):
        partition.write_partial(
            pl.DataFrame({"perturbation": ["A"], "metric": ["mae"], "value": [1.0]}),
            str(tmp_path), subset_id="../escape", meta={},
        )


def test_aggregate_rejects_sidecar_missing_guard_fields(tmp_path):
    """A sidecar missing real_ref_fingerprint/config_hash must be rejected, not silently
    pass the cross-partial guard (PR #56 review)."""
    import json
    import os

    import polars as pl
    import pytest

    from cell_eval2 import partition

    out = str(tmp_path)
    partition.write_partial(
        pl.DataFrame({"perturbation": ["A"], "metric": ["mae"], "value": [1.0]}),
        out, subset_id="s0",
        meta={"real_ref_fingerprint": "rf", "config_hash": "cf", "comparator": "lognorm",
              "perturbations": ["A"]},
    )
    pl.DataFrame({"perturbation": ["B"], "metric": ["mae"], "value": [2.0]}).write_parquet(
        os.path.join(out, "s1.parquet")
    )
    with open(os.path.join(out, "s1.json"), "w", encoding="utf-8") as fh:
        json.dump({"subset_id": "s1", "real_ref_fingerprint": "rf"}, fh)  # no config_hash
    with pytest.raises(ValueError, match="missing"):
        partition.aggregate_partials(out)


def test_aggregate_partials_rejects_sidecars_with_different_comparators(tmp_path):
    _write_comparator_sidecar(tmp_path, "a", "bulk_lognorm")
    _write_comparator_sidecar(tmp_path, "b", "lognorm")
    with pytest.raises(ValueError, match="comparator"):
        partition.aggregate_partials(str(tmp_path))


@pytest.mark.parametrize("drop", ["a", "b", "both"])
def test_aggregate_partials_fails_closed_on_a_missing_comparator(tmp_path, drop):
    _write_comparator_sidecar(
        tmp_path, "a", None if drop in ("a", "both") else "bulk_lognorm")
    _write_comparator_sidecar(
        tmp_path, "b", None if drop in ("b", "both") else "bulk_lognorm")
    with pytest.raises(ValueError, match="comparator"):
        partition.aggregate_partials(str(tmp_path))


# --- #246: the cross-partial RESULT-SEMANTICS guard --------------------------------------------

def _sem_partial(root, subset_id, *, semantics, metrics=("mae",), value=1.0):
    """`metrics=None` omits the key, modelling a sidecar that predates it -- the one state where
    the cross-sidecar semantics check is the only guard that can fire."""
    meta = {"real_ref_fingerprint": "rf", "config_hash": "cf", "comparator": "lognorm",
            "perturbations": [subset_id]}
    if metrics is not None:
        meta["metrics"] = sorted(metrics)
    if semantics is not None:
        meta[partition.PARTIAL_SEMANTICS_KEY] = semantics
    partition.write_partial(
        pl.DataFrame({"perturbation": [subset_id],
                      "metric": [(metrics or ("mae",))[0]], "value": [value]}),
        str(root), subset_id=subset_id, meta=meta,
    )


def test_the_purity_floor_moves_the_partial_semantics_payload(monkeypatch):
    """The floor must reach the payload, or two partials computed under 0.975 and 0.9 reduce
    together into a plausible aggregate over incompatible values -- #246's exact failure, for a
    scored `vcc2026` member.

    ⚠️ A PAYLOAD-mutation test, not an integration one: it does not write partials or call
    `aggregate_partials`. The refusal itself is already covered generically by
    `_semantics_diff`/`aggregate_partials` tests in this file, so what is unproven without this
    is only whether the floor is IN the payload those tests compare (codex round 3 caught the
    earlier name and docstring claiming the integration)."""
    import cell_eval2.metrics.direction as direction
    from cell_eval2.catalog import resolve_metrics
    names, _ = resolve_metrics("vcc2026", version="v2")
    before = partition.result_semantics(names, comparator="bulk_lognorm")
    monkeypatch.setattr(direction, "REACH_PURITY_FLOOR", 0.975)
    after = partition.result_semantics(names, comparator="bulk_lognorm")
    assert before != after, (
        "result_semantics does not carry the purity floor, so a partial directory straddling "
        "the 0.975 -> 0.9 move would aggregate silently"
    )
    assert partition._semantics_diff(before, after) == ["reach_purity_floor"]


def test_result_semantics_payload_covers_agg_derived_normalization_and_worst_value():
    """#246's minimum is "resolved metric names and their agg"; baseline.config_digest's #231
    model is what the payload follows, so the derived components and the RESOLVED per-metric
    normalization travel too -- plus `worst_value`, which codex-review caught missing: it is the v2
    no-droppable-NaN fill and `run._fill_no_drop` REPLACES a per-perturbation value with it, so it
    moves emitted numbers, not just their reduction."""
    from cell_eval2.catalog import CATALOG, resolve_metrics
    names, _ = resolve_metrics("vcc2026", version="v2")
    sem = partition.result_semantics(names, comparator="bulk_lognorm")
    assert {"schema", "metric_agg", "metric_derived", "metric_normalization",
            "metric_worst_value", "metric_kind", "metric_needs_moments",
            "de_rank_semantics", "pds_exclusion_semantics",
            # #172's counter, the third sibling: three scored vcc2026 members stopped scoring
            # each perturbation's own target gene, and none of the four cross-partial fields
            # can see that (they describe what was ASKED FOR).
            "ontarget_exclusion_semantics",
            # The raw `direction_reach` purity floor -- same idea, carried as the VALUE rather
            # than a counter because here the semantics ARE one number.
            "reach_purity_floor",
            # #271's counter, the fourth sibling and a rung LOWER than the other three: not a
            # member's policy nor even its arithmetic, but the PSEUDOBULK that arithmetic reads.
            # `prep._grouped_sums` reduces WIDE, so two pieces either side of it carry bulks
            # rounded differently under one metric name.
            "grouped_sum_reduction_semantics"} == set(sem)
    # The schema number is pinned EXACTLY. `<= {True, False}` and a `current - 1` stale test both
    # pass for a reverted constant or an all-False mapping, which codex-review round 3 called out as
    # non-discriminating. Bumped 2 -> 3 by #172, 3 -> 4 by the `reach_purity_floor` term and
    # 4 -> 5 by #271's `grouped_sum_reduction_semantics`, per this module's own "bump when the
    # payload gains a term" rule.
    assert sem["schema"] == 5
    # kind routes the dispatch; needs_moments selects a DIFFERENT cache artifact and computation.
    assert dict(sem["metric_kind"])["pds_cosine"] == "anndata"
    assert dict(sem["metric_kind"])["de_wilcoxon_lfc_nmae"] == "de"
    moments = dict(sem["metric_needs_moments"])
    # Expected keys from `names`, NOT from the mapping under test: deriving them from `moments`
    # itself means omitting any metric other than the two checked below still passes
    # (codex-review round 4).
    assert moments == {n: bool(CATALOG[n].needs_moments) for n in names}
    # Both polarities are present in this one profile, so the term cannot be an all-False (or
    # all-True) constant that any implementation reproduces.
    assert moments["pds_cosine"] is False
    assert moments["expr_mse_unbiased"] is True, moments
    agg = dict(sem["metric_agg"])
    assert agg["expr_mse_unbiased_capped_norm"] == "ratio_of_sums"   # not flattened to "mean"
    assert agg["pds_cosine"] == "mean"
    # The derived member's identity carries the two components it divides, not just its name.
    assert sem["metric_derived"] == [["expr_mse_unbiased_capped_norm", "ratio_of_sums",
                                     "expr_mse_unbiased_capped", "expr_distance_unbiased"]]
    norm = dict(sem["metric_normalization"])
    assert norm["pds_cosine"] == "bulk_lognorm" and norm["de_wilcoxon_lfc_nmae"] is None
    # worst_value is read from the catalog, not defaulted. MEASURED: no vcc2026 member sets one
    # (all ten are None), so the term is only non-trivial off the competition profile -- 25 catalog
    # entries do set it. Asserted on one of those, or this column would be an all-None constant
    # that any implementation reproduces.
    worst = dict(sem["metric_worst_value"])
    assert worst == {n: CATALOG[n].worst_value for n in worst}
    assert set(worst.values()) == {None}, "vcc2026 sets no worst_value -- recheck if this changes"
    off_profile = partition.result_semantics(["delta_pearson", "mae"], comparator="lognorm")
    assert dict(off_profile["metric_worst_value"])["delta_pearson"] == -1.0


def test_result_semantics_is_order_insensitive_but_content_sensitive():
    """Ordered pairs, sorted names: a request-order change must not move the payload, while a
    real change must."""
    a = partition.result_semantics(["expr_mae", "pds_cosine"], comparator="lognorm")
    b = partition.result_semantics(["pds_cosine", "expr_mae"], comparator="lognorm")
    assert a == b
    assert a != partition.result_semantics(["expr_mae", "pds_cosine"],
                                           comparator="bulk_lognorm")


def test_aggregate_refuses_a_partial_whose_semantics_THIS_BUILD_does_not_produce(tmp_path):
    """⚠️ The hole codex-review found in the first version of this guard, and the one that matters
    most. Cross-sidecar agreement is NOT sufficient: if EVERY sidecar declares the same OLD payload
    they agree with each other, the guard passed, and `aggregate_metrics` then reduced them with the
    CURRENT catalog -- so an all-`mean` set could be median-reduced after a catalog change with no
    error. Exactly the silent failure #246 is about, moved one step.

    Both sidecars here are mutually CONSISTENT and both are stale, which is why only the
    against-this-build check can catch them."""
    out = str(tmp_path)
    stale = {**partition.result_semantics(["mae"], comparator="lognorm"),
             "metric_agg": [["mae", "median"]]}
    _sem_partial(out, "s0", semantics=stale)
    _sem_partial(out, "s1", semantics=stale, value=3.0)
    with pytest.raises(ValueError, match="semantics this build does not produce"):
        partition.aggregate_partials(out)


def test_a_stale_SCHEMA_alone_is_refused(tmp_path):
    """The schema number is part of the payload precisely so a payload-shape change is itself a
    semantics change. A sidecar from a build with a different schema must not reduce."""
    out = str(tmp_path)
    sem = partition.result_semantics(["mae"], comparator="lognorm")
    _sem_partial(out, "s0", semantics={**sem, "schema": 1})   # literal, not current-1
    with pytest.raises(ValueError, match=r"does not produce.*\['schema'\]"):
        partition.aggregate_partials(out)


def test_the_cross_sidecar_semantics_check_reports_disagreeing_payloads():
    """The CROSS-sidecar arm, called DIRECTLY -- and that is the honest way to test it now.

    ⚠️ It used to be driven through `aggregate_partials` with sidecars carrying `result_semantics`
    but no `metrics` key. Round 2 of codex-review pointed out that this state is exactly the hole
    the against-this-build check skipped, so the test was reaching its target THROUGH a defect. The
    state is now refused outright (`metrics` predates the semantics key, so no writer emits one
    without the other), which leaves this arm unreachable end to end: two self-consistent sidecars
    sharing a metric set and comparator have EQUAL payloads by construction.

    So it is kept as a direct unit test of a defence-in-depth branch, and labelled as one, rather
    than dressed up as an integration test of a path that cannot occur.
    """
    a = partition.result_semantics(["mae"], comparator="lognorm")
    b = {**a, "metric_agg": [["mae", "median"]]}
    with pytest.raises(ValueError, match="differ in RESULT SEMANTICS"):
        partition._check_result_semantics(
            "d", {"a": ("s0.json", a), "b": ("s1.json", b)}, [], n_sidecars=2)


def test_the_semantics_mismatch_message_names_the_terms_that_differ():
    """A bare digest would say only that two payloads disagree. Naming the term is what makes the
    error actionable -- an agg change and a normalization change call for completely different
    investigations. Asserted on both arms: cross-sidecar and against-this-build."""
    a = partition.result_semantics(["mae"], comparator="lognorm")
    b = {**a, "pds_exclusion_semantics": 99}
    with pytest.raises(ValueError, match=r"disagree on \['pds_exclusion_semantics'\]"):
        partition._check_result_semantics(
            "d", {"a": ("s0.json", a), "b": ("s1.json", b)}, [], n_sidecars=2)
    with pytest.raises(ValueError, match=r"does not produce.*\['pds_exclusion_semantics'\]"):
        partition._check_result_semantics(
            "d", {"b": ("s0.json", b)}, [], n_sidecars=1,
            declared=[("s0.json", ["mae"], "lognorm", b)])


def test_a_sidecar_with_semantics_but_no_metrics_is_REFUSED_not_skipped(tmp_path):
    """codex-review round 2's strongest point: the against-this-build loop used to `continue` when
    `metrics` was absent, so a stale sidecar -- or a whole directory of them -- was accepted with no
    check at all. `metrics` predates the semantics key, so the combination means a hand-edited or
    corrupt sidecar, not a legacy one."""
    out = str(tmp_path)
    _sem_partial(out, "s0", semantics=partition.result_semantics(["mae"], comparator="lognorm"),
                 metrics=None)
    with pytest.raises(ValueError, match="must also declare the metric list"):
        partition.aggregate_partials(out)


def test_aggregate_refuses_a_MIX_of_declared_and_undeclared_semantics(tmp_path):
    """Case 2 of the legacy policy, and it is the straddle #246 actually describes: an old
    partial predates the key, so pairing it with a new one cannot be verified. "No disagreement
    observed" would be a statement about the schema, not about the numbers."""
    out = str(tmp_path)
    _sem_partial(out, "s0", semantics=partition.result_semantics(["mae"], comparator="lognorm"))
    _sem_partial(out, "s1", semantics=None)
    with pytest.raises(ValueError, match="MIX declared and undeclared result semantics"):
        partition.aggregate_partials(out)


def test_an_ALL_legacy_partial_directory_warns_and_still_aggregates(tmp_path, caplog):
    """Case 3: refusing an all-legacy directory would break every warm partial dir to protect
    against a mix that, by construction, is not present. Mirrors the lenient/strict split the
    `metrics` key already uses -- and says so out loud, so the missing guard is not silent."""
    import logging
    out = str(tmp_path)
    _sem_partial(out, "s0", semantics=None)
    _sem_partial(out, "s1", semantics=None, value=3.0)
    with caplog.at_level(logging.WARNING, logger="cell_eval2.partition"):
        full, agg = partition.aggregate_partials(out)
    assert set(full["perturbation"]) == {"s0", "s1"}
    assert agg.filter(pl.col("metric") == "mae")["mean"][0] == 2.0
    assert "result-semantics guard" in caplog.text


def test_matching_semantics_aggregate_normally(tmp_path):
    """The guard must not reject the ordinary case -- two partials from one run."""
    out = str(tmp_path)
    sem = partition.result_semantics(["mae"], comparator="lognorm")
    _sem_partial(out, "s0", semantics=sem)
    _sem_partial(out, "s1", semantics=sem, value=3.0)
    full, agg = partition.aggregate_partials(out)
    assert set(full["perturbation"]) == {"s0", "s1"}
    assert agg.filter(pl.col("metric") == "mae")["mean"][0] == 2.0


def test_mixed_metric_MEMBERSHIP_is_still_refused_by_the_selection_check(tmp_path):
    """#246 asks for tests of BOTH reduction hazards. Membership was already guarded (#263 closed
    on that evidence); this pins that the older check still fires, and independently of the new
    one -- both partials here declare the same semantics SHAPE for their own selection."""
    out = str(tmp_path)
    # Each sidecar's semantics match ITS OWN metric list, so both pass the against-this-build
    # check and the selection check is the one that fires. The first version of this test gave the
    # two-metric sidecar semantics for one metric -- codex-review's point that it "proves the hole"
    # rather than the selection guard.
    _sem_partial(out, "s0", metrics=("mae",),
                 semantics=partition.result_semantics(["mae"], comparator="lognorm"))
    _sem_partial(out, "s1", metrics=("mae", "pds_cosine"),
                 semantics=partition.result_semantics(["mae", "pds_cosine"],
                                                     comparator="lognorm"))
    with pytest.raises(ValueError, match="differ in metric selections"):
        partition.aggregate_partials(out)
