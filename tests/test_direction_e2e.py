import math

import polars as pl
import pytest

from cell_eval2.de import prepare_de
from cell_eval2.metrics.direction import (
    de_direction_coverage,
    de_direction_fidelity,
    de_direction_fidelity_raw,
    de_direction_fidelity_yield,
    de_direction_reach,
)

SIG, NS = 0.01, 0.90


def _tbl(rows):
    return pl.DataFrame(
        {
            "target": [r[0] for r in rows],
            "feature": [r[1] for r in rows],
            "log2_fold_change": [float(r[2]) for r in rows],
            "p_adj": [float(r[3]) for r in rows],
        },
        schema={"target": pl.String, "feature": pl.String,
                "log2_fold_change": pl.Float64, "p_adj": pl.Float64},
    )


def _resolution(rows, mode):
    """mode='self' (default) -> every target maps to its own label; None -> derive."""
    from cell_eval2.de import TargetResolution
    if mode is None:
        return None
    if mode != "self":
        return mode
    targets = sorted({r[0] for r in rows})
    return TargetResolution({t: t for t in targets}, len(targets))


def _prep(pred, real, *, resolution="self"):
    """Default to a SELF-MAP resolution -- see the plan's fixture-trap note. Tests that
    need REAL derivation (the construct-ID and single-target cases) pass resolution=None."""
    return prepare_de(_tbl(pred), _tbl(real), control="non-targeting",
                      p_adj_threshold=0.05,
                      target_resolution=_resolution(real, resolution))


def _reference(n_genes=20, n_up=15):
    """One target 'T' over n_genes features, all reference-significant, n_up of them UP.
    q = max(n_up, n_genes-n_up)/n_genes."""
    return [("T", f"g{i}", 1.0 if i < n_up else -1.0, SIG) for i in range(n_genes)]


def test_identity_fidelity_equals_corrected_raw_on_its_domain():
    """Spec 8: domain N_conf > 0 AND n_pred > 0 AND q defined. Asserting it
    unconditionally over a degenerate target FAILS -- spec 5's conventions override the
    algebra, they are not consequences of it."""
    real = _reference()
    pred = [("T", f"g{i}", 1.0, SIG) for i in range(10)]
    p = _prep(pred, real)
    from cell_eval2.metrics.direction import _components
    comp = _components(p).row(0, named=True)
    assert comp["n_conf"] > 0 and comp["n_pred"] > 0 and comp["q"] is not None
    expected = (de_direction_fidelity_raw(p)["T"] - comp["q"]) / comp["d"]
    assert de_direction_fidelity(p)["T"] == pytest.approx(expected)


def test_identity_fidelity_yield_equals_capped_coverage_times_fidelity():
    real = _reference()
    pred = [("T", f"g{i}", 1.0, SIG) for i in range(10)]
    p = _prep(pred, real)
    expected = min(1.0, de_direction_coverage(p)["T"]) * de_direction_fidelity(p)["T"]
    assert de_direction_fidelity_yield(p)["T"] == pytest.approx(expected)


def test_attack_strategic_abstention_buys_nothing_on_the_scored_metrics():
    """Spec 5/8: n_pred = 0 makes fidelity NaN because the MODEL made no calls -- a
    model-side NaN, not a truth-side one. It is not a score-evasion hole because the
    SCORED metrics still charge for it."""
    real = _reference()
    honest = [("T", f"g{i}", 1.0, SIG) for i in range(10)]
    abstain = [("T", f"g{i}", 1.0, NS) for i in range(10)]
    p_honest, p_abstain = _prep(honest, real), _prep(abstain, real)
    assert math.isnan(de_direction_fidelity(p_abstain)["T"])
    assert de_direction_fidelity_yield(p_abstain)["T"] == pytest.approx(0.0)
    assert de_direction_fidelity_yield(p_honest)["T"] > 0.0
    # reach never reads n_pred, so abstention cannot help there either
    assert de_direction_reach(p_abstain)["T"] == pytest.approx(
        de_direction_reach(p_honest)["T"])


def test_attack_constant_sign_scores_near_zero_after_correction():
    """A model that predicts the majority sign everywhere earns ~q raw, which the
    correction maps to ~0."""
    real = _reference(n_genes=20, n_up=15)
    pred = [("T", f"g{i}", 1.0, SIG) for i in range(20)]   # everything UP
    p = _prep(pred, real)
    assert de_direction_fidelity_raw(p)["T"] == pytest.approx(0.75)
    assert de_direction_fidelity(p)["T"] == pytest.approx(0.0, abs=1e-9)


def test_attack_few_gene_calling_is_penalised_by_the_coverage_cap():
    """Calling one gene correctly gives perfect fidelity but low coverage, so the
    SCORED fidelity_yield stays small."""
    real = _reference()
    p_few = _prep([("T", "g0", 1.0, SIG)], real)
    p_many = _prep([("T", f"g{i}", 1.0, SIG) for i in range(15)], real)
    assert de_direction_fidelity(p_few)["T"] > 0
    assert de_direction_fidelity_yield(p_few)["T"] < de_direction_fidelity_yield(p_many)["T"]


def _reference_with_slack(n_conf=20, n_up=15, n_extra=20):
    """As _reference, plus `n_extra` shared genes the reference does NOT call significant.

    ⚠️ Needed for the padding and spray attacks: with a universe of exactly N_conf genes
    the model cannot call more than the reference's budget, so coverage tops out at 1 and
    the test cannot tell a capped implementation from an uncapped one. The slack genes
    carry a real direction (so they are adjudicable and count in n_pred and k) but are
    reference-NON-significant (so they are outside N_conf and q).
    """
    conf = [("T", f"g{i}", 1.0 if i < n_up else -1.0, SIG) for i in range(n_conf)]
    slack = [("T", f"x{i}", -1.0, NS) for i in range(n_extra)]
    return conf + slack


def test_attack_padding_cannot_beat_honest_calling():
    """Spec 8 attack 2. Padding with genes outside the reference's budget drives coverage
    ABOVE 1, and the cap must stop that buying anything.

    Honest: the 15 real-UP confident genes -> n_pred=15, k=15, raw=1, fidelity=1,
    coverage=15/20=0.75, fy=0.75.
    Padded: those plus all 20 slack genes, every one called UP while the slack genes are
    really DOWN -> n_pred=35, k=15, raw=15/35, coverage=35/20=1.75 (>1, so the cap binds).
    """
    real = _reference_with_slack()
    honest = [("T", f"g{i}", 1.0, SIG) for i in range(15)]
    padded = honest + [("T", f"x{i}", 1.0, SIG) for i in range(20)]
    p_h, p_p = _prep(honest, real), _prep(padded, real)
    assert de_direction_coverage(p_h)["T"] == pytest.approx(0.75)
    assert de_direction_coverage(p_p)["T"] == pytest.approx(1.75)   # > 1: the cap binds
    assert de_direction_fidelity_yield(p_h)["T"] == pytest.approx(0.75)

    # n_pred = 35, k = 15 -> raw = 3/7, fidelity = (3/7 - 3/4)/(1/4) = -9/7.
    fid_p = de_direction_fidelity(p_p)["T"]
    fy_p = de_direction_fidelity_yield(p_p)["T"]
    assert fid_p == pytest.approx(-9 / 7)
    # ⚠️ THE discriminating assertion. Once coverage exceeds 1 the cap makes
    # fidelity_yield EQUAL fidelity. An uncapped implementation would give
    # 1.75 * (-9/7) = -9/4 -- which is also negative and also worse than the honest
    # model, so every `< 0` / `< honest` comparison passes against the bug.
    assert fy_p == pytest.approx(-9 / 7)
    assert fy_p == pytest.approx(fid_p)
    assert fy_p != pytest.approx(-9 / 4)      # the uncapped value


def test_attack_on_target_only_scores_nothing():
    """Spec 8 attack 3. A model that calls ONLY the knocked-down gene -- and gets it
    right -- must earn nothing, because the target gene is excluded from the scored set,
    N_conf and both reach pools.

    The prediction contains ONLY the target row. ⚠️ It must not merely mark the off-target
    genes non-significant: `_purity_curve` ranks every row of the excluded frame regardless
    of PREDICTION significance (spec 2.6), so leaving directionally-correct g0/g1 rows in
    the pred table gives purity 1.0, k* = 2 and reach = 1.0 -- the opposite of what this
    attack should score. Omitting them empties the pool, which is the honest depth for a
    model that ranked nothing off-target.

    Uses REAL derivation so the on-target row is genuinely resolved and removed.
    """
    real = [("T", "T", 1.0, SIG), ("T", "g0", 1.0, SIG), ("T", "g1", 1.0, SIG)]
    pred = [("T", "T", 1.0, SIG)]
    p = _prep(pred, real, resolution=None)
    assert p.target_resolution.mapping["T"] == "T"
    from cell_eval2.metrics.direction import _components
    comp = _components(p).row(0, named=True)
    assert (comp["n_pred"], comp["n_conf"]) == (0, 2)   # T excluded from BOTH
    assert math.isnan(de_direction_fidelity_raw(p)["T"])       # 0/0
    assert de_direction_fidelity_yield(p)["T"] == pytest.approx(0.0)  # charged, not NaN
    assert de_direction_reach(p)["T"] == pytest.approx(0.0)   # empty pool -> k* = 0


def test_attack_spray_plus_generic_response_scores_below_chance():
    """Spec 8 attack 4. Spraying the WHOLE shared universe with the reference's majority
    sign scores BELOW chance, not at it: the slack genes it also calls are adjudicable and
    wrong, so they enter n_pred and drag raw fidelity under q.

    Distinct from the constant-sign attack, which calls only the confident universe and
    lands exactly at 0. This one also drives coverage to 2.0, so it probes the cap.
    """
    real = _reference_with_slack()
    spray = ([("T", f"g{i}", 1.0, SIG) for i in range(20)]
             + [("T", f"x{i}", 1.0, SIG) for i in range(20)])
    p = _prep(spray, real)
    assert de_direction_coverage(p)["T"] == pytest.approx(2.0)   # 40/20, cap binds
    # n_pred = 40 (every call is adjudicable), k = 15 (the real-UP confident genes).
    assert de_direction_fidelity_raw(p)["T"] == pytest.approx(15 / 40)
    fid = de_direction_fidelity(p)["T"]
    fy = de_direction_fidelity_yield(p)["T"]
    assert fid == pytest.approx(-3 / 2)     # (0.375 - 0.75)/0.25
    # The cap again: fy must EQUAL fid, not 2.0 * fid = -3.
    assert fy == pytest.approx(-3 / 2)
    assert fy == pytest.approx(fid)
    assert fy != pytest.approx(-3.0)        # the uncapped value


def test_q_is_invariant_when_prediction_rows_are_REMOVED():
    """Spec 8: the inner join makes row-removal the failure mode that matters -- q and
    N_conf come from real_df alone and must not move."""
    from cell_eval2.metrics.direction import _reference_stats
    real = _reference()
    full = [("T", f"g{i}", 1.0, SIG) for i in range(20)]
    trimmed = full[:5]
    a = _reference_stats(_prep(full, real)).row(0, named=True)
    b = _reference_stats(_prep(trimmed, real)).row(0, named=True)
    assert (a["q"], a["n_conf"]) == (b["q"], b["n_conf"])


def test_target_gene_map_changes_a_number():
    """Spec 8, and the regression test for spec 2.7c. The fixture must be MIXED: an
    all-construct-ID fixture resolves zero targets with no map and raises at the gate, so
    there would be no unmapped number to compare against."""
    rows = [
        ("A", "A", 1.0, SIG), ("A", "B", 1.0, SIG),          # A resolves exactly
        ("GENEX-1", "GENEX", 1.0, SIG), ("GENEX-1", "B", -1.0, SIG),
    ]
    pred = [
        ("A", "A", 1.0, SIG), ("A", "B", 1.0, SIG),
        ("GENEX-1", "GENEX", 1.0, SIG), ("GENEX-1", "B", 1.0, SIG),
    ]
    from cell_eval2.de import TargetResolution
    # resolution=None -> REAL derivation: 'A' resolves exactly, 'GENEX-1' does not, so
    # n_resolved = 1 and the gate does not fire. That mix is the point (spec 8): an
    # all-construct-ID fixture would raise in this arm and there would be no unmapped
    # number to compare against.
    unmapped = de_direction_fidelity_raw(_prep(pred, rows, resolution=None))["GENEX-1"]
    mapped = de_direction_fidelity_raw(
        _prep(pred, rows,
              resolution=TargetResolution({"A": "A", "GENEX-1": "GENEX"}, 2))
    )["GENEX-1"]
    assert unmapped == pytest.approx(0.5)   # GENEX matches, B does not
    assert mapped == pytest.approx(0.0)     # GENEX excluded, only the mismatch remains
    assert unmapped != pytest.approx(mapped)


def test_h1_cgs_shape_neither_raises_nor_excludes():
    """Spec 8: every target resolves against the GLOBAL index, but no target's own gene
    is among its OWN rows. The per-target-check regression test.

    resolution=None so this exercises REAL derivation -- with the self-map default it
    would assert nothing about the resolver.
    """
    rows = [("A", "B", 1.0, SIG), ("A", "C", 1.0, SIG),
            ("B", "A", 1.0, SIG), ("B", "C", 1.0, SIG),
            ("C", "A", 1.0, SIG), ("C", "B", 1.0, SIG)]
    p = _prep(rows, rows, resolution=None)
    assert p.target_resolution.n_resolved == 3
    from cell_eval2.metrics.direction import _direction_frame, _ontarget_excluded_frame
    assert _ontarget_excluded_frame(p).height == _direction_frame(p).height


def test_construct_id_fixture_raises_and_passes_with_a_map():
    from cell_eval2.de import TargetResolution
    rows = [("GENEX-1", "GENEX", 1.0, SIG), ("GENEY-2", "GENEY", 1.0, SIG)]
    # Construction succeeds; the METRIC raises. The gate is at the eleven metrics' entry,
    # not at PreparedDE construction -- so the `raises` block must wrap a metric call.
    bare = _prep(rows, rows, resolution=None)
    assert bare.target_resolution.n_resolved == 0
    with pytest.raises(ValueError, match="no target resolves"):
        de_direction_fidelity_raw(bare)
    p = _prep(rows, rows, resolution=TargetResolution(
        {"GENEX-1": "GENEX", "GENEY-2": "GENEY"}, 2))
    assert set(de_direction_fidelity_raw(p)) == {"GENEX-1", "GENEY-2"}


def test_single_target_dataset_raises_and_the_map_rescues_it():
    """Spec 10 open 4 + spec 2.1's map-bypasses-index clause. The mapped gene is
    deliberately ABSENT from the index and must be accepted rather than re-checked."""
    from cell_eval2.de import TargetResolution
    rows = [("T", "g0", 1.0, SIG), ("T", "g1", 1.0, SIG)]
    with pytest.raises(ValueError, match="no target resolves"):
        de_direction_fidelity_raw(_prep(rows, rows, resolution=None))
    # The mapped gene 'T' is deliberately absent from the index and must be accepted.
    p = _prep(rows, rows, resolution=TargetResolution({"T": "T"}, 1))
    assert not math.isnan(de_direction_fidelity_raw(p)["T"])


def test_an_empty_resolution_does_not_bypass_the_gate():
    """The bypass a construction-time gate would have left open: PreparedDE's field
    DEFAULTS to TargetResolution({}, 0), so a check keyed on n_targets would pass it.
    The gate keys on prepared.perturbations instead."""
    from cell_eval2.de import TargetResolution
    rows = [("A", "A", 1.0, SIG), ("A", "B", 1.0, SIG)]
    p = _prep(rows, rows, resolution=TargetResolution({}, 0))   # n_targets == 0
    assert p.perturbations == ["A"]
    with pytest.raises(ValueError, match="no target resolves"):
        de_direction_fidelity_raw(p)


def test_memos_are_isolated_per_resolution():
    """Spec 8: two PreparedDEs with different target_resolution never share memo values.
    Assert on all three memos individually -- a shared-memo bug shows up in only one of
    them at a time."""
    from cell_eval2.de import TargetResolution
    from cell_eval2.metrics.direction import (
        _components, _ontarget_excluded_frame, _reference_stats)
    rows = [("A", "A", 1.0, SIG), ("A", "B", 1.0, SIG)]
    with_excl = _prep(rows, rows, resolution=TargetResolution({"A": "A"}, 1))
    without = _prep(rows, rows, resolution=TargetResolution({}, 1))
    assert _ontarget_excluded_frame(with_excl).height != \
        _ontarget_excluded_frame(without).height
    assert _reference_stats(with_excl)["n_conf"].to_list() != \
        _reference_stats(without)["n_conf"].to_list()
    assert _components(with_excl)["n_pred"].to_list() != \
        _components(without)["n_pred"].to_list()


def test_the_eleven_are_absent_from_the_published_wide_csv_under_v1():
    """Spec 8: metric_output_names + aggregate_metrics_wide are the path a
    dispatch-only gate misses -- test the final wide CSV, not only the tidy rows.

    Compare against the EXACT eleven-name set. A substring check on
    'direction_fidelity'/'direction_reach' silently misses direction_coverage,
    direction_yield and direction_yield_raw.
    """
    from cell_eval2.catalog import CATALOG
    from cell_eval2.config import EvalConfig
    from cell_eval2.run import aggregate_metrics_wide, metric_output_names

    # NAMED, not derived from `not v1_available`. That derivation happened to equal the
    # eleven only while they were the sole metrics with the flag cleared; once v1
    # availability became a property of v1_name it covered 57 entries, and the test would
    # have kept "passing" while no longer checking the eleven in particular.
    eleven = {f"de_{m}_{s}"
              for m in ("wilcoxon", "deseq2")
              for s in ("direction_fidelity", "direction_fidelity_raw", "direction_coverage",
                        "direction_yield", "direction_yield_raw", "direction_fidelity_yield",
                        "direction_fidelity_yield_raw", "direction_reach",
                        "direction_reach_raw", "direction_reach_unbounded",
                        "direction_reach_unbounded_raw")}
    assert len(eleven) == 22, "11 suffixes x {wilcoxon, deseq2}"
    assert eleven <= {n for n, s in CATALOG.items() if not s.v1_available}
    assert all(n in CATALOG for n in eleven)

    names = metric_output_names(EvalConfig(metrics="full", version="v1"))
    assert not (set(names) & eleven)
    wide = aggregate_metrics_wide(
        pl.DataFrame({"perturbation": [], "metric": [], "value": []},
                     schema={"perturbation": pl.String, "metric": pl.String,
                             "value": pl.Float64}),
        metrics=names,
    )
    assert not (set(wide.columns) & eleven)

    # ...and they ARE present under v2, so the assertion above is not vacuous.
    #
    # #212: this used to read `set(v2_names) & eleven` -- a non-empty INTERSECTION, which
    # ONE of the 22 being present satisfies. The sibling assertion under v1 above is correct
    # as written (there an EMPTY intersection IS the whole claim); only this direction was
    # weak.
    #
    # ⚠️ The issue prescribed `eleven <= set(v2_names)` and that is UNACHIEVABLE, measured:
    # `full` under v2 emits the eleven `de_wilcoxon_*` names and none of the eleven
    # `de_deseq2_*` mirrors, which carry `profiles=()` and are reached only by the
    # backend-driven relabel (see test_deseq2_family_absent_from_profiles). So the property
    # is asserted per family: ALL eleven of the emitted one, and NONE of the opt-in one --
    # which is strictly stronger than the intersection and, unlike the prescribed form,
    # true.
    v2_names = metric_output_names(EvalConfig(metrics="full", version="v2"))
    wilcoxon_eleven = {n for n in eleven if n.startswith("de_wilcoxon_")}
    deseq2_eleven = eleven - wilcoxon_eleven
    assert len(wilcoxon_eleven) == 11 and len(deseq2_eleven) == 11
    assert wilcoxon_eleven <= set(v2_names)
    assert not (set(v2_names) & deseq2_eleven)


def test_target_gene_map_flows_from_EvalConfig_through_a_construction_site():
    """Spec 8 wants the map exercised as CONFIG, not just as a hand-built
    TargetResolution: the wiring from EvalConfig into the four PreparedDE construction
    sites is the part that can silently not happen.

    Drive `_prepare_de_cached` -- the main compute_metrics path (run.py:466) -- with a
    construct-ID fixture that resolves ZERO targets without a map, and assert (a) it
    resolves nothing without one and (b) the map rescues it.
    """
    from dataclasses import replace
    from cell_eval2.config import EvalConfig
    from cell_eval2.run import _prepare_de_cached

    tbl = _tbl([("GENEX-1", "GENEX", 1.0, SIG), ("GENEY-2", "GENEY", 1.0, SIG)])
    base = EvalConfig()
    kw = dict(real_store=None, pred_store=None,
              de_real_supplied=True, de_pred_supplied=True)

    bare = _prepare_de_cached(tbl, tbl, cfg=base, **kw)
    assert bare.target_resolution.n_resolved == 0

    mapped_cfg = replace(base, target_gene_map={"GENEX-1": "GENEX", "GENEY-2": "GENEY"})
    mapped = _prepare_de_cached(tbl, tbl, cfg=mapped_cfg, **kw)
    assert mapped.target_resolution.mapping["GENEX-1"] == "GENEX"
    # ...and the metrics now run rather than raising at the gate.
    assert set(de_direction_fidelity_raw(mapped)) == {"GENEX-1", "GENEY-2"}
    with pytest.raises(ValueError, match="no target resolves"):
        de_direction_fidelity_raw(bare)


def test_cli_writes_the_metric_aggregation_sidecar(synthetic_pair, tmp_path):
    """Spec 4's sidecar can be omitted or malformed while every other test passes,
    because nothing else in this suite reads it. Drive the CLI `run` subcommand end to end.

    Follows tests/test_cli.py::test_cli_run_writes_results, with two changes that matter:
      * profile `de`, not `vcc` -- the vcc profile deliberately excludes the eleven
        (spec 6), so a vcc run would exercise none of the direction family;
      * the DE tables' feature set INCLUDES the target labels, so resolution succeeds.
        test_cli.py's own tables use targets GENE1..3 against features g0..g3, which
        resolves zero targets and would raise at the gate.
    """
    from cell_eval2.catalog import CATALOG
    from cell_eval2.cli import main

    pred, real = synthetic_pair
    pp, rp = tmp_path / "pred.h5ad", tmp_path / "real.h5ad"
    pred.write_h5ad(pp)
    real.write_h5ad(rp)

    # Targets are also features -> every target resolves.
    de_rows = {
        "target": ["GENE1", "GENE1", "GENE2", "GENE2", "GENE3", "GENE3"],
        "feature": ["GENE1", "g1", "GENE2", "g2", "GENE3", "g3"],
        "log2_fold_change": [2.0, 1.0, 1.5, -1.0, 1.0, 1.0],
        "p_adj": [0.01, 0.01, 0.01, 0.01, 0.01, 0.01],
    }
    de_real_p, de_pred_p = tmp_path / "real_de.parquet", tmp_path / "pred_de.parquet"
    pl.DataFrame(de_rows).write_parquet(de_real_p)
    pl.DataFrame({**de_rows, "log2_fold_change": [1.8, 0.9, 1.2, 1.0, 0.9, -1.0]}
                 ).write_parquet(de_pred_p)

    out = tmp_path / "out"
    main(["run", "-ap", str(pp), "-ar", str(rp), "--profile", "de",
          "--pert-col", "target", "--control", "non-targeting",
          "--input-type", "lognorm",
          "--de-pred", str(de_pred_p), "--de-real", str(de_real_p), "-o", str(out)])

    sidecar = out / "metric_aggregation.csv"
    assert sidecar.exists()
    side = pl.read_csv(sidecar)
    # #239 appended the cohort columns. `metric` and `agg` stay FIRST and keep their meaning --
    # the two internal readers take what they need by name (internal:tools/metricval/strata.py
    # checks `{"metric","agg"} <= columns`; internal:tools/metricval/report.py zips the two by
    # name), so appending is compatible with
    # both. Asserted as a prefix rather than an exact list, so a future column does not fail here
    # while a writer that dropped or reordered the original two still does.
    assert side.columns[:2] == ["metric", "agg"]
    assert {"n_used", "n_rows", "n_nan", "n_null", "derived"} <= set(side.columns)

    wide = pl.read_csv(out / "agg_results.csv")
    metric_cols = [c for c in wide.columns if c != "statistic"]
    assert set(side["metric"].to_list()) == set(metric_cols)

    # Since #231 the catalog has ONE statistic, so the sidecar's whole `agg` column is
    # "mean". Pinned as a set equality rather than dropped: the sidecar exists precisely so a
    # consumer of agg_results.csv alone can tell which statistic each column holds, and a
    # writer that emitted the wrong constant -- or that stopped writing the column -- must
    # still fail here. (The per-row cross-check against CATALOG below is the same fact keyed
    # per metric; this line is the one that reads as a statement about the RUN's output.)
    assert set(side["agg"].unique().to_list()) == {"mean"}
    for name, agg in zip(side["metric"].to_list(), side["agg"].to_list()):
        spec = CATALOG.get(name)
        assert agg == (spec.agg if spec is not None else "mean"), name

    # ...and the wide CSV is still strictly numeric after the round trip.
    for c in metric_cols:
        assert wide[c].dtype in (pl.Float64, pl.Int64), c


@pytest.mark.parametrize("driver_name", ["_score_streaming_de", "_score_streaming_cell_de"])
def test_scale_drivers_resolve_targets_before_slicing(driver_name):
    """Dataset-level resolution must be computed from the unsliced real DE table."""
    import inspect

    from cell_eval2 import scale

    src = inspect.getsource(getattr(scale, driver_name))
    resolve_at = src.index("resolve_target_genes(")
    filter_at = src.index('.filter(pl.col("target").is_in(')
    assert resolve_at < filter_at, f"{driver_name} resolves target genes after slicing"
