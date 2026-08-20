import json
from dataclasses import replace

import numpy as np
import polars as pl
import pytest

from cell_eval2 import EvalConfig
from cell_eval2.baseline import (
    BaselineResult,
    baseline_config,
    build_generic_baseline,
    config_digest,
)
from cell_eval2.score import score_metrics

_METRICS = ["expr_mae", "delta_pearson"]


def _cfg(**kw):
    # These legacy builder tests use a lognorm reference. Dispersed emission is counts-only
    # (§3.2a), and their subjects are config/stamp/gate plumbing, so each build selects tile.
    base = dict(metrics=_METRICS, pert_col="target", control="non-targeting",
                input_type="lognorm", validate_input=False)
    base.update(kw)
    return EvalConfig(**base)


def _de_features(features, lfc=3.0):
    """A supplied real-side DE table over conftest's GENE1..3 perturbations."""
    return pl.DataFrame([
        {"target": t, "feature": f, "log2_fold_change": lfc, "p_adj": 0.001}
        for t in ("GENE1", "GENE2", "GENE3") for f in features
    ])


# --------------------------------------------------------------- forced knobs

def test_forces_allow_fractional_counts():
    assert baseline_config(_cfg()).allow_fractional_counts is True


def test_does_NOT_touch_control_source():
    """Reverses an earlier draft. Task 3's real-control rows make both settings give the
    same numbers, so forcing would change the v1 estimand for no gain."""
    for source in ("pred", "real"):
        assert baseline_config(_cfg(control_source=source)).control_source == source


def test_rejects_the_deseq2_backend():
    cfg = _cfg()
    cfg = replace(cfg, de=replace(cfg.de, backend="deseq2"))
    with pytest.raises(ValueError, match="deseq2"):
        baseline_config(cfg)


def test_disables_cache_pred_and_warns(caplog, tmp_path):
    cfg = _cfg(cache_pred=str(tmp_path / "pcache"), cache_real=str(tmp_path / "rcache"))
    with caplog.at_level("WARNING"):
        eff = baseline_config(cfg)
    assert eff.cache_pred is None
    assert eff.cache_real == cfg.cache_real          # the real side IS reusable
    assert any("cache_pred" in r.message for r in caplog.records)


def test_cache_pred_guard_is_load_bearing(synthetic_pair, tmp_path):
    """Discriminating: with the guard BYPASSED the two exclude_target_gene arms collide
    in a warm pred cache and return identical (wrong) results; with the guard they
    differ. Proves the forcing does real work rather than being decorative.

    Non-strict fingerprint_adata hashes only shape, dtype, var index and per-cell
    labels -- never X -- and the baseline prediction mirrors the reference, so both
    arms fingerprint identically regardless of their profile values.
    """
    from cell_eval2.baseline import build_baseline_prediction, generic_response_profile
    from cell_eval2.run import compute_metrics

    _, real = synthetic_pair
    real = real.copy()
    # make perturbation labels gene names so the exclusion has something to bite on
    mapping = {"GENE1": "g0", "GENE2": "g1", "GENE3": "g2"}
    real.obs["target"] = [mapping.get(v, v) for v in real.obs["target"]]

    shared = str(tmp_path / "shared_pred_cache")
    bypassed = replace(_cfg(allow_fractional_counts=True),
                       cache_pred=shared, cache_real=str(tmp_path / "r"))
    got = []
    for flag in (True, False):
        prof = generic_response_profile(real, pert_col="target",
                                        control="non-targeting",
                                        exclude_target_gene=flag)
        pred = build_baseline_prediction(prof, real, pert_col="target",
                                         control="non-targeting", emit="tile")
        got.append(compute_metrics(pred, real, config=bypassed))
    # the profiles genuinely differ...
    p_on = generic_response_profile(real, pert_col="target", control="non-targeting",
                                    exclude_target_gene=True)
    p_off = generic_response_profile(real, pert_col="target", control="non-targeting",
                                     exclude_target_gene=False)
    assert not np.allclose(p_on.values, p_off.values)
    # ...yet the cached run reports them identical: the defect the guard prevents
    assert got[0].equals(got[1])

    # with the guard, cache_pred is dropped and the arms separate
    guarded = baseline_config(bypassed)
    assert guarded.cache_pred is None
    fresh = []
    for prof in (p_on, p_off):
        pred = build_baseline_prediction(prof, real, pert_col="target",
                                         control="non-targeting", emit="tile")
        fresh.append(compute_metrics(pred, real, config=guarded))
    assert not fresh[0].equals(fresh[1])


# --------------------------------------------------------------- digest

def test_digest_ignores_performance_only_knobs():
    a = _cfg(num_threads=1, gather_threads=1, outdir="/tmp/a", cache_strict=True)
    b = _cfg(num_threads=8, gather_threads=8, outdir="/tmp/b", cache_strict=False)
    assert config_digest(a, comparator="lognorm") == config_digest(b, comparator="lognorm")


def test_digest_uses_the_RESOLVED_device_not_the_spelling():
    """device is value-affecting (run.py:398-403 keys the pseudobulk cache on it, because
    fp32-GPU and fp64-CPU means differ), so it is digested -- but as its RESOLUTION, like
    run._result_config_digest does. On one host 'auto' and the concrete device it resolves
    to produce identical numbers and must not mismatch; two hosts that resolve differently
    still must."""
    from cell_eval2.run import _cache_device
    resolved = _cache_device(_cfg())                      # 'auto' -> 'cpu' or 'cuda' here
    assert config_digest(_cfg(), comparator="lognorm") == config_digest(
        _cfg(device=resolved), comparator="lognorm",
    )
    other = "cuda" if resolved == "cpu" else "cpu"
    assert config_digest(_cfg(), comparator="lognorm") != config_digest(
        _cfg(device=other), comparator="lognorm",
    )


def test_digest_CHANGES_with_pert_chunk():
    """Not machine-resolvable, so it is digested verbatim: it governs GPU reduction
    blocking and can change the numbers."""
    assert config_digest(_cfg(), comparator="lognorm") != config_digest(
        _cfg(pert_chunk=64), comparator="lognorm",
    )


def test_digest_CHANGES_with_control_source():
    """No longer exempt: nothing forces it, so it must match like any scoring knob."""
    assert config_digest(
        _cfg(control_source="pred"), comparator="lognorm",
    ) != config_digest(_cfg(control_source="real"), comparator="lognorm")


def test_digest_ignores_allow_fractional_counts():
    """A validation ALLOWANCE, not a scoring semantic -- it changes no metric's math. If
    it were digested, every baseline would mismatch every ordinary run by construction."""
    assert config_digest(_cfg(), comparator="lognorm") == config_digest(
        _cfg(allow_fractional_counts=True), comparator="lognorm",
    )


def test_digest_CHANGES_with_allow_discrete():
    """It is value-affecting, so it stays IN the digest. What makes forcing it safe is
    that build_generic_baseline digests the REQUESTED config -- see the next test."""
    assert config_digest(_cfg(), comparator="lognorm") != config_digest(
        _cfg(allow_discrete=True), comparator="lognorm",
    )


def test_digest_is_taken_over_the_REQUESTED_config(synthetic_counts_pair):
    """The forced knobs must not reach the digest, or every baseline mismatches every
    ordinary run by construction (codex #2). Exempting them instead would be unsound in the
    other direction, so the stamp digests what the caller ASKED FOR."""
    _, real = synthetic_counts_pair
    requested = _cfg(input_type="counts", version="v1", validate_input=True)
    res = build_generic_baseline(real, config=requested, exclude_target_gene=False)
    assert res.meta["allow_discrete_effective"] is True        # it really was forced
    assert requested.allow_discrete is False                   # ...and requested was not
    assert res.meta["config_digest"] == config_digest(requested, comparator="lognorm")
    # discriminating: digesting the EFFECTIVE config would have given a different answer
    effective = EvalConfig.from_dict(res.meta["config"])
    assert config_digest(effective, comparator="lognorm") != res.meta["config_digest"]
    assert res.meta["config_requested"] == requested.to_dict()


def test_digest_changes_with_a_scoring_knob():
    """A DE metric is selected, so de.p_adj_threshold genuinely changes the numbers.

    Note the digest is CONSERVATIVE: it covers the whole `de` block even for a run that
    selects no DE metric, so two such configs differing only in a DE knob will mismatch
    although they would have produced identical numbers. That direction is loud and
    --allow-config-mismatch overrides it; the opposite direction (silently accepting an
    incomparable pair) is what a correctness gate must never do.
    """
    a = _cfg(metrics=["expr_mae", "de_wilcoxon_overlap"])
    b = replace(a, de=replace(a.de, p_adj_threshold=0.01))
    assert config_digest(a, comparator="lognorm") != config_digest(b, comparator="lognorm")


def test_digest_normalizes_profile_name_to_resolved_metrics():
    """'vcc' and the explicit list it resolves to must agree, or the mismatch check
    fires on two runs that are actually identical."""
    from cell_eval2.catalog import resolve_metrics
    names, _ = resolve_metrics("vcc")
    assert config_digest(_cfg(metrics="vcc"), comparator="lognorm") == config_digest(
        _cfg(metrics=names), comparator="lognorm",
    )


# ------------------------------------------------- the agg mapping in the digest (#231)


def _patched_agg(monkeypatch, name, agg):
    """Give ONE catalog entry a different aggregation statistic, in place."""
    from cell_eval2.catalog import CATALOG
    monkeypatch.setitem(CATALOG, name, replace(CATALOG[name], agg=agg))


def test_digest_CHANGES_when_a_SELECTED_metric_changes_its_agg(monkeypatch):
    """#231's whole point. The 18 entries that moved from median to mean kept their names
    and their profile membership, so the resolved NAME list -- everything the digest recorded
    before -- is bit-identical across the change while every whole-cohort number moves. A
    0.7 baseline scored against a 0.8 run would have produced silently wrong margins.

    Not covered by the version stamp: `cell_eval2_version` resolves through the INSTALLED
    distribution metadata, which in a dev tree need not describe the tree under test.
    """
    cfg = _cfg(metrics=["expr_mae", "delta_pearson"])
    before = config_digest(cfg, comparator="lognorm")
    _patched_agg(monkeypatch, "expr_mae", "median")
    assert config_digest(cfg, comparator="lognorm") != before


def test_digest_is_INERT_to_an_UNSELECTED_metric_changing_its_agg(monkeypatch):
    """Scoped to the resolved list, like `metrics` itself. Digesting the whole catalog would
    make every baseline mismatch every run after any unrelated enrolment."""
    cfg = _cfg(metrics=["expr_mae", "delta_pearson"])
    before = config_digest(cfg, comparator="lognorm")
    _patched_agg(monkeypatch, "de_wilcoxon_direction_reach", "median")   # not selected
    assert config_digest(cfg, comparator="lognorm") == before


def test_digest_is_stable_across_calls_for_one_config():
    """The mapping is emitted as an ORDERED list of pairs, so no mapping-iteration order can
    leak into the payload and make an unchanged config mismatch itself."""
    cfg = _cfg(metrics=["expr_mae", "delta_pearson"])
    assert len({config_digest(cfg, comparator="lognorm") for _ in range(5)}) == 1


def test_the_agg_mapping_does_NOT_reach_the_RESULT_cache_key(monkeypatch):
    """The deliberate asymmetry (#231). The result cache stores the PER-PERTURBATION tidy
    frame; `agg` is applied to it only afterwards, so an agg-only change leaves every cached
    value correct and invalidating it would buy nothing but a recompute. Two independent
    mechanisms already keep it out -- `cache.config_hash` strips `metrics` outright, and the
    resolved names reach the key through `result_fingerprint`, not the config digest.

    Asserted on the FINGERPRINT, which is what `compute_metrics` looks the cache up by, and
    paired with the discriminating half: changing the selected metric LIST still misses.
    """
    from cell_eval2.cache import result_fingerprint
    from cell_eval2.catalog import resolve_metrics
    from cell_eval2.run import _result_config_digest

    cfg = EvalConfig(metrics=["expr_mae", "delta_pearson"], device="cpu")
    names, _ = resolve_metrics(cfg.metrics, version=cfg.version)

    def fp(config, metric_names):
        return result_fingerprint(
            real_fp="real", pred_fp="pred", de_fps=["no-de-real", "no-de-pred"],
            config_digest=_result_config_digest(
                config, de_backend_used=False, comparator="lognorm",
            ),
            metric_names=metric_names)

    before = fp(cfg, names)
    _patched_agg(monkeypatch, "expr_mae", "median")
    assert fp(cfg, names) == before, "an agg-only change must not invalidate the result cache"

    other = EvalConfig(metrics=["expr_mae"], device="cpu")
    other_names, _ = resolve_metrics(other.metrics, version=other.version)
    assert fp(other, other_names) != before, "a changed metric LIST must still miss"


# --------------------------------------------------------------- orchestrator

def test_build_returns_results_agg_profile_and_stamp(synthetic_pair):
    _, real = synthetic_pair
    res = build_generic_baseline(real, config=_cfg(), emit="tile")
    assert isinstance(res, BaselineResult)
    assert res.results.columns == ["perturbation", "metric", "value"]
    assert res.agg.columns[0] == "statistic"
    assert set(res.agg.columns[1:]) == set(_METRICS)
    assert res.profile.n_perturbations == 3

    for field in ("cell_eval2_version", "created_utc", "source", "source_fingerprint",
                  "pert_col", "control", "n_perturbations", "n_genes",
                  "exclude_target_gene", "n_excluded", "control_source_requested",
                  "control_source_effective", "input_type_real_effective",
                  "input_type_pred_effective", "comparator", "allow_discrete_effective",
                  "resolved_device", "resolved_de_backend", "de_real_supplied",
                  "degenerate_metrics", "metrics", "config_requested", "config",
                  "config_digest"):
        assert field in res.meta, field
    assert res.meta["n_excluded"] == res.profile.n_excluded
    assert res.meta["degenerate_metrics"] == []
    assert res.meta["de_real_supplied"] is False
    # STRICT JSON: json.dump emits a bare `NaN` token by default, which is not valid JSON.
    # allow_nan=False is what the CLI writes with, so assert it here too.
    json.dumps(res.meta, allow_nan=False)


def test_build_records_the_exclusion_that_happened(synthetic_pair):
    """conftest's var index now MEASURES GENE1..3 (see conftest._var_names, #248), so the
    exclusion genuinely fires and the stamp must report the count it actually removed."""
    _, real = synthetic_pair
    res = build_generic_baseline(real, config=_cfg(), exclude_target_gene=True, emit="tile")
    assert res.meta["exclude_target_gene"] is True
    assert res.meta["n_excluded"] == 3


def _unresolvable_panel():
    """A panel whose labels name no measured gene -- the construct-ID shape, minus the map.

    Built locally rather than from `synthetic_pair`, whose var index MEASURES GENE1..3
    (conftest._var_names, #248) and therefore resolves.
    """
    import anndata as ad
    import pandas as pd
    rng = np.random.default_rng(0)
    labels = [p for p in ("non-targeting", "GENE1", "GENE2", "GENE3") for _ in range(25)]
    X = np.log1p(rng.gamma(shape=1.0, scale=1.0, size=(len(labels), 12)))
    return ad.AnnData(
        X=X,
        obs=pd.DataFrame({"target": labels}, index=[f"c{i}" for i in range(len(labels))]),
        var=pd.DataFrame(index=[f"g{j}" for j in range(12)]),  # no GENE* -> nothing matches
    )


def test_build_REFUSES_a_panel_where_the_exclusion_would_be_a_no_op():
    """#285. This test previously asserted the opposite -- a warning, a plain-mean baseline,
    and `n_excluded == 0` stamped as the after-the-fact proof -- and its own docstring
    recorded the reason ("this builder still ignores EvalConfig.target_gene_map ... tracked
    separately"). That is now #253/#285, fixed: the builder routes through
    `distances.resolve_exclusion_columns` and inherits its zero-resolve raise.

    The stamp was a weaker guarantee than it looked. A generic-response baseline is the 0 end
    of the competition scale, so "excluded nothing, and it is visible in the metadata if you
    read it" is not the same as refusing to build."""
    with pytest.raises(ValueError, match="NO perturbation resolves"):
        build_generic_baseline(_unresolvable_panel(), config=_cfg(),
                               exclude_target_gene=True, emit="tile")


def test_build_on_that_same_panel_is_fine_with_the_exclusion_OFF():
    """The refusal is scoped to the request. Asking for no exclusion asks nothing of the
    labels, so the panel that cannot resolve is still perfectly scoreable."""
    res = build_generic_baseline(_unresolvable_panel(), config=_cfg(),
                                 exclude_target_gene=False, emit="tile")
    assert res.meta["exclude_target_gene"] is False
    assert res.meta["n_excluded"] == 0


def test_build_THREADS_the_config_target_gene_map():
    """#253's live half, end to end through the builder rather than through the profile
    helper. A genuine construct-ID panel: labels `GENE1-1`, genes `GENE1`.

    ⚠️ Both arms are asserted. Without the map the panel is REFUSED, with it `n_excluded`
    is 3 -- so the map is demonstrably what did the work. Asserting only the passing arm
    would also pass on a panel the raw-label fallback resolves by itself, which is the shape
    this test would silently degrade into.

    The map was ALREADY in this builder's `config_digest` before the fix (dropped only when
    None), so two baselines differing only in the map digested differently and came out
    numerically identical. This is the assertion that the digest describes the artifact."""
    real = _unresolvable_panel()
    real.obs["target"] = [lab if lab == "non-targeting" else f"{lab}-1"
                          for lab in real.obs["target"]]
    real.var.index = ["GENE1", "GENE2", "GENE3"] + [f"g{j}" for j in range(9)]

    with pytest.raises(ValueError, match="NO perturbation resolves"):
        build_generic_baseline(real, config=_cfg(), exclude_target_gene=True, emit="tile")

    cfg = _cfg(target_gene_map={"GENE1-1": "GENE1", "GENE2-1": "GENE2",
                                "GENE3-1": "GENE3"})
    res = build_generic_baseline(real, config=cfg, exclude_target_gene=True, emit="tile")
    assert res.meta["n_excluded"] == 3


def test_build_saves_pred_when_asked(synthetic_pair, tmp_path):
    import anndata as ad
    _, real = synthetic_pair
    out = tmp_path / "baseline_pred.h5ad"
    res = build_generic_baseline(real, config=_cfg(), save_pred=str(out), emit="tile")
    assert out.exists()
    saved = ad.read_h5ad(out)
    assert saved.shape == real.shape
    non_ctrl = np.asarray(real.obs["target"]) != "non-targeting"
    saved_non_ctrl = saved.X[non_ctrl]
    mean = (np.asarray(saved_non_ctrl.mean(axis=0)).ravel()
            if hasattr(saved_non_ctrl, "toarray") else np.asarray(saved_non_ctrl).mean(axis=0))
    # Grouping accumulates the saved float32 rows; the measured reduction error is 6.6e-7.
    np.testing.assert_allclose(mean, res.profile.values.astype(np.float32), rtol=2e-6)


def test_build_scored_against_itself_is_zero(synthetic_pair):
    from cell_eval2 import score_metrics
    _, real = synthetic_pair
    res = build_generic_baseline(real, config=_cfg(), emit="tile")
    mae = dict(zip(res.agg["statistic"], res.agg["expr_mae"]))["mean"]
    assert mae > 0, "score_metrics fails loud on a base <= 0 for a lower-is-better metric"
    scored = score_metrics(res.agg, res.agg)
    assert scored.filter(pl.col("metric") == "avg_score")["from_baseline"][0] == pytest.approx(0.0)


def test_de_real_is_CONSUMED_not_ignored(synthetic_pair):
    """--de-real was parsed and silently dropped. Two different supplied real-side DE
    tables must produce different numbers; identical output would mean the argument never
    reached compute_metrics."""
    _, real = synthetic_pair
    cfg = _cfg(metrics=["de_wilcoxon_overlap"])
    # Both feature sets must land NON-degenerate values, or "different" could just be
    # 0.0 vs 0.0 and the test would pass vacuously. Since #248 made conftest's panel
    # measure GENE1..3, the baseline's own DE ranking shifted and the previous
    # g0..g5 sets both overlap zero -- so pick sets that demonstrably do not.
    # Measured here: the 3 target genes -> 0.333, the whole 40-gene panel -> 0.025.
    all_features = list(real.var.index)
    a = build_generic_baseline(real, config=cfg, de_real=_de_features(all_features[:3]),
                               emit="tile")
    b = build_generic_baseline(real, config=cfg, de_real=_de_features(all_features),
                               emit="tile")
    assert a.meta["de_real_supplied"] is True
    a_mean = dict(zip(a.agg["statistic"], a.agg["de_wilcoxon_overlap"]))["mean"]
    b_mean = dict(zip(b.agg["statistic"], b.agg["de_wilcoxon_overlap"]))["mean"]
    assert a_mean > 0 and b_mean > 0, "both arms must be non-degenerate"
    assert not a.agg.equals(b.agg)


def test_backend_identity_keys_on_the_REAL_side_only(synthetic_pair):
    """A `run --de-pred P --de-real R` computes NEITHER side, while the baseline's synthetic
    prediction ALWAYS computes its own DE. Keying the compared backend on "either side
    computed" would record pdex vs null and reject that supported pairing -- for a
    difference on the PREDICTION side, which is exactly what de_pred_fingerprint is recorded
    but not compared for (design section 6)."""
    from cell_eval2.baseline import _de_backend_used
    from cell_eval2.catalog import resolve_metrics

    cfg = _cfg(metrics=["de_wilcoxon_overlap"])
    names = resolve_metrics(cfg.metrics)[0]
    assert _de_backend_used(cfg, names, None) is True            # real computed
    assert _de_backend_used(cfg, names, _de_features(["g0"])) is False   # real supplied
    # no DE metric at all -> never resolve an engine the run does not need
    assert _de_backend_used(_cfg(), resolve_metrics(_cfg().metrics)[0], None) is False
    # ...and the digest follows it, so a both-supplied run and a computed-DE baseline
    # differ on de_real provenance rather than on the prediction side
    assert config_digest(cfg, comparator="lognorm") != config_digest(
        cfg, comparator="lognorm", de_real=_de_features(["g0"]),
    )


def test_runs_under_v1_counts_with_validation_ENABLED(synthetic_counts_pair):
    """The two forcings and the lock together: v1 auto-detects both sides, the profile is
    fractional, and validation is on. Without lock_matrix_space the two sides land in
    different spaces; without allow_fractional_counts the pred side fails validation."""
    _, real = synthetic_counts_pair
    res = build_generic_baseline(
        real, config=_cfg(input_type="counts", version="v1", validate_input=True),
        exclude_target_gene=False,
    )
    assert res.meta["input_type_real_effective"] == "counts"
    assert res.meta["input_type_pred_effective"] == "counts"
    assert res.meta["allow_discrete_effective"] is True
    mae = dict(zip(res.agg["statistic"], res.agg[res.agg.columns[1]]))["mean"]
    assert np.isfinite(mae)


def test_runs_under_v2_autodetect_with_validation_ENABLED(synthetic_counts_pair):
    _, real = synthetic_counts_pair
    res = build_generic_baseline(
        real,
        config=_cfg(input_type="counts", autodetect_input_type=True, validate_input=True),
        exclude_target_gene=False,
    )
    assert res.meta["input_type_pred_effective"] == "counts"


# --------------------------------------------------------------- section 7.1 gate

def _degenerate_reference():
    """Two IDENTICAL non-control perturbations + a control. The profile then equals every
    perturbation's own pseudobulk, so expr_mae is exactly 0.0 -- an anchor-0 baseline whose
    denominator is 0, which score_metrics rejects outright."""
    import anndata as ad
    import pandas as pd
    X = np.array([[1., 2., 3.], [1., 2., 3.], [9., 8., 7.]], dtype=np.float64)
    obs = pd.DataFrame({"target": ["pA", "pB", "non-targeting"]},
                       index=["c0", "c1", "c2"])
    return ad.AnnData(X=X, obs=obs, var=pd.DataFrame(index=["g0", "g1", "g2"]))


def _mae_only():
    # The degeneracy fixture is lognorm. Dispersed is counts-only (§3.2a); tile preserves
    # the exact-zero denominator these tests isolate.
    return EvalConfig(metrics=["expr_mae"], pert_col="target", control="non-targeting",
                      input_type="lognorm", validate_input=False)


def test_degenerate_baseline_raises_before_returning():
    with pytest.raises(ValueError, match="expr_mae"):
        build_generic_baseline(_degenerate_reference(), config=_mae_only(),
                               exclude_target_gene=False, emit="tile")


def test_allow_degenerate_now_WAIVES_THE_ZERO_CLASS_but_the_artifact_is_unscoreable(caplog):
    """The behaviour change, end to end. `_degenerate_reference()` gives expr_mae == 0.0 --
    today the ERROR class the flag explicitly could NOT waive, so this build raises. Now the
    flag waives it: the build returns, the offender is recorded in the stamp, and the
    refusal happens where the denominator is actually used. Asserted through
    build_generic_baseline rather than through _degenerate_message, because the message is
    not the behaviour and a string assertion would keep passing if the gate stopped working.
    """
    with caplog.at_level("WARNING"):
        res = build_generic_baseline(_degenerate_reference(), config=_mae_only(),
                                     exclude_target_gene=False, allow_degenerate=True,
                                     emit="tile")
    assert [d["metric"] for d in res.meta["degenerate_metrics"]] == ["expr_mae"]
    assert res.meta["degenerate_metrics"][0]["direction"] == "lower"   # key renamed from best_value
    # expr_mae is DECISIVE (v1-available and in vcc), so the artifact is refused at scoring
    # time rather than partially scored -- the stamp records which side of that split it is on.
    assert res.meta["degenerate_metrics"][0]["decisive"] is True
    assert any("REFUSES" in r.message and "expr_mae" in r.message for r in caplog.records)

    # ...and without the flag it is still refused at the gate
    with pytest.raises(ValueError, match="expr_mae"):
        build_generic_baseline(_degenerate_reference(), config=_mae_only(),
                               exclude_target_gene=False, emit="tile")

    # ...and the artifact the waiver produced is refused by the scorer
    user = pl.DataFrame({"statistic": ["mean"], "expr_mae": [0.5]})
    base = pl.DataFrame({"statistic": ["mean"], "expr_mae": [0.0]})
    with pytest.raises(ValueError, match="degenerate baseline"):
        score_metrics(user, base)


def test_a_purely_DIAGNOSTIC_profile_is_rejected():
    """aggregate_metrics_wide only refuses ZERO metric columns and _degenerate_metrics skips
    every unscored column as inert, so a profile of purely diagnostic metrics sails through
    both and reaches score_metrics -- which scores nothing and falls back to a vacuous
    avg_score of 0.0, a number that reads like a result."""
    _, real = None, _degenerate_reference()
    cfg = EvalConfig(metrics=["de_wilcoxon_nsig_counts_real"], pert_col="target",
                     control="non-targeting", input_type="lognorm", validate_input=False)
    with pytest.raises(ValueError, match="no scoreable metric"):
        build_generic_baseline(real, config=cfg, exclude_target_gene=False, emit="tile")


def test_degenerate_baseline_writes_NOTHING_the_caller_asked_for(tmp_path):
    """'Before the artifact is written' is literal: a rejected baseline must not leave a
    --save-pred h5ad behind. (compute_metrics writes its own run_params.yaml when outdir is
    set; that is a record of the attempt, not an artifact anyone scores.)"""
    out = tmp_path / "should_not_exist.h5ad"
    with pytest.raises(ValueError, match="expr_mae"):
        build_generic_baseline(_degenerate_reference(), config=_mae_only(),
                               exclude_target_gene=False, save_pred=str(out), emit="tile")
    assert not out.exists()


def test_gate_covers_the_STATISTIC_score_will_actually_use():
    """score --comparison-statistic accepts any row, and a std row is NaN wherever the
    sample std is undefined -- which re-creates the silent zeroing on a baseline whose mean
    row is healthy. The helper therefore takes the statistic; Task 6 makes `score` pass the
    one it was asked for."""
    from cell_eval2.baseline import _degenerate_metrics
    agg = pl.DataFrame({"statistic": ["mean", "std"],
                        "delta_pearson": [0.4, float("nan")]})
    assert _degenerate_metrics(agg) == []                       # mean is fine
    bad = _degenerate_metrics(agg, statistic="std")
    assert [d["metric"] for d in bad] == ["delta_pearson"]
    assert bad[0]["statistic"] == "std" and bad[0]["value"] is None


def test_gate_catches_the_FORMERLY_SILENT_case_too():
    """The case that motivated the gate: an anchor-1 base of NaN or 1.0 used NOT to raise
    in score_metrics -- _norm_by_one returned NaN, which was clipped to 0.0, so every
    submission scored exactly 0 on that metric forever with no error. score_metrics now
    refuses it too (spec 6), and this gate still reports it earlier and better-located.
    Asserted directly on the helper, since an all-NaN aggregate is hard to arrange live."""
    from cell_eval2.baseline import _degenerate_metrics
    agg = pl.DataFrame({"statistic": ["mean"], "delta_pearson": [1.0],
                        "pds_cosine": [float("nan")], "expr_mae": [0.5]})
    found = {d["metric"]: d for d in _degenerate_metrics(agg)}
    assert set(found) == {"delta_pearson", "pds_cosine"}
    # non-finite is recorded as None, not NaN: json.dump would otherwise emit a bare `NaN`
    assert found["pds_cosine"]["value"] is None
    json.dumps(found["pds_cosine"], allow_nan=False)
