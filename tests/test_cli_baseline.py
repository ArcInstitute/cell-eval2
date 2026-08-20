import json

import anndata as ad
import numpy as np
import polars as pl
import pytest
import scipy.sparse as sp

from cell_eval2.cli import main


@pytest.fixture
def real_path(synthetic_pair, tmp_path):
    _, real = synthetic_pair
    p = tmp_path / "real.h5ad"
    real.write_h5ad(p)
    return p


_COMMON = ["--profile", "vcc", "--pert-col", "target",
           "--control", "non-targeting", "--input-type", "lognorm",
           "--set", "validate_input=false"]
# Existing baseline CLI tests use this lognorm fixture. Dispersed is counts-only (§3.2a),
# and these tests cover artifact/config plumbing, so the baseline arm explicitly stays tile.
_BASELINE_COMMON = [*_COMMON, "--emit", "tile"]


def test_run_also_writes_agg_results(synthetic_pair, tmp_path):
    pred, real = synthetic_pair
    pp, rp = tmp_path / "p.h5ad", tmp_path / "r.h5ad"
    pred.write_h5ad(pp)
    real.write_h5ad(rp)
    out = tmp_path / "run_out"
    main(["run", "-ap", str(pp), "-ar", str(rp), "-o", str(out), *_COMMON])
    assert (out / "results.csv").exists()
    agg = pl.read_csv(out / "agg_results.csv")
    assert agg.columns[0] == "statistic"
    assert "mean" in agg["statistic"].to_list()


def test_run_and_baseline_agg_columns_match_exactly(synthetic_pair, real_path, tmp_path):
    """score_metrics compares the column lists as ORDERED lists. Both sides pass the same
    expected metric names, so they match by construction rather than by luck."""
    pred, _ = synthetic_pair
    pp = tmp_path / "p.h5ad"
    pred.write_h5ad(pp)
    run_out, bl_out = tmp_path / "u0", tmp_path / "b0"
    main(["run", "-ap", str(pp), "-ar", str(real_path), "-o", str(run_out), *_COMMON])
    main(["baseline", "-ar", str(real_path), "-o", str(bl_out), *_BASELINE_COMMON])
    assert (pl.read_csv(run_out / "agg_results.csv").columns
            == pl.read_csv(bl_out / "baseline_agg.csv").columns)


def test_run_writes_run_meta_with_resolved_identity(synthetic_pair, tmp_path):
    """Without this file `score` has nothing on the user side to compare the baseline's
    resolved backend/device/reference against, and those stamped fields are decorative."""
    pred, real = synthetic_pair
    pp, rp = tmp_path / "p.h5ad", tmp_path / "r.h5ad"
    pred.write_h5ad(pp)
    real.write_h5ad(rp)
    out = tmp_path / "rm"
    main(["run", "-ap", str(pp), "-ar", str(rp), "-o", str(out), *_COMMON])
    meta = json.loads((out / "run_meta.json").read_text())
    for field in ("source_fingerprint", "resolved_device", "resolved_de_backend",
                  "input_type_real_effective", "input_type_pred_effective",
                  "comparator", "config_digest"):
        assert field in meta, field
    assert meta["resolved_device"] in ("cpu", "cuda")     # RESOLVED, never "auto"
    assert meta["comparator"] == "lognorm"


def test_score_detects_a_DIFFERENT_REFERENCE(synthetic_pair, real_path, tmp_path, caplog):
    """The failure design section 6 exists to prevent: identical configs, different
    reference. BOTH arms, because the level of the check is a real limit, not a footnote.

    fingerprint_adata is metadata-only unless cache_strict (cache.py:88-100) -- deliberately,
    because build_run_meta must not make ordinary `run` content-hash both sides. So a
    reference differing ONLY in X is caught under --cache-strict and is NOT caught without
    it; the second arm measures that rather than leaving it as a claim.
    """
    pred, real = synthetic_pair
    pp = tmp_path / "p.h5ad"
    pred.write_h5ad(pp)
    other = real.copy()
    np.asarray(other.X)[0, 0] += 1.0                     # same shape, labels, dtype, var
    op = tmp_path / "other.h5ad"
    other.write_h5ad(op)

    def _pair(tag, extra):
        run_out, bl_out = tmp_path / f"u6{tag}", tmp_path / f"b6{tag}"
        main(["run", "-ap", str(pp), "-ar", str(real_path), "-o", str(run_out),
              *_COMMON, *extra])
        main(["baseline", "-ar", str(op), "-o", str(bl_out), *_BASELINE_COMMON, *extra])
        return ["score", "--user-agg", str(run_out / "agg_results.csv"),
                "--baseline-agg", str(bl_out / "baseline_agg.csv")]

    strict = _pair("s", ["--cache-strict"])
    with pytest.raises(SystemExit, match="source_fingerprint"):
        main(strict)
    main([*strict, "--allow-config-mismatch"])

    # ...and the documented limit, measured: metadata-only cannot see an X-only change
    loose = _pair("l", [])
    with caplog.at_level("WARNING"):
        main([*loose, "-o", str(tmp_path / "loose.csv")])
    assert any("metadata level" in r.message for r in caplog.records)


def test_score_is_FAIL_CLOSED_on_an_incomplete_record(synthetic_pair, real_path, tmp_path):
    """A missing key must be a mismatch, not a match: comparing with .get() would let two
    empty JSON objects pass as 'fully verified'."""
    pred, _ = synthetic_pair
    pp = tmp_path / "p.h5ad"
    pred.write_h5ad(pp)
    run_out, bl_out = tmp_path / "u8", tmp_path / "b8"
    main(["run", "-ap", str(pp), "-ar", str(real_path), "-o", str(run_out), *_COMMON])
    main(["baseline", "-ar", str(real_path), "-o", str(bl_out), *_BASELINE_COMMON])
    # delete from BOTH records: removing it from only one would also be caught by a plain
    # .get() comparison (value vs None), so that weaker version would not discriminate.
    for f in (run_out / "run_meta.json", bl_out / "baseline_meta.json"):
        meta = json.loads(f.read_text())
        del meta["source_fingerprint"]
        f.write_text(json.dumps(meta))
    with pytest.raises(SystemExit, match="source_fingerprint"):
        main(["score", "--user-agg", str(run_out / "agg_results.csv"),
              "--baseline-agg", str(bl_out / "baseline_agg.csv")])


def test_score_says_NOT_VERIFIED_when_run_meta_is_absent(real_path, tmp_path, caplog):
    """No fallback to run_params.yaml: it is written POST target_sum resolution, so
    digesting it would mismatch two runs that both requested None. Say so instead."""
    bl = tmp_path / "b9"
    main(["baseline", "-ar", str(real_path), "-o", str(bl), *_BASELINE_COMMON])
    agg = str(bl / "baseline_agg.csv")
    with caplog.at_level("WARNING"):
        main(["score", "--user-agg", agg, "--baseline-agg", agg,
              "-o", str(tmp_path / "nv.csv")])
    assert any("NOT verified" in r.message for r in caplog.records)


def test_score_detects_a_different_SUPPLIED_de_real(synthetic_pair, real_path, tmp_path):
    """A supplied real-side DE table changes the numbers (see the builder's de_real test)
    but is invisible to the config, so only its fingerprint catches the mismatch."""
    pred, _ = synthetic_pair
    pp = tmp_path / "p.h5ad"
    pred.write_h5ad(pp)
    de = tmp_path / "de.csv"
    pl.DataFrame([{"target": t, "feature": f, "log2_fold_change": 3.0, "p_adj": 0.001}
                  for t in ("GENE1", "GENE2", "GENE3") for f in ("g0", "g1", "g2")]
                 ).write_csv(de)
    other = tmp_path / "de_other.csv"
    pl.DataFrame([{"target": t, "feature": f, "log2_fold_change": 3.0, "p_adj": 0.001}
                  for t in ("GENE1", "GENE2", "GENE3") for f in ("g3", "g4", "g5")]
                 ).write_csv(other)
    run_out, bl_out = tmp_path / "u10", tmp_path / "b10"
    main(["run", "-ap", str(pp), "-ar", str(real_path), "-o", str(run_out),
          "--de-real", str(de), *_COMMON])
    # DIFFERENT non-null tables on both sides: a boolean "supplied" marker, or a constant,
    # would pass this. Only a content fingerprint separates them.
    main(["baseline", "-ar", str(real_path), "-o", str(bl_out),
          "--de-real", str(other), *_BASELINE_COMMON])
    args = ["score", "--user-agg", str(run_out / "agg_results.csv"),
            "--baseline-agg", str(bl_out / "baseline_agg.csv")]
    with pytest.raises(SystemExit, match="de_real_fingerprint"):
        main(args)
    # ...and the SAME table on both sides must pass
    bl_same = tmp_path / "b10s"
    main(["baseline", "-ar", str(real_path), "-o", str(bl_same),
          "--de-real", str(de), *_BASELINE_COMMON])
    main(["score", "--user-agg", str(run_out / "agg_results.csv"),
          "--baseline-agg", str(bl_same / "baseline_agg.csv"),
          "-o", str(tmp_path / "de_ok.csv")])


def test_score_rejects_a_degenerate_COMPARISON_STATISTIC(real_path, tmp_path):
    """The build-time gate validates `mean`; `--comparison-statistic std` is a different
    row, and a NaN there re-creates the silent zeroing."""
    bl = tmp_path / "b7"
    main(["baseline", "-ar", str(real_path), "-o", str(bl), *_BASELINE_COMMON])
    agg_path = bl / "baseline_agg.csv"
    agg = pl.read_csv(agg_path)
    # de_wilcoxon_overlap is anchor-1 (the formerly SILENT class: NaN -> clipped to 0.0
    # forever). It is also DECISIVE -- it is v1-available AND in the vcc profile -- so
    # score_metrics refuses a baseline degenerate on it (spec 6), and
    # --allow-degenerate-baseline does not rescue it: the flag governs WRITING a diagnostic
    # baseline, not scoring against one. (Not every degenerate metric is refused now; a
    # v2-native full/de-only one is dropped from avg_score instead. This one is not that.)
    col = "de_wilcoxon_overlap"
    agg.with_columns(
        pl.when(pl.col("statistic") == "std").then(float("nan")).otherwise(pl.col(col))
        .alias(col)
    ).write_csv(agg_path)
    args = ["score", "--user-agg", str(agg_path), "--baseline-agg", str(agg_path),
            "--comparison-statistic", "std", "-o", str(tmp_path / "s.csv")]
    with pytest.raises(SystemExit, match="degenerate"):
        main(args)
    # ...and the flag does NOT rescue it: the gate is unconditional now.
    with pytest.raises(SystemExit, match="does not cover scoring"):
        main([*args, "--allow-degenerate-baseline"])


def test_baseline_does_not_clobber_the_resolved_run_params(real_path, tmp_path):
    """target_sum=None is resolved to the real control pool's median INSIDE
    compute_metrics (run.py:765-773), which then writes that resolved config to
    run_params.yaml (run.py:905). Rewriting the file from the stamp's `config` would put
    `null` back."""
    import yaml
    out = tmp_path / "bl_ts"
    main(["baseline", "-ar", str(real_path), "-o", str(out), *_BASELINE_COMMON,
          "--set", "target_sum=null", "--set", "input_type=counts",
          "--set", "allow_fractional_counts=true"])
    written = yaml.safe_load((out / "run_params.yaml").read_text())
    assert written["target_sum"] is not None, "run_params.yaml lost the resolved target_sum"


def test_baseline_writes_all_four_artifacts(real_path, tmp_path):
    out = tmp_path / "bl"
    main(["baseline", "-ar", str(real_path), "-o", str(out), *_BASELINE_COMMON])
    for name in ("baseline_agg.csv", "baseline_results.csv",
                 "baseline_meta.json", "run_params.yaml"):
        assert (out / name).exists(), name
    meta = json.loads((out / "baseline_meta.json").read_text())
    assert meta["exclude_target_gene"] is True
    assert meta["degenerate_metrics"] == []
    assert "config_digest" in meta
    assert "source_fingerprint" in meta
    assert meta["comparator"] == "lognorm"


def test_baseline_no_exclude_flag_is_recorded(real_path, tmp_path):
    out = tmp_path / "bl_off"
    main(["baseline", "-ar", str(real_path), "-o", str(out),
          "--no-exclude-target-gene", *_BASELINE_COMMON])
    meta = json.loads((out / "baseline_meta.json").read_text())
    assert meta["exclude_target_gene"] is False
    assert meta["n_excluded"] == 0


def test_baseline_save_pred(real_path, tmp_path):
    out = tmp_path / "bl_sp"
    pred_path = tmp_path / "bl_pred.h5ad"
    main(["baseline", "-ar", str(real_path), "-o", str(out),
          "--save-pred", str(pred_path), *_BASELINE_COMMON])
    assert pred_path.exists()


def test_baseline_cli_emission_default_tile_override_and_seed(synthetic_counts_pair, tmp_path):
    """End-to-end counts coverage for #234: omitting --emit must select dispersed,
    unequal --seed values must change its saved cells, and --emit tile must select the
    legacy matrix. Construction choices belong in baseline_meta, not config_digest."""
    import anndata as ad

    _, real = synthetic_counts_pair
    real_file = tmp_path / "counts.h5ad"
    real.write_h5ad(real_file)
    common = [
        # `minimal`, not `anndata`: see tests/test_cli.py::test_cli_run_with_cache_dirs --
        # #257's derived metric refuses this null fixture's non-positive denominator sum.
        "--profile", "minimal", "--pert-col", "target", "--control", "non-targeting",
        "--input-type", "counts", "--no-exclude-target-gene", "--set", "device=cpu",
    ]

    def run(tag, seed, extra=()):
        out = tmp_path / tag
        pred = tmp_path / f"{tag}.h5ad"
        main([
            "baseline", "-ar", str(real_file), "-o", str(out), "--seed", str(seed),
            "--save-pred", str(pred), *extra, *common,
        ])
        return json.loads((out / "baseline_meta.json").read_text()), ad.read_h5ad(pred)

    def dense(adata):
        """The prediction MIRRORS the template's sparsity now, so a saved `X` is dense only
        because `synthetic_counts_pair` is. Read through an accessor rather than depending
        on that: `np.array_equal` on two scipy sparse matrices does not compare contents --
        measured, it raises `ValueError: The truth value of an array ... is ambiguous`, so a
        sparse fixture would turn these into an unrelated-looking crash. (Gemini called it a
        vacuous `False` on PR #241; that mechanism is wrong, and its suggested `.toarray()`
        patch would itself `AttributeError` on the ndarray this fixture actually produces.)"""
        X = adata.X
        return X.toarray() if sp.issparse(X) else np.asarray(X)

    first_meta, first = run("default17", 17)
    second_meta, second = run("default18", 18)
    tile_meta, tile = run("tile", 23, ("--emit", "tile"))

    assert first_meta["emit"] == second_meta["emit"] == "dispersed"
    assert first_meta["seed"] == 17 and second_meta["seed"] == 18
    assert first_meta["baseline_emission"]["seed"] == 17
    # discriminating in both directions: a self-comparison must be True, or "the seeds
    # differ" would hold for a comparison that can never report equality at all
    assert np.array_equal(dense(first), dense(first))
    assert not np.array_equal(dense(first), dense(second))
    assert tile_meta["emit"] == tile_meta["baseline_emission"]["emit"] == "tile"
    assert tile_meta["seed"] == 23
    assert not np.array_equal(dense(first), dense(tile))
    assert first_meta["config_digest"] == tile_meta["config_digest"]


def test_baseline_forwards_de_real(real_path, tmp_path):
    """--de-real was accepted and silently ignored. A garbage path must fail, proving the
    argument reaches the loader; a valid table must be recorded in the stamp."""
    out = tmp_path / "bl_de"
    with pytest.raises(Exception):
        main(["baseline", "-ar", str(real_path), "-o", str(out),
              "--de-real", str(tmp_path / "nope.csv"), *_BASELINE_COMMON])
    de = tmp_path / "de_real.csv"
    pl.DataFrame([{"target": t, "feature": f, "log2_fold_change": 3.0, "p_adj": 0.001}
                  for t in ("GENE1", "GENE2", "GENE3") for f in ("g0", "g1", "g2")]
                 ).write_csv(de)
    out2 = tmp_path / "bl_de2"
    main(["baseline", "-ar", str(real_path), "-o", str(out2),
          "--de-real", str(de), *_BASELINE_COMMON])
    assert json.loads((out2 / "baseline_meta.json").read_text())["de_real_supplied"] is True


def test_score_verb_end_to_end(synthetic_pair, real_path, tmp_path):
    pred, _ = synthetic_pair
    pp = tmp_path / "p.h5ad"
    pred.write_h5ad(pp)
    run_out, bl_out = tmp_path / "u", tmp_path / "b"
    main(["run", "-ap", str(pp), "-ar", str(real_path), "-o", str(run_out), *_COMMON])
    main(["baseline", "-ar", str(real_path), "-o", str(bl_out), *_BASELINE_COMMON])
    scored = tmp_path / "scored.csv"
    main(["score", "--user-agg", str(run_out / "agg_results.csv"),
          "--baseline-agg", str(bl_out / "baseline_agg.csv"), "-o", str(scored)])
    df = pl.read_csv(scored)
    assert df.columns == ["metric", "from_baseline"]
    assert "avg_score" in df["metric"].to_list()


def test_score_cli_lfc_nmae_ref_emits_from_reference(tmp_path):
    """The parser + dispatch, not just score_metrics: without this the flag and its
    dispatch kwarg could BOTH be deleted and the suite would stay green.

    Deliberately does NOT reuse this file's `_COMMON`: its profile is `vcc`, and #208 keeps
    the member OUT of vcc on purpose, so the aggregate would have no such column and the
    filter below would return an empty frame. Hand-built aggregates instead -- this test is
    about argument plumbing, and a real `run` would make the assertion depend on whether a
    random fixture happened to produce ten gated genes.
    """
    import polars as pl
    user = tmp_path / "user_agg.csv"
    base = tmp_path / "base_agg.csv"
    ref = tmp_path / "ref_agg.csv"
    pl.DataFrame({"statistic": ["mean"], "de_wilcoxon_lfc_nmae": [0.68]}).write_csv(user)
    pl.DataFrame({"statistic": ["mean"], "de_wilcoxon_lfc_nmae": [0.96]}).write_csv(base)
    pl.DataFrame({"statistic": ["mean"], "nmae_ref_raw": [0.9], "nmae_ref_sqrt2": [0.4],
                  "n_perturbations": [3]}).write_csv(ref)
    scored = tmp_path / "scored.csv"
    main(["score", "--user-agg", str(user), "--baseline-agg", str(base),
          "--lfc-nmae-ref", str(ref), "-o", str(scored)])
    df = pl.read_csv(scored)
    assert df.columns == ["metric", "from_baseline", "from_reference"]
    got = df.filter(pl.col("metric") == "de_wilcoxon_lfc_nmae")["from_reference"].item()
    assert got == pytest.approx((1 - 0.68) / (1 - 0.9), abs=1e-12)   # RAW denominator
    # and DISTINGUISHABLE from the old sqrt(2) arithmetic, so this can fail
    assert got != pytest.approx((1 - 0.68) / (1 - 0.4), abs=1e-12)


def test_score_cli_without_the_flag_has_two_columns(tmp_path):
    """The complement, so the conditional column is pinned from BOTH sides at the CLI."""
    import polars as pl
    user, base = tmp_path / "u.csv", tmp_path / "b.csv"
    pl.DataFrame({"statistic": ["mean"], "de_wilcoxon_lfc_nmae": [0.68]}).write_csv(user)
    pl.DataFrame({"statistic": ["mean"], "de_wilcoxon_lfc_nmae": [0.96]}).write_csv(base)
    scored = tmp_path / "s.csv"
    main(["score", "--user-agg", str(user), "--baseline-agg", str(base), "-o", str(scored)])
    assert pl.read_csv(scored).columns == ["metric", "from_baseline"]


def test_score_baseline_against_itself_is_zero(real_path, tmp_path):
    bl = tmp_path / "b2"
    main(["baseline", "-ar", str(real_path), "-o", str(bl), *_BASELINE_COMMON])
    agg = str(bl / "baseline_agg.csv")
    scored = tmp_path / "self.csv"
    main(["score", "--user-agg", agg, "--baseline-agg", agg, "-o", str(scored)])
    df = pl.read_csv(scored)
    got = df.filter(pl.col("metric") == "avg_score")["from_baseline"][0]
    assert got == pytest.approx(0.0)


def test_score_detects_a_config_mismatch(synthetic_pair, real_path, tmp_path):
    """A different p_adj_threshold produces identically-shaped frames, so only the
    digest can catch it."""
    pred, _ = synthetic_pair
    pp = tmp_path / "p.h5ad"
    pred.write_h5ad(pp)
    run_out, bl_out = tmp_path / "u3", tmp_path / "b3"
    main(["run", "-ap", str(pp), "-ar", str(real_path), "-o", str(run_out), *_COMMON])
    main(["baseline", "-ar", str(real_path), "-o", str(bl_out), *_BASELINE_COMMON,
          "--set", "de.p_adj_threshold=0.01"])
    args = ["score", "--user-agg", str(run_out / "agg_results.csv"),
            "--baseline-agg", str(bl_out / "baseline_agg.csv")]
    with pytest.raises(SystemExit, match="config_digest"):
        main(args)
    main([*args, "--allow-config-mismatch"])          # override works


def test_score_matching_configs_pass_the_check(synthetic_pair, real_path, tmp_path):
    """The forced knobs must all be digest-exempt, or this fails by construction: the
    baseline forces allow_fractional_counts while the user run keeps the default."""
    pred, _ = synthetic_pair
    pp = tmp_path / "p.h5ad"
    pred.write_h5ad(pp)
    run_out, bl_out = tmp_path / "u4", tmp_path / "b4"
    main(["run", "-ap", str(pp), "-ar", str(real_path), "-o", str(run_out), *_COMMON])
    main(["baseline", "-ar", str(real_path), "-o", str(bl_out), *_BASELINE_COMMON])
    main(["score", "--user-agg", str(run_out / "agg_results.csv"),
          "--baseline-agg", str(bl_out / "baseline_agg.csv"),
          "-o", str(tmp_path / "ok.csv")])


def test_score_matching_configs_pass_under_v1_counts(synthetic_counts_pair, tmp_path):
    """The v1 arm of the same guarantee: here the baseline additionally forces
    allow_discrete (design 3.0), which must NOT perturb the digest -- otherwise every v1
    baseline mismatches every v1 submission and --allow-config-mismatch becomes routine,
    which defeats the check."""
    pred, real = synthetic_counts_pair
    pp, rp = tmp_path / "p.h5ad", tmp_path / "r.h5ad"
    pred.write_h5ad(pp)
    real.write_h5ad(rp)
    common = ["--profile", "minimal", "--pert-col", "target", "--control", "non-targeting",
              "--input-type", "counts", "--version", "v1"]
    run_out, bl_out = tmp_path / "u5", tmp_path / "b5"
    main(["run", "-ap", str(pp), "-ar", str(rp), "-o", str(run_out), *common])
    main(["baseline", "-ar", str(rp), "-o", str(bl_out), "--no-exclude-target-gene", *common])
    meta = json.loads((bl_out / "baseline_meta.json").read_text())
    assert meta["allow_discrete_effective"] is True      # the forcing did happen
    main(["score", "--user-agg", str(run_out / "agg_results.csv"),
          "--baseline-agg", str(bl_out / "baseline_agg.csv"),
          "-o", str(tmp_path / "ok_v1.csv")])


_VCC2026_COMMON = ["--profile", "vcc2026", "--pert-col", "target",
                   "--control", "non-targeting", "--input-type", "lognorm",
                   "--set", "validate_input=false"]


def _vcc2026_aggs(synthetic_pair, tmp_path, tag):
    """A user agg and a baseline agg that between them cover every metric the scale names.

    Takes ``synthetic_pair`` and NOT the file-backed ``real_path`` fixture: the reference has
    to be widened below, so this helper writes its own copy. An earlier signature accepted
    ``real_path`` and ignored it, which invited the reading that the on-disk fixture was what
    got scored.

    ⚠️ The 8x widening REPEATS the deterministic groups rather than drawing more cells: at 25
    cells the six-metric DE profile has an empty real gate and four degenerate baseline cells,
    and the plan's Task 4 contingency is to widen the fixture rather than weaken the raise.
    Duplicated cells inflate n without adding information, so the DE statistics here are
    anticonservative -- these tests assert COLUMN SHAPE only and never a metric value. Do not
    read a number off this fixture.
    """
    pred, real = synthetic_pair
    pred = ad.concat([pred] * 8, index_unique="-")
    real = ad.concat([real] * 8, index_unique="-")
    pp, rp = tmp_path / f"p_{tag}.h5ad", tmp_path / f"r_{tag}.h5ad"
    pred.write_h5ad(pp)
    real.write_h5ad(rp)
    run_out, bl_out = tmp_path / f"u_{tag}", tmp_path / f"b_{tag}"
    main(["run", "-ap", str(pp), "-ar", str(rp), "-o", str(run_out),
          *_VCC2026_COMMON])
    main(["baseline", "-ar", str(rp), "-o", str(bl_out),
          *_VCC2026_COMMON, "--emit", "tile"])
    return str(run_out / "agg_results.csv"), str(bl_out / "baseline_agg.csv")


def test_cli_score_with_scale_adds_the_column(synthetic_pair, tmp_path):
    user_agg, base_agg = _vcc2026_aggs(synthetic_pair, tmp_path, "s1")
    out = tmp_path / "scored1.csv"
    main(["score", "--user-agg", user_agg, "--baseline-agg", base_agg,
          "--scale", "low-random_high-1_v10", "-o", str(out)])
    assert pl.read_csv(out).columns == ["metric", "from_baseline",
                                        "low-random_high-1_v10"]


def test_cli_score_scale_only_needs_no_baseline(synthetic_pair, tmp_path):
    user_agg, _ = _vcc2026_aggs(synthetic_pair, tmp_path, "s2")
    out = tmp_path / "scored2.csv"
    main(["score", "--user-agg", user_agg, "--scale", "low-random_high-1_v10",
          "-o", str(out)])
    assert pl.read_csv(out).columns == ["metric", "low-random_high-1_v10"]


def test_cli_score_without_baseline_or_scale_exits(tmp_path):
    with pytest.raises(SystemExit, match="nothing to score against"):
        main(["score", "--user-agg", str(tmp_path / "unused.csv")])
