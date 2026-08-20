import polars as pl
import pytest

from cell_eval2 import EvalConfig

PROFILE_KW = dict(metrics="anndata", pert_col="target", input_type="lognorm",
                  validate_input=False)


def _wide(**cols):
    return pl.DataFrame({"statistic": ["mean"], **{k: [v] for k, v in cols.items()}})


def _anchor_frame(**vals):
    from cell_eval2.anchor import _ANCHOR_SCHEMA, SPLIT_HALF_RAW

    n = len(vals)
    return pl.DataFrame(
        {"metric": list(vals), "replicate": [float(v) for v in vals.values()],
         "replicate_sd": [0.0] * n, "replicate_min": list(vals.values()),
         "replicate_max": list(vals.values()), "n_perturbations_min": [5] * n,
         "n_perturbations_max": [5] * n, "estimator": [SPLIT_HALF_RAW] * n},
        schema=_ANCHOR_SCHEMA)


def _col(row_names, u, base, anchor):
    """The replicate column under test, on plain values. Same shape as the existing
    `_from_reference_column` unit tests -- the ARITHMETIC is tested here, on the private
    builders, so the public API can require a validated bundle without these becoming
    end-to-end tests.

    Since #276 part C this composes the two halves the implementation uses: the entries
    builder (which resolves each member's policy against the measured replicate) and the
    shared reference column (which scores and averages them).

    ⚠️ `avg_score` is APPENDED here rather than required of every caller. The shared column
    writes its mean to the last row by position and raises if that row is not `avg_score`,
    while the existing callers pass bare metric lists -- so without this the migrated helper
    would raise for all of them.
    """
    from cell_eval2.score import _reference_column, _replicate_entries

    names = list(u)
    entries = _replicate_entries(base, anchor)
    rows = list(row_names) + (["avg_score"] if "avg_score" not in row_names else [])
    return dict(zip(rows, _reference_column(
        rows, [u[n] for n in names], names, entries,
        column="from_replicate", label="the replicate anchor").to_list()))


def test_from_replicate_puts_zero_at_the_baseline_and_one_at_the_anchor():
    """Spec 6.6 and #276 comment 1's scale. Checked at BOTH ends and at a midpoint, on a
    lower-is-better (expr_mae) and a higher-is-better (pds_cosine) metric, so a sign error
    cannot pass."""
    rows = ["expr_mae", "pds_cosine"]
    base = {"expr_mae": 0.50, "pds_cosine": 0.20}
    anchor = _anchor_frame(expr_mae=0.10, pds_cosine=0.90)

    at_base = _col(rows, {"expr_mae": 0.50, "pds_cosine": 0.20}, base, anchor)
    at_anchor = _col(rows, {"expr_mae": 0.10, "pds_cosine": 0.90}, base, anchor)
    midway = _col(rows, {"expr_mae": 0.30, "pds_cosine": 0.55}, base, anchor)

    for metric in rows:
        assert at_base[metric] == pytest.approx(0.0)
        assert at_anchor[metric] == pytest.approx(1.0)
        assert midway[metric] == pytest.approx(0.5)


def test_from_replicate_is_not_clamped_above_one():
    """Spec 7: the raw split-half anchor is measured at half depth, so it is an easier bar
    than a full-depth replicate and >1 is expected. Whether the official column clamps is
    part C's call -- B must not decide it silently."""
    got = _col(["pds_cosine", "expr_mse_unbiased_capped_norm"],
               {"pds_cosine": 1.0, "expr_mse_unbiased_capped_norm": 0.02},
               {"pds_cosine": 0.20, "expr_mse_unbiased_capped_norm": 0.50},
               _anchor_frame(pds_cosine=0.90, expr_mse_unbiased_capped_norm=0.10))
    assert got["pds_cosine"] > 1.0
    assert got["expr_mse_unbiased_capped_norm"] == 1.0


def test_rows_the_anchor_does_not_name_are_null():
    got = _col(["expr_mae", "pds_cosine"], {"expr_mae": 0.30, "pds_cosine": 0.4},
               {"expr_mae": 0.50, "pds_cosine": 0.2}, _anchor_frame(expr_mae=0.10))
    assert got["pds_cosine"] is None          # in the aggregate, NOT in the anchor
    assert got["expr_mae"] is not None
    # ...and the average covers only the metric the anchor names.
    assert got["avg_score"] == pytest.approx(got["expr_mae"])


def test_a_degenerate_scale_raises_for_a_decisive_member(caplog):
    """A decisive metric fails loud; a non-decisive one warns and is omitted."""
    import logging

    with pytest.raises(ValueError, match="expr_mae"):
        _col(["expr_mae"], {"expr_mae": 0.30}, {"expr_mae": 0.50},
             _anchor_frame(expr_mae=0.50))

    with caplog.at_level(logging.WARNING):
        got = _col(["de_wilcoxon_direction_yield", "pds_cosine"],
                   {"de_wilcoxon_direction_yield": 0.30, "pds_cosine": 0.55},
                   {"de_wilcoxon_direction_yield": 0.50, "pds_cosine": 0.20},
                   _anchor_frame(de_wilcoxon_direction_yield=0.50, pds_cosine=0.90))
    assert got["de_wilcoxon_direction_yield"] is None
    assert got["pds_cosine"] is not None
    assert "de_wilcoxon_direction_yield" in caplog.text


def test_the_unclamped_linear_core_is_the_replicate_ratio():
    """#276 part C's structural claim, stated exactly: with `Scoring.anchor` set to the
    measured replicate, `score_one`'s UNCLAMPED LINEAR CORE is `(u - b) / (r - b)`.

    This is the core only. The shipped column deliberately diverges from it wherever a policy
    applies -- floors, MSE's ceiling, the non-finite floor -- and each of those divergences is
    pinned by its own test below. The Box-Cox tail is deliberately NOT among them, and the
    reason is about OUTPUT rather than reachability: `expr_mse_unbiased_capped_norm` does still
    execute the tail branch for a bad enough finite value, but its `clamp_low=0.0` clips the
    result to 0, and `de_wilcoxon_lfc_nmae` moved to `ERROR_LINEAR`, so no `vcc2026` number is
    numerically affected by the tail any more. (This file is not `vcc2026`-only -- `PROFILE_KW`
    is `metrics="anndata"` and the cases below include `expr_mae`, which DOES carry the tail --
    so the claim is about the members these divergence tests pin, not about the file.)
    `tests/test_scoring_catalog.py` covers the tail on the `ERROR` metrics. Bit-exact for the four
    higher-is-better members; the two lower-is-better ones carry score_one's frozen
    `1 - (u-a)/(b-a)` order (kept for v1 parity) and agree to within 1 ULP.
    """
    from dataclasses import replace

    from cell_eval2.catalog import CATALOG
    from cell_eval2.scoring import score_one

    NEG, POS = float("-inf"), float("inf")
    exact = [("pds_cosine", 0.70, 0.4931, 0.9701),
             ("de_wilcoxon_sig_jaccard", 0.50, 0.0, 0.4533),
             ("de_wilcoxon_direction_reach_raw", 0.40, 0.0780, 0.8923),
             ("de_wilcoxon_direction_fidelity_yield_raw", 0.55, 0.0028, 0.7990)]
    for name, u, b, rep in exact:
        pol = replace(CATALOG[name].scoring, anchor=rep)
        assert score_one(u, b, pol, penalty="none", clamp_low=NEG,
                         clamp_high=POS) == (u - b) / (rep - b), name

    for name, u, b, rep in [("de_wilcoxon_lfc_nmae", 0.70, 0.9946, 0.3608),
                            ("expr_mse_unbiased_capped_norm", 0.50, 1.0394, 0.0690)]:
        pol = replace(CATALOG[name].scoring, anchor=rep)
        assert score_one(u, b, pol, penalty="none", clamp_low=NEG,
                         clamp_high=POS) == pytest.approx((u - b) / (rep - b), abs=2.3e-16)


def test_the_policy_divergences_from_that_core_are_the_point():
    """Each is a knob the raw ratio did not have, on CCL_1's measured ends."""
    ends = {"pds_cosine": (0.4931, 0.9701),
            "de_wilcoxon_lfc_nmae": (0.9946, 0.3608),
            "expr_mse_unbiased_capped_norm": (1.0394, 0.0690)}
    got = _col(list(ends) + ["avg_score"],
               {"pds_cosine": 0.30, "de_wilcoxon_lfc_nmae": 5.00,
                "expr_mse_unbiased_capped_norm": 0.0},
               {k: v[0] for k, v in ends.items()},
               _anchor_frame(**{k: v[1] for k, v in ends.items()}))
    # UNFLOORED since the clip removal: `pds_cosine` carries clamp_low=None + metric_min=0.0,
    # so the raw below-comparator score now stands instead of being clipped to 0.0. That makes
    # this row a NON-clamp divergence, which is the point -- the two below it are the clamps.
    assert got["pds_cosine"] == pytest.approx(-0.40482180293501047)
    # The nmae user value is 5.00, not the 3.30 this test used while the family carried the
    # Box-Cox tail: under `ERROR_LINEAR` 3.30 lands at -3.637 -- a real number, but then this
    # row would demonstrate no clamp at all, which is what the test is for. 5.00 is past the
    # r = 7 the line reaches the floor at, so the divergence being pinned is still a CLAMP.
    assert got["de_wilcoxon_lfc_nmae"] == -6.0            # floored, raw -6.320
    assert got["expr_mse_unbiased_capped_norm"] == 1.0    # ceiling, raw 1.071


def test_a_metric_whose_policy_forbids_an_anchor_is_still_scored():
    """`de_wilcoxon_direction_yield` carries allow_negative_baseline=True, which
    `Scoring.__post_init__` forbids alongside an anchor -- with an anchor the baseline's side
    IS checkable, so the flag is meaningless rather than conflicting. Clearing it must not
    silently drop a legitimate scored metric from a `full`-profile anchored run."""
    from cell_eval2.score import _replicate_entries

    entries = _replicate_entries({"de_wilcoxon_direction_yield": 0.1},
                                 _anchor_frame(de_wilcoxon_direction_yield=0.6))
    assert "de_wilcoxon_direction_yield" in entries
    assert entries["de_wilcoxon_direction_yield"].scoring.allow_negative_baseline is False


def test_the_anchor_may_not_name_one_metric_twice_under_two_spellings():
    from cell_eval2.score import _replicate_entries

    frame = _anchor_frame(**{"pds_cosine": 0.97})
    frame = pl.concat([frame, frame.with_columns(
        pl.lit("discrimination_score_cosine").alias("metric"))])
    with pytest.raises(ValueError, match="twice"):
        _replicate_entries({"pds_cosine": 0.49}, frame)


def test_restored_rows_land_INSIDE_their_direction_group():
    """`_insert_metric_rows` must not append after both groups: the frame's contract is
    lower-is-better, then higher-is-better, then avg_score, and the payload freezes that
    order. Appending would put a restored lower-is-better metric after the higher block."""
    import polars as pl

    from cell_eval2.score import _insert_metric_rows

    out = pl.DataFrame({"metric": ["de_wilcoxon_lfc_nmae", "pds_cosine", "avg_score"],
                        "from_baseline": [0.1, 0.2, 0.15]})
    got = _insert_metric_rows(out, ["expr_mae", "de_wilcoxon_sig_jaccard"])["metric"].to_list()
    # Each direction group is SORTED, not merely grouped: the frozen rule's
    # `member_order_in_frame` describes sorted groups, and appending would leave the frame in
    # an order it does not describe.
    assert got == ["de_wilcoxon_lfc_nmae", "expr_mae",
                   "de_wilcoxon_sig_jaccard", "pds_cosine", "avg_score"]
    # A metric already present is not duplicated -- a scale and an anchor may name the same
    # dropped metric -- and an already-ordered frame with nothing to restore comes back
    # IDENTICAL, which is what makes the unconditional call safe.
    assert _insert_metric_rows(out, ["pds_cosine"])["metric"].to_list() == out["metric"].to_list()
    assert _insert_metric_rows(out, []).equals(out)
    # ...but a frame whose group order was disturbed (as the SCALE path's append-restoration
    # leaves it) IS normalized, with nothing to add. This is why the function does not early-
    # return on an empty list.
    jumbled = pl.DataFrame({"metric": ["pds_cosine", "de_wilcoxon_sig_jaccard", "avg_score"],
                            "from_baseline": [0.2, 0.3, 0.25]})
    assert _insert_metric_rows(jumbled, [])["metric"].to_list() == [
        "de_wilcoxon_sig_jaccard", "pds_cosine", "avg_score"]


def _score_inputs(meta, frame):
    """A (user, base) wide pair covering exactly the anchor's metric list.

    ⚠️ The baseline is derived from the anchor's MEASURED replicate, per metric and in that
    metric's own direction. A flat constant across metrics is direction-blind, and since #276
    part C that is fatal rather than merely odd: `_replicate_entries` REFUSES a decisive member
    whose baseline leaves no headroom over the replicate, and the old flat 0.5 sat ABOVE
    `pds_cosine`'s measured replicate (0.4167) -- a scale running backwards, which a real
    baseline never is. The user value sits halfway between the two ends, so every member scores
    ~0.5 on the replicate scale while staying non-degenerate on the baseline scale too
    (anchor-0 metrics keep base > 0; anchor-1 metrics keep base < 1).
    """
    from cell_eval2.catalog import CATALOG

    rep = dict(zip(frame["metric"].to_list(), frame["replicate"].to_list()))
    user, base = {}, {}
    # SORTED, because `aggregate_metrics_wide` sorts its metric columns by name and a real
    # aggregate is what this pair stands in for. `_insert_metric_rows` normalizes each
    # direction group into name order, so a fixture in profile order would report a row-order
    # change that no production frame can ever see.
    for n in sorted(meta["metric_names"]):
        direction = CATALOG[n].scoring.direction
        r = rep.get(n)
        if direction is None or r is None:      # a diagnostic: neither pass scores it
            user[n], base[n] = 0.30, 0.50
            continue
        r = float(r)
        b = r + 0.25 if direction == "lower" else (r / 2.0 if r > 0.0 else r - 0.25)
        user[n], base[n] = (b + r) / 2.0, b
    return _wide(**user), _wide(**base)


def build_anchor_dir(real, outdir, **cfg_kw):
    """Duplicated from tests/test_anchor_resolution.py on purpose -- never import across
    test modules (a test module is not an importable API, and pytest's import modes make it
    unreliable). Body identical to the helper there."""
    import os

    from cell_eval2.anchor import (_derive_seeds, build_meta, compute_replicate_anchor,
                                   write_anchor)
    from cell_eval2.catalog import resolve_metrics
    from cell_eval2.run import _resolve_config, metric_output_names

    os.makedirs(str(outdir), exist_ok=True)
    cfg_in = EvalConfig(**{**PROFILE_KW, **cfg_kw})
    resolved = _resolve_config(cfg_in, {})
    names = list(resolve_metrics(resolved.metrics, version=resolved.version)[0])
    splits, anchor = compute_replicate_anchor(real, config=cfg_in, base_seed=0, n_splits=2)
    meta = build_meta(real_ad=real, cfg=resolved, names=names, base_seed=0, n_splits=2,
                      seeds=_derive_seeds(0, 2), metrics=metric_output_names(resolved))
    write_anchor(str(outdir), splits, anchor, meta=meta)
    return str(outdir)


def _expect_for(meta):
    from cell_eval2.anchor import AnchorExpect

    return AnchorExpect(fingerprint=meta["real_fingerprint"],
                        semantic_identity=meta["semantic_identity"],
                        version=meta["cell_eval2_version"],
                        metrics=tuple(meta["metric_names"]))


def test_anchor_source_and_digest_are_stamped_into_the_frame(synthetic_pair_with_effect,
                                                             tmp_path):
    """Spec 4.4: which door was used is a property of the OUTPUT, not of the invocation.
    Exercised through the real supplied door so the label is not hand-written."""
    from cell_eval2.anchor import anchor_digest, read_anchor
    from cell_eval2.score import score_metrics

    _pred, real = synthetic_pair_with_effect
    outdir = build_anchor_dir(real, tmp_path)
    frame, _splits, meta = read_anchor(outdir)
    user, base = _score_inputs(meta, frame)

    out = score_metrics(user, base, anchor=outdir, anchor_expect=_expect_for(meta))
    assert set(out["anchor_source"].drop_nulls().to_list()) == {"supplied"}
    assert set(out["anchor_digest"].drop_nulls().to_list()) == {anchor_digest(frame, meta)}
    assert out["from_replicate"].drop_nulls().len() > 0


def test_score_validates_the_supplied_anchor_against_THIS_RUNs_expectations(
        synthetic_pair_with_effect, tmp_path):
    """The expectation must come from the SCORING run, not from the anchor's own sidecar --
    handing an artifact its own metadata back validates it against itself and passes for any
    artifact whatsoever."""
    from dataclasses import replace as _replace

    from cell_eval2.anchor import read_anchor
    from cell_eval2.score import score_metrics

    _pred, real = synthetic_pair_with_effect
    outdir = build_anchor_dir(real, tmp_path)
    frame, _splits, meta = read_anchor(outdir)
    user, base = _score_inputs(meta, frame)
    foreign = _replace(_expect_for(meta), fingerprint="another-dataset")
    with pytest.raises(ValueError, match="fingerprint"):
        score_metrics(user, base, anchor=outdir, anchor_expect=foreign)


GOOD_RUN_META = {
    "source_fingerprint": "fp-strict", "source_fingerprint_strict": True,
    "cell_eval2_version": "9.9.9", "anchor_semantic_identity": "sem",
    "anchor_metric_names": ["expr_mae"],
}


def test_expect_from_run_meta_builds_a_complete_AnchorExpect():
    from cell_eval2.score import expect_from_run_meta

    exp = expect_from_run_meta(dict(GOOD_RUN_META))
    assert exp.fingerprint == "fp-strict"
    assert exp.semantic_identity == "sem"
    assert exp.version == "9.9.9"
    assert exp.metrics == ("expr_mae",)


def test_a_METADATA_ONLY_user_run_cannot_score_against_an_anchor():
    """The gate is the strict content hash. `build_run_meta` computes source_fingerprint at
    strict=cfg.cache_strict (baseline.py:793), FALSE by default, so a default run carries the
    metadata hash -- under which two datasets with identical structure and different X are
    indistinguishable. Refuse and name the flag rather than silently weakening the gate."""
    from cell_eval2.score import expect_from_run_meta

    meta = dict(GOOD_RUN_META, source_fingerprint_strict=False,
                source_fingerprint="fp-meta")
    with pytest.raises(ValueError, match="cache-strict|cache_strict"):
        expect_from_run_meta(meta)


@pytest.mark.parametrize("missing", list(GOOD_RUN_META))
def test_expect_from_run_meta_is_FAIL_CLOSED(missing):
    """A missing key is a mismatch, not a match -- the same rule `_check_baseline_config`
    states at cli.py:120. `.get()` would let an empty JSON object pass as fully verified."""
    from cell_eval2.score import expect_from_run_meta

    meta = dict(GOOD_RUN_META)
    meta.pop(missing)
    with pytest.raises(ValueError, match=missing):
        expect_from_run_meta(meta)


def test_build_run_meta_records_the_anchor_identity(synthetic_pair_with_effect):
    """`expect_from_run_meta` reads fields that must actually be written. Without this the
    two records drift and the score-side gate reads None."""
    from cell_eval2 import EvalConfig
    from cell_eval2.anchor import semantic_identity
    from cell_eval2.baseline import build_run_meta
    from cell_eval2.catalog import resolve_metrics
    from cell_eval2.run import _resolve_config, metric_output_names

    pred, real = synthetic_pair_with_effect
    cfg = EvalConfig(metrics="anndata", pert_col="target", input_type="lognorm",
                     validate_input=False)
    meta = build_run_meta(cfg, real, pred)
    resolved = _resolve_config(cfg, {})
    names = list(resolve_metrics(resolved.metrics, version=resolved.version)[0])

    assert meta["anchor_semantic_identity"] == semantic_identity(resolved, real, names)
    assert meta["anchor_metric_names"] == metric_output_names(resolved)


def test_build_run_meta_records_the_anchor_identity_from_a_BACKED_path(
        synthetic_pair_with_effect, tmp_path):
    """The path input is the real one: `build_run_meta` opens both sides BACKED and closes
    them in a `finally`, so the anchor identity has to be computed inside the real-side
    iteration. Computed after the loop it would read the PRED handle, or a closed one."""
    from cell_eval2 import EvalConfig
    from cell_eval2.baseline import build_run_meta
    from cell_eval2.score import expect_from_run_meta

    pred, real = synthetic_pair_with_effect
    pp, rp = str(tmp_path / "p.h5ad"), str(tmp_path / "r.h5ad")
    pred.write_h5ad(pp)
    real.write_h5ad(rp)
    cfg = EvalConfig(metrics="anndata", pert_col="target", input_type="lognorm",
                     validate_input=False, cache_strict=True)

    meta = build_run_meta(cfg, rp, pp)
    assert meta["anchor_semantic_identity"] and meta["anchor_metric_names"]
    exp = expect_from_run_meta(meta)          # the real consumer, end to end
    assert exp.semantic_identity == meta["anchor_semantic_identity"]


def test_from_replicate_is_NOT_enrolled_in_avg_score(synthetic_pair_with_effect, tmp_path):
    """Spec 4.4. Passing an anchor must not move a single existing number."""
    from cell_eval2.anchor import read_anchor
    from cell_eval2.score import score_metrics

    _pred, real = synthetic_pair_with_effect
    outdir = build_anchor_dir(real, tmp_path)
    frame, _s, meta = read_anchor(outdir)
    user, base = _score_inputs(meta, frame)

    without = score_metrics(user, base)
    with_anchor = score_metrics(user, base, anchor=outdir,
                                anchor_expect=_expect_for(meta))

    assert "from_replicate" not in without.columns
    assert with_anchor["metric"].to_list() == without["metric"].to_list()
    assert with_anchor["from_baseline"].to_list() == pytest.approx(
        without["from_baseline"].to_list())


def test_score_raises_when_no_anchor_is_available(synthetic_pair_with_effect, tmp_path):
    """Spec 6.5, at the score boundary: with neither door it must refuse rather than
    recompute."""
    from cell_eval2.anchor import read_anchor
    from cell_eval2.score import score_metrics

    _pred, real = synthetic_pair_with_effect
    outdir = build_anchor_dir(real, tmp_path)
    frame, _s, meta = read_anchor(outdir)
    user, base = _score_inputs(meta, frame)
    empty = {"root": str(tmp_path / "empty-cache"), "key": "replicate_anchor",
             "kind": "json", "fingerprint": "nope", "params": {"p": 1}}
    with pytest.raises(ValueError, match="no anchor"):
        score_metrics(user, base, anchor=None, anchor_cache=empty,
                      anchor_expect=_expect_for(meta))


def test_anchor_without_a_baseline_raises(synthetic_pair_with_effect, tmp_path):
    """`from_replicate` needs BOTH ends. Scale-only scoring has no baseline frame, so the 0
    end does not exist and the column would be undefined rather than merely null."""
    from cell_eval2.anchor import read_anchor
    from cell_eval2.score import score_metrics

    from cell_eval2.scales import SCALES

    _pred, real = synthetic_pair_with_effect
    outdir = build_anchor_dir(real, tmp_path)
    frame, _s, meta = read_anchor(outdir)
    user, _base = _score_inputs(meta, frame)
    # ANY currently-shipped scale, taken from the registry rather than named literally. What
    # this test needs is a scale-only call (no baseline frame); WHICH scale is irrelevant to
    # it. Hardcoding one couples the test to the registry's lifecycle -- scales are retired
    # and re-minted by unrelated work (#257 retired _v1, #282's PR mints _v6 and retires
    # _v5), and this test would then fail for a reason that has nothing to do with anchors.
    any_scale = next(iter(SCALES))
    with pytest.raises(ValueError, match="baseline"):
        score_metrics(user, None, scale=any_scale, anchor=outdir,
                      anchor_expect=_expect_for(meta))


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_a_NON_FINITE_user_value_takes_clamp_low(bad):
    """A non-finite model output takes the policy floor inherited from `score_one`."""
    got = _col(["expr_mae"], {"expr_mae": bad}, {"expr_mae": 0.50},
               _anchor_frame(expr_mae=0.10))
    assert got["expr_mae"] == -6.0


def test_a_NON_FINITE_baseline_or_anchor_is_nulled_too():
    """Bad scale ends are omitted for non-decisive metrics and raise for decisive ones."""
    from cell_eval2.score import _replicate_entries

    with pytest.raises(ValueError, match="expr_mae"):
        _replicate_entries({"expr_mae": float("nan")}, _anchor_frame(expr_mae=0.10))
    with pytest.raises(ValueError, match="expr_mae"):
        _replicate_entries({"expr_mae": 0.50}, _anchor_frame(expr_mae=float("nan")))

    entries = _replicate_entries({"de_wilcoxon_direction_yield": float("nan")},
                                 _anchor_frame(de_wilcoxon_direction_yield=0.10))
    assert "de_wilcoxon_direction_yield" not in entries
    entries = _replicate_entries({"de_wilcoxon_direction_yield": 0.50},
                                 _anchor_frame(de_wilcoxon_direction_yield=float("nan")))
    assert "de_wilcoxon_direction_yield" not in entries


def test_supplied_wins_WITHOUT_EVER_TOUCHING_the_cache(synthetic_pair_with_effect,
                                                       tmp_path, monkeypatch):
    """'Supplied wins' has to mean the cache is never OPENED. Eagerly evaluating the cached
    bundle let an inaccessible cache root abort scoring that had a perfectly good artifact
    in hand (codex checkpoint-2 P1).

    Driven by a CacheStore that raises the moment it is constructed, so a regression is a
    hard failure rather than a silent extra read."""
    import cell_eval2.cache as cache_mod
    from cell_eval2.anchor import read_anchor
    from cell_eval2.score import score_metrics

    _pred, real = synthetic_pair_with_effect
    outdir = build_anchor_dir(real, tmp_path)
    frame, _s, meta = read_anchor(outdir)
    user, base = _score_inputs(meta, frame)

    def boom(*a, **k):
        raise AssertionError("the cached door was opened despite a supplied anchor")

    monkeypatch.setattr(cache_mod, "CacheStore", boom)
    descriptor = {"root": str(tmp_path / "nope"), "key": "replicate_anchor",
                  "kind": "json", "fingerprint": "x", "params": {"p": 1}}
    out = score_metrics(user, base, anchor=outdir, anchor_cache=descriptor,
                        anchor_expect=_expect_for(meta))
    assert set(out["anchor_source"].drop_nulls().to_list()) == {"supplied"}


def _de_features(features=("g0", "g1"), lfc=3.0):
    """A supplied real-side DE table over conftest's GENE1..3 perturbations. Same shape as
    tests/test_baseline_build.py's helper, duplicated rather than imported across modules."""
    return pl.DataFrame([
        {"target": t, "feature": f, "log2_fold_change": lfc, "p_adj": 0.001}
        for t in ("GENE1", "GENE2", "GENE3") for f in features
    ])


def test_build_run_meta_OMITS_the_anchor_identity_when_no_engine_can_be_resolved(
        synthetic_pair_with_effect, monkeypatch):
    """An ORDINARY run that supplies both DE tables needs no engine, and `_de_backend_used`
    is written so it never resolves `backend="auto"` -- which would demand an installed
    backend and raise on a CUDA host without gpudge. The anchor identity must not
    reintroduce that failure on a supported path (codex checkpoint-2 P1).

    The SUPPORTED path is exercised, not a proxy for it: both DE tables are supplied, and
    the poison goes into `de_compute._resolve_backend` -- the one function that actually
    fails on a backend-free host -- rather than into `anchor._cache_backend`. Patching the
    anchor's binding alone would leave `baseline._cache_backend` live, so on a genuinely
    backend-free host `build_run_meta` could fail EARLIER, outside the new handler, and the
    test would not have covered the path it names.

    The run must still SUCCEED, both fields must be absent TOGETHER, and `score --anchor`
    against it must then fail CLOSED."""
    import cell_eval2.de_compute as de_compute
    from cell_eval2 import EvalConfig
    from cell_eval2.baseline import build_run_meta
    from cell_eval2.score import expect_from_run_meta

    pred, real = synthetic_pair_with_effect

    def poisoned(backend):
        raise RuntimeError("no DE backend is installed on this host")

    monkeypatch.setattr(de_compute, "_resolve_backend", poisoned)
    de = _de_features()
    cfg = EvalConfig(metrics=["expr_mae", "de_wilcoxon_overlap"], pert_col="target",
                     input_type="lognorm", validate_input=False)
    meta = build_run_meta(cfg, real, pred, de_real=de, de_pred=de)   # must NOT raise
    # BOTH absent, checked directly. `"expr_mae" not in str(...)` would pass on a partially
    # populated meta, which is exactly the contract the atomic update exists to hold.
    assert "anchor_semantic_identity" not in meta
    assert "anchor_metric_names" not in meta
    with pytest.raises(ValueError, match="anchor_semantic_identity"):
        expect_from_run_meta(meta)                  # fail-closed downstream


def test_a_programming_error_in_the_anchor_identity_is_NOT_swallowed(
        synthetic_pair_with_effect, monkeypatch):
    """`build_run_meta` tolerates ONE failure -- an unresolvable DE backend -- and nothing
    else. A broad `except Exception` there would turn a genuine bug in the identity
    computation into a silent omission that only surfaces much later, at score time
    (codex checkpoint-2 P2)."""
    import cell_eval2.baseline as baseline_mod
    from cell_eval2 import EvalConfig
    from cell_eval2.baseline import build_run_meta

    pred, real = synthetic_pair_with_effect

    def boom(*a, **k):
        raise AttributeError("a genuine bug, not a missing backend")

    monkeypatch.setattr(baseline_mod, "metric_output_names", boom)
    cfg = EvalConfig(metrics=["expr_mae"], pert_col="target", input_type="lognorm",
                     validate_input=False)
    with pytest.raises(AttributeError, match="a genuine bug"):
        build_run_meta(cfg, real, pred)


def test_a_TRUNCATED_cache_descriptor_is_named_not_a_KeyError(synthetic_pair_with_effect,
                                                              tmp_path):
    """The descriptor comes from `run_meta.json`, so it can be hand-edited or written by an
    older version. A missing key must say WHICH one rather than surfacing as a KeyError with
    a stack trace pointing at the cache layer (Copilot, PR #284)."""
    from cell_eval2.anchor import read_anchor
    from cell_eval2.score import score_metrics

    _pred, real = synthetic_pair_with_effect
    outdir = build_anchor_dir(real, tmp_path)
    frame, _s, meta = read_anchor(outdir)
    user, base = _score_inputs(meta, frame)
    truncated = {"root": str(tmp_path / "cache"), "key": "replicate_anchor"}  # no fp/params
    with pytest.raises(ValueError, match="fingerprint.*params|params.*fingerprint"):
        score_metrics(user, base, anchor=None, anchor_cache=truncated,
                      anchor_expect=_expect_for(meta))


def test_a_CORRUPT_cached_anchor_warns_before_it_becomes_no_anchor_available(
        synthetic_pair_with_effect, tmp_path, caplog):
    """Returning None silently turns cache CORRUPTION into the generic "no anchor available",
    which sends the reader looking for a missing file rather than a broken one (Copilot,
    PR #284). `score` cannot repair it -- it has no AnnData to recompute from -- so it must at
    least name it."""
    import glob
    import json
    import logging
    import os

    import cell_eval2.anchor as anchor_mod
    from cell_eval2.cache import CacheStore
    from cell_eval2.run import _resolve_config
    from cell_eval2.score import score_metrics

    _pred, real = synthetic_pair_with_effect
    cfg = _resolve_config(EvalConfig(**PROFILE_KW), {})
    root = str(tmp_path / "cache")
    store = CacheStore(root)
    anchor_mod.cached_anchor(real, cfg, store=store, base_seed=0, n_splits=2)

    artifacts = [p for p in glob.glob(os.path.join(root, "*.json"))
                 if os.path.basename(p) != "manifest.json"]
    assert len(artifacts) == 1
    open(artifacts[0], "w").write(json.dumps({"not": "a bundle"}))

    outdir = build_anchor_dir(real, tmp_path / "anchordir")
    frame, _s, meta = anchor_mod.read_anchor(outdir)
    user, base = _score_inputs(meta, frame)
    names = list(anchor_mod.resolve_metrics(cfg.metrics, version=cfg.version)[0])
    descriptor = {
        "root": root, "key": anchor_mod.ANCHOR_CACHE_KEY, "kind": "json",
        "fingerprint": anchor_mod.fingerprint_adata(real, pert_col=cfg.pert_col,
                                                    strict=True),
        "params": anchor_mod.anchor_cache_params(cfg, real, names, base_seed=0, n_splits=2,
                                                 metrics=list(meta["metric_names"])),
    }
    with caplog.at_level(logging.WARNING):
        with pytest.raises(ValueError, match="no anchor"):
            score_metrics(user, base, anchor=None, anchor_cache=descriptor,
                          anchor_expect=_expect_for(meta))
    assert "unusable" in caplog.text and root in caplog.text
