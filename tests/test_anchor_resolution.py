import polars as pl
import pytest

from cell_eval2 import EvalConfig

PROFILE_KW = dict(metrics="anndata", pert_col="target", input_type="lognorm",
                  validate_input=False)


def build_anchor_dir(real, outdir, **cfg_kw):
    """Local helper. Same shape as tests/test_anchor_artifact.py's -- duplicated on purpose
    rather than shared through a new module or a conftest import."""
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


def _expect(outdir, **override):
    """The expectation object, built the way a PRODUCER builds it (strict fingerprint)."""
    from dataclasses import replace as _replace

    from cell_eval2.anchor import AnchorExpect, read_anchor

    frame, _splits, meta = read_anchor(outdir)
    exp = AnchorExpect(fingerprint=meta["real_fingerprint"],
                       semantic_identity=meta["semantic_identity"],
                       version=meta["cell_eval2_version"],
                       metrics=tuple(meta["metric_names"]))
    return frame, (_replace(exp, **override) if override else exp)


def _corrupt(outdir, fn):
    """Rewrite the anchor parquet in place with `fn(frame)`."""
    path = f"{outdir}/anchor_agg.parquet"
    fn(pl.read_parquet(path)).write_parquet(path)
    return outdir


def _corrupt_meta(outdir, fields):
    """`fields` is a dict of overrides (None => delete the key), or the sentinel string
    "__replace_with_a_list__" to write a sidecar that is not an object at all."""
    import json
    path = f"{outdir}/anchor_meta.json"
    if fields == "__replace_with_a_list__":
        open(path, "w").write(json.dumps(["not", "an", "object"]))
        return outdir
    meta = json.loads(open(path).read())
    meta.update(fields)
    for k, v in list(fields.items()):
        if v is None:
            meta.pop(k, None)
    open(path, "w").write(json.dumps(meta))
    return outdir


def test_validate_anchor_refuses_a_shared_control_anchor(synthetic_pair_with_effect,
                                                         tmp_path):
    """A correlated-halves anchor is measurably optimistic (0.5-2.3% on lfc_nmae) and would
    inflate the TOP of the competition scale with no other signal.

    Built with this file's real producer helper and read back off disk -- a hand-made meta
    dict would agree with whatever the validator expects."""
    from cell_eval2.anchor import read_anchor, validate_anchor

    _pred, real = synthetic_pair_with_effect
    outdir = build_anchor_dir(real, tmp_path / "a")
    frame, expect = _expect(outdir)
    _f, _s, meta = read_anchor(outdir)
    validate_anchor(frame, meta, expect, source="supplied")          # the happy path first
    with pytest.raises(ValueError, match="control_source_effective"):
        validate_anchor(frame, {**meta, "control_source_effective": "real"}, expect,
                        source="supplied")


# (pattern, expectation override, frame corruption, meta corruption)
REJECTIONS = [
    ("fingerprint", dict(fingerprint="not-this-dataset"), None, None),
    ("semantic_identity", dict(semantic_identity="not-this-config"), None, None),
    ("cell_eval2_version", dict(version="0.0.0-not-this"), None, None),
    ("duplicate", {}, lambda f: pl.concat([f, f.head(1)]), None),
    ("missing", {}, lambda f: f.head(f.height - 1), None),
    ("unexpected|extra", {}, lambda f: pl.concat(
        [f, f.head(1).with_columns(metric=pl.lit("not_a_metric"))]), None),
    ("empty", {}, lambda f: f.head(0), None),
    ("dtype", {}, lambda f: f.with_columns(pl.col("replicate").cast(pl.Float32)), None),
    ("non-finite", {}, lambda f: f.with_columns(
        replicate=pl.lit(float("nan"), dtype=pl.Float64)), None),
    ("estimator", {}, lambda f: f.with_columns(
        estimator=pl.lit("hand_edited", dtype=pl.Utf8)), None),
    ("metric_names", {}, None, dict(metric_names=["not_the_frames_metrics"])),
    # gate field absent. SUPPLIED-DOOR ONLY -- see the skip below.
    ("fingerprint", {}, None, dict(real_fingerprint=None)),
    # A sidecar that is not an object at all. `validate_anchor` reaches for `meta.get`, so
    # without `read_anchor`'s type check this dies with an AttributeError naming the
    # validator instead of the source.
    ("sidecar|object", {}, None, "__replace_with_a_list__"),
]


@pytest.mark.parametrize("door", ["supplied", "cached"])
@pytest.mark.parametrize("pattern,override,corrupt,meta_corrupt", REJECTIONS,
                         ids=[f"{i}-{r[0]}" for i, r in enumerate(REJECTIONS)])
def test_every_rejection_runs_on_BOTH_doors(synthetic_pair_with_effect, tmp_path, door,
                                            pattern, override, corrupt, meta_corrupt):
    """Spec 6.4. Exercised through `resolve_anchor`, not through `validate_anchor` directly:
    a direct call proves the guard works, not that the door CALLS it -- and the cached door
    silently skipping validation is the exact bug this shape exists to catch.

    The CACHED door is driven in its REAL shape (the three-part in-memory bundle
    `cached_anchor` returns), not as a directory. Passing a directory through the cached
    label would leave the shape the cache actually produces untested."""
    from cell_eval2.anchor import read_anchor, resolve_anchor

    _pred, real = synthetic_pair_with_effect
    outdir = build_anchor_dir(real, tmp_path / "a")
    _f, expect = _expect(outdir, **override)
    if corrupt is not None:
        _corrupt(outdir, corrupt)
    if meta_corrupt is not None:
        _corrupt_meta(outdir, meta_corrupt)

    # The cached door is driven in the shape the cache produces. A sidecar that is not an
    # object, or that is missing one of `_REQUIRED_META`, has no cached analogue --
    # `_bundle_from_obj` turns both into a MISS, covered in Task 8 -- so they are
    # supplied-door cases only. MEASURED, not assumed: driving the missing-field case
    # through the cached door fails in SETUP, because `read_anchor` (which builds the
    # in-memory bundle) refuses it before `resolve_anchor` is ever reached. That is the
    # correct boundary, but it makes the case unrunnable on that door.
    _meta_shape_case = (meta_corrupt == "__replace_with_a_list__"
                        or (isinstance(meta_corrupt, dict)
                            and any(v is None for v in meta_corrupt.values())))
    if _meta_shape_case and door == "cached":
        pytest.skip("a malformed sidecar is a cache MISS, not a rejection (Task 8)")
    src = outdir if door == "supplied" else read_anchor(outdir)   # (frame, splits, meta)
    with pytest.raises(ValueError, match=pattern):
        resolve_anchor(expect, **{door: src})


def test_the_lfc_nmae_estimator_label_is_enforced_not_just_typed(graded_counts_real,
                                                                 tmp_path):
    """Spec 4.2 makes `estimator` the artifact's own record of the settled raw/full-gate
    decision. An anchor claiming `split_half_raw` for de_wilcoxon_lfc_nmae is asserting it
    was built by the estimator section 3.2 rules out -- on a real panel that is a 21-35%
    cohort mismatch, and `score` sees only the two scalars."""
    from cell_eval2.anchor import SPLIT_HALF_RAW, resolve_anchor

    outdir = build_anchor_dir(graded_counts_real, tmp_path,
                              metrics=["de_wilcoxon_lfc_nmae", "de_wilcoxon_overlap"],
                              input_type="counts", validate_input=True,
                              device="cpu", de={"backend": "pdex"})
    _f, expect = _expect(outdir)
    _corrupt(outdir, lambda f: f.with_columns(
        estimator=pl.when(pl.col("metric") == "de_wilcoxon_lfc_nmae")
        .then(pl.lit(SPLIT_HALF_RAW)).otherwise(pl.col("estimator"))))
    with pytest.raises(ValueError, match="full_gate_raw|estimator"):
        resolve_anchor(expect, supplied=outdir)


def test_supplied_wins_over_cached_and_the_VALUES_prove_it(synthetic_pair_with_effect,
                                                           tmp_path):
    """The two artifacts are made OBSERVABLY different, so the assertion is about which one
    was returned -- not only about which label came back. Identical artifacts would let a
    door that returned the wrong frame pass on the label alone."""
    from cell_eval2.anchor import read_anchor, resolve_anchor

    _pred, real = synthetic_pair_with_effect
    a = build_anchor_dir(real, tmp_path / "supplied")
    b = build_anchor_dir(real, tmp_path / "cached")
    _corrupt(b, lambda f: f.with_columns(replicate=pl.col("replicate") + 0.25))

    want, expect = _expect(a)
    got, _m, source = resolve_anchor(expect, supplied=a, cached=read_anchor(b))
    assert source == "supplied"
    assert got["replicate"].to_list() == pytest.approx(want["replicate"].to_list())


def test_neither_door_raises_and_never_recomputes(synthetic_pair_with_effect, tmp_path):
    """Spec 6.5. Silently recomputing at score time would derive the anchor from whatever
    data is at hand -- the plausible-wrong-number shape this scheme exists to close."""
    from cell_eval2.anchor import resolve_anchor

    _pred, real = synthetic_pair_with_effect
    outdir = build_anchor_dir(real, tmp_path)
    _frame, expect = _expect(outdir)
    with pytest.raises(ValueError, match="no anchor"):
        resolve_anchor(expect, supplied=None, cached=None)


@pytest.mark.parametrize("door", ["supplied", "cached"])
def test_a_valid_anchor_passes_both_doors(synthetic_pair_with_effect, tmp_path, door):
    """Guards the guard: if a correct artifact were rejected, every raise above would pass
    for the wrong reason."""
    from cell_eval2.anchor import read_anchor, resolve_anchor

    _pred, real = synthetic_pair_with_effect
    outdir = build_anchor_dir(real, tmp_path)
    frame, expect = _expect(outdir)
    src = outdir if door == "supplied" else read_anchor(outdir)
    got, meta, source = resolve_anchor(expect, **{door: src})
    assert source == door
    assert got.equals(frame)
    assert meta["semantic_identity"] == expect.semantic_identity


def test_the_METADATA_hash_is_NEVER_accepted_as_the_gate(synthetic_pair_with_effect,
                                                         tmp_path):
    """The gate is `real_fingerprint`, the strict content hash, and nothing else.

    An earlier draft let the expectation name a STRENGTH and compared whichever field
    matched -- which, because `build_run_meta` is metadata-only by default
    (baseline.py:793), made the weak hash the DEFAULT gate. The metadata hash is stamped in
    the sidecar as provenance; offering it as the expectation must be rejected."""
    import json
    from dataclasses import replace as _replace

    from cell_eval2.anchor import resolve_anchor

    _pred, real = synthetic_pair_with_effect
    outdir = build_anchor_dir(real, tmp_path)
    meta = json.loads(open(f"{outdir}/anchor_meta.json").read())
    _f, expect = _expect(outdir)

    loose = _replace(expect, fingerprint=meta["real_fingerprint_meta"])
    with pytest.raises(ValueError, match="fingerprint"):
        resolve_anchor(loose, supplied=outdir)


def test_anchor_digest_moves_with_the_values(synthetic_pair_with_effect, tmp_path):
    """The digest `score` stamps must be a function of the NUMBERS, not only of the meta --
    a digest over provenance alone cannot tell two anchors of the same dataset apart."""
    from cell_eval2.anchor import anchor_digest, read_anchor

    _pred, real = synthetic_pair_with_effect
    outdir = build_anchor_dir(real, tmp_path)
    frame, _splits, meta = read_anchor(outdir)
    moved = frame.with_columns(replicate=pl.col("replicate") + 1.0)
    assert anchor_digest(frame, meta) != anchor_digest(moved, meta)
    assert anchor_digest(frame, meta) == anchor_digest(frame, meta)   # stable
