import os

import polars as pl
import pytest

from cell_eval2 import EvalConfig

PROFILE_KW = dict(metrics="anndata", pert_col="target", input_type="lognorm",
                  validate_input=False)


def _cfg(**kw):
    from cell_eval2.run import _resolve_config
    return _resolve_config(EvalConfig(**{**PROFILE_KW, **kw}), {})


def _names(resolved):
    from cell_eval2.catalog import resolve_metrics
    return list(resolve_metrics(resolved.metrics, version=resolved.version)[0])


def test_anchor_cache_params_carry_the_production_parameters(synthetic_pair_with_effect):
    """A key narrower than the value's true dependencies is a false hit that ships another
    configuration's anchor into a competition score.

    This asserts only the PRODUCTION half. The semantic half is covered by the mutation
    tests in tests/test_anchor_artifact.py -- "every key `anchor_semantic_params` returns is
    also in the cache params" would prove the two functions agree with each other, not that
    either covers the anchor's real dependencies, which is the failure #10 actually was."""
    from cell_eval2.anchor import anchor_cache_params

    _pred, real = synthetic_pair_with_effect
    cfg = _cfg()
    p = anchor_cache_params(cfg, real, _names(cfg), base_seed=0, n_splits=5,
                            metrics=["expr_mae"])
    for key in ("base_seed", "n_splits", "seed_derivation", "metric_names",
                "cell_eval2_version", "validate_input", "bulk_target_sum", "comparator"):
        assert key in p, f"anchor cache params omit {key!r}: {sorted(p)}"


@pytest.mark.parametrize("differ", [
    dict(base_seed=1), dict(n_splits=3), dict(metrics=["expr_mse"]),
])
def test_anchor_cache_params_move_with_every_production_parameter(
        synthetic_pair_with_effect, differ):
    from cell_eval2.anchor import anchor_cache_params

    _pred, real = synthetic_pair_with_effect
    cfg = _cfg()
    common = dict(base_seed=0, n_splits=5, metrics=["expr_mae"])
    a = anchor_cache_params(cfg, real, _names(cfg), **common)
    b = anchor_cache_params(cfg, real, _names(cfg), **{**common, **differ})
    assert a != b


def test_a_parseable_but_MALFORMED_cache_entry_misses_instead_of_raising(
        synthetic_pair_with_effect, tmp_path):
    """`CacheStore.get` catches read failures and returns MISS (cache.py:383), but that catch
    only spans the LOAD. Decoding the bundle happens after, so a JSON object that parses but
    lacks "anchor" would abort a scoring run on a corrupt cache entry rather than recompute.
    A cache is an optimization; a corrupt one must never be fatal."""
    import cell_eval2.anchor as anchor_mod
    from cell_eval2.cache import CacheStore

    _pred, real = synthetic_pair_with_effect
    cfg = _cfg()
    store = CacheStore(str(tmp_path / "cache"))
    anchor_mod.cached_anchor(real, cfg, store=store, base_seed=0, n_splits=2)

    # Overwrite the stored ARTIFACT with valid JSON that is not a bundle. Filter the
    # manifest BEFORE unpacking: a populated cache always contains manifest.json
    # (cache.py:123) alongside the artifact, so a bare `(path,) = glob(...)` raises here.
    import glob
    import json
    artifacts = [p for p in glob.glob(str(tmp_path / "cache" / "*.json"))
                 if os.path.basename(p) != "manifest.json"]
    assert len(artifacts) == 1, f"expected one cached artifact, got {artifacts}"
    open(artifacts[0], "w").write(json.dumps({"not": "a bundle"}))

    frame, _splits, meta = anchor_mod.cached_anchor(real, cfg, store=store, base_seed=0,
                                                    n_splits=2)
    assert frame.height > 0 and meta["metric_names"]


def test_a_bundle_with_a_NON_DICT_meta_MISSES_RECOMPUTES_and_REPLACES(
        synthetic_pair_with_effect, tmp_path, monkeypatch):
    """Frames that decode plus `meta=[]` escape a keys-only check and crash inside
    validation instead.

    Asserts the whole remedy, not just the return type: a decoder that coerced `[]` to `{}`
    would satisfy "meta is a dict" while having recomputed nothing. Count the closure, prove
    the entry was REPLACED (the next call is warm), and check the meta actually carries the
    gate's fields."""
    import glob
    import json

    import cell_eval2.anchor as anchor_mod
    from cell_eval2.cache import CacheStore

    _pred, real = synthetic_pair_with_effect
    cfg = _cfg()
    store = CacheStore(str(tmp_path / "cache"))
    anchor_mod.cached_anchor(real, cfg, store=store, base_seed=0, n_splits=2)

    artifacts = [p for p in glob.glob(str(tmp_path / "cache" / "*.json"))
                 if os.path.basename(p) != "manifest.json"]
    obj = json.loads(open(artifacts[0]).read())
    obj["meta"] = []
    open(artifacts[0], "w").write(json.dumps(obj))

    calls = {"n": 0}
    inner = anchor_mod.compute_replicate_anchor

    def counted(*a, **k):
        calls["n"] += 1
        return inner(*a, **k)

    monkeypatch.setattr(anchor_mod, "compute_replicate_anchor", counted)
    frame, _splits, meta = anchor_mod.cached_anchor(real, cfg, store=store, base_seed=0,
                                                    n_splits=2)
    assert calls["n"] == 1, "the corrupt entry was accepted instead of recomputed"
    assert isinstance(meta, dict) and frame.height > 0
    for field in ("real_fingerprint", "semantic_identity", "cell_eval2_version",
                  "metric_names"):
        assert meta.get(field), f"recomputed meta is missing {field!r}"

    anchor_mod.cached_anchor(real, cfg, store=store, base_seed=0, n_splits=2)
    assert calls["n"] == 1, "the recomputed bundle was not written back over the bad entry"


def test_cold_warm_and_param_miss_count_the_COMPUTE_CLOSURE(synthetic_pair_with_effect,
                                                            tmp_path, monkeypatch):
    """Spec 6.8. Counts invocations of the closure through a REAL CacheStore -- comparing
    two params dicts proves the dicts differ, not that the cache ever served anything."""
    import cell_eval2.anchor as anchor_mod
    from cell_eval2.cache import CacheStore

    _pred, real = synthetic_pair_with_effect
    calls = {"n": 0}
    inner = anchor_mod.compute_replicate_anchor

    def counted(*a, **k):
        calls["n"] += 1
        return inner(*a, **k)

    monkeypatch.setattr(anchor_mod, "compute_replicate_anchor", counted)
    store = CacheStore(str(tmp_path / "cache"))
    cfg = _cfg()

    a, _s, _m = anchor_mod.cached_anchor(real, cfg, store=store, base_seed=0, n_splits=2)
    assert calls["n"] == 1, "cold run did not compute"

    b, _s, _m = anchor_mod.cached_anchor(real, cfg, store=store, base_seed=0, n_splits=2)
    assert calls["n"] == 1, "warm run recomputed instead of hitting the cache"
    assert b.sort("metric").equals(a.sort("metric"))
    assert b.schema == a.schema, "the cached round-trip changed a dtype"

    anchor_mod.cached_anchor(real, cfg, store=store, base_seed=1, n_splits=2)
    assert calls["n"] == 2, "a different base seed served the cached anchor"

    anchor_mod.cached_anchor(real, _cfg(bulk_target_sum=1e6), store=store, base_seed=0,
                             n_splits=2)
    assert calls["n"] == 3, "a different bulk_target_sum served the cached anchor"


def test_cached_anchor_returns_a_validatable_bundle(synthetic_pair_with_effect, tmp_path):
    """`get_or_compute` returns the VALUE, not a directory -- so the cached door must get
    back a frame, its splits and its meta, and that meta must pass the same guard the
    supplied door runs."""
    import cell_eval2.anchor as anchor_mod
    from cell_eval2.cache import CacheStore

    _pred, real = synthetic_pair_with_effect
    cfg = _cfg()
    store = CacheStore(str(tmp_path / "cache"))
    bundle = anchor_mod.cached_anchor(real, cfg, store=store, base_seed=0, n_splits=2)
    frame, splits, meta = bundle
    expect = anchor_mod.AnchorExpect(
        fingerprint=meta["real_fingerprint"],
        semantic_identity=anchor_mod.semantic_identity(cfg, real, _names(cfg)),
        version=meta["cell_eval2_version"], metrics=tuple(meta["metric_names"]))
    # the bundle goes through the DOOR, in the shape the cache produces
    got, _m, source = anchor_mod.resolve_anchor(expect, cached=bundle)
    assert source == "cached"
    assert got.equals(frame)
    assert set(splits.columns) == {"split_index", "seed", "metric", "value",
                                   "n_perturbations"}
    assert splits.schema["n_perturbations"] == pl.Int64
