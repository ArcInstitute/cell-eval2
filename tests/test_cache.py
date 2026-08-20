from dataclasses import replace          # REQUIRED: the swap below uses it

import anndata as ad
import json
import numpy as np
import pandas as pd
import polars as pl
import pytest

from cell_eval2.cache import CACHE_FORMAT_VERSION, MISS, CacheStore, fingerprint_adata, fingerprint_de_table
from cell_eval2.cache import _dump_npz_moments, _load_npz_moments
from cell_eval2.catalog import CATALOG, DerivedAgg, derived_policy
from cell_eval2 import EvalConfig, compute_metrics
from cell_eval2.moments import GroupMoments


def _adata(seed=0, scale=1.0, n_genes=6, n_cells=12):
    rng = np.random.default_rng(seed)
    X = np.log1p(rng.gamma(1.0, scale, size=(n_cells, n_genes))).astype(np.float64)
    labels = (["non-targeting", "GENE1", "GENE2"] * n_cells)[:n_cells]
    obs = pd.DataFrame({"target": labels}, index=[f"c{i}" for i in range(n_cells)])
    var = pd.DataFrame(index=[f"g{j}" for j in range(n_genes)])
    return ad.AnnData(X=X, obs=obs, var=var)


def test_fingerprint_adata_stable_and_structural():
    a = _adata()
    assert fingerprint_adata(a, pert_col="target") == fingerprint_adata(a, pert_col="target")
    # different gene names -> different fingerprint
    b = a.copy()
    b.var.index = [f"x{j}" for j in range(b.n_vars)]
    assert fingerprint_adata(b, pert_col="target") != fingerprint_adata(a, pert_col="target")


def test_fingerprint_adata_structural_ignores_values_strict_catches():
    a = _adata()
    c = a.copy()
    c.X = c.X * 2.0  # same shape/dtype/var/value-counts, different values
    assert fingerprint_adata(c, pert_col="target") == fingerprint_adata(a, pert_col="target")
    assert (fingerprint_adata(c, pert_col="target", strict=True)
            != fingerprint_adata(a, pert_col="target", strict=True))


def test_fingerprint_de_table_structural_and_strict():
    df = pl.DataFrame({"target": ["GENE1", "GENE2"], "feature": ["g0", "g1"],
                       "log2_fold_change": [1.0, -2.0], "p_adj": [0.01, 0.2]})
    assert fingerprint_de_table(df) == fingerprint_de_table(df)
    df2 = df.with_columns(pl.col("log2_fold_change") * 3)  # value-only change
    assert fingerprint_de_table(df2) == fingerprint_de_table(df)          # structural: same
    assert fingerprint_de_table(df2, strict=True) != fingerprint_de_table(df, strict=True)


def test_cachestore_npz_cold_then_warm(tmp_path):
    root = str(tmp_path / "real")
    calls = []

    def compute():
        calls.append(1)
        return (np.array(["GENE1", "GENE2"], dtype=str), np.arange(6.0).reshape(2, 3))

    s1 = CacheStore(root)
    perts, means = s1.get_or_compute("pseudobulk_lognorm", fingerprint="fp1",
                                     params={"pert_col": "target"}, kind="npz", compute=compute)
    assert list(perts) == ["GENE1", "GENE2"]
    assert (tmp_path / "real" / "manifest.json").exists()
    assert list((tmp_path / "real").glob("pseudobulk_lognorm*.npz"))  # content-addressed filename

    s2 = CacheStore(root)  # fresh store, warm cache
    perts2, means2 = s2.get_or_compute("pseudobulk_lognorm", fingerprint="fp1",
                                       params={"pert_col": "target"}, kind="npz", compute=compute)
    np.testing.assert_array_equal(means, means2)
    assert calls == [1]  # compute ran exactly once


def test_cachestore_fingerprint_mismatch_recomputes(tmp_path):
    root = str(tmp_path / "real")
    s = CacheStore(root)
    s.put("pseudobulk_lognorm", (np.array(["A"], dtype=str), np.ones((1, 2))),
          fingerprint="fp1", params={"pert_col": "target"}, kind="npz")
    assert s.get("pseudobulk_lognorm", fingerprint="fp1", params={"pert_col": "target"},
                 kind="npz") is not MISS
    assert s.get("pseudobulk_lognorm", fingerprint="DIFFERENT", params={"pert_col": "target"},
                 kind="npz") is MISS
    assert s.get("pseudobulk_lognorm", fingerprint="fp1", params={"pert_col": "OTHER"},
                 kind="npz") is MISS


def test_cachestore_corrupt_artifact_and_manifest_recompute(tmp_path):
    root = tmp_path / "real"
    s = CacheStore(str(root))
    s.put("p", (np.array(["A"], dtype=str), np.ones((1, 2))), fingerprint="fp",
          params={}, kind="npz")
    fn = json.loads((root / "manifest.json").read_text())["artifacts"]["p"]["filename"]
    (root / fn).write_bytes(b"garbage")                    # corrupt the manifest-referenced artifact
    assert s.get("p", fingerprint="fp", params={}, kind="npz") is MISS
    (root / "manifest.json").write_text("{ not json")      # corrupt manifest
    s2 = CacheStore(str(root))                              # must not raise
    assert s2.get("p", fingerprint="fp", params={}, kind="npz") is MISS


def test_cachestore_version_bump_invalidates(tmp_path):
    root = tmp_path / "real"
    s = CacheStore(str(root))
    s.put("p", (np.array(["A"], dtype=str), np.ones((1, 2))), fingerprint="fp",
          params={}, kind="npz")
    data = json.loads((root / "manifest.json").read_text())
    data["cache_format_version"] = CACHE_FORMAT_VERSION + 99
    (root / "manifest.json").write_text(json.dumps(data))
    s2 = CacheStore(str(root))
    assert s2.get("p", fingerprint="fp", params={}, kind="npz") is MISS


def test_cachestore_parquet_roundtrip_and_no_tmp(tmp_path):
    root = tmp_path / "pred"
    s = CacheStore(str(root))
    df = pl.DataFrame({"perturbation": ["GENE1"], "metric": ["expr_mae"], "value": [0.5]})
    got = s.get_or_compute("results", fingerprint="fp", params={}, kind="parquet",
                           compute=lambda: df)
    assert got.equals(df)
    s2 = CacheStore(str(root))
    assert s2.get("results", fingerprint="fp", params={}, kind="parquet").equals(df)
    assert not list(root.glob(".*tmp*"))  # no stray temp files


def test_cachestore_malformed_entry_and_npz_are_misses(tmp_path):
    root = tmp_path / "real"
    s = CacheStore(str(root))
    s.put("p", (np.array(["A"], dtype=str), np.ones((1, 2))), fingerprint="fp",
          params={}, kind="npz")
    # manifest entry missing 'filename' -> miss, NOT KeyError
    m = json.loads((root / "manifest.json").read_text())
    del m["artifacts"]["p"]["filename"]
    (root / "manifest.json").write_text(json.dumps(m))
    assert CacheStore(str(root)).get("p", fingerprint="fp", params={}, kind="npz") is MISS
    # parseable npz with mismatched shapes (perts len 2 vs means rows 1) -> miss
    s2 = CacheStore(str(root))
    s2.put("q", (np.array(["A", "B"], dtype=str), np.ones((1, 2))), fingerprint="fp",
           params={}, kind="npz")
    assert s2.get("q", fingerprint="fp", params={}, kind="npz") is MISS


def test_cachestore_non_dict_manifest_and_entry_dont_crash(tmp_path):
    # valid JSON that isn't an object -> ignored, fresh manifest, no AttributeError
    root = tmp_path / "real"
    root.mkdir()
    (root / "manifest.json").write_text("[1, 2, 3]")
    s = CacheStore(str(root))  # must not raise
    assert s.get("p", fingerprint="fp", params={}, kind="npz") is MISS
    # a non-dict artifacts entry -> miss, not AttributeError
    s.put("p", (np.array(["A"], dtype=str), np.ones((1, 2))), fingerprint="fp",
          params={}, kind="npz")
    m = json.loads((root / "manifest.json").read_text())
    m["artifacts"]["p"] = "not-a-dict"
    (root / "manifest.json").write_text(json.dumps(m))
    assert CacheStore(str(root)).get("p", fingerprint="fp", params={}, kind="npz") is MISS


def test_cachestore_unknown_kind_raises_valueerror(tmp_path):
    s = CacheStore(str(tmp_path / "r"))
    with pytest.raises(ValueError, match="unknown cache kind"):
        s.get("p", fingerprint="fp", params={}, kind="bogus")
    with pytest.raises(ValueError, match="unknown cache kind"):
        s.put("p", None, fingerprint="fp", params={}, kind="bogus")


def test_cachestore_rejects_unsafe_key_and_tampered_filename(tmp_path):
    root = tmp_path / "real"
    s = CacheStore(str(root))
    with pytest.raises(ValueError, match="bare name"):
        s.put("../escape", (np.array(["A"], dtype=str), np.ones((1, 2))),
              fingerprint="fp", params={}, kind="npz")
    # a tampered manifest filename with a path component -> treated as a miss, no traversal read
    s.put("p", (np.array(["A"], dtype=str), np.ones((1, 2))), fingerprint="fp", params={}, kind="npz")
    m = json.loads((root / "manifest.json").read_text())
    m["artifacts"]["p"]["filename"] = "../p.npz"
    (root / "manifest.json").write_text(json.dumps(m))
    assert CacheStore(str(root)).get("p", fingerprint="fp", params={}, kind="npz") is MISS


def test_fingerprint_adata_detects_label_permutation():
    a = _adata()
    b = a.copy()
    b.obs["target"] = list(reversed(a.obs["target"].tolist()))  # same multiset, different assignment
    assert fingerprint_adata(a, pert_col="target") != fingerprint_adata(b, pert_col="target")


def test_cachestore_put_survives_write_failure(tmp_path, monkeypatch):
    import cell_eval2.cache as cache
    s = cache.CacheStore(str(tmp_path / "r"))

    def boom(*a, **k):
        raise OSError("disk full")

    monkeypatch.setattr(cache, "_atomic_write", boom)
    # a write failure must NOT raise — the result is valid, just uncached
    s.put("p", (np.array(["A"], dtype=str), np.ones((1, 2))), fingerprint="fp", params={}, kind="npz")
    assert s.get("p", fingerprint="fp", params={}, kind="npz") is MISS


def test_side_cache_params_include_validation_flags(tmp_path, synthetic_counts_pair):
    # Codex #2 + P2: the pseudobulk and DE-table side caches must key on the new validation
    # flags (and max_counts_per_cell for the DE table) so a permissive cache fill cannot be
    # reused by a stricter run that would then skip validation on the hit.
    import cell_eval2.cache as cache_mod
    pred, real = synthetic_counts_pair
    captured = []
    orig = cache_mod.CacheStore.get_or_compute

    def spy(self, key, *, fingerprint, params, kind, compute):
        captured.append((key, dict(params)))
        return orig(self, key, fingerprint=fingerprint, params=params, kind=kind, compute=compute)

    cache_mod.CacheStore.get_or_compute = spy
    try:
        compute_metrics(pred, real, config=EvalConfig(
            metrics=["mae", "de_wilcoxon_overlap"], input_type="counts",
            cache_real=str(tmp_path / "r"), cache_pred=str(tmp_path / "p"),
            autodetect_input_type=True, allow_fractional_counts=True,
            de={"backend": "scanpy"},
        ))
    finally:
        cache_mod.CacheStore.get_or_compute = orig

    pseudobulk_params = [p for k, p in captured if k.startswith("pseudobulk_")]
    de_params = [p for k, p in captured if k.startswith("de_") and k.endswith("_table")]
    assert pseudobulk_params, "expected pseudobulk cache calls"
    assert de_params, "expected DE-table cache calls"
    for p in pseudobulk_params:
        assert {"autodetect_input_type", "allow_fractional_counts", "pert_col"} <= set(p)
    # DE table depends on the reference group (control) and groupby (pert_col); both must key the
    # cache or a config change silently hits a stale table (Gemini PR #35).
    for p in de_params:
        assert {"autodetect_input_type", "allow_fractional_counts", "max_counts_per_cell",
                "control", "pert_col"} <= set(p)


def test_pseudobulk_cache_cold_warm_equivalence(tmp_path, synthetic_pair):
    pred, real = synthetic_pair
    cr, cp = str(tmp_path / "r"), str(tmp_path / "p")
    base = dict(metrics=["mae"], pert_col="target", control="non-targeting",
                input_type="lognorm")
    df_nocache = compute_metrics(pred, real, **base)
    cfg = EvalConfig(cache_real=cr, cache_pred=cp, **base)
    df_cold = compute_metrics(pred, real, config=cfg)
    assert list((tmp_path / "r").glob("pseudobulk_lognorm*.npz"))
    assert df_cold.equals(df_nocache)              # cache changes nothing about values

    import cell_eval2.run as run
    calls = []
    orig = run.pseudobulk
    run.pseudobulk = lambda *a, **k: (calls.append(1), orig(*a, **k))[1]
    try:
        df_warm = compute_metrics(pred, real, config=cfg)
    finally:
        run.pseudobulk = orig
    assert df_warm.equals(df_nocache)
    assert calls == []                              # warm: pseudobulk never recomputed


def test_pseudobulk_real_cache_reused_across_preds(tmp_path, synthetic_pair):
    pred, real = synthetic_pair
    pred2 = pred.copy()
    pred2.X = pred2.X * 1.3                          # a different prediction
    cr = str(tmp_path / "r")
    compute_metrics(pred, real, config=EvalConfig(metrics=["mae"], cache_real=cr,
                                                  cache_pred=str(tmp_path / "p1"),
                                                  input_type="lognorm"))
    import cell_eval2.run as run
    seen = []
    orig = run._materialize
    run._materialize = lambda src: seen.append(id(src)) or orig(src)
    try:
        compute_metrics(pred2, real, config=EvalConfig(metrics=["mae"], cache_real=cr,
                                                       cache_pred=str(tmp_path / "p2"),
                                                       input_type="lognorm"))
    finally:
        run._materialize = orig
    # real pseudobulk is a cache hit -> real is never materialized; only pred2 is
    assert id(real) not in seen
    assert id(pred2) in seen


def test_backed_inmemory_object_is_validated(tmp_path, synthetic_pair):
    # An already-loaded *backed* AnnData passed directly must still be validated on a
    # cache miss (it escapes the up-front check, so full() must catch it).
    import anndata as ad
    pred, real = synthetic_pair
    bad = real.copy()
    bad.X = np.rint(bad.X) + 1.0          # all-integer, but declared lognorm (default)
    rp = tmp_path / "bad.h5ad"
    bad.write_h5ad(rp)
    backed = ad.read_h5ad(str(rp), backed="r")
    with pytest.raises(ValueError, match="lognorm"):
        compute_metrics(pred, backed, metrics=["mae"], control="non-targeting",
                        input_type="lognorm")


def _de_for_synth(scale_lfc=1.0):
    rows = []
    for t in ("GENE1", "GENE2", "GENE3"):
        rows += [{"target": t, "feature": "g0", "log2_fold_change": 3.0 * scale_lfc, "p_adj": 0.001},
                 {"target": t, "feature": "g1", "log2_fold_change": -2.0 * scale_lfc, "p_adj": 0.02},
                 {"target": t, "feature": "g2", "log2_fold_change": 0.05, "p_adj": 0.8}]
    return pl.DataFrame(rows)


def test_de_rank_cache_cold_warm_equivalence(tmp_path, synthetic_pair):
    pred, real = synthetic_pair
    de = _de_for_synth()
    cr, cp = str(tmp_path / "r"), str(tmp_path / "p")
    base = dict(metrics=["overlap_at_N"], control="non-targeting", pert_col="target",
                input_type="lognorm")
    df_nocache = compute_metrics(pred, real, de_pred=de, de_real=de, **base)
    cfg = EvalConfig(cache_real=cr, cache_pred=cp, **base)
    df_cold = compute_metrics(pred, real, de_pred=de, de_real=de, config=cfg)
    assert list((tmp_path / "r").glob("de_wilcoxon_rank*.parquet"))
    assert list((tmp_path / "p").glob("de_wilcoxon_rank*.parquet"))
    assert df_cold.equals(df_nocache)

    # warm run: a second DE metric busts the RESULT cache (different metric set) but the
    # rank matrices (same DE tables + DE params) are reused, so the rank pivot never reruns.
    import cell_eval2.run as run
    calls = []
    orig = run.rank_de_side
    run.rank_de_side = lambda *a, **k: (calls.append(1), orig(*a, **k))[1]
    try:
        compute_metrics(pred, real, de_pred=de, de_real=de,
                        config=EvalConfig(metrics=["overlap_at_N", "precision_at_N"],
                                          control="non-targeting", pert_col="target",
                                          cache_real=cr, cache_pred=cp,
                                          input_type="lognorm"))
    finally:
        run.rank_de_side = orig
    assert calls == []  # both sides' rank matrices came from cache


def test_real_de_table_reused_across_preds(tmp_path, monkeypatch, synthetic_counts_pair):
    from cell_eval2 import de_compute
    from cell_eval2.config import EvalConfig
    from cell_eval2.run import compute_metrics
    pred, real = synthetic_counts_pair
    pred2 = pred.copy()
    pred2.X = pred2.X + 1            # a different pred (stays integer counts)
    cr = str(tmp_path / "real")
    base = dict(metrics=["de_wilcoxon_overlap"], input_type="counts",
                cache_real=cr, de={"backend": "scanpy"})
    # Run 1 (different cache_pred) populates the real-side DE table in cache_real.
    compute_metrics(pred, real, config=EvalConfig(cache_pred=str(tmp_path / "p1"), **base))
    # Spy on compute_de; _compute_de_side does a call-time `from .de_compute import
    # compute_de`, so patching the de_compute attribute is seen.
    calls = {"n": 0}
    orig = de_compute.compute_de

    def spy(*a, **k):
        calls["n"] += 1
        return orig(*a, **k)

    monkeypatch.setattr("cell_eval2.de_compute.compute_de", spy)
    # Run 2: a fresh cache_pred forces a result-cache miss (so the real side is actually
    # consulted), but cache_real is shared -> the real DE table is reused, only pred2 computes.
    compute_metrics(pred2, real, config=EvalConfig(cache_pred=str(tmp_path / "p2"), **base))
    assert calls["n"] == 1          # exactly one compute (pred2); real served from cache


def test_de_only_path_input_still_validated(tmp_path, synthetic_pair):
    # No-cache DE-only run with path inputs must still validate input-type, exactly
    # as before (regression guard for the §1 invariant).
    pred, real = synthetic_pair
    bad = real.copy()
    bad.X = np.rint(bad.X) + 1.0          # all-integer values, but declared lognorm (default)
    rp, pp = tmp_path / "bad_real.h5ad", tmp_path / "pred.h5ad"
    bad.write_h5ad(rp)
    pred.write_h5ad(pp)
    de = _de_for_synth()
    with pytest.raises(ValueError, match="lognorm"):
        compute_metrics(str(pp), str(rp), de_pred=de, de_real=de,
                        metrics=["overlap_at_N"], control="non-targeting",
                        input_type="lognorm")


def test_de_only_with_cache_skips_anndata_materialization(tmp_path, synthetic_pair):
    # In cache mode, a DE-only run uses only anndata metadata, so neither side is materialized.
    import cell_eval2.run as run
    pred, real = synthetic_pair
    pp, rp = tmp_path / "p.h5ad", tmp_path / "r.h5ad"
    pred.write_h5ad(pp)
    real.write_h5ad(rp)
    de = _de_for_synth()
    cfg = EvalConfig(metrics=["overlap_at_N"], control="non-targeting",
                     cache_real=str(tmp_path / "r"), cache_pred=str(tmp_path / "p"),
                     input_type="lognorm")
    seen = []
    orig = run._materialize
    run._materialize = lambda src: seen.append(src) or orig(src)
    try:
        compute_metrics(str(pp), str(rp), de_pred=de, de_real=de, config=cfg)
    finally:
        run._materialize = orig
    assert seen == []  # DE-only + cache: neither anndata side is loaded fully


def test_result_cache_short_circuits_all_compute(tmp_path, synthetic_pair):
    pred, real = synthetic_pair
    cr, cp = str(tmp_path / "r"), str(tmp_path / "p")
    cfg = EvalConfig(metrics=["mae"], cache_real=cr, cache_pred=cp, input_type="lognorm")
    df1 = compute_metrics(pred, real, config=cfg)
    assert list((tmp_path / "p").glob("results*.parquet"))

    import cell_eval2.run as run
    calls = []
    orig = run.pseudobulk
    run.pseudobulk = lambda *a, **k: (calls.append(1), orig(*a, **k))[1]
    try:
        df2 = compute_metrics(pred, real, config=cfg)
    finally:
        run.pseudobulk = orig
    assert df2.equals(df1)
    assert calls == []  # full result hit: no intermediate computation at all


def _spy(monkeypatch, name):
    """Count calls into run.<name> without changing its behaviour."""
    import cell_eval2.run as run
    calls = []
    orig = getattr(run, name)
    monkeypatch.setattr(run, name, lambda *a, **k: (calls.append(1), orig(*a, **k))[1])
    return calls


def test_write_de_keeps_the_result_cache_short_circuit(tmp_path, monkeypatch, synthetic_pair):
    # write_de asks for an extra ARTIFACT, not for different numbers, so it must not cost the
    # results hit: the DE tables come from their own caches on the cache-hit path, and the
    # metric loop -- exactly what the results cache exists to skip -- must not run again.
    pred, real = synthetic_pair
    # `metrics="full"` now includes the eleven chance-corrected direction metrics (#195),
    # which require each target to name a MEASURED gene and fail loud otherwise. The shared
    # fixture's targets are GENE1..3 while its var index is g0..g39, so nothing resolves.
    # Rename the first three features locally rather than editing the heavily shared
    # conftest fixture -- the same local-override rule #195 used for its own fixtures.
    genes = ["GENE1", "GENE2", "GENE3", *real.var_names[3:].to_list()]
    pred.var_names = genes
    real.var_names = genes
    base = dict(metrics="full", cache_real=str(tmp_path / "r"), cache_pred=str(tmp_path / "p"),
                input_type="lognorm")
    cold, warm = tmp_path / "cold", tmp_path / "warm"
    df1 = compute_metrics(pred, real, config=EvalConfig(**base, outdir=str(cold)), write_de=True)

    de_calls = _spy(monkeypatch, "dispatch_de_metrics")
    ad_calls = _spy(monkeypatch, "dispatch_anndata_metrics")
    df2 = compute_metrics(pred, real, config=EvalConfig(**base, outdir=str(warm)), write_de=True)

    assert df2.equals(df1)
    assert de_calls == [] and ad_calls == []  # results hit preserved
    # ...and the artifact is still produced, byte-for-byte what the cold run wrote.
    for name in ("de_real.parquet", "de_pred.parquet"):
        assert (warm / name).exists()
        assert pl.read_parquet(warm / name).equals(pl.read_parquet(cold / name))


def test_write_de_requires_an_outdir(tmp_path, synthetic_pair):
    # outdir=None means the API writes nothing, so write_de has nowhere to put the tables.
    # Raise up front rather than silently creating ./cell-eval2-outdir in the caller's CWD.
    pred, real = synthetic_pair
    with pytest.raises(ValueError, match="requires an output directory"):
        compute_metrics(pred, real, config=EvalConfig(metrics=["mae"], input_type="lognorm"),
                        write_de=True)
    # ...and it fails BEFORE doing the work, so nothing was written anywhere.
    assert list(tmp_path.iterdir()) == []


def test_write_de_without_de_metrics_still_short_circuits(tmp_path, monkeypatch, synthetic_pair):
    # A DE-free profile writes no DE tables at all, so write_de has nothing to materialize and
    # must not disturb the cache hit -- otherwise the flag costs a full recompute for nothing.
    pred, real = synthetic_pair
    base = dict(metrics=["mae"], cache_real=str(tmp_path / "r"), cache_pred=str(tmp_path / "p"),
                input_type="lognorm")
    compute_metrics(pred, real, config=EvalConfig(**base, outdir=str(tmp_path / "cold")))

    warm = tmp_path / "warm"
    ad_calls = _spy(monkeypatch, "dispatch_anndata_metrics")
    compute_metrics(pred, real, config=EvalConfig(**base, outdir=str(warm)), write_de=True)

    assert ad_calls == []
    assert sorted(f.name for f in warm.glob("de_*.parquet")) == []


def test_result_cache_misses_on_metric_change(tmp_path, synthetic_pair):
    pred, real = synthetic_pair
    cr, cp = str(tmp_path / "r"), str(tmp_path / "p")
    compute_metrics(
        pred, real,
        config=EvalConfig(metrics=["mae"], cache_real=cr, cache_pred=cp, input_type="lognorm"),
    )
    # a different metric set must not return the mae-only cached result
    df = compute_metrics(
        pred, real,
        config=EvalConfig(metrics=["pds_l1"], cache_real=cr, cache_pred=cp, input_type="lognorm"),
    )
    assert set(df["metric"].unique()) == {"pds_l1"}


def test_precompute_real_then_pair_run_skips_real(tmp_path, synthetic_pair):
    from cell_eval2.run import precompute_cache
    pred, real = synthetic_pair
    rp = tmp_path / "real.h5ad"
    real.write_h5ad(rp)
    cr, cp = str(tmp_path / "r"), str(tmp_path / "p")

    precompute_cache(str(rp), side="real",
                     config=EvalConfig(metrics=["mae"], cache_real=cr, cache_pred=cp,
                                       input_type="lognorm"), comparator="lognorm")
    assert list((tmp_path / "r").glob("pseudobulk_lognorm*.npz"))

    import cell_eval2.run as run
    materialized = []
    orig = run._materialize
    run._materialize = lambda src: materialized.append(src) or orig(src)
    try:
        df = compute_metrics(pred, str(rp),
                             config=EvalConfig(metrics=["mae"], cache_real=cr, cache_pred=cp,
                                               input_type="lognorm"))
    finally:
        run._materialize = orig
    assert set(df["metric"].unique()) == {"expr_mae"}
    # warm real cache: only pred is materialized; real (a cache hit) is never loaded fully
    assert [id(x) for x in materialized] == [id(pred)]


def test_precompute_requires_side_folder():
    from cell_eval2.run import precompute_cache
    with pytest.raises(ValueError, match="requires config.cache_real"):
        precompute_cache(object(), side="real", config=EvalConfig(metrics=["mae"],
                                                                  input_type="lognorm"))


def test_precompute_validates_inmemory_input(synthetic_pair, tmp_path):
    from cell_eval2.run import precompute_cache
    _, real = synthetic_pair
    bad = real.copy()
    bad.X = np.rint(bad.X) + 1.0  # all-integer values but declared lognorm (default)
    cfg = EvalConfig(metrics=["mae"], cache_real=str(tmp_path / "r"), input_type="lognorm")
    with pytest.raises(ValueError, match="lognorm"):
        precompute_cache(bad, side="real", config=cfg)


def test_concurrent_stores_do_not_clobber_manifest(tmp_path):
    """Two CacheStores sharing a dir (e.g. concurrent processes) must not drop each
    other's manifest entries: put() re-reads + merges the on-disk manifest."""
    root = str(tmp_path / "shared")
    a = CacheStore(root)
    b = CacheStore(root)  # both snapshot the (empty) manifest at construction
    val = (["p"], np.zeros((1, 2)))
    a.put("k_a", val, fingerprint="fa", params={}, kind="npz")
    b.put("k_b", val, fingerprint="fb", params={}, kind="npz")  # must NOT clobber k_a
    fresh = CacheStore(root)
    assert fresh.get("k_a", fingerprint="fa", params={}, kind="npz") is not MISS
    assert fresh.get("k_b", fingerprint="fb", params={}, kind="npz") is not MISS


def _de_features(features, lfc=3.0):
    rows = []
    for t in ("GENE1", "GENE2", "GENE3"):
        for f in features:
            rows.append({"target": t, "feature": f, "log2_fold_change": lfc, "p_adj": 0.001})
    return pl.DataFrame(rows)


def test_result_cache_keys_on_single_supplied_de_side(tmp_path, synthetic_pair):
    """A supplied de_real must enter the result-cache key even when de_pred is computed.
    Re-running with the SAME de_real hits the result cache (no recompute), but a different
    de_real must MISS and recompute rather than return the first run's stale cached result.
    Detected via a spy on _prepare_de_cached, which runs only on a result-cache miss."""
    import cell_eval2.run as run
    pred, real = synthetic_pair
    de_A = _de_features(["g0", "g1", "g2"])
    de_B = _de_features(["g3", "g4", "g5"])  # different feature multiset -> different fingerprint
    cp = str(tmp_path / "p")
    base = dict(metrics=["overlap_at_N"], control="non-targeting", pert_col="target",
                input_type="lognorm", version="v1")
    compute_metrics(pred, real, de_real=de_A, config=EvalConfig(cache_pred=cp, **base))  # populate

    def _spy_run(de_real):
        calls = []
        orig = run._prepare_de_cached
        run._prepare_de_cached = lambda *a, **k: (calls.append(1), orig(*a, **k))[1]
        try:
            compute_metrics(pred, real, de_real=de_real, config=EvalConfig(cache_pred=cp, **base))
        finally:
            run._prepare_de_cached = orig
        return calls

    assert _spy_run(de_A) == []    # same de_real -> result-cache HIT (no recompute)
    assert _spy_run(de_B) == [1]   # different de_real -> MISS -> recompute (no stale hit)


def _de_stats(g0_padj):
    # 3 targets x 3 features, identical (target, feature) STRUCTURE regardless of g0_padj; only g0's
    # significance toggles (threshold 0.05), so tables differ ONLY in a stat column.
    return pl.DataFrame([
        {"target": t, "feature": f, "log2_fold_change": lfc, "p_adj": p}
        for t in ("GENE1", "GENE2", "GENE3")
        for f, lfc, p in (("g0", 3.0, g0_padj), ("g1", -2.0, 0.02), ("g2", 0.05, 0.8))
    ])


def test_supplied_de_stats_change_busts_result_and_rank_caches(tmp_path, synthetic_pair):
    """F9.1: two SUPPLIED DE tables with identical (target, feature) structure but a different p_adj
    must not collide -- on the result cache OR the rank cache. A supplied table's stats are external
    input that nothing else in the cache key characterizes (unlike a COMPUTED table, pinned by the
    adata fingerprint + DE config + backend), so supplied tables are fingerprinted STRICTLY even at
    the default cache_strict=False. Re-running the SAME table must still HIT (no false-miss)."""
    import cell_eval2.run as run
    pred, real = synthetic_pair
    pred_de = _de_stats(0.001)  # pred side fixed: g0 significant, across every run below
    cp = str(tmp_path / "p")
    base = dict(metrics=["overlap_at_N"], control="non-targeting", pert_col="target",
                input_type="lognorm")

    def run_with(real_de):
        return compute_metrics(pred, real, de_pred=pred_de, de_real=real_de,
                               config=EvalConfig(cache_pred=cp, **base))

    def spy_with(real_de):  # _prepare_de_cached runs ONLY on a result-cache miss
        calls = []
        orig = run._prepare_de_cached
        run._prepare_de_cached = lambda *a, **k: (calls.append(1), orig(*a, **k))[1]
        try:
            df = run_with(real_de)
        finally:
            run._prepare_de_cached = orig
        return calls, df

    r_sig = run_with(_de_stats(0.001))               # g0 significant -> populate result + rank caches
    hit_calls, r_sig_warm = spy_with(_de_stats(0.001))   # identical table -> result-cache HIT
    miss_calls, r_nonsig = spy_with(_de_stats(0.9))      # g0 NOT significant -> MISS + recompute

    assert hit_calls == [], "identical supplied DE table false-missed the result cache"
    assert r_sig_warm.equals(r_sig), "warm result-cache hit returned a different frame"
    assert miss_calls == [1], "a p_adj change in a supplied DE table did not bust the result cache"
    assert not r_nonsig.equals(r_sig), \
        "different-p_adj supplied DE tables returned the same result (stale result and/or rank cache)"


def test_precompute_supplied_de_rank_reused_by_run(tmp_path, synthetic_pair):
    """F9.1 (precompute consistency): precompute_cache must write a SUPPLIED table's rank under the
    same strict fingerprint _prepare_de_cached reads with, so a later run reuses the precomputed
    rank. Only the REAL side is precomputed here, so a correct run recomputes exactly the PRED rank
    (one rank_de_side call); a second call would mean the precomputed real rank key did not match."""
    import cell_eval2.run as run
    from cell_eval2.run import precompute_cache
    pred, real = synthetic_pair
    de = _de_stats(0.001)
    cr, cp = str(tmp_path / "r"), str(tmp_path / "p")
    base = dict(metrics=["overlap_at_N"], control="non-targeting", pert_col="target",
                input_type="lognorm")
    precompute_cache(real, side="real", de=de, config=EvalConfig(cache_real=cr, **base))
    assert list((tmp_path / "r").glob("de_wilcoxon_rank*.parquet"))

    calls = []
    orig = run.rank_de_side
    run.rank_de_side = lambda *a, **k: (calls.append(1), orig(*a, **k))[1]
    try:
        compute_metrics(pred, real, de_pred=de, de_real=de,
                        config=EvalConfig(cache_real=cr, cache_pred=cp, **base))
    finally:
        run.rank_de_side = orig
    assert calls == [1], \
        "precomputed supplied-DE real rank was not reused (precompute/run fingerprint mismatch)"


def test_concurrent_store_does_not_revert_others_update(tmp_path):
    """When a store saves its own (unrelated) key, it must NOT revert another process's
    UPDATE to a pre-existing key: only the store's own dirty keys are merged onto disk."""
    root = str(tmp_path / "shared")
    a = CacheStore(root)
    a.put("k", (["p"], np.zeros((1, 2))), fingerprint="f1", params={}, kind="npz")  # k@f1
    b = CacheStore(root)  # b snapshots {k@f1}
    CacheStore(root).put("k", (["p"], np.ones((1, 2))), fingerprint="f2", params={}, kind="npz")  # k -> f2
    b.put("other", (["p"], np.zeros((1, 2))), fingerprint="fo", params={}, kind="npz")  # b saves only "other"
    fresh = CacheStore(root)
    assert fresh.get("k", fingerprint="f2", params={}, kind="npz") is not MISS      # update preserved
    assert fresh.get("other", fingerprint="fo", params={}, kind="npz") is not MISS  # b's own key written


def test_failed_manifest_write_retried_on_next_put(tmp_path, monkeypatch):
    """A transient manifest-write failure must not drop the entry: the key stays dirty
    and is re-persisted on the next successful save (in-memory state commits only after write)."""
    import cell_eval2.cache as cache
    s = cache.CacheStore(str(tmp_path / "r"))
    real_aw = cache._atomic_write
    state = {"failed": False}

    def flaky(path, write_fn):
        if path.endswith("manifest.json") and not state["failed"]:
            state["failed"] = True
            raise OSError("transient manifest write failure")
        return real_aw(path, write_fn)

    monkeypatch.setattr(cache, "_atomic_write", flaky)
    val = (["p"], np.zeros((1, 2)))
    s.put("k1", val, fingerprint="f1", params={}, kind="npz")  # manifest write fails (data file ok)
    s.put("k2", val, fingerprint="f2", params={}, kind="npz")  # succeeds; must also re-persist k1
    fresh = cache.CacheStore(str(tmp_path / "r"))
    assert fresh.get("k1", fingerprint="f1", params={}, kind="npz") is not MISS
    assert fresh.get("k2", fingerprint="f2", params={}, kind="npz") is not MISS


def _cache_files(root, ext):
    from pathlib import Path
    return sorted(p.name for p in Path(str(root)).iterdir() if p.name.endswith(ext))


def test_put_different_content_uses_distinct_files(tmp_path):
    # F9.2: two puts under ONE key with different content (fingerprint/params) must land in DISTINCT
    # files. A fingerprint-independent filename lets a concurrent manifest merge point one entry at
    # the other's file (wrong-content hit); content-addressed filenames prevent it.
    root = str(tmp_path / "c")
    s = CacheStore(root)
    df1 = pl.DataFrame({"perturbation": ["A"], "metric": ["m"], "value": [1.0]})
    df2 = pl.DataFrame({"perturbation": ["A"], "metric": ["m"], "value": [2.0]})
    s.put("results", df1, fingerprint="R1", params={}, kind="parquet")
    f1 = s._manifest["artifacts"]["results"]["filename"]
    s.put("results", df2, fingerprint="R2", params={"a": 1}, kind="parquet")
    f2 = s._manifest["artifacts"]["results"]["filename"]
    assert f1 != f2, "different-content puts under one key shared a filename (F9.2 wrong-content hit)"


def test_concurrent_puts_no_wrong_content_hit(tmp_path):
    # F9.2 race: two processes put DIFFERENT content under the same key. With a fingerprint-
    # independent filename, an interleaved manifest merge where P1 commits LAST (its entry wins)
    # leaves that entry pointing at P2's file -> get(P1) returns P2's content. Simulate "P1 saves
    # last" by deferring s1's manifest commit until after s2 commits.
    root = str(tmp_path / "c")
    df1 = pl.DataFrame({"perturbation": ["A"], "metric": ["m"], "value": [1.0]})
    df2 = pl.DataFrame({"perturbation": ["A"], "metric": ["m"], "value": [2.0]})
    s1 = CacheStore(root)
    s2 = CacheStore(root)
    saved = s1._save_manifest
    s1._save_manifest = lambda: None                                     # defer P1's manifest commit
    s1.put("results", df1, fingerprint="R1", params={}, kind="parquet")  # writes P1's artifact
    s1._save_manifest = saved
    s2.put("results", df2, fingerprint="R2", params={}, kind="parquet")  # P2 commits R2
    s1._save_manifest()                                                  # P1 commits R1 last -> wins merge
    fresh = CacheStore(root)
    # P1 committed its manifest last, so `results` maps to R1: get(R1) must HIT and return R1's OWN
    # content (df1), never P2's df2. R2 lost the merge, so its entry is gone -> a clean MISS (a miss
    # is safe; a wrong-content hit is the bug). Assert both explicitly so a total cache failure
    # (both MISS) cannot let the test silently pass (Gemini/Copilot #116).
    got1 = fresh.get("results", fingerprint="R1", params={}, kind="parquet")
    assert got1 is not MISS and got1.equals(df1), "R1 must hit and return its own content, not df2"
    got2 = fresh.get("results", fingerprint="R2", params={}, kind="parquet")
    assert got2 is MISS, "R2 lost the manifest merge -> clean miss (never a wrong-content hit)"


def test_put_gcs_superseded_file_for_key(tmp_path):
    # F9.2 follow-through: content-addressed names would otherwise accumulate a file per distinct
    # content. Re-putting a key with NEW content must GC the file it superseded (no longer manifest-
    # referenced), so a single-writer key keeps exactly one artifact file.
    root = str(tmp_path / "c")
    s = CacheStore(root)
    df1 = pl.DataFrame({"perturbation": ["A"], "metric": ["m"], "value": [1.0]})
    df2 = pl.DataFrame({"perturbation": ["A"], "metric": ["m"], "value": [2.0]})
    s.put("results", df1, fingerprint="R1", params={}, kind="parquet")
    s.put("results", df2, fingerprint="R2", params={}, kind="parquet")
    parquets = _cache_files(root, ".parquet")
    assert len(parquets) == 1, f"superseded artifact not GC'd; files={parquets}"
    assert s.get("results", fingerprint="R2", params={}, kind="parquet").equals(df2)
    assert s.get("results", fingerprint="R1", params={}, kind="parquet") is MISS  # old fp -> miss


def test_put_gc_does_not_traverse_outside_root(tmp_path):
    # F9.2 hardening (Copilot #116): the GC delete path must apply the same traversal guard as the
    # read path (_entry_valid), so a tampered/legacy manifest filename with path separators cannot
    # make os.remove delete a file OUTSIDE the cache root.
    root = tmp_path / "c"
    victim = tmp_path / "victim.npz"
    victim.write_bytes(b"important")
    s = CacheStore(str(root))
    s.put("p", (np.array(["A"], dtype=str), np.ones((1, 2))), fingerprint="f1", params={}, kind="npz")
    # Tamper: make p's recorded filename a traversal path pointing at the sibling victim.
    m = json.loads((root / "manifest.json").read_text())
    m["artifacts"]["p"]["filename"] = "../victim.npz"
    (root / "manifest.json").write_text(json.dumps(m))
    s2 = CacheStore(str(root))  # reads the tampered manifest
    # A new-content put supersedes the tampered entry; GC must NOT delete via the traversal path.
    s2.put("p", (np.array(["B"], dtype=str), np.ones((1, 2))), fingerprint="f2", params={}, kind="npz")
    assert victim.exists(), "GC deleted a file outside the cache root via a traversal filename"


def test_derived_policy_records_both_component_names():
    spec = CATALOG["expr_mse_unbiased_capped_norm"]
    assert isinstance(spec.derived, DerivedAgg), spec.derived
    got = derived_policy(["expr_mse_unbiased_capped_norm"])
    assert got == [["expr_mse_unbiased_capped_norm", "ratio_of_sums",
                    "expr_mse_unbiased_capped", "expr_distance_unbiased"]]


def test_derived_policy_is_empty_for_a_plain_metric():
    assert derived_policy(["expr_mae"]) == []


@pytest.mark.parametrize("field", ["numerator", "denominator"])
def test_swapping_either_component_moves_the_result_fingerprint(monkeypatch, field):
    from cell_eval2.cache import result_fingerprint

    names = ["expr_mse_unbiased_capped_norm"]
    kw = dict(real_fp="r", pred_fp="p", de_fps=(), config_digest="c", metric_names=names)
    before = result_fingerprint(**kw)

    spec = CATALOG[names[0]]
    swapped = dict(CATALOG)
    swapped[names[0]] = replace(
        spec, derived=replace(spec.derived, **{field: "expr_mae"}))
    # Patch cell_eval2.CATALOG where `derived_policy` READS it -- the catalog module. Patching
    # `cell_eval2.cache.CATALOG` cannot work: `derived_policy` is defined in catalog.py and
    # resolves that module's global, so the fingerprint would not move and this test would
    # report a bug that is not there (codex round 2).
    monkeypatch.setattr("cell_eval2.catalog.CATALOG", swapped)
    after = result_fingerprint(**kw)

    assert before != after, (
        f"changing the derived {field} left the fingerprint at {before}; a cached result "
        "computed under the old components would be served for the new definition"
    )


def test_the_purity_floor_moves_the_result_config_digest(monkeypatch):
    """A release that retunes a metric CONSTANT changes no input, no config and no metric name,
    so nothing else in the result key moves.

    ⚠️ Asserted with `cell_eval2.__version__` HELD FIXED, deliberately. `__version__` reads the
    INSTALLED distribution metadata, which an editable install froze at install time, so a
    source tree can run new code while reporting an old version -- a version-based guard is
    exactly the one that fails in the environment this repo develops in. The scoped semantics
    term is computed from the constant itself and cannot go stale that way.
    """
    import cell_eval2.metrics.direction as direction
    from cell_eval2.config import EvalConfig
    from cell_eval2.run import _reach_floor_used, _result_config_digest

    names = ["de_wilcoxon_direction_reach_raw"]
    assert _reach_floor_used(names), "the predicate must fire for a raw reach run"
    assert not _reach_floor_used(["expr_mae"]), "and must not fire for a run without reach"

    cfg = EvalConfig.v2()
    kw = dict(de_backend_used=True, comparator="bulk_lognorm", reach_floor_used=True)
    before = _result_config_digest(cfg, **kw)
    monkeypatch.setattr(direction, "REACH_PURITY_FLOOR", 0.975)
    after = _result_config_digest(cfg, **kw)
    assert before != after, (
        "the purity floor does not reach the result cache key, so a result computed under the "
        "old floor would be served -- and re-stamped with the current version -- for a metric "
        "whose definition changed"
    )


def test_a_run_without_reach_keeps_its_warm_key_when_the_floor_moves(monkeypatch):
    """The term is SCOPED, like its three siblings: a run that selects no `direction_reach*`
    metric could not have been affected and must not lose its warm cache."""
    import cell_eval2.metrics.direction as direction
    from cell_eval2.config import EvalConfig
    from cell_eval2.run import _result_config_digest

    cfg = EvalConfig.v2()
    kw = dict(de_backend_used=True, comparator="bulk_lognorm", reach_floor_used=False)
    before = _result_config_digest(cfg, **kw)
    monkeypatch.setattr(direction, "REACH_PURITY_FLOOR", 0.975)
    assert _result_config_digest(cfg, **kw) == before


@pytest.mark.parametrize("field", ["numerator", "denominator"])
def test_swapping_either_component_moves_the_baseline_digest(monkeypatch, field):
    # The cache and the baseline are two independent identities; covering only the cache would
    # let the baseline half of Task 7 be omitted with every test still green.
    from cell_eval2.baseline import _baseline_policy_dict   # the builder around baseline.py:710

    names = ["expr_mse_unbiased_capped_norm"]
    before = _baseline_policy_dict(names, comparator="bulk_lognorm")
    spec = CATALOG[names[0]]
    swapped = dict(CATALOG)
    swapped[names[0]] = replace(spec, derived=replace(spec.derived, **{field: "expr_mae"}))
    monkeypatch.setattr("cell_eval2.catalog.CATALOG", swapped)
    assert _baseline_policy_dict(names, comparator="bulk_lognorm") != before, (
        f"changing the derived {field} left the baseline policy at {before}"
    )


def test_a_PR1_era_baseline_cannot_pair_with_a_PR2_run(monkeypatch):
    """#264 PR2. A PR1 artifact and a PR2 one are otherwise INDISTINGUISHABLE.

    Both stamp `comparator="bulk_lognorm"`; both carry the same `config_digest` inputs; and
    inside one unreleased cycle both carry the same `cell_eval2_version`, which resolves
    through the installed distribution metadata. But PR1 computed the six remaining `expr_*`
    metrics on `lognorm` -- their catalog `normalization` had not moved yet -- so pairing the
    two publishes margins between numbers taken in different spaces, and `cli.py`'s check
    (which compares exactly those stamped fields) would have accepted it.

    The digest now covers the RESOLVED per-metric normalization, so the two differ. This
    simulates the PR1 catalog by putting one metric back on `lognorm`, which is precisely
    what the flip in `catalog.py` changed.
    """
    from cell_eval2 import norm as _norm
    from cell_eval2.baseline import _baseline_policy_dict

    names = ["expr_mse_unbiased_capped", "expr_distance_unbiased"]
    pr2 = _baseline_policy_dict(names, comparator="bulk_lognorm")
    from cell_eval2.baseline import config_digest as _cd
    from cell_eval2.config import EvalConfig as _EC
    pr2_digest = _cd(_EC(metrics=names), comparator="bulk_lognorm")
    pr1_catalog = dict(CATALOG)
    for n in names:
        pr1_catalog[n] = replace(CATALOG[n], normalization="lognorm")
    monkeypatch.setattr("cell_eval2.catalog.CATALOG", pr1_catalog)
    monkeypatch.setattr("cell_eval2.baseline.CATALOG", pr1_catalog)
    pr1 = _baseline_policy_dict(names, comparator="bulk_lognorm")
    # MUTATION-CHECK: without the metric_normalization entry the two are equal, which is the
    # state this test exists to forbid.
    assert {k: v for k, v in pr1.items() if k != "metric_normalization"} == \
           {k: v for k, v in pr2.items() if k != "metric_normalization"}, (
               "the other policy fields already differ, so this test would pass without the "
               "normalization entry -- it must be the discriminating one")
    assert pr1 != pr2, (
        "a PR1-era baseline (expr_* on lognorm) and a PR2 one hash identically at the same "
        f"comparator: {pr2}"
    )
    # And the mapping says which space each metric was actually in, not just that it differs.
    assert dict(pr2["metric_normalization"]) == {n: "bulk_lognorm" for n in names}
    assert dict(pr1["metric_normalization"]) == {n: "lognorm" for n in names}
    # The declaration itself is unchanged by the comparator -- only the RESOLUTION is.
    assert _norm.EXPR_COMPARATOR == "expr_comparator"

    # And the same through the SHIPPED digest, which is what cli.py actually compares --
    # the policy dict is an implementation detail of it.
    from cell_eval2.baseline import config_digest
    from cell_eval2.config import EvalConfig
    cfg = EvalConfig(metrics=names)
    assert config_digest(cfg, comparator="bulk_lognorm") != pr2_digest, (
        "a PR1-era baseline and a PR2 one produce the same config_digest, so cli.py would "
        "pair them and publish the margins"
    )


def _pair(jk):
    perts = np.array(["a", "b"])
    return (perts, np.array([[1.0, 2.0], [3.0, 4.0]])), GroupMoments(
        perts=perts, counts=np.array([5.0, 6.0]), sumsq=np.array([7.0, 8.0]), jk=jk)


def test_moments_artifact_round_trips_jk(tmp_path):
    p = str(tmp_path / "m.npz")
    _dump_npz_moments(_pair(np.array([0.25, 0.5])), p)
    np.testing.assert_allclose(_load_npz_moments(p)[1].jk, [0.25, 0.5])


def test_moments_artifact_round_trips_absent_jk_as_none(tmp_path):
    """None must come back as None, not as an empty or zero array -- a zeroed jk is a
    silently VALID-looking correction of exactly zero."""
    p = str(tmp_path / "m.npz")
    _dump_npz_moments(_pair(None), p)
    assert _load_npz_moments(p)[1].jk is None


@pytest.mark.parametrize("keys", [
    {"jk": np.array([1.0, 2.0]), "jk_absent": np.array(True)},   # both
    {},                                                          # neither
    {"jk_absent": np.array(False)},                              # a false sentinel
])
def test_a_malformed_moments_artifact_raises_rather_than_loading(tmp_path, keys):
    """Rev 1's loader treated any jk_absent key as authoritative, so 'both' loaded silently
    as jk=None -- i.e. scored with no correction. CacheStore.get treats a raise as a MISS."""
    p = str(tmp_path / "m.npz")
    perts = np.array(["a", "b"])
    np.savez(p, perts=perts, means=np.array([[1.0, 2.0], [3.0, 4.0]]),
             counts=np.array([5.0, 6.0]), sumsq=np.array([7.0, 8.0]), **keys)
    with pytest.raises((ValueError, KeyError)):
        _load_npz_moments(p)


def test_cache_format_version_was_bumped():
    """A pre-PR2 artifact has no jk key at all; it must be rejected by VERSION rather than
    loaded with a missing correction (spec §5 item 3)."""
    assert CACHE_FORMAT_VERSION == 2


def test_json_kind_roundtrips_a_nested_object(tmp_path):
    """The anchor bundle is three objects (aggregate, splits, meta) under one key, and the
    store keeps one value per key. npz cannot hold a dict and parquet cannot hold two
    differently-shaped frames, so the store gains a json kind."""
    from cell_eval2.cache import CacheStore

    store = CacheStore(str(tmp_path))
    value = {"a": [{"metric": "x", "replicate": 0.5, "n": None}], "meta": {"k": [1, 2]}}
    store.put("k", value, fingerprint="fp", params={"p": 1}, kind="json")
    got = store.get("k", fingerprint="fp", params={"p": 1}, kind="json")
    assert got == value


def test_json_kind_misses_on_a_different_param(tmp_path):
    from cell_eval2.cache import MISS, CacheStore

    store = CacheStore(str(tmp_path))
    store.put("k", {"v": 1}, fingerprint="fp", params={"p": 1}, kind="json")
    assert store.get("k", fingerprint="fp", params={"p": 2}, kind="json") is MISS
