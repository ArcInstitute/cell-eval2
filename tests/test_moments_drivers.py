import anndata as ad
import numpy as np
import pandas as pd
import polars as pl
import pytest
from scipy.sparse import csr_matrix

from cell_eval2 import norm as _norm
from cell_eval2.cache import MISS, CacheStore
from cell_eval2.config import EvalConfig
from cell_eval2.gpu.bulk import GroupedMeanAccumulator
from cell_eval2.moments import GroupMoments
from cell_eval2.prep import pseudobulk_with_moments
from cell_eval2.run import compute_metrics
from cell_eval2.streaming_bulk import inmem_pseudobulk


def _blocks(seed=0, n_cells=60, n_genes=12, n_groups=3):
    rng = np.random.default_rng(seed)
    X = rng.poisson(3.0, size=(n_cells, n_genes)).astype(np.float64)
    groups = rng.integers(0, n_groups, size=n_cells)
    return csr_matrix(X), groups


def test_accumulator_moments_none_when_not_requested():
    acc = GroupedMeanAccumulator(3, 12, normalizations=["counts"], target_sum=1e6,
                                 device="cpu")
    X, g = _blocks()
    acc.update(X, g)
    assert acc.moments() is None


def test_accumulator_counts_moments_match_numpy():
    X, g = _blocks()
    acc = GroupedMeanAccumulator(3, 12, normalizations=["counts"], target_sum=1e6,
                                 device="cpu", with_moments=True)
    acc.update(X, g)
    counts, sumsq = acc.moments()["counts"]
    D = X.toarray()
    for k in range(3):
        sub = D[g == k]
        assert counts[k] == pytest.approx(float(sub.shape[0]))
        assert sumsq[k] == pytest.approx(float(np.sum(sub * sub)), rel=1e-12)


def test_accumulator_lognorm_moments_match_numpy():
    X, g = _blocks(seed=4)
    acc = GroupedMeanAccumulator(3, 12, normalizations=["lognorm"], target_sum=1e6,
                                 device="cpu", with_moments=True)
    acc.update(X, g)
    counts, sumsq = acc.moments()["lognorm"]
    D = X.toarray()
    libs = D.sum(axis=1)
    libs[libs == 0] = 1.0
    L = np.log1p(D * (1e6 / libs)[:, None])
    for k in range(3):
        sub = L[g == k]
        assert sumsq[k] == pytest.approx(float(np.sum(sub * sub)), rel=1e-10)


def test_accumulator_moments_additive_across_blocks():
    X, g = _blocks(seed=5, n_cells=60)
    one = GroupedMeanAccumulator(3, 12, normalizations=["counts"], target_sum=1e6,
                                 device="cpu", with_moments=True)
    one.update(X, g)
    split = GroupedMeanAccumulator(3, 12, normalizations=["counts"], target_sum=1e6,
                                   device="cpu", with_moments=True)
    split.update(X[:25], g[:25])
    split.update(X[25:], g[25:])
    for a, b in zip(one.moments()["counts"], split.moments()["counts"]):
        np.testing.assert_allclose(a, b, rtol=1e-12, atol=0)


def test_finalize_shape_unchanged_with_moments():
    X, g = _blocks(seed=6)
    acc = GroupedMeanAccumulator(3, 12, normalizations=["counts", "lognorm"],
                                 target_sum=1e6, device="cpu", with_moments=True)
    acc.update(X, g)
    out = acc.finalize()
    assert set(out) == {"counts", "lognorm"}
    idx, means = out["counts"]
    assert idx.shape == (3,) and means.shape == (3, 12) and means.dtype == np.float32


def _counts_adata(seed=11, n_cells=90, n_genes=15, dtype=np.float32):
    rng = np.random.default_rng(seed)
    labels = np.repeat(["non-targeting", "A", "B"], n_cells // 3)
    X = rng.poisson(4.0, size=(labels.size, n_genes)).astype(dtype)
    X[X < 2] = 0.0  # real sparsity
    obs = pd.DataFrame({"target": labels}, index=[str(i) for i in range(labels.size)])
    var = pd.DataFrame(index=[f"g{i}" for i in range(n_genes)])
    return ad.AnnData(X=csr_matrix(X), obs=obs, var=var)


def test_inmem_pseudobulk_moments_off_by_default():
    out = inmem_pseudobulk(_counts_adata(), pert_col="target", norms=["lognorm"],
                           target_sum=1e6)
    assert isinstance(out, dict) and set(out) == {"lognorm"}


# The two tolerances below are NOT arbitrary. `to_normalization` goes through
# scanpy's normalize_total, which divides in the INPUT dtype, so the prep path does CPM in
# fp32 on an fp32 fixture; the streaming/accumulator paths cast data to fp64 BEFORE the CPM
# scale. Measured: max relative sumsq difference 1.0e-8 on fp32 input, 2.1e-15 on fp64. The
# same pre-existing gap is why test_streaming_bulk.py:149 compares MEANS at rtol=1e-4.
# So: an fp64 fixture pins the ALGEBRA exactly, and an fp32 fixture pins what actually ships.

def test_inmem_pseudobulk_moments_match_prep_reference_exactly_in_fp64():
    """Driver parity, algebra: on fp64 input, where no normalize_total precision is lost, the
    accumulator and the in-memory reference agree to fp64 round-off.
    (#198 acceptance criterion 6.)"""
    adata = _counts_adata(seed=12, dtype=np.float64)
    _, moments = inmem_pseudobulk(adata, pert_col="target", norms=["lognorm"],
                                  target_sum=1e6, with_moments=True)
    ref_perts, _, ref_mom = pseudobulk_with_moments(
        _norm.to_normalization(adata, "counts", "lognorm", target_sum=1e6), "target")
    got = moments["lognorm"]
    assert list(got.perts) == list(ref_perts)
    np.testing.assert_allclose(got.counts, ref_mom.counts, rtol=0, atol=0)
    np.testing.assert_allclose(got.sumsq, ref_mom.sumsq, rtol=1e-12, atol=0)


def test_inmem_pseudobulk_moments_match_prep_reference_on_fp32_input():
    """Driver parity, as shipped: fp32 input is the real case, and the residual is the
    fp32-normalize gap described above -- not a moments defect."""
    adata = _counts_adata(seed=12, dtype=np.float32)
    _, moments = inmem_pseudobulk(adata, pert_col="target", norms=["lognorm"],
                                  target_sum=1e6, with_moments=True)
    _, _, ref_mom = pseudobulk_with_moments(
        _norm.to_normalization(adata, "counts", "lognorm", target_sum=1e6), "target")
    np.testing.assert_allclose(moments["lognorm"].counts, ref_mom.counts, rtol=0, atol=0)
    np.testing.assert_allclose(moments["lognorm"].sumsq, ref_mom.sumsq, rtol=1e-6, atol=0)


def test_inmem_pseudobulk_moments_multi_block_identical():
    """Single- vs multi-block accumulation must agree (the shard analogue)."""
    adata = _counts_adata(seed=13, n_cells=90)
    _, one = inmem_pseudobulk(adata, pert_col="target", norms=["lognorm"], target_sum=1e6,
                              with_moments=True)
    _, many = inmem_pseudobulk(adata, pert_col="target", norms=["lognorm"], target_sum=1e6,
                               with_moments=True, block_rows=7)
    np.testing.assert_allclose(one["lognorm"].sumsq, many["lognorm"].sumsq, rtol=1e-12, atol=0)


def test_duplicate_bearing_csc_is_rejected_not_silently_wrong():
    """The canonical-format guard must cover CSC, not just CSR.

    Copilot (PR #209) suggested narrowing the guard to `X.format == "csr"`, reasoning that a
    non-CSR input is converted per block and the conversion canonicalizes. Measured: it does
    NOT. `csc.tocsr()` on a duplicate-bearing CSC returns a matrix that is still not canonical
    -- its `.data` stays [1., 2., 4., 5.] rather than collapsing to [3., 9.] -- so a CSR-only
    guard would let this through and produce a silently wrong Σ‖x‖². anndata accepts exactly
    CSR and CSC (`coerce_array` rejects lil/dok/dia/coo), so those two are the whole reachable
    surface and both expose `has_canonical_format`."""
    from scipy.sparse import csc_matrix

    dup = csc_matrix((np.array([1.0, 2.0, 4.0, 5.0]), np.array([0, 0, 1, 1]),
                      np.array([0, 2, 4])), shape=(2, 2))
    assert dup.has_canonical_format is False
    assert dup.tocsr().has_canonical_format is False          # the trap, pinned
    np.testing.assert_allclose(dup.tocsr().data, [1.0, 2.0, 4.0, 5.0])   # duplicates survive

    obs = pd.DataFrame({"target": ["a", "b"]}, index=["0", "1"])
    var = pd.DataFrame(index=["g0", "g1"])
    adata = ad.AnnData(X=dup, obs=obs, var=var)
    with pytest.raises(ValueError, match="duplicate coordinates"):
        inmem_pseudobulk(adata, pert_col="target", norms=["counts"], target_sum=1e6,
                         with_moments=True)
    # ...and without moments the pre-existing behaviour is untouched.
    out = inmem_pseudobulk(adata, pert_col="target", norms=["counts"], target_sum=1e6)
    assert set(out) == {"counts"}


def test_inmem_moments_include_the_control():
    """The invariant: moments carry EVERY group, control included."""
    adata = _counts_adata(seed=14)
    _, moments = inmem_pseudobulk(adata, pert_col="target", norms=["lognorm"],
                                  target_sum=1e6, with_moments=True)
    assert "non-targeting" in set(map(str, moments["lognorm"].perts))


def _streaming_cpu_moments(adata, *, n_blocks=2):
    """Run the streaming CPU accumulator over `n_blocks` contiguous 'shards'."""
    from cell_eval2.streaming_bulk import _streaming_pseudobulk_cpu

    labels = adata.obs["target"].to_numpy().astype(str)
    perts = np.unique(labels)
    step = -(-adata.n_obs // n_blocks)
    blocks = [(adata.X[i:i + step], labels[i:i + step])
              for i in range(0, adata.n_obs, step)]
    _, moments = _streaming_pseudobulk_cpu(
        iter(blocks), perts, perts.size, adata.n_vars, ["lognorm"], 1e6, with_moments=True)
    return perts, moments["lognorm"]


def test_streaming_cpu_moments_match_prep_reference():
    """The THIRD accumulator: _streaming_pseudobulk_cpu is a separate implementation from the
    GroupedMeanAccumulator that inmem_pseudobulk uses, so it needs its own parity check.
    fp64 fixture, so the normalize_total precision gap described above is not in play.
    (#198 acceptance criterion 6.)"""
    adata = _counts_adata(seed=15, dtype=np.float64)
    perts, got = _streaming_cpu_moments(adata)
    _, _, ref_mom = pseudobulk_with_moments(
        _norm.to_normalization(adata, "counts", "lognorm", target_sum=1e6), "target")
    assert list(got.perts) == list(perts)
    np.testing.assert_allclose(got.counts, ref_mom.counts, rtol=0, atol=0)
    np.testing.assert_allclose(got.sumsq, ref_mom.sumsq, rtol=1e-12, atol=0)


def test_streaming_cpu_and_accumulator_agree():
    """The two STREAMING implementations compute the per-cell library size differently --
    scipy's row sum vs a weighted bincount over the resident data -- so pin that they still
    land on the same sumsq. Measured identical (0.0) on integer counts, where an fp64 sum of
    ints below 2^53 is order-independent."""
    adata = _counts_adata(seed=17, dtype=np.float64)
    _, streamed = _streaming_cpu_moments(adata, n_blocks=3)
    _, accum = inmem_pseudobulk(adata, pert_col="target", norms=["lognorm"],
                                target_sum=1e6, with_moments=True, block_rows=40)
    np.testing.assert_allclose(streamed.counts, accum["lognorm"].counts, rtol=0, atol=0)
    np.testing.assert_allclose(streamed.sumsq, accum["lognorm"].sumsq, rtol=1e-12, atol=0)


def test_streaming_cpu_moments_survive_an_all_zero_cell():
    """A cell with no counts hits the `libs == 0 -> 1.0` floor in every driver; its log1p(CPM)
    row is all zeros, so it contributes 0 to sumsq but 1 to the count."""
    adata = _counts_adata(seed=18, dtype=np.float64)
    X = adata.X.toarray()
    X[3] = 0.0
    adata.X = csr_matrix(X)
    perts, got = _streaming_cpu_moments(adata)
    _, _, ref_mom = pseudobulk_with_moments(
        _norm.to_normalization(adata, "counts", "lognorm", target_sum=1e6), "target")
    np.testing.assert_allclose(got.counts, ref_mom.counts, rtol=0, atol=0)
    np.testing.assert_allclose(got.sumsq, ref_mom.sumsq, rtol=1e-12, atol=0)


def test_streaming_cpu_moments_off_by_default():
    from cell_eval2.streaming_bulk import _streaming_pseudobulk_cpu

    adata = _counts_adata(seed=16)
    labels = adata.obs["target"].to_numpy().astype(str)
    perts = np.unique(labels)
    out = _streaming_pseudobulk_cpu(iter([(adata.X, labels)]), perts, perts.size,
                                    adata.n_vars, ["lognorm"], 1e6)
    assert isinstance(out, dict) and set(out) == {"lognorm"}


def test_npz_moments_roundtrips_all_four_arrays(tmp_path):
    store = CacheStore(str(tmp_path))
    perts = np.array(["A", "B", "ctrl"])
    means = np.arange(9, dtype=np.float32).reshape(3, 3)
    mom = GroupMoments(perts=perts, counts=np.array([1.0, 2.0, 3.0]),
                       sumsq=np.array([4.0, 5.0, 6.0]))
    store.put("pseudobulk_moments_lognorm", ((perts, means), mom),
              fingerprint="fp", params={"a": 1}, kind="npz_moments")
    (gp, gm), gmom = store.get("pseudobulk_moments_lognorm",
                               fingerprint="fp", params={"a": 1}, kind="npz_moments")
    assert list(gp) == list(perts)
    np.testing.assert_allclose(gm, means, rtol=0, atol=0)
    np.testing.assert_allclose(gmom.counts, mom.counts, rtol=0, atol=0)
    np.testing.assert_allclose(gmom.sumsq, mom.sumsq, rtol=0, atol=0)


def test_plain_pseudobulk_cache_is_never_served_to_a_moments_run(tmp_path):
    """#198 acceptance criterion 7 -- structural: the keys differ, so a pre-change cache
    cannot be reused as if it carried moments."""
    store = CacheStore(str(tmp_path))
    perts = np.array(["A", "ctrl"])
    means = np.zeros((2, 3), dtype=np.float32)
    store.put("pseudobulk_lognorm", (perts, means),
              fingerprint="fp", params={"a": 1}, kind="npz")
    assert store.get("pseudobulk_moments_lognorm",
                     fingerprint="fp", params={"a": 1}, kind="npz_moments") is MISS
    # ...and the plain entry still hits, untouched.
    assert store.get("pseudobulk_lognorm", fingerprint="fp", params={"a": 1},
                     kind="npz") is not MISS


def test_npz_moments_rejects_a_truncated_file(tmp_path):
    store = CacheStore(str(tmp_path))
    perts = np.array(["A", "B"])
    bad = GroupMoments(perts=perts, counts=np.array([1.0, 2.0]), sumsq=np.array([1.0, 2.0]))
    store.put("k", ((perts, np.zeros((2, 3))), bad), fingerprint="fp", params={},
              kind="npz_moments")
    path = tmp_path / store._manifest["artifacts"]["k"]["filename"]
    np.savez(path, perts=perts, means=np.zeros((2, 3)))  # drop counts/sumsq
    assert store.get("k", fingerprint="fp", params={}, kind="npz_moments") is MISS


def test_npz_moments_refuses_a_bulk_moments_label_mismatch(tmp_path):
    """The artifact stores ONE label vector. A permuted-but-correctly-labelled GroupMoments
    would round-trip mislabelled, so dumping it must fail rather than corrupt silently."""
    store = CacheStore(str(tmp_path))
    perts = np.array(["A", "B"])
    permuted = GroupMoments(perts=np.array(["B", "A"]), counts=np.array([1.0, 2.0]),
                            sumsq=np.array([3.0, 4.0]))
    with pytest.raises(ValueError, match="labels differ"):
        store.put("k2", ((perts, np.zeros((2, 3))), permuted), fingerprint="fp", params={},
                  kind="npz_moments")


def test_a_moments_entry_is_never_decoded_by_the_plain_npz_loader(tmp_path):
    """The manifest holds ONE entry per key and the loaders are not interchangeable: without a
    kind check, _load_npz would read a .moments.npz and silently drop counts/sumsq."""
    store = CacheStore(str(tmp_path))
    perts = np.array(["A", "B"])
    means = np.zeros((2, 3), dtype=np.float32)
    mom = GroupMoments(perts=perts, counts=np.array([2.0, 2.0]), sumsq=np.array([1.0, 1.0]))
    store.put("shared", ((perts, means), mom), fingerprint="fp", params={}, kind="npz_moments")
    assert store.get("shared", fingerprint="fp", params={}, kind="npz") is MISS
    assert store.get("shared", fingerprint="fp", params={}, kind="npz_moments") is not MISS


def test_legacy_manifest_entries_without_a_kind_still_hit(tmp_path):
    """Backward compatibility: entries written before the kind field carry none and must not
    be invalidated wholesale."""
    store = CacheStore(str(tmp_path))
    perts, means = np.array(["A"]), np.zeros((1, 3), dtype=np.float32)
    store.put("legacy", (perts, means), fingerprint="fp", params={}, kind="npz")
    store._manifest["artifacts"]["legacy"].pop("kind")      # simulate a pre-change manifest
    assert store.get("legacy", fingerprint="fp", params={}, kind="npz") is not MISS


def test_npz_and_npz_moments_use_distinct_extensions(tmp_path):
    """Same key + same fingerprint + same params under two kinds must not share a file, so a
    plain `npz` loader can never read a moments artifact and drop its extra arrays."""
    store = CacheStore(str(tmp_path))
    perts = np.array(["A", "B"])
    means = np.zeros((2, 3), dtype=np.float32)
    mom = GroupMoments(perts=perts, counts=np.array([2.0, 2.0]), sumsq=np.array([1.0, 1.0]))
    store.put("same", (perts, means), fingerprint="fp", params={}, kind="npz")
    plain_file = store._manifest["artifacts"]["same"]["filename"]
    store.put("same", ((perts, means), mom), fingerprint="fp", params={}, kind="npz_moments")
    moments_file = store._manifest["artifacts"]["same"]["filename"]
    assert plain_file != moments_file
    assert plain_file.endswith(".npz") and moments_file.endswith(".moments.npz")


def _paired_adata(seed):
    rng = np.random.default_rng(seed)
    labels = np.repeat(["non-targeting", "A", "B"], 40)
    X = rng.poisson(5.0, size=(labels.size, 20)).astype(np.float32)
    obs = pd.DataFrame({"target": labels}, index=[str(i) for i in range(labels.size)])
    # Targets 'A' and 'B' name MEASURED genes. Issue #172 made the `expr_mse_unbiased*` /
    # `expr_distance_unbiased` family drop each perturbation's own column and RAISE when NO
    # target resolves against the gene panel, so the suite's usual "targets A/B, genes g0.."
    # shape now fails for the construct-ID reason. This is the shape every real panel has.
    var = pd.DataFrame(index=["A", "B"] + [f"g{i}" for i in range(2, 20)])
    return ad.AnnData(X=csr_matrix(X), obs=obs, var=var)


def test_compute_metrics_end_to_end_emits_expr_mse_unbiased_capped():
    real, pred = _paired_adata(1), _paired_adata(2)
    cfg = EvalConfig(metrics=["expr_mse", "expr_mse_unbiased_capped"], control="non-targeting",
                     pert_col="target", input_type="counts", device="cpu")
    df = compute_metrics(pred, real, config=cfg)
    got = set(df["metric"].to_list())
    assert "expr_mse_unbiased_capped" in got and "expr_mse" in got
    vals = df.filter(pl.col("metric") == "expr_mse_unbiased_capped")["value"].to_list()
    plain = df.filter(pl.col("metric") == "expr_mse")["value"].to_list()
    assert len(vals) == 2                       # A and B; control excluded
    # same population both sides -> the correction must actually bite
    assert float(np.mean(vals)) < float(np.mean(plain))


def test_compute_metrics_without_the_metric_asks_for_no_moments(monkeypatch):
    """The moments pass is opt-in: a run that doesn't need them must not pay for them."""
    import cell_eval2.run as run_mod

    seen = {}
    original = run_mod._side_bulks

    def spy(*a, **kw):
        seen["moment_norms"] = set(kw.get("moment_norms") or ())
        return original(*a, **kw)

    monkeypatch.setattr(run_mod, "_side_bulks", spy)
    cfg = EvalConfig(metrics=["expr_mse"], control="non-targeting", pert_col="target",
                     input_type="counts", device="cpu")
    compute_metrics(_paired_adata(3), _paired_adata(4), config=cfg)
    assert seen["moment_norms"] == set()


def test_restrict_takes_bulks_only_and_never_moments():
    """Pins the §4.6 invariant at its most dangerous site: _restrict subsets the bulks to
    the chosen perturbations, and moments must not travel through it -- #202 needs the
    control's trace AFTER the control has been dropped from the bulk."""
    import inspect

    from cell_eval2.scale import _restrict

    assert list(inspect.signature(_restrict).parameters) == ["bulks", "chosen"]
    perts = np.array(["A", "B", "non-targeting"])
    means = np.arange(9, dtype=np.float64).reshape(3, 3)
    out = _restrict({"lognorm": (perts, means)}, ["A"])
    got_perts, got_means = out["lognorm"]
    assert list(got_perts) == ["A"] and got_means.shape == (1, 3)


def test_score_piece_rejects_moments_metrics():
    from cell_eval2.run import _reject_moments_metrics

    with pytest.raises(NotImplementedError, match="partitioned in-memory"):
        _reject_moments_metrics(["expr_mse_unbiased_capped"],
                                driver="score_piece (partitioned in-memory)")


def test_the_accumulator_serves_bulk_lognorm_moments_directly():
    """The inline guard at gpu/bulk.py:63-66 is unreachable through any driver (the shared
    helper always fires first), so only a direct construction proves it is gone. This is also
    the shape test_moments_drivers.py already uses everywhere else in this file."""
    X, g = _blocks()
    acc = GroupedMeanAccumulator(3, 12, normalizations=["bulk_lognorm"], target_sum=None,
                                 device="cpu", with_moments=True, bulk_target_sum=1e6)
    acc.update(X, g)
    acc.jackknife(lambda: iter([(X, g)]))
    counts, sumsq = acc.moments()["bulk_lognorm"]      # still a 2-TUPLE, see Task 4(e)
    jk = acc.jackknife_by_norm()["bulk_lognorm"]
    assert counts.shape == sumsq.shape == jk.shape == (3,)
    assert np.all(jk >= 0.0)
