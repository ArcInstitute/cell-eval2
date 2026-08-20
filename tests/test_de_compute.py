import anndata as ad
import numpy as np
import pandas as pd
import polars as pl
import pytest
from scipy.stats import false_discovery_control

from cell_eval2.de import REQUIRED_COLS
from cell_eval2.de_compute import compute_de


def test_lfc_clip_zero_means_matches_pdex_semantics():
    from cell_eval2.de_compute import _lfc_from_means

    genes = np.array(["g0", "g1", "g2", "g3"])
    means = {
        "ref": np.array([1.0, 0.0, 2.0, 0.0]),
        "T": np.array([2.0, 1.0, 0.0, 0.0]),
    }
    df = _lfc_from_means(means, genes, reference="ref", epsilon=0.0, clip_value=20.0)
    lfc = dict(zip(df["feature"], df["log2_fold_change"]))
    assert np.isclose(lfc["g0"], np.log2(2.0))      # both nonzero -> normal ratio
    assert np.isclose(lfc["g1"], np.log2(20.0))     # ref==0 -> clip_value
    assert np.isclose(lfc["g2"], np.log2(1.0 / 20.0)) # tgt==0 -> 1/clip_value
    assert np.isclose(lfc["g3"], 0.0)               # both zero -> 1 -> log2(1)=0


def test_lfc_clip_none_keeps_inf():
    from cell_eval2.de_compute import _lfc_from_means

    genes = np.array(["g0", "g1"])
    means = {"ref": np.array([0.0, 2.0]), "T": np.array([2.0, 0.0])}
    df = _lfc_from_means(means, genes, reference="ref", epsilon=0.0, clip_value=None)
    vals = df["log2_fold_change"].to_numpy()
    assert np.isposinf(vals[0]) and np.isneginf(vals[1])  # unchanged behavior


def test_clipped_log2fc_clip_semantics():
    from cell_eval2.de_compute import _clipped_log2fc

    mt  = np.array([2.0, 0.0, 0.0, 4.0, 5.0])
    ref = np.array([1.0, 3.0, 0.0, 2.0, 0.0])
    lfc = _clipped_log2fc(mt, ref, epsilon=0.0, clip_value=20.0)
    assert np.isclose(lfc[0], np.log2(2.0 / 1.0))      # both nonzero -> ratio
    assert np.isclose(lfc[1], np.log2(1.0 / 20.0))     # tgt==0 -> 1/clip
    assert np.isclose(lfc[2], 0.0)                     # both zero -> ratio 1 -> log2(1)=0
    assert np.isclose(lfc[3], np.log2(4.0 / 2.0))      # both nonzero -> ratio
    assert np.isclose(lfc[4], np.log2(20.0))           # ref==0 -> clip
    assert np.isfinite(lfc).all()


def test_clipped_log2fc_none_keeps_inf():
    import warnings

    from cell_eval2.de_compute import _clipped_log2fc

    mt  = np.array([0.0, 2.0, 0.0])
    ref = np.array([1.0, 0.0, 0.0])
    # The expected inf/NaN on this path must NOT emit numpy divide/invalid warnings
    # (np.errstate wrap); promote RuntimeWarning to an error to pin the suppression.
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        lfc = _clipped_log2fc(mt, ref, epsilon=0.0, clip_value=None)
    assert np.isneginf(lfc[0])   # log2(0/1)
    assert np.isposinf(lfc[1])   # log2(2/0)
    assert np.isnan(lfc[2])      # log2(0/0)


@pytest.mark.parametrize("clip_value", [20.0, None])
def test_lfc_from_means_equals_clipped_log2fc(clip_value):
    # Exact-equality pin of the refactor: _lfc_from_means must produce exactly what
    # _clipped_log2fc returns per non-reference group (guards against future divergence).
    from cell_eval2.de_compute import _clipped_log2fc, _lfc_from_means

    genes = np.array(["g0", "g1", "g2", "g3"])
    means = {
        "ref": np.array([1.0, 0.0, 2.0, 0.0]),
        "A":   np.array([2.0, 1.0, 0.0, 0.0]),
        "B":   np.array([0.5, 3.0, 4.0, 1.0]),
    }
    df = _lfc_from_means(means, genes, reference="ref", epsilon=0.0, clip_value=clip_value)
    for g in ("A", "B"):
        got = df.filter(pl.col("target") == g)["log2_fold_change"].to_numpy()
        exp = _clipped_log2fc(means[g], means["ref"], epsilon=0.0, clip_value=clip_value)
        assert np.array_equal(got, exp, equal_nan=True)


def _toy_linear():
    # 2 groups, 2 genes, hand-computable arithmetic means.
    X = np.array([[2.0, 0.0],
                  [4.0, 0.0],     # GENE1: gene0 mean=3, gene1 mean=0
                  [1.0, 10.0],
                  [1.0, 20.0]],   # non-targeting: gene0 mean=1, gene1 mean=15
                 dtype=float)
    obs = pd.DataFrame({"target": ["GENE1", "GENE1", "non-targeting", "non-targeting"]},
                       index=[f"c{i}" for i in range(4)])
    var = pd.DataFrame(index=["g0", "g1"])
    return ad.AnnData(X=X, obs=obs, var=var)


def _sig_sets(df, thr=0.05):
    out = {}
    for t in df["target"].unique().to_list():
        sub = df.filter((pl.col("target") == t) & (pl.col("p_adj") < thr))
        out[t] = set(sub["feature"].to_list())
    return out


def _jaccard(a, b):
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


def test_group_means_linear_arithmetic_and_geometric():
    from cell_eval2.de_compute import _group_means_linear
    a = _toy_linear()
    arr = _group_means_linear(a, "target", "arithmetic")
    assert np.allclose(arr["GENE1"], [3.0, 0.0])
    assert np.allclose(arr["non-targeting"], [1.0, 15.0])
    geo = _group_means_linear(a, "target", "geometric")
    # geometric gene0 of GENE1 = expm1(mean(log1p([2,4]))); gene1 is all-zero -> 0
    assert np.isclose(geo["GENE1"][0], np.expm1(np.log1p(np.array([2.0, 4.0])).mean()))
    assert np.isclose(geo["GENE1"][1], 0.0)


def test_compute_lfc_table_arithmetic():
    from cell_eval2.de_compute import compute_lfc_table
    a = _toy_linear()
    df = compute_lfc_table(a, groupby="target", reference="non-targeting",
                           mean_calc="arithmetic", epsilon=1e-9)
    assert set(df["target"].unique()) == {"GENE1"}                 # reference excluded
    row = {r["feature"]: r["log2_fold_change"] for r in df.iter_rows(named=True)}
    assert np.isclose(row["g0"], np.log2((3.0 + 1e-9) / (1.0 + 1e-9)))  # ~log2(3)


def test_to_linear_counts_is_cpm_and_lognorm_is_expm1():
    from cell_eval2.de_compute import _to_linear
    counts = ad.AnnData(
        X=np.array([[1.0, 1.0], [3.0, 1.0]]),
        obs=pd.DataFrame({"target": ["a", "b"]}, index=["c0", "c1"]),
        var=pd.DataFrame(index=["g0", "g1"]))
    lin = _to_linear(counts, "counts")
    assert np.allclose(np.asarray(lin.X).sum(axis=1), [1e6, 1e6])    # CPM rows sum to 1e6
    log = ad.AnnData(
        X=np.log1p(np.array([[5.0, 0.0]])),
        obs=pd.DataFrame({"target": ["a"]}, index=["c0"]),
        var=pd.DataFrame(index=["g0", "g1"]))
    lin2 = _to_linear(log, "lognorm")
    assert np.allclose(np.asarray(lin2.X), [[5.0, 0.0]])            # expm1 recovers linear


def test_to_linear_target_sum_none_is_median(synthetic_counts_pair):
    from cell_eval2.de_compute import _to_linear

    real, _ = synthetic_counts_pair
    lin = _to_linear(real, "counts", target_sum=None)     # median normalization
    sums = np.asarray(lin.X.sum(axis=1)).ravel()
    assert np.allclose(sums, np.median(sums), rtol=1e-5)  # all cells -> median lib size


def test_to_linear_sparse_integer_lognorm_does_not_raise():
    # A sparse integer matrix declared lognorm must not crash: expm1 cannot write float
    # results into an int .data in place -> cast to float first (Gemini HIGH on PR #10).
    from scipy.sparse import csr_matrix
    from cell_eval2.de_compute import _to_linear
    a = ad.AnnData(
        X=csr_matrix(np.array([[1, 0], [0, 2]], dtype=np.int64)),
        obs=pd.DataFrame({"target": ["x", "y"]}, index=["c0", "c1"]),
        var=pd.DataFrame(index=["g0", "g1"]))
    out = _to_linear(a, "lognorm")
    assert np.allclose(np.asarray(out.X.todense()), np.expm1([[1.0, 0.0], [0.0, 2.0]]))


def test_to_linear_lognorm_float32_preserved_and_matches_fp64():
    # The streaming/row-store path feeds float32 (scaled_log1p). _to_linear must preserve
    # float32 (skip the fp64 upcast) while matching the fp64 expm1 within the accepted DE
    # residual. O(1e4) linear values (expm1 of log1p up to target_sum) stress the precision.
    from cell_eval2.de_compute import _to_linear
    rng = np.random.default_rng(0)
    linear = rng.uniform(0.0, 1e4, size=(200, 50)).astype(np.float32)
    x32 = np.log1p(linear).astype(np.float32)                       # float32 lognorm input
    a = ad.AnnData(
        X=x32.copy(), obs=pd.DataFrame({"target": ["p"] * 200}),
        var=pd.DataFrame(index=[f"g{j}" for j in range(50)]))
    out = _to_linear(a, "lognorm")
    assert out.X.dtype == np.float32                                # NEW: no fp64 upcast
    ref_fp64 = np.expm1(x32.astype(np.float64))                     # fp64 expm1 of the SAME input
    np.testing.assert_allclose(np.asarray(out.X), ref_fp64, rtol=1e-5, atol=1e-5)


def test_to_linear_lognorm_float64_and_int_stay_float64_bit_identical():
    # preserve-dtype must leave float64 and int inputs bit-identical -- only float32 changes.
    from scipy.sparse import csr_matrix
    from cell_eval2.de_compute import _to_linear
    x64 = np.log1p(np.array([[5.0, 0.0, 1000.0]]))                  # float64 lognorm
    a64 = ad.AnnData(
        X=x64.copy(), obs=pd.DataFrame({"target": ["p"]}),
        var=pd.DataFrame(index=["g0", "g1", "g2"]))
    out64 = _to_linear(a64, "lognorm")
    assert out64.X.dtype == np.float64
    np.testing.assert_array_equal(np.asarray(out64.X), np.expm1(x64))   # bit-identical
    aint = ad.AnnData(
        X=csr_matrix(np.array([[1, 0], [0, 2]], dtype=np.int64)),
        obs=pd.DataFrame({"target": ["x", "y"]}),
        var=pd.DataFrame(index=["g0", "g1"]))
    outint = _to_linear(aint, "lognorm")
    assert outint.X.dtype == np.float64
    np.testing.assert_array_equal(
        outint.X.toarray(), np.expm1(np.array([[1.0, 0.0], [0.0, 2.0]])))


def test_compute_de_missing_reference_raises_clear_error(synthetic_counts_pair):
    # A reference absent from the groupby column must raise a clear ValueError, not a
    # cryptic KeyError from compute_lfc_table's means[reference] (Gemini MEDIUM on PR #10).
    _pred, real = synthetic_counts_pair
    with pytest.raises(ValueError, match="reference group 'ABSENT' not found"):
        compute_de(real, backend="scanpy", groupby="target", reference="ABSENT",
                   mean_calc="arithmetic", epsilon=1e-9, input_type="counts",
                   filter_gene_min_cpm_cell=None)


def test_compute_de_rejects_unknown_input_type_and_mean_calc(synthetic_counts_pair):
    # A typo'd convention would otherwise silently misroute and emit wrong numbers
    # (Copilot on PR #10): unknown input_type -> expm1 path, unknown mean_calc -> arithmetic.
    _pred, real = synthetic_counts_pair
    base = dict(backend="scanpy", groupby="target", reference="non-targeting",
                epsilon=1e-9, filter_gene_min_cpm_cell=None)
    with pytest.raises(ValueError, match="input_type must be"):
        compute_de(real, mean_calc="arithmetic", input_type="lognrm", **base)
    with pytest.raises(ValueError, match="mean_calc must be"):
        compute_de(real, mean_calc="geomtric", input_type="counts", **base)


def test_fdr_scope_global_pools_all_perturbations(synthetic_counts_pair):
    real, _ = synthetic_counts_pair
    df = compute_de(
        real,
        backend="pdex",
        groupby="target",
        reference="non-targeting",
        mean_calc="geometric",
        epsilon=0.0,
        input_type="counts",
        target_sum=None,
        clip_value=20.0,
        filter_gene_min_cpm_cell=None,
        fdr_scope="global",
    )
    expected = false_discovery_control(df["p_value"].to_numpy().astype(float), method="bh")
    assert np.allclose(df["p_adj"].to_numpy(), expected, equal_nan=True)  # ONE global pool


def test_fdr_scope_per_pert_is_per_target(synthetic_counts_pair):
    real, _ = synthetic_counts_pair
    df = compute_de(
        real,
        backend="pdex",
        groupby="target",
        reference="non-targeting",
        mean_calc="arithmetic",
        epsilon=1e-9,
        input_type="counts",
        filter_gene_min_cpm_cell=None,
        fdr_scope="per_pert",
    )
    # per-target BH: recomputing within each target reproduces p_adj, global does not (in general)
    for t in df["target"].unique().to_list():
        sub = df.filter(pl.col("target") == t)
        exp = false_discovery_control(sub["p_value"].to_numpy().astype(float), method="bh")
        assert np.allclose(sub["p_adj"].to_numpy(), exp, equal_nan=True)


def test_compute_lfc_table_edge_cases():
    # Public helper (tests import it directly): clear error on a missing reference,
    # typed-empty frame when only the reference group is present (no pl.concat([]) crash).
    from cell_eval2.de_compute import compute_lfc_table
    only_ref = ad.AnnData(
        X=np.array([[1.0, 2.0], [3.0, 4.0]]),
        obs=pd.DataFrame({"target": ["non-targeting", "non-targeting"]}, index=["c0", "c1"]),
        var=pd.DataFrame(index=["g0", "g1"]))
    empty = compute_lfc_table(only_ref, groupby="target", reference="non-targeting",
                              mean_calc="arithmetic", epsilon=1e-9)
    assert empty.height == 0
    assert empty.columns == ["target", "feature", "log2_fold_change"]
    no_ref = ad.AnnData(
        X=np.array([[1.0, 2.0], [3.0, 4.0]]),
        obs=pd.DataFrame({"target": ["GENE1", "GENE1"]}, index=["c0", "c1"]),
        var=pd.DataFrame(index=["g0", "g1"]))
    with pytest.raises(ValueError, match="reference group 'non-targeting' not found"):
        compute_lfc_table(no_ref, groupby="target", reference="non-targeting",
                          mean_calc="arithmetic", epsilon=1e-9)
    # Unknown mean_calc must raise at the branch point, not silently fall through to
    # arithmetic when the helper is called directly (Copilot re-review on PR #10).
    with pytest.raises(ValueError, match="mean_calc must be"):
        compute_lfc_table(only_ref, groupby="target", reference="non-targeting",
                          mean_calc="geomtric", epsilon=1e-9)


def test_scanpy_backend_returns_canonical_schema(synthetic_counts_pair):
    _pred, real = synthetic_counts_pair
    df = compute_de(real, backend="scanpy", groupby="target",
                    reference="non-targeting", mean_calc="geometric",
                    epsilon=1e-9, input_type="counts",
                    filter_gene_min_cpm_cell=None)
    assert isinstance(df, pl.DataFrame)
    assert all(c in df.columns for c in REQUIRED_COLS)        # target, feature, log2_fold_change, p_adj
    assert set(df["target"].unique()) == {"GENE1", "GENE2", "GENE3"}  # control excluded
    assert df.height > 0


def test_explicit_unavailable_backend_raises(monkeypatch):
    # Force the backend unavailable so the assertion is deterministic regardless of the
    # runner: gpudge IS importable + usable on a GPU node (where this suite runs), so a
    # bare backend="gpudge" no longer fails at resolution there. Monkeypatch _available so
    # the test exercises the real "explicitly-requested unavailable backend -> raise" path.
    import cell_eval2.de_compute as dc

    monkeypatch.setattr(dc, "_available", lambda backend: False)
    with pytest.raises((ImportError, RuntimeError)):
        compute_de(None, backend="gpudge", groupby="target",
                   reference="non-targeting", mean_calc="geometric",
                   epsilon=1e-9, input_type="counts",
                   filter_gene_min_cpm_cell=None)


def test_pdex_backend_returns_canonical_schema(synthetic_counts_pair):
    _pred, real = synthetic_counts_pair
    df = compute_de(real, backend="pdex", groupby="target",
                    reference="non-targeting", mean_calc="geometric",
                    epsilon=1e-9, input_type="counts",
                    filter_gene_min_cpm_cell=None)
    assert all(c in df.columns for c in REQUIRED_COLS)
    assert "p_adj" in df.columns and "fdr" not in df.columns   # fdr->p_adj normalized
    assert set(df["target"].unique()) == {"GENE1", "GENE2", "GENE3"}


def test_pdex_and_scanpy_lfc_are_identical(synthetic_counts_pair):
    # LFC is cell_eval2's own (same code, same array) -> pdex-path and scanpy-path
    # log2_fold_change must be exactly equal for every shared (target, feature).
    _pred, real = synthetic_counts_pair
    kw = dict(groupby="target", reference="non-targeting", mean_calc="arithmetic",
              epsilon=1e-9, input_type="counts", filter_gene_min_cpm_cell=None)
    p = compute_de(real, backend="pdex", **kw).sort(["target", "feature"])
    s = compute_de(real, backend="scanpy", **kw).sort(["target", "feature"])
    j = p.select(["target", "feature", "log2_fold_change"]).join(
        s.select(["target", "feature", "log2_fold_change"]),
        on=["target", "feature"], suffix="_s")
    assert j.height > 0
    assert (j["log2_fold_change"] == j["log2_fold_change_s"]).all()


def test_gpudge_lognorm_input_converted_to_linear(monkeypatch, synthetic_pair):
    # gpudge assumes linear input; lognorm must be expm1'd to linear before gpudge sees it
    # (spec §4; Gemini HIGH on PR #9). Stub gpudge.de and assert the handed-in X is linear.
    import sys
    import types
    import numpy as np
    _pred, real = synthetic_pair  # lognorm
    seen = {}

    def fake_de(adata, *, groupby, reference, mean_calc, epsilon,
                cpm_normalize, filter_gene_min_cpm_cell, normalize_target_sum=None):
        seen["X"] = np.asarray(adata.X).copy()
        seen["cpm_normalize"] = cpm_normalize
        return pl.DataFrame({"target": ["GENE1"], "feature": ["g0"],
                             "log2_fold_change": [1.0], "p_value": [0.01], "p_adj": [0.02]})

    fake_mod = types.ModuleType("gpudge")
    fake_mod.de = fake_de
    monkeypatch.setitem(sys.modules, "gpudge", fake_mod)
    from cell_eval2 import de_compute
    monkeypatch.setattr(de_compute, "_resolve_backend", lambda b: "gpudge")
    de_compute.compute_de(real, backend="gpudge", groupby="target",
                          reference="non-targeting", mean_calc="arithmetic", epsilon=1e-9,
                          input_type="lognorm", filter_gene_min_cpm_cell=None)
    assert seen["cpm_normalize"] is False                       # lognorm -> no internal CPM
    assert np.allclose(seen["X"], np.expm1(np.asarray(real.X)))  # handed linear (expm1), not lognorm


def test_gpudge_counts_median_routes_cpm_normalize_false(monkeypatch, synthetic_counts_pair):
    # v1 (target_sum=None = median): gpudge has no target_sum knob, so compute_de must
    # pre-normalize to median on the CPU and pass cpm_normalize=False (#21). Assert the stub
    # sees a median-normalized X (NOT raw counts) and cpm_normalize=False.
    import sys
    import types
    import numpy as np
    import polars as pl  # also module-level in tests/test_de_compute.py; local for snippet self-containedness
    from cell_eval2.de_compute import _to_linear
    _pred, real = synthetic_counts_pair  # dense float32 counts
    seen = {}

    def fake_de(adata, *, groupby, reference, mean_calc, epsilon,
                cpm_normalize, filter_gene_min_cpm_cell, normalize_target_sum=None):
        seen["X"] = np.asarray(adata.X).copy()
        seen["cpm_normalize"] = cpm_normalize
        return pl.DataFrame({"target": ["GENE1"], "feature": ["g0"],
                             "log2_fold_change": [1.0], "p_value": [0.01], "p_adj": [0.02]})

    fake_mod = types.ModuleType("gpudge")
    fake_mod.de = fake_de
    monkeypatch.setitem(sys.modules, "gpudge", fake_mod)
    from cell_eval2 import de_compute
    monkeypatch.setattr(de_compute, "_resolve_backend", lambda b: "gpudge")
    de_compute.compute_de(real, backend="gpudge", groupby="target",
                          reference="non-targeting", mean_calc="geometric", epsilon=0.0,
                          input_type="counts", target_sum=None,
                          filter_gene_min_cpm_cell=None)
    assert seen["cpm_normalize"] is False                       # median -> no internal CPM
    expected = np.asarray(_to_linear(real, "counts", target_sum=None).X)
    assert np.allclose(seen["X"], expected)                     # handed median-normalized, not raw counts


def test_gpudge_counts_custom_target_sum_prenormalizes(monkeypatch, synthetic_counts_pair):
    # Any counts target_sum other than 1e6 must also pre-normalize (gpudge can only do CPM
    # natively). Confirms the fix generalizes beyond v1's median (#21).
    import sys
    import types
    import numpy as np
    import polars as pl  # also module-level in tests/test_de_compute.py; local for snippet self-containedness
    from cell_eval2.de_compute import _to_linear
    _pred, real = synthetic_counts_pair
    seen = {}

    def fake_de(adata, *, groupby, reference, mean_calc, epsilon,
                cpm_normalize, filter_gene_min_cpm_cell, normalize_target_sum=None):
        seen["X"] = np.asarray(adata.X).copy()
        seen["cpm_normalize"] = cpm_normalize
        return pl.DataFrame({"target": ["GENE1"], "feature": ["g0"],
                             "log2_fold_change": [1.0], "p_value": [0.01], "p_adj": [0.02]})

    fake_mod = types.ModuleType("gpudge")
    fake_mod.de = fake_de
    monkeypatch.setitem(sys.modules, "gpudge", fake_mod)
    from cell_eval2 import de_compute
    monkeypatch.setattr(de_compute, "_resolve_backend", lambda b: "gpudge")
    de_compute.compute_de(real, backend="gpudge", groupby="target",
                          reference="non-targeting", mean_calc="arithmetic", epsilon=1e-9,
                          input_type="counts", target_sum=5e5,
                          filter_gene_min_cpm_cell=None)
    assert seen["cpm_normalize"] is False
    expected = np.asarray(_to_linear(real, "counts", target_sum=5e5).X)
    assert np.allclose(seen["X"], expected)


def test_gpudge_counts_norm_mapping():
    # The (cpm_normalize, normalize_target_sum) mapping the native path shares with
    # compute_de_streaming (#142): 1e6 -> CPM, None -> median, else -> that library size.
    from cell_eval2.de_compute import _gpudge_counts_norm
    assert _gpudge_counts_norm(1e6) == (True, None)
    assert _gpudge_counts_norm(1e4) == (False, 1e4)
    assert _gpudge_counts_norm(None) == (False, "median")
    assert _gpudge_counts_norm(500.0) == (False, 500.0)


def test_gpudge_counts_norm_rejects_bad_target():
    # native path fails loud on a bad target_sum instead of coercing it into gpudge's on-GPU
    # normalize (Copilot #143): bool / non-finite / non-positive.
    from cell_eval2.de_compute import _gpudge_counts_norm
    for bad in (0.0, -1.0, float("nan"), float("inf"), True):
        with pytest.raises(ValueError, match="positive, finite"):
            _gpudge_counts_norm(bad)


def test_gpudge_native_normalize_counts_hands_raw_and_target(monkeypatch, synthetic_counts_pair):
    # native_gpu_normalize=True (issue #142): gpudge normalizes RAW counts on-GPU, so compute_de
    # hands it the raw counts (NOT _to_linear'd) + cpm_normalize=False + normalize_target_sum=N
    # for a non-1e6 target. Contrast with test_gpudge_counts_custom_target_sum_prenormalizes,
    # which is the default (flag-off) CPU-prenormalize path.
    import sys
    import types
    import numpy as np
    _pred, real = synthetic_counts_pair
    seen = {}

    def fake_de(adata, *, groupby, reference, mean_calc, epsilon,
                cpm_normalize, filter_gene_min_cpm_cell, normalize_target_sum=None):
        seen["X"] = np.asarray(adata.X).copy()
        seen["cpm_normalize"] = cpm_normalize
        seen["normalize_target_sum"] = normalize_target_sum
        return pl.DataFrame({"target": ["GENE1"], "feature": ["g0"],
                             "log2_fold_change": [1.0], "p_value": [0.01], "p_adj": [0.02]})

    fake_mod = types.ModuleType("gpudge")
    fake_mod.de = fake_de
    monkeypatch.setitem(sys.modules, "gpudge", fake_mod)
    from cell_eval2 import de_compute
    monkeypatch.setattr(de_compute, "_resolve_backend", lambda b: "gpudge")
    de_compute.compute_de(real, backend="gpudge", groupby="target",
                          reference="non-targeting", mean_calc="arithmetic", epsilon=1e-9,
                          input_type="counts", target_sum=1e4,
                          filter_gene_min_cpm_cell=None, native_gpu_normalize=True)
    assert seen["cpm_normalize"] is False
    assert seen["normalize_target_sum"] == 1e4                     # gpudge CPMs on-GPU
    assert np.allclose(seen["X"], np.asarray(real.X))             # RAW counts, not pre-normalized


def test_gpudge_native_normalize_lognorm_still_expm1s(monkeypatch, synthetic_pair):
    # native_gpu_normalize applies to counts only; lognorm still expm1's via _to_linear (gpudge
    # can't invert log1p) and passes normalize_target_sum=None.
    import sys
    import types
    import numpy as np
    _pred, real = synthetic_pair  # lognorm
    seen = {}

    def fake_de(adata, *, groupby, reference, mean_calc, epsilon,
                cpm_normalize, filter_gene_min_cpm_cell, normalize_target_sum=None):
        seen["X"] = np.asarray(adata.X).copy()
        seen["normalize_target_sum"] = normalize_target_sum
        return pl.DataFrame({"target": ["GENE1"], "feature": ["g0"],
                             "log2_fold_change": [1.0], "p_value": [0.01], "p_adj": [0.02]})

    fake_mod = types.ModuleType("gpudge")
    fake_mod.de = fake_de
    monkeypatch.setitem(sys.modules, "gpudge", fake_mod)
    from cell_eval2 import de_compute
    monkeypatch.setattr(de_compute, "_resolve_backend", lambda b: "gpudge")
    de_compute.compute_de(real, backend="gpudge", groupby="target",
                          reference="non-targeting", mean_calc="arithmetic", epsilon=1e-9,
                          input_type="lognorm", filter_gene_min_cpm_cell=None,
                          native_gpu_normalize=True)
    assert seen["normalize_target_sum"] is None                   # counts-only; lognorm bypasses
    assert np.allclose(seen["X"], np.expm1(np.asarray(real.X)))   # still expm1'd to linear


def test_scanpy_pvalues_match_direct_rank_genes_groups(synthetic_counts_pair):
    # TIGHT layer: our scanpy-backend p-values reproduce a direct scanpy reference call
    # (guards our plumbing; same MWU implementation, so essentially exact).
    import scanpy as sc
    from cell_eval2.de_compute import _cpm_log1p
    _pred, real = synthetic_counts_pair
    ours = compute_de(real, backend="scanpy", groupby="target",
                      reference="non-targeting", mean_calc="arithmetic", epsilon=1e-9,
                      input_type="counts", filter_gene_min_cpm_cell=None
                      ).sort(["target", "feature"])
    a = _cpm_log1p(real)
    sc.tl.rank_genes_groups(a, groupby="target", reference="non-targeting",
                            method="wilcoxon", tie_correct=True, n_genes=a.n_vars)
    res = a.uns["rank_genes_groups"]
    ref = pl.concat([
        pl.DataFrame({"target": g, "feature": np.asarray(res["names"][g], dtype=str),
                      "p_value": np.asarray(res["pvals"][g], dtype=float)})
        for g in res["names"].dtype.names]).sort(["target", "feature"])
    j = ours.select(["target", "feature", "p_value"]).join(
        ref, on=["target", "feature"], suffix="_ref")
    assert np.allclose(j["p_value"].to_numpy(), j["p_value_ref"].to_numpy(), rtol=1e-9, atol=0)


def test_pdex_vs_scanpy_significant_gene_jaccard(synthetic_counts_pair):
    # JACCARD layer: different MWU implementations agree on the significant-gene set.
    _pred, real = synthetic_counts_pair
    kw = dict(groupby="target", reference="non-targeting", mean_calc="arithmetic",
              epsilon=1e-9, input_type="counts", filter_gene_min_cpm_cell=None)
    p = _sig_sets(compute_de(real, backend="pdex", **kw))
    s = _sig_sets(compute_de(real, backend="scanpy", **kw))
    jac = [_jaccard(p[t], s[t]) for t in p]
    # Measured fixture min = 1.0; keep the bound just under the measured minimum.
    assert min(jac) >= 0.99, f"pdex vs scanpy sig-gene Jaccard too low: {jac}"


def test_counts_vs_lognorm_consistent_lfc_and_pvalues(synthetic_counts_pair):
    # Feed the SAME data as counts, and as its lognorm (CPM+log1p) transform. LFC and
    # raw p-values must be consistent (CPM is the operative normalization; log1p is monotone).
    from cell_eval2.de_compute import _cpm_log1p
    _pred, real = synthetic_counts_pair
    as_counts = compute_de(real, backend="scanpy", groupby="target",
                           reference="non-targeting", mean_calc="arithmetic", epsilon=1e-9,
                           input_type="counts", filter_gene_min_cpm_cell=None
                           ).sort(["target", "feature"])
    real_log = _cpm_log1p(real)  # CPM+log1p == lognorm of the same library normalization
    as_log = compute_de(real_log, backend="scanpy", groupby="target",
                        reference="non-targeting", mean_calc="arithmetic", epsilon=1e-9,
                        input_type="lognorm", filter_gene_min_cpm_cell=None
                        ).sort(["target", "feature"])
    j = as_counts.join(as_log, on=["target", "feature"], suffix="_log")
    assert np.allclose(j["log2_fold_change"].to_numpy(),
                       j["log2_fold_change_log"].to_numpy(), rtol=1e-6, atol=1e-9)
    assert np.allclose(j["p_value"].to_numpy(),
                       j["p_value_log"].to_numpy(), rtol=1e-9, atol=1e-12)


def test_cpm_filter_skipped_on_lognorm_changes_padj_universe(synthetic_counts_pair, caplog):
    # The CPM filter needs counts; on lognorm it skips (info-log) -> the BH universe is
    # larger -> p_adj differs from the counts path even when raw p-values match.
    import logging
    from cell_eval2.de_compute import _cpm_log1p
    _pred, real = synthetic_counts_pair
    counts_filt = compute_de(real, backend="scanpy", groupby="target",
                             reference="non-targeting", mean_calc="arithmetic", epsilon=1e-9,
                             input_type="counts", filter_gene_min_cpm_cell=5.0)
    with caplog.at_level(logging.INFO, logger="cell_eval2.de_compute"):
        log_filt = compute_de(_cpm_log1p(real), backend="scanpy", groupby="target",
                              reference="non-targeting", mean_calc="arithmetic", epsilon=1e-9,
                              input_type="lognorm", filter_gene_min_cpm_cell=5.0)
    assert any("skipping the CPM gate" in r.message for r in caplog.records)
    # Filter skipped on lognorm -> more rows retained than the counts-filtered path.
    assert log_filt.height >= counts_filt.height


def test_gpudge_adapter_forwards_args(monkeypatch, synthetic_counts_pair):
    import sys
    import types
    _pred, real = synthetic_counts_pair
    captured = {}

    def fake_de(adata, *, groupby, reference, mean_calc, epsilon,
                cpm_normalize, filter_gene_min_cpm_cell, normalize_target_sum=None):
        captured.update(groupby=groupby, reference=reference, mean_calc=mean_calc,
                        epsilon=epsilon, cpm_normalize=cpm_normalize,
                        filter_gene_min_cpm_cell=filter_gene_min_cpm_cell,
                        normalize_target_sum=normalize_target_sum)
        return pl.DataFrame({
            "target": ["GENE1"], "feature": ["g0"], "log2_fold_change": [1.0],
            "p_value": [0.01], "p_adj": [0.02],
        })

    fake_mod = types.ModuleType("gpudge")
    fake_mod.de = fake_de
    monkeypatch.setitem(sys.modules, "gpudge", fake_mod)

    from cell_eval2 import de_compute
    monkeypatch.setattr(de_compute, "_resolve_backend", lambda b: "gpudge")
    df = de_compute.compute_de(real, backend="gpudge", groupby="target",
                               reference="non-targeting", mean_calc="geometric",
                               epsilon=1e-9, input_type="counts",
                               filter_gene_min_cpm_cell=5.0)
    assert captured == {"groupby": "target", "reference": "non-targeting",
                        "mean_calc": "geometric", "epsilon": 1e-9,
                        "cpm_normalize": True,
                        # ⚠️ None, not the requested 5.0, and the gate is still APPLIED -- just by
                        # cell_eval2 rather than by gpudge (#351). This call is geometric, so
                        # `_gpudge_gate_plan` takes its MATRIX route (the frame's `ref_mean` is the
                        # geometric control mean there, not the arithmetic one gpudge gates on and
                        # never returns), and on that route gpudge's own `(target OR ref)` mask is
                        # muted so it cannot drop a boundary row for one target that cell_eval2
                        # keeps for another. Every other argument still forwards verbatim, which is
                        # what this test is for.
                        "filter_gene_min_cpm_cell": None,
                        "normalize_target_sum": None}  # counts@1e6 native CPM (default path)
    assert "abs_log2_fold_change" in df.columns   # normalize_de_schema ran


def _stub_gpudge_de(monkeypatch, frame):
    """Install a fake gpudge module whose de() returns `frame`, and force the
    gpudge backend. Mirrors the existing gpudge-stub tests."""
    import sys
    import types

    def fake_de(adata, *, groupby, reference, mean_calc, epsilon,
                cpm_normalize, filter_gene_min_cpm_cell, normalize_target_sum=None):
        return frame

    fake_mod = types.ModuleType("gpudge")
    fake_mod.de = fake_de
    monkeypatch.setitem(sys.modules, "gpudge", fake_mod)
    from cell_eval2 import de_compute
    monkeypatch.setattr(de_compute, "_resolve_backend", lambda b: "gpudge")
    return de_compute


def test_gpudge_v1_clip_applied(monkeypatch, synthetic_counts_pair):
    # gpudge with v1 epsilon=0 emits +/-inf/NaN on zero-mean genes; the v1 clip
    # must map them to pdex's finite clip values, reusing gpudge's own means.
    _pred, real = synthetic_counts_pair
    frame = pl.DataFrame({
        "target": ["GENE1"] * 5,
        "feature": ["g0", "g1", "g2", "g3", "g4"],
        "target_mean": [2.0, 0.0, 0.0, 4.0, 5.0],
        "ref_mean":    [1.0, 3.0, 0.0, 2.0, 0.0],
        "log2_fold_change": [1.0, float("-inf"), float("nan"), 1.0, float("inf")],
        "p_value": [0.01, 0.2, 0.5, 0.02, 0.03],
        "p_adj":   [0.02, 0.3, 0.6, 0.03, 0.04],
    })
    de_compute = _stub_gpudge_de(monkeypatch, frame)
    df = de_compute.compute_de(
        real, backend="gpudge", groupby="target", reference="non-targeting",
        mean_calc="geometric", epsilon=0.0, input_type="counts",
        clip_value=20.0, filter_gene_min_cpm_cell=None, fdr_scope="global")  # v1's real fdr_scope
    lfc = dict(zip(df["feature"].to_list(), df["log2_fold_change"].to_list()))
    assert np.isfinite(np.array(list(lfc.values()))).all()    # no inf/NaN survive
    assert np.isclose(lfc["g0"], np.log2(2.0 / 1.0))          # finite -> log2(mt/ref)
    assert np.isclose(lfc["g1"], np.log2(1.0 / 20.0))         # tgt0 -> 1/clip
    assert np.isclose(lfc["g2"], 0.0)                         # both0 -> ratio 1 -> log2(1)=0
    assert np.isclose(lfc["g3"], np.log2(4.0 / 2.0))          # finite
    assert np.isclose(lfc["g4"], np.log2(20.0))               # ref0 -> clip
    # abs_log2_fold_change derives from the clipped LFC
    abs_g4 = dict(zip(df["feature"].to_list(), df["abs_log2_fold_change"].to_list()))["g4"]
    assert np.isclose(abs_g4, np.log2(20.0))


def test_gpudge_v2_no_clip_passthrough(monkeypatch, synthetic_counts_pair):
    # clip_value=None (v2) -> gpudge's native LFC column is passed through unchanged.
    _pred, real = synthetic_counts_pair
    frame = pl.DataFrame({
        "target": ["GENE1"] * 3,
        "feature": ["g0", "g1", "g2"],
        "target_mean": [2.0, 0.0, 0.0],
        "ref_mean":    [1.0, 3.0, 0.0],
        "log2_fold_change": [1.0, float("-inf"), float("nan")],
        "p_value": [0.01, 0.2, 0.5],
        "p_adj":   [0.02, 0.3, 0.6],
    })
    de_compute = _stub_gpudge_de(monkeypatch, frame)
    df = de_compute.compute_de(
        real, backend="gpudge", groupby="target", reference="non-targeting",
        mean_calc="arithmetic", epsilon=1e-9, input_type="counts",
        clip_value=None, filter_gene_min_cpm_cell=None, fdr_scope="per_pert")
    lfc = dict(zip(df["feature"].to_list(), df["log2_fold_change"].to_list()))
    assert lfc["g0"] == 1.0
    assert np.isneginf(lfc["g1"])     # unchanged
    assert np.isnan(lfc["g2"])        # unchanged


def test_gpudge_v1_clip_requires_mean_columns(monkeypatch, synthetic_counts_pair):
    # If gpudge ever omits target_mean/ref_mean, the v1 clip path fails loudly.
    _pred, real = synthetic_counts_pair
    frame = pl.DataFrame({
        "target": ["GENE1"], "feature": ["g0"],
        "log2_fold_change": [float("inf")], "p_value": [0.01], "p_adj": [0.02],
    })
    de_compute = _stub_gpudge_de(monkeypatch, frame)
    with pytest.raises(ValueError, match="target_mean"):
        de_compute.compute_de(
            real, backend="gpudge", groupby="target", reference="non-targeting",
            mean_calc="geometric", epsilon=0.0, input_type="counts",
            clip_value=20.0, filter_gene_min_cpm_cell=None, fdr_scope="per_pert")


def test_gpudge_lfc_matches_cpu_reference_when_cuda():
    # Permanent regression guard; runs only where CUDA + gpudge are present
    # (locally skipped; exercised on a slurm GPU node -- see internal:tools/gpudge_parity.sbatch).
    torch = pytest.importorskip("torch")
    gpudge = pytest.importorskip("gpudge")
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device")
    import anndata as ad
    import numpy as np
    import pandas as pd
    import scanpy as sc
    rng = np.random.default_rng(1)
    blocks, labels = [], []
    for p in ("non-targeting", "GENE1", "GENE2", "GENE3"):
        blocks.append(rng.poisson(3.0, size=(40, 30)))
        labels += [p] * 40
    X = np.vstack(blocks).astype(np.float32)
    adata = ad.AnnData(
        X=X, obs=pd.DataFrame({"target": labels}, index=[f"c{i}" for i in range(X.shape[0])]),
        var=pd.DataFrame(index=[f"g{j}" for j in range(30)]))
    g = gpudge.de(adata, groupby="target", reference="non-targeting",
                  mean_calc="arithmetic", epsilon=1e-9, cpm_normalize=True,
                  filter_gene_min_cpm_cell=None).to_pandas()
    a = adata.copy()
    sc.pp.normalize_total(a, target_sum=1e6)
    Xl = np.asarray(a.X, dtype=float)
    lab = a.obs["target"].to_numpy().astype(str)
    means = {gp: Xl[lab == gp].mean(axis=0) for gp in np.unique(lab)}
    ref = means["non-targeting"]
    for _, r in g.iterrows():
        exp = np.log2((means[r["target"]][int(r["feature"][1:])] + 1e-9) / (ref[int(r["feature"][1:])] + 1e-9))
        assert abs(r["log2_fold_change"] - exp) < 1e-4


def test_auto_prefers_pdex_then_scanpy_without_gpu(monkeypatch):
    from cell_eval2 import de_compute
    avail = {"gpudge": False, "pdex": True, "scanpy": True}
    monkeypatch.setattr(de_compute, "_available", lambda b: avail[b])
    # The test's name says "without_gpu": pin the new CUDA probe too, or it raises on a GPU node.
    monkeypatch.setattr(de_compute, "_cuda_device_present", lambda: False)
    assert de_compute._resolve_backend("auto") == "pdex"
    avail["pdex"] = False
    assert de_compute._resolve_backend("auto") == "scanpy"
    avail["scanpy"] = False
    with pytest.raises(RuntimeError):
        de_compute._resolve_backend("auto")


def test_cpm_filter_drops_rows_and_recomputes_bh(synthetic_counts_pair):
    _pred, real = synthetic_counts_pair
    base = compute_de(real, backend="scanpy", groupby="target",
                      reference="non-targeting", mean_calc="geometric",
                      epsilon=1e-9, input_type="counts", filter_gene_min_cpm_cell=None)
    # A high threshold drops most genes; survivors get BH recomputed over a smaller set.
    filtered = compute_de(real, backend="scanpy", groupby="target",
                          reference="non-targeting", mean_calc="geometric",
                          epsilon=1e-9, input_type="counts",
                          filter_gene_min_cpm_cell=1e6)
    assert filtered.height < base.height


def test_cpm_filter_skipped_on_lognorm_input(synthetic_pair):
    _pred, real = synthetic_pair   # lognorm
    # filter active but input is lognorm -> skipped (no error); same rows as no filter.
    a = compute_de(real, backend="scanpy", groupby="target",
                   reference="non-targeting", mean_calc="geometric",
                   epsilon=1e-9, input_type="lognorm", filter_gene_min_cpm_cell=5.0)
    b = compute_de(real, backend="scanpy", groupby="target",
                   reference="non-targeting", mean_calc="geometric",
                   epsilon=1e-9, input_type="lognorm", filter_gene_min_cpm_cell=None)
    assert a.height == b.height


def test_unknown_backend_raises_valueerror():
    # An unknown backend name must give a clear ValueError, not a bare KeyError (Copilot #3).
    with pytest.raises(ValueError, match="unknown DE backend"):
        compute_de(None, backend="badbackend", groupby="target",
                   reference="non-targeting", mean_calc="geometric",
                   epsilon=1e-9, input_type="counts", filter_gene_min_cpm_cell=None)


def test_only_reference_group_raises_clear_error():
    # Data with only the reference group (no perturbations to test) must raise a clear
    # ValueError, not a cryptic backend-internal error (Copilot #4).
    import anndata as ad
    import numpy as np
    import pandas as pd

    rng = np.random.default_rng(0)
    X = rng.poisson(3.0, size=(20, 6)).astype(np.float32)
    obs = pd.DataFrame({"target": ["non-targeting"] * 20},
                       index=[f"c{i}" for i in range(20)])
    var = pd.DataFrame(index=[f"g{j}" for j in range(6)])
    only_ctrl = ad.AnnData(X=X, obs=obs, var=var)
    with pytest.raises(ValueError, match="no non-reference groups"):
        compute_de(only_ctrl, backend="scanpy", groupby="target",
                   reference="non-targeting", mean_calc="geometric",
                   epsilon=1e-9, input_type="counts", filter_gene_min_cpm_cell=None)


def test_apply_cpm_filter_handles_empty_df():
    # An empty DE frame (e.g. no non-control groups survive upstream) must not crash
    # the per-target keep-frame concat. Regression for pl.concat on an empty list.
    import anndata as ad
    import numpy as np
    import pandas as pd

    from cell_eval2.de_compute import _apply_cpm_filter

    rng = np.random.default_rng(0)
    X = rng.poisson(3.0, size=(20, 5)).astype(np.float32)
    obs = pd.DataFrame({"target": ["non-targeting"] * 10 + ["GENE1"] * 10},
                       index=[f"c{i}" for i in range(20)])
    var = pd.DataFrame(index=[f"g{j}" for j in range(5)])
    adata = ad.AnnData(X=X, obs=obs, var=var)
    empty = pl.DataFrame({"target": [], "feature": [], "p_value": []},
                         schema={"target": pl.Utf8, "feature": pl.Utf8, "p_value": pl.Float64})
    out = _apply_cpm_filter(empty, adata, groupby="target",
                            reference="non-targeting", threshold=5.0)
    assert out.height == 0


def test_apply_cpm_filter_handles_nan_pvalue():
    # A NaN p-value among CPM-filter survivors must not crash the per-target BH recompute
    # (scipy's false_discovery_control rejects NaN); the NaN row keeps NaN p_adj while the
    # valid rows get a finite BH value (Gemini critical re-review finding).
    import anndata as ad
    import numpy as np
    import pandas as pd

    from cell_eval2.de_compute import _apply_cpm_filter

    X = (np.ones((20, 3)) * 1000.0).astype(np.float32)  # all genes survive any low threshold
    obs = pd.DataFrame({"target": ["non-targeting"] * 10 + ["GENE1"] * 10},
                       index=[f"c{i}" for i in range(20)])
    var = pd.DataFrame(index=["g0", "g1", "g2"])
    adata = ad.AnnData(X=X, obs=obs, var=var)
    df = pl.DataFrame({
        "target": ["GENE1", "GENE1", "GENE1"],
        "feature": ["g0", "g1", "g2"],
        "log2_fold_change": [1.0, 2.0, 0.5],
        "p_value": [0.01, float("nan"), 0.2],
        "p_adj": [0.02, float("nan"), 0.3],
    })
    out = _apply_cpm_filter(df, adata, groupby="target",
                            reference="non-targeting", threshold=0.0)
    padj = {r["feature"]: r["p_adj"] for r in out.iter_rows(named=True)}
    assert np.isnan(padj["g1"])                              # NaN p stays NaN (non-significant)
    assert np.isfinite(padj["g0"]) and np.isfinite(padj["g2"])


def test_gate_keepset_matches_manual_cpm():
    # Independent reference for the expression gate: a mixed-expression counts adata
    # (some genes well above the 5-CPM gate, some well below) where the manual per-cell
    # CPM (diags(1e6/L)@X) and scanpy normalize_total agree on the kept set. compute_de's
    # filtered output must keep exactly that set. Guards the gate through the CPM-source
    # change (manual -> scanpy) introduced later in this plan.
    import numpy as np
    import pandas as pd
    import anndata as ad

    rng = np.random.default_rng(7)
    n = 30
    hi = rng.poisson(50, size=(2 * n, 3))            # g0..g2: high expression -> kept
    lo = np.zeros((2 * n, 3), dtype=float)           # g3..g5: zero -> below 5 CPM -> dropped
    X = np.hstack([hi, lo]).astype(np.float64)
    labels = ["non-targeting"] * n + ["GENE1"] * n
    adata = ad.AnnData(
        X=X,
        obs=pd.DataFrame({"target": labels}, index=[f"c{i}" for i in range(2 * n)]),
        var=pd.DataFrame(index=[f"g{j}" for j in range(6)]),
    )

    thr = 5.0
    out = compute_de(adata, backend="scanpy", groupby="target", reference="non-targeting",
                     mean_calc="arithmetic", epsilon=1e-9, input_type="counts",
                     filter_gene_min_cpm_cell=thr)
    kept = set(zip(out["target"].to_list(), out["feature"].to_list()))

    # Manual reference: per-cell CPM = X/L*1e6, per-group arithmetic mean, keep (target,gene)
    # if target-mean OR ref-mean CPM > thr.
    lab = adata.obs["target"].to_numpy().astype(str)
    genes = np.asarray(adata.var_names, dtype=str)
    L = X.sum(axis=1)
    cpm = X * (1e6 / np.where(L == 0, 1.0, L))[:, None]
    means = {g: cpm[lab == g].mean(axis=0) for g in np.unique(lab)}
    ref = means["non-targeting"]
    expected = {
        ("GENE1", gene)
        for gi, gene in enumerate(genes)
        if (means["GENE1"][gi] > thr) or (ref[gi] > thr)
    }
    assert kept == expected
    assert {f for (_t, f) in kept} == {"g0", "g1", "g2"}   # low-expression genes dropped


def test_cpm_gate_keepset_is_true_cpm_independent_of_target_sum():
    # F4.1: the CPU CPM gate must threshold TRUE CPM (per 1e6), so its kept-gene set is INDEPENDENT
    # of the LFC target_sum. Before the fix it thresholded target_sum-normalized means, so a non-1e6
    # target_sum silently kept/dropped different genes than gpudge (which recovers true CPM), making
    # the DEG set + BH universe backend-dependent. A moderate gene (true CPM ~200) is kept at the
    # 5-CPM threshold, but its target_sum=1e4-normalized mean (~2) would fall below 5 without rescaling.
    import numpy as np
    import pandas as pd
    import anndata as ad

    rng = np.random.default_rng(11)
    n = 40
    bg = rng.poisson(250, size=(2 * n, 40))        # ~10000 counts/cell -> large library
    mid = rng.poisson(2, size=(2 * n, 1))          # true CPM ~200: kept at thr=5, but ~2 per-1e4
    zero = np.zeros((2 * n, 1))
    X = np.hstack([bg, mid, zero]).astype(np.float64)
    genes = [f"bg{i}" for i in range(40)] + ["gmid", "gzero"]
    labels = ["non-targeting"] * n + ["GENE1"] * n
    adata = ad.AnnData(X=X, obs=pd.DataFrame({"target": labels}, index=[f"c{i}" for i in range(2 * n)]),
                       var=pd.DataFrame(index=genes))

    def keepset(ts):
        out = compute_de(adata, backend="scanpy", groupby="target", reference="non-targeting",
                         mean_calc="arithmetic", epsilon=1e-9, input_type="counts",
                         target_sum=ts, filter_gene_min_cpm_cell=5.0)
        return set(zip(out["target"].to_list(), out["feature"].to_list()))

    ks_1e6 = keepset(1e6)
    ks_1e4 = keepset(1e4)
    ks_none = keepset(None)   # median normalization: eff_target recovered from `linear` row-sums
    assert ("GENE1", "gmid") in ks_1e6, "sanity: gmid (true CPM ~200) is above the 5-CPM gate"
    assert ks_1e4 == ks_1e6, "CPU CPM gate keepset changed with target_sum (not thresholding true CPM)"
    assert ks_none == ks_1e6, "CPU CPM gate keepset changed under median (target_sum=None) normalization"
    assert ("GENE1", "gmid") in ks_1e4, "gmid wrongly dropped at target_sum=1e4 (gate not true CPM)"
    assert ("GENE1", "gzero") not in ks_1e4                # zero-expression gene dropped either way


def test_cpm_gate_rejects_nonpositive_or_nonfinite_target_sum():
    # F4.1 hardening (Gemini/Copilot #118): the true-CPM rescale divides by eff_target, so a direct
    # compute_de call (bypassing config validation) with a non-positive or non-finite target_sum must
    # raise a clear ValueError -- not a bare ZeroDivisionError (target_sum=0) nor a silently
    # degenerate all-NaN gate (target_sum=NaN, which otherwise completes with the wrong keepset).
    import numpy as np
    import pandas as pd
    import anndata as ad

    rng = np.random.default_rng(0)
    X = rng.poisson(50, size=(20, 6)).astype(float)
    adata = ad.AnnData(
        X=X, obs=pd.DataFrame({"target": ["non-targeting"] * 10 + ["GENE1"] * 10},
                              index=[f"c{i}" for i in range(20)]),
        var=pd.DataFrame(index=[f"g{j}" for j in range(6)]),
    )
    for bad in (0.0, float("nan")):
        with pytest.raises(ValueError, match="positive, finite"):
            compute_de(adata, backend="scanpy", groupby="target", reference="non-targeting",
                       mean_calc="arithmetic", epsilon=1e-9, input_type="counts",
                       target_sum=bad, filter_gene_min_cpm_cell=5.0)


def test_counts_cpu_path_normalizes_once(monkeypatch, synthetic_counts_pair):
    # The counts CPU path must call scanpy normalize_total exactly once (the single CPM
    # source feeds both the LFC means and the log1p p-value view). Guards against
    # re-introducing the double normalization.
    import scanpy as sc
    _pred, real = synthetic_counts_pair
    calls = {"n": 0}
    real_norm = sc.pp.normalize_total

    def counting(*a, **k):
        calls["n"] += 1
        return real_norm(*a, **k)

    monkeypatch.setattr(sc.pp, "normalize_total", counting)
    compute_de(real, backend="scanpy", groupby="target", reference="non-targeting",
               mean_calc="arithmetic", epsilon=1e-9, input_type="counts",
               filter_gene_min_cpm_cell=None)
    assert calls["n"] == 1


def test_de_scanpy_pvalues_takes_log_space_input(synthetic_counts_pair):
    # The engine takes PRE-LOG-NORMALIZED input directly (no internal CPM/log1p) and
    # returns the canonical p-value schema; its p-values match a direct rank_genes_groups
    # on the same log-space adata.
    import scanpy as sc
    from cell_eval2.de_compute import _cpm_log1p, _de_scanpy_pvalues
    _pred, real = synthetic_counts_pair
    log_adata = _cpm_log1p(real)  # counts -> CPM + log1p (log space)
    got = _de_scanpy_pvalues(log_adata, groupby="target", reference="non-targeting"
                             ).sort(["target", "feature"])
    assert got.columns == ["target", "feature", "p_value", "p_adj"]

    a = log_adata.copy()
    sc.tl.rank_genes_groups(a, groupby="target", reference="non-targeting",
                            method="wilcoxon", tie_correct=True, n_genes=a.n_vars)
    res = a.uns["rank_genes_groups"]
    ref = pl.concat([
        pl.DataFrame({"target": g, "feature": np.asarray(res["names"][g], dtype=str),
                      "p_value": np.asarray(res["pvals"][g], dtype=float)})
        for g in res["names"].dtype.names]).sort(["target", "feature"])
    j = got.join(ref, on=["target", "feature"], suffix="_ref")
    assert np.allclose(j["p_value"].to_numpy(), j["p_value_ref"].to_numpy(), rtol=1e-9, atol=0)


def test_de_pdex_pvalues_takes_log_space_input(synthetic_counts_pair):
    # pdex engine takes PRE-LOG-NORMALIZED input directly and returns the canonical schema.
    from cell_eval2.de_compute import _cpm_log1p, _de_pdex_pvalues
    _pred, real = synthetic_counts_pair
    log_adata = _cpm_log1p(real)
    got = _de_pdex_pvalues(log_adata, groupby="target", reference="non-targeting", threads=1)
    assert got.columns == ["target", "feature", "p_value", "p_adj"]
    assert got.height > 0


def _ref_group_means_linear(adata_linear, groupby, mean_calc):
    """Pre-change reference impl of _group_means_linear, inlined, to assert bit-identity."""
    from scipy.sparse import issparse
    X = adata_linear.X
    labels = adata_linear.obs[groupby].to_numpy().astype(str)
    out = {}
    for g in np.unique(labels):
        sub = X[np.where(labels == g)[0]]
        if mean_calc == "geometric":
            if issparse(sub):
                m = sub.copy()
                np.log1p(m.data, out=m.data)
                log_mean = np.asarray(m.mean(axis=0)).ravel()
            else:
                log_mean = np.log1p(np.asarray(sub, dtype=float)).mean(axis=0)
            out[g] = np.expm1(log_mean)
        else:
            out[g] = np.asarray(sub.mean(axis=0)).ravel()
    return out


@pytest.mark.parametrize("dense", [True, False])
@pytest.mark.parametrize("mean_calc", ["arithmetic", "geometric"])
def test_group_means_linear_bit_identical(dense, mean_calc):
    from scipy.sparse import csr_matrix
    from cell_eval2.de_compute import _group_means_linear
    rng = np.random.default_rng(3)
    Xv = np.abs(rng.standard_normal((1500, 40)))
    X = Xv if dense else csr_matrix(Xv * (rng.random((1500, 40)) < 0.3))
    labels = np.array([f"t{rng.integers(30)}" for _ in range(1500)]).astype(str)
    adata = ad.AnnData(X=X, obs=pd.DataFrame({"target": labels}))
    got = _group_means_linear(adata, "target", mean_calc)
    ref = _ref_group_means_linear(adata, "target", mean_calc)
    assert got.keys() == ref.keys()
    for k in ref:
        np.testing.assert_array_equal(got[k], ref[k])  # exact, not allclose



def test_scanpy_threads_notice_emitted_once(caplog):
    import logging
    from cell_eval2.de_compute import _notice_scanpy_ignores_threads
    _notice_scanpy_ignores_threads.cache_clear()
    try:
        with caplog.at_level(logging.INFO, logger="cell_eval2.de_compute"):
            _notice_scanpy_ignores_threads()
            _notice_scanpy_ignores_threads()
        hits = [r for r in caplog.records if "ignores num_threads" in r.message]
        assert len(hits) == 1  # functools.cache -> emitted once per process
    finally:
        _notice_scanpy_ignores_threads.cache_clear()  # don't leak cache state


def test_group_means_geometric_integer_sparse_does_not_raise():
    import numpy as np
    import anndata as ad
    import pandas as pd
    import scipy.sparse as sp
    from cell_eval2.de_compute import compute_lfc_table, _group_means_linear

    # Integer-dtype CSR (raw counts). The geometric mean path must cast to float,
    # not write log1p into an int .data buffer (UFuncTypeError before PR #54).
    X = sp.csr_matrix(np.array([[1, 0, 3], [2, 1, 0], [0, 4, 5], [3, 0, 1]], dtype=np.int64))
    adata = ad.AnnData(
        X=X,
        obs=pd.DataFrame({"target": ["ctrl", "A", "A", "ctrl"]},
                         index=[f"c{i}" for i in range(4)]),
        var=pd.DataFrame(index=["g0", "g1", "g2"]),
    )
    # Direct grouped-means helper: must not raise on int-sparse geometric.
    means = _group_means_linear(adata, "target", "geometric")
    assert set(means) == {"ctrl", "A"}
    assert all(np.all(np.isfinite(v)) for v in means.values())
    # Public LFC table entry point: must not raise either.
    lfc = compute_lfc_table(adata, groupby="target", reference="ctrl",
                            mean_calc="geometric", epsilon=1e-9)
    assert lfc.height > 0
    # Bit-identical to the explicit float-cast input (the fix is a no-op cast).
    adata_f = adata.copy()
    adata_f.X = adata_f.X.astype(np.float64)
    means_f = _group_means_linear(adata_f, "target", "geometric")
    for k in means:
        np.testing.assert_array_equal(means[k], means_f[k])


def _fake_gpudge_df():
    # gpudge raw schema (subset): target/feature/target_mean/ref_mean/log2_fold_change/p_value/p_adj
    return pl.DataFrame({
        "target": ["A", "A", "B", "B"],
        "feature": ["g0", "g1", "g0", "g1"],
        "target_mean": [0.0, 2.0, 3.0, 0.0],
        "ref_mean": [1.0, 0.0, 4.0, 0.0],
        "log2_fold_change": [float("-inf"), float("inf"), -0.415, float("nan")],
        "p_value": [0.01, 0.2, 0.03, 0.5],
        "p_adj": [0.02, 0.2, 0.03, 0.5],
    })


def test_finalize_gpudge_de_v2_passthrough_schema():
    # v2: clip_value=None, fdr_scope="per_pert" -> just schema-normalize (adds abs LFC), keep p_adj
    from cell_eval2.de_compute import _finalize_gpudge_de

    out = _finalize_gpudge_de(_fake_gpudge_df(), epsilon=1e-9, clip_value=None, fdr_scope="per_pert")
    assert "abs_log2_fold_change" in out.columns
    assert out["p_adj"].to_list() == [0.02, 0.2, 0.03, 0.5]  # unchanged (per-pert, no global BH)


def test_finalize_gpudge_de_v1_clip_replaces_inf():
    # v1: clip_value=20 -> zero-mean genes get pdex clip instead of +/-inf
    from cell_eval2.de_compute import _finalize_gpudge_de

    out = _finalize_gpudge_de(_fake_gpudge_df(), epsilon=0.0, clip_value=20.0, fdr_scope="global")
    lfc = dict(zip(zip(out["target"], out["feature"]), out["log2_fold_change"]))
    assert lfc[("A", "g0")] == np.log2(1.0 / 20.0)   # tgt-zero -> 1/clip
    assert lfc[("A", "g1")] == np.log2(20.0)         # ref-zero -> clip
    assert np.isfinite(lfc[("A", "g0")]) and np.isfinite(lfc[("A", "g1")])


from cell_eval2.gpu import resolve_device  # noqa: E402

_HAS_GPU = resolve_device("auto") == "cuda"


def _small_counts_adata(seed=11, n=600, g=20):
    import scipy.sparse as sp

    rng = np.random.default_rng(seed)
    labels = np.repeat(["non-targeting", "A", "B", "C", "D", "E"], n // 6)
    X = sp.csr_matrix(rng.poisson(1.5, size=(len(labels), g)).astype(np.float32))
    return ad.AnnData(X=X, obs=pd.DataFrame({"target_gene": labels}),
                      var=pd.DataFrame(index=[f"g{i}" for i in range(g)]))


@pytest.mark.skipif(not _HAS_GPU, reason="no usable CUDA GPU")
def test_compute_de_streaming_mode1_matches_in_memory(tmp_path):
    # `_HAS_GPU` alone is not enough: this test WRITES a `.shad`, so it needs the optional `scale`
    # extra too. On a GPU host without it the bare `import cellstream` was a collection ERROR, not
    # a skip -- and CPU CI never saw it because `_HAS_GPU` skips there. That is the same shape as
    # the h5ad-manifest defect the publish runbook exists to catch: GPU-gated, so invisible
    # everywhere the GPU tests do not run (Gemini, #366).
    pytest.importorskip("cellstream")
    import cellstream
    from cell_eval2.de_compute import compute_de, compute_de_streaming

    adata = _small_counts_adata()
    shd = str(tmp_path / "ref.shad")
    cellstream.write_sharded(adata, shd, group_by="target_gene", reference="non-targeting")
    im = compute_de(adata, backend="gpudge", groupby="target_gene", reference="non-targeting",
                    mean_calc="arithmetic", epsilon=1e-9, input_type="counts", target_sum=1e6,
                    clip_value=None, filter_gene_min_cpm_cell=5.0, fdr_scope="per_pert"
                    ).sort(["target", "feature"])
    st = compute_de_streaming(shd, backend="gpudge", reference=None, groupby="target_gene",
                              mean_calc="arithmetic", epsilon=1e-9, target_sum=1e6,
                              clip_value=None, fdr_scope="per_pert", filter_gene_min_cpm_cell=5.0
                              ).sort(["target", "feature"])
    j = im.join(st, on=["target", "feature"], suffix="_s")
    assert j.height == im.height > 0
    for col in ("log2_fold_change", "p_value", "p_adj"):
        a = j[col].to_numpy()
        b = j[f"{col}_s"].to_numpy()
        fin = np.isfinite(a) & np.isfinite(b)
        assert np.all(np.abs(a[fin] - b[fin]) <= 1e-4 * np.abs(a[fin]) + 1e-6)


def test_compute_de_streaming_rejects_cpu_backend():
    # CPU-ok: explicit pdex/scanpy on the streaming path errors before touching the GPU
    from cell_eval2.de_compute import compute_de_streaming

    with pytest.raises(ValueError, match="gpudge"):
        compute_de_streaming("x.shad", backend="pdex", reference=None, groupby="target_gene",
                             mean_calc="arithmetic", epsilon=1e-9, target_sum=1e6,
                             clip_value=None, fdr_scope="per_pert", filter_gene_min_cpm_cell=5.0)


# ---- in-memory external-reference DE (gpudge_arc #67), CPU-runnable error paths ----

def _tiny_counts_adata(n=40, g=6, control="non-targeting"):
    from scipy.sparse import csr_matrix
    rng = np.random.default_rng(0)
    X = csr_matrix(rng.integers(0, 30, size=(n, g)).astype("float32"))
    labels = ([control] * (n // 2)) + ["P1"] * (n - n // 2)
    obs = pd.DataFrame({"target": labels})
    var = pd.DataFrame(index=[f"g{i}" for i in range(g)])
    return ad.AnnData(X=X, obs=obs, var=var)


def test_anndata_reference_requires_gpudge_backend():
    a = _tiny_counts_adata()
    ref = a[a.obs["target"] == "non-targeting"].copy()
    tgt = a[a.obs["target"] != "non-targeting"].copy()
    with pytest.raises(ValueError, match="requires the gpudge backend"):
        compute_de(tgt, backend="scanpy", groupby="target", reference=ref,
                   mean_calc="arithmetic", epsilon=1e-9, input_type="counts",
                   filter_gene_min_cpm_cell=None)


def test_anndata_reference_var_mismatch_raises(monkeypatch):
    import cell_eval2.de_compute as dc
    monkeypatch.setattr(dc, "_resolve_backend", lambda b: "gpudge")
    monkeypatch.setattr(dc, "_gpudge_supports_inmem_external_ref", lambda: True)
    a = _tiny_counts_adata()
    ref = a[a.obs["target"] == "non-targeting"][:, [0, 1, 2]].copy()  # fewer genes -> mismatch
    tgt = a[a.obs["target"] != "non-targeting"].copy()
    with pytest.raises(ValueError, match="var_names"):
        dc.compute_de(tgt, backend="gpudge", groupby="target", reference=ref,
                      mean_calc="arithmetic", epsilon=1e-9, input_type="counts",
                      filter_gene_min_cpm_cell=None)


def test_anndata_reference_needs_gpudge_67(monkeypatch):
    import cell_eval2.de_compute as dc
    monkeypatch.setattr(dc, "_resolve_backend", lambda b: "gpudge")
    monkeypatch.setattr(dc, "_gpudge_supports_inmem_external_ref", lambda: False)
    a = _tiny_counts_adata()
    ref = a[a.obs["target"] == "non-targeting"].copy()
    tgt = a[a.obs["target"] != "non-targeting"].copy()
    # The clause has to be UNIQUE to this message: "external-reference DE" alone also appears in
    # `partition_inmem`'s NotImplementedError and in the sibling ValueError above, so matching on
    # it would pass for a different failure (codex, PR #359).
    with pytest.raises(RuntimeError,
                       match="requires a gpudge build that supports an in-memory AnnData "
                             "reference pool"):
        dc.compute_de(tgt, backend="gpudge", groupby="target", reference=ref,
                      mean_calc="arithmetic", epsilon=1e-9, input_type="counts",
                      filter_gene_min_cpm_cell=None)


@pytest.mark.skipif(not _HAS_GPU, reason="no usable CUDA GPU")
def test_inmem_external_ref_matches_concat_gpudge():
    """gpudge #67 in-memory external-ref DE must equal the concat DE on the same cells.

    Option A: the external-ref call takes the FULL adata (INCLUDING the control group) plus
    control_group, mirroring run._pred_de_input — it passes the full pred to gpudge (no
    subset copy, which OOMs at ~5M cells) and drops the control group's spurious
    control-vs-refpool rows from the output. Tested under BOTH FDR scopes: 'global' proves the
    drop happens BEFORE the BH pool (else the retained targets' p_adj would shift vs concat)."""
    from polars.testing import assert_frame_equal

    from cell_eval2.de_compute import compute_de

    a = _small_counts_adata()  # counts; control 'non-targeting' + perts A-E; groupby target_gene
    ref = a[a.obs["target_gene"] == "non-targeting"].copy()              # separate control pool
    key = ["target", "feature"]
    for fdr_scope in ("per_pert", "global"):
        kw = dict(backend="gpudge", groupby="target_gene", mean_calc="arithmetic", epsilon=1e-9,
                  input_type="counts", target_sum=1e6, filter_gene_min_cpm_cell=None,
                  fdr_scope=fdr_scope)
        concat_df = compute_de(a, reference="non-targeting", **kw)        # control as an in-adata group
        # FULL adata (incl. control) + control_group -> gpudge #67 external-ref, control dropped
        ext_df = compute_de(a, reference=ref, control_group="non-targeting", **kw)
        assert "non-targeting" not in set(ext_df["target"].to_list())     # control group dropped
        assert_frame_equal(concat_df.sort(key), ext_df.sort(key), check_exact=False,
                           abs_tol=1e-6, rel_tol=1e-4)


def test_release_gpu_pool_is_noop_safe():
    # On CPU nodes cupy is absent or has no device -> the helper must return None and never
    # raise (it guards both the import and the free_all_blocks calls).
    from cell_eval2.de_compute import _release_gpu_pool

    assert _release_gpu_pool() is None


def test_de_gpudge_releases_gpu_pool_around_de(monkeypatch):
    # _de_gpudge frees the pool BEFORE de() (so gpudge's chunk-sizer sees the full GPU, not a
    # pool-starved "free" -> a tiny chunk -> ~10-20x slower de_pred) AND AFTER de() (finally, so
    # the next same-process GPU phase's cuBLAS can allocate). Both gpudge_arc#76.
    gpudge = pytest.importorskip("gpudge")

    import cell_eval2.de_compute as dc

    order = []
    monkeypatch.setattr(gpudge, "de", lambda *a, **k: order.append("de") or "DF", raising=False)
    monkeypatch.setattr(dc, "_release_gpu_pool", lambda: order.append("release"))
    out = dc._de_gpudge(object(), groupby="g", reference="r", mean_calc="arithmetic",
                        epsilon=1.0, cpm_normalize=True, filter_gene_min_cpm_cell=None)
    assert out == "DF"
    assert order == ["release", "de", "release"]  # freed before (sizing) and after (handback)


def test_de_gpudge_releases_gpu_pool_on_exception(monkeypatch):
    # The release must run even if gpudge.de() raises (finally), else a DE error leaves VRAM
    # clogged for downstream / retry work (Gemini on PR #75).
    gpudge = pytest.importorskip("gpudge")

    import cell_eval2.de_compute as dc

    order = []

    def fake_de(*a, **k):
        order.append("de_fail")
        raise RuntimeError("GPUDGE FAILED")

    monkeypatch.setattr(gpudge, "de", fake_de, raising=False)
    monkeypatch.setattr(dc, "_release_gpu_pool", lambda: order.append("release"))
    with pytest.raises(RuntimeError, match="GPUDGE FAILED"):
        dc._de_gpudge(object(), groupby="g", reference="r", mean_calc="arithmetic",
                      epsilon=1.0, cpm_normalize=True, filter_gene_min_cpm_cell=None)
    assert order == ["release", "de_fail", "release"]  # freed before; finally frees despite raise


def test_gpudge_external_ref_counts_median_raises(monkeypatch, synthetic_counts_pair):
    # #155: on the DEFAULT branch the target block and the control pool are normalized by two
    # independent _to_linear calls, so every log2FC shifts by log2(T_target/T_ref).
    import sys
    import types
    pred, real = synthetic_counts_pair
    ctrl = real[real.obs["target"] == "non-targeting"].copy()
    fake_mod = types.ModuleType("gpudge")
    fake_mod.de = lambda *a, **k: pytest.fail("must not reach gpudge")
    monkeypatch.setitem(sys.modules, "gpudge", fake_mod)
    from cell_eval2 import de_compute
    monkeypatch.setattr(de_compute, "_resolve_backend", lambda b: "gpudge")
    with pytest.raises(ValueError, match="target_sum=None"):
        de_compute.compute_de(pred, backend="gpudge", groupby="target", reference=ctrl,
                              control_group="non-targeting", mean_calc="arithmetic",
                              epsilon=1e-9, input_type="counts", target_sum=None,
                              filter_gene_min_cpm_cell=None)


def test_gpudge_external_ref_median_allowed_under_native_normalize(monkeypatch,
                                                                   synthetic_counts_pair):
    # NOT a defect: gpudge resolves ONE union median over reference + all target cells
    # (_refpool.py:536-546), so both halves share a target. The guard must not reject it, and
    # compute_de must hand gpudge the raw matrices with normalize_target_sum="median".
    import sys
    import types
    import numpy as np
    import polars as pl
    pred, real = synthetic_counts_pair
    ctrl = real[real.obs["target"] == "non-targeting"].copy()
    seen = {}

    def fake_de(adata, *, groupby, reference, mean_calc, epsilon,
                cpm_normalize, filter_gene_min_cpm_cell, normalize_target_sum=None):
        seen["normalize_target_sum"] = normalize_target_sum
        seen["cpm_normalize"] = cpm_normalize
        seen["raw"] = np.allclose(np.asarray(adata.X), np.asarray(pred.X))
        return pl.DataFrame({"target": ["GENE1"], "feature": ["g0"],
                             "log2_fold_change": [1.0], "p_value": [0.01], "p_adj": [0.02]})

    fake_mod = types.ModuleType("gpudge")
    fake_mod.de = fake_de
    monkeypatch.setitem(sys.modules, "gpudge", fake_mod)
    from cell_eval2 import de_compute
    monkeypatch.setattr(de_compute, "_resolve_backend", lambda b: "gpudge")
    monkeypatch.setattr(de_compute, "_gpudge_supports_inmem_external_ref", lambda: True)
    de_compute.compute_de(pred, backend="gpudge", groupby="target", reference=ctrl,
                          control_group="non-targeting", mean_calc="arithmetic", epsilon=1e-9,
                          input_type="counts", target_sum=None, filter_gene_min_cpm_cell=None,
                          native_gpu_normalize=True)
    assert seen["normalize_target_sum"] == "median"
    assert seen["cpm_normalize"] is False
    assert seen["raw"], "native path must hand gpudge RAW counts, not a pre-normalized matrix"


def test_gpudge_in_adata_reference_counts_median_still_allowed(monkeypatch,
                                                               synthetic_counts_pair):
    # The guard is external-reference-only: a string reference means ONE matrix, one target.
    import sys
    import types
    import numpy as np
    import polars as pl
    _pred, real = synthetic_counts_pair
    seen = {}

    def fake_de(adata, *, groupby, reference, mean_calc, epsilon,
                cpm_normalize, filter_gene_min_cpm_cell, normalize_target_sum=None):
        seen["X"] = np.asarray(adata.X).copy()
        return pl.DataFrame({"target": ["GENE1"], "feature": ["g0"],
                             "log2_fold_change": [1.0], "p_value": [0.01], "p_adj": [0.02]})

    fake_mod = types.ModuleType("gpudge")
    fake_mod.de = fake_de
    monkeypatch.setitem(sys.modules, "gpudge", fake_mod)
    from cell_eval2 import de_compute
    monkeypatch.setattr(de_compute, "_resolve_backend", lambda b: "gpudge")
    de_compute.compute_de(real, backend="gpudge", groupby="target",
                          reference="non-targeting", mean_calc="geometric", epsilon=0.0,
                          input_type="counts", target_sum=None, filter_gene_min_cpm_cell=None)
    assert "X" in seen
