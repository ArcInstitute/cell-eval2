import numpy as np
import pytest
from _helpers import _counts_adata_fp64, _r_zero_adata_fp64
from cell_eval2.gpu import resolve_device
from cell_eval2.prep import bulk_lognorm_means, pseudobulk_bulk_lognorm


def test_mass_ratio_is_one_under_bulk_lognorm():
    from cell_eval2.metrics.delta import real_mass_ratio

    bulk = bulk_lognorm_means(
        np.array([[3.0, 5.0, 2.0], [1.0, 1.0, 8.0]]), 1e6,
    )
    got = real_mass_ratio(
        real_bulk=(np.array(["A", "B"]), bulk), mass_target=1e6, control="A",
    )
    assert got["B"] == pytest.approx(1.0, abs=1e-9)


def test_mass_ratio_uses_the_direct_lognorm_fallback_oracle_at_28118():
    from cell_eval2.metrics.delta import real_mass_ratio

    rng = np.random.default_rng(0)
    cells = rng.poisson(3.0, size=(200, 400)).astype(np.float64)
    ts = 28_118.0
    lognorm_bulk = np.log1p(cells * (ts / cells.sum(axis=1, keepdims=True))).mean(
        axis=0, keepdims=True,
    )
    expected = float(np.expm1(lognorm_bulk).sum() / ts)
    got = real_mass_ratio(
        real_bulk=(np.array(["A", "B"]), np.vstack([lognorm_bulk, lognorm_bulk])),
        mass_target=ts,
        control="A",
    )
    assert got["B"] == pytest.approx(expected, abs=1e-12)
    assert expected < 1.0


def test_mass_ratio_is_nan_when_the_target_is_unresolvable():
    from cell_eval2.metrics.delta import real_mass_ratio

    got = real_mass_ratio(
        real_bulk=(np.array(["A", "B"]), np.zeros((2, 3))),
        mass_target=None,
        control="A",
    )
    assert np.isnan(got["B"])


def _per_cell_lognorm(adata, target_sum):
    out = adata.copy()
    values = np.asarray(out.X, dtype=np.float64)
    out.X = np.log1p(values * (target_sum / values.sum(axis=1, keepdims=True)))
    return out


def test_compute_metrics_mass_ratio_is_one_at_a_nondefault_bulk_target(
        synthetic_counts_pair):
    from cell_eval2.config import EvalConfig
    from cell_eval2.run import compute_metrics

    pred, real = synthetic_counts_pair
    got = compute_metrics(
        pred,
        real,
        config=EvalConfig(
            metrics=["expr_real_mass_ratio"], bulk_target_sum=28_000.0, device="cpu",
        ),
    )
    assert got.height
    assert got["value"].to_list() == pytest.approx([1.0] * got.height, abs=1e-9)


def test_compute_metrics_mass_ratio_uses_the_resolved_per_cell_target_on_lognorm_input(
        synthetic_counts_pair):
    from cell_eval2.config import EvalConfig
    from cell_eval2.run import compute_metrics

    pred_counts, real_counts = synthetic_counts_pair
    ts = 28_118.0
    pred = _per_cell_lognorm(pred_counts, ts)
    real = _per_cell_lognorm(real_counts, ts)
    got = compute_metrics(
        pred,
        real,
        config=EvalConfig(
            metrics=["expr_real_mass_ratio"], input_type="lognorm", target_sum=ts,
            device="cpu",
        ),
    )
    expected = {}
    labels = real.obs["target"].astype(str).to_numpy()
    values = np.asarray(real.X, dtype=np.float64)
    for pert in sorted(set(labels) - {"non-targeting"}):
        mean = values[labels == pert].mean(axis=0)
        expected[pert] = float(np.expm1(mean).sum() / ts)
    observed = dict(zip(got["perturbation"].to_list(), got["value"].to_list()))
    assert observed == pytest.approx(expected, abs=1e-12)


def test_score_piece_filters_real_side_diagnostic_rows_without_narrowing_the_real_bulk(
        tmp_path, synthetic_counts_pair, monkeypatch):
    from cell_eval2 import partition_inmem
    from cell_eval2.config import EvalConfig
    from cell_eval2.metrics.discrimination import discrimination_score
    from cell_eval2.partition_inmem import build_reference, score_piece

    # Partition eligibility is currently gated on the DE backend even for an AnnData-only
    # metric; isolate this row-filter test from machine GPU availability.
    monkeypatch.setattr(partition_inmem, "_resolve_backend", lambda _name: "gpudge")
    pred, real = synthetic_counts_pair
    cfg = EvalConfig(metrics=["expr_real_mass_ratio", "pds_cosine"], device="cpu")
    # ⚠️ The oracle bulks take `cfg.bulk_target_sum`, NOT a literal. `build_reference` and
    # `score_piece` below run off `cfg`, so a hard-coded target here would compare two
    # normalizations; it read 1e6 against a 5e4 cfg after #268 and passed only because this
    # fixture happens to select the same perturbation at both (Codex review, finding 3).
    real_bulk = pseudobulk_bulk_lognorm(real, "target", bulk_target_sum=cfg.bulk_target_sum)
    genes = np.asarray(real.var.index.values, dtype=str)
    pert = None
    expected_pds = narrowed_pds = None
    for candidate in sorted(set(pred.obs["target"].astype(str)) - {"non-targeting"}):
        candidate_piece = pred[pred.obs["target"].astype(str) == candidate].copy()
        pred_bulk = pseudobulk_bulk_lognorm(
            candidate_piece, "target", bulk_target_sum=cfg.bulk_target_sum,
        )
        full = discrimination_score(
            pred_bulk=pred_bulk, real_bulk=real_bulk, control=cfg.control,
            control_source=cfg.control_source, distance="cosine", genes=genes,
            rank_denominator=cfg.discrimination.rank_denominator,
            exclude_target_gene=cfg.discrimination.exclude_target_gene,
        )[candidate]
        keep = np.isin(real_bulk[0].astype(str), ["non-targeting", candidate])
        narrow = discrimination_score(
            pred_bulk=pred_bulk,
            real_bulk=(real_bulk[0][keep], real_bulk[1][keep]),
            control=cfg.control, control_source=cfg.control_source, distance="cosine", genes=genes,
            rank_denominator=cfg.discrimination.rank_denominator,
            exclude_target_gene=cfg.discrimination.exclude_target_gene,
        )[candidate]
        if full != pytest.approx(narrow):
            pert, expected_pds, narrowed_pds = candidate, full, narrow
            break
    assert pert is not None, "fixture does not distinguish full-panel from narrowed PDS"
    piece = pred[pred.obs["target"].astype(str) == pert].copy()
    cache = str(tmp_path / "reference")
    build_reference(
        real,
        config=cfg,
        cache_dir=cache,
        control_format="h5ad",
        comparator="bulk_lognorm",
    )
    got = score_piece(piece, cache, config=cfg, comparator="bulk_lognorm")
    assert set(got["perturbation"].to_list()) == {pert}
    observed = dict(zip(got["metric"].to_list(), got["value"].to_list()))
    assert observed["expr_real_mass_ratio"] == pytest.approx(1.0, abs=1e-9)
    assert observed["pds_cosine"] == pytest.approx(expected_pds)
    assert observed["pds_cosine"] != pytest.approx(narrowed_pds)


def test_expm1_of_the_bulk_sums_to_the_target_exactly():
    """The property the whole change exists for: a point mass is representable."""
    rng = np.random.default_rng(0)
    sums = rng.poisson(30.0, size=(4, 500)).astype(np.float64)
    got = np.expm1(bulk_lognorm_means(sums, 1e6)).sum(axis=1) / 1e6
    assert np.allclose(got, 1.0, rtol=0, atol=1e-9), got


def test_a_tiled_group_and_a_dispersed_group_with_the_same_sum_agree():
    """bulk_lognorm reads the SUM, so emission cannot move it. This is the objective."""
    import anndata as ad

    rng = np.random.default_rng(1)
    cells = rng.poisson(5.0, size=(50, 200)).astype(np.float64)
    total = cells.sum(axis=0, keepdims=True)
    tiled = np.repeat(total / 50.0, 50, axis=0)
    dispersed_ad = ad.AnnData(cells, obs={"target": ["A"] * cells.shape[0]})
    tiled_ad = ad.AnnData(tiled, obs={"target": ["A"] * tiled.shape[0]})
    dispersed = pseudobulk_bulk_lognorm(
        dispersed_ad, "target", bulk_target_sum=1e6,
    )
    uniform = pseudobulk_bulk_lognorm(
        tiled_ad, "target", bulk_target_sum=1e6,
    )
    assert dispersed[0].tolist() == uniform[0].tolist() == ["A"]
    assert np.allclose(dispersed[1], uniform[1], rtol=0, atol=1e-12)


def test_an_all_zero_group_does_not_divide_by_zero():
    assert np.all(bulk_lognorm_means(np.zeros((1, 5)), 1e6) == 0.0)


def test_pseudobulk_bulk_lognorm_is_on_the_shell(synthetic_counts_pair):
    _pred, real = synthetic_counts_pair
    perts, means = pseudobulk_bulk_lognorm(real, "target", bulk_target_sum=1e6)
    assert means.shape[0] == perts.size
    assert np.allclose(np.expm1(means).sum(axis=1) / 1e6, 1.0, atol=1e-9)


def test_cpu_and_accumulator_drivers_agree_on_bulk_lognorm(synthetic_counts_pair):
    """The plumbing is where this breaks, not the formula."""
    from cell_eval2.streaming_bulk import inmem_pseudobulk

    _pred, real = synthetic_counts_pair
    perts_ref, ref = pseudobulk_bulk_lognorm(real, "target", bulk_target_sum=1e6)
    got = inmem_pseudobulk(
        real, pert_col="target", norms=["bulk_lognorm"], target_sum=None,
        bulk_target_sum=1e6, device="cpu",
    )["bulk_lognorm"]
    assert list(got[0]) == list(perts_ref)
    assert np.allclose(got[1], ref, rtol=1e-5, atol=1e-6)


def test_bulk_lognorm_alone_needs_no_target_sum(synthetic_counts_pair):
    """target_sum=None must NOT raise when no requested norm consumes it."""
    from cell_eval2.streaming_bulk import inmem_pseudobulk

    _pred, real = synthetic_counts_pair
    inmem_pseudobulk(
        real, pert_col="target", norms=["bulk_lognorm"], target_sum=None,
        bulk_target_sum=1e6, device="cpu",
    )


def test_side_bulks_builds_bulk_lognorm_without_to_normalization(synthetic_counts_pair):
    """_side_bulks' CPU path is to_normalization, which raises for this target."""
    from cell_eval2.config import EvalConfig
    from cell_eval2.run import _side_bulks

    _pred, real = synthetic_counts_pair
    out = _side_bulks(
        real, fp=None, store=None, norms=["bulk_lognorm"], cfg=EvalConfig(device="cpu"),
        side="real",
    )
    assert set(out) == {"bulk_lognorm"}


def test_bulk_lognorm_moments_name_issue_264_pr2(synthetic_counts_pair):
    from cell_eval2.streaming_bulk import inmem_pseudobulk

    _pred, real = synthetic_counts_pair
    out, moments = inmem_pseudobulk(
        real, pert_col="target", norms=["bulk_lognorm"], target_sum=None,
        with_moments=True,
    )
    perts, means = out["bulk_lognorm"]
    mom = moments["bulk_lognorm"]
    assert means.ndim == 2 and means.shape[0] == perts.size
    assert mom.jk is not None and mom.jk.shape == (perts.size,)


def test_moments_are_requested_for_the_resolved_comparator_key_only():
    from cell_eval2.run import _moment_normalizations

    assert _moment_normalizations(
        ["pds_cosine", "expr_mse_unbiased"], comparator="bulk_lognorm"
    ) == {"bulk_lognorm"}
    assert _moment_normalizations(
        ["pds_cosine"], comparator="bulk_lognorm"
    ) == set()


def test_a_mixed_profile_gets_moments_only_where_they_are_needed(synthetic_counts_pair):
    """pds_* on bulk_lognorm and expr_* on lognorm-with-moments, in one run."""
    from cell_eval2.config import EvalConfig
    from cell_eval2.run import compute_metrics

    pred, real = synthetic_counts_pair
    df = compute_metrics(
        pred,
        real,
        config=EvalConfig(metrics=["pds_cosine", "expr_mse_unbiased"], device="cpu"),
    )
    assert {"pds_cosine", "expr_mse_unbiased"} <= set(df["metric"].unique())


def test_pds_cosine_reads_the_bulk_lognorm_values_not_the_old_lognorm_bulk(
        synthetic_counts_pair):
    """The runtime must read the resolved bulk, not merely build and identify it."""
    from cell_eval2 import norm
    from cell_eval2.config import EvalConfig
    from cell_eval2.metrics.discrimination import discrimination_score
    from cell_eval2.prep import pseudobulk
    from cell_eval2.run import compute_metrics

    pred, real = synthetic_counts_pair
    cfg = EvalConfig(metrics=["pds_cosine"], device="cpu")
    got_df = compute_metrics(pred, real, config=cfg)
    got = dict(zip(got_df["perturbation"].to_list(), got_df["value"].to_list()))

    direct_kwargs = {
        "control": cfg.control,
        "control_source": cfg.control_source,
        "distance": "cosine",
        "rank_denominator": cfg.discrimination.rank_denominator,
        "exclude_target_gene": cfg.discrimination.exclude_target_gene,
        "genes": np.asarray(real.var.index.values, dtype=str),
    }
    expected = discrimination_score(
        pred_bulk=pseudobulk_bulk_lognorm(
            pred, cfg.pert_col, bulk_target_sum=cfg.bulk_target_sum,
        ),
        real_bulk=pseudobulk_bulk_lognorm(
            real, cfg.pert_col, bulk_target_sum=cfg.bulk_target_sum,
        ),
        **direct_kwargs,
    )
    old = discrimination_score(
        pred_bulk=pseudobulk(
            norm.to_normalization(
                pred, "counts", "lognorm", target_sum=cfg.target_sum,
            ),
            cfg.pert_col,
        ),
        real_bulk=pseudobulk(
            norm.to_normalization(
                real, "counts", "lognorm", target_sum=cfg.target_sum,
            ),
            cfg.pert_col,
        ),
        **direct_kwargs,
    )

    assert expected != old, "fixture does not distinguish the two rank comparators"
    assert got == pytest.approx(expected)
    assert got != pytest.approx(old)


def _score_via(driver, pred, real, *, bulk_target_sum, tmp_path, monkeypatch):
    from cell_eval2.config import EvalConfig

    cfg = EvalConfig(
        metrics=["delta_mse"], device="cuda" if driver == "gpu" else "cpu",
        bulk_target_sum=bulk_target_sum,
    )

    if driver in {"inmem_cpu", "gpu"}:
        from cell_eval2.run import compute_metrics

        scored = compute_metrics(pred, real, config=cfg)

    elif driver == "shard":
        pytest.importorskip("cellstream")
        from cellstream import write_sharded
        from cell_eval2.scale import score_streaming

        pred_path = tmp_path / "bulk-pred.shad"
        real_path = tmp_path / "bulk-real.shad"
        write_sharded(pred, pred_path, group_by="target", reference="non-targeting")
        write_sharded(real, real_path, group_by="target", reference="non-targeting")
        scored = score_streaming(pred_path, real_path, config=cfg)

    elif driver == "cell_layout":
        pytest.importorskip("cellstream")
        from cellstream.cell import write_cell_archive
        from cell_eval2.scale import score_streaming_cell

        pred_path = tmp_path / "bulk-pred-cell.shad"
        real_path = tmp_path / "bulk-real-cell.shad"
        pred_ref = (pred.obs["target"].astype(str) == "non-targeting").to_numpy()
        real_ref = (real.obs["target"].astype(str) == "non-targeting").to_numpy()
        write_cell_archive(
            pred, pred_path, group_by="target", reference=pred_ref, codec="zstd", overwrite=True,
        )
        write_cell_archive(
            real, real_path, group_by="target", reference=real_ref, codec="zstd", overwrite=True,
        )
        scored = score_streaming_cell(pred_path, real_path, config=cfg)

    elif driver == "partition":
        import polars as pl
        from cell_eval2 import partition_inmem
        from cell_eval2.partition_inmem import build_reference, score_piece

        # The partition API currently gates on gpudge and builds real DE even when no DE metric
        # is selected. Keep this comparator acceptance test about the partitioned pseudobulk and
        # dispatch paths, independent of host GPU availability.
        monkeypatch.setattr(partition_inmem, "_resolve_backend", lambda _name: "gpudge")
        monkeypatch.setattr(
            partition_inmem,
            "compute_de",
            lambda *args, **kwargs: pl.DataFrame(
                {"target": []}, schema={"target": pl.Utf8},
            ),
        )
        cache = str(tmp_path / "bulk-partition")
        build_reference(
            real,
            config=cfg,
            cache_dir=cache,
            control_format="h5ad",
            comparator="bulk_lognorm",
        )
        piece = pred[pred.obs["target"].astype(str) != cfg.control].copy()
        scored = score_piece(
            piece,
            cache,
            config=cfg,
            comparator="bulk_lognorm",
        )

    else:
        raise AssertionError(driver)

    metric_rows = scored.filter(scored["metric"] == "delta_mse")
    return dict(zip(metric_rows["perturbation"].to_list(), metric_rows["value"].to_list()))


@pytest.mark.parametrize("driver", [
    "inmem_cpu",
    "shard",
    "cell_layout",
    # The repo gates GPU tests with skipif on the resolved device (test_gpu_device.py:40),
    # not a custom marker -- an unregistered `gpu` marker only emits
    # PytestUnknownMarkWarning and gates nothing.
    pytest.param("gpu", marks=pytest.mark.skipif(
        resolve_device("auto") != "cuda", reason="no usable CUDA GPU")),
    "partition",
])
def test_a_non_default_bulk_target_sum_reaches_every_driver(
        driver, synthetic_counts_pair, tmp_path, monkeypatch):
    """A wrapper default must not hide a forgotten production pass-through."""
    from cell_eval2.metrics.delta import mse_delta

    pred, real = synthetic_counts_pair

    def oracle(target):
        return mse_delta(
            pred_bulk=pseudobulk_bulk_lognorm(
                pred, "target", bulk_target_sum=target,
            ),
            real_bulk=pseudobulk_bulk_lognorm(
                real, "target", bulk_target_sum=target,
            ),
            control="non-targeting", control_source="real",
        )

    # ⚠️ The comparison value must be the ACTUAL shipped default, not a literal: this test
    # exists to catch a driver that silently falls back to it, and a driver falls back to
    # whatever the default IS. Pinning 1e6 here left the guard aimed at a value no wrapper
    # defaults to any more once #268 moved it to 5e4 (Codex review, finding 2).
    from cell_eval2.moments import DEFAULT_BULK_TARGET_SUM
    expected = oracle(28_000.0)
    wrapper_default = oracle(DEFAULT_BULK_TARGET_SUM)
    assert 28_000.0 != DEFAULT_BULK_TARGET_SUM, "the requested target must not BE the default"
    assert expected != pytest.approx(wrapper_default), \
        "fixture does not distinguish the requested target from the wrapper default"
    got = _score_via(
        driver,
        pred,
        real,
        bulk_target_sum=28_000.0,
        tmp_path=tmp_path,
        monkeypatch=monkeypatch,
    )
    assert got == pytest.approx(expected, rel=1e-5, abs=1e-6)
    assert got != pytest.approx(wrapper_default, rel=1e-5, abs=1e-6)


def test_lognorm_fallback_is_resolved_stamped_and_rejected_by_a_counts_baseline(
        tmp_path, synthetic_counts_pair, monkeypatch):
    from cell_eval2 import run
    from cell_eval2.baseline import build_generic_baseline, build_run_meta, write_json_meta
    from cell_eval2.cli import main
    from cell_eval2.config import EvalConfig
    from cell_eval2.run import aggregate_metrics_wide, compute_metrics, metric_output_names

    pred_counts, real_counts = synthetic_counts_pair
    target_sum = 28_118.0
    pred = _per_cell_lognorm(pred_counts, target_sum)
    real = _per_cell_lognorm(real_counts, target_sum)
    log_cfg = EvalConfig(
        metrics=["delta_mse"], input_type="lognorm", target_sum=target_sum, device="cpu",
    )

    resolved = []
    original_dispatch = run.dispatch_anndata_metrics

    def capture(*args, **kwargs):
        resolved.append(kwargs["comparator"])
        return original_dispatch(*args, **kwargs)

    monkeypatch.setattr(run, "dispatch_anndata_metrics", capture)
    results = compute_metrics(pred, real, config=log_cfg)
    assert resolved == ["lognorm"]

    user_dir = tmp_path / "lognorm-run"
    user_dir.mkdir()
    user_agg = aggregate_metrics_wide(results, metrics=metric_output_names(log_cfg))
    user_agg.write_csv(user_dir / "agg_results.csv")
    user_meta = build_run_meta(log_cfg, real, pred)
    assert user_meta["input_type_real_effective"] == "lognorm"
    assert user_meta["input_type_pred_effective"] == "lognorm"
    assert user_meta["comparator"] == "lognorm"
    write_json_meta(user_meta, user_dir / "run_meta.json")

    counts_cfg = EvalConfig(metrics=["delta_mse"], input_type="counts", device="cpu")
    baseline = build_generic_baseline(
        real_counts,
        config=counts_cfg,
        exclude_target_gene=False,
        emit="dispersed",
    )
    assert baseline.meta["comparator"] == "bulk_lognorm"
    baseline_dir = tmp_path / "counts-baseline"
    baseline_dir.mkdir()
    baseline.agg.write_csv(baseline_dir / "baseline_agg.csv")
    write_json_meta(baseline.meta, baseline_dir / "baseline_meta.json")

    with pytest.raises(SystemExit, match="comparator"):
        main([
            "score",
            "--user-agg", str(user_dir / "agg_results.csv"),
            "--baseline-agg", str(baseline_dir / "baseline_agg.csv"),
        ])


def test_a_warm_cache_is_not_reused_across_bulk_target_sums(
        tmp_path, synthetic_counts_pair):
    """Exercise both the in-memory and shard-reference L2 caches."""
    pytest.importorskip("cellstream")
    from cellstream import write_sharded
    from cell_eval2.cache import CacheStore
    from cell_eval2.config import EvalConfig
    from cell_eval2.run import _side_bulks
    from cell_eval2.scale import _real_reference

    _pred, real = synthetic_counts_pair
    inmem_cache = tmp_path / "inmem-cache"
    cfg_a = EvalConfig(cache_real=str(inmem_cache), bulk_target_sum=1e6, device="cpu")
    cfg_b = EvalConfig(cache_real=str(inmem_cache), bulk_target_sum=28_000.0, device="cpu")
    warm = _side_bulks(
        real, fp="fp", store=CacheStore(str(inmem_cache)), norms=["bulk_lognorm"],
        cfg=cfg_a, side="real",
    )["bulk_lognorm"][1]
    cold = _side_bulks(
        real, fp="fp", store=CacheStore(str(inmem_cache)), norms=["bulk_lognorm"],
        cfg=cfg_b, side="real",
    )["bulk_lognorm"][1]
    assert not np.allclose(warm, cold, rtol=1e-5)

    shad = tmp_path / "warm.shad"
    write_sharded(real, shad, group_by="target", reference="non-targeting")
    shard_cache = tmp_path / "shard-cache"
    shard_a = EvalConfig(cache_real=str(shard_cache), bulk_target_sum=1e6, device="cpu")
    shard_b = EvalConfig(
        cache_real=str(shard_cache), bulk_target_sum=28_000.0, device="cpu",
    )
    shard_warm = _real_reference(
        shad, cfg=shard_a, norms=["bulk_lognorm"], real_fp="fp",
    )["bulk_lognorm"][1]
    shard_cold = _real_reference(
        shad, cfg=shard_b, norms=["bulk_lognorm"], real_fp="fp",
    )["bulk_lognorm"][1]
    ref = pseudobulk_bulk_lognorm(real, "target", bulk_target_sum=28_000.0)[1]
    assert not np.allclose(shard_warm, shard_cold, rtol=1e-5)
    assert np.allclose(cold, ref, rtol=1e-5, atol=1e-6)
    assert np.allclose(shard_cold, ref, rtol=1e-5, atol=1e-6)


def test_compute_metrics_serves_expr_moments_through_the_cpu_in_memory_path(
        synthetic_counts_pair):
    """run._side_bulks:533 is the CPU path's ONLY route to bulk_lognorm and does not go
    through inmem_pseudobulk, so every driver-level test above can pass while the default
    scoring path is broken. Force device='cpu' and ask for a moments metric directly."""
    from cell_eval2.config import EvalConfig
    from cell_eval2.run import _side_bulks

    real, _ = synthetic_counts_pair
    cfg = EvalConfig(version="v2", input_type="counts", device="cpu")
    out, moments = _side_bulks(real, fp=None, store=None, norms=["bulk_lognorm"],
                               moment_norms={"bulk_lognorm"}, cfg=cfg, side="real")
    perts, means = out["bulk_lognorm"]          # NOT (perts, moments) -- rev 1's bug
    assert means.ndim == 2 and means.shape[0] == perts.size
    assert moments["bulk_lognorm"].jk is not None
    assert moments["bulk_lognorm"].jk.shape == (perts.size,)


def _split_half_ratio(*, corrected: bool, per_half: int = 200, g: int = 60):
    """The scored ratio of sums; zeroing both C arrays gives the uncorrected comparator.

    This is `expr_mse_unbiased_capped_norm` as `_derived_value` computes it (#257) -- the sum
    of the numerator over perturbations over the sum of the denominator, NOT a mean of
    per-perturbation ratios.
    """
    import anndata as ad
    import pandas as pd
    import scipy.sparse as sp

    from cell_eval2.metrics.delta import distance_unbiased, mse_unbiased_capped
    from cell_eval2.moments import GroupMoments
    from cell_eval2.prep import pseudobulk_bulk_lognorm_with_moments

    rng = np.random.default_rng(3)
    labels = ["non-targeting", "A", "B", "C"]
    base = rng.gamma(2.0, 3.0, size=(len(labels), g))
    halves = []
    for _ in range(2):
        X = np.vstack([
            rng.poisson(base[i], size=(per_half, g))
            for i in range(len(labels))
        ]).astype(np.float64)
        obs = pd.DataFrame({"target": np.repeat(labels, per_half)})
        halves.append(ad.AnnData(X=sp.csr_matrix(X), obs=obs))
    out = []
    for pred_ad, real_ad in ((halves[0], halves[1]), (halves[1], halves[0])):
        pp, pm, pmom = pseudobulk_bulk_lognorm_with_moments(
            pred_ad, "target", bulk_target_sum=1e6,
        )
        rp, rm, rmom = pseudobulk_bulk_lognorm_with_moments(
            real_ad, "target", bulk_target_sum=1e6,
        )
        if not corrected:
            pmom = GroupMoments(
                perts=pmom.perts, counts=pmom.counts, sumsq=pmom.sumsq,
                jk=np.zeros_like(pmom.jk),
            )
            rmom = GroupMoments(
                perts=rmom.perts, counts=rmom.counts, sumsq=rmom.sumsq,
                jk=np.zeros_like(rmom.jk),
            )
        num = mse_unbiased_capped(
            pred_bulk=(pp, pm), real_bulk=(rp, rm),
            pred_moments=pmom, real_moments=rmom,
            control="non-targeting", comparator="bulk_lognorm", driver="test",
        )
        den = distance_unbiased(
            real_bulk=(rp, rm), real_moments=rmom,
            control="non-targeting", comparator="bulk_lognorm", driver="test",
        )
        out.append(sum(num.values()) / sum(den.values()))
    return float(np.mean(out))


def test_the_split_half_replicate_is_scorable_only_with_the_correction():
    """Spec section 6 test 4, asserted as the MECHANISM rather than as a level.

    ⚠️ THE 1.0206 / ~0.02 FIGURES IN SPEC SECTION 2 ARE REAL-DATA NUMBERS -- 200 constructs,
    500 cells each, 18,533 genes -- and they belong to the section 6.2 harness
    (`internal:tools/metricval/emission_neutrality_264.py`), NOT here. This synthetic fixture cannot
    reproduce them and must not pretend to. (That harness MEASURES the ~0.02 corrected
    ceiling against a real run; the 1.0206 uncorrected arm is not computable from a scored
    run at all, since nothing in the pipeline disables `C`. Neither is asserted anywhere --
    they are panel-specific, which is the whole lesson of #268.) It sits in the opposite regime: sampling noise
    over a large fixed biological denominator, because the four labels are independent gamma
    draws. MEASURED here, uncorrected, as `per_half` grows -- 20: 0.0234, 50: 0.00982,
    200: 0.00210, 1000: 0.000398 -- i.e. the clean 1/n of a noise term over a constant, where
    the real panel's uncorrected value sits ABOVE the no-change point. Rebuilding the fixture
    to hit 1.0206 would be tuning a fixture against a target number.

    What this fixture CAN establish, and does: the correction cuts the residual by ~10x, and
    a no-op correction cannot pass. MEASURED at seed 3, per_half=200, g=60, TS=1e6:

        uncorrected  0.002099392691393939
        corrected    0.00021870519810520244        ratio 9.5992

    A zeroed C gives corrected == uncorrected exactly, i.e. a ratio of 1.0, so the reduction
    is what discriminates -- and `test_a_zeroed_correction_fails_the_split_half_test` proves
    that by mutation rather than by argument. Both assertions are sign-agnostic on `corrected`
    deliberately: the jackknife can OVER-correct a pure replicate, since the true excess is
    ~0 and noise signs it either way. MEASURED on this same fixture at other widths, the
    corrected ratio is NEGATIVE -- g=20: -1.526e-4, g=500: -1.014e-4. That is recorded, not
    asserted; #268 and #271 own any question about bounding it.

    Rev 1 asserted only a per-perturbation `corrected < raw`, which any small arbitrary
    subtraction passes.
    """
    raw = _split_half_ratio(corrected=False)
    corrected = _split_half_ratio(corrected=True)
    # Fixture identity. Loose enough for platform float drift, tight enough that a changed
    # fixture or kernel says so here instead of silently moving the claims above.
    assert raw == pytest.approx(0.002099392691393939, rel=5e-3), f"fixture moved: {raw}"
    # THE CLAIM: a replicate becomes scorable -- what is left after correction is small
    # against what was there before it.
    assert abs(corrected) < 0.15 * raw, f"correction left too much: {raw} -> {corrected}"
    # THE MECHANISM: a ~10x cut. Wide enough for drift, and 1.0 (a no-op C) is nowhere near.
    assert 5.0 < raw / abs(corrected) < 20.0, f"reduction not ~9.6x: {raw} -> {corrected}"


def test_a_zeroed_correction_fails_the_split_half_test(monkeypatch):
    """Section 6 test 4's own requirement: 'any test that passes against a deliberately
    zeroed C is not testing the correction'.

    ⚠️ Patch `cell_eval2.prep.jackknife_correction`, NOT `cell_eval2.moments....`: prep binds
    it by direct import, so patching the original leaves prep's reference intact and the
    mutation silently does nothing. Rev 1 patched the wrong symbol and passed vacuously.
    """
    monkeypatch.setattr(
        "cell_eval2.prep.jackknife_correction",
        lambda X, codes, n, ts, **kw: np.zeros(n, dtype=np.float64),
    )
    # Both legs of the test above fail under this mutation: corrected == raw exactly, so
    # abs(corrected) is 1.0x raw rather than <0.15x, and the ratio is 1.0 rather than ~9.6.
    assert _split_half_ratio(corrected=True) == _split_half_ratio(corrected=False)
    with pytest.raises(AssertionError):
        test_the_split_half_replicate_is_scorable_only_with_the_correction()


def test_the_bulk_lognorm_correction_is_the_jackknife_not_the_analytic_trace():
    """Spec section 6 test 3's second half -- 'a test that would catch a regression to the
    analytic form'. Its first half (accuracy against a brute-force delete-1 loop, asked for
    at ~0.1%) is `tests/test_jackknife.py::test_jackknife_matches_brute_force_delete_one`,
    which pins it at rel=1e-10 on two target sums.

    The regression this catches is a driver, or a future refactor of `correction_for`, that
    fills `jk` from `trace_over_n_for` -- an analytic per-cell trace. That is not a small
    error: the two are estimators of different quantities in different spaces, and the
    `bulk_lognorm` GroupMoments deliberately keeps `counts`/`sumsq` in COUNTS space, so the
    analytic form over it is a number rather than a crash. MEASURED on this fixture
    (seed 1, 10 cells x 4 groups, 18 genes, TS=1e6):

        jackknife C       0.6529    0.8753    0.4954    0.3435
        tr(Sigma-hat)/n  -182.29   -149.02   -138.65   -113.17

    Two decades apart in magnitude, and NEGATIVE -- `sumsq/n - mean^2` over counts-space
    sumsq and group-sum-space means is not a variance of anything. A correction that came out
    negative would be ADDED to the metric as a bonus. (`correction_for` refuses the other
    direction of the same confusion outright: reading a jk-bearing artifact as `lognorm`
    raises, `tests/test_jackknife.py`.) The assertion below is on magnitude, not sign, so it
    still fires on a fixture where the analytic form happens to come out positive.

    RECORDED, from spec section 2 -- the ALTERNATIVE analytic correction, the delta method
    (first-order, not this per-cell trace), is much closer and still not good enough: it
    differs from the jackknife by only 1.4% on `C`, yet leaves the real split-half ceiling at
    0.0881 where the jackknife reaches 0.0487 and an exact correction gives 0.0291. The score
    is a small residual between large terms (raw numerator ~45x the result), so `C` must be
    right to ~0.1%. That sensitivity is why this family gets a two-pass O(n*G) correction
    instead of a closed form, and it is not reproducible on a synthetic fixture this size.
    """
    from cell_eval2.moments import correction_for, trace_over_n_for
    from cell_eval2.prep import pseudobulk_bulk_lognorm_with_moments

    adata = _counts_adata_fp64(seed=1, per_group=10, g=18)
    perts, means, mom = pseudobulk_bulk_lognorm_with_moments(
        adata, "target", bulk_target_sum=1e6,
    )
    jk = correction_for(mom, perts, means, comparator="bulk_lognorm")
    analytic = trace_over_n_for(mom, perts, means)
    np.testing.assert_allclose(jk, mom.jk, rtol=0, atol=0)
    assert np.all(jk > 0.0), f"a degenerate fixture cannot discriminate: {jk}"
    # MEASURED ratios on this fixture: 279, 170, 280, 329. A jk filled from the analytic
    # trace would read 1.0 here.
    assert np.min(np.abs(analytic) / jk) > 50.0, (
        f"the two corrections are no longer far apart -- has jk become the analytic trace? "
        f"jk={jk}, tr/n={analytic}"
    )


def _jk_by_driver(tmp_path, adata, *, device="cpu"):
    """Jackknife arrays aligned to sorted labels for every public non-partitioned entry."""
    # ⚠️ `cellstream` is the OPTIONAL `scale` extra and CI does not install it, so this must skip
    # rather than error -- exactly as tests/test_cell_source.py:33 and tests/test_cellstream.py:22
    # already do. It is installed in the dev venv, which is why the local suite never showed it:
    # the first CI run on this branch failed here with ModuleNotFoundError on both 3.11 and 3.12.
    pytest.importorskip("cellstream")
    from cellstream import write_sharded
    from cellstream.cell import write_cell_archive

    from cell_eval2._cell_archive import open_cell_store
    from cell_eval2.cell_source import cell_pseudobulk
    from cell_eval2.config import EvalConfig
    from cell_eval2.run import _side_bulks
    from cell_eval2.streaming_bulk import streaming_pseudobulk

    got = {}
    # ⚠️ `bulk_target_sum` is pinned EXPLICITLY, not inherited from the default. Every other
    # driver below is given 1e6 by hand and `want` is computed at 1e6, so a cfg that followed
    # the default would compare two different target sums and report it as a driver
    # disagreement. That is exactly what #268's 1e6 -> 5e4 move did to this test. The constant
    # is arbitrary -- what this test asserts is that the drivers AGREE at whatever it is.
    cfg = EvalConfig(
        version="v2", input_type="counts", device=device, pert_col="target",
        bulk_target_sum=1e6,
    )
    _, mom = _side_bulks(
        adata, fp=None, store=None, norms=["bulk_lognorm"],
        moment_norms={"bulk_lognorm"}, cfg=cfg, side="real",
    )
    got["in_memory"] = mom

    spath = str(tmp_path / f"a_{device}.shad")
    write_sharded(adata, spath, group_by="target", reference="non-targeting")
    _, mom = streaming_pseudobulk(
        spath, pert_col="target", norms=["bulk_lognorm"], target_sum=None,
        device=device, with_moments=True, bulk_target_sum=1e6,
    )
    got["shard_streaming"] = mom

    cpath = str(tmp_path / f"a_{device}.csad")
    # codec="zstd" and an explicit reference mask, exactly as tests/test_cell_source.py:42
    # and tests/test_cellstream.py:44 write theirs: the default pfordelta codec needs
    # pyfastpfor, which is not installed here, and the writer raises ImportError without it.
    write_cell_archive(adata, cpath, group_by="target",
                       reference=(adata.obs["target"] == "non-targeting").to_numpy(),
                       codec="zstd", overwrite=True)
    _, mom = cell_pseudobulk(
        open_cell_store(cpath), pert_col="target", norms=["bulk_lognorm"],
        target_sum=None, device=device, with_moments=True, bulk_target_sum=1e6,
    )
    got["cell_streaming"] = mom

    out = {}
    for name, moments in got.items():
        gm = moments["bulk_lognorm"]
        out[name] = gm.jk[np.argsort(gm.perts)]
    return out


def test_every_driver_agrees_on_the_jackknife(tmp_path):
    """Spec section 6 test 7 exercises the two-pass plumbing through public entry points."""
    from cell_eval2.prep import pseudobulk_bulk_lognorm_with_moments

    adata = _counts_adata_fp64(seed=2, per_group=25, g=20)
    _, _, ref = pseudobulk_bulk_lognorm_with_moments(
        adata, "target", bulk_target_sum=1e6,
    )
    want = ref.jk[np.argsort(ref.perts)]
    for name, jk in _jk_by_driver(tmp_path, adata).items():
        np.testing.assert_allclose(jk, want, rtol=1e-8, err_msg=name)
    # The partitioned driver does not expose GroupMoments; #272 is postponed to land with #270.


def test_every_driver_agrees_on_the_r_zero_edge_CASE(tmp_path):
    """The exact r_i == 0 oracle fixture reaches resident, shard, and cell routes."""
    from cell_eval2.prep import pseudobulk_bulk_lognorm_with_moments

    adata = _r_zero_adata_fp64()
    _, _, ref = pseudobulk_bulk_lognorm_with_moments(
        adata, "target", bulk_target_sum=1e6,
    )
    order = np.argsort(ref.perts)
    want = ref.jk[order]
    # The oracle Task 1 pinned for [[4, 0], [0, 0]] at TS=1e6, reached here through the
    # public in-memory entry point. `np.any(want > 40)` would also pass on a fixture that
    # had drifted into some other large value; this says WHICH number is expected, and the
    # group it belongs to (sorted labels: A, B, non-targeting).
    assert want[0] == pytest.approx(47.71708990205765, rel=1e-12), (
        f"the fixture no longer contains the r_i == 0 group: {want}"
    )
    for name, jk in _jk_by_driver(tmp_path, adata).items():
        np.testing.assert_allclose(jk, want, rtol=1e-10, err_msg=name)


@pytest.mark.skipif(resolve_device("auto") != "cuda", reason="requires a GPU")
def test_the_gpu_drivers_agree_on_the_jackknife(tmp_path):
    """The GPU public routes exercise the accumulator's streaming second pass."""
    from cell_eval2.prep import pseudobulk_bulk_lognorm_with_moments

    adata = _counts_adata_fp64(seed=2, per_group=25, g=20)
    _, _, ref = pseudobulk_bulk_lognorm_with_moments(
        adata, "target", bulk_target_sum=1e6,
    )
    want = ref.jk[np.argsort(ref.perts)]
    for name, jk in _jk_by_driver(tmp_path, adata, device="cuda").items():
        np.testing.assert_allclose(jk, want, rtol=1e-5, err_msg=name)


def test_the_metrics_refuse_ANNDATA_input_under_bulk_lognorm():
    """Copilot, PR #269. The hybrid AnnData form would silently mix two spaces.

    `_resolve_bulks` computes `prep.pseudobulk` -- a plain arithmetic mean of `adata.X` --
    and it has no `bulk_target_sum` with which to build `log1p(TS * P_g / sum_g P_g)`. The
    correction does not catch it either: `correction_for` returns `jk` without consulting
    `means` on this branch, so the metric would subtract a log-space jackknife from a
    raw-mean-space distance and return a plausible number instead of raising.

    No driver reaches this -- every one passes precomputed bulks (`run.py:275`) -- so this
    pins a PUBLIC-API path, and the same call under `lognorm` must still work, because there
    `prep.pseudobulk` IS the comparator.
    """
    from cell_eval2.metrics.delta import (
        distance_unbiased,
        mse_unbiased,
        mse_unbiased_capped,
    )
    from cell_eval2.prep import pseudobulk_bulk_lognorm_with_moments

    adata = _counts_adata_fp64(seed=4, per_group=6, g=10)
    perts, means, mom = pseudobulk_bulk_lognorm_with_moments(
        adata, "target", bulk_target_sum=1e6,
    )
    for fn in (mse_unbiased, mse_unbiased_capped):
        with pytest.raises(ValueError, match="precomputed bulks"):
            fn(pred=adata, real=adata, comparator="bulk_lognorm",
               pred_moments=mom, real_moments=mom, pert_col="target")
    with pytest.raises(ValueError, match="precomputed bulks"):
        distance_unbiased(real=adata, comparator="bulk_lognorm", real_moments=mom,
                          pert_col="target")

    # Precomputed bulks are the supported form and still work.
    out = mse_unbiased_capped(pred_bulk=(perts, means), real_bulk=(perts, means),
                              pred_moments=mom, real_moments=mom,
                              comparator="bulk_lognorm", control="non-targeting",
                              driver="test")
    assert set(out) == {"A", "B", "C"}

    # ...and the guard is scoped to the group-sum comparator: under `lognorm` the AnnData
    # form is exactly right, so it must NOT raise. (jk=None there, as an artifact built for
    # that comparator would carry.)
    from cell_eval2.moments import GroupMoments
    lm = GroupMoments(perts=mom.perts, counts=mom.counts, sumsq=mom.sumsq, jk=None)
    got = distance_unbiased(real=adata, comparator="lognorm", real_moments=lm,
                            pert_col="target")
    assert set(got) == {"A", "B", "C"}


def test_the_resident_and_streaming_bulks_AGREE_over_the_fp32_boundary():
    """#271. `streaming_bulk._streaming_pseudobulk_cpu` accumulates group sums into an fp64
    array from fp64-cast `.data`, and `gpu.bulk.GroupedMeanAccumulator` does the same -- so
    `prep._grouped_sums` reducing in the INPUT dtype made the resident path the odd one out,
    not the strict one. MEASURED before the fix: the two bulks diverged by 3.53e-10 on fp32
    input and agreed exactly on fp64, i.e. the same submission scored differently depending on
    which driver ran it.

    Bit equality on BOTH dtypes is the assertion: a tolerance would have passed before the fix
    too (3.53e-10 is inside any sane rtol), which is why the original divergence went unseen."""
    import anndata as anndata_mod
    import pandas as pd
    import scipy.sparse as sp

    from cell_eval2.prep import pseudobulk_bulk_lognorm
    from cell_eval2.streaming_bulk import _streaming_pseudobulk_cpu

    ts = 50_000.0
    for dtype in (np.float32, np.float64):
        # per-gene group sum 16,777,217 -- one above 2**24, where an fp32 reduction rounds
        X = np.array([[16777216.0, 1.0], [1.0, 1.0], [3.0, 4.0], [5.0, 6.0]], dtype=dtype)
        labels = np.array(["A", "A", "ctrl", "ctrl"])
        adata = anndata_mod.AnnData(
            X=sp.csr_matrix(X), obs=pd.DataFrame({"target": labels}),
            var=pd.DataFrame(index=["g0", "g1"]),
        )
        perts_res, means_res = pseudobulk_bulk_lognorm(adata, "target", bulk_target_sum=ts)
        out = _streaming_pseudobulk_cpu(
            [(adata.X, labels)], np.unique(labels), 2, adata.n_vars,
            ["bulk_lognorm"], None, bulk_target_sum=ts,
        )
        perts_str, means_str = out["bulk_lognorm"]
        np.testing.assert_array_equal(perts_res, perts_str)
        np.testing.assert_array_equal(
            means_res, means_str,
            err_msg=f"resident and streaming bulks disagree on {np.dtype(dtype).name} input",
        )
