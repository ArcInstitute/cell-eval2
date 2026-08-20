import anndata as ad
import numpy as np
import pandas as pd
import polars as pl
import pytest
from cell_eval2.config import EvalConfig
from cell_eval2 import partition_inmem
from _helpers import full_minus_moments, resolved_comparator


def _no_gpu():
    try:
        import cupy
        return cupy.cuda.runtime.getDeviceCount() == 0
    except Exception:
        return True


# Both tests here build a reference bundle first (needs gpudge DE), so this whole
# module skips on a no-GPU node -- pytest.importorskip("gpudge") would NOT skip
# (gpudge imports fine without a GPU) and the test would instead error inside the
# DE call. Run on the H100 node to actually exercise these.
pytestmark = pytest.mark.skipif(_no_gpu(), reason="needs a CUDA GPU (gpudge DE)")


def _cfg():
    from dataclasses import replace
    # Moment-consuming expression metrics are unavailable on this driver (#198).
    return replace(EvalConfig.v2(), pert_col="target_gene",
                   metrics=full_minus_moments())  # replace() keeps nested dataclasses


def _genes():
    return ["A", "B", "C", "D"] + [f"g{i}" for i in range(4, 12)]


def _real():
    rng = np.random.default_rng(0)
    perts = ["non-targeting"] * 60 + sum(([g] * 40 for g in ["A", "B", "C", "D"]), [])
    X = rng.poisson(3, size=(len(perts), 12)).astype(np.float32)
    genes = _genes()
    return ad.AnnData(
        X=X,
        obs={"target_gene": perts},
        var=pd.DataFrame({"gene": genes}, index=genes),
    )


def _pred_piece(perts):  # perturbed cells only, NO controls
    rng = np.random.default_rng(1)
    labels = sum(([g] * 40 for g in perts), [])
    X = rng.poisson(3, size=(len(labels), 12)).astype(np.float32)
    genes = _genes()
    return ad.AnnData(
        X=X,
        obs={"target_gene": labels},
        var=pd.DataFrame({"gene": genes}, index=genes),
    )


def test_score_piece_emits_partial_without_nsig_spearman(tmp_path):
    cfg = _cfg()
    partition_inmem.build_reference(_real(), config=cfg, cache_dir=str(tmp_path / "ref"),
                                    control_format="h5ad",
                                    comparator=resolved_comparator(cfg))
    df = partition_inmem.score_piece(_pred_piece(["A", "B"]), str(tmp_path / "ref"),
                                     config=cfg, piece_id="ab",
                                     partial_out=str(tmp_path / "partials"),
                                     comparator=resolved_comparator(cfg))
    assert set(df.filter(pl.col("metric") == "expr_mae")["perturbation"]) == {"A", "B"}
    assert df.filter(pl.col("metric") == "de_wilcoxon_nsig_spearman").height == 0
    # score_piece's contract is "{piece_id}.parquet + {piece_id}.json", so name them: a bare
    # `.glob(...)` is a generator and therefore truthy whether or not anything matched.
    assert (tmp_path / "partials" / "ab.parquet").is_file()
    assert (tmp_path / "partials" / "ab.json").is_file()


def test_score_piece_rejects_controls_in_piece(tmp_path):
    cfg = _cfg()
    partition_inmem.build_reference(_real(), config=cfg, cache_dir=str(tmp_path / "ref"),
                                    control_format="h5ad",
                                    comparator=resolved_comparator(cfg))
    bad = _pred_piece(["A"])
    bad.obs.loc[bad.obs.index[0], "target_gene"] = "non-targeting"
    with pytest.raises(ValueError, match="control"):
        partition_inmem.score_piece(
            bad, str(tmp_path / "ref"), config=cfg,
            comparator=resolved_comparator(cfg))


def test_bundle_reads_the_control_once_across_pieces(tmp_path, monkeypatch):
    cfg = _cfg()
    ref = str(tmp_path / "ref")
    partition_inmem.build_reference(
        _real(), config=cfg, cache_dir=ref, control_format="h5ad",
        comparator=resolved_comparator(cfg))
    real_load = partition_inmem.load_anndata
    loads = []

    def spy(*a, **k):
        loads.append(a[0])
        return real_load(*a, **k)

    monkeypatch.setattr(partition_inmem, "load_anndata", spy)
    bundle = partition_inmem._RefBundle(ref, cfg)
    for i, perts in enumerate([["A"], ["B"], ["C"]]):
        partition_inmem.score_piece(_pred_piece(perts), ref, config=cfg,
                                    piece_id=f"p{i}", bundle=bundle,
                                    comparator=resolved_comparator(cfg))
    control_loads = [p for p in loads if isinstance(p, str) and p.endswith("real_control.h5ad")]
    assert len(control_loads) == 1


def test_bundle_control_is_not_mutated_across_pieces(tmp_path):
    cfg = _cfg()
    ref = str(tmp_path / "ref")
    partition_inmem.build_reference(
        _real(), config=cfg, cache_dir=ref, control_format="h5ad",
        comparator=resolved_comparator(cfg))
    bundle = partition_inmem._RefBundle(ref, cfg)
    before = np.asarray(bundle.control_ad.X.todense() if hasattr(bundle.control_ad.X, "todense")
                        else bundle.control_ad.X).copy()
    shape = bundle.control_ad.shape
    for i, perts in enumerate([["A"], ["B"]]):
        partition_inmem.score_piece(_pred_piece(perts), ref, config=cfg,
                                    piece_id=f"p{i}", bundle=bundle,
                                    comparator=resolved_comparator(cfg))
    after = np.asarray(bundle.control_ad.X.todense() if hasattr(bundle.control_ad.X, "todense")
                       else bundle.control_ad.X)
    assert bundle.control_ad.shape == shape
    np.testing.assert_array_equal(before, after)


def test_bundle_gives_identical_results_to_no_bundle(tmp_path):
    cfg = _cfg()
    ref = str(tmp_path / "ref")
    partition_inmem.build_reference(
        _real(), config=cfg, cache_dir=ref, control_format="h5ad",
        comparator=resolved_comparator(cfg))
    without = partition_inmem.score_piece(
        _pred_piece(["A", "B"]), ref, config=cfg, piece_id="x",
        comparator=resolved_comparator(cfg))
    bundle = partition_inmem._RefBundle(ref, cfg)
    with_bundle = partition_inmem.score_piece(_pred_piece(["A", "B"]), ref, config=cfg,
                                              piece_id="x", bundle=bundle,
                                              comparator=resolved_comparator(cfg))
    assert without.equals(with_bundle)


def test_h1_shape_matches_whole_scoring_in_single_target_pieces(tmp_path):
    """Whole-data target resolution must survive every single-target DE slice."""
    from polars.testing import assert_frame_equal

    cfg = _cfg()
    ref = str(tmp_path / "ref")
    partition_inmem.build_reference(_real(), config=cfg, cache_dir=ref,
                                    control_format="h5ad",
                                    comparator=resolved_comparator(cfg))

    # H1_CGS shape: every target is a measured feature globally, but its own gene is
    # absent from that target's rows.
    real_de_path = tmp_path / "ref" / "real_de.parquet"
    real_de = pl.read_parquet(real_de_path)
    real_de.filter(pl.col("target") != pl.col("feature")).write_parquet(real_de_path)

    pred = _pred_piece(["A", "B", "C", "D"])
    bundle = partition_inmem._RefBundle(ref, cfg)
    whole = partition_inmem.score_piece(
        pred, ref, config=cfg, bundle=bundle, comparator=resolved_comparator(cfg))

    pieces = []
    for target in ["A", "B", "C", "D"]:
        piece = pred[pred.obs["target_gene"] == target].copy()
        pieces.append(partition_inmem.score_piece(
            piece, ref, config=cfg, bundle=bundle, comparator=resolved_comparator(cfg)))
    piecewise = pl.concat(pieces)

    direction_names = {
        name for name in whole["metric"].unique().to_list()
        if "direction_fidelity" in name or "direction_reach" in name
        or name.endswith("direction_coverage") or name.endswith("direction_yield")
        or name.endswith("direction_yield_raw")
    }
    assert direction_names
    whole_direction = whole.filter(pl.col("metric").is_in(direction_names)).sort(
        "perturbation", "metric")
    piecewise_direction = piecewise.filter(pl.col("metric").is_in(direction_names)).sort(
        "perturbation", "metric")
    assert_frame_equal(piecewise_direction, whole_direction)
