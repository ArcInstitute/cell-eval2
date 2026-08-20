import numpy as np
import pytest

from _helpers import _dispatch_cfg
from cell_eval2.catalog import CATALOG
from cell_eval2.config import EvalConfig
from cell_eval2.norm import EXPR_COMPARATOR
from cell_eval2.run import effective_normalization, precompute_cache


MOVED = [
    "expr_mae",
    "expr_mse",
    "expr_mse_unbiased",
    "expr_mse_unbiased_capped",
    "expr_distance_unbiased",
    "expr_mse_unbiased_capped_norm",
]


@pytest.mark.parametrize("name", MOVED)
def test_expr_metrics_declare_the_comparator_token(name):
    assert CATALOG[name].normalization == EXPR_COMPARATOR


@pytest.mark.parametrize("name", MOVED)
def test_expr_metrics_resolve_to_bulk_lognorm_on_v2_counts(name):
    assert effective_normalization(CATALOG[name], "bulk_lognorm") == "bulk_lognorm"


@pytest.mark.parametrize("name", MOVED)
def test_expr_metrics_still_resolve_to_lognorm_on_the_fallback(name):
    """v1 and any run with a lognorm side keep the old comparator and correction."""
    assert effective_normalization(CATALOG[name], "lognorm") == "lognorm"


EVERY_ANNDATA_METRIC = [
    "delta_mae",
    "delta_mse",
    "delta_pearson",
    "expr_distance_unbiased",
    "expr_mae",
    "expr_mse",
    "expr_mse_unbiased",
    "expr_mse_unbiased_capped",
    "expr_mse_unbiased_capped_norm",
    "expr_real_mass_ratio",
    "pds_cosine",
    "pds_l1",
    "pds_l2",
]


def test_every_anndata_metric_declares_the_comparator_token():
    """All 13 anndata metrics must positively declare the comparator token."""
    anndata = sorted(n for n, spec in CATALOG.items() if spec.kind == "anndata")
    assert anndata == sorted(EVERY_ANNDATA_METRIC)
    assert [n for n in anndata if CATALOG[n].normalization != EXPR_COMPARATOR] == []


def test_exactly_ONE_scale_is_shipped_and_it_is_the_current_one():
    """Was `test_v5_replaces_v4_and_v4_is_gone`, back when this file's own change minted `_v5`.
    The name went stale four mints ago while the assertion stayed correct, which is the argument
    for naming it after the INVARIANT rather than after whichever pair is current -- the registry
    ships exactly one scale, and `tests/test_scales.py` owns which one and why it was minted."""
    from cell_eval2.scales import SCALES

    assert list(SCALES) == ["low-random_high-1_v10"]


def test_the_dispatcher_now_resolves_the_moved_metric_to_bulk_lognorm(tmp_path):
    """The moved metric reads the jk-bearing artifact under its effective key."""
    from cell_eval2.moments import GroupMoments
    from cell_eval2.run import dispatch_anndata_metrics

    perts = np.array(["non-targeting", "A"])
    # ⚠️ Issue #172: this metric drops each perturbation's own gene and RAISES when no target
    # resolves, so the panel needs a gene named 'A'. The appended column is ALL ZERO, which
    # leaves the expected 1.75 EXACT -- it adds 0 to the squared distance and, being the excluded
    # column, takes the divisor back to 2. `jk` is supplied directly here, so the correction is
    # unaffected by construction.
    pred_bulk = np.array([[1.0, 2.0, 0.0], [3.0, 5.0, 0.0]])
    real_bulk = np.array([[1.0, 2.0, 0.0], [4.0, 7.0, 0.0]])

    def moments(sumsq, jk):
        return GroupMoments(
            perts=perts,
            counts=np.array([4.0, 4.0]),
            sumsq=np.array(sumsq),
            jk=np.array(jk),
        )

    rows = dispatch_anndata_metrics(
        ["expr_mse_unbiased"],
        {"bulk_lognorm": (perts, pred_bulk)},
        {"bulk_lognorm": (perts, real_bulk)},
        np.array(["g0", "g1", "A"]),
        _dispatch_cfg(),
        comparator="bulk_lognorm",
        pred_moments={"bulk_lognorm": moments([30.0, 140.0], [0.0, 0.9])},
        real_moments={"bulk_lognorm": moments([30.0, 270.0], [0.0, 0.6])},
        driver="test",
    )
    assert {row["perturbation"]: row["value"] for row in rows}["A"] == pytest.approx(1.75)


def _assert_bulk_lognorm_cache_carries_jk(cache_dir):
    paths = list(cache_dir.glob("pseudobulk_moments_bulk_lognorm*.npz"))
    assert len(paths) == 1
    with np.load(paths[0]) as artifact:
        assert set(artifact.files) == {"perts", "means", "counts", "sumsq", "jk"}
        assert artifact["jk"].shape == artifact["perts"].shape
        assert np.all(artifact["jk"] >= 0)
        assert np.any(artifact["jk"] > 0)


def test_the_real_bulk_lognorm_cache_persists_the_jackknife(
    tmp_path, synthetic_counts_pair
):
    _pred, real = synthetic_counts_pair
    cache_dir = tmp_path / "real"
    cfg = EvalConfig(
        metrics=["expr_mse_unbiased"], cache_real=str(cache_dir), device="cpu"
    )
    precompute_cache(real, side="real", config=cfg, comparator="bulk_lognorm")
    _assert_bulk_lognorm_cache_carries_jk(cache_dir)


def test_the_pred_bulk_lognorm_cache_persists_the_jackknife(
    tmp_path, synthetic_counts_pair
):
    pred, _real = synthetic_counts_pair
    cache_dir = tmp_path / "pred"
    cfg = EvalConfig(
        metrics=["expr_mse_unbiased"], cache_pred=str(cache_dir), device="cpu"
    )
    precompute_cache(pred, side="pred", config=cfg, comparator="bulk_lognorm")
    _assert_bulk_lognorm_cache_carries_jk(cache_dir)
