import anndata as ad
import numpy as np
import pandas as pd
import pytest

from cell_eval2 import EvalConfig, compute_metrics
from cell_eval2.baseline import (
    build_baseline_prediction,
    build_constant_prediction,
    generic_response_profile,
)


def _adata(values, labels, genes):
    X = np.asarray(values, dtype=np.float64)
    obs = pd.DataFrame({"target": list(labels), "guide": [f"gd{i}" for i in range(X.shape[0])]},
                       index=[f"c{i}" for i in range(X.shape[0])])
    var = pd.DataFrame({"gene_name": list(genes)}, index=list(genes))
    return ad.AnnData(X=X, obs=obs, var=var)


@pytest.fixture
def ref():
    return _adata([[-9., 1., 1., 1.], [2., -9., 2., 2.], [3., 3., -9., 3.],
                   [99., 98., 97., 96.]],
                  ["g0", "g1", "g2", "non-targeting"], ["g0", "g1", "g2", "g3"])


def _pred(ref):
    prof = generic_response_profile(ref, pert_col="target", control="non-targeting")
    return prof, build_baseline_prediction(prof, ref, pert_col="target",
                                           control="non-targeting")


def _tile_pred(ref):
    prof = generic_response_profile(ref, pert_col="target", control="non-targeting")
    return prof, build_baseline_prediction(prof, ref, pert_col="target",
                                           control="non-targeting", emit="tile")


def test_non_control_group_mean_follows_the_r_arithmetic(ref):
    """With one control donor, ``donor * (profile / ctrl)`` equals the profile exactly.
    This pins the r arithmetic only: identical/one-donor controls do not discriminate
    dispersed emission from tiling, which the variance and order-of-operations tests do."""
    prof, pred = _pred(ref)
    assert pred.shape == ref.shape
    assert pred.X.dtype == np.float32
    expected = prof.values.astype(np.float32)
    is_ctrl = np.asarray(ref.obs["target"]) == "non-targeting"
    np.testing.assert_array_equal(np.asarray(pred.X)[~is_ctrl].mean(axis=0), expected)


def test_tile_non_control_rows_are_the_profile(ref):
    """The legacy arm remains byte-for-byte reproducible for pre-#234 comparisons."""
    prof, pred = _tile_pred(ref)
    expected = prof.values.astype(np.float32)
    is_ctrl = np.asarray(ref.obs["target"]) == "non-targeting"
    for i in np.flatnonzero(~is_ctrl):
        np.testing.assert_array_equal(np.asarray(pred.X)[i], expected)


def test_control_rows_carry_the_REAL_CONTROL_CELLS(ref):
    """Reverses an earlier draft. The profile in the control rows makes the prediction's
    own control constant, so under control_source='pred' the predicted delta is
    identically zero and the DE is fully tied (measured delta_pearson -1.0). The
    generic-response hypothesis is about PERTURBATION responses and says nothing about
    the control, so handing the baseline a correct control removes an unrelated handicap
    rather than granting charity."""
    prof, pred = _pred(ref)
    is_ctrl = np.asarray(ref.obs["target"]) == "non-targeting"
    ctrl = np.asarray(pred.X)[is_ctrl]
    # EXACT at float32: the cast is the only transformation applied to the control cells
    assert np.array_equal(ctrl, np.asarray(ref.X)[is_ctrl].astype(np.float32))
    # ...and they are NOT the profile -- the rejected implementation, asserted against
    assert not np.allclose(ctrl[0], prof.values.astype(np.float32))


def test_mirrors_the_WHOLE_obs_and_var(ref):
    """The whole obs is copied, not obs[[pert_col]]: a pert_col-only copy drops
    de.replicate_col, which the deseq2 backend requires in the prediction's obs
    (deseq2_de.py:26-32), and the real-control concat keeps only shared columns so it
    would vanish. Cell counts per perturbation must match too -- they drive the DE
    p-values, so a baseline with fabricated counts is a comparator for a different
    experiment."""
    _, pred = _pred(ref)
    assert list(pred.obs.columns) == list(ref.obs.columns)
    assert "guide" in pred.obs.columns
    assert list(pred.obs.index) == list(ref.obs.index)
    assert pred.obs["target"].tolist() == ref.obs["target"].tolist()
    assert list(pred.var.index) == list(ref.var.index)
    assert list(pred.var.columns) == list(ref.var.columns)


def test_both_control_sources_AGREE_with_real_control_rows(synthetic_pair):
    """The decisive property (design section 4.2), asserted as a difference rather than
    assumed: with the real control's cells in the control rows the two control_source
    settings give the same numbers, so the baseline needs no forced override. With the
    profile in them (the rejected build) they diverge.

    The DE arm is included on purpose: control_source governs the DE REFERENCE, and on a
    GPU-less host that is the ad.concat(pred_non_ctrl, real_ctrl) path at run.py:546, which
    a delta-only test never touches. Measured: the DE metrics agree EXACTLY; delta_pearson
    agrees to a relative 6.5e-9, because X is float32 and this fixture is float64 (design
    section 4.2) -- so the two families get different tolerances rather than one loose one.
    """
    _, real = synthetic_pair
    prof = generic_response_profile(real, pert_col="target", control="non-targeting")
    good = build_baseline_prediction(prof, real, pert_col="target",
                                     control="non-targeting")
    # the rejected build: profile everywhere, including the control rows
    bad = good.copy()
    np.asarray(bad.X)[:] = prof.values.astype(np.float32)

    def _score(pred, source, metrics):
        cfg = EvalConfig(metrics=metrics, pert_col="target",
                         control="non-targeting", input_type="lognorm",
                         validate_input=False, control_source=source,
                         allow_fractional_counts=True)
        return (compute_metrics(pred, real, config=cfg)
                .sort(["metric", "perturbation"])["value"].to_list())

    # rel=1e-6 for BOTH families. The measured pdex result was EXACT for the DE metrics,
    # but control_source="real" and "pred" take structurally different paths (gpudge
    # external-ref at run.py:472 vs the CPU concat at run.py:533), so exact equality is not
    # a backend-independent guarantee and must not be asserted as one.
    de = ["de_wilcoxon_overlap", "de_wilcoxon_precision"]
    assert _score(good, "real", de) == pytest.approx(_score(good, "pred", de), rel=1e-6,
                                                     nan_ok=True)
    delta_real = _score(good, "real", ["delta_pearson"])
    delta_pred = _score(good, "pred", ["delta_pearson"])
    assert delta_real == pytest.approx(delta_pred, rel=1e-6, nan_ok=True)
    # the rejected build collapses to the degenerate -1.0
    assert _score(bad, "pred", ["delta_pearson"]) != pytest.approx(delta_pred, rel=1e-6,
                                                                  nan_ok=True)


def test_is_writable_not_a_broadcast_view(ref):
    """np.broadcast_to would save no memory (the pipeline densifies anyway) and would
    make .X read-only, so an in-place downstream write would crash."""
    _, pred = _tile_pred(ref)
    X = np.asarray(pred.X)
    X[0, 0] = 123.0                      # must not raise
    assert X[0, 0] == 123.0


def test_gene_axis_mismatch_raises(ref):
    prof = generic_response_profile(ref, pert_col="target", control="non-targeting")
    other = _adata([[1., 2.], [3., 4.]], ["g0", "non-targeting"], ["gX", "gY"])
    with pytest.raises(ValueError, match="var index"):
        build_baseline_prediction(prof, other, pert_col="target",
                                  control="non-targeting")


def test_missing_pert_col_in_template_raises(ref):
    prof = generic_response_profile(ref, pert_col="target", control="non-targeting")
    with pytest.raises(ValueError, match="nope"):
        build_baseline_prediction(prof, ref, pert_col="nope", control="non-targeting")


def test_missing_control_in_template_raises(ref):
    prof = generic_response_profile(ref, pert_col="target", control="non-targeting")
    with pytest.raises(ValueError, match="control"):
        build_baseline_prediction(prof, ref, pert_col="target", control="nope")


def test_sparse_template_yields_sparse_prediction(ref):
    """Dispersed emission must retain sparse storage; densification was the old tile ceiling."""
    import scipy.sparse as sp
    sparse = ad.AnnData(X=sp.csr_matrix(ref.X), obs=ref.obs.copy(), var=ref.var.copy())
    prof = generic_response_profile(sparse, pert_col="target", control="non-targeting")
    pred = build_baseline_prediction(prof, sparse, pert_col="target",
                                     control="non-targeting")
    is_ctrl = np.asarray(ref.obs["target"]) == "non-targeting"
    assert sp.issparse(pred.X)
    np.testing.assert_allclose(pred.X[is_ctrl].toarray(),
                               np.asarray(ref.X)[is_ctrl].astype(np.float32))


def test_sparse_h5ad_path_template_works(ref, tmp_path):
    """Same reason as the profile's sparse-path test: the template is read NON-backed,
    because a backed sparse read breaks the grouped-mean path and gains nothing here."""
    import scipy.sparse as sp
    p = tmp_path / "ref.h5ad"
    ad.AnnData(X=sp.csr_matrix(ref.X), obs=ref.obs.copy(), var=ref.var.copy()).write_h5ad(p)
    prof = generic_response_profile(ref, pert_col="target", control="non-targeting")
    pred = build_baseline_prediction(prof, str(p), pert_col="target",
                                     control="non-targeting")
    assert pred.shape == ref.shape
    is_ctrl = np.asarray(ref.obs["target"]) == "non-targeting"
    assert sp.issparse(pred.X)
    assert np.array_equal(pred.X[is_ctrl].toarray(),
                          np.asarray(ref.X)[is_ctrl].astype(np.float32))
    ref.write_h5ad(p)          # reopenable for writing -> nothing holds the file


def test_build_constant_prediction_warns_and_still_tiles(ref):
    """The deprecated entry point remains the exact legacy escape hatch, never the default."""
    prof = generic_response_profile(ref, pert_col="target", control="non-targeting")
    with pytest.warns(DeprecationWarning, match="known-biased"):
        pred = build_constant_prediction(prof, ref, pert_col="target",
                                         control="non-targeting")
    expected = prof.values.astype(np.float32)
    is_ctrl = np.asarray(ref.obs["target"]) == "non-targeting"
    np.testing.assert_array_equal(np.asarray(pred.X)[~is_ctrl],
                                  np.tile(expected, ((~is_ctrl).sum(), 1)))
