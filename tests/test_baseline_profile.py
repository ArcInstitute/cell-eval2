import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from cell_eval2.baseline import GenericProfile, generic_response_profile


def _adata(values, labels, genes):
    """One cell per row; `values` is (n_cells, n_genes)."""
    X = np.asarray(values, dtype=np.float64)
    obs = pd.DataFrame({"target": list(labels)},
                       index=[f"c{i}" for i in range(X.shape[0])])
    var = pd.DataFrame(index=list(genes))
    return ad.AnnData(X=X, obs=obs, var=var)


def _hand_fixture():
    """3 perturbations named after genes g0,g1,g2 + a control, 4 genes, 1 cell each.

    Per-perturbation pseudobulk (one cell each, so the row IS the mean):
        g0 : [-9,  1,  1,  1]     <- extreme self-knockdown at its own gene
        g1 : [ 2, -9,  2,  2]
        g2 : [ 3,  3, -9,  3]
        ctrl: [99, 99, 99, 99]    <- must be excluded entirely
    """
    return _adata(
        [[-9., 1., 1., 1.], [2., -9., 2., 2.], [3., 3., -9., 3.], [99.]*4],
        ["g0", "g1", "g2", "non-targeting"],
        ["g0", "g1", "g2", "g3"],
    )


def test_plain_mean_excludes_control_only():
    p = generic_response_profile(_hand_fixture(), pert_col="target",
                                 control="non-targeting", exclude_target_gene=False)
    assert isinstance(p, GenericProfile)
    assert p.n_perturbations == 3
    assert p.n_excluded == 0
    assert p.exclude_target_gene is False
    # column means over the three non-control rows
    np.testing.assert_allclose(p.values, [(-9+2+3)/3, (1-9+3)/3, (1+2-9)/3, (1+2+3)/3])
    np.testing.assert_array_equal(p.genes, ["g0", "g1", "g2", "g3"])


def test_self_target_exclusion_closed_form():
    p = generic_response_profile(_hand_fixture(), pert_col="target",
                                 control="non-targeting", exclude_target_gene=True)
    assert p.n_excluded == 3          # g0, g1, g2 each dropped from their own gene
    # g0: drop pert g0's -9 -> mean(2,3); g1: drop -9 -> mean(1,3);
    # g2: drop -9 -> mean(1,2);  g3 is nobody's target -> unchanged mean of 3
    np.testing.assert_allclose(p.values, [(2+3)/2, (1+3)/2, (1+2)/2, (1+2+3)/3])


def test_exclusion_DISCRIMINATES():
    """The two arms must actually differ, and by the predicted amount.

    A fixture that merely *contains* a self-target gene proves nothing -- the rejected
    implementation (no exclusion) is computed here and the difference asserted, per
    section 8.1 of the direction-metrics spec.
    """
    on = generic_response_profile(_hand_fixture(), pert_col="target",
                                  control="non-targeting", exclude_target_gene=True)
    off = generic_response_profile(_hand_fixture(), pert_col="target",
                                   control="non-targeting", exclude_target_gene=False)
    assert not np.allclose(on.values, off.values)
    # the three targeted genes move; g3 does not
    assert not np.isclose(on.values[0], off.values[0])
    assert np.isclose(on.values[3], off.values[3])
    # exact: excluding one term x from a mean of n shifts it to (n*m - x)/(n-1)
    n, m, x = 3, off.values[0], -9.0
    assert on.values[0] == pytest.approx((n * m - x) / (n - 1))


def test_a_zero_resolving_panel_RAISES(caplog):
    """#285. Guide-ID labels match no gene, so the flag would do nothing -- and this builder
    is the 0 END of the competition scale, not a diagnostic.

    ⚠️ This test used to assert the OPPOSITE: a warning, `n_excluded == 0`, and a profile
    numerically identical to `exclude_target_gene=False`. The stamp made the no-op auditable
    after the fact, which is weaker than refusing it. Routing through the shared
    `resolve_exclusion_columns` (#253) brings its zero-resolve raise with it, and the two
    cannot be separated without re-splitting the behaviour #248 unified."""
    a = _adata([[1., 2.], [3., 4.], [5., 6.]],
               ["GUIDE_A", "GUIDE_B", "non-targeting"], ["g0", "g1"])
    with pytest.raises(ValueError, match="NO perturbation resolves"):
        generic_response_profile(a, pert_col="target", control="non-targeting",
                                 exclude_target_gene=True)
    # ...and the message names BOTH escape hatches, because the remedy is one of them
    with pytest.raises(ValueError) as e:
        generic_response_profile(a, pert_col="target", control="non-targeting",
                                 exclude_target_gene=True)
    assert "target_gene_map" in str(e.value) and "exclude_target_gene=False" in str(e.value)
    # ...and it states THIS caller's consequence, not the shared resolver's. The helper's own
    # text is written for the discrimination metrics ("the ranked vector", "inflates the
    # discrimination score"), and a baseline builder computes neither: what it loses is the
    # exclusion itself, on the arm that defines the 0 end of the scale.
    assert "0 END of the competition scale" in str(e.value)
    assert "NO perturbation resolves" in str(e.value)      # the resolver's diagnosis, kept
    assert e.value.__cause__ is not None                   # chained, not swallowed
    # the flag being OFF still works: nothing is resolved, so nothing can fail to resolve
    off = generic_response_profile(a, pert_col="target", control="non-targeting",
                                   exclude_target_gene=False)
    np.testing.assert_allclose(off.values, [2., 3.])
    assert off.n_excluded == 0


def test_a_construct_ID_panel_WITH_a_map_excludes_exactly_as_the_symbol_panel_does():
    """#253/#285's whole point: the map makes a guide-level panel score the same as the
    gene-symbol panel it stands for. Before the fix `n_excluded` was 0 here and the profile
    was the plain mean -- with each perturbation's own on-target knockdown left in.

    Asserted BOTH ways, and the independent one is load-bearing: against the closed form
    `test_self_target_exclusion_closed_form` pins for this same fixture (a hand-typed vector,
    computed from the pre-change definition), and against the symbol panel's own answer. The
    symbol-panel comparison alone would only show that two invocations of the CURRENT
    implementation agree with each other, which is true of any implementation."""
    x = [[-9., 1., 1., 1.], [2., -9., 2., 2.], [3., 3., -9., 3.], [99.] * 4]
    genes = ["g0", "g1", "g2", "g3"]
    symbols = _adata(x, ["g0", "g1", "g2", "non-targeting"], genes)
    guides = _adata(x, ["g0-1", "g1-1", "g2-1", "non-targeting"], genes)

    want = generic_response_profile(symbols, pert_col="target", control="non-targeting",
                                    exclude_target_gene=True)
    got = generic_response_profile(guides, pert_col="target", control="non-targeting",
                                   exclude_target_gene=True,
                                   target_gene_map={"g0-1": "g0", "g1-1": "g1",
                                                    "g2-1": "g2"})
    # g0: drop pert g0's -9 -> mean(2,3); g1: drop -9 -> mean(1,3); g2: drop -9 -> mean(1,2);
    # g3 is nobody's target -> unchanged mean of 3. The OLD algorithm's answer, hand-computed.
    np.testing.assert_array_equal(got.values, [(2 + 3) / 2, (1 + 3) / 2, (1 + 2) / 2,
                                               (1 + 2 + 3) / 3])
    np.testing.assert_array_equal(got.values, want.values)   # BIT-identical, not approx
    assert got.n_excluded == want.n_excluded == 3


def test_a_gene_symbol_panel_is_UNCHANGED_by_the_map_parameter():
    """The other half of the safety argument. A panel whose labels ARE symbols resolves
    through the raw-label fallback exactly as it did before, with or without a map -- which
    is why the three official val panels (all 300 labels resolve) cannot move.

    Pinned to the hand-computed closed form as well, not only to each other: an
    implementation that ignored the map entirely would satisfy the two-invocation comparison
    on its own."""
    plain = generic_response_profile(_hand_fixture(), pert_col="target",
                                     control="non-targeting", exclude_target_gene=True)
    mapped = generic_response_profile(_hand_fixture(), pert_col="target",
                                      control="non-targeting", exclude_target_gene=True,
                                      target_gene_map={"nothing": "relevant"})
    np.testing.assert_array_equal(plain.values, [(2 + 3) / 2, (1 + 3) / 2, (1 + 2) / 2,
                                                 (1 + 2 + 3) / 3])
    np.testing.assert_array_equal(plain.values, mapped.values)
    assert plain.n_excluded == mapped.n_excluded == 3


def test_a_PARTIALLY_resolving_panel_does_not_raise():
    """The gate is zero-resolution, not full resolution: a target that is simply not a
    measured gene is ordinary biology (or the CPM filter), and excluding the ones that do
    resolve is strictly better than excluding none."""
    a = _adata([[1., 2., 3.], [4., 5., 6.], [7., 8., 9.], [0., 0., 0.]],
               ["g0-1", "g1-1", "offpanel-1", "non-targeting"], ["g0", "g1", "g2"])
    p = generic_response_profile(a, pert_col="target", control="non-targeting",
                                 exclude_target_gene=True,
                                 target_gene_map={"g0-1": "g0", "g1-1": "g1",
                                                  "offpanel-1": "offpanel"})
    assert p.n_excluded == 2
    # g0 drops pert 0's 1.0 -> mean(4,7); g1 drops pert 1's 5.0 -> mean(2,8);
    # g2 is nobody's resolved target -> the plain mean of all three
    np.testing.assert_allclose(p.values, [(4 + 7) / 2, (2 + 8) / 2, (3 + 6 + 9) / 3])


def test_a_gene_targeted_by_EVERY_perturbation_raises():
    """Newly reachable through the map, which is many-to-one where a raw label was not:
    two guides for one gene on a two-perturbation panel leaves `count == 0` for that gene.
    Measured as a NaN in the middle of the profile before the guard -- the `n < 2` raise one
    granularity down."""
    a = _adata([[1., 2.], [3., 4.], [0., 0.]], ["g0-1", "g0-2", "non-targeting"],
               ["g0", "g1"])
    with pytest.raises(ValueError, match="no perturbation contributing"):
        generic_response_profile(a, pert_col="target", control="non-targeting",
                                 exclude_target_gene=True,
                                 target_gene_map={"g0-1": "g0", "g0-2": "g0"})


def test_no_non_control_perturbations_raises():
    a = _adata([[1., 2.]], ["non-targeting"], ["g0", "g1"])
    with pytest.raises(ValueError, match="non-control perturbation"):
        generic_response_profile(a, pert_col="target", control="non-targeting")


def test_single_perturbation_with_exclusion_raises():
    a = _adata([[1., 2.], [3., 4.]], ["g0", "non-targeting"], ["g0", "g1"])
    with pytest.raises(ValueError, match="at least 2 non-control"):
        generic_response_profile(a, pert_col="target", control="non-targeting",
                                 exclude_target_gene=True)
    # ...but it is fine with the exclusion off
    p = generic_response_profile(a, pert_col="target", control="non-targeting",
                                 exclude_target_gene=False)
    np.testing.assert_allclose(p.values, [1., 2.])


def test_missing_control_raises():
    a = _adata([[1., 2.], [3., 4.]], ["g0", "g1"], ["g0", "g1"])
    with pytest.raises(ValueError, match="control"):
        generic_response_profile(a, pert_col="target", control="non-targeting")


def test_missing_pert_col_raises():
    a = _adata([[1., 2.]], ["g0"], ["g0", "g1"])
    with pytest.raises(ValueError, match="wrong_col"):
        generic_response_profile(a, pert_col="wrong_col", control="non-targeting")


def test_duplicate_var_names_raise_only_with_exclusion():
    a = _adata([[1., 2.], [3., 4.], [5., 6.]],
               ["g0", "g0dup", "non-targeting"], ["g0", "g0"])
    with pytest.raises(ValueError, match="unique"):
        generic_response_profile(a, pert_col="target", control="non-targeting",
                                 exclude_target_gene=True)
    # the plain mean does not need to identify a target gene, so it is allowed
    p = generic_response_profile(a, pert_col="target", control="non-targeting",
                                 exclude_target_gene=False)
    np.testing.assert_allclose(p.values, [2., 3.])


def test_sparse_equals_dense():
    dense = _hand_fixture()
    sparse = ad.AnnData(X=sp.csr_matrix(dense.X), obs=dense.obs.copy(),
                        var=dense.var.copy())
    for flag in (True, False):
        d = generic_response_profile(dense, pert_col="target", control="non-targeting",
                                     exclude_target_gene=flag)
        s = generic_response_profile(sparse, pert_col="target", control="non-targeting",
                                     exclude_target_gene=flag)
        np.testing.assert_allclose(d.values, s.values)


def test_multi_cell_perturbations_use_pert_means_not_pooled():
    """Equal weight per perturbation (Notion section 1.4), not per cell: a perturbation
    with many cells must not dominate. Pooled would give a different answer."""
    a = _adata([[0.], [0.], [0.], [12.], [99.]],
               ["g0", "g0", "g0", "gB", "non-targeting"], ["g0"])
    p = generic_response_profile(a, pert_col="target", control="non-targeting",
                                 exclude_target_gene=False)
    np.testing.assert_allclose(p.values, [6.0])       # mean(0, 12), NOT mean(0,0,0,12)=3


def test_path_input_does_not_hold_the_file_open(tmp_path):
    """A path input is read NON-backed, so nothing holds the file. Asserted by reopening
    it for writing, which is the observable consequence."""
    p = tmp_path / "ref.h5ad"
    _hand_fixture().write_h5ad(p)
    prof = generic_response_profile(str(p), pert_col="target", control="non-targeting",
                                    exclude_target_gene=False)
    assert prof.n_perturbations == 3
    _hand_fixture().write_h5ad(p)


def test_SPARSE_h5ad_path_works(tmp_path):
    """Regression for the reason the reference is never opened backed: prep._grouped_means
    decides sparsity once with issparse(X), which is False for a backed h5ad CSR matrix
    (an AnnData _CSRDataset), after which each X[rows] slice returns a scipy CSR and hits
    the DENSE branch at prep.py:67. Measured failure on a backed sparse reference:
    `ValueError: setting an array element with a sequence`. Sparse is the ordinary
    real-world h5ad layout, so this must be a test, not a comment."""
    dense = _hand_fixture()
    p = tmp_path / "sparse.h5ad"
    ad.AnnData(X=sp.csr_matrix(dense.X), obs=dense.obs.copy(),
               var=dense.var.copy()).write_h5ad(p)
    got = generic_response_profile(str(p), pert_col="target", control="non-targeting",
                                   exclude_target_gene=True)
    want = generic_response_profile(dense, pert_col="target", control="non-targeting",
                                    exclude_target_gene=True)
    np.testing.assert_allclose(got.values, want.values)


def test_caller_supplied_backed_object_is_materialized_locally(tmp_path):
    """A caller may hand us an already-backed AnnData. We materialize a local copy rather
    than crash -- and we do NOT close their handle, which is theirs, not ours."""
    import anndata
    p = tmp_path / "backed.h5ad"
    dense = _hand_fixture()
    ad.AnnData(X=sp.csr_matrix(dense.X), obs=dense.obs.copy(),
               var=dense.var.copy()).write_h5ad(p)
    handle = anndata.read_h5ad(p, backed="r")
    try:
        prof = generic_response_profile(handle, pert_col="target",
                                        control="non-targeting", exclude_target_gene=False)
        assert prof.n_perturbations == 3
        # isbacked stays True after close (AnnData keeps the filename), so assert the FILE
        assert handle.file.is_open                # still open: we did not close it
    finally:
        handle.file.close()


def test_missing_control_is_detected_before_the_pseudobulk_pass(monkeypatch):
    """The control check reads obs, so it must fire BEFORE the (expensive) grouped-mean
    pass -- otherwise a mistyped control label costs a full scan of the reference first."""
    import cell_eval2.baseline as bl

    def _boom(*a, **k):
        raise AssertionError("pseudobulk ran before the control label was validated")

    monkeypatch.setattr(bl, "pseudobulk", _boom)
    a = _adata([[1., 2.], [3., 4.]], ["g0", "g1"], ["g0", "g1"])
    with pytest.raises(ValueError, match="control"):
        generic_response_profile(a, pert_col="target", control="non-targeting")
