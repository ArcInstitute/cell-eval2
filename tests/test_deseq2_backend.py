"""Tests for the opt-in ``deseq2`` DE backend (deseq2_gpu NB-GLM adapter).

Pure-adapter tests (config, ``_grouped_sums``, registry resolution via monkeypatch,
``_validate``/``_pseudobulk``) run in CI without the engine. Every test that fits via
deseq2_gpu starts with ``pytest.importorskip("deseq2_gpu")`` so it SKIPS in CI (where the
private engine is absent) and runs locally against an editable install. Raw-count toy
AnnData comes from the shared ``toy_de_adata`` factory fixture (tests/conftest.py).
"""
from __future__ import annotations

import numpy as np
import polars as pl
import pytest
from scipy.sparse import csr_matrix

from cell_eval2 import de_compute, deseq2_de
from cell_eval2.config import DEParams
from cell_eval2.prep import _group_row_index, _grouped_sums


def test_deparams_accepts_deseq2_backend_and_replicate_col():
    p = DEParams(backend="deseq2", replicate_col="guide")
    assert p.backend == "deseq2"
    assert p.replicate_col == "guide"


def test_deparams_method_tracks_backend():
    # method is DERIVED provenance: it always tracks the backend (deseq2 -> "deseq2";
    # auto/rank -> "wilcoxon") so the de_<method>_* cache keys + run_params can't drift.
    assert DEParams(backend="deseq2", replicate_col="g").method == "deseq2"
    assert DEParams(backend="pdex").method == "wilcoxon"
    assert DEParams().method == "wilcoxon"                            # default backend=auto
    # an inconsistent explicit method is canonicalized to the backend (backend wins)
    assert DEParams(backend="pdex", method="deseq2").method == "wilcoxon"


def test_deparams_frozen_method_cannot_drift():
    # method is derived once in __post_init__, so DEParams must be frozen — else mutating
    # backend after construction would leave a stale method and reintroduce the rank-cache
    # collision (Codex). dataclasses.replace (how every internal update is done) reruns it.
    from dataclasses import FrozenInstanceError, replace
    assert replace(DEParams(backend="auto"), backend="deseq2").method == "deseq2"
    assert replace(DEParams(backend="deseq2", replicate_col="g"), backend="pdex").method == "wilcoxon"
    with pytest.raises(FrozenInstanceError):
        DEParams(backend="auto").backend = "deseq2"
    with pytest.raises(FrozenInstanceError):
        DEParams(backend="deseq2", replicate_col="g").method = "wilcoxon"


def test_deparams_replicate_col_defaults_none():
    assert DEParams().replicate_col is None


def test_deparams_rejects_unknown_backend():
    with pytest.raises(ValueError, match="backend must be one of"):
        DEParams(backend="bogus")


def test_deparams_rejects_non_string_replicate_col():
    with pytest.raises(ValueError, match="replicate_col"):
        DEParams(replicate_col=123)


def test_grouped_sums_dense_and_sparse_match_manual():
    X = np.array([[1., 2.], [3., 4.], [10., 0.], [0., 5.]])
    labels = np.array(["a", "a", "b", "b"])
    uniq, order, bounds = _group_row_index(labels)
    dense = _grouped_sums(X, order, bounds, uniq.size)
    sparse = _grouped_sums(csr_matrix(X), order, bounds, uniq.size)
    # groups sorted -> uniq == ['a','b']; a-sum=[4,6], b-sum=[10,5]
    exp = {"a": np.array([4., 6.]), "b": np.array([10., 5.])}
    for i, g in enumerate(uniq):
        np.testing.assert_allclose(dense[i], exp[g])
        np.testing.assert_allclose(sparse[i], exp[g])


def test_the_deseq2_pseudobulk_REDUCES_WIDE_TOO_and_that_is_the_ruling():
    """#271. `prep._grouped_sums` is shared by the `bulk_lognorm` comparator and by this backend's
    pseudobulk, and it reduces WIDE for both -- ONE policy, no per-caller `widen=` flag a future
    caller can forget. Ruled deliberately (Alex 2026-08-18) rather than inherited: #264 PR2 left
    the reduction narrow partly because widening moves deseq2's numbers.

    Driven through `deseq2_de._pseudobulk` rather than through the shared helper, so it fails if
    this backend stops calling it or reduces narrowly itself -- the first cut asserted
    `prep._grouped_sums is _grouped_sums`, which compares one import with the same import and
    would have stayed green either way (codex review).

    The fixture is fp32 counts whose per-(condition, replicate) sum crosses float32's 2**24, which
    is where a narrow reduction loses a count. No engine needed: `_pseudobulk` is pure adapter."""
    import anndata as ad_mod
    import pandas as pd_mod

    X = csr_matrix(np.array([[16777216.0, 1.0],
                             [1.0, 1.0],
                             [5.0, 3.0],
                             [7.0, 2.0]], dtype=np.float32))
    obs = pd_mod.DataFrame(
        {"target": ["A", "A", "ctrl", "ctrl"], "guide": ["A1", "A1", "c1", "c2"]},
        index=[f"c{i}" for i in range(4)])
    adata = ad_mod.AnnData(X=X, obs=obs, var=pd_mod.DataFrame(index=["g0", "g1"]))

    counts_gxs, conditions, replicates, genes = deseq2_de._pseudobulk(
        adata, pert_col="target", control="ctrl", replicate_col="guide")
    # genes x samples; the (A, A1) sample is the one that crosses the boundary
    col = [i for i, (c, r) in enumerate(zip(conditions, replicates)) if (c, r) == ("A", "A1")]
    assert len(col) == 1, f"expected one (A, A1) sample; got {list(zip(conditions, replicates))}"
    assert counts_gxs[0, col[0]] == 16777217.0, (
        "the deseq2 pseudobulk reduced in the input dtype -- if this path was deliberately "
        "narrowed, that decision belongs in deseq2_de._pseudobulk's docstring too")
    assert list(genes) == ["g0", "g1"]

    # the pre-#271 answer on the same rows, so the fixture is known to discriminate
    narrow = np.asarray(X[np.array([0, 1])].sum(axis=0), dtype=np.float64).ravel()
    assert narrow[0] == 16777216.0


def test_auto_never_resolves_deseq2(monkeypatch):
    # force all backends "available" -> auto must still prefer gpudge/pdex/scanpy order
    monkeypatch.setattr(de_compute, "_available", lambda b: True)
    assert de_compute._resolve_backend("auto") in ("gpudge", "pdex", "scanpy")
    assert de_compute._resolve_backend("auto") != "deseq2"


def test_auto_excludes_deseq2_even_when_only_deseq2_available(monkeypatch):
    # stronger gate: if deseq2 were (wrongly) an auto fallback this would resolve to it. Make
    # ONLY deseq2 available -> auto must NOT fall back to it (raises, since no rank backend is).
    monkeypatch.setattr(de_compute, "_available", lambda b: b == "deseq2")
    with pytest.raises(RuntimeError):
        de_compute._resolve_backend("auto")


def test_resolve_explicit_deseq2(monkeypatch):
    monkeypatch.setattr(de_compute, "_available", lambda b: b == "deseq2")
    assert de_compute._resolve_backend("deseq2") == "deseq2"


def test_resolve_deseq2_unavailable_errors(monkeypatch):
    monkeypatch.setattr(de_compute, "_available", lambda b: False)
    with pytest.raises(RuntimeError, match="deseq2"):
        de_compute._resolve_backend("deseq2")


def test_validate_requires_counts(toy_de_adata):
    a = toy_de_adata()
    with pytest.raises(ValueError, match="counts"):
        deseq2_de._validate(a, pert_col="target_gene", control="non-targeting",
                            replicate_col="guide", input_type="lognorm")


def test_validate_requires_two_control_replicates(toy_de_adata):
    a = toy_de_adata(n_ctrl_guides=1)
    with pytest.raises(ValueError, match="control.*replicate"):
        deseq2_de._validate(a, pert_col="target_gene", control="non-targeting",
                            replicate_col="guide", input_type="counts")


def test_validate_missing_replicate_col(toy_de_adata):
    a = toy_de_adata()
    with pytest.raises(ValueError, match="replicate_col"):
        deseq2_de._validate(a, pert_col="target_gene", control="non-targeting",
                            replicate_col="nope", input_type="counts")


def test_pseudobulk_shapes_and_sums(toy_de_adata):
    a = toy_de_adata(n_ctrl_guides=3, n_pert=2, cells_per=5, n_genes=4)
    counts, cond, rep, genes = deseq2_de._pseudobulk(
        a, pert_col="target_gene", control="non-targeting", replicate_col="guide")
    assert counts.shape == (4, 5)          # 4 genes x (3 ctrl + 2 pert) samples
    assert cond.count("non-targeting") == 3 and len(set(rep)) == 5
    # column sum equals summing the matching cells
    for j, (c, r) in enumerate(zip(cond, rep)):
        mask = (a.obs["target_gene"] == c) & (a.obs["guide"] == r)
        np.testing.assert_allclose(counts[:, j], np.asarray(a.X[mask.values].sum(0)).ravel())


def test_contrasts_layout(toy_de_adata):
    pytest.importorskip("deseq2_gpu")  # _size_factors calls into deseq2_gpu -> skip in CI
    a = toy_de_adata(n_ctrl_guides=3, n_pert=2, cells_per=5, n_genes=4)
    counts, cond, rep, genes = deseq2_de._pseudobulk(
        a, pert_col="target_gene", control="non-targeting", replicate_col="guide")
    sf = deseq2_de._size_factors(counts)
    assert sf.shape == (5,) and np.all(sf > 0)
    contrasts = deseq2_de._contrasts(counts, cond, control="non-targeting", size_factors=sf)
    names = sorted(name for name, *_ in contrasts)
    assert names == ["GENE_0", "GENE_1"]
    for name, sub, design, sfsub in contrasts:
        assert sub.shape[0] == 4                      # genes
        assert design.shape == (sub.shape[1], 2)      # samples x [intercept, cond]
        assert set(np.unique(design[:, 0])) == {1.0}  # intercept
        assert design[:, 1].sum() == 1                # one perturbation sample (n=1)
        assert (design[:, 1] == 0).sum() == 3         # three control samples
        assert sfsub.shape[0] == sub.shape[1]


def test_run_deseq2_de_end_to_end_cpu(toy_de_adata):
    pytest.importorskip("deseq2_gpu")
    a = toy_de_adata(n_ctrl_guides=4, n_pert=3, cells_per=8, n_genes=6, seed=1)
    df = deseq2_de.run_deseq2_de(
        a, pert_col="target_gene", control="non-targeting",
        replicate_col="guide", input_type="counts", use_gpu=False)
    assert set(df.columns) >= {"target", "feature", "log2_fold_change", "p_value", "p_adj"}
    assert sorted(df["target"].unique().to_list()) == ["GENE_0", "GENE_1", "GENE_2"]
    assert df.height == 3 * 6                                   # perts x genes
    assert df["log2_fold_change"].is_finite().sum() > 0
    # p_adj in [0,1] where present
    pa = df["p_adj"].drop_nulls().drop_nans().to_numpy()
    assert ((pa >= 0) & (pa <= 1)).all()


def test_run_deseq2_de_matches_direct_fit(toy_de_adata):
    pytest.importorskip("deseq2_gpu")
    from deseq2_gpu import fit_nb_glm, results
    a = toy_de_adata(n_ctrl_guides=4, n_pert=1, cells_per=8, n_genes=6, seed=2)
    df = deseq2_de.run_deseq2_de(
        a, pert_col="target_gene", control="non-targeting",
        replicate_col="guide", input_type="counts", use_gpu=False)
    counts, cond, rep, genes = deseq2_de._pseudobulk(
        a, pert_col="target_gene", control="non-targeting", replicate_col="guide")
    sf = deseq2_de._size_factors(counts)
    (name, sub, design, sfsub), = deseq2_de._contrasts(counts, cond, control="non-targeting", size_factors=sf)
    fit = fit_nb_glm(sub, design, size_factors=sfsub, tool="deseq2", backend="numpy")
    ref = results(fit, np.array([0.0, 1.0]))
    sub_df = df.filter(pl.col("target") == name)
    # run_deseq2_de emits features in var_names order, and the direct-fit `ref` rows are in the
    # same order (results() synthesizes gene names row-wise for ndarray input). Assert the feature
    # order explicitly, then compare LFC ELEMENTWISE (no value-sorting) so a gene permutation is
    # actually detected -- sorting both sides would mask exactly the misalignment this test gates.
    assert sub_df["feature"].to_list() == list(a.var_names)
    got = sub_df["log2_fold_change"].to_numpy()
    exp = ref["log2FoldChange"].to_numpy()
    np.testing.assert_allclose(got, exp, rtol=1e-9, atol=1e-9)


def test_compute_de_deseq2_branch_end_to_end(toy_de_adata):
    pytest.importorskip("deseq2_gpu")
    from cell_eval2.de_compute import compute_de
    a = toy_de_adata(n_ctrl_guides=4, n_pert=2, cells_per=8, n_genes=5, seed=3)
    df = compute_de(
        a, backend="deseq2", groupby="target_gene", reference="non-targeting",
        mean_calc="arithmetic", epsilon=1e-9, input_type="counts",
        filter_gene_min_cpm_cell=None, replicate_col="guide")
    assert set(df.columns) >= {"target", "feature", "log2_fold_change", "p_adj"}
    assert sorted(df["target"].unique().to_list()) == ["GENE_0", "GENE_1"]


def test_compute_de_deseq2_requires_replicate_col(toy_de_adata):
    pytest.importorskip("deseq2_gpu")
    from cell_eval2.de_compute import compute_de
    a = toy_de_adata()
    with pytest.raises(ValueError, match="replicate_col"):
        compute_de(a, backend="deseq2", groupby="target_gene", reference="non-targeting",
                   mean_calc="arithmetic", epsilon=1e-9, input_type="counts",
                   filter_gene_min_cpm_cell=None, replicate_col=None)


def test_compute_de_deseq2_warns_on_fdr_scope(toy_de_adata, caplog):
    pytest.importorskip("deseq2_gpu")
    from cell_eval2.de_compute import compute_de
    a = toy_de_adata(n_ctrl_guides=4, n_pert=1, cells_per=6, n_genes=4, seed=4)
    with caplog.at_level("WARNING"):
        compute_de(a, backend="deseq2", groupby="target_gene", reference="non-targeting",
                   mean_calc="arithmetic", epsilon=1e-9, input_type="counts",
                   filter_gene_min_cpm_cell=None, replicate_col="guide", fdr_scope="global")
    assert any("fdr_scope" in r.message for r in caplog.records)


def test_de_cache_key_includes_replicate_col():
    from cell_eval2 import run as cev_run
    from cell_eval2.config import DEParams, EvalConfig
    base = EvalConfig(pert_col="target_gene", control="non-targeting", input_type="counts",
                      de=DEParams(backend="deseq2", replicate_col="guideA"))
    other = EvalConfig(pert_col="target_gene", control="non-targeting", input_type="counts",
                       de=DEParams(backend="deseq2", replicate_col="guideB"))
    d1 = cev_run._result_config_digest(
        base, de_backend_used=True, comparator="bulk_lognorm",
    )
    d2 = cev_run._result_config_digest(
        other, de_backend_used=True, comparator="bulk_lognorm",
    )
    assert d1 != d2


def test_replicate_col_does_not_split_cache_for_non_deseq2():
    # replicate_col is inert for non-deseq2 backends, so setting it must NOT change the result-cache
    # digest -- otherwise every non-deseq2 config that happens to set replicate_col loses cache hits.
    from cell_eval2 import run as cev_run
    from cell_eval2.config import DEParams, EvalConfig
    without = EvalConfig(de=DEParams(backend="pdex"))
    with_rep = EvalConfig(de=DEParams(backend="pdex", replicate_col="guide"))
    assert (cev_run._result_config_digest(
        without, de_backend_used=True, comparator="lognorm",
    ) == cev_run._result_config_digest(
        with_rep, de_backend_used=True, comparator="lognorm",
    ))
