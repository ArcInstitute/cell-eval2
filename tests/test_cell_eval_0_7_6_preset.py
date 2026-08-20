import pytest

from cell_eval2.config import EvalConfig


@pytest.mark.parametrize("name", ["cell-eval-0.7.6", "cell_eval_0_7_6"])
def test_preset_knobs(name):
    cfg = EvalConfig.from_preset(name)
    assert cfg.version == "v1"                       # -> upstream metric names
    assert cfg.control_source == "pred"
    assert cfg.input_type == "lognorm"
    assert cfg.target_sum == 1e4
    assert cfg.max_counts_per_cell == 1e9
    assert cfg.filter.filter_gene_min_cpm_cell == 5.0
    assert cfg.de.mean_calc == "geometric"
    assert cfg.de.epsilon == 1e-9
    assert cfg.de.clip_value is None
    assert cfg.de.fdr_scope == "per_pert"
    assert cfg.de.nan_lfc_policy == "keep"
    assert cfg.de.auc_pval_floor == "clip"
    assert cfg.de.auc_pval_floor_value == 1e-10
    assert cfg.de.sort_by == "abs_log2_fold_change"
    assert cfg.discrimination.rank_denominator == "n"
    assert cfg.discrimination.exclude_target_gene is True


def test_preset_yaml_roundtrip(tmp_path):
    cfg = EvalConfig.from_preset("cell-eval-0.7.6")
    p = tmp_path / "c.yaml"
    cfg.to_yaml(str(p))
    assert EvalConfig.from_yaml(str(p)) == cfg


def _synth_pair(n_perts=4, n_cells=40, n_genes=15, seed=0):
    import anndata as ad
    import numpy as np
    import pandas as pd
    labels = ["non-targeting"] + [f"P{i}" for i in range(n_perts)]
    obs_labels = np.repeat(labels, n_cells)

    def mk(s):
        r = np.random.default_rng(s)
        x = r.poisson(2.0, size=(len(obs_labels), n_genes)).astype("float32")
        lib = x.sum(1, keepdims=True)
        lib[lib == 0] = 1.0
        x = np.log1p(x * (1e4 / lib))       # lognorm input (CPM-1e4 + log1p)
        # The perturbations P0..P{n-1} are gene knockdowns, so their own genes are
        # MEASURED -- the leading var labels are the perturbation labels. Since #248
        # exclude_target_gene=True (the preset's default) raises on a panel where no
        # perturbation resolves to a gene rather than scoring with nothing excluded.
        var_names = [f"P{i}" for i in range(n_perts)] + \
                    [f"g{j}" for j in range(n_genes - n_perts)]
        return ad.AnnData(x, obs=pd.DataFrame({"target": obs_labels}),
                          var=pd.DataFrame(index=var_names))

    return mk(seed), mk(seed + 1)


def test_preset_emits_v1_metric_names():
    from cell_eval2.run import compute_metrics
    pred, real = _synth_pair()
    cfg = EvalConfig.from_preset("cell-eval-0.7.6")  # backend=auto -> pdex on CPU
    out = compute_metrics(pred, real, config=cfg)
    names = set(out["metric"].to_list())
    for m in ("overlap_at_N", "pr_auc", "roc_auc", "pearson_delta", "mae",
              "discrimination_score_l1", "de_spearman_sig", "de_nsig_counts_real"):
        assert m in names, f"missing upstream metric name {m!r} (got {sorted(names)[:8]}...)"
    # metrics that HAVE a v1 name must emit the v1 name, not the v2 canonical name.
    # (cev2-only metrics with no v1 equivalent legitimately keep their canonical name.)
    for v1name, v2name in [("overlap_at_N", "de_wilcoxon_overlap"),
                           ("mae", "expr_mae"),
                           ("discrimination_score_l1", "pds_l1")]:
        assert v1name in names and v2name not in names, \
            f"{v1name!r} should replace {v2name!r} under version='v1'"


def _no_gpu():
    try:
        import cupy
        return cupy.cuda.runtime.getDeviceCount() == 0
    except Exception:
        return True


@pytest.mark.skipif(_no_gpu(), reason="needs a CUDA GPU (gpudge DE)")
def test_gpudge_cpm_filter_applies_on_lognorm():
    import anndata as ad
    import numpy as np
    import pandas as pd
    from cell_eval2.de_compute import compute_de
    r = np.random.default_rng(1)
    labels = np.repeat(["non-targeting", "P0", "P1", "P2"], 200)
    n_genes = 40
    counts = r.poisson(2.0, size=(labels.size, n_genes)).astype("float32")
    counts[:, 20:] = 0.0                       # 20 never-expressed genes -> < 5 CPM -> dropped
    lib = counts.sum(1, keepdims=True)
    lib[lib == 0] = 1.0
    x = np.log1p(counts * (1e4 / lib))         # lognorm input
    real = ad.AnnData(x, obs=pd.DataFrame({"target": labels}),
                      var=pd.DataFrame(index=[f"g{j}" for j in range(n_genes)]))
    kw = dict(groupby="target", reference="non-targeting", mean_calc="geometric",
              epsilon=1e-9, input_type="lognorm", clip_value=None, fdr_scope="per_pert")
    unfiltered = compute_de(real, backend="gpudge", filter_gene_min_cpm_cell=None, **kw)
    filtered = compute_de(real, backend="gpudge", filter_gene_min_cpm_cell=5.0, **kw)
    assert filtered.height < unfiltered.height       # the fix: filter drops low-CPM genes on lognorm
    j = filtered.join(unfiltered, on=["target", "feature"], suffix="_u", how="inner")
    assert np.allclose(j["log2_fold_change"].to_numpy(),
                       j["log2_fold_change_u"].to_numpy(), rtol=1e-5, atol=1e-6)
