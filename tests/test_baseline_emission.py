from dataclasses import replace

import anndata as ad
import numpy as np
import pandas as pd
import polars as pl
import pytest
import scipy.sparse as sp

import cell_eval2.norm as norm
from cell_eval2 import EvalConfig
from cell_eval2.baseline import (
    GenericProfile,
    _emission_scale,
    _emit_scaled_resample,
    build_baseline_prediction,
    build_generic_baseline,
    generic_response_profile,
)
from cell_eval2.config import DEParams
from cell_eval2.moments import trace_sigma
from cell_eval2.prep import pseudobulk, pseudobulk_with_moments


PERTS = ("non-targeting", "GENE1", "GENE2", "GENE3")


def _adata(X, labels, *, genes=None, sparse=False):
    X = np.asarray(X, dtype=np.float32)
    if sparse:
        X = sp.csr_matrix(X)
    obs = pd.DataFrame(
        {"target": labels}, index=[f"c{i}" for i in range(len(labels))]
    )
    if genes is None:
        genes = [f"g{i}" for i in range(X.shape[1])]
    return ad.AnnData(X=X, obs=obs, var=pd.DataFrame(index=genes))


def _counts_cfg(metrics, **kwargs):
    base = dict(
        metrics=metrics,
        pert_col="target",
        control="non-targeting",
        input_type="counts",
        version="v2",
        allow_fractional_counts=True,
        allow_discrete=True,
        device="cpu",
    )
    base.update(kwargs)
    return EvalConfig(**base)


def _overdispersed_counts(
    seed=0, n_genes=120, n_cells_per=30, shape=0.08, base=4.0, lib_sigma=1.0
):
    """The ~86% zeros make log1p's concavity term large; unequal totals keep the CPM
    weighting term live. Measured totals are 1--481 and tile/dispersed expr_mse is 40.6x
    at seed 0; a dense over-dispersed fixture measured only 3.5x."""
    rng = np.random.default_rng(seed)
    blocks, labels = [], []
    for k, pert in enumerate(PERTS):
        libs = rng.lognormal(mean=0.0, sigma=lib_sigma, size=n_cells_per)[:, None]
        mu = rng.gamma(
            shape=shape,
            scale=base * (1.0 + 0.35 * k),
            size=(n_cells_per, n_genes),
        )
        blocks.append(rng.poisson(mu * libs).astype(np.float32))
        labels += [pert] * n_cells_per
    # Every non-control target names a MEASURED gene, as a real panel does. Since issue #172 the
    # `expr_mse_unbiased*` / `expr_distance_unbiased` family drops each perturbation's own column
    # and RAISES when no target resolves, so the suite's usual "targets GENE1.., genes g0.."
    # shape fails for the construct-ID reason. The genes are i.i.d. here, so WHICH indices carry
    # the names is arbitrary; the tail is used so no index-based expectation moves.
    genes = [f"g{i}" for i in range(n_genes)]
    named = [p for p in PERTS if p != "non-targeting"]
    for j, pert in enumerate(named, start=1):
        genes[n_genes - j] = pert
    return _adata(np.vstack(blocks), labels, genes=genes)


def _mean_metric(result, name):
    return float(result.results.filter(pl.col("metric") == name)["value"].mean())


def _heterogeneous_counts(*, sparse=False, unequal=False, constant_noncontrol=False):
    """Small positive-support counts panel whose control donors have distinct supports."""
    rng = np.random.default_rng(12)
    sizes = {"GENE1": 17, "GENE2": 29, "GENE3": 11} if unequal else {
        "GENE1": 24,
        "GENE2": 24,
        "GENE3": 24,
    }
    controls = np.array(
        [[8, 0, 3, 1, 6], [0, 9, 2, 7, 1], [4, 2, 10, 0, 3], [1, 7, 0, 8, 5]],
        dtype=np.float32,
    )
    blocks = []
    labels = []
    for pert, size in sizes.items():
        if constant_noncontrol:
            block = np.tile(np.array([4, 5, 6, 3, 7], dtype=np.float32), (size, 1))
        else:
            block = rng.poisson([4, 5, 6, 3, 7], size=(size, 5)).astype(np.float32)
        blocks.append(block)
        labels += [pert] * size
    blocks.append(controls)
    labels += ["non-targeting"] * len(controls)
    # Every target names a MEASURED gene, as a real panel does. Since issue #172 the
    # `expr_mse_unbiased*` / `expr_distance_unbiased` family drops each perturbation's own
    # column and RAISES when no target resolves against the gene panel, so the suite's usual
    # "targets GENE1.., genes g0.." shape fails here for the construct-ID reason.
    return _adata(np.vstack(blocks), labels, sparse=sparse,
                  genes=["GENE1", "GENE2", "GENE3", "g3", "g4"])


def _unreachable_reference():
    """Counts panel with positive perturbation mass in ``unreachable`` and zero controls."""
    rng = np.random.default_rng(41)
    blocks, labels = [], []
    for pert in PERTS:
        block = rng.poisson(6, size=(18, 12)).astype(np.float32) + 1
        # Keep the unreachable gene below the default 5 CPM in every real cell while
        # retaining positive mass: 1 / ~300,000 * 1e6 is about 3.3 CPM.
        block[:, 0] = 300_000
        if pert == "non-targeting":
            block[:, -1] = 0
        else:
            block[:, -1] = 1
        blocks.append(block)
        labels += [pert] * len(block)
    genes = [f"g{i}" for i in range(11)] + ["unreachable"]
    return _adata(np.vstack(blocks), labels, genes=genes, sparse=True)


def test_the_comparator_move_INVERTS_which_emission_model_expr_mse_prefers():
    """⚠️ THE DIRECTION OF THIS ASSERTION FLIPPED IN #264 PR2. That is the headline result of
    the comparator move, not an incidental fixture update, and it is why this test was renamed
    rather than re-tuned.

    MEASURED on this frozen 85.6%-zero over-dispersed fixture at seed 0:

        comparator      TS     tile_mse  dispersed_mse  ratio          preferred arm
        lognorm         --      ~40.6x      1x         tile/disp 40.6   DISPERSED
        bulk_lognorm    1e6     1.0348     12.3205     disp/tile 11.9   TILE
        bulk_lognorm    5e4     0.8010      5.9414     disp/tile  7.4   TILE

    ⚠️ **#268's 1e6 -> 5e4 move PARTIALLY MITIGATES the inversion** -- 11.9x down to 7.4x on
    this fixture. It does not remove it: tile is still the preferred arm, which is the accepted
    outcome below. Noted because the acceptance decision was taken against the 11.9x figure.

    WHY it inverts, and why it is intended. Under `lognorm` the comparator is a per-cell mean
    of `log1p(CPM)`, which is a DISPERSION FUNCTIONAL rather than a mean (#260): Jensen's
    inequality means no zero-dispersion prediction can reproduce a real bulk at all, so a tiled
    arm is penalized for a property the metric does not claim to measure. #258 is the same
    finding from the other side -- `emit='tile'` cannot represent a lognorm bulk as a matter of
    RANGE, not scale, and the real control itself scored 13x when tiled. Under `bulk_lognorm`
    the bulk is `log1p(TS * P_g / sum_g P_g)` computed from the GROUP SUM, which a tiled arm
    reproduces exactly, so the pathology is gone. Removing it is the point of #264.

    WHAT IT COSTS, stated so nobody has to rediscover it: a degenerate zero-dispersion
    submission is no longer penalized by `expr_mse`. That penalty was never this metric's job
    -- it was an artifact -- but it WAS load-bearing in practice (#234, #259), so the gameability
    it used to suppress is no longer suppressed here. DECIDED (Alex, 2026-08-11): accepted, no
    replacement guard. `expr_mse` never claimed to measure cell-to-cell variation, and the
    metric family that would was measured at a +2.0% ceiling against the mean's +14.5% and
    rejected. Do not re-add a dispersion penalty to THIS metric to recover it.

    The old name (`test_dispersed_fixes_the_order_of_operations_regression`) asserted that
    DISPERSED was the fix. Under the shipped comparator it is not, so keeping that name with an
    inverted body would have been actively misleading."""
    real = _overdispersed_counts()
    cfg = _counts_cfg(["expr_mse", "expr_mse_unbiased_capped"])
    tile = build_generic_baseline(
        real, config=cfg, exclude_target_gene=False, emit="tile"
    )
    dispersed = build_generic_baseline(
        real, config=cfg, exclude_target_gene=False, seed=0
    )
    tile_mse = _mean_metric(tile, "expr_mse")
    dispersed_mse = _mean_metric(dispersed, "expr_mse")

    # Direction deliberately reversed vs pre-#264 -- see the docstring. Kept as a SEPARATION
    # band rather than a pinned ratio so it fails on a collapse in either direction, not on
    # ordinary drift. Loosened 10x -> 5x by #268: the band tracks the shipped
    # `bulk_target_sum`, which moved the measured ratio from 11.9x to 7.4x. 5x still fails on
    # a collapse; it is not tuned to just-pass 7.4x.
    assert dispersed_mse > 5.0 * tile_mse
    assert np.mean(np.asarray(real.X) == 0) > 0.8
    totals = np.asarray(real.X.sum(axis=1)).ravel()
    assert totals.max() / totals.min() > 100


def test_builder_emits_nonzero_predicted_covariance(tmp_path):
    """Assert through build_generic_baseline + saved artifacts: calling the tiled helper
    inside the builder would leave tr(Sigma-hat_pred) exactly zero despite a new constructor."""
    # Integer profile rows make the tile's algebraic zero bit-exact in the fp64 moments
    # reduction, while heterogeneous controls still give dispersed cells positive variance.
    real = _heterogeneous_counts(sparse=True, constant_noncontrol=True)
    cfg = _counts_cfg(["expr_mse", "expr_mse_unbiased_capped"])
    paths = {emit: tmp_path / f"{emit}.h5ad" for emit in ("tile", "dispersed")}
    for emit, path in paths.items():
        build_generic_baseline(
            real,
            config=cfg,
            exclude_target_gene=False,
            emit=emit,
            seed=3,
            save_pred=path,
            allow_degenerate=True,
        )

    traces = {}
    for emit, path in paths.items():
        pred = ad.read_h5ad(path)
        perts, means, moments = pseudobulk_with_moments(pred, "target")
        values = trace_sigma(moments.counts, moments.sumsq, means)
        traces[emit] = values[perts != "non-targeting"]

    assert np.all(traces["dispersed"] > 0)
    np.testing.assert_array_equal(traces["tile"], 0.0)


def test_seed_reproducibility_and_sampling_mean():
    """Seeds select iid donor draws: equal seeds are bit-identical and unequal seeds are
    not. The 5% mean bound is deliberately Monte Carlo, not a float-precision claim; each
    group has 10,000 draws, so it is loose relative to the measured 1/sqrt(n) behavior."""
    controls = np.array([[2, 12, 4, 8], [10, 2, 8, 4], [6, 6, 2, 12]], dtype=np.float32)
    profile_rows = np.tile(np.array([7, 5, 9, 3], dtype=np.float32), (20_000, 1))
    labels = ["GENE1"] * 10_000 + ["GENE2"] * 10_000 + ["non-targeting"] * 3
    real = _adata(np.vstack([profile_rows, controls]), labels)
    profile = generic_response_profile(
        real, pert_col="target", control="non-targeting", exclude_target_gene=False
    )
    a = build_baseline_prediction(
        profile, real, pert_col="target", control="non-targeting", seed=91
    )
    b = build_baseline_prediction(
        profile, real, pert_col="target", control="non-targeting", seed=91
    )
    c = build_baseline_prediction(
        profile, real, pert_col="target", control="non-targeting", seed=92
    )

    np.testing.assert_array_equal(a.X, b.X)
    assert not np.array_equal(a.X, c.X)
    for pred in (a, c):
        perts, means = pseudobulk(pred, "target")
        got = means[perts != "non-targeting"]
        expected = np.tile(profile.values, (got.shape[0], 1))
        np.testing.assert_allclose(got, expected, rtol=0.05)


def test_control_zero_gene_is_reported_and_default_gate_excludes_it(monkeypatch, tmp_path):
    """A control-zero gene is unreachable, not support removed: every donor was already
    zero. The v2 5-CPM gate removes it from both computed DE tables, while diagnostics keep
    the measured lost profile mass visible in the prediction and builder stamp."""
    import cell_eval2.run as run

    real = _unreachable_reference()
    profile = generic_response_profile(
        real, pert_col="target", control="non-targeting", exclude_target_gene=False
    )
    pred = build_baseline_prediction(
        profile, real, pert_col="target", control="non-targeting", seed=0
    )
    non_ctrl = np.asarray(pred.obs["target"]) != "non-targeting"
    assert pred[:, "unreachable"].X[non_ctrl].nnz == 0
    diag = pred.uns["baseline_emission"]
    assert diag["genes_control_zero"] == ["unreachable"]
    assert diag["profile_mass_unreachable"] > 0

    seen = {}
    original = run._compute_de_side

    def capture(*args, **kwargs):
        table = original(*args, **kwargs)
        seen[kwargs["side"]] = table
        return table

    monkeypatch.setattr(run, "_compute_de_side", capture)
    out = tmp_path / "pred.h5ad"
    cfg = _counts_cfg(
        ["expr_mse", "de_wilcoxon_overlap"],
        de=DEParams(backend="scanpy"),
    )
    result = build_generic_baseline(
        real,
        config=cfg,
        exclude_target_gene=False,
        seed=0,
        save_pred=out,
        allow_degenerate=True,
    )
    assert set(seen) == {"real", "pred"}
    for table in seen.values():
        assert "unreachable" not in table["feature"].to_list()
    stamp = result.meta["baseline_emission"]
    assert stamp["genes_control_zero"] == ["unreachable"]
    assert stamp["profile_mass_unreachable"] == pytest.approx(
        diag["profile_mass_unreachable"]
    )


def test_stamp_lists_all_unreachable_genes_beyond_the_old_100_name_cap():
    """The gene axis bounds the JSON payload, while truncation defeats the stamp's purpose.
    Use 137 unreachable genes so a reintroduced 100-name cap cannot pass silently."""
    unreachable = [f"unreachable_{i:03d}" for i in range(137)]
    genes = unreachable + ["reachable"]
    perturbed = np.ones((8, len(genes)), dtype=np.float32)
    controls = np.zeros((4, len(genes)), dtype=np.float32)
    controls[:, -1] = 2.0
    real = _adata(
        np.vstack([perturbed, controls]),
        ["GENE1"] * 4 + ["GENE2"] * 4 + ["non-targeting"] * 4,
        genes=genes,
        sparse=True,
    )

    result = build_generic_baseline(
        real,
        config=_counts_cfg(["expr_mse"]),
        exclude_target_gene=False,
        seed=0,
        allow_degenerate=True,
    )
    stamp = result.meta["baseline_emission"]
    assert stamp["n_genes_control_zero"] == len(unreachable)
    assert stamp["genes_control_zero"] == unreachable
    assert "genes_control_zero_limit" not in stamp


def test_unreachable_profile_fraction_is_l1_mass_for_mixed_signs():
    """A general reference may have mixed signs, so signed sums can report negative lost
    mass. The unreachable fraction is abs(unreachable) / abs(all), here 3 / 10."""
    _, diagnostics = _emission_scale(
        np.array([-3.0, 2.0, 5.0]),
        np.array([0.0, 1.0, 1.0]),
        np.array(["unreachable", "g1", "g2"]),
    )

    assert diagnostics["profile_mass_unreachable"] == pytest.approx(0.3)


def test_profile_zero_gene_removes_real_donor_support():
    """This is the distinct r=0 case: profile==0<ctrl actively deletes donor entries;
    the control-zero case above cannot demonstrate support removal."""
    real = _adata(
        [[3, 8, 1], [4, 5, 2], [2, 7, 6], [1, 9, 3], [5, 6, 4]],
        ["GENE1", "GENE1", "GENE2", "non-targeting", "non-targeting"],
        sparse=True,
    )
    profile = GenericProfile(
        values=np.array([3.0, 0.0, 2.0]),
        genes=np.asarray(real.var.index).astype(str),
        n_perturbations=2,
        exclude_target_gene=False,
        n_excluded=0,
    )
    pred = build_baseline_prediction(
        profile, real, pert_col="target", control="non-targeting", seed=2
    )
    non_ctrl = np.asarray(pred.obs["target"]) != "non-targeting"
    assert np.all(real[~non_ctrl, 1].X.toarray() > 0)
    assert pred[non_ctrl, 1].X.nnz == 0
    assert pred.uns["baseline_emission"]["n_genes_profile_zero_control_positive"] == 1
    assert pred.uns["baseline_emission"]["n_explicit_zeros_removed"] > 0


def test_default_scale_gate_rejects_rare_gene_high_r():
    """With M=2 controls, one count a=1 and profile b=600,000, r=bM/a=1.2e6.
    Seed 0 draws the rare donor, so the emitted 1.2m row must hit the default 1m gate and
    the wrapped error must preserve both the gate text and construction diagnostics."""
    X = np.array(
        [
            [600_000, 1],
            [600_000, 2],
            [600_000, 1],
            [600_000, 2],
            [0, 1],
            [1, 1],
        ],
        dtype=np.float32,
    )
    labels = ["GENE1", "GENE1", "GENE2", "GENE2", "non-targeting", "non-targeting"]
    real = _adata(X, labels)
    cfg = _counts_cfg(["expr_mse"], allow_discrete=False)
    with pytest.raises(ValueError) as caught:
        build_generic_baseline(
            real, config=cfg, exclude_target_gene=False, seed=0
        )
    assert isinstance(caught.value, norm.ScaleLimitError)
    message = str(caught.value)
    assert "per-cell total 1200001.5" in message
    assert "exceeds max_counts_per_cell=1000000" in message
    assert "r_max=1200000" in message
    assert "max_scaled_noncontrol_row_total=1200001.5" in message


def test_unrelated_compute_metrics_value_error_subclass_propagates_untouched(monkeypatch):
    """Only the typed scale gate merits emission diagnostics; every other metric failure
    must retain its exact subclass, instance and message instead of being relabelled."""
    class SentinelMetricsError(ValueError):
        pass

    sentinel = SentinelMetricsError("sentinel metric failure")

    def fail_metrics(*args, **kwargs):
        raise sentinel

    monkeypatch.setattr("cell_eval2.baseline.compute_metrics", fail_metrics)
    real = _heterogeneous_counts()
    with pytest.raises(SentinelMetricsError) as caught:
        build_generic_baseline(
            real,
            config=_counts_cfg(["expr_mse"]),
            exclude_target_gene=False,
        )

    assert caught.value is sentinel
    assert str(caught.value) == "sentinel metric failure"
    assert "Dispersed-emission diagnostics" not in str(caught.value)


def test_a_REAL_side_scale_rejection_is_not_labelled_a_dispersed_emission_failure(monkeypatch):
    """`compute_metrics` gates the REAL side too (run.py:851, ``side == "real"``), so the
    type alone does not identify the side. Narrowing the checkpoint-2 catch to
    ScaleLimitError was not enough: a reference whose own cells break the cap would still
    have been annotated with this construction's diagnostics, pointing at the wrong matrix.
    Monkeypatched rather than fixture-driven on purpose -- a reference large enough to trip
    the real gate necessarily produces a profile whose emitted rows trip the pred gate too,
    so no fixture can isolate this branch."""
    real_side = norm.ScaleLimitError(
        "per-cell total 9999999.0 exceeds max_counts_per_cell=1000000.0"
    )

    def fail_metrics(*args, **kwargs):
        raise real_side

    monkeypatch.setattr("cell_eval2.baseline.compute_metrics", fail_metrics)
    real = _heterogeneous_counts()
    with pytest.raises(norm.ScaleLimitError) as caught:
        build_generic_baseline(
            real, config=_counts_cfg(["expr_mse"]), exclude_target_gene=False, seed=0
        )

    # discriminating: this fixture's own prediction is comfortably INSIDE the cap, which is
    # what makes the real side the only possible source -- assert that premise, or the test
    # would also pass on a build that simply never annotates anything
    pred_max = build_baseline_prediction(
        generic_response_profile(real, pert_col="target", control="non-targeting",
                                 exclude_target_gene=False),
        real, pert_col="target", control="non-targeting", seed=0,
    ).uns["baseline_emission"]["max_row_total_full_prediction"]
    assert pred_max <= _counts_cfg(["expr_mse"]).max_counts_per_cell
    assert caught.value is real_side
    assert "Dispersed-emission diagnostics" not in str(caught.value)


def test_kernel_support_is_the_scaled_donor_subset_without_stored_zeros():
    """This is the load-bearing kernel VALUE assertion: support cannot detect a wrong
    multiplier, wrapper parity repeats a shared-kernel bug, and sampling means tolerate 5%.
    Build donor[source] * scale in float32 and require exact equality after zero removal."""
    donors = sp.csr_matrix(
        (
            np.array([2, 4, 0, 8, 3, 5, 1, 7, 2, 6], dtype=np.float32),
            np.array([0, 1, 2, 3, 1, 2, 3, 0, 2, 3]),
            np.array([0, 4, 7, 10]),
        ),
        shape=(3, 4),
    )
    assert donors.nnz == 10  # includes the explicit zero: the old nnz-only test counted it
    scale = np.array([2.0, 0.0, 0.5, 3.0])
    out, source, _ = _emit_scaled_resample(
        donors, [17, 9], scale, np.random.default_rng(7)
    )
    clean = donors.copy()
    clean.eliminate_zeros()
    selected = clean[source]
    expected = donors[source].copy()
    expected.data = (expected.data * scale[expected.indices]).astype(np.float32)
    expected.eliminate_zeros()

    assert (out != expected).nnz == 0
    assert out.nnz <= selected.nnz
    assert np.all(out.data != 0)
    for row, donor_row in zip(out, selected):
        got = set(row.indices.tolist())
        allowed = {j for j in donor_row.indices.tolist() if scale[j] > 0}
        assert got <= allowed
        assert got == allowed  # finite nonzero scales here: no underflow


def test_n_cells_mirrors_unequal_template_groups():
    """The library contract mirrors the template rather than adopting the tool's flat-N
    sensitivity mode; unequal 17/29/11 groups make a fixed-size implementation fail."""
    real = _heterogeneous_counts(unequal=True)
    profile = generic_response_profile(
        real, pert_col="target", control="non-targeting", exclude_target_gene=False
    )
    pred = build_baseline_prediction(
        profile, real, pert_col="target", control="non-targeting", seed=4
    )
    assert pred.obs["target"].value_counts().sort_index().to_dict() == {
        "GENE1": 17,
        "GENE2": 29,
        "GENE3": 11,
        "non-targeting": 4,
    }


def test_library_and_tool_wrapper_have_kernel_parity(tmp_path):
    """The wrappers order rows differently by design. This template is deliberately
    group-major (sorted perturbations, then controls), matching the tool's explicit order,
    so an exact positional X comparison isolates their shared kernel."""
    # `tools/baselineval/` is internal and does not travel to the public cut, so the one test in
    # this module that reaches for it is guarded: it runs here and skips where the tool is
    # absent, instead of taking a 17-test module down with it. Same pattern, and the same
    # explicit `exc_type`, as tests/test_score.py::test_compare_vcc_select_scorer -- see the
    # reasoning there.
    make_baselines = pytest.importorskip(
        "baselineval.make_baselines", exc_type=ModuleNotFoundError).main

    real = _heterogeneous_counts(sparse=True, unequal=True)
    order = np.argsort(
        pd.Categorical(
            real.obs["target"],
            categories=["GENE1", "GENE2", "GENE3", "non-targeting"],
            ordered=True,
        ).codes,
        kind="stable",
    )
    real = real[order].copy()
    source = tmp_path / "real.h5ad"
    tool_path = tmp_path / "tool.h5ad"
    real.write_h5ad(source)
    profile = generic_response_profile(
        real, pert_col="target", control="non-targeting", exclude_target_gene=False
    )
    library = build_baseline_prediction(
        profile, real, pert_col="target", control="non-targeting", seed=19
    )
    make_baselines(
        [
            "--data",
            str(source),
            "--kind",
            "dispersed",
            "--out",
            str(tool_path),
            "--emit",
            "h5ad",
            "--n-cells-from-input",
            "--pert-col",
            "target",
            "--control",
            "non-targeting",
            "--seed",
            "19",
            "--no-exclude-target-gene",
            "--with-controls",
        ]
    )
    tool = ad.read_h5ad(tool_path)

    assert library.obs["target"].astype(str).tolist() == tool.obs["target"].astype(str).tolist()
    np.testing.assert_array_equal(library.X.toarray(), tool.X.toarray())


def test_numeric_and_none_target_sum_paths_both_score():
    """The v1-over-counts fallback keeps target_sum live after v2 moves to bulk_lognorm."""
    real = _heterogeneous_counts()
    cfg = _counts_cfg(["expr_mse"], version="v1")
    numeric = build_generic_baseline(
        real,
        config=replace(cfg, target_sum=1234.0),
        exclude_target_gene=False,
        seed=5,
    )
    resolved = build_generic_baseline(
        real,
        config=replace(cfg, target_sum=None),
        exclude_target_gene=False,
        seed=5,
    )
    a = numeric.results["value"].to_numpy()
    b = resolved.results["value"].to_numpy()
    assert np.all(np.isfinite(a)) and np.all(np.isfinite(b))
    assert not np.allclose(a, b)


def test_all_zero_profile_has_exact_zero_scale_and_empty_emitted_block():
    """State exact zero-denominator values: no 0/0 or empty-median NaN. The prediction is
    not empty because copied control rows remain; only the non-control emitted CSR block is."""
    real = _heterogeneous_counts(sparse=True)
    profile = GenericProfile(
        values=np.zeros(real.n_vars),
        genes=np.asarray(real.var.index).astype(str),
        n_perturbations=3,
        exclude_target_gene=True,
        n_excluded=3,
    )
    ctrl = np.asarray(real.obs["target"]) == "non-targeting"
    ctrl_pb = np.asarray(real[ctrl].X.mean(axis=0)).ravel()
    scale, scale_diag = _emission_scale(profile.values, ctrl_pb, profile.genes)
    pred = build_baseline_prediction(
        profile, real, pert_col="target", control="non-targeting", seed=0
    )

    np.testing.assert_array_equal(scale, 0.0)
    assert scale_diag["r_max"] == 0.0
    assert scale_diag["r_median_nonzero"] == 0.0
    assert scale_diag["profile_mass_unreachable"] == 0.0
    assert pred[~ctrl].X.nnz == 0
    assert pred[ctrl].X.nnz > 0
    assert pred.uns["baseline_emission"]["r_max"] == 0.0
    assert pred.uns["baseline_emission"]["r_median_nonzero"] == 0.0


@pytest.mark.parametrize(
    "cfg",
    [
        EvalConfig(
            metrics=["expr_mse"],
            pert_col="target",
            control="non-targeting",
            version="v2",
            input_type="lognorm",
            validate_input=False,
        ),
        EvalConfig(
            metrics=["expr_mse"],
            pert_col="target",
            control="non-targeting",
            version="v1",
            input_type="counts",
            validate_input=False,
        ),
    ],
    ids=["v2-declared-lognorm", "v1-autodetected-fractional"],
)
def test_lognorm_effective_reference_rejects_dispersed_but_tile_works(cfg):
    """The guard keys on effective type, not just the declared string: cover v2's direct
    lognorm route and v1's fractional-row-total autodetection on the same reference."""
    rng = np.random.default_rng(8)
    real = _adata(
        np.log1p(rng.gamma(1.2, 2.0, size=(48, 9))),
        [pert for pert in PERTS for _ in range(12)],
    )
    with pytest.raises(ValueError, match='emit="tile"'):
        build_generic_baseline(real, config=cfg, exclude_target_gene=False)
    result = build_generic_baseline(
        real,
        config=cfg,
        exclude_target_gene=False,
        emit="tile",
        allow_degenerate=True,
    )
    assert result.meta["emit"] == "tile"


def test_unreachable_genes_stay_finite_gate_off_and_with_supplied_de_real():
    """The v2 computed-DE exclusion is narrow: v1 disables the CPM gate, supplied de_real
    bypasses its real-side computation, and expr/delta/PDS read pseudobulks directly. Assert
    finite values across those families plus DE; this is not an expression-metrics-only claim."""
    real = _unreachable_reference()
    metrics = ["expr_mse", "delta_mse", "pds_cosine", "de_wilcoxon_overlap"]
    v1_base = EvalConfig.for_version("v1")
    v1 = replace(
        v1_base,
        metrics=metrics,
        pert_col="target",
        control="non-targeting",
        input_type="counts",
        allow_fractional_counts=True,
        allow_discrete=True,
        device="cpu",
        de=replace(v1_base.de, backend="scanpy"),
        # This fixture's gene identities are load-bearing (g0 carries 300k counts, the
        # last gene is `unreachable`), so its panel deliberately shares no label with
        # PERTS and pds_cosine can exclude nothing. Stated explicitly, matching the
        # exclude_target_gene=False already passed to the builder below: since #248 an
        # unstated zero-resolve exclusion raises instead of silently scoring. This test
        # is about finiteness with the CPM gate off.
        discrimination=replace(v1_base.discrimination, exclude_target_gene=False),
    )
    gate_off = build_generic_baseline(
        real,
        config=v1,
        exclude_target_gene=False,
        seed=11,
        allow_degenerate=True,
    )
    assert v1.filter.filter_gene_min_cpm_cell is None
    assert np.all(np.isfinite(gate_off.results["value"].to_numpy()))

    rows = []
    for pert in PERTS[1:]:
        for feature in np.asarray(real.var.index).astype(str):
            rows.append(
                {
                    "target": pert,
                    "feature": feature,
                    "log2_fold_change": 1.0 if feature == "unreachable" else 0.2,
                    "p_adj": 0.001 if feature == "unreachable" else 0.5,
                }
            )
    supplied = pl.DataFrame(rows)
    assert "unreachable" in supplied["feature"].to_list()
    v2 = _counts_cfg(metrics, de=DEParams(backend="scanpy"))
    # same reason as the v1 arm above: this fixture's panel resolves no perturbation,
    # so the exclusion must be declined explicitly rather than silently no-op (#248)
    v2 = replace(v2, discrimination=replace(v2.discrimination,
                                            exclude_target_gene=False))
    bypass = build_generic_baseline(
        real,
        config=v2,
        exclude_target_gene=False,
        seed=11,
        de_real=supplied,
        allow_degenerate=True,
    )
    assert bypass.meta["de_real_supplied"] is True
    assert np.all(np.isfinite(bypass.results["value"].to_numpy()))
