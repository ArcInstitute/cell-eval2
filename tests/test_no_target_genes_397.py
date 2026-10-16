"""Issue #397: a panel whose perturbations have no target gene -- drug doses, added cytokines.

The declaration is a `None` value in `EvalConfig.target_gene_map`, and `{"*": None}` covers every
label without an entry of its own (the CLI's `--no-target-genes`). A declared label counts as
resolved, excludes nothing, and is never matched against the features by name -- at all five
places that exclude a target gene: `pds_*` (both scopes, CPU and GPU), the moments metrics, the
DE metrics, the `de_lfc_nmae` reference and the generic-response baseline.

Two fixtures carry most of the weight, and they are the same matrices under two label sets:

* the CYTOKINE panel labels its perturbations GENE1..GENE5, which ARE measured genes, so name
  matching would exclude them -- the 9-of-90 parse_pbmc case;
* the DRUG panel renames them DRUG1..DRUG5, which match nothing, so nothing can be excluded
  whatever the declaration means.

Declared, the two must score identically; that is the "never matched by name" property, and the
undeclared cytokine panel is what shows the comparison can fail.
"""
import logging
from dataclasses import replace

import anndata as ad
import numpy as np
import pandas as pd
import polars as pl
import pytest
import yaml

from _helpers import counts_bundle_inputs, full_minus_moments
from cell_eval2.config import (NO_TARGET_GENE_WILDCARD, UNMAPPED, EvalConfig,
                               target_gene_entry)
from cell_eval2.de import (TargetResolution, exclude_on_target, require_resolution,
                           resolve_target_genes)
from cell_eval2.distances import resolve_exclusion_columns, resolve_panel_columns
from cell_eval2.metrics.discrimination import discrimination_score

DECLARE_ALL = {NO_TARGET_GENE_WILDCARD: None}
_DRUGS = {f"GENE{i}": f"DRUG{i}" for i in range(1, 6)}

# MEASURED on main af46ac73: of the 54 `full` names, exactly these 20 raise on the drug panel
# below (19 computed metrics and the derived member built from two of them). They are also the
# only ones a target-gene exclusion can move.
_RAISED_ON_MAIN = frozenset({
    "pds_l1", "pds_l2", "pds_cosine",
    "expr_mse_unbiased", "expr_mse_unbiased_capped", "expr_distance_unbiased",
    "expr_mse_unbiased_capped_norm",
    "de_wilcoxon_sig_jaccard", "de_wilcoxon_lfc_nmae",
    "de_wilcoxon_direction_fidelity", "de_wilcoxon_direction_fidelity_raw",
    "de_wilcoxon_direction_coverage", "de_wilcoxon_direction_yield",
    "de_wilcoxon_direction_yield_raw", "de_wilcoxon_direction_fidelity_yield",
    "de_wilcoxon_direction_fidelity_yield_raw", "de_wilcoxon_direction_reach",
    "de_wilcoxon_direction_reach_raw", "de_wilcoxon_direction_reach_unbounded",
    "de_wilcoxon_direction_reach_unbounded_raw",
})


# ---- the map's one reading --------------------------------------------------------------------

def test_target_gene_entry_reads_the_own_entry_then_the_wildcard():
    m = {"A-1": "A", "DRUG1": None, NO_TARGET_GENE_WILDCARD: None}
    assert target_gene_entry(m, "A-1") == "A"          # an own entry wins over the wildcard
    assert target_gene_entry(m, "DRUG1") is None        # declared by its own entry
    assert target_gene_entry(m, "DRUG2") is None        # declared by the wildcard
    assert target_gene_entry({"A-1": "A"}, "B") is UNMAPPED
    assert target_gene_entry(None, "B") is UNMAPPED
    assert target_gene_entry({}, "B") is UNMAPPED


@pytest.mark.parametrize("value", ["GENEX", ""])
def test_the_wildcard_is_defined_for_None_only(value):
    """Before #397 a "*" key was an ordinary label. `{"*": "GENEX"}` must not read as "map every
    label to GENEX", which nothing implements."""
    with pytest.raises(ValueError, match="must be None"):
        EvalConfig(target_gene_map={NO_TARGET_GENE_WILDCARD: value})
    with pytest.raises(ValueError, match="must be None"):
        EvalConfig.from_dict({"target_gene_map": {NO_TARGET_GENE_WILDCARD: value}})
    EvalConfig(target_gene_map={NO_TARGET_GENE_WILDCARD: None, "A-1": "A"})   # accepted


@pytest.mark.parametrize("tgm", [
    {123: None},                          # an unquoted numeric YAML label
    {"123": None, 123: "GENEX"},          # PDS keeps GENEX, DE keeps the declaration
    {123: "GENEX", "*": None},            # PDS maps 123 to GENEX, DE declares it
])
def test_a_map_that_declares_needs_string_labels_throughout(tgm):
    """The PDS-side resolver str()s the map's keys and the DE-side one does not, so a non-string
    key next to a declaration is read two ways (codex review). A map that declares nothing is
    left as it was before #397."""
    with pytest.raises(ValueError, match="every label in it must be a string"):
        EvalConfig(target_gene_map=tgm)


def test_a_string_labelled_declaration_and_an_undeclaring_map_are_accepted():
    EvalConfig(target_gene_map={"123": None})
    EvalConfig(target_gene_map={123: "GENEX"})                  # unchanged from main
    with pytest.raises(ValueError, match="every label in it must be a string"):
        EvalConfig.from_dict(yaml.safe_load("target_gene_map: {123: null}"))
    EvalConfig.from_dict(yaml.safe_load("target_gene_map: {'123': null}"))


def test_the_declaration_round_trips_through_yaml(tmp_path):
    path = tmp_path / "c.yaml"
    EvalConfig(target_gene_map={**DECLARE_ALL, "A-1": "A"}).to_yaml(str(path))
    assert yaml.safe_load(path.read_text())["target_gene_map"] == {"*": None, "A-1": "A"}
    assert EvalConfig.from_yaml(str(path)).target_gene_map == {"*": None, "A-1": "A"}


# ---- distances.resolve_exclusion_columns: pds_*, the moments metrics, the baseline ------------

def test_resolver_every_label_declared_excludes_nothing_and_passes_the_gate(caplog):
    with caplog.at_level(logging.WARNING, logger="cell_eval2.distances"):
        assert resolve_exclusion_columns(["DRUG1", "DRUG2"], ["g0", "g1"],
                                         target_gene_map=DECLARE_ALL) == {}
        assert resolve_panel_columns(["DRUG1", "DRUG2"], ["g0", "g1"],
                                     target_gene_map=DECLARE_ALL).size == 0
    assert not caplog.records            # deliberate, so no partial-resolution warning either


def test_resolver_never_matches_a_declared_label_by_name():
    """A cytokine named after a measured gene: by name it would drop column 0."""
    genes = ["IL6", "B", "g2"]
    assert resolve_exclusion_columns(["IL6", "B"], genes) == {0: 0, 1: 1}
    assert resolve_exclusion_columns(["IL6", "B"], genes,
                                     target_gene_map={"IL6": None}) == {1: 1}
    assert resolve_exclusion_columns(["IL6", "B"], genes, target_gene_map=DECLARE_ALL) == {}


def test_resolver_an_own_entry_beats_the_wildcard():
    cols = resolve_exclusion_columns(["B-1", "DRUG1"], ["A", "B"],
                                     target_gene_map={**DECLARE_ALL, "B-1": "B"})
    assert cols == {0: 1}


def test_resolver_still_raises_when_nothing_resolves_and_nothing_is_declared():
    """#248's protection is unchanged, and the message now names the declaration."""
    with pytest.raises(ValueError, match="NO perturbation resolves") as exc:
        resolve_exclusion_columns(["GENEX-1", "GENEY-1"], ["GENEX", "GENEY"])
    assert "--no-target-genes" in str(exc.value)


def test_resolver_a_declared_gate_label_counts_as_resolved():
    """The gate is judged on the whole panel (`gate_labels`), so a shard holding only an
    unresolved label passes when the panel declares the rest -- and the unresolved one is still
    reported, because it is NOT declared."""
    m = {"DRUG1": None, "DRUG2": None}
    assert resolve_exclusion_columns(["X-1"], ["g0"], target_gene_map=m,
                                     gate_labels=["X-1", "DRUG1", "DRUG2"]) == {}
    with pytest.raises(ValueError, match="NO perturbation resolves"):
        resolve_exclusion_columns(["X-1"], ["g0"], gate_labels=["X-1", "Y-1"])


def test_resolver_a_declared_label_carries_a_mixed_panel(caplog):
    """"Counts as resolved": a panel with declared labels and no resolved one passes the gate, and
    the undeclared, unresolved labels are reported as before."""
    with caplog.at_level(logging.WARNING, logger="cell_eval2.distances"):
        cols = resolve_exclusion_columns(["DRUG1", "X-1"], ["g0"],
                                         target_gene_map={"DRUG1": None})
    assert cols == {}
    rendered = " ".join(r.getMessage() for r in caplog.records)
    assert "0/1 labels resolved" in rendered and "['X-1']" in rendered


# ---- pds_*: the declaration is exclusion switched off, under either scope ----------------------

_PERTS = np.array(["ctrl", "IL6", "B"])     # both labels ARE measured genes
_GENES = np.asarray(["IL6", "B", "C"], dtype=str)
_REAL = np.array([[0.0, 0.0, 0.0], [-9.0, 1.0, 0.0], [0.0, 5.0, 5.0]])
_PRED = np.array([[0.0, 0.0, 0.0], [-9.0, 4.0, 4.0], [0.0, 5.0, 5.0]])


def _pds(**kw):
    return discrimination_score(pred_bulk=(_PERTS, _PRED), real_bulk=(_PERTS, _REAL),
                                control="ctrl", rank_denominator="n", genes=_GENES,
                                control_source="pred", **kw)


@pytest.mark.parametrize("scope", ["panel", "row"])
@pytest.mark.parametrize("distance", ["l1", "l2", "cosine"])
def test_pds_declared_equals_exclusion_off(scope, distance):
    declared = _pds(distance=distance, exclusion_scope=scope, target_gene_map=DECLARE_ALL)
    off = _pds(distance=distance, exclusion_scope=scope, exclude_target_gene=False)
    assert declared == off                                   # bit-identical, not approx
    assert declared == {"IL6": 1.0, "B": 1.0}
    # ...and the comparison can fail: by name, IL6's own transcript leaves and B wins its row.
    assert _pds(distance=distance, exclusion_scope=scope) == {"IL6": 0.5, "B": 1.0}


@pytest.mark.parametrize("scope", ["panel", "row"])
def test_pds_xp_kernel_matches_cpu_with_a_declaration(scope):
    """The kernel cupy runs, on numpy: the GPU path reaches the same resolver host-side."""
    from cell_eval2.gpu.distances import _discrimination_ranks_xp

    for tgm in (DECLARE_ALL, {"IL6": None}):
        cpu = _pds(distance="cosine", exclusion_scope=scope, target_gene_map=tgm)
        got = _discrimination_ranks_xp(
            np, (_PERTS, _REAL), (_PERTS, _PRED), genes=_GENES, metric="cosine",
            exclude_target_gene=True, exclusion_scope=scope, rank_denominator="n",
            tie_policy="midrank", pert_chunk=2, control="ctrl", control_source="pred",
            target_gene_map=tgm)
        assert got == pytest.approx(cpu, rel=1e-12, abs=1e-12)


@pytest.mark.parametrize("scope", ["panel", "row"])
@pytest.mark.parametrize("distance", ["l1", "l2", "cosine"])
def test_pds_declared_is_bit_identical_to_exclusion_off_on_float_data(scope, distance):
    """The 3-gene integer panel above cannot see summation order; this one can. With every
    label declared, panel scope removes no column, and the reduced operand must keep the
    unreduced one's memory layout: an F-ordered copy changes what BLAS sums first, and the
    l2/cosine distances then differ in the last bits (measured, codex review)."""
    from cell_eval2.distances import pairwise_full, panel_reduced
    from cell_eval2.gpu.distances import _discrimination_ranks_xp

    rng = np.random.default_rng(0)
    names = np.array([f"D{i}" for i in range(80)] + ["ctrl"])
    genes = np.array([f"g{i}" for i in range(400)])
    real = (names, rng.normal(size=(names.size, genes.size)))
    pred = (names, rng.normal(size=(names.size, genes.size)))
    eff = rng.normal(size=(names.size, genes.size))
    reduced = panel_reduced(eff, np.array([], dtype=np.intp))
    assert reduced.flags.c_contiguous and reduced is not eff
    assert np.array_equal(pairwise_full(reduced, reduced, distance),
                          pairwise_full(eff, eff, distance))
    kw = dict(control="ctrl", distance=distance, rank_denominator="n-1", tie_policy="midrank",
              control_source="real", genes=genes, exclusion_scope=scope)
    off = discrimination_score(pred_bulk=pred, real_bulk=real, exclude_target_gene=False, **kw)
    assert discrimination_score(pred_bulk=pred, real_bulk=real, target_gene_map=DECLARE_ALL,
                                **kw) == off
    xp = _discrimination_ranks_xp(
        np, real, pred, genes=genes, metric=distance, exclude_target_gene=True,
        exclusion_scope=scope, rank_denominator="n-1", tie_policy="midrank", pert_chunk=32,
        control="ctrl", control_source="real", target_gene_map=DECLARE_ALL)
    assert xp == _discrimination_ranks_xp(
        np, real, pred, genes=genes, metric=distance, exclude_target_gene=False,
        exclusion_scope=scope, rank_denominator="n-1", tie_policy="midrank", pert_chunk=32,
        control="ctrl", control_source="real")


def test_pds_cuda_kernel_honours_the_declaration():
    from cell_eval2.gpu import resolve_device
    from cell_eval2.gpu.distances import discrimination_ranks

    if resolve_device("auto") != "cuda":
        pytest.skip("no CUDA device")
    for scope in ("panel", "row"):
        got = discrimination_ranks(
            (_PERTS, _REAL), (_PERTS, _PRED), perts=None, genes=_GENES, metric="l1",
            exclude_target_gene=True, exclusion_scope=scope, rank_denominator="n",
            tie_policy="midrank", pert_chunk=2, device="cuda", control="ctrl",
            control_source="pred", target_gene_map=DECLARE_ALL)
        assert got == pytest.approx({"IL6": 1.0, "B": 1.0}, rel=1e-12, abs=1e-12)


# ---- the moments metrics: declared == the unexcluded definition (genes=None) ------------------

def _moments_panel(n_perts=4, n_genes=12, seed=3):
    """Hand-built bulks + moments whose labels ARE measured genes, each knocked down hard."""
    from cell_eval2.moments import GroupMoments

    rng = np.random.default_rng(seed)
    genes = np.array([f"g{i}" for i in range(n_genes)])
    perts = np.array([str(g) for g in genes[:n_perts]] + ["non-targeting"])
    mu_ctrl = rng.uniform(0.5, 4.0, size=n_genes)
    mu_real = np.tile(mu_ctrl, (n_perts + 1, 1))
    for p in range(n_perts):
        mu_real[p] += rng.normal(0.0, 0.2, size=n_genes)
        mu_real[p, p] -= 3.0
    mu_pred = mu_real + rng.normal(0.0, 0.3, size=mu_real.shape)
    mom = GroupMoments(perts=perts, counts=np.full(n_perts + 1, 500.0),
                       sumsq=np.zeros(n_perts + 1), jk=np.full(n_perts + 1, 0.01))
    return perts, genes, mu_real, mu_pred, mom


@pytest.mark.parametrize("name", ["expr_mse_unbiased", "expr_mse_unbiased_capped",
                                  "expr_distance_unbiased"])
def test_moments_declared_equal_the_unexcluded_definition(name, caplog):
    from cell_eval2.catalog import CATALOG

    perts, genes, mu_real, mu_pred, mom = _moments_panel()
    func = CATALOG[name].func
    kw = dict(comparator="bulk_lognorm", pert_col="target", control="non-targeting",
              pred_bulk=(perts, mu_pred), real_bulk=(perts, mu_real),
              pred_moments=mom, real_moments=mom)
    declared = func(**kw, genes=genes, target_gene_map=DECLARE_ALL)
    with caplog.at_level(logging.WARNING, logger="cell_eval2.metrics.delta"):
        unexcluded = func(**kw, genes=None)      # the documented "excludes nothing" route
    assert declared == unexcluded
    assert func(**kw, genes=genes) != declared   # by name, each label's own gene would leave


# ---- the DE side: resolution, the gate, the anti-join -----------------------------------------

def _real(rows):
    return pl.DataFrame({"target": [r[0] for r in rows], "feature": [r[1] for r in rows]},
                        schema={"target": pl.String, "feature": pl.String})


def test_de_resolution_counts_a_declared_target_and_never_matches_it_by_name():
    real = _real([("IL6", "IL6"), ("IL6", "g0"), ("B", "B"), ("DRUG1", "g0")])
    res = resolve_target_genes(real, ["B", "DRUG1", "IL6"],
                               target_gene_map={"IL6": None, "DRUG1": None})
    assert res.mapping == {"B": "B"}                # IL6 IS a feature, and is not matched
    assert (res.n_resolved, res.n_declared, res.unresolved) == (1, 2, ())
    every = resolve_target_genes(real, ["B", "DRUG1", "IL6"], target_gene_map=DECLARE_ALL)
    assert (every.n_resolved, every.n_declared, every.mapping) == (0, 3, {})
    own = resolve_target_genes(real, ["B", "DRUG1"],
                               target_gene_map={**DECLARE_ALL, "B": "g0"})
    assert (own.mapping, own.n_declared) == ({"B": "g0"}, 1)


def test_de_gate_passes_a_declared_panel_and_still_raises_an_undeclared_one():
    perts = ["DRUG1", "DRUG2"]
    real = _real([("DRUG1", "g0"), ("DRUG2", "g1")])
    require_resolution(perts, resolve_target_genes(real, perts, target_gene_map=DECLARE_ALL))
    with pytest.raises(ValueError, match="no target resolves") as exc:
        require_resolution(perts, resolve_target_genes(real, perts))
    assert "--no-target-genes" in str(exc.value)


def test_de_declared_targets_exclude_nothing():
    df = pl.DataFrame({"target": ["IL6", "IL6", "B"], "feature": ["IL6", "g0", "B"]})
    real = _real(df.select("target", "feature").rows())
    declared = resolve_target_genes(real, ["B", "IL6"], target_gene_map=DECLARE_ALL)
    assert exclude_on_target(df, declared).equals(df)
    by_name = resolve_target_genes(real, ["B", "IL6"])
    assert exclude_on_target(df, by_name).height == 1           # the comparison can fail


def test_de_resolution_with_a_declaration_copies_and_replaces():
    import copy
    import dataclasses

    res = TargetResolution({"B": "B"}, 3, n_features=4, n_declared=2)
    assert copy.deepcopy(res) == res
    assert dataclasses.replace(res, n_targets=5).n_declared == 2
    assert dataclasses.asdict(res)["n_declared"] == 2
    assert TargetResolution({}, 0).n_declared == 0               # positional callers unchanged


# ---- end to end: the full v2 profile on an .h5ad pair -----------------------------------------

def _relabel(a, ren):
    a = a.copy()
    a.obs["target"] = [ren.get(t, t) for t in a.obs["target"].astype(str)]
    return a


@pytest.fixture(scope="module")
def panels():
    _, real, sub = counts_bundle_inputs()
    return {"cytokine": (sub, real),
            "drug": (_relabel(sub, _DRUGS), _relabel(real, _DRUGS))}


def _values(df, ren=None):
    ren = ren or {}
    return {(ren.get(p, p), m): v for p, m, v in
            df.select("perturbation", "metric", "value").iter_rows()}


def _same(a, b):
    return a == b or (a != a and b != b)          # NaN-aware, exact


@pytest.fixture(scope="module")
def full_runs(panels):
    from cell_eval2.run import compute_metrics

    cfg = EvalConfig(metrics="full", input_type="counts", device="cpu")
    declared = replace(cfg, target_gene_map=DECLARE_ALL)
    return {"cfg": cfg,
            "drug": compute_metrics(*panels["drug"], config=declared),
            "cytokine": compute_metrics(*panels["cytokine"], config=declared),
            "by_name": compute_metrics(*panels["cytokine"], config=cfg)}


def test_full_profile_raises_on_a_drug_panel_without_the_declaration(panels, full_runs):
    from cell_eval2.run import compute_metrics

    with pytest.raises(ValueError, match="resolves to a gene"):
        compute_metrics(*panels["drug"], config=full_runs["cfg"])


def test_full_profile_scores_every_metric_on_a_declared_drug_panel(full_runs):
    from cell_eval2.run import aggregate_metrics_wide, metric_output_names

    names = metric_output_names(full_runs["cfg"])
    assert _RAISED_ON_MAIN <= set(names)
    wide = aggregate_metrics_wide(full_runs["drug"], metrics=names)
    assert set(wide.columns[1:]) == set(names)
    mean = wide.filter(pl.col("statistic") == "mean")
    for m in sorted(_RAISED_ON_MAIN):
        assert np.isfinite(mean[m][0]), m


def test_a_declared_label_is_never_matched_by_name_end_to_end(full_runs):
    """The cytokine panel's labels are measured genes; declared, it scores exactly as the drug
    panel, whose labels match nothing."""
    drug, cyt = _values(full_runs["drug"]), _values(full_runs["cytokine"], _DRUGS)
    assert drug.keys() == cyt.keys()
    assert all(_same(drug[k], cyt[k]) for k in drug)
    # ...and name matching would have moved them, so the comparison above can fail.
    by_name = _values(full_runs["by_name"], _DRUGS)
    moved = {m for (p, m), v in cyt.items() if not _same(v, by_name[(p, m)])}
    assert {"expr_mse_unbiased", "expr_distance_unbiased", "de_wilcoxon_sig_jaccard",
            "de_wilcoxon_lfc_nmae"} <= moved
    assert moved <= _RAISED_ON_MAIN, moved - _RAISED_ON_MAIN


def test_the_lfc_nmae_reference_honours_the_declaration(panels):
    from cell_eval2.lfc_nmae_ref import compute_lfc_nmae_reference

    cfg = EvalConfig(input_type="counts", device="cpu", target_gene_map=DECLARE_ALL)
    _, cyt_real = panels["cytokine"]
    _, drug_real = panels["drug"]
    cyt, _ = compute_lfc_nmae_reference(cyt_real, config=cfg)
    drug, _ = compute_lfc_nmae_reference(drug_real, config=cfg)
    cyt = cyt.with_columns(pl.Series("perturbation",
                                     [_DRUGS.get(p, p) for p in cyt["perturbation"]]))
    assert cyt.sort("perturbation").equals(drug.sort("perturbation"))
    by_name, _ = compute_lfc_nmae_reference(
        cyt_real, config=replace(cfg, target_gene_map=None))
    by_name = by_name.with_columns(pl.Series(
        "perturbation", [_DRUGS.get(p, p) for p in by_name["perturbation"]]))
    assert not by_name.sort("perturbation").equals(cyt.sort("perturbation"))


def test_the_generic_response_baseline_honours_the_declaration(panels):
    from cell_eval2.baseline import generic_response_profile

    _, cyt_real = panels["cytokine"]
    _, drug_real = panels["drug"]
    kw = dict(pert_col="target", control="non-targeting", exclude_target_gene=True)
    declared = generic_response_profile(cyt_real, **kw, target_gene_map=DECLARE_ALL)
    plain = generic_response_profile(cyt_real, pert_col="target", control="non-targeting",
                                     exclude_target_gene=False)
    assert declared.n_excluded == 0
    np.testing.assert_array_equal(declared.values, plain.values)
    assert generic_response_profile(cyt_real, **kw).n_excluded == 5   # by name: all five
    with pytest.raises(ValueError, match="NO perturbation resolves"):
        generic_response_profile(drug_real, **kw)
    assert generic_response_profile(drug_real, **kw, target_gene_map=DECLARE_ALL).n_excluded == 0


def test_the_baseline_two_perturbation_guard_skips_a_declared_label():
    """One perturbation loses no contribution when it has no target gene, so its profile is
    defined; undeclared, the guard still raises."""
    from cell_eval2.baseline import generic_response_profile

    X = np.array([[1.0, 2.0], [3.0, 4.0], [9.0, 9.0]])
    a = ad.AnnData(X=X, obs=pd.DataFrame({"target": ["DRUG1", "DRUG1", "non-targeting"]},
                                         index=["c0", "c1", "c2"]),
                   var=pd.DataFrame(index=["g0", "g1"]))
    kw = dict(pert_col="target", control="non-targeting", exclude_target_gene=True)
    for tgm in (DECLARE_ALL, {"DRUG1": None}):
        p = generic_response_profile(a, **kw, target_gene_map=tgm)
        np.testing.assert_array_equal(p.values, [2.0, 3.0])
        assert (p.n_perturbations, p.n_excluded) == (1, 0)
    for tgm in (None, {"DRUG1": "g0"}):
        with pytest.raises(ValueError, match="needs at least 2 non-control perturbations"):
            generic_response_profile(a, **kw, target_gene_map=tgm)


# ---- the CLI flag ------------------------------------------------------------------------------

def _cfg_for(argv):
    import shlex

    from cell_eval2.cli import _build_cfg, _build_parser

    return _build_cfg(_build_parser().parse_args(shlex.split(argv)))


def test_the_flag_sets_the_wildcard_on_every_config_subcommand():
    for argv in ("run -ap p -ar r", "prep-cache --side real --adata r",
                 "prep-real-bundle --real r --baseline b -o o", "baseline -ar r"):
        assert _cfg_for(f"{argv} --no-target-genes").target_gene_map == DECLARE_ALL, argv
        assert _cfg_for(argv).target_gene_map is None, argv


def test_the_flag_merges_into_a_map_the_config_already_holds():
    cfg = _cfg_for("run -ap p -ar r --set 'target_gene_map={B-1: B}' --no-target-genes")
    assert cfg.target_gene_map == {"B-1": "B", "*": None}


def test_the_flag_on_a_map_with_a_numeric_label_exits_cleanly(tmp_path):
    path = tmp_path / "c.yaml"
    path.write_text("target_gene_map: {123: GENEX}\n")
    assert _cfg_for(f"run -ap p -ar r --config {path}").target_gene_map == {123: "GENEX"}
    with pytest.raises(SystemExit, match="--no-target-genes: .*must be a string"):
        _cfg_for(f"run -ap p -ar r --config {path} --no-target-genes")


def test_set_cannot_give_the_wildcard_a_gene():
    with pytest.raises(SystemExit, match="must be None"):
        _cfg_for("run -ap p -ar r --set 'target_gene_map={\"*\": GENEX}'")


def test_cli_run_scores_an_h5ad_drug_pair_and_records_the_declaration(panels, tmp_path):
    from cell_eval2.cli import main

    pred, real = panels["drug"]
    pp, rp = tmp_path / "pred.h5ad", tmp_path / "real.h5ad"
    pred.write_h5ad(pp)
    real.write_h5ad(rp)
    argv = ["run", "-ap", str(pp), "-ar", str(rp), "--set", "device=cpu"]
    with pytest.raises(ValueError, match="resolves to a gene"):
        main([*argv, "-o", str(tmp_path / "plain")])
    out = tmp_path / "declared"
    main([*argv, "-o", str(out), "--no-target-genes"])
    assert pl.read_csv(out / "results.csv").height > 0
    params = yaml.safe_load((out / "run_params.yaml").read_text())
    assert params["target_gene_map"] == {"*": None}


# ---- what the declaration does to identity -----------------------------------------------------

def test_the_declaration_takes_a_run_off_the_competition_rule():
    from cell_eval2 import competition

    comp = EvalConfig.from_preset("vcc2026")
    assert competition.rule_config_hash(replace(comp, target_gene_map=DECLARE_ALL)) != (
        competition.rule_config_hash(comp))


def test_the_declaration_keys_apart_from_every_other_map():
    from cell_eval2.baseline import config_digest
    from cell_eval2.run import _result_config_digest

    maps = (None, {}, DECLARE_ALL, {"DRUG1": None}, {"DRUG1": "__no_target_gene__"})
    results = {_result_config_digest(EvalConfig(target_gene_map=m), de_backend_used=False,
                                     comparator="bulk_lognorm") for m in maps}
    baselines = {config_digest(EvalConfig(metrics="minimal", target_gene_map=m),
                               comparator="bulk_lognorm") for m in maps}
    assert len(results) == len(maps) and len(baselines) == len(maps)


# A "*" key was an ordinary label before #397, so `{"*": None}` digests the same on both sides of
# it while computing something else; `run._NO_TARGET_GENE_SEMANTICS` is what retires an entry
# built from the old reading. Scoped to maps that hold a `None` value.
_DECLARING = (DECLARE_ALL, {"A": None}, {"A": None, "B": "GENE2"})
_NOT_DECLARING = (None, {}, {"A": "GENE1"}, {"A": "__no_target_gene__"})


def test_the_semantics_term_is_read_off_the_map_alone():
    from cell_eval2.run import _no_target_gene_declared

    assert all(_no_target_gene_declared(EvalConfig(target_gene_map=m)) for m in _DECLARING)
    assert not any(_no_target_gene_declared(EvalConfig(target_gene_map=m))
                   for m in _NOT_DECLARING)


def test_the_result_cache_key_moves_with_the_counter_only_for_a_declaring_map(monkeypatch):
    from cell_eval2 import run

    cfgs = ([EvalConfig(target_gene_map=m) for m in _DECLARING + _NOT_DECLARING]
            + [EvalConfig.from_preset("vcc2026")])

    def keys():
        return [run._result_config_digest(c, de_backend_used=True, comparator="bulk_lognorm")
                for c in cfgs]

    before = keys()
    monkeypatch.setattr(run, "_NO_TARGET_GENE_SEMANTICS", run._NO_TARGET_GENE_SEMANTICS + 1)
    moved = [b != a for b, a in zip(before, keys())]
    assert moved == [True] * len(_DECLARING) + [False] * (len(_NOT_DECLARING) + 1)


def test_the_anchor_identity_carries_the_term_only_where_the_map_enters_it():
    from _helpers import _counts_adata_fp64

    from cell_eval2.anchor import anchor_semantic_params
    from cell_eval2.catalog import resolve_metrics
    from cell_eval2.run import _NO_TARGET_GENE_SEMANTICS

    real = _counts_adata_fp64(seed=0, per_group=6, g=5)
    # `expr_mae` reads no target gene, so neither the map nor the term enters its identity.
    for metrics, m, present in ((["pds_cosine"], DECLARE_ALL, True),
                                (["de_wilcoxon_sig_jaccard"], {"A": None}, True),
                                (["pds_cosine"], {"A": "GENE1"}, False),
                                (["pds_cosine"], None, False),
                                (["expr_mae"], DECLARE_ALL, False)):
        cfg = EvalConfig(metrics=metrics, version="v2", device="cpu", input_type="counts",
                         pert_col="target", target_gene_map=m)
        names = list(resolve_metrics(cfg.metrics, version=cfg.version)[0])
        params = anchor_semantic_params(cfg, real, names)
        assert params.get("no_target_gene_semantics") == (
            _NO_TARGET_GENE_SEMANTICS if present else None), (metrics, m)


def test_partials_carry_the_term(monkeypatch):
    from cell_eval2 import run
    from cell_eval2.partition import _PARTIAL_SEMANTICS_SCHEMA, result_semantics

    names = ["pds_cosine", "expr_mae"]
    sem = result_semantics(names, comparator="lognorm")
    assert sem["no_target_gene_semantics"] == run._NO_TARGET_GENE_SEMANTICS == 1
    assert sem["schema"] == _PARTIAL_SEMANTICS_SCHEMA >= 7
    monkeypatch.setattr(run, "_NO_TARGET_GENE_SEMANTICS", 2)
    assert result_semantics(names, comparator="lognorm") != sem


# ---- the streaming path (score_cellstream), where arcbench-nf scores these panels --------------

def _write_drug_cell(tmp_path, name, seed, *, n_per=30, n_genes=50):
    pytest.importorskip("cellstream")
    import scipy.sparse as sp
    from cellstream.cell import write_cell_archive

    perts = ["control", "DRUG_A", "DRUG_B", "DRUG_C"]
    rng = np.random.default_rng(seed)
    n = n_per * len(perts)
    X = sp.csr_matrix(rng.poisson(0.5, size=(n, n_genes)).astype(np.float32))
    obs = pd.DataFrame({"perturbation": np.repeat(perts, n_per), "context": "ctxA",
                        "control_value": "control", "dataset": "dsX"},
                       index=[f"{name}_c{i}" for i in range(n)])
    adata = ad.AnnData(X=X, obs=obs, var=pd.DataFrame(index=[f"g{j}" for j in range(n_genes)]))
    shad = tmp_path / f"{name}.shad"
    write_cell_archive(adata, shad, group_by="perturbation",
                       reference=(obs["perturbation"] == "control").to_numpy(),
                       codec="zstd", overwrite=True)
    return shad


def test_score_cellstream_scores_a_declared_drug_panel_gpu(tmp_path):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("score_cellstream requires a CUDA device (gpudge external-reference DE)")
    pytest.importorskip("gpudge")
    from cell_eval2.cellstream import score_cellstream
    from cell_eval2.h5ad_manifest import MemBudget
    from cell_eval2.run import compute_metrics

    pred, real = _write_drug_cell(tmp_path, "pred", 1), _write_drug_cell(tmp_path, "real", 2)
    big = MemBudget(host_bytes=10**12, gpu_bytes=10**12)
    cfg = EvalConfig(device="cuda", pert_col="perturbation", control="control",
                     metrics=full_minus_moments())
    cfg = replace(cfg, de=replace(cfg.de, auc_pval_floor="replace_zero",
                                  auc_pval_floor_value=1e-10, fdr_scope="per_pert"))
    with pytest.raises(ValueError, match="resolves to a gene"):
        score_cellstream(pred, real, config=cfg, mem_budget=big)
    declared = replace(cfg, target_gene_map=DECLARE_ALL)
    got = score_cellstream(pred, real, config=declared, mem_budget=big).per_pert
    want = compute_metrics(pred, real, config=declared)
    g, w = _values(got), _values(want)
    assert g.keys() == w.keys()
    for k in g:
        assert g[k] == pytest.approx(w[k], rel=1e-7, abs=1e-7, nan_ok=True), k
