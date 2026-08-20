"""Issue #343: the target-gene exclusion is applied to the PANEL, not to the row.

Under the legacy rule (`exclusion_scope="row"`, kept for upstream cell-eval parity)
`correct_excluded_gene` rewrites the whole row `full[i, :]` dropping only `target(p_i)`. So
in cell `(i, j)` of the distance matrix, reference perturbation `j`'s OWN knockdown -- the
largest and most predictable coordinate of `real_eff[j]` -- is still there. A submission that
spikes the panel's *other* target genes is then anti-correlated with every off-diagonal
competitor while its own diagonal, where its gene IS dropped, sits at cosine 0; the self-match
wins by construction, on nothing but the target list every participant is handed.

`exclusion_scope="panel"` (the v2 default) removes every panel target gene from the ranked
feature space once, so no cell of the matrix can see any of them.

The arm below is the shape measured on the three official val contexts, where it took
`pds_cosine` to 0.7982 / 0.7570 / 0.7614 against baselines of 0.5304 / 0.5284 / 0.5102.
"""

import numpy as np
import pytest
from cell_eval2.config import DiscriminationParams, EvalConfig
from cell_eval2.distances import panel_reduced, resolve_panel_columns
from cell_eval2.metrics.discrimination import discrimination_score
from cell_eval2.run import _result_config_digest

CTRL = "ctrl"
_KW = dict(control=CTRL, distance="cosine", rank_denominator="n-1", tie_policy="midrank",
           exclude_target_gene=True, control_source="pred")


def _panel(n_targets=12, n_extra=40, seed=0):
    """A panel whose first `n_targets` genes ARE the targets, plus unrelated biology.

    Each real perturbation knocks its own gene down hard and moves a handful of other genes;
    that off-target biology is what a real submission would have to get right, and what the
    attack arm deliberately does not have.
    """
    rng = np.random.default_rng(seed)
    n_genes = n_targets + n_extra
    genes = np.array([f"g{i:03d}" for i in range(n_genes)], dtype=str)
    targets = [f"g{i:03d}" for i in range(n_targets)]
    labels = np.array([CTRL] + targets)
    ctrl = rng.uniform(0.5, 3.0, size=n_genes)
    real = np.tile(ctrl, (len(labels), 1))
    for i in range(n_targets):
        d = np.zeros(n_genes)
        d[rng.choice(n_genes, 6, replace=False)] = rng.normal(0, 0.4, 6)
        d[i] = -1.5                                   # the on-target knockdown
        real[i + 1] += d
    return genes, labels, ctrl, real, targets


def _antitarget_arm(genes, labels, ctrl, spike=0.05, kd=1.5):
    """Spike every panel target gene EXCEPT this perturbation's own; knock its own down."""
    n_targets = len(labels) - 1
    pred = np.tile(ctrl, (len(labels), 1))
    for i in range(n_targets):
        d = np.zeros(genes.size)
        d[:n_targets] = spike
        d[i] = -kd
        pred[i + 1] += d
    return pred


def _mean(scores):
    return float(np.mean(list(scores.values())))


# ------------------------------------------------------------------ the defect and the fix

def test_row_scope_scores_the_antitarget_arm_far_above_chance():
    """The legacy rule is what the attack needs; this pins that it really is exploitable."""
    genes, labels, ctrl, real, _ = _panel()
    arm = _antitarget_arm(genes, labels, ctrl)
    row = _mean(discrimination_score(pred_bulk=(labels, arm), real_bulk=(labels, real),
                                     genes=genes, exclusion_scope="row", **_KW))
    assert row > 0.85, f"the row-scope channel should be wide open, got {row}"


def test_panel_scope_returns_the_antitarget_arm_to_chance():
    genes, labels, ctrl, real, _ = _panel()
    arm = _antitarget_arm(genes, labels, ctrl)
    panel = _mean(discrimination_score(pred_bulk=(labels, arm), real_bulk=(labels, real),
                                       genes=genes, exclusion_scope="panel", **_KW))
    # Exactly 0.5: with every panel gene gone the arm's delta is identically zero, every row
    # ties at cosine distance 1, and midrank puts the match at the middle of one tied block.
    assert panel == pytest.approx(0.5, abs=1e-12), panel


def test_panel_scope_is_the_v2_default():
    """A caller that says nothing gets the corrected rule, not the legacy one."""
    genes, labels, ctrl, real, _ = _panel()
    arm = _antitarget_arm(genes, labels, ctrl)
    assert DiscriminationParams().exclusion_scope == "panel"
    assert _mean(discrimination_score(pred_bulk=(labels, arm), real_bulk=(labels, real),
                                      genes=genes, **_KW)) == pytest.approx(0.5, abs=1e-12)


def test_the_spike_amplitude_does_not_matter_under_either_scope():
    """Cosine is scale-invariant, so the attacker can make the spike arbitrarily small --
    which is exactly what keeps the expression-error members at 'no change'. The fix must not
    depend on the amplitude being large enough to notice."""
    genes, labels, ctrl, real, _ = _panel()
    for spike in (1e-4, 1e-2, 1.0):
        arm = _antitarget_arm(genes, labels, ctrl, spike=spike)
        assert _mean(discrimination_score(
            pred_bulk=(labels, arm), real_bulk=(labels, real), genes=genes,
            exclusion_scope="row", **_KW)) > 0.85
        assert _mean(discrimination_score(
            pred_bulk=(labels, arm), real_bulk=(labels, real), genes=genes,
            exclusion_scope="panel", **_KW)) == pytest.approx(0.5, abs=1e-12)


# ------------------------------------------------------------------ the legitimate end

def test_a_perfect_submission_still_scores_one_under_panel_scope():
    genes, labels, _, real, _ = _panel()
    assert discrimination_score(pred_bulk=(labels, real), real_bulk=(labels, real),
                                genes=genes, exclusion_scope="panel", **_KW) == {
        t: 1.0 for t in labels[1:]}


def test_a_control_paste_still_scores_chance_under_panel_scope():
    genes, labels, ctrl, real, _ = _panel()
    paste = np.tile(ctrl, (len(labels), 1))
    assert _mean(discrimination_score(pred_bulk=(labels, paste), real_bulk=(labels, real),
                                      genes=genes, exclusion_scope="panel",
                                      **_KW)) == pytest.approx(0.5, abs=1e-12)


def test_off_panel_biology_still_discriminates():
    """The fix removes the panel's targets, NOT the metric's power: a submission carrying the
    real off-target response still ranks itself first."""
    genes, labels, ctrl, real, _ = _panel()
    n_targets = len(labels) - 1
    off_panel_only = np.tile(ctrl, (len(labels), 1))
    off_panel_only[1:, n_targets:] = real[1:, n_targets:]     # keep only the non-target genes
    assert _mean(discrimination_score(
        pred_bulk=(labels, off_panel_only), real_bulk=(labels, real), genes=genes,
        exclusion_scope="panel", **_KW)) == pytest.approx(1.0)


# ------------------------------------------------------------------ structure

def test_a_pred_shard_is_scored_in_the_same_feature_space_as_the_whole():
    """`scale.py` restricts the pred bulks and `partition_inmem.py` passes one piece at a
    time, while both hand the REAL bulks over whole. The excluded set comes from the real
    panel for exactly that reason, so a shard's scores are bit-identical to the whole's."""
    genes, labels, _, real, targets = _panel()
    whole = discrimination_score(pred_bulk=(labels, real), real_bulk=(labels, real),
                                 genes=genes, exclusion_scope="panel", **_KW)
    keep = [0, 2, 5]                                   # ctrl + two targets
    shard = discrimination_score(
        pred_bulk=(labels[keep], real[keep]), real_bulk=(labels, real),
        genes=genes, exclusion_scope="panel", **_KW)
    assert set(shard) == {labels[2], labels[5]}
    for k in shard:
        assert shard[k] == whole[k]


def test_panel_columns_resolve_through_target_gene_map():
    """Construct-ID labels ('ADNP-1') resolve through the map, the same route #248 built for
    the row scope -- one definition, so the two scopes cannot disagree on what a target is."""
    genes = np.array(["A", "B", "C"], dtype=str)
    perts = np.array(["A-1", "B-1", "ZZZ-1"])
    cols = resolve_panel_columns(perts, genes,
                                 target_gene_map={"A-1": "A", "B-1": "B", "ZZZ-1": "NOPE"})
    np.testing.assert_array_equal(cols, np.array([0, 1]))   # ZZZ-1 resolves to nothing


def test_duplicate_target_labels_collapse_to_one_column():
    genes = np.array(["A", "B", "C"], dtype=str)
    cols = resolve_panel_columns(np.array(["A", "A", "B"]), genes)
    np.testing.assert_array_equal(cols, np.array([0, 1]))


def test_a_panel_covering_every_gene_raises_rather_than_returning_a_flat_half():
    """With no features left every row ties at distance 1 and the metric returns exactly 0.5
    for every perturbation regardless of the submission -- silence dressed as a number."""
    genes = np.array(["A", "B"], dtype=str)
    labels = np.array([CTRL, "A", "B"])
    real = np.array([[0.0, 0.0], [2.0, 0.0], [0.0, 2.0]])
    with pytest.raises(ValueError, match="would remove every one of the"):
        discrimination_score(pred_bulk=(labels, real), real_bulk=(labels, real),
                             genes=genes, exclusion_scope="panel", **_KW)


def test_panel_reduced_raises_on_an_empty_residue():
    with pytest.raises(ValueError, match="ranked feature space is empty"):
        panel_reduced(np.zeros((2, 2)), np.array([0, 1]))


def test_an_unknown_scope_is_rejected_at_both_the_metric_and_the_config():
    genes, labels, _, real, _ = _panel()
    with pytest.raises(ValueError, match="exclusion_scope must be"):
        discrimination_score(pred_bulk=(labels, real), real_bulk=(labels, real),
                             genes=genes, exclusion_scope="rows", **_KW)
    with pytest.raises(ValueError, match="exclusion_scope must be one of"):
        DiscriminationParams(exclusion_scope="rows")


def test_the_scope_is_inert_when_exclusion_is_off():
    genes, labels, ctrl, real, _ = _panel()
    arm = _antitarget_arm(genes, labels, ctrl)
    kw = {**_KW, "exclude_target_gene": False}
    a = discrimination_score(pred_bulk=(labels, arm), real_bulk=(labels, real),
                             genes=genes, exclusion_scope="row", **kw)
    b = discrimination_score(pred_bulk=(labels, arm), real_bulk=(labels, real),
                             genes=genes, exclusion_scope="panel", **kw)
    assert a == b


# ------------------------------------------------------------------ presets and cache

@pytest.mark.parametrize("preset,scope", [("v1", "row"), ("cell-eval-0.7.6", "row"),
                                          ("v2", "panel"), ("vcc2026", "panel")])
def test_presets_pin_the_scope_they_mean(preset, scope):
    """v1 and its family reproduce upstream cell-eval byte-for-byte, so they keep the legacy
    rule; the corrected presets take the panel rule."""
    assert EvalConfig.from_preset(preset).discrimination.exclusion_scope == scope


def test_the_scope_keys_the_result_cache_apart_for_pds_runs():
    cfg = EvalConfig.from_preset("vcc2026")
    other = EvalConfig.from_dict({**cfg.to_dict(),
                                  "discrimination": {**cfg.to_dict()["discrimination"],
                                                     "exclusion_scope": "row"}})
    kw = dict(de_backend_used=False, comparator="bulk_lognorm", pds_exclusion_used=True)
    assert _result_config_digest(cfg, **kw) != _result_config_digest(other, **kw)


def test_the_scope_is_dropped_from_the_key_where_it_cannot_move_a_number():
    """A run with no pds_* metric, or with exclusion off, keeps its warm cache -- the same
    inert-field rule target_gene_map and replicate_col already follow."""
    cfg = EvalConfig.from_preset("vcc2026")
    other = EvalConfig.from_dict({**cfg.to_dict(),
                                  "discrimination": {**cfg.to_dict()["discrimination"],
                                                     "exclusion_scope": "row"}})
    kw = dict(de_backend_used=True, comparator="bulk_lognorm", pds_exclusion_used=False)
    assert _result_config_digest(cfg, **kw) == _result_config_digest(other, **kw)


def test_the_anchor_semantic_identity_moves_with_the_scope():
    """An anchor frozen under one scope must not enrol against a run scored under the other.

    Asserts the identity actually MOVES, not merely that the field is in the tuple: tuple
    membership proves the constant was edited, not that anything reads it (cross-provider
    review of #343). The end-to-end mutation matrix lives in test_anchor_artifact.py.
    """
    from dataclasses import replace

    import anndata as ad
    import pandas as pd

    from cell_eval2.anchor import anchor_semantic_params

    genes, labels, _, real, _ = _panel(n_targets=3, n_extra=5)
    obs = pd.DataFrame({"perturbation": np.repeat(labels, 2).astype(str)})
    real_ad = ad.AnnData(np.repeat(np.rint(real * 10), 2, axis=0).astype(np.float32),
                         obs=obs, var=pd.DataFrame(index=genes))
    cfg = EvalConfig.from_preset("vcc2026")
    names = ["pds_cosine"]
    panel = anchor_semantic_params(cfg, real_ad, names)
    row = anchor_semantic_params(
        replace(cfg, discrimination=replace(cfg.discrimination, exclusion_scope="row")),
        real_ad, names)
    assert panel != row
    assert panel["discrimination.exclusion_scope"] == "panel"
    assert row["discrimination.exclusion_scope"] == "row"
    # and the pre-existing sibling hole the same review turned up
    assert panel["discrimination.tie_policy"] == "midrank"


# ------------------------------------------------------------------ the two kernels agree

@pytest.mark.parametrize("scope", ["row", "panel"])
@pytest.mark.parametrize("metric", ["l1", "l2", "cosine"])
def test_the_gpu_kernel_reproduces_the_cpu_reference_under_both_scopes(scope, metric):
    """#248 was a lookup written independently in the two discrimination kernels and wrong in
    both. `exclusion_scope` is resolved through the same shared helper for exactly that reason,
    and the xp kernel run on numpy IS the code cupy runs -- so this is the check that the panel
    branch did not land on one side only."""
    from cell_eval2.gpu.distances import _discrimination_ranks_xp

    genes, labels, ctrl, real, _ = _panel(n_targets=9, n_extra=25, seed=3)
    arm = _antitarget_arm(genes, labels, ctrl)
    kw = dict(control=CTRL, rank_denominator="n-1", tie_policy="midrank",
              exclude_target_gene=True, exclusion_scope=scope, control_source="pred",
              genes=genes)
    cpu = discrimination_score(pred_bulk=(labels, arm), real_bulk=(labels, real),
                               distance=metric, **kw)
    gpu = _discrimination_ranks_xp(np, (labels, real), (labels, arm), metric=metric,
                                   pert_chunk=4, **kw)
    assert cpu == gpu
