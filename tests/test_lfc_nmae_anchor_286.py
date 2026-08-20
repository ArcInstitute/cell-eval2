"""Issue #286 -- `de_lfc_nmae`'s documented 1.0 anchor is exact in LOG2FC space, not in
SUBMISSION space, under `control_source="real"` (the v2 default, and what `vcc2026` scores).

The docs quote the effect on the committed fixture (0.9397-1.0397 across five perturbations,
three of them BELOW 1.0) and on the official val panels. Neither is a test: the first reads a
5 MB h5ad no other test touches, the second needs the panel. This is the mechanism itself,
constructed to isolate it exactly.

⚠️ The mechanism is NOT "heterogeneous library sizes", which is how #286 and §1.2 both phrase
it. With `pi_c = x_c / L_c` the per-cell composition,

    CPM(mean_c x_c) - mean_c CPM(x_c)  =  1e6 * Cov_c(L_c, pi_c) / E[L]

so depth heterogeneity is NECESSARY (no spread, no covariance) but nowhere near SUFFICIENT: a
panel whose cells differ 10x in depth while sharing one composition has a discrepancy of
exactly zero, because every cell's CPM vector is literally the same vector.

The panels below make that the only difference. Both carry the SAME depth multiset and the
SAME composition multiset; only the PAIRING between them changes. Consequences, all verified
by the guard test rather than asserted: identical library-size CV, and -- because a DE mean is
a mean of per-cell CPM, which depends on the composition multiset alone -- the same real-side
DE GATE and the same real log2FCs, hence the same `nmae` denominator. That last one is checked
by running the production DE path on both panels and comparing its output, not by an invariant
on the count matrix. The pred side sees the depth-
weighted MARGINAL mean, which is what the pairing moves.
"""
from __future__ import annotations

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from cell_eval2 import EvalConfig, compute_metrics
from cell_eval2.config import DEParams

G, A = 300, 30            # A cells per stratum, four strata per group
PERT = "P1"
DEPTH_RATIO = 10          # deep cells are EXACTLY 10x, so nothing is ever rounded

_rng = np.random.default_rng(0)
C0 = (_rng.gamma(2.0, 20.0, size=G) + 5.0).round().astype(np.int64)
C1 = np.roll(C0, 137)                    # the SAME count multiset, a different composition
D0 = C0.copy()
D0[100:160] *= 3                         # the real perturbation's effect
D1 = np.roll(D0, 137)


def _group(low, high, *, pairing):
    """4A cells: 2A at depth L and 2A at 10L, 2A carrying `low` and 2A carrying `high`.

    `pairing="independent"` splits each composition evenly across both depths, so
    `Cov(L, pi) = 0` exactly. `"low_deep"` / `"high_deep"` put one composition entirely at the
    deep end, which is the only thing that changes.
    """
    if pairing == "independent":
        blocks = [(low, 1), (high, 1), (low, DEPTH_RATIO), (high, DEPTH_RATIO)]
        reps = A
    elif pairing == "low_deep":
        blocks, reps = [(high, 1), (low, DEPTH_RATIO)], 2 * A
    elif pairing == "high_deep":
        blocks, reps = [(low, 1), (high, DEPTH_RATIO)], 2 * A
    else:  # pragma: no cover - guard against a typo in a parametrize
        raise ValueError(pairing)
    return np.vstack([np.tile(profile * mult, (reps, 1))
                      for profile, mult in blocks]).astype(np.float32)


def _panel(pairing):
    ctrl = _group(C0, C1, pairing=pairing)
    pert = _group(D0, D1, pairing=pairing)
    n = ctrl.shape[0]
    obs = pd.DataFrame({"target": ["non-targeting"] * n + [PERT] * n},
                       index=[f"c{i}" for i in range(2 * n)])
    var = pd.DataFrame(index=[f"g{i}" for i in range(G)])
    real = ad.AnnData(sp.csr_matrix(np.vstack([ctrl, pert])), obs=obs.copy(), var=var.copy())
    # The most honest possible "predict no change": the EXACT unrounded control mean,
    # broadcast to every cell. Privileged information no real contestant has.
    pred_x = np.repeat(ctrl.mean(axis=0, keepdims=True), 2 * n, axis=0).astype(np.float32)
    pred = ad.AnnData(sp.csr_matrix(pred_x), obs=obs.copy(), var=var.copy())
    return pred, real, ctrl, pert


def _cfg(control_source):
    """ONE config, so the guard's DE and the scored DE cannot drift apart. `_real_de` reads
    every knob back off this object rather than restating it -- an earlier version passed
    `filter_gene_min_cpm_cell=None` by hand while this default is 5.0, so the guard was
    checking a DE the metric never ran."""
    return EvalConfig(metrics=["de_wilcoxon_lfc_nmae"], pert_col="target",
                      control="non-targeting", allow_fractional_counts=True,
                      control_source=control_source, device="cpu", num_threads=1,
                      # ⚠️ #172: `de_lfc_nmae` excludes each target's own gene and RAISES when
                      # NO target resolves. `P1` is a LABEL on a synthetic depth/composition
                      # panel, not a gene symbol, and this map says so -- `resolve_target_genes`
                      # treats a map entry as AUTHORITATIVE and does not re-check it against the
                      # feature index, which is exactly the "correctly-named but deliberately
                      # absent gene" case it documents. The target resolves, the anti-join finds
                      # no `(P1, __unmeasured__)` row, and every reading below is unchanged --
                      # which matters here because they are MEASURED constants (1.2010 /
                      # 1.1936), not properties that would survive a re-derivation.
                      target_gene_map={PERT: "__unmeasured__"},
                      de=DEParams(backend="pdex", p_adj_threshold=1.0))


def _nmae(pred, real, control_source):
    cfg = _cfg(control_source)
    rows = [r for r in compute_metrics(pred, real, config=cfg).iter_rows() if r[0] == PERT]
    assert rows, "the perturbation was omitted from the frame"
    return rows[0][2]


def _real_de(real_ad):
    """The REAL side's DE table, through the production path, every knob read off `_cfg` --
    mirroring `run._compute_de_side`. This is what the guard has to compare: a proxy
    invariant on the count matrix cannot pin a rank test's tie handling."""
    from cell_eval2.de_compute import compute_de
    cfg = _cfg("real")
    return compute_de(real_ad, backend=cfg.de.backend, groupby=cfg.pert_col,
                      reference=cfg.control, mean_calc=cfg.de.mean_calc,
                      epsilon=cfg.de.epsilon, input_type=cfg.input_type,
                      target_sum=cfg.target_sum, clip_value=cfg.de.clip_value,
                      filter_gene_min_cpm_cell=cfg.filter.filter_gene_min_cpm_cell,
                      fdr_scope=cfg.de.fdr_scope,
                      threads=cfg.num_threads).sort(["target", "feature"])


@pytest.mark.parametrize("pairing", ["low_deep", "high_deep"])
def test_the_panels_are_matched_on_everything_but_the_pairing(pairing):
    """Guard the guard, and it is the whole basis of the attribution below. Without this the
    correlated panel could differ in depth spread, in composition, or in the real log2FCs, and
    the score difference would prove nothing about covariance.

    ⚠️ The real-side comparison is made on the DE TABLE, not on the count matrix. Earlier
    versions compared per-gene means (which cannot imply equal p-values at all) and then
    per-gene value multisets to a tolerance (which cannot pin Mann-Whitney tie groups: two
    values one ULP apart split a tie the exact values would share, moving the tie correction).
    A rank test's inputs are only pinned by running the rank test."""
    _, real_flat, flat_c, flat_p = _panel("independent")
    _, real_corr, corr_c, corr_p = _panel(pairing)
    for flat, corr in ((flat_c, corr_c), (flat_p, corr_p)):
        # identical DEPTH multiset, so the library-size spread is held fixed...
        assert sorted(flat.sum(axis=1).tolist()) == sorted(corr.sum(axis=1).tolist())
        assert (flat.sum(1).std() / flat.sum(1).mean()) == pytest.approx(
            corr.sum(1).std() / corr.sum(1).mean())
        # ...and it is a real spread, not a degenerate one
        assert flat.sum(1).std() / flat.sum(1).mean() > 0.8

    de_flat, de_corr = _real_de(real_flat), _real_de(real_corr)
    assert de_flat["target"].to_list() == de_corr["target"].to_list()
    assert de_flat["feature"].to_list() == de_corr["feature"].to_list()
    # The p-values come out BIT-IDENTICAL -- the rank test sees the same value multiset in a
    # different row order, and Mann-Whitney is order-invariant -- so assert that rather than
    # the weaker gate membership it implies.
    assert de_flat["p_adj"].to_list() == de_corr["p_adj"].to_list()
    # ⚠️ And the gate that assertion pins is NOT vacuous at `_nmae`'s p_adj_threshold = 1.0:
    # 95 of these 300 rows sit at p_adj >= 1.0 and are excluded, in all three panels alike.
    assert 0 < (de_flat["p_adj"] < 1.0).sum() < de_flat.height
    # The real log2FCs -- the nmae DENOMINATOR -- are not bit-identical, because the group
    # MEAN reduces in a different order. float64 through pdex; measured max relative
    # difference 2.0e-13, so the tolerance below is ~5x that rather than slack.
    np.testing.assert_allclose(de_flat["log2_fold_change"].to_numpy(),
                               de_corr["log2_fold_change"].to_numpy(), rtol=1e-12, atol=1e-13)


@pytest.mark.parametrize("pairing", ["independent", "low_deep", "high_deep"])
def test_anchor_is_exact_under_control_source_pred(pairing):
    """Both sides of the ratio come from the same predicted matrix, so lfc_hat is exactly 0
    and the numerator IS the denominator. This half of the claim is airtight, and holds
    whatever the panel's composition structure."""
    pred, real, _, _ = _panel(pairing)
    assert _nmae(pred, real, "pred") == pytest.approx(1.0, abs=1e-12)


def test_depth_alone_leaves_the_anchor_exact():
    """`Cov(L, pi) = 0` with a 10x depth spread still present: `mean_c CPM = CPM(mean_c)`, so
    the documented anchor survives `control_source="real"` EXACTLY. This is what rules out
    "heterogeneous library sizes" as the mechanism -- the tolerance is float32 storage of the
    count matrix, not a residual effect."""
    pred, real, _, _ = _panel("independent")
    assert _nmae(pred, real, "real") == pytest.approx(1.0, abs=1e-7)


@pytest.mark.parametrize(("pairing", "expected"), [("low_deep", 1.2010), ("high_deep", 1.1936)])
def test_depth_composition_covariance_breaks_the_anchor(pairing, expected):
    """#286. Same depths, same compositions, same library-size CV, same real log2FCs -- only
    the pairing differs, and the exact-control-mean submission stops reading 1.0.

    ⚠️ The direction is NOT asserted as "below 1.0", and both pairings here read ABOVE it.
    Per gene, moving off `lfc_hat = 0` improves the error only when the predicted and true
    log2FCs share a sign AND `|lfc_hat| < 2|lfc_real|`; the aggregate drops below 1 only when
    those improvements outweigh the rest. The artifact here is LARGE (a 25.9% CPM gap) and
    unrelated to the planted effect, so it worsens most genes. The below-1.0 cases in the docs
    are measured on real data -- the committed fixture (3 of 5) and a dispersed context-mean
    arm at 0.9909 -- where the artifact is ~1% and correlates with the true effect.

    The expected values are pinned, not merely bounded, so the figures quoted in the docstring
    and in §4.3 are the ones this fixture actually produces."""
    pred, real, _, _ = _panel(pairing)
    got = _nmae(pred, real, "real")
    assert abs(got - 1.0) > 0.05
    # abs=5e-5, not a relative tolerance: the point is to pin the four decimals the
    # docstring and §4.3 quote, and rel=1e-3 would admit +-0.0012 around 1.20.
    assert got == pytest.approx(expected, rel=0, abs=5e-5)
