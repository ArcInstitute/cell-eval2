"""Issue #172: the perturbed gene's own row leaves the three scored `vcc2026` members that
still passed it through -- `de_wilcoxon_sig_jaccard`, `de_wilcoxon_lfc_nmae` and
`expr_mse_unbiased_capped_norm` (both legs). Ruled by Alex 2026-08-17.

The adversary the issue describes is the same on both sides: **predict nothing (or predict the
control) everywhere, and predict the truth at your own target gene.** That is not a prediction
-- the knockdown is the experiment's premise -- and before this change it was worth 0.02 raw on
`sig_jaccard` and 6-11% of `expr_mse_unbiased_capped_norm`'s [0, 1] range on the official val
contexts. Every test below is written as "the adversary must gain exactly nothing", which is
the property, rather than as a golden value, which is an artifact of the fixture.

Scope is deliberate and is asserted at the bottom: `de_wilcoxon_overlap`,
`de_wilcoxon_precision` and `de_wilcoxon_sig_recall` move under exclusion too (measured
-0.030 / -0.019 / -0.019 in `docs/metrics.md` §4) but are NOT scored by `vcc2026`, so they are
out of scope and must keep passing the row through.
"""
import logging

import numpy as np
import polars as pl
import pytest

from cell_eval2.catalog import CATALOG, PROFILES, resolve_metrics
from cell_eval2.config import EvalConfig
from cell_eval2.de import prepare_de, resolve_target_genes
from cell_eval2.metrics.de import (
    de_lfc_nmae,
    de_overlap,
    de_sig_jaccard,
    de_sig_recall,
)
from cell_eval2.metrics.delta import distance_unbiased, mse_unbiased_capped
from cell_eval2.moments import GroupMoments
from cell_eval2.run import _ontarget_exclusion_used, _result_config_digest

CONTROL = "non-targeting"

# ---------------------------------------------------------------------------------------
# DE half
# ---------------------------------------------------------------------------------------

_OFF = [f"g{i}" for i in range(12)]      # off-target genes, 12 so min_gate_size=10 clears


def _de_frame(targets, *, sig, on_target_sig, lfc=2.0, on_target_lfc=2.0):
    """One row per (target, feature) over `_OFF` + each target's OWN gene.

    `sig` is the set of off-target features called significant; `on_target_sig` says whether
    each target's own gene is called significant. Every target's own gene is PRESENT in the
    frame either way, so the target resolves and the exclusion has something to remove.
    """
    rows = {"target": [], "feature": [], "log2_fold_change": [], "p_adj": []}
    for t in targets:
        for f in _OFF:
            rows["target"].append(t)
            rows["feature"].append(f)
            rows["log2_fold_change"].append(lfc)
            rows["p_adj"].append(0.001 if f in sig else 0.9)
        rows["target"].append(t)
        rows["feature"].append(t)                      # the target's OWN gene
        rows["log2_fold_change"].append(on_target_lfc)
        rows["p_adj"].append(0.001 if on_target_sig else 0.9)
    return pl.DataFrame(rows)


def _prep(real, pred, **kw):
    return prepare_de(pred, real, control=CONTROL, **kw)


def test_sig_jaccard_the_adversary_gains_exactly_nothing():
    """A submission calling ONLY its own target gene significant scores 0.0, not 1/|R∪P|.

    The reference calls the target's own gene plus four off-target genes; the adversary calls
    the target's own gene and nothing else. Pre-#172 that was a guaranteed intersection of 1
    over a union of 5.
    """
    targets = ["ABCA1", "BRCA2"]
    real = _de_frame(targets, sig={"g0", "g1", "g2", "g3"}, on_target_sig=True)
    silent = _de_frame(targets, sig=set(), on_target_sig=False)
    adversary = _de_frame(targets, sig=set(), on_target_sig=True)

    for t in targets:
        assert de_sig_jaccard(_prep(real, adversary))[t] == 0.0
        assert de_sig_jaccard(_prep(real, silent))[t] == 0.0
    # ... and the two submissions are indistinguishable, which is the property.
    assert de_sig_jaccard(_prep(real, adversary)) == de_sig_jaccard(_prep(real, silent))


def test_sig_jaccard_excludes_from_BOTH_sides():
    """Asymmetry would be a bug in either direction: dropping the pair from the reference
    alone leaves an on-target call in |P| as a pure penalty, and dropping it from the
    prediction alone leaves it in |R| as a guaranteed miss. A perfect off-target prediction
    must read 1.0 whatever either side says about the on-target row."""
    targets = ["ABCA1"]
    real = _de_frame(targets, sig={"g0", "g1"}, on_target_sig=True)
    for pred_on_target in (True, False):
        pred = _de_frame(targets, sig={"g0", "g1"}, on_target_sig=pred_on_target)
        assert de_sig_jaccard(_prep(real, pred))["ABCA1"] == 1.0


def test_lfc_nmae_the_adversary_gains_exactly_nothing():
    """The on-target row leaves the GATE, so a prediction that nails it and nothing else is
    worth exactly the same as one predicting no change at all: 1.0."""
    targets = ["ABCA1"]
    # Reference: off-target LFCs of 2.0 and a huge on-target LFC the adversary reproduces.
    real = _de_frame(targets, sig={f for f in _OFF}, on_target_sig=True, on_target_lfc=-8.0)
    silent = _de_frame(targets, sig=set(), on_target_sig=False, lfc=0.0, on_target_lfc=0.0)
    adversary = _de_frame(targets, sig=set(), on_target_sig=False, lfc=0.0,
                          on_target_lfc=-8.0)
    assert de_lfc_nmae(de_pred=silent, de_real=real)["ABCA1"] == 1.0
    assert de_lfc_nmae(de_pred=adversary, de_real=real)["ABCA1"] == 1.0


def test_lfc_nmae_the_gate_shrinks_by_one_per_resolved_target():
    """The gate SIZE is what `min_gate_size` reads, so exclusion has to shrink it. Exactly
    `min_gate_size` off-target genes plus the on-target row: scored at 10, omitted at 11."""
    real = _de_frame(["ABCA1"], sig=set(_OFF[:10]), on_target_sig=True)
    pred = _de_frame(["ABCA1"], sig=set(), on_target_sig=True, lfc=0.0)
    assert de_lfc_nmae(de_pred=pred, de_real=real, min_gate_size=10)["ABCA1"] == 1.0
    assert "ABCA1" not in de_lfc_nmae(de_pred=pred, de_real=real, min_gate_size=11)


@pytest.mark.parametrize("func", [de_sig_jaccard, de_lfc_nmae])
def test_de_zero_resolve_raises(func):
    """#248's failure mode: construct-ID labels with no map resolve to nothing, the anti-join
    removes nothing, and the metric silently keeps its pre-#172 meaning. It must raise."""
    rows = {"target": [], "feature": [], "log2_fold_change": [], "p_adj": []}
    for t in ("ABCA1-1", "BRCA2-1"):
        for f in _OFF:
            rows["target"].append(t)
            rows["feature"].append(f)
            rows["log2_fold_change"].append(2.0)
            rows["p_adj"].append(0.001)
    frame = pl.DataFrame(rows)
    with pytest.raises(ValueError, match="no target resolves to a gene"):
        func(_prep(frame, frame))


@pytest.mark.parametrize("func", [de_sig_jaccard, de_lfc_nmae])
def test_de_target_gene_map_routes_the_exclusion(func):
    """The construct-ID escape hatch: with the map, guide-level labels give the SAME values
    as symbol-labelled ones. Without it the previous test shows they raise."""
    real_sym = _de_frame(["ABCA1"], sig=set(_OFF), on_target_sig=True, on_target_lfc=-8.0)
    pred_sym = _de_frame(["ABCA1"], sig=set(_OFF[:6]), on_target_sig=True, lfc=1.0,
                         on_target_lfc=-8.0)
    # Only the TARGET label becomes a construct ID; `feature` stays a gene symbol, which is
    # exactly the shape that makes raw matching fail and the map necessary.
    ren = {"ABCA1": "ABCA1-1"}
    real_cid = real_sym.with_columns(pl.col("target").replace(ren))
    pred_cid = pred_sym.with_columns(pl.col("target").replace(ren))
    prep_cid = prepare_de(
        pred_cid, real_cid, control=CONTROL,
        target_resolution=resolve_target_genes(
            real_cid, ["ABCA1-1"], target_gene_map={"ABCA1-1": "ABCA1"}),
    )
    assert func(prep_cid)["ABCA1-1"] == pytest.approx(func(_prep(real_sym, pred_sym))["ABCA1"])


def test_de_partial_resolution_excludes_only_the_resolved_targets():
    """A target whose own gene is not measured drops nothing, and that is data rather than an
    error (`de.TargetResolution`). The resolved sibling must still be excluded."""
    # The two targets are built IDENTICALLY -- each calls only its own gene significant, while
    # the reference calls that gene plus g0. ABCA1's own gene is measured (so it resolves and is
    # excluded); ZZZZ's is not in the frame at all (so it resolves to nothing and is kept).
    # Same input shape, different answers: that difference IS the assertion, and it is exactly
    # what a no-op anti-join would erase.
    real_abca1 = _de_frame(["ABCA1"], sig={"g0"}, on_target_sig=True)
    real_zzzz = pl.DataFrame({
        "target": ["ZZZZ"] * (len(_OFF) + 1),
        "feature": list(_OFF) + ["ZZZZ_GENE"],           # its own gene is NOT this label
        "log2_fold_change": [2.0] * (len(_OFF) + 1),
        "p_adj": [0.001 if f == "g0" else 0.9 for f in _OFF] + [0.001],
    })
    pred_abca1 = _de_frame(["ABCA1"], sig=set(), on_target_sig=True)   # only its own gene
    pred_zzzz = pl.DataFrame({
        "target": ["ZZZZ"] * (len(_OFF) + 1),
        "feature": list(_OFF) + ["ZZZZ_GENE"],
        "log2_fold_change": [2.0] * (len(_OFF) + 1),
        "p_adj": [0.9] * len(_OFF) + [0.001],            # only ZZZZ_GENE
    })
    prepared = _prep(pl.concat([real_abca1, real_zzzz]),
                     pl.concat([pred_abca1, pred_zzzz]))
    assert prepared.target_resolution.n_resolved == 1
    assert set(prepared.target_resolution.mapping) == {"ABCA1"}
    out = de_sig_jaccard(prepared)
    # ABCA1 RESOLVES: its own pair leaves both sides -> real {g0} vs pred {} -> 0/1 = 0.0.
    # ZZZZ does NOT: nothing is dropped -> real {g0, ZZZZ_GENE} vs pred {ZZZZ_GENE} -> 1/2.
    # Without the anti-join ABCA1 would read 1/2 as well, and this test would not separate them.
    assert out == {"ABCA1": 0.0, "ZZZZ": 0.5}


def test_de_exclusion_count_is_reported(caplog):
    """#248's tripwire, DE-side analogue of `baseline_meta.json`'s `n_excluded`: a stamped
    "excluded" beside a zero count is the only after-the-fact proof a run excluded nothing."""
    real = _de_frame(["ABCA1", "BRCA2"], sig={"g0"}, on_target_sig=True)
    with caplog.at_level(logging.INFO, logger="cell_eval2.metrics.de"):
        de_sig_jaccard(_prep(real, real))
    rendered = " ".join(r.getMessage() for r in caplog.records)
    assert "2 row(s) removed" in rendered
    assert "2/2 targets resolved" in rendered


@pytest.mark.parametrize(("func", "kw"), [
    (de_overlap, {"k": None, "metric": "overlap"}),
    (de_overlap, {"k": None, "metric": "precision"}),
    (de_sig_recall, {}),
])
def test_the_unscored_set_members_still_pass_the_row_through(func, kw):
    """SCOPE, asserted. These three move under exclusion (`docs/metrics.md` §4) but are not
    `vcc2026` members, and the 2026-08-17 ruling is scoped to the competition six. A
    submission calling ONLY its own target gene must still score above zero here -- if one of
    them starts excluding, this test is the signal that the scope moved."""
    # The on-target row carries the largest |log2FC| so it heads the rank on both sides --
    # `de_overlap` is a TOP-K metric, and with tied effect sizes its top-1 is arbitrary.
    real = _de_frame(["ABCA1"], sig={"g0", "g1", "g2", "g3"}, on_target_sig=True,
                     on_target_lfc=9.0)
    adversary = _de_frame(["ABCA1"], sig=set(), on_target_sig=True, on_target_lfc=9.0)
    assert func(_prep(real, adversary), **kw)["ABCA1"] > 0.0


def test_the_dispatch_will_keep_passing_the_gene_labels():
    """The one silent-regression path the value tests cannot see.

    `run.dispatch_anndata_metrics` builds its kwargs by SIGNATURE FILTERING -- it offers
    `genes` and `target_gene_map` and each metric receives them only if it declares them. So
    deleting either parameter from a leg of `expr_mse_unbiased_capped_norm` does not break any
    call: the metric silently stops excluding and keeps its pre-#172 meaning, exactly the
    failure #172 exists to fix. Assert the contract on the CATALOG's funcs, which is what the
    dispatch actually inspects.
    """
    import inspect

    for name in ("expr_mse_unbiased_capped", "expr_distance_unbiased"):
        params = inspect.signature(CATALOG[name].func).parameters
        assert "genes" in params, name
        assert "target_gene_map" in params, name

    # The scored member itself is derived from exactly those two legs, so both must be present
    # and both must exclude -- see `test_mse_only_the_numerator_excluding_would_break_the_anchor`.
    derived = CATALOG["expr_mse_unbiased_capped_norm"].derived
    assert (derived.numerator, derived.denominator) == (
        "expr_mse_unbiased_capped", "expr_distance_unbiased")
    # `test_competition_rule.py` owns the assertion that these six ARE the scored members; this
    # only pins that the two #172 touched on the DE side are among them.
    assert {"de_wilcoxon_sig_jaccard", "de_wilcoxon_lfc_nmae"} <= set(PROFILES["vcc2026"])


# ---------------------------------------------------------------------------------------
# expression half -- expr_mse_unbiased_capped_norm's two legs
# ---------------------------------------------------------------------------------------

def _panel(n_perts=6, n_genes=40, seed=7, construct_ids=False):
    """Hand-built bulks + moments. The metric reads only `(perts, means)` and GroupMoments,
    so no cells are needed -- and the knocked-down gene is made the largest-moving one, which
    is what the official contexts measured it to be in 57-66% of perturbations."""
    rng = np.random.default_rng(seed)
    genes = np.array([f"g{i}" for i in range(n_genes)])
    labels = [(f"{genes[i]}-1" if construct_ids else str(genes[i])) for i in range(n_perts)]
    perts = np.array(labels + [CONTROL])
    ic = n_perts

    mu_ctrl = rng.uniform(0.5, 4.0, size=n_genes)
    mu_real = np.tile(mu_ctrl, (n_perts + 1, 1))
    for p in range(n_perts):
        mu_real[p] += rng.normal(0.0, 0.2, size=n_genes)
        mu_real[p, p] -= 3.0                     # the knockdown, and the largest mover
    mu_real[ic] = mu_ctrl

    # The control's correction is the SMALLEST, so #247's cap does not bind for a submission
    # emitting the control -- the one condition the 1.0 anchor needs (`mse_unbiased_capped`).
    jk = rng.uniform(0.04, 0.08, size=n_perts + 1)
    jk[ic] = 0.01
    mom = GroupMoments(perts=perts, counts=np.full(n_perts + 1, 500.0),
                       sumsq=np.zeros(n_perts + 1), jk=jk)
    return perts, genes, mu_ctrl, mu_real, mom, ic


def _ratio(perts, mu_pred, pred_mom, mu_real, real_mom, **extra):
    """The derived member: `ratio_of_sums` over its numerator and denominator legs."""
    kw = dict(comparator="bulk_lognorm", pert_col="perturbation", control=CONTROL,
              real_bulk=(perts, mu_real), real_moments=real_mom, **extra)
    num = mse_unbiased_capped(pred_bulk=(perts, mu_pred), pred_moments=pred_mom, **kw)
    den = distance_unbiased(**kw)
    return sum(num.values()) / sum(den.values()), num, den


def test_mse_the_adversary_gains_exactly_nothing():
    """THE #172 exploit, on the expression side: predict the control everywhere except your
    own target gene, where you predict the truth. Measured at 10.21% / 11.30% / 6.07% of this
    member's range on the three official val contexts. Post-exclusion the adversary and the
    plain predict-the-control arm are the SAME submission, so the gift must be exactly 0."""
    perts, genes, mu_ctrl, mu_real, mom, ic = _panel()
    control_arm = np.tile(mu_ctrl, (perts.size, 1))
    adversary = control_arm.copy()
    for p in range(perts.size - 1):
        adversary[p, p] = mu_real[p, p]
    pred_mom = GroupMoments(perts=perts, counts=np.full(perts.size, 500.0),
                            sumsq=np.zeros(perts.size), jk=np.full(perts.size, mom.jk[ic]))

    without, _, _ = _ratio(perts, control_arm, pred_mom, mu_real, mom, genes=genes)
    with_gift, _, _ = _ratio(perts, adversary, pred_mom, mu_real, mom, genes=genes)
    assert with_gift == pytest.approx(without, abs=1e-12)

    # ... and the gift was real before the exclusion, so this test can fail.
    pre_without, _, _ = _ratio(perts, control_arm, pred_mom, mu_real, mom)
    pre_with, _, _ = _ratio(perts, adversary, pred_mom, mu_real, mom)
    assert pre_without - pre_with > 0.1


def test_mse_the_1_0_anchor_is_exact_under_exclusion(monkeypatch):
    """The `1.0 = predicted the control` anchor is what BOTH legs dropping the same gene buys. A
    submission emitting the real control's own cells has mu_pred = mu_ctrl and C_pred =
    C_ctrl, so numerator and denominator must agree PER PERTURBATION, bit for bit.

    ⚠️ The arm below is the tile of ONE control profile, so it also has zero
    across-perturbation spread -- which is what #348's bound reads. That is not incidental: the
    exact identity needs `mu_pred = mu_ctrl` on every row, and any arm satisfying it exactly has
    no across-perturbation variation to show. #348 therefore withholds this arm's correction, on
    the ground that a reused control cell block and a pinned aggregate are indistinguishable to
    an estimator seeing one submission (`tests/test_pred_correction_bound_348.py`). #172's
    identity is what is under test here, so the bound is disabled for it and asserted separately
    below -- the two are orthogonal and both are exact.
    """
    perts, genes, mu_ctrl, mu_real, mom, ic = _panel()
    control_arm = np.tile(mu_ctrl, (perts.size, 1))
    pred_mom = GroupMoments(perts=perts, counts=np.full(perts.size, 500.0),
                            sumsq=np.zeros(perts.size), jk=np.full(perts.size, mom.jk[ic]))
    with monkeypatch.context() as m:
        m.setattr("cell_eval2.metrics.delta._across_pert_budget", lambda *_a, **_k: float("inf"))
        value, num, den = _ratio(perts, control_arm, pred_mom, mu_real, mom, genes=genes)
    assert value == 1.0
    assert max(abs(num[p] - den[p]) for p in den) == 0.0

    # With the bound live the whole correction is withheld -- exactly, and nothing else moves.
    _, num_bounded, _ = _ratio(perts, control_arm, pred_mom, mu_real, mom, genes=genes)
    zero_mom = GroupMoments(perts=perts, counts=pred_mom.counts, sumsq=pred_mom.sumsq,
                            jk=np.zeros(perts.size))
    _, num_uncorrected, _ = _ratio(perts, control_arm, zero_mom, mu_real, mom, genes=genes)
    assert num_bounded == num_uncorrected


def test_mse_only_the_numerator_excluding_would_break_the_anchor():
    """The reason `distance_unbiased` had to change too. With the denominator alone keeping
    the gene, a submission that predicted the control EXACTLY reads below 1.0 for free."""
    perts, genes, mu_ctrl, mu_real, mom, ic = _panel()
    control_arm = np.tile(mu_ctrl, (perts.size, 1))
    pred_mom = GroupMoments(perts=perts, counts=np.full(perts.size, 500.0),
                            sumsq=np.zeros(perts.size), jk=np.full(perts.size, mom.jk[ic]))
    kw = dict(comparator="bulk_lognorm", pert_col="perturbation", control=CONTROL,
              real_bulk=(perts, mu_real), real_moments=mom)
    num = mse_unbiased_capped(pred_bulk=(perts, control_arm), pred_moments=pred_mom,
                              genes=genes, **kw)
    den_unexcluded = distance_unbiased(**kw)          # genes=None -> keeps the gene
    mixed = sum(num.values()) / sum(den_unexcluded.values())
    assert mixed < 0.95, mixed


def test_mse_divides_by_the_genes_it_actually_summed():
    """Gene-averaged means averaged over the genes in the sum. With one gene excluded the
    divisor is G-1, which is checked here against a hand-computed value."""
    perts, genes, mu_ctrl, mu_real, mom, ic = _panel(n_perts=2, n_genes=5)
    got = distance_unbiased(comparator="bulk_lognorm", pert_col="perturbation",
                            control=CONTROL, real_bulk=(perts, mu_real),
                            real_moments=mom, genes=genes)
    for p in range(2):
        d = mu_real[p] - mu_real[ic]
        keep = np.ones(5, dtype=bool)
        keep[p] = False                              # perturbation p targets gene p
        expected = (float(d[keep] @ d[keep]) - mom.jk[p] - mom.jk[ic]) / 4
        assert got[str(perts[p])] == pytest.approx(expected, rel=1e-12)


def test_mse_zero_resolve_raises_and_names_the_metric():
    """#248's tripwire on the expression side. The message must not tell the caller to unset
    `exclude_target_gene` -- these two metrics have no such knob."""
    perts, genes, mu_ctrl, mu_real, mom, ic = _panel(construct_ids=True)
    with pytest.raises(ValueError, match="NO perturbation resolves") as exc:
        distance_unbiased(comparator="bulk_lognorm", pert_col="perturbation",
                          control=CONTROL, real_bulk=(perts, mu_real),
                          real_moments=mom, genes=genes)
    assert "expr_distance_unbiased" in str(exc.value)
    assert "exclude_target_gene=False" not in str(exc.value)


def test_mse_target_gene_map_routes_the_exclusion():
    """Construct-ID labels + the map give the same values as symbol-labelled ones."""
    perts_c, genes, _, mu_real, mom_c, _ = _panel(construct_ids=True)
    perts_s, _, _, mu_real_s, mom_s, _ = _panel(construct_ids=False)
    np.testing.assert_array_equal(mu_real, mu_real_s)     # same panel, different labels
    tgm = {str(p): str(p)[:-2] for p in perts_c if p != CONTROL}
    mapped = distance_unbiased(comparator="bulk_lognorm", pert_col="perturbation",
                               control=CONTROL, real_bulk=(perts_c, mu_real),
                               real_moments=mom_c, genes=genes, target_gene_map=tgm)
    plain = distance_unbiased(comparator="bulk_lognorm", pert_col="perturbation",
                              control=CONTROL, real_bulk=(perts_s, mu_real_s),
                              real_moments=mom_s, genes=genes)
    for p in perts_c:
        if p != CONTROL:
            assert mapped[str(p)] == pytest.approx(plain[str(p)[:-2]], rel=1e-15)


def test_mse_no_gene_labels_warns_and_does_not_exclude(caplog):
    """`genes=None` is the one route that silently excludes nothing, so it warns. Every
    production driver passes the var index; a direct caller may not have it."""
    perts, genes, _, mu_real, mom, _ = _panel()
    with caplog.at_level(logging.WARNING, logger="cell_eval2.metrics.delta"):
        without = distance_unbiased(comparator="bulk_lognorm", pert_col="perturbation",
                                    control=CONTROL, real_bulk=(perts, mu_real),
                                    real_moments=mom)
    assert "genes=None" in " ".join(r.getMessage() for r in caplog.records)
    with_genes = distance_unbiased(comparator="bulk_lognorm", pert_col="perturbation",
                                   control=CONTROL, real_bulk=(perts, mu_real),
                                   real_moments=mom, genes=genes)
    # Excluding the largest-moving gene has to LOWER the summed distance on every row.
    assert all(with_genes[p] < without[p] for p in with_genes)


# ---------------------------------------------------------------------------------------
# result-cache invalidation -- the silent half
# ---------------------------------------------------------------------------------------

def _cache_cfg(**kw):
    return EvalConfig(metrics=["expr_mae"], version="v2", device="cpu", **kw)


def test_a_pre_172_warm_cache_cannot_be_served_to_the_excluding_code():
    """THE REGRESSION TEST for the invisible half of this change.

    The result cache keys on (inputs + config), and `cell_eval2_version` is deliberately NOT in
    that key -- neither `cache.py` nor `EvalConfig` carries it. So a pre-#172 run at the same
    version reproduces the key exactly and its cached per-perturbation frame, now known to sum a
    gene the metric no longer scores, would be served in preference to recomputing. Identical
    config and identical inputs here; the only difference is whether the run is one #172 could
    have moved.
    """
    cfg = _cache_cfg()
    assert _result_config_digest(
        cfg, de_backend_used=False, comparator="bulk_lognorm", ontarget_exclusion_used=True,
    ) != _result_config_digest(
        cfg, de_backend_used=False, comparator="bulk_lognorm", ontarget_exclusion_used=False,
    )


def test_the_semantics_term_is_scoped_to_the_metrics_172_moved():
    """Scoped by metric FUNC, so both backend families are covered by one entry and an
    unaffected run keeps its warm cache rather than cold-starting every cache in the project."""
    assert _ontarget_exclusion_used(["de_wilcoxon_sig_jaccard"]) is True
    assert _ontarget_exclusion_used(["de_deseq2_lfc_nmae"]) is True          # same func
    assert _ontarget_exclusion_used(["expr_mse_unbiased_capped"]) is True
    assert _ontarget_exclusion_used(["expr_distance_unbiased"]) is True
    assert _ontarget_exclusion_used(["expr_mse_unbiased"]) is True           # shares _numerator
    # Nothing #172 touched -> the run keeps its cache.
    assert _ontarget_exclusion_used(["expr_mae", "expr_mse", "pds_cosine"]) is False
    assert _ontarget_exclusion_used([]) is False


def test_the_derived_member_is_covered_through_its_components():
    """`expr_mse_unbiased_capped_norm` has no func of its own, and needs none: the catalog
    requires a derived metric's components to sit in every profile it claims, so a `vcc2026`
    run carries both legs in `names` and takes the new key through them."""
    names = resolve_metrics("vcc2026", version="v2")[0]
    assert "expr_mse_unbiased_capped" in names and "expr_distance_unbiased" in names
    assert _ontarget_exclusion_used(names) is True


def test_the_semantics_term_is_actually_WIRED_into_the_driver(monkeypatch, tmp_path):
    """The tests above call `_result_config_digest` directly, so deleting the
    `ontarget_exclusion_used=` argument at the driver's call site would leave every one of them
    green while warm pre-#172 caches were served again. This one goes through
    `compute_metrics`, and separates the two halves: bumping the semantics constant must move
    the result fingerprint for a #172 metric and must NOT move it for one #172 never touched.
    """
    import cell_eval2.run as run_mod

    seen: list[str] = []
    real_fp = run_mod.result_fingerprint

    def spy(**kw):
        seen.append(kw["config_digest"])
        return real_fp(**kw)

    monkeypatch.setattr(run_mod, "result_fingerprint", spy)
    perts, genes, _, mu_real, mom, _ = _panel(n_perts=3, n_genes=12)

    def digests(metrics):
        seen.clear()
        ad_obj = _tiny_adata(perts, genes)
        # `cache_pred` is what makes the driver compute a result fingerprint at all
        # (`run.py`: `pred_store = CacheStore(cfg.cache_pred) if cfg.cache_pred else None`),
        # and the key this test is about only exists on that path.
        cfg = EvalConfig(metrics=metrics, pert_col="perturbation", control=CONTROL,
                         input_type="counts", device="cpu",
                         cache_pred=str(tmp_path / "cache"))
        try:
            run_mod.compute_metrics(ad_obj, ad_obj, config=cfg)
        except Exception:
            pass                       # the digest is computed before any metric runs
        return list(seen)

    for metrics, must_move in ((["expr_mse_unbiased_capped"], True),
                               (["expr_mae"], False)):
        before = digests(metrics)
        monkeypatch.setattr(run_mod, "_ONTARGET_EXCLUSION_SEMANTICS",
                            run_mod._ONTARGET_EXCLUSION_SEMANTICS + 1)
        after = digests(metrics)
        monkeypatch.setattr(run_mod, "_ONTARGET_EXCLUSION_SEMANTICS",
                            run_mod._ONTARGET_EXCLUSION_SEMANTICS - 1)
        assert before and after, f"no fingerprint was computed for {metrics}"
        moved = before != after
        assert moved is must_move, (
            f"{metrics}: semantics bump {'did not move' if must_move else 'moved'} the result "
            f"cache key; the driver wiring is wrong"
        )


def _tiny_adata(perts, genes):
    """A minimal counts AnnData whose targets name measured genes, for driver-level tests."""
    import anndata as ad
    import pandas as pd

    rng = np.random.default_rng(3)
    labels = np.repeat(perts, 6)
    X = rng.poisson(4.0, size=(labels.size, genes.size)).astype(np.float32)
    return ad.AnnData(
        X=X,
        obs=pd.DataFrame({"perturbation": labels},
                         index=[f"c{i}" for i in range(labels.size)]),
        var=pd.DataFrame(index=list(genes)),
    )


def test_an_unaffected_run_keeps_its_warm_cache_end_to_end():
    """The scoping half, through the same helper the driver calls."""
    cfg = _cache_cfg()
    unaffected = ["expr_mae"]
    assert _result_config_digest(
        cfg, de_backend_used=False, comparator="bulk_lognorm",
        ontarget_exclusion_used=_ontarget_exclusion_used(unaffected),
    ) == _result_config_digest(
        cfg, de_backend_used=False, comparator="bulk_lognorm",
    )


def test_mse_a_single_gene_panel_gives_nan_not_a_divide_error():
    """G == 1 AND that gene being the perturbation's own leaves an EMPTY scored gene set.
    NaN is `safe_mse`'s empty-input contract; an inf or a ZeroDivisionError would not be."""
    perts = np.array(["g0", CONTROL])
    mu = np.array([[2.0], [1.0]])
    mom = GroupMoments(perts=perts, counts=np.array([500.0, 500.0]),
                       sumsq=np.zeros(2), jk=np.array([0.01, 0.01]))
    with np.errstate(all="raise"):
        got = distance_unbiased(comparator="bulk_lognorm", pert_col="perturbation",
                                control=CONTROL, real_bulk=(perts, mu),
                                real_moments=mom, genes=np.array(["g0"]))
    assert np.isnan(got["g0"])


# --- the anchor's identity ---------------------------------------------------------------
#
# `anchor_semantic_params` is the SAME subset the cache key and `validate_anchor` both use, so
# a hole here is two holes: a false cache hit AND a supplied artifact that validates against a
# run it does not belong to. #172 opened one on each side -- `target_gene_map` was gated on
# "pds_* or DE selected", and neither fires for an EXPRESSION-only anchor whose legs now
# resolve through that map.

_ANCHOR_KW = dict(metrics=["expr_mae"], pert_col="target", input_type="lognorm",
                  validate_input=False)


def _anchor_params(real, **kw):
    from cell_eval2.anchor import anchor_semantic_params
    from cell_eval2.catalog import resolve_metrics as _resolve
    from cell_eval2.run import _resolve_config

    cfg = _resolve_config(EvalConfig(**{**_ANCHOR_KW, **kw}), {})
    names = list(_resolve(cfg.metrics, version=cfg.version)[0])
    return anchor_semantic_params(cfg, real, names)


#: An ACTIVE override, not merely a different serialization. The fixture's perturbations are
#: `GENE1`-`GENE3` and its var index is `GENE1, GENE2, GENE3, g0, ...`, so every target already
#: resolves to its own gene by name; this REDIRECTS GENE1's exclusion to a different column, so
#: the two configs genuinely score different gene sets rather than differing in map bytes alone.
#: A map keyed on a label the panel does not carry (`{"t": ...}`) would move the digest without
#: moving a number, which proves nothing about the gate (codex checkpoint-2 round 2 P1).
_ACTIVE_MAP = {"GENE1": "g0"}


def _identity(real, **kw):
    from cell_eval2.anchor import semantic_identity
    from cell_eval2.catalog import resolve_metrics as _resolve
    from cell_eval2.run import _resolve_config

    cfg = _resolve_config(EvalConfig(**{**_ANCHOR_KW, **kw}), {})
    return semantic_identity(cfg, real, list(_resolve(cfg.metrics, version=cfg.version)[0]))


def test_the_anchor_key_carries_the_map_for_an_EXPRESSION_only_selection(
        synthetic_pair_with_effect):
    """The gap #172 opened: `expr_mse_unbiased_capped`/`expr_distance_unbiased` resolve each
    perturbation's own gene through `cfg.target_gene_map`, and an anchor over them selects
    neither a pds_* nor a DE metric -- so the old `is_pds or is_de` gate left the map out of
    the key entirely.

    Requests the DERIVED name rather than its two components, which is the path a real caller
    takes: it also proves `_ontarget_exclusion_used` is reached through `resolve_metrics`'
    component expansion at THIS call site and not only at run.py's."""
    _pred, real = synthetic_pair_with_effect
    derived = ["expr_mse_unbiased_capped_norm"]
    assert "target_gene_map" in _anchor_params(real, metrics=derived)
    assert _identity(real, metrics=derived) != \
        _identity(real, metrics=derived, target_gene_map=_ACTIVE_MAP), (
        "a target_gene_map change left an expression-only anchor's identity standing still"
    )


def test_the_map_stays_OUT_of_an_anchor_no_excluding_metric_reaches(
        synthetic_pair_with_effect):
    """The gate must not widen to unconditional: keying an `expr_mae`/`expr_mse` anchor on a
    map no selected metric reads would reject a valid artifact for a change that provably
    cannot move one of its numbers. This is the same principle the DE and pds_* blocks
    already follow.

    The identity assertion is what gives this teeth -- absent keys alone would still pass if
    the map leaked in under a different name."""
    _pred, real = synthetic_pair_with_effect
    plain = ["expr_mae", "expr_mse"]
    params = _anchor_params(real, metrics=plain)
    assert "target_gene_map" not in params
    assert "ontarget_exclusion_semantics" not in params
    assert _identity(real, metrics=plain) == \
        _identity(real, metrics=plain, target_gene_map=_ACTIVE_MAP), (
        "a map no selected metric reads moved an anchor's identity"
    )


def test_the_anchor_key_carries_the_172_semantics_term(synthetic_pair_with_effect):
    """`anchor_cache_params` stamps `cell_eval2_version`, and it is not a substitute: #172
    lands WITHIN 0.13.0, so a warm anchor built before it carries the same version string as
    a run after it. Same reasoning as the result cache's term, and the two must carry the
    same value -- bumping one alone would silently half-invalidate."""
    from cell_eval2.run import _ONTARGET_EXCLUSION_SEMANTICS

    _pred, real = synthetic_pair_with_effect
    params = _anchor_params(real, metrics=["expr_mse_unbiased_capped_norm"])
    assert params["ontarget_exclusion_semantics"] == _ONTARGET_EXCLUSION_SEMANTICS


def test_the_anchor_semantics_term_reaches_the_CACHE_params(synthetic_pair_with_effect):
    """`anchor_semantic_params` is only half the story -- the cache door reads
    `anchor_cache_params`. `test_anchor_artifact.py` pins "every key the semantic subset
    returns is in the cache params"; this asserts it for THIS key rather than trusting that
    invariant stays true."""
    from cell_eval2.anchor import anchor_cache_params
    from cell_eval2.catalog import resolve_metrics as _resolve
    from cell_eval2.run import _resolve_config, metric_output_names

    _pred, real = synthetic_pair_with_effect
    cfg = _resolve_config(EvalConfig(**{**_ANCHOR_KW,
                                        "metrics": ["expr_mse_unbiased_capped_norm"]}), {})
    names = list(_resolve(cfg.metrics, version=cfg.version)[0])
    params = anchor_cache_params(cfg, real, names, base_seed=0, n_splits=2,
                                 metrics=metric_output_names(cfg))
    assert "ontarget_exclusion_semantics" in params
    assert "target_gene_map" in params


# --- the partial sidecar's semantics payload ---------------------------------------------
#
# #307 landed `partition.result_semantics` (#246) while this branch was in review, so #172's
# counter belongs in its payload rather than in a fifth cross-partial field: the four fields
# `aggregate_partials` compares -- real_ref_fingerprint, config_hash, comparator, metrics -- all
# describe what was ASKED FOR, and two shards straddling this change agree on every one.


def test_the_partial_payload_carries_the_172_counter():
    """Alongside its two siblings, and UNCONDITIONALLY -- the file's own rule, since a partial is
    a transient intermediate with no warm-cache cost to protect."""
    from cell_eval2.partition import result_semantics
    from cell_eval2.run import _ONTARGET_EXCLUSION_SEMANTICS

    for metrics in (["expr_mse_unbiased_capped_norm"], ["expr_mae"], []):
        sem = result_semantics(metrics, comparator="bulk_lognorm")
        assert sem["ontarget_exclusion_semantics"] == _ONTARGET_EXCLUSION_SEMANTICS, metrics
        assert "de_rank_semantics" in sem and "pds_exclusion_semantics" in sem


def test_a_pre_172_partial_cannot_be_aggregated_with_a_post_172_one(monkeypatch):
    """The property, through the guard rather than through the payload: a sidecar declaring the
    OLD payload must be refused, and the message must NAME the term that moved.

    `_check_result_semantics` validates each declared payload against what THIS build would
    produce, so a stale sidecar is caught even when every sidecar in the directory agrees with
    every other -- which is the hole #307's codex round found and the reason this test asserts
    through the guard."""
    from cell_eval2 import partition as part

    metrics = ["expr_mse_unbiased_capped_norm"]
    current = part.result_semantics(metrics, comparator="bulk_lognorm")
    stale = {**current, "ontarget_exclusion_semantics": 0}      # what a pre-#172 build wrote
    with pytest.raises(ValueError, match="ontarget_exclusion_semantics"):
        part._check_result_semantics(
            "/tmp/does-not-matter", {}, [], n_sidecars=1,
            declared=[("s0.json", metrics, "bulk_lognorm", stale)],
        )


def test_the_schema_counter_was_bumped_for_the_new_term():
    """The file's rule: "Bump when the RESULT SEMANTICS payload gains, loses or re-spells a
    term." Round 1 of #307's own review caught an unbumped addition, so this pins it -- the
    schema is IN the payload, which is what makes every pre-bump partial compare unequal."""
    from cell_eval2.partition import _PARTIAL_SEMANTICS_SCHEMA, result_semantics

    assert _PARTIAL_SEMANTICS_SCHEMA >= 3
    assert result_semantics(["expr_mae"],
                            comparator="bulk_lognorm")["schema"] == _PARTIAL_SEMANTICS_SCHEMA


# --- duplicate gene labels ----------------------------------------------------------------
#
# `resolve_exclusion_columns` keys its gene->column lookup on the LABEL, so a duplicated label
# keeps only the LAST column. #172 added a fourth caller to that resolver and it shipped without
# the uniqueness check its three siblings (`metrics.discrimination`, `gpu.distances`, `baseline`)
# already carried -- so the exclusion dropped the WRONG coordinate, silently. Duplicate symbols
# are ordinary in real var indices; `AnnData.var_names_make_unique()` exists for that reason.
# Found by Copilot on PR #316, in the SUPPRESSED block.


def _dup_panel():
    """A 4-gene panel whose target label appears TWICE, at columns 0 and 2, with very different
    distances so the choice of column is visible in the value."""
    perts = np.array(["GENE1", CONTROL])
    genes = np.array(["GENE1", "g1", "GENE1", "g3"])
    mu = np.array([[10.0, 1.0, 1.0, 1.0], [0.0, 0.0, 0.0, 0.0]])
    mom = GroupMoments(perts=perts, counts=np.array([500.0, 500.0]),
                       sumsq=np.zeros(2), jk=np.zeros(2))
    return perts, genes, mu, mom


def test_a_duplicated_gene_label_RAISES_rather_than_excluding_the_wrong_column():
    """The regression test. Before the guard this returned 34.0 -- the value you get by dropping
    the DUPLICATE at column 2 -- where dropping the real target at column 0 gives 1.0. A 34x
    error on a scored member, with no error and no log line."""
    perts, genes, mu, mom = _dup_panel()
    with pytest.raises(ValueError, match="duplicate gene names"):
        distance_unbiased(comparator="bulk_lognorm", pert_col="perturbation", control=CONTROL,
                          real_bulk=(perts, mu), real_moments=mom, genes=genes)


def test_the_duplicate_guard_names_the_count_and_the_remedy():
    """An error a user cannot act on is barely better than a wrong number: it must say HOW MANY
    collided and what to do, matching the three sibling guards' wording.

    ⚠️ This asserts the RESOLVER's message, because the caller-side copy of this check was
    measured and removed (Gemini, PR #316: 8.6 ms at G = 20000, against a resolver check that is
    free). `who` is what carries the metric name through, so the operand assertion below is also
    what proves the resolver-only arrangement did not make the error anonymous."""
    perts, genes, mu, mom = _dup_panel()
    with pytest.raises(ValueError) as e:
        distance_unbiased(comparator="bulk_lognorm", pert_col="perturbation", control=CONTROL,
                          real_bulk=(perts, mu), real_moments=mom, genes=genes)
    msg = str(e.value)
    assert "1 duplicate gene name" in msg          # 4 labels, 3 distinct
    assert "var_names_make_unique" in msg
    assert "expr_distance_unbiased" in msg         # names the operand, not just "a metric"


def test_a_unique_panel_is_unaffected_by_the_guard():
    """The guard must not fire on the ordinary case -- and the value must still be the one that
    drops the REAL target column, which is what makes this test discriminating rather than a
    smoke test."""
    perts = np.array(["GENE1", CONTROL])
    genes = np.array(["GENE1", "g1", "g2", "g3"])
    mu = np.array([[10.0, 1.0, 1.0, 1.0], [0.0, 0.0, 0.0, 0.0]])
    mom = GroupMoments(perts=perts, counts=np.array([500.0, 500.0]),
                       sumsq=np.zeros(2), jk=np.zeros(2))
    got = distance_unbiased(comparator="bulk_lognorm", pert_col="perturbation", control=CONTROL,
                            real_bulk=(perts, mu), real_moments=mom, genes=genes)
    # sum of squares 103; drop the target's 100 -> 3 over the remaining 3 genes.
    assert got["GENE1"] == pytest.approx(1.0)


def test_the_resolver_backstops_a_caller_that_forgets():
    """The contract "callers validate `genes` first" was forgettable once, so the resolver now
    checks too. Asserted directly, because a caller-side guard alone would leave the FIFTH
    caller free to repeat #172's mistake."""
    from cell_eval2.distances import resolve_exclusion_columns

    with pytest.raises(ValueError, match="duplicate gene names"):
        resolve_exclusion_columns(["GENE1"], np.array(["GENE1", "g1", "GENE1"]))


def test_a_gene_label_count_that_does_not_match_the_means_RAISES():
    """The other half of the resolver's input contract. The resolver backstops UNIQUENESS but
    cannot check the DIMENSION -- it never sees the means. A short `genes` maps only a prefix of
    the coordinates, so a target whose own gene lies past the end resolves to nothing while the
    global gate still passes on the targets that did resolve, and that row silently keeps its own
    gene in the sum: warned-but-wrong rather than an error (codex round 3)."""
    perts = np.array(["GENE1", CONTROL])
    mu = np.array([[10.0, 1.0, 1.0, 1.0], [0.0, 0.0, 0.0, 0.0]])       # 4 coordinates
    mom = GroupMoments(perts=perts, counts=np.array([500.0, 500.0]),
                       sumsq=np.zeros(2), jk=np.zeros(2))
    for genes in (np.array(["GENE1", "g1", "g2"]),                     # short
                  np.array(["GENE1", "g1", "g2", "g3", "g4"])):        # long
        with pytest.raises(ValueError, match="label|coordinate"):
            distance_unbiased(comparator="bulk_lognorm", pert_col="perturbation",
                              control=CONTROL, real_bulk=(perts, mu), real_moments=mom,
                              genes=genes)


def test_the_dimension_guard_does_not_fire_on_the_matched_case():
    """And the matched case still drops the REAL target column, so this pins the value rather
    than only the absence of a raise."""
    perts = np.array(["GENE1", CONTROL])
    genes = np.array(["GENE1", "g1", "g2", "g3"])
    mu = np.array([[10.0, 1.0, 1.0, 1.0], [0.0, 0.0, 0.0, 0.0]])
    mom = GroupMoments(perts=perts, counts=np.array([500.0, 500.0]),
                       sumsq=np.zeros(2), jk=np.zeros(2))
    got = distance_unbiased(comparator="bulk_lognorm", pert_col="perturbation", control=CONTROL,
                            real_bulk=(perts, mu), real_moments=mom, genes=genes)
    assert got["GENE1"] == pytest.approx(1.0)


def test_an_ALL_legacy_partial_directory_still_only_WARNS(caplog):
    """Pins #246's deliberate case-3 leniency, and records exactly how far #172 is closed.

    `_check_result_semantics` raises when declared payloads disagree (case 1) and when declared
    and undeclared are MIXED (case 2), but only warns when NONE declares (case 3) -- because an
    all-legacy directory predates the key entirely and refusing it would break every warm
    partial directory to protect against a mix that is usually not present.

    ⚠️ #172 narrows "usually" a little, and this test is where that is written down. A shard
    lacks the key only if written by a build predating #307. Post-#172 semantics exist only on
    this branch, and only on it BEFORE the merge that brought #307 in -- so the one directory
    that can straddle #172 undetected is `pre-#307 main` + `this branch pre-merge`. Once this
    lands, every build carries both, so every new shard declares and any mix with a legacy shard
    hits case 2 and raises. Tightening case 3 further is a change to #246's ruling, not to #172
    ⚠️ **RULED (Alex, 2026-08-17): leave case 3 as #246 set it — warn, do not raise.** codex
    round 3 rated this release-blocking; the ruling stands against that on the reachability above.
    Both alternatives were on the table and both were declined: rejecting all-legacy directories
    outright breaks every warm partial directory, and rejecting them only when the metric set
    includes an excluding member still breaks them for the competition profile, which is the
    common case. So this test is the RECORD of a decision, not a placeholder for one — if it
    starts failing, someone changed #246's policy and should say so here."""
    from cell_eval2 import partition as part

    with caplog.at_level(logging.WARNING):
        part._check_result_semantics("/tmp/does-not-matter", {}, ["s0.json", "s1.json"],
                                     n_sidecars=2, declared=[])
    assert any("semantics" in r.message.lower() or "semantics" in str(r.msg).lower()
               for r in caplog.records), "an all-legacy directory must at least WARN"
