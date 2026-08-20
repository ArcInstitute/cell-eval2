import anndata as ad
import numpy as np
import pandas as pd
import pytest

PERTS = ("non-targeting", "GENE1", "GENE2", "GENE3")


def _var_names(n_genes):
    """Var index whose leading labels ARE the non-control perturbation labels.

    PERTS are gene knockdowns, so their own genes must be MEASURED -- that is what a real
    panel looks like, and since #248 `exclude_target_gene=True` (the default) raises on a
    panel where NO perturbation resolves to a gene, rather than scoring with nothing
    excluded. Mirrors what the DE-side zero-resolve gate already forced on its fixtures
    (tests/test_run.py::test_de_profile_emits_all_new_metrics sets `feats = perts`).

    Total gene count is unchanged, so feature-dimension expectations hold -- but the `g{i}`
    labels are RENUMBERED, not replaced in place: the index is
    `GENE1, GENE2, GENE3, g0 .. g{n_genes-4}`, so `g0` now sits at column 3 and the last
    three `g{i}` labels no longer exist.

    That is safe today because **no consumer of this fixture reaches a gene POSITIONALLY**.
    Consumers do name `g0`-`g2` -- e.g. the supplied-DE tables in `test_cache.py` and
    `test_baseline_build.py` -- but by NAME, and those names still resolve since the labels
    still exist, merely at different columns. Nothing names `g37`-`g39`, which no longer do.
    A new test must not assume `g{i}` is at column `i` here.
    """
    named = [p for p in PERTS if p != "non-targeting"]
    return named + [f"g{j}" for j in range(n_genes - len(named))]


def _build(rng, scale, n_genes, n_cells_per):
    blocks, labels = [], []
    for p in PERTS:
        counts = rng.gamma(shape=1.0, scale=scale, size=(n_cells_per, n_genes))
        blocks.append(np.log1p(counts))  # lognorm-like: fractional, >=0, small
        labels += [p] * n_cells_per
    X = np.vstack(blocks).astype(np.float64)
    obs = pd.DataFrame({"target": labels},
                       index=[f"c{i}" for i in range(X.shape[0])])
    var = pd.DataFrame(index=_var_names(n_genes))
    return ad.AnnData(X=X, obs=obs, var=var)


@pytest.fixture
def synthetic_pair():
    """Deterministic (pred, real) AnnData pair, log-normalized, pert_col='target'."""
    rng = np.random.default_rng(0)
    real = _build(rng, scale=1.0, n_genes=40, n_cells_per=25)
    pred = _build(rng, scale=1.05, n_genes=40, n_cells_per=25)
    return pred, real


@pytest.fixture
def synthetic_pair_with_effect():
    """`synthetic_pair` with a REAL perturbation effect on both sides.

    `_build` draws every group from one distribution, so a panel built from it has no
    measurable aggregate effect: `sum(expr_distance_unbiased)` is non-positive there and
    `run._derived_value` refuses it (#257) -- a property of the reference, by design. Any
    test that runs a profile carrying `expr_mse_unbiased_capped_norm` needs a reference whose
    perturbations actually moved. The shift is non-negative so the matrix stays lognorm-like.
    """
    rng = np.random.default_rng(7)
    real = _build(rng, scale=1.0, n_genes=40, n_cells_per=25)
    pred = _build(rng, scale=1.05, n_genes=40, n_cells_per=25)
    shift = np.abs(rng.normal(0.0, 0.5, size=real.shape[1]))
    mask = (real.obs["target"] != "non-targeting").to_numpy()
    real.X[mask] = real.X[mask] + shift
    pred.X[mask] = pred.X[mask] + shift
    return pred, real


def _build_counts(rng, lam, n_genes, n_cells_per):
    blocks, labels = [], []
    for p in PERTS:
        counts = rng.poisson(lam=lam, size=(n_cells_per, n_genes))
        blocks.append(counts)
        labels += [p] * n_cells_per
    X = np.vstack(blocks).astype(np.float32)
    obs = pd.DataFrame({"target": labels},
                       index=[f"c{i}" for i in range(X.shape[0])])
    var = pd.DataFrame(index=_var_names(n_genes))
    return ad.AnnData(X=X, obs=obs, var=var)


@pytest.fixture
def synthetic_counts_pair():
    """Deterministic (pred, real) integer-count AnnData pair, pert_col='target'."""
    rng = np.random.default_rng(1)
    real = _build_counts(rng, lam=3.0, n_genes=30, n_cells_per=40)
    pred = _build_counts(rng, lam=3.2, n_genes=30, n_cells_per=40)
    return pred, real


@pytest.fixture
def counts_bundle_inputs():
    """(baseline_pred, real, submission_pred): integer counts that define a USABLE
    baseline -> replicate scale for all six vcc2026 members.

    Three properties, each load-bearing and each learned the hard way:

    * PER-TARGET DISTINCT effects. `synthetic_counts_pair` draws every group from one Poisson,
      so `sum(expr_distance_unbiased)` is non-positive and #257 refuses the MSE member. But a
      single SHARED effect is not enough either: measured, it leaves fewer than
      `min_gate_size=10` genes clearing the real-side gate for every target, and
      `compute_replicate_anchor` RAISES rather than returning a weak anchor.
    * A NO-SKILL baseline arm. The 0 end must be a prediction with no effect at all. An
      effect-bearing full-depth prediction beats the half-depth replicate, which puts the
      baseline on the wrong side of the anchor and makes `is_degenerate` refuse every member.
    * TARGETS PRESENT IN `var_names` -- `_build_counts` already does this via `_var_names`, and
      dropping it is the #195 fixture trap: `exclude_target_gene` resolves nothing and PDS
      raises "no target resolves to a gene in the reference feature index".

    Measured on this fixture at the competition's k=5: all six members yield a usable scale
    (baseline 0.65 / 1.0532 / 0.9429 / 0.0 / 0.0638 / 0.0 against replicate 1.0 / 0.0233 /
    0.4395 / 0.8082 / 0.9867 / 0.1948), and `_replicate_entries` returns exactly six entries.
    """
    # SELF-CONTAINED, not the module-level PERTS. That tuple carries three targeted
    # perturbations and is shared by every other fixture in this file; five are needed here
    # (measured: with fewer, too few genes clear `min_gate_size=10` and the anchor raises),
    # and widening the shared tuple would perturb every existing test for an unrelated reason.
    perts = ("non-targeting", "GENE1", "GENE2", "GENE3", "GENE4", "GENE5")
    n_genes, n_cells = 40, 30
    named = [p for p in perts if p != "non-targeting"]
    # The #195 trap: each target must BE a measured gene.
    var = pd.DataFrame(index=named + [f"g{j}" for j in range(n_genes - len(named))])
    rng = np.random.default_rng(11)
    effects = {p: rng.integers(0, 8, size=n_genes).astype(np.float32) for p in named}

    def _build(seed, eff):
        r = np.random.default_rng(seed)
        blocks, labels = [], []
        for p in perts:
            X = r.poisson(lam=8.0, size=(n_cells, n_genes)).astype(np.float32)
            if eff and p in eff:
                X = X + eff[p]
            blocks.append(X)
            labels += [p] * n_cells
        X = np.vstack(blocks).astype(np.float32)
        return ad.AnnData(X=X, var=var.copy(),
                          obs=pd.DataFrame({"target": labels},
                                           index=[f"c{i}" for i in range(X.shape[0])]))

    real = _build(11, effects)
    baseline_pred = _build(13, None)          # no skill: the 0 end
    submission_pred = _build(12, effects)     # a good prediction: something to score
    return baseline_pred, real, submission_pred


def _toy_de_adata(n_ctrl_guides=3, n_pert=2, cells_per=5, n_genes=4, seed=0):
    """Raw-count AnnData for the deseq2 backend: an NTC-replicated control (``n_ctrl_guides``
    guide levels) + ``n_pert`` unreplicated (n=1) perturbations. pert_col='target_gene',
    replicate_col='guide', control='non-targeting'."""
    rng = np.random.default_rng(seed)
    rows, tg, guide = [], [], []
    for k in range(n_ctrl_guides):
        rows.append(rng.poisson(5, size=(cells_per, n_genes)))
        tg += ["non-targeting"] * cells_per
        guide += [f"NTC_{k}"] * cells_per
    for j in range(n_pert):
        rows.append(rng.poisson(8, size=(cells_per, n_genes)))
        tg += [f"GENE_{j}"] * cells_per
        guide += [f"GENE_{j}_g1"] * cells_per
    X = np.vstack(rows).astype(float)
    obs = pd.DataFrame({"target_gene": tg, "guide": guide})
    genes = [f"GENE_{i}" if i < n_pert else f"g{i}" for i in range(n_genes)]
    var = pd.DataFrame(index=genes)
    return ad.AnnData(X=X, obs=obs, var=var)


@pytest.fixture
def toy_de_adata():
    """Factory fixture: call as ``toy_de_adata(n_ctrl_guides=..., n_pert=..., ...)`` to build a
    raw-count deseq2 AnnData. Shared by the deseq2 backend + metric-family tests so neither
    needs a brittle cross-test-module import of the builder."""
    return _toy_de_adata


#: (n_up, n_down, up_fold, down_fold) per perturbation, GRADED from strong to weak. The weak
#: tail is the point: it clears min_gate_size=10 on the FULL real table but not on a half,
#: which is the exact condition the lfc_nmae substitution exists for. Measured on this
#: fixture: uniform split-half core scores 3 perturbations, the full-gate reference and the
#: member both score 4, on every one of the five derived seeds. A flat fixture scores 4/4/4
#: and would make the cohort assertion unfalsifiable.
#:
#: ⚠️ UNCHANGED by issue #172, deliberately, and the grading is why. The weak tail's full-table
#: gate is exactly 10 -- the boundary IS the fixture -- so removing one gate row would omit it
#: and break the guard for a reason this fixture is not about. #172's exclusion is kept clear of
#: the grading instead: `graded_counts_real` places each target's own gene in the gene range no
#: block moves (see there). Re-tuning the effects to absorb the lost row was measured and
#: rejected: (7, 3, 1.30, 0.8) restores the member's 4 but also lifts the HALF-data core to 4,
#: which is exactly the 4/4 state this grading exists to avoid.
_GRADED_EFFECTS = ((15, 10, 4.0, 0.2), (12, 8, 2.5, 0.4),
                   (8, 4, 1.5, 0.7), (6, 3, 1.25, 0.8))


@pytest.fixture
def graded_counts_real():
    """Counts with GRADED differential signal, for the gate-dependent DE metrics.

    `_build_counts` draws every perturbation from one Poisson, so NOTHING is significant and
    de_lfc_nmae's real-side gate is empty for every target -- its reference is empty and it
    emits no tidy rows at all. This fixture moves a distinct gene block per perturbation, with
    the effect size decreasing across the four, so the half-data gate loses the weakest one
    while the full-real gate keeps it.

    ⚠️ **Each target's OWN gene is measured, and sits in the 85..119 range no block moves.**
    Issue #172 made `de_lfc_nmae` drop that row from its gate and RAISE when NO target resolves
    against the reference feature index, so a `GENE1..GENE4` / `g0..g119` panel -- the suite's
    usual convention -- would hard-fail here for the construct-ID reason rather than for
    anything this fixture is about. Resolving every target fixes that; placing the gene outside
    every moved block keeps the exclusion clear of the GRADING, which is what this fixture is
    for and which lives on a one-gene margin (`_GRADED_EFFECTS`). The strong perturbations shift
    library sizes enough that the row is significant for them anyway and does leave their gates;
    what it must not do is push the weak tail under `min_gate_size`.
    """
    rng = np.random.default_rng(0)
    n_genes, n_cells_per = 120, 200
    base = rng.gamma(4.0, 2.0, size=n_genes)
    labels = ["non-targeting"] * n_cells_per
    blocks = [rng.poisson(base, size=(n_cells_per, n_genes))]
    genes = [f"g{i}" for i in range(n_genes)]
    for k, (n_up, n_dn, up, dn) in enumerate(_GRADED_EFFECTS, start=1):
        rate = base.copy()
        lo = (k - 1) * 25
        rate[lo:lo + n_up] *= up
        rate[lo + n_up:lo + n_up + n_dn] *= dn
        blocks.append(rng.poisson(rate, size=(n_cells_per, n_genes)))
        labels += [f"GENE{k}"] * n_cells_per
        genes[n_genes - k] = f"GENE{k}"    # own gene, outside every moved block (85..119)
    X = np.vstack(blocks).astype(np.float32)
    obs = pd.DataFrame({"target": labels}, index=[f"c{i}" for i in range(X.shape[0])])
    var = pd.DataFrame(index=genes)
    return ad.AnnData(X=X, obs=obs, var=var)
