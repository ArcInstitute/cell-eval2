"""B4, the responding-gene covariance family (issue #386 §4): ``cell_cov_corr`` and
``cell_cov_eig_ratio``.

The fixture is built so the family's whole argument is testable in one place. Each
perturbation raises 100 responding genes 4x above the control, and it does so in two ways
that are INDISTINGUISHABLE to a pseudobulk and opposite to a covariance:

  * ``correlated=True`` drives the responders off ONE latent per CELL, so they co-vary
    across cells -- the mixture-of-responding-states shape real perturbation data has.
  * ``correlated=False`` draws the same per-gene lognormal-Poisson marginals with an
    INDEPENDENT latent per gene, so every gene's mean (and hence the pseudobulk) is
    unchanged while the gene-gene structure is gone.

``test_the_pseudobulk_cannot_tell_but_the_cells_can`` is the measurement that makes the
family worth having; the rest pin the gene set, the depth equalization, the omission
rules, and the wiring.
"""

from __future__ import annotations

import anndata as ad
import numpy as np
import pandas as pd
import pytest

from cell_eval2.catalog import CATALOG, PROFILES
from cell_eval2.cell_rowsource import AnnDataSource
from cell_eval2.cells_prep import CellsPrep, pert_rng
from cell_eval2.config import EvalConfig
from cell_eval2.metrics.cells import (
    ccc_all_genes,
    cov_agreement,
    cov_ccc_metric,
    cov_metric,
    responding_genes,
)
from cell_eval2.gpu import resolve_device
from cell_eval2.run import _reject_cell_metrics, compute_metrics, dispatch_cell_metrics

_HAS_GPU = resolve_device("auto") == "cuda"

CTRL = "non-targeting"
#: Perturbation labels ARE gene names in the panel: `resolve_exclusion_columns` refuses a
#: panel where NO label resolves (the construct-ID mismatch), which is the shared
#: target-gene definition this family inherits -- see `test_target_gene_is_never_scored`.
PERTS = ("g0", "g40", "g80")
#: 100 of 500 genes respond. The responder FRACTION matters and is deliberately a
#: minority: CP10K is compositional, so a panel where half the genes rise 4x deflates
#: every other gene's CP10K value enough to give the NON-responders the larger
#: |log2FC| -- real, but not the regime `top_genes` is built for.
NG, NRESP, NCELL = 500, 100, 300
#: The latent loading on the responding genes. 1.6 puts the true off-diagonal correlations
#: well clear of the sampling noise at NCELL cells, so `correlated` vs `independent` is a
#: separation the test can assert on rather than a trend.
ALPHA = 1.6


def _panel(rng, *, correlated, n_cells=NCELL, n_genes=NG, perts=PERTS):
    blocks, labels = [], []
    base = np.full(n_genes, 1.0)
    blocks.append(rng.poisson(base, size=(n_cells, n_genes)).astype("float32"))
    labels += [CTRL] * n_cells
    for pi, p in enumerate(perts):
        rate = base.copy()
        resp = slice(pi * 20, pi * 20 + NRESP)      # overlapping, shifted responder sets
        rate[resp] = 4.0
        alpha = np.zeros(n_genes)
        alpha[resp] = ALPHA
        # exp(a*z - a^2/2) has mean 1 for z~N(0,1), so the per-gene MEAN rate is `rate`
        # either way and only the across-gene dependence differs.
        z = rng.normal(size=(n_cells, 1)) if correlated else rng.normal(size=(n_cells, n_genes))
        lam = rate[None, :] * np.exp(alpha[None, :] * z - 0.5 * alpha[None, :] ** 2)
        blocks.append(rng.poisson(lam).astype("float32"))
        labels += [p] * n_cells
    X = np.vstack(blocks)
    obs = pd.DataFrame({"target": labels}, index=[f"c{i}" for i in range(X.shape[0])])
    return ad.AnnData(X=X, obs=obs, var=pd.DataFrame(index=[f"g{j}" for j in range(n_genes)]))


@pytest.fixture(scope="module")
def panels():
    rng = np.random.default_rng(0)
    return {
        "real": _panel(rng, correlated=True),
        "match": _panel(rng, correlated=True),      # same process, independent draw
        "indep": _panel(rng, correlated=False),     # same marginals, no gene-gene structure
    }


def _both(pred, real, **kw):
    """``(corr_table, eig_table)`` through the metric functions, sharing one cache."""
    cache: dict = {}
    common = dict(pred_source=AnnDataSource(pred), real_source=AnnDataSource(real),
                  _cell_metric_cache=cache, pert_col="target", control=CTRL, **kw)
    return cov_metric(which="corr", **common), cov_metric(which="eig_ratio", **common)


def _prep(pred, real, **kw):
    return CellsPrep(AnnDataSource(pred), AnnDataSource(real),
                     pert_col="target", control=CTRL, **kw)


# ── catalog registration ──────────────────────────────────────────────────────

def test_registered_as_unscored_cell_kind_in_the_cell_profile():
    names = {"cell_cov_corr", "cell_cov_eig_ratio"}
    assert names <= set(PROFILES["cell"])
    for n in names:
        assert CATALOG[n].kind == "cell"
        assert CATALOG[n].normalization is None
        assert CATALOG[n].agg == "mean"
        # UNSCORED, on the docs/metrics.md §6a rule: a family grading emission realism
        # earns enrolment on its own measured ceiling and baseline, not by registration.
        assert not CATALOG[n].scoring.scored
        assert CATALOG[n].scoring.direction is None
        assert CATALOG[n].worst_value is None
    # No shipped profile moves -- this is what keeps `avg_score` and the competition
    # digest untouched (the digest itself is pinned by test_competition_rule.py).
    for profile in ("full", "minimal", "vcc", "vcc2026", "de", "anndata", "pds", "dispersion"):
        assert not (names & set(PROFILES.get(profile, [])))


# ── the measurement the family exists for ─────────────────────────────────────

def test_the_pseudobulk_cannot_tell_but_the_cells_can(panels):
    """The two predictions agree on the pseudobulk and differ completely on the covariance.

    This is the spec's §6 test 4 and the family's reason to exist: `expr_mae` does not
    penalise a prediction with no gene-gene structure against one with the real structure
    (both are the same per-gene marginals), while `cell_cov_corr` goes from ~0.97 to ~0.0.
    """
    cfg = EvalConfig(metrics=["expr_mae", "cell_cov_corr", "cell_cov_eig_ratio"],
                     pert_col="target", control=CTRL, input_type="counts",
                     validate_input=False, device="cpu")
    got = {}
    for name in ("match", "indep"):
        df = compute_metrics(panels[name], panels["real"], config=cfg).to_pandas()
        wide = df.pivot_table(index="perturbation", columns="metric", values="value")
        assert set(wide.index) == set(PERTS)          # control excluded, every pert scored
        got[name] = {c: float(wide[c].mean()) for c in wide.columns}

    # The pseudobulk metric cannot penalise the missing structure: the structure-free
    # prediction's error is not even 1.5x the matched one's. One-sided on purpose -- measured
    # 0.129 independent vs 0.256 matched: one latent per cell correlates the matched
    # prediction's pseudobulk errors across genes, so its expr_mae swings more per draw.
    assert got["indep"]["expr_mae"] < 1.5 * got["match"]["expr_mae"]
    # The covariance metrics are decisive.
    assert got["match"]["cell_cov_corr"] > 0.8
    assert abs(got["indep"]["cell_cov_corr"]) < 0.1
    # ... and the eigenvalue ratio reports the missing shared program as a deficit.
    assert 0.8 < got["match"]["cell_cov_eig_ratio"] < 1.3
    assert got["indep"]["cell_cov_eig_ratio"] < 0.3


def test_predicting_the_real_cells_is_the_exact_identity(panels):
    """pred IS real, and both sides fit under `max_cells`, so the two blocks are the same
    cells and both metrics are exactly 1. Pins that no normalization or sampling step
    perturbs one side relative to the other."""
    corr, eig = _both(panels["real"], panels["real"])
    assert set(corr) == set(PERTS)
    for p in PERTS:
        assert corr[p] == pytest.approx(1.0, abs=1e-12)
        assert eig[p] == pytest.approx(1.0, abs=1e-12)


# ── the gene set ──────────────────────────────────────────────────────────────

def _independent_log2fc(adata, pert):
    """`log2((cp10k_mean_p + 1) / (cp10k_mean_ctrl + 1))`, recomputed from the raw matrix.

    Deliberately NOT via `CellsPrep.real_log2fc` -- checking the selection against the same
    function that drives it would be a tautology. This is the spec's definition written out:
    CP10K per cell, mean over the group's cells, pseudocount 1, log2 of the ratio.
    """
    labels = adata.obs["target"].astype(str).to_numpy()
    X = np.asarray(adata.X, dtype=np.float64)
    cp10k = 1e4 * X / X.sum(axis=1, keepdims=True)
    mean_p = cp10k[labels == str(pert)].mean(axis=0)
    mean_c = cp10k[labels == CTRL].mean(axis=0)
    return np.log2((mean_p + 1.0) / (mean_c + 1.0))


def test_responding_genes_are_the_top_absolute_lfc_genes(panels):
    """The selection IS "top `top_genes` of G_p by |log2FC_real|", checked against an
    independent recomputation of the CP10K/pseudocount-1 fold change."""
    prep = _prep(panels["match"], panels["real"])
    genes = list(panels["real"].var_names)
    for p in PERTS:
        cols = responding_genes(prep, p, top_genes=100, min_nonzero_cells=10)
        assert cols.size == 100
        absfc = np.abs(_independent_log2fc(panels["real"], p))
        eligible = [c for c in prep.scored_genes(p).tolist()
                    if prep.stats("real", p).nonzero_cells[c] >= 10]
        want = sorted(sorted(eligible, key=lambda c: (-absfc[c], c))[:100])
        assert cols.tolist() == want
        assert genes.index(p) not in want          # and the target gene is not eligible


def test_the_selected_genes_are_mostly_the_ones_that_actually_moved(panels):
    """A biology check on top of the contract check above: the top-|log2FC| set is dominated
    by the genes the fixture actually raised 4x.

    Not 100 of 100, and that is the estimator rather than the selection: the responders are
    driven by a heavy-tailed lognormal latent (ALPHA=1.6), so their CP10K group mean is
    noisy over NCELL cells, and CP10K's compositionality pushes the non-responders to a
    |log2FC| of ~0.7 against the responders' ~1.3, so some genes cross over. Measured at
    90 / 96 / 81 of 100 for the three perturbations, hence a floor of 75 rather than a
    number that pins this particular draw.
    """
    prep = _prep(panels["match"], panels["real"])
    for pi, p in enumerate(PERTS):
        cols = set(responding_genes(prep, p, top_genes=100, min_nonzero_cells=10).tolist())
        planted = set(range(pi * 20, pi * 20 + NRESP))
        assert len(cols & planted) >= 75


def test_target_gene_is_never_scored(panels):
    """Each perturbation's OWN gene is excluded from `G_p`, through the same
    `distances.resolve_exclusion_columns` panel-scope definition `pds_*` uses (spec §2.3).

    Here the label `g0` IS gene column 0 and `g0` is inside its own responder block, so
    without the exclusion it would be scored -- a coordinate that is predictable from the
    label alone.
    """
    prep = _prep(panels["match"], panels["real"])
    genes = list(panels["real"].var_names)
    for p in PERTS:
        own = genes.index(p)
        assert own not in prep.scored_genes(p).tolist()
        assert own not in responding_genes(prep, p, top_genes=100,
                                           min_nonzero_cells=10).tolist()


def test_a_gene_zero_on_both_real_sides_is_dropped(panels):
    """Spec §2.3: a gene with zero counts in the real control AND the real perturbation has
    no comparison to support and leaves `G_p`."""
    real = panels["real"].copy()
    real.X[:, 7] = 0.0                    # dead in every group, control included
    prep = _prep(panels["match"], real)
    for p in PERTS:
        assert 7 not in prep.scored_genes(p).tolist()
    # A gene that is zero in the perturbation but LIVE in the control stays: that is a
    # real, and strong, perturbation effect.
    real2 = panels["real"].copy()
    rows = (real2.obs["target"].astype(str) == PERTS[0]).to_numpy()
    real2.X[rows, 9] = 0.0
    assert 9 in _prep(panels["match"], real2).scored_genes(PERTS[0]).tolist()


def test_a_gene_expressed_in_too_few_real_cells_is_not_scored(panels):
    """The `min_nonzero_cells` filter runs BEFORE the |log2FC| ranking: on a CP10K
    pseudobulk with pseudocount 1 a gene seen in a handful of cells can carry a large fold
    change, and a correlation across mostly-zero cells is noise with a defined value."""
    real = panels["real"].copy()
    rows = np.flatnonzero((real.obs["target"].astype(str) == PERTS[0]).to_numpy())
    real.X[:, 11] = 0.0
    real.X[rows[:4], 11] = 500.0          # huge |log2FC|, only 4 cells expressing
    prep = _prep(panels["match"], real)
    cols = responding_genes(prep, PERTS[0], top_genes=100, min_nonzero_cells=10)
    assert 11 not in cols.tolist()
    # ... and it IS picked up once enough cells express it.
    assert 11 in responding_genes(prep, PERTS[0], top_genes=100,
                                  min_nonzero_cells=4).tolist()


# ── depth equalization and omission rules ─────────────────────────────────────

def test_both_sides_are_cut_to_the_same_depth(panels):
    """`min(n_real, n_pred, max_cells)` per side. Equal depth is load-bearing: the metric
    compares two correlation matrices ESTIMATED from cells, and their sampling noise is a
    function of n, so unequal depths would be a confound in the comparison itself."""
    prep = _prep(panels["match"], panels["real"], max_cells=120)
    assert prep.depth(PERTS[0]) == 120
    pair = prep.pair(PERTS[0], np.arange(30))
    assert pair.real.shape == pair.pred.shape == (120, 30)

    # A thin pred side pulls BOTH sides down to its own count, not just itself.
    thin = panels["match"][
        ~((panels["match"].obs["target"].astype(str) == PERTS[0])
          & (np.arange(panels["match"].n_obs) % 5 != 0))
    ].copy()
    prep_thin = _prep(thin, panels["real"], max_cells=500)
    n_pred = int((thin.obs["target"].astype(str) == PERTS[0]).sum())
    assert prep_thin.depth(PERTS[0]) == n_pred < NCELL
    pair = prep_thin.pair(PERTS[0], np.arange(30))
    assert pair.real.shape[0] == pair.pred.shape[0] == n_pred


def test_a_perturbation_with_too_few_cells_is_omitted(panels):
    """Below `min_cells` on either side the perturbation is OMITTED, not filled: `agg='mean'`
    skips it, and both metrics are unscored, so an omission costs no score -- while a
    filled-in number would be a correlation the data cannot support."""
    corr, eig = _both(panels["match"], panels["real"], min_cells=NCELL + 1)
    assert corr == {} and eig == {}
    corr, _ = _both(panels["match"], panels["real"], min_cells=NCELL)
    assert set(corr) == set(PERTS)


def test_too_few_responding_genes_omits_the_perturbation(panels):
    corr, eig = _both(panels["match"], panels["real"], top_genes=10, min_genes=20)
    assert corr == {} and eig == {}


def test_constant_genes_are_dropped_and_can_omit_the_perturbation(panels):
    """A gene with no variance in the cells actually drawn has no correlation with
    anything. It is dropped after sampling and the gene floor is re-applied, so a
    prediction of constant cells is omitted rather than scored NaN-by-accident."""
    flat = panels["match"].copy()
    rows = (flat.obs["target"].astype(str) == PERTS[0]).to_numpy()
    flat.X[rows, :] = 1.0                 # every drawn pred cell identical on every gene
    corr, _ = _both(flat, panels["real"])
    assert PERTS[0] not in corr
    assert set(corr) == set(PERTS) - {PERTS[0]}


# ── the kernel, directly ──────────────────────────────────────────────────────

def test_cov_agreement_on_hand_built_blocks():
    rng = np.random.default_rng(7)
    block = rng.normal(size=(200, 12))
    same = cov_agreement(block, block.copy())
    assert same["corr"] == pytest.approx(1.0, abs=1e-12)
    assert same["eig_ratio"] == pytest.approx(1.0, abs=1e-12)

    # Sign-flipping one gene negates its whole row/column of R, so the upper triangles
    # disagree on exactly those entries: correlation drops well below 1 while the
    # eigenvalue MAGNITUDE is untouched -- the two metrics reading different things.
    flipped = block.copy()
    flipped[:, 0] *= -1.0
    got = cov_agreement(block, flipped)
    assert got["corr"] < 0.9
    assert got["eig_ratio"] == pytest.approx(1.0, abs=1e-12)

    # A constant upper triangle has no correlation to report -> NaN, not a number.
    const = np.repeat(rng.normal(size=(200, 1)), 12, axis=1)
    assert np.isnan(cov_agreement(block, const)["corr"])


# ── wiring: shared table, determinism, refusals ───────────────────────────────

def test_both_members_share_one_table(panels, monkeypatch):
    """The two entries must cost ONE pass over the cells, not two: the correlation matrices
    are computed once and both numbers read off them."""
    import cell_eval2.metrics.cells as mod

    calls = {"n": 0}
    real_agreement = mod.cov_agreement

    def counting(real_block, pred_block):
        calls["n"] += 1
        return real_agreement(real_block, pred_block)

    monkeypatch.setattr(mod, "cov_agreement", counting)
    _both(panels["match"], panels["real"])
    assert calls["n"] == len(PERTS)        # 3, not 6


def test_dispatch_emits_both_metrics_and_is_deterministic(panels):
    cfg = EvalConfig(metrics=["cell_cov_corr", "cell_cov_eig_ratio"], pert_col="target",
                     control=CTRL, input_type="counts", validate_input=False, device="cpu")
    args = (list(cfg.metrics), AnnDataSource(panels["match"]), AnnDataSource(panels["real"]),
            cfg)
    kwargs = dict(effective_types={"pred": "counts", "real": "counts"},
                  genes=np.asarray(panels["real"].var_names, dtype=str))
    rows = dispatch_cell_metrics(*args, **kwargs)
    again = dispatch_cell_metrics(*args, **kwargs)
    assert rows == again                                   # same seed -> identical rows
    df = pd.DataFrame(rows)
    assert set(df["metric"]) == {"cell_cov_corr", "cell_cov_eig_ratio"}
    assert set(df["perturbation"]) == set(PERTS)


def test_a_sparse_pair_gives_bit_comparable_numbers(panels):
    """Real submissions arrive as CSR. The gather path column-slices BEFORE densifying and
    takes library sizes from the FULL-panel row sum, so a sparse pair must agree with the
    dense one -- measured identical to 1e-6."""
    import scipy.sparse as sp

    real, match = panels["real"].copy(), panels["match"].copy()
    dense_corr, dense_eig = _both(match, real)
    real.X, match.X = sp.csr_matrix(real.X), sp.csr_matrix(match.X)
    sparse_corr, sparse_eig = _both(match, real)
    for p in PERTS:
        assert sparse_corr[p] == pytest.approx(dense_corr[p], abs=1e-6)
        assert sparse_eig[p] == pytest.approx(dense_eig[p], abs=1e-6)


def test_the_seeded_draw_does_not_depend_on_iteration_order():
    """`pert_rng` is keyed on (seed, pert), so adding a perturbation to a panel cannot move
    another one's cell sample -- which a single advancing generator would do."""
    a = pert_rng(0, "g40").integers(0, 10_000, size=5)
    b = pert_rng(0, "g40").integers(0, 10_000, size=5)
    c = pert_rng(0, "g80").integers(0, 10_000, size=5)
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)
    assert not np.array_equal(a, pert_rng(1, "g40").integers(0, 10_000, size=5))


def test_a_lognorm_pair_is_refused_not_silently_expm1d(panels):
    """Spec §2.1: Part B is defined on counts. A per-cell library size and a CP10K lognorm
    cannot be recovered from an already-normalized matrix, so the pair is refused."""
    with pytest.raises(ValueError, match="need raw counts"):
        _prep(panels["match"], panels["real"], input_type_pred="lognorm")
    with pytest.raises(ValueError, match="need raw counts"):
        _prep(panels["match"], panels["real"], input_type_real="lognorm")


@pytest.mark.parametrize("name", ["cell_cov_corr", "cell_cov_eig_ratio"])
def test_the_cellless_drivers_refuse_them(name):
    with pytest.raises(NotImplementedError, match="raw single cells"):
        _reject_cell_metrics([name, "expr_mae"], driver="score_streaming")


# ── cell_cov_ccc: Lin's CCC over the all-genes correlation matrix ──────────────

def _ccc(pred, real, **kw):
    """The ``cell_cov_ccc`` per-perturbation table, on the CPU path (numpy ``xp``)."""
    return cov_ccc_metric(
        pred_source=AnnDataSource(pred), real_source=AnnDataSource(real),
        _cell_metric_cache={}, pert_col="target", control=CTRL, device="cpu", **kw)


def test_ccc_registered_as_unscored_cell_kind_in_the_cell_profile():
    assert "cell_cov_ccc" in PROFILES["cell"]
    spec = CATALOG["cell_cov_ccc"]
    assert spec.kind == "cell"
    assert spec.normalization is None
    assert spec.agg == "mean"
    assert not spec.scoring.scored          # DIAG, docs/metrics.md §6a
    assert spec.scoring.direction is None
    assert spec.worst_value is None
    for profile in ("full", "minimal", "vcc", "vcc2026", "de", "anndata", "pds", "dispersion"):
        assert "cell_cov_ccc" not in set(PROFILES.get(profile, []))


def test_ccc_pseudobulk_blind_cells_decisive(panels):
    """The family's reason to exist, read through CCC over ALL genes: the matched
    (co-varying) prediction agrees, the structure-free one collapses toward 0 -- while the
    two are indistinguishable to a pseudobulk (asserted for `cell_cov_corr` already)."""
    match = _ccc(panels["match"], panels["real"])
    indep = _ccc(panels["indep"], panels["real"])
    assert set(match) == set(PERTS)
    assert float(np.mean(list(match.values()))) > 0.5
    assert abs(float(np.mean(list(indep.values())))) < 0.2
    # CCC is at most Pearson for the same vectors; on the matched pair (right pattern, right
    # scale) the two agree closely, which is the all-genes analogue of cell_cov_corr ~ 0.9.
    assert match[PERTS[0]] <= 1.0 + 1e-9


def test_ccc_identity_is_one(panels):
    """pred IS real and both sides fit under `max_cells`, so the two blocks are the same
    cells and the CCC of a matrix with itself is exactly 1."""
    got = _ccc(panels["real"], panels["real"])
    assert set(got) == set(PERTS)
    for p in PERTS:
        assert got[p] == pytest.approx(1.0, abs=1e-6)


def test_ccc_penalises_a_weaker_but_correctly_patterned_prediction():
    """CCC's point over Pearson, and the reason this metric exists alongside `cell_cov_corr`:
    a prediction whose correlation PATTERN is right but systematically weaker (what per-gene-
    independent sampling from a fitted marginal produces) reads ~1 on Pearson of the two
    triangles and is penalised by CCC, because CCC grades agreement about the identity line."""
    import cell_eval2.metrics.cells as mod

    rng = np.random.default_rng(3)
    load = rng.uniform(0.5, 1.5, size=(1, 60))
    z = rng.normal(size=(1500, 1))
    real = (z @ load + 0.25 * rng.normal(size=(1500, 60))).astype("float32")
    # Same latent and loadings (so the SIGN pattern of the gene-gene correlations matches),
    # but a far larger independent component -> every off-diagonal correlation shrinks.
    weak = (z @ load + 1.0 * rng.normal(size=(1500, 60))).astype("float32")

    # Pearson between the two correlation matrices' off-diagonal entries: pattern only.
    Ar, Br = mod._device_corr(real, np), mod._device_corr(weak, np)
    fin = np.isfinite(Ar) & np.isfinite(Br)
    pearson = float(np.corrcoef(Ar[fin], Br[fin])[0, 1])
    ccc = ccc_all_genes(real, weak, device="cpu")

    assert pearson > 0.85            # the pattern is largely preserved ...
    assert ccc < pearson - 0.1       # ... but CCC docks the systematic shrink Pearson ignores
    assert ccc_all_genes(real, real.copy(), device="cpu") == pytest.approx(1.0, abs=1e-6)


def test_ccc_a_pert_with_too_few_cells_is_omitted(panels):
    assert _ccc(panels["match"], panels["real"], min_cells=NCELL + 1) == {}
    assert set(_ccc(panels["match"], panels["real"], min_cells=NCELL)) == set(PERTS)


def test_ccc_intersection_is_the_default_and_differs_from_all_pairs(panels):
    """The shipped metric gates to pairs significant on BOTH sides (``significance='intersection'``).
    It is the default, and it is a different number from the un-gated all-pairs CCC."""
    default = _ccc(panels["match"], panels["real"])                 # no significance= -> default
    explicit = cov_ccc_metric(
        pred_source=AnnDataSource(panels["match"]), real_source=AnnDataSource(panels["real"]),
        _cell_metric_cache={}, pert_col="target", control=CTRL, device="cpu",
        significance="intersection")
    allpairs = cov_ccc_metric(
        pred_source=AnnDataSource(panels["match"]), real_source=AnnDataSource(panels["real"]),
        _cell_metric_cache={}, pert_col="target", control=CTRL, device="cpu",
        significance="none")
    assert set(default) == set(PERTS)
    for p in PERTS:
        assert default[p] == pytest.approx(explicit[p], abs=1e-12)   # default IS intersection
        assert default[p] != pytest.approx(allpairs[p], abs=1e-6)    # gate changes the value


def test_ccc_significance_gate_kernel():
    """`ccc_all_genes` keeps only |r|>=r_crit pairs on both sides under intersection; identity is
    still 1.0 (a matrix is trivially significant against itself on its significant pairs); an
    unknown mode raises."""
    rng = np.random.default_rng(1)
    z = rng.normal(size=(300, 1))
    load = rng.uniform(0.5, 1.5, size=(1, 25))
    # Half the genes co-vary off a latent (significant pairs), half are pure noise (non-significant):
    # so the intersection gate removes a real fraction and the two numbers must differ.
    def _block():
        return np.hstack([z @ load + 0.5 * rng.normal(size=(300, 25)),
                          rng.normal(size=(300, 25))]).astype("float32")
    real, pred = _block(), _block()
    none = ccc_all_genes(real, pred, device="cpu", significance="none")
    inter = ccc_all_genes(real, pred, device="cpu", significance="intersection")
    assert np.isfinite(none) and np.isfinite(inter)
    assert inter != pytest.approx(none, abs=1e-6)
    assert ccc_all_genes(real, real.copy(), device="cpu",
                         significance="intersection") == pytest.approx(1.0, abs=1e-6)
    with pytest.raises(ValueError, match="significance"):
        ccc_all_genes(real, pred, device="cpu", significance="bogus")


@pytest.mark.skipif(not _HAS_GPU, reason="no usable CUDA GPU")
def test_ccc_cuda_matches_cpu(panels):
    """``device="auto"`` puts cell_cov_ccc on the GPU wherever CUDA is present, while every other
    test here pins the CPU path. The two compute the G x G products in different float32 BLAS;
    on a 300-target, 18,533-gene real pair they agreed to 5.3e-6 per perturbation."""
    cpu = _ccc(panels["match"], panels["real"])
    gpu = cov_ccc_metric(
        pred_source=AnnDataSource(panels["match"]), real_source=AnnDataSource(panels["real"]),
        _cell_metric_cache={}, pert_col="target", control=CTRL, device="cuda")
    assert set(gpu) == set(cpu) == set(PERTS)
    for p in PERTS:
        assert gpu[p] == pytest.approx(cpu[p], abs=1e-4)
    from cell_eval2.cells_prep import lognorm

    counts = np.tile(np.arange(1, 41, dtype=np.float32), (300, 1))
    const = lognorm(counts, counts.sum(axis=1).astype(np.float64))
    assert np.isnan(ccc_all_genes(const, const.copy(), device="cuda"))


def test_ccc_a_constant_prediction_has_no_correlation_to_score():
    """Bit-identical cells carry no gene-gene correlation, so the CCC is NaN (left out of the
    aggregate) rather than a number. Centring in float32 left a constant column an SD of
    2.5e-5, above `_STD_FLOOR`: every off-diagonal entry survived and a constant block scored
    ~1.0 against itself (measured before the fix)."""
    import cell_eval2.metrics.cells as mod
    from cell_eval2.cells_prep import lognorm

    counts = np.tile(np.arange(1, 41, dtype=np.float32), (300, 1))
    const = lognorm(counts, counts.sum(axis=1).astype(np.float64))
    assert np.isnan(mod._device_corr(const, np)).all()
    rng = np.random.default_rng(0)
    live = rng.poisson(2.0, size=(300, 40)).astype("float32")
    live = lognorm(live, live.sum(axis=1).astype(np.float64))
    assert np.isnan(ccc_all_genes(live, const, device="cpu"))
    assert np.isnan(ccc_all_genes(const, const.copy(), device="cpu"))


def test_ccc_needs_three_distinct_gene_pairs():
    """The pair mask is symmetric, so the three-pair minimum is six mask entries: two
    co-varying gene pairs (and nothing else significant on both sides) give NaN, three give a
    number. The fixture's own precondition -- exactly k significant pairs -- is asserted."""
    import cell_eval2.metrics.cells as mod

    rng = np.random.default_rng(0)
    n = 300
    for k, defined in ((2, False), (3, True)):
        cols = []
        for _ in range(k):
            z = rng.normal(size=n)
            cols += [z + 0.1 * rng.normal(size=n), z + 0.1 * rng.normal(size=n)]
        real = np.column_stack(cols).astype("float32")
        pred = (real + 0.05 * rng.normal(size=real.shape)).astype("float32")
        A, B = mod._device_corr(real, np), mod._device_corr(pred, np)
        rc = mod._r_crit(n, A.shape[0] * (A.shape[0] - 1) / 2, alpha=0.05, correction="none")
        both = np.isfinite(A) & np.isfinite(B) & (np.abs(A) >= rc) & (np.abs(B) >= rc)
        assert int(both.sum()) // 2 == k
        assert bool(np.isfinite(ccc_all_genes(real, pred, device="cpu"))) is defined


def test_ccc_scores_all_genes_and_needs_no_target_gene_map(panels):
    """`cell_cov_ccc` scores over the WHOLE panel and excludes no gene, so it needs no
    target_gene_map: the result is identical with or without one (a supplied map is ignored),
    and a gene zero on both real sides -- which the responding-gene family drops -- simply
    contributes a NaN-masked column rather than an omission. This is the whole reason the metric
    runs on the NGN2 .shad (symbol labels, ENSG panel) with nothing but the two archives."""
    real = panels["real"].copy()
    real.X[:, 7] = 0.0                      # zero on every real group (incl control)
    no_map = _ccc(panels["match"], real)    # target_gene_map=None
    with_map = cov_ccc_metric(
        pred_source=AnnDataSource(panels["match"]), real_source=AnnDataSource(real),
        _cell_metric_cache={}, pert_col="target", control=CTRL, device="cpu",
        target_gene_map={p: p for p in PERTS})     # a map is accepted but not applied
    assert set(no_map) == set(PERTS)
    for p in PERTS:
        assert np.isfinite(no_map[p])
        assert no_map[p] == pytest.approx(with_map[p], abs=1e-6)   # exclusion not applied


def test_ccc_construction_needs_no_map_even_when_labels_are_unresolvable(panels):
    """CellsPrep resolves the target-gene exclusion LAZILY, so building it and running the
    all-genes cell_cov_ccc never requires a map -- even on a panel where NO label resolves to a
    gene (the symbol-vs-ENSG case). `scored_genes` would still raise; `cell_cov_ccc` never calls
    it."""
    real = panels["real"].copy()
    # Rename genes so NO perturbation label matches a gene id and there is no map: the eager
    # resolve this used to do in __init__ would raise here.
    real.var_names = [f"ENS{i}" for i in range(real.n_vars)]
    match = panels["match"].copy()
    match.var_names = list(real.var_names)
    got = _ccc(match, real)
    assert set(got) == set(PERTS) and all(np.isfinite(v) for v in got.values())
    # the exclusion machinery still raises for a metric that actually asks for it
    prep = _prep(match, real)
    with pytest.raises(ValueError):
        prep.scored_genes(PERTS[0])


def test_a_shared_cache_never_serves_another_seeds_table(panels):
    """The per-dispatch cache is keyed on every knob that changes a table, `max_cells` and `seed`
    included: two calls sharing one cache with different draws must each get their own numbers,
    equal to a fresh-cache call's. `max_cells` below the 300 cells per label makes `seed` bite."""
    common = dict(pred_source=AnnDataSource(panels["match"]),
                  real_source=AnnDataSource(panels["real"]), pert_col="target", control=CTRL)
    shared: dict = {}
    for fn, kw in ((cov_ccc_metric, dict(device="cpu")), (cov_metric, dict(which="corr"))):
        first = fn(_cell_metric_cache=shared, max_cells=100, seed=0, **common, **kw)
        second = fn(_cell_metric_cache=shared, max_cells=100, seed=1, **common, **kw)
        fresh = fn(_cell_metric_cache={}, max_cells=100, seed=1, **common, **kw)
        assert second == fresh
        assert second != first
        third = fn(_cell_metric_cache=shared, max_cells=120, seed=1, **common, **kw)
        assert third == fn(_cell_metric_cache={}, max_cells=120, seed=1, **common, **kw)


def test_ccc_dispatch_emits_and_is_deterministic(panels):
    cfg = EvalConfig(metrics=["cell_cov_ccc"], pert_col="target", control=CTRL,
                     input_type="counts", validate_input=False, device="cpu")
    args = (list(cfg.metrics), AnnDataSource(panels["match"]), AnnDataSource(panels["real"]),
            cfg)
    kwargs = dict(effective_types={"pred": "counts", "real": "counts"},
                  genes=np.asarray(panels["real"].var_names, dtype=str))
    rows = dispatch_cell_metrics(*args, **kwargs)
    assert rows == dispatch_cell_metrics(*args, **kwargs)
    df = pd.DataFrame(rows)
    assert set(df["metric"]) == {"cell_cov_ccc"}
    assert set(df["perturbation"]) == set(PERTS)


def test_ccc_refused_by_cellless_drivers():
    with pytest.raises(NotImplementedError, match="raw single cells"):
        _reject_cell_metrics(["cell_cov_ccc", "expr_mae"], driver="score_streaming")


def test_ccc_matches_direct_all_genes_computation(panels):
    """The metric equals a direct all-genes CCC over the same sampled cells -- pinning that it
    scores the WHOLE panel (`pair` with no cols) and nothing is quietly sliced out."""
    prep = _prep(panels["match"], panels["real"])
    got = _ccc(panels["match"], panels["real"])
    for p in PERTS:
        pair = prep.pair(p)              # all genes, same seed -> same cells as the metric
        ref = ccc_all_genes(pair.real, pair.pred, device="cpu")
        assert got[p] == pytest.approx(ref, abs=1e-6)


# ── streaming .shad source (CellStoreSource) ───────────────────────────────────

def _write_shad_pair(tmp_path):
    """A (pred, real) cell-layout .shad pair over the fixture panels. Skips without cellstream."""
    pytest.importorskip("cellstream")
    from cellstream.cell import write_cell_archive

    rng = np.random.default_rng(0)
    pred, real = _panel(rng, correlated=True), _panel(rng, correlated=True)
    paths = {}
    for side, adata in (("pred", pred), ("real", real)):
        shad = tmp_path / f"{side}.shad"
        ref = (adata.obs["target"].astype(str) == CTRL).to_numpy()
        write_cell_archive(adata, shad, group_by="target", reference=ref,
                           codec="zstd", overwrite=True)
        paths[side] = (adata, shad)
    return paths


def test_cell_store_source_gather_matches_anndata(tmp_path):
    """CellStoreSource.gather_rows returns the same cells as the in-memory AnnData (the streaming
    source is a drop-in for AnnDataSource)."""
    from cell_eval2.cell_rowsource import AnnDataSource, CellStoreSource

    paths = _write_shad_pair(tmp_path)
    adata, shad = paths["pred"]
    src = CellStoreSource(str(shad))
    try:
        assert src.n_vars == adata.n_vars and src.n_obs == adata.n_obs
        assert list(src.var_names) == list(adata.var_names)
        # Sorted and unsorted: both come back in the REQUESTED order (the CellSource contract).
        for rows in (np.array([3, 17, 50, 120]), np.array([120, 3, 50, 17])):
            got = src.gather_rows(rows).toarray()
            want = AnnDataSource(adata).gather_rows(rows)
            want = want.toarray() if hasattr(want, "toarray") else np.asarray(want)
            assert np.allclose(got, want)
    finally:
        src.close()


def test_streaming_fast_path_matches_materialized(tmp_path):
    """compute_metrics on a .shad pair takes the streaming fast path (no full materialize) and
    returns exactly what the in-memory path returns when the SAME archives are materialized --
    identical cells (same archive row order + same seed), so identical cell_cov_ccc values."""
    from cell_eval2._cell_archive import materialize_cell
    from cell_eval2.run import compute_metrics, _cell_stream_eligible

    paths = _write_shad_pair(tmp_path)
    pred_shad, real_shad = str(paths["pred"][1]), str(paths["real"][1])
    cfg = EvalConfig(metrics=["cell_cov_ccc"], pert_col="target", control=CTRL,
                     input_type="counts", validate_input=False, device="cpu")

    assert _cell_stream_eligible(pred_shad, real_shad, cfg)          # routes to streaming
    stream = compute_metrics(pred_shad, real_shad, config=cfg).to_pandas()

    mem = compute_metrics(materialize_cell(pred_shad), materialize_cell(real_shad),
                          config=cfg).to_pandas()
    s = stream.set_index("perturbation")["value"].sort_index()
    m = mem.set_index("perturbation")["value"].sort_index()
    assert list(s.index) == list(m.index)
    assert np.allclose(s.to_numpy(), m.to_numpy(), atol=1e-6)


def test_mixed_selection_does_not_take_streaming_path(tmp_path):
    """A selection that also includes a pseudobulk metric is NOT streaming-eligible: those metrics
    need the in-memory pseudobulk path, so the whole run falls through to it."""
    from cell_eval2.run import _cell_stream_eligible

    paths = _write_shad_pair(tmp_path)
    pred_shad, real_shad = str(paths["pred"][1]), str(paths["real"][1])
    cfg = EvalConfig(metrics=["cell_cov_ccc", "expr_mae"], pert_col="target", control=CTRL,
                     input_type="counts", validate_input=False, device="cpu")
    assert not _cell_stream_eligible(pred_shad, real_shad, cfg)


def test_run_meta_stub_reproduces_materialized_fingerprint(tmp_path):
    """build_run_meta's provenance pass is the SECOND place a .shad was materialized (no backed
    reader). Its metadata-only cell stub reads no payload (0 nnz) but reproduces
    fingerprint_adata(strict=False) bit-for-bit against a materialized read -- so the CLI `run`
    path records identical provenance without the full load."""
    from cell_eval2._cell_archive import materialize_cell
    from cell_eval2.baseline import _cell_metadata_adata, build_run_meta
    from cell_eval2.cache import fingerprint_adata

    paths = _write_shad_pair(tmp_path)
    pred_shad, real_shad = str(paths["pred"][1]), str(paths["real"][1])

    stub, full = _cell_metadata_adata(pred_shad), materialize_cell(pred_shad)
    assert stub.shape == full.shape and str(stub.X.dtype) == str(full.X.dtype)
    assert stub.X.nnz == 0
    assert (fingerprint_adata(stub, pert_col="target")
            == fingerprint_adata(full, pert_col="target"))

    cfg = EvalConfig(metrics=["cell_cov_ccc"], pert_col="target", control=CTRL,
                     input_type="counts", version="v2", validate_input=False, device="cpu")
    meta = build_run_meta(cfg, real_shad, pred_shad)
    assert meta["source_fingerprint"] == fingerprint_adata(
        materialize_cell(real_shad), pert_col="target")
    assert meta["input_type_real_effective"] == "counts"


def test_run_meta_stub_is_not_used_when_autodetect_reads_the_real_matrix(tmp_path):
    """Under ``autodetect_input_type`` the REAL side's metadata pass reads X too: the anchor
    semantic identity resolves the PRED convention on the real matrix
    (``anchor.anchor_semantic_params``), and a 0-nnz stub would report the convention of an
    empty matrix. On a LOGNORM real archive that recorded a different identity from the
    materialized read (measured before the fix), so ``score --anchor`` would refuse the run."""
    pytest.importorskip("cellstream")
    from cellstream.cell import write_cell_archive

    from cell_eval2._cell_archive import materialize_cell
    from cell_eval2.anchor import semantic_identity
    from cell_eval2.baseline import build_run_meta
    from cell_eval2.catalog import resolve_metrics

    counts = _panel(np.random.default_rng(0), correlated=True).X.astype(np.float64)
    lib = counts.sum(axis=1, keepdims=True)
    lib[lib == 0] = 1.0
    lognorm = np.log1p(1e4 * counts / lib).astype("float32")
    template = _panel(np.random.default_rng(0), correlated=True)
    paths = {}
    for side in ("pred", "real"):
        adata = ad.AnnData(X=lognorm.copy(), obs=template.obs.copy(), var=template.var.copy())
        paths[side] = str(tmp_path / f"{side}.shad")
        write_cell_archive(adata, paths[side], group_by="target",
                           reference=(adata.obs["target"].astype(str) == CTRL).to_numpy(),
                           codec="zstd", overwrite=True)

    cfg = EvalConfig(metrics="vcc2026", pert_col="target", control=CTRL, input_type="lognorm",
                     autodetect_input_type=True, device="cpu")
    names, _ = resolve_metrics(cfg.metrics, version=cfg.version)
    meta = build_run_meta(cfg, paths["real"], paths["pred"])
    assert meta["anchor_semantic_identity"] == semantic_identity(
        cfg, materialize_cell(paths["real"]), list(names))


# ── review fixes on the cell plumbing (Codex, PR #395) ─────────────────────────

def _write_shad(tmp_path, name, adata, *, group_by="target"):
    """One cell-layout archive of ``adata``. Skips without cellstream."""
    pytest.importorskip("cellstream")
    from cellstream.cell import write_cell_archive

    path = str(tmp_path / f"{name}.shad")
    write_cell_archive(adata, path, group_by=group_by,
                       reference=(adata.obs[group_by].astype(str) == CTRL).to_numpy(),
                       codec="zstd", overwrite=True)
    return path


def _ccc_cfg(**kw):
    return EvalConfig(metrics=["cell_cov_ccc"], pert_col="target", control=CTRL, device="cpu",
                      **kw)


def test_the_fast_path_validates_the_cells_it_reads(tmp_path):
    """The cell-layout fast path reads only the cells it samples, so no whole-matrix pass ever
    validated them: negative counts, and a library above ``max_counts_per_cell``, used to score.
    Every gathered block is now validated as the in-memory path validates the matrix -- which
    refuses the same pair."""
    from cell_eval2._cell_archive import materialize_cell
    from cell_eval2.run import _cell_stream_eligible

    rng = np.random.default_rng(0)
    real, pred = _panel(rng, correlated=True), _panel(rng, correlated=True)
    negative = pred.copy()
    negative.X = negative.X - 1.0
    real_p = _write_shad(tmp_path, "real", real)
    for name, adata, cfg, match in (
        ("negative", negative, _ccc_cfg(), "negative"),
        ("deep", pred, _ccc_cfg(max_counts_per_cell=100.0), "max_counts_per_cell|exceed"),
    ):
        pred_p = _write_shad(tmp_path, name, adata)
        assert _cell_stream_eligible(pred_p, real_p, cfg)
        with pytest.raises(ValueError, match=match):
            compute_metrics(pred_p, real_p, config=cfg)
        with pytest.raises(ValueError, match=match):
            compute_metrics(materialize_cell(pred_p), materialize_cell(real_p), config=cfg)


def test_a_backed_cell_only_run_validates_its_cells(tmp_path):
    """The in-memory path validates a BACKED input lazily, inside the pseudobulk and DE passes,
    or -- for a run needing neither -- in a pass it skips whenever a cache is configured, because
    such a run reads no X. A cell-metric-only run DOES read X, so with a cache a fractional
    'counts' prediction used to score; it is now refused, as the same matrix in memory is."""
    rng = np.random.default_rng(0)
    real, pred = _panel(rng, correlated=True), _panel(rng, correlated=True)
    pred.X = pred.X * 1.5
    real.write_h5ad(tmp_path / "real.h5ad")
    pred.write_h5ad(tmp_path / "pred.h5ad")
    for cfg in (_ccc_cfg(cache_pred=str(tmp_path / "cache")), _ccc_cfg()):
        with pytest.raises(ValueError, match="fractional"):
            compute_metrics(str(tmp_path / "pred.h5ad"), str(tmp_path / "real.h5ad"),
                            config=cfg)
    with pytest.raises(ValueError, match="fractional"):
        compute_metrics(pred, real, config=_ccc_cfg())


def test_the_fast_path_resolves_input_types_as_the_in_memory_path_does(tmp_path):
    """The DECLARED type, not a peek-autodetect, unless ``autodetect_input_type`` asks: a
    fractional prediction declared counts under ``allow_fractional_counts`` scores in memory, and
    the fast path used to re-type it 'lognorm' and refuse it."""
    from cell_eval2._cell_archive import materialize_cell

    rng = np.random.default_rng(0)
    real, pred = _panel(rng, correlated=True), _panel(rng, correlated=True)
    pred.X = pred.X * 1.5        # fractional row sums, which the peek-autodetect reads as lognorm
    real_p, pred_p = _write_shad(tmp_path, "real", real), _write_shad(tmp_path, "pred", pred)
    cfg = _ccc_cfg(allow_fractional_counts=True)
    stream = compute_metrics(pred_p, real_p, config=cfg).to_pandas()
    mem = compute_metrics(materialize_cell(pred_p), materialize_cell(real_p),
                          config=cfg).to_pandas()
    s = stream.set_index("perturbation")["value"].sort_index()
    m = mem.set_index("perturbation")["value"].sort_index()
    assert list(s.index) == list(m.index) == sorted(PERTS)
    assert np.allclose(s.to_numpy(), m.to_numpy(), atol=1e-6)


def test_the_fast_path_autodetects_from_the_rows_the_in_memory_path_samples(tmp_path):
    """Under ``autodetect_input_type`` the fast path once peeked at the first 2,000 archive rows.
    An archive is grouped, so those can all be integer while a later perturbation is fractional:
    the in-memory path (500 rows sampled across the matrix) calls such a pred side 'lognorm' and
    the cell metrics refuse it, while the archive path called it 'counts' and scored it."""
    from cell_eval2._cell_archive import materialize_cell

    rng = np.random.default_rng(0)
    sizes = {CTRL: 300, "g0": 1000, "g40": 1100, "g80": 300}       # g80 sorts last, past row 2000
    obs = pd.DataFrame({"target": np.concatenate([[p] * n for p, n in sizes.items()])},
                       index=[f"c{i}" for i in range(sum(sizes.values()))])
    var = pd.DataFrame(index=[f"g{j}" for j in range(50)])
    # Small values: as 'lognorm' they must stay under log1p(max_counts_per_cell), or the in-memory
    # path refuses them on scale before the cell metrics are reached.
    pred = ad.AnnData(X=np.vstack([
        rng.poisson(1.0, size=(n, 50)).astype("float32") * (1.5 if p == "g80" else 1.0)
        for p, n in sizes.items()]), obs=obs, var=var)
    real = ad.AnnData(X=np.vstack([rng.poisson(1.0, size=(n, 50)).astype("float32")
                                   for n in sizes.values()]), obs=obs.copy(), var=var.copy())
    pred_p, real_p = _write_shad(tmp_path, "pred", pred), _write_shad(tmp_path, "real", real)
    cfg = _ccc_cfg(autodetect_input_type=True, allow_fractional_counts=True)
    with pytest.raises(ValueError, match="need raw counts"):
        compute_metrics(materialize_cell(pred_p), materialize_cell(real_p), config=cfg)
    with pytest.raises(ValueError, match="need raw counts"):
        compute_metrics(pred_p, real_p, config=cfg)


def test_the_fast_path_groups_by_pert_col_whatever_the_storage_grouping(tmp_path):
    """The cell metrics gather rows by LABEL, so the fast path validates and groups by
    ``obs[pert_col]`` (``io.validate_pair``, as the in-memory path does) and an archive STORED
    grouped by another column scores as its materialized read. It once adopted the storage
    column whenever ``pert_col`` was left at 'target' -- scoring 'merged' buckets instead of
    targets -- and then refused such a pair outright. The storage column is adopted only when
    NEITHER archive has a 'target' column."""
    from cell_eval2._cell_archive import materialize_cell

    rng = np.random.default_rng(0)
    real, pred = _panel(rng, correlated=True), _panel(rng, correlated=True)
    for adata in (real, pred):
        labels = adata.obs["target"].astype(str)
        adata.obs["bucket"] = np.where(labels == CTRL, CTRL, "merged")
    real_p = _write_shad(tmp_path, "real", real, group_by="bucket")
    pred_p = _write_shad(tmp_path, "pred", pred, group_by="bucket")
    stream = compute_metrics(pred_p, real_p, config=_ccc_cfg()).to_pandas()
    mem = compute_metrics(materialize_cell(pred_p), materialize_cell(real_p),
                          config=_ccc_cfg()).to_pandas()
    s = stream.set_index("perturbation")["value"].sort_index()
    m = mem.set_index("perturbation")["value"].sort_index()
    assert list(s.index) == list(m.index) == sorted(PERTS)
    assert np.allclose(s.to_numpy(), m.to_numpy(), atol=1e-6)

    # Only the REAL archive carries 'target': nothing is adopted, so the pred side is refused.
    bare = pred.copy()
    bare.obs = bare.obs[["bucket"]]
    with pytest.raises(ValueError, match="missing"):
        compute_metrics(_write_shad(tmp_path, "bare", bare, group_by="bucket"), real_p,
                        config=_ccc_cfg())

    # The convenience itself still works: no 'target' column, so the archive's grouping is used.
    for adata in (real, pred):
        adata.obs = adata.obs[["target"]].rename(columns={"target": "perturbation"})
    real_p = _write_shad(tmp_path, "real2", real, group_by="perturbation")
    pred_p = _write_shad(tmp_path, "pred2", pred, group_by="perturbation")
    got = compute_metrics(pred_p, real_p, config=_ccc_cfg()).to_pandas()
    assert set(got["perturbation"]) == set(PERTS)


def test_the_fast_path_gathers_at_the_materialized_dtype(tmp_path):
    """A count above 65,535 makes the archive uint32 when materialized, while the store's own
    gathers are float32 -- whose row sums round above 2**24. A row of 16,777,217 counts then
    passed ``max_counts_per_cell=2**24`` on the archive path and failed it in memory (Codex,
    review round 3). Gathering at the materialized dtype makes the two agree."""
    from cell_eval2._cell_archive import materialize_cell
    from cell_eval2.cell_rowsource import CellStoreSource

    rng = np.random.default_rng(0)
    real, pred = _panel(rng, correlated=True), _panel(rng, correlated=True)
    row = int(np.flatnonzero(pred.obs["target"].astype(str).to_numpy() == PERTS[0])[0])
    pred.X[row, :] = 0.0
    pred.X[row, 0], pred.X[row, 1] = 8_388_608.0, 8_388_609.0      # sums to 2**24 + 1
    real_p, pred_p = _write_shad(tmp_path, "real", real), _write_shad(tmp_path, "pred", pred)
    src = CellStoreSource(pred_p)
    try:
        assert src.gather_rows(np.array([row])).dtype == materialize_cell(pred_p).X.dtype
    finally:
        src.close()
    cfg = _ccc_cfg(max_counts_per_cell=float(2 ** 24))
    with pytest.raises(ValueError, match="max_counts_per_cell|exceed"):
        compute_metrics(materialize_cell(pred_p), materialize_cell(real_p), config=cfg)
    with pytest.raises(ValueError, match="max_counts_per_cell|exceed"):
        compute_metrics(pred_p, real_p, config=cfg)


def test_the_fast_path_closes_the_pred_archive_when_the_real_one_fails(tmp_path, monkeypatch):
    import os

    import cell_eval2.cell_rowsource as rs

    paths = _write_shad_pair(tmp_path)
    closed = []

    class Recording(rs.CellStoreSource):
        def __init__(self, path, **kw):
            if os.path.basename(str(path)).startswith("real"):
                raise OSError("corrupt archive")
            super().__init__(path, **kw)

        def close(self):
            closed.append(True)
            super().close()

    monkeypatch.setattr(rs, "CellStoreSource", Recording)
    with pytest.raises(OSError, match="corrupt archive"):
        compute_metrics(str(paths["pred"][1]), str(paths["real"][1]), config=_ccc_cfg())
    assert closed == [True]


def test_run_meta_stub_matches_a_materialized_uint32_archive(tmp_path):
    """A count above 65,535 makes the archive's recorded minimum dtype uint32, which the
    whole-archive reader keeps while a row gather returns float32. The stub used to probe a
    gather, so its fingerprint differed from the materialized read's."""
    from cell_eval2._cell_archive import materialize_cell
    from cell_eval2.baseline import _cell_metadata_adata
    from cell_eval2.cache import fingerprint_adata

    adata = _panel(np.random.default_rng(0), correlated=True)
    adata.X[0, 0] = 70_000.0
    path = _write_shad(tmp_path, "deep", adata)
    stub, full = _cell_metadata_adata(path), materialize_cell(path)
    assert str(stub.X.dtype) == str(full.X.dtype) == "uint32"
    assert fingerprint_adata(stub, pert_col="target") == fingerprint_adata(full, pert_col="target")


def test_the_result_cache_keys_on_the_cell_metrics_knobs(tmp_path, panels, monkeypatch):
    """The knobs are catalog-bound defaults, not config fields, so `config_hash` cannot see them:
    a result cached under ``significance='intersection'`` used to be served after the binding
    changed to ``'none'``."""
    import dataclasses
    import functools

    from cell_eval2 import catalog as cat

    cfg = _ccc_cfg(cache_pred=str(tmp_path / "cache"))
    first = compute_metrics(panels["match"], panels["real"], config=cfg)
    spec = cat.CATALOG["cell_cov_ccc"]
    monkeypatch.setitem(cat.CATALOG, "cell_cov_ccc", dataclasses.replace(
        spec, func=functools.partial(cov_ccc_metric, significance="none")))
    second = compute_metrics(panels["match"], panels["real"], config=cfg)
    uncached = compute_metrics(panels["match"], panels["real"],
                               config=dataclasses.replace(cfg, cache_pred=None))
    assert second["value"].to_list() == uncached["value"].to_list()
    assert second["value"].to_list() != first["value"].to_list()


def test_group_stats_does_not_depend_on_the_chunking(panels, monkeypatch):
    """The stats pass is chunked by BYTES now (a 20,000-row chunk densified to ~9 GB at 18.5k
    genes); any chunking gives the same summary."""
    import cell_eval2.cells_prep as cp

    class Recording(AnnDataSource):
        def __init__(self, adata):
            super().__init__(adata)
            self.gathered = []

        def gather_rows(self, rows):
            self.gathered.append(np.asarray(rows).copy())
            return super().gather_rows(rows)

    rows = np.flatnonzero(panels["real"].obs["target"].astype(str).to_numpy() == CTRL)
    whole = cp.group_stats(Recording(panels["real"]), rows)
    monkeypatch.setattr(cp, "_STATS_CHUNK_BYTES", 8 * panels["real"].n_vars * 7)   # 7 rows
    src = Recording(panels["real"])
    chunked = cp.group_stats(src, rows)
    assert max(len(g) for g in src.gathered) == 7
    assert len(src.gathered) == -(-rows.size // 7)
    assert np.array_equal(np.concatenate(src.gathered), rows)      # every row, once, in order
    assert np.array_equal(whole.nonzero_cells, chunked.nonzero_cells)
    np.testing.assert_allclose(whole.cp10k_mean, chunked.cp10k_mean, rtol=1e-12)


# ── review fixes, Copilot round 1 on #395 ──────────────────────────────────────

def test_the_cli_resolves_the_archive_grouping_before_provenance(tmp_path, monkeypatch):
    """`cell-eval2 run` builds `run_meta.json` BEFORE scoring, from the requested `pert_col`, so
    the fast path's adoption of the archives' grouping never reached the CLI: two archives
    grouped by 'perturbation' with no 'target' column failed provenance with "perturbation column
    'target' missing". The CLI now resolves the column first, through the same resolver."""
    import json
    import sys

    import polars as pl

    from cell_eval2 import cli

    rng = np.random.default_rng(0)
    paths = {}
    for side in ("pred", "real"):
        adata = _panel(rng, correlated=True)
        adata.obs = adata.obs.rename(columns={"target": "perturbation"})
        paths[side] = _write_shad(tmp_path, side, adata, group_by="perturbation")
    out = tmp_path / "out"
    monkeypatch.setattr(sys, "argv", ["cell-eval2", "run", "-ap", paths["pred"],
                                      "-ar", paths["real"], "--profile", "cell",
                                      "-o", str(out), "--set", "device=cpu"])
    cli.main()
    results = pl.read_csv(out / "results.csv")
    assert set(results["perturbation"]) == set(PERTS)
    assert set(results["metric"]) == {"cell_cov_ccc", "cell_cov_corr", "cell_cov_eig_ratio"}
    assert json.loads((out / "run_meta.json").read_text())["source_fingerprint"]


def test_the_anchor_identity_keys_on_the_cell_metrics_knobs(panels, monkeypatch):
    """The result cache keys on the cell metrics' bound knobs; a replicate anchor -- cached and
    validated by its semantic identity -- did not, so an anchor computed under one binding was
    accepted under another. Selections without a cell metric carry no such term, so no shipped
    profile's anchor identity moves."""
    import dataclasses
    import functools

    from cell_eval2.anchor import anchor_semantic_params, semantic_identity
    from cell_eval2.catalog import resolve_metrics

    cfg = _ccc_cfg()
    before = semantic_identity(cfg, panels["real"], ["cell_cov_ccc"])
    monkeypatch.setitem(CATALOG, "cell_cov_ccc", dataclasses.replace(
        CATALOG["cell_cov_ccc"], func=functools.partial(cov_ccc_metric, significance="none")))
    assert semantic_identity(cfg, panels["real"], ["cell_cov_ccc"]) != before
    for profile in ("full", "vcc2026"):
        names, _ = resolve_metrics(profile)
        params = anchor_semantic_params(dataclasses.replace(cfg, metrics=profile),
                                        panels["real"], names)
        assert "cell_metric_params" not in params


def test_the_fast_path_says_it_keeps_no_cache(tmp_path, caplog):
    """The archive fast path reads both archives directly and caches nothing; a configured cache
    used to be ignored without a word."""
    import logging

    paths = _write_shad_pair(tmp_path)
    cfg = _ccc_cfg(cache_pred=str(tmp_path / "cache"))
    with caplog.at_level(logging.WARNING, logger="cell_eval2.run"):
        compute_metrics(str(paths["pred"][1]), str(paths["real"][1]), config=cfg)
    assert any("cache_pred" in r.getMessage() and "not used" in r.getMessage()
               for r in caplog.records)


def test_a_backed_cell_only_run_reads_rows_without_materializing(tmp_path, panels,
                                                                monkeypatch):
    """The cell dispatch used to materialize BOTH backed matrices to sample at most 500 cells per
    perturbation. It now reads the backed handles row block by row block, nothing else
    materializes a side, and it scores what the in-memory pair scores."""
    import anndata as _ad

    import cell_eval2.run as run

    panels["real"].write_h5ad(tmp_path / "real.h5ad")
    panels["match"].write_h5ad(tmp_path / "pred.h5ad")
    calls = []
    real_materialize = run._materialize

    def counting(source):
        calls.append(source)
        return real_materialize(source)

    monkeypatch.setattr(run, "_materialize", counting)
    mem = compute_metrics(panels["match"], panels["real"], config=_ccc_cfg()).to_pandas()
    m = mem.set_index("perturbation")["value"].sort_index()
    # With a cache, and -- since the no-cache pass that validates a backed side by materializing
    # it now skips cell-only selections (Copilot, #395) -- without one, validation on or off.
    # target_sum=None too: the metrics read no target_sum, so resolving it -- which materializes
    # the real control pool -- is skipped for a cell-only selection (Copilot, #395).
    for cfg in (_ccc_cfg(cache_pred=str(tmp_path / "cache")), _ccc_cfg(),
                _ccc_cfg(validate_input=False), _ccc_cfg(target_sum=None)):
        backed = compute_metrics(str(tmp_path / "pred.h5ad"), str(tmp_path / "real.h5ad"),
                                 config=cfg).to_pandas()
        assert calls == []
        b = backed.set_index("perturbation")["value"].sort_index()
        assert list(b.index) == list(m.index) == sorted(PERTS)
        assert np.allclose(b.to_numpy(), m.to_numpy(), atol=1e-6)

    # A backed X is read through h5py, which wants increasing indices: the source sorts, then
    # restores the requested order.
    handle = _ad.read_h5ad(tmp_path / "real.h5ad", backed="r")
    try:
        rows = np.array([120, 3, 50, 17])
        got = AnnDataSource(handle).gather_rows(rows)
        assert np.allclose(np.asarray(got), np.asarray(panels["real"].X[rows]))
    finally:
        handle.file.close()


def test_backed_inputs_are_read_by_rows_only_where_that_is_bounded(tmp_path, panels):
    """A backed CSR (or dense) X answers a sorted row request with just those rows, so it is
    wrapped as it is. A CSC-backed X is read WHOLE by anndata for a row subset, and a backed VIEW
    materializes the entire view on every ``.X`` access -- wrapped directly, either would repeat a
    whole-matrix read per perturbation (Codex, round 5), so each is materialized once instead.
    Every route scores what the in-memory pair scores."""
    import anndata as _ad
    import scipy.sparse as sp

    from cell_eval2.run import _cell_row_source

    real = panels["real"]
    for fmt, X in (("csr", sp.csr_matrix(real.X)), ("csc", sp.csc_matrix(real.X)),
                   ("dense", np.asarray(real.X))):
        path = tmp_path / f"{fmt}.h5ad"
        _ad.AnnData(X=X, obs=real.obs.copy(), var=real.var.copy()).write_h5ad(path)
        handle = _ad.read_h5ad(path, backed="r")
        try:
            src = _cell_row_source(handle)
            assert getattr(src._adata, "isbacked", False) is (fmt != "csc"), fmt
            view = handle[handle.obs["target"].astype(str) != CTRL]
            assert view.is_view and not getattr(_cell_row_source(view)._adata, "is_view", False)
        finally:
            handle.file.close()

    csc_real, csc_pred = tmp_path / "real_csc.h5ad", tmp_path / "pred_csc.h5ad"
    _ad.AnnData(X=sp.csc_matrix(real.X), obs=real.obs.copy(),
                var=real.var.copy()).write_h5ad(csc_real)
    match = panels["match"]
    _ad.AnnData(X=sp.csc_matrix(match.X), obs=match.obs.copy(),
                var=match.var.copy()).write_h5ad(csc_pred)
    backed = compute_metrics(str(csc_pred), str(csc_real),
                             config=_ccc_cfg(cache_pred=str(tmp_path / "cache"))).to_pandas()
    mem = compute_metrics(match, real, config=_ccc_cfg()).to_pandas()
    b = backed.set_index("perturbation")["value"].sort_index()
    m = mem.set_index("perturbation")["value"].sort_index()
    assert np.allclose(b.to_numpy(), m.to_numpy(), atol=1e-6)


def test_the_anchor_identity_carries_the_target_gene_map_for_cell_metrics(panels):
    """`cell_cov_corr` / `cell_cov_eig_ratio` exclude each perturbation's own gene through
    `target_gene_map`, but the anchor identity carried the map only for pds/DE/on-target
    selections: an anchor computed under one map was served under another (Codex, round 5)."""
    import dataclasses

    from cell_eval2.anchor import semantic_identity

    cfg = _ccc_cfg()
    own = dataclasses.replace(cfg, target_gene_map={p: p for p in PERTS})
    other = dataclasses.replace(cfg, target_gene_map={p: "g499" for p in PERTS})
    assert (semantic_identity(own, panels["real"], ["cell_cov_corr"])
            != semantic_identity(other, panels["real"], ["cell_cov_corr"]))


def test_an_empty_side_indexes_to_no_perturbations():
    """`_index_by_pert` on a source with no cells raised IndexError (Gemini, #395)."""
    empty = ad.AnnData(X=np.zeros((0, 5), dtype="float32"),
                       obs=pd.DataFrame({"target": pd.Series([], dtype=str)}),
                       var=pd.DataFrame(index=[f"g{j}" for j in range(5)]))
    prep = CellsPrep(AnnDataSource(empty), AnnDataSource(empty), pert_col="target", control=CTRL)
    assert prep.perts == []


def test_backed_global_stats_take_the_chunked_pass(tmp_path, panels):
    """On a BACKED X the direct reductions would read the whole matrix (or fail on anndata's
    backed dataset); they take the base class's chunked pass instead and agree with a resident
    X (Gemini, #395)."""
    import anndata as _ad

    panels["real"].write_h5ad(tmp_path / "real.h5ad")
    handle = _ad.read_h5ad(tmp_path / "real.h5ad", backed="r")
    try:
        backed, resident = AnnDataSource(handle), AnnDataSource(panels["real"])
        np.testing.assert_allclose(backed.col_mean(), resident.col_mean(), rtol=1e-6)
        np.testing.assert_allclose(backed.row_sum(), resident.row_sum(), rtol=1e-6)
        assert backed.global_max() == pytest.approx(resident.global_max())
    finally:
        handle.file.close()


def test_column_sliced_chunks_read_rows_first():
    """`iter_row_chunks(cols=...)` sliced every row by column up front, which for a backed X reads
    the whole matrix before the first chunk (Copilot, #395). It now reads row slices only."""
    import types

    dense = np.arange(60, dtype="float32").reshape(20, 3)

    class RecordingX:
        def __init__(self, X):
            self.X, self.keys = X, []

        def __getitem__(self, key):
            self.keys.append(key)
            return self.X[key]

    rec = RecordingX(dense)
    fake = types.SimpleNamespace(X=rec, obs=pd.DataFrame(index=range(20)),
                                 var=pd.DataFrame(index=["a", "b", "c"]),
                                 var_names=pd.Index(["a", "b", "c"]), n_obs=20, n_vars=3,
                                 isbacked=True)
    chunks = list(AnnDataSource(fake).iter_row_chunks(7, cols=np.array([0, 2])))
    assert all(isinstance(k, slice) for k in rec.keys)
    assert np.array_equal(np.vstack([c for _, _, c in chunks]), dense[:, [0, 2]])


def test_archive_gathers_use_the_configured_decode_threads(tmp_path, monkeypatch):
    """The archive source gathered with cellstream's serial default; it now passes
    `resolve_gather_threads(rows, EvalConfig.gather_threads)` on every gather (Copilot, #395)."""
    import cell_eval2.cell_rowsource as rs
    from cell_eval2._threads import resolve_gather_threads

    paths = _write_shad_pair(tmp_path)
    src = rs.CellStoreSource(str(paths["pred"][1]), gather_threads=3)
    try:
        seen = []
        real_gather = src._store.gather_rows

        def recording(rows, **kw):
            seen.append(kw.get("n_threads"))
            return real_gather(rows, **kw)

        src._store.gather_rows = recording
        rows = np.arange(src.n_obs)
        src.gather_rows(rows)
        list(src.iter_row_chunks(500))
        assert seen and seen[0] == resolve_gather_threads(rows.size, 3)
        assert all(t is not None and t >= 1 for t in seen)
    finally:
        src.close()

    built = []

    class Recording(rs.CellStoreSource):
        def __init__(self, path, **kw):
            built.append(kw.get("gather_threads"))
            super().__init__(path, **kw)

    monkeypatch.setattr(rs, "CellStoreSource", Recording)
    compute_metrics(str(paths["pred"][1]), str(paths["real"][1]),
                    config=_ccc_cfg(gather_threads=5))
    assert built and all(t == 5 for t in built)


def test_the_cellless_drivers_refuse_before_reading_anything(monkeypatch, tmp_path):
    """`score_streaming` refused a cell metric only after resolving input types -- which, with
    target_sum=None, reads the real archive's whole control pool -- and `score_piece` only after
    loading its bundle and piece (Copilot, #395). Both now refuse first."""
    import cell_eval2.partition_inmem as pim
    import cell_eval2.scale as scale

    def no_reads(*a, **k):
        raise AssertionError("read input before refusing a cell metric")

    monkeypatch.setattr(scale, "_shard_effective_input_type", no_reads)
    monkeypatch.setattr(pim, "_RefBundle", no_reads)
    cfg = EvalConfig(metrics=["cell_cov_ccc"], target_sum=None, device="cpu")
    with pytest.raises(NotImplementedError, match="raw single cells"):
        scale.score_streaming("pred.shad", "real.shad", config=cfg)
    # On a host without gpudge the partition-config check refuses first; either way nothing is
    # read (`_RefBundle` would raise the AssertionError above).
    with pytest.raises(NotImplementedError):
        pim.score_piece(panels_free_piece(), str(tmp_path), config=cfg, piece_id="p0")


def panels_free_piece():
    """A tiny pred piece; it must never be read."""
    return ad.AnnData(X=np.ones((2, 3), dtype="float32"),
                      obs=pd.DataFrame({"target": ["g0", "g0"]}),
                      var=pd.DataFrame(index=["g0", "g1", "g2"]))


def test_an_iterator_selection_survives_the_cell_layout_routing(tmp_path):
    """`_cell_stream_eligible` resolves the selection to route a `.shad` pair, and the scorer it
    routes to resolves it again: a one-shot iterator came out empty the second time, on the fast
    path AND on the in-memory fallback (Copilot, #395)."""
    paths = _write_shad_pair(tmp_path)
    pred_p, real_p = str(paths["pred"][1]), str(paths["real"][1])
    for selection, metric in ((["cell_cov_ccc"], "cell_cov_ccc"), (["expr_mae"], "expr_mae")):
        cfg = EvalConfig(metrics=iter(selection), pert_col="target", control=CTRL,
                         device="cpu", validate_input=False)
        got = compute_metrics(pred_p, real_p, config=cfg).to_pandas()
        assert set(got["metric"]) == {metric}


def test_the_global_stats_pass_is_chunked_by_bytes(panels, monkeypatch):
    """`_ensure_global_stats` streamed 50,000-row dense chunks whatever the gene count (3.7 GB at
    18.5k genes); it now takes its rows per chunk from a byte budget (Gemini, #395)."""
    import cell_eval2.cell_rowsource as rs

    seen = []

    class Recording(rs.CellStoreSource.__mro__[1]):        # the CellSource base
        def __init__(self, adata):
            super().__init__()
            self._inner = AnnDataSource(adata)
            self.obs, self.var = adata.obs, adata.var
            self.var_names, self.n_obs, self.n_vars = adata.var_names, adata.n_obs, adata.n_vars

        def iter_row_chunks(self, chunk_size, cols=None):
            seen.append(chunk_size)
            return self._inner.iter_row_chunks(chunk_size, cols)

    monkeypatch.setattr(rs, "_GLOBAL_STATS_CHUNK_BYTES", 4 * panels["real"].n_vars * 11)
    src = Recording(panels["real"])
    np.testing.assert_allclose(src.col_mean(), AnnDataSource(panels["real"]).col_mean(),
                               rtol=1e-6)
    assert seen == [11]


def test_the_reference_builders_refuse_cell_metrics_before_reading(monkeypatch, tmp_path):
    """`score_cellstream`, `score_h5ad_manifest` and `score_rowstore` build the real reference --
    its real-side DE included -- before `score_piece` runs, so a cell metric was refused only
    after that whole pass (Copilot, #395). The builders now refuse first. `_resolve_backend` is
    forced to 'gpudge' so the partition-config check passes without a CUDA device, as in
    tests/test_ground_truth_cache_streaming.py."""
    import cell_eval2.de_compute as de_compute
    import cell_eval2.partition_inmem as pim

    monkeypatch.setattr(pim, "_resolve_backend", lambda backend: "gpudge")
    monkeypatch.setattr(de_compute, "_resolve_backend", lambda backend: "gpudge")
    cfg = EvalConfig(metrics=iter(["cell_cov_ccc", "expr_mae"]), device="cpu")

    class NoReads:
        def __getattr__(self, name):
            raise AssertionError(f"read source.{name} before refusing a cell metric")

    with pytest.raises(NotImplementedError, match="raw single cells"):
        pim.build_reference("unused.h5ad", config=cfg, cache_dir=str(tmp_path / "a"))
    cell_cfg = EvalConfig(metrics=["cell_cov_ccc"], device="cpu")
    with pytest.raises(NotImplementedError, match="raw single cells"):
        pim._build_reference_streaming_core(NoReads(), config=cell_cfg,
                                            cache_dir=str(tmp_path / "b"), mem_budget=None)
    with pytest.raises(NotImplementedError, match="raw single cells"):
        pim._build_pred_control_reference_core(NoReads(), config=cell_cfg,
                                               cache_dir=str(tmp_path / "c"))


def test_the_partitioned_scorers_refuse_cell_metrics_before_reading(monkeypatch, tmp_path):
    """`score_cellstream`, `score_h5ad_manifest` and `score_rowstore` sampled, peeked at or (with
    target_sum=None) materialized their inputs before the reference builders refused a cell
    metric (Codex, #395). Each now refuses at its own boundary: the inputs here do not exist, so
    any read would raise something else first."""
    import cell_eval2.de_compute as de_compute
    import cell_eval2.partition_inmem as pim
    from cell_eval2.cellstream import score_cellstream
    from cell_eval2.h5ad_manifest import score_h5ad_manifest
    from cell_eval2.rowstore import score_rowstore

    monkeypatch.setattr(pim, "_resolve_backend", lambda backend: "gpudge")
    monkeypatch.setattr(de_compute, "_resolve_backend", lambda backend: "gpudge")
    missing = str(tmp_path / "does-not-exist")
    for selection in (["cell_cov_ccc"], iter(["cell_cov_ccc", "expr_mae"])):
        cfg = EvalConfig(metrics=selection, device="cpu", target_sum=None)
        with pytest.raises(NotImplementedError, match="raw single cells"):
            score_cellstream(missing, missing, config=cfg, mem_budget=None)
    for scorer in (score_h5ad_manifest, score_rowstore):
        with pytest.raises(NotImplementedError, match="raw single cells"):
            scorer(missing, config=EvalConfig(metrics=["cell_cov_ccc"], device="cpu",
                                              target_sum=None), mem_budget=None)
