"""Sampled-cell (Part B) metrics -- issue #386 §4. Currently B4, the responding-gene
gene-gene covariance family.

**What this family is for.** No metric in the shipped profiles looks at how genes co-vary
across cells: ``expr_*``, ``delta_*`` and ``pds_*`` compare pseudobulks (``docs/metrics.md``
§1.1) -- means, and a mean cannot see a covariance -- and ``de_*`` tests one gene at a time.
So two predictions with the same per-gene behaviour and different gene-gene structure can
score alike on all of them. B4 measures that structure directly: whether the genes that MOVE
under a perturbation co-vary across single cells the way they do in the real data.

  * ``cell_cov_corr`` -- Pearson correlation between the upper triangles of the real and
    predicted gene-gene correlation matrices, over the responding genes. 1.0 is identical
    co-expression structure; 0.0 is unrelated structure. This is the metric of the pair.
  * ``cell_cov_eig_ratio`` -- ``λ1(R_pred) / λ1(R_real)``, the leading eigenvalue ratio.
    A one-number read on HOW MUCH shared structure there is rather than whether it is the
    right structure: 1.0 matches the real data's dominant-program strength, < 1 means the
    prediction's responding genes are more independent than the real ones (what
    independent per-gene sampling from a fitted marginal produces), > 1 means more
    correlated. ⚠️ It is a ratio of magnitudes and says NOTHING about agreement --
    a prediction whose dominant program is as strong as the real one but points somewhere
    else reads 1.0 here and ~0.0 on ``cell_cov_corr``. Read them together; that is why
    both ship.

Both are UNSCORED (``DIAG``) and live in the opt-in ``cell`` profile. The reason is on the
record in ``docs/metrics.md`` §6a: the one previous metric family that would have graded
emission realism was measured at a +2.0% ceiling against the mean's +14.5% and rejected on
that basis, and the ruling was that anything grading emission realism has to make that case
on its own evidence. So these ship as diagnostics, and their split-half ceiling and
generic-response baseline on the three official validation contexts are what an enrolment
discussion would start from -- not this docstring.

B4 measures the SHAPE of the co-expression among responding genes against the other side (a
[k, k] correlation matrix per side: does gene i move with gene j), which a pseudobulk mean
cannot see. A prediction can match the real per-gene marginals exactly with every gene
independent, and it can get the covariance pattern right at the wrong overall scale.
"""

from __future__ import annotations

import logging

import numpy as np

from ..cells_prep import prep_for
from ..gpu import _release_gpu_pool, resolve_device, xp_for

logger = logging.getLogger(__name__)

_CACHE_KEY = "cell_cov_table"

#: Below this per-gene SD a lognorm column counts as CONSTANT and leaves the gene set.
#: ⚠️ NOT `> 0`: the cell blocks are float32, and `std` over a column whose values are all
#: bit-identical returns 2.4e-07 rather than 0 (measured on a 300-cell constant block --
#: the mean subtraction rounds). An exact-zero test therefore let a genuinely constant
#: prediction through to `np.corrcoef`, which divided by that round-off and returned a
#: correlation built out of floating-point noise. Lognorm values are O(1) and a gene
#: expressed in even 2% of cells has an SD near 0.3, so 1e-6 is orders of magnitude clear
#: of anything real while sitting well above the round-off.
_STD_FLOOR = 1e-6

#: The same idea one level up, for the upper triangle of a correlation matrix. Entries are
#: in [-1, 1], and `np.corrcoef` on perfectly collinear genes returns 1.0 +/- ~1e-16, so a
#: constant triangle's SD is ~1e-16 and not 0. A triangle with no spread carries no
#: information to correlate against, and Pearson on it is 0/0.
_TRIANGLE_STD_FLOOR = 1e-10


def responding_genes(
    prep, pert: str, *, top_genes: int, min_nonzero_cells: int,
) -> np.ndarray:
    """The responding-gene columns for ``pert``: the ``top_genes`` genes of ``G_p`` with the
    largest ``|log2FC_real|`` that at least ``min_nonzero_cells`` real cells of ``p`` express.

    The expression filter comes FIRST, before the ranking, and it is what keeps the set
    interpretable: on a CP10K pseudobulk with pseudocount 1 a gene seen in three cells can
    carry a large fold change, and a correlation estimated across cells that are almost all
    zero is noise with a defined value. Ranking is by ``|log2FC|`` on the REAL side only
    (spec §4 B4), so the scored gene set cannot be chosen by the submission.

    Ties in ``|log2FC|`` break by column index (stable argsort over an ascending candidate
    array), so the set is a function of the data and not of the panel's row order.
    """
    scored = prep.scored_genes(pert)
    expressed = prep.stats("real", pert).nonzero_cells[scored] >= int(min_nonzero_cells)
    candidates = scored[expressed]
    if candidates.size == 0:
        return candidates
    absfc = np.abs(prep.real_log2fc(pert)[candidates])
    order = np.argsort(-absfc, kind="stable")
    return np.sort(candidates[order[:int(top_genes)]])


def _upper(matrix: np.ndarray) -> np.ndarray:
    """The strict upper triangle of a square matrix, flattened."""
    i, j = np.triu_indices(matrix.shape[0], k=1)
    return matrix[i, j]


def _corr_matrix(block: np.ndarray) -> np.ndarray:
    """Gene-gene Pearson correlation of a [cells, genes] lognorm block, as [genes, genes]."""
    # rowvar=False: variables are the COLUMNS (genes). np.corrcoef would emit a
    # divide-by-zero warning on a zero-variance gene; those columns are dropped by the
    # caller before this is reached, so the errstate is a belt-and-braces silence rather
    # than a mask over an expected condition.
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.corrcoef(np.asarray(block, dtype=np.float64), rowvar=False)


def cov_agreement(real_block: np.ndarray, pred_block: np.ndarray) -> dict[str, float]:
    """``{'corr': ..., 'eig_ratio': ...}`` for one perturbation's two lognorm cell blocks.
    Both blocks are [n, k] over the SAME genes in the same order and at the same n (see
    ``cells_prep.CellsPrep.pair`` on why the depths are equalized). Returns NaN for either
    entry that is undefined rather than a plausible number: a constant upper triangle has no
    correlation to report, and a non-positive ``λ1(R_real)`` has no ratio to take.
    """
    r_real, r_pred = _corr_matrix(real_block), _corr_matrix(pred_block)
    u_real, u_pred = _upper(r_real), _upper(r_pred)

    finite = np.isfinite(u_real) & np.isfinite(u_pred)
    if (finite.sum() < 2 or np.std(u_real[finite]) <= _TRIANGLE_STD_FLOOR
            or np.std(u_pred[finite]) <= _TRIANGLE_STD_FLOOR):
        corr = float("nan")
    else:
        corr = float(np.corrcoef(u_real[finite], u_pred[finite])[0, 1])

    # eigvalsh, not eig: a correlation matrix is symmetric, so the Hermitian solver is both
    # faster and free of the spurious imaginary parts the general one returns.
    ratio = float("nan")
    if np.isfinite(r_real).all() and np.isfinite(r_pred).all():
        l1_real = float(np.linalg.eigvalsh(r_real)[-1])
        l1_pred = float(np.linalg.eigvalsh(r_pred)[-1])
        if l1_real > 0:
            ratio = l1_pred / l1_real
    return {"corr": corr, "eig_ratio": ratio}


def _cov_table(prep, *, top_genes: int, min_genes: int, min_cells: int,
               min_nonzero_cells: int) -> dict[str, dict[str, float]]:
    """``{pert -> {'corr', 'eig_ratio'}}`` for every scorable perturbation.

    Perturbations that cannot support the estimate are OMITTED rather than filled: fewer
    than ``min_cells`` cells on a side, or fewer than ``min_genes`` genes surviving the
    responding-gene filters. ``agg='mean'`` skips NaN and a missing key alike, and both
    metrics are unscored, so an omission costs no score -- while a filled-in number would
    be a correlation this data cannot support.
    """
    out: dict[str, dict[str, float]] = {}
    for pert in prep.perts:
        n = prep.depth(pert)
        if n < int(min_cells):
            logger.debug("cell_cov: %s skipped, common depth %d < %d", pert, n, min_cells)
            continue
        cols = responding_genes(prep, pert, top_genes=top_genes,
                                min_nonzero_cells=min_nonzero_cells)
        if cols.size < int(min_genes):
            logger.debug("cell_cov: %s skipped, %d responding genes < %d",
                         pert, cols.size, min_genes)
            continue
        pair = prep.pair(pert, cols)
        # A gene with no variance on EITHER side has no correlation with anything; drop it
        # here (after sampling, since variance is a property of the cells actually drawn)
        # and re-apply the gene floor to the surviving set.
        # float64 for the SD: the blocks are float32 and a float32 variance over a
        # constant column does not round to zero (see _STD_FLOOR).
        real_sd = np.asarray(pair.real, dtype=np.float64).std(axis=0)
        pred_sd = np.asarray(pair.pred, dtype=np.float64).std(axis=0)
        keep = (real_sd > _STD_FLOOR) & (pred_sd > _STD_FLOOR)
        if int(keep.sum()) < int(min_genes):
            logger.debug("cell_cov: %s skipped, %d genes varied on both sides < %d",
                         pert, int(keep.sum()), min_genes)
            continue
        out[str(pert)] = cov_agreement(pair.real[:, keep], pair.pred[:, keep])
    return out


def cov_metric(
    *,
    which: str,
    pred_source,
    real_source,
    _cell_metric_cache: dict,
    pert_col: str = "target",
    control: str = "non-targeting",
    genes=None,
    target_gene_map: dict[str, str] | None = None,
    input_type_pred: str = "counts",
    input_type_real: str = "counts",
    max_cells: int = 500,
    seed: int = 0,
    top_genes: int = 100,
    min_genes: int = 20,
    min_cells: int = 50,
    min_nonzero_cells: int = 10,
    _validate_cells=None,
) -> dict[str, float]:
    """One member of the B4 family, per perturbation.

    ``which`` is ``'corr'`` (``cell_cov_corr``) or ``'eig_ratio'``
    (``cell_cov_eig_ratio``); bound per catalog entry via ``functools.partial``. Both
    members share ONE table -- the correlation matrices are computed once per dispatch and
    both numbers read off them -- through ``_cell_metric_cache``.

    ⚠️ The knobs are function defaults rather than ``EvalConfig`` fields. That is not a style
    preference: ``cache.config_hash`` digests
    every non-skipped ``EvalConfig`` field, ``competition.rule_config_hash`` is that hash,
    and ``competition_payload()`` embeds it -- so adding a ``cells:`` section to the config
    MOVES ``competition_digest()`` (measured: it does) and breaks the frozen digest that
    certifies two bundles as mutually comparable. An unscored diagnostic must not be able
    to do that. Override them by naming the metrics through a partial, or lift them to the
    config in a release that bumps the rule deliberately.
    """
    if which not in ("corr", "eig_ratio"):
        raise ValueError(f"which must be 'corr' or 'eig_ratio', got {which!r}")
    # Every knob that changes the table, `prep_for`'s two included: a second call on the same
    # cache with a different `max_cells` or `seed` must not be served the first call's table.
    key = (_CACHE_KEY, max_cells, seed, top_genes, min_genes, min_cells, min_nonzero_cells)
    table = _cell_metric_cache.get(key)
    if table is None:
        prep = prep_for(
            _cell_metric_cache, pred_source=pred_source, real_source=real_source,
            pert_col=pert_col, control=control, genes=genes,
            target_gene_map=target_gene_map, input_type_pred=input_type_pred,
            input_type_real=input_type_real, max_cells=max_cells, seed=seed,
            validate_cells=_validate_cells,
        )
        table = _cell_metric_cache[key] = _cov_table(
            prep, top_genes=top_genes, min_genes=min_genes, min_cells=min_cells,
            min_nonzero_cells=min_nonzero_cells,
        )
    return {p: vals[which] for p, vals in table.items()}


# ── cell_cov_ccc: Lin's CCC over the ALL-GENES gene-gene correlation matrix ────
#
# Sibling of the B4 ``cell_cov_corr`` above, differing on three deliberate axes (chosen
# over the responding-gene Pearson version after a side-by-side of the two):
#
#   * AGREEMENT STAT -- Lin's concordance correlation coefficient, not Pearson. CCC is
#     ``2·cov / (var_a + var_b + (mean_a - mean_b)^2)``: a prediction whose correlation
#     PATTERN is right but systematically weaker (the signature of per-gene-independent
#     sampling from a fitted marginal) reads near 1.0 on Pearson and is penalised by CCC,
#     because CCC is agreement about the 45-degree line, not about any line.
#   * GENE SET -- all genes (no responding-gene / top-|log2FC| restriction, no target-gene
#     exclusion). This asks whether WHOLE-transcriptome co-expression structure is right, a
#     different hypothesis from "did the genes that moved co-vary correctly".
#   * SIGNIFICANCE GATE -- by default the CCC is taken only over gene pairs significantly
#     correlated on BOTH sides (``|r| >= r_crit`` at the sampled n, raw alpha=0.05; the
#     "intersection" mode of the bench prototype's sweep). Most of a G² matrix is sampling
#     noise around zero; scoring the shared SIGNIFICANT pairs is what makes the number track
#     real co-expression rather than noise agreement -- measured to ~double whole-matrix
#     discrimination (ceiling − control) where structure is weak (NGN2), and ~neutral where it
#     is already strong (sci0048 T-cell cytokines). ``significance='none'`` recovers the
#     un-gated all-pairs CCC.
#
# The all-genes matrix is ~G×G with G≈18k on a real panel (~1.3 GB float32), so this runs
# on the shared ``gpu`` abstraction (cupy on CUDA, numpy fallback). The symmetric off-
# diagonal is used in place of ``triu_indices`` -- a correlation matrix is symmetric and
# its diagonal is dropped, so moments over all finite off-diagonal entries equal moments
# over the strict upper triangle, and we avoid materialising two G²/2 index arrays
# (~2.6 GB at G=18k) that the triangle extraction would otherwise cost.
#
# UNSCORED (``DIAG``), ``cell`` profile -- the §6a rule that governs the whole family: a
# metric grading emission realism earns enrolment on its own measured ceiling/baseline.

_CCC_CACHE_KEY = "cell_cov_ccc_table"

#: Lin's-CCC denominator floor, mirroring the bench prototype this is ported from. Guards
#: the 0/0 when two identical constant matrices are compared (both variances and the mean
#: gap are zero); any real pair clears it by many orders of magnitude.
_CCC_EPS = 1e-12

#: The significance gates cell_cov_ccc can apply before the CCC (see ``ccc_all_genes``).
CCC_SIGNIFICANCE = ("intersection", "none")


def _r_crit(n: int, n_pairs: float, *, alpha: float, correction: str) -> float:
    """Critical |r| for a two-sided Pearson test across ``n`` cells (ported from the bench
    ``gene_corr_ceiling.r_crit``).

    At fixed ``n`` the Pearson t statistic ``t = r·sqrt((n-2)/(1-r^2))`` is monotone in ``|r|``,
    so "significantly correlated, either sign" is exactly ``|r| >= r_crit`` -- one cell-count-
    adaptive magnitude cut. ``correction='bonferroni'`` divides ``alpha`` by ``n_pairs`` (~G²/2);
    ``'none'`` leaves it raw. Bonferroni over ~1e8 pairs keeps <0.1% of pairs and empties the
    metric, so ``none`` is the shipped default (see the family's measured sweep)."""
    from scipy.stats import t as _student_t

    df = max(int(n) - 2, 1)
    a = alpha / max(n_pairs, 1.0) if correction == "bonferroni" else alpha
    tc = float(_student_t.ppf(1.0 - a / 2.0, df))
    return tc / float(np.sqrt(df + tc * tc))


def _device_corr(block: np.ndarray, xp):
    """Gene-gene Pearson correlation across cells of a [n, G] lognorm block, on ``xp``.

    Returns the [G, G] matrix with the diagonal and every row/column of a zero-variance
    gene set to NaN -- so a downstream finite-mask drops both self-correlation and genes
    that cannot be correlated. ``xp`` is numpy (CPU) or cupy (CUDA); the block is moved to
    the device here.
    """
    # Centre and take the SD in float64, as `_cov_table` does: a float32 mean of bit-identical
    # values does not subtract to zero, so a CONSTANT column read an SD up to 2.5e-5 (> _STD_FLOOR)
    # and a constant prediction scored a finite CCC built from round-off. The [n, G] block is
    # small; only the standardized copy is cast to float32 for the G x G product.
    X = xp.asarray(block, dtype=xp.float64)
    n = X.shape[0]
    mu = X.mean(0, keepdims=True)
    Xc = X - mu
    sd = xp.sqrt((Xc * Xc).mean(0, keepdims=True))          # (1, G) population sd
    valid = sd[0] > _STD_FLOOR
    sd_safe = sd.copy()
    sd_safe[sd <= _STD_FLOOR] = 1.0
    Xs = (Xc / sd_safe).astype(xp.float32)
    C = (Xs.T @ Xs) / n                                     # (G, G) float32 correlation
    G = C.shape[0]
    bad = ~valid
    C[bad, :] = xp.nan
    C[:, bad] = xp.nan
    diag = xp.arange(G)
    C[diag, diag] = xp.nan                                  # drop self-correlation
    return C


def ccc_all_genes(
    real_block: np.ndarray, pred_block: np.ndarray, *, device: str = "auto",
    significance: str = "intersection", alpha: float = 0.05, correction: str = "none",
) -> float:
    """Lin's CCC between the real and predicted ALL-GENES gene-gene correlation matrices.

    Both blocks are [n, G] lognorm over the SAME genes at the same n (``CellsPrep.pair``). The CCC
    is taken over the kept off-diagonal entries present on BOTH matrices -- which equals the strict-
    upper-triangle CCC because a correlation matrix is symmetric (see the section comment).

    ``significance`` selects which pairs are kept:
      * ``'intersection'`` (default) -- only pairs whose correlation is significantly nonzero on
        BOTH sides (``|r| >= r_crit`` at this ``n``, ``alpha``, ``correction``). This focuses the
        CCC on the co-expression structure both sides actually resolve; measured to roughly double
        whole-matrix discrimination (ceiling − control) where structure is weak, and to be ~neutral
        where it is already strong. The shipped default -- the metric scores the shared SIGNIFICANT
        structure, not every sampling-noise pair.
      * ``'none'`` -- every jointly-finite pair (the un-gated all-pairs CCC).

    Returns NaN when fewer than three pairs survive (e.g. a structureless prediction shares almost
    no significant pairs with the real side) -- ``agg='mean'`` skips it and the metric is unscored.
    """
    if significance not in CCC_SIGNIFICANCE:
        raise ValueError(f"significance must be one of {CCC_SIGNIFICANCE}, got {significance!r}")
    xp = xp_for(resolve_device(device))
    A = _device_corr(real_block, xp)
    B = _device_corr(pred_block, xp)
    m = xp.isfinite(A) & xp.isfinite(B)
    if significance == "intersection":
        G = A.shape[0]
        rc = _r_crit(int(real_block.shape[0]), G * (G - 1) / 2.0,
                     alpha=alpha, correction=correction)
        # `|NaN| >= rc` is False, so this also keeps the finite/off-diagonal restriction.
        m = m & (xp.abs(A) >= rc) & (xp.abs(B) >= rc)
    # The mask is symmetric, so every gene pair is counted twice: three pairs are six entries.
    cnt = float(m.sum())
    if cnt < 2 * 3:
        return float("nan")
    # Moments over the kept entries only: compacting them to 1-D first means no further G x G
    # temporary (the masked-`where` form held five), and under the default gate the kept set
    # is a small fraction of G^2 (Gemini, #395). Sums accumulate in float64; the deviations
    # stay float32.
    a, b = A[m], B[m]
    del A, B, m
    mA = float(a.sum(dtype=xp.float64) / cnt)
    mB = float(b.sum(dtype=xp.float64) / cnt)
    dA = a - xp.float32(mA)
    dB = b - xp.float32(mB)
    vA = float((dA * dA).sum(dtype=xp.float64) / cnt)
    vB = float((dB * dB).sum(dtype=xp.float64) / cnt)
    cov = float((dA * dB).sum(dtype=xp.float64) / cnt)
    return 2.0 * cov / (vA + vB + (mA - mB) ** 2 + _CCC_EPS)


def _ccc_table(prep, *, min_cells: int, device: str,
               significance: str, alpha: float, correction: str) -> dict[str, float]:
    """``{pert -> cell_cov_ccc}`` for every scorable perturbation.

    Scores over ALL genes with no gene set to resolve -- NOT ``scored_genes``. The perturbed gene
    is deliberately NOT excluded: this is a whole-transcriptome co-expression agreement over ~18k
    genes, so one gene is negligible and the target gene's own correlations are legitimate
    structure (unlike pds_*/DE, where the target is a label-predictable coordinate that must be
    dropped). That removes the ``target_gene_map`` dependency entirely -- the metric needs nothing
    but the two archives. By default (``significance='intersection'``) only pairs significantly
    correlated on BOTH sides enter the CCC; zero-variance and non-significant pairs drop out.
    Reading only the ≤``max_cells`` cells :meth:`pair` samples per side, a streaming source
    (``CellStoreSource``) touches no control pool and no full group. A perturbation is OMITTED (no
    key) below ``min_cells`` common depth, and is NaN when fewer than three pairs survive the gate;
    ``agg='mean'`` skips both and the metric is unscored, so neither costs a score.
    The GPU caching pool is released after each perturbation so one pert's ~G×G matrices do not
    accumulate across the loop.
    """
    out: dict[str, float] = {}
    for pert in prep.perts:
        n = prep.depth(pert)
        if n < int(min_cells):
            logger.debug("cell_cov_ccc: %s skipped, common depth %d < %d", pert, n, min_cells)
            continue
        pair = prep.pair(pert)                       # all genes, no exclusion
        out[str(pert)] = ccc_all_genes(pair.real, pair.pred, device=device,
                                       significance=significance, alpha=alpha,
                                       correction=correction)
        _release_gpu_pool()
    return out


def cov_ccc_metric(
    *,
    pred_source,
    real_source,
    _cell_metric_cache: dict,
    pert_col: str = "target",
    control: str = "non-targeting",
    genes=None,
    target_gene_map: dict[str, str] | None = None,
    input_type_pred: str = "counts",
    input_type_real: str = "counts",
    max_cells: int = 500,
    seed: int = 0,
    min_cells: int = 50,
    significance: str = "intersection",
    alpha: float = 0.05,
    correction: str = "none",
    device: str = "auto",
    _validate_cells=None,
) -> dict[str, float]:
    """``cell_cov_ccc``, per perturbation: Lin's CCC over the all-genes gene-gene
    correlation matrices of the real and predicted cells, restricted by default to the pairs
    significantly correlated on BOTH sides (``significance='intersection'``, raw ``alpha=0.05``).

    Shares the run's :class:`~cell_eval2.cells_prep.CellsPrep` (same cells and depth
    equalisation as ``cell_cov_corr``) through ``_cell_metric_cache``;
    the per-pert CCC table is itself memoised there, keyed on the knobs that change it. The
    knobs are function defaults rather than ``EvalConfig`` fields for the same reason the rest
    of the family's are -- see ``cov_metric`` -- so this unscored diagnostic cannot move the
    frozen competition digest. Pass ``significance='none'`` for the un-gated all-pairs CCC.
    """
    key = (_CCC_CACHE_KEY, max_cells, seed, min_cells, significance, alpha, correction,
           str(device))
    table = _cell_metric_cache.get(key)
    if table is None:
        prep = prep_for(
            _cell_metric_cache, pred_source=pred_source, real_source=real_source,
            pert_col=pert_col, control=control, genes=genes,
            target_gene_map=target_gene_map, input_type_pred=input_type_pred,
            input_type_real=input_type_real, max_cells=max_cells, seed=seed,
            validate_cells=_validate_cells,
        )
        table = _cell_metric_cache[key] = _ccc_table(
            prep, min_cells=min_cells, device=device,
            significance=significance, alpha=alpha, correction=correction)
    return dict(table)
