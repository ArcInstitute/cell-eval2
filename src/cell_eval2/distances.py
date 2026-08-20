from __future__ import annotations

import logging

import numpy as np
from scipy.spatial.distance import cdist

logger = logging.getLogger(__name__)


def resolve_exclusion_columns(
    labels,
    genes,
    *,
    target_gene_map: dict[str, str] | None = None,
    gate_labels=None,
    who: str = "exclude_target_gene=True",
    escape: str = ("or pass exclude_target_gene=False to score without exclusion "
                   "deliberately."),
) -> dict[int, int]:
    """``{row index -> gene column}`` for each perturbation whose OWN gene is measured.

    The single definition of "which column is this perturbation's target gene", shared by
    the CPU (:func:`cell_eval2.metrics.discrimination.discrimination_score`) and GPU
    (:func:`cell_eval2.gpu.distances._discrimination_ranks_xp`) discrimination kernels.
    It lives here, and not once in each of them, because the two are separately
    implemented and the bug this function exists to close (issue #248) was precisely a
    lookup that was wrong in both.

    Resolution order per label mirrors :func:`cell_eval2.de.resolve_target_genes` exactly,
    so ``pds_*`` and the eleven chance-corrected DE metrics agree on what "the target
    gene" means inside one scoring run:

      1. ``target_gene_map[label]`` if present -- AUTHORITATIVE, the construct-ID escape
         hatch (``'ADNP-1'`` -> ``'ADNP'``).
      2. the raw label matched against the gene panel, so gene-level panels whose labels
         ARE symbols keep resolving with no map and no score change.

    A label that resolves to nothing is omitted and reported. But ZERO resolution raises: that is
    the construct-ID-vs-symbol mismatch, where the caller asked for exclusion, nothing was
    excluded, and the score silently changed meaning. Same placement and same zero-resolve rule as
    ``metrics.direction._require_resolution``, for the same reason -- a metric whose meaning
    depends on label format, with no signal to the caller, is the worst failure mode available to
    it.

    ⚠️ **The partial case is tolerated but is NOT known to be benign (issue #289).** The old report
    sat at INFO and asserted the harmless reading -- "its own gene is simply not measured, ordinary
    biology, or the CPM filter dropped it" -- of a condition it never tested. Two structurally
    different states share this path and this function cannot tell them apart:

      (a) the label does not resolve AND the gene is genuinely absent from the panel -> there is
          nothing to exclude, and omitting it is right;
      (b) the label does not resolve BUT the gene is present under a different string -> the column
          stays in the ranked vector, and it is a coordinate predictable from the label alone.

    (b) IS the mismatch the raise was written for; it simply stops counting as one once any single
    label resolves, because the gate is an all-quantifier. MEASURED on a 40-target uniform
    construct-ID panel against a zero-DOWNSTREAM-knowledge model -- off-target biology drawn
    independently of the truth, but the on-target knockdown reproduced -- whose correct
    ``pds_cosine`` is 0.7808: with a map covering 30 of 40 the panel mean reads 0.8468, and every
    *unresolved* target reads exactly 1.0000 while resolved ones read 0.7885. The same genes
    genuinely absent from the panel give 0.7000, i.e. no leak. So the tolerance is right for (a)
    and wrong for (b).

    ⚠️ 1.0000 is what that model gets, not a universal. The leaked coordinate is only decisive for
    a submission that predicts the knockdown; one with no effect at all ties every real target at
    cosine distance 1 and scores 0.5 under v2 midrank whether or not its gene was excluded. The
    warning says "can inflate" for that reason (codex review).

    What is done about it here is the REPORTING, not the gate: the line is a WARNING, it names how
    many labels are affected, it states what an unresolved target actually gets, and it says
    outright that this function cannot separate (a) from (b) -- so the operator is told to check
    the named labels against ``var_names`` rather than being told the miss was benign.

    Separating (a) from (b) automatically was tried and dropped. "Unresolved AND absent from a
    ``target_gene_map`` that is in use" looks like the map gap #289's repro is built from, and on
    the metric path it would be one -- ``discrimination_score`` passes ``pred_keys``, which
    ``prep.delta`` has already stripped of the control. But a DIRECT caller may pass the control
    label in with the perturbations (``tests/`` do), and the control never resolves because it has
    no target gene, so it would show up as a spurious map gap on exactly the calls that have no
    dispatcher to protect them. The resolver is not told which label is the control and cannot
    exclude it without a signature change -- and that signature is the contract ``baseline.py``'s
    call site (#253/#285) resolves against. The warning therefore states the ambiguity instead of
    guessing at it.

    ⚠️ **RULED (Alex, 2026-08-16): warn only. The gate does NOT move.** A raise needs a rule, and
    all three candidates are unsound. Rate-based needs a threshold, and the harm is continuous --
    one unresolved target in 40 is +0.00128 ``pds_cosine``, 25% unmapped is +0.066 -- so no value
    is principled; worse, the rate does not measure whether the misses are (a) or (b), and a small
    panel can legitimately have half its targets unmeasured. "Raise when a ``target_gene_map`` is
    supplied but incomplete" false-positives on a MIXED panel, where a symbol-labelled target whose
    gene is not measured is neither mapped nor directly resolvable -- the documented benign case --
    as well as on the control-label route above. A ``strict`` opt-in on ``exclude_target_gene``
    would be sound, but it is a ``config.DiscriminationParams`` field and so a different owner.

    What settled it: on all three official competition contexts **200 of 200 targets resolve**
    against the 18,533-gene panel, with ``target_gene_map: null`` -- the labels already ARE gene
    symbols. ``filter_gene_min_cpm_cell`` is DE-only (``de_compute.py``) and never shrinks the panel
    this kernel sees, so it cannot make a target unresolvable either. This warning does not fire on
    the competition path at all, so a raise would buy nothing there while being able to hard-fail a
    legitimate small or heavily-filtered research panel.

    The return value and the zero-resolve raise are UNCHANGED here.

    ⚠️ **The zero-resolve GATE is global; the RESOLUTION is per row.** ``gate_labels`` is the
    label set the raise is judged on -- pass the REAL (reference) keys, which every driver
    carries whole. ``labels`` may be a SUBSET: ``scale.py`` restricts the pred bulks to a
    shard (``_restrict(pred_bulks, chosen | {control})``) and ``partition_inmem.py`` passes
    one piece at a time, while both hand the real bulks over unrestricted. Gating on
    ``labels`` would make the raise depend on how the data happened to be chunked -- a panel
    that resolves 2 of 3 targets scores whole, then hard-fails on the shard that happens to
    hold only the unresolved one. The panel either has the construct-ID mismatch or it does
    not; that is a property of the reference, not of the partition.

    ``gate_labels=None`` gates on ``labels`` itself, which is right for a direct call where
    the caller has the whole panel in hand.

    ``genes`` must already be validated (unique, and matching the feature dimension); the
    callers do that before calling here, where the errors can name the right operand.

    ⚠️ That contract is now BACKSTOPPED below, because it turned out to be forgettable: #172
    added a fourth caller (``metrics.delta._exclusion_cols``) and it shipped without the
    uniqueness check its three siblings carry. ``gene_pos`` is a dict keyed on the gene label,
    so a duplicated label silently keeps only the LAST column and the exclusion drops the wrong
    coordinate -- measured at 34.0 where the correct answer was 1.0 (Copilot, PR #316, in the
    suppressed block). The callers keep their own checks, which fire first and name their own
    operands; this one exists so that a caller which forgets gets an ERROR rather than a wrong
    number. O(G) once per metric call -- a dict build plus a `len` comparison, and the dict is built
    above regardless, against a metric that pseudobulks millions of cells. (⚠️ It said O(G log G)
    first: that is `np.unique`'s cost, which is what the CALLERS use, not this check's. Copilot,
    PR #316.)

    ⚠️ ``who`` and ``escape`` exist because issue #172 gave this resolver a SECOND family of
    callers: the two legs of ``expr_mse_unbiased_capped_norm`` (``metrics.delta``). Those have
    no ``exclude_target_gene`` knob -- exclusion is part of the metric, as it is for the eleven
    chance-corrected DE metrics -- so the zero-resolve message must not open by asserting a flag
    they do not have, nor close by advising the caller to unset it. The defaults reproduce the
    discrimination wording verbatim, so ``pds_*``'s error is unchanged.
    """
    gene_pos = {str(g): k for k, g in enumerate(genes)}
    # The backstop the docstring describes. Compared against `gene_pos` rather than via
    # `np.unique(genes)` for two reasons, and NOT for the one I first wrote down: I claimed
    # np.unique would miss a collision this catches (labels differing only by np.str_-vs-str),
    # and that is FALSE -- measured, np.unique sees `['A', np.str_('A')]` as a duplicate too
    # (Gemini, PR #316). The real reasons are that this comparison is FREE, since `gene_pos` is
    # built above whatever happens, where `np.unique` on a 20000-label string panel costs 8.6 ms;
    # and that `gene_pos` is by definition the object that decides which column gets dropped, so
    # checking it cannot drift from what the lookup actually does.
    if len(gene_pos) != len(genes):
        raise ValueError(
            f"{who}: duplicate gene names in `genes` are not supported -- "
            f"{len(genes) - len(gene_pos)} duplicate gene name(s) found. Resolving each "
            "perturbation's own gene needs a unique gene->column mapping, and a duplicated label "
            "silently resolves to the LAST matching column, so the wrong coordinate would be "
            "excluded with no error at all. Deduplicate the var index (e.g. "
            f"AnnData.var_names_make_unique()), {escape}"
        )
    # str-keyed both sides: `genes` may hold np.str_ or object dtype, and a YAML-loaded
    # map may hold anything str()-able. Matching the CPU path's existing str(p) lookup.
    mapping = {str(k): str(v) for k, v in (target_gene_map or {}).items()}

    cols: dict[int, int] = {}
    n_unresolved = 0
    sample: list[str] = []  # capped: only 5 are ever reported, and a CCL-scale panel
                            # would otherwise build a full-length list to slice [:5] off
    for i, label in enumerate(labels):
        key = str(label)
        # A mapped label that names a gene absent from the panel is UNRESOLVED here, not
        # authoritative-and-done as it is on the DE side: there is no column to drop, so
        # nothing can be excluded. Counting it as resolved would re-open the silent hole
        # for a map that is present but wrong.
        col = gene_pos.get(mapping.get(key, key))
        if col is None:
            n_unresolved += 1
            if len(sample) < 5:
                sample.append(key)
        else:
            cols[i] = col

    # THE GATE, judged on the whole panel (`gate_labels`), never on this call's slice.
    # Resolving the gate set separately costs one dict lookup per reference label and buys
    # partition-independence: the raise now fires on the construct-ID mismatch and nothing
    # else. When gate_labels is None (or IS labels) this collapses to the counts above.
    if gate_labels is None or gate_labels is labels:
        n_gate_resolved, n_gate, gate_sample = len(cols), len(cols) + n_unresolved, sample
    else:
        n_gate = 0
        n_gate_resolved = 0
        # Sampled from the GATE set, not from this call's slice. The raise is a statement
        # about the panel, so the labels it names have to come from the panel -- quoting the
        # slice would print an empty or unrepresentative sample on exactly the shard case this
        # gate exists to handle (Copilot, PR #250).
        gate_sample: list[str] = []
        for label in gate_labels:
            n_gate += 1
            key = str(label)
            if gene_pos.get(mapping.get(key, key)) is not None:
                n_gate_resolved += 1
            elif len(gate_sample) < 5:
                gate_sample.append(key)

    n_labels = n_gate
    if n_gate and not n_gate_resolved:
        raise ValueError(
            f"{who} but NO perturbation resolves to a gene in the "
            f"feature panel: 0 of {n_labels} labels matched any of {len(gene_pos)} "
            f"genes. This is the construct-ID-vs-gene-symbol mismatch (e.g. "
            f"perturbation 'GENEX-1' vs gene 'GENEX'). Scoring on would exclude nothing "
            f"and return a plausible wrong number: every perturbation would keep its own "
            f"transcript in the scored vector, a coordinate predictable from the label "
            f"alone, which CAN inflate its score -- decisively for a submission "
            f"that predicts the knockdown, not at all for one with no effect. Unresolved sample: "
            f"{gate_sample}. Supply EvalConfig.target_gene_map={{perturbation: gene}} "
            f"to map construct IDs to their genes (the same map the DE metrics use), "
            f"{escape}"
        )
    if n_unresolved:
        # Denominator is THIS call's label count, not the gate set's -- mixing a local
        # numerator with a global denominator would misreport a shard as mostly unresolved.
        #
        # WARNING, not INFO (#289). What this reports is that some targets are scored WITHOUT
        # the exclusion the caller asked for, which is a change in what the metric means for
        # them. The wording no longer asserts the benign reading of a condition this function
        # cannot test, and it no longer says 1.0000 is what EVERY submission gets there -- that
        # figure belongs to a model reproducing the knockdown; a zero-effect prediction ties
        # every real target and midranks to 0.5 either way (codex review round 2).
        logger.warning(
            "target-gene exclusion: %d/%d labels resolved to a measured gene. The other %d are "
            "scored WITHOUT exclusion -- each keeps its own transcript in the ranked vector, a "
            "coordinate predictable from the label alone, which CAN inflate its score (measured "
            "on #289's panel: unresolved targets read pds_cosine 1.0000 against a model that "
            "reproduces the knockdown, where resolved ones read 0.7885). That "
            "is harmless ONLY if those genes are genuinely absent from the panel; if any is "
            "present under a different string it is the #248 leak on that subset (issue #289), "
            # "the gene panel passed in" rather than bare "var_names" (Copilot review): this
            # resolver takes a `genes` array and holds no AnnData, so a DIRECT caller has no
            # `var_names` to check against.
            "and this function cannot tell the two apart. Check them against the gene panel "
            "passed in (an AnnData caller's var_names), and "
            "supply EvalConfig.target_gene_map={perturbation: gene} for any that are present. "
            "Unresolved sample: %s",
            len(cols), len(cols) + n_unresolved, n_unresolved, sample,
        )
    return cols


def resolve_panel_columns(
    panel_labels,
    genes,
    *,
    target_gene_map: dict[str, str] | None = None,
    who: str = "exclude_target_gene=True with exclusion_scope='panel'",
    escape: str = ("or pass exclude_target_gene=False to score without exclusion "
                   "deliberately."),
) -> np.ndarray:
    """The sorted gene columns of EVERY target on the panel (issue #343).

    The ``exclusion_scope="panel"`` counterpart of :func:`resolve_exclusion_columns`, and a
    thin one on purpose: it resolves through that function so the two scopes cannot drift on
    what "this perturbation's target gene" means. Resolution order, the zero-resolve raise
    and the partial-resolution warning are therefore identical, and shared with the DE side's
    ``de.resolve_target_genes``.

    ``panel_labels`` must be the REFERENCE's non-control perturbations, never the
    prediction's. The excluded feature set is a property of the panel, so a prediction that
    arrives as a shard (``scale.py`` restricts the pred bulks; ``partition_inmem.py`` passes
    one piece at a time) is scored in the SAME feature space as the whole -- which is what
    keeps the partition-parity tests exact. It is also why no ``gate_labels`` argument is
    offered here: the labels passed in already ARE the gate set.

    Duplicate target labels collapse: two perturbations of one gene name one column, and the
    returned array is unique and sorted so the column set is a canonical, order-independent
    identity of the reduced space.
    """
    cols = resolve_exclusion_columns(
        panel_labels, genes, target_gene_map=target_gene_map, who=who, escape=escape,
    )
    return np.unique(np.fromiter(cols.values(), dtype=np.intp, count=len(cols)))


def panel_reduced(
    eff: np.ndarray, panel_cols: np.ndarray, *, who: str = "exclude_target_gene=True"
) -> np.ndarray:
    """``eff`` with the panel's target-gene columns removed, for ``exclusion_scope='panel'``.

    Raises when that leaves NO features. An empty feature space is not a degenerate score to
    be reported -- every row's norm is zero, every cosine similarity falls to the zero-norm
    convention, and the whole matrix ties at distance 1, so the metric returns exactly 0.5 for
    every perturbation no matter what was submitted. That is silence dressed as a number, and
    it is reachable on a small research panel whose perturbations cover the measured genes,
    never on the competition panel (300 targets, 18,533 genes).

    ``panel_cols`` is non-empty by contract: the only caller resolves it through
    :func:`resolve_panel_columns`, whose zero-resolve gate RAISES before this function is
    reached. It is therefore deliberately not special-cased -- an empty ``panel_cols`` would
    make the mask all-True and return an ordinary full-width copy, which is the right answer
    for a hypothetical direct caller and one no production path can ask for. (Gemini suggested
    short-circuiting that case to return ``eff`` itself; declined in PR #345 because it is
    unreachable through the real caller and would make the return value sometimes a view and
    sometimes a copy, which is the kind of aliasing contract a shared helper should not have.)
    """
    # A boolean mask, not `np.setdiff1d`: O(G) instead of O(G log G), it preserves column
    # order by construction, and it needs NO uniqueness or sortedness precondition on
    # `panel_cols` -- so this helper stays correct for a caller that does not come through
    # `resolve_panel_columns`. MEASURED on an 18,533-gene panel with 300 targets: 1.8 us,
    # against 79.8 us for `setdiff1d(assume_unique=True)` and 2,021 us for the safe
    # `assume_unique=False`. (Copilot and Gemini both flagged the setdiff1d; Gemini proposed
    # `assume_unique=True`, which is correct for today's single caller but buys 44x less and
    # silently returns the wrong columns for a duplicated input. PR #345.)
    keep = np.ones(eff.shape[1], dtype=bool)
    keep[panel_cols] = False
    if not keep.any():
        raise ValueError(
            f"{who} with exclusion_scope='panel' would remove every one of the "
            f"{eff.shape[1]} measured gene(s): each is the target of some perturbation on "
            "the panel, so the ranked feature space is empty and the score would be exactly "
            "0.5 for every perturbation regardless of the submission. Score this panel with "
            "exclusion_scope='row', or with exclude_target_gene=False."
        )
    return eff[:, keep]


def pairwise_to_vector(
    matrix: np.ndarray, vector: np.ndarray, metric: str
) -> np.ndarray:
    """Distance from each row of `matrix` to `vector`, as a 1-D array [n_rows].

    Matches ``sklearn.metrics.pairwise_distances(matrix, vector[None, :],
    metric=metric).flatten()`` for metric in {"l1", "l2", "cosine"}, including the
    cosine zero-norm convention (a zero-norm operand gives similarity 0 ⇒ distance
    1) and the [0, 2] clip. numpy/float64.
    """
    matrix = np.asarray(matrix, dtype=np.float64)
    vector = np.asarray(vector, dtype=np.float64).ravel()
    if metric in ("l1", "manhattan", "cityblock"):
        return np.sum(np.abs(matrix - vector), axis=1)
    if metric in ("l2", "euclidean"):
        return np.sqrt(np.sum((matrix - vector) ** 2, axis=1))
    if metric == "cosine":
        nv = np.linalg.norm(vector)
        nm = np.linalg.norm(matrix, axis=1)
        denom = nm * nv
        sim = np.zeros(matrix.shape[0], dtype=np.float64)
        nonzero = denom != 0
        sim[nonzero] = (matrix[nonzero] @ vector) / denom[nonzero]
        return np.clip(1.0 - sim, 0.0, 2.0)
    raise ValueError(f"unsupported distance metric {metric!r}; use l1, l2, or cosine")


def cosine_distance_from_parts(
    sim: np.ndarray, pred_norm_squares: np.ndarray, real_norm_squares: np.ndarray
) -> np.ndarray:
    """[n_pred, n_real] cosine distance from precomputed parts.

    ``sim`` = ``pred @ real.T``; the norm-squares are ``sum(.**2, axis=1)`` for each
    operand. A zero-norm operand yields similarity 0 ⇒ distance 1; the result is
    clipped to [0, 2]. Sharing this with :func:`pairwise_full` keeps a single
    definition of the cosine convention and lets a caller reuse ``sim``/norms (e.g.
    for the drop-gene correction) without a second ``pred @ real.T`` matmul.
    """
    denom = np.sqrt(pred_norm_squares)[:, None] * np.sqrt(real_norm_squares)[None, :]
    out = np.zeros_like(sim)
    nz = denom != 0
    out[nz] = sim[nz] / denom[nz]  # zero-norm operand -> sim 0 -> distance 1
    return np.clip(1.0 - out, 0.0, 2.0)


def pairwise_full(pred: np.ndarray, real: np.ndarray, metric: str) -> np.ndarray:
    """Full [n_pred, n_real] distance matrix from each `pred` row to each `real` row.

    Matches ``sklearn.metrics.pairwise_distances(pred, real, metric=metric)`` for
    metric in {"l1", "l2", "cosine"}, including the cosine zero-norm convention (a
    zero-norm operand gives similarity 0 ⇒ distance 1) and the [0, 2] clip — the
    same conventions as :func:`pairwise_to_vector`. numpy/float64.

    ``l2``/``cosine`` collapse to a single BLAS matmul (``‖a‖²+‖b‖²−2·ABᵀ`` /
    ``A·Bᵀ``); ``l1`` accumulates per ``pred`` row (copy-free, no per-row temporary
    beyond one ``[n_real]`` vector).
    """
    pred = np.asarray(pred, dtype=np.float64)
    real = np.asarray(real, dtype=np.float64)
    if metric in ("l2", "euclidean"):
        # ||a-b||^2 = ||a||^2 + ||b||^2 - 2 a.b ; clip tiny negatives from roundoff
        d2 = (
            np.einsum("ig,ig->i", pred, pred)[:, None]
            + np.einsum("jg,jg->j", real, real)[None, :]
            - 2.0 * (pred @ real.T)
        )
        return np.sqrt(np.maximum(d2, 0.0))
    if metric == "cosine":
        sim = pred @ real.T
        pred_norm_squares = np.einsum("ig,ig->i", pred, pred)
        real_norm_squares = np.einsum("jg,jg->j", real, real)
        return cosine_distance_from_parts(sim, pred_norm_squares, real_norm_squares)
    if metric in ("l1", "manhattan", "cityblock"):
        # no matmul identity; scipy's C cdist streams over genes without a
        # [n_real, n_genes] temporary (~4x faster + lower memory than a numpy
        # per-row loop at scale, measured on P=1000/G=18533).
        return cdist(pred, real, metric="cityblock")
    raise ValueError(f"unsupported distance metric {metric!r}; use l1, l2, or cosine")


def correct_excluded_gene(
    dist: np.ndarray,
    pred: np.ndarray,
    real: np.ndarray,
    metric: str,
    row_idx: int,
    col: int,
    *,
    real_norm_squares: np.ndarray | None = None,
    pred_norm_squares: np.ndarray | None = None,
    sim: np.ndarray | None = None,
) -> None:
    """In-place: rewrite ``dist[row_idx, :]`` to the distance with gene ``col`` removed.

    ``dist`` is a full ``pairwise_full(pred, real, metric)`` matrix; this corrects the
    single ``pred`` row ``row_idx`` so it equals the distance computed with column
    ``col`` dropped from both operands. Exact for l1/l2 (subtract the gene's
    contribution); cosine recomputes the row's similarity from the reduced dot/norms.

    Cosine fast path: pass the loop-invariants ``real_norm_squares``
    (``sum(real**2, axis=1)``), ``pred_norm_squares`` (``sum(pred**2, axis=1)``), and
    ``sim`` (``pred @ real.T``) so the per-row work is O(n_real) rather than
    O(n_real·G). When omitted they are recomputed internally, so the function stays
    self-contained for standalone/test use.
    """
    a = pred[row_idx, col]
    b = real[:, col]
    if metric in ("l1", "manhattan", "cityblock"):
        # clip tiny negatives from summation roundoff (mirrors the l2 sqrt(max(.,0)))
        dist[row_idx] = np.maximum(dist[row_idx] - np.abs(b - a), 0.0)
    elif metric in ("l2", "euclidean"):
        d2 = dist[row_idx] ** 2 - (b - a) ** 2
        dist[row_idx] = np.sqrt(np.maximum(d2, 0.0))
    elif metric == "cosine":
        # recompute the row's similarity from dot/norms with gene `col` dropped
        dot_full = sim[row_idx] if sim is not None else pred[row_idx] @ real.T
        dot = dot_full - a * b
        pnsq = (
            pred_norm_squares[row_idx]
            if pred_norm_squares is not None
            else pred[row_idx] @ pred[row_idx]
        )
        npd = np.sqrt(max(pnsq - a * a, 0.0))
        if real_norm_squares is None:
            real_norm_squares = np.einsum("jg,jg->j", real, real)
        nre = np.sqrt(np.maximum(real_norm_squares - b * b, 0.0))
        denom = npd * nre
        sim_row = np.zeros_like(denom)
        nz = denom != 0
        sim_row[nz] = dot[nz] / denom[nz]  # zero-norm operand -> sim 0 -> distance 1
        dist[row_idx] = np.clip(1.0 - sim_row, 0.0, 2.0)
    else:
        raise ValueError(f"unsupported distance metric {metric!r}; use l1, l2, or cosine")
