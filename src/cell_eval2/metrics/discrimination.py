"""Perturbation discrimination in the run's resolved expression comparator.

For v2 counts/counts runs, PDS ranks effects in ``bulk_lognorm`` space::

    log1p(bulk_target_sum * group_gene_count_sum / group_total_count)

If either input is already log-normalized, and for every v1 run, it falls back to the
legacy comparator ``mean_c log1p(target_sum * C_cg / L_c)``. The dispatcher resolves that
choice once per run and supplies the corresponding precomputed pseudobulks here.
"""

from __future__ import annotations

from typing import Literal

import anndata as ad
import numpy as np

from ..distances import (
    correct_excluded_gene,
    cosine_distance_from_parts,
    pairwise_full,
    panel_reduced,
    resolve_exclusion_columns,
    resolve_panel_columns,
)
from ..prep import delta, pseudobulk


def discrimination_score(
    pred: ad.AnnData | None = None,
    real: ad.AnnData | None = None,
    *,
    pred_bulk: tuple[np.ndarray, np.ndarray] | None = None,
    real_bulk: tuple[np.ndarray, np.ndarray] | None = None,
    pert_col: str = "target",
    control: str = "non-targeting",
    distance: Literal["l1", "l2", "cosine"] = "l1",
    rank_denominator: Literal["n", "n-1"] = "n",
    tie_policy: Literal["midrank", "position"] = "midrank",
    exclude_target_gene: bool = True,
    exclusion_scope: Literal["row", "panel"] = "panel",
    control_source: Literal["pred", "real"] = "pred",
    genes: np.ndarray | None = None,
    target_gene_map: dict[str, str] | None = None,
    embed_key: str | None = None,
) -> dict[str, float]:
    """Perturbation Discrimination Score (PDS), per perturbation.

    A rank-based retrieval score: for each non-control perturbation, rank how
    close the predicted signed delta (effect) is to the matching real delta
    versus all other real deltas. Score = ``1 - rank / D``; D = number of
    non-control perturbations ``n`` (``rank_denominator="n"``) or ``n-1``.

    Ties (``tie_policy``): when real deltas are equidistant from the query, every
    member of the tied block takes the AVERAGE of the positions that block spans
    (``"midrank"``, the v2 default) -- the standard tie convention. Rows with no
    exact tie are unaffected: a block of size 1 has mid-rank == position, so this
    is bit-identical to the legacy rule wherever distances are distinct.

    ``"position"`` is the legacy rule (argsort position), retained for upstream
    cell_eval v1 parity. ⚠️ It is NOT a neutral tie-break. ``np.argsort`` returns
    the identity permutation for an all-equal row, and ``prep.pseudobulk`` returns
    SORTED labels, so a tied row resolves to the target's ALPHABETICAL index:
    ``PDS = 1 - index/D``, i.e. 1.0 for an early-alphabet target and 0.0 for a
    late one, rather than the 0.5 a no-information prediction earns. That is
    reachable in practice -- under ``cosine`` a zero-norm predicted effect (a
    submission pasting the reference control cells for a target) ties the entire
    row at distance 1.0 -- and it is gameable, since the ordering is public.
    Issue #282; ``"midrank"`` scores every such target exactly 0.5.

    A FULLY constant prediction self-corrects under either policy (the ranks are a
    bijection onto ``{0..n-1}``, so the panel mean is 0.5); a PARTIALLY constant one
    does not, which is what made the legacy rule a live scoring defect rather than a
    curiosity.

    Hybrid input: pass raw AnnData (`pred`, `real`) or precomputed pseudobulk
    tuples (`pred_bulk`, `real_bulk`) from `prep.pseudobulk`. Expression space
    only in this unit; a non-None `embed_key` (obsm-space PDS) is not yet
    supported. When precomputed bulk is passed and `exclude_target_gene=True`,
    `genes` (the var index) is required.

    ``exclusion_scope`` -- WHICH target genes ``exclude_target_gene=True`` removes, and from
    where (issue #343). Inert when exclusion is off.

    ``"panel"`` (the v2 default) removes EVERY panel target gene from the ranked feature
    space, once, before any distance is computed, so all n^2 cells of the distance matrix
    are computed on one fixed set of genes and the n distances ranked within a row are
    directly comparable. The columns are resolved from the REFERENCE's perturbations, so a
    prediction arriving as a shard is scored in the same space as the whole.

    ``"row"`` is the legacy rule, retained for upstream cell_eval v1 parity: for prediction
    row ``i`` only ``target(p_i)`` is dropped -- and it is dropped from that row's comparison
    against EVERY reference perturbation, so in cell ``(i, j)`` reference perturbation ``j``'s
    own knockdown is still there. ⚠️ That asymmetry is scoreable, and not marginally. A
    submission that spikes the panel's OTHER target genes is anti-correlated with every
    off-diagonal competitor while its own diagonal, where its gene IS dropped, sits at cosine
    0; the self-match is nearest by construction, on no information beyond the target list
    every participant is given. MEASURED on the three official val contexts at 0.14.0:
    ``pds_cosine`` 0.7982 / 0.7570 / 0.7614 against baselines 0.5304 / 0.5284 / 0.5102, i.e.
    +0.57 / +0.49 / +0.51 of member score for a content-free submission. Under ``"panel"``
    those arms measure exactly 0.5000. The amplitude of the spike is irrelevant -- cosine is
    scale-invariant -- so it costs the attacker nothing on the expression-error members.

    ``"row"`` also hands out a smaller standing gift, to every submission rather than to an
    attacker: only the DIAGONAL's reference-side norm loses its knockdown, so the self-distance
    is computed against a systematically deflated denominator. On a submission carrying the
    panel-average response that is worth ~+0.017 raw ``pds_cosine``.

    `target_gene_map`: explicit `{perturbation: gene}` override for target-gene
    exclusion, the SAME `EvalConfig.target_gene_map` the DE metrics resolve through.
    Required for guide-level panels, whose labels are construct IDs (`ADNP-1`) rather
    than gene symbols; without it those labels match no gene and `exclude_target_gene`
    now raises rather than silently scoring with nothing excluded (issue #248).

    `control_source`: "pred" subtracts the predicted control from predicted
    means; "real" subtracts the real control. The real side always uses the
    real control. The control perturbation is excluded from the result.
    """
    if embed_key is not None:
        raise NotImplementedError(
            "embed_key (obsm-space discrimination) is not implemented in this unit; "
            "expression-space PDS only (embed_key=None)"
        )
    if rank_denominator not in ("n", "n-1"):
        raise ValueError(
            f"rank_denominator must be 'n' or 'n-1', got {rank_denominator!r}"
        )
    if tie_policy not in ("midrank", "position"):
        raise ValueError(
            f"tie_policy must be 'midrank' or 'position', got {tie_policy!r}"
        )
    if exclusion_scope not in ("row", "panel"):
        raise ValueError(
            f"exclusion_scope must be 'row' or 'panel', got {exclusion_scope!r}"
        )
    if control_source not in ("pred", "real"):
        raise ValueError(
            f"control_source must be 'pred' or 'real', got {control_source!r}"
        )

    if (pred_bulk is None) != (real_bulk is None):
        raise ValueError(
            "provide both pred_bulk and real_bulk, or neither (a single precomputed "
            "bulk would be silently ignored and recomputed from AnnData)"
        )
    if pred_bulk is None or real_bulk is None:
        if pred is None or real is None:
            raise ValueError("provide pred+real AnnData or pred_bulk+real_bulk tuples")
        pred_bulk = pseudobulk(pred, pert_col)
        real_bulk = pseudobulk(real, pert_col)
        if genes is None:
            genes = np.asarray(real.var.index.values, dtype=str)

    if exclude_target_gene and genes is None:
        raise ValueError(
            "genes (the var index) is required when exclude_target_gene=True and "
            "precomputed bulk is passed"
        )

    real_perts = np.asarray(real_bulk[0]).astype(str)
    pred_perts = np.asarray(pred_bulk[0]).astype(str)
    real_means = np.asarray(real_bulk[1], dtype=np.float64)
    pred_means = np.asarray(pred_bulk[1], dtype=np.float64)

    if real_means.shape[0] != real_perts.shape[0] or pred_means.shape[0] != pred_perts.shape[0]:
        raise ValueError(
            "bulk perts/means row mismatch: "
            f"real ({real_perts.shape[0]} perts vs {real_means.shape[0]} rows), "
            f"pred ({pred_perts.shape[0]} perts vs {pred_means.shape[0]} rows)"
        )
    if real_means.shape[1] != pred_means.shape[1]:
        raise ValueError(
            f"pred/real feature dimension mismatch: pred {pred_means.shape[1]} != "
            f"real {real_means.shape[1]}"
        )

    # Real-side effects (always against the real control).
    perts, real_eff = delta(real_means, real_perts, control)

    # Predicted-side effects: control_source selects which control is subtracted.
    if control_source == "pred":
        pred_keys, pred_eff = delta(pred_means, pred_perts, control)
    else:  # "real": measure predicted effect against the real control
        ctrl_hits = np.flatnonzero(real_perts == control)
        if ctrl_hits.size == 0:
            raise ValueError(f"control {control!r} not found in real perturbations")
        real_ctrl = real_means[ctrl_hits[0]]
        mask = pred_perts != control
        pred_keys, pred_eff = pred_perts[mask], pred_means[mask] - real_ctrl

    # Generalize: pred keys must be a SUBSET of real keys (equality is the whole-prediction
    # case). Each pred effect is ranked against ALL real effects; the denominator is the
    # full real perturbation count. When pred_keys == perts this is byte-identical to the
    # prior behavior (match_col == i for every i).
    real_keys = np.asarray(perts).astype(str)
    pred_keys = np.asarray(pred_keys).astype(str)
    real_pos = {p: j for j, p in enumerate(real_keys)}
    missing = [p for p in pred_keys if p not in real_pos]
    if missing:
        raise ValueError(
            "pred non-control perturbations must be a subset of real; "
            f"labels missing from real: {missing[:10]}"
        )

    if genes is not None:
        genes = np.asarray(genes).astype(str)
        if exclude_target_gene and genes.shape[0] != real_eff.shape[1]:
            raise ValueError(
                f"genes length ({genes.shape[0]}) does not match the pseudobulk feature "
                f"dimension ({real_eff.shape[1]}); cannot exclude target genes"
            )

    n = real_keys.size                      # denominator uses the FULL real count
    D = n if rank_denominator == "n" else n - 1

    # Full-matrix path: compute the [n_pred, n_real] distance once (l2/cosine collapse
    # to a BLAS matmul) and correct each row for its single excluded target gene,
    # instead of boolean-indexing a fresh ~[n_perts, n_genes-1] copy per perturbation.
    # Exact parity with the prior per-pert loop. `exclude_target_gene` drops one column
    # per perturbation by matching the pert label to a gene position, which requires the
    # gene panel (`genes`) to have unique labels (true for a var index); duplicates are
    # rejected up front rather than silently corrected for only one occurrence.
    out: dict[str, float] = {}
    do_exclude = exclude_target_gene and genes is not None
    # ---- exclusion_scope="panel" (issue #343) --------------------------------------
    # Drop EVERY panel target gene from both operands once, before any distance is
    # computed, and then rank in that one fixed feature space with no per-row correction
    # at all. The columns come from `real_keys`, the reference's whole panel, so a shard
    # of the prediction is scored in the same space as the whole (see
    # `resolve_panel_columns`).
    #
    # Why this and not the per-row rule below: `correct_excluded_gene` rewrites the WHOLE
    # row `full[i, :]` dropping only `target(p_i)`, so in cell (i, j) the reference
    # perturbation j's OWN knockdown -- the largest and most predictable coordinate of
    # `real_eff[j]` -- stays fully visible. A submission that spikes the panel's other
    # targets is then anti-correlated with every off-diagonal competitor while its own
    # diagonal, where its gene IS dropped, sits at cosine 0. The self-match wins by
    # construction, on no biology beyond the target list every participant is given.
    # MEASURED on the three official val contexts at v0.14.0: pds_cosine 0.7982 / 0.7570 /
    # 0.7614 against baselines 0.5304 / 0.5284 / 0.5102, i.e. +0.57 / +0.49 / +0.51 of
    # member score for a submission carrying no information at all. Under this scope the
    # same arms measure 0.5000 on all three -- exactly the control-paste floor -- while a
    # perfect submission still measures 1.0000 and a partial one moves <= 0.01 of member
    # score. The same rule also removes a smaller standing gift: under the per-row rule the
    # DIAGONAL's real-side norm alone loses its knockdown, so every submission with a
    # non-zero delta was scored against a systematically deflated self-distance.
    if do_exclude and exclusion_scope == "panel":
        panel_cols = resolve_panel_columns(real_keys, genes,
                                           target_gene_map=target_gene_map)
        pred_eff = panel_reduced(pred_eff, panel_cols)
        real_eff = panel_reduced(real_eff, panel_cols)
        full = pairwise_full(pred_eff, real_eff, distance)   # [n_pred, n_real]
        for i, p in enumerate(pred_keys):
            match_col = real_pos[str(p)]
            rank = _match_rank(full[i], match_col, tie_policy)
            out[str(p)] = 1.0 if D <= 0 else 1.0 - rank / D
        return out
    # ---- exclusion_scope="row": the legacy per-row rule, v1 parity only --------------
    if do_exclude and np.unique(genes).size != genes.shape[0]:
        raise ValueError(
            "duplicate gene names in `genes` are not supported when "
            "exclude_target_gene=True: the full-matrix path needs a unique "
            "gene->column mapping (deduplicate the var index, e.g. "
            "AnnData.var_names_make_unique())"
        )
    # Which column is each pert's OWN gene. Resolved through the shared helper (and NOT
    # by a local `{gene: col}` lookup of the raw label) so the construct-ID case resolves
    # via target_gene_map and a zero-resolve run raises instead of silently excluding
    # nothing -- and so the GPU kernel cannot drift from this one. Issue #248.
    # gate_labels=real_keys: the raise is a statement about the PANEL, and pred_keys may be
    # a shard (scale.py restricts the pred bulks; partition_inmem.py passes one piece) while
    # the real bulks always arrive whole. Gating on pred_keys made the failure depend on
    # partition composition.
    exclusion_cols = (
        resolve_exclusion_columns(pred_keys, genes, target_gene_map=target_gene_map,
                                  gate_labels=real_keys)
        if do_exclude else {}
    )
    # The cosine drop-gene correction reuses the real/pred row-norms^2 and the
    # pred.real^T dot matrix, all loop-invariant. For cosine + exclusion, compute them
    # once and build the base distance from the same parts (cosine_distance_from_parts)
    # so the O(n_perts^2 * n_genes) matmul is done ONCE, not again inside pairwise_full.
    real_norm_squares = pred_norm_squares = sim = None
    if distance == "cosine" and do_exclude:
        real_norm_squares = np.einsum("jg,jg->j", real_eff, real_eff)
        pred_norm_squares = np.einsum("ig,ig->i", pred_eff, pred_eff)
        sim = pred_eff @ real_eff.T
        full = cosine_distance_from_parts(sim, pred_norm_squares, real_norm_squares)
    else:
        full = pairwise_full(pred_eff, real_eff, distance)  # [n_pred, n_real]
    for i, p in enumerate(pred_keys):
        col = exclusion_cols.get(i)
        if col is not None:  # this pert's target gene is in the panel
            correct_excluded_gene(
                full, pred_eff, real_eff, distance, i, col,
                real_norm_squares=real_norm_squares,
                pred_norm_squares=pred_norm_squares, sim=sim,
            )
        match_col = real_pos[str(p)]        # this pred pert's own effect among the real effects
        rank = _match_rank(full[i], match_col, tie_policy)
        out[str(p)] = 1.0 if D <= 0 else 1.0 - rank / D
    return out


def _match_rank(row: np.ndarray, match_col: int, tie_policy: str) -> float:
    """0-based rank of ``row[match_col]`` within ``row``, under ``tie_policy``.

    ``"midrank"``: ``n_less + (n_equal - 1) / 2`` -- the average of the positions the
    tied block spans. Equals the argsort position exactly when the match is untied
    (``n_equal == 1``), so every existing number over distinct distances is preserved
    bit-for-bit. It is also O(n) rather than the O(n log n) sort it replaces.

    ``"position"``: the legacy ``argsort`` position (see the caller's docstring).

    NaN ordering matches ``np.sort``'s (NaN last) under both policies. For a finite
    match distance that falls out for free -- ``NaN < x`` and ``NaN == x`` are both
    False, so NaN competitors count as neither closer nor tied, i.e. as farther. A NaN
    match distance is handled explicitly: it sits inside the trailing NaN block.
    """
    if tie_policy == "position":
        return float(np.flatnonzero(np.argsort(row) == match_col)[0])
    match = row[match_col]
    if np.isnan(match):
        n_less = int(np.count_nonzero(~np.isnan(row)))
        n_equal = int(row.size - n_less)
    else:
        n_less = int(np.count_nonzero(row < match))
        n_equal = int(np.count_nonzero(row == match))
    return n_less + (n_equal - 1) / 2.0
