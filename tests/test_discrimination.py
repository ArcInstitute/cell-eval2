import numpy as np
import pytest

from cell_eval2.metrics import discrimination_score

# Bulk tuples are (perts, means) exactly as prep.pseudobulk returns them.
# Perturbations are sorted; the control row is "ctrl".


def _bulk(perts, rows):
    return np.array(perts), np.array(rows, dtype=np.float64)


def test_rank_denominator_n_vs_n_minus_1():
    # pred effect for A is closer to real B than real A -> rank 1 for A; B,C rank 0.
    real = _bulk(["A", "B", "C", "ctrl"],
                 [[4, 0], [0, 4], [-4, -4], [0, 0]])
    pred = _bulk(["A", "B", "C", "ctrl"],
                 [[1, 3], [0, 4], [-4, -4], [0, 0]])
    # A/B/C are not gene names -> nothing to exclude. Stated EXPLICITLY rather than left
    # to the default: since #248 a zero-resolve run with exclude_target_gene=True raises,
    # because "asked to exclude, excluded nothing" is the silent-wrong-number this test
    # would otherwise be quietly relying on. This test is about the denominator.
    genes = np.array(["g0", "g1"])
    out_n = discrimination_score(pred_bulk=pred, real_bulk=real, control="ctrl",
                                 distance="l1", rank_denominator="n",
                                 control_source="pred", genes=genes,
                                 exclude_target_gene=False)
    out_nm1 = discrimination_score(pred_bulk=pred, real_bulk=real, control="ctrl",
                                   distance="l1", rank_denominator="n-1",
                                   control_source="pred", genes=genes,
                                   exclude_target_gene=False)
    assert out_n == pytest.approx({"A": 2 / 3, "B": 1.0, "C": 1.0})
    assert out_nm1 == pytest.approx({"A": 0.5, "B": 1.0, "C": 1.0})


def test_control_source_real_vs_pred():
    # Predicted ABSOLUTE means are perfect, but the predicted control is wrong.
    # control_source="real" measures the pred effect against the (correct) real
    # control -> perfect; "pred" uses the bad predicted control -> A degrades.
    real = _bulk(["A", "B", "ctrl"], [[10, 0], [0, 10], [0, 0]])
    pred = _bulk(["A", "B", "ctrl"], [[10, 0], [0, 10], [6, -6]])  # bad pred ctrl
    out_real = discrimination_score(pred_bulk=pred, real_bulk=real, control="ctrl",
                                    distance="l1", rank_denominator="n",
                                    control_source="real", exclude_target_gene=False)
    out_pred = discrimination_score(pred_bulk=pred, real_bulk=real, control="ctrl",
                                    distance="l1", rank_denominator="n",
                                    control_source="pred", exclude_target_gene=False)
    assert out_real == pytest.approx({"A": 1.0, "B": 1.0})
    assert out_pred == pytest.approx({"A": 0.5, "B": 1.0})


def test_exclude_target_gene_drops_named_column():
    # genes are named "A","B"; perturbations "A","B" share those names.
    # Excluding the target gene flips A's nearest neighbour (0.75 -> 0.5).
    #
    # ⚠️ `keep["A"]` was 1.0 before issue #282 and that value ENCODED THE DEFECT: pred A's
    # effect [2,2] is exactly equidistant from real A's [1,9] and real B's [9,1] (L1 = 8
    # to both), so A is a genuine tie that the prediction cannot resolve at all -- and the
    # legacy argsort rule handed it rank 0, a perfect score, purely because "A" sorts
    # first. Under mid-rank the tied pair shares rank 0.5, giving 1 - 0.5/2 = 0.75 with
    # this test's D = n = 2. (0.75 is the no-information point for a 2-target panel under
    # the "n" denominator, whose range is [0.5, 1.0].) `drop` is untied and unchanged.
    real = _bulk(["A", "B", "ctrl"], [[1, 9], [9, 1], [0, 0]])
    pred = _bulk(["A", "B", "ctrl"], [[2, 2], [9, 1], [0, 0]])
    genes = np.array(["A", "B"])
    keep = discrimination_score(pred_bulk=pred, real_bulk=real, control="ctrl",
                                distance="l1", rank_denominator="n",
                                control_source="pred", exclude_target_gene=False,
                                genes=genes)
    # `exclusion_scope="row"`: this test IS the per-row rule's definition -- one NAMED column
    # dropped, not the panel. Under the v2 default ("panel", #343) both of this 2-gene
    # panel's genes are targets, so the ranked space would be empty and `panel_reduced`
    # raises. The panel rule's own behaviour is pinned in test_pds_panel_exclusion_343.py.
    drop = discrimination_score(pred_bulk=pred, real_bulk=real, control="ctrl",
                                distance="l1", rank_denominator="n",
                                control_source="pred", exclude_target_gene=True,
                                exclusion_scope="row", genes=genes)
    assert keep == pytest.approx({"A": 0.75, "B": 1.0})
    assert drop == pytest.approx({"A": 0.5, "B": 1.0})
    # and the legacy rule still reproduces the pre-#282 number, so v1 parity is pinned
    legacy = discrimination_score(pred_bulk=pred, real_bulk=real, control="ctrl",
                                  distance="l1", rank_denominator="n",
                                  control_source="pred", exclude_target_gene=False,
                                  genes=genes, tie_policy="position")
    assert legacy == pytest.approx({"A": 1.0, "B": 1.0})


def test_cosine_zero_norm_pred_effect_is_finite():
    # A's predicted effect is the zero vector -> cosine distance 1.0 to every real
    # effect; must not raise or produce NaN.
    #
    # ⚠️ This assertion alone is NOT sufficient and was not (issue #282): the legacy
    # argsort rule scored A by its ALPHABETICAL position, which is finite and non-NaN,
    # so this test passed on the defect for its whole life. The value is pinned in
    # test_all_tied_row_scores_the_no_information_point below. Keep both.
    real = _bulk(["A", "B", "ctrl"], [[3, 1], [1, 3], [0, 0]])
    pred = _bulk(["A", "B", "ctrl"], [[5, 5], [1, 3], [5, 5]])  # A pred effect = [0,0]
    out = discrimination_score(pred_bulk=pred, real_bulk=real, control="ctrl",
                               distance="cosine", rank_denominator="n",
                               control_source="pred", exclude_target_gene=False)
    assert set(out) == {"A", "B"}
    assert all(np.isfinite(v) for v in out.values())


# --------------------------------------------------------------------------------------
# issue #282: tie policy
# --------------------------------------------------------------------------------------

def _tied_panel(n):
    """n targets named aa, bb, cc, ... plus a control; every predicted effect is EXACTLY
    the zero vector, so under cosine every distance is 1.0 and the whole matrix ties."""
    names = [chr(ord("a") + i) * 2 for i in range(n)]
    perts = names + ["ctrl"]
    real = _bulk(perts, [[i + 1, n - i] for i in range(n)] + [[0, 0]])
    pred = _bulk(perts, [[7, 7]] * (n + 1))          # pred effect == 0 for every target
    return names, real, pred


@pytest.mark.parametrize("n", [2, 3, 8, 26])
def test_all_tied_row_scores_the_no_information_point(n):
    """Every target of an all-tied panel scores EXACTLY 0.5 -- not its alphabetical index.

    The defect this pins (#282): a zero-norm predicted effect ties the entire cosine row,
    the legacy argsort rule resolved that to the target's position in the SORTED label
    array, and the result was 1.0 for the first target and 0.0 for the last.
    """
    names, real, pred = _tied_panel(n)
    out = discrimination_score(pred_bulk=pred, real_bulk=real, control="ctrl",
                               distance="cosine", rank_denominator="n-1",
                               control_source="pred", exclude_target_gene=False)
    assert out == pytest.approx({name: 0.5 for name in names})


def test_legacy_policy_still_reproduces_the_alphabetical_ranking():
    """v1 parity: tie_policy="position" must keep the exact legacy behaviour, defect and
    all. If this ever changes, upstream cell-eval parity has silently moved."""
    names, real, pred = _tied_panel(6)
    out = discrimination_score(pred_bulk=pred, real_bulk=real, control="ctrl",
                               distance="cosine", rank_denominator="n-1",
                               control_source="pred", exclude_target_gene=False,
                               tie_policy="position")
    assert out == pytest.approx(dict(zip(names, [1.0, 0.8, 0.6, 0.4, 0.2, 0.0])))


def test_partial_ties_do_not_move_with_target_names():
    """The regression that makes #282 a scoring defect rather than a curiosity.

    A FULLY tied panel self-corrects under either policy -- the ranks are a bijection, so
    the mean is 0.5 regardless. Zeroing only a SUBSET breaks that cancellation, and under
    the legacy rule the subset keeps whichever half of the range its names land in. The
    panel mean must not depend on WHICH targets were zeroed.

    The comparison is only sound because the NON-zeroed targets are predicted exactly, so
    they score 1.0 whichever ones they are; otherwise the two panels would differ merely by
    holding different targets and the assertion would be measuring the wrong thing.
    Mutation-checked -- under tie_policy="position" this reads 0.9464 vs 0.6786 and the
    assertion fails, so the test discriminates rather than passing vacuously.
    """
    n = 8
    names = [chr(ord("a") + i) * 2 for i in range(n)]
    perts = names + ["ctrl"]
    real = _bulk(perts, [[i + 1, n - i] for i in range(n)] + [[0, 0]])

    def panel(zeroed_idx):
        rows = []
        for i in range(n):
            rows.append([7, 7] if i in zeroed_idx else [7 + (i + 1), 7 + (n - i)])
        pred = _bulk(perts, rows + [[7, 7]])
        out = discrimination_score(pred_bulk=pred, real_bulk=real, control="ctrl",
                                   distance="cosine", rank_denominator="n-1",
                                   control_source="pred", exclude_target_gene=False)
        return float(np.mean([out[k] for k in names]))

    early, late = panel({0, 1, 2}), panel({n - 3, n - 2, n - 1})
    assert early == pytest.approx(late), (
        f"zeroing early- vs late-alphabet targets moved the panel: {early} vs {late}"
    )


@pytest.mark.parametrize("distance", ["l1", "l2", "cosine"])
def test_midrank_is_bit_identical_when_no_distance_is_tied(distance):
    """The correction must not move a single number on ordinary input. A tied block of
    size 1 has mid-rank == argsort position, so the two policies agree exactly."""
    rng = np.random.default_rng(0)
    perts = [f"p{i}" for i in range(12)] + ["ctrl"]
    real = _bulk(perts, rng.random((13, 20)) + 1.0)
    pred = _bulk(perts, rng.random((13, 20)) + 1.0)
    kw = dict(pred_bulk=pred, real_bulk=real, control="ctrl", distance=distance,
              rank_denominator="n-1", control_source="pred", exclude_target_gene=False)
    assert (discrimination_score(**kw, tie_policy="midrank")
            == discrimination_score(**kw, tie_policy="position"))


@pytest.mark.parametrize("policy", ["midrank", "position"])
def test_match_rank_orders_NaN_last_on_both_paths(policy):
    """The NaN branch, which nothing covered directly (Codex, #282 review).

    Two claims are under test. (a) For a FINITE match, NaN competitors need no special
    handling: `NaN < x` and `NaN == x` are both False, so a NaN counts as neither closer
    nor tied -- i.e. implicitly farther, which is exactly `np.sort`'s NaN-last ordering.
    (b) A NaN MATCH is handled explicitly and lands inside the trailing NaN block. Both
    must hold identically on the CPU helper and its xp twin.
    """
    from cell_eval2.gpu.distances import _match_ranks_xp
    from cell_eval2.metrics.discrimination import _match_rank

    row = np.array([0.5, np.nan, 0.2, np.nan, 0.9])
    #                 ^finite     ^smallest        ^largest finite
    expected = {
        # finite match: 1 value below (0.2), no ties -> rank 1. NaNs count as farther.
        "midrank": {0: 1.0, 2: 0.0, 4: 2.0,
                    # NaN match: 3 finite values sort first, then the 2-NaN block -> 3.5
                    1: 3.5, 3: 3.5},
        # legacy argsort also puts NaN last, but splits the block by position
        "position": {0: 1.0, 2: 0.0, 4: 2.0, 1: 3.0, 3: 4.0},
    }[policy]
    for col, want in expected.items():
        got = _match_rank(row, col, policy)
        assert got == pytest.approx(want), f"col {col}: {got} != {want}"
        xp_got = _match_ranks_xp(np, row[None, :], np.array([col]), policy)
        assert float(np.asarray(xp_got)[0]) == pytest.approx(want), "xp twin disagrees"


def test_an_all_NaN_row_does_not_produce_a_negative_rank():
    """The degenerate corner of the NaN branch: with no finite competitor, `n_less` is 0
    and the whole row is one tied NaN block. A naive implementation reads rank -0.5 here
    (n_equal == 0 because NaN != NaN), which would score ABOVE perfect."""
    from cell_eval2.metrics.discrimination import _match_rank

    row = np.full(4, np.nan)
    assert _match_rank(row, 0, "midrank") == pytest.approx(1.5)   # (4 - 1) / 2
    assert _match_rank(row, 3, "midrank") == pytest.approx(1.5)


def test_tie_policy_is_validated():
    names, real, pred = _tied_panel(3)
    with pytest.raises(ValueError, match="tie_policy"):
        discrimination_score(pred_bulk=pred, real_bulk=real, control="ctrl",
                             distance="cosine", exclude_target_gene=False,
                             tie_policy="alphabetical")


def test_single_perturbation_corrected_denominator_zero():
    # n == 1 -> corrected D = n-1 = 0; must not divide by zero. Single pert -> 1.0.
    real = _bulk(["A", "ctrl"], [[2, 0], [1, 1]])
    pred = _bulk(["A", "ctrl"], [[1.8, 0.1], [0.9, 1.1]])
    out = discrimination_score(pred_bulk=pred, real_bulk=real, control="ctrl",
                               distance="cosine", rank_denominator="n-1",
                               control_source="real", exclude_target_gene=False)
    assert out == {"A": 1.0}


def test_bulk_path_requires_genes_when_excluding():
    real = _bulk(["A", "B", "ctrl"], [[1, 9], [9, 1], [0, 0]])
    pred = _bulk(["A", "B", "ctrl"], [[2, 2], [9, 1], [0, 0]])
    with pytest.raises(ValueError, match="genes"):
        discrimination_score(pred_bulk=pred, real_bulk=real, control="ctrl",
                             distance="l1", exclude_target_gene=True)  # genes=None


def test_duplicate_gene_names_raise_when_excluding():
    # The full-matrix path maps each pert to a single gene column; duplicate gene
    # names would silently correct only one occurrence (the old loop dropped all).
    # Raise a clear error instead of producing a wrong result.
    real = _bulk(["A", "B", "ctrl"], [[1, 9, 0], [9, 1, 0], [0, 0, 0]])
    pred = _bulk(["A", "B", "ctrl"], [[2, 2, 0], [9, 1, 0], [0, 0, 0]])
    dup_genes = np.array(["A", "A", "B"])  # duplicate "A"; len matches feature dim 3
    with pytest.raises(ValueError, match="duplicate"):
        discrimination_score(pred_bulk=pred, real_bulk=real, control="ctrl",
                             distance="l1", exclude_target_gene=True, genes=dup_genes)


def test_duplicate_gene_names_ok_when_not_excluding():
    # Without exclusion the gene panel is never indexed, so duplicates are harmless.
    real = _bulk(["A", "B", "ctrl"], [[1, 9, 0], [9, 1, 0], [0, 0, 0]])
    pred = _bulk(["A", "B", "ctrl"], [[2, 2, 0], [9, 1, 0], [0, 0, 0]])
    dup_genes = np.array(["A", "A", "B"])
    out = discrimination_score(pred_bulk=pred, real_bulk=real, control="ctrl",
                               distance="l1", exclude_target_gene=False,
                               genes=dup_genes)
    assert set(out) == {"A", "B"}


def test_genes_length_mismatch_raises_clear_error():
    # 2-feature bulk but a 3-element genes array -> clear ValueError (not a cryptic
    # numpy IndexError from boolean indexing). Only reachable via precomputed bulk.
    real = _bulk(["A", "B", "ctrl"], [[1, 9], [9, 1], [0, 0]])
    pred = _bulk(["A", "B", "ctrl"], [[2, 2], [9, 1], [0, 0]])
    bad_genes = np.array(["A", "B", "C"])  # length 3 != 2 features
    with pytest.raises(ValueError, match="genes length"):
        discrimination_score(pred_bulk=pred, real_bulk=real, control="ctrl",
                             distance="l1", exclude_target_gene=True, genes=bad_genes)


def test_partial_bulk_input_raises():
    # Providing only one of pred_bulk/real_bulk must error rather than silently
    # ignoring it and recomputing from AnnData.
    real = _bulk(["A", "B", "ctrl"], [[1, 9], [9, 1], [0, 0]])
    with pytest.raises(ValueError, match="both pred_bulk and real_bulk"):
        discrimination_score(real_bulk=real, control="ctrl", distance="l1",
                             exclude_target_gene=False)


def test_bulk_feature_dim_mismatch_raises():
    real = _bulk(["A", "B", "ctrl"], [[1, 9], [9, 1], [0, 0]])       # 2 features
    pred = _bulk(["A", "B", "ctrl"], [[2, 2, 0], [9, 1, 0], [0, 0, 0]])  # 3 features
    with pytest.raises(ValueError, match="feature dimension mismatch"):
        discrimination_score(pred_bulk=pred, real_bulk=real, control="ctrl",
                             distance="l1", exclude_target_gene=False)


def test_embed_key_not_implemented():
    real = _bulk(["A", "ctrl"], [[2, 0], [1, 1]])
    pred = _bulk(["A", "ctrl"], [[1.8, 0.1], [0.9, 1.1]])
    with pytest.raises(NotImplementedError, match="embed_key"):
        discrimination_score(pred_bulk=pred, real_bulk=real, control="ctrl",
                             embed_key="X_pca", exclude_target_gene=False)


def _slow_discrimination(pred_bulk, real_bulk, *, control, distance, rank_denominator,
                         control_source, exclude_target_gene, genes):
    # reference: the pre-restructure per-pert loop (vendored here to pin exact parity)
    from cell_eval2.distances import pairwise_to_vector
    from cell_eval2.prep import delta
    rp = np.asarray(real_bulk[0]).astype(str)
    pp = np.asarray(pred_bulk[0]).astype(str)
    rm = np.asarray(real_bulk[1], float)
    pm = np.asarray(pred_bulk[1], float)
    perts, real_eff = delta(rm, rp, control)
    if control_source == "pred":
        _, pred_eff = delta(pm, pp, control)
    else:
        rc = rm[np.flatnonzero(rp == control)[0]]
        m = pp != control
        pred_eff = pm[m] - rc
    g = np.asarray(genes).astype(str)
    n = perts.size
    D = n if rank_denominator == "n" else n - 1
    out = {}
    for i, p in enumerate(perts):
        if exclude_target_gene and bool((g == p).any()):
            inc = g != p
            ref = real_eff[:, inc]
            q = pred_eff[i, inc]
        else:
            ref = real_eff
            q = pred_eff[i]
        order = np.argsort(pairwise_to_vector(ref, q, distance))
        out[str(p)] = 1.0 if D <= 0 else 1.0 - int(np.flatnonzero(order == i)[0]) / D
    return out


@pytest.mark.parametrize("distance", ["l1", "l2", "cosine"])
@pytest.mark.parametrize("excl", [True, False])
@pytest.mark.parametrize("csrc", ["pred", "real"])
def test_restructured_matches_slow_loop(distance, excl, csrc):
    rng = np.random.default_rng(7)
    P, G = 12, 8
    names = [f"G{i}" for i in range(P - 1)] + ["ctrl"]
    genes = np.array([f"G{i}" for i in range(G)])  # some perts share gene names -> exclusion fires
    real = (np.array(names), rng.normal(size=(P, G)))
    pred = (np.array(names), rng.normal(size=(P, G)))
    # `_slow_discrimination` below vendors the PRE-RESTRUCTURE per-pert loop, which is the
    # per-row rule; the scope is pinned to match it. The panel rule has no such optimization
    # to pin -- it is one plain matmul on a narrower matrix -- and its CPU/GPU agreement is
    # covered in test_pds_panel_exclusion_343.py.
    kw = dict(control="ctrl", distance=distance, rank_denominator="n",
              control_source=csrc, exclude_target_gene=excl, exclusion_scope="row",
              genes=genes)
    got = discrimination_score(pred_bulk=pred, real_bulk=real, **kw)
    exp = _slow_discrimination(pred, real, **{k: v for k, v in kw.items()
                                              if k != "exclusion_scope"})
    assert got.keys() == exp.keys()
    for k in got:
        assert got[k] == pytest.approx(exp[k], rel=1e-4, abs=1e-6)


def test_hybrid_anndata_equals_bulk(synthetic_pair):
    from cell_eval2.prep import pseudobulk
    pred, real = synthetic_pair
    genes = np.asarray(real.var.index.values, dtype=str)
    # exclude_target_gene=False explicitly: the fixture's targets (GENE1..GENE3) share no
    # label with its genes (g0..g39), so exclusion would resolve nothing and raise since
    # #248. This test is about AnnData-vs-bulk input equivalence, not exclusion.
    from_ad = discrimination_score(pred=pred, real=real, pert_col="target",
                                   control="non-targeting", distance="l1",
                                   rank_denominator="n", control_source="pred",
                                   exclude_target_gene=False)
    from_bulk = discrimination_score(pred_bulk=pseudobulk(pred, "target"),
                                     real_bulk=pseudobulk(real, "target"),
                                     control="non-targeting", distance="l1",
                                     rank_denominator="n", control_source="pred",
                                     genes=genes, exclude_target_gene=False)
    assert from_ad == from_bulk


# ---------------------------------------------------------------------------
# Issue #248: exclude_target_gene silently no-ops on guide-level (SYMBOL-N) labels.
#
# The shape that matters: a guide-level panel labels perturbations by CONSTRUCT
# ("ADNP-1"), not by gene symbol ("ADNP"). Before the fix the label missed the gene
# index, the exclusion was skipped with no signal, and every perturbation kept its own
# transcript -- a coordinate predictable from the label alone -- inside the ranked
# vector. These tests pin the three things that fix requires: the map resolves, a
# gene-level panel is untouched, and a run that resolves NOTHING raises.
# ---------------------------------------------------------------------------

def _guide_panel():
    """Guide-level panel: labels SYMBOL-N, genes bare symbols. The #248 shape.

    Constructed so the exclusion is DECISIVE for A-1: on the full 3-gene vector A-1's
    predicted effect is nearest real A-1 (rank 0), but that is carried entirely by the
    shared "own transcript goes down" coordinate at gene A. Drop gene A and the
    remaining biology (genes B, C) is nearer real B-1 -> rank 1. So the score moves iff
    the exclusion actually fires. With D = n = 2 non-control perts: 1.0 with the leak
    (rank 0), 0.5 without it (rank 1).

    By hand, l1, for pred A-1 = [-9, 4, 4]:
      full vector    -> d(real A-1 [-9,1,0]) = 7  <  d(real B-1 [0,5,5]) = 11  -> rank 0
      gene A dropped -> d(real A-1 [1,0])    = 7  >  d(real B-1 [5,5])   = 2   -> rank 1
    The entire rank-0 result is bought by the one coordinate the model gets for free.
    """
    #                      gene:   A     B     C
    real = _bulk(["A-1", "B-1", "ctrl"],
                 [[-9.0,  1.0,  0.0],     # A-1: own transcript down, weak B response
                  [ 0.0,  5.0,  5.0],     # B-1: no A knockdown, strong B/C response
                  [ 0.0,  0.0,  0.0]])
    pred = _bulk(["A-1", "B-1", "ctrl"],
                 [[-9.0,  4.0,  4.0],     # nails the free on-target coordinate; biology
                                          # actually looks like B-1's
                  [ 0.0,  5.0,  5.0],
                  [ 0.0,  0.0,  0.0]])
    genes = np.array(["A", "B", "C"])
    tgm = {"A-1": "A", "B-1": "B"}
    return real, pred, genes, tgm


def test_target_gene_map_drops_the_mapped_column_for_guide_labels():
    # THE REGRESSION TEST for #248. Same data, same exclude_target_gene=True; the only
    # difference is whether the construct->gene map is supplied.
    real, pred, genes, tgm = _guide_panel()
    kw = dict(pred_bulk=pred, real_bulk=real, control="ctrl", distance="l1",
              rank_denominator="n", control_source="pred", genes=genes,
              exclude_target_gene=True)

    mapped = discrimination_score(**kw, target_gene_map=tgm)

    # With the map the on-target column is genuinely dropped, and A-1 is no longer
    # rescued by its own transcript: 1.0 -> 0.5.
    assert mapped == pytest.approx({"A-1": 0.5, "B-1": 1.0})

    # The leak, quantified on this panel: scoring the SAME data with exclusion off
    # returns 1.0 for A-1. That 0.5 gap is the free signal #248 was handing out.
    leaked = discrimination_score(**{**kw, "exclude_target_gene": False})
    assert leaked == pytest.approx({"A-1": 1.0, "B-1": 1.0})

    # And it equals the score of the SAME panel relabelled to bare symbols, which is the
    # path that always worked -- the map buys label-format independence, nothing else.
    real_sym = (np.array(["A", "B", "ctrl"]), real[1])
    pred_sym = (np.array(["A", "B", "ctrl"]), pred[1])
    symbol_level = discrimination_score(
        pred_bulk=pred_sym, real_bulk=real_sym, control="ctrl", distance="l1",
        rank_denominator="n", control_source="pred", genes=genes,
        exclude_target_gene=True,
    )
    assert mapped["A-1"] == pytest.approx(symbol_level["A"])
    assert mapped["B-1"] == pytest.approx(symbol_level["B"])


def test_guide_labels_without_a_map_raise_instead_of_silently_scoring():
    # The actual bug: this call used to return {"A-1": 1.0, "B-1": 1.0} -- a plausible
    # number computed with nothing excluded, despite exclude_target_gene=True.
    real, pred, genes, _ = _guide_panel()
    with pytest.raises(ValueError, match="NO perturbation resolves"):
        discrimination_score(pred_bulk=pred, real_bulk=real, control="ctrl",
                             distance="l1", rank_denominator="n",
                             control_source="pred", genes=genes,
                             exclude_target_gene=True)


def test_zero_resolve_error_names_the_override_and_the_opt_out():
    # The message has to tell the caller how to proceed, both ways.
    real, pred, genes, _ = _guide_panel()
    with pytest.raises(ValueError) as exc:
        discrimination_score(pred_bulk=pred, real_bulk=real, control="ctrl",
                             distance="l1", exclude_target_gene=True, genes=genes)
    msg = str(exc.value)
    assert "target_gene_map" in msg
    assert "exclude_target_gene=False" in msg
    assert "A-1" in msg  # names an actually-unresolved label


def test_gene_level_panel_unchanged_without_a_map():
    # The no-regression control: bare-symbol labels resolved before the fix and must
    # still resolve, with no map and the identical score.
    real = _bulk(["A", "B", "ctrl"], [[1, 9], [9, 1], [0, 0]])
    pred = _bulk(["A", "B", "ctrl"], [[2, 2], [9, 1], [0, 0]])
    genes = np.array(["A", "B"])
    out = discrimination_score(pred_bulk=pred, real_bulk=real, control="ctrl",
                               distance="l1", rank_denominator="n",
                               control_source="pred", exclude_target_gene=True,
                               exclusion_scope="row", genes=genes)
    # identical to test_exclude_target_gene_drops_named_column's `drop` -- same 2-gene panel,
    # so same reason for pinning the scope
    assert out == pytest.approx({"A": 0.5, "B": 1.0})


def test_partial_resolution_is_allowed_and_logged(caplog):
    # A target whose own gene is not measured (or was CPM-filtered) excludes nothing.
    # That is ordinary, so it must NOT raise -- it must be reported. Mirrors the DE
    # side's partial-resolution behavior exactly.
    #
    # #289 raised that report from INFO to WARNING and rewrote it: it no longer asserts the
    # benign reading of a condition the resolver cannot test. Level and wording are pinned in
    # tests/test_exclusion_reporting_289.py. What this test owns is unchanged -- that a partial
    # resolution SCORES rather than raising, and that it is reported at all.
    real = _bulk(["A-1", "Z-9", "ctrl"], [[1, 9, 0], [9, 1, 0], [0, 0, 0]])
    pred = _bulk(["A-1", "Z-9", "ctrl"], [[2, 2, 0], [9, 1, 0], [0, 0, 0]])
    genes = np.array(["A", "B", "C"])
    with caplog.at_level("INFO", logger="cell_eval2.distances"):
        out = discrimination_score(pred_bulk=pred, real_bulk=real, control="ctrl",
                                   distance="l1", rank_denominator="n",
                                   control_source="pred", exclude_target_gene=True,
                                   genes=genes, target_gene_map={"A-1": "A"})
    assert set(out) == {"A-1", "Z-9"}
    # 1/2, not 1/3: the control is stripped by `prep.delta` before the resolver ever sees it.
    assert "1/2 labels resolved" in caplog.text


def test_map_pointing_at_an_unmeasured_gene_does_not_count_as_resolved():
    # A map that is present but wrong must not re-open the silent hole: mapping every
    # label to a gene outside the panel resolves nothing, so it raises.
    real, pred, genes, _ = _guide_panel()
    with pytest.raises(ValueError, match="NO perturbation resolves"):
        discrimination_score(pred_bulk=pred, real_bulk=real, control="ctrl",
                             distance="l1", exclude_target_gene=True, genes=genes,
                             target_gene_map={"A-1": "NOT_MEASURED",
                                              "B-1": "ALSO_ABSENT"})


def test_map_is_ignored_when_exclusion_is_off():
    # exclude_target_gene=False means no column is ever indexed; supplying a map must
    # not change the score (and must not raise on an unresolvable one).
    real, pred, genes, tgm = _guide_panel()
    kw = dict(pred_bulk=pred, real_bulk=real, control="ctrl", distance="l1",
              rank_denominator="n", control_source="pred", genes=genes,
              exclude_target_gene=False)
    assert discrimination_score(**kw, target_gene_map=tgm) == \
        pytest.approx(discrimination_score(**kw))
