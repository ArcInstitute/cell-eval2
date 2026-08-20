"""#289: a PARTIAL target-gene resolution is reported as a warning, and says what it costs.

`resolve_exclusion_columns` raises only when ZERO labels resolve (the all-quantifier that #248
installed). One resolving label satisfies that gate for the whole panel, while `cols` is still
built per row -- so every unresolved label keeps its own transcript in the ranked cosine vector,
which is exactly the #248 condition, on the subset the gate exists to protect. #289 measured it:
against a zero-DOWNSTREAM-knowledge model -- off-target biology drawn independently of the truth,
but the on-target knockdown reproduced -- whose correct `pds_cosine` is 0.7808, a map covering 30
of 40 targets reads 0.8468, and each unresolved target reads exactly 1.0000.

That 1.0000 is what THAT model gets, not a universal: the leaked coordinate is only decisive for a
submission that predicts the knockdown. A zero-effect prediction ties every real target at cosine
distance 1 and scores 0.5 under v2 midrank whether or not its gene was excluded. Both are pinned
below, because "keeping the column always wins" would be the easy wrong summary (codex review).

The old report sat at INFO and asserted the benign reading -- "ordinary when a target is not a
measured gene, or the CPM filter dropped it" -- of a condition the function never tested. This file
pins the new one.

**The GATE is unchanged, and that is a RULING, not an omission** (Alex, 2026-08-16: warn only).
All three raise rules are unsound -- see the `resolve_exclusion_columns` docstring -- and the
measurement that settled it is that all 200 competition targets resolve on all three official
contexts with no map at all, so this warning never fires on the scored path. `baseline.py`'s call
site (#253/#285, chunk 3) therefore resolves against an unchanged contract.

WHICH OF THESE ARE REGRESSION TESTS: the level test and the wording tests, which go red if the
INFO line is restored. The full-resolution, unchanged-return, unchanged-zero-raise and leak tests
pass on the old code too -- they pin the premise and the contract rather than the fix
(codex review).

A new file rather than an addition to `tests/test_target_resolution.py`, which chunk 3 owns.
"""

import logging

import numpy as np
import pytest

from cell_eval2 import distances
from cell_eval2.distances import resolve_exclusion_columns

_GENES = np.asarray(["A", "B", "C"], dtype=str)
_PERTS = np.array(["ctrl", "A-1", "B-1", "ZZZ-1"])
_MAP_FULL = {"A-1": "A", "B-1": "B", "ZZZ-1": "ZZZ_NOT_MEASURED"}
_MAP_GAPPED = {"A-1": "A", "B-1": "B"}                 # ZZZ-1 has no entry at all


def _resolve(caplog, labels, genes, **kw):
    caplog.clear()
    with caplog.at_level(logging.DEBUG, logger=distances.__name__):
        cols = resolve_exclusion_columns(labels, genes, **kw)
    return cols, caplog.records


def test_partial_resolution_reports_at_WARNING_not_INFO(caplog):
    """The level is the point. What is being reported is that some targets are scored WITHOUT the
    exclusion the caller asked for -- a change in what the metric means for them -- and an INFO
    line is not where an operator looks for that. Captured at DEBUG so an INFO regression shows up
    as a wrong level rather than as silence."""
    _, records = _resolve(caplog, _PERTS, _GENES, target_gene_map=_MAP_FULL)
    assert len(records) == 1
    assert records[0].levelno == logging.WARNING, (
        f"expected WARNING, got {records[0].levelname}")


def test_the_warning_states_what_an_unresolved_target_actually_gets(caplog):
    """The old wording asserted the harmless reading. The new one has to name the cost, or it is
    no more actionable than silence."""
    _, records = _resolve(caplog, _PERTS, _GENES, target_gene_map=_MAP_FULL)
    msg = records[0].getMessage()
    assert "WITHOUT exclusion" in msg
    assert "keeps its own transcript" in msg
    assert "1.0000" in msg, "the measured consequence belongs in the line"
    assert "#289" in msg
    assert "ZZZ-1" in msg, "the unresolved labels have to be nameable"
    # POSITIVE + NEGATIVE, so reverting the round-2 rewording alone goes red. The claim has to be
    # hedged ("CAN inflate") and attributed to the model that produces 1.0000, never stated as
    # something every submission gets (codex review round 2).
    assert "CAN inflate" in msg
    assert "against a model that" in msg, "the figure must be attributed, not universal"
    assert "whatever the submission predicts" not in msg
    assert "regardless of the submission" not in msg


def test_the_ZERO_RESOLVE_error_is_hedged_the_same_way():
    """The raise message made the same universal claim ("inflates the discrimination score") and
    was missed by the first pass -- the two messages describe the same mechanism and have to agree.
    """
    with pytest.raises(ValueError) as exc:
        resolve_exclusion_columns(np.array(["X-1", "Y-1"]), _GENES, target_gene_map=None)
    msg = str(exc.value)
    assert "CAN inflate" in msg
    assert "not at all for one with no effect" in msg


def test_the_warning_does_NOT_claim_to_know_which_case_it_is(caplog):
    """The defect was a line asserting the benign reading of an untested condition. Whatever
    replaces it must say the opposite: that (a) and (b) are indistinguishable here, and that the
    operator has to look. Same wording whether a map is in use or not -- an automatic split was
    tried and dropped (see the docstring: the control label always looks like a map gap)."""
    cases = [
        (_PERTS, {"target_gene_map": _MAP_FULL}),
        (_PERTS, {"target_gene_map": _MAP_GAPPED}),
        # no map at all: gene-level labels, one of which is not measured
        (np.array(["ctrl", "A", "NOPE"]), {"target_gene_map": None}),
    ]
    for labels, kw in cases:
        _, records = _resolve(caplog, labels, _GENES, **kw)
        msg = records[0].getMessage()
        assert "cannot tell the two apart" in msg
        # "the gene panel passed in", not bare "var_names": this resolver takes a `genes` array
        # and holds no AnnData, so a DIRECT caller has no `var_names` (Copilot review, PR #302).
        assert "Check them against the gene panel passed in" in msg
        assert "harmless ONLY if" in msg


def test_the_count_is_over_the_labels_this_call_was_HANDED(caplog):
    """A direct call that includes the control counts it as unresolved -- it has no target gene.
    The metric path does not: `discrimination_score` passes `pred_keys`, which `prep.delta` has
    already stripped of the control. Pinned because it is why an automatic map-gap split was
    dropped (see the resolver's docstring)."""
    _, direct = _resolve(caplog, _PERTS, _GENES, target_gene_map=_MAP_FULL)
    assert "2/4 labels resolved" in direct[0].getMessage()      # ctrl and ZZZ-1 unresolved
    _, no_ctrl = _resolve(caplog, _PERTS[1:], _GENES, target_gene_map=_MAP_FULL)
    assert "2/3 labels resolved" in no_ctrl[0].getMessage()     # only ZZZ-1 unresolved


def test_full_resolution_says_nothing(caplog):
    """No unresolved labels, no line. A warning on every clean run is a warning nobody reads."""
    cols, records = _resolve(caplog, np.array(["A", "B"]), _GENES)
    assert cols == {0: 0, 1: 1}
    assert records == []


# ----------------------------------------------------- the contract chunk 3 resolves against

def test_the_RETURN_VALUE_is_unchanged_by_the_reporting_change():
    """`baseline.py`'s call site (#253/#285) is chunk 3's work and resolves against this
    function's contract, not its log output. Pinned so the two can move independently."""
    assert resolve_exclusion_columns(_PERTS, _GENES, target_gene_map=_MAP_FULL) == {1: 0, 2: 1}
    assert resolve_exclusion_columns(_PERTS, _GENES, target_gene_map=_MAP_GAPPED) == {1: 0, 2: 1}
    # the slice/gate split is untouched
    assert resolve_exclusion_columns(
        np.array(["ZZZ-1"]), _GENES, target_gene_map=_MAP_FULL, gate_labels=_PERTS,
    ) == {}


def test_the_ZERO_resolve_raise_is_unchanged():
    """#289 argues the tolerance is unsound for one of the two cases it covers; it does not argue
    the raise should move. Whether a partial resolution should ALSO raise was ruled: warn only
    (2026-08-16), so this raise is the only one there is."""
    with pytest.raises(ValueError, match="NO perturbation resolves"):
        resolve_exclusion_columns(np.array(["X-1", "Y-1"]), _GENES, target_gene_map=None)


# ----------------------------------------------------- the leak itself, so the premise is tested

def test_an_unresolved_target_keeps_its_own_gene_and_scores_1_0():
    """#289's per-target decomposition, at the smallest size that shows it: a target whose label
    does not resolve is ranked with its own knocked-down transcript still in the vector, which is
    a coordinate predictable from the label alone. Its `pds_cosine` is 1.0000 while the resolved
    targets -- same model, same data -- are not.

    Deliberately a MODEL WITH NO DOWNSTREAM KNOWLEDGE: the predicted off-target biology is drawn
    independently of the real one, so any score above chance comes from the on-target dip alone.
    """
    from cell_eval2.metrics.discrimination import discrimination_score

    n_pert, n_bg, base, dip = 12, 30, 60.0, 0.45
    syms = [f"SYM{i}" for i in range(n_pert)]
    genes = np.asarray(syms + [f"bg{j}" for j in range(n_bg)], dtype=str)
    constructs = [f"{s}-1" for s in syms]
    g = len(genes)

    rng = np.random.default_rng(11)
    true_bio = rng.normal(0.0, 3.0, size=(n_pert, g))
    cheat_bio = rng.normal(0.0, 3.0, size=(n_pert, g))     # independent -> zero knowledge

    def bulks(bio):
        rows = [np.full(g, base)]                          # the control row
        for i in range(n_pert):
            mu = np.full(g, base) + bio[i]
            mu[i] *= dip                                   # the on-target knockdown
            rows.append(np.maximum(mu, 0.1))
        return np.array(["non-targeting", *constructs]), np.vstack(rows)

    real_bulk, pred_bulk = bulks(true_bio), bulks(cheat_bio)
    half = {constructs[i]: syms[i] for i in range(n_pert // 2)}   # covers 6 of 12

    got = discrimination_score(
        pred_bulk=pred_bulk, real_bulk=real_bulk, control="non-targeting",
        distance="cosine", rank_denominator="n-1", exclude_target_gene=True,
        control_source="real", genes=genes, target_gene_map=half,
    )
    resolved = [got[c] for c in constructs[:n_pert // 2]]
    unresolved = [got[c] for c in constructs[n_pert // 2:]]
    assert all(v == pytest.approx(1.0) for v in unresolved), (
        f"the leak is the premise of #289; unresolved targets read {unresolved}")
    assert max(resolved) < 1.0, (
        f"resolved targets must NOT be pinned at 1.0 for a zero-knowledge model: {resolved}")
    assert np.mean(unresolved) > np.mean(resolved)


def test_a_ZERO_EFFECT_prediction_is_NOT_helped_by_the_leak():
    """The counterexample that keeps the claim honest. The leaked coordinate only helps a
    submission that reproduces the knockdown; a prediction with no effect at all is equidistant
    from every real target (cosine distance 1 to all of them), so v2 midrank gives it 0.5 whether
    or not its own gene was excluded.

    Without this, "an unresolved target reads 1.0000" reads as a universal, and it is not
    (codex review). The warning in `distances.py` says "can inflate" for exactly this reason.
    """
    from cell_eval2.metrics.discrimination import discrimination_score

    n_pert, n_bg, base = 8, 20, 60.0
    syms = [f"SYM{i}" for i in range(n_pert)]
    genes = np.asarray(syms + [f"bg{j}" for j in range(n_bg)], dtype=str)
    constructs = [f"{s}-1" for s in syms]
    g = len(genes)

    rng = np.random.default_rng(3)
    true_bio = rng.normal(0.0, 3.0, size=(n_pert, g))

    real_rows = [np.full(g, base)]
    for i in range(n_pert):
        mu = np.full(g, base) + true_bio[i]
        mu[i] *= 0.45
        real_rows.append(np.maximum(mu, 0.1))
    real_bulk = (np.array(["non-targeting", *constructs]), np.vstack(real_rows))
    # every predicted profile IDENTICAL to the control: zero predicted effect everywhere
    pred_bulk = (np.array(["non-targeting", *constructs]),
                 np.vstack([np.full(g, base)] * (n_pert + 1)))

    half = {constructs[i]: syms[i] for i in range(n_pert // 2)}
    got = discrimination_score(
        pred_bulk=pred_bulk, real_bulk=real_bulk, control="non-targeting",
        distance="cosine", rank_denominator="n-1", exclude_target_gene=True,
        # tie_policy passed EXPLICITLY so the 0.5 below is a statement about midrank rather than
        # about whatever the signature happens to default to. It does NOT pin `vcc2026.yaml` --
        # changing the preset would not redden this; the preset tests already cover that drift
        # (codex review round 3 corrected the round-2 claim).
        tie_policy="midrank",
        control_source="real", genes=genes, target_gene_map=half,
    )
    resolved = [got[c] for c in constructs[:n_pert // 2]]
    unresolved = [got[c] for c in constructs[n_pert // 2:]]
    assert all(v == pytest.approx(0.5) for v in resolved + unresolved), (
        f"an all-tied row must midrank to 0.5; resolved {resolved} unresolved {unresolved}")
    assert np.mean(unresolved) == pytest.approx(np.mean(resolved)), (
        "the leak must give a zero-effect submission NO advantage")
