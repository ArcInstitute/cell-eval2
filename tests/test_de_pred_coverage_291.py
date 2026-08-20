"""Issue #291 -- the direction family's omission invariant is DENOMINATOR-ONLY.

These pin the measured behaviour of the SHIPPED pool (a pred/real inner join) so that a
future change to it is deliberate and its cost is visible. They are characterization tests:
the values asserted for `direction_reach_raw` under omission are the ones the metric returns
today, and three of them are the defect. Changing the pool changes them, which is the point
-- see `direction._reference_stats` for why the obvious repair scores omission HIGHER still,
and `de._warn_pred_gene_coverage` for the diagnostic that ships instead.

⚠️ This file is deliberately THRESHOLD-AGNOSTIC: `N_HEAD` is sized from
`REACH_PURITY_FLOOR`, so every assertion rescales and the suite stays green under either
floor. It characterizes the omission defect, not the floor -- the floor is pinned in
`tests/test_direction_reach.py`. Do not add a bare `N_HEAD == 9` here; that would make this
file fail for a reason it does not describe.
"""
from __future__ import annotations

import logging

import polars as pl
import pytest

from cell_eval2.de import TargetResolution, _warn_pred_gene_coverage, prepare_de
from cell_eval2.metrics.direction import (
    REACH_PURITY_FLOOR,
    de_direction_fidelity_yield_raw,
    de_direction_reach,
)

N_ADJUDICABLE = 80
TARGET = "T1"
# The head-miss block has to be big enough that purity NEVER recovers to the purity floor,
# which is what holds k* at 0 and makes deleting the block pay. k* = 0 needs
# (N - m)/N < P0, i.e. m > (1 - P0)*N -- 3 of 80 under the old P0 = 0.975, and 9 of 80 under
# REACH_PURITY_FLOOR = 0.9. Sized from the constant so this cannot drift silently again: at
# m = 3 the whole defect is invisible at the new floor (measured: the panel scores 1.0 both
# with and without the rows), which would have turned this file green while testing nothing.
def _min_head_misses(n: int, p0: float) -> int:
    """Smallest m with (n - m)/n < p0. SEARCHED, not `int((1-p0)*n) + 1`: 1 - 0.9 is
    0.09999999999999998 in binary, so the closed form returns 8 here and the panel then
    scores 1.0 instead of 0.0 -- the fixture would stop exercising the defect with nothing
    failing to say so (measured)."""
    m = 1
    while (n - m) / n >= p0:
        m += 1
    return m


N_HEAD = _min_head_misses(N_ADJUDICABLE, REACH_PURITY_FLOOR)   # 9 at P0 = 0.9
HEAD = tuple(f"g{i}" for i in range(N_HEAD))   # the WRONG calls, ranked first
TAIL = ("g78", "g79")              # two correct calls at the end of the ranking


def _tbl(rows: list[tuple]) -> pl.DataFrame:
    return pl.DataFrame(
        {"target": [r[0] for r in rows], "feature": [r[1] for r in rows],
         "log2_fold_change": [float(r[2]) for r in rows], "p_adj": [float(r[3]) for r in rows]},
        schema={"target": pl.String, "feature": pl.String,
                "log2_fold_change": pl.Float64, "p_adj": pl.Float64},
    )


def _panel(n_overcall: int = 0) -> tuple[list[tuple], list[tuple]]:
    """One target. The reference calls `N_ADJUDICABLE` genes significant, all up.

    The prediction agrees on every one EXCEPT the `N_HEAD` it ranks FIRST -- so
    `misses(k) <= (1 - P0)k` fails at every k <= 80 and `k*` is 0. `n_overcall` adds pred-side
    significant calls on genes the reference did NOT call, which is the regime where
    `fidelity_yield_raw`'s `max(n_pred, N_conf)` denominator is the pred-side one.
    """
    real: list[tuple] = []
    pred: list[tuple] = []
    for i in range(N_ADJUDICABLE):
        real.append((TARGET, f"g{i}", 1.0, 0.001))
        pred.append((TARGET, f"g{i}", -1.0 if i < N_HEAD else 1.0, 1e-6 + i * 1e-9))
    for j in range(n_overcall):
        real.append((TARGET, f"x{j}", 1.0, 1.0))
        pred.append((TARGET, f"x{j}", 1.0, 1e-5 + j * 1e-9))
    # The target's own gene, so `_require_resolution` passes. Anti-joined away by both
    # `_reference_stats` and `_ontarget_excluded_frame`, so it touches nothing measured.
    real.append((TARGET, TARGET, 0.0, 1.0))
    pred.append((TARGET, TARGET, 0.0, 1.0))
    return real, pred


def _score(real_rows, pred_rows) -> tuple[float, float]:
    prepared = prepare_de(
        _tbl(pred_rows), _tbl(real_rows), control="non-targeting", p_adj_threshold=0.05,
        target_resolution=TargetResolution({TARGET: TARGET}, 1),
    )
    reach = de_direction_reach(prepared, corrected=False, universe="adjudicated")
    return reach[TARGET], de_direction_fidelity_yield_raw(prepared)[TARGET]


def _drop(rows, features):
    return [r for r in rows if r[1] not in features]


def test_deleting_head_misses_raises_reach_raw():
    """The defect. `N_HEAD` (9 at the shipped floor) wrong calls at the head hold k* at 0;
    deleting exactly those rows
    leaves a pure prefix and k* jumps to it against an unchanged N_conf = 80.

    ⚠️ The jump SHRANK when the purity floor moved to 0.9 -- 0.0 -> 0.9625 became
    0.0 -> 0.8875 -- but the defect did not go away: it now takes 9 head misses instead of 3,
    because purity recovers to the floor once misses(80) <= (1 - P0)*80."""
    real, pred = _panel()
    assert _score(real, pred)[0] == 0.0
    assert _score(real, _drop(pred, HEAD))[0] == pytest.approx((80 - N_HEAD) / 80)


def test_n_conf_itself_is_omission_proof():
    """The premise that survives. `N_conf` is read from the real table alone, so the
    denominator does not move when the prediction drops rows -- reach_raw's jump above is
    entirely numerator, not a shrinking budget."""
    real, pred = _panel()
    from cell_eval2.metrics.direction import _reference_stats
    full = prepare_de(_tbl(pred), _tbl(real), control="non-targeting", p_adj_threshold=0.05,
                      target_resolution=TargetResolution({TARGET: TARGET}, 1))
    omitted = prepare_de(_tbl(_drop(pred, HEAD)), _tbl(real), control="non-targeting",
                         p_adj_threshold=0.05,
                         target_resolution=TargetResolution({TARGET: TARGET}, 1))
    assert (_reference_stats(full)["n_conf"].to_list()
            == _reference_stats(omitted)["n_conf"].to_list() == [N_ADJUDICABLE])


def test_omission_is_targeted_not_monotone():
    """Deleting TAIL misses LOWERS the score, so no blanket "omit rows" rule pays. A miss
    also buys depth: 39 matches then one miss gives purity(40) = 39/40 = 0.975 >= P0, so the
    miss ADDS a depth. Omission wins only where the deleted rows hold misses(k) > (1 - P0)k
    for every k -- i.e. at the head.

    ⚠️ Built from an otherwise-CORRECT panel, deliberately. Starting from `_panel()` leaves
    the N_HEAD head misses in place, which pins reach_raw at 0.0 both with and without the
    tail rows -- a strict inequality would be unprovable and a `<=` would pass on 0.0 == 0.0,
    testing nothing."""
    real, pred = _panel()
    # every call correct, so k* = 80 and reach_raw = 1.0
    pred_clean = [(t, f, abs(lfc) if f.startswith("g") else lfc, a)
                  for t, f, lfc, a in pred]
    assert _score(real, pred_clean)[0] == pytest.approx(1.0)
    # now make ONLY the two tail rows misses
    pred_tail_wrong = [(t, f, -lfc if f in TAIL else lfc, a) for t, f, lfc, a in pred_clean]
    with_tail = _score(real, pred_tail_wrong)[0]
    without_tail = _score(real, _drop(pred_tail_wrong, TAIL))[0]
    # BOTH ends asserted absolutely: 78 matches then 2 misses still reaches P0 at every
    # depth (purity(80) = 78/80 = 0.975 >= 0.9), so the misses BUY depth and reach stays 1.0.
    assert with_tail == pytest.approx(1.0)
    assert without_tail == pytest.approx(78 / 80)
    assert without_tail < with_tail


def test_deleting_a_row_is_weaker_than_declining_to_call_it():
    """⚠️ Deletion is NOT the strongest form of this move, so a fix aimed only at row
    absence would miss. `universe='adjudicated'` filters the pool on the REFERENCE's
    significance but RANKS it by the PREDICTION's p_adj, so setting p_adj_pred = 1.0 demotes
    a known-wrong call to the tail while keeping the row -- and scores ABOVE deleting it.
    This is also why re-admitting an omitted pair on a left join does not fix anything: a
    null p_adj_pred sorts to that same tail."""
    real, pred = _panel()
    demoted = [(t, f, lfc, 1.0 if f in HEAD else a) for t, f, lfc, a in pred]
    # Demoted to the tail, the N_HEAD misses buy depth instead of destroying it: the deepest
    # k still clearing the floor is 78 (71/78 = 0.910; 71/79 = 0.899 falls short).
    assert _score(real, demoted)[0] == pytest.approx(78 / 80)
    assert _score(real, demoted)[0] > _score(real, _drop(pred, HEAD))[0]


def test_fidelity_yield_raw_moves_only_when_over_calling():
    """`k/max(n_pred, N_conf)` is omission-proof exactly while n_pred <= N_conf, because the
    denominator is then the real-side one."""
    real, pred = _panel()
    assert _score(real, pred)[1] == _score(real, _drop(pred, HEAD))[1]
    real_oc, pred_oc = _panel(n_overcall=120)
    # n_pred = 80 + 120 = 200 > N_conf = 80, so the denominator is the PRED side's; the only
    # non-matches are the N_HEAD head misses. Written from N_HEAD rather than as a literal so
    # it tracks the purity floor the fixture is sized from.
    assert _score(real_oc, pred_oc)[1] == pytest.approx((200 - N_HEAD) / 200)
    assert _score(real_oc, _drop(pred_oc, HEAD))[1] == pytest.approx(1.0)


def test_coverage_warning_reports_the_omitted_significant_pairs(caplog):
    real, pred = _panel()
    with caplog.at_level(logging.WARNING, logger="cell_eval2.de"):
        _warn_pred_gene_coverage(_tbl(real), _tbl(_drop(pred, HEAD)), p_adj_threshold=0.05)
    assert f"omits {N_HEAD} of the 80 reference-significant" in caplog.text
    assert "#291" in caplog.text


def test_coverage_warning_is_silent_on_complete_coverage(caplog):
    real, pred = _panel()
    with caplog.at_level(logging.WARNING, logger="cell_eval2.de"):
        _warn_pred_gene_coverage(_tbl(real), _tbl(pred), p_adj_threshold=0.05)
    assert "reference-significant" not in caplog.text


def _prepare(pred_rows, real_rows, resolution):
    return prepare_de(_tbl(pred_rows), _tbl(real_rows), control="non-targeting",
                      p_adj_threshold=0.05, target_resolution=resolution)


def test_coverage_warning_is_WIRED_into_the_metrics_that_it_describes(caplog):
    """The helper-level tests above would all stay green if the production call were deleted,
    so this one drives a real metric.

    ⚠️ It is emitted from `_warn_coverage_once`, NOT from a constructor: that helper carries
    its own memo and is invoked by the eleven target-excluding direction metrics, which are
    the ones the message is about. Constructing the PreparedDE must therefore be SILENT, and
    running one of the eleven must emit -- both halves asserted here, because the first is
    what makes the placement observable."""
    real, pred = _panel()
    resolution = TargetResolution({TARGET: TARGET}, 1)
    with caplog.at_level(logging.WARNING, logger="cell_eval2.de"):
        prepared = _prepare(_drop(pred, HEAD), real, resolution)
    assert "reference-significant" not in caplog.text, "constructing must not warn"
    with caplog.at_level(logging.WARNING, logger="cell_eval2.de"):
        de_direction_reach(prepared, corrected=False, universe="adjudicated")
    assert f"omits {N_HEAD} of the 80 reference-significant" in caplog.text


def test_coverage_warning_is_silent_for_the_v0_5_0_direction_metrics(caplog):
    """The three v0.5.0 metrics read the UNFILTERED `_direction_frame` and never reach
    `_reference_stats`, so the on-target-excluded population this warning counts over is not
    theirs. Selecting only those must therefore stay quiet."""
    from cell_eval2.metrics.direction import de_direction_precision
    real, pred = _panel()
    prepared = _prepare(_drop(pred, HEAD), real, TargetResolution({TARGET: TARGET}, 1))
    with caplog.at_level(logging.WARNING, logger="cell_eval2.de"):
        de_direction_precision(prepared)
    assert "reference-significant" not in caplog.text


def test_the_resolution_is_FORWARDED_to_the_warning(caplog):
    """Wiring test for the `target_resolution=` argument specifically, which the test above
    cannot cover: `T1 -> T1` maps onto a non-significant row, so the anti-join removes
    nothing and dropping the argument would change no count.

    Here the target resolves to `g0` -- a reference-SIGNIFICANT feature -- and the pred table
    omits exactly that one row. With the resolution forwarded, that is an on-target pair
    `_reference_stats` never counts, so nothing is reported. Without it, the run reports a
    one-pair gap. Note `feature != target` would also miss this: the mapped feature is not
    the target's label, which is why the exclusion follows `target_resolution`."""
    real, pred = _panel()
    omitted = _drop(pred, ("g0",))
    with caplog.at_level(logging.WARNING, logger="cell_eval2.de"):
        de_direction_reach(_prepare(omitted, real, TargetResolution({TARGET: "g0"}, 1)),
                           corrected=False, universe="adjudicated")
    assert "reference-significant" not in caplog.text
    # ...and the same input, resolving the target to its own non-significant row instead,
    # leaves the g0 omission visible
    with caplog.at_level(logging.WARNING, logger="cell_eval2.de"):
        de_direction_reach(_prepare(omitted, real, TargetResolution({TARGET: TARGET}, 1)),
                           corrected=False, universe="adjudicated")
    assert "omits 1 of the 80 reference-significant" in caplog.text


def test_coverage_warning_ignores_pairs_the_reference_did_not_call(caplog):
    """Real-side significance decides what is counted, so an omitted NON-significant pair is
    not a coverage gap -- otherwise ordinary gene-filter differences would drown the signal."""
    real, pred = _panel(n_overcall=5)
    with caplog.at_level(logging.WARNING, logger="cell_eval2.de"):
        _warn_pred_gene_coverage(_tbl(real), _tbl(_drop(pred, ("x0", "x1", "x2"))),
                                 p_adj_threshold=0.05)
    assert "reference-significant" not in caplog.text


def test_reading_reference_stats_does_not_swallow_the_warning(caplog):
    """The warning has its OWN memo rather than riding on `_reference_stats`'s build.

    `_reference_stats` has direct callers that are not metrics -- a diagnostic reading
    `n_conf`, or `test_n_conf_itself_is_omission_proof` above. If the emission lived inside
    that build, such a call would consume the memo and leave a scored metric run later on the
    same PreparedDE silent, which is worst exactly when the diagnostic call had logging
    suppressed. Drive that order explicitly."""
    from cell_eval2.metrics.direction import _reference_stats
    real, pred = _panel()
    prepared = _prepare(_drop(pred, HEAD), real, TargetResolution({TARGET: TARGET}, 1))
    with caplog.at_level(logging.CRITICAL, logger="cell_eval2.de"):
        _reference_stats(prepared)          # a non-metric caller, warnings suppressed
    with caplog.at_level(logging.WARNING, logger="cell_eval2.de"):
        de_direction_reach(prepared, corrected=False, universe="adjudicated")
    assert f"omits {N_HEAD} of the 80 reference-significant" in caplog.text
