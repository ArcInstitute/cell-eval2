"""#288: an ASYMMETRIC v2 run falls off the group-sum comparator, and now says so.

A new file rather than an addition to ``tests/test_comparator_resolution.py``: that one is a
cross-chunk integration file with a single owner, and #193 is working the same mixed
counts/lognorm behaviour there. The resolution TABLE is asserted over there and is unchanged;
what is asserted here is only the log line #288 asked for.

The defect #288 measured is that the declared/detected input type -- not the predicted biology --
decides which comparator scores ``expr_mse_unbiased_capped_norm``, and the two disagree by +0.8 to
+1.4 for the same predicted expression, largest where the prediction is most accurate. The
resolution itself is the documented rule and is not being changed; falling back SILENTLY is what
is being changed.

WHICH OF THESE ARE REGRESSION TESTS: only `test_v2_asymmetric_declaration_warns_and_names_the_issue`
goes red if the warning is reverted. The silent / symmetric / v1 / preset tests below pass on the
old code too -- they are invariant guards pinning what must NOT have changed (the resolution table
itself, and that the new warning does not fire where it should not). Both kinds earn their place;
they are labelled so nobody cites the second kind as evidence the fix works (codex review).
"""

import logging

import pytest

from cell_eval2 import norm


def _resolve(caplog, *, version, pred, real):
    caplog.clear()
    with caplog.at_level(logging.WARNING, logger=norm.__name__):
        got = norm.resolve_comparator(
            version=version, pred_input_type=pred, real_input_type=real,
        )
    return got, [r for r in caplog.records if r.levelno >= logging.WARNING]


@pytest.mark.parametrize("pred,real", [("counts", "lognorm"), ("lognorm", "counts")])
def test_v2_asymmetric_declaration_warns_and_names_the_issue(caplog, pred, real):
    got, warnings = _resolve(caplog, version="v2", pred=pred, real=real)
    assert got == "lognorm"
    assert len(warnings) == 1, f"expected exactly one warning, got {warnings}"
    msg = warnings[0].getMessage()
    assert "bulk_lognorm" in msg and "lognorm" in msg
    assert "#288" in msg, "the warning has to be traceable to the measurement behind it"
    assert repr(pred) in msg and repr(real) in msg, "it must name which side was declared what"
    # The remedy must read as a DATA requirement, not a declaration one (Copilot review of PR
    # #302). "Declare both sides as counts to stay on 'bulk_lognorm'" invited exactly the wrong
    # fix -- re-declaring a log-normalized matrix as counts. `vcc2026` catches that, since
    # `validate_input_type` rejects non-integer data declared counts, but a run with
    # `allow_fractional_counts=True`, or any v1 run, would score nonsense silently.
    assert "must actually BE counts" in msg
    assert "re-declaring it does not reconstruct them" in msg
    assert "Declare both sides as counts" not in msg


def test_v2_counts_on_both_sides_stays_on_bulk_lognorm_and_is_silent(caplog):
    got, warnings = _resolve(caplog, version="v2", pred="counts", real="counts")
    assert got == "bulk_lognorm"
    assert warnings == []


def test_v2_lognorm_on_BOTH_sides_is_silent(caplog):
    """Deliberately not warned about. The warning is for a run whose comparator changed because
    ONE side's declaration changed; a run that declared both sides lognorm was never going to get
    the group-sum comparator, so there is no surprise to report."""
    got, warnings = _resolve(caplog, version="v2", pred="lognorm", real="lognorm")
    assert got == "lognorm"
    assert warnings == []


@pytest.mark.parametrize("pred,real", [
    ("counts", "counts"), ("counts", "lognorm"),
    ("lognorm", "counts"), ("lognorm", "lognorm"),
])
def test_v1_never_warns(caplog, pred, real):
    """v1 never had the group-sum comparator to fall off -- it reproduces upstream cell-eval and
    is pinned by tests/test_v1_gate.py. Warning there would be noise on every v1 run."""
    got, warnings = _resolve(caplog, version="v1", pred=pred, real=real)
    assert got == "lognorm"
    assert warnings == []


def test_the_competition_preset_cannot_reach_the_fallback():
    """The reachability claim, asserted rather than asserted-about. `vcc2026` pins both knobs
    that would be needed, and both are inside the frozen rule digest."""
    from cell_eval2.config import EvalConfig

    cfg = EvalConfig.from_preset("vcc2026")
    assert cfg.input_type == "counts"
    assert cfg.autodetect_input_type is False
    assert norm.resolve_comparator(
        version=cfg.version, pred_input_type=cfg.input_type, real_input_type=cfg.input_type,
    ) == "bulk_lognorm"
