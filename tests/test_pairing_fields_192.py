"""`cli._check_baseline_config` — which fields a LOOSE baseline/submission pairing compares.

The bundle path's copy of this list is covered in `tests/test_real_bundle_score.py`. This file
is the other copy, at `cli.py:194`, and it is a separate file rather than an addition to
`tests/test_cli_baseline.py` because `cli.py` is a region-split file: only the pairing loop
belongs to this work, and a new file cannot collide with an edit to the rest of it.

⚠️ Driven DIRECTLY, with hand-written meta dicts, rather than through `run` + `baseline`. The
CLI-level tests in `test_cli_baseline.py` pay for two full metric runs to produce a pair of
stamps; what is under test here is which KEYS the loop reads, and a real run cannot make two
stamps differ in exactly one chosen field without also moving `config_digest`.
"""
import json
from argparse import Namespace

import pytest

from cell_eval2.cli import _check_baseline_config

# A pairing that agrees on everything. Values are arbitrary -- the loop compares them to each
# other, never against the world.
AGREED = {
    "cell_eval2_version": "0.13.0",
    "config_digest": "d" * 8,
    "comparator": "bulk_lognorm",
    "source_fingerprint": "s" * 8,
    "source_fingerprint_strict": True,
    "resolved_device": "cpu",
    "resolved_de_backend": None,
    "input_type_real_effective": "counts",
    "input_type_pred_effective": "counts",
    "de_real_fingerprint": None,
    "de_pred_fingerprint": None,
}


def _args(tmp_path, base, user, allow=False):
    b, u = tmp_path / "baseline_meta.json", tmp_path / "run_meta.json"
    b.write_text(json.dumps(base))
    u.write_text(json.dumps(user))
    return Namespace(baseline_meta=str(b), user_meta=str(u),
                     baseline_agg=str(tmp_path / "baseline_agg.csv"),
                     user_agg=str(tmp_path / "agg_results.csv"),
                     allow_config_mismatch=allow)


def test_an_agreeing_pair_passes(tmp_path):
    """The control. Without it, every refusal test below would also pass on a loop that
    refused unconditionally."""
    _check_baseline_config(_args(tmp_path, AGREED, AGREED))


def test_a_pred_side_input_type_difference_is_ACCEPTED(tmp_path):
    """#192, on the loose path. The baseline's prediction is a fractional mean of counts that
    the matrix-space lock pulls back to `counts`; a VCC submission is commonly log-normalized.
    Both values are correct and both sides are converted into the same metric space before any
    metric is computed.

    ⚠️ The point is not that it warns -- it is that it does not raise. Before #192 this pairing
    aborted, and the only way past was `--allow-config-mismatch`, which ALSO downgrades
    config_digest, source_fingerprint, resolved_de_backend, resolved_device and
    de_real_fingerprint. The common case forced users to disarm the five checks the function
    exists for, which is what makes this a fix rather than a relaxation."""
    _check_baseline_config(
        _args(tmp_path, AGREED, {**AGREED, "input_type_pred_effective": "lognorm"}))


def test_a_pred_side_DE_fingerprint_difference_is_ACCEPTED(tmp_path):
    """Its neighbour, and the precedent #192 is applying. Unlike the bundle path (#291), the
    loose path deliberately still permits a supplied `--de-pred`: `run --de-pred` against a
    diagnostic baseline is the isolator the metric campaigns are built on, and there is no
    enrolment here to protect."""
    _check_baseline_config(
        _args(tmp_path, AGREED, {**AGREED, "de_pred_fingerprint": "f" * 8}))


@pytest.mark.parametrize("field,bad", [
    ("input_type_real_effective", "lognorm"),   # the side the two runs genuinely SHARE
    ("cell_eval2_version", "0.12.0"),
    ("config_digest", "nonesuch"),
    ("comparator", "lognorm"),
    ("source_fingerprint", "nonesuch"),
    ("source_fingerprint_strict", False),
    ("resolved_device", "cuda"),
    ("resolved_de_backend", "gpudge"),
    ("de_real_fingerprint", "f" * 8),
])
def test_every_OTHER_field_is_still_fatal(tmp_path, field, bad):
    """The other edge of #192, and what keeps it honest. `input_type_real_effective` is first
    on purpose: it is one letter away from the field that was dropped, and without this the
    suite would stay green if BOTH went."""
    with pytest.raises(SystemExit, match=field):
        _check_baseline_config(_args(tmp_path, AGREED, {**AGREED, field: bad}))


def test_a_dropped_field_is_not_reported_as_MISSING_either(tmp_path):
    """The loop is fail-closed on an absent key, so a field it no longer reads must be gone
    from the list rather than merely unequal -- otherwise a stamp predating the field would
    abort on `input_type_pred_effective: missing from ...`."""
    without = {k: v for k, v in AGREED.items() if k != "input_type_pred_effective"}
    _check_baseline_config(_args(tmp_path, without, without))


def test_allow_config_mismatch_still_waives_the_fields_that_remain(tmp_path, caplog):
    """#192 removed a field from the comparison; it did not touch the waiver. The blanket flag
    is still exactly as blanket as before for everything left in the list."""
    with caplog.at_level("WARNING"):
        _check_baseline_config(
            _args(tmp_path, AGREED, {**AGREED, "config_digest": "nonesuch"}, allow=True))
    assert any("config_digest" in r.getMessage() for r in caplog.records)
