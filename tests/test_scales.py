import hashlib
import json
import pytest

from cell_eval2.catalog import CATALOG, PROFILES
from cell_eval2.scales import SCALES, Scale, ScaleEntry, build_scale, resolve_scales
from cell_eval2.scoring import Scoring

NAME = "low-random_high-1_v10"

# (metric, base, direction, anchor, clamp_low) -- the table the spec froze.
EXPECTED = [
    ("expr_mse_unbiased_capped_norm", 1.0, "lower", 0.0, -6.0),
    ("de_wilcoxon_lfc_nmae", 1.0, "lower", 0.0, -1.0),
    ("pds_cosine", 0.5, "higher", 1.0, -1.0),
    ("de_wilcoxon_direction_fidelity_yield_raw", 0.5, "higher", 1.0, -1.0),
    ("de_wilcoxon_direction_reach_raw", 0.0, "higher", 1.0, 0.0),
    ("de_wilcoxon_sig_jaccard", 0.0, "higher", 1.0, 0.0),
]


def _up(base=0.5):
    return ScaleEntry(base, Scoring(scored=True, direction="higher", anchor=1.0))


def test_registry_contains_the_shipped_scale():
    assert NAME in SCALES
    assert isinstance(SCALES[NAME], Scale)


def test_shipped_scale_matches_the_frozen_table():
    entries = SCALES[NAME].entries
    assert set(entries) == {m for m, *_ in EXPECTED}
    for metric, base, direction, anchor, clamp_low in EXPECTED:
        e = entries[metric]
        assert e.base == base, metric
        assert e.scoring.direction == direction, metric
        assert e.scoring.anchor == anchor, metric
        assert e.scoring.penalty == "none", metric
        assert e.scoring.clamp_low == clamp_low, metric
        assert e.scoring.clamp_high == 1.0, metric
        assert e.scoring.scored is True, metric


def test_shipped_scale_covers_exactly_the_SCORED_vcc2026_members():
    # Widened by #257 and again by #264: the profile carries FOUR unscored diagnostics --
    # the derived metric's two components, the uncapped audit sibling, and
    # `expr_real_mass_ratio` -- which a scale must NOT carry: they are in gene-averaged
    # expression units (or a bare ratio) and one is submitter-gameable. Coverage is of what
    # the profile SCORES.
    scored = {m for m in PROFILES["vcc2026"] if CATALOG[m].scoring.scored}
    unscored = set(PROFILES["vcc2026"]) - scored
    assert unscored, "no unscored members: this test has degenerated back into the old one"
    assert set(SCALES[NAME].entries) == scored
    assert not (set(SCALES[NAME].entries) & unscored), (
        f"the scale scores unscored diagnostics: {sorted(set(SCALES[NAME].entries) & unscored)}"
    )


def test_v1_through_v9_are_retired_and_v10_is_the_only_shipped_scale():
    from cell_eval2.scales import SCALES
    assert "low-random_high-1_v1" not in SCALES, (
        "v1 keys expr_mse_unbiased_norm, which #257 removed; build_scale validates at import "
        "that every key names a catalog metric, so keeping it makes the PACKAGE unimportable"
    )
    assert "low-random_high-1_v2" not in SCALES, (
        "v2 contains pds_cosine, whose comparator moved in #264; keeping the name would "
        "silently redefine a shipped scale"
    )
    assert "low-random_high-1_v3" not in SCALES, (
        "v3 contains expr_mse_unbiased_capped_norm, whose comparator moved in #264; keeping "
        "the name would silently redefine a shipped scale"
    )
    assert "low-random_high-1_v4" not in SCALES, (
        "v4 was scored at bulk_target_sum=1e6; #268 moved it to 5e4, which shifts every "
        "scored value. Keeping the name would let an old column bind to new numbers"
    )
    assert "low-random_high-1_v5" not in SCALES, (
        "v5 keys pds_cosine, whose TIE HANDLING changed in #282 -- an all-tied row scored "
        "the target's alphabetical index and now scores 0.5. The table is byte-identical, "
        "which is exactly the 'what a keyed metric MEANS changed' case the immutability "
        "rule names: keeping the name would let an old column bind to a new definition"
    )
    assert "low-random_high-1_v6" not in SCALES, (
        "v6 keys de_wilcoxon_sig_jaccard, de_wilcoxon_lfc_nmae and "
        "expr_mse_unbiased_capped_norm, all THREE of which stopped scoring each perturbation's "
        "own target gene in #172 -- so every one of them is computed over a different gene set "
        "than v6 published. The table is byte-identical for the third mint running, which is "
        "the same 'what a keyed metric MEANS changed' case as v5: keeping the name would let a "
        "v6-headed column bind to the new definition"
    )
    assert "low-random_high-1_v7" not in SCALES, (
        "v7 keys de_wilcoxon_direction_reach_raw, whose purity floor moved 1 - alpha/2 -> "
        "REACH_PURITY_FLOOR = 0.9, so it is computed under a different rule than v7 "
        "published. Byte-identical table for the fourth mint running -- the same 'what a "
        "keyed metric MEANS changed' case as v6 -> v7: keeping the name would let a "
        "v7-headed column bind to the new definition"
    )
    assert "low-random_high-1_v8" not in SCALES, (
        "v8 was minted for the direction_reach purity floor and retired by #271: "
        "`prep._grouped_sums` reduces WIDE, which moves the values keyed under this name for "
        "pds_cosine and expr_mse_unbiased_capped_norm while the table stands still -- so a "
        "v8-headed column would span two GROUP-SUM eras. ⚠️ Neither v7 nor v8 ever shipped in a "
        "RELEASE (v0.13.0 ships v6); both were minted inside this same unreleased cycle, so "
        "retiring them orphans no column defined by a TAGGED release"
    )
    assert list(SCALES) == ["low-random_high-1_v10"]


def test_the_v10_table_is_BYTE_IDENTICAL_to_v8s_minus_the_name():
    """The #343 + #348 mint changed the NAME and nothing else -- proved, not asserted in prose.

    `_v10` is another mint whose table is byte-identical to its predecessor's, and the
    claim matters twice over: it is what makes "the registry versions the name, not the numbers"
    true, and it is what makes the moved `scales_digest()` attributable to the rename alone.

    The comparison is against `_v8`'s payload as a LITERAL rather than against `EXPECTED` above:
    `EXPECTED` pins five fields of each entry, so comparing to it would prove only that this file
    agrees with itself. The literal covers every field `scale_payload` serializes -- including
    `metric_min`, which the `_v7` -> `_v8` cycle added, and `allow_negative_baseline` -- so a
    future schema change moves it too, which is the point.

    ⚠️ The literal was GENERATED from `_v8`'s own payload at the previous commit and checked equal
    to `_v8`'s at mint time, not hand-typed. The first hand-typed attempt got two fields wrong
    (`allow_negative_baseline` missing, `fidelity_yield_raw`'s `clamp_low` written as 0.0 instead
    of -1.0) and this assertion caught both, which is the only reason to trust it now.
    """
    from cell_eval2.scales import SCALES, scale_payload

    v8_payload_minus_name = {
        "entries": {
            "expr_mse_unbiased_capped_norm": {
                "base": 1.0, "scored": True, "direction": "lower", "anchor": 0.0,
                "penalty": "none", "penalty_exponent": None, "penalty_cap": None,
                "clamp_low": -6.0, "clamp_high": 1.0, "metric_min": None,
                "allow_negative_baseline": False
            },
            "de_wilcoxon_lfc_nmae": {
                "base": 1.0, "scored": True, "direction": "lower", "anchor": 0.0,
                "penalty": "none", "penalty_exponent": None, "penalty_cap": None,
                "clamp_low": -1.0, "clamp_high": 1.0, "metric_min": None,
                "allow_negative_baseline": False
            },
            "pds_cosine": {
                "base": 0.5, "scored": True, "direction": "higher", "anchor": 1.0,
                "penalty": "none", "penalty_exponent": None, "penalty_cap": None,
                "clamp_low": -1.0, "clamp_high": 1.0, "metric_min": None,
                "allow_negative_baseline": False
            },
            "de_wilcoxon_direction_fidelity_yield_raw": {
                "base": 0.5, "scored": True, "direction": "higher", "anchor": 1.0,
                "penalty": "none", "penalty_exponent": None, "penalty_cap": None,
                "clamp_low": -1.0, "clamp_high": 1.0, "metric_min": None,
                "allow_negative_baseline": False
            },
            "de_wilcoxon_direction_reach_raw": {
                "base": 0.0, "scored": True, "direction": "higher", "anchor": 1.0,
                "penalty": "none", "penalty_exponent": None, "penalty_cap": None,
                "clamp_low": 0.0, "clamp_high": 1.0, "metric_min": None,
                "allow_negative_baseline": False
            },
            "de_wilcoxon_sig_jaccard": {
                "base": 0.0, "scored": True, "direction": "higher", "anchor": 1.0,
                "penalty": "none", "penalty_exponent": None, "penalty_cap": None,
                "clamp_low": 0.0, "clamp_high": 1.0, "metric_min": None,
                "allow_negative_baseline": False
            },
        },
    }
    def canonical(payload) -> bytes:
        """Exactly how `scales_digest` serializes, so this compares what the digest compares."""
        return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()

    # ⚠️ BYTES, not `==` on dicts. Python holds `1 == 1.0`, `True == 1` and `0.0 == -0.0`, all of
    # which JSON serializes DIFFERENTLY -- so a dict comparison can pass while the digest moves,
    # which is the one thing this test exists to rule out (codex review round 7).
    got = scale_payload(SCALES[NAME])
    assert got["name"] == "low-random_high-1_v10"
    got_minus_name = {k: v for k, v in got.items() if k != "name"}
    assert canonical(got_minus_name) == canonical(v8_payload_minus_name), (
        "the _v10 table is NOT byte-identical to _v8's (and so to _v9's). Either a number moved with the mint -- in "
        "which case say so in scales.py's history, since the whole point of a mint is that only "
        "the name changed -- or scale_payload gained a field, which is its own kind of bump"
    )

    # And the literal is TIED TO a shipped identity rather than being a hopeful transcription:
    # with a previous name put back, its digest must be the one that name shipped under. Those
    # numbers are in this file's own history comment and in the CHANGELOG, so a wrong literal
    # cannot agree with them by accident. Both predecessors are checked, because `_v10` is the
    # second mint running to reuse the same table.
    for prev, want in (("low-random_high-1_v8",
                        "22b3d6b1402dd28650e01b44d15d07f509571ba0460630b7fccf311149cef5a8"),
                       ("low-random_high-1_v9",
                        "8542ae142610caef580859c2e1526bc1754b3ee2d61ffd274a2236117d36d83c")):
        prev_digest = hashlib.sha256(canonical([{**v8_payload_minus_name,
                                                 "name": prev}])).hexdigest()
        assert prev_digest == want, (
            f"the literal above is not {prev}'s payload: it digests to {prev_digest}, not the "
            f"digest {prev} shipped under. Fix the literal, not this assertion"
        )
    # ...and the same payload under the new name is what the registry now digests to.
    assert hashlib.sha256(canonical([got])).hexdigest() == FROZEN_DIGEST


def test_v10_keys_the_derived_metric_with_an_unchanged_base():
    from cell_eval2.scales import SCALES
    entry = SCALES[NAME].entries["expr_mse_unbiased_capped_norm"]
    assert entry.base == 1.0, "the base is unchanged in VALUE -- what changed is that it is true"
    assert entry.scoring.clamp_low == -6.0
    assert entry.scoring.clamp_high == 1.0
    assert entry.scoring.direction == "lower"
    assert entry.scoring.anchor == 0.0


def test_entries_are_read_only():
    with pytest.raises(TypeError):
        SCALES[NAME].entries["pds_cosine"] = _up()


def test_registry_is_read_only():
    with pytest.raises(TypeError):
        SCALES["another"] = SCALES[NAME]


def test_build_rejects_a_degenerate_entry():
    # base == anchor -> D == 0. Exactly what a measured baseline can hit and a scale must not.
    with pytest.raises(ValueError, match="degenerate"):
        build_scale("x_v1", "d", {"pds_cosine": _up(1.0)})


def test_build_rejects_a_wrong_side_base():
    # For higher/anchor-1 the base must be BELOW the anchor. 1.2 gives D = -0.2, which would
    # invert the ranking silently rather than fail.
    with pytest.raises(ValueError, match="degenerate"):
        build_scale("x_v1", "d", {"pds_cosine": _up(1.2)})


def test_build_rejects_an_unscored_entry():
    with pytest.raises(ValueError, match="scored=True"):
        build_scale("x_v1", "d", {"pds_cosine": ScaleEntry(0.5, Scoring(scored=False))})


def test_build_rejects_an_unknown_metric():
    with pytest.raises(ValueError, match="unknown metric"):
        build_scale("x_v1", "d", {"not_a_metric": _up()})


def test_build_rejects_two_spellings_of_one_metric():
    with pytest.raises(ValueError, match="twice"):
        build_scale("x_v1", "d",
                    {"pds_cosine": _up(), "discrimination_score_cosine": _up()})


def test_build_canonicalizes_a_v1_spelling():
    s = build_scale("x_v1", "d", {"discrimination_score_cosine": _up()})
    assert set(s.entries) == {"pds_cosine"}


@pytest.mark.parametrize("reserved", ["metric", "from_baseline", "from_reference"])
def test_build_rejects_a_reserved_column_name(reserved):
    with pytest.raises(ValueError, match="reserved"):
        build_scale(reserved, "d", {"pds_cosine": _up()})


def test_build_rejects_an_empty_scale():
    with pytest.raises(ValueError, match="at least one"):
        build_scale("x_v1", "d", {})


def test_build_rejects_a_non_entry_value():
    with pytest.raises(TypeError, match="ScaleEntry"):
        build_scale("x_v1", "d", {"pds_cosine": 0.5})


def test_resolve_none_is_empty():
    assert resolve_scales(None) == []


def test_resolve_accepts_a_bare_string():
    assert [s.name for s in resolve_scales(NAME)] == [NAME]


def test_resolve_accepts_a_sequence():
    assert [s.name for s in resolve_scales([NAME])] == [NAME]


def test_resolve_rejects_an_unknown_name():
    with pytest.raises(ValueError, match="unknown scale"):
        resolve_scales("nope_v1")


def test_resolve_rejects_a_repeated_name():
    with pytest.raises(ValueError, match="twice"):
        resolve_scales([NAME, NAME])


# --- freeze -----------------------------------------------------------------------------
#
# A shipped scale is IMMUTABLE: any change to any field that moves a number mints a new
# _v<n> rather than redefining a name. This digest is what makes that structural instead of
# a convention. If you are here because this test failed, the question is NOT "what is the
# new hash" -- it is "should this have been a _v2?". Update the constant only for a scale
# that has never been published.
#
# Moved for #172 (Alex, 2026-08-17): `_v6` -> `_v7`. The registry TABLE is byte-identical --
# `scale_payload` carries `name`, so what moved this digest is the RENAME, not a field. That is
# precisely the point: three keyed members (`de_wilcoxon_sig_jaccard`, `de_wilcoxon_lfc_nmae`,
# `expr_mse_unbiased_capped_norm`) stopped scoring each perturbation's own target gene, so a
# `_v6`-headed column produced under the new definition would be the silent rebinding this
# freeze exists to prevent. Precedent: #282's `_v5` -> `_v6`, the same "identical table, new
# meaning" case.
#
# Moved AGAIN, and this one is a SCHEMA change that moved no number: `Scoring` gained
# `metric_min` and `scale_payload` now serializes it, so every entry's dict gained a
# `"metric_min": null` key. No shipped scale is unfloored, so none carries a non-None value and
# none of their SCORES moved -- proved BIT-EXACTLY by
# `test_the_metric_min_bump_moved_no_shipped_score` below, which replays the pre-change values
# via `float.hex()`. A `_v8` would have been the wrong answer here: minting one says the
# reference points changed, and unlike the `_v7` rename above, nothing about what a member
# means moved with it.
#
# Moved AGAIN for the purity floor (Alex, 2026-08-17): `_v7` -> `_v8`. Unlike the `metric_min`
# bump above, this one IS a mint -- `de_wilcoxon_direction_reach_raw`'s purity floor moved
# `1 - alpha/2` -> `REACH_PURITY_FLOOR`, so what a keyed member MEANS changed while the table
# stayed byte-identical. Both moves are in that digest: the rename AND the new payload key.
#
# Moved AGAIN for #271 (Alex, 2026-08-18): `_v8` -> `_v9`. A mint one rung LOWER than the three
# before it -- `prep._grouped_sums` reduces WIDE, so what moved is not a member's policy nor even
# its arithmetic but the PSEUDOBULK that arithmetic reads, for `pds_cosine` and
# `expr_mse_unbiased_capped_norm`. Table byte-identical again: PROVED, not asserted --
# `test_the_v10_table_is_BYTE_IDENTICAL_to_v8s_minus_the_name` ABOVE compares the canonical JSON
# bytes, and ties the literal it compares against to the digest `_v8` shipped under. So the
# ONLY thing moving this digest is the rename, which is exactly what a mint is for.
# Moved AGAIN for #343 + #348 (2026-08-19): `_v9` -> `_v10`, the FOURTH "what a keyed metric MEANS
# changed" mint and the first to cover two such changes in one wave. #343 removed every panel target
# gene from `pds_cosine`'s feature space (and shipped in `1c05408` with NO mint, so `_v9` had already
# begun to span two definitions -- this pays that debt), and #348 bounds
# `expr_mse_unbiased_capped`'s prediction-side correction by the submission's own
# across-perturbation centred sum of squares. Table byte-identical for the fifth mint running:
# PROVED, not asserted, by the byte-comparison test above, which now ties its literal to BOTH
# predecessors' shipped digests. So the only thing moving this digest is the rename.
FROZEN_DIGEST = "c49271e6e2840acf0431fc4288773c1443526f6c85a138a83f8804dd1ab544f1"

def test_registry_digest_is_frozen():
    from cell_eval2.scales import scales_digest

    assert scales_digest() == FROZEN_DIGEST, (
        "the scale registry changed. A shipped scale is immutable -- mint a new _v<n> "
        "instead of editing this one, and only update FROZEN_DIGEST for a scale that has "
        "never been published."
    )


# Every value differs from what `expr_mse_unbiased_capped_norm` already carries -- a no-op mutation
# would leave the payload identical and fail for the wrong reason. `allow_negative_baseline`
# is absent on purpose: it requires `anchor=None`, which no shipped entry has, so it cannot be
# varied legally here and is covered structurally by the next test instead.
@pytest.mark.parametrize("field,new", [
    ("base", 0.25), ("anchor", -0.75), ("clamp_low", -3.0), ("clamp_high", 2.0),
    ("penalty", "boxcox"), ("direction", "higher"), ("scored", False),
    ("penalty_exponent", 3.0), ("penalty_cap", 5.0),
    # Legal on this entry precisely because it is the LOWER/anchor-0 one: `metric_min` must
    # sit on the worse side of the anchor, i.e. >= 0.0 here.
    ("metric_min", 5.0),
])
def test_digest_moves_when_any_scoring_field_moves(field, new):
    """`scale_payload` must SERIALIZE every field that can change a score.

    ⚠️ An earlier version mutated the already-serialized dict and re-hashed it. That could
    not fail: hashing any changed dict changes its digest, so it held even if `scale_payload`
    hard-coded or omitted the field entirely. It has to start from a modified `Scale` OBJECT
    and serialize that (codex checkpoint-2 P3).

    Mutations are applied to `expr_mse_unbiased_capped_norm` (lower / anchor 0.0), the one shipped
    entry on which every field below is a LEGAL change -- `dataclasses.replace` re-runs
    `Scoring.__post_init__`, so a boxcox penalty on a higher-is-better member would be
    rejected before the serializer was ever reached. `allow_negative_baseline` cannot vary
    legally on any shipped entry (it requires `anchor=None`); it is covered by
    `test_scale_payload_serializes_every_scoring_field` instead.
    """
    import dataclasses

    from cell_eval2.scales import SCALES, ScaleEntry, scale_payload

    original = SCALES[NAME]
    metric = "expr_mse_unbiased_capped_norm"
    entry = original.entries[metric]
    mutated_entry = (
        ScaleEntry(base=new, scoring=entry.scoring) if field == "base"
        else ScaleEntry(base=entry.base,
                        scoring=dataclasses.replace(entry.scoring, **{field: new}))
    )
    mutated = dataclasses.replace(
        original, entries={**dict(original.entries), metric: mutated_entry})

    assert scale_payload(mutated) != scale_payload(original), (
        f"scale_payload does not serialize {field!r}, so scales_digest() cannot freeze it"
    )


def test_scale_payload_serializes_every_scoring_field():
    """Structural counterpart to the test above, and the one that survives a new field.

    The behavioural test can only mutate fields that are LEGAL on a shipped entry. This one
    needs no legal mutation: it asserts the payload's key set is exactly ``base`` plus every
    field of ``Scoring``. Add a field to ``Scoring`` and forget it in ``scale_payload`` and
    this fails immediately -- which is the failure mode that would let a frozen scale's
    numbers move while its digest stood still.
    """
    import dataclasses

    from cell_eval2.scales import SCALES, scale_payload
    from cell_eval2.scoring import Scoring

    payload = scale_payload(SCALES[NAME])
    expected = {"base"} | {f.name for f in dataclasses.fields(Scoring)}
    for metric, serialized in payload["entries"].items():
        assert set(serialized) == expected, metric


def test_digest_ignores_the_description():
    """Prose must be improvable without minting a _v2."""
    from cell_eval2.scales import SCALES, scale_payload

    assert "description" not in scale_payload(SCALES[NAME])


# Captured from the registry as it stood BEFORE `Scoring` gained `metric_min`, i.e. under
# FROZEN_DIGEST 964cd53a..., as EXACT bit patterns (`float.hex()`) over inputs chosen to be
# NON-DYADIC. An earlier revision of this replay used {0, .25, .5, .75, 1} with
# `pytest.approx`, which cannot detect the drift it exists to detect: every one of those
# lands on a dyadic rational that survives any reasonable rearrangement, and `approx` would
# forgive a 1-ulp move anyway (codex round 1).
_PRE_BUMP_INPUTS = (0.0, 1e-7, 0.1, 1 / 3, 0.4999999999999999, 0.5000000000000001,
                    0.6180339887498949, 0.7071067811865476, 0.9999999999999999, 1.0,
                    1.0000000000000002, 3.7)
_PRE_BUMP_HEX = {
    "de_wilcoxon_direction_fidelity_yield_raw": [
        '-0x1.0000000000000p+0', '-0x1.fffff94a03595p-1', '-0x1.999999999999ap-1',
        '-0x1.5555555555556p-2', '-0x1.0000000000000p-52', '0x1.0000000000000p-52',
        '0x1.e3779b97f4a80p-3', '0x1.a827999fcef34p-2', '0x1.ffffffffffffep-1',
        '0x1.0000000000000p+0', '0x1.0000000000000p+0', '0x1.0000000000000p+0',
    ],
    "de_wilcoxon_direction_reach_raw": [
        '0x0.0p+0', '0x1.ad7f29abcaf48p-24', '0x1.999999999999ap-4',
        '0x1.5555555555555p-2', '0x1.ffffffffffffep-2', '0x1.0000000000001p-1',
        '0x1.3c6ef372fe950p-1', '0x1.6a09e667f3bcdp-1', '0x1.fffffffffffffp-1',
        '0x1.0000000000000p+0', '0x1.0000000000000p+0', '0x1.0000000000000p+0',
    ],
    "de_wilcoxon_lfc_nmae": [
        '0x1.0000000000000p+0', '0x1.fffffca501acbp-1', '0x1.ccccccccccccdp-1',
        '0x1.5555555555556p-1', '0x1.0000000000001p-1', '0x1.ffffffffffffep-2',
        '0x1.8722191a02d60p-2', '0x1.2bec333018866p-2', '0x1.0000000000000p-53',
        '0x0.0p+0', '-0x1.0000000000000p-52', '-0x1.0000000000000p+0',
    ],
    "de_wilcoxon_sig_jaccard": [
        '0x0.0p+0', '0x1.ad7f29abcaf48p-24', '0x1.999999999999ap-4',
        '0x1.5555555555555p-2', '0x1.ffffffffffffep-2', '0x1.0000000000001p-1',
        '0x1.3c6ef372fe950p-1', '0x1.6a09e667f3bcdp-1', '0x1.fffffffffffffp-1',
        '0x1.0000000000000p+0', '0x1.0000000000000p+0', '0x1.0000000000000p+0',
    ],
    "expr_mse_unbiased_capped_norm": [
        '0x1.0000000000000p+0', '0x1.fffffca501acbp-1', '0x1.ccccccccccccdp-1',
        '0x1.5555555555556p-1', '0x1.0000000000001p-1', '0x1.ffffffffffffep-2',
        '0x1.8722191a02d60p-2', '0x1.2bec333018866p-2', '0x1.0000000000000p-53',
        '0x0.0p+0', '-0x1.0000000000000p-52', '-0x1.599999999999ap+1',
    ],
    "pds_cosine": [
        '-0x1.0000000000000p+0', '-0x1.fffff94a03595p-1', '-0x1.999999999999ap-1',
        '-0x1.5555555555556p-2', '-0x1.0000000000000p-52', '0x1.0000000000000p-52',
        '0x1.e3779b97f4a80p-3', '0x1.a827999fcef34p-2', '0x1.ffffffffffffep-1',
        '0x1.0000000000000p+0', '0x1.0000000000000p+0', '0x1.0000000000000p+0',
    ],
}


def test_the_metric_min_bump_moved_no_shipped_score():
    """`FROZEN_DIGEST` was bumped for a SCHEMA change; this is the evidence it was not a
    number change. Without it the bump is indistinguishable from a silent re-tune, which is
    the one thing the freeze exists to prevent -- so the replay has to live in the repo, not
    in a commit message.

    BIT-EXACT (`float.hex()`), not `approx`: the claim being made is that nothing moved at
    all, and a tolerance would not support it.
    """
    from cell_eval2.scales import SCALES
    from cell_eval2.scoring import score_one

    scale = SCALES[NAME]
    assert set(scale.entries) == set(_PRE_BUMP_HEX), f"the {NAME} membership moved too"
    for metric, entry in scale.entries.items():
        assert entry.scoring.metric_min is None, (
            f"{metric} declares metric_min; a frozen scale that is unfloored needs its own "
            "_v<n>, not a bump"
        )
        got = [score_one(u, entry.base, entry.scoring).hex() for u in _PRE_BUMP_INPUTS]
        assert got == _PRE_BUMP_HEX[metric], metric


def test_no_shipped_scale_entry_is_unfloored():
    """A frozen scale's floor is part of its published definition. `clamp_low=None` would
    make it `-inf` (or `-penalty_cap`), i.e. a floor that follows a module constant -- the
    same hazard `build_scale` already refuses for `penalty_exponent`/`penalty_cap`."""
    from cell_eval2.scales import SCALES

    for scale in SCALES.values():
        for metric, entry in scale.entries.items():
            assert entry.scoring.clamp_low is not None, f"{scale.name}/{metric} is unfloored"
