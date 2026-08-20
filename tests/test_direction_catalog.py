import pytest

from cell_eval2.catalog import CATALOG, PROFILES

SUFFIXES = [
    "direction_fidelity", "direction_fidelity_raw", "direction_coverage",
    "direction_yield", "direction_yield_raw", "direction_fidelity_yield",
    "direction_fidelity_yield_raw", "direction_reach", "direction_reach_raw",
    "direction_reach_unbounded", "direction_reach_unbounded_raw",
]
# All eleven are scored. NINE of them are newly enrolled -- direction_fidelity_yield and
# direction_reach were already scored, being the two the old token could spell. The other
# nine were diagnostic-only because best_value could not say "higher is better but not
# enrolled", so enrolment and boundedness had to share one token. With direction and anchor
# recorded separately enrolment is its own decision, and every metric in this family has a
# direction, so every one of them is scored.
SCORED = set(SUFFIXES)

#: The two members of this family the 2026 competition profile scores. The other nine stay
#: full/de-only, so `vcc2026` picks up exactly two of the eleven. Since #231 these are the
#: RAW (chance-UNcorrected) pair: the corrected `direction_fidelity_yield`'s no-skill point
#: is negative and panel-sensitive, while the raw one sits at ~0.50 on every line and panel
#: measured. The corrected pair stays in `full`/`de`.
#:
#: Profile membership is wilcoxon-only (`_register_de_family` gives the deseq2 family
#: `profiles=()`), so this tuple governs only the wilcoxon spellings. It no longer says
#: anything about `agg`: every entry in the catalog aggregates by mean (#231), which is
#: asserted unconditionally below and pinned catalog-wide in
#: `test_scoring_catalog.py::test_the_catalog_has_exactly_one_aggregation_statistic`.
IN_VCC2026 = ("direction_fidelity_yield_raw", "direction_reach_raw")


@pytest.mark.parametrize("suffix", SUFFIXES)
@pytest.mark.parametrize("method", ["wilcoxon", "deseq2"])
def test_entry_exists_with_the_specified_wiring(method, suffix):
    spec = CATALOG[f"de_{method}_{suffix}"]
    assert spec.kind == "de"
    assert spec.normalization is None
    assert spec.v1_name is None
    # Unconditional since #231: all eleven aggregate by MEAN, in both families. `agg` is
    # backend-invariant by construction (`_register_de_family` passes one `agg` per sibling
    # pair), so this holds for the deseq2 spellings too even though their `profiles` is empty.
    assert spec.agg == "mean"
    assert spec.worst_value is None
    assert spec.v1_available is False
    assert spec.scoring.scored is (suffix in SCORED)
    # every member of this family is up-is-better; only enrolment differs, which is
    # exactly the fact best_value="none" used to erase along with the direction.
    assert spec.scoring.direction == "higher"


@pytest.mark.parametrize("suffix", SUFFIXES)
def test_profiles_are_full_and_de_on_wilcoxon_only(suffix):
    expected = ("full", "de", "vcc2026") if suffix in IN_VCC2026 else ("full", "de")
    assert CATALOG[f"de_wilcoxon_{suffix}"].profiles == expected
    assert CATALOG[f"de_deseq2_{suffix}"].profiles == ()


def test_exactly_two_of_the_family_are_in_vcc2026():
    # Pinned as a SET rather than a count: a count would still pass if one member were
    # swapped for another, and which two are scored is the whole content of the choice.
    assert {m for m in PROFILES["vcc2026"] if any(s in m for s in SUFFIXES)} == {
        f"de_wilcoxon_{s}" for s in IN_VCC2026
    }


def test_the_vcc_profile_is_untouched():
    assert not any(s in m for m in PROFILES["vcc"] for s in SUFFIXES)


def test_entries_are_appended_after_the_v0_5_0_direction_metrics():
    names = list(CATALOG)
    assert names.index("de_wilcoxon_direction_sensitivity_universe") < names.index(
        "de_wilcoxon_direction_fidelity"
    )


def test_the_v0_5_0_entries_are_unchanged():
    for suffix in ("direction_precision", "direction_sensitivity",
                   "direction_sensitivity_universe"):
        spec = CATALOG[f"de_wilcoxon_{suffix}"]
        assert spec.worst_value == 0.0
        assert spec.agg == "mean"
        # v2-native (v1_name=None), so NOT offered under version="v1". These three were
        # v1-available only because v1_available was hand-flagged and nobody set it; it is
        # derived from v1_name now, so a new v2 metric cannot leak into v1 by omission.
        assert spec.v1_name is None
        assert spec.v1_available is False
