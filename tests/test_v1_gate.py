import pytest

from cell_eval2.catalog import CATALOG, PROFILES, resolve_metrics


def _v2_only_names():
    return [n for n, s in CATALOG.items() if not s.v1_available]


def test_v2_default_is_unchanged():
    assert resolve_metrics("full") == resolve_metrics("full", version="v2")


def test_a_v1_profile_run_silently_filters_and_does_not_raise():
    """Spec 4: compat expands the profile to an EXPLICIT list before dispatch
    (compat/__init__.py:108), so a resolver that raised on explicit v2-only names would
    raise on an ordinary v1 profile run. Filter at profile expansion; raise only on
    caller-supplied names."""
    names, _ = resolve_metrics("full", version="v1")
    assert not set(names) & set(_v2_only_names())


def test_explicitly_requesting_a_v2_only_metric_under_v1_raises():
    v2_only = _v2_only_names()
    if not v2_only:
        pytest.skip("no v2-only metrics registered yet")
    with pytest.raises(ValueError, match="not available under version='v1'"):
        resolve_metrics([v2_only[0]], version="v1")


# What v1 actually emits, per profile, NAMED. Every entry has an upstream cell-eval
# counterpart, which is what "available under v1" means.
_V1_EMITS = {
    "anndata": ["delta_mae", "delta_mse", "delta_pearson", "expr_mae", "expr_mse",
                "pds_cosine", "pds_l1", "pds_l2"],
    "de": ["de_wilcoxon_direction_match", "de_wilcoxon_lfc_spearman",
           "de_wilcoxon_lfc_spearman_neg", "de_wilcoxon_lfc_spearman_pos",
           "de_wilcoxon_model_direction_match", "de_wilcoxon_nsig_counts_pred",
           "de_wilcoxon_nsig_counts_real", "de_wilcoxon_nsig_spearman",
           "de_wilcoxon_overlap", "de_wilcoxon_overlap_top100", "de_wilcoxon_overlap_top200",
           "de_wilcoxon_overlap_top50", "de_wilcoxon_overlap_top500", "de_wilcoxon_pr_auc",
           "de_wilcoxon_precision", "de_wilcoxon_precision_top100",
           "de_wilcoxon_precision_top200", "de_wilcoxon_precision_top50",
           "de_wilcoxon_precision_top500", "de_wilcoxon_roc_auc", "de_wilcoxon_sig_recall"],
    "minimal": ["de_wilcoxon_nsig_counts_pred", "de_wilcoxon_nsig_counts_real",
                "de_wilcoxon_overlap", "de_wilcoxon_precision", "delta_pearson",
                "expr_mae", "expr_mse", "pds_l1"],
    "pds": ["pds_l1"],
    "vcc": ["de_wilcoxon_overlap", "expr_mae", "pds_l1"],
    # ⚠️ ONE of six. Five of vcc2026's members are v2-native, and `resolve_metrics` filters a
    # not-v1-available name SILENTLY when it arrived via a PROFILE (only an explicitly named
    # one raises), so `--profile vcc2026 --version v1` scores `pds_cosine` alone and still
    # calls itself vcc2026. Pinned here so that collapse is a stated fact rather than a
    # surprise; v1 is the upstream-cell-eval compatibility surface and a 2026 profile has
    # nothing to be compatible WITH.
    "vcc2026": ["pds_cosine"],
}
_V1_EMITS["full"] = sorted(set(_V1_EMITS["de"]) | set(_V1_EMITS["anndata"]))


def test_the_v1_emitted_set_is_pinned_by_name():
    """A NAMED golden, not a derivation. The predecessor asserted that everything dropped
    under v1 was `not v1_available` -- but `resolve_metrics` drops exactly the
    `not v1_available` names, so the assertion restated the implementation and could not
    fail for any definition of the flag. It was billed as "the direct guard against marking
    availability with `v1_name is None`", and when availability WAS switched to derive from
    `v1_name` (removing 35 entries from v1) it passed unchanged.

    Pinning the names instead means any change to what v1 emits has to edit this list, which
    is the point: v1 output is a compatibility surface, so it moves deliberately or not at all.
    """
    for profile in sorted(PROFILES):
        assert sorted(resolve_metrics(profile, version="v1")[0]) == sorted(_V1_EMITS[profile]), \
            f"the v1-emitted set for profile {profile!r} changed"


def test_every_v1_emitted_metric_has_an_upstream_name():
    """The golden above is a list of names; this is the RULE it encodes. Both are kept: the
    rule alone would drift silently with the catalog, the list alone would not say why."""
    for profile, names in _V1_EMITS.items():
        for n in names:
            assert CATALOG[n].v1_name is not None, f"{n} has no upstream cell-eval name"
            assert CATALOG[n].v1_available, n


def test_a_v1_explicit_list_of_v1_metrics_is_unaffected():
    assert resolve_metrics(["expr_mae"], version="v1") == (["expr_mae"], [])


@pytest.mark.parametrize(
    "module_name",
    ["cell_eval2.run", "cell_eval2.compat", "cell_eval2.baseline",
     "cell_eval2.scale", "cell_eval2.partition_inmem", "cell_eval2.ceiling"],
)
def test_every_caller_passes_version_through(module_name):
    """Spec 8: 'each resolve_metrics caller sees the same gated list -- parametrize
    rather than testing run alone.'

    A gate applied at one call site leaves the other thirteen emitting the metric, and
    the failure is silent: the metric simply appears in that path's output. Reading the
    source is the cheap structural check; the behavioural one is
    test_the_eleven_are_absent_from_the_published_wide_csv_under_v1 in test_direction_e2e.
    """
    import importlib
    import inspect
    import re

    # None of these five modules DEFINES resolve_metrics (it lives in catalog), and every
    # call site is single-line, so a flat scan is sufficient.
    src = inspect.getsource(importlib.import_module(module_name))
    calls = re.findall(r"resolve_metrics\([^)]*\)", src)
    assert calls, f"{module_name} no longer calls resolve_metrics -- update this test"
    missing = [c for c in calls if "version=" not in c]
    assert not missing, f"{module_name} calls resolve_metrics without version=: {missing}"
