"""A gated metric no perturbation of the reference can score leaves avg_score (issue #391, §10.4).

Spec: internal:docs/internal/superpowers/specs/2026-10-08-real-side-nan-not-scored-design.md.
"""
import dataclasses
import logging

import polars as pl
import pytest

from cell_eval2.baseline import _degenerate_metrics, _require_scoreable
from cell_eval2.catalog import CATALOG
from cell_eval2.real_gate import RealGate
from cell_eval2.score import _replicate_entries, score_metrics

_NAME = "de_wilcoxon_lfc_spearman_pos"      # decisive: it has a v1 name
_NAN = float("nan")


@pytest.fixture
def gated(monkeypatch):
    monkeypatch.setitem(CATALOG, _NAME, dataclasses.replace(
        CATALOG[_NAME], real_gate=RealGate("rank_corr", lfc_direction="pos"), worst_value=-1.0))


def _frames(user_val, base_val):
    user = pl.DataFrame({"statistic": ["mean"], "expr_mae": [0.5], _NAME: [user_val]})
    base = pl.DataFrame({"statistic": ["mean"], "expr_mae": [1.0], _NAME: [base_val]})
    return user, base


def test_nan_on_both_sides_is_excluded_with_a_warning(gated, caplog):
    with caplog.at_level(logging.WARNING, logger="cell_eval2.score"):
        out = score_metrics(*_frames(_NAN, _NAN))
    assert set(out["metric"]) == {"expr_mae", "avg_score"}
    assert out.filter(pl.col("metric") == "avg_score")["from_baseline"].item() == 0.5
    assert "no perturbation its reference-only gate admits" in caplog.text


@pytest.mark.parametrize("statistic", ["mean", "count"])
def test_a_one_sided_empty_cohort_is_refused_whatever_the_statistic(gated, statistic):
    """Empty on one side only cannot happen over one reference: refused as a pairing error, even
    when the compared row (`count`) is finite on both sides (Copilot, PR #401)."""
    rows = ["mean", "count"]
    user = pl.DataFrame({"statistic": rows, "expr_mae": [0.5, 3.0], _NAME: [0.4, 3.0]})
    base = pl.DataFrame({"statistic": rows, "expr_mae": [1.0, 3.0], _NAME: [_NAN, 3.0]})
    with pytest.raises(ValueError, match="one side only"):
        score_metrics(user, base, comparison_statistic=statistic)
    with pytest.raises(ValueError, match="one side only"):
        score_metrics(base, user, comparison_statistic=statistic)


def test_an_ungated_metric_nan_on_both_sides_still_raises():
    user = pl.DataFrame({"statistic": ["mean"], "pds_l1": [_NAN]})
    base = pl.DataFrame({"statistic": ["mean"], "pds_l1": [_NAN]})
    with pytest.raises(ValueError, match="degenerate baseline"):
        score_metrics(user, base)


def test_nothing_left_to_score_still_raises(gated):
    user = pl.DataFrame({"statistic": ["mean"], _NAME: [_NAN]})
    base = pl.DataFrame({"statistic": ["mean"], _NAME: [_NAN]})
    with pytest.raises(ValueError, match="nothing left to score"):
        score_metrics(user, base)


def test_the_baseline_gate_skips_a_gated_nan_but_not_an_ungated_one(gated):
    agg = pl.DataFrame({"statistic": ["mean"], _NAME: [_NAN], "pds_l1": [_NAN]})
    assert [d["metric"] for d in _degenerate_metrics(agg)] == ["pds_l1"]


def test_a_v1_spelling_keeps_the_old_refusal(gated):
    """The gate runs only under v2, which emits canonical names. A v1 artifact spells this
    metric by its v1 alias and keeps the existing refusal and report -- v1 is unchanged."""
    v1 = CATALOG[_NAME].v1_name
    user = pl.DataFrame({"statistic": ["mean"], v1: [_NAN]})
    base = pl.DataFrame({"statistic": ["mean"], v1: [_NAN]})
    with pytest.raises(ValueError, match="degenerate baseline"):
        score_metrics(user, base)
    assert [d["metric"] for d in _degenerate_metrics(base)] == [v1]


def test_the_baseline_gate_still_reports_a_finite_degenerate_value_of_a_gated_metric(gated):
    """The exemption is for a NON-FINITE aggregate only. A bounded metric's baseline AT its
    anchor (1.0) is degenerate for a different reason and must still be reported."""
    agg = pl.DataFrame({"statistic": ["mean"], _NAME: [1.0]})
    assert [d["metric"] for d in _degenerate_metrics(agg)] == [_NAME]


def test_an_undefined_std_beside_a_finite_mean_is_still_a_degenerate_baseline(gated):
    """One admitted perturbation: the mean is finite and the sample std undefined. That is NOT an
    empty cohort, so the exemption must not fire on the std row -- it raises as before (codex,
    round 1)."""
    user = pl.DataFrame({"statistic": ["mean", "std"], _NAME: [0.4, _NAN]})
    base = pl.DataFrame({"statistic": ["mean", "std"], _NAME: [0.1, _NAN]})
    with pytest.raises(ValueError, match="degenerate baseline"):
        score_metrics(user, base, comparison_statistic="std")
    assert [d["metric"] for d in _degenerate_metrics(base, statistic="std")] == [_NAME]


def test_an_empty_cohort_is_exempt_on_every_statistic(gated):
    agg = pl.DataFrame({"statistic": ["mean", "std"], _NAME: [_NAN, _NAN]})
    assert _degenerate_metrics(agg, statistic="std") == []


def test_a_baseline_with_nothing_left_to_score_is_refused_at_build(gated):
    """Only a gated metric, admitting nothing: the build must not accept an artifact every
    scoring attempt would then refuse (codex, round 1)."""
    agg = pl.DataFrame({"statistic": ["mean"], _NAME: [_NAN]})
    with pytest.raises(ValueError, match="no scoreable metric"):
        _require_scoreable(agg)
    _require_scoreable(agg.with_columns(pds_l1=pl.lit(0.5)))      # one usable metric suffices


def test_the_replicate_scale_skips_a_metric_the_baseline_pass_excluded(gated, caplog):
    """An anchor can name a gated metric whose baseline admitted nothing (e.g. split halves
    behave differently from the full reference). The baseline pass excluded it; the replicate
    pass must not then refuse the run over the same NaN (codex, round 1)."""
    frame = pl.DataFrame({"metric": [_NAME], "replicate": [0.8]})
    with pytest.raises(ValueError, match="no usable replicate scale"):
        _replicate_entries({_NAME: _NAN}, frame)
    with caplog.at_level(logging.WARNING, logger="cell_eval2.score"):
        assert _replicate_entries({_NAME: _NAN}, frame, exclude={_NAME}) == {}
    assert "left off the replicate scale too" in caplog.text


def test_a_missing_mean_row_is_not_an_empty_cohort(gated):
    """Frames carrying only a `std` row say nothing about whether the cohort was empty, so the
    exemption must not fire -- the same answer `_degenerate_metrics` gives (codex, checkpoint 1
    round 2)."""
    user = pl.DataFrame({"statistic": ["std"], "expr_mae": [0.5], _NAME: [_NAN]})
    base = pl.DataFrame({"statistic": ["std"], "expr_mae": [1.0], _NAME: [_NAN]})
    with pytest.raises(ValueError, match="degenerate baseline"):
        score_metrics(user, base, comparison_statistic="std")


def test_the_replicate_exclusion_holds_under_a_scoring_override(gated, monkeypatch):
    """With the gated metric overridden to scored=False the baseline pass never reaches the
    exemption -- the replicate scale must still skip it, not refuse over its NaN baseline
    (codex, checkpoint 1 round 2). Driven through `score_metrics`, with the anchor resolution
    stubbed so no real bundle is needed."""
    import cell_eval2.score as score_mod
    from cell_eval2.scoring import DIAG

    frame = pl.DataFrame({"metric": ["expr_mae", _NAME], "replicate": [0.2, 0.8]})
    monkeypatch.setattr(score_mod, "resolve_anchor",
                        lambda expect, supplied=None, cached=None: (frame, {}, "stub"))
    monkeypatch.setattr(score_mod, "anchor_digest", lambda f, m: "stub")
    user = pl.DataFrame({"statistic": ["mean"], "expr_mae": [0.5], _NAME: [_NAN]})
    base = pl.DataFrame({"statistic": ["mean"], "expr_mae": [1.0], _NAME: [_NAN]})
    out = score_metrics(user, base, anchor="stub", anchor_expect=object(),
                        overrides={_NAME: DIAG})
    assert "from_replicate" in out.columns


def test_the_statistic_column_need_not_come_first(gated):
    """`score_metrics` finds the statistic column by name, so the shared predicate must too: with
    it last, a healthy metric still scores and the empty cohort is still excluded (codex,
    checkpoint 1 round 3)."""
    user = pl.DataFrame({"expr_mae": [0.5], "statistic": ["mean"], _NAME: [_NAN]})
    base = pl.DataFrame({"expr_mae": [1.0], "statistic": ["mean"], _NAME: [_NAN]})
    out = score_metrics(user, base)
    assert set(out["metric"]) == {"expr_mae", "avg_score"}


@pytest.mark.parametrize("statistic", ["count", "null_count"])
def test_an_empty_cohort_leaves_avg_score_on_a_finite_statistic_too(gated, statistic):
    """`count` and `null_count` are finite over an all-NaN column, so the exemption must key on
    the empty cohort alone, not on the compared row (Copilot, PR #401)."""
    rows = ["mean", "count", "null_count"]
    # expr_mae's rows are only there to leave something scoreable; positive on every row, so its
    # own baseline is never the degenerate one.
    user = pl.DataFrame({"statistic": rows, "expr_mae": [0.5, 3.0, 1.0], _NAME: [_NAN, 3.0, 0.0]})
    base = pl.DataFrame({"statistic": rows, "expr_mae": [1.0, 3.0, 1.0], _NAME: [_NAN, 3.0, 0.0]})
    out = score_metrics(user, base, comparison_statistic=statistic)
    assert _NAME not in set(out["metric"])
    assert _NAME not in [d["metric"] for d in _degenerate_metrics(base, statistic=statistic)]
