"""The shared core behind `from_replicate` and every scale column.

Its raises are the membership contract: a competition average must never silently average
over fewer metrics than it names. The row-presence check is the one the scale path lacked --
a metric can sit in the AGGREGATE (so the missing-column check passes) and still have no
output ROW, because the baseline pass dropped it.
"""
import pytest

from cell_eval2.scales import ScaleEntry
from cell_eval2.score import _reference_column
from cell_eval2.scoring import Scoring

_UP = Scoring(scored=True, direction="higher", anchor=1.0, penalty="none", clamp_low=0.0)


def _entries():
    return {"pds_cosine": ScaleEntry(base=0.5, scoring=_UP),
            "de_wilcoxon_sig_jaccard": ScaleEntry(base=0.0, scoring=_UP)}


def test_scores_named_metrics_nulls_others_and_averages_its_own_membership():
    rows = ["pds_cosine", "de_wilcoxon_sig_jaccard", "expr_mae", "avg_score"]
    s = _reference_column(rows, (0.75, 0.5, 9.9),
                          ["pds_cosine", "de_wilcoxon_sig_jaccard", "expr_mae"],
                          _entries(), column="from_replicate", label="the anchor")
    assert s.name == "from_replicate"
    assert s.to_list()[0] == pytest.approx(0.5)      # (0.75 - 0.5) / (1 - 0.5)
    assert s.to_list()[1] == pytest.approx(0.5)      # (0.5  - 0.0) / (1 - 0.0)
    assert s.to_list()[2] is None                    # unnamed -> null, never averaged
    assert s.to_list()[3] == pytest.approx(0.5)      # avg over the TWO named metrics


def test_a_named_metric_absent_from_the_aggregate_raises():
    with pytest.raises(ValueError, match="absent from the aggregate"):
        _reference_column(["pds_cosine", "avg_score"], (0.75,), ["pds_cosine"],
                          _entries(), column="from_replicate", label="the anchor")


def test_a_named_metric_with_no_OUTPUT_ROW_raises():
    """The reachable narrowing: the metric IS in the aggregate, so the column check passes,
    but the baseline pass gave it no row -- e.g. an `overrides={m: Scoring(scored=False)}`, or
    a non-decisive metric excluded for a degenerate baseline. Averaging the survivors would
    silently report a five-member competition score under a six-member anchor."""
    with pytest.raises(ValueError, match="no output row"):
        _reference_column(["pds_cosine", "avg_score"], (0.75, 0.5),
                          ["pds_cosine", "de_wilcoxon_sig_jaccard"],
                          _entries(), column="from_replicate", label="the anchor")


def test_two_columns_spelling_one_metric_raise():
    entries = {"pds_cosine": ScaleEntry(base=0.5, scoring=_UP)}
    with pytest.raises(ValueError, match="two columns that both name"):
        _reference_column(["pds_cosine", "avg_score"], (0.75, 0.25),
                          ["pds_cosine", "discrimination_score_cosine"],
                          entries, column="from_replicate", label="the anchor")


def test_the_last_row_must_be_avg_score():
    """The mean is written to the LAST row by position; a caller that reordered the frame
    would silently overwrite a metric's cell with an average."""
    with pytest.raises(ValueError, match="avg_score"):
        _reference_column(["avg_score", "pds_cosine"], (0.75,), ["pds_cosine"],
                          {"pds_cosine": ScaleEntry(base=0.5, scoring=_UP)},
                          column="from_replicate", label="the anchor")
