import numpy as np
from cell_eval2.metrics.discrimination import discrimination_score

def _bulk(labels, means):
    return (np.asarray(labels, dtype=object), np.asarray(means, dtype=np.float64))

def _toy():
    # 4 non-control perts + control; distinct effect vectors so ranks are unambiguous.
    genes = np.array(["gA", "gB", "gC"], dtype=str)
    labels = ["ctrl", "A", "B", "C", "D"]
    real = np.array([[0, 0, 0], [3, 0, 0], [0, 3, 0], [0, 0, 3], [3, 3, 0]], dtype=float)
    pred = real.copy()  # perfect prediction -> every score should be 1.0
    return genes, labels, real, pred

def test_subset_matches_whole_for_predicted_perts():
    genes, labels, real, pred = _toy()
    whole = discrimination_score(
        pred_bulk=_bulk(labels, pred), real_bulk=_bulk(labels, real),
        control="ctrl", distance="l2", rank_denominator="n-1",
        exclude_target_gene=False, control_source="real", genes=genes,
    )
    # Score only perts {A, C} as a "piece": pred holds ctrl + A + C only; real is full.
    keep = [0, 1, 3]  # ctrl, A, C
    piece = discrimination_score(
        pred_bulk=_bulk([labels[i] for i in keep], pred[keep]),
        real_bulk=_bulk(labels, real),
        control="ctrl", distance="l2", rank_denominator="n-1",
        exclude_target_gene=False, control_source="real", genes=genes,
    )
    assert set(piece) == {"A", "C"}
    assert piece["A"] == whole["A"]
    assert piece["C"] == whole["C"]

def test_whole_prediction_behavior_unchanged():
    genes, labels, real, pred = _toy()
    whole = discrimination_score(
        pred_bulk=_bulk(labels, pred), real_bulk=_bulk(labels, real),
        control="ctrl", distance="l2", rank_denominator="n-1",
        exclude_target_gene=False, control_source="real", genes=genes,
    )
    assert whole == {"A": 1.0, "B": 1.0, "C": 1.0, "D": 1.0}  # perfect prediction
