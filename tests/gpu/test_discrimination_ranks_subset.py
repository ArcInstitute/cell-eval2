import numpy as np

from cell_eval2.gpu.distances import _discrimination_ranks_xp


def _bulk(labels, means):
    return (np.asarray(labels, dtype=object), np.asarray(means, dtype=np.float64))


def test_xp_subset_matches_whole():
    genes = np.array(["gA", "gB", "gC"], dtype=str)
    labels = ["ctrl", "A", "B", "C", "D"]
    real = np.array(
        [[0, 0, 0], [3, 0, 0], [0, 3, 0], [0, 0, 3], [3, 3, 0]], dtype=float
    )
    common = dict(
        genes=genes,
        metric="l2",
        exclude_target_gene=False, exclusion_scope="panel",
        rank_denominator="n-1",
        tie_policy="midrank",
        pert_chunk=2,
        control="ctrl",
        control_source="real",
    )
    whole = _discrimination_ranks_xp(np, _bulk(labels, real), _bulk(labels, real), **common)
    keep = [0, 1, 3]
    piece = _discrimination_ranks_xp(
        np, _bulk(labels, real), _bulk([labels[i] for i in keep], real[keep]), **common
    )
    assert set(piece) == {"A", "C"}
    assert piece["A"] == whole["A"] and piece["C"] == whole["C"]
