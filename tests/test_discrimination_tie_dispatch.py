"""End-to-end tie behaviour through `compute_metrics`, i.e. through the real dispatch.

Issue #282. The other #282 tests call `discrimination_score` / `_discrimination_ranks_xp`
directly and pass `tie_policy` themselves, so **deleting the dispatch wiring in `run.py`
would leave them all green** while every real run silently fell back to the low-level
default. Codex flagged that on review; these tests close it by never naming the knob --
they name a CONFIG and assert the number.
"""
import anndata as ad
import numpy as np
import pandas as pd
import pytest

from cell_eval2 import compute_metrics
from cell_eval2.config import EvalConfig

NCELL = 6
GENES = np.array([f"gene{i}" for i in range(5)])
# Six targets whose ALPHABETICAL order is unmistakable. The legacy policy scores a tied
# target by its index here; the corrected one scores every one of them 0.5.
NAMES = ["aa", "bb", "cc", "dd", "ee", "ff"]


def _pair():
    """A reference with real effects, and a prediction that pastes the reference CONTROL
    for every target -- so every predicted effect is bit-exactly zero and, under cosine,
    every distance is exactly 1.0."""
    rng = np.random.default_rng(0)
    base = rng.random(len(GENES)) + 5.0
    ctrl_cells = np.tile(base, (NCELL, 1))

    real_rows, pred_rows, labels = [ctrl_cells], [ctrl_cells.copy()], ["non-targeting"] * NCELL
    for i, n in enumerate(NAMES):
        eff = np.zeros(len(GENES))
        eff[i % len(GENES)] = i + 1.0
        real_rows.append(np.tile(base + eff, (NCELL, 1)))
        pred_rows.append(ctrl_cells.copy())          # the paste
        labels += [n] * NCELL

    def build(rows):
        a = ad.AnnData(X=np.vstack(rows).astype(np.float32),
                       obs=pd.DataFrame({"target": labels}),
                       var=pd.DataFrame(index=GENES))
        a.obs_names = [str(i) for i in range(a.n_obs)]
        return a

    return build(pred_rows), build(real_rows)


def _pds(cfg):
    pred, real = _pair()
    df = compute_metrics(pred, real, config=cfg)
    col = "pds_cosine" if "pds_cosine" in set(df["metric"]) else "discrimination_score_cosine"
    sub = df.filter(df["metric"] == col)
    return dict(zip(sub["perturbation"].to_list(), sub["value"].to_list()))


def _cfg(preset, **over):
    """The preset, restricted to pds_cosine. `exclude_target_gene` is turned OFF
    DELIBERATELY: these target labels are not gene names, and since #248 a zero-resolve
    run raises rather than silently excluding nothing. This test is about the tie policy,
    not exclusion -- and crucially it leaves `tie_policy` at whatever the PRESET says, so
    the dispatch is what carries it."""
    from dataclasses import replace

    base = EvalConfig.from_preset(preset)
    return replace(base, metrics=["pds_cosine"], pert_col="target",
                   control="non-targeting", input_type="lognorm",
                   discrimination=replace(base.discrimination, exclude_target_gene=False),
                   **over)


def test_v2_preset_scores_every_pasted_target_at_the_no_information_point():
    """The headline: a submission pasting the control gets 0.5 on every target, not a
    ladder from 1.0 to 0.0 keyed on the target's name."""
    got = _pds(_cfg("v2"))
    assert got == pytest.approx({n: 0.5 for n in NAMES}), (
        "a pasted-control submission must score the no-information point on every target"
    )


def test_v2_preset_result_is_invariant_to_renaming_the_targets():
    """Stronger and independent of the 0.5 constant: relabel the targets so their
    alphabetical ORDER reverses, and no target's score may move."""
    got = _pds(_cfg("v2"))
    rename = {n: chr(ord("z") - i) * 2 for i, n in enumerate(NAMES)}   # aa->zz, bb->yy, ...

    pred, real = _pair()
    for a in (pred, real):
        a.obs["target"] = [rename.get(t, t) for t in a.obs["target"]]
    from dataclasses import replace
    df = compute_metrics(pred, real, config=replace(_cfg("v2")))
    sub = df.filter(df["metric"] == "pds_cosine")
    renamed = dict(zip(sub["perturbation"].to_list(), sub["value"].to_list()))

    for original, new in rename.items():
        assert renamed[new] == pytest.approx(got[original]), (
            f"{original} scored {got[original]} but reads {renamed[new]} once renamed to "
            f"{new} -- the score still depends on the target's name"
        )


def test_v1_preset_still_reaches_the_legacy_policy_through_the_dispatch():
    """The wiring carries the v1 convention too. If `run.py` stopped passing tie_policy,
    this would silently become the v2 answer (0.5 everywhere) and fail."""
    got = _pds(_cfg("v1"))
    # v1: rank_denominator="n", so D = 6 and the ladder is 1 - index/6.
    assert got == pytest.approx({n: 1.0 - i / 6 for i, n in enumerate(NAMES)}), (
        "the v1 preset must still reproduce upstream cell-eval's argsort tie ordering"
    )
