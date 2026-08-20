import dataclasses

import anndata as ad
import numpy as np
import pandas as pd
import pytest

from cell_eval2 import EvalConfig, compute_metrics


def _pair():
    # fractional "counts" pred vs an integer-counts real; 2 perts + control, enough genes/cells
    # that v2's mae pipeline computes cleanly.
    rng = np.random.default_rng(0)
    labels = (["g1"] * 4) + (["g2"] * 4) + (["non-targeting"] * 4)
    obs = pd.DataFrame({"target_gene": labels})
    var = pd.DataFrame(index=[f"gene{i}" for i in range(8)])
    real_X = rng.integers(1, 20, size=(12, 8)).astype(np.float32)
    real = ad.AnnData(X=real_X, obs=obs.copy(), var=var.copy())
    pred = ad.AnnData(X=(real_X + 0.5), obs=obs.copy(), var=var.copy())  # fractional
    return pred, real


def _pair_real_fractional():
    # the inverse of _pair: integer-counts pred vs a FRACTIONAL real, to assert the REAL side
    # stays strictly validated even when allow_fractional_counts (pred-side) is set.
    rng = np.random.default_rng(0)
    labels = (["g1"] * 4) + (["g2"] * 4) + (["non-targeting"] * 4)
    obs = pd.DataFrame({"target_gene": labels})
    var = pd.DataFrame(index=[f"gene{i}" for i in range(8)])
    base = rng.integers(1, 20, size=(12, 8)).astype(np.float32)
    pred = ad.AnnData(X=base.copy(), obs=obs.copy(), var=var.copy())   # integer counts
    real = ad.AnnData(X=(base + 0.5), obs=obs.copy(), var=var.copy())  # fractional
    return pred, real


def _v2_cfg(**over):
    return dataclasses.replace(EvalConfig.v2(), metrics=["mae"], pert_col="target_gene", **over)


def test_v2_rejects_fractional_counts_by_default():
    pred, real = _pair()
    with pytest.raises(ValueError, match="fractional"):
        compute_metrics(pred, real, config=_v2_cfg())


def test_v2_allows_fractional_counts_when_flagged():
    pred, real = _pair()
    out = compute_metrics(pred, real, config=_v2_cfg(allow_fractional_counts=True))  # no raise
    assert out.height > 0


def test_v2_autodetect_scores_fractional_pred():
    # autodetect re-types a genuinely log-norm pred (non-integer per-cell totals, which
    # guess_is_lognorm keys on) as lognorm — cell-eval style — so v2 scores it instead of
    # rejecting it; 1e10 gate gives headroom for the expm1 of lognorm values.
    import scanpy as sc

    from cell_eval2.norm import guess_is_lognorm

    _, real = _pair()
    pred = real.copy()
    sc.pp.normalize_total(pred)
    sc.pp.log1p(pred)
    assert guess_is_lognorm(pred) is True  # precondition: autodetect must see lognorm
    out = compute_metrics(
        pred, real,
        config=_v2_cfg(autodetect_input_type=True, max_counts_per_cell=1e10),
    )  # no raise
    assert out.height > 0


def test_allow_fractional_is_pred_side_only():
    # Codex #1: allow_fractional_counts must NOT relax the REAL side; a fractional real still
    # rejects even with the flag on (real is always validated strictly).
    pred, real = _pair_real_fractional()
    with pytest.raises(ValueError, match="fractional"):
        compute_metrics(pred, real, config=_v2_cfg(allow_fractional_counts=True))


def test_autodetect_keeps_real_side_strict():
    # Gemini PR #35: autodetect re-types the PRED submission ONLY; the REAL side keeps the declared
    # type (counts in v2), so a non-counts (genuinely log-norm) real is still rejected even with
    # autodetect on. Uses a real whose per-cell totals are non-integer (guess_is_lognorm -> True),
    # which would be accepted as lognorm if autodetect were (wrongly) applied to the real side.
    import scanpy as sc

    from cell_eval2.norm import guess_is_lognorm

    _, real_counts = _pair()          # integer-counts real
    real = real_counts.copy()
    sc.pp.normalize_total(real)
    sc.pp.log1p(real)                 # genuine log-norm real (fractional per-cell totals)
    assert guess_is_lognorm(real) is True
    pred = real_counts.copy()         # integer-counts pred
    cfg = _v2_cfg(autodetect_input_type=True, max_counts_per_cell=1e10)
    with pytest.raises(ValueError, match="fractional"):
        compute_metrics(pred, real, config=cfg)


def test_pred_de_input_strict_validates_real_control():
    # Codex P1: _pred_de_input substitutes the real control into the pred-DE matrix; that real
    # slice must be validated strictly even when allow_fractional_counts (pred-side) is set, so
    # the narrow backed/cache/de_real path can't let a fractional real control through.
    from cell_eval2.run import _pred_de_input

    pred, real = _pair_real_fractional()  # pred integer, real fractional; v2 -> control_source="real"
    cfg = _v2_cfg(allow_fractional_counts=True)
    with pytest.raises(ValueError, match="fractional"):
        _pred_de_input(pred, real, cfg=cfg)


def _pair_integer():
    # integer-counts pred AND real; used to exercise the scale-limit guard in isolation
    rng = np.random.default_rng(0)
    labels = (["g1"] * 4) + (["g2"] * 4) + (["non-targeting"] * 4)
    obs = pd.DataFrame({"target_gene": labels})
    var = pd.DataFrame(index=[f"gene{i}" for i in range(8)])
    X = rng.integers(1, 20, size=(12, 8)).astype(np.float32)
    return (ad.AnnData(X=X, obs=obs.copy(), var=var.copy()),
            ad.AnnData(X=X.copy(), obs=obs.copy(), var=var.copy()))


def test_validate_input_false_skips_mislabel_guard():
    pred, real = _pair()  # fractional "counts" pred -> raises by default
    out = compute_metrics(pred, real, config=_v2_cfg(validate_input=False))
    assert out.height > 0


def test_validate_input_default_still_rejects_mislabel():
    pred, real = _pair()
    with pytest.raises(ValueError, match="fractional"):
        compute_metrics(pred, real, config=_v2_cfg())  # default validate_input=True


def test_validate_input_false_skips_scale_limit():
    pred, real = _pair_integer()  # integer counts; tiny cap -> per-cell totals exceed it
    with pytest.raises(ValueError, match="exceeds"):
        compute_metrics(pred, real, config=_v2_cfg(max_counts_per_cell=1.0))
    out = compute_metrics(pred, real, config=_v2_cfg(max_counts_per_cell=1.0, validate_input=False))
    assert out.height > 0
