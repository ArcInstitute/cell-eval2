"""``score_piece`` validates each pred piece before scoring it (``validate_input``,
``allow_fractional_counts``, ``max_counts_per_cell``), directly and through ``score_cellstream``.

CPU-only: DE is stubbed and the gpudge backend check is forced, so no GPU or cellstream is needed.
"""
import re

import anndata as ad
import numpy as np
import pandas as pd
import polars as pl
import pytest

from cell_eval2.config import EvalConfig

_NORM_FRACTIONAL = "declared input_type='counts' but values are fractional"


@pytest.fixture
def stub_gpudge(monkeypatch):
    """These tests consume no DE; avoid an unrelated GPU dependency."""
    from cell_eval2 import partition_inmem

    monkeypatch.setattr(
        partition_inmem, "compute_de",
        lambda *args, **kwargs: pl.DataFrame({"target": []}, schema={"target": pl.Utf8}),
    )
    monkeypatch.setattr(partition_inmem, "_resolve_backend", lambda _name: "gpudge")


def _whole(text):
    return rf"^{re.escape(text)}$"


def _cfg(**overrides):
    return EvalConfig(metrics=["pds_cosine"], **overrides)


def _score(tmp_path, real, piece, cfg):
    """Build the reference from ``real`` under ``cfg`` and score ``piece`` under the same cfg."""
    from cell_eval2.partition_inmem import build_reference, score_piece

    cache = tmp_path / f"ref_{len(list(tmp_path.iterdir()))}"
    build_reference(real, config=cfg, cache_dir=str(cache), control_format="h5ad",
                    comparator="bulk_lognorm")
    return score_piece(piece, str(cache), config=cfg, comparator="bulk_lognorm")


def _piece(pred, transform=None):
    """A fresh perturbed-only piece (the validation memo lives on the AnnData object)."""
    piece = pred[pred.obs["target"] != "non-targeting"].copy()
    if transform is not None:
        piece.X = transform(np.asarray(piece.X))
    return piece


def test_a_log1p_piece_declared_counts_is_refused_with_the_pred_side_fix(
        tmp_path, synthetic_counts_pair, stub_gpudge):
    from cell_eval2.partition_inmem import _PIECE_FRACTIONAL_FIX

    pred, real = synthetic_counts_pair
    with pytest.raises(ValueError) as info:
        _score(tmp_path, real, _piece(pred, np.log1p), _cfg())
    assert type(info.value) is ValueError
    assert re.match(_whole(f"{_NORM_FRACTIONAL} (pred side). {_PIECE_FRACTIONAL_FIX}"),
                    str(info.value))
    assert "never for log-normalized data" in str(info.value)


@pytest.mark.parametrize("overrides", [{"allow_fractional_counts": True},
                                       {"validate_input": False}])
def test_a_log1p_piece_scores_when_the_check_is_relaxed_or_off(
        tmp_path, synthetic_counts_pair, stub_gpudge, overrides):
    pred, real = synthetic_counts_pair
    out = _score(tmp_path, real, _piece(pred, np.log1p), _cfg(**overrides))
    assert out.height > 0
    assert set(out["metric"].to_list()) == {"pds_cosine"}


def test_a_counts_piece_scores_as_with_validation_off(tmp_path, synthetic_counts_pair,
                                                      stub_gpudge):
    pred, real = synthetic_counts_pair
    on = _score(tmp_path, real, _piece(pred), _cfg())
    off = _score(tmp_path, real, _piece(pred), _cfg(validate_input=False))
    assert on.height > 0
    assert on.sort(["perturbation", "metric"]).equals(off.sort(["perturbation", "metric"]))


def test_a_piece_above_max_counts_per_cell_is_refused(tmp_path, synthetic_counts_pair,
                                                      stub_gpudge):
    from cell_eval2.norm import ScaleLimitError

    pred, real = synthetic_counts_pair
    with pytest.raises(ScaleLimitError):
        _score(tmp_path, real, _piece(pred, lambda x: x * 1000.0),
               _cfg(max_counts_per_cell=1e4))


# ── score_cellstream end to end, on fake cell stores ────────────────────────────────────────

_PERTS = ("control", "A", "B")


class _FakeCellStore:
    """A reference-first cell-store stand-in holding ``X`` (rows grouped by ``_PERTS``)."""

    def __init__(self, X, *, id_prefix):
        n_per = X.shape[0] // len(_PERTS)
        self.n_obs, self.n_vars = X.shape
        self.manifest = {"group_by": "perturbation", "value_dtype_on_disk": "float32"}
        self._X = X.astype(np.float32)
        self.obs = pd.DataFrame(
            {"perturbation": np.repeat(np.asarray(_PERTS, dtype=object), n_per),
             "control_value": "control"},
            index=[f"{id_prefix}{i}" for i in range(self.n_obs)])
        self.var = pd.DataFrame(index=["A", "B"] + [f"g{j}" for j in range(self.n_vars - 2)])

    def group_labels(self):
        return list(_PERTS)

    def gather_rows_adata(self, row_ids, n_threads=1):
        rows = np.atleast_1d(np.asarray(row_ids))
        return ad.AnnData(X=self._X[rows].copy(), obs=self.obs.iloc[rows].copy(),
                          var=self.var.copy())

    def close(self):
        pass


def _cellstream_pair(monkeypatch):
    """Patch ``open_cell_store`` to a counts real archive and a mean-baseline-like pred archive
    (fractional, non-negative, varying across cells)."""
    import cell_eval2.cellstream as cs
    from cell_eval2.h5ad_manifest import MemBudget

    rng = np.random.default_rng(0)
    real_x = rng.poisson(3.0, size=(60, 8)).astype(np.float32)
    pred_x = real_x * rng.uniform(0.5, 1.5, size=real_x.shape).astype(np.float32) + 0.25
    stores = {"real.shad": _FakeCellStore(real_x, id_prefix="r"),
              "pred.shad": _FakeCellStore(pred_x, id_prefix="p")}
    monkeypatch.setattr(cs, "open_cell_store", lambda p: stores[p])
    return MemBudget(host_bytes=10**12, gpu_bytes=10**12)


def _cellstream_cfg(**overrides):
    return EvalConfig(pert_col="perturbation", metrics=["pds_cosine"], **overrides)


def test_score_cellstream_scores_a_fractional_pred_with_allow_fractional_counts(
        monkeypatch, stub_gpudge):
    from cell_eval2.cellstream import score_cellstream

    budget = _cellstream_pair(monkeypatch)
    res = score_cellstream("pred.shad", "real.shad",
                           config=_cellstream_cfg(allow_fractional_counts=True),
                           mem_budget=budget)
    assert sorted(res.per_pert["perturbation"].unique().to_list()) == ["A", "B"]


_FRACTIONAL_HINT = (
    "If the prediction holds fractional expected counts (for example a mean baseline), set "
    "allow_fractional_counts=True; that setting is for fractional expected counts, never for "
    "log-normalized data."
)
_DECLARE = 'input_type is one setting for both sides: declare input_type="lognorm" if both sides are lognorm'
_PRED_HINTS = {
    # The hint suggests autodetect only where autodetect would not refuse again.
    "autodetect_ok": f"{_DECLARE}; otherwise set autodetect_input_type=True.",
    "allow_discrete": f"{_DECLARE}.",
    "de_real": f"{_DECLARE}.",
    "de_pred": f"{_DECLARE}; otherwise set autodetect_input_type=True.",
    "autodetect_on": None,
}
_PRED_CASES = {
    "autodetect_ok": dict(metrics=["pds_cosine"]),
    "allow_discrete": dict(metrics=["pds_cosine"], allow_discrete=True),
    "de_real": dict(metrics=["de_wilcoxon_overlap"]),
    "de_pred": dict(metrics=["de_wilcoxon_overlap"], control_source="pred"),
    "autodetect_on": dict(metrics=["pds_cosine"], autodetect_input_type=True,
                          allow_discrete=True),
}


@pytest.mark.parametrize("case, pred_control", [
    ("autodetect_ok", False), ("allow_discrete", False), ("de_real", False), ("de_pred", False),
    ("autodetect_on", False), ("autodetect_ok", True), ("allow_discrete", True),
    ("autodetect_on", True),
])
def test_score_cellstream_refuses_a_fractional_pred_with_the_pred_side_fix(
        case, pred_control, monkeypatch, stub_gpudge):
    """Each case's text. With ``pred_control`` (control_source="pred") the pred-control builder
    refuses first, with the same text as ``score_piece``."""
    import cell_eval2.partition_inmem as pinmem
    from cell_eval2.cellstream import score_cellstream

    kw = dict(_PRED_CASES[case])
    if pred_control:
        kw["control_source"] = "pred"
        monkeypatch.setattr(pinmem, "score_piece",
                            lambda *a, **k: pytest.fail("reached score_piece"))
    hint = _PRED_HINTS[case]
    tail = _FRACTIONAL_HINT if hint is None else f"{_FRACTIONAL_HINT} {hint}"
    budget = _cellstream_pair(monkeypatch)
    cfg = EvalConfig(pert_col="perturbation", **kw)
    want = f"{_NORM_FRACTIONAL} (pred side). {tail}"
    with pytest.raises(ValueError, match=_whole(want)):
        score_cellstream("pred.shad", "real.shad", config=cfg, mem_budget=budget)


def test_score_cellstream_logs_the_fractional_allowance_once_per_run(
        monkeypatch, stub_gpudge, caplog):
    """A run over several pred pieces logs "the allowance was load-bearing" once, not per piece."""
    import logging

    import cell_eval2.partition_inmem as pinmem
    from cell_eval2.cellstream import score_cellstream
    from cell_eval2.h5ad_manifest import MemBudget

    budget = _cellstream_pair(monkeypatch)
    # Room for the control block and one pert (96 B per cell), not two: one piece per pert.
    one_pert = MemBudget(host_bytes=96 * 45, gpu_bytes=budget.gpu_bytes)
    pieces = []
    real_score_piece = pinmem.score_piece
    monkeypatch.setattr(pinmem, "score_piece",
                        lambda *a, **k: pieces.append(k["piece_id"]) or real_score_piece(*a, **k))
    cfg = _cellstream_cfg(allow_fractional_counts=True)
    with caplog.at_level(logging.WARNING, logger="cell_eval2.run"):
        score_cellstream("pred.shad", "real.shad", config=cfg, mem_budget=one_pert)
    assert len(pieces) == 2
    assert sum("was LOAD-BEARING" in r.getMessage() for r in caplog.records) == 1

    caplog.clear()
    pieces.clear()
    with caplog.at_level(logging.WARNING, logger="cell_eval2.run"):
        for _ in range(2):   # a new run warns again
            score_cellstream("pred.shad", "real.shad", config=cfg, mem_budget=one_pert)
    assert sum("was LOAD-BEARING" in r.getMessage() for r in caplog.records) == 2
