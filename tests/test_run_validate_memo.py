import anndata as ad
import numpy as np

from cell_eval2 import norm, run


def _a(X):
    return ad.AnnData(X=np.asarray(X, dtype=np.float64))


def test_validate_runs_once_per_matrix_and_type(monkeypatch):
    calls = []
    real = norm.validate_input_type
    def spy(adata, input_type, **kw):
        calls.append((id(adata), input_type))
        return real(adata, input_type, **kw)
    monkeypatch.setattr(norm, "validate_input_type", spy)

    # all-zero -> passes "counts" (all-integer) AND "lognorm" (max==0, so the all-integer
    # mislabel guard does not fire), letting us exercise the per-type memo without a raise.
    a = _a([[0.0, 0.0], [0.0, 0.0]])
    run._validate_input_once(a, "counts", allow_fractional=False)
    run._validate_input_once(a, "counts", allow_fractional=False)
    assert len(calls) == 1                      # second call deduped

    run._validate_input_once(a, "lognorm", allow_fractional=False)
    assert len(calls) == 2                      # different effective type -> re-validated

    b = _a([[1.0, 2.0]])
    run._validate_input_once(b, "counts", allow_fractional=False)
    assert len(calls) == 3                      # different object -> validated


def test_check_scale_limit_runs_once_per_matrix(monkeypatch):
    calls = []
    real = norm.check_scale_limit
    def spy(adata, input_type, mx, *, precomputed_row_total_max=None):
        calls.append((id(adata), input_type, mx))
        return real(adata, input_type, mx, precomputed_row_total_max=precomputed_row_total_max)
    monkeypatch.setattr(norm, "check_scale_limit", spy)

    a = _a([[1.0, 2.0], [3.0, 4.0]])
    run._check_scale_limit_once(a, "counts", 1e6)
    run._check_scale_limit_once(a, "counts", 1e6)
    assert len(calls) == 1                       # second call deduped
    run._check_scale_limit_once(a, "counts", 10.0)
    assert len(calls) == 2                       # different cap -> re-checked
