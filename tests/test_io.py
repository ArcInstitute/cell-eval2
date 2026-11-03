import numpy as np
import pandas as pd
import pytest

from _helpers import MISSING_LABEL_KINDS, assert_missing_labels_error, make_labels, set_labels
from cell_eval2.io import load_anndata, validate_pair


def test_load_passthrough(synthetic_pair):
    pred, _ = synthetic_pair
    assert load_anndata(pred) is pred


def test_validate_pair_ok(synthetic_pair):
    pred, real = synthetic_pair
    validate_pair(pred, real, pert_col="target", control="non-targeting")


def test_validate_pair_gene_mismatch(synthetic_pair):
    pred, real = synthetic_pair
    pred2 = pred[:, ::-1].copy()  # reversed gene order
    with pytest.raises(ValueError, match="gene"):
        validate_pair(pred2, real, pert_col="target", control="non-targeting")


def test_validate_pair_missing_control(synthetic_pair):
    pred, real = synthetic_pair
    with pytest.raises(ValueError, match="control"):
        validate_pair(pred, real, pert_col="target", control="absent-ctrl")


def test_load_anndata_backed_from_path(tmp_path, synthetic_pair):
    pred, _ = synthetic_pair
    p = tmp_path / "pred.h5ad"
    pred.write_h5ad(p)
    backed = load_anndata(str(p), backed=True)
    assert backed.isbacked
    assert backed.n_obs == pred.n_obs
    assert list(backed.var.index) == list(pred.var.index)  # metadata available


def test_load_anndata_inmemory_passthrough(synthetic_pair):
    pred, _ = synthetic_pair
    assert load_anndata(pred, backed=True) is pred  # objects pass through unchanged


def test_validate_pair_pert_mismatch_message_is_truncated():
    import numpy as np
    import anndata as ad
    import pandas as pd
    import pytest
    from cell_eval2.io import validate_pair

    genes = ["g0", "g1"]

    def _ad(perts):
        n = len(perts)
        return ad.AnnData(
            X=np.zeros((n, 2), dtype=np.float32),
            obs=pd.DataFrame({"perturbation": perts}, index=[f"c{i}" for i in range(n)]),
            var=pd.DataFrame(index=genes),
        )

    pred = _ad(["ctrl", "A", "B", "EXTRA_PRED"])
    real = _ad(["ctrl", "A", "B", "ONLY_REAL"])
    with pytest.raises(ValueError) as exc:
        validate_pair(pred, real, pert_col="perturbation", control="ctrl")
    msg = str(exc.value)
    assert "perturbation sets differ" in msg
    assert "EXTRA_PRED" in msg and "ONLY_REAL" in msg     # symmetric diff reported
    assert "pred-only" in msg and "real-only" in msg      # diff format, not raw arrays


@pytest.mark.parametrize("kind", MISSING_LABEL_KINDS)
def test_require_pert_labels_rejects_each_missing_kind(kind):
    from cell_eval2.io import require_pert_labels

    labels = make_labels(kind, ["a", "b", "c", "d", "e"], [1, 3])
    with pytest.raises(ValueError) as exc:
        require_pert_labels(labels, "target", side="real")
    msg = assert_missing_labels_error(exc, n_missing=2, n_cells=5, column="target", side="real")
    assert "null" in msg


@pytest.mark.parametrize("wrap", [lambda s: s, lambda s: s.to_numpy(), lambda s: pd.Index(s)])
def test_require_pert_labels_accepts_series_array_and_index(wrap):
    from cell_eval2.io import require_pert_labels

    labels = make_labels("object_none", ["a", "b", "c"], [0])
    with pytest.raises(ValueError, match="1 of 3 cells"):
        require_pert_labels(wrap(labels), "target", side="pred")


@pytest.mark.parametrize("kind", ["object", "string", "str", "categorical"])
def test_require_pert_labels_accepts_text_that_looks_missing(kind):
    from cell_eval2.io import require_pert_labels

    values = ["", "nan", "None", "<NA>", "a"]
    if kind == "str":    # make_labels skips where pd.StringDtype(na_value=...) is unavailable
        labels = make_labels("str", values, [])
    else:
        dtype = {"object": object, "string": "string", "categorical": "category"}[kind]
        labels = pd.Series(values, dtype=dtype)
    require_pert_labels(labels, "target", side="real")


def test_require_pert_labels_names_the_side_given():
    from cell_eval2.io import require_pert_labels

    with pytest.raises(ValueError, match="cells in /data/x.h5ad have"):
        require_pert_labels(np.array([None, "a"], dtype=object), "perturbation",
                            side="/data/x.h5ad")


@pytest.mark.parametrize("kind", MISSING_LABEL_KINDS)
@pytest.mark.parametrize("bad, other", [("pred", "real"), ("real", "pred")])
def test_validate_pair_rejects_missing_labels_on_either_side(synthetic_pair, kind, bad, other):
    pred, real = synthetic_pair
    n = pred.n_obs
    set_labels({"pred": pred, "real": real}[bad], "target", kind, [3, n - 2])
    with pytest.raises(ValueError) as exc:
        validate_pair(pred, real, pert_col="target", control="non-targeting")
    assert_missing_labels_error(exc, n_missing=2, n_cells=n, column="target", side=bad,
                                other_side=other)


def test_validate_pair_column_absent_error_comes_before_missing_labels(synthetic_pair):
    pred, real = synthetic_pair
    set_labels(real, "target", "object_none", [0, 1])
    pred.obs = pred.obs.rename(columns={"target": "other"})
    with pytest.raises(ValueError, match="column 'target' missing from pred.obs"):
        validate_pair(pred, real, pert_col="target", control="non-targeting")
