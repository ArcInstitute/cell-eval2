"""Shared helpers for the test suite.

A normal importable module rather than ``conftest.py``. pytest loads ``conftest.py`` as a
PLUGIN; importing it directly happens to work under the default ``prepend`` import mode but
is unsupported, and breaks outright under ``--import-mode=importlib``, which inserts nothing
into ``sys.path``. ``pyproject.toml``'s ``[tool.pytest.ini_options] pythonpath`` lists
``tests`` so this module is importable under ANY import mode. (Copilot flagged the direct
conftest import at seven sites on PR #209; moving the helper is the durable half of the fix,
the ``pythonpath`` entry is the other half -- ``tests/_helpers.py`` alone would still rely on
``prepend`` putting ``tests/`` on the path.)
"""
from __future__ import annotations


def _counts_adata_fp64(*, seed: int, per_group: int, g: int):
    """Small fp64 counts panel for algebraic parity across public drivers."""
    import anndata as ad
    import numpy as np
    import pandas as pd
    import scipy.sparse as sp

    rng = np.random.default_rng(seed)
    labels = np.array(["non-targeting", "A", "B", "C"])
    rates = rng.gamma(2.0, 3.0, size=(labels.size, g))
    X = np.vstack([
        rng.poisson(rates[i], size=(per_group, g))
        for i in range(labels.size)
    ]).astype(np.float64)
    obs = pd.DataFrame({"target": np.repeat(labels, per_group)})
    return ad.AnnData(X=sp.csr_matrix(X), obs=obs)


def _r_zero_adata_fp64():
    """fp64 counts panel containing the ``[[4, 0], [0, 0]]`` r_i == 0 group."""
    import anndata as ad
    import numpy as np
    import pandas as pd
    import scipy.sparse as sp

    X = np.array([
        [3.0, 1.0], [2.0, 2.0], [1.0, 4.0],
        [4.0, 0.0], [0.0, 0.0],
        [1.0, 3.0], [2.0, 1.0], [4.0, 2.0],
    ], dtype=np.float64)
    obs = pd.DataFrame({
        "target": ["non-targeting"] * 3 + ["A"] * 2 + ["B"] * 3,
    })
    return ad.AnnData(X=sp.csr_matrix(X), obs=obs)


def _dispatch_cfg():
    """Minimal v2 counts config shared by direct dispatcher tests."""
    from cell_eval2.config import EvalConfig

    return EvalConfig(
        version="v2",
        input_type="counts",
        control="non-targeting",
        device="cpu",
        pert_col="target",
    )


def resolved_comparator(cfg, *, pred_input_type=None, real_input_type=None) -> str:
    """Resolve a test run's comparator from its config and effective input types."""
    from cell_eval2.norm import resolve_comparator

    return resolve_comparator(
        version=cfg.version,
        pred_input_type=pred_input_type or cfg.input_type,
        real_input_type=real_input_type or cfg.input_type,
    )


def full_minus_moments() -> list[str]:
    """The `full` profile minus every moments-consuming metric (issue #198).

    The partitioned in-memory scorer (`partition_inmem.score_piece`) cannot supply per-group
    moments, and all three public entry points -- `cellstream.score_cellstream`,
    `h5ad_manifest.score_h5ad_manifest` and `rowstore.score_rowstore` -- run through it, so
    their success and parity tests score `full` minus those metrics and derived metrics whose
    components need moments. Derived from PROFILES rather than hard-coded, so a later
    full-profile moments metric is picked up here too. Issue #272 deletes this helper entirely.
    """
    from cell_eval2.catalog import PROFILES, needs_moments_transitively

    return [m for m in PROFILES["full"] if not needs_moments_transitively(m)]


def counts_bundle_inputs():
    """`(baseline_pred, real, submission_pred)`: integer counts defining a USABLE baseline ->
    replicate scale for all six `vcc2026` members.

    Lives here rather than only in `conftest.py` because `tests/test_golden_score.py` needs it at
    MODULE scope and a module-scoped fixture cannot depend on a function-scoped one. Every
    property of these inputs is load-bearing; see `conftest.counts_bundle_inputs`' docstring,
    which is the record of why (per-target distinct effects, a no-skill baseline arm, and targets
    present in `var_names`).
    """
    import anndata as ad
    import numpy as np
    import pandas as pd

    perts = ("non-targeting", "GENE1", "GENE2", "GENE3", "GENE4", "GENE5")
    n_genes, n_cells = 40, 30
    named = [p for p in perts if p != "non-targeting"]
    var = pd.DataFrame(index=named + [f"g{j}" for j in range(n_genes - len(named))])
    rng = np.random.default_rng(11)
    effects = {p: rng.integers(0, 8, size=n_genes).astype(np.float32) for p in named}

    def _build(seed, eff):
        r = np.random.default_rng(seed)
        blocks, labels = [], []
        for p in perts:
            X = r.poisson(lam=8.0, size=(n_cells, n_genes)).astype(np.float32)
            if eff and p in eff:
                X = X + eff[p]
            blocks.append(X)
            labels += [p] * n_cells
        X = np.vstack(blocks).astype(np.float32)
        return ad.AnnData(X=X, var=var.copy(),
                          obs=pd.DataFrame({"target": labels},
                                           index=[f"c{i}" for i in range(X.shape[0])]))

    real = _build(11, effects)
    baseline_pred = _build(13, None)          # no skill: the 0 end
    submission_pred = _build(12, effects)     # a good prediction: something to score
    return baseline_pred, real, submission_pred


# ---- #391 gate fixtures -----------------------------------------------------------------------
# One 60-gene DE block per target. Sixty because the polars defect the gate's constancy check
# guards against (internal:docs/internal/superpowers/specs/2026-10-08-real-side-nan-not-scored-design.md
# §5.2) first shows at a gate of 49 genes, and the floor is 10.

GATE_GENES = [f"g{i:02d}" for i in range(60)]
GATE_NS = 0.9                     # a non-significant p_adj


def gate_block(target, lfc, padj, genes=None):
    """A DE table block for one target: one row per gene. ``None`` stays null."""
    import polars as pl

    genes = GATE_GENES if genes is None else list(genes)
    return pl.DataFrame({"target": [target] * len(genes), "feature": genes,
                         "log2_fold_change": [None if x is None else float(x) for x in lfc],
                         "p_adj": [None if x is None else float(x) for x in padj]},
                        schema={"target": pl.String, "feature": pl.String,
                                "log2_fold_change": pl.Float64, "p_adj": pl.Float64})


def gate_sig(k, n=len(GATE_GENES)):
    """p_adj for n genes, the first k of them significant."""
    return [1e-4] * k + [GATE_NS] * (n - k)


def gate_lfc(seed, n=len(GATE_GENES)):
    """n log2FCs drawn from a fixed seed -- distinct values, both signs."""
    import numpy as np

    return np.random.default_rng(seed).normal(size=n).tolist()


def gate_prep(real_blocks, pred_blocks):
    """A PreparedDE from per-target blocks."""
    import polars as pl

    from cell_eval2.de import prepare_de

    return prepare_de(pl.concat(pred_blocks), pl.concat(real_blocks), control="non-targeting")


def gate_v2(names, prep, **cfg):
    """{metric: {perturbation: value}} from the v2 DE dispatch."""
    from cell_eval2.config import EvalConfig
    from cell_eval2.run import dispatch_de_metrics

    rows = dispatch_de_metrics(names, prep, EvalConfig(version="v2", device="cpu", **cfg))
    out: dict[str, dict[str, float]] = {}
    for r in rows:
        out.setdefault(r["metric"], {})[r["perturbation"]] = r["value"]
    return out


# Label column kinds that a missing perturbation label can arrive in.
MISSING_LABEL_KINDS = ("object_none", "object_nan", "string_na", "str", "categorical")


def make_labels(kind, labels, missing_at, *, index=None):
    """A label Series of ``kind`` that holds a missing value at each position in ``missing_at``.

    Asserts the realised dtype, so a kind that pandas or anndata silently converts to another
    fails here and not as a vacuous pass in the test that uses it. ``str`` is
    ``pd.StringDtype(storage="python", na_value=np.nan)``: a pandas 3 string dtype,
    spelled so that pandas 2.3 builds the same dtype.
    """
    import numpy as np
    import pandas as pd

    values = [None if i in set(missing_at) else v for i, v in enumerate(labels)]
    if kind == "object_none":
        s = pd.Series(values, index=index, dtype=object)
        assert s.dtype == object and all(values[i] is None for i in missing_at)
    elif kind == "object_nan":
        s = pd.Series([np.nan if v is None else v for v in values], index=index, dtype=object)
        assert s.dtype == object and all(isinstance(s.iloc[i], float) for i in missing_at)
    elif kind == "string_na":
        s = pd.Series(pd.array(values, dtype="string"), index=index)
        assert isinstance(s.dtype, pd.StringDtype) and s.dtype.na_value is pd.NA
        assert all(s.iloc[i] is pd.NA for i in missing_at)
    elif kind == "str":
        import pytest

        if tuple(int(v) for v in pd.__version__.split(".")[:2]) < (2, 3):
            pytest.skip("pd.StringDtype(na_value=...) needs pandas >= 2.3")
        dtype = pd.StringDtype(storage="python", na_value=np.nan)
        s = pd.Series(values, index=index, dtype=dtype)
        assert s.dtype == dtype
    elif kind == "categorical":
        s = pd.Series(pd.Categorical(values), index=index)
        assert isinstance(s.dtype, pd.CategoricalDtype)
    else:
        raise ValueError(f"unknown label kind {kind!r}")
    assert int(s.isna().sum()) == len(set(missing_at))
    return s


def set_labels(adata, column, kind, missing_at):
    """Replace ``adata.obs[column]`` by a ``kind`` column with missing values; return ``adata``."""
    labels = adata.obs[column].astype(object).tolist()
    new = make_labels(kind, labels, missing_at, index=adata.obs.index)
    adata.obs[column] = new
    assert adata.obs[column].dtype == new.dtype, (kind, adata.obs[column].dtype)
    return adata


def assert_missing_labels_error(excinfo, *, n_missing, n_cells, column, side, other_side=None):
    """The ``require_pert_labels`` message: the count, the column and the side, and for a
    one-sided failure not the other side's name. Returns the message."""
    msg = str(excinfo.value)
    assert f"{n_missing} of {n_cells} cells in {side} have a missing (null) perturbation label" in msg
    assert repr(column) in msg
    if other_side is not None:
        assert other_side not in msg
    return msg
