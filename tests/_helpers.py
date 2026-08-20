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
