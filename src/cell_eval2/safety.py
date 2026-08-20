from __future__ import annotations

import numpy as np
from scipy.stats import pearsonr


def require_finite(arr: np.ndarray, *, name: str = "array") -> None:
    """Raise if `arr` contains NaN or inf."""
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains non-finite values (NaN/inf)")


def safe_mae(x: np.ndarray, y: np.ndarray) -> float:
    """Mean absolute error between two equal-shaped arrays; NaN if either is empty."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.size == 0 or y.size == 0:
        return float("nan")
    if x.shape != y.shape:
        raise ValueError(f"safe_mae shape mismatch: {x.shape} != {y.shape}")
    return float(np.mean(np.abs(x - y)))


def safe_mse(x: np.ndarray, y: np.ndarray) -> float:
    """Mean squared error between two equal-shaped arrays; NaN if either is empty."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.size == 0 or y.size == 0:
        return float("nan")
    if x.shape != y.shape:
        raise ValueError(f"safe_mse shape mismatch: {x.shape} != {y.shape}")
    return float(np.mean((x - y) ** 2))


def safe_pearson(x: np.ndarray, y: np.ndarray) -> float:
    """Pearson correlation coefficient between two equal-shaped 1-D arrays.

    Uses ``scipy.stats.pearsonr`` (exactly upstream cell_eval's call) and returns only
    the coefficient. Fewer than 2 points -> NaN (would otherwise raise); a zero-variance
    input yields scipy's NaN, which is propagated (faithful to upstream, which does no
    special handling - unreachable for real pseudobulk deltas over many genes).
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.size < 2 or y.size < 2:        # size/empty check first, like safe_mae/safe_mse
        return float("nan")
    if x.shape != y.shape:
        raise ValueError(f"safe_pearson shape mismatch: {x.shape} != {y.shape}")
    return float(pearsonr(x, y)[0])
