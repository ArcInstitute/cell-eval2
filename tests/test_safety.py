import math
import warnings

import numpy as np
import pytest
from sklearn.metrics import mean_absolute_error

from cell_eval2.safety import require_finite, safe_mae, safe_mse, safe_pearson


def test_safe_mae_matches_sklearn():
    x = np.array([1.0, 2.0, 3.0])
    y = np.array([1.5, 1.0, 5.0])
    assert safe_mae(x, y) == pytest.approx(mean_absolute_error(y, x))


def test_safe_mae_empty_is_nan():
    assert np.isnan(safe_mae(np.array([]), np.array([])))


def test_safe_mae_empty_y_is_nan():
    assert np.isnan(safe_mae(np.array([1.0, 2.0]), np.array([])))


def test_safe_mae_shape_mismatch_raises():
    with pytest.raises(ValueError, match="shape mismatch"):
        safe_mae(np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0]))


def test_require_finite_raises_on_nan():
    with pytest.raises(ValueError, match="non-finite"):
        require_finite(np.array([1.0, np.nan]), name="vec")


def test_safe_mse_matches_mean_of_squares():
    x = np.array([1.0, 2.0, 3.0])
    y = np.array([1.0, 0.0, 6.0])
    assert safe_mse(x, y) == pytest.approx((0.0 + 4.0 + 9.0) / 3.0)


def test_safe_mse_matches_sklearn():
    skm = pytest.importorskip("sklearn.metrics")
    rng = np.random.default_rng(0)
    x, y = rng.normal(size=50), rng.normal(size=50)
    assert safe_mse(x, y) == pytest.approx(skm.mean_squared_error(x, y), rel=1e-6, abs=1e-9)


def test_safe_mse_empty_is_nan():
    assert math.isnan(safe_mse(np.array([]), np.array([])))


def test_safe_mse_shape_mismatch_raises():
    with pytest.raises(ValueError, match="shape mismatch"):
        safe_mse(np.array([1.0, 2.0]), np.array([1.0]))


def test_safe_pearson_matches_scipy():
    stats = pytest.importorskip("scipy.stats")
    rng = np.random.default_rng(1)
    x, y = rng.normal(size=40), rng.normal(size=40)
    assert safe_pearson(x, y) == pytest.approx(stats.pearsonr(x, y)[0], rel=1e-12)


def test_safe_pearson_zero_variance_is_nan():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # scipy ConstantInputWarning; robust if filterwarnings=error
        assert math.isnan(safe_pearson(np.zeros(5), np.array([1.0, 2.0, 3.0, 4.0, 5.0])))


def test_safe_pearson_too_short_is_nan():
    assert math.isnan(safe_pearson(np.array([1.0]), np.array([2.0])))


def test_safe_pearson_shape_mismatch_raises():
    with pytest.raises(ValueError, match="shape mismatch"):
        safe_pearson(np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0]))
