import numpy as np
import pandas as pd
import pytest
from src.calibration.beta_prediction import BetaPredictor

@pytest.fixture
def predictor():
    return BetaPredictor()

@pytest.fixture
def beta_series():
    return pd.Series([0.001, 0.002, 0.004, 0.003, 0.002])

METHODS = ["last_value", "rolling_mean", "expanding_mean", "biexponential", "median_beta"]

@pytest.mark.parametrize("method", METHODS)
def test_output_length(predictor, beta_series, method):
    for n_ahead in [1, 3, 5]:
        result = predictor.predict(method, beta_series, n_ahead=n_ahead)
        assert len(result) == n_ahead, f"{method} returned {len(result)}, expected {n_ahead}"

@pytest.mark.parametrize("method", METHODS)
def test_values_positive(predictor, beta_series, method):
    result = predictor.predict(method, beta_series, n_ahead=5)
    assert (result >= 0).all(), f"{method} produced negative values"

@pytest.mark.parametrize("method", METHODS)
def test_no_nan(predictor, beta_series, method):
    result = predictor.predict(method, beta_series, n_ahead=5)
    assert not result.isna().any(), f"{method} produced NaN"

def test_rolling_mean_value(predictor, beta_series):
    result = predictor.predict("rolling_mean", beta_series, n_ahead=1, window=3)
    expected = beta_series.iloc[-3:].mean()
    assert abs(result.iloc[0] - expected) < 1e-10

def test_median_beta_value(predictor, beta_series):
    result = predictor.predict("median_beta", beta_series, n_ahead=1)
    expected = float(np.median(beta_series.values))
    assert abs(result.iloc[0] - expected) < 1e-10

@pytest.mark.parametrize("method", METHODS)
def test_no_division_by_n(method):
    import inspect
    from src.calibration.beta_prediction import BetaPredictor
    method_map = {
        "last_value": "_last_value",
        "rolling_mean": "_rolling_mean",
        "expanding_mean": "_expanding_mean",
        "biexponential": "_biexponential",
        "median_beta": "_median_beta",
    }
    src = inspect.getsource(getattr(BetaPredictor, method_map[method]))
    assert "/ N" not in src and "/N" not in src, f"{method} divides by N"

def test_lstm_raises_without_model(predictor, beta_series):
    with pytest.raises(FileNotFoundError, match="trained model not found"):
        predictor.predict("lstm_day_e_prev_i", beta_series, n_ahead=1)

def test_unknown_method_raises(predictor, beta_series):
    with pytest.raises(ValueError, match="unknown beta prediction method"):
        predictor.predict("nonexistent_method", beta_series, n_ahead=1)
