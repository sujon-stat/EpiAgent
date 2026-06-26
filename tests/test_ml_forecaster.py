import numpy as np
import pytest
from epiagent.engines.ml_forecaster import forecast_ensemble, create_lag_features

class TestMLForecaster:
    def test_create_lag_features(self):
        cases = np.arange(100, dtype=float)
        df = create_lag_features(cases, n_lags=7)
        assert not df.empty
        assert "lag_1" in df.columns
        assert "lag_7" in df.columns
        assert "rolling_7d_mean" in df.columns
        assert len(df) == 100 - 7

    def test_forecast_ensemble_basic(self):
        # Generate a sine wave to forecast
        t = np.arange(200)
        cases = 100 + 50 * np.sin(t * 2 * np.pi / 30) + np.random.normal(0, 5, 200)
        
        ensemble, individual = forecast_ensemble(cases, horizon=7)
        assert "Ensemble" in ensemble.model_name
        assert len(ensemble.predicted) == 7
        assert len(ensemble.lower_bound) == 7
        assert len(ensemble.upper_bound) == 7
        assert ensemble.rmse is not None
        assert len(individual) >= 1  # At least XGBoost

    def test_forecast_short_series(self):
        cases = np.arange(10, dtype=float)
        with pytest.raises(RuntimeError):
            forecast_ensemble(cases)
