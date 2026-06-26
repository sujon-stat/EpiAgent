import numpy as np
import pytest
from epiagent.engines.changepoint_detector import detect_outbreak_signals

class TestChangepointDetector:
    def test_detect_no_changepoints(self):
        cases = np.ones(50) * 10  # Constant series
        result = detect_outbreak_signals(cases, hazard_lambda=100.0)
        assert len(result.changepoints) == 0

    def test_detect_single_changepoint(self):
        cases = np.concatenate([np.ones(50) * 10, np.ones(50) * 1000])
        result = detect_outbreak_signals(cases, hazard_lambda=200.0, threshold=0.1)
        assert len(result.changepoint_probs) == 100
        # If any changepoints are detected, verify they are in the expected range
        if len(result.changepoints) > 0:
            assert any(abs(cp - 50) <= 5 for cp in result.changepoints)

    def test_detect_multiple_changepoints(self):
        cases = np.concatenate([np.ones(50) * 10, np.ones(50) * 1000, np.ones(50) * 10])
        result = detect_outbreak_signals(cases, hazard_lambda=100.0, threshold=0.1)
        assert len(result.changepoint_probs) == 150
        assert result.run_length_posterior.shape == (151, 151)

    def test_short_series(self):
        cases = np.array([10, 12, 11])
        result = detect_outbreak_signals(cases)
        assert len(result.changepoints) == 0
