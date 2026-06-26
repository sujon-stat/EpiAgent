import numpy as np
import pytest
from epiagent.engines.epi_metrics import (
    compute_cfr,
    compute_incidence_rate,
    compute_doubling_time,
    compute_attack_rate,
    compute_growth_rate
)

class TestEpiMetrics:
    def test_compute_cfr(self):
        res = compute_cfr(150, 10000)
        assert np.isclose(res.value, 0.015)
        assert res.lower_ci < res.value < res.upper_ci
        assert "Wilson" in res.method

    def test_compute_incidence_rate(self):
        res = compute_incidence_rate(500, 100000, per=100000)
        assert np.isclose(res.value, 500.0)
        assert res.lower_ci < res.value < res.upper_ci

    def test_compute_doubling_time_growing(self):
        cases = np.array([10, 15, 22, 33, 50, 75, 110])
        res = compute_doubling_time(cases)
        assert res.value > 0
        assert res.lower_ci < res.value < res.upper_ci

    def test_compute_doubling_time_declining(self):
        cases = np.array([100, 80, 60, 45, 30, 20, 10])
        res = compute_doubling_time(cases)
        assert np.isnan(res.value)  # Declining should result in NaN doubling time

    def test_compute_attack_rate(self):
        res = compute_attack_rate(1000, 10000)
        assert np.isclose(res.value, 0.1)
        assert res.lower_ci < res.value < res.upper_ci

    def test_zero_cases_deaths(self):
        res = compute_cfr(0, 100)
        assert res.value == 0.0
        assert np.isclose(res.lower_ci, 0.0)
        assert res.upper_ci > 0.0

    def test_zero_population(self):
        with pytest.raises(ValueError):
            compute_incidence_rate(100, 0)
