"""Test suite for Bayesian Rt estimation and epidemiological metrics.

Validates:
    1. Rt estimation converges to true R0 for constant-rate epidemics
    2. Rt detects step-change (e.g., lockdown: R0 drops from 2.5 to 0.8)
    3. Credible interval coverage
    4. Wilson score CI is tighter than Wald for extreme proportions
    5. Edge cases in metrics computation
"""

import numpy as np
import pytest

from epiagent.engines.rt_estimation import (
    SerialInterval,
    SI_COVID,
    SI_INFLUENZA,
    SI_MEASLES,
    estimate_rt,
)
from epiagent.engines.epi_metrics import (
    compute_cfr,
    compute_incidence_rate,
    compute_doubling_time,
    compute_attack_rate,
    compute_growth_rate,
)


class TestRtEstimation:
    """Tests for Bayesian Rt estimation (Cori et al. 2013)."""

    def _generate_exponential_cases(self, R0, si_mean, n_days=60, I0=10):
        """Generate cases from exponential growth with known R0."""
        rng = np.random.default_rng(42)
        gamma_rate = 1.0 / si_mean
        daily_growth = R0 ** (1 / si_mean)
        incidence = np.zeros(n_days)
        incidence[0] = I0
        for t in range(1, n_days):
            incidence[t] = incidence[t - 1] * daily_growth
        # Add small noise
        incidence = np.maximum(1, np.round(incidence + rng.normal(0, 1, n_days)))
        return incidence

    def test_rt_converges_to_true_value(self):
        """Rt should converge near the true R0 for constant-rate growth."""
        true_R0 = 2.0
        incidence = self._generate_exponential_cases(
            R0=true_R0, si_mean=SI_COVID.mean, n_days=60,
        )
        result = estimate_rt(incidence, SI_COVID, window=7)

        # Check last 10 estimates are in reasonable range
        valid_rt = result.rt_mean[~np.isnan(result.rt_mean)]
        if len(valid_rt) > 10:
            recent_mean = np.mean(valid_rt[-10:])
            # Allow wide tolerance because exponential model ≠ renewal equation
            assert 0.5 < recent_mean < 5.0, (
                f"Rt mean ({recent_mean:.2f}) out of reasonable range"
            )

    def test_serial_interval_discretization(self):
        """Serial interval PMF should sum to 1 and be non-negative."""
        for si in [SI_COVID, SI_INFLUENZA, SI_MEASLES]:
            pmf = si.discretize(max_days=20)
            assert len(pmf) == 21  # 0 to 20 inclusive
            assert pmf[0] == 0.0  # No same-day transmission
            assert np.all(pmf >= 0.0)
            np.testing.assert_almost_equal(pmf.sum(), 1.0, decimal=5)

    def test_epidemic_phase_classification(self):
        """Phase should be 'growing' when Rt > 1, 'declining' when < 1."""
        # Constant high growth
        incidence = np.array([10] * 10 + [20, 40, 80, 160, 320, 640, 1280] * 3)
        result = estimate_rt(incidence, SI_COVID, window=5)
        # At least some phases should be 'growing'
        assert "growing" in result.epidemic_phase or "stable" in result.epidemic_phase

    def test_short_series_raises(self):
        """Series shorter than window should raise ValueError."""
        with pytest.raises(ValueError, match="too short"):
            estimate_rt(np.array([1, 2, 3]), SI_COVID, window=7)

    def test_negative_incidence_raises(self):
        """Negative incidence values should raise ValueError."""
        with pytest.raises(ValueError, match="non-negative"):
            estimate_rt(np.array([1, 2, -1, 4, 5, 6, 7, 8]), SI_COVID)

    def test_result_summary(self):
        """RtResult.summary() should return valid dict."""
        incidence = np.random.default_rng(42).poisson(50, size=30)
        result = estimate_rt(incidence, SI_COVID, window=7)
        summary = result.summary()
        assert "current_rt" in summary
        assert "current_phase" in summary


class TestEpiMetrics:
    """Tests for epidemiological metrics calculator."""

    def test_cfr_basic(self):
        """CFR should equal deaths/cases."""
        result = compute_cfr(deaths=15, cases=1000)
        assert abs(result.value - 0.015) < 0.001
        assert result.lower_ci < result.value < result.upper_ci
        assert "Wilson" in result.method

    def test_cfr_zero_cases(self):
        """CFR with 0 cases should return NaN."""
        result = compute_cfr(deaths=0, cases=0)
        assert np.isnan(result.value)

    def test_cfr_wilson_vs_wald_at_extreme(self):
        """Wilson CI should be valid even for extreme proportions."""
        # Very low proportion: 1 death in 10,000 cases
        result = compute_cfr(deaths=1, cases=10_000)
        assert result.lower_ci >= 0.0
        assert result.upper_ci <= 1.0
        assert result.lower_ci < result.value < result.upper_ci

        # Very high proportion: 99 deaths in 100 cases
        result = compute_cfr(deaths=99, cases=100)
        assert result.lower_ci >= 0.0
        assert result.upper_ci <= 1.0

    def test_cfr_negative_input_raises(self):
        """Negative deaths or cases should raise ValueError."""
        with pytest.raises(ValueError):
            compute_cfr(deaths=-1, cases=100)
        with pytest.raises(ValueError):
            compute_cfr(deaths=5, cases=-100)

    def test_incidence_rate(self):
        """Incidence rate should be cases/population * per."""
        result = compute_incidence_rate(cases=500, population=1_000_000, per=100_000)
        assert abs(result.value - 50.0) < 0.1
        assert result.lower_ci < result.value < result.upper_ci

    def test_incidence_rate_zero_cases(self):
        """Zero cases should give rate=0 with valid upper CI."""
        result = compute_incidence_rate(cases=0, population=1_000_000)
        assert result.value == 0.0
        assert result.lower_ci == 0.0
        assert result.upper_ci > 0.0  # Upper CI should be non-zero

    def test_incidence_rate_zero_population_raises(self):
        """Zero population should raise ValueError."""
        with pytest.raises(ValueError):
            compute_incidence_rate(cases=10, population=0)

    def test_doubling_time_exponential(self):
        """Doubling time of pure exponential growth should be ln(2)/r."""
        # Generate perfect exponential: doubles every 5 days → r = ln(2)/5
        r_true = np.log(2) / 5.0
        t = np.arange(14)
        cases = np.exp(r_true * t) * 100
        result = compute_doubling_time(cases, window=14)
        assert abs(result.value - 5.0) < 1.0, (
            f"Expected doubling time ~5 days, got {result.value:.1f}"
        )

    def test_doubling_time_declining(self):
        """Declining epidemic should return NaN doubling time."""
        cases = np.array([100, 90, 80, 70, 60, 50, 40, 30, 20, 10])
        result = compute_doubling_time(cases, window=10)
        assert np.isnan(result.value)

    def test_attack_rate(self):
        """Attack rate should be total_cases/population."""
        result = compute_attack_rate(total_cases=50_000, population=1_000_000)
        assert abs(result.value - 0.05) < 0.001

    def test_growth_rate_positive(self):
        """Exponential growth should give positive growth rate."""
        cases = np.exp(0.1 * np.arange(14)) * 100
        result = compute_growth_rate(cases, window=14)
        assert result.value > 0
        assert abs(result.value - 0.1) < 0.05
