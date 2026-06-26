"""Test suite for the SEIR compartmental model.

Validates:
    1. Conservation law: S + E + I + R = N at all time steps
    2. Equilibrium: If R0 < 1, epidemic dies out
    3. Peak timing scales correctly with R0
    4. Parameter fitting recovers known parameters
    5. Edge cases: zero infections, extreme R0
"""

import numpy as np
import pytest

from epiagent.engines.seir_model import (
    SEIRParameters,
    SEIRResult,
    run_seir,
    fit_seir,
)


class TestSEIRModel:
    """Tests for SEIR model simulation."""

    def test_conservation_law(self):
        """S + E + I + R must equal N at every time step."""
        params = SEIRParameters.from_epi_params(
            R0=2.5, latent_period=5.2, infectious_period=2.9,
            population=1_000_000, initial_infected=10,
        )
        result = run_seir(params, t_max=365)
        total = result.S + result.E + result.I + result.R
        np.testing.assert_allclose(total, params.N, atol=1.0)

    def test_subcritical_r0_dies_out(self):
        """If R0 < 1, the epidemic should die out (I → 0)."""
        params = SEIRParameters.from_epi_params(
            R0=0.8, latent_period=5.0, infectious_period=3.0,
            population=100_000, initial_infected=100,
        )
        result = run_seir(params, t_max=365)
        # Infectious should decline to near zero
        assert result.I[-1] < 1.0, (
            f"Epidemic should die out with R0={params.R0}, "
            f"but I[-1]={result.I[-1]:.2f}"
        )

    def test_supercritical_r0_epidemic(self):
        """If R0 > 1, there should be a significant epidemic."""
        params = SEIRParameters.from_epi_params(
            R0=3.0, latent_period=5.0, infectious_period=3.0,
            population=100_000, initial_infected=10,
        )
        result = run_seir(params, t_max=365)
        # Should infect a large fraction of population
        total_infected = result.R[-1]
        attack_rate = total_infected / params.N
        assert attack_rate > 0.5, (
            f"With R0=3.0, attack rate should be >50%, got {attack_rate:.1%}"
        )

    def test_higher_r0_earlier_peak(self):
        """Higher R0 should produce an earlier epidemic peak."""
        params_low = SEIRParameters.from_epi_params(
            R0=1.5, latent_period=5.0, infectious_period=3.0,
            population=100_000, initial_infected=10,
        )
        params_high = SEIRParameters.from_epi_params(
            R0=5.0, latent_period=5.0, infectious_period=3.0,
            population=100_000, initial_infected=10,
        )
        result_low = run_seir(params_low, t_max=365)
        result_high = run_seir(params_high, t_max=365)

        assert result_high.peak_day < result_low.peak_day, (
            f"Higher R0 should peak earlier: R0=5.0 peaked at day "
            f"{result_high.peak_day}, R0=1.5 at day {result_low.peak_day}"
        )

    def test_no_initial_infection(self):
        """With I0=0 and E0=0, no epidemic should occur."""
        params = SEIRParameters.from_epi_params(
            R0=5.0, latent_period=5.0, infectious_period=3.0,
            population=100_000, initial_infected=0,
        )
        result = run_seir(params, t_max=100)
        assert result.peak_cases == 0.0
        np.testing.assert_array_equal(result.I, 0.0)

    def test_seir_result_summary(self):
        """SEIRResult.summary() should return a valid dict."""
        params = SEIRParameters.from_epi_params(
            R0=2.5, latent_period=5.2, infectious_period=2.9,
            population=1_000_000, initial_infected=10,
        )
        result = run_seir(params, t_max=180)
        summary = result.summary()
        assert "R0" in summary
        assert "peak_day" in summary
        assert "attack_rate" in summary
        assert 0.0 <= summary["attack_rate"] <= 1.0

    def test_from_epi_params_factory(self):
        """Test the from_epi_params class method."""
        params = SEIRParameters.from_epi_params(
            R0=2.5, latent_period=5.2, infectious_period=2.9,
            population=1_000_000, initial_infected=10,
        )
        assert abs(params.R0 - 2.5) < 0.01
        assert abs(params.latent_period - 5.2) < 0.01
        assert abs(params.infectious_period - 2.9) < 0.01
        assert params.N == 1_000_000

    def test_negative_rates_raise_error(self):
        """Negative rate parameters should raise ValueError."""
        params = SEIRParameters(
            beta=-0.5, sigma=0.2, gamma=0.1, N=100_000, I0=10,
        )
        with pytest.raises(ValueError, match="non-negative"):
            run_seir(params)


class TestSEIRFitting:
    """Tests for SEIR parameter fitting."""

    def test_fit_recovers_known_parameters(self):
        """Fitting should approximately recover known parameters."""
        # Generate data with known R0
        true_R0 = 2.0
        params = SEIRParameters.from_epi_params(
            R0=true_R0, latent_period=5.0, infectious_period=3.0,
            population=500_000, initial_infected=10,
        )
        result = run_seir(params, t_max=120)
        observed = result.daily_incidence

        # Fit
        fitted_params, fitted_result, metrics = fit_seir(
            observed, population=500_000,
            initial_guess={"R0": 3.0, "latent_period": 7.0, "infectious_period": 5.0},
        )

        # R0 should be within 50% of true value (optimization is approximate)
        assert abs(fitted_params.R0 - true_R0) / true_R0 < 0.5, (
            f"Fitted R0={fitted_params.R0:.2f} too far from true R0={true_R0}"
        )

    def test_fit_too_few_points_raises(self):
        """Fitting with <7 data points should raise ValueError."""
        with pytest.raises(ValueError, match="at least 7"):
            fit_seir(np.array([1, 2, 3]), population=100_000)
