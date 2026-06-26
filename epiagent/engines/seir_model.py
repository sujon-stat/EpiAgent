"""SEIR Compartmental Model for Epidemic Simulation.

Implements the Susceptible-Exposed-Infectious-Recovered model using
scipy.integrate.solve_ivp with RK45 method.

Differential Equations:
    dS/dt = -β·S·I/N
    dE/dt =  β·S·I/N - σ·E
    dI/dt =  σ·E - γ·I
    dR/dt =  γ·I

Where:
    β = transmission rate = R0 · γ
    σ = 1/latent_period (rate of progression from E→I)
    γ = 1/infectious_period (recovery rate)
    R0 = basic reproduction number = β/γ

References:
    Kermack WO, McKendrick AG. (1927) "A Contribution to the Mathematical
    Theory of Epidemics." Proceedings of the Royal Society A, 115(772):700-721.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize

logger = logging.getLogger(__name__)


@dataclass
class SEIRParameters:
    """Parameters for the SEIR compartmental model.

    Attributes:
        beta: Transmission rate. Related to R0 by β = R0 · γ.
        sigma: Rate of progression from Exposed to Infectious (1/latent_period).
        gamma: Recovery rate (1/infectious_period).
        N: Total population (constant).
        S0: Initial number of susceptible individuals.
        E0: Initial number of exposed individuals.
        I0: Initial number of infectious individuals.
        R0_init: Initial number of recovered individuals.
    """
    beta: float
    sigma: float
    gamma: float
    N: int
    S0: float = 0.0
    E0: float = 0.0
    I0: float = 10.0
    R0_init: float = 0.0

    def __post_init__(self):
        if self.S0 == 0.0:
            self.S0 = self.N - self.E0 - self.I0 - self.R0_init

    @property
    def R0(self) -> float:
        """Basic reproduction number."""
        if self.gamma == 0:
            return float("inf")
        return self.beta / self.gamma

    @property
    def latent_period(self) -> float:
        """Average latent period in days."""
        if self.sigma == 0:
            return float("inf")
        return 1.0 / self.sigma

    @property
    def infectious_period(self) -> float:
        """Average infectious period in days."""
        if self.gamma == 0:
            return float("inf")
        return 1.0 / self.gamma

    @property
    def serial_interval(self) -> float:
        """Serial interval = latent_period + infectious_period."""
        return self.latent_period + self.infectious_period

    @classmethod
    def from_epi_params(
        cls,
        R0: float,
        latent_period: float,
        infectious_period: float,
        population: int,
        initial_infected: int = 10,
        initial_exposed: int = 0,
    ) -> SEIRParameters:
        """Create parameters from intuitive epidemiological values.

        Args:
            R0: Basic reproduction number.
            latent_period: Average latent period in days.
            infectious_period: Average infectious period in days.
            population: Total population.
            initial_infected: Initial infectious count.
            initial_exposed: Initial exposed count.

        Returns:
            SEIRParameters instance.
        """
        gamma = 1.0 / infectious_period
        sigma = 1.0 / latent_period
        beta = R0 * gamma

        return cls(
            beta=beta,
            sigma=sigma,
            gamma=gamma,
            N=population,
            E0=float(initial_exposed),
            I0=float(initial_infected),
            R0_init=0.0,
        )


@dataclass
class SEIRResult:
    """Results from SEIR model simulation.

    Attributes:
        t: Time points (days).
        S: Susceptible compartment over time.
        E: Exposed compartment over time.
        I: Infectious compartment over time.
        R: Recovered compartment over time.
        daily_incidence: Daily new infections (flow from S→E).
        R0: Basic reproduction number used.
        peak_day: Day of peak infections.
        peak_cases: Maximum number of infectious individuals.
    """
    t: np.ndarray
    S: np.ndarray
    E: np.ndarray
    I: np.ndarray
    R: np.ndarray
    daily_incidence: np.ndarray
    R0: float
    peak_day: int
    peak_cases: float

    def summary(self) -> dict:
        """Return concise summary dict for agent state."""
        return {
            "R0": round(self.R0, 2),
            "peak_day": int(self.peak_day),
            "peak_cases": int(round(self.peak_cases)),
            "total_infected": int(round(self.R[-1])) if len(self.R) > 0 else 0,
            "attack_rate": round(float(self.R[-1] / (self.S[0] + self.E[0] + self.I[0] + self.R[0])), 4)
            if len(self.R) > 0
            else 0.0,
            "duration_days": int(self.t[-1]) if len(self.t) > 0 else 0,
        }


def _seir_odes(t: float, y: list[float], N: int, beta: float, sigma: float, gamma: float):
    """SEIR system of ordinary differential equations.

    Args:
        t: Current time (not used explicitly, required by solve_ivp).
        y: State vector [S, E, I, R].
        N: Total population.
        beta: Transmission rate.
        sigma: Incubation rate (1/latent_period).
        gamma: Recovery rate (1/infectious_period).

    Returns:
        List of derivatives [dS/dt, dE/dt, dI/dt, dR/dt].
    """
    S, E, I, R = y

    # Force of infection
    force_of_infection = beta * S * I / N

    dS = -force_of_infection
    dE = force_of_infection - sigma * E
    dI = sigma * E - gamma * I
    dR = gamma * I

    return [dS, dE, dI, dR]


def run_seir(params: SEIRParameters, t_max: int = 365) -> SEIRResult:
    """Run the SEIR model forward simulation.

    Args:
        params: SEIRParameters defining the model.
        t_max: Maximum simulation time in days.

    Returns:
        SEIRResult with full compartment trajectories.

    Raises:
        ValueError: If parameters are invalid (negative rates, etc.).
    """
    # Validate parameters
    if params.beta < 0 or params.sigma < 0 or params.gamma < 0:
        raise ValueError(
            f"Rates must be non-negative: beta={params.beta}, "
            f"sigma={params.sigma}, gamma={params.gamma}"
        )
    if params.N <= 0:
        raise ValueError(f"Population must be positive: N={params.N}")

    # Handle edge case: no initial infections → no epidemic
    if params.I0 == 0 and params.E0 == 0:
        t = np.arange(t_max + 1, dtype=float)
        n = len(t)
        return SEIRResult(
            t=t,
            S=np.full(n, params.S0),
            E=np.zeros(n),
            I=np.zeros(n),
            R=np.full(n, params.R0_init),
            daily_incidence=np.zeros(n),
            R0=params.R0,
            peak_day=0,
            peak_cases=0.0,
        )

    # Initial conditions
    y0 = [params.S0, params.E0, params.I0, params.R0_init]

    # Time span
    t_span = (0, t_max)
    t_eval = np.arange(t_max + 1, dtype=float)

    # Solve ODE system
    sol = solve_ivp(
        fun=_seir_odes,
        t_span=t_span,
        y0=y0,
        args=(params.N, params.beta, params.sigma, params.gamma),
        t_eval=t_eval,
        method="RK45",
        rtol=1e-8,
        atol=1e-8,
    )

    if not sol.success:
        raise RuntimeError(f"ODE solver failed: {sol.message}")

    S, E, I, R = sol.y

    # Conservation law check: S + E + I + R = N
    total = S + E + I + R
    max_deviation = np.max(np.abs(total - params.N))
    if max_deviation > 1e-3:
        logger.warning(
            "Conservation law violation: max deviation = %.6f (threshold=1e-3)",
            max_deviation,
        )

    # Compute daily incidence (new infections per day = flow from S to E)
    daily_incidence = compute_daily_incidence(S, params.beta, I, params.N)

    # Find peak
    peak_idx = int(np.argmax(I))

    result = SEIRResult(
        t=sol.t,
        S=S,
        E=E,
        I=I,
        R=R,
        daily_incidence=daily_incidence,
        R0=params.R0,
        peak_day=peak_idx,
        peak_cases=float(I[peak_idx]),
    )

    logger.info(
        "SEIR simulation complete: R0=%.2f, peak at day %d with %d cases",
        params.R0, peak_idx, int(I[peak_idx]),
    )

    return result


def compute_daily_incidence(
    S: np.ndarray, beta: float, I: np.ndarray, N: int
) -> np.ndarray:
    """Compute daily new infections from SEIR compartments.

    Daily incidence = β·S·I/N (the flow from S to E per unit time).

    Args:
        S: Susceptible compartment array.
        beta: Transmission rate.
        I: Infectious compartment array.
        N: Total population.

    Returns:
        Array of daily incidence values.
    """
    if N == 0:
        return np.zeros_like(S)
    incidence = beta * S * I / N
    return np.maximum(incidence, 0.0)


def fit_seir(
    observed_cases: np.ndarray,
    population: int,
    initial_guess: dict | None = None,
    method: str = "Nelder-Mead",
) -> tuple[SEIRParameters, SEIRResult, dict]:
    """Fit SEIR model parameters to observed daily case counts.

    Uses scipy.optimize.minimize to find β, σ, γ that minimize the
    RMSE between model-predicted daily incidence and observed cases.

    Args:
        observed_cases: Array of daily observed case counts.
        population: Total population.
        initial_guess: Optional dict with keys 'R0', 'latent_period',
            'infectious_period'. Defaults to COVID-like parameters.
        method: Optimization method (default: Nelder-Mead).

    Returns:
        Tuple of (fitted_params, fitted_result, fit_metrics).
        fit_metrics contains 'rmse', 'r_squared', 'n_iterations'.
    """
    observed = np.asarray(observed_cases, dtype=float)
    T = len(observed)

    if T < 7:
        raise ValueError(f"Need at least 7 data points for fitting, got {T}")

    # Default initial guess: COVID-like parameters
    if initial_guess is None:
        initial_guess = {
            "R0": 2.5,
            "latent_period": 5.2,
            "infectious_period": 2.9,
        }

    # Estimate initial infected from first non-zero observation
    first_nonzero = np.argmax(observed > 0)
    I0 = max(observed[first_nonzero], 1.0) if first_nonzero < T else 1.0

    def objective(x):
        """Objective function: RMSE between model and observed."""
        R0_est, lat_est, inf_est = x

        # Parameter bounds enforcement
        if R0_est <= 0 or lat_est <= 0 or inf_est <= 0:
            return 1e10

        try:
            params = SEIRParameters.from_epi_params(
                R0=R0_est,
                latent_period=lat_est,
                infectious_period=inf_est,
                population=population,
                initial_infected=int(I0),
            )
            result = run_seir(params, t_max=T - 1)

            # Compare daily incidence
            model_incidence = result.daily_incidence[:T]
            if len(model_incidence) < T:
                return 1e10

            # RMSE
            rmse = np.sqrt(np.mean((model_incidence - observed) ** 2))
            return rmse

        except (ValueError, RuntimeError):
            return 1e10

    # Initial parameter vector: [R0, latent_period, infectious_period]
    x0 = [
        initial_guess["R0"],
        initial_guess["latent_period"],
        initial_guess["infectious_period"],
    ]

    # Optimize
    opt_result = minimize(
        objective,
        x0=x0,
        method=method,
        options={"maxiter": 2000, "xatol": 1e-6, "fatol": 1e-6},
    )

    # Extract fitted parameters
    R0_fit, lat_fit, inf_fit = opt_result.x

    # Clamp to reasonable ranges
    R0_fit = max(0.1, min(R0_fit, 30.0))
    lat_fit = max(0.5, min(lat_fit, 30.0))
    inf_fit = max(0.5, min(inf_fit, 30.0))

    fitted_params = SEIRParameters.from_epi_params(
        R0=R0_fit,
        latent_period=lat_fit,
        infectious_period=inf_fit,
        population=population,
        initial_infected=int(I0),
    )

    fitted_result = run_seir(fitted_params, t_max=T - 1)

    # Compute fit metrics
    model_incidence = fitted_result.daily_incidence[:T]
    rmse = float(np.sqrt(np.mean((model_incidence - observed) ** 2)))

    # R-squared
    ss_res = np.sum((observed - model_incidence) ** 2)
    ss_tot = np.sum((observed - np.mean(observed)) ** 2)
    r_squared = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0

    fit_metrics = {
        "rmse": round(rmse, 4),
        "r_squared": round(r_squared, 4),
        "n_iterations": int(opt_result.nit),
        "converged": bool(opt_result.success),
        "fitted_R0": round(R0_fit, 3),
        "fitted_latent_period": round(lat_fit, 3),
        "fitted_infectious_period": round(inf_fit, 3),
    }

    logger.info(
        "SEIR fitting complete: R0=%.2f, RMSE=%.2f, R²=%.4f, converged=%s",
        R0_fit, rmse, r_squared, opt_result.success,
    )

    return fitted_params, fitted_result, fit_metrics
