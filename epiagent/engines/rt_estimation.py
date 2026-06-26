"""Bayesian Real-Time Effective Reproduction Number (Rt) Estimation.

Implements the Cori et al. (2013) method — the WHO/CDC gold standard
for real-time Rt estimation during epidemics.

Reference:
    Cori A, Ferguson NM, Fraser C, Cauchemez S. (2013)
    "A New Framework and Software to Estimate Time-Varying
    Reproduction Numbers During Epidemics."
    American Journal of Epidemiology, 178(9):1505-1512.

Method:
    The instantaneous reproduction number Rt is estimated using a
    Bayesian framework with a Gamma conjugate prior:

    Likelihood: I_t ~ Poisson(R_t · Λ_t)
    Prior: R_t ~ Gamma(a, scale=b)
    Posterior: R_t | data ~ Gamma(a_post, scale=b_post)

    Where:
        a_post = a_prior + Σ I_t  (sum of cases over sliding window)
        b_post = 1 / (1/b_prior + Σ Λ_t)  (total infectiousness)
        Λ_t = Σ_{s=1}^{T} I_{t-s} · w_s  (total infectiousness at time t)
        w_s = serial interval distribution (discretized Gamma PMF)

    The 95% credible interval is computed from the Gamma posterior quantiles.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
from scipy.stats import gamma as gamma_dist

logger = logging.getLogger(__name__)


@dataclass
class SerialInterval:
    """Serial interval distribution for a pathogen.

    The serial interval is the time between symptom onset in a primary
    case and symptom onset in a secondary case.

    Attributes:
        mean: Mean serial interval in days.
        std: Standard deviation of serial interval in days.
    """
    mean: float
    std: float

    def discretize(self, max_days: int = 20) -> np.ndarray:
        """Discretize the serial interval into a probability mass function.

        Uses a Gamma distribution parameterized by the mean and std,
        evaluated at integer day values and normalized to sum to 1.

        Args:
            max_days: Maximum number of days to include in the PMF.

        Returns:
            Array of probabilities: w[s] = P(serial interval = s days).
            w[0] = 0 by convention (no same-day transmission).
        """
        if self.mean <= 0 or self.std <= 0:
            raise ValueError(
                f"Mean and std must be positive: mean={self.mean}, std={self.std}"
            )

        # Gamma shape and scale from mean and std
        # mean = shape * scale, var = shape * scale^2
        # → shape = (mean/std)^2, scale = std^2/mean
        shape = (self.mean / self.std) ** 2
        scale = self.std ** 2 / self.mean

        # Evaluate Gamma PDF at integer day values
        days = np.arange(0, max_days + 1)
        pmf = gamma_dist.pdf(days, a=shape, scale=scale)

        # Zero out day 0 (no same-day transmission)
        pmf[0] = 0.0

        # Normalize to ensure PMF sums to 1
        total = pmf.sum()
        if total > 0:
            pmf /= total
        else:
            # Fallback: uniform distribution if parameterization fails
            pmf[1:] = 1.0 / max_days

        return pmf


# ---------------------------------------------------------------------------
# Preset serial interval distributions
# ---------------------------------------------------------------------------

SI_INFLUENZA = SerialInterval(mean=2.6, std=1.5)
"""Influenza serial interval. Ref: Ferguson et al. (2005)"""

SI_COVID = SerialInterval(mean=4.7, std=2.9)
"""COVID-19 serial interval. Ref: Nishiura et al. (2020)"""

SI_MEASLES = SerialInterval(mean=11.5, std=2.0)
"""Measles serial interval. Ref: Fine (2003)"""


@dataclass
class RtResult:
    """Result of Bayesian Rt estimation.

    Attributes:
        t: Time indices (aligned with input incidence array).
        rt_mean: Posterior mean of Rt at each time step.
        rt_lower: Lower bound of 95% credible interval.
        rt_upper: Upper bound of 95% credible interval.
        epidemic_phase: Classification at each step: 'growing', 'declining', 'stable'.
        posterior_shape: Posterior Gamma shape parameter at each step.
        posterior_scale: Posterior Gamma scale parameter at each step.
    """
    t: np.ndarray
    rt_mean: np.ndarray
    rt_lower: np.ndarray
    rt_upper: np.ndarray
    epidemic_phase: list[str]
    posterior_shape: np.ndarray
    posterior_scale: np.ndarray

    @property
    def current_rt(self) -> float:
        """Most recent Rt estimate (last non-NaN value)."""
        valid = ~np.isnan(self.rt_mean)
        if valid.any():
            return float(self.rt_mean[valid][-1])
        return float("nan")

    @property
    def current_phase(self) -> str:
        """Most recent epidemic phase classification."""
        for phase in reversed(self.epidemic_phase):
            if phase in ("growing", "declining", "stable"):
                return phase
        return "unknown"

    def summary(self) -> dict:
        """Return concise summary dict for agent state."""
        return {
            "current_rt": round(self.current_rt, 3),
            "current_phase": self.current_phase,
            "rt_range": [
                round(float(np.nanmin(self.rt_mean)), 3),
                round(float(np.nanmax(self.rt_mean)), 3),
            ],
            "n_valid_estimates": int(np.sum(~np.isnan(self.rt_mean))),
        }


def estimate_rt(
    incidence: np.ndarray,
    serial_interval: SerialInterval,
    window: int = 7,
    prior_shape: float = 1.0,
    prior_scale: float = 5.0,
    confidence: float = 0.95,
) -> RtResult:
    """Estimate time-varying Rt using the Cori et al. (2013) method.

    Args:
        incidence: Array of daily case counts (non-negative integers).
        serial_interval: SerialInterval distribution for the pathogen.
        window: Sliding window size (τ) in days for smoothing.
            Larger windows → smoother but more lagged estimates.
            Typical values: 7 (default), 14 for noisy data.
        prior_shape: Gamma prior shape parameter (a). Default 1.0
            gives a weakly informative prior.
        prior_scale: Gamma prior scale parameter (b). Default 5.0
            gives prior mean = a*b = 5.0 (broad uncertainty).
        confidence: Confidence level for credible interval (default 0.95).

    Returns:
        RtResult with time-varying Rt estimates and credible intervals.

    Raises:
        ValueError: If incidence contains negative values or is too short.
    """
    incidence = np.asarray(incidence, dtype=float)
    T = len(incidence)

    if T < window + 1:
        raise ValueError(
            f"Incidence series too short: need at least {window + 1} days, "
            f"got {T}"
        )

    if np.any(incidence < 0):
        raise ValueError("Incidence values must be non-negative")

    # Discretize serial interval
    si = serial_interval.discretize(max_days=min(T, 20))

    # -----------------------------------------------------------------------
    # Compute total infectiousness Λ_t for each day
    # Λ_t = Σ_{s=1}^{T} I_{t-s} · w_s
    # -----------------------------------------------------------------------
    lambda_t = np.zeros(T)
    for t in range(1, T):
        for s in range(1, min(t + 1, len(si))):
            lambda_t[t] += incidence[t - s] * si[s]

    # -----------------------------------------------------------------------
    # Estimate Rt for each sliding window [t-τ+1, t]
    # -----------------------------------------------------------------------
    alpha_half = (1.0 - confidence) / 2.0

    rt_mean = np.full(T, np.nan)
    rt_lower = np.full(T, np.nan)
    rt_upper = np.full(T, np.nan)
    post_shape = np.full(T, np.nan)
    post_scale = np.full(T, np.nan)

    for t in range(window, T):
        t_start = t - window + 1
        t_end = t + 1

        # Sum of cases in window
        sum_I = np.sum(incidence[t_start:t_end])
        # Sum of total infectiousness in window
        sum_lambda = np.sum(lambda_t[t_start:t_end])

        if sum_lambda == 0:
            # Cannot estimate Rt if no prior infectiousness
            continue

        # Posterior Gamma parameters (conjugate update)
        a_post = prior_shape + sum_I
        b_post = 1.0 / (1.0 / prior_scale + sum_lambda)

        # Store posterior parameters
        post_shape[t] = a_post
        post_scale[t] = b_post

        # Posterior mean
        rt_mean[t] = a_post * b_post

        # Credible interval from Gamma quantiles
        rt_lower[t] = gamma_dist.ppf(alpha_half, a=a_post, scale=b_post)
        rt_upper[t] = gamma_dist.ppf(1.0 - alpha_half, a=a_post, scale=b_post)

    # -----------------------------------------------------------------------
    # Classify epidemic phase at each time step
    # -----------------------------------------------------------------------
    epidemic_phase = []
    for t in range(T):
        if np.isnan(rt_mean[t]):
            epidemic_phase.append("unknown")
        elif rt_lower[t] > 1.0:
            # Entire CI above 1 → confidently growing
            epidemic_phase.append("growing")
        elif rt_upper[t] < 1.0:
            # Entire CI below 1 → confidently declining
            epidemic_phase.append("declining")
        else:
            # CI straddles 1 → uncertain, classify as stable
            epidemic_phase.append("stable")

    result = RtResult(
        t=np.arange(T),
        rt_mean=rt_mean,
        rt_lower=rt_lower,
        rt_upper=rt_upper,
        epidemic_phase=epidemic_phase,
        posterior_shape=post_shape,
        posterior_scale=post_scale,
    )

    logger.info(
        "Rt estimation complete: T=%d, current Rt=%.2f (%s), "
        "%d valid estimates",
        T, result.current_rt, result.current_phase,
        int(np.sum(~np.isnan(rt_mean))),
    )

    return result
