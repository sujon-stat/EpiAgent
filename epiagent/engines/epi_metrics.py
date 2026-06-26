"""Deterministic Epidemiological Metrics Calculator.

Computes standard surveillance metrics with proper confidence intervals.
All computations are deterministic — no LLM involvement.

Metrics:
    - Case Fatality Rate (CFR) with Wilson score interval
    - Incidence Rate with exact Poisson confidence interval
    - Doubling Time via log-linear regression with delta method CI
    - Attack Rate with Wilson score interval
    - Exponential Growth Rate via log-linear regression

References:
    Wilson EB. (1927) "Probable Inference, the Law of Succession, and
    Statistical Inference." JASA, 22(158):209-212.

    Agresti A, Coull BA. (1998) "Approximate is Better than 'Exact' for
    Interval Estimation of Binomial Proportions." The American Statistician,
    52(2):119-126.
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass

import numpy as np
from scipy import stats

logger = logging.getLogger(__name__)


@dataclass
class MetricResult:
    """A single epidemiological metric with confidence interval.

    Attributes:
        value: Point estimate.
        lower_ci: Lower bound of confidence interval.
        upper_ci: Upper bound of confidence interval.
        method: Description of the CI method used.
    """
    value: float
    lower_ci: float
    upper_ci: float
    method: str

    def summary(self) -> dict:
        """Return concise summary dict."""
        return {
            "value": round(self.value, 6),
            "ci": [round(self.lower_ci, 6), round(self.upper_ci, 6)],
            "method": self.method,
        }


# ---------------------------------------------------------------------------
# Wilson Score Interval (helper)
# ---------------------------------------------------------------------------

def _wilson_score_interval(
    successes: int,
    trials: int,
    confidence: float = 0.95,
) -> tuple[float, float, float]:
    """Compute Wilson score interval for a binomial proportion.

    The Wilson score interval has better coverage properties than the
    Wald interval, especially for small n or extreme p.

    Formula:
        CI = (p̂ + z²/2n ± z·√(p̂(1-p̂)/n + z²/4n²)) / (1 + z²/n)

    Args:
        successes: Number of successes (e.g., deaths).
        trials: Number of trials (e.g., cases).
        confidence: Confidence level (default 0.95).

    Returns:
        Tuple of (point_estimate, lower_ci, upper_ci).
    """
    if trials <= 0:
        return (float("nan"), float("nan"), float("nan"))

    p_hat = successes / trials
    z = stats.norm.ppf(1 - (1 - confidence) / 2)
    z2 = z ** 2

    denominator = 1 + z2 / trials
    center = p_hat + z2 / (2 * trials)
    margin = z * np.sqrt(p_hat * (1 - p_hat) / trials + z2 / (4 * trials ** 2))

    lower = (center - margin) / denominator
    upper = (center + margin) / denominator

    # Clamp to [0, 1]
    lower = max(0.0, lower)
    upper = min(1.0, upper)

    return (p_hat, lower, upper)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_cfr(
    deaths: int,
    cases: int,
    confidence: float = 0.95,
) -> MetricResult:
    """Compute Case Fatality Rate with Wilson score confidence interval.

    CFR = deaths / cases

    Uses the Wilson score interval which has proper coverage even for
    small sample sizes and extreme proportions — a significant improvement
    over the Wald interval (p̂ ± z√(p̂(1-p̂)/n)) which fails at boundaries.

    Args:
        deaths: Number of deaths.
        cases: Number of cases (denominator).
        confidence: Confidence level (default 0.95).

    Returns:
        MetricResult with CFR and Wilson score CI.

    Raises:
        ValueError: If deaths or cases are negative.
    """
    if deaths < 0 or cases < 0:
        raise ValueError(f"Counts must be non-negative: deaths={deaths}, cases={cases}")

    if cases == 0:
        logger.warning("CFR undefined: 0 cases. Returning NaN.")
        return MetricResult(
            value=float("nan"),
            lower_ci=float("nan"),
            upper_ci=float("nan"),
            method="Wilson score (undefined: 0 cases)",
        )

    if deaths > cases:
        logger.warning(
            "Deaths (%d) exceed cases (%d). CFR will exceed 1.0. "
            "This may indicate reporting lag or data error.",
            deaths, cases,
        )

    cfr, lower, upper = _wilson_score_interval(deaths, cases, confidence)

    return MetricResult(
        value=cfr,
        lower_ci=lower,
        upper_ci=upper,
        method=f"Wilson score interval ({confidence:.0%} CI)",
    )


def compute_incidence_rate(
    cases: int,
    population: int,
    per: int = 100_000,
    confidence: float = 0.95,
) -> MetricResult:
    """Compute incidence rate with exact Poisson confidence interval.

    Rate = (cases / population) × per

    Uses the exact Poisson CI based on the chi-squared distribution:
        Lower = χ²(α/2, 2k) / (2n)
        Upper = χ²(1-α/2, 2k+2) / (2n)
    where k = cases and n = population/per.

    Args:
        cases: Number of new cases.
        population: Population at risk.
        per: Rate multiplier (default 100,000 for "per 100k").
        confidence: Confidence level (default 0.95).

    Returns:
        MetricResult with incidence rate and exact Poisson CI.

    Raises:
        ValueError: If cases is negative or population is non-positive.
    """
    if cases < 0:
        raise ValueError(f"Cases must be non-negative: {cases}")
    if population <= 0:
        raise ValueError(f"Population must be positive: {population}")

    rate = (cases / population) * per
    alpha = 1 - confidence

    if cases == 0:
        lower = 0.0
        upper = (stats.chi2.ppf(1 - alpha / 2, 2) / (2 * population)) * per
    else:
        lower = (stats.chi2.ppf(alpha / 2, 2 * cases) / (2 * population)) * per
        upper = (stats.chi2.ppf(1 - alpha / 2, 2 * (cases + 1)) / (2 * population)) * per

    return MetricResult(
        value=rate,
        lower_ci=lower,
        upper_ci=upper,
        method=f"Exact Poisson CI ({confidence:.0%}), per {per:,}",
    )


def compute_doubling_time(
    case_series: np.ndarray,
    window: int = 7,
    confidence: float = 0.95,
) -> MetricResult:
    """Compute epidemic doubling time via log-linear regression.

    Fits log(cases) = a + r·t to the most recent `window` days,
    then Td = ln(2) / r.

    The confidence interval uses the delta method:
        SE(Td) = ln(2) / r² · SE(r)

    Args:
        case_series: Array of daily case counts.
        window: Number of recent days to use for fitting (default 7).
        confidence: Confidence level (default 0.95).

    Returns:
        MetricResult with doubling time in days.
        Returns NaN if growth rate is non-positive (epidemic declining).
    """
    series = np.asarray(case_series, dtype=float)

    if len(series) < window:
        return MetricResult(
            value=float("nan"),
            lower_ci=float("nan"),
            upper_ci=float("nan"),
            method=f"Log-linear regression (insufficient data: {len(series)} < {window})",
        )

    # Use last `window` days
    recent = series[-window:]

    # Filter out zeros and negative values for log transform
    valid_mask = recent > 0
    if valid_mask.sum() < 3:
        return MetricResult(
            value=float("nan"),
            lower_ci=float("nan"),
            upper_ci=float("nan"),
            method="Log-linear regression (insufficient positive values)",
        )

    t = np.arange(window)[valid_mask]
    y = np.log(recent[valid_mask])

    # Linear regression: log(cases) = intercept + r * t
    result = stats.linregress(t, y)
    r = result.slope
    se_r = result.stderr

    if r <= 0:
        # Epidemic is declining — doubling time is not meaningful
        return MetricResult(
            value=float("nan"),
            lower_ci=float("nan"),
            upper_ci=float("nan"),
            method="Log-linear regression (growth rate ≤ 0, epidemic declining)",
        )

    # Doubling time
    Td = np.log(2) / r

    # Delta method CI for Td = ln(2)/r
    # SE(Td) = ln(2) / r² · SE(r)
    z = stats.norm.ppf(1 - (1 - confidence) / 2)
    se_Td = np.log(2) / (r ** 2) * se_r
    lower = max(0.0, Td - z * se_Td)
    upper = Td + z * se_Td

    return MetricResult(
        value=Td,
        lower_ci=lower,
        upper_ci=upper,
        method=f"Log-linear regression, delta method ({confidence:.0%} CI), "
               f"window={window} days, r={r:.4f}",
    )


def compute_attack_rate(
    total_cases: int,
    population: int,
    confidence: float = 0.95,
) -> MetricResult:
    """Compute attack rate (cumulative incidence proportion) with Wilson CI.

    Attack Rate = total_cases / population

    Args:
        total_cases: Total cumulative cases.
        population: Population at risk.
        confidence: Confidence level (default 0.95).

    Returns:
        MetricResult with attack rate and Wilson score CI.

    Raises:
        ValueError: If inputs are invalid.
    """
    if total_cases < 0:
        raise ValueError(f"Total cases must be non-negative: {total_cases}")
    if population <= 0:
        raise ValueError(f"Population must be positive: {population}")

    ar, lower, upper = _wilson_score_interval(total_cases, population, confidence)

    return MetricResult(
        value=ar,
        lower_ci=lower,
        upper_ci=upper,
        method=f"Wilson score interval ({confidence:.0%} CI)",
    )


def compute_growth_rate(
    case_series: np.ndarray,
    window: int = 7,
    confidence: float = 0.95,
) -> MetricResult:
    """Compute exponential growth rate via log-linear regression.

    Fits log(cases) = a + r·t to the most recent `window` days.
    Positive r → exponential growth; negative r → exponential decay.

    Args:
        case_series: Array of daily case counts.
        window: Number of recent days to use (default 7).
        confidence: Confidence level (default 0.95).

    Returns:
        MetricResult with growth rate r (per day).
    """
    series = np.asarray(case_series, dtype=float)

    if len(series) < window:
        return MetricResult(
            value=float("nan"),
            lower_ci=float("nan"),
            upper_ci=float("nan"),
            method=f"Log-linear regression (insufficient data: {len(series)} < {window})",
        )

    recent = series[-window:]

    valid_mask = recent > 0
    if valid_mask.sum() < 3:
        return MetricResult(
            value=float("nan"),
            lower_ci=float("nan"),
            upper_ci=float("nan"),
            method="Log-linear regression (insufficient positive values)",
        )

    t = np.arange(window)[valid_mask]
    y = np.log(recent[valid_mask])

    result = stats.linregress(t, y)
    r = result.slope
    se_r = result.stderr

    z = stats.norm.ppf(1 - (1 - confidence) / 2)
    lower = r - z * se_r
    upper = r + z * se_r

    return MetricResult(
        value=r,
        lower_ci=lower,
        upper_ci=upper,
        method=f"Log-linear regression ({confidence:.0%} CI), window={window} days",
    )
