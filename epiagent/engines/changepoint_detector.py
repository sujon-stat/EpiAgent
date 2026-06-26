"""Bayesian Online Changepoint Detection (BOCPD).

Implements the Adams & MacKay (2007) algorithm for detecting structural
breaks in epidemic time series — outbreak onset, peak, decline, and
resurgence.

Reference:
    Adams RP, MacKay DJC. (2007)
    "Bayesian Online Changepoint Detection."
    arXiv:0710.3742

Method:
    Maintains a posterior distribution over the "run length" r_t
    (number of time steps since the last changepoint). At each
    new observation x_t:
    
    1. Compute predictive probability P(x_t | r_{t-1}) using a
       conjugate Normal-Inverse-Gamma model
    2. Update growth probabilities (r_t = r_{t-1} + 1)
    3. Update changepoint probability (r_t = 0)
    4. Normalize to get posterior P(r_t | x_{1:t})
    5. P(changepoint at t) = P(r_t = 0 | x_{1:t})

    Uses a constant hazard function H = 1/λ where λ is the
    expected run length between changepoints.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
from scipy.stats import t as student_t

logger = logging.getLogger(__name__)


@dataclass
class ChangePointResult:
    """Result of Bayesian online changepoint detection.

    Attributes:
        changepoint_probs: Array of changepoint probability at each time step.
        changepoints: Indices where changepoint probability exceeds threshold.
        run_length_posterior: Full run length posterior matrix R[t, r].
        threshold: Threshold used for changepoint identification.
    """
    changepoint_probs: np.ndarray
    changepoints: list[int]
    run_length_posterior: np.ndarray
    threshold: float

    def summary(self) -> dict:
        """Return concise summary for agent state."""
        return {
            "n_changepoints": len(self.changepoints),
            "changepoint_indices": self.changepoints,
            "max_changepoint_prob": float(np.nanmax(self.changepoint_probs))
            if len(self.changepoint_probs) > 0
            else 0.0,
            "threshold": self.threshold,
        }


class BOCPDetector:
    """Bayesian Online Changepoint Detector.

    Uses a Normal-Inverse-Gamma conjugate prior for the observation model,
    meaning it assumes observations are Gaussian with unknown mean and variance.

    For epidemic case counts, it's recommended to apply a log-transform
    or Box-Cox transform before running detection to better approximate
    normality.

    Args:
        hazard_lambda: Expected run length between changepoints.
            Smaller values make the detector more sensitive.
            Typical values: 50-250 for daily data.
        mu0: Prior mean of the Normal-Inverse-Gamma distribution.
        kappa0: Prior precision scaling (higher = more confident in mu0).
        alpha0: Prior shape of the Inverse-Gamma (must be > 0).
        beta0: Prior scale of the Inverse-Gamma (must be > 0).
    """

    def __init__(
        self,
        hazard_lambda: float = 100.0,
        mu0: float = 0.0,
        kappa0: float = 1.0,
        alpha0: float = 1.0,
        beta0: float = 1.0,
    ):
        if hazard_lambda <= 0:
            raise ValueError(f"hazard_lambda must be > 0, got {hazard_lambda}")
        if alpha0 <= 0 or beta0 <= 0:
            raise ValueError(f"alpha0 and beta0 must be > 0, got alpha0={alpha0}, beta0={beta0}")

        self.hazard = 1.0 / hazard_lambda
        self.mu0 = mu0
        self.kappa0 = kappa0
        self.alpha0 = alpha0
        self.beta0 = beta0

    def detect(
        self,
        data: np.ndarray,
        threshold: float = 0.5,
    ) -> ChangePointResult:
        """Run BOCPD on a 1-D time series.

        Args:
            data: 1-D array of observations (e.g., daily case counts,
                  ideally log-transformed).
            threshold: Probability threshold for declaring a changepoint.
                       Lower values = more sensitive detection.

        Returns:
            ChangePointResult with probabilities and detected changepoints.
        """
        data = np.asarray(data, dtype=float)
        T = len(data)

        if T == 0:
            return ChangePointResult(
                changepoint_probs=np.array([]),
                changepoints=[],
                run_length_posterior=np.zeros((1, 1)),
                threshold=threshold,
            )

        # Run length posterior: R[t, r] = P(r_t = r | x_{1:t})
        # We only need the current and previous time step for memory efficiency,
        # but we store the full matrix for visualization purposes.
        R = np.zeros((T + 1, T + 1))
        R[0, 0] = 1.0  # Prior: run length 0 with probability 1

        # Sufficient statistics for each run length hypothesis
        # Normal-Inverse-Gamma: track (mu, kappa, alpha, beta)
        mu = np.array([self.mu0])
        kappa = np.array([self.kappa0])
        alpha = np.array([self.alpha0])
        beta = np.array([self.beta0])

        changepoint_probs = np.zeros(T)

        for t in range(T):
            x = data[t]

            # ---------------------------------------------------------------
            # 1. Predictive probability under each run length hypothesis
            #    Using Student-t distribution (conjugate predictive)
            # ---------------------------------------------------------------
            df = 2.0 * alpha
            loc = mu
            scale = np.sqrt(beta * (kappa + 1.0) / (alpha * kappa))

            # Clamp scale to avoid numerical issues
            scale = np.maximum(scale, 1e-10)

            pred_probs = student_t.pdf(x, df=df, loc=loc, scale=scale)

            # Clamp to avoid zero probabilities
            pred_probs = np.maximum(pred_probs, 1e-300)

            # ---------------------------------------------------------------
            # 2. Growth probabilities: r_t = r_{t-1} + 1
            # ---------------------------------------------------------------
            growth = R[t, : t + 1] * pred_probs * (1.0 - self.hazard)

            # ---------------------------------------------------------------
            # 3. Changepoint probability: r_t = 0
            # ---------------------------------------------------------------
            cp = np.sum(R[t, : t + 1] * pred_probs * self.hazard)

            # ---------------------------------------------------------------
            # 4. Update run length distribution
            # ---------------------------------------------------------------
            R[t + 1, 0] = cp
            R[t + 1, 1 : t + 2] = growth

            # ---------------------------------------------------------------
            # 5. Normalize
            # ---------------------------------------------------------------
            evidence = R[t + 1, : t + 2].sum()
            if evidence > 0:
                R[t + 1, : t + 2] /= evidence
            else:
                # Fallback: reset to uniform if numerical underflow
                R[t + 1, 0] = 1.0

            changepoint_probs[t] = R[t + 1, 0]

            # ---------------------------------------------------------------
            # 6. Update sufficient statistics for each run length
            # ---------------------------------------------------------------
            new_kappa = np.append(self.kappa0, kappa + 1.0)
            new_alpha = np.append(self.alpha0, alpha + 0.5)
            new_mu = np.append(
                self.mu0,
                (kappa * mu + x) / (kappa + 1.0),
            )
            new_beta = np.append(
                self.beta0,
                beta + kappa * (x - mu) ** 2 / (2.0 * (kappa + 1.0)),
            )

            mu, kappa, alpha, beta = new_mu, new_kappa, new_alpha, new_beta

        # ---------------------------------------------------------------
        # Identify changepoints above threshold
        # ---------------------------------------------------------------
        changepoints = list(np.where(changepoint_probs > threshold)[0])

        logger.info(
            "BOCPD complete: T=%d, detected %d changepoints (threshold=%.2f)",
            T, len(changepoints), threshold,
        )

        return ChangePointResult(
            changepoint_probs=changepoint_probs,
            changepoints=changepoints,
            run_length_posterior=R,
            threshold=threshold,
        )


def detect_outbreak_signals(
    case_series: np.ndarray,
    *,
    hazard_lambda: float = 100.0,
    threshold: float = 0.5,
    log_transform: bool = True,
) -> ChangePointResult:
    """Convenience function for outbreak signal detection in case counts.

    Applies optional log-transform to better approximate normality
    (case counts are typically right-skewed), then runs BOCPD.

    Args:
        case_series: Array of daily case counts.
        hazard_lambda: Expected run length between changepoints.
        threshold: Changepoint probability threshold.
        log_transform: If True, apply log(1 + x) transform before detection.

    Returns:
        ChangePointResult with detected outbreak signals.
    """
    case_series = np.asarray(case_series, dtype=float)

    if log_transform:
        # log(1 + x) handles zeros and makes distribution more Gaussian
        transformed = np.log1p(np.maximum(case_series, 0))
    else:
        transformed = case_series.copy()

    # Replace NaN/Inf with 0
    transformed = np.nan_to_num(transformed, nan=0.0, posinf=0.0, neginf=0.0)

    detector = BOCPDetector(hazard_lambda=hazard_lambda)
    return detector.detect(transformed, threshold=threshold)
