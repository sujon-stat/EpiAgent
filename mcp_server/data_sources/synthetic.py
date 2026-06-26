"""Synthetic Epidemic Data Generator.

Generates realistic synthetic surveillance data using SEIR dynamics
with configurable noise for three pathogen profiles:
    - Influenza-like: R0≈1.3, serial interval 2.6 days, CFR≈0.1%
    - COVID-like: R0≈2.5, serial interval 4.7 days, CFR≈1.5%
    - Measles-like: R0≈12, serial interval 11.5 days, CFR≈0.2%

The generator produces daily surveillance records with:
    - Realistic case counts from SEIR dynamics + Poisson noise
    - Deaths derived from cases with appropriate CFR + binomial noise
    - Configurable reporting delays and weekend effects
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date, timedelta

import numpy as np
from scipy.integrate import solve_ivp

logger = logging.getLogger(__name__)


@dataclass
class PathogenProfile:
    """Configuration for a pathogen's epidemiological characteristics."""
    name: str
    R0: float
    latent_period: float      # days
    infectious_period: float  # days
    cfr: float                # case fatality rate (0-1)
    population: int = 1_000_000


# ---------------------------------------------------------------------------
# Preset pathogen profiles
# ---------------------------------------------------------------------------

INFLUENZA = PathogenProfile(
    name="influenza",
    R0=1.3,
    latent_period=2.0,
    infectious_period=3.0,
    cfr=0.001,
    population=1_000_000,
)

COVID = PathogenProfile(
    name="covid-19",
    R0=2.5,
    latent_period=5.2,
    infectious_period=2.9,
    cfr=0.015,
    population=1_000_000,
)

MEASLES = PathogenProfile(
    name="measles",
    R0=12.0,
    latent_period=10.0,
    infectious_period=8.0,
    cfr=0.002,
    population=1_000_000,
)


@dataclass
class SurveillanceRecord:
    """A single daily surveillance observation."""
    date: str
    region: str
    pathogen: str
    new_cases: int
    cumulative_cases: int
    new_deaths: int
    cumulative_deaths: int
    population: int
    source: str = "synthetic"

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "date": self.date,
            "region": self.region,
            "pathogen": self.pathogen,
            "new_cases": self.new_cases,
            "cumulative_cases": self.cumulative_cases,
            "new_deaths": self.new_deaths,
            "cumulative_deaths": self.cumulative_deaths,
            "population": self.population,
            "source": self.source,
            "metadata": {},
        }


def _run_seir_internal(
    N: int,
    beta: float,
    sigma: float,
    gamma: float,
    I0: float,
    E0: float,
    t_max: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Run SEIR model and return daily incidence.

    Self-contained ODE solver (doesn't import from engines to keep
    MCP server independent for deployment).

    Returns:
        Tuple of (time_array, daily_incidence_array).
    """
    S0 = N - E0 - I0

    def odes(t, y):
        S, E, I, R = y
        dS = -beta * S * I / N
        dE = beta * S * I / N - sigma * E
        dI = sigma * E - gamma * I
        dR = gamma * I
        return [dS, dE, dI, dR]

    t_eval = np.arange(t_max + 1, dtype=float)
    sol = solve_ivp(
        odes,
        t_span=(0, t_max),
        y0=[S0, E0, I0, 0.0],
        t_eval=t_eval,
        method="RK45",
        rtol=1e-8,
        atol=1e-8,
    )

    S = sol.y[0]
    I = sol.y[2]

    # Daily incidence = β·S·I/N
    incidence = beta * S * I / N
    return sol.t, np.maximum(incidence, 0.0)


def generate_epidemic(
    profile: PathogenProfile,
    duration_days: int = 180,
    initial_infected: int = 10,
    noise_level: float = 0.1,
    seed: int = 42,
    region: str = "synthetic_region",
    start_date: str | None = None,
) -> list[SurveillanceRecord]:
    """Generate synthetic epidemic surveillance data.

    Args:
        profile: PathogenProfile defining the disease characteristics.
        duration_days: Number of days to simulate.
        initial_infected: Number of initially infectious individuals.
        noise_level: Poisson noise multiplier (0 = deterministic).
        seed: Random seed for reproducibility.
        region: Region name for records.
        start_date: ISO date string for day 0 (defaults to today - duration).

    Returns:
        List of SurveillanceRecord objects.
    """
    rng = np.random.default_rng(seed)

    # SEIR parameters
    gamma = 1.0 / profile.infectious_period
    sigma = 1.0 / profile.latent_period
    beta = profile.R0 * gamma

    # Run deterministic SEIR
    t, true_incidence = _run_seir_internal(
        N=profile.population,
        beta=beta,
        sigma=sigma,
        gamma=gamma,
        I0=float(initial_infected),
        E0=0.0,
        t_max=duration_days,
    )

    # Start date
    if start_date:
        base_date = date.fromisoformat(start_date)
    else:
        base_date = date.today() - timedelta(days=duration_days)

    # Generate noisy observations
    records = []
    cumulative_cases = 0
    cumulative_deaths = 0

    for day_idx in range(len(true_incidence)):
        current_date = base_date + timedelta(days=day_idx)

        # Add Poisson noise to incidence
        expected = true_incidence[day_idx]
        if noise_level > 0 and expected > 0:
            noisy = rng.poisson(max(0, expected * (1.0 + noise_level * rng.standard_normal())))
        else:
            noisy = int(round(expected))
        new_cases = max(0, int(noisy))

        # Weekend reporting effect: reduce by 30%, redistribute to Monday
        weekday = current_date.weekday()
        if weekday in (5, 6):  # Saturday, Sunday
            weekend_reduction = int(new_cases * 0.3)
            new_cases -= weekend_reduction
        elif weekday == 0:  # Monday
            # Add back some weekend cases (approximation)
            new_cases = int(new_cases * 1.4)

        # Derive deaths from cases using binomial sampling
        if new_cases > 0 and profile.cfr > 0:
            new_deaths = int(rng.binomial(new_cases, profile.cfr))
        else:
            new_deaths = 0

        cumulative_cases += new_cases
        cumulative_deaths += new_deaths

        record = SurveillanceRecord(
            date=current_date.isoformat(),
            region=region,
            pathogen=profile.name,
            new_cases=new_cases,
            cumulative_cases=cumulative_cases,
            new_deaths=new_deaths,
            cumulative_deaths=cumulative_deaths,
            population=profile.population,
            source="synthetic",
        )
        records.append(record)

    logger.info(
        "Generated %d-day %s epidemic: total cases=%d, total deaths=%d",
        duration_days, profile.name, cumulative_cases, cumulative_deaths,
    )
    return records


def generate_multi_wave(
    profile: PathogenProfile,
    num_waves: int = 2,
    duration_days: int = 365,
    wave_gap_days: int = 60,
    r0_decay: float = 0.7,
    seed: int = 42,
    region: str = "synthetic_region",
) -> list[SurveillanceRecord]:
    """Generate multi-wave epidemic data.

    Each subsequent wave has a reduced R0 (simulating immunity/vaccination).

    Args:
        profile: Base PathogenProfile.
        num_waves: Number of epidemic waves.
        duration_days: Total duration.
        wave_gap_days: Gap between wave starts.
        r0_decay: R0 multiplier for each subsequent wave.
        seed: Random seed.
        region: Region name.

    Returns:
        List of SurveillanceRecord objects.
    """
    rng = np.random.default_rng(seed)
    base_date = date.today() - timedelta(days=duration_days)

    # Generate each wave separately and combine
    all_incidence = np.zeros(duration_days + 1)

    for wave in range(num_waves):
        wave_r0 = profile.R0 * (r0_decay ** wave)
        gamma = 1.0 / profile.infectious_period
        sigma = 1.0 / profile.latent_period
        beta = wave_r0 * gamma

        wave_start = wave * wave_gap_days
        wave_duration = duration_days - wave_start

        if wave_duration <= 0:
            break

        _, wave_incidence = _run_seir_internal(
            N=profile.population,
            beta=beta,
            sigma=sigma,
            gamma=gamma,
            I0=10.0,
            E0=0.0,
            t_max=wave_duration,
        )

        # Add to total incidence
        for i, inc in enumerate(wave_incidence):
            if wave_start + i < len(all_incidence):
                all_incidence[wave_start + i] += inc

    # Generate records from combined incidence
    records = []
    cumulative_cases = 0
    cumulative_deaths = 0

    for day_idx in range(len(all_incidence)):
        current_date = base_date + timedelta(days=day_idx)
        expected = all_incidence[day_idx]

        new_cases = max(0, int(rng.poisson(max(0, expected))))

        if new_cases > 0 and profile.cfr > 0:
            new_deaths = int(rng.binomial(new_cases, profile.cfr))
        else:
            new_deaths = 0

        cumulative_cases += new_cases
        cumulative_deaths += new_deaths

        records.append(SurveillanceRecord(
            date=current_date.isoformat(),
            region=region,
            pathogen=profile.name,
            new_cases=new_cases,
            cumulative_cases=cumulative_cases,
            new_deaths=new_deaths,
            cumulative_deaths=cumulative_deaths,
            population=profile.population,
            source="synthetic",
        ))

    logger.info(
        "Generated %d-wave %s epidemic over %d days: "
        "total cases=%d, total deaths=%d",
        num_waves, profile.name, duration_days,
        cumulative_cases, cumulative_deaths,
    )
    return records
