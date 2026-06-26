"""FunctionTool wrappers for EpiAgent computational engines.

Each function is designed to be invoked as a deterministic FunctionTool
by an LLM agent. The LLM agent decides WHEN to call these tools and
INTERPRETS the results — but the computation itself is fully deterministic.

This is the critical architectural pattern:
    LLM decides → FunctionTool computes → LLM interprets
    (No hallucinated math. Ever.)
"""

from __future__ import annotations

import json
import logging
import traceback

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data Retrieval Tools
# ---------------------------------------------------------------------------

def fetch_synthetic_data(
    pathogen: str = "covid-19",
    duration_days: int = 180,
    region: str = "synthetic_region",
    noise_level: float = 0.1,
    seed: int = 42,
) -> dict:
    """Fetch synthetic epidemic surveillance data.

    Generates realistic epidemic data using SEIR dynamics with
    configurable Poisson noise for testing and demonstration.

    Args:
        pathogen: One of 'influenza', 'covid-19', 'measles'.
        duration_days: Number of days to simulate.
        region: Region name for records.
        noise_level: Noise multiplier (0=deterministic, 0.1=default).
        seed: Random seed for reproducibility.

    Returns:
        Dict with 'records' (list of surveillance dicts), 'summary' stats.
    """
    from mcp_server.data_sources.synthetic import (
        INFLUENZA, COVID, MEASLES, generate_epidemic,
    )

    profiles = {"influenza": INFLUENZA, "covid-19": COVID, "measles": MEASLES}
    profile = profiles.get(pathogen.lower())
    if not profile:
        return {"error": f"Unknown pathogen '{pathogen}'. Options: {list(profiles.keys())}"}

    records = generate_epidemic(
        profile=profile,
        duration_days=duration_days,
        noise_level=noise_level,
        seed=seed,
        region=region,
    )

    record_dicts = [r.to_dict() for r in records]

    total_cases = sum(r.new_cases for r in records)
    total_deaths = sum(r.new_deaths for r in records)
    peak_day = max(range(len(records)), key=lambda i: records[i].new_cases)

    return {
        "record_count": len(record_dicts),
        "pathogen": pathogen,
        "region": region,
        "date_range": f"{records[0].date} to {records[-1].date}",
        "total_cases": total_cases,
        "total_deaths": total_deaths,
        "peak_day": records[peak_day].date,
        "peak_cases": records[peak_day].new_cases,
        "records": record_dicts,
    }


def fetch_cdc_data(
    regions: str = "nat",
    epiweeks: str = "202301-202352",
) -> dict:
    """Fetch real CDC FluView ILINet surveillance data.

    Retrieves weekly ILI data from the CMU Delphi Epidata API.

    Args:
        regions: Region codes: 'nat' for national, 'hhs1'-'hhs10'.
        epiweeks: Epiweek range (YYYYWW-YYYYWW).

    Returns:
        Dict with 'records' and summary stats.
    """
    from mcp_server.data_sources.cdc_fluview import fetch_fluview

    records = fetch_fluview(regions=regions, epiweeks=epiweeks)
    record_dicts = [r.to_dict() for r in records]

    if not records:
        return {"record_count": 0, "records": [], "warning": "No data returned from CDC API"}

    return {
        "record_count": len(record_dicts),
        "source": "cdc_fluview",
        "regions": regions,
        "records": record_dicts,
    }


# ---------------------------------------------------------------------------
# Validation Tools
# ---------------------------------------------------------------------------

def validate_data(records_json: str) -> dict:
    """Run epidemiological data quality validation.

    Applies 8-point plausibility checks on surveillance data.

    Args:
        records_json: JSON string of surveillance records list.

    Returns:
        Dict with quality_score, issues, and recommendations.
    """
    from epiagent.validators.epi_validator import validate_surveillance_data

    try:
        records = json.loads(records_json) if isinstance(records_json, str) else records_json
    except json.JSONDecodeError as e:
        return {"error": f"Invalid JSON: {e}"}

    report = validate_surveillance_data(records)

    return {
        "quality_score": report.quality_score,
        "is_acceptable": report.is_acceptable,
        "total_records": report.total_records,
        "valid_records": report.valid_records,
        "error_count": report.error_count,
        "warning_count": report.warning_count,
        "issues": [
            {
                "severity": i.severity,
                "field": i.field,
                "record_index": i.record_index,
                "message": i.message,
            }
            for i in report.issues[:20]  # Limit to 20 for LLM context
        ],
    }


# ---------------------------------------------------------------------------
# Security Tools
# ---------------------------------------------------------------------------

def run_security_audit(records_json: str) -> dict:
    """Run HIPAA-aligned security audit on surveillance data.

    Scans for all 18 HIPAA identifier types and validates schema.

    Args:
        records_json: JSON string of surveillance records list.

    Returns:
        Dict with PII findings, schema validation, and data hash.
    """
    from epiagent.guardrails.security import (
        strip_pii_from_records, create_security_report,
    )

    try:
        records = json.loads(records_json) if isinstance(records_json, str) else records_json
    except json.JSONDecodeError as e:
        return {"error": f"Invalid JSON: {e}"}

    # Audit
    report = create_security_report(records)

    # Strip PII if found
    cleaned_records = records
    if report.pii_detected:
        cleaned_records, strip_report = strip_pii_from_records(records)
        report = strip_report

    return {
        "pii_detected": report.pii_detected,
        "pii_types_found": report.pii_types_found,
        "items_redacted": report.items_redacted,
        "data_hash": report.data_hash,
        "schema_valid": report.schema_valid,
        "schema_errors": report.schema_errors[:10],
        "cleaned_records": cleaned_records,
    }


# ---------------------------------------------------------------------------
# Epidemiological Analysis Tools
# ---------------------------------------------------------------------------

def run_seir_model(
    population: int = 1_000_000,
    R0: float = 2.5,
    latent_period: float = 5.2,
    infectious_period: float = 2.9,
    initial_infected: int = 10,
    t_max: int = 365,
) -> dict:
    """Run SEIR compartmental model simulation.

    Solves the SEIR differential equations using RK45.

    Args:
        population: Total population size.
        R0: Basic reproduction number.
        latent_period: Average latent period (days).
        infectious_period: Average infectious period (days).
        initial_infected: Initial number of infectious individuals.
        t_max: Simulation duration (days).

    Returns:
        Dict with model outputs, peak timing, and attack rate.
    """
    from epiagent.engines.seir_model import SEIRParameters, run_seir

    params = SEIRParameters.from_epi_params(
        R0=R0,
        latent_period=latent_period,
        infectious_period=infectious_period,
        population=population,
        initial_infected=initial_infected,
    )
    result = run_seir(params, t_max=t_max)

    return {
        "summary": result.summary(),
        "daily_incidence": result.daily_incidence[:60].tolist(),  # First 60 days
        "peak_day": int(result.peak_day),
        "peak_cases": float(result.peak_cases),
        "total_infected": float(result.R[-1]),
        "attack_rate": float(result.R[-1] / params.N),
    }


def estimate_rt(
    incidence_json: str,
    pathogen: str = "covid-19",
    window: int = 7,
) -> dict:
    """Estimate time-varying effective reproduction number Rt.

    Uses the Cori et al. (2013) Bayesian method — the WHO/CDC standard.

    Args:
        incidence_json: JSON array of daily case counts.
        pathogen: Pathogen name to select serial interval.
        window: Sliding window size in days.

    Returns:
        Dict with Rt estimates, credible intervals, and phase classification.
    """
    from epiagent.engines.rt_estimation import (
        estimate_rt as _estimate_rt,
        SI_COVID, SI_INFLUENZA, SI_MEASLES,
    )

    try:
        incidence = np.array(json.loads(incidence_json), dtype=float)
    except (json.JSONDecodeError, TypeError) as e:
        return {"error": f"Invalid incidence data: {e}"}

    si_map = {
        "covid-19": SI_COVID, "covid": SI_COVID,
        "influenza": SI_INFLUENZA, "flu": SI_INFLUENZA,
        "measles": SI_MEASLES,
    }
    si = si_map.get(pathogen.lower(), SI_COVID)

    try:
        result = _estimate_rt(incidence, si, window=window)
    except ValueError as e:
        return {"error": str(e)}

    # Return last 30 days of Rt
    n = min(30, len(result.rt_mean))
    return {
        "current_rt": result.current_rt,
        "current_phase": result.current_phase,
        "rt_last_30d": {
            "rt_mean": [round(x, 3) if not np.isnan(x) else None for x in result.rt_mean[-n:]],
            "rt_lower": [round(x, 3) if not np.isnan(x) else None for x in result.rt_lower[-n:]],
            "rt_upper": [round(x, 3) if not np.isnan(x) else None for x in result.rt_upper[-n:]],
            "phase": result.epidemic_phase[-n:],
        },
        "summary": result.summary(),
    }


def compute_epi_metrics(
    total_cases: int,
    total_deaths: int,
    population: int,
    case_series_json: str = "[]",
) -> dict:
    """Compute standard epidemiological metrics with confidence intervals.

    Computes: CFR, incidence rate, attack rate, doubling time, growth rate.

    Args:
        total_cases: Total cumulative cases.
        total_deaths: Total cumulative deaths.
        population: Population at risk.
        case_series_json: JSON array of daily case counts (for time-based metrics).

    Returns:
        Dict with all metrics and their confidence intervals.
    """
    from epiagent.engines.epi_metrics import (
        compute_cfr, compute_incidence_rate, compute_attack_rate,
        compute_doubling_time, compute_growth_rate,
    )

    results = {}

    # CFR
    cfr = compute_cfr(total_deaths, total_cases)
    results["cfr"] = cfr.summary()

    # Incidence rate
    inc = compute_incidence_rate(total_cases, population)
    results["incidence_rate_per_100k"] = inc.summary()

    # Attack rate
    ar = compute_attack_rate(total_cases, population)
    results["attack_rate"] = ar.summary()

    # Time-based metrics (if series provided)
    try:
        case_series = np.array(json.loads(case_series_json), dtype=float)
        if len(case_series) >= 7:
            dt = compute_doubling_time(case_series)
            results["doubling_time_days"] = dt.summary()

            gr = compute_growth_rate(case_series)
            results["growth_rate_per_day"] = gr.summary()
    except (json.JSONDecodeError, TypeError):
        pass

    return results


def detect_changepoints(
    case_series_json: str,
    hazard_lambda: float = 100.0,
) -> dict:
    """Detect outbreak changepoints using Bayesian Online Changepoint Detection.

    Identifies abrupt changes in epidemic dynamics (e.g., outbreak onset,
    lockdown effects, new variant emergence).

    Args:
        case_series_json: JSON array of daily case counts.
        hazard_lambda: Expected run length (higher = fewer changepoints).

    Returns:
        Dict with detected changepoints and their confidence scores.
    """
    from epiagent.engines.changepoint_detector import (
        detect_outbreak_signals as _detect_cps,
    )

    try:
        case_series = np.array(json.loads(case_series_json), dtype=float)
    except (json.JSONDecodeError, TypeError) as e:
        return {"error": f"Invalid data: {e}"}

    try:
        result = _detect_cps(case_series, hazard_lambda=hazard_lambda)
        # Extract confidence scores from changepoint_probs at detected indices
        confidence_scores = [
            float(result.changepoint_probs[cp])
            for cp in result.changepoints
            if cp < len(result.changepoint_probs)
        ]
        return {
            "changepoints": result.changepoints,
            "confidence_scores": [round(s, 3) for s in confidence_scores],
            "n_changepoints": len(result.changepoints),
            "summary": result.summary(),
        }
    except Exception as e:
        return {"error": str(e)}


# ---------------------------------------------------------------------------
# ML Forecasting Tools
# ---------------------------------------------------------------------------

def run_ml_forecast(
    case_series_json: str,
    horizon: int = 14,
    dates_json: str = "[]",
) -> dict:
    """Run ML ensemble forecast (XGBoost + Prophet if available).

    Args:
        case_series_json: JSON array of daily case counts.
        horizon: Number of days to forecast ahead.
        dates_json: Optional JSON array of ISO date strings.

    Returns:
        Dict with ensemble predictions, individual model results, and feature importance.
    """
    from epiagent.engines.ml_forecaster import forecast_ensemble

    try:
        case_series = np.array(json.loads(case_series_json), dtype=float)
    except (json.JSONDecodeError, TypeError) as e:
        return {"error": f"Invalid data: {e}"}

    try:
        dates_list = json.loads(dates_json) if dates_json != "[]" else None
    except json.JSONDecodeError:
        dates_list = None

    try:
        ensemble, individual = forecast_ensemble(
            case_series, dates=dates_list, horizon=horizon,
        )
        return {
            "ensemble": ensemble.to_dict(),
            "individual_models": [fc.to_dict() for fc in individual],
            "model_count": len(individual),
        }
    except Exception as e:
        logger.error("Forecast failed: %s\n%s", e, traceback.format_exc())
        return {"error": str(e)}


def run_shap_analysis(
    case_series_json: str,
    top_k: int = 10,
) -> dict:
    """Run SHAP explainability analysis on epidemic forecast.

    Computes SHAP values to explain which features drive the forecast.

    Args:
        case_series_json: JSON array of daily case counts.
        top_k: Number of top features to return.

    Returns:
        Dict with top drivers, feature importance, and narrative.
    """
    from epiagent.engines.explainability import explain_xgboost_forecast

    try:
        case_series = np.array(json.loads(case_series_json), dtype=float)
    except (json.JSONDecodeError, TypeError) as e:
        return {"error": f"Invalid data: {e}"}

    try:
        explanation, metrics = explain_xgboost_forecast(
            case_series, top_k=top_k,
        )
        return {
            "top_drivers": explanation.top_drivers,
            "narrative": explanation.narrative(),
            "metrics": metrics,
            "global_importance": {
                k: round(v, 4)
                for k, v in list(explanation.global_importance.items())[:top_k]
            },
        }
    except Exception as e:
        logger.error("SHAP analysis failed: %s", e)
        return {"error": str(e)}
