"""Multi-Pathogen Scenario Testing.

Runs the full EpiAgent pipeline across three pathogen profiles
to demonstrate the system's generalizability:
    1. COVID-19 (R0=2.5, SI=4.7d)
    2. Influenza (R0=1.3, SI=2.6d)
    3. Measles (R0=12.0, SI=11.5d)

Generates per-pathogen dashboards and a comparative summary.
"""

import json
import sys
import time
import io
from pathlib import Path

import numpy as np

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

from epiagent.agents.tools import (
    fetch_synthetic_data,
    validate_data,
    run_security_audit,
    run_seir_model,
    estimate_rt,
    compute_epi_metrics,
    detect_changepoints,
    run_ml_forecast,
    run_shap_analysis,
)
from epiagent.dashboard.generator import generate_dashboard


SCENARIOS = [
    {
        "pathogen": "covid-19",
        "display_name": "COVID-19",
        "R0": 2.5,
        "latent_period": 5.2,
        "infectious_period": 2.9,
        "duration": 180,
    },
    {
        "pathogen": "influenza",
        "display_name": "Influenza",
        "R0": 1.3,
        "latent_period": 2.0,
        "infectious_period": 3.0,
        "duration": 120,
    },
    {
        "pathogen": "measles",
        "display_name": "Measles",
        "R0": 12.0,
        "latent_period": 10.0,
        "infectious_period": 8.0,
        "duration": 180,
    },
]


def run_scenario(scenario: dict) -> dict:
    """Run full pipeline for one pathogen scenario."""
    name = scenario["display_name"]
    print(f"\n{'='*60}")
    print(f"  SCENARIO: {name} (R0={scenario['R0']})")
    print(f"{'='*60}")

    t0 = time.time()

    # Step 1: Data
    data = fetch_synthetic_data(
        pathogen=scenario["pathogen"],
        duration_days=scenario["duration"],
        region=f"{name.lower()}_region",
    )
    records = data["records"]
    case_series = [r["new_cases"] for r in records]
    dates = [r["date"] for r in records]
    case_json = json.dumps(case_series)

    print(f"  Data: {data['record_count']} records, {data['total_cases']:,} cases")

    # Step 2: Security
    security = run_security_audit(records)
    print(f"  Security: {'CLEAN' if not security['pii_detected'] else 'PII FOUND'}")

    # Step 3: Validation
    quality = validate_data(records)
    print(f"  Quality: {quality['quality_score']:.0%}")

    # Step 4: Analysis
    seir = run_seir_model(
        population=1_000_000,
        R0=scenario["R0"],
        latent_period=scenario["latent_period"],
        infectious_period=scenario["infectious_period"],
        initial_infected=10,
        t_max=scenario["duration"],
    )

    rt = estimate_rt(case_json, pathogen=scenario["pathogen"], window=7)

    total_cases = sum(case_series)
    total_deaths = sum(r["new_deaths"] for r in records)
    metrics = compute_epi_metrics(
        total_cases=total_cases,
        total_deaths=total_deaths,
        population=1_000_000,
        case_series_json=case_json,
    )

    cp = detect_changepoints(case_json)

    print(f"  Rt: {rt.get('current_rt', 'N/A'):.3f}" if isinstance(rt.get('current_rt'), float) else f"  Rt: {rt.get('current_rt', 'N/A')}")
    print(f"  Phase: {rt.get('current_phase', 'N/A')}")

    # Step 5: Forecast
    forecast = run_ml_forecast(case_json, horizon=14, dates_json=json.dumps(dates))
    shap = run_shap_analysis(case_json, top_k=5)

    if "error" not in forecast:
        ensemble = forecast["ensemble"]
        print(f"  Forecast RMSE: {ensemble.get('rmse', 'N/A'):.2f}" if ensemble.get('rmse') else "  Forecast: done")

    # Step 6: Dashboard
    dashboard_path = generate_dashboard(
        surveillance_data=data,
        rt_results=rt if "error" not in rt else None,
        seir_results=seir,
        forecast_results=forecast if "error" not in forecast else None,
        shap_results=shap if "error" not in shap else None,
        changepoint_results=cp if "error" not in cp else None,
        epi_metrics=metrics,
        security_report=security,
        output_path=str(project_root / f"dashboard_{scenario['pathogen'].replace('-','_')}.html"),
    )

    elapsed = time.time() - t0
    print(f"  Dashboard: {dashboard_path}")
    print(f"  Time: {elapsed:.1f}s")

    return {
        "pathogen": name,
        "R0": scenario["R0"],
        "total_cases": total_cases,
        "total_deaths": total_deaths,
        "cfr": metrics.get("cfr", {}).get("value", float("nan")),
        "current_rt": rt.get("current_rt", float("nan")),
        "phase": rt.get("current_phase", "unknown"),
        "quality_score": quality["quality_score"],
        "forecast_rmse": forecast.get("ensemble", {}).get("rmse") if "error" not in forecast else None,
        "dashboard": dashboard_path,
        "elapsed": elapsed,
    }


def main():
    print("=" * 60)
    print("  EpiAgent Multi-Pathogen Scenario Testing")
    print("=" * 60)

    results = []
    for scenario in SCENARIOS:
        result = run_scenario(scenario)
        results.append(result)

    # Comparative summary
    print("\n" + "=" * 60)
    print("  COMPARATIVE SUMMARY")
    print("=" * 60)
    print(f"\n  {'Pathogen':<12} {'R0':>5} {'Cases':>12} {'Deaths':>8} {'CFR':>8} {'Rt':>6} {'Phase':<10} {'Time':>5}")
    print("  " + "-" * 75)
    for r in results:
        cfr_str = f"{r['cfr']:.4f}" if not np.isnan(r['cfr']) else "N/A"
        rt_str = f"{r['current_rt']:.3f}" if isinstance(r['current_rt'], float) and not np.isnan(r['current_rt']) else "N/A"
        print(f"  {r['pathogen']:<12} {r['R0']:>5.1f} {r['total_cases']:>12,} {r['total_deaths']:>8,} {cfr_str:>8} {rt_str:>6} {r['phase']:<10} {r['elapsed']:>4.1f}s")

    total_time = sum(r["elapsed"] for r in results)
    print(f"\n  Total time: {total_time:.1f}s for {len(results)} scenarios")
    print(f"  Dashboards generated in project root directory")

    return results


if __name__ == "__main__":
    main()
