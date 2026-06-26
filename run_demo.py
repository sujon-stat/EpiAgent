"""EpiAgent Full Pipeline Demo — No API Key Required.

Runs the complete EpiAgent analysis pipeline using direct function calls
(bypassing the LLM agents). This demonstrates all computational engines
and generates the interactive dashboard.

This script is perfect for:
    1. Kaggle competition submission (runs without API keys)
    2. Testing all engines end-to-end
    3. Generating the dashboard for screenshots/video
    4. Validating the full pipeline before ADK agent wiring
"""

import json
import logging
import sys
import time
from pathlib import Path
from datetime import datetime

import numpy as np

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("epiagent.demo")

# Fix Windows console encoding
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')


def print_banner(text: str, char: str = "="):
    """Print a styled banner."""
    width = 70
    border = char * width
    padding = (width - len(text) - 2) // 2
    print(f"\n{border}")
    print(f"{char}{' ' * padding}{text}{' ' * (width - padding - len(text) - 2)}{char}")
    print(f"{border}")


def print_metric(name: str, value: str, detail: str = ""):
    """Print a metric line."""
    print(f"  {'•'} {name:.<35} {value}")
    if detail:
        print(f"    {'└─'} {detail}")


def main():
    start_time = time.time()

    print_banner("EpiAgent — Full Pipeline Demo")
    print(f"  Timestamp: {datetime.utcnow().isoformat()} UTC")
    print(f"  Python: {sys.version.split()[0]}")
    print()

    # =======================================================================
    # STEP 1: Data Retrieval
    # =======================================================================
    print_banner("STEP 1: Data Retrieval", "─")
    print("  Generating synthetic COVID-19 epidemic data...")

    data = fetch_synthetic_data(
        pathogen="covid-19",
        duration_days=180,
        region="demo_region",
        noise_level=0.1,
        seed=42,
    )
    records = data["records"]

    print_metric("Records generated", str(data["record_count"]))
    print_metric("Date range", data["date_range"])
    print_metric("Total cases", f"{data['total_cases']:,}")
    print_metric("Total deaths", f"{data['total_deaths']:,}")
    print_metric("Peak day", f"{data['peak_day']} ({data['peak_cases']:,} cases)")

    # =======================================================================
    # STEP 2: Security Audit
    # =======================================================================
    print_banner("STEP 2: Security Audit (HIPAA)", "─")
    print("  Scanning for PII across 18 HIPAA identifier types...")

    security_result = run_security_audit(records)

    pii_status = "🔴 DETECTED" if security_result["pii_detected"] else "🟢 CLEAN"
    print_metric("PII Status", pii_status)
    print_metric("Data Hash", security_result["data_hash"][:32] + "...")
    print_metric("Schema Valid", str(security_result["schema_valid"]))

    if security_result["pii_detected"]:
        print_metric("PII Types Found", ", ".join(security_result["pii_types_found"]))
        print_metric("Items Redacted", str(security_result["items_redacted"]))
        records = security_result["cleaned_records"]

    # =======================================================================
    # STEP 3: Data Validation
    # =======================================================================
    print_banner("STEP 3: Data Quality Validation", "─")
    print("  Running 8-point epidemiological plausibility checks...")

    quality_result = validate_data(records)

    score = quality_result["quality_score"]
    score_bar = "█" * int(score * 20) + "░" * (20 - int(score * 20))
    score_emoji = "🟢" if score >= 0.7 else ("🟡" if score >= 0.5 else "🔴")

    print_metric("Quality Score", f"{score:.1%} {score_emoji}")
    print(f"    [{score_bar}]")
    print_metric("Valid Records", f"{quality_result['valid_records']}/{quality_result['total_records']}")
    print_metric("Errors", str(quality_result["error_count"]))
    print_metric("Warnings", str(quality_result["warning_count"]))

    if quality_result["issues"]:
        print("\n  Issues found:")
        for issue in quality_result["issues"][:5]:
            icon = "❌" if issue["severity"] == "error" else "⚠️"
            print(f"    {icon} [{issue['severity']}] {issue['message']}")

    # =======================================================================
    # STEP 4: Epidemiological Analysis
    # =======================================================================
    print_banner("STEP 4: Epidemiological Analysis", "─")

    # Extract case series
    case_series = [r["new_cases"] for r in records]
    dates = [r["date"] for r in records]
    case_json = json.dumps(case_series)

    # 4a: SEIR Model
    print("\n  ▸ Running SEIR compartmental model...")
    seir_result = run_seir_model(
        population=1_000_000, R0=2.5,
        latent_period=5.2, infectious_period=2.9,
        initial_infected=10, t_max=180,
    )
    summary = seir_result["summary"]
    print_metric("SEIR R0", f"{summary['R0']:.2f}")
    print_metric("Peak Day", str(seir_result["peak_day"]))
    print_metric("Peak Cases", f"{seir_result['peak_cases']:,.0f}")
    print_metric("Attack Rate", f"{seir_result['attack_rate']:.1%}")

    # 4b: Rt Estimation
    print("\n  ▸ Estimating Rt (Cori et al. 2013)...")
    rt_result = estimate_rt(case_json, pathogen="covid-19", window=7)

    if "error" not in rt_result:
        print_metric("Current Rt", f"{rt_result['current_rt']:.3f}")
        print_metric("Epidemic Phase", rt_result["current_phase"].upper())
        rt_summary = rt_result.get("summary", {})
        if "rt_range" in rt_summary:
            print_metric("Rt Range", f"{rt_summary['rt_range'][0]:.2f} – {rt_summary['rt_range'][1]:.2f}")
    else:
        print(f"  ⚠️ Rt estimation: {rt_result['error']}")

    # 4c: Epi Metrics
    print("\n  ▸ Computing epidemiological metrics...")
    total_cases = sum(case_series)
    total_deaths = sum(r["new_deaths"] for r in records)

    metrics_result = compute_epi_metrics(
        total_cases=total_cases,
        total_deaths=total_deaths,
        population=1_000_000,
        case_series_json=case_json,
    )

    if "cfr" in metrics_result:
        cfr = metrics_result["cfr"]
        print_metric("CFR", f"{cfr['value']:.4f}",
                     f"95% CI: [{cfr['ci'][0]:.4f}, {cfr['ci'][1]:.4f}] ({cfr['method']})")
    if "incidence_rate_per_100k" in metrics_result:
        inc = metrics_result["incidence_rate_per_100k"]
        print_metric("Incidence Rate", f"{inc['value']:.1f} per 100k",
                     f"95% CI: [{inc['ci'][0]:.1f}, {inc['ci'][1]:.1f}]")
    if "doubling_time_days" in metrics_result:
        dt = metrics_result["doubling_time_days"]
        dt_val = f"{dt['value']:.1f} days" if not np.isnan(dt["value"]) else "N/A (declining)"
        print_metric("Doubling Time", dt_val)

    # 4d: Changepoint Detection
    print("\n  ▸ Detecting changepoints (BOCPD)...")
    cp_result = detect_changepoints(case_json, hazard_lambda=100.0)

    if "error" not in cp_result:
        print_metric("Changepoints Found", str(cp_result["n_changepoints"]))
        for i, (cp, score) in enumerate(zip(
            cp_result["changepoints"][:5],
            cp_result["confidence_scores"][:5],
        )):
            cp_date = dates[cp] if cp < len(dates) else f"day {cp}"
            print(f"    CP {i+1}: Day {cp} ({cp_date}) — confidence: {score:.0%}")
    else:
        print(f"  ⚠️ Changepoint detection: {cp_result['error']}")

    # =======================================================================
    # STEP 5: ML Forecasting + SHAP
    # =======================================================================
    print_banner("STEP 5: ML Forecasting & Explainability", "─")

    # 5a: Forecast
    print("\n  ▸ Running XGBoost+Prophet ensemble forecast...")
    forecast_result = run_ml_forecast(case_json, horizon=14, dates_json=json.dumps(dates))

    if "error" not in forecast_result:
        ensemble = forecast_result["ensemble"]
        print_metric("Forecast Model", ensemble["model_name"])
        print_metric("Forecast RMSE", f"{ensemble['rmse']:.2f}" if ensemble.get("rmse") else "N/A")
        print_metric("14-Day Forecast Range",
                     f"{min(ensemble['predicted']):,.0f} – {max(ensemble['predicted']):,.0f}")
        print_metric("Models in Ensemble", str(forecast_result["model_count"]))
    else:
        print(f"  ⚠️ Forecast: {forecast_result['error']}")
        forecast_result = None

    # 5b: SHAP
    print("\n  ▸ Running SHAP explainability analysis...")
    shap_result = run_shap_analysis(case_json, top_k=10)

    if "error" not in shap_result:
        print(f"\n  {shap_result.get('narrative', 'No narrative')}")
    else:
        print(f"  ⚠️ SHAP: {shap_result['error']}")
        shap_result = None

    # =======================================================================
    # STEP 6: Dashboard Generation
    # =======================================================================
    print_banner("STEP 6: Dashboard Generation", "─")
    print("  Generating interactive Plotly dashboard...")

    dashboard_path = generate_dashboard(
        surveillance_data=data,
        rt_results=rt_result if "error" not in rt_result else None,
        seir_results=seir_result,
        forecast_results=forecast_result,
        shap_results=shap_result,
        changepoint_results=cp_result if "error" not in cp_result else None,
        epi_metrics=metrics_result,
        security_report=security_result,
        output_path=str(project_root / "epiagent_dashboard.html"),
    )

    print_metric("Dashboard Path", dashboard_path)
    print_metric("File Size", f"{Path(dashboard_path).stat().st_size / 1024:.0f} KB")

    # =======================================================================
    # Summary
    # =======================================================================
    elapsed = time.time() - start_time
    print_banner("PIPELINE COMPLETE")
    print(f"  ⏱️  Total execution time: {elapsed:.1f} seconds")
    print(f"  📊 Dashboard: {dashboard_path}")
    print(f"  🔒 HIPAA: {'CLEAN' if not security_result['pii_detected'] else 'REDACTED'}")
    print(f"  ✅ Quality: {quality_result['quality_score']:.0%}")
    print(f"  📈 Current Rt: {rt_result.get('current_rt', 'N/A')}")
    print()
    print("  Open the dashboard in your browser to explore interactive visualizations!")
    print()

    return {
        "data": data,
        "security": security_result,
        "quality": quality_result,
        "seir": seir_result,
        "rt": rt_result,
        "metrics": metrics_result,
        "changepoints": cp_result,
        "forecast": forecast_result,
        "shap": shap_result,
        "dashboard_path": dashboard_path,
    }


if __name__ == "__main__":
    main()
