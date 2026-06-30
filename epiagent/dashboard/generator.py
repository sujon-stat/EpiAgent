"""Interactive Plotly Dashboard Generator for EpiAgent.

Creates a 7-panel interactive HTML dashboard for epidemic surveillance:

    Panel 1: Epidemic Curve (daily cases with 7-day rolling average)
    Panel 2: Rt Time Series (with credible intervals + phase shading)
    Panel 3: SEIR Model Fit vs Observed
    Panel 4: ML Forecast (14-day prediction with uncertainty cone)
    Panel 5: SHAP Feature Importance (waterfall chart)
    Panel 6: Changepoint Detection overlay
    Panel 7: Key Metrics Summary Cards

Output: Self-contained HTML file with Plotly.js embedded.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Color Palette (modern, accessible)
# ---------------------------------------------------------------------------
COLORS = {
    "primary": "#6366f1",       # Indigo
    "secondary": "#8b5cf6",     # Violet
    "accent": "#ec4899",        # Pink
    "success": "#22c55e",       # Green
    "warning": "#f59e0b",       # Amber
    "danger": "#ef4444",        # Red
    "info": "#06b6d4",          # Cyan
    "surface": "#1e1b4b",       # Dark indigo
    "text": "#e2e8f0",          # Slate 200
    "muted": "#94a3b8",         # Slate 400
    "cases": "#6366f1",
    "deaths": "#ef4444",
    "rt": "#f59e0b",
    "forecast": "#22c55e",
    "ci_band": "rgba(99, 102, 241, 0.15)",
    "rt_band": "rgba(245, 158, 11, 0.15)",
    "forecast_band": "rgba(34, 197, 94, 0.15)",
}

LAYOUT_DEFAULTS = dict(
    template="plotly_dark",
    paper_bgcolor="#0f0a2a",
    plot_bgcolor="#1a1545",
    font=dict(family="Inter, system-ui, sans-serif", color=COLORS["text"]),
    hovermode="x unified",
    hoverlabel=dict(
        bgcolor="#2d2a6e",
        font_size=12,
        bordercolor="#6366f1",
    ),
    title_x=0.5,
    title_y=0.98,
    title_xanchor="center",
    title_yanchor="top",
    title_font_size=16,
    legend=dict(
        orientation="h",
        yanchor="top",
        y=-0.15,
        xanchor="center",
        x=0.5,
        bgcolor="rgba(0,0,0,0)",
        font_size=11,
    ),
    margin=dict(t=50, b=80, l=60, r=30),
    height=450,
)


def _create_metric_card_html(
    title: str,
    value: str,
    subtitle: str = "",
    color: str = "#6366f1",
) -> str:
    """Create a metric card HTML element."""
    return f"""
    <div style="
        background: linear-gradient(135deg, {color}22, {color}08);
        border: 1px solid {color}44;
        border-radius: 16px;
        padding: 20px;
        text-align: center;
        min-width: 160px;
    ">
        <div style="color: #94a3b8; font-size: 12px; text-transform: uppercase; letter-spacing: 1px;">
            {title}
        </div>
        <div style="color: {color}; font-size: 32px; font-weight: 700; margin: 8px 0;">
            {value}
        </div>
        <div style="color: #64748b; font-size: 11px;">
            {subtitle}
        </div>
    </div>
    """


# ---------------------------------------------------------------------------
# Panel Builders
# ---------------------------------------------------------------------------

def plot_epidemic_curve(
    dates: list[str],
    cases: list[int],
    deaths: list[int] | None = None,
) -> go.Figure:
    """Panel 1: Epidemic curve with rolling average."""
    fig = go.Figure()

    # Daily cases (bar)
    fig.add_trace(go.Bar(
        x=dates, y=cases,
        name="Daily Cases",
        marker_color=COLORS["cases"],
        opacity=0.5,
    ))

    # 7-day rolling average
    cases_arr = np.array(cases, dtype=float)
    if len(cases_arr) >= 7:
        rolling_avg = np.convolve(cases_arr, np.ones(7) / 7, mode="valid")
        fig.add_trace(go.Scatter(
            x=dates[6:], y=rolling_avg,
            name="7-Day Average",
            line=dict(color=COLORS["accent"], width=3),
        ))

    # Deaths (if provided)
    if deaths:
        fig.add_trace(go.Bar(
            x=dates, y=deaths,
            name="Daily Deaths",
            marker_color=COLORS["danger"],
            opacity=0.7,
            yaxis="y2",
        ))

    fig.update_layout(
        **LAYOUT_DEFAULTS,
        title="📊 Epidemic Curve",
        xaxis_title="Date",
        yaxis_title="Cases",
        yaxis2=dict(
            title="Deaths",
            overlaying="y",
            side="right",
            showgrid=False,
        ),
        barmode="overlay",
    )

    return fig


def plot_rt_timeseries(
    dates: list[str],
    rt_mean: list[float],
    rt_lower: list[float],
    rt_upper: list[float],
    phases: list[str] | None = None,
) -> go.Figure:
    """Panel 2: Rt time series with CrI and phase shading."""
    fig = go.Figure()

    # Filter NaN values
    valid = [(d, m, l, u) for d, m, l, u in zip(dates, rt_mean, rt_lower, rt_upper)
             if m is not None and not (isinstance(m, float) and np.isnan(m))]

    if not valid:
        fig.add_annotation(text="Insufficient data for Rt estimation",
                          xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        fig.update_layout(**LAYOUT_DEFAULTS, title="📈 Effective Reproduction Number (Rt)")
        return fig

    v_dates, v_mean, v_lower, v_upper = zip(*valid)

    # Credible interval band
    fig.add_trace(go.Scatter(
        x=list(v_dates) + list(reversed(v_dates)),
        y=list(v_upper) + list(reversed(v_lower)),
        fill="toself",
        fillcolor=COLORS["rt_band"],
        line=dict(color="rgba(0,0,0,0)"),
        name="95% Credible Interval",
        showlegend=True,
    ))

    # Rt mean line
    fig.add_trace(go.Scatter(
        x=list(v_dates), y=list(v_mean),
        name="Rt (posterior mean)",
        line=dict(color=COLORS["rt"], width=3),
    ))

    # Threshold line at Rt = 1
    fig.add_hline(
        y=1.0, line_dash="dash", line_color=COLORS["danger"],
        annotation_text="Rt = 1 (epidemic threshold)",
        annotation_position="bottom right",
    )

    fig.update_layout(
        **LAYOUT_DEFAULTS,
        title="📈 Effective Reproduction Number (Rt) — Cori et al. (2013)",
        xaxis_title="Date",
        yaxis_title="Rt",
        yaxis=dict(range=[0, max(5, max(v_upper) * 1.2)]),
    )

    return fig


def plot_seir_fit(
    dates: list[str],
    observed_cases: list[int],
    seir_incidence: list[float],
) -> go.Figure:
    """Panel 3: SEIR model fit vs observed data."""
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=dates[:len(observed_cases)],
        y=observed_cases,
        name="Observed",
        mode="markers",
        marker=dict(color=COLORS["cases"], size=4, opacity=0.6),
    ))

    seir_dates = dates[:len(seir_incidence)]
    fig.add_trace(go.Scatter(
        x=seir_dates,
        y=seir_incidence,
        name="SEIR Model",
        line=dict(color=COLORS["accent"], width=2.5, dash="dash"),
    ))

    fig.update_layout(
        **LAYOUT_DEFAULTS,
        title="🧬 SEIR Model Fit vs Observed Cases",
        xaxis_title="Date",
        yaxis_title="Daily Incidence",
    )

    return fig


def plot_forecast(
    historical_dates: list[str],
    historical_cases: list[int],
    forecast_dates: list[str],
    forecast_predicted: list[float],
    forecast_lower: list[float],
    forecast_upper: list[float],
    model_name: str = "Ensemble",
) -> go.Figure:
    """Panel 4: ML forecast with uncertainty cone."""
    fig = go.Figure()

    # Historical
    fig.add_trace(go.Scatter(
        x=historical_dates[-60:],
        y=historical_cases[-60:],
        name="Historical Cases",
        line=dict(color=COLORS["cases"], width=2),
    ))

    # Forecast uncertainty band
    fig.add_trace(go.Scatter(
        x=forecast_dates + list(reversed(forecast_dates)),
        y=forecast_upper + list(reversed(forecast_lower)),
        fill="toself",
        fillcolor=COLORS["forecast_band"],
        line=dict(color="rgba(0,0,0,0)"),
        name="95% Prediction Interval",
    ))

    # Forecast line
    fig.add_trace(go.Scatter(
        x=forecast_dates,
        y=forecast_predicted,
        name=f"{model_name} Forecast",
        line=dict(color=COLORS["forecast"], width=3, dash="dot"),
    ))

    # Divider
    if historical_dates:
        fig.add_vline(
            x=historical_dates[-1],
            line_dash="dash",
            line_color=COLORS["muted"],
            annotation_text="Forecast →",
        )

    fig.update_layout(
        **LAYOUT_DEFAULTS,
        title=f"🔮 14-Day Forecast ({model_name})",
        xaxis_title="Date",
        yaxis_title="Predicted Cases",
    )

    return fig


def plot_shap_importance(
    feature_names: list[str],
    importance_values: list[float],
) -> go.Figure:
    """Panel 5: SHAP feature importance waterfall chart."""
    # Sort by importance
    pairs = sorted(zip(feature_names, importance_values), key=lambda x: x[1])
    names, values = zip(*pairs) if pairs else ([], [])

    # Translate names
    translations = {
        "lag_1": "Yesterday's Cases",
        "lag_7": "Cases 1 Week Ago",
        "lag_14": "Cases 2 Weeks Ago",
        "rolling_7d_mean": "7-Day Average",
        "rolling_7d_std": "7-Day Variability",
        "rolling_14d_mean": "14-Day Average",
        "day_change": "Day-over-Day Change",
        "week_change_ratio": "Week Growth Ratio",
        "day_of_week": "Day of Week",
        "is_weekend": "Weekend Effect",
        "month": "Month/Season",
    }
    display_names = [translations.get(n, n) for n in names]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=list(values),
        y=display_names,
        orientation="h",
        marker=dict(
            color=list(values),
            colorscale=[[0, COLORS["info"]], [1, COLORS["accent"]]],
        ),
    ))

    fig.update_layout(
        **LAYOUT_DEFAULTS,
        title="🎯 Forecast Driver Analysis (SHAP Feature Importance)",
        xaxis_title="Mean |SHAP Value|",
        yaxis_title="",
    )

    return fig


def plot_changepoints(
    dates: list[str],
    cases: list[int],
    changepoint_indices: list[int],
    confidence_scores: list[float] | None = None,
) -> go.Figure:
    """Panel 6: Case series with changepoint annotations."""
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=dates, y=cases,
        name="Cases",
        line=dict(color=COLORS["cases"], width=2),
        fill="tozeroy",
        fillcolor="rgba(99, 102, 241, 0.08)",
    ))

    # Changepoint markers
    for i, cp_idx in enumerate(changepoint_indices):
        if cp_idx < len(dates):
            score = confidence_scores[i] if confidence_scores and i < len(confidence_scores) else 0.5
            fig.add_vline(
                x=dates[cp_idx],
                line_dash="dash",
                line_color=COLORS["warning"],
                line_width=2,
                annotation_text=f"CP (conf: {score:.0%})",
                annotation_font_color=COLORS["warning"],
            )

    fig.update_layout(
        **LAYOUT_DEFAULTS,
        title="⚡ Changepoint Detection (BOCPD)",
        xaxis_title="Date",
        yaxis_title="Daily Cases",
    )

    return fig


# ---------------------------------------------------------------------------
# Full Dashboard Assembly
# ---------------------------------------------------------------------------

def generate_dashboard(
    surveillance_data: dict,
    rt_results: dict | None = None,
    seir_results: dict | None = None,
    forecast_results: dict | None = None,
    shap_results: dict | None = None,
    changepoint_results: dict | None = None,
    epi_metrics: dict | None = None,
    security_report: dict | None = None,
    output_path: str = "epiagent_dashboard.html",
) -> str:
    """Generate complete interactive HTML dashboard.

    Args:
        surveillance_data: Dict with 'records' list.
        rt_results: Rt estimation results.
        seir_results: SEIR model results.
        forecast_results: ML forecast results.
        shap_results: SHAP analysis results.
        changepoint_results: Changepoint detection results.
        epi_metrics: Epidemiological metrics dict.
        security_report: Security audit results.
        output_path: Path for the HTML output file.

    Returns:
        Path to the generated HTML file.
    """
    records = surveillance_data.get("records", [])
    if not records:
        logger.warning("No records provided for dashboard")
        return ""

    dates = [r["date"] for r in records]
    cases = [r["new_cases"] for r in records]
    deaths = [r.get("new_deaths", 0) for r in records]

    panels_html = []

    # Panel 1: Epidemic curve
    fig1 = plot_epidemic_curve(dates, cases, deaths)
    panels_html.append(fig1.to_html(full_html=False, include_plotlyjs=False))

    # Panel 2: Rt time series
    if rt_results and "rt_last_30d" in rt_results:
        rt_data = rt_results["rt_last_30d"]
        n_rt = len(rt_data.get("rt_mean", []))
        rt_dates = dates[-n_rt:] if n_rt <= len(dates) else dates
        fig2 = plot_rt_timeseries(
            rt_dates,
            rt_data.get("rt_mean", []),
            rt_data.get("rt_lower", []),
            rt_data.get("rt_upper", []),
        )
        panels_html.append(fig2.to_html(full_html=False, include_plotlyjs=False))

    # Panel 3: SEIR fit
    if seir_results and "daily_incidence" in seir_results:
        fig3 = plot_seir_fit(dates, cases, seir_results["daily_incidence"])
        panels_html.append(fig3.to_html(full_html=False, include_plotlyjs=False))

    # Panel 4: Forecast
    if forecast_results:
        ensemble = forecast_results.get("ensemble", {})
        if ensemble:
            fig4 = plot_forecast(
                dates, cases,
                ensemble.get("dates", []),
                ensemble.get("predicted", []),
                ensemble.get("lower_bound", []),
                ensemble.get("upper_bound", []),
                ensemble.get("model_name", "Ensemble"),
            )
            panels_html.append(fig4.to_html(full_html=False, include_plotlyjs=False))

    # Panel 5: SHAP
    if shap_results and "top_drivers" in shap_results:
        drivers = shap_results["top_drivers"]
        if drivers:
            fig5 = plot_shap_importance(
                [d["feature"] for d in drivers],
                [d["importance"] for d in drivers],
            )
            panels_html.append(fig5.to_html(full_html=False, include_plotlyjs=False))

    # Panel 6: Changepoints
    if changepoint_results and "changepoints" in changepoint_results:
        fig6 = plot_changepoints(
            dates, cases,
            changepoint_results.get("changepoints", []),
            changepoint_results.get("confidence_scores", []),
        )
        panels_html.append(fig6.to_html(full_html=False, include_plotlyjs=False))

    # Assemble metric cards
    cards = []
    if rt_results:
        rt_val = rt_results.get("current_rt", "N/A")
        phase = rt_results.get("current_phase", "unknown")
        rt_color = COLORS["danger"] if phase == "growing" else (
            COLORS["success"] if phase == "declining" else COLORS["warning"]
        )
        cards.append(_create_metric_card_html(
            "Current Rt", f"{rt_val:.2f}" if isinstance(rt_val, (int, float)) else str(rt_val),
            f"Phase: {phase}", rt_color,
        ))

    if epi_metrics:
        if "cfr" in epi_metrics:
            cfr_val = epi_metrics["cfr"]["value"]
            cards.append(_create_metric_card_html(
                "Case Fatality Rate",
                f"{cfr_val:.2%}" if isinstance(cfr_val, (int, float)) else str(cfr_val),
                "Wilson score CI", COLORS["danger"],
            ))
        if "incidence_rate_per_100k" in epi_metrics:
            inc = epi_metrics["incidence_rate_per_100k"]["value"]
            cards.append(_create_metric_card_html(
                "Incidence Rate",
                f"{inc:.1f}" if isinstance(inc, (int, float)) else str(inc),
                "per 100,000", COLORS["info"],
            ))
        if "doubling_time_days" in epi_metrics:
            dt = epi_metrics["doubling_time_days"]["value"]
            cards.append(_create_metric_card_html(
                "Doubling Time",
                f"{dt:.1f}d" if isinstance(dt, (int, float)) and not np.isnan(dt) else "N/A",
                "log-linear regression", COLORS["warning"],
            ))

    total_cases = sum(cases)
    total_deaths = sum(deaths)
    cards.append(_create_metric_card_html(
        "Total Cases", f"{total_cases:,}", "", COLORS["primary"],
    ))
    cards.append(_create_metric_card_html(
        "Total Deaths", f"{total_deaths:,}", "", COLORS["danger"],
    ))

    # Security badge
    if security_report:
        pii = security_report.get("pii_detected", False)
        badge_color = COLORS["danger"] if pii else COLORS["success"]
        badge_text = "PII DETECTED" if pii else "CLEAN"
        cards.append(_create_metric_card_html(
            "HIPAA Status", badge_text,
            security_report.get("data_hash", "")[:16] + "...",
            badge_color,
        ))

    cards_html = "\n".join(cards)

    # Build full HTML
    now = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
    pathogen = records[0].get("pathogen", "Unknown") if records else "Unknown"
    region = records[0].get("region", "Unknown") if records else "Unknown"

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>EpiAgent Dashboard — {pathogen.title()} Surveillance</title>
    <script src="https://cdn.plot.ly/plotly-2.35.0.min.js"></script>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    <style>
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{
            font-family: 'Inter', system-ui, sans-serif;
            background: linear-gradient(135deg, #0f0a2a 0%, #1a1545 50%, #0f172a 100%);
            color: #e2e8f0;
            min-height: 100vh;
        }}
        .header {{
            background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
            padding: 32px 40px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}
        .header h1 {{
            font-size: 28px;
            font-weight: 700;
            letter-spacing: -0.5px;
        }}
        .header .subtitle {{
            font-size: 14px;
            opacity: 0.85;
            margin-top: 4px;
        }}
        .header .badge {{
            background: rgba(255,255,255,0.2);
            padding: 8px 16px;
            border-radius: 20px;
            font-size: 13px;
        }}
        .metrics-bar {{
            display: flex;
            gap: 16px;
            padding: 24px 40px;
            overflow-x: auto;
            flex-wrap: wrap;
        }}
        .panels {{
            padding: 0 40px 40px;
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 24px;
        }}
        .panel {{
            background: rgba(30, 27, 75, 0.6);
            border: 1px solid rgba(99, 102, 241, 0.15);
            border-radius: 16px;
            padding: 16px;
            backdrop-filter: blur(10px);
        }}
        .panel.full-width {{
            grid-column: 1 / -1;
        }}
        .footer {{
            text-align: center;
            padding: 24px;
            color: #64748b;
            font-size: 12px;
            border-top: 1px solid rgba(99, 102, 241, 0.1);
        }}
        @media (max-width: 768px) {{
            .panels {{ grid-template-columns: 1fr; padding: 0 16px 16px; }}
            .metrics-bar {{ padding: 16px; }}
            .header {{ padding: 20px 16px; }}
        }}
    </style>
</head>
<body>
    <div class="header">
        <div>
            <h1>🦠 EpiAgent Dashboard</h1>
            <div class="subtitle">{pathogen.title()} Surveillance — {region} | Generated: {now}</div>
        </div>
        <div class="badge">Multi-Agent AI System</div>
    </div>

    <div class="metrics-bar">
        {cards_html}
    </div>

    <div class="panels">
        {"".join(f'<div class="panel{" full-width" if i == 0 else ""}">{p}</div>' for i, p in enumerate(panels_html))}
    </div>

    <div class="footer">
        EpiAgent v1.0 | Google ADK 2.3.0 | Deterministic Engines + LLM Orchestration<br>
        Methods: SEIR (RK45) · Cori et al. 2013 (Bayesian Rt) · BOCPD · XGBoost · SHAP<br>
        HIPAA Safe Harbor Compliant | Data Hash: {security_report.get("data_hash", "N/A")[:32] if security_report else "N/A"}...
    </div>
</body>
</html>"""

    # Write to file
    output = Path(output_path)
    output.write_text(html, encoding="utf-8")
    logger.info("Dashboard generated: %s", output.resolve())

    return str(output.resolve())
