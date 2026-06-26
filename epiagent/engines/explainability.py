"""SHAP Explainability Module for Epidemic Forecasting Models.

Provides post-hoc explainability for XGBoost epidemic forecasts using
SHAP (SHapley Additive exPlanations) values.

SHAP provides:
    1. Global feature importance — which features matter most overall
    2. Local explanations — why was this specific prediction made
    3. Feature interaction effects — how features combine
    4. Temporal attribution — how driver importance changes over time

This module wraps SHAP's TreeExplainer for fast, exact computation
of Shapley values for tree-based models (XGBoost, Random Forest).

References:
    Lundberg SM, Lee SI. (2017) "A Unified Approach to Interpreting
    Model Predictions." NeurIPS.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class SHAPExplanation:
    """SHAP analysis results for a forecast model.

    Attributes:
        feature_names: List of feature names.
        global_importance: Mean |SHAP| for each feature (global ranking).
        shap_values: Full SHAP value matrix (n_samples × n_features).
        top_drivers: Top-k most important features with their scores.
        base_value: Expected model output (baseline prediction).
        temporal_importance: Feature importance over time windows.
    """
    feature_names: list[str]
    global_importance: dict[str, float]
    shap_values: np.ndarray
    top_drivers: list[dict[str, float]]
    base_value: float
    temporal_importance: dict[str, list[float]] = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Convert to JSON-serializable dict (excluding raw SHAP matrix)."""
        return {
            "top_drivers": self.top_drivers,
            "global_importance": {
                k: round(v, 4) for k, v in self.global_importance.items()
            },
            "base_value": round(self.base_value, 2),
            "temporal_importance": {
                k: [round(x, 4) for x in v]
                for k, v in self.temporal_importance.items()
            },
        }

    def narrative(self) -> str:
        """Generate human-readable narrative of SHAP findings.

        Returns:
            Multi-sentence explanation suitable for SitRep inclusion.
        """
        if not self.top_drivers:
            return "Insufficient data for SHAP analysis."

        lines = ["**Forecast Driver Analysis (SHAP):**"]

        for i, driver in enumerate(self.top_drivers[:5], 1):
            name = driver["feature"]
            importance = driver["importance"]

            # Translate feature names to epidemiological terms
            epi_name = _translate_feature_name(name)
            lines.append(
                f"{i}. **{epi_name}** (importance: {importance:.3f})"
            )

        # Add temporal trend if available
        if self.temporal_importance:
            top_feat = self.top_drivers[0]["feature"]
            if top_feat in self.temporal_importance:
                recent = self.temporal_importance[top_feat][-3:]
                if len(recent) >= 2:
                    trend = "increasing" if recent[-1] > recent[0] else "decreasing"
                    lines.append(
                        f"\nThe influence of {_translate_feature_name(top_feat)} "
                        f"has been **{trend}** over recent time windows."
                    )

        return "\n".join(lines)


def _translate_feature_name(name: str) -> str:
    """Translate ML feature names to epidemiological terms."""
    translations = {
        "lag_1": "yesterday's case count",
        "lag_2": "cases 2 days ago",
        "lag_7": "cases 1 week ago",
        "lag_14": "cases 2 weeks ago",
        "rolling_7d_mean": "7-day average case count",
        "rolling_7d_std": "7-day case count variability",
        "rolling_14d_mean": "14-day average case count",
        "rolling_14d_std": "14-day case count variability",
        "rolling_7d_min": "7-day minimum daily cases",
        "rolling_7d_max": "7-day peak daily cases",
        "rolling_14d_min": "14-day minimum daily cases",
        "rolling_14d_max": "14-day peak daily cases",
        "day_change": "day-over-day change in cases",
        "week_change_ratio": "week-over-week growth ratio",
        "day_of_week": "day of the week (reporting pattern)",
        "month": "month of year (seasonality)",
        "is_weekend": "weekend reporting effect",
    }
    return translations.get(name, name)


def explain_forecast(
    model,
    X_data: np.ndarray,
    feature_names: list[str],
    top_k: int = 10,
    n_temporal_windows: int = 4,
) -> SHAPExplanation:
    """Compute SHAP explanations for an XGBoost forecast model.

    Args:
        model: Trained XGBoost model (or any tree-based model).
        X_data: Feature matrix (n_samples × n_features).
        feature_names: List of feature names.
        top_k: Number of top features to highlight.
        n_temporal_windows: Number of time windows for temporal analysis.

    Returns:
        SHAPExplanation with global and local explanations.
    """
    try:
        import shap
    except ImportError:
        logger.warning(
            "SHAP not installed. Falling back to model feature_importances_."
        )
        return _fallback_importance(model, feature_names, top_k)

    # TreeExplainer for exact, fast Shapley values
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_data)

    # Base value
    base_value = float(explainer.expected_value)

    # Global importance: mean |SHAP| per feature
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    global_importance = dict(zip(feature_names, mean_abs_shap.tolist()))

    # Sort by importance
    sorted_features = sorted(
        global_importance.items(), key=lambda x: x[1], reverse=True
    )

    # Top-k drivers
    top_drivers = [
        {"feature": name, "importance": score}
        for name, score in sorted_features[:top_k]
    ]

    # Temporal importance: split data into windows and compute per-window SHAP
    temporal_importance = {}
    if len(X_data) >= n_temporal_windows * 5:
        window_size = len(X_data) // n_temporal_windows
        top_feature_names = [d["feature"] for d in top_drivers[:5]]

        for feat_name in top_feature_names:
            feat_idx = feature_names.index(feat_name)
            window_importances = []

            for w in range(n_temporal_windows):
                start = w * window_size
                end = (w + 1) * window_size if w < n_temporal_windows - 1 else len(X_data)
                window_shap = np.abs(shap_values[start:end, feat_idx]).mean()
                window_importances.append(float(window_shap))

            temporal_importance[feat_name] = window_importances

    return SHAPExplanation(
        feature_names=feature_names,
        global_importance=dict(sorted_features),
        shap_values=shap_values,
        top_drivers=top_drivers,
        base_value=base_value,
        temporal_importance=temporal_importance,
    )


def _fallback_importance(
    model,
    feature_names: list[str],
    top_k: int,
) -> SHAPExplanation:
    """Fallback when SHAP is not available: use model's feature_importances_.

    Args:
        model: Tree model with feature_importances_ attribute.
        feature_names: Feature names.
        top_k: Number of top features.

    Returns:
        SHAPExplanation with feature importance (no SHAP values).
    """
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_.tolist()
    else:
        logger.warning("Model has no feature_importances_. Using uniform.")
        importances = [1.0 / len(feature_names)] * len(feature_names)

    global_importance = dict(zip(feature_names, importances))
    sorted_features = sorted(
        global_importance.items(), key=lambda x: x[1], reverse=True
    )

    top_drivers = [
        {"feature": name, "importance": score}
        for name, score in sorted_features[:top_k]
    ]

    return SHAPExplanation(
        feature_names=feature_names,
        global_importance=dict(sorted_features),
        shap_values=np.array([]),
        top_drivers=top_drivers,
        base_value=0.0,
    )


def explain_xgboost_forecast(
    case_series: np.ndarray,
    dates: list[str] | None = None,
    n_lags: int = 14,
    top_k: int = 10,
) -> tuple[SHAPExplanation, dict]:
    """Convenience: train XGBoost + compute SHAP in one call.

    Args:
        case_series: Daily case count array.
        dates: Optional ISO date strings.
        n_lags: Number of lag features.
        top_k: Number of top features.

    Returns:
        Tuple of (SHAPExplanation, model_metrics_dict).
    """
    try:
        import xgboost as xgb
    except ImportError:
        raise ImportError("XGBoost required for this function")

    from .ml_forecaster import create_lag_features

    df = create_lag_features(case_series, dates, n_lags=n_lags)

    feature_cols = [
        c for c in df.columns if c not in ("date", "target", "cases")
    ]

    X = df[feature_cols].values
    y = df["target"].values

    # Train
    model = xgb.XGBRegressor(
        n_estimators=200, max_depth=6, learning_rate=0.1,
        random_state=42, verbosity=0,
    )
    model.fit(X, y, verbose=False)

    # SHAP
    explanation = explain_forecast(model, X, feature_cols, top_k=top_k)

    metrics = {
        "n_samples": len(X),
        "n_features": len(feature_cols),
        "model": "XGBoost",
    }

    return explanation, metrics
