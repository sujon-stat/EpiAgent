"""ML Forecasting Ensemble for Epidemic Time Series.

Implements a two-model ensemble for epidemic case forecasting:

1. XGBoost Gradient Boosted Trees — Feature-engineered supervised model
   using lag features, rolling statistics, and epidemiological covariates.
   Provides the primary forecast with SHAP explainability.

2. Prophet (Meta/Facebook) — Additive decomposition model with
   trend + seasonality + holiday components. Optional dependency.

Ensemble Strategy:
    Weighted average using inverse RMSE from validation set.
    If only one model is available, uses that model alone.

The forecaster is designed to be called as a deterministic FunctionTool
by the ML Forecasting Agent — the LLM interprets results, not computes them.
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error

logger = logging.getLogger(__name__)


@dataclass
class ForecastOutput:
    """Output from a single forecasting model.

    Attributes:
        dates: Forecast date strings (ISO 8601).
        predicted: Point predictions.
        lower_bound: Lower prediction interval (approx 95%).
        upper_bound: Upper prediction interval (approx 95%).
        model_name: Name of the model.
        rmse: Root Mean Squared Error on validation data.
        mae: Mean Absolute Error on validation data.
        feature_importance: Feature importance dict (for tree models).
    """
    dates: list[str]
    predicted: list[float]
    lower_bound: list[float]
    upper_bound: list[float]
    model_name: str
    rmse: float | None = None
    mae: float | None = None
    feature_importance: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Convert to JSON-serializable dict."""
        return {
            "dates": self.dates,
            "predicted": [round(p, 2) for p in self.predicted],
            "lower_bound": [round(lb, 2) for lb in self.lower_bound],
            "upper_bound": [round(ub, 2) for ub in self.upper_bound],
            "model_name": self.model_name,
            "rmse": round(self.rmse, 4) if self.rmse else None,
            "mae": round(self.mae, 4) if self.mae else None,
            "feature_importance": {
                k: round(v, 4) for k, v in self.feature_importance.items()
            },
        }


# ---------------------------------------------------------------------------
# Feature Engineering
# ---------------------------------------------------------------------------

def create_lag_features(
    case_series: np.ndarray,
    dates: list[str] | None = None,
    n_lags: int = 14,
) -> pd.DataFrame:
    """Create supervised learning features from epidemic time series.

    Features:
        - Lag features: cases at t-1, t-2, ..., t-n_lags
        - Rolling statistics: 7-day and 14-day mean, std, min, max
        - Growth features: day-over-day change, 7-day change ratio
        - Calendar features: day of week, month, is_weekend

    Args:
        case_series: Array of daily case counts.
        dates: Optional list of ISO date strings.
        n_lags: Number of lag features to create.

    Returns:
        DataFrame with features and 'target' column.
    """
    n = len(case_series)
    df = pd.DataFrame({"cases": case_series.astype(float)})

    if dates is not None:
        df["date"] = pd.to_datetime(dates)
    else:
        df["date"] = pd.date_range(end=datetime.today(), periods=n, freq="D")

    # Lag features
    for lag in range(1, n_lags + 1):
        df[f"lag_{lag}"] = df["cases"].shift(lag)

    # Rolling statistics (shifted by 1 to prevent data leakage)
    for window in [7, 14]:
        rolling = df["cases"].shift(1).rolling(window=window, min_periods=1)
        df[f"rolling_{window}d_mean"] = rolling.mean()
        df[f"rolling_{window}d_std"] = rolling.std().fillna(0)
        df[f"rolling_{window}d_min"] = rolling.min()
        df[f"rolling_{window}d_max"] = rolling.max()

    # Growth features
    df["day_change"] = df["cases"].diff().shift(1)
    prev_7d = df["cases"].shift(7)
    df["week_change_ratio"] = np.where(
        prev_7d > 0,
        df["cases"].shift(1) / prev_7d,
        1.0,
    )

    # Calendar features
    df["day_of_week"] = df["date"].dt.dayofweek
    df["month"] = df["date"].dt.month
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)

    # Target
    df["target"] = df["cases"]

    # Drop rows with NaN from lagging
    df = df.dropna().reset_index(drop=True)

    return df


# ---------------------------------------------------------------------------
# XGBoost Forecaster
# ---------------------------------------------------------------------------

def forecast_xgboost(
    case_series: np.ndarray,
    dates: list[str] | None = None,
    horizon: int = 14,
    n_lags: int = 14,
    test_size: int = 14,
) -> ForecastOutput:
    """Forecast epidemic cases using XGBoost with engineered features.

    Args:
        case_series: Array of daily case counts.
        dates: Optional list of ISO date strings.
        horizon: Number of days to forecast ahead.
        n_lags: Number of lag features.
        test_size: Number of recent days for validation.

    Returns:
        ForecastOutput with predictions and feature importance.
    """
    try:
        import xgboost as xgb
    except ImportError:
        logger.error("XGBoost not installed. Run: pip install xgboost")
        raise

    # Create features
    df = create_lag_features(case_series, dates, n_lags=n_lags)

    if len(df) < test_size + 10:
        raise ValueError(
            f"Insufficient data for XGBoost: need at least "
            f"{test_size + 10} rows after feature engineering, got {len(df)}"
        )

    # Feature columns (exclude date, target, cases)
    feature_cols = [
        c for c in df.columns
        if c not in ("date", "target", "cases")
    ]

    # Train/validation split (temporal)
    train_df = df.iloc[:-test_size]
    val_df = df.iloc[-test_size:]

    X_train = train_df[feature_cols].values
    y_train = train_df["target"].values
    X_val = val_df[feature_cols].values
    y_val = val_df["target"].values

    # Train XGBoost
    model = xgb.XGBRegressor(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=3,
        random_state=42,
        verbosity=0,
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

    # Validation metrics
    val_pred = model.predict(X_val)
    rmse = float(np.sqrt(mean_squared_error(y_val, val_pred)))
    mae = float(mean_absolute_error(y_val, val_pred))

    # Feature importance
    importance = dict(zip(feature_cols, model.feature_importances_.tolist()))
    # Sort by importance
    importance = dict(
        sorted(importance.items(), key=lambda x: x[1], reverse=True)[:15]
    )

    # Recursive multi-step forecast
    last_known = df.iloc[-1].copy()
    predictions = []
    last_date = df["date"].iloc[-1]

    for step in range(horizon):
        # Build feature vector for next day
        feat_dict = {}

        # Update lags: shift everything forward
        for lag in range(n_lags, 1, -1):
            feat_dict[f"lag_{lag}"] = last_known.get(f"lag_{lag - 1}", 0)
        feat_dict["lag_1"] = last_known.get("target", last_known.get("cases", 0))

        # Rolling stats (approximate using recent predictions)
        recent_cases = list(case_series[-13:]) + predictions
        recent = np.array(recent_cases[-14:], dtype=float)
        for window in [7, 14]:
            w = min(window, len(recent))
            feat_dict[f"rolling_{window}d_mean"] = float(np.mean(recent[-w:]))
            feat_dict[f"rolling_{window}d_std"] = float(np.std(recent[-w:])) if w > 1 else 0.0
            feat_dict[f"rolling_{window}d_min"] = float(np.min(recent[-w:]))
            feat_dict[f"rolling_{window}d_max"] = float(np.max(recent[-w:]))

        # Growth features
        if len(predictions) > 0:
            feat_dict["day_change"] = predictions[-1] - (
                predictions[-2] if len(predictions) > 1 else case_series[-1]
            )
        else:
            feat_dict["day_change"] = float(np.diff(case_series[-2:]).item()) if len(case_series) > 1 else 0.0

        prev_week = recent[-7] if len(recent) >= 7 else recent[0]
        feat_dict["week_change_ratio"] = (
            recent[-1] / prev_week if prev_week > 0 else 1.0
        )

        # Calendar features
        next_date = last_date + timedelta(days=step + 1)
        feat_dict["day_of_week"] = next_date.weekday()
        feat_dict["month"] = next_date.month
        feat_dict["is_weekend"] = 1 if next_date.weekday() in (5, 6) else 0

        # Predict
        X_next = np.array([[feat_dict.get(c, 0) for c in feature_cols]])
        pred = float(model.predict(X_next)[0])
        pred = max(0.0, pred)  # Non-negative cases
        predictions.append(pred)

        # Update last_known for next step
        for k, v in feat_dict.items():
            last_known[k] = v
        last_known["target"] = pred
        last_known["cases"] = pred

    # Generate forecast dates
    forecast_dates = [
        (last_date + timedelta(days=i + 1)).strftime("%Y-%m-%d")
        for i in range(horizon)
    ]

    # Uncertainty estimation (naive: based on validation residuals)
    val_residuals_std = float(np.std(y_val - val_pred))
    lower = [max(0, p - 1.96 * val_residuals_std) for p in predictions]
    upper = [p + 1.96 * val_residuals_std for p in predictions]

    return ForecastOutput(
        dates=forecast_dates,
        predicted=predictions,
        lower_bound=lower,
        upper_bound=upper,
        model_name="XGBoost",
        rmse=rmse,
        mae=mae,
        feature_importance=importance,
    )


# ---------------------------------------------------------------------------
# Prophet Forecaster (optional dependency)
# ---------------------------------------------------------------------------

def forecast_prophet(
    case_series: np.ndarray,
    dates: list[str] | None = None,
    horizon: int = 14,
) -> ForecastOutput | None:
    """Forecast using Facebook Prophet (optional dependency).

    Args:
        case_series: Array of daily case counts.
        dates: Optional list of ISO date strings.
        horizon: Forecast horizon in days.

    Returns:
        ForecastOutput or None if Prophet is not installed.
    """
    try:
        from prophet import Prophet
    except ImportError:
        logger.warning(
            "Prophet not installed. Skipping Prophet forecast. "
            "Install with: pip install prophet"
        )
        return None

    # Prepare Prophet dataframe
    if dates is not None:
        ds = pd.to_datetime(dates)
    else:
        ds = pd.date_range(end=datetime.today(), periods=len(case_series), freq="D")

    df = pd.DataFrame({"ds": ds, "y": case_series.astype(float)})

    # Remove zeros/negatives for better Prophet fit
    df["y"] = df["y"].clip(lower=0)

    # Validation: hold out last 14 days
    train_df = df.iloc[:-14] if len(df) > 28 else df
    val_df = df.iloc[-14:] if len(df) > 28 else None

    # Fit Prophet
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = Prophet(
            growth="linear",
            seasonality_mode="multiplicative",
            changepoint_prior_scale=0.05,
            yearly_seasonality=False,
            weekly_seasonality=True,
            daily_seasonality=False,
        )
        model.fit(train_df)

    # Validation metrics
    rmse = None
    mae = None
    if val_df is not None:
        val_pred = model.predict(val_df[["ds"]])
        val_y = val_df["y"].values
        val_yhat = val_pred["yhat"].values
        rmse = float(np.sqrt(mean_squared_error(val_y, val_yhat)))
        mae = float(mean_absolute_error(val_y, val_yhat))

    # Forecast
    future = model.make_future_dataframe(periods=horizon)
    forecast = model.predict(future)

    # Extract forecast period
    fc = forecast.iloc[-horizon:]
    forecast_dates = fc["ds"].dt.strftime("%Y-%m-%d").tolist()
    predicted = fc["yhat"].clip(lower=0).tolist()
    lower = fc["yhat_lower"].clip(lower=0).tolist()
    upper = fc["yhat_upper"].clip(lower=0).tolist()

    return ForecastOutput(
        dates=forecast_dates,
        predicted=predicted,
        lower_bound=lower,
        upper_bound=upper,
        model_name="Prophet",
        rmse=rmse,
        mae=mae,
    )


# ---------------------------------------------------------------------------
# Ensemble Forecaster
# ---------------------------------------------------------------------------

def forecast_ensemble(
    case_series: np.ndarray,
    dates: list[str] | None = None,
    horizon: int = 14,
    n_lags: int = 14,
) -> tuple[ForecastOutput, list[ForecastOutput]]:
    """Run ensemble forecast combining XGBoost and Prophet.

    Uses inverse-RMSE weighting: models with lower validation RMSE
    get higher weight in the ensemble.

    Args:
        case_series: Array of daily case counts.
        dates: Optional list of ISO date strings.
        horizon: Forecast horizon in days.
        n_lags: Number of lag features for XGBoost.

    Returns:
        Tuple of (ensemble_forecast, list_of_individual_forecasts).
    """
    individual_forecasts: list[ForecastOutput] = []

    # XGBoost (always available)
    try:
        xgb_forecast = forecast_xgboost(
            case_series, dates, horizon=horizon, n_lags=n_lags,
        )
        individual_forecasts.append(xgb_forecast)
        logger.info("XGBoost forecast complete: RMSE=%.2f", xgb_forecast.rmse or 0)
    except Exception as e:
        logger.error("XGBoost forecast failed: %s", e)

    # Prophet (optional)
    try:
        prophet_forecast = forecast_prophet(case_series, dates, horizon=horizon)
        if prophet_forecast is not None:
            individual_forecasts.append(prophet_forecast)
            logger.info("Prophet forecast complete: RMSE=%.2f", prophet_forecast.rmse or 0)
    except Exception as e:
        logger.warning("Prophet forecast failed: %s", e)

    if not individual_forecasts:
        raise RuntimeError("All forecasting models failed")

    # Single model → return directly
    if len(individual_forecasts) == 1:
        fc = individual_forecasts[0]
        ensemble = ForecastOutput(
            dates=fc.dates,
            predicted=fc.predicted,
            lower_bound=fc.lower_bound,
            upper_bound=fc.upper_bound,
            model_name=f"Ensemble ({fc.model_name} only)",
            rmse=fc.rmse,
            mae=fc.mae,
            feature_importance=fc.feature_importance,
        )
        return ensemble, individual_forecasts

    # Inverse-RMSE weighting
    rmses = []
    for fc in individual_forecasts:
        rmses.append(fc.rmse if fc.rmse and fc.rmse > 0 else 1e6)

    inv_rmses = [1.0 / r for r in rmses]
    total_weight = sum(inv_rmses)
    weights = [w / total_weight for w in inv_rmses]

    logger.info(
        "Ensemble weights: %s",
        {fc.model_name: f"{w:.2%}" for fc, w in zip(individual_forecasts, weights)},
    )

    # Weighted average
    n = len(individual_forecasts[0].dates)
    ensemble_pred = np.zeros(n)
    ensemble_lower = np.zeros(n)
    ensemble_upper = np.zeros(n)

    for fc, w in zip(individual_forecasts, weights):
        pred = np.array(fc.predicted[:n])
        lower = np.array(fc.lower_bound[:n])
        upper = np.array(fc.upper_bound[:n])
        ensemble_pred += w * pred
        ensemble_lower += w * lower
        ensemble_upper += w * upper

    # Ensemble RMSE (weighted average of individual RMSEs)
    ensemble_rmse = sum(w * r for w, r in zip(weights, rmses))

    ensemble = ForecastOutput(
        dates=individual_forecasts[0].dates,
        predicted=ensemble_pred.tolist(),
        lower_bound=ensemble_lower.tolist(),
        upper_bound=ensemble_upper.tolist(),
        model_name=f"Ensemble ({'+'.join(fc.model_name for fc in individual_forecasts)})",
        rmse=ensemble_rmse,
        feature_importance=individual_forecasts[0].feature_importance,  # from XGBoost
    )

    return ensemble, individual_forecasts
