"""Pydantic Data Schemas for Epidemic Surveillance.

Defines the canonical data models used throughout the EpiAgent pipeline.
All inter-agent communication uses these schemas for type safety and validation.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Core data models
# ---------------------------------------------------------------------------

class SurveillanceRecord(BaseModel):
    """Single day/week surveillance observation."""
    date: str = Field(..., description="ISO 8601 date string (YYYY-MM-DD)")
    region: str = Field(..., description="Geographic region identifier")
    pathogen: str = Field(..., description="Pathogen name (e.g., 'influenza', 'covid-19')")
    new_cases: int = Field(..., ge=0, description="New cases reported this period")
    cumulative_cases: int = Field(..., ge=0, description="Total cumulative cases")
    new_deaths: int = Field(0, ge=0, description="New deaths reported this period")
    cumulative_deaths: int = Field(0, ge=0, description="Total cumulative deaths")
    population: int = Field(..., gt=0, description="Population at risk")
    source: str = Field(..., description="Data source identifier")
    metadata: dict[str, Any] = Field(default_factory=dict)


class SurveillanceDataset(BaseModel):
    """Collection of surveillance records with provenance metadata."""
    records: list[SurveillanceRecord]
    fetch_timestamp: str = Field(
        default_factory=lambda: datetime.utcnow().isoformat(),
        description="ISO 8601 datetime when data was fetched",
    )
    source_hash: str = Field("", description="SHA-256 hash of raw data for audit trail")

    @property
    def record_count(self) -> int:
        return len(self.records)

    @property
    def date_range(self) -> tuple[str, str] | None:
        if not self.records:
            return None
        dates = sorted(r.date for r in self.records)
        return (dates[0], dates[-1])

    @property
    def regions(self) -> list[str]:
        return sorted(set(r.region for r in self.records))

    @property
    def pathogens(self) -> list[str]:
        return sorted(set(r.pathogen for r in self.records))


# ---------------------------------------------------------------------------
# Data quality models
# ---------------------------------------------------------------------------

class DataQualityIssue(BaseModel):
    """Individual validation issue found during data quality checks."""
    severity: Literal["error", "warning", "info"]
    field: str
    record_index: int
    message: str
    value: Any = None


class DataQualityReport(BaseModel):
    """Output of epidemiological data validation."""
    quality_score: float = Field(..., ge=0.0, le=1.0)
    total_records: int
    valid_records: int
    issues: list[DataQualityIssue] = Field(default_factory=list)

    @property
    def is_acceptable(self) -> bool:
        return self.quality_score >= 0.7


# ---------------------------------------------------------------------------
# Metric models
# ---------------------------------------------------------------------------

class MetricWithCI(BaseModel):
    """Single metric with confidence interval."""
    value: float
    lower_ci: float
    upper_ci: float
    method: str


class EpiMetrics(BaseModel):
    """Bundle of epidemiological metrics."""
    cfr: MetricWithCI
    incidence_rate: MetricWithCI
    doubling_time: MetricWithCI | None = None
    attack_rate: MetricWithCI
    growth_rate: MetricWithCI | None = None


# ---------------------------------------------------------------------------
# Rt estimation models
# ---------------------------------------------------------------------------

class RtEstimate(BaseModel):
    """Rt estimation result for serialization."""
    dates: list[str]
    rt_mean: list[float]
    rt_lower: list[float]
    rt_upper: list[float]
    current_rt: float
    current_phase: Literal["growing", "declining", "stable", "unknown"]


# ---------------------------------------------------------------------------
# Forecast models
# ---------------------------------------------------------------------------

class ForecastResult(BaseModel):
    """ML forecast output."""
    dates: list[str]
    predicted: list[float]
    lower_bound: list[float]
    upper_bound: list[float]
    model_name: str
    rmse: float | None = None
    feature_importance: dict[str, float] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Alert models
# ---------------------------------------------------------------------------

class AlertLevel(str, Enum):
    """Public health alert level classification."""
    GREEN = "GREEN"     # Normal activity, Rt < 0.8
    YELLOW = "YELLOW"   # Elevated activity, 0.8 <= Rt < 1.0
    ORANGE = "ORANGE"   # High activity, 1.0 <= Rt < 1.5
    RED = "RED"         # Critical activity, Rt >= 1.5 or rapid growth


# ---------------------------------------------------------------------------
# Situation Report
# ---------------------------------------------------------------------------

class SituationReport(BaseModel):
    """Executive Situation Report (SitRep) — the final pipeline output."""
    report_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique report identifier",
    )
    generated_at: str = Field(
        default_factory=lambda: datetime.utcnow().isoformat(),
        description="ISO 8601 datetime of report generation",
    )
    region: str
    pathogen: str
    alert_level: AlertLevel
    executive_summary: str = Field(
        ..., description="1-paragraph executive summary for decision makers"
    )
    epi_metrics: EpiMetrics
    rt_estimate: RtEstimate
    forecasts: list[ForecastResult] = Field(default_factory=list)
    data_quality: DataQualityReport
    recommendations: list[str] = Field(default_factory=list)
    methodology_notes: str = Field(
        "",
        description="Technical methodology description for reproducibility",
    )
    reproducibility: dict[str, Any] = Field(
        default_factory=dict,
        description="Software versions, data hash, parameters for full reproducibility",
    )
