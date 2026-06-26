"""Epidemiological Data Validator.

Validates surveillance data for epidemiological plausibility BEFORE it reaches
the analysis engines. This is the critical "data quality firewall" that prevents
garbage-in-gospel-out failures.

Validation Checks:
    1. Non-negativity: cases >= 0, deaths >= 0
    2. Biological plausibility: deaths <= cases (CFR ∈ [0, 1])
    3. Population denominator: population > 0
    4. Temporal spike detection: week-over-week change > 300% flagged
    5. Completeness scoring: > 20% missing data degrades confidence
    6. Date consistency: monotonically increasing, no future dates
    7. Logical consistency: cumulative >= new (per day)

The validator produces a DataQualityReport with a composite quality score
Q ∈ [0, 1] that propagates through the pipeline to modulate confidence
in downstream outputs.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, date
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ValidationIssue:
    """A single data quality issue found during validation."""
    severity: str  # 'error', 'warning', 'info'
    field: str
    record_index: int
    message: str
    value: Any = None


@dataclass
class DataQualityReport:
    """Results of epidemiological data validation.

    Attributes:
        quality_score: Composite quality score in [0, 1]. Higher is better.
        total_records: Number of records examined.
        valid_records: Number of records passing all checks.
        issues: List of all validation issues found.
        is_acceptable: True if quality_score >= 0.7 (configurable threshold).
    """
    quality_score: float
    total_records: int
    valid_records: int
    issues: list[ValidationIssue] = field(default_factory=list)
    is_acceptable: bool = True

    @property
    def error_count(self) -> int:
        return sum(1 for i in self.issues if i.severity == "error")

    @property
    def warning_count(self) -> int:
        return sum(1 for i in self.issues if i.severity == "warning")

    def summary(self) -> dict:
        """Return a concise summary dict for agent state."""
        return {
            "quality_score": round(self.quality_score, 3),
            "total_records": self.total_records,
            "valid_records": self.valid_records,
            "errors": self.error_count,
            "warnings": self.warning_count,
            "is_acceptable": self.is_acceptable,
        }


# ---------------------------------------------------------------------------
# Spike detection threshold
# ---------------------------------------------------------------------------
_DEFAULT_SPIKE_THRESHOLD = 3.0       # 300% week-over-week increase
_DEFAULT_COMPLETENESS_THRESHOLD = 0.8  # At least 80% non-missing records
_DEFAULT_QUALITY_THRESHOLD = 0.7       # Minimum acceptable quality score


def validate_surveillance_data(
    records: list[dict],
    *,
    spike_threshold: float = _DEFAULT_SPIKE_THRESHOLD,
    completeness_threshold: float = _DEFAULT_COMPLETENESS_THRESHOLD,
    quality_threshold: float = _DEFAULT_QUALITY_THRESHOLD,
) -> DataQualityReport:
    """Validate a list of surveillance records for epidemiological plausibility.

    Args:
        records: List of surveillance record dicts. Expected keys:
            date, region, pathogen, new_cases, cumulative_cases,
            new_deaths, cumulative_deaths, population, source.
        spike_threshold: Maximum acceptable fold-change in new_cases
            between consecutive records (default 3.0 = 300%).
        completeness_threshold: Minimum fraction of non-missing records
            required (default 0.8 = 80%).
        quality_threshold: Minimum quality_score for is_acceptable (default 0.7).

    Returns:
        DataQualityReport with composite quality score and all issues found.
    """
    if not records:
        return DataQualityReport(
            quality_score=0.0,
            total_records=0,
            valid_records=0,
            issues=[ValidationIssue("error", "records", -1, "Empty dataset")],
            is_acceptable=False,
        )

    issues: list[ValidationIssue] = []
    total = len(records)
    record_valid = [True] * total  # Track per-record validity

    # -----------------------------------------------------------------------
    # Pass 1: Per-record checks
    # -----------------------------------------------------------------------
    for idx, rec in enumerate(records):
        # 1. Non-negativity check
        for fld in ("new_cases", "cumulative_cases", "new_deaths", "cumulative_deaths"):
            val = rec.get(fld)
            if val is not None and val < 0:
                issues.append(ValidationIssue(
                    "error", fld, idx,
                    f"Negative value: {fld}={val}. Epidemiological counts cannot be negative.",
                    value=val,
                ))
                record_valid[idx] = False

        # 2. Biological plausibility: deaths cannot exceed cases
        new_cases = rec.get("new_cases", 0) or 0
        new_deaths = rec.get("new_deaths", 0) or 0
        if new_deaths > new_cases and new_cases > 0:
            issues.append(ValidationIssue(
                "warning", "new_deaths", idx,
                f"Deaths ({new_deaths}) exceed cases ({new_cases}). "
                f"Implied CFR={new_deaths / new_cases:.1%} > 100%. "
                "Possible reporting lag or data error.",
                value={"new_deaths": new_deaths, "new_cases": new_cases},
            ))

        cum_cases = rec.get("cumulative_cases", 0) or 0
        cum_deaths = rec.get("cumulative_deaths", 0) or 0
        if cum_deaths > cum_cases and cum_cases > 0:
            issues.append(ValidationIssue(
                "warning", "cumulative_deaths", idx,
                f"Cumulative deaths ({cum_deaths}) exceed cumulative cases ({cum_cases}).",
                value={"cumulative_deaths": cum_deaths, "cumulative_cases": cum_cases},
            ))

        # 3. Population denominator guard
        population = rec.get("population")
        if population is None or population <= 0:
            issues.append(ValidationIssue(
                "error", "population", idx,
                f"Invalid population={population}. Must be > 0 to prevent "
                "division-by-zero in incidence calculations.",
                value=population,
            ))
            record_valid[idx] = False

        # 4. Logical consistency: cumulative >= new
        if cum_cases < new_cases:
            issues.append(ValidationIssue(
                "warning", "cumulative_cases", idx,
                f"Cumulative cases ({cum_cases}) less than new cases ({new_cases}).",
                value={"cumulative_cases": cum_cases, "new_cases": new_cases},
            ))

        # 5. Date validation
        date_str = rec.get("date")
        if date_str:
            try:
                rec_date = datetime.fromisoformat(date_str).date()
                if rec_date > date.today():
                    issues.append(ValidationIssue(
                        "warning", "date", idx,
                        f"Future date detected: {date_str}. "
                        "Surveillance data should not contain future dates.",
                        value=date_str,
                    ))
            except (ValueError, TypeError):
                issues.append(ValidationIssue(
                    "error", "date", idx,
                    f"Invalid date format: {date_str}. Expected ISO 8601.",
                    value=date_str,
                ))
                record_valid[idx] = False
        else:
            issues.append(ValidationIssue(
                "error", "date", idx,
                "Missing date field.",
            ))
            record_valid[idx] = False

    # -----------------------------------------------------------------------
    # Pass 2: Temporal checks (across records)
    # -----------------------------------------------------------------------
    case_series = []
    for rec in records:
        val = rec.get("new_cases")
        case_series.append(val if val is not None else np.nan)

    case_arr = np.array(case_series, dtype=float)

    # 6. Temporal spike detection
    for i in range(1, len(case_arr)):
        prev = case_arr[i - 1]
        curr = case_arr[i]
        if np.isnan(prev) or np.isnan(curr) or prev <= 0:
            continue
        fold_change = curr / prev
        if fold_change > spike_threshold:
            issues.append(ValidationIssue(
                "warning", "new_cases", i,
                f"Temporal spike: {fold_change:.1f}x increase from day {i-1} "
                f"({prev:.0f}) to day {i} ({curr:.0f}). "
                f"Exceeds {spike_threshold:.0f}x threshold. "
                "May be a reporting artifact or data dump.",
                value={"fold_change": round(fold_change, 2)},
            ))

    # 7. Monotonicity check on cumulative fields
    for fld in ("cumulative_cases", "cumulative_deaths"):
        prev_val = None
        for idx, rec in enumerate(records):
            val = rec.get(fld)
            if val is not None and prev_val is not None:
                if val < prev_val:
                    issues.append(ValidationIssue(
                        "warning", fld, idx,
                        f"Non-monotonic {fld}: decreased from {prev_val} to {val}. "
                        "Cumulative counts should never decrease (possible data correction).",
                        value={"previous": prev_val, "current": val},
                    ))
            prev_val = val

    # 8. Date ordering
    dates_parsed = []
    for idx, rec in enumerate(records):
        try:
            dates_parsed.append(datetime.fromisoformat(rec.get("date", "")).date())
        except (ValueError, TypeError):
            dates_parsed.append(None)

    for i in range(1, len(dates_parsed)):
        if dates_parsed[i] is not None and dates_parsed[i - 1] is not None:
            if dates_parsed[i] < dates_parsed[i - 1]:
                issues.append(ValidationIssue(
                    "warning", "date", i,
                    f"Non-chronological dates: {dates_parsed[i]} follows {dates_parsed[i-1]}.",
                ))

    # -----------------------------------------------------------------------
    # Pass 3: Completeness scoring
    # -----------------------------------------------------------------------
    required_fields = [
        "date", "region", "pathogen", "new_cases", "population"
    ]
    missing_count = 0
    total_field_checks = total * len(required_fields)

    for idx, rec in enumerate(records):
        for fld in required_fields:
            if rec.get(fld) is None:
                missing_count += 1

    completeness = 1.0 - (missing_count / total_field_checks) if total_field_checks > 0 else 0.0

    if completeness < completeness_threshold:
        issues.append(ValidationIssue(
            "warning", "completeness", -1,
            f"Data completeness ({completeness:.1%}) below threshold "
            f"({completeness_threshold:.1%}). Missing {missing_count} values "
            f"across {total_field_checks} field checks.",
            value={"completeness": round(completeness, 3)},
        ))

    # -----------------------------------------------------------------------
    # Compute composite quality score
    # -----------------------------------------------------------------------
    valid_count = sum(record_valid)
    error_count = sum(1 for i in issues if i.severity == "error")
    warning_count = sum(1 for i in issues if i.severity == "warning")

    # Quality score components:
    #   - Record validity ratio (weight: 0.5)
    #   - Completeness ratio (weight: 0.3)
    #   - Warning penalty (weight: 0.2) — each warning reduces by 0.02, capped
    validity_ratio = valid_count / total if total > 0 else 0.0
    warning_penalty = min(warning_count * 0.02, 0.2)

    quality_score = (
        0.5 * validity_ratio
        + 0.3 * completeness
        + 0.2 * (1.0 - warning_penalty)
    )
    quality_score = max(0.0, min(1.0, quality_score))

    # If any errors exist, cap quality at 0.5
    if error_count > 0:
        quality_score = min(quality_score, 0.5)

    report = DataQualityReport(
        quality_score=quality_score,
        total_records=total,
        valid_records=valid_count,
        issues=issues,
        is_acceptable=quality_score >= quality_threshold,
    )

    logger.info(
        "Validation complete: score=%.3f, records=%d/%d valid, "
        "errors=%d, warnings=%d, acceptable=%s",
        quality_score, valid_count, total, error_count, warning_count,
        report.is_acceptable,
    )

    return report


def filter_valid_records(
    records: list[dict],
    report: DataQualityReport,
) -> list[dict]:
    """Return only records that passed all error-level checks.

    Args:
        records: Original list of surveillance record dicts.
        report: DataQualityReport from validate_surveillance_data().

    Returns:
        Filtered list of records with no error-level issues.
    """
    error_indices = {
        issue.record_index
        for issue in report.issues
        if issue.severity == "error" and issue.record_index >= 0
    }
    return [
        rec for idx, rec in enumerate(records)
        if idx not in error_indices
    ]
