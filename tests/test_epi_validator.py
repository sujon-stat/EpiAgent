"""Test suite for the Epidemiological Data Validator.

Validates:
    1. Negative cases → rejection (error)
    2. Deaths > cases → flagged (warning)
    3. Temporal spike detection → flagged
    4. All zeros → low completeness/quality
    5. Population = 0 → rejection
    6. Quality score computation
    7. Record filtering
"""

import pytest

from epiagent.validators.epi_validator import (
    validate_surveillance_data,
    filter_valid_records,
    DataQualityReport,
)


def _make_record(
    date="2024-01-15",
    new_cases=100,
    cumulative_cases=500,
    new_deaths=2,
    cumulative_deaths=10,
    population=1_000_000,
    region="test_region",
    pathogen="test_pathogen",
):
    """Helper to create a valid surveillance record dict."""
    return {
        "date": date,
        "region": region,
        "pathogen": pathogen,
        "new_cases": new_cases,
        "cumulative_cases": cumulative_cases,
        "new_deaths": new_deaths,
        "cumulative_deaths": cumulative_deaths,
        "population": population,
        "source": "test",
    }


class TestEpiValidator:
    """Tests for epidemiological data validation."""

    def test_valid_data_passes(self):
        """Clean, valid data should produce high quality score."""
        records = [
            _make_record(date=f"2024-01-{d:02d}", new_cases=100 + d)
            for d in range(1, 15)
        ]
        report = validate_surveillance_data(records)
        assert report.is_acceptable
        assert report.quality_score > 0.8
        assert report.valid_records == len(records)

    def test_negative_cases_rejected(self):
        """Negative case counts should trigger error and reduce quality."""
        records = [
            _make_record(new_cases=-500),
        ]
        report = validate_surveillance_data(records)
        assert report.error_count > 0
        assert report.quality_score <= 0.5  # Errors cap quality at 0.5
        # Check specific error message
        error_msgs = [i.message for i in report.issues if i.severity == "error"]
        assert any("Negative" in msg for msg in error_msgs)

    def test_deaths_exceed_cases_flagged(self):
        """Deaths > cases should trigger a warning."""
        records = [
            _make_record(new_cases=5, new_deaths=100),
        ]
        report = validate_surveillance_data(records)
        warning_msgs = [i.message for i in report.issues if i.severity == "warning"]
        assert any("exceed" in msg.lower() or "cfr" in msg.lower() for msg in warning_msgs)

    def test_zero_population_rejected(self):
        """Population = 0 should trigger an error."""
        records = [_make_record(population=0)]
        report = validate_surveillance_data(records)
        assert report.error_count > 0
        error_msgs = [i.message for i in report.issues if i.severity == "error"]
        assert any("population" in msg.lower() for msg in error_msgs)

    def test_temporal_spike_detected(self):
        """10x spike in cases should be flagged as potential artifact."""
        records = [
            _make_record(date="2024-01-01", new_cases=100),
            _make_record(date="2024-01-02", new_cases=100),
            _make_record(date="2024-01-03", new_cases=5000),  # 50x spike!
        ]
        report = validate_surveillance_data(records)
        warning_msgs = [i.message for i in report.issues if i.severity == "warning"]
        assert any("spike" in msg.lower() for msg in warning_msgs)

    def test_empty_dataset(self):
        """Empty dataset should produce quality score = 0."""
        report = validate_surveillance_data([])
        assert report.quality_score == 0.0
        assert not report.is_acceptable
        assert report.total_records == 0

    def test_missing_date_rejected(self):
        """Records without a date should trigger an error."""
        records = [{"new_cases": 100, "population": 1000000}]
        report = validate_surveillance_data(records)
        error_msgs = [i.message for i in report.issues if i.severity == "error"]
        assert any("date" in msg.lower() for msg in error_msgs)

    def test_filter_valid_records(self):
        """filter_valid_records should remove records with errors."""
        records = [
            _make_record(date="2024-01-01", new_cases=100),   # Valid
            _make_record(date="2024-01-02", new_cases=-50),    # Invalid
            _make_record(date="2024-01-03", new_cases=200),    # Valid
        ]
        report = validate_surveillance_data(records)
        filtered = filter_valid_records(records, report)
        assert len(filtered) == 2
        assert all(r["new_cases"] >= 0 for r in filtered)

    def test_quality_score_range(self):
        """Quality score should always be in [0, 1]."""
        # All valid
        records = [_make_record(date=f"2024-01-{d:02d}") for d in range(1, 10)]
        report = validate_surveillance_data(records)
        assert 0.0 <= report.quality_score <= 1.0

        # All invalid
        bad_records = [_make_record(new_cases=-1, population=0) for _ in range(5)]
        report = validate_surveillance_data(bad_records)
        assert 0.0 <= report.quality_score <= 1.0

    def test_non_monotonic_cumulative_flagged(self):
        """Decreasing cumulative counts should trigger a warning."""
        records = [
            _make_record(date="2024-01-01", cumulative_cases=100),
            _make_record(date="2024-01-02", cumulative_cases=200),
            _make_record(date="2024-01-03", cumulative_cases=150),  # Decrease!
        ]
        report = validate_surveillance_data(records)
        warning_msgs = [i.message for i in report.issues if i.severity == "warning"]
        assert any("monotonic" in msg.lower() or "decrease" in msg.lower() for msg in warning_msgs)
