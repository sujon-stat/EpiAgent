"""CDC FluView ILINet Data Client.

Fetches Influenza-Like Illness (ILI) surveillance data from the
CMU Delphi Epidata API — the standard programmatic interface for
CDC ILINet data used by epidemiology researchers.

API Docs: https://cmu-delphi.github.io/delphi-epidata/
Endpoint: https://api.delphi.cmu.edu/epidata/fluview/

Data Available:
    - Weekly ILI percentages by HHS region and nationally
    - Patient counts, provider counts
    - Available from 1997 to present
"""

from __future__ import annotations

import logging
import time
from datetime import date, timedelta

import requests

from .synthetic import SurveillanceRecord

logger = logging.getLogger(__name__)

BASE_URL = "https://api.delphi.cmu.edu/epidata"

# Rate limiting: max 1 request per second
_last_request_time = 0.0


def _rate_limit():
    """Enforce 1 request per second rate limiting."""
    global _last_request_time
    elapsed = time.time() - _last_request_time
    if elapsed < 1.0:
        time.sleep(1.0 - elapsed)
    _last_request_time = time.time()


def epiweek_to_date(epiweek: int) -> str:
    """Convert CDC epiweek integer (YYYYWW) to ISO date string.

    The CDC epiweek starts on Sunday. We return the date of the
    Saturday (end of the epiweek) for consistency.

    Args:
        epiweek: Integer in YYYYWW format (e.g., 202301).

    Returns:
        ISO date string (YYYY-MM-DD).
    """
    year = epiweek // 100
    week = epiweek % 100

    # January 4th is always in week 1 (ISO standard)
    jan4 = date(year, 1, 4)
    # Find the Monday of ISO week 1
    iso_week1_monday = jan4 - timedelta(days=jan4.weekday())

    # CDC weeks start on Sunday, so subtract 1 day from Monday
    cdc_week1_start = iso_week1_monday - timedelta(days=1)

    # Target week start (Sunday)
    target_start = cdc_week1_start + timedelta(weeks=week - 1)
    # End of week (Saturday)
    target_end = target_start + timedelta(days=6)

    return target_end.isoformat()


def fetch_fluview(
    regions: str = "nat",
    epiweeks: str = "202301-202352",
) -> list[SurveillanceRecord]:
    """Fetch ILINet data from CMU Delphi Epidata API.

    Args:
        regions: Comma-separated region codes.
            'nat' = national, 'hhs1'-'hhs10' = HHS regions,
            or 2-letter state codes.
        epiweeks: Epiweek range in YYYYWW-YYYYWW format.

    Returns:
        List of SurveillanceRecord objects. Empty list on error.
    """
    _rate_limit()

    url = f"{BASE_URL}/fluview/"
    params = {
        "regions": regions,
        "epiweeks": epiweeks,
    }

    try:
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()

        if data.get("result") != 1:
            logger.warning(
                "Delphi API returned non-success: %s",
                data.get("message", "Unknown error"),
            )
            return []

        records = []
        epi_data = data.get("epidata", [])

        for entry in epi_data:
            # Extract relevant fields
            epiweek = entry.get("epiweek", 0)
            region = entry.get("region", "unknown")
            num_ili = entry.get("num_ili", 0)
            num_patients = entry.get("num_patients", 0)

            # Convert epiweek to date
            date_str = epiweek_to_date(epiweek)

            # ILINet doesn't report deaths directly; set to 0
            # num_ili = number of ILI cases seen by sentinel providers
            record = SurveillanceRecord(
                date=date_str,
                region=region,
                pathogen="influenza",
                new_cases=int(num_ili) if num_ili else 0,
                cumulative_cases=0,  # ILINet reports weekly, not cumulative
                new_deaths=0,
                cumulative_deaths=0,
                population=330_000_000 if region == "nat" else 33_000_000,
                source="cdc_fluview",
            )
            records.append(record)

        logger.info(
            "Fetched %d records from CDC FluView (regions=%s, epiweeks=%s)",
            len(records), regions, epiweeks,
        )
        return records

    except requests.RequestException as e:
        logger.error("Failed to fetch CDC FluView data: %s", e)
        return []
    except (ValueError, KeyError) as e:
        logger.error("Failed to parse CDC FluView response: %s", e)
        return []


def fetch_fluview_meta() -> dict:
    """Fetch metadata about available FluView data.

    Returns:
        Dict with metadata or empty dict on error.
    """
    _rate_limit()

    url = f"{BASE_URL}/fluview_meta/"

    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        data = response.json()
        return data.get("epidata", {})
    except requests.RequestException as e:
        logger.error("Failed to fetch FluView metadata: %s", e)
        return {}
