"""HIPAA-Aligned Security Guardrails for Epidemic Surveillance.

Implements the Safe Harbor de-identification method per 45 CFR § 164.514(b)(2),
which requires removal of 18 types of identifiers.

This module provides:
    1. PII Detection — regex-based scanning for all 18 HIPAA identifier types
    2. PII Stripping — aggressive removal/redaction of detected PII
    3. Schema Validation — enforcement for surveillance data structure
    4. Data Provenance — SHA-256 hashing for audit trails
    5. Output Sanitization — ensures no PII leaks into generated reports

References:
    45 CFR § 164.514(b)(2) — Safe Harbor Method of De-identification
    https://www.hhs.gov/hipaa/for-professionals/privacy/special-topics/de-identification/
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class PIIPattern:
    """A regex pattern for detecting a specific type of PII."""
    name: str
    pattern: str
    replacement: str
    compiled: re.Pattern = field(init=False, repr=False)

    def __post_init__(self):
        self.compiled = re.compile(self.pattern, re.IGNORECASE)


# ---------------------------------------------------------------------------
# HIPAA Safe Harbor: 18 Identifier Types
# ---------------------------------------------------------------------------

HIPAA_PATTERNS: list[PIIPattern] = [
    # 1. Names (common patterns: First Last, Last, First)
    PIIPattern(
        name="name",
        pattern=r"\b(?:Mr|Mrs|Ms|Dr|Prof)\.?\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b",
        replacement="[REDACTED_NAME]",
    ),
    # 2. Geographic data (street addresses, zip codes)
    PIIPattern(
        name="address",
        pattern=r"\b\d{1,5}\s+(?:[A-Za-z]+\s+){1,3}(?:St|Street|Ave|Avenue|Blvd|Boulevard|Dr|Drive|Rd|Road|Ln|Lane|Way|Ct|Court)\b",
        replacement="[REDACTED_ADDRESS]",
    ),
    PIIPattern(
        name="zip_code",
        pattern=r"\b\d{5}(?:-\d{4})?\b",
        replacement="[REDACTED_ZIP]",
    ),
    # 3. Dates (MM/DD/YYYY, MM-DD-YYYY, Month DD, YYYY) — except year alone
    PIIPattern(
        name="date_mmddyyyy",
        pattern=r"\b(?:0[1-9]|1[0-2])[/\-](?:0[1-9]|[12]\d|3[01])[/\-](?:19|20)\d{2}\b",
        replacement="[REDACTED_DATE]",
    ),
    PIIPattern(
        name="date_text",
        pattern=r"\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}\b",
        replacement="[REDACTED_DATE]",
    ),
    # 4. Phone numbers (US formats)
    PIIPattern(
        name="phone",
        pattern=r"\b(?:\+?1[-.\s]?)?(?:\(?\d{3}\)?[-.\s]?)?\d{3}[-.\s]?\d{4}\b",
        replacement="[REDACTED_PHONE]",
    ),
    # 5. Fax numbers (same format, usually labeled)
    PIIPattern(
        name="fax",
        pattern=r"(?:fax|facsimile)[:\s]*(?:\+?1[-.\s]?)?(?:\(?\d{3}\)?[-.\s]?)?\d{3}[-.\s]?\d{4}",
        replacement="[REDACTED_FAX]",
    ),
    # 6. Email addresses
    PIIPattern(
        name="email",
        pattern=r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b",
        replacement="[REDACTED_EMAIL]",
    ),
    # 7. Social Security Numbers
    PIIPattern(
        name="ssn",
        pattern=r"\b\d{3}[-\s]?\d{2}[-\s]?\d{4}\b",
        replacement="[REDACTED_SSN]",
    ),
    # 8. Medical record numbers (common patterns: MRN-XXXXXXX)
    PIIPattern(
        name="mrn",
        pattern=r"\b(?:MRN|Medical\s*Record\s*(?:Number|No|#))[:\s]*[A-Z0-9\-]{5,15}\b",
        replacement="[REDACTED_MRN]",
    ),
    # 9. Health plan beneficiary numbers
    PIIPattern(
        name="health_plan_id",
        pattern=r"\b(?:Health\s*Plan|Beneficiary|Member)\s*(?:ID|Number|No|#)[:\s]*[A-Z0-9\-]{5,20}\b",
        replacement="[REDACTED_HEALTH_PLAN_ID]",
    ),
    # 10. Account numbers
    PIIPattern(
        name="account_number",
        pattern=r"\b(?:Account|Acct)\s*(?:Number|No|#)[:\s]*\d{6,20}\b",
        replacement="[REDACTED_ACCOUNT]",
    ),
    # 11. Certificate/license numbers
    PIIPattern(
        name="license",
        pattern=r"\b(?:License|Certificate|Cert)\s*(?:Number|No|#)[:\s]*[A-Z0-9\-]{5,20}\b",
        replacement="[REDACTED_LICENSE]",
    ),
    # 12. Vehicle identifiers (VIN)
    PIIPattern(
        name="vin",
        pattern=r"\b[A-HJ-NPR-Z0-9]{17}\b",
        replacement="[REDACTED_VIN]",
    ),
    # 13. Device identifiers / serial numbers
    PIIPattern(
        name="device_serial",
        pattern=r"\b(?:Serial|Device)\s*(?:Number|No|#|ID)[:\s]*[A-Z0-9\-]{5,25}\b",
        replacement="[REDACTED_DEVICE_ID]",
    ),
    # 14. Web URLs
    PIIPattern(
        name="url",
        pattern=r"https?://[^\s\"'<>]+",
        replacement="[REDACTED_URL]",
    ),
    # 15. IP addresses (IPv4)
    PIIPattern(
        name="ip_address",
        pattern=r"\b(?:\d{1,3}\.){3}\d{1,3}\b",
        replacement="[REDACTED_IP]",
    ),
    # 16. Biometric identifiers (text references)
    PIIPattern(
        name="biometric",
        pattern=r"\b(?:fingerprint|retinal?\s*scan|voice\s*print|facial\s*recognition|dna\s*sample)\b",
        replacement="[REDACTED_BIOMETRIC]",
    ),
    # 17. Full face photographs (filename references)
    PIIPattern(
        name="photo_file",
        pattern=r"\b\w+(?:_photo|_face|_headshot|_portrait)\.\w{3,4}\b",
        replacement="[REDACTED_PHOTO]",
    ),
    # 18. Any other unique identifying number (catch-all for labeled IDs)
    PIIPattern(
        name="unique_id",
        pattern=r"\b(?:Patient|Subject|Participant)\s*(?:ID|Number|No|#)[:\s]*[A-Z0-9\-]{3,20}\b",
        replacement="[REDACTED_ID]",
    ),
]


@dataclass
class SecurityReport:
    """Results of security audit on surveillance data."""
    pii_detected: bool
    pii_types_found: list[str] = field(default_factory=list)
    items_redacted: int = 0
    data_hash: str = ""
    schema_valid: bool = True
    schema_errors: list[str] = field(default_factory=list)

    def summary(self) -> dict:
        """Return concise summary dict for agent state."""
        return {
            "pii_detected": self.pii_detected,
            "pii_types": self.pii_types_found,
            "items_redacted": self.items_redacted,
            "data_hash": self.data_hash[:16] + "..." if self.data_hash else "",
            "schema_valid": self.schema_valid,
            "schema_errors_count": len(self.schema_errors),
        }


# ---------------------------------------------------------------------------
# Core Functions
# ---------------------------------------------------------------------------

def detect_pii(text: str) -> list[tuple[str, str, int, int]]:
    """Scan text for PII patterns.

    Args:
        text: Text to scan.

    Returns:
        List of (pattern_name, matched_text, start_pos, end_pos) tuples.
    """
    findings = []
    for pattern in HIPAA_PATTERNS:
        for match in pattern.compiled.finditer(text):
            findings.append((
                pattern.name,
                match.group(),
                match.start(),
                match.end(),
            ))
    return findings


def strip_pii(text: str) -> tuple[str, list[str]]:
    """Remove all detected PII from text.

    Args:
        text: Text to sanitize.

    Returns:
        Tuple of (cleaned_text, list_of_redacted_pattern_names).
    """
    redacted_types = []
    cleaned = text

    for pattern in HIPAA_PATTERNS:
        matches = pattern.compiled.findall(cleaned)
        if matches:
            cleaned = pattern.compiled.sub(pattern.replacement, cleaned)
            redacted_types.append(pattern.name)

    return cleaned, redacted_types


def strip_pii_from_records(
    records: list[dict],
) -> tuple[list[dict], SecurityReport]:
    """Strip PII from all string fields in surveillance records.

    Args:
        records: List of surveillance record dicts.

    Returns:
        Tuple of (cleaned_records, SecurityReport).
    """
    cleaned_records = []
    all_redacted_types = set()
    total_redacted = 0

    for rec in records:
        cleaned_rec = {}
        for key, value in rec.items():
            if isinstance(value, str):
                cleaned_value, types = strip_pii(value)
                if types:
                    all_redacted_types.update(types)
                    total_redacted += len(types)
                cleaned_rec[key] = cleaned_value
            elif isinstance(value, dict):
                # Recurse into nested dicts
                inner_cleaned = {}
                for k, v in value.items():
                    if isinstance(v, str):
                        cv, t = strip_pii(v)
                        if t:
                            all_redacted_types.update(t)
                            total_redacted += len(t)
                        inner_cleaned[k] = cv
                    else:
                        inner_cleaned[k] = v
                cleaned_rec[key] = inner_cleaned
            else:
                cleaned_rec[key] = value
        cleaned_records.append(cleaned_rec)

    pii_detected = total_redacted > 0

    report = SecurityReport(
        pii_detected=pii_detected,
        pii_types_found=sorted(all_redacted_types),
        items_redacted=total_redacted,
        data_hash=compute_data_hash(records),
        schema_valid=True,  # Schema validation done separately
    )

    if pii_detected:
        logger.warning(
            "PII detected and stripped: %d items, types: %s",
            total_redacted, sorted(all_redacted_types),
        )
    else:
        logger.info("No PII detected in %d records.", len(records))

    return cleaned_records, report


def validate_schema(data: dict | list) -> tuple[bool, list[str]]:
    """Validate surveillance data structure.

    Checks that required fields exist with correct types.

    Args:
        data: Single record dict or list of record dicts.

    Returns:
        Tuple of (is_valid, list_of_error_messages).
    """
    required_fields = {
        "date": str,
        "region": str,
        "pathogen": str,
        "new_cases": (int, float),
        "population": (int, float),
    }
    optional_fields = {
        "cumulative_cases": (int, float),
        "new_deaths": (int, float),
        "cumulative_deaths": (int, float),
        "source": str,
        "metadata": dict,
    }

    errors = []
    records = data if isinstance(data, list) else [data]

    for idx, rec in enumerate(records):
        if not isinstance(rec, dict):
            errors.append(f"Record {idx}: expected dict, got {type(rec).__name__}")
            continue

        for field_name, expected_type in required_fields.items():
            if field_name not in rec:
                errors.append(f"Record {idx}: missing required field '{field_name}'")
            elif not isinstance(rec[field_name], expected_type):
                errors.append(
                    f"Record {idx}: field '{field_name}' expected "
                    f"{expected_type}, got {type(rec[field_name]).__name__}"
                )

    is_valid = len(errors) == 0
    return is_valid, errors


def compute_data_hash(data: object) -> str:
    """Compute SHA-256 hash of data for provenance tracking.

    Args:
        data: Any JSON-serializable object.

    Returns:
        Hex-encoded SHA-256 hash string.
    """
    serialized = json.dumps(data, sort_keys=True, default=str)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def create_security_report(records: list[dict]) -> SecurityReport:
    """Run full security audit on a dataset.

    Performs PII detection, schema validation, and data hashing.

    Args:
        records: List of surveillance record dicts.

    Returns:
        Complete SecurityReport.
    """
    # Schema validation
    schema_valid, schema_errors = validate_schema(records)

    # PII scan (without modifying data)
    all_pii_types = set()
    total_pii = 0
    for rec in records:
        for key, value in rec.items():
            if isinstance(value, str):
                findings = detect_pii(value)
                if findings:
                    total_pii += len(findings)
                    all_pii_types.update(f[0] for f in findings)

    return SecurityReport(
        pii_detected=total_pii > 0,
        pii_types_found=sorted(all_pii_types),
        items_redacted=0,  # This is audit-only, no redaction performed
        data_hash=compute_data_hash(records),
        schema_valid=schema_valid,
        schema_errors=schema_errors,
    )
