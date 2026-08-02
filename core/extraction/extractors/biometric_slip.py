from __future__ import annotations

from ..utils import extract_regex
import re

OCR_SEPARATOR = r"[\s|‘’'\"`()\[\]{}:;,.=—–-]+"

BIOMETRIC_BOUNDARY = (
    rf"(?={OCR_SEPARATOR}(?:"
    r"Case\s+(?:ID|1D|10)|"
    r"Applicant|"
    r"Species\s+Match|"
    r"Biometric\s+confidence|"
    r"Observed\s+flags|"
    r"SCAN\s+IMAGE|"
    r"Packet"
    r"Synthetic\s+hiring"
    r")\b|$)"
)


def extract_biometric_slip(text: str) -> dict:
    """Extract fields from Form B-13 biometric scan slips."""

    fields = {
        "case_id": extract_regex(
            r"Case\s+(?:ID|1D|10)\s*[:;.-]?\s*"
            r"([A-Z0-9-]+)",
            text,
        ),

        "applicant": extract_regex(
            r"Applicant\s*[:;.-]?\s*"
            r"(.+?)"
            + BIOMETRIC_BOUNDARY,
            text,
            flags=re.IGNORECASE | re.DOTALL,
        ),

        "species_match": extract_regex(
            r"Species\s+Match\s*[:;.-]?\s*"
            r"(.+?)"
            + BIOMETRIC_BOUNDARY,
            text,
            flags=re.IGNORECASE | re.DOTALL,
        ),

        "biometric_confidence": extract_regex(
            r"Biometric\s+confidence\s*[:;.-]?\s*"
            r"(\d{1,3}\s*%)",
            text,
        ),

        "observed_flags": extract_regex(
            r"Observed\s+flags\s*[:;.-]?\s*"
            # A slip can list more than one flag, pipe-delimited
            # (e.g. "illegible_biometrics|sponsor_mismatch"). The
            # character class must include "|" or multi-flag values
            # silently truncate to the first flag.
            r"([a-z0-9_|]+)",
            text,
        ),
    }

    return {
        "document_type": "biometric_slip",
        "fields": fields,
    }