from __future__ import annotations

from ..utils import extract_regex
import re

INTAKE_FIELD_BOUNDARY = (
    r"(?=\s+(?:"
    r"Case\s+(?:ID|1D|10)|"
    r"Applicant|"
    r"Species\s+Code|"
    r"Home\s+World|"
    r"Visa\s+(?:Class|Close)|"
    r"Sponsor\s+(?:ID|1D|10)|"
    r"(?:Arrival|Antval|Amival|Arnval|Ariival)\s+Date|"
    r"Declared\s+Purpose|"
    r"PASSPORT\s+IMAGE|"
    r"Manual\s+correction\s*:|"
    r"SAMPLE\s+(?:DENIAL|APPROVAL)|"
    r"Packet"
    r")\b|$)"
)

INTAKE_PURPOSE_BOUNDARY = (
    r"(?=\s+(?:"
    r"Manual\s+correction\s*:|"
    r"SAMPLE\s+(?:DENIAL|APPROVAL)|"
    r"Packet\s+MIB-\d+\s*/\s*page"
    r")\b|$)"
)

def extract_intake_form(text: str) -> dict:
    """Extract fields from Form I-8090."""

    fields = {
        "case_id": extract_regex(
            r"Case\s+(?:ID|1D|10)\s*[:;.-]?\s*"
            r"(M[I1]B-\d+)",
            text,
        ),

        "applicant": extract_regex(
            r"Applicant\s*[:;.-]?\s*"
            r"(.+?)"
            + INTAKE_FIELD_BOUNDARY,
            text,
            flags=re.IGNORECASE | re.DOTALL,
        ),

        "species_code": extract_regex(
            r"Species\s+Code\s*[:;.-]?\s*"
            r"([A-Z][A-Z0-9_]+)",
            text,
        ),

        "home_world": extract_regex(
            r"Home\s+World\s*[:;.-]?\s*"
            r"(.+?)"
            + INTAKE_FIELD_BOUNDARY,
            text,
            flags=re.IGNORECASE | re.DOTALL,
        ),

        "visa_class": extract_regex(
            r"Visa\s+(?:Class|Close)\s*[:;.-]?\s*"
            r"([A-Z0-9]+(?:[-.][A-Z0-9]+)?)",
            text,
        ),

        "sponsor_id": extract_regex(
            r"Sponsor\s+(?:ID|1D|10)\s*[:;.=]?\s*"
            r"(SPN[- ]?\d+)",
            text,
        ),

        "arrival_date": extract_regex(
            r"(?:Arrival|Antval|Amival|Arnval|Ariival)\s+Date"
            r"\s*[:;.-]?\s*"
            r"(\d{4}[-.]\d{2}[-.]\d{2})",
            text,
        ),

        "declared_purpose": extract_regex(
            r"Declared\s+Purpose\s*[:;.-]?\s*"
            r"(.+?)"
            + INTAKE_PURPOSE_BOUNDARY,
            text,
            flags=re.IGNORECASE | re.DOTALL,
        ),
    }

    return {
        "document_type": "intake_form",
        "fields": fields,
    }