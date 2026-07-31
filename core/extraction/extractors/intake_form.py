from ..utils import extract_regex
import re

INTAKE_BOUNDARY = (
    r"(?=\s+(?:"
    r"Case\s+ID|"
    r"Applicant|"
    r"Species\s+Code|"
    r"Home\s+World|"
    r"Visa\s+(?:Class|Close)|"
    r"Sponsor\s+ID|"
    r"Arrival\s+Date|"
    r"Declared\s+Purpose|"
    r"Packet"
    r")\b|$)"
)
INTAKE_PURPOSE_BOUNDARY = (
    r"(?=\s+(?:"
    r"Manual\s+correction\s*:|"
    r"SAMPLE\s+DENIAL|"
    r"SAMPLE\s+APPROVAL|"
    r"Packet\s+MIB-\d+\s*/\s*page"
    r")\b|$)"
)

def extract_intake_form(text: str) -> dict:
    """Extract fields from Form I-8090."""

    fields = {
        "case_id": extract_regex(
            r"Case\s+ID\s*:?\s*"
            r"(MIB-\d+)",
            text,
        ),

        "applicant": extract_regex(
            r"Applicant\s*:?\s*"
            r"(.+?)"
            + INTAKE_BOUNDARY,
            text,
        ),

       "species_code": extract_regex(
            r"Species\s+Code\s*:?\s*"
            r"([A-Z][A-Z0-9_]+)",
            text,
        ),


        "home_world": extract_regex(
            r"Home\s+World\s*:?\s*"
            r"(.+?)"
            + INTAKE_BOUNDARY,
            text,
        ),

        "visa_class": extract_regex(
            r"Visa\s+Class\s*:?\s*"
            r"([A-Z0-9-]+)"
            + INTAKE_BOUNDARY,
            text,
        ),

       "sponsor_id": extract_regex(
            r"Sponsor\s+ID\s*:?\s*"
            r"(SPN[- ]?\d+)",
            text,
        ),

       "arrival_date": extract_regex(
            r"Arrival\s+Date\s*:?\s*"
            r"(\d{4}-\d{2}-\d{2})",
            text,
        ),

        "declared_purpose": extract_regex(
            r"Declared\s+Purpose\s*:?\s*(.+?)"
            + INTAKE_PURPOSE_BOUNDARY,
            text,
            flags=re.IGNORECASE | re.DOTALL,
        ),
    }

    return {
        "document_type": "intake_form",
        "fields": fields,
    }