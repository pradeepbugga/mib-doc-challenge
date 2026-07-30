from ..utils import extract_regex


def extract_intake_form(text: str) -> dict:
    """Extract fields from Form I-8090."""

    fields = {
        "case_id": extract_regex(
            r"Case\s+ID\s*:?\s*"
            r"(MIB-\d+)"
            r"(?=\s+Applicant\b|$)",
            text,
        ),
        "applicant": extract_regex(
            r"Applicant\s*:?\s*"
            r"(.+?)"
            r"(?=\s+Species\s+Code\b|$)",
            text,
        ),
        "species_code": extract_regex(
            r"Species\s+Code\s*:?\s*"
            r"([A-Z0-9_ -]+?)"
            r"(?=\s+Home\s+World\b|$)",
            text,
        ),
        "home_world": extract_regex(
            r"Home\s+World\s*:?\s*"
            r"(.+?)"
            r"(?=\s+Visa\s+Class\b|$)",
            text,
        ),
        "visa_class": extract_regex(
            r"Visa\s+Class\s*:?\s*"
            r"([A-Z0-9-]+)"
            r"(?=\s+Sponsor\s+ID\b|$)",
            text,
        ),
        "sponsor_id": extract_regex(
            r"Sponsor\s+ID\s*:?\s*"
            r"(SPN[- ]?\d+)"
            r"(?=\s+Arrival\s+Date\b|$)",
            text,
        ),
        "arrival_date": extract_regex(
            r"Arrival\s+Date\s*:?\s*"
            r"(\d{4}-\d{2}-\d{2})"
            r"(?=\s+Declared\s+Purpose\b|$)",
            text,
        ),
        "declared_purpose": extract_regex(
            r"Declared\s+Purpose\s*:?\s*"
            r"(.+?)"
            r"(?=\s+(?:SAMPLE\s+DENIAL|SAMPLE\s+APPROVAL|Packet)\b|$)",
            text,
        ),
    }

    return {
        "document_type": "intake_form",
        "fields": fields,
    }