from ..utils import extract_regex


def extract_biometric_slip(text: str) -> dict:
    """Extract fields from Form B-13 biometric scan slips."""

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
            r"(?=\s+Species\s+Match\b|$)",
            text,
        ),
        "species_match": extract_regex(
            r"Species\s+Match\s*:?\s*"
            r"([A-Z0-9_ -]+?)"
            r"(?=\s+Biometric\s+confidence\b|$)",
            text,
        ),
        "biometric_confidence": extract_regex(
            r"Biometric\s+confidence\s*:?\s*"
            r"(\d{1,3}%)"
            r"(?=\s+Observed\s+flags\b|$)",
            text,
        ),
        "observed_flags": extract_regex(
            r"Observed\s+flags\s*:?\s*"
            r"(.+?)"
            r"(?=\s+(?:SCAN\s+IMAGE|Packet)\b|$)",
            text,
        ),
    }

    return {
        "document_type": "biometric_slip",
        "fields": fields,
    }