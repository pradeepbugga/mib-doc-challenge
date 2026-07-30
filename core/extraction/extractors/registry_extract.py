from ..patterns import *
from ..utils import extract_regex


def extract_registry_extract(text: str) -> dict:
    """
    Extract fields from a planetary registry extract.

    Supports both:
        Field: Value

    and:
        Field
        Value
    """

    fields = {
        "registry_name": extract_regex(
            r"Registry\s+Name\s*:?\s*"
            r"(.+?)"
            r"(?=\s+Home\s+World\b|$)",
            text,
        ),
        "home_world": extract_regex(
            r"Home\s+World\s*:?\s*"
            r"(.+?)"
            r"(?=\s+Species\s+Code\b|$)",
            text,
        ),
        "species_code": extract_regex(
            r"Species\s+Code\s*:?\s*"
            r"([A-Z0-9_ -]+?)"
            r"(?=\s+Registry\s+Status\b|$)",
            text,
        ),
        "registry_status": extract_regex(
            r"Registry\s+Status\s*:?\s*"
            r"(.+?)"
            r"(?=\s+Arrival\s+Date\b|$)",
            text,
        ),
        "arrival_date": extract_regex(
            r"Arrival\s+Date\s*:?\s*"
            r"(\d{4}[-/]\d{1,2}[-/]\d{1,2})",
            text,
        ),
    }

    return {
        "document_type": "registry_extract",
        "fields": fields,
    }