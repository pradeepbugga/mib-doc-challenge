import re

from ..utils import extract_regex


REGISTRY_BOUNDARY = (
    r"(?=\s+(?:"
    r"Applicant|"
    r"Registry\s+Name|"
    r"Home\s+World|"
    r"Species\s+Code|"
    r"Registry\s+Status|"
    r"(?:Arrival|Arival|Amival|Antval)\s+Date|"
    r"REGISTRY\s+IMAGE|"
    r"Packet"
    r")\b|$)"
)


def extract_registry_extract(text: str) -> dict:
    fields = {
        "applicant": extract_regex(
            r"(?:Applicant|Registry\s+Name)\s*[:;.-]?\s*"
            r"(.+?)"
            + REGISTRY_BOUNDARY,
            text,
            flags=re.IGNORECASE | re.DOTALL,
        ),

        "home_world": extract_regex(
            r"Home\s+World\s*[:;.-]?\s*"
            r"(.+?)"
            + REGISTRY_BOUNDARY,
            text,
            flags=re.IGNORECASE | re.DOTALL,
        ),

        "species_code": extract_regex(
            r"Species\s+Code\s*[:;.-]?\s*"
            r"([A-Z][A-Z0-9_]+)",
            text,
        ),

        "registry_status": extract_regex(
            r"Registry\s+Status\s*[:;.-]?\s*"
            r"(.+?)"
            + REGISTRY_BOUNDARY,
            text,
            flags=re.IGNORECASE | re.DOTALL,
        ),

        "arrival_date": extract_regex(
            r"(?:Arrival|Arival|Amival|Antval)\s+Date"
            r"\s*[:;.-]?\s*"
            r"(\d{4}[-/.]\d{1,2}[-/.]\d{1,2})",
            text,
        ),
    }

    return {
        "document_type": "registry_extract",
        "fields": fields,
    }