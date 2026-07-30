import re

from ..utils import extract_regex


def extract_adjudication_note(text: str) -> dict:
    """
    Extract fields from an adjudication note.

    Extraction preserves the text as read from the page. Formatting and
    conflict correction should occur during normalization and corroboration.
    """

    fields = {
        "applicant": extract_regex(
            r"(?:Applicant|Applicant Name|Subject)\s*[:\-]\s*"
            r"([^\n\r]+)",
            text,
        ),
        "case_id": extract_regex(
            r"(?:Case ID|Case Number|Packet ID)\s*[:\-]\s*"
            r"([A-Z0-9\-]+)",
            text,
        ),
        "decision": extract_regex(
            r"(?:Decision|Disposition|Outcome)\s*[:\-]\s*"
            r"(approved|denied|pending|none|needs[ _-]?review)",
            text,
        ),
        "risk_flag": extract_regex(
            r"(?:Risk Flag|Risk Status|Risk Level)\s*[:\-]\s*"
            r"([^\n\r]+)",
            text,
        ),
        "adjudicator": extract_regex(
            r"(?:Adjudicator|Reviewed By|Officer)\s*[:\-]\s*"
            r"([^\n\r]+)",
            text,
        ),
        "adjudication_date": extract_regex(
            r"(?:Adjudication Date|Decision Date|Review Date|Date)\s*[:\-]\s*"
            r"(\d{4}[-/]\d{1,2}[-/]\d{1,2}"
            r"|\d{1,2}[-/]\d{1,2}[-/]\d{2,4})",
            text,
        ),
        "reason": extract_regex(
            r"(?:Reason|Rationale|Notes?|Comments?)\s*[:\-]\s*"
            r"(.+?)(?=\n\s*[A-Za-z][A-Za-z ]{1,30}\s*[:\-]|\Z)",
            text,
            flags=re.IGNORECASE | re.MULTILINE | re.DOTALL,
        ),
    }

    return {
        "document_type": "adjudication_note",
        "fields": fields,
    }