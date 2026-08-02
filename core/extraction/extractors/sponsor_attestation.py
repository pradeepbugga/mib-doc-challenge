from __future__ import annotations

from ..utils import extract_regex, extract_first
import re 

SPONSOR_BOUNDARY = (
    r"(?=\s+(?:"
    r"Case\s+ID|"
    r"Sponsor\s+ID|"
    r"Applicant|"
    r"Purpose|"
    r"Visa\s+Class|"
    r"Packet\s+MIB-\d+\s*/\s*page|"
    r"Synthetic\s+hiring"
    r")\b|$)"
)



def extract_sponsor_attestation(text: str) -> dict:
    fields = {
        "case_id": extract_regex(
            r"\b(MIB-\d+)\b",
            text,
        ),

        "sponsor_id": extract_first(
            [
                # Table or labeled layout
                r"Sponsor\s+ID\s*:?\s*"
                r"(SPN[- ]?\d(?:[\d ]*\d)?)",

                # Narrative letter
                r"Sponsor\s+"
                r"(SPN[- ]?\d(?:[\d ]*\d)?)"
                r"\s+attests\b",
            ],
            text,
            flags=re.IGNORECASE,
        ),

        "applicant": extract_first(
            [
                # Table or labeled layout
                r"Applicant\s*:?\s*(.+?)" + SPONSOR_BOUNDARY,

                # Narrative letter
                r"attests\s+that\s+(.+?)\s+is\s+expected\b",
            ],
            text,
        ),

        "purpose": extract_first(
    [
        r"Purpose\s*:?\s*(.+?)" + SPONSOR_BOUNDARY,
                r"is\s+expected\s+on\s+Earth\s+for\s+(.+?)\.",
            ],
            text,
            flags=re.IGNORECASE | re.DOTALL,
        ),
        "visa_class": extract_first(
            [
                # Table or labeled layout
                r"Visa\s+Class\s*:?\s*([A-Z0-9-]+)",

                # Narrative letter
                r"responsibility\s+for\s+class\s+"
                r"([A-Z0-9-]+)\s+compliance",
            ],
            text,
        ),
    }

    return {
        "document_type": "sponsor_attestation",
        "fields": fields,
    }