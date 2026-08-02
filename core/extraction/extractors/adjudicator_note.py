from __future__ import annotations

import re

from ..utils import extract_regex, extract_first

ADJUDICATOR_FOOTER = (
    r"(?=\s+Packet\s+MIB-\d+\s*/\s*page\b|$)"
)
def extract_adjudicator_note(text: str) -> dict:
    """Extract fields from a manual adjudicator note."""

    
    fields = {
        "case_id": extract_regex(
            r"\b(MIB-\d+)\b",
            text,
        ),
        "decision": extract_first(
            [
                r"Finding\s*:\s*(APPROVED|DENIED|NEEDS_REVIEW)\b",
                r"^\s*(APPROVED|DENIED|REVIEW)\s*$",
            ],
            text,
            flags=re.IGNORECASE | re.MULTILINE,
        ),
        "reason": extract_regex(
            r"Reason\s*:\s*(.+?)" + ADJUDICATOR_FOOTER,
            text,
            flags=re.IGNORECASE | re.DOTALL,
        ),
    }

    return {
        "document_type": "adjudicator_note",
        "fields": fields,
    }