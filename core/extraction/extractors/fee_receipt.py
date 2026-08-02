from __future__ import annotations

import re

from ..utils import extract_regex, extract_first


FEE_RECEIPT_BOUNDARY = (
    r"(?=[\s|‘’'\"`]+(?:"
    r"Case\s+(?:ID|1D|10)|"
    r"Fee\s+Status|"
    r"Amount|"
    r"Waiver\s+Code|"
    r"Packet|"
    r"Synthetic\s+hiring"
    r")\b|$)"
)


def extract_fee_receipt(text: str) -> dict:
    """Extract fields from an MIB fee receipt."""

    fields = {
        "case_id": extract_regex(
            r"Case\s+(?:ID|1D|10)\s*[:;.-]?\s*"
            r"(MIB-\d+)",
            text,
            flags=re.IGNORECASE,
        ),

        "fee_status": extract_regex(
            r"Fee\s+Status\s*[:;.-]?\s*"
            r"([A-Za-z_]+)",
            text,
            flags=re.IGNORECASE,
        ),

        "amount": extract_regex(
            r"Amount\s*[:;.-]?\s*"
            r"\$?\s*([\d,]+(?:\.\d{2})?)",
            text,
            flags=re.IGNORECASE,
        ),

        "waiver_code": extract_regex(
            r"Waiver\s+Code\s*[:;.-]?\s*"
            r"([A-Z0-9/_-]+)",
            text,
            flags=re.IGNORECASE,
        ),
    }

    return {
        "document_type": "fee_receipt",
        "fields": fields,
    }