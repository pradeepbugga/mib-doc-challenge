from ..utils import extract_regex

def extract_fee_receipt(text: str) -> dict:
    """Extract fields from an MIB fee receipt."""

    fields = {
        "case_id": extract_regex(
            r"Case\s+ID\s*:?\s*"
            r"(MIB-\d+)"
            r"(?=\s+Fee\s+Status\b|$)",
            text,
        ),
        "fee_status": extract_regex(
            r"Fee\s+Status\s*:?\s*"
            r"(.+?)"
            r"(?=\s+Amount\b|$)",
            text,
        ),
        "amount": extract_regex(
            r"Amount\s*:?\s*"
            r"\$?\s*([\d,]+(?:\.\d{2})?)"
            r"(?=\s+Waiver\s+Code\b|$)",
            text,
        ),
        "waiver_code": extract_regex(
            r"Waiver\s+Code\s*:?\s*"
            r"(.+?)"
            r"(?=\s+Packet\b|$)",
            text,
        ),
    }

    return {
        "document_type": "fee_receipt",
        "fields": fields,
    }