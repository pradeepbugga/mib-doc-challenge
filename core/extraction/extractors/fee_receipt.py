from ..utils import extract_regex


def extract_fee_receipt(text: str) -> dict:
    """
    Extract payment fields from a fee receipt.
    """

    fields = {
        "applicant": extract_regex(
            r"(?:Applicant|Applicant Name|Payer|Customer)\s*[:\-]\s*"
            r"([^\n\r]+)",
            text,
        ),
        "case_id": extract_regex(
            r"(?:Case ID|Case Number|Packet ID)\s*[:\-]\s*"
            r"([A-Z0-9\-]+)",
            text,
        ),
        "receipt_number": extract_regex(
            r"(?:Receipt Number|Receipt No\.?|Receipt ID|Transaction ID)\s*"
            r"[:#\-]?\s*([A-Z0-9\-]+)",
            text,
        ),
        "payment_date": extract_regex(
            r"(?:Payment Date|Transaction Date|Receipt Date|Date Paid|Date)\s*"
            r"[:\-]\s*(\d{4}[-/]\d{1,2}[-/]\d{1,2}"
            r"|\d{1,2}[-/]\d{1,2}[-/]\d{2,4})",
            text,
        ),
        "amount": extract_regex(
            r"(?:Amount Paid|Payment Amount|Total|Amount)\s*[:\-]?\s*"
            r"(?:[A-Z]{3}\s*)?[$€£]?\s*"
            r"(\d+(?:,\d{3})*(?:\.\d{2})?)",
            text,
        ),
        "currency": extract_regex(
            r"(?:Currency)\s*[:\-]\s*([A-Z]{3})",
            text,
        ),
        "payment_method": extract_regex(
            r"(?:Payment Method|Method|Paid By)\s*[:\-]\s*"
            r"([^\n\r]+)",
            text,
        ),
        "payment_status": extract_regex(
            r"(?:Payment Status|Transaction Status|Status)\s*[:\-]\s*"
            r"([^\n\r]+)",
            text,
        ),
        "fee_type": extract_regex(
            r"(?:Fee Type|Payment For|Description|Service)\s*[:\-]\s*"
            r"([^\n\r]+)",
            text,
        ),
    }

    return {
        "document_type": "fee_receipt",
        "fields": fields,
    }