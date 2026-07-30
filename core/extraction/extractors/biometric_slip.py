from ..utils import extract_regex


def extract_biometric_slip(text: str) -> dict:
    """
    Extract fields from a biometric collection or verification slip.
    """

    fields = {
        "applicant": extract_regex(
            r"(?:Applicant|Applicant Name|Subject|Name)\s*[:\-]\s*"
            r"([^\n\r]+)",
            text,
        ),
        "case_id": extract_regex(
            r"(?:Case ID|Case Number|Packet ID)\s*[:\-]\s*"
            r"([A-Z0-9\-]+)",
            text,
        ),
        "biometric_id": extract_regex(
            r"(?:Biometric ID|Biometric Reference|Capture ID|Record ID)\s*"
            r"[:\-]\s*([A-Z0-9\-]+)",
            text,
        ),
        "capture_date": extract_regex(
            r"(?:Capture Date|Collection Date|Biometric Date|Date)\s*[:\-]\s*"
            r"(\d{4}[-/]\d{1,2}[-/]\d{1,2}"
            r"|\d{1,2}[-/]\d{1,2}[-/]\d{2,4})",
            text,
        ),
        "biometric_type": extract_regex(
            r"(?:Biometric Type|Capture Type|Modality)\s*[:\-]\s*"
            r"([^\n\r]+)",
            text,
        ),
        "fingerprint_status": extract_regex(
            r"(?:Fingerprint Status|Fingerprints?|Print Status)\s*[:\-]\s*"
            r"([^\n\r]+)",
            text,
        ),
        "verification_status": extract_regex(
            r"(?:Verification Status|Match Status|Biometric Status|Status)\s*"
            r"[:\-]\s*([^\n\r]+)",
            text,
        ),
        "collection_site": extract_regex(
            r"(?:Collection Site|Capture Site|Location|Facility)\s*[:\-]\s*"
            r"([^\n\r]+)",
            text,
        ),
    }

    return {
        "document_type": "biometric_slip",
        "fields": fields,
    }