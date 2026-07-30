from ..utils import extract_regex


def extract_intake_form(text: str) -> dict:
    """
    Extract fields from an intake form.

    This stage only extracts values as they appear on the page.
    Normalization and conflict resolution occur later.
    """

    fields = {

        "applicant": extract_regex(
            r"(?:Applicant|Applicant Name|Name)\s*[:\-]\s*([^\n\r]+)",
            text,
        ),

        "case_id": extract_regex(
            r"(?:Case ID|Case Number|Packet ID)\s*[:\-]\s*([A-Z0-9\-]+)",
            text,
        ),

        "species": extract_regex(
            r"(?:Species|Species Code)\s*[:\-]\s*([^\n\r]+)",
            text,
        ),

        "home_world": extract_regex(
            r"(?:Home World|Planet|Origin)\s*[:\-]\s*([^\n\r]+)",
            text,
        ),

        "arrival_date": extract_regex(
            r"(?:Arrival Date|Date of Arrival)\s*[:\-]\s*"
            r"(\d{4}[-/]\d{1,2}[-/]\d{1,2}"
            r"|\d{1,2}[-/]\d{1,2}[-/]\d{2,4})",
            text,
        ),

        "purpose": extract_regex(
            r"(?:Purpose|Reason for Visit|Visit Purpose)\s*[:\-]\s*([^\n\r]+)",
            text,
        ),

        "visa_class": extract_regex(
            r"(?:Visa Class|Visa Type)\s*[:\-]\s*([A-Z]{2}-?\d+)",
            text,
        ),

        "sponsor_id": extract_regex(
            r"(?:Sponsor ID|Sponsor)\s*[:\-]\s*(SPN[- ]?\d+)",
            text,
        ),

        "registry_name": extract_regex(
            r"(?:Registry Name|Registry)\s*[:\-]\s*([^\n\r]+)",
            text,
        ),

        "registry_status": extract_regex(
            r"(?:Registry Status|Status)\s*[:\-]\s*([^\n\r]+)",
            text,
        ),

        "application_date": extract_regex(
            r"(?:Application Date|Submission Date|Date)\s*[:\-]\s*"
            r"(\d{4}[-/]\d{1,2}[-/]\d{1,2}"
            r"|\d{1,2}[-/]\d{1,2}[-/]\d{2,4})",
            text,
        ),

    }

    return {
        "document_type": "intake_form",
        "fields": fields,
    }