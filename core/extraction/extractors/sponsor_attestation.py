from ..utils import extract_regex


def extract_sponsor_attestation(text: str) -> dict:
    """
    Extract fields from a sponsor attestation page.

    Patterns support OCR text where line breaks between fields may be lost.
    """

    fields = {
        "sponsor_id": extract_regex(
            r"Sponsor\s*(?:ID|Identifier)?\s*:?\s*"
            r"(SPN[- ]?\d+)",
            text,
        ),
        "applicant": extract_regex(
            r"Applicant(?:\s+Name)?\s*:?\s*"
            r"(.+?)"
            r"(?=\s+(?:Purpose|Visa\s+Class|Sponsor\s+ID|Packet)\b|$)",
            text,
        ),
        "purpose": extract_regex(
            r"Purpose\s*:?\s*"
            r"(.+?)"
            r"(?=\s+(?:Visa\s+Class|Sponsor\s+ID|Applicant|Packet)\b|$)",
            text,
        ),
        "visa_class": extract_regex(
            r"Visa\s+Class\s*:?\s*"
            r"([A-Z]{1,4}[- ]?\d+)",
            text,
        ),
    }

    return {
        "document_type": "sponsor_attestation",
        "fields": fields,
    }