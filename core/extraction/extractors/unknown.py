from __future__ import annotations

from .adjudicator_note import extract_adjudicator_note
from .biometric_slip import extract_biometric_slip
from .fee_receipt import extract_fee_receipt
from .intake_form import extract_intake_form
from .registry_extract import extract_registry_extract
from .sponsor_attestation import extract_sponsor_attestation

# Every specific-form extractor, tried in turn. Each one's regexes are keyed to
# that form's own labels ("Fee Status", "Registry Status", "Observed flags",
# ...), so running all of them against a page that failed to classify is safe:
# a label from one form essentially never appears on another, and a page that
# classified as "unknown" usually did so because OCR noise dropped the one or
# two cues the classifier needed, not because every field is unrecoverable.
FALLBACK_EXTRACTORS = (
    extract_fee_receipt,
    extract_intake_form,
    extract_registry_extract,
    extract_biometric_slip,
    extract_sponsor_attestation,
    extract_adjudicator_note,
)


def extract_unknown(text: str) -> dict:
    """
    Best-effort extraction for a page that didn't classify into a known form.

    Merges whatever fields any specific-form extractor manages to recover from
    the raw text, keeping the first non-null value found per field name.
    """
    fields: dict[str, str | None] = {}

    for extractor in FALLBACK_EXTRACTORS:
        for field_name, value in extractor(text).get("fields", {}).items():
            if value is not None and fields.get(field_name) is None:
                fields[field_name] = value

    return {
        "document_type": "unknown",
        "fields": fields,
    }
