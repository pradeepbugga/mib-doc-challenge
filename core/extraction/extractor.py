from __future__ import annotations

from core.extraction.fuzzy_labels import (
    FIELD_VOCABULARIES,
    extract_with_fuzzy_labels,
)
from core.extraction.registry import EXTRACTORS


def extract_fields(
    document_type: str,
    text: str,
) -> dict:
    """
    Extract structured fields from a document page.

    The registered extractor runs first. Any field it could not fill is then
    retried with fuzzy label matching, which tolerates the garbled labels that
    damaged scans produce ("Home Workt", "Foe Status") while the strict
    extractors require the label verbatim.

    Parameters
    ----------
    document_type
        Predicted page type.

    text
        OCR or native text.

    Returns
    -------
    dict
        Structured extracted fields.
    """

    extractor = EXTRACTORS.get(document_type)

    if extractor is None:
        result = {
            "document_type": document_type,
            "fields": {},
        }
    else:
        result = extractor(text)

    fields = result.get("fields", {})

    def is_empty(value) -> bool:
        return value is None or str(value).strip() == ""

    missing = {name for name, value in fields.items() if is_empty(value)}

    # Controlled-vocabulary fields are retried even when the strict extractor
    # produced something, because on a damaged scan what it produces is often
    # the value plus whatever noise followed it
    # ("xencbotany | --| vat PO Synthetic hiring..."). A term the fallback can
    # match against the ontology is strictly better than that.
    retryable = missing | (set(fields) & set(FIELD_VOCABULARIES))

    # An unclassified page has no declared field set, so let the fallback offer
    # everything it knows about rather than nothing.
    recovered = extract_with_fuzzy_labels(
        text=text,
        document_type=document_type,
        wanted=retryable or None,
    )

    for name, value in recovered.items():
        if is_empty(fields.get(name)):
            fields[name] = value
        elif name in FIELD_VOCABULARIES and value in FIELD_VOCABULARIES[name]:
            # Only override with an exact vocabulary term, never with more
            # unvalidated text.
            fields[name] = value

    result["fields"] = fields

    return result
