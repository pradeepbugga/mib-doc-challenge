from __future__ import annotations

from core.extraction.registry import EXTRACTORS


def extract_fields(
    document_type: str,
    text: str,
) -> dict:
    """
    Extract structured fields from a document page.

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
        return {
            "document_type": document_type,
            "fields": {},
        }

    return extractor(text)