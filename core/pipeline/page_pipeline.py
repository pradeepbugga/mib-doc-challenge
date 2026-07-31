from core.quality.assessment import assess_page_quality
from core.quality.quality_router import route_page_quality
from core.ocr.engine import run_ocr
from core.classification.page_classifier import classify_document_type
from core.pipeline.case_assignment import resolve_page_case_assignment

import fitz  # PyMuPDF

def process_page(
    doc: fitz.Document,
    page: fitz.Page,
    filename_case_id: str | None = None,
) -> dict:
    """
    Process one PDF page through quality assessment, routing,
    text extraction, and document-type classification.
    """

    native_text = page.get_text("text")

    assessment = assess_page_quality(
        doc=doc,
        page=page,
    )

    route = route_page_quality(assessment)

    ocr_result = None

    if route.use_native_text:
        selected_text = native_text
        text_source = "native_text"

    elif route.run_ocr:
        ocr_result = run_ocr(
            page=page,
            route=route,
        )

        selected_text = ocr_result.text
        text_source = "ocr"

    else:
        selected_text = ""
        text_source = "none"

    case_assignment = resolve_page_case_assignment(
        page_number=page.number + 1,
        text=selected_text,
        filename_case_id=filename_case_id,
    )

    classification = classify_document_type(selected_text)

    return {
        "assessment": assessment,
        "route": route,
        "ocr_result": ocr_result,
        "text": selected_text,
        "text_source": text_source,
        "classification": classification,
        "case_assignment": case_assignment,
    }


def main() -> None:
    pdf_path = "./data/train/MIB-000009.pdf"

    with fitz.open(pdf_path) as doc:
        for page in doc:
            result = process_page(
                doc=doc,
                page=page,
            )

            print(f"Page {page.number + 1}:")
            print(f"  Quality: {result['assessment'].quality_class}")
            print(f"  Text source: {result['text_source']}")
            print(f"  OCR strategy: {result['route'].ocr_strategy.value}")
            print(f"  Classification: {result['classification']}")
            print(f"  OCR diagnostic: {result['ocr_result']}")


if __name__ == "__main__":
    main()