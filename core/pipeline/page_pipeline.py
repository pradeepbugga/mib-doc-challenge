from __future__ import annotations

import fitz

from core.classification.page_classifier import classify_document_type

from core.extraction.extractor import extract_fields

from core.ocr.engine import (
    render_page,
    run_ocr,
    run_ocr_image,
    select_ocr_profile,
)
from core.ocr.orientation import (
    score_orientation_candidate,
    try_alternate_orientations,
)
from core.pipeline.case_assignment import (
    extract_case_id_candidates,
)
from core.quality.assessment import assess_page_quality
from core.quality.quality_router import route_page_quality


def has_usable_fields(extraction: dict) -> bool:
    """Return whether extraction produced at least one non-null field."""
    fields = extraction.get("fields", {})

    return any(
        value is not None and str(value).strip() != ""
        for value in fields.values()
    )


def should_retry_orientation(
    *,
    classification,
    extraction: dict,
    ocr_result,
) -> bool:
    """
    Retry only when the original orientation appears unsuccessful.
    """
    if classification.document_type == "unknown":
        return True

    if not has_usable_fields(extraction):
        return True

    # Keep this threshold conservative. A low-confidence page that still
    # classified and extracted useful fields does not necessarily need retry.
    if (
        ocr_result is not None
        and ocr_result.average_confidence is not None
        and ocr_result.average_confidence < 35
    ):
        return True

    return False


def process_page(
    doc: fitz.Document,
    page: fitz.Page,
) -> dict:
    """
    Process one PDF page through quality routing, text selection,
    selective orientation retry, classification, and extraction.
    """
    native_text = page.get_text("text")

    assessment = assess_page_quality(
        doc=doc,
        page=page,
    )
    route = route_page_quality(assessment)

    ocr_result = None
    selected_rotation = 0
    orientation_retry_attempted = False

    if route.use_native_text:
        selected_text = native_text
        text_source = "native_text"

        classification = classify_document_type(
            selected_text
        )
        extraction = extract_fields(
            document_type=classification.document_type,
            text=selected_text,
        )

    elif route.run_ocr:
        selected_profile = select_ocr_profile(route)

        # Initial attempt using your existing OSD/preprocessing behavior.
        ocr_result = run_ocr(
            page=page,
            route=route,
            profile=selected_profile,
        )
        selected_text = ocr_result.text
        text_source = "ocr"

        classification = classify_document_type(
            selected_text
        )
        extraction = extract_fields(
            document_type=classification.document_type,
            text=selected_text,
        )

        initial_score = score_orientation_candidate(
            text=selected_text,
            ocr_result=ocr_result,
            classification=classification,
            extraction=extraction,
        )

        if should_retry_orientation(
            classification=classification,
            extraction=extraction,
            ocr_result=ocr_result,
        ):
            orientation_retry_attempted = True

            # Render once. All alternate rotations reuse this image.
            original_image = render_page(
                page=page,
                dpi=selected_profile.render_dpi,
            )

            def retry_ocr_image(image):
                return run_ocr_image(
                    image=image,
                    route=route,
                    profile=selected_profile,
                    render_dpi=selected_profile.render_dpi,
                    apply_orientation_detection=False,
                )

            alternate = try_alternate_orientations(
                original_image=original_image,
                ocr_image_fn=retry_ocr_image,
                classify_fn=classify_document_type,
                extract_fn=extract_fields,
            )

            if (
                alternate is not None
                and alternate.score > initial_score
            ):
                selected_rotation = alternate.rotation
                selected_text = alternate.text
                ocr_result = alternate.ocr_result
                classification = alternate.classification
                extraction = alternate.extraction

    else:
        selected_text = ""
        text_source = "none"

        classification = classify_document_type(
            selected_text
        )
        extraction = extract_fields(
            document_type=classification.document_type,
            text=selected_text,
        )

    case_id_candidates = extract_case_id_candidates(
        selected_text
    )

    return {
        "assessment": assessment,
        "route": route,
        "ocr_result": ocr_result,
        "text": selected_text,
        "text_source": text_source,
        "classification": classification,
        "extraction": extraction,
        "case_id_candidates": case_id_candidates,
        "selected_rotation": selected_rotation,
        "orientation_retry_attempted": orientation_retry_attempted,
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