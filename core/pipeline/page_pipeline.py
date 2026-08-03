from __future__ import annotations

import cv2
import fitz

from core.classification.page_classifier import classify_document_type

from core.extraction.extractor import extract_fields

from core.ocr.engine import (
    MIN_DESKEW_ANGLE,
    render_page,
    rotate_image,
    run_ocr,
    run_ocr_image,
    select_ocr_profile,
)
from core.ocr.orientation import (
    OrientationCandidate,
    score_orientation_candidate,
    try_alternate_orientations,
)
from core.ocr.geometry import correct_page_geometry, measure_page_geometry
from core.ocr.tear_repair import align_text_lines, looks_torn, repair_tear
from core.ocr.text_layer import get_hidden_text, get_visible_text
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


def try_tear_repair(
    *,
    page,
    route,
    profile,
    rotation: int,
):
    """
    Evaluate repaired renders of the page and return the best candidate.

    Two repairs are offered because the damage comes in two shapes. Band
    repair handles a tear that cuts through glyphs, where the page border
    tracks the displacement. Line alignment handles a tear that falls between
    text lines, where the border test has nothing to corroborate against and
    every real boundary is rejected — the case on MIB-000039 page 2, whose
    field lines are each visibly offset yet yielded 0 of 37 confirmed
    boundaries.

    Both are scored like an alternate orientation, so the caller keeps
    whichever actually reads better and neither can make a page worse.
    """
    try:
        image = render_page(page=page, dpi=profile.render_dpi)

        if rotation:
            image = rotate_image(image, clockwise_degrees=rotation)

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        variants = []

        if looks_torn(gray):
            repaired, moved = repair_tear(gray)

            if moved:
                variants.append(repaired)

        aligned, lines_moved = align_text_lines(gray)

        if lines_moved:
            variants.append(aligned)

        best = None

        for variant in variants:
            ocr_result = run_ocr_image(
                image=variant,
                route=route,
                profile=profile,
                render_dpi=profile.render_dpi,
                apply_orientation_detection=False,
            )
            text = ocr_result.text or ""
            classification = classify_document_type(text)
            extraction = extract_fields(
                document_type=classification.document_type,
                text=text,
            )

            candidate = OrientationCandidate(
                rotation=rotation,
                text=text,
                ocr_result=ocr_result,
                classification=classification,
                extraction=extraction,
                score=score_orientation_candidate(
                    text=text,
                    ocr_result=ocr_result,
                    classification=classification,
                    extraction=extraction,
                ),
            )

            if best is None or candidate.score > best.score:
                best = candidate

        return best
    except Exception:
        # A failed repair must never abort the page.
        return None


def try_geometry_correction(
    *,
    page,
    route,
    profile,
    rotation: int,
):
    """
    Evaluate a rotation+shear-corrected render of the page as a candidate.

    `estimate_skew_angle` inside `run_ocr_image`'s preprocessing
    (`cv2.minAreaRect` over all thresholded foreground) measures rotation
    only and is unreliable on degraded pages -- see `core/ocr/geometry.py`.
    The line-detection-based replacement is more accurate, but was measured
    on the full training set to be a net *regression* when applied
    unconditionally to every page (-0.08 net score: it helps some pages and
    hurts others). Scored the same way as tear repair and orientation, so
    the caller keeps whichever actually reads better and this can never make
    a page worse -- the property that made the unconditional version unsafe.
    """
    try:
        image = render_page(page=page, dpi=profile.render_dpi)

        if rotation:
            image = rotate_image(image, clockwise_degrees=rotation)

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        geometry = measure_page_geometry(gray)

        if geometry is None:
            return None

        if (
            abs(geometry.rotation) < MIN_DESKEW_ANGLE
            and abs(geometry.shear) < MIN_DESKEW_ANGLE
        ):
            return None

        corrected = correct_page_geometry(image, geometry)

        ocr_result = run_ocr_image(
            image=corrected,
            route=route,
            profile=profile,
            render_dpi=profile.render_dpi,
            apply_orientation_detection=False,
        )
        text = ocr_result.text or ""
        classification = classify_document_type(text)
        extraction = extract_fields(
            document_type=classification.document_type,
            text=text,
        )

        return OrientationCandidate(
            rotation=rotation,
            text=text,
            ocr_result=ocr_result,
            classification=classification,
            extraction=extraction,
            score=score_orientation_candidate(
                text=text,
                ocr_result=ocr_result,
                classification=classification,
                extraction=extraction,
            ),
        )
    except Exception:
        # A failed correction must never abort the page.
        return None


def process_page(
    doc: fitz.Document,
    page: fitz.Page,
) -> dict:
    """
    Process one PDF page through quality routing, text selection,
    selective orientation retry, classification, and extraction.
    """
    # Only the visible text layer is evidence. White-on-white spans and text
    # outside the page crop carry prompt injection and fake answer keys.
    native_text = get_visible_text(page)
    hidden_text = get_hidden_text(page)

    assessment = assess_page_quality(
        doc=doc,
        page=page,
    )
    route = route_page_quality(assessment)

    ocr_result = None
    selected_rotation = 0
    orientation_retry_attempted = False
    tear_repair_attempted = False
    tear_repair_applied = False
    geometry_correction_attempted = False
    geometry_correction_applied = False

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

                initial_score = alternate.score

        # Scanline-tear retry. Repairing band displacement helps some pages a
        # lot and destroys readable text on others, so it is scored the same
        # way as an alternate orientation and kept only when it wins.
        tear_candidate = try_tear_repair(
            page=page,
            route=route,
            profile=selected_profile,
            rotation=selected_rotation,
        )

        current_score = initial_score

        if tear_candidate is not None:
            tear_repair_attempted = True

            if tear_candidate.score > current_score:
                tear_repair_applied = True
                selected_text = tear_candidate.text
                ocr_result = tear_candidate.ocr_result
                classification = tear_candidate.classification
                extraction = tear_candidate.extraction

                current_score = tear_candidate.score

        # Rotation+shear correction. Measured as a net regression when
        # applied unconditionally to every page (helps many, hurts some), so
        # -- like tear repair -- it is only kept when it scores better than
        # whatever the current best result is.
        geometry_candidate = try_geometry_correction(
            page=page,
            route=route,
            profile=selected_profile,
            rotation=selected_rotation,
        )

        if geometry_candidate is not None:
            geometry_correction_attempted = True

            if geometry_candidate.score > current_score:
                geometry_correction_applied = True
                selected_text = geometry_candidate.text
                ocr_result = geometry_candidate.ocr_result
                classification = geometry_candidate.classification
                extraction = geometry_candidate.extraction

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
        "hidden_text": hidden_text,
        "text_source": text_source,
        "classification": classification,
        "extraction": extraction,
        "case_id_candidates": case_id_candidates,
        "selected_rotation": selected_rotation,
        "orientation_retry_attempted": orientation_retry_attempted,
        "tear_repair_attempted": tear_repair_attempted,
        "tear_repair_applied": tear_repair_applied,
        "geometry_correction_attempted": geometry_correction_attempted,
        "geometry_correction_applied": geometry_correction_applied,
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