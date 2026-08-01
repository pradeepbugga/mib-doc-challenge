from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from core.ocr.engine import OCRResult, rotate_image


ALTERNATE_ROTATIONS = (90, 180, 270)


@dataclass(frozen=True)
class OrientationCandidate:
    """
    Downstream result for one explicit page orientation.

    rotation is clockwise relative to the rendered PDF page.
    """

    rotation: int
    text: str
    ocr_result: OCRResult
    classification: object
    extraction: dict
    score: float


def count_non_null_fields(extraction: dict) -> int:
    """Count successfully extracted field values."""
    fields = extraction.get("fields", {})

    return sum(
        value is not None and str(value).strip() != ""
        for value in fields.values()
    )


def score_orientation_candidate(
    *,
    text: str,
    ocr_result: OCRResult,
    classification: object,
    extraction: dict,
) -> float:
    """
    Score an orientation by downstream usefulness.

    Document recognition and field recovery dominate. OCR confidence
    and text volume are tie-breakers.
    """
    document_type = getattr(
        classification,
        "document_type",
        "unknown",
    )
    classification_score = float(
        getattr(classification, "score", 0.0) or 0.0
    )
    classification_confidence = float(
        getattr(classification, "confidence", 0.0) or 0.0
    )
    ocr_confidence = float(
        ocr_result.average_confidence or 0.0
    )
    non_null_fields = count_non_null_fields(extraction)

    score = 0.0

    if document_type != "unknown":
        score += 25.0

    score += classification_score
    score += 10.0 * classification_confidence
    score += 6.0 * non_null_fields

    # Weak tie-breakers only.
    score += 0.02 * ocr_confidence
    score += 0.005 * min(len(text.strip()), 500)

    return score


def evaluate_orientation(
    *,
    original_image: np.ndarray,
    rotation: int,
    ocr_image_fn: Callable[[np.ndarray], OCRResult],
    classify_fn: Callable[[str], object],
    extract_fn: Callable[..., dict],
) -> OrientationCandidate:
    """Rotate and fully evaluate one candidate orientation."""
    rotated_image = rotate_image(
        image=original_image,
        clockwise_degrees=rotation,
    )

    ocr_result = ocr_image_fn(rotated_image)
    text = ocr_result.text or ""

    classification = classify_fn(text)

    document_type = getattr(
        classification,
        "document_type",
        "unknown",
    )

    extraction = extract_fn(
        document_type=document_type,
        text=text,
    )

    score = score_orientation_candidate(
        text=text,
        ocr_result=ocr_result,
        classification=classification,
        extraction=extraction,
    )

    return OrientationCandidate(
        rotation=rotation,
        text=text,
        ocr_result=ocr_result,
        classification=classification,
        extraction=extraction,
        score=score,
    )


def try_alternate_orientations(
    *,
    original_image: np.ndarray,
    ocr_image_fn: Callable[[np.ndarray], OCRResult],
    classify_fn: Callable[[str], object],
    extract_fn: Callable[..., dict],
    rotations: tuple[int, ...] = ALTERNATE_ROTATIONS,
) -> OrientationCandidate | None:
    """
    Try 90, 180, and 270 degrees and return the best candidate.

    The caller is expected to have already evaluated the original
    zero-degree orientation.
    """
    candidates: list[OrientationCandidate] = []

    for rotation in rotations:
        try:
            candidate = evaluate_orientation(
                original_image=original_image,
                rotation=rotation,
                ocr_image_fn=ocr_image_fn,
                classify_fn=classify_fn,
                extract_fn=extract_fn,
            )
        except Exception:
            # One failed orientation should not abort the page.
            continue

        candidates.append(candidate)

    if not candidates:
        return None

    return max(
        candidates,
        key=lambda candidate: candidate.score,
    )