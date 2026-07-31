from __future__ import annotations

from core.quality.models import PageQualityAssessment
from enum import Enum
from typing import TYPE_CHECKING

from dataclasses import dataclass

if TYPE_CHECKING:
    # Replace this import with the actual location of your dataclass.
    from page_quality_check import PageQualityAssessment


class ExtractionMethod(str, Enum):
    """Primary method used to obtain trustworthy text from a page."""

    NATIVE_TEXT = "native_text"
    STANDARD_OCR = "standard_ocr"
    DEGRADED_SCAN_OCR = "degraded_scan_ocr"
    OCR_FALLBACK = "ocr_fallback"


class OCRStrategy(str, Enum):
    """OCR processing strategy selected by the quality router."""

    NONE = "none"
    STANDARD = "standard"
    DEGRADED_SCAN = "degraded_scan"
    SUSPICIOUS_TEXT_LAYER = "suspicious_text_layer"
    FALLBACK = "fallback"


@dataclass(frozen=True)
class QualityRoute:
    """
    Processing instructions selected from a PageQualityAssessment.

    The router decides what should happen next. It does not perform
    native-text extraction, image preprocessing, or OCR itself.
    """

    extraction_method: ExtractionMethod
    ocr_strategy: OCRStrategy

    use_native_text: bool
    render_page: bool
    run_ocr: bool

    ignore_native_text: bool

    detect_orientation: bool
    rotate: bool
    deskew: bool
    enhance_contrast: bool
    denoise: bool
    threshold: bool

    reason: str


def route_page_quality(
    assessment: PageQualityAssessment,
) -> QualityRoute:
    """
    Select a text-extraction route from a page-quality assessment.

    Parameters
    ----------
    assessment
        Output from the page-quality assessment stage.

    Returns
    -------
    QualityRoute
        Instructions for the next stage of the document pipeline.
    """

    # -------------------------------------------------------------
    # Route 1: trusted native PDF text
    # -------------------------------------------------------------

    if assessment.native_text_usable:
        return QualityRoute(
            extraction_method=ExtractionMethod.NATIVE_TEXT,
            ocr_strategy=OCRStrategy.NONE,
            use_native_text=True,
            render_page=False,
            run_ocr=False,
            ignore_native_text=False,
            detect_orientation=False,
            rotate=False,
            deskew=False,
            enhance_contrast=False,
            denoise=False,
            threshold=False,
            reason=(
                "The native PDF text layer is sufficiently populated "
                "and trusted."
            ),
        )

    # -------------------------------------------------------------
    # Route 2: suspicious or adversarial native text layer
    # -------------------------------------------------------------

    if assessment.suspicious_text_layer:
        return QualityRoute(
            extraction_method=ExtractionMethod.STANDARD_OCR,
            ocr_strategy=OCRStrategy.SUSPICIOUS_TEXT_LAYER,
            use_native_text=False,
            render_page=True,
            run_ocr=True,
            ignore_native_text=True,
            detect_orientation=True,
            rotate=True,
            deskew=True,
            enhance_contrast=assessment.low_contrast,
            denoise=assessment.low_sharpness,
            threshold=assessment.low_contrast,
            reason=(
                "The native text layer contains suspicious instruction-like "
                "content. The visible page must be rendered and OCR must be "
                "performed without using or merging the native text."
            ),
        )

    # -------------------------------------------------------------
    # Route 3: readable scanned page
    # -------------------------------------------------------------

    if assessment.quality_class == "scan_readable":
        return QualityRoute(
            extraction_method=ExtractionMethod.STANDARD_OCR,
            ocr_strategy=OCRStrategy.STANDARD,
            use_native_text=False,
            render_page=True,
            run_ocr=True,
            ignore_native_text=True,
            detect_orientation=True,
            rotate=True,
            deskew=True,
            enhance_contrast=False,
            denoise=False,
            threshold=False,
            reason=(
                "The page is image-based but has adequate visual quality "
                "for standard OCR."
            ),
        )

    # -------------------------------------------------------------
    # Route 4: degraded scanned page
    # -------------------------------------------------------------

    if assessment.quality_class == "scan_degraded":
        return QualityRoute(
            extraction_method=ExtractionMethod.DEGRADED_SCAN_OCR,
            ocr_strategy=OCRStrategy.DEGRADED_SCAN,
            use_native_text=False,
            render_page=True,
            run_ocr=True,
            ignore_native_text=True,
            detect_orientation=True,
            rotate=True,
            deskew=True,
            enhance_contrast=False, # we make this False because we want to enhance contrast in the OCR step, not here
            denoise=True,
            threshold=True,
            reason=(
                "The page is image-based and visually degraded. OCR should "
                "use preprocessing and may require multiple attempts."
            ),
        )

    # -------------------------------------------------------------
    # Route 5: mixed page whose native text was rejected
    # -------------------------------------------------------------

    if assessment.quality_class == "mixed_content":
        return QualityRoute(
            extraction_method=ExtractionMethod.STANDARD_OCR,
            ocr_strategy=OCRStrategy.STANDARD,
            use_native_text=False,
            render_page=True,
            run_ocr=True,
            ignore_native_text=True,
            detect_orientation=True,
            rotate=True,
            deskew=True,
            enhance_contrast=assessment.low_contrast,
            denoise=False,
            threshold=False,
            reason=(
                "The page contains mixed visual and textual content, but its "
                "native text was not accepted. OCR the rendered page."
            ),
        )

    # -------------------------------------------------------------
    # Route 6: insufficient or unrecognized evidence
    # -------------------------------------------------------------

    return QualityRoute(
        extraction_method=ExtractionMethod.OCR_FALLBACK,
        ocr_strategy=OCRStrategy.FALLBACK,
        use_native_text=False,
        render_page=True,
        run_ocr=True,
        ignore_native_text=True,
        detect_orientation=True,
        rotate=True,
        deskew=True,
        enhance_contrast=True,
        denoise=True,
        threshold=True,
        reason=(
            "The page does not provide enough trusted native text and does "
            "not match a more specific route. Use the OCR fallback pipeline."
        ),
    )