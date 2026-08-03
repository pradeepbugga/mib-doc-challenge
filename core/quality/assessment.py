from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import fitz  # PyMuPDF
import numpy as np
from PIL import Image
from core.ocr.text_layer import get_visible_text
from core.quality.models import PageQualityAssessment

# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

MIN_NATIVE_WORDS = 15
MIN_NATIVE_CHARACTERS = 80

# If a single embedded image occupies this fraction of the page,
# the page is probably a scan or image-based document.
LARGE_IMAGE_COVERAGE_THRESHOLD = 0.70

# Visual metric thresholds. These are starting points and should be
# adjusted after reviewing results across the training corpus.
LOW_CONTRAST_THRESHOLD = 25.0
LOW_SHARPNESS_THRESHOLD = 20.0

SUSPICIOUS_TEXT_PATTERNS = (
    r"\bsystem\s*:",
    r"\bignore\s+(the\s+)?visible\s+evidence\b",
    r"\bignore\s+(all\s+)?previous\s+instructions\b",
    r"\boutput\s+(this\s+)?answer\s+key\b",
    r"\bfake\s+answer\s+key\b",
    r"\bhidden\s+text\b",
    r"\bprompt\s+injection\b",
)




# ---------------------------------------------------------------------
# Text analysis
# ---------------------------------------------------------------------

def normalize_text(text: str) -> str:
    """Normalize whitespace without altering the underlying content."""
    return " ".join(text.split())


def detect_suspicious_text(text: str) -> tuple[bool, list[str]]:
    """
    Detect text-layer content that resembles hidden instructions,
    answer keys, or prompt injection.

    This does not prove that the text is hidden. It marks the page for
    visual/OCR-based inspection instead of trusting the native layer.
    """
    normalized = normalize_text(text).lower()
    matched_patterns: list[str] = []

    for pattern in SUSPICIOUS_TEXT_PATTERNS:
        if re.search(pattern, normalized, flags=re.IGNORECASE):
            matched_patterns.append(pattern)

    return bool(matched_patterns), matched_patterns


# ---------------------------------------------------------------------
# PDF geometry and image analysis
# ---------------------------------------------------------------------

def rectangle_area(rect: fitz.Rect) -> float:
    """Return the non-negative area of a PyMuPDF rectangle."""
    width = max(0.0, rect.width)
    height = max(0.0, rect.height)
    return width * height


def calculate_maximum_image_coverage(
    doc: fitz.Document,
    page: fitz.Page,
) -> tuple[int, float]:
    """
    Estimate how much of the page is occupied by the largest embedded image.

    Returns
    -------
    tuple[int, float]
        Number of embedded image objects and maximum page coverage from
        any single image placement.
    """
    page_area = rectangle_area(page.rect)

    if page_area <= 0:
        return 0, 0.0

    images = page.get_images(full=True)
    maximum_coverage = 0.0

    seen_placements: set[tuple[int, float, float, float, float]] = set()

    for image in images:
        xref = image[0]

        try:
            placements = page.get_image_rects(xref)
        except Exception:
            placements = []

        for rect in placements:
            placement_key = (
                xref,
                round(rect.x0, 2),
                round(rect.y0, 2),
                round(rect.x1, 2),
                round(rect.y1, 2),
            )

            if placement_key in seen_placements:
                continue

            seen_placements.add(placement_key)

            coverage = rectangle_area(rect) / page_area
            maximum_coverage = max(maximum_coverage, coverage)

    return len(images), min(maximum_coverage, 1.0)


# ---------------------------------------------------------------------
# Page rendering and visual metrics
# ---------------------------------------------------------------------

def render_page_as_grayscale(
    page: fitz.Page,
    dpi: int = 150,
) -> np.ndarray:
    """
    Render a PDF page as an 8-bit grayscale NumPy array.
    """
    zoom = dpi / 72.0
    matrix = fitz.Matrix(zoom, zoom)

    pixmap = page.get_pixmap(
        matrix=matrix,
        colorspace=fitz.csGRAY,
        alpha=False,
    )

    image = Image.frombytes(
        "L",
        (pixmap.width, pixmap.height),
        pixmap.samples,
    )

    return np.asarray(image, dtype=np.float32)


def calculate_visual_contrast(gray_image: np.ndarray) -> float:
    """
    Estimate page contrast using pixel intensity standard deviation.

    Higher values generally indicate stronger separation between text
    and background.
    """
    if gray_image.size == 0:
        return 0.0

    return float(np.std(gray_image))


def calculate_visual_sharpness(gray_image: np.ndarray) -> float:
    """
    Estimate sharpness using the variance of horizontal and vertical
    pixel gradients.

    Higher values generally indicate sharper edges.
    """
    if gray_image.size == 0:
        return 0.0

    horizontal_gradient = np.diff(gray_image, axis=1)
    vertical_gradient = np.diff(gray_image, axis=0)

    horizontal_variance = float(np.var(horizontal_gradient))
    vertical_variance = float(np.var(vertical_gradient))

    return horizontal_variance + vertical_variance


# ---------------------------------------------------------------------
# Page-quality assessment
# ---------------------------------------------------------------------

def assess_page_quality(
    doc: fitz.Document,
    page: fitz.Page,
    render_dpi: int = 150,
) -> PageQualityAssessment:
    """
    Assess the extraction quality of a single PDF page.
    """
    native_text = page.get_text("text") or ""
    normalized_native_text = normalize_text(native_text)

    words = page.get_text("words") or []
    blocks = page.get_text("blocks") or []

    native_character_count = len(normalized_native_text)
    native_word_count = len(words)
    native_block_count = len(blocks)

    suspicious_text_layer, suspicious_matches = detect_suspicious_text(
        native_text
    )

    image_count, maximum_image_coverage = (
        calculate_maximum_image_coverage(doc, page)
    )

    has_large_page_image = (
        maximum_image_coverage >= LARGE_IMAGE_COVERAGE_THRESHOLD
    )

    appears_scanned = (
        has_large_page_image
        or (
            image_count > 0
            and native_word_count < MIN_NATIVE_WORDS
        )
    )

    rendered_page = render_page_as_grayscale(
        page,
        dpi=render_dpi,
    )

    visual_contrast = calculate_visual_contrast(rendered_page)
    visual_sharpness = calculate_visual_sharpness(rendered_page)

    # Sufficiency for trusting the text layer is judged on the *visible*
    # text, not the raw text -- get_visible_text already strips injected
    # spans (white-on-white, off-crop) the same way process_page's actual
    # extraction does, so a page with real content plus an injected
    # instruction should be judged on the real content alone. Confirmed on
    # MIB-000890 p2: raw native text tripped detect_suspicious_text, which
    # discarded the entire native layer and forced OCR -- but the visible
    # text alone (Registry Name / Orinax Miravara / ...) was clean and
    # complete; the OCR fallback produced a scrambled read of the same page.
    visible_text = get_visible_text(page)
    visible_character_count = len(normalize_text(visible_text))
    visible_word_count = len(visible_text.split())

    has_enough_native_text = (
        visible_word_count >= MIN_NATIVE_WORDS
        and visible_character_count >= MIN_NATIVE_CHARACTERS
    )

    low_contrast = visual_contrast < LOW_CONTRAST_THRESHOLD
    low_sharpness = visual_sharpness < LOW_SHARPNESS_THRESHOLD

    reasons: list[str] = []

    if suspicious_text_layer:
        reasons.append(
            "Native text contains suspicious instruction-like content."
        )

        for matched_pattern in suspicious_matches:
            reasons.append(
                f"Suspicious text pattern matched: {matched_pattern}"
            )

    if has_large_page_image:
        reasons.append(
            "A large embedded image covers most of the page."
        )

    if native_word_count < MIN_NATIVE_WORDS:
        reasons.append(
            f"Only {native_word_count} native words were extracted."
        )

    if native_character_count < MIN_NATIVE_CHARACTERS:
        reasons.append(
            f"Only {native_character_count} native characters were extracted."
        )

    if appears_scanned and low_contrast:
        reasons.append(
            f"Visual contrast is low: {visual_contrast:.2f}."
        )

    if appears_scanned and low_sharpness:
        reasons.append(
            f"Visual sharpness is low: {visual_sharpness:.2f}."
        )

# -------------------------------------------------------------
# Visual quality classification
# -------------------------------------------------------------



    if has_enough_native_text and not appears_scanned:
        quality_class = "digital_clean"

        reasons.append(
            "The page contains sufficient native text and does not "
            "appear to be dominated by a scanned image."
        )

    elif appears_scanned and not low_contrast and not low_sharpness:
        quality_class = "scan_readable"

        reasons.append(
            "The page appears scanned, but its visual quality should "
            "support OCR."
        )

    elif appears_scanned:
        quality_class = "scan_degraded"

        reasons.append(
            "The page appears scanned and has weak visual-quality "
            "measurements."
        )

    elif has_enough_native_text:
        quality_class = "mixed_content"

        reasons.append(
            "The page contains usable native text together with "
            "significant visual content."
        )

    else:
        quality_class = "insufficient_evidence"

        reasons.append(
            "The page does not contain enough reliable native text "
            "for classification."
        )


    # -------------------------------------------------------------
    # Text-layer trust and extraction routing
    # -------------------------------------------------------------

    # suspicious_text_layer no longer vetoes usability on its own --
    # has_enough_native_text is already computed on the visible (injection-
    # filtered) text above, so a page can carry an injected instruction and
    # still have its real content trusted. suspicious_text_layer is kept on
    # the assessment for diagnostics/reporting.
    native_text_usable = (
        has_enough_native_text
        and quality_class in {
            "digital_clean",
            "mixed_content",
        }
    )

    ocr_required = not native_text_usable


    if suspicious_text_layer:
        reasons.append(
            "The native text layer is not trusted, so visible-page OCR "
            "must be used."
        )

    elif native_text_usable:
        reasons.append(
            "The native text layer is suitable for downstream extraction."
        )

    else:
        reasons.append(
            "The native text layer is insufficient for downstream "
            "extraction, so OCR is required."
        )

    return PageQualityAssessment(
        page_number=page.number + 1,
        quality_class=quality_class,
        native_text_usable=native_text_usable,
        ocr_required=ocr_required,
        suspicious_text_layer=suspicious_text_layer,
        appears_scanned=appears_scanned,
        has_large_page_image=has_large_page_image,
        native_character_count=native_character_count,
        native_word_count=native_word_count,
        native_block_count=native_block_count,
        image_count=image_count,
        maximum_image_coverage=round(maximum_image_coverage, 4),
        visual_contrast=round(visual_contrast, 2),
        visual_sharpness=round(visual_sharpness, 2),
        low_contrast=low_contrast,
        low_sharpness=low_sharpness,
        reasons=reasons,
    )


# ---------------------------------------------------------------------
# Document-level processing
# ---------------------------------------------------------------------

def assess_pdf_quality(
    pdf_path: str | Path,
    render_dpi: int = 150,
) -> dict[str, Any]:
    """
    Assess every page in a PDF.
    """
    pdf_path = Path(pdf_path)

    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF does not exist: {pdf_path}")

    if pdf_path.suffix.lower() != ".pdf":
        raise ValueError(f"Expected a PDF file: {pdf_path}")

    page_assessments: list[dict[str, Any]] = []

    with fitz.open(pdf_path) as doc:
        if doc.needs_pass:
            raise ValueError(
                f"PDF is encrypted and requires a password: {pdf_path}"
            )

        for page in doc:
            assessment = assess_page_quality(
                doc=doc,
                page=page,
                render_dpi=render_dpi,
            )

            page_assessments.append(asdict(assessment))

        document_summary = summarize_document_quality(page_assessments)

        return {
            "file_name": pdf_path.name,
            "file_path": str(pdf_path.resolve()),
            "page_count": doc.page_count,
            "summary": document_summary,
            "pages": page_assessments,
        }


def summarize_document_quality(
    page_assessments: list[dict[str, Any]],
) -> dict[str, Any]:
    """Create a basic document-level summary."""
    class_counts: dict[str, int] = {}

    for page in page_assessments:
        quality_class = page["quality_class"]
        class_counts[quality_class] = (
            class_counts.get(quality_class, 0) + 1
        )

    ocr_pages = [
        page["page_number"]
        for page in page_assessments
        if page["ocr_required"]
    ]

    suspicious_pages = [
        page["page_number"]
        for page in page_assessments
        if page["suspicious_text_layer"]
    ]

    return {
        "quality_class_counts": class_counts,
        "ocr_required_pages": ocr_pages,
        "suspicious_text_layer_pages": suspicious_pages,
    }


# ---------------------------------------------------------------------
# Console display
# ---------------------------------------------------------------------

def print_report(report: dict[str, Any]) -> None:
    """Print a human-readable page-quality report."""
    print(f"\nFile: {report['file_name']}")
    print(f"Pages: {report['page_count']}")
    print("-" * 72)

    for page in report["pages"]:
        print(
            f"Page {page['page_number']}: "
            f"{page['quality_class']}"
        )

        print(
            f"  Native text usable: "
            f"{page['native_text_usable']}"
        )

        print(
            f"  OCR required: "
            f"{page['ocr_required']}"
        )

        print(
            f"  Suspicious text layer: "
            f"{page['suspicious_text_layer']}"
        )

        print(
            f"  Native words/chars: "
            f"{page['native_word_count']}/"
            f"{page['native_character_count']}"
        )

        print(
            f"  Maximum image coverage: "
            f"{page['maximum_image_coverage']:.1%}"
        )

        print(
            f"  Contrast: {page['visual_contrast']:.2f}"
        )

        print(
            f"  Sharpness: {page['visual_sharpness']:.2f}"
        )

        if page["reasons"]:
            print("  Reasons:")

            for reason in page["reasons"]:
                print(f"    - {reason}")

        print()

    print("Summary")
    print("-" * 72)

    print(
        "Quality classes:",
        report["summary"]["quality_class_counts"],
    )

    print(
        "OCR-required pages:",
        report["summary"]["ocr_required_pages"],
    )

    print(
        "Suspicious text-layer pages:",
        report["summary"]["suspicious_text_layer_pages"],
    )


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Assess PDF page quality and determine whether native text "
            "or OCR should be used."
        )
    )

    parser.add_argument(
        "pdf_path",
        type=Path,
        help="Path to the PDF file.",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path for the output JSON report.",
    )

    parser.add_argument(
        "--dpi",
        type=int,
        default=150,
        help="Rendering resolution used for visual analysis.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_arguments()

    try:
        report = assess_pdf_quality(
            pdf_path=args.pdf_path,
            render_dpi=args.dpi,
        )
    except Exception as exc:
        raise SystemExit(f"Page-quality assessment failed: {exc}") from exc

    print_report(report)

    if args.output is not None:
        args.output.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with args.output.open(
            "w",
            encoding="utf-8",
        ) as output_file:
            json.dump(
                report,
                output_file,
                indent=2,
                ensure_ascii=False,
            )

        print(f"\nJSON report saved to: {args.output}")


if __name__ == "__main__":
    main()