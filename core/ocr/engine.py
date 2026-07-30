
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import cv2
import fitz
import numpy as np
import pytesseract
from PIL import Image
from pytesseract import Output

if TYPE_CHECKING:
    from core.quality.quality_router import QualityRoute


DEFAULT_RENDER_DPI = 300
DEFAULT_TESSERACT_CONFIG = "--oem 3 --psm 6"


@dataclass(frozen=True)
class OCRResult:
    """
    Result returned by the OCR engine.
    """

    text: str
    average_confidence: float | None
    word_count: int
    preprocessing_steps: list[str]
    render_dpi: int


def render_page(
    page: fitz.Page,
    dpi: int = DEFAULT_RENDER_DPI,
) -> np.ndarray:
    """
    Render a PDF page into a BGR OpenCV image.
    """

    if dpi <= 0:
        raise ValueError("dpi must be greater than zero")

    zoom = dpi / 72.0
    matrix = fitz.Matrix(zoom, zoom)

    pixmap = page.get_pixmap(
        matrix=matrix,
        alpha=False,
        colorspace=fitz.csRGB,
    )

    image = np.frombuffer(
        pixmap.samples,
        dtype=np.uint8,
    ).reshape(
        pixmap.height,
        pixmap.width,
        pixmap.n,
    )

    if pixmap.n == 4:
        return cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)

    return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)


def detect_orientation(image: np.ndarray) -> int:
    """
    Detect the clockwise rotation required to make text upright.

    Returns
    -------
    int
        One of 0, 90, 180, or 270.
    """

    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(rgb_image)

    try:
        orientation_data = pytesseract.image_to_osd(
            pil_image,
            output_type=Output.DICT,
        )
    except pytesseract.TesseractError:
        return 0

    rotation = int(orientation_data.get("rotate", 0))

    if rotation not in {0, 90, 180, 270}:
        return 0

    return rotation


def rotate_image(
    image: np.ndarray,
    clockwise_degrees: int,
) -> np.ndarray:
    """
    Rotate an image clockwise by a right angle.
    """

    if clockwise_degrees == 0:
        return image

    if clockwise_degrees == 90:
        return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)

    if clockwise_degrees == 180:
        return cv2.rotate(image, cv2.ROTATE_180)

    if clockwise_degrees == 270:
        return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)

    raise ValueError(
        "clockwise_degrees must be one of 0, 90, 180, or 270"
    )


def estimate_skew_angle(image: np.ndarray) -> float:
    """
    Estimate the page's small skew angle in degrees.
    """

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    inverted = cv2.bitwise_not(gray)

    thresholded = cv2.threshold(
        inverted,
        0,
        255,
        cv2.THRESH_BINARY | cv2.THRESH_OTSU,
    )[1]

    coordinates = np.column_stack(
        np.where(thresholded > 0)
    )

    if len(coordinates) < 100:
        return 0.0

    angle = cv2.minAreaRect(coordinates)[-1]

    if angle < -45:
        angle = 90 + angle
    else:
        angle = angle

    return float(-angle)


def deskew_image(
    image: np.ndarray,
    angle: float,
) -> np.ndarray:
    """
    Rotate an image by a small angle to correct skew.
    """

    if abs(angle) < 0.5:
        return image

    height, width = image.shape[:2]
    center = (width / 2.0, height / 2.0)

    matrix = cv2.getRotationMatrix2D(
        center=center,
        angle=angle,
        scale=1.0,
    )

    return cv2.warpAffine(
        image,
        matrix,
        (width, height),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )


def enhance_contrast(image: np.ndarray) -> np.ndarray:
    """
    Enhance local contrast using CLAHE.
    """

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8),
    )

    return clahe.apply(gray)


def denoise_image(image: np.ndarray) -> np.ndarray:
    """
    Reduce common scan noise while retaining character edges.
    """

    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    return cv2.fastNlMeansDenoising(
        gray,
        None,
        h=10,
        templateWindowSize=7,
        searchWindowSize=21,
    )


def threshold_image(image: np.ndarray) -> np.ndarray:
    """
    Convert an image to adaptive black-and-white text.
    """

    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    return cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        15,
    )


def preprocess_for_ocr(
    image: np.ndarray,
    route: QualityRoute,
) -> tuple[np.ndarray, list[str]]:
    """
    Apply only the transformations selected by the quality router.
    """

    processed = image.copy()
    steps: list[str] = []

    if route.detect_orientation:
        rotation = detect_orientation(processed)

        if route.rotate and rotation:
            processed = rotate_image(
                processed,
                clockwise_degrees=rotation,
            )
            steps.append(f"rotate_{rotation}")

    if route.deskew:
        skew_angle = estimate_skew_angle(processed)

        if abs(skew_angle) >= 0.5:
            processed = deskew_image(
                processed,
                angle=skew_angle,
            )
            steps.append(f"deskew_{skew_angle:.2f}")

    if route.enhance_contrast:
        processed = enhance_contrast(processed)
        steps.append("enhance_contrast")

    if route.denoise:
        processed = denoise_image(processed)
        steps.append("denoise")

    if route.threshold:
        processed = threshold_image(processed)
        steps.append("adaptive_threshold")

    return processed, steps


def extract_ocr_data(
    image: np.ndarray,
    config: str = DEFAULT_TESSERACT_CONFIG,
) -> tuple[str, float | None, int]:
    """
    Run Tesseract and return text plus basic confidence metrics.
    """

    if image.ndim == 3:
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    else:
        rgb_image = image

    pil_image = Image.fromarray(rgb_image)

    data = pytesseract.image_to_data(
        pil_image,
        config=config,
        output_type=Output.DICT,
    )

    words: list[str] = []
    confidences: list[float] = []

    for text, confidence in zip(
        data["text"],
        data["conf"],
        strict=True,
    ):
        text = text.strip()

        try:
            numeric_confidence = float(confidence)
        except (TypeError, ValueError):
            numeric_confidence = -1.0

        if not text:
            continue

        words.append(text)

        if numeric_confidence >= 0:
            confidences.append(numeric_confidence)

    text = " ".join(words).strip()

    average_confidence = (
        sum(confidences) / len(confidences)
        if confidences
        else None
    )

    return text, average_confidence, len(words)


def run_ocr(
    page: fitz.Page,
    route: QualityRoute,
    render_dpi: int = DEFAULT_RENDER_DPI,
    tesseract_config: str = DEFAULT_TESSERACT_CONFIG,
) -> OCRResult:
    """
    Render a PDF page, apply route-selected preprocessing, and run OCR.

    Parameters
    ----------
    page
        PyMuPDF page to process.
    route
        Instructions returned by route_page_quality().
    render_dpi
        Resolution used to render the PDF page.
    tesseract_config
        Tesseract command-line configuration.

    Returns
    -------
    OCRResult
        Extracted text and OCR diagnostics.
    """

    if not route.run_ocr:
        raise ValueError(
            "run_ocr() received a route for which run_ocr is False"
        )

    image = render_page(
        page=page,
        dpi=render_dpi,
    )

    processed_image, preprocessing_steps = preprocess_for_ocr(
        image=image,
        route=route,
    )

    text, average_confidence, word_count = extract_ocr_data(
        image=processed_image,
        config=tesseract_config,
    )

    return OCRResult(
        text=text,
        average_confidence=average_confidence,
        word_count=word_count,
        preprocessing_steps=preprocessing_steps,
        render_dpi=render_dpi,
    )