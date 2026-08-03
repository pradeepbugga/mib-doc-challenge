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


DEFAULT_RENDER_DPI = 200
DEFAULT_TESSERACT_CONFIG = "--oem 3 --psm 11"

MIN_DESKEW_ANGLE = 2.0

# Grey level at or above which a pixel is treated as blank page rather than
# ink. Real printed text on these packets renders very dark (as low as 41),
# while the adversarial white-on-white layer renders around 226-255 — visually
# invisible, but only a few grey levels off the background. CLAHE amplifies
# local contrast, so without this clamp it can pull that hidden layer up into
# readable glyphs and feed prompt-injection text straight into OCR output,
# bypassing the visible-text filter that protects the native text layer.
# Clamping first destroys the faint layer and also removes scan haze and
# half-toned grid rules that blur character segmentation.
FAINT_INK_FLOOR = 195

# On a genuinely faint scan the real ink itself can be washed out into the
# same 195-255 band the floor exists to erase -- confirmed on MIB-000045,
# MIB-000670, and MIB-000294 (field text measured at ~228, indistinguishable
# by absolute intensity from the injection band). A fixed floor cannot serve
# both cases at once. ADAPTIVE_FLOOR_MARGIN/CAP bound how far a per-page
# floor (see estimate_ink_floor) is allowed to rise above the global default,
# so a faint page can recover its own ink without opening the floor all the
# way up to where injection typically renders.
ADAPTIVE_FLOOR_MARGIN = 12
ADAPTIVE_FLOOR_CAP = 230

# A page needs at least this share of non-white pixels for its own histogram
# to be a trustworthy basis for a floor -- an almost-blank page has too little
# ink to estimate anything from, and should fall back to the global floor.
MIN_INK_FRACTION_FOR_ADAPTIVE_FLOOR = 0.005

@dataclass(frozen=True)
class OCRProfile:
    """Empirically selected OCR settings for the challenge documents."""

    name: str
    render_dpi: int
    tesseract_config: str
    use_clahe: bool


STANDARD_OCR_PROFILE = OCRProfile(
    name="standard",
    render_dpi=200,
    tesseract_config="--oem 3 --psm 11",
    use_clahe=False,
)

DEGRADED_OCR_PROFILE = OCRProfile(
    name="degraded_scan",
    render_dpi=200,
    tesseract_config="--oem 3 --psm 11",
    use_clahe=True,
)


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
    profile_name: str
    tesseract_config: str


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

    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

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


def suppress_faint_ink(
    image: np.ndarray,
    floor: int = FAINT_INK_FLOOR,
) -> np.ndarray:
    """
    Clamp near-white pixels to pure white and return a grayscale image.

    Run this before any contrast enhancement. See `FAINT_INK_FLOOR`.
    """

    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    suppressed = gray.copy()
    suppressed[suppressed >= floor] = 255

    return suppressed


def estimate_ink_floor(image: np.ndarray) -> int:
    """
    Derive a per-page faint-ink floor from the page's own intensity
    histogram, for pages where the global FAINT_INK_FLOOR would erase real
    but faint content along with it. See FAINT_INK_FLOOR and
    ADAPTIVE_FLOOR_MARGIN/CAP.

    Only ever raises the floor above the global default, never lowers it --
    a page whose own ink is already dark (the common case) must keep exactly
    the injection-suppression behavior that floor was tuned for.
    """

    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    non_white = gray[gray < 255]

    if non_white.size < gray.size * MIN_INK_FRACTION_FOR_ADAPTIVE_FLOOR:
        return FAINT_INK_FLOOR

    otsu_threshold, _ = cv2.threshold(
        non_white.reshape(-1, 1),
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU,
    )

    candidate = int(otsu_threshold) + ADAPTIVE_FLOOR_MARGIN

    return max(FAINT_INK_FLOOR, min(candidate, ADAPTIVE_FLOOR_CAP))


def enhance_contrast(image: np.ndarray) -> np.ndarray:
    """
    Enhance local contrast using CLAHE.
    """

    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

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
    profile: OCRProfile,
    apply_orientation_detection: bool = True,
) -> tuple[np.ndarray, list[str]]:
    """
    Apply orientation/deskew corrections plus the selected OCR profile.

    Benchmarking showed that denoising and adaptive thresholding generally
    reduced recognition quality. The primary profile therefore uses the
    rendered image directly, while the degraded-scan fallback adds only
    CLAHE contrast enhancement.
    """

    processed = image.copy()
    steps: list[str] = []

    if apply_orientation_detection and route.detect_orientation:
        rotation = detect_orientation(processed)

        if route.rotate and rotation:
            processed = rotate_image(
                processed,
                clockwise_degrees=rotation,
            )
            steps.append(f"rotate_{rotation}")

    if route.deskew:
        skew_angle = estimate_skew_angle(processed)

        if abs(skew_angle) >= MIN_DESKEW_ANGLE:
            processed = deskew_image(
                processed,
                angle=skew_angle,
            )
            steps.append(f"deskew_{skew_angle:.2f}")

    # Must run after orientation/deskew, which need the colour image, and
    # before CLAHE, which would otherwise amplify the faint layer. Only pages
    # already on the CLAHE path get an adaptive floor -- everything else
    # keeps the exact global-floor behavior the injection suppression was
    # tuned against.
    if profile.use_clahe:
        floor = estimate_ink_floor(processed)
        processed = suppress_faint_ink(processed, floor=floor)
        steps.append(f"suppress_faint_ink_{floor}")
        processed = enhance_contrast(processed)
        steps.append("clahe")
    else:
        processed = suppress_faint_ink(processed)
        steps.append("suppress_faint_ink")

    return processed, steps


def select_ocr_profile(route: QualityRoute) -> OCRProfile:
    """
    Select the empirically preferred OCR profile for a quality route.

    The degraded route uses CLAHE as the only image-enhancement fallback.
    Denoising and thresholding flags from older routes are intentionally not
    applied because the benchmark consistently found them harmful.
    """

    if route.enhance_contrast:
        return DEGRADED_OCR_PROFILE

    return STANDARD_OCR_PROFILE


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

def run_ocr_image(
    image: np.ndarray,
    route: QualityRoute,
    profile: OCRProfile | None = None,
    render_dpi: int | None = None,
    tesseract_config: str | None = None,
    apply_orientation_detection: bool = False,
) -> OCRResult:
    """
    OCR an already-rendered page image.

    This is used for orientation retries, where the caller has already
    rotated the image to a candidate orientation.

    Parameters
    ----------
    image
        BGR or grayscale OpenCV image.
    route
        Quality route controlling deskew and enhancement.
    profile
        Optional OCR profile override.
    render_dpi
        DPI associated with the rendered image.
    tesseract_config
        Optional Tesseract configuration override.
    apply_orientation_detection
        Whether to run Tesseract OSD before OCR. This should normally be
        False during explicit 90/180/270-degree retries.
    """
    if not route.run_ocr:
        raise ValueError(
            "run_ocr_image() received a route for which run_ocr is False"
        )

    selected_profile = profile or select_ocr_profile(route)
    selected_render_dpi = render_dpi or selected_profile.render_dpi
    selected_tesseract_config = (
        tesseract_config or selected_profile.tesseract_config
    )

    processed = image.copy()
    preprocessing_steps: list[str] = []

    # During explicit orientation retries, the caller already selected
    # the page rotation, so OSD should not rotate it again.
    if apply_orientation_detection and route.detect_orientation:
        rotation = detect_orientation(processed)

        if route.rotate and rotation:
            processed = rotate_image(
                processed,
                clockwise_degrees=rotation,
            )
            preprocessing_steps.append(f"rotate_{rotation}")

    if route.deskew:
        skew_angle = estimate_skew_angle(processed)

        if abs(skew_angle) >= MIN_DESKEW_ANGLE:
            processed = deskew_image(
                processed,
                angle=skew_angle,
            )
            preprocessing_steps.append(
                f"deskew_{skew_angle:.2f}"
            )

    # Duplicates preprocess_for_ocr's faint-ink/CLAHE handling rather than
    # calling it, since this path also runs deskew and orientation inline
    # above -- kept in sync with the same adaptive-floor logic so retry
    # candidates (geometry correction, orientation retries) get the same
    # faint-scan recovery as the primary OCR pass.
    if selected_profile.use_clahe:
        floor = estimate_ink_floor(processed)
        processed = suppress_faint_ink(processed, floor=floor)
        preprocessing_steps.append(f"suppress_faint_ink_{floor}")
        processed = enhance_contrast(processed)
        preprocessing_steps.append("clahe")
    else:
        processed = suppress_faint_ink(processed)
        preprocessing_steps.append("suppress_faint_ink")

    text, average_confidence, word_count = extract_ocr_data(
        image=processed,
        config=selected_tesseract_config,
    )

    return OCRResult(
        text=text,
        average_confidence=average_confidence,
        word_count=word_count,
        preprocessing_steps=preprocessing_steps,
        render_dpi=selected_render_dpi,
        profile_name=selected_profile.name,
        tesseract_config=selected_tesseract_config,
    )

def run_ocr(
    page: fitz.Page,
    route: QualityRoute,
    profile: OCRProfile | None = None,
    render_dpi: int | None = None,
    tesseract_config: str | None = None,
) -> OCRResult:
    """
    Render a PDF page, apply the selected benchmark-derived profile, and OCR.

    By default, the profile is selected from the quality route:

    - standard: 200 DPI, no CLAHE/denoise/threshold, PSM 11
    - degraded_scan: 200 DPI, CLAHE only, PSM 11
    """

    if not route.run_ocr:
        raise ValueError(
            "run_ocr() received a route for which run_ocr is False"
        )

    selected_profile = profile or select_ocr_profile(route)
    selected_render_dpi = render_dpi or selected_profile.render_dpi
    selected_tesseract_config = (
        tesseract_config or selected_profile.tesseract_config
    )

    image = render_page(
        page=page,
        dpi=selected_render_dpi,
    )

    processed_image, preprocessing_steps = preprocess_for_ocr(
        image=image,
        route=route,
        profile=selected_profile,
        apply_orientation_detection=False
    )

    text, average_confidence, word_count = extract_ocr_data(
        image=processed_image,
        config=selected_tesseract_config,
    )

    return OCRResult(
        text=text,
        average_confidence=average_confidence,
        word_count=word_count,
        preprocessing_steps=preprocessing_steps,
        render_dpi=selected_render_dpi,
        profile_name=selected_profile.name,
        tesseract_config=selected_tesseract_config,
    )
