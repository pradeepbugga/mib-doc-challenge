from __future__ import annotations

import argparse
import csv
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping

import cv2
import fitz
import numpy as np
import pytesseract
from PIL import Image
from pytesseract import Output


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class NormalizedRegion:
    """
    Region coordinates expressed as fractions of page width and height.

    Values must be between 0.0 and 1.0.

    Example
    -------
    x1=0.50 means the crop begins halfway across the page.
    """

    x1: float
    y1: float
    x2: float
    y2: float

    def validate(self) -> None:
        values = (self.x1, self.y1, self.x2, self.y2)

        if not all(0.0 <= value <= 1.0 for value in values):
            raise ValueError(f"Normalized coordinates must be between 0 and 1: {self}")

        if self.x2 <= self.x1:
            raise ValueError(f"x2 must be greater than x1: {self}")

        if self.y2 <= self.y1:
            raise ValueError(f"y2 must be greater than y1: {self}")


@dataclass(frozen=True)
class RegionOCRConfig:
    region: NormalizedRegion
    psm: int
    grayscale: bool = True
    clahe: bool = False
    denoise: bool = False
    threshold_method: str = "none"
    invert: bool = False
    padding_pixels: int = 10
    whitelist: str | None = None


@dataclass(frozen=True)
class RegionOCRResult:
    document_type: str
    field_name: str
    page_number: int

    text: str
    normalized_text: str

    average_confidence: float
    word_count: int

    psm: int
    threshold_method: str
    clahe: bool
    denoise: bool
    invert: bool

    x1: int
    y1: int
    x2: int
    y2: int

    crop_path: str


# ---------------------------------------------------------------------------
# Template configuration
# ---------------------------------------------------------------------------
#
# IMPORTANT:
# These coordinates are starting placeholders.
#
# Use --preview to draw the regions onto the page, inspect the output image,
# and adjust the coordinates until each crop isolates the intended value.
#
# Coordinates are normalized:
#
#   0.0 -------------------------- 1.0
#    |
#    |
#    |
#    |
#   1.0
#
# Region values should generally include the VALUE but exclude the label when
# possible.
# ---------------------------------------------------------------------------


REGION_TEMPLATES: dict[
    str,
    dict[str, RegionOCRConfig],
] = {
    "intake_form": {
        "case_id": RegionOCRConfig(
            region=NormalizedRegion(
                x1=0.60,
                y1=0.06,
                x2=0.94,
                y2=0.13,
            ),
            psm=7,
            clahe=True,
            whitelist=(
                "ABCDEFGHIJKLMNOPQRSTUVWXYZ" "abcdefghijklmnopqrstuvwxyz" "0123456789-"
            ),
        ),
        "applicant_name": RegionOCRConfig(
            region=NormalizedRegion(
                x1=0.42,
                y1=0.18,
                x2=0.91,
                y2=0.25,
            ),
            psm=7,
            clahe=True,
        ),
        "species_code": RegionOCRConfig(
            region=NormalizedRegion(
                x1=0.42,
                y1=0.27,
                x2=0.91,
                y2=0.34,
            ),
            psm=7,
            clahe=True,
            whitelist=(
                "ABCDEFGHIJKLMNOPQRSTUVWXYZ" "abcdefghijklmnopqrstuvwxyz" "0123456789_-"
            ),
        ),
        "home_world": RegionOCRConfig(
            region=NormalizedRegion(
                x1=0.42,
                y1=0.36,
                x2=0.91,
                y2=0.43,
            ),
            psm=7,
            clahe=True,
            whitelist=(
                "ABCDEFGHIJKLMNOPQRSTUVWXYZ" "abcdefghijklmnopqrstuvwxyz" "0123456789-"
            ),
        ),
        "visa_class": RegionOCRConfig(
            region=NormalizedRegion(
                x1=0.42,
                y1=0.45,
                x2=0.70,
                y2=0.52,
            ),
            psm=7,
            clahe=True,
            whitelist=(
                "ABCDEFGHIJKLMNOPQRSTUVWXYZ" "abcdefghijklmnopqrstuvwxyz" "0123456789-"
            ),
        ),
        "sponsor_id": RegionOCRConfig(
            region=NormalizedRegion(
                x1=0.42,
                y1=0.54,
                x2=0.80,
                y2=0.61,
            ),
            psm=7,
            clahe=True,
            whitelist=(
                "ABCDEFGHIJKLMNOPQRSTUVWXYZ" "abcdefghijklmnopqrstuvwxyz" "0123456789-"
            ),
        ),
        "arrival_date": RegionOCRConfig(
            region=NormalizedRegion(
                x1=0.42,
                y1=0.63,
                x2=0.80,
                y2=0.70,
            ),
            psm=7,
            clahe=True,
            whitelist="0123456789-/.",
        ),
        "declared_purpose": RegionOCRConfig(
            region=NormalizedRegion(
                x1=0.42,
                y1=0.72,
                x2=0.91,
                y2=0.80,
            ),
            psm=7,
            clahe=True,
        ),
    },
    "fee_receipt": {
        "case_id": RegionOCRConfig(
            region=NormalizedRegion(
                x1=0.48,
                y1=0.20,
                x2=0.90,
                y2=0.30,
            ),
            psm=7,
            clahe=True,
            whitelist=(
                "ABCDEFGHIJKLMNOPQRSTUVWXYZ" "abcdefghijklmnopqrstuvwxyz" "0123456789-"
            ),
        ),
        "fee_status": RegionOCRConfig(
            region=NormalizedRegion(
                x1=0.48,
                y1=0.30,
                x2=0.90,
                y2=0.41,
            ),
            psm=7,
            clahe=True,
        ),
    },
    "adjudicator_note": {
        "decision": RegionOCRConfig(
            region=NormalizedRegion(
                x1=0.58,
                y1=0.14,
                x2=0.93,
                y2=0.30,
            ),
            psm=11,
            clahe=False,
            threshold_method="none",
            whitelist=("ABCDEFGHIJKLMNOPQRSTUVWXYZ" "abcdefghijklmnopqrstuvwxyz"),
        ),
        "reason": RegionOCRConfig(
            region=NormalizedRegion(
                x1=0.12,
                y1=0.27,
                x2=0.92,
                y2=0.58,
            ),
            psm=6,
            clahe=False,
            threshold_method="none",
        ),
    },
    "biometric_slip": {
        "biometric_status": RegionOCRConfig(
            region=NormalizedRegion(
                x1=0.20,
                y1=0.20,
                x2=0.88,
                y2=0.45,
            ),
            psm=11,
            clahe=True,
        ),
    },
}


# ---------------------------------------------------------------------------
# Rendering and cropping
# ---------------------------------------------------------------------------


def render_page(
    page: fitz.Page,
    dpi: int,
) -> np.ndarray:
    """
    Render one PDF page as an OpenCV BGR image.
    """

    if dpi <= 0:
        raise ValueError("dpi must be greater than zero")

    zoom = dpi / 72.0

    pixmap = page.get_pixmap(
        matrix=fitz.Matrix(zoom, zoom),
        colorspace=fitz.csRGB,
        alpha=False,
    )

    image = np.frombuffer(
        pixmap.samples,
        dtype=np.uint8,
    ).reshape(
        pixmap.height,
        pixmap.width,
        pixmap.n,
    )

    return cv2.cvtColor(
        image,
        cv2.COLOR_RGB2BGR,
    )


def normalized_to_pixels(
    region: NormalizedRegion,
    image_width: int,
    image_height: int,
    padding_pixels: int = 0,
) -> tuple[int, int, int, int]:
    """
    Convert normalized region coordinates into pixel coordinates.
    """

    region.validate()

    x1 = int(region.x1 * image_width)
    y1 = int(region.y1 * image_height)
    x2 = int(region.x2 * image_width)
    y2 = int(region.y2 * image_height)

    x1 = max(0, x1 - padding_pixels)
    y1 = max(0, y1 - padding_pixels)
    x2 = min(image_width, x2 + padding_pixels)
    y2 = min(image_height, y2 + padding_pixels)

    return x1, y1, x2, y2


def crop_region(
    image: np.ndarray,
    config: RegionOCRConfig,
) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    """
    Crop a configured region from a rendered page.
    """

    image_height, image_width = image.shape[:2]

    coordinates = normalized_to_pixels(
        region=config.region,
        image_width=image_width,
        image_height=image_height,
        padding_pixels=config.padding_pixels,
    )

    x1, y1, x2, y2 = coordinates

    crop = image[y1:y2, x1:x2].copy()

    if crop.size == 0:
        raise ValueError(f"Empty crop generated for region {config.region}")

    return crop, coordinates


# ---------------------------------------------------------------------------
# Image preprocessing
# ---------------------------------------------------------------------------


def to_grayscale(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image

    return cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY,
    )


def apply_clahe(image: np.ndarray) -> np.ndarray:
    gray = to_grayscale(image)

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8),
    )

    return clahe.apply(gray)


def apply_denoising(image: np.ndarray) -> np.ndarray:
    gray = to_grayscale(image)

    return cv2.fastNlMeansDenoising(
        gray,
        None,
        h=10,
        templateWindowSize=7,
        searchWindowSize=21,
    )


def apply_threshold(
    image: np.ndarray,
    threshold_method: str,
) -> np.ndarray:
    gray = to_grayscale(image)

    if threshold_method == "none":
        return gray

    if threshold_method == "otsu":
        return cv2.threshold(
            gray,
            0,
            255,
            cv2.THRESH_BINARY | cv2.THRESH_OTSU,
        )[1]

    if threshold_method == "adaptive_gaussian":
        return cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            31,
            15,
        )

    if threshold_method == "adaptive_mean":
        return cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_MEAN_C,
            cv2.THRESH_BINARY,
            31,
            15,
        )

    raise ValueError(f"Unsupported threshold method: {threshold_method}")


def preprocess_crop(
    crop: np.ndarray,
    config: RegionOCRConfig,
) -> np.ndarray:
    """
    Apply field-specific preprocessing.
    """

    processed = crop.copy()

    if config.grayscale:
        processed = to_grayscale(processed)

    if config.clahe:
        processed = apply_clahe(processed)

    if config.denoise:
        processed = apply_denoising(processed)

    processed = apply_threshold(
        image=processed,
        threshold_method=config.threshold_method,
    )

    if config.invert:
        processed = cv2.bitwise_not(processed)

    return processed


# ---------------------------------------------------------------------------
# OCR and normalization
# ---------------------------------------------------------------------------


def build_tesseract_config(
    config: RegionOCRConfig,
) -> str:
    parts = [
        "--oem 3",
        f"--psm {config.psm}",
    ]

    if config.whitelist:
        parts.append(f"-c tessedit_char_whitelist={config.whitelist}")

    return " ".join(parts)


def run_tesseract(
    image: np.ndarray,
    config: RegionOCRConfig,
) -> tuple[str, float, int]:
    """
    OCR one preprocessed crop.
    """

    if image.ndim == 3:
        image = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB,
        )

    pil_image = Image.fromarray(image)

    data = pytesseract.image_to_data(
        pil_image,
        config=build_tesseract_config(config),
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

        if not text:
            continue

        try:
            confidence_value = float(confidence)
        except (TypeError, ValueError):
            confidence_value = -1.0

        words.append(text)

        if confidence_value >= 0:
            confidences.append(confidence_value)

    extracted_text = " ".join(words).strip()

    average_confidence = sum(confidences) / len(confidences) if confidences else 0.0

    return (
        extracted_text,
        average_confidence,
        len(words),
    )


def normalize_generic(text: str) -> str:
    return " ".join(text.strip().split())


def normalize_identifier(text: str) -> str:
    """
    Normalize IDs while preserving hyphens and underscores.
    """

    text = text.strip()
    text = re.sub(r"\s*-\s*", "-", text)
    text = re.sub(r"\s*_\s*", "_", text)
    text = re.sub(r"\s+", "", text)

    return text.upper()


def normalize_date(text: str) -> str:
    """
    Normalize common OCR date formats toward YYYY-MM-DD.
    """

    text = text.strip()

    match = re.search(
        r"\b(20\d{2})[\s./-]+(\d{1,2})[\s./-]+(\d{1,2})\b",
        text,
    )

    if not match:
        compact_match = re.search(
            r"\b(20\d{2})(\d{2})(\d{2})\b",
            re.sub(r"\D", "", text),
        )

        if not compact_match:
            return normalize_generic(text)

        year, month, day = compact_match.groups()

    else:
        year, month, day = match.groups()

    return f"{year}-{int(month):02d}-{int(day):02d}"


def normalize_decision(text: str) -> str:
    normalized = re.sub(
        r"[^a-z]",
        "",
        text.lower(),
    )

    if "denied" in normalized:
        return "DENIED"

    if "approved" in normalized:
        return "APPROVED"

    if "pending" in normalized:
        return "PENDING"

    return normalize_generic(text).upper()


def normalize_fee_status(text: str) -> str:
    normalized = re.sub(
        r"[^a-z]",
        "",
        text.lower(),
    )

    if "unpaid" in normalized:
        return "unpaid"

    if "paid" in normalized:
        return "paid"

    if "waived" in normalized:
        return "waived"

    return normalize_generic(text).lower()


def normalize_field_value(
    field_name: str,
    text: str,
) -> str:
    """
    Apply field-specific normalization after OCR.
    """

    identifier_fields = {
        "case_id",
        "species_code",
        "home_world",
        "visa_class",
        "sponsor_id",
    }

    if field_name in identifier_fields:
        return normalize_identifier(text)

    if field_name == "arrival_date":
        return normalize_date(text)

    if field_name == "decision":
        return normalize_decision(text)

    if field_name == "fee_status":
        return normalize_fee_status(text)

    return normalize_generic(text)


# ---------------------------------------------------------------------------
# Region extraction
# ---------------------------------------------------------------------------


def extract_regions(
    image: np.ndarray,
    document_type: str,
    page_number: int,
    crop_output_dir: Path,
    fields: set[str] | None = None,
) -> list[RegionOCRResult]:
    """
    Extract all configured regions for one document type.
    """

    template = REGION_TEMPLATES.get(document_type)

    if template is None:
        raise ValueError(
            f"No region template configured for document type " f"'{document_type}'"
        )

    crop_output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    results: list[RegionOCRResult] = []

    for field_name, config in template.items():
        if fields is not None and field_name not in fields:
            continue

        crop, coordinates = crop_region(
            image=image,
            config=config,
        )

        processed_crop = preprocess_crop(
            crop=crop,
            config=config,
        )

        text, average_confidence, word_count = run_tesseract(
            image=processed_crop,
            config=config,
        )

        normalized_text = normalize_field_value(
            field_name=field_name,
            text=text,
        )

        crop_path = (
            crop_output_dir / f"page_{page_number}_{document_type}_{field_name}.png"
        )

        cv2.imwrite(
            str(crop_path),
            processed_crop,
        )

        x1, y1, x2, y2 = coordinates

        results.append(
            RegionOCRResult(
                document_type=document_type,
                field_name=field_name,
                page_number=page_number,
                text=text,
                normalized_text=normalized_text,
                average_confidence=round(
                    average_confidence,
                    3,
                ),
                word_count=word_count,
                psm=config.psm,
                threshold_method=config.threshold_method,
                clahe=config.clahe,
                denoise=config.denoise,
                invert=config.invert,
                x1=x1,
                y1=y1,
                x2=x2,
                y2=y2,
                crop_path=str(crop_path),
            )
        )

    return results


# ---------------------------------------------------------------------------
# Region preview
# ---------------------------------------------------------------------------


def create_region_preview(
    image: np.ndarray,
    document_type: str,
    output_path: Path,
) -> None:
    """
    Draw all configured crop regions onto the rendered page.
    """

    template = REGION_TEMPLATES.get(document_type)

    if template is None:
        raise ValueError(f"No region template configured for '{document_type}'")

    preview = image.copy()

    image_height, image_width = preview.shape[:2]

    for field_name, config in template.items():
        x1, y1, x2, y2 = normalized_to_pixels(
            region=config.region,
            image_width=image_width,
            image_height=image_height,
            padding_pixels=0,
        )

        cv2.rectangle(
            preview,
            (x1, y1),
            (x2, y2),
            (0, 0, 255),
            3,
        )

        cv2.putText(
            preview,
            field_name,
            (x1, max(25, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    cv2.imwrite(
        str(output_path),
        preview,
    )


# ---------------------------------------------------------------------------
# Saving output
# ---------------------------------------------------------------------------


def save_results_csv(
    results: list[RegionOCRResult],
    output_path: Path,
) -> None:
    if not results:
        raise ValueError("No region OCR results were produced")

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(asdict(results[0]).keys()),
        )

        writer.writeheader()

        for result in results:
            writer.writerow(asdict(result))


def save_results_json(
    results: list[RegionOCRResult],
    output_path: Path,
) -> None:
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    payload = {
        result.field_name: {
            "value": result.normalized_text,
            "raw_text": result.text,
            "confidence": result.average_confidence,
            "word_count": result.word_count,
            "crop_path": result.crop_path,
        }
        for result in results
    }

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            payload,
            file,
            indent=2,
        )


def print_results(
    results: list[RegionOCRResult],
) -> None:
    print()

    for result in results:
        print(f"{result.field_name}:")
        print(f"  Raw: {result.text!r}")
        print(f"  Normalized: {result.normalized_text!r}")
        print(f"  Confidence: " f"{result.average_confidence:.1f}")
        print(f"  Words: {result.word_count}")
        print(f"  Crop: {result.crop_path}")
        print()


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------


def parse_fields(
    fields_argument: str | None,
) -> set[str] | None:
    if fields_argument is None:
        return None

    fields = {field.strip() for field in fields_argument.split(",") if field.strip()}

    return fields or None


def main() -> None:
    parser = argparse.ArgumentParser(
        description=("Run template-based region OCR on one PDF page.")
    )

    parser.add_argument(
        "pdf_path",
        type=Path,
        help="Path to the PDF.",
    )

    parser.add_argument(
        "--page",
        type=int,
        required=True,
        help="One-indexed page number.",
    )

    parser.add_argument(
        "--document-type",
        required=True,
        choices=sorted(REGION_TEMPLATES),
        help="Template used to define OCR regions.",
    )

    parser.add_argument(
        "--dpi",
        type=int,
        default=200,
        help="PDF rendering DPI.",
    )

    parser.add_argument(
        "--fields",
        default=None,
        help=(
            "Optional comma-separated fields to extract. "
            "By default all configured fields are processed."
        ),
    )

    parser.add_argument(
        "--crop-dir",
        type=Path,
        default=Path("evaluation/region_ocr/crops"),
        help="Directory for saved crop images.",
    )

    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("evaluation/region_ocr/results.csv"),
    )

    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("evaluation/region_ocr/results.json"),
    )

    parser.add_argument(
        "--preview",
        type=Path,
        default=None,
        help=("Optional path for a page image showing region boxes."),
    )

    args = parser.parse_args()

    if not args.pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {args.pdf_path}")

    fields = parse_fields(args.fields)

    with fitz.open(args.pdf_path) as document:
        if args.page < 1 or args.page > len(document):
            raise ValueError(
                f"Page {args.page} is outside the range " f"1-{len(document)}"
            )

        page = document[args.page - 1]

        image = render_page(
            page=page,
            dpi=args.dpi,
        )

    if args.preview is not None:
        create_region_preview(
            image=image,
            document_type=args.document_type,
            output_path=args.preview,
        )

        print(f"Region preview saved to: {args.preview}")

    results = extract_regions(
        image=image,
        document_type=args.document_type,
        page_number=args.page,
        crop_output_dir=args.crop_dir,
        fields=fields,
    )

    save_results_csv(
        results=results,
        output_path=args.output_csv,
    )

    save_results_json(
        results=results,
        output_path=args.output_json,
    )

    print_results(results)

    print(f"CSV saved to: {args.output_csv}")
    print(f"JSON saved to: {args.output_json}")


if __name__ == "__main__":
    main()
