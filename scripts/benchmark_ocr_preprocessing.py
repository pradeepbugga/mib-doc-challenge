# scripts/benchmark_ocr_preprocessing.py

from __future__ import annotations

import argparse
import csv
import itertools
import re
import time
from dataclasses import asdict, dataclass
from difflib import SequenceMatcher
from pathlib import Path

import cv2
import fitz
import numpy as np
import pytesseract
from PIL import Image
from pytesseract import Output


@dataclass(frozen=True)
class OCRConfiguration:
    dpi: int
    grayscale: bool
    clahe: bool
    denoise: bool

    sharpen_method: str
    upscale: float

    threshold_method: str
    invert: bool
    psm: int


@dataclass(frozen=True)
class OCRBenchmarkResult:
    page_number: int
    dpi: int
    grayscale: bool
    clahe: bool
    denoise: bool
    sharpen_method: str
    upscale: float

    threshold_method: str
    invert: bool
    psm: int

    elapsed_seconds: float
    average_confidence: float
    word_count: int

    expected_phrase_score: float
    matched_phrases: str

    field_value_score: float
    exact_field_matches: int
    fuzzy_field_matches: int
    matched_field_values: str

    meaningful_token_ratio: float
    combined_score: float
    text: str


EXPECTED_PHRASES_BY_PAGE = {
    1: [
        "form i 8090",
        "case id",
        "applicant",
        "species code",
        "home world",
        "visa class",
        "sponsor id",
        "arrival date",
        "declared purpose",
    ],
    3: [
        "manual adjudicator note",
        "adjudicator note",
        "denied",
        "illegible biometrics",
        "sponsor mismatch",
    ],
}


EXPECTED_FIELDS_BY_PAGE = {
    1: {
        "case_id": "MIB-000003",
        "applicant_name": "Solix Qorquell",
        "species_code": "LUNA_SECURID",
        "home_world": "Wolf-1061c",
        "visa_class": "XW-1",
        "sponsor_id": "SPN-6799",
        "arrival_date": "2026-03-15",
        "declared_purpose": "xenobotany",
    },
    3: {
        "decision": "DENIED",
        "risk_flag_1": "illegible biometrics",
        "risk_flag_2": "sponsor mismatch",
    },
}


def extract_document_id(pdf_path: Path) -> str:
    """
    Extract the canonical MIB document ID from a filename.

    Examples
    --------
    MIB-000002.pdf     -> MIB-000002
    MIB-000002(2).pdf  -> MIB-000002
    copy_MIB-000002.pdf -> MIB-000002
    """

    match = re.search(
        r"MIB-\d{6}",
        pdf_path.stem,
        flags=re.IGNORECASE,
    )

    if not match:
        raise ValueError(
            f"Could not extract an MIB document ID from " f"filename: {pdf_path.name}"
        )

    return match.group(0).upper()


EXPECTED_BY_DOCUMENT: dict[str, dict[int, dict[str, object]]] = {
    "MIB-000002": {
        5: {
            "phrases": [
                "sponsor attestation letter",
                "sponsor id",
                "applicant",
                "purpose",
                "visa class",
            ],
            "fields": {
                "sponsor_id": "SPN-6712",
                "applicant_name": "Miraquell Ixovara",
                "declared_purpose": "cultural exchange",
                "visa_class": "DIP-1",
            },
        },
    },
    "MIB-000003": {
        1: {
            "phrases": [
                "form i 8090",
                "case id",
                "applicant",
                "species code",
                "home world",
                "visa class",
                "sponsor id",
                "arrival date",
                "declared purpose",
            ],
            "fields": {
                "case_id": "MIB-000003",
                "applicant_name": "Solix Qorquell",
                "species_code": "LUNA_SECURID",
                "home_world": "Wolf-1061c",
                "visa_class": "XW-1",
                "sponsor_id": "SPN-6799",
                "arrival_date": "2026-03-15",
                "declared_purpose": "xenobotany",
            },
        },
        3: {
            "phrases": [
                "manual adjudicator note",
                "adjudicator note",
                "denied",
                "illegible biometrics",
                "sponsor mismatch",
            ],
            "fields": {
                "decision": "DENIED",
                "risk_flag_1": "illegible biometrics",
                "risk_flag_2": "sponsor mismatch",
            },
        },
    },
    "MIB-000005": {
        6: {
            "phrases": [
                "planetary registry extract",
                "applicant",
                "home world",
                "species code",
                "arrival date",
                "registry image",
            ],
            "fields": {
                "applicant_name": "Aridane Zavoss",
                "home_world": "Proxima-b",
                "species_code": "ARCTURIAN",
                "arrival_date": "2026-04-29",
            },
        },
    },
    "MIB-000008": {
        1: {
            "phrases": [
                "form i 8090",
                "extraterrestrial work authorization intake",
                "case id",
                "applicant",
                "species code",
                "home world",
                "visa class",
                "sponsor id",
                "arrival date",
                "declared purpose",
            ],
            "fields": {
                "case_id": "MIB-000008",
                "applicant_name": "Qorvoss Qormora",
                "species_code": "JOVIAN_GASFORM",
                "home_world": "Gliese-581g",
                "visa_class": "XW-2",
                "sponsor_id": "SPN-2313",
                "arrival_date": "2026-02-21",
                "declared_purpose": "field repair",
            },
        },
        2: {
            "phrases": [
                "sponsor attestation letter",
                "sponsor id",
                "applicant",
                "purpose",
                "visa class",
            ],
            "fields": {
                "sponsor_id": "SPN-2313",
                "applicant_name": "Qorvoss Qormora",
                "declared_purpose": "field repair",
                "visa_class": "XW-2",
            },
        },
        3: {
            "phrases": [
                "reason",
                "fee status unknown",
            ],
            "fields": {
                "reason": "Fee status unknown",
            },
        },
    },
}


def normalize_text(text: str) -> str:
    """
    Normalize prose for phrase and fuzzy matching.
    """

    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return " ".join(text.split())


def normalize_compact(text: str) -> str:
    """
    Normalize structured values while ignoring punctuation.

    Examples
    --------
    SPN-6799     -> spn6799
    2026-03-15   -> 20260315
    LUNA_SECURID -> lunasecurid
    """

    return re.sub(r"[^a-z0-9]", "", text.lower())


def render_page(
    page: fitz.Page,
    dpi: int,
) -> np.ndarray:
    """
    Render a PDF page as a BGR OpenCV image.
    """

    if dpi <= 0:
        raise ValueError("dpi must be greater than zero")

    zoom = dpi / 72.0

    pixmap = page.get_pixmap(
        matrix=fitz.Matrix(zoom, zoom),
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

    return cv2.cvtColor(
        image,
        cv2.COLOR_RGB2BGR,
    )


def to_grayscale(image: np.ndarray) -> np.ndarray:
    """
    Convert an image to grayscale when needed.
    """

    if image.ndim == 2:
        return image

    return cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY,
    )
def upscale_image(
    image: np.ndarray,
    scale: float,
) -> np.ndarray:

    if scale == 1.0:
        return image

    return cv2.resize(
        image,
        None,
        fx=scale,
        fy=scale,
        interpolation=cv2.INTER_CUBIC,
    )
def apply_unsharp_mask(
    image: np.ndarray,
    sigma: float = 1.0,
    amount: float = 1.5,
) -> np.ndarray:
    gray = to_grayscale(image)

    blurred = cv2.GaussianBlur(
        gray,
        (0, 0),
        sigmaX=sigma,
    )

    return cv2.addWeighted(
        gray,
        1.0 + amount,
        blurred,
        -amount,
        0,
    )

def apply_laplacian_sharpening(
    image: np.ndarray,
    strength: float = 0.5,
) -> np.ndarray:
    gray = to_grayscale(image)

    laplacian = cv2.Laplacian(
        gray,
        cv2.CV_32F,
        ksize=3,
    )

    sharpened = gray.astype(np.float32) - strength * laplacian

    return np.clip(
        sharpened,
        0,
        255,
    ).astype(np.uint8)

def apply_clahe(image: np.ndarray) -> np.ndarray:
    """
    Apply local contrast enhancement.
    """

    gray = to_grayscale(image)

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8),
    )

    return clahe.apply(gray)


def apply_denoising(image: np.ndarray) -> np.ndarray:
    """
    Apply non-local means denoising.
    """

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
    """
    Apply the selected thresholding operation.
    """

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


def preprocess_image(
    image: np.ndarray,
    config: OCRConfiguration,
) -> np.ndarray:
    """
    Apply one preprocessing configuration.
    """

    processed = image.copy()

    if config.grayscale:
        processed = to_grayscale(processed)

    if config.clahe:
        processed = apply_clahe(processed)

    if config.denoise:
        processed = apply_denoising(processed)

    processed = upscale_image(
        image=processed,
        scale=config.upscale,
    )
    if config.sharpen_method == "unsharp":
        processed = apply_unsharp_mask(processed)

    if config.sharpen_method == "laplacian":
        processed = apply_laplacian_sharpening(processed)


    processed = apply_threshold(
        image=processed,
        threshold_method=config.threshold_method,
    )

    if config.invert:
        processed = cv2.bitwise_not(processed)

    return processed


def run_tesseract(
    image: np.ndarray,
    psm: int,
) -> tuple[str, float, int]:
    """
    Run Tesseract and return text, mean confidence, and word count.
    """

    if image.ndim == 3:
        image = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB,
        )

    pil_image = Image.fromarray(image)

    data = pytesseract.image_to_data(
        pil_image,
        config=f"--oem 3 --psm {psm}",
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

    extracted_text = " ".join(words).strip()

    average_confidence = sum(confidences) / len(confidences) if confidences else 0.0

    return (
        extracted_text,
        average_confidence,
        len(words),
    )


def phrase_match_score(
    text: str,
    expected_phrases: list[str],
) -> tuple[float, list[str]]:
    """
    Calculate exact normalized phrase recovery.
    """

    if not expected_phrases:
        return 0.0, []

    normalized_text = normalize_text(text)

    matched_phrases = [
        phrase
        for phrase in expected_phrases
        if normalize_text(phrase) in normalized_text
    ]

    score = len(matched_phrases) / len(expected_phrases)

    return score, matched_phrases


def meaningful_token_ratio(text: str) -> float:
    """
    Estimate how much OCR output consists of plausible words.
    """

    tokens = text.split()

    if not tokens:
        return 0.0

    meaningful_count = sum(
        bool(re.fullmatch(r"[A-Za-z]{3,}", token)) for token in tokens
    )

    return meaningful_count / len(tokens)


def best_fuzzy_window_score(
    expected_value: str,
    extracted_text: str,
) -> tuple[float, str]:
    """
    Find the OCR token window most similar to an expected value.

    Returns
    -------
    tuple[float, str]
        Best similarity score and corresponding OCR candidate.
    """

    expected = normalize_text(expected_value)
    extracted = normalize_text(extracted_text)

    expected_tokens = expected.split()
    extracted_tokens = extracted.split()

    if not expected_tokens or not extracted_tokens:
        return 0.0, ""

    expected_token_count = len(expected_tokens)

    minimum_window_size = max(
        1,
        expected_token_count - 1,
    )
    maximum_window_size = expected_token_count + 2

    best_score = 0.0
    best_candidate = ""

    for window_size in range(
        minimum_window_size,
        maximum_window_size + 1,
    ):
        if window_size > len(extracted_tokens):
            continue

        for start_index in range(len(extracted_tokens) - window_size + 1):
            candidate = " ".join(
                extracted_tokens[start_index : start_index + window_size]
            )

            similarity = SequenceMatcher(
                None,
                expected,
                candidate,
            ).ratio()

            if similarity > best_score:
                best_score = similarity
                best_candidate = candidate

    return best_score, best_candidate


def score_field_values(
    text: str,
    expected_fields: dict[str, str],
    fuzzy_threshold: float = 0.72,
) -> tuple[float, int, int, list[str]]:
    """
    Score recovery of required field values.

    Exact matches receive 1.0 credit. Fuzzy matches receive their
    similarity score when they exceed the supplied threshold.
    """

    if not expected_fields:
        return 0.0, 0, 0, []

    normalized_text = normalize_text(text)
    compact_text = normalize_compact(text)

    total_score = 0.0
    exact_matches = 0
    fuzzy_matches = 0
    match_details: list[str] = []

    for field_name, expected_value in expected_fields.items():
        normalized_expected = normalize_text(expected_value)
        compact_expected = normalize_compact(expected_value)

        exact_match = (
            normalized_expected in normalized_text or compact_expected in compact_text
        )

        if exact_match:
            total_score += 1.0
            exact_matches += 1

            match_details.append(f"{field_name}=exact:{expected_value}")
            continue

        fuzzy_score, candidate = best_fuzzy_window_score(
            expected_value=expected_value,
            extracted_text=text,
        )

        if fuzzy_score >= fuzzy_threshold:
            total_score += fuzzy_score
            fuzzy_matches += 1

            match_details.append(
                f"{field_name}=fuzzy:{fuzzy_score:.3f}:" f"{candidate}"
            )
        else:
            match_details.append(f"{field_name}=miss:{fuzzy_score:.3f}:" f"{candidate}")

    field_value_score = total_score / len(expected_fields)

    return (
        field_value_score,
        exact_matches,
        fuzzy_matches,
        match_details,
    )


def calculate_combined_score(
    phrase_score: float,
    field_value_score: float,
    confidence: float,
    meaningful_ratio: float,
    elapsed_seconds: float,
) -> float:
    """
    Rank configurations primarily by recovery of required field values.

    Runtime receives a small penalty because the final scorer allows
    approximately six seconds per PDF on average.
    """

    normalized_confidence = min(
        max(confidence / 100.0, 0.0),
        1.0,
    )

    runtime_penalty = min(
        elapsed_seconds / 3.0,
        1.0,
    )

    return (
        0.55 * field_value_score
        + 0.20 * phrase_score
        + 0.15 * meaningful_ratio
        + 0.10 * normalized_confidence
        - 0.05 * runtime_penalty
    )


def build_grid() -> list[OCRConfiguration]:
    """
    Build the OCR preprocessing parameter grid.
    """

    parameter_grid = {
        "dpi": [200, 300, 400],
        "grayscale": [True],
        "clahe": [False, True],
        "denoise": [False],
        "upscale": [1.0, 1.5, 2.0],
        "sharpen_method": ["none", "unsharp", "laplacian"],
        "threshold_method": [
            "none"
        ],
        "invert": [False],
        "psm": [3, 6, 11, 12],
    }

    keys = list(parameter_grid)

    configurations: list[OCRConfiguration] = []

    for values in itertools.product(*(parameter_grid[key] for key in keys)):
        parameters = dict(
            zip(
                keys,
                values,
                strict=True,
            )
        )

        configurations.append(OCRConfiguration(**parameters))

    return configurations


def benchmark_page(
    page: fitz.Page,
    expected_phrases: list[str],
    expected_fields: dict[str, str],
    configurations: list[OCRConfiguration],
) -> list[OCRBenchmarkResult]:
    """
    Benchmark all OCR configurations for one PDF page.
    """

    rendered_images: dict[int, np.ndarray] = {}
    results: list[OCRBenchmarkResult] = []

    for index, config in enumerate(
        configurations,
        start=1,
    ):
        if config.dpi not in rendered_images:
            rendered_images[config.dpi] = render_page(
                page=page,
                dpi=config.dpi,
            )

        original_image = rendered_images[config.dpi]

        start_time = time.perf_counter()

        processed_image = preprocess_image(
            image=original_image,
            config=config,
        )

        text, confidence, word_count = run_tesseract(
            image=processed_image,
            psm=config.psm,
        )

        elapsed_seconds = time.perf_counter() - start_time

        phrase_score, matched_phrases = phrase_match_score(
            text=text,
            expected_phrases=expected_phrases,
        )

        (
            field_value_score,
            exact_matches,
            fuzzy_matches,
            field_match_details,
        ) = score_field_values(
            text=text,
            expected_fields=expected_fields,
        )

        token_ratio = meaningful_token_ratio(text)

        combined_score = calculate_combined_score(
            phrase_score=phrase_score,
            field_value_score=field_value_score,
            confidence=confidence,
            meaningful_ratio=token_ratio,
            elapsed_seconds=elapsed_seconds,
        )

        results.append(
            OCRBenchmarkResult(
                page_number=page.number + 1,
                dpi=config.dpi,
                grayscale=config.grayscale,
                clahe=config.clahe,
                denoise=config.denoise,
                sharpen_method=config.sharpen_method,
                upscale=config.upscale, 
                threshold_method=config.threshold_method,
                invert=config.invert,
                psm=config.psm,
                elapsed_seconds=round(
                    elapsed_seconds,
                    4,
                ),
                average_confidence=round(
                    confidence,
                    3,
                ),
                word_count=word_count,
                expected_phrase_score=round(
                    phrase_score,
                    4,
                ),
                matched_phrases="|".join(matched_phrases),
                field_value_score=round(
                    field_value_score,
                    4,
                ),
                exact_field_matches=exact_matches,
                fuzzy_field_matches=fuzzy_matches,
                matched_field_values="|".join(field_match_details),
                meaningful_token_ratio=round(
                    token_ratio,
                    4,
                ),
                combined_score=round(
                    combined_score,
                    4,
                ),
                text=text,
            )
        )

        print(
            f"\rTested {index}/{len(configurations)} configurations",
            end="",
            flush=True,
        )

    print()

    return sorted(
        results,
        key=lambda result: (
            result.combined_score,
            result.field_value_score,
            result.exact_field_matches,
            -result.elapsed_seconds,
        ),
        reverse=True,
    )


def save_results(
    results: list[OCRBenchmarkResult],
    output_path: Path,
) -> None:
    """
    Save ranked benchmark results as CSV.
    """

    if not results:
        raise ValueError("No OCR benchmark results were produced")

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


def print_top_results(
    results: list[OCRBenchmarkResult],
    limit: int = 10,
) -> None:
    """
    Print the highest-ranked OCR configurations.
    """

    print()

    for rank, result in enumerate(
        results[:limit],
        start=1,
    ):
        print(
            f"{rank}. "
            f"combined={result.combined_score:.4f} "
            f"values={result.field_value_score:.4f} "
            f"exact={result.exact_field_matches} "
            f"fuzzy={result.fuzzy_field_matches} "
            f"phrases={result.expected_phrase_score:.4f} "
            f"confidence={result.average_confidence:.1f} "
            f"dpi={result.dpi} "
            f"clahe={result.clahe} "
            f"denoise={result.denoise} "
            f"threshold={result.threshold_method} "
            f"psm={result.psm} "
            f"time={result.elapsed_seconds:.3f}s"
        )

        print(f"   Fields: {result.matched_field_values}")

        print(f"   Phrases: {result.matched_phrases or 'none'}")

        print(f"   Text: {result.text[:300]}")

        print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Grid-search OCR preprocessing and Tesseract settings "
            "for a selected PDF page."
        )
    )

    parser.add_argument(
        "pdf_path",
        type=Path,
        help="Path to the source PDF.",
    )

    parser.add_argument(
        "--page",
        type=int,
        required=True,
        help="One-indexed page number.",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path("ocr_grid_results.csv"),
        help="Path for the ranked CSV output.",
    )

    parser.add_argument(
        "--top",
        type=int,
        default=10,
        help="Number of top results to print.",
    )

    args = parser.parse_args()

    if not args.pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {args.pdf_path}")

    document_id = extract_document_id(args.pdf_path)

    document_ground_truth = EXPECTED_BY_DOCUMENT.get(
        document_id,
        {},
    )

    page_ground_truth = document_ground_truth.get(
        args.page,
        {},
    )

    expected_phrases = list(page_ground_truth.get("phrases", []))

    expected_fields = dict(page_ground_truth.get("fields", {}))

    if not expected_phrases and not expected_fields:
        raise ValueError(
            f"No expected phrases or fields configured for "
            f"{document_id}, page {args.page}"
        )

    configurations = build_grid()

    print(f"Testing {len(configurations)} configurations " f"on page {args.page}...")

    with fitz.open(args.pdf_path) as document:
        if args.page < 1 or args.page > len(document):
            raise ValueError(
                f"Page {args.page} is outside the PDF range " f"1-{len(document)}"
            )

        page = document[args.page - 1]

        results = benchmark_page(
            page=page,
            expected_phrases=expected_phrases,
            expected_fields=expected_fields,
            configurations=configurations,
        )

    save_results(
        results=results,
        output_path=args.output,
    )

    print_top_results(
        results=results,
        limit=args.top,
    )

    print(f"Full ranked results saved to: {args.output}")


if __name__ == "__main__":
    main()
