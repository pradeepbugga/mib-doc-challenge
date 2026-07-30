from __future__ import annotations

import fitz  # PyMuPDF
import argparse
import json
from pathlib import Path
from typing import Any

# Update these imports to match your project.
from core.extraction.extractor import extract_fields
from core.pipeline.page_pipeline import process_page


def serialize_classification(classification: Any) -> dict:
    """
    Convert a classification result into JSON-safe data.

    Supports dataclasses or simple objects with attributes.
    """
    if classification is None:
        return {
            "document_type": "unknown",
            "score": 0,
            "confidence": 0.0,
            "matched_cues": [],
        }

    return {
        "document_type": getattr(
            classification,
            "document_type",
            "unknown",
        ),
        "score": getattr(classification, "score", 0),
        "confidence": getattr(classification, "confidence", 0.0),
        "matched_cues": list(
            getattr(classification, "matched_cues", ())
        ),
    }


def test_extraction(pdf_path: Path) -> list[dict]:
    """
    Process every page in a PDF and run field extraction.

    Parameters
    ----------
    pdf_path
        Path to the PDF packet.

    Returns
    -------
    list[dict]
        Page-level processing and extraction results.
    """
    results: list[dict] = []

    with fitz.open(pdf_path) as doc:
        for page in doc:
            page_result = process_page(
                doc=doc,
                page=page,
            )

            classification = page_result["classification"]
            document_type = classification.document_type

            # Use the final text selected by process_page:
            # either native PDF text or OCR text.
            page_text = page_result["text"]

            extraction = extract_fields(
                document_type=document_type,
                text=page_text,
            )

            results.append(
                {
                    "page_number": page.number + 1,
                    "quality": page_result[
                        "assessment"
                    ].quality_class,
                    "text_source": page_result["text_source"],
                    "ocr_strategy": page_result[
                        "route"
                    ].ocr_strategy.value,
                    "classification": {
                        "document_type": classification.document_type,
                        "score": classification.score,
                        "confidence": classification.confidence,
                        "matched_cues": list(
                            classification.matched_cues
                        ),
                    },
                    "extraction": extraction,
                    "page_text": page_text,
                }
            )

    return results

def print_results(results: list[dict]) -> None:
    """Print page-level extraction results."""
    for result in results:
        classification = result["classification"]
        fields = result["extraction"].get("fields", {})

        print("=" * 80)
        print(f"Page {result['page_number']}:")
        print(f"  Quality: {result['quality']}")
        print(f"  Text source: {result['text_source']}")
        print(f"  OCR strategy: {result['ocr_strategy']}")
        print(
            "  Classification: "
            f"{classification['document_type']} "
            f"(confidence={classification['confidence']:.2f})"
        )

        print("  Extracted fields:")

        if not fields:
            print("    None")
        else:
            for field_name, value in fields.items():
                print(f"    {field_name}: {value!r}")

        print("  Page text:")
        print(f"    {result['page_text'][:500]}")


def save_results(
    results: list[dict],
    output_path: Path,
) -> None:
    """Save results to JSON."""
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            results,
            file,
            indent=2,
            ensure_ascii=False,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run page processing and field extraction "
            "on an MIB packet."
        )
    )

    parser.add_argument(
        "pdf_path",
        type=Path,
        help="Path to the PDF packet.",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path for output JSON.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.pdf_path.is_file():
        raise FileNotFoundError(
            f"PDF not found: {args.pdf_path}"
        )

    results = test_extraction(args.pdf_path)

    print_results(results)

    if args.output is not None:
        save_results(
            results=results,
            output_path=args.output,
        )
        print(f"\nSaved results to {args.output}")


if __name__ == "__main__":
    main()