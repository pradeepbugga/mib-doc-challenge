from __future__ import annotations

import fitz  # PyMuPDF
import argparse
import json
from pathlib import Path
from typing import Any

# Update these imports to match your project.
from core.extraction.extractor import extract_fields
from core.pipeline.page_pipeline import process_page
from core.pipeline.case_assignment import normalize_case_id_candidate
from core.pipeline.identity_resolution import resolve_initial_identities


def extract_filename_case_id(pdf_path: Path) -> str | None:
    return normalize_case_id_candidate(pdf_path.stem)

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

        filename_case_id = extract_filename_case_id(pdf_path)

        for page in doc:
            page_result = process_page(
                doc=doc,
                page=page)

            classification = page_result["classification"]
            document_type = classification.document_type
            case_id_candidates = page_result["case_id_candidates"]

            # Use the final text selected by process_page:
            # either native PDF text or OCR text.
            page_text = page_result["text"]

            extraction = page_result["extraction"]
            

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
                    "case_id_candidates": case_id_candidates,
                    "extraction": extraction,
                    "selected_rotation": page_result["selected_rotation"],
                    "orientation_retry_attempted": page_result[
                        "orientation_retry_attempted"
                    ],
                    "page_text": page_text,
                }
            )
        identity_result = resolve_initial_identities(results)

    return results, identity_result

def print_results(results: list[dict], identity_result: Any) -> None:
    """Print page-level extraction results."""
    for result in results:
        classification = result["classification"]
        fields = result["extraction"].get("fields", {})
        candidates = result["case_id_candidates"]

        page_number = result["page_number"]
        assignment = identity_result.assignments[page_number]

        print("=" * 80)
        print(f"Page {page_number}:")
        print(f"  Quality: {result['quality']}")
        print(f"  Text source: {result['text_source']}")
        print(f"  OCR strategy: {result['ocr_strategy']}")

        print(
            "  Classification: "
            f"{classification['document_type']} "
            f"(confidence={classification['confidence']:.2f})"
        )

        print(
            "  Case ID candidates: "
            f"header={candidates.header_case_id!r}, "
            f"footer={candidates.footer_case_id!r}, "
            f"internal={candidates.internal_case_id!r}, "
            f"mismatch={candidates.has_mismatch}"
        )

        print(
            "  Provisional case assignment: "
            f"case_id={assignment.case_id!r}, "
            f"method={assignment.assignment_method!r}, "
            f"mismatch={assignment.mismatch}"
        )

        print("  Extracted fields:")

        if not fields:
            print("    None")
        else:
            for field_name, value in fields.items():
                print(f"    {field_name}: {value!r}")

        print(
            "  Selected rotation: "
            f"{result['selected_rotation']} degrees"
        )

        print(
            "  Orientation retry attempted: "
            f"{result['orientation_retry_attempted']}"
        )

        print("  Page text:")
        print(f"    {result['page_text'][:500]}")

    print("\n=== INITIAL IDENTITY GROUPS ===")

    if identity_result.pages_by_case_id:
        for case_id, page_numbers in (
            identity_result.pages_by_case_id.items()
        ):
            print(f"{case_id}: pages {page_numbers}")
    else:
        print("No case groups resolved.")

    print(
        "Unassigned pages: "
        f"{identity_result.unassigned_pages}"
    )

def serialize_case_candidates(candidates) -> dict:
    return {
        "header_case_id": candidates.header_case_id,
        "footer_case_id": candidates.footer_case_id,
        "internal_case_id": candidates.internal_case_id,
        "mismatch": candidates.has_mismatch,
    }

def save_results(
    results: list[dict],
    output_path: Path,
) -> None:
    """Save results to JSON."""

    serializable_results = []

    for result in results:
        serializable_result = result.copy()
        serializable_result["case_id_candidates"] = (
            serialize_case_candidates(
                result["case_id_candidates"]
            )
        )
        serializable_results.append(serializable_result)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            serializable_results,
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

    results, identity_result = test_extraction(
        args.pdf_path
    )

    print_results(
        results=results,
        identity_result=identity_result,
    )

    if args.output is not None:
        save_results(
            results=results,
            output_path=args.output,
        )
        print(f"\nSaved results to {args.output}")


if __name__ == "__main__":
    main()