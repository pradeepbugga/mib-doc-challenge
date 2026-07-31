from __future__ import annotations

import argparse
from pathlib import Path
from pprint import pprint

from scripts.test_extraction import test_extraction

from core.adjudication.models import FieldObservation
from core.adjudication.normalizers import normalize_observations
from core.adjudication.corroborator import corroborate_packet

FIELD_ALIASES = {
    "declared_purpose": "purpose",
    "purpose": "purpose",

    "species_code": "species",
    "species_match": "species",
}

def build_observations(
    extraction_results: list[dict],
) -> list[FieldObservation]:
    """
    Convert page-level extraction results into canonical field observations.
    """
    observations: list[FieldObservation] = []

    for page_result in extraction_results:
        page_number = page_result["page_number"]
        text_source = page_result["text_source"]

        document_type = page_result[
            "classification"
        ]["document_type"]

        fields = page_result["extraction"].get(
            "fields",
            {},
        )

        for extracted_field, raw_value in fields.items():
            canonical_field = FIELD_ALIASES.get(
                extracted_field,
                extracted_field,
            )

            observations.append(
                FieldObservation(
                    field=canonical_field,
                    raw_value=raw_value,
                    document_type=document_type,
                    page_number=page_number,
                    text_source=text_source,
                )
            )

    return observations


def run_pipeline(pdf_path: Path):
    """
    Run extraction, normalization, and corroboration on one PDF packet.
    """

    # 1. Run your existing page-level extraction pipeline.
    extraction_results = test_extraction(pdf_path)

    # 2. Convert extraction dictionaries into observations.
    observations = build_observations(extraction_results)

    print("\n=== RAW OBSERVATIONS ===")

    for observation in observations:
        pprint(observation)

    # 3. Normalize field values.
    normalized_observations = normalize_observations(observations)

    print("\n=== NORMALIZED OBSERVATIONS ===")

    for observation in normalized_observations:
        pprint(observation)

    # 4. Corroborate observations across pages.
    packet = corroborate_packet(normalized_observations)

    print("\n=== RESOLVED PACKET ===")

    for field_name, resolved_field in packet.fields.items():
        print(
            f"{field_name}: "
            f"value={resolved_field.resolved_value!r}, "
            f"status={resolved_field.status}, "
            f"method={resolved_field.resolution_method}"
        )

    return packet


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run extraction, normalization, and corroboration "
            "on an MIB packet."
        )
    )

    parser.add_argument(
        "pdf_path",
        type=Path,
        help="Path to the PDF packet.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.pdf_path.is_file():
        raise FileNotFoundError(
            f"PDF not found: {args.pdf_path}"
        )

    run_pipeline(args.pdf_path)


if __name__ == "__main__":
    main()