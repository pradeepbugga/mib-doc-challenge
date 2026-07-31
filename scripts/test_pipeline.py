from __future__ import annotations

import argparse
from pathlib import Path
from pprint import pprint
from collections import defaultdict

from scripts.test_extraction import test_extraction

from core.adjudication.models import FieldObservation, IdentityResolutionResult
from core.adjudication.normalizers import normalize_observations
from core.adjudication.corroborator import corroborate_packet

FIELD_ALIASES = {
    # Names
    "applicant": "applicant_name",
    "registry_name": "applicant_name",

    # Species
    "species_match": "species_code",
    "species": "species_code",

    # Purpose
    "purpose": "declared_purpose",

    # Risk flags
    "observed_flags": "risk_flags",

    # Human adjudicator note (keep separate)
    "decision": "adjudicator_decision",
    "reason": "adjudicator_reason",
}
def build_observations(
    extraction_results: list[dict],
    identity_result: IdentityResolutionResult
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

        assignment = identity_result.assignments[page_number]


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
                    case_id=assignment.case_id
                )
            )

    return observations


def run_pipeline(pdf_path: Path):
    """
    Run extraction, provisional identity grouping, normalization,
    and corroboration for every case found in one PDF.
    """

    # 1. Page-level extraction plus initial case-ID grouping.
    extraction_results, identity_result = test_extraction(
        pdf_path
    )

    # 2. Convert page results into observations using the
    # provisional PDF-level case assignments.
    observations = build_observations(
        extraction_results=extraction_results,
        identity_result=identity_result,
    )

    assigned_observations = [
        observation
        for observation in observations
        if observation.case_id is not None
    ]

    unassigned_observations = [
        observation
        for observation in observations
        if observation.case_id is None
    ]

    print("\n=== RAW OBSERVATIONS ===")

    for observation in observations:
        pprint(observation)

    # 3. Group assigned observations by case ID.
    observations_by_case_id = defaultdict(list)

    for observation in assigned_observations:
        observations_by_case_id[
            observation.case_id
        ].append(observation)

    # 4. Normalize and corroborate each case separately.
    resolved_cases = {}

    for case_id, case_observations in (
        observations_by_case_id.items()
    ):
        normalized_case_observations = normalize_observations(
            case_observations
        )

        print(
            f"\n=== NORMALIZED OBSERVATIONS: "
            f"{case_id} ==="
        )

        for observation in normalized_case_observations:
            pprint(observation)

        resolved_cases[case_id] = corroborate_packet(
            normalized_case_observations
        )

    # 5. Print provisional resolved cases.
    print("\n=== PROVISIONAL RESOLVED CASES ===")

    if not resolved_cases:
        print("No cases were resolved.")

    for case_id, packet in resolved_cases.items():
        print(f"\nCASE: {case_id}")

        for field_name, resolved_field in (
            packet.fields.items()
        ):
            print(
                f"{field_name}: "
                f"value={resolved_field.resolved_value!r}, "
                f"status={resolved_field.status}, "
                f"method={resolved_field.resolution_method}"
            )

            for observation in (
                resolved_field.supporting_observations
            ):
                print(
                    "    supported by: "
                    f"page={observation.page_number}, "
                    f"document={observation.document_type}, "
                    f"text_source={observation.text_source}, "
                    f"raw={observation.raw_value!r}, "
                    f"normalized="
                    f"{observation.normalized_value!r}"
                )

    # 6. Keep unresolved-page evidence separate for the later
    # metadata-linkage pass.
    print("\n=== UNASSIGNED OBSERVATIONS ===")

    print(
    "\nUnassigned pages:",
        identity_result.unassigned_pages,
    )

    print("\nUnassigned observations:")

    if not unassigned_observations:
        print("None")
    else:
        for observation in unassigned_observations:
            pprint(observation)

    return resolved_cases, unassigned_observations


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