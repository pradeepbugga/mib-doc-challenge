from __future__ import annotations

import argparse
from pathlib import Path
from pprint import pprint
from collections import defaultdict

from scripts.test_extraction import test_extraction

from core.adjudication.models import FieldObservation, IdentityResolutionResult
from core.adjudication.normalizers import normalize_observations
from core.adjudication.corroborator import corroborate_packet
from core.pipeline.identity_resolution import refine_identities_with_metadata

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

def apply_identity_assignments(
    observations: list[FieldObservation],
    identity_result,
) -> None:
    """
    Update observation case IDs from the current page assignments.
    """
    for observation in observations:
        assignment = identity_result.assignments[
            observation.page_number
        ]

        observation.case_id = assignment.case_id

def resolve_cases(
    observations: list[FieldObservation],
) -> dict[str, Packet]:
    """
    Group normalized observations by case ID and corroborate each case.
    """
    observations_by_case_id = defaultdict(list)

    for observation in observations:
        if observation.case_id is None:
            continue

        observations_by_case_id[
            observation.case_id
        ].append(observation)

    return {
        case_id: corroborate_packet(case_observations)
        for case_id, case_observations
        in observations_by_case_id.items()
    }



def run_pipeline(pdf_path: Path):
    """
    Run extraction, two-pass identity resolution, normalization,
    and final corroboration.
    """

    extraction_results, initial_identity_result = (
        test_extraction(pdf_path)
    )

    observations = build_observations(
        extraction_results=extraction_results,
        identity_result=initial_identity_result,
    )

    # Normalize every observation once, including observations
    # from currently unassigned pages.
    normalize_observations(observations)

    print("\n=== NORMALIZED OBSERVATIONS ===")

    for observation in observations:
        pprint(observation)

    # First-pass case records use structural case-ID assignment only.
    provisional_cases = resolve_cases(observations)

    print("\n=== PROVISIONAL CASES ===")

    print_resolved_cases(provisional_cases)

    # Second identity pass: link unassigned pages through metadata.
    refinement = refine_identities_with_metadata(
        initial_result=initial_identity_result,
        observations=observations,
        provisional_cases=provisional_cases,
    )

    refined_identity_result = refinement.identity_result

    print("\n=== METADATA LINKAGE ===")

    for page_number, linkage in (
        refinement.linkage_results.items()
    ):
        print(
            f"Page {page_number}: "
            f"case_id={linkage.assigned_case_id!r}, "
            f"method={linkage.assignment_method}, "
            f"score={linkage.best_score:.1f}, "
            f"second={linkage.second_best_score:.1f}, "
            f"matched={linkage.matched_fields}, "
            f"conflicts={linkage.conflicting_fields}"
        )

    # Apply second-pass assignments to every observation.
    apply_identity_assignments(
        observations=observations,
        identity_result=refined_identity_result,
    )

    # Rebuild case records now that more pages may be attached.
    final_cases = resolve_cases(observations)

    print("\n=== FINAL RESOLVED CASES ===")

    print_resolved_cases(final_cases)

    unassigned_observations = [
        observation
        for observation in observations
        if observation.case_id is None
    ]

    print(
        "\nUnassigned pages:",
        refined_identity_result.unassigned_pages,
    )

    print("\nUnassigned observations:")

    if not unassigned_observations:
        print("None")
    else:
        for observation in unassigned_observations:
            pprint(observation)

    return (
        final_cases,
        refined_identity_result,
        unassigned_observations,
    )


def print_resolved_cases(
    resolved_cases: dict[str, Packet],
) -> None:
    if not resolved_cases:
        print("No cases resolved.")
        return

    for case_id, packet in resolved_cases.items():
        print(f"\nCASE: {case_id}")

        for field_name, resolved_field in packet.fields.items():
            print(
                f"{field_name}: "
                f"value={resolved_field.resolved_value!r}, "
                f"status={resolved_field.status}, "
                f"method={resolved_field.resolution_method}"
            )

            
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