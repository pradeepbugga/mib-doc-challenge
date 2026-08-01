from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field, replace
from typing import Any

from core.adjudication.models import FieldObservation, Packet

from core.pipeline.case_assignment import CaseIdCandidates


@dataclass
class PageIdentityAssignment:
    page_number: int
    case_id: str | None
    assignment_method: str
    candidates: CaseIdCandidates
    mismatch: bool = False
    linkage_fields: tuple[str, ...] = ()


@dataclass
class IdentityResolutionResult:
    assignments: dict[int, PageIdentityAssignment]
    pages_by_case_id: dict[str, list[int]]
    unassigned_pages: list[int] = field(default_factory=list)


@dataclass(frozen=True)
class MetadataLinkResult:
    page_number: int
    assigned_case_id: str | None
    best_score: float
    second_best_score: float
    matched_fields: tuple[str, ...] = ()
    conflicting_fields: tuple[str, ...] = ()
    assignment_method: str = "unassigned"


@dataclass
class IdentityRefinementResult:
    identity_result: IdentityResolutionResult
    linkage_results: dict[int, MetadataLinkResult]

METADATA_MATCH_WEIGHTS = {
    "sponsor_id": 6.0,
    "applicant_name": 5.0,
    "species_code": 3.0,
    "arrival_date": 3.0,
    "home_world": 2.0,
    "visa_class": 2.0,
    "declared_purpose": 1.0,
}

METADATA_CONFLICT_PENALTIES = {
    "sponsor_id": 7.0,
    "applicant_name": 5.0,
    "species_code": 3.0,
    "arrival_date": 3.0,
    "home_world": 2.0,
    "visa_class": 2.0,
    "declared_purpose": 1.0,
}


def choose_structural_case_id(
    candidates: CaseIdCandidates,
) -> tuple[str | None, str]:
    """
    Resolve a page from page-local structural evidence only.

    Precedence:
        footer
        header
        internal field
        unassigned
    """
    if candidates.footer_case_id is not None:
        return candidates.footer_case_id, "footer"

    if candidates.header_case_id is not None:
        return candidates.header_case_id, "header"

    if candidates.internal_case_id is not None:
        return candidates.internal_case_id, "internal_field"

    return None, "unassigned"


def resolve_initial_identities(
    page_results: list[dict[str, Any]],
) -> IdentityResolutionResult:
    assignments: dict[int, PageIdentityAssignment] = {}
    pages_by_case_id: dict[str, list[int]] = defaultdict(list)
    unassigned_pages: list[int] = []

    for result in page_results:
        page_number = result["page_number"]
        candidates = result["case_id_candidates"]

        case_id, method = choose_structural_case_id(
            candidates
        )

        assignment = PageIdentityAssignment(
            page_number=page_number,
            case_id=case_id,
            assignment_method=method,
            candidates=candidates,
            mismatch=candidates.has_mismatch,
        )

        assignments[page_number] = assignment

        if case_id is None:
            unassigned_pages.append(page_number)
        else:
            pages_by_case_id[case_id].append(page_number)

    return IdentityResolutionResult(
        assignments=assignments,
        pages_by_case_id=dict(pages_by_case_id),
        unassigned_pages=unassigned_pages,
    )

def normalized_value(
    observation: FieldObservation,
) -> str | None:
    value = observation.normalized_value

    if value is None:
        return None

    value = str(value).strip()

    return value or None

def resolved_case_values(
    packet: Packet,
) -> dict[str, str]:
    """
    Return usable resolved metadata for one provisional case.
    """
    values: dict[str, str] = {}

    for field_name in METADATA_MATCH_WEIGHTS:
        resolved_field = packet.fields.get(field_name)

        if (
            resolved_field is None
            or resolved_field.resolved_value is None
        ):
            continue

        value = str(
            resolved_field.resolved_value
        ).strip()

        if value:
            values[field_name] = value

    return values

def page_metadata_values(
    observations: list[FieldObservation],
) -> dict[str, set[str]]:
    """
    Collect normalized metadata values found on one page.
    """
    values: dict[str, set[str]] = {}

    for observation in observations:
        if observation.field not in METADATA_MATCH_WEIGHTS:
            continue

        value = normalized_value(observation)

        if value is None:
            continue

        values.setdefault(
            observation.field,
            set(),
        ).add(value)

    return values

def score_page_against_case(
    page_observations: list[FieldObservation],
    packet: Packet,
) -> tuple[float, tuple[str, ...], tuple[str, ...]]:
    """
    Score one unassigned page against one provisional case.

    Matches add field-specific weight. Explicit differing values
    subtract a field-specific penalty.
    """
    page_values = page_metadata_values(
        page_observations
    )
    case_values = resolved_case_values(packet)

    score = 0.0
    matched_fields: list[str] = []
    conflicting_fields: list[str] = []

    for field_name, page_field_values in page_values.items():
        case_value = case_values.get(field_name)

        if case_value is None:
            continue

        if case_value in page_field_values:
            score += METADATA_MATCH_WEIGHTS[field_name]
            matched_fields.append(field_name)
        else:
            score -= METADATA_CONFLICT_PENALTIES[
                field_name
            ]
            conflicting_fields.append(field_name)

    return (
        score,
        tuple(matched_fields),
        tuple(conflicting_fields),
    )
def metadata_assignment_is_safe(
    *,
    score: float,
    second_best_score: float,
    matched_fields: tuple[str, ...],
    conflicting_fields: tuple[str, ...],
) -> bool:
    """
    Require strong, non-ambiguous metadata evidence.

    Initial policy:
    - at least two matching fields
    - score of at least 7
    - margin of at least 3 over the next case
    - no conflict in applicant_name or sponsor_id
    """
    if len(matched_fields) < 2:
        return False

    if score < 7.0:
        return False

    if score - second_best_score < 3.0:
        return False

    critical_conflicts = {
        "applicant_name",
        "sponsor_id",
    }

    if critical_conflicts.intersection(
        conflicting_fields
    ):
        return False

    return True

def refine_identities_with_metadata(
    *,
    initial_result: IdentityResolutionResult,
    observations: list[FieldObservation],
    provisional_cases: dict[str, Packet],
) -> IdentityRefinementResult:
    """
    Assign previously unassigned pages using corroborated metadata.

    Pages that already have a structural case ID are never reassigned.
    """
    observations_by_page: dict[
        int,
        list[FieldObservation],
    ] = {}

    for observation in observations:
        observations_by_page.setdefault(
            observation.page_number,
            [],
        ).append(observation)

    assignments = dict(initial_result.assignments)

    pages_by_case_id = {
        case_id: list(page_numbers)
        for case_id, page_numbers
        in initial_result.pages_by_case_id.items()
    }

    remaining_unassigned: list[int] = []
    linkage_results: dict[int, MetadataLinkResult] = {}

    for page_number in initial_result.unassigned_pages:
        page_observations = observations_by_page.get(
            page_number,
            [],
        )

        if not page_observations:
            remaining_unassigned.append(page_number)

            linkage_results[page_number] = MetadataLinkResult(
                page_number=page_number,
                assigned_case_id=None,
                best_score=0.0,
                second_best_score=0.0,
                assignment_method="no_metadata",
            )
            continue

        candidates: list[
            tuple[
                float,
                str,
                tuple[str, ...],
                tuple[str, ...],
            ]
        ] = []

        for case_id, packet in provisional_cases.items():
            (
                score,
                matched_fields,
                conflicting_fields,
            ) = score_page_against_case(
                page_observations=page_observations,
                packet=packet,
            )

            candidates.append(
                (
                    score,
                    case_id,
                    matched_fields,
                    conflicting_fields,
                )
            )

        candidates.sort(
            key=lambda candidate: candidate[0],
            reverse=True,
        )

        if not candidates:
            remaining_unassigned.append(page_number)

            linkage_results[page_number] = MetadataLinkResult(
                page_number=page_number,
                assigned_case_id=None,
                best_score=0.0,
                second_best_score=0.0,
                assignment_method="no_candidate_cases",
            )
            continue

        (
            best_score,
            best_case_id,
            matched_fields,
            conflicting_fields,
        ) = candidates[0]

        second_best_score = (
            candidates[1][0]
            if len(candidates) > 1
            else 0.0
        )

        safe = metadata_assignment_is_safe(
            score=best_score,
            second_best_score=second_best_score,
            matched_fields=matched_fields,
            conflicting_fields=conflicting_fields,
        )

        if not safe:
            remaining_unassigned.append(page_number)

            linkage_results[page_number] = MetadataLinkResult(
                page_number=page_number,
                assigned_case_id=None,
                best_score=best_score,
                second_best_score=second_best_score,
                matched_fields=matched_fields,
                conflicting_fields=conflicting_fields,
                assignment_method="ambiguous_metadata",
            )
            continue

        previous_assignment = assignments[page_number]

        assignments[page_number] = replace(
            previous_assignment,
            case_id=best_case_id,
            assignment_method="metadata_linkage",
            linkage_fields=matched_fields,
        )

        pages_by_case_id.setdefault(
            best_case_id,
            [],
        ).append(page_number)

        linkage_results[page_number] = MetadataLinkResult(
            page_number=page_number,
            assigned_case_id=best_case_id,
            best_score=best_score,
            second_best_score=second_best_score,
            matched_fields=matched_fields,
            conflicting_fields=conflicting_fields,
            assignment_method="metadata_linkage",
        )

    refined_result = IdentityResolutionResult(
        assignments=assignments,
        pages_by_case_id=pages_by_case_id,
        unassigned_pages=remaining_unassigned,
    )

    return IdentityRefinementResult(
        identity_result=refined_result,
        linkage_results=linkage_results,
    )