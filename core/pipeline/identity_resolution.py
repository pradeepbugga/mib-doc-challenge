from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

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