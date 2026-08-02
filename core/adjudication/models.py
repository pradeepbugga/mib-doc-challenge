from __future__ import annotations

from dataclasses import dataclass, field as dataclass_field

@dataclass
class FieldObservation:
    field: str
    raw_value: str | None
    document_type: str
    page_number: int
    normalized_value: str | None = None
    trusted: bool = True
    text_source: str | None = None
    case_id: str | None = None

@dataclass
class ResolvedField:
    field: str
    resolved_value: str | None
    status: str
    observations: list[FieldObservation]
    supporting_observations: list[FieldObservation] = dataclass_field(
        default_factory=list
    )
    resolution_method: str | None = None

@dataclass
class Packet:
    fields: dict[str, ResolvedField]

@dataclass
class IdentityResolutionResult:
    assignments: dict[int, PageIdentityAssignment]
    pages_by_case_id: dict[str, list[int]]
    unassigned_pages: list[int] = dataclass_field(default_factory=list)
