from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field, replace
from difflib import SequenceMatcher
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


# A page's own case id reading this close to a case id shared by other pages
# in the same packet is OCR noise, not a second applicant -- confirmed on
# MIB-000557 page 4 (biometric_slip, OCR): its internal field read
# "MIB-O00657" against the packet's actual "MIB-000557" (0.90 similar), which
# silently formed its own single-page phantom case and discarded a
# correctly-read biohazard_red flag along with it. Native text is trusted
# exactly here, same as FUZZY_MATCH_FIELDS does for applicant_name --
# relaxing a native-text case id disagreement would defeat the reason this
# system exists (telling apart two real applicants sharing one packet).
CASE_ID_SIMILARITY = 0.85


def resolve_initial_identities(
    page_results: list[dict[str, Any]],
) -> IdentityResolutionResult:
    assignments: dict[int, PageIdentityAssignment] = {}
    pages_by_case_id: dict[str, list[int]] = defaultdict(list)
    unassigned_pages: list[int] = []

    raw_assignments: list[
        tuple[int, str | None, str, CaseIdCandidates, str | None]
    ] = []
    id_counts: Counter[str] = Counter()

    for result in page_results:
        page_number = result["page_number"]
        candidates = result["case_id_candidates"]
        text_source = result.get("text_source")

        case_id, method = choose_structural_case_id(
            candidates
        )

        raw_assignments.append(
            (page_number, case_id, method, candidates, text_source)
        )

        if case_id is not None:
            id_counts[case_id] += 1

    # Case ids two or more pages agree on are the packet's real identity --
    # only those are trusted as a target for the fuzzy rescue below.
    established_ids = [
        case_id for case_id, count in id_counts.items() if count > 1
    ]

    for page_number, case_id, method, candidates, text_source in raw_assignments:
        if (
            case_id is not None
            and id_counts[case_id] == 1
            and text_source == "ocr"
        ):
            for established_id in established_ids:
                if (
                    established_id != case_id
                    and SequenceMatcher(
                        None, case_id, established_id
                    ).ratio()
                    >= CASE_ID_SIMILARITY
                ):
                    case_id = None
                    method = "ocr_near_miss_relaxed"
                    break

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

def resolved_case_value_sets(
    packet: Packet,
) -> dict[str, set[str]]:
    """
    Return every distinct value observed for one provisional case's
    metadata, not only whichever one evidence-priority picked as the
    resolved value.

    A page's own name conflicting with the *resolved* value does not mean
    it conflicts with the case -- if the intake form and another page
    disagree on applicant_name (a known decoy pattern for that field, see
    mib-intake-name-decoy), a third page's reading only needs to match
    either one to plausibly belong to this case. Comparing against just the
    precedence winner manufactures a conflict out of a disagreement the
    packet already contained before this page showed up.
    """
    values: dict[str, set[str]] = {}

    for field_name in METADATA_MATCH_WEIGHTS:
        resolved_field = packet.fields.get(field_name)

        if resolved_field is None:
            continue

        field_values: set[str] = set()

        if resolved_field.resolved_value is not None:
            value = str(resolved_field.resolved_value).strip()

            if value:
                field_values.add(value)

        for observation in resolved_field.observations:
            value = normalized_value(observation)

            if value:
                field_values.add(value)

        if field_values:
            values[field_name] = field_values

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

# A page value this similar to the case value is the same value read badly.
SAME_VALUE_SIMILARITY = 0.85

# Share of the case value that must survive inside a longer page value.
CONTAINED_COVERAGE = 0.80

# Only free-text fields are compared loosely. Every other metadata field either
# has a canonicalising normalizer (species_code, home_world, visa_class) or a
# fixed shape where a single character carries meaning: SPN-1680 and SPN-1690
# are 87% similar and are different sponsors.
FUZZY_MATCH_FIELDS = frozenset({"applicant_name"})


def _value_compatible_with_one(
    field_name: str,
    case_value: str,
    page_values: set[str],
) -> bool:
    """Return whether a page's values agree with one candidate case value."""
    if case_value in page_values:
        return True

    if field_name not in FUZZY_MATCH_FIELDS:
        return False

    case_lowered = case_value.casefold().strip()

    if not case_lowered:
        return False

    for page_value in page_values:
        page_lowered = page_value.casefold().strip()

        if not page_lowered:
            continue

        matcher = SequenceMatcher(None, case_lowered, page_lowered)

        if matcher.ratio() >= SAME_VALUE_SIMILARITY:
            return True

        covered = sum(
            block.size for block in matcher.get_matching_blocks()
        )

        if covered / len(case_lowered) >= CONTAINED_COVERAGE:
            return True

    return False


def values_compatible(
    field_name: str,
    case_values: set[str],
    page_values: set[str],
) -> bool:
    """
    Return whether a page's values agree with the case, allowing for OCR noise.

    Comparing exactly manufactures conflicts out of damage. On MIB-000063 the
    intake form's applicant read as 'Orirx Orivoss Spr~- "te. URION_GRAYS' --
    one dropped letter, plus the next field's text swallowed by a boundary
    regex whose anchor had garbled. Against the case's 'Oririx Orivoss' that
    scored as a conflicting applicant, which blocks assignment outright, so the
    page holding the only copy of declared_purpose was discarded.

    Two allowances for free-text fields: near-identical strings, and a case
    value that survives largely intact inside a longer page value. Matching
    blocks are summed rather than taking the longest, because a single dropped
    letter splits one run into two. Different applicants share little and still
    register as conflicts.

    `case_values` is every distinct value observed for this case (see
    `resolved_case_value_sets`), not just the precedence winner -- a match
    against any one of them counts as compatible.
    """
    return any(
        _value_compatible_with_one(field_name, case_value, page_values)
        for case_value in case_values
    )


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
    case_values = resolved_case_value_sets(packet)

    score = 0.0
    matched_fields: list[str] = []
    conflicting_fields: list[str] = []

    for field_name, page_field_values in page_values.items():
        case_value_set = case_values.get(field_name)

        if case_value_set is None:
            continue

        if values_compatible(field_name, case_value_set, page_field_values):
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

    # A packet that resolved to exactly one case has no ambiguity for an
    # orphan page to fall foul of.
    sole_case_id = (
        next(iter(provisional_cases))
        if len(provisional_cases) == 1
        else None
    )

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

        assignment_method = "metadata_linkage"

        if not safe and sole_case_id is not None and not conflicting_fields:
            # The strict gate exists to stop a page being attached to the wrong
            # applicant when a packet holds several. With exactly one case in
            # the packet there is no wrong case to choose, and the only real
            # risk — a page belonging to some other applicant — shows up as a
            # metadata conflict, which is excluded here.
            #
            # Without this, a page whose footer failed to OCR is dropped whole:
            # MIB-000008 page 2 matched the sole case on applicant_name but
            # scored 5.0 against a threshold of 7.0 with one matched field
            # against a minimum of two, so its sponsor_id, visa_class and
            # declared_purpose were all discarded despite being read correctly.
            safe = True
            best_case_id = sole_case_id
            assignment_method = "sole_case_fallback"

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
            assignment_method=assignment_method,
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
            assignment_method=assignment_method,
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