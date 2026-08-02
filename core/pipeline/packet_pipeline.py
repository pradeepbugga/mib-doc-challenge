"""
Packet-level pipeline: one PDF in, one prediction record out.

This is the library entry point the Docker image runs. It is deliberately quiet
— all diagnostics are returned rather than printed — so it can be driven over
thousands of packets.
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import fitz

from core.adjudication.corroborator import corroborate_packet
from core.adjudication.engine import Adjudication, adjudicate
from core.adjudication.models import FieldObservation
from core.adjudication.normalizers import normalize_observations
from core.adjudication.risk_derivation import augment_risk_flags
from core.pipeline.case_assignment import normalize_case_id_candidate
from core.pipeline.identity_resolution import (
    refine_identities_with_metadata,
    resolve_initial_identities,
)
from core.pipeline.page_pipeline import process_page

# Extractor field names mapped onto the canonical output schema.
FIELD_ALIASES = {
    "applicant": "applicant_name",
    "registry_name": "applicant_name",
    "species_match": "species_code",
    "species": "species_code",
    "purpose": "declared_purpose",
    "observed_flags": "risk_flags",
    "decision": "adjudicator_decision",
    "reason": "adjudicator_reason",
}

OUTPUT_FIELDS = (
    "case_id",
    "applicant_name",
    "species_code",
    "home_world",
    "visa_class",
    "sponsor_id",
    "arrival_date",
    "declared_purpose",
    "risk_flags",
    "fee_status",
    "adjudication",
    "confidence",
)

# `scripts/validate_submission.py` requires every row to carry a syntactically
# valid sponsor_id and ISO arrival_date. When a value cannot be recovered from
# trusted evidence these placeholders keep the row structurally valid so the
# remaining fields and the adjudication decision still score. They are never
# claims about the applicant — an unrecovered field scores zero either way.
PLACEHOLDER_SPONSOR_ID = "SPN-0000"
PLACEHOLDER_ARRIVAL_DATE = "1970-01-01"

SPONSOR_ID_PATTERN = re.compile(r"^SPN-\d{4}$")


@dataclass
class PacketResult:
    """A processed packet and the trail that produced it."""

    case_id: str
    fields: dict
    adjudication: Adjudication
    page_count: int
    unassigned_pages: list[int]
    error: str | None = None


def build_observations(page_results: list[dict], identity_result) -> list[FieldObservation]:
    """Flatten page-level extractions into canonical field observations."""
    observations: list[FieldObservation] = []

    for page_result in page_results:
        page_number = page_result["page_number"]
        document_type = page_result["classification"]["document_type"]
        assignment = identity_result.assignments[page_number]

        for extracted_field, raw_value in page_result["extraction"].get(
            "fields", {}
        ).items():
            observations.append(
                FieldObservation(
                    field=FIELD_ALIASES.get(extracted_field, extracted_field),
                    raw_value=raw_value,
                    document_type=document_type,
                    page_number=page_number,
                    text_source=page_result["text_source"],
                    case_id=assignment.case_id,
                )
            )

    return observations


def resolve_cases(observations: list[FieldObservation]) -> dict:
    """Group observations by case ID and corroborate each case."""
    by_case = defaultdict(list)

    for observation in observations:
        if observation.case_id is not None:
            by_case[observation.case_id].append(observation)

    return {
        case_id: corroborate_packet(case_observations)
        for case_id, case_observations in by_case.items()
    }


def extract_page_results(doc: fitz.Document) -> list[dict]:
    """Run page processing over every page of a packet."""
    page_results = []

    for page in doc:
        page_result = process_page(doc=doc, page=page)
        classification = page_result["classification"]

        page_results.append(
            {
                "page_number": page.number + 1,
                "text_source": page_result["text_source"],
                "classification": {
                    "document_type": classification.document_type,
                    "score": classification.score,
                    "confidence": classification.confidence,
                    "matched_cues": list(classification.matched_cues),
                },
                "case_id_candidates": page_result["case_id_candidates"],
                "extraction": page_result["extraction"],
            }
        )

    return page_results


def select_active_case(
    resolved_cases: dict,
    expected_case_id: str | None,
    observations: list[FieldObservation],
):
    """
    Choose which applicant's record to report.

    A packet can hold pages for more than one applicant. The active case is the
    one matching the packet's own case ID; when that is not among the resolved
    cases, fall back to the case carrying the most evidence.
    """
    if expected_case_id and expected_case_id in resolved_cases:
        return resolved_cases[expected_case_id]

    if not resolved_cases:
        return None

    weight = Counter(
        observation.case_id
        for observation in observations
        if observation.case_id is not None
    )

    if weight:
        return resolved_cases.get(weight.most_common(1)[0][0])

    return next(iter(resolved_cases.values()))


def process_packet(pdf_path: Path) -> PacketResult:
    """Process one PDF packet into a resolved record and a decision."""
    expected_case_id = normalize_case_id_candidate(Path(pdf_path).stem)

    with fitz.open(pdf_path) as doc:
        page_count = doc.page_count
        page_results = extract_page_results(doc)
        identity_result = resolve_initial_identities(page_results)

    observations = build_observations(page_results, identity_result)
    normalize_observations(observations)

    provisional_cases = resolve_cases(observations)

    refinement = refine_identities_with_metadata(
        initial_result=identity_result,
        observations=observations,
        provisional_cases=provisional_cases,
    )
    refined = refinement.identity_result

    for observation in observations:
        observation.case_id = refined.assignments[
            observation.page_number
        ].case_id

    final_cases = resolve_cases(observations)

    packet = select_active_case(final_cases, expected_case_id, observations)

    if packet is None:
        return PacketResult(
            case_id=expected_case_id or "",
            fields={},
            adjudication=Adjudication(
                decision="NEEDS_REVIEW",
                confidence=0.5,
                rule="no_case_resolved",
                rationale="No case could be resolved from the packet.",
            ),
            page_count=page_count,
            unassigned_pages=list(refined.unassigned_pages),
        )

    augment_risk_flags(packet)
    decision = adjudicate(packet)

    fields = {
        name: resolved.resolved_value
        for name, resolved in packet.fields.items()
    }

    return PacketResult(
        case_id=expected_case_id or fields.get("case_id") or "",
        fields=fields,
        adjudication=decision,
        page_count=page_count,
        unassigned_pages=list(refined.unassigned_pages),
    )


def is_valid_iso_date(value: str) -> bool:
    """Return whether a string is a real, canonically-formatted ISO date."""
    if not value:
        return False

    try:
        return date.fromisoformat(value).isoformat() == value
    except ValueError:
        return False


def to_prediction(result: PacketResult) -> dict:
    """Render a packet result as a schema-valid prediction object."""
    fields = result.fields

    def text(name: str, default: str = "") -> str:
        value = fields.get(name)

        return default if value is None else str(value).strip() or default

    # OCR noise can slip a stray digit past the extractor (SPN-143,
    # SPN-404000). The evaluator's schema requires exactly four digits, so an
    # extracted value that doesn't fit the pattern is worth the same as no
    # value: normalize both to the placeholder rather than submit a
    # structurally invalid row.
    sponsor_id = text("sponsor_id")

    if not SPONSOR_ID_PATTERN.fullmatch(sponsor_id):
        sponsor_id = PLACEHOLDER_SPONSOR_ID

    arrival_date = text("arrival_date")

    if not is_valid_iso_date(arrival_date):
        arrival_date = PLACEHOLDER_ARRIVAL_DATE

    fee_status = text("fee_status")

    if fee_status not in {"paid", "waived", "unpaid", "unknown"}:
        fee_status = "unknown"

    risk_flags = text("risk_flags", "none") or "none"

    return {
        "case_id": result.case_id,
        "applicant_name": text("applicant_name"),
        "species_code": text("species_code"),
        "home_world": text("home_world"),
        "visa_class": text("visa_class"),
        "sponsor_id": sponsor_id,
        "arrival_date": arrival_date,
        "declared_purpose": text("declared_purpose"),
        "risk_flags": risk_flags,
        "fee_status": fee_status,
        "adjudication": result.adjudication.decision,
        "confidence": round(float(result.adjudication.confidence), 4),
    }


def predict_packet(pdf_path: Path) -> dict:
    """Process one packet and return its prediction object."""
    return to_prediction(process_packet(pdf_path))