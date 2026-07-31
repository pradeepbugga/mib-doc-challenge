from __future__ import annotations

import re
from dataclasses import dataclass


CASE_ID_VALUE = r"MIB-[A-Z0-9]{6}"


@dataclass
class PageCaseAssignment:
    page_number: int
    resolved_case_id: str | None
    header_case_id: str | None = None
    footer_case_id: str | None = None
    internal_case_id: str | None = None
    filename_case_id: str | None = None
    assignment_method: str | None = None
    mismatch: bool = False


def normalize_case_id_candidate(value: str | None) -> str | None:
    if value is None:
        return None

    value = value.upper().strip()
    value = re.sub(r"\s+", "", value)

    match = re.fullmatch(r"MIB-([A-Z0-9]{6})", value)

    if match is None:
        return None

    suffix = match.group(1).translate(
        str.maketrans(
            {
                "O": "0",
                "C": "0",
                "I": "1",
                "L": "1",
            }
        )
    )

    if not suffix.isdigit():
        return None

    return f"MIB-{suffix}"


def extract_case_id_candidates(text: str) -> dict[str, str | None]:
    footer_match = re.search(
        rf"\bPacket\s+({CASE_ID_VALUE})\s*/\s*page\b",
        text,
        flags=re.IGNORECASE,
    )

    header_match = re.search(
        rf"\b({CASE_ID_VALUE})\s*\|\s*MIB\s+Eyes\s+Only\b",
        text,
        flags=re.IGNORECASE,
    )

    internal_match = re.search(
        rf"\bCase\s+(?:ID|1D|10)\s*[:;.-]?\s*({CASE_ID_VALUE})\b",
        text,
        flags=re.IGNORECASE,
    )

    return {
        "footer_case_id": (
            normalize_case_id_candidate(footer_match.group(1))
            if footer_match
            else None
        ),
        "header_case_id": (
            normalize_case_id_candidate(header_match.group(1))
            if header_match
            else None
        ),
        "internal_case_id": (
            normalize_case_id_candidate(internal_match.group(1))
            if internal_match
            else None
        ),
    }


def resolve_page_case_assignment(
    *,
    page_number: int,
    text: str,
    filename_case_id: str | None,
) -> PageCaseAssignment:
    candidates = extract_case_id_candidates(text)

    footer_case_id = candidates["footer_case_id"]
    header_case_id = candidates["header_case_id"]
    internal_case_id = candidates["internal_case_id"]

    structural_values = [
        value
        for value in (footer_case_id, header_case_id)
        if value is not None
    ]

    if structural_values:
        resolved_case_id = structural_values[0]
        assignment_method = "header_footer"
    elif internal_case_id is not None:
        resolved_case_id = internal_case_id
        assignment_method = "internal_field"
    else:
        resolved_case_id = filename_case_id
        assignment_method = (
            "filename_fallback"
            if filename_case_id is not None
            else None
        )

    observed_values = {
        value
        for value in (
            footer_case_id,
            header_case_id,
            internal_case_id,
        )
        if value is not None
    }

    mismatch = len(observed_values) > 1

    return PageCaseAssignment(
        page_number=page_number,
        resolved_case_id=resolved_case_id,
        header_case_id=header_case_id,
        footer_case_id=footer_case_id,
        internal_case_id=internal_case_id,
        filename_case_id=filename_case_id,
        assignment_method=assignment_method,
        mismatch=mismatch,
    )