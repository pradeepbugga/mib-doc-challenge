from __future__ import annotations

import re
from dataclasses import dataclass


CASE_ID_VALUE = r"MIB-[A-Z0-9]{6}"


@dataclass
class CaseIdCandidates:
    header_case_id: str | None = None
    footer_case_id: str | None = None
    internal_case_id: str | None = None

    @property
    def observed_values(self) -> set[str]:
        return {
            value
            for value in (
                self.header_case_id,
                self.footer_case_id,
                self.internal_case_id,
            )
            if value is not None
        }

    @property
    def has_mismatch(self) -> bool:
        return len(self.observed_values) > 1


def normalize_case_id_candidate(
    value: str | None,
) -> str | None:
    if value is None:
        return None

    value = value.upper().strip()
    value = re.sub(r"\s+", "", value)

    match = re.fullmatch(
        r"MIB-([A-Z0-9]{6})",
        value,
    )

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


def extract_case_id_candidates(
    text: str,
) -> CaseIdCandidates:
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
        rf"\b(?:Case|Cave)\s+(?:ID|1D|10)"
        rf"\s*[:;.-]?\s*({CASE_ID_VALUE})\b",
        text,
        flags=re.IGNORECASE,
    )

    return CaseIdCandidates(
        footer_case_id=(
            normalize_case_id_candidate(footer_match.group(1))
            if footer_match
            else None
        ),
        header_case_id=(
            normalize_case_id_candidate(header_match.group(1))
            if header_match
            else None
        ),
        internal_case_id=(
            normalize_case_id_candidate(internal_match.group(1))
            if internal_match
            else None
        ),
    )