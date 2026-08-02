from __future__ import annotations

from datetime import datetime
from difflib import SequenceMatcher
from core.adjudication.ontology import (
    VALID_DECLARED_PURPOSES,
    VALID_HOME_WORLDS,
    VALID_RISK_FLAGS,
    VALID_SPECIES_CODES,
    VALID_VISA_CLASSES,
    best_vocabulary_match,
)

import unicodedata
import re


def strip_accents(value: str) -> str:
    """Convert accented characters to their unaccented equivalents."""
    decomposed = unicodedata.normalize("NFKD", value)

    return "".join(
        character
        for character in decomposed
        if not unicodedata.combining(character)
    )

def normalize_comparison_text(value: str) -> str:
    """
    Normalize noisy OCR text for vocabulary comparison.
    """
    value = strip_accents(value)
    value = value.casefold().strip()

    value = re.sub(r"[\s_-]+", " ", value)
    value = re.sub(r"[^a-z0-9 ]+", "", value)
    value = re.sub(r"\s+", " ", value)

    return value.strip()

def normalize_case_id(case_id: str | None) -> str | None:
    """Normalize an extracted MIB case ID for comparison."""

    if not case_id:
        return None

    value = case_id.upper().strip()

    value = re.sub(r"\s+", "", value)

    value = re.sub(r"^MIB[-_]?", "MIB-", value)

    suffix = value[4:]

    suffix = suffix.replace("O", "0")
    suffix = suffix.replace("C", "0")
    suffix = suffix.replace("I", "1")
    suffix = suffix.replace("L", "1")

    return "MIB-" + suffix
def normalize_declared_purpose(
    value: str | None,
) -> str | None:
    if value is None:
        return None

    cleaned = value.strip().casefold()
    cleaned = re.sub(r"\s+", " ", cleaned)

    if cleaned in VALID_DECLARED_PURPOSES:
        return cleaned

    return best_vocabulary_match(
        cleaned,
        VALID_DECLARED_PURPOSES,
        minimum_score=0.72,
        minimum_margin=0.08,
    )

def normalize_sponsor_id(sponsor: str | None) -> str | None:
    """Normalize an extracted sponsor ID."""

    if not sponsor:
        return None

    value = sponsor.upper().strip()

    value = re.sub(r"\s+", "", value)

    value = re.sub(r"^SPN[-_]?", "SPN-", value)

    suffix = value[4:]

    suffix = suffix.replace("O", "0")
    suffix = suffix.replace("I", "1")
    suffix = suffix.replace("L", "1")

    return "SPN-" + suffix

def normalize_visa_class(
    value: str | None,
) -> str | None:
    if value is None:
        return None

    cleaned = value.strip().upper()
    cleaned = re.sub(r"\s+", "", cleaned)

    aliases = {
        "XW1": "XW-1",
        "XW2": "XW-2",
        "MED3": "MED-3",
        "DIP1": "DIP-1",
        "TRANSIT7": "TRANSIT-7",
    }

    cleaned = aliases.get(cleaned, cleaned)

    if cleaned in VALID_VISA_CLASSES:
        return cleaned

    return best_vocabulary_match(
        cleaned,
        VALID_VISA_CLASSES,
        minimum_score=0.75,
        minimum_margin=0.10,
    )

def normalize_date(date: str | None) -> str | None:
    """Normalize dates to ISO format."""

    if not date:
        return None

    for fmt in (
        "%Y-%m-%d",
        "%Y/%m/%d",
        "%Y.%m.%d",
        "%Y-%m-%d",
        "%Y-%m-%d",
    ):
        try:
            return datetime.strptime(date, fmt).strftime("%Y-%m-%d")
        except ValueError:
            pass

    return date

def normalize_fee_status(status: str | None) -> str | None:
    if not status:
        return None

    value = status.lower().strip()

    value = value.replace("1", "i")
    value = value.replace("0", "o")

    aliases = {
        "paid": "paid",
        "waived": "waived",
        "unpaid": "unpaid",
        "unknown": "unknown",
    }

    return aliases.get(value, value)

def normalize_species_code(
    value: str | None,
) -> str | None:
    if value is None:
        return None

    cleaned = value.strip().upper()
    cleaned = re.sub(r"[\s-]+", "_", cleaned)
    cleaned = re.sub(r"_+", "_", cleaned)
    cleaned = cleaned.strip("_")

    if cleaned in VALID_SPECIES_CODES:
        return cleaned

    return best_vocabulary_match(
        cleaned,
        VALID_SPECIES_CODES,
        minimum_score=0.72,
        minimum_margin=0.08,
    )

def normalize_waiver_code(value: str | None) -> str | None:
    if value is None:
        return None

    normalized = value.strip().upper()

    aliases = {
        "NA": "N/A",
        "N-A": "N/A",
        "NOT APPLICABLE": "N/A",
    }

    return aliases.get(normalized, normalized)

def normalize_home_world(
    value: str | None,
) -> str | None:
    if value is None:
        return None

    cleaned = strip_accents(value)
    cleaned = " ".join(cleaned.split()).strip()

    exact_lookup = {
        normalize_comparison_text(candidate): candidate
        for candidate in VALID_HOME_WORLDS
    }

    normalized_cleaned = normalize_comparison_text(cleaned)

    exact_match = exact_lookup.get(normalized_cleaned)

    if exact_match is not None:
        return exact_match

    return best_vocabulary_match(
        cleaned,
        VALID_HOME_WORLDS,
        minimum_score=0.72,
        minimum_margin=0.08,
    )

DISQUALIFYING_RISK_FLAGS = {
    "memory_tampering",
    "planetary_embargo",
    "active_warrant",
    "biohazard_red",
}

REVIEW_ONLY_RISK_FLAGS = {
    "identity_conflict",
    "sponsor_mismatch",
    "illegible_biometrics",
    "rescinded_denial",
}

VALID_RISK_FLAGS = (
    DISQUALIFYING_RISK_FLAGS
    | REVIEW_ONLY_RISK_FLAGS
    | {"none"}
)


def best_flag_match(value: str) -> str | None:
    scores = sorted(
        (
            (
                SequenceMatcher(None, value, candidate).ratio(),
                candidate,
            )
            for candidate in VALID_RISK_FLAGS
        ),
        reverse=True,
    )

    best_score, best_flag = scores[0]
    second_score = scores[1][0]

    if best_score < 0.72:
        return None

    if best_score - second_score < 0.08:
        return None

    return best_flag




def normalize_risk_flag(value: str | None) -> str | None:
    if value is None:
        return None

    normalized = value.strip().lower()
    normalized = re.sub(r"[\s-]+", "_", normalized)
    normalized = re.sub(r"_+", "_", normalized)
    normalized = normalized.strip("_")

    if normalized == "none":
        return "none"

    if normalized in VALID_RISK_FLAGS:
        return normalized

    return best_flag_match(normalized)

def normalize_risk_flags(value: str | None) -> str | None:
    if value is None:
        return None

    raw_flags = re.split(r"[|,;]+", value)

    normalized_flags = []

    for raw_flag in raw_flags:
        normalized = normalize_risk_flag(raw_flag)

        if normalized is not None:
            normalized_flags.append(normalized)

    if not normalized_flags:
        return None

    unique_flags = sorted(set(normalized_flags))

    if unique_flags == ["none"]:
        return "none"

    unique_flags = [
        flag
        for flag in unique_flags
        if flag != "none"
    ]

    return "|".join(unique_flags)


NORMALIZERS = {
    "case_id": normalize_case_id,
    "sponsor_id": normalize_sponsor_id,
    "visa_class": normalize_visa_class,
    "arrival_date": normalize_date,
    "species_code": normalize_species_code,
    "home_world": normalize_home_world,
    "declared_purpose": normalize_declared_purpose,
    "fee_status": normalize_fee_status,
    "risk_flags": normalize_risk_flags,
}

def normalize_observations(
    observations: list[FieldObservation],
) -> list[FieldObservation]:
    """
    Normalize every observation in a packet.
    """

    for observation in observations:

        normalizer = NORMALIZERS.get(observation.field)

        if normalizer is not None:
            observation.normalized_value = normalizer(observation.raw_value)
        else:
            observation.normalized_value = observation.raw_value

    return observations

def main():
    raw_value='GiieSé-5619)'

    # Normalize the raw home world value
    normalized_value = normalize_home_world(raw_value)
    print(f"Raw value: {raw_value}")
    print(f"Normalized value: {normalized_value}")

if __name__ == "__main__":
    main()