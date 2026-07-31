from datetime import datetime
from difflib import SequenceMatcher


import re


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

def normalize_visa_class(visa: str | None) -> str |None:
    """Normalize visa classes."""

    if not visa:
        return None

    value = visa.upper()

    value = re.sub(r"\s+", "", value)

    value = value.replace("_", "-")

    patterns = [
        (r"^XW-?([12])$", r"XW-\1"),
        (r"^DIP-?1$", "DIP-1"),
        (r"^MED-?3$", "MED-3"),
        (r"^TRANSIT-?7$", "TRANSIT-7"),
    ]

    for pattern, replacement in patterns:
        if re.match(pattern, value):
            return re.sub(pattern, replacement, value)

    return value


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

def normalize_species_code(code: str | None) -> str | None:
    if not code:
        return None

    value = code.upper().strip()

    value = re.sub(r"\s+", "_", value)

    value = value.replace("__", "_")

    return value
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



def normalize_text(
    value: str | None,
) -> str | None:
    if value is None:
        return None

    normalized = " ".join(value.split()).strip()

    return normalized or None

NORMALIZERS = {
    "case_id": normalize_case_id,
    "sponsor_id": normalize_sponsor_id,
    "visa_class": normalize_visa_class,
    "arrival_date": normalize_date,
    "species_code": normalize_species_code,
    "fee_status": normalize_fee_status,
    "waiver_code": normalize_waiver_code,
    "purpose": normalize_text,
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

