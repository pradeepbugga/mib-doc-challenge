from __future__ import annotations

from difflib import SequenceMatcher
import re
from typing import Iterable


VALID_SPECIES_CODES = {
    "TRIANGULAN",
    "JOVIAN_GASFORM",
    "CENTAURI_SYNTH",
    "LUNA_SECURID",
    "KAIJU_MICRO",
    "ORION_GRAYS",
    "ALPHA_DRACONIAN",
    "SIRIUS_AVIAN",
    "VENUSIAN_MYCELIAL",
    "AQUARIAN_MANTIS",
    "ARCTURIAN",
    "ANDROMEDAN",
}

VALID_DECLARED_PURPOSES = {
    "reactor maintenance",
    "field repair",
    "medical consult",
    "research",
    "cultural exchange",
    "translation",
    "archive audit",
    "xenobotany",
    "diplomatic",
    "transit",
}

VALID_HOME_WORLDS = {
    "Luyten-b",
    "Europa Station",
    "Titan Freeport",
    "Barnard-c",
    "Gliese-581g",
    "Mars Dome-7",
    "Kepler-186f",
    "Sirius Outpost",
    "Wolf-1061c",
    "Proxima-b",
    "Zeta Reticuli",
    "TRAPPIST-1e",
    "Eris Relay",
}

VALID_VISA_CLASSES = {
    "MED-3",
    "DIP-1",
    "XW-1",
    "XW-2",
    "TRANSIT-7",
}

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


def normalize_comparison_text(value: str) -> str:
    """
    Normalize text for ontology matching only.

    This does not determine the final display format.
    """
    value = value.casefold().strip()
    value = re.sub(r"[\s_-]+", " ", value)
    value = re.sub(r"[^a-z0-9 ]+", "", value)
    value = re.sub(r"\s+", " ", value)

    return value.strip()


def best_vocabulary_match(
    value: str,
    vocabulary: Iterable[str],
    *,
    minimum_score: float = 0.72,
    minimum_margin: float = 0.08,
) -> str | None:
    """
    Return the strongest unambiguous vocabulary match.

    A result is accepted only when:
    - the best similarity exceeds minimum_score; and
    - it exceeds the second-best match by minimum_margin.
    """
    normalized_value = normalize_comparison_text(value)

    if not normalized_value:
        return None

    scored_matches = sorted(
        (
            (
                SequenceMatcher(
                    None,
                    normalized_value,
                    normalize_comparison_text(candidate),
                ).ratio(),
                candidate,
            )
            for candidate in vocabulary
        ),
        reverse=True,
    )

    if not scored_matches:
        return None

    best_score, best_candidate = scored_matches[0]

    second_score = (
        scored_matches[1][0]
        if len(scored_matches) > 1
        else 0.0
    )

    if best_score < minimum_score:
        return None

    if best_score - second_score < minimum_margin:
        return None

    return best_candidate