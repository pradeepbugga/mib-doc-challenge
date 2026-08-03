"""
Recover fields whose label was garbled by OCR.

The strict extractors anchor on an exact label — `Fee\\s+Status`, `Home\\s+World`
— and return nothing when OCR mangles it. On damaged scans that happens
constantly while the *value* survives intact, because a value is short enough
to sit inside one scan band whereas a longer label straddles a band boundary.
Observed on real pages: `Home Workt Proxima-b` (value perfect, label broken),
`Foe Status: paid`, `Appiicent Zevese Qorul`, `Declared Pursosa: xencbotany`.

This module applies the same idea `core.adjudication.ontology` already uses for
field *values* — accept the strongest unambiguous fuzzy match — to field
*labels*. It runs only as a fallback, for fields the strict extractor could not
fill, so pages that OCR cleanly are unaffected.
"""

from __future__ import annotations

import re

from core.adjudication.ontology import (
    VALID_DECLARED_PURPOSES,
    VALID_HOME_WORLDS,
    VALID_RISK_FLAGS,
    VALID_SPECIES_CODES,
    VALID_VISA_CLASSES,
    best_vocabulary_match,
    normalize_comparison_text,
)

# Fields whose value is drawn from a closed vocabulary. For these the value can
# be recovered by searching the text after the label for the best vocabulary
# match, instead of trusting where the value happens to end — OCR noise around
# the value ("paid (mcnive fom) —— ld") otherwise leaks into the field.
FIELD_VOCABULARIES: dict[str, set[str]] = {
    "species_code": VALID_SPECIES_CODES,
    "species_match": VALID_SPECIES_CODES,
    "home_world": VALID_HOME_WORLDS,
    "visa_class": VALID_VISA_CLASSES,
    "declared_purpose": VALID_DECLARED_PURPOSES,
    "purpose": VALID_DECLARED_PURPOSES,
    "fee_status": {"paid", "waived", "unpaid", "unknown"},
    "observed_flags": VALID_RISK_FLAGS,
    "decision": {"APPROVED", "DENIED", "NEEDS_REVIEW", "REVIEW"},
}

# Longest vocabulary entry, in words ("reactor maintenance", "Europa Station").
MAX_VOCABULARY_WORDS = 2

# Fields with a fixed shape. Matching the shape inside the window stops the
# next line's text being swallowed when its label fails to match and therefore
# does not terminate the value.
FIELD_PATTERNS: dict[str, str] = {
    "case_id": r"\bM[I1l]B[-\s]?[0-9OoIl]{6}\b",
    "sponsor_id": r"\bSPN[-\s]?[0-9OoIl]{4}\b",
    "arrival_date": r"\b\d{4}[-/.]\d{1,2}[-/.]\d{1,2}\b",
    "amount": r"\$?\s*[\d,]+\.\d{2}\b",
    "biometric_confidence": r"\b\d{1,3}\s*%",
}

# Canonical labels per document type, mapped to the extractor field they fill.
# Order matters only for readability; matching is by similarity.
DOCUMENT_LABELS: dict[str, dict[str, str]] = {
    "intake_form": {
        "Case ID": "case_id",
        "Applicant": "applicant",
        "Species Code": "species_code",
        "Home World": "home_world",
        "Visa Class": "visa_class",
        "Sponsor ID": "sponsor_id",
        "Arrival Date": "arrival_date",
        "Declared Purpose": "declared_purpose",
        # Damage often takes the first word of a two-word label with it, and
        # the surviving half is then too short to match the full phrase:
        # "Declared Purpose" reaching OCR as "Pei Pupose" matches neither.
        # Only distinctive second words are aliased — "Code", "Class", "ID",
        # "Date" and "Status" each belong to several labels and would
        # mis-assign.
        "Purpose": "declared_purpose",
        "World": "home_world",
    },
    "fee_receipt": {
        "Case ID": "case_id",
        "Fee Status": "fee_status",
        "Amount": "amount",
        "Waiver Code": "waiver_code",
    },
    "registry_extract": {
        "Registry Name": "applicant",
        "Home World": "home_world",
        "Species Code": "species_code",
        "Registry Status": "registry_status",
        "Arrival Date": "arrival_date",
    },
    "biometric_slip": {
        "Case ID": "case_id",
        "Applicant": "applicant",
        "Species Match": "species_match",
        "Biometric confidence": "biometric_confidence",
        "Observed flags": "observed_flags",
    },
    "sponsor_attestation": {
        "Case ID": "case_id",
        "Sponsor ID": "sponsor_id",
        "Applicant": "applicant",
        "Purpose": "purpose",
        "Visa Class": "visa_class",
    },
    "adjudicator_note": {
        "Finding": "decision",
        "Reason": "reason",
    },
}

# An unclassified page could be any form, so offer every label.
ALL_LABELS: dict[str, str] = {
    label: field
    for labels in DOCUMENT_LABELS.values()
    for label, field in labels.items()
}

# Labels are matched more strictly than values. A label is a fixed phrase we
# expect verbatim, so a weak match is far more likely to be a coincidence than
# a damaged label, and a wrong label match silently mis-assigns a value.
MINIMUM_LABEL_SCORE = 0.78
MINIMUM_LABEL_MARGIN = 0.10

# A candidate label is at most this many words. "Biometric confidence" is two;
# nothing in the vocabulary is longer than two.
MAX_LABEL_WORDS = 2

# Value text is cut at the next matched label, or this many characters.
MAX_VALUE_LENGTH = 80


def candidate_label_spans(text: str) -> list[tuple[int, int, str]]:
    """Yield (start, end, matched_text) for one- and two-word label candidates."""
    spans: list[tuple[int, int, str]] = []
    words = list(re.finditer(r"[A-Za-z][A-Za-z0-9]*", text))

    for index, word in enumerate(words):
        for length in range(1, MAX_LABEL_WORDS + 1):
            if index + length > len(words):
                break

            end_word = words[index + length - 1]
            spans.append((word.start(), end_word.end(),
                          text[word.start():end_word.end()]))

    return spans


def match_label(fragment: str, labels: dict[str, str]) -> str | None:
    """Return the field a text fragment names, if it unambiguously names one."""
    if not normalize_comparison_text(fragment):
        return None

    matched = best_vocabulary_match(
        fragment,
        labels.keys(),
        minimum_score=MINIMUM_LABEL_SCORE,
        minimum_margin=MINIMUM_LABEL_MARGIN,
    )

    return labels.get(matched) if matched else None


def best_vocabulary_value(window: str, vocabulary: set[str]) -> str | None:
    """
    Return the vocabulary term the text after a label most likely names.

    Word n-grams are scored rather than the whole window, so trailing OCR noise
    does not drag an otherwise clean value below the acceptance threshold.
    """
    words = re.findall(r"[A-Za-z0-9][A-Za-z0-9_/-]*", window)

    if not words:
        return None

    best: tuple[float, str] | None = None

    for start in range(len(words)):
        for length in range(1, MAX_VOCABULARY_WORDS + 1):
            if start + length > len(words):
                break

            fragment = " ".join(words[start:start + length])
            match = best_vocabulary_match(fragment, vocabulary)

            if match is None:
                continue

            score = _similarity(fragment, match)

            if best is None or score > best[0]:
                best = (score, match)

    return best[1] if best else None


def _similarity(value: str, candidate: str) -> float:
    from difflib import SequenceMatcher

    return SequenceMatcher(
        None,
        normalize_comparison_text(value),
        normalize_comparison_text(candidate),
    ).ratio()


def extract_with_fuzzy_labels(
    text: str,
    document_type: str,
    wanted: set[str] | None = None,
) -> dict[str, str]:
    """
    Extract fields by fuzzy-matching labels rather than requiring exact ones.

    `wanted` restricts the search to fields the strict extractor left empty.
    """
    labels = DOCUMENT_LABELS.get(document_type, ALL_LABELS)

    if wanted is not None:
        labels = {
            label: field
            for label, field in labels.items()
            if field in wanted
        }

    if not labels or not text:
        return {}

    # Locate every fragment that names a label, keeping the longest match at
    # each start position so "Home World" wins over "Home".
    hits: list[tuple[int, int, str]] = []

    for start, end, fragment in candidate_label_spans(text):
        field = match_label(fragment, labels)

        if field is None:
            continue

        if hits and start == hits[-1][0]:
            if end > hits[-1][1]:
                hits[-1] = (start, end, field)
            continue

        if hits and start < hits[-1][1]:
            continue

        hits.append((start, end, field))

    found: dict[str, str] = {}

    for position, (start, end, field) in enumerate(hits):
        if field in found:
            continue

        stop = (
            hits[position + 1][0]
            if position + 1 < len(hits)
            else len(text)
        )
        window = text[end:min(stop, end + MAX_VALUE_LENGTH)]

        vocabulary = FIELD_VOCABULARIES.get(field)
        pattern = FIELD_PATTERNS.get(field)

        if pattern is not None:
            found_pattern = re.search(pattern, window, flags=re.IGNORECASE)
            value = found_pattern.group(0).strip() if found_pattern else None
        elif vocabulary is not None:
            value = best_vocabulary_value(window, vocabulary)
        else:
            # Free-form field: drop the separator after the label, then cut at
            # the first run of OCR noise.
            value = re.sub(r"^\s*[:;.,=-]+\s*", "", window).strip()
            value = re.split(r"\s{3,}|[|\\(\[]|[—_]{2,}", value)[0].strip()

        if value:
            found[field] = value

    return found
