from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class ClassificationResult:
    document_type: str
    score: int
    confidence: float
    matched_cues: tuple[str, ...]


DOCUMENT_TYPE_RULES = {
    "intake_form": {
        "headings": {
            "form i 8090": 8,
            "extraterrestrial work authorization intake": 8,
            "mib intake form": 8,
            "temporary work authorization application": 8,
            "applicant intake": 6,
        },
        "fields": {
            "case id": 1,
            "applicant": 1,
            "species code": 2,
            "home world": 2,
            "visa class": 1,
            "sponsor id": 1,
            "arrival date": 2,
            "declared purpose": 3,
        },
        "minimum_score": 5,
    },

    "biometric_slip": {
        "headings": {
            "form b 13": 8,
            "biometric scan slip": 8,
            "biometric verification": 8,
            "biometric slip": 7,
        },
        "fields": {
            "biometric confidence": 4,
            "species match": 3,
            "observed flags": 3,
            "biometric status": 3,
            "identity match": 3,
            "biohazard check": 3,
            "scan image": 1,
        },
        "minimum_score": 5,
    },

    "sponsor_attestation": {
        "headings": {
            "sponsor attestation letter": 10,
            "sponsor attestation": 9,
            "sponsor declaration": 8,
        },
        "fields": {
            "sponsor id": 4,
            "applicant": 1,
            "purpose": 2,
            "visa class": 2,
            "sponsor signature": 3,
            "applicant sponsored": 3,
            "cultural exchange": 2,
        },
        "minimum_score": 6,
    },

    "registry_extract": {
        "headings": {
            "planetary registry extract": 10,
            "registry extract": 8,
            "mib registry record": 8,
        },
        "fields": {
            "registry name": 3,
            "registry status": 4,
            "home world": 1,
            "species code": 1,
            "arrival date": 1,
            "warrant status": 3,
            "embargo status": 3,
        },
        "minimum_score": 5,
    },

    "fee_receipt": {
        "headings": {
            "mib fee receipt": 10,
            "payment receipt": 8,
            "fee receipt": 8,
        },
        "fields": {
            "fee status": 4,
            "amount": 2,
            "amount paid": 3,
            "waiver code": 3,
            "payment reference": 3,
        },
        "minimum_score": 5,
    },

    "adjudicator_note": {
        "headings": {
            "adjudicator note": 9,
            "manual review note": 8,
            "decision note": 7,
        },
        "fields": {
            "adjudicator": 4,
            "decision": 3,
            "approved": 2,
            "denied": 2,
            "signed": 1,
            "review findings": 3,
        },
        "minimum_score": 5,
    },
}

OCR_REPLACEMENTS = {
    "applieant": "applicant",
    "applicantt": "applicant",
    "sponsor 1d": "sponsor id",
    "sponsor ld": "sponsor id",
    "visa gass": "visa class",
    "visa dass": "visa class",
    "registry narne": "registry name",
}


def normalize_text(text: str) -> str:
    normalized = text.lower()

    for incorrect, corrected in OCR_REPLACEMENTS.items():
        normalized = normalized.replace(incorrect, corrected)

    # Convert punctuation and separators to spaces.
    normalized = re.sub(r"[_|:;/\\\-]+", " ", normalized)

    # Remove remaining unusual characters.
    normalized = re.sub(r"[^a-z0-9\s]", " ", normalized)

    # Collapse whitespace.
    return " ".join(normalized.split())

def compute_confidence(
    score: int,
    margin: int,
    matched_cues: tuple[str, ...],
) -> float:
    heading_matched = any(
        cue.startswith("heading:")
        for cue in matched_cues
    )

    confidence = (
        0.35
        + 0.035 * score
        + 0.04 * margin
    )

    if not heading_matched:
        confidence = min(confidence, 0.90)

    return round(min(confidence, 1.0), 3)

def classify_document_type(text: str) -> ClassificationResult:
    normalized_text = normalize_text(text)

    results: list[ClassificationResult] = []

    for document_type, rules in DOCUMENT_TYPE_RULES.items():
        score = 0
        matched_cues: list[str] = []

        heading_matches = [
            (heading, weight)
            for heading, weight in rules["headings"].items()
            if normalize_text(heading) in normalized_text
        ]

        if heading_matches:
            best_heading, best_heading_weight = max(
                heading_matches,
                key=lambda item: item[1],
            )

            score += best_heading_weight
            matched_cues.append(f"heading:{best_heading}")

        for field, weight in rules["fields"].items():
            normalized_field = normalize_text(field)

            if normalized_field in normalized_text:
                score += weight
                matched_cues.append(f"field:{field}")

        results.append(
            ClassificationResult(
                document_type=document_type,
                score=score,
                confidence=0.0,
                matched_cues=tuple(matched_cues),
            )
        )

    results.sort(
        key=lambda result: result.score,
        reverse=True,
    )

    best = results[0]
    second_best_score = results[1].score if len(results) > 1 else 0

    minimum_score = DOCUMENT_TYPE_RULES[
        best.document_type
    ]["minimum_score"]

    if best.score < minimum_score:
        return ClassificationResult(
            document_type="unknown",
            score=best.score,
            confidence=0.0,
            matched_cues=best.matched_cues,
        )

    # A heuristic confidence based on absolute score and separation
    # from the next-best document type.
    margin = best.score - second_best_score
    confidence = compute_confidence(
        score=best.score,
        margin=margin,
        matched_cues=best.matched_cues,
    )

    return ClassificationResult(
        document_type=best.document_type,
        score=best.score,
        confidence=round(confidence, 3),
        matched_cues=best.matched_cues,
    )


def main() -> None:
    import fitz

    doc = fitz.open("./data/train/MIB-000009.pdf")

    for page in doc:
        text = page.get_text("text")
        document_type = classify_document_type(text)
        print(f"Page {page.number + 1}: Classified as '{document_type.document_type}'")

if __name__ == "__main__":
    main()