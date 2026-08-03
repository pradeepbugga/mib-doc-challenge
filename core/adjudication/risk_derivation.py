"""
Derive risk flags that aren't literal text on any single field.

`risk_flags` in the training labels is not always a value some extractor reads
off a "Observed flags:" line. Checked against `data/train_labels.csv`:
`MIB-000054` is labeled `planetary_embargo` from a 3-page packet with no
biometric_slip/sponsor_attestation/adjudicator_note page at all — the only
visible evidence is `Registry Status: EMBARGO REVIEW` on the registry_extract
page, in a field the pipeline already extracts into `registry_status` but never
maps to `risk_flags`. Sampling registry_status across the training set found
exactly two values (`CLEAR`, `EMBARGO REVIEW`); `EMBARGO REVIEW` co-occurs with
a risk flag in 11/13 sampled cases vs. 47/119 for `CLEAR`.

This module adds that mapping (and is the place to add others like it) as a
post-corroboration step, so the literal biometric_slip flags and derived flags
merge into one `risk_flags` field before adjudication runs.
"""

from __future__ import annotations

import re
from difflib import SequenceMatcher

from core.adjudication.models import Packet
from core.adjudication.ontology import VALID_RISK_FLAGS

# Every genuine applicant_name in the training labels is exactly two
# alphabetic words, 10-18 characters total. A badly OCR'd page can still
# populate the "applicant" field with boilerplate leakage (page footers,
# "COPY ARTIFACT", scan-label text) that isn't a name at all -- confirmed on
# MIB-000215 and MIB-000098, where a garbled registry_extract OCR read footer
# text as the applicant, and that junk value being treated as evidence
# produced false identity_conflict/sponsor_mismatch. Reject anything that
# doesn't fit the real shape before it counts as corroboration either way.
_PLAUSIBLE_NAME = re.compile(r"^[A-Za-z\-]+ [A-Za-z\-]+$")


def _looks_like_name(value: str) -> bool:
    return bool(_PLAUSIBLE_NAME.match(value)) and 10 <= len(value) <= 18


def registry_status_flag(packet: Packet) -> str | None:
    """Return the risk flag implied by a non-clear registry status, if any."""
    resolved = packet.fields.get("registry_status")

    if resolved is None or resolved.resolved_value is None:
        return None

    status = str(resolved.resolved_value).strip().upper()

    if "EMBARGO" in status:
        return "planetary_embargo"

    return None


_SUSPECT_DOCUMENT_TYPES = ("intake_form", "sponsor_attestation")

# Below the training-set decoy ceiling (0.643, two clean-but-different native
# names) and comfortably below a confirmed OCR-noise same-identity pair
# (0.897: registry_extract OCR "Orikesh Antari" vs intake_form native
# "Orikesh Aritari", MIB-000175). See mib-intake-name-decoy memory.
FUZZY_NAME_MATCH_THRESHOLD = 0.70


def _applicant_name_by_doctype(
    packet: Packet,
    *,
    native_only: bool,
) -> dict[str, set[str]]:
    """Applicant_name values observed, grouped by document type."""
    applicant_field = packet.fields.get("applicant_name")

    if applicant_field is None:
        return {}

    by_doctype: dict[str, set[str]] = {}

    for observation in applicant_field.observations:
        if native_only and observation.text_source != "native_text":
            continue

        if not observation.normalized_value:
            continue

        value = observation.normalized_value.strip()

        if not _looks_like_name(value):
            continue
        by_doctype.setdefault(observation.document_type, set()).add(value)

    return by_doctype


def _fuzzy_match(value: str, candidates: set[str]) -> bool:
    return any(
        SequenceMatcher(None, value.lower(), candidate.lower()).ratio()
        >= FUZZY_NAME_MATCH_THRESHOLD
        for candidate in candidates
    )


def _outlier_document_flag(
    packet: Packet,
    document_type: str,
    flag: str,
) -> str | None:
    """
    Return `flag` if `document_type`'s native applicant_name disagrees with a
    third-party-confirmed value.

    Measured 2026-08-03: in 23 training packets with 2+ distinct native-text
    applicant_name readings, the document type holding the outlier value
    cleanly separates two flags -- intake_form disagreeing means
    identity_conflict, sponsor_attestation disagreeing means sponsor_mismatch.
    This is the same mechanism as the intake-form name decoy: the applicant's
    own name is sometimes wrong on exactly one page while every other field
    on that page, and every other page's name, stays truthful.

    The SUSPECT document's own reading must be native_text -- we only want to
    flag a clean, unambiguous disagreement, not an OCR misread. The
    third-party corroborator may be native OR OCR, matched fuzzily: verified
    on MIB-000175, whose only non-suspect page (registry_extract) needed OCR
    and read "Orikesh Antari" against intake_form's clean "Orikesh Aritari" --
    an OCR-noise pair, not a genuine second identity (0.897 similarity, well
    above the 0.643 ceiling measured across all known decoys).

    Requires a genuine third-party corroborator -- a document type that is
    neither intake_form nor sponsor_attestation. Without one, a bare
    disagreement between just those two is symmetric: checking intake_form
    against sponsor_attestation and checking sponsor_attestation against
    intake_form each look identical, so there is no way to tell which one is
    the actual outlier. Better to miss that case than guess and risk firing
    both flags on one disagreement (confirmed as a real bug on MIB-000175
    before this function required a third party at all).
    """
    native_by_doctype = _applicant_name_by_doctype(packet, native_only=True)
    target_values = native_by_doctype.get(document_type)

    if not target_values:
        return None

    all_by_doctype = _applicant_name_by_doctype(packet, native_only=False)
    third_party_values = {
        value
        for doctype, values in all_by_doctype.items()
        if doctype not in _SUSPECT_DOCUMENT_TYPES
        for value in values
    }

    if not third_party_values:
        return None

    if target_values & third_party_values:
        return None

    if any(_fuzzy_match(value, third_party_values) for value in target_values):
        return None

    return flag


def identity_conflict_flag(packet: Packet) -> str | None:
    """Return identity_conflict when intake_form's applicant_name is an outlier."""
    return _outlier_document_flag(packet, "intake_form", "identity_conflict")


def sponsor_mismatch_flag(packet: Packet) -> str | None:
    """Return sponsor_mismatch when sponsor_attestation's applicant_name is an outlier."""
    return _outlier_document_flag(
        packet, "sponsor_attestation", "sponsor_mismatch"
    )


def adjudicator_reason_flag(packet: Packet) -> str | None:
    """
    Return a risk flag literally named in the adjudicator note's reason text.

    Found on MIB-000657: the note's reason field reads "Review-only risk flag
    present: identity_conflict." -- restating the flag directly. `reason` is
    already extracted (aliased to `adjudicator_reason`) but was never mined
    for this. The field manual treats a signed adjudicator note as
    precedence-1 evidence, so this is at least as trustworthy as a
    biometric_slip's "Observed flags:" line -- it just needed reading.

    Conservative: only fires when exactly one valid flag name appears in the
    text, so an edge case combining multiple flags is left undetermined
    rather than guessed at.
    """
    reason_field = packet.fields.get("adjudicator_reason")

    if reason_field is None or not reason_field.resolved_value:
        return None

    text = str(reason_field.resolved_value).lower()
    mentioned = {
        flag
        for flag in VALID_RISK_FLAGS
        if flag != "none" and flag in text
    }

    if len(mentioned) == 1:
        return mentioned.pop()

    return None


DERIVATION_RULES = (
    registry_status_flag,
    identity_conflict_flag,
    sponsor_mismatch_flag,
    adjudicator_reason_flag,
)


def merge_risk_flags(existing: str | None, derived: set[str]) -> str:
    """Combine an existing risk_flags string with newly derived flags."""
    flags = {
        flag.strip()
        for flag in (existing or "").split("|")
        if flag.strip() and flag.strip() != "none"
    }
    flags |= derived

    return "|".join(sorted(flags)) if flags else "none"


def augment_risk_flags(packet: Packet) -> None:
    """
    Fold derived risk flags into the packet's `risk_flags` field in place.

    Mutates the resolved field's value only; the underlying observations are
    left untouched so the corroboration trail still reflects what was actually
    read off the page.
    """
    derived = {
        flag
        for rule in DERIVATION_RULES
        if (flag := rule(packet)) is not None
    }

    if not derived:
        return

    existing_field = packet.fields.get("risk_flags")
    existing_value = (
        existing_field.resolved_value if existing_field is not None else None
    )
    merged = merge_risk_flags(existing_value, derived)

    if existing_field is not None:
        existing_field.resolved_value = merged
        existing_field.status = "corroborated"
    else:
        from core.adjudication.models import ResolvedField

        packet.fields["risk_flags"] = ResolvedField(
            field="risk_flags",
            resolved_value=merged,
            status="derived",
            observations=[],
            supporting_observations=[],
            resolution_method="registry_status_derivation",
        )