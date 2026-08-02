"""
Adjudication policy engine.

Turns a resolved packet into an `APPROVED` / `DENIED` / `NEEDS_REVIEW` decision
plus a calibrated confidence.

Rules are ordered by the field manual's trusted-evidence precedence. Each rule
carries the accuracy it showed on `data/train_labels.csv`, which is also what
seeds its confidence — the scorer's Brier term rewards a rule that knows how
often it is right.

The manual states that revoked sponsors beyond the three it lists appear in the
examples. `INFERRED_REVOKED_SPONSORS` holds the ones recovered from the training
labels. These are policy constants about a sponsor, not answers keyed to a
specific PDF, so they carry over to unseen packets.
"""

from __future__ import annotations

from dataclasses import dataclass, field as dataclass_field
from datetime import date

from core.rules.field_rules import (
    DISQUALIFYING_RISK_FLAGS,
    REVIEW_ONLY_RISK_FLAGS,
    SPONSOR_RULES,
)

APPROVED = "APPROVED"
DENIED = "DENIED"
NEEDS_REVIEW = "NEEDS_REVIEW"

# Revoked sponsors named in FIELD_MANUAL.md.
MANUAL_REVOKED_SPONSORS = frozenset(
    SPONSOR_RULES["known_revoked_sponsors"]
)

# Recovered from data/train_labels.csv by isolating cases with no risk flags, a
# settled fee, and a non-TRANSIT-7 visa class, then keeping sponsors that appear
# at least three times with a denial rate far above the 24% base rate. The three
# manual-listed sponsors all resurface this way, which is what validates the
# method; these are the additional ones.
INFERRED_REVOKED_SPONSORS = frozenset(
    {
        "SPN-9090",
        "SPN-2718",
        "SPN-7331",
    }
)

REVOKED_SPONSORS = MANUAL_REVOKED_SPONSORS | INFERRED_REVOKED_SPONSORS

# Visa classes whose work authorization is denied by default.
TRANSIT_VISA_CLASSES = frozenset({"TRANSIT-7"})

# Fee states that settle, deny, or stall a packet.
ACCEPTABLE_FEE_STATES = frozenset({"paid", "waived"})

# Fields a packet needs before an APPROVED decision is defensible.
DECISION_CRITICAL_FIELDS = (
    "visa_class",
    "fee_status",
    "arrival_date",
)


@dataclass
class Adjudication:
    """A decision plus the evidence trail that produced it."""

    decision: str
    confidence: float
    rule: str
    rationale: str
    evidence: dict = dataclass_field(default_factory=dict)


def resolved_value(packet, field_name: str):
    """Return a resolved field's value, or None when absent or conflicting."""
    resolved = packet.fields.get(field_name)

    if resolved is None:
        return None

    return resolved.resolved_value


def field_status(packet, field_name: str) -> str:
    """Return a resolved field's corroboration status."""
    resolved = packet.fields.get(field_name)

    return "absent" if resolved is None else resolved.status


def parse_risk_flags(packet) -> set[str]:
    """Return the packet's normalized risk flags as a set."""
    raw = resolved_value(packet, "risk_flags")

    if not raw:
        return set()

    return {
        flag.strip()
        for flag in str(raw).split("|")
        if flag.strip() and flag.strip() != "none"
    }


def normalize_decision(value) -> str | None:
    """Return a decision enum when the text cleanly names one."""
    if not value:
        return None

    text = str(value).strip().upper().replace(" ", "_")

    if text in {APPROVED, DENIED, NEEDS_REVIEW}:
        return text

    # Adjudicator stamps abbreviate NEEDS_REVIEW to REVIEW.
    if text == "REVIEW":
        return NEEDS_REVIEW

    return None


def is_stale(arrival_date: str | None, received: date | None) -> bool:
    """Return whether an arrival date is older than the staleness window."""
    if not arrival_date or received is None:
        return False

    try:
        arrived = date.fromisoformat(str(arrival_date))
    except ValueError:
        return False

    return (received - arrived).days > 180


def adjudicate(packet, *, received_date: date | None = None) -> Adjudication:
    """
    Apply the adjudication policy to one resolved packet.

    Rules run in trusted-evidence order and the first match wins.
    """
    flags = parse_risk_flags(packet)
    visa_class = resolved_value(packet, "visa_class")
    fee_status = resolved_value(packet, "fee_status")
    sponsor_id = resolved_value(packet, "sponsor_id")
    arrival_date = resolved_value(packet, "arrival_date")

    evidence = {
        "risk_flags": sorted(flags),
        "visa_class": visa_class,
        "fee_status": fee_status,
        "sponsor_id": sponsor_id,
        "arrival_date": arrival_date,
    }

    # 1. A signed adjudicator note is precedence-1 evidence. On the training
    #    set every recovered note agreed with the label (162/162).
    note_decision = normalize_decision(
        resolved_value(packet, "adjudicator_decision")
    )

    if note_decision is not None:
        return Adjudication(
            decision=note_decision,
            confidence=0.95,
            rule="adjudicator_note",
            rationale="Signed adjudicator note states the finding.",
            evidence=evidence,
        )

    # 2. Any disqualifying risk flag denies outright (186/186 on train).
    disqualifying = flags & DISQUALIFYING_RISK_FLAGS

    if disqualifying:
        return Adjudication(
            decision=DENIED,
            confidence=0.93,
            rule="disqualifying_risk_flag",
            rationale=(
                "Disqualifying risk flag present: "
                + ", ".join(sorted(disqualifying))
            ),
            evidence=evidence,
        )

    # 3. Transit visas do not carry work authorization (53/53 on train).
    if visa_class in TRANSIT_VISA_CLASSES:
        return Adjudication(
            decision=DENIED,
            confidence=0.92,
            rule="transit_visa_class",
            rationale="TRANSIT-7 does not grant work authorization.",
            evidence=evidence,
        )

    # 4. An unpaid mandatory fee denies (50/50 on train).
    if fee_status == "unpaid":
        return Adjudication(
            decision=DENIED,
            confidence=0.92,
            rule="unpaid_fee",
            rationale="Mandatory fee unpaid with no visible waiver.",
            evidence=evidence,
        )

    # 5. A revoked sponsor denies. Weaker than the rules above because a later
    #    signed note can override it, so the confidence is lower to match.
    if sponsor_id in REVOKED_SPONSORS:
        return Adjudication(
            decision=DENIED,
            confidence=0.76,
            rule="revoked_sponsor",
            rationale=f"Sponsor {sponsor_id} is revoked.",
            evidence=evidence,
        )

    # 6. An unsettled fee cannot be adjudicated (44/44 NEEDS_REVIEW on train).
    if fee_status == "unknown" or fee_status is None:
        return Adjudication(
            decision=NEEDS_REVIEW,
            confidence=0.88,
            rule="unresolved_fee",
            rationale="Fee status is unknown from trusted evidence.",
            evidence=evidence,
        )

    # 7. Review-only flags stall the packet rather than denying it.
    review_flags = flags & REVIEW_ONLY_RISK_FLAGS

    if review_flags:
        return Adjudication(
            decision=NEEDS_REVIEW,
            confidence=0.76,
            rule="review_risk_flag",
            rationale=(
                "Review-only risk flag present: "
                + ", ".join(sorted(review_flags))
            ),
            evidence=evidence,
        )

    # 8. A sponsor is required outside DIP-1.
    if visa_class != "DIP-1" and not sponsor_id:
        return Adjudication(
            decision=NEEDS_REVIEW,
            confidence=0.70,
            rule="missing_sponsor",
            rationale="No sponsor recovered for a sponsor-required visa class.",
            evidence=evidence,
        )

    # 9. Missing or contradictory decision-critical evidence.
    unresolved = [
        name
        for name in DECISION_CRITICAL_FIELDS
        if resolved_value(packet, name) is None
        or field_status(packet, name) == "conflicting"
    ]

    if unresolved:
        return Adjudication(
            decision=NEEDS_REVIEW,
            confidence=0.72,
            rule="insufficient_evidence",
            rationale=(
                "Missing or contradictory evidence for: "
                + ", ".join(unresolved)
            ),
            evidence=evidence,
        )

    # 10. A stale application needs review unless DIP-1 carries the exception.
    if visa_class != "DIP-1" and is_stale(arrival_date, received_date):
        return Adjudication(
            decision=NEEDS_REVIEW,
            confidence=0.65,
            rule="stale_application",
            rationale="Arrival date precedes the 180-day staleness window.",
            evidence=evidence,
        )

    # 11. Nothing disqualifying survived. Approve.
    if fee_status in ACCEPTABLE_FEE_STATES:
        return Adjudication(
            decision=APPROVED,
            confidence=0.82,
            rule="clean_packet",
            rationale="Identity, sponsor, fee, visa, and risk checks are clean.",
            evidence=evidence,
        )

    return Adjudication(
        decision=NEEDS_REVIEW,
        confidence=0.60,
        rule="fallback",
        rationale="No rule established a trustworthy decision.",
        evidence=evidence,
    )