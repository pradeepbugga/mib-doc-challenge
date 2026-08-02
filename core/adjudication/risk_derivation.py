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

from core.adjudication.models import Packet


def registry_status_flag(packet: Packet) -> str | None:
    """Return the risk flag implied by a non-clear registry status, if any."""
    resolved = packet.fields.get("registry_status")

    if resolved is None or resolved.resolved_value is None:
        return None

    status = str(resolved.resolved_value).strip().upper()

    if "EMBARGO" in status:
        return "planetary_embargo"

    return None


DERIVATION_RULES = (registry_status_flag,)


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