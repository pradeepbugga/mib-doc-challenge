"""
Derive `fee_status` from the sibling fields on a fee receipt.

`fee_status` is not always readable from its own label. Of 429 training cases
where the label was lost, 47 have a fee receipt page whose `Fee Status` line
specifically did not survive OCR while the rest of the receipt did, and another
61 have a status word floating in the text with no label attached to it.

The receipt carries the same fact twice. `Amount` and `Waiver Code` are short,
distinctive tokens — `$809.00`, `DIP-WAIVER` — that survive damage far better
than the phrase "Fee Status", and on the training set they determine the status
exactly:

    amount > 0,  waiver N/A          -> paid    (118/118)
    amount == 0, waiver DIP-WAIVER   -> waived  (37/37)
    amount == 0, waiver N/A          -> ambiguous: unpaid 12 / unknown 10

The third combination is left alone. Splitting evenly between `unpaid` and
`unknown` means the receipt genuinely does not say, and guessing `unpaid` there
would flip the adjudication to DENIED on a coin toss.

The derived value wins even when a `Fee Status` label was already read
successfully, not only when the label is missing. Some receipts print a
`Fee Status` word that plainly contradicts their own `Amount`/`Waiver Code` --
confirmed on 26 training cases where the label disagreed with truth: 10 of
those also had a readable amount/waiver pair, and the derivation was correct
on all 10 while the label was wrong on all 10. The label is not just harder to
read than the sibling fields, it is sometimes an outright decoy.

This mirrors `core.adjudication.risk_derivation`: a post-corroboration step that
fills a field from sibling evidence rather than from its own label.
"""

from __future__ import annotations

import re

from core.adjudication.models import Packet, ResolvedField

# A waiver code naming a waiver programme, as opposed to "N/A"/absent.
WAIVER_PRESENT = re.compile(r"WAIVER|WVR", re.IGNORECASE)

# Codes that explicitly record the absence of a waiver.
WAIVER_ABSENT = {"N/A", "NA", "NONE", "-", ""}


def parse_amount(value) -> float | None:
    """Return a receipt amount as a number, or None when unreadable."""
    if value is None:
        return None

    match = re.search(r"[\d,]+(?:\.\d{1,2})?", str(value))

    if match is None:
        return None

    try:
        return float(match.group(0).replace(",", ""))
    except ValueError:
        return None


def classify_waiver(value) -> str:
    """Return 'present', 'absent', or 'unknown' for a waiver code."""
    if value is None:
        return "unknown"

    text = str(value).strip().upper()

    if not text:
        return "unknown"

    if WAIVER_PRESENT.search(text):
        return "present"

    if text in WAIVER_ABSENT:
        return "absent"

    return "unknown"


def derive_fee_status(packet: Packet) -> str | None:
    """
    Return the fee status the receipt's amount and waiver code imply.

    Returns None when the pair does not determine a status, so the caller
    leaves the field unresolved rather than guessing.
    """
    amount_field = packet.fields.get("amount")
    waiver_field = packet.fields.get("waiver_code")

    amount = parse_amount(
        amount_field.resolved_value if amount_field is not None else None
    )
    waiver = classify_waiver(
        waiver_field.resolved_value if waiver_field is not None else None
    )

    if amount is None:
        # A waiver code alone still settles a waived fee.
        return "waived" if waiver == "present" else None

    if amount > 0:
        # Money changed hands. A waiver alongside a non-zero amount is
        # contradictory, so only the clean case is claimed.
        return "paid" if waiver in {"absent", "unknown"} else None

    if waiver == "present":
        return "waived"

    # amount == 0 with no waiver: the receipt does not say whether the fee is
    # unpaid or merely unrecorded.
    return None


def augment_fee_status(packet: Packet) -> None:
    """Fill or correct `fee_status` from the sibling amount/waiver fields.

    Overrides an already-read `Fee Status` label whenever the sibling fields
    determine a value -- see the module docstring for why the label is not
    trusted over them.
    """
    existing = packet.fields.get("fee_status")

    derived = derive_fee_status(packet)

    if derived is None:
        return

    if existing is not None:
        existing.resolved_value = derived
        existing.status = "derived"
        existing.resolution_method = "fee_receipt_derivation"
    else:
        packet.fields["fee_status"] = ResolvedField(
            field="fee_status",
            resolved_value=derived,
            status="derived",
            observations=[],
            supporting_observations=[],
            resolution_method="fee_receipt_derivation",
        )
