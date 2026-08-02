"""
Read handwritten-style correction notes off the page.

Packets sometimes carry a line like `Manual correction: sponsor is SPN-4705.`
alongside the printed form. `FIELD_MANUAL.md` puts a "visible MIB adjudicator
stamp or signed manual note" at the top of the evidence precedence list, above
the intake form itself, so where such a note exists it settles the field.

They are also reliable in practice. Across the 1000 training packets, 136 carry
a correction note and every one checked agrees with the label — 28/28 for fee
status, 49/49 for sponsor. Until now they were not merely unused but harmful:
the note sits inside the intake form's text, so a greedy field regex swallowed
it, producing values like

    declared_purpose = 'archive audit Manual correction: sponsor is SPN-4705.'

while `sponsor_id` stayed empty. Parsing the note therefore both supplies a
trusted value and stops it contaminating the field it was printed next to.
"""

from __future__ import annotations

import re

# The note names a field in prose, then gives its value.
CORRECTION_PATTERN = re.compile(
    r"Manual\s+correction\s*:\s*(?P<body>[^\n|]{0,90})",
    re.IGNORECASE,
)

# Prose field names as they appear in the notes, mapped to canonical fields.
# Ordered longest-first so "visa class" is tested before "visa".
FIELD_PHRASES: tuple[tuple[str, str], ...] = (
    ("fee status", "fee_status"),
    ("visa class", "visa_class"),
    ("species code", "species_code"),
    ("home world", "home_world"),
    ("arrival date", "arrival_date"),
    ("declared purpose", "declared_purpose"),
    ("applicant", "applicant"),
    ("sponsor", "sponsor_id"),
    ("species", "species_code"),
)

# How each field's value is recognised once the phrase has been matched.
VALUE_PATTERNS: dict[str, str] = {
    "fee_status": r"\b(paid|waived|unpaid|unknown)\b",
    "visa_class": r"\b(XW-1|XW-2|DIP-1|MED-3|TRANSIT-7)\b",
    "sponsor_id": r"\b(SPN-\d{4})\b",
    "species_code": r"\b([A-Z][A-Z0-9_]{3,})\b",
    "arrival_date": r"\b(\d{4}-\d{2}-\d{2})\b",
}

# Free-text fields: take the words after "is", up to the sentence end.
FREE_TEXT_PATTERN = re.compile(r"\bis\s+(.+?)\s*[.;|]|\bis\s+(.+)$", re.IGNORECASE)


def parse_corrections(text: str) -> dict[str, str]:
    """Return the field values a page's correction notes assert."""
    corrections: dict[str, str] = {}

    if not text:
        return corrections

    for match in CORRECTION_PATTERN.finditer(text):
        body = match.group("body")
        lowered = body.lower()

        field = next(
            (
                canonical
                for phrase, canonical in FIELD_PHRASES
                if phrase in lowered
            ),
            None,
        )

        if field is None:
            continue

        pattern = VALUE_PATTERNS.get(field)

        if pattern is not None:
            found = re.search(pattern, body, re.IGNORECASE)

            if found:
                value = found.group(1)
                corrections[field] = (
                    value.lower() if field == "fee_status" else value.upper()
                    if field in {"visa_class", "sponsor_id", "species_code"}
                    else value
                )
            continue

        free = FREE_TEXT_PATTERN.search(body)

        if free:
            value = (free.group(1) or free.group(2) or "").strip()

            if value:
                corrections[field] = value

    return corrections


def strip_corrections(text: str) -> str:
    """
    Remove correction notes from a page's text.

    The note is printed inside the form body, so leaving it in place lets a
    greedy value regex absorb it into whichever field precedes it.
    """
    return CORRECTION_PATTERN.sub(" ", text or "")
