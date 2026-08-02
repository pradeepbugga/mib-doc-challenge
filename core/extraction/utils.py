from __future__ import annotations

import re


def extract_regex(
    pattern: str,
    text: str,
    flags=re.IGNORECASE,
):
    """
    Return first regex capture group.

    Returns None if no match.
    """
    match = re.search(pattern, text, flags)

    if match:
        return match.group(1).strip()

    return None

def extract_first(
    patterns: list[str],
    text: str,
    flags=re.IGNORECASE,
) -> str | None:
    """Return the first successful extraction."""
    for pattern in patterns:
        value = extract_regex(
            pattern,
            text,
            flags=flags,
        )

        if value is not None:
            return value

    return None