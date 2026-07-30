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