"""
Localized retry for a page whose OCR mostly succeeded but a specific
expected field's own line stayed garbled.

Motivating case: MIB-000890 p5 (biometric_slip). Three of its four field
lines sit on one background shade (~235); the fourth ("Observed flags",
the one field that actually decides the packet) sits on a distinctly darker
one (~204). A single page-wide contrast floor cannot serve both regions --
whichever floor is right for the 235 rows clips or misses the 204 row.
No whole-page fix (adaptive floor, CLAHE, percentile stretch, geometry
correction, alternate PSM -- all tried first) recovered that one line.
Cropping just that line and deriving its own local floor did.

The trigger is specifically "a required field is missing AND there is
leftover OCR text Tesseract could not attribute to any already-extracted
field" -- not "field is missing" alone. A page with no evidence at all for
a field (no biometric_slip page in the packet, for instance) produces no
orphan line and this never fires, so it can't burn OCR budget chasing pages
that were never going to recover.

Strictly additive: only ever fills a field that extraction left empty,
never overwrites one already read. A wrong guess into an empty field costs
nothing beyond what it already cost, but must still be validated at scale
before trusting -- the same rule that applied to every other retry
candidate in this pipeline.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import cv2
import numpy as np
import pytesseract
from PIL import Image
from pytesseract import Output

from core.extraction.extractor import extract_fields
from core.extraction.schema import DOCUMENT_REQUIRED_FIELDS
from core.ocr.engine import (
    enhance_contrast,
    estimate_ink_floor,
    suppress_faint_ink,
)

# Page furniture that legitimately has no field behind it -- never a retry
# candidate even when it doesn't match any extracted value.
BOILERPLATE_LINE = re.compile(
    r"packet|page\s+\d|synthetic\s+hiring|mib\s+eyes\s+only|scan\s+image|"
    r"passport\s+image|registry\s+image|form\s+[bi][-\s]?\d+",
    re.IGNORECASE,
)

# Padding added above/below a candidate line's own Tesseract bounding box
# before cropping -- covers a line whose height a garbled read under-measured,
# without pulling in the neighboring line.
LINE_CROP_PADDING = 10

MIN_LINE_TEXT_LENGTH = 3

# Every genuine applicant name in the training labels is exactly two
# alphabetic words, 10-18 characters total (see
# core.adjudication.risk_derivation._looks_like_name, which this mirrors).
# A name-shaped field holding something outside that shape -- a truncated
# read like "Orin" for "Orinax Miravara", confirmed on MIB-000890 p5 -- is
# functionally as unusable as an empty one, so it counts as missing for
# retry purposes even though the field isn't literally None.
NAME_FIELDS = frozenset({"applicant", "registry_name"})
_PLAUSIBLE_NAME = re.compile(r"^[A-Za-z\-]+ [A-Za-z\-]+$")


def _is_missing(field_name: str, value) -> bool:
    if not value:
        return True

    if field_name in NAME_FIELDS:
        text = str(value)
        return not (
            _PLAUSIBLE_NAME.match(text) and 10 <= len(text) <= 18
        )

    return False

# Tried in order per crop; the first to recover a field wins. 11 (sparse
# text) and 4 (single variable-size column) cover the two shapes measured
# so far -- a short isolated line, and a normal label:value line that PSM
# 11 mis-segments for reasons unrelated to contrast. 3 is Tesseract's own
# general-purpose default, kept as a final fallback.
LINE_RETRY_TESSERACT_CONFIGS = (
    "--oem 3 --psm 11",
    "--oem 3 --psm 4",
    "--oem 3 --psm 3",
)


@dataclass(frozen=True)
class OCRLine:
    """One Tesseract-detected line, with its page-pixel bounding box."""

    text: str
    left: int
    top: int
    width: int
    height: int
    confidence: float | None


def extract_ocr_lines(
    image: np.ndarray,
    config: str = LINE_RETRY_TESSERACT_CONFIGS[0],
) -> list[OCRLine]:
    """Group Tesseract's word-level output into per-line records."""
    if image.ndim == 3:
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    else:
        rgb_image = image

    data = pytesseract.image_to_data(
        Image.fromarray(rgb_image),
        config=config,
        output_type=Output.DICT,
    )

    grouped: dict[tuple[int, int, int], dict] = {}

    for i in range(len(data["text"])):
        text = data["text"][i].strip()

        if not text:
            continue

        key = (
            data["block_num"][i],
            data["par_num"][i],
            data["line_num"][i],
        )
        entry = grouped.setdefault(
            key,
            {
                "words": [],
                "lefts": [],
                "tops": [],
                "rights": [],
                "bottoms": [],
                "confs": [],
            },
        )
        entry["words"].append(text)
        entry["lefts"].append(data["left"][i])
        entry["tops"].append(data["top"][i])
        entry["rights"].append(data["left"][i] + data["width"][i])
        entry["bottoms"].append(data["top"][i] + data["height"][i])

        try:
            conf = float(data["conf"][i])
        except (TypeError, ValueError):
            conf = -1.0

        if conf >= 0:
            entry["confs"].append(conf)

    lines = [
        OCRLine(
            text=" ".join(entry["words"]),
            left=min(entry["lefts"]),
            top=min(entry["tops"]),
            width=max(entry["rights"]) - min(entry["lefts"]),
            height=max(entry["bottoms"]) - min(entry["tops"]),
            confidence=(
                sum(entry["confs"]) / len(entry["confs"])
                if entry["confs"]
                else None
            ),
        )
        for entry in grouped.values()
    ]

    lines.sort(key=lambda line: line.top)

    return lines


def _line_is_accounted_for(
    line: OCRLine,
    extracted_values: list[str],
) -> bool:
    text_lower = line.text.lower()

    return any(
        value and value.lower() in text_lower
        for value in extracted_values
    )


def find_orphan_lines(
    lines: list[OCRLine],
    extracted_values: list[str],
) -> list[OCRLine]:
    """
    Lines Tesseract detected that don't correspond to any already-extracted
    field value and aren't page furniture -- candidates for a garbled
    expected field.
    """
    return [
        line
        for line in lines
        if len(line.text) >= MIN_LINE_TEXT_LENGTH
        and not BOILERPLATE_LINE.search(line.text)
        and not _line_is_accounted_for(line, extracted_values)
    ]


def retry_missing_fields(
    *,
    image: np.ndarray,
    document_type: str,
    fields: dict,
    lines: list[OCRLine],
) -> dict:
    """
    For each required field still missing after the primary OCR pass, retry
    every orphan line in isolation with its own locally-derived contrast
    floor and see whether it now extracts.

    Returns {field_name: recovered_value} for fields this recovered --
    callers must only use it to fill gaps, never to overwrite an existing
    value.
    """
    required = DOCUMENT_REQUIRED_FIELDS.get(document_type)

    if not required:
        return {}

    missing = [
        name
        for name in required
        if _is_missing(name, fields.get(name))
    ]

    if not missing:
        return {}

    # Exclude values for fields we're treating as missing -- an implausible
    # name like "Orin" must not mark its own source line as already
    # accounted for, or that line would never be offered as a retry target.
    extracted_values = [
        str(value)
        for name, value in fields.items()
        if value and name not in missing
    ]
    orphans = find_orphan_lines(lines, extracted_values)

    if not orphans:
        return {}

    height, width = image.shape[:2]
    recovered: dict = {}

    for line in orphans:
        if len(recovered) == len(missing):
            break

        top = max(0, line.top - LINE_CROP_PADDING)
        bottom = min(height, line.top + line.height + LINE_CROP_PADDING)
        crop = image[top:bottom, 0:width]

        if crop.size == 0:
            continue

        floor = estimate_ink_floor(crop)
        suppressed = suppress_faint_ink(crop, floor=floor)
        enhanced = enhance_contrast(suppressed)

        # A single PSM does not serve every crop -- confirmed on MIB-000890
        # p5, where the "Observed flags" line only reads under --psm 11
        # (sparse text) and the "Applicant" line on the same page only
        # reads under --psm 3/4 (page/column-aware layout), each returning
        # nothing at all under the other's mode. Try each and keep whichever
        # crop actually recovers a missing field.
        for config in LINE_RETRY_TESSERACT_CONFIGS:
            crop_data = pytesseract.image_to_data(
                Image.fromarray(enhanced),
                config=config,
                output_type=Output.DICT,
            )
            crop_text = " ".join(
                t.strip() for t in crop_data["text"] if t.strip()
            )

            if not crop_text:
                continue

            candidate = extract_fields(
                document_type=document_type,
                text=crop_text,
            )
            candidate_fields = candidate.get("fields", {})

            for name in missing:
                if name in recovered:
                    continue

                value = candidate_fields.get(name)

                if value:
                    recovered[name] = value

            if len(recovered) == len(missing):
                break

    return recovered
