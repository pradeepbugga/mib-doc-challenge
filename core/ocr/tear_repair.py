"""
Repair horizontal scanline tearing.

Some scanned packets are torn into horizontal bands that are each displaced
sideways by a different amount, so a field's label and its value can end up on
different x offsets. OCR reads the values but mangles the labels, and the
label-anchored extractors then fail on a page a human can read easily.

The page border rules are the landmark. They sit at a fixed x in an undamaged
page, so a band's border segment carries that band's displacement. Correcting
band by band (rather than row by row) matters: glyphs are ~25px tall and bands
are ~45px, so a per-row correction shears characters apart and makes OCR worse.

Repair is offered as a retry candidate, never applied unconditionally — see
`core.ocr.orientation.score_orientation_candidate`. On some pages the repair
helps a lot and on others it destroys readable text, so the caller keeps
whichever version actually scores better downstream.
"""

from __future__ import annotations

import cv2
import numpy as np

# Fraction of page width at each edge searched for the border rule.
SIDE_FRACTION = 0.16

# Minimum vertical run to count as a rule segment rather than a letter stem.
MIN_RULE_LENGTH = 12

# Largest believable band displacement, in pixels at the render DPI.
MAX_SHIFT = 200

# A new band starts when the offset jumps by more than this.
BAND_JUMP = 12

# Bands shorter than this are noise.
MIN_BAND_HEIGHT = 10

# Offset spread below this means the page is not meaningfully torn.
TEAR_SPREAD_THRESHOLD = 30.0


# How far a tracked border may drift between adjacent rows and still be the
# same object. Wider than this and we are looking at a different mark.
TRACK_TOLERANCE = 6

# Discontinuities smaller than this are noise, not a tear.
MIN_JUMP = 8

# Rows either side of a candidate boundary used to test whether body text moved
# by the same amount as the border.
CORROBORATION_WINDOW = 14

# Border and text displacement may differ by this much and still corroborate.
CORROBORATION_TOLERANCE = 8

# Skew below this is not worth correcting, in pixels of drift per pixel of height.
MIN_SKEW_SLOPE = 0.002

# Pairs sampled when fitting the border slope.
SKEW_SAMPLES = 4000

# Sampled pairs must be at least this far apart vertically to give a stable slope.
MIN_SKEW_BASELINE = 50


def _vertical_strokes(gray: np.ndarray) -> np.ndarray:
    """Keep only vertical ink runs, dropping text and horizontal rules."""
    binary = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )[1]

    return cv2.morphologyEx(
        binary,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (1, MIN_RULE_LENGTH)),
        iterations=1,
    )


def leftmost_marks(gray: np.ndarray) -> np.ndarray:
    """x of the leftmost vertical mark on each scanline, or NaN."""
    strokes = _vertical_strokes(gray)
    height, width = gray.shape
    left_limit = int(width * SIDE_FRACTION)

    marks = np.full(height, np.nan)

    for y in range(height):
        xs = np.nonzero(strokes[y, :left_limit])[0]

        if len(xs):
            marks[y] = xs[0]

    return marks


def estimate_border_skew(gray: np.ndarray) -> float:
    """
    Return the border rule's slope, dx per dy.

    The band offsets and the page skew both move the border sideways, and on a
    skewed page the skew component is not small: MIB-000003 page 1 drifts 116px
    top to bottom, comparable to the tear itself. Left uncorrected it is read
    as tear and the repair shifts bands that were never displaced.

    The slope is fitted with a median of pairwise slopes, which ignores the
    piecewise jumps the tear introduces. `estimate_skew_angle` in the OCR engine
    is not usable here — its whole-page `minAreaRect` is thrown off by torn
    content, reporting +0.88 degrees for a page that is skewed -3.01.
    """
    marks = leftmost_marks(gray)
    measured = ~np.isnan(marks)

    if measured.sum() < 20:
        return 0.0

    ys = np.nonzero(measured)[0].astype(float)
    xs = marks[measured]

    generator = np.random.default_rng(0)
    first = generator.integers(0, len(ys), SKEW_SAMPLES)
    second = generator.integers(0, len(ys), SKEW_SAMPLES)

    usable = np.abs(ys[first] - ys[second]) > MIN_SKEW_BASELINE

    if usable.sum() < 10:
        return 0.0

    slopes = (xs[first][usable] - xs[second][usable]) / (
        ys[first][usable] - ys[second][usable]
    )

    return float(np.median(slopes))


def deskew_by_shear(gray: np.ndarray, slope: float) -> np.ndarray:
    """
    Straighten the page with a horizontal shear so vertical rules stand upright.

    A shear is used rather than a rotation because only horizontal alignment
    matters here, and shearing leaves rows intact instead of resampling them.
    """
    if abs(slope) < MIN_SKEW_SLOPE:
        return gray

    height, width = gray.shape

    # warpAffine takes a forward src->dst map, so removing a slope of +s
    # requires shearing by -s.
    matrix = np.float32([[1.0, -slope, 0.0], [0.0, 1.0, 0.0]])

    return cv2.warpAffine(
        gray,
        matrix,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=255,
    )


def estimate_offsets(gray: np.ndarray) -> np.ndarray:
    """
    Return each scanline's displacement from the left border rule, or NaN.

    Only the left edge is used. Measured on a torn page the right edge
    contributes almost nothing on its own (7 of 2200 rows), while requiring the
    two edges to agree discards 128 otherwise-good rows and garbles field
    labels that a left-only correction keeps intact. The left rule is also the
    one that matters: labels and values are left-aligned, so aligning to it is
    what puts a label back beside its value.

    Within the left strip the border is taken to be the *leftmost* mark. The
    strip has to be wide enough to contain the border across its full travel
    (roughly +/-100px here), which means it also catches letter stems, and a
    band can offer half a dozen candidates. Picking the one nearest the
    reference guesses wrong exactly where the form fields are: rows 588-605
    resolved to a 0px shift while the identical border position one band below
    resolved to -50px. The border is by definition the leftmost ink on the
    page, so taking the first mark needs no guess and merges those bands
    correctly.
    """
    strokes = _vertical_strokes(gray)
    height, width = gray.shape

    left_limit = int(width * SIDE_FRACTION)

    first_mark = np.full(height, np.nan)

    for y in range(height):
        xs = np.nonzero(strokes[y, :left_limit])[0]

        if len(xs):
            first_mark[y] = xs[0]

    measured = first_mark[~np.isnan(first_mark)]

    if not len(measured):
        return first_mark

    offsets = first_mark - float(np.median(measured))
    offsets[np.abs(offsets) > MAX_SHIFT] = np.nan

    return offsets


def track_border(gray: np.ndarray) -> tuple[np.ndarray, list[tuple[int, float]]]:
    """
    Follow a single border object down the page.

    Taking the leftmost mark on each row independently silently switches
    between objects — the border, a table rule, a letter stem — and reports the
    switch as a tear. On MIB-000003 that invented a 24px boundary in the middle
    of an intact title line and shredded it. Here each row picks the candidate
    nearest the previous row's position, so the tracker stays on one object;
    anything outside `TRACK_TOLERANCE` is recorded as a candidate tear rather
    than silently accepted.
    """
    strokes = _vertical_strokes(gray)
    height, width = gray.shape
    left_limit = int(width * SIDE_FRACTION)

    marks = [np.nonzero(strokes[y, :left_limit])[0] for y in range(height)]
    seeds = np.array([m[0] for m in marks if len(m)], dtype=float)

    positions = np.full(height, np.nan)
    jumps: list[tuple[int, float]] = []

    if not len(seeds):
        return positions, jumps

    current = float(np.median(seeds))

    for y in range(height):
        row_marks = marks[y]

        if not len(row_marks):
            continue

        distances = np.abs(row_marks - current)
        nearest = int(np.argmin(distances))

        if distances[nearest] > TRACK_TOLERANCE:
            jump = float(row_marks[nearest]) - current

            if MIN_JUMP <= abs(jump) <= MAX_SHIFT:
                jumps.append((y, jump))

        current = float(row_marks[nearest])
        positions[y] = current

    return positions, jumps


def text_shift_across(gray: np.ndarray, row: int) -> int | None:
    """
    Estimate how far body text moved across a row, by column-profile correlation.
    """
    height, width = gray.shape

    above_start, above_end = max(0, row - CORROBORATION_WINDOW), row
    below_start, below_end = row, min(height, row + CORROBORATION_WINDOW)

    if above_end - above_start < 4 or below_end - below_start < 4:
        return None

    body = slice(int(width * SIDE_FRACTION), int(width * 0.85))

    above = (gray[above_start:above_end, body] < 128).sum(axis=0).astype(float)
    below = (gray[below_start:below_end, body] < 128).sum(axis=0).astype(float)

    if above.sum() < 20 or below.sum() < 20:
        return None

    correlation = np.correlate(below - below.mean(), above - above.mean(),
                               mode="full")
    centre = len(above) - 1
    low = max(0, centre - MAX_SHIFT)
    window = correlation[low: centre + MAX_SHIFT + 1]

    if not len(window) or not np.any(window > 0):
        return None

    return int(np.argmax(window)) - (centre - low)


def corroborated_boundaries(gray: np.ndarray) -> tuple[np.ndarray, list[int]]:
    """
    Return tracked border positions and the boundaries a genuine tear explains.

    A real tear moves everything in the band, so the body text shifts by the
    same amount as the border. A tracking artefact moves only the measurement.
    Requiring the two to agree rejected all 28 candidate boundaries on
    MIB-000003 (whose title line is intact) while confirming 8 on MIB-000457,
    where border and text move together to within a few pixels.
    """
    positions, jumps = track_border(gray)
    confirmed: list[int] = []

    for row, jump in jumps:
        shift = text_shift_across(gray, row)

        if shift is None:
            continue

        if abs(shift - jump) <= CORROBORATION_TOLERANCE:
            confirmed.append(row)

    return positions, sorted(confirmed)


def looks_torn(gray: np.ndarray) -> bool:
    """Return whether a page shows enough band displacement to be worth repairing."""
    offsets = estimate_offsets(gray)

    if not np.any(~np.isnan(offsets)):
        return False

    spread = float(np.nanmax(offsets) - np.nanmin(offsets))

    return spread >= TEAR_SPREAD_THRESHOLD


def segment_bands(offsets: np.ndarray) -> list[tuple[int, int, float]]:
    """Group scanlines into bands sharing a near-constant offset."""
    bands: list[tuple[int, int, float]] = []
    start = 0
    current: list[float] = []

    for y, offset in enumerate(offsets):
        if np.isnan(offset):
            continue

        if current and abs(offset - float(np.median(current))) > BAND_JUMP:
            if y - start >= MIN_BAND_HEIGHT:
                bands.append((start, y, float(np.median(current))))

            start, current = y, [offset]
        else:
            if not current:
                start = y

            current.append(offset)

    if current and len(offsets) - start >= MIN_BAND_HEIGHT:
        bands.append((start, len(offsets), float(np.median(current))))

    return bands


def text_rows(gray: np.ndarray) -> np.ndarray:
    """Boolean mask of rows carrying body text, ignoring the border strip."""
    binary = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )[1]
    width = gray.shape[1]
    body = binary[:, int(width * SIDE_FRACTION): int(width * 0.75)]
    ink = (body > 0).sum(axis=1)

    return ink > max(3, ink.max() * 0.01)


def snap_bands_to_text_gaps(
    bands: list[tuple[int, int, float]],
    is_text: np.ndarray,
    search: int = 30,
) -> list[tuple[int, int, float]]:
    """
    Move band boundaries out of text lines and into the gaps between them.

    A boundary that lands mid-line splits that line between two different
    shifts, which tears the glyphs apart horizontally and reliably destroys
    the line. Measured on MIB-000003 page 1, 8 of 25 boundaries cut a text
    line, and the affected lines were exactly the ones OCR lost.
    """
    if not bands:
        return bands

    height = len(is_text)
    snapped: list[tuple[int, int, float]] = []

    for index, (top, bottom, shift) in enumerate(bands):
        new_top = top

        if 0 < top < height and is_text[top]:
            for distance in range(1, search + 1):
                if top - distance > 0 and not is_text[top - distance]:
                    new_top = top - distance
                    break

                if top + distance < height and not is_text[top + distance]:
                    new_top = top + distance
                    break

        if snapped:
            previous_top, _, previous_shift = snapped[-1]

            if new_top <= previous_top:
                new_top = top

            snapped[-1] = (previous_top, new_top, previous_shift)

        snapped.append((new_top, bottom, shift))

    return [band for band in snapped if band[1] > band[0]]


def repair_tear(gray: np.ndarray) -> tuple[np.ndarray, int]:
    """
    Undo band displacement, using only text-corroborated tear boundaries.

    Bands are delimited by boundaries where the body text moved with the
    border. Everything else is left alone: an unverified boundary that lands
    inside a text line splits the glyphs and costs more than the tear did.
    """
    positions, boundaries = corroborated_boundaries(gray)

    if not boundaries:
        return gray, 0

    height, width = gray.shape
    edges = [0] + boundaries + [height]

    bands: list[tuple[int, int, float]] = []

    for index in range(len(edges) - 1):
        top, bottom = edges[index], edges[index + 1]
        segment = positions[top:bottom]
        segment = segment[~np.isnan(segment)]

        if len(segment) >= 5:
            bands.append((top, bottom, float(np.median(segment))))

    if not bands:
        return gray, 0

    reference = float(np.median([band[2] for band in bands]))

    repaired = gray.copy()
    moved = 0

    for top, bottom, position in bands:
        shift = int(round(position - reference))

        if shift == 0 or abs(shift) > MAX_SHIFT:
            continue

        band = gray[top:bottom]
        shifted = np.full_like(band, 255)

        if shift > 0:
            shifted[:, : width - shift] = band[:, shift:]
        else:
            shifted[:, -shift:] = band[:, : width + shift]

        repaired[top:bottom] = shifted
        moved += 1

    return repaired, moved
