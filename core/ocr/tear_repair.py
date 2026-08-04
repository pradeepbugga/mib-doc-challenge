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

# Row must carry more than this much ink to count as part of a text line.
MIN_LINE_INK = 4

# Shorter runs of inked rows are speckle, not a line of text.
MIN_LINE_HEIGHT = 6

# Left-edge alignment needs a few lines before its median means anything.
MIN_ALIGNABLE_LINES = 3

# Left-edge displacement below this is within normal glyph variation.
MIN_LINE_SHIFT = 3


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


# --- Validated per-row mark detection (replaces the old leftmost_marks path
# for band offset estimation; see TEAR_REPAIR_TODO.md) ---

# A qualifying mark run's width band, in pixels: thin enough to be a rule
# stroke rather than a text blob, wide enough not to be a speckle.
MARK_MIN_WIDTH = 3
MARK_MAX_WIDTH = 12

# Absolute grey level below which a pixel counts as ink for mark detection.
MARK_DARKNESS = 180

# A candidate mark must have a similar x (within this many px) in at least
# this fraction of rows within +/- MARK_CONTINUITY_WINDOW to survive.
MARK_CONTINUITY_TOLERANCE = 6
MARK_CONTINUITY_WINDOW = 10
MARK_CONTINUITY_MIN_FRACTION = 0.5

# Left margin coverage (fraction of rows with a surviving mark) required to
# trust it outright; below this, the right margin is tried as a fallback and
# whichever side covers more rows wins. See TEAR_REPAIR_TODO.md item 6 --
# tested directly on MIB-000030 p2, where the visually-cleaner right edge
# produced nonsensical shifts and the messier left edge (72.5% coverage)
# recovered the field the right edge lost.
MIN_SIDE_COVERAGE = 0.60

# A short band sandwiched between two neighbors that agree with each other
# but not with it is noise (a hole-punch, a stamp), not a real tear boundary.
OUTLIER_BAND_MAX_HEIGHT = 40
OUTLIER_NEIGHBOR_AGREEMENT = 10
OUTLIER_DEVIATION = 30

# Rows from each end of a band used to compute its local top/bottom edge
# value for cascade repair, rather than one flat whole-band median.
EDGE_SAMPLE_ROWS = 15


def _row_marks(gray: np.ndarray, side: str) -> np.ndarray:
    """
    x of the qualifying thin dark run nearest the given margin on each row.

    Per-row scanning for a thin run (width MARK_MIN_WIDTH..MARK_MAX_WIDTH,
    absolute intensity < MARK_DARKNESS) rather than connected-components:
    a vertical border segment touching the page's horizontal top border at a
    corner merges into one L-shaped blob under connected-components, and a
    whole-blob width filter wrongly rejects a genuine long line. Per-row
    scanning has no such blind spot.
    """
    height, width = gray.shape
    margin = int(width * SIDE_FRACTION)

    if side == "left":
        strip = gray[:, :margin] < MARK_DARKNESS
    else:
        strip = gray[:, width - margin:] < MARK_DARKNESS

    marks = np.full(height, np.nan)

    for y in range(height):
        row = strip[y]
        xs = np.nonzero(row)[0]

        if not len(xs):
            continue

        # Split into contiguous runs and keep the first (left) / last (right)
        # one whose width qualifies.
        breaks = np.nonzero(np.diff(xs) > 1)[0]
        run_starts = np.concatenate(([0], breaks + 1))
        run_ends = np.concatenate((breaks, [len(xs) - 1]))

        run_indices = range(len(run_starts)) if side == "left" else reversed(range(len(run_starts)))

        for i in run_indices:
            run_width = xs[run_ends[i]] - xs[run_starts[i]] + 1

            if MARK_MIN_WIDTH <= run_width <= MARK_MAX_WIDTH:
                position = xs[run_starts[i]]
                marks[y] = float(position if side == "left" else width - margin + position)
                break

    return marks


def _apply_continuity_filter(marks: np.ndarray) -> np.ndarray:
    """
    Drop any mark without a similar x in most nearby rows.

    Rejects isolated text characters and other one-off contamination (a
    stray letter picked up near a text line) while preserving genuine
    multi-row strokes, which is what a real border or tear edge is.
    """
    height = len(marks)
    filtered = marks.copy()

    for y in range(height):
        if np.isnan(marks[y]):
            continue

        low = max(0, y - MARK_CONTINUITY_WINDOW)
        high = min(height, y + MARK_CONTINUITY_WINDOW + 1)
        window = marks[low:high]
        window = window[~np.isnan(window)]

        if len(window) == 0:
            filtered[y] = np.nan
            continue

        agreeing = np.sum(np.abs(window - marks[y]) <= MARK_CONTINUITY_TOLERANCE)

        if agreeing / len(window) < MARK_CONTINUITY_MIN_FRACTION:
            filtered[y] = np.nan

    return filtered


def side_marks_and_coverage(gray: np.ndarray, side: str) -> tuple[np.ndarray, float]:
    """Return continuity-filtered marks for one margin and their row coverage."""
    marks = _apply_continuity_filter(_row_marks(gray, side))
    coverage = float(np.sum(~np.isnan(marks))) / len(marks) if len(marks) else 0.0

    return marks, coverage


def estimate_offsets_v2(gray: np.ndarray) -> np.ndarray:
    """
    Return each scanline's displacement from its own margin's median mark.

    Tracks the left margin first; if its coverage is below MIN_SIDE_COVERAGE
    the right margin is tried and whichever side covers more rows is used.
    Do not prefer whichever side "looks cleaner" -- see MIN_SIDE_COVERAGE.
    """
    left_marks, left_coverage = side_marks_and_coverage(gray, "left")

    if left_coverage >= MIN_SIDE_COVERAGE:
        marks = left_marks
    else:
        right_marks, right_coverage = side_marks_and_coverage(gray, "right")
        marks = right_marks if right_coverage > left_coverage else left_marks

    measured = marks[~np.isnan(marks)]

    if not len(measured):
        return marks

    offsets = marks - float(np.median(measured))
    offsets[np.abs(offsets) > MAX_SHIFT] = np.nan

    return offsets


def clean_outlier_bands(
    bands: list[tuple[int, int, float]],
) -> list[tuple[int, int, float]]:
    """
    Absorb a short band sandwiched between two agreeing neighbors into its
    predecessor.

    A hole-punch circle or stamp interrupting a border for a few dozen rows
    creates a spurious band reading something else entirely (e.g. a leftmost
    text character). It is noise, not a real tear boundary.
    """
    if len(bands) < 3:
        return bands

    cleaned = [bands[0]]

    for index in range(1, len(bands) - 1):
        top, bottom, offset = bands[index]
        prev_offset = cleaned[-1][2]
        next_offset = bands[index + 1][2]

        is_short = (bottom - top) <= OUTLIER_BAND_MAX_HEIGHT
        neighbors_agree = abs(prev_offset - next_offset) <= OUTLIER_NEIGHBOR_AGREEMENT
        this_deviates = (
            abs(offset - prev_offset) >= OUTLIER_DEVIATION
            and abs(offset - next_offset) >= OUTLIER_DEVIATION
        )

        if is_short and neighbors_agree and this_deviates:
            prev_top, _, prev_offset_value = cleaned[-1]
            cleaned[-1] = (prev_top, bottom, prev_offset_value)
        else:
            cleaned.append((top, bottom, offset))

    cleaned.append(bands[-1])

    return cleaned


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
    offsets = estimate_offsets_v2(gray)

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


def detect_text_lines(
    gray: np.ndarray,
    left_margin: float = 0.05,
    right_margin: float = 0.80,
) -> tuple[list[tuple[int, int]], np.ndarray, int]:
    """Group rows into text lines, ignoring page margins and horizontal rules."""
    height, width = gray.shape

    binary = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )[1]

    rules = cv2.morphologyEx(
        binary,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (80, 1)),
        iterations=1,
    )
    text = cv2.subtract(binary, rules)

    start_x = int(width * left_margin)
    ink = (text[:, start_x: int(width * right_margin)] > 0).sum(axis=1)
    inked = ink > MIN_LINE_INK

    lines: list[tuple[int, int]] = []
    y = 0

    while y < height:
        if not inked[y]:
            y += 1
            continue

        top = y

        while y < height and inked[y]:
            y += 1

        if y - top >= MIN_LINE_HEIGHT:
            lines.append((top, y))

    return lines, text, start_x


def align_text_lines(gray: np.ndarray) -> tuple[np.ndarray, int]:
    """
    Repair tearing by aligning each text line's left edge.

    The band-based repair measures the page border and asks whether body text
    moved with it. That test only works when a tear cuts through a glyph, so
    both sides of the boundary are the same letters. When the tear falls
    between lines — the common case — it compares the bottom of one line with
    the top of a different one, which correlates at nothing, and every real
    boundary is rejected. On MIB-000039 page 2 that left 0 of 37 candidates
    confirmed on a page whose every field line is visibly offset.

    These forms label every field at the same left margin, so a line's left
    edge is its displacement directly, with no boundary detection needed.
    """
    edges: list[tuple[int, int, int | None]] = []
    lines, text, start_x = detect_text_lines(gray)

    for top, bottom in lines:
        columns = np.nonzero(text[top:bottom].sum(axis=0))[0]
        columns = columns[columns >= start_x]
        edges.append(
            (top, bottom, int(columns[0]) if len(columns) else None)
        )

    measured = [left for _, _, left in edges if left is not None]

    if len(measured) < MIN_ALIGNABLE_LINES:
        return gray, 0

    reference = int(np.median(measured))
    height, width = gray.shape

    aligned = gray.copy()
    moved = 0

    for top, bottom, left in edges:
        if left is None:
            continue

        shift = left - reference

        if abs(shift) < MIN_LINE_SHIFT or abs(shift) > MAX_SHIFT:
            continue

        line = gray[top:bottom]
        shifted = np.full_like(line, 255)

        if shift > 0:
            shifted[:, : width - shift] = line[:, shift:]
        else:
            shifted[:, -shift:] = line[:, : width + shift]

        aligned[top:bottom] = shifted
        moved += 1

    return aligned, moved


def corroborate_band_boundaries(
    gray: np.ndarray,
    bands: list[tuple[int, int, float]],
) -> list[tuple[int, int, float]]:
    """
    Merge any band boundary body text does not corroborate.

    The continuity filter in _row_marks/_apply_continuity_filter rejects
    isolated noise, but a multi-row artefact (a decorative rule, a stamp
    edge) can still pass it and look like a genuine band split. A real tear
    moves everything in the band, so the body text shifts by the same amount
    as the tracked margin mark at the boundary row; an artefact moves only
    the measurement. This mirrors the old track_border-based
    corroborated_boundaries check, reapplied to the new detection's band
    boundaries instead of individual row jumps -- restores the precision
    that check gave (rejected all 28 spurious candidates on MIB-000003) that
    plain continuity-filtering alone does not provide, confirmed by a
    full-training-set regression (-0.14) when this step was left out.

    Unlike the old check, a boundary with nothing to corroborate against (no
    text on one side) is kept rather than rejected, not treated as
    equivalent to a disagreement. The old check only ever evaluated a
    handful of individual jump rows flagged by track_border; segment_bands
    here produces many more candidate boundaries, most from the blank lower
    portion of a typical page, and requiring positive evidence for every one
    of them collapsed the entire page to a single band (confirmed directly:
    both MIB-000670 p2 and MIB-000030 p2 dropped to 0 bands moved). Only
    reject a boundary body text actively disagrees with. Tears that fall
    between text lines, where there is nothing to corroborate against on
    principle, are handled by the separate align_text_lines candidate
    instead; see try_tear_repair in page_pipeline.py.
    """
    if len(bands) < 2:
        return bands

    merged: list[tuple[int, int, float]] = [bands[0]]

    for top, bottom, offset in bands[1:]:
        prev_top, prev_bottom, prev_offset = merged[-1]
        boundary_row = top  # == prev_bottom, bands are contiguous

        jump = offset - prev_offset
        shift = text_shift_across(gray, boundary_row)
        disagrees = shift is not None and abs(shift - jump) > CORROBORATION_TOLERANCE

        if not disagrees:
            merged.append((top, bottom, offset))
        else:
            prev_height = prev_bottom - prev_top
            this_height = bottom - top
            total = prev_height + this_height
            blended_offset = (
                (prev_offset * prev_height + offset * this_height) / total
                if total
                else prev_offset
            )
            merged[-1] = (prev_top, bottom, blended_offset)

    return merged


def _local_edge_values(
    gray: np.ndarray,
    offsets: np.ndarray,
    top: int,
    bottom: int,
) -> tuple[float, float]:
    """
    Return a band's own local top-edge and bottom-edge offset.

    A flat whole-band median collapses cascade repair to the same wrong
    "everything equals one global value" result a single global reference
    gives, since a flat band's top and bottom are identical. Using the
    first/last rows' own median instead preserves real internal drift (e.g.
    a genuinely continuous but slightly rotated border) and lets adjacent
    bands be chained edge-to-edge instead of centre-to-centre.
    """
    segment = offsets[top:bottom]
    sample = min(EDGE_SAMPLE_ROWS, len(segment) // 2 or len(segment))

    head = segment[:sample]
    tail = segment[-sample:] if sample else segment

    head = head[~np.isnan(head)]
    tail = tail[~np.isnan(tail)]

    whole = segment[~np.isnan(segment)]
    fallback = float(np.median(whole)) if len(whole) else 0.0

    top_value = float(np.median(head)) if len(head) else fallback
    bottom_value = float(np.median(tail)) if len(tail) else fallback

    return top_value, bottom_value


def _cascade_shifts(
    edge_values: list[tuple[float, float]],
) -> list[float]:
    """
    Return one shift per band, anchored at the most typical (offset closest
    to zero) band and cascaded edge-to-edge through its neighbors.

    `estimate_offsets_v2` already centres offsets on the page's own median
    mark position, so the band whose local offset sits closest to zero is,
    by construction, closest to this page's dominant/majority border
    position -- the best available proxy for "confirmed undamaged" without
    per-page manual inspection. Zero shift is assigned there; each other
    band's shift is chosen so its edge touching an already-corrected
    neighbor lines up with that neighbor's corrected edge, cascading
    outward. A flat global reference was the confirmed bug this replaces:
    averaging over mostly-noise bands (blank page regions, stamps) pulled an
    undamaged header 25px off its true position.
    """
    if not edge_values:
        return []

    band_medians = [(top + bottom) / 2 for top, bottom in edge_values]
    anchor = int(np.argmin(np.abs(band_medians)))

    shifts = [0.0] * len(edge_values)

    for i in range(anchor - 1, -1, -1):
        # band i is above band i+1; i's bottom touches (i+1)'s (corrected) top.
        next_top_corrected = edge_values[i + 1][0] + shifts[i + 1]
        shifts[i] = next_top_corrected - edge_values[i][1]

    for i in range(anchor + 1, len(edge_values)):
        # band i is below band i-1; i's top touches (i-1)'s (corrected) bottom.
        prev_bottom_corrected = edge_values[i - 1][1] + shifts[i - 1]
        shifts[i] = prev_bottom_corrected - edge_values[i][0]

    return shifts


def repair_tear(gray: np.ndarray) -> tuple[np.ndarray, int]:
    """
    Undo band displacement with per-row mark detection and cascade repair.

    Offsets come from a continuity-filtered per-row mark scan of whichever
    margin covers more of the page (see estimate_offsets_v2), segmented into
    bands, cleaned of short noise bands, and snapped off text lines so a
    boundary never cuts through a glyph. Each band's shift is then chained
    from its own local top/bottom edge values, anchored at the page's most
    typical band, rather than pulled toward one flat global reference.
    """
    offsets = estimate_offsets_v2(gray)

    if not np.any(~np.isnan(offsets)):
        return gray, 0

    bands = segment_bands(offsets)

    if not bands:
        return gray, 0

    bands = clean_outlier_bands(bands)
    # Corroborate at the raw boundary positions, near wherever the actual
    # discontinuity is, before snapping moves them into text-free whitespace
    # gaps -- text_shift_across needs nearby text to correlate against, which
    # whitespace by definition doesn't have.
    bands = corroborate_band_boundaries(gray, bands)
    bands = snap_bands_to_text_gaps(bands, text_rows(gray))

    if not bands:
        return gray, 0

    edge_values = [
        _local_edge_values(gray, offsets, top, bottom) for top, bottom, _ in bands
    ]
    shifts = _cascade_shifts(edge_values)

    height, width = gray.shape
    repaired = gray.copy()
    moved = 0

    for (top, bottom, _), shift_value in zip(bands, shifts):
        shift = int(round(shift_value))

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
