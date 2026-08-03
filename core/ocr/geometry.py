"""
Page-level rotation and shear correction from detected straight lines.

`estimate_skew_angle` in `engine.py` (`cv2.minAreaRect` over all thresholded
foreground) measures rotation only and is unreliable on degraded pages: it
returned 0.0 on `MIB-000045` page 1 despite ~3.4 degrees of real skew, because
edge artifacts and scan bands made the foreground's bounding box axis-aligned
regardless of the page's true tilt. It also has no concept of shear at all,
so a sheared-but-not-rotated page (measured on `MIB-000038` page 5: rotation
~0, shear ~2.5 degrees) is never corrected.

This module instead detects every sufficiently long, sufficiently solid,
sufficiently dark straight line on the page -- the printed border AND
interior ruled lines, which share the same page-level rotation/shear as the
border and provide far more independent measurements than the border alone
-- and pools their individual angles. Validated against a directly-measured
physical border (`MIB-000407`, corner-angle decomposition: rotation +1.96,
shear +3.18) to within 0.2 degrees, and against three text-based
cross-checks. See mib-scanline-skew-unhandled and mib-skew-alone-doesnt-pay
memory for the investigation history, including two earlier interventions
that measured near-null: this method is different in kind (direct geometric
measurement, no angle search) rather than a retuning of either of them.
"""

from __future__ import annotations

import math

import cv2
import numpy as np

MIN_LINE_LENGTH_FRACTION = 0.15
DARKEST_PERCENTILE = 55
MIN_FILL_RATIO = 0.97
MIN_LINES_PER_ORIENTATION = 3
VIGNETTE_MARGIN_FRACTION = 0.03


def _fill_ratios(dilated: np.ndarray, lines: np.ndarray) -> np.ndarray:
    """
    Fraction of each line's own path that is actually inked, vectorized.

    A genuine ruled line (border or table rule) is solid: this is close to
    1.0. A run of text characters that Hough's gap tolerance bridged into one
    "line" (e.g. a footer) has real gaps between letters and words: this is
    well below 1.0. `dilated` is the binary foreground mask dilated by one
    pixel so pointwise indexing tolerates the same anti-aliasing slack the
    original per-point 3x3 window check did, without a per-line Python loop.
    """
    h, w = dilated.shape
    ratios = np.empty(len(lines), dtype=float)

    for i, (x1, y1, x2, y2) in enumerate(lines):
        length = int(max(abs(x2 - x1), abs(y2 - y1)))

        if length < 2:
            ratios[i] = 1.0
            continue

        xs = np.clip(np.linspace(x1, x2, length).astype(int), 0, w - 1)
        ys = np.clip(np.linspace(y1, y2, length).astype(int), 0, h - 1)
        ratios[i] = float(np.mean(dilated[ys, xs] > 0))

    return ratios


def _line_darkness(gray: np.ndarray, lines: np.ndarray, samples: int = 25) -> np.ndarray:
    h, w = gray.shape
    darkness = np.empty(len(lines), dtype=float)

    for i, (x1, y1, x2, y2) in enumerate(lines):
        xs = np.clip(np.linspace(x1, x2, samples).astype(int), 0, w - 1)
        ys = np.clip(np.linspace(y1, y2, samples).astype(int), 0, h - 1)
        darkness[i] = float(np.mean(gray[ys, xs]))

    return darkness


def _robust_median(values: list[float], k: float = 3.0) -> float:
    arr = np.asarray(values, dtype=float)
    median = np.median(arr)
    mad = np.median(np.abs(arr - median)) + 1e-6
    kept = arr[np.abs(arr - median) < k * mad]
    return float(np.median(kept)) if len(kept) else float(median)


def _robust_fit(a_values: list[float], b_values: list[float], iterations: int = 4):
    a = np.asarray(a_values, dtype=float)
    b = np.asarray(b_values, dtype=float)
    slope, intercept = np.polyfit(a, b, 1)

    for _ in range(iterations):
        residual = np.abs(b - (slope * a + intercept))
        spread = residual.std()

        if spread < 1e-9:
            break

        keep = residual < 2.5 * spread

        if keep.sum() < max(4, int(0.3 * len(a))):
            break

        a, b = a[keep], b[keep]
        slope, intercept = np.polyfit(a, b, 1)

    return slope, intercept


class PageGeometry:
    """Measured rotation and shear, in degrees, for one rendered page."""

    __slots__ = ("rotation", "shear")

    def __init__(self, rotation: float, shear: float):
        self.rotation = rotation
        self.shear = shear


def measure_page_geometry(gray: np.ndarray) -> PageGeometry | None:
    """
    Measure page rotation and shear from detected straight lines.

    Returns None when there aren't enough usable lines of both orientations
    to measure confidently -- callers should treat that the same as "no
    correction needed" rather than guessing.
    """
    height, width = gray.shape
    binary = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )[1]
    dilated = cv2.dilate(binary, np.ones((3, 3), np.uint8))

    min_length = int(min(height, width) * MIN_LINE_LENGTH_FRACTION)
    detected = cv2.HoughLinesP(
        binary,
        rho=1,
        theta=np.pi / 720,
        threshold=60,
        minLineLength=min_length,
        maxLineGap=int(min_length * 0.15),
    )

    if detected is None:
        return None

    lines = detected.reshape(-1, 4)
    fill_ratios = _fill_ratios(dilated, lines)
    solid = lines[fill_ratios >= MIN_FILL_RATIO]

    if len(solid) < 2 * MIN_LINES_PER_ORIENTATION:
        return None

    darkness = _line_darkness(gray, solid)
    margin_x = width * VIGNETTE_MARGIN_FRACTION
    margin_y = height * VIGNETTE_MARGIN_FRACTION

    horizontal: list[dict] = []
    vertical: list[dict] = []

    for (x1, y1, x2, y2), dark in zip(solid, darkness):
        dx, dy = x2 - x1, y2 - y1
        angle_horizontal = math.degrees(math.atan2(dy, dx))
        angle_vertical = math.degrees(math.atan2(dx, dy))
        # Normalize both to (-90, 90] -- a Hough segment's endpoint order is
        # arbitrary, so a line and its reverse must read the same angle.
        normalized_horizontal = ((angle_horizontal + 90) % 180) - 90
        normalized_vertical = ((angle_vertical + 90) % 180) - 90
        center_x, center_y = (x1 + x2) / 2.0, (y1 + y2) / 2.0

        if abs(normalized_horizontal) <= 30:
            # Top/bottom borders legitimately sit close to the page edge --
            # no margin exclusion here (it wrongly dropped a genuine bottom
            # border on one measured page).
            horizontal.append(dict(
                angle=normalized_horizontal, center_x=center_x, center_y=center_y,
                dark=dark, edge_distance=min(center_y, height - center_y),
                x1=x1, y1=y1, x2=x2, y2=y2,
            ))
        elif abs(normalized_vertical) <= 30:
            # Left/right scan-edge vignette artifacts are specifically a
            # vertical-line problem -- exclude only here, not symmetrically.
            if center_x < margin_x or center_x > width - margin_x:
                continue

            vertical.append(dict(
                angle=normalized_vertical, center_x=center_x, center_y=center_y,
                dark=dark, edge_distance=min(center_x, width - center_x),
                x1=x1, y1=y1, x2=x2, y2=y2,
            ))

    if len(horizontal) < MIN_LINES_PER_ORIENTATION:
        # Shear needs a rotation reference to be measured relative to, but
        # rotation alone does not need verticals at all -- requiring both
        # orientations discarded good rotation-only data on pages whose left
        # border is disconnected tick/dash marks rather than one continuous
        # line (measured on MIB-000670 p2: a scan-line tear left 0 vertical
        # lines but 11 horizontal ones, tightly agreeing at -1.20 to -1.26
        # degrees -- thrown away entirely under the old all-or-nothing rule).
        return None

    def darkest_tier(items: list[dict]) -> list[dict]:
        threshold = np.percentile([item["dark"] for item in items], DARKEST_PERCENTILE)
        kept = [item for item in items if item["dark"] <= threshold]
        return kept if len(kept) >= MIN_LINES_PER_ORIENTATION else items

    horizontal = darkest_tier(horizontal)
    rotation = _robust_median([item["angle"] for item in horizontal])

    if len(vertical) < MIN_LINES_PER_ORIENTATION:
        return PageGeometry(rotation=rotation, shear=0.0)

    vertical = darkest_tier(vertical)
    vertical_tilt = _robust_median([item["angle"] for item in vertical])
    # Verified against a directly-measured physical border (MIB-000407):
    # shear = -(vertical_tilt + rotation) reproduces +3.18 degrees to within
    # 0.02; the naive `vertical_tilt - rotation` gives -7.10, wrong sign and
    # magnitude.
    shear = -(vertical_tilt + rotation)

    return PageGeometry(rotation=rotation, shear=shear)


def correct_page_geometry(
    image: np.ndarray,
    geometry: PageGeometry,
) -> np.ndarray:
    """
    Undo the measured rotation and shear with a single affine warp.

    Unlike a rotation-only deskew, this also straightens shear -- measured on
    MIB-000407 page 3 through the actual OCR/extraction pipeline: baseline
    3/7 fields correct, rotation-only correction 1/7, this combined
    correction 6/7 (recovered sponsor_id and arrival_date digit misreads and
    let declared_purpose extract at all).
    """
    height, width = image.shape[:2]
    rotation_radians = math.radians(geometry.rotation)
    vertical_tilt_radians = math.radians(-(geometry.shear + geometry.rotation))

    # Forward model: a clean, axis-aligned page's local (u, v) axes map to
    # image-space directions (cos(rotation), sin(rotation)) and
    # (sin(vertical_tilt), cos(vertical_tilt)). Correcting the page is the
    # inverse of that mapping.
    forward = np.array([
        [math.cos(rotation_radians), math.sin(vertical_tilt_radians)],
        [math.sin(rotation_radians), math.cos(vertical_tilt_radians)],
    ])
    inverse = np.linalg.inv(forward)

    center = np.array([width / 2.0, height / 2.0])
    affine = np.zeros((2, 3))
    affine[:, :2] = inverse
    affine[:, 2] = center - inverse @ center

    return cv2.warpAffine(
        image,
        affine,
        (width, height),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255) if image.ndim == 3 else 255,
    )
