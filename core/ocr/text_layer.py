"""
Visible-text extraction from a PDF text layer.

The packets embed adversarial content in the native text layer: white-on-white
spans, text positioned outside the page crop, and fake "answer key" blocks. The
field manual is explicit that this content is not trusted evidence, so the
pipeline must never read it as if it were printed on the page.

`get_visible_text` returns only the spans a human would actually see.
"""

from __future__ import annotations

import fitz


# sRGB integer for pure white. Spans at or near this are invisible against the
# page background.
WHITE = 0xFFFFFF

# A channel value above this counts as "effectively white".
NEAR_WHITE_CHANNEL = 0.92

# Spans must overlap the visible page area by at least this fraction of their
# own area to count as on-page.
MIN_ON_PAGE_OVERLAP = 0.5


def unpack_rgb(color: int) -> tuple[float, float, float]:
    """Split an sRGB integer into channels normalized to 0..1."""
    return (
        ((color >> 16) & 0xFF) / 255.0,
        ((color >> 8) & 0xFF) / 255.0,
        (color & 0xFF) / 255.0,
    )


def is_invisible_color(color: int) -> bool:
    """
    Return whether a span colour is effectively invisible on a white page.

    Only near-white is treated as invisible. Light greys are kept because the
    packets legitimately use grey for footers and secondary labels.
    """
    red, green, blue = unpack_rgb(color)

    return (
        red >= NEAR_WHITE_CHANNEL
        and green >= NEAR_WHITE_CHANNEL
        and blue >= NEAR_WHITE_CHANNEL
    )


def is_on_page(span_bbox, page_rect: fitz.Rect) -> bool:
    """Return whether a span sits inside the visible page crop."""
    span_rect = fitz.Rect(span_bbox)

    if span_rect.is_empty or span_rect.is_infinite:
        return False

    overlap = span_rect & page_rect

    if overlap.is_empty:
        return False

    span_area = abs(span_rect.get_area())

    if span_area == 0:
        return False

    return abs(overlap.get_area()) / span_area >= MIN_ON_PAGE_OVERLAP


def iter_visible_spans(page: fitz.Page):
    """Yield the spans on a page that a human reader would see."""
    page_rect = page.rect
    text_dict = page.get_text("dict")

    for block in text_dict.get("blocks", []):
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                if not span.get("text", "").strip():
                    continue

                if is_invisible_color(span.get("color", 0)):
                    continue

                if not is_on_page(span["bbox"], page_rect):
                    continue

                yield span


def get_visible_text(page: fitz.Page) -> str:
    """
    Return the page's native text layer with hidden content removed.

    Blocks and lines are emitted in the order PyMuPDF reports them, which is the
    same reading order `page.get_text("text")` produces. Re-sorting by geometry
    would separate table labels from their values — these packets lay out
    "Fee Status" and "paid" on the same visual row — and every label-then-value
    extractor regex depends on that order surviving.
    """
    page_rect = page.rect
    text_dict = page.get_text("dict")

    lines: list[str] = []

    for block in text_dict.get("blocks", []):
        for line in block.get("lines", []):
            parts = [
                span["text"]
                for span in line.get("spans", [])
                if span.get("text", "").strip()
                and not is_invisible_color(span.get("color", 0))
                and is_on_page(span["bbox"], page_rect)
            ]

            if parts:
                lines.append("".join(parts))

    return "\n".join(lines)


def get_hidden_text(page: fitz.Page) -> str:
    """
    Return only the text that `get_visible_text` filtered out.

    This is never used as evidence. It exists so the pipeline can tell
    "unknown from trusted evidence" apart from "supplied by prompt injection",
    which the field manual requires be treated differently.
    """
    page_rect = page.rect
    text_dict = page.get_text("dict")

    hidden: list[str] = []

    for block in text_dict.get("blocks", []):
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                text = span.get("text", "")

                if not text.strip():
                    continue

                if is_invisible_color(
                    span.get("color", 0)
                ) or not is_on_page(span["bbox"], page_rect):
                    hidden.append(text)

    return "\n".join(hidden)