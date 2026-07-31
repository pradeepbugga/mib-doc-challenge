from __future__ import annotations

from collections import defaultdict
from typing import Iterable
import re

from core.adjudication.models import FieldObservation, Packet, ResolvedField

def normalize_for_comparison(
    value: str,
) -> str:
    """
    Normalize text for matching without changing stored evidence.
    """
    value = value.casefold()
    value = " ".join(value.split())

    return value.strip()

def source_priority(
    observation: FieldObservation,
) -> int:
    """
    Rank extraction text sources by expected transcription quality.

    This ranks extraction reliability, not document authority.
    """
    priorities = {
        "native_text": 3,
        "ocr": 2,
    }

    return priorities.get(
        observation.text_source or "",
        1,
    )

def is_low_information_ocr_suffix(
    suffix: str,
) -> bool:
    """
    Return True when trailing OCR text appears to be noise rather
    than meaningful additional content.
    """
    alphanumeric_text = re.sub(
        r"[^a-z0-9]+",
        "",
        suffix.casefold(),
    )

    return len(alphanumeric_text) <= 3

def native_value_matches_ocr(
    native_value: str,
    ocr_value: str,
) -> bool:
    """
    Determine whether a clean native-text value is supported by
    a noisier OCR value.

    The OCR value may contain a small trailing noise suffix, but
    not meaningful additional text.
    """
    native = normalize_for_comparison(native_value)
    ocr = normalize_for_comparison(ocr_value)

    if not native or not ocr:
        return False

    if native == ocr:
        return True

    if not ocr.startswith(native):
        return False

    suffix = ocr[len(native):].strip()

    return is_low_information_ocr_suffix(suffix)

def observations_match(
    first: FieldObservation,
    second: FieldObservation,
) -> bool:
    """
    Compare two observations using exact normalized matching first,
    then conservative source-aware native/OCR matching.
    """
    first_value = first.normalized_value
    second_value = second.normalized_value

    if first_value is None or second_value is None:
        return False

    first_comparison = normalize_for_comparison(
        first_value
    )
    second_comparison = normalize_for_comparison(
        second_value
    )

    if first_comparison == second_comparison:
        return True

    if (
        first.text_source == "native_text"
        and second.text_source == "ocr"
    ):
        return native_value_matches_ocr(
            first_value,
            second_value,
        )

    if (
        first.text_source == "ocr"
        and second.text_source == "native_text"
    ):
        return native_value_matches_ocr(
            second_value,
            first_value,
        )

    return False
def corroborate_field(
    field_name: str,
    observations: list[FieldObservation],
) -> ResolvedField:
    """
    Resolve one canonical field from all usable observations.
    """
    usable = [
        observation
        for observation in observations
        if observation.trusted
        and observation.normalized_value is not None
    ]

    if not usable:
        return ResolvedField(
            field=field_name,
            resolved_value=None,
            status="missing",
            observations=observations,
            resolution_method=None,
        )

    ranked = sorted(
        usable,
        key=source_priority,
        reverse=True,
    )

    candidate = ranked[0]

    matching = [
        observation
        for observation in usable
        if observations_match(
            candidate,
            observation,
        )
    ]

    conflicting = [
        observation
        for observation in usable
        if not observations_match(
            candidate,
            observation,
        )
    ]

    if conflicting:
        return ResolvedField(
            field=field_name,
            resolved_value=None,
            status="conflicting",
            observations=observations,
            resolution_method=None,
        )

    distinct_document_types = {
        observation.document_type
        for observation in matching
    }

    if len(distinct_document_types) > 1:
        status = "corroborated"
    else:
        status = "single_source"

    has_native_ocr_pair = (
        any(
            observation.text_source == "native_text"
            for observation in matching
        )
        and any(
            observation.text_source == "ocr"
            for observation in matching
        )
    )

    comparison_values = {
        normalize_for_comparison(
            observation.normalized_value
        )
        for observation in matching
        if observation.normalized_value is not None
    }

    if len(matching) == 1:
        resolution_method = "single_observation"

    elif len(comparison_values) == 1:
        resolution_method = "exact_match"

    elif has_native_ocr_pair:
        resolution_method = "native_ocr_match"

    else:
        resolution_method = "source_aware_match"

    return ResolvedField(
        field=field_name,
        resolved_value=candidate.normalized_value,
        status=status,
        observations=observations,
        resolution_method=resolution_method,
    )

def corroborate_packet(
    observations: list[FieldObservation],
) -> Packet:
    """
    Group observations by canonical field and resolve each field.
    """
    grouped: dict[str, list[FieldObservation]] = defaultdict(
        list
    )

    for observation in observations:
        grouped[observation.field].append(
            observation
        )

    resolved_fields = {
        field_name: corroborate_field(
            field_name=field_name,
            observations=field_observations,
        )
        for field_name, field_observations in grouped.items()
    }

    return Packet(fields=resolved_fields)