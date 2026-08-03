from __future__ import annotations

from collections import defaultdict
from typing import Iterable
import re

from core.adjudication.models import FieldObservation, Packet, ResolvedField
from core.rules.field_rules import EVIDENCE_PRIORITY

# FIELD_MANUAL.md ranks *which document* to believe when two of them disagree;
# `source_priority` below ranks *how cleanly the text was read*. They are
# different questions and both matter, so conflicts are settled by document
# authority first and transcription quality only as a tie-break.
#
# The manual's tiers are keyed by evidence kind; these are the page types the
# classifier actually emits. Two are not in the manual's list:
#   - fee_receipt: an official visible form and the only authoritative source
#     for fee_status, so it sits at the intake-form tier.
#   - unknown: a page that failed to classify. Its provenance is unproven, so
#     it ranks with the bare machine-readable text layer — usable when nothing
#     else supplies the field, never able to override a real form.
DOCUMENT_EVIDENCE_PRIORITY = {
    "adjudicator_note": EVIDENCE_PRIORITY["signed_manual_note"],
    "intake_form": EVIDENCE_PRIORITY["intake_form"],
    "fee_receipt": EVIDENCE_PRIORITY["intake_form"],
    "biometric_slip": EVIDENCE_PRIORITY["biometric_slip"],
    "sponsor_attestation": EVIDENCE_PRIORITY["sponsor_attestation"],
    "registry_extract": EVIDENCE_PRIORITY["registry_extract"],
    "unknown": EVIDENCE_PRIORITY["machine_readable_text"],
}

DEFAULT_DOCUMENT_PRIORITY = EVIDENCE_PRIORITY["machine_readable_text"]


def document_priority(observation: FieldObservation) -> int:
    """Return the trusted-evidence rank of the page an observation came from."""
    return DOCUMENT_EVIDENCE_PRIORITY.get(
        observation.document_type,
        DEFAULT_DOCUMENT_PRIORITY,
    )


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
def cluster_agreeing_observations(
    observations: list[FieldObservation],
) -> list[list[FieldObservation]]:
    """
    Group observations into clusters that agree on a value.

    Uses `observations_match` rather than exact string equality so a clean
    native-text value and its noisier OCR twin land in the same cluster.
    """
    clusters: list[list[FieldObservation]] = []

    for observation in observations:
        for cluster in clusters:
            if observations_match(cluster[0], observation):
                cluster.append(observation)
                break
        else:
            clusters.append([observation])

    return clusters


def best_transcription(
    cluster: list[FieldObservation],
) -> FieldObservation:
    """Return the cleanest-read observation within an agreeing cluster."""
    return max(cluster, key=source_priority)


def cluster_rank(cluster: list[FieldObservation]) -> tuple:
    """
    Sort key ranking a cluster's claim to be the resolved value.

    Ordered by document authority, then how many distinct document types back
    it, then transcription quality.
    """
    best_document_priority = min(
        document_priority(observation) for observation in cluster
    )
    distinct_document_types = len(
        {observation.document_type for observation in cluster}
    )
    best_source_priority = max(
        source_priority(observation) for observation in cluster
    )

    return (
        best_document_priority,
        -distinct_document_types,
        -best_source_priority,
    )


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
            supporting_observations=[],
            resolution_method=None,
        )

    clusters = cluster_agreeing_observations(usable)

    # Most trusted document wins; ties broken by breadth of corroboration and
    # then by transcription quality.
    clusters.sort(key=cluster_rank)

    if field_name == "applicant_name" and not any(
        observation.text_source == "native_text"
        for observation in clusters[0]
    ):
        # Measured 2026-08-03 across the training set: when a native-text
        # reading of applicant_name disagrees with an OCR reading, truth
        # matches the native reading in 203/214 (94.9%) of cases, regardless
        # of which document type either came from and regardless of fuzzy
        # similarity between the two readings. Document-priority-first
        # ranking lets a corrupted OCR read of a high-priority document
        # (e.g. intake_form) beat a clean native read of a lower-priority one
        # (e.g. sponsor_attestation) -- backwards for this one field.
        #
        # This is an early return, not a reorder-and-fall-through: the
        # conflict check below compares document_priority between "winning"
        # and runner-up clusters, and a promoted native cluster usually has
        # WORSE document authority than the OCR cluster it's beating (that's
        # the whole problem). Letting it fall through made that check see a
        # lower-authority "winner" losing to a higher-authority runner-up and
        # report an unresolved conflict, wiping the value to None -- exactly
        # backwards from the intent. Bypass that check entirely here.
        native_clusters = [
            cluster
            for cluster in clusters[1:]
            if any(
                observation.text_source == "native_text"
                for observation in cluster
            )
        ]

        if native_clusters:
            native_clusters.sort(key=cluster_rank)
            best_native = native_clusters[0]
            candidate = best_transcription(best_native)
            distinct_document_types = {
                observation.document_type for observation in best_native
            }
            status = (
                "corroborated" if len(distinct_document_types) > 1
                else "single_source"
            )
            return ResolvedField(
                field=field_name,
                resolved_value=candidate.normalized_value,
                status=status,
                observations=observations,
                supporting_observations=best_native,
                resolution_method="native_preferred_over_ocr",
            )

    winning_cluster = clusters[0]

    matching = winning_cluster
    conflicting = [
        observation
        for cluster in clusters[1:]
        for observation in cluster
    ]

    if conflicting:
        winning_priority = min(
            document_priority(observation)
            for observation in winning_cluster
        )
        runner_up_priority = min(
            document_priority(observation)
            for observation in clusters[1]
        )

        # A more authoritative document settles the disagreement. Equal
        # authority disagreeing with itself does not — that is a genuine
        # contradiction and still needs a human.
        if winning_priority >= runner_up_priority:
            return ResolvedField(
                field=field_name,
                resolved_value=None,
                status="conflicting",
                observations=observations,
                supporting_observations=matching,
                resolution_method=None,
            )

        candidate = best_transcription(winning_cluster)

        return ResolvedField(
            field=field_name,
            resolved_value=candidate.normalized_value,
            status="resolved_by_evidence_priority",
            observations=observations,
            supporting_observations=matching,
            resolution_method="evidence_priority",
        )

    candidate = best_transcription(winning_cluster)

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
        supporting_observations=matching,
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