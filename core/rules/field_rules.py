from __future__ import annotations

from typing import Final


# -------------------------------------------------------------------
# Final output schema
# -------------------------------------------------------------------

OUTPUT_FIELDS: Final[tuple[str, ...]] = (
    "case_id",
    "applicant_name",
    "species_code",
    "home_world",
    "visa_class",
    "sponsor_id",
    "arrival_date",
    "declared_purpose",
    "risk_flags",
    "fee_status",
    "adjudication",
)


# -------------------------------------------------------------------
# Trusted evidence precedence
#
# Lower number means more trusted.
# -------------------------------------------------------------------

EVIDENCE_PRIORITY: Final[dict[str, int]] = {
    "adjudicator_stamp": 1,
    "signed_manual_note": 1,
    "intake_form": 2,
    "biometric_slip": 3,
    "sponsor_attestation": 4,
    "registry_extract": 5,
    "machine_readable_text": 6,
}


UNTRUSTED_EVIDENCE_TYPES: Final[set[str]] = {
    "hidden_white_text",
    "text_outside_page_crop",
    "fake_answer_key",
    "barcode_instruction",
}


# -------------------------------------------------------------------
# Field registry
# -------------------------------------------------------------------

FIELD_RULES: Final[dict[str, dict]] = {
    "case_id": {
        "type": "string",
        "required": True,
        "pattern": r"^MIB-\d{6}$",
        "normalizer": "normalize_case_id",
        "notes": (
            "Use the applicant associated with the active case_id when "
            "a packet contains records for multiple applicants."
        ),
    },

    "applicant_name": {
        "type": "string",
        "required": True,
        "normalizer": "normalize_name",
        "conflict_policy": "use_highest_priority_trusted_evidence",
    },

    "species_code": {
        "type": "string",
        "required": True,
        "normalizer": "normalize_species_code",
        "conflict_policy": "use_highest_priority_trusted_evidence",
    },

    "home_world": {
        "type": "string",
        "required": True,
        "normalizer": "normalize_home_world",
        "conflict_policy": "use_highest_priority_trusted_evidence",
    },

    "visa_class": {
        "type": "category",
        "required": True,
        "allowed_values": {
            "XW-1",
            "XW-2",
            "DIP-1",
            "MED-3",
            "TRANSIT-7",
        },
        "normalizer": "normalize_visa_class",
        "conflict_policy": "use_highest_priority_trusted_evidence",
    },

    "sponsor_id": {
        "type": "string",
        "required": False,
        "pattern": r"^SPN-\d{4}$",
        "normalizer": "normalize_sponsor_id",
        "conflict_policy": "use_highest_priority_trusted_evidence",
        "notes": "Not required for DIP-1 applicants.",
    },

    "arrival_date": {
        "type": "date",
        "required": True,
        "output_format": "YYYY-MM-DD",
        "normalizer": "normalize_date",
        "conflict_policy": "use_highest_priority_trusted_evidence",
        "missing_result": "NEEDS_REVIEW",
        "hidden_text_only_result": "NEEDS_REVIEW",
    },

    "declared_purpose": {
        "type": "string",
        "required": True,
        "normalizer": "normalize_declared_purpose",
        "conflict_policy": "use_highest_priority_trusted_evidence",
    },

    "risk_flags": {
        "type": "list[string]",
        "required": True,
        "normalizer": "normalize_risk_flags",
        "empty_serialization": "none",
        "multiple_value_separator": "|",
    },

    "fee_status": {
        "type": "category",
        "required": True,
        "allowed_values": {
            "paid",
            "waived",
            "unpaid",
            "unknown",
        },
        "normalizer": "normalize_fee_status",
        "conflict_policy": "use_highest_priority_trusted_evidence",
    },

    "adjudication": {
        "type": "category",
        "required": True,
        "allowed_values": {
            "APPROVED",
            "DENIED",
            "NEEDS_REVIEW",
        },
        "derived": True,
        "notes": "Calculated from resolved case facts and policy rules.",
    },
}

# -------------------------------------------------------------------
# Visa class rules
# -------------------------------------------------------------------


VISA_RULES: Final[dict[str, dict]] = {
    "XW-1": {
        "description": "short-term technical work",
        "maximum_duration_days": 30,
        "fee_waiver_allowed_by_class": False,
    },

    "XW-2": {
        "description": "extended technical work",
        "maximum_duration_days": 180,
        "fee_waiver_allowed_by_class": False,
    },

    "DIP-1": {
        "description": "diplomatic mission",
        "maximum_duration_days": None,
        "sponsor_required": False,
        "fee_waiver_allowed_by_class": True,
        "diplomatic_stale_date_exception": True,
    },

    "MED-3": {
        "description": "medical or biological consultation",
        "maximum_duration_days": None,
        "requires_clean_biohazard_check": True,
        "fee_waiver_allowed_by_class": False,
    },

    "TRANSIT-7": {
        "description": "transit only",
        "maximum_duration_days": None,
        "work_authorization_normally_denied": True,
        "fee_waiver_allowed_by_class": False,
    },
}


# -------------------------------------------------------------------
# Sponsor rules
# -------------------------------------------------------------------

SPONSOR_RULES: Final[dict[str, object]] = {
    "pattern": r"^SPN-\d{4}$",
    "required_unless_visa_class": {"DIP-1"},
    "known_revoked_sponsors": {
        "SPN-0007",
        "SPN-0139",
        "SPN-4040",
    },
    "unknown_revoked_sponsors_may_exist": True,
}

# -------------------------------------------------------------------
# Fee rules
# -------------------------------------------------------------------


FEE_RULES: Final[dict[str, dict]] = {
    "paid": {
        "result": "acceptable",
    },

    "waived": {
        "result": "acceptable_if",
        "conditions": {
            "visa_class_is_DIP-1",
            "visible_hardship_waiver",
        },
    },

    "unpaid": {
        "result": "DENIED",
        "exception": "visible_valid_waiver",
    },

    "unknown": {
        "result": "NEEDS_REVIEW",
    },
}

# -------------------------------------------------------------------
# Risk rules
# -------------------------------------------------------------------

DISQUALIFYING_RISK_FLAGS: Final[set[str]] = {
    "memory_tampering",
    "planetary_embargo",
    "active_warrant",
    "biohazard_red",
}


REVIEW_ONLY_RISK_FLAGS: Final[set[str]] = {
    "identity_conflict",
    "sponsor_mismatch",
    "illegible_biometrics",
    "rescinded_denial",
}


RISK_RULES: Final[dict[str, object]] = {
    "disqualifying": DISQUALIFYING_RISK_FLAGS,
    "review_only": REVIEW_ONLY_RISK_FLAGS,

    # The manual confirms this possibility but does not provide the
    # exact combinations. These must be inferred from training data.
    "multiple_review_flags_may_cause_denial": True,
    "review_flag_denial_combinations": set(),
}


# -------------------------------------------------------------------
# Date rules
# -------------------------------------------------------------------

DATE_RULES: Final[dict[str, object]] = {
    "stale_after_days": 180,

    "stale_exception": {
        "visa_class": "DIP-1",
        "requires_valid_diplomatic_note": True,
    },

    "missing_arrival_date_result": "NEEDS_REVIEW",
    "hidden_text_only_arrival_date_result": "NEEDS_REVIEW",
}

# -------------------------------------------------------------------
# Trusted document trap rules
# -------------------------------------------------------------------


DOCUMENT_TRAP_RULES: Final[dict[str, dict]] = {
    "sample_denial_watermark": {
        "trusted": False,
        "creates_denial": False,
    },

    "crossed_out_denial_with_later_signed_approval": {
        "automatic_denial": False,
        "resolution": "evaluate_later_signed_note",
    },

    "barcode_registry_metadata": {
        "metadata_may_be_used": True,
        "embedded_instructions_trusted": False,
    },

    "multiple_applicants": {
        "resolution": "use_applicant_linked_to_active_case_id",
    },
}

# -------------------------------------------------------------------
# Rule registry
# -------------------------------------------------------------------

RULE_REGISTRY: Final[dict[str, object]] = {
    "output_fields": OUTPUT_FIELDS,
    "fields": FIELD_RULES,
    "evidence_priority": EVIDENCE_PRIORITY,
    "untrusted_evidence_types": UNTRUSTED_EVIDENCE_TYPES,
    "visa_rules": VISA_RULES,
    "sponsor_rules": SPONSOR_RULES,
    "fee_rules": FEE_RULES,
    "risk_rules": RISK_RULES,
    "date_rules": DATE_RULES,
    "document_traps": DOCUMENT_TRAP_RULES,
}