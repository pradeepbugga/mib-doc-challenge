DOCUMENT_POSSIBLE_FIELDS = {
    "intake_form": {
        "case_id",
        "applicant",
        "species_code",
        "home_world",
        "visa_class",
        "sponsor_id",
        "arrival_date",
        "declared_purpose",
        "applicant_correction",
        "species_code_correction",
        "home_world_correction",
        "visa_class_correction",
        "sponsor_id_correction",
        "arrival_date_correction",
        "declared_purpose_correction",
    },

    "biometric_slip": {
        "case_id",
        "applicant",
        "species_match",
        "biometric_confidence",
        "observed_flags",
    },

    "registry_extract": {
        "applicant",
        "home_world",
        "species_code",
        "arrival_date",
        "registry_status",
    },

    "sponsor_attestation": {
        "case_id",
        "sponsor_id",
        "applicant",
        "purpose",
        "visa_class",
    },

    "fee_receipt": {
        "case_id",
        "fee_status",
        "amount",
        "waiver_code",
    },

    "adjudicator_note": {
        "case_id",
        "decision",
        "reason",
        "adjudicator",
    },
}

DOCUMENT_REQUIRED_FIELDS = {
    "intake_form": {
        "case_id",
        "applicant",
        "species_code",
        "home_world",
        "visa_class",
        "sponsor_id",
        "arrival_date",
        "declared_purpose",
    },

    "biometric_slip": {
        "case_id",
        "applicant",
        "species_match",
        "observed_flags",
    },

    "registry_extract": {
        "applicant",
        "home_world",
        "species_code",
        "arrival_date",
    },

    "sponsor_attestation": {
        "sponsor_id",
        "applicant",
        "purpose",
        "visa_class",
    },

    "fee_receipt": {
        "case_id",
        "fee_status",
    },

    "adjudicator_note": {
        "decision",
        "reason",
    },
}