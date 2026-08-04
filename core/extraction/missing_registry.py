from __future__ import annotations

# Keyed by the raw per-document field name (DOCUMENT_REQUIRED_FIELDS), not the
# canonical/aliased field name -- e.g. sponsor_attestation's purpose field is
# "purpose", not "declared_purpose", and both need the same marker registered
# under their own key.
#
# Markers are exact literal placeholder strings the synthetic generator writes
# in place of a genuinely-unrecoverable value (confirmed by cross-referencing
# every native-text occurrence of each marker against train_labels.csv: the
# marked field is never present elsewhere in the packet, text or image, even
# though ground truth is non-blank). On OCR-routed pages these markers get
# garbled past recognition (e.g. "[SPECIES WHITEOUT]" -> "[SPE"), so they only
# ever match on native-text pages -- that's expected, not a gap to fix.
EXPECTED_MISSING_MARKERS = {
    "arrival_date": {
        "UNREADABLE",
        "[DATE WASHED OUT]",
    },
    "sponsor_id": {
        "[SPONSOR ID BLANK]",
    },
    "species_code": {
        "[SPECIES WHITEOUT]",
    },
    "species_match": {
        "[SPECIES WHITEOUT]",
    },
    "home_world": {
        "[REGISTRY LOST]",
    },
    "visa_class": {
        "[VISA CLASS TORN]",
    },
    "declared_purpose": {
        "[PURPOSE ILLEGIBLE]",
    },
    "purpose": {
        "[PURPOSE ILLEGIBLE]",
    },
    "applicant": {
        "[NAME CUT OUT]",
    },
    "observed_flags": {
        "[RISK PANEL MISSING]",
    },
    "fee_status": {
        "[FEE STATUS OBSCURED]",
    },
}