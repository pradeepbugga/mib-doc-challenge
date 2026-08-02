from __future__ import annotations

from .extractors.sponsor_attestation import extract_sponsor_attestation
from .extractors.registry_extract import extract_registry_extract
from .extractors.adjudicator_note import extract_adjudicator_note
from .extractors.intake_form import extract_intake_form
from .extractors.biometric_slip import extract_biometric_slip
from .extractors.fee_receipt import extract_fee_receipt
from .extractors.unknown import extract_unknown

EXTRACTORS = {
    "sponsor_attestation": extract_sponsor_attestation,
    "registry_extract": extract_registry_extract,
    "adjudicator_note": extract_adjudicator_note,
    "intake_form": extract_intake_form,
    "biometric_slip": extract_biometric_slip,
    "fee_receipt": extract_fee_receipt,
    "unknown": extract_unknown,
}