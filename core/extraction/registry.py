from .extractors.sponsor_attestation import extract_sponsor_attestation
from .extractors.registry_extract import extract_registry_extract
from .extractors.adjudication_note import extract_adjudication_note
from .extractors.intake_form import extract_intake_form
from .extractors.biometric_slip import extract_biometric_slip
from .extractors.fee_receipt import extract_fee_receipt
from .extractors.unknown import extract_unknown

EXTRACTORS = {
    "sponsor_attestation": extract_sponsor_attestation,
    "registry_extract": extract_registry_extract,
    "adjudication_note": extract_adjudication_note,
    "intake_form": extract_intake_form,
    "biometric_slip": extract_biometric_slip,
    "fee_receipt": extract_fee_receipt,
    "unknown": extract_unknown,
}