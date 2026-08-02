from pprint import pprint

from core.adjudication.models import FieldObservation
from core.adjudication.normalizers import normalize_observations
from core.adjudication.corroborator import corroborate_packet


def main():

    observations = [
        FieldObservation(
            field="sponsor_id",
            raw_value="SPN-68 18",
            document_type="intake_form",
            page_number=1,
        ),
        FieldObservation(
            field="sponsor_id",
            raw_value="SPN-6818",
            document_type="sponsor_attestation",
            page_number=4,
        ),
        FieldObservation(
            field="fee_status",
            raw_value="PAID",
            document_type="intake_form",
            page_number=1,
        ),
        FieldObservation(
            field="fee_status",
            raw_value="paid",
            document_type="receipt",
            page_number=2,
        ),
        FieldObservation(
            field="visa_class",
            raw_value="xw1",
            document_type="intake_form",
            page_number=1,
        ),
    ]

    observations = normalize_observations(observations)

    packet = corroborate_packet(observations)

    print("\nResolved Fields\n")

    for field, resolved in packet.fields.items():
        print(f"{field}:")
        pprint(resolved)
        print()


if __name__ == "__main__":
    main()