DOCUMENT_TYPE_RULES = {
    "intake_form": {
        "headings": {
            "mib intake form",
            "temporary work authorization application",
            "applicant intake",
        },
        "field_labels": {
            "applicant name",
            "species code",
            "home world",
            "visa class",
            "arrival date",
            "declared purpose",
        },
    },

    "biometric_slip": {
        "headings": {
            "biometric verification",
            "biometric slip",
            "biometric scan slip"
        },
        "field_labels": {
            "biometric status",
            "identity match",
            "biohazard check",
        },
    },

    "sponsor_attestation": {
        "headings": {
            "sponsor attestation",
            "sponsor declaration",
        },
        "field_labels": {
            "sponsor id",
            "sponsor signature",
            "applicant sponsored",
        },
    },

    "registry_extract": {
        "headings": {
            "registry extract",
            "mib registry record",
        },
        "field_labels": {
            "registry status",
            "warrant status",
            "embargo status",
        },
    },

    "fee_receipt": {
        "headings": {
            "mib fee receipt",
            "payment receipt",
        },
        "field_labels": {
            "fee status",
            "amount paid",
            "payment reference",
        },
    },

    "adjudicator_note": {
        "headings": {
            "adjudicator note",
            "manual review note",
            "decision note",
        },
        "field_labels": {
            "approved",
            "denied",
            "signed",
            "adjudicator",
        },
    },
}


def classify_document_type(text: str) -> str:
    normalized_text = " ".join(text.lower().split())

    best_type = "unknown"
    best_score = 0

    for document_type, rules in DOCUMENT_TYPE_RULES.items():
        score = 0

        for heading in rules.get("headings", set()):
            if heading in normalized_text:
                score += 3

        for label in rules.get("field_labels", set()):
            if label in normalized_text:
                score += 1

        if score > best_score:
            best_score = score
            best_type = document_type

    return best_type if best_score > 0 else "unknown"


def main() -> None:
    import fitz

    doc = fitz.open("./data/train/MIB-000002.pdf")

    for page in doc:
        text = page.get_text("text")
        document_type = classify_document_type(text)
        print(f"Page {page.number + 1}: Classified as '{document_type}'")

if __name__ == "__main__":
    main()