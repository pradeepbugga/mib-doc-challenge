from pathlib import Path
from scripts.test_extraction import test_extraction
from tqdm import tqdm
from core.extraction.missing_registry import EXPECTED_MISSING_MARKERS
from core.extraction.schema import DOCUMENT_REQUIRED_FIELDS, DOCUMENT_POSSIBLE_FIELDS

pdf_dir = Path("./data/train")


def find_expected_missing_marker(
    field_name: str,
    page_text: str,
) -> str | None:
    """
    Return the explicit missing-value marker found for a field.

    None means no recognized marker was found.
    """
    markers = EXPECTED_MISSING_MARKERS.get(field_name, set())
    normalized_text = page_text.upper()

    for marker in markers:
        if marker.upper() in normalized_text:
            return marker

    return None

for pdf_path in tqdm(sorted(pdf_dir.glob("*.pdf")), desc="Processing PDFs"):
    results = test_extraction(pdf_path)

    for page in results:

        #if page["text_source"] == "native_text":
        #    continue

        fields = page["extraction"]["fields"]

        document_type = page["classification"]["document_type"]
        required = DOCUMENT_REQUIRED_FIELDS.get(document_type, set())

        unexpected_missing = []
        expected_unavailable = {}

        for field_name in required:

            value = fields.get(field_name)

            if value is not None:
                continue

            marker = find_expected_missing_marker(
                field_name=field_name,
                page_text=page["page_text"],
            )

            if marker is not None:
                expected_unavailable[field_name] = marker
            else:
                unexpected_missing.append(field_name)

        if unexpected_missing:
            print("=" * 80)
            print(pdf_path.name)
            print(
                f"Page {page['page_number']} - Document Type: "
                f"({page['classification']['document_type']})"
            )
            print(f"Missing: {', '.join(unexpected_missing)}")
            print()
            print(page["page_text"][:500])
            print()