from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from scripts.test_pipeline import run_pipeline


DEFAULT_LABELS_PATH = Path("./data/train_labels.csv")
DEFAULT_PDF_DIR = Path("./data/train")
DEFAULT_OUTPUT_PATH = Path(
    "./outputs/extraction_error_analysis.csv"
)

COMPARE_FIELDS = (
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
)

# Adjudication is omitted for now because the policy engine
# has not yet been implemented.
IGNORED_FIELDS = {
    "adjudication",
}


def normalize_comparison_value(
    field_name: str,
    value: Any,
) -> str | None:
    """
    Normalize values only for evaluation comparison.

    This does not alter pipeline output.
    """
    if value is None:
        return None

    text = str(value).strip()

    if not text:
        return None

    if field_name == "risk_flags":
        flags = {
            flag.strip()
            for flag in text.split("|")
            if flag.strip()
        }

        if not flags:
            return None

        return "|".join(sorted(flags))

    return text


def values_match(
    field_name: str,
    expected: Any,
    predicted: Any,
) -> bool:
    """
    Compare expected and predicted values.

    Risk-flag ordering is ignored.
    """
    expected_normalized = normalize_comparison_value(
        field_name,
        expected,
    )
    predicted_normalized = normalize_comparison_value(
        field_name,
        predicted,
    )

    return expected_normalized == predicted_normalized


def get_resolved_field(
    packet: Any,
    field_name: str,
) -> Any | None:
    """Return a ResolvedField from a packet when present."""
    if packet is None:
        return None

    return packet.fields.get(field_name)


def serialize_observations(
    observations: list[Any],
) -> str:
    """
    Serialize observations compactly for one CSV cell.
    """
    records = []

    for observation in observations:
        records.append(
            {
                "page": observation.page_number,
                "document_type": observation.document_type,
                "text_source": observation.text_source,
                "raw_value": observation.raw_value,
                "normalized_value": observation.normalized_value,
            }
        )

    return json.dumps(
        records,
        ensure_ascii=False,
    )


def infer_error_category(
    *,
    expected_value: str | None,
    predicted_value: str | None,
    resolved_field: Any | None,
) -> str:
    """
    Assign a preliminary error bucket.

    These categories are intended to accelerate manual review.
    """
    if resolved_field is None:
        return "field_not_extracted"

    status = resolved_field.status

    if status == "missing":
        return "missing_value"

    if status == "conflicting":
        return "corroboration_conflict"

    if predicted_value is None:
        return "normalization_returned_none"

    if expected_value is None:
        return "unexpected_prediction"

    return "incorrect_value"


def load_labels(
    csv_path: Path,
) -> list[dict[str, str]]:
    """Load training labels from CSV."""
    with csv_path.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        return list(csv.DictReader(file))


def locate_pdf(
    pdf_dir: Path,
    case_id: str,
) -> Path | None:
    """
    Locate a training PDF for a case ID.

    Supports normal names and names with suffixes such as:
        MIB-000013(1).pdf
    """
    exact_path = pdf_dir / f"{case_id}.pdf"

    if exact_path.is_file():
        return exact_path

    candidates = sorted(
        pdf_dir.glob(f"{case_id}*.pdf")
    )

    return candidates[0] if candidates else None


def evaluate_case(
    *,
    label_row: dict[str, str],
    pdf_path: Path,
) -> list[dict[str, Any]]:
    """
    Run one PDF and return discrepancy records.
    """
    case_id = label_row["case_id"]

    (
        final_cases,
        identity_result,
        unassigned_observations,
    ) = run_pipeline(pdf_path)

    packet = final_cases.get(case_id)

    discrepancy_rows: list[dict[str, Any]] = []

    for field_name in COMPARE_FIELDS:
        expected_value = normalize_comparison_value(
            field_name,
            label_row.get(field_name),
        )

        resolved_field = get_resolved_field(
            packet,
            field_name,
        )

        predicted_value = (
            normalize_comparison_value(
                field_name,
                resolved_field.resolved_value,
            )
            if resolved_field is not None
            else None
        )

        if values_match(
            field_name,
            expected_value,
            predicted_value,
        ):
            continue

        all_observations = (
            resolved_field.observations
            if resolved_field is not None
            else []
        )

        supporting_observations = (
            resolved_field.supporting_observations
            if resolved_field is not None
            else []
        )

        conflicting_observations = getattr(
            resolved_field,
            "conflicting_observations",
            [],
        ) if resolved_field is not None else []

        discrepancy_rows.append(
            {
                "case_id": case_id,
                "pdf_path": str(pdf_path),
                "field": field_name,
                "expected_value": expected_value,
                "predicted_value": predicted_value,
                "matches": False,
                "resolution_status": (
                    resolved_field.status
                    if resolved_field is not None
                    else "field_absent"
                ),
                "resolution_method": (
                    resolved_field.resolution_method
                    if resolved_field is not None
                    else None
                ),
                "error_category": infer_error_category(
                    expected_value=expected_value,
                    predicted_value=predicted_value,
                    resolved_field=resolved_field,
                ),
                "all_observations": serialize_observations(
                    all_observations
                ),
                "supporting_observations": serialize_observations(
                    supporting_observations
                ),
                "conflicting_observations": serialize_observations(
                    conflicting_observations
                ),
                "unassigned_pages": "|".join(
                    str(page_number)
                    for page_number
                    in identity_result.unassigned_pages
                ),
                "unassigned_observation_count": len(
                    unassigned_observations
                ),
                # Fill these in during manual error analysis.
                "root_cause": "",
                "fix_now": "",
                "notes": "",
            }
        )

    return discrepancy_rows


def write_csv(
    rows: list[dict[str, Any]],
    output_path: Path,
) -> None:
    """Write discrepancy rows to CSV."""
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fieldnames = [
        "case_id",
        "pdf_path",
        "field",
        "expected_value",
        "predicted_value",
        "matches",
        "resolution_status",
        "resolution_method",
        "error_category",
        "all_observations",
        "supporting_observations",
        "conflicting_observations",
        "unassigned_pages",
        "unassigned_observation_count",
        "root_cause",
        "fix_now",
        "notes",
    ]

    with output_path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare extracted MIB fields against training labels "
            "and write a field-level error-analysis CSV."
        )
    )

    parser.add_argument(
        "--labels",
        type=Path,
        default=DEFAULT_LABELS_PATH,
        help="Path to train_labels.csv.",
    )

    parser.add_argument(
        "--pdf-dir",
        type=Path,
        default=DEFAULT_PDF_DIR,
        help="Directory containing training PDFs.",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Destination error-analysis CSV.",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional maximum number of labeled cases to evaluate.",
    )

    parser.add_argument(
        "--case-id",
        type=str,
        default=None,
        help="Optional single case ID to evaluate.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    labels = load_labels(args.labels)

    if args.case_id is not None:
        labels = [
            row
            for row in labels
            if row["case_id"] == args.case_id
        ]

    if args.limit is not None:
        labels = labels[: args.limit]

    discrepancy_rows: list[dict[str, Any]] = []
    missing_pdfs: list[str] = []

    total_cases = len(labels)

    for index, label_row in enumerate(
        labels,
        start=1,
    ):
        case_id = label_row["case_id"]

        print(
            f"[{index}/{total_cases}] "
            f"Evaluating {case_id}"
        )

        pdf_path = locate_pdf(
            pdf_dir=args.pdf_dir,
            case_id=case_id,
        )

        if pdf_path is None:
            missing_pdfs.append(case_id)
            print(f"  PDF not found: {case_id}")
            continue

        try:
            case_discrepancies = evaluate_case(
                label_row=label_row,
                pdf_path=pdf_path,
            )
        except Exception as error:
            discrepancy_rows.append(
                {
                    "case_id": case_id,
                    "pdf_path": str(pdf_path),
                    "field": "__pipeline__",
                    "expected_value": None,
                    "predicted_value": None,
                    "matches": False,
                    "resolution_status": "pipeline_error",
                    "resolution_method": None,
                    "error_category": "pipeline_error",
                    "all_observations": "",
                    "supporting_observations": "",
                    "conflicting_observations": "",
                    "unassigned_pages": "",
                    "unassigned_observation_count": 0,
                    "root_cause": "",
                    "fix_now": "",
                    "notes": repr(error),
                }
            )

            print(f"  Pipeline error: {error!r}")
            continue

        discrepancy_rows.extend(
            case_discrepancies
        )

        print(
            f"  Discrepancies: "
            f"{len(case_discrepancies)}"
        )

    write_csv(
        rows=discrepancy_rows,
        output_path=args.output,
    )

    print()
    print(f"Cases evaluated: {total_cases}")
    print(f"Discrepancy rows: {len(discrepancy_rows)}")
    print(f"Missing PDFs: {len(missing_pdfs)}")
    print(f"Saved: {args.output}")

    if missing_pdfs:
        print(
            "Missing case IDs:",
            ", ".join(missing_pdfs),
        )


if __name__ == "__main__":
    main()