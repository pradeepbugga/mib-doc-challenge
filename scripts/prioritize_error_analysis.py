from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


DEFAULT_INPUT_PATH = Path(
    "./outputs/extraction_error_analysis.csv"
)

DEFAULT_REVIEW_PATH = Path(
    "./outputs/error_review_queue.csv"
)

DEFAULT_CASE_SUMMARY_PATH = Path(
    "./outputs/error_case_summary.csv"
)

DEFAULT_ROOT_CAUSE_SUMMARY_PATH = Path(
    "./outputs/error_root_cause_summary.csv"
)


# These match the public evaluator's extraction weights.
FIELD_WEIGHTS = {
    "applicant_name": 5,
    "species_code": 6,
    "home_world": 5,
    "visa_class": 5,
    "sponsor_id": 5,
    "arrival_date": 4,
    "declared_purpose": 3,
    "risk_flags": 8,
    "fee_status": 4,
    "case_id": 0,
    "__pipeline__": 20,
}


# Use these exact labels during manual review.
ROOT_CAUSE_OPTIONS = {
    "",
    "ocr_characters",
    "ocr_layout",
    "regex",
    "classification",
    "orientation",
    "case_assignment",
    "corroboration",
    "evidence_precedence",
    "policy",
    "visual_evidence",
    "unrecoverable",
    "possible_label_issue",
    "pipeline_error",
    "other",
}


RECOVERABILITY_OPTIONS = {
    "",
    "yes",
    "probably",
    "no",
}


FIX_PRIORITY_OPTIONS = {
    "",
    "now",
    "later",
    "do_not_fix",
}


def parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value

    return str(value).strip().casefold() in {
        "true",
        "1",
        "yes",
    }


def parse_int(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def field_weight(field_name: str) -> int:
    return FIELD_WEIGHTS.get(field_name, 1)


def calculate_priority_score(row: dict[str, str]) -> float:
    """
    Calculate a review-priority score for one discrepancy.

    Higher scores are reviewed first.
    """
    field_name = row.get("field", "")
    score = float(field_weight(field_name))

    error_category = row.get(
        "error_category",
        "",
    )

    resolution_status = row.get(
        "resolution_status",
        "",
    )

    unassigned_pages = row.get(
        "unassigned_pages",
        "",
    ).strip()

    unassigned_observation_count = parse_int(
        row.get("unassigned_observation_count")
    )

    # Missing/absent observations are currently the largest
    # architectural bottleneck.
    category_bonus = {
        "pipeline_error": 10.0,
        "field_not_extracted": 5.0,
        "missing_value": 4.0,
        "normalization_returned_none": 3.5,
        "corroboration_conflict": 3.0,
        "incorrect_value": 2.0,
        "unexpected_prediction": 1.0,
    }

    score += category_bonus.get(
        error_category,
        0.0,
    )

    if resolution_status == "conflicting":
        score += 2.0

    if unassigned_pages:
        score += 3.0

    if unassigned_observation_count > 0:
        score += min(
            unassigned_observation_count,
            5,
        ) * 0.5

    # Risk flags influence both extraction and adjudication.
    if field_name == "risk_flags":
        score += 4.0

    # Applicant conflicts were a major observed failure mode.
    if (
        field_name == "applicant_name"
        and resolution_status == "conflicting"
    ):
        score += 2.0

    return round(score, 2)


def load_rows(
    input_path: Path,
) -> list[dict[str, str]]:
    with input_path.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        return list(csv.DictReader(file))


def prepare_review_rows(
    rows: list[dict[str, str]],
) -> list[dict[str, Any]]:
    review_rows: list[dict[str, Any]] = []

    for row in rows:
        review_row = dict(row)

        review_row["field_weight"] = field_weight(
            row.get("field", "")
        )

        review_row["priority_score"] = (
            calculate_priority_score(row)
        )

        # Preserve any existing manual annotations.
        review_row["root_cause"] = (
            row.get("root_cause", "") or ""
        )

        review_row["recoverable"] = (
            row.get("recoverable", "") or ""
        )

        review_row["fix_priority"] = (
            row.get("fix_priority", "")
            or row.get("fix_now", "")
            or ""
        )

        review_row["visual_inspection_needed"] = (
            row.get(
                "visual_inspection_needed",
                "",
            )
            or ""
        )

        review_row["reviewed"] = (
            row.get("reviewed", "") or ""
        )

        review_row["notes"] = (
            row.get("notes", "") or ""
        )

        review_rows.append(review_row)

    review_rows.sort(
        key=lambda row: (
            -float(row["priority_score"]),
            row.get("case_id", ""),
            row.get("field", ""),
        )
    )

    for rank, row in enumerate(
        review_rows,
        start=1,
    ):
        row["review_rank"] = rank

    return review_rows


def build_case_summary(
    rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    by_case: dict[
        str,
        list[dict[str, Any]],
    ] = defaultdict(list)

    for row in rows:
        by_case[row["case_id"]].append(row)

    summary_rows: list[dict[str, Any]] = []

    for case_id, case_rows in by_case.items():
        fields = sorted(
            {
                row.get("field", "")
                for row in case_rows
                if row.get("field")
            }
        )

        error_categories = sorted(
            {
                row.get("error_category", "")
                for row in case_rows
                if row.get("error_category")
            }
        )

        root_causes = sorted(
            {
                row.get("root_cause", "")
                for row in case_rows
                if row.get("root_cause")
            }
        )

        unassigned_pages = sorted(
            {
                page
                for row in case_rows
                for page in (
                    row.get(
                        "unassigned_pages",
                        "",
                    ).split("|")
                )
                if page.strip()
            }
        )

        weighted_error_score = sum(
            field_weight(
                row.get("field", "")
            )
            for row in case_rows
        )

        review_priority_score = sum(
            float(
                row.get(
                    "priority_score",
                    0,
                )
            )
            for row in case_rows
        )

        reviewed_count = sum(
            parse_bool(
                row.get("reviewed")
            )
            for row in case_rows
        )

        summary_rows.append(
            {
                "case_id": case_id,
                "pdf_path": case_rows[0].get(
                    "pdf_path",
                    "",
                ),
                "discrepancy_count": len(
                    case_rows
                ),
                "weighted_error_score": (
                    weighted_error_score
                ),
                "review_priority_score": round(
                    review_priority_score,
                    2,
                ),
                "failed_fields": "|".join(
                    fields
                ),
                "error_categories": "|".join(
                    error_categories
                ),
                "root_causes": "|".join(
                    root_causes
                ),
                "unassigned_pages": "|".join(
                    unassigned_pages
                ),
                "reviewed_discrepancies": (
                    reviewed_count
                ),
                "all_discrepancies_reviewed": (
                    reviewed_count
                    == len(case_rows)
                ),
            }
        )

    summary_rows.sort(
        key=lambda row: (
            -float(
                row[
                    "review_priority_score"
                ]
            ),
            -int(
                row[
                    "weighted_error_score"
                ]
            ),
            row["case_id"],
        )
    )

    return summary_rows


def build_root_cause_summary(
    rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    reviewed_rows = [
        row
        for row in rows
        if str(
            row.get("root_cause", "")
        ).strip()
    ]

    counts = Counter(
        row["root_cause"]
        for row in reviewed_rows
    )

    weighted_counts = defaultdict(int)

    for row in reviewed_rows:
        weighted_counts[
            row["root_cause"]
        ] += field_weight(
            row.get("field", "")
        )

    total_reviewed = len(reviewed_rows)
    total_weighted = sum(
        weighted_counts.values()
    )

    summary: list[dict[str, Any]] = []

    for root_cause, count in counts.most_common():
        summary.append(
            {
                "root_cause": root_cause,
                "discrepancy_count": count,
                "discrepancy_percent": round(
                    (
                        count
                        / total_reviewed
                        * 100
                    )
                    if total_reviewed
                    else 0.0,
                    2,
                ),
                "weighted_error_total": (
                    weighted_counts[root_cause]
                ),
                "weighted_error_percent": round(
                    (
                        weighted_counts[
                            root_cause
                        ]
                        / total_weighted
                        * 100
                    )
                    if total_weighted
                    else 0.0,
                    2,
                ),
            }
        )

    return summary


def write_csv(
    path: Path,
    rows: list[dict[str, Any]],
    fieldnames: list[str],
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
            extrasaction="ignore",
        )

        writer.writeheader()
        writer.writerows(rows)


def validate_manual_labels(
    rows: list[dict[str, Any]],
) -> None:
    for row in rows:
        root_cause = str(
            row.get(
                "root_cause",
                "",
            )
        ).strip()

        recoverable = str(
            row.get(
                "recoverable",
                "",
            )
        ).strip()

        fix_priority = str(
            row.get(
                "fix_priority",
                "",
            )
        ).strip()

        if root_cause not in ROOT_CAUSE_OPTIONS:
            raise ValueError(
                f"Invalid root_cause "
                f"{root_cause!r} for "
                f"{row.get('case_id')} "
                f"{row.get('field')}"
            )

        if (
            recoverable
            not in RECOVERABILITY_OPTIONS
        ):
            raise ValueError(
                f"Invalid recoverable value "
                f"{recoverable!r} for "
                f"{row.get('case_id')} "
                f"{row.get('field')}"
            )

        if (
            fix_priority
            not in FIX_PRIORITY_OPTIONS
        ):
            raise ValueError(
                f"Invalid fix_priority "
                f"{fix_priority!r} for "
                f"{row.get('case_id')} "
                f"{row.get('field')}"
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prioritize MIB extraction errors "
            "and generate a manual review queue."
        )
    )

    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT_PATH,
    )

    parser.add_argument(
        "--review-output",
        type=Path,
        default=DEFAULT_REVIEW_PATH,
    )

    parser.add_argument(
        "--case-summary-output",
        type=Path,
        default=DEFAULT_CASE_SUMMARY_PATH,
    )

    parser.add_argument(
        "--root-cause-output",
        type=Path,
        default=(
            DEFAULT_ROOT_CAUSE_SUMMARY_PATH
        ),
    )

    parser.add_argument(
        "--top",
        type=int,
        default=None,
        help=(
            "Optionally limit the review queue "
            "to the top N discrepancies."
        ),
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    source_rows = load_rows(
        args.input
    )

    review_rows = prepare_review_rows(
        source_rows
    )

    validate_manual_labels(
        review_rows
    )

    if args.top is not None:
        review_rows = review_rows[
            : args.top
        ]

    case_summary_rows = build_case_summary(
        review_rows
    )

    root_cause_summary_rows = (
        build_root_cause_summary(
            review_rows
        )
    )

    review_fieldnames = [
        "review_rank",
        "priority_score",
        "field_weight",
        "case_id",
        "pdf_path",
        "field",
        "expected_value",
        "predicted_value",
        "resolution_status",
        "resolution_method",
        "error_category",
        "unassigned_pages",
        "unassigned_observation_count",
        "all_observations",
        "supporting_observations",
        "conflicting_observations",
        "root_cause",
        "recoverable",
        "fix_priority",
        "visual_inspection_needed",
        "reviewed",
        "notes",
    ]

    case_summary_fieldnames = [
        "case_id",
        "pdf_path",
        "discrepancy_count",
        "weighted_error_score",
        "review_priority_score",
        "failed_fields",
        "error_categories",
        "root_causes",
        "unassigned_pages",
        "reviewed_discrepancies",
        "all_discrepancies_reviewed",
    ]

    root_cause_fieldnames = [
        "root_cause",
        "discrepancy_count",
        "discrepancy_percent",
        "weighted_error_total",
        "weighted_error_percent",
    ]

    write_csv(
        args.review_output,
        review_rows,
        review_fieldnames,
    )

    write_csv(
        args.case_summary_output,
        case_summary_rows,
        case_summary_fieldnames,
    )

    write_csv(
        args.root_cause_output,
        root_cause_summary_rows,
        root_cause_fieldnames,
    )

    print(
        f"Input discrepancies: "
        f"{len(source_rows)}"
    )

    print(
        f"Review rows written: "
        f"{len(review_rows)}"
    )

    print(
        f"Cases represented: "
        f"{len(case_summary_rows)}"
    )

    print(
        f"Saved review queue: "
        f"{args.review_output}"
    )

    print(
        f"Saved case summary: "
        f"{args.case_summary_output}"
    )

    print(
        f"Saved root-cause summary: "
        f"{args.root_cause_output}"
    )


if __name__ == "__main__":
    main()