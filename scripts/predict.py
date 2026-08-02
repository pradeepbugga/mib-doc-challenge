"""
Submission entry point: a directory of PDF packets in, predictions.jsonl out.

Usage:
    python3 -m scripts.predict <input_pdf_dir> <output_predictions_path>

Runs fully offline. A packet that fails to process still emits a conservative
NEEDS_REVIEW row rather than being dropped: the scorer pays 1-2 points for a
cautious decision but applies a missing-case penalty for an absent one.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from core.pipeline.packet_pipeline import (
    PLACEHOLDER_ARRIVAL_DATE,
    PLACEHOLDER_SPONSOR_ID,
    predict_packet,
)
from core.pipeline.case_assignment import normalize_case_id_candidate

# Per-packet wall-clock ceiling. The scoring contract allows 6 seconds per PDF
# on average; a single pathological packet must not consume that budget for the
# whole run.
PACKET_TIMEOUT_SECONDS = 55


def fallback_prediction(case_id: str) -> dict:
    """Return a conservative, schema-valid row for a packet that failed."""
    return {
        "case_id": case_id,
        "applicant_name": "",
        "species_code": "",
        "home_world": "",
        "visa_class": "",
        "sponsor_id": PLACEHOLDER_SPONSOR_ID,
        "arrival_date": PLACEHOLDER_ARRIVAL_DATE,
        "declared_purpose": "",
        "risk_flags": "none",
        "fee_status": "unknown",
        "adjudication": "NEEDS_REVIEW",
        "confidence": 0.4,
    }


class PacketTimeout(Exception):
    """Raised when a single packet exceeds its wall-clock budget."""


def _raise_timeout(signum, frame):
    raise PacketTimeout


def predict_one(pdf_path_text: str) -> tuple[dict, str | None]:
    """
    Process one packet, returning its prediction and any error text.

    Runs inside a worker process, so it must not raise.
    """
    pdf_path = Path(pdf_path_text)
    case_id = normalize_case_id_candidate(pdf_path.stem) or pdf_path.stem

    previous_handler = None
    has_alarm = hasattr(signal, "SIGALRM")

    if has_alarm:
        previous_handler = signal.signal(signal.SIGALRM, _raise_timeout)
        signal.alarm(PACKET_TIMEOUT_SECONDS)

    try:
        return predict_packet(pdf_path), None
    except PacketTimeout:
        return fallback_prediction(case_id), f"{case_id}: timed out"
    except Exception:
        return (
            fallback_prediction(case_id),
            f"{case_id}: {traceback.format_exc(limit=3)}",
        )
    finally:
        if has_alarm:
            signal.alarm(0)

            if previous_handler is not None:
                signal.signal(signal.SIGALRM, previous_handler)


def find_pdfs(input_dir: Path) -> list[Path]:
    """Return every PDF in the input directory, in stable order."""
    return sorted(
        path
        for path in input_dir.iterdir()
        if path.is_file() and path.suffix.lower() == ".pdf"
    )


def default_worker_count() -> int:
    """Pick a worker count that fits the scoring container's 4 vCPUs."""
    return max(1, min(4, os.cpu_count() or 1))


def run(input_dir: Path, output_path: Path, workers: int, quiet: bool) -> int:
    pdf_paths = find_pdfs(input_dir)

    if not pdf_paths:
        print(f"No PDFs found in {input_dir}", file=sys.stderr)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    predictions: dict[str, dict] = {}
    errors: list[str] = []

    if workers == 1:
        for index, pdf_path in enumerate(pdf_paths, start=1):
            prediction, error = predict_one(str(pdf_path))
            predictions[prediction["case_id"]] = prediction

            if error:
                errors.append(error)

            if not quiet and index % 100 == 0:
                print(f"  {index}/{len(pdf_paths)}", file=sys.stderr)
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(predict_one, str(pdf_path)): pdf_path
                for pdf_path in pdf_paths
            }

            for index, future in enumerate(as_completed(futures), start=1):
                pdf_path = futures[future]

                try:
                    prediction, error = future.result()
                except Exception:
                    case_id = (
                        normalize_case_id_candidate(pdf_path.stem)
                        or pdf_path.stem
                    )
                    prediction = fallback_prediction(case_id)
                    error = f"{case_id}: worker crashed"

                predictions[prediction["case_id"]] = prediction

                if error:
                    errors.append(error)

                if not quiet and index % 100 == 0:
                    print(f"  {index}/{len(pdf_paths)}", file=sys.stderr)

    with output_path.open("w", encoding="utf-8") as handle:
        for case_id in sorted(predictions):
            handle.write(
                json.dumps(predictions[case_id], ensure_ascii=False) + "\n"
            )

    if not quiet:
        print(
            f"Wrote {len(predictions)} predictions to {output_path}",
            file=sys.stderr,
        )

        if errors:
            print(
                f"{len(errors)} packet(s) fell back to NEEDS_REVIEW",
                file=sys.stderr,
            )

            for error in errors[:5]:
                print(f"  {error.splitlines()[0]}", file=sys.stderr)

    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_dir", type=Path)
    parser.add_argument("output_path", type=Path)
    parser.add_argument(
        "--workers",
        type=int,
        default=default_worker_count(),
        help="Parallel worker processes (default: min(4, cpu count)).",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--quiet", action="store_true")

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.input_dir.is_dir():
        print(f"Input directory not found: {args.input_dir}", file=sys.stderr)
        raise SystemExit(1)

    if args.limit is not None:
        pdf_paths = find_pdfs(args.input_dir)[: args.limit]
        predictions = {}

        args.output_path.parent.mkdir(parents=True, exist_ok=True)

        for pdf_path in pdf_paths:
            prediction, _ = predict_one(str(pdf_path))
            predictions[prediction["case_id"]] = prediction

        with args.output_path.open("w", encoding="utf-8") as handle:
            for case_id in sorted(predictions):
                handle.write(
                    json.dumps(predictions[case_id], ensure_ascii=False) + "\n"
                )

        print(f"Wrote {len(predictions)} predictions", file=sys.stderr)
        raise SystemExit(0)

    raise SystemExit(
        run(args.input_dir, args.output_path, args.workers, args.quiet)
    )


if __name__ == "__main__":
    main()
