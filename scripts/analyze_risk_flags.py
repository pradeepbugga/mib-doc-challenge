"""
EDA on risk_flags vs. other fields in the training labels.

The public field manual lists disqualifying and review-only risk flags but
says "multiple review-only flags may combine into a denial in edge cases"
without specifying which combinations. This script surfaces the flag/field
crosstabs needed to infer those combinations from data/train_labels.csv.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

DEFAULT_LABELS_PATH = Path("./data/train_labels.csv")

CROSSTAB_FIELDS = (
    "visa_class",
    "declared_purpose",
    "home_world",
    "species_code",
    "adjudication",
)


def explode_risk_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Return a long-form frame with one row per (case, risk_flag)."""
    flags = df["risk_flags"].str.split("|", regex=False)
    exploded = df.assign(risk_flags=flags).explode(
        "risk_flags", ignore_index=True
    )
    exploded["risk_flags"] = exploded["risk_flags"].str.strip()

    return exploded[exploded["risk_flags"] != "none"]


def print_value_counts(df: pd.DataFrame, column: str) -> None:
    print(f"\nDistribution of {column} values:")
    print(df[column].value_counts())


def plot_crosstab(
    exploded: pd.DataFrame,
    against: str,
    output_dir: Path,
) -> None:
    crosstab = pd.crosstab(exploded["risk_flags"], exploded[against])

    plt.figure(figsize=(10, 6))
    sns.heatmap(crosstab, annot=True, fmt="d", cmap="YlGnBu")
    plt.title(f"Risk Flags vs {against}")
    plt.xlabel(against)
    plt.ylabel("Risk Flags")
    plt.tight_layout()

    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_dir / f"risk_flags_vs_{against}.png")
    plt.close()


def analyze_risk_flag_denial_combinations(exploded: pd.DataFrame) -> None:
    """
    Show which multi-review-flag combinations co-occur with DENIED, to help
    infer the undocumented review_flag_denial_combinations rule.
    """
    review_only = {
        "identity_conflict",
        "sponsor_mismatch",
        "illegible_biometrics",
        "rescinded_denial",
    }

    grouped = exploded.groupby("case_id")["risk_flags"].apply(
        lambda flags: frozenset(f for f in flags if f in review_only)
    )
    multi_flag_cases = grouped[grouped.apply(len) >= 2]

    if multi_flag_cases.empty:
        print("\nNo cases with 2+ review-only risk flags found.")
        return

    combo_df = multi_flag_cases.rename("flag_combo").reset_index()
    labels = exploded.drop_duplicates("case_id")[["case_id", "adjudication"]]
    combo_df = combo_df.merge(labels, on="case_id")

    print("\nMulti review-flag combinations vs adjudication outcome:")
    print(
        combo_df.groupby("flag_combo")["adjudication"]
        .value_counts()
        .unstack(fill_value=0)
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--labels",
        type=Path,
        default=DEFAULT_LABELS_PATH,
        help="Path to train_labels.csv.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./outputs/risk_flag_analysis"),
        help="Directory to write crosstab heatmaps to.",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip writing heatmap PNGs; only print value counts/tables.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.labels)

    for column in ("species_code", "declared_purpose", "home_world", "adjudication"):
        print_value_counts(df, column)

    exploded = explode_risk_flags(df)
    print("\nDistribution of individual risk_flags values:")
    print(exploded["risk_flags"].value_counts())

    analyze_risk_flag_denial_combinations(exploded)

    if not args.no_plots:
        for field in CROSSTAB_FIELDS:
            plot_crosstab(exploded, field, args.output_dir)
        print(f"\nWrote crosstab heatmaps to {args.output_dir}")


if __name__ == "__main__":
    main()
