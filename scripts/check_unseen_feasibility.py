#!/usr/bin/env python3
"""Estimate whether selected ODPs are plausibly unseen for models with known cutoffs."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path


@dataclass
class ModelCutoff:
    model_name: str
    cutoff_date: date
    notes: str


@dataclass
class SelectedODP:
    odp_id: str
    title: str
    publication_date: date


def load_model_cutoffs(path: Path) -> list[ModelCutoff]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return [
            ModelCutoff(
                model_name=row["model_name"].strip(),
                cutoff_date=date.fromisoformat(row["cutoff_date"].strip()),
                notes=row.get("notes", "").strip(),
            )
            for row in reader
        ]


def load_selected_odps(path: Path) -> list[SelectedODP]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return [
            SelectedODP(
                odp_id=row["odp_id"].strip(),
                title=row["title"].strip(),
                publication_date=date.fromisoformat(row["publication_date"].strip()),
            )
            for row in reader
        ]


def assign_label(publication_date: date, latest_cutoff: date, uncertainty_days: int) -> str:
    if publication_date <= latest_cutoff:
        return "likely_seen"
    if publication_date > latest_cutoff + timedelta(days=uncertainty_days):
        return "plausible_unseen"
    return "uncertain"


def write_report(
    report_path: Path,
    selected: list[SelectedODP],
    cutoffs: list[ModelCutoff],
    uncertainty_days: int,
) -> None:
    latest_cutoff = max(cutoff.cutoff_date for cutoff in cutoffs)
    with report_path.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = [
            "odp_id",
            "title",
            "publication_date",
            "latest_model_cutoff",
            "days_after_latest_cutoff",
            "confidence_label",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for odp in selected:
            days_after = (odp.publication_date - latest_cutoff).days
            writer.writerow(
                {
                    "odp_id": odp.odp_id,
                    "title": odp.title,
                    "publication_date": odp.publication_date.isoformat(),
                    "latest_model_cutoff": latest_cutoff.isoformat(),
                    "days_after_latest_cutoff": days_after,
                    "confidence_label": assign_label(
                        odp.publication_date,
                        latest_cutoff,
                        uncertainty_days,
                    ),
                }
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--selected",
        type=Path,
        default=Path("data/selected_odps.csv"),
        help="Shortlisted ODP CSV to evaluate.",
    )
    parser.add_argument(
        "--cutoffs",
        type=Path,
        default=Path("data/model_cutoffs.csv"),
        help="CSV with model_name and cutoff_date columns.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/unseen_feasibility.csv"),
        help="Output CSV report path.",
    )
    parser.add_argument(
        "--uncertainty-days",
        type=int,
        default=90,
        help="Days after the latest cutoff to keep labels as uncertain.",
    )
    args = parser.parse_args()

    cutoffs = load_model_cutoffs(args.cutoffs)
    selected = load_selected_odps(args.selected)
    if not cutoffs:
        raise ValueError("No model cutoffs found.")
    if not selected:
        raise ValueError("No selected ODPs found.")

    latest_cutoff = max(cutoff.cutoff_date for cutoff in cutoffs)
    earliest_cutoff = min(cutoff.cutoff_date for cutoff in cutoffs)

    write_report(args.output, selected, cutoffs, args.uncertainty_days)

    print(f"Loaded {len(selected)} selected ODPs and {len(cutoffs)} model cutoffs.")
    print(
        f"Cutoff range: {earliest_cutoff.isoformat()} to {latest_cutoff.isoformat()} "
        f"(latest used for confidence labels)."
    )
    print(
        "Confidence policy: likely_seen (<= latest cutoff), "
        "uncertain (within uncertainty window), plausible_unseen (after window)."
    )
    print(f"Feasibility report written to {args.output}")


if __name__ == "__main__":
    main()
