#!/usr/bin/env python3
"""Filter ODP records by publication window and create an experiment shortlist."""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from dataclasses import dataclass
from datetime import date
from pathlib import Path


TRUE_VALUES = {"1", "true", "yes", "y"}


@dataclass
class ODPRecord:
    odp_id: str
    title: str
    publication_date: date
    source_url: str
    has_scenario: bool
    has_cq: bool
    notes: str

    @property
    def experiment_score(self) -> int:
        return int(self.has_scenario) + int(self.has_cq)


def parse_bool(value: str) -> bool:
    return value.strip().lower() in TRUE_VALUES


def parse_record(row: dict[str, str]) -> ODPRecord:
    return ODPRecord(
        odp_id=row["odp_id"].strip(),
        title=row["title"].strip(),
        publication_date=date.fromisoformat(row["publication_date"].strip()),
        source_url=row["source_url"].strip(),
        has_scenario=parse_bool(row["has_scenario"]),
        has_cq=parse_bool(row["has_cq"]),
        notes=row.get("notes", "").strip(),
    )


def load_catalog(path: Path) -> list[ODPRecord]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {
            "odp_id",
            "title",
            "publication_date",
            "source_url",
            "has_scenario",
            "has_cq",
            "notes",
        }
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Catalog is missing required fields: {sorted(missing)}")
        return [parse_record(row) for row in reader]


def summarize(records: list[ODPRecord]) -> Counter:
    summary = Counter()
    for record in records:
        summary["total"] += 1
        summary["with_scenario"] += int(record.has_scenario)
        summary["without_scenario"] += int(not record.has_scenario)
        summary["with_cq"] += int(record.has_cq)
        summary["without_cq"] += int(not record.has_cq)
        summary["with_both"] += int(record.has_scenario and record.has_cq)
    return summary


def write_selection(path: Path, records: list[ODPRecord]) -> None:
    headers = [
        "odp_id",
        "title",
        "publication_date",
        "source_url",
        "has_scenario",
        "has_cq",
        "notes",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers)
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    "odp_id": record.odp_id,
                    "title": record.title,
                    "publication_date": record.publication_date.isoformat(),
                    "source_url": record.source_url,
                    "has_scenario": "yes" if record.has_scenario else "no",
                    "has_cq": "yes" if record.has_cq else "no",
                    "notes": record.notes,
                }
            )


def build_shortlist(filtered: list[ODPRecord], max_items: int) -> list[ODPRecord]:
    ranked = sorted(
        filtered,
        key=lambda row: (row.experiment_score, row.publication_date, row.odp_id),
        reverse=True,
    )
    return ranked[:max_items]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--catalog",
        type=Path,
        default=Path("data/odp_catalog.csv"),
        help="Path to canonical ODP registry CSV.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/selected_odps.csv"),
        help="Path to write shortlisted ODPs for experiments.",
    )
    parser.add_argument(
        "--since",
        default="2023-01-01",
        help="Inclusive lower bound publication date (YYYY-MM-DD).",
    )
    parser.add_argument(
        "--max-items",
        type=int,
        default=10,
        help="Maximum number of ODPs to include in shortlist.",
    )
    args = parser.parse_args()

    since = date.fromisoformat(args.since)
    catalog = load_catalog(args.catalog)
    filtered = [row for row in catalog if row.publication_date >= since]
    stats = summarize(filtered)

    shortlist = build_shortlist(filtered, args.max_items)
    write_selection(args.output, shortlist)

    print(f"Catalog records: {len(catalog)}")
    print(f"Records published since {since.isoformat()}: {stats['total']}")
    print(
        "Scenario coverage: "
        f"{stats['with_scenario']} with / {stats['without_scenario']} without"
    )
    print(f"CQ coverage: {stats['with_cq']} with / {stats['without_cq']} without")
    print(f"With both scenario and CQ: {stats['with_both']}")
    print(f"Shortlist written to {args.output} ({len(shortlist)} rows)")


if __name__ == "__main__":
    main()
