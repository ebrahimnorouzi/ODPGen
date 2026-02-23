#!/usr/bin/env python3
import argparse
import csv
from datetime import date
from pathlib import Path


def parse_iso(d: str) -> date:
    y, m, day = [int(x) for x in d.split("-")]
    return date(y, m, day)


def main() -> None:
    parser = argparse.ArgumentParser(description="Filter ODP catalog by publication window.")
    parser.add_argument("--catalog", default="data/odp_catalog.csv", type=Path)
    parser.add_argument("--after", default="2023-01-01")
    parser.add_argument("--out", default="data/selected_odps.csv", type=Path)
    args = parser.parse_args()

    threshold = parse_iso(args.after)
    with args.catalog.open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))

    selected = [r for r in rows if parse_iso(r["publication_date"]) > threshold]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(selected)

    with_scenarios = sum(1 for r in selected if r["has_scenario"].lower() == "yes")
    with_cq = sum(1 for r in selected if r["has_cq"].lower() == "yes")
    print(f"Selected: {len(selected)}")
    print(f"With scenarios: {with_scenarios}")
    print(f"With CQs: {with_cq}")


if __name__ == "__main__":
    main()
