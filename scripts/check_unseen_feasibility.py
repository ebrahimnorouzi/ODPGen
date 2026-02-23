#!/usr/bin/env python3
import argparse
import csv
from datetime import date
from pathlib import Path


def parse_iso(d: str) -> date:
    y, m, day = [int(x) for x in d.split("-")]
    return date(y, m, day)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare ODP publication dates against model cutoffs.")
    parser.add_argument("--odps", default="data/selected_odps.csv", type=Path)
    parser.add_argument("--models", default="data/model_cutoffs.csv", type=Path)
    parser.add_argument("--out", default="results/unseen_feasibility.csv", type=Path)
    args = parser.parse_args()

    with args.odps.open(encoding="utf-8", newline="") as f:
        odps = list(csv.DictReader(f))
    with args.models.open(encoding="utf-8", newline="") as f:
        models = list(csv.DictReader(f))

    rows = []
    for odp in odps:
        pub = parse_iso(odp["publication_date"])
        for model in models:
            cutoff = parse_iso(model["training_cutoff_date"])
            if cutoff < pub:
                label = "plausible_unseen"
            elif cutoff == pub:
                label = "uncertain"
            else:
                label = "likely_seen"
            rows.append(
                {
                    "odp_id": odp["odp_id"],
                    "model_id": model["model_id"],
                    "publication_date": odp["publication_date"],
                    "training_cutoff_date": model["training_cutoff_date"],
                    "cutoff_confidence": model["cutoff_confidence"],
                    "feasibility_label": label,
                }
            )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
