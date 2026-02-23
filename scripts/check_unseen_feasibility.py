#!/usr/bin/env python3
import argparse
import csv
from datetime import date
from pathlib import Path


def parse_iso(d: str | None) -> date | None:
    if d is None:
        return None
    text = d.strip()
    if not text:
        return None
    try:
        y, m, day = [int(x) for x in text.split("-")]
        return date(y, m, day)
    except (TypeError, ValueError):
        return None


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
    skipped_odps = 0
    skipped_models = 0
    for odp in odps:
        pub = parse_iso(odp.get("publication_date"))
        if pub is None:
            skipped_odps += 1
            continue
        for model in models:
            cutoff = parse_iso(model.get("training_cutoff_date"))
            if cutoff is None:
                skipped_models += 1
                continue
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
        if not rows:
            raise SystemExit("No feasibility rows generated (check publication_date/training_cutoff_date values).")
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    if skipped_odps:
        print(f"Skipped ODP rows with missing/invalid publication_date: {skipped_odps}")
    if skipped_models:
        print(f"Skipped model comparisons with missing/invalid training_cutoff_date: {skipped_models}")


if __name__ == "__main__":
    main()
