#!/usr/bin/env python3
import argparse
import csv
from collections import defaultdict
from pathlib import Path


def as_float(value: str):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate human evaluation scores.")
    parser.add_argument("--sheet", default="results/human_eval_sheet.csv", type=Path)
    parser.add_argument("--out", default="results/human_eval_summary.csv", type=Path)
    parser.add_argument("--report", default="results/human_eval_summary.md", type=Path)
    args = parser.parse_args()

    with args.sheet.open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))

    grouped = defaultdict(lambda: {"n": 0, "correctness": [], "completeness": [], "clarity": []})
    for r in rows:
        key = (r["internal_model"], r["internal_config"])
        grouped[key]["n"] += 1
        c1 = as_float(r.get("correctness_1_5", ""))
        c2 = as_float(r.get("completeness_1_5", ""))
        c3 = as_float(r.get("clarity_reusability_1_5", ""))
        if c1 is not None:
            grouped[key]["correctness"].append(c1)
        if c2 is not None:
            grouped[key]["completeness"].append(c2)
        if c3 is not None:
            grouped[key]["clarity"].append(c3)

    summary = []
    for (model, config), vals in grouped.items():
        def mean(items):
            return round(sum(items) / len(items), 3) if items else "NA"
        summary.append(
            {
                "model": model,
                "config": config,
                "items": vals["n"],
                "mean_correctness": mean(vals["correctness"]),
                "mean_completeness": mean(vals["completeness"]),
                "mean_clarity": mean(vals["clarity"]),
                "agreement_note": "not_computed",
            }
        )

    with args.out.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=summary[0].keys())
        writer.writeheader()
        writer.writerows(summary)

    lines = ["# Human Evaluation Summary", "", "| model | config | items | correctness | completeness | clarity |", "|---|---:|---:|---:|---:|---:|"]
    for row in summary:
        lines.append(
            f"| {row['model']} | {row['config']} | {row['items']} | {row['mean_correctness']} | {row['mean_completeness']} | {row['mean_clarity']} |"
        )
    lines.append("")
    lines.append("Inter-rater agreement: not computed in this bootstrap run.")
    args.report.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
