#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a blinded human evaluation sheet.")
    parser.add_argument("--outputs-dir", default="outputs", type=Path)
    parser.add_argument("--out", default="results/human_eval_sheet.csv", type=Path)
    args = parser.parse_args()

    rows = []
    idx = 1
    for ontology_path in sorted(args.outputs_dir.glob("*/*/*/ontology.ttl")):
        model, config, scenario_id = ontology_path.parts[-4], ontology_path.parts[-3], ontology_path.parts[-2]
        rows.append(
            {
                "blind_id": f"B{idx:04d}",
                "scenario_id": scenario_id,
                "output_ref": str(ontology_path),
                "model_hidden": "masked",
                "config_hidden": "masked",
                "correctness_1_5": "",
                "completeness_1_5": "",
                "clarity_reusability_1_5": "",
                "comments": "",
                "internal_model": model,
                "internal_config": config,
            }
        )
        idx += 1

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
