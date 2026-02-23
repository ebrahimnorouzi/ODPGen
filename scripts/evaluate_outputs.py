#!/usr/bin/env python3
import argparse
import csv
import json
from pathlib import Path

from eval.faithfulness import evaluate_faithfulness
from eval.formal_quality import evaluate_formal_quality
from eval.similarity import evaluate_structure_similarity


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate generated ODP outputs.")
    parser.add_argument("--outputs-dir", default="outputs", type=Path)
    parser.add_argument("--data", default="data/scenarios/pattern_scenarios.json", type=Path)
    parser.add_argument("--results-dir", default="results", type=Path)
    args = parser.parse_args()

    scenarios = {item["scenario_id"]: item for item in json.loads(args.data.read_text(encoding="utf-8"))}
    args.results_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    for ontology_path in args.outputs_dir.glob("*/*/*/ontology.ttl"):
        model, config, scenario_id = ontology_path.parts[-4], ontology_path.parts[-3], ontology_path.parts[-2]
        if scenario_id not in scenarios:
            continue

        ontology_text = ontology_path.read_text(encoding="utf-8")
        formal = evaluate_formal_quality(ontology_path)
        faith = evaluate_faithfulness(
            scenarios[scenario_id]["scenario_text"],
            scenarios[scenario_id].get("cq_list", []),
            ontology_text,
        )
        sim = evaluate_structure_similarity(ontology_text)

        metrics = {
            "model": model,
            "config": config,
            "scenario_id": scenario_id,
            **formal,
            **faith,
            **sim,
        }

        result_file = args.results_dir / model / config / f"{scenario_id}.json"
        result_file.parent.mkdir(parents=True, exist_ok=True)
        result_file.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        summary_rows.append(metrics)

    summary_path = args.results_dir / "summary.csv"
    if summary_rows:
        fieldnames = list(summary_rows[0].keys())
        with summary_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(summary_rows)


if __name__ == "__main__":
    main()
