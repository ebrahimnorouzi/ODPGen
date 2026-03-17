#!/usr/bin/env python3
"""
Evaluate all generated ODP outputs against ground truth ontologies.

Usage:
    python scripts/evaluate_outputs.py
    python scripts/evaluate_outputs.py --ground-truth-dir data/ground_truth
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import rdflib

from eval.faithfulness import evaluate_faithfulness
from eval.formal_quality import evaluate_formal_quality
from eval.rdf_utils import load_graph
from eval.similarity import evaluate_structure_similarity

# ---------------------------------------------------------------------------
# Ground truth loading
# ---------------------------------------------------------------------------

_RDF_EXTENSIONS = [".ttl", ".owl", ".rdf", ".n3", ".nt"]


def _load_ground_truth_graphs(
    scenarios: dict[str, dict],
    gt_dir: Path,
) -> dict[str, rdflib.Graph | None]:
    """
    For each scenario_id, find and parse the downloaded ground truth file.
    Returns None for any scenario where the file is missing or unparseable.
    """
    graphs: dict[str, rdflib.Graph | None] = {}
    for sid in scenarios:
        path = _find_gt_file(gt_dir, sid)
        if path is None:
            print(f"  [GT] {sid}: no ground truth file found in {gt_dir}")
            graphs[sid] = None
            continue
        g = load_graph(path)
        if len(g) > 0:
            graphs[sid] = g
            print(f"  [GT] {sid}: loaded {len(g)} triples from {path.name}")
        else:
            print(f"  [GT] {sid}: failed to parse {path}")
            graphs[sid] = None
    return graphs


def _find_gt_file(gt_dir: Path, sid: str) -> Path | None:
    for ext in _RDF_EXTENSIONS:
        p = gt_dir / f"{sid}{ext}"
        if p.exists():
            return p
    return None


# ---------------------------------------------------------------------------
# CSV serialisation helpers
# ---------------------------------------------------------------------------

def _flatten_for_csv(metrics: dict) -> dict:
    """
    Produce a flat dict safe for CSV. List-valued keys (cq_details,
    matched_class_names, matched_property_names) are JSON-encoded strings.
    """
    flat = {}
    for k, v in metrics.items():
        if isinstance(v, list):
            flat[k] = json.dumps(v, ensure_ascii=False)
        else:
            flat[k] = v
    return flat


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate generated ODP outputs.")
    parser.add_argument("--outputs-dir", default="outputs", type=Path)
    parser.add_argument("--data", default="data/scenarios/pattern_scenarios.json", type=Path)
    parser.add_argument("--results-dir", default="results", type=Path)
    parser.add_argument(
        "--ground-truth-dir",
        default="data/ground_truth",
        type=Path,
        help="Directory containing downloaded ground truth ontologies (from download_ground_truth.py).",
    )
    args = parser.parse_args()

    scenarios = {
        item["scenario_id"]: item
        for item in json.loads(args.data.read_text(encoding="utf-8"))
    }
    args.results_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading ground truth ontologies from {args.ground_truth_dir}/")
    gt_graphs = _load_ground_truth_graphs(scenarios, args.ground_truth_dir)
    print()

    summary_rows: list[dict] = []

    for ontology_path in sorted(args.outputs_dir.glob("*/*/*/ontology.ttl")):
        parts = ontology_path.parts
        model, config, scenario_id = parts[-4], parts[-3], parts[-2]
        if scenario_id not in scenarios:
            print(f"  [skip] {scenario_id} not in scenarios")
            continue

        scenario = scenarios[scenario_id]
        print(f"  Evaluating {model} / {config} / {scenario_id} ...", end=" ", flush=True)

        formal = evaluate_formal_quality(ontology_path)
        faith = evaluate_faithfulness(
            scenario_text=scenario["scenario_text"],
            cq_list=scenario.get("cq_list", []),
            ontology_path=ontology_path,
        )
        sim = evaluate_structure_similarity(
            ontology_path=ontology_path,
            gt_graph=gt_graphs.get(scenario_id),
        )

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
        result_file.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
        summary_rows.append(metrics)

        parse_ok = formal.get("parse_success", False)
        cq_cov = faith.get("cq_coverage_ratio", 0.0)
        cf1 = sim.get("combined_f1", 0.0)
        print(f"parse={'ok' if parse_ok else 'FAIL'} | cq_cov={cq_cov:.2f} | gt_f1={cf1:.2f}")

    if not summary_rows:
        print("\nNo outputs found. Run run_generation.py first.")
        return

    # Write CSV summary (list fields JSON-encoded)
    summary_path = args.results_dir / "summary.csv"
    flat_rows = [_flatten_for_csv(r) for r in summary_rows]
    fieldnames = list(flat_rows[0].keys())
    with summary_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(flat_rows)

    print(f"\nEvaluated {len(summary_rows)} outputs → {summary_path}")


if __name__ == "__main__":
    main()
