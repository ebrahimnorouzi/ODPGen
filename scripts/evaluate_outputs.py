#!/usr/bin/env python3
"""Evaluate generated artifacts and emit per-scenario JSON + summary CSV.

Expected input layout (configurable root):
  <input_root>/<model>/<config>/<scenario_id>/
      requirements.json   # optional list of atomic requirements (strings or objects)
      reference.txt       # optional reference output
      outputs/            # optional folder of candidate outputs
      *.txt|*.md|*.json|*.py  # candidate outputs if outputs/ absent

Output layout:
  results/<model>/<config>/<scenario_id>.json
  results/summary.csv
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List

from eval.faithfulness import evaluate_faithfulness
from eval.formal_quality import evaluate_formal_quality
from eval.similarity import evaluate_similarity


def _load_requirements(scenario_dir: Path) -> List[Any]:
    for name in ("requirements.json", "atomic_requirements.json"):
        p = scenario_dir / name
        if p.exists():
            with p.open("r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                data = data.get("requirements", data.get("atomic_requirements", []))
            return data if isinstance(data, list) else []
    return []


def _load_reference(scenario_dir: Path) -> str | None:
    for name in ("reference.txt", "target.txt", "gold.txt"):
        p = scenario_dir / name
        if p.exists():
            return p.read_text(encoding="utf-8")
    return None


def _iter_output_files(scenario_dir: Path) -> Iterable[Path]:
    outputs_dir = scenario_dir / "outputs"
    if outputs_dir.exists() and outputs_dir.is_dir():
        candidates = sorted(outputs_dir.iterdir())
    else:
        candidates = sorted(scenario_dir.iterdir())

    for p in candidates:
        if not p.is_file():
            continue
        if p.name in {"requirements.json", "atomic_requirements.json", "reference.txt", "target.txt", "gold.txt"}:
            continue
        if p.suffix.lower() in {".txt", ".md", ".json", ".py"}:
            yield p


def _scenario_metrics(model: str, config: str, scenario_id: str, scenario_dir: Path) -> Dict[str, Any]:
    atomic_requirements = _load_requirements(scenario_dir)
    reference_text = _load_reference(scenario_dir)

    outputs_payload: List[Dict[str, Any]] = []
    for out_file in _iter_output_files(scenario_dir):
        text = out_file.read_text(encoding="utf-8")
        output_metrics = {
            "output_id": out_file.stem,
            "output_file": str(out_file.relative_to(scenario_dir)),
            "formal_quality": evaluate_formal_quality(out_file, text),
            "faithfulness": evaluate_faithfulness(text, atomic_requirements),
            "similarity": evaluate_similarity(text, reference_text),
        }
        outputs_payload.append(output_metrics)

    return {
        "model": model,
        "config": config,
        "scenario_id": scenario_id,
        "source_dir": str(scenario_dir),
        "requirements_present": bool(atomic_requirements),
        "reference_present": reference_text is not None,
        "outputs_evaluated": len(outputs_payload),
        "outputs": outputs_payload,
    }


def _summary_rows(scenario_result: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for output in scenario_result.get("outputs", []):
        formal = output["formal_quality"]
        faith = output["faithfulness"]
        sim = output["similarity"]
        rows.append(
            {
                "model": scenario_result["model"],
                "config": scenario_result["config"],
                "scenario_id": scenario_result["scenario_id"],
                "output_id": output["output_id"],
                "syntax_parse": formal["syntax_parse"]["value"],
                "syntax_parse_evaluated": formal["syntax_parse"]["evaluated"],
                "coverage": faith["coverage"]["value"],
                "coverage_evaluated": faith["coverage"]["evaluated"],
                "unsupported_additions": faith["unsupported_additions"]["value"],
                "unsupported_additions_evaluated": faith["unsupported_additions"]["evaluated"],
                "token_jaccard": sim["token_jaccard"]["value"],
                "token_jaccard_evaluated": sim["token_jaccard"]["evaluated"],
                "axiom_similarity_evaluated": sim["axiom_similarity"]["evaluated"],
                "structure_similarity_evaluated": sim["structure_similarity"]["evaluated"],
                "consistency_evaluated": formal["consistency"]["evaluated"],
                "coherence_evaluated": formal["coherence"]["evaluated"],
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", default="generated", help="Root dir containing model/config/scenario folders")
    parser.add_argument("--results-root", default="results", help="Output root for evaluation results")
    args = parser.parse_args()

    input_root = Path(args.input_root)
    results_root = Path(args.results_root)
    results_root.mkdir(parents=True, exist_ok=True)

    all_summary_rows: List[Dict[str, Any]] = []

    if not input_root.exists():
        raise SystemExit(f"Input root does not exist: {input_root}")

    for model_dir in sorted([p for p in input_root.iterdir() if p.is_dir()]):
        model = model_dir.name
        for config_dir in sorted([p for p in model_dir.iterdir() if p.is_dir()]):
            config = config_dir.name
            for scenario_dir in sorted([p for p in config_dir.iterdir() if p.is_dir()]):
                scenario_id = scenario_dir.name
                result = _scenario_metrics(model, config, scenario_id, scenario_dir)

                out_dir = results_root / model / config
                out_dir.mkdir(parents=True, exist_ok=True)
                out_file = out_dir / f"{scenario_id}.json"
                out_file.write_text(json.dumps(result, indent=2), encoding="utf-8")

                all_summary_rows.extend(_summary_rows(result))

    summary_file = results_root / "summary.csv"
    fieldnames = [
        "model",
        "config",
        "scenario_id",
        "output_id",
        "syntax_parse",
        "syntax_parse_evaluated",
        "coverage",
        "coverage_evaluated",
        "unsupported_additions",
        "unsupported_additions_evaluated",
        "token_jaccard",
        "token_jaccard_evaluated",
        "axiom_similarity_evaluated",
        "structure_similarity_evaluated",
        "consistency_evaluated",
        "coherence_evaluated",
    ]
    with summary_file.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_summary_rows)


if __name__ == "__main__":
    main()
