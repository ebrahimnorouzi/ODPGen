#!/usr/bin/env python3
"""CLI for ontology prompt generation runs."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

CONFIG_CHOICES = [
    "scenario-only",
    "cq-only",
    "scenario-cq",
    "scenario-cq-reasoning",
    "scenario-cq-constraints",
]


@dataclass
class ScenarioInput:
    scenario_id: str
    scenario_text: str
    cq_list: List[str]


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def load_scenarios(data_dir: Path) -> List[ScenarioInput]:
    scenario_files = sorted(data_dir.glob("*.json"))
    if not scenario_files:
        raise ValueError(f"No scenario files found in {data_dir}")

    scenarios: List[ScenarioInput] = []
    for path in scenario_files:
        payload = json.loads(path.read_text(encoding="utf-8"))
        for key in ("scenario_id", "scenario_text", "cq_list"):
            if key not in payload:
                raise ValueError(f"{path} missing required key: {key}")
        if not isinstance(payload["cq_list"], list):
            raise ValueError(f"{path} has non-list cq_list")
        scenarios.append(
            ScenarioInput(
                scenario_id=str(payload["scenario_id"]),
                scenario_text=str(payload["scenario_text"]),
                cq_list=[str(item) for item in payload["cq_list"]],
            )
        )
    return scenarios


def render_prompt(template: str, scenario: ScenarioInput, config: str) -> str:
    cq_block = "\n".join(f"- {cq}" for cq in scenario.cq_list) or "- (none)"
    context: Dict[str, Any] = {
        "scenario_id": scenario.scenario_id,
        "scenario_text": scenario.scenario_text,
        "cq_block": cq_block,
        "reasoning_block": "Include explicit reasoning steps for each modeling decision.",
        "constraints_block": "Respect closed-world assumptions and cardinality constraints where needed.",
        "config": config,
    }
    return template.format(**context).strip() + "\n"


def extract_ontology_text(raw_response: str) -> str:
    pattern = re.compile(r"```(?:ontology|owl|ttl|turtle|rdf|xml)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)
    match = pattern.search(raw_response)
    if match:
        return match.group(1).strip() + "\n"
    return raw_response.strip() + "\n"


def call_model(prompt: str, args: argparse.Namespace) -> str:
    if args.mock_response is not None:
        return args.mock_response
    if args.mock_response_file is not None:
        return Path(args.mock_response_file).read_text(encoding="utf-8")
    raise ValueError(
        "No model backend configured. Provide --mock-response or --mock-response-file, "
        "or run with --dry-run to inspect prompts."
    )


def write_outputs(
    output_root: Path,
    model: str,
    config: str,
    scenario: ScenarioInput,
    prompt: str,
    response: str,
    metadata: Dict[str, Any],
) -> None:
    run_dir = output_root / model / config / scenario.scenario_id
    run_dir.mkdir(parents=True, exist_ok=True)

    (run_dir / "prompt.txt").write_text(prompt, encoding="utf-8")
    (run_dir / "raw_response.txt").write_text(response, encoding="utf-8")
    (run_dir / "ontology.txt").write_text(extract_ontology_text(response), encoding="utf-8")
    (run_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run scenario/CQ prompt generation workflow.")
    parser.add_argument("--config", required=True, choices=CONFIG_CHOICES)
    parser.add_argument("--model", default="mock-model")
    parser.add_argument("--data-dir", default="data/scenarios")
    parser.add_argument("--prompts-dir", default="prompts")
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--mock-response", default=None)
    parser.add_argument("--mock-response-file", default=None)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.temperature is not None and not (0 <= args.temperature <= 2):
        raise ValueError("--temperature must be between 0 and 2")
    if args.mock_response and args.mock_response_file:
        raise ValueError("Use only one of --mock-response or --mock-response-file")

    template_path = Path(args.prompts_dir) / f"{args.config}.txt"
    if not template_path.exists():
        raise ValueError(f"Template file not found: {template_path}")

    if not Path(args.data_dir).exists():
        raise ValueError(f"Data directory not found: {args.data_dir}")


def run(args: argparse.Namespace) -> None:
    validate_args(args)
    template_path = Path(args.prompts_dir) / f"{args.config}.txt"
    template_text = template_path.read_text(encoding="utf-8")
    template_hash = sha256_text(template_text)

    scenarios = load_scenarios(Path(args.data_dir))
    config_fingerprint = {
        "config": args.config,
        "template_hash": template_hash,
        "temperature": args.temperature,
        "seed": args.seed,
    }
    config_hash = sha256_text(json.dumps(config_fingerprint, sort_keys=True))

    for scenario in scenarios:
        prompt = render_prompt(template_text, scenario, args.config)
        prompt_hash = sha256_text(prompt)

        if args.dry_run:
            print(f"=== DRY RUN: {scenario.scenario_id} ===")
            print(prompt)
            continue

        response = call_model(prompt, args)
        metadata = {
            "model": args.model,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "config": args.config,
            "config_hash": config_hash,
            "prompt_hash": prompt_hash,
            "template_hash": template_hash,
            "seed": args.seed,
            "temperature": args.temperature,
            "scenario_id": scenario.scenario_id,
        }
        write_outputs(Path(args.output_dir), args.model, args.config, scenario, prompt, response, metadata)
        print(f"Wrote outputs for {scenario.scenario_id}")


def main() -> None:
    args = parse_args()
    run(args)


if __name__ == "__main__":
    main()
