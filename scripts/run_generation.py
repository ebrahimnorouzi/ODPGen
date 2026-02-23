#!/usr/bin/env python3
import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

CONFIG_TEMPLATE = {
    "scenario-only": "scenario_only.txt",
    "cq-only": "cq_only.txt",
    "scenario-cq": "scenario_cq.txt",
    "scenario-cq-reasoning": "scenario_cq_reasoning.txt",
    "scenario-cq-constraints": "scenario_cq_constraints.txt",
}


def render(template: str, scenario_text: str, cq_list: list[str]) -> str:
    cq_block = "\n".join(f"- {cq}" for cq in cq_list) if cq_list else "- (none provided)"
    return template.replace("{{SCENARIO_TEXT}}", scenario_text).replace("{{CQ_LIST}}", cq_block)


def slug(text: str) -> str:
    return "".join(ch for ch in text.title() if ch.isalnum()) or "Entity"


def mock_ontology(scenario: dict[str, Any], config: str) -> str:
    sid = scenario["scenario_id"]
    core = slug(sid)
    cq_entities = [slug(cq.split()[0]) for cq in scenario.get("cq_list", [])][:3]
    classes = [core, f"{core}Event", f"{core}Participant", *cq_entities]
    classes = list(dict.fromkeys(classes))

    lines = [
        "@prefix : <http://example.org/odp/> .",
        "@prefix owl: <http://www.w3.org/2002/07/owl#> .",
        "@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .",
        "",
        ":Pattern a owl:Ontology .",
    ]
    for c in classes:
        lines.append(f":{c} a owl:Class .")
    lines.extend(
        [
            ":hasParticipant a owl:ObjectProperty ; rdfs:domain :Pattern ; rdfs:range :" + classes[2] + " .",
            ":hasTimestamp a owl:DatatypeProperty ; rdfs:domain :Pattern ; rdfs:range rdfs:Literal .",
        ]
    )
    lines.append("")
    lines.append("# Documentation")
    lines.append(f"# config: {config}")
    lines.append(f"# scenario_id: {sid}")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run ODP generation experiments.")
    parser.add_argument("--data", default="data/scenarios/pattern_scenarios.json", type=Path)
    parser.add_argument("--prompts-dir", default="prompts", type=Path)
    parser.add_argument("--outputs-dir", default="outputs", type=Path)
    parser.add_argument("--model", default="mock-odp-generator")
    parser.add_argument("--config", choices=list(CONFIG_TEMPLATE.keys()) + ["all"], default="all")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.0)
    args = parser.parse_args()

    scenarios = json.loads(args.data.read_text(encoding="utf-8"))
    configs = list(CONFIG_TEMPLATE.keys()) if args.config == "all" else [args.config]

    for config in configs:
        template_path = args.prompts_dir / CONFIG_TEMPLATE[config]
        template = template_path.read_text(encoding="utf-8")
        for scenario in scenarios:
            prompt = render(template, scenario["scenario_text"], scenario.get("cq_list", []))
            prompt_hash = hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:12]
            output_dir = args.outputs_dir / args.model / config / scenario["scenario_id"]
            output_dir.mkdir(parents=True, exist_ok=True)

            if args.dry_run:
                (output_dir / "prompt.txt").write_text(prompt, encoding="utf-8")
                continue

            response = mock_ontology(scenario, config)
            response_hash = hashlib.sha256(response.encode("utf-8")).hexdigest()[:12]
            metadata = {
                "model": args.model,
                "config": config,
                "scenario_id": scenario["scenario_id"],
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "temperature": args.temperature,
                "prompt_hash": prompt_hash,
                "response_hash": response_hash,
                "mock_mode": True,
            }
            (output_dir / "prompt.txt").write_text(prompt, encoding="utf-8")
            (output_dir / "raw_response.txt").write_text(response, encoding="utf-8")
            (output_dir / "ontology.ttl").write_text(response, encoding="utf-8")
            (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
