#!/usr/bin/env python3
"""Render a prompt template from JSON input with placeholder validation."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

PLACEHOLDER_PATTERN = re.compile(r"\{\{([A-Z0-9_]+)\}\}")

TEMPLATE_REQUIREMENTS = {
    "scenario_only.txt": {"SCENARIO_TEXT"},
    "cq_only.txt": {"CQ_LIST"},
    "scenario_cq.txt": {"SCENARIO_TEXT", "CQ_LIST"},
    "scenario_cq_reasoning.txt": {"SCENARIO_TEXT", "CQ_LIST", "REASONING_GUIDANCE"},
    "scenario_cq_constraints.txt": {"SCENARIO_TEXT", "CQ_LIST", "OUTPUT_CONSTRAINTS"},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render a prompt template with required placeholder checks."
    )
    parser.add_argument("--template", required=True, type=Path, help="Path to template file")
    parser.add_argument("--input", required=True, type=Path, help="Path to JSON input data")
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional output path. If omitted, rendered prompt is written to stdout.",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict[str, str]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise ValueError(f"Input file not found: {path}") from None
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in input file {path}: {exc}") from None

    if not isinstance(data, dict):
        raise ValueError("Input JSON must be an object with placeholder keys.")
    return data


def required_placeholders_for(template_path: Path) -> set[str]:
    name = template_path.name
    if name not in TEMPLATE_REQUIREMENTS:
        raise ValueError(
            f"Unsupported template '{name}'. Supported templates: {sorted(TEMPLATE_REQUIREMENTS)}"
        )
    return TEMPLATE_REQUIREMENTS[name]


def extract_placeholders(template_text: str) -> set[str]:
    return set(PLACEHOLDER_PATTERN.findall(template_text))


def validate_schema(template_path: Path, template_text: str, data: dict[str, str]) -> set[str]:
    required = required_placeholders_for(template_path)
    placeholders_in_template = extract_placeholders(template_text)

    missing_in_template = sorted(required - placeholders_in_template)
    if missing_in_template:
        raise ValueError(
            "Template is missing required placeholder(s): " + ", ".join(missing_in_template)
        )

    missing_in_input = sorted(key for key in required if key not in data or not str(data[key]).strip())
    if missing_in_input:
        raise ValueError(
            "Input JSON is missing required field(s): " + ", ".join(missing_in_input)
        )

    return placeholders_in_template


def render(template_text: str, placeholders: set[str], data: dict[str, str]) -> str:
    rendered = template_text
    for key in placeholders:
        value = str(data.get(key, ""))
        rendered = rendered.replace(f"{{{{{key}}}}}", value)
    return rendered


def main() -> int:
    args = parse_args()

    try:
        template_text = args.template.read_text(encoding="utf-8")
    except FileNotFoundError:
        print(f"Template file not found: {args.template}", file=sys.stderr)
        return 1

    try:
        data = load_json(args.input)
        placeholders = validate_schema(args.template, template_text, data)
        rendered = render(template_text, placeholders, data)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    if args.output:
        args.output.write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
