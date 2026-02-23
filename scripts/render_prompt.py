#!/usr/bin/env python3
import argparse
from pathlib import Path


def render(template: str, scenario_text: str, cq_list: list[str]) -> str:
    cq_block = "\n".join(f"- {cq}" for cq in cq_list) if cq_list else "- (none provided)"
    return (
        template.replace("{{SCENARIO_TEXT}}", scenario_text.strip())
        .replace("{{CQ_LIST}}", cq_block)
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Render prompt templates for ODP generation.")
    parser.add_argument("--template", required=True, type=Path)
    parser.add_argument("--scenario", default="")
    parser.add_argument("--cq", action="append", default=[])
    args = parser.parse_args()

    template = args.template.read_text(encoding="utf-8")
    required_placeholders = ["{{SCENARIO_TEXT}}", "{{CQ_LIST}}"]
    missing = [p for p in required_placeholders if p not in template]
    if missing:
        raise SystemExit(f"Template missing required placeholder(s): {', '.join(missing)}")

    print(render(template, args.scenario, args.cq))


if __name__ == "__main__":
    main()
