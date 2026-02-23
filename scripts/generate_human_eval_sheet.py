#!/usr/bin/env python3
"""Generate blinded human/LLM evaluation packets from model outputs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

RATING_COLUMNS = [
    "correctness_1_to_5",
    "completeness_1_to_5",
    "clarity_reusability_1_to_5",
    "comments_optional",
]
OUTPUT_CANDIDATE_FILES = [
    "output.txt",
    "response.txt",
    "answer.txt",
    "output.md",
    "response.md",
    "output.json",
    "response.json",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outputs-dir", default="outputs", help="Root output directory")
    parser.add_argument("--results-dir", default="results", help="Destination for packet + mappings")
    parser.add_argument("--format", choices=["csv", "xlsx"], default="csv", help="Packet file format")
    parser.add_argument("--packet-name", default="human_eval_packet", help="Packet filename stem")
    parser.add_argument("--seed", default="human_eval", help="Seed used for deterministic anonymized IDs")
    parser.add_argument(
        "--include-output-text",
        action="store_true",
        help="Include best-effort model output text in the packet (useful for LLM-as-judge).",
    )
    parser.add_argument(
        "--output-max-chars",
        type=int,
        default=8000,
        help="Max output text length when --include-output-text is enabled.",
    )
    return parser.parse_args()


def discover_output_dirs(outputs_dir: Path) -> List[Tuple[str, str, str, Path]]:
    records: List[Tuple[str, str, str, Path]] = []
    if not outputs_dir.exists():
        return records
    for model_dir in sorted([p for p in outputs_dir.iterdir() if p.is_dir()]):
        for config_dir in sorted([p for p in model_dir.iterdir() if p.is_dir()]):
            for scenario_dir in sorted([p for p in config_dir.iterdir() if p.is_dir()]):
                records.append((model_dir.name, config_dir.name, scenario_dir.name, scenario_dir))
    return records


def extract_snippet_refs(scenario_dir: Path) -> str:
    candidates = [
        scenario_dir / "metadata.json",
        scenario_dir / "meta.json",
        scenario_dir / "scenario.json",
        scenario_dir / "references.json",
    ]
    keys = ["ontology_refs", "documentation_refs", "snippet_refs", "snippet_references", "references"]

    for file_path in candidates:
        if not file_path.exists():
            continue
        try:
            payload = json.loads(file_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue

        refs: List[str] = []
        for key in keys:
            value = payload.get(key)
            if isinstance(value, list):
                refs.extend(str(item) for item in value)
            elif isinstance(value, str) and value.strip():
                refs.append(value.strip())
        if refs:
            return " | ".join(dict.fromkeys(refs))

    txt_ref = scenario_dir / "references.txt"
    if txt_ref.exists():
        try:
            lines = [ln.strip() for ln in txt_ref.read_text(encoding="utf-8").splitlines() if ln.strip()]
            if lines:
                return " | ".join(lines)
        except OSError:
            pass

    return ""


def extract_output_text(scenario_dir: Path, max_chars: int) -> str:
    for filename in OUTPUT_CANDIDATE_FILES:
        file_path = scenario_dir / filename
        if not file_path.exists():
            continue
        try:
            if file_path.suffix.lower() == ".json":
                payload = json.loads(file_path.read_text(encoding="utf-8"))
                text = json.dumps(payload, ensure_ascii=False, indent=2)
            else:
                text = file_path.read_text(encoding="utf-8")
        except (OSError, json.JSONDecodeError):
            continue
        text = text.strip()
        return text[:max_chars]
    return ""


def anonymized_id(seed: str, model: str, config: str, scenario_id: str) -> str:
    raw = f"{seed}:{model}:{config}:{scenario_id}".encode("utf-8")
    return f"OUT_{hashlib.sha256(raw).hexdigest()[:12].upper()}"


def write_csv(path: Path, rows: Sequence[Dict[str, str]], fieldnames: Sequence[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        writer.writerows(rows)


def write_xlsx(path: Path, rows: Sequence[Dict[str, str]], fieldnames: Sequence[str]) -> None:
    try:
        from openpyxl import Workbook
    except ImportError as exc:
        raise RuntimeError("XLSX output requires openpyxl. Install it or use --format csv.") from exc

    wb = Workbook()
    ws = wb.active
    ws.title = "eval_packet"
    ws.append(list(fieldnames))
    for row in rows:
        ws.append([row.get(col, "") for col in fieldnames])
    wb.save(path)


def main() -> int:
    args = parse_args()
    outputs_dir = Path(args.outputs_dir)
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    found = discover_output_dirs(outputs_dir)
    if not found:
        raise SystemExit(f"No scenario outputs found under: {outputs_dir}")

    packet_rows: List[Dict[str, str]] = []
    mapping_rows: List[Dict[str, str]] = []

    for model, config, scenario_id, scenario_dir in found:
        anon_id = anonymized_id(args.seed, model, config, scenario_id)
        row = {
            "scenario_id": scenario_id,
            "anonymized_output_id": anon_id,
            "snippet_references": extract_snippet_refs(scenario_dir),
            **{col: "" for col in RATING_COLUMNS},
        }
        if args.include_output_text:
            row["output_text"] = extract_output_text(scenario_dir, args.output_max_chars)
        packet_rows.append(row)

        mapping_rows.append(
            {
                "anonymized_output_id": anon_id,
                "model": model,
                "config": config,
                "scenario_id": scenario_id,
                "output_path": f"outputs/{model}/{config}/{scenario_id}",
                "absolute_output_path": str(scenario_dir.resolve()),
            }
        )

    packet_fields = ["scenario_id", "anonymized_output_id", "snippet_references"]
    if args.include_output_text:
        packet_fields.append("output_text")
    packet_fields.extend(RATING_COLUMNS)

    packet_path = results_dir / f"{args.packet_name}.{args.format}"
    mapping_csv_path = results_dir / "human_eval_mapping.csv"
    mapping_json_path = results_dir / "human_eval_mapping.json"

    if args.format == "csv":
        write_csv(packet_path, packet_rows, packet_fields)
    else:
        write_xlsx(packet_path, packet_rows, packet_fields)

    mapping_fields = [
        "anonymized_output_id",
        "model",
        "config",
        "scenario_id",
        "output_path",
        "absolute_output_path",
    ]
    write_csv(mapping_csv_path, mapping_rows, mapping_fields)
    mapping_json_path.write_text(json.dumps(mapping_rows, indent=2), encoding="utf-8")

    print(f"Generated packet: {packet_path}")
    print(f"Generated mapping: {mapping_csv_path}")
    print(f"Generated mapping: {mapping_json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
