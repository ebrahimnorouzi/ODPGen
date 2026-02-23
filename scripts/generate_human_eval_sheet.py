#!/usr/bin/env python3
"""Generate blinded human evaluation packets from model outputs.

This script scans `outputs/{model}/{config}/{scenario_id}` directories, builds
an anonymized output identifier for each scenario output, writes a blinded
rating packet (CSV or XLSX), and stores mapping files for traceability.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Dict, List, Tuple

RATING_COLUMNS = [
    "correctness_1_to_5",
    "completeness_1_to_5",
    "clarity_reusability_1_to_5",
    "comments_optional",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outputs-dir", default="outputs", help="Root output directory")
    parser.add_argument("--results-dir", default="results", help="Destination for packet + mappings")
    parser.add_argument(
        "--format",
        choices=["csv", "xlsx"],
        default="csv",
        help="Packet file format",
    )
    parser.add_argument(
        "--packet-name",
        default="human_eval_packet",
        help="Packet filename stem (without extension)",
    )
    parser.add_argument(
        "--seed",
        default="human_eval",
        help="Seed used for deterministic anonymized IDs",
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
    """Best-effort extraction for ontology/documentation snippet references."""
    candidates = [
        scenario_dir / "metadata.json",
        scenario_dir / "meta.json",
        scenario_dir / "scenario.json",
        scenario_dir / "references.json",
    ]
    keys = [
        "ontology_refs",
        "documentation_refs",
        "snippet_refs",
        "snippet_references",
        "references",
    ]

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

    # Optional plaintext fallback for hand-authored references.
    txt_ref = scenario_dir / "references.txt"
    if txt_ref.exists():
        try:
            lines = [ln.strip() for ln in txt_ref.read_text(encoding="utf-8").splitlines() if ln.strip()]
            if lines:
                return " | ".join(lines)
        except OSError:
            pass

    return ""


def anonymized_id(seed: str, model: str, config: str, scenario_id: str) -> str:
    raw = f"{seed}:{model}:{config}:{scenario_id}".encode("utf-8")
    digest = hashlib.sha256(raw).hexdigest()[:12].upper()
    return f"OUT_{digest}"


def write_csv(path: Path, rows: List[Dict[str, str]], fieldnames: List[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_xlsx(path: Path, rows: List[Dict[str, str]], fieldnames: List[str]) -> None:
    try:
        from openpyxl import Workbook
    except ImportError as exc:
        raise RuntimeError(
            "XLSX output requires openpyxl. Install it or use --format csv."
        ) from exc

    wb = Workbook()
    ws = wb.active
    ws.title = "human_eval"
    ws.append(fieldnames)
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
        refs = extract_snippet_refs(scenario_dir)

        packet_rows.append(
            {
                "scenario_id": scenario_id,
                "anonymized_output_id": anon_id,
                "snippet_references": refs,
                **{col: "" for col in RATING_COLUMNS},
            }
        )
        mapping_rows.append(
            {
                "anonymized_output_id": anon_id,
                "model": model,
                "config": config,
                "scenario_id": scenario_id,
                "output_path": str(scenario_dir),
            }
        )

    packet_fields = [
        "scenario_id",
        "anonymized_output_id",
        "snippet_references",
        *RATING_COLUMNS,
    ]
    mapping_fields = ["anonymized_output_id", "model", "config", "scenario_id", "output_path"]

    packet_path = results_dir / f"{args.packet_name}.{args.format}"
    mapping_csv_path = results_dir / "human_eval_mapping.csv"
    mapping_json_path = results_dir / "human_eval_mapping.json"

    if args.format == "csv":
        write_csv(packet_path, packet_rows, packet_fields)
    else:
        write_xlsx(packet_path, packet_rows, packet_fields)

    write_csv(mapping_csv_path, mapping_rows, mapping_fields)
    mapping_json_path.write_text(json.dumps(mapping_rows, indent=2), encoding="utf-8")

    print(f"Generated packet: {packet_path}")
    print(f"Generated mapping: {mapping_csv_path}")
    print(f"Generated mapping: {mapping_json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
