#!/usr/bin/env python3
"""
Aggregate LLM-as-a-Judge evaluation results into summary tables and reports.

Reads per-instance JSON results produced by run_evaluation.py and generates:
  1. Per-model mean scores (across all configs and scenarios)
  2. Per-config mean scores (across all models and scenarios)
  3. Per-scenario mean scores
  4. Full cross-tabulation (model × config)
  5. Markdown summary report

Usage:
    python -m eval_judge.aggregate_results
    python -m eval_judge.aggregate_results --results-dir eval_judge_results
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# ─────────────────────────────────────────────────────────────────────────────
# Score dimensions
# ─────────────────────────────────────────────────────────────────────────────

SCORE_FIELDS = [
    "syntactic_correctness_score",
    "semantic_accuracy_score",
    "logical_consistency_score",
    "functional_adequacy_score",
    "overall_score",
]

SCORE_LABELS = {
    "syntactic_correctness_score": "Syntactic",
    "semantic_accuracy_score": "Semantic",
    "logical_consistency_score": "Logical",
    "functional_adequacy_score": "Functional",
    "overall_score": "Overall",
}


# ─────────────────────────────────────────────────────────────────────────────
# Load results
# ─────────────────────────────────────────────────────────────────────────────

def load_single_results(results_dir: Path) -> list[dict]:
    """Load all single-candidate evaluation JSONs."""
    single_dir = results_dir / "single"
    if not single_dir.exists():
        print(f"[aggregate] No single results found in {single_dir}")
        return []

    rows = []
    for json_path in sorted(single_dir.glob("*/*/*.json")):
        if json_path.suffix == ".json" and not json_path.name.endswith(".raw.txt"):
            data = json.loads(json_path.read_text(encoding="utf-8"))
            rows.append(data)
    print(f"[aggregate] Loaded {len(rows)} single evaluation results")
    return rows


# ─────────────────────────────────────────────────────────────────────────────
# Aggregation functions
# ─────────────────────────────────────────────────────────────────────────────

def _aggregate_group(rows: list[dict], group_key: str) -> list[dict]:
    """Group rows by a key and compute mean ± std for each score dimension."""
    groups: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        groups[r[group_key]].append(r)

    table = []
    for key in sorted(groups.keys()):
        group = groups[key]
        entry = {group_key: key, "n": len(group)}
        for field in SCORE_FIELDS:
            vals = [r[field] for r in group if isinstance(r.get(field), (int, float))]
            if vals:
                entry[f"{field}_mean"] = round(mean(vals), 2)
                entry[f"{field}_std"] = round(stdev(vals), 2) if len(vals) > 1 else 0.0
            else:
                entry[f"{field}_mean"] = None
                entry[f"{field}_std"] = None
        table.append(entry)
    return table


def aggregate_by_model(rows: list[dict]) -> list[dict]:
    return _aggregate_group(rows, "model")


def aggregate_by_config(rows: list[dict]) -> list[dict]:
    return _aggregate_group(rows, "config")


def aggregate_by_scenario(rows: list[dict]) -> list[dict]:
    return _aggregate_group(rows, "scenario_id")


def cross_tabulation(rows: list[dict]) -> dict[tuple[str, str], dict]:
    """Model × Config cross-tab of mean overall scores."""
    groups: dict[tuple[str, str], list[int]] = defaultdict(list)
    for r in rows:
        k = (r["model"], r["config"])
        if isinstance(r.get("overall_score"), (int, float)):
            groups[k].append(r["overall_score"])

    return {
        k: {"mean": round(mean(v), 2), "n": len(v)}
        for k, v in sorted(groups.items())
    }


# ─────────────────────────────────────────────────────────────────────────────
# Markdown report
# ─────────────────────────────────────────────────────────────────────────────

def _md_table(headers: list[str], rows_data: list[list[str]]) -> str:
    """Format a simple markdown table."""
    lines = ["| " + " | ".join(headers) + " |"]
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for row in rows_data:
        lines.append("| " + " | ".join(str(c) for c in row) + " |")
    return "\n".join(lines)


def generate_report(rows: list[dict], results_dir: Path) -> str:
    """Generate a markdown summary report."""
    parts = ["# LLM-as-a-Judge Evaluation Report\n"]

    if not rows:
        parts.append("No evaluation results found.\n")
        return "\n".join(parts)

    judge_models = set(r.get("judge_model", "unknown") for r in rows)
    parts.append(f"**Judge model(s):** {', '.join(sorted(judge_models))}  ")
    parts.append(f"**Total evaluations:** {len(rows)}\n")

    # ── By Model ─────────────────────────────────────────────────────────
    parts.append("## Scores by Model\n")
    by_model = aggregate_by_model(rows)
    headers = ["Model", "N"] + [SCORE_LABELS[f] for f in SCORE_FIELDS]
    table_rows = []
    for entry in by_model:
        row = [entry["model"], str(entry["n"])]
        for f in SCORE_FIELDS:
            m = entry.get(f"{f}_mean")
            s = entry.get(f"{f}_std")
            row.append(f"{m:.2f} ± {s:.2f}" if m is not None else "—")
        table_rows.append(row)
    parts.append(_md_table(headers, table_rows))
    parts.append("")

    # ── By Config ────────────────────────────────────────────────────────
    parts.append("## Scores by Prompt Configuration\n")
    by_config = aggregate_by_config(rows)
    headers = ["Config", "N"] + [SCORE_LABELS[f] for f in SCORE_FIELDS]
    table_rows = []
    for entry in by_config:
        row = [entry["config"], str(entry["n"])]
        for f in SCORE_FIELDS:
            m = entry.get(f"{f}_mean")
            s = entry.get(f"{f}_std")
            row.append(f"{m:.2f} ± {s:.2f}" if m is not None else "—")
        table_rows.append(row)
    parts.append(_md_table(headers, table_rows))
    parts.append("")

    # ── By Scenario ──────────────────────────────────────────────────────
    parts.append("## Scores by Scenario\n")
    by_scenario = aggregate_by_scenario(rows)
    headers = ["Scenario", "N"] + [SCORE_LABELS[f] for f in SCORE_FIELDS]
    table_rows = []
    for entry in by_scenario:
        row = [entry["scenario_id"], str(entry["n"])]
        for f in SCORE_FIELDS:
            m = entry.get(f"{f}_mean")
            s = entry.get(f"{f}_std")
            row.append(f"{m:.2f} ± {s:.2f}" if m is not None else "—")
        table_rows.append(row)
    parts.append(_md_table(headers, table_rows))
    parts.append("")

    # ── Cross-tab (Model × Config) ──────────────────────────────────────
    parts.append("## Overall Score: Model × Config\n")
    xtab = cross_tabulation(rows)
    models = sorted(set(r["model"] for r in rows))
    configs = sorted(set(r["config"] for r in rows))
    headers = ["Model"] + configs
    table_rows = []
    for model in models:
        row = [model]
        for config in configs:
            entry = xtab.get((model, config))
            row.append(f"{entry['mean']:.1f} (n={entry['n']})" if entry else "—")
        table_rows.append(row)
    parts.append(_md_table(headers, table_rows))
    parts.append("")

    return "\n".join(parts)


# ─────────────────────────────────────────────────────────────────────────────
# CSV writers
# ─────────────────────────────────────────────────────────────────────────────

def write_aggregation_csv(table: list[dict], path: Path) -> None:
    if not table:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(table[0].keys()))
        writer.writeheader()
        writer.writerows(table)
    print(f"  → {path}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(description="Aggregate LLM-as-a-Judge results.")
    p.add_argument(
        "--results-dir", type=Path,
        default=Path(__file__).resolve().parent.parent / "eval_judge_results",
    )
    args = p.parse_args()

    rows = load_single_results(args.results_dir)
    if not rows:
        print("[aggregate] Nothing to aggregate.")
        return

    out = args.results_dir / "aggregated"
    out.mkdir(parents=True, exist_ok=True)

    print("\n[aggregate] Writing aggregation CSVs:")
    write_aggregation_csv(aggregate_by_model(rows), out / "by_model.csv")
    write_aggregation_csv(aggregate_by_config(rows), out / "by_config.csv")
    write_aggregation_csv(aggregate_by_scenario(rows), out / "by_scenario.csv")

    report = generate_report(rows, args.results_dir)
    report_path = out / "judge_report.md"
    report_path.write_text(report, encoding="utf-8")
    print(f"  → {report_path}")

    # Also print report to stdout
    print("\n" + "=" * 72)
    print(report)

    print("\n[aggregate] Done.")


if __name__ == "__main__":
    main()
