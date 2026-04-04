#!/usr/bin/env python3
"""
LLM-as-a-Judge evaluation pipeline — main orchestrator.

Iterates over all generated ODPs, evaluates each against its ground-truth
ontology using a judge LLM, and writes per-instance results + a summary CSV.

Usage:
    # Single-candidate evaluation (default)
    python -m eval_judge.run_evaluation

    # Pairwise evaluation between two models
    python -m eval_judge.run_evaluation --mode pairwise \
        --model-a gpt-5.4 --model-b meta-llama_Llama-3.1-8B-Instruct

    # Override judge model
    python -m eval_judge.run_evaluation --judge-model claude-opus-4-6

    # Filter to specific configs or scenarios
    python -m eval_judge.run_evaluation --configs cq-only scenario-cq \
        --scenarios 2023-133-01 2023-134-01
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import os
from datetime import datetime, timezone
from pathlib import Path

# Allow running as `python -m eval_judge.run_evaluation`
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

from eval_judge.data_loader import (
    load_scenarios,
    load_all_generated_odps,
    DEFAULT_OUTPUTS_DIR,
    DEFAULT_GT_DIR,
    DEFAULT_SCENARIOS_PATH,
)
from eval_judge.judge import JudgeLLM, check_judge_independence


# ─────────────────────────────────────────────────────────────────────────────
# Single-candidate evaluation
# ─────────────────────────────────────────────────────────────────────────────

def run_single_evaluation(
    judge: JudgeLLM,
    scenarios_path: Path,
    gt_dir: Path,
    outputs_dir: Path,
    results_dir: Path,
    configs_filter: list[str] | None = None,
    scenarios_filter: list[str] | None = None,
    skip_existing: bool = True,
) -> list[dict]:
    """Evaluate every generated ODP individually against its ground truth."""
    print(f"[pipeline] Loading scenarios from {scenarios_path}")
    scenarios = load_scenarios(scenarios_path, gt_dir)
    print(f"[pipeline] Found {len(scenarios)} scenarios with ground truth\n")

    print(f"[pipeline] Discovering generated ODPs in {outputs_dir}/")
    odps = load_all_generated_odps(outputs_dir)
    print(f"[pipeline] Found {len(odps)} generated ODPs\n")

    # Apply filters
    if configs_filter:
        odps = [o for o in odps if o.config in configs_filter]
    if scenarios_filter:
        odps = [o for o in odps if o.scenario_id in scenarios_filter]

    print(f"[pipeline] Evaluating {len(odps)} ODPs with judge={judge.model}\n")

    summary_rows: list[dict] = []
    total = len(odps)

    for idx, odp in enumerate(odps, 1):
        scenario = scenarios.get(odp.scenario_id)
        if scenario is None or scenario.ground_truth_text is None:
            print(f"  [{idx}/{total}] SKIP {odp.model}/{odp.config}/{odp.scenario_id} — no ground truth")
            continue

        # Check judge independence
        if not check_judge_independence(judge.model, odp.model):
            print(f"  [{idx}/{total}] SKIP {odp.model}/{odp.config}/{odp.scenario_id} — judge conflicts with generator family")
            continue

        # Check for existing result
        result_path = results_dir / "single" / odp.model / odp.config / f"{odp.scenario_id}.json"
        if skip_existing and result_path.exists():
            print(f"  [{idx}/{total}] CACHED {odp.model}/{odp.config}/{odp.scenario_id}")
            cached = json.loads(result_path.read_text(encoding="utf-8"))
            summary_rows.append(cached)
            continue

        print(f"  [{idx}/{total}] Evaluating {odp.model} / {odp.config} / {odp.scenario_id} ...", end=" ", flush=True)

        try:
            result = judge.evaluate_single(
                ground_truth=scenario.ground_truth_text,
                cq_list=scenario.cq_list,
                candidate=odp.ontology_text,
            )

            row = {
                "model": odp.model,
                "config": odp.config,
                "scenario_id": odp.scenario_id,
                "judge_model": judge.model,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                **result.to_dict(),
            }

            # Save per-instance result
            result_path.parent.mkdir(parents=True, exist_ok=True)
            result_path.write_text(
                json.dumps(row, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )

            # Also save raw judge response for auditability
            raw_path = result_path.with_suffix(".raw.txt")
            raw_path.write_text(result.raw_response, encoding="utf-8")

            summary_rows.append(row)
            print(
                f"syn={result.syntactic_correctness_score} "
                f"sem={result.semantic_accuracy_score} "
                f"log={result.logical_consistency_score} "
                f"func={result.functional_adequacy_score} "
                f"overall={result.overall_score}"
            )

        except Exception as e:
            print(f"ERROR: {e}")
            continue

    return summary_rows


# ─────────────────────────────────────────────────────────────────────────────
# Pairwise evaluation
# ─────────────────────────────────────────────────────────────────────────────

def run_pairwise_evaluation(
    judge: JudgeLLM,
    model_a: str,
    model_b: str,
    scenarios_path: Path,
    gt_dir: Path,
    outputs_dir: Path,
    results_dir: Path,
    configs_filter: list[str] | None = None,
    scenarios_filter: list[str] | None = None,
    skip_existing: bool = True,
) -> list[dict]:
    """Run pairwise evaluation between two models with position-swap debiasing."""
    print(f"[pipeline] Pairwise: {model_a} vs {model_b}")
    scenarios = load_scenarios(scenarios_path, gt_dir)
    odps = load_all_generated_odps(outputs_dir)

    if configs_filter:
        odps = [o for o in odps if o.config in configs_filter]
    if scenarios_filter:
        odps = [o for o in odps if o.scenario_id in scenarios_filter]

    # Group by (config, scenario_id)
    odps_a = {(o.config, o.scenario_id): o for o in odps if o.model == model_a}
    odps_b = {(o.config, o.scenario_id): o for o in odps if o.model == model_b}
    common_keys = sorted(set(odps_a.keys()) & set(odps_b.keys()))

    print(f"[pipeline] Found {len(common_keys)} common (config, scenario) pairs\n")

    summary_rows: list[dict] = []
    total = len(common_keys)

    for idx, (config, scenario_id) in enumerate(common_keys, 1):
        scenario = scenarios.get(scenario_id)
        if scenario is None or scenario.ground_truth_text is None:
            continue

        result_path = (
            results_dir / "pairwise" / f"{model_a}_vs_{model_b}"
            / config / f"{scenario_id}.json"
        )
        if skip_existing and result_path.exists():
            print(f"  [{idx}/{total}] CACHED {config}/{scenario_id}")
            cached = json.loads(result_path.read_text(encoding="utf-8"))
            summary_rows.append(cached)
            continue

        print(f"  [{idx}/{total}] Pairwise {config}/{scenario_id} ...", end=" ", flush=True)

        try:
            result = judge.evaluate_pair(
                ground_truth=scenario.ground_truth_text,
                cq_list=scenario.cq_list,
                candidate_1=odps_a[(config, scenario_id)].ontology_text,
                candidate_2=odps_b[(config, scenario_id)].ontology_text,
            )

            row = {
                "model_a": model_a,
                "model_b": model_b,
                "config": config,
                "scenario_id": scenario_id,
                "judge_model": judge.model,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                f"{model_a}_score": result.candidate_1_score,
                f"{model_b}_score": result.candidate_2_score,
                **result.to_dict(),
            }

            result_path.parent.mkdir(parents=True, exist_ok=True)
            result_path.write_text(
                json.dumps(row, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            summary_rows.append(row)
            print(f"{model_a}={result.candidate_1_score} | {model_b}={result.candidate_2_score}")

        except Exception as e:
            print(f"ERROR: {e}")
            continue

    return summary_rows


# ─────────────────────────────────────────────────────────────────────────────
# CSV writer
# ─────────────────────────────────────────────────────────────────────────────

def write_summary_csv(rows: list[dict], path: Path) -> None:
    """Write a list of result dicts to a CSV file."""
    if not rows:
        print("[pipeline] No results to write.")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n[pipeline] Summary written to {path}  ({len(rows)} rows)")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="LLM-as-a-Judge evaluation pipeline for generated ODPs.",
    )
    p.add_argument(
        "--mode", choices=["single", "pairwise"], default="single",
        help="Evaluation mode (default: single).",
    )
    p.add_argument("--judge-model", default=None, help="Override JUDGE_MODEL from .env.")
    p.add_argument("--temperature", type=float, default=None, help="Override JUDGE_TEMPERATURE.")
    p.add_argument("--outputs-dir", type=Path, default=DEFAULT_OUTPUTS_DIR)
    p.add_argument("--data", type=Path, default=DEFAULT_SCENARIOS_PATH)
    p.add_argument("--ground-truth-dir", type=Path, default=DEFAULT_GT_DIR)
    p.add_argument(
        "--results-dir", type=Path,
        default=Path(__file__).resolve().parent.parent / "eval_judge_results",
    )
    p.add_argument("--configs", nargs="*", default=None, help="Filter to specific configs.")
    p.add_argument("--scenarios", nargs="*", default=None, help="Filter to specific scenario IDs.")
    p.add_argument("--no-skip-existing", action="store_true", help="Re-evaluate already cached results.")

    # Pairwise-specific
    p.add_argument("--model-a", default=None, help="First model for pairwise comparison.")
    p.add_argument("--model-b", default=None, help="Second model for pairwise comparison.")

    return p.parse_args()


def main() -> None:
    args = parse_args()

    judge = JudgeLLM(
        model=args.judge_model,
        temperature=args.temperature,
    )

    if args.mode == "single":
        rows = run_single_evaluation(
            judge=judge,
            scenarios_path=args.data,
            gt_dir=args.ground_truth_dir,
            outputs_dir=args.outputs_dir,
            results_dir=args.results_dir,
            configs_filter=args.configs,
            scenarios_filter=args.scenarios,
            skip_existing=not args.no_skip_existing,
        )
        write_summary_csv(
            rows,
            args.results_dir / "single_summary.csv",
        )

    elif args.mode == "pairwise":
        if not args.model_a or not args.model_b:
            print("ERROR: --model-a and --model-b required for pairwise mode.")
            sys.exit(1)
        rows = run_pairwise_evaluation(
            judge=judge,
            model_a=args.model_a,
            model_b=args.model_b,
            scenarios_path=args.data,
            gt_dir=args.ground_truth_dir,
            outputs_dir=args.outputs_dir,
            results_dir=args.results_dir,
            configs_filter=args.configs,
            scenarios_filter=args.scenarios,
            skip_existing=not args.no_skip_existing,
        )
        write_summary_csv(
            rows,
            args.results_dir / f"pairwise_{args.model_a}_vs_{args.model_b}.csv",
        )

    print("\n[pipeline] Done.")


if __name__ == "__main__":
    main()
