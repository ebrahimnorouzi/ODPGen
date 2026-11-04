#!/usr/bin/env bash
# LLM-as-a-judge evaluation of the WOP 2026 extension with the settings of the paper
# (Sec. "Reference-Based Evaluation with an LLM Judge"): GPT-4o as judge, temperature 0,
# single-candidate mode. GPT-5.4 is skipped automatically (same model family as the
# judge) and BLOOMZ-7B is excluded (it produces no Turtle).
#
#   nohup bash scripts/run_wop2026_judge.sh > logs/wop2026_judge.log 2>&1 &
#
# Safe to start while generation is still running: each judged ODP is cached under
# $RESULTS/<judge>/single/<model>/<config>/<scenario>.json, so running the script again
# later judges only outputs that are new or whose ontology.ttl changed since.
#
# Variables: JUDGE (default gpt-4o), OUT (outputs_wop2026), DATA, GT (data/ground_truth),
# RESULTS (eval_judge_results_wop2026), EXCLUDE (space-separated model directories,
# default bigscience_bloomz-7b1), PYTHON (python3).
# Needs the judge's API key (OPENAI_API_KEY for GPT judges; read from .env) and
# pip install -r requirements_judge.txt
set -uo pipefail
cd "$(dirname "$0")/.."

JUDGE="${JUDGE:-gpt-4o}"
OUT="${OUT:-outputs_wop2026}"
DATA="${DATA:-data/scenarios/pattern_scenarios_wop2026.json}"
GT="${GT:-data/ground_truth}"
RESULTS="${RESULTS:-eval_judge_results_wop2026}"
EXCLUDE="${EXCLUDE-bigscience_bloomz-7b1}"
PY="${PYTHON:-python3}"

stamp() { date '+%F %T'; }
echo "[$(stamp)] judge=$JUDGE outputs=$OUT data=$DATA results=$RESULTS exclude=${EXCLUDE:-none}"

# Dependencies and API key (run_evaluation.py loads .env itself; nothing is printed but the key name).
if ! "$PY" - "$JUDGE" <<'EOF'
import os, sys
from pathlib import Path
try:
    import dotenv
except ImportError:
    sys.exit("python-dotenv missing: pip install --user -r requirements_judge.txt")
dotenv.load_dotenv(Path(".env"))
judge = sys.argv[1].lower()
if "claude" in judge:
    mod, key = "anthropic", "ANTHROPIC_API_KEY"
elif "gemini" in judge:
    mod, key = "google.genai", "GOOGLE_API_KEY"
else:
    mod, key = "openai", "OPENAI_API_KEY"
try:
    __import__(mod)
except ImportError:
    sys.exit(f"{mod} missing: pip install --user -r requirements_judge.txt")
print(f"{key} set: {bool(os.environ.get(key))}")
sys.exit(0 if os.environ.get(key) else 1)
EOF
then
  echo "[$(stamp)] PREFLIGHT FAILED"
  exit 1
fi

# Generation status: how many non-empty ontologies each model has so far.
for m in "$OUT"/*/; do
  [[ -d "$m" ]] || continue
  echo "  $(basename "$m"): $(find "$m" -name ontology.ttl -size +0 | wc -l) non-empty / $(find "$m" -name ontology.ttl | wc -l) ontology.ttl"
done

EX=()
# shellcheck disable=SC2206  # word splitting of EXCLUDE is intended
[[ -n "$EXCLUDE" ]] && EX=(--exclude-models $EXCLUDE)

echo "[$(stamp)] START single-candidate evaluation"
"$PY" -m eval_judge.run_evaluation --mode single --judge-model "$JUDGE" --temperature 0 \
  --outputs-dir "$OUT" --data "$DATA" --ground-truth-dir "$GT" --results-dir "$RESULTS" \
  ${EX[@]+"${EX[@]}"} || echo "[$(stamp)] FAILED single-candidate evaluation"

echo "[$(stamp)] START aggregation"
"$PY" -m eval_judge.aggregate_results --results-dir "$RESULTS"

echo "[$(stamp)] finished; report: $RESULTS/${JUDGE//[\/:]/_}/aggregated/judge_report.md"
