#!/usr/bin/env bash
# ══════════════════════════════════════════════════════════════════════════════
# LLM-as-a-Judge Evaluation Pipeline — Run All Experiments
# ══════════════════════════════════════════════════════════════════════════════
#
# This script runs the complete evaluation pipeline:
#   Step 1: Single-candidate evaluation (every ODP against its ground truth)
#   Step 2: Pairwise evaluation (selected model pairs with position-swap debiasing)
#   Step 3: Aggregate results into summary tables and a Markdown report
#
# Prerequisites:
#   1. Copy .env.sample to .env and fill in your API keys
#   2. uv pip install -r requirements_judge.txt
#
# Usage:
#   chmod +x eval_judge/run_all_experiments.sh
#   ./eval_judge/run_all_experiments.sh
#
# Background with logging:
#   nohup ./eval_judge/run_all_experiments.sh > judge.log 2>&1 &
#
# ══════════════════════════════════════════════════════════════════════════════

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

# ── Load .env ────────────────────────────────────────────────────────────────
if [ ! -f ".env" ]; then
    echo "ERROR: .env file not found. Copy .env.sample to .env and fill in API keys."
    exit 1
fi

set -a
source .env
set +a

# ── Configuration ────────────────────────────────────────────────────────────
RESULTS_DIR="eval_judge_results"
JUDGE_MODEL="${JUDGE_MODEL:-gemini-2.5-flash}"

echo "══════════════════════════════════════════════════════════════════════"
echo " LLM-as-a-Judge Evaluation Pipeline"
echo " Judge model : $JUDGE_MODEL"
echo " Results dir : $RESULTS_DIR"
echo " Started     : $(date -Iseconds)"
echo "══════════════════════════════════════════════════════════════════════"
echo ""

# ── Create venv and install dependencies ─────────────────────────────────────
VENV_DIR="$PROJECT_ROOT/.venv_judge"
if [ ! -d "$VENV_DIR" ]; then
    echo " Creating virtual environment at $VENV_DIR ..."
    uv venv "$VENV_DIR" --python python3
fi
source "$VENV_DIR/bin/activate"
echo " Python: $(which python3) ($(python3 --version 2>&1))"
echo " Installing dependencies with uv..."
uv pip install -r requirements_judge.txt --quiet || {
    echo "ERROR: uv pip install failed. Ensure uv is installed: https://docs.astral.sh/uv/"
    exit 1
}

# Verify imports
python3 -c "import dotenv, google.genai" 2>/dev/null || {
    echo "ERROR: Import check failed."
    exit 1
}

# ══════════════════════════════════════════════════════════════════════════════
# STEP 1: Single-candidate evaluation
# ══════════════════════════════════════════════════════════════════════════════
echo "────────────────────────────────────────────────────────────────────"
echo " STEP 1: Single-candidate evaluation (all models × configs × scenarios)"
echo "────────────────────────────────────────────────────────────────────"
echo ""

python3 -m eval_judge.run_evaluation \
    --mode single \
    --judge-model "$JUDGE_MODEL" \
    --results-dir "$RESULTS_DIR"

echo ""
echo " Step 1 complete."
echo ""

# ══════════════════════════════════════════════════════════════════════════════
# STEP 2: Pairwise evaluation (selected model pairs)
# ══════════════════════════════════════════════════════════════════════════════
echo "────────────────────────────────────────────────────────────────────"
echo " STEP 2: Pairwise evaluation with position-swap debiasing"
echo "────────────────────────────────────────────────────────────────────"
echo ""

# Define the model pairs to compare pairwise.
# These are the most scientifically interesting comparisons:
#   - Large proprietary vs large proprietary
#   - Large proprietary vs best open-source
#   - Open-source vs open-source (size comparison)
PAIRS=(
    "gpt-5.4|gemini-3.1-pro-preview"
    "gpt-5.4|meta-llama_Llama-3.1-8B-Instruct"
    "gemini-3.1-pro-preview|meta-llama_Llama-3.1-8B-Instruct"
    "meta-llama_Llama-3.1-8B-Instruct|mistralai_Mistral-7B-Instruct-v0.3"
    "meta-llama_Llama-2-70b-chat-hf|meta-llama_Llama-3.1-8B-Instruct"
)

for pair in "${PAIRS[@]}"; do
    IFS='|' read -r MODEL_A MODEL_B <<< "$pair"
    echo "  Comparing: $MODEL_A  vs  $MODEL_B"
    python3 -m eval_judge.run_evaluation \
        --mode pairwise \
        --judge-model "$JUDGE_MODEL" \
        --model-a "$MODEL_A" \
        --model-b "$MODEL_B" \
        --results-dir "$RESULTS_DIR"
    echo ""
done

echo " Step 2 complete."
echo ""

# ══════════════════════════════════════════════════════════════════════════════
# STEP 3: Aggregate results
# ══════════════════════════════════════════════════════════════════════════════
echo "────────────────────────────────────────────────────────────────────"
echo " STEP 3: Aggregating results and generating report"
echo "────────────────────────────────────────────────────────────────────"
echo ""

python3 -m eval_judge.aggregate_results \
    --results-dir "$RESULTS_DIR"

echo ""
echo "══════════════════════════════════════════════════════════════════════"
echo " Pipeline complete!"
echo " Results   : $RESULTS_DIR/"
echo " Report    : $RESULTS_DIR/aggregated/judge_report.md"
echo " Finished  : $(date -Iseconds)"
echo "══════════════════════════════════════════════════════════════════════"
