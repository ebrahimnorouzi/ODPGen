#!/usr/bin/env bash

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -d "$ROOT_DIR/scripts" ]]; then
  cd "$ROOT_DIR"
elif [[ -d "$PWD/scripts" ]]; then
  ROOT_DIR="$PWD"
else
  echo "Could not find repository root containing scripts/. Run this script from the repo root or place it there."
  exit 1
fi

TEMPERATURE="${TEMPERATURE:-0}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-1024}"
LOG_DIR="${LOG_DIR:-logs}"
RUN_DOWNLOAD_GROUND_TRUTH="${RUN_DOWNLOAD_GROUND_TRUTH:-0}"
RUN_EVALUATION="${RUN_EVALUATION:-0}"

mkdir -p "$LOG_DIR"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
MASTER_LOG="$LOG_DIR/run_all_${TIMESTAMP}.log"
FAILED_LOG="$LOG_DIR/run_all_${TIMESTAMP}_failed.log"

CONFIGS=(
  "scenario-only"
  "cq-only"
  "scenario-cq"
  "scenario-cq-reasoning"
  "scenario-cq-constraints"
)

# Format per entry:
# backend|model|extra args...
MODEL_SPECS=(
  "openai|gpt-5.4|--fix-common-turtle-issues"
  "huggingface|meta-llama/Llama-3.1-8B-Instruct|--hf-token ${HF_TOKEN:-} --fix-common-turtle-issues"
  "huggingface|meta-llama/Llama-3.1-8B-Instruct|--quantize 4bit --hf-token ${HF_TOKEN:-} --fix-common-turtle-issues"
  "huggingface|meta-llama/Llama-2-70b-chat-hf|--quantize 4bit --hf-token ${HF_TOKEN:-} --fix-common-turtle-issues"
  "huggingface|mistralai/Mistral-7B-Instruct-v0.3|--fix-common-turtle-issues"
  "huggingface|bigscience/bloomz-7b1|--fix-common-turtle-issues"
)

log() {
  echo "[$(date '+%F %T')] $*" | tee -a "$MASTER_LOG"
}

run_cmd() {
  local backend="$1"
  local model="$2"
  local config="$3"
  shift 3
  local extra_args=("$@")

  local safe_model
  safe_model="$(echo "$model" | tr '/:' '__')"
  local run_log="$LOG_DIR/${TIMESTAMP}_${backend}_${safe_model}_${config}.log"

  log "START backend=$backend model=$model config=$config"

  if python3 scripts/run_generation.py \
      --backend "$backend" \
      --model "$model" \
      --config "$config" \
      --temperature "$TEMPERATURE" \
      --max-new-tokens "$MAX_NEW_TOKENS" \
      "${extra_args[@]}" \
      > >(tee -a "$run_log" "$MASTER_LOG") 2> >(tee -a "$run_log" "$MASTER_LOG" >&2); then
    log "DONE  backend=$backend model=$model config=$config"
  else
    log "FAIL  backend=$backend model=$model config=$config"
    echo "$backend|$model|$config" >> "$FAILED_LOG"
  fi
}

if [[ "$RUN_DOWNLOAD_GROUND_TRUTH" == "1" ]]; then
  log "Downloading ground truth ontologies"
  python3 scripts/download_ground_truth.py \
    > >(tee -a "$MASTER_LOG") 2> >(tee -a "$MASTER_LOG" >&2)
fi

for spec in "${MODEL_SPECS[@]}"; do
  IFS='|' read -r backend model extra <<< "$spec"

  if [[ "$backend" == "openai" && -z "${OPENAI_API_KEY:-}" ]]; then
    log "SKIP  backend=openai model=$model reason=OPENAI_API_KEY not set"
    continue
  fi

  if [[ "$backend" == "huggingface" && "$model" == meta-llama/* && -z "${HF_TOKEN:-}" ]]; then
    log "SKIP  backend=huggingface model=$model reason=HF_TOKEN not set for gated model"
    continue
  fi

  # Split the extra-args string into an array.
  read -r -a extra_args <<< "$extra"

  for config in "${CONFIGS[@]}"; do
    run_cmd "$backend" "$model" "$config" "${extra_args[@]}"
  done

done

if [[ "$RUN_EVALUATION" == "1" ]]; then
  log "Running evaluation pipeline"
  python3 scripts/evaluate_outputs.py \
    > >(tee -a "$MASTER_LOG") 2> >(tee -a "$MASTER_LOG" >&2)
  python3 scripts/check_unseen_feasibility.py \
    > >(tee -a "$MASTER_LOG") 2> >(tee -a "$MASTER_LOG" >&2)
  python3 scripts/generate_human_eval_sheet.py \
    > >(tee -a "$MASTER_LOG") 2> >(tee -a "$MASTER_LOG" >&2)
  python3 scripts/aggregate_human_scores.py \
    > >(tee -a "$MASTER_LOG") 2> >(tee -a "$MASTER_LOG" >&2)
fi

log "All requested runs finished."
if [[ -f "$FAILED_LOG" ]]; then
  log "Some runs failed. See: $FAILED_LOG"
else
  log "No failed runs recorded."
fi
