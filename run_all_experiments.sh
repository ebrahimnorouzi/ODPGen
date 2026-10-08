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

# Output token budget.  Deliberately UNSET by default: scripts/run_generation.py
# owns the default (DEFAULT_MAX_NEW_TOKENS), and this driver used to pin it to
# 1024 and pass --max-new-tokens unconditionally, which made the Python default
# unreachable through the repository's own documented entry point -- every one
# of the 420 recorded runs was capped at 1024 tokens and truncated mid-answer.
# Leave it empty and the flag is not passed at all, so the two can never drift
# apart again.  Set MAX_NEW_TOKENS in the environment to override for a run.
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-}"

# ECHO_COMMANDS=1 prints the exact command line each run would execute and
# runs nothing.  Use it to verify what the driver actually passes.
ECHO_COMMANDS="${ECHO_COMMANDS:-0}"
LOG_DIR="${LOG_DIR:-logs}"
RUN_DOWNLOAD_GROUND_TRUTH="${RUN_DOWNLOAD_GROUND_TRUTH:-0}"
RUN_EVALUATION="${RUN_EVALUATION:-0}"

# ---------------------------------------------------------------------------
# Local, offline evaluation of the generated tree.
#
# RUN_LOCAL_EVAL=1 scores whatever is in OUTPUTS_DIR *on disk* and writes CSVs.
# It is a separate switch from RUN_EVALUATION on purpose: you almost always want
# to score the tree you just generated, and you can run it on its own with
#
#     RUN_LOCAL_EVAL=1 GENERATE=0 ./run_all_experiments.sh
#
# Both drivers below are offline and make ZERO network calls:
#   * batch_evaluate.py --local-root  walks OUTPUTS_DIR/**/ontology.ttl and adds
#     the truncation verdict to every record and aggregate.
#   * scripts/eval_local.py           the socket-guarded cross-check, which
#     records the unavailable OOPS! pitfall count as null rather than as zero.
# Running both and comparing them is the point; neither is a substitute.
# ---------------------------------------------------------------------------
GENERATE="${GENERATE:-1}"
RUN_LOCAL_EVAL="${RUN_LOCAL_EVAL:-0}"
OUTPUTS_DIR="${OUTPUTS_DIR:-outputs}"
BATCH_EVAL_OUT="${BATCH_EVAL_OUT:-odp_eval}"
EVAL_LOCAL_OUT="${EVAL_LOCAL_OUT:-eval_local}"
# RERUN_FAILED=1 adds a repair pass that re-scores only the failed artefacts.
RERUN_FAILED="${RERUN_FAILED:-0}"

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

  # Build the command once, then either show it or run it, so echo mode can
  # never disagree with what is actually executed.
  local cmd=(
    python3 scripts/run_generation.py
    --backend "$backend"
    --model "$model"
    --config "$config"
    --temperature "$TEMPERATURE"
  )
  if [[ -n "$MAX_NEW_TOKENS" ]]; then
    cmd+=(--max-new-tokens "$MAX_NEW_TOKENS")
  fi
  cmd+=("${extra_args[@]}")

  if [[ "$ECHO_COMMANDS" == "1" ]]; then
    printf '%s\n' "${cmd[*]}" | tee -a "$run_log" "$MASTER_LOG"
    return 0
  fi

  log "START backend=$backend model=$model config=$config"

  if "${cmd[@]}" \
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
  if [[ "$GENERATE" != "1" ]]; then
    break
  fi
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

if [[ "$RUN_LOCAL_EVAL" == "1" ]]; then
  if [[ ! -d "$OUTPUTS_DIR" ]]; then
    log "SKIP  local evaluation reason=$OUTPUTS_DIR does not exist"
  else
    n_onto="$(find "$OUTPUTS_DIR" -name ontology.ttl | wc -l | tr -d ' ')"
    log "Local evaluation: $n_onto ontology.ttl under $OUTPUTS_DIR (offline, no network)"

    # 1. batch_evaluate.py over the LOCAL tree.  --local-root is what makes a
    #    recovered or regenerated output scoreable at all: the default path
    #    enumerates the corpus over the GitHub API and can only ever see what is
    #    on the remote branch.  Writes per-ID JSON plus summary.csv (one row per
    #    artefact, with its truncation verdict) and aggregate.csv (one row per
    #    model/config, with n_truncated beside every mean).
    log "  -> batch_evaluate.py --local-root $OUTPUTS_DIR --out $BATCH_EVAL_OUT"
    python3 batch_evaluate.py --local-root "$OUTPUTS_DIR" --out "$BATCH_EVAL_OUT" \
      > >(tee -a "$MASTER_LOG") 2> >(tee -a "$MASTER_LOG" >&2)

    # 1b. The repair loop.  RERUN_FAILED=1 re-scores ONLY the artefacts whose
    #     record shows a parse error -- what you run after recover_outputs.py or
    #     a regeneration has changed files in the working tree.  summary.csv and
    #     aggregate.csv still cover the whole corpus: rows the pass did not touch
    #     are read back from $BATCH_EVAL_OUT and marked record_origin=carried_over,
    #     and run_manifest.json reports n_evaluated_this_run and n_carried_over
    #     beside n_files.  A partial pass can never shrink the scored corpus.
    if [[ "${RERUN_FAILED:-0}" == "1" ]]; then
      log "  -> batch_evaluate.py --local-root $OUTPUTS_DIR --out $BATCH_EVAL_OUT --rerun-failed"
      python3 batch_evaluate.py --local-root "$OUTPUTS_DIR" --out "$BATCH_EVAL_OUT" \
        --rerun-failed \
        > >(tee -a "$MASTER_LOG") 2> >(tee -a "$MASTER_LOG" >&2)
    fi

    # 2. scripts/eval_local.py -- the socket-guarded cross-check.  Same scoring
    #    functions, different bookkeeping: it records the OOPS! pitfall count as
    #    null (unavailable offline) instead of letting a missing scan read as
    #    "zero pitfalls found".
    log "  -> scripts/eval_local.py --outputs $OUTPUTS_DIR --out $EVAL_LOCAL_OUT"
    python3 scripts/eval_local.py --outputs "$OUTPUTS_DIR" --out "$EVAL_LOCAL_OUT" --quiet \
      > >(tee -a "$MASTER_LOG") 2> >(tee -a "$MASTER_LOG" >&2)

    log "Local evaluation CSVs: $BATCH_EVAL_OUT/summary.csv, $BATCH_EVAL_OUT/aggregate.csv, $EVAL_LOCAL_OUT/summary.csv"
  fi
fi

if [[ "$RUN_EVALUATION" == "1" ]]; then
  log "Running the legacy evaluation pipeline"
  for legacy in \
      scripts/evaluate_outputs.py \
      scripts/check_unseen_feasibility.py \
      scripts/generate_human_eval_sheet.py \
      scripts/aggregate_human_scores.py
  do
    # Three of these four have never existed in this repository, so the block
    # used to abort partway through.  Say which are missing instead.
    if [[ -f "$legacy" ]]; then
      log "  -> $legacy"
      python3 "$legacy" \
        > >(tee -a "$MASTER_LOG") 2> >(tee -a "$MASTER_LOG" >&2)
    else
      log "  SKIP $legacy (not present in this repository)"
    fi
  done
fi

log "All requested runs finished."
if [[ -f "$FAILED_LOG" ]]; then
  log "Some runs failed. See: $FAILED_LOG"
else
  log "No failed runs recorded."
fi
