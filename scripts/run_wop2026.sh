#!/usr/bin/env bash
# Generate ODPs for the WOP 2026 extension (data/scenarios/pattern_scenarios_wop2026.json)
# with the models and prompting configurations of the paper.
#
#   GROUP=open nohup bash scripts/run_wop2026.sh > logs/wop2026_open.log 2>&1 &
#   GROUP=api  nohup bash scripts/run_wop2026.sh > logs/wop2026_api.log  2>&1 &
#
# Variables: GROUP=open|api|all (default all), TOK (max new tokens; default 1024 as in
# the paper, the generator recommends >= 4096), OUT (default outputs_wop2026),
# DATA, PYTHON (default python3), HF_TOKEN (needed for the gated Llama models).
# A failing model is logged and the next one still runs.
set -uo pipefail
cd "$(dirname "$0")/.."

DATA="${DATA:-data/scenarios/pattern_scenarios_wop2026.json}"
OUT="${OUT:-outputs_wop2026}"
TOK="${TOK:-1024}"
GROUP="${GROUP:-all}"
PY="${PYTHON:-python3}"
COMMON=(--data "$DATA" --outputs-dir "$OUT" --config all --temperature 0
        --max-new-tokens "$TOK" --fix-common-turtle-issues)
HF=()
[[ -n "${HF_TOKEN:-}" ]] && HF=(--hf-token "$HF_TOKEN")

stamp() { date '+%F %T'; }
run() {
  echo "[$(stamp)] START $*"
  if "$PY" scripts/run_generation.py "$@" "${COMMON[@]}"; then
    echo "[$(stamp)] DONE  $*"
  else
    echo "[$(stamp)] FAILED $*"
  fi
}

echo "[$(stamp)] data=$DATA out=$OUT max_new_tokens=$TOK group=$GROUP"
"$PY" -c "import json,sys; d=json.load(open(sys.argv[1])); print(len(d), 'scenarios:', ' '.join(x['scenario_id'] for x in d))" "$DATA"

# Reference ontologies (quick; skips files that are already there)
"$PY" scripts/download_ground_truth.py --data "$DATA"

if [[ "$GROUP" == open || "$GROUP" == all ]]; then
  # Refuse to start the open models on CPU or without 4-bit support.
  if ! "$PY" - <<'EOF'
import importlib.util, sys
import torch
cuda = torch.cuda.is_available()
bnb = importlib.util.find_spec("bitsandbytes") is not None
print(f"torch {torch.__version__} | CUDA available: {cuda} | GPUs: {torch.cuda.device_count()} | bitsandbytes: {bnb}")
sys.exit(0 if cuda and bnb else 1)
EOF
  then
    echo "[$(stamp)] PREFLIGHT FAILED: open models need a CUDA-enabled torch and bitsandbytes."
    [[ "$GROUP" == open ]] && exit 1
  else
    run --backend huggingface --model meta-llama/Llama-3.1-8B-Instruct --quantize 4bit "${HF[@]}"
    run --backend huggingface --model mistralai/Mistral-7B-Instruct-v0.3
    run --backend huggingface --model bigscience/bloomz-7b1
    run --backend huggingface --model meta-llama/Llama-2-70b-chat-hf --quantize 4bit "${HF[@]}"
  fi
fi

if [[ "$GROUP" == api || "$GROUP" == all ]]; then
  run --backend openai --model gpt-5.4
  run --backend gemini --model gemini-3.1-pro-preview
fi
echo "[$(stamp)] finished"
