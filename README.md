# ODPGen Experiment Bootstrap

This repository now includes executable scripts to run and evaluate the ODP + LLM research plan.

## Prompt/template contract

Prompt templates are stored in `prompts/` and must include both placeholders:
- `{{SCENARIO_TEXT}}`
- `{{CQ_LIST}}`

Use `scripts/render_prompt.py` for standalone prompt rendering.

## Data

- `data/odp_catalog.csv` — candidate patterns and publication metadata.
- `data/scenarios/pattern_scenarios.json` — ground-truth scenarios and competency questions (primary input to generation).

## Outputs

- Generated artifacts: `outputs/{model}/{config}/{scenario_id}/`
- Automatic evaluation: `results/summary.csv` and `results/{model}/{config}/*.json`
- Unseen-feasibility matrix: `results/unseen_feasibility.csv`
- Human evaluation sheets: `results/human_eval_sheet.csv`, `results/human_eval_summary.*`

## Installation

```bash
pip install -r requirements.txt
```

## Running generation

### GPT (OpenAI)

```bash
export OPENAI_API_KEY="sk-..."
python3 scripts/run_generation.py \
  --backend openai \
  --model gpt-5.4 \
  --config scenario-cq \
  --temperature 0  \
  --fix-common-turtle-issues
```

### Llama 3.1 8B Instruct (Hugging Face — recommended open-source)

Llama 3.1 8B fits on a single 24 GB GPU without quantization, or on 12 GB with `--quantize 4bit`.
It is instruction-tuned and follows structured output instructions reliably.
Request access at https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct.

```bash
export HF_TOKEN="hf_..."
python3 scripts/run_generation.py \
  --backend huggingface \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --config scenario-cq \
  --temperature 0 \
  --hf-token $HF_TOKEN \
  --fix-common-turtle-issues
```

With quantization for limited VRAM:

```bash
python3 scripts/run_generation.py \
  --backend huggingface \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --config scenario-cq \
  --temperature 0 \
  --quantize 4bit \
  --hf-token $HF_TOKEN \
  --fix-common-turtle-issues
```

### Llama 2 70B Chat (Hugging Face — large model, requires 2×A100 with 4-bit)

> **Note:** Llama 2 70B produces a systematic bug where standard prefixes are written as
> `:rdfs:label` instead of `rdfs:label`. The pipeline automatically corrects this for all
> HuggingFace backends. Prefer Llama 3.1 8B for single-GPU setups.

Llama 2 is a gated model. Request access at https://huggingface.co/meta-llama/Llama-2-70b-chat-hf.

```bash
export HF_TOKEN="hf_..."
python3 scripts/run_generation.py \
  --backend huggingface \
  --model meta-llama/Llama-2-70b-chat-hf \
  --config scenario-cq \
  --temperature 0 \
  --quantize 4bit \
  --hf-token $HF_TOKEN \
  --fix-common-turtle-issues
```

### Mistral 7B Instruct (Hugging Face — open access, fast)

```bash
python3 scripts/run_generation.py \
  --backend huggingface \
  --model mistralai/Mistral-7B-Instruct-v0.3 \
  --config scenario-cq \
  --temperature 0  \
  --fix-common-turtle-issues
```

### BLOOMZ 7B (Hugging Face — open access, instruction-tuned)

> **Note:** Use `bigscience/bloomz-7b1` (instruction-tuned), not `bigscience/bloom-*` (base models).
> Base BLOOM models repeat prompt text instead of generating Turtle.

```bash
python3 scripts/run_generation.py \
  --backend huggingface \
  --model bigscience/bloomz-7b1 \
  --config scenario-cq \
  --temperature 0  \
  --fix-common-turtle-issues
```

### Download ground truth ontologies (run once)

Downloads all 13 reference ontologies from the URLs in `pattern_scenarios.json`
into `data/ground_truth/`. Required before evaluation.

```bash
python3 scripts/download_ground_truth.py
```

### Post-generation evaluation (all models)

```bash
python3 scripts/evaluate_outputs.py
python3 scripts/check_unseen_feasibility.py
python3 scripts/generate_human_eval_sheet.py
python3 scripts/aggregate_human_scores.py
```

### Dry-run (render and save prompts without calling any model)

```bash
python3 scripts/run_generation.py \
  --backend openai \
  --model gpt-3.5-turbo \
  --config scenario-cq \
  --dry-run
```
