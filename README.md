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

---

## LLM-as-a-Judge Evaluation Pipeline

An automated evaluation pipeline that uses a judge LLM to score generated ODPs against ground-truth ontologies. The judge is always a different model family than the generator to prevent self-evaluation bias.

### Evaluation Criteria (Rubric)

Each generated ODP is scored on four semiotic dimensions (1–10 scale):

| Dimension | What it measures |
|---|---|
| **Syntactic Correctness** | Valid OWL 2 DL / RDF Turtle syntax, well-formed triples, declared prefixes |
| **Semantic Accuracy** | Concept precision and recall against ground truth (synonyms allowed) |
| **Logical Consistency** | No unsatisfiable classes, contradictory axioms, or domain/range violations |
| **Functional Adequacy** | Structural capacity to answer the scenario's Competency Questions |

Score bands:
- **1–2**: Syntactically invalid or logically incoherent. Fails all CQs.
- **3–4**: Major semantic drift, missing core classes, severe logical errors.
- **5–6**: Basic taxonomy captured but complex restrictions missing. Fails complex CQs.
- **7–8**: Logically sound, highly similar semantics. Answers almost all CQs.
- **9–10**: Semantically isomorphic to ground truth. Flawless logic, all CQs satisfied.

### Bias Mitigation

**Positional bias** is mitigated in pairwise evaluations by running each comparison twice with candidates swapped (A↔B), then averaging the scores.

**Judge independence** is enforced automatically: the pipeline detects the generating model's family and skips evaluation if the judge belongs to the same family.

### Pipeline Structure

```
eval_judge/
├── __init__.py
├── prompt_templates.py        # G-Eval CoT prompts (single + pairwise)
├── data_loader.py             # Load ground truth, CQs, generated outputs
├── judge.py                   # Core engine: API calls, position-swap debiasing
├── run_evaluation.py          # Main orchestrator (single + pairwise modes)
├── aggregate_results.py       # Summary stats, cross-tabs, Markdown report
└── run_all_experiments.sh     # One-shot shell script to run everything

.env.sample                    # Template for API keys
requirements_judge.txt         # Dependencies: anthropic, openai, python-dotenv
```

### Quick Start

```bash
# 1. Install dependencies
pip install -r requirements_judge.txt

# 2. Configure API keys
cp .env.sample .env
# Edit .env — set ANTHROPIC_API_KEY (and/or OPENAI_API_KEY)

# 3. Run the full pipeline (single eval + pairwise + aggregation)
chmod +x eval_judge/run_all_experiments.sh
./eval_judge/run_all_experiments.sh
```

### Running Individual Steps

```bash
# Single-candidate evaluation (each ODP scored against its ground truth)
python -m eval_judge.run_evaluation --mode single

# Override the judge model
python -m eval_judge.run_evaluation --mode single --judge-model claude-opus-4-6

# Filter to specific configs or scenarios
python -m eval_judge.run_evaluation --mode single \
    --configs cq-only scenario-cq \
    --scenarios 2023-133-01 2023-134-01

# Pairwise evaluation between two models (with position-swap debiasing)
python -m eval_judge.run_evaluation --mode pairwise \
    --model-a gpt-5.4 --model-b gemini-3.1-pro-preview

# Aggregate results into summary tables and Markdown report
python -m eval_judge.aggregate_results
```

### Output

Results are written to `eval_judge_results/`:

```
eval_judge_results/
├── single/                         # Per-instance JSON results
│   └── {model}/{config}/{scenario_id}.json
├── pairwise/                       # Pairwise comparison results
│   └── {modelA}_vs_{modelB}/{config}/{scenario_id}.json
├── single_summary.csv              # Flat CSV of all single evaluations
├── pairwise_*.csv                  # Flat CSV per model pair
└── aggregated/
    ├── by_model.csv                # Mean scores grouped by model
    ├── by_config.csv               # Mean scores grouped by prompt config
    ├── by_scenario.csv             # Mean scores grouped by scenario
    └── judge_report.md             # Full Markdown summary report
```

Results are cached — re-running skips already-evaluated instances. Use `--no-skip-existing` to force re-evaluation.
