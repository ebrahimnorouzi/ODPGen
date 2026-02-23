# ODPGen Experiment Bootstrap

This repository now includes executable scripts to run and evaluate the ODP + LLM research plan.

## Quickstart

```bash
python3 scripts/curate_odp_window.py --after 2023-01-01
python3 scripts/run_generation.py --config all --model mock-odp-generator
python3 scripts/evaluate_outputs.py
python3 scripts/check_unseen_feasibility.py
python3 scripts/generate_human_eval_sheet.py
python3 scripts/aggregate_human_scores.py
```

## Prompt/template contract

Prompt templates are stored in `prompts/` and must include both placeholders:
- `{{SCENARIO_TEXT}}`
- `{{CQ_LIST}}`

Use `scripts/render_prompt.py` for standalone prompt rendering.

## Data fallback strategy

If no curated ODP dataset is available, use:
- `data/odp_catalog.csv` for candidate patterns and publication metadata.
- `data/scenarios/pattern_scenarios.json` for scenario + CQ seeds.

These are bootstrapped candidates and should be replaced with verified pattern sources before publication.

## Outputs

- Generated artifacts: `outputs/{model}/{config}/{scenario_id}/`
- Automatic evaluation: `results/summary.csv` and `results/{model}/{config}/*.json`
- Unseen-feasibility matrix: `results/unseen_feasibility.csv`
- Human evaluation sheets: `results/human_eval_sheet.csv`, `results/human_eval_summary.*`

## Re-running for a PR refresh

If you need to re-submit the same PR after review comments, re-run the pipeline to refresh generated
artifacts and then commit any updated files:

```bash
python3 scripts/curate_odp_window.py --after 2023-01-01
python3 scripts/run_generation.py --config all --model mock-odp-generator
python3 scripts/evaluate_outputs.py
python3 scripts/check_unseen_feasibility.py
python3 scripts/generate_human_eval_sheet.py
python3 scripts/aggregate_human_scores.py
```

Note: the default generator is deterministic, so unchanged inputs produce stable outputs.

## Using open-source Hugging Face models (e.g., Llama)

You can switch from the built-in mock generator to open-source models hosted on Hugging Face:

```bash
pip install transformers torch
python3 scripts/run_generation.py \
  --backend huggingface \
  --model meta-llama/Llama-3.2-3B-Instruct \
  --config scenario-cq \
  --temperature 0.2 \
  --max-new-tokens 900
```

Notes:
- `--backend mock` (default) keeps the deterministic local mock output.
- `--backend huggingface` calls `transformers` text-generation pipeline with your selected model.
- Choose a model that fits your local GPU/CPU memory budget.
