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
