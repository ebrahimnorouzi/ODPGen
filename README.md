# ODPGen Prompt Templates

This repository standardizes prompt assembly through fixed templates in `prompts/` and a renderer script in `scripts/render_prompt.py`.

## Templates and required input fields

Each template has a strict contract. Required JSON keys must be present and non-empty.

- `prompts/scenario_only.txt`
  - Required keys: `SCENARIO_TEXT`
- `prompts/cq_only.txt`
  - Required keys: `CQ_LIST`
- `prompts/scenario_cq.txt`
  - Required keys: `SCENARIO_TEXT`, `CQ_LIST`
- `prompts/scenario_cq_reasoning.txt`
  - Required keys: `SCENARIO_TEXT`, `CQ_LIST`, `REASONING_GUIDANCE`
- `prompts/scenario_cq_constraints.txt`
  - Required keys: `SCENARIO_TEXT`, `CQ_LIST`, `OUTPUT_CONSTRAINTS`

## Renderer usage

```bash
python3 scripts/render_prompt.py \
  --template prompts/scenario_cq.txt \
  --input input.json \
  --output rendered_prompt.txt
```

If `--output` is omitted, the rendered prompt is written to stdout.

## Input format

`input.json` must be a JSON object:

```json
{
  "SCENARIO_TEXT": "...",
  "CQ_LIST": "- CQ1\n- CQ2",
  "REASONING_GUIDANCE": "Explain assumptions before conclusions.",
  "OUTPUT_CONSTRAINTS": "Use exactly 3 bullet points."
}
```

Only keys required by the selected template are enforced. Extra keys are ignored.

## Validation behavior (fail fast)

The renderer exits with an error when:

1. The template name is unsupported.
2. A required placeholder is missing from the template file itself.
3. A required input field is absent or blank in `input.json`.
4. The input file is missing or contains invalid JSON.

This contract ensures identical output instructions across runs and prevents format drift from ad-hoc prompt changes.
