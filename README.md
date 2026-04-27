# ODPGen — Generating Ontology Design Patterns with Large Language Models

This repository accompanies a study on the automatic generation and evaluation of Ontology Design Patterns (ODPs) using Large Language Models. It contains the code and intermediate artefacts needed to reproduce the experiments: generation, automatic structural and functional evaluation, an LLM-as-a-Judge pipeline, and the analysis of the user-based evaluation collected through a companion online platform.

The companion **ODP evaluation platform** (used to host the questionnaires shown to experts and students) is released separately at `<ANON-PLATFORM-REPO>`. Anonymised CSV exports from that platform are consumed by the analysis scripts in `odp-platform-results/`.

> The associated manuscript is currently under double-blind review. To preserve anonymity, all author-identifying URLs in this README have been replaced by `<ANON-...>` placeholders. The corresponding links will be filled in for the camera-ready version.

---

## Repository layout

```
ODPGen/
├── data/
│   ├── odp.csv                       # source-paper metadata for the dataset
│   ├── scenarios/pattern_scenarios.json  # 14 scenarios + competency questions
│   ├── ground_truth/                 # reference ontologies (downloaded on demand)
│   └── ontologies_retrived/          # additional retrieved ontologies
├── prompts/                          # the five prompting configurations
│   ├── cq_only.txt
│   ├── scenario_only.txt
│   ├── scenario_cq.txt
│   ├── scenario_cq_constraints.txt
│   └── scenario_cq_reasoning.txt
├── scripts/
│   ├── run_generation.py             # main generation entry point
│   ├── evaluate_outputs.py           # structural + functional evaluation
│   ├── download_ground_truth.py      # one-shot reference-ontology fetcher
│   ├── render_prompt.py              # standalone prompt-rendering helper
│   ├── curate_odp_window.py          # dataset curation utility
│   └── eval/                         # parsing, OOPS!, similarity, rdf utils
├── outputs/{model}/{config}/{scenario_id}/   # generated ODPs
├── results/{model}/{config}/                 # per-instance evaluation JSON
│   └── summary.csv                           # aggregate ranking
├── eval_judge/                       # LLM-as-a-Judge pipeline
│   ├── prompt_templates.py
│   ├── data_loader.py
│   ├── judge.py
│   ├── run_evaluation.py
│   ├── aggregate_results.py
│   └── run_all_experiments.sh
├── eval_judge_results/               # judge outputs (single + pairwise)
├── odp-platform-results/             # user-evaluation analysis pipeline
│   ├── analyze_user_eval.py          # Likert / IAA / per-pattern statistics
│   ├── theme_analysis.py             # thematic tagging of free-text comments
│   └── run.sh                        # one-shot runner
├── requirements.txt                  # generation + automatic evaluation
└── requirements_judge.txt            # LLM-as-a-Judge dependencies
```

---

## Installation

```bash
# Generation + automatic structural / functional evaluation
pip install -r requirements.txt

# Optional: LLM-as-a-Judge dependencies
pip install -r requirements_judge.txt
```

API keys are read from a `.env` file (a template is provided in `.env.sample`). The relevant variables are `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GEMINI_API_KEY`, and `HF_TOKEN`, depending on the models used.

---

## Reproducing the experiments

### 1. Prepare the reference ontologies

```bash
python3 scripts/download_ground_truth.py
```

This populates `data/ground_truth/` with the reference ontologies referenced from `pattern_scenarios.json`. It only needs to be run once.

### 2. Generate ODPs

The generation script supports OpenAI, Google Gemini, and any Hugging Face causal LM. The five prompting configurations (`cq-only`, `scenario-only`, `scenario-cq`, `scenario-cq-reasoning`, `scenario-cq-constraints`) are selected with `--config`.

```bash
# Example: GPT
python3 scripts/run_generation.py \
    --backend openai --model gpt-5.4 \
    --config scenario-cq --temperature 0 \
    --fix-common-turtle-issues

# Example: open-source via Hugging Face
python3 scripts/run_generation.py \
    --backend huggingface --model meta-llama/Llama-3.1-8B-Instruct \
    --config scenario-only --temperature 0 \
    --fix-common-turtle-issues

# 70B model with 4-bit quantisation
python3 scripts/run_generation.py \
    --backend huggingface --model meta-llama/Llama-2-70b-chat-hf \
    --config cq-only --temperature 0 \
    --quantize 4bit --fix-common-turtle-issues
```

A `--dry-run` flag renders the prompts without calling any model, useful for inspecting the inputs.

### 3. Automatic structural and functional evaluation

```bash
python3 scripts/evaluate_outputs.py
```

This runs OWL-RL consistency checks via `owlrl`, OOPS! pitfall detection, and CQ verification, and writes per-model/per-config JSON results into `results/`, plus the aggregate ranking in `results/summary.csv`.

### 4. LLM-as-a-Judge evaluation (optional)

`eval_judge/` provides a self-contained pipeline that scores generated ODPs along four semiotic dimensions (syntactic correctness, semantic accuracy, logical consistency, functional adequacy) using a judge LLM from a different model family than the generator. Pairwise comparisons are debiased by running each pair twice with the candidates swapped.

```bash
# Run the whole pipeline
chmod +x eval_judge/run_all_experiments.sh
./eval_judge/run_all_experiments.sh

# Or run individual steps
python -m eval_judge.run_evaluation --mode single
python -m eval_judge.run_evaluation --mode pairwise \
    --model-a gpt-5.4 --model-b gemini-3.1-pro-preview
python -m eval_judge.aggregate_results
```

Outputs land in `eval_judge_results/`.

### 5. User-based evaluation analysis

The questionnaires shown to expert and student evaluators were hosted on the companion ODP evaluation platform (`<ANON-PLATFORM-REPO>`). The platform exports one CSV file per evaluator pool. These CSVs are kept locally for analysis but are **not committed to this repository**, since they contain raw qualitative comments and evaluator tokens.

To regenerate the descriptive statistics, the inter-annotator agreement, and the thematic analysis of free-text comments:

```bash
bash odp-platform-results/run.sh
```

This runs `analyze_user_eval.py` (Likert summaries, Mann–Whitney U between tracks, Krippendorff's α, per-pattern means) and `theme_analysis.py` (regex-based tagging of recurring issues in free-text fields) and prints a console summary.

---

## Companion repository: ODP evaluation platform

The online platform that collected the user evaluations is released separately at `<ANON-PLATFORM-REPO>`. It is a containerised FastAPI application that hosts per-evaluator questionnaire forms, randomly assigns patterns, and exports the responses as CSV. The evaluator-form question schemas used in this study (expert and student) are also archived in that repository.

---

## Data note

- `data/scenarios/pattern_scenarios.json` is the primary input: 14 scenarios derived from 10 peer-reviewed ODP papers, each with its competency questions and reference ontology URL.
- Generated outputs (`outputs/`) and automatic-evaluation results (`results/`) are committed so reviewers can inspect the experimental artefacts without re-running every model.
- Raw user-evaluation CSV exports (`odp-platform-results/*.csv`) are gitignored, since they contain free-text comments and evaluator tokens. They will be released alongside the camera-ready version of the manuscript.

---

## Citation

A citation entry will be added once the manuscript is accepted. In the meantime, please refer to this repository and the companion evaluation platform when reusing the code or the dataset.
