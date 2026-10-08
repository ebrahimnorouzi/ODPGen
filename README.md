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
├── batch_evaluate.py                 # scorer; `--local-root` evaluates outputs/
├── run_all_experiments.sh            # generation + local evaluation driver
├── scripts/
│   ├── run_generation.py             # main generation entry point
│   ├── eval_local.py                 # offline evaluation of the local tree
│   ├── evaluate_outputs.py           # structural + functional evaluation
│   ├── download_ground_truth.py      # one-shot reference-ontology fetcher
│   ├── render_prompt.py              # standalone prompt-rendering helper
│   ├── curate_odp_window.py          # dataset curation utility
│   └── eval/                         # parsing, OOPS!, similarity, rdf utils
├── outputs/{model}/{config}/{scenario_id}/   # generated ODPs
├── odp_eval/                                 # batch_evaluate.py --local-root output
│   ├── summary.csv                           # one row per artefact
│   └── aggregate.csv                         # one row per model/config
├── eval_local/                               # scripts/eval_local.py output
│   └── summary.csv
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

**From `outputs/` to a scored CSV, offline, in one command:**

```bash
python3 batch_evaluate.py --local-root outputs --out odp_eval
```

This walks `outputs/**/ontology.ttl` **on disk**, runs OWL-RL consistency checks via `owlrl` and offline CQ verification (SEV), and writes:

| File | Contents |
| --- | --- |
| `odp_eval/{model}/{config}/{id}.json` | the full per-artefact record |
| `odp_eval/summary.csv` | one row per artefact |
| `odp_eval/aggregate.csv` | one row per (model, config) |
| `odp_eval/run_manifest.json` | corpus size, status counts, truncation split, refused network calls |

`--local-root` is the path to use. Without it, `batch_evaluate.py` enumerates the corpus over the GitHub API (`https://api.github.com/repos/.../git/trees/<branch>`) and fetches every file from `raw.githubusercontent.com`. That remote path still works, but it can only ever see what is committed on the remote branch, so **an output you have just generated, recovered, or regenerated in your working tree cannot be scored through it.** The local path also makes the run offline: a socket guard is armed for its duration and the number of refused connections is printed (it should be `0`).

A second, independent offline driver cross-checks the same tree:

```bash
python3 scripts/eval_local.py --outputs outputs --out eval_local
```

`scripts/eval_local.py` scores through the same `batch_evaluate` functions but keeps stricter bookkeeping around unavailable measurements: the OOPS! pitfall count needs a web service, so offline it is recorded as `null` and **excluded** from the structural score rather than read as "zero pitfalls found". Its results land in `eval_local/summary.csv` and `eval_local/aggregate.csv`, each aggregate stating the denominator its mean was taken over.

Both drivers are wired into `run_all_experiments.sh`:

```bash
# score the existing outputs/ tree without regenerating anything
RUN_LOCAL_EVAL=1 GENERATE=0 ./run_all_experiments.sh
```

#### Truncated generations are flagged, not hidden

`scripts/run_generation.py` detects a generation that ran out of output budget and records it three ways: `"truncated"` / `"truncation_signals"` in `metadata.json`, a `# ODPGEN-TRUNCATED` banner prepended to `ontology.ttl`, and — for runs recorded before the flag existed — the unterminated code fence still present in `raw_response.txt`. 160 of the 420 committed responses carry such a signal, all of them from the 1024-token cap that the generation driver used to pin.

The evaluation reads all three. Every per-artefact record carries `truncated` (`true` / `false` / `null` when no evidence survives) and a `truncation` block naming the evidence it used, and every aggregate reports:

- `n_truncated`, `n_complete`, `n_truncation_unknown` — which always sum to `n_files`;
- `mean_structural_score` / `mean_sev_score` over the **whole** group, with `structural_denominator` / `sev_denominator`;
- `mean_structural_score_complete` / `mean_sev_score_complete` over the artefacts known to be complete, with their own denominators.

Truncated artefacts are never removed from a denominator. To report on complete generations only, use the `_complete` columns, which state exactly how much of the corpus they cover. `truncated = null` means *unknown*, and is counted separately from *complete* on purpose.

#### Repairing an artefact and re-scoring only it

The reason `--local-root` exists is that a repaired output lives in the working
tree. The loop is:

```bash
python3 batch_evaluate.py --local-root outputs --out odp_eval              # score everything
python3 scripts/recover_outputs.py --apply                                 # repair what can be repaired
python3 batch_evaluate.py --local-root outputs --out odp_eval --rerun-failed
```

The third command re-scores only the artefacts whose existing record shows a
parse error. `summary.csv`, `aggregate.csv` and `run_manifest.json` still
describe the **whole** corpus: every record the pass did not touch is read back
from `odp_eval/` and carried, and each row says which it was in a
`record_origin` column (`this_run` / `carried_over`). `run_manifest.json` adds
`n_evaluated_this_run`, `n_carried_over` and `n_unreadable_records` beside
`n_files`, so a partial pass can never be mistaken for a full one — and can
never silently shrink the scored corpus. `--patch-oops` rebuilds the two CSVs
the same way, so they always agree with the JSON records beside them.

#### `--oops-url` with `--local-root`

The socket guard armed by `--local-root` refuses connections **the run was not
given**. An OOPS! endpoint passed explicitly with `--oops-url` is resolved
before the guard is armed and allowed through; `run_manifest.json` reports
`network_calls` (refused) and `network_allowed_calls` (permitted) separately,
alongside the `network_allowlist` it honoured. Without `--oops-url` the
allowlist is empty and `network_calls` should read `0`.

#### Legacy driver

```bash
python3 scripts/evaluate_outputs.py
```

The original driver, kept for the artefacts already in `results/`. It writes per-model/per-config JSON into `results/` plus the aggregate ranking in `results/summary.csv`, and does not carry the truncation verdict.

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
