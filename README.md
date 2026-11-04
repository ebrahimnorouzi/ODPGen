# ODPGen — Generating Ontology Design Patterns with Large Language Models

This repository accompanies the article *Generating Ontology Design Patterns with LLMs: An Empirical Evaluation on Recently Published Patterns* by Anna Sofia Lippolis, Ebrahim Norouzi, Andrea Giovanni Nuzzolese, Harald Sack and Aldo Gangemi. It contains the code, data and intermediate artefacts needed to reproduce the experiments: generation, automatic structural and functional evaluation, an LLM-as-a-Judge pipeline, and the analysis of the user-based evaluation collected through a companion online platform.

The companion **ODP evaluation platform**, which hosted the questionnaires shown to experts, students and pattern authors, is released at <https://github.com/ebrahimnorouzi/odp-platform>. The deployments used in the study were <http://odp.ise.fiz-karlsruhe.de/evaluate?s=z9IJcI1hs2vzhg> (experts) and <http://odp.ise.fiz-karlsruhe.de/evaluate?s=zT_6wILobV_nug> (students). Pseudonymised CSV exports from the platform are analysed by the scripts in `odp-platform-results/`.

---

## Repository layout

```text
ODPGen/
├── data/
│   ├── odp.csv                           # surveyed WOP papers (2023–2026) and inclusion notes
│   ├── scenarios/
│   │   ├── pattern_scenarios.json        # the 14 scenarios + CQs used in the paper
│   │   ├── pattern_scenarios_wop2026.json  # WOP 2026 held-out extension (see below)
│   │   ├── cq_signatures.json            # CQ signatures used by offline CQ verification
│   │   └── cq_audit_report.json          # audit of CQ authenticity (scripts/audit_cqs.py)
│   ├── ground_truth/                     # reference ODPs + per-scenario fetch status
│   └── ontologies_retrived/              # reference ODPs recovered by hand
├── prompts/                              # the five prompting configurations
├── scripts/
│   ├── run_generation.py                 # main generation entry point
│   ├── download_ground_truth.py          # one-shot reference-ontology fetcher
│   ├── aggregate_eval.py                 # writes eval/*.csv (structural + functional tables)
│   ├── eval_local.py                     # offline evaluation of the local tree
│   ├── evaluate_outputs.py               # legacy structural + functional driver
│   ├── odp_structural.py                 # structural metrics
│   ├── recover_outputs.py                # repairs truncated/broken generations
│   ├── audit_artifacts.py, audit_cqs.py  # corpus and CQ audits
│   ├── render_prompt.py, curate_odp_window.py
│   └── eval/                             # parsing, OOPS!, similarity, rdf utils
├── batch_evaluate.py                     # scorer; `--local-root` evaluates outputs/
├── run_all_experiments.sh                # generation + local evaluation driver
├── outputs/{model}/{config}/{scenario_id}/   # generated ODPs (420 artefacts)
├── eval/                                 # aggregate rankings behind the paper's tables
├── eval_results/odp_eval/                # per-artefact records (+ manual evaluation)
├── eval_local/, eval_local_post/         # scripts/eval_local.py output (before/after recovery)
├── eval_connectivity/                    # class/property connectivity per output
├── results/{model}/{config}/             # legacy per-instance evaluation JSON
├── eval_judge/                           # LLM-as-a-Judge pipeline
├── eval_judge_results/                   # judge outputs (GPT-4o; single + pairwise)
├── outputs_wop2026/, eval_results_wop2026/, eval_judge_results_wop2026/, eval_wop2026/
│                                         # WOP 2026 extension: ODPs, structural records, judge, tables
├── odp-platform-results/                 # user evaluation: pseudonymised exports + analysis
│   ├── expert-evaluation-responses.csv, us-responses.csv, students-responses.csv,
│   │   author_<scenario>-responses.csv
│   ├── analyze_user_eval.py              # Likert, Wilcoxon, Krippendorff's α, per-pattern
│   ├── analyze_author_eval.py            # pattern-author assessment of their own ODP
│   ├── analyze_students_eval.py          # student report (supplementary/)
│   ├── theme_analysis.py                 # thematic tagging of free-text comments
│   └── run.sh                            # one-shot runner
├── supplementary/student_odp_platform.pdf  # detailed student results
├── tests/                                # pytest suite for parsing and evaluation
├── audit_report.json, recovery_report.json,
│   truncated_cells.txt, remaining_cells.txt  # provenance of the truncation audit and recovery
├── requirements.txt, requirements_judge.txt
├── LICENSE, CITATION.cff
```

---

## From the paper to the files

| Paper | Source in this repository | How to regenerate |
| --- | --- | --- |
| Table 1 (dataset) | `data/odp.csv`, `data/scenarios/pattern_scenarios.json` | — |
| Table 2 (structural evaluation) | `eval/odp_eval_ranking.csv`, per artefact `eval/odp_eval_results.csv` | `python3 scripts/aggregate_eval.py --in eval_results --out eval` (add `--compare eval` to check) |
| Table 3 (functional evaluation) | `eval/functional_eval_ranking.csv`, per ODP `eval/functional_eval_detail.csv`; manual superfluous-element annotations in `eval_results/odp_eval/bigscience_bloomz-7b1/manualeval.csv` | `python3 scripts/functional_score.py --compare` (see note below) |
| Tables 4–6 (expert IAA, Likert, per pattern), adoption and time saved | `odp-platform-results/*.csv` | `bash odp-platform-results/run.sh` |
| Pattern-author and student results | `odp-platform-results/author_*.csv`, `students-responses.csv`, `supplementary/student_odp_platform.pdf` | `bash odp-platform-results/run.sh` |
| Recurring issues in comments (Recommendations) | `odp-platform-results/*.csv` | `python3 odp-platform-results/theme_analysis.py` |
| Tables 7–8 (LLM judge) | `eval_judge_results/single_summary.csv` | `python -m eval_judge.aggregate_results` (writes `eval_judge_results/aggregated/by_model.csv`, `by_config.csv`) |
| Table 9 (WOP 2026, structural) | `eval_results_wop2026/odp_eval/`, `eval_wop2026/structural_by_model.csv` | `python3 scripts/structural_table.py --in eval_results_wop2026/odp_eval` |
| Table 10 (WOP 2026, LLM judge) | `eval_judge_results_wop2026/gpt-4o/` | `bash scripts/run_wop2026_judge.sh` |
| Generated ODPs and prompts | `outputs/`, `prompts/` | `scripts/run_generation.py` |

**Structural table.** `scripts/structural_table.py` prints Table 2 from the per-artefact records: consistency is the share of all outputs that parse and pass OWL-RL screening, and the OOPS! column is the mean, over the outputs OOPS! scanned, of `max(0, 1 − (important + 2 · critical) / 10)`. `python3 scripts/structural_table.py --in eval_results/odp_eval --count-empty-as-parsed` reproduces the Consistency, OOPS! and Overall columns of Table 2 exactly. The same records give parse rates equal to the consistency rates (every output that parses is consistent), not the 100 % the published Parse column shows for the Llama models and Mistral. `scripts/aggregate_eval.py` writes a different OOPS! component, `1 / (1 + important + 2 · critical)`, averaged over all outputs.

**Functional score.** `scripts/functional_score.py` implements the formula of the paper (superfluous penalty `1.0 · numbnoscen + 0.3 · (numb − numbnoscen)`, manual score `1 / (1 + 0.5 · penalty)`, functional score `0.5 · manual + 0.5 · CQ rate`) and reproduces both committed CSVs exactly. The `cq_pass_rate` column of those CSVs is the SPARQL pass rate of the offline CQ verification; Table 3 of the paper combines the same manual scores with the Yes/No labels of the LLM-based CQ verification (`gpt-4o-mini`), e.g. `python3 scripts/functional_score.py --cq-rate llama70b=0.916 --cq-rate gemini=0.935 --cq-rate gpt=0.738 --cq-rate llama8b=0.750`.

---

## Models

| Role | Model (as named in `outputs/`) |
| --- | --- |
| Open generators | `bigscience/bloomz-7b1`, `meta-llama/Llama-2-70b-chat-hf`, `meta-llama/Llama-3.1-8B-Instruct`, `mistralai/Mistral-7B-Instruct-v0.3` |
| Proprietary reference generators | `gemini-3.1-pro-preview`, `gpt-5.4` |
| CQ verification | `gpt-4o-mini` |
| LLM judge | `gpt-4o` (evaluations of the GPT family are skipped) |

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

The questionnaires shown to experts, students and pattern authors were hosted on the companion ODP evaluation platform (<https://github.com/ebrahimnorouzi/odp-platform>). The platform exports one CSV file per evaluator pool; these exports are committed in `odp-platform-results/`:

| File | Pool |
| --- | --- |
| `expert-evaluation-responses.csv` | external experts |
| `us-responses.csv` | the two paper authors who evaluated all patterns with the expert form |
| `students-responses.csv` | students |
| `author_<scenario>-responses.csv` | pattern authors (their own pattern plus one extra pattern) |

Evaluators are identified only by a pseudonymous platform token (`evaluator_token`, `session_id`); the files contain no names or e-mail addresses. Free-text answers are kept verbatim.

To regenerate every user-evaluation statistic reported in the paper:

```bash
bash odp-platform-results/run.sh
```

This runs `analyze_user_eval.py` (Likert summaries, paired Wilcoxon signed-rank tests with rank-biserial effect sizes, Krippendorff's α, per-pattern means, adoption and time saved), `analyze_author_eval.py`, `analyze_students_eval.py` and `theme_analysis.py` (regex-based tagging of recurring issues in free-text fields). The LaTeX tables are written to `odp-platform-results/latex/` (set `ODPGEN_LATEX_DIR` to write into a paper checkout instead) and are identical to the ones in the paper.

---

## Companion repository: ODP evaluation platform

The online platform that collected the user evaluations is released separately at <https://github.com/ebrahimnorouzi/odp-platform>. It is a containerised FastAPI application that hosts per-evaluator questionnaire forms, randomly assigns patterns, and exports the responses as CSV. The evaluator-form question schemas used in this study (expert and student) are also archived in that repository.

---

## Data note

- `data/scenarios/pattern_scenarios.json` is the primary input: 14 scenarios derived from 10 peer-reviewed ODP papers (WOP 2023–2025 and the Cultural Heritage Survey ODP, SemDH 2026, CEUR-WS Vol-4212), each with its competency questions and reference ontology URL. It is the single source of truth for the CQs used in scoring.
- Generated outputs (`outputs/`) and automatic-evaluation results (`eval/`, `eval_results/`, `eval_local*/`, `results/`) are committed so reviewers can inspect the experimental artefacts without re-running every model.
- The pseudonymised user-evaluation exports are committed in `odp-platform-results/` (see above).
- `audit_report.json`, `recovery_report.json`, `truncated_cells.txt` and `remaining_cells.txt` record the audit of truncated generations and the recovery pass (`scripts/audit_artifacts.py`, `scripts/recover_outputs.py`).

---

## Dataset extension: WOP 2026 (held out, automatic evaluation only)

The 14 papers of the [WOP 2026 workshop](https://odpa.github.io/workshop-on-ontology-design-and-patterns/2026/) (ISWC 2026, Bari) were screened with the same inclusion criteria as the original dataset: the paper proposes or discusses an ODP; the pattern is publicly available; and the paper reports both user stories (or an equivalent use-case narrative) and CQs. All 14 are listed in `data/odp.csv`.

Items that meet the criteria and have been extracted are in `data/scenarios/pattern_scenarios_wop2026.json`, in the same format as `pattern_scenarios.json`. Scenario text and CQs are verbatim from the paper or the CQ file it links to. Each reference ontology was downloaded and parsed with rdflib.

| Scenario | Paper | CQs | Reference ontology |
| --- | --- | --- | --- |
| `2026-w01-01` | LCAnet: A Modular Ontology Network for Life Cycle Assessment Knowledge Modeling | 10 | Core module `core/ontology.ttl`, which imports the five other modules |
| `2026-w04-01` | MAGE-Ontology: A Domain Ontology for Ghidra Malware Analysis | 10 | `symbol-ontology.owl` (Turtle), pinned commit |
| `2026-w05-01` | PEO: An Ontology to Scaffold Prompt Engineering for Foundation Models | 16 | `rbarile17/peo` `peo.owl` |
| `2026-w06-01` | Modeling User Requirements for Inclusive Pedestrian Routing with the PIIRRITE Ontology | 29 | `modelet_5/TBox.ttl` (modelets 1–5) |
| `2026-w07-01` | APLO: A Modular Adaptive Pedagogical Learning Orchestration Ontology for Neurosymbolic Educational Systems | 94 | `schema/full-schema.owl` (Turtle; all four modules) |
| `2026-w09-01` | Formalizing Data Structures Without Overdoing It (HACID data ontology) | 37 | `data.owl`, pinned to the commit matching the paper |
| `2026-w10-01` | An Ontology Design Pattern for Software Requirements | 5 | ODPA `SoftwareRequirements.ttl` |
| `2026-w11-01` | WISDOM: A Wearable Integrated Stress Data Ontology Model | 41 | `WISDOM.owl` |

In total there are 8 items and 242 CQs.

Notes on the extraction:

- **CQs from linked CQ files.** Where the paper prints only a subset and links the full list (PIIRRITE, APLO, WISDOM, HACID), the CQs come from that file. HACID's CQ file is pinned to the commit with the 37 CQs the paper describes.
- **MAGE-Ontology scenario (borderline).** The scenario comes from the repository's `documentation/use-case.md`, which is the paper's supplementary material. The paper itself gives only a one-sentence use case.
- **ArCo4Archives excluded.** Its 8 CQs were "elicited from the scenario", but the paper never states that scenario.

All 8 reference ontologies download and parse with `scripts/download_ground_truth.py`.

These items keep the leakage control of the study: they were published after the training cutoffs of all four open models. The paper reports them as a held-out extension (Section 7, Tables 9–10): structural screening and the LLM judge only, since the superfluous-element annotation and the user evaluation were not repeated. To generate them:

```bash
mkdir -p logs
GROUP=open nohup bash scripts/run_wop2026.sh > logs/wop2026_open.log 2>&1 &   # GPU: 4 open models
GROUP=api  nohup bash scripts/run_wop2026.sh > logs/wop2026_api.log  2>&1 &   # GPT-5.4, Gemini
```

`scripts/run_wop2026.sh` downloads the reference ontologies, checks for a CUDA-enabled PyTorch and `bitsandbytes` before it loads any open model, and writes to `outputs_wop2026/`. It uses the paper's settings by default (1024-token cap; both Llama models in 4-bit); set `TOK=4096` to raise the cap. APLO (94 CQs), WISDOM (41) and HACID (37) are larger than any item in the paper, so expect truncation at that cap. The committed `outputs_wop2026/` was generated with `TOK=4096` (the generator raises Gemini to 8192); 70 of the 240 outputs are still flagged as truncated (39 with a cap hit reported by the model, the rest by an unclosed code block or an incomplete final statement).

**Structural screening.** `batch_evaluate.py` reads CQs only from `data/scenarios/pattern_scenarios.json`, but the structural screening needs none, so run it with `--no-cq`:

```bash
python3 batch_evaluate.py --local-root outputs_wop2026 --no-cq --out eval_results_wop2026/odp_eval
python3 scripts/structural_table.py --in eval_results_wop2026/odp_eval --csv eval_wop2026/structural_by_model.csv
```

An output that parses to an empty graph counts as unparseable here (all 40 BLOOMZ outputs are empty files).

**OOPS! for the WOP 2026 outputs.** The committed records carry no OOPS! scan: the OOPS! web service did not respond on 4 November 2026. When an endpoint is available, add the scans and print the full table:

```bash
python3 batch_evaluate.py --local-root outputs_wop2026 --out eval_results_wop2026/odp_eval --patch-oops --oops-url https://oops.linkeddata.es/rest
python3 scripts/structural_table.py --in eval_results_wop2026/odp_eval --csv eval_wop2026/structural_by_model.csv
```

Then fill in the OOPS! and Overall structural columns of Table 9.

The LLM judge takes the scenario file as an argument, so it runs on the extension directly:

```bash
nohup bash scripts/run_wop2026_judge.sh > logs/wop2026_judge.log 2>&1 &
```

It uses the paper's judge settings: GPT-4o at temperature 0, single-candidate mode, GPT-5.4 skipped as same-family, BLOOMZ excluded. It needs `OPENAI_API_KEY` in `.env` and `pip install -r requirements_judge.txt`. Results go to `eval_judge_results_wop2026/gpt-4o/`, with the report in `aggregated/judge_report.md`. Each judged ODP is cached together with the SHA-256 of its `ontology.ttl`, so the script can be started while generation is still running. Re-running it judges only new outputs or outputs whose file changed. The judge reads `ontology.ttl` without the `# ODPGEN-TRUNCATED` banner that the generator prepends to truncated outputs: the banner tells the reader not to score the file, the judge followed it, and the paper's outputs carry no banner.

---

## Tests

```bash
pip install pytest
pytest tests/
```

---

## Licence

The source code is released under the MIT licence (`LICENSE`). The data produced in this study (generated ODPs, evaluation results, annotations and the pseudonymised questionnaire exports) are released under CC BY 4.0. Third-party material keeps its original licence: the reference ODPs in `data/ground_truth/` and `data/ontologies_retrived/`, and the scenarios and competency questions quoted from the source papers listed in `data/odp.csv`.

---

## Citation

Citation metadata for this repository is in `CITATION.cff` (GitHub shows a "Cite this repository" button). Please also cite the accompanying article:

> A. S. Lippolis, E. Norouzi, A. G. Nuzzolese, H. Sack, A. Gangemi. *Generating Ontology Design Patterns with LLMs: An Empirical Evaluation on Recently Published Patterns.* Manuscript, 2026.
