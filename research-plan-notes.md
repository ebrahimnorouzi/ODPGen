# ODP + LLM Study Notes (Consolidated)

## 1) Core study direction

**Main question:** To what extent can LLMs construct an Ontology Design Pattern (ODP) from a scenario, and how valid/useful is the result?

Working focus:
- formal validity (syntax, consistency, coherence)
- faithfulness to scenario requirements (coverage vs unsupported additions)
- usefulness/reusability for humans (expert evaluation)
- impact of prompt/tooling configurations

---

## 2) Candidate research questions

- **RQ1:** How well can LLMs generate an ODP from natural-language scenario descriptions?
- **RQ2:** To what extent are generated ODPs logically consistent and free of common modeling errors?
- **RQ3:** Do CQs, explicit reasoning guidance, and syntax/constraint instructions improve ODP quality and reduce hallucinations?
- **RQ4:** How faithful are generated ODPs to scenario ground truth (coverage of supported statements vs unsupported additions)?
- **RQ5:** How do experts rate usability/reusability of LLM-generated ODPs compared with expert-designed baselines?
- **RQ6:** Can models generate good patterns for recently published ODPs likely unseen during model training?

---

## 3) Experimental configurations (prompting conditions)

1. **Scenario-only generation**
2. **CQ-only generation**
3. **Scenario + CQs**
4. **Scenario + CQs + explicit reasoning steps**
5. **Scenario + syntax/constraint checking instructions**

Note: Keep output format consistent (avoid inconsistency where output may unexpectedly switch to markdown-only). Define one strict output contract for all conditions.

---

## 4) Five prompt templates (one per configuration)

### Prompt 1 — Scenario-only

You are an ontology engineer. Generate an OWL ontology design pattern (ODP) from the scenario below.

Requirements:
- Stay strictly within the scenario scope.
- Use clear class and property names.
- Include minimal but sufficient axioms.
- Output in Turtle syntax.
- After the ontology, provide a short documentation block with: intent, covered requirements, and known assumptions.

Scenario:
{{SCENARIO_TEXT}}

### Prompt 2 — CQ-only

You are an ontology engineer. Generate an OWL ontology design pattern (ODP) that can answer the following competency questions.

Requirements:
- Model only what is needed to support these CQs.
- Avoid adding domain facts not implied by the CQs.
- Use clear class and property names.
- Output in Turtle syntax.
- After the ontology, map each CQ to the ontology elements that support it.

Competency Questions:
{{CQ_LIST}}

### Prompt 3 — Scenario + CQs

You are an ontology engineer. Generate an OWL ontology design pattern (ODP) using both the scenario and competency questions.

Requirements:
- Prioritize semantic faithfulness to the scenario.
- Ensure the ODP supports all CQs.
- Explicitly avoid unsupported additions (hallucinations).
- Output in Turtle syntax.
- Then provide:
  1) requirement-to-axiom mapping,
  2) CQ-to-axiom mapping,
  3) list of intentionally excluded concepts.

Scenario:
{{SCENARIO_TEXT}}

Competency Questions:
{{CQ_LIST}}

### Prompt 4 — Scenario + CQs + explicit reasoning steps

You are an ontology engineer. Create an OWL ontology design pattern from the scenario and CQs.

Follow these steps explicitly:
1. Extract atomic requirements from the scenario.
2. Propose candidate classes/properties.
3. Draft axioms for each requirement.
4. Verify each CQ is supported.
5. Remove unsupported elements.
6. Output final ontology.

Constraints:
- Keep modeling minimal and reusable.
- No domain claims beyond scenario/CQs.
- Output final ontology in Turtle syntax.
- Include a traceability table (requirement -> axiom, CQ -> axiom).

Scenario:
{{SCENARIO_TEXT}}

Competency Questions:
{{CQ_LIST}}

### Prompt 5 — Scenario + syntax/constraint instructions

You are an ontology engineer. Generate an OWL ontology design pattern from the scenario and CQs, with strict syntax/constraint validation.

Validation checklist before final output:
- Turtle syntax validity.
- No undeclared prefixes.
- Domain/range used consistently.
- Class expressions coherent.
- Avoid contradictory restrictions.
- Keep ontology satisfiable and reusable.

Output format (strict):
1) Final ontology in Turtle.
2) "Validation report" with pass/fail per checklist item.
3) "Scope report" listing included requirements and rejected out-of-scope elements.

Scenario:
{{SCENARIO_TEXT}}

Competency Questions:
{{CQ_LIST}}

---

## 5) Evaluation framework (multi-layer)

### A. Formal quality
- syntax validity
- logical consistency (reasoner)
- class satisfiability/coherence
- anti-pattern checks (common OWL modeling mistakes)

### B. Scenario faithfulness
- requirement coverage (how many scenario statements are represented)
- unsupported additions (hallucinated claims)
- scope closeness (especially if scenarios are decomposed into atomic statements)

### C. Semantic similarity to reference pattern
- axiom-level/structure-aware comparison (not only lexical similarity)
- optional hybrid similarity (symbolic + embedding-assisted)

### D. Human-centered utility
Adapted 1–5 scales for:
- **Correctness** (from generation failure to syntactically+semantically correct and intended)
- **Completeness** (manual rework needed)
- **Clarity/reusability** (documentation quality, ease of adoption)

Recommendation: if expert capacity is limited, run human evaluation on top-performing configurations only (e.g., best 20× outputs).

---

## 6) Study design constraints and TODOs

### Immediate TODOs
- Select ODPs published **after 2023**.
- Count how many have user stories/CQs already available.
- For missing user stories/CQs:
  1) contact authors, or
  2) derive from existing CQs / documentation.
- Add a requirements-check column in tracking spreadsheet.
- Define and report query efficiency metrics.
- Clarify and operationalize "helpfulness" of a pattern.
- Coordinate calibration among evaluators (including possible student involvement and bias controls).

### Time-window logic for "unseen" evaluation
- Choose a defensible publication window for ODPs (e.g., last 2–3 years).
- Prefer models with training cutoffs plausibly before those ODPs.
- Explicitly document uncertainty for actively updated APIs.

Candidate model families mentioned in notes:
- GPT-3.5 / instruct variants
- LLaMA 2 / Code LLaMA
- BLOOM
- DeepSeek (early releases)
- Claude 3 Haiku (cutoff caution)
- 4o/o-series mini variants (cutoff caution)
- LLaMA 3.x, Qwen 2.5

---

## 7) Related evidence to cite in framing

- Reported benefits of pattern-based ontology engineering include simplified design, reuse, interoperability, and query/automation advantages.
- Prior studies suggest ODP reuse can improve speed and correctness compared with non-pattern workflows.
- Historical pain point: limited tooling for finding/selecting/specializing/composing suitable patterns.

(Use your existing bibliography entries for gangemi2009ontology, hammar2014ontology, hammar2010information, blomqvist2009semi, and newer examples.)

---

## 8) Additional future directions (from brainstorming)

- pattern recommendation systems from scenario or ontology input
- suggesting missing patterns for existing ontologies (factoring opportunities)
- extracting patterns from existing ontologies
- studying pattern evolution/lifecycle/versioning and adoption over time
- annotating ontology repositories with pattern usage metadata (e.g., OPLA-X style)

---

## 9) Pilot study results (March 2026)

### 9.1 Structural evaluation — top configurations

**Proprietary models (top 4 overall):**
1. Gemini 3.1 Pro — scenario-only (OOPS score 0.9643, structural score 0.9821)
2. Gemini 3.1 Pro — scenario-cq-reasoning (OOPS 0.7143, structural 0.8571)
3. GPT-5.4 — scenario-only (structural 0.4643)
4. GPT-5.4 — scenario-cq-constraints (structural 0.2929 overall, but top GPT config)

**Open-source models (top 3):**
1. Llama-3.1-8B — scenario-cq (structural 0.6161, parse rate 85.7%) 
2. Llama-3.1-8B — scenario-only (structural 0.5952, parse rate 78.6%) ← **creativity track winner**
3. Llama-2-70B — cq-only (structural 0.5893, parse rate 85.7%)

**Anomaly detected:** One Llama-3.1-8B output for scenario 2025-151-01 (scenario-cq config) repeated the same triples 22 times — degenerate output, retained in dataset as an edge case.

### 9.2 Functional evaluation (pilot)

Functional evaluation measures practical utility: whether generated ODPs answer competency questions without over-committing to unsupported ontological elements.

**Scoring formula:**
```
superfluous_penalty  = numbnoscen × 1.0 + (numb − numbnoscen) × 0.3
manual_score         = 1 / (1 + 0.5 × superfluous_penalty)    [0–1, higher = fewer superfluous elements]
functional_score     = 0.5 × manual_score + 0.5 × cq_pass_rate
```

- `numb`: total elements added beyond ground truth
- `numbnoscen`: extra elements with no grounding in the scenario (weighted ×1.0)
- Scenario-justified extra elements weighted ×0.3 (defensible over-specificity)

**Results (4 configurations evaluated):**

| Model | Config | Avg Extra | Manual Score | CQ Pass Rate | Functional Score |
|-------|--------|-----------|-------------|-------------|-----------------|
| Llama-2-70B | cq-only | 0.79 | 0.866 | 0.202 | **0.534** ← winner |
| GPT-5.4 | scenario-cq-constraints | 1.00 | 0.862 | 0.191 | 0.527 |
| Llama-3.1-8B | scenario-cq | 4.50 | 0.719 | 0.133 | 0.426 |
| Gemini 3.1 Pro | scenario-cq-reasoning | 2.21 | 0.774 | 0.023 | 0.399 |

### 9.3 Evaluation gap identified

Llama-2-70B (cq-only) achieves the highest functional score despite being described as "incredibly bad modeling" (low OOPS score 0.2476, structural score 0.4524 overall). It wins because it produces extremely minimal ontologies (avg_numb=0.79) with almost no unsupported additions, yielding high manual_score.

**Implication:** Automated functional scores reward parsimony but cannot detect ontological modeling deficiencies (incorrect domain/range, missing restrictions, semantically incoherent axioms). Expert evaluation is necessary to determine whether these outputs are actually usable.

---

## 10) Open-source model selection for human evaluation

### Decision

Two open-source model/configuration pairs selected for the human evaluation phase, each representing a distinct quality dimension:

| Track | Model | Config | Rationale |
|-------|-------|--------|-----------|
| **Scenario-only (Structural)** | Llama-3.1-8B | scenario-only | Creativity track winner; best structural quality for 8B model without CQ guidance; tests scenario sufficiency |
| **CQ-only (Functional)** | Llama-2-70B | cq-only | Functional evaluation winner (score 0.534); highest CQ pass rate among open-source; minimal unsupported additions |

### Rationale for configuration scope reduction

Reducing from 5 configurations to 2 for open-source models:
- Makes human evaluation feasible without overburdening participants
- The two selected configs represent opposite ends of the prompt spectrum (scenario-only vs CQ-only)
- Each is a clear winner in its respective quality dimension
- The structural/functional divergence in winners makes this an informative comparison

### Research value

The Llama-70B case presents a natural experiment: human experts can assess whether its functional metric advantage (0.534 score, 0.87 manual score) survives real-world scrutiny despite known structural weaknesses. This will help calibrate the validity of automated functional metrics.

---

## 11) Human evaluation plan (in progress)

### Participant pool
- Pattern authors: ~2 (incl. 1 who is also a student)
- Ontology engineering experts: ~5 (some are also pattern authors)
- MSc/PhD students: ~2
- ISWC workshop participants: ~8
- Target: ~38 participants (experts) + ~38 (students)
- Each evaluator assesses ~3 ontologies; pattern authors: ~2

### Evaluation forms (3 types)

**Pattern authors:** Author role, intent match (1–5), scope appropriateness (1–5), reuse readiness (1–5), adoption threshold (Yes/Maybe/No + justification), time saved (0%/1–25%/26–50%/51–75%/76–100%), missing core elements (free text)

**Experts:** Overall helpfulness (1–5), reusability (1–5), clarity/documentation (1–5), adoption threshold (Yes/Maybe/No + justification), time saved, free text observations

**Students:** Academic level (MSc/PhD), time to understand (minutes), documentation clarity (1–5), naming clarity (1–5), adoption (Yes/Maybe/No + justification), time saved

### Notes and open questions
- Very long CQ lists may discourage participation — consider trimming to representative subset (~3 CQs)
- Whether to evaluate ground truth ODPs alongside generated ones
- Evaluation platform: https://github.com/ebrahimnorouzi/odp-platform
