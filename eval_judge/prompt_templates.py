"""
G-Eval prompt templates for LLM-as-a-Judge ontology evaluation.

Two modes:
  1. Single-candidate evaluation: one candidate vs. ground truth.
  2. Pairwise evaluation: two candidates vs. ground truth (with position-swap
     bias mitigation).

Each template follows the G-Eval Chain-of-Thought (CoT) paradigm: the judge
is forced to reason step-by-step before emitting a score.
"""

# ─────────────────────────────────────────────────────────────────────────────
# System prompt shared by both modes
# ─────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = (
    "You are an expert Semantic Web Logician and Ontology Engineer. "
    "You evaluate candidate Ontology Design Patterns (ODPs) against a "
    "Ground Truth ODP using a rigorous, multi-dimensional rubric."
)

# ─────────────────────────────────────────────────────────────────────────────
# Rubric (shared)
# ─────────────────────────────────────────────────────────────────────────────

RUBRIC = """\
RUBRIC (1-10 scale):
  1-2 : Syntactically invalid or logically incoherent. Fails all CQs.
  3-4 : Major semantic drift, misses core classes, severe logical errors.
  5-6 : Captures basic taxonomy but misses complex restrictions. Fails complex CQs.
  7-8 : Logically sound, highly similar semantics. Answers almost all CQs. Minor structural divergences.
  9-10: Semantically isomorphic to the Ground Truth. Flawless logic, satisfies all CQs."""

# ─────────────────────────────────────────────────────────────────────────────
# Single-candidate evaluation template
# ─────────────────────────────────────────────────────────────────────────────

SINGLE_EVAL_TEMPLATE = """\
Evaluate the following Candidate Ontology Design Pattern against the Ground Truth ODP on four dimensions and assign a 1-10 score for each dimension plus an overall score.

=== INPUTS ===

[Ground Truth ODP]:
{ground_truth}

[Competency Questions]:
{competency_questions}

[Candidate ODP]:
{candidate}

=== EVALUATION STEPS (follow in order) ===

1. **Parse the Ground Truth**: Identify core classes, object/data properties, axioms (domain/range, cardinality, restrictions), and the overall modelling intent.

2. **Syntactic Correctness** (Structural): Verify the Candidate uses valid OWL 2 DL / RDF Turtle syntax. Check for malformed triples, undeclared prefixes, or broken URIs.

3. **Semantic Accuracy** (Functional): Compare the Candidate's classes and properties against the Ground Truth. Compute conceptual precision (how many Candidate concepts are relevant) and recall (how many Ground Truth concepts are captured). Allow valid synonyms and minor naming differences.

4. **Logical Consistency**: Detect unsatisfiable classes, contradictory axioms, domain/range violations, or circular definitions in the Candidate.

5. **Functional Adequacy** (Pragmatics): For each Competency Question, determine whether the Candidate contains the logical structure (classes, properties, axioms) required to answer it.

6. **Overall Score**: Synthesise dimensions 2-5 into an overall quality score.

""" + RUBRIC + """

=== OUTPUT FORMAT ===
Respond strictly as a JSON object — no markdown fences, no extra text:
{{
  "syntactic_correctness_analysis": "<reasoning>",
  "syntactic_correctness_score": <int 1-10>,
  "semantic_accuracy_analysis": "<reasoning>",
  "semantic_accuracy_score": <int 1-10>,
  "logical_consistency_analysis": "<reasoning>",
  "logical_consistency_score": <int 1-10>,
  "functional_adequacy_analysis": "<reasoning>",
  "functional_adequacy_score": <int 1-10>,
  "overall_analysis": "<1-2 sentence summary>",
  "overall_score": <int 1-10>
}}"""

# ─────────────────────────────────────────────────────────────────────────────
# Pairwise evaluation template (with position-swap bias mitigation)
# ─────────────────────────────────────────────────────────────────────────────

PAIRWISE_EVAL_TEMPLATE = """\
Evaluate two candidate Ontology Design Patterns (Candidate A and Candidate B) against a Ground Truth ODP on a scale of 1 to 10.

=== INPUTS ===

[Ground Truth ODP]:
{ground_truth}

[Competency Questions]:
{competency_questions}

[Candidate A]:
{candidate_a}

[Candidate B]:
{candidate_b}

=== EVALUATION STEPS (follow in order) ===

1. **Parse the Ground Truth** to identify core classes, properties, and axioms.

2. **Syntactic Correctness**: Verify valid OWL/RDF syntax for both candidates.

3. **Semantic Alignment**: Compare Candidate A and B concepts against the Ground Truth (allow valid synonyms).

4. **Logical Consistency**: Identify any unsatisfiable classes or contradictory axioms.

5. **Functional Adequacy**: Assess if the candidates contain the structure to answer the Competency Questions.

6. **Scoring**: Assign a 1-10 score for each candidate based on the rubric.

""" + RUBRIC + """

=== OUTPUT FORMAT ===
Respond strictly as a JSON object — no markdown fences, no extra text:
{{
  "Candidate_A_Analysis": "<step-by-step reasoning for A>",
  "Candidate_A_Score": <int 1-10>,
  "Candidate_B_Analysis": "<step-by-step reasoning for B>",
  "Candidate_B_Score": <int 1-10>
}}"""
