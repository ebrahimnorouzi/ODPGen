"""
Core LLM-as-a-Judge evaluation engine.

Supports two evaluation modes:
  - Single-candidate: evaluate one generated ODP against ground truth.
  - Pairwise: evaluate two candidates with position-swap bias mitigation.

The judge LLM is configurable and must NOT be the same model family
that generated the ODP being evaluated.
"""
from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, field

from .prompt_templates import (
    PAIRWISE_EVAL_TEMPLATE,
    SINGLE_EVAL_TEMPLATE,
    SYSTEM_PROMPT,
)
from .data_loader import format_cqs

# ─────────────────────────────────────────────────────────────────────────────
# Result dataclasses
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SingleEvalResult:
    """Result from a single-candidate evaluation."""
    syntactic_correctness_score: int = 0
    semantic_accuracy_score: int = 0
    logical_consistency_score: int = 0
    functional_adequacy_score: int = 0
    overall_score: int = 0
    syntactic_correctness_analysis: str = ""
    semantic_accuracy_analysis: str = ""
    logical_consistency_analysis: str = ""
    functional_adequacy_analysis: str = ""
    overall_analysis: str = ""
    raw_response: str = ""
    judge_model: str = ""

    def to_dict(self) -> dict:
        return {
            "syntactic_correctness_score": self.syntactic_correctness_score,
            "semantic_accuracy_score": self.semantic_accuracy_score,
            "logical_consistency_score": self.logical_consistency_score,
            "functional_adequacy_score": self.functional_adequacy_score,
            "overall_score": self.overall_score,
            "syntactic_correctness_analysis": self.syntactic_correctness_analysis,
            "semantic_accuracy_analysis": self.semantic_accuracy_analysis,
            "logical_consistency_analysis": self.logical_consistency_analysis,
            "functional_adequacy_analysis": self.functional_adequacy_analysis,
            "overall_analysis": self.overall_analysis,
            "judge_model": self.judge_model,
        }


@dataclass
class PairwiseEvalResult:
    """Result from a pairwise evaluation with position-swap debiasing."""
    candidate_1_score: float = 0.0  # averaged across both positions
    candidate_2_score: float = 0.0
    candidate_1_analysis_ab: str = ""  # analysis when cand1=A
    candidate_1_analysis_ba: str = ""  # analysis when cand1=B
    candidate_2_analysis_ab: str = ""  # analysis when cand2=B
    candidate_2_analysis_ba: str = ""  # analysis when cand2=A
    raw_run_ab: str = ""
    raw_run_ba: str = ""
    judge_model: str = ""

    def to_dict(self) -> dict:
        return {
            "candidate_1_score": self.candidate_1_score,
            "candidate_2_score": self.candidate_2_score,
            "candidate_1_analysis_ab": self.candidate_1_analysis_ab,
            "candidate_1_analysis_ba": self.candidate_1_analysis_ba,
            "candidate_2_analysis_ab": self.candidate_2_analysis_ab,
            "candidate_2_analysis_ba": self.candidate_2_analysis_ba,
            "judge_model": self.judge_model,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Judge engine
# ─────────────────────────────────────────────────────────────────────────────

# Generating model family → judge exclusion mapping
_MODEL_FAMILY = {
    "gpt": "openai",
    "gemini": "google",
    "llama": "meta",
    "mistral": "mistral",
    "bloom": "bigscience",
    "claude": "anthropic",
}


def _detect_family(model_name: str) -> str:
    """Detect the model family from a model name string."""
    name_lower = model_name.lower()
    for key, family in _MODEL_FAMILY.items():
        if key in name_lower:
            return family
    return "unknown"


def _judge_backend(judge_model: str) -> str:
    """Determine the API backend for the judge model."""
    if "claude" in judge_model.lower():
        return "anthropic"
    if "gpt" in judge_model.lower() or "o1" in judge_model.lower():
        return "openai"
    raise ValueError(f"Unknown judge model backend: {judge_model}")


def check_judge_independence(judge_model: str, generating_model: str) -> bool:
    """
    Verify that the judge is NOT from the same family as the generator.
    Returns True if independent, False if conflict detected.
    """
    judge_family = _detect_family(judge_model)
    gen_family = _detect_family(generating_model)
    return judge_family != gen_family


class JudgeLLM:
    """
    Wrapper around LLM API calls for evaluation.
    Supports Anthropic (Claude) and OpenAI (GPT) backends.
    """

    def __init__(
        self,
        model: str | None = None,
        temperature: float | None = None,
        max_retries: int = 3,
        retry_delay: float = 5.0,
    ):
        self.model = model or os.getenv("JUDGE_MODEL", "claude-sonnet-4-6")
        self.temperature = (
            temperature if temperature is not None
            else float(os.getenv("JUDGE_TEMPERATURE", "0.0"))
        )
        self.backend = _judge_backend(self.model)
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self._client = None

    def _get_client(self):
        """Lazy-init the API client."""
        if self._client is not None:
            return self._client

        if self.backend == "anthropic":
            import anthropic
            self._client = anthropic.Anthropic(
                api_key=os.getenv("ANTHROPIC_API_KEY"),
            )
        elif self.backend == "openai":
            import openai
            self._client = openai.OpenAI(
                api_key=os.getenv("OPENAI_API_KEY"),
            )
        return self._client

    def _call_llm(self, system: str, user: str) -> str:
        """Send a prompt to the judge LLM and return the response text."""
        client = self._get_client()

        for attempt in range(self.max_retries):
            try:
                if self.backend == "anthropic":
                    response = client.messages.create(
                        model=self.model,
                        max_tokens=8192,
                        temperature=self.temperature,
                        system=system,
                        messages=[{"role": "user", "content": user}],
                    )
                    if response.stop_reason == "max_tokens":
                        print("\n    [judge] WARNING: response truncated (max_tokens)")
                    return response.content[0].text

                elif self.backend == "openai":
                    response = client.chat.completions.create(
                        model=self.model,
                        temperature=self.temperature,
                        max_tokens=8192,
                        messages=[
                            {"role": "system", "content": system},
                            {"role": "user", "content": user},
                        ],
                    )
                    if response.choices[0].finish_reason == "length":
                        print("\n    [judge] WARNING: response truncated (max_tokens)")
                    return response.choices[0].message.content

            except Exception as e:
                print(f"    [judge] API error (attempt {attempt+1}/{self.max_retries}): {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay * (attempt + 1))
                else:
                    raise

    # ── Single-candidate evaluation ──────────────────────────────────────

    def evaluate_single(
        self,
        ground_truth: str,
        cq_list: list[str],
        candidate: str,
    ) -> SingleEvalResult:
        """
        Evaluate a single candidate ODP against the ground truth.
        Returns dimensional scores and analyses.
        """
        prompt = SINGLE_EVAL_TEMPLATE.format(
            ground_truth=ground_truth,
            competency_questions=format_cqs(cq_list),
            candidate=candidate,
        )

        raw = self._call_llm(SYSTEM_PROMPT, prompt)
        parsed = _parse_json_response(raw)

        return SingleEvalResult(
            syntactic_correctness_score=_safe_int(parsed.get("syntactic_correctness_score")),
            semantic_accuracy_score=_safe_int(parsed.get("semantic_accuracy_score")),
            logical_consistency_score=_safe_int(parsed.get("logical_consistency_score")),
            functional_adequacy_score=_safe_int(parsed.get("functional_adequacy_score")),
            overall_score=_safe_int(parsed.get("overall_score")),
            syntactic_correctness_analysis=parsed.get("syntactic_correctness_analysis", ""),
            semantic_accuracy_analysis=parsed.get("semantic_accuracy_analysis", ""),
            logical_consistency_analysis=parsed.get("logical_consistency_analysis", ""),
            functional_adequacy_analysis=parsed.get("functional_adequacy_analysis", ""),
            overall_analysis=parsed.get("overall_analysis", ""),
            raw_response=raw,
            judge_model=self.model,
        )

    # ── Pairwise evaluation with position-swap debiasing ─────────────────

    def evaluate_pair(
        self,
        ground_truth: str,
        cq_list: list[str],
        candidate_1: str,
        candidate_2: str,
    ) -> PairwiseEvalResult:
        """
        Evaluate two candidate ODPs with position-swap bias mitigation.

        Run 1: candidate_1 = A, candidate_2 = B
        Run 2: candidate_2 = A, candidate_1 = B  (swapped)
        Final scores are averaged across both runs.
        """
        cqs_formatted = format_cqs(cq_list)

        # ── Run 1: original order (cand1=A, cand2=B) ────────────────────
        prompt_ab = PAIRWISE_EVAL_TEMPLATE.format(
            ground_truth=ground_truth,
            competency_questions=cqs_formatted,
            candidate_a=candidate_1,
            candidate_b=candidate_2,
        )
        raw_ab = self._call_llm(SYSTEM_PROMPT, prompt_ab)
        parsed_ab = _parse_json_response(raw_ab)

        # ── Run 2: swapped order (cand2=A, cand1=B) ─────────────────────
        prompt_ba = PAIRWISE_EVAL_TEMPLATE.format(
            ground_truth=ground_truth,
            competency_questions=cqs_formatted,
            candidate_a=candidate_2,
            candidate_b=candidate_1,
        )
        raw_ba = self._call_llm(SYSTEM_PROMPT, prompt_ba)
        parsed_ba = _parse_json_response(raw_ba)

        # ── Average scores across both position orderings ────────────────
        # Run 1: cand1 was A, cand2 was B
        cand1_score_run1 = _safe_int(parsed_ab.get("Candidate_A_Score"))
        cand2_score_run1 = _safe_int(parsed_ab.get("Candidate_B_Score"))

        # Run 2: cand2 was A, cand1 was B
        cand1_score_run2 = _safe_int(parsed_ba.get("Candidate_B_Score"))
        cand2_score_run2 = _safe_int(parsed_ba.get("Candidate_A_Score"))

        return PairwiseEvalResult(
            candidate_1_score=round((cand1_score_run1 + cand1_score_run2) / 2, 2),
            candidate_2_score=round((cand2_score_run1 + cand2_score_run2) / 2, 2),
            candidate_1_analysis_ab=parsed_ab.get("Candidate_A_Analysis", ""),
            candidate_1_analysis_ba=parsed_ba.get("Candidate_B_Analysis", ""),
            candidate_2_analysis_ab=parsed_ab.get("Candidate_B_Analysis", ""),
            candidate_2_analysis_ba=parsed_ba.get("Candidate_A_Analysis", ""),
            raw_run_ab=raw_ab,
            raw_run_ba=raw_ba,
            judge_model=self.model,
        )


# ─────────────────────────────────────────────────────────────────────────────
# JSON parsing helpers
# ─────────────────────────────────────────────────────────────────────────────

def _parse_json_response(raw: str) -> dict:
    """
    Extract JSON from LLM response, handling markdown fences,
    trailing text, and truncated responses.
    """
    # Try direct parse first
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

    # Strip markdown code fences
    cleaned = re.sub(r"```(?:json)?\s*", "", raw)
    cleaned = re.sub(r"```\s*$", "", cleaned)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # Find first { ... last }
    start = raw.find("{")
    end = raw.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(raw[start : end + 1])
        except json.JSONDecodeError:
            pass

    # Handle truncated JSON: find partial object and try to close it
    json_start = raw.find("{")
    if json_start != -1:
        fragment = raw[json_start:]
        # Try adding closing brace(s)
        for suffix in ["\"}", "\n}", "}", "0}", "\"\"}"]:
            try:
                return json.loads(fragment + suffix)
            except json.JSONDecodeError:
                continue

    print(f"    [judge] WARNING: could not parse JSON from response ({len(raw)} chars)")
    return {}


def _safe_int(val, default: int = 0) -> int:
    """Safely convert a value to int, clamping to 1-10 range."""
    try:
        v = int(val)
        return max(1, min(10, v))
    except (TypeError, ValueError):
        return default
