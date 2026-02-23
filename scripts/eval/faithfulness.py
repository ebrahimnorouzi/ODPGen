"""Faithfulness metrics against atomic requirements."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List

WORD_RE = re.compile(r"[A-Za-z0-9_]+")
STOPWORDS = {
    "a", "an", "the", "and", "or", "of", "to", "in", "on", "for", "with", "is", "are", "be", "as", "by",
    "it", "this", "that", "from", "at", "if", "then", "than", "into", "about",
}


@dataclass
class MetricResult:
    value: Any
    evaluated: bool
    reason: str | None = None

    def as_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {"value": self.value, "evaluated": self.evaluated}
        if self.reason:
            out["reason"] = self.reason
        return out


def _tokens(text: str) -> List[str]:
    return [t.lower() for t in WORD_RE.findall(text)]


def _content_tokens(text: str) -> set[str]:
    return {t for t in _tokens(text) if t not in STOPWORDS}


def _normalize_requirements(requirements: Iterable[Any]) -> List[str]:
    normalized: List[str] = []
    for req in requirements:
        if isinstance(req, str):
            normalized.append(req)
        elif isinstance(req, dict):
            normalized.append(str(req.get("requirement") or req.get("text") or "").strip())
        else:
            normalized.append(str(req))
    return [r for r in normalized if r]


def evaluate_faithfulness(output_text: str, atomic_requirements: Iterable[Any]) -> Dict[str, Any]:
    requirements = _normalize_requirements(atomic_requirements)
    if not requirements:
        return {
            "coverage": MetricResult(None, False, "Not evaluated: missing atomic requirements.").as_dict(),
            "unsupported_additions": MetricResult(None, False, "Not evaluated: missing atomic requirements.").as_dict(),
            "details": {"covered_requirements": 0, "total_requirements": 0},
        }

    output_content = _content_tokens(output_text)
    covered = 0
    for req in requirements:
        req_tokens = _content_tokens(req)
        if not req_tokens:
            continue
        overlap = len(req_tokens & output_content) / len(req_tokens)
        if overlap >= 0.6:
            covered += 1

    coverage = covered / len(requirements) if requirements else 0.0

    req_vocab = _content_tokens(" ".join(requirements))
    output_vocab = _content_tokens(output_text)
    extra_vocab = output_vocab - req_vocab
    unsupported = len(extra_vocab) / len(output_vocab) if output_vocab else 0.0

    return {
        "coverage": MetricResult(round(coverage, 4), True).as_dict(),
        "unsupported_additions": MetricResult(round(unsupported, 4), True).as_dict(),
        "details": {
            "covered_requirements": covered,
            "total_requirements": len(requirements),
            "requirement_vocab_size": len(req_vocab),
            "output_vocab_size": len(output_vocab),
            "extra_vocab_size": len(extra_vocab),
        },
    }
import re


def tokenize(text: str) -> set[str]:
    tokens = re.findall(r"[A-Za-z][A-Za-z0-9_-]{2,}", text.lower())
    stop = {"the", "and", "for", "with", "that", "this", "which", "from", "into", "what", "who"}
    return {t for t in tokens if t not in stop}


def evaluate_faithfulness(scenario_text: str, cq_list: list[str], ontology_text: str) -> dict:
    source_terms = tokenize(scenario_text + " " + " ".join(cq_list))
    onto_terms = tokenize(ontology_text)
    if not source_terms:
        coverage = 0.0
    else:
        coverage = len(source_terms & onto_terms) / len(source_terms)
    unsupported = sorted(list(onto_terms - source_terms))
    unsupported_trimmed = unsupported[:30]
    return {
        "coverage_ratio": round(coverage, 3),
        "unsupported_addition_count": len(unsupported),
        "unsupported_additions_sample": unsupported_trimmed,
    }
