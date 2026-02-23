"""Similarity metrics with axiom/structure-aware extension hooks."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict

WORD_RE = re.compile(r"[A-Za-z0-9_]+")


@dataclass
class MetricResult:
    value: Any
    evaluated: bool
    reason: str | None = None

    def as_dict(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {"value": self.value, "evaluated": self.evaluated}
        if self.reason:
            payload["reason"] = self.reason
        return payload


def _token_set(text: str) -> set[str]:
    return {w.lower() for w in WORD_RE.findall(text)}


def evaluate_similarity(output_text: str, reference_text: str | None) -> Dict[str, Any]:
    if not reference_text:
        return {
            "token_jaccard": MetricResult(None, False, "Not evaluated: missing reference output.").as_dict(),
            "axiom_similarity": MetricResult(None, False, "Not evaluated: requires axiom parser/tooling.").as_dict(),
            "structure_similarity": MetricResult(None, False, "Not evaluated: requires structure-aware comparator.").as_dict(),
        }

    out_tokens = _token_set(output_text)
    ref_tokens = _token_set(reference_text)
    union = out_tokens | ref_tokens
    inter = out_tokens & ref_tokens
    jaccard = len(inter) / len(union) if union else 1.0

    return {
        "token_jaccard": MetricResult(round(jaccard, 4), True).as_dict(),
        "axiom_similarity": MetricResult(None, False, "Not evaluated: requires axiom parser/tooling.").as_dict(),
        "structure_similarity": MetricResult(None, False, "Not evaluated: requires structure-aware comparator.").as_dict(),
    }
