"""Formal quality metrics for generated outputs.

Some metrics are placeholders when external tooling is unavailable.
"""

from __future__ import annotations

import ast
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict


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


def _try_parse_output(path: Path, text: str) -> MetricResult:
    suffix = path.suffix.lower()
    try:
        if suffix == ".json":
            json.loads(text)
            return MetricResult(value=True, evaluated=True)
        if suffix == ".py":
            ast.parse(text)
            return MetricResult(value=True, evaluated=True)
        # Generic fallback: lightweight structural check for balanced brackets.
        pairs = {')': '(', ']': '[', '}': '{'}
        stack = []
        for ch in text:
            if ch in '([{':
                stack.append(ch)
            elif ch in pairs:
                if not stack or stack[-1] != pairs[ch]:
                    return MetricResult(value=False, evaluated=True, reason="Unbalanced delimiters")
                stack.pop()
        if stack:
            return MetricResult(value=False, evaluated=True, reason="Unbalanced delimiters")
        return MetricResult(value=True, evaluated=True)
    except Exception as exc:  # parse error
        return MetricResult(value=False, evaluated=True, reason=f"Parse failed: {exc}")


def evaluate_formal_quality(output_path: Path, output_text: str) -> Dict[str, Any]:
    """Return formal-quality metrics with explicit not-evaluated placeholders."""
    syntax = _try_parse_output(output_path, output_text).as_dict()

    consistency = MetricResult(
        value=None,
        evaluated=False,
        reason="Not evaluated: requires external reasoner integration.",
    ).as_dict()
    coherence = MetricResult(
        value=None,
        evaluated=False,
        reason="Not evaluated: requires discourse/coherence model.",
    ).as_dict()

    return {
        "syntax_parse": syntax,
        "consistency": consistency,
        "coherence": coherence,
    }
