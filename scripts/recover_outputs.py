#!/usr/bin/env python3
"""
recover_outputs.py -- offline salvage of ontology.ttl files destroyed by the F1
extractor bug.

Background
----------
``scripts/run_generation.py`` wrote every generation to
``outputs/{model}/{config}/{scenario_id}/`` as four files: ``prompt.txt``,
``raw_response.txt``, ``ontology.ttl`` and ``metadata.json``.

F1 -- ``extract_turtle()`` selected the *first* fenced code span in the model
reply with ``re.search``.  The prompts instruct the model to wrap its answer in
a turtle-tagged code fence, and models routinely open their reply by echoing
that instruction verbatim.  The echo is therefore the first fenced span and the
capture group returns the three literal characters ``...`` -- the real ontology
further down was never reached.  70 generations in this corpus were saved as
3-byte ``ontology.ttl`` files.

F2 -- generation ran with ``max_new_tokens=1024`` while the prompts ask for four
deliverables, so many replies were cut off mid-axiom and the closing fence was
never emitted.  Those cannot be salvaged from disk at all.

What this tool does
-------------------
The full model reply is still on disk in ``raw_response.txt``.  This tool walks
the corpus, finds the damaged ``ontology.ttl`` files, re-extracts the Turtle
from the sibling ``raw_response.txt`` using the *corrected* extractor, validates
the result by actually parsing it with rdflib, and (only with ``--apply``)
writes it back -- preserving the damaged original as ``ontology.ttl.orig-empty``.

It performs **no API calls of any kind**.  Everything it needs is already on
disk.  It is a dry run by default and it is idempotent: a recovered directory is
no longer damaged, so a second run is a no-op.

Extractor sharing
-----------------
The extraction logic is *imported* from ``scripts/run_generation.py`` rather
than copied, so the generator and this recovery tool can never drift apart.
Because the upstream fix may not have landed yet, a local
``fallback_extract_turtle`` implements the corrected behaviour (longest fenced
block containing ``@prefix`` / ``@base`` / ``" a owl:"``) and is used *only when
the imported function fails to produce parseable Turtle*.  Every recovery record
in the report names which extractor produced it, so ``"extractor":
"fallback_longest_fence"`` is a live signal that ``run_generation.extract_turtle``
is still broken.  Once it is fixed the imported function wins and the fallback
goes cold on its own.

Safety rules
------------
A recovery tool must never invent content, so three rules bound what it will
write:

1. Nothing is written that rdflib has not already parsed into a graph with at
   least one triple.  (An empty graph parses fine and scores maximally under the
   current structural metric -- F3 -- so it is rejected explicitly.)
2. When the reply contains closed Turtle fences, the accepted text must be one
   of them *verbatim*.  ``run_generation.extract_turtle`` has an unfenced
   fallback that walks the raw text to end-of-buffer; that is the right call at
   generation time, but here it would splice a truncated tail onto an otherwise
   complete block.
3. A reply cut off inside an open fence (F2) yields nothing at all, even though
   ``maybe_fix_common_turtle_issues`` could trim the dangling statement into
   something that parses.  A partial ODP written back as ``ontology.ttl`` would
   look complete and quietly misrepresent what the model produced; those
   scenarios need regeneration, which is out of scope here.

Usage
-----
    python scripts/recover_outputs.py                  # dry run (default)
    python scripts/recover_outputs.py --apply          # actually write files
    python scripts/recover_outputs.py --report out.json
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import rdflib

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SCRIPTS_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPTS_DIR.parent
DEFAULT_OUTPUTS_ROOT = REPO_ROOT / "outputs"
DEFAULT_REPORT_PATH = REPO_ROOT / "recovery_report.json"
RUN_GENERATION_PATH = SCRIPTS_DIR / "run_generation.py"

ONTOLOGY_NAME = "ontology.ttl"
RAW_RESPONSE_NAME = "raw_response.txt"
#: Suffix under which a damaged original is preserved before being replaced.
ORIG_SUFFIX = ".orig-empty"

#: An ontology.ttl below this size is damaged by definition.  The F1 bug
#: produced 3-byte files ("..."); 50 bytes is well below the smallest possible
#: real ODP (a single @prefix line is already ~45 bytes) and well above 3.
MIN_ONTOLOGY_BYTES = 50

#: Substrings that make a chunk of text plausibly Turtle.  Deliberately narrow:
#: a bare ":Foo" bullet in prose must NOT qualify.
TURTLE_MARKERS = ("@prefix", "@base", " a owl:")

#: A fenced code block.  The opening fence must be followed by an optional info
#: string and then a newline, which is what keeps the prompt-echo fence (an
#: inline "turtle ... " span written inside one sentence) from ever matching --
#: that echo is the exact thing the F1 bug tripped over.
_FENCE_BLOCK_RE = re.compile(
    r"```[ \t]*[A-Za-z0-9_+.-]*[ \t]*\r?\n(.*?)```",
    re.DOTALL,
)
_FENCE_TOKEN_RE = re.compile(r"```")

#: An opening fence whose closing partner never arrives -- the F2 signature.
#: Everything from the newline after the info string to end-of-input is the
#: partial ontology the model had emitted when the token budget ran out.
_OPEN_FENCE_TAIL_RE = re.compile(
    r"```[ \t]*[A-Za-z0-9_+.-]*[ \t]*\r?\n((?:(?!```).)*)\Z",
    re.DOTALL,
)

#: Stamped as the first line of any ontology recovered from a truncated reply.
#: This is what makes truncation recovery safe: the artefact announces its own
#: provenance, so no downstream reader can mistake a partial ODP for a complete
#: one.  ``run_generation`` writes the sibling ``# ODPGEN-TRUNCATED`` banner.
TRUNCATION_BANNER = "# ODPGEN-RECOVERED-FROM-TRUNCATED-RESPONSE"


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------

def looks_like_turtle(text: str) -> bool:
    """True if *text* plausibly contains Turtle (not just prose naming a class)."""
    if not text:
        return False
    lowered = text.lower()
    return any(marker in lowered for marker in TURTLE_MARKERS)


def fenced_blocks(response: str) -> List[str]:
    """All fenced code-block bodies in *response*, in document order."""
    return [block.strip() for block in _FENCE_BLOCK_RE.findall(response)]


def turtle_fenced_blocks(response: str) -> List[str]:
    """Closed fenced blocks that plausibly contain Turtle, longest first."""
    blocks = [block for block in fenced_blocks(response) if looks_like_turtle(block)]
    return sorted(blocks, key=len, reverse=True)


def fallback_extract_turtle(response: str) -> str:
    """
    The corrected extraction rule: the **longest** fenced block that plausibly
    contains Turtle.

    This exists only as a safety net for the window in which
    ``run_generation.extract_turtle`` is still the buggy first-match version.
    It is never used when the imported extractor yields parseable Turtle.
    """
    candidates = turtle_fenced_blocks(response)
    return candidates[0] if candidates else ""


def open_fence_tail(response: str) -> str:
    """
    The body of an unterminated Turtle fence, or ``""``.

    This is the F2 case: the reply opened a fence and the token budget ran out
    before it closed.  The text is *partial by construction* -- it is only ever
    accepted alongside :data:`TRUNCATION_BANNER`.
    """
    match = _OPEN_FENCE_TAIL_RE.search(response)
    if match is None:
        return ""
    body = match.group(1).strip()
    return body if looks_like_turtle(body) else ""


def drop_incomplete_tail(text: str) -> str:
    """
    Drop the trailing statement a truncation cut in half.

    Turtle statements end in ``.``; a reply severed mid-axiom leaves a fragment
    that cannot parse.  Removing whole lines from the end until the remainder
    terminates cleanly is the least-invasive repair that can succeed, and it
    only ever *removes* model output -- it never invents any.
    """
    lines = text.rstrip().splitlines()
    while lines:
        candidate = "\n".join(lines).rstrip()
        if candidate.endswith("."):
            return candidate
        lines.pop()
    return ""


def response_is_truncated(response: str) -> bool:
    """
    True when the reply was cut off inside a code fence (the F2 signature).

    An odd number of fence tokens means one fence was opened and never closed;
    if Turtle follows that last opening, the ontology on disk is incomplete.
    """
    fences = list(_FENCE_TOKEN_RE.finditer(response))
    if len(fences) % 2 == 0:
        return False
    return looks_like_turtle(response[fences[-1].end():])


def load_run_generation():
    """
    Import ``scripts/run_generation.py`` as a module.

    Loaded by file location under a private module name so that importing it
    neither depends on nor pollutes ``sys.path``.  ``run_generation`` guards its
    entry point with ``if __name__ == "__main__"`` and its module-level imports
    are stdlib only, so importing it is side-effect free and never touches an
    API key.
    """
    spec = importlib.util.spec_from_file_location(
        "odpgen_run_generation", RUN_GENERATION_PATH
    )
    if spec is None or spec.loader is None:  # pragma: no cover - defensive
        raise ImportError("cannot load %s" % RUN_GENERATION_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_extract_turtle() -> Callable[[str], str]:
    """The shared extractor, imported from run_generation (never copied)."""
    return load_run_generation().extract_turtle


def load_turtle_repair() -> Optional[Callable[[str], str]]:
    """
    ``run_generation.maybe_fix_common_turtle_issues`` if available.

    The generation pipeline applies this to every HuggingFace output, so a
    faithful recovery applies it too -- but only as a second attempt, after the
    verbatim block has already failed to parse.
    """
    return getattr(load_run_generation(), "maybe_fix_common_turtle_issues", None)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_turtle(text: str) -> Tuple[bool, int, str]:
    """
    Parse *text* as Turtle with rdflib.

    Returns ``(ok, n_triples, error)``.  ``ok`` is True only for text that
    parses **and** yields at least one triple -- an empty graph is exactly the
    degenerate artefact this whole exercise is trying to stop producing (F3).
    """
    if not text or not text.strip():
        return False, 0, "empty text"
    graph = rdflib.Graph()
    try:
        graph.parse(data=text, format="turtle")
    except Exception as exc:  # rdflib raises a zoo of parser exceptions
        detail = str(exc).strip().splitlines()
        head = detail[0] if detail else ""
        return False, 0, ("%s: %s" % (type(exc).__name__, head))[:400]
    n = len(graph)
    if n == 0:
        return False, 0, "parsed but graph is empty (0 triples)"
    return True, n, ""


def validate_turtle_file(path: Path) -> Tuple[bool, int, str]:
    """Same as :func:`validate_turtle` but reading from disk."""
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:  # pragma: no cover - defensive
        return False, 0, "unreadable: %s" % exc
    return validate_turtle(text)


# ---------------------------------------------------------------------------
# Damage classification
# ---------------------------------------------------------------------------

STATE_MISSING = "missing"
STATE_EMPTY = "empty"
STATE_TOO_SMALL = "too_small"
STATE_UNPARSEABLE = "unparseable"
STATE_OK = "ok"


def classify_ontology(
    path: Path, min_bytes: int = MIN_ONTOLOGY_BYTES
) -> Tuple[str, int, int]:
    """
    Classify the on-disk ``ontology.ttl``.

    Returns ``(state, size_bytes, n_triples)``.
    """
    if not path.exists():
        return STATE_MISSING, 0, 0
    size = path.stat().st_size
    if size == 0:
        return STATE_EMPTY, 0, 0
    if size < min_bytes:
        return STATE_TOO_SMALL, size, 0
    ok, n_triples, _ = validate_turtle_file(path)
    if not ok:
        return STATE_UNPARSEABLE, size, 0
    return STATE_OK, size, n_triples


# Unrecoverable reasons -----------------------------------------------------
REASON_NO_RAW = "raw_response_missing"
REASON_RAW_EMPTY = "raw_response_empty"
REASON_NO_TURTLE = "no_turtle_content_in_raw_response"
REASON_UNTERMINATED = "unterminated_code_fence"
REASON_NO_FENCED_BLOCK = "no_fenced_turtle_block"
REASON_EXTRACTOR_EMPTY = "extractor_returned_nothing"
REASON_OUTSIDE_FENCE = "extraction_fell_outside_closed_fences"
REASON_TOO_SHORT = "recovered_text_below_min_bytes"
REASON_UNPARSEABLE = "recovered_text_does_not_parse"
REASON_NOT_BETTER = "recovered_text_not_better_than_original"


def diagnose_raw(raw_text: str) -> Tuple[str, str]:
    """
    Explain why no usable Turtle could be pulled out of *raw_text*.

    Returns ``(reason, detail)``.
    """
    if not raw_text.strip():
        return REASON_RAW_EMPTY, "raw_response.txt is blank"

    # A closed Turtle fence exists, so the reply was not cut short -- the block
    # simply did not survive rdflib.  The caller replaces the detail with the
    # actual parser error.
    blocks = turtle_fenced_blocks(raw_text)
    if blocks:
        return (
            REASON_UNPARSEABLE,
            "the longest closed Turtle fence (%d B) does not parse" % len(blocks[0]),
        )

    if response_is_truncated(raw_text):
        return (
            REASON_UNTERMINATED,
            "response was cut off inside the turtle code fence (F2: "
            "max_new_tokens=1024), so the closing fence was never emitted; "
            "the ontology is incomplete and regeneration is required",
        )

    if len(list(_FENCE_TOKEN_RE.finditer(raw_text))) % 2 == 1:
        return (
            REASON_UNTERMINATED,
            "odd number of code fences and no Turtle after the last one",
        )

    if not looks_like_turtle(raw_text):
        return (
            REASON_NO_TURTLE,
            "raw_response.txt (%d chars) contains no @prefix/@base/' a owl:' "
            "anywhere -- the model never produced an ontology" % len(raw_text),
        )

    return (
        REASON_NO_FENCED_BLOCK,
        "Turtle markers are present but not inside any closed code fence",
    )


# ---------------------------------------------------------------------------
# Per-directory recovery
# ---------------------------------------------------------------------------

EXTRACTOR_IMPORTED = "run_generation.extract_turtle"
EXTRACTOR_FALLBACK = "fallback_longest_fence"
#: Recovery from an unterminated fence.  Always paired with TRUNCATION_BANNER;
#: an artefact carrying this provenance is partial and must never be counted as
#: a complete generation.
EXTRACTOR_TRUNCATED = "open_fence_tail_truncated"


class Extractor:
    """
    Composed extractor: the imported ``run_generation.extract_turtle`` first,
    the local corrected fallback only if that produces nothing parseable.
    """

    def __init__(
        self,
        primary: Optional[Callable[[str], str]] = None,
        repair: Optional[Callable[[str], str]] = None,
        use_fallback: bool = True,
        use_repair: bool = True,
        recover_truncated: bool = False,
    ):
        # Off by default, and deliberately so.  Writing a partial ODP back as
        # ontology.ttl would misrepresent what the model produced -- unless the
        # artefact is stamped with TRUNCATION_BANNER, which is what this mode
        # does and why it is safe to offer at all.
        self.recover_truncated = recover_truncated
        self.primary = primary
        self.primary_error: Optional[str] = None
        if self.primary is None:
            try:
                self.primary = load_extract_turtle()
            except Exception as exc:  # pragma: no cover - defensive
                self.primary_error = "%s: %s" % (type(exc).__name__, exc)
        self.repair = repair
        if self.repair is None and use_repair:
            try:
                self.repair = load_turtle_repair()
            except Exception:  # pragma: no cover - defensive
                self.repair = None
        self.use_fallback = use_fallback
        self.use_repair = use_repair

    def _attempts(self, raw_text: str):
        """Yield ``(extractor_name, text)`` candidates in preference order."""
        if self.primary is not None:
            try:
                yield EXTRACTOR_IMPORTED, self.primary(raw_text) or ""
            except Exception as exc:  # pragma: no cover - defensive
                self.primary_error = "%s: %s" % (type(exc).__name__, exc)
        if self.use_fallback:
            yield EXTRACTOR_FALLBACK, fallback_extract_turtle(raw_text)

    def extract(self, raw_text: str, min_bytes: int = MIN_ONTOLOGY_BYTES) -> Dict:
        """
        Return a dict describing the best validated extraction.

        Keys: ``ok``, ``text``, ``extractor``, ``repaired``, ``dropped_bytes``,
        ``triples``, ``error``.  ``ok`` is False when nothing parseable could be
        produced.

        Two guards keep a *recovery* tool from inventing content:

        * When the reply contains closed Turtle fences, the accepted text must
          be one of them verbatim.  ``run_generation.extract_turtle`` has its
          own unfenced fallback that walks the raw text to end-of-buffer; that
          is the right call at generation time but here it would splice a
          truncated tail onto an otherwise complete block.
        * When the reply contains no closed Turtle fence and was cut off inside
          an open one (F2), nothing is accepted at all.  A partial ODP written
          back as ``ontology.ttl`` would look complete and quietly misrepresent
          what the model actually produced.
        """
        allowed = set(turtle_fenced_blocks(raw_text))
        truncated = response_is_truncated(raw_text)

        last_error = ""
        seen = set()
        for name, text in self._attempts(raw_text):
            text = (text or "").strip()
            if not text or text in seen:
                continue
            seen.add(text)
            if allowed:
                if text not in allowed:
                    # The extractor strayed outside the closed fences.
                    last_error = last_error or REASON_OUTSIDE_FENCE
                    continue
            elif truncated:
                last_error = last_error or REASON_UNTERMINATED
                continue
            if len(text.encode("utf-8")) < min_bytes:
                last_error = last_error or REASON_TOO_SHORT
                continue
            ok, n_triples, err = validate_turtle(text)
            if ok:
                return {
                    "ok": True,
                    "text": text,
                    "extractor": name,
                    "repaired": False,
                    "dropped_bytes": 0,
                    "triples": n_triples,
                    "error": "",
                }
            last_error = err
            # Second chance: the same conservative cleanup the generation
            # pipeline already applies to HuggingFace output.
            if self.use_repair and self.repair is not None:
                try:
                    repaired = (self.repair(text) or "").strip()
                except Exception:  # pragma: no cover - defensive
                    repaired = ""
                if (
                    repaired
                    and repaired != text
                    and len(repaired.encode("utf-8")) >= min_bytes
                ):
                    ok, n_triples, err2 = validate_turtle(repaired)
                    if ok:
                        return {
                            "ok": True,
                            "text": repaired,
                            "extractor": name,
                            "repaired": True,
                            "dropped_bytes": len(text.encode("utf-8"))
                            - len(repaired.encode("utf-8")),
                            "triples": n_triples,
                            "error": "",
                        }
                    last_error = err2 or last_error
        # Last resort: the reply was cut off inside an open fence, so no closed
        # block exists to accept verbatim.  Recovering here is only defensible
        # because the artefact is stamped with TRUNCATION_BANNER on the way to
        # disk -- a partial ODP that announces itself cannot be mistaken for a
        # complete one, which is the whole objection this guard exists to raise.
        if self.recover_truncated and truncated and not allowed:
            tail = open_fence_tail(raw_text)
            if tail:
                for candidate, repaired in ((tail, False), (drop_incomplete_tail(tail), True)):
                    if not candidate:
                        continue
                    if len(candidate.encode("utf-8")) < min_bytes:
                        continue
                    ok, n_triples, err = validate_turtle(candidate)
                    if ok:
                        return {
                            "ok": True,
                            "text": candidate,
                            "extractor": EXTRACTOR_TRUNCATED,
                            "repaired": repaired,
                            "dropped_bytes": len(tail.encode("utf-8"))
                            - len(candidate.encode("utf-8")),
                            "triples": n_triples,
                            "truncated_recovery": True,
                            "error": "",
                        }
                    last_error = err or last_error

        return {
            "ok": False,
            "text": "",
            "extractor": None,
            "repaired": False,
            "dropped_bytes": 0,
            "triples": 0,
            "truncated_recovery": False,
            "error": last_error,
        }


def process_scenario_dir(
    scenario_dir: Path,
    extractor: "Extractor",
    apply: bool = False,
    min_bytes: int = MIN_ONTOLOGY_BYTES,
    include_unparseable: bool = False,
) -> Dict:
    """
    Inspect (and with ``apply=True`` repair) a single
    ``outputs/{model}/{config}/{scenario_id}/`` directory.

    Returns a record whose ``status`` is one of ``healthy``, ``recovered``,
    ``unrecoverable`` or ``skipped_unparseable``.  Nothing is written unless
    ``apply`` is True *and* the replacement text has already been parsed
    successfully by rdflib.
    """
    ontology_path = scenario_dir / ONTOLOGY_NAME
    raw_path = scenario_dir / RAW_RESPONSE_NAME

    state, size, triples = classify_ontology(ontology_path, min_bytes=min_bytes)

    record: Dict = {
        "model": scenario_dir.parent.parent.name,
        "config": scenario_dir.parent.name,
        "scenario_id": scenario_dir.name,
        "path": str(ontology_path),
        "original_state": state,
        "original_bytes": size,
        "original_triples": triples,
    }

    if state == STATE_OK:
        record["status"] = "healthy"
        return record

    # ---- gather the raw response -----------------------------------------
    raw_text = (
        raw_path.read_text(encoding="utf-8", errors="replace")
        if raw_path.exists()
        else None
    )
    record["raw_response_bytes"] = len(raw_text.encode("utf-8")) if raw_text else 0

    if raw_text is None:
        result = {
            "ok": False,
            "error": "",
            "text": "",
            "extractor": None,
            "repaired": False,
            "triples": 0,
        }
        reason, detail = REASON_NO_RAW, "%s does not exist" % raw_path
    else:
        result = extractor.extract(raw_text, min_bytes=min_bytes)
        reason, detail = "", ""
        if not result["ok"]:
            reason, detail = diagnose_raw(raw_text)
            if reason == REASON_UNPARSEABLE and result["error"]:
                detail = result["error"]

    # An ontology.ttl that is big enough but does not parse is a genuine model
    # failure, not F1 damage.  Only treat it as damaged when the raw response
    # actually holds something strictly better; otherwise leave it alone so the
    # report never conflates "the extractor ate it" with "the model failed".
    if state == STATE_UNPARSEABLE and not result["ok"] and not include_unparseable:
        record["status"] = "skipped_unparseable"
        record["reason"] = reason or REASON_UNPARSEABLE
        record["detail"] = detail
        return record

    record["damaged"] = True

    if not result["ok"]:
        record["status"] = "unrecoverable"
        record["reason"] = reason or REASON_EXTRACTOR_EMPTY
        record["detail"] = detail
        return record

    recovered_text = result["text"]
    recovered_bytes = len(recovered_text.encode("utf-8"))

    # Never trade a larger file for a smaller one when the original was merely
    # unparseable rather than destroyed.
    if state == STATE_UNPARSEABLE and recovered_bytes <= size:
        record["status"] = "unrecoverable"
        record["reason"] = REASON_NOT_BETTER
        record["detail"] = (
            "candidate (%d B) is no larger than the %d B file already on disk"
            % (recovered_bytes, size)
        )
        return record

    record.update(
        {
            "status": "recovered",
            "extractor": result["extractor"],
            "repaired": result["repaired"],
            "repair_dropped_bytes": result.get("dropped_bytes", 0),
            "recovered_bytes": recovered_bytes,
            "recovered_triples": result["triples"],
            "backup_path": str(ontology_path.with_name(ONTOLOGY_NAME + ORIG_SUFFIX)),
        }
    )

    truncated_recovery = bool(result.get("truncated_recovery"))
    record["truncated_recovery"] = truncated_recovery
    record["written"] = (
        _write_recovery(ontology_path, recovered_text, truncated=truncated_recovery)
        if apply
        else False
    )
    return record


def _write_recovery(ontology_path: Path, text: str, truncated: bool = False) -> bool:
    """
    Preserve the damaged original, then write *text*.

    The backup is written first and is **never** overwritten, so re-running
    ``--apply`` can only ever add provenance, never clobber it.
    """
    backup_path = ontology_path.with_name(ONTOLOGY_NAME + ORIG_SUFFIX)
    if not backup_path.exists():
        original = ""
        if ontology_path.exists():
            original = ontology_path.read_text(encoding="utf-8", errors="replace")
        backup_path.write_text(original, encoding="utf-8", newline="\n")
    payload = text if text.endswith("\n") else text + "\n"
    if truncated and not payload.lstrip().startswith(TRUNCATION_BANNER):
        # Provenance travels with the artefact, not merely in a side report:
        # anyone who opens this file, and any scorer that greps it, sees that
        # the generation was cut short and the ontology is partial.
        payload = (
            TRUNCATION_BANNER + "\n"
            "# The model's reply was truncated by the generation token budget.\n"
            "# Recovered from raw_response.txt; the trailing axiom(s) are missing.\n"
            "# Do NOT count this artefact as a complete generation.\n"
            + payload
        )
    ontology_path.write_text(payload, encoding="utf-8", newline="\n")
    return True


# ---------------------------------------------------------------------------
# Corpus walk
# ---------------------------------------------------------------------------

def iter_scenario_dirs(outputs_root: Path):
    """Yield every ``outputs/{model}/{config}/{scenario_id}/`` directory."""
    outputs_root = Path(outputs_root)
    if not outputs_root.is_dir():
        return
    for model_dir in sorted(p for p in outputs_root.iterdir() if p.is_dir()):
        for config_dir in sorted(p for p in model_dir.iterdir() if p.is_dir()):
            for scenario_dir in sorted(p for p in config_dir.iterdir() if p.is_dir()):
                yield scenario_dir


def _blank_bucket() -> Dict:
    return {
        "scanned": 0,
        "healthy": 0,
        "damaged": 0,
        "recovered": 0,
        "unrecoverable": 0,
        "skipped_unparseable": 0,
        "reasons": {},
    }


def run_recovery(
    outputs_root: Path,
    apply: bool = False,
    min_bytes: int = MIN_ONTOLOGY_BYTES,
    include_unparseable: bool = False,
    extractor: Optional["Extractor"] = None,
) -> Dict:
    """
    Walk the corpus and build the report.

    With ``apply=False`` (the default) this function touches nothing on disk.
    """
    outputs_root = Path(outputs_root)
    extractor = extractor or Extractor()

    records: List[Dict] = []
    by_model_config: Dict[str, Dict[str, Dict]] = {}
    totals = _blank_bucket()
    extractor_usage: Dict[str, int] = {}

    for scenario_dir in iter_scenario_dirs(outputs_root):
        record = process_scenario_dir(
            scenario_dir,
            extractor=extractor,
            apply=apply,
            min_bytes=min_bytes,
            include_unparseable=include_unparseable,
        )
        records.append(record)

        bucket = by_model_config.setdefault(record["model"], {}).setdefault(
            record["config"], _blank_bucket()
        )
        targets = (totals, bucket)
        for target in targets:
            target["scanned"] += 1

        status = record["status"]
        if status == "healthy":
            for target in targets:
                target["healthy"] += 1
        elif status == "skipped_unparseable":
            for target in targets:
                target["skipped_unparseable"] += 1
        elif status == "recovered":
            for target in targets:
                target["damaged"] += 1
                target["recovered"] += 1
            name = record.get("extractor") or "unknown"
            extractor_usage[name] = extractor_usage.get(name, 0) + 1
        elif status == "unrecoverable":
            reason = record.get("reason", "unknown")
            for target in targets:
                target["damaged"] += 1
                target["unrecoverable"] += 1
                target["reasons"][reason] = target["reasons"].get(reason, 0) + 1

    recovered = [r for r in records if r["status"] == "recovered"]
    unrecoverable = [r for r in records if r["status"] == "unrecoverable"]
    skipped = [r for r in records if r["status"] == "skipped_unparseable"]

    upstream_fixed = (
        extractor.primary is not None
        and extractor_usage.get(EXTRACTOR_FALLBACK, 0) == 0
    )

    return {
        "tool": "scripts/recover_outputs.py",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "outputs_root": str(outputs_root),
        "dry_run": not apply,
        "applied": bool(apply),
        "min_ontology_bytes": min_bytes,
        "include_unparseable": include_unparseable,
        "backup_suffix": ORIG_SUFFIX,
        "extractor": {
            "imported_from": str(RUN_GENERATION_PATH),
            "import_ok": extractor.primary is not None,
            "import_error": extractor.primary_error,
            "fallback_enabled": extractor.use_fallback,
            "repair_enabled": bool(extractor.use_repair and extractor.repair is not None),
            "usage": extractor_usage,
            "upstream_looks_fixed": upstream_fixed,
            "note": (
                "Extraction is imported from run_generation so the two cannot "
                "drift. A non-zero '" + EXTRACTOR_FALLBACK + "' count means "
                "run_generation.extract_turtle is still the buggy first-match "
                "version (F1)."
            ),
        },
        "totals": totals,
        "by_model_config": by_model_config,
        "recovered": recovered,
        "unrecoverable": unrecoverable,
        "skipped_unparseable": skipped,
    }


def write_report(report: Dict, report_path: Path) -> None:
    report_path = Path(report_path)
    if report_path.parent and not report_path.parent.exists():
        report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def format_summary(report: Dict) -> str:
    lines: List[str] = []
    mode = "APPLY (files written)" if report["applied"] else "DRY RUN (nothing written)"
    lines.append("mode: %s" % mode)
    lines.append("outputs root: %s" % report["outputs_root"])
    ext = report["extractor"]
    lines.append(
        "extractor: import_ok=%s usage=%s upstream_looks_fixed=%s"
        % (ext["import_ok"], ext["usage"] or {}, ext["upstream_looks_fixed"])
    )
    lines.append("")
    header = "%-34s %-24s %5s %5s %5s %6s" % (
        "model",
        "config",
        "scan",
        "dmg",
        "rec",
        "unrec",
    )
    lines.append(header)
    lines.append("-" * len(header))
    for model in sorted(report["by_model_config"]):
        for config in sorted(report["by_model_config"][model]):
            b = report["by_model_config"][model][config]
            if b["damaged"] == 0:
                continue
            lines.append(
                "%-34s %-24s %5d %5d %5d %6d"
                % (model, config, b["scanned"], b["damaged"], b["recovered"], b["unrecoverable"])
            )
    t = report["totals"]
    lines.append("-" * len(header))
    lines.append(
        "%-34s %-24s %5d %5d %5d %6d"
        % ("TOTAL", "", t["scanned"], t["damaged"], t["recovered"], t["unrecoverable"])
    )
    lines.append("")

    reasons: Dict[str, int] = {}
    for rec in report["unrecoverable"]:
        key = rec.get("reason", "unknown")
        reasons[key] = reasons.get(key, 0) + 1
    if reasons:
        lines.append("unrecoverable by reason:")
        for reason, count in sorted(reasons.items(), key=lambda kv: -kv[1]):
            lines.append("  %4d  %s" % (count, reason))
    if t["skipped_unparseable"]:
        lines.append("")
        lines.append(
            "%d ontology.ttl files are large enough but do not parse and have nothing"
            % t["skipped_unparseable"]
        )
        lines.append(
            "better in raw_response.txt; they are genuine model failures rather than F1"
        )
        lines.append(
            "damage and were left untouched (--include-unparseable to list them)."
        )
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Recover ontology.ttl files destroyed by the F1 extractor bug by "
            "re-extracting Turtle from the raw_response.txt already on disk. "
            "Makes no API calls. Dry run unless --apply is given."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--outputs",
        type=Path,
        default=DEFAULT_OUTPUTS_ROOT,
        help="root of the generation corpus (default: %s)" % DEFAULT_OUTPUTS_ROOT,
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=DEFAULT_REPORT_PATH,
        help="where to write the JSON report (default: %s)" % DEFAULT_REPORT_PATH,
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="actually write recovered files (default: dry run, writes nothing)",
    )
    parser.add_argument(
        "--no-report", action="store_true", help="do not write the JSON report"
    )
    parser.add_argument(
        "--min-bytes",
        type=int,
        default=MIN_ONTOLOGY_BYTES,
        help="ontology.ttl below this size counts as damaged (default: %d)"
        % MIN_ONTOLOGY_BYTES,
    )
    parser.add_argument(
        "--include-unparseable",
        action="store_true",
        help=(
            "also count large-but-unparseable ontology.ttl files as damaged even "
            "when nothing better exists in raw_response.txt (audit mode)"
        ),
    )
    parser.add_argument(
        "--no-fallback",
        action="store_true",
        help=(
            "use only run_generation.extract_turtle; do not fall back to the "
            "local corrected extractor when it returns nothing usable"
        ),
    )
    parser.add_argument(
        "--no-repair",
        action="store_true",
        help=(
            "do not try run_generation.maybe_fix_common_turtle_issues on a block "
            "that fails to parse"
        ),
    )
    parser.add_argument(
        "--recover-truncated",
        action="store_true",
        help=(
            "also recover replies cut off inside an open code fence (F2). The "
            "recovered ontology is PARTIAL and is stamped with a "
            "'%s' banner so it can never be counted as a complete generation. "
            "Off by default." % TRUNCATION_BANNER
        ),
    )
    parser.add_argument("--quiet", action="store_true", help="suppress the summary table")
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)

    # rdflib is noisy about the malformed IRIs these broken outputs are full of.
    logging.getLogger("rdflib").setLevel(logging.CRITICAL)

    outputs_root = Path(args.outputs)
    if not outputs_root.is_dir():
        print("error: outputs root not found: %s" % outputs_root, file=sys.stderr)
        return 2

    extractor = Extractor(
        use_fallback=not args.no_fallback,
        use_repair=not args.no_repair,
        recover_truncated=args.recover_truncated,
    )
    report = run_recovery(
        outputs_root,
        apply=args.apply,
        min_bytes=args.min_bytes,
        include_unparseable=args.include_unparseable,
        extractor=extractor,
    )

    if not args.no_report:
        write_report(report, Path(args.report))

    if not args.quiet:
        print(format_summary(report))
        if not args.no_report:
            print("\nreport: %s" % Path(args.report))
        if not args.apply:
            print(
                "\nDRY RUN -- no ontology.ttl was modified. "
                "Re-run with --apply to write."
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
