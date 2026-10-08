#!/usr/bin/env python3
"""
audit_artifacts.py -- read-only integrity auditor for the ODPGen artifact corpus.

The ODPGen repository ships several evaluation artifact sets that disagree with each
other, and several evaluation metrics that are degenerate (they reward empty files).
This auditor re-derives the ground facts directly from the artifacts on disk so that
the defects cannot be reintroduced silently.  It never writes to the corpus, never
calls an LLM or any network service, and exits non-zero the moment a check fails, so
it can be wired straight into CI.

Checks
------
  empty-outputs        outputs/<model>/<config>/<sid>/ontology.ttl files that are
                       byte-empty, unparseable, or parse to zero triples.  The
                       artifacts on disk were produced by a Turtle extractor that
                       took the FIRST fenced span in the reply; since the prompt's
                       own output-format instruction contains a literal
                       ```turtle ... ``` example, the capture returned the prompt's
                       placeholder rather than the model's ontology.
  cq-contamination     eval_results/odp_eval/**/*.json record a "cqs" list that is
                       compared here against the authoritative cq_list in
                       data/scenarios/pattern_scenarios.json.  The recorded lists
                       were produced by a scraper that began collecting bullets at
                       the first line matching /competency questions?/i -- in most
                       prompts the opening instruction sentence -- and so harvested
                       the "Modeling constraints" bullets instead.
  artifact-disagreement
                       results/<model>/<config>/<sid>.json vs
                       eval_results/odp_eval/<model>/<config>/<sid>.json on parse
                       success and triple count.  Nothing in the repo marks which
                       set is authoritative.
  degenerate-scores    Outputs that earn a high structural score while being empty
                       or near-empty, plus models that declare no rdfs:domain and
                       no rdfs:range anywhere (such a model cannot produce a
                       well-formed ODP and must never be selected as a winner).

Usage
-----
    python scripts/audit_artifacts.py
    python scripts/audit_artifacts.py --check cq-contamination --check empty-outputs
    python scripts/audit_artifacts.py --self-test

Exit codes
----------
    0  every requested check passed
    1  at least one check reported a problem
    2  the auditor could not run (missing corpus, unreadable input)
"""

from __future__ import annotations

import argparse
import csv
import datetime as _dt
import json
import logging
import os
import re
import sys
import warnings
from collections import Counter, defaultdict
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

# rdflib is chatty about the malformed base URIs in this corpus; the auditor
# reports parse failures itself, so silence the library's own logging.
warnings.filterwarnings("ignore")
for _name in ("rdflib", "rdflib.term", "rdflib.plugins.parsers.notation3"):
    logging.getLogger(_name).setLevel(logging.CRITICAL)

try:
    from rdflib import Graph
    from rdflib.namespace import RDFS
except ImportError:  # pragma: no cover - dependency is declared in requirements.txt
    sys.stderr.write("audit_artifacts.py requires rdflib (pip install rdflib)\n")
    raise SystemExit(2)


# --------------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------------

TOOL_VERSION = "1.0"

#: A file this small cannot hold a Turtle ontology.  The F1 extraction bug writes
#: exactly three bytes ("...", the prompt's own placeholder), so the threshold is
#: deliberately generous.
EMPTY_BYTES_DEFAULT = 50

#: At or below this triple count a graph carries no reusable pattern.
DEGENERATE_TRIPLES_DEFAULT = 20

#: Structural score at or above which an output counts as "scoring well".
DEGENERATE_SCORE_DEFAULT = 0.75

#: The parse formats batch_evaluate.py itself tries, in the same order.
PARSE_FORMATS = ("turtle", "xml", "n3")

#: Prompt configurations that supply no competency questions by design, so
#: recording zero CQs for them is correct rather than a contamination symptom.
CQ_FREE_CONFIGS = frozenset({"scenario-only"})

#: The four (model, config) pairs reported in the paper's functional-evaluation
#: table (ISWC2026_ODP_paper/tables/tab_functional_results.tex).  Their CQ pass
#: rates are the paper's headline functional numbers.
PAPER_FUNCTIONAL_CONFIGS: Tuple[Tuple[str, str], ...] = (
    ("gemini-3.1-pro-preview", "scenario-cq-reasoning"),
    ("gpt-5.4", "scenario-cq-constraints"),
    ("meta-llama_Llama-3.1-8B-Instruct", "scenario-cq"),
    ("meta-llama_Llama-2-70b-chat-hf", "cq-only"),
)

CHECK_IDS = ("empty-outputs", "cq-contamination", "artifact-disagreement", "degenerate-scores")

_TURTLE_FENCE_CLOSED = re.compile(r"```(?:turtle|ttl|rdf)[ \t]*\r?\n(.*?)```", re.IGNORECASE | re.DOTALL)
_TURTLE_FENCE_OPEN = re.compile(r"```(?:turtle|ttl|rdf)[ \t]*\r?\n", re.IGNORECASE)


# --------------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------------

def _posix(path: str) -> str:
    return path.replace("\\", "/")


def _rel(root: str, path: str) -> str:
    try:
        return _posix(os.path.relpath(path, root))
    except ValueError:  # different drives on Windows
        return _posix(path)


def _read_text(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        return fh.read()


def _read_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        return json.load(fh)


def parse_rdf(text: str) -> Tuple[Optional[Graph], Optional[str]]:
    """Parse `text` with the same format ladder batch_evaluate.py uses."""
    if not text.strip():
        # rdflib happily parses whitespace as an empty graph; make that explicit.
        return Graph(), "empty"
    for fmt in PARSE_FORMATS:
        try:
            g = Graph()
            g.parse(data=text, format=fmt)
            return g, fmt
        except Exception:
            continue
    return None, None


def recover_turtle_from_raw(raw_text: str) -> Tuple[Optional[Graph], str]:
    """
    Best-effort recovery of the ontology a model actually emitted.

    Returns (graph, mode) where mode is one of:
      "closed_fence"  a complete ```turtle ... ``` block parsed cleanly
      "truncated"     only the text after the last opening fence parsed, after
                      dropping trailing lines -- the response was cut off
      "none"          nothing recoverable
    """
    # 1. properly closed fences (ignore the prompt's own "..." placeholder)
    blocks = [b for b in _TURTLE_FENCE_CLOSED.findall(raw_text) if b.strip() not in ("", "...")]
    for block in reversed(blocks):
        g, _ = parse_rdf(block)
        if g is not None and len(g) > 0:
            return g, "closed_fence"

    # 2. an unterminated fence: the generation hit the token ceiling (F2)
    last_open = None
    for match in _TURTLE_FENCE_OPEN.finditer(raw_text):
        last_open = match
    if last_open is not None:
        tail = raw_text[last_open.end():].split("```")[0]
        lines = tail.splitlines()
        for drop in range(0, min(len(lines), 60)):
            candidate = "\n".join(lines[: len(lines) - drop])
            if not candidate.strip():
                break
            g, _ = parse_rdf(candidate)
            if g is not None and len(g) > 0:
                return g, "truncated"
    return None, "none"


def normalise_cq(text: str) -> str:
    """
    Canonical form for competency-question comparison.

    batch_evaluate.py:extract_cqs() appends "?" to every bullet it scrapes, and the
    prompts use backticks around code spans, so both are stripped before matching.
    """
    s = str(text).strip().lower()
    s = s.replace("`", "").replace("’", "'")
    s = re.sub(r"\s+", " ", s)
    s = s.strip(" \t\r\n")
    s = s.rstrip("?.!;:, ")
    return s


def structural_score(record: Dict[str, Any]) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """
    Recompute the paper's per-file structural score from an eval_results record.

    The published aggregate is the mean of the consistency rate and the average
    OOPS! score (ISWC2026_ODP_paper/tables/tab_structural_evaluation.tex).  The
    per-file OOPS score is 1 / (1 + pitfalls_total); that formula reproduces every
    avg_oops_score in eval/odp_eval_ranking.csv exactly.

    Returns (consistency_component, oops_component, structural_score); any element
    is None when the underlying stage errored out.
    """
    reasoner = record.get("reasoner") or {}
    oops = record.get("oops") or {}

    consistency: Optional[float]
    if "error" in reasoner or "consistent" not in reasoner:
        consistency = None
    else:
        consistency = 1.0 if reasoner.get("consistent") else 0.0

    oops_component: Optional[float]
    if "error" in oops or oops.get("status_code") != 200:
        oops_component = None
    else:
        oops_component = 1.0 / (1.0 + float(oops.get("pitfalls_total", 0) or 0))

    if consistency is None or oops_component is None:
        return consistency, oops_component, None
    return consistency, oops_component, (consistency + oops_component) / 2.0


# --------------------------------------------------------------------------------
# Corpus loading
# --------------------------------------------------------------------------------

def load_scenarios(root: str) -> Dict[str, Dict[str, Any]]:
    """scenario_id -> {"cq_list": [...], "normalised": frozenset}"""
    path = os.path.join(root, "data", "scenarios", "pattern_scenarios.json")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"authoritative CQ source not found: {path}")
    raw = _read_json(path)
    if isinstance(raw, dict):
        raw = list(raw.values())
    out: Dict[str, Dict[str, Any]] = {}
    for item in raw:
        sid = str(item.get("scenario_id", "")).strip()
        if not sid:
            continue
        cqs = [str(c) for c in (item.get("cq_list") or [])]
        out[sid] = {"cq_list": cqs, "normalised": frozenset(normalise_cq(c) for c in cqs)}
    return out


def load_prompt_bullets(root: str) -> Dict[str, str]:
    """normalised bullet text -> prompt file it came from (evidence for F4)."""
    bullets: Dict[str, str] = {}
    prompt_dir = os.path.join(root, "prompts")
    if not os.path.isdir(prompt_dir):
        return bullets
    for name in sorted(os.listdir(prompt_dir)):
        if not name.endswith(".txt"):
            continue
        for line in _read_text(os.path.join(prompt_dir, name)).splitlines():
            stripped = line.strip()
            if stripped.startswith("-") or re.match(r"^\d+\.\s", stripped):
                text = re.sub(r"^(-+|\d+\.)\s*", "", stripped)
                key = normalise_cq(text)
                if key and key not in bullets:
                    bullets[key] = f"prompts/{name}"
    return bullets


def iter_generation_outputs(root: str) -> Iterable[Tuple[str, str, str, str]]:
    """Yield (model, config, scenario_id, ontology_path) for outputs/*/*/*/ontology.ttl."""
    base = os.path.join(root, "outputs")
    if not os.path.isdir(base):
        return
    for model in sorted(os.listdir(base)):
        model_dir = os.path.join(base, model)
        if not os.path.isdir(model_dir):
            continue
        for config in sorted(os.listdir(model_dir)):
            config_dir = os.path.join(model_dir, config)
            if not os.path.isdir(config_dir):
                continue
            for sid in sorted(os.listdir(config_dir)):
                onto = os.path.join(config_dir, sid, "ontology.ttl")
                if os.path.isfile(onto):
                    yield model, config, sid, onto


def iter_eval_records(root: str) -> Iterable[Tuple[str, str, str, str]]:
    """Yield (model, config, scenario_id, json_path) for eval_results/odp_eval/*/*/*.json."""
    base = os.path.join(root, "eval_results", "odp_eval")
    if not os.path.isdir(base):
        return
    for model in sorted(os.listdir(base)):
        model_dir = os.path.join(base, model)
        if not os.path.isdir(model_dir):
            continue
        for config in sorted(os.listdir(model_dir)):
            config_dir = os.path.join(model_dir, config)
            if not os.path.isdir(config_dir):
                continue
            for name in sorted(os.listdir(config_dir)):
                if name.endswith(".json"):
                    yield model, config, os.path.splitext(name)[0], os.path.join(config_dir, name)


def iter_result_records(root: str) -> Iterable[Tuple[str, str, str, str]]:
    """Yield (model, config, scenario_id, json_path) for results/*/*/*.json."""
    base = os.path.join(root, "results")
    if not os.path.isdir(base):
        return
    for model in sorted(os.listdir(base)):
        model_dir = os.path.join(base, model)
        if not os.path.isdir(model_dir):
            continue
        for config in sorted(os.listdir(model_dir)):
            config_dir = os.path.join(model_dir, config)
            if not os.path.isdir(config_dir):
                continue
            for name in sorted(os.listdir(config_dir)):
                if name.endswith(".json"):
                    yield model, config, os.path.splitext(name)[0], os.path.join(config_dir, name)


# --------------------------------------------------------------------------------
# Check 1 -- empty outputs
# --------------------------------------------------------------------------------

def check_empty_outputs(root: str, empty_bytes: int) -> Dict[str, Any]:
    per_pair: Dict[Tuple[str, str], Counter] = defaultdict(Counter)
    per_model: Dict[str, Counter] = defaultdict(Counter)
    offenders: List[Dict[str, Any]] = []

    for model, config, sid, path in iter_generation_outputs(root):
        size = os.path.getsize(path)
        text = _read_text(path)
        graph, fmt = parse_rdf(text)
        triples = len(graph) if graph is not None else 0

        too_small = size < empty_bytes
        unparseable = graph is None
        zero_triples = graph is not None and triples == 0
        degenerate = too_small or unparseable or zero_triples

        for bucket in (per_pair[(model, config)], per_model[model]):
            bucket["total"] += 1
            bucket["bytes_below_threshold"] += int(too_small)
            bucket["unparseable"] += int(unparseable)
            bucket["parses_to_zero_triples"] += int(zero_triples)
            bucket["empty_or_unusable"] += int(degenerate)

        if not degenerate:
            continue

        raw_path = os.path.join(os.path.dirname(path), "raw_response.txt")
        recovery = "no_raw_response"
        recovered_triples = 0
        if os.path.isfile(raw_path):
            rec_graph, mode = recover_turtle_from_raw(_read_text(raw_path))
            recovery = mode
            recovered_triples = len(rec_graph) if rec_graph is not None else 0

        for bucket in (per_pair[(model, config)], per_model[model]):
            bucket[f"recovery_{recovery}"] += 1

        offenders.append({
            "model": model,
            "config": config,
            "scenario_id": sid,
            "path": _rel(root, path),
            "bytes": size,
            "parsed": graph is not None,
            "parse_format": fmt,
            "triples": triples,
            "content_preview": text.strip()[:60],
            "raw_response_recovery": recovery,
            "recovered_triples": recovered_triples,
        })

    total = sum(c["total"] for c in per_model.values())
    bad = sum(c["empty_or_unusable"] for c in per_model.values())

    findings: List[str] = []
    for model in sorted(per_model):
        c = per_model[model]
        if c["empty_or_unusable"]:
            findings.append(
                f"{model}: {c['empty_or_unusable']}/{c['total']} outputs are empty or unusable "
                f"({c['bytes_below_threshold']} under {empty_bytes} bytes, "
                f"{c['unparseable']} unparseable, {c['parses_to_zero_triples']} parse to zero triples)"
            )

    if total == 0:
        status = "SKIP"
        headline = "no outputs/<model>/<config>/<scenario>/ontology.ttl files found"
    elif bad:
        status = "FAIL"
        headline = f"{bad}/{total} generated ontologies are empty or unusable"
    else:
        status = "PASS"
        headline = f"all {total} generated ontologies carry parseable content"

    return {
        "id": "empty-outputs",
        "title": "Empty / unusable ontology.ttl outputs",
        "status": status,
        "headline": headline,
        "threshold_bytes": empty_bytes,
        "totals": {"files": total, "empty_or_unusable": bad},
        "per_model": {m: dict(c) for m, c in sorted(per_model.items())},
        "per_config": {f"{m}/{c}": dict(v) for (m, c), v in sorted(per_pair.items())},
        "findings": findings,
        "offenders": offenders,
        "root_cause": (
            "These ontology.ttl files were written by a Turtle extractor that took the "
            "FIRST ```turtle ... ``` span in the reply.  The prompt's own output-format "
            "instruction contains that literal fence, so the capture returned the prompt's "
            "placeholder ('...') instead of the ontology below it.  A "
            "raw_response_recovery of 'truncated' additionally means the generation hit "
            "max_new_tokens before closing its fence, so no complete ontology exists in "
            "the raw response either and that output needs regeneration.  If this check "
            "fails again after the extractor is fixed, the outputs must be regenerated."
        ),
    }


# --------------------------------------------------------------------------------
# Check 2 -- competency-question contamination
# --------------------------------------------------------------------------------

def check_cq_contamination(root: str) -> Dict[str, Any]:
    scenarios = load_scenarios(root)
    prompt_bullets = load_prompt_bullets(root)

    per_pair: Dict[Tuple[str, str], Dict[str, Any]] = {}
    fake_counter: Counter = Counter()
    unknown_scenarios: Counter = Counter()

    for model, config, sid, path in iter_eval_records(root):
        key = (model, config)
        stats = per_pair.setdefault(key, {
            "files": 0,
            "files_with_cq_field": 0,
            "files_with_contamination": 0,
            "files_with_zero_real_cqs": 0,
            "recorded_cqs": 0,
            "real_cqs_evaluated": 0,
            "fake_cqs_recorded": 0,
            "real_cqs_available": 0,
            "cq_verification_ran": 0,
            "examples": [],
        })
        stats["files"] += 1

        try:
            record = _read_json(path)
        except Exception as exc:
            stats["examples"].append({"scenario_id": sid, "error": f"unreadable: {exc}"})
            continue

        verification = record.get("cq_verification") or {}
        if not verification.get("skipped"):
            stats["cq_verification_ran"] += 1

        if "cqs" not in record:
            continue
        stats["files_with_cq_field"] += 1

        recorded = [str(c) for c in (record.get("cqs") or [])]
        stats["recorded_cqs"] += len(recorded)

        truth = scenarios.get(sid)
        if truth is None:
            unknown_scenarios[sid] += 1
            continue
        stats["real_cqs_available"] += len(truth["cq_list"])

        real_hits = 0
        fakes: List[str] = []
        for cq in recorded:
            if normalise_cq(cq) in truth["normalised"]:
                real_hits += 1
            else:
                fakes.append(cq)
                fake_counter[cq] += 1

        stats["real_cqs_evaluated"] += real_hits
        stats["fake_cqs_recorded"] += len(fakes)
        if fakes:
            stats["files_with_contamination"] += 1
            if len(stats["examples"]) < 3:
                stats["examples"].append({
                    "scenario_id": sid,
                    "path": _rel(root, path),
                    "real_cqs_evaluated": real_hits,
                    "fake_cqs": fakes[:3],
                })
        if real_hits == 0 and recorded:
            stats["files_with_zero_real_cqs"] += 1

    # classify every (model, config)
    per_config_report: Dict[str, Any] = {}
    findings: List[str] = []
    failing = 0
    for (model, config), s in sorted(per_pair.items()):
        contaminated = s["fake_cqs_recorded"] > 0
        cq_free_by_design = config in CQ_FREE_CONFIGS
        if contaminated and s["real_cqs_evaluated"] == 0:
            verdict = "CRITICAL: zero real CQs evaluated, every recorded CQ is prompt boilerplate"
        elif contaminated:
            verdict = "CONTAMINATED: recorded CQ list mixes real CQs with prompt boilerplate"
        elif cq_free_by_design and s["recorded_cqs"] == 0:
            verdict = "OK (config supplies no CQs by design)"
        elif s["recorded_cqs"] == 0:
            verdict = "NO SIGNAL: no CQ list was recorded for any file"
        else:
            verdict = "OK"
        if contaminated:
            failing += 1
            findings.append(
                f"{model}/{config}: {s['files_with_contamination']}/{s['files']} files contain "
                f"boilerplate CQs; {s['real_cqs_evaluated']}/{s['real_cqs_available']} real CQs evaluated"
            )
        per_config_report[f"{model}/{config}"] = {**s, "verdict": verdict}

    # what the paper's functional table actually rests on
    paper_view: Dict[str, Any] = {}
    zero_real_paper_configs: List[str] = []
    for model, config in PAPER_FUNCTIONAL_CONFIGS:
        s = per_pair.get((model, config))
        label = f"{model}/{config}"
        if s is None:
            paper_view[label] = {"status": "MISSING", "note": "no eval_results records"}
            continue
        entry = {
            "real_cqs_evaluated": s["real_cqs_evaluated"],
            "real_cqs_available": s["real_cqs_available"],
            "fake_cqs_recorded": s["fake_cqs_recorded"],
            "files_with_contamination": s["files_with_contamination"],
            "files": s["files"],
        }
        if s["real_cqs_evaluated"] == 0:
            entry["status"] = "ZERO REAL CQs EVALUATED"
            zero_real_paper_configs.append(label)
        elif s["fake_cqs_recorded"]:
            entry["status"] = "PARTIALLY CONTAMINATED"
        else:
            entry["status"] = "CLEAN"
        paper_view[label] = entry

    if zero_real_paper_configs:
        findings.insert(0, (
            f"{len(zero_real_paper_configs)} of {len(PAPER_FUNCTIONAL_CONFIGS)} paper-reported "
            f"functional configurations evaluated ZERO real competency questions: "
            + ", ".join(zero_real_paper_configs)
        ))

    top_fakes = []
    for text, count in fake_counter.most_common(15):
        top_fakes.append({
            "text": text,
            "occurrences": count,
            "source": prompt_bullets.get(normalise_cq(text), "unmatched (not a prompt bullet)"),
        })

    if not per_pair:
        status, headline = "SKIP", "no eval_results/odp_eval records found"
    elif failing:
        total_fake = sum(s["fake_cqs_recorded"] for s in per_pair.values())
        total_real = sum(s["real_cqs_evaluated"] for s in per_pair.values())
        status = "FAIL"
        headline = (
            f"{failing} of {len(per_pair)} model/config pairs evaluated contaminated CQ lists "
            f"({total_fake} boilerplate CQs vs {total_real} real CQs across the corpus)"
        )
    else:
        status = "PASS"
        headline = "every recorded CQ matches data/scenarios/pattern_scenarios.json"

    return {
        "id": "cq-contamination",
        "title": "Competency-question contamination",
        "status": status,
        "headline": headline,
        "authoritative_source": "data/scenarios/pattern_scenarios.json (cq_list per scenario)",
        "scenarios_known": len(scenarios),
        "real_cqs_defined": sum(len(v["cq_list"]) for v in scenarios.values()),
        "per_config": per_config_report,
        "paper_functional_configs": paper_view,
        "top_boilerplate_cqs": top_fakes,
        "unknown_scenario_ids": dict(unknown_scenarios),
        "findings": findings,
        "root_cause": (
            "The recorded CQ lists came from a scraper that flipped into 'CQ section' mode "
            "at the first line matching /competency questions?/i.  In most prompts that is "
            "the opening instruction sentence, so it harvested the 'Modeling constraints' "
            "bullets that follow instead of the real competency questions.  Any CQ pass "
            "rate computed over those strings measures nothing about the ontology; the "
            "authoritative CQs are the cq_list entries in "
            "data/scenarios/pattern_scenarios.json."
        ),
    }


# --------------------------------------------------------------------------------
# Check 3 -- artifact disagreement
# --------------------------------------------------------------------------------

def _results_parse_flag(record: Dict[str, Any]) -> Tuple[Optional[bool], int]:
    return record.get("parse_success"), int(record.get("triple_count") or 0)


def _eval_parse_flag(record: Dict[str, Any]) -> Tuple[Optional[bool], int]:
    metrics = record.get("ontometrics") or {}
    if "error" in metrics:
        return False, 0
    if "triples_count" not in metrics:
        return None, 0
    triples = int(metrics.get("triples_count") or 0)
    return triples > 0, triples


def _summary_csv_report(root: str, results_index: Dict[Tuple[str, str, str], Dict[str, Any]]) -> Dict[str, Any]:
    path = os.path.join(root, "results", "summary.csv")
    if not os.path.isfile(path):
        return {"present": False}
    rows = 0
    mismatches: List[Dict[str, Any]] = []
    missing_json = 0
    with open(path, "r", encoding="utf-8", errors="replace", newline="") as fh:
        for row in csv.DictReader(fh):
            rows += 1
            key = (row.get("model", ""), row.get("config", ""), row.get("scenario_id", ""))
            per_file = results_index.get(key)
            if per_file is None:
                missing_json += 1
                continue
            csv_parse = str(row.get("parse_success", "")).strip().lower() == "true"
            json_parse = bool(per_file.get("parse_success"))
            if csv_parse != json_parse and len(mismatches) < 20:
                mismatches.append({"key": "/".join(key), "summary_csv": csv_parse, "results_json": json_parse})
    return {
        "present": True,
        "rows": rows,
        "rows_without_matching_results_json": missing_json,
        "parse_flag_mismatches_vs_results_json": len(mismatches),
        "examples": mismatches,
    }


def _third_artifact_set_report(root: str, eval_index: Dict[Tuple[str, str, str], Dict[str, Any]]) -> Dict[str, Any]:
    """eval/odp_eval_results.csv is a THIRD derived artifact set; report drift against eval_results/."""
    path = os.path.join(root, "eval", "odp_eval_results.csv")
    if not os.path.isfile(path):
        return {"present": False}
    rows = 0
    disagree: List[Dict[str, Any]] = []
    per_pair: Counter = Counter()
    with open(path, "r", encoding="utf-8", errors="replace", newline="") as fh:
        for row in csv.DictReader(fh):
            rows += 1
            key = (row.get("model", ""), row.get("config", ""), row.get("id", ""))
            record = eval_index.get(key)
            if record is None:
                continue
            csv_parseable = str(row.get("parseable", "")).strip().lower() == "true"
            json_parseable, _ = _eval_parse_flag(record)
            if json_parseable is not None and csv_parseable != json_parseable:
                per_pair[f"{key[0]}/{key[1]}"] += 1
                if len(disagree) < 10:
                    disagree.append({
                        "key": "/".join(key),
                        "eval_csv_parseable": csv_parseable,
                        "eval_results_json_parseable": json_parseable,
                    })
    return {
        "present": True,
        "path": "eval/odp_eval_results.csv",
        "rows": rows,
        "parse_flag_disagreements_vs_eval_results_json": sum(per_pair.values()),
        "per_config": dict(sorted(per_pair.items())),
        "examples": disagree,
        "note": (
            "eval/odp_eval_results.csv and eval/odp_eval_ranking.csv are a third derived set. "
            "Where they disagree with eval_results/odp_eval/*.json they predate the "
            "'extracted_from_raw' re-extraction, so the published ranking CSVs and the "
            "per-file JSONs cannot both be the source of the paper's numbers."
        ),
    }


def check_artifact_disagreement(root: str) -> Dict[str, Any]:
    results_index: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
    for model, config, sid, path in iter_result_records(root):
        try:
            results_index[(model, config, sid)] = _read_json(path)
        except Exception:
            results_index[(model, config, sid)] = {"_unreadable": True}

    eval_index: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
    for model, config, sid, path in iter_eval_records(root):
        try:
            eval_index[(model, config, sid)] = _read_json(path)
        except Exception:
            eval_index[(model, config, sid)] = {"_unreadable": True}

    per_pair: Dict[Tuple[str, str], Counter] = defaultdict(Counter)
    disagreements: List[Dict[str, Any]] = []

    for key in sorted(set(results_index) | set(eval_index)):
        model, config, sid = key
        bucket = per_pair[(model, config)]
        in_results = key in results_index
        in_eval = key in eval_index
        if in_results and not in_eval:
            bucket["only_in_results"] += 1
            continue
        if in_eval and not in_results:
            bucket["only_in_eval_results"] += 1
            continue

        bucket["in_both"] += 1
        r_parse, r_triples = _results_parse_flag(results_index[key])
        e_parse, e_triples = _eval_parse_flag(eval_index[key])
        extracted = bool(eval_index[key].get("extracted_from_raw"))

        parse_conflict = (r_parse is not None and e_parse is not None and bool(r_parse) != bool(e_parse))
        triple_conflict = r_triples != e_triples
        if parse_conflict:
            bucket["parse_success_disagreements"] += 1
        if triple_conflict:
            bucket["triple_count_disagreements"] += 1
        if extracted:
            bucket["eval_extracted_from_raw"] += 1
        if parse_conflict or triple_conflict:
            disagreements.append({
                "model": model,
                "config": config,
                "scenario_id": sid,
                "results_parse_success": r_parse,
                "eval_parse_success": e_parse,
                "results_triple_count": r_triples,
                "eval_triple_count": e_triples,
                "eval_extracted_from_raw": extracted,
            })

    totals: Counter = Counter()
    for bucket in per_pair.values():
        totals.update(bucket)

    findings: List[str] = []
    for (model, config), bucket in sorted(per_pair.items()):
        parts = []
        if bucket["parse_success_disagreements"]:
            parts.append(f"{bucket['parse_success_disagreements']} parse-flag conflicts")
        if bucket["triple_count_disagreements"]:
            parts.append(f"{bucket['triple_count_disagreements']} triple-count conflicts")
        if bucket["only_in_results"]:
            parts.append(f"{bucket['only_in_results']} records only in results/")
        if bucket["only_in_eval_results"]:
            parts.append(f"{bucket['only_in_eval_results']} records only in eval_results/")
        if parts:
            findings.append(f"{model}/{config}: " + ", ".join(parts))

    problem_count = (
        totals["parse_success_disagreements"]
        + totals["triple_count_disagreements"]
        + totals["only_in_results"]
        + totals["only_in_eval_results"]
    )

    if not results_index and not eval_index:
        status, headline = "SKIP", "neither results/ nor eval_results/ contains per-file records"
    elif problem_count:
        status = "FAIL"
        headline = (
            f"results/ and eval_results/ disagree on {totals['parse_success_disagreements']} parse "
            f"verdicts and {totals['triple_count_disagreements']} triple counts; "
            f"{totals['only_in_results'] + totals['only_in_eval_results']} records exist in only one set"
        )
    else:
        status, headline = "PASS", "results/ and eval_results/ agree on every shared record"

    return {
        "id": "artifact-disagreement",
        "title": "Disagreement between results/ and eval_results/",
        "status": status,
        "headline": headline,
        "totals": dict(totals),
        "per_config": {f"{m}/{c}": dict(v) for (m, c), v in sorted(per_pair.items())},
        "disagreements": disagreements,
        "summary_csv": _summary_csv_report(root, results_index),
        "additional_observations": {"third_artifact_set": _third_artifact_set_report(root, eval_index)},
        "findings": findings,
        "root_cause": (
            "results/ scores outputs/<...>/ontology.ttl as written by the broken extractor, "
            "while eval_results/ silently re-extracts the ontology from raw_response.txt and "
            "records extracted_from_raw=true.  No file in the repo declares which set is "
            "authoritative, so a reader cannot tell which one the paper's tables were built from."
        ),
    }


# --------------------------------------------------------------------------------
# Check 4 -- degenerate structural scores
# --------------------------------------------------------------------------------

def _domain_range_census(root: str) -> Dict[str, Dict[str, Any]]:
    """
    Count rdfs:domain / rdfs:range declarations per model.

    Counted twice: from ontology.ttl as stored, and again after recovering the
    ontology from raw_response.txt.  A model is only accused of declaring none
    when BOTH counts are zero -- otherwise the extraction bug (F1), not the model,
    would be taking the blame.
    """
    census: Dict[str, Dict[str, Any]] = {}
    for model, config, sid, path in iter_generation_outputs(root):
        entry = census.setdefault(model, {
            "files": 0,
            "domain_declarations": 0,
            "range_declarations": 0,
            "files_with_domain_or_range": 0,
            "domain_declarations_with_raw_recovery": 0,
            "range_declarations_with_raw_recovery": 0,
            "files_with_domain_or_range_with_raw_recovery": 0,
        })
        entry["files"] += 1

        graph, _ = parse_rdf(_read_text(path))
        stored_d = stored_r = 0
        if graph is not None:
            stored_d = sum(1 for _ in graph.triples((None, RDFS.domain, None)))
            stored_r = sum(1 for _ in graph.triples((None, RDFS.range, None)))
        entry["domain_declarations"] += stored_d
        entry["range_declarations"] += stored_r
        entry["files_with_domain_or_range"] += int(stored_d + stored_r > 0)

        best_d, best_r = stored_d, stored_r
        if graph is None or len(graph) == 0:
            raw_path = os.path.join(os.path.dirname(path), "raw_response.txt")
            if os.path.isfile(raw_path):
                rec, _mode = recover_turtle_from_raw(_read_text(raw_path))
                if rec is not None:
                    best_d = max(best_d, sum(1 for _ in rec.triples((None, RDFS.domain, None))))
                    best_r = max(best_r, sum(1 for _ in rec.triples((None, RDFS.range, None))))
        entry["domain_declarations_with_raw_recovery"] += best_d
        entry["range_declarations_with_raw_recovery"] += best_r
        entry["files_with_domain_or_range_with_raw_recovery"] += int(best_d + best_r > 0)
    return census


def _empty_graph_control() -> Dict[str, Any]:
    """
    Offline proof that the structural metric's optimum is an empty file.

    No network and no LLM: rdflib parses, owlrl reasons, and the OOPS component is
    taken from the metric's own definition (1 / (1 + pitfalls_total)) for a graph
    that declares nothing an OOPS! pitfall could attach to.
    """
    control: Dict[str, Any] = {
        "description": (
            "An empty graph and a header-only graph are both parseable and trivially "
            "consistent, so they saturate the consistency half of the structural score."
        ),
    }
    samples = {
        "empty_file": "",
        "header_only": (
            '@prefix owl:  <http://www.w3.org/2002/07/owl#> .\n'
            '@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .\n'
            '@prefix :     <http://example.org/odp#> .\n'
            ':Ontology a owl:Ontology ; rdfs:label "x" ; rdfs:comment "y" .\n'
        ),
    }
    for name, text in samples.items():
        graph, fmt = parse_rdf(text)
        entry: Dict[str, Any] = {
            "parses": graph is not None,
            "parse_format": fmt,
            "triples": len(graph) if graph is not None else None,
            "classes": 0,
            "properties": 0,
        }
        if graph is not None:
            try:
                import owlrl
                closure = Graph()
                for triple in graph:
                    closure.add(triple)
                owlrl.DeductiveClosure(
                    owlrl.OWLRL_Semantics,
                    rdfs_closure=True,
                    axiomatic_triples=False,
                    datatype_axioms=False,
                ).expand(closure)
                entry["consistent"] = True
                entry["consistency_component"] = 1.0
            except ImportError:
                entry["consistent"] = None
                entry["note"] = "owlrl not installed; consistency component not verified locally"
            except Exception as exc:  # an inconsistency would be a genuine surprise here
                entry["consistent"] = False
                entry["consistency_component"] = 0.0
                entry["reasoner_error"] = str(exc)[:200]
        if entry.get("consistency_component") == 1.0:
            entry["oops_component_if_zero_pitfalls"] = 1.0
            entry["structural_score_if_zero_pitfalls"] = 1.0
        control[name] = entry
    return control


def check_degenerate_scores(root: str, max_triples: int, min_score: float) -> Dict[str, Any]:
    flagged: List[Dict[str, Any]] = []
    zero_triple_consistent: List[Dict[str, Any]] = []
    scored = 0
    best_by_score: List[Tuple[float, int, str]] = []

    for model, config, sid, path in iter_eval_records(root):
        try:
            record = _read_json(path)
        except Exception:
            continue
        metrics = record.get("ontometrics") or {}
        if "error" in metrics or "triples_count" not in metrics:
            continue

        triples = int(metrics.get("triples_count") or 0)
        classes = int(metrics.get("classes_count") or 0)
        properties = (
            int(metrics.get("object_properties_count") or 0)
            + int(metrics.get("datatype_properties_count") or 0)
        )
        consistency, oops_component, score = structural_score(record)

        if consistency == 1.0 and triples == 0:
            zero_triple_consistent.append({
                "model": model, "config": config, "scenario_id": sid,
                "triples": 0, "consistency_component": 1.0,
                "oops_component": oops_component, "structural_score": score,
            })

        if score is None:
            continue
        scored += 1
        best_by_score.append((score, triples, f"{model}/{config}/{sid}"))

        near_empty = triples <= max_triples or classes == 0 or (classes + properties) == 0
        if near_empty and score >= min_score:
            flagged.append({
                "model": model, "config": config, "scenario_id": sid,
                "path": _rel(root, path),
                "triples": triples, "classes": classes, "properties": properties,
                "subclass_axioms": int(metrics.get("subclass_axioms") or 0),
                "restriction_axioms": int(metrics.get("restriction_axioms") or 0),
                "consistency_component": consistency,
                "oops_component": oops_component,
                "structural_score": round(score, 4),
            })

    flagged.sort(key=lambda f: (-f["structural_score"], f["triples"]))

    # the smallest graph in the corpus that attains a perfect structural score
    perfect = sorted((t for t in best_by_score if t[0] >= 1.0), key=lambda t: t[1])
    smallest_perfect = None
    if perfect:
        score, triples, label = perfect[0]
        smallest_perfect = {"key": label, "triples": triples, "structural_score": score}

    census = _domain_range_census(root)
    zero_dr_models: List[str] = []
    for model, entry in sorted(census.items()):
        stored = entry["domain_declarations"] + entry["range_declarations"]
        recovered = (
            entry["domain_declarations_with_raw_recovery"]
            + entry["range_declarations_with_raw_recovery"]
        )
        entry["declares_no_domain_or_range"] = (stored == 0 and recovered == 0)
        if entry["declares_no_domain_or_range"]:
            zero_dr_models.append(model)

    findings: List[str] = []
    if flagged:
        worst = flagged[0]
        findings.append(
            f"{len(flagged)} outputs score >= {min_score} structurally while holding "
            f"<= {max_triples} triples (or no classes/properties); worst offender "
            f"{worst['model']}/{worst['config']}/{worst['scenario_id']} scores "
            f"{worst['structural_score']} on {worst['triples']} triples, "
            f"{worst['classes']} classes, {worst['properties']} properties"
        )
    if zero_triple_consistent:
        findings.append(
            f"{len(zero_triple_consistent)} outputs hold ZERO triples yet are recorded "
            f"consistent=true, taking full marks on the consistency half of the metric"
        )
    for model in zero_dr_models:
        entry = census[model]
        findings.append(
            f"{model} declares ZERO rdfs:domain and ZERO rdfs:range across all "
            f"{entry['files']} outputs (also zero after recovering the ontology from "
            f"raw_response.txt) -- it cannot produce a well-formed ODP and must never "
            f"be selected as a winner"
        )
    if smallest_perfect and smallest_perfect["triples"] <= max_triples:
        findings.append(
            f"the metric's optimum is reachable by an almost-empty file: "
            f"{smallest_perfect['key']} attains the maximum structural score of 1.0 "
            f"on {smallest_perfect['triples']} triples"
        )

    if scored == 0 and not census:
        status, headline = "SKIP", "no scoreable eval_results records found"
    elif findings:
        status = "FAIL"
        headline = (
            f"{len(flagged)} near-empty outputs score >= {min_score}; "
            f"{len(zero_dr_models)} model(s) declare no rdfs:domain/rdfs:range at all"
        )
    else:
        status, headline = "PASS", "no degenerate structural scores detected"

    return {
        "id": "degenerate-scores",
        "title": "Degenerate structural scores and unusable models",
        "status": status,
        "headline": headline,
        "score_definition": (
            "structural_score = mean(consistency, 1/(1+oops_pitfalls_total)); this "
            "reproduces every avg_oops_score in eval/odp_eval_ranking.csv exactly."
        ),
        "thresholds": {"max_triples": max_triples, "min_structural_score": min_score},
        "records_scored": scored,
        "flagged_outputs": flagged,
        "zero_triple_but_consistent": zero_triple_consistent,
        "smallest_output_with_perfect_score": smallest_perfect,
        "domain_range_census": census,
        "models_declaring_no_domain_or_range": zero_dr_models,
        "empty_graph_control": _empty_graph_control(),
        "findings": findings,
        "root_cause": (
            "An empty or near-empty graph parses, is trivially consistent, and trips no "
            "OOPS! pitfalls, so it earns the maximum structural score.  The metric's "
            "optimum is therefore the empty file, and the score cannot distinguish a "
            "truncated stub from a complete pattern."
        ),
    }


# --------------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------------

_STATUS_MARK = {"PASS": "[ OK ]", "FAIL": "[FAIL]", "SKIP": "[SKIP]"}


def render_text_report(report: Dict[str, Any]) -> str:
    lines: List[str] = []
    add = lines.append
    add("=" * 78)
    add("ODPGen artifact integrity audit")
    add(f"root: {report['root']}")
    add(f"generated: {report['generated_at']}")
    add("=" * 78)

    for check in report["checks"]:
        add("")
        add(f"{_STATUS_MARK.get(check['status'], '[?]')} {check['id']} -- {check['title']}")
        add(f"       {check['headline']}")
        for finding in check.get("findings", []):
            add(f"       * {finding}")

        if check["id"] == "empty-outputs" and check.get("per_model"):
            add("")
            add("       per model (empty_or_unusable / total):")
            for model, counts in check["per_model"].items():
                bad, total = counts.get("empty_or_unusable", 0), counts.get("total", 0)
                extra = []
                for mode, label in (("recovery_closed_fence", "recoverable"),
                                    ("recovery_truncated", "truncated"),
                                    ("recovery_none", "unrecoverable")):
                    if counts.get(mode):
                        extra.append(f"{counts[mode]} {label}")
                suffix = f"   [{', '.join(extra)}]" if extra else ""
                add(f"         {model:<40} {bad:>3}/{total:<3}{suffix}")

        if check["id"] == "cq-contamination" and check.get("paper_functional_configs"):
            add("")
            add("       paper-reported functional configurations:")
            for label, entry in check["paper_functional_configs"].items():
                if "real_cqs_evaluated" in entry:
                    add(f"         {label:<52} {entry['status']} "
                        f"({entry['real_cqs_evaluated']}/{entry['real_cqs_available']} real CQs)")
                else:
                    add(f"         {label:<52} {entry['status']}")
            if check.get("top_boilerplate_cqs"):
                add("")
                add("       most frequent boilerplate 'CQs':")
                for item in check["top_boilerplate_cqs"][:5]:
                    text = item["text"].replace("\n", " ")
                    if len(text) > 76:
                        text = text[:73] + "..."
                    add(f"         {item['occurrences']:>4}x  {text}")
                    add(f"                 source: {item['source']}")

        if check["id"] == "artifact-disagreement":
            for label, counts in check.get("per_config", {}).items():
                notable = {k: v for k, v in counts.items() if k != "in_both" and v}
                if notable:
                    add(f"         {label:<52} {notable}")
            third = check.get("additional_observations", {}).get("third_artifact_set", {})
            if third.get("present") and third.get("parse_flag_disagreements_vs_eval_results_json"):
                add("")
                add(f"       note: a THIRD set ({third['path']}) disagrees with "
                    f"eval_results/ on {third['parse_flag_disagreements_vs_eval_results_json']} "
                    f"parse verdicts")

        if check["id"] == "degenerate-scores":
            if check.get("flagged_outputs"):
                add("")
                add("       near-empty outputs with high structural scores:")
                for item in check["flagged_outputs"][:8]:
                    add(f"         {item['model']}/{item['config']}/{item['scenario_id']:<14} "
                        f"score={item['structural_score']:<6} triples={item['triples']:<4} "
                        f"classes={item['classes']:<3} properties={item['properties']}")
            if check.get("domain_range_census"):
                add("")
                add("       rdfs:domain / rdfs:range census (stored | after raw recovery):")
                for model, entry in check["domain_range_census"].items():
                    flag = "  <-- NONE" if entry.get("declares_no_domain_or_range") else ""
                    add(f"         {model:<40} "
                        f"{entry['domain_declarations']:>4}/{entry['range_declarations']:<4} | "
                        f"{entry['domain_declarations_with_raw_recovery']:>4}/"
                        f"{entry['range_declarations_with_raw_recovery']:<4}"
                        f"  files_with_dr={entry['files_with_domain_or_range_with_raw_recovery']}"
                        f"/{entry['files']}{flag}")

    summary = report["summary"]
    add("")
    add("-" * 78)
    add(f"checks run: {summary['checks_run']}   passed: {summary['checks_passed']}   "
        f"failed: {summary['checks_failed']}   skipped: {summary['checks_skipped']}")
    if summary["failed_checks"]:
        add(f"FAILED: {', '.join(summary['failed_checks'])}")
    add(f"exit code: {summary['exit_code']}")
    add("-" * 78)
    return "\n".join(lines)


def run_audit(root: str, checks: Sequence[str], empty_bytes: int,
              max_triples: int, min_score: float) -> Dict[str, Any]:
    results: List[Dict[str, Any]] = []
    if "empty-outputs" in checks:
        results.append(check_empty_outputs(root, empty_bytes))
    if "cq-contamination" in checks:
        results.append(check_cq_contamination(root))
    if "artifact-disagreement" in checks:
        results.append(check_artifact_disagreement(root))
    if "degenerate-scores" in checks:
        results.append(check_degenerate_scores(root, max_triples, min_score))

    failed = [c["id"] for c in results if c["status"] == "FAIL"]
    passed = [c["id"] for c in results if c["status"] == "PASS"]
    skipped = [c["id"] for c in results if c["status"] == "SKIP"]

    return {
        "tool": "scripts/audit_artifacts.py",
        "tool_version": TOOL_VERSION,
        "generated_at": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "root": _posix(os.path.abspath(root)),
        "read_only": True,
        "thresholds": {
            "empty_bytes": empty_bytes,
            "degenerate_max_triples": max_triples,
            "degenerate_min_structural_score": min_score,
        },
        "checks": results,
        "summary": {
            "checks_run": len(results),
            "checks_passed": len(passed),
            "checks_failed": len(failed),
            "checks_skipped": len(skipped),
            "failed_checks": failed,
            "skipped_checks": skipped,
            "exit_code": 1 if failed else 0,
        },
    }


# --------------------------------------------------------------------------------
# Self-test (offline, synthetic corpus, no network and no LLM)
# --------------------------------------------------------------------------------

_GOOD_TTL = """@prefix owl:  <http://www.w3.org/2002/07/owl#> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
@prefix :     <http://example.org/odp#> .

:Ontology a owl:Ontology ; rdfs:label "T" ; rdfs:comment "T" .
:Event a owl:Class ; rdfs:label "Event" ; rdfs:comment "An event." .
:Place a owl:Class ; rdfs:label "Place" ; rdfs:comment "A place." .
:happensAt a owl:ObjectProperty ; rdfs:label "happens at" ; rdfs:comment "c" ;
    rdfs:domain :Event ; rdfs:range :Place .
"""

_STUB_TTL = """@prefix owl:  <http://www.w3.org/2002/07/owl#> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
@prefix :     <http://example.org/odp#> .

:Ontology a owl:Ontology ; rdfs:label "T" ; rdfs:comment "T" .
"""

_REAL_CQS = ["What events are responsible for the outcome?", "Where does the event happen?"]
_FAKE_CQS = ["Keep the ontology minimal and reusable.?", "Use clear, self-explanatory class and property names.?"]


def _write(path: str, text: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(text)


def _build_fixture(root: str, *, dirty: bool) -> None:
    """Build a tiny synthetic corpus with known-answer defects (or none)."""
    scenarios = [{
        "scenario_id": "S-01",
        "scenario_text": "A scenario.",
        "cq_list": list(_REAL_CQS),
    }]
    _write(os.path.join(root, "data", "scenarios", "pattern_scenarios.json"),
           json.dumps(scenarios, indent=2))
    _write(os.path.join(root, "prompts", "scenario_cq_constraints.txt"),
           "Generate an ODP from the scenario and competency questions below.\n"
           "- Keep the ontology minimal and reusable.\n"
           "- Use clear, self-explanatory class and property names.\n")

    # ---- outputs/
    _write(os.path.join(root, "outputs", "good-model", "scenario-cq", "S-01", "ontology.ttl"), _GOOD_TTL)
    _write(os.path.join(root, "outputs", "good-model", "scenario-cq", "S-01", "raw_response.txt"),
           "```turtle\n" + _GOOD_TTL + "```\n")
    if dirty:
        # F1 victim: the extractor captured the prompt's own placeholder
        _write(os.path.join(root, "outputs", "bad-model", "scenario-cq", "S-01", "ontology.ttl"), "...")
        _write(os.path.join(root, "outputs", "bad-model", "scenario-cq", "S-01", "raw_response.txt"),
               "Format: ```turtle ... ```\nHere it is:\n```turtle\n" + _GOOD_TTL + "```\n")
        # a model that declares no domain/range anywhere
        _write(os.path.join(root, "outputs", "nodr-model", "scenario-cq", "S-01", "ontology.ttl"), _STUB_TTL)
        _write(os.path.join(root, "outputs", "nodr-model", "scenario-cq", "S-01", "raw_response.txt"),
               "```turtle\n" + _STUB_TTL + "```\n")

    # ---- results/  and eval_results/
    def result_json(parse_ok: bool, triples: int) -> str:
        return json.dumps({"model": "m", "config": "c", "scenario_id": "S-01",
                           "parse_success": parse_ok, "triple_count": triples}, indent=2)

    def eval_json(triples: int, cqs: List[str], pitfalls: int, *, extracted: bool = False,
                  classes: int = 2, props: int = 1) -> str:
        payload: Dict[str, Any] = {
            "id": "S-01", "model": "m", "config": "c",
            "ontometrics": {
                "triples_count": triples, "classes_count": classes,
                "object_properties_count": props, "datatype_properties_count": 0,
                "subclass_axioms": 0, "restriction_axioms": 0,
            },
            "reasoner": {"consistent": True, "triples_before": triples,
                         "triples_after": triples, "inferred_triples": 0,
                         "unsatisfiable_classes": []},
            "oops": {"status_code": 200, "pitfalls_total": pitfalls, "pitfall_codes": []},
            "cqs": cqs, "cqs_count": len(cqs),
            "cq_verification": {"cqs_total": len(cqs), "cqs_passed": 0,
                                "cqs_failed": len(cqs), "pass_rate": 0.0, "results": []},
        }
        if extracted:
            payload["extracted_from_raw"] = True
        return json.dumps(payload, indent=2)

    _write(os.path.join(root, "results", "good-model", "scenario-cq", "S-01.json"), result_json(True, 12))
    _write(os.path.join(root, "eval_results", "odp_eval", "good-model", "scenario-cq", "S-01.json"),
           eval_json(12, list(_REAL_CQS), pitfalls=2))

    if dirty:
        # results/ says unparseable, eval_results/ says 12 triples via raw re-extraction
        _write(os.path.join(root, "results", "bad-model", "scenario-cq", "S-01.json"), result_json(False, 0))
        _write(os.path.join(root, "eval_results", "odp_eval", "bad-model", "scenario-cq", "S-01.json"),
               eval_json(12, list(_FAKE_CQS), pitfalls=2, extracted=True))
        # a 4-triple stub with zero pitfalls -> perfect structural score
        _write(os.path.join(root, "results", "nodr-model", "scenario-cq", "S-01.json"), result_json(True, 4))
        _write(os.path.join(root, "eval_results", "odp_eval", "nodr-model", "scenario-cq", "S-01.json"),
               eval_json(4, list(_REAL_CQS), pitfalls=0, classes=0, props=0))


def _self_test() -> int:
    import shutil
    import tempfile

    failures: List[str] = []
    checks_run = 0

    def _brief(value: Any, limit: int = 90) -> str:
        text = repr(value)
        return text if len(text) <= limit else text[: limit - 3] + "..."

    def expect(label: str, actual: Any, wanted: Any) -> None:
        nonlocal checks_run
        checks_run += 1
        if actual == wanted:
            print(f"  ok   {label}: {_brief(actual)}")
        else:
            print(f"  FAIL {label}: got {_brief(actual, 400)}, expected {_brief(wanted, 400)}")
            failures.append(label)

    def expect_true(label: str, actual: Any) -> None:
        expect(label, bool(actual), True)

    tmp = tempfile.mkdtemp(prefix="odpgen-audit-selftest-")
    try:
        # ---------------- unit-level helpers ----------------
        print("unit: helpers")
        expect("normalise_cq strips appended '?'",
               normalise_cq("Keep the ontology minimal and reusable.?"),
               normalise_cq("Keep the ontology minimal and reusable."))
        expect("normalise_cq strips backticks",
               normalise_cq("Declare `a owl:Class`."), "declare a owl:class")
        expect("normalise_cq keeps distinct CQs distinct",
               normalise_cq("What events?") == normalise_cq("Where does it happen?"), False)
        expect("parse_rdf rejects the F1 placeholder", parse_rdf("...")[0], None)
        expect("parse_rdf accepts a real ontology", parse_rdf(_GOOD_TTL)[0] is not None, True)
        expect("empty text parses to an empty graph", len(parse_rdf("")[0]), 0)

        g, mode = recover_turtle_from_raw("Format: ```turtle ... ```\n```turtle\n" + _GOOD_TTL + "```\n")
        expect("recovery skips the prompt placeholder and finds the real block", mode, "closed_fence")
        expect_true("recovered graph has triples", g is not None and len(g) > 0)
        truncated_raw = "```turtle\n" + _GOOD_TTL.rsplit("\n", 3)[0] + "\n:dangling"
        expect("unterminated fence is reported as truncated",
               recover_turtle_from_raw(truncated_raw)[1], "truncated")
        expect("nothing recoverable from prose", recover_turtle_from_raw("no code here")[1], "none")

        expect("structural score of a consistent, pitfall-free record",
               structural_score({"reasoner": {"consistent": True},
                                 "oops": {"status_code": 200, "pitfalls_total": 0}})[2], 1.0)
        expect("structural score with one pitfall",
               structural_score({"reasoner": {"consistent": True},
                                 "oops": {"status_code": 200, "pitfalls_total": 1}})[2], 0.75)
        expect("structural score is None when OOPS errored",
               structural_score({"reasoner": {"consistent": True}, "oops": {"error": "x"}})[2], None)

        # ---------------- dirty fixture: every check must FAIL ----------------
        print("integration: corpus with known defects")
        dirty = os.path.join(tmp, "dirty")
        _build_fixture(dirty, dirty=True)
        report = run_audit(dirty, CHECK_IDS, EMPTY_BYTES_DEFAULT,
                           DEGENERATE_TRIPLES_DEFAULT, DEGENERATE_SCORE_DEFAULT)
        by_id = {c["id"]: c for c in report["checks"]}

        empty = by_id["empty-outputs"]
        expect("empty-outputs status", empty["status"], "FAIL")
        expect("empty-outputs counts the placeholder file", empty["totals"]["empty_or_unusable"], 1)
        expect("empty-outputs attributes it to bad-model",
               empty["per_model"]["bad-model"]["empty_or_unusable"], 1)
        expect("empty-outputs leaves the good model alone",
               empty["per_model"]["good-model"]["empty_or_unusable"], 0)
        expect("empty-outputs records recoverability",
               empty["offenders"][0]["raw_response_recovery"], "closed_fence")

        cq = by_id["cq-contamination"]
        expect("cq-contamination status", cq["status"], "FAIL")
        expect("bad-model CQs are all boilerplate",
               cq["per_config"]["bad-model/scenario-cq"]["real_cqs_evaluated"], 0)
        expect("bad-model boilerplate CQ count",
               cq["per_config"]["bad-model/scenario-cq"]["fake_cqs_recorded"], 2)
        expect("good-model CQs are real",
               cq["per_config"]["good-model/scenario-cq"]["fake_cqs_recorded"], 0)
        expect("good-model real CQ count",
               cq["per_config"]["good-model/scenario-cq"]["real_cqs_evaluated"], 2)
        expect("boilerplate is traced back to the prompt file",
               cq["top_boilerplate_cqs"][0]["source"], "prompts/scenario_cq_constraints.txt")

        dis = by_id["artifact-disagreement"]
        expect("artifact-disagreement status", dis["status"], "FAIL")
        expect("one parse verdict conflict", dis["totals"]["parse_success_disagreements"], 1)
        expect("one triple-count conflict", dis["totals"]["triple_count_disagreements"], 1)
        expect("the conflict is flagged as re-extracted",
               dis["disagreements"][0]["eval_extracted_from_raw"], True)

        deg = by_id["degenerate-scores"]
        expect("degenerate-scores status", deg["status"], "FAIL")
        expect("the 4-triple stub is flagged", len(deg["flagged_outputs"]), 1)
        expect("the stub scored the maximum", deg["flagged_outputs"][0]["structural_score"], 1.0)
        expect("nodr-model is named", deg["models_declaring_no_domain_or_range"], ["nodr-model"])
        expect("good-model is not accused",
               deg["domain_range_census"]["good-model"]["declares_no_domain_or_range"], False)
        expect("bad-model is cleared by raw recovery (F1, not the model, emptied the file)",
               deg["domain_range_census"]["bad-model"]["declares_no_domain_or_range"], False)
        expect("empty-file control is consistent",
               deg["empty_graph_control"]["empty_file"].get("consistent"), True)
        expect("empty-file control would score the maximum",
               deg["empty_graph_control"]["empty_file"].get("structural_score_if_zero_pitfalls"), 1.0)

        expect("dirty corpus exits non-zero", report["summary"]["exit_code"], 1)
        expect("all four checks failed", sorted(report["summary"]["failed_checks"]), sorted(CHECK_IDS))
        expect_true("text report renders", len(render_text_report(report)) > 0)

        # ---------------- clean fixture: every check must PASS ----------------
        print("integration: corpus without defects")
        clean = os.path.join(tmp, "clean")
        _build_fixture(clean, dirty=False)
        clean_report = run_audit(clean, CHECK_IDS, EMPTY_BYTES_DEFAULT,
                                 DEGENERATE_TRIPLES_DEFAULT, DEGENERATE_SCORE_DEFAULT)
        clean_by_id = {c["id"]: c for c in clean_report["checks"]}
        for cid in CHECK_IDS:
            expect(f"clean corpus: {cid}", clean_by_id[cid]["status"], "PASS")
        expect("clean corpus exits zero", clean_report["summary"]["exit_code"], 0)

        # ---------------- read-only guarantee ----------------
        print("invariant: the auditor never writes to the corpus")

        def snapshot(base: str) -> Dict[str, Tuple[int, str]]:
            out: Dict[str, Tuple[int, str]] = {}
            for dirpath, _dirnames, filenames in os.walk(base):
                for name in filenames:
                    p = os.path.join(dirpath, name)
                    out[p] = (os.path.getsize(p), _read_text(p))
            return out

        before = snapshot(dirty)
        run_audit(dirty, CHECK_IDS, EMPTY_BYTES_DEFAULT,
                  DEGENERATE_TRIPLES_DEFAULT, DEGENERATE_SCORE_DEFAULT)
        expect("corpus is byte-identical after an audit run", snapshot(dirty), before)

        # ---------------- empty corpus degrades to SKIP, not a false pass ----------------
        print("invariant: an absent corpus skips rather than passes")
        bare = os.path.join(tmp, "bare")
        _write(os.path.join(bare, "data", "scenarios", "pattern_scenarios.json"), "[]")
        bare_report = run_audit(bare, CHECK_IDS, EMPTY_BYTES_DEFAULT,
                                DEGENERATE_TRIPLES_DEFAULT, DEGENERATE_SCORE_DEFAULT)
        expect("absent corpus yields no PASS verdicts",
               [c["status"] for c in bare_report["checks"]], ["SKIP"] * 4)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    print("")
    print(f"self-test: {checks_run - len(failures)}/{checks_run} assertions passed")
    if failures:
        print("failed assertions:")
        for name in failures:
            print(f"  - {name}")
        return 1
    return 0


# --------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    default_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    parser = argparse.ArgumentParser(
        prog="audit_artifacts.py",
        description="Read-only integrity auditor for the ODPGen artifact corpus.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--root", default=default_root,
                        help="repository root (default: the parent of scripts/)")
    parser.add_argument("--json-out", default=None,
                        help="path for the machine-readable report "
                             "(default: <root>/audit_report.json)")
    parser.add_argument("--check", action="append", choices=CHECK_IDS, dest="checks",
                        help="run only this check; repeatable (default: all)")
    parser.add_argument("--empty-bytes", type=int, default=EMPTY_BYTES_DEFAULT,
                        help=f"ontology.ttl smaller than this is empty (default: {EMPTY_BYTES_DEFAULT})")
    parser.add_argument("--degenerate-triples", type=int, default=DEGENERATE_TRIPLES_DEFAULT,
                        help=f"a graph with at most this many triples is near-empty "
                             f"(default: {DEGENERATE_TRIPLES_DEFAULT})")
    parser.add_argument("--degenerate-score", type=float, default=DEGENERATE_SCORE_DEFAULT,
                        help=f"structural score at or above which an output 'scores well' "
                             f"(default: {DEGENERATE_SCORE_DEFAULT})")
    parser.add_argument("--quiet", action="store_true",
                        help="write only the JSON report; suppress the stdout summary")
    parser.add_argument("--self-test", action="store_true",
                        help="run the built-in offline self-test and exit")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)

    if args.self_test:
        return _self_test()

    root = os.path.abspath(args.root)
    if not os.path.isdir(root):
        sys.stderr.write(f"audit_artifacts.py: no such directory: {root}\n")
        return 2

    checks = tuple(dict.fromkeys(args.checks)) if args.checks else CHECK_IDS

    try:
        report = run_audit(root, checks, args.empty_bytes,
                           args.degenerate_triples, args.degenerate_score)
    except FileNotFoundError as exc:
        sys.stderr.write(f"audit_artifacts.py: {exc}\n")
        return 2

    json_out = args.json_out or os.path.join(root, "audit_report.json")
    parent = os.path.dirname(os.path.abspath(json_out))
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(json_out, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(report, fh, indent=2, sort_keys=False)
        fh.write("\n")

    if not args.quiet:
        print(render_text_report(report))
        print(f"machine-readable report written to {_posix(os.path.abspath(json_out))}")

    return report["summary"]["exit_code"]


if __name__ == "__main__":
    raise SystemExit(main())
