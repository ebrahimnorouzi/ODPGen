#!/usr/bin/env python3
"""Regenerate eval/odp_eval_results.csv and eval/odp_eval_ranking.csv.

Defect V8: those two CSVs were committed as artifacts with **no producer**.  No
script in the repository read the per-file evaluation records and emitted them,
so the paper's structural table could not be regenerated from the evidence it
claims to summarise.  This script is that producer.

Input
-----
``--in DIR`` is walked recursively for ``*.json`` per-file evaluation records,
i.e. the objects ``batch_evaluate.evaluate_one`` produces (``id``, ``model``,
``config``, ``ontometrics``, ``reasoner``, ``cq_verification``, ``oops``).  Any
JSON that is not such a record is ignored, so ``--in eval_results`` and
``--in eval_local`` both work, as does pointing at a single model directory.

Output
------
``--out DIR`` receives ``odp_eval_results.csv`` (one row per evaluated file, the
same 35 columns as the committed artifact) and ``odp_eval_ranking.csv`` (the
per-model / per-config aggregate the paper's structural table is drawn from).

The OOPS! component
-------------------
The per-file OOPS! component is

    1 / (1 + important_count + 2 * critical_count)

Critical pitfalls weigh double; minor pitfalls do not count at all (OOPS!
reports a great many of them and they are recorded separately as
``minor_pitfalls_skipped``).  The naive ``1 / (1 + pitfalls_total)`` is
**wrong** and is retained only under ``--oops-formula total`` so that the two
can be measured against each other.  Measured against the committed
``eval/odp_eval_ranking.csv``: the weighted formula reproduces all 36 published
``avg_oops_score`` cells exactly; ``1/(1+pitfalls_total)`` misses 8 of them,
because 8 evaluated files carry a critical pitfall.

An OOPS! block has three states, not two, and the difference matters:

``scanned``      no ``skipped`` flag, no ``error``, ``status_code == 200``.
                 Believe it; score it with the chosen formula.  (This mirrors
                 ``batch_evaluate.compute_structural_score`` exactly.)
``failed``       the block carries an ``error``.  The file WAS submitted and
                 OOPS! could make nothing of it, which is an observation about
                 the file: it scores 0.0 and stays in the denominator.  The 151
                 unparseable rows of the committed corpus are of this kind and
                 the published ``avg_oops_score`` averages them in at 0.0.
``unavailable``  ``skipped`` is set, the block is absent, or the HTTP status is
                 not 200.  Nobody ever looked.  That is a MISSING component,
                 not a clean bill of health and not a failure: it is dropped
                 from the mean and the remaining weights renormalise.  Every
                 record ``scripts/eval_local.py`` writes is of this kind (OOPS!
                 is a web service and eval_local runs offline), which is why
                 ``--in eval_local`` used to abort with a ValueError.  A group
                 in which nothing was ever scanned reports a BLANK
                 ``avg_oops_score``, never 0.0.

The structural score  (defect V1 — the empty file at rank 1)
------------------------------------------------------------
Per file, in order of preference:

1. the record's OWN structural block — ``record["structural"]``
   (``batch_evaluate.compute_structural_score``) or ``record["scores"]``
   (``eval_local``).  The evaluator that produced the record is authoritative;
   the aggregator must not award a file more than its evaluator did.
2. otherwise, derived as the mean of the components that are actually
   available — the **gated** consistency credit and the OOPS! component —
   then floored by the content gate.

The gate is the point.  OWL-RL returns ``consistent: true`` for a graph with
zero triples and OOPS! returns zero pitfalls for it, because there is nothing
there to contradict anything and nothing there to be wrong.  Taking either
verdict at face value makes the empty file the metric's optimum: driving 70
zero-triple records through the previous version of this script produced
``structural_score`` 1.0 at rank 1.  So consistency credit comes from
``reasoner["consistency_credit"]`` when the record carries it and is otherwise
re-derived (0.0 when ``triples_before == 0``), and a file that is unparseable
or empty is floored at 0.0 outright.  The committed corpus contains five such
records — every "parseable" file bloomz produced — and they are the entire
source of that model's published structural score.

Group structural_score is the mean of the per-file scores over ALL files in the
group, unparseable ones included at 0.  Files are never dropped from the
denominator: an ontology that cannot be parsed is the metric's worst case, not
a missing observation.  (When every file in a group has both components, that
mean is arithmetically identical to the published
``(consistency_rate + avg_oops_score) / 2``, so every non-vacuous published cell
is reproduced unchanged.)

``consistency_rate`` and ``n_consistent`` keep reporting the LITERAL OWL-RL
verdict — that is what the committed artifact reports and what a reader checks
against the per-file CSV.  The gate removes score, it does not falsify a
verdict.  ``--include-diagnostics`` adds the gated rate and the vacuous count
as extra columns so the two can be seen side by side.

Usage
-----
    python scripts/aggregate_eval.py --in eval_results --out eval_regen
    python scripts/aggregate_eval.py --in eval_local   --out eval_local_regen
    python scripts/aggregate_eval.py --in eval_results --out eval_regen \\
        --compare eval          # report every cell that drifted
    python scripts/aggregate_eval.py --in eval_results --out /tmp/x \\
        --oops-formula total    # the wrong formula, for audit only
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

RESULTS_BASENAME = "odp_eval_results.csv"
RANKING_BASENAME = "odp_eval_ranking.csv"

ALL_CONFIG = "ALL (aggregate)"

OOPS_FORMULAS = ("weighted", "total")

ONTOMETRIC_FIELDS = (
    "triples_count",
    "classes_count",
    "object_properties_count",
    "datatype_properties_count",
    "annotation_properties_count",
    "individuals_count",
    "subclass_axioms",
    "equivalence_axioms",
    "disjoint_axioms",
    "restriction_axioms",
    "attribute_richness",
    "relationship_richness",
    "avg_subclass_per_class",
)

RESULT_FIELDS = (
    "id",
    "model",
    "config",
) + ONTOMETRIC_FIELDS + (
    "consistent",
    "triples_before",
    "triples_after",
    "inferred_triples",
    "unsatisfiable_classes_count",
    "cqs_total",
    "cqs_passed",
    "cqs_failed",
    "cq_pass_rate",
    "cq_skipped",
    "oops_scanned",
    "oops_pitfalls_total",
    "oops_important_count",
    "oops_critical_count",
    "oops_minor_skipped",
    "oops_pitfall_codes",
    "oops_error",
    "parseable",
    "consistency_score",
)

# Optional SEV columns, appended only with --include-sev.  Off by default so the
# default output stays byte-comparable with the committed artifact.
SEV_FIELDS = ("sev_score", "sev_coverage", "sev_connectivity",
              "cqs_authored", "cqs_derived")

RANKING_FIELDS = (
    "rank_overall",
    "rank_in_config",
    "model",
    "config",
    "n_total",
    "n_parseable",
    "n_consistent",
    "n_oops_scanned",
    "parse_rate",
    "consistency_rate",
    "avg_oops_score",
    "structural_score",
    "avg_cq_pass_rate",
)

# Optional ranking columns, appended only with --include-diagnostics.  Off by
# default so the header stays byte-comparable with the committed artifact.
RANKING_DIAGNOSTIC_FIELDS = (
    "n_vacuous",
    "n_oops_unavailable",
    "consistency_credit_rate",
)

ROUND = 4


# ── record loading ───────────────────────────────────────────────────────────

def is_record(obj: Any) -> bool:
    """True for a per-file evaluation record produced by batch_evaluate."""
    if not isinstance(obj, dict):
        return False
    if not all(isinstance(obj.get(k), str) for k in ("id", "model", "config")):
        return False
    # Must carry at least one evaluation section, otherwise it is some other
    # JSON that happens to have those three keys.
    return any(k in obj for k in
               ("ontometrics", "reasoner", "cq_verification", "oops", "error"))


def load_records(indir) -> List[Dict[str, Any]]:
    """Load every per-file evaluation record under ``indir`` (recursively).

    Raises ValueError on a duplicated (model, config, id) key: silently keeping
    one of two conflicting records would corrupt every aggregate downstream.
    """
    root = Path(indir)
    if not root.exists():
        raise FileNotFoundError(f"input directory does not exist: {root}")
    records: List[Dict[str, Any]] = []
    seen: Dict[tuple, Path] = {}
    for path in sorted(root.rglob("*.json")):
        try:
            obj = json.loads(path.read_text(encoding="utf-8"))
        except (ValueError, OSError):
            continue
        if not is_record(obj):
            continue
        key = (obj["model"], obj["config"], obj["id"])
        if key in seen:
            raise ValueError(
                f"duplicate evaluation record for {key}: "
                f"{seen[key]} and {path}")
        seen[key] = path
        obj["_source_path"] = str(path)
        records.append(obj)
    return records


# ── per-file derived quantities ──────────────────────────────────────────────

def _section(rec: Dict[str, Any], name: str) -> Dict[str, Any]:
    sec = rec.get(name)
    return sec if isinstance(sec, dict) else {}


def is_parseable(rec: Dict[str, Any]) -> bool:
    om = _section(rec, "ontometrics")
    return bool(om) and "error" not in om


def is_consistent(rec: Dict[str, Any]) -> bool:
    """The LITERAL reasoner verdict, for reporting.

    True for a zero-triple graph, because OWL-RL really does say so.  Never use
    this to award score — use :func:`consistency_credit`.
    """
    return _section(rec, "reasoner").get("consistent") is True


def is_vacuous(rec: Dict[str, Any]) -> bool:
    """True when the graph has nothing in it, so every verdict about it is free.

    Trusts ``reasoner["vacuous"]`` when batch_evaluate recorded it, and
    otherwise re-derives the fact from the triple counts, because none of the
    records in the committed eval_results/ corpus carry that field.
    """
    rz = _section(rec, "reasoner")
    if rz.get("vacuous") is True:
        return True
    if "error" not in rz:
        triples = rz.get("triples_before")
        if triples is not None and int(triples) == 0:
            return True
    om = _section(rec, "ontometrics")
    if om and "error" not in om:
        triples = om.get("triples_count")
        if triples is not None and int(triples) == 0:
            return True
    return False


def consistency_credit(rec: Dict[str, Any]) -> Optional[float]:
    """The consistency component the structural score is allowed to use.

    None when no reasoner verdict exists at all (unparseable, read error): that
    is a missing component, and the content gate floors the file anyway.
    0.0 for a vacuously consistent empty graph — this is the V1/F3 fix, and it
    is batch_evaluate's own ``consistency_credit`` whenever the record has one.
    """
    rz = _section(rec, "reasoner")
    if not rz or "error" in rz:
        return None
    if rz.get("consistency_credit") is not None:
        return float(rz["consistency_credit"])
    if "consistent" not in rz:
        return None
    if is_vacuous(rec):
        return 0.0
    return 1.0 if rz.get("consistent") is True else 0.0


# The three states of an OOPS! block.  See the module docstring.
OOPS_SCANNED = "scanned"
OOPS_FAILED = "failed"
OOPS_UNAVAILABLE = "unavailable"


def oops_state(rec: Dict[str, Any]) -> str:
    """scanned | failed | unavailable.

    Mirrors ``batch_evaluate.compute_structural_score``: a scan is believed only
    when it is not ``skipped``, carries no ``error``, and returned HTTP 200.
    """
    oops = _section(rec, "oops")
    if not oops:
        return OOPS_UNAVAILABLE
    if oops.get("skipped"):
        return OOPS_UNAVAILABLE
    if "error" in oops:
        return OOPS_FAILED
    if oops.get("status_code") != 200:
        return OOPS_UNAVAILABLE
    return OOPS_SCANNED


def is_oops_scanned(rec: Dict[str, Any]) -> bool:
    """True only for a believable scan; drives the ``oops_scanned`` column."""
    return oops_state(rec) == OOPS_SCANNED


def cq_pass_rate(rec: Dict[str, Any]) -> Optional[float]:
    cv = _section(rec, "cq_verification")
    if not cv or cv.get("skipped") or "error" in cv:
        return None
    rate = cv.get("pass_rate")
    return float(rate) if rate is not None else None


def oops_component(rec: Dict[str, Any], formula: str = "weighted") -> Optional[float]:
    """Per-file OOPS! component in [0, 1], or None when it is MISSING.

    weighted (default, correct):  1 / (1 + important + 2 * critical)
    total    (audit only, wrong): 1 / (1 + pitfalls_total)

    failed      (the scan was attempted and OOPS! could make nothing of the
                file) -> 0.0.  An observation about the file; stays in the mean.
    unavailable (skipped / no block / not HTTP 200: nobody ever looked)
                -> None.  Excluded from the mean, weights renormalised.  Never
                0.0, which would punish the file for the tool's absence, and
                never 1.0, which would award it a clean bill nobody issued.
    A file that WAS scanned but whose record lacks the severity counts the
    chosen formula needs raises ValueError rather than being silently scored.
    """
    if formula not in OOPS_FORMULAS:
        raise ValueError(f"unknown oops formula {formula!r}; "
                         f"expected one of {OOPS_FORMULAS}")
    oops = _section(rec, "oops")
    state = oops_state(rec)
    if state == OOPS_UNAVAILABLE:
        return None
    if state == OOPS_FAILED:
        return 0.0
    if formula == "total":
        total = oops.get("pitfalls_total")
        if total is None:
            raise ValueError(
                f"scanned OOPS record for {rec.get('model')}/{rec.get('config')}/"
                f"{rec.get('id')} has no 'pitfalls_total'")
        return 1.0 / (1.0 + int(total))
    important = oops.get("important_count")
    critical = oops.get("critical_count")
    if important is None or critical is None:
        raise ValueError(
            f"scanned OOPS record for {rec.get('model')}/{rec.get('config')}/"
            f"{rec.get('id')} has no important_count/critical_count; refusing to "
            f"guess a severity-weighted score (re-run the OOPS! scan, or use "
            f"--oops-formula total and say so in the paper)")
    return 1.0 / (1.0 + int(important) + 2 * int(critical))


# ── per-file structural score (the V1 fix) ───────────────────────────────────

def recorded_structural_score(rec: Dict[str, Any]) -> Optional[float]:
    """The record's OWN structural score, if the evaluator computed one.

    ``record["structural"]`` is batch_evaluate's block (already gated and
    capped); ``record["scores"]`` is eval_local's.  A null value there means the
    driver declined to score the file, so we fall back to deriving one.
    """
    for name in ("structural", "scores"):
        sec = _section(rec, name)
        if "structural_score" in sec:
            value = sec["structural_score"]
            if value is not None:
                return float(value)
    return None


def structural_cap(rec: Dict[str, Any]) -> float:
    """The most a file may score, given how much it actually says.

    0.0 for unparseable and for zero-triple graphs — a vacuously consistent,
    vacuously pitfall-free empty file is the metric's WORST case, not its best.
    Deliberately narrower than ``batch_evaluate.structural_content_gate``: the
    near-empty cap is not re-derived here, only read off a record that already
    carries it (see :func:`recorded_structural_score`), so that this script
    reproduces every published non-vacuous cell unchanged.
    """
    if not is_parseable(rec):
        return 0.0
    if is_vacuous(rec):
        return 0.0
    return 1.0


def structural_component(rec: Dict[str, Any],
                         oops_formula: str = "weighted") -> float:
    """Per-file structural score in [0, 1].  Never None: the floor is 0.0.

    The evaluator's own score wins when there is one.  Otherwise: the mean of
    the components actually available (gated consistency credit, OOPS!
    component), renormalised over however many that is, then floored by the
    content gate.  A file with no available component at all scores 0.0 and
    stays in the denominator.
    """
    own = recorded_structural_score(rec)
    if own is not None:
        return own
    parts = [c for c in (consistency_credit(rec),
                         oops_component(rec, oops_formula)) if c is not None]
    raw = (sum(parts) / len(parts)) if parts else 0.0
    return min(raw, structural_cap(rec))


# ── per-file rows ────────────────────────────────────────────────────────────

def build_results_rows(records: Iterable[Dict[str, Any]],
                       include_sev: bool = False) -> List[Dict[str, Any]]:
    rows = []
    for rec in records:
        om = _section(rec, "ontometrics")
        rz = _section(rec, "reasoner")
        cv = _section(rec, "cq_verification")
        oo = _section(rec, "oops")
        row: Dict[str, Any] = {
            "id": rec["id"],
            "model": rec["model"],
            "config": rec["config"],
        }
        for field in ONTOMETRIC_FIELDS:
            row[field] = om.get(field, "")
        row["consistent"] = rz.get("consistent", "")
        row["triples_before"] = rz.get("triples_before", "")
        row["triples_after"] = rz.get("triples_after", "")
        row["inferred_triples"] = rz.get("inferred_triples", "")
        row["unsatisfiable_classes_count"] = len(rz.get("unsatisfiable_classes") or [])
        row["cqs_total"] = cv.get("cqs_total", "")
        row["cqs_passed"] = cv.get("cqs_passed", "")
        row["cqs_failed"] = cv.get("cqs_failed", "")
        row["cq_pass_rate"] = cv.get("pass_rate", "")
        row["cq_skipped"] = bool(cv.get("skipped", False))
        row["oops_scanned"] = is_oops_scanned(rec)
        row["oops_pitfalls_total"] = oo.get("pitfalls_total", "")
        row["oops_important_count"] = oo.get("important_count", "")
        row["oops_critical_count"] = oo.get("critical_count", "")
        row["oops_minor_skipped"] = oo.get("minor_pitfalls_skipped", "")
        row["oops_pitfall_codes"] = "|".join(oo.get("pitfall_codes") or [])
        row["oops_error"] = oo.get("error", "")
        row["parseable"] = is_parseable(rec)
        row["consistency_score"] = 1 if is_consistent(rec) else 0
        if include_sev:
            row["sev_score"] = cv.get("sev_score", "")
            row["sev_coverage"] = cv.get("sev_coverage", "")
            row["sev_connectivity"] = cv.get("sev_connectivity", "")
            row["cqs_authored"] = cv.get("cqs_authored", "")
            row["cqs_derived"] = cv.get("cqs_derived", "")
        rows.append(row)
    rows.sort(key=lambda r: (r["model"], r["config"], r["id"]))
    return rows


# ── aggregate rows ───────────────────────────────────────────────────────────

def _blank_ranking_row() -> Dict[str, Any]:
    return {f: "" for f in RANKING_FIELDS + RANKING_DIAGNOSTIC_FIELDS}


def _summarise(records: Sequence[Dict[str, Any]], model: str, config: str,
               oops_formula: str) -> Dict[str, Any]:
    n = len(records)
    n_parseable = sum(1 for r in records if is_parseable(r))
    n_consistent = sum(1 for r in records if is_consistent(r))
    n_scanned = sum(1 for r in records if is_oops_scanned(r))
    n_vacuous = sum(1 for r in records if is_vacuous(r))
    n_unavailable = sum(1 for r in records
                        if oops_state(r) == OOPS_UNAVAILABLE)
    parse_rate = n_parseable / n if n else 0.0

    # Reported verdicts stay literal: this is what the committed artifact says
    # and what a reader checks against the per-file CSV.
    consistency_rate = n_consistent / n if n else 0.0

    # The OOPS! column averages the components that EXIST.  A group in which
    # nothing was ever scanned reports blank, not 0.0.
    components = [c for c in (oops_component(r, oops_formula) for r in records)
                  if c is not None]
    avg_oops = (sum(components) / len(components)) if components else None

    # The SCORE is gated per file, then averaged over the whole group.  When
    # every file has both components this is arithmetically identical to the
    # published (consistency_rate + avg_oops)/2; it differs exactly where a
    # verdict was vacuous or a component was missing.
    structural = (sum(structural_component(r, oops_formula) for r in records) / n) \
        if n else 0.0

    credits = [c for c in (consistency_credit(r) for r in records)
               if c is not None]
    credit_rate = (sum(credits) / n) if n else 0.0

    rates = [v for v in (cq_pass_rate(r) for r in records) if v is not None]
    avg_cq = (sum(rates) / len(rates)) if rates else None
    row = _blank_ranking_row()
    row.update({
        "model": model,
        "config": config,
        "n_total": n,
        "n_parseable": n_parseable,
        "n_consistent": n_consistent,
        "n_oops_scanned": n_scanned,
        "parse_rate": round(parse_rate, ROUND),
        "consistency_rate": round(consistency_rate, ROUND),
        "avg_oops_score": round(avg_oops, ROUND) if avg_oops is not None else "",
        "structural_score": round(structural, ROUND),
        "avg_cq_pass_rate": round(avg_cq, ROUND) if avg_cq is not None else "",
        "n_vacuous": n_vacuous,
        "n_oops_unavailable": n_unavailable,
        "consistency_credit_rate": round(credit_rate, ROUND),
    })
    # kept for callers/tests, not a CSV column
    row["_structural_unrounded"] = structural
    return row


def build_ranking_rows(records: Iterable[Dict[str, Any]],
                       oops_formula: str = "weighted") -> List[Dict[str, Any]]:
    """Per-model aggregate rows, then one block per config.

    Blocks are separated by an all-blank row, exactly as in the committed
    artifact.  Ranking is by structural_score descending, ties broken by model
    name ascending so the output is deterministic.
    """
    records = list(records)
    models = sorted({r["model"] for r in records})
    configs = sorted({r["config"] for r in records})

    rows: List[Dict[str, Any]] = []

    overall = [_summarise([r for r in records if r["model"] == m], m, ALL_CONFIG,
                          oops_formula)
               for m in models]
    overall.sort(key=lambda r: (-r["structural_score"], r["model"]))
    for i, row in enumerate(overall, start=1):
        row["rank_overall"] = i
    rows.extend(overall)
    rows.append(_blank_ranking_row())

    for config in configs:
        block = [_summarise([r for r in records
                             if r["model"] == m and r["config"] == config],
                            m, config, oops_formula)
                 for m in models
                 if any(r["model"] == m and r["config"] == config for r in records)]
        block.sort(key=lambda r: (-r["structural_score"], r["model"]))
        for i, row in enumerate(block, start=1):
            row["rank_in_config"] = i
        rows.extend(block)
        rows.append(_blank_ranking_row())

    return rows


# ── rendering ────────────────────────────────────────────────────────────────

def render_cell(value: Any) -> str:
    """Render one cell exactly as the committed CSVs render it."""
    if value is None or value == "":
        return ""
    if isinstance(value, bool):
        return "True" if value else "False"
    if isinstance(value, str):
        return value
    return str(value)


def _write_csv(path: Path, fields: Sequence[str],
               rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh, lineterminator="\n")
        writer.writerow(list(fields))
        for row in rows:
            writer.writerow([render_cell(row.get(f, "")) for f in fields])


# ── comparison against a committed artifact ──────────────────────────────────

def compare_results(rows: Sequence[Dict[str, Any]], committed_csv) -> List[Dict[str, Any]]:
    """Every per-file cell where the regenerated value differs from ``committed_csv``."""
    committed = list(csv.DictReader(Path(committed_csv).open(encoding="utf-8")))
    mine = {(r["model"], r["config"], r["id"]): r for r in rows}
    diffs: List[Dict[str, Any]] = []
    for crow in committed:
        key = (crow["model"], crow["config"], crow["id"])
        row = mine.get(key)
        if row is None:
            diffs.append({"model": key[0], "config": key[1], "id": key[2],
                          "field": "*", "committed": "<row present>",
                          "regenerated": "<no record>"})
            continue
        for field in crow:
            got = render_cell(row.get(field, ""))
            if got != crow[field]:
                diffs.append({"model": key[0], "config": key[1], "id": key[2],
                              "field": field, "committed": crow[field],
                              "regenerated": got})
    extra = set(mine) - {(c["model"], c["config"], c["id"]) for c in committed}
    for key in sorted(extra):
        diffs.append({"model": key[0], "config": key[1], "id": key[2],
                      "field": "*", "committed": "<no row>",
                      "regenerated": "<record present>"})
    return diffs


def compare_ranking(rows: Sequence[Dict[str, Any]], committed_csv) -> List[Dict[str, Any]]:
    """Every ranking cell where the regenerated value differs from ``committed_csv``."""
    committed = {(r["model"], r["config"]): r
                 for r in csv.DictReader(Path(committed_csv).open(encoding="utf-8"))
                 if r["model"]}
    diffs: List[Dict[str, Any]] = []
    seen = set()
    for row in rows:
        if not row.get("model"):
            continue
        key = (row["model"], row["config"])
        seen.add(key)
        crow = committed.get(key)
        if crow is None:
            diffs.append({"model": key[0], "config": key[1], "field": "*",
                          "committed": "<no row>", "regenerated": "<row present>"})
            continue
        for field in RANKING_FIELDS:
            got = render_cell(row.get(field, ""))
            if got != crow[field]:
                diffs.append({"model": key[0], "config": key[1], "field": field,
                              "committed": crow[field], "regenerated": got})
    for key in sorted(set(committed) - seen):
        diffs.append({"model": key[0], "config": key[1], "field": "*",
                      "committed": "<row present>", "regenerated": "<no row>"})
    return diffs


def _report(diffs: Sequence[Dict[str, Any]], label: str, stream=sys.stdout) -> None:
    if not diffs:
        print(f"[compare] {label}: regenerated output matches the committed file "
              f"cell for cell.", file=stream)
        return
    keys = {(d["model"], d.get("config", ""), d.get("id", "")) for d in diffs}
    models = sorted({d["model"] for d in diffs})
    print(f"[compare] {label}: {len(diffs)} differing cells across {len(keys)} rows "
          f"(models: {', '.join(models)})", file=stream)
    for d in diffs:
        loc = "/".join(x for x in (d["model"], d.get("config", ""), d.get("id", "")) if x)
        print(f"  {loc} :: {d['field']}: committed={d['committed']!r} "
              f"regenerated={d['regenerated']!r}", file=stream)


# ── CLI ──────────────────────────────────────────────────────────────────────

def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="aggregate_eval.py",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--in", dest="indir", required=True, metavar="DIR",
                        help="directory of per-file evaluation JSON records "
                             "(e.g. eval_results/ or eval_local/), walked recursively")
    parser.add_argument("--out", dest="outdir", required=True, metavar="DIR",
                        help=f"directory to write {RESULTS_BASENAME} and "
                             f"{RANKING_BASENAME} into")
    parser.add_argument("--oops-formula", choices=OOPS_FORMULAS, default="weighted",
                        help="'weighted' (default) = 1/(1+important+2*critical); "
                             "'total' = 1/(1+pitfalls_total), retained for audit "
                             "only and known to be wrong")
    parser.add_argument("--compare", metavar="DIR", default=None,
                        help="directory holding a committed odp_eval_results.csv / "
                             "odp_eval_ranking.csv; report every differing cell")
    parser.add_argument("--include-sev", action="store_true",
                        help="append the SEV columns to the per-file CSV "
                             "(off by default so output stays comparable to the "
                             "committed artifact)")
    parser.add_argument("--include-diagnostics", action="store_true",
                        help="append n_vacuous / n_oops_unavailable / "
                             "consistency_credit_rate to the ranking CSV, so the "
                             "gated score can be reconciled with the literal "
                             "verdicts it is computed from (off by default so "
                             "the header stays comparable to the committed "
                             "artifact)")
    parser.add_argument("--fail-on-diff", action="store_true",
                        help="with --compare, exit 1 if any cell differs")
    args = parser.parse_args(argv)

    records = load_records(args.indir)
    if not records:
        print(f"error: no evaluation records found under {args.indir}", file=sys.stderr)
        return 2

    result_fields = list(RESULT_FIELDS) + (list(SEV_FIELDS) if args.include_sev else [])
    results = build_results_rows(records, include_sev=args.include_sev)
    ranking = build_ranking_rows(records, oops_formula=args.oops_formula)

    ranking_fields = list(RANKING_FIELDS) + (
        list(RANKING_DIAGNOSTIC_FIELDS) if args.include_diagnostics else [])

    outdir = Path(args.outdir)
    _write_csv(outdir / RESULTS_BASENAME, result_fields, results)
    _write_csv(outdir / RANKING_BASENAME, ranking_fields, ranking)

    models = sorted({r["model"] for r in records})
    print(f"[aggregate_eval] {len(records)} records, {len(models)} models "
          f"-> {outdir / RESULTS_BASENAME}, {outdir / RANKING_BASENAME}")
    print(f"[aggregate_eval] oops formula: {args.oops_formula}")

    n_vacuous = sum(1 for r in records if is_vacuous(r))
    n_unavailable = sum(1 for r in records
                        if oops_state(r) == OOPS_UNAVAILABLE)
    if n_vacuous:
        print(f"[aggregate_eval] {n_vacuous} record(s) have a zero-triple graph: "
              f"their consistency verdict is vacuous and earns no structural "
              f"credit (V1)")
    if n_unavailable:
        print(f"[aggregate_eval] {n_unavailable} record(s) were never scanned by "
              f"OOPS!: that component is MISSING, excluded from the mean, and "
              f"the remaining weight renormalised")

    exit_code = 0
    if args.compare:
        cdir = Path(args.compare)
        rdiffs = compare_results(results, cdir / RESULTS_BASENAME)
        _report(rdiffs, RESULTS_BASENAME)
        kdiffs = compare_ranking(ranking, cdir / RANKING_BASENAME)
        _report(kdiffs, RANKING_BASENAME)
        if (rdiffs or kdiffs) and args.fail_on_diff:
            exit_code = 1
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
