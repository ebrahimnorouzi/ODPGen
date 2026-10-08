#!/usr/bin/env python3
"""
eval_local.py -- evaluate the LOCAL outputs/ tree, offline, with no GitHub.

Why this exists (defect V5)
---------------------------
batch_evaluate.py main() enumerates the corpus with list_ontology_paths(), which
calls https://api.github.com/repos/<REPO>/git/trees/<BRANCH>?recursive=1 and then
fetches every ontology over raw.githubusercontent.com.  Two consequences:

  1. A file that exists only in the local working tree -- a recovered output, a
     regenerated output, the result of ANY repair -- can never be scored.  The
     repair is unmeasurable by construction.
  2. Every downstream artifact is pinned to whatever is on the remote branch at
     the moment the script ran, not to the tree the reader has checked out.

This driver walks `outputs/**/ontology.ttl` on disk instead and scores each file
through the SAME functions batch_evaluate.py uses -- compute_ontometrics,
run_reasoner, run_cq_verification.  batch_evaluate is imported lazily and by
name, never copied, so fixes landing in that module are picked up automatically.

Guarantees
----------
* Offline.  No HTTP, no API key, no OOPS! web service.  A socket guard is armed
  for the duration of the run and any connect attempt is refused and counted.
* Nothing is skipped.  Empty, 3-byte, and unparseable ontologies are evaluated
  and land IN the denominator with a SEV score of 0.0.  They are the metric's
  worst case, not an absence of data.
* Unknown is never scored as perfect.  The OOPS! pitfall count is unavailable
  offline, so it is recorded as null and EXCLUDED from the structural score
  rather than read as "zero pitfalls" (that substitution is the F3 bug: it makes
  the empty file the metric's optimum).  For the same reason a graph with zero
  triples gets consistency = null, not 1.0: an empty graph is vacuously
  consistent, and vacuous is not an achievement.

It writes to eval_local/ by default and never touches results/, eval_results/,
eval/ or odp-platform-results/, which are evidence.

Usage
-----
    python scripts/eval_local.py                        # outputs/ -> eval_local/
    python scripts/eval_local.py --outputs outputs --out eval_local
    python scripts/eval_local.py --models gpt-5.4 --configs cq-only
    python scripts/eval_local.py --no-cq                # structure only, fast
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import socket
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_OUTPUTS_DIR = REPO_ROOT / "outputs"
DEFAULT_OUT_DIR = REPO_ROOT / "eval_local"

ONTOLOGY_FILENAME = "ontology.ttl"

#: How a missing measurement is written into CSV.  Deliberately NOT "0" and not
#: an empty cell: pandas reads "null" as NaN, and a human reads it as "we do not
#: know", which is the whole point.
NULL_TOKEN = "null"

#: Trees that hold published evidence.  Refusing to write into them is cheap
#: insurance against clobbering the artifacts this repair is being judged against.
PROTECTED_DIRS = {"results", "eval_results", "eval", "odp-platform-results",
                  "outputs", "data", "odp_eval", "eval_judge_results"}

DRIVER_VERSION = "eval_local/1.0"


# ── batch_evaluate, imported lazily and defensively ───────────────────────────

def load_batch_evaluate():
    """Import batch_evaluate from the working tree.

    Deliberately lazy: batch_evaluate.py is being edited concurrently, and this
    driver must (a) not fail at import time because of that, and (b) pick up
    whatever the module currently defines, so fixes to the scoring functions are
    inherited rather than duplicated here.
    """
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    try:
        import batch_evaluate  # noqa: WPS433 (intentional runtime import)
    except Exception as exc:  # pragma: no cover - only when that module is broken
        raise RuntimeError(
            f"could not import batch_evaluate from {REPO_ROOT}: {exc}. "
            "eval_local.py deliberately reuses that module's scoring functions "
            "instead of reimplementing them; fix the import there."
        ) from exc
    return batch_evaluate


def _require(be, name: str):
    fn = getattr(be, name, None)
    if fn is None:
        raise RuntimeError(
            f"batch_evaluate.{name} is missing; eval_local.py scores through "
            "batch_evaluate and will not silently substitute its own copy."
        )
    return fn


# ── offline guard ─────────────────────────────────────────────────────────────

class NetworkAccessRefused(RuntimeError):
    """Raised when something in the run tries to open a socket."""


class _OfflineGuard:
    """Refuse (and count) every outbound connection while armed."""

    def __init__(self) -> None:
        self.attempts: List[str] = []

    @property
    def n_attempts(self) -> int:
        return len(self.attempts)


@contextmanager
def offline_guard(enabled: bool = True):
    guard = _OfflineGuard()
    if not enabled:
        yield guard
        return

    real_connect = socket.socket.connect
    real_connect_ex = socket.socket.connect_ex
    real_create = socket.create_connection

    def refuse(where: str, target: Any) -> None:
        guard.attempts.append(f"{where}:{target!r}")
        raise NetworkAccessRefused(
            f"eval_local.py is offline by construction; refused {where} to "
            f"{target!r}. If a code path needs the network, it does not belong "
            "in the local evaluation driver."
        )

    def guarded_connect(self, address, *a, **k):   # noqa: ANN001
        refuse("socket.connect", address)

    def guarded_connect_ex(self, address, *a, **k):  # noqa: ANN001
        refuse("socket.connect_ex", address)

    def guarded_create(address, *a, **k):          # noqa: ANN001
        refuse("socket.create_connection", address)

    socket.socket.connect = guarded_connect          # type: ignore[assignment]
    socket.socket.connect_ex = guarded_connect_ex    # type: ignore[assignment]
    socket.create_connection = guarded_create        # type: ignore[assignment]
    try:
        yield guard
    finally:
        socket.socket.connect = real_connect          # type: ignore[assignment]
        socket.socket.connect_ex = real_connect_ex    # type: ignore[assignment]
        socket.create_connection = real_create        # type: ignore[assignment]


# ── the walker: local filesystem, never the GitHub API ────────────────────────

def iter_local_ontologies(outputs_root: Path | str,
                          models: Optional[Iterable[str]] = None,
                          configs: Optional[Iterable[str]] = None,
                          scenarios: Optional[Iterable[str]] = None,
                          ) -> List[Dict[str, Any]]:
    """Every outputs/<model>/<config>/<scenario_id>/ontology.ttl on disk.

    This is the whole point of the file: the corpus is what the working tree
    contains, not what a remote branch listing happens to report.
    """
    root = Path(outputs_root).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"outputs root not found: {root}")

    want_models = set(models) if models else None
    want_configs = set(configs) if configs else None
    want_scenarios = set(scenarios) if scenarios else None

    records: List[Dict[str, Any]] = []
    for path in root.rglob(ONTOLOGY_FILENAME):
        if not path.is_file():
            continue
        rel = path.relative_to(root)
        parts = rel.parts
        if len(parts) < 4:
            # Unexpected depth: keep it, do not drop it, and say so.  Silently
            # dropping files is how V5 hid a whole tree in the first place.
            model = parts[0] if len(parts) > 1 else "<unknown-model>"
            config = parts[1] if len(parts) > 2 else "<unknown-config>"
            scenario_id = path.parent.name
        else:
            model, config, scenario_id = parts[-4], parts[-3], parts[-2]

        if want_models and model not in want_models:
            continue
        if want_configs and config not in want_configs:
            continue
        if want_scenarios and scenario_id not in want_scenarios:
            continue

        records.append({
            "model": model,
            "config": config,
            "scenario_id": scenario_id,
            "path": str(path),
            "rel_path": rel.as_posix(),
        })

    records.sort(key=lambda r: (r["model"], r["config"], r["scenario_id"]))
    return records


# ── reading and classifying one file ──────────────────────────────────────────

def read_ontology(path: Path | str) -> Dict[str, Any]:
    """Bytes, text and sha256 of one ontology file; never raises."""
    p = Path(path)
    try:
        raw = p.read_bytes()
    except OSError as exc:
        return {"bytes": None, "sha256": None, "text": None,
                "read_error": str(exc)}
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        text = raw.decode("utf-8", errors="replace")
    return {"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest(),
            "text": text, "read_error": None}


def classify(be, text: Optional[str]):
    """(graph_or_None, format_or_None, status).

    status is one of: empty | unparseable | empty_graph | ok.  Every one of them
    is a RESULT, and every one of them is scored.  None is a skip.
    """
    if text is None:
        return None, None, "read_error"
    if not text.strip():
        return None, None, "empty"
    parse = _require(be, "_parse_graph")
    try:
        g, fmt = parse(text)
    except Exception:
        return None, None, "unparseable"
    if len(g) == 0:
        return g, fmt, "empty_graph"
    return g, fmt, "ok"


# ── scoring one file ──────────────────────────────────────────────────────────

def _offline_oops() -> Dict[str, Any]:
    """OOPS! is a web service.  Offline its verdict is unknown, not clean.

    pitfalls_total is null, NOT 0.  Recording "unknown" as "no pitfalls found"
    is the F3 bug: it awards a perfect 1/(1+0) OOPS component to files nobody
    ever scanned, including empty ones.
    """
    return {
        "skipped": True,
        "reason": "OOPS! is an online service; eval_local.py runs offline",
        "pitfalls_total": None,
        "important_count": None,
        "critical_count": None,
    }


def compute_scores(status: str,
                   reasoner: Dict[str, Any],
                   oops: Dict[str, Any],
                   cq_verification: Dict[str, Any]) -> Dict[str, Any]:
    """Per-file scores, with every unavailable component explicitly null.

    consistency
        1.0 / 0.0 for a graph that actually says something; null when the graph
        is empty, unparseable or unreadable.  An empty graph IS consistent --
        vacuously -- and crediting that vacuity with a perfect mark is precisely
        how the published structural metric came to have the empty file as its
        optimum (V1/F3).
    oops_component
        null offline.  Excluded from the mean; never defaulted to 1.0.
    structural_score
        the mean of the components that are actually available, or null when
        none are.  Null propagates into the aggregate as an excluded row with a
        stated denominator, so a reader can always see how much of the corpus a
        headline number is really about.
    """
    consistency: Optional[float]
    consistency_note: Optional[str]
    if status != "ok":
        consistency = None
        consistency_note = (
            f"status={status}: no non-empty graph to reason over; "
            "'vacuously consistent' is not evidence of consistency"
        )
    elif "error" in reasoner or "consistent" not in reasoner:
        consistency = None
        consistency_note = reasoner.get("error", "reasoner produced no verdict")
    else:
        consistency = 1.0 if reasoner.get("consistent") else 0.0
        consistency_note = None

    if oops.get("skipped") or "error" in oops or oops.get("pitfalls_total") is None:
        oops_component: Optional[float] = None
        oops_note: Optional[str] = oops.get("reason") or oops.get("error") \
            or "no pitfall count available"
    else:
        oops_component = 1.0 / (1.0 + float(oops["pitfalls_total"]))
        oops_note = None

    used: List[str] = []
    parts: List[float] = []
    if consistency is not None:
        used.append("consistency")
        parts.append(consistency)
    if oops_component is not None:
        used.append("oops")
        parts.append(oops_component)
    structural = round(sum(parts) / len(parts), 4) if parts else None

    sev = cq_verification.get("sev_score")
    if cq_verification.get("skipped"):
        sev_score: Optional[float] = None
    elif sev is None:
        # SEV ran but produced nothing -> the file is worthless, which is 0.0,
        # not "no data".  Only an explicit --no-cq yields null here.
        sev_score = 0.0
    else:
        sev_score = float(sev)

    return {
        "sev_score": sev_score,
        "consistency": consistency,
        "consistency_note": consistency_note,
        "oops_component": oops_component,
        "oops_note": oops_note,
        "structural_score": structural,
        "structural_components_used": used,
    }


def evaluate_local_file(record: Dict[str, Any], be=None,
                        run_cqs: bool = True) -> Dict[str, Any]:
    """Score one local ontology through batch_evaluate's own functions."""
    be = be or load_batch_evaluate()
    sid = record["scenario_id"]

    result: Dict[str, Any] = {
        "id": sid,
        "model": record["model"],
        "config": record["config"],
        "path": record["path"],
        "rel_path": record["rel_path"],
        "source": "local filesystem",
        "driver": DRIVER_VERSION,
    }

    blob = read_ontology(record["path"])
    result["bytes"] = blob["bytes"]
    result["sha256"] = blob["sha256"]
    if blob["read_error"]:
        result["read_error"] = blob["read_error"]

    text = blob["text"]
    graph, fmt, status = classify(be, text)
    result["status"] = status
    result["parse_format"] = fmt

    # 1. ontometrics -- batch_evaluate's own
    if status in ("ok", "empty_graph"):
        result["ontometrics"] = _require(be, "compute_ontometrics")(graph)
    else:
        result["ontometrics"] = {"error": f"no graph ({status})"}

    # 2. reasoner -- batch_evaluate's own.  Run even on the empty graph, because
    #    its output is the evidence for V1; it just does not earn a score.
    if status in ("ok", "empty_graph"):
        result["reasoner"] = _require(be, "run_reasoner")(graph)
        if status == "empty_graph":
            result["reasoner"]["vacuous"] = True
    else:
        result["reasoner"] = {"error": f"no graph ({status})"}

    # 3. CQ verification (SEV) -- offline and deterministic.  Runs for empty and
    #    unparseable files too: they score 0.0 and stay in the denominator.
    if not run_cqs:
        result["cq_verification"] = {"skipped": True, "reason": "--no-cq"}
    else:
        try:
            verify = _require(be, "run_cq_verification")
            result["cq_verification"] = verify(text if text is not None else "", sid)
        except KeyError as exc:
            result["cq_verification"] = {
                "skipped": True,
                "reason": f"scenario {sid} not in pattern_scenarios.json: {exc}",
            }
        except NetworkAccessRefused:
            raise
        except Exception as exc:
            # A contaminated CQ or a stale signature must be loud, not a 0.0.
            name = type(exc).__name__
            if name in ("CQContaminationError", "SignatureError"):
                raise
            result["cq_verification"] = {"error": f"SEV failed: {exc}"}

    # 4. OOPS -- unavailable offline; recorded as unknown, not as clean.
    result["oops"] = _offline_oops()

    result["scores"] = compute_scores(status, result["reasoner"],
                                      result["oops"], result["cq_verification"])
    return result


# ── CSV shaping ───────────────────────────────────────────────────────────────

SUMMARY_COLUMNS = [
    "model", "config", "id", "rel_path", "bytes", "sha256", "status",
    "parse_format", "triples_count", "classes_count", "object_properties_count",
    "datatype_properties_count", "subclass_axioms", "restriction_axioms",
    "consistent", "inferred_triples", "cqs_total", "cqs_passed", "sev_score",
    "sev_coverage", "sev_connectivity", "oops_pitfalls_total", "consistency_score",
    "oops_component", "structural_score",
]

AGGREGATE_COLUMNS = [
    "model", "config", "n_files", "n_ok", "n_empty", "n_empty_graph",
    "n_unparseable", "n_read_error", "mean_sev_score", "sev_denominator",
    "n_consistent", "consistency_denominator", "mean_structural_score",
    "structural_denominator", "oops_available",
]


def _cell(value: Any) -> Any:
    return NULL_TOKEN if value is None else value


def summary_row(rec: Dict[str, Any]) -> Dict[str, Any]:
    om = rec.get("ontometrics") or {}
    rsn = rec.get("reasoner") or {}
    cqv = rec.get("cq_verification") or {}
    sc = rec.get("scores") or {}
    ok = "error" not in om
    return {
        "model": rec["model"],
        "config": rec["config"],
        "id": rec["id"],
        "rel_path": rec["rel_path"],
        "bytes": _cell(rec.get("bytes")),
        "sha256": _cell(rec.get("sha256")),
        "status": rec.get("status"),
        "parse_format": _cell(rec.get("parse_format")),
        "triples_count": _cell(om.get("triples_count") if ok else None),
        "classes_count": _cell(om.get("classes_count") if ok else None),
        "object_properties_count": _cell(om.get("object_properties_count") if ok else None),
        "datatype_properties_count": _cell(om.get("datatype_properties_count") if ok else None),
        "subclass_axioms": _cell(om.get("subclass_axioms") if ok else None),
        "restriction_axioms": _cell(om.get("restriction_axioms") if ok else None),
        "consistent": _cell(rsn.get("consistent") if "error" not in rsn else None),
        "inferred_triples": _cell(rsn.get("inferred_triples") if "error" not in rsn else None),
        "cqs_total": _cell(cqv.get("cqs_total")),
        "cqs_passed": _cell(cqv.get("cqs_passed")),
        "sev_score": _cell(sc.get("sev_score")),
        "sev_coverage": _cell(cqv.get("sev_coverage")),
        "sev_connectivity": _cell(cqv.get("sev_connectivity")),
        "oops_pitfalls_total": _cell((rec.get("oops") or {}).get("pitfalls_total")),
        "consistency_score": _cell(sc.get("consistency")),
        "oops_component": _cell(sc.get("oops_component")),
        "structural_score": _cell(sc.get("structural_score")),
    }


def aggregate_rows(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """One row per (model, config).

    Every denominator is written out beside its mean.  n_files is the number of
    ontologies on disk; nothing is filtered out of it.  A mean whose denominator
    is smaller than n_files is telling you how much of the corpus it ignores.
    """
    groups: Dict[Any, List[Dict[str, Any]]] = {}
    for rec in records:
        groups.setdefault((rec["model"], rec["config"]), []).append(rec)

    rows: List[Dict[str, Any]] = []
    for (model, config) in sorted(groups):
        group = groups[(model, config)]
        statuses = [r.get("status") for r in group]
        sevs = [r["scores"]["sev_score"] for r in group
                if r["scores"]["sev_score"] is not None]
        cons = [r["scores"]["consistency"] for r in group
                if r["scores"]["consistency"] is not None]
        strs = [r["scores"]["structural_score"] for r in group
                if r["scores"]["structural_score"] is not None]
        rows.append({
            "model": model,
            "config": config,
            "n_files": len(group),
            "n_ok": statuses.count("ok"),
            "n_empty": statuses.count("empty"),
            "n_empty_graph": statuses.count("empty_graph"),
            "n_unparseable": statuses.count("unparseable"),
            "n_read_error": statuses.count("read_error"),
            "mean_sev_score": round(sum(sevs) / len(sevs), 4) if sevs else NULL_TOKEN,
            "sev_denominator": len(sevs),
            "n_consistent": int(sum(cons)),
            "consistency_denominator": len(cons),
            "mean_structural_score": round(sum(strs) / len(strs), 4) if strs else NULL_TOKEN,
            "structural_denominator": len(strs),
            "oops_available": False,
        })
    return rows


def _write_csv(path: Path, columns: List[str], rows: List[Dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _assert_safe_out_dir(out_dir: Path) -> None:
    resolved = out_dir.resolve()
    if resolved.parent == REPO_ROOT and resolved.name in PROTECTED_DIRS:
        raise ValueError(
            f"refusing to write into {resolved.name}/: that tree is published "
            "evidence. Use --out eval_local (the default) or another new path."
        )


# ── the run ───────────────────────────────────────────────────────────────────

def run(outputs_root: Path | str = DEFAULT_OUTPUTS_DIR,
        out_dir: Path | str = DEFAULT_OUT_DIR,
        run_cqs: bool = True,
        models: Optional[Iterable[str]] = None,
        configs: Optional[Iterable[str]] = None,
        scenarios: Optional[Iterable[str]] = None,
        limit: Optional[int] = None,
        network_guard: bool = True,
        progress: bool = False) -> Dict[str, Any]:
    """Walk the local tree, score everything, write JSON + CSV. Never networks."""
    out = Path(out_dir)
    _assert_safe_out_dir(out)
    out.mkdir(parents=True, exist_ok=True)

    be = load_batch_evaluate()
    found = iter_local_ontologies(outputs_root, models, configs, scenarios)
    selected = found[:limit] if limit else found

    records: List[Dict[str, Any]] = []
    failures: List[Dict[str, str]] = []

    with offline_guard(network_guard) as guard:
        for i, rec in enumerate(selected, 1):
            try:
                result = evaluate_local_file(rec, be, run_cqs=run_cqs)
            except NetworkAccessRefused as exc:
                # Loud on purpose: a network call inside the offline driver is a
                # bug in the scoring path, not a per-file data problem.
                raise
            except Exception as exc:            # pragma: no cover - defensive
                result = dict(rec)
                result.update({"id": rec["scenario_id"], "status": "driver_error",
                               "error": f"{type(exc).__name__}: {exc}",
                               "ontometrics": {"error": str(exc)},
                               "reasoner": {"error": str(exc)},
                               "cq_verification": {"error": str(exc)},
                               "oops": _offline_oops()})
                result["scores"] = compute_scores("driver_error",
                                                  result["reasoner"],
                                                  result["oops"],
                                                  result["cq_verification"])
                failures.append({"rel_path": rec["rel_path"], "error": str(exc)})

            target = out / rec["model"] / rec["config"]
            target.mkdir(parents=True, exist_ok=True)
            (target / f"{rec['scenario_id']}.json").write_text(
                json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
            records.append(result)

            if progress:
                sc = result.get("scores", {})
                print(f"[{i:>4}/{len(selected)}] {rec['rel_path']}  "
                      f"status={result.get('status')} "
                      f"sev={sc.get('sev_score')} "
                      f"struct={sc.get('structural_score')}", flush=True)

        network_attempts = guard.n_attempts

    _write_csv(out / "summary.csv", SUMMARY_COLUMNS,
               [summary_row(r) for r in records])
    agg = aggregate_rows(records)
    _write_csv(out / "aggregate.csv", AGGREGATE_COLUMNS, agg)

    status_counts: Dict[str, int] = {}
    for r in records:
        status_counts[r.get("status", "?")] = status_counts.get(r.get("status", "?"), 0) + 1

    all_sev = [r["scores"]["sev_score"] for r in records
               if r["scores"]["sev_score"] is not None]
    all_struct = [r["scores"]["structural_score"] for r in records
                  if r["scores"]["structural_score"] is not None]

    summary = {
        "driver": DRIVER_VERSION,
        "source": "local filesystem",
        "outputs_root": str(Path(outputs_root).resolve()),
        "out_dir": str(out.resolve()),
        "n_files": len(selected),
        "n_found": len(found),
        "n_evaluated": len(records),
        "n_skipped": len(selected) - len(records),
        "status_counts": status_counts,
        "cq_verification": "SEV" if run_cqs else "off (--no-cq)",
        "oops": "unavailable offline; recorded as null and excluded from the score",
        "corpus_sev_mean": round(sum(all_sev) / len(all_sev), 4) if all_sev else None,
        "corpus_sev_denominator": len(all_sev),
        "corpus_structural_mean": round(sum(all_struct) / len(all_struct), 4)
        if all_struct else None,
        "corpus_structural_denominator": len(all_struct),
        "network_calls_attempted": network_attempts,
        "driver_errors": failures,
        "aggregate": agg,
    }
    (out / "run_manifest.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


# ── CLI ───────────────────────────────────────────────────────────────────────

def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--outputs", default=str(DEFAULT_OUTPUTS_DIR),
                        help="local outputs tree to walk (default: outputs/)")
    parser.add_argument("--out", default=str(DEFAULT_OUT_DIR),
                        help="where to write results (default: eval_local/)")
    parser.add_argument("--no-cq", action="store_true",
                        help="skip SEV CQ verification (structure only, faster)")
    parser.add_argument("--models", nargs="*", default=None,
                        help="restrict to these model directories")
    parser.add_argument("--configs", nargs="*", default=None,
                        help="restrict to these config directories")
    parser.add_argument("--scenarios", nargs="*", default=None,
                        help="restrict to these scenario ids")
    parser.add_argument("--limit", type=int, default=None,
                        help="evaluate at most N files (smoke tests)")
    parser.add_argument("--no-network-guard", action="store_true",
                        help="do not arm the socket guard (diagnostics only)")
    parser.add_argument("--quiet", action="store_true",
                        help="do not print per-file progress")
    args = parser.parse_args(argv)

    summary = run(outputs_root=args.outputs, out_dir=args.out,
                  run_cqs=not args.no_cq, models=args.models,
                  configs=args.configs, scenarios=args.scenarios,
                  limit=args.limit, network_guard=not args.no_network_guard,
                  progress=not args.quiet)

    counts = ", ".join(f"{k}={v}" for k, v in sorted(summary["status_counts"].items()))
    print(f"\n{summary['n_evaluated']} local ontologies evaluated "
          f"({counts or 'none'}); 0 skipped.", flush=True)
    print(f"SEV mean over {summary['corpus_sev_denominator']} of "
          f"{summary['n_files']} files: {summary['corpus_sev_mean']}", flush=True)
    print(f"Structural mean over {summary['corpus_structural_denominator']} of "
          f"{summary['n_files']} files: {summary['corpus_structural_mean']} "
          "(OOPS! excluded: unavailable offline, recorded as null not zero)",
          flush=True)
    print(f"Network calls attempted: {summary['network_calls_attempted']}",
          flush=True)
    print(f"Results in: {summary['out_dir']}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
