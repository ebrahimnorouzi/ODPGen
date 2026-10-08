#!/usr/bin/env python3
"""
Batch ontology evaluator for ODPGen outputs.
Runs four evaluations per ontology:
  1. Ontometrics    (structural counts + richness ratios)
  2. OWL Reasoner   (OWL-RL via owlrl: consistency + inferred triples)
  3. CQ Verification (SEV — Schema-Entailment Verification; offline, rdflib only)
  4. OOPS! pitfalls (optional, needs --oops-url)

CQ verification changed in engine version 1.0.0 (see the SEV section below):

  * Competency questions now come ONLY from
    data/scenarios/pattern_scenarios.json ("cq_list"), keyed by scenario_id.
    They used to be scraped out of outputs/*/*/*/prompt.txt, which pulled prompt
    boilerplate ("Use clear, self-explanatory class and property names?") into
    the CQ set and, for three of the four configurations reported in the paper,
    meant ZERO real CQs were evaluated.  A CQ that is not in cq_list now raises
    CQContaminationError instead of being scored.

  * A CQ is no longer "passed" because an LLM-written instance-level SPARQL
    SELECT returned a row.  The generation prompts say "Do NOT create individual
    instances; output schema axioms only", so every generated ODP is a T-Box and
    an instance-level query returns nothing however good the ontology is.  SEV
    lifts the T-Box into a schema-level graph and ASKs whether the vocabulary
    each CQ needs exists and is connected there, awarding graded credit.

  * There is no LLM and no network call anywhere in CQ verification, so no API
    key is needed and results are deterministic and reproducible.

WHERE THE CORPUS COMES FROM (--local-root):

  The DOCUMENTED workflow evaluates the LOCAL working tree:

      python3 batch_evaluate.py --local-root outputs --out odp_eval

  which walks outputs/**/ontology.ttl from disk, scores every file, and writes
  odp_eval/summary.csv (per artefact) and odp_eval/aggregate.csv (per
  model/config) alongside the per-ID JSON.  It makes no network call at all: a
  guard is armed for the run and the refused-call count is printed.

  Without --local-root the corpus is still enumerated over the GitHub API and
  fetched from raw.githubusercontent.com — that path is kept working, but it can
  only ever see what is on the remote branch, so a recovered or regenerated
  output in the working tree is invisible to it.

TRUNCATION:

  scripts/run_generation.py flags a generation that ran out of output budget
  (metadata.json "truncated"/"truncation_signals", plus a "# ODPGEN-TRUNCATED"
  banner on the ontology).  Every record produced here carries that verdict as
  `truncated` (True / False / None-for-unknown) and a `truncation` block naming
  the evidence, and every aggregate reports n_truncated / n_complete /
  n_truncation_unknown beside both a whole-corpus mean and a complete-only mean.
  Truncated artefacts are NEVER removed from a denominator.

Output: odp_eval/{model}/{config}/{id}.json  (one file per ID)
        odp_eval/summary.csv, odp_eval/aggregate.csv, odp_eval/run_manifest.json
Usage:
  python3 batch_evaluate.py --local-root outputs      # LOCAL tree, offline
  python3 batch_evaluate.py                           # remote (GitHub) corpus
  python3 batch_evaluate.py --no-cq                   # skip CQ verification
  python3 batch_evaluate.py --gold-calibration        # run the SEV validity gate
  python3 batch_evaluate.py --rerun-failed            # redo only error/parse-failed IDs
"""

import argparse
import csv
import hashlib
import importlib.util
import json
import posixpath
import re
import socket
import sys
import threading
import time
import urllib.parse
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import requests
from rdflib import BNode, Graph, Literal, URIRef
from rdflib.namespace import OWL, RDF, RDFS, SKOS, Namespace

# ── Config ────────────────────────────────────────────────────────────────────
GITHUB_TOKEN = ""
REPO         = "ebrahimnorouzi/ODPGen"
BRANCH       = "main"
OUT_DIR      = Path("odp_eval")
MAX_WORKERS  = 8    # parallel HTTP + reasoner tasks

GH_HEADERS = {
    "Authorization": f"token {GITHUB_TOKEN}",
    "Accept": "application/vnd.github+json",
}

# ── GitHub helpers ─────────────────────────────────────────────────────────────

def raw_url(path: str) -> str:
    return f"https://raw.githubusercontent.com/{REPO}/{BRANCH}/{path}"


def fetch_text(path: str, retries: int = 3) -> Optional[str]:
    for attempt in range(retries):
        try:
            r = requests.get(raw_url(path),
                             headers={"Authorization": f"token {GITHUB_TOKEN}"},
                             timeout=30)
            if r.status_code == 200:
                return r.text
            if r.status_code == 404:
                return None
        except requests.RequestException:
            time.sleep(2 ** attempt)
    return None


def list_ontology_paths() -> List[str]:
    """Enumerate the corpus over the GitHub API (the REMOTE path).

    Kept working on purpose, but it is no longer the documented workflow: see
    list_local_ontology_paths and `--local-root` below for why.
    """
    url = f"https://api.github.com/repos/{REPO}/git/trees/{BRANCH}?recursive=1"
    r = requests.get(url, headers=GH_HEADERS, timeout=30)
    r.raise_for_status()
    return [t["path"] for t in r.json().get("tree", [])
            if t["path"].endswith("/ontology.ttl")]


# ── D1: the LOCAL corpus — the working tree, not a remote branch ──────────────
#
# list_ontology_paths() above enumerates ontologies over
# https://api.github.com/repos/<repo>/git/trees/<branch> and every file is then
# fetched over raw.githubusercontent.com.  Two consequences, and they are the
# whole reason this section exists:
#
#   1. A file that exists only in the working tree — a recovered output, a
#      regenerated output, the result of ANY repair — can never be scored.  The
#      repair is unmeasurable by construction.
#   2. Every artefact the paper derives is pinned to whatever is on the remote
#      branch at the moment the script ran, not to the tree the reader has
#      checked out.
#
# `--local-root PATH` walks PATH/**/ontology.ttl from disk instead.  It scores
# through exactly the same functions, so the two paths cannot drift apart, and
# it is the path the README and run_all_experiments.sh document.

ONTOLOGY_FILENAME = "ontology.ttl"


def list_local_ontology_paths(local_root: Path | str) -> List[str]:
    """Every <local_root>/**/ontology.ttl on disk, sorted, as filesystem paths.

    The corpus is what the working tree contains.  Nothing is filtered out and
    nothing is silently dropped: a file at an unexpected depth is still returned
    and still scored, because silently dropping files is how a whole tree became
    invisible in the first place.
    """
    root = Path(local_root)
    if not root.is_dir():
        raise FileNotFoundError(f"--local-root is not a directory: {root}")
    found = [p for p in root.rglob(ONTOLOGY_FILENAME) if p.is_file()]
    return sorted(str(p) for p in found)


def split_artifact_path(onto_path: str) -> Tuple[str, str, str]:
    """(model, config, scenario_id) from …/<model>/<config>/<id>/ontology.ttl.

    Reads the TAIL of the path, so it gives the same answer for the remote
    "outputs/m/c/i/ontology.ttl" as for an absolute local path.
    """
    parts = [p for p in str(onto_path).replace("\\", "/").split("/")
             if p and p != "."]
    while len(parts) < 4:
        parts.insert(0, "<unknown>")
    return parts[-4], parts[-3], parts[-2]


def read_local_text(path: Path | str) -> Optional[str]:
    """Text of a local file, or None when it is not there. Never raises."""
    p = Path(path)
    try:
        return p.read_text(encoding="utf-8")
    except FileNotFoundError:
        return None
    except UnicodeDecodeError:
        try:
            return p.read_bytes().decode("utf-8", errors="replace")
        except OSError:
            return None
    except OSError:
        return None


class NetworkAccessDuringLocalRun(RuntimeError):
    """Something tried to open a socket during a --local-root run."""


class _NetworkGuard:
    """Counts what it refused, and what it let through, while armed."""

    def __init__(self, allowlist: Optional[List[str]] = None) -> None:
        #: connections nobody asked for — these are refused.
        self.attempts: List[str] = []
        #: connections to an endpoint this run was explicitly given.
        self.allowed: List[str] = []
        self.allowlist: List[str] = list(allowlist or [])

    @property
    def n_attempts(self) -> int:
        return len(self.attempts)

    @property
    def n_allowed(self) -> int:
        return len(self.allowed)


def _endpoint_allowlist(urls: Optional[List[str]]) -> set:
    """(host, port) pairs reachable for an explicitly requested service URL.

    Every address the hostname resolves to is included, because urllib3 calls
    socket.connect() with the RESOLVED address, not the hostname.  Resolution
    happens here, before the guard is armed.
    """
    allowed: set = set()
    for url in urls or []:
        if not url:
            continue
        try:
            parsed = urllib.parse.urlsplit(url if "//" in url else "//" + url)
            host = parsed.hostname
            port = parsed.port or (443 if parsed.scheme == "https" else 80)
        except ValueError:
            continue
        if not host:
            continue
        allowed.add((host.lower(), int(port)))
        try:
            for info in socket.getaddrinfo(host, port, type=socket.SOCK_STREAM):
                sa = info[4]
                if sa:
                    allowed.add((str(sa[0]).lower(), int(sa[1])))
        except OSError:
            pass
    return allowed


@contextmanager
def network_guard(enabled: bool = True, allow: Optional[List[str]] = None):
    """Refuse every UNREQUESTED socket connection for a local run.

    The point is that "0 network calls" is MEASURED rather than asserted: the
    guard counts what it refused and main() reports the number.

    `allow` lists service URLs this run was explicitly given on the command
    line — today that is only --oops-url.  Blanket refusal used to include
    those, and because NetworkAccessDuringLocalRun is not a
    requests.RequestException it tore straight out of run_oops_scan and
    evaluate_one: `--local-root --oops-url` turned every artefact in the corpus
    into a bare error record while still exiting 0.  Refusing what the user
    asked for is not offline-by-construction, it is data loss.  The invariant
    that carries the weight is unchanged and still measured: nothing the run
    was NOT given is dialled, and both counts are reported separately.
    """
    allowlist = _endpoint_allowlist(allow)
    guard = _NetworkGuard([u for u in (allow or []) if u])
    if not enabled:
        yield guard
        return

    real_connect = socket.socket.connect
    real_connect_ex = socket.socket.connect_ex
    real_create = socket.create_connection
    permit = threading.local()

    def _key(address) -> Optional[Tuple[str, int]]:     # noqa: ANN001
        if isinstance(address, (tuple, list)) and len(address) >= 2:
            try:
                return (str(address[0]).lower(), int(address[1]))
            except (TypeError, ValueError):
                return None
        return None

    def permitted(address) -> bool:                     # noqa: ANN001
        # Inside an already-permitted create_connection, the inner connect() is
        # the same dial and must not be second-guessed.
        if getattr(permit, "depth", 0) > 0:
            return True
        return _key(address) in allowlist

    def refuse(where: str, target: Any):
        guard.attempts.append(f"{where}:{target!r}")
        raise NetworkAccessDuringLocalRun(
            f"--local-root is offline by construction; refused {where} to "
            f"{target!r}. The local corpus is read from disk."
        )

    def guarded_connect(self, address, *a, **k):        # noqa: ANN001
        if permitted(address):
            guard.allowed.append(f"socket.connect:{address!r}")
            return real_connect(self, address, *a, **k)
        refuse("socket.connect", address)

    def guarded_connect_ex(self, address, *a, **k):     # noqa: ANN001
        if permitted(address):
            guard.allowed.append(f"socket.connect_ex:{address!r}")
            return real_connect_ex(self, address, *a, **k)
        refuse("socket.connect_ex", address)

    def guarded_create(address, *a, **k):               # noqa: ANN001
        if permitted(address):
            guard.allowed.append(f"socket.create_connection:{address!r}")
            permit.depth = getattr(permit, "depth", 0) + 1
            try:
                return real_create(address, *a, **k)
            finally:
                permit.depth = getattr(permit, "depth", 1) - 1
        refuse("socket.create_connection", address)

    socket.socket.connect = guarded_connect             # type: ignore[assignment]
    socket.socket.connect_ex = guarded_connect_ex       # type: ignore[assignment]
    socket.create_connection = guarded_create           # type: ignore[assignment]
    try:
        yield guard
    finally:
        socket.socket.connect = real_connect            # type: ignore[assignment]
        socket.socket.connect_ex = real_connect_ex      # type: ignore[assignment]
        socket.create_connection = real_create          # type: ignore[assignment]


# ── D3: truncation is DETECTED at generation time — now it is CONSUMED ────────
#
# scripts/run_generation.py already decides whether a generation was cut off
# mid-answer and records the verdict three ways:
#
#   * metadata.json  "truncated": bool and "truncation_signals": [...]
#   * a "# ODPGEN-TRUNCATED" banner prepended to ontology.ttl (a Turtle comment,
#     so the file still parses — the banner is the copy that travels with the
#     artefact)
#   * the raw response itself, whose unterminated code fence is the tell that
#     survives even when neither of the other two was written
#
# Nothing downstream read any of it, so a generation that ran out of output
# budget was scored as though the model had chosen to stop there.  160 of the
# 420 recorded responses carry a signal.  Every one of the 420 metadata.json
# files on disk predates the flag, which is why the raw-response fallback below
# is not optional: without it the fix would be invisible on the actual corpus.
#
# The verdict is carried into every per-artefact record and every aggregate.
# Truncated artefacts are NOT dropped from any denominator — that would hide
# exactly the failure this exists to show.  Each aggregate reports the whole
# corpus AND a complete-only mean, each with its own stated denominator.

TRUNCATION_BANNER_PREFIX = "# ODPGEN-TRUNCATED"
TRUNCATION_SIGNALS_PREFIX = "# ODPGEN-TRUNCATION-SIGNALS:"
METADATA_FILENAME = "metadata.json"
RAW_RESPONSE_FILENAME = "raw_response.txt"

#: How many leading lines of an ontology may hold the banner.
_BANNER_SCAN_LINES = 10

_RUN_GENERATION_MODULE: Optional[Any] = None
_RUN_GENERATION_ERROR: Optional[str] = None


def load_run_generation() -> Optional[Any]:
    """Import scripts/run_generation.py by path, once, defensively.

    The truncation detector is DELIBERATELY not reimplemented here.  Copying it
    would let the scorer's idea of "truncated" drift away from the generator's,
    and then the two would disagree about the same file.  If the module cannot
    be imported the verdict is `None` (unknown) with a note — never a silent
    "not truncated".
    """
    global _RUN_GENERATION_MODULE, _RUN_GENERATION_ERROR
    if _RUN_GENERATION_MODULE is not None or _RUN_GENERATION_ERROR is not None:
        return _RUN_GENERATION_MODULE
    path = Path(__file__).resolve().parent / "scripts" / "run_generation.py"
    try:
        spec = importlib.util.spec_from_file_location("odpgen_run_generation", path)
        if spec is None or spec.loader is None:
            raise ImportError(f"no loader for {path}")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        for name in ("truncation_signals", "extract_turtle",
                     "maybe_fix_common_turtle_issues"):
            if not hasattr(mod, name):
                raise ImportError(f"{path} has no {name}()")
    except Exception as exc:                     # pragma: no cover - defensive
        _RUN_GENERATION_ERROR = f"{type(exc).__name__}: {exc}"
        return None
    _RUN_GENERATION_MODULE = mod
    return mod


def banner_truncation_signals(ontology_text: Optional[str]) -> Optional[List[str]]:
    """Signals named by the ODPGEN-TRUNCATED banner, or None if there is none."""
    if not ontology_text:
        return None
    signals: List[str] = []
    seen_banner = False
    for line in ontology_text.splitlines()[:_BANNER_SCAN_LINES]:
        stripped = line.strip()
        if stripped.startswith(TRUNCATION_BANNER_PREFIX):
            seen_banner = True
        elif stripped.startswith(TRUNCATION_SIGNALS_PREFIX):
            body = stripped[len(TRUNCATION_SIGNALS_PREFIX):]
            signals = [s.strip() for s in body.split(",") if s.strip()]
    if not seen_banner:
        return None
    return signals


def derive_truncation_signals(raw_response: str,
                              finish_reason: Optional[str]) -> Optional[List[str]]:
    """Re-run the generator's own detector over a recorded raw response.

    Returns None when the detector is unavailable — which is "unknown", not
    "complete".
    """
    rg = load_run_generation()
    if rg is None:
        return None
    try:
        turtle = rg.extract_turtle(raw_response)
        _, repairs = rg.maybe_fix_common_turtle_issues(turtle, report=True)
        return list(rg.truncation_signals(raw_response, finish_reason, repairs))
    except Exception:                            # pragma: no cover - defensive
        return None


def _local_sibling_reader(onto_path: Path | str) -> Callable[[str], Optional[str]]:
    base = Path(onto_path).parent

    def read(name: str) -> Optional[str]:
        return read_local_text(base / name)

    return read


def _remote_sibling_reader(onto_path: str) -> Callable[[str], Optional[str]]:
    base = posixpath.dirname(str(onto_path).replace("\\", "/"))

    def read(name: str) -> Optional[str]:
        return fetch_text(posixpath.join(base, name), retries=1)

    return read


def read_truncation(onto_path: Path | str,
                    ontology_text: Optional[str] = None,
                    reader: Optional[Callable[[str], Optional[str]]] = None,
                    ) -> Dict[str, Any]:
    """The truncation verdict for one artefact, and where it came from.

    Returns {"truncated": True|False|None, "signals": [...], "source": str,
             "finish_reason": str|None, "note": str|None}.

    `truncated is None` means UNKNOWN and is reported as such everywhere
    downstream.  Unknown is never laundered into False: an artefact with no
    surviving evidence has not been shown to be complete.
    """
    reader = reader or _local_sibling_reader(onto_path)
    record: Dict[str, Any] = {"truncated": None, "signals": [],
                              "source": "unavailable", "finish_reason": None,
                              "note": None}

    meta_text = reader(METADATA_FILENAME)
    meta: Dict[str, Any] = {}
    if meta_text:
        try:
            loaded = json.loads(meta_text)
            if isinstance(loaded, dict):
                meta = loaded
        except ValueError:
            record["note"] = f"{METADATA_FILENAME} is not valid JSON"

    record["finish_reason"] = meta.get("finish_reason")

    # 1. the authoritative record.
    if "truncated" in meta:
        record["truncated"] = bool(meta["truncated"])
        record["signals"] = list(meta.get("truncation_signals") or [])
        record["source"] = "metadata"
        record["note"] = None
        return record

    # 2. the copy that travels with the artefact.
    if ontology_text is None:
        ontology_text = reader(Path(str(onto_path)).name)
    banner = banner_truncation_signals(ontology_text)
    if banner is not None:
        record["truncated"] = True
        record["signals"] = banner or ["ontology_banner"]
        record["source"] = "ontology_banner"
        record["note"] = None
        return record

    # 3. the raw response — the only evidence the 420 recorded runs still have.
    raw = reader(RAW_RESPONSE_FILENAME)
    if raw is not None:
        derived = derive_truncation_signals(raw, record["finish_reason"])
        if derived is not None:
            record["truncated"] = bool(derived)
            record["signals"] = derived
            record["source"] = "derived_from_raw_response"
            record["note"] = (
                "metadata.json predates the truncated flag; verdict re-derived "
                "from raw_response.txt with the generator's own detector"
            )
            return record
        record["source"] = "detector_unavailable"
        record["note"] = (
            "could not import scripts/run_generation.py to re-derive the "
            f"verdict ({_RUN_GENERATION_ERROR}); completeness is UNKNOWN"
        )
        return record

    record["source"] = "no_evidence"
    record["note"] = (
        f"no 'truncated' key in {METADATA_FILENAME}, no {TRUNCATION_BANNER_PREFIX} "
        f"banner and no {RAW_RESPONSE_FILENAME}: this generation's completeness "
        "is UNKNOWN, which is not the same as complete"
    )
    return record


# ── Shared: parse ontology ─────────────────────────────────────────────────────

def _parse_graph(ontology_text: str):
    """Return (graph, format_used) or raise ValueError if unparseable."""
    from rdflib import Graph
    for fmt in ("turtle", "xml", "n3"):
        try:
            g = Graph()
            g.parse(data=ontology_text, format=fmt)
            return g, fmt
        except Exception:
            continue
    raise ValueError("Could not parse ontology in any known format")


# ── 1. Ontometrics ─────────────────────────────────────────────────────────────

def compute_ontometrics(g) -> Dict[str, Any]:
    from rdflib.namespace import OWL, RDF, RDFS

    classes        = set(g.subjects(RDF.type, OWL.Class)) | set(g.subjects(RDF.type, RDFS.Class))
    obj_props      = set(g.subjects(RDF.type, OWL.ObjectProperty))
    data_props     = set(g.subjects(RDF.type, OWL.DatatypeProperty))
    annot_props    = set(g.subjects(RDF.type, OWL.AnnotationProperty))
    individuals    = set(g.subjects(RDF.type, OWL.NamedIndividual))
    subclass_ax    = sum(1 for _ in g.triples((None, RDFS.subClassOf, None)))
    equiv_ax       = sum(1 for _ in g.triples((None, OWL.equivalentClass, None)))
    disjoint_ax    = sum(1 for _ in g.triples((None, OWL.disjointWith, None)))
    restriction_ax = sum(1 for _ in g.triples((None, RDF.type, OWL.Restriction)))

    nc  = len(classes)
    nop = len(obj_props)
    ndp = len(data_props)

    return {
        "triples_count":               len(g),
        "classes_count":               nc,
        "object_properties_count":     nop,
        "datatype_properties_count":   ndp,
        "annotation_properties_count": len(annot_props),
        "individuals_count":           len(individuals),
        "subclass_axioms":             subclass_ax,
        "equivalence_axioms":          equiv_ax,
        "disjoint_axioms":             disjoint_ax,
        "restriction_axioms":          restriction_ax,
        "attribute_richness":          ndp / nc if nc else 0.0,
        "relationship_richness":       nop / (nop + ndp) if (nop + ndp) else 0.0,
        "avg_subclass_per_class":      subclass_ax / nc if nc else 0.0,
    }


# ── 2. OWL Reasoner (OWL-RL via owlrl) ────────────────────────────────────────

def run_reasoner(g) -> Dict[str, Any]:
    """
    Run OWL-RL reasoning on a copy of the graph.
    Returns consistency verdict, inferred triple count, unsatisfiable classes.
    """
    import owlrl
    from rdflib import Graph, RDF
    from rdflib.namespace import OWL, RDFS

    # work on a copy so the original graph is not mutated
    g2 = Graph()
    for triple in g:
        g2.add(triple)
    for prefix, ns in g.namespaces():
        g2.bind(prefix, ns)

    triples_before = len(g2)
    consistent = True
    inconsistency_reason = None
    unsat_classes: List[str] = []

    try:
        owlrl.DeductiveClosure(owlrl.OWLRL_Semantics,
                               rdfs_closure=True,
                               axiomatic_triples=False,
                               datatype_axioms=False).expand(g2)
    except owlrl.InconsistencyError as e:
        consistent = False
        inconsistency_reason = str(e)[:300]
    except Exception as e:
        return {"error": f"Reasoner failed: {e}"}

    triples_after = len(g2)
    inferred = triples_after - triples_before

    if consistent:
        # owl:Nothing instances → inconsistency detected post-hoc
        nothing_instances = list(g2.subjects(RDF.type, OWL.Nothing))
        if nothing_instances:
            consistent = False
            inconsistency_reason = f"owl:Nothing has instances: {[str(x) for x in nothing_instances[:5]]}"

        # classes explicitly subsumed by owl:Nothing → unsatisfiable
        unsat_classes = [
            str(c) for c in g2.subjects(RDFS.subClassOf, OWL.Nothing)
            if str(c) != str(OWL.Nothing)
        ]

    # F3 / V1 — VACUOUS CONSISTENCY.
    # OWL-RL reports consistent=True on a graph with zero triples: there is
    # nothing there to contradict anything.  That verdict is vacuously true and
    # is NOT evidence of quality, so it must not be convertible into full
    # consistency credit downstream.  `consistent` keeps reporting the literal
    # reasoner verdict (it really is consistent); `consistency_credit` is the
    # number the structural score is allowed to use, and it is 0.0 when the
    # verdict is vacuous.  See structural_content_gate() below.
    vacuous = triples_before == 0
    if vacuous:
        consistency_credit = 0.0
    else:
        consistency_credit = 1.0 if consistent else 0.0

    result: Dict[str, Any] = {
        "consistent":             consistent,
        "vacuous":                vacuous,
        "consistency_credit":     consistency_credit,
        "triples_before":         triples_before,
        "triples_after":          triples_after,
        "inferred_triples":       inferred,
        "unsatisfiable_classes":  unsat_classes,
    }
    if vacuous:
        result["vacuous_reason"] = (
            "0 triples: consistency is vacuously true and earns no credit"
        )
    if inconsistency_reason:
        result["inconsistency_reason"] = inconsistency_reason
    return result


# ── 2b. Structural content gate (fixes F3 / V1) ───────────────────────────────
#
# THE DEFECT.  The published structural score is
#
#     structural_score = mean(consistency, 1 / (1 + oops_pitfalls_total))
#
# (see ISWC2026_ODP_paper/tables/tab_structural_evaluation.tex and the
# reproduction in scripts/audit_artifacts.py).  Both halves are SATISFIED BY
# SAYING NOTHING:
#
#   * An empty graph is trivially consistent — OWL-RL has nothing to
#     contradict — so the consistency half pays 1.0.
#   * An empty graph cannot commit a modelling pitfall, so OOPS! reports zero
#     pitfalls and the OOPS half pays 1 / (1 + 0) = 1.0.
#
# The metric's OPTIMUM was therefore the empty file, at 1.0 — tying the
# published gold reference patterns.  This is not hypothetical: five of the 70
# bigscience_bloomz-7b1 outputs in this repository are comment-only files that
# parse to zero triples, and they scored 1.0.  Under a Turtle extractor that
# correctly rejects prose, all 70 would, and the corpus's worst model would be
# published as structurally perfect.
#
# THE FIX.  Consistency and pitfall-freedom are NECESSARY conditions for a good
# ontology, never sufficient ones, and they are vacuous on a graph with no
# content.  So the score is now gated on content before those signals are read:
#
#   unparseable / 0 triples  -> cap 0.0   (the FLOOR, counted in the denominator)
#   near-empty               -> cap 0.5   (cannot reach the top of the scale)
#   otherwise                -> cap 1.0   (the metric behaves as published)
#
# A floored file is scored 0.0 and stays IN the denominator.  It is never
# "skipped" and never excluded: dropping empty outputs from a model's mean is
# the same bug wearing a different hat, since a model that emits nothing would
# simply vanish from its own average.

# An ontology below this many triples is near-empty.  Justification: 10 is the
# triple count of the SMALLEST published reference pattern in the corpus's own
# gold set (data/ground_truth/2023-134-02.ttl and 2023-134-03.ttl, 10 triples
# each).  Nothing at least as large as the smallest artifact the authors
# themselves treat as a real ODP is capped for size alone.
MIN_STRUCTURAL_TRIPLES = 10

# Cap applied to a near-empty ontology.  It can still be ranked above an empty
# file (a 6-triple stub is more than nothing) but it cannot reach the top half
# of the scale, where only ontologies with actual class/property structure and
# a clean reasoner + OOPS! verdict belong.
NEAR_EMPTY_SCORE_CAP = 0.5

# Types that count as declaring a class.  owl:Restriction is deliberately NOT
# here: an anonymous restriction is a class expression ABOUT declared terms, not
# a vocabulary term of its own.
_CLASS_DECL_TYPES = (OWL.Class, RDFS.Class)

# Types that count as declaring a property.  owl:AnnotationProperty is
# deliberately EXCLUDED: annotation properties carry no logical structure, so a
# file that declares nothing but annotation properties is still structurally
# empty and must not escape the near-empty cap.  rdfs:Property is a
# non-standard spelling of rdf:Property that appears in the gold set
# (2025-151-02.ttl), so it is accepted too.
_PROPERTY_DECL_TYPES = (
    OWL.ObjectProperty, OWL.DatatypeProperty, OWL.FunctionalProperty,
    OWL.InverseFunctionalProperty, OWL.TransitiveProperty, OWL.SymmetricProperty,
    RDF.Property, URIRef(str(RDFS) + "Property"),
)


def _declared_vocabulary(g) -> Tuple[int, int]:
    """(class-like terms, property-like terms) declared in the graph.

    Counted generously on purpose.  The gate exists to catch ontologies that say
    NOTHING, so it must not misfire on a real pattern that happens to use a
    less common spelling (rdfs:Class, rdfs:Property, or a bare rdfs:domain
    declaration with no explicit rdf:type).
    """
    class_like = set()
    for t in _CLASS_DECL_TYPES:
        class_like |= set(g.subjects(RDF.type, t))
    for s, o in g.subject_objects(RDFS.subClassOf):
        class_like.add(s)
        class_like.add(o)
    for pred in (OWL.equivalentClass, OWL.disjointWith):
        for s, o in g.subject_objects(pred):
            class_like.add(s)
            class_like.add(o)

    prop_like = set()
    for t in _PROPERTY_DECL_TYPES:
        prop_like |= set(g.subjects(RDF.type, t))
    for pred in (RDFS.domain, RDFS.range, RDFS.subPropertyOf):
        prop_like |= set(g.subjects(pred, None))
    # a property used inside an owl:Restriction is a declared edge too
    prop_like |= set(g.objects(None, OWL.onProperty))

    return len(class_like), len(prop_like)


def structural_content_gate(g, parse_error: Optional[str] = None) -> Dict[str, Any]:
    """Decide how much of the structural scale an ontology is eligible for.

    Returns {"status", "cap", "reason", "triples_count", "class_terms",
             "property_terms"}.  `cap` is the maximum structural score the file
    may attain; see the module comment above for why.
    """
    if parse_error is not None or g is None:
        return {
            "status": "unparseable", "cap": 0.0,
            "reason": parse_error or "ontology could not be parsed",
            "triples_count": 0, "class_terms": 0, "property_terms": 0,
        }

    triples = len(g)
    if triples == 0:
        return {
            "status": "empty", "cap": 0.0,
            "reason": ("0 triples: consistency and zero-pitfall verdicts are "
                       "vacuous, so the file scores at the floor"),
            "triples_count": 0, "class_terms": 0, "property_terms": 0,
        }

    n_classes, n_props = _declared_vocabulary(g)
    reasons = []
    if triples < MIN_STRUCTURAL_TRIPLES:
        reasons.append(f"{triples} triples < {MIN_STRUCTURAL_TRIPLES}")
    if n_classes == 0:
        reasons.append("declares no classes")
    if n_props == 0:
        reasons.append("declares no properties")

    if reasons:
        return {
            "status": "near_empty", "cap": NEAR_EMPTY_SCORE_CAP,
            "reason": ("near-empty (" + "; ".join(reasons) + "): a clean "
                       "reasoner and OOPS! verdict are close to vacuous here, "
                       f"so the score is capped at {NEAR_EMPTY_SCORE_CAP}"),
            "triples_count": triples, "class_terms": n_classes,
            "property_terms": n_props,
        }

    return {
        "status": "ok", "cap": 1.0, "reason": "has class and property structure",
        "triples_count": triples, "class_terms": n_classes,
        "property_terms": n_props,
    }


def compute_structural_score(reasoner: Optional[Dict[str, Any]],
                             oops: Optional[Dict[str, Any]],
                             gate: Dict[str, Any]) -> Dict[str, Any]:
    """Combine the consistency and OOPS! halves under the content gate.

    The uncapped formula is the published one — mean of the available
    components — so a real ontology is scored exactly as before.  The gate only
    ever removes credit that was awarded for saying nothing.
    """
    cap = float(gate["cap"])

    consistency: Optional[float] = None
    if reasoner and "error" not in reasoner and "consistent" in reasoner:
        # prefer consistency_credit: it is already 0.0 for a vacuous verdict
        consistency = float(reasoner.get(
            "consistency_credit", 1.0 if reasoner.get("consistent") else 0.0))

    oops_component: Optional[float] = None
    if (oops and "error" not in oops and not oops.get("skipped")
            and oops.get("status_code") == 200):
        pitfalls = float(oops.get("pitfalls_total", 0) or 0)
        oops_component = 1.0 / (1.0 + pitfalls)

    if cap == 0.0:
        # Nothing in the file, so neither half may pay out. Report the zeroed
        # components rather than the vacuous 1.0s they would otherwise be.
        consistency = 0.0 if consistency is not None else None
        oops_component = 0.0 if oops_component is not None else None

    available = [c for c in (consistency, oops_component) if c is not None]
    raw = sum(available) / len(available) if available else 0.0
    score = min(raw, cap)

    return {
        "structural_score":      round(score, 4),
        "consistency_component": consistency if consistency is not None else 0.0,
        "oops_component":        oops_component,
        "raw_score":             round(raw, 4),
        "gate":                  gate["status"],
        "gate_cap":              cap,
        "gate_reason":           gate["reason"],
        "triples_count":         gate["triples_count"],
        "class_terms":           gate["class_terms"],
        "property_terms":        gate["property_terms"],
        # A floored file is COUNTED, never skipped: excluding empty outputs from
        # a model's mean would hide exactly the failure this gate exists to show.
        "counted":               True,
        "skipped":               False,
    }


def score_structural(ontology_text: Optional[str],
                     oops: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Structural score for one ontology, from raw text. The real scoring path.

    A missing, empty or unparseable ontology returns structural_score 0.0 with
    counted=True — the floor, in the denominator.
    """
    if ontology_text is None:
        gate = structural_content_gate(None, parse_error="ontology.ttl not found")
        return compute_structural_score(None, None, gate)
    try:
        g, _ = _parse_graph(ontology_text)
    except ValueError as e:
        gate = structural_content_gate(None, parse_error=str(e))
        return compute_structural_score(None, None, gate)

    gate = structural_content_gate(g)
    return compute_structural_score(run_reasoner(g), oops, gate)


# ── 3. CQ Verification — SEV (Schema-Entailment Verification) ─────────────────
#
# Replaces the previous "ask an LLM for a SPARQL SELECT, pass iff it returns
# rows" verifier, which was invalid by construction (F5): the generation prompts
# say "Do NOT create individual instances; output schema axioms only", so every
# generated ODP is a T-Box.  An instance-level SELECT returns zero rows against a
# schema-only graph no matter how good the ontology is.
#
# SEV instead lifts the T-Box into a schema-level graph (classes are nodes,
# declared properties are edges) and ASKs whether the vocabulary each CQ needs
# exists AND is connected there.  Fully offline: rdflib only, no LLM, no network,
# no API key.
#
# It also fixes F4: competency questions are read ONLY from
# data/scenarios/pattern_scenarios.json ("cq_list"), never from prompt.txt, and
# every CQ about to be scored is checked against that list — a CQ that is not in
# it raises CQContaminationError instead of being silently evaluated.
#
# And it closes F3's free ride for empty files: an empty / 3-byte / unparseable
# ontology gets status no_ontology|empty|unparseable and score 0.0, counted IN
# the denominator, never "skipped".

SEV_ENGINE_VERSION = "1.0.0"

_REPO_ROOT       = Path(__file__).resolve().parent
SCENARIOS_PATH   = _REPO_ROOT / "data" / "scenarios" / "pattern_scenarios.json"
# Optional frozen signature artifact.  When present it OVERRIDES the built-in
# signatures below (per scenario).  Authoring the full 175-CQ artifact is the
# committed, version-controlled follow-up; the built-ins cover the scenarios that
# can be gold-calibrated today.
SIGNATURES_PATH  = _REPO_ROOT / "data" / "scenarios" / "cq_signatures.json"
GROUND_TRUTH_DIR = _REPO_ROOT / "data" / "ground_truth"

# Gold-calibration gate: a signature is only trusted if the published reference
# pattern for its scenario scores at least this much on it.
GOLD_CALIBRATION_THRESHOLD = 0.85


class CQContaminationError(ValueError):
    """A CQ that is not in the scenario's authoritative cq_list was supplied.

    This is the F4 bug class: prompt boilerplate ("Use clear, self-explanatory
    class and property names?") was scraped out of prompt.txt and evaluated as if
    it were a competency question, which silently invalidated the paper's
    headline metric.  It must fail loudly, never warn-and-continue.
    """


class SignatureError(ValueError):
    """A CQ Requirement Signature is malformed or refers to an unknown CQ."""


# ── 3a. CQ loading — pattern_scenarios.json is the SINGLE source of truth ─────

_SCENARIO_CACHE: Dict[str, Dict[str, Any]] = {}


def load_scenarios(path: Optional[Path] = None) -> Dict[str, Dict[str, Any]]:
    """Load data/scenarios/pattern_scenarios.json keyed by scenario_id."""
    p = Path(path) if path else SCENARIOS_PATH
    key = str(p.resolve()) if p.exists() else str(p)
    if key in _SCENARIO_CACHE:
        return _SCENARIO_CACHE[key]
    if not p.exists():
        raise FileNotFoundError(f"pattern_scenarios.json not found at {p}")
    raw = json.loads(p.read_text(encoding="utf-8"))
    if isinstance(raw, dict):
        raw = list(raw.values())
    scenarios = {}
    for item in raw:
        sid = item.get("scenario_id")
        if not sid:
            continue
        scenarios[str(sid)] = item
    _SCENARIO_CACHE[key] = scenarios
    return scenarios


def load_cq_list(scenario_id: str, path: Optional[Path] = None) -> List[str]:
    """The authoritative competency questions for a scenario.

    This is the ONLY place CQs may come from.  extract_cqs() (which scraped
    prompt.txt and pulled in prompt boilerplate) has been deleted.
    """
    scenarios = load_scenarios(path)
    if scenario_id not in scenarios:
        raise KeyError(
            f"scenario_id {scenario_id!r} not present in pattern_scenarios.json "
            f"(known: {sorted(scenarios)})"
        )
    cqs = scenarios[scenario_id].get("cq_list") or []
    if not isinstance(cqs, list):
        raise SignatureError(f"cq_list for {scenario_id} is not a list")
    return [str(c) for c in cqs]


def _norm_cq(text: str) -> str:
    """Normalise a CQ for identity comparison (whitespace / case / final mark)."""
    s = re.sub(r"\s+", " ", str(text)).strip().lower()
    return s.rstrip(" ?.!")


def assert_cq_authentic(scenario_id: str, cq: str,
                        path: Optional[Path] = None) -> None:
    """HARD GUARD (F4). Raise unless `cq` is verbatim in the scenario's cq_list."""
    allowed = {_norm_cq(c): c for c in load_cq_list(scenario_id, path)}
    if _norm_cq(cq) not in allowed:
        raise CQContaminationError(
            "Refusing to evaluate a competency question that is not in "
            f"pattern_scenarios.json cq_list for scenario {scenario_id!r}.\n"
            f"  offending CQ: {cq!r}\n"
            "  This is the F4 contamination bug: prompt boilerplate scraped out "
            "of prompts/*.txt was being scored as if it were a real CQ.\n"
            f"  authoritative cq_list ({len(allowed)}): "
            + json.dumps(list(allowed.values()), ensure_ascii=False)
        )


def assert_cqs_authentic(scenario_id: str, cqs: List[str],
                         path: Optional[Path] = None) -> None:
    for cq in cqs:
        assert_cq_authentic(scenario_id, cq, path)


# ── 3b. CQ Requirement Signatures ────────────────────────────────────────────
#
# A signature decomposes ONE competency question into the vocabulary the ODP must
# declare in order to be able to answer it, plus the relations that must connect
# that vocabulary:
#
#   {"cq": "...",
#    "cq_type": "structural" | "data_dependent" | "meta" | "out_of_scope",
#    "slots":     {"EVENT": {"kind": "class",
#                            "surface": ["event", "occurrence", ...],
#                            "anchors": ["http://www.w3.org/ns/prov#Activity"]}},
#    "relations": [{"from": "EVENT", "to": "OUTCOME" | "external" | "literal",
#                   "surface": ["causes", "has outcome", ...]}]}
#
# `to: "external"` means the CQ asks for the filler of a role on an anchor class
# and the filler legitimately comes from a reused vocabulary the pattern does not
# redefine — so anchoring the property's DOMAIN is the whole requirement.  This
# mode is not optional: the published reference pattern 2025-151-01 declares
# rdfs:domain on all five of its properties and rdfs:range on none of them, and a
# metric that demanded both would score a correct, published ODP at zero.
#
# Signatures are CQ-specific, never ODP-specific: the same signature scores all
# 30 model x config outputs for that scenario, so no configuration can be
# advantaged.  They are authored from the CQ text, the scenario text and the
# published reference pattern's own labels ONLY — never from any generated
# output, and in particular never from the "CQ-to-axiom mapping table" that the
# models emit in raw_response.txt (that would feed the system under test's own
# self-report back into its evaluator).
#
# Every signature must clear the gold-calibration gate: run_gold_calibration()
# scores it against data/ground_truth/<scenario>.* and requires >= 0.85.  A CQ
# the published reference genuinely cannot answer is marked gold_unsupported and
# excluded from the primary aggregate, with the exclusion count reported.

# The catalogue itself is no longer a code literal.  It lives in
# data/scenarios/cq_signatures.json — a committed, version-controlled data file
# — so that signatures_sha() certifies something an auditor can diff and a paper
# can cite.  Hashing a dict embedded in mutable source certified nothing: any
# edit to batch_evaluate.py silently changed the "provenance" hash, and any edit
# to the signatures was invisible in the artifact tree.  The name
# _BUILTIN_CQ_SIGNATURES is kept because it is part of this module's public API;
# it is now populated by reading that file at import time.

def _load_frozen_signatures(path: Optional[Path] = None
                            ) -> Dict[str, List[Dict[str, Any]]]:
    """Read the frozen signature artifact.  Returns {} when it is absent."""
    p = Path(path) if path else SIGNATURES_PATH
    if not p.exists():
        return {}
    raw = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise SignatureError(f"{p} must be a JSON object keyed by scenario_id")
    for sid, records in raw.items():
        if not isinstance(records, list):
            raise SignatureError(f"{p}: signatures for {sid} must be a list")
        for rec in records:
            if not isinstance(rec, dict) or "cq" not in rec:
                raise SignatureError(
                    f"{p}: every signature for {sid} needs a 'cq' field: {rec!r}")
    return raw


_BUILTIN_CQ_SIGNATURES: Dict[str, List[Dict[str, Any]]] = _load_frozen_signatures()


# When True, CQs without a hand-authored signature get a deterministic derived
# one so the whole corpus stays scorable; they are always flagged
# signature_source="derived" and are excluded from the *_authored aggregates.
# Set to False to score only the gold-calibrated, hand-authored subset.
SCORE_UNSIGNED_CQS = True

_SIGNATURE_CACHE: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}


def load_cq_signatures(path: Optional[Path] = None) -> Dict[str, List[Dict[str, Any]]]:
    """Load the frozen signature artifact, data/scenarios/cq_signatures.json.

    That file IS the catalogue (see _load_frozen_signatures); an explicitly
    supplied `path` overrides it per scenario, which is how tests and
    alternative signature sets are swapped in.
    """
    p = Path(path) if path else SIGNATURES_PATH
    key = str(p)
    if key in _SIGNATURE_CACHE:
        return _SIGNATURE_CACHE[key]
    sigs: Dict[str, List[Dict[str, Any]]] = {
        k: list(v) for k, v in _BUILTIN_CQ_SIGNATURES.items()
    }
    if p.exists():
        for sid, records in _load_frozen_signatures(p).items():
            sigs[str(sid)] = records
    elif not sigs:
        raise SignatureError(
            f"the frozen CQ signature artifact is missing: {p}\n"
            "  The hand-authored CQ Requirement Signatures live in "
            "data/scenarios/cq_signatures.json and are the metric's answer key; "
            "without it every CQ silently falls back to the weak derived "
            "signature and signatures_sha() certifies an empty catalogue.")
    _SIGNATURE_CACHE[key] = sigs
    return sigs


def signatures_sha(path: Optional[Path] = None) -> str:
    """Stable digest of the signature artifact actually in use (auditability).

    Because the catalogue is now a data file rather than a literal inside this
    module, this digest identifies a diffable artifact: two runs reporting the
    same signatures_sha were scored against byte-identical signatures.
    """
    payload = json.dumps(load_cq_signatures(path), sort_keys=True,
                         ensure_ascii=False, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


_META_CQ_RE = re.compile(
    r"\b(can|does|is)\s+(the\s+)?(ontology|pattern|model|odp)\b|"
    r"\bcan\s+(the\s+)?ontology\s+represent\b", re.IGNORECASE)
_DATA_CQ_RE = re.compile(
    r"^\s*how\s+many\b|\bmost\s+common\b|\bthe\s+most\b|\baverage\b|"
    r"\btotal\s+number\b|\bhow\s+much\s+\w+\s+(is|are|was|were)\b", re.IGNORECASE)


def classify_cq(cq: str) -> str:
    """structural | data_dependent | meta — a signature's cq_type overrides this.

    Some CQs in the corpus are not structural questions at all: 2025-149-01 asks
    "Can the ontology represent X?" (near-tautological under SEV) and 2025-147-01
    contains data questions ("How many container ships are registered under the
    German flag?") that no T-Box can answer.  They are scored but reported
    separately so they cannot silently depress the primary aggregate the way the
    old metric let them.
    """
    if _META_CQ_RE.search(cq):
        return "meta"
    if _DATA_CQ_RE.search(cq):
        return "data_dependent"
    return "structural"


_DERIVE_SPLITS = [" associated with ", " related to ", " belongs to ",
                  " used in ", " performed on ", " of the ", " for the ",
                  " in the ", " on the ", " of a ", " of an ", " of ",
                  " for ", " in ", " on ", " about "]
_DERIVE_LEAD_RE = re.compile(
    r"^\s*(what|which|who|whom|whose|when|where|why|how(\s+\w+)?)\b"
    r"(\s+(is|are|was|were|do|does|did|can|could|would|should|has|have|had))?"
    r"(\s+(the|a|an))?\s*", re.IGNORECASE)


def derive_signature(cq: str) -> Dict[str, Any]:
    """Deterministic fallback signature for a CQ with no hand-authored record.

    Deliberately weak and always flagged: it probes only whether the ODP declares
    a class matching some CQ term AND a property matching some CQ term that is
    anchored on that class.  A pure glossary (classes but no domain/range) still
    scores near the floor, but this is NOT a gold-calibrated signature and its
    results are excluded from the *_authored aggregates.
    """
    body = _DERIVE_LEAD_RE.sub("", str(cq).strip()).strip(" ?.!,")
    body = re.sub(r"\s+", " ", body)
    lowered = " " + body.lower() + " "
    head, tail = body, body
    for sep in _DERIVE_SPLITS:
        idx = lowered.rfind(sep)
        if idx > 0:
            head = body[:idx].strip(" ?.!,")
            tail = body[idx + len(sep) - 1:].strip(" ?.!,")
            break

    def phrases(text: str) -> List[str]:
        words = [w for w in re.findall(r"[A-Za-z][A-Za-z0-9\-]*", text.lower())
                 if w not in _DERIVE_STOP and len(w) > 2]
        out = list(dict.fromkeys(words))
        out += [f"{a} {b}" for a, b in zip(words, words[1:])]
        return list(dict.fromkeys(out))[:12]

    subject = phrases(tail) or phrases(body) or ["thing"]
    relation = phrases(head) or subject
    return {
        "cq": cq,
        "cq_type": classify_cq(cq),
        "derived": True,
        "slots": {"SUBJECT": {"kind": "class", "surface": subject}},
        "relations": [{"from": "SUBJECT", "to": "external", "surface": relation}],
    }


_DERIVE_STOP = {
    "the", "a", "an", "of", "in", "on", "to", "for", "by", "with", "and", "or",
    "is", "are", "was", "were", "be", "been", "being", "that", "this", "these",
    "those", "it", "its", "their", "there", "any", "some", "all", "can", "could",
    "would", "should", "has", "have", "had", "do", "does", "did", "given",
    "specific", "particular", "e.g", "eg", "etc", "if", "not", "what", "which",
    "who", "when", "where", "why", "how", "at", "from", "as", "into", "about",
}


def signatures_for(scenario_id: str,
                   cqs: Optional[List[str]] = None,
                   scenarios_path: Optional[Path] = None,
                   signatures_path: Optional[Path] = None) -> List[Dict[str, Any]]:
    """Signatures aligned 1:1 with the scenario's authoritative cq_list.

    Applies the F4 hard guard to every CQ, in both directions:
      * a CQ that is not in cq_list raises CQContaminationError;
      * a signature whose `cq` is not in cq_list raises SignatureError (a stale
        or hand-mistyped signature must not silently score nothing).
    """
    authoritative = load_cq_list(scenario_id, scenarios_path)
    if cqs is None:
        cqs = authoritative
    else:
        assert_cqs_authentic(scenario_id, cqs, scenarios_path)

    authored = load_cq_signatures(signatures_path).get(scenario_id, [])
    allowed = {_norm_cq(c) for c in authoritative}
    by_cq: Dict[str, Dict[str, Any]] = {}
    for rec in authored:
        if "cq" not in rec:
            raise SignatureError(
                f"signature for {scenario_id} has no 'cq' field: {rec!r}")
        key = _norm_cq(rec["cq"])
        if key not in allowed:
            raise SignatureError(
                f"signature for scenario {scenario_id} refers to a CQ that is not "
                f"in pattern_scenarios.json cq_list: {rec['cq']!r}")
        by_cq[key] = rec

    out: List[Dict[str, Any]] = []
    for cq in cqs:
        rec = by_cq.get(_norm_cq(cq))
        if rec is not None:
            sig = dict(rec)
            sig["cq"] = cq
            sig.setdefault("cq_type", classify_cq(cq))
            sig["signature_source"] = "authored"
        elif SCORE_UNSIGNED_CQS:
            sig = derive_signature(cq)
            sig["signature_source"] = "derived"
        else:
            sig = {"cq": cq, "cq_type": classify_cq(cq),
                   "slots": {}, "relations": [],
                   "signature_source": "unsigned"}
        out.append(sig)
    return out


# ── 3c. SEV engine — schema lifting, binding, connectivity ───────────────────

SEV_NS = Namespace("urn:sev:")
SEV_ANY = SEV_NS.Anything        # sentinel: endpoint left open (no domain/range)
SEV_LIT = SEV_NS.LiteralValue    # sentinel: datatype endpoint

_CAMEL_RE = re.compile(r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])")
_TOKEN_STOP = {"has", "is", "the", "a", "an", "of", "in", "on", "to", "for",
               "by", "with", "and", "or", "was", "were", "be"}
_BAD_IRI_RE = re.compile(r"[\s<>\"{}|\\^`]")

_META_NAMESPACES = (
    "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
    "http://www.w3.org/2000/01/rdf-schema#",
    "http://www.w3.org/2002/07/owl#",
    "http://www.w3.org/2001/XMLSchema#",
)


def _suffix_strip(t: str) -> str:
    for suf, keep in (("ies", 3), ("ses", 2), ("es", 2), ("s", 1), ("ing", 3),
                      ("ed", 2), ("ation", 5), ("tion", 4)):
        if len(t) > len(suf) + 2 and t.endswith(suf):
            return t[:-keep]
    return t


def sev_tokens(s: Any) -> set:
    s = _CAMEL_RE.sub(" ", str(s))
    s = re.sub(r"[^A-Za-z0-9]+", " ", s).lower()
    ts = [t for t in s.split() if t and t not in _TOKEN_STOP]
    return set(ts) | {_suffix_strip(t) for t in ts}


def sev_flat(s: Any) -> str:
    return re.sub(r"[^a-z0-9]", "", str(s).lower())


def _is_meta(u) -> bool:
    s = str(u)
    return any(s.startswith(x) for x in _META_NAMESPACES)


class SchemaView:
    """The T-Box of a candidate ODP, as classes + properties + closures."""

    def __init__(self, g):
        self.g = g
        self.classes = set()
        self.objprops = set()
        self.dataprops = set()
        self._build()

    def _build(self):
        g = self.g
        for c in list(g.subjects(RDF.type, OWL.Class)) + list(g.subjects(RDF.type, RDFS.Class)):
            if isinstance(c, URIRef):
                self.classes.add(c)
        for p in g.subjects(RDF.type, OWL.ObjectProperty):
            if isinstance(p, URIRef):
                self.objprops.add(p)
        for p in g.subjects(RDF.type, OWL.DatatypeProperty):
            if isinstance(p, URIRef):
                self.dataprops.add(p)
        for _, _, o in g.triples((None, RDFS.subClassOf, None)):
            if isinstance(o, URIRef):
                self.classes.add(o)
        for _, _, o in g.triples((None, RDFS.domain, None)):
            if isinstance(o, URIRef):
                self.classes.add(o)
        for _, _, o in g.triples((None, RDFS.range, None)):
            if isinstance(o, URIRef) and "XMLSchema" not in str(o):
                self.classes.add(o)
        for pred in (OWL.someValuesFrom, OWL.allValuesFrom, OWL.onClass):
            for _, _, o in g.triples((None, pred, None)):
                if isinstance(o, URIRef):
                    self.classes.add(o)
        # PUNNING: ":X a :Y" with a local :Y promotes :Y to a class but leaves :X
        # an individual, so an ODP written purely as instance data earns credit
        # only for the types it names — the honest verdict under a prompt that
        # says "output schema axioms only".
        for _, _, o in g.triples((None, RDF.type, None)):
            if isinstance(o, URIRef) and not _is_meta(o):
                self.classes.add(o)
        # properties implied by structural position
        preds = {p for _, p, _ in g if isinstance(p, URIRef) and not _is_meta(p)}
        for p in preds:
            if p in self.dataprops or p in self.objprops:
                continue
            objs = list(g.objects(None, p))
            if objs and all(isinstance(o, Literal) for o in objs):
                self.dataprops.add(p)
            else:
                self.objprops.add(p)
        for p in set(g.subjects(RDFS.domain, None)) | set(g.subjects(RDFS.range, None)):
            if isinstance(p, URIRef) and p not in self.dataprops and p not in self.objprops:
                self.objprops.add(p)
        self.classes -= (self.objprops | self.dataprops)
        # Only RDF/RDFS/OWL/XSD are meta-vocabulary.  dcterms, skos, sosa, odrl,
        # prov and friends stay bindable: they are legitimate ODP building blocks
        # and filtering them drops a published gold pattern below the gate.
        self.classes = {c for c in self.classes if not _is_meta(c)}
        self.objprops = {p for p in self.objprops if not _is_meta(p)}
        self.dataprops = {p for p in self.dataprops if not _is_meta(p)}
        # Locally-declared owl:Thing substitutes (see SEV_HUB_FRACTION).  Must be
        # computed last: it reads the finished class set.
        self.universal_hubs = self._find_universal_hubs()

    def _find_universal_hubs(self) -> set:
        """Named classes that subsume so much of the ontology that anchoring a
        property on them says nothing — the mega-hub exploit."""
        n = len(self.classes)
        if n < 2:
            return set()
        hubs = set()
        for c in self.classes:
            proper = len(self.descendants(c)) - 1
            if (proper >= SEV_HUB_MIN_DESCENDANTS
                    and proper / float(n) > SEV_HUB_FRACTION):
                hubs.add(c)
        return hubs

    def declared_endpoints(self, p) -> Tuple[set, set]:
        """(domains, ranges) of p as schema-graph nodes, after sub-property
        closure — the same reading lift() uses, exposed for fan-out scoring."""
        doms, rngs = set(), set()
        for q in self.superprops(p):
            for d in self.g.objects(q, RDFS.domain):
                if isinstance(d, URIRef):
                    doms.add(self._endpoint(d))
            for r in self.g.objects(q, RDFS.range):
                if isinstance(r, URIRef):
                    rngs.add(self._endpoint(r))
        if p in self.dataprops and not rngs:
            rngs.add(SEV_LIT)
        return (doms or {SEV_ANY}), (rngs or {SEV_ANY})

    # subsumption / sub-property closures ---------------------------------
    def ancestors(self, c, depth: int = 4) -> set:
        out, frontier = {c}, {c}
        for _ in range(depth):
            nxt = set()
            for x in frontier:
                for o in self.g.objects(x, RDFS.subClassOf):
                    if isinstance(o, URIRef) and o not in out:
                        nxt.add(o)
                for o in self.g.objects(x, OWL.equivalentClass):
                    if isinstance(o, URIRef) and o not in out:
                        nxt.add(o)
            out |= nxt
            frontier = nxt
            if not nxt:
                break
        return out

    def descendants(self, c, depth: int = 4) -> set:
        out, frontier = {c}, {c}
        for _ in range(depth):
            nxt = set()
            for x in frontier:
                for s in self.g.subjects(RDFS.subClassOf, x):
                    if isinstance(s, URIRef) and s not in out:
                        nxt.add(s)
            out |= nxt
            frontier = nxt
            if not nxt:
                break
        return out

    def cone(self, c) -> set:
        """Ancestors + descendants.  Neither owl:Thing nor a locally-declared
        universal superclass ever joins a cone: a mega-hub must not become a
        universal connector.  Without the second exclusion, declaring
        `:X rdfs:subClassOf :Hub` for every class and hanging every property off
        :Hub scores a name-stuffed ontology 1.0."""
        return {x for x in (self.ancestors(c) | self.descendants(c))
                if not _is_meta(x) and x not in self.universal_hubs}

    def superprops(self, p, depth: int = 3) -> set:
        out, fr = {p}, {p}
        for _ in range(depth):
            nxt = set()
            for x in fr:
                for o in self.g.objects(x, RDFS.subPropertyOf):
                    if isinstance(o, URIRef) and o not in out:
                        nxt.add(o)
            out |= nxt
            fr = nxt
            if not nxt:
                break
        return out

    # schema-lifted edge graph --------------------------------------------
    def _endpoint(self, u):
        """Map a declared endpoint onto a schema-graph node.

        owl:Thing is an *open* endpoint, not a class: the OWL-API idiom
        `owl:Thing rdfs:subClassOf [ onProperty p ; allValuesFrom F ]` is how a
        global range is written (the published reference 2023-135-01 does exactly
        this), and treating owl:Thing as a real node would also let a single
        mega-hub connect everything.

        A locally-declared universal superclass is owl:Thing under another IRI,
        so it is mapped the same way; otherwise `p rdfs:domain :Hub` buys the
        same universal connectivity that owl:Thing is already refused.
        """
        if u == OWL.Thing:
            return SEV_ANY
        if "XMLSchema" in str(u):
            return SEV_LIT
        if u in getattr(self, "universal_hubs", ()):
            return SEV_ANY
        return u

    def lift(self):
        """Turn the T-Box into a schema-level A-Box of (class, prop, class) edges."""
        g, SG = self.g, Graph()
        for p in sorted(self.objprops | self.dataprops, key=str):
            doms, rngs = set(), set()
            for q in self.superprops(p):
                for d in g.objects(q, RDFS.domain):
                    if isinstance(d, URIRef):
                        doms.add(self._endpoint(d))
                for r in g.objects(q, RDFS.range):
                    if isinstance(r, URIRef):
                        rngs.add(self._endpoint(r))
            if p in self.dataprops and not rngs:
                rngs.add(SEV_LIT)
            doms = doms or {SEV_ANY}
            rngs = rngs or {SEV_ANY}
            inverses = [q for q in (list(g.objects(p, OWL.inverseOf)) +
                                    list(g.subjects(OWL.inverseOf, p)))
                        if isinstance(q, URIRef)]
            symmetric = (p, RDF.type, OWL.SymmetricProperty) in g
            for d in doms:
                for r in rngs:
                    SG.add((d, p, r))
                    for q in inverses:
                        SG.add((r, q, d))
                    if symmetric:
                        SG.add((r, p, d))
        # C rdfs:subClassOf [ onProperty p ; someValuesFrom/allValuesFrom/onClass F ]
        for c, _, restr in g.triples((None, RDFS.subClassOf, None)):
            if not isinstance(c, URIRef):
                continue
            self._lift_restriction(SG, self._endpoint(c), restr)
        # the inverted OWL-API general-axiom form:
        #   [ onProperty p ; someValuesFrom F ] rdfs:subClassOf C
        # i.e. "anything with p to F is a C" — an implicit domain declaration.
        for restr, _, c in g.triples((None, RDFS.subClassOf, None)):
            if isinstance(restr, BNode) and isinstance(c, URIRef):
                self._lift_restriction(SG, self._endpoint(c), restr)
        for c, _, restr in g.triples((None, OWL.equivalentClass, None)):
            if isinstance(c, URIRef):
                self._lift_restriction(SG, self._endpoint(c), restr)
        return SG

    def _lift_restriction(self, SG, class_node, restr):
        props = [p for p in self.g.objects(restr, OWL.onProperty)
                 if isinstance(p, URIRef)]
        if not props:
            return
        fillers = [f for f in (list(self.g.objects(restr, OWL.someValuesFrom)) +
                               list(self.g.objects(restr, OWL.allValuesFrom)) +
                               list(self.g.objects(restr, OWL.onClass)) +
                               list(self.g.objects(restr, OWL.onDataRange)))
                   if isinstance(f, URIRef)] or [SEV_ANY]
        for p in props:
            for f in fillers:
                SG.add((class_node, p, self._endpoint(f)))

    # lexical index --------------------------------------------------------
    def index(self) -> Dict[Any, Tuple[set, set]]:
        """Token / flat-string index per term.

        Built from the CamelCase-split local name plus rdfs:label,
        skos:prefLabel and skos:altLabel — deliberately NOT rdfs:comment, so a
        model cannot earn credit by echoing the scenario prose into comments.
        """
        idx: Dict[Any, Tuple[set, set]] = {}
        for u in self.classes | self.objprops | self.dataprops:
            local = str(u).rsplit("#", 1)[-1].rsplit("/", 1)[-1]
            bag, flats = sev_tokens(local), {sev_flat(local)}
            for lp in (RDFS.label, SKOS.prefLabel, SKOS.altLabel):
                for lab in self.g.objects(u, lp):
                    bag |= sev_tokens(lab)
                    flats.add(sev_flat(lab))
            idx[u] = (bag, flats)
        return idx


# ── binding ──────────────────────────────────────────────────────────────────

def bind_slot(sv: SchemaView, idx, kind: str, surfaces: List[str],
              anchors: Optional[List[str]] = None) -> Tuple[Any, float, str]:
    """Bind a signature slot to a term of the candidate ODP.

    Tiers: exact normalised string 1.00, full token-subset 0.85 (so
    "AbstractEvent" binds "event"), >= 50% token overlap 0.70, else 0.
    A declared external anchor URI binds at 1.0 regardless of lexical form.
    """
    if kind == "class":
        pool = sv.classes
    elif kind == "objprop":
        pool = sv.objprops
    elif kind == "dataprop":
        pool = sv.dataprops
    elif kind == "prop":
        pool = sv.objprops | sv.dataprops
    else:
        pool = sv.classes | sv.objprops | sv.dataprops

    for a in (anchors or []):
        au = URIRef(a)
        if au in pool:
            return au, 1.0, "anchor"

    # Ties are broken by surface-form ORDER, then by URI, so the binding is both
    # deterministic and steered by the signature author's canonical term: the
    # first surface form listed is the one the pattern is expected to use.
    prepared = [(sev_flat(x), sev_tokens(x)) for x in surfaces]
    best: Tuple[Any, float, str] = (None, 0.0, "none")
    best_rank = len(prepared) + 1
    for u in sorted(pool, key=str):          # sorted => deterministic ties
        bag, flats = idx[u]
        u_score, u_rank, u_tier = 0.0, len(prepared) + 1, "none"
        for rank, (fs, st) in enumerate(prepared):
            if fs and fs in flats:
                sc, tier = 1.00, "exact"
            elif st and st <= bag:
                sc, tier = 0.85, "token-subset"
            elif st and (st & bag) and len(st & bag) / len(st) >= 0.5:
                sc, tier = 0.70, "partial"
            else:
                continue
            if (sc, -rank) > (u_score, -u_rank):
                u_score, u_rank, u_tier = sc, rank, tier
        if (u_score, -u_rank) > (best[1], -best_rank):
            best, best_rank = (u, u_score, u_tier), u_rank
    return best


def _name_matches(idx, u, surfaces) -> bool:
    if u not in idx or not surfaces:
        return False
    bag, flats = idx[u]
    if {sev_flat(s) for s in surfaces} & flats:
        return True
    for s in surfaces:
        st = sev_tokens(s)
        if st and (st <= bag or len(st & bag) / len(st) >= 0.5):
            return True
    return False


def match_props(sv: SchemaView, idx, surfaces: List[str]) -> set:
    pool = sv.objprops | sv.dataprops
    if not surfaces:
        return set(pool)
    sfl = {sev_flat(s) for s in surfaces}
    sft = [sev_tokens(s) for s in surfaces]
    P = set()
    for u in pool:
        bag, flats = idx[u]
        if sfl & flats:
            P.add(u)
            continue
        for st in sft:
            if st and (st <= bag or len(st & bag) / len(st) >= 0.5):
                P.add(u)
                break
    return P


# ── connectivity: SPARQL ASK over the lifted schema graph ────────────────────

SEV_TIERS = {"direct": 1.00, "reified": 1.00,
             "external": 0.85, "indirect2": 0.85,
             "half": 0.75, "external_uncertified": 0.60,
             "external_open": 0.40,
             "path3": 0.60, "floating": 0.15, "none": 0.00}

# ── anti-degeneracy: fan-out and universal hubs ─────────────────────────────
#
# V3: a mechanically-built cross-product ontology (one owl:Class per signature
# surface term, one owl:ObjectProperty per relation surface, every class
# declared as both rdfs:domain and rdfs:range of every property) scored 1.0 —
# STRICTLY ABOVE the published gold reference pattern.  Two independent holes
# let it through, and both are closed here.
#
# HOLE 1 — the cross product.  lift() reads a property's several rdfs:domain
# axioms DISJUNCTIVELY, emitting one schema edge per (domain, range) pair.  RDFS
# says the opposite: multiple rdfs:domain axioms are a CONJUNCTION (every
# subject of p belongs to *all* of them).  So an ontology that names 14 classes
# as domains of one property is not saying "p connects any of these"; it is
# saying almost nothing, and pricing it as 14 separate anchored connections is
# unsound in the direction that rewards the attacker.  Rather than change
# lift()'s shape (restrictions and inverses depend on it) the credit for a
# property is discounted by its endpoint fan-out: the number of pairwise
# INCOMPARABLE named classes it declares on each side.  A subsumption chain
# counts once — its RDFS intersection is simply its most specific member, which
# is a coherent, committed declaration.
#
# SEV_FANOUT_FREE is the per-side allowance that costs nothing.  It is set to 2
# rather than 1 because no property in any of the six calibratable published
# reference patterns declares more than ONE named domain or ONE named range
# (measured), so 2 leaves a full extra degree of freedom for honest sloppiness —
# a generated ODP that names two alternative domains keeps full credit — while
# the attack, which needs breadth to stuff names, collapses: 14 x 14 = 196
# incomparable pairs discount to 4/196 = 0.02.
#
# HOLE 2 — the mega-hub.  Declaring `:X rdfs:subClassOf :Hub` for every class and
# `p rdfs:domain :Hub ; rdfs:range :Hub` gives every property exactly one domain
# and one range, so no fan-out counter sees it; connectivity comes entirely from
# climbing the subsumption cone into a class that subsumes the whole ontology.
# _endpoint() already refuses to treat owl:Thing as a real node for exactly this
# reason ("treating owl:Thing as a real node would let a single mega-hub connect
# everything").  A locally-declared universal superclass is the same object with
# a different IRI, so the same rule is applied to it: a named class that
# subsumes more than SEV_HUB_FRACTION of the ontology's classes is an open
# endpoint, not a connector.
#
# SEV_HUB_MIN_DESCENDANTS keeps the rule off small honest patterns, where a top
# class legitimately covers most of a handful of classes.  Measured on the six
# calibratable references, the most subsuming class is `Event` in the causal
# pattern with 2 proper descendants out of 5 classes (0.40) — under both bars.
SEV_FANOUT_FREE = 2
SEV_HUB_FRACTION = 0.5
SEV_HUB_MIN_DESCENDANTS = 4


def _incomparable_endpoints(sv, nodes) -> int:
    """How many pairwise-incomparable NAMED classes this endpoint set declares.

    Open endpoints (SEV_ANY / SEV_LIT) and universal hubs are not commitments and
    do not count.  A subsumption chain collapses to one representative.
    """
    named = [n for n in nodes
             if n not in (SEV_ANY, SEV_LIT) and n not in sv.universal_hubs]
    if len(named) <= 1:
        return 1
    reps = 0
    for n in named:
        anc = sv.ancestors(n)
        if any(o != n and o in anc for o in named):
            continue                 # subsumed by another declared endpoint
        reps += 1
    return max(1, reps)


def prop_specificity(sv, p) -> float:
    """Fan-out discount in (0, 1] for one property.

    1.0 while the property stays within SEV_FANOUT_FREE incomparable endpoints
    per side; then 1/x decay in the number of (domain, range) pairs it claims.
    """
    doms, rngs = sv.declared_endpoints(p)
    fan = _incomparable_endpoints(sv, doms) * _incomparable_endpoints(sv, rngs)
    free = SEV_FANOUT_FREE ** 2
    return 1.0 if fan <= free else free / float(fan)


def _values(uris) -> str:
    safe = [u for u in uris if not _BAD_IRI_RE.search(str(u))]
    return " ".join("<%s>" % u for u in sorted(map(str, safe)))


def _ask(SG, q: str) -> bool:
    try:
        return bool(SG.query(q))
    except Exception:
        return False


def relation_credit(sv, SG, idx, A_uri, B_uri, B_mode, rel_surfaces,
                    uncertified: bool = False):
    """Strongest structural connection between two bound endpoints.

    The ladder itself is _relation_tier(); this wrapper applies the fan-out
    discount (see SEV_FANOUT_FREE).  Because the discount is per PROPERTY and the
    ladder ASKs over a whole VALUES set, the matched properties are partitioned
    by specificity and the ladder is run once per partition, taking the best
    tier x specificity.  There is normally one partition, so this costs nothing
    on real ontologies; a cross-product name-stuffer lands in a partition whose
    specificity is near zero and cannot borrow credit from an honest property.
    """
    if A_uri is None:
        return "none", 0.0, None
    Ac = sv.cone(A_uri)
    P = match_props(sv, idx, rel_surfaces)
    if not Ac or not P:
        return "none", 0.0, None

    buckets: Dict[float, set] = {}
    for p in P:
        buckets.setdefault(round(prop_specificity(sv, p), 6), set()).add(p)

    best: Tuple[str, float, Any] = ("none", 0.0, None)
    for spec in sorted(buckets, reverse=True):
        if spec <= best[1] + 1e-9:
            break                      # nothing left can beat what we have
        tier, value, why = _relation_tier(sv, SG, idx, A_uri, B_uri, B_mode,
                                          buckets[spec], rel_surfaces,
                                          uncertified)
        credit = value * spec
        if credit > best[1]:
            note = why
            if why is not None and spec < 1.0:
                note = "%s (fan-out discounted x%.3f)" % (why, spec)
            best = (tier, credit, note)
    return best


def _relation_tier(sv, SG, idx, A_uri, B_uri, B_mode, P, rel_surfaces,
                   uncertified=False):
    """The tier ladder, over one set of candidate properties."""
    Ac = sv.cone(A_uri)
    if not Ac or not P:
        return "none", 0.0, None
    if B_mode == "literal":
        Bc = {SEV_LIT}
    elif B_mode == "external":
        Bc = None
    else:
        if B_uri is None:
            return "none", 0.0, None
        Bc = sv.cone(B_uri)
        if not Bc:
            return "none", 0.0, None

    VA, VP = _values(Ac), _values(P)
    if not VA or not VP:
        return "none", 0.0, None

    # T1 direct: rdfs:domain/range, or a restriction, links the two endpoints.
    if Bc is not None:
        VB = _values(Bc)
        if VB and _ask(SG, "ASK { VALUES ?a {%s} VALUES ?p {%s} VALUES ?b {%s} "
                            "?a ?p ?b }" % (VA, VP, VB)):
            return "direct", SEV_TIERS["direct"], "domain/range or restriction"
    else:
        # Open filler: the CQ asks for the filler of a role whose value comes
        # from a vocabulary the pattern deliberately does not redefine, so
        # anchoring the role on its domain class is most of the requirement (the
        # reference 2025-151-01 declares rdfs:domain on all five of its
        # properties and rdfs:range on none).
        #
        # It is NOT the direct tier, though.  `direct` means both endpoint
        # constraints of the CQ were verified against the ontology; `external`
        # verifies exactly one of them and takes the other on trust.  Pricing
        # them the same is V4: it let a single rdfs:domain buy full connectivity
        # credit.  0.85 is the ladder's existing price for "connected, but not by
        # a directly anchored domain+range pair" (indirect2), and it is the
        # HIGHEST value that still lets the range-free published reference
        # 2025-151-01 clear the 0.85 gold gate
        # (0.25 * 1.0 + 0.75 * 0.85 = 0.8875) — so it is the maximum defensible
        # discount rather than an arbitrary one.
        #
        # `uncertified` marks a relation from a DERIVED signature.  `external`
        # in a hand-authored signature is a curator's determination that the
        # filler genuinely lies outside the pattern; derive_signature() emits
        # `to: "external"` unconditionally for every CQ, precisely because it
        # cannot make that determination — and it discards the CQ's own object
        # while doing so, so nothing on the far end is ever checked.  An
        # uncertified claim must therefore land strictly below
        # SEV_ANSWERABLE_L2 (0.75): an uncalibrated fallback signature may never
        # report a CQ as answerable.  Derived signatures cover ~79% of the
        # corpus, and the gold gate only ever scores authored signatures, so this
        # discount touches no published reference pattern.
        if not uncertified:
            if _ask(SG, "ASK { VALUES ?a {%s} VALUES ?p {%s} ?a ?p ?b }"
                        % (VA, VP)):
                return "external", SEV_TIERS["external"], \
                       "role anchored on domain class, filler outside the pattern"
        else:
            # An uncertified external claim is still GRADED, so the fallback
            # keeps discriminating between real ontologies instead of
            # flattening 79% of the corpus onto one number.  Two rungs:
            #   external_uncertified (0.60) — the property is anchored at
            #     BOTH ends (a declared rdfs:range, a restriction filler, or
            #     a datatype); only the IDENTITY of the far end is
            #     unverified, because the deriver discarded the CQ object.
            #   external_open (0.40) — a bare rdfs:domain and nothing
            #     whatever on the far end.  This two-axiom stub is what used
            #     to score a perfect 1.00 (V4).  It sits above `floating`
            #     (0.15, nothing anchored at all) and below `path3` (0.60,
            #     where BOTH endpoints are named and a real route between
            #     them was found), which is exactly the evidence it carries.
            if _ask(SG, "ASK { VALUES ?a {%s} VALUES ?p {%s} ?a ?p ?b "
                        "FILTER(?b != <%s>) }" % (VA, VP, SEV_ANY)):
                return "external_uncertified", \
                       SEV_TIERS["external_uncertified"], \
                       "role anchored on domain class (uncertified filler)"
            if _ask(SG, "ASK { VALUES ?a {%s} VALUES ?p {%s} ?a ?p ?b }"
                        % (VA, VP)):
                return "external_open", SEV_TIERS["external_open"], \
                       "domain anchored only; far end undeclared and uncertified"

    if Bc is not None:
        VB = _values(Bc)
        q2 = ("ASK { VALUES ?a {%s} VALUES ?b {%s} VALUES ?pm {%s} "
              "{ ?a ?p1 ?x . ?x ?p2 ?b } UNION { ?x ?p1 ?a . ?x ?p2 ?b } "
              "UNION { ?a ?p1 ?x . ?b ?p2 ?x } "
              "FILTER(?x != <%s> && ?x != <%s> && ?x != ?a && ?x != ?b) %s }")
        # T2 reified: the n-ary-relation idiom — a hub class whose OWN name
        # matches the relation (e.g. :Causes with hasTreatment/hasOutcome).
        # Full credit, no hop penalty: this is how the gold answers its CQs.
        hubs = {c for c in sv.classes
                if c not in sv.universal_hubs
                and _name_matches(idx, c, rel_surfaces)}
        if hubs and VB:
            vh = _values(hubs)
            if vh and _ask(SG, q2 % (VA, VB, VP, SEV_ANY, SEV_LIT,
                                     "VALUES ?x {%s}" % vh)):
                return "reified", SEV_TIERS["reified"], \
                       "n-ary reification via relation-named hub"
        # T3 generic 2-hop with at least one matching property on the path
        if VB and _ask(SG, q2 % (VA, VB, VP, SEV_ANY, SEV_LIT,
                                 "FILTER(?p1 = ?pm || ?p2 = ?pm)")):
            return "indirect2", SEV_TIERS["indirect2"], \
                   "2-hop path via intermediate class"

    # T4 half-anchored: domain declared, range left open (or the mirror image).
    if _ask(SG, "ASK { VALUES ?a {%s} VALUES ?p {%s} ?a ?p <%s> }"
                % (VA, VP, SEV_ANY)):
        return "half", SEV_TIERS["half"], "domain anchored, range undeclared"
    if Bc is not None:
        VB = _values(Bc)
        if VB and _ask(SG, "ASK { VALUES ?p {%s} VALUES ?b {%s} <%s> ?p ?b }"
                           % (VP, VB, SEV_ANY)):
            return "half", SEV_TIERS["half"], "range anchored, domain undeclared"
        # T5 3-hop fully-anchored path
        if VB and _ask(SG, "ASK { VALUES ?a {%s} VALUES ?b {%s} "
                           "?a ?p1 ?x . ?x ?p2 ?y . ?y ?p3 ?b . "
                           "FILTER(?x != <%s> && ?y != <%s> && ?x != <%s> && "
                           "?y != <%s>) }"
                           % (VA, VB, SEV_ANY, SEV_ANY, SEV_LIT, SEV_LIT)):
            return "path3", SEV_TIERS["path3"], "3-hop path"

    # T6 floating: the property exists by name but declares no domain, no range
    # and no restriction.  This is the tier every Llama-2 output lands on.
    if _ask(SG, "ASK { VALUES ?p {%s} <%s> ?p <%s> }" % (VP, SEV_ANY, SEV_ANY)):
        return "floating", SEV_TIERS["floating"], \
               "property declared but unanchored"
    return "none", 0.0, None


# ── per-CQ verdict ───────────────────────────────────────────────────────────

SEV_W_COVERAGE = 0.25
SEV_W_CONNECTIVITY = 0.75
SEV_ANSWERABLE_L1 = 0.75
SEV_ANSWERABLE_L2 = 0.75

# Naming the right things and connecting none of them must not be worth a full
# quarter of the score.  When the best relation on a CQ reaches no higher than
# the "floating" tier (a property that exists by name but declares no domain, no
# range and no restriction), the coverage term contributes only half its weight.
# This is what keeps a pure glossary, an unanchored vocabulary and an adversarial
# name-stuffer inside the "lexical-only" band instead of drifting up into
# "partial".  L1 itself is still reported unmodified: the cap applies to the
# convenience scalar, never to the component vector.
SEV_UNANCHORED_TIER = 0.15
SEV_UNANCHORED_COVERAGE_CAP = 0.5


def _label_for(score: float) -> str:
    if score < 0.05:
        return "unsupported"
    if score < 0.30:
        return "lexical-only"
    if score < 0.70:
        return "partial"
    return "supported"


def score_cq(sv, SG, idx, sig: Dict[str, Any]) -> Dict[str, Any]:
    slots = sig.get("slots", {}) or {}
    binds: Dict[str, Dict[str, Any]] = {}
    for sid in sorted(slots):
        spec = slots[sid]
        u, sc, tier = bind_slot(sv, idx, spec.get("kind", "class"),
                                spec.get("surface", []), spec.get("anchors"))
        binds[sid] = {"uri": str(u) if u is not None else None,
                      "score": round(sc, 3), "match": tier}

    rels = sig.get("relations", []) or []
    rel_out, rel_scores = [], []
    for r in rels:
        a = binds.get(r.get("from"), {}).get("uri")
        A = URIRef(a) if a else None
        target = r.get("to")
        if target == "literal":
            mode, B = "literal", None
        elif target == "external":
            mode, B = "external", None
        else:
            mode = "slot"
            b = binds.get(target, {}).get("uri")
            B = URIRef(b) if b else None
        uncertified = bool(sig.get("derived")
                           or sig.get("signature_source") == "derived")
        tier, credit, why = relation_credit(sv, SG, idx, A, B, mode,
                                            r.get("surface", []), uncertified)
        # gate on binding confidence: an unbound or lexically weak endpoint
        # cannot inherit strong relational credit.
        gate = binds.get(r.get("from"), {}).get("score", 0.0)
        if mode == "slot":
            gate = min(gate, binds.get(target, {}).get("score", 0.0))
        credit *= gate
        rel_out.append({"from": r.get("from"), "to": target, "tier": tier,
                        "credit": round(credit, 3), "why": why})
        rel_scores.append(credit)

    cov_parts = [b["score"] for b in binds.values()]
    for r in rels:
        if r.get("surface"):
            _, sc, _ = bind_slot(sv, idx, "prop", r["surface"])
            cov_parts.append(sc)
    L1 = sum(cov_parts) / len(cov_parts) if cov_parts else 0.0
    L2 = sum(rel_scores) / len(rel_scores) if rel_scores else 0.0
    best_credit = max(rel_scores) if rel_scores else 0.0
    w_cov = SEV_W_COVERAGE
    if best_credit <= SEV_UNANCHORED_TIER + 1e-9:
        w_cov *= SEV_UNANCHORED_COVERAGE_CAP
    s = w_cov * L1 + SEV_W_CONNECTIVITY * L2
    return {
        "cq": sig.get("cq"),
        "cq_type": sig.get("cq_type", "structural"),
        "signature_source": sig.get("signature_source",
                                    "derived" if sig.get("derived") else "authored"),
        "score": round(s, 4),
        "coverage": round(L1, 4),
        "connectivity": round(L2, 4),
        "label": _label_for(s),
        "answerable": bool(L2 >= SEV_ANSWERABLE_L2 and L1 >= SEV_ANSWERABLE_L1),
        "status": "pass" if (L2 >= SEV_ANSWERABLE_L2 and L1 >= SEV_ANSWERABLE_L1) else "fail",
        "bindings": binds,
        "relations": rel_out,
    }


def _zero_cq(sig: Dict[str, Any], status: str) -> Dict[str, Any]:
    return {"cq": sig.get("cq"), "cq_type": sig.get("cq_type", "structural"),
            "signature_source": sig.get("signature_source", "authored"),
            "score": 0.0, "coverage": 0.0, "connectivity": 0.0,
            "label": "unsupported", "answerable": False, "status": status,
            "bindings": {}, "relations": []}


def _parse_for_sev(text: str):
    """(graph, status). status is ok | no_ontology | empty | unparseable."""
    if text is None or not str(text).strip():
        return None, "no_ontology"
    stripped = str(text).strip()
    # F1/F2 left 3-byte "..." files behind; they are worthless, not "skipped".
    if len(stripped) < 20:
        return None, "no_ontology"
    for fmt in ("turtle", "xml", "n3"):
        g = Graph()
        try:
            g.parse(data=text, format=fmt)
        except Exception:
            continue
        if len(g) == 0:
            return g, "empty"
        return g, "ok"
    return None, "unparseable"


def score_ontology(ontology, sigs: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Score one candidate ODP against a list of CQ Requirement Signatures.

    `ontology` may be Turtle/RDF-XML text or an already-parsed rdflib Graph.
    An empty / truncated / unparseable ontology scores 0.0 on every CQ and those
    zeros stay IN the denominator: under SEV the empty file is the metric's worst
    case, not (as under the structural metric, F3) its optimum.
    """
    if isinstance(ontology, Graph):
        g, status = ontology, ("empty" if len(ontology) == 0 else "ok")
    else:
        g, status = _parse_for_sev(ontology)

    if status != "ok":
        per = [_zero_cq(s, status) for s in sigs]
        return _aggregate(per, status, 0, 0, 0)

    sv = SchemaView(g)
    SG = sv.lift()
    idx = sv.index()
    per = [score_cq(sv, SG, idx, s) for s in sigs]
    return _aggregate(per, "ok", len(sv.classes), len(sv.objprops),
                      len(sv.dataprops))


def _mean(values) -> float:
    values = list(values)
    return sum(values) / len(values) if values else 0.0


def _aggregate(per: List[Dict[str, Any]], status: str,
               n_classes: int, n_objprops: int, n_dataprops: int) -> Dict[str, Any]:
    n = len(per)
    authored = [p for p in per if p.get("signature_source") == "authored"]
    structural = [p for p in per if p.get("cq_type") == "structural"]
    tiers: Dict[str, int] = {}
    for p in per:
        for r in p.get("relations", []):
            tiers[r["tier"]] = tiers.get(r["tier"], 0) + 1
    type_counts: Dict[str, int] = {}
    for p in per:
        t = p.get("cq_type", "structural")
        type_counts[t] = type_counts.get(t, 0) + 1
    return {
        "status": status,
        "score": round(_mean(p["score"] for p in per), 4),
        "coverage": round(_mean(p["coverage"] for p in per), 4),
        "connectivity": round(_mean(p["connectivity"] for p in per), 4),
        "answerable_rate": round(
            (sum(1 for p in per if p["answerable"]) / n) if n else 0.0, 4),
        "score_authored": round(_mean(p["score"] for p in authored), 4)
        if authored else None,
        "score_structural": round(_mean(p["score"] for p in structural), 4)
        if structural else None,
        "n_cqs": n,
        "n_authored": len(authored),
        "n_derived": sum(1 for p in per if p.get("signature_source") == "derived"),
        "cq_type_counts": type_counts,
        "tier_histogram": dict(sorted(tiers.items())),
        "n_classes": n_classes,
        "n_objprops": n_objprops,
        "n_dataprops": n_dataprops,
        "per_cq": per,
    }


# ── public entry point (replaces the LLM→SPARQL verifier) ────────────────────

def run_cq_verification(ontology, scenario_id: str,
                        cqs: Optional[List[str]] = None,
                        scenarios_path: Optional[Path] = None,
                        signatures_path: Optional[Path] = None) -> Dict[str, Any]:
    """SEV verification of a candidate ODP against a scenario's real CQs.

    Offline, deterministic, no LLM and no network.

    Result shape keeps the fields the paper's tables read — cqs_total,
    cqs_passed, cqs_failed, pass_rate, results — where "passed" now means the CQ
    is *answerable* (connectivity >= 0.75 AND coverage >= 0.75) rather than "an
    LLM-written instance-level SPARQL happened to return a row".  The graded SEV
    score and its two components are added alongside.
    """
    sigs = signatures_for(scenario_id, cqs, scenarios_path, signatures_path)
    agg = score_ontology(ontology, sigs)
    per = agg["per_cq"]
    total = len(per)                     # ALWAYS len(cq_list): nothing is skipped
    passed = sum(1 for p in per if p["answerable"])
    out: Dict[str, Any] = {
        "method": "SEV",
        "engine_version": SEV_ENGINE_VERSION,
        "signatures_sha": signatures_sha(signatures_path),
        "scenario_id": scenario_id,
        "cq_source": "data/scenarios/pattern_scenarios.json#cq_list",
        "status": agg["status"],
        # backward-compatible headline
        "cqs_total": total,
        "cqs_passed": passed,
        "cqs_failed": total - passed,
        "pass_rate": round(passed / total, 4) if total else 0.0,
        # graded SEV
        "sev_score": agg["score"],
        "sev_coverage": agg["coverage"],
        "sev_connectivity": agg["connectivity"],
        "sev_score_authored": agg["score_authored"],
        "sev_score_structural": agg["score_structural"],
        "cqs_authored": agg["n_authored"],
        "cqs_derived": agg["n_derived"],
        "cq_type_counts": agg["cq_type_counts"],
        "tier_histogram": agg["tier_histogram"],
        "results": per,
    }
    return out


# ── gold-calibration gate ────────────────────────────────────────────────────

_GOLD_EXEMPT = {
    # A-Box instance examples, not T-Boxes: 0 owl:Class and 0 property
    # declarations, so any schema-graph gate scores its own reference at ~0.
    "2023-134-01": "reference is an A-Box ODRL instance example (no T-Box)",
    "2023-134-02": "reference is an A-Box ODRL instance example (no T-Box)",
    "2023-134-03": "reference is an A-Box ODRL instance example (no T-Box)",
    # OWL/XML functional-style syntax: rdflib parses both to 1 triple and errors.
    "2025-147-01": "reference is OWL/XML functional syntax; rdflib cannot parse it",
    "2025-150-01": "reference is OWL/XML functional syntax; rdflib cannot parse it",
}


def find_ground_truth(scenario_id: str,
                      directory: Optional[Path] = None) -> Optional[Path]:
    d = Path(directory) if directory else GROUND_TRUTH_DIR
    for ext in (".ttl", ".owl", ".rdf", ".xml", ".n3"):
        p = d / f"{scenario_id}{ext}"
        if p.exists():
            return p
    return None


def run_gold_calibration(scenario_ids: Optional[List[str]] = None,
                         threshold: float = GOLD_CALIBRATION_THRESHOLD,
                         scenarios_path: Optional[Path] = None,
                         signatures_path: Optional[Path] = None,
                         ground_truth_dir: Optional[Path] = None
                         ) -> Dict[str, Any]:
    """Score every authored signature against its published reference pattern.

    A signature that cannot reach `threshold` on the gold is either a bad
    signature (fix it) or a CQ the reference genuinely does not cover — the
    latter is flagged gold_unsupported, excluded from the primary aggregate, and
    the exclusion count is reported.  This is what turns "is the signature fair?"
    into a falsifiable check.
    """
    sigs_all = load_cq_signatures(signatures_path)
    ids = scenario_ids if scenario_ids is not None else sorted(sigs_all)
    report: Dict[str, Any] = {"threshold": threshold, "scenarios": {},
                              "exempt": {}, "gold_unsupported": [],
                              "passed": 0, "failed": 0,
                              "failed_unexpected": 0}
    for sid in ids:
        if sid in _GOLD_EXEMPT:
            report["exempt"][sid] = _GOLD_EXEMPT[sid]
            continue
        gt = find_ground_truth(sid, ground_truth_dir)
        if gt is None:
            report["exempt"][sid] = "no reference pattern in data/ground_truth"
            continue
        authored = [s for s in signatures_for(sid, None, scenarios_path,
                                              signatures_path)
                    if s.get("signature_source") == "authored"]
        if not authored:
            continue
        text = gt.read_text(encoding="utf-8", errors="replace")
        agg = score_ontology(text, authored)
        if agg["status"] != "ok":
            report["exempt"][sid] = f"reference unparseable ({agg['status']})"
            continue
        entry = {"reference": gt.name, "status": agg["status"], "cqs": []}
        for p in agg["per_cq"]:
            ok = p["score"] >= threshold
            entry["cqs"].append({"cq": p["cq"], "score": p["score"],
                                 "cq_type": p["cq_type"], "gold_pass": ok})
            if ok:
                report["passed"] += 1
            else:
                report["failed"] += 1
                if p["cq_type"] != "out_of_scope":
                    # A CQ the reference cannot answer is only acceptable when
                    # the signature already declares it out of the pattern's
                    # scope; anything else is a signature (or engine) bug.
                    report["failed_unexpected"] += 1
                report["gold_unsupported"].append(
                    {"scenario_id": sid, "cq": p["cq"], "score": p["score"],
                     "cq_type": p["cq_type"]})
        entry["mean_score"] = round(
            _mean(c["score"] for c in entry["cqs"]), 4)
        report["scenarios"][sid] = entry
    return report


# ── 4. OOPS! Pitfall Scanner ──────────────────────────────────────────────────

_OOPS_NS = Namespace("http://oops.linkeddata.es/def#")


def _to_rdfxml(ontology_text: str) -> str:
    for fmt in ("turtle", "xml", "n3", "nt", "json-ld"):
        try:
            g = Graph()
            g.parse(data=ontology_text, format=fmt)
            return g.serialize(format="xml")
        except Exception:
            continue
    raise ValueError("Failed to convert ontology to RDF/XML for OOPS")


def _strip_xml_declaration(text: str) -> str:
    cleaned = text.lstrip("\ufeff").lstrip()
    return re.sub(r"(?is)^<\?xml[^>]*\?>\s*", "", cleaned).lstrip()


def _wrap_cdata(text: str) -> str:
    return text.replace("]]>", "]]]]><![CDATA[>")


def _build_oops_request_xml(ontology_text: str) -> str:
    rdfxml = _strip_xml_declaration(_to_rdfxml(ontology_text)).strip()
    # OntologyURI must always be present — OOPS readXML crashes with
    # StringIndexOutOfBoundsException when the tag is missing (indexOf returns -1,
    # then substring(-1 - offset) throws).
    return "\n".join([
        '<?xml version="1.0" encoding="UTF-8"?>',
        "<OOPSRequest>",
        "  <OntologyURI>urn:local:ontology</OntologyURI>",
        f"  <OntologyContent><![CDATA[{_wrap_cdata(rdfxml)}]]></OntologyContent>",
        "  <Pitfalls></Pitfalls>",
        "  <OutputFormat>RDF/XML</OutputFormat>",
        "</OOPSRequest>",
    ]) + "\n"


_IMPORTANT_OR_CRITICAL = {"Important", "Critical"}


def _parse_oops_response(raw_rdfxml: str) -> Dict[str, Any]:
    summary: Dict[str, Any] = {
        "pitfall_codes": [], "pitfall_instance_counts": {},
        "pitfall_affected_elements_counts": {}, "pitfalls_total": 0, "pitfalls": [],
        "minor_pitfalls_skipped": 0,
    }
    if not raw_rdfxml:
        return summary
    g = Graph()
    g.parse(data=raw_rdfxml, format="xml")
    instance_counts: Dict[str, int] = {}
    affected_counts: Dict[str, int] = {}
    skipped_minor = 0
    for pitfall in sorted(g.subjects(RDF.type, _OOPS_NS.pitfall), key=str):
        code_node = g.value(pitfall, _OOPS_NS.hasCode)
        if not code_node:
            continue
        code = str(code_node).strip()
        importance_node = g.value(pitfall, _OOPS_NS.hasImportanceLevel)
        importance = str(importance_node).strip() if importance_node else None

        # skip Minor pitfalls
        if importance and importance not in _IMPORTANT_OR_CRITICAL:
            skipped_minor += 1
            continue

        name_node = g.value(pitfall, _OOPS_NS.hasName)
        desc_node  = g.value(pitfall, _OOPS_NS.hasDescription)
        reported   = g.value(pitfall, _OOPS_NS.hasNumberAffectedElements)
        affected_elements = [str(o) for o in g.objects(pitfall, _OOPS_NS.hasAffectedElement)]
        affected_count = int(str(reported).strip()) if reported else len(affected_elements)

        pit: Dict[str, Any] = {"code": code, "affected_elements_count": affected_count}
        if importance:
            pit["importance"] = importance
        if name_node:
            pit["name"] = str(name_node).strip()
        if desc_node:
            pit["description"] = str(desc_node).strip()
        if affected_elements:
            pit["affected_elements_sample"] = affected_elements[:50]

        summary["pitfalls"].append(pit)
        instance_counts[code] = instance_counts.get(code, 0) + 1
        affected_counts[code] = affected_counts.get(code, 0) + affected_count

    summary["pitfalls_total"] = len(summary["pitfalls"])
    summary["important_count"] = sum(1 for p in summary["pitfalls"] if p.get("importance") == "Important")
    summary["critical_count"] = sum(1 for p in summary["pitfalls"] if p.get("importance") == "Critical")
    summary["minor_pitfalls_skipped"] = skipped_minor
    summary["pitfall_codes"] = sorted(instance_counts)
    summary["pitfall_instance_counts"] = dict(sorted(instance_counts.items()))
    summary["pitfall_affected_elements_counts"] = dict(sorted(affected_counts.items()))
    return summary


def run_oops_scan(ontology_text: str, oops_url: str, timeout: float = 120) -> Dict[str, Any]:
    if not oops_url:
        return {"skipped": True, "reason": "pass --oops-url to enable"}
    try:
        body = _build_oops_request_xml(ontology_text)
    except Exception as e:
        return {"error": f"RDF/XML conversion failed: {e}"}
    last_status, last_text = 0, ""
    for content_type in ["text/xml; charset=utf-8", "application/xml"]:
        try:
            r = requests.post(oops_url, data=body.encode("utf-8"),
                              headers={"Content-Type": content_type}, timeout=timeout)
            last_status, last_text = r.status_code, r.text
            low = r.text.lower()
            if r.status_code < 400 and "wrong_execution" not in low and "unexpected_error" not in low:
                result: Dict[str, Any] = {"status_code": r.status_code}
                try:
                    result.update(_parse_oops_response(r.text))
                except Exception as e:
                    result["parse_error"] = str(e)
                    result["raw_response"] = r.text
                return result
        except requests.RequestException as e:
            return {"error": str(e)}
    return {"error": f"OOPS request failed (status {last_status})", "raw_response": last_text[:500]}


# ── Per-ID evaluation ──────────────────────────────────────────────────────────

def evaluate_one(onto_path: str, cq_model: Optional[str] = None,
                 openai_key: Optional[str] = None,
                 oops_url: Optional[str] = None,
                 run_cqs: bool = True,
                 local: bool = False) -> Tuple[str, Dict[str, Any]]:
    """Evaluate one <model>/<config>/<scenario_id>/ontology.ttl.

    `local=True` reads the artefact and its sidecars off disk (D1); the default
    keeps the original GitHub behaviour.

    `cq_model` and `openai_key` are accepted only so existing invocations keep
    working; CQ verification is offline now and ignores both.
    """
    model, config, id_ = split_artifact_path(onto_path)

    result: Dict[str, Any] = {
        "id": id_, "model": model, "config": config,
        "source": "local" if local else "github",
        "rel_path": f"{model}/{config}/{id_}/{ONTOLOGY_FILENAME}",
    }

    # The CQs are a property of the SCENARIO, not of the generated file, so they
    # are loaded (and audited) even when the ontology is missing or unparseable.
    try:
        cqs = load_cq_list(id_)
    except KeyError as e:
        cqs = []
        result["cq_error"] = str(e)
    result["cqs"]        = cqs
    result["cqs_count"]  = len(cqs)
    result["cq_source"]  = "data/scenarios/pattern_scenarios.json#cq_list"

    def _cq(text) -> Dict[str, Any]:
        """CQ verification, including for empty/unparseable ontologies.

        An empty or 3-byte ontology.ttl scores 0.0 on every CQ and those zeros
        stay in the denominator: it is the metric's worst case, never a skip.
        """
        if not run_cqs:
            return {"skipped": True, "reason": "--no-cq"}
        if not cqs:
            return {"skipped": True,
                    "reason": f"scenario {id_} has no cq_list in pattern_scenarios.json"}
        try:
            return run_cq_verification(text, id_)
        except (CQContaminationError, SignatureError):
            # A contaminated CQ or a stale signature invalidates the whole run:
            # never swallow it into a per-file "error" field.
            raise
        except Exception as e:                       # pragma: no cover - defensive
            return {"error": f"SEV failed: {e}"}

    # fetch ontology — from disk in local mode, over HTTP otherwise
    onto_text = read_local_text(onto_path) if local else fetch_text(onto_path)

    # D3: the truncation verdict is attached to EVERY record, including the
    # missing-file and unparseable ones, before any early return below.  A
    # generation that ran out of output budget must never look complete.
    sibling_reader = (_local_sibling_reader(onto_path) if local
                      else _remote_sibling_reader(onto_path))
    truncation = read_truncation(onto_path, onto_text, sibling_reader)
    result["truncation"] = truncation
    result["truncated"] = truncation["truncated"]

    if onto_text is None:
        result["error"] = "ontology.ttl not found"
        result["cq_verification"] = _cq(None)
        # F3: a missing file lands on the structural FLOOR, in the denominator.
        result["structural"] = compute_structural_score(
            None, None,
            structural_content_gate(None, parse_error="ontology.ttl not found"))
        return onto_path, result

    # parse once, reuse graph
    try:
        g, _ = _parse_graph(onto_text)
    except ValueError as e:
        result["ontometrics"]     = {"error": str(e)}
        result["reasoner"]        = {"error": str(e)}
        result["cq_verification"] = _cq(onto_text)
        result["oops"]            = {"error": str(e)}
        # F3: unparseable scores 0.0 — never "skipped", never excluded.
        result["structural"] = compute_structural_score(
            None, None, structural_content_gate(None, parse_error=str(e)))
        return onto_path, result

    # 1. ontometrics
    result["ontometrics"] = compute_ontometrics(g)

    # 2. reasoner
    result["reasoner"] = run_reasoner(g)

    # 3. CQ verification (SEV) — offline, deterministic
    result["cq_verification"] = _cq(onto_text)

    # 4. OOPS
    result["oops"] = run_oops_scan(onto_text, oops_url or "")

    # 5. structural score, under the F3 content gate (see structural_content_gate)
    result["structural"] = compute_structural_score(
        result["reasoner"], result["oops"], structural_content_gate(g))

    return onto_path, result


# ── Records and aggregates (D3: truncation is carried into both) ─────────────

SUMMARY_COLUMNS = [
    "model", "config", "id", "rel_path", "source", "record_origin", "status",
    "truncated", "truncation_source", "truncation_signals", "finish_reason",
    "triples_count", "classes_count", "object_properties_count",
    "datatype_properties_count", "consistent", "inferred_triples",
    "cqs_total", "cqs_passed", "sev_score",
    "structural_score", "structural_gate", "oops_pitfalls_total",
]

AGGREGATE_COLUMNS = [
    "model", "config", "n_files",
    # Truncation, surfaced beside every mean it could distort.  n_files is the
    # number of artefacts on disk and nothing is filtered out of it.
    "n_truncated", "n_complete", "n_truncation_unknown",
    "n_parse_errors",
    "mean_structural_score", "structural_denominator",
    "mean_structural_score_complete", "structural_denominator_complete",
    "mean_sev_score", "sev_denominator",
    "mean_sev_score_complete", "sev_denominator_complete",
]

NULL_TOKEN = "null"


def _cell(value: Any) -> Any:
    return NULL_TOKEN if value is None else value


def record_status(result: Dict[str, Any]) -> str:
    """ok | parse_error | missing | error — never a silent blank."""
    if result.get("error") == "ontology.ttl not found":
        return "missing"
    if "error" in result:
        return "error"
    if "error" in (result.get("ontometrics") or {}):
        return "parse_error"
    return "ok"


def summary_row(result: Dict[str, Any]) -> Dict[str, Any]:
    om = result.get("ontometrics") or {}
    rsn = result.get("reasoner") or {}
    cqv = result.get("cq_verification") or {}
    st = result.get("structural") or {}
    tr = result.get("truncation") or {}
    ok = "error" not in om
    return {
        "model": result.get("model"),
        "config": result.get("config"),
        "id": result.get("id"),
        "rel_path": result.get("rel_path"),
        "source": result.get("source"),
        "record_origin": result.get("record_origin", "this_run"),
        "status": record_status(result),
        "truncated": _cell(result.get("truncated")),
        "truncation_source": _cell(tr.get("source")),
        "truncation_signals": "; ".join(tr.get("signals") or []),
        "finish_reason": _cell(tr.get("finish_reason")),
        "triples_count": _cell(om.get("triples_count") if ok else None),
        "classes_count": _cell(om.get("classes_count") if ok else None),
        "object_properties_count": _cell(om.get("object_properties_count") if ok else None),
        "datatype_properties_count": _cell(om.get("datatype_properties_count") if ok else None),
        "consistent": _cell(rsn.get("consistent") if "error" not in rsn else None),
        "inferred_triples": _cell(rsn.get("inferred_triples") if "error" not in rsn else None),
        "cqs_total": _cell(cqv.get("cqs_total")),
        "cqs_passed": _cell(cqv.get("cqs_passed")),
        "sev_score": _cell(cqv.get("sev_score")),
        "structural_score": _cell(st.get("structural_score")),
        "structural_gate": _cell(st.get("gate")),
        "oops_pitfalls_total": _cell((result.get("oops") or {}).get("pitfalls_total")),
    }


def _mean_or_null(values: List[float]) -> Any:
    return round(sum(values) / len(values), 4) if values else NULL_TOKEN


def aggregate_rows(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """One row per (model, config), with truncation surfaced not subtracted.

    Every mean is written twice: over the whole group, and over the artefacts
    known to be complete — each beside its own denominator.  A reader can then
    see exactly how much of a headline number is generations that were cut off
    mid-axiom, WITHOUT any artefact being quietly removed from n_files.
    """
    groups: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    for r in results:
        groups.setdefault((r.get("model", "?"), r.get("config", "?")), []).append(r)

    rows: List[Dict[str, Any]] = []
    for key in sorted(groups):
        group = groups[key]
        complete = [r for r in group if r.get("truncated") is False]

        def structural(rs):
            return [float(r["structural"]["structural_score"]) for r in rs
                    if isinstance(r.get("structural"), dict)
                    and r["structural"].get("structural_score") is not None]

        def sev(rs):
            out = []
            for r in rs:
                cqv = r.get("cq_verification") or {}
                if cqv.get("skipped") or "error" in cqv:
                    continue
                if cqv.get("sev_score") is not None:
                    out.append(float(cqv["sev_score"]))
            return out

        st_all, st_ok = structural(group), structural(complete)
        sev_all, sev_ok = sev(group), sev(complete)
        rows.append({
            "model": key[0],
            "config": key[1],
            "n_files": len(group),
            "n_truncated": sum(1 for r in group if r.get("truncated") is True),
            "n_complete": len(complete),
            "n_truncation_unknown": sum(1 for r in group
                                        if r.get("truncated") is None),
            "n_parse_errors": sum(1 for r in group
                                  if record_status(r) in ("parse_error", "error",
                                                          "missing")),
            "mean_structural_score": _mean_or_null(st_all),
            "structural_denominator": len(st_all),
            "mean_structural_score_complete": _mean_or_null(st_ok),
            "structural_denominator_complete": len(st_ok),
            "mean_sev_score": _mean_or_null(sev_all),
            "sev_denominator": len(sev_all),
            "mean_sev_score_complete": _mean_or_null(sev_ok),
            "sev_denominator_complete": len(sev_ok),
        })
    return rows


def record_key(result: Dict[str, Any]) -> Tuple[str, str, str]:
    """The identity of one artefact: (model, config, scenario id)."""
    return (str(result.get("model")), str(result.get("config")),
            str(result.get("id")))


def load_existing_records(out_dir: Path,
                          skip: Optional[set] = None,
                          ) -> Tuple[List[Dict[str, Any]], List[str]]:
    """Per-artefact records already scored into out_dir, minus `skip`.

    D4.  A repair pass (`--rerun-failed`, `--patch-oops`) re-scores only the
    artefacts that needed it, but summary.csv, aggregate.csv and
    run_manifest.json describe the CORPUS.  Writing them from the pass's own
    handful of results silently replaced the full-corpus CSVs — and a no-op
    pass, with nothing left to repair, cut them down to a bare header and a
    manifest claiming a corpus of zero files, while every per-artefact JSON sat
    untouched on disk.

    So the rows a pass did not touch are read back and carried, each labelled
    `record_origin: "carried_over"` so no reader mistakes a stale score for a
    fresh one.  A record that cannot be read is NOT skipped quietly: its path is
    returned and the manifest reports it.
    """
    skip = skip or set()
    carried: List[Dict[str, Any]] = []
    unreadable: List[str] = []
    if not out_dir.is_dir():
        return carried, unreadable
    for path in sorted(out_dir.glob("*/*/*.json")):
        if path.name == "run_manifest.json":
            continue
        try:
            rec = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            unreadable.append(str(path))
            continue
        if not isinstance(rec, dict):
            unreadable.append(str(path))
            continue
        rec.setdefault("model", path.parent.parent.name)
        rec.setdefault("config", path.parent.name)
        rec.setdefault("id", path.stem)
        if record_key(rec) in skip:
            continue
        rec["record_origin"] = "carried_over"
        carried.append(rec)
    return carried, unreadable


def write_csv(path: Path, columns: List[str], rows: List[Dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_run_artifacts(out_dir: Path, results: List[Dict[str, Any]],
                        *, source: str, corpus_root: Optional[str],
                        network_calls: Optional[int],
                        run_cqs: bool,
                        network_allowed_calls: Optional[int] = None,
                        network_allowlist: Optional[List[str]] = None,
                        n_evaluated_this_run: Optional[int] = None,
                        n_carried_over: int = 0,
                        unreadable_records: Optional[List[str]] = None,
                        ) -> Dict[str, Any]:
    """summary.csv + aggregate.csv + run_manifest.json for one run.

    `results` is the WHOLE scored corpus in out_dir — what this pass scored plus
    what earlier passes left behind (see load_existing_records).  n_files is
    therefore the corpus, and `n_evaluated_this_run` / `n_carried_over` say how
    much of it this pass actually touched.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(out_dir / "summary.csv", SUMMARY_COLUMNS,
              [summary_row(r) for r in results])
    agg = aggregate_rows(results)
    write_csv(out_dir / "aggregate.csv", AGGREGATE_COLUMNS, agg)

    statuses: Dict[str, int] = {}
    for r in results:
        s = record_status(r)
        statuses[s] = statuses.get(s, 0) + 1
    trunc_sources: Dict[str, int] = {}
    for r in results:
        s = (r.get("truncation") or {}).get("source", "unavailable")
        trunc_sources[s] = trunc_sources.get(s, 0) + 1

    manifest = {
        "source": source,
        "corpus_root": corpus_root,
        "out_dir": str(out_dir.resolve()),
        # D4: the corpus, not just this pass.
        "n_files": len(results),
        "n_evaluated_this_run": (len(results) if n_evaluated_this_run is None
                                 else n_evaluated_this_run),
        "n_carried_over": n_carried_over,
        "n_unreadable_records": len(unreadable_records or []),
        "unreadable_records": list(unreadable_records or []),
        "status_counts": statuses,
        "cq_verification": "SEV" if run_cqs else "off (--no-cq)",
        # D3.  These three always sum to n_files: a truncated artefact is
        # reported, never removed from the corpus it belongs to.
        "n_truncated": sum(1 for r in results if r.get("truncated") is True),
        "n_complete": sum(1 for r in results if r.get("truncated") is False),
        "n_truncation_unknown": sum(1 for r in results
                                    if r.get("truncated") is None),
        "truncation_sources": trunc_sources,
        "truncation_note": (
            "truncated artefacts stay in every denominator; each aggregate also "
            "reports a complete-only mean beside its own denominator"
        ),
        # D5: refused (nobody asked for it) and allowed (the run was given the
        # endpoint on the command line) are counted separately, never merged.
        "network_calls": network_calls,
        "network_allowed_calls": network_allowed_calls,
        "network_allowlist": list(network_allowlist or []),
        "aggregate": agg,
    }
    (out_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return manifest


# ── Main ──────────────────────────────────────────────────────────────────────

def _make_console_lossy() -> None:
    """Never let the terminal's codepage abort a scoring run.

    The progress line carries U+2713 for "reasoner says consistent".  On a
    console that cannot represent it — cp1252, the Windows default — the print
    raises UnicodeEncodeError partway through the corpus, after artefacts have
    been scored and before any CSV is written.  Degrading one glyph is always
    the right trade against losing the run.
    """
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors="replace")     # type: ignore[union-attr]
        except (AttributeError, ValueError, OSError):
            pass


def main():
    _make_console_lossy()
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cq", metavar="MODEL", nargs="?", const="", default=None,
                        help="DEPRECATED and ignored. CQ verification (SEV) is offline "
                             "and always on; use --no-cq to skip it.")
    parser.add_argument("--no-cq", action="store_true",
                        help="Skip CQ verification (SEV)")
    parser.add_argument("--openai-key", default=None,
                        help="DEPRECATED and ignored. No API key is used anywhere.")
    parser.add_argument("--gold-calibration", action="store_true",
                        help="Run the SEV validity gate (score every authored CQ "
                             "signature against its reference pattern in "
                             "data/ground_truth/) and exit")
    parser.add_argument("--rerun-failed", action="store_true",
                        help="Only re-evaluate IDs whose existing JSON has a parse error")
    parser.add_argument("--oops-url", metavar="URL", default=None,
                        help="OOPS! REST endpoint (e.g. http://localhost:8080/OOPS/rest)")
    parser.add_argument("--patch-oops", action="store_true",
                        help="Only (re-)run OOPS on existing JSONs that are missing it; skip all other metrics")
    parser.add_argument("--workers", type=int, default=None,
                        help="Override number of parallel workers")
    parser.add_argument("--local-root", metavar="PATH", default=None,
                        help="Evaluate the LOCAL working tree: walk "
                             "PATH/**/ontology.ttl from disk instead of "
                             "enumerating the corpus over the GitHub API. This "
                             "is the documented workflow (see README.md); "
                             "recovered and regenerated outputs can only be "
                             "scored this way. Offline: a network guard is "
                             "armed and the refused-call count is reported.")
    parser.add_argument("--out", metavar="DIR", default=None,
                        help=f"Where to write results (default: {OUT_DIR})")
    args = parser.parse_args()

    if args.cq is not None or args.openai_key:
        print("NOTE: --cq / --openai-key are deprecated and ignored. CQ verification "
              "is now SEV: offline, deterministic, no LLM and no API key.", flush=True)

    if args.gold_calibration:
        report = run_gold_calibration()
        print(json.dumps(report, indent=2, ensure_ascii=False))
        print(f"\nGold calibration: {report['passed']} pass / {report['failed']} "
              f"below threshold {report['threshold']} "
              f"({report['failed_unexpected']} not declared out_of_scope); "
              f"{len(report['exempt'])} scenario(s) exempt.", flush=True)
        sys.exit(0 if report["failed_unexpected"] == 0 else 1)

    run_cqs = not args.no_cq
    n_workers = args.workers or MAX_WORKERS
    out_dir = Path(args.out) if args.out else OUT_DIR

    local = args.local_root is not None
    if local:
        local_root = Path(args.local_root)
        print(f"Walking local tree {local_root} ...", flush=True)
        onto_paths = list_local_ontology_paths(local_root)
    else:
        print("Fetching repo tree ...", flush=True)
        onto_paths = list_ontology_paths()
    print(f"Found {len(onto_paths)} ontology.ttl files", flush=True)

    if args.rerun_failed:
        def is_failed(p: str) -> bool:
            m, c, i = split_artifact_path(p)
            f = out_dir / m / c / f"{i}.json"
            if not f.exists():
                return True
            try:
                d = json.loads(f.read_text())
                return "error" in d.get("ontometrics", {}) or "error" in d
            except Exception:
                return True
        onto_paths = [p for p in onto_paths if is_failed(p)]
        print(f"Re-running {len(onto_paths)} failed/missing IDs")

    if args.patch_oops:
        if not args.oops_url:
            print("ERROR: --patch-oops requires --oops-url.")
            sys.exit(1)

        def needs_oops(p: str) -> bool:
            m, c, i = split_artifact_path(p)
            f = out_dir / m / c / f"{i}.json"
            if not f.exists():
                return False  # no existing result to patch
            try:
                d = json.loads(f.read_text())
                oops = d.get("oops", {})
                return oops.get("skipped", False) or "error" in oops or not oops
            except Exception:
                return False
        onto_paths = [p for p in onto_paths if needs_oops(p)]
        print(f"Patching OOPS on {len(onto_paths)} existing JSONs")

        def patch_one(onto_path: str) -> Tuple[str, Optional[str]]:
            onto_text = read_local_text(onto_path) if local else fetch_text(onto_path)
            if onto_text is None:
                return onto_path, None
            return onto_path, onto_text

        print(f"Workers: {n_workers}  OOPS url: {args.oops_url}\n", flush=True)
        done = 0
        total = len(onto_paths)
        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            futures = {pool.submit(patch_one, p): p for p in onto_paths}
            for future in as_completed(futures):
                onto_path = futures[future]
                model, config, id_ = split_artifact_path(onto_path)
                json_path = out_dir / model / config / f"{id_}.json"
                try:
                    _, onto_text = future.result()
                    if onto_text is None:
                        oops_result = {"error": "ontology.ttl not found"}
                    else:
                        oops_result = run_oops_scan(onto_text, args.oops_url)
                except Exception as e:
                    oops_result = {"error": str(e)}
                try:
                    d = json.loads(json_path.read_text())
                    d["oops"] = oops_result
                    json_path.write_text(json.dumps(d, indent=2))
                except Exception as e:
                    print(f"  [WARN] could not patch {json_path}: {e}", flush=True)
                done += 1
                imp  = oops_result.get("important_count", "?")
                crit = oops_result.get("critical_count", "?")
                tag  = f"oops=I:{imp}/C:{crit}" if not oops_result.get("error") and not oops_result.get("skipped") else str(oops_result.get("error") or "skipped")
                print(f"[{done:>3}/{total}] {model}/{config}/{id_}  {tag}", flush=True)
        # D4: the JSONs just changed, so the CSVs built from them are stale.
        # Rebuild both from everything in out_dir rather than leaving a
        # summary.csv that disagrees with the records beside it.
        patched, unreadable = load_existing_records(out_dir)
        for rec in patched:
            rec["record_origin"] = "carried_over"
        write_run_artifacts(
            out_dir, patched,
            source="local filesystem" if local else f"github:{REPO}@{BRANCH}",
            corpus_root=(str(Path(args.local_root).resolve()) if local else None),
            network_calls=None, run_cqs=run_cqs,
            n_evaluated_this_run=0, n_carried_over=len(patched),
            unreadable_records=unreadable)
        print(f"\nDone. {done} JSONs patched with OOPS results.")
        print(f"Rebuilt {out_dir / 'summary.csv'} over {len(patched)} records.")
        return

    print(f"Workers: {n_workers}  CQ verification: {'SEV' if run_cqs else 'off'}  "
          f"corpus: {'LOCAL ' + str(args.local_root) if local else 'GitHub ' + REPO}\n",
          flush=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    done = 0
    errors = 0
    total = len(onto_paths)
    results: List[Dict[str, Any]] = []

    # A --local-root run must not touch the network; the guard makes that a
    # measured fact rather than a claim.
    allow_urls = [args.oops_url] if args.oops_url else []
    with network_guard(local, allow=allow_urls) as guard, \
            ThreadPoolExecutor(max_workers=n_workers) as pool:
        futures = {
            pool.submit(evaluate_one, p, None, None, args.oops_url, run_cqs, local): p
            for p in onto_paths
        }
        for future in as_completed(futures):
            try:
                onto_path, result = future.result()
            except Exception as e:
                onto_path = futures[future]
                m, c, i = split_artifact_path(onto_path)
                result = {"id": i, "model": m, "config": c,
                          "source": "local" if local else "github",
                          "rel_path": f"{m}/{c}/{i}/{ONTOLOGY_FILENAME}",
                          "error": str(e),
                          "truncated": None,
                          "truncation": {"truncated": None, "signals": [],
                                         "source": "unavailable",
                                         "finish_reason": None,
                                         "note": f"evaluation failed: {e}"}}

            model, config, id_ = split_artifact_path(onto_path)
            results.append(result)

            target = out_dir / model / config
            target.mkdir(parents=True, exist_ok=True)
            with open(target / f"{id_}.json", "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2)

            done += 1
            om  = result.get("ontometrics", {})
            rsn = result.get("reasoner", {})
            cqv = result.get("cq_verification", {})
            trunc = result.get("truncated")
            trunc_str = ("  [TRUNCATED]" if trunc is True
                         else "  [truncation UNKNOWN]" if trunc is None else "")

            if "error" in om or "error" in result:
                errors += 1
                tag = "[PARSE ERROR]"
            else:
                nc       = om.get("classes_count", "?")
                nop      = om.get("object_properties_count", "?")
                inf      = rsn.get("inferred_triples", "?")
                ok_str   = "ok" if rsn.get("consistent", True) else "INCONSISTENT"
                cq_str   = ""
                if not cqv.get("skipped") and not cqv.get("error"):
                    cp = cqv.get("cqs_passed", "?")
                    ct = cqv.get("cqs_total", "?")
                    sev = cqv.get("sev_score", "?")
                    cq_str = f" cq={cp}/{ct} sev={sev}"
                oops_str = ""
                oops = result.get("oops", {})
                if not oops.get("skipped") and not oops.get("error"):
                    imp = oops.get("important_count", 0)
                    crit = oops.get("critical_count", 0)
                    oops_str = f" oops=I:{imp}/C:{crit}"
                tag = f"cls={nc} op={nop} inf={inf} rsn={ok_str}{cq_str}{oops_str}"

            print(f"[{done:>3}/{total}] {model}/{config}/{id_}  {tag}{trunc_str}",
                  flush=True)

        network_calls = guard.n_attempts if local else None
        network_allowed = guard.n_allowed if local else None
        network_allowlist = guard.allowlist if local else []

    # D4: this pass may have scored only part of the corpus (--rerun-failed).
    # The CSVs describe the corpus, so everything it did not touch is read back
    # from out_dir and carried, labelled as carried over.
    for r in results:
        r["record_origin"] = "this_run"
    carried, unreadable = load_existing_records(
        out_dir, skip={record_key(r) for r in results})
    all_results = results + carried

    manifest = write_run_artifacts(
        out_dir, all_results,
        source="local filesystem" if local else f"github:{REPO}@{BRANCH}",
        corpus_root=str(Path(args.local_root).resolve()) if local else None,
        network_calls=network_calls, run_cqs=run_cqs,
        network_allowed_calls=network_allowed,
        network_allowlist=network_allowlist,
        n_evaluated_this_run=len(results), n_carried_over=len(carried),
        unreadable_records=unreadable)

    print(f"\nDone. {done} evaluated, {errors} parse errors.")
    if carried or unreadable:
        print(f"Corpus in {out_dir}: {manifest['n_files']} artefacts "
              f"({len(results)} re-scored this run, {len(carried)} carried over"
              + (f", {len(unreadable)} unreadable" if unreadable else "") + ")")
    # D3: say the truncation split out loud.  The three numbers sum to n_files —
    # truncated artefacts are reported, never dropped from the corpus.
    print(f"Truncation: {manifest['n_truncated']} truncated, "
          f"{manifest['n_complete']} complete, "
          f"{manifest['n_truncation_unknown']} unknown "
          f"(of {manifest['n_files']}); sources: {manifest['truncation_sources']}")
    if local:
        print(f"Network calls attempted: {manifest['network_calls']}")
    print(f"Results in: {out_dir.resolve()}/")
    print(f"  per-artefact CSV: {(out_dir / 'summary.csv').resolve()}")
    print(f"  aggregate CSV:    {(out_dir / 'aggregate.csv').resolve()}")


if __name__ == "__main__":
    main()
