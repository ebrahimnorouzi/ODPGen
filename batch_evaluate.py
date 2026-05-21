#!/usr/bin/env python3
"""
Batch ontology evaluator for ODPGen outputs.
Runs three evaluations per ontology:
  1. Ontometrics  (structural counts + richness ratios)
  2. OWL Reasoner (OWL-RL via owlrl: consistency + inferred triples)
  3. CQ Verification (LLM→SPARQL→run on ontology, requires --openai-key)

Output: odp_eval/{model}/{config}/{id}.json  (one file per ID)
Usage:
  python3 batch_evaluate.py                          # ontometrics + reasoner only
  python3 batch_evaluate.py --cq gpt-4o-mini        # + CQ verification
  python3 batch_evaluate.py --rerun-failed           # redo only error/parse-failed IDs
"""

import argparse
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests
from rdflib import Graph
from rdflib.namespace import RDF, Namespace
from rdflib.term import Node

# ── Config ────────────────────────────────────────────────────────────────────
GITHUB_TOKEN = ""
REPO         = "ebrahimnorouzi/ODPGen"
BRANCH       = "main"
OUT_DIR      = Path("odp_eval")
MAX_WORKERS  = 8    # parallel HTTP + reasoner tasks
CQ_WORKERS   = 4    # fewer workers when making LLM calls (rate-limit friendly)

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
    url = f"https://api.github.com/repos/{REPO}/git/trees/{BRANCH}?recursive=1"
    r = requests.get(url, headers=GH_HEADERS, timeout=30)
    r.raise_for_status()
    return [t["path"] for t in r.json().get("tree", [])
            if t["path"].endswith("/ontology.ttl")]


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

    result: Dict[str, Any] = {
        "consistent":             consistent,
        "triples_before":         triples_before,
        "triples_after":          triples_after,
        "inferred_triples":       inferred,
        "unsatisfiable_classes":  unsat_classes,
    }
    if inconsistency_reason:
        result["inconsistency_reason"] = inconsistency_reason
    return result


# ── 3. CQ Verification (LLM → SPARQL → run) ──────────────────────────────────

def extract_cqs(prompt_text: str) -> List[str]:
    """Extract bullet CQs from the ODPGen prompt.txt."""
    lines: List[str] = []
    in_section = False
    for line in prompt_text.splitlines():
        if re.search(r"competency questions?", line, re.IGNORECASE):
            in_section = True
            continue
        if in_section:
            stripped = line.strip()
            if stripped.startswith("-"):
                cq = stripped.lstrip("- ").strip()
                if cq:
                    lines.append(cq if cq.endswith("?") else cq + "?")
            elif stripped and not stripped.startswith("#") and lines:
                break
    return lines


def _ontology_summary(g) -> str:
    """Compact class/property summary with prefix:name terms for the LLM."""
    from rdflib.namespace import OWL, RDF, RDFS

    ns_to_prefix: Dict[str, str] = {}
    for p, ns in g.namespaces():
        ns_str = str(ns)
        if p and (ns_str not in ns_to_prefix or len(str(p)) < len(ns_to_prefix[ns_str])):
            ns_to_prefix[ns_str] = str(p)

    def qname(uri: str) -> str:
        for ns_str, pfx in sorted(ns_to_prefix.items(), key=lambda x: -len(x[0])):
            if uri.startswith(ns_str):
                local = uri[len(ns_str):]
                if local:
                    return f"{pfx}:{local}"
        return f"<{uri}>"

    skip_ns = {"http://www.w3.org/2002/07/owl#", "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
                "http://www.w3.org/2000/01/rdf-schema#", "http://www.w3.org/2001/XMLSchema#"}

    def terms(subjects):
        return sorted({qname(str(s)) for s in subjects
                       if not any(str(s).startswith(ns) for ns in skip_ns)})

    classes   = terms(list(g.subjects(RDF.type, OWL.Class)) +
                      list(g.subjects(RDF.type, RDFS.Class)))
    obj_props = terms(g.subjects(RDF.type, OWL.ObjectProperty))
    dat_props = terms(g.subjects(RDF.type, OWL.DatatypeProperty))

    used_pfxs = {t.split(":")[0] for t in classes + obj_props + dat_props
                 if ":" in t and not t.startswith("<")}
    prefix_block = "\n".join(
        f"PREFIX {p}: <{ns}>"
        for ns, p in sorted(ns_to_prefix.items())
        if p in used_pfxs
    )

    return "\n".join([
        "Use these PREFIX declarations verbatim:",
        prefix_block,
        "",
        f"Classes (use exactly as written): {', '.join(classes) or 'none'}",
        f"Object properties:                {', '.join(obj_props) or 'none'}",
        f"Datatype properties:              {', '.join(dat_props) or 'none'}",
    ])


def _cq_to_sparql(cq: str, summary: str, model: str, client) -> str:
    prompt = (
        "You are an ontology engineering expert. Write a SPARQL SELECT query to answer "
        "the given competency question against the described ontology.\n\n"
        f"=== ONTOLOGY ===\n{summary}\n=== END ONTOLOGY ===\n\n"
        f"Competency question: {cq}\n\n"
        "STRICT RULES:\n"
        "1. ONLY use prefix names and URIs listed under 'Prefixes' above. "
        "   Do NOT invent prefixes (no brick:, schema:, etc.).\n"
        "2. ONLY use terms listed under Classes / Object properties / Datatype properties.\n"
        "3. Output ONLY the raw SPARQL, no explanation, no markdown fences.\n"
        "4. Use SELECT … WHERE { … } LIMIT 10.\n"
        "5. Use OPTIONAL for properties that may not exist on every instance.\n"
        "6. Do NOT add a trailing dot after LIMIT N."
    )
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "Output only a valid SPARQL query."},
            {"role": "user",   "content": prompt},
        ],
        max_tokens=400, temperature=0,
    )
    raw = resp.choices[0].message.content.strip()
    raw = re.sub(r"^```[a-z]*\n?", "", raw, flags=re.IGNORECASE)
    raw = re.sub(r"\n?```$", "", raw)
    raw = re.sub(r"\bLIMIT\s+(\d+)\s*\.\s*$", r"LIMIT \1", raw.strip())
    return raw.strip()


def _prepend_prefixes(sparql: str, g) -> str:
    """Auto-inject missing PREFIX declarations from the ontology's namespace map."""
    ns_map = {str(p): str(ns) for p, ns in g.namespaces() if p}
    declared = set(re.findall(r"PREFIX\s+(\w+)\s*:", sparql, re.IGNORECASE))
    used = set(re.findall(r"\b(\w+):[A-Za-z_]", sparql))
    missing = used - declared - {"http", "https", "ftp"}
    extra = "".join(f"PREFIX {p}: <{ns_map[p]}>\n" for p in sorted(missing) if p in ns_map)
    return extra + sparql


def _suggest_fix(cq: str, sparql: str, error: Optional[str], model: str, client) -> str:
    reason = f"Query error: {error}" if error else "Query returned no results."
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are an ontology engineering expert."},
            {"role": "user",   "content": (
                f"Competency question: {cq}\n\n"
                f"SPARQL attempted:\n{sparql}\n\n"
                f"Result: {reason}\n\n"
                "Suggest concretely what classes, properties, or axioms should be added "
                "to the ontology to answer this CQ. Be specific and brief (3-5 bullet points max)."
            )},
        ],
        max_tokens=300, temperature=0,
    )
    return resp.choices[0].message.content.strip()


def run_cq_verification(g, cqs: List[str], model: str, openai_key: str) -> Dict[str, Any]:
    if not cqs:
        return {"skipped": True, "reason": "no CQs available for this config"}

    import openai
    client = openai.OpenAI(api_key=openai_key)
    summary = _ontology_summary(g)

    results = []
    passed = 0

    for cq in cqs:
        entry: Dict[str, Any] = {"cq": cq}
        try:
            sparql = _cq_to_sparql(cq, summary, model, client)
            sparql_fixed = _prepend_prefixes(sparql, g)
            entry["sparql"] = sparql
        except Exception as e:
            entry["status"] = "error"
            entry["error"] = f"SPARQL generation failed: {e}"
            results.append(entry)
            continue

        # run query
        try:
            rows = list(g.query(sparql_fixed))
            exec_error = None
        except Exception as e:
            rows = []
            exec_error = str(e)

        if exec_error is None and rows:
            entry["status"] = "pass"
            entry["result_count"] = len(rows)
            passed += 1
        else:
            entry["status"] = "fail"
            entry["result_count"] = 0
            if exec_error:
                entry["sparql_error"] = exec_error
            try:
                entry["suggestion"] = _suggest_fix(cq, sparql, exec_error, model, client)
            except Exception as e:
                entry["suggestion_error"] = str(e)

        results.append(entry)

    total = len([r for r in results if r.get("status") != "skipped"])
    return {
        "cqs_total":  len(cqs),
        "cqs_passed": passed,
        "cqs_failed": total - passed,
        "pass_rate":  round(passed / total, 4) if total else 0.0,
        "results":    results,
    }


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

def evaluate_one(onto_path: str, cq_model: Optional[str],
                 openai_key: Optional[str],
                 oops_url: Optional[str] = None) -> Tuple[str, Dict[str, Any]]:
    parts    = onto_path.split("/")
    model, config, id_ = parts[1], parts[2], parts[3]

    result: Dict[str, Any] = {"id": id_, "model": model, "config": config}

    # fetch ontology
    onto_text = fetch_text(onto_path)
    if onto_text is None:
        result["error"] = "ontology.ttl not found"
        return onto_path, result

    # parse once, reuse graph
    try:
        g, _ = _parse_graph(onto_text)
    except ValueError as e:
        result["ontometrics"]    = {"error": str(e)}
        result["reasoner"]       = {"error": str(e)}
        result["cq_verification"] = {"error": str(e)}
        return onto_path, result

    # 1. ontometrics
    result["ontometrics"] = compute_ontometrics(g)

    # 2. reasoner
    result["reasoner"] = run_reasoner(g)

    # 3. CQs: always extract; verify only if --cq given
    prompt_path  = onto_path.replace("ontology.ttl", "prompt.txt")
    prompt_text  = fetch_text(prompt_path)
    cqs          = extract_cqs(prompt_text) if prompt_text else []
    result["cqs"]       = cqs
    result["cqs_count"] = len(cqs)

    if cq_model and openai_key:
        result["cq_verification"] = run_cq_verification(g, cqs, cq_model, openai_key)
    else:
        result["cq_verification"] = {
            "skipped": True,
            "reason": "pass --cq MODEL to enable"
        }

    # 4. OOPS
    result["oops"] = run_oops_scan(onto_text, oops_url or "")

    return onto_path, result


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cq", metavar="MODEL", default=None,
                        help="Enable CQ verification with this OpenAI model (e.g. gpt-4o-mini)")
    parser.add_argument("--openai-key", default=None,
                        help="OpenAI API key (defaults to OPENAI_API_KEY env var)")
    parser.add_argument("--rerun-failed", action="store_true",
                        help="Only re-evaluate IDs whose existing JSON has a parse error")
    parser.add_argument("--oops-url", metavar="URL", default=None,
                        help="OOPS! REST endpoint (e.g. http://localhost:8080/OOPS/rest)")
    parser.add_argument("--patch-oops", action="store_true",
                        help="Only (re-)run OOPS on existing JSONs that are missing it; skip all other metrics")
    parser.add_argument("--workers", type=int, default=None,
                        help="Override number of parallel workers")
    args = parser.parse_args()

    openai_key = args.openai_key or os.environ.get("OPENAI_API_KEY")
    if args.cq and not openai_key:
        # try to read from local .env files
        for env_file in [Path.home() / "text2sparql/.env",
                         Path.home() / "work-with-sciments/.env",
                         Path.home() / "GenCQs/.env"]:
            if env_file.exists():
                for line in env_file.read_text().splitlines():
                    if line.startswith("OPENAI_API_KEY"):
                        openai_key = line.split("=", 1)[1].strip().strip('"').strip("'")
                        break
            if openai_key:
                break
        if not openai_key:
            print("ERROR: --cq requires an OpenAI key. Set OPENAI_API_KEY or pass --openai-key.")
            sys.exit(1)

    n_workers = args.workers or (CQ_WORKERS if args.cq else MAX_WORKERS)

    print("Fetching repo tree …", flush=True)
    onto_paths = list_ontology_paths()
    print(f"Found {len(onto_paths)} ontology.ttl files", flush=True)

    if args.rerun_failed:
        def is_failed(p: str) -> bool:
            parts = p.split("/")
            f = OUT_DIR / parts[1] / parts[2] / f"{parts[3]}.json"
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
            parts = p.split("/")
            f = OUT_DIR / parts[1] / parts[2] / f"{parts[3]}.json"
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
            onto_text = fetch_text(onto_path)
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
                parts = onto_path.split("/")
                model, config, id_ = parts[1], parts[2], parts[3]
                json_path = OUT_DIR / model / config / f"{id_}.json"
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
        print(f"\nDone. {done} JSONs patched with OOPS results.")
        return

    print(f"Workers: {n_workers}  CQ model: {args.cq or 'none'}\n", flush=True)
    OUT_DIR.mkdir(exist_ok=True)

    done = 0
    errors = 0
    total = len(onto_paths)

    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        futures = {
            pool.submit(evaluate_one, p, args.cq, openai_key, args.oops_url): p
            for p in onto_paths
        }
        for future in as_completed(futures):
            try:
                onto_path, result = future.result()
            except Exception as e:
                onto_path = futures[future]
                parts = onto_path.split("/")
                result = {"id": parts[3], "model": parts[1], "config": parts[2],
                          "error": str(e)}

            parts  = onto_path.split("/")
            model, config, id_ = parts[1], parts[2], parts[3]

            out_dir = OUT_DIR / model / config
            out_dir.mkdir(parents=True, exist_ok=True)
            with open(out_dir / f"{id_}.json", "w") as f:
                json.dump(result, f, indent=2)

            done += 1
            om  = result.get("ontometrics", {})
            rsn = result.get("reasoner", {})
            cqv = result.get("cq_verification", {})

            if "error" in om or "error" in result:
                errors += 1
                tag = "[PARSE ERROR]"
            else:
                nc       = om.get("classes_count", "?")
                nop      = om.get("object_properties_count", "?")
                inf      = rsn.get("inferred_triples", "?")
                ok_str   = "✓" if rsn.get("consistent", True) else "✗ INCONSISTENT"
                cq_str   = ""
                if not cqv.get("skipped"):
                    cp = cqv.get("cqs_passed", "?")
                    ct = cqv.get("cqs_total", "?")
                    cq_str = f" cq={cp}/{ct}"
                oops_str = ""
                oops = result.get("oops", {})
                if not oops.get("skipped") and not oops.get("error"):
                    imp = oops.get("important_count", 0)
                    crit = oops.get("critical_count", 0)
                    oops_str = f" oops=I:{imp}/C:{crit}"
                tag = f"cls={nc} op={nop} inf={inf} rsn={ok_str}{cq_str}{oops_str}"

            print(f"[{done:>3}/{total}] {model}/{config}/{id_}  {tag}", flush=True)

    print(f"\nDone. {done} evaluated, {errors} parse errors.")
    print(f"Results in: {OUT_DIR.resolve()}/")


if __name__ == "__main__":
    main()
