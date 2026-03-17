#!/usr/bin/env python3
"""
Download ground truth ontologies from the URLs in pattern_scenarios.json.

Usage:
    python scripts/download_ground_truth.py
    python scripts/download_ground_truth.py --force   # re-download existing files
"""
from __future__ import annotations

import argparse
import io
import json
import re
import sys
import zipfile
from pathlib import Path

import xml.etree.ElementTree as ET

import requests
import rdflib
from rdflib import OWL, RDF, RDFS, URIRef, Literal

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_RDF_EXTENSIONS = {".ttl": "turtle", ".owl": "xml", ".rdf": "xml", ".n3": "n3", ".nt": "nt"}
_FORMAT_CONTENT_TYPES = {
    "text/turtle": "turtle",
    "application/rdf+xml": "xml",
    "application/owl+xml": "xml",
    "text/n3": "n3",
}
_PARSE_ORDER = ["turtle", "xml", "n3", "nt"]

# Local fallback for files that live in this repo but may not be pushed yet.
# Maps scenario_id -> relative path from repo root.
_LOCAL_FALLBACKS: dict[str, Path] = {
    "2025-153-01": Path("data/ontologies_retrived/temporalindirection-reconstructed.ttl"),
    "2025-151-01": Path("data/ontologies_retrived/evaluation-request-pattern-v0.1.ttl"),
    "2025-151-02": Path("data/ontologies_retrived/state-of-the-world-pattern-v0.1.ttl"),
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ext_from_url(url: str) -> str:
    """Best-effort file extension from a URL, ignoring query strings."""
    path_part = url.split("?")[0].rstrip("/")
    return Path(path_part).suffix.lower()


def _format_hint_from_url(url: str) -> str:
    ext = _ext_from_url(url)
    if ".zip" in url:
        return "zip"
    return _RDF_EXTENSIONS.get(ext, "turtle")


def _format_hint_from_response(response: requests.Response, url: str) -> str:
    ct = response.headers.get("Content-Type", "")
    for ct_key, fmt in _FORMAT_CONTENT_TYPES.items():
        if ct_key in ct:
            return fmt
    if "html" in ct.lower() or response.text.lstrip().startswith("<!"):
        return "html"
    return _format_hint_from_url(url)


def _try_parse_owlxml(data: bytes) -> rdflib.Graph | None:
    """
    Parse OWL/XML (OWLAPI functional-style XML) into a minimal rdflib Graph.
    OWL/XML uses <Declaration><Class IRI="..."/></Declaration> style,
    which is NOT RDF/XML and cannot be parsed by rdflib natively.
    Returns None if the content is not OWL/XML.
    """
    OWL_NS = "http://www.w3.org/2002/07/owl#"
    try:
        root = ET.fromstring(data)
    except ET.ParseError:
        return None

    # Must be an OWL/XML Ontology element
    if root.tag not in (f"{{{OWL_NS}}}Ontology", "Ontology"):
        return None

    g = rdflib.Graph()
    g.bind("owl", OWL)
    g.bind("rdfs", RDFS)

    type_map = {
        f"{{{OWL_NS}}}Class": OWL.Class,
        f"{{{OWL_NS}}}ObjectProperty": OWL.ObjectProperty,
        f"{{{OWL_NS}}}DataProperty": OWL.DatatypeProperty,
        f"{{{OWL_NS}}}AnnotationProperty": OWL.AnnotationProperty,
        f"{{{OWL_NS}}}NamedIndividual": OWL.NamedIndividual,
    }

    onto_iri = root.get("ontologyIRI") or root.get("{http://www.w3.org/2002/07/owl#}ontologyIRI")
    onto_node = URIRef(onto_iri) if onto_iri else URIRef("http://example.org/ontology")
    g.add((onto_node, RDF.type, OWL.Ontology))

    for decl in root.iter(f"{{{OWL_NS}}}Declaration"):
        for child in decl:
            owl_type = type_map.get(child.tag)
            if owl_type:
                iri = child.get("IRI") or child.get("abbreviatedIRI")
                if iri and iri.startswith("http"):
                    node = URIRef(iri)
                    g.add((node, RDF.type, owl_type))

    # Extract annotation assertions (labels, comments)
    for ann in root.iter(f"{{{OWL_NS}}}AnnotationAssertion"):
        children = list(ann)
        if len(children) >= 3:
            prop_el, subj_el, val_el = children[0], children[1], children[2]
            prop_iri = prop_el.get("IRI") or prop_el.get("abbreviatedIRI", "")
            subj_iri = subj_el.get("IRI") or subj_el.get("abbreviatedIRI", "")
            val_text = val_el.text or ""
            if subj_iri.startswith("http") and val_text:
                pred = URIRef(prop_iri) if prop_iri.startswith("http") else None
                if str(RDFS.label) in prop_iri or "label" in prop_iri.lower():
                    pred = RDFS.label
                elif str(RDFS.comment) in prop_iri or "comment" in prop_iri.lower():
                    pred = RDFS.comment
                if pred:
                    g.add((URIRef(subj_iri), pred, Literal(val_text)))

    return g if len(g) > 0 else None


def _try_parse_rdf(data: bytes, fmt_hint: str, source_label: str = "") -> tuple[rdflib.Graph, str]:
    """
    Try to parse RDF bytes, attempting multiple formats.
    Returns (graph, format_used). Raises ValueError if all fail.
    Falls back to OWL/XML parser for OWLAPI-style files.
    """
    order = [fmt_hint] + [f for f in _PARSE_ORDER if f != fmt_hint]
    last_err = None
    for fmt in order:
        try:
            g = rdflib.Graph()
            g.parse(data=data.decode("utf-8", errors="replace"), format=fmt)
            if len(g) > 0:
                return g, fmt
        except Exception as e:
            last_err = e

    # Fallback: try OWL/XML (OWLAPI functional-style)
    g = _try_parse_owlxml(data)
    if g is not None:
        return g, "owlxml"

    raise ValueError(f"Could not parse {source_label} in any format. Last error: {last_err}")


def _extract_rdf_from_zip(raw: bytes) -> tuple[bytes, str, str]:
    """
    Find the first parseable RDF file in a ZIP archive.
    Returns (rdf_bytes, filename, format_hint).
    Raises ValueError if no RDF file found.
    """
    with zipfile.ZipFile(io.BytesIO(raw)) as zf:
        rdf_entries = [
            name for name in zf.namelist()
            if Path(name).suffix.lower() in _RDF_EXTENSIONS
            and not name.startswith("__MACOSX")
        ]
        if not rdf_entries:
            raise ValueError("No RDF files found in ZIP archive.")
        # Prefer .ttl over .owl over .rdf
        for preferred_ext in [".ttl", ".owl", ".rdf", ".n3", ".nt"]:
            for name in rdf_entries:
                if Path(name).suffix.lower() == preferred_ext:
                    data = zf.read(name)
                    return data, Path(name).name, _RDF_EXTENSIONS[preferred_ext]
        # Fallback: first entry
        name = rdf_entries[0]
        data = zf.read(name)
        return data, Path(name).name, _format_hint_from_url(name)


def _fetch_and_parse(url: str, timeout: int) -> tuple[bytes, str, rdflib.Graph, str]:
    """
    Download url, detect format, parse with rdflib.
    Returns (raw_bytes, ext_to_save, graph, format_used).
    Raises on HTTP error, HTML response, or parse failure.
    """
    resp = requests.get(url, timeout=timeout, headers={"Accept": "text/turtle, application/rdf+xml;q=0.9, */*;q=0.8"})
    resp.raise_for_status()

    fmt_hint = _format_hint_from_response(resp, url)
    if fmt_hint == "html":
        raise ValueError(f"Server returned HTML instead of RDF for {url}")

    raw = resp.content

    if fmt_hint == "zip" or ".zip" in url:
        rdf_bytes, filename, fmt_hint = _extract_rdf_from_zip(raw)
        ext = Path(filename).suffix or ".ttl"
        g, fmt_used = _try_parse_rdf(rdf_bytes, fmt_hint, url)
        return rdf_bytes, ext, g, fmt_used

    g, fmt_used = _try_parse_rdf(raw, fmt_hint, url)
    ext = _ext_from_url(url) or (".ttl" if fmt_used == "turtle" else ".owl")
    return raw, ext, g, fmt_used


# ---------------------------------------------------------------------------
# Main downloader
# ---------------------------------------------------------------------------

def download_ground_truth(
    scenarios: list[dict],
    output_dir: Path,
    timeout: int = 30,
    force: bool = False,
) -> dict[str, Path | None]:
    """
    Download ground truth ontologies for all scenarios.
    Returns scenario_id -> local Path (or None on failure).
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    results: dict[str, Path | None] = {}

    # Track already-fetched URLs to avoid duplicate HTTP requests
    url_cache: dict[str, tuple[bytes, str, str]] = {}  # url -> (raw, ext, fmt)

    for scenario in scenarios:
        sid = scenario["scenario_id"]
        url = scenario.get("ontology", "").strip()
        status_path = output_dir / f"{sid}.status.json"

        if not url:
            _write_status(status_path, url, "skipped", "No ontology URL in scenario.", None, None, None)
            results[sid] = None
            continue

        # Check existing file
        existing = _find_existing(output_dir, sid)
        if existing and not force:
            results[sid] = existing
            print(f"  [{sid}] already downloaded → {existing}")
            continue

        print(f"  [{sid}] downloading {url} ...", end=" ", flush=True)

        # After failed HTTP — try local fallback before giving up
        def _try_local_fallback(sid: str, status_path: Path) -> Path | None:
            local = _LOCAL_FALLBACKS.get(sid)
            if local and local.exists():
                dest = output_dir / f"{sid}{local.suffix}"
                dest.write_bytes(local.read_bytes())
                try:
                    g2 = rdflib.Graph()
                    g2.parse(str(dest))
                    _write_status(status_path, str(local), "ok", None, str(dest), "local_copy", len(g2))
                    print(f"ok (local copy) [{len(g2)} triples]")
                    return dest
                except Exception:
                    pass
            return None

        # Use cache for duplicate URLs (e.g., two scenarios with the same ontology)
        if url in url_cache:
            raw, ext, fmt_used = url_cache[url]
            try:
                g, _ = _try_parse_rdf(raw, fmt_used, url)
                dest = output_dir / f"{sid}{ext}"
                dest.write_bytes(raw)
                _write_status(status_path, url, "ok", None, str(dest), fmt_used, len(g))
                results[sid] = dest
                print(f"ok (cached) [{len(g)} triples, {fmt_used}]")
            except Exception as e:
                _write_status(status_path, url, "failed", str(e), None, None, None)
                results[sid] = None
                print(f"FAILED (cache re-parse): {e}")
            continue

        try:
            raw, ext, g, fmt_used = _fetch_and_parse(url, timeout)
            dest = output_dir / f"{sid}{ext}"
            dest.write_bytes(raw)
            url_cache[url] = (raw, ext, fmt_used)
            _write_status(status_path, url, "ok", None, str(dest), fmt_used, len(g))
            results[sid] = dest
            print(f"ok [{len(g)} triples, {fmt_used}]")
        except Exception as e:
            fb = _try_local_fallback(sid, status_path)
            if fb:
                results[sid] = fb
            else:
                _write_status(status_path, url, "failed", str(e), None, None, None)
                results[sid] = None
                print(f"FAILED: {e}")

    return results


def _find_existing(directory: Path, sid: str) -> Path | None:
    for ext in _RDF_EXTENSIONS:
        p = directory / f"{sid}{ext}"
        if p.exists():
            return p
    return None


def _write_status(
    path: Path,
    url: str,
    status: str,
    reason: str | None,
    dest: str | None,
    fmt: str | None,
    triple_count: int | None,
) -> None:
    path.write_text(json.dumps({
        "url": url,
        "status": status,
        "reason": reason,
        "path": dest,
        "format": fmt,
        "triple_count": triple_count,
    }, indent=2), encoding="utf-8")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Download ground truth ontologies from pattern_scenarios.json.")
    parser.add_argument("--data", default="data/scenarios/pattern_scenarios.json", type=Path)
    parser.add_argument("--output-dir", default="data/ground_truth", type=Path)
    parser.add_argument("--force", action="store_true", help="Re-download even if file already exists.")
    parser.add_argument("--timeout", type=int, default=30)
    args = parser.parse_args()

    scenarios = json.loads(args.data.read_text(encoding="utf-8"))
    print(f"Downloading ground truth ontologies for {len(scenarios)} scenarios → {args.output_dir}/\n")
    results = download_ground_truth(scenarios, args.output_dir, timeout=args.timeout, force=args.force)

    ok = sum(1 for v in results.values() if v is not None)
    failed = len(results) - ok
    print(f"\nDone: {ok} downloaded, {failed} failed/skipped.")
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
