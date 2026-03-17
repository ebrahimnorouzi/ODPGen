"""
Structural similarity between a generated OWL ontology and its ground truth.

Compares class and property names using camelCase-normalized token-set Jaccard
matching so that minor naming variations (abbreviations, synonyms) are tolerated.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

import rdflib
from rdflib import OWL, RDF, URIRef

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MATCH_THRESHOLD = 0.5  # Jaccard threshold to count two names as "matching"

_PROPERTY_TYPES = [OWL.ObjectProperty, OWL.DatatypeProperty, OWL.AnnotationProperty]

_STOPWORDS = {
    "a", "an", "the", "of", "to", "in", "on", "for", "with", "is", "are",
    "be", "as", "by", "it", "this", "that", "from", "at", "or", "and",
}


# ---------------------------------------------------------------------------
# Name normalisation
# ---------------------------------------------------------------------------

def split_camel_case(uri: str) -> str:
    """
    Extract the local name from a URI and normalize camelCase to
    space-separated lowercase tokens.

    Examples
    --------
    >>> split_camel_case("http://example.org/odp#hasParticipant")
    'has participant'
    >>> split_camel_case("http://example.org/odp#RoleDependentName")
    'role dependent name'
    >>> split_camel_case("http://example.org/odp#MQTT4SSN")
    'mqtt4ssn'
    """
    # Extract local name after # or last /
    local = uri.split("#")[-1] if "#" in uri else uri.rstrip("/").rsplit("/", 1)[-1]
    # Replace underscores and hyphens with spaces
    local = re.sub(r"[_\-]+", " ", local)
    # Insert space before uppercase following lowercase/digit (e.g. hasParticipant → has Participant)
    local = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", local)
    # Insert space between consecutive caps and following lowercase (e.g. AISHIPOntology → AISHIP Ontology)
    local = re.sub(r"(?<=[A-Z])(?=[A-Z][a-z])", " ", local)
    return local.lower().strip()


def _token_set(name: str) -> frozenset[str]:
    tokens = {t for t in name.split() if len(t) >= 2 and t not in _STOPWORDS}
    return frozenset(tokens)


def _jaccard(a: frozenset, b: frozenset) -> float:
    if not a and not b:
        return 1.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


# ---------------------------------------------------------------------------
# Graph helpers
# ---------------------------------------------------------------------------

def _load_graph(path: Path) -> rdflib.Graph:
    g = rdflib.Graph()
    for fmt in ("turtle", "xml", "n3"):
        try:
            g.parse(str(path), format=fmt)
            if len(g) > 0:
                return g
        except Exception:
            g = rdflib.Graph()
    return g


def _extract_names(g: rdflib.Graph, type_uris: list) -> list[str]:
    """Normalized name strings for all URIRef subjects typed as any of type_uris."""
    seen: set[str] = set()
    names: list[str] = []
    for rdf_type in type_uris:
        for s, _, _ in g.triples((None, RDF.type, rdf_type)):
            if isinstance(s, URIRef):
                norm = split_camel_case(str(s))
                if norm and norm not in seen:
                    seen.add(norm)
                    names.append(norm)
    return names


# ---------------------------------------------------------------------------
# F1 computation
# ---------------------------------------------------------------------------

def _f1_from_name_lists(
    gen_names: list[str],
    gt_names: list[str],
    threshold: float = MATCH_THRESHOLD,
) -> tuple[float, float, float, list[str]]:
    """
    Compute (precision, recall, f1, matched_gen_names) via greedy
    token-set Jaccard matching.

    Each generated name can match at most one ground-truth name (greedy),
    and vice versa.
    """
    if not gen_names and not gt_names:
        return 1.0, 1.0, 1.0, []
    if not gen_names or not gt_names:
        return 0.0, 0.0, 0.0, []

    gen_sets = [(n, _token_set(n)) for n in gen_names]
    gt_sets = [_token_set(n) for n in gt_names]
    matched_gt: set[int] = set()
    matched_gen: list[str] = []

    for gen_name, gen_toks in gen_sets:
        best_j, best_idx = 0.0, -1
        for i, gt_toks in enumerate(gt_sets):
            if i in matched_gt:
                continue
            j = _jaccard(gen_toks, gt_toks)
            if j > best_j:
                best_j, best_idx = j, i
        if best_j >= threshold:
            matched_gt.add(best_idx)
            matched_gen.append(gen_name)

    tp = len(matched_gen)
    precision = tp / len(gen_names)
    recall = tp / len(gt_names)
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    return round(precision, 4), round(recall, 4), round(f1, 4), matched_gen


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def evaluate_structure_similarity(
    ontology_path: Path,
    gt_graph: Optional[rdflib.Graph] = None,
) -> dict:
    """
    Compare the generated ontology at ontology_path against gt_graph.

    If gt_graph is None or empty, ground-truth metrics are 0 / empty.

    Returns
    -------
    dict with keys:
      gt_available, gen_class_count, gt_class_count,
      gen_property_count, gt_property_count,
      class_name_precision, class_name_recall, class_name_f1,
      property_name_precision, property_name_recall, property_name_f1,
      class_jaccard, property_jaccard, combined_f1,
      matched_class_names, matched_property_names
    """
    gen_g = _load_graph(ontology_path)
    gen_classes = _extract_names(gen_g, [OWL.Class])
    gen_props = _extract_names(gen_g, _PROPERTY_TYPES)

    gt_available = gt_graph is not None and len(gt_graph) > 0

    if not gt_available:
        return {
            "gt_available": False,
            "gen_class_count": len(gen_classes),
            "gt_class_count": 0,
            "gen_property_count": len(gen_props),
            "gt_property_count": 0,
            "class_name_precision": 0.0,
            "class_name_recall": 0.0,
            "class_name_f1": 0.0,
            "property_name_precision": 0.0,
            "property_name_recall": 0.0,
            "property_name_f1": 0.0,
            "class_jaccard": 0.0,
            "property_jaccard": 0.0,
            "combined_f1": 0.0,
            "matched_class_names": [],
            "matched_property_names": [],
        }

    gt_classes = _extract_names(gt_graph, [OWL.Class])
    gt_props = _extract_names(gt_graph, _PROPERTY_TYPES)

    c_p, c_r, c_f1, matched_c = _f1_from_name_lists(gen_classes, gt_classes)
    p_p, p_r, p_f1, matched_p = _f1_from_name_lists(gen_props, gt_props)

    # Token-level Jaccard across all class / property names
    gen_class_tokens = frozenset(t for n in gen_classes for t in _token_set(n))
    gt_class_tokens = frozenset(t for n in gt_classes for t in _token_set(n))
    gen_prop_tokens = frozenset(t for n in gen_props for t in _token_set(n))
    gt_prop_tokens = frozenset(t for n in gt_props for t in _token_set(n))

    class_jac = round(_jaccard(gen_class_tokens, gt_class_tokens), 4)
    prop_jac = round(_jaccard(gen_prop_tokens, gt_prop_tokens), 4)
    combined_f1 = round((c_f1 + p_f1) / 2, 4)

    return {
        "gt_available": True,
        "gen_class_count": len(gen_classes),
        "gt_class_count": len(gt_classes),
        "gen_property_count": len(gen_props),
        "gt_property_count": len(gt_props),
        "class_name_precision": c_p,
        "class_name_recall": c_r,
        "class_name_f1": c_f1,
        "property_name_precision": p_p,
        "property_name_recall": p_r,
        "property_name_f1": p_f1,
        "class_jaccard": class_jac,
        "property_jaccard": prop_jac,
        "combined_f1": combined_f1,
        "matched_class_names": matched_c,
        "matched_property_names": matched_p,
    }
