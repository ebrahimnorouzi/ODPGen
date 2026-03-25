"""
Formal quality metrics for a generated OWL ontology.

Parses the Turtle file with rdflib and measures structural completeness
and annotation coverage. No external reasoner is required.
"""
from __future__ import annotations

from pathlib import Path

import rdflib
from rdflib import OWL, RDF, RDFS, URIRef

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_STANDARD_NS = {str(OWL), str(RDF), str(RDFS), "http://www.w3.org/2001/XMLSchema#"}


def _load_graph(ontology_path: Path) -> tuple[rdflib.Graph, str | None]:
    """
    Parse path as Turtle (fallback: RDF/XML).
    Returns (graph, error_msg); error_msg is None on success.
    """
    g = rdflib.Graph()
    for fmt in ("turtle", "xml", "n3"):
        try:
            g.parse(str(ontology_path), format=fmt)
            if len(g) > 0:
                return g, None
        except Exception:
            g = rdflib.Graph()
    return g, f"Could not parse {ontology_path.name} in turtle/xml/n3 formats."


def _named_typed(g: rdflib.Graph, rdf_type: URIRef) -> set[URIRef]:
    """URIRef subjects typed as rdf_type (excludes blank nodes)."""
    return {
        s for s, _, _ in g.triples((None, RDF.type, rdf_type))
        if isinstance(s, URIRef)
    }


def _fraction_with_predicate(g: rdflib.Graph, subjects: set[URIRef], predicate: URIRef) -> float:
    if not subjects:
        return 0.0
    covered = sum(1 for s in subjects if (s, predicate, None) in g)
    return round(covered / len(subjects), 4)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def evaluate_formal_quality(ontology_path: Path) -> dict:
    """
    Parse ontology_path with rdflib and compute structural/annotation metrics.

    Returns
    -------
    dict with keys:
      parse_success, parse_error, triple_count,
      ontology_declared, ontology_label_present, ontology_comment_present,
      class_count, object_property_count, datatype_property_count,
      annotation_property_count, label_coverage_ratio, comment_coverage_ratio,
      class_label_ratio, property_label_ratio,
      domain_range_ratio, has_prefix_declarations, uses_standard_prefixes
    """
    raw_text = ontology_path.read_text(encoding="utf-8", errors="replace")
    has_prefix = any(line.lstrip().startswith("@prefix") for line in raw_text.splitlines())

    g, error = _load_graph(ontology_path)
    parse_ok = error is None and len(g) > 0

    if not parse_ok:
        return {
            "parse_success": False,
            "parse_error": error or "Empty graph after parsing.",
            "triple_count": 0,
            "ontology_declared": False,
            "ontology_label_present": False,
            "ontology_comment_present": False,
            "class_count": 0,
            "object_property_count": 0,
            "datatype_property_count": 0,
            "annotation_property_count": 0,
            "label_coverage_ratio": 0.0,
            "comment_coverage_ratio": 0.0,
            "class_label_ratio": 0.0,
            "property_label_ratio": 0.0,
            "domain_range_ratio": 0.0,
            "has_prefix_declarations": has_prefix,
            "uses_standard_prefixes": False,
        }

    # --- Ontology header ---
    onto_nodes = _named_typed(g, OWL.Ontology)
    ontology_declared = len(onto_nodes) > 0
    onto_node = next(iter(onto_nodes), None)
    onto_label = onto_node is not None and (onto_node, RDFS.label, None) in g
    onto_comment = onto_node is not None and (onto_node, RDFS.comment, None) in g

    # --- Typed entities ---
    classes = _named_typed(g, OWL.Class)
    obj_props = _named_typed(g, OWL.ObjectProperty)
    data_props = _named_typed(g, OWL.DatatypeProperty)
    ann_props = _named_typed(g, OWL.AnnotationProperty)
    all_terms = classes | obj_props | data_props | ann_props

    # --- Annotation coverage ---
    label_cov = _fraction_with_predicate(g, all_terms, RDFS.label)
    comment_cov = _fraction_with_predicate(g, all_terms, RDFS.comment)

    # --- Per-type label coverage ---
    class_label_ratio = _fraction_with_predicate(g, classes, RDFS.label)
    property_label_ratio = _fraction_with_predicate(g, obj_props | data_props | ann_props, RDFS.label)

    # --- Domain/range coverage (object + datatype properties) ---
    props_with_dr = obj_props | data_props
    if props_with_dr:
        with_both = sum(
            1 for p in props_with_dr
            if (p, RDFS.domain, None) in g and (p, RDFS.range, None) in g
        )
        domain_range_ratio = round(with_both / len(props_with_dr), 4)
    else:
        domain_range_ratio = 0.0

    # --- Standard prefix check ---
    used_ns = {str(uri).rsplit("#", 1)[0].rsplit("/", 1)[0] + "/" for uri in g.subjects()
               if isinstance(uri, URIRef)}
    uses_standard = any(ns in used_ns for ns in _STANDARD_NS) or ontology_declared

    return {
        "parse_success": True,
        "parse_error": None,
        "triple_count": len(g),
        "ontology_declared": ontology_declared,
        "ontology_label_present": bool(onto_label),
        "ontology_comment_present": bool(onto_comment),
        "class_count": len(classes),
        "object_property_count": len(obj_props),
        "datatype_property_count": len(data_props),
        "annotation_property_count": len(ann_props),
        "label_coverage_ratio": label_cov,
        "comment_coverage_ratio": comment_cov,
        "class_label_ratio": class_label_ratio,
        "property_label_ratio": property_label_ratio,
        "domain_range_ratio": domain_range_ratio,
        "has_prefix_declarations": has_prefix,
        "uses_standard_prefixes": uses_standard,
    }
