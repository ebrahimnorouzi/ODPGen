"""
Shared RDF parsing utilities for evaluation and downloading.

Provides a robust multi-format parser that handles:
  - Turtle (.ttl)
  - RDF/XML (.rdf, standard .owl)
  - OWL/XML (OWLAPI functional-style .owl — NOT natively supported by rdflib)
  - N3, N-Triples
"""
from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import rdflib
from rdflib import OWL, RDF, RDFS, URIRef, Literal

_PARSE_ORDER = ["turtle", "xml", "n3", "nt"]


def parse_owlxml(data: bytes) -> rdflib.Graph | None:
    """
    Parse OWL/XML (OWLAPI-style) into a minimal rdflib Graph with class/property
    declarations and rdfs:label / rdfs:comment annotations.

    OWL/XML uses <Declaration><Class IRI="..."/></Declaration> which is NOT
    RDF/XML and cannot be parsed by rdflib natively.

    Returns None if the content is not recognized as OWL/XML.
    """
    OWL_NS = "http://www.w3.org/2002/07/owl#"
    try:
        root = ET.fromstring(data)
    except ET.ParseError:
        return None

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

    onto_iri = root.get("ontologyIRI") or root.get(f"{{{OWL_NS}}}ontologyIRI")
    onto_node = URIRef(onto_iri) if onto_iri else URIRef("http://example.org/ontology")
    g.add((onto_node, RDF.type, OWL.Ontology))

    for decl in root.iter(f"{{{OWL_NS}}}Declaration"):
        for child in decl:
            owl_type = type_map.get(child.tag)
            if owl_type:
                iri = child.get("IRI") or child.get("abbreviatedIRI", "")
                if iri.startswith("http"):
                    g.add((URIRef(iri), RDF.type, owl_type))

    for ann in root.iter(f"{{{OWL_NS}}}AnnotationAssertion"):
        children = list(ann)
        if len(children) >= 3:
            prop_iri = children[0].get("IRI") or children[0].get("abbreviatedIRI", "")
            subj_iri = children[1].get("IRI") or children[1].get("abbreviatedIRI", "")
            val_text = children[2].text or ""
            if subj_iri.startswith("http") and val_text:
                pred = None
                if "label" in prop_iri.lower():
                    pred = RDFS.label
                elif "comment" in prop_iri.lower():
                    pred = RDFS.comment
                if pred:
                    g.add((URIRef(subj_iri), pred, Literal(val_text)))

    return g if len(g) > 0 else None


def load_graph(path: Path) -> rdflib.Graph:
    """
    Robustly parse an RDF file at path, trying multiple formats.
    Falls back to OWL/XML parser for OWLAPI-style files.
    Returns an empty Graph on total failure (never raises).
    """
    data = path.read_bytes()
    for fmt in _PARSE_ORDER:
        try:
            g = rdflib.Graph()
            g.parse(data=data.decode("utf-8", errors="replace"), format=fmt)
            if len(g) > 0:
                return g
        except Exception:
            pass

    g = parse_owlxml(data)
    if g is not None:
        return g

    return rdflib.Graph()
