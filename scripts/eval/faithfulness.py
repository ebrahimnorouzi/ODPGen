"""
Faithfulness metrics: how well does the generated ontology cover the scenario and CQs?

For each CQ, we check whether the ontology's vocabulary (class/property names
and rdfs:label values) contains enough tokens to plausibly answer it.
This is a lexical proxy for structural CQ coverage.
"""
from __future__ import annotations

import re
from pathlib import Path

import rdflib
from rdflib import OWL, RDF, RDFS, URIRef

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CQ_COVERAGE_THRESHOLD = 0.4  # Fraction of CQ tokens that must appear in onto vocab

# Extended stopword list for ontology + NL text
_STOPWORDS = {
    "a", "an", "the", "and", "or", "of", "to", "in", "on", "for", "with",
    "is", "are", "be", "as", "by", "it", "this", "that", "from", "at",
    "if", "then", "than", "into", "about", "what", "which", "when", "where",
    "how", "who", "can", "does", "have", "has", "been", "used", "given",
    "type", "types", "list", "show", "each", "any", "all", "not", "no",
    "its", "their", "was", "were", "do", "did", "will", "would", "could",
    "should", "shall", "may", "might", "use", "using", "per", "via",
}

# Namespace tokens to exclude from hallucination calculation
_NAMESPACE_TOKENS = {
    "owl", "rdf", "rdfs", "xsd", "dc", "dcterms", "skos", "foaf",
    "prov", "schema", "dcat", "void", "http", "https", "www", "org",
    "com", "net", "example", "ontology", "namespace", "prefix",
}


# ---------------------------------------------------------------------------
# Tokenisation helpers
# ---------------------------------------------------------------------------

def _content_tokens(text: str) -> set[str]:
    """Lowercase alpha-num tokens, min length 3, stopwords removed."""
    raw = re.findall(r"[A-Za-z][A-Za-z0-9]*", text.lower())
    return {t for t in raw if len(t) >= 3 and t not in _STOPWORDS}


def _split_camel(name: str) -> list[str]:
    """Split a camelCase or PascalCase local name into tokens."""
    name = re.sub(r"[_\-]+", " ", name)
    name = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", name)
    name = re.sub(r"(?<=[A-Z])(?=[A-Z][a-z])", " ", name)
    return [t.lower() for t in name.split() if len(t) >= 3]


def _local_name(uri: str) -> str:
    return uri.split("#")[-1] if "#" in uri else uri.rstrip("/").rsplit("/", 1)[-1]


# ---------------------------------------------------------------------------
# Ontology vocabulary extraction
# ---------------------------------------------------------------------------

def _build_ontology_vocab(g: rdflib.Graph) -> set[str]:
    """
    Build a set of content tokens from the graph.

    Sources:
      - Local names of all named (URIRef) classes and properties (camelCase split)
      - All rdfs:label and rdfs:comment literal values
    Excludes standard namespace tokens.
    """
    vocab: set[str] = set()

    type_uris = [OWL.Class, OWL.ObjectProperty, OWL.DatatypeProperty, OWL.AnnotationProperty]
    for rdf_type in type_uris:
        for s, _, _ in g.triples((None, RDF.type, rdf_type)):
            if isinstance(s, URIRef):
                vocab.update(t for t in _split_camel(_local_name(str(s)))
                             if t not in _NAMESPACE_TOKENS)

    for predicate in (RDFS.label, RDFS.comment):
        for _, _, obj in g.triples((None, predicate, None)):
            vocab.update(t for t in _content_tokens(str(obj))
                         if t not in _NAMESPACE_TOKENS)

    return vocab


def _fallback_vocab_from_text(ontology_text: str) -> set[str]:
    """
    When rdflib parse fails, extract vocabulary from raw TTL text using regex.
    Picks up rdfs:label values and local names after '#' or '/'.
    """
    vocab: set[str] = set()
    # rdfs:label "..." or rdfs:comment "..."
    for m in re.finditer(r'rdfs:(?:label|comment)\s+"([^"]+)"', ontology_text, re.IGNORECASE):
        vocab.update(_content_tokens(m.group(1)))
    # Local names after # in URIs or bare :LocalName patterns
    for m in re.finditer(r"[#/:]([A-Za-z][A-Za-z0-9_]{2,})", ontology_text):
        vocab.update(t for t in _split_camel(m.group(1)) if t not in _NAMESPACE_TOKENS)
    return vocab


# ---------------------------------------------------------------------------
# Per-CQ evaluation
# ---------------------------------------------------------------------------

def _evaluate_single_cq(
    cq_text: str,
    cq_index: int,
    onto_vocab: set[str],
    threshold: float = CQ_COVERAGE_THRESHOLD,
) -> dict:
    cq_tokens = _content_tokens(cq_text)
    if not cq_tokens:
        return {
            "cq_index": cq_index,
            "cq_text": cq_text,
            "cq_token_count": 0,
            "covered_token_count": 0,
            "token_recall": 0.0,
            "covered": False,
        }
    covered = cq_tokens & onto_vocab
    recall = len(covered) / len(cq_tokens)
    return {
        "cq_index": cq_index,
        "cq_text": cq_text,
        "cq_token_count": len(cq_tokens),
        "covered_token_count": len(covered),
        "token_recall": round(recall, 4),
        "covered": recall >= threshold,
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def evaluate_faithfulness(
    scenario_text: str,
    cq_list: list[str],
    ontology_path: Path,
) -> dict:
    """
    Evaluate whether the generated ontology vocabulary covers the scenario and CQs.

    Parameters
    ----------
    scenario_text : str
        Full scenario description text.
    cq_list : list[str]
        List of competency question strings.
    ontology_path : Path
        Path to the generated Turtle file.

    Returns
    -------
    dict with keys:
      cq_count, cq_coverage_ratio, cq_mean_token_recall, cq_details,
      ontology_vocab_size, scenario_vocab_size,
      scenario_coverage_ratio, hallucination_ratio
    """
    # Parse the generated ontology
    raw_text = ontology_path.read_text(encoding="utf-8", errors="replace")
    g = rdflib.Graph()
    parsed_ok = False
    for fmt in ("turtle", "xml", "n3"):
        try:
            g.parse(data=raw_text, format=fmt)
            if len(g) > 0:
                parsed_ok = True
                break
        except Exception:
            g = rdflib.Graph()

    onto_vocab = _build_ontology_vocab(g) if parsed_ok else _fallback_vocab_from_text(raw_text)

    # Per-CQ evaluation
    cq_details = [
        _evaluate_single_cq(cq, i, onto_vocab)
        for i, cq in enumerate(cq_list)
    ]
    n_cqs = len(cq_details)
    n_covered = sum(1 for d in cq_details if d["covered"])
    cq_coverage_ratio = round(n_covered / n_cqs, 4) if n_cqs else 0.0
    cq_mean_recall = round(
        sum(d["token_recall"] for d in cq_details) / n_cqs, 4
    ) if n_cqs else 0.0

    # Scenario-level coverage
    scenario_tokens = _content_tokens(scenario_text)
    scenario_vocab_size = len(scenario_tokens)
    scenario_covered = scenario_tokens & onto_vocab
    scenario_coverage = round(len(scenario_covered) / scenario_vocab_size, 4) if scenario_tokens else 0.0

    # Hallucination: ontology vocab tokens not grounded in scenario+CQs
    source_vocab = scenario_tokens | {
        t for cq in cq_list for t in _content_tokens(cq)
    }
    hallucination_tokens = onto_vocab - source_vocab
    hallucination_ratio = round(len(hallucination_tokens) / len(onto_vocab), 4) if onto_vocab else 0.0

    return {
        "cq_count": n_cqs,
        "cq_coverage_ratio": cq_coverage_ratio,
        "cq_mean_token_recall": cq_mean_recall,
        "cq_details": cq_details,
        "ontology_vocab_size": len(onto_vocab),
        "scenario_vocab_size": scenario_vocab_size,
        "scenario_coverage_ratio": scenario_coverage,
        "hallucination_ratio": hallucination_ratio,
    }
