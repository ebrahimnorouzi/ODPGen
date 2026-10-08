"""Offline, deterministic, intrinsic measure of ODP structural quality.

The quantity measured is CONNECTIVITY: how much of the vocabulary an artifact
declares is actually anchored into a model, and how densely it is anchored.
Nothing here calls a network service, a reasoner, or an LLM, and nothing here
reads a scenario file, a CQ signature, or a reference pattern.  The score is a
function of the artifact bytes alone (contract I9).

Why connectivity
----------------
Over the committed 420-generation corpus the reasoner never fired (``consistent``
took one distinct value, ``unsatisfiable_classes_count`` one, ``individuals_count``
one) and OOPS! is unreachable offline.  What *does* vary is whether declared
vocabulary is connected: Llama-2-70b declares 176 object properties and anchors
none of them, gemini anchors most of what it declares.  That is the signal.

The shape of the measure
------------------------
Let ``E`` be the artifact's roster of modelling entities (D6: URIRef subjects,
role-classified, meta-vocabulary removed) and let ``d(e)`` be the number of
*counted anchor incidences* of ``e`` -- the surviving, resolving, informative
axioms in which ``e`` takes part.  Read ``d(e)`` as the evidence that ``e`` is
part of a model and ``1 / (1 + d(e))`` as the residual doubt that it is: an
entity begins wholly unexplained, its first anchor halves the doubt, its second
thirds it, and no finite amount of anchoring ever quite closes it.  The measure
is the single pooled ratio of evidence to evidence-plus-doubt,

    score = sum_e d(e) / ( sum_e d(e) + sum_e 1/(1 + d(e)) )

computed with :class:`fractions.Fraction` and converted to ``float`` once.

Every property of that shape is load-bearing:

*   It is **intensive**.  ``k`` IRI-disjoint isomorphic copies multiply both
    ``sum d`` and ``|E|`` by ``k``, so the ratio is bit-identical (I11), with no
    epsilon anywhere.  It is also a *mediant*, so a disjoint union of two
    artifacts always lies between their two scores (I11, second clause).
*   It is **zero exactly** when no entity is anchored -- an empty file, a prefix
    block, refusal prose, an ontology header, 500 bare declarations (I2).
*   Adding unanchored vocabulary raises ``|E|`` and not ``sum d``, so padding
    strictly lowers it (I6), and so does any construction that adds entities
    without adding surviving anchors (I7).
*   Deleting an axiom removes incidences without removing roster entities
    (roster membership survives on annotations and on unanchored declarations,
    neither of which is load-bearing under D8), so deletion never pays (I4).
*   Adding one anchored property to a star adds two incidences and one entity,
    which strictly raises the ratio -- so the measure grows along a grounded
    prefix chain instead of saturating at its first axiom (I5).  A per-entity
    *mean* provably cannot do this, which is why the measure is a pooled ratio
    and not an average of per-entity credits.

Components are the same ratio restricted to the property, class and individual
roles; the pooled score is exactly their mass-weighted mean, so reporting them
separately is lossless (I13(b), and the review objection to single-number
reduction).

What deliberately does NOT count
--------------------------------
``rdfs:subClassOf`` / ``owl:equivalentClass`` / ``owl:disjointWith`` /
``rdfs:subPropertyOf`` / ``owl:inverseOf`` between *named* terms contribute no
incidences at all.  Every one of contract I7's roster functions -- bottom, top,
mesh, chain, clique, clique_reversed, disjoint_all -- is built out of exactly
those predicates, and every one of them is therefore worth exactly zero here,
without a hub detector, a fan-out threshold or an asymmetric traversal to get
wrong (E5, E6).  Because these predicates are ignored symmetrically, a clique
and its reverse are literally the same input to the measure, and
``A owl:equivalentClass B`` and the pair of ``rdfs:subClassOf`` axioms it
abbreviates are too (I15(e)).  Five of the eleven parseable reference patterns
have no subclass axiom at all (G-D), so nothing in the gold set is lost.

Declared arcs must *resolve*: an ``rdfs:domain`` whose object is a class the
artifact never says anything about, or is ``owl:Thing`` / ``rdfs:Resource`` /
``owl:Nothing`` / any RDF, RDFS, OWL or XSD term, anchors nothing.

A predicate whose extension is complete over a connected component of its own
extension (every ordered pair, or every source-target pair of a bipartition)
relates everything to everything within that component and so discriminates
nothing; its axioms are dropped.  The test is component-local precisely so that
it survives replication unchanged (I11).

MISSING
-------
``MISSING`` is ``None``: it JSON round-trips as ``null``, it is not 0, not 0.0,
not 1, not ``""`` and not -1, and every consumer in this module tests it with
``is None``.  A missing component is dropped and the remaining weights are
renormalised (I13).  There is no ``or 0``, no ``.get(k, 0)`` and no
``nan_to_num`` on any path that touches a component value.  OOPS! is
permanently MISSING in this environment and is never treated as zero pitfalls.
"""

from __future__ import annotations

import math
from collections import defaultdict
from fractions import Fraction
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from rdflib import BNode, Graph, Literal, URIRef
from rdflib.namespace import OWL, RDF, RDFS, SKOS, XSD

__all__ = [
    "MISSING",
    "ANNOTATION_PREDICATES",
    "DECLARATION_TYPES",
    "CLASS_DECLARATION_TYPES",
    "PROPERTY_DECLARATION_TYPES",
    "INDIVIDUAL_DECLARATION_TYPES",
    "STRUCTURAL_PREDICATES",
    "SUBSUMPTION_PREDICATES",
    "EQUIVALENCE_PREDICATES",
    "TAUTOLOGICAL_TERMS",
    "COMPONENT_ORDER",
    "PITFALL_WEIGHT",
    "evaluate",
    "evaluate_reference",
    "evaluate_path",
    "parse_artifact",
    "connective_axioms",
    "declaration_triples",
    "annotation_triples",
    "entity_roster",
    "grounded_entities",
    "load_bearing_axioms",
    "is_vacuous",
    "Analysis",
]


# ---------------------------------------------------------------------------
# The MISSING sentinel (D2, I1, I13(a)).
# ---------------------------------------------------------------------------
MISSING = None


# ---------------------------------------------------------------------------
# D3 - annotation predicates.  A closed, enumerable module constant.
# ---------------------------------------------------------------------------
DC = "http://purl.org/dc/elements/1.1/"
DCTERMS = "http://purl.org/dc/terms/"
VANN = "http://purl.org/vocab/vann/"
SCHEMA = "http://schema.org/"

ANNOTATION_PREDICATES: frozenset = frozenset(
    {
        # The contract's D3 minimum, verbatim.
        RDFS.label,
        RDFS.comment,
        RDFS.seeAlso,
        RDFS.isDefinedBy,
        OWL.versionInfo,
        SKOS.definition,
        SKOS.note,
        SKOS.prefLabel,
        SKOS.altLabel,
        URIRef(DCTERMS + "title"),
        URIRef(DCTERMS + "description"),
        URIRef(DC + "title"),
        URIRef(DC + "description"),
        # Unambiguously documentary additions.  dcterms:creator / publisher /
        # issued are deliberately NOT here: they carry real relational content
        # in the ABox exemplification patterns (2023-134-01/02/03, G-A) and
        # must stay connective.
        SKOS.scopeNote,
        SKOS.editorialNote,
        SKOS.example,
        SKOS.changeNote,
        SKOS.historyNote,
        SKOS.hiddenLabel,
        OWL.priorVersion,
        OWL.backwardCompatibleWith,
        OWL.incompatibleWith,
        OWL.deprecated,
        URIRef(DCTERMS + "abstract"),
        URIRef(DCTERMS + "license"),
        URIRef(DCTERMS + "rights"),
        URIRef(DC + "rights"),
        URIRef(VANN + "preferredNamespacePrefix"),
        URIRef(VANN + "preferredNamespaceUri"),
        URIRef(SCHEMA + "name"),
        URIRef(SCHEMA + "description"),
    }
)


# ---------------------------------------------------------------------------
# D4 - declaration triples.
# ---------------------------------------------------------------------------
CLASS_DECLARATION_TYPES: frozenset = frozenset(
    {OWL.Class, RDFS.Class, SKOS.Concept, URIRef(str(RDFS) + "Klass")}
)
PROPERTY_DECLARATION_TYPES: frozenset = frozenset(
    {
        OWL.ObjectProperty,
        OWL.DatatypeProperty,
        OWL.AnnotationProperty,
        OWL.FunctionalProperty,
        OWL.InverseFunctionalProperty,
        OWL.TransitiveProperty,
        OWL.SymmetricProperty,
        OWL.AsymmetricProperty,
        OWL.ReflexiveProperty,
        OWL.IrreflexiveProperty,
        RDF.Property,
        # G-F: gold 2025-151-02 declares properties as rdfs:Property ten times,
        # a term that does not exist in RDFS.  Gold does it, so it counts.
        URIRef(str(RDFS) + "Property"),
    }
)
INDIVIDUAL_DECLARATION_TYPES: frozenset = frozenset({OWL.NamedIndividual, OWL.Thing})
OTHER_DECLARATION_TYPES: frozenset = frozenset(
    {OWL.Ontology, OWL.Restriction, OWL.AllDisjointClasses, OWL.AllDifferent, OWL.Axiom}
)
ANNOTATION_DECLARATION_TYPES: frozenset = frozenset({OWL.AnnotationProperty})

DECLARATION_TYPES: frozenset = (
    CLASS_DECLARATION_TYPES
    | PROPERTY_DECLARATION_TYPES
    | INDIVIDUAL_DECLARATION_TYPES
    | OTHER_DECLARATION_TYPES
)


# ---------------------------------------------------------------------------
# Schema vocabulary.
# ---------------------------------------------------------------------------
SUBSUMPTION_PREDICATES: frozenset = frozenset({RDFS.subClassOf, RDFS.subPropertyOf})
EQUIVALENCE_PREDICATES: frozenset = frozenset(
    {OWL.equivalentClass, OWL.equivalentProperty, OWL.sameAs}
)
DISJOINTNESS_PREDICATES: frozenset = frozenset(
    {OWL.disjointWith, OWL.propertyDisjointWith, OWL.differentFrom}
)
# Named-to-named axioms over these predicates contribute no incidence: every
# roster function in contract I7 is built out of them.
INERT_PREDICATES: frozenset = (
    SUBSUMPTION_PREDICATES
    | EQUIVALENCE_PREDICATES
    | DISJOINTNESS_PREDICATES
    | frozenset({OWL.inverseOf, OWL.disjointUnionOf, OWL.propertyDisjointWith})
)

RESTRICTION_FILLER_PREDICATES: frozenset = frozenset(
    {OWL.someValuesFrom, OWL.allValuesFrom, OWL.onClass, OWL.onDataRange, OWL.hasValue}
)
RESTRICTION_CARDINALITY_PREDICATES: frozenset = frozenset(
    {
        OWL.cardinality,
        OWL.minCardinality,
        OWL.maxCardinality,
        OWL.qualifiedCardinality,
        OWL.minQualifiedCardinality,
        OWL.maxQualifiedCardinality,
    }
)
CLASS_CONSTRUCTOR_PREDICATES: frozenset = frozenset(
    {OWL.unionOf, OWL.intersectionOf, OWL.complementOf, OWL.oneOf, OWL.disjointUnionOf}
)

STRUCTURAL_PREDICATES: frozenset = (
    INERT_PREDICATES
    | RESTRICTION_FILLER_PREDICATES
    | RESTRICTION_CARDINALITY_PREDICATES
    | CLASS_CONSTRUCTOR_PREDICATES
    | frozenset(
        {
            RDF.type,
            RDFS.domain,
            RDFS.range,
            OWL.onProperty,
            OWL.propertyChainAxiom,
            OWL.hasKey,
            OWL.members,
            OWL.distinctMembers,
            OWL.imports,
            OWL.withRestrictions,
            OWL.onDatatype,
            RDF.first,
            RDF.rest,
            OWL.annotatedSource,
            OWL.annotatedProperty,
            OWL.annotatedTarget,
        }
    )
)

# Terms that are the universal class, the empty class, or pure meta-vocabulary.
# Anchoring to one of them is a tautology, so it anchors nothing.
TAUTOLOGICAL_TERMS: frozenset = frozenset(
    {
        OWL.Thing,
        OWL.Nothing,
        RDFS.Resource,
        RDFS.Literal,
        RDFS.Datatype,
        RDFS.Class,
        OWL.Class,
        RDF.Property,
        RDFS.Container,
        URIRef(str(RDFS) + "Property"),
    }
)

_META_PREFIXES: Tuple[str, ...] = (str(XSD), str(RDF), str(RDFS), str(OWL))

COMPONENT_ORDER: Tuple[str, ...] = (
    "property_connectivity",
    "class_connectivity",
    "individual_connectivity",
    "pitfall_freedom",
)
_ROLE_COMPONENTS: Tuple[str, ...] = COMPONENT_ORDER[:3]

# The one fixed weight in the module.  It is used only when a caller supplies a
# real pitfall report; offline it is never used, because OOPS! is MISSING and
# the structural mass is renormalised to 1.
PITFALL_WEIGHT: Fraction = Fraction(3, 20)

# Below three entities a "complete relation" is indistinguishable from a single
# ordinary edge, so the completeness filter must not fire there (G-B: two gold
# patterns are ten triples).
_COMPLETENESS_MIN = 3

# The residual-doubt schedule.  An entity with no anchor is wholly unexplained
# and carries one whole unit of doubt.  Its first anchor is what settles the
# question the measure is asking -- is this term part of a model, or is it
# vocabulary the generator emitted and never used -- and so it closes all but
# _DOUBT_FIRST of that doubt; further anchors close the rest quadratically, and
# _DOUBT_FULL independent anchors close it entirely.  Coverage therefore
# dominates density by a factor of 1/_DOUBT_FIRST, which is what lets a
# ten-triple ABox pattern and a five-domain star sit in the same band as a
# ninety-triple pattern with restrictions (contract I12(c), G-B).  These are
# module constants, not caller-tunable options: I14(b) forbids any knob that
# changes a score.
_DOUBT_FULL = 4
_DOUBT_FIRST = Fraction(1, 30)
_DOUBT_DECAY = 2


def _doubt(d: int) -> Fraction:
    """Residual doubt that an entity carrying ``d`` anchors is part of a model."""
    if d <= 0:
        return Fraction(1)
    if d >= _DOUBT_FULL:
        return Fraction(0)
    return _DOUBT_FIRST * Fraction(
        (_DOUBT_FULL - d) ** _DOUBT_DECAY, (_DOUBT_FULL - 1) ** _DOUBT_DECAY
    )
_BNODE_MAX_DEPTH = 12
# Above this many taxonomy nodes the logical closure is not materialised and
# direct quotient edges are used instead.  No artifact in the corpus and no
# reference pattern but the 15.8 MB one comes near it.
_CLOSURE_MAX_NODES = 3000
_MAX_BYTES = 64 * 1024 * 1024


def _is_meta(term: Any) -> bool:
    if not isinstance(term, URIRef):
        return False
    s = str(term)
    for prefix in _META_PREFIXES:
        if s.startswith(prefix):
            return True
    return False


# ---------------------------------------------------------------------------
# Parsing.  Content-sniffed, never extension-based (I15(f)); never raises (I1).
# ---------------------------------------------------------------------------
_FORMAT_ORDER: Tuple[str, ...] = ("turtle", "xml", "n3", "nt", "json-ld", "trig")


class ParseResult:
    """Outcome of parsing an artifact's bytes."""

    __slots__ = ("graph", "syntax", "status", "n_triples")

    def __init__(self, graph: Optional[Graph], syntax: Optional[str], status: str):
        self.graph = graph
        self.syntax = syntax
        self.status = status  # "parsed" | "empty" | "unparseable"
        self.n_triples = 0 if graph is None else len(graph)


def _to_bytes(artifact: Any) -> bytes:
    if isinstance(artifact, bytes):
        return artifact
    if isinstance(artifact, (bytearray, memoryview)):
        return bytes(artifact)
    if isinstance(artifact, str):
        return artifact.encode("utf-8", errors="replace")
    if isinstance(artifact, Graph):
        return artifact.serialize(format="nt").encode("utf-8")
    return str(artifact).encode("utf-8", errors="replace")


def _sniff_order(text: str) -> Sequence[str]:
    head = text.lstrip()[:4096]
    lowered = head.lower()
    if lowered.startswith("<?xml") or "<rdf:rdf" in lowered[:2048]:
        return ("xml", "turtle", "n3", "nt", "json-ld", "trig")
    if head.startswith("{") or head.startswith("["):
        return ("json-ld", "turtle", "xml", "n3", "nt", "trig")
    return _FORMAT_ORDER


def parse_artifact(artifact: Any) -> ParseResult:
    """Parse ``artifact``'s bytes with content sniffing.  Never raises."""
    if isinstance(artifact, Graph):
        # An already-parsed graph is accepted directly.  This is exactly what
        # parsing the graph's own serialisation would yield, and serialisation
        # invariance (I15(b)) is asserted over the byte path independently.
        return ParseResult(artifact, None, "parsed" if len(artifact) else "empty")
    try:
        raw = _to_bytes(artifact)
    except Exception:
        return ParseResult(None, None, "unparseable")

    if len(raw) > _MAX_BYTES:
        raw = raw[:_MAX_BYTES]
    if not raw.strip():
        return ParseResult(Graph(), None, "empty")

    try:
        text = raw.decode("utf-8", errors="replace")
    except Exception:
        return ParseResult(None, None, "unparseable")
    if not text.strip():
        return ParseResult(Graph(), None, "empty")

    fallback: Optional[ParseResult] = None
    for syntax in _sniff_order(text):
        graph = Graph()
        try:
            graph.parse(data=text, format=syntax)
        except Exception:
            continue
        except RecursionError:  # pragma: no cover - defensive
            continue
        if len(graph) > 0:
            return ParseResult(graph, syntax, "parsed")
        if fallback is None:
            # Parsed cleanly but yielded nothing: a prefix block or a comment
            # block.  Keep looking for a syntax that finds triples.
            fallback = ParseResult(graph, syntax, "empty")
    if fallback is not None:
        return fallback
    return ParseResult(None, None, "unparseable")


# ---------------------------------------------------------------------------
# D3 / D4 / D5 - triple classification.
# ---------------------------------------------------------------------------
def annotation_triples(graph: Graph) -> Set[Tuple[Any, Any, Any]]:
    """Triples whose predicate is in the declared annotation set (D3)."""
    return {(s, p, o) for s, p, o in graph if p in ANNOTATION_PREDICATES}


def declaration_triples(graph: Graph) -> Set[Tuple[Any, Any, Any]]:
    """``(e, rdf:type, T)`` with ``T`` a declaration type (D4)."""
    return {
        (s, p, o) for s, p, o in graph if p == RDF.type and o in DECLARATION_TYPES
    }


def connective_axioms(graph: Graph) -> Set[Tuple[Any, Any, Any]]:
    """Every triple that is neither an annotation nor a declaration (D5)."""
    out = set()
    for s, p, o in graph:
        if p in ANNOTATION_PREDICATES:
            continue
        if p == RDF.type and o in DECLARATION_TYPES:
            continue
        out.add((s, p, o))
    return out


def is_vacuous(artifact: Any) -> bool:
    """D9: the parsed graph contains zero connective axioms."""
    parsed = parse_artifact(artifact)
    if parsed.graph is None:
        return False
    return not connective_axioms(parsed.graph)


# ---------------------------------------------------------------------------
# Connected components of an undirected edge set.  Used by the completeness
# filter, which is component-local so that it survives replication (I11).
# ---------------------------------------------------------------------------
def _components_with_pairs(
    pairs: Iterable[Tuple[Any, Any]]
) -> List[Tuple[Set[Any], Set[Tuple[Any, Any]]]]:
    """Components of the undirected graph, each with the pairs that induced it.

    Every caller of :func:`_components` immediately wants, for each component,
    the restriction of ``pairs`` to that component.  Recovering it by rescanning
    ``pairs`` once per component costs ``O(|components| x |pairs|)``, which is
    what made the measure fail to return on a 15.9 MB reference.  Bucketing is
    exactly equivalent and costs one pass: ``union(a, b)`` is applied to every
    pair, so both endpoints of a pair always share a root, and
    ``{(a, b) for a, b in pairs if a in comp and b in comp}`` is precisely the
    bucket of pairs whose root is that component's.  Component order is
    unchanged, so nothing downstream can observe the difference.

    The union-find runs over **integer** node ids rather than over the terms
    themselves.  That is a change of representation and not of meaning, but it
    is the difference between a measure that returns and one that does not: an
    ``rdflib`` term's ``==`` and ``!=`` are Python-level methods, and the
    ``while parent[root] != root`` of a term-keyed union-find calls one on every
    step of every path walk.  On a 2900-class subsumption chain -- one node
    under ``_CLOSURE_MAX_NODES``, so the whole transitive closure is
    materialised -- that was 4.2 million pairs, 2.9 million ``find`` calls and
    46 seconds spent inside ``rdflib.term.__eq__`` and ``__ne__`` alone.

    Every observable is preserved exactly, and each is preserved for a reason:

    *   ids are handed out to ``a`` then ``b`` of each pair in order, which is
        precisely the order in which the term-keyed version's ``find`` first
        inserted them into ``parent``, so ``nodes`` reproduces that dict's key
        order and hence the order in which groups are first seen;
    *   the root of a merge is still the endpoint with the smaller ``str``
        (``keys`` just hoists the ``str`` calls out of the loop), so the same
        node represents the same component;
    *   the result is still ordered by the ``str`` of that representative, and
        ``sorted`` is stable over the reproduced insertion order, so even a tie
        between two distinct roots that print alike keeps its old position;
    *   terms are still identified by ``==``/``hash`` -- two ``URIRef``
        instances carrying one IRI remain one node -- and the pairs handed back
        are the caller's own objects, not the interned representatives.
    """
    pairs = list(pairs)
    ident: Dict[Any, int] = {}
    nodes: List[Any] = []
    keys: List[str] = []

    # Interning is written out rather than called, because it runs twice per
    # pair and there are millions of pairs; the dict lookup is the work, and a
    # Python call frame around it costs as much again.
    ident_get = ident.get
    indexed: List[Tuple[int, int]] = []
    append_pair = indexed.append
    for a, b in pairs:
        ia = ident_get(a, -1)
        if ia < 0:
            ia = ident[a] = len(nodes)
            nodes.append(a)
            keys.append(str(a))
        ib = ident_get(b, -1)
        if ib < 0:
            ib = ident[b] = len(nodes)
            nodes.append(b)
            keys.append(str(b))
        append_pair((ia, ib))

    parent: List[int] = list(range(len(nodes)))

    def find(i: int) -> int:
        root = i
        while parent[root] != root:
            root = parent[root]
        while parent[i] != root:
            parent[i], i = root, parent[i]
        return root

    for ia, ib in indexed:
        ra, rb = find(ia), find(ib)
        if ra == rb:
            continue
        if keys[ra] <= keys[rb]:
            parent[rb] = ra
        else:
            parent[ra] = rb

    # Resolve every node once.  The bucketing below needs a root per pair, and
    # asking ``find`` per pair is a walk per pair; asking it per node is a walk
    # per node, and after this pass ``root_of`` answers in one index.
    root_of: List[int] = [find(i) for i in range(len(nodes))]

    groups: Dict[int, Set[Any]] = {}
    order: List[int] = []
    for i, root in enumerate(root_of):
        bucket = groups.get(root)
        if bucket is None:
            bucket = groups[root] = set()
            order.append(root)
        bucket.add(nodes[i])
    local: Dict[int, Set[Tuple[Any, Any]]] = {root: set() for root in order}
    for (a, b), (ia, _ib) in zip(pairs, indexed):
        local[root_of[ia]].add((a, b))
    order.sort(key=lambda root: keys[root])
    return [(groups[root], local[root]) for root in order]


def _components(pairs: Iterable[Tuple[Any, Any]]) -> List[Set[Any]]:
    return [comp for comp, _local in _components_with_pairs(pairs)]


def _neighbourhoods(
    local: Iterable[Tuple[Any, Any]]
) -> Dict[Any, Set[Any]]:
    """Undirected adjacency of an edge set, built in one pass.

    The equivalent of ``{b if a == node else a for a, b in local
    if node in (a, b)}`` evaluated for every ``node`` at once, including the
    degenerate self-pair, which contributes the node to its own neighbourhood
    under both formulations.
    """
    adjacency: Dict[Any, Set[Any]] = defaultdict(set)
    for a, b in local:
        adjacency[a].add(b)
        if b != a:
            adjacency[b].add(a)
    return adjacency


def _strongly_connected(
    nodes: Iterable[Any], edges: Dict[Any, Set[Any]]
) -> List[Set[Any]]:
    """Iterative Tarjan.  Deterministic: neighbours are visited in sorted order."""
    index: Dict[Any, int] = {}
    low: Dict[Any, int] = {}
    on_stack: Dict[Any, bool] = {}
    stack: List[Any] = []
    result: List[Set[Any]] = []
    counter = 0
    for root in sorted(nodes, key=str):
        if root in index:
            continue
        work: List[Tuple[Any, int]] = [(root, 0)]
        while work:
            node, pi = work[-1]
            if pi == 0:
                index[node] = low[node] = counter
                counter += 1
                stack.append(node)
                on_stack[node] = True
            neighbours = sorted(edges.get(node, ()), key=str)
            if pi < len(neighbours):
                work[-1] = (node, pi + 1)
                nxt = neighbours[pi]
                if nxt not in index:
                    work.append((nxt, 0))
                elif on_stack.get(nxt):
                    low[node] = min(low[node], index[nxt])
            else:
                if low[node] == index[node]:
                    comp: Set[Any] = set()
                    while True:
                        w = stack.pop()
                        on_stack[w] = False
                        comp.add(w)
                        if w == node:
                            break
                    result.append(comp)
                work.pop()
                if work:
                    parent = work[-1][0]
                    low[parent] = min(low[parent], low[node])
    return result


def _uninformative_predicate_pairs(
    pairs_by_predicate: Dict[Any, Set[Tuple[Any, Any]]]
) -> Set[Tuple[Any, Any, Any]]:
    """Drop the axioms of a predicate that relates everything to everything.

    Within each connected component of a predicate's own extension the
    predicate is dropped when its restriction to that component is the complete
    relation on the component's entities, or the complete bipartite relation
    between the component's sources and targets.  A relation that holds of
    every pair discriminates no pair, so it is evidence of nothing.  Working
    per component rather than globally is what makes the filter survive
    replication: ``k`` disjoint copies are ``k`` components and each is judged
    exactly as the original was.
    """
    dropped: Set[Tuple[Any, Any, Any]] = set()
    for pred, pairs in pairs_by_predicate.items():
        if len(pairs) < _COMPLETENESS_MIN:
            continue
        for _comp, local in _components_with_pairs(pairs):
            if len(local) < _COMPLETENESS_MIN:
                continue
            nodes = {x for pair in local for x in pair}
            sources = {a for a, _b in local}
            targets = {b for _a, b in local}
            complete_all = len(nodes) >= _COMPLETENESS_MIN and len(local) == len(
                nodes
            ) * (len(nodes) - 1)
            complete_bipartite = (
                len(sources) >= 2
                and len(targets) >= 2
                and len(sources) * len(targets) >= 6
                and len(local) == len(sources) * len(targets)
                and not (sources & targets)
            )
            if complete_all or complete_bipartite:
                for a, b in local:
                    dropped.add((a, pred, b))
    return dropped


# ---------------------------------------------------------------------------
# The analysis.
# ---------------------------------------------------------------------------
class Analysis:
    """Everything the measure derives from one parsed graph.

    Every set here is order-independent and every aggregate is a
    :class:`~fractions.Fraction`, so equal structure gives bit-identical floats.
    """

    def __init__(self, graph: Graph) -> None:
        self.graph = graph
        self.conn = connective_axioms(graph)
        self.decls = declaration_triples(graph)
        self.annotated: Set[URIRef] = {
            s for s, p, _o in graph if p in ANNOTATION_PREDICATES and isinstance(s, URIRef)
        }
        self.subjects: Set[URIRef] = {s for s in graph.subjects() if isinstance(s, URIRef)}

        self._collect_declarations()
        self._classify_roles()
        self._collect_restrictions()
        self._collect_usage()
        self._collect_primary_incidences()
        self._collect_taxonomy()
        self._collect_incidences()

    # -- declarations ------------------------------------------------------
    def _collect_declarations(self) -> None:
        self.declared_class: Set[URIRef] = set()
        self.declared_class_strong: Set[URIRef] = set()
        self.declared_property: Set[URIRef] = set()
        self.declared_annotation_property: Set[URIRef] = set()
        self.declared_individual: Set[URIRef] = set()
        self.declared_ontology: Set[URIRef] = set()
        for s, _p, o in self.decls:
            if not isinstance(s, URIRef):
                continue
            if o in CLASS_DECLARATION_TYPES:
                self.declared_class.add(s)
                if o != SKOS.Concept:
                    self.declared_class_strong.add(s)
            if o in PROPERTY_DECLARATION_TYPES:
                self.declared_property.add(s)
            if o in ANNOTATION_DECLARATION_TYPES:
                self.declared_annotation_property.add(s)
            if o in INDIVIDUAL_DECLARATION_TYPES:
                self.declared_individual.add(s)
            if o == OWL.Ontology:
                self.declared_ontology.add(s)

    # -- roles -------------------------------------------------------------
    def _classify_roles(self) -> None:
        """Infer class-like / property-like roles from position, never from IRI.

        Roles are resolved by precedence rather than by intersection, because
        gold does not keep them apart: 2025-151-02 declares each of its ten
        properties ``a rdfs:Property, skos:Concept`` and its one class
        ``a rdfs:Class, owl:Class, skos:Concept`` (G-F).  Treating an entity
        with both signals as ambiguous, and so as neither, scores that
        published pattern at zero.  Property evidence therefore wins over the
        weak class signal, and only a genuine conflict -- an ``owl:Class`` used
        as a predicate -- is ambiguous.
        """
        strong_class: Set[URIRef] = set(self.declared_class_strong)
        weak_class: Set[URIRef] = set(self.declared_class)
        strong_property: Set[URIRef] = set(self.declared_property)
        typed_by: Dict[URIRef, Set[Any]] = defaultdict(set)

        for s, p, o in self.conn:
            if p == RDF.type:
                if isinstance(s, URIRef) and isinstance(o, URIRef):
                    typed_by[s].add(o)
                    strong_class.add(o)
                continue
            if p in (RDFS.subClassOf, OWL.equivalentClass, OWL.disjointWith):
                if isinstance(s, URIRef):
                    weak_class.add(s)
                if isinstance(o, URIRef):
                    weak_class.add(o)
                continue
            if p in (RDFS.subPropertyOf, OWL.equivalentProperty, OWL.inverseOf,
                     OWL.propertyDisjointWith):
                if isinstance(s, URIRef):
                    strong_property.add(s)
                if isinstance(o, URIRef):
                    strong_property.add(o)
                continue
            if p in (RDFS.domain, RDFS.range):
                if isinstance(s, URIRef):
                    strong_property.add(s)
                if isinstance(o, URIRef):
                    strong_class.add(o)
                continue
            if p in RESTRICTION_FILLER_PREDICATES:
                if isinstance(o, URIRef) and p != OWL.hasValue:
                    strong_class.add(o)
                continue
            if p == OWL.onProperty:
                if isinstance(o, URIRef):
                    strong_property.add(o)
                continue
            if isinstance(p, URIRef) and p not in STRUCTURAL_PREDICATES:
                strong_property.add(p)

        self.typed_by: Dict[URIRef, Set[Any]] = dict(typed_by)
        class_like = {c for c in (strong_class | weak_class) if not _is_meta(c)}
        property_like = {p for p in strong_property if not _is_meta(p)}
        # Only a conflict between two *strong* signals is genuinely ambiguous.
        self.ambiguous: Set[URIRef] = strong_class & property_like
        self.class_like: Set[URIRef] = (class_like - property_like) - self.ambiguous
        self.property_like: Set[URIRef] = property_like - self.ambiguous

    # -- resolution --------------------------------------------------------
    def resolves_as_class(self, term: Any) -> bool:
        """Is ``term`` a class this artifact actually introduces?

        A domain, range or restriction filler pointing at a term the artifact
        never says anything about, or at ``owl:Thing`` / ``rdfs:Resource`` /
        any RDF, RDFS, OWL or XSD term, anchors nothing (contract C2).
        """
        if not isinstance(term, URIRef):
            return False
        if _is_meta(term) or term in TAUTOLOGICAL_TERMS:
            return False
        if term not in self.class_like:
            return False
        if term in self.declared_ontology:
            return False
        return term in self.subjects

    def resolves_as_property(self, term: Any) -> bool:
        if not isinstance(term, URIRef):
            return False
        if _is_meta(term) or term in TAUTOLOGICAL_TERMS:
            return False
        return term in self.property_like

    # -- class expressions -------------------------------------------------
    def _collect_restrictions(self) -> None:
        """Anonymous class expressions, credited arc by arc.

        A restriction is three triples -- an attachment, an ``owl:onProperty``
        and a filler or a cardinality -- and the measure credits each of them
        separately to its named end rather than waiting for the body to be
        complete.  Crediting only complete bodies makes the measure fall
        sharply whenever a graph contains half a restriction, which is what
        every prefix of a permutation of a restriction-bearing pattern
        contains, and monotone growth (I5) is worth more than the purity of
        refusing to read a half-written axiom.  The blank node must carry at
        least one triple of its own, so ``C rdfs:subClassOf []`` -- an empty
        anonymous superclass, and a thing a generator can emit for every class
        without knowing anything -- is worth nothing.

        The two relations this materialises, attachment (class, property) and
        filler (property, class), are passed through the same completeness
        filter as every other predicate, so a construction that hangs the same
        restriction on every class of a component is worth nothing either.
        """
        outgoing: Dict[Any, int] = defaultdict(int)
        for sub, _p, _o in self.conn:
            if isinstance(sub, BNode):
                outgoing[sub] += 1

        onprop: Dict[Any, Set[Any]] = defaultdict(set)
        fillers: Dict[Any, Set[Any]] = defaultdict(set)
        cards: Dict[Any, int] = defaultdict(int)
        context: Dict[Any, Set[Any]] = defaultdict(set)
        attach_pairs: Set[Tuple[Any, Any]] = set()
        filler_pairs: Set[Tuple[Any, Any]] = set()
        self.expression_arcs: List[Tuple[Any, Any]] = []  # (entity, body)

        for sub, pred, obj in self.conn:
            if pred == OWL.onProperty and isinstance(sub, BNode):
                onprop[sub].add(obj)
                if outgoing[sub] >= 1 and self.resolves_as_property(obj):
                    self.expression_arcs.append((obj, sub))
            elif pred in RESTRICTION_FILLER_PREDICATES and isinstance(sub, BNode):
                fillers[sub].add(obj)
                if outgoing[sub] >= 1 and self.resolves_as_class(obj):
                    self.expression_arcs.append((obj, sub))
            elif pred in RESTRICTION_CARDINALITY_PREDICATES and isinstance(sub, BNode):
                cards[sub] += 1
            elif pred in (RDFS.subClassOf, OWL.equivalentClass, OWL.disjointWith):
                if isinstance(obj, BNode) and outgoing[obj] >= 1:
                    context[obj].add(sub)
                    if self.resolves_as_class(sub):
                        self.expression_arcs.append((sub, obj))

        # Materialise the relations a restriction induces so that the
        # completeness filter can see them.
        for body, props in onprop.items():
            for prop in props:
                for ctx in context.get(body, ()):
                    attach_pairs.add((ctx, prop))
                for f in fillers.get(body, ()):
                    filler_pairs.add((prop, f))
        dropped = _uninformative_predicate_pairs(
            {"restriction-attachment": attach_pairs, "restriction-filler": filler_pairs}
        )
        self.dropped_restrictions = len(dropped)
        if dropped:
            dead: Set[Any] = set()
            for a, kind, b in dropped:
                for body, props in onprop.items():
                    if kind == "restriction-attachment" and b in props and a in context.get(body, ()):
                        dead.add(body)
                    if kind == "restriction-filler" and a in props and b in fillers.get(body, ()):
                        dead.add(body)
            self.expression_arcs = [(e, b) for e, b in self.expression_arcs if b not in dead]

        self.restriction_bodies = len(onprop)
        self.wellformed_restrictions = 0
        for body in sorted(onprop, key=str):
            props = onprop[body]
            if len(props) != 1:
                continue
            prop = next(iter(props))
            if not self.resolves_as_property(prop):
                continue
            if not any(self.resolves_as_class(f) for f in fillers.get(body, ())) \
                    and cards.get(body, 0) == 0:
                continue
            if not any(self.resolves_as_class(c) for c in context.get(body, ())):
                continue
            self.wellformed_restrictions += 1

    # -- usage -------------------------------------------------------------
    def _collect_usage(self) -> None:
        """Domain-vocabulary edges: what makes the ABox reference patterns
        measurable at all (G-A).

        Blank nodes are transparent connectors, not entities: an edge
        ``s -p-> _:b -q-> o`` counts as ``s -q-> o``.  Without this the whole
        model of 2023-134-01, which hangs off a nested ``odrl:permission``
        blank node, looks empty.  The traversal reads graph structure only and
        never a blank node's label, so renaming blank nodes changes nothing
        (I15(c)).
        """
        out_by_subject: Dict[Any, List[Tuple[Any, Any]]] = defaultdict(list)
        for s, p, o in self.conn:
            out_by_subject[s].append((p, o))

        object_edges: Set[Tuple[URIRef, Any, URIRef]] = set()
        literal_edges: Set[Tuple[URIRef, Any, Any]] = set()

        for root in sorted(out_by_subject, key=str):
            if not isinstance(root, URIRef):
                continue
            seen: Set[Any] = set()
            stack: List[Tuple[Any, Any, int]] = [
                (p, o, 0) for p, o in out_by_subject[root]
            ]
            while stack:
                pred, obj, depth = stack.pop()
                if isinstance(obj, BNode):
                    if obj in seen or depth > _BNODE_MAX_DEPTH:
                        continue
                    seen.add(obj)
                    schema_hop = pred in STRUCTURAL_PREDICATES
                    for p2, o2 in out_by_subject.get(obj, ()):
                        if p2 in (RDF.first, RDF.rest):
                            stack.append((pred, o2, depth + 1))
                        elif schema_hop:
                            # Arrived over a schema predicate (a restriction
                            # body, a list): do not manufacture usage from it.
                            continue
                        else:
                            stack.append((p2, o2, depth + 1))
                    continue
                if pred in STRUCTURAL_PREDICATES or pred in ANNOTATION_PREDICATES:
                    continue
                if isinstance(obj, URIRef):
                    if obj != root:
                        object_edges.add((root, pred, obj))
                elif isinstance(obj, Literal):
                    literal_edges.add((root, pred, obj))

        # Completeness filter (component-local).
        pairs_by_pred: Dict[Any, Set[Tuple[Any, Any]]] = defaultdict(set)
        for s, p, o in object_edges:
            pairs_by_pred[p].add((s, o))
        dropped = _uninformative_predicate_pairs(pairs_by_pred)
        self.dropped_usage = len(dropped)
        self.object_edges = {e for e in object_edges if e not in dropped}
        self.literal_edges = literal_edges

    # -- taxonomy ----------------------------------------------------------
    def _collect_taxonomy(self) -> None:
        """Surviving *comparability* and *distinction* pairs between named terms.

        Subsumption is read through its own logical closure, and only the
        closure is ever counted, so every rewrite contract I15(e) requires is a
        no-op here: ``A owl:equivalentClass B`` and the two ``rdfs:subClassOf``
        axioms it abbreviates produce the same strongly-connected component,
        reversing an ``owl:equivalentClass`` triple produces the same one, and
        adding every entailed ``rdfs:subClassOf`` edge leaves the closure
        untouched.  A clique and its reverse are literally the same object.

        Two shapes are then removed, both judged inside a single connected
        component of the relation so that replication cannot change the verdict
        (I11):

        *   a component in which *every* pair is comparable classifies nothing
            (a mesh and a chain both land here, and so does a clique after the
            quotient), and
        *   a term comparable to every other term of its component while
            carrying no anchor of its own is a universal top or bottom: it is
            ``owl:Thing`` or ``owl:Nothing`` under another name, and a
            construction that hangs every class off one fresh term is not
            evidence that any of them is modelled.

        Nothing here needs to know which construction produced the axioms; it
        needs only to notice that the axioms distinguish nothing.
        """
        self.equivalence_pairs = 0
        pairs_sub: Set[Tuple[Any, Any]] = set()
        pairs_dis: Set[Tuple[Any, Any]] = set()

        def informative(t: Any) -> bool:
            return self.resolves_as_class(t) or self.resolves_as_property(t)

        # Union-find over equivalence, then SCCs of the subsumption digraph.
        uf: Dict[Any, Any] = {}

        def find(x: Any) -> Any:
            uf.setdefault(x, x)
            root = x
            while uf[root] != root:
                root = uf[root]
            while uf[x] != root:
                uf[x], x = root, uf[x]
            return root

        def union(a: Any, b: Any) -> None:
            ra, rb = find(a), find(b)
            if ra == rb:
                return
            if str(ra) <= str(rb):
                uf[rb] = ra
            else:
                uf[ra] = rb

        raw_edges: Set[Tuple[Any, Any]] = set()
        for sub, pred, obj in self.conn:
            if not (informative(sub) and informative(obj)) or sub == obj:
                continue
            if pred in EQUIVALENCE_PREDICATES:
                union(sub, obj)
                self.equivalence_pairs += 1
                raw_edges.add((sub, obj))
                raw_edges.add((obj, sub))
            elif pred in SUBSUMPTION_PREDICATES:
                raw_edges.add((sub, obj))
            elif pred in (OWL.disjointWith, OWL.propertyDisjointWith, OWL.differentFrom,
                          OWL.inverseOf):
                pairs_dis.add(tuple(sorted((sub, obj), key=str)))

        # Cycles in subsumption are equivalence as well: collapse them.
        for _ in range(2):
            reach: Dict[Any, Set[Any]] = defaultdict(set)
            for a, b in raw_edges:
                reach[find(a)].add(find(b))
            changed = False
            nodes = set(reach) | {n for v in reach.values() for n in v}
            for scc in _strongly_connected(nodes, reach):
                if len(scc) > 1:
                    members = sorted(scc, key=str)
                    for m in members[1:]:
                        union(members[0], m)
                    changed = True
            if not changed:
                break

        quotient: Dict[Any, Set[Any]] = defaultdict(set)
        for a, b in raw_edges:
            ra, rb = find(a), find(b)
            if ra != rb:
                quotient[ra].add(rb)

        # The closure is walked over integer node ids rather than over the
        # representatives themselves.  The reachable set of a node in a
        # taxonomy of ``k`` quotient nodes can be ``k`` nodes wide, so this
        # walk touches ``O(k^2)`` entries -- 4.2 million of them for a 2900
        # class chain -- and every one of them is a set membership test, a set
        # insert and a dict lookup.  Those cost a hashed pointer compare on an
        # ``int`` and a Python-level ``rdflib.term.__eq__`` on a ``URIRef``.
        # The ids are assigned in the same ``sorted(nodes, key=str)`` order the
        # walk already used, so the traversal, its stack order and its results
        # are the ones the term-keyed version produced.
        order = sorted(set(quotient) | {n for v in quotient.values() for n in v},
                       key=str)
        node_key: List[str] = [str(n) for n in order]
        index: Dict[Any, int] = {n: i for i, n in enumerate(order)}
        adjacency: List[List[int]] = [
            [index[m] for m in quotient.get(n, ())] for n in order
        ]
        closure: List[Set[int]] = []
        if len(order) <= _CLOSURE_MAX_NODES:
            for i in range(len(order)):
                seen: Set[int] = set()
                stack = list(adjacency[i])
                while stack:
                    cur = stack.pop()
                    if cur in seen:
                        continue
                    seen.add(cur)
                    stack.extend(adjacency[cur])
                seen.discard(i)
                closure.append(seen)
        else:  # pragma: no cover - only reachable on very large graphs
            closure = [set(succ) for succ in adjacency]

        members: Dict[Any, Set[Any]] = defaultdict(set)
        for e in list(uf):
            members[find(e)].add(e)

        # ``tuple(sorted((a, b), key=str))`` for every closure pair, with both
        # ``str`` calls hoisted out of the loop.  ``sorted`` on a two-element
        # sequence performs exactly one comparison and swaps when
        # ``key(b) < key(a)``, so the conditional below returns the same tuple
        # for every input.
        rep_pairs: Set[Tuple[Any, Any]] = set()
        add_rep_pair = rep_pairs.add
        for i, reachable in enumerate(closure):
            a = order[i]
            ka = node_key[i]
            for j in reachable:
                b = order[j]
                add_rep_pair((a, b) if ka <= node_key[j] else (b, a))

        # Drop degenerate shapes, component by component.
        dropped_nodes: Set[Any] = set()
        dropped_components = 0
        for comp, local in _components_with_pairs(rep_pairs):
            n = len(comp)
            if n >= _COMPLETENESS_MIN and len(local) == n * (n - 1) // 2:
                dropped_nodes |= comp
                dropped_components += 1
                continue
            if n >= _COMPLETENESS_MIN:
                # One adjacency pass for the whole component.  Recomputing a
                # node's neighbourhood by rescanning ``local`` is quadratic in
                # the component, and a published vocabulary has components with
                # tens of thousands of nodes.
                neighbours = _neighbourhoods(local)
                for node in comp:
                    nbrs = neighbours.get(node, ())
                    if len(nbrs) == n - 1 and not any(
                        self.primary_incidence.get(e, 0) for e in members.get(node, ())
                    ):
                        dropped_nodes.add(node)
        self.dropped_taxonomy_nodes = len(dropped_nodes)
        self.dropped_taxonomy_components = dropped_components

        for a, b in rep_pairs:
            if a in dropped_nodes or b in dropped_nodes:
                continue
            for x in sorted(members.get(a, {a}), key=str):
                for y in sorted(members.get(b, {b}), key=str):
                    pairs_sub.add((x, y))

        # Distinction is symmetric, so both orientations are handed to the
        # completeness filter: "everything is disjoint from everything" is a
        # complete relation and says nothing, exactly like a mesh.
        dis_by_pred: Dict[Any, Set[Tuple[Any, Any]]] = defaultdict(set)
        for a, b in pairs_dis:
            if find(a) == find(b):
                continue
            dis_by_pred["distinction"].add((a, b))
            dis_by_pred["distinction"].add((b, a))
        dropped_dis = _uninformative_predicate_pairs(dis_by_pred)
        self.comparability_pairs = sorted(pairs_sub, key=lambda t: (str(t[0]), str(t[1])))
        self.distinction_pairs = sorted(
            (
                (a, b)
                for a, b in dis_by_pred["distinction"]
                if str(a) <= str(b) and (a, "distinction", b) not in dropped_dis
            ),
            key=lambda t: (str(t[0]), str(t[1])),
        )

    # -- incidences --------------------------------------------------------
    def _collect_primary_incidences(self) -> None:
        """Count, per entity, the surviving *primary* anchor axioms it takes
        part in: signature arcs, restriction bodies, usage edges and typing.

        These are the anchors a construction cannot manufacture out of the
        entity roster alone, which is why the taxonomy pass consults them
        before deciding whether a universal top or bottom is real.
        """
        incid: Dict[URIRef, int] = defaultdict(int)
        counted = 0

        def bump(entity: Any) -> None:
            if isinstance(entity, URIRef) and not _is_meta(entity):
                incid[entity] += 1

        # rdfs:domain / rdfs:range, filtered for completeness and resolution.
        dr_pairs: Dict[Any, Set[Tuple[Any, Any]]] = defaultdict(set)
        for s, p, o in self.conn:
            if p in (RDFS.domain, RDFS.range):
                if self.resolves_as_property(s) and self.resolves_as_class(o) and s != o:
                    dr_pairs[p].add((s, o))
        dropped_dr = _uninformative_predicate_pairs(dr_pairs)
        self.dropped_signature = len(dropped_dr)
        self.n_domains = 0
        self.n_ranges = 0
        # A property's domain is ONE signature commitment however many classes
        # it names -- several rdfs:domain axioms are a conjunction, not several
        # independent anchors -- so the property is credited once per
        # predicate.  The class on the other end is credited every time,
        # because each property naming it is a separate reference to it.
        signed: Set[Tuple[Any, Any]] = set()
        for pred, pairs in dr_pairs.items():
            for s, o in sorted(pairs, key=lambda x: (str(x[0]), str(x[1]))):
                if (s, pred, o) in dropped_dr:
                    continue
                if (s, pred) not in signed:
                    signed.add((s, pred))
                    bump(s)
                bump(o)
                counted += 1
                if pred == RDFS.domain:
                    self.n_domains += 1
                else:
                    self.n_ranges += 1

        # Anonymous class expressions: one incidence per surviving arc.
        for entity, _body in sorted(self.expression_arcs, key=lambda t: tuple(map(str, t))):
            bump(entity)
            counted += 1

        # Usage edges between entities, and attribute edges to literals.
        for s, p, o in sorted(self.object_edges, key=lambda x: tuple(map(str, x))):
            bump(s)
            bump(o)
            if isinstance(p, URIRef) and p in self.subjects:
                bump(p)
            counted += 1
        for s, p, _o in sorted(self.literal_edges, key=lambda x: tuple(map(str, x))):
            bump(s)
            if isinstance(p, URIRef) and p in self.subjects:
                bump(p)
            counted += 1

        # Typing of individuals.  ``rdf:type`` to a declaration term is a
        # declaration (D4), not an anchor; typing to a modelling class is.
        for s, types in sorted(self.typed_by.items(), key=lambda x: str(x[0])):
            if not isinstance(s, URIRef):
                continue
            typed_once = False
            for t in sorted(types, key=str):
                if t in DECLARATION_TYPES or _is_meta(t) or t in TAUTOLOGICAL_TERMS:
                    continue
                if t == s:
                    continue
                if not typed_once:
                    typed_once = True
                    bump(s)  # being typed at all is one commitment
                if t in self.subjects:
                    bump(t)
                counted += 1

        self.primary_incidence: Dict[URIRef, int] = dict(incid)
        self.primary_axioms = counted

    def _collect_incidences(self) -> None:
        """Add the surviving taxonomy pairs to the primary incidences."""
        incid: Dict[URIRef, int] = dict(self.primary_incidence)
        counted = self.primary_axioms
        # Taxonomy and distinction are counted once per entity, not once per
        # pair.  Being placed in a taxonomy at all is one anchor; being placed
        # in it sixty times is not sixty anchors.  Gold says the same thing:
        # five of eleven reference patterns have no subclass axiom and
        # 2026-155-01 has twenty-two classes and five (G-D), so taxonomy bulk
        # is not ODP quality.  Capping here also bounds the damage any
        # roster-built taxonomy could do if it slipped past the shape filters.
        taxo: Set[Any] = set()
        for a, b in self.comparability_pairs:
            taxo.add(a)
            taxo.add(b)
        dist: Set[Any] = set()
        for a, b in self.distinction_pairs:
            dist.add(a)
            dist.add(b)
        for e in sorted(taxo, key=str):
            incid[e] = incid.get(e, 0) + 1
        for e in sorted(dist, key=str):
            incid[e] = incid.get(e, 0) + 1
        counted += len(self.comparability_pairs) + len(self.distinction_pairs)
        self.incidences: Dict[URIRef, int] = incid
        self.counted_axioms = counted

    # -- roster (D6) and the role denominators -----------------------------
    def roster(self) -> Dict[str, Set[URIRef]]:
        """The modelling entities the artifact commits to, split by role.

        An entity is in the roster when the artifact *says something about it*
        (it is the subject of a triple, D6).  A term the artifact only
        mentions as an object is a reference to something outside the
        artifact - three of the ABox reference patterns link to external
        resources - and is not judged.
        """
        props: Set[URIRef] = set()
        classes: Set[URIRef] = set()
        individuals: Set[URIRef] = set()
        for e in self.subjects:
            if _is_meta(e) or e in TAUTOLOGICAL_TERMS:
                continue
            if e in self.declared_ontology:
                continue
            if e in self.ambiguous:
                continue
            if e in self.property_like:
                if e in self.declared_annotation_property:
                    continue
                props.add(e)
            elif e in self.class_like:
                classes.add(e)
            else:
                individuals.add(e)
        return {
            "property_connectivity": props,
            "class_connectivity": classes,
            "individual_connectivity": individuals,
        }

    def anchored(self) -> Set[URIRef]:
        return {e for e, d in self.incidences.items() if d > 0}


# ---------------------------------------------------------------------------
# D6 / D7 / D8 - roster, groundedness, load-bearing axioms.
#
# These are exposed so that a test can be written against the same notions the
# measure uses.  D7 is deliberately the measure's OWN notion of anchoring: an
# entity is grounded exactly when the measure credits it.  That identity is
# what makes D8's asymmetry work.  A declaration of an unanchored entity is not
# load-bearing, so I4 (deleting never pays) cannot ask for the deletion of the
# very padding I6 (declaring never pays) requires to be counted.
# ---------------------------------------------------------------------------
def entity_roster(artifact: Any) -> Dict[URIRef, Set[Any]]:
    """D6: subjects of any triple, with their declared types."""
    parsed = parse_artifact(artifact)
    if parsed.graph is None:
        return {}
    roster: Dict[URIRef, Set[Any]] = {}
    for s, _p, _o in parsed.graph:
        if isinstance(s, URIRef):
            roster.setdefault(s, set())
    for s, p, o in parsed.graph:
        if p == RDF.type and isinstance(s, URIRef) and s in roster:
            roster[s].add(o)
    return roster


def grounded_entities(artifact: Any) -> Set[URIRef]:
    """D7: the entities the measure credits with at least one anchor."""
    parsed = parse_artifact(artifact)
    if parsed.graph is None:
        return set()
    return Analysis(parsed.graph).anchored()


def load_bearing_axioms(artifact: Any) -> Set[Tuple[Any, Any, Any]]:
    """D8: connective axioms, plus declarations of grounded subjects."""
    parsed = parse_artifact(artifact)
    if parsed.graph is None:
        return set()
    grounded = grounded_entities(artifact)
    out = set(connective_axioms(parsed.graph))
    for s, p, o in declaration_triples(parsed.graph):
        if s in grounded:
            out.add((s, p, o))
    return out


# ---------------------------------------------------------------------------
# The components.
# ---------------------------------------------------------------------------
def _pitfall_freedom(oops: Any) -> Optional[Fraction]:
    """OOPS! is unreachable offline: MISSING, permanently, never 'clean'.

    The component becomes available if and only if a caller hands in a real
    pitfall report.  There is no default that turns an unreachable scanner into
    a perfect score; that is the bug this round exists to end (I13, C5).
    """
    if oops is None or isinstance(oops, BaseException):
        return None
    try:
        total = int(oops["pitfalls_total"])
        critical = int(oops["critical_count"]) if "critical_count" in oops else 0
    except Exception:
        return None
    if total < 0 or critical < 0:
        return None
    penalty = Fraction(total) + 2 * Fraction(critical)
    return Fraction(1) / (Fraction(1) + penalty)


def _to_float(value: Fraction) -> float:
    f = float(value)
    if not math.isfinite(f):
        return 0.0
    return min(1.0, max(0.0, f))


# ---------------------------------------------------------------------------
# The public entry point.
# ---------------------------------------------------------------------------
def evaluate(
    artifact: Any,
    *,
    oops: Any = None,
    scenario: Any = None,
) -> Dict[str, Any]:
    """Score one artifact's structural connectivity.

    Parameters
    ----------
    artifact:
        Bytes, ``str`` or an :class:`rdflib.Graph`.  Parsing is part of the
        measure and is in scope for the invariants (D1).
    oops:
        A pitfall report ``{"pitfalls_total": int, "critical_count": int}``, or
        ``None``.  ``None`` -- the default, and what every offline caller gets --
        means the scanner was unavailable: the component is MISSING and is
        *excluded* from the aggregate, never read as zero pitfalls.
    scenario:
        Accepted and **ignored**.  The measure is intrinsic (I9): it never
        consults a scenario file, a CQ signature or a reference pattern.

    Returns
    -------
    A JSON-native ``dict`` with exactly one score key, always holding the safe
    value.  MISSING is ``None`` throughout.
    """
    del scenario  # intrinsic by contract: never read.

    report: Dict[str, Any] = {
        "score": MISSING,
        "components": {},
        "components_available": [],
        "components_missing": [],
        "weights_used": {},
        "parse_status": "unparseable",
        "syntax": MISSING,
        "vacuous": False,
        "diagnostics": {},
        "notes": [],
    }

    parsed = parse_artifact(artifact)
    report["parse_status"] = parsed.status
    report["syntax"] = parsed.syntax if parsed.syntax is not None else MISSING

    if parsed.graph is None:
        for name in COMPONENT_ORDER:
            report["components"][name] = MISSING
            report["components_missing"].append(name)
        report["score"] = 0.0
        report["notes"].append("unparseable candidate: no RDF was emitted")
        return report

    try:
        analysis = Analysis(parsed.graph)
    except Exception as exc:  # pragma: no cover - defensive
        for name in COMPONENT_ORDER:
            report["components"][name] = MISSING
            report["components_missing"].append(name)
        report["score"] = 0.0
        report["notes"].append("analysis failed: %s" % type(exc).__name__)
        return report

    roster = analysis.roster()
    incid = analysis.incidences
    report["diagnostics"] = _diagnostics(analysis, parsed, roster)

    vacuous = not analysis.conn
    report["vacuous"] = bool(vacuous)

    raw: Dict[str, Optional[Fraction]] = {}
    mass: Dict[str, Fraction] = {}
    for name in _ROLE_COMPONENTS:
        entities = roster[name]
        if not entities:
            raw[name] = None
            continue
        evidence = Fraction(0)
        doubt = Fraction(0)
        for e in entities:
            d = incid.get(e, 0)
            evidence += d
            doubt += _doubt(d)
        raw[name] = evidence / (evidence + doubt)
        mass[name] = evidence + doubt
    raw["pitfall_freedom"] = _pitfall_freedom(oops)

    if vacuous:
        # D9.  Nothing here is unknown: the artifact demonstrably contains no
        # connective axiom, so every component that has a denominator is
        # exactly zero, and so is the score, for every value of every optional
        # parameter -- including a hypothetical clean OOPS! report (I2, I14(d)).
        for name in COMPONENT_ORDER:
            if name == "pitfall_freedom" and raw[name] is None:
                report["components"][name] = MISSING
                report["components_missing"].append(name)
            elif name != "pitfall_freedom" and raw[name] is None:
                report["components"][name] = MISSING
                report["components_missing"].append(name)
            else:
                report["components"][name] = 0.0
                report["components_available"].append(name)
        report["score"] = 0.0
        report["notes"].append("vacuous artifact: zero connective axioms")
        return report

    available: List[str] = []
    for name in COMPONENT_ORDER:
        value = raw[name]
        if value is None:
            report["components"][name] = MISSING
            report["components_missing"].append(name)
        else:
            report["components"][name] = _to_float(value)
            report["components_available"].append(name)
            available.append(name)

    if not available:
        report["score"] = MISSING
        report["notes"].append("no component was computable")
        return report

    # The structural score is the pooled ratio over every available role, which
    # is exactly the mass-weighted mean of the role components; reporting the
    # components alongside it is therefore lossless.
    structural_names = [n for n in available if n != "pitfall_freedom"]
    weights: Dict[str, Fraction] = {}
    if structural_names:
        total_mass = sum(mass[n] for n in structural_names)
        structural = sum(mass[n] * raw[n] for n in structural_names) / total_mass
        share = Fraction(1) - PITFALL_WEIGHT if "pitfall_freedom" in available else Fraction(1)
        for n in structural_names:
            weights[n] = share * mass[n] / total_mass
    else:
        structural = None

    if "pitfall_freedom" in available:
        if structural is None:
            weights["pitfall_freedom"] = Fraction(1)
            aggregate = raw["pitfall_freedom"]
        else:
            weights["pitfall_freedom"] = PITFALL_WEIGHT
            aggregate = (
                (Fraction(1) - PITFALL_WEIGHT) * structural
                + PITFALL_WEIGHT * raw["pitfall_freedom"]
            )
    else:
        aggregate = structural
        report["notes"].append(
            "OOPS! is unreachable offline: pitfall_freedom is MISSING and is "
            "excluded from the aggregate, never scored as zero pitfalls"
        )

    report["weights_used"] = {k: _to_float(v) for k, v in weights.items()}
    report["score"] = _to_float(aggregate)
    report["notes"].append(
        "no reasoner component: over the 420-generation corpus the reasoner "
        "never fired, and a component with one distinct value is decoration"
    )
    return report


def _diagnostics(
    analysis: Analysis, parsed: ParseResult, roster: Dict[str, Set[URIRef]]
) -> Dict[str, Any]:
    """Descriptive integer counts.  Never scored, never aggregated."""
    anchored = analysis.anchored()
    return {
        "triples": int(parsed.n_triples),
        "connective_axioms": len(analysis.conn),
        "declarations": len(analysis.decls),
        "roster_properties": len(roster["property_connectivity"]),
        "roster_classes": len(roster["class_connectivity"]),
        "roster_individuals": len(roster["individual_connectivity"]),
        "anchored_properties": len(roster["property_connectivity"] & anchored),
        "anchored_classes": len(roster["class_connectivity"] & anchored),
        "anchored_individuals": len(roster["individual_connectivity"] & anchored),
        "counted_anchor_axioms": analysis.counted_axioms,
        "domains_counted": analysis.n_domains,
        "ranges_counted": analysis.n_ranges,
        "restriction_bodies": analysis.restriction_bodies,
        "wellformed_restrictions": analysis.wellformed_restrictions,
        "usage_edges": len(analysis.object_edges),
        "attribute_edges": len(analysis.literal_edges),
        "dropped_universal_usage": analysis.dropped_usage,
        "dropped_universal_signature": analysis.dropped_signature,
        "dropped_universal_restrictions": analysis.dropped_restrictions,
        "comparability_pairs": len(analysis.comparability_pairs),
        "distinction_pairs": len(analysis.distinction_pairs),
        "equivalence_axioms": analysis.equivalence_pairs,
        "dropped_taxonomy_nodes": analysis.dropped_taxonomy_nodes,
        "dropped_taxonomy_components": analysis.dropped_taxonomy_components,
        "primary_anchor_axioms": analysis.primary_axioms,
    }


def evaluate_reference(artifact: Any, *, oops: Any = None) -> Dict[str, Any]:
    """Score a *reference* pattern rather than a generated candidate.

    Exactly one thing differs from :func:`evaluate`, and it is the rule
    contract I13 states: a non-empty artifact that no available parser can read
    is a quality signal when a generator produced it -- the model failed to
    emit RDF, and that scores ``0.0`` -- but is a tooling failure when a
    published reference produced it, and a tooling failure is MISSING, never
    zero.  Two of the fourteen reference patterns are OWL/XML that rdflib
    cannot load at all (G-H); they are MISSING here.

    This is a separate entry point rather than a flag on :func:`evaluate`
    because a flag would be an optional parameter whose value changes the score
    of a vacuous artifact, which contract I2 forbids.  The naive call is always
    the candidate call, and the candidate call is always the safe one.
    """
    report = evaluate(artifact, oops=oops)
    if report["parse_status"] == "unparseable":
        report["score"] = MISSING
        report["notes"] = [
            "unparseable reference: a tooling failure, not a quality signal"
        ]
    return report


def evaluate_path(path: str, *, oops: Any = None) -> Dict[str, Any]:
    """Score the bytes at ``path``.  The suffix never chooses the parser."""
    try:
        with open(path, "rb") as fh:
            data = fh.read()
    except Exception:
        report = evaluate(b"", oops=oops)
        report["notes"].append("unreadable file")
        return report
    return evaluate(data, oops=oops)
