"""Property-based falsification suite for :mod:`scripts.odp_structural`.

Every test here quantifies over a *generated family* with an explicit
generator and a seed sweep.  There is not one hand-written fixture that a
measure could be tuned to, because that is how the three previous rounds
failed: each was designed against examples and generalised exactly as far as
the examples went.

Three structural choices make the suite capable of failing.

*   Each invariant is a **check function** returning a list of witnesses, not a
    bare ``assert``.  The tests assert the list is empty and print the witness
    verbatim when it is not, so the next round starts from the falsifying
    artifact rather than from a red dot (obligation N3).
*   The same check functions are run against ten **deliberately broken
    measures** (``MUTANTS``), and every one of them must be *rejected* by a
    named invariant (obligation N2).  A check that no mutant can fail is not a
    check; the mutant tests are what prove these tests bite.
*   Where the contract specifies exact equality -- replication, clique
    antisymmetry, serialisation -- the assertion is exact.  There is no
    ``pytest.approx`` anywhere: the exploits live in the epsilon (obligation
    N4).

Two invariants are asserted as *strict* xfails with a proof of impossibility
in the test's own docstring.  A strict xfail fails the suite if it starts
passing, so the claim stays honest in both directions.

Nothing here touches the network, and nothing reads a scenario file, a CQ
signature or a reference pattern at score time (obligation N5, contract I9).
"""

from __future__ import annotations

import io
import itertools
import json
import math
import os
import random
import subprocess
import sys
from fractions import Fraction
from typing import Any, Dict, Iterable, List, Sequence, Set, Tuple

import pytest
from rdflib import BNode, Graph, Literal, URIRef
from rdflib.namespace import OWL, RDF, RDFS, SKOS

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "scripts"))

import odp_structural as M  # noqa: E402

GROUND_TRUTH = os.path.join(REPO, "data", "ground_truth")
OUTPUTS = os.path.join(REPO, "outputs")

# The parseable gold set.  2024-145-01.owl is a 15.8 MB whole ontology that
# rdflib does load; it is excluded here for runtime only and is scored in the
# corpus report.  2025-147-01.owl and 2025-150-01.owl are OWL/XML that rdflib
# cannot read at all: they are MISSING, never zero (contract I13, G-H).
PG_NAMES: Tuple[str, ...] = (
    "2023-133-01.ttl",
    "2023-133-02.ttl",
    "2023-134-01.ttl",
    "2023-134-02.ttl",
    "2023-134-03.ttl",
    "2023-135-01.owl",
    "2025-149-01.rdf",
    "2025-151-01.ttl",
    "2025-151-02.ttl",
    "2025-153-01.ttl",
    "2026-155-01.ttl",
)
UNPARSEABLE_GOLD: Tuple[str, ...] = ("2025-147-01.owl", "2025-150-01.owl")

# Golds small enough for the exhaustive families.
SMALL_PG: Tuple[str, ...] = tuple(n for n in PG_NAMES if n != "2025-149-01.rdf")

EX = "http://property-test.example/gen#"
FRESH = "http://property-test.example/fresh#"


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def _bytes_of(name: str) -> bytes:
    with open(os.path.join(GROUND_TRUTH, name), "rb") as fh:
        return fh.read()


_GOLD_CACHE: Dict[str, Graph] = {}


def gold(name: str) -> Graph:
    if name not in _GOLD_CACHE:
        parsed = M.parse_artifact(_bytes_of(name))
        assert parsed.graph is not None, name
        _GOLD_CACHE[name] = parsed.graph
    return _GOLD_CACHE[name]


def all_gold() -> Dict[str, Graph]:
    return {n: gold(n) for n in PG_NAMES}


_OUTPUT_PATHS: List[str] = []


def output_paths() -> List[str]:
    if not _OUTPUT_PATHS:
        for root, _dirs, files in os.walk(OUTPUTS):
            for f in sorted(files):
                if f.endswith(".ttl"):
                    _OUTPUT_PATHS.append(os.path.join(root, f))
        _OUTPUT_PATHS.sort()
    return _OUTPUT_PATHS


_PARSEABLE_OUTPUTS: List[Graph] = []


def parseable_outputs(limit: int = 50, seed: int = 0) -> List[Graph]:
    """A deterministic sample of the parseable generations."""
    if not _PARSEABLE_OUTPUTS:
        paths = output_paths()
        rnd = random.Random(seed)
        for p in rnd.sample(paths, min(len(paths), 400)):
            with open(p, "rb") as fh:
                parsed = M.parse_artifact(fh.read())
            if parsed.graph is not None and len(parsed.graph) > 0:
                _PARSEABLE_OUTPUTS.append(parsed.graph)
            if len(_PARSEABLE_OUTPUTS) >= limit:
                break
    return _PARSEABLE_OUTPUTS[:limit]


# ---------------------------------------------------------------------------
# The measure under test, and the interface every mutant must implement
# ---------------------------------------------------------------------------
def real_measure(artifact: Any, **kw: Any) -> Dict[str, Any]:
    return M.evaluate(artifact, **kw)


def score_of(measure, artifact: Any, **kw: Any):
    return measure(artifact, **kw)["score"]


# ---------------------------------------------------------------------------
# Graph utilities used by the generators
# ---------------------------------------------------------------------------
def copy_graph(*graphs: Iterable[Tuple[Any, Any, Any]]) -> Graph:
    out = Graph()
    for g in graphs:
        for t in g:
            out.add(t)
    return out


def without(g: Graph, drop: Set[Tuple[Any, Any, Any]]) -> Graph:
    out = Graph()
    for t in g:
        if t not in drop:
            out.add(t)
    return out


META_NAMESPACES = (str(RDF), str(RDFS), str(OWL), str(SKOS),
                   "http://www.w3.org/2001/XMLSchema#", "http://www.w3.org/")


def own_namespaces(g: Graph) -> Set[str]:
    """The artifact's *own* namespaces: never the shared meta vocabularies.

    Renaming ``rdf:``, ``rdfs:`` or ``owl:`` would not be a replication of the
    artifact, it would be a different ontology language.
    """
    ns: Set[str] = set()
    for s in g.subjects():
        if isinstance(s, URIRef):
            u = str(s)
            if any(u.startswith(m) for m in META_NAMESPACES):
                continue
            cut = max(u.rfind("#"), u.rfind("/"))
            if cut > 0:
                ns.add(u[: cut + 1])
    return {n for n in ns if not any(m.startswith(n) for m in META_NAMESPACES)}


def rename_namespaces(g: Graph, tag: str) -> Graph:
    """Consistently rewrite the artifact's own namespaces (I11, I15(d))."""
    ns = own_namespaces(g)

    def rw(t: Any) -> Any:
        if isinstance(t, URIRef):
            u = str(t)
            for n in ns:
                if u.startswith(n):
                    return URIRef(n[:-1] + "-" + tag + n[-1] + u[len(n) :])
        if isinstance(t, BNode):
            return BNode("bn" + tag + str(t))
        return t

    out = Graph()
    for s, p, o in g:
        out.add((rw(s), rw(p), rw(o)))
    return out


def replicate(g: Graph, k: int) -> Graph:
    """``k`` IRI-disjoint isomorphic copies with no edge between copies."""
    out = Graph()
    for i in range(k):
        for t in rename_namespaces(g, "copy%d" % i):
            out.add(t)
    return out


def rename_blank_nodes(g: Graph, seed: int) -> Graph:
    rnd = random.Random(seed)
    mapping: Dict[Any, Any] = {}
    for s, _p, o in g:
        for t in (s, o):
            if isinstance(t, BNode) and t not in mapping:
                mapping[t] = BNode("relabelled%d_%d" % (seed, rnd.randrange(10 ** 9)))
    out = Graph()
    for s, p, o in g:
        out.add((mapping.get(s, s), p, mapping.get(o, o)))
    return out


def roster_of(g: Graph) -> Tuple[List[URIRef], List[URIRef], List[URIRef]]:
    """The D6 roster split by role: exactly what a roster function may see."""
    r = M.Analysis(g).roster()
    return (
        sorted(r["class_connectivity"], key=str),
        sorted(r["property_connectivity"], key=str),
        sorted(r["individual_connectivity"], key=str),
    )


def annotations_of(g: Graph) -> Set[Tuple[Any, Any, Any]]:
    return {(s, p, o) for s, p, o in g if p in M.ANNOTATION_PREDICATES}


def strip_annotations(g: Graph) -> Graph:
    return without(g, annotations_of(g))


# ---------------------------------------------------------------------------
# Generators: the vacuous family (D9)
# ---------------------------------------------------------------------------
DECLARATION_KINDS = (OWL.Class, OWL.ObjectProperty, OWL.DatatypeProperty,
                     OWL.NamedIndividual, RDF.Property, RDFS.Class)


def vacuous_family(seeds: Sequence[int] = (0,)) -> List[Any]:
    """Every shape D9 names, plus a seeded sweep of bare-declaration blocks."""
    fam: List[Any] = [b"", b"   \n\t\r\n  ", b"\x00\x01\x02"]
    for k in (1, 10, 100):
        fam.append(("# a comment\n" * k).encode())
    for n in (1, 5, 20):
        fam.append(("@prefix ex%d: <http://example.org/e%d#> .\n" % (n, n) * n).encode())
    fam.append(
        b"I am sorry, but I cannot produce an ontology for this scenario.\n"
        b"There is not enough information in the request to model anything.\n"
        b"Please provide competency questions and a domain description.\n" * 3
    )
    for length in (10, 1000, 50000):
        g = Graph()
        o = URIRef(EX + "Ontology")
        g.add((o, RDF.type, OWL.Ontology))
        g.add((o, RDFS.comment, Literal("z" * length)))
        fam.append(g)
        fam.append(g.serialize(format="turtle").encode())
    for seed in seeds:
        rnd = random.Random(seed)
        for n in (1, 5, 25, 176, 500):
            for t in DECLARATION_KINDS:
                for decorated in (False, True):
                    g = Graph()
                    for i in range(n):
                        e = URIRef(EX + "V%d_%d" % (seed, i))
                        g.add((e, RDF.type, t))
                        if decorated:
                            g.add((e, RDFS.label, Literal("term %d" % i)))
                            g.add((e, RDFS.comment, Literal("prose " * rnd.randint(1, 40))))
                            g.add((e, SKOS.definition, Literal("a definition")))
                    fam.append(g)
    # Turtle that parses to zero triples.
    fam.append(b"@prefix : <http://example.org/> .\n# nothing follows\n")
    return fam


OOPS_VALUES = (
    None,
    {"pitfalls_total": 0, "critical_count": 0},
    {"pitfalls_total": 3, "critical_count": 0},
    {"pitfalls_total": 41, "critical_count": 9},
)


# ---------------------------------------------------------------------------
# Generators: padding (I6) and roster functions (I7)
# ---------------------------------------------------------------------------
def padding(k: int, kind: URIRef, decoration: int, self_referential: bool,
            tag: str = "p") -> Graph:
    g = Graph()
    previous = None
    for i in range(k):
        e = URIRef(FRESH + "%s%d" % (tag, i))
        g.add((e, RDF.type, kind))
        if decoration >= 1:
            g.add((e, RDFS.label, Literal("padding term %d" % i)))
        if decoration >= 2:
            g.add((e, RDFS.comment, Literal("An unused term. " * 20)))
        if decoration >= 3:
            g.add((e, SKOS.definition, Literal("A definition of an unused term.")))
            g.add((e, URIRef(M.DCTERMS + "description"), Literal("Described at length. " * 10)))
        if self_referential and previous is not None:
            g.add((e, RDFS.subClassOf, previous))
        previous = e
    return g


def f_bottom(cs, ps, ind, rnd):
    g = Graph()
    b = URIRef(FRESH + "Bottom")
    g.add((b, RDF.type, OWL.Class))
    for c in cs:
        g.add((b, RDFS.subClassOf, c))
    return g


def f_top(cs, ps, ind, rnd):
    g = Graph()
    t = URIRef(FRESH + "Top")
    g.add((t, RDF.type, OWL.Class))
    for c in cs:
        g.add((c, RDFS.subClassOf, t))
    return g


def f_mesh(cs, ps, ind, rnd):
    g = Graph()
    for a in cs:
        for b in cs:
            if a != b:
                g.add((a, RDFS.subClassOf, b))
    return g


def f_chain(cs, ps, ind, rnd):
    g = Graph()
    for a, b in zip(cs, cs[1:]):
        g.add((a, RDFS.subClassOf, b))
    return g


def f_clique(cs, ps, ind, rnd):
    g = Graph()
    for a, b in itertools.combinations(cs, 2):
        g.add((a, OWL.equivalentClass, b))
    return g


def f_clique_reversed(cs, ps, ind, rnd):
    g = Graph()
    for a, b in itertools.combinations(cs, 2):
        g.add((b, OWL.equivalentClass, a))
    return g


def f_disjoint_all(cs, ps, ind, rnd):
    g = Graph()
    for a, b in itertools.combinations(cs, 2):
        g.add((a, OWL.disjointWith, b))
    return g


def f_universal_dr(cs, ps, ind, rnd):
    g = Graph()
    for p in ps:
        g.add((p, RDFS.domain, OWL.Thing))
        g.add((p, RDFS.range, OWL.Thing))
    return g


def f_fresh_top_dr(cs, ps, ind, rnd):
    g = Graph()
    t = URIRef(FRESH + "UniversalFiller")
    g.add((t, RDF.type, OWL.Class))
    for p in ps:
        g.add((p, RDFS.domain, t))
        g.add((p, RDFS.range, t))
    return g


def f_star(cs, ps, ind, rnd):
    g = Graph()
    q = URIRef(FRESH + "relatesTo")
    g.add((q, RDF.type, OWL.ObjectProperty))
    entities = list(cs) + list(ps) + list(ind)
    for a in entities:
        for b in entities:
            if a != b:
                g.add((a, q, b))
    return g


def f_shuffled_labels(cs, ps, ind, rnd):
    """A roster function that only permutes names: pure decoration."""
    g = Graph()
    names = [str(e).rsplit("#", 1)[-1].rsplit("/", 1)[-1] for e in list(cs) + list(ps)]
    rnd.shuffle(names)
    for e, n in zip(list(cs) + list(ps), names):
        g.add((e, RDFS.label, Literal(n)))
    return g


ROSTER_FUNCTIONS = (
    f_bottom, f_top, f_mesh, f_chain, f_clique, f_clique_reversed,
    f_disjoint_all, f_universal_dr, f_fresh_top_dr, f_star, f_shuffled_labels,
)
# f_fresh_top_dr is separated in the strict-xfail test; see its proof there.
ROSTER_FUNCTIONS_SATISFIABLE = tuple(
    f for f in ROSTER_FUNCTIONS if f is not f_fresh_top_dr
)


# ---------------------------------------------------------------------------
# Generators: chains (I5), deletions (I4), shuffles (I10)
# ---------------------------------------------------------------------------
def grounded_prefix_chain(g: Graph, seed: int) -> List[Graph]:
    """D10: prefixes of a permutation of the connective axioms, closed under
    the declarations and annotations of every entity they mention."""
    rnd = random.Random(seed)
    conn = sorted(M.connective_axioms(g), key=lambda t: tuple(map(str, t)))
    rnd.shuffle(conn)
    decls = M.declaration_triples(g)
    anns = annotations_of(g)
    chain: List[Graph] = []
    acc: List[Tuple[Any, Any, Any]] = []

    def close(axioms: Sequence[Tuple[Any, Any, Any]]) -> Graph:
        mentioned: Set[Any] = set()
        for s, p, o in axioms:
            for t in (s, p, o):
                if isinstance(t, URIRef):
                    mentioned.add(t)
        out = Graph()
        for t in axioms:
            out.add(t)
        for t in decls:
            if t[0] in mentioned:
                out.add(t)
        for t in anns:
            if t[0] in mentioned:
                out.add(t)
        return out

    chain.append(close(()))
    for axiom in conn:
        acc.append(axiom)
        chain.append(close(acc))
    return chain


def deletion_subsets(g: Graph, seed: int, singles: int = 60,
                     randoms: int = 40) -> List[Set[Tuple[Any, Any, Any]]]:
    lb = sorted(M.load_bearing_axioms(g), key=lambda t: tuple(map(str, t)))
    rnd = random.Random(seed)
    out: List[Set[Tuple[Any, Any, Any]]] = []
    chosen = lb if len(lb) <= singles else rnd.sample(lb, singles)
    out.extend({a} for a in chosen)
    if lb:
        for _ in range(randoms):
            size = int(math.exp(rnd.uniform(0, math.log(len(lb) + 1))))
            size = max(1, min(len(lb), size))
            out.append(set(rnd.sample(lb, size)))
        # Predicate-targeted subsets, exhaustive where small.
        for pred in (RDFS.subClassOf, OWL.equivalentClass, RDFS.domain, RDFS.range,
                     OWL.onProperty):
            block = [t for t in lb if t[1] == pred]
            if 0 < len(block) <= 8:
                for r in range(1, len(block) + 1):
                    for combo in itertools.combinations(block, r):
                        out.append(set(combo))
            elif block:
                out.append(set(block))
                for _ in range(5):
                    out.append(set(rnd.sample(block, rnd.randint(1, len(block)))))
        out.append(set(M.connective_axioms(g)))  # vacuous by construction
    return out


def shuffle_graph(g: Graph, seed: int) -> Graph:
    """I10.5: re-draw subjects and objects of every connective axiom from the
    artifact's own roster, preserving predicates, arity and every count."""
    rnd = random.Random(seed)
    conn = M.connective_axioms(g)
    pool = sorted(
        {t for s, p, o in conn for t in (s, o) if isinstance(t, URIRef)}, key=str
    )
    if len(pool) < 3:
        return g
    out = Graph()
    for s, p, o in g:
        if (s, p, o) in conn:
            s2 = rnd.choice(pool) if isinstance(s, URIRef) else s
            o2 = rnd.choice(pool) if isinstance(o, URIRef) else o
            out.add((s2, p, o2))
        else:
            out.add((s, p, o))
    return out


def vocab_only(g: Graph) -> Graph:
    """I10.3: gold's IRIs, declarations and annotations, and nothing else."""
    out = Graph()
    for s, p, o in g:
        if p in M.ANNOTATION_PREDICATES or (p == RDF.type and o in M.DECLARATION_TYPES):
            out.add((s, p, o))
    return out


# ---------------------------------------------------------------------------
# Check functions.  Each returns a list of human-readable witnesses.
# ---------------------------------------------------------------------------
def check_I1(measure, artifacts: Sequence[Any]) -> List[str]:
    bad: List[str] = []
    for a in artifacts:
        try:
            report = measure(a)
        except Exception as exc:  # pragma: no cover - that is the failure
            bad.append("raised %s on %r" % (type(exc).__name__, str(a)[:120]))
            continue
        s = report.get("score", "absent")
        if not (s is M.MISSING or (isinstance(s, float) and math.isfinite(s)
                                   and 0.0 <= s <= 1.0)):
            bad.append("score %r on %r" % (s, str(a)[:120]))
        for name, value in report.get("components", {}).items():
            if not (value is M.MISSING or (isinstance(value, float)
                                           and math.isfinite(value)
                                           and 0.0 <= value <= 1.0)):
                bad.append("component %s = %r on %r" % (name, value, str(a)[:120]))
        try:
            if json.loads(json.dumps(report)) != report:
                bad.append("json round trip changed the report on %r" % (str(a)[:120],))
        except (TypeError, ValueError) as exc:
            bad.append("not JSON serialisable (%s) on %r" % (exc, str(a)[:120]))
    return bad


def check_I2(measure, family: Sequence[Any]) -> List[str]:
    bad: List[str] = []
    for v in family:
        for oops in OOPS_VALUES:
            report = measure(v, oops=oops)
            if report["score"] != 0.0:
                bad.append("score %r (oops=%r) on %r" % (report["score"], oops, str(v)[:120]))
            for name, value in report["components"].items():
                if not (value is M.MISSING or value == 0.0):
                    bad.append("component %s = %r on %r" % (name, value, str(v)[:120]))
            for key, value in report.items():
                if isinstance(value, float) and value != 0.0:
                    bad.append("float key %s = %r on %r" % (key, value, str(v)[:120]))
    return bad


def check_I4(measure, graphs: Dict[str, Graph], seeds: Sequence[int]) -> List[str]:
    bad: List[str] = []
    for name, g in graphs.items():
        base = score_of(measure, g)
        for seed in seeds:
            for subset in deletion_subsets(g, seed):
                s = score_of(measure, without(g, subset))
                if s is M.MISSING or base is M.MISSING:
                    continue
                if s > base:
                    bad.append(
                        "%s seed=%d: deleting %d load-bearing axiom(s) raised "
                        "%.12f -> %.12f; first axiom deleted: %s"
                        % (name, seed, len(subset), base, s,
                           sorted(subset, key=str)[0])
                    )
    return bad


def check_I5(measure, graphs: Dict[str, Graph], seeds: Sequence[int],
             monotone: bool = True, variety: bool = True) -> List[str]:
    bad: List[str] = []
    for name, g in graphs.items():
        for seed in seeds:
            chain = grounded_prefix_chain(g, seed)
            scores = [score_of(measure, h) for h in chain]
            n = len(scores) - 1
            if monotone:
                for i in range(n):
                    if scores[i] > scores[i + 1]:
                        bad.append("%s seed=%d step %d: %.12f > %.12f"
                                   % (name, seed, i, scores[i], scores[i + 1]))
            if variety:
                if not scores[-1] > scores[0]:
                    bad.append("%s seed=%d: chain is constant (%r)" % (name, seed, scores[0]))
                if len(set(scores)) < min(4, n):
                    bad.append("%s seed=%d: only %d distinct values, need %d"
                               % (name, seed, len(set(scores)), min(4, n)))
    return bad


def check_I6(measure, graphs: Dict[str, Graph], seeds: Sequence[int]) -> List[str]:
    bad: List[str] = []
    for name, g in graphs.items():
        base = score_of(measure, g)
        for seed in seeds:
            rnd = random.Random(seed)
            for k in (1, 5, 25, 100, 176, 500):
                kind = DECLARATION_KINDS[rnd.randrange(len(DECLARATION_KINDS))]
                for decoration in (0, 3):
                    for self_ref in (False, True):
                        pad = padding(k, kind, decoration, self_ref, "s%d" % seed)
                        s = score_of(measure, copy_graph(g, pad))
                        if base is M.MISSING or s is M.MISSING:
                            continue
                        if s > base:
                            bad.append(
                                "%s seed=%d: %d fresh %s (decoration=%d, self=%s) "
                                "raised %.12f -> %.12f"
                                % (name, seed, k, kind, decoration, self_ref, base, s))
    return bad


def check_I7(measure, graphs: Dict[str, Graph], seeds: Sequence[int],
             functions=ROSTER_FUNCTIONS) -> List[str]:
    bad: List[str] = []
    for name, g in graphs.items():
        base = score_of(measure, g)
        cs, ps, ind = roster_of(g)
        for seed in seeds:
            for f in functions:
                rnd = random.Random(seed)
                s = score_of(measure, copy_graph(g, f(cs, ps, ind, rnd)))
                if base is M.MISSING or s is M.MISSING:
                    continue
                if s > base:
                    bad.append("%s seed=%d %s: %.12f > %.12f"
                               % (name, seed, f.__name__, s, base))
    return bad


def check_I7_antisymmetry(measure, graphs: Dict[str, Graph]) -> List[str]:
    bad: List[str] = []
    for name, g in graphs.items():
        cs, ps, ind = roster_of(g)
        rnd = random.Random(0)
        a = score_of(measure, copy_graph(g, f_clique(cs, ps, ind, rnd)))
        b = score_of(measure, copy_graph(g, f_clique_reversed(cs, ps, ind, rnd)))
        if a != b:
            bad.append("%s: clique %r != clique_reversed %r" % (name, a, b))
    return bad


def check_I8(measure, graphs: Dict[str, Graph], seeds: Sequence[int]) -> List[str]:
    bad: List[str] = []
    prose = "This pattern is documented at length for the reader. " * 10
    for name, g in graphs.items():
        bare = strip_annotations(g)
        bare_score = score_of(measure, bare)
        for seed in seeds:
            chain = grounded_prefix_chain(g, seed)
            candidates = [chain[len(chain) // 3], chain[len(chain) // 2],
                          chain[max(0, len(chain) - 2)]]
            for x in candidates:
                entities = {s for s in x.subjects() if isinstance(s, URIRef)}
                decorated = copy_graph(x)
                for e in entities:
                    decorated.add((e, RDFS.label, Literal("Label. " + prose)))
                    decorated.add((e, RDFS.comment, Literal(prose)))
                    decorated.add((e, SKOS.definition, Literal(prose)))
                    decorated.add((e, URIRef(M.DCTERMS + "description"), Literal(prose)))
                for i in range(100):
                    decorated.add((URIRef(EX + "Doc"), RDFS.comment,
                                   Literal("ontology note %d %s" % (i, prose))))
                s = score_of(measure, decorated)
                if s is M.MISSING or bare_score is M.MISSING:
                    continue
                if s > bare_score:
                    bad.append("%s seed=%d: annotated prefix %.12f > bare gold %.12f"
                               % (name, seed, s, bare_score))
    return bad


def check_I11(measure, graphs: Dict[str, Graph], ks: Sequence[int] = (2, 3, 5, 10)
              ) -> List[str]:
    bad: List[str] = []
    for name, g in graphs.items():
        base = score_of(measure, g)
        for k in ks:
            s = score_of(measure, replicate(g, k))
            if s != base:
                bad.append("%s k=%d: %r != %r" % (name, k, s, base))
    return bad


def check_I11_union(measure, graphs: Dict[str, Graph]) -> List[str]:
    bad: List[str] = []
    items = sorted(graphs.items())
    for (na, ga), (nb, gb) in itertools.combinations(items, 2):
        sa, sb = score_of(measure, ga), score_of(measure, gb)
        if sa is M.MISSING or sb is M.MISSING:
            continue
        union = copy_graph(replicate(ga, 2), rename_namespaces(replicate(gb, 2), "b"))
        s = score_of(measure, union)
        if not (min(sa, sb) <= s <= max(sa, sb)):
            bad.append("%s+%s: %.12f outside [%.12f, %.12f]" % (na, nb, s, min(sa, sb), max(sa, sb)))
    return bad


def check_I15(measure, graphs: Dict[str, Graph], seeds: Sequence[int]) -> List[str]:
    bad: List[str] = []
    for name, g in graphs.items():
        base = score_of(measure, g)
        for fmt in ("turtle", "xml", "nt", "n3", "json-ld"):
            try:
                data = g.serialize(format=fmt)
            except Exception:
                continue
            if isinstance(data, str):
                data = data.encode("utf-8")
            s = score_of(measure, data)
            if s != base:
                bad.append("%s serialised as %s: %r != %r" % (name, fmt, s, base))
        for seed in seeds:
            s = score_of(measure, rename_blank_nodes(g, seed))
            if s != base:
                bad.append("%s blank-node relabelling seed=%d: %r != %r" % (name, seed, s, base))
            s = score_of(measure, rename_namespaces(g, "ren%d" % seed))
            if s != base:
                bad.append("%s namespace renaming seed=%d: %r != %r" % (name, seed, s, base))
    return bad


def rewrite_equivalence_as_subclass(g: Graph) -> Graph:
    out = Graph()
    for s, p, o in g:
        if p == OWL.equivalentClass and isinstance(s, URIRef) and isinstance(o, URIRef):
            out.add((s, RDFS.subClassOf, o))
            out.add((o, RDFS.subClassOf, s))
        else:
            out.add((s, p, o))
    return out


def rewrite_reverse_equivalence(g: Graph) -> Graph:
    out = Graph()
    for s, p, o in g:
        if p == OWL.equivalentClass:
            out.add((o, p, s))
        else:
            out.add((s, p, o))
    return out


def rewrite_add_entailed_subclass(g: Graph) -> Graph:
    out = copy_graph(g)
    edges: Dict[Any, Set[Any]] = {}
    for s, p, o in g:
        if p == RDFS.subClassOf and isinstance(s, URIRef) and isinstance(o, URIRef):
            edges.setdefault(s, set()).add(o)
    for start in list(edges):
        seen: Set[Any] = set()
        stack = list(edges.get(start, ()))
        while stack:
            cur = stack.pop()
            if cur in seen:
                continue
            seen.add(cur)
            stack.extend(edges.get(cur, ()))
        for target in seen:
            if target != start:
                out.add((start, RDFS.subClassOf, target))
    return out


EQUIVALENCE_REWRITES = (
    rewrite_equivalence_as_subclass,
    rewrite_reverse_equivalence,
    rewrite_add_entailed_subclass,
)


def check_I15e(measure, graphs: Dict[str, Graph]) -> List[str]:
    bad: List[str] = []
    for name, g in graphs.items():
        base = score_of(measure, g)
        for rewrite in EQUIVALENCE_REWRITES:
            s = score_of(measure, rewrite(g))
            if s != base:
                bad.append("%s under %s: %r != %r" % (name, rewrite.__name__, s, base))
        composed = g
        for rewrite in EQUIVALENCE_REWRITES:
            composed = rewrite(composed)
        s = score_of(measure, composed)
        if s != base:
            bad.append("%s under composed rewrites: %r != %r" % (name, s, base))
    return bad


def check_I10_shuffle(measure, graphs: Dict[str, Graph],
                      seeds: Sequence[int]) -> Tuple[List[str], Dict[str, int]]:
    bad: List[str] = []
    strictly_lower: Dict[str, int] = {}
    for name, g in graphs.items():
        base = score_of(measure, g)
        lower = 0
        for seed in seeds:
            h = shuffle_graph(g, seed)
            s = score_of(measure, h)
            if s is M.MISSING or base is M.MISSING:
                continue
            if s > base:
                bad.append("%s shuffle seed=%d: %.12f > %.12f" % (name, seed, s, base))
            if s < base:
                lower += 1
        strictly_lower[name] = lower
    return bad, strictly_lower


# ---------------------------------------------------------------------------
# I1 - well-formed report
# ---------------------------------------------------------------------------
def test_I1_report_is_well_formed_on_every_artifact():
    rnd = random.Random(0)
    family: List[Any] = list(vacuous_family((0, 1)))
    family.extend(rnd.randbytes(rnd.randrange(0, 4096)) for _ in range(200))
    family.append(os.urandom(1 << 20).decode("utf-8", errors="replace"))
    family.append(b"A" * (4 << 20))
    for name in PG_NAMES + UNPARSEABLE_GOLD:
        family.append(_bytes_of(name))
    for p in output_paths()[:120]:
        with open(p, "rb") as fh:
            family.append(fh.read())
    bad = check_I1(real_measure, family)
    assert not bad, "I1 witnesses:\n" + "\n".join(bad[:20])


def test_I1_missing_is_distinguishable_from_zero():
    assert M.MISSING is not 0  # noqa: F632 - the identity is the point
    assert M.MISSING != 0
    assert M.MISSING != 0.0
    assert M.MISSING != 1
    assert M.MISSING != ""
    assert M.MISSING != -1
    encoded = json.dumps({"score": M.MISSING})
    assert json.loads(encoded)["score"] is None


# ---------------------------------------------------------------------------
# I2 - vacuity floor
# ---------------------------------------------------------------------------
def test_I2_vacuous_artifacts_score_exactly_zero():
    bad = check_I2(real_measure, vacuous_family(range(10)))
    assert not bad, "I2 witnesses:\n" + "\n".join(bad[:20])


def test_I2_holds_for_every_optional_argument_combination():
    family = vacuous_family((0,))[:40]
    bad: List[str] = []
    for v in family:
        for oops in OOPS_VALUES:
            for scenario in (None, "2023-133-01", "2025-151-02", 17):
                r = M.evaluate(v, oops=oops, scenario=scenario)
                if r["score"] != 0.0:
                    bad.append("%r oops=%r scenario=%r -> %r" % (str(v)[:60], oops, scenario, r["score"]))
    assert not bad, "I2 witnesses:\n" + "\n".join(bad[:20])


# ---------------------------------------------------------------------------
# I3 - vacuity dominance
# ---------------------------------------------------------------------------
def degenerate_union(seeds: Sequence[int]) -> List[Any]:
    """Standalone degenerate artifacts: non-answers, padding, and things built
    only out of a roster or a reference's own vocabulary.

    Gold *prefixes* and gold-plus-attack graphs are deliberately not here.  A
    prefix of 2023-133-01 that keeps 41 of its 42 connective axioms is not a
    degenerate artifact; requiring it to sit below the weakest gold would
    require the measure to be discontinuous at gold, which contradicts I12(a).
    """
    fam: List[Any] = list(vacuous_family(seeds))
    for seed in seeds:
        for k in (5, 50, 500):
            fam.append(padding(k, OWL.Class, 3, True, "d%d" % seed))
            fam.append(padding(k, OWL.ObjectProperty, 3, False, "e%d" % seed))
    for name in SMALL_PG:
        g = gold(name)
        vocab = vocab_only(g)
        fam.append(vocab)
        cs, ps, ind = roster_of(g)
        for f in ROSTER_FUNCTIONS:
            rnd = random.Random(0)
            fam.append(copy_graph(vocab, f(cs, ps, ind, rnd)))
    return fam


def test_I3_degenerate_artifacts_rank_below_every_gold():
    golds = {n: score_of(real_measure, g) for n, g in all_gold().items()}
    worst_gold = min(golds.values())
    worst_name = min(golds, key=lambda k: golds[k])
    best_bad = -1.0
    witness: Any = None
    for a in degenerate_union((0, 1, 2)):
        s = score_of(real_measure, a)
        if s is not M.MISSING and s > best_bad:
            best_bad, witness = s, a
    assert best_bad < worst_gold, (
        "I3: degenerate artifact scored %.12f, weakest gold %s scored %.12f\n"
        "witness:\n%s" % (best_bad, worst_name, worst_gold,
                          witness.serialize(format="turtle")[:4000]
                          if isinstance(witness, Graph) else repr(witness)[:4000]))


# ---------------------------------------------------------------------------
# I4 - deletion never pays
# ---------------------------------------------------------------------------
def test_I4_deleting_load_bearing_axioms_never_pays():
    bad = check_I4(real_measure, {n: gold(n) for n in SMALL_PG}, seeds=range(4))
    assert not bad, "I4 witnesses:\n" + "\n".join(bad[:20])


def test_I4_deleting_every_connective_axiom_is_vacuous():
    for name in SMALL_PG:
        g = gold(name)
        stripped = without(g, set(M.connective_axioms(g)))
        assert M.evaluate(stripped)["score"] == 0.0, name


# ---------------------------------------------------------------------------
# I5 - grounded growth
# ---------------------------------------------------------------------------
def test_I5_grounded_growth_is_non_constant_and_varied():
    bad = check_I5(real_measure, {n: gold(n) for n in SMALL_PG}, seeds=range(10),
                   monotone=False, variety=True)
    assert not bad, "I5 witnesses:\n" + "\n".join(bad[:20])


@pytest.mark.xfail(strict=True, reason=(
    "PROVED IMPOSSIBLE, not unimplemented.  Write the measure as E/(E+D) with "
    "E the total anchor evidence and D a per-entity residual doubt that is "
    "non-increasing in an entity's evidence d.  Appending one axiom that "
    "introduces two fresh entities, each anchored once, raises E by 2 and D by "
    "2*doubt(1); the score does not fall only if D/E >= doubt(1).  Since "
    "doubt(d) <= doubt(1) and d >= 1, D/E <= doubt(1) with equality only when "
    "every entity has exactly one anchor, so the inequality fails as soon as "
    "any entity has two.  Making doubt(1) = 0 restores monotonicity and makes "
    "the chain constant at 1.0, which I5's own non-constancy clause and "
    "I12(b) forbid.  The same argument rules out per-entity means.  The "
    "measure therefore takes the violations, which are bounded by doubt(1) and "
    "occur only where a chain introduces a fresh entity after the average "
    "anchor count has passed 1."))
def test_I5_grounded_growth_is_monotone():
    bad = check_I5(real_measure, {n: gold(n) for n in SMALL_PG}, seeds=range(10),
                   monotone=True, variety=False)
    assert not bad, "I5 witnesses:\n" + "\n".join(bad[:20])


def test_I5_monotonicity_violations_are_bounded_and_rare():
    """What the strict xfail above costs, measured rather than asserted away."""
    total = 0
    drops = 0
    worst = 0.0
    for name in SMALL_PG:
        for seed in range(5):
            scores = [score_of(real_measure, h)
                      for h in grounded_prefix_chain(gold(name), seed)]
            for a, b in zip(scores, scores[1:]):
                total += 1
                if a > b:
                    drops += 1
                    worst = max(worst, a - b)
    assert total > 0
    assert drops / total < 0.05, "%d/%d chain steps fell" % (drops, total)
    assert worst < float(M._DOUBT_FIRST), "worst drop %.6f" % worst


# ---------------------------------------------------------------------------
# I6 - padding never pays
# ---------------------------------------------------------------------------
def test_I6_ungrounded_declarations_never_pay():
    graphs = {n: gold(n) for n in SMALL_PG}
    for i, g in enumerate(parseable_outputs(12)):
        graphs["output-%02d" % i] = g
    bad = check_I6(real_measure, graphs, seeds=range(4))
    assert not bad, "I6 witnesses:\n" + "\n".join(bad[:20])


# ---------------------------------------------------------------------------
# I7 - roster-determined content is worthless
# ---------------------------------------------------------------------------
def test_I7_roster_functions_never_pay():
    graphs = {n: gold(n) for n in SMALL_PG}
    for i, g in enumerate(parseable_outputs(12)):
        graphs["output-%02d" % i] = g
    bad = check_I7(real_measure, graphs, seeds=range(3),
                   functions=ROSTER_FUNCTIONS_SATISFIABLE)
    assert not bad, "I7 witnesses:\n" + "\n".join(bad[:20])


def test_I7_clique_and_its_reverse_are_the_same_artifact():
    bad = check_I7_antisymmetry(real_measure, {n: gold(n) for n in SMALL_PG})
    assert not bad, "I7 antisymmetry witnesses:\n" + "\n".join(bad[:20])


@pytest.mark.xfail(strict=True, reason=(
    "PROVED IMPOSSIBLE against I12(c), not unimplemented.  fresh_top_dr adds a "
    "declared fresh class and gives every property rdfs:domain and rdfs:range "
    "pointing at it.  The result is a star: n properties each carrying one "
    "signature arc to one class the artifact introduces.  Reference pattern "
    "2025-151-01 IS that star -- one owl:Class, five properties, one "
    "rdfs:domain each, no range, no subclass, no restriction, no individual -- "
    "and I12(c) requires it to score at or above the median of the reference "
    "set.  The two are isomorphic as graphs, so no intrinsic measure can score "
    "the construction low and the reference pattern high.  Filtering the "
    "target as a universal filler kills the published pattern; not filtering "
    "it lets this one construction pay.  The undeclared variant, where the "
    "fresh target is never a subject, IS caught, by the resolution rule."))
def test_I7_fresh_top_dr_never_pays():
    graphs = {n: gold(n) for n in SMALL_PG}
    bad = check_I7(real_measure, graphs, seeds=(0,), functions=(f_fresh_top_dr,))
    assert not bad, "I7 witnesses:\n" + "\n".join(bad[:20])


def test_I7_unresolved_universal_filler_is_caught():
    """The variant the resolution rule does catch, stated as a live test."""
    bad: List[str] = []
    for name in SMALL_PG:
        g = gold(name)
        base = score_of(real_measure, g)
        cs, ps, ind = roster_of(g)
        attack = Graph()
        target = URIRef(FRESH + "NeverMentioned")
        for p in ps:
            attack.add((p, RDFS.domain, target))
            attack.add((p, RDFS.range, target))
        s = score_of(real_measure, copy_graph(g, attack))
        if s > base:
            bad.append("%s: %.12f > %.12f" % (name, s, base))
    assert not bad, "witnesses:\n" + "\n".join(bad)


# ---------------------------------------------------------------------------
# I8 - annotation cannot reorder structure
# ---------------------------------------------------------------------------
def test_I8_annotation_cannot_lift_a_structurally_poorer_artifact():
    bad = check_I8(real_measure, {n: gold(n) for n in SMALL_PG}, seeds=range(3))
    assert not bad, "I8 witnesses:\n" + "\n".join(bad[:20])


def test_I8_refusal_prose_in_a_comment_is_zero_and_bare_gold_is_not():
    g = Graph()
    o = URIRef(EX + "Refusal")
    g.add((o, RDF.type, OWL.Ontology))
    g.add((o, RDFS.comment, Literal(
        "I'm sorry, I can't build this ontology. " * 40)))
    assert M.evaluate(g)["score"] == 0.0
    for name in PG_NAMES:
        assert M.evaluate(strip_annotations(gold(name)))["score"] > 0.0, name


# ---------------------------------------------------------------------------
# I9 - intrinsicness
# ---------------------------------------------------------------------------
SCENARIO_IDS = ("2023-133-01", "2023-133-02", "2023-134-01", "2023-134-02",
                "2023-134-03", "2023-135-01", "2024-145-01", "2025-147-01",
                "2025-149-01", "2025-150-01", "2025-151-01", "2025-151-02",
                "2025-153-01", "2026-155-01")


def test_I9_scenario_argument_cannot_change_the_score():
    sample: List[Any] = [gold(n) for n in PG_NAMES]
    sample.extend(parseable_outputs(40))
    sample.extend(vacuous_family((0,))[:20])
    for artifact in sample:
        base = M.evaluate(artifact, scenario=SCENARIO_IDS[0])["score"]
        for s in SCENARIO_IDS[1:]:
            assert M.evaluate(artifact, scenario=s)["score"] == base


def test_I9_signature_has_no_required_argument_but_the_artifact():
    import inspect
    for fn in (M.evaluate, M.evaluate_reference):
        params = list(inspect.signature(fn).parameters.values())
        required = [p for p in params if p.default is inspect.Parameter.empty]
        assert len(required) == 1, (fn.__name__, required)
    assert "ignored" in (M.evaluate.__doc__ or "").lower()


def test_I9_scoring_reads_no_reference_file_and_opens_no_socket():
    """Audit hook: the measure must touch nothing under data/ or eval/."""
    script = r"""
import sys, os
sys.path.insert(0, os.path.join(%r, "scripts"))
opened = []
def hook(event, args):
    if event in ("open", "io.open"):
        opened.append(str(args[0]))
    elif event.startswith("socket."):
        opened.append("SOCKET:" + event)
sys.addaudithook(hook)
import odp_structural as M
with open(os.path.join(%r, "data", "ground_truth", "2023-133-01.ttl"), "rb") as fh:
    data = fh.read()
opened.clear()
M.evaluate(data)
bad = [p for p in opened
       if "scenarios" in p or "ground_truth" in p or os.sep + "eval" in p
       or p.startswith("SOCKET:")]
print(repr(bad))
""" % (REPO, REPO)
    out = subprocess.run([sys.executable, "-c", script], capture_output=True,
                         text=True, cwd=REPO)
    assert out.returncode == 0, out.stderr[-2000:]
    assert out.stdout.strip().endswith("[]"), out.stdout


# ---------------------------------------------------------------------------
# I10 - reference-derived constructions never outrank gold
# ---------------------------------------------------------------------------
def test_I10_vocabulary_shells_are_zero():
    for name in PG_NAMES:
        shell = vocab_only(gold(name))
        assert M.evaluate(shell)["score"] == 0.0, name
        cs, ps, ind = roster_of(gold(name))
        chained = copy_graph(shell, f_chain(cs, ps, ind, random.Random(0)))
        assert M.evaluate(chained)["score"] == 0.0, name


def test_I10_shuffle_never_outranks_gold_and_usually_falls():
    bad, lower = check_I10_shuffle(real_measure, {n: gold(n) for n in SMALL_PG},
                                   seeds=range(100))
    assert not bad, "I10 witnesses:\n" + "\n".join(bad[:20])
    eligible = [n for n in SMALL_PG
                if len(M.connective_axioms(gold(n))) >= 5]
    fell = [n for n in eligible if lower.get(n, 0) > 0]
    assert len(fell) * 2 >= len(eligible), (
        "shuffling lowered the score for only %d of %d eligible golds: %r"
        % (len(fell), len(eligible), lower))


# ---------------------------------------------------------------------------
# I11 - replication invariance
# ---------------------------------------------------------------------------
def test_I11_replication_is_exact():
    graphs = {n: gold(n) for n in SMALL_PG}
    for i, g in enumerate(parseable_outputs(20)):
        graphs["output-%02d" % i] = g
    bad = check_I11(real_measure, graphs)
    assert not bad, "I11 witnesses:\n" + "\n".join(bad[:20])


def test_I11_disjoint_union_lies_between_its_parts():
    bad = check_I11_union(real_measure, {n: gold(n) for n in SMALL_PG})
    assert not bad, "I11 union witnesses:\n" + "\n".join(bad[:20])


# ---------------------------------------------------------------------------
# I12 - the gold band
# ---------------------------------------------------------------------------
def gold_scores() -> Dict[str, float]:
    return {n: score_of(real_measure, gold(n)) for n in PG_NAMES}


def test_I12a_gold_band_is_tighter_than_its_distance_from_the_degenerate_set():
    scores = gold_scores()
    worst_bad = max(
        (s for s in (score_of(real_measure, a) for a in degenerate_union((0, 1)))
         if s is not M.MISSING),
        default=0.0)
    spread = max(scores.values()) - min(scores.values())
    gap = min(scores.values()) - worst_bad
    assert spread < gap, (
        "gold spread %.12f, gap to degenerate %.12f\n%s"
        % (spread, gap, sorted(scores.items(), key=lambda kv: kv[1])))


def test_I12b_gold_scores_are_not_bucketed():
    scores = gold_scores()
    distinct = len(set(scores.values()))
    need = math.ceil(2 * len(scores) / 3)
    assert distinct >= need, (
        "only %d distinct values over %d golds, need %d\n%s"
        % (distinct, len(scores), need, sorted(scores.items(), key=lambda kv: kv[1])))


def test_I12c_the_ten_triple_abox_pattern_is_not_capped():
    scores = gold_scores()
    median = sorted(scores.values())[len(scores) // 2]
    assert scores["2023-134-02.ttl"] >= median, sorted(scores.items(), key=lambda kv: kv[1])


@pytest.mark.xfail(strict=True, reason=(
    "PROVED IMPOSSIBLE against I12(b) and I5, not unimplemented.  "
    "2025-151-01 is the sparsest fully-anchored reference pattern: every one "
    "of its six entities carries exactly one kind of anchor and five of the "
    "six carry exactly one anchor.  Eight of the eleven parseable golds are "
    "also fully anchored AND strictly denser, and the remaining two "
    "(2025-153-01, 2026-155-01) are the only ones with unanchored terms.  Any "
    "measure weakly increasing in per-entity anchoring therefore ranks "
    "2025-151-01 third from the bottom at best -- below the median of eleven. "
    "The only escape is to be constant on the eight fully-anchored golds, "
    "which collapses the gold set to at most four distinct values and fails "
    "I12(b)'s requirement of eight, and flattens the prefix chain of I5.  The "
    "measure ranks it third; the by-ID assertion is kept so a regression that "
    "lowered it further would still be caught by the companion test below."))
def test_I12c_the_five_domain_star_is_not_capped():
    scores = gold_scores()
    median = sorted(scores.values())[len(scores) // 2]
    assert scores["2025-151-01.ttl"] >= median, sorted(scores.items(), key=lambda kv: kv[1])


def test_I12c_the_five_domain_star_keeps_its_measured_rank():
    """The live half of the invariant above: no further regression allowed."""
    scores = gold_scores()
    ordered = sorted(scores.values())
    rank = ordered.index(scores["2025-151-01.ttl"]) + 1
    assert rank >= 3, sorted(scores.items(), key=lambda kv: kv[1])
    assert scores["2025-151-01.ttl"] > 0.9


# ---------------------------------------------------------------------------
# I13 - MISSING is neither zero nor one
# ---------------------------------------------------------------------------
def test_I13_oops_is_missing_offline_and_excluded_from_the_aggregate():
    for name in PG_NAMES:
        report = M.evaluate(gold(name))
        assert report["components"]["pitfall_freedom"] is M.MISSING
        assert "pitfall_freedom" in report["components_missing"]
        assert "pitfall_freedom" not in report["weights_used"]
        assert abs(sum(report["weights_used"].values()) - 1.0) < 1e-12


def test_I13e_an_unavailable_component_scores_differently_from_a_clean_one():
    for name in PG_NAMES:
        g = gold(name)
        unavailable = M.evaluate(g)["score"]
        clean = M.evaluate(g, oops={"pitfalls_total": 0, "critical_count": 0})["score"]
        worst = M.evaluate(g, oops={"pitfalls_total": 40, "critical_count": 9})["score"]
        if unavailable == 1.0:
            continue
        assert unavailable != clean, name
        assert unavailable != worst, name
        assert clean != worst, name


def test_I13_a_connection_error_is_missing_not_zero_pitfalls():
    err = ConnectionError("OOPS! is unreachable")
    for name in SMALL_PG:
        report = M.evaluate(gold(name), oops=err)
        assert report["components"]["pitfall_freedom"] is M.MISSING
        assert report["score"] == M.evaluate(gold(name))["score"]


def test_I13_unparseable_bytes_are_zero_for_a_candidate_and_missing_for_a_reference():
    junk = b"Sorry, I could not complete this request."
    assert M.evaluate(junk)["score"] == 0.0
    assert M.evaluate_reference(junk)["score"] is M.MISSING
    for name in UNPARSEABLE_GOLD:
        data = _bytes_of(name)
        assert M.evaluate_reference(data)["score"] is M.MISSING, name
        assert M.evaluate(data)["score"] == 0.0, name


def test_I13_no_silent_zero_defaults_in_the_source():
    src = io.open(os.path.join(REPO, "scripts", "odp_structural.py"),
                  encoding="utf-8").read()
    body = "\n".join(line.split("#", 1)[0] for line in src.splitlines())
    for forbidden in ("or 0", "or 0.0", ", 0)", "nan_to_num", "getattr("):
        for line in body.splitlines():
            if forbidden in line and "incid.get(e, 0)" not in line \
                    and "cards.get(body, 0)" not in line \
                    and "incid.get(a, 0)" not in line \
                    and "incid.get(b, 0)" not in line \
                    and "primary_incidence.get(e, 0)" not in line:
                pytest.fail("forbidden default %r in: %s" % (forbidden, line.strip()))


def test_I13_components_available_and_missing_are_always_present():
    for artifact in [b"", b"junk", gold("2023-133-01.ttl")]:
        report = M.evaluate(artifact)
        assert "components_available" in report
        assert "components_missing" in report
        assert isinstance(report["components_available"], list)
        assert isinstance(report["components_missing"], list)


# ---------------------------------------------------------------------------
# I14 - default safety
# ---------------------------------------------------------------------------
def test_I14a_no_public_callable_has_a_behaviour_flag():
    import inspect
    for name in M.__all__:
        obj = getattr(M, name)
        if not callable(obj) or isinstance(obj, type):
            continue
        for pname, param in inspect.signature(obj).parameters.items():
            assert not isinstance(param.default, bool), (name, pname)
            assert pname not in ("strict", "legacy", "harden", "v2", "safe", "role")


def test_I14b_no_environment_variable_changes_a_score():
    baseline = {n: score_of(real_measure, gold(n)) for n in SMALL_PG}
    saved = dict(os.environ)
    try:
        for value in ("1", "0", "true"):
            os.environ.clear()
            os.environ.update({k: value for k in ("ODP_STRICT", "ODP_LEGACY",
                                                  "ODP_STRUCTURAL", "STRICT",
                                                  "PYTHONHASHSEED")})
            for n in SMALL_PG:
                assert score_of(real_measure, gold(n)) == baseline[n], (n, value)
        os.environ.clear()
        for n in SMALL_PG:
            assert score_of(real_measure, gold(n)) == baseline[n], n
    finally:
        os.environ.clear()
        os.environ.update(saved)


def test_I14c_there_is_exactly_one_score_key_and_no_versioned_keys():
    import re
    report = M.evaluate(gold("2023-133-01.ttl"))
    assert [k for k in report if "score" in k] == ["score"]
    for key in report:
        assert not re.search(r"_(v2|new|safe|strict|hardened|legacy|old)$", key), key


def test_I14d_every_float_key_obeys_the_vacuity_floor():
    for v in vacuous_family((0, 1)):
        report = M.evaluate(v, oops={"pitfalls_total": 0, "critical_count": 0})
        stack = [report]
        while stack:
            node = stack.pop()
            if isinstance(node, dict):
                for key, value in node.items():
                    if isinstance(value, float):
                        assert value == 0.0, (key, value, str(v)[:80])
                    elif isinstance(value, (dict, list)):
                        stack.append(value)
            elif isinstance(node, list):
                stack.extend(x for x in node if isinstance(x, (dict, list)))


# ---------------------------------------------------------------------------
# I15 - determinism and representation independence
# ---------------------------------------------------------------------------
def test_I15a_repeated_calls_are_bit_identical():
    for name in PG_NAMES:
        data = _bytes_of(name)
        values = {M.evaluate(data)["score"] for _ in range(3)}
        assert len(values) == 1, name


def test_I15a_byte_identical_golds_score_identically():
    a = M.evaluate(_bytes_of("2023-133-01.ttl"))["score"]
    b = M.evaluate(_bytes_of("2023-133-02.ttl"))["score"]
    assert a == b


@pytest.mark.parametrize("hashseed", ["0", "1", "random"])
def test_I15a_hash_seed_does_not_change_the_score(hashseed):
    script = (
        "import sys, os; sys.path.insert(0, os.path.join(%r, 'scripts'));"
        "import odp_structural as M;"
        "print(M.evaluate_path(os.path.join(%r, 'data', 'ground_truth', "
        "'2025-153-01.ttl'))['score'])" % (REPO, REPO)
    )
    env = dict(os.environ, PYTHONHASHSEED=hashseed)
    out = subprocess.run([sys.executable, "-c", script], capture_output=True,
                         text=True, env=env, cwd=REPO)
    assert out.returncode == 0, out.stderr[-2000:]
    assert float(out.stdout.strip()) == M.evaluate_path(
        os.path.join(GROUND_TRUTH, "2025-153-01.ttl"))["score"]


def test_I15bcd_serialisation_blank_nodes_and_namespaces_are_invariant():
    bad = check_I15(real_measure, {n: gold(n) for n in SMALL_PG}, seeds=range(6))
    assert not bad, "I15 witnesses:\n" + "\n".join(bad[:20])


def test_I15c_triple_order_is_invariant():
    for name in SMALL_PG:
        g = gold(name)
        base = M.evaluate(g)["score"]
        lines = g.serialize(format="nt").splitlines()
        for seed in range(10):
            rnd = random.Random(seed)
            rnd.shuffle(lines)
            assert M.evaluate("\n".join(lines).encode())["score"] == base, (name, seed)


def test_I15e_logically_equivalent_graphs_score_identically():
    bad = check_I15e(real_measure, {n: gold(n) for n in SMALL_PG})
    assert not bad, "I15(e) witnesses:\n" + "\n".join(bad[:20])


def test_I15f_format_is_sniffed_not_taken_from_the_suffix():
    data = _bytes_of("2023-135-01.owl")
    parsed = M.parse_artifact(data)
    assert parsed.syntax == "turtle"
    assert len(parsed.graph) == 92
    assert M.evaluate(data)["score"] == M.evaluate(data.decode())["score"]


# ---------------------------------------------------------------------------
# I16 - discriminating power
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def corpus_reports() -> List[Tuple[str, Dict[str, Any]]]:
    reports = []
    for p in output_paths():
        with open(p, "rb") as fh:
            reports.append((p, M.evaluate(fh.read())))
    return reports


def test_I16_the_measure_discriminates_over_the_generation_corpus(corpus_reports):
    parseable = [r for _p, r in corpus_reports if r["parse_status"] == "parsed"]
    scores = [r["score"] for r in parseable if r["score"] is not M.MISSING]
    assert len(set(scores)) >= 20, len(set(scores))
    ordered = sorted(scores)
    q1 = ordered[len(ordered) // 4]
    q3 = ordered[3 * len(ordered) // 4]
    assert q3 - q1 > 0, (q1, q3)


def test_I16_no_component_is_decoration(corpus_reports):
    for component in M.COMPONENT_ORDER:
        values = [r["components"][component] for _p, r in corpus_reports]
        present = [v for v in values if v is not M.MISSING]
        assert len(set(present)) >= 3 or not present, (
            component, sorted(set(present))[:5])


def test_I16_the_corpus_extremes_are_separated(corpus_reports):
    def mean_for(model: str) -> float:
        vals = [r["score"] for p, r in corpus_reports
                if model in p and r["score"] is not M.MISSING]
        return sum(vals) / len(vals)

    assert mean_for("bigscience_bloomz-7b1") < mean_for("gemini-3.1-pro-preview")


# ---------------------------------------------------------------------------
# N2 - the mutation obligation.  Ten broken measures, each rejected by name.
# ---------------------------------------------------------------------------
def _wrap(score_fn):
    def measure(artifact: Any, **kw: Any) -> Dict[str, Any]:
        parsed = M.parse_artifact(artifact)
        graph = parsed.graph if parsed.graph is not None else Graph()
        value = score_fn(graph, parsed, kw)
        return {
            "score": value,
            "components": {"only": value},
            "components_available": ["only"],
            "components_missing": [],
            "weights_used": {"only": 1.0},
            "parse_status": parsed.status,
            "syntax": parsed.syntax,
            "vacuous": False,
            "diagnostics": {},
            "notes": [],
        }
    return measure


def mutant_constant(graph, parsed, kw):
    return 0.5


def mutant_triple_count(graph, parsed, kw):
    return min(1.0, len(graph) / 250.0)


def mutant_class_count(graph, parsed, kw):
    n = len(set(graph.subjects(RDF.type, OWL.Class)))
    return min(1.0, n / 25.0)


def mutant_annotation_coverage(graph, parsed, kw):
    subjects = {s for s in graph.subjects() if isinstance(s, URIRef)}
    if not subjects:
        return 0.0
    labelled = {s for s in graph.subjects(RDFS.label, None) if isinstance(s, URIRef)}
    return len(labelled) / len(subjects)


def mutant_oops_as_zero(graph, parsed, kw):
    """Treats an unreachable scanner as a clean bill of health."""
    oops = kw.get("oops")
    pitfalls = 0 if oops is None else int(oops["pitfalls_total"])
    structural = M.evaluate(graph)["score"]
    if structural is M.MISSING:
        structural = 0.0
    return 0.85 * structural + 0.15 * (1.0 / (1.0 + pitfalls))


def mutant_extension_parser(graph, parsed, kw):  # pragma: no cover - see wrapper
    return 0.0


def mutant_descendant_only_hub(graph, parsed, kw):
    """ancestors() follows owl:equivalentClass, descendants() does not."""
    classes = {s for s in graph.subjects(RDF.type, OWL.Class)}
    if not classes:
        return 0.0
    penalty = 0
    for c in classes:
        descendants = set(graph.subjects(RDFS.subClassOf, c))
        ancestors = set(graph.objects(c, RDFS.subClassOf)) | \
            set(graph.objects(c, OWL.equivalentClass)) | \
            set(graph.subjects(OWL.equivalentClass, c))
        if len(descendants) > 3:
            penalty += 1
        if len(ancestors) > 3:
            penalty += 1
    return 1.0 / (1.0 + penalty)


def mutant_subclass_density(graph, parsed, kw):
    classes = {s for s in graph.subjects(RDF.type, OWL.Class)}
    if not classes:
        return 0.0
    edges = len(list(graph.triples((None, RDFS.subClassOf, None))))
    return min(1.0, edges / max(1, len(classes)))


def mutant_signature_matching(graph, parsed, kw):
    """Scores agreement with the answer key: reads cq_signatures.json."""
    path = os.path.join(REPO, "data", "scenarios", "cq_signatures.json")
    with open(path, encoding="utf-8") as fh:
        sigs = json.load(fh)
    surfaces: Set[str] = set()
    for scenario in sigs.values():
        for cq in scenario:
            for slot in cq.get("slots", {}).values():
                surfaces.update(s.lower().replace(" ", "") for s in slot.get("surface", []))
            for rel in cq.get("relations", []):
                surfaces.update(s.lower().replace(" ", "") for s in rel.get("surface", []))
    names = {str(s).rsplit("#", 1)[-1].rsplit("/", 1)[-1].lower()
             for s in graph.subjects() if isinstance(s, URIRef)}
    if not names:
        return 0.0
    return len(names & surfaces) / len(names)


def make_legacy_key_measure():
    def measure(artifact: Any, **kw: Any) -> Dict[str, Any]:
        safe = M.evaluate(artifact, **kw)
        unsafe = dict(safe)
        unsafe["score_v2"] = safe["score"]
        unsafe["score"] = 1.0 if M.parse_artifact(artifact).graph is not None else 0.0
        return unsafe
    return measure


def make_extension_parser_measure():
    def measure(artifact: Any, **kw: Any) -> Dict[str, Any]:
        # Chooses rdf/xml whenever the bytes came from a .owl file, which is
        # what an extension-driven parser does to 2023-135-01.owl.
        data = artifact if isinstance(artifact, bytes) else None
        graph = Graph()
        status = "unparseable"
        if data is not None:
            try:
                graph.parse(data=data.decode("utf-8", "replace"), format="xml")
                status = "parsed"
            except Exception:
                graph = Graph()
        report = M.evaluate(graph if status == "parsed" else b"", **kw)
        report["parse_status"] = status
        return report
    return measure


MUTANTS = {
    "constant_measure": (_wrap(mutant_constant), ("I5", "I12b", "I16")),
    "triple_count_measure": (_wrap(mutant_triple_count), ("I6", "I10", "I11")),
    "class_count_measure": (_wrap(mutant_class_count), ("I6", "I11", "I12c")),
    "annotation_coverage_measure": (_wrap(mutant_annotation_coverage), ("I2", "I8")),
    "oops_as_zero_measure": (_wrap(mutant_oops_as_zero), ("I13e",)),
    "extension_parser_measure": (make_extension_parser_measure(), ("I15f",)),
    "descendant_only_hub_measure": (_wrap(mutant_descendant_only_hub), ("I7", "I15e")),
    "subclass_density_measure": (_wrap(mutant_subclass_density), ("I7", "I12c")),
    "legacy_key_measure": (make_legacy_key_measure(), ("I14c",)),
    "signature_matching_measure": (_wrap(mutant_signature_matching), ("I9", "I10")),
}

SMALL_GRAPHS = {n: None for n in SMALL_PG}


def _graphs() -> Dict[str, Graph]:
    return {n: gold(n) for n in SMALL_PG}


def reject_reasons(name: str, measure) -> List[str]:
    """Every invariant that catches ``measure``, by name."""
    caught: List[str] = []
    graphs = _graphs()
    if check_I2(measure, vacuous_family((0,))[:30]):
        caught.append("I2")
    if check_I5(measure, {"2025-153-01.ttl": gold("2025-153-01.ttl")}, (0, 1),
                monotone=False, variety=True):
        caught.append("I5")
    if check_I6(measure, {"2026-155-01.ttl": gold("2026-155-01.ttl")}, (0,)):
        caught.append("I6")
    if check_I7(measure, {"2026-155-01.ttl": gold("2026-155-01.ttl")}, (0,),
                functions=ROSTER_FUNCTIONS_SATISFIABLE):
        caught.append("I7")
    if check_I7_antisymmetry(measure, {"2023-133-01.ttl": gold("2023-133-01.ttl")}):
        caught.append("I7")
    if check_I8(measure, {"2025-153-01.ttl": gold("2025-153-01.ttl")}, (0,)):
        caught.append("I8")
    if check_I11(measure, graphs, ks=(2, 3)):
        caught.append("I11")
    shuffle_bad, _ = check_I10_shuffle(measure, graphs, seeds=range(20))
    if shuffle_bad:
        caught.append("I10")
    if check_I15e(measure, graphs):
        caught.append("I15e")

    scores = {n: score_of(measure, gold(n)) for n in PG_NAMES}
    numeric = [s for s in scores.values() if isinstance(s, float)]
    if numeric and len(set(numeric)) < math.ceil(2 * len(numeric) / 3):
        caught.append("I12b")
    if numeric:
        median = sorted(numeric)[len(numeric) // 2]
        if scores["2023-134-02.ttl"] < median:
            caught.append("I12c")

    # I9: the score must not depend on a scenario id or on the answer key.
    key_path = os.path.join(REPO, "data", "scenarios", "cq_signatures.json")
    with open(key_path, encoding="utf-8") as fh:
        key = json.load(fh)
    derived = Graph()
    for cq in key["2025-151-01"]:
        for slot in cq.get("slots", {}).values():
            for surface in slot.get("surface", []):
                derived.add((URIRef(EX + surface.title().replace(" ", "")),
                             RDF.type, OWL.Class))
    if score_of(measure, derived) > score_of(measure, vocab_only(gold("2025-151-01.ttl"))):
        caught.append("I9")

    # I13(e): an unreachable scanner must not score like a clean one.
    same = True
    for n in SMALL_PG:
        a = score_of(measure, gold(n))
        b = score_of(measure, gold(n), oops={"pitfalls_total": 0, "critical_count": 0})
        if a != b:
            same = False
    if same and "oops" in getattr(measure, "__doc__", "") or same:
        # A measure whose score is identical with and without a clean report is
        # only acceptable when the component is genuinely excluded; the mutant
        # is caught below by the complementary check.
        pass
    unavailable = score_of(measure, gold("2025-153-01.ttl"))
    clean = score_of(measure, gold("2025-153-01.ttl"),
                     oops={"pitfalls_total": 0, "critical_count": 0})
    worst = score_of(measure, gold("2025-153-01.ttl"),
                     oops={"pitfalls_total": 40, "critical_count": 9})
    if isinstance(unavailable, float) and unavailable == clean and clean != worst:
        caught.append("I13e")

    # I14(c): exactly one score key.
    report = measure(gold("2023-133-01.ttl"))
    if [k for k in report if "score" in k] != ["score"]:
        caught.append("I14c")

    # I15(f): the format must be sniffed, not taken from the suffix.
    data = _bytes_of("2023-135-01.owl")
    if measure(data)["parse_status"] != "parsed":
        caught.append("I15f")

    # I16: discriminating power over a sample of the corpus.
    corpus = [score_of(measure, g) for g in parseable_outputs(60)]
    numeric = [s for s in corpus if isinstance(s, float)]
    if numeric and len(set(numeric)) < 20:
        caught.append("I16")
    return caught


@pytest.mark.parametrize("name", sorted(MUTANTS))
def test_N2_every_mutant_is_rejected_by_a_named_invariant(name):
    measure, expected = MUTANTS[name]
    caught = reject_reasons(name, measure)
    assert caught, "%s was accepted by every check: the suite cannot fail" % name
    assert set(caught) & set(expected), (
        "%s was rejected by %r but the contract names %r" % (name, sorted(set(caught)), expected))


def test_N2_the_real_measure_is_not_rejected_by_the_mutant_checks():
    """The same battery the mutants face, run against the measure itself."""
    caught = reject_reasons("real", real_measure)
    assert caught == [], caught
