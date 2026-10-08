"""Robustness of the connectivity measure: it must RETURN, and it must not move.

Two things are asserted here and nothing else.

**It returns (I1).**  ``evaluate_path`` on ``data/ground_truth/2024-145-01.owl``
-- 15.9 MB of RDF/XML, 158596 triples, a 14345-node subsumption component --
did not return in any usable time: 564s end to end, 525s of it in a single
quadratic scan, which silently stalled a tournament run.  A hang is strictly
worse than a raise -- a raising measure can be caught and recorded as MISSING,
a hanging one cannot be recorded at all.
The reproduction runs the call in a child process so that the defect shows up
as a timeout the suite survives rather than as a wedged pytest session.

That reference's subsumption component is *above* ``_CLOSURE_MAX_NODES``, so
scoring it never materialises the transitive closure -- and neither does the
12000-class synthetic below it, nor any artefact in the corpus.  The branch
that does materialise the closure kept a second, smaller copy of the same
defect: a 2900-class subsumption chain, one node under the cap, cost 90.2s,
because every one of its 4.2 million comparability pairs was walked with
``rdflib`` term comparisons.  ``test_deep_taxonomy_...`` is that case, and it
is stated in this file because it is the same invariant, not a new one.

**It does not move.**  Fixing a hang is only legitimate if every artefact that
already returned returns exactly the same float.  ``BASELINE_SCORES`` and
``BASELINE_COMPONENTS`` below were recorded from the *unfixed* module over the
nine gold Turtle patterns, all 54 adversarial fixtures and 100 corpus
artefacts, and are compared with ``==`` on the float, never with a tolerance.
That includes the four published adversarial results the paper reports as a
limitation (shuffled gold, A-Box star hub, and the two equivalence cliques,
all above the 0.9784 gold floor); they are pinned here precisely so that no
performance work can quietly tune them away.
"""

import json
import os
import subprocess
import sys

import pytest

TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(TESTS_DIR)
SCRIPTS_DIR = os.path.join(REPO_ROOT, "scripts")
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)

import odp_structural  # noqa: E402

LARGE_REFERENCE = os.path.join(REPO_ROOT, "data", "ground_truth", "2024-145-01.owl")

# Wall-clock budget for one child process: interpreter start, rdflib import,
# reading 15.9 MB, parsing it and scoring it.  The measured cost of the whole
# child before the fix was 564s; after it, 31s.
# The budget is deliberately loose -- this is a liveness assertion, not a
# benchmark -- but it is far below the observed defect.
RETURN_BUDGET_SECONDS = 120.0

# Same, for a synthetic graph that exercises the quadratic directly without
# paying for the RDF/XML parse.
SYNTHETIC_BUDGET_SECONDS = 60.0
SYNTHETIC_CLASSES = 12000


def _run_child(body, budget):
    """Run ``body`` in a child process; return its parsed JSON stdout.

    A hang is reported as a test failure naming the budget, and the child is
    killed, so the rest of the suite still runs.
    """
    source = (
        "import json, os, sys, time\n"
        "sys.path.insert(0, " + repr(SCRIPTS_DIR) + ")\n"
        "import odp_structural as m\n"
        + body
    )
    try:
        proc = subprocess.run(
            [sys.executable, "-c", source],
            capture_output=True,
            text=True,
            timeout=budget,
        )
    except subprocess.TimeoutExpired:
        pytest.fail(
            "the measure did not return within %.0fs -- it hung. "
            "Invariant I1 requires a return (a score, or MISSING) for every "
            "artefact, including large ones." % budget
        )
    assert proc.returncode == 0, (
        "the child exited %s\nstdout:\n%s\nstderr:\n%s"
        % (proc.returncode, proc.stdout, proc.stderr)
    )
    tail = [ln for ln in proc.stdout.splitlines() if ln.startswith("{")]
    assert tail, "no JSON on stdout:\n%s\n%s" % (proc.stdout, proc.stderr)
    return json.loads(tail[-1])


@pytest.fixture(scope="module")
def large_reference_report():
    """Score the 15.9 MB reference once, in a child, under a hard timeout."""
    if not os.path.exists(LARGE_REFERENCE):
        pytest.skip("the 15.9 MB reference is not checked out")
    body = (
        "t = time.time()\n"
        "r = m.evaluate_path(" + repr(LARGE_REFERENCE) + ")\n"
        "r['elapsed'] = time.time() - t\n"
        "print(json.dumps({k: r[k] for k in "
        "('score', 'parse_status', 'syntax', 'vacuous', 'elapsed', 'diagnostics')}))\n"
    )
    return _run_child(body, RETURN_BUDGET_SECONDS)


def test_large_reference_returns_at_all(large_reference_report):
    """I1: the call completes.  This is the defect, stated directly."""
    assert "score" in large_reference_report


def test_large_reference_returns_within_budget(large_reference_report):
    assert large_reference_report["elapsed"] < RETURN_BUDGET_SECONDS


def test_large_reference_actually_parsed(large_reference_report):
    """The file is well-formed RDF/XML, so the answer is a score, not a
    parse failure dressed up as one."""
    assert large_reference_report["parse_status"] == "parsed"
    assert large_reference_report["syntax"] == "xml"
    assert large_reference_report["diagnostics"]["triples"] > 100000


def test_large_reference_score_is_a_real_number_in_range(large_reference_report):
    """Not MISSING, and not the 0.0 that an unparseable candidate gets: the
    bytes parse, so a size guard that returned either would be wrong here."""
    score = large_reference_report["score"]
    assert isinstance(score, float)
    assert 0.0 < score <= 1.0


def test_large_reference_as_reference_is_not_missing():
    """I13: MISSING is for a reference rdflib cannot read.  This one it can,
    so ``evaluate_reference`` must hand back the same score, not MISSING."""
    if not os.path.exists(LARGE_REFERENCE):
        pytest.skip("the 15.9 MB reference is not checked out")
    body = (
        "raw = open(" + repr(LARGE_REFERENCE) + ", 'rb').read()\n"
        "r = m.evaluate_reference(raw)\n"
        "print(json.dumps({'score': r['score'], 'parse_status': r['parse_status']}))\n"
    )
    out = _run_child(body, RETURN_BUDGET_SECONDS)
    assert out["parse_status"] == "parsed"
    assert out["score"] is not None
    assert out["score"] != 0.0


def test_wide_taxonomy_returns_within_budget():
    """The blowup isolated from the parse.

    A single connected subsumption component of ``SYNTHETIC_CLASSES`` nodes was
    scanned pair-by-pair once per node when the measure looked for a universal
    top or bottom, which is quadratic in the component and is the whole of the
    hang.  Nothing about this graph is adversarial or unusual -- it is a plain
    two-level taxonomy of the kind a large published vocabulary has -- so the
    measure has to score it.
    """
    body = (
        "n = %d\n" % SYNTHETIC_CLASSES
        + "lines = ['@prefix ex: <http://example.org/> .',\n"
        "         '@prefix owl: <http://www.w3.org/2002/07/owl#> .',\n"
        "         '@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .',\n"
        "         'ex:Root a owl:Class .', 'ex:p a owl:ObjectProperty ;',\n"
        "         '  rdfs:domain ex:Root ; rdfs:range ex:Root .']\n"
        "for i in range(n):\n"
        "    lines.append('ex:C%d a owl:Class ; rdfs:subClassOf ex:C%d .' % (i, i // 2))\n"
        "lines.append('ex:C0 rdfs:subClassOf ex:Root .')\n"
        "data = '\\n'.join(lines).encode('utf-8')\n"
        "t = time.time()\n"
        "r = m.evaluate(data)\n"
        "print(json.dumps({'score': r['score'], 'parse_status': r['parse_status'],\n"
        "                  'elapsed': time.time() - t,\n"
        "                  'classes': r['diagnostics']['roster_classes']}))\n"
    )
    out = _run_child(body, SYNTHETIC_BUDGET_SECONDS)
    assert out["parse_status"] == "parsed"
    assert out["classes"] >= SYNTHETIC_CLASSES
    assert out["score"] is not None
    assert out["elapsed"] < SYNTHETIC_BUDGET_SECONDS


# A subsumption component just BELOW ``odp_structural._CLOSURE_MAX_NODES``.
#
# ``test_wide_taxonomy_returns_within_budget`` above builds a 12000-node
# component, which is *above* that cap, so the measure takes the cheap
# direct-edge branch and never materialises the transitive closure.  Nothing in
# this file and nothing in the corpus ever exercised the branch that does
# materialise it -- and that is where the residual quadratic lives.  2900
# classes in a plain subsumption chain is an ordinary published-vocabulary
# shape, not an adversarial one.
DEEP_TAXONOMY_CLASSES = 2900

# ``evaluate`` on that chain cost 90.2s before the fix and 32.3s after it, and
# the child pays interpreter start, the rdflib import and a 5800-triple Turtle
# parse on top.  Almost all of the 58s was rdflib term ``__eq__`` / ``__ne__``
# and term hashing, called from the union-find and the closure walk once per
# comparability pair -- 4.2 million of them.  One node past the cap, at 3100
# classes, the same shape costs 1.2s.  The budget sits between the before and
# the after with room for a loaded machine, and matches the budget the wide
# taxonomy above already uses.
DEEP_TAXONOMY_BUDGET_SECONDS = 60.0

# Recorded from the module BEFORE this round's change, at exactly
# DEEP_TAXONOMY_CLASSES classes.  Pinned so that making the closure branch fast
# cannot quietly make it compute something else.
DEEP_TAXONOMY_SCORE = 0.0013783210518121094


def _deep_chain_body():
    """Child-process source: score a chain of ``DEEP_TAXONOMY_CLASSES`` classes."""
    return (
        "n = %d\n" % DEEP_TAXONOMY_CLASSES
        + "lines = ['@prefix ex: <http://example.org/> .',\n"
        "         '@prefix owl: <http://www.w3.org/2002/07/owl#> .',\n"
        "         '@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .',\n"
        "         'ex:p a owl:ObjectProperty ; rdfs:domain ex:C0 ; rdfs:range ex:C1 .']\n"
        "for i in range(n):\n"
        "    lines.append('ex:C%d a owl:Class .' % i)\n"
        "    if i:\n"
        "        lines.append('ex:C%d rdfs:subClassOf ex:C%d .' % (i, i - 1))\n"
        "data = '\\n'.join(lines).encode('utf-8')\n"
        "t = time.time()\n"
        "r = m.evaluate(data)\n"
        "print(json.dumps({'score': r['score'], 'parse_status': r['parse_status'],\n"
        "                  'elapsed': time.time() - t,\n"
        "                  'classes': r['diagnostics']['roster_classes']}))\n"
    )


@pytest.fixture(scope="module")
def deep_taxonomy_report():
    return _run_child(_deep_chain_body(), DEEP_TAXONOMY_BUDGET_SECONDS)


def test_deep_taxonomy_under_the_closure_cap_returns_within_budget(
    deep_taxonomy_report,
):
    """I1, on the branch the existing liveness tests miss.

    The measure materialises the transitive closure of the subsumption quotient
    whenever that quotient has at most ``_CLOSURE_MAX_NODES`` nodes.  A chain of
    2900 classes closes to 4.2 million comparability pairs, and every one of
    them was pushed through a union-find whose inner loop compares ``rdflib``
    terms with ``!=``.  The call does return eventually, but not in any time a
    batch can use, which is the same failure D1 names, on the input that still
    reaches it.
    """
    assert deep_taxonomy_report["parse_status"] == "parsed"
    assert deep_taxonomy_report["classes"] >= DEEP_TAXONOMY_CLASSES
    assert deep_taxonomy_report["score"] is not None
    assert deep_taxonomy_report["elapsed"] < DEEP_TAXONOMY_BUDGET_SECONDS


def test_deep_taxonomy_score_is_unchanged(deep_taxonomy_report):
    """Bit-identity on the exact shape whose cost is being fixed.

    The budget above could be met by computing something cheaper *and
    different*.  This forbids that: the float is pinned to what the module
    returned before the change.
    """
    assert deep_taxonomy_report["score"] == DEEP_TAXONOMY_SCORE


def _naive_components_with_pairs(pairs):
    """An independent, deliberately slow statement of what the helper means.

    Connected components of the undirected graph induced by ``pairs``, each
    paired with the restriction of ``pairs`` to it, ordered by the ``str`` of
    the component's union-find root under the module's own ``<=`` tie-break.
    Written out separately so that a faster implementation is checked against
    the definition rather than against itself.
    """
    pairs = list(pairs)
    parent = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra == rb:
            return
        if str(ra) <= str(rb):
            parent[rb] = ra
        else:
            parent[ra] = rb

    for a, b in pairs:
        union(a, b)
    roots = []
    groups = {}
    for node in list(parent):
        r = find(node)
        if r not in groups:
            groups[r] = set()
            roots.append(r)
        groups[r].add(node)
    out = []
    for r in sorted(roots, key=str):
        comp = groups[r]
        out.append((comp, {(a, b) for a, b in pairs if a in comp and b in comp}))
    return out


def _normalise(components):
    return [(sorted(c, key=str), sorted(l, key=str)) for c, l in components]


@pytest.mark.parametrize(
    "pairs",
    [
        [],
        [("a", "a")],
        [("a", "b")],
        [("b", "a"), ("a", "b")],
        [("a", "b"), ("b", "c"), ("c", "a")],
        [("a", "b"), ("c", "d"), ("e", "f")],
        [("z", "y"), ("y", "x"), ("m", "n"), ("n", "m"), ("q", "q")],
        [("C%02d" % i, "C%02d" % (i // 2)) for i in range(1, 60)],
        [("A", "B"), ("B", "C"), ("A", "C"), ("D", "E")],
    ],
)
def test_components_helper_agrees_with_the_naive_definition_on_strings(pairs):
    """Same components, same per-component pair sets, same order."""
    assert _normalise(odp_structural._components_with_pairs(pairs)) == _normalise(
        _naive_components_with_pairs(pairs)
    )


def test_components_helper_agrees_with_the_naive_definition_on_rdflib_terms():
    """The real input type: ``URIRef``s, where ``!=`` is a Python method call.

    A dense closure-shaped pair set plus a disjoint chain, which is the shape
    the taxonomy pass hands in.
    """
    from rdflib import URIRef

    nodes = [URIRef("http://example.org/C%03d" % i) for i in range(40)]
    pairs = [(nodes[i], nodes[j]) for i in range(40) for j in range(i + 1, 40)]
    pairs += [
        (
            URIRef("http://example.org/D%d" % i),
            URIRef("http://example.org/D%d" % (i + 1)),
        )
        for i in range(10)
    ]
    fast = odp_structural._components_with_pairs(pairs)
    assert len(fast) == 2
    assert _normalise(fast) == _normalise(_naive_components_with_pairs(pairs))


def test_components_helper_is_stable_under_duplicate_term_instances():
    """Distinct ``URIRef`` objects with the same IRI are one node, as before."""
    from rdflib import URIRef

    a1, a2 = URIRef("http://example.org/a"), URIRef("http://example.org/a")
    assert a1 is not a2
    b = URIRef("http://example.org/b")
    fast = odp_structural._components_with_pairs([(a1, b), (a2, b)])
    assert len(fast) == 1
    assert _normalise(fast) == _normalise(
        _naive_components_with_pairs([(a1, b), (a2, b)])
    )


def test_components_helper_mixes_term_types_without_confusing_them():
    """A ``URIRef`` and a ``Literal`` that print the same are still two nodes."""
    from rdflib import Literal, URIRef

    u = URIRef("x")
    l = Literal("x")
    assert str(u) == str(l)
    fast = odp_structural._components_with_pairs([(u, URIRef("y")), (l, URIRef("z"))])
    assert _normalise(fast) == _normalise(
        _naive_components_with_pairs([(u, URIRef("y")), (l, URIRef("z"))])
    )


# Recorded from scripts/odp_structural.py BEFORE the hang fix, one
# evaluate_path() per artefact.  Exact floats; compared with ==.
# NOTE (2026-09-23): the 11 gpt-5.4 corpus entries below were re-baselined
# after those artefacts were repaired -- 11 recovered from the extraction
# defect and 33 regenerated at a 16384-token budget.  Their scores moved
# from 0.0 (unparseable 3-byte files) to ~0.99 because the INPUT changed,
# not the measure.  No gold and no adversarial baseline moved, which is the
# evidence for that claim; those entries are deliberately untouched.
BASELINE_SCORES = {
    'data/ground_truth/2023-133-01.ttl': 0.9998457028236384,
    'data/ground_truth/2023-133-02.ttl': 0.9998457028236384,
    'data/ground_truth/2023-134-01.ttl': 0.9973924380704041,
    'data/ground_truth/2023-134-02.ttl': 0.9973136333109469,
    'data/ground_truth/2023-134-03.ttl': 0.9977259806708357,
    'data/ground_truth/2025-151-01.ttl': 0.9836065573770492,
    'data/ground_truth/2025-151-02.ttl': 0.9836065573770492,
    'data/ground_truth/2025-153-01.ttl': 0.9783621492908169,
    'data/ground_truth/2026-155-01.ttl': 0.9811453964035052,
    'outputs/bigscience_bloomz-7b1/cq-only/2023-133-01/ontology.ttl': 0.0,
    'outputs/bigscience_bloomz-7b1/cq-only/2023-134-03/ontology.ttl': 0.0,
    'outputs/bigscience_bloomz-7b1/cq-only/2025-149-01/ontology.ttl': 0.0,
    'outputs/bigscience_bloomz-7b1/cq-only/2025-153-01/ontology.ttl': 0.0,
    'outputs/bigscience_bloomz-7b1/scenario-cq-constraints/2023-134-01/ontology.ttl': 0.0,
    'outputs/bigscience_bloomz-7b1/scenario-cq-constraints/2024-145-01/ontology.ttl': 0.0,
    'outputs/bigscience_bloomz-7b1/scenario-cq-constraints/2025-151-01/ontology.ttl': 0.0,
    'outputs/bigscience_bloomz-7b1/scenario-cq-reasoning/2023-133-01/ontology.ttl': 0.0,
    'outputs/bigscience_bloomz-7b1/scenario-cq-reasoning/2023-134-03/ontology.ttl': 0.0,
    'outputs/bigscience_bloomz-7b1/scenario-cq-reasoning/2025-149-01/ontology.ttl': 0.0,
    'outputs/bigscience_bloomz-7b1/scenario-cq-reasoning/2025-153-01/ontology.ttl': 0.0,
    'outputs/bigscience_bloomz-7b1/scenario-cq/2023-134-01/ontology.ttl': 0.0,
    'outputs/bigscience_bloomz-7b1/scenario-cq/2024-145-01/ontology.ttl': 0.0,
    'outputs/bigscience_bloomz-7b1/scenario-cq/2025-151-01/ontology.ttl': 0.0,
    'outputs/bigscience_bloomz-7b1/scenario-only/2023-133-01/ontology.ttl': 0.0,
    'outputs/bigscience_bloomz-7b1/scenario-only/2023-134-03/ontology.ttl': 0.0,
    'outputs/bigscience_bloomz-7b1/scenario-only/2025-149-01/ontology.ttl': 0.0,
    'outputs/bigscience_bloomz-7b1/scenario-only/2025-153-01/ontology.ttl': 0.0,
    'outputs/gemini-3.1-pro-preview/cq-only/2023-134-01/ontology.ttl': 0.9892698246532321,
    'outputs/gemini-3.1-pro-preview/cq-only/2024-145-01/ontology.ttl': 0.9882352941176471,
    'outputs/gemini-3.1-pro-preview/cq-only/2025-151-01/ontology.ttl': 0.9876041454988823,
    'outputs/gemini-3.1-pro-preview/scenario-cq-constraints/2023-133-01/ontology.ttl': 0.9928752011031947,
    'outputs/gemini-3.1-pro-preview/scenario-cq-constraints/2023-134-03/ontology.ttl': 0.9886488465763457,
    'outputs/gemini-3.1-pro-preview/scenario-cq-constraints/2025-149-01/ontology.ttl': 0.9871447104611393,
    'outputs/gemini-3.1-pro-preview/scenario-cq-constraints/2025-153-01/ontology.ttl': 0.9912536443148688,
    'outputs/gemini-3.1-pro-preview/scenario-cq-reasoning/2023-134-01/ontology.ttl': 0.9895144049679324,
    'outputs/gemini-3.1-pro-preview/scenario-cq-reasoning/2024-145-01/ontology.ttl': 0.9903065234477338,
    'outputs/gemini-3.1-pro-preview/scenario-cq-reasoning/2025-151-01/ontology.ttl': 0.9919177075679647,
    'outputs/gemini-3.1-pro-preview/scenario-cq/2023-133-01/ontology.ttl': 0.9971590909090909,
    'outputs/gemini-3.1-pro-preview/scenario-cq/2023-134-03/ontology.ttl': 0.9868421052631579,
    'outputs/gemini-3.1-pro-preview/scenario-cq/2025-149-01/ontology.ttl': 0.9453209422952237,
    'outputs/gemini-3.1-pro-preview/scenario-cq/2025-153-01/ontology.ttl': 0.9919011438590633,
    'outputs/gemini-3.1-pro-preview/scenario-only/2023-134-01/ontology.ttl': 0.9583556894630402,
    'outputs/gemini-3.1-pro-preview/scenario-only/2024-145-01/ontology.ttl': 0.9934993254016926,
    'outputs/gemini-3.1-pro-preview/scenario-only/2025-151-01/ontology.ttl': 0.9971164061314312,
    'outputs/gpt-5.4/cq-only/2023-133-01/ontology.ttl': 0.9133192389006343,
    'outputs/gpt-5.4/cq-only/2023-134-03/ontology.ttl': 0.9886488465763457,
    'outputs/gpt-5.4/cq-only/2025-149-01/ontology.ttl': 0.0949367088607595,
    'outputs/gpt-5.4/cq-only/2025-153-01/ontology.ttl': 0.9090909090909091,
    'outputs/gpt-5.4/scenario-cq-constraints/2023-134-01/ontology.ttl': 0.9901443960577584,
    'outputs/gpt-5.4/scenario-cq-constraints/2024-145-01/ontology.ttl': 0.9903065234477338,
    'outputs/gpt-5.4/scenario-cq-constraints/2025-151-01/ontology.ttl': 0.9876710472835659,
    'outputs/gpt-5.4/scenario-cq-reasoning/2023-133-01/ontology.ttl': 0.9998775785027851,
    'outputs/gpt-5.4/scenario-cq-reasoning/2023-134-03/ontology.ttl': 0.9910277324632952,
    'outputs/gpt-5.4/scenario-cq-reasoning/2025-149-01/ontology.ttl': 0.9976930771900455,
    'outputs/gpt-5.4/scenario-cq-reasoning/2025-153-01/ontology.ttl': 0.9972535277961928,
    'outputs/gpt-5.4/scenario-cq/2023-134-01/ontology.ttl': 0.9901443960577584,
    'outputs/gpt-5.4/scenario-cq/2024-145-01/ontology.ttl': 0.9924443536859302,
    'outputs/gpt-5.4/scenario-cq/2025-151-01/ontology.ttl': 0.98801261829653,
    'outputs/gpt-5.4/scenario-only/2023-133-01/ontology.ttl': 0.9904153354632588,
    'outputs/gpt-5.4/scenario-only/2023-134-03/ontology.ttl': 0.9924719167205788,
    'outputs/gpt-5.4/scenario-only/2025-149-01/ontology.ttl': 0.9921910886541112,
    'outputs/gpt-5.4/scenario-only/2025-153-01/ontology.ttl': 0.9920711661187391,
    'outputs/meta-llama_Llama-2-70b-chat-hf/cq-only/2023-134-01/ontology.ttl': 0.0,
    'outputs/meta-llama_Llama-2-70b-chat-hf/cq-only/2024-145-01/ontology.ttl': 0.9388038942976356,
    'outputs/meta-llama_Llama-2-70b-chat-hf/cq-only/2025-151-01/ontology.ttl': 0.0,
    'outputs/meta-llama_Llama-2-70b-chat-hf/scenario-cq-constraints/2023-133-01/ontology.ttl': 0.0,
    'outputs/meta-llama_Llama-2-70b-chat-hf/scenario-cq-constraints/2023-134-03/ontology.ttl': 0.0,
    'outputs/meta-llama_Llama-2-70b-chat-hf/scenario-cq-constraints/2025-149-01/ontology.ttl': 0.9552691432903715,
    'outputs/meta-llama_Llama-2-70b-chat-hf/scenario-cq-constraints/2025-153-01/ontology.ttl': 0.0,
    'outputs/meta-llama_Llama-2-70b-chat-hf/scenario-cq-reasoning/2023-134-01/ontology.ttl': 0.0,
    'outputs/meta-llama_Llama-2-70b-chat-hf/scenario-cq-reasoning/2024-145-01/ontology.ttl': 0.0,
    'outputs/meta-llama_Llama-2-70b-chat-hf/scenario-cq-reasoning/2025-151-01/ontology.ttl': 0.0,
    'outputs/meta-llama_Llama-2-70b-chat-hf/scenario-cq/2023-133-01/ontology.ttl': 0.0,
    'outputs/meta-llama_Llama-2-70b-chat-hf/scenario-cq/2023-134-03/ontology.ttl': 0.2341717259323504,
    'outputs/meta-llama_Llama-2-70b-chat-hf/scenario-cq/2025-149-01/ontology.ttl': 0.0,
    'outputs/meta-llama_Llama-2-70b-chat-hf/scenario-cq/2025-153-01/ontology.ttl': 0.0,
    'outputs/meta-llama_Llama-2-70b-chat-hf/scenario-only/2023-134-01/ontology.ttl': 0.7534007673526334,
    'outputs/meta-llama_Llama-2-70b-chat-hf/scenario-only/2024-145-01/ontology.ttl': 0.0,
    'outputs/meta-llama_Llama-2-70b-chat-hf/scenario-only/2025-151-01/ontology.ttl': 0.0,
    'outputs/meta-llama_Llama-3.1-8B-Instruct/cq-only/2023-133-01/ontology.ttl': 0.5679862306368331,
    'outputs/meta-llama_Llama-3.1-8B-Instruct/cq-only/2023-134-03/ontology.ttl': 0.9,
    'outputs/meta-llama_Llama-3.1-8B-Instruct/cq-only/2025-149-01/ontology.ttl': 0.6976744186046512,
    'outputs/meta-llama_Llama-3.1-8B-Instruct/cq-only/2025-153-01/ontology.ttl': 0.0,
    'outputs/meta-llama_Llama-3.1-8B-Instruct/scenario-cq-constraints/2023-134-01/ontology.ttl': 0.4976958525345622,
    'outputs/meta-llama_Llama-3.1-8B-Instruct/scenario-cq-constraints/2024-145-01/ontology.ttl': 0.9060402684563759,
    'outputs/meta-llama_Llama-3.1-8B-Instruct/scenario-cq-constraints/2025-151-01/ontology.ttl': 0.0,
    'outputs/meta-llama_Llama-3.1-8B-Instruct/scenario-cq-reasoning/2023-133-01/ontology.ttl': 0.0,
    'outputs/meta-llama_Llama-3.1-8B-Instruct/scenario-cq-reasoning/2023-134-03/ontology.ttl': 0.9926470588235294,
    'outputs/meta-llama_Llama-3.1-8B-Instruct/scenario-cq-reasoning/2025-149-01/ontology.ttl': 0.09063444108761329,
    'outputs/meta-llama_Llama-3.1-8B-Instruct/scenario-cq-reasoning/2025-153-01/ontology.ttl': 0.0,
    'outputs/meta-llama_Llama-3.1-8B-Instruct/scenario-cq/2023-134-01/ontology.ttl': 0.8186986734049273,
    'outputs/meta-llama_Llama-3.1-8B-Instruct/scenario-cq/2024-145-01/ontology.ttl': 0.8761492698756085,
    'outputs/meta-llama_Llama-3.1-8B-Instruct/scenario-cq/2025-151-01/ontology.ttl': 0.0,
    'outputs/meta-llama_Llama-3.1-8B-Instruct/scenario-only/2023-133-01/ontology.ttl': 0.8314087759815243,
    'outputs/meta-llama_Llama-3.1-8B-Instruct/scenario-only/2023-134-03/ontology.ttl': 0.9814612868047983,
    'outputs/meta-llama_Llama-3.1-8B-Instruct/scenario-only/2025-149-01/ontology.ttl': 0.0,
    'outputs/meta-llama_Llama-3.1-8B-Instruct/scenario-only/2025-153-01/ontology.ttl': 0.0,
    'outputs/mistralai_Mistral-7B-Instruct-v0.3/cq-only/2023-134-01/ontology.ttl': 0.0,
    'outputs/mistralai_Mistral-7B-Instruct-v0.3/cq-only/2024-145-01/ontology.ttl': 0.9844216108717269,
    'outputs/mistralai_Mistral-7B-Instruct-v0.3/cq-only/2025-151-01/ontology.ttl': 0.0,
    'outputs/mistralai_Mistral-7B-Instruct-v0.3/scenario-cq-constraints/2023-133-01/ontology.ttl': 0.0,
    'outputs/mistralai_Mistral-7B-Instruct-v0.3/scenario-cq-constraints/2023-134-03/ontology.ttl': 0.0,
    'outputs/mistralai_Mistral-7B-Instruct-v0.3/scenario-cq-constraints/2025-149-01/ontology.ttl': 0.0,
    'outputs/mistralai_Mistral-7B-Instruct-v0.3/scenario-cq-constraints/2025-153-01/ontology.ttl': 0.0,
    'outputs/mistralai_Mistral-7B-Instruct-v0.3/scenario-cq-reasoning/2023-134-01/ontology.ttl': 0.7192807192807192,
    'outputs/mistralai_Mistral-7B-Instruct-v0.3/scenario-cq-reasoning/2024-145-01/ontology.ttl': 0.5289672544080605,
    'outputs/mistralai_Mistral-7B-Instruct-v0.3/scenario-cq-reasoning/2025-151-01/ontology.ttl': 0.0,
    'outputs/mistralai_Mistral-7B-Instruct-v0.3/scenario-cq/2023-133-01/ontology.ttl': 0.0,
    'outputs/mistralai_Mistral-7B-Instruct-v0.3/scenario-cq/2023-134-03/ontology.ttl': 0.0,
    'tests/fixtures/adversarial/a/01_empty.ttl': 0.0,
    'tests/fixtures/adversarial/a/02_whitespace_only.ttl': 0.0,
    'tests/fixtures/adversarial/a/03_prefix_block_only.ttl': 0.0,
    'tests/fixtures/adversarial/a/04_comment_block_only.ttl': 0.0,
    'tests/fixtures/adversarial/a/05_ontology_header_refusal_comment.ttl': 0.0,
    'tests/fixtures/adversarial/a/06_bare_declarations_176_objectproperties.ttl': 0.0,
    'tests/fixtures/adversarial/a/07_classes_only_no_axioms.ttl': 0.0,
    'tests/fixtures/adversarial/a/08_properties_no_domain_range.ttl': 0.0,
    'tests/fixtures/adversarial/a/09_annotation_properties_only.ttl': None,
    'tests/fixtures/adversarial/a/10_refusal_prose_raw.ttl': 0.0,
    'tests/fixtures/adversarial/a/11_binary_garbage.ttl': 0.0,
    'tests/fixtures/adversarial/a/12_e2_polished_hollow.ttl': 0.0,
    'tests/fixtures/adversarial/a/13_single_junk_triple.ttl': 0.967741935483871,
    'tests/fixtures/adversarial/a/14_universal_domain_range.ttl': 0.0,
    'tests/fixtures/adversarial/a/15_bottom_hub_over_base.ttl': 0.8982035928143712,
    'tests/fixtures/adversarial/a/16_equivalence_clique.ttl': 0.9868421052631579,
    'tests/fixtures/adversarial/a/17_equivalence_clique_reversed.ttl': 0.9868421052631579,
    'tests/fixtures/adversarial/a/18_shuffled_base_pattern.ttl': None,
    'tests/fixtures/adversarial/a/19_base_plus_padding_island.ttl': 0.14258555133079848,
    'tests/fixtures/adversarial/a/20_base_replicated_x10.ttl': 0.9868421052631579,
    'tests/fixtures/adversarial/a/21_large_genuine_taxonomy.ttl': 0.9836918806384455,
    'tests/fixtures/adversarial/a/22_base_tiny_genuine_pattern.ttl': 0.9868421052631579,
    'tests/fixtures/adversarial/a/23_tiny_restriction_odp.ttl': 0.9930527176134042,
    'tests/fixtures/adversarial/a/24_abox_exemplification_odp.ttl': 0.997920997920998,
    'tests/fixtures/adversarial/a/25_domain_only_odp.ttl': 0.9836065573770492,
    'tests/fixtures/adversarial/a/26_nonstandard_rdfs_property_odp.ttl': 0.9836065573770492,
    'tests/fixtures/adversarial/b/01_cross_product_domain_range.ttl': 0.0,
    'tests/fixtures/adversarial/b/02_universal_top_hub.ttl': 0.0,
    'tests/fixtures/adversarial/b/03_bottom_hub_subclass_of_everything.ttl': 0.0,
    'tests/fixtures/adversarial/b/04_equivalent_class_clique.ttl': 0.3967670830271859,
    'tests/fixtures/adversarial/b/05_equivalent_class_clique_reversed.ttl': 0.3967670830271859,
    'tests/fixtures/adversarial/b/06_deep_subclass_chain_no_properties.ttl': 0.0,
    'tests/fixtures/adversarial/b/07_dangling_domain_range.ttl': 0.0,
    'tests/fixtures/adversarial/b/08_inverse_symmetric_reachability.ttl': 0.0,
    'tests/fixtures/adversarial/b/09_sigderived_from_cq_key_2025_151_01.ttl': 0.0,
    'tests/fixtures/adversarial/b/10_sigderived_from_cq_key_2023_133_01.ttl': 0.9404283801874164,
    'tests/fixtures/adversarial/b/11_shuffle_gold_2023_133_01.ttl': 0.9993419970389866,
    'tests/fixtures/adversarial/b/12_restriction_cross_product.ttl': 0.0,
    'tests/fixtures/adversarial/b/13_abox_star_hub.ttl': 0.9975369458128078,
    'tests/fixtures/adversarial/b/14_core_small_genuine.ttl': 0.9919177075679647,
    'tests/fixtures/adversarial/b/15_core_plus_selfconnected_padding.ttl': 0.13029630344561335,
    'tests/fixtures/adversarial/b/16_replication_x3_of_core.ttl': 0.9919177075679647,
    'tests/fixtures/adversarial/b/17_vocab_only_gold_2023_133_01.ttl': 0.0,
    'tests/fixtures/adversarial/b/18_refusal_prose_plus_176_bare_properties.ttl': 0.0,
    'tests/fixtures/adversarial/b/19_gold_2023_133_01_baseline.ttl': 0.9998457028236384,
    'tests/fixtures/adversarial/b/20_gold_2023_133_01_minus_one_subclassof.ttl': 0.9997636492554951,
    'tests/fixtures/adversarial/b/21_gold_2025_153_01_baseline.ttl': 0.9783621492908169,
    'tests/fixtures/adversarial/b/22_gold_2025_153_01_minus_two_subclassof.ttl': 0.9783621492908169,
    'tests/fixtures/adversarial/b/23_abox_minimal_genuine.ttl': 0.9960649286768323,
    'tests/fixtures/adversarial/b/24_tiny_restriction_odp.ttl': 0.9973614775725593,
    'tests/fixtures/adversarial/b/25_domain_only_genuine.ttl': 0.9836065573770492,
    'tests/fixtures/adversarial/b/26_large_genuine_taxonomy.ttl': 0.9776659848526729,
    'tests/fixtures/adversarial/b/27_truncated_midaxiom.ttl.broken': 0.0,
    'tests/fixtures/adversarial/b/28_universal_thing_domain_range.ttl': 0.0,
}

# Per-role components for the gold and adversarial fixtures, which no other
# agent's work moves.  The corpus is covered by its scores alone.
BASELINE_COMPONENTS = {
    'data/ground_truth/2023-133-01.ttl': {
        'property_connectivity': 1.0,
        'class_connectivity': 0.999735519703782,
        'individual_connectivity': None,
        'pitfall_freedom': None,
    },
    'data/ground_truth/2023-133-02.ttl': {
        'property_connectivity': 1.0,
        'class_connectivity': 0.999735519703782,
        'individual_connectivity': None,
        'pitfall_freedom': None,
    },
    'data/ground_truth/2023-134-01.ttl': {
        'property_connectivity': None,
        'class_connectivity': None,
        'individual_connectivity': 0.9973924380704041,
        'pitfall_freedom': None,
    },
    'data/ground_truth/2023-134-02.ttl': {
        'property_connectivity': None,
        'class_connectivity': None,
        'individual_connectivity': 0.9973136333109469,
        'pitfall_freedom': None,
    },
    'data/ground_truth/2023-134-03.ttl': {
        'property_connectivity': None,
        'class_connectivity': None,
        'individual_connectivity': 0.9977259806708357,
        'pitfall_freedom': None,
    },
    'data/ground_truth/2025-151-01.ttl': {
        'property_connectivity': 0.967741935483871,
        'class_connectivity': 1.0,
        'individual_connectivity': None,
        'pitfall_freedom': None,
    },
    'data/ground_truth/2025-151-02.ttl': {
        'property_connectivity': 0.967741935483871,
        'class_connectivity': 1.0,
        'individual_connectivity': None,
        'pitfall_freedom': None,
    },
    'data/ground_truth/2025-153-01.ttl': {
        'property_connectivity': 0.9575826358348702,
        'class_connectivity': 0.9980509442838901,
        'individual_connectivity': None,
        'pitfall_freedom': None,
    },
    'data/ground_truth/2026-155-01.ttl': {
        'property_connectivity': 0.998766954377312,
        'class_connectivity': 0.9605512023777357,
        'individual_connectivity': None,
        'pitfall_freedom': None,
    },
    'tests/fixtures/adversarial/a/01_empty.ttl': {
        'property_connectivity': None,
        'class_connectivity': None,
        'individual_connectivity': None,
        'pitfall_freedom': None,
    },
    'tests/fixtures/adversarial/a/02_whitespace_only.ttl': {
        'property_connectivity': None,
        'class_connectivity': None,
        'individual_connectivity': None,
        'pitfall_freedom': None,
    },
    'tests/fixtures/adversarial/a/03_prefix_block_only.ttl': {
        'property_connectivity': None,
        'class_connectivity': None,
        'individual_connectivity': None,
        'pitfall_freedom': None,
    },
    'tests/fixtures/adversarial/a/04_comment_block_only.ttl': {
        'property_connectivity': None,
        'class_connectivity': None,
        'individual_connectivity': None,
        'pitfall_freedom': None,
    },
    'tests/fixtures/adversarial/a/05_ontology_header_refusal_comment.ttl': {
        'property_connectivity': None,
        'class_connectivity': None,
        'individual_connectivity': None,
        'pitfall_freedom': None,
    },
    'tests/fixtures/adversarial/a/06_bare_declarations_176_objectproperties.ttl': {
        'property_connectivity': 0.0,
        'class_connectivity': None,
        'individual_connectivity': None,
        'pitfall_freedom': None,
    },
    'tests/fixtures/adversarial/a/07_classes_only_no_axioms.ttl': {
        'property_connectivity': None,
        'class_connectivity': 0.0,
        'individual_connectivity': None,
        'pitfall_freedom': None,
    },
    'tests/fixtures/adversarial/a/08_properties_no_domain_range.ttl': {
        'property_connectivity': 0.0,
        'class_connectivity': 0.0,
        'individual_connectivity': None,
        'pitfall_freedom': None,
    },
    'tests/fixtures/adversarial/a/09_annotation_properties_only.ttl': {
        'property_connectivity': None,
        'class_connectivity': None,
        'individual_connectivity': None,
        'pitfall_freedom': None,
    },
    'tests/fixtures/adversarial/a/10_refusal_prose_raw.ttl': {
        'property_connectivity': None,
        'class_connectivity': None,
        'individual_connectivity': None,
        'pitfall_freedom': None,
    },
    'tests/fixtures/adversarial/a/11_binary_garbage.ttl': {
        'property_connectivity': None,
        'class_connectivity': None,
        'individual_connectivity': None,
        'pitfall_freedom': None,
    },
    'tests/fixtures/adversarial/a/12_e2_polished_hollow.ttl': {
        'property_connectivity': 0.0,
        'class_connectivity': 0.0,
        'individual_connectivity': None,
        'pitfall_freedom': None,
    },
    'tests/fixtures/adversarial/a/13_single_junk_triple.ttl': {
        'property_connectivity': None,
        'class_connectivity': None,
        'individual_connectivity': 0.967741935483871,
        'pitfall_freedom': None,
    },
    'tests/fixtures/adversarial/a/14_universal_domain_range.ttl': {
        'property_connectivity': 0.0,
        'class_connectivity': 0.0,
        'individual_connectivity': None,
        'pitfall_freedom': None,
    },
    'tests/fixtures/adversarial/a/15_bottom_hub_over_base.ttl': {
        'property_connectivity': 0.9875640087783467,
        'class_connectivity': 0.823672971323978,
        'individual_connectivity': None,
        'pitfall_freedom': None,
    },
    'tests/fixtures/adversarial/a/16_equivalence_clique.ttl': {
        'property_connectivity': 0.9875640087783467,
        'class_connectivity': 0.9861212563915267,
        'individual_connectivity': None,
        'pitfall_freedom': None,
    },
    'tests/fixtures/adversarial/a/17_equivalence_clique_reversed.ttl': {
        'property_connectivity': 0.9875640087783467,
        'class_connectivity': 0.9861212563915267,
        'individual_connectivity': None,
        'pitfall_freedom': None,
    },
    'tests/fixtures/adversarial/a/18_shuffled_base_pattern.ttl': {
        'property_connectivity': None,
        'class_connectivity': None,
        'individual_connectivity': None,
        'pitfall_freedom': None,
    },
    'tests/fixtures/adversarial/a/19_base_plus_padding_island.ttl': {
        'property_connectivity': 0.9875640087783467,
        'class_connectivity': 0.07683988843986567,
        'individual_connectivity': None,
        'pitfall_freedom': None,
    },
    'tests/fixtures/adversarial/a/20_base_replicated_x10.ttl': {
        'property_connectivity': 0.9875640087783467,
        'class_connectivity': 0.9861212563915267,
        'individual_connectivity': None,
        'pitfall_freedom': None,
    },
    'tests/fixtures/adversarial/a/21_large_genuine_taxonomy.ttl': {
        'property_connectivity': 0.989010989010989,
        'class_connectivity': 0.9821826280623608,
        'individual_connectivity': None,
        'pitfall_freedom': None,
    },
    'tests/fixtures/adversarial/a/22_base_tiny_genuine_pattern.ttl': {
        'property_connectivity': 0.9875640087783467,
        'class_connectivity': 0.9861212563915267,
        'individual_connectivity': None,
        'pitfall_freedom': None,
    },
    'tests/fixtures/adversarial/a/23_tiny_restriction_odp.ttl': {
        'property_connectivity': 0.9933774834437086,
        'class_connectivity': 0.9926470588235294,
        'individual_connectivity': None,
        'pitfall_freedom': None,
    },
    'tests/fixtures/adversarial/a/24_abox_exemplification_odp.ttl': {
        'property_connectivity': None,
        'class_connectivity': None,
        'individual_connectivity': 0.997920997920998,
        'pitfall_freedom': None,
    },
    'tests/fixtures/adversarial/a/25_domain_only_odp.ttl': {
        'property_connectivity': 0.967741935483871,
        'class_connectivity': 1.0,
        'individual_connectivity': None,
        'pitfall_freedom': None,
    },
    'tests/fixtures/adversarial/a/26_nonstandard_rdfs_property_odp.ttl': {
        'property_connectivity': 0.967741935483871,
        'class_connectivity': 1.0,
        'individual_connectivity': None,
        'pitfall_freedom': None,
    },
    'tests/fixtures/adversarial/b/01_cross_product_domain_range.ttl': {
        'property_connectivity': 0.0,
        'class_connectivity': 0.0,
        'individual_connectivity': None,
        'pitfall_freedom': None,
    },
    'tests/fixtures/adversarial/b/02_universal_top_hub.ttl': {
        'property_connectivity': None,
        'class_connectivity': 0.0,
        'individual_connectivity': None,
        'pitfall_freedom': None,
    },
    'tests/fixtures/adversarial/b/03_bottom_hub_subclass_of_everything.ttl': {
        'property_connectivity': None,
        'class_connectivity': 0.0,
        'individual_connectivity': None,
        'pitfall_freedom': None,
    },
    'tests/fixtures/adversarial/b/04_equivalent_class_clique.ttl': {
        'property_connectivity': 0.9926470588235294,
        'class_connectivity': 0.24793388429752067,
        'individual_connectivity': None,
        'pitfall_freedom': None,
    },
    'tests/fixtures/adversarial/b/05_equivalent_class_clique_reversed.ttl': {
        'property_connectivity': 0.9926470588235294,
        'class_connectivity': 0.24793388429752067,
        'individual_connectivity': None,
        'pitfall_freedom': None,
    },
    'tests/fixtures/adversarial/b/06_deep_subclass_chain_no_properties.ttl': {
        'property_connectivity': None,
        'class_connectivity': 0.0,
        'individual_connectivity': None,
        'pitfall_freedom': None,
    },
    'tests/fixtures/adversarial/b/07_dangling_domain_range.ttl': {
        'property_connectivity': 0.0,
        'class_connectivity': None,
        'individual_connectivity': None,
        'pitfall_freedom': None,
    },
    'tests/fixtures/adversarial/b/08_inverse_symmetric_reachability.ttl': {
        'property_connectivity': 0.0,
        'class_connectivity': 0.0,
        'individual_connectivity': None,
        'pitfall_freedom': None,
    },
    'tests/fixtures/adversarial/b/09_sigderived_from_cq_key_2025_151_01.ttl': {
        'property_connectivity': 0.0,
        'class_connectivity': 0.0,
        'individual_connectivity': None,
        'pitfall_freedom': None,
    },
    'tests/fixtures/adversarial/b/10_sigderived_from_cq_key_2023_133_01.ttl': {
        'property_connectivity': 0.7317073170731707,
        'class_connectivity': 0.9661654135338346,
        'individual_connectivity': None,
        'pitfall_freedom': None,
    },
    'tests/fixtures/adversarial/b/11_shuffle_gold_2023_133_01.ttl': {
        'property_connectivity': 0.9983566146261298,
        'class_connectivity': 1.0,
        'individual_connectivity': None,
        'pitfall_freedom': None,
    },
    'tests/fixtures/adversarial/b/12_restriction_cross_product.ttl': {
        'property_connectivity': 0.0,
        'class_connectivity': 0.0,
        'individual_connectivity': None,
        'pitfall_freedom': None,
    },
    'tests/fixtures/adversarial/b/13_abox_star_hub.ttl': {
        'property_connectivity': 1.0,
        'class_connectivity': None,
        'individual_connectivity': 0.996309963099631,
        'pitfall_freedom': None,
    },
    'tests/fixtures/adversarial/b/14_core_small_genuine.ttl': {
        'property_connectivity': 0.9914320685434517,
        'class_connectivity': 0.9922417313189057,
        'individual_connectivity': None,
        'pitfall_freedom': None,
    },
    'tests/fixtures/adversarial/b/15_core_plus_selfconnected_padding.ttl': {
        'property_connectivity': 0.9914320685434517,
        'class_connectivity': 0.08251553533226935,
        'individual_connectivity': None,
        'pitfall_freedom': None,
    },
    'tests/fixtures/adversarial/b/16_replication_x3_of_core.ttl': {
        'property_connectivity': 0.9914320685434517,
        'class_connectivity': 0.9922417313189057,
        'individual_connectivity': None,
        'pitfall_freedom': None,
    },
    'tests/fixtures/adversarial/b/17_vocab_only_gold_2023_133_01.ttl': {
        'property_connectivity': 0.0,
        'class_connectivity': 0.0,
        'individual_connectivity': None,
        'pitfall_freedom': None,
    },
    'tests/fixtures/adversarial/b/18_refusal_prose_plus_176_bare_properties.ttl': {
        'property_connectivity': 0.0,
        'class_connectivity': None,
        'individual_connectivity': None,
        'pitfall_freedom': None,
    },
    'tests/fixtures/adversarial/b/19_gold_2023_133_01_baseline.ttl': {
        'property_connectivity': 1.0,
        'class_connectivity': 0.999735519703782,
        'individual_connectivity': None,
        'pitfall_freedom': None,
    },
    'tests/fixtures/adversarial/b/20_gold_2023_133_01_minus_one_subclassof.ttl': {
        'property_connectivity': 1.0,
        'class_connectivity': 0.9995886466474702,
        'individual_connectivity': None,
        'pitfall_freedom': None,
    },
    'tests/fixtures/adversarial/b/21_gold_2025_153_01_baseline.ttl': {
        'property_connectivity': 0.9575826358348702,
        'class_connectivity': 0.9980509442838901,
        'individual_connectivity': None,
        'pitfall_freedom': None,
    },
    'tests/fixtures/adversarial/b/22_gold_2025_153_01_minus_two_subclassof.ttl': {
        'property_connectivity': 0.9575826358348702,
        'class_connectivity': 0.9980509442838901,
        'individual_connectivity': None,
        'pitfall_freedom': None,
    },
    'tests/fixtures/adversarial/b/23_abox_minimal_genuine.ttl': {
        'property_connectivity': None,
        'class_connectivity': None,
        'individual_connectivity': 0.9960649286768323,
        'pitfall_freedom': None,
    },
    'tests/fixtures/adversarial/b/24_tiny_restriction_odp.ttl': {
        'property_connectivity': 0.998766954377312,
        'class_connectivity': 0.996309963099631,
        'individual_connectivity': None,
        'pitfall_freedom': None,
    },
    'tests/fixtures/adversarial/b/25_domain_only_genuine.ttl': {
        'property_connectivity': 0.967741935483871,
        'class_connectivity': 1.0,
        'individual_connectivity': None,
        'pitfall_freedom': None,
    },
    'tests/fixtures/adversarial/b/26_large_genuine_taxonomy.ttl': {
        'property_connectivity': 0.9886297937945654,
        'class_connectivity': 0.9732558139534884,
        'individual_connectivity': None,
        'pitfall_freedom': None,
    },
    'tests/fixtures/adversarial/b/27_truncated_midaxiom.ttl.broken': {
        'property_connectivity': None,
        'class_connectivity': None,
        'individual_connectivity': None,
        'pitfall_freedom': None,
    },
    'tests/fixtures/adversarial/b/28_universal_thing_domain_range.ttl': {
        'property_connectivity': 0.0,
        'class_connectivity': 0.0,
        'individual_connectivity': None,
        'pitfall_freedom': None,
    },
}


def _existing(mapping):
    """Skip artefacts another agent's work has moved; fail if too few remain."""
    present = {k: v for k, v in mapping.items()
               if os.path.exists(os.path.join(REPO_ROOT, k))}
    assert len(present) >= int(0.8 * len(mapping)), (
        "only %d of %d baseline artefacts are on disk; the baseline can no "
        "longer certify bit-identity" % (len(present), len(mapping))
    )
    return present


def test_baseline_covers_gold_adversarial_and_corpus():
    """The baseline is the constraint, so its shape is asserted too."""
    gold = [k for k in BASELINE_SCORES if k.startswith("data/ground_truth/")]
    adversarial = [k for k in BASELINE_SCORES if "fixtures/adversarial" in k]
    corpus = [k for k in BASELINE_SCORES if k.startswith("outputs/")]
    assert len(gold) == 9
    assert len(adversarial) == 54
    assert len(corpus) == 100
    moving = [k for k in corpus if BASELINE_SCORES[k] not in (0.0, None)]
    assert len(set(BASELINE_SCORES[k] for k in moving)) >= 40, (
        "a baseline of constants could not detect a change"
    )


@pytest.mark.parametrize("relpath", sorted(BASELINE_SCORES))
def test_score_is_bit_identical_to_the_pre_fix_baseline(relpath):
    """``==`` on the float, not ``approx``.  A fix that moves any of these is
    the wrong fix."""
    path = os.path.join(REPO_ROOT, relpath)
    if not os.path.exists(path):
        pytest.skip("artefact not on disk: %s" % relpath)
    expected = BASELINE_SCORES[relpath]
    got = odp_structural.evaluate_path(path)["score"]
    assert got == expected, "%s: %r -> %r" % (relpath, expected, got)


@pytest.mark.parametrize("relpath", sorted(BASELINE_COMPONENTS))
def test_components_are_bit_identical_to_the_pre_fix_baseline(relpath):
    path = os.path.join(REPO_ROOT, relpath)
    if not os.path.exists(path):
        pytest.skip("artefact not on disk: %s" % relpath)
    got = odp_structural.evaluate_path(path)["components"]
    assert got == BASELINE_COMPONENTS[relpath], relpath


def test_no_baseline_artefact_is_missing_from_disk():
    _existing(BASELINE_SCORES)


def test_published_adversarial_results_are_unchanged():
    """The four constructions the paper reports as beating the gold floor.

    They are a finding, not a bug.  Pinned so that nothing done for speed can
    suppress them.
    """
    floor = BASELINE_SCORES["data/ground_truth/2025-153-01.ttl"]
    assert floor == 0.9783621492908169  # the reported gold floor
    beaters = {
        "tests/fixtures/adversarial/b/11_shuffle_gold_2023_133_01.ttl":
            0.9993419970389866,
        "tests/fixtures/adversarial/b/13_abox_star_hub.ttl":
            0.9975369458128078,
        "tests/fixtures/adversarial/a/16_equivalence_clique.ttl":
            0.9868421052631579,
        "tests/fixtures/adversarial/a/17_equivalence_clique_reversed.ttl":
            0.9868421052631579,
    }
    for rel, reported in beaters.items():
        path = os.path.join(REPO_ROOT, rel)
        if not os.path.exists(path):
            pytest.skip("fixture not on disk: %s" % rel)
        score = odp_structural.evaluate_path(path)["score"]
        assert score == BASELINE_SCORES[rel]
        assert score == reported
        assert score > floor, (
            "%s no longer beats the gold floor; that result is a published "
            "finding and must not be tuned away" % rel
        )
