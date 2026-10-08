"""Negative controls for the structural metric (defect F3 / V1).

Before the content gate landed, the structural metric's OPTIMUM was the empty
file:

    structural_score = mean(consistency, 1 / (1 + oops_pitfalls_total))

An empty graph is trivially consistent (OWL-RL reports ``consistent: True`` on
zero triples) and has zero OOPS! pitfalls, so it scored ``mean(1.0, 1.0) = 1.0``
-- strictly at the top of the scale, tying the published gold reference
patterns.  Five of the 70 ``bigscience_bloomz-7b1`` outputs in this repository
are comment-only files that parse to zero triples and therefore scored 1.0
today; under a fixed Turtle extractor all 70 would.

Every test in this file is a NEGATIVE CONTROL: it asserts that a worthless
ontology cannot reach the top of the scale.  Two positive controls guard the
other direction, so the gate cannot be "passed" by a metric that simply punishes
everything.

The tests drive the real scoring path, ``batch_evaluate.score_structural``.
When that function is absent the helper below falls back to the pre-fix
published formula, so this file fails loudly (reporting the actual degenerate
1.0) rather than erroring out on an AttributeError.
"""

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import batch_evaluate as be  # noqa: E402

GROUND_TRUTH = REPO_ROOT / "data" / "ground_truth"
BLOOMZ_DIR = REPO_ROOT / "outputs" / "bigscience_bloomz-7b1"

# Gold reference patterns used as the positive control.  Excluded:
#   * 2024-145-01.owl  -- 15 MB / ~1.6 M triples; OWL-RL closure on it does not
#     terminate in test time.  Its size is not in question here.
#   * 2025-147-01.owl, 2025-150-01.owl -- OWL/XML serialisation, which rdflib
#     (and therefore ``_parse_graph``) cannot read at all.  That is a separate,
#     real defect in the harness and is reported as such; including them here
#     would only measure that bug, not this one.
GOLD_REFERENCES = [
    "2023-133-01.ttl", "2023-133-02.ttl", "2023-134-01.ttl", "2023-134-02.ttl",
    "2023-134-03.ttl", "2023-135-01.owl", "2025-149-01.rdf", "2025-151-01.ttl",
    "2025-151-02.ttl", "2025-153-01.ttl", "2026-155-01.ttl",
]

# A real ODP from the gold set: 90 triples, 5 owl:Class, 5 owl:ObjectProperty.
GOOD_ONTOLOGY = GROUND_TRUTH / "2023-133-01.ttl"

EMPTY = ""
PREFIX_ONLY = "@prefix : <http://ex.org/#> ."
UNPARSEABLE = "- What if the pavement is wet, how would it effect the pavement being slippery?"

# 6 triples, exactly one class, zero properties of any kind.
SIX_TRIPLE_ONE_CLASS = """\
@prefix :    <http://ex.org/#> .
@prefix owl: <http://www.w3.org/2002/07/owl#> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .

:Onto a owl:Ontology ;
      rdfs:label "stub" ;
      rdfs:comment "a stub" .
:Thing a owl:Class ;
       rdfs:label "Thing" ;
       rdfs:comment "a thing" .
"""


# ── helpers ───────────────────────────────────────────────────────────────────

def _legacy_published_score(ontology_text, oops=None):
    """The PRE-FIX formula, reproduced from scripts/audit_artifacts.py.

    Kept only so this file reports the real degenerate value (1.0) when the gate
    is missing, instead of dying on AttributeError.  Once
    ``score_structural`` exists this branch is never taken.
    """
    try:
        g, _ = be._parse_graph(ontology_text or "")
    except ValueError:
        return {"structural_score": None, "gate": "unparseable"}
    reasoner = be.run_reasoner(g)
    consistency = 1.0 if reasoner.get("consistent") else 0.0
    pitfalls = 0 if oops is None else float(oops.get("pitfalls_total", 0) or 0)
    oops_component = 1.0 / (1.0 + pitfalls)
    return {"structural_score": (consistency + oops_component) / 2.0,
            "gate": "legacy-no-gate"}


def score(ontology_text, oops=None):
    """Score one ontology through the real structural path."""
    fn = getattr(be, "score_structural", None)
    if fn is None:
        return _legacy_published_score(ontology_text, oops)
    return fn(ontology_text, oops=oops)


def score_value(ontology_text, oops=None):
    return score(ontology_text, oops)["structural_score"]


def _read(path):
    return path.read_text(encoding="utf-8", errors="replace")


def _mean(values):
    values = list(values)
    return sum(values) / len(values) if values else 0.0


# ── negative controls: the degenerate optima ─────────────────────────────────

def test_empty_string_scores_zero():
    """An empty ontology.ttl is the worst possible output, not the best."""
    assert score_value(EMPTY) == 0.0


def test_parseable_but_zero_triple_ontology_scores_zero():
    """Parses fine, says nothing.  Zero triples means zero structural credit."""
    result = score(PREFIX_ONLY)
    assert result["structural_score"] == 0.0


def test_unparseable_ontology_scores_zero():
    """A prose blob is not an ontology.  It floors; it does not error out."""
    assert score_value(UNPARSEABLE) == 0.0


def test_zero_triple_ontology_stays_in_the_denominator():
    """The floor must be a real 0.0 that is COUNTED, never a skip/None.

    Excluding empty files from the aggregate is the same bug wearing a
    different hat: it lets a model that emits nothing vanish from its own mean.
    """
    for text in (EMPTY, PREFIX_ONLY, UNPARSEABLE):
        result = score(text)
        assert result["structural_score"] == 0.0
        assert result.get("counted") is True
        assert not result.get("skipped")


def test_empty_graph_consistency_is_not_full_credit():
    """OWL-RL says consistent:True on 0 triples.  That is vacuous, not quality."""
    result = score(PREFIX_ONLY)
    assert result["consistency_component"] == 0.0


def test_zero_pitfalls_on_an_empty_file_earns_no_oops_credit():
    """An empty file cannot commit a pitfall, so 0 pitfalls proves nothing."""
    clean_oops = {"status_code": 200, "pitfalls_total": 0,
                  "important_count": 0, "critical_count": 0}
    assert score_value(PREFIX_ONLY, oops=clean_oops) == 0.0
    assert score_value(EMPTY, oops=clean_oops) == 0.0


def test_six_triple_one_class_no_properties_cannot_reach_075():
    """A near-empty stub must not be able to climb the top of the scale."""
    clean_oops = {"status_code": 200, "pitfalls_total": 0}
    assert score_value(SIX_TRIPLE_ONE_CLASS, oops=clean_oops) < 0.75
    assert score_value(SIX_TRIPLE_ONE_CLASS) < 0.75


def test_run_reasoner_flags_vacuous_consistency_on_an_empty_graph():
    """run_reasoner must say out loud that its verdict is vacuous."""
    from rdflib import Graph
    result = be.run_reasoner(Graph())
    assert result["consistent"] is True          # unchanged: it IS consistent
    assert result["vacuous"] is True             # ...but vacuously so
    assert result["consistency_credit"] == 0.0   # so it earns nothing


def test_run_reasoner_does_not_flag_a_real_ontology_as_vacuous():
    g, _ = be._parse_graph(_read(GOOD_ONTOLOGY))
    result = be.run_reasoner(g)
    assert result["vacuous"] is False
    assert result["consistency_credit"] == 1.0


# ── the gate's two tunable constants ─────────────────────────────────────────
#
# Everything above tests the gate at its extremes -- 0 triples on one side, a
# real 90-triple ODP on the other -- and every one of those tests still passes
# with MIN_STRUCTURAL_TRIPLES lowered from 10 to 1.  That is exactly the hole:
# the near-empty region the gate exists to close could be silently reopened by
# editing one number, and the suite would stay green.  The tests below pin both
# constants: the value, the justification it is documented with, and the
# behaviour at the boundary in both directions.

_GATE_PREFIXES = [
    "@prefix : <http://ex.org/#> .",
    "@prefix owl: <http://www.w3.org/2002/07/owl#> .",
    "@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .",
]


def _ontology_with_triples(n):
    """A T-Box with exactly `n` triples that declares a class AND a property.

    Both other near-empty reasons ("declares no classes", "declares no
    properties") are therefore satisfied, so the ONLY thing the gate can object
    to is the triple count.  That isolates MIN_STRUCTURAL_TRIPLES.
    """
    if n < 6:
        raise ValueError("the base T-Box is already 6 triples")
    lines = list(_GATE_PREFIXES) + [
        ":Onto a owl:Ontology .",
        ":A a owl:Class .",
        ":B a owl:Class .",
        ":p a owl:ObjectProperty .",
        ":p rdfs:domain :A .",
        ":p rdfs:range :B .",
    ]
    lines += [':Onto rdfs:comment "pad %d" .' % i for i in range(n - 6)]
    return "\n".join(lines) + "\n"


# Exactly one triple below / exactly at the documented threshold of 10.
NINE_TRIPLES = _ontology_with_triples(9)
TEN_TRIPLES = _ontology_with_triples(10)

CLEAN_OOPS = {"status_code": 200, "pitfalls_total": 0}


def test_the_boundary_fixtures_really_are_9_and_10_triples():
    """The two fixtures below are only meaningful at those exact sizes."""
    assert len(be._parse_graph(NINE_TRIPLES)[0]) == 9
    assert len(be._parse_graph(TEN_TRIPLES)[0]) == 10
    for text in (NINE_TRIPLES, TEN_TRIPLES):
        gate = be.structural_content_gate(be._parse_graph(text)[0])
        assert gate["class_terms"] >= 1 and gate["property_terms"] >= 1, (
            "the fixture must be near-empty for its SIZE alone, not because it "
            "declares no vocabulary")


def test_min_structural_triples_constant_is_pinned():
    """The near-empty threshold is a published number, not a free parameter."""
    assert be.MIN_STRUCTURAL_TRIPLES == 10


def test_near_empty_score_cap_constant_is_pinned():
    assert be.NEAR_EMPTY_SCORE_CAP == 0.5


def test_one_triple_below_the_threshold_is_still_gated_near_empty():
    """Lowering MIN_STRUCTURAL_TRIPLES reopens the near-empty region.

    A 9-triple stub with one property and two classes is a plausible "looks like
    an ontology" output.  With a clean reasoner and a clean OOPS! verdict its
    UNCAPPED score is 1.0 -- top of the scale.  The gate is the only thing
    standing between that file and the published reference patterns.
    """
    result = be.score_structural(NINE_TRIPLES, oops=CLEAN_OOPS)
    assert result["triples_count"] == 9
    assert result["gate"] == "near_empty", result
    assert result["gate_cap"] == be.NEAR_EMPTY_SCORE_CAP, result
    assert result["raw_score"] == 1.0, (
        "precondition: without the cap this stub scores at the top of the scale")
    assert result["structural_score"] == be.NEAR_EMPTY_SCORE_CAP, result
    assert result["structural_score"] < score_value(_read(GOOD_ONTOLOGY),
                                                    oops=CLEAN_OOPS)


def test_at_the_threshold_the_gate_opens_again():
    """The other direction: raising the constant must not start capping real
    patterns.  10 triples is the documented threshold, so 10 triples passes."""
    result = be.score_structural(TEN_TRIPLES, oops=CLEAN_OOPS)
    assert result["triples_count"] == 10
    assert result["gate"] == "ok", result
    assert result["gate_cap"] == 1.0, result
    assert result["structural_score"] > be.NEAR_EMPTY_SCORE_CAP, result


def test_the_near_empty_cap_is_the_value_a_capped_file_actually_receives():
    """NEAR_EMPTY_SCORE_CAP must be the cap in force, not a decorative constant.

    Both near-empty shapes -- too few triples, and no declared properties -- have
    an uncapped score of 1.0 here, so whatever they come out at IS the cap.
    """
    for text in (NINE_TRIPLES, SIX_TRIPLE_ONE_CLASS):
        result = be.score_structural(text, oops=CLEAN_OOPS)
        assert result["gate"] == "near_empty", result
        assert result["raw_score"] == 1.0, result
        assert result["structural_score"] == be.NEAR_EMPTY_SCORE_CAP, result
    assert be.NEAR_EMPTY_SCORE_CAP < 0.75, (
        "a near-empty file must stay out of the top half of the scale")


def test_min_structural_triples_matches_its_documented_justification():
    """10 is documented as the size of the SMALLEST published reference pattern,
    so that nothing at least as large as an artifact the authors themselves treat
    as a real ODP is capped for size alone.  Pin that: no gold reference may be
    penalised by the triple threshold.
    """
    sizes = {}
    for name in GOLD_REFERENCES:
        g, _ = be._parse_graph(_read(GROUND_TRUTH / name))
        sizes[name] = len(g)
    assert min(sizes.values()) == be.MIN_STRUCTURAL_TRIPLES, sizes

    for name, n_triples in sizes.items():
        gate = be.structural_content_gate(be._parse_graph(_read(GROUND_TRUTH / name))[0])
        assert f"triples < {be.MIN_STRUCTURAL_TRIPLES}" not in gate["reason"], (
            f"{name} ({n_triples} triples) is a published reference pattern but "
            f"the threshold penalises it for size: {gate['reason']}")


# ── positive controls: do not over-correct ───────────────────────────────────

def test_a_genuinely_good_ontology_still_scores_high():
    """Guard against a gate that simply punishes everything."""
    clean_oops = {"status_code": 200, "pitfalls_total": 0}
    assert score_value(_read(GOOD_ONTOLOGY), oops=clean_oops) >= 0.9
    assert score_value(_read(GOOD_ONTOLOGY)) >= 0.9


def test_real_ontologies_are_ranked_above_near_empty_stubs():
    good = score_value(_read(GOOD_ONTOLOGY))
    assert good > score_value(SIX_TRIPLE_ONE_CLASS) > score_value(EMPTY)


# ── the corpus-level control ─────────────────────────────────────────────────

@pytest.fixture(scope="module")
def bloomz_scores():
    paths = sorted(BLOOMZ_DIR.rglob("ontology.ttl"))
    assert len(paths) == 70, f"expected 70 bloomz outputs, found {len(paths)}"
    return {p.relative_to(BLOOMZ_DIR).as_posix(): score_value(_read(p)) for p in paths}


@pytest.fixture(scope="module")
def gold_scores():
    out = {}
    for name in GOLD_REFERENCES:
        path = GROUND_TRUTH / name
        assert path.exists(), f"missing gold reference {path}"
        out[name] = score_value(_read(path))
    return out


def test_no_bloomz_output_beats_the_best_gold_reference(bloomz_scores, gold_scores):
    assert max(bloomz_scores.values()) < max(gold_scores.values())


def test_bloomz_corpus_mean_is_far_below_the_gold_mean(bloomz_scores, gold_scores):
    assert _mean(bloomz_scores.values()) < _mean(gold_scores.values())


def test_every_bloomz_output_is_floored(bloomz_scores):
    """All 70 are prose blobs or comment-only stubs.  None of them is an ODP."""
    offenders = {k: v for k, v in bloomz_scores.items() if v != 0.0}
    assert not offenders, f"non-zero structural score for empty outputs: {offenders}"
