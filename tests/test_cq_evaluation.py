"""Tests for the CQ-evaluation half of batch_evaluate.py (defects F4 and F5).

F4 — competency questions must come ONLY from
     data/scenarios/pattern_scenarios.json ("cq_list"), keyed by scenario_id.
     Anything else (in particular the prompt boilerplate that used to be scraped
     out of prompt.txt) must raise, loudly, forever.

F5 — a CQ is verified by SEV (Schema-Entailment Verification) over the lifted
     T-Box, not by running an LLM-written instance-level SPARQL SELECT and
     checking for rows.  The ODPs are schema-only by construction, so the old
     criterion measured nothing.

Everything here is offline: rdflib only, no LLM, no network, no API key.
"""

import copy
import importlib.util
import json
import sys
from pathlib import Path

import pytest
from rdflib import Graph

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _load_batch_evaluate():
    spec = importlib.util.spec_from_file_location(
        "batch_evaluate", REPO_ROOT / "batch_evaluate.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["batch_evaluate"] = mod
    spec.loader.exec_module(mod)
    return mod


be = _load_batch_evaluate()

SCENARIO = "2023-133-01"          # the causal pattern; exactly 4 real CQs
GROUND_TRUTH = REPO_ROOT / "data" / "ground_truth"


# ── fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def sigs():
    """Signatures aligned to the authoritative cq_list of the causal scenario."""
    return be.signatures_for(SCENARIO)


# A correct, hand-written schema-only ODP for the causal scenario: the n-ary
# reification idiom (:Causes links events through role properties) plus an
# effect-weight role.  Contains no individuals at all, exactly as the generation
# prompts demand.
CORRECT_ODP = """
@prefix owl:  <http://www.w3.org/2002/07/owl#> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
@prefix :     <http://example.org/causal#> .

:Event        a owl:Class ; rdfs:label "Event" .
:Causes       a owl:Class ; rdfs:label "Causes" .
:EffectWeight a owl:Class ; rdfs:label "Effect Weight" .

:hasTreatment    a owl:ObjectProperty ; rdfs:domain :Causes ; rdfs:range :Event .
:hasOutcome      a owl:ObjectProperty ; rdfs:domain :Causes ; rdfs:range :Event .
:hasMediator     a owl:ObjectProperty ; rdfs:domain :Causes ; rdfs:range :Event .
:hasEffectWeight a owl:ObjectProperty ; rdfs:domain :Causes ; rdfs:range :EffectWeight .
"""

# The right nouns, and nothing that connects them.
CLASSES_ONLY_ODP = """
@prefix owl: <http://www.w3.org/2002/07/owl#> .
@prefix :    <http://example.org/causal#> .

:Event a owl:Class .
:Causes a owl:Class .
:Outcome a owl:Class .
:EffectWeight a owl:Class .
:Mediator a owl:Class .
"""

# The Llama-2 style: perfectly named vocabulary, zero rdfs:domain, zero
# rdfs:range, zero restrictions.  46/46 parseable Llama-2 outputs look like this.
NO_DOMAIN_RANGE_ODP = """
@prefix owl:  <http://www.w3.org/2002/07/owl#> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
@prefix :     <http://example.org/causal#> .

:Event a owl:Class ; rdfs:comment "An event that causes an outcome." .
:Causes a owl:Class .
:EffectWeight a owl:Class .
:hasTreatment a owl:ObjectProperty .
:hasOutcome a owl:ObjectProperty .
:hasMediator a owl:ObjectProperty .
:hasEffectWeight a owl:ObjectProperty .
"""

# Adversarial: declare every plausible name, anchor nothing anywhere.
NAME_STUFFING_ODP = """
@prefix owl:  <http://www.w3.org/2002/07/owl#> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
@prefix :     <http://example.org/causal#> .

:Event a owl:Class . :Occurrence a owl:Class . :Situation a owl:Class .
:State a owl:Class . :Node a owl:Class . :Outcome a owl:Class .
:Effect a owl:Class . :Consequence a owl:Class . :Result a owl:Class .
:Causes a owl:Class . :CausalRelation a owl:Class . :CausalLink a owl:Class .
:Mediator a owl:Class . :EffectWeight a owl:Class . :Strength a owl:Class .
:causes a owl:ObjectProperty ; rdfs:domain owl:Thing ; rdfs:range owl:Thing .
:hasOutcome a owl:ObjectProperty ; rdfs:domain owl:Thing ; rdfs:range owl:Thing .
:hasMediator a owl:ObjectProperty ; rdfs:domain owl:Thing ; rdfs:range owl:Thing .
:hasEffectWeight a owl:ObjectProperty ; rdfs:domain owl:Thing ; rdfs:range owl:Thing .
"""


# ── F4: CQ loading is sourced only from pattern_scenarios.json ───────────────

def test_load_cq_list_returns_exactly_the_scenario_cq_list():
    expected = json.loads(
        (REPO_ROOT / "data" / "scenarios" / "pattern_scenarios.json")
        .read_text(encoding="utf-8"))
    expected = next(s for s in expected if s["scenario_id"] == SCENARIO)["cq_list"]

    got = be.load_cq_list(SCENARIO)

    assert got == expected
    assert len(got) == 4, "2023-133-01 has exactly four real competency questions"
    assert got[0] == "What events are responsible for the pavement being slippery?"


def test_load_cq_list_rejects_unknown_scenario():
    with pytest.raises(KeyError):
        be.load_cq_list("9999-999-99")


def test_extract_cqs_from_prompt_text_is_gone():
    """The prompt.txt scraper is the F4 bug; it must not exist any more."""
    assert not hasattr(be, "extract_cqs")
    # prompt.txt may still be *mentioned* in the prose that explains the bug, but
    # no code may name it as a path any more.
    import ast
    tree = ast.parse((REPO_ROOT / "batch_evaluate.py").read_text(encoding="utf-8"))
    literals = [n.value for n in ast.walk(tree)
                if isinstance(n, ast.Constant) and isinstance(n.value, str)]
    assert "prompt.txt" not in literals
    # CQ verification must advertise where its CQs came from.
    result = be.run_cq_verification(CORRECT_ODP, SCENARIO)
    assert result["cq_source"] == "data/scenarios/pattern_scenarios.json#cq_list"


# ── F4: the contamination guard must RAISE, never warn-and-continue ──────────

PROMPT_BOILERPLATE = [
    "Use clear, self-explanatory class and property names?",
    "Do not add concepts or axioms not grounded in the scenario or CQs?",
    "Output valid Turtle syntax?",
    "Ensure the ODP structurally supports every CQ?",
]


@pytest.mark.parametrize("boilerplate", PROMPT_BOILERPLATE)
def test_prompt_boilerplate_raises(boilerplate):
    with pytest.raises(be.CQContaminationError):
        be.assert_cq_authentic(SCENARIO, boilerplate)


@pytest.mark.parametrize("boilerplate", PROMPT_BOILERPLATE)
def test_prompt_boilerplate_raises_through_the_scoring_entry_point(boilerplate):
    """The guard has to sit on the path that actually scores, not beside it."""
    with pytest.raises(be.CQContaminationError):
        be.run_cq_verification(CORRECT_ODP, SCENARIO, cqs=[boilerplate])


def test_contamination_error_names_the_offending_cq_and_the_real_list():
    with pytest.raises(be.CQContaminationError) as excinfo:
        be.assert_cq_authentic(
            SCENARIO, "Use clear, self-explanatory class and property names?")
    message = str(excinfo.value)
    assert "Use clear, self-explanatory class and property names?" in message
    assert "What events are responsible for the pavement being slippery?" in message


def test_a_single_contaminated_cq_in_an_otherwise_clean_batch_raises():
    real = be.load_cq_list(SCENARIO)
    with pytest.raises(be.CQContaminationError):
        be.assert_cqs_authentic(
            SCENARIO, real + ["Use clear, self-explanatory class and property names?"])


def test_authentic_cqs_do_not_raise():
    be.assert_cqs_authentic(SCENARIO, be.load_cq_list(SCENARIO))
    # trailing whitespace / casing / a missing question mark are formatting, not
    # contamination
    be.assert_cq_authentic(
        SCENARIO, "  what events are responsible for the pavement being slippery  ")


def test_stale_signature_referring_to_an_unknown_cq_raises(tmp_path):
    artifact = tmp_path / "cq_signatures.json"
    artifact.write_text(json.dumps({
        SCENARIO: [{"cq": "Use clear, self-explanatory class and property names?",
                    "slots": {}, "relations": []}]
    }), encoding="utf-8")
    with pytest.raises(be.SignatureError):
        be.signatures_for(SCENARIO, None, None, artifact)


def test_signatures_are_aligned_one_to_one_with_the_cq_list(sigs):
    cq_list = be.load_cq_list(SCENARIO)
    assert [s["cq"] for s in sigs] == cq_list
    assert all(s["signature_source"] == "authored" for s in sigs)


# ── F5 sanity case 1: a correct hand-written ODP scores near max ─────────────

def test_correct_hand_written_odp_scores_near_max(sigs):
    result = be.score_ontology(CORRECT_ODP, sigs)
    assert result["status"] == "ok"
    assert result["score"] >= 0.85, result
    assert result["connectivity"] >= 0.85


def test_correct_odp_is_schema_only_and_still_scores_near_max(sigs):
    """The whole point of F5: no individuals are needed to answer a CQ."""
    g = Graph()
    g.parse(data=CORRECT_ODP, format="turtle")
    from rdflib.namespace import OWL, RDF
    assert not list(g.subjects(RDF.type, OWL.NamedIndividual))
    assert be.score_ontology(g, sigs)["score"] >= 0.85


@pytest.mark.parametrize("scenario_id", sorted(be._BUILTIN_CQ_SIGNATURES))
def test_published_reference_pattern_clears_the_gold_calibration_gate(scenario_id):
    """Validity gate: a signature the published ODP cannot satisfy is a bad
    signature (or a CQ the reference genuinely does not cover, which must be
    declared out_of_scope rather than quietly dropped)."""
    report = be.run_gold_calibration([scenario_id])
    if scenario_id in report["exempt"]:
        pytest.skip(f"{scenario_id}: {report['exempt'][scenario_id]}")
    entry = report["scenarios"][scenario_id]
    failures = [c for c in entry["cqs"]
                if not c["gold_pass"] and c["cq_type"] != "out_of_scope"]
    assert not failures, (
        f"{scenario_id}: signatures below the {report['threshold']} gold gate: "
        + json.dumps(failures, indent=2))


def test_gold_calibration_flags_the_out_of_scope_cq_rather_than_hiding_it():
    report = be.run_gold_calibration(["2023-135-01"])
    unsupported = [c["cq"] for c in report["gold_unsupported"]]
    assert "Show the trajectories of rivers which cross national parks?" in unsupported
    # ... and a CQ that is declared out of the pattern's scope does not count as
    # a signature bug, while anything else would.
    assert report["failed"] == 1
    assert report["failed_unexpected"] == 0


def test_whole_signature_catalogue_passes_the_gate():
    report = be.run_gold_calibration()
    assert report["failed_unexpected"] == 0, json.dumps(
        report["gold_unsupported"], indent=2)
    assert report["passed"] >= 35


# ── F5 sanity case 2: an EMPTY ontology scores ZERO (and is not skipped) ─────

@pytest.mark.parametrize("ontology,expected_status", [
    ("", "no_ontology"),
    ("   \n  ", "no_ontology"),
    ("...", "no_ontology"),                       # the 3-byte F1/F2 artefact
    ("```turtle\n...\n```", "no_ontology"),
    ("@prefix : <http://example.org/> .", "empty"),
    ("this is not rdf at all, it is an apology from a chat model", "unparseable"),
])
def test_empty_or_broken_ontology_scores_zero(sigs, ontology, expected_status):
    result = be.score_ontology(ontology, sigs)
    assert result["status"] == expected_status
    assert result["score"] == 0.0
    assert result["coverage"] == 0.0
    assert result["connectivity"] == 0.0


def test_empty_ontology_is_counted_in_the_denominator_not_skipped():
    """F3 pointed the other way: the empty file must be the worst case, never a
    free pass and never an exclusion."""
    result = be.run_cq_verification("", SCENARIO)
    assert "skipped" not in result
    assert result["cqs_total"] == len(be.load_cq_list(SCENARIO)) == 4
    assert result["cqs_passed"] == 0
    assert result["pass_rate"] == 0.0
    assert result["sev_score"] == 0.0
    assert len(result["results"]) == 4
    assert all(r["status"] == "no_ontology" for r in result["results"])


def test_empty_ontology_scores_strictly_below_every_non_empty_candidate(sigs):
    empty = be.score_ontology("", sigs)["score"]
    for candidate in (CLASSES_ONLY_ODP, NO_DOMAIN_RANGE_ODP,
                      NAME_STUFFING_ODP, CORRECT_ODP):
        assert be.score_ontology(candidate, sigs)["score"] > empty


# ── F5 sanity case 3: right classes, nothing connecting them → low ───────────

def test_classes_without_connecting_properties_score_low(sigs):
    result = be.score_ontology(CLASSES_ONLY_ODP, sigs)
    assert result["status"] == "ok"
    assert result["coverage"] > 0.0, "the class names really are there"
    assert result["connectivity"] == 0.0, "but nothing connects them"
    assert result["score"] < 0.30, result
    assert result["score"] < be.score_ontology(CORRECT_ODP, sigs)["score"]


def test_name_stuffing_and_mega_hub_do_not_buy_connectivity(sigs):
    """owl:Thing is an open endpoint, not a class, so hanging every property off
    it must not connect everything to everything."""
    result = be.score_ontology(NAME_STUFFING_ODP, sigs)
    assert result["coverage"] > 0.8, "name stuffing does buy lexical coverage"
    assert result["connectivity"] < 0.20
    assert result["score"] < 0.30, result


def test_comment_stuffing_earns_nothing(sigs):
    """rdfs:comment is deliberately excluded from the term index, so echoing the
    scenario prose into comments cannot buy score."""
    plain = be.score_ontology(CLASSES_ONLY_ODP, sigs)["score"]
    stuffed = CLASSES_ONLY_ODP + """
@prefix rdfs2: <http://www.w3.org/2000/01/rdf-schema#> .
:Event rdfs2:comment "causes outcome effect weight mediator causal relation" .
"""
    assert be.score_ontology(stuffed, sigs)["score"] == pytest.approx(plain)


# ── an ontology with NO rdfs:domain / rdfs:range must not crash ──────────────

def test_ontology_without_any_domain_or_range_is_handled(sigs):
    g = Graph()
    g.parse(data=NO_DOMAIN_RANGE_ODP, format="turtle")
    from rdflib.namespace import OWL, RDF, RDFS
    assert not list(g.triples((None, RDFS.domain, None)))
    assert not list(g.triples((None, RDFS.range, None)))
    assert not list(g.subjects(RDF.type, OWL.Restriction))

    result = be.score_ontology(NO_DOMAIN_RANGE_ODP, sigs)

    assert result["status"] == "ok"
    assert 0.0 <= result["score"] <= 1.0
    # the vocabulary is there ...
    assert result["coverage"] > 0.8
    # ... but every property is floating, so it earns only the floor tier
    assert result["connectivity"] <= 0.20
    assert set(result["tier_histogram"]) <= {"floating", "none"}
    assert result["score"] < be.score_ontology(CORRECT_ODP, sigs)["score"]


@pytest.mark.parametrize("path", sorted(
    p for p in (GROUND_TRUTH.glob("*.ttl") if GROUND_TRUTH.exists() else [])))
def test_real_reference_patterns_never_crash_the_engine(path, sigs):
    """Robustness sweep over every reference pattern, including the three that
    are A-Box-only instance data and the one with no domain/range at all."""
    result = be.score_ontology(path.read_text(encoding="utf-8", errors="replace"),
                               sigs)
    assert result["status"] in {"ok", "empty", "unparseable", "no_ontology"}
    assert 0.0 <= result["score"] <= 1.0


def test_real_llama2_style_output_from_the_corpus_is_handled():
    """The corpus's dominant degenerate style, scored end to end."""
    candidate = (REPO_ROOT / "outputs" / "meta-llama_Llama-2-70b-chat-hf" /
                 "scenario-cq" / "2025-151-01" / "ontology.ttl")
    if not candidate.exists():
        pytest.skip("generation corpus not available")
    result = be.run_cq_verification(
        candidate.read_text(encoding="utf-8", errors="replace"), "2025-151-01")
    assert result["status"] == "ok"
    assert result["cqs_total"] == 5
    assert result["sev_coverage"] > 0.5, "it does name the right things"
    assert result["sev_connectivity"] <= 0.20, "but declares no structure"
    assert result["sev_score"] < 0.40


# ── structural regressions the design must keep satisfying ──────────────────

SELF_LOOP_ODP = """
@prefix owl:  <http://www.w3.org/2002/07/owl#> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
@prefix :     <http://example.org/causal#> .

:Event a owl:Class .
:causes a owl:ObjectProperty ; rdfs:domain :Event ; rdfs:range :Event .
"""


def test_self_loop_event_causes_event_earns_full_direct_credit():
    """Variables, not nodes: a property whose domain and range are the SAME class
    is a correct direct answer to "what events are responsible for X".  A path
    search with a visited-node set would silently score it zero."""
    sig = [s for s in be.signatures_for(SCENARIO)
           if s["cq"].startswith("What events are responsible")]
    result = be.score_ontology(SELF_LOOP_ODP, sig)
    assert result["per_cq"][0]["relations"][0]["tier"] == "direct"
    assert result["per_cq"][0]["connectivity"] == pytest.approx(1.0)


def test_restriction_only_reference_pattern_is_lifted():
    """data/ground_truth/2023-135-01.owl declares 5 classes, 5 object properties,
    ZERO rdfs:domain, ZERO rdfs:range and 10 owl:Restriction.  All of its
    structure comes from restrictions; if lift() ever stops reading them, a
    published ODP silently drops to zero."""
    gt = GROUND_TRUTH / "2023-135-01.owl"
    if not gt.exists():
        pytest.skip("reference pattern not available")
    g = Graph()
    g.parse(data=gt.read_text(encoding="utf-8"), format="turtle")
    from rdflib.namespace import RDFS
    assert not list(g.triples((None, RDFS.domain, None)))
    assert not list(g.triples((None, RDFS.range, None)))

    result = be.score_ontology(gt.read_text(encoding="utf-8"),
                               [s for s in be.signatures_for("2023-135-01")
                                if s.get("cq_type") != "out_of_scope"])
    assert result["connectivity"] > 0.75, result
    assert result["score"] >= 0.85, result


def test_reified_n_ary_relation_earns_full_credit_like_the_gold(sigs):
    """The gold answers "what events are responsible for X" through a :Causes hub
    with hasTreatment / hasOutcome, not through an Event->Event property.  That
    idiom must pay full credit or the metric ranks the published pattern below a
    model that emits surface nouns."""
    result = be.score_ontology(CORRECT_ODP, sigs)
    tiers = {r["tier"] for p in result["per_cq"] for r in p["relations"]}
    assert "reified" in tiers
    assert result["tier_histogram"].get("none", 0) == 0


def test_domain_only_property_earns_the_half_anchored_tier():
    """The published reference 2025-151-01 declares rdfs:domain on all of its
    properties and rdfs:range on none.  A metric that demanded both would score a
    correct, published ODP at zero, so `external` fillers must be full credit and
    a genuinely open range must still be worth the half tier."""
    odp = """
@prefix owl:  <http://www.w3.org/2002/07/owl#> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
@prefix :     <http://example.org/causal#> .
:Causes a owl:Class .
:Event a owl:Class .
:hasMediator a owl:ObjectProperty ; rdfs:domain :Causes .
"""
    sig = [s for s in be.signatures_for(SCENARIO) if s["cq"].startswith("What if")]
    result = be.score_ontology(odp, sig)
    assert result["per_cq"][0]["relations"][0]["tier"] == "half"
    assert result["per_cq"][0]["connectivity"] == pytest.approx(0.75)


# ── result shape, determinism, and "no LLM anywhere" ────────────────────────

def test_result_keeps_the_backward_compatible_headline_fields():
    result = be.run_cq_verification(CORRECT_ODP, SCENARIO)
    for key in ("cqs_total", "cqs_passed", "cqs_failed", "pass_rate", "results"):
        assert key in result
    assert result["cqs_total"] == result["cqs_passed"] + result["cqs_failed"]
    assert result["pass_rate"] == pytest.approx(
        result["cqs_passed"] / result["cqs_total"])
    # and adds the graded score alongside
    for key in ("sev_score", "sev_coverage", "sev_connectivity",
                "signatures_sha", "tier_histogram"):
        assert key in result
    assert result["method"] == "SEV"


def test_scoring_is_deterministic():
    first = be.run_cq_verification(CORRECT_ODP, SCENARIO)
    second = be.run_cq_verification(CORRECT_ODP, SCENARIO)
    assert json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True)


def test_signatures_sha_is_stable_and_reported():
    result = be.run_cq_verification(CORRECT_ODP, SCENARIO)
    assert result["signatures_sha"] == be.signatures_sha()
    assert len(result["signatures_sha"]) == 16


def test_cq_verification_makes_no_network_or_llm_calls(monkeypatch):
    """There is no LLM to mock: the code path must not exist at all.  Break
    `requests` and `openai` outright and confirm scoring is unaffected."""
    def explode(*args, **kwargs):                       # pragma: no cover
        raise AssertionError("CQ verification must not touch the network")

    monkeypatch.setattr(be.requests, "get", explode)
    monkeypatch.setattr(be.requests, "post", explode)
    monkeypatch.setitem(sys.modules, "openai", None)

    result = be.run_cq_verification(CORRECT_ODP, SCENARIO)
    assert result["sev_score"] > 0.0

    source = (REPO_ROOT / "batch_evaluate.py").read_text(encoding="utf-8")
    assert "import openai" not in source
    assert "chat.completions.create" not in source


def test_external_signature_artifact_overrides_the_builtins(tmp_path):
    """data/scenarios/cq_signatures.json is the frozen, committed artifact; when
    it exists it must win over the in-module defaults."""
    artifact = tmp_path / "cq_signatures.json"
    artifact.write_text(json.dumps({
        SCENARIO: [{"cq": "What events are responsible for the pavement being slippery?",
                    "cq_type": "structural",
                    "slots": {"EVENT": {"kind": "class", "surface": ["event"]}},
                    "relations": []}]
    }), encoding="utf-8")
    loaded = be.signatures_for(SCENARIO, None, None, artifact)
    assert loaded[0]["signature_source"] == "authored"
    assert loaded[0]["relations"] == []
    # the remaining three CQs have no record in the artifact and fall back
    assert [s["signature_source"] for s in loaded[1:]] == ["derived"] * 3


def test_unsigned_cqs_are_flagged_not_silently_counted_as_authored():
    """Scenarios whose 175-CQ signatures are not authored yet must be visibly
    marked, so the paper can report the calibrated subset separately."""
    result = be.run_cq_verification(CORRECT_ODP, "2025-149-01")
    assert result["cqs_total"] == len(be.load_cq_list("2025-149-01"))
    assert result["cqs_authored"] == 0
    assert result["cqs_derived"] == result["cqs_total"]
    assert result["sev_score_authored"] is None
    assert all(r["signature_source"] == "derived" for r in result["results"])


def test_non_structural_cqs_are_classified_so_they_can_be_excluded():
    assert be.classify_cq("Can the ontology represent a causal relation?") == "meta"
    assert be.classify_cq(
        "How many container ships are registered under the German flag?"
    ) == "data_dependent"
    assert be.classify_cq(
        "What events are responsible for the pavement being slippery?") == "structural"


def test_signature_records_are_not_mutated_across_calls(sigs):
    """The signature catalogue is frozen: scoring must not edit it in place."""
    before = copy.deepcopy(be._BUILTIN_CQ_SIGNATURES[SCENARIO])
    be.score_ontology(CORRECT_ODP, be.signatures_for(SCENARIO))
    be.run_cq_verification(NO_DOMAIN_RANGE_ODP, SCENARIO)
    assert be._BUILTIN_CQ_SIGNATURES[SCENARIO] == before
