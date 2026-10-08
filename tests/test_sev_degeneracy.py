"""Degeneracy tests for the SEV (Schema-Entailment Verification) CQ metric.

Why this file exists
--------------------
The structural metric was replaced because it had a degenerate optimum (F3: the
empty file took full marks).  SEV was built to remove that class of flaw — and
reproduced it.  Two live defects:

V3  A mechanically-built *cross-product* ontology — take the signature surface
    forms for a scenario, declare one owl:Class per surface term and one
    owl:ObjectProperty per relation surface with rdfs:domain and rdfs:range
    spanning every class — scored 1.0, STRICTLY ABOVE the published gold
    reference pattern at 0.9953.  The same held for a *mega-hub* ontology, where
    one ``:Hub`` class subsumes everything and every property is declared
    ``rdfs:domain :Hub ; rdfs:range :Hub``.

V4  ``derive_signature()`` always emits exactly one relation with
    ``to: "external"``, and ``relation_credit()``'s external branch returned the
    ``direct`` tier (credit 1.00) for the bare ``ASK { ?a ?p ?b }``.  Declaring a
    single ``rdfs:domain`` therefore bought FULL connectivity credit.  Derived
    signatures cover ~79% of the corpus, so that inflation dominated the
    headline number.

The two constraints these tests pin down together are the whole problem:

    (1) a name-stuffed cross-product / mega-hub ontology must score LOW, and
    (2) the published gold reference patterns must still score >= 0.85,

and every adversarial fixture must sit strictly below every gold pattern.

Everything here is offline: rdflib only, no LLM, no network, no API key.
"""

import json
import importlib.util
import re
import sys
from pathlib import Path

import pytest

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

GROUND_TRUTH = REPO_ROOT / "data" / "ground_truth"
SIGNATURES_JSON = REPO_ROOT / "data" / "scenarios" / "cq_signatures.json"

# Scenarios whose published reference pattern is a real, rdflib-parseable T-Box
# and therefore can be gold-calibrated.  (The ODRL A-Box examples and the two
# OWL/XML functional-syntax files are exempt; see be._GOLD_EXEMPT.)
CALIBRATABLE = ["2023-133-01", "2023-133-02", "2023-135-01",
                "2025-151-01", "2025-151-02", "2025-153-01"]

# Keeping the adversarial fixtures to two scenarios keeps the suite fast; the
# construction is scenario-generic and is applied to both a small signature set
# (2023-133-01, 4 CQs) and a multi-relation one (2023-135-01, chained slots).
ADVERSARIAL_SCENARIOS = ["2023-133-01", "2023-135-01"]

_PREFIXES = [
    "@prefix : <http://example.org/adversary#> .",
    "@prefix owl: <http://www.w3.org/2002/07/owl#> .",
    "@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .",
    "@prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .",
]

# Cap only so the fixtures stay quick to score; 14 x 14 is still a full
# cross-product and still stuffs every high-value surface form.
_MAX_TERMS = 14


def _class_name(term):
    return "".join(w.capitalize() for w in re.findall(r"[A-Za-z0-9]+", term)) or "Thing"


def _prop_name(term):
    parts = re.findall(r"[A-Za-z0-9]+", term)
    if not parts:
        return "prop"
    return parts[0].lower() + "".join(w.capitalize() for w in parts[1:])


def calibrated_signatures(scenario_id):
    """The authored signatures the gold gate actually scores."""
    return [s for s in be.signatures_for(scenario_id)
            if s.get("signature_source") == "authored"
            and s.get("cq_type") != "out_of_scope"]


def signature_surfaces(scenario_id):
    """Every surface form the scenario's signatures look for.

    This is the attacker's whole budget: the metric's own answer key, read out
    of the committed signature artifact.  Nothing else is used.
    """
    classes, props = [], []
    for sig in calibrated_signatures(scenario_id):
        for spec in (sig.get("slots") or {}).values():
            classes += list(spec.get("surface") or [])
        for rel in (sig.get("relations") or []):
            props += list(rel.get("surface") or [])
    classes = list(dict.fromkeys(_class_name(c) for c in classes))[:_MAX_TERMS]
    props = list(dict.fromkeys(_prop_name(p) for p in props))[:_MAX_TERMS]
    return classes, props


# ── the three adversarial fixtures ───────────────────────────────────────────

def cross_product_odp(scenario_id):
    """V3: name stuffing + a full domain/range cross product.

    Declares one class per surface term and one object property per relation
    surface, then declares EVERY class as both rdfs:domain and rdfs:range of
    EVERY property.  It commits to nothing: it asserts that everything relates
    to everything.  Under RDFS semantics multiple rdfs:domain axioms are a
    CONJUNCTION, so this ontology does not even say what the exploit relies on
    it saying.
    """
    classes, props = signature_surfaces(scenario_id)
    lines = list(_PREFIXES)
    for c in classes:
        lines.append(":%s a owl:Class ." % c)
    for p in props:
        lines.append(":%s a owl:ObjectProperty ." % p)
        for c in classes:
            lines.append(":%s rdfs:domain :%s ." % (p, c))
            lines.append(":%s rdfs:range :%s ." % (p, c))
    return "\n".join(lines)


def mega_hub_odp(scenario_id):
    """V3 variant: one universal superclass, every property hung off it.

    Every stuffed class is rdfs:subClassOf :Hub and every property is declared
    ``rdfs:domain :Hub ; rdfs:range :Hub``.  Each property now has exactly ONE
    declared domain and ONE declared range, so a pure fan-out counter does not
    see it; the connectivity comes entirely from climbing the subsumption cone
    into a class that subsumes the whole ontology.  This is the locally-declared
    version of the owl:Thing exploit the engine already refuses.
    """
    classes, props = signature_surfaces(scenario_id)
    lines = list(_PREFIXES)
    lines.append(":Hub a owl:Class .")
    for c in classes:
        lines.append(":%s a owl:Class ; rdfs:subClassOf :Hub ." % c)
    for p in props:
        lines.append(":%s a owl:ObjectProperty ; rdfs:domain :Hub ; "
                     "rdfs:range :Hub ." % p)
    return "\n".join(lines)


def floating_property_odp(scenario_id):
    """The Llama-2 style: perfect vocabulary, zero domain, zero range.

    The control fixture: it must stay at the floor, and it must not score above
    the two fixtures that add fake structure on top of the same vocabulary.
    """
    classes, props = signature_surfaces(scenario_id)
    lines = list(_PREFIXES)
    for c in classes:
        lines.append(":%s a owl:Class ." % c)
    for p in props:
        lines.append(":%s a owl:ObjectProperty ." % p)
    return "\n".join(lines)


ADVERSARIES = {
    "cross_product": cross_product_odp,
    "mega_hub": mega_hub_odp,
    "floating": floating_property_odp,
}

# A name-stuffed, structure-faking ontology may keep its lexical coverage — that
# is honest, the words really are there — but it must land in the "lexical-only"
# band, not anywhere near a pattern that actually models the domain.
ADVERSARIAL_CEILING = 0.30


def score_of(text, scenario_id):
    return be.score_ontology(text, calibrated_signatures(scenario_id))["score"]


def gold_score(scenario_id):
    gt = be.find_ground_truth(scenario_id)
    assert gt is not None, f"no reference pattern for {scenario_id}"
    return be.score_ontology(gt.read_text(encoding="utf-8", errors="replace"),
                             calibrated_signatures(scenario_id))["score"]


# ── V3: the degenerate optimum ───────────────────────────────────────────────

@pytest.mark.parametrize("scenario_id", ADVERSARIAL_SCENARIOS)
def test_cross_product_name_stuffer_does_not_beat_the_gold(scenario_id):
    """V3, exactly as reported: the cross product scored 1.0 vs gold 0.9953."""
    adv = score_of(cross_product_odp(scenario_id), scenario_id)
    gold = gold_score(scenario_id)
    assert adv < gold, (
        f"{scenario_id}: a mechanically-built cross-product ontology scores "
        f"{adv} against a published reference pattern's {gold}")
    assert adv < ADVERSARIAL_CEILING, (
        f"{scenario_id}: cross-product ontology scores {adv}, which is above the "
        f"lexical-only band ({ADVERSARIAL_CEILING})")


@pytest.mark.parametrize("scenario_id", ADVERSARIAL_SCENARIOS)
def test_mega_hub_does_not_buy_connectivity(scenario_id):
    """A single subsuming class must not connect everything to everything."""
    result = be.score_ontology(mega_hub_odp(scenario_id),
                               calibrated_signatures(scenario_id))
    assert result["coverage"] > 0.5, "the fixture really does stuff the names"
    assert result["connectivity"] < 0.30, (
        f"{scenario_id}: a mega-hub bought connectivity {result['connectivity']}")
    assert result["score"] < ADVERSARIAL_CEILING, result["score"]
    assert result["score"] < gold_score(scenario_id)


@pytest.mark.parametrize("scenario_id", ADVERSARIAL_SCENARIOS)
def test_floating_properties_stay_at_the_floor(scenario_id):
    result = be.score_ontology(floating_property_odp(scenario_id),
                               calibrated_signatures(scenario_id))
    assert result["connectivity"] <= 0.20
    assert result["score"] < ADVERSARIAL_CEILING
    assert result["score"] < gold_score(scenario_id)


def test_adding_fake_structure_never_pays_more_than_declaring_nothing():
    """The load-bearing invariant.

    All three fixtures share the SAME stuffed vocabulary.  The only difference is
    that two of them bolt on domain/range axioms that commit to nothing.  If
    faking structure scores higher than admitting there is none, the metric is
    rewarding the fake — which is precisely how the cross product reached 1.0.
    """
    for scenario_id in ADVERSARIAL_SCENARIOS:
        floor = score_of(floating_property_odp(scenario_id), scenario_id)
        for name in ("cross_product", "mega_hub"):
            faked = score_of(ADVERSARIES[name](scenario_id), scenario_id)
            assert faked <= floor + 0.10, (
                f"{scenario_id}: {name} scores {faked} vs {floor} for the same "
                f"vocabulary with no structure at all — faking structure pays")


# ── the two hub SHAPES, on the scenario the rest of the suite uses ───────────
#
# `tests/test_cq_evaluation.py::test_name_stuffing_and_mega_hub_do_not_buy_
# connectivity` is named for two constructions but scores only one of them:
# NAME_STUFFING_ODP, which hangs every property off `owl:Thing`.  No mega-hub
# fixture appears in it, so the second half of its name is uncovered there and a
# reader auditing coverage by test name is misled.  That file belongs to another
# brief; the gap it leaves is closed here, on the SAME scenario and the SAME
# signature set, with BOTH shapes side by side and named for what they are.
#
# The two shapes are different exploits against the same defence:
#   * owl:Thing hub      -- the built-in universal class as an endpoint;
#                           refused by _endpoint(), which treats owl:Thing as an
#                           OPEN endpoint rather than a node.
#   * locally-declared   -- `:Hub`, subsuming every class in the file; refused by
#     mega-hub              _find_universal_hubs()/SEV_HUB_FRACTION, which is a
#                           separate code path with its own tunable constants.
# Testing only the first leaves the second undefended.

HUB_SCENARIO = "2023-133-01"          # the causal pattern; 4 authoritative CQs

# The same stuffed causal vocabulary in all three fixtures below, so the ONLY
# difference between them is which hub they hang the properties off.
_HUB_CLASSES = ["Event", "Occurrence", "Situation", "State", "Node", "Outcome",
                "Effect", "Consequence", "Result", "Causes", "CausalRelation",
                "CausalLink", "Mediator", "EffectWeight", "Strength"]
_HUB_PROPS = ["causes", "hasOutcome", "hasMediator", "hasEffectWeight"]

_HUB_PREFIXES = [
    "@prefix owl:  <http://www.w3.org/2002/07/owl#> .",
    "@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .",
    "@prefix :     <http://example.org/causal#> .",
]


def owl_thing_hub_odp():
    """Every property anchored at both ends -- to owl:Thing."""
    lines = list(_HUB_PREFIXES)
    lines += [":%s a owl:Class ." % c for c in _HUB_CLASSES]
    lines += [":%s a owl:ObjectProperty ; rdfs:domain owl:Thing ; "
              "rdfs:range owl:Thing ." % p for p in _HUB_PROPS]
    return "\n".join(lines) + "\n"


def declared_mega_hub_odp():
    """The fixture the sibling test's name promises: a LOCAL universal class.

    `:Hub` is an ordinary owl:Class, so nothing in the vocabulary marks it as
    universal; it is universal only because every other class in the file is
    rdfs:subClassOf it.  Each property has exactly one declared domain and one
    declared range, so a fan-out counter sees nothing wrong.
    """
    lines = list(_HUB_PREFIXES) + [":Hub a owl:Class ."]
    lines += [":%s a owl:Class ; rdfs:subClassOf :Hub ." % c for c in _HUB_CLASSES]
    lines += [":%s a owl:ObjectProperty ; rdfs:domain :Hub ; rdfs:range :Hub ."
              % p for p in _HUB_PROPS]
    return "\n".join(lines) + "\n"


def no_hub_odp():
    """The control: identical vocabulary, no domain and no range at all."""
    lines = list(_HUB_PREFIXES)
    lines += [":%s a owl:Class ." % c for c in _HUB_CLASSES]
    lines += [":%s a owl:ObjectProperty ." % p for p in _HUB_PROPS]
    return "\n".join(lines) + "\n"


HUB_SHAPES = {"owl_thing_hub": owl_thing_hub_odp,
              "declared_mega_hub": declared_mega_hub_odp}


def test_the_declared_mega_hub_fixture_really_is_a_universal_hub():
    """Precondition.  If `:Hub` stopped subsuming the ontology the test below
    would pass for the wrong reason -- there would be no hub to refuse."""
    g, status = be._parse_for_sev(declared_mega_hub_odp())
    assert status == "ok", status
    sv = be.SchemaView(g)
    hub = [c for c in sv.universal_hubs if str(c).endswith("#Hub")]
    assert hub, (
        f"the fixture's :Hub is not detected as a universal hub "
        f"({sv.universal_hubs}); the mega-hub defence is not being exercised")


@pytest.mark.parametrize("shape", sorted(HUB_SHAPES))
def test_neither_hub_shape_buys_connectivity_on_the_causal_scenario(shape):
    """Name stuffing may keep its lexical coverage; a hub may not buy structure."""
    sigs = be.signatures_for(HUB_SCENARIO)
    result = be.score_ontology(HUB_SHAPES[shape](), sigs)
    assert result["coverage"] > 0.8, "the fixture really does stuff the names"
    assert result["connectivity"] < 0.20, (
        f"{shape} bought connectivity {result['connectivity']}")
    assert result["score"] < 0.30, result
    assert set(result["tier_histogram"]) <= {"floating", "none"}, (
        f"{shape} was credited a connected tier: {result['tier_histogram']}")


@pytest.mark.parametrize("shape", sorted(HUB_SHAPES))
def test_neither_hub_shape_beats_the_causal_gold_pattern(shape):
    sigs = be.signatures_for(HUB_SCENARIO)
    adv = be.score_ontology(HUB_SHAPES[shape](), sigs)["score"]
    gt = be.find_ground_truth(HUB_SCENARIO)
    gold = be.score_ontology(gt.read_text(encoding="utf-8", errors="replace"),
                             sigs)["score"]
    assert adv < gold, f"{shape} scores {adv} against the gold pattern's {gold}"


@pytest.mark.parametrize("shape", sorted(HUB_SHAPES))
def test_hanging_properties_off_a_hub_pays_no_more_than_declaring_nothing(shape):
    """The load-bearing invariant, restated for the hub shapes specifically."""
    sigs = be.signatures_for(HUB_SCENARIO)
    floor = be.score_ontology(no_hub_odp(), sigs)["score"]
    hubbed = be.score_ontology(HUB_SHAPES[shape](), sigs)["score"]
    assert hubbed <= floor, (
        f"{shape} scores {hubbed} against {floor} for the same vocabulary with "
        "no domain/range at all -- a hub is being paid for as structure")


# ── the gold-calibration regression (the other half of the problem) ──────────

@pytest.mark.parametrize("scenario_id", CALIBRATABLE)
def test_gold_reference_pattern_still_clears_the_calibration_threshold(scenario_id):
    """Killing the exploit must not kill the published patterns."""
    gold = gold_score(scenario_id)
    assert gold >= be.GOLD_CALIBRATION_THRESHOLD, (
        f"{scenario_id}: the published reference pattern fell to {gold}, below "
        f"the {be.GOLD_CALIBRATION_THRESHOLD} gold gate")


def test_whole_signature_catalogue_still_passes_the_gold_gate():
    report = be.run_gold_calibration()
    assert report["failed_unexpected"] == 0, json.dumps(
        report["gold_unsupported"], indent=2)
    assert report["passed"] >= 35


@pytest.mark.parametrize("scenario_id", ADVERSARIAL_SCENARIOS)
def test_every_adversary_is_strictly_below_every_gold(scenario_id):
    """The two constraints, stated together in one place."""
    golds = {sid: gold_score(sid) for sid in CALIBRATABLE}
    worst_gold = min(golds.values())
    for name, build in ADVERSARIES.items():
        adv = score_of(build(scenario_id), scenario_id)
        assert adv < worst_gold, (
            f"{scenario_id}/{name} scores {adv}; the weakest published reference "
            f"pattern scores {worst_gold} ({golds})")


# ── V4: derived-signature inflation ──────────────────────────────────────────

DERIVED_CQ = "What is the trajectory of a river?"

SINGLE_DOMAIN_ODP = """
@prefix : <http://example.org/adversary#> .
@prefix owl: <http://www.w3.org/2002/07/owl#> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .

:River a owl:Class .
:trajectory a owl:ObjectProperty ; rdfs:domain :River .
"""


def _score_derived(cq, odp):
    sig = be.derive_signature(cq)
    sig["signature_source"] = "derived"
    g, status = be._parse_for_sev(odp)
    assert status == "ok", status
    sv = be.SchemaView(g)
    return be.score_cq(sv, sv.lift(), sv.index(), sig)


def test_derive_signature_still_emits_the_external_relation():
    """Pin the shape the defect lives in, so the test cannot go stale."""
    sig = be.derive_signature(DERIVED_CQ)
    assert sig["derived"] is True
    assert [r["to"] for r in sig["relations"]] == ["external"]


DOMAIN_AND_RANGE_ODP = """
@prefix : <http://example.org/adversary#> .
@prefix owl: <http://www.w3.org/2002/07/owl#> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .

:River a owl:Class .
:Path a owl:Class .
:trajectory a owl:ObjectProperty ; rdfs:domain :River ; rdfs:range :Path .
"""


def test_a_single_rdfs_domain_does_not_buy_full_connectivity_credit():
    """V4.  One class + one property + one rdfs:domain scored a perfect 1.0."""
    result = _score_derived(DERIVED_CQ, SINGLE_DOMAIN_ODP)
    tier = result["relations"][0]["tier"]
    assert tier != "direct", (
        "an uncertified `external` relation from a derived signature is being "
        "priced at the strongest tier in the ladder")
    assert result["relations"][0]["credit"] <= be.SEV_TIERS["half"], result
    # An uncalibrated fallback signature may never call a CQ answerable.
    assert result["connectivity"] < be.SEV_ANSWERABLE_L2, result
    assert result["answerable"] is False, result
    assert result["score"] < be.GOLD_CALIBRATION_THRESHOLD, result
    assert result["label"] != "supported", result


def test_the_derived_fallback_still_discriminates_between_real_ontologies():
    """Lowering the ceiling must not flatten 79% of the corpus onto one number.

    An ODP that anchors the property at BOTH ends has to out-score the two-axiom
    stub, which in turn has to out-score declaring the property and nothing else.
    """
    stub = _score_derived(DERIVED_CQ, SINGLE_DOMAIN_ODP)
    anchored = _score_derived(DERIVED_CQ, DOMAIN_AND_RANGE_ODP)
    floating = _score_derived(DERIVED_CQ, """
@prefix : <http://example.org/adversary#> .
@prefix owl: <http://www.w3.org/2002/07/owl#> .
:River a owl:Class .
:trajectory a owl:ObjectProperty .
""")
    assert floating["score"] < stub["score"] < anchored["score"], (
        floating["score"], stub["score"], anchored["score"])
    # ... and every one of them below a curator-certified external relation.
    assert anchored["relations"][0]["credit"] < be.SEV_TIERS["external"]


def test_the_external_tier_is_strictly_weaker_than_a_real_anchored_connection():
    """`external` verifies ONE of the two endpoint constraints; `direct` verifies
    both.  It must therefore be priced strictly below `direct`."""
    assert "external" in be.SEV_TIERS
    assert be.SEV_TIERS["external"] < be.SEV_TIERS["direct"]
    # ... and an uncertified (derived) external claim weaker still.
    assert be.SEV_TIERS["external_uncertified"] < be.SEV_TIERS["external"]
    # ... but not so weak that a range-free published reference fails the gate:
    # 0.25 * 1.0 + 0.75 * SEV_TIERS["external"] must stay >= 0.85.
    assert 0.25 + 0.75 * be.SEV_TIERS["external"] >= be.GOLD_CALIBRATION_THRESHOLD


def test_authored_external_relations_are_still_credited_above_uncertified_ones():
    """2025-151-01 declares rdfs:domain on all five properties and rdfs:range on
    none; a curator certified that its fillers come from reused ODRL vocabulary.
    That certification is what separates it from the deriver's guess."""
    gt = be.find_ground_truth("2025-151-01")
    result = be.score_ontology(gt.read_text(encoding="utf-8"),
                               calibrated_signatures("2025-151-01"))
    tiers = {r["tier"] for p in result["per_cq"] for r in p["relations"]}
    assert tiers == {"external"}, tiers
    assert result["score"] >= be.GOLD_CALIBRATION_THRESHOLD


# ── (c) the signature catalogue must be a frozen data file ───────────────────

def test_signature_catalogue_lives_in_a_committed_data_file():
    """signatures_sha() must hash a file on disk, not a dict embedded in code."""
    assert SIGNATURES_JSON.exists(), (
        "data/scenarios/cq_signatures.json is missing: signatures_sha() would be "
        "certifying a mutable code literal")
    frozen = json.loads(SIGNATURES_JSON.read_text(encoding="utf-8"))
    assert isinstance(frozen, dict) and frozen
    assert sorted(frozen) == sorted(be.load_cq_signatures())
    for sid, records in frozen.items():
        assert records, sid
        for rec in records:
            assert "cq" in rec and "slots" in rec and "relations" in rec


def test_batch_evaluate_no_longer_embeds_the_signature_catalogue():
    source = (REPO_ROOT / "batch_evaluate.py").read_text(encoding="utf-8")
    assert "_BUILTIN_CQ_SIGNATURES" in source, "the name is still part of the API"
    # ... but the surface-form catalogue itself must not be a code literal.
    for needle in ("What events are responsible for the pavement being slippery?",
                   "Under what pseudonym did the author"):
        assert needle not in source, (
            f"the signature catalogue is still embedded in batch_evaluate.py "
            f"({needle!r}); signatures_sha() would hash mutable code")


def test_signatures_sha_tracks_the_frozen_file():
    payload = json.dumps(be.load_cq_signatures(), sort_keys=True,
                         ensure_ascii=False, default=str)
    import hashlib
    assert be.signatures_sha() == hashlib.sha256(
        payload.encode("utf-8")).hexdigest()[:16]
    frozen = json.loads(SIGNATURES_JSON.read_text(encoding="utf-8"))
    assert be.load_cq_signatures() == frozen
