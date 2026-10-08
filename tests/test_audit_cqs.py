"""Tests for scripts/audit_cqs.py -- the offline CQ provenance auditor.

Three contracts are asserted here, in increasing order of how much they matter.

1.  SEPARATION.  `test_known_bad_cq_is_flagged` /
    `test_known_good_cq_from_same_scenario_is_not_flagged`: the auditor must flag
    the river/national-parks CQ that sits under the RoleDependentNames (author
    pseudonym) pattern in the real corpus, and must NOT flag its legitimate
    siblings.  A detector that cannot separate those two cases is worthless, so
    both directions are asserted.

2.  MONOTONICITY.  `test_injecting_more_foreign_cqs_never_reduces_detection` and
    `test_the_reported_regression_second_intruder_does_not_hide_the_first`: more
    contamination must never produce less detection.  Before the rule was rebuilt,
    adding ONE more river CQ to 2023-135-01 made the confirmed-bad CQ #3 stop being
    flagged and flipped the CI exit code from 1 to 0, because each CQ was partly
    anchored against its SIBLINGS.  That failure direction is the dangerous one: it
    makes a low misfiled count evidence of heavy contamination rather than of a
    clean corpus.  `test_verdicts_do_not_depend_on_sibling_cqs` asserts the
    structural property that guarantees monotonicity.

3.  RECALL.  `test_cross_pattern_intruder_recall` injects every real CQ of every
    scenario, one at a time, into every other scenario (2067 lone-intruder
    injections) and prints recall, precision and the native false-positive rate in
    full.  The printed numbers are the honest ones; the assertions are floors, not
    the measurement.
"""

import copy
import importlib.util
import json
import os
import subprocess
import sys

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPT = os.path.join(REPO, "scripts", "audit_cqs.py")
SCENARIOS = os.path.join(REPO, "data", "scenarios", "pattern_scenarios.json")


def _load_module():
    spec = importlib.util.spec_from_file_location("audit_cqs", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


audit_cqs = _load_module()

FLAGGED = {"suspicious", "almost_certainly_misfiled", "malformed"}


@pytest.fixture(scope="module")
def real_report():
    scenarios = audit_cqs.load_scenarios(SCENARIOS)
    return audit_cqs.audit_corpus(
        scenarios, ground_truth_dir=os.path.join(REPO, "data", "ground_truth")
    )


def _cq(report, sid, index):
    for sc in report["scenarios"]:
        if sc["scenario_id"] == sid:
            for cq in sc["cqs"]:
                if cq["index"] == index:
                    return cq
    raise AssertionError("no such CQ: %s #%s" % (sid, index))


# --------------------------------------------------------------------------
# The known case: 2023-135-01 is RoleDependentNames (author pseudonyms).
#   index 0 -> "Under what pseudonym did the author C. S. Lewis published ..."  GOOD
#   index 2 -> "Show the trajectories of rivers which cross national parks?"    BAD
# --------------------------------------------------------------------------


def test_known_bad_cq_is_flagged(real_report):
    cq = _cq(real_report, "2023-135-01", 2)
    assert "river" in cq["text"].lower(), "corpus changed; test targets the wrong CQ"
    assert cq["verdict"] == "almost_certainly_misfiled", (
        "the river CQ under the author-pseudonym pattern must be flagged as "
        "almost_certainly_misfiled, got %r (suspicion=%s, flags=%s)"
        % (cq["verdict"], cq.get("suspicion"), [f["code"] for f in cq.get("flags", [])])
    )


def test_known_good_cq_from_same_scenario_is_not_flagged(real_report):
    cq = _cq(real_report, "2023-135-01", 0)
    assert "pseudonym" in cq["text"].lower()
    assert cq["verdict"] == "clean", (
        "the pseudonym CQ genuinely belongs to RoleDependentNames and must stay "
        "clean, got %r (flags=%s)" % (cq["verdict"], [f["code"] for f in cq.get("flags", [])])
    )


def test_second_good_cq_from_same_scenario_is_not_flagged(real_report):
    cq = _cq(real_report, "2023-135-01", 1)
    assert cq["verdict"] not in ("almost_certainly_misfiled", "malformed")


def test_detector_separates_the_two_directions(real_report):
    """A detector that flags everything, or nothing, fails this."""
    sc = [s for s in real_report["scenarios"] if s["scenario_id"] == "2023-135-01"][0]
    verdicts = [c["verdict"] for c in sc["cqs"]]
    assert verdicts.count("almost_certainly_misfiled") == 1, verdicts
    assert verdicts.count("clean") >= 1, verdicts


# --------------------------------------------------------------------------
# Synthetic controls -- the detector must generalise, not memorise one string.
# --------------------------------------------------------------------------

SYNTH = [
    {
        "scenario_id": "SYN-01",
        "scenario_text": "A pizza restaurant serves pizzas with different toppings; "
        "customers order pizzas and the kitchen prepares each topping.",
        "ontology": "http://example.org/PizzaTopping.ttl",
        "cq_list": [
            "Which toppings are on a given pizza?",
            "Which customer ordered which pizza?",
            "How long does the kitchen take to prepare a pizza?",
            "Which spacecraft launched from which orbital platform?",
        ],
    },
    {
        "scenario_id": "SYN-02",
        "scenario_text": "Spacecraft launches are recorded with their orbital platform "
        "and launch window.",
        "ontology": "http://example.org/Launch.ttl",
        "cq_list": [
            "Which spacecraft launched from which orbital platform?",
            "What is the launch window of a spacecraft?",
            "Which orbital platform hosted the most launches?",
        ],
    },
]


def test_synthetic_intruder_is_flagged():
    rep = audit_cqs.audit_corpus(SYNTH, ground_truth_dir=None)
    intruder = _cq(rep, "SYN-01", 3)
    assert intruder["verdict"] == "almost_certainly_misfiled", intruder


def test_synthetic_natives_are_clean():
    rep = audit_cqs.audit_corpus(SYNTH, ground_truth_dir=None)
    for i in (0, 1, 2):
        assert _cq(rep, "SYN-01", i)["verdict"] == "clean", _cq(rep, "SYN-01", i)


def test_cross_scenario_duplicate_is_reported():
    rep = audit_cqs.audit_corpus(SYNTH, ground_truth_dir=None)
    dupes = rep["cross_scenario_near_duplicates"]
    pairs = {(d["a"]["scenario_id"], d["b"]["scenario_id"]) for d in dupes}
    assert ("SYN-01", "SYN-02") in pairs or ("SYN-02", "SYN-01") in pairs, dupes


def test_better_fit_elsewhere_is_recorded():
    rep = audit_cqs.audit_corpus(SYNTH, ground_truth_dir=None)
    intruder = _cq(rep, "SYN-01", 3)
    codes = [f["code"] for f in intruder["flags"]]
    assert "better_fit_elsewhere" in codes, codes


def test_empty_and_malformed_entries():
    bad = [
        {
            "scenario_id": "SYN-03",
            "scenario_text": "Books are written by authors and published by publishers.",
            "ontology": "http://example.org/Book.ttl",
            "cq_list": [
                "Who is the author of a given book?",
                "",
                "   ",
                None,
                "Which publisher published a given book?",
            ],
        }
    ]
    rep = audit_cqs.audit_corpus(bad, ground_truth_dir=None)
    verdicts = [c["verdict"] for c in rep["scenarios"][0]["cqs"]]
    assert verdicts[1] == "malformed" and verdicts[2] == "malformed"
    assert verdicts[3] == "malformed"
    assert verdicts[0] == "clean" and verdicts[4] == "clean"


def test_boilerplate_is_flagged():
    bad = [
        {
            "scenario_id": "SYN-04",
            "scenario_text": "Books are written by authors and published by publishers.",
            "ontology": "http://example.org/Book.ttl",
            "cq_list": [
                "Who is the author of a given book?",
                "Which publisher published a given book?",
                "Output only the ontology in Turtle syntax inside a ```turtle code block.",
            ],
        }
    ]
    rep = audit_cqs.audit_corpus(bad, ground_truth_dir=None)
    cq = rep["scenarios"][0]["cqs"][2]
    codes = [f["code"] for f in cq["flags"]]
    assert "prompt_boilerplate" in codes, cq
    assert cq["verdict"] in FLAGGED


def test_auditor_never_mutates_the_corpus(tmp_path):
    before = open(SCENARIOS, "rb").read()
    audit_cqs.audit_corpus(audit_cqs.load_scenarios(SCENARIOS), ground_truth_dir=None)
    assert open(SCENARIOS, "rb").read() == before


# --------------------------------------------------------------------------
# CLI contract: writes a report, prints a summary, gates CI with exit code.
# --------------------------------------------------------------------------


def test_cli_exits_nonzero_on_real_corpus_and_writes_report(tmp_path):
    out = tmp_path / "report.json"
    proc = subprocess.run(
        [sys.executable, SCRIPT, "--scenarios", SCENARIOS, "--out", str(out)],
        capture_output=True,
        text=True,
        cwd=REPO,
    )
    assert proc.returncode != 0, "the real corpus contains a misfiled CQ; CI must fail"
    assert out.exists()
    rep = json.loads(out.read_text(encoding="utf-8"))
    assert rep["totals"]["cqs"] == 175
    assert rep["totals"]["almost_certainly_misfiled"] >= 1
    assert "2023-135-01" in proc.stdout


def test_cli_exits_zero_on_a_clean_corpus(tmp_path):
    clean = tmp_path / "clean.json"
    clean.write_text(json.dumps([SYNTH[1]]), encoding="utf-8")
    out = tmp_path / "report.json"
    proc = subprocess.run(
        [sys.executable, SCRIPT, "--scenarios", str(clean), "--out", str(out)],
        capture_output=True,
        text=True,
        cwd=REPO,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr


# ==========================================================================
# D1 -- ANTI-MONOTONICITY
#
# The auditor used to anchor each CQ partly against its SIBLING CQs.  That made
# the decision rule anti-monotone in contamination: a second foreign CQ anchored
# the first one, the first one's verdict fell back from
# almost_certainly_misfiled to suspicious, and the CI gate flipped red -> green.
# More contamination produced less detection, so "1 of 175 misfiled" could be an
# artefact of the contamination being large rather than small.
#
# The contract asserted below: injecting k additional foreign CQs into a
# scenario must never remove a CQ from the flagged set and never soften the exit
# code.
# ==========================================================================

GT_DIR = os.path.join(REPO, "data", "ground_truth")

GATING = ("almost_certainly_misfiled", "malformed")


def _audit(scenarios):
    return audit_cqs.audit_corpus(scenarios, ground_truth_dir=GT_DIR)


def _gating_set(report):
    """The CQs whose verdict makes CI fail -- the set that must never shrink."""
    return {
        (sc["scenario_id"], cq["index"])
        for sc in report["scenarios"]
        for cq in sc["cqs"]
        if cq["verdict"] in GATING
    }


def _nonclean_set(report):
    return {
        (sc["scenario_id"], cq["index"])
        for sc in report["scenarios"]
        for cq in sc["cqs"]
        if cq["verdict"] != "clean"
    }


def _exit_severity(report):
    t = report["totals"]
    return 1 if (t["almost_certainly_misfiled"] or t["malformed"]) else 0


@pytest.fixture(scope="module")
def real_scenarios():
    return audit_cqs.load_scenarios(SCENARIOS)


def _foreign_pool(scenarios, target_sid, want, base_report):
    """`want` real CQs harvested from OTHER scenarios, deterministically chosen.

    They are ranked by lexical similarity to the already-flagged CQs of the target
    scenario.  That is the realistic contamination model -- a harvest error pulls a
    coherent BLOCK of questions from one neighbouring pattern, not one stray line --
    and it is precisely the case a sibling-anchored rule gets wrong: the block
    anchors itself and the whole block looks locally consistent.
    """
    anchors = [
        set(audit_cqs.content_tokens(cq["text"]))
        for sc in base_report["scenarios"] if sc["scenario_id"] == target_sid
        for cq in sc["cqs"] if cq["verdict"] != "clean"
    ]
    cands = []
    for s in scenarios:
        if s["scenario_id"] == target_sid:
            continue
        for i, c in enumerate(s.get("cq_list") or []):
            if not isinstance(c, str) or not c.strip():
                continue
            toks = set(audit_cqs.content_tokens(c))
            if not toks:
                continue
            sim = max(
                (len(toks & a) / float(len(toks | a)) for a in anchors if a), default=0.0
            )
            cands.append((-sim, s["scenario_id"], i, c))
    cands.sort()
    return [c[3] for c in cands[:want]]


@pytest.mark.parametrize("k", [1, 2, 3, 5])
def test_injecting_more_foreign_cqs_never_reduces_detection(real_scenarios, k):
    """MONOTONICITY PROPERTY, checked for every scenario in the corpus.

    Adding foreign CQs may only ever add detections.  It must never un-flag a CQ
    that was flagged before, never turn a non-clean CQ clean, and never soften the
    exit code.
    """
    base = _audit(real_scenarios)
    base_gating = _gating_set(base)
    base_nonclean = _nonclean_set(base)
    base_exit = _exit_severity(base)

    for sc in real_scenarios:
        sid = sc["scenario_id"]
        intruders = _foreign_pool(real_scenarios, sid, k, base)
        assert len(intruders) == k
        contaminated = copy.deepcopy(real_scenarios)
        for s in contaminated:
            if s["scenario_id"] == sid:
                s["cq_list"] = list(s["cq_list"]) + intruders
        rep = _audit(contaminated)

        lost = base_gating - _gating_set(rep)
        assert not lost, (
            "ANTI-MONOTONE: injecting %d foreign CQ(s) into %s un-flagged %s"
            % (k, sid, sorted(lost))
        )
        lost_soft = base_nonclean - _nonclean_set(rep)
        assert not lost_soft, (
            "ANTI-MONOTONE: injecting %d foreign CQ(s) into %s exonerated %s"
            % (k, sid, sorted(lost_soft))
        )
        assert _exit_severity(rep) >= base_exit, (
            "ANTI-MONOTONE: injecting %d foreign CQ(s) into %s softened the exit code"
            % (k, sid)
        )


def test_the_reported_regression_second_intruder_does_not_hide_the_first(real_scenarios):
    """The exact measured symptom, pinned as its own regression test.

    2023-135-01 CQ #3 is confirmed contamination.  Adding one more river CQ used
    to make it stop being flagged and flipped the exit code from 1 to 0.
    """
    second = "Which rivers flow through the national park boundaries?"
    contaminated = copy.deepcopy(real_scenarios)
    for s in contaminated:
        if s["scenario_id"] == "2023-135-01":
            s["cq_list"] = list(s["cq_list"]) + [second]
    rep = _audit(contaminated)
    river = _cq(rep, "2023-135-01", 2)
    assert river["verdict"] == "almost_certainly_misfiled", (
        "a second river CQ must not exonerate the first, got %r" % river["verdict"]
    )
    assert _exit_severity(rep) == 1


def test_verdicts_do_not_depend_on_sibling_cqs(real_scenarios):
    """Anchoring must come from the scenario text and pattern vocabulary only.

    Deleting every sibling of a CQ must not change that CQ's verdict; if it does,
    the rule is anchored on the very list it is auditing.
    """
    base = _audit(real_scenarios)
    for sc in real_scenarios:
        sid = sc["scenario_id"]
        cqs = sc.get("cq_list") or []
        for i in range(len(cqs)):
            if not isinstance(cqs[i], str) or not cqs[i].strip():
                continue
            # Every other scenario keeps its text and pattern vocabulary (the
            # best-fit signal needs them) but loses its CQs, so the audit is cheap
            # and no sibling of any kind survives.
            solo = copy.deepcopy(real_scenarios)
            for s in solo:
                s["cq_list"] = [cqs[i]] if s["scenario_id"] == sid else []
            rep = _audit(solo)
            got = _cq(rep, sid, 0)["verdict"]
            want = _cq(base, sid, i)["verdict"]
            # duplicate flags legitimately vanish when the siblings do; only the
            # anchoring verdict is compared, so ignore duplicate-only downgrades.
            if want == "suspicious" and got == "clean":
                codes = {f["code"] for f in _cq(base, sid, i)["flags"]}
                if codes <= {"duplicate_within_scenario", "duplicate_across_scenarios",
                             "foreign_cq_island"}:
                    continue
            assert got == want, (
                "%s #%d: verdict %r alone vs %r with siblings -- the rule is "
                "anchored on the CQ list it audits" % (sid, i, got, want)
            )


def test_island_of_two_foreign_cqs_is_more_damning_not_less(real_scenarios):
    """Connected-component detection: a coherent island must raise suspicion."""
    second = "Which rivers flow through the national park boundaries?"
    base = _audit(real_scenarios)
    lone = _cq(base, "2023-135-01", 2)

    contaminated = copy.deepcopy(real_scenarios)
    for s in contaminated:
        if s["scenario_id"] == "2023-135-01":
            s["cq_list"] = list(s["cq_list"]) + [second]
    rep = _audit(contaminated)
    isle = _cq(rep, "2023-135-01", 2)
    partner = _cq(rep, "2023-135-01", 3)

    assert isle["suspicion"] >= lone["suspicion"]
    codes = {f["code"] for f in isle["flags"]}
    assert "foreign_cq_island" in codes, codes
    assert partner["verdict"] == "almost_certainly_misfiled", partner


def test_known_bad_cq_holds_top_suspicion(real_report):
    """The river CQ must sit at the top of the ranking, not merely be flagged."""
    river = _cq(real_report, "2023-135-01", 2)
    every = [
        cq for sc in real_report["scenarios"] for cq in sc["cqs"]
    ]
    assert river["suspicion"] == max(c["suspicion"] for c in every), (
        "river CQ suspicion %.3f is not the maximum" % river["suspicion"]
    )
    top = real_report["ranked_by_suspicion"][0]
    assert (top["scenario_id"], top["index"]) == ("2023-135-01", 2), top


# ==========================================================================
# D2 -- RECALL against injected cross-pattern intruders
#
# Every real CQ of scenario X is injected, one at a time, into every other
# scenario Y, and we ask whether the auditor calls it contaminated.  Injections
# are batched so each audit sees at most ONE intruder per scenario: this measures
# the LONE-intruder recall, which is the hard case and the honest one (the
# connected-component signal cannot help when there is only one).
# ==========================================================================


KNOWN_CONTAMINATED = [("2023-135-01", 2)]  # the river CQ: genuine, not a false alarm


@pytest.fixture(scope="module")
def recall_measurement(real_scenarios):
    return audit_cqs.measure_injection_recall(
        real_scenarios, ground_truth_dir=GT_DIR,
        known_contaminated=KNOWN_CONTAMINATED,
    )


def test_cross_pattern_intruder_recall(recall_measurement, capsys):
    """The honest recall number, printed in full whether or not it flatters us."""
    r = recall_measurement
    with capsys.disabled():
        print("")
        print("  CROSS-PATTERN INTRUDER RECALL (%s)" % r["design"])
        print("    injections                 %d" % r["injections"])
        print("    detected (TP)              %d" % r["true_positives"])
        print("    missed (FN)                %d" % r["false_negatives"])
        print("    RECALL                     %.4f" % r["recall"])
        print("    precision                  %.4f" % r["precision"])
        print("    native gating verdicts     %s" % r["native_gating_verdicts"])
        print("    native false positives     %d (%.4f of 175 CQs)"
              % (len(r["native_false_positives"]), r["native_false_positive_rate"]))
        for fp in r["native_false_positives"]:
            print("      native FP: %s #%d" % (fp[0], fp[1] + 1))
        for miss in r["missed_examples"][:5]:
            print("      missed:    %s #%d -> %s  %r"
                  % (miss["source_scenario"], miss["source_index"] + 1,
                     miss["injected_into"], miss["text"][:64]))

    assert r["injections"] > 2000, r["injections"]
    assert r["recall"] >= 0.80, (
        "lone-intruder recall %.4f is below the 0.80 contract" % r["recall"]
    )
    assert r["precision"] >= 0.80, r["precision"]


def test_the_known_intruder_is_among_the_native_gating_verdicts(recall_measurement):
    assert ["2023-135-01", 2] in recall_measurement["native_gating_verdicts"]


def test_native_false_positive_rate_is_bounded(recall_measurement):
    """False alarms on the uncontaminated corpus, excluding the genuine intruder."""
    r = recall_measurement
    assert r["native_false_positive_rate"] <= 0.02, r["native_false_positives"]


# --------------------------------------------------------------------------
# Sibling agreement: may downgrade suspicion, must never exonerate.
# --------------------------------------------------------------------------

CORROB = [
    {
        "scenario_id": "COR-01",
        "scenario_text": "A pizza restaurant serves pizzas with different toppings; "
        "customers order pizzas and the kitchen prepares each topping.",
        "ontology": "http://example.org/PizzaTopping.ttl",
        "cq_list": [
            "Which toppings are on a given pizza?",
            "Which spacecraft launched from which orbital platform?",
        ],
    },
    {
        "scenario_id": "COR-02",
        "scenario_text": "Spacecraft launches are recorded with their orbital platform "
        "and launch window.",
        "ontology": "http://example.org/Launch.ttl",
        "cq_list": ["What is the launch window of a spacecraft?"],
    },
]


def test_a_zero_anchored_cq_can_never_be_corroborated():
    """The exoneration channel that caused D1 is closed by construction.

    A CQ with no home anchoring has no home-vocabulary word to share, so no
    sibling -- however well anchored, however many -- can corroborate it.
    """
    rep = audit_cqs.audit_corpus(CORROB, ground_truth_dir=None)
    intruder = _cq(rep, "COR-01", 1)
    assert intruder["evidence"]["home_support"] == 0.0
    assert intruder["evidence"]["corroborating_siblings"] == []
    assert intruder["verdict"] == "almost_certainly_misfiled"

    # Pile on well-anchored siblings; the verdict must not move.
    piled = copy.deepcopy(CORROB)
    piled[0]["cq_list"] = piled[0]["cq_list"] + [
        "Which customer ordered which pizza?",
        "How long does the kitchen take to prepare a pizza?",
        "Which topping does the kitchen prepare first?",
    ]
    rep2 = audit_cqs.audit_corpus(piled, ground_truth_dir=None)
    still = _cq(rep2, "COR-01", 1)
    assert still["verdict"] == "almost_certainly_misfiled", still
    assert still["evidence"]["corroborating_siblings"] == []


def test_corroboration_lowers_suspicion_without_changing_the_verdict():
    """A weakly anchored but genuine CQ may have its suspicion reduced only."""
    lonely = [
        {
            "scenario_id": "COR-03",
            "scenario_text": "Books are written by authors and published by publishers.",
            "ontology": "http://example.org/Book.ttl",
            "cq_list": ["Which publisher published the book of a prolific author?"],
        }
    ]
    with_sibling = copy.deepcopy(lonely)
    with_sibling[0]["cq_list"] = with_sibling[0]["cq_list"] + [
        "Who is the author of a given book?"
    ]
    a = _cq(audit_cqs.audit_corpus(lonely, ground_truth_dir=None), "COR-03", 0)
    b = _cq(audit_cqs.audit_corpus(with_sibling, ground_truth_dir=None), "COR-03", 0)
    assert b["evidence"]["corroborating_siblings"], b["evidence"]
    assert b["suspicion"] < a["suspicion"]
    assert b["verdict"] == a["verdict"]
