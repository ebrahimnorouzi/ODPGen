"""Tests for scripts/aggregate_eval.py — the missing producer of eval/odp_eval_results.csv
and eval/odp_eval_ranking.csv (defect V8).

The central test here is `test_oops_component_discriminates_between_formulas`: it uses a
fixture on which the two candidate OOPS! component formulas give DIFFERENT answers, so the
suite actually distinguishes

    A)  1 / (1 + pitfalls_total)                  <- previous round's auditor, WRONG
    B)  1 / (1 + important_count + 2*critical)    <- correct

and `test_committed_corpus_is_reproduced_by_weighted_formula_only`, which measures the two
formulas against the committed artifact rather than taking anyone's word for it.
"""

from __future__ import annotations

import csv
import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "aggregate_eval.py"
COMMITTED_RESULTS = REPO / "eval" / "odp_eval_results.csv"
COMMITTED_RANKING = REPO / "eval" / "odp_eval_ranking.csv"
CORPUS = REPO / "eval_results"

# The one model whose per-file records under eval_results/ were re-evaluated after
# eval/*.csv was committed.  Every other model must reproduce cell for cell.
STALE_MODEL = "gpt-5.4"


def _load_module():
    spec = importlib.util.spec_from_file_location("aggregate_eval", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["aggregate_eval"] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def agg():
    if not SCRIPT.exists():
        pytest.fail(f"{SCRIPT} does not exist — V8 is unfixed")
    return _load_module()


# ── record fixtures ──────────────────────────────────────────────────────────

def _record(model, config, id_, *, parseable=True, consistent=True,
            important=0, critical=0, minor=0, scanned=True, codes=None,
            pass_rate=None, cq_skipped=False):
    rec = {"id": id_, "model": model, "config": config}
    if parseable:
        rec["ontometrics"] = {
            "triples_count": 10, "classes_count": 2, "object_properties_count": 1,
            "datatype_properties_count": 0, "annotation_properties_count": 0,
            "individuals_count": 0, "subclass_axioms": 1, "equivalence_axioms": 0,
            "disjoint_axioms": 0, "restriction_axioms": 0,
            "attribute_richness": 0.0, "relationship_richness": 0.5,
            "avg_subclass_per_class": 0.5,
        }
        rec["reasoner"] = {
            "consistent": consistent, "triples_before": 10, "triples_after": 20,
            "inferred_triples": 10, "unsatisfiable_classes": [],
        }
    else:
        err = "Could not parse ontology in any known format"
        rec["ontometrics"] = {"error": err}
        rec["reasoner"] = {"error": err}
    if scanned:
        rec["oops"] = {
            "status_code": 200,
            "pitfall_codes": list(codes or []),
            "pitfalls_total": important + critical,
            "pitfalls": [],
            "minor_pitfalls_skipped": minor,
            "important_count": important,
            "critical_count": critical,
        }
    else:
        rec["oops"] = {"error": "RDF/XML conversion failed"}
    if cq_skipped:
        rec["cq_verification"] = {"skipped": True, "reason": "no CQs available for this config"}
    elif pass_rate is None:
        rec["cq_verification"] = {"error": "unparseable"}
    else:
        rec["cq_verification"] = {
            "cqs_total": 4, "cqs_passed": int(round(pass_rate * 4)),
            "cqs_failed": 4 - int(round(pass_rate * 4)),
            "pass_rate": pass_rate, "results": [],
        }
    return rec


def _write_tree(root: Path, records):
    for rec in records:
        d = root / rec["model"] / rec["config"]
        d.mkdir(parents=True, exist_ok=True)
        (d / f"{rec['id']}.json").write_text(json.dumps(rec), encoding="utf-8")
    return root


# ── 1. the discriminating OOPS! test ─────────────────────────────────────────

def test_oops_component_discriminates_between_formulas(agg):
    """One important + one critical pitfall.

    1/(1 + pitfalls_total)              = 1/3  = 0.3333...   (WRONG)
    1/(1 + important + 2*critical)      = 1/4  = 0.25        (CORRECT)

    A test that cannot tell these apart is worthless, so assert the exact value
    AND assert it is not the other formula's value.
    """
    rec = _record("m", "c", "x", important=1, critical=1, minor=3)
    got = agg.oops_component(rec)
    assert got == pytest.approx(0.25), (
        "critical pitfalls must weigh double: expected 1/(1+1+2*1)=0.25")
    assert got != pytest.approx(1.0 / 3.0), (
        "component collapsed to 1/(1+pitfalls_total) — critical weight lost")
    assert agg.oops_component(rec, formula="total") == pytest.approx(1.0 / 3.0)


def test_oops_component_ignores_minor_pitfalls(agg):
    rec = _record("m", "c", "x", important=0, critical=0, minor=7)
    assert agg.oops_component(rec) == pytest.approx(1.0)


def test_oops_component_two_criticals(agg):
    rec = _record("m", "c", "x", important=0, critical=2)
    assert agg.oops_component(rec) == pytest.approx(1.0 / 5.0)
    assert agg.oops_component(rec, formula="total") == pytest.approx(1.0 / 3.0)


def test_oops_component_unscanned_is_zero(agg):
    rec = _record("m", "c", "x", parseable=False, scanned=False)
    assert agg.oops_component(rec) == 0.0


def test_oops_component_rejects_scanned_record_without_severity_counts(agg):
    """A scanned record missing important/critical must NOT silently score 0 or 1."""
    rec = _record("m", "c", "x", important=1, critical=0)
    del rec["oops"]["important_count"]
    del rec["oops"]["critical_count"]
    with pytest.raises(ValueError):
        agg.oops_component(rec)


# ── 2. schema reconciliation with the committed artifacts ────────────────────

@pytest.mark.skipif(not COMMITTED_RESULTS.exists(), reason="committed artifact absent")
def test_results_columns_match_committed_header(agg):
    committed = COMMITTED_RESULTS.read_text(encoding="utf-8").splitlines()[0]
    assert list(agg.RESULT_FIELDS) == committed.split(",")


@pytest.mark.skipif(not COMMITTED_RANKING.exists(), reason="committed artifact absent")
def test_ranking_columns_match_committed_header(agg):
    committed = COMMITTED_RANKING.read_text(encoding="utf-8").splitlines()[0]
    assert list(agg.RANKING_FIELDS) == committed.split(",")


# ── 3. aggregate arithmetic ──────────────────────────────────────────────────

def test_structural_score_is_mean_of_consistency_rate_and_avg_oops(agg, tmp_path):
    recs = [
        _record("m", "cq-only", "a", consistent=True, important=1, critical=1),
        _record("m", "cq-only", "b", consistent=True, important=0, critical=0),
        _record("m", "cq-only", "c", parseable=False, scanned=False),
        _record("m", "cq-only", "d", parseable=False, scanned=False),
    ]
    _write_tree(tmp_path, recs)
    rows = agg.build_ranking_rows(agg.load_records(tmp_path))
    row = [r for r in rows if r.get("config") == "cq-only"][0]
    # consistency 2/4 = 0.5 ; oops (0.25 + 1.0 + 0 + 0)/4 = 0.3125
    assert float(row["consistency_rate"]) == pytest.approx(0.5)
    assert float(row["avg_oops_score"]) == pytest.approx(0.3125)
    # unrounded internal value is the exact mean; the CSV cell is it rounded to 4dp
    assert row["_structural_unrounded"] == pytest.approx((0.5 + 0.3125) / 2)
    assert float(row["structural_score"]) == round((0.5 + 0.3125) / 2, 4)


def test_unscored_files_stay_in_the_denominator(agg, tmp_path):
    """Unparseable files must lower the average, never be dropped from it."""
    recs = [_record("m", "cq-only", "a", important=0, critical=0)]
    recs += [_record("m", "cq-only", f"z{i}", parseable=False, scanned=False)
             for i in range(3)]
    _write_tree(tmp_path, recs)
    row = [r for r in agg.build_ranking_rows(agg.load_records(tmp_path))
           if r.get("config") == "cq-only"][0]
    assert int(row["n_total"]) == 4
    assert float(row["avg_oops_score"]) == pytest.approx(0.25)


def test_avg_cq_pass_rate_blank_when_every_file_skipped(agg, tmp_path):
    recs = [_record("m", "scenario-only", i, cq_skipped=True) for i in "ab"]
    _write_tree(tmp_path, recs)
    row = [r for r in agg.build_ranking_rows(agg.load_records(tmp_path))
           if r.get("config") == "scenario-only"][0]
    assert row["avg_cq_pass_rate"] == ""


def test_duplicate_record_key_is_an_error(agg, tmp_path):
    (tmp_path / "one").mkdir()
    (tmp_path / "two").mkdir()
    rec = _record("m", "cq-only", "a")
    (tmp_path / "one" / "a.json").write_text(json.dumps(rec), encoding="utf-8")
    (tmp_path / "two" / "a.json").write_text(json.dumps(rec), encoding="utf-8")
    with pytest.raises(ValueError, match="[Dd]uplicate"):
        agg.load_records(tmp_path)


def test_ranking_orders_by_structural_score_then_model_name(agg, tmp_path):
    recs = []
    for m, imp in (("zmodel", 0), ("amodel", 0), ("bmodel", 1)):
        recs += [_record(m, "cq-only", "a", important=imp),
                 _record(m, "cq-only", "b", important=imp)]
    _write_tree(tmp_path, recs)
    rows = [r for r in agg.build_ranking_rows(agg.load_records(tmp_path))
            if r.get("config") == "cq-only"]
    assert [r["model"] for r in rows] == ["amodel", "zmodel", "bmodel"]
    assert [r["rank_in_config"] for r in rows] == [1, 2, 3]


# ── 4. end-to-end CLI ────────────────────────────────────────────────────────

def test_cli_writes_both_csvs(agg, tmp_path):
    indir = _write_tree(tmp_path / "in", [
        _record("m", "cq-only", "a", important=1, critical=1, pass_rate=0.5),
        _record("m", "cq-only", "b"),
    ])
    outdir = tmp_path / "out"
    rc = agg.main(["--in", str(indir), "--out", str(outdir)])
    assert rc == 0
    res = list(csv.DictReader((outdir / "odp_eval_results.csv").open(encoding="utf-8")))
    assert len(res) == 2
    assert res[0]["oops_critical_count"] == "1"
    assert res[0]["parseable"] == "True"
    rank_lines = (outdir / "odp_eval_ranking.csv").read_text(encoding="utf-8").splitlines()
    assert rank_lines[0] == ",".join(agg.RANKING_FIELDS)
    # blank separator row after the ALL block
    assert any(set(line) == {","} for line in rank_lines[1:])


# ── 5. the measurement that matters: which formula reproduces the artifact ───

@pytest.mark.skipif(not (CORPUS.exists() and COMMITTED_RANKING.exists()),
                    reason="committed corpus absent")
def test_committed_corpus_is_reproduced_by_weighted_formula_only(agg):
    """Recompute avg_oops_score over the committed per-file records under both
    formulas and check against eval/odp_eval_ranking.csv.

    Restricted to the five models whose per-file records still correspond to the
    committed CSV (gpt-5.4 was re-evaluated afterwards; see the stale-artifact
    test below).
    """
    records = [r for r in agg.load_records(CORPUS) if r["model"] != STALE_MODEL]
    committed = {(r["model"], r["config"]): r
                 for r in csv.DictReader(COMMITTED_RANKING.open(encoding="utf-8"))
                 if r["model"]}

    def mismatches(formula):
        bad = []
        for row in agg.build_ranking_rows(records, oops_formula=formula):
            key = (row["model"], row["config"])
            if key not in committed:
                continue
            got = agg.render_cell(row["avg_oops_score"])
            if got != committed[key]["avg_oops_score"]:
                bad.append((key, committed[key]["avg_oops_score"], got))
        return bad

    weighted = mismatches("weighted")
    total = mismatches("total")
    assert weighted == [], f"weighted formula failed to reproduce: {weighted}"
    assert total, "the two formulas were indistinguishable on this corpus"


@pytest.mark.skipif(not (CORPUS.exists() and COMMITTED_RESULTS.exists()),
                    reason="committed corpus absent")
def test_per_file_rows_reproduce_committed_csv_except_the_stale_model(agg):
    records = agg.load_records(CORPUS)
    regenerated = {(r["model"], r["config"], r["id"]): r
                   for r in agg.build_results_rows(records)}
    committed = list(csv.DictReader(COMMITTED_RESULTS.open(encoding="utf-8")))
    assert len(committed) == len(regenerated)

    differing_models = set()
    for crow in committed:
        key = (crow["model"], crow["config"], crow["id"])
        assert key in regenerated, f"no record regenerates committed row {key}"
        mine = regenerated[key]
        for field in agg.RESULT_FIELDS:
            if agg.render_cell(mine[field]) != crow[field]:
                differing_models.add(crow["model"])
    assert differing_models <= {STALE_MODEL}, (
        f"unexpected models differ from the committed artifact: {differing_models}")


@pytest.mark.skipif(not (CORPUS.exists() and COMMITTED_RANKING.exists()),
                    reason="committed corpus absent")
def test_ranking_reproduces_committed_rows_except_the_stale_model(agg):
    records = [r for r in agg.load_records(CORPUS) if r["model"] != STALE_MODEL]
    committed = {(r["model"], r["config"]): r
                 for r in csv.DictReader(COMMITTED_RANKING.open(encoding="utf-8"))
                 if r["model"]}
    checked = 0
    drift = []
    for row in agg.build_ranking_rows(records):
        key = (row["model"], row["config"])
        if key not in committed:
            continue
        checked += 1
        for field in ("n_total", "n_parseable", "n_consistent", "n_oops_scanned",
                      "parse_rate", "consistency_rate", "avg_oops_score",
                      "structural_score", "avg_cq_pass_rate"):
            if agg.render_cell(row[field]) != committed[key][field]:
                drift.append((key, field, committed[key][field], row[field]))
    assert checked == 30, f"expected 30 comparable ranking rows, compared {checked}"

    # The V1 fix moves exactly one thing and nothing else: the structural score
    # of the one model whose only "parseable" outputs are zero-triple graphs.
    # Every other cell of every other row still reproduces the committed
    # artifact byte for byte — asserted here as an exact set, so a future change
    # that quietly moves a second cell fails this test rather than passing it.
    assert {(k, f) for k, f, _, _ in drift} == {
        ((VACUOUS_MODEL, agg.ALL_CONFIG), "structural_score"),
        ((VACUOUS_MODEL, "scenario-only"), "structural_score"),
    }, f"unexpected drift from the committed ranking: {drift}"
    for _, _, was, now in drift:
        assert float(now) < float(was), (
            "the gate may only ever remove credit awarded for saying nothing")


@pytest.mark.skipif(not (CORPUS.exists() and COMMITTED_RESULTS.exists()),
                    reason="committed corpus absent")
def test_compare_reports_the_stale_gpt_rows(agg, tmp_path):
    """--compare must surface, not hide, the drift between the committed artifact
    and the per-file records it was supposedly derived from."""
    diffs = agg.compare_results(agg.build_results_rows(agg.load_records(CORPUS)),
                                COMMITTED_RESULTS)
    assert diffs, "compare found no drift, but gpt-5.4 was re-evaluated after publication"
    assert {d["model"] for d in diffs} == {STALE_MODEL}
    assert len({(d["model"], d["config"], d["id"]) for d in diffs}) == 44


# ═════════════════════════════════════════════════════════════════════════════
# 6. THE V1 BYPASS — empty ontologies must not win the paper's structural table
# ═════════════════════════════════════════════════════════════════════════════
#
# batch_evaluate.py already refuses to pay a 0-triple graph for its vacuous
# `consistent: true` (run_reasoner emits consistency_credit=0.0, and
# structural_content_gate caps the file at 0.0).  aggregate_eval.py is the ONLY
# producer of eval/odp_eval_ranking.csv — the paper's structural table — and it
# bypassed both: is_consistent() read the raw reasoner verdict and _summarise()
# recomputed (consistency_rate + avg_oops)/2 from scratch, ignoring the record's
# own structural block.  70 empty files therefore scored 1.0 and ranked FIRST.
#
# These are the tests whose absence let that survive a full round.


def _empty_graph_record(model, config, id_, *, structural_block=None,
                        credit=True):
    """A file that parsed but says NOTHING: 0 triples.

    The reasoner really does return consistent=True on it (there is nothing to
    contradict) and OOPS! really does return zero pitfalls.  Both verdicts are
    vacuous.  `credit=True` emits batch_evaluate's consistency_credit field;
    credit=False simulates an older record that carries only the raw verdict, so
    the gate has to be re-derived from triples_before.
    """
    rec = {"id": id_, "model": model, "config": config}
    rec["ontometrics"] = {
        "triples_count": 0, "classes_count": 0, "object_properties_count": 0,
        "datatype_properties_count": 0, "annotation_properties_count": 0,
        "individuals_count": 0, "subclass_axioms": 0, "equivalence_axioms": 0,
        "disjoint_axioms": 0, "restriction_axioms": 0,
        "attribute_richness": 0.0, "relationship_richness": 0.0,
        "avg_subclass_per_class": 0.0,
    }
    rec["reasoner"] = {
        "consistent": True, "triples_before": 0, "triples_after": 0,
        "inferred_triples": 0, "unsatisfiable_classes": [],
    }
    if credit:
        rec["reasoner"]["vacuous"] = True
        rec["reasoner"]["consistency_credit"] = 0.0
    rec["oops"] = {
        "status_code": 200, "pitfall_codes": [], "pitfalls_total": 0,
        "pitfalls": [], "minor_pitfalls_skipped": 0,
        "important_count": 0, "critical_count": 0,
    }
    rec["cq_verification"] = {"cqs_total": 4, "cqs_passed": 0, "cqs_failed": 4,
                              "pass_rate": 0.0, "results": []}
    if structural_block is not None:
        rec["structural"] = structural_block
    return rec


def test_seventy_empty_ontologies_do_not_rank_first(agg, tmp_path):
    """THE blocking regression.

    A model that emitted 70 zero-triple files is handed a vacuous
    `consistent: true` by OWL-RL and a vacuous "0 pitfalls" by OOPS!.  Scoring
    those at face value gives structural_score 1.0 and rank 1 — the empty file
    published as the best ontology in the corpus.
    """
    recs = [_empty_graph_record("emptymodel", "cq-only", f"e{i:02d}")
            for i in range(70)]
    # A model that actually produced ontologies, and is imperfect: 2 important
    # pitfalls each -> oops component 1/3, consistency credit 1.0 -> 0.6667.
    recs += [_record("realmodel", "cq-only", f"r{i:02d}", important=2)
             for i in range(70)]
    _write_tree(tmp_path, recs)

    rows = agg.build_ranking_rows(agg.load_records(tmp_path))
    overall = [r for r in rows if r.get("config") == agg.ALL_CONFIG]
    ranked = [r["model"] for r in overall]

    empty = [r for r in overall if r["model"] == "emptymodel"][0]
    real = [r for r in overall if r["model"] == "realmodel"][0]

    assert float(empty["structural_score"]) == 0.0, (
        "70 zero-triple files scored above the floor: the vacuous reasoner "
        "verdict and the vacuous zero-pitfall verdict were both paid out")
    assert float(real["structural_score"]) > float(empty["structural_score"])
    assert ranked[0] == "realmodel", (
        f"the empty-output model ranked {ranked.index('emptymodel') + 1} of "
        f"{len(ranked)}; order was {ranked}")
    assert empty["rank_overall"] == len(ranked)


def test_empty_ontology_does_not_rank_first_without_a_credit_field(agg, tmp_path):
    """Same, for records predating batch_evaluate's consistency_credit field.

    The gate must be re-derived from triples_before == 0, not trusted to be
    pre-computed — every record in the committed eval_results/ corpus is of this
    older shape.
    """
    recs = [_empty_graph_record("emptymodel", "cq-only", f"e{i}", credit=False)
            for i in range(5)]
    recs += [_record("realmodel", "cq-only", f"r{i}", important=3)
             for i in range(5)]
    _write_tree(tmp_path, recs)
    overall = [r for r in agg.build_ranking_rows(agg.load_records(tmp_path))
               if r.get("config") == agg.ALL_CONFIG]
    empty = [r for r in overall if r["model"] == "emptymodel"][0]
    assert float(empty["structural_score"]) == 0.0
    assert overall[0]["model"] == "realmodel"


def test_records_own_structural_block_is_authoritative(agg, tmp_path):
    """When batch_evaluate already scored the file, use ITS number.

    The record below says structural_score 0.0 (gate: empty) while its raw
    reasoner verdict says consistent.  The aggregator must not recompute a
    better score than the evaluator that produced the record.
    """
    block = {"structural_score": 0.0, "consistency_component": 0.0,
             "oops_component": 0.0, "raw_score": 1.0, "gate": "empty",
             "gate_cap": 0.0, "counted": True, "skipped": False}
    recs = [_empty_graph_record("m", "cq-only", "a", structural_block=block)]
    _write_tree(tmp_path, recs)
    row = [r for r in agg.build_ranking_rows(agg.load_records(tmp_path))
           if r.get("config") == "cq-only"][0]
    assert float(row["structural_score"]) == 0.0
    assert agg.structural_component(recs[0]) == 0.0


def test_near_empty_cap_from_the_records_own_block_is_honoured(agg, tmp_path):
    """A non-zero authoritative score is used verbatim too, cap included."""
    rec = _record("m", "cq-only", "a", important=0, critical=0)
    rec["structural"] = {"structural_score": 0.5, "gate": "near_empty",
                         "gate_cap": 0.5, "raw_score": 1.0}
    _write_tree(tmp_path, [rec])
    row = [r for r in agg.build_ranking_rows(agg.load_records(tmp_path))
           if r.get("config") == "cq-only"][0]
    assert float(row["structural_score"]) == 0.5


def test_consistency_rate_still_reports_the_literal_reasoner_verdict(agg, tmp_path):
    """The gate removes SCORE, it does not falsify the reported verdict.

    consistency_rate / n_consistent stay the literal OWL-RL answer — that is
    what the committed artifact reports and what a reader checks against the
    per-file CSV.  Only structural_score is gated.
    """
    recs = [_empty_graph_record("m", "cq-only", f"e{i}") for i in range(4)]
    _write_tree(tmp_path, recs)
    row = [r for r in agg.build_ranking_rows(agg.load_records(tmp_path))
           if r.get("config") == "cq-only"][0]
    assert int(row["n_consistent"]) == 4
    assert float(row["consistency_rate"]) == 1.0
    assert float(row["structural_score"]) == 0.0


# ═════════════════════════════════════════════════════════════════════════════
# 7. eval_local records — an unavailable OOPS! scan is MISSING, not zero, not one
# ═════════════════════════════════════════════════════════════════════════════

def _eval_local_record(model, config, id_, *, status="ok", triples=60,
                       consistent=True):
    """The shape scripts/eval_local.py writes.

    OOPS! is a web service; eval_local runs offline, so it records
    skipped=True with pitfalls_total / important_count / critical_count all
    null.  That is "we never looked", which is neither 0 pitfalls nor infinitely
    many, and the aggregator must not turn it into either.
    """
    rec = {"id": id_, "model": model, "config": config,
           "rel_path": f"{model}/{config}/{id_}/ontology.ttl",
           "source": "local filesystem", "driver": "eval_local/1.0",
           "bytes": 512, "sha256": "0" * 64, "status": status,
           "parse_format": "turtle" if status != "unparseable" else None}
    if status == "unparseable":
        rec["ontometrics"] = {"error": "no graph (unparseable)"}
        rec["reasoner"] = {"error": "no graph (unparseable)"}
        scores = {"sev_score": 0.0, "consistency": None,
                  "oops_component": None, "structural_score": None,
                  "structural_components_used": []}
    else:
        rec["ontometrics"] = {
            "triples_count": triples, "classes_count": 5,
            "object_properties_count": 3, "datatype_properties_count": 1,
            "annotation_properties_count": 0, "individuals_count": 0,
            "subclass_axioms": 4, "equivalence_axioms": 0,
            "disjoint_axioms": 0, "restriction_axioms": 2,
            "attribute_richness": 0.2, "relationship_richness": 0.6,
            "avg_subclass_per_class": 0.8,
        }
        credit = 0.0 if triples == 0 else (1.0 if consistent else 0.0)
        rec["reasoner"] = {"consistent": consistent, "vacuous": triples == 0,
                           "consistency_credit": credit,
                           "triples_before": triples, "triples_after": triples * 2,
                           "inferred_triples": triples, "unsatisfiable_classes": []}
        scores = {"sev_score": 0.3, "consistency": credit,
                  "oops_component": None,
                  "structural_score": credit,
                  "structural_components_used": ["consistency"]}
    rec["cq_verification"] = {"method": "SEV", "cqs_total": 4, "cqs_passed": 1,
                              "cqs_failed": 3, "pass_rate": 0.25,
                              "sev_score": 0.3, "results": []}
    rec["oops"] = {
        "skipped": True,
        "reason": "OOPS! is an online service; eval_local.py runs offline",
        "pitfalls_total": None, "important_count": None, "critical_count": None,
    }
    rec["scores"] = scores
    return rec


def test_oops_component_is_missing_not_zero_when_never_scanned(agg):
    """skipped=True + null counts is a MISSING observation."""
    rec = _eval_local_record("m", "cq-only", "a")
    got = agg.oops_component(rec)
    assert got is None, (
        f"an OOPS! scan that never ran was scored {got!r}; it must be excluded")


def test_oops_component_rejects_a_skipped_scan_even_with_counts_present(agg):
    """skipped wins over any counts left in the block (batch_evaluate's rule)."""
    rec = _record("m", "cq-only", "a", important=0, critical=0)
    rec["oops"]["skipped"] = True
    assert agg.oops_component(rec) is None


def test_oops_component_requires_http_200_before_believing_a_scan(agg):
    rec = _record("m", "cq-only", "a", important=0, critical=0)
    rec["oops"]["status_code"] = 503
    assert agg.oops_component(rec) is None


def test_cli_reads_eval_local_records(agg, tmp_path):
    """`aggregate_eval.py --in eval_local --out DIR` must produce CSVs.

    It died with an uncaught ValueError ("scanned OOPS record ... has no
    important_count/critical_count") and wrote nothing at all.
    """
    recs = [_eval_local_record("m", "cq-only", f"a{i}") for i in range(3)]
    recs += [_eval_local_record("m", "cq-only", f"b{i}", status="unparseable")
             for i in range(1)]
    indir = _write_tree(tmp_path / "in", recs)
    outdir = tmp_path / "out"

    rc = agg.main(["--in", str(indir), "--out", str(outdir)])
    assert rc == 0
    assert (outdir / "odp_eval_results.csv").exists()
    assert (outdir / "odp_eval_ranking.csv").exists()

    rank = [r for r in csv.DictReader((outdir / "odp_eval_ranking.csv")
                                      .open(encoding="utf-8")) if r["model"]]
    row = [r for r in rank if r["config"] == "cq-only"][0]
    assert row["n_oops_scanned"] == "0"
    assert row["avg_oops_score"] == "", (
        "an OOPS! column with no scans anywhere must be blank, not 0.0")
    # weights renormalise onto consistency alone: 3 of 4 files consistent
    assert float(row["structural_score"]) == pytest.approx(0.75), (
        "with OOPS! missing the structural score must be the consistency "
        "credit alone, not that credit halved by a phantom 0.0 OOPS term")


def test_missing_oops_renormalises_rather_than_scoring_zero_or_one(agg, tmp_path):
    """The three wrong answers, all excluded explicitly.

    consistency credit 1.0, OOPS! never run:
        wrong (0.0 for the missing half)  -> 0.5
        wrong (1.0 for the missing half)  -> 1.0 ... and equal to a clean scan
        right (renormalised)              -> 1.0 on consistency alone
    Distinguish the last two by pairing with an inconsistent file, which the
    "1.0 for missing" rule would still lift to 0.5.
    """
    recs = [_eval_local_record("m", "cq-only", "ok1"),
            _eval_local_record("m", "cq-only", "bad1", consistent=False)]
    _write_tree(tmp_path, recs)
    row = [r for r in agg.build_ranking_rows(agg.load_records(tmp_path))
           if r.get("config") == "cq-only"][0]
    assert float(row["structural_score"]) == pytest.approx(0.5)
    assert agg.structural_component(recs[0]) == pytest.approx(1.0)
    assert agg.structural_component(recs[1]) == pytest.approx(0.0)


def test_failed_oops_scan_still_scores_zero(agg):
    """A scan that was attempted and could not be run on the artefact is 0.0.

    This is NOT the same as "never scanned": the file was submitted and OOPS!
    could make nothing of it, which is an observation about the file.  The
    committed corpus's 151 unparseable rows are of this kind and the published
    avg_oops_score averages them in at 0.0.
    """
    rec = _record("m", "cq-only", "a", parseable=False, scanned=False)
    assert agg.oops_component(rec) == 0.0


@pytest.mark.skipif(not (REPO / "eval_local").exists(),
                    reason="eval_local corpus absent")
def test_real_eval_local_corpus_aggregates(agg, tmp_path):
    """End to end on the real offline corpus — the command from the brief."""
    rc = agg.main(["--in", str(REPO / "eval_local"), "--out", str(tmp_path)])
    assert rc == 0
    rank = [r for r in csv.DictReader((tmp_path / "odp_eval_ranking.csv")
                                      .open(encoding="utf-8")) if r["model"]]
    assert rank, "no ranking rows produced from eval_local/"
    assert all(r["avg_oops_score"] == "" for r in rank), (
        "eval_local never scans OOPS!; every avg_oops_score cell must be blank")
    assert all(r["n_oops_scanned"] == "0" for r in rank)


# ═════════════════════════════════════════════════════════════════════════════
# 8. D3 — which OOPS! formula reproduces the committed artifact
# ═════════════════════════════════════════════════════════════════════════════

@pytest.mark.skipif(not (COMMITTED_RESULTS.exists() and COMMITTED_RANKING.exists()),
                    reason="committed artifacts absent")
def test_committed_ranking_is_reproduced_from_the_committed_results_csv():
    """Recompute avg_oops_score straight out of eval/odp_eval_results.csv.

    Deliberately does NOT go through aggregate_eval: it reads the two committed
    CSVs and arbitrates the formulas on the published numbers alone, so the
    answer cannot be an artifact of the code under test.
    """
    rows = list(csv.DictReader(COMMITTED_RESULTS.open(encoding="utf-8")))
    ranking = {(r["model"], r["config"]): r
               for r in csv.DictReader(COMMITTED_RANKING.open(encoding="utf-8"))
               if r["model"]}

    def component(row, formula):
        if row["oops_scanned"] != "True":
            return 0.0
        if formula == "total":
            return 1.0 / (1.0 + int(row["oops_pitfalls_total"]))
        return 1.0 / (1.0 + int(row["oops_important_count"])
                      + 2 * int(row["oops_critical_count"]))

    def misses(formula):
        bad = []
        for (model, config), crow in ranking.items():
            group = [r for r in rows if r["model"] == model
                     and (config == "ALL (aggregate)" or r["config"] == config)]
            if not group:
                continue
            avg = sum(component(r, formula) for r in group) / len(group)
            if f"{round(avg, 4)}" != crow["avg_oops_score"]:
                bad.append(((model, config), crow["avg_oops_score"], round(avg, 4)))
        return bad

    weighted = misses("weighted")
    total = misses("total")
    assert weighted == [], (
        "1/(1+important+2*critical) does NOT reproduce the committed "
        f"avg_oops_score: {weighted}")
    assert total, "the two formulas are indistinguishable on the committed data"


def test_the_two_formulas_disagree_on_a_group_not_only_on_one_file(agg, tmp_path):
    """A group-level fixture where the formulas give different avg_oops_score.

    Without a critical pitfall anywhere the two are identical, so a suite built
    only on important-pitfall fixtures cannot tell them apart.
    """
    recs = [_record("m", "cq-only", "a", important=1, critical=1),   # 0.25 vs 1/3
            _record("m", "cq-only", "b", important=0, critical=2),   # 0.20 vs 1/3
            _record("m", "cq-only", "c", important=2, critical=0)]   # 1/3 vs 1/3
    _write_tree(tmp_path, recs)
    loaded = agg.load_records(tmp_path)
    w = [r for r in agg.build_ranking_rows(loaded, oops_formula="weighted")
         if r.get("config") == "cq-only"][0]
    t = [r for r in agg.build_ranking_rows(loaded, oops_formula="total")
         if r.get("config") == "cq-only"][0]
    assert float(w["avg_oops_score"]) == pytest.approx(
        round((0.25 + 0.2 + 1 / 3) / 3, 4))
    assert float(t["avg_oops_score"]) == pytest.approx(round(1 / 3, 4))
    assert w["avg_oops_score"] != t["avg_oops_score"]
    assert float(w["structural_score"]) != float(t["structural_score"])


# ═════════════════════════════════════════════════════════════════════════════
# 9. what the fix changes in the committed artifact, stated exactly
# ═════════════════════════════════════════════════════════════════════════════

#: The one model whose published structural score came entirely from empty
#: graphs: all 5 of its "parseable" files are 0-triple.
VACUOUS_MODEL = "bigscience_bloomz-7b1"


@pytest.mark.skipif(not CORPUS.exists(), reason="committed corpus absent")
def test_the_committed_corpus_contains_vacuous_records(agg):
    """The premise of the whole fix, measured rather than assumed."""
    records = agg.load_records(CORPUS)
    vacuous = [r for r in records if agg.is_vacuous(r)]
    assert len(vacuous) == 5
    assert {r["model"] for r in vacuous} == {VACUOUS_MODEL}
    assert {r["config"] for r in vacuous} == {"scenario-only"}
    # OWL-RL calls every one of them consistent — on a graph with nothing in it
    assert all(agg.is_consistent(r) for r in vacuous)
    assert all(_section(r, "reasoner")["triples_before"] == 0 for r in vacuous)
    # OOPS! still scanned them and still returned a component above the floor
    # (P38/P39: one important + one critical -> 0.25), so the published
    # avg_oops_score cell is genuinely non-zero and is left exactly as it is.
    assert all(agg.oops_component(r) == pytest.approx(0.25) for r in vacuous)
    # ... and none of it may be paid out
    assert all(agg.consistency_credit(r) == 0.0 for r in vacuous)
    assert all(agg.structural_cap(r) == 0.0 for r in vacuous)
    assert all(agg.structural_component(r) == 0.0 for r in vacuous)


def _section(rec, name):
    return rec.get(name) or {}


@pytest.mark.skipif(not (CORPUS.exists() and COMMITTED_RANKING.exists()),
                    reason="committed corpus absent")
def test_vacuous_model_structural_score_collapses_to_the_floor(agg):
    """bloomz's published 0.0446 was paid entirely for five empty files."""
    committed = {(r["model"], r["config"]): r
                 for r in csv.DictReader(COMMITTED_RANKING.open(encoding="utf-8"))
                 if r["model"]}
    assert float(committed[(VACUOUS_MODEL, agg.ALL_CONFIG)]["structural_score"]) > 0
    rows = {(r["model"], r["config"]): r
            for r in agg.build_ranking_rows(agg.load_records(CORPUS))
            if r.get("model")}
    for config in (agg.ALL_CONFIG, "scenario-only"):
        row = rows[(VACUOUS_MODEL, config)]
        assert float(row["structural_score"]) == 0.0
        # the literal verdicts it was paid for are still reported, unchanged
        assert row["consistency_rate"] == float(
            committed[(VACUOUS_MODEL, config)]["consistency_rate"])
        assert row["avg_oops_score"] == float(
            committed[(VACUOUS_MODEL, config)]["avg_oops_score"])
