"""
Tests for scripts/eval_local.py -- the LOCAL-filesystem evaluation driver.

Background (defect V5)
---------------------
batch_evaluate.py main() calls list_ontology_paths(), which enumerates the
corpus over https://api.github.com/repos/<repo>/git/trees/<branch>.  Every
artifact the paper derives is therefore pinned to whatever happens to be on the
remote branch, and a file that exists ONLY in the local working tree -- a
recovered output, a regenerated output, the result of any repair -- can never be
scored at all.  In other words: nothing in the repair can be measured.

scripts/eval_local.py fixes that by walking the local tree and scoring each
ontology through the SAME functions batch_evaluate.py uses.  These tests pin the
three properties that make it trustworthy:

  (i)   the walker finds local files that the GitHub path would miss;
  (ii)  empty and unparseable ontologies land IN the denominator, never skipped;
  (iii) no network call is made -- requests/urllib/socket are all booby-trapped
        and the run still completes.

Plus the F3 guard: an ontology whose graph is empty must NOT come out
structurally perfect, and an unavailable OOPS! pitfall count must be recorded as
null and EXCLUDED from the score, never silently read as "zero pitfalls".
"""

import csv
import importlib.util
import json
import socket
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
EVAL_LOCAL_PATH = REPO_ROOT / "scripts" / "eval_local.py"


# ── module loader ─────────────────────────────────────────────────────────────

def _load_eval_local():
    """Import scripts/eval_local.py by path (scripts/ is not a package)."""
    if not EVAL_LOCAL_PATH.exists():
        pytest.fail(
            f"{EVAL_LOCAL_PATH} does not exist. V5 is unfixed: evaluation can "
            "only be driven over the GitHub API, so nothing in the local working "
            "tree can be measured."
        )
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    spec = importlib.util.spec_from_file_location("eval_local_under_test",
                                                  EVAL_LOCAL_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def el():
    return _load_eval_local()


@pytest.fixture(scope="module")
def be():
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    import batch_evaluate
    return batch_evaluate


# ── corpus fixture ────────────────────────────────────────────────────────────

# A tiny but real T-Box for scenario 2023-133-01 (the causal-event pattern):
# >20 bytes, parses, non-zero triples, and it genuinely answers 3 of the 4 CQs
# so its SEV score is strictly positive (0.7708 with the current engine).
GOOD_TTL = """@prefix owl: <http://www.w3.org/2002/07/owl#> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
@prefix ex: <http://example.org/odp#> .

ex:Event a owl:Class ; rdfs:label "Event" .
ex:Outcome a owl:Class ; rdfs:label "Outcome" .
ex:CausalLink a owl:Class ; rdfs:label "Causal link" .
ex:causes a owl:ObjectProperty ; rdfs:label "causes" ;
    rdfs:domain ex:Event ; rdfs:range ex:Outcome .
ex:hasOutcome a owl:ObjectProperty ; rdfs:label "has outcome" ;
    rdfs:domain ex:Event ; rdfs:range ex:Outcome .
ex:weight a owl:DatatypeProperty ; rdfs:label "weight" ;
    rdfs:domain ex:CausalLink .
"""

# Parses cleanly, but yields ZERO triples: prefixes only.  This is the V1/F3
# degenerate case -- the reasoner calls it "consistent" and OOPS! finds no
# pitfalls in it, so under the published structural metric it is PERFECT.
PREFIX_ONLY_TTL = """@prefix owl: <http://www.w3.org/2002/07/owl#> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
@prefix ex: <http://example.org/odp#> .
"""

# The 3-byte residue F1/F2 left behind.
STUB_TTL = "...\n"

# Not RDF in any syntax.
GARBAGE_TTL = "I'm sorry, I cannot produce an ontology for this scenario. {{{ <<<\n"


def _write(root: Path, model: str, config: str, sid: str, text: str) -> Path:
    d = root / model / config / sid
    d.mkdir(parents=True, exist_ok=True)
    p = d / "ontology.ttl"
    p.write_text(text, encoding="utf-8")
    return p


@pytest.fixture
def corpus(tmp_path):
    """A local outputs/ tree of six files, four of which are worthless.

    Scenario ids are real ids from data/scenarios/pattern_scenarios.json so the
    offline CQ verification has an authoritative cq_list to work with.
    """
    root = tmp_path / "outputs"
    # a model that IS on the remote
    _write(root, "published-model", "cq-only", "2023-133-01", GOOD_TTL)
    _write(root, "published-model", "cq-only", "2023-133-02", "")
    _write(root, "published-model", "cq-only", "2023-134-01", GARBAGE_TTL)
    _write(root, "published-model", "scenario-cq", "2023-135-01", PREFIX_ONLY_TTL)
    _write(root, "published-model", "scenario-cq", "2023-134-02", STUB_TTL)
    # a model that exists ONLY in the local working tree (e.g. recovered/regenerated)
    _write(root, "recovered-model", "cq-only", "2023-133-01", GOOD_TTL)
    return root


# ── (i) the walker finds local files the GitHub path would miss ───────────────

def test_walker_finds_local_files_that_the_github_path_would_miss(el, be, corpus,
                                                                  monkeypatch):
    """The remote tree is stale; the local tree is the truth."""
    # Whatever the remote says, it does not know about recovered-model.
    remote_paths = [
        "outputs/published-model/cq-only/2023-133-01/ontology.ttl",
        "outputs/published-model/cq-only/2023-133-02/ontology.ttl",
    ]

    found = el.iter_local_ontologies(corpus)
    keys = {(r["model"], r["config"], r["scenario_id"]) for r in found}

    assert ("recovered-model", "cq-only", "2023-133-01") in keys, (
        "the local walker missed a file that exists only in the working tree"
    )
    assert len(found) == 6, f"expected all 6 local ontologies, got {len(found)}"

    remote_keys = {(p.split("/")[1], p.split("/")[2], p.split("/")[3])
                   for p in remote_paths}
    missed_by_github = keys - remote_keys
    assert len(missed_by_github) == 4, (
        "the local walker must surface strictly more than the remote listing"
    )

    # every record is addressable on disk and carries a usable relative path
    for r in found:
        assert Path(r["path"]).is_file()
        assert r["rel_path"].endswith("/ontology.ttl")

    # deterministic ordering, so reruns diff cleanly
    assert found == sorted(
        found, key=lambda r: (r["model"], r["config"], r["scenario_id"]))


def test_driver_never_consults_list_ontology_paths(el, be, corpus, monkeypatch,
                                                   tmp_path):
    """The GitHub enumerator must not be on the local code path at all."""
    def boom(*a, **k):
        raise AssertionError("list_ontology_paths() was called by eval_local")

    monkeypatch.setattr(be, "list_ontology_paths", boom)
    monkeypatch.setattr(be, "fetch_text", boom)

    summary = el.run(outputs_root=corpus, out_dir=tmp_path / "out",
                     run_cqs=False)
    assert summary["n_files"] == 6


# ── (ii) empty and unparseable land in the denominator ────────────────────────

def test_empty_and_unparseable_are_counted_not_skipped(el, corpus, tmp_path):
    out = tmp_path / "out"
    summary = el.run(outputs_root=corpus, out_dir=out, run_cqs=True)

    # `n_skipped` is `len(selected) - len(records)` and eval_local.run() appends
    # to `records` on every iteration of the loop, so it is 0 by construction and
    # asserting it pins nothing.  Compare against the files that are actually on
    # disk instead: that is the claim ("nothing is skipped") in its testable form.
    on_disk = sorted(p.relative_to(corpus).as_posix()
                     for p in corpus.rglob("ontology.ttl") if p.is_file())
    assert len(on_disk) == 6, on_disk
    assert summary["n_files"] == 6
    assert summary["n_found"] == len(on_disk), (
        f"the walker found {summary['n_found']} ontologies but {len(on_disk)} "
        f"are on disk: {on_disk}"
    )
    assert summary["n_evaluated"] == len(on_disk), (
        "a file that exists on disk never reached the records list, so it was "
        "dropped out of every denominator"
    )

    by_status = summary["status_counts"]
    # "" -> empty ; garbage and "..." -> unparseable ; prefixes only -> empty_graph
    assert by_status.get("ok") == 2
    assert by_status.get("empty") == 1
    assert by_status.get("unparseable") == 2
    assert by_status.get("empty_graph") == 1

    # every one of the six has a per-file JSON on disk
    written = sorted(p.name for p in out.rglob("*.json") if p.name != "run_manifest.json")
    assert len(written) == 6, written

    # the SEV denominator is the file count, and the worthless files score 0.0
    rows = list(csv.DictReader((out / "summary.csv").open(encoding="utf-8")))
    assert len(rows) == 6
    assert sorted(r["rel_path"] for r in rows) == on_disk, (
        "summary.csv does not cover exactly the ontologies on disk"
    )
    zero_rows = [r for r in rows if r["status"] != "ok"]
    assert len(zero_rows) == 4
    for r in zero_rows:
        assert float(r["sev_score"]) == 0.0, (
            f"{r['model']}/{r['config']}/{r['id']} is {r['status']} but did not "
            "score 0.0 -- it was skipped out of the denominator"
        )

    agg = {(r["model"], r["config"]): r
           for r in csv.DictReader((out / "aggregate.csv").open(encoding="utf-8"))}
    pm = agg[("published-model", "cq-only")]
    assert int(pm["n_files"]) == 3
    assert int(pm["sev_denominator"]) == 3, (
        "the SEV mean must be taken over all 3 files, not just the parseable one"
    )
    # 1 good file + 2 zeros => mean is strictly below the good file's own score
    assert 0.0 < float(pm["mean_sev_score"]) < 1.0


def test_unparseable_does_not_vanish_from_the_json(el, corpus, tmp_path):
    out = tmp_path / "out"
    el.run(outputs_root=corpus, out_dir=out, run_cqs=True)
    rec = json.loads(
        (out / "published-model" / "cq-only" / "2023-134-01.json").read_text(encoding="utf-8"))
    assert rec["status"] == "unparseable"
    assert "error" in rec["ontometrics"]
    assert "error" in rec["reasoner"]
    assert rec["scores"]["sev_score"] == 0.0
    assert rec["cq_verification"]["cqs_total"] > 0
    assert rec["cq_verification"]["cqs_passed"] == 0


# ── (iii) no network ──────────────────────────────────────────────────────────

def test_no_network_call_is_made(el, be, corpus, tmp_path, monkeypatch):
    """requests, urllib and raw sockets are all booby-trapped."""
    import urllib.request

    import requests

    def boom(*a, **k):
        raise AssertionError("eval_local attempted a network call")

    monkeypatch.setattr(requests, "get", boom)
    monkeypatch.setattr(requests, "post", boom)
    monkeypatch.setattr(requests, "request", boom)
    monkeypatch.setattr(requests.Session, "request", boom)
    monkeypatch.setattr(urllib.request, "urlopen", boom)
    monkeypatch.setattr(socket, "create_connection", boom)
    monkeypatch.setattr(socket.socket, "connect", boom)
    monkeypatch.setattr(socket, "getaddrinfo", boom)

    summary = el.run(outputs_root=corpus, out_dir=tmp_path / "out", run_cqs=True)

    assert summary["n_files"] == 6
    assert summary["n_evaluated"] == 6
    assert summary["network_calls_attempted"] == 0
    assert (tmp_path / "out" / "summary.csv").is_file()

    # A run that "completes" because every file blew up inside a broad except
    # is not an offline run; it is a swallowed network error wearing a green
    # tick.  Demand real verdicts.
    assert summary["driver_errors"] == [], summary["driver_errors"]
    assert "driver_error" not in summary["status_counts"]
    assert summary["status_counts"].get("ok") == 2
    assert summary["corpus_sev_denominator"] == 6


def test_oops_is_never_attempted_over_the_web(el, corpus, tmp_path):
    out = tmp_path / "out"
    el.run(outputs_root=corpus, out_dir=out, run_cqs=False)
    rec = json.loads(
        (out / "published-model" / "cq-only" / "2023-133-01.json").read_text(encoding="utf-8"))
    assert rec["oops"]["skipped"] is True
    assert rec["oops"]["pitfalls_total"] is None, (
        "an unavailable pitfall count must be null, never 0"
    )


# ── the F3 guard: unknown is not perfect, empty is not optimal ────────────────

def test_unavailable_oops_is_null_and_excluded_from_the_score(el, corpus, tmp_path):
    out = tmp_path / "out"
    el.run(outputs_root=corpus, out_dir=out, run_cqs=False)
    rec = json.loads(
        (out / "published-model" / "cq-only" / "2023-133-01.json").read_text(encoding="utf-8"))
    scores = rec["scores"]
    assert scores["oops_component"] is None
    assert "oops" not in scores["structural_components_used"]

    # The published formula would have read the unavailable pitfall count as
    # "zero pitfalls" and paid out 1/(1+0) = 1.0 for it.  Pin the behaviour that
    # rules that out: consistency is the ONLY component, and the score is that
    # component and nothing else.
    assert scores["structural_components_used"] == ["consistency"], scores
    assert scores["structural_score"] == scores["consistency"], scores

    # ... and the exclusion has to hold where it is visible.  This file is
    # consistent, so both the correct score and the F3 score are 1.0 and the
    # equality above cannot tell them apart on its own.  Drive compute_scores
    # directly with an INCONSISTENT verdict: correct behaviour is 0.0, while
    # substituting a clean OOPS! component lifts it to mean(0.0, 1.0) = 0.5.
    inconsistent = el.compute_scores("ok", {"consistent": False},
                                     el._offline_oops(), {"skipped": True})
    assert inconsistent["oops_component"] is None, inconsistent
    assert inconsistent["structural_components_used"] == ["consistency"], inconsistent
    assert inconsistent["structural_score"] == 0.0, inconsistent

    row = [r for r in csv.DictReader((out / "summary.csv").open(encoding="utf-8"))
           if r["id"] == "2023-133-01" and r["model"] == "published-model"][0]
    assert row["oops_pitfalls_total"] == "null"
    assert row["oops_component"] == "null"


def test_empty_graph_is_not_structurally_perfect(el, corpus, tmp_path):
    """V1/F3: the empty file must not be the metric's optimum."""
    out = tmp_path / "out"
    el.run(outputs_root=corpus, out_dir=out, run_cqs=True)

    empty_graph = json.loads(
        (out / "published-model" / "scenario-cq" / "2023-135-01.json").read_text(encoding="utf-8"))
    assert empty_graph["status"] == "empty_graph"
    assert empty_graph["scores"]["consistency"] is None, (
        "a 0-triple graph is vacuously 'consistent'; recording that as 1.0 is "
        "exactly the F3 bug"
    )
    assert empty_graph["scores"]["structural_score"] is None
    assert empty_graph["scores"]["sev_score"] == 0.0

    good = json.loads(
        (out / "published-model" / "cq-only" / "2023-133-01.json").read_text(encoding="utf-8"))
    assert good["scores"]["sev_score"] > empty_graph["scores"]["sev_score"]

    # ... and it still counts against the model
    agg = {(r["model"], r["config"]): r
           for r in csv.DictReader((out / "aggregate.csv").open(encoding="utf-8"))}
    sc = agg[("published-model", "scenario-cq")]
    assert int(sc["n_files"]) == 2
    assert int(sc["structural_denominator"]) == 0
    assert sc["mean_structural_score"] == "null"
    assert float(sc["mean_sev_score"]) == 0.0


# ── it must reuse batch_evaluate, not reimplement it ──────────────────────────

def test_scoring_goes_through_batch_evaluate(el, be, corpus, tmp_path, monkeypatch):
    """Concurrent fixes to batch_evaluate must be picked up automatically."""
    real = be.compute_ontometrics
    calls = []

    def spy(g):
        calls.append(len(g))
        d = real(g)
        d["_spy"] = "batch_evaluate.compute_ontometrics"
        return d

    monkeypatch.setattr(be, "compute_ontometrics", spy)
    out = tmp_path / "out"
    el.run(outputs_root=corpus, out_dir=out, run_cqs=False)

    assert calls, "eval_local did not call batch_evaluate.compute_ontometrics"
    rec = json.loads(
        (out / "published-model" / "cq-only" / "2023-133-01.json").read_text(encoding="utf-8"))
    assert rec["ontometrics"]["_spy"] == "batch_evaluate.compute_ontometrics"


# ── it must not clobber the evidence trees ────────────────────────────────────

def test_default_out_dir_is_eval_local(el):
    assert Path(el.DEFAULT_OUT_DIR).name == "eval_local"
    for evidence in ("results", "eval_results", "odp_eval", "eval"):
        assert Path(el.DEFAULT_OUT_DIR).name != evidence


@pytest.mark.parametrize("evidence", ["results", "eval_results", "eval",
                                      "odp-platform-results", "outputs"])
def test_refuses_to_write_into_the_evidence_trees(el, evidence):
    """No --out can point the driver at a published-evidence tree.

    Checked on the path alone -- nothing is written -- so the test cannot itself
    damage the evidence it is protecting.
    """
    with pytest.raises(ValueError):
        el._assert_safe_out_dir(Path(el.REPO_ROOT) / evidence)
    el._assert_safe_out_dir(Path(el.REPO_ROOT) / "eval_local")


def test_run_writes_only_under_out_dir(el, corpus, tmp_path):
    out = tmp_path / "out"
    sentinel = tmp_path / "results"
    sentinel.mkdir()
    (sentinel / "keep.csv").write_text("evidence", encoding="utf-8")
    el.run(outputs_root=corpus, out_dir=out, run_cqs=False)
    assert (sentinel / "keep.csv").read_text(encoding="utf-8") == "evidence"
    assert sorted(p.name for p in sentinel.iterdir()) == ["keep.csv"]


# ── CLI ───────────────────────────────────────────────────────────────────────

def test_cli_runs_offline_end_to_end(corpus, tmp_path):
    out = tmp_path / "cli-out"
    proc = subprocess.run(
        [sys.executable, str(EVAL_LOCAL_PATH),
         "--outputs", str(corpus), "--out", str(out), "--no-cq"],
        cwd=str(REPO_ROOT), capture_output=True, text=True, timeout=600,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert (out / "summary.csv").is_file()
    assert (out / "aggregate.csv").is_file()
    manifest = json.loads((out / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["n_files"] == 6
    assert manifest["source"] == "local filesystem"
