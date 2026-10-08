"""
Tests for the LOCAL evaluation pipeline: outputs/ -> scored records, offline.

Three defects are pinned here.

D1 -- batch_evaluate.main() could only enumerate the corpus over
     https://api.github.com/repos/<repo>/git/trees/<branch> (list_ontology_paths)
     and fetch each file over raw.githubusercontent.com.  A file that exists only
     in the working tree -- a recovered output, a regenerated output, the result
     of any repair -- could never be scored, and every artefact was pinned to a
     remote branch rather than to the tree the reader has checked out.
     `--local-root PATH` walks `<root>/**/ontology.ttl` from disk instead.

D2 -- scripts/eval_local.py existed but was referenced by neither README.md nor
     run_all_experiments.sh, so nobody would find it.  Both must now document a
     command that goes from outputs/ to a scored CSV.

D3 -- run_generation.py records a `truncated` flag and `truncation_signals` in
     metadata.json and prepends a `# ODPGEN-TRUNCATED` banner to the ontology,
     and 127 of the 420 recorded responses carry a truncation signal.  Nothing
     downstream read any of it, so a generation that was cut off mid-axiom was
     scored as though the model had chosen to stop there.  The evaluation must
     now carry the verdict into every per-artefact record and every aggregate --
     and must NOT drop truncated artefacts from the denominators, only report
     them separately.

Every test in this file runs with the network booby-trapped: `requests`,
`urllib` and the raw socket layer all raise on use.  A test that needs the
network to pass is a test that has not fixed D1.
"""

from __future__ import annotations

import csv
import importlib.util
import json
import socket
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
BATCH_EVALUATE_PATH = REPO_ROOT / "batch_evaluate.py"
README_PATH = REPO_ROOT / "README.md"
RUNNER_PATH = REPO_ROOT / "run_all_experiments.sh"
EVAL_LOCAL_REL = "scripts/eval_local.py"
ONTOLOGY = "ontology.ttl"


# ── module under test ─────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def be():
    """Import batch_evaluate.py from the working tree."""
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    spec = importlib.util.spec_from_file_location("batch_evaluate_under_test",
                                                  BATCH_EVALUATE_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ── the network booby-trap ────────────────────────────────────────────────────

class NetworkUsed(AssertionError):
    """Raised by the booby-trap when anything tries to reach the network."""


@pytest.fixture
def no_network(monkeypatch, be):
    """Make every outbound call explode, and record that it was attempted.

    This is the load-bearing fixture of the whole file: the point of --local-root
    is that the corpus comes off disk, so a run that touches requests, urllib or
    a socket has not fixed D1 no matter what it writes.
    """
    attempts: list[str] = []

    def boom(where):
        def _boom(*a, **k):
            attempts.append(f"{where}{a[:1]}")
            raise NetworkUsed(
                f"network used ({where}); the local pipeline must read from disk"
            )
        return _boom

    import urllib.request

    import requests

    for name in ("get", "post", "put", "head", "request"):
        monkeypatch.setattr(requests, name, boom(f"requests.{name}"),
                            raising=False)
    monkeypatch.setattr(requests.Session, "request",
                        boom("requests.Session.request"), raising=False)
    # batch_evaluate holds its own reference to the module; patch that too so a
    # `from requests import get`-style binding cannot slip past.
    monkeypatch.setattr(be, "requests", requests, raising=False)
    monkeypatch.setattr(urllib.request, "urlopen", boom("urllib.urlopen"))
    monkeypatch.setattr(socket.socket, "connect", boom("socket.connect"))
    monkeypatch.setattr(socket.socket, "connect_ex", boom("socket.connect_ex"))
    monkeypatch.setattr(socket, "create_connection", boom("socket.create_connection"))
    return attempts


# ── corpus builders ───────────────────────────────────────────────────────────

COMPLETE_TTL = """@prefix : <http://example.org/odp#> .
@prefix owl: <http://www.w3.org/2002/07/owl#> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .

:Sensor a owl:Class ; rdfs:label "Sensor" .
:Observation a owl:Class ; rdfs:label "Observation" .
:Feature a owl:Class ; rdfs:label "Feature" .
:madeObservation a owl:ObjectProperty ;
    rdfs:domain :Sensor ; rdfs:range :Observation .
:observedFeature a owl:ObjectProperty ;
    rdfs:domain :Observation ; rdfs:range :Feature .
:Observation rdfs:subClassOf :Feature .
"""

# Same content, but the generation that produced it ran out of output budget.
TRUNCATED_TTL_BODY = """@prefix : <http://example.org/odp#> .
@prefix owl: <http://www.w3.org/2002/07/owl#> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .

:Sensor a owl:Class ; rdfs:label "Sensor" .
:Observation a owl:Class ; rdfs:label "Observation" .
:Feature a owl:Class ; rdfs:label "Feature" .
:madeObservation a owl:ObjectProperty ;
    rdfs:domain :Sensor ; rdfs:range :Observation .
:observedFeature a owl:ObjectProperty ;
    rdfs:domain :Observation ; rdfs:range :Feature .
"""

TRUNCATION_BANNER = (
    "# ODPGEN-TRUNCATED: this generation was cut off before the model finished.\n"
    "# ODPGEN-TRUNCATION-SIGNALS: finish_reason=length, unterminated_code_fence\n"
    "# Do NOT score this file as a complete ontology; see metadata.json.\n"
)


def _write_artifact(root: Path, model: str, config: str, scenario: str,
                    ttl: str, metadata: dict | None = None,
                    raw_response: str | None = None) -> Path:
    d = root / model / config / scenario
    d.mkdir(parents=True, exist_ok=True)
    (d / "ontology.ttl").write_text(ttl, encoding="utf-8")
    if metadata is not None:
        (d / "metadata.json").write_text(json.dumps(metadata, indent=2),
                                         encoding="utf-8")
    if raw_response is not None:
        (d / "raw_response.txt").write_text(raw_response, encoding="utf-8")
    return d


@pytest.fixture
def corpus(tmp_path: Path) -> Path:
    """A four-artefact local tree that exists ONLY on disk.

    Nothing here is on any remote branch, which is exactly the situation D1
    made unmeasurable.
    """
    root = tmp_path / "outputs"

    # 1. complete, and metadata says so.
    _write_artifact(
        root, "modelA", "cq-only", "2023-133-01", COMPLETE_TTL,
        metadata={"model": "modelA", "config": "cq-only",
                  "scenario_id": "2023-133-01", "finish_reason": "stop",
                  "truncated": False, "truncation_signals": []},
        raw_response="```turtle\n" + COMPLETE_TTL + "```\n",
    )

    # 2. truncated, and metadata says so.
    _write_artifact(
        root, "modelA", "cq-only", "2023-133-02",
        TRUNCATION_BANNER + TRUNCATED_TTL_BODY,
        metadata={"model": "modelA", "config": "cq-only",
                  "scenario_id": "2023-133-02", "finish_reason": "length",
                  "truncated": True,
                  "truncation_signals": ["finish_reason=length",
                                         "unterminated_code_fence"]},
        raw_response="```turtle\n" + TRUNCATED_TTL_BODY,
    )

    # 3. LEGACY artefact: metadata predates the truncation flag entirely (this
    #    is what all 420 recorded metadata.json files look like), but the raw
    #    response still carries the unterminated-fence tell.
    _write_artifact(
        root, "modelB", "scenario-cq", "2023-134-01", TRUNCATED_TTL_BODY,
        metadata={"model": "modelB", "config": "scenario-cq",
                  "scenario_id": "2023-134-01", "max_new_tokens": 1024},
        raw_response="```turtle\n" + TRUNCATED_TTL_BODY,
    )

    # 4. LEGACY artefact that is genuinely complete: closed fence, no tell.
    _write_artifact(
        root, "modelB", "scenario-cq", "2023-134-02", COMPLETE_TTL,
        metadata={"model": "modelB", "config": "scenario-cq",
                  "scenario_id": "2023-134-02", "max_new_tokens": 1024},
        raw_response="```turtle\n" + COMPLETE_TTL + "```\n",
    )
    return root


def _run_local(be, corpus: Path, out: Path, extra: list[str] | None = None):
    argv = ["batch_evaluate.py", "--local-root", str(corpus),
            "--out", str(out), "--no-cq", "--workers", "1"]
    argv.extend(extra or [])
    old = sys.argv
    sys.argv = argv
    try:
        be.main()
    finally:
        sys.argv = old


def _records(out: Path) -> dict:
    return {p.relative_to(out).as_posix(): json.loads(p.read_text(encoding="utf-8"))
            for p in sorted(out.rglob("*.json")) if p.name != "run_manifest.json"}


def _csv_rows(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


# ── D1: the local walker ──────────────────────────────────────────────────────

def test_local_walker_exists(be):
    """batch_evaluate must expose a local enumerator, not only the GitHub one."""
    assert hasattr(be, "list_local_ontology_paths"), (
        "batch_evaluate.list_local_ontology_paths is missing: the corpus can "
        "still only be enumerated over the GitHub API (D1)."
    )
    assert hasattr(be, "list_ontology_paths"), (
        "the remote enumerator was deleted; D1 says keep it working."
    )


def test_local_walker_finds_files_the_github_path_would_miss(be, corpus, no_network):
    """The walker sees on-disk artefacts; the GitHub path cannot even be called."""
    found = be.list_local_ontology_paths(corpus)
    assert len(found) == 4, f"expected 4 local ontologies, got {found}"
    tails = {Path(p).parent.name for p in found}
    assert tails == {"2023-133-01", "2023-133-02", "2023-134-01", "2023-134-02"}

    # The remote enumerator, on the same corpus, gets nothing -- it explodes.
    with pytest.raises(Exception):
        be.list_ontology_paths()
    assert no_network, "the remote path was expected to attempt a network call"


def test_local_run_scores_everything_with_zero_network_calls(be, corpus, tmp_path,
                                                             no_network):
    """A full --local-root run completes offline and scores every artefact."""
    out = tmp_path / "odp_eval_local"
    _run_local(be, corpus, out)

    recs = _records(out)
    assert len(recs) == 4, f"expected 4 scored records, got {sorted(recs)}"
    assert no_network == [], f"the local run touched the network: {no_network}"

    for key, rec in recs.items():
        assert rec.get("source") == "local", f"{key} is not marked local: {rec}"
        assert "structural" in rec, f"{key} has no structural score"
        assert rec["structural"]["counted"] is True


def test_local_run_scores_a_file_that_is_not_in_the_repository(be, tmp_path,
                                                               no_network):
    """The regenerated-output case: a brand-new artefact, on disk only."""
    root = tmp_path / "outputs"
    _write_artifact(root, "regenerated", "scenario-cq", "9999-999-99",
                    COMPLETE_TTL,
                    metadata={"truncated": False, "truncation_signals": []})
    out = tmp_path / "odp_eval_local"
    _run_local(be, root, out)

    scored = out / "regenerated" / "scenario-cq" / "9999-999-99.json"
    assert scored.exists(), (
        "an artefact that exists only in the working tree was not scored; "
        "the corpus is still pinned to the remote branch (D1)."
    )
    rec = json.loads(scored.read_text(encoding="utf-8"))
    assert rec["model"] == "regenerated"
    assert rec["config"] == "scenario-cq"
    assert rec["id"] == "9999-999-99"
    assert rec["structural"]["structural_score"] > 0.0


def test_remote_path_is_still_the_default(be, monkeypatch, tmp_path):
    """D1 says keep the remote path: with no --local-root it must still be used."""
    called = {"n": 0}

    def fake_list():
        called["n"] += 1
        return []

    monkeypatch.setattr(be, "list_ontology_paths", fake_list)
    out = tmp_path / "odp_eval_remote"
    old = sys.argv
    sys.argv = ["batch_evaluate.py", "--out", str(out), "--no-cq", "--workers", "1"]
    try:
        be.main()
    finally:
        sys.argv = old
    assert called["n"] == 1, "the remote enumerator was not used without --local-root"


# ── D3: truncation is read and carried ────────────────────────────────────────

def test_truncated_artifact_is_distinguishable_from_a_complete_one(be, corpus,
                                                                   tmp_path,
                                                                   no_network):
    """The core D3 property: the record says which artefacts were cut off."""
    out = tmp_path / "odp_eval_local"
    _run_local(be, corpus, out)

    complete = json.loads(
        (out / "modelA" / "cq-only" / "2023-133-01.json").read_text(encoding="utf-8"))
    truncated = json.loads(
        (out / "modelA" / "cq-only" / "2023-133-02.json").read_text(encoding="utf-8"))

    assert truncated["truncated"] is True, (
        "a generation that ran out of output budget is scored as a complete "
        "answer: nothing reads metadata.json's truncated flag (D3)."
    )
    assert complete["truncated"] is False
    assert truncated["truncation"]["source"] == "metadata"
    assert "finish_reason=length" in truncated["truncation"]["signals"]
    assert complete["truncation"]["signals"] == []


def test_legacy_truncation_is_derived_when_metadata_predates_the_flag(be, corpus,
                                                                     tmp_path,
                                                                     no_network):
    """All 420 recorded metadata.json files predate the flag; the tell survives.

    modelB/2023-134-01 has an unterminated code fence in raw_response.txt and no
    `truncated` key in metadata.json.  It must still come out truncated, and its
    sibling with a closed fence must not.
    """
    out = tmp_path / "odp_eval_local"
    _run_local(be, corpus, out)

    legacy_cut = json.loads(
        (out / "modelB" / "scenario-cq" / "2023-134-01.json").read_text(encoding="utf-8"))
    legacy_ok = json.loads(
        (out / "modelB" / "scenario-cq" / "2023-134-02.json").read_text(encoding="utf-8"))

    assert legacy_cut["truncated"] is True, (
        "a legacy artefact whose metadata predates the truncated flag is read "
        "as complete; the raw_response.txt tell is never consulted (D3)."
    )
    assert legacy_cut["truncation"]["source"].startswith("derived")
    assert "unterminated_code_fence" in legacy_cut["truncation"]["signals"]
    assert legacy_ok["truncated"] is False


def test_truncation_banner_alone_is_enough(be, tmp_path, no_network):
    """The banner travels with the artefact; metadata may be gone entirely."""
    root = tmp_path / "outputs"
    _write_artifact(root, "modelC", "cq-only", "2023-133-01",
                    TRUNCATION_BANNER + TRUNCATED_TTL_BODY)  # no metadata, no raw
    out = tmp_path / "odp_eval_local"
    _run_local(be, root, out)

    rec = json.loads(
        (out / "modelC" / "cq-only" / "2023-133-01.json").read_text(encoding="utf-8"))
    assert rec["truncated"] is True
    assert rec["truncation"]["source"] == "ontology_banner"


def test_truncation_is_unknown_not_false_when_there_is_no_evidence(be, tmp_path,
                                                                   no_network):
    """No metadata, no banner, no raw response -> unknown, never a clean bill."""
    root = tmp_path / "outputs"
    _write_artifact(root, "modelD", "cq-only", "2023-133-01", COMPLETE_TTL)
    out = tmp_path / "odp_eval_local"
    _run_local(be, root, out)

    rec = json.loads(
        (out / "modelD" / "cq-only" / "2023-133-01.json").read_text(encoding="utf-8"))
    assert rec["truncated"] is None, (
        "an artefact with no truncation evidence at all was reported as "
        "complete; unknown must not be laundered into False."
    )
    assert rec["truncation"]["note"]


# ── D3: truncation reaches the aggregates, without leaving the denominator ────

def test_per_artifact_csv_carries_the_truncation_columns(be, corpus, tmp_path,
                                                         no_network):
    out = tmp_path / "odp_eval_local"
    _run_local(be, corpus, out)

    rows = _csv_rows(out / "summary.csv")
    assert len(rows) == 4
    assert "truncated" in rows[0], f"summary.csv columns: {sorted(rows[0])}"
    assert "truncation_source" in rows[0]
    by_id = {r["id"]: r for r in rows}
    assert by_id["2023-133-02"]["truncated"] == "True"
    assert by_id["2023-133-01"]["truncated"] == "False"


def test_aggregate_reports_truncation_and_keeps_it_in_the_denominator(be, corpus,
                                                                      tmp_path,
                                                                      no_network):
    """Truncated artefacts are surfaced, NOT silently dropped from n_files."""
    out = tmp_path / "odp_eval_local"
    _run_local(be, corpus, out)

    rows = _csv_rows(out / "aggregate.csv")
    by_group = {(r["model"], r["config"]): r for r in rows}
    a = by_group[("modelA", "cq-only")]

    assert int(a["n_files"]) == 2, (
        "truncated artefacts were dropped from the denominator; D3 says "
        "surface them, do not hide them."
    )
    assert int(a["n_truncated"]) == 1
    assert int(a["n_complete"]) == 1
    # both a corpus-wide mean and a complete-only mean, each with its own stated
    # denominator, so a reader can see exactly what a headline number covers.
    assert a["mean_structural_score"] not in ("", None)
    assert int(a["structural_denominator"]) == 2
    assert int(a["structural_denominator_complete"]) == 1

    manifest = json.loads((out / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["n_truncated"] == 2
    assert manifest["n_complete"] == 2
    assert manifest["n_files"] == 4
    assert manifest["network_calls"] == 0


def test_unknown_truncation_is_counted_separately(be, tmp_path, no_network):
    root = tmp_path / "outputs"
    _write_artifact(root, "modelD", "cq-only", "2023-133-01", COMPLETE_TTL)
    out = tmp_path / "odp_eval_local"
    _run_local(be, root, out)

    rows = _csv_rows(out / "aggregate.csv")
    assert int(rows[0]["n_truncation_unknown"]) == 1
    assert int(rows[0]["n_files"]) == 1


# ── D3 on the real corpus ─────────────────────────────────────────────────────

@pytest.mark.skipif(not (REPO_ROOT / "outputs").is_dir(),
                    reason="outputs/ not present")
def test_real_corpus_truncation_verdict_is_reachable(be):
    """The recorded corpus: 420 artefacts, 127 of them carrying a signal.

    Was 160 until 2026-09-23, when the 33 truncated gpt-5.4 cells were
    regenerated at a 16384-token budget and came back complete.
    160 - 33 = 127; the arithmetic is the check that nothing else moved.

    This is the number D3 quotes.  If reading the corpus yields 0 truncated,
    the flag is being ignored again.
    """
    paths = be.list_local_ontology_paths(REPO_ROOT / "outputs")
    assert len(paths) == 420, f"expected 420 local ontologies, got {len(paths)}"
    n_trunc = 0
    for p in paths:
        verdict = be.read_truncation(p)
        if verdict["truncated"]:
            n_trunc += 1
    assert n_trunc == 127, (
        f"expected 127 truncated artefacts in the recorded corpus, got {n_trunc}"
    )


# ── the documented command must actually run ──────────────────────────────────

def test_documented_local_command_survives_a_non_utf8_console(tmp_path):
    """`--local-root` must not die on the progress line it prints.

    The per-file progress tag contains U+2713 (the reasoner-consistent tick).
    On a console whose encoding cannot represent it -- cp1252, the Windows
    default -- the print raises UnicodeEncodeError and takes the whole run down
    partway through, after some artefacts have been scored and before any CSV is
    written.  The documented workflow must not depend on the terminal's codepage.
    """
    root = tmp_path / "outputs"
    _write_artifact(root, "modelA", "cq-only", "2023-133-01", COMPLETE_TTL,
                    metadata={"truncated": False, "truncation_signals": []})
    out = tmp_path / "odp_eval_local"

    import os
    import subprocess

    env = dict(os.environ, PYTHONIOENCODING="cp1252")
    proc = subprocess.run(
        [sys.executable, str(BATCH_EVALUATE_PATH),
         "--local-root", str(root), "--out", str(out), "--no-cq", "--workers", "1"],
        cwd=str(REPO_ROOT), env=env, capture_output=True, text=True,
        encoding="utf-8", errors="replace",
    )
    assert proc.returncode == 0, (
        "the documented --local-root command crashed on a cp1252 console:\n"
        f"{proc.stdout}\n{proc.stderr}"
    )
    assert (out / "summary.csv").exists(), "the run died before writing its CSV"


# ── D2: the local path is documented and wired into the runner ────────────────

def test_readme_documents_the_local_evaluation_command():
    text = README_PATH.read_text(encoding="utf-8")
    assert EVAL_LOCAL_REL in text, (
        "README.md never mentions scripts/eval_local.py, so nobody would find "
        "the offline local evaluator (D2)."
    )
    assert "--local-root" in text, (
        "README.md does not document batch_evaluate.py --local-root, so the "
        "documented workflow is still the GitHub one (D1)."
    )
    assert "eval_local/summary.csv" in text or "summary.csv" in text, (
        "README.md does not say where the scored CSV lands (D2)."
    )


def test_runner_wires_in_the_local_evaluation():
    text = RUNNER_PATH.read_text(encoding="utf-8")
    assert EVAL_LOCAL_REL in text, (
        "run_all_experiments.sh never invokes scripts/eval_local.py (D2)."
    )
    assert "--local-root" in text, (
        "run_all_experiments.sh does not run the local batch evaluation (D1)."
    )


# ── D4: a repair pass must not destroy the scored corpus ──────────────────────
#
# `--local-root` exists so an output repaired in the working tree can be scored
# (D1).  The repair loop is: regenerate or recover the damaged artefacts, then
# re-score just those with `--rerun-failed`.  That second pass wrote
# summary.csv, aggregate.csv and run_manifest.json from ONLY the artefacts it
# had re-scored, silently replacing the full-corpus CSVs.  A no-op repair pass
# (nothing left to fix) reduced them to a bare header and a manifest claiming a
# corpus of zero files, while the per-artefact JSONs sat untouched on disk.

BROKEN_TTL = """@prefix : <http://example.org/odp#> .
:Sensor a owl:Class ;
"""


def _fresh_corpus(root: Path) -> Path:
    """Three good artefacts plus one that does not parse."""
    meta = {"truncated": False, "truncation_signals": []}
    for i in (1, 2, 3):
        _write_artifact(root, "modelA", "cq-only", f"2023-133-0{i}",
                        COMPLETE_TTL, metadata=dict(meta))
    _write_artifact(root, "modelA", "cq-only", "2023-133-09", BROKEN_TTL,
                    metadata=dict(meta))
    return root


def test_a_repair_pass_keeps_the_whole_scored_corpus_in_the_csv(be, tmp_path,
                                                                no_network):
    """Repair one artefact on disk, re-score only it, keep all four rows."""
    root = _fresh_corpus(tmp_path / "outputs")
    out = tmp_path / "odp_eval_local"

    _run_local(be, root, out)
    assert len(_csv_rows(out / "summary.csv")) == 4

    # The D1 use case: the damaged output is repaired in the WORKING TREE and
    # re-scored.  Nothing else changed, so nothing else needs re-scoring.
    (root / "modelA" / "cq-only" / "2023-133-09" / ONTOLOGY).write_text(
        COMPLETE_TTL, encoding="utf-8")
    _run_local(be, root, out, ["--rerun-failed"])

    rows = _csv_rows(out / "summary.csv")
    assert len(rows) == 4, (
        "a --rerun-failed repair pass replaced the full-corpus summary.csv "
        f"with only the artefacts it re-scored: {[r['id'] for r in rows]}"
    )
    by_id = {r["id"]: r for r in rows}
    assert by_id["2023-133-09"]["status"] == "ok", (
        "the repaired artefact was not re-scored")
    assert by_id["2023-133-09"]["record_origin"] == "this_run"
    assert by_id["2023-133-01"]["record_origin"] == "carried_over", (
        "a row that survived a partial pass must say it was carried over, so "
        "no reader mistakes a stale score for a fresh one")

    manifest = json.loads((out / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["n_files"] == 4, (
        f"run_manifest.json reports a corpus of {manifest['n_files']} after a "
        "one-artefact repair pass")
    assert manifest["n_evaluated_this_run"] == 1
    assert manifest["n_carried_over"] == 3

    agg = _csv_rows(out / "aggregate.csv")
    assert len(agg) == 1
    assert int(agg[0]["n_files"]) == 4


def test_a_no_op_repair_pass_does_not_empty_the_csv(be, corpus, tmp_path,
                                                    no_network):
    """Nothing left to repair must mean nothing changes -- not an empty CSV.

    This is the destructive case: every artefact already scores clean, so
    --rerun-failed selects zero files, and the scored corpus was overwritten
    with a header-only summary.csv and a manifest saying n_files: 0.
    """
    out = tmp_path / "odp_eval_local"
    _run_local(be, corpus, out)
    _run_local(be, corpus, out, ["--rerun-failed"])

    rows = _csv_rows(out / "summary.csv")
    assert len(rows) == 4, (
        f"a no-op repair pass wiped summary.csv down to {len(rows)} rows while "
        "the per-artefact JSONs were still on disk")

    manifest = json.loads((out / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["n_files"] == 4
    assert manifest["n_evaluated_this_run"] == 0
    assert manifest["n_carried_over"] == 4
    # D3 must survive the carry-over: the truncation verdict is part of the
    # record, so a re-read record still knows it was cut off.
    assert manifest["n_truncated"] == 2
    assert manifest["n_complete"] == 2
    assert int(_csv_rows(out / "aggregate.csv")[0]["n_truncated"]) == 1


# ── D5: an explicitly requested service must survive the offline guard ────────
#
# `--local-root` arms a socket guard so "the corpus came off disk" is measured
# rather than asserted.  The guard refused EVERY connection, including the OOPS!
# endpoint the user had just asked for with --oops-url.  It raised
# NetworkAccessDuringLocalRun, which is not a requests.RequestException, so it
# tore straight through run_oops_scan and out of evaluate_one: every artefact in
# the corpus came back as a bare {"error": ...} record with no ontometrics, no
# reasoner verdict, no structural score and no truncation verdict -- and the run
# still exited 0.  A flag combination that silently destroys the whole result
# set is worse than either flag alone.

def _closed_loopback_port() -> int:
    """A loopback port with nothing listening on it."""
    s = socket.socket()
    try:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])
    finally:
        s.close()


def test_local_run_with_an_oops_url_still_scores_every_artifact(be, tmp_path):
    """--local-root --oops-url must score the corpus, not annihilate it.

    Deliberately NOT using the no_network fixture: the point is the guard's own
    behaviour.  The endpoint is a closed loopback port, so the connection is
    attempted and refused by the OS -- a transport error OOPS can report -- and
    nothing leaves the machine.
    """
    root = tmp_path / "outputs"
    _write_artifact(root, "modelA", "cq-only", "2023-133-01", COMPLETE_TTL,
                    metadata={"truncated": False, "truncation_signals": []})
    out = tmp_path / "odp_eval_local"
    url = f"http://127.0.0.1:{_closed_loopback_port()}/OOPS/rest"

    _run_local(be, root, out, ["--oops-url", url])

    rec = json.loads(
        (out / "modelA" / "cq-only" / "2023-133-01.json").read_text(encoding="utf-8"))
    assert "error" not in rec, (
        "the offline guard refused the OOPS! endpoint the run had explicitly "
        f"been given, and took the whole artefact down with it: {rec.get('error')}")
    assert rec["structural"]["structural_score"] > 0.0, (
        "the artefact lost its structural score to the guard")
    assert rec["truncated"] is False
    assert "error" in (rec.get("oops") or {}), (
        "a closed port should surface as an OOPS transport error, not as a scan")

    manifest = json.loads((out / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["n_files"] == 1
    # The invariant that matters is unchanged: nothing UNEXPECTED was dialled.
    assert manifest["network_calls"] == 0, (
        f"an unexpected connection was attempted: {manifest}")
    assert manifest["network_allowed_calls"] >= 1, (
        "the run never even tried the endpoint it was given")
    assert url in " ".join(manifest["network_allowlist"]) or any(
        "127.0.0.1" in a for a in manifest["network_allowlist"]), (
        "the manifest does not record which endpoint was allowed through")


def test_the_guard_still_refuses_an_endpoint_that_was_never_requested(be, tmp_path):
    """Allowing the requested endpoint must not disarm the guard generally."""
    with be.network_guard(True, allow=["http://127.0.0.1:9/OOPS/rest"]) as guard:
        s = socket.socket()
        try:
            with pytest.raises(be.NetworkAccessDuringLocalRun):
                s.connect(("93.184.216.34", 80))
        finally:
            s.close()
    assert guard.n_attempts == 1
