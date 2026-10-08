"""
Tests for scripts/recover_outputs.py.

Every test builds its own synthetic outputs/ tree under tmp_path.  The real
outputs/ corpus is never read or written by this module.

No test performs a network call.  recover_outputs itself never calls an API --
it only re-parses raw_response.txt files that are already on disk -- so there is
nothing to mock; the tests simply assert that no such machinery is reachable.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import recover_outputs as ro  # noqa: E402


# ---------------------------------------------------------------------------
# Fixture material
# ---------------------------------------------------------------------------

#: The three characters the F1 bug wrote to disk.
DAMAGED_TTL = "..."

#: The prompt boilerplate models echo back.  It contains an inline turtle fence,
#: which is precisely the span re.search grabbed first (F1).
PROMPT_ECHO = (
    "1. The complete Turtle ontology wrapped in a ```turtle ... ``` code block. "
    "Output this FIRST, before any explanation.\n\n"
)

GOOD_TURTLE = """@prefix owl:  <http://www.w3.org/2002/07/owl#> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
@prefix :     <http://example.org/odp#> .

:Ontology a owl:Ontology ;
    rdfs:label "Test ODP" .

:Patient a owl:Class ;
    rdfs:label "Patient" .

:Symptom a owl:Class ;
    rdfs:label "Symptom" .

:experiencesSymptom a owl:ObjectProperty ;
    rdfs:domain :Patient ;
    rdfs:range :Symptom ."""

#: A complete reply: prompt echo, then a properly closed turtle fence, then the
#: mapping-table prose the prompts also ask for.
RECOVERABLE_RESPONSE = (
    PROMPT_ECHO
    + "```turtle\n"
    + GOOD_TURTLE
    + "\n```\n\n"
    + "2. Scenario-requirement-to-axiom mapping\n\n"
    + "| Requirement | Axiom |\n| --- | --- |\n| R1 | :Patient a owl:Class |\n"
)

#: F2 damage: the reply runs out of tokens inside the fence, so the closing
#: fence is never emitted.
TRUNCATED_RESPONSE = (
    PROMPT_ECHO
    + "```turtle\n"
    + GOOD_TURTLE
    + "\n\n:hasMedicationIntake a owl:ObjectProperty ;\n"
    + "    rdfs:comment \"Relates a patient to a medication intake\" ;\n"
    + "    rdfs:subPropertyOf :"
)

#: A weak model that never produced Turtle at all (bloomz-style).
NO_TURTLE_RESPONSE = "- What is the scope of the policy?"


def make_scenario(
    outputs_root: Path,
    model: str,
    config: str,
    scenario_id: str,
    ontology: str,
    raw_response: str = None,
) -> Path:
    """Create one outputs/{model}/{config}/{scenario_id}/ directory."""
    scenario_dir = outputs_root / model / config / scenario_id
    scenario_dir.mkdir(parents=True, exist_ok=True)
    if ontology is not None:
        (scenario_dir / "ontology.ttl").write_text(ontology, encoding="utf-8", newline="\n")
    if raw_response is not None:
        (scenario_dir / "raw_response.txt").write_text(
            raw_response, encoding="utf-8", newline="\n"
        )
    (scenario_dir / "prompt.txt").write_text("prompt", encoding="utf-8", newline="\n")
    (scenario_dir / "metadata.json").write_text(
        json.dumps({"model": model, "config": config, "scenario_id": scenario_id}),
        encoding="utf-8",
    )
    return scenario_dir


@pytest.fixture
def corpus(tmp_path: Path) -> Path:
    """
    A miniature corpus holding one of each interesting case.

    outputs/
      good-model/cfg-a/S-healthy      -> intact ontology, must be left alone
      broken-model/cfg-a/S-recover    -> 3-byte F1 damage, closed fence in raw
      broken-model/cfg-a/S-truncated  -> 3-byte F1 damage, F2 unterminated fence
      broken-model/cfg-b/S-noturtle   -> 3-byte F1 damage, raw has no Turtle
      broken-model/cfg-b/S-missing    -> ontology.ttl absent, closed fence in raw
    """
    outputs_root = tmp_path / "outputs"
    make_scenario(outputs_root, "good-model", "cfg-a", "S-healthy", GOOD_TURTLE,
                  RECOVERABLE_RESPONSE)
    make_scenario(outputs_root, "broken-model", "cfg-a", "S-recover", DAMAGED_TTL,
                  RECOVERABLE_RESPONSE)
    make_scenario(outputs_root, "broken-model", "cfg-a", "S-truncated", DAMAGED_TTL,
                  TRUNCATED_RESPONSE)
    make_scenario(outputs_root, "broken-model", "cfg-b", "S-noturtle", DAMAGED_TTL,
                  NO_TURTLE_RESPONSE)
    make_scenario(outputs_root, "broken-model", "cfg-b", "S-missing", None,
                  RECOVERABLE_RESPONSE)
    return outputs_root


def snapshot(root: Path) -> dict:
    """Path -> sha256 for every file under *root*, for exact change detection."""
    return {
        str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(root.rglob("*"))
        if p.is_file()
    }


def by_scenario(report: dict, key: str) -> dict:
    return {r["scenario_id"]: r for r in report[key]}


# ---------------------------------------------------------------------------
# The extractor is imported, not copied
# ---------------------------------------------------------------------------

def test_extract_turtle_is_imported_from_run_generation():
    """
    The whole point of importing rather than copying is that the two can never
    drift.  Assert we really are reaching into run_generation.py.
    """
    fn = ro.load_extract_turtle()
    assert callable(fn)
    assert Path(fn.__code__.co_filename).resolve() == (SCRIPTS_DIR / "run_generation.py").resolve()


def test_composed_extractor_returns_the_longest_fenced_turtle_block():
    """
    The corrected behaviour, asserted end-to-end on the composed extractor so
    the test holds both before and after run_generation.extract_turtle is fixed.

    The response opens with the prompt echo containing an inline turtle fence --
    the exact span the F1 bug returned "..." from.
    """
    result = ro.Extractor().extract(RECOVERABLE_RESPONSE)
    assert result["ok"] is True
    assert result["text"].startswith("@prefix owl:")
    assert ":experiencesSymptom" in result["text"]
    assert "..." not in result["text"]
    assert result["triples"] > 0


def test_fallback_extractor_ignores_prompt_echo_and_picks_longest_block():
    """The local corrected rule, tested directly."""
    response = (
        PROMPT_ECHO
        + "```turtle\n@prefix : <http://example.org/> .\n```\n\n"
        + "```turtle\n"
        + GOOD_TURTLE
        + "\n```\n"
    )
    extracted = ro.fallback_extract_turtle(response)
    assert extracted.strip() == GOOD_TURTLE.strip()


def test_fallback_extractor_returns_nothing_for_unterminated_fence():
    assert ro.fallback_extract_turtle(TRUNCATED_RESPONSE) == ""


def test_response_is_truncated_detects_f2_signature():
    assert ro.response_is_truncated(TRUNCATED_RESPONSE) is True
    assert ro.response_is_truncated(RECOVERABLE_RESPONSE) is False
    assert ro.response_is_truncated(NO_TURTLE_RESPONSE) is False


def test_truncated_response_never_yields_a_candidate():
    """
    run_generation.extract_turtle has an unfenced fallback that walks the raw
    text to end-of-buffer, and maybe_fix_common_turtle_issues will happily trim
    the dangling final statement.  Together they would turn an F2-truncated
    reply into an ontology that *looks* complete.  A recovery tool must refuse.
    """
    result = ro.Extractor().extract(TRUNCATED_RESPONSE)
    assert result["ok"] is False
    assert result["text"] == ""


def test_extraction_is_restricted_to_closed_fences():
    """
    When closed Turtle fences exist, the accepted text must be one of them
    verbatim -- never a splice reaching past the closing fence.
    """
    response = (
        PROMPT_ECHO
        + "```turtle\n"
        + GOOD_TURTLE
        + "\n```\n\n"
        + "Notes: @prefix trailing prose that must not be spliced in.\n"
    )
    result = ro.Extractor().extract(response)
    assert result["ok"] is True
    assert result["text"] == GOOD_TURTLE.strip()
    assert "trailing prose" not in result["text"]


def _buggy_first_match_extractor(response: str) -> str:
    """The pre-fix run_generation.extract_turtle, reproduced for the fallback test."""
    import re

    match = re.search(
        r"```(?:turtle|ttl|rdf)\s*(.*?)```", response, flags=re.IGNORECASE | re.DOTALL
    )
    return match.group(1).strip() if match else ""


def test_fallback_covers_a_still_buggy_upstream_extractor():
    """
    If run_generation.extract_turtle is (or reverts to) the first-match version,
    it returns the "..." of the echoed instruction.  The local corrected
    extractor must still pull out the real ontology, and the report must name it
    so the stale upstream is visible.
    """
    assert _buggy_first_match_extractor(RECOVERABLE_RESPONSE) == "..."

    extractor = ro.Extractor(primary=_buggy_first_match_extractor)
    result = extractor.extract(RECOVERABLE_RESPONSE)
    assert result["ok"] is True
    assert result["extractor"] == ro.EXTRACTOR_FALLBACK
    assert result["text"].startswith("@prefix owl:")


def test_no_fallback_leaves_a_buggy_upstream_unrescued():
    extractor = ro.Extractor(primary=_buggy_first_match_extractor, use_fallback=False)
    assert extractor.extract(RECOVERABLE_RESPONSE)["ok"] is False


def test_looks_like_turtle_rejects_prose_bullets():
    assert not ro.looks_like_turtle("- Use clear, self-explanatory class names")
    assert not ro.looks_like_turtle(":Patient and :Symptom are the key concepts")
    assert ro.looks_like_turtle("@prefix : <http://example.org/> .")
    assert ro.looks_like_turtle(":Patient a owl:Class .")


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def test_validate_turtle_rejects_empty_graph():
    """F3: an empty graph parses cleanly and must never be written."""
    ok, n, err = ro.validate_turtle("@prefix : <http://example.org/> .")
    assert ok is False
    assert n == 0
    assert "empty" in err.lower()


def test_validate_turtle_rejects_broken_syntax():
    ok, n, err = ro.validate_turtle("@prefix : <http://example.org/> .\n:A a owl:Class")
    assert ok is False
    assert err


def test_validate_turtle_accepts_real_ontology():
    ok, n, err = ro.validate_turtle(GOOD_TURTLE)
    assert ok is True
    assert n > 0
    assert err == ""


# ---------------------------------------------------------------------------
# Recoverable case
# ---------------------------------------------------------------------------

def test_apply_recovers_damaged_ontology(corpus: Path):
    report = ro.run_recovery(corpus, apply=True)

    recovered = by_scenario(report, "recovered")
    assert set(recovered) == {"S-recover", "S-missing"}

    target = corpus / "broken-model" / "cfg-a" / "S-recover" / "ontology.ttl"
    text = target.read_text(encoding="utf-8")
    assert text.startswith("@prefix owl:")
    assert ":experiencesSymptom" in text

    ok, n_triples, err = ro.validate_turtle(text)
    assert ok is True, err
    assert n_triples == recovered["S-recover"]["recovered_triples"]


def test_apply_preserves_original_as_orig_empty(corpus: Path):
    ro.run_recovery(corpus, apply=True)
    scenario_dir = corpus / "broken-model" / "cfg-a" / "S-recover"
    backup = scenario_dir / "ontology.ttl.orig-empty"
    assert backup.exists()
    assert backup.read_text(encoding="utf-8") == DAMAGED_TTL


def test_existing_orig_empty_is_never_overwritten(corpus: Path):
    scenario_dir = corpus / "broken-model" / "cfg-a" / "S-recover"
    backup = scenario_dir / "ontology.ttl.orig-empty"
    backup.write_text("PRE-EXISTING PROVENANCE", encoding="utf-8", newline="\n")

    ro.run_recovery(corpus, apply=True)

    assert backup.read_text(encoding="utf-8") == "PRE-EXISTING PROVENANCE"
    assert (scenario_dir / "ontology.ttl").read_text(encoding="utf-8").startswith("@prefix")


def test_healthy_ontology_is_left_untouched(corpus: Path):
    healthy = corpus / "good-model" / "cfg-a" / "S-healthy" / "ontology.ttl"
    before = healthy.read_bytes()
    report = ro.run_recovery(corpus, apply=True)
    assert healthy.read_bytes() == before
    assert not (healthy.parent / "ontology.ttl.orig-empty").exists()
    assert by_scenario(report, "recovered").get("S-healthy") is None
    assert report["by_model_config"]["good-model"]["cfg-a"]["healthy"] == 1


def test_missing_ontology_file_is_recovered(corpus: Path):
    report = ro.run_recovery(corpus, apply=True)
    rec = by_scenario(report, "recovered")["S-missing"]
    assert rec["original_state"] == ro.STATE_MISSING
    written = corpus / "broken-model" / "cfg-b" / "S-missing" / "ontology.ttl"
    assert ro.validate_turtle(written.read_text(encoding="utf-8"))[0] is True


# ---------------------------------------------------------------------------
# Unrecoverable cases
# ---------------------------------------------------------------------------

def test_truncated_response_is_unrecoverable_and_not_written(corpus: Path):
    report = ro.run_recovery(corpus, apply=True)

    unrec = by_scenario(report, "unrecoverable")
    assert "S-truncated" in unrec
    assert unrec["S-truncated"]["reason"] == ro.REASON_UNTERMINATED
    assert "max_new_tokens" in unrec["S-truncated"]["detail"]

    scenario_dir = corpus / "broken-model" / "cfg-a" / "S-truncated"
    # The damaged 3-byte file is left exactly as it was; nothing is fabricated.
    assert (scenario_dir / "ontology.ttl").read_text(encoding="utf-8") == DAMAGED_TTL
    assert not (scenario_dir / "ontology.ttl.orig-empty").exists()


def test_response_without_turtle_is_unrecoverable(corpus: Path):
    report = ro.run_recovery(corpus, apply=True)
    unrec = by_scenario(report, "unrecoverable")
    assert unrec["S-noturtle"]["reason"] == ro.REASON_NO_TURTLE
    scenario_dir = corpus / "broken-model" / "cfg-b" / "S-noturtle"
    assert (scenario_dir / "ontology.ttl").read_text(encoding="utf-8") == DAMAGED_TTL
    assert not (scenario_dir / "ontology.ttl.orig-empty").exists()


def test_missing_raw_response_is_unrecoverable(tmp_path: Path):
    outputs_root = tmp_path / "outputs"
    make_scenario(outputs_root, "m", "c", "S", DAMAGED_TTL, raw_response=None)
    report = ro.run_recovery(outputs_root, apply=True)
    assert by_scenario(report, "unrecoverable")["S"]["reason"] == ro.REASON_NO_RAW


def test_unparseable_candidate_is_never_written(tmp_path: Path):
    """A fenced block that rdflib refuses must not reach disk."""
    outputs_root = tmp_path / "outputs"
    broken_block = (
        "@prefix owl: <http://www.w3.org/2002/07/owl#> .\n"
        "@prefix : <http://example.org/odp#> .\n"
        ":Patient a owl:Class ;;; rdfs:label \n"
        ":Symptom a owl:Class\n"
    )
    raw = PROMPT_ECHO + "```turtle\n" + broken_block + "```\n"
    scenario_dir = make_scenario(outputs_root, "m", "c", "S", DAMAGED_TTL, raw)

    report = ro.run_recovery(outputs_root, apply=True)

    assert by_scenario(report, "recovered") == {}
    assert "S" in by_scenario(report, "unrecoverable")
    assert (scenario_dir / "ontology.ttl").read_text(encoding="utf-8") == DAMAGED_TTL


def test_large_unparseable_ontology_is_not_touched_when_nothing_better_exists(tmp_path: Path):
    """
    A big-but-broken ontology.ttl is a model failure, not F1 damage.  It must be
    reported separately and left alone rather than being replaced by garbage.
    """
    outputs_root = tmp_path / "outputs"
    junk = "<!DOCTYPE html>\n<html><body>not an ontology at all, but long</body></html>\n"
    scenario_dir = make_scenario(outputs_root, "m", "c", "S", junk, NO_TURTLE_RESPONSE)

    report = ro.run_recovery(outputs_root, apply=True)

    assert report["totals"]["damaged"] == 0
    assert report["totals"]["skipped_unparseable"] == 1
    assert (scenario_dir / "ontology.ttl").read_text(encoding="utf-8") == junk
    assert not (scenario_dir / "ontology.ttl.orig-empty").exists()


# ---------------------------------------------------------------------------
# Dry run writes nothing
# ---------------------------------------------------------------------------

def test_dry_run_writes_nothing(corpus: Path):
    before = snapshot(corpus)
    report = ro.run_recovery(corpus, apply=False)
    after = snapshot(corpus)

    assert after == before
    assert report["dry_run"] is True
    assert report["applied"] is False
    # It still *reports* what it would have done.
    assert report["totals"]["recovered"] == 2
    assert all(r["written"] is False for r in report["recovered"])
    assert not list(corpus.rglob("*.orig-empty"))


def test_cli_defaults_to_dry_run(corpus: Path, tmp_path: Path, capsys):
    before = snapshot(corpus)
    report_path = tmp_path / "report.json"

    exit_code = ro.main(["--outputs", str(corpus), "--report", str(report_path)])

    assert exit_code == 0
    assert snapshot(corpus) == before
    assert "DRY RUN" in capsys.readouterr().out

    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["dry_run"] is True
    assert report["totals"]["recovered"] == 2


def test_cli_apply_writes(corpus: Path, tmp_path: Path):
    report_path = tmp_path / "report.json"
    exit_code = ro.main(
        ["--outputs", str(corpus), "--report", str(report_path), "--apply", "--quiet"]
    )
    assert exit_code == 0
    recovered = corpus / "broken-model" / "cfg-a" / "S-recover" / "ontology.ttl"
    assert recovered.read_text(encoding="utf-8").startswith("@prefix")
    assert json.loads(report_path.read_text(encoding="utf-8"))["applied"] is True


def test_cli_no_report_writes_no_report(corpus: Path, tmp_path: Path):
    report_path = tmp_path / "report.json"
    ro.main(
        ["--outputs", str(corpus), "--report", str(report_path), "--no-report", "--quiet"]
    )
    assert not report_path.exists()


# ---------------------------------------------------------------------------
# Idempotency
# ---------------------------------------------------------------------------

def test_second_apply_changes_nothing(corpus: Path):
    first = ro.run_recovery(corpus, apply=True)
    after_first = snapshot(corpus)

    second = ro.run_recovery(corpus, apply=True)
    after_second = snapshot(corpus)

    assert after_second == after_first, "second --apply run must be a no-op"
    assert first["totals"]["recovered"] == 2
    assert second["totals"]["recovered"] == 0
    assert second["totals"]["healthy"] == first["totals"]["healthy"] + 2
    # The unrecoverable ones stay unrecoverable, with the same reasons.
    assert {r["scenario_id"]: r["reason"] for r in second["unrecoverable"]} == {
        r["scenario_id"]: r["reason"] for r in first["unrecoverable"]
    }


def test_dry_run_after_apply_reports_clean(corpus: Path):
    ro.run_recovery(corpus, apply=True)
    report = ro.run_recovery(corpus, apply=False)
    assert report["totals"]["recovered"] == 0
    assert report["recovered"] == []


# ---------------------------------------------------------------------------
# Report shape
# ---------------------------------------------------------------------------

def test_report_has_per_model_config_counts_and_reasons(corpus: Path):
    report = ro.run_recovery(corpus, apply=False)

    cfg_a = report["by_model_config"]["broken-model"]["cfg-a"]
    assert cfg_a["scanned"] == 2
    assert cfg_a["damaged"] == 2
    assert cfg_a["recovered"] == 1
    assert cfg_a["unrecoverable"] == 1
    assert cfg_a["reasons"] == {ro.REASON_UNTERMINATED: 1}

    cfg_b = report["by_model_config"]["broken-model"]["cfg-b"]
    assert cfg_b["damaged"] == 2
    assert cfg_b["recovered"] == 1
    assert cfg_b["reasons"] == {ro.REASON_NO_TURTLE: 1}

    totals = report["totals"]
    assert totals["scanned"] == 5
    assert totals["damaged"] == 4
    assert totals["recovered"] == 2
    assert totals["unrecoverable"] == 2

    for rec in report["unrecoverable"]:
        assert rec["reason"]
        assert rec["detail"]


def test_report_is_json_serialisable(corpus: Path, tmp_path: Path):
    report = ro.run_recovery(corpus, apply=False)
    path = tmp_path / "nested" / "report.json"
    ro.write_report(report, path)
    assert json.loads(path.read_text(encoding="utf-8"))["tool"].endswith(
        "recover_outputs.py"
    )


def test_report_records_which_extractor_was_used(corpus: Path):
    report = ro.run_recovery(corpus, apply=False)
    usage = report["extractor"]["usage"]
    assert sum(usage.values()) == 2
    assert set(usage) <= {ro.EXTRACTOR_IMPORTED, ro.EXTRACTOR_FALLBACK}
    assert report["extractor"]["import_ok"] is True


# ---------------------------------------------------------------------------
# No API calls
# ---------------------------------------------------------------------------

def test_module_imports_no_llm_clients():
    """Recovery is a pure disk operation; no provider SDK may be reachable."""
    source = (SCRIPTS_DIR / "recover_outputs.py").read_text(encoding="utf-8")
    for forbidden in ("import openai", "from openai", "import anthropic",
                      "google.generativeai", "huggingface_hub", "requests.post"):
        assert forbidden not in source


def test_no_network_during_recovery(corpus: Path, monkeypatch):
    """Hard-fail the socket layer, then run a full apply pass."""
    import socket

    def _blocked(*args, **kwargs):
        raise AssertionError("recover_outputs must not open a network connection")

    monkeypatch.setattr(socket.socket, "connect", _blocked)
    monkeypatch.setattr(socket, "create_connection", _blocked)

    report = ro.run_recovery(corpus, apply=True)
    assert report["totals"]["recovered"] == 2
