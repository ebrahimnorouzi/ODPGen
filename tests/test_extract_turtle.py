"""
Regression tests for scripts/run_generation.py :: extract_turtle().

Background
----------
The generation script used to select the Turtle block with

    re.search(r"```(?:turtle|ttl|rdf)\\s*(.*?)```", ...)

i.e. the FIRST fenced span in the response.  Every prompt template ends with a
format instruction that itself contains a literal ```turtle ... ``` example,
and the models dutifully echo that line before answering.  The echo is
therefore the first fenced span, so the extractor captured the three
characters "..." and never reached the real ontology below it.  That produced
44 three-byte ontology.ttl files for gpt-5.4 and 26 for bloomz.

These tests pin the fixed behaviour: pick, among ALL fenced blocks, the
longest one that actually looks like Turtle; never return the "..." echo.

No network and no model is involved: the fixture is a real recorded response
committed under outputs/.
"""

import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "run_generation.py"

# The exact real-world failure: gpt-5.4 echoed the format instruction, and the
# committed ontology.ttl for this scenario is 3 bytes ("...").
REGRESSION_RESPONSE = (
    REPO_ROOT
    / "outputs"
    / "gpt-5.4"
    / "scenario-cq-constraints"
    / "2023-133-01"
    / "raw_response.txt"
)

FENCE = "```"


def _load_module():
    """Import scripts/run_generation.py without needing it to be a package."""
    spec = importlib.util.spec_from_file_location("run_generation", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules["run_generation"] = module
    spec.loader.exec_module(module)
    return module


rg = _load_module()


# ---------------------------------------------------------------------------
# 1. The real-world regression
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def echoed_instruction_response() -> str:
    assert REGRESSION_RESPONSE.is_file(), f"missing fixture: {REGRESSION_RESPONSE}"
    return REGRESSION_RESPONSE.read_text(encoding="utf-8")


def test_fixture_really_contains_the_echoed_instruction(echoed_instruction_response):
    """Guard the premise: the fixture must still start with the echoed fence."""
    first_line = echoed_instruction_response.splitlines()[0]
    assert FENCE + "turtle ... " + FENCE in first_line
    # ... and the genuine ontology must live in a later, separate fence.
    assert echoed_instruction_response.count(FENCE) == 4


def test_echoed_instruction_does_not_win(echoed_instruction_response):
    """The old extractor returned '...'. The fixed one must not."""
    ttl = rg.extract_turtle(echoed_instruction_response)
    assert ttl != "..."
    assert not ttl.startswith("...")
    assert len(ttl) > 1000


def test_real_ontology_is_recovered(echoed_instruction_response):
    ttl = rg.extract_turtle(echoed_instruction_response)
    assert ttl.startswith("@prefix owl:")
    assert ":CausalRelation a owl:Class" in ttl
    assert ttl.rstrip().endswith("rdfs:subClassOf :Event .")
    # No stray fence markers leaked into the extracted ontology.
    assert FENCE not in ttl
    # The trailing validation-report prose must be left behind.
    assert "PASS" not in ttl


def test_recovered_ontology_parses_to_59_triples(echoed_instruction_response):
    rdflib = pytest.importorskip("rdflib")
    graph = rdflib.Graph()
    graph.parse(data=rg.extract_turtle(echoed_instruction_response), format="turtle")
    assert len(graph) == 59


def test_committed_ontology_ttl_is_the_broken_three_byte_artifact():
    """
    Documents the damage this fix addresses.  Recovering the files on disk is
    a separate task; this test only records that the file is still the 3-byte
    '...' and will need rewriting.  It is skipped once that has happened.
    """
    committed = REGRESSION_RESPONSE.with_name("ontology.ttl")
    if not committed.is_file():
        pytest.skip("ontology.ttl not present")
    if committed.read_text(encoding="utf-8").strip() != "...":
        pytest.skip("ontology.ttl has already been recovered")
    assert committed.stat().st_size < 50


# ---------------------------------------------------------------------------
# 2. A single well-formed turtle fence
# ---------------------------------------------------------------------------

SIMPLE_TURTLE = """@prefix : <http://example.org/odp#> .
@prefix owl: <http://www.w3.org/2002/07/owl#> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .

:Event a owl:Class ;
  rdfs:label "Event" .

:causes a owl:ObjectProperty ;
  rdfs:domain :Event ;
  rdfs:range :Event ."""


def test_single_well_formed_fence():
    response = (
        "Here is the ontology design pattern you asked for.\n\n"
        + FENCE + "turtle\n"
        + SIMPLE_TURTLE + "\n"
        + FENCE + "\n\n"
        "Let me know if you would like the mapping tables as well.\n"
    )
    assert rg.extract_turtle(response) == SIMPLE_TURTLE


@pytest.mark.parametrize("info", ["turtle", "ttl", "rdf", "TURTLE", ""])
def test_fence_language_tags_all_accepted(info):
    response = FENCE + info + "\n" + SIMPLE_TURTLE + "\n" + FENCE + "\n"
    assert rg.extract_turtle(response) == SIMPLE_TURTLE


# ---------------------------------------------------------------------------
# 3. No fence at all — raw turtle
# ---------------------------------------------------------------------------

def test_raw_turtle_without_any_fence():
    assert rg.extract_turtle(SIMPLE_TURTLE) == SIMPLE_TURTLE


def test_raw_turtle_after_a_prose_preamble():
    response = "Sure! Here is the pattern:\n\n" + SIMPLE_TURTLE
    assert rg.extract_turtle(response) == SIMPLE_TURTLE


# ---------------------------------------------------------------------------
# 4. Truncated response — fence opened, never closed
# ---------------------------------------------------------------------------

TRUNCATED_RESPONSE = (
    "1. The complete Turtle ontology wrapped in a "
    + FENCE + "turtle ... " + FENCE + " code block.\n\n"
    + FENCE + "turtle\n"
    "@prefix : <http://example.org/odp#> .\n"
    "@prefix owl: <http://www.w3.org/2002/07/owl#> .\n"
    "\n"
    ":Event a owl:Class ;\n"
    '  rdfs:label "Ev'
)


def test_truncated_unterminated_fence_still_yields_the_partial_ontology():
    ttl = rg.extract_turtle(TRUNCATED_RESPONSE)
    assert ttl.startswith("@prefix :")
    assert ":Event a owl:Class ;" in ttl
    assert ttl != "..."
    assert FENCE not in ttl


def test_truncation_is_detectable():
    assert rg.has_unterminated_fence(TRUNCATED_RESPONSE) is True
    signals = rg.truncation_signals(TRUNCATED_RESPONSE)
    assert "unterminated_code_fence" in signals


def test_complete_response_is_not_flagged_as_truncated():
    response = FENCE + "turtle\n" + SIMPLE_TURTLE + "\n" + FENCE + "\n"
    assert rg.has_unterminated_fence(response) is False
    assert rg.truncation_signals(response, "stop") == []


@pytest.mark.parametrize(
    "finish_reason",
    ["length", "MAX_TOKENS", "FinishReason.MAX_TOKENS", "incomplete:max_output_tokens"],
)
def test_length_finish_reasons_are_recorded_as_truncation(finish_reason):
    response = FENCE + "turtle\n" + SIMPLE_TURTLE + "\n" + FENCE + "\n"
    signals = rg.truncation_signals(response, finish_reason)
    assert signals == [f"finish_reason={finish_reason}"]


# ---------------------------------------------------------------------------
# 5. Echoed instruction only — no real ontology anywhere
# ---------------------------------------------------------------------------

ECHO_ONLY_RESPONSE = (
    "1. The complete Turtle ontology wrapped in a "
    + FENCE + "turtle ... " + FENCE + " code block.\n"
    "2. A scenario-requirement-to-axiom mapping table.\n"
    "3. A CQ-to-axiom mapping table.\n"
    "4. A validation report.\n\n"
    "I am unable to produce the ontology for this scenario.\n"
)


def test_echo_only_response_yields_empty_not_dots():
    ttl = rg.extract_turtle(ECHO_ONLY_RESPONSE)
    assert ttl != "..."
    assert ttl == ""


def test_pure_prose_yields_empty():
    assert rg.extract_turtle("I'm sorry, I cannot help with that request.") == ""


def test_empty_response_yields_empty():
    assert rg.extract_turtle("") == ""


# ---------------------------------------------------------------------------
# 6. Block selection details
# ---------------------------------------------------------------------------

def test_longest_qualifying_block_wins_over_a_short_one():
    short = "@prefix : <http://example.org/odp#> ."
    response = (
        "First, the prefixes:\n\n"
        + FENCE + "turtle\n" + short + "\n" + FENCE + "\n\n"
        "And here is the full pattern:\n\n"
        + FENCE + "turtle\n" + SIMPLE_TURTLE + "\n" + FENCE + "\n"
    )
    assert rg.extract_turtle(response) == SIMPLE_TURTLE


def test_non_turtle_blocks_are_ignored_however_long():
    long_table = "\n".join(f"| requirement {i} | axiom {i} |" for i in range(200))
    response = (
        FENCE + "markdown\n" + long_table + "\n" + FENCE + "\n\n"
        + FENCE + "turtle\n" + SIMPLE_TURTLE + "\n" + FENCE + "\n"
    )
    assert rg.extract_turtle(response) == SIMPLE_TURTLE


def test_sparql_block_is_not_preferred_over_a_turtle_block():
    """A long SPARQL query can carry ' a owl:' too; the turtle fence still wins."""
    patterns = "\n".join(f"  ?e{i} a owl:Class ; :causes ?f{i} ." for i in range(20))
    sparql = "PREFIX : <http://example.org/odp#>\nSELECT * WHERE {\n" + patterns + "\n}"
    # Guard the premise: without the language preference the query would win
    # on length alone.
    assert rg.looks_like_turtle(sparql)
    assert len(sparql) > len(SIMPLE_TURTLE)
    response = (
        FENCE + "sparql\n" + sparql + "\n" + FENCE + "\n\n"
        + FENCE + "turtle\n" + SIMPLE_TURTLE + "\n" + FENCE + "\n"
    )
    assert rg.extract_turtle(response) == SIMPLE_TURTLE


def test_find_code_blocks_reports_closed_state():
    blocks = rg.find_code_blocks(TRUNCATED_RESPONSE)
    assert [b.closed for b in blocks] == [True, False]
    assert blocks[0].body.strip() == "..."
    assert blocks[1].info == "turtle"


def test_looks_like_turtle_rejects_the_echo():
    assert rg.looks_like_turtle("...") is False
    assert rg.looks_like_turtle("@prefix : <http://example.org/> .") is True
    assert rg.looks_like_turtle(":Event a owl:Class .") is True


# ---------------------------------------------------------------------------
# 7. Token budget (F2)
# ---------------------------------------------------------------------------

def test_default_token_budget_is_no_longer_1024():
    assert rg.DEFAULT_MAX_NEW_TOKENS >= 4096
    assert rg.RECOMMENDED_MIN_NEW_TOKENS >= 4096


def test_cli_default_matches_the_constant_and_stays_overridable():
    """--help is enough to prove the wiring; it never contacts a model."""
    proc = subprocess.run(
        [sys.executable, str(SCRIPT_PATH), "--help"],
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
    )
    assert proc.returncode == 0, proc.stderr
    help_text = " ".join(proc.stdout.split())  # undo argparse line wrapping
    assert "--max-new-tokens N" in help_text
    assert f"default: {rg.DEFAULT_MAX_NEW_TOKENS}" in help_text


# ---------------------------------------------------------------------------
# 8. Block selection: the FIRST strictly-tagged Turtle fence wins
#
#    Regression (b).  The rule "longest block carrying a Turtle signature
#    wins" is activated by the raised token budget: with 4096 tokens the model
#    now has room to emit the CQ-to-axiom mapping table it was always asked
#    for, and that table quotes Turtle in its Axiom column, so it carries a
#    Turtle signature and is far longer than the ODP itself.  The prompts
#    instruct the model to output the ontology FIRST; the table is prose about
#    the ontology, not the ontology.
# ---------------------------------------------------------------------------

SHORT_ODP = """@prefix : <http://example.org/odp#> .
@prefix owl: <http://www.w3.org/2002/07/owl#> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .

:Event a owl:Class ;
  rdfs:label "Event" .

:Participant a owl:Class ;
  rdfs:label "Participant" .

:hasParticipant a owl:ObjectProperty ;
  rdfs:domain :Event ;
  rdfs:range :Participant ."""


def _cq_axiom_mapping_table(rows: int = 40) -> str:
    """A markdown table whose Axiom column quotes Turtle, as the prompts ask."""
    header = "| # | Competency question | Axiom |\n|---|---|---|"
    body = "\n".join(
        f"| {i} | Which participants take part in event {i}? "
        f'| `:C{i} a owl:Class ; rdfs:label "c{i}" .` and '
        f"`:p{i} a owl:ObjectProperty ; rdfs:domain :C{i} ; rdfs:range :C{i} .` |"
        for i in range(rows)
    )
    return header + "\n" + body


ODP_THEN_LONGER_UNTAGGED_TABLE = (
    "Here is the ontology, followed by the CQ-to-axiom mapping table.\n\n"
    + FENCE + "turtle\n" + SHORT_ODP + "\n" + FENCE + "\n\n"
    "### CQ-to-axiom mapping\n\n"
    + FENCE + "\n" + _cq_axiom_mapping_table() + "\n" + FENCE + "\n"
)


def test_premise_the_untagged_table_is_longer_and_looks_like_turtle():
    """Guard the premise of the regression: the table really would win on length."""
    blocks = rg.find_code_blocks(ODP_THEN_LONGER_UNTAGGED_TABLE)
    assert [b.info for b in blocks] == ["turtle", ""]
    assert all(b.closed for b in blocks)
    assert rg.looks_like_turtle(blocks[1].body)
    assert len(blocks[1].body.strip()) > len(blocks[0].body.strip())


def test_first_strictly_tagged_turtle_fence_beats_a_longer_untagged_table():
    ttl = rg.extract_turtle(ODP_THEN_LONGER_UNTAGGED_TABLE)
    assert "Competency question" not in ttl
    assert "|" not in ttl
    assert ttl == SHORT_ODP


def test_untagged_turtle_is_still_used_when_no_strict_tag_exists():
    """The strict preference must not throw away a genuinely untagged ontology."""
    response = FENCE + "\n" + SHORT_ODP + "\n" + FENCE + "\n"
    assert rg.extract_turtle(response) == SHORT_ODP


def test_a_prefix_only_fence_does_not_shadow_the_real_ontology():
    """
    'First' means first block that actually declares something.  A fence
    holding nothing but @prefix lines is a preamble, not an ontology.
    """
    response = (
        FENCE + "turtle\n@prefix : <http://example.org/odp#> .\n" + FENCE + "\n\n"
        + FENCE + "turtle\n" + SHORT_ODP + "\n" + FENCE + "\n"
    )
    assert rg.extract_turtle(response) == SHORT_ODP


# ---------------------------------------------------------------------------
# 9. Truncation must never be silently repaired into a clean ontology
#
#    Regression (c).  A reply cut off mid-axiom at 'rdfs:range :Ev' reaches
#    maybe_fix_common_turtle_issues, which drops the incomplete trailing
#    statement, and the result parses perfectly.  A truncated generation then
#    becomes indistinguishable from a complete one and is scored as if whole.
# ---------------------------------------------------------------------------

TRUNCATED_MID_AXIOM = (
    "Here is the ontology design pattern.\n\n"
    + FENCE + "turtle\n"
    "@prefix : <http://example.org/odp#> .\n"
    "@prefix owl: <http://www.w3.org/2002/07/owl#> .\n"
    "@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .\n"
    "\n"
    ":Event a owl:Class ;\n"
    '  rdfs:label "Event" .\n'
    "\n"
    ":causes a owl:ObjectProperty ;\n"
    "  rdfs:domain :Event ;\n"
    "  rdfs:range :Ev"
)


def test_premise_the_truncated_tail_is_repaired_into_clean_turtle():
    """State the danger as a measured fact, so the flagging below has a point."""
    rdflib = pytest.importorskip("rdflib")
    repaired = rg.maybe_fix_common_turtle_issues(rg.extract_turtle(TRUNCATED_MID_AXIOM))
    graph = rdflib.Graph()
    graph.parse(data=repaired, format="turtle")  # parses cleanly -- looks complete
    assert len(graph) > 0
    assert "rdfs:range :Ev" not in repaired


def test_repair_reports_the_incomplete_statement_it_dropped():
    _, repairs = rg.maybe_fix_common_turtle_issues(
        rg.extract_turtle(TRUNCATED_MID_AXIOM), report=True
    )
    assert "dropped_incomplete_trailing_statement" in repairs


def test_repair_reports_nothing_for_complete_turtle():
    fixed, repairs = rg.maybe_fix_common_turtle_issues(SIMPLE_TURTLE, report=True)
    assert repairs == []
    assert fixed.strip() == SIMPLE_TURTLE.strip()


def test_extract_ontology_flags_a_truncated_generation():
    result = rg.extract_ontology(
        TRUNCATED_MID_AXIOM, finish_reason="stop", apply_repairs=True
    )
    assert result.truncated is True
    assert "unterminated_code_fence" in result.truncation_signals
    assert "repair_dropped_incomplete_statement" in result.truncation_signals
    assert result.empty is False


def test_extract_ontology_leaves_a_complete_generation_unflagged():
    response = FENCE + "turtle\n" + SIMPLE_TURTLE + "\n" + FENCE + "\n"
    result = rg.extract_ontology(response, finish_reason="stop", apply_repairs=True)
    assert result.truncated is False
    assert result.truncation_signals == []
    assert result.repairs == []
    assert result.turtle == SIMPLE_TURTLE


def test_truncated_ontology_carries_a_visible_marker():
    result = rg.extract_ontology(
        TRUNCATED_MID_AXIOM, finish_reason="length", apply_repairs=True
    )
    annotated = rg.annotate_truncated_turtle(result.turtle, result.truncation_signals)
    assert annotated.splitlines()[0].startswith("# ODPGEN-TRUNCATED")
    assert "finish_reason=length" in annotated
    assert "repair_dropped_incomplete_statement" in annotated
    # The marker is a Turtle comment, so the file still parses -- it is the
    # metadata, not the syntax, that tells the scorer to discount it.
    rdflib = pytest.importorskip("rdflib")
    rdflib.Graph().parse(data=annotated, format="turtle")


def test_complete_ontology_is_not_annotated():
    assert rg.annotate_truncated_turtle(SIMPLE_TURTLE, []) == SIMPLE_TURTLE


# ---------------------------------------------------------------------------
# 9b. End-to-end: main() must write the truncation verdict to metadata.json
# ---------------------------------------------------------------------------

def _minimal_workspace(tmp_path: Path) -> Path:
    (tmp_path / "data").mkdir(parents=True, exist_ok=True)
    (tmp_path / "data" / "scenarios.json").write_text(
        json.dumps(
            [
                {
                    "scenario_id": "2023-999-01",
                    "scenario_text": "A test scenario.",
                    "cq_list": ["What happened?"],
                }
            ]
        ),
        encoding="utf-8",
    )
    prompts = tmp_path / "prompts"
    prompts.mkdir(parents=True, exist_ok=True)
    (prompts / "scenario_only.txt").write_text(
        "{{SCENARIO_TEXT}}\n{{CQ_LIST}}\n", encoding="utf-8"
    )
    return tmp_path


def _run_main(tmp_path, monkeypatch, response_text, finish_reason):
    """Drive main() with the network backend replaced. No API call is made."""
    _minimal_workspace(tmp_path)
    monkeypatch.setattr(
        rg,
        "generate_with_openai",
        lambda **kwargs: rg.GenerationResult(response_text, finish_reason),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_generation.py",
            "--backend", "openai",
            "--model", "stub-model",
            "--config", "scenario-only",
            "--data", str(tmp_path / "data" / "scenarios.json"),
            "--prompts-dir", str(tmp_path / "prompts"),
            "--outputs-dir", str(tmp_path / "outputs"),
            "--fix-common-turtle-issues",
        ],
    )
    rg.main()
    out = tmp_path / "outputs" / "stub-model" / "scenario-only" / "2023-999-01"
    meta = json.loads((out / "metadata.json").read_text(encoding="utf-8"))
    return out, meta


def test_main_records_truncation_for_a_cut_off_generation(tmp_path, monkeypatch):
    out, meta = _run_main(
        tmp_path, monkeypatch, TRUNCATED_MID_AXIOM, "incomplete:max_output_tokens"
    )
    assert meta["truncated"] is True
    assert "finish_reason=incomplete:max_output_tokens" in meta["truncation_signals"]
    assert "unterminated_code_fence" in meta["truncation_signals"]
    assert "repair_dropped_incomplete_statement" in meta["truncation_signals"]
    assert meta["ontology_repairs"] == ["dropped_incomplete_trailing_statement"]
    ttl = (out / "ontology.ttl").read_text(encoding="utf-8")
    assert ttl.startswith("# ODPGEN-TRUNCATED")


def test_main_leaves_a_complete_generation_unmarked(tmp_path, monkeypatch):
    response = FENCE + "turtle\n" + SIMPLE_TURTLE + "\n" + FENCE + "\n"
    out, meta = _run_main(tmp_path, monkeypatch, response, "completed")
    assert meta["truncated"] is False
    assert meta["truncation_signals"] == []
    assert meta["ontology_repairs"] == []
    ttl = (out / "ontology.ttl").read_text(encoding="utf-8")
    assert not ttl.startswith("#")
    assert ttl.strip() == SIMPLE_TURTLE


# ---------------------------------------------------------------------------
# 10. The documented batch driver must not pin the token budget (V2 / F2)
# ---------------------------------------------------------------------------

DRIVER = REPO_ROOT / "run_all_experiments.sh"

# The driver is executed for real, but with a fake ``python3`` first on PATH:
# it records the argv it was handed and exits.  No model, no API, no network --
# whatever the driver decides to run, nothing can actually generate.
_PYTHON3_SHIM = """#!/usr/bin/env bash
printf '%s\\n' "$*" >> "$CMDLOG"
exit 0
"""


def _driver_command_lines(tmp_path, extra_env=None) -> list[str]:
    """Invoke the real driver and return the command lines it resolved."""
    bash = shutil.which("bash")
    if bash is None:
        pytest.skip("bash is not available on this machine")

    bindir = tmp_path / "bin"
    bindir.mkdir(parents=True, exist_ok=True)
    shim = bindir / "python3"
    shim.write_bytes(_PYTHON3_SHIM.encode("utf-8"))
    shim.chmod(0o755)

    cmdlog = tmp_path / "cmdlog.txt"

    env = dict(os.environ)
    # A real generation must be impossible even if every other guard fails.
    for key in ("OPENAI_API_KEY", "HF_TOKEN", "GEMINI_API_KEY", "MAX_NEW_TOKENS"):
        env.pop(key, None)
    env["PATH"] = str(bindir) + os.pathsep + env.get("PATH", "")
    env["CMDLOG"] = str(cmdlog)
    env["ECHO_COMMANDS"] = "1"
    env["LOG_DIR"] = str(tmp_path / "logs")
    if extra_env:
        env.update(extra_env)

    proc = subprocess.run(
        [bash, str(DRIVER)],
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
        env=env,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr

    lines = [l.strip() for l in proc.stdout.splitlines() if "run_generation.py" in l]
    if cmdlog.is_file():
        lines += [
            l.strip()
            for l in cmdlog.read_text(encoding="utf-8").splitlines()
            if "run_generation.py" in l
        ]
    return lines


def test_driver_resolves_command_lines_we_can_inspect(tmp_path):
    cmds = _driver_command_lines(tmp_path)
    assert cmds, "the driver produced no inspectable command line at all"
    assert any("bigscience/bloomz-7b1" in c for c in cmds)


def test_driver_does_not_pin_max_new_tokens(tmp_path):
    """
    V2: the driver pinned MAX_NEW_TOKENS=1024 and passed --max-new-tokens
    unconditionally, so run_generation.DEFAULT_MAX_NEW_TOKENS (4096) was
    unreachable through the repository's own documented entry point.
    """
    cmds = _driver_command_lines(tmp_path)
    assert cmds
    offenders = [c for c in cmds if "--max-new-tokens" in c]
    assert offenders == [], offenders


def test_driver_still_honours_an_explicit_override(tmp_path):
    cmds = _driver_command_lines(tmp_path, {"MAX_NEW_TOKENS": "8192"})
    assert cmds
    assert all("--max-new-tokens 8192" in c for c in cmds), cmds


def test_driver_source_has_no_live_1024_default():
    """
    Only executable lines matter -- the comment explaining the old 1024 cap is
    documentation, and deleting the explanation is how the bug comes back.
    """
    code = [
        line
        for line in DRIVER.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    offenders = [line for line in code if "1024" in line]
    assert offenders == [], offenders
