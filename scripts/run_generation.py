#!/usr/bin/env python3
import argparse
import gc
import hashlib
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import NamedTuple

# ---------------------------------------------------------------------------
# Generation-length defaults
# ---------------------------------------------------------------------------

# Every prompt template in prompts/ asks for FOUR deliverables: the Turtle
# ontology plus three mapping / validation tables.  A complete answer for a
# mid-sized ODP is ~1500-2500 tokens of Turtle alone, and the tables add
# roughly as much again.  The previous default of 1024 tokens cut verbose
# models off mid-axiom, so the closing fence was never emitted and the recorded
# ontology was a fragment -- or, combined with the fence-selection bug fixed
# below, nothing at all.  4096 is the floor at which the four deliverables fit
# for the terser models; reasoning / thinking models need more, which
# --max-new-tokens supplies on the CLI.
DEFAULT_MAX_NEW_TOKENS = 4096

# Below this we warn loudly and flag the run in metadata: any generation made
# with a smaller budget is at high risk of silent truncation.
RECOMMENDED_MIN_NEW_TOKENS = 4096

# Finish-reason substrings that mean "the model ran out of output budget".
_LENGTH_FINISH_MARKERS = (
    "length",
    "max_tokens",
    "max_output_tokens",
    "maxtokens",
    "incomplete",
)


# ---------------------------------------------------------------------------
# Chat-model detection helpers
# ---------------------------------------------------------------------------

_CHAT_KEYWORDS = (
    "chat",
    "instruct",
    "it",
    "-rl",
    "assistant",
    "tulu",
    "vicuna",
    "alpaca",
)


def _is_chat_model(model_name: str) -> bool:
    """Heuristic: model names containing chat/instruct keywords use a chat template."""
    lower = model_name.lower()
    return any(kw in lower for kw in _CHAT_KEYWORDS)


def _best_torch_dtype(torch):
    """Pick a sensible dtype for local inference."""
    if torch.cuda.is_available():
        if hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16
    return torch.float32


# ---------------------------------------------------------------------------
# Output cleanup helpers
# ---------------------------------------------------------------------------

class CodeBlock(NamedTuple):
    """One fenced code block found in a model response."""

    info: str      # info string after the opening fence ("turtle", "", ...)
    body: str      # block content, verbatim
    closed: bool   # False when the response ended before the closing fence


# An opening fence with an optional info string, the body, and then either a
# closing fence or the end of the response.  A response cut off by the token
# limit never emits its closing fence, so the \Z alternative is what lets us
# both recover such a block and *detect* that it was truncated.
_FENCE_RE = re.compile(
    r"```[ \t]*(?P<info>[A-Za-z0-9_.+-]*)[ \t]*\r?\n?"
    r"(?P<body>.*?)"
    r"(?:(?P<close>```)|\Z)",
    re.DOTALL,
)

# What a block must contain before we believe it really is Turtle.  Models
# routinely open their reply by echoing the prompt's own format instruction --
# "wrapped in a ```turtle ... ``` code block" -- which is itself a fenced block
# whose body is literally "...".  That echo is the FIRST fence in the response,
# so first-match selection returned three dots and threw the real ontology
# away.  Requiring a Turtle signature is what disqualifies it.
_TURTLE_SIGNATURE_RE = re.compile(r"@prefix\b|\sa\s+owl:", re.IGNORECASE)

# Info strings that identify a block as Turtle *by declaration*.  A model that
# writes ```turtle has told us what the block is; nothing later in the reply
# outranks that.
_STRICT_TURTLE_FENCE_LANGS = frozenset({"turtle", "ttl", "rdf", "n3", "trig", "owl"})

# Info strings that merely fail to contradict Turtle.  An untagged fence is the
# usual home of the CQ-to-axiom mapping table, whose Axiom column quotes real
# Turtle and is therefore far longer than the ontology it describes.
_AMBIGUOUS_TURTLE_FENCE_LANGS = frozenset({"", "text", "txt", "plaintext"})

# Kept as the union for callers that only ask "could this fence hold Turtle?".
_TURTLE_FENCE_LANGS = _STRICT_TURTLE_FENCE_LANGS | _AMBIGUOUS_TURTLE_FENCE_LANGS

# A Turtle directive: @prefix / @base, or their SPARQL-style spellings.  A block
# containing nothing but directives is a preamble, not an ontology.
_TURTLE_DIRECTIVE_RE = re.compile(r"^@?(?:prefix|base)\b", re.IGNORECASE)


def find_code_blocks(text: str) -> list[CodeBlock]:
    """Return every fenced code block in ``text``, in order of appearance."""
    return [
        CodeBlock(
            info=(m.group("info") or "").lower(),
            body=m.group("body"),
            closed=m.group("close") is not None,
        )
        for m in _FENCE_RE.finditer(text)
    ]


def looks_like_turtle(text: str) -> bool:
    """True when ``text`` carries an unambiguous Turtle/OWL signature."""
    return bool(_TURTLE_SIGNATURE_RE.search(text))


def has_turtle_statement(text: str) -> bool:
    """
    True when ``text`` actually declares something.

    ``looks_like_turtle`` is satisfied by a lone "@prefix" line, so a fence
    holding only the prefix preamble would otherwise count as the ontology and
    shadow the real one a few lines further down.
    """
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if _TURTLE_DIRECTIVE_RE.match(stripped):
            continue
        return True
    return False


def has_unterminated_fence(text: str) -> bool:
    """True when a code fence was opened but never closed (a truncation tell)."""
    return any(not block.closed for block in find_code_blocks(text))


def extract_turtle(response: str) -> str:
    """
    Extract Turtle content from a model response.

    Selection order:
    1) Among the fenced blocks carrying a Turtle signature ("@prefix" or
       " a owl:") AND declaring at least one non-directive statement, the FIRST
       whose info tag is in the strict Turtle family (turtle/ttl/rdf/n3/trig/
       owl).  Every prompt template asks for the ontology FIRST and the mapping
       tables after it, and a ```turtle tag is the model telling us which block
       is the ontology.
    2) Only when no strictly-tagged block qualifies do we fall back to the
       longest among the ambiguously-tagged ones.  Preferring length here is a
       last resort, not the rule: with a realistic token budget the
       CQ-to-axiom mapping table -- which quotes Turtle in its Axiom column and
       is emitted untagged -- is reliably longer than the ODP itself, so
       "longest wins" hands back the table.
    3) If no fenced block qualifies, the raw text from the first line that
       looks like Turtle onwards.
    4) If nothing in the response resembles Turtle, the empty string.  We
       deliberately do NOT fall back to "whatever was inside the first fence":
       returning "..." or a paragraph of prose disguises a failed generation
       as an ontology.  Callers record the empty extraction in metadata.
    """
    text = response.strip()

    candidates = [b for b in find_code_blocks(text) if looks_like_turtle(b.body)]
    if candidates:
        # Prefer blocks that declare something over bare prefix preambles.
        substantive = [b for b in candidates if has_turtle_statement(b.body)]
        pool = substantive or candidates

        strict = [b for b in pool if b.info in _STRICT_TURTLE_FENCE_LANGS]
        if strict:
            return strict[0].body.strip()

        ambiguous = [b for b in pool if b.info in _AMBIGUOUS_TURTLE_FENCE_LANGS]
        best = max(ambiguous or pool, key=lambda b: len(b.body.strip()))
        return best.body.strip()

    # Fallback: no fenced block qualified -- work on the raw text
    lines = text.splitlines()

    # Start from the first line that looks like Turtle.
    # We intentionally exclude bare ":" from markers -- it is too broad and
    # would match ":ClassName" in reasoning bullet points.  Instead we
    # look for unambiguous Turtle starts:
    #   • @prefix / @base / prefix / base declarations
    #   • Absolute URI references  <http://…>
    #   • A line that looks like a typed OWL declaration:
    #       :Something  a  owl:…   (must contain " a owl:" or " a rdf:")
    turtle_start = None
    _SAFE_MARKERS = ("@prefix", "@base", "prefix ", "base ", "<http://", "<https://")

    for i, line in enumerate(lines):
        stripped = line.strip()
        if not stripped:
            continue
        lower = stripped.lower()
        if lower.startswith(_SAFE_MARKERS):
            turtle_start = i
            break
        # Accept ":Foo a owl:…" declarations but NOT bare ":Foo" bullet references
        if stripped.startswith(":") and (" a owl:" in lower or " a rdf:" in lower):
            turtle_start = i
            break

    if turtle_start is None:
        # Nothing in the whole response resembles Turtle.  Return nothing
        # rather than prose, or the "..." of an echoed instruction, which
        # would masquerade downstream as a real (badly parsing) ontology.
        return ""

    text = "\n".join(lines[turtle_start:]).strip()

    # Remove stray code fences if any remain
    text = re.sub(r"^```[a-zA-Z0-9_-]*\s*", "", text)
    text = re.sub(r"\s*```$", "", text)

    return text.strip()


# Repair names recorded in metadata.  ``DROPPED_INCOMPLETE_STATEMENT`` is the
# load-bearing one: it means the document ended mid-statement, which only
# happens when the generation was cut off.
REPAIR_DROPPED_INCOMPLETE_STATEMENT = "dropped_incomplete_trailing_statement"
REPAIR_NORMALIZED_PREFIX_QNAMES = "normalized_standard_prefix_qnames"
REPAIR_COMMENTED_STRAY_SEMICOLON = "commented_out_stray_semicolon_heading"
REPAIR_PREFIXED_BARE_PREDICATE = "prefixed_bare_predicate"


def maybe_fix_common_turtle_issues(ttl: str, *, report: bool = False):
    """
    Conservative cleanup for common model errors in generated Turtle.

    With ``report=True`` returns ``(fixed_text, repairs)`` where ``repairs`` is
    the list of repair names applied; otherwise returns just the text, so the
    long-standing single-argument callers keep working.

    Repair 4 -- dropping an incomplete trailing statement -- is the one that
    used to launder a truncated generation into a clean-parsing ontology.  It
    is still applied (rdflib would otherwise refuse the whole file), but it is
    now *reported*, so the caller can flag the artifact instead of shipping a
    cut-off answer that is indistinguishable from a finished one.

    Fixes applied (in order):

    1. Spurious leading colon on standard prefixes:
           :rdfs:label  ->  rdfs:label
           :owl:Class   ->  owl:Class
           :rdf:type    ->  rdf:type
           :xsd:string  ->  xsd:string
       This is a systematic bug produced by Llama 2 70B.

    2. Standalone ';' lines between statements (Llama 2 section headings):
           . <blank>
           ; :Scenario-requirement-to-axiom mapping
        -> # :Scenario-requirement-to-axiom mapping
       A ';' predicate separator is only valid *within* a subject block.
       Between statements (after '.') it is invalid Turtle -- convert to comments.

    3. Missing colon on custom predicates after ';':
           ; has_method :Method1 .
        -> ; :has_method :Method1 .
       Only applied to tokens that do NOT start with a known standard prefix.
    """
    repairs: list[str] = []

    # Fix 1: :rdfs: / :owl: / :rdf: / :xsd: / :skos: / :dc: spurious leading colon
    _STANDARD = ("rdf", "rdfs", "owl", "xsd", "skos", "dc", "dcterms", "sh", "prov")
    ttl, n_prefix = re.subn(
        r":(rdf|rdfs|owl|xsd|skos|dc|dcterms|sh|prov):",
        r"\1:",
        ttl,
    )
    if n_prefix:
        repairs.append(REPAIR_NORMALIZED_PREFIX_QNAMES)

    # Fix 2: standalone ';' section-heading lines between Turtle statements
    lines = ttl.splitlines()
    fixed_lines = []
    between_statements = True  # True at start-of-file and after every '.'

    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            fixed_lines.append(line)
            continue

        if stripped.startswith(";") and between_statements:
            # Section heading written as '; text' between subjects -> comment
            fixed_lines.append("# " + stripped[1:].strip())
            repairs.append(REPAIR_COMMENTED_STRAY_SEMICOLON)
            continue

        fixed_lines.append(line)

        # Track whether we're between top-level statements
        if stripped.endswith("."):
            between_statements = True
        elif stripped.endswith(";") or stripped.endswith(","):
            between_statements = False

    ttl = "\n".join(fixed_lines)

    # Fix 3: custom predicate after ';' missing the ':' prefix
    before_fix3 = ttl
    ttl = re.sub(
        r"(^\s*[;]\s*)([A-Za-z_][A-Za-z0-9_-]*)(\s+)",
        lambda m: f"{m.group(1)}:{m.group(2)}{m.group(3)}"
        if not m.group(2).startswith(_STANDARD)
        else m.group(0),
        ttl,
        flags=re.MULTILINE,
    )
    if ttl != before_fix3:
        repairs.append(REPAIR_PREFIXED_BARE_PREDICATE)

    # Fix 4: truncated output -- if the document ends without a closing '.', strip the
    # incomplete trailing statement so rdflib does not crash at end-of-buffer.
    lines = ttl.splitlines()
    last_dot = -1
    for i in range(len(lines) - 1, -1, -1):
        stripped = lines[i].strip()
        if stripped.endswith(".") and not stripped.startswith("#"):
            last_dot = i
            break
    if last_dot >= 0 and last_dot < len(lines) - 1:
        # Everything after the last complete statement is a fragment.  If any of
        # it is non-blank the model was still writing when it stopped.
        if any(line.strip() for line in lines[last_dot + 1:]):
            repairs.append(REPAIR_DROPPED_INCOMPLETE_STATEMENT)
        ttl = "\n".join(lines[: last_dot + 1])

    # De-duplicate while preserving order (fix 2 can fire on many lines).
    repairs = list(dict.fromkeys(repairs))

    if report:
        return ttl, repairs
    return ttl


# ---------------------------------------------------------------------------
# Truncation bookkeeping
# ---------------------------------------------------------------------------

class GenerationResult(NamedTuple):
    """A model response plus whatever the backend told us about how it ended."""

    text: str
    finish_reason: str | None = None


SIGNAL_UNTERMINATED_FENCE = "unterminated_code_fence"
SIGNAL_REPAIR_DROPPED_STATEMENT = "repair_dropped_incomplete_statement"


def truncation_signals(
    response_text: str,
    finish_reason: str | None = None,
    repairs: list[str] | None = None,
) -> list[str]:
    """
    Return the reasons to believe this response was cut off mid-generation.

    An empty list means "no evidence of truncation".  Three independent tells:

    * the backend's own finish reason ("length", "MAX_TOKENS",
      "incomplete:max_output_tokens", ...),
    * a code fence that was opened and never closed, which is what a
      mid-axiom cut looks like in the raw text, and
    * a repair pass that had to drop an incomplete trailing statement -- the
      tell that survives even when the backend reports "stop" and the fence
      happens to be closed.

    The result is written into metadata.json so a truncated generation is
    never silently scored as if it were a complete one.
    """
    signals: list[str] = []

    if finish_reason:
        lowered = str(finish_reason).lower()
        if any(marker in lowered for marker in _LENGTH_FINISH_MARKERS):
            signals.append(f"finish_reason={finish_reason}")

    if has_unterminated_fence(response_text):
        signals.append(SIGNAL_UNTERMINATED_FENCE)

    if repairs and REPAIR_DROPPED_INCOMPLETE_STATEMENT in repairs:
        signals.append(SIGNAL_REPAIR_DROPPED_STATEMENT)

    return signals


# ---------------------------------------------------------------------------
# Extraction, repair and truncation verdict in one place
# ---------------------------------------------------------------------------

class ExtractionResult(NamedTuple):
    """What we extracted, what we had to repair, and whether to trust it."""

    turtle: str
    repairs: list[str]
    truncation_signals: list[str]
    truncated: bool
    empty: bool


def extract_ontology(
    response: str,
    finish_reason: str | None = None,
    apply_repairs: bool = False,
) -> ExtractionResult:
    """
    Extract the ontology AND decide whether the generation was complete.

    The repair pass is always *run* -- its report is evidence about truncation
    even when we do not keep its output -- but its result is only adopted when
    ``apply_repairs`` is set.  This is the single choke point that stops a
    truncated response from being silently upgraded into a clean one.
    """
    turtle = extract_turtle(response)
    repaired, repairs = maybe_fix_common_turtle_issues(turtle, report=True)
    if apply_repairs:
        turtle = repaired

    signals = truncation_signals(response, finish_reason, repairs)
    return ExtractionResult(
        turtle=turtle,
        repairs=repairs,
        truncation_signals=signals,
        truncated=bool(signals),
        empty=not turtle.strip(),
    )


TRUNCATION_BANNER_PREFIX = "# ODPGEN-TRUNCATED"


def annotate_truncated_turtle(ttl: str, signals: list[str]) -> str:
    """
    Prepend a machine-greppable banner to a truncated ontology.

    The banner is a Turtle comment, so the file still parses -- the point is
    not to break the scorer but to make it impossible for a downstream reader
    to look at ontology.ttl and not know the generation was cut off.  The
    authoritative record is metadata.json; this is the copy that travels with
    the artifact.
    """
    if not signals:
        return ttl

    banner = (
        f"{TRUNCATION_BANNER_PREFIX}: this generation was cut off before the model "
        "finished.\n"
        f"# ODPGEN-TRUNCATION-SIGNALS: {', '.join(signals)}\n"
        "# Do NOT score this file as a complete ontology; see metadata.json.\n"
    )
    return banner + ttl


# ---------------------------------------------------------------------------
# Backends
# ---------------------------------------------------------------------------

def load_huggingface_pipeline(
    model: str,
    hf_token: str | None = None,
    quantize: str = "none",
):
    """
    Load the tokenizer + model + pipeline once and reuse them for all prompts.
    This avoids re-loading a 70B model for every scenario.
    """
    try:
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            BitsAndBytesConfig,
            pipeline,
        )
        import torch
    except ImportError as exc:
        raise RuntimeError(
            "Hugging Face backend requires 'transformers' and 'torch'. "
            "Install with: pip install transformers torch accelerate"
        ) from exc

    token = hf_token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")

    quant_cfg = None
    if quantize == "4bit":
        try:
            quant_cfg = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=_best_torch_dtype(torch),
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
        except Exception as exc:
            raise RuntimeError(
                "4-bit quantization requires 'bitsandbytes'. "
                "Install with: pip install bitsandbytes"
            ) from exc
    elif quantize == "8bit":
        try:
            quant_cfg = BitsAndBytesConfig(load_in_8bit=True)
        except Exception as exc:
            raise RuntimeError(
                "8-bit quantization requires 'bitsandbytes'. "
                "Install with: pip install bitsandbytes"
            ) from exc

    tokenizer = AutoTokenizer.from_pretrained(model, token=token)

    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    torch_dtype = None if quant_cfg is not None else _best_torch_dtype(torch)

    try:
        model_obj = AutoModelForCausalLM.from_pretrained(
            model,
            device_map="auto",
            quantization_config=quant_cfg,
            torch_dtype=torch_dtype,
            token=token,
        )
    except ValueError as exc:
        msg = str(exc)
        if quantize in {"4bit", "8bit"} and (
            "Some modules are dispatched on the CPU or the disk" in msg
            or "llm_int8_enable_fp32_cpu_offload" in msg
        ):
            raise RuntimeError(
                "The quantized model still does not fit on the available GPU(s). "
                "Your options are:\n"
                "1) use a smaller model,\n"
                "2) add more GPU memory,\n"
                "3) configure explicit CPU offload with a custom device_map,\n"
                "4) reduce concurrent GPU memory use.\n\n"
                "This script already avoids repeated model loads; if you still see this "
                "at first load, the hardware is the bottleneck."
            ) from exc
        raise

    if getattr(model_obj.config, "pad_token_id", None) is None and tokenizer.pad_token_id is not None:
        model_obj.config.pad_token_id = tokenizer.pad_token_id

    pipe = pipeline(
        "text-generation",
        model=model_obj,
        tokenizer=tokenizer,
    )

    return tokenizer, pipe


def generate_with_huggingface(
    pipe,
    tokenizer,
    model_name: str,
    prompt: str,
    temperature: float,
    max_new_tokens: int,
) -> GenerationResult:
    """
    Generate text with a preloaded Hugging Face pipeline.
    Avoids passing sampling-only args when do_sample=False.

    The text-generation pipeline does not expose a finish reason, so we infer
    one: a completion whose length reaches the requested budget almost
    certainly stopped because it ran out of room, not because it was done.
    """
    do_sample = temperature > 0

    gen_kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "return_full_text": False,
    }

    if do_sample:
        gen_kwargs["temperature"] = temperature
    else:
        gen_kwargs["temperature"] = 1.0
        gen_kwargs["top_p"] = 1.0

    if _is_chat_model(model_name) and getattr(tokenizer, "chat_template", None):
        messages = [{"role": "user", "content": prompt}]
        output = pipe(messages, **gen_kwargs)
        text = output[0]["generated_text"]

        if isinstance(text, list):
            text = text[-1].get("content", "")
    else:
        output = pipe(prompt, **gen_kwargs)
        text = output[0]["generated_text"]

    finish_reason = "stop"
    try:
        generated_tokens = len(tokenizer(text, add_special_tokens=False)["input_ids"])
        # Tokenising the decoded text is approximate, hence the small margin.
        if generated_tokens >= max_new_tokens - 2:
            finish_reason = "length"
    except Exception:
        finish_reason = "unknown"

    return GenerationResult(text.strip(), finish_reason)


def generate_with_openai(
    prompt: str,
    model: str,
    temperature: float,
    max_new_tokens: int,
) -> GenerationResult:
    """
    Generate text with the OpenAI Responses API.

    The Responses API reports truncation through ``status == "incomplete"``
    together with ``incomplete_details.reason == "max_output_tokens"``; we pass
    both up so the caller can record them.
    """
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OpenAI backend requires the OPENAI_API_KEY environment variable to be set."
        )

    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError(
            "OpenAI backend requires the 'openai' package. "
            "Install with: pip install openai"
        ) from exc

    client = OpenAI(api_key=api_key)

    response = client.responses.create(
        model=model,
        input=prompt,
        temperature=temperature,
        max_output_tokens=max_new_tokens,
    )

    finish_reason = getattr(response, "status", None) or "unknown"
    details = getattr(response, "incomplete_details", None)
    detail_reason = getattr(details, "reason", None) if details is not None else None
    if detail_reason:
        finish_reason = f"{finish_reason}:{detail_reason}"

    if "incomplete" in str(finish_reason).lower():
        print(
            f"  [openai] WARNING: {finish_reason} -- output is truncated; "
            f"raise --max-new-tokens (currently {max_new_tokens})."
        )

    text = response.output_text.strip() if response.output_text else ""
    return GenerationResult(text, finish_reason)


_GEMINI_MIN_OUTPUT_TOKENS = 8192   # Thinking models need far more than 1024


def generate_with_gemini(
    prompt: str,
    model: str,
    temperature: float,
    max_new_tokens: int,
    thinking_budget: int | None = None,
) -> GenerationResult:
    """
    Generate text with the Google Gemini API (google-genai SDK).

    Notes
    -----
    * Gemini 2.5-family "thinking" models (e.g. gemini-2.5-pro-preview,
      gemini-3.1-pro-preview) reason internally before producing output.
      The thinking tokens do NOT count against max_output_tokens, but the
      thinking budget itself can be large, so max_output_tokens < 4096 often
      results in the model running out of tokens before finishing the Turtle block.
      We enforce a minimum of 8 192 output tokens.
    * temperature=0 is valid for all Gemini models; no clamping required.
    * thinking_budget=0 disables the thinking phase entirely (faster, cheaper,
      but lower quality). Leave as None to use the model's default budget.
    """
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "Gemini backend requires the GEMINI_API_KEY environment variable to be set."
        )

    try:
        import google.genai as genai
        from google.genai import types
    except ImportError as exc:
        raise RuntimeError(
            "Gemini backend requires the 'google-genai' package. "
            "Install with: pip install google-genai"
        ) from exc

    effective_tokens = max(max_new_tokens, _GEMINI_MIN_OUTPUT_TOKENS)
    if max_new_tokens < _GEMINI_MIN_OUTPUT_TOKENS:
        print(
            f"  [gemini] WARNING: --max-new-tokens {max_new_tokens} is too low for a "
            f"thinking model; raising to {effective_tokens}."
        )

    gen_config_kwargs: dict = dict(
        temperature=temperature,
        max_output_tokens=effective_tokens,
    )
    if thinking_budget is not None:
        gen_config_kwargs["thinking_config"] = types.ThinkingConfig(
            thinking_budget=thinking_budget
        )

    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model=model,
        contents=prompt,
        config=types.GenerateContentConfig(**gen_config_kwargs),
    )

    # Surface finish reason so the caller can spot MAX_TOKENS truncations
    finish_reason = "unknown"
    if response.candidates:
        finish = response.candidates[0].finish_reason
        finish_reason = str(finish)
        if finish_reason not in ("FinishReason.STOP", "STOP", "1"):
            print(f"  [gemini] WARNING: finish_reason={finish} -- output may be truncated.")

    text = response.text.strip() if response.text else ""
    return GenerationResult(text, finish_reason)


# ---------------------------------------------------------------------------
# Config / template map
# ---------------------------------------------------------------------------

CONFIG_TEMPLATE = {
    "scenario-only": "scenario_only.txt",
    "cq-only": "cq_only.txt",
    "scenario-cq": "scenario_cq.txt",
    "scenario-cq-reasoning": "scenario_cq_reasoning.txt",
    "scenario-cq-constraints": "scenario_cq_constraints.txt",
}


def render(template: str, scenario_text: str, cq_list: list[str]) -> str:
    cq_block = "\n".join(f"- {cq}" for cq in cq_list) if cq_list else "- (none provided)"
    return (
        template.replace("{{SCENARIO_TEXT}}", scenario_text)
        .replace("{{CQ_LIST}}", cq_block)
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Run ODP generation experiments.")
    parser.add_argument("--data", default="data/scenarios/pattern_scenarios.json", type=Path)
    parser.add_argument("--prompts-dir", default="prompts", type=Path)
    parser.add_argument("--outputs-dir", default="outputs", type=Path)
    parser.add_argument(
        "--model",
        required=True,
        help=(
            "Model name "
            "(e.g. gpt-3.5-turbo, meta-llama/Llama-2-70b-chat-hf, bigscience/bloom)."
        ),
    )
    parser.add_argument(
        "--backend",
        choices=["huggingface", "openai", "gemini"],
        required=True,
        help=(
            "Generation backend. "
            "'huggingface' for open-source LLMs from Hugging Face Hub. "
            "'openai' for OpenAI models (requires OPENAI_API_KEY env var). "
            "'gemini' for Google Gemini models (requires GEMINI_API_KEY env var)."
        ),
    )
    parser.add_argument("--config", choices=list(CONFIG_TEMPLATE.keys()) + ["all"], default="all")
    parser.add_argument(
        "--only",
        metavar="CONFIG:SCENARIO",
        action="append",
        default=None,
        help=(
            "Restrict generation to specific (config, scenario) cells, e.g. "
            "--only scenario-cq:2023-133-02. Repeatable, or comma-separated. "
            "Every other cell is skipped and its artefacts are left untouched, "
            "so a handful of damaged generations can be repaired without "
            "re-running (and re-billing) the whole grid."
        ),
    )
    parser.add_argument(
        "--only-file",
        type=Path,
        default=None,
        help="File of CONFIG:SCENARIO lines to restrict generation to; '#' comments allowed.",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=DEFAULT_MAX_NEW_TOKENS,
        metavar="N",
        help=(
            f"Output token budget per generation (default: {DEFAULT_MAX_NEW_TOKENS}). "
            "The prompts ask for four deliverables (ontology + three tables), so a "
            f"budget below {RECOMMENDED_MIN_NEW_TOKENS} truncates verbose models "
            "mid-axiom; runs made below that floor are warned about and flagged in "
            "metadata.json. Raise it for reasoning / thinking models."
        ),
    )
    parser.add_argument(
        "--hf-token",
        default=None,
        help=(
            "Hugging Face access token for gated models. "
            "Can also be set via HF_TOKEN or HUGGING_FACE_HUB_TOKEN."
        ),
    )
    parser.add_argument(
        "--quantize",
        choices=["none", "4bit", "8bit"],
        default="none",
        help=(
            "Quantization mode for large HuggingFace models. "
            "Requires 'bitsandbytes'."
        ),
    )
    parser.add_argument(
        "--fix-common-turtle-issues",
        action="store_true",
        help="Apply conservative cleanup for common QName predicate mistakes like '; has_method' -> '; :has_method'.",
    )
    parser.add_argument(
        "--thinking-budget",
        type=int,
        default=None,
        metavar="N",
        help=(
            "Gemini only. Set the thinking token budget (0 = disable thinking, "
            "e.g. 8192 = default budget). Leave unset to use the model's default. "
            "Useful for gemini-2.5-pro / gemini-3.1-pro-preview thinking models."
        ),
    )
    args = parser.parse_args()

    budget_too_low = args.max_new_tokens < RECOMMENDED_MIN_NEW_TOKENS
    if budget_too_low:
        print(
            f"WARNING: --max-new-tokens {args.max_new_tokens} is below the "
            f"recommended floor of {RECOMMENDED_MIN_NEW_TOKENS}. The prompts ask "
            "for four deliverables; verbose models will be cut off mid-axiom and "
            "the extracted ontology will be incomplete. Every generation from this "
            'run is flagged with "max_new_tokens_below_recommended": true.'
        )

    # Normalise --only / --only-file into a set of (config, scenario_id) pairs.
    only_pairs = set()
    raw_only = list(args.only or [])
    if args.only_file:
        for line in args.only_file.read_text(encoding="utf-8").splitlines():
            line = line.split("#", 1)[0].strip()
            if line:
                raw_only.append(line)
    for entry in raw_only:
        for piece in str(entry).split(","):
            piece = piece.strip()
            if not piece:
                continue
            if ":" not in piece:
                parser.error("--only expects CONFIG:SCENARIO, got %r" % piece)
            cfg, sid = piece.split(":", 1)
            only_pairs.add((cfg.strip(), sid.strip()))
    args.only = only_pairs or None
    if args.only:
        print("[run_generation] restricted to %d cell(s)" % len(args.only))

    scenarios = json.loads(args.data.read_text(encoding="utf-8"))
    configs = list(CONFIG_TEMPLATE.keys()) if args.config == "all" else [args.config]

    model_dir_name = args.model.replace("/", "_").replace(":", "_")

    hf_tokenizer = None
    hf_pipe = None
    if not args.dry_run and args.backend == "huggingface":
        hf_tokenizer, hf_pipe = load_huggingface_pipeline(
            model=args.model,
            hf_token=args.hf_token,
            quantize=args.quantize,
        )

    try:
        for config in configs:
            template_path = args.prompts_dir / CONFIG_TEMPLATE[config]
            template = template_path.read_text(encoding="utf-8")

            for scenario in scenarios:
                # Targeted re-generation: repair a handful of damaged artefacts
                # without re-running -- and re-billing -- the whole grid, and
                # without overwriting artefacts that are already sound.
                if args.only and (config, scenario["scenario_id"]) not in args.only:
                    continue

                prompt = render(
                    template,
                    scenario["scenario_text"],
                    scenario.get("cq_list", []),
                )
                prompt_hash = hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:12]

                output_dir = args.outputs_dir / model_dir_name / config / scenario["scenario_id"]
                output_dir.mkdir(parents=True, exist_ok=True)

                if args.dry_run:
                    (output_dir / "prompt.txt").write_text(prompt, encoding="utf-8")
                    print(f"[dry-run] wrote prompt -> {output_dir / 'prompt.txt'}")
                    continue

                if args.backend == "huggingface":
                    result = generate_with_huggingface(
                        pipe=hf_pipe,
                        tokenizer=hf_tokenizer,
                        model_name=args.model,
                        prompt=prompt,
                        temperature=args.temperature,
                        max_new_tokens=args.max_new_tokens,
                    )
                elif args.backend == "gemini":
                    result = generate_with_gemini(
                        prompt=prompt,
                        model=args.model,
                        temperature=args.temperature,
                        max_new_tokens=args.max_new_tokens,
                        thinking_budget=args.thinking_budget,
                    )
                else:
                    result = generate_with_openai(
                        prompt=prompt,
                        model=args.model,
                        temperature=args.temperature,
                        max_new_tokens=args.max_new_tokens,
                    )

                response = result.text

                # Always fix for HuggingFace (systematic prefix bugs); the flag
                # enables it for OpenAI too.  Either way the repair report is
                # computed, because a dropped trailing statement is evidence of
                # truncation whether or not we keep the repaired text.
                apply_repairs = (
                    args.fix_common_turtle_issues or args.backend == "huggingface"
                )
                extraction = extract_ontology(
                    response,
                    finish_reason=result.finish_reason,
                    apply_repairs=apply_repairs,
                )
                ontology_ttl = extraction.turtle
                signals = extraction.truncation_signals

                response_hash = hashlib.sha256(response.encode("utf-8")).hexdigest()[:12]
                ontology_hash = hashlib.sha256(ontology_ttl.encode("utf-8")).hexdigest()[:12]

                effective_max_new_tokens = args.max_new_tokens
                if args.backend == "gemini":
                    effective_max_new_tokens = max(
                        args.max_new_tokens, _GEMINI_MIN_OUTPUT_TOKENS
                    )

                ontology_empty = extraction.empty

                metadata = {
                    "model": args.model,
                    "backend": args.backend,
                    "config": config,
                    "scenario_id": scenario["scenario_id"],
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "temperature": args.temperature,
                    "max_new_tokens": args.max_new_tokens,
                    "max_new_tokens_effective": effective_max_new_tokens,
                    "max_new_tokens_below_recommended": budget_too_low,
                    "prompt_hash": prompt_hash,
                    "response_hash": response_hash,
                    "ontology_hash": ontology_hash,
                    # Truncation bookkeeping: a generation that ran out of output
                    # budget is incomplete by construction and must not be scored
                    # as if the model had chosen to stop there.
                    "finish_reason": result.finish_reason,
                    "truncated": extraction.truncated,
                    "truncation_signals": signals,
                    # Repair bookkeeping: which cleanups the text needed, and
                    # whether the file on disk carries the truncation banner.
                    "ontology_repairs": extraction.repairs,
                    "ontology_repairs_applied": apply_repairs,
                    # Extraction bookkeeping: an empty ontology parses cleanly and
                    # would otherwise look like a flawless generation.
                    "response_chars": len(response),
                    "ontology_chars": len(ontology_ttl),
                    "ontology_empty": ontology_empty,
                    "ontology_file_annotated": bool(signals),
                    "quantize": args.quantize if args.backend == "huggingface" else "n/a",
                    "thinking_budget": args.thinking_budget if args.backend == "gemini" else "n/a",
                }

                (output_dir / "prompt.txt").write_text(prompt, encoding="utf-8")
                (output_dir / "raw_response.txt").write_text(response, encoding="utf-8")
                # A truncated generation must never leave here looking clean.
                (output_dir / "ontology.ttl").write_text(
                    annotate_truncated_turtle(ontology_ttl, signals), encoding="utf-8"
                )
                (output_dir / "metadata.json").write_text(
                    json.dumps(metadata, indent=2),
                    encoding="utf-8",
                )

                flags = ""
                if signals:
                    flags += f"  [TRUNCATED: {', '.join(signals)}]"
                if ontology_empty:
                    flags += "  [EMPTY ONTOLOGY: no Turtle found in the response]"

                print(
                    f"[{args.backend}] {args.model} | {config} | "
                    f"{scenario['scenario_id']} -> {output_dir}{flags}"
                )
    finally:
        if args.backend == "huggingface":
            del hf_pipe
            del hf_tokenizer
            gc.collect()
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass


if __name__ == "__main__":
    main()