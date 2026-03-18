#!/usr/bin/env python3
import argparse
import gc
import hashlib
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path

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

def extract_turtle(response: str) -> str:
    """
    Extract Turtle content from a model response.

    Preference order:
    1) fenced ```turtle / ```ttl / ```rdf block
    2) any fenced code block
    3) raw text stripped of leading/trailing markdown noise
    """
    text = response.strip()

    fenced_patterns = [
        r"```(?:turtle|ttl|rdf)\s*(.*?)```",
        r"```(?:sparql)?\s*(.*?)```",
    ]

    for pattern in fenced_patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE | re.DOTALL)
        if match:
            return match.group(1).strip()

    # Fallback: remove a leading explanatory sentence if present
    lines = text.splitlines()

    # Start from the first line that looks like Turtle
    turtle_start = None
    turtle_markers = (
        "@prefix",
        "@base",
        "prefix ",
        "base ",
        "<http://",
        "<https://",
        ":",
    )

    for i, line in enumerate(lines):
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.lower().startswith(turtle_markers):
            turtle_start = i
            break

    if turtle_start is not None:
        text = "\n".join(lines[turtle_start:]).strip()

    # Remove stray code fences if any remain
    text = re.sub(r"^```[a-zA-Z0-9_-]*\s*", "", text)
    text = re.sub(r"\s*```$", "", text)

    return text.strip()


def maybe_fix_common_turtle_issues(ttl: str) -> str:
    """
    Conservative cleanup for common model errors in generated Turtle.

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
       Between statements (after '.') it is invalid Turtle — convert to comments.

    3. Missing colon on custom predicates after ';':
           ; has_method :Method1 .
        -> ; :has_method :Method1 .
       Only applied to tokens that do NOT start with a known standard prefix.
    """
    # Fix 1: :rdfs: / :owl: / :rdf: / :xsd: / :skos: / :dc: spurious leading colon
    _STANDARD = ("rdf", "rdfs", "owl", "xsd", "skos", "dc", "dcterms", "sh", "prov")
    ttl = re.sub(
        r":(rdf|rdfs|owl|xsd|skos|dc|dcterms|sh|prov):",
        r"\1:",
        ttl,
    )

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
            # Section heading written as '; text' between subjects → comment
            fixed_lines.append("# " + stripped[1:].strip())
            continue

        fixed_lines.append(line)

        # Track whether we're between top-level statements
        if stripped.endswith("."):
            between_statements = True
        elif stripped.endswith(";") or stripped.endswith(","):
            between_statements = False

    ttl = "\n".join(fixed_lines)

    # Fix 3: custom predicate after ';' missing the ':' prefix
    ttl = re.sub(
        r"(^\s*[;]\s*)([A-Za-z_][A-Za-z0-9_-]*)(\s+)",
        lambda m: f"{m.group(1)}:{m.group(2)}{m.group(3)}"
        if not m.group(2).startswith(_STANDARD)
        else m.group(0),
        ttl,
        flags=re.MULTILINE,
    )

    # Fix 4: truncated output — if the document ends without a closing '.', strip the
    # incomplete trailing statement so rdflib does not crash at end-of-buffer.
    lines = ttl.splitlines()
    last_dot = -1
    for i in range(len(lines) - 1, -1, -1):
        stripped = lines[i].strip()
        if stripped.endswith(".") and not stripped.startswith("#"):
            last_dot = i
            break
    if last_dot >= 0 and last_dot < len(lines) - 1:
        ttl = "\n".join(lines[: last_dot + 1])

    return ttl


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
) -> str:
    """
    Generate text with a preloaded Hugging Face pipeline.
    Avoids passing sampling-only args when do_sample=False.
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

    return text.strip()


def generate_with_openai(
    prompt: str,
    model: str,
    temperature: float,
    max_new_tokens: int,
) -> str:
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

    return response.output_text.strip() if response.output_text else ""

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
        choices=["huggingface", "openai"],
        required=True,
        help=(
            "Generation backend. "
            "'huggingface' for open-source LLMs from Hugging Face Hub. "
            "'openai' for OpenAI models (requires OPENAI_API_KEY env var)."
        ),
    )
    parser.add_argument("--config", choices=list(CONFIG_TEMPLATE.keys()) + ["all"], default="all")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
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
    args = parser.parse_args()

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
                    print(f"[dry-run] wrote prompt → {output_dir / 'prompt.txt'}")
                    continue

                if args.backend == "huggingface":
                    response = generate_with_huggingface(
                        pipe=hf_pipe,
                        tokenizer=hf_tokenizer,
                        model_name=args.model,
                        prompt=prompt,
                        temperature=args.temperature,
                        max_new_tokens=args.max_new_tokens,
                    )
                else:
                    response = generate_with_openai(
                        prompt=prompt,
                        model=args.model,
                        temperature=args.temperature,
                        max_new_tokens=args.max_new_tokens,
                    )

                ontology_ttl = extract_turtle(response)
                # Always fix for HuggingFace (systematic prefix bugs); flag enables it for OpenAI too
                if args.fix_common_turtle_issues or args.backend == "huggingface":
                    ontology_ttl = maybe_fix_common_turtle_issues(ontology_ttl)

                response_hash = hashlib.sha256(response.encode("utf-8")).hexdigest()[:12]
                ontology_hash = hashlib.sha256(ontology_ttl.encode("utf-8")).hexdigest()[:12]

                metadata = {
                    "model": args.model,
                    "backend": args.backend,
                    "config": config,
                    "scenario_id": scenario["scenario_id"],
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "temperature": args.temperature,
                    "max_new_tokens": args.max_new_tokens,
                    "prompt_hash": prompt_hash,
                    "response_hash": response_hash,
                    "ontology_hash": ontology_hash,
                    "quantize": args.quantize if args.backend == "huggingface" else "n/a",
                }

                (output_dir / "prompt.txt").write_text(prompt, encoding="utf-8")
                (output_dir / "raw_response.txt").write_text(response, encoding="utf-8")
                (output_dir / "ontology.ttl").write_text(ontology_ttl, encoding="utf-8")
                (output_dir / "metadata.json").write_text(
                    json.dumps(metadata, indent=2),
                    encoding="utf-8",
                )

                print(
                    f"[{args.backend}] {args.model} | {config} | "
                    f"{scenario['scenario_id']} → {output_dir}"
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