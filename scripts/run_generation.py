#!/usr/bin/env python3
import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

# ---------------------------------------------------------------------------
# Chat-model detection helpers
# ---------------------------------------------------------------------------

_CHAT_KEYWORDS = ("chat", "instruct", "it", "-rl", "assistant", "tulu", "vicuna", "alpaca")

def _is_chat_model(model_name: str) -> bool:
    """Heuristic: model names containing chat/instruct keywords use a chat template."""
    lower = model_name.lower()
    return any(kw in lower for kw in _CHAT_KEYWORDS)


# ---------------------------------------------------------------------------
# Backends
# ---------------------------------------------------------------------------

def generate_with_huggingface(
    prompt: str,
    model: str,
    temperature: float,
    max_new_tokens: int,
    hf_token: str | None = None,
    quantize: str = "none",
) -> str:
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline
        import torch
    except ImportError as exc:
        raise RuntimeError(
            "Hugging Face backend requires 'transformers' and 'torch'. "
            "Install with: pip install transformers torch accelerate"
        ) from exc

    token = hf_token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")

    # Build quantization config
    quant_cfg = None
    if quantize == "4bit":
        try:
            quant_cfg = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
        except Exception:
            raise RuntimeError(
                "4-bit quantization requires 'bitsandbytes'. "
                "Install with: pip install bitsandbytes"
            )
    elif quantize == "8bit":
        try:
            quant_cfg = BitsAndBytesConfig(load_in_8bit=True)
        except Exception:
            raise RuntimeError(
                "8-bit quantization requires 'bitsandbytes'. "
                "Install with: pip install bitsandbytes"
            )

    # Load tokenizer + model with device_map for multi-GPU / CPU offload
    tokenizer = AutoTokenizer.from_pretrained(model, token=token)
    model_obj = AutoModelForCausalLM.from_pretrained(
        model,
        device_map="auto",
        quantization_config=quant_cfg,
        torch_dtype=torch.float16 if quant_cfg is None else None,
        token=token,
    )

    pipe = pipeline(
        "text-generation",
        model=model_obj,
        tokenizer=tokenizer,
    )

    # Use chat template for instruction-tuned models when supported
    if _is_chat_model(model) and hasattr(tokenizer, "chat_template") and tokenizer.chat_template:
        messages = [{"role": "user", "content": prompt}]
        output = pipe(
            messages,
            max_new_tokens=max_new_tokens,
            temperature=max(temperature, 1e-5),
            do_sample=temperature > 0,
            return_full_text=False,
        )
        text = output[0]["generated_text"]
        # pipeline with chat template returns a list of message dicts
        if isinstance(text, list):
            text = text[-1].get("content", "")
    else:
        output = pipe(
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=max(temperature, 1e-5),
            do_sample=temperature > 0,
            return_full_text=False,
        )
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
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=max_new_tokens,
    )
    return response.choices[0].message.content.strip()


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
    return template.replace("{{SCENARIO_TEXT}}", scenario_text).replace("{{CQ_LIST}}", cq_block)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Run ODP generation experiments.")
    parser.add_argument("--data", default="data/scenarios/pattern_scenarios.json", type=Path)
    parser.add_argument("--prompts-dir", default="prompts", type=Path)
    parser.add_argument("--outputs-dir", default="outputs", type=Path)
    parser.add_argument("--model", required=True, help="Model name (e.g. gpt-3.5-turbo, meta-llama/Llama-2-70b-chat-hf, bigscience/bloom).")
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
        help="Hugging Face access token for gated models (e.g. Llama 2). "
             "Can also be set via HF_TOKEN environment variable.",
    )
    parser.add_argument(
        "--quantize",
        choices=["none", "4bit", "8bit"],
        default="none",
        help="Quantization mode for large HuggingFace models. "
             "Requires 'bitsandbytes'. Recommended: 4bit for 70B+ models.",
    )
    args = parser.parse_args()

    scenarios = json.loads(args.data.read_text(encoding="utf-8"))
    configs = list(CONFIG_TEMPLATE.keys()) if args.config == "all" else [args.config]

    # Sanitise model name for use as a directory (replace / and : with _)
    model_dir_name = args.model.replace("/", "_").replace(":", "_")

    for config in configs:
        template_path = args.prompts_dir / CONFIG_TEMPLATE[config]
        template = template_path.read_text(encoding="utf-8")
        for scenario in scenarios:
            prompt = render(template, scenario["scenario_text"], scenario.get("cq_list", []))
            prompt_hash = hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:12]
            output_dir = args.outputs_dir / model_dir_name / config / scenario["scenario_id"]
            output_dir.mkdir(parents=True, exist_ok=True)

            if args.dry_run:
                (output_dir / "prompt.txt").write_text(prompt, encoding="utf-8")
                print(f"[dry-run] wrote prompt → {output_dir / 'prompt.txt'}")
                continue

            if args.backend == "huggingface":
                response = generate_with_huggingface(
                    prompt=prompt,
                    model=args.model,
                    temperature=args.temperature,
                    max_new_tokens=args.max_new_tokens,
                    hf_token=args.hf_token,
                    quantize=args.quantize,
                )
            else:  # openai
                response = generate_with_openai(
                    prompt=prompt,
                    model=args.model,
                    temperature=args.temperature,
                    max_new_tokens=args.max_new_tokens,
                )

            response_hash = hashlib.sha256(response.encode("utf-8")).hexdigest()[:12]
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
                "quantize": args.quantize if args.backend == "huggingface" else "n/a",
            }
            (output_dir / "prompt.txt").write_text(prompt, encoding="utf-8")
            (output_dir / "raw_response.txt").write_text(response, encoding="utf-8")
            (output_dir / "ontology.ttl").write_text(response, encoding="utf-8")
            (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
            print(f"[{args.backend}] {args.model} | {config} | {scenario['scenario_id']} → {output_dir}")


if __name__ == "__main__":
    main()
