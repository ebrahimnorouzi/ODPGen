from pathlib import Path


def evaluate_formal_quality(ontology_path: Path) -> dict:
    text = ontology_path.read_text(encoding="utf-8")
    lines = [line.strip() for line in text.splitlines() if line.strip() and not line.strip().startswith("#")]
    has_prefix = any(line.startswith("@prefix") for line in lines)
    ttl_like = all(line.endswith(".") or line.startswith("@prefix") for line in lines)
    has_ontology_decl = "owl:Ontology" in text

    return {
        "syntax_valid": bool(has_prefix and ttl_like),
        "ontology_declared": bool(has_ontology_decl),
        "consistency_check": "not_evaluated_no_reasoner",
        "coherence_check": "not_evaluated_no_reasoner",
    }
