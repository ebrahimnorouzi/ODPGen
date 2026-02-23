import re


def tokenize(text: str) -> set[str]:
    tokens = re.findall(r"[A-Za-z][A-Za-z0-9_-]{2,}", text.lower())
    stop = {"the", "and", "for", "with", "that", "this", "which", "from", "into", "what", "who"}
    return {t for t in tokens if t not in stop}


def evaluate_faithfulness(scenario_text: str, cq_list: list[str], ontology_text: str) -> dict:
    source_terms = tokenize(scenario_text + " " + " ".join(cq_list))
    onto_terms = tokenize(ontology_text)
    if not source_terms:
        coverage = 0.0
    else:
        coverage = len(source_terms & onto_terms) / len(source_terms)
    unsupported = sorted(list(onto_terms - source_terms))
    unsupported_trimmed = unsupported[:30]
    return {
        "coverage_ratio": round(coverage, 3),
        "unsupported_addition_count": len(unsupported),
        "unsupported_additions_sample": unsupported_trimmed,
    }
