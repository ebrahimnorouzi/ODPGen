"""
Data loader for the LLM-as-a-Judge evaluation pipeline.

Loads ground-truth ontologies, competency questions, and generated ODP outputs
from the project's directory structure.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SCENARIOS_PATH = PROJECT_ROOT / "data" / "scenarios" / "pattern_scenarios.json"
DEFAULT_GT_DIR = PROJECT_ROOT / "data" / "ground_truth"
DEFAULT_OUTPUTS_DIR = PROJECT_ROOT / "outputs"

_RDF_EXTENSIONS = [".ttl", ".owl", ".rdf", ".n3", ".nt"]


# ─────────────────────────────────────────────────────────────────────────────
# Data classes
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Scenario:
    """A single evaluation scenario with ground-truth and competency questions."""
    scenario_id: str
    scenario_text: str
    cq_list: list[str]
    ground_truth_path: Path | None = None
    ground_truth_text: str | None = None


@dataclass
class GeneratedODP:
    """A single generated ontology output."""
    model: str
    config: str
    scenario_id: str
    ontology_path: Path
    ontology_text: str = ""
    metadata: dict = field(default_factory=dict)


# ─────────────────────────────────────────────────────────────────────────────
# Loaders
# ─────────────────────────────────────────────────────────────────────────────

def load_scenarios(
    scenarios_path: Path = DEFAULT_SCENARIOS_PATH,
    gt_dir: Path = DEFAULT_GT_DIR,
) -> dict[str, Scenario]:
    """Load scenario definitions and attach ground-truth ontology text."""
    raw = json.loads(scenarios_path.read_text(encoding="utf-8"))
    scenarios: dict[str, Scenario] = {}

    for item in raw:
        sid = item["scenario_id"]
        gt_path = _find_gt_file(gt_dir, sid)
        gt_text = None
        if gt_path is not None and gt_path.exists():
            gt_text = gt_path.read_text(encoding="utf-8", errors="replace")

        scenarios[sid] = Scenario(
            scenario_id=sid,
            scenario_text=item.get("scenario_text", ""),
            cq_list=item.get("cq_list", []),
            ground_truth_path=gt_path,
            ground_truth_text=gt_text,
        )
    return scenarios


def load_all_generated_odps(
    outputs_dir: Path = DEFAULT_OUTPUTS_DIR,
) -> list[GeneratedODP]:
    """
    Discover all generated ontology.ttl files under outputs/.
    Directory layout: outputs/{model}/{config}/{scenario_id}/ontology.ttl
    """
    results: list[GeneratedODP] = []
    for ontology_path in sorted(outputs_dir.glob("*/*/*/ontology.ttl")):
        parts = ontology_path.parts
        model = parts[-4]
        config = parts[-3]
        scenario_id = parts[-2]

        meta_path = ontology_path.parent / "metadata.json"
        metadata = {}
        if meta_path.exists():
            metadata = json.loads(meta_path.read_text(encoding="utf-8"))

        odp = GeneratedODP(
            model=model,
            config=config,
            scenario_id=scenario_id,
            ontology_path=ontology_path,
            ontology_text=ontology_path.read_text(encoding="utf-8", errors="replace"),
            metadata=metadata,
        )
        results.append(odp)
    return results


def load_generated_odps_for_scenario(
    scenario_id: str,
    outputs_dir: Path = DEFAULT_OUTPUTS_DIR,
) -> list[GeneratedODP]:
    """Load all generated ODPs for a specific scenario."""
    return [
        odp for odp in load_all_generated_odps(outputs_dir)
        if odp.scenario_id == scenario_id
    ]


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _find_gt_file(gt_dir: Path, sid: str) -> Path | None:
    """Find the ground-truth file for a given scenario ID."""
    for ext in _RDF_EXTENSIONS:
        p = gt_dir / f"{sid}{ext}"
        if p.exists():
            return p
    return None


def format_cqs(cq_list: list[str]) -> str:
    """Format competency questions as a numbered list for prompt insertion."""
    return "\n".join(f"  CQ{i+1}. {cq}" for i, cq in enumerate(cq_list))
