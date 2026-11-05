#!/usr/bin/env python3
"""Per-model structural table in the form of the paper's Table 2.

Reads the per-artefact records written by ``batch_evaluate.py`` (``--in``, walked
recursively) and prints, per generating model:

    n            artefacts evaluated
    parse        share that loads as RDF
    consistency  share that loads as RDF and passes OWL-RL screening (n in the denominator)
    oops         mean OOPS! component over the artefacts OOPS! scanned, where the
                 component is max(0, 1 - (important + 2 * critical) / 10)
    structural   0.5 * consistency + 0.5 * oops

This OOPS! component and the scanned-only mean are what the paper's Table 2
reports: applied to ``eval_results/odp_eval`` with ``--count-empty-as-parsed``
they reproduce its consistency, OOPS! and structural columns for Gemini, both
Llama models, Mistral and BLOOMZ (GPT-5.4 was regenerated after those records
were written). ``scripts/aggregate_eval.py`` writes the different component
1 / (1 + important + 2 * critical) averaged over all artefacts.

By default an artefact that parses to an empty graph counts as unparseable:
OWL-RL and OOPS! find nothing wrong with an empty file, and counting it as
parsed would rank a model that writes nothing at the top. The flag
``--count-empty-as-parsed`` restores the convention the published table used.

    python3 scripts/structural_table.py --in eval_results_wop2026/odp_eval
    python3 scripts/structural_table.py --in eval_results/odp_eval --count-empty-as-parsed
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path


def oops_component(oops: dict | None) -> float | None:
    """Paper formula; None when OOPS! never scanned the file."""
    if not oops or oops.get("skipped") or "error" in oops or oops.get("status_code") != 200:
        return None
    important = int(oops.get("important_count") or 0)
    critical = int(oops.get("critical_count") or 0)
    return max(0.0, 1.0 - (important + 2 * critical) / 10.0)


def load_records(root: Path) -> list[dict]:
    recs = []
    for p in sorted(root.rglob("*.json")):
        try:
            r = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if isinstance(r, dict) and {"id", "model", "config"} <= r.keys():
            recs.append(r)
    return recs


def parsed(rec: dict, count_empty: bool) -> bool:
    om = rec.get("ontometrics") or {}
    if "error" in rec or not om or "error" in om:
        return False
    return count_empty or (om.get("triples_count") or 0) > 0


def table(recs: list[dict], count_empty: bool, by_config: bool) -> list[dict]:
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for r in recs:
        groups[(r["model"], r["config"] if by_config else "ALL")].append(r)
    rows = []
    for (model, config), rs in groups.items():
        n = len(rs)
        n_parse = sum(parsed(r, count_empty) for r in rs)
        n_cons = sum(parsed(r, count_empty) and (r.get("reasoner") or {}).get("consistent") is True
                     for r in rs)
        comps = [c for c in (oops_component(r.get("oops")) for r in rs
                             if parsed(r, count_empty)) if c is not None]
        oops = sum(comps) / len(comps) if comps else None
        cons = n_cons / n
        rows.append({
            "model": model, "config": config, "n": n,
            "n_parse": n_parse, "parse": round(n_parse / n, 4),
            "n_consistent": n_cons, "consistency": round(cons, 4),
            "n_oops_scanned": len(comps),
            "oops": None if oops is None else round(oops, 4),
            "structural": None if oops is None else round(0.5 * cons + 0.5 * oops, 4),
            "n_truncated": sum(1 for r in rs if r.get("truncated") is True),
        })
    rows.sort(key=lambda x: (x["config"], -(x["structural"] if x["structural"] is not None
                                             else x["consistency"] / 2)))
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="inp", type=Path, required=True,
                    help="directory of batch_evaluate.py records")
    ap.add_argument("--by-config", action="store_true", help="one row per (model, config)")
    ap.add_argument("--count-empty-as-parsed", action="store_true",
                    help="count empty graphs as parsed (the published Table 2 convention)")
    ap.add_argument("--csv", type=Path, default=None, help="also write the table to this CSV")
    args = ap.parse_args()

    recs = load_records(args.inp)
    if not recs:
        sys.exit(f"no evaluation records under {args.inp}")
    rows = table(recs, args.count_empty_as_parsed, args.by_config)

    def fmt(v):
        return "-" if v is None else (f"{v:.3f}" if isinstance(v, float) else str(v))
    cols = list(rows[0])
    print("  ".join(cols))
    for r in rows:
        print("  ".join(fmt(r[c]) for c in cols))
    if any(r["oops"] is None for r in rows):
        print("\nNOTE: '-' in oops/structural = OOPS! scanned none of the group's files; run\n"
              "  python3 batch_evaluate.py --local-root <outputs> --out <records> "
              "--patch-oops --oops-url <OOPS! REST endpoint>")
    if args.csv:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with args.csv.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            w.writerows(rows)
        print(f"\nwritten: {args.csv}")


if __name__ == "__main__":
    main()
