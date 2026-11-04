#!/usr/bin/env python3
"""Recompute the functional-evaluation scores (paper, Sec. 3.4 and Table 3).

eval/functional_eval_detail.csv and eval/functional_eval_ranking.csv had no
producer in the repository. This script is that producer.

Per ODP (one row per id / model / config):

    numb                 superfluous elements (manual annotation, see
                         eval_results/odp_eval/bigscience_bloomz-7b1/manualeval.csv)
    numbnoscen           ... of which not justified by the scenario
    cq_pass_rate         CQ verification rate in [0, 1]

    superfluous_penalty = 1.0 * numbnoscen + 0.3 * (numb - numbnoscen)
    manual_score        = 1 / (1 + 0.5 * superfluous_penalty)
    functional_score    = 0.5 * manual_score + 0.5 * cq_pass_rate

The ranking averages these per (model, config).

Usage:
    python3 scripts/functional_score.py --compare            # check the committed CSVs
    python3 scripts/functional_score.py --out eval           # rewrite them
    python3 scripts/functional_score.py --cq-rate llama70b=0.916 --cq-rate gemini=0.935 ...
        # model-level score with an externally measured CQ verification rate
        # (e.g. the Yes/No labels of the LLM-based CQ verification)
"""
from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DETAIL = "functional_eval_detail.csv"
RANKING = "functional_eval_ranking.csv"
W_UNSUPPORTED, W_SCENARIO_JUSTIFIED, SCALE = 1.0, 0.3, 0.5


def score(numb: float, numbnoscen: float, cq_rate: float) -> tuple[float, float, float]:
    penalty = W_UNSUPPORTED * numbnoscen + W_SCENARIO_JUSTIFIED * (numb - numbnoscen)
    manual = 1.0 / (1.0 + SCALE * penalty)
    return penalty, manual, 0.5 * manual + 0.5 * cq_rate


def load(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def recompute(rows: list[dict]) -> tuple[list[dict], list[dict]]:
    detail = []
    for r in rows:
        numb, nn, cq = float(r["numb"]), float(r["numbnoscen"]), float(r["cq_pass_rate"])
        pen, man, fun = score(numb, nn, cq)
        detail.append({"id": r["id"], "model": r["model"], "config": r["config"],
                       "numb": r["numb"], "numbnoscen": r["numbnoscen"],
                       "superfluous_penalty": round(pen, 4), "manual_score": round(man, 4),
                       "cq_pass_rate": r["cq_pass_rate"], "functional_score": round(fun, 4)})
    groups = defaultdict(list)
    for r in detail:
        groups[(r["model"], r["config"])].append(r)
    ranking = []
    for (model, config), rs in groups.items():
        mean = lambda k: sum(float(x[k]) for x in rs) / len(rs)  # noqa: E731
        ranking.append({"model": model, "config": config,
                        "avg_numb": round(mean("numb"), 2), "avg_numbnoscen": round(mean("numbnoscen"), 2),
                        "avg_manual_score": round(mean("manual_score"), 4),
                        "avg_cq_pass_rate": round(mean("cq_pass_rate"), 4),
                        "avg_functional_score": round(mean("functional_score"), 4)})
    ranking.sort(key=lambda r: r["avg_functional_score"], reverse=True)
    return detail, ranking


def compare(new: list[dict], old: list[dict], keys: list[str], cols: list[str], tol: float) -> int:
    index = {tuple(r[k] for k in keys): r for r in old}
    bad = 0
    for r in new:
        o = index.get(tuple(str(r[k]) for k in keys))
        if o is None:
            print(f"  missing row {[r[k] for k in keys]}")
            bad += 1
            continue
        for c in cols:
            if abs(float(r[c]) - float(o[c])) > tol:
                print(f"  {[r[k] for k in keys]} {c}: recomputed {r[c]} vs committed {o[c]}")
                bad += 1
    return bad


def write(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--detail", type=Path, default=REPO / "eval" / DETAIL, help="input with numb, numbnoscen, cq_pass_rate")
    p.add_argument("--out", type=Path, help="directory to write the two CSVs to")
    p.add_argument("--compare", action="store_true", help="compare with the CSVs next to --detail")
    p.add_argument("--cq-rate", action="append", default=[], metavar="MODEL=RATE",
                   help="model-level CQ verification rate to combine with the mean manual score")
    a = p.parse_args()

    detail, ranking = recompute(load(a.detail))
    status = 0
    if a.compare:
        d_bad = compare(detail, load(a.detail), ["id", "model", "config"],
                        ["superfluous_penalty", "manual_score", "functional_score"], 1e-4)
        r_bad = compare(ranking, load(a.detail.parent / RANKING), ["model", "config"],
                        ["avg_manual_score", "avg_cq_pass_rate", "avg_functional_score"], 1e-3)
        print(f"detail: {len(detail)} rows, {d_bad} mismatches; ranking: {len(ranking)} rows, {r_bad} mismatches")
        status = int(bool(d_bad or r_bad))
    if a.out:
        a.out.mkdir(parents=True, exist_ok=True)
        write(a.out / DETAIL, detail)
        write(a.out / RANKING, ranking)
        print(f"wrote {a.out / DETAIL} and {a.out / RANKING}")
    for spec in a.cq_rate:
        model, rate = spec.split("=")
        for r in ranking:
            if r["model"] == model:
                fs = 0.5 * r["avg_manual_score"] + 0.5 * float(rate)
                print(f"{model:10s} {r['config']:26s} manual={r['avg_manual_score']:.4f} cq={float(rate):.3f} functional={fs:.3f}")
    if not (a.compare or a.out or a.cq_rate):
        for r in ranking:
            print(r)
    return status


if __name__ == "__main__":
    sys.exit(main())
