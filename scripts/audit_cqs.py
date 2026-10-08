#!/usr/bin/env python3
"""audit_cqs.py -- read-only, fully offline provenance auditor for the 175 competency
questions in data/scenarios/pattern_scenarios.json.

Why this exists
---------------
The CQ-evaluation fix (F4) replaced a broken prompt.txt scraper with "read cq_list
from data/scenarios/pattern_scenarios.json".  That is only a fix if cq_list is clean,
and nobody ever checked.  It is not clean: scenario 2023-135-01 is the
RoleDependentNames / author-pseudonym pattern (CQ #1 asks under what pseudonym
C. S. Lewis published a collection of poems) yet its CQ #3 is "Show the trajectories
of rivers which cross national parks?" -- plainly harvested from a different pattern.

This script audits every CQ against its own scenario using only signals computable
offline.  It calls no LLM and no network service.  It NEVER edits, reorders or
deletes anything: it is an auditor, not a cleaner.  Deciding what to do about a
flagged CQ is a human call.

The anchoring rule is MONOTONE in contamination
-----------------------------------------------
An earlier version of this auditor anchored each CQ partly against its SIBLING CQs.
That made it anti-monotone: injecting a SECOND foreign question into a scenario
anchored the FIRST one, its verdict fell back from almost_certainly_misfiled to
suspicious, and the CI gate flipped from red to green.  More contamination produced
less detection, which means a low misfiled count could be an artefact of the
contamination being LARGE rather than small -- exactly the wrong failure direction
for a number that is quoted as a measurement.

The rule below therefore anchors every CQ only against signals that are INDEPENDENT
of the CQ list being audited:

  * the scenario_text, and
  * the pattern vocabulary (the ontology URL slug plus, when
    data/ground_truth/<scenario_id>.{ttl,owl,rdf,...} is cached, the local names and
    short literals declared by the ground-truth ontology),

for the CQ's own scenario and, for the comparison signal, for every other scenario.
No verdict reads any cq_list.  Consequences:

  * appending CQs to a scenario can never SOFTEN an existing CQ's verdict.  It
    is not true that the verdict cannot change at all: island membership grows
    as CQs are added, so a verdict can become more severe.  Monotonicity here
    means one-directional, not fixed;
  * the flagged set and the exit code are monotone non-decreasing in contamination.

Sibling CQs are still used, but only in the two directions that keep monotonicity:

  * CORROBORATION may lower a CQ's numeric suspicion, never its verdict, and only
    when the corroborating sibling is itself well anchored at home AND the shared
    word is itself part of the home vocabulary.  A CQ with zero home anchoring can
    therefore never be corroborated by anything.  Sibling agreement downgrades
    suspicion; it never exonerates.
  * CONNECTED COMPONENTS make a coherent island of foreign CQs MORE damning, not
    less.  Weakly anchored CQs of one scenario are joined when they share a
    non-generic content word that is absent from that scenario's home vocabulary.
    A component of two or more such CQs is a harvested block, and every member is
    promoted to almost_certainly_misfiled.  Components can only grow when CQs are
    added, so this signal is monotone too.

Signals (all lexical, all deterministic)
----------------------------------------
  home support       fraction of the CQ's content words found in its own scenario's
                     scenario_text or pattern vocabulary.  The primary signal.
  best-fit elsewhere the same fraction recomputed against every OTHER scenario's
                     scenario_text and pattern vocabulary.  A CQ anchored much
                     better somewhere else is positive evidence of misfiling rather
                     than merely of odd wording, and it is the signal that carries
                     most of the recall.
  foreign island     connected component of weakly anchored CQs of one scenario
                     agreeing on vocabulary foreign to that scenario.
  corroboration      well-anchored siblings sharing home vocabulary; lowers the
                     numeric suspicion only.
  near duplicates    CQs repeated (Jaccard >= 0.80 on content words) under two
                     different scenarios.
  boilerplate        prompt/output-format instructions masquerading as questions.
  malformed          empty, whitespace-only, non-string, or zero-content-word entries.

Verdicts
--------
  clean                       well anchored at home, no other flag
  suspicious                  weakly anchored, duplicated, boilerplate, anchored only
                              by generic modelling words, or somewhat better anchored
                              under a different scenario
  almost_certainly_misfiled   >= 3 content words and either NO home anchoring at all,
                              or a large better-fit margin under another scenario, or
                              membership of a foreign island of >= 2 CQs
  malformed                   structurally unusable entry

Measured behaviour on this corpus (see tests/test_audit_cqs.py)
---------------------------------------------------------------
Recall is measured by injecting every real CQ of every scenario, one at a time, into
every other scenario -- 2067 lone-intruder injections -- and asking whether the
auditor calls the injected question contaminated.  The numbers this rule scores are
reported by test_cross_pattern_intruder_recall and are recorded under
"injection_experiment" in the JSON report when --measure-recall is passed.

Exit status
-----------
  0  no almost_certainly_misfiled and no malformed CQ  (suspicious CQs do not gate)
  1  at least one almost_certainly_misfiled or malformed CQ -- suitable for CI
  2  usage / IO error

Usage
-----
  python scripts/audit_cqs.py
  python scripts/audit_cqs.py --scenarios data/scenarios/pattern_scenarios.json \
                              --out data/scenarios/cq_audit_report.json
  python scripts/audit_cqs.py --show clean        # print every CQ, not just flagged
  python scripts/audit_cqs.py --measure-recall    # run the injection experiment too
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys

DEFAULT_SCENARIOS = os.path.join("data", "scenarios", "pattern_scenarios.json")
DEFAULT_OUT = os.path.join("data", "scenarios", "cq_audit_report.json")
DEFAULT_GT_DIR = os.path.join("data", "ground_truth")

# Words carrying no topical signal: function words plus the interrogative /
# imperative frame every competency question is built out of.
STOPWORDS = {
    "a", "about", "above", "after", "all", "already", "also", "am", "an", "and", "any",
    "anything", "are", "as", "at", "available", "back", "be", "because", "been", "before",
    "being", "below", "between", "both", "but", "by", "can", "cannot", "could", "did",
    "do", "does", "doing", "done", "down", "during", "each", "either", "else", "etc",
    "even", "ever", "every", "example", "far", "few", "for", "from", "further", "get",
    "give", "given", "go", "had", "has", "have", "having", "he", "her", "here", "hers",
    "herself", "him", "himself", "his", "how", "however", "i", "if", "in", "include",
    "included", "includes", "including", "into", "is", "it", "its", "itself", "just",
    "kind", "kinds", "know", "known", "let", "like", "list", "long", "made", "make",
    "many", "may", "me", "might", "more", "most", "much", "must", "my", "myself", "need",
    "needs", "no", "nor", "not", "now", "of", "off", "on", "once", "one", "only", "or",
    "other", "others", "otherwise", "our", "ours", "out", "over", "own", "part", "per",
    "possible", "provide", "provided", "return", "returns", "same", "say", "see", "shall",
    "she", "should", "show", "showing", "shown", "shows", "since", "so", "some", "such",
    "take", "taken", "than", "that", "the", "their", "theirs", "them", "themselves",
    "then", "there", "therefore", "these", "they", "this", "those", "through", "to",
    "too", "under", "until", "up", "upon", "us", "use", "used", "uses", "using", "very",
    "via", "was", "we", "were", "what", "whats", "when", "where", "whether", "which",
    "while", "who", "whom", "whose", "why", "will", "with", "within", "without", "would",
    "you", "your", "yours",
}

# Content words that are generic across ODP modelling questions.  They still count as
# content (dropping them would over-flag), but a CQ anchored ONLY by these is weakly
# anchored and is reported as such, and they are never allowed to join two CQs into a
# "foreign island": agreeing on the word "information" is not evidence of a shared
# provenance.
GENERIC_MODELING = {
    "ontology", "ontolog", "pattern", "class", "classe", "property", "propertie",
    "represent", "model", "express", "describe", "description", "specify", "specific",
    "information", "instance", "entity", "entitie", "type", "kind", "value", "attribute",
    "relation", "relationship", "concept", "term", "axiom", "individual",
}

BOILERPLATE_PATTERNS = [
    (r"```", "contains a markdown code fence"),
    (r"\bturtle\b", "mentions the Turtle serialisation"),
    (r"\bowl\s*:", "contains an OWL qname"),
    (r"\brdfs?\s*:", "contains an RDF(S) qname"),
    (r"@prefix", "contains a Turtle prefix declaration"),
    (r"\bcompetency question", "refers to competency questions themselves"),
    (r"\bmodeling constraint|\bmodelling constraint", "is a modelling-constraint bullet"),
    (r"\boutput (only|the|a)\b", "is an output-format instruction"),
    (r"\bdo not\b.*\b(explain|output|include|add)\b", "is a prohibition instruction"),
    (r"\byou are\b|\bas an ai\b|\byour task\b", "is prompt persona text"),
    (r"\bscenario\b.*\bfollow", "is prompt framing text"),
    (r"^\s*(generate|produce|write|create|return)\b.*\b(ontology|owl|rdf|ttl)\b",
     "instructs the model to emit an ontology"),
]

# --------------------------------------------------------------------------------
# tokenisation
# --------------------------------------------------------------------------------

_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9]*")
_CAMEL_RE = re.compile(r"[A-Z]+(?![a-z])|[A-Z][a-z0-9]*|[a-z0-9]+")


def stem(word: str) -> str:
    """Deliberately crude, order-dependent suffix stripping.

    Only needs to make morphological variants of the SAME word collide
    (names/name, trajectories/trajectory, published/publish).  It is not a
    linguistic stemmer and does not need to be one.
    """
    w = word.lower()
    if w.endswith("'s"):
        w = w[:-2]
    if len(w) > 5 and w.endswith("ies"):
        return w[:-3] + "y"
    if len(w) > 5 and (w.endswith("ches") or w.endswith("shes") or w.endswith("sses")
                       or w.endswith("xes")):
        return w[:-2]
    if len(w) >= 5 and w.endswith("ing"):
        return w[:-3]
    if len(w) >= 5 and w.endswith("ed") and not w.endswith("eed"):
        return w[:-2]
    if len(w) >= 4 and w.endswith("s") and not w.endswith(("ss", "us", "is", "as")):
        return w[:-1]
    return w


def content_tokens(text: str) -> list:
    """Ordered, de-duplicated content stems of `text`."""
    out, seen = [], set()
    for raw in _TOKEN_RE.findall(text or ""):
        low = raw.lower()
        if len(low) < 3 or low in STOPWORDS:
            continue
        s = stem(low)
        if len(s) < 3 or s in STOPWORDS:
            continue
        if s not in seen:
            seen.add(s)
            out.append(s)
    return out


def split_identifier(name: str) -> list:
    """RoleDependentNames -> [role, dependent, name]; data-request -> [data, request]."""
    parts = []
    for chunk in re.split(r"[^A-Za-z0-9]+", name or ""):
        parts.extend(_CAMEL_RE.findall(chunk))
    return [stem(p.lower()) for p in parts if len(p) >= 3]


# --------------------------------------------------------------------------------
# pattern vocabulary
# --------------------------------------------------------------------------------

_LOCALNAME_RE = re.compile(r"[#/]([A-Za-z][A-Za-z0-9_]{2,60})")
_LITERAL_RE = re.compile(r'"([^"\\]{3,120})"')
_GT_EXTENSIONS = (".ttl", ".owl", ".rdf", ".xml", ".n3", ".nt", ".jsonld")
_GT_READ_CAP = 16 * 1024 * 1024  # bytes; the largest ground truth here is ~15 MB


def ontology_url_vocab(url: str) -> set:
    if not url:
        return set()
    tail = url.rstrip("/").split("/")[-1]
    tail = re.sub(r"\.(ttl|owl|rdf|xml|n3|nt|jsonld)$", "", tail, flags=re.I)
    vocab = set(split_identifier(tail))
    return {v for v in vocab if v not in STOPWORDS and len(v) >= 3}


_GT_CACHE = {}


def ground_truth_vocab(scenario_id: str, gt_dir: str):
    """Local names + short literals declared by the locally cached ground-truth file.

    Memoised on (path, mtime, size).  The largest ground truth in the corpus is
    ~15 MB and re-reading it dominates the runtime of the injection experiments in
    tests/test_audit_cqs.py.  The cache is a pure speed-up: the key includes the
    file's mtime and size, so an edited ground truth is re-read.
    """
    if not gt_dir or not os.path.isdir(gt_dir):
        return set(), None
    for ext in _GT_EXTENSIONS:
        path = os.path.join(gt_dir, scenario_id + ext)
        if os.path.isfile(path):
            break
    else:
        return set(), None
    try:
        st = os.stat(path)
        key = (os.path.abspath(path), st.st_mtime_ns, st.st_size)
    except OSError:
        return set(), None
    hit = _GT_CACHE.get(key)
    if hit is not None:
        return set(hit[0]), hit[1]
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            text = fh.read(_GT_READ_CAP)
    except OSError:
        return set(), None
    vocab = set()
    for local in _LOCALNAME_RE.findall(text):
        vocab.update(split_identifier(local))
    for lit in _LITERAL_RE.findall(text):
        vocab.update(content_tokens(lit))
    vocab = {v for v in vocab if len(v) >= 3 and v not in STOPWORDS}
    _GT_CACHE[key] = (frozenset(vocab), os.path.basename(path))
    return vocab, os.path.basename(path)


# --------------------------------------------------------------------------------
# thresholds
# --------------------------------------------------------------------------------

MISFILED_MIN_TOKENS = 3          # below this, zero overlap is not informative

# Home anchoring.  Calibrated against the corpus: the median CQ anchors 0.75 of its
# content words at home, and exactly one CQ (the known river intruder) anchors none.
WEAK_HOME_SUPPORT = 0.25         # strictly below this -> 'weak_home_anchor'

# Better fit elsewhere.  This is the signal that carries the recall: an intruder
# harvested from another pattern is, by construction, anchored there.
BETTER_FIT_SOFT_MARGIN = 0.20    # >= this -> suspicious
BETTER_FIT_MARGIN = 0.34         # >= this (with the two guards below) -> misfiled
BETTER_FIT_MIN_ELSEWHERE = 0.30  # the other scenario must actually anchor it
BETTER_FIT_HOME_CEILING = 0.60   # a CQ well anchored at home is never called misfiled

# Foreign islands (connected components of mutually agreeing, weakly anchored CQs).
ISLAND_HOME_CEILING = 0.34
ISLAND_MIN_SIZE = 2

# Corroboration by well-anchored siblings.  Lowers the numeric suspicion only.
CORROBORATION_MIN_HOME = 0.50
CORROBORATION_MAX_DISCOUNT = 0.15

NEAR_DUPLICATE_JACCARD = 0.80
SIMILARITY_REPORT_FLOOR = 0.40   # merely listed as evidence, never a flag on its own

# A ground-truth ontology that ships a large ABox contributes tens of thousands of
# vocabulary terms and will anchor almost any CQ, so a misfiled CQ under such a
# scenario can be masked.  Report that explicitly rather than pretend otherwise.
VOCAB_DILUTION_LIMIT = 5000

GATING_VERDICTS = ("almost_certainly_misfiled", "malformed")

_VERDICT_RANK = {"malformed": 0, "almost_certainly_misfiled": 1, "suspicious": 2,
                 "clean": 3}


def _jaccard(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / float(len(a | b))


# --------------------------------------------------------------------------------
# audit
# --------------------------------------------------------------------------------


def build_contexts(scenarios, ground_truth_dir=DEFAULT_GT_DIR):
    """Per-scenario anchoring context.  `home_vocab` never contains any CQ text."""
    ctx = []
    for sc in scenarios:
        sid = sc.get("scenario_id")
        scen_tokens = set(content_tokens(sc.get("scenario_text", "")))
        url_vocab = ontology_url_vocab(sc.get("ontology", ""))
        gt_vocab, gt_file = ground_truth_vocab(sid, ground_truth_dir)
        pattern_vocab = url_vocab | gt_vocab
        raw_cqs = sc.get("cq_list", []) or []
        ctx.append(
            {
                "sc": sc,
                "sid": sid,
                "scen_tokens": scen_tokens,
                "url_vocab": url_vocab,
                "gt_vocab": gt_vocab,
                "gt_file": gt_file,
                "pattern_vocab": pattern_vocab,
                "home_vocab": scen_tokens | pattern_vocab,
                "raw_cqs": raw_cqs,
                "cq_tokens": [
                    set(content_tokens(c)) if isinstance(c, str) else set()
                    for c in raw_cqs
                ],
            }
        )
    return ctx


def home_support(tokens, c):
    """How much of `tokens` scenario `c` anchors, using NO CQ text whatsoever."""
    if not tokens:
        return {
            "matched_scenario_text": [], "matched_pattern_vocabulary": [],
            "unanchored_content_words": [], "home_support": 0.0,
        }
    matched_scen = sorted(tokens & c["scen_tokens"])
    matched_pat = sorted(tokens & c["pattern_vocab"])
    anchored = set(matched_scen) | set(matched_pat)
    return {
        "matched_scenario_text": matched_scen,
        "matched_pattern_vocabulary": matched_pat,
        "unanchored_content_words": sorted(tokens - anchored),
        "home_support": round(len(anchored) / float(len(tokens)), 3),
    }


def _foreign_islands(c):
    """Connected components of weakly anchored CQs agreeing on foreign vocabulary.

    Nodes  CQs of this scenario whose home support is <= ISLAND_HOME_CEILING.
    Edges  two nodes share at least one content word that is (a) absent from the
           scenario's home vocabulary and (b) not a generic modelling word.

    A component of >= ISLAND_MIN_SIZE members is a harvested block.  Because adding
    CQs can only add nodes and edges, components never shrink -- so this signal is
    monotone in contamination, which is the whole point.
    """
    home = c["home_vocab"]
    nodes = []
    for i, toks in enumerate(c["cq_tokens"]):
        if not toks:
            continue
        support = len(toks & home) / float(len(toks))
        if support <= ISLAND_HOME_CEILING:
            foreign = {t for t in toks - home if t not in GENERIC_MODELING}
            if foreign:
                nodes.append((i, foreign))

    parent = {i: i for i, _ in nodes}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    shared = {}
    for a in range(len(nodes)):
        for b in range(a + 1, len(nodes)):
            common = nodes[a][1] & nodes[b][1]
            if common:
                parent[find(nodes[a][0])] = find(nodes[b][0])
                shared.setdefault(nodes[a][0], set()).update(common)
                shared.setdefault(nodes[b][0], set()).update(common)

    groups = {}
    for i, _ in nodes:
        groups.setdefault(find(i), []).append(i)

    islands = {}
    for members in groups.values():
        if len(members) >= ISLAND_MIN_SIZE:
            members = sorted(members)
            for i in members:
                islands[i] = {
                    "members": members,
                    "shared_foreign_words": sorted(shared.get(i, set())),
                }
    return islands


def _corroborating_siblings(i, toks, c, home_ratio):
    """Well-anchored siblings sharing HOME vocabulary with this CQ.

    Only ever lowers the numeric suspicion.  Because the shared word must itself be
    part of the home vocabulary, a CQ with zero home anchoring can never be
    corroborated: sibling agreement downgrades suspicion, it never exonerates.
    """
    if home_ratio <= 0.0:
        return []
    home = c["home_vocab"]
    mine = toks & home
    out = []
    for j, other in enumerate(c["cq_tokens"]):
        if j == i or not other:
            continue
        if len(other & home) / float(len(other)) < CORROBORATION_MIN_HOME:
            continue
        common = sorted(w for w in (mine & other) if w not in GENERIC_MODELING)
        if common:
            out.append({"index": j, "shared_home_words": common})
    return out


def audit_corpus(scenarios, ground_truth_dir=DEFAULT_GT_DIR):
    """Audit every CQ of every scenario.  Pure function: reads, never writes.

    Every verdict is a function of the CQ's own text and of scenario-level context
    that contains no CQ text, so appending CQs can never SOFTEN an existing CQ's
    verdict.  It can harden one: `foreign_cq_island` membership grows as CQs are
    added.  That is the intended direction -- more contamination must not yield
    less detection -- but it is one-directional monotonicity, not invariance, and
    an island alone never convicts (see the verdict rule).  The corroboration
    discount touches only the numeric suspicion, never the verdict.
    """
    ctx = build_contexts(scenarios, ground_truth_dir)

    # ---- cross-scenario near duplicates -----------------------------------
    flat = []
    for c in ctx:
        for i, toks in enumerate(c["cq_tokens"]):
            if toks:
                flat.append((c["sid"], i, c["raw_cqs"][i], toks))
    near_dupes = []
    similarities = []
    dupe_index = {}
    for a in range(len(flat)):
        for b in range(a + 1, len(flat)):
            if flat[a][0] == flat[b][0]:
                continue
            j = _jaccard(flat[a][3], flat[b][3])
            if j >= SIMILARITY_REPORT_FLOOR:
                similarities.append(
                    {
                        "jaccard": round(j, 3),
                        "flagged": j >= NEAR_DUPLICATE_JACCARD,
                        "a": {"scenario_id": flat[a][0], "index": flat[a][1], "text": flat[a][2]},
                        "b": {"scenario_id": flat[b][0], "index": flat[b][1], "text": flat[b][2]},
                    }
                )
            if j >= NEAR_DUPLICATE_JACCARD:
                near_dupes.append(
                    {
                        "jaccard": round(j, 3),
                        "a": {"scenario_id": flat[a][0], "index": flat[a][1], "text": flat[a][2]},
                        "b": {"scenario_id": flat[b][0], "index": flat[b][1], "text": flat[b][2]},
                    }
                )
                dupe_index.setdefault((flat[a][0], flat[a][1]), []).append(
                    (flat[b][0], flat[b][1], round(j, 3))
                )
                dupe_index.setdefault((flat[b][0], flat[b][1]), []).append(
                    (flat[a][0], flat[a][1], round(j, 3))
                )
    near_dupes.sort(key=lambda d: -d["jaccard"])
    similarities.sort(key=lambda d: -d["jaccard"])

    # ---- per CQ ------------------------------------------------------------
    out_scenarios = []
    totals = {"clean": 0, "suspicious": 0, "almost_certainly_misfiled": 0, "malformed": 0}
    n_cqs = 0

    for c in ctx:
        seen_exact = {}
        cq_rows = []
        counts = {"clean": 0, "suspicious": 0, "almost_certainly_misfiled": 0, "malformed": 0}
        islands = _foreign_islands(c)

        for i, raw in enumerate(c["raw_cqs"]):
            n_cqs += 1
            flags = []

            if not isinstance(raw, str) or not raw.strip():
                flags.append(
                    {"code": "empty_or_non_string",
                     "detail": "cq_list entry is %s" % ("blank" if isinstance(raw, str)
                                                        else type(raw).__name__)}
                )
                cq_rows.append(
                    {"index": i, "text": raw if isinstance(raw, str) else "",
                     "verdict": "malformed", "suspicion": 1.0, "flags": flags,
                     "evidence": {}}
                )
                counts["malformed"] += 1
                totals["malformed"] += 1
                continue

            text = raw.strip()
            toks = c["cq_tokens"][i]

            if not toks:
                flags.append({"code": "no_content_words",
                              "detail": "entry has no scorable content word: %r" % text})
                cq_rows.append({"index": i, "text": text, "verdict": "malformed",
                                "suspicion": 1.0, "flags": flags,
                                "evidence": {"content_words": []}})
                counts["malformed"] += 1
                totals["malformed"] += 1
                continue

            ev = home_support(toks, c)
            ratio = ev["home_support"]
            evidence = dict(ev)
            evidence["content_words"] = sorted(toks)
            evidence["pattern_vocabulary_source"] = {
                "ontology_url_terms": sorted(c["url_vocab"]),
                "ground_truth_file": c["gt_file"],
                "ground_truth_vocabulary_size": len(c["gt_vocab"]),
            }

            # -- best fit under a different scenario (uses no CQ text either) --
            best = None
            for other in ctx:
                if other is c:
                    continue
                r = home_support(toks, other)["home_support"]
                if best is None or r > best[1]:
                    best = (other["sid"], r)
            best_sid, best_ratio = best if best else (None, 0.0)
            delta = round(best_ratio - ratio, 3)
            evidence["best_other_scenario"] = (
                {"scenario_id": best_sid, "home_support": best_ratio, "margin": delta}
                if best else None
            )

            # -- sibling-derived evidence (never exonerating) -----------------
            island = islands.get(i)
            evidence["foreign_cq_island"] = island
            corroborators = _corroborating_siblings(i, toks, c, ratio)
            evidence["corroborating_siblings"] = corroborators

            # -- flags ---------------------------------------------------------
            low = text.lower()
            for pat, why in BOILERPLATE_PATTERNS:
                if re.search(pat, low):
                    flags.append({"code": "prompt_boilerplate",
                                  "detail": "text %s (matched /%s/)" % (why, pat)})
                    break

            key = re.sub(r"\W+", " ", low).strip()
            if key in seen_exact:
                flags.append({"code": "duplicate_within_scenario",
                              "detail": "identical to CQ #%d of the same scenario"
                                        % (seen_exact[key] + 1)})
            else:
                seen_exact[key] = i

            for other_sid, other_i, j in dupe_index.get((c["sid"], i), []):
                flags.append(
                    {"code": "duplicate_across_scenarios",
                     "detail": "Jaccard %.2f with %s CQ #%d" % (j, other_sid, other_i + 1)}
                )

            if ratio == 0.0 and len(toks) >= MISFILED_MIN_TOKENS:
                flags.append(
                    {"code": "no_home_anchor",
                     "detail": "none of the %d content words (%s) occurs in this "
                               "scenario's text or pattern vocabulary"
                               % (len(toks), ", ".join(sorted(toks)))}
                )
            elif ratio < WEAK_HOME_SUPPORT:
                flags.append(
                    {"code": "weak_home_anchor",
                     "detail": "only %d/%d content words anchored (%s); unanchored: %s"
                               % (len(toks) - len(ev["unanchored_content_words"]),
                                  len(toks),
                                  ", ".join(sorted(set(ev["matched_scenario_text"])
                                                   | set(ev["matched_pattern_vocabulary"])))
                                  or "-",
                                  ", ".join(ev["unanchored_content_words"]))}
                )

            anchored_terms = (set(ev["matched_scenario_text"])
                              | set(ev["matched_pattern_vocabulary"]))
            if anchored_terms and anchored_terms <= GENERIC_MODELING:
                flags.append({"code": "anchored_only_by_generic_terms",
                              "detail": "sole anchors are generic modelling words: %s"
                                        % ", ".join(sorted(anchored_terms))})

            strong_better_fit = (
                delta >= BETTER_FIT_MARGIN
                and best_ratio >= BETTER_FIT_MIN_ELSEWHERE
                and ratio <= BETTER_FIT_HOME_CEILING
            )
            if delta >= BETTER_FIT_SOFT_MARGIN:
                flags.append(
                    {"code": "better_fit_elsewhere",
                     "detail": "home support %.2f vs %.2f under %s (margin %.2f)%s"
                               % (ratio, best_ratio, best_sid, delta,
                                  "" if strong_better_fit else " -- below the "
                                  "misfiling margin, reported as evidence only")}
                )
            if island:
                flags.append(
                    {"code": "foreign_cq_island",
                     "detail": "one of %d weakly anchored CQs (#%s) of this scenario "
                               "agreeing on vocabulary foreign to it: %s"
                               % (len(island["members"]),
                                  ", ".join("%d" % (m + 1) for m in island["members"]),
                                  ", ".join(island["shared_foreign_words"]))}
                )

            codes = {f["code"] for f in flags}

            # -- verdict -------------------------------------------------------
            # Every disjunct below reads only `toks`, scenario-level context, and
            # the island membership (which only grows).  None can be undone by
            # adding CQs, so the verdict is monotone in contamination.
            # `foreign_cq_island` is corroborating evidence, never a sufficient
            # one on its own.  Island membership is reachable by a CQ that is
            # merely thinly anchored (home_support <= ISLAND_HOME_CEILING, not
            # zero) and that happens to share one ordinary noun with an
            # intruder, so promoting on it alone convicts innocent questions:
            # 2023-135-01 #1, a genuine question about the pseudonym C. S. Lewis
            # published a *collection* of poems under, is flagged the moment an
            # unrelated CQ mentioning an "asset collection" is appended.  An
            # island only convicts alongside an independent signal.
            misfiled = (
                len(toks) >= MISFILED_MIN_TOKENS
                and "prompt_boilerplate" not in codes
                and (
                    "no_home_anchor" in codes
                    or strong_better_fit
                )
            )
            if misfiled:
                verdict = "almost_certainly_misfiled"
            elif codes:
                verdict = "suspicious"
            else:
                verdict = "clean"

            # -- suspicion score -----------------------------------------------
            score = (1.0 - ratio) * 0.5
            score += 0.25 if "no_home_anchor" in codes else 0.0
            score += 0.20 if strong_better_fit else 0.0
            score += 0.10 if "better_fit_elsewhere" in codes else 0.0
            if island:
                score += 0.15 + 0.05 * min(len(island["members"]) - ISLAND_MIN_SIZE, 4)
            score += 0.15 if "prompt_boilerplate" in codes else 0.0
            score += 0.10 if "duplicate_across_scenarios" in codes else 0.0
            score += 0.10 if "duplicate_within_scenario" in codes else 0.0
            score += 0.05 if "anchored_only_by_generic_terms" in codes else 0.0
            # Corroboration by well-anchored siblings downgrades suspicion.  It can
            # never reach zero-anchored CQs (see _corroborating_siblings) and never
            # touches the verdict.
            if corroborators:
                score -= min(CORROBORATION_MAX_DISCOUNT, 0.05 * len(corroborators))
            suspicion = round(min(1.0, max(0.0, score)), 3)

            counts[verdict] += 1
            totals[verdict] += 1
            cq_rows.append({"index": i, "text": text, "verdict": verdict,
                            "suspicion": suspicion, "flags": flags, "evidence": evidence})

        out_scenarios.append(
            {
                "scenario_id": c["sid"],
                "ontology": c["sc"].get("ontology"),
                "scenario_text_content_words": len(c["scen_tokens"]),
                "pattern_vocabulary_terms": len(c["pattern_vocab"]),
                "home_vocabulary_terms": len(c["home_vocab"]),
                "ground_truth_file": c["gt_file"],
                "n_cqs": len(cq_rows),
                "counts": counts,
                "foreign_cq_islands": [
                    list(t) for t in sorted({tuple(v["members"])
                                             for v in islands.values()})
                ],
                "vocabulary_dilution_warning": (
                    len(c["pattern_vocab"]) > VOCAB_DILUTION_LIMIT
                ),
                "cqs": cq_rows,
            }
        )

    totals["scenarios"] = len(scenarios)
    totals["cqs"] = n_cqs
    ranked = []
    for sc in out_scenarios:
        for cq in sc["cqs"]:
            if cq["verdict"] != "clean":
                ranked.append({"scenario_id": sc["scenario_id"], "index": cq["index"],
                               "verdict": cq["verdict"], "suspicion": cq["suspicion"],
                               "text": cq["text"],
                               "flags": [f["code"] for f in cq["flags"]]})
    ranked.sort(key=lambda r: (-r["suspicion"], _VERDICT_RANK[r["verdict"]],
                               r["scenario_id"], r["index"]))

    return {
        "tool": "scripts/audit_cqs.py",
        "read_only": True,
        "offline": True,
        "source": None,
        "ground_truth_dir": ground_truth_dir,
        "decision_rule": (
            "monotone: every verdict is a function of the CQ text and of "
            "scenario-level context (scenario_text + pattern vocabulary) that "
            "contains no CQ text.  Appending CQs to a scenario cannot un-flag an "
            "existing CQ.  Sibling CQs may only lower a numeric suspicion "
            "(corroboration) or raise it (foreign_cq_island); they never exonerate."
        ),
        "thresholds": {
            "misfiled_min_content_words": MISFILED_MIN_TOKENS,
            "weak_home_support": WEAK_HOME_SUPPORT,
            "better_fit_soft_margin": BETTER_FIT_SOFT_MARGIN,
            "better_fit_margin": BETTER_FIT_MARGIN,
            "better_fit_min_elsewhere": BETTER_FIT_MIN_ELSEWHERE,
            "better_fit_home_ceiling": BETTER_FIT_HOME_CEILING,
            "island_home_ceiling": ISLAND_HOME_CEILING,
            "island_min_size": ISLAND_MIN_SIZE,
            "corroboration_min_home": CORROBORATION_MIN_HOME,
            "near_duplicate_jaccard": NEAR_DUPLICATE_JACCARD,
        },
        "verdict_definitions": {
            "clean": "at least %.0f%% of its content words are anchored in the "
                     "scenario text or the pattern vocabulary, and no other flag fired"
                     % (WEAK_HOME_SUPPORT * 100),
            "suspicious": "weakly anchored, duplicated, boilerplate-shaped, anchored "
                          "only by generic modelling words, or somewhat better "
                          "anchored under a different scenario -- needs a human look, "
                          "not necessarily wrong",
            "almost_certainly_misfiled": "at least %d content words and either none of "
                                         "them anchored at home, or a margin of >= "
                                         "%.2f in favour of another scenario, or "
                                         "membership of a foreign island of >= %d CQs"
                                         % (MISFILED_MIN_TOKENS, BETTER_FIT_MARGIN,
                                            ISLAND_MIN_SIZE),
            "malformed": "empty, whitespace-only, non-string, or no scorable content word",
        },
        "exit_code_contract": "1 if any CQ is almost_certainly_misfiled or malformed, "
                              "else 0; suspicious CQs do not gate CI",
        "totals": totals,
        "ranked_by_suspicion": ranked,
        "cross_scenario_near_duplicates": near_dupes,
        "top_cross_scenario_similarities": similarities[:25],
        "limitations": [
            "Purely lexical.  A CQ that is topically wrong but shares surface "
            "vocabulary with its scenario (e.g. a policy question misfiled between "
            "the three 2023-134-* policy patterns) still cannot be detected this way; "
            "the measured recall below quantifies how often that happens.",
            "The better-fit signal asks whether the CQ is anchored better under some "
            "OTHER scenario OF THIS CORPUS.  An intruder harvested from a pattern "
            "that is not in the corpus has no such home and can only be caught by the "
            "home-anchoring and island signals, so the measured recall is an upper "
            "bound for out-of-corpus contamination.",
            "Scenarios whose ground-truth ontology ships a large ABox contribute a "
            "huge vocabulary that anchors nearly anything; those scenarios carry "
            "vocabulary_dilution_warning=true and their clean verdicts are weaker.",
            "'suspicious' means 'weakly anchored', not 'contaminated'.  Only "
            "'almost_certainly_misfiled' is a claim of contamination.",
            "This tool never edits the corpus.  Removing or re-assigning a flagged CQ "
            "is a human decision.",
        ],
        "scenarios": out_scenarios,
    }


# --------------------------------------------------------------------------------
# injection experiment (recall measurement)
# --------------------------------------------------------------------------------


def gating_set(report):
    """(scenario_id, index) of every CQ whose verdict makes CI fail."""
    return {
        (sc["scenario_id"], cq["index"])
        for sc in report["scenarios"]
        for cq in sc["cqs"]
        if cq["verdict"] in GATING_VERDICTS
    }


def measure_injection_recall(scenarios, ground_truth_dir=DEFAULT_GT_DIR,
                             known_contaminated=()):
    """Inject every real CQ of every scenario into every other scenario, alone.

    Injections are batched so that any single audit sees at most ONE intruder per
    scenario.  That measures the LONE-intruder recall: the connected-component
    signal cannot help, which is the hard case and the honest one to quote.

    Returns recall, precision, and the verdicts the auditor already reaches on the
    uncontaminated corpus (the river CQ among them is a genuine intruder, not a
    false positive, so `known_contaminated` is subtracted before the false-positive
    rate is computed).
    """
    import copy as _copy

    sids = [s.get("scenario_id") for s in scenarios]
    queues = {}
    for tsid in sids:
        q = []
        for src in scenarios:
            if src.get("scenario_id") == tsid:
                continue
            for i, c in enumerate(src.get("cq_list") or []):
                if isinstance(c, str) and len(content_tokens(c)) >= MISFILED_MIN_TOKENS:
                    q.append((src.get("scenario_id"), i, c))
        queues[tsid] = q

    base = audit_corpus(scenarios, ground_truth_dir=ground_truth_dir)
    native = sorted(gating_set(base))
    known = {tuple(k) for k in known_contaminated}
    native_fp = [n for n in native if n not in known]

    rounds = max((len(q) for q in queues.values()), default=0)
    tp = fn = fp_occurrences = 0
    missed = []
    for r in range(rounds):
        contaminated = _copy.deepcopy(scenarios)
        injected = {}
        for s in contaminated:
            q = queues[s.get("scenario_id")]
            if r < len(q):
                src_sid, src_i, text = q[r]
                injected[s.get("scenario_id")] = (len(s["cq_list"]), src_sid, src_i, text)
                s["cq_list"] = list(s["cq_list"]) + [text]
        if not injected:
            continue
        rep = audit_corpus(contaminated, ground_truth_dir=ground_truth_dir)
        gating = gating_set(rep)
        inj_ids = {(t, v[0]) for t, v in injected.items()}
        for tsid, (idx, src_sid, src_i, text) in injected.items():
            if (tsid, idx) in gating:
                tp += 1
            else:
                fn += 1
                missed.append({"source_scenario": src_sid, "source_index": src_i,
                               "injected_into": tsid, "text": text})
        fp_occurrences += len(gating - inj_ids - known)

    total = tp + fn
    return {
        "design": "every real CQ injected alone into every other scenario",
        "injections": total,
        "true_positives": tp,
        "false_negatives": fn,
        "recall": round(tp / float(total), 4) if total else 0.0,
        "precision": (round(tp / float(tp + fp_occurrences), 4)
                      if (tp + fp_occurrences) else 0.0),
        "false_positive_occurrences": fp_occurrences,
        "native_gating_verdicts": [list(n) for n in native],
        "native_false_positives": [list(n) for n in native_fp],
        "native_false_positive_rate": (round(len(native_fp) / float(base["totals"]["cqs"]), 4)
                                       if base["totals"]["cqs"] else 0.0),
        "missed_examples": missed[:40],
    }


# --------------------------------------------------------------------------------
# reporting
# --------------------------------------------------------------------------------

SYMBOL = {"clean": "ok  ", "suspicious": "SUSP", "almost_certainly_misfiled": "MISF",
          "malformed": "MALF"}


def print_summary(report, show_clean=False, stream=sys.stdout):
    w = stream.write
    t = report["totals"]
    w("=" * 78 + "\n")
    w("CQ PROVENANCE AUDIT -- %s\n" % report["source"])
    w("offline, read-only; nothing in the corpus was modified\n")
    w("decision rule is MONOTONE in contamination: more contamination can only "
      "produce\nmore detection, never less\n")
    w("=" * 78 + "\n\n")

    for sc in report["scenarios"]:
        c = sc["counts"]
        w("%-12s  %2d CQs | clean %2d  suspicious %2d  misfiled %2d  malformed %2d\n"
          % (sc["scenario_id"], sc["n_cqs"], c["clean"], c["suspicious"],
             c["almost_certainly_misfiled"], c["malformed"]))
        w("%-12s  home vocab: %d terms (%s)%s\n"
          % ("", sc["home_vocabulary_terms"],
             sc["ground_truth_file"] or "URL slug only",
             "  << DILUTED: vocabulary too large to be discriminative; clean verdicts "
             "here are weak" if sc.get("vocabulary_dilution_warning") else ""))
        for isle in sc.get("foreign_cq_islands") or []:
            w("%-12s  FOREIGN ISLAND: CQs #%s agree on vocabulary foreign to this "
              "scenario\n" % ("", ", ".join(str(m + 1) for m in isle)))
        for cq in sc["cqs"]:
            if cq["verdict"] == "clean" and not show_clean:
                continue
            w("    [%s] #%d (suspicion %.2f) %s\n"
              % (SYMBOL[cq["verdict"]], cq["index"] + 1, cq["suspicion"], cq["text"]))
            for f in cq["flags"]:
                w("           - %s: %s\n" % (f["code"], f["detail"]))
        w("\n")

    w("-" * 78 + "\n")
    w("CROSS-SCENARIO CQ SIMILARITY (flag threshold J>=%.2f, listed from J>=%.2f)\n"
      % (report["thresholds"]["near_duplicate_jaccard"], SIMILARITY_REPORT_FLOOR))
    sims = report.get("top_cross_scenario_similarities", [])
    if not sims:
        w("  none above J=%.2f -- no CQ is shared between two scenarios\n"
          % SIMILARITY_REPORT_FLOOR)
    for d in sims:
        w("  J=%.2f%s  %s #%d  <->  %s #%d\n"
          % (d["jaccard"], "  [FLAGGED]" if d["flagged"] else "",
             d["a"]["scenario_id"], d["a"]["index"] + 1,
             d["b"]["scenario_id"], d["b"]["index"] + 1))
        w("          %s\n          %s\n" % (d["a"]["text"], d["b"]["text"]))
    w("\n")

    w("-" * 78 + "\n")
    w("TOP SUSPICIONS\n")
    for r in report["ranked_by_suspicion"][:15]:
        w("  %.2f  [%s]  %s #%d  %s\n"
          % (r["suspicion"], SYMBOL[r["verdict"]].strip(), r["scenario_id"],
             r["index"] + 1, r["text"][:88]))
    if not report["ranked_by_suspicion"]:
        w("  (none)\n")
    w("\n")

    exp = report.get("injection_experiment")
    if exp:
        w("-" * 78 + "\n")
        w("INJECTION EXPERIMENT -- %s\n" % exp["design"])
        w("  injections              %d\n" % exp["injections"])
        w("  detected                %d\n" % exp["true_positives"])
        w("  missed                  %d\n" % exp["false_negatives"])
        w("  RECALL                  %.4f\n" % exp["recall"])
        w("  precision               %.4f\n" % exp["precision"])
        w("  native false positives  %d (%.4f of the corpus)\n"
          % (len(exp["native_false_positives"]), exp["native_false_positive_rate"]))
        w("\n")

    w("-" * 78 + "\n")
    w("TOTAL  %d scenarios, %d CQs: clean %d | suspicious %d | "
      "almost-certainly-misfiled %d | malformed %d\n"
      % (t["scenarios"], t["cqs"], t["clean"], t["suspicious"],
         t["almost_certainly_misfiled"], t["malformed"]))
    w("This tool deletes nothing.  Flagged CQs require a human decision.\n")
    w("\nLIMITATIONS OF THIS AUDIT\n")
    for line in report.get("limitations", []):
        w("  - %s\n" % line)


def load_scenarios(path):
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, list):
        raise ValueError("expected a JSON list of scenarios, got %s" % type(data).__name__)
    return data


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--scenarios", default=DEFAULT_SCENARIOS)
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--ground-truth-dir", default=DEFAULT_GT_DIR)
    ap.add_argument("--show", choices=["flagged", "clean"], default="flagged",
                    help="'clean' also prints CQs that passed every check")
    ap.add_argument("--no-write", action="store_true", help="print only, write no report")
    ap.add_argument("--measure-recall", action="store_true",
                    help="also run the lone-intruder injection experiment and record "
                         "its recall/precision in the report (takes ~1 min)")
    ap.add_argument("--known-contaminated", default="2023-135-01:2",
                    help="comma-separated scenario_id:index pairs that are known "
                         "genuine contamination, so the recall experiment does not "
                         "count them as false positives; '' to disable")
    args = ap.parse_args(argv)

    try:
        scenarios = load_scenarios(args.scenarios)
    except (OSError, ValueError) as exc:
        sys.stderr.write("cannot read %s: %s\n" % (args.scenarios, exc))
        return 2

    report = audit_corpus(scenarios, ground_truth_dir=args.ground_truth_dir)
    report["source"] = args.scenarios

    if args.measure_recall:
        known = []
        for item in (args.known_contaminated or "").split(","):
            item = item.strip()
            if not item:
                continue
            sid, _, idx = item.partition(":")
            try:
                known.append((sid, int(idx)))
            except ValueError:
                sys.stderr.write("ignoring --known-contaminated entry %r\n" % item)
        report["injection_experiment"] = measure_injection_recall(
            scenarios, ground_truth_dir=args.ground_truth_dir, known_contaminated=known
        )

    if not args.no_write:
        try:
            d = os.path.dirname(os.path.abspath(args.out))
            if d and not os.path.isdir(d):
                os.makedirs(d)
            with open(args.out, "w", encoding="utf-8") as fh:
                json.dump(report, fh, indent=2, ensure_ascii=False)
                fh.write("\n")
        except OSError as exc:
            sys.stderr.write("cannot write %s: %s\n" % (args.out, exc))
            return 2

    print_summary(report, show_clean=(args.show == "clean"))
    if not args.no_write:
        sys.stdout.write("report written to %s\n" % args.out)

    t = report["totals"]
    return 1 if (t["almost_certainly_misfiled"] or t["malformed"]) else 0


if __name__ == "__main__":
    sys.exit(main())
