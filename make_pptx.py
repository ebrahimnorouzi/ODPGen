"""
ODPGen PowerPoint Presentation Generator
Run: python3 make_pptx.py
Output: ODPGen_Presentation.pptx
"""

from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.util import Inches, Pt
import copy

# ── Palette ───────────────────────────────────────────────
BG        = RGBColor(0x0f, 0x11, 0x17)
SURFACE   = RGBColor(0x1a, 0x1d, 0x2e)
SURFACE2  = RGBColor(0x23, 0x27, 0x40)
ACCENT    = RGBColor(0x6c, 0x8e, 0xbf)
ACCENT2   = RGBColor(0x8a, 0xb4, 0xf8)
GREEN     = RGBColor(0x4c, 0xaf, 0x7d)
YELLOW    = RGBColor(0xf0, 0xc0, 0x60)
RED       = RGBColor(0xe0, 0x6c, 0x6c)
PURPLE    = RGBColor(0xb3, 0x9d, 0xdb)
WHITE     = RGBColor(0xe8, 0xea, 0xf0)
MUTED     = RGBColor(0x88, 0x92, 0xa4)
DARK_BLUE = RGBColor(0x12, 0x18, 0x30)

W = Inches(13.33)   # widescreen 16:9
H = Inches(7.5)

# ── Helpers ───────────────────────────────────────────────

def new_prs():
    prs = Presentation()
    prs.slide_width  = W
    prs.slide_height = H
    return prs

def blank_slide(prs):
    blank = prs.slide_layouts[6]   # completely blank
    return prs.slides.add_slide(blank)

def bg(slide, color=BG):
    fill = slide.background.fill
    fill.solid()
    fill.fore_color.rgb = color

def box(slide, x, y, w, h, fill_color=None, border_color=None, border_pt=1.5):
    shape = slide.shapes.add_shape(1, x, y, w, h)   # MSO_SHAPE_TYPE.RECTANGLE = 1
    shape.line.width = Pt(border_pt)
    if fill_color:
        shape.fill.solid()
        shape.fill.fore_color.rgb = fill_color
    else:
        shape.fill.background()
    if border_color:
        shape.line.color.rgb = border_color
    else:
        shape.line.fill.background()
    return shape

def txbox(slide, x, y, w, h, text, size=16, bold=False, color=WHITE,
          align=PP_ALIGN.LEFT, wrap=True, italic=False):
    tb = slide.shapes.add_textbox(x, y, w, h)
    tf = tb.text_frame
    tf.word_wrap = wrap
    p  = tf.paragraphs[0]
    p.alignment = align
    run = p.add_run()
    run.text = text
    run.font.size   = Pt(size)
    run.font.bold   = bold
    run.font.italic = italic
    run.font.color.rgb = color
    return tb

def label_value(slide, x, y, w, label, value, lsize=10, vsize=20,
                lcolor=MUTED, vcolor=ACCENT2):
    txbox(slide, x, y,      w, Inches(0.3), label, lsize, color=lcolor)
    txbox(slide, x, y+Inches(0.3), w, Inches(0.4), value, vsize, bold=True, color=vcolor)

def slide_header(slide, num_str, title, y=Inches(0.55)):
    # number chip
    c = box(slide, Inches(0.5), y, Inches(0.55), Inches(0.35), SURFACE2)
    txbox(slide, Inches(0.5), y+Pt(2), Inches(0.55), Inches(0.35),
          num_str, 11, color=MUTED, align=PP_ALIGN.CENTER)
    # title
    txbox(slide, Inches(1.2), y-Pt(2), Inches(11.5), Inches(0.45),
          title, 28, bold=True, color=ACCENT2)
    # divider line
    line = slide.shapes.add_shape(1, Inches(0.5), y+Inches(0.42), Inches(12.3), Pt(1))
    line.fill.solid(); line.fill.fore_color.rgb = SURFACE2
    line.line.fill.background()

def stat_card(slide, x, y, w, h, number, label, num_color=ACCENT2):
    box(slide, x, y, w, h, SURFACE, SURFACE2)
    txbox(slide, x, y+Inches(0.18), w, Inches(0.7),
          number, 44, bold=True, color=num_color, align=PP_ALIGN.CENTER)
    txbox(slide, x, y+Inches(0.85), w, Inches(0.35),
          label, 12, color=MUTED, align=PP_ALIGN.CENTER)

def card(slide, x, y, w, h, title, body_lines, title_color=YELLOW,
         border=SURFACE2, bg_color=SURFACE):
    box(slide, x, y, w, h, bg_color, border)
    txbox(slide, x+Inches(0.15), y+Inches(0.12), w-Inches(0.3),
          Inches(0.32), title, 13, bold=True, color=title_color)
    body = "\n".join(f"• {l}" for l in body_lines)
    txbox(slide, x+Inches(0.15), y+Inches(0.42), w-Inches(0.3),
          h-Inches(0.55), body, 11, color=WHITE)

def callout(slide, x, y, w, h, text, accent=ACCENT):
    box(slide, x, y, w, h, RGBColor(accent[0]//6, accent[1]//6, accent[2]//6), accent, 2)
    txbox(slide, x+Inches(0.18), y+Inches(0.1), w-Inches(0.28),
          h-Inches(0.2), text, 12, color=WHITE)

def hbar(slide, x, y, w, h, pct, label, val_text, color=GREEN):
    """Horizontal bar: label | [====] val_text"""
    txbox(slide, x, y+Pt(3), Inches(2.4), h, label, 12,
          color=WHITE, align=PP_ALIGN.RIGHT)
    track = box(slide, x+Inches(2.5), y, w, h, SURFACE2)
    track.line.fill.background()
    fill_w = max(Pt(2), int(w * pct))
    bar = box(slide, x+Inches(2.5), y, fill_w, h, color)
    bar.line.fill.background()
    txbox(slide, x+Inches(2.5)+fill_w+Inches(0.05), y+Pt(3),
          Inches(1.5), h, val_text, 11, bold=True, color=WHITE)


# ═══════════════════════════════════════════════════════════
# BUILD SLIDES
# ═══════════════════════════════════════════════════════════

prs = new_prs()

# ────────────────────────────────────────────────────────────
# SLIDE 1 — TITLE
# ────────────────────────────────────────────────────────────
s = blank_slide(prs)
bg(s, BG)

# gradient-ish left panel
left = box(s, 0, 0, Inches(6.5), H, DARK_BLUE)
left.line.fill.background()

# accent bar
bar = box(s, 0, 0, Inches(0.08), H, ACCENT)
bar.line.fill.background()

# badge
txbox(s, Inches(0.5), Inches(0.9), Inches(3.5), Inches(0.35),
      "RESEARCH PRESENTATION", 10, color=ACCENT)

# title
txbox(s, Inches(0.5), Inches(1.35), Inches(5.8), Inches(1.4),
      "ODPGen", 72, bold=True, color=WHITE)

# subtitle
txbox(s, Inches(0.5), Inches(2.85), Inches(5.8), Inches(0.8),
      "Benchmarking Large Language Models on\nOntology Design Pattern Generation",
      18, color=MUTED, italic=True)

# divider
d = box(s, Inches(0.5), Inches(3.78), Inches(5.5), Pt(1.5), ACCENT)
d.line.fill.background()

# meta grid
meta = [
    ("Models Evaluated",   "6 LLMs"),
    ("Prompt Strategies",  "5 Configurations"),
    ("ODP Scenarios",      "14 Benchmarks"),
    ("Total Runs",         "350 Evaluated"),
]
for i, (lbl, val) in enumerate(meta):
    col = i % 2
    row = i // 2
    lx = Inches(0.5 + col * 2.8)
    ly = Inches(3.95 + row * 0.85)
    txbox(s, lx, ly,          Inches(2.6), Inches(0.3), lbl, 10, color=MUTED)
    txbox(s, lx, ly+Inches(0.28), Inches(2.6), Inches(0.4), val, 18, bold=True, color=ACCENT2)

# right panel — decorative stats
rdata = [("350", "evaluated outputs"), ("14", "ODP scenarios"),
         ("5",  "prompt configs"),    ("5",  "LLMs tested")]
for i, (n, l) in enumerate(rdata):
    rx = Inches(7.2 + (i % 2) * 2.85)
    ry = Inches(1.6 + (i // 2) * 2.4)
    stat_card(s, rx, ry, Inches(2.5), Inches(1.9), n, l)

txbox(s, Inches(6.8), Inches(6.8), Inches(6.0), Inches(0.4),
      "ODPGen  ·  Ontology Design Pattern Generation Benchmark", 10, color=MUTED)


# ────────────────────────────────────────────────────────────
# SLIDE 2 — OVERVIEW
# ────────────────────────────────────────────────────────────
s = blank_slide(prs)
bg(s)
slide_header(s, "01", "Overview & Research Questions")

txbox(s, Inches(0.5), Inches(1.2), Inches(12.3), Inches(0.4),
      "ODPGen evaluates how well LLMs generate Ontology Design Patterns (ODPs) from natural-language "
      "scenarios and competency questions — measuring syntactic validity, faithfulness, and semantic quality.",
      13, color=MUTED)

# left: what are ODPs
card(s, Inches(0.5), Inches(1.75), Inches(4.1), Inches(2.6),
     "What are ODPs?",
     ["Reusable ontology templates encoding solutions to common modelling problems",
      "Building blocks for ontology engineering (like SW design patterns)",
      "Define classes, properties & axioms grounded in a scenario",
      "Verified by Competency Questions (CQs) — must-answer queries"],
     title_color=ACCENT2, border=ACCENT)

# right: RQs
card(s, Inches(4.8), Inches(1.75), Inches(8.0), Inches(2.6),
     "Research Questions",
     ["RQ1  How well can LLMs generate valid OWL/Turtle from natural-language scenarios?",
      "RQ2  How logically consistent and error-free are the outputs?",
      "RQ3  What is the impact of prompting strategy (CQs, reasoning, constraints)?",
      "RQ4  How faithful are the outputs to the ground-truth scenario?",
      "RQ5  Can models generalize to recently published ODPs unseen during training?"],
     title_color=ACCENT2, border=ACCENT)

# callout
callout(s, Inches(0.5), Inches(4.5), Inches(12.3), Inches(0.6),
        "Core hypothesis: Structured prompting (CQs + reasoning or self-validation) should "
        "improve faithfulness — but may trade off against syntactic correctness for weaker models.")

# stat row
for i, (n, l, c) in enumerate([("6", "LLMs", ACCENT2), ("5", "Prompt Configs", GREEN),
                                ("14", "Scenarios", YELLOW), ("350", "Runs", PURPLE)]):
    stat_card(s, Inches(0.5 + i * 3.1), Inches(5.3), Inches(2.8), Inches(1.9), n, l, c)


# ────────────────────────────────────────────────────────────
# SLIDE 3 — DATASET
# ────────────────────────────────────────────────────────────
s = blank_slide(prs)
bg(s)
slide_header(s, "02", "Dataset — 14 ODP Scenarios (2023–2026)")

txbox(s, Inches(0.5), Inches(1.2), Inches(12.3), Inches(0.35),
      "Scenarios sourced from published ODP papers, each pairing a domain description, "
      "Competency Questions, and a ground-truth OWL ontology.", 12, color=MUTED)

# table header
cols  = [Inches(0.5), Inches(2.45), Inches(6.95), Inches(8.35), Inches(10.0), Inches(11.5)]
cwidths = [Inches(1.9), Inches(4.45), Inches(1.35), Inches(1.6), Inches(1.45), Inches(1.3)]
headers = ["Scenario ID", "Domain", "CQs", "Year", "Status", ""]
for j, (hx, hw, ht) in enumerate(zip(cols, cwidths, headers)):
    box(s, hx, Inches(1.65), hw, Inches(0.32), SURFACE2).line.fill.background()
    txbox(s, hx+Pt(4), Inches(1.68), hw, Inches(0.28), ht, 10, bold=True, color=ACCENT2)

rows = [
    ("2023-133-01", "Causal Bayesian Networks (Sprinkler)", "~8",  "2023", "Likely Seen",    GREEN),
    ("2023-133-02", "Causal Bayesian Networks (Asthma)",   "~6",  "2023", "Likely Seen",    GREEN),
    ("2023-134-01", "Data Governance / SOLID Pods",        "~10", "2023", "Likely Seen",    GREEN),
    ("2023-134-02", "GDPR Compliance Policies",            "~8",  "2023", "Likely Seen",    GREEN),
    ("2023-134-03", "Access Control / ARGOS Database",     "~7",  "2023", "Likely Seen",    GREEN),
    ("2023-135-01", "Role-Dependent Naming",               "~9",  "2023", "Likely Seen",    GREEN),
    ("2024-145-01", "Maritime / AISHIP Vessel Data",       "~12", "2024", "Borderline",     YELLOW),
    ("2025-147-01", "Legal Ontology (Indian Courts)",      "~15", "2025", "Plausible Unseen", RED),
    ("2025-149-01", "Temporal Indirection (Finance)",      "~11", "2025", "Plausible Unseen", RED),
    ("2025-150-01", "MQTT Protocol Semantics",             "~9",  "2025", "Plausible Unseen", RED),
    ("2025-151-01", "Circular Economy (Product Lifecycle)","~14", "2025", "Plausible Unseen", RED),
    ("2025-151-02", "Causality in Environmental Policy",   "~10", "2025", "Plausible Unseen", RED),
    ("2025-153-01", "Governance Roles & Responsibilities", "~13", "2025", "Plausible Unseen", RED),
    ("2026-155-01", "State of the World Pattern",          "~11", "2026", "Plausible Unseen", RED),
]
for i, (sid, domain, cqs, yr, status, sc) in enumerate(rows):
    ry = Inches(2.0) + i * Inches(0.345)
    row_bg = SURFACE if i % 2 == 0 else BG
    for j, (hx, hw) in enumerate(zip(cols, cwidths)):
        box(s, hx, ry, hw, Inches(0.34), row_bg).line.fill.background()
    vals = [sid, domain, cqs, yr, status]
    colors = [MUTED, WHITE, MUTED, MUTED, sc]
    for j, (hx, hw, val, col) in enumerate(zip(cols, cwidths, vals, colors)):
        txbox(s, hx+Pt(4), ry+Pt(4), hw-Pt(4), Inches(0.28), val, 10, color=col)

callout(s, Inches(0.5), Inches(6.98), Inches(12.3), Inches(0.42),
        "9 of 14 scenarios are plausible_unseen by most models — enabling generalization vs. memorization analysis. "
        "Unseen status is determined by comparing ODP publication date to model training cutoff.",
        YELLOW)


# ────────────────────────────────────────────────────────────
# SLIDE 4 — MODELS
# ────────────────────────────────────────────────────────────
s = blank_slide(prs)
bg(s)
slide_header(s, "03", "Models Evaluated")

model_data = [
    ("BLOOM-3B",       "bigscience/bloom-3b",          "3B",        "Base (no RLHF)",     "Jul 2022",  "1",  RED),
    ("BLOOMZ-7B1",     "bigscience/bloomz-7b1",         "7B",        "Instruction-tuned",  "Jul 2022",  "5",  RED),
    ("Mistral-7B",     "Mistral-7B-Instruct-v0.3",      "7B",        "Instruction-tuned",  "2023",      "5",  YELLOW),
    ("Llama-3.1-8B ★", "Llama-3.1-8B-Instruct",         "8B",        "Instruction-tuned",  "Dec 2023",  "5",  GREEN),
    ("Llama-2-70B",    "Llama-2-70b-chat-hf",           "70B",       "Chat (RLHF)",        "Sep 2022",  "5",  ACCENT2),
    ("GPT-5.4",        "OpenAI API",                    "Proprietary","RLHF + API",         "~2024",     "5",  PURPLE),
]

# header row
hdr_labels = ["Model", "Checkpoint / API", "Parameters", "Type", "Cutoff", "Configs"]
hdr_xs     = [Inches(0.5), Inches(2.2), Inches(5.3), Inches(6.5), Inches(8.6), Inches(10.2)]
hdr_ws     = [Inches(1.65), Inches(3.05), Inches(1.15), Inches(2.05), Inches(1.55), Inches(1.2)]
for hx, hw, hl in zip(hdr_xs, hdr_ws, hdr_labels):
    box(s, hx, Inches(1.22), hw, Inches(0.35), SURFACE2).line.fill.background()
    txbox(s, hx+Pt(4), Inches(1.25), hw, Inches(0.28), hl, 10, bold=True, color=ACCENT2)

for i, (name, ckpt, params, typ, cutoff, cfgs, nc) in enumerate(model_data):
    ry = Inches(1.6) + i * Inches(0.83)
    bg_c = RGBColor(nc[0]//12, nc[1]//12, nc[2]//12)
    box(s, Inches(0.5), ry, Inches(12.3), Inches(0.78), bg_c,
        RGBColor(nc[0]//3, nc[1]//3, nc[2]//3), 1)
    vals = [name, ckpt, params, typ, cutoff, cfgs]
    cols2 = [nc, MUTED, WHITE, WHITE, MUTED, WHITE]
    bolds = [True, False, False, False, False, False]
    for hx, hw, val, col, bold in zip(hdr_xs, hdr_ws, vals, cols2, bolds):
        txbox(s, hx+Pt(6), ry+Pt(8), hw-Pt(6), Inches(0.6), val, 12,
              bold=bold, color=col)

callout(s, Inches(0.5), Inches(6.7), Inches(12.3), Inches(0.65),
        "Execution: Open-source models via Hugging Face Transformers on GPU (4-bit quant for 70B). "
        "GPT-5.4 via OpenAI API. Temperature = 0.0, max_new_tokens = 1024 for all runs.")


# ────────────────────────────────────────────────────────────
# SLIDE 5 — PROMPTING STRATEGIES
# ────────────────────────────────────────────────────────────
s = blank_slide(prs)
bg(s)
slide_header(s, "04", "Prompting Strategies — 5 Configurations")

configs = [
    ("1. scenario-only",
     ["Model receives only the domain scenario text",
      "Must infer modeling scope from narrative alone",
      "Highest scenario vocabulary coverage (0.440)"],
     ACCENT2, Inches(0.5), Inches(1.22)),
    ("2. cq-only",
     ["Model receives only the Competency Questions",
      "Highest CQ coverage (0.844 avg)",
      "Also highest hallucination ratio (0.548)"],
     ACCENT2, Inches(4.1), Inches(1.22)),
    ("3. scenario-cq  [Balanced]",
     ["Both scenario text and CQs provided",
      "Requires scenario→axiom & CQ→axiom mapping",
      "Best balance of CQ + scenario coverage"],
     GREEN, Inches(7.7), Inches(1.22)),
    ("4. scenario-cq-constraints  [Validation]",
     ["Adds 6-item self-validation checklist",
      "Syntax, domain/range, coherence, scope checks",
      "Lowest hallucination after reasoning (0.392)",
      "Lowest parse success (29%) — extra text confuses parsers"],
     YELLOW, Inches(0.5), Inches(3.5)),
    ("5. scenario-cq-reasoning  [Best Faithfulness]",
     ["6-step internal reasoning: extract requirements → propose classes →",
      "  draft axioms → verify CQs → remove unsupported → output ODP",
      "Lowest hallucination overall: 0.367",
      "Best scenario coverage among CQ-aware configs (0.379)",
      "Good CQ coverage: 0.748 on successful parses"],
     GREEN, Inches(4.5), Inches(3.5)),
]
for title, bullets, tc, cx, cy in configs:
    w = Inches(3.55) if cx < Inches(4) else (Inches(3.55) if cx < Inches(7) else Inches(5.1))
    if cx == Inches(4.5) and cy > Inches(3):
        w = Inches(8.3)
    h = Inches(2.1)
    card(s, cx, cy, w, h, title, bullets, tc)

callout(s, Inches(0.5), Inches(6.88), Inches(12.3), Inches(0.52),
        "All prompts use standard prefixes (owl:, rdf:, rdfs:, xsd:, :) and require "
        "a mandatory owl:Ontology header with rdfs:label and rdfs:comment.", ACCENT)


# ────────────────────────────────────────────────────────────
# SLIDE 6 — EVALUATION METHODOLOGY
# ────────────────────────────────────────────────────────────
s = blank_slide(prs)
bg(s)
slide_header(s, "05", "Evaluation Methodology — 4 Layers")

layers = [
    ("Layer 1 — Formal / Syntactic Quality", ACCENT,
     ["parse_success — Valid Turtle syntax?",
      "triple_count — Total RDF triples",
      "ontology_declared — owl:Ontology header?",
      "label_coverage_ratio — all named entities with rdfs:label",
      "class_label_ratio [NEW] — owl:Class entities with rdfs:label",
      "property_label_ratio [NEW] — Obj/Data/Ann properties with rdfs:label",
      "comment_coverage_ratio — entities with rdfs:comment",
      "domain_range_ratio — rdfs:domain & rdfs:range",
      "class_count, property_count — pattern size"]),
    ("Layer 2 — Scenario Faithfulness", GREEN,
     ["cq_coverage_ratio — CQs answerable (token overlap)",
      "cq_mean_token_recall — avg recall per CQ",
      "scenario_coverage_ratio — scenario vocab in ontology",
      "hallucination_ratio — unsupported vocab fraction",
      "ontology_vocab_size"]),
    ("Layer 3 — Semantic Similarity (vs Ground Truth)", YELLOW,
     ["class_name_precision / recall / F1 — name match vs ground truth",
      "property_name_precision / recall / F1 — same for properties",
      "class_jaccard, property_jaccard — |A∩B|/|A∪B| set similarity",
      "combined_f1 — macro avg of class + property F1",
      "matched_class_names — class names in both generated & GT",
      "matched_property_names — property names in both generated & GT",
      "  e.g. 'event', 'has effect', 'relational database'"]),
    ("Layer 4 — Feasibility & Unseen Analysis", PURPLE,
     ["(model, ODP) labelled plausible_unseen or likely_seen",
      "Compares ODP pub date vs. model training cutoff",
      "Confidence: high / medium / low",
      "Enables generalization vs. memorization analysis",
      "Future: human eval — Correctness, Completeness,",
      "  Clarity/Reusability (1–5 scale)"]),
]
for i, (title, color, bullets) in enumerate(layers):
    col = i % 2
    row = i // 2
    cx = Inches(0.5 + col * 6.45)
    cy = Inches(1.22 + row * 2.65)
    card(s, cx, cy, Inches(6.1), Inches(2.5), title, bullets, color,
         RGBColor(color[0]//3, color[1]//3, color[2]//3))

callout(s, Inches(0.5), Inches(6.78), Inches(12.3), Inches(0.62),
        "Pipeline: evaluate_outputs.py → eval/formal_quality.py + eval/faithfulness.py "
        "+ eval/similarity.py → results/summary.csv  (350 rows × 44 columns, "
        "incl. class_label_ratio & property_label_ratio)")


# ────────────────────────────────────────────────────────────
# SLIDE 7 — RAW OUTPUT EXAMPLES
# ────────────────────────────────────────────────────────────
s = blank_slide(prs)
bg(s)
slide_header(s, "06", "Raw Output Examples — Scenario 2023-133-01 (Causal Bayesian Network)")

txbox(s, Inches(0.5), Inches(1.2), Inches(12.3), Inches(0.35),
      'Scenario: "Model a causal Bayesian network over events (RainySeason, SprinklerOn, Raining, '
      'WetPavement, SlipperyPavement) with weighted causal edges and mediators as an ODP."',
      11, color=MUTED, italic=True)

examples = [
    ("GPT-5.4",       "SUCCESS", GREEN,
     ":Event a owl:Class .\n:CausalRelation a owl:Class .\n:MediatedCausalRelation\n  rdfs:subClassOf :CausalRelation .\n:hasCause a owl:ObjectProperty .\n:effectWeight a owl:DatatypeProperty .\n# Full mapping tables included",
     "Valid Turtle · Full class hierarchy · CQ→axiom mapping"),
    ("Llama-3.1-8B",  "SUCCESS", GREEN,
     ":Event a owl:Class .\n:RainySeasonEvent rdfs:subClassOf :Event .\n:WetPavement rdfs:subClassOf :Event .\n:Cause a owl:ObjectProperty .\n:Weight a owl:DatatypeProperty .\n# Causal instances with weights 0.6, 0.8, 0.9, 0.7",
     "Valid Turtle · Instance-level causal chain · Lowest hallucination"),
    ("Llama-2-70B",   "SUCCESS", GREEN,
     ":Event a owl:Class .\n:RainySeason rdfs:subClassOf :Event .\n:SprinklerOn rdfs:subClassOf :Event .\n:CausalRelation a owl:ObjectProperty .\n:EffectWeight a owl:DatatypeProperty ;\n  rdfs:range xsd:decimal .",
     "Valid Turtle · Simpler structure · Scenario mapping section"),
    ("Mistral-7B",    "PARTIAL", YELLOW,
     ":Event a owl:Class .\n:CausalLink a owl:Class .\n:causes a owl:ObjectProperty .\n:effectWeight a owl:DatatypeProperty .\n# Uses Manchester-style 'some' restriction\n# → Parser fails on 71% of scenario-cq runs",
     "Mixes Manchester syntax in Turtle → parse failures"),
    ("BLOOMZ-7B1",    "FAIL", RED,
     '"What if the pavement is wet, how would\nit effect the pavement being slippery?"\n\n-- Model rephrases the CQ instead of\n   generating an ontology --',
     "0% parse rate · Outputs natural language only"),
    ("BLOOM-3B",      "FAIL", RED,
     '"If the pavement is wet, the pavement is wet.\nIf the pavement is wet, the pavement is not wet.\nIf the pavement is not wet, the pavement is wet…"\n\n-- Repetitive hallucinations --',
     "Base model · No instruction following · Degenerate output"),
]
for i, (model, status, sc, code, note) in enumerate(examples):
    col = i % 3
    row = i // 3
    cx = Inches(0.5 + col * 4.3)
    cy = Inches(1.65 + row * 2.7)
    cw, ch = Inches(4.05), Inches(2.55)
    bg_c = RGBColor(sc[0]//14, sc[1]//14, sc[2]//14)
    box(s, cx, cy, cw, ch, bg_c, RGBColor(sc[0]//3, sc[1]//3, sc[2]//3), 1)
    txbox(s, cx+Pt(6), cy+Pt(6), cw-Pt(60), Inches(0.28), model, 12, bold=True, color=sc)
    txbox(s, cx+cw-Inches(0.9), cy+Pt(8), Inches(0.82), Inches(0.24), status, 10,
          bold=True, color=sc, align=PP_ALIGN.CENTER)
    # code box
    code_bg = box(s, cx+Pt(5), cy+Inches(0.38), cw-Pt(10), Inches(1.6), SURFACE2)
    code_bg.line.fill.background()
    txbox(s, cx+Pt(8), cy+Inches(0.41), cw-Pt(16), Inches(1.55), code, 8,
          color=RGBColor(0xc3, 0xc8, 0xd8))
    txbox(s, cx+Pt(6), cy+Inches(2.05), cw-Pt(12), Inches(0.42), f"→ {note}", 9, color=MUTED, italic=True)


# ────────────────────────────────────────────────────────────
# SLIDE 8 — PARSE SUCCESS RESULTS
# ────────────────────────────────────────────────────────────
s = blank_slide(prs)
bg(s)
slide_header(s, "07", "Results — Parse Success Rate")

txbox(s, Inches(0.5), Inches(1.2), Inches(12.3), Inches(0.32),
      "Parse success is the primary gate metric — an ontology must be valid Turtle before any quality metrics apply.  Total: 364 runs.",
      12, color=MUTED)

# by model
txbox(s, Inches(0.5), Inches(1.58), Inches(6.0), Inches(0.3), "By Model", 14, bold=True, color=ACCENT2)
bars_m = [
    ("Llama-3.1-8B-Instruct",  0.757, "75.7%  (53/70)", GREEN),
    ("Llama-2-70B-chat",       0.657, "65.7%  (46/70)", GREEN),
    ("Mistral-7B-Instruct",    0.414, "41.4%  (29/70)", YELLOW),
    ("GPT-5.4",                0.371, "37.1%  (26/70)", YELLOW),
    ("BLOOMZ-7B1",             0.00,  "0%  (0/70)",     RED),
    ("BLOOM-3B",               0.00,  "0%  (0/14)",     RED),
]
for i, (lbl, pct, txt, col) in enumerate(bars_m):
    hbar(s, Inches(0.5), Inches(1.95) + i*Inches(0.62), Inches(5.8), Inches(0.44),
         pct, lbl, txt, col)

# by config
txbox(s, Inches(7.0), Inches(1.58), Inches(5.8), Inches(0.3), "By Configuration", 14, bold=True, color=ACCENT2)
bars_c = [
    ("cq-only",                  0.60, "60.0%  (42/70)", GREEN),
    ("scenario-only",            0.514,"51.4%  (36/70)", GREEN),
    ("scenario-cq",              0.393,"39.3%  (33/84)", ACCENT2),
    ("scenario-cq-reasoning",    0.329,"32.9%  (23/70)", YELLOW),
    ("scenario-cq-constraints",  0.286,"28.6%  (20/70)", YELLOW),
]
for i, (lbl, pct, txt, col) in enumerate(bars_c):
    hbar(s, Inches(7.0), Inches(1.95) + i*Inches(0.62), Inches(5.8), Inches(0.44),
         pct, lbl, txt, col)

callout(s, Inches(0.5), Inches(5.85), Inches(12.3), Inches(0.68),
        "Insight: Complex prompts (constraints, reasoning) reduce parse success for smaller models — "
        "extra output structure interferes with Turtle extraction. GPT-5.4 scores 0% on reasoning/constraints "
        "configs, while Llama-3.1-8B maintains 64–71% even there.", YELLOW)

callout(s, Inches(0.5), Inches(6.65), Inches(12.3), Inches(0.68),
        "GPT paradox: GPT-5.4 achieves 86% parse rate on cq-only (best in class!) but only 37% overall "
        "because complex prompts wrap the ontology in non-Turtle formatting the parser cannot extract.", ACCENT)


# ────────────────────────────────────────────────────────────
# SLIDE 9 — QUALITY METRICS TABLE
# ────────────────────────────────────────────────────────────
s = blank_slide(prs)
bg(s)
slide_header(s, "08", "Results — Quality Metrics (Successful Parses Only)")

txbox(s, Inches(0.5), Inches(1.2), Inches(12.3), Inches(0.3),
      "Metrics computed only when parse_success = True.  350 runs × 44 metrics.  "
      "New: class_label_ratio & property_label_ratio split from the aggregate label_coverage_ratio.",
      11, color=MUTED)

# ── Model table ──────────────────────────────────────────
# cols: Model | Parse | CQ Cov | Scen Cov | Halluc | Cls Lbl★ | Prop Lbl★ | F1
col_labels = ["Model", "Parse Rate", "CQ Coverage", "Scenario Cov.",
              "Hallucination", "Class Label ★", "Prop Label ★", "Combined F1"]
col_xs = [Inches(0.5),  Inches(2.35), Inches(3.6),  Inches(4.9),
          Inches(6.15), Inches(7.45), Inches(8.75), Inches(10.1)]
col_ws = [Inches(1.8),  Inches(1.2),  Inches(1.25), Inches(1.2),
          Inches(1.25), Inches(1.25), Inches(1.3),  Inches(1.15)]

for hx, hw, hl in zip(col_xs, col_ws, col_labels):
    box(s, hx, Inches(1.6), hw, Inches(0.33), SURFACE2).line.fill.background()
    txbox(s, hx+Pt(3), Inches(1.63), hw, Inches(0.28), hl, 8, bold=True, color=ACCENT2)

# (name, parse, cq, scen, halluc, cls_lbl, prop_lbl, f1, model_color, good_cols, bad_cols)
model_rows = [
    ("GPT-5.4",        "37.1%", "0.804", "0.444", "0.585", "1.000", "0.885", "0.110",
     YELLOW, [2, 5, 7], [4]),
    ("Llama-3.1-8B ★", "75.7%", "0.729", "0.335", "0.345", "1.000", "0.830", "0.071",
     GREEN,  [1, 5],    [3]),          # best parse & halluc
    ("Llama-2-70B",    "65.7%", "0.689", "0.258", "0.489", "1.000", "0.717", "0.072",
     ACCENT2, [5],     [6]),           # worst prop label
    ("Mistral-7B",     "41.4%", "0.615", "0.279", "0.459", "1.000", "0.966", "0.059",
     YELLOW,  [5, 6],  []),            # best prop label
    ("BLOOMZ-7B1",     "0%",    "—",     "—",     "—",     "—",     "—",     "—",
     RED,     [],      []),
]
for i, (mname, *vals, mc, highs, lows) in enumerate(model_rows):
    ry = Inches(1.97) + i * Inches(0.57)
    row_bg = RGBColor(mc[0]//16, mc[1]//16, mc[2]//16) if i < 4 else SURFACE
    box(s, Inches(0.5), ry, Inches(10.75), Inches(0.53), row_bg).line.fill.background()
    all_vals = [mname] + vals
    for j, (hx, hw, val) in enumerate(zip(col_xs, col_ws, all_vals)):
        col2 = mc if j == 0 else (GREEN if j in highs else (RED if j in lows else WHITE))
        bold = j == 0 or j in highs or j in lows
        txbox(s, hx+Pt(4), ry+Pt(8), hw-Pt(4), Inches(0.4), val, 10,
              bold=bold, color=col2)

# label insight callout
callout(s, Inches(0.5), Inches(5.02), Inches(12.3), Inches(0.52),
        "class_label_ratio = 1.000 for ALL models — classes always annotated.  "
        "property_label_ratio reveals the gap: Llama-2-70B leaves 28% of properties unlabelled (0.717); "
        "Mistral-7B is the most diligent (0.966).  The old aggregate label_coverage_ratio (always 1.0) masked this.",
        YELLOW)

# ── Config table ────────────────────────────────────────
txbox(s, Inches(0.5), Inches(5.65), Inches(12.3), Inches(0.28),
      "Average Quality Metrics by Configuration (successful parses)", 12, bold=True, color=ACCENT2)

cfg_labels = ["Config", "CQ Cov.", "Scen. Cov.", "Halluc.", "Cls Lbl", "Prop Lbl ★", "F1"]
cfg_xs = [Inches(0.5), Inches(3.0), Inches(4.55), Inches(6.1), Inches(7.65), Inches(8.95), Inches(10.55)]
cfg_ws = [Inches(2.45), Inches(1.5), Inches(1.5),  Inches(1.5), Inches(1.25), Inches(1.55), Inches(1.2)]
for hx, hw, hl in zip(cfg_xs, cfg_ws, cfg_labels):
    box(s, hx, Inches(5.98), hw, Inches(0.28), SURFACE2).line.fill.background()
    txbox(s, hx+Pt(3), Inches(6.01), hw, Inches(0.24), hl, 8, bold=True, color=ACCENT2)

# (name, cq, scen, halluc, cls_lbl, prop_lbl, f1, color)
cfg_rows = [
    ("cq-only",                "0.844", "0.182", "0.548", "1.000", "0.833", "0.088", RED),
    ("scenario-cq",            "0.819", "0.346", "0.416", "1.000", "0.788", "0.086", ACCENT2),
    ("scenario-cq-constraints","0.836", "0.282", "0.392", "1.000", "0.850", "0.056", YELLOW),
    ("scenario-cq-reasoning ★","0.748", "0.379", "0.367", "1.000", "0.826", "0.082", GREEN),
    ("scenario-only",          "0.352", "0.440", "0.451", "1.000", "0.861", "0.059", MUTED),
]
for i, (cname, cq, sc, hal, clbl, plbl, f1, cc) in enumerate(cfg_rows):
    ry = Inches(6.3) + i * Inches(0.23)
    box(s, Inches(0.5), ry, Inches(11.25), Inches(0.21),
        SURFACE if i % 2 == 0 else BG).line.fill.background()
    for hx, hw, val, bold in zip(cfg_xs, cfg_ws,
                                  [cname, cq, sc, hal, clbl, plbl, f1],
                                  [True,  False, False, False, False, False, False]):
        vc = cc if bold else (GREEN if val == "1.000" else WHITE)
        txbox(s, hx+Pt(3), ry+Pt(1), hw, Inches(0.19), val, 9, bold=bold, color=vc)


# ────────────────────────────────────────────────────────────
# SLIDE 9b — MATCHED CLASS & PROPERTY NAMES
# ────────────────────────────────────────────────────────────
s = blank_slide(prs)
bg(s)
slide_header(s, "09", "Matched Class & Property Names vs. Ground Truth")

txbox(s, Inches(0.5), Inches(1.2), Inches(12.3), Inches(0.35),
      "Names found in BOTH the generated ontology and the ground-truth (case-folded, CamelCase split). "
      "Low F1 does not mean zero conceptual overlap — models do capture key domain terms.",
      12, color=MUTED)

# explain matching logic
card(s, Inches(0.5), Inches(1.65), Inches(4.0), Inches(1.8),
     "How matching works",
     ["Local name extracted from URI (:hasCause → 'has cause')",
      "CamelCase split: HasEffect → 'has effect'",
      "Case-folded and whitespace-normalised",
      "Precision = |matched| / |generated|",
      "Recall    = |matched| / |ground truth|",
      "F1        = harmonic mean of P & R"],
     ACCENT2, ACCENT)

# examples table
txbox(s, Inches(4.7), Inches(1.65), Inches(8.1), Inches(0.3),
      "Examples — GPT-5.4 / cq-only (successful parses with ground truth)", 12, bold=True, color=ACCENT2)

ex_headers = ["Scenario", "Matched Classes", "Matched Properties"]
ex_xs = [Inches(4.7), Inches(6.8), Inches(9.7)]
ex_ws = [Inches(2.05), Inches(2.85), Inches(3.3)]
for hx, hw, hl in zip(ex_xs, ex_ws, ex_headers):
    box(s, hx, Inches(2.0), hw, Inches(0.3), SURFACE2).line.fill.background()
    txbox(s, hx+Pt(4), Inches(2.03), hw, Inches(0.26), hl, 9, bold=True, color=ACCENT2)

ex_rows = [
    ("2023-133-01\nCausal BN",       "event",                          "has effect, has outcome"),
    ("2023-135-01\nRole Naming",      "name usage",                     "has name usage"),
    ("2025-149-01\nTemporal Indir.",  "relational database,\ntable, column", "—"),
    ("2025-151-01\nEval Request",     "evaluation request",             "requested action, issued by"),
    ("2025-153-01\nGovernance Roles", "entity, temporal binding,\ntransition rule",
                                      "has temporal binding,\nbinds identifier, has transition"),
]
for i, (sid, cls, prop) in enumerate(ex_rows):
    ry = Inches(2.34) + i * Inches(0.7)
    rbg = SURFACE if i % 2 == 0 else BG
    for hx, hw in zip(ex_xs, ex_ws):
        box(s, hx, ry, hw, Inches(0.66), rbg).line.fill.background()
    for hx, hw, val, vc in zip(ex_xs, ex_ws,
                                [sid, cls, prop],
                                [MUTED, GREEN, ACCENT2]):
        txbox(s, hx+Pt(4), ry+Pt(4), hw-Pt(8), Inches(0.58), val, 10, color=vc)

# property label breakdown chart
txbox(s, Inches(0.5), Inches(3.6), Inches(4.0), Inches(0.3),
      "Property Label Ratio by Model", 13, bold=True, color=ACCENT2)
prop_bars = [
    ("Mistral-7B",     0.966, "0.966", GREEN),
    ("GPT-5.4",        0.885, "0.885", GREEN),
    ("Llama-3.1-8B",   0.830, "0.830", YELLOW),
    ("Llama-2-70B",    0.717, "0.717", RED),
]
for i, (lbl, pct, txt, col) in enumerate(prop_bars):
    hbar(s, Inches(0.5), Inches(4.0) + i * Inches(0.68), Inches(3.8), Inches(0.5),
         pct, lbl, txt, col)

callout(s, Inches(0.5), Inches(6.88), Inches(12.3), Inches(0.52),
        "class_label_ratio = 1.000 for ALL models on ALL successful parses.  "
        "The split metric reveals that Llama-2-70B neglects property annotations (0.717); "
        "scenario-cq config yields the worst property labelling (0.788) while scenario-only is best (0.861).",
        YELLOW)


# ────────────────────────────────────────────────────────────
# SLIDE 10 — HEATMAP
# ────────────────────────────────────────────────────────────
s = blank_slide(prs)
bg(s)
slide_header(s, "10", "Model × Config Parse Success Heatmap")

txbox(s, Inches(0.5), Inches(1.2), Inches(12.3), Inches(0.3),
      "Parse success rate for each (model, config) pair. 14 scenarios per cell.", 12, color=MUTED)

heat_configs = ["cq-only", "scenario-cq", "scenario-cq-\nconstraints", "scenario-cq-\nreasoning", "scenario-only", "Average"]
heat_models  = ["BLOOMZ-7B1", "GPT-5.4", "Llama-2-70B", "Llama-3.1-8B ★", "Mistral-7B"]
heat_data = [
    ["0%",  "0%",  "0%",  "0%",  "0%",  "0%"],
    ["86%", "50%", "0%",  "0%",  "50%", "37%"],
    ["86%", "71%", "71%", "50%", "50%", "66%"],
    ["79%", "86%", "64%", "71%", "79%", "76%"],
    ["50%", "29%", "7%",  "43%", "79%", "41%"],
]
heat_colors = [
    [RED,    RED,    RED,    RED,    RED,    RED],
    [GREEN,  YELLOW, RED,    RED,    YELLOW, YELLOW],
    [GREEN,  GREEN,  GREEN,  YELLOW, YELLOW, GREEN],
    [GREEN,  GREEN,  YELLOW, GREEN,  GREEN,  GREEN],
    [YELLOW, RED,    RED,    YELLOW, GREEN,  YELLOW],
]

cell_w = Inches(1.72)
cell_h = Inches(0.68)
start_x = Inches(2.5)
start_y = Inches(1.6)

# column headers
for j, cfg in enumerate(heat_configs):
    cx = start_x + j * cell_w
    box(s, cx, start_y, cell_w, Inches(0.55), SURFACE2).line.fill.background()
    txbox(s, cx+Pt(2), start_y+Pt(3), cell_w-Pt(4), Inches(0.5), cfg,
          9, bold=True, color=ACCENT2, align=PP_ALIGN.CENTER)

for i, (model, row, colors) in enumerate(zip(heat_models, heat_data, heat_colors)):
    ry = start_y + Inches(0.58) + i * cell_h
    # model label
    box(s, Inches(0.5), ry, Inches(1.95), cell_h,
        RGBColor(0x1a, 0x1d, 0x2e)).line.fill.background()
    txbox(s, Inches(0.52), ry+Pt(8), Inches(1.9), cell_h-Pt(10), model,
          11, bold=(i==3), color=(GREEN if i==3 else WHITE))
    for j, (val, vc) in enumerate(zip(row, colors)):
        cx = start_x + j * cell_w
        bg_c = RGBColor(vc[0]//8, vc[1]//8, vc[2]//8)
        box(s, cx, ry, cell_w, cell_h, bg_c,
            RGBColor(vc[0]//4, vc[1]//4, vc[2]//4), 1)
        txbox(s, cx, ry+Pt(10), cell_w, cell_h-Pt(10), val,
              18, bold=True, color=vc, align=PP_ALIGN.CENTER)

callout(s, Inches(0.5), Inches(6.55), Inches(7.5), Inches(0.82),
        "Llama-3.1-8B is the only model maintaining >60% parse success "
        "across ALL 5 configs — including difficult constraint/reasoning prompts.", GREEN)
callout(s, Inches(8.2), Inches(6.55), Inches(4.6), Inches(0.82),
        "GPT-5.4: 86% on cq-only (best!) but 0% on reasoning & constraints — prompt format breaks Turtle extraction.", YELLOW)


# ────────────────────────────────────────────────────────────
# SLIDE 11 — WINNER / SUMMARY
# ────────────────────────────────────────────────────────────
s = blank_slide(prs)
bg(s)
slide_header(s, "11", "Summary & Best Performing Model")

# winner card
box(s, Inches(0.5), Inches(1.2), Inches(5.5), Inches(3.6),
    RGBColor(0x07, 0x18, 0x0e), RGBColor(0x2a, 0x6e, 0x4a), 2)
txbox(s, Inches(0.65), Inches(1.3), Inches(5.2), Inches(0.4),
      "★  Best Overall", 16, bold=True, color=GREEN)
txbox(s, Inches(0.65), Inches(1.72), Inches(5.2), Inches(0.45),
      "Llama-3.1-8B-Instruct", 22, bold=True, color=WHITE)
winner_pts = [
    "Parse success:    75.7%  (highest across all configs)",
    "Hallucination:    0.345  (lowest — most faithful)",
    "CQ coverage:      0.729  (good on successful runs)",
    "Consistent across ALL 5 prompt configurations",
    "Open-source · 8B params · GPU-deployable",
]
for k, pt in enumerate(winner_pts):
    txbox(s, Inches(0.65), Inches(2.22)+k*Inches(0.42), Inches(5.2), Inches(0.38),
          f"• {pt}", 11, color=WHITE)

# silver
box(s, Inches(0.5), Inches(4.95), Inches(2.65), Inches(2.38),
    RGBColor(0x12, 0x12, 0x0a), RGBColor(0x80, 0x70, 0x20), 2)
txbox(s, Inches(0.62), Inches(5.05), Inches(2.4), Inches(0.35),
      "🥈  Best Quality When Parsed", 10, bold=True, color=YELLOW)
txbox(s, Inches(0.62), Inches(5.38), Inches(2.4), Inches(0.3),
      "GPT-5.4", 16, bold=True, color=WHITE)
for k, pt in enumerate(["CQ coverage: 0.804 (best)", "Combined F1: 0.110 (best)",
                         "But: 0% parse rate on complex configs",
                         "High hallucination: 0.585"]):
    txbox(s, Inches(0.62), Inches(5.72)+k*Inches(0.38), Inches(2.4), Inches(0.35),
          f"• {pt}", 10, color=WHITE)

# bronze
box(s, Inches(3.3), Inches(4.95), Inches(2.65), Inches(2.38),
    RGBColor(0x10, 0x0d, 0x06), RGBColor(0x80, 0x50, 0x20), 2)
txbox(s, Inches(3.42), Inches(5.05), Inches(2.4), Inches(0.35),
      "🥉  Best Runner-Up", 10, bold=True, color=ACCENT2)
txbox(s, Inches(3.42), Inches(5.38), Inches(2.4), Inches(0.3),
      "Llama-2-70B-chat", 16, bold=True, color=WHITE)
for k, pt in enumerate(["Parse rate: 65.7% (2nd)", "CQ coverage: 0.689",
                         "Consistent across 4/5 configs",
                         "More params ≠ better than 3.1-8B"]):
    txbox(s, Inches(3.42), Inches(5.72)+k*Inches(0.38), Inches(2.4), Inches(0.35),
          f"• {pt}", 10, color=WHITE)

# key findings
txbox(s, Inches(6.3), Inches(1.2), Inches(6.5), Inches(0.35),
      "Key Findings", 16, bold=True, color=ACCENT2)
findings = [
    (GREEN,  "F1", "Model size ≠ performance. Llama-3.1-8B (8B) outperforms Llama-2-70B (70B) and GPT-5.4 on parse reliability."),
    (GREEN,  "F2", "Instruction-following quality matters more than raw capacity for structured Turtle generation."),
    (YELLOW, "F3", "CQ-only prompt maximises CQ coverage (0.844) but introduces most hallucination (0.548). Scenario context reduces hallucination by 24%."),
    (ACCENT2,"F4", "Reasoning-guided prompts (scenario-cq-reasoning) achieve lowest hallucination (0.367) — best faithfulness-quality tradeoff."),
    (RED,    "F5", "BLOOM-family models (3B, 7B): 0% parse success. Base models without instruction tuning are unsuitable for ODP generation."),
    (MUTED,  "F6", "Ground-truth F1 universally low (0.06–0.11): LLMs diverge in naming from expert ontologists even when semantically correct."),
]
for k, (fc, tag, text) in enumerate(findings):
    fy = Inches(1.65) + k * Inches(0.8)
    box(s, Inches(6.3), fy, Inches(0.5), Inches(0.65),
        RGBColor(fc[0]//8, fc[1]//8, fc[2]//8),
        RGBColor(fc[0]//3, fc[1]//3, fc[2]//3), 1)
    txbox(s, Inches(6.3), fy+Pt(8), Inches(0.5), Inches(0.5),
          tag, 10, bold=True, color=fc, align=PP_ALIGN.CENTER)
    txbox(s, Inches(6.88), fy+Pt(8), Inches(5.85), Inches(0.58),
          text, 11, color=WHITE)

callout(s, Inches(0.5), Inches(7.08), Inches(12.3), Inches(0.32),
        "Recommendation: Llama-3.1-8B-Instruct + scenario-cq-reasoning = best balance of syntactic validity, "
        "faithfulness & hallucination control for open-source ODP generation.", GREEN)


# ────────────────────────────────────────────────────────────
# SAVE
# ────────────────────────────────────────────────────────────
out = "ODPGen_Presentation.pptx"
prs.save(out)
print(f"Saved: {out}")
print(f"Slides: {len(prs.slides)}")
