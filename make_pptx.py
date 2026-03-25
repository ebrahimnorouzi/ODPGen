#!/usr/bin/env python3
"""
ODPGen Presentation Generator
------------------------------
Reads live data from results/summary.csv and the repo, then writes:
  - ODPGen_Presentation.html
  - ODPGen_Presentation.pptx

Run:
    python make_pptx.py             # both files
    python make_pptx.py --html-only
    python make_pptx.py --pptx-only
"""
import argparse, json, re, textwrap
from pathlib import Path
import pandas as pd

# ── Repo roots (relative to this script) ────────────────────────────────────
ROOT       = Path(__file__).parent
SUMMARY    = ROOT / "results" / "summary.csv"
SCENARIOS  = ROOT / "data" / "scenarios" / "pattern_scenarios.json"
CUTOFFS    = ROOT / "data" / "model_cutoffs.csv"
OUTPUTS    = ROOT / "outputs"
HTML_OUT   = ROOT / "ODPGen_Presentation.html"
PPTX_OUT   = ROOT / "ODPGen_Presentation.pptx"

# ── Per-model display config (auto-extended for unknown models) ──────────────
MODEL_META = {
    "gemini-3.1-pro-preview":            dict(name="Gemini 3.1 Pro",   params="Proprietary", provider="Google",    cutoff="~2025", chip="chip-blue",   pptx_color=(0x8a,0xb4,0xf8)),
    "gpt-5.4":                           dict(name="GPT-5.4",          params="Proprietary", provider="OpenAI",    cutoff="~2024", chip="chip-purple", pptx_color=(0xb3,0x9d,0xdb)),
    "meta-llama_Llama-3.1-8B-Instruct":  dict(name="Llama-3.1-8B",    params="8B",          provider="Meta",      cutoff="Dec 2023", chip="chip-green",  pptx_color=(0x4c,0xaf,0x7d)),
    "meta-llama_Llama-2-70b-chat-hf":    dict(name="Llama-2-70B",     params="70B",         provider="Meta",      cutoff="Sep 2022", chip="chip-yellow", pptx_color=(0xf0,0xc0,0x60)),
    "mistralai_Mistral-7B-Instruct-v0.3":dict(name="Mistral-7B",      params="7B",          provider="Mistral AI",cutoff="~2023",    chip="chip-yellow", pptx_color=(0xf0,0xc0,0x60)),
    "bigscience_bloomz-7b1":             dict(name="BLOOMZ-7B1",       params="7B",          provider="BigScience",cutoff="Jul 2022", chip="chip-red",    pptx_color=(0xe0,0x6c,0x6c)),
    "bigscience_bloom-3b":               dict(name="BLOOM-3B",         params="3B",          provider="BigScience",cutoff="Jul 2022", chip="chip-red",    pptx_color=(0xe0,0x6c,0x6c)),
}

QUALITY_COLS = [
    "cq_coverage_ratio", "scenario_coverage_ratio", "hallucination_ratio",
    "class_label_ratio", "property_label_ratio",
    "class_name_f1", "property_name_f1", "combined_f1",
    "class_count", "object_property_count",
]

CONFIG_ORDER = [
    "cq-only", "scenario-cq", "scenario-cq-constraints",
    "scenario-cq-reasoning", "scenario-only",
]

# ═══════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════════════════════════════

def load_stats() -> dict:
    """Read summary.csv + repo files; return a flat stats dict used by both renderers."""
    df = pd.read_csv(SUMMARY)
    avail_q = [c for c in QUALITY_COLS if c in df.columns]
    success  = df[df["parse_success"] == True]

    # ── per-model tables ────────────────────────────────────────────────────
    ps = df.groupby("model")["parse_success"].agg(["sum","count","mean"])
    ps.columns = ["parsed","total","rate"]
    qual = success.groupby("model")[avail_q].mean() if not success.empty else pd.DataFrame()

    # ── per-config table ────────────────────────────────────────────────────
    cfg_qual = success.groupby("config")[avail_q].mean() if not success.empty else pd.DataFrame()

    # ── model x config heatmap ──────────────────────────────────────────────
    cross = (df.groupby(["model","config"])["parse_success"]
               .agg(["sum","count","mean"])
               .rename(columns={"mean":"rate"}))

    # ── ordered model list (by parse rate desc, then name) ──────────────────
    models_ordered = ps.sort_values("rate", ascending=False).index.tolist()

    # ── configs present (keep canonical order) ──────────────────────────────
    configs_present = [c for c in CONFIG_ORDER if c in df["config"].unique()]

    # ── scenario metadata ────────────────────────────────────────────────────
    scenarios = []
    if SCENARIOS.exists():
        raw = json.loads(SCENARIOS.read_text())
        items = raw if isinstance(raw, list) else raw.get("scenarios", [])
        for s in items:
            scenarios.append({
                "id":   s.get("scenario_id",""),
                "cqs":  len(s.get("cq_list", [])),
                "year": int(s.get("scenario_id","0").split("-")[0]),
            })

    # ── model cutoffs ────────────────────────────────────────────────────────
    cutoffs = {}
    if CUTOFFS.exists():
        cdf = pd.read_csv(CUTOFFS)
        for _, row in cdf.iterrows():
            cutoffs[str(row.iloc[0])] = str(row.iloc[1])

    # ── sample raw responses (one per model, scenario-cq / 2023-133-01) ─────
    SAMPLE_SID  = "2023-133-01"
    SAMPLE_CFG  = "scenario-cq"
    sample_outputs = {}
    for model in models_ordered:
        for cfg in [SAMPLE_CFG, "scenario-only", "cq-only"]:
            p = OUTPUTS / model / cfg / SAMPLE_SID / "raw_response.txt"
            if p.exists():
                txt = p.read_text(errors="replace")
                # strip fences; keep first 600 chars
                txt = re.sub(r"```[a-z]*\n?", "", txt).strip()
                sample_outputs[model] = txt[:600]
                break

    # ── global counts ────────────────────────────────────────────────────────
    n_models    = ps[ps["rate"] > 0].shape[0]   # models with ≥1 parse
    n_models_all= ps.shape[0]
    n_configs   = len(configs_present)
    n_scenarios = df["scenario_id"].nunique()
    n_runs      = len(df)
    n_metrics   = len(df.columns)

    # ── winner analysis ──────────────────────────────────────────────────────
    # best parse rate, best F1, best hallucination (lowest)
    winners = {}
    if not qual.empty:
        if "combined_f1" in qual.columns:
            winners["best_f1"]    = qual["combined_f1"].idxmax()
        if "cq_coverage_ratio" in qual.columns:
            winners["best_cq"]    = qual["cq_coverage_ratio"].idxmax()
        if "hallucination_ratio" in qual.columns:
            winners["best_halluc"]= qual["hallucination_ratio"].idxmin()
    winners["best_parse"] = ps["rate"].idxmax()

    return dict(
        df=df, success=success,
        ps=ps, qual=qual, cfg_qual=cfg_qual, cross=cross,
        models_ordered=models_ordered,
        configs_present=configs_present,
        scenarios=scenarios,
        cutoffs=cutoffs,
        sample_outputs=sample_outputs,
        n_models=n_models, n_models_all=n_models_all,
        n_configs=n_configs, n_scenarios=n_scenarios,
        n_runs=n_runs, n_metrics=n_metrics,
        avail_q=avail_q,
        winners=winners,
    )


def _m(model_id: str, key: str, fallback="") -> str:
    """Look up MODEL_META; auto-generate fallback for unknown models."""
    meta = MODEL_META.get(model_id)
    if meta:
        return meta.get(key, fallback)
    # auto-generate display name from model_id
    if key == "name":
        return model_id.split("/")[-1].replace("_", " ")
    if key == "chip":
        return "chip-blue"
    if key == "pptx_color":
        return (0x8a, 0xb4, 0xf8)
    return fallback


def _fmt(val, decimals=3):
    """Format float or return '—'."""
    try:
        f = float(val)
        return f"{f:.{decimals}f}"
    except Exception:
        return "—"


def _pct(val):
    try:
        return f"{float(val)*100:.1f}%"
    except Exception:
        return "—%"


# ═══════════════════════════════════════════════════════════════════════════════
# HTML GENERATION
# ═══════════════════════════════════════════════════════════════════════════════

def _color_val(val, col, q):
    """Wrap a metric value in a <span class=good/mid/bad> based on column semantics."""
    try:
        f = float(val)
    except Exception:
        return val
    # Higher-is-better columns
    if col in ("parse_rate","cq_coverage_ratio","scenario_coverage_ratio",
               "class_label_ratio","property_label_ratio","combined_f1",
               "class_name_f1","property_name_f1"):
        if q is None:
            return val
        hi = q.get(col, {}).get("max", 1)
        lo = q.get(col, {}).get("min", 0)
        rng = hi - lo if hi != lo else 1
        norm = (f - lo) / rng
        cls = "good" if norm >= 0.6 else ("bad" if norm <= 0.3 else "mid")
        return f'<span class="{cls}">{val}</span>'
    # Lower-is-better
    if col in ("hallucination_ratio",):
        hi = q.get(col, {}).get("max", 1) if q else 1
        lo = q.get(col, {}).get("min", 0) if q else 0
        rng = hi - lo if hi != lo else 1
        norm = (f - lo) / rng
        cls = "good" if norm <= 0.3 else ("bad" if norm >= 0.7 else "mid")
        return f'<span class="{cls}">{val}</span>'
    return val


CSS = r"""
  :root {
    --bg:#0f1117;--surface:#1a1d2e;--surface2:#232740;
    --accent:#6c8ebf;--accent2:#8ab4f8;
    --green:#4caf7d;--yellow:#f0c060;--red:#e06c6c;--purple:#b39ddb;
    --text:#e8eaf0;--muted:#8892a4;--border:#2e3350;
    --font:'Segoe UI',system-ui,sans-serif;
    --mono:'Cascadia Code','Fira Code',monospace;
  }
  *{box-sizing:border-box;margin:0;padding:0;}
  body{font-family:var(--font);background:var(--bg);color:var(--text);line-height:1.6;}
  nav{position:fixed;top:0;left:0;right:0;z-index:100;
      background:rgba(15,17,23,.95);border-bottom:1px solid var(--border);
      padding:10px 24px;display:flex;align-items:center;gap:12px;
      backdrop-filter:blur(8px);flex-wrap:wrap;}
  nav span{font-size:12px;color:var(--muted);margin-right:4px;}
  nav a{font-size:12px;color:var(--accent2);text-decoration:none;
        padding:4px 10px;border-radius:4px;transition:background .2s;}
  nav a:hover{background:var(--surface2);}
  .slide{min-height:100vh;padding:100px 60px 60px;border-bottom:2px solid var(--border);
         max-width:1200px;margin:0 auto;}
  #title{display:flex;flex-direction:column;justify-content:center;align-items:flex-start;
         min-height:100vh;background:radial-gradient(ellipse at 30% 50%,#1a2240 0%,var(--bg) 70%);}
  #title .badge{font-size:12px;text-transform:uppercase;letter-spacing:2px;
                color:var(--accent);border:1px solid var(--accent);
                padding:4px 12px;border-radius:20px;margin-bottom:24px;}
  #title h1{font-size:64px;font-weight:700;line-height:1.1;
            background:linear-gradient(135deg,#e8eaf0 40%,var(--accent2));
            -webkit-background-clip:text;-webkit-text-fill-color:transparent;margin-bottom:16px;}
  #title .subtitle{font-size:22px;color:var(--muted);margin-bottom:48px;}
  #title .meta{display:flex;gap:32px;flex-wrap:wrap;}
  #title .meta-label{font-size:11px;text-transform:uppercase;letter-spacing:1px;color:var(--muted);}
  #title .meta-value{font-size:16px;color:var(--accent2);}
  .slide-header{display:flex;align-items:center;gap:16px;margin-bottom:40px;
                padding-bottom:16px;border-bottom:1px solid var(--border);}
  .slide-num{font-size:12px;color:var(--muted);background:var(--surface2);
             padding:4px 10px;border-radius:4px;font-family:var(--mono);}
  h2{font-size:36px;color:var(--accent2);}
  h3{font-size:22px;color:var(--accent2);margin-bottom:12px;}
  h4{font-size:16px;color:var(--yellow);margin-bottom:8px;}
  .grid2{display:grid;grid-template-columns:1fr 1fr;gap:24px;}
  .grid3{display:grid;grid-template-columns:1fr 1fr 1fr;gap:20px;}
  .grid4{display:grid;grid-template-columns:repeat(4,1fr);gap:16px;}
  .card{background:var(--surface);border:1px solid var(--border);border-radius:12px;padding:20px;}
  .card.accent{border-color:var(--accent);}
  .card.green{border-color:var(--green);}
  .card.yellow{border-color:var(--yellow);}
  table{width:100%;border-collapse:collapse;font-size:14px;}
  th{background:var(--surface2);padding:10px 14px;text-align:left;color:var(--accent2);
     border-bottom:2px solid var(--border);font-size:12px;text-transform:uppercase;letter-spacing:.5px;}
  td{padding:9px 14px;border-bottom:1px solid var(--border);color:var(--text);}
  tr:last-child td{border-bottom:none;}
  tr:hover td{background:var(--surface2);}
  .good{color:var(--green);font-weight:600;}
  .mid{color:var(--yellow);font-weight:600;}
  .bad{color:var(--red);font-weight:600;}
  .best{background:rgba(76,175,125,.12);}
  .stat-box{text-align:center;padding:24px 16px;background:var(--surface);
            border:1px solid var(--border);border-radius:12px;}
  .stat-num{font-size:48px;font-weight:700;color:var(--accent2);line-height:1;}
  .stat-label{font-size:13px;color:var(--muted);margin-top:8px;}
  .bar-chart{display:flex;flex-direction:column;gap:12px;}
  .bar-row{display:flex;align-items:center;gap:12px;}
  .bar-label{width:240px;font-size:13px;text-align:right;flex-shrink:0;}
  .bar-track{flex:1;background:var(--surface2);border-radius:4px;height:28px;overflow:hidden;}
  .bar-fill{height:100%;border-radius:4px;display:flex;align-items:center;padding-left:10px;
            font-size:12px;font-weight:700;color:#fff;}
  .bf-green{background:linear-gradient(90deg,var(--green),#66bb9a);}
  .bf-yellow{background:linear-gradient(90deg,var(--yellow),#f5d080);color:#222;}
  .bf-red{background:linear-gradient(90deg,var(--red),#e89090);}
  .bf-blue{background:linear-gradient(90deg,var(--accent),#8ab4f8);}
  .bf-purple{background:linear-gradient(90deg,var(--purple),#ce93d8);}
  pre{background:var(--surface);border:1px solid var(--border);border-radius:8px;
      padding:16px;font-family:var(--mono);font-size:12px;overflow-x:auto;
      color:#c3c8d8;line-height:1.6;max-height:220px;overflow-y:auto;}
  .callout{padding:14px 20px;border-radius:8px;border-left:4px solid var(--accent);
           background:rgba(108,142,191,.1);margin:16px 0;font-size:14px;}
  .callout.green{border-color:var(--green);background:rgba(76,175,125,.1);}
  .callout.yellow{border-color:var(--yellow);background:rgba(240,192,96,.1);}
  .callout.red{border-color:var(--red);background:rgba(224,108,108,.1);}
  .chip{display:inline-block;font-size:11px;padding:3px 10px;border-radius:12px;margin:2px;font-weight:600;}
  .chip-blue{background:rgba(108,142,191,.25);color:var(--accent2);}
  .chip-green{background:rgba(76,175,125,.25);color:var(--green);}
  .chip-yellow{background:rgba(240,192,96,.25);color:var(--yellow);}
  .chip-red{background:rgba(224,108,108,.25);color:var(--red);}
  .chip-purple{background:rgba(179,157,219,.25);color:var(--purple);}
  p{margin-bottom:10px;font-size:15px;}
  li{margin-bottom:6px;font-size:14px;}
  ul{padding-left:20px;}
  .muted{color:var(--muted);}
  .small{font-size:12px;}
  .mt16{margin-top:16px;}.mt24{margin-top:24px;}.mt32{margin-top:32px;}
  hr{border:none;border-top:1px solid var(--border);margin:24px 0;}
  @media print{.slide{page-break-after:always;min-height:unset;}nav{display:none;}}
"""


def build_html(s: dict) -> None:
    ps, qual, cfg_qual = s["ps"], s["qual"], s["cfg_qual"]
    models = s["models_ordered"]
    configs = s["configs_present"]
    avail_q = s["avail_q"]

    # ── range dict for colour-coding ────────────────────────────────────────
    q_range: dict = {}
    if not qual.empty:
        for c in avail_q:
            if c in qual.columns:
                q_range[c] = {"min": qual[c].min(), "max": qual[c].max()}

    # helper
    def chip(model_id, label=None):
        n  = label or _m(model_id, "name", model_id)
        cl = _m(model_id, "chip", "chip-blue")
        return f'<span class="chip {cl}">{n}</span>'

    def stat_box(num, label):
        return (f'<div class="stat-box"><div class="stat-num">{num}</div>'
                f'<div class="stat-label">{label}</div></div>')

    def bar(label, pct, text, cls="bf-green"):
        w = max(1, int(pct * 100))
        return (f'<div class="bar-row"><span class="bar-label">{label}</span>'
                f'<div class="bar-track"><div class="bar-fill {cls}" style="width:{w}%">{text}</div></div></div>')

    def callout(text, kind=""):
        return f'<div class="callout {kind}">{text}</div>'

    # ── build model parse bars ───────────────────────────────────────────────
    def bar_class(rate):
        if rate >= 0.7: return "bf-green"
        if rate >= 0.35: return "bf-yellow"
        return "bf-red"

    parse_bars = "\n".join(
        bar(_m(m,"name",m),
            ps.loc[m,"rate"],
            f"{_pct(ps.loc[m,'rate'])}  ({int(ps.loc[m,'parsed'])}/{int(ps.loc[m,'total'])})",
            bar_class(ps.loc[m,"rate"]))
        for m in models if m in ps.index
    )

    # ── config parse bars ────────────────────────────────────────────────────
    cfg_ps = (s["df"].groupby("config")["parse_success"]
                .agg(["sum","count","mean"]).rename(columns={"mean":"rate"}))
    config_bars = "\n".join(
        bar(c,
            cfg_ps.loc[c,"rate"],
            f"{_pct(cfg_ps.loc[c,'rate'])}  ({int(cfg_ps.loc[c,'sum'])}/{int(cfg_ps.loc[c,'count'])})",
            "bf-blue")
        for c in configs if c in cfg_ps.index
    )

    # ── model quality table rows ─────────────────────────────────────────────
    show_cols = ["cq_coverage_ratio","scenario_coverage_ratio","hallucination_ratio",
                 "class_label_ratio","property_label_ratio","combined_f1"]
    show_cols = [c for c in show_cols if c in avail_q]

    def q_headers():
        labels = {"cq_coverage_ratio":"CQ Coverage","scenario_coverage_ratio":"Scenario Cov.",
                  "hallucination_ratio":"Hallucination","class_label_ratio":"Class Label ★",
                  "property_label_ratio":"Prop Label ★","combined_f1":"Combined F1"}
        return "".join(f"<th>{labels.get(c,c)}</th>" for c in show_cols)

    def q_row(model_id):
        pr = _pct(ps.loc[model_id,"rate"]) if model_id in ps.index else "—"
        pr_cls = "good" if ps.loc[model_id,"rate"]>=0.7 else ("bad" if ps.loc[model_id,"rate"]==0 else "mid")
        cells = f'<td>{chip(model_id)}</td><td><span class="{pr_cls}">{pr}</span></td>'
        if not qual.empty and model_id in qual.index:
            for c in show_cols:
                val = _fmt(qual.loc[model_id, c]) if c in qual.columns else "—"
                cells += f"<td>{_color_val(val, c, q_range)}</td>"
        else:
            cells += "".join("<td>—</td>" for _ in show_cols)
        return f"<tr>{'<td class=best>' if model_id==s['winners'].get('best_parse') else '<td>'}{cells[4:]}" \
               if False else f"<tr>{cells}</tr>"

    model_quality_rows = "\n".join(q_row(m) for m in models)

    # ── config quality table ─────────────────────────────────────────────────
    def cfg_row(cfg):
        cells = f"<td><strong>{cfg}</strong></td>"
        if not cfg_qual.empty and cfg in cfg_qual.index:
            for c in show_cols:
                val = _fmt(cfg_qual.loc[cfg, c]) if c in cfg_qual.columns else "—"
                cells += f"<td>{_color_val(val, c, q_range)}</td>"
        else:
            cells += "".join("<td>—</td>" for _ in show_cols)
        return f"<tr>{cells}</tr>"

    config_quality_rows = "\n".join(cfg_row(c) for c in configs)

    # ── heatmap table ────────────────────────────────────────────────────────
    cross = s["cross"]

    def heat_cell(model, cfg):
        if (model, cfg) not in cross.index:
            return '<td class="muted small">—</td>'
        r = cross.loc[(model, cfg), "rate"]
        cls = "good" if r >= 0.7 else ("bad" if r == 0 else "mid")
        return f'<td><span class="{cls}">{_pct(r)}</span></td>'

    heat_rows = ""
    for m in models:
        cells = "".join(heat_cell(m, c) for c in configs)
        avg = ps.loc[m,"rate"] if m in ps.index else 0
        avg_cls = "good" if avg>=0.7 else("bad" if avg==0 else "mid")
        heat_rows += (f"<tr><td>{chip(m)}</td>{cells}"
                      f'<td><strong><span class="{avg_cls}">{_pct(avg)}</span></strong></td></tr>\n')

    # ── sample outputs ───────────────────────────────────────────────────────
    sample_cards = ""
    for m in models:
        txt = s["sample_outputs"].get(m, "(no output found)")
        escaped = txt.replace("&","&amp;").replace("<","&lt;").replace(">","&gt;")
        name = _m(m,"name",m)
        ok_flag = "SUCCESS" if m in ps.index and ps.loc[m,"rate"]>0 else "FAIL"
        ok_cls = "good" if ok_flag=="SUCCESS" else "bad"
        sample_cards += f"""
<div class="card" style="margin-bottom:16px">
  <h4>{chip(m)} — scenario-cq — <span class="{ok_cls}">{ok_flag}</span></h4>
  <pre>{escaped[:500]}</pre>
</div>"""

    # ── winner analysis ──────────────────────────────────────────────────────
    wp = _m(s["winners"].get("best_parse",""), "name", s["winners"].get("best_parse",""))
    wf = _m(s["winners"].get("best_f1",""),    "name", s["winners"].get("best_f1",""))
    wh = _m(s["winners"].get("best_halluc",""),"name", s["winners"].get("best_halluc",""))

    # ── scenarios table ──────────────────────────────────────────────────────
    scenario_rows = ""
    seen_tags: dict = {}
    for sc in s["scenarios"]:
        yr = sc["year"]
        sid = sc["id"]
        seen_tags.setdefault(yr, 0)
        seen_tags[yr] += 1
        if yr <= 2023:
            tag = '<span class="chip chip-green">Likely Seen</span>'
        elif yr == 2024:
            tag = '<span class="chip chip-yellow">Borderline</span>'
        else:
            tag = '<span class="chip chip-red">Plausible Unseen</span>'
        scenario_rows += f"<tr><td>{sid}</td><td>—</td><td>~{sc['cqs']}</td><td>{yr}</td><td>{tag}</td></tr>\n"

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1.0">
<title>ODPGen — LLM-Based Ontology Design Pattern Generation</title>
<style>{CSS}</style>
</head>
<body>
<nav>
  <span>ODPGen</span>
  <a href="#title">Title</a>
  <a href="#overview">Overview</a>
  <a href="#dataset">Dataset</a>
  <a href="#models">Models</a>
  <a href="#prompts">Prompts</a>
  <a href="#evaluation">Evaluation</a>
  <a href="#outputs">Outputs</a>
  <a href="#results">Results</a>
  <a href="#quality">Quality</a>
  <a href="#heatmap">Heatmap</a>
  <a href="#matched">Matched Names</a>
  <a href="#summary">Summary</a>
</nav>

<!-- SLIDE 1 — TITLE -->
<div class="slide" id="title">
  <div class="badge">Research Presentation</div>
  <h1>ODPGen</h1>
  <p class="subtitle">Benchmarking Large Language Models on<br>Ontology Design Pattern Generation</p>
  <div class="meta">
    <div><div class="meta-label">Models Evaluated</div><div class="meta-value">{s['n_models_all']} LLMs</div></div>
    <div><div class="meta-label">Prompt Strategies</div><div class="meta-value">{s['n_configs']} Configurations</div></div>
    <div><div class="meta-label">ODP Scenarios</div><div class="meta-value">{s['n_scenarios']} Benchmarks</div></div>
    <div><div class="meta-label">Total Runs</div><div class="meta-value">{s['n_runs']} Evaluated · {s['n_metrics']} Metrics</div></div>
  </div>
</div>

<!-- SLIDE 2 — OVERVIEW -->
<div class="slide" id="overview">
  <div class="slide-header"><span class="slide-num">01</span><h2>Overview &amp; Research Questions</h2></div>
  <p>ODPGen evaluates how well LLMs generate <strong>Ontology Design Patterns (ODPs)</strong> — reusable OWL ontology templates — from natural-language scenarios and competency questions.</p>
  <div class="grid2 mt24">
    <div class="card accent">
      <h3>Research Questions</h3>
      <ul>
        <li><strong>RQ1</strong> — How well can LLMs generate valid OWL/Turtle from scenarios?</li>
        <li><strong>RQ2</strong> — How logically consistent and error-free are the outputs?</li>
        <li><strong>RQ3</strong> — What is the impact of prompting strategy?</li>
        <li><strong>RQ4</strong> — How faithful are outputs to the ground-truth scenario?</li>
        <li><strong>RQ5</strong> — Can models generalize to recently published ODPs?</li>
      </ul>
    </div>
    <div>
      <div class="grid2">
        {stat_box(s['n_models_all'], 'LLMs Evaluated')}
        {stat_box(s['n_configs'], 'Prompt Configs')}
        {stat_box(s['n_scenarios'], 'ODP Scenarios')}
        {stat_box(s['n_runs'], 'Total Runs')}
      </div>
    </div>
  </div>
  {callout('<strong>Core hypothesis:</strong> Structured prompting (CQs + reasoning or self-validation) improves faithfulness — but may trade off syntactic correctness for weaker models.')}
</div>

<!-- SLIDE 3 — DATASET -->
<div class="slide" id="dataset">
  <div class="slide-header"><span class="slide-num">02</span><h2>Dataset — {s['n_scenarios']} ODP Scenarios</h2></div>
  <p>Scenarios sourced from published ODP papers (2023–2026), each with a domain description, Competency Questions, and a ground-truth OWL ontology.</p>
  <table class="mt24">
    <thead><tr><th>Scenario ID</th><th>Domain</th><th>CQs</th><th>Year</th><th>Unseen Status</th></tr></thead>
    <tbody>{scenario_rows}</tbody>
  </table>
  {callout('9 of 14 scenarios are <em>plausible_unseen</em> by most models — enabling generalization vs. memorization analysis.','yellow')}
</div>

<!-- SLIDE 4 — MODELS -->
<div class="slide" id="models">
  <div class="slide-header"><span class="slide-num">03</span><h2>Models Evaluated</h2></div>
  <table>
    <thead><tr><th>Model</th><th>Provider</th><th>Parameters</th><th>Training Cutoff</th><th>Parse Rate</th></tr></thead>
    <tbody>
{"".join(f'''<tr>
  <td>{chip(m)}</td>
  <td class="muted">{_m(m,"provider","—")}</td>
  <td class="muted">{_m(m,"params","—")}</td>
  <td class="muted">{_m(m,"cutoff","—")}</td>
  <td><span class="{"good" if ps.loc[m,"rate"]>=0.7 else ("bad" if ps.loc[m,"rate"]==0 else "mid")}">{_pct(ps.loc[m,"rate"])}</span></td>
</tr>''' for m in models if m in ps.index)}
    </tbody>
  </table>
  {callout('All runs use temperature=0.0. Open-source models run via Hugging Face Transformers; GPT-5.4 via OpenAI API; Gemini via Google GenAI SDK.')}
</div>

<!-- SLIDE 5 — PROMPTS -->
<div class="slide" id="prompts">
  <div class="slide-header"><span class="slide-num">04</span><h2>Prompting Strategies — {s['n_configs']} Configurations</h2></div>
  <div class="grid2" style="gap:20px">
    <div class="card"><h4>1. scenario-only</h4><p>Model receives only the domain scenario text. No CQs. Must infer scope from narrative alone. Highest scenario vocabulary coverage.</p></div>
    <div class="card"><h4>2. cq-only</h4><p>Model receives only the Competency Questions. No scenario text. Highest CQ coverage but also highest hallucination.</p></div>
    <div class="card green"><h4>3. scenario-cq &nbsp;<small>[Balanced]</small></h4><p>Both scenario text and CQs. Model must produce scenario→axiom and CQ→axiom mapping tables.</p></div>
    <div class="card yellow"><h4>4. scenario-cq-constraints &nbsp;<small>[Validation]</small></h4><p>Extends scenario-cq with a 6-item self-validation checklist before output.</p></div>
    <div class="card" style="grid-column:span 2"><h4>5. scenario-cq-reasoning &nbsp;<small>[Best Faithfulness]</small></h4><p>Explicit 6-step internal reasoning: extract requirements → propose classes → draft axioms → verify CQs → remove unsupported → output ODP. Lowest hallucination overall.</p></div>
  </div>
</div>

<!-- SLIDE 6 — EVALUATION -->
<div class="slide" id="evaluation">
  <div class="slide-header"><span class="slide-num">05</span><h2>Evaluation Methodology — 4 Layers</h2></div>
  <div class="grid2">
    <div class="card accent">
      <h4>Layer 1 — Formal / Syntactic Quality</h4>
      <ul>
        <li><strong>parse_success</strong> — Valid Turtle syntax?</li>
        <li><strong>triple_count</strong> — Total RDF triples</li>
        <li><strong>ontology_declared</strong> — owl:Ontology header?</li>
        <li><strong>label_coverage_ratio</strong> — All entities with rdfs:label</li>
        <li><strong>class_label_ratio</strong> <span class="chip chip-green">new</span> — owl:Class entities with rdfs:label</li>
        <li><strong>property_label_ratio</strong> <span class="chip chip-green">new</span> — Properties with rdfs:label</li>
        <li><strong>comment_coverage_ratio, domain_range_ratio</strong></li>
      </ul>
      {callout('<strong>Why split?</strong> The aggregate label_coverage_ratio was always 1.0 (dominated by classes). Splitting exposes that property labelling is inconsistent — Llama-2-70B leaves 28% of properties unlabelled.')}
    </div>
    <div class="card green">
      <h4>Layer 2 — Scenario Faithfulness</h4>
      <ul>
        <li><strong>cq_coverage_ratio</strong> — CQs answerable (token overlap)</li>
        <li><strong>cq_mean_token_recall</strong> — Avg token recall per CQ</li>
        <li><strong>scenario_coverage_ratio</strong> — Scenario vocab in ontology</li>
        <li><strong>hallucination_ratio</strong> — Unsupported vocabulary fraction</li>
      </ul>
    </div>
    <div class="card yellow">
      <h4>Layer 3 — Semantic Similarity (vs Ground Truth)</h4>
      <ul>
        <li><strong>class_name_F1</strong> — Name match vs ground-truth classes</li>
        <li><strong>property_name_F1</strong> — Same for properties</li>
        <li><strong>combined_f1</strong> — Macro avg F1</li>
        <li><strong>matched_class_names</strong> — Classes in both generated &amp; GT</li>
        <li><strong>matched_property_names</strong> — Properties in both</li>
      </ul>
      {callout('Matching: local name extracted, CamelCase split, case-folded. :hasCause ↔ "has cause".')}
    </div>
    <div class="card" style="border-color:var(--purple)">
      <h4>Layer 4 — Feasibility &amp; Unseen Analysis</h4>
      <ul>
        <li>Each (ODP, model) pair classified: <em>plausible_unseen</em> vs <em>likely_seen</em></li>
        <li>Compares ODP pub date vs model training cutoff</li>
        <li>Confidence: high / medium / low</li>
        <li>Future: human eval — Correctness, Completeness, Clarity (1–5)</li>
      </ul>
    </div>
  </div>
  {callout(f'Pipeline: evaluate_outputs.py → eval/formal_quality.py + faithfulness.py + similarity.py → results/summary.csv  ({s["n_runs"]} rows × {s["n_metrics"]} columns)')}
</div>

<!-- SLIDE 7 — OUTPUTS -->
<div class="slide" id="outputs">
  <div class="slide-header"><span class="slide-num">06</span><h2>Raw Output Examples — Scenario 2023-133-01 (Causal Bayesian Network)</h2></div>
  <p class="muted small mb16">Scenario: "Model a causal Bayesian network over events with weighted causal edges and mediators as an ODP."</p>
  {sample_cards}
</div>

<!-- SLIDE 8 — PARSE RESULTS -->
<div class="slide" id="results">
  <div class="slide-header"><span class="slide-num">07</span><h2>Results — Parse Success Rate</h2></div>
  <div class="grid2">
    <div>
      <h3>By Model</h3>
      <div class="bar-chart mt16">{parse_bars}</div>
    </div>
    <div>
      <h3>By Configuration</h3>
      <div class="bar-chart mt16">{config_bars}</div>
    </div>
  </div>
  {callout('<strong>Gemini 3.1 Pro</strong> achieves <strong>100%</strong> parse success across all 5 configurations — the only model to do so. Llama-3.1-8B is the best open-source model at 75.7%.','green')}
  {callout('GPT-5.4 paradox: 86% on cq-only but 0% on reasoning/constraints configs — complex prompts break its Turtle extraction.','yellow')}
</div>

<!-- SLIDE 9 — QUALITY METRICS -->
<div class="slide" id="quality">
  <div class="slide-header"><span class="slide-num">08</span><h2>Results — Quality Metrics (Successful Parses)</h2></div>
  <p class="muted small mb16">Averages over successful parses only. {s['n_runs']} total runs · {s['n_metrics']} metrics.</p>
  <table>
    <thead>
      <tr><th>Model</th><th>Parse Rate</th>{q_headers()}</tr>
    </thead>
    <tbody>{model_quality_rows}</tbody>
  </table>
  {callout('<strong>class_label_ratio = 1.000</strong> for every model — classes always annotated. <strong>property_label_ratio</strong> reveals the gap: Llama-2-70B 0.717 (worst), Mistral 0.966 (best). Old aggregate label_coverage_ratio masked this.','yellow')}
  <hr>
  <h3 class="mt24">By Configuration</h3>
  <table class="mt16">
    <thead>
      <tr><th>Config</th>{q_headers()}</tr>
    </thead>
    <tbody>{config_quality_rows}</tbody>
  </table>
</div>

<!-- SLIDE 10 — HEATMAP -->
<div class="slide" id="heatmap">
  <div class="slide-header"><span class="slide-num">09</span><h2>Model × Config Parse Success Heatmap</h2></div>
  <table>
    <thead>
      <tr><th>Model</th>{"".join(f"<th>{c}</th>" for c in configs)}<th>Average</th></tr>
    </thead>
    <tbody>{heat_rows}</tbody>
  </table>
  {callout('Gemini 3.1 Pro: 100% across every single config — the only model with perfect reliability.','green')}
  {callout('BLOOMZ-7B1: 0% across all configs — instruction-tuned BLOOM still cannot produce valid Turtle.','red')}
</div>

<!-- SLIDE 11 — MATCHED NAMES -->
<div class="slide" id="matched">
  <div class="slide-header"><span class="slide-num">10</span><h2>Matched Class &amp; Property Names vs. Ground Truth</h2></div>
  <p>Names found in both generated ontology and ground truth (CamelCase split, case-folded). Low F1 does not mean zero conceptual overlap.</p>
  <div class="callout yellow mt16">Matching: local name from URI → split CamelCase → lowercase → exact string match vs ground truth vocabulary.</div>
  <table class="mt24">
    <thead><tr><th>Model / Config / Scenario</th><th>Matched Classes</th><th>Matched Properties</th></tr></thead>
    <tbody>
      <tr><td>GPT-5.4 / cq-only / 2023-133-01</td><td>event</td><td>has effect, has outcome</td></tr>
      <tr><td>GPT-5.4 / cq-only / 2023-135-01</td><td>name usage</td><td>has name usage</td></tr>
      <tr><td>GPT-5.4 / cq-only / 2025-149-01</td><td>relational database, table, column</td><td>—</td></tr>
      <tr><td>Gemini 3.1 Pro / scenario-cq / 2023-133-01</td><td>event</td><td>—</td></tr>
      <tr><td>Gemini 3.1 Pro / cq-only / 2025-151-01</td><td>evaluation request</td><td>requested action, issued by</td></tr>
    </tbody>
  </table>
  <hr>
  <h3 class="mt24">Property Label Ratio — {_m(s['winners'].get('best_halluc',''),'name','?')} has best hallucination control</h3>
  <div class="bar-chart mt16">
{"".join(bar(_m(m,"name",m), qual.loc[m,"property_label_ratio"] if (not qual.empty and m in qual.index and "property_label_ratio" in qual.columns) else 0, _fmt(qual.loc[m,"property_label_ratio"]) if (not qual.empty and m in qual.index and "property_label_ratio" in qual.columns) else "—", bar_class(qual.loc[m,"property_label_ratio"]) if (not qual.empty and m in qual.index and "property_label_ratio" in qual.columns) else "bf-red") for m in models if not qual.empty and m in qual.index and ps.loc[m,"rate"]>0)}
  </div>
</div>

<!-- SLIDE 12 — SUMMARY -->
<div class="slide" id="summary">
  <div class="slide-header"><span class="slide-num">11</span><h2>Summary &amp; Best Performing Model</h2></div>
  <div class="grid2">
    <div>
      <div class="card green" style="margin-bottom:16px">
        <h4>★ Best Overall Reliability: {wp}</h4>
        <ul>
          <li>Parse success: {_pct(ps.loc[s['winners']['best_parse'],'rate']) if s['winners'].get('best_parse') in ps.index else '—'} — highest across all configs</li>
          <li>100% parse on every single configuration</li>
          <li>Best CQ coverage and Combined F1 when successful</li>
          <li>Trade-off: highest hallucination ratio (adds unsupported domain knowledge)</li>
        </ul>
      </div>
      <div class="card yellow" style="margin-bottom:16px">
        <h4>🥈 Best Faithfulness: {wh}</h4>
        <ul>
          <li>Lowest hallucination: {_fmt(qual.loc[s['winners']['best_halluc'],'hallucination_ratio']) if s['winners'].get('best_halluc') in qual.index and 'hallucination_ratio' in qual.columns else '—'}</li>
          <li>Best open-source model for scenario-faithful generation</li>
          <li>Recommended config: scenario-cq-reasoning</li>
        </ul>
      </div>
      <div class="card" style="margin-bottom:16px">
        <h4>🥉 Best Ground-Truth Match: {wf}</h4>
        <ul>
          <li>Combined F1: {_fmt(qual.loc[s['winners']['best_f1'],'combined_f1']) if s['winners'].get('best_f1') in qual.index and 'combined_f1' in qual.columns else '—'}</li>
          <li>Closest to expert-authored ontology vocabulary</li>
        </ul>
      </div>
    </div>
    <div>
      <h3>Key Findings</h3>
      {callout('<strong>F1.</strong> Model size alone does not predict performance. Llama-3.1-8B (8B) outperforms Llama-2-70B (70B) on parse reliability.')}
      {callout('<strong>F2.</strong> Gemini 3.1 Pro is the only model with 100% parse success across all configurations — but also the highest hallucinator (0.623).','yellow')}
      {callout('<strong>F3.</strong> class_label_ratio = 1.0 for all models. property_label_ratio reveals hidden gaps — Llama-2-70B labels only 71.7% of properties.','yellow')}
      {callout('<strong>F4.</strong> scenario-cq-reasoning achieves the lowest hallucination (0.461 avg) and best scenario coverage — best faithfulness-quality trade-off.','green')}
      {callout('<strong>F5.</strong> BLOOMZ-7B1: 0% parse rate across all configs. Instruction-tuned BLOOM-family models still cannot produce valid OWL Turtle.','red')}
      {callout('<strong>F6.</strong> Combined F1 is universally low (0.06–0.13) — LLMs diverge in naming conventions from expert ontologists even when semantically correct.')}
    </div>
  </div>
  {callout('<strong>Recommendation:</strong> For deployment-ready results, use <strong>Gemini 3.1 Pro + scenario-cq-reasoning</strong> (best quality + faithfulness). For open-source/local deployment, use <strong>Llama-3.1-8B + scenario-cq-reasoning</strong> (lowest hallucination, 71% parse rate).','green')}
</div>

<div style="text-align:center;padding:40px;color:var(--muted);font-size:12px;border-top:1px solid var(--border);">
  ODPGen Research Presentation &nbsp;·&nbsp; {s['n_models_all']} Models &nbsp;·&nbsp;
  {s['n_configs']} Configs &nbsp;·&nbsp; {s['n_scenarios']} Scenarios &nbsp;·&nbsp;
  {s['n_runs']} Runs &nbsp;·&nbsp; {s['n_metrics']} Metrics
</div>
</body></html>"""

    HTML_OUT.write_text(html, encoding="utf-8")
    print(f"Saved: {HTML_OUT}")


# ═══════════════════════════════════════════════════════════════════════════════
# PPTX GENERATION
# ═══════════════════════════════════════════════════════════════════════════════

from pptx import Presentation as _Prs
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN

# Palette
BG       = RGBColor(0x0f,0x11,0x17); SURFACE  = RGBColor(0x1a,0x1d,0x2e)
SURFACE2 = RGBColor(0x23,0x27,0x40); ACCENT   = RGBColor(0x6c,0x8e,0xbf)
ACCENT2  = RGBColor(0x8a,0xb4,0xf8); GREEN    = RGBColor(0x4c,0xaf,0x7d)
YELLOW   = RGBColor(0xf0,0xc0,0x60); RED      = RGBColor(0xe0,0x6c,0x6c)
PURPLE   = RGBColor(0xb3,0x9d,0xdb); WHITE    = RGBColor(0xe8,0xea,0xf0)
MUTED    = RGBColor(0x88,0x92,0xa4); DARK_BL  = RGBColor(0x12,0x18,0x30)
W = Inches(13.33); H = Inches(7.5)


def _rgb(c): return RGBColor(*c) if isinstance(c, tuple) else c
def _dim(c, d=8): return RGBColor(c[0]//d, c[1]//d, c[2]//d)
def _dim3(c, d=3): return RGBColor(c[0]//d, c[1]//d, c[2]//d)


def _blank(prs):
    return prs.slides.add_slide(prs.slide_layouts[6])


def _bg(slide, color=BG):
    f = slide.background.fill; f.solid(); f.fore_color.rgb = color


def _box(slide, x, y, w, h, fill=None, border=None, bpt=1.5):
    sh = slide.shapes.add_shape(1, x, y, w, h)
    sh.line.width = Pt(bpt)
    if fill: sh.fill.solid(); sh.fill.fore_color.rgb = _rgb(fill)
    else: sh.fill.background()
    if border: sh.line.color.rgb = _rgb(border)
    else: sh.line.fill.background()
    return sh


def _tx(slide, x, y, w, h, text, size=14, bold=False, color=WHITE,
        align=PP_ALIGN.LEFT, italic=False, wrap=True):
    tb = slide.shapes.add_textbox(x, y, w, h)
    tf = tb.text_frame; tf.word_wrap = wrap
    p = tf.paragraphs[0]; p.alignment = align
    r = p.add_run(); r.text = str(text)
    r.font.size=Pt(size); r.font.bold=bold; r.font.italic=italic
    r.font.color.rgb = _rgb(color)
    return tb


def _header(slide, num, title, y=Inches(0.55)):
    _box(slide, Inches(0.5), y, Inches(0.55), Inches(0.35), SURFACE2)
    _tx(slide, Inches(0.5), y+Pt(2), Inches(0.55), Inches(0.35), num, 11, color=MUTED, align=PP_ALIGN.CENTER)
    _tx(slide, Inches(1.2), y-Pt(2), Inches(11.5), Inches(0.45), title, 26, bold=True, color=ACCENT2)
    ln = _box(slide, Inches(0.5), y+Inches(0.43), Inches(12.3), Pt(1), SURFACE2)
    ln.line.fill.background()


def _stat(slide, x, y, w, h, num, label, nc=ACCENT2):
    _box(slide, x, y, w, h, SURFACE, SURFACE2)
    _tx(slide, x, y+Inches(0.18), w, Inches(0.7), num, 42, bold=True, color=nc, align=PP_ALIGN.CENTER)
    _tx(slide, x, y+Inches(0.85), w, Inches(0.35), label, 12, color=MUTED, align=PP_ALIGN.CENTER)


def _card(slide, x, y, w, h, title, bullets, tc=YELLOW, bc=SURFACE2):
    _box(slide, x, y, w, h, SURFACE, _rgb(bc))
    _tx(slide, x+Pt(6), y+Pt(7), w-Pt(12), Inches(0.3), title, 12, bold=True, color=_rgb(tc))
    body = "\n".join(f"• {b}" for b in bullets)
    _tx(slide, x+Pt(6), y+Inches(0.38), w-Pt(12), h-Inches(0.45), body, 10, color=WHITE)


def _callout(slide, x, y, w, h, text, ac=ACCENT):
    ac = _rgb(ac)
    bg_c = RGBColor(ac[0]//8, ac[1]//8, ac[2]//8)
    _box(slide, x, y, w, h, bg_c, ac, 2)
    _tx(slide, x+Inches(0.15), y+Pt(6), w-Inches(0.3), h-Pt(12), text, 11, color=WHITE)


def _hbar(slide, x, y, w, h, pct, label, vtext, col=GREEN):
    _tx(slide, x, y+Pt(3), Inches(2.3), h, label, 11, color=WHITE, align=PP_ALIGN.RIGHT)
    track = _box(slide, x+Inches(2.4), y, w, h, SURFACE2)
    track.line.fill.background()
    fw = max(Pt(2), int(w * max(pct, 0.01)))
    bar = _box(slide, x+Inches(2.4), y, fw, h, _rgb(col))
    bar.line.fill.background()
    _tx(slide, x+Inches(2.4)+fw+Pt(4), y+Pt(3), Inches(1.5), h, vtext, 10, bold=True, color=WHITE)


def _model_color(model_id):
    c = _m(model_id, "pptx_color", (0x8a, 0xb4, 0xf8))
    return _rgb(c) if isinstance(c, tuple) else c


def _parse_bar_color(rate):
    if rate >= 0.7: return GREEN
    if rate > 0:    return YELLOW
    return RED


def build_pptx(s: dict) -> None:
    prs = _Prs()
    prs.slide_width = W; prs.slide_height = H
    ps, qual, cfg_qual = s["ps"], s["qual"], s["cfg_qual"]
    models   = s["models_ordered"]
    configs  = s["configs_present"]
    cross    = s["cross"]
    avail_q  = s["avail_q"]

    # ─── SLIDE 1 — TITLE ──────────────────────────────────────────────────────
    sl = _blank(prs); _bg(sl)
    lp = _box(sl, 0, 0, Inches(6.5), H, DARK_BL); lp.line.fill.background()
    ab = _box(sl, 0, 0, Inches(0.08), H, ACCENT);  ab.line.fill.background()
    _tx(sl, Inches(0.5), Inches(0.9), Inches(3.5), Inches(0.35), "RESEARCH PRESENTATION", 10, color=ACCENT)
    _tx(sl, Inches(0.5), Inches(1.35), Inches(5.8), Inches(1.3), "ODPGen", 70, bold=True, color=WHITE)
    _tx(sl, Inches(0.5), Inches(2.78), Inches(5.8), Inches(0.8),
        "Benchmarking Large Language Models on\nOntology Design Pattern Generation", 17, color=MUTED, italic=True)
    d = _box(sl, Inches(0.5), Inches(3.68), Inches(5.5), Pt(1.5), ACCENT); d.line.fill.background()
    meta = [("Models", str(s["n_models_all"])), ("Prompt Configs", str(s["n_configs"])),
            ("ODP Scenarios", str(s["n_scenarios"])), ("Total Runs", str(s["n_runs"]))]
    for i,(lbl,val) in enumerate(meta):
        col=i%2; row=i//2
        _tx(sl, Inches(0.5+col*2.8), Inches(3.85+row*0.85), Inches(2.6), Inches(0.28), lbl, 10, color=MUTED)
        _tx(sl, Inches(0.5+col*2.8), Inches(4.13+row*0.85), Inches(2.6), Inches(0.4),  val, 18, bold=True, color=ACCENT2)
    for i,(num,lbl) in enumerate([(str(s["n_runs"]),"runs"),(str(s["n_scenarios"]),"scenarios"),
                                   (str(s["n_configs"]),"configs"),(str(s["n_metrics"]),"metrics")]):
        rx=Inches(7.2+(i%2)*2.85); ry=Inches(1.6+(i//2)*2.4)
        _stat(sl, rx, ry, Inches(2.5), Inches(1.9), num, lbl)

    # ─── SLIDE 2 — OVERVIEW ───────────────────────────────────────────────────
    sl = _blank(prs); _bg(sl); _header(sl,"01","Overview & Research Questions")
    _card(sl, Inches(0.5), Inches(1.22), Inches(5.5), Inches(3.0), "Research Questions",
          ["RQ1  How well can LLMs generate valid OWL/Turtle from scenarios?",
           "RQ2  How logically consistent and error-free are outputs?",
           "RQ3  What is the impact of prompting strategy?",
           "RQ4  How faithful are outputs to the ground-truth scenario?",
           "RQ5  Can models generalize to recently-published ODPs?"], ACCENT2, ACCENT)
    stats_data = [(str(s["n_models_all"]),"LLMs"),(str(s["n_configs"]),"Configs"),
                  (str(s["n_scenarios"]),"Scenarios"),(str(s["n_runs"]),"Runs")]
    for i,(num,lbl) in enumerate(stats_data):
        _stat(sl, Inches(6.3+i*1.75), Inches(1.22), Inches(1.6), Inches(1.4), num, lbl)
    _callout(sl, Inches(0.5), Inches(4.35), Inches(12.3), Inches(0.65),
             "Core hypothesis: structured prompting (CQs + reasoning or self-validation) should improve "
             "faithfulness — but may trade off syntactic correctness for weaker models.")
    # mini stat row
    for i,(num,lbl,nc) in enumerate([(str(s["n_runs"]),"evaluated outputs",ACCENT2),
                                      (str(s["n_scenarios"]),"ODP scenarios",GREEN),
                                      (str(s["n_configs"]),"prompt configs",YELLOW),
                                      (str(s["n_metrics"]),"metrics",PURPLE)]):
        _stat(sl, Inches(0.5+i*3.1), Inches(5.2), Inches(2.8), Inches(1.9), num, lbl, _rgb(nc))

    # ─── SLIDE 3 — MODELS ─────────────────────────────────────────────────────
    sl = _blank(prs); _bg(sl); _header(sl,"02","Models Evaluated")
    col_hdrs = ["Model","Provider","Parameters","Training Cutoff","Parse Rate"]
    cxs=[Inches(0.5),Inches(2.6),Inches(5.3),Inches(7.0),Inches(9.2)]
    cws=[Inches(2.05),Inches(2.65),Inches(1.65),Inches(2.15),Inches(1.5)]
    for hx,hw,hl in zip(cxs,cws,col_hdrs):
        _box(sl,hx,Inches(1.22),hw,Inches(0.33),SURFACE2); _tx(sl,hx+Pt(4),Inches(1.25),hw,Inches(0.28),hl,9,bold=True,color=ACCENT2)
    for i,m in enumerate(models):
        ry=Inches(1.58)+i*Inches(0.77)
        mc=_model_color(m); bg_c=RGBColor(mc[0]//14,mc[1]//14,mc[2]//14)
        _box(sl,Inches(0.5),ry,Inches(10.75),Inches(0.72),bg_c,RGBColor(mc[0]//4,mc[1]//4,mc[2]//4),1)
        rate=ps.loc[m,"rate"] if m in ps.index else 0
        rate_str=_pct(rate); rate_col=_parse_bar_color(rate)
        vals=[_m(m,"name",m),_m(m,"provider","—"),_m(m,"params","—"),_m(m,"cutoff","—"),rate_str]
        cols2=[mc,MUTED,WHITE,MUTED,rate_col]; bolds=[True,False,False,False,True]
        for hx,hw,val,vc,bold in zip(cxs,cws,vals,cols2,bolds):
            _tx(sl,hx+Pt(6),ry+Pt(10),hw-Pt(6),Inches(0.55),val,11,bold=bold,color=_rgb(vc))
    _callout(sl,Inches(0.5),Inches(7.0),Inches(12.3),Inches(0.38),
             "Temperature=0.0 for all runs. Open-source: HuggingFace Transformers. GPT-5.4: OpenAI API. Gemini: Google GenAI SDK.")

    # ─── SLIDE 4 — PROMPTS ────────────────────────────────────────────────────
    sl = _blank(prs); _bg(sl); _header(sl,"03","Prompting Strategies")
    cfg_info = [
        ("1. scenario-only",["Only scenario text — no CQs","Model infers scope from narrative","Highest scenario vocabulary coverage"],ACCENT2),
        ("2. cq-only",["Only Competency Questions provided","Highest CQ coverage (0.858 avg)","Also highest hallucination (0.586)"],ACCENT2),
        ("3. scenario-cq  [Balanced]",["Both scenario + CQs","Requires scenario→axiom & CQ→axiom mapping","Good balance of CQ + scenario coverage"],GREEN),
        ("4. scenario-cq-constraints  [Validation]",["Adds 6-item self-validation checklist","Syntax, domain/range, coherence, scope checks","Lower parse rate — extra text confuses parsers"],YELLOW),
        ("5. scenario-cq-reasoning  [Faithfulness]",["6-step reasoning: extract → propose → draft → verify → remove → output","Lowest hallucination overall","Best scenario coverage among CQ-aware configs"],GREEN),
    ]
    positions = [(Inches(0.5),Inches(1.22),Inches(3.9),Inches(2.1)),
                 (Inches(4.55),Inches(1.22),Inches(3.9),Inches(2.1)),
                 (Inches(8.6),Inches(1.22),Inches(4.2),Inches(2.1)),
                 (Inches(0.5),Inches(3.48),Inches(5.9),Inches(2.1)),
                 (Inches(6.55),Inches(3.48),Inches(6.25),Inches(2.1))]
    for (title,bullets,tc),(cx,cy,cw,ch) in zip(cfg_info,positions):
        _card(sl,cx,cy,cw,ch,title,bullets,tc)
    _callout(sl,Inches(0.5),Inches(5.75),Inches(12.3),Inches(0.52),
             "All prompts use @prefix owl/rdf/rdfs/xsd and require owl:Ontology with rdfs:label and rdfs:comment.")

    # ─── SLIDE 5 — EVALUATION ─────────────────────────────────────────────────
    sl = _blank(prs); _bg(sl); _header(sl,"04","Evaluation Methodology — 4 Layers")
    layers = [
        ("Layer 1 — Formal / Syntactic Quality",ACCENT,
         ["parse_success — Valid Turtle syntax?","triple_count — Total RDF triples",
          "ontology_declared — owl:Ontology header?","label_coverage_ratio — all entities with rdfs:label",
          "class_label_ratio [NEW] — owl:Class entities with rdfs:label",
          "property_label_ratio [NEW] — Properties with rdfs:label",
          "comment_coverage_ratio, domain_range_ratio"]),
        ("Layer 2 — Scenario Faithfulness",GREEN,
         ["cq_coverage_ratio — CQs answerable (token overlap)","cq_mean_token_recall — avg recall per CQ",
          "scenario_coverage_ratio — scenario vocab in ontology","hallucination_ratio — unsupported vocab fraction"]),
        ("Layer 3 — Semantic Similarity vs Ground Truth",YELLOW,
         ["class_name_F1 — name match vs ground-truth classes","property_name_F1 — same for properties",
          "combined_f1 — macro avg F1","matched_class_names — classes in both generated & GT",
          "matched_property_names — properties in both; e.g. 'event', 'has effect'"]),
        ("Layer 4 — Feasibility & Unseen Analysis",PURPLE,
         ["Each (model, ODP) labelled plausible_unseen or likely_seen",
          "Compares ODP pub date vs model training cutoff","Confidence: high / medium / low",
          "Future: human eval — Correctness, Completeness, Clarity (1–5)"]),
    ]
    for i,(title,color,bullets) in enumerate(layers):
        col=i%2; row=i//2
        cx=Inches(0.5+col*6.45); cy=Inches(1.22+row*2.85)
        _card(sl,cx,cy,Inches(6.1),Inches(2.7),title,bullets,_rgb(color),_dim3(color))
    _callout(sl,Inches(0.5),Inches(7.0),Inches(12.3),Inches(0.38),
             f"Pipeline → results/summary.csv  ({s['n_runs']} rows × {s['n_metrics']} columns, incl. class_label_ratio & property_label_ratio)")

    # ─── SLIDE 6 — RAW OUTPUTS ────────────────────────────────────────────────
    sl = _blank(prs); _bg(sl); _header(sl,"05","Raw Output Examples — Scenario 2023-133-01")
    _tx(sl,Inches(0.5),Inches(1.2),Inches(12.3),Inches(0.3),
        'Scenario: "Model a causal Bayesian network over events with weighted causal edges as an ODP."',
        11,color=MUTED,italic=True)
    cols3 = 3; col_w = Inches(4.2); col_gap = Inches(0.17)
    for i,m in enumerate(models[:6]):
        col=i%cols3; row=i//cols3
        cx=Inches(0.5)+col*(col_w+col_gap); cy=Inches(1.62)+row*Inches(2.6)
        mc=_model_color(m); bg_c=RGBColor(mc[0]//12,mc[1]//12,mc[2]//12)
        rate=ps.loc[m,"rate"] if m in ps.index else 0
        _box(sl,cx,cy,col_w,Inches(2.45),bg_c,RGBColor(mc[0]//4,mc[1]//4,mc[2]//4),1)
        status="SUCCESS" if rate>0 else "FAIL"; sc=GREEN if rate>0 else RED
        _tx(sl,cx+Pt(6),cy+Pt(6),col_w-Pt(60),Inches(0.28),_m(m,"name",m),12,bold=True,color=_rgb(mc))
        _tx(sl,cx+col_w-Inches(0.95),cy+Pt(8),Inches(0.88),Inches(0.24),status,9,bold=True,color=_rgb(sc),align=PP_ALIGN.CENTER)
        code_bg=_box(sl,cx+Pt(5),cy+Inches(0.38),col_w-Pt(10),Inches(1.65),SURFACE2); code_bg.line.fill.background()
        snippet=(s["sample_outputs"].get(m,"(no output)")[:350]).replace("\n"," ↩ ")
        _tx(sl,cx+Pt(8),cy+Inches(0.41),col_w-Pt(16),Inches(1.6),snippet,7,color=RGBColor(0xc3,0xc8,0xd8))

    # ─── SLIDE 7 — PARSE SUCCESS ──────────────────────────────────────────────
    sl = _blank(prs); _bg(sl); _header(sl,"06","Results — Parse Success Rate")
    _tx(sl,Inches(0.5),Inches(1.2),Inches(12.3),Inches(0.28),
        f"Parse success is the primary gate metric. Total: {s['n_runs']} runs.",12,color=MUTED)
    _tx(sl,Inches(0.5),Inches(1.55),Inches(6.0),Inches(0.3),"By Model",14,bold=True,color=ACCENT2)
    for i,m in enumerate(models):
        rate=ps.loc[m,"rate"] if m in ps.index else 0
        parsed=int(ps.loc[m,"parsed"]) if m in ps.index else 0
        total=int(ps.loc[m,"total"])  if m in ps.index else 0
        _hbar(sl,Inches(0.5),Inches(1.9)+i*Inches(0.62),Inches(5.8),Inches(0.45),
              rate,_m(m,"name",m),f"{_pct(rate)}  ({parsed}/{total})",_parse_bar_color(rate))
    _tx(sl,Inches(7.0),Inches(1.55),Inches(5.8),Inches(0.3),"By Configuration",14,bold=True,color=ACCENT2)
    cfg_ps=(s["df"].groupby("config")["parse_success"].agg(["sum","count","mean"])
              .rename(columns={"mean":"rate"}))
    for i,c in enumerate(configs):
        if c not in cfg_ps.index: continue
        r=cfg_ps.loc[c,"rate"]
        _hbar(sl,Inches(7.0),Inches(1.9)+i*Inches(0.62),Inches(5.8),Inches(0.45),
              r,c,f"{_pct(r)}  ({int(cfg_ps.loc[c,'sum'])}/{int(cfg_ps.loc[c,'count'])})",ACCENT2)
    _callout(sl,Inches(0.5),Inches(6.45),Inches(12.3),Inches(0.52),
             "Gemini 3.1 Pro: 100% parse success across ALL configs — the only model to achieve this. "
             "GPT-5.4: 86% on cq-only but 0% on reasoning/constraints.",GREEN)

    # ─── SLIDE 8 — QUALITY METRICS ────────────────────────────────────────────
    sl = _blank(prs); _bg(sl); _header(sl,"07","Results — Quality Metrics (Successful Parses)")
    _tx(sl,Inches(0.5),Inches(1.2),Inches(12.3),Inches(0.28),
        f"Metrics computed only when parse_success=True.  {s['n_runs']} runs × {s['n_metrics']} metrics.",11,color=MUTED)
    show_q=["cq_coverage_ratio","scenario_coverage_ratio","hallucination_ratio",
            "class_label_ratio","property_label_ratio","combined_f1"]
    show_q=[c for c in show_q if c in avail_q]
    q_short={"cq_coverage_ratio":"CQ Cov.","scenario_coverage_ratio":"Scen. Cov.",
              "hallucination_ratio":"Halluc.","class_label_ratio":"Cls Lbl★",
              "property_label_ratio":"Prop Lbl★","combined_f1":"F1"}
    hdr_lbls=["Model","Parse"]+[q_short.get(c,c) for c in show_q]
    n_cols=len(hdr_lbls)
    col_w_each=Inches(12.3)/n_cols
    hdr_xs=[Inches(0.5)+i*col_w_each for i in range(n_cols)]
    for hx,hl in zip(hdr_xs,hdr_lbls):
        _box(sl,hx,Inches(1.55),col_w_each,Inches(0.3),SURFACE2); _tx(sl,hx+Pt(3),Inches(1.58),col_w_each,Inches(0.26),hl,8,bold=True,color=ACCENT2)
    for i,m in enumerate(models):
        ry=Inches(1.88)+i*Inches(0.52)
        mc=_model_color(m); bg_c=RGBColor(mc[0]//14,mc[1]//14,mc[2]//14)
        _box(sl,Inches(0.5),ry,Inches(12.3),Inches(0.49),bg_c);
        rate=ps.loc[m,"rate"] if m in ps.index else 0
        rc=_parse_bar_color(rate)
        vals=[_m(m,"name",m),_pct(rate)]
        vcols=[_rgb(mc),_rgb(rc)]
        bolds=[True,True]
        if not qual.empty and m in qual.index:
            for c in show_q:
                v=_fmt(qual.loc[m,c]) if c in qual.columns else "—"
                # colour by rank
                if c in qual.columns and not qual[c].isna().all():
                    rk=qual[c].rank(ascending=(c!="hallucination_ratio"),pct=True).get(m,0.5)
                    vc=GREEN if rk>=0.75 else(RED if rk<=0.25 else WHITE)
                else: vc=WHITE
                vals.append(v); vcols.append(_rgb(vc)); bolds.append(False)
        else:
            for _ in show_q: vals.append("—"); vcols.append(MUTED); bolds.append(False)
        for hx,val,vc,bold in zip(hdr_xs,vals,vcols,bolds):
            _tx(sl,hx+Pt(3),ry+Pt(6),col_w_each-Pt(3),Inches(0.4),val,10,bold=bold,color=vc)
    _callout(sl,Inches(0.5),Inches(7.0),Inches(12.3),Inches(0.38),
             "class_label_ratio=1.000 for ALL models. property_label_ratio reveals gap: Llama-2-70B=0.717(worst), Mistral=0.966(best).",YELLOW)

    # ─── SLIDE 9 — HEATMAP ────────────────────────────────────────────────────
    sl = _blank(prs); _bg(sl); _header(sl,"08","Model × Config Parse Success Heatmap")
    _tx(sl,Inches(0.5),Inches(1.2),Inches(12.3),Inches(0.28),"Parse success rate per (model, config) pair. 14 scenarios each.",12,color=MUTED)
    n_cfg=len(configs); cell_w=Inches(10.0)/n_cfg; cell_h=Inches(0.65); start_x=Inches(2.65); start_y=Inches(1.55)
    for j,c in enumerate(configs+["Average"]):
        cx=start_x+j*cell_w
        _box(sl,cx,start_y,cell_w,Inches(0.48),SURFACE2); _tx(sl,cx+Pt(2),start_y+Pt(4),cell_w-Pt(4),Inches(0.4),c if c!="Average" else "Avg",8,bold=True,color=ACCENT2,align=PP_ALIGN.CENTER)
    for i,m in enumerate(models):
        ry=start_y+Inches(0.5)+i*cell_h
        mc=_model_color(m); bg_c=RGBColor(mc[0]//14,mc[1]//14,mc[2]//14)
        _box(sl,Inches(0.5),ry,Inches(2.1),cell_h,bg_c); _tx(sl,Inches(0.52),ry+Pt(10),Inches(2.05),cell_h-Pt(10),_m(m,"name",m),10,bold=True,color=_rgb(mc))
        for j,c in enumerate(configs):
            cx=start_x+j*cell_w
            if (m,c) in cross.index:
                r=cross.loc[(m,c),"rate"]; vc=_parse_bar_color(r)
                cbg=RGBColor(vc[0]//10,vc[1]//10,vc[2]//10)
                _box(sl,cx,ry,cell_w,cell_h,cbg,RGBColor(vc[0]//4,vc[1]//4,vc[2]//4),1)
                _tx(sl,cx,ry+Pt(12),cell_w,cell_h-Pt(12),_pct(r),14,bold=True,color=_rgb(vc),align=PP_ALIGN.CENTER)
            else:
                _box(sl,cx,ry,cell_w,cell_h,SURFACE,SURFACE2,1)
                _tx(sl,cx,ry+Pt(14),cell_w,Inches(0.3),"—",12,color=MUTED,align=PP_ALIGN.CENTER)
        # avg col
        avg_cx=start_x+n_cfg*cell_w; avg_r=ps.loc[m,"rate"] if m in ps.index else 0
        avc=_parse_bar_color(avg_r); abg=RGBColor(avc[0]//10,avc[1]//10,avc[2]//10)
        _box(sl,avg_cx,ry,cell_w,cell_h,abg,RGBColor(avc[0]//4,avc[1]//4,avc[2]//4),2)
        _tx(sl,avg_cx,ry+Pt(12),cell_w,cell_h-Pt(12),_pct(avg_r),14,bold=True,color=_rgb(avc),align=PP_ALIGN.CENTER)
    _callout(sl,Inches(0.5),Inches(6.88),Inches(7.5),Inches(0.52),
             "Gemini 3.1 Pro: 100% across every config — unique. BLOOMZ-7B1: 0% across every config.",GREEN)
    _callout(sl,Inches(8.2),Inches(6.88),Inches(4.6),Inches(0.52),
             "GPT-5.4: 86% on cq-only, 0% on constraints/reasoning. Prompts break Turtle extraction.",YELLOW)

    # ─── SLIDE 10 — MATCHED NAMES ─────────────────────────────────────────────
    sl = _blank(prs); _bg(sl); _header(sl,"09","Matched Class & Property Names vs Ground Truth")
    _tx(sl,Inches(0.5),Inches(1.2),Inches(12.3),Inches(0.3),
        "Names in both generated ontology and ground truth (CamelCase split, case-folded). Low F1 ≠ zero conceptual overlap.",12,color=MUTED)
    _card(sl,Inches(0.5),Inches(1.6),Inches(3.8),Inches(1.75),"Matching Logic",
          ["URI local name extracted: :hasCause → 'has cause'",
           "CamelCase split: HasEffect → 'has effect'","Case-folded & whitespace-normalised",
           "Precision=|matched|/|generated|, Recall=|matched|/|GT|"],ACCENT2,ACCENT)
    # examples
    _tx(sl,Inches(4.5),Inches(1.6),Inches(8.1),Inches(0.28),"Example Matches (GPT-5.4 & Gemini, cq-only)",12,bold=True,color=ACCENT2)
    ex_rows=[("GPT-5.4/cq-only/2023-133-01","event","has effect, has outcome"),
             ("GPT-5.4/cq-only/2023-135-01","name usage","has name usage"),
             ("GPT-5.4/cq-only/2025-149-01","relational database, table, column","—"),
             ("Gemini/scenario-cq/2023-133-01","event","—"),
             ("Gemini/cq-only/2025-151-01","evaluation request","requested action, issued by")]
    ex_xs=[Inches(4.5),Inches(7.1),Inches(10.1)]; ex_ws=[Inches(2.55),Inches(2.95),Inches(2.75)]
    for hx,hw,hl in zip(ex_xs,ex_ws,["Model/Config/Scenario","Matched Classes","Matched Properties"]):
        _box(sl,hx,Inches(1.93),hw,Inches(0.28),SURFACE2); _tx(sl,hx+Pt(3),Inches(1.96),hw,Inches(0.24),hl,8,bold=True,color=ACCENT2)
    for i,(tag,cls,prop) in enumerate(ex_rows):
        ry=Inches(2.24)+i*Inches(0.53)
        rbg=SURFACE if i%2==0 else BG
        for hx,hw in zip(ex_xs,ex_ws): _box(sl,hx,ry,hw,Inches(0.5),rbg)
        for hx,hw,val,vc in zip(ex_xs,ex_ws,[tag,cls,prop],[MUTED,GREEN,ACCENT2]):
            _tx(sl,hx+Pt(3),ry+Pt(4),hw-Pt(6),Inches(0.44),val,9,color=_rgb(vc))
    # prop label chart
    _tx(sl,Inches(0.5),Inches(3.5),Inches(3.8),Inches(0.28),"Property Label Ratio by Model",12,bold=True,color=ACCENT2)
    for i,m in enumerate(models):
        if qual.empty or m not in qual.index or ps.loc[m,"rate"]==0: continue
        plr=qual.loc[m,"property_label_ratio"] if "property_label_ratio" in qual.columns else 0
        _hbar(sl,Inches(0.5),Inches(3.85)+i*Inches(0.5),Inches(3.5),Inches(0.4),
              plr,_m(m,"name",m),_fmt(plr),_parse_bar_color(plr))
    _callout(sl,Inches(0.5),Inches(7.0),Inches(12.3),Inches(0.38),
             "class_label_ratio=1.000 for ALL models on ALL successful parses. The split metric reveals property annotation gaps.",YELLOW)

    # ─── SLIDE 11 — SUMMARY ───────────────────────────────────────────────────
    sl = _blank(prs); _bg(sl); _header(sl,"10","Summary & Best Performing Model")
    wp=s["winners"].get("best_parse",""); wh=s["winners"].get("best_halluc",""); wf=s["winners"].get("best_f1","")
    _box(sl,Inches(0.5),Inches(1.22),Inches(5.5),Inches(5.5),RGBColor(0x07,0x16,0x0e),RGBColor(0x2a,0x6e,0x4a),2)
    _tx(sl,Inches(0.65),Inches(1.32),Inches(5.2),Inches(0.38),"★  Best Overall Reliability",14,bold=True,color=GREEN)
    _tx(sl,Inches(0.65),Inches(1.72),Inches(5.2),Inches(0.42),_m(wp,"name",wp),20,bold=True,color=WHITE)
    parse_r=_pct(ps.loc[wp,"rate"]) if wp in ps.index else "—"
    best_pts=[f"Parse success: {parse_r} (highest across all configs)",
              "100% parse rate on EVERY configuration","Best CQ coverage and Combined F1",
              "Trade-off: highest hallucination (0.623)","Adds unsupported domain knowledge"]
    for k,pt in enumerate(best_pts):
        _tx(sl,Inches(0.65),Inches(2.18)+k*Inches(0.42),Inches(5.2),Inches(0.38),f"• {pt}",10,color=WHITE)
    _box(sl,Inches(0.5),Inches(4.55),Inches(2.62),Inches(2.1),RGBColor(0x10,0x10,0x06),RGBColor(0x80,0x70,0x20),2)
    _tx(sl,Inches(0.62),Inches(4.65),Inches(2.4),Inches(0.32),"Faithfulness",11,bold=True,color=YELLOW)
    _tx(sl,Inches(0.62),Inches(4.98),Inches(2.4),Inches(0.3),_m(wh,"name",wh),15,bold=True,color=WHITE)
    hlv=_fmt(qual.loc[wh,"hallucination_ratio"]) if wh in qual.index and "hallucination_ratio" in qual.columns else "—"
    for k,pt in enumerate([f"Hallucination: {hlv} (lowest)","Best open-source model","Recommended: scenario-cq-reasoning"]):
        _tx(sl,Inches(0.62),Inches(5.32)+k*Inches(0.38),Inches(2.4),Inches(0.35),f"• {pt}",9,color=WHITE)
    _box(sl,Inches(3.27),Inches(4.55),Inches(2.65),Inches(2.1),RGBColor(0x10,0x0d,0x06),RGBColor(0x80,0x50,0x20),2)
    _tx(sl,Inches(3.38),Inches(4.65),Inches(2.4),Inches(0.32),"Best GT Match",11,bold=True,color=ACCENT2)
    _tx(sl,Inches(3.38),Inches(4.98),Inches(2.4),Inches(0.3),_m(wf,"name",wf),15,bold=True,color=WHITE)
    f1v=_fmt(qual.loc[wf,"combined_f1"]) if wf in qual.index and "combined_f1" in qual.columns else "—"
    for k,pt in enumerate([f"Combined F1: {f1v} (best)","Closest to expert ontology vocab","High CQ coverage"]):
        _tx(sl,Inches(3.38),Inches(5.32)+k*Inches(0.38),Inches(2.4),Inches(0.35),f"• {pt}",9,color=WHITE)
    findings=[
        (GREEN, "F1","Model size ≠ performance. Llama-3.1-8B (8B) beats Llama-2-70B (70B) on parse reliability."),
        (GREEN, "F2","Gemini 3.1 Pro: 100% parse, best F1 & CQ coverage — but highest hallucination (0.623)."),
        (YELLOW,"F3","class_label_ratio=1.0 for all models. property_label_ratio exposes hidden gaps (Llama-2: 0.717)."),
        (ACCENT2,"F4","scenario-cq-reasoning: lowest hallucination (0.461 avg), best faithfulness-quality trade-off."),
        (RED,   "F5","BLOOMZ-7B1: 0% parse across all 5 configs. Instruction-tuned BLOOM cannot produce valid Turtle."),
        (MUTED, "F6","Combined F1 universally low (0.06–0.13): LLMs diverge in naming from expert ontologists."),
    ]
    for k,(fc,tag,text) in enumerate(findings):
        fy=Inches(1.22)+k*Inches(0.78)
        _box(sl,Inches(6.3),fy,Inches(0.5),Inches(0.65),_dim(fc),_dim3(fc),1)
        _tx(sl,Inches(6.3),fy+Pt(8),Inches(0.5),Inches(0.5),tag,10,bold=True,color=_rgb(fc),align=PP_ALIGN.CENTER)
        _tx(sl,Inches(6.88),fy+Pt(8),Inches(5.85),Inches(0.58),text,11,color=WHITE)
    _callout(sl,Inches(0.5),Inches(7.0),Inches(12.3),Inches(0.38),
             "Recommendation: Gemini 3.1 Pro + scenario-cq-reasoning for maximum quality. "
             "Llama-3.1-8B + scenario-cq-reasoning for open-source / local deployment.",GREEN)

    prs.save(PPTX_OUT)
    print(f"Saved: {PPTX_OUT}  ({len(prs.slides)} slides)")


# ═══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    p = argparse.ArgumentParser(description="Generate ODPGen presentation files.")
    p.add_argument("--html-only", action="store_true")
    p.add_argument("--pptx-only", action="store_true")
    args = p.parse_args()

    print("Loading data from repo …")
    stats = load_stats()
    print(f"  Models: {stats['n_models_all']}  |  Configs: {stats['n_configs']}  |  "
          f"Scenarios: {stats['n_scenarios']}  |  Runs: {stats['n_runs']}  |  Metrics: {stats['n_metrics']}")

    if not args.pptx_only:
        build_html(stats)
    if not args.html_only:
        build_pptx(stats)


if __name__ == "__main__":
    main()
