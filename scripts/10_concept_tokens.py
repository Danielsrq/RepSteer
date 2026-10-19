"""Per-token multi-concept visualization.

Generate a few paragraphs from NEUTRAL prompts (unsteered), then for every token
compute the signed projection onto each of the 10 unit concept vectors (after
subtracting the grand mean) at the primary layer:
    score[t,k] = (h_t - grand_mean) . concept_hat[k]
Render a self-contained HTML: text wraps naturally, each wrapped visual line gets
its own aligned SVG sparkline (10 colored concept lines), and each word is
highlighted by its argmax (most-present) concept. Alignment is done CLIENT-SIDE
because wrap points depend on browser width.

Needs data/concepts/concept_vectors.pt (from 09).

Usage:
    python scripts/10_concept_tokens.py
"""

import argparse
import html
import json
import os

import numpy as np
import torch

import common

NEUTRAL_PROMPTS = [
    "Write a short paragraph describing a walk through a city at the end of the day.",
    "Write a short paragraph about returning to your childhood home.",
    "Write a short paragraph describing a train arriving at a station.",
]

# 10-color set (validated set is 8; 10 categories exceed the CVD-safe cap, so
# identity is carried primarily by the legend + argmax word highlight + hover +
# click-to-isolate, per the dataviz skill's secondary-encoding rule).
# Order matches the concept index order from 09.
CONCEPT_COLORS = [
    "#eda100",  # happy      - yellow
    "#2a78d6",  # sad        - blue
    "#e34948",  # angry      - red
    "#e87ba4",  # excited    - magenta
    "#1baf7a",  # calm       - aqua
    "#898781",  # nonchalant - gray
    "#4a3aa7",  # skeptical  - violet
    "#eb6834",  # belligerent- orange
    "#008300",  # scheming   - green
    "#8a5a2b",  # paranoid   - brown
    "#17becf",  # humorous   - cyan
    "#b0b0a8",  # neutral    - light gray (matrices only; excluded from token lines)
]

def _blend(f, r, g, b):
    f = max(0.0, min(1.0, f))
    return (f"rgb({round(255*(1-f)+r*f)},{round(255*(1-f)+g*f)},{round(255*(1-f)+b*f)})",
            "#fff" if f > 0.55 else "#0b0b0b")


def matrix_html(mat, concepts, colors, kind):
    """kind='diverging' (cosine, blue<->red around 0) or 'sequential' (counts, blue ramp)."""
    ab = [c[:4] for c in concepts]
    vmax = max(1.0, float(mat.max())) if kind == "sequential" else 1.0
    out = ['<table class="matrix"><tr><th></th>']
    out += [f'<th title="{html.escape(c)}"><span class="sw" style="background:{colors[j]}"></span>{ab[j]}</th>'
            for j, c in enumerate(concepts)]
    out.append("</tr>")
    for i, c in enumerate(concepts):
        row = [f'<th class="rowh" title="{html.escape(c)}"><span class="sw" style="background:{colors[i]}"></span>{html.escape(c)}</th>']
        for j in range(len(concepts)):
            v = float(mat[i, j])
            if kind == "diverging":
                bg, fg = _blend(abs(v), *( (214, 40, 40) if v >= 0 else (33, 102, 172) ))
                txt = f"{v:.2f}"
            else:
                bg, fg = _blend(v / vmax, 13, 54, 107)
                txt = f"{int(round(v))}"
            diag = " diag" if i == j else ""
            row.append(f'<td class="c{diag}" style="background:{bg};color:{fg}">{txt}</td>')
        out.append("<tr>" + "".join(row) + "</tr>")
    out.append("</table>")
    return "".join(out)


def overlap_table_html(names, colors_full, emo_idx, cos2_n, cos2_s, Ks, D):
    """Cumulative Sum cos^2 (R^2) of each concept in the top-K PCs, neutral vs superset."""
    def cell(v):
        bg, fg = _blend(min(v, 1.0), 13, 54, 107)
        return f'<td class="c" style="background:{bg};color:{fg}">{v:.2f}</td>'

    def rowcells(cos2, j):
        return "".join(cell(cos2[j, :k].sum()) for k in Ks)

    out = ['<table class="matrix ov">']
    out.append(f'<tr><th></th><th colspan="{len(Ks)}">neutral PCA (top-K)</th>'
               f'<th colspan="{len(Ks)}">superset PCA (top-K)</th></tr>')
    out.append('<tr><th class="rowh">concept</th>'
               + "".join(f"<th>K={k}</th>" for k in Ks) * 2 + "</tr>")
    for j, cidx in enumerate(emo_idx):
        out.append(f'<tr><th class="rowh"><span class="sw" style="background:{colors_full[cidx]}"></span>'
                   f'{html.escape(names[j])}</th>' + rowcells(cos2_n, j) + rowcells(cos2_s, j) + "</tr>")
    navg = [cos2_n[:, :k].sum(1).mean() for k in Ks]
    savg = [cos2_s[:, :k].sum(1).mean() for k in Ks]
    out.append('<tr class="avg"><th class="rowh">AVERAGE</th>'
               + "".join(cell(v) for v in navg) + "".join(cell(v) for v in savg) + "</tr>")
    rnd = [k / D for k in Ks]
    out.append('<tr><th class="rowh">random</th>'
               + "".join(f'<td class="c">{v:.3f}</td>' for v in rnd) * 2 + "</tr>")
    out.append("</table>")
    return "".join(out)


def denoise_tabs_html(cos_by_k, ks, concepts_full, colors):
    """Tabbed 12x12 cosine matrices, one per k (only one shown at a time)."""
    btns, panels = [], []
    for i, k in enumerate(ks):
        act = " active" if i == 0 else ""
        hid = "" if i == 0 else " hidden"
        btns.append(f'<button class="tab{act}" data-k="{k}">k={k}</button>')
        tbl = matrix_html(cos_by_k[str(k)].numpy(), concepts_full, colors, "diverging")
        panels.append(f'<div class="panel" data-k="{k}"{hid}><div class="wrap">{tbl}</div></div>')
    return '<div class="tabs">' + "".join(btns) + "</div>" + "".join(panels)


HTML_TEMPLATE = r"""<!doctype html><meta charset="utf-8">
<title>Per-token concept content - layer __LAYER__</title>
<style>
 :root{--surface:#fcfcfb;--ink:#0b0b0b;--ink2:#52514e;--muted:#898781;--grid:#e1e0d9;--axis:#c3c2b7}
 body{font:15px/1.7 system-ui,-apple-system,"Segoe UI",sans-serif;max-width:1000px;margin:1.5rem auto;padding:0 1rem;color:var(--ink);background:var(--surface)}
 h1{font-size:1.2rem;margin:.2rem 0}
 .note{color:var(--ink2);font-size:.85rem;margin:.3rem 0 1rem}
 .legend{display:flex;flex-wrap:wrap;gap:.4rem .8rem;margin:.6rem 0 1rem}
 .legend .item{display:flex;align-items:center;gap:.35rem;cursor:pointer;font-size:.85rem;padding:2px 6px;border-radius:6px;user-select:none}
 .legend .item.dim{opacity:.35}
 .legend .sw{width:14px;height:14px;border-radius:3px;box-shadow:0 0 0 1px rgba(11,11,11,.12)}
 .controls{margin:.5rem 0 1rem} select{font-size:1rem;padding:.3rem .5rem}
 .ptext{color:var(--ink2);font-style:italic;margin:.4rem 0}
 .linetext{position:relative;white-space:nowrap}
 .tok{border-radius:2px;padding:0 1px}
 svg.spark{display:block;margin:0 0 .35rem;overflow:visible}
 .flow{white-space:normal;overflow-wrap:anywhere}
 h2{font-size:1.05rem;margin:1.8rem 0 .2rem}
 .wrap{overflow-x:auto}
 .matrix{border-collapse:collapse;font-size:.78rem;margin:.3rem 0 1rem;font-variant-numeric:tabular-nums}
 .matrix th,.matrix td{border:1px solid #fff;padding:4px 7px;text-align:center}
 .matrix th{font-weight:600;color:var(--ink2)}
 .matrix th.rowh{text-align:right;white-space:nowrap}
 .matrix td.c{min-width:34px}
 .matrix td.diag{outline:2px solid rgba(11,11,11,.4);outline-offset:-2px}
 .matrix .sw{display:inline-block;width:10px;height:10px;border-radius:2px;margin-right:4px;vertical-align:middle;box-shadow:0 0 0 1px rgba(11,11,11,.12)}
 .ov td:nth-child(6),.ov th:nth-child(6){border-left:2px solid #bbb}
 .ov tr.avg td,.ov tr.avg th{font-weight:700;border-top:2px solid #bbb}
 .tabs{display:flex;gap:.3rem;margin:.5rem 0}
 .tab{font:inherit;font-size:.85rem;padding:.3rem .8rem;border:1px solid #ccc;background:#f2f2ef;border-radius:6px;cursor:pointer}
 .tab.active{background:#2a78d6;color:#fff;border-color:#2a78d6}
 .panel[hidden]{display:none}
 .toptabs{display:flex;gap:.4rem;margin:.6rem 0 1rem;border-bottom:2px solid #eee}
 .toptab{font:inherit;padding:.4rem 1.1rem;border:none;background:none;cursor:pointer;font-weight:600;color:var(--ink2);border-bottom:3px solid transparent;margin-bottom:-2px}
 .toptab.active{color:var(--ink);border-bottom-color:#2a78d6}
 .toppanel[hidden]{display:none}
 .grp{font-weight:700;margin:1.1rem 0 .3rem;color:var(--ink2)}
 .switch{display:inline-flex;align-items:center;gap:.4rem;margin:.4rem 0 1rem;font-size:.9rem;cursor:pointer}
 .baselinebox{border:2px solid #2a78d6;border-radius:10px;padding:.4rem .7rem .1rem;margin:.4rem 0 1.2rem;background:#f5f9ff}
 .badge{display:inline-block;font-size:.7rem;font-weight:700;letter-spacing:.04em;color:#2a78d6;background:#e3eefb;padding:2px 8px;border-radius:10px;margin:.2rem 0 .3rem}
 .proj{color:var(--ink2);font-weight:600}
 .sepfaint{border-top:1px dashed #ddd;margin:.5rem 0}
 .sepstrong{border-top:3px solid #b8b8b0;margin:1.3rem 0 .6rem}
 .pills{display:flex;flex-wrap:wrap;gap:.35rem;margin:.3rem 0 1rem}
 .pill{font:inherit;font-size:.82rem;display:inline-flex;align-items:center;gap:.3rem;padding:.25rem .6rem;border:1px solid #ddd;background:#f7f7f8;border-radius:999px;cursor:pointer}
 .pill.active{border-color:#0b0b0b;background:#fff;font-weight:600}
 .pill .sw{width:10px;height:10px;border-radius:3px;box-shadow:0 0 0 1px rgba(11,11,11,.12)}
 .steerhdr{display:flex;align-items:center;gap:.8rem;flex-wrap:wrap}
 .pbars{flex:0 0 auto}
</style>
<h1>Per-token concept content &mdash; layer __LAYER__</h1>
<div class="legend" id="legend"></div>
<div class="controls"><label>Prompt: <select id="sel"></select></label><div class="ptext" id="ptext"></div></div>
<div class="toptabs">
 <button class="toptab active" data-tab="analysis">Analysis</button>
 <button class="toptab" data-tab="steer">Steering</button>
</div>
<div class="toppanel" data-tab="analysis">
<div class="note">Each token's word is highlighted by its <b>strongest</b> concept (argmax). Beneath each wrapped line, colored lines show the signed projection
 <code>(h<sub>t</sub> &minus; grand_mean) &middot; concept&#770;<sub>k</sub></code> per token. Click a legend entry to isolate one concept. Hover a word for its top concept. <b>Note:</b> many categorical colors exceed CVD-safe limits &mdash; use the legend/labels, not hue alone.</div>
<div id="out"></div>
<h2>Concept-vector cosine similarity &mdash; layer __LAYER__</h2>
<div class="note">Diverging: <b style="color:#d62828">red</b> = aligned directions, <b style="color:#2166ac">blue</b> = anti-aligned, white &asymp; orthogonal. Cosines of the grand-mean-centered concept vectors.</div>
<div class="wrap">__COSINE__</div>
<h2>Nearest-concept confusion (in-sample) &mdash; rows = true, cols = predicted</h2>
<div class="note">Sequential blue: darker = more paragraphs. Outlined diagonal = correctly classified. Off-diagonal shows which concepts get mixed up.</div>
<div class="wrap">__CONFUSION__</div>
<h2>Concept vs PCA subspace &mdash; how much of each concept the top-K PCs reconstruct</h2>
<div class="note">Cumulative &Sigma;cos&sup2; (the R&sup2; of rebuilding a concept direction from the top-K principal components) at layer __PCALAYER__.
 1.00 = concept lies fully inside that subspace; &asymp;<i>random</i> = orthogonal to it. Emotion explains <b>__ETA__%</b> of the superset's activation variance (inflated by design: balanced tone contrast over shared topics).
 <b>Superset PCs are the concept subspace</b> (&asymp;0.98 by K=20) &mdash; PCA of the emotional cloud recovers the concepts, so it is a poor nuisance basis.
 <b>Neutral PCs are largely orthogonal</b> (&asymp;0.33 by K=20) &mdash; the untoned cloud varies by style/topic, so it is the basis to denoise against (removing them costs only ~1/3 of the concept).</div>
<div class="wrap">__OVERLAP__</div>
__DENOISE__
</div>
<div class="toppanel" data-tab="steer" hidden>
<div class="note">Steering the residual stream at layer __LAYER__ with each <b>emotional concept vector</b> at &plusmn;0.1/0.2/0.4/0.6&middot;&#8214;h&#8214; (norm-relative, every token position). Pick the steering concept with the pills. Words are colored by their strongest concept (argmax); each header's mini bar chart shows the mean projection onto <b>all 11 concepts</b> (the steered one outlined) &mdash; watch related concepts move too. Toggle raw vs denoised (top-2 neutral PCs removed) vectors.</div>
<div class="pills" id="steerpills"></div>
<label class="switch"><input type="checkbox" id="dentoggle"> Use denoised steering vectors (top-2 neutral PCs removed)</label>
<div id="steerout"></div>
</div>
<script>
const DATA=__DATA__, COLORS=__COLORS__, CONCEPTS=__CONCEPTS__, SCALE=__SCALE__, STEER=__STEER__, STEER_MAGS=__STEER_MAGS__, PROJ_SCALE=__PROJ_SCALE__;
let active=null;   // isolated concept index, or null
const H=54, PAD=8;

function mixWhite(hex,f){ // f = fraction of color (rest white)
  const n=parseInt(hex.slice(1),16), r=n>>16, g=(n>>8)&255, b=n&255;
  const m=v=>Math.round(255*(1-f)+v*f);
  return `rgb(${m(r)},${m(g)},${m(b)})`;
}
function tooltip(pd,i){
  const pairs=CONCEPTS.map((c,k)=>[c,pd.scores[i][k]]).sort((a,b)=>b[1]-a[1]);
  return pairs.map(([c,v])=>`${c}: ${v>=0?'+':''}${v.toFixed(2)}`).join('\n');
}
function yMap(v){ const c=Math.max(-1,Math.min(1,v/SCALE)); return H/2 - c*(H/2-PAD); }

function drawSvg(lineIdx, spans, pd, width){
  const svgNS="http://www.w3.org/2000/svg";
  const svg=document.createElementNS(svgNS,"svg");
  svg.setAttribute("class","spark"); svg.setAttribute("width",width); svg.setAttribute("height",H);
  // baseline (grand mean = 0)
  const base=document.createElementNS(svgNS,"line");
  base.setAttribute("x1",0);base.setAttribute("x2",width);base.setAttribute("y1",H/2);base.setAttribute("y2",H/2);
  base.setAttribute("stroke","var(--grid)");base.setAttribute("stroke-width","1");
  svg.appendChild(base);
  const xs=lineIdx.map(i=>spans[i].offsetLeft+spans[i].offsetWidth/2);
  for(let k=0;k<CONCEPTS.length;k++){
    const pts=lineIdx.map((i,j)=>`${xs[j].toFixed(1)},${yMap(pd.scores[i][k]).toFixed(1)}`).join(" ");
    const pl=document.createElementNS(svgNS,"polyline");
    pl.setAttribute("points",pts);pl.setAttribute("fill","none");
    pl.setAttribute("stroke",COLORS[k]);
    const on = active===null || active===k;
    pl.setAttribute("stroke-width", active===k?"2.5":(active===null?"1.6":"1"));
    pl.setAttribute("opacity", on?"0.95":"0.10");
    svg.appendChild(pl);
  }
  return svg;
}

function rebuild(pd, root){
  root.innerHTML="";
  const flow=document.createElement("div"); flow.className="flow"; root.appendChild(flow);
  const spans=pd.tokens.map((t,i)=>{
    const s=document.createElement("span"); s.className="tok"; s.textContent=t;
    s.style.background=mixWhite(COLORS[pd.argmax[i]],0.32); s.title=tooltip(pd,i);
    return s;
  });
  spans.forEach(s=>flow.appendChild(s));
  // group tokens into visual lines by offsetTop
  const lines=[]; let cur=null, top=null;
  spans.forEach((s,i)=>{ if(s.offsetTop!==top){cur=[];lines.push(cur);top=s.offsetTop;} cur.push(i); });
  root.removeChild(flow);
  const W=root.clientWidth;
  for(const line of lines){
    const lt=document.createElement("div"); lt.className="linetext";
    line.forEach(i=>lt.appendChild(spans[i]));
    root.appendChild(lt);                       // layout now valid: offsetLeft relative to lt
    root.appendChild(drawSvg(line, spans, pd, W));
  }
}

let curPrompt=0, curSteerConcept=null;
function renderCurrent(){ rebuild(DATA.prompts[curPrompt], document.getElementById("out")); }

// --- steering tab ---
function projBars(proj, steeredIdx){
  const svgNS="http://www.w3.org/2000/svg", n=proj.length, bw=11, W=n*bw, Hh=30, mid=Hh/2, pad=3;
  const svg=document.createElementNS(svgNS,"svg"); svg.setAttribute("width",W); svg.setAttribute("height",Hh); svg.setAttribute("class","pbars");
  const ml=document.createElementNS(svgNS,"line"); ml.setAttribute("x1",0);ml.setAttribute("x2",W);ml.setAttribute("y1",mid);ml.setAttribute("y2",mid);ml.setAttribute("stroke","#ccc");ml.setAttribute("stroke-width","1"); svg.appendChild(ml);
  for(let k=0;k<n;k++){
    const v=Math.max(-1,Math.min(1,proj[k]/PROJ_SCALE)), bh=Math.abs(v)*(mid-pad), y=v>=0?mid-bh:mid;
    const r=document.createElementNS(svgNS,"rect");
    r.setAttribute("x",k*bw+0.5);r.setAttribute("width",bw-1);r.setAttribute("y",y);r.setAttribute("height",Math.max(bh,0.6));
    r.setAttribute("fill",COLORS[k]);
    if(k===steeredIdx){r.setAttribute("stroke","#0b0b0b");r.setAttribute("stroke-width","1.3");}
    const ttl=document.createElementNS(svgNS,"title"); ttl.textContent=CONCEPTS[k]+": "+(proj[k]>=0?"+":"")+proj[k].toFixed(2); r.appendChild(ttl);
    svg.appendChild(r);
  }
  return svg;
}
function steerCard(label, res, steeredIdx){
  const card=document.createElement("div"); card.className="cell";
  const hdr=document.createElement("div"); hdr.className="hdr steerhdr";
  const lab=document.createElement("span"); lab.textContent=label; hdr.appendChild(lab);
  hdr.appendChild(projBars(res.proj, steeredIdx));
  const txt=document.createElement("div"); txt.className="txt";
  res.tokens.forEach((t,i)=>{const s=document.createElement("span"); s.className="tok";
    s.style.background=mixWhite(COLORS[res.argmax[i]],0.32); s.title=CONCEPTS[res.argmax[i]];
    s.textContent=t; txt.appendChild(s);});
  card.appendChild(hdr); card.appendChild(txt); return card;
}
function grp(label){const d=document.createElement("div"); d.className="grp"; d.textContent=label; return d;}
function sep(cls){const d=document.createElement("div"); d.className=cls; return d;}
function renderSteer(){
  if(curSteerConcept===null) curSteerConcept=CONCEPTS[0];
  const p=STEER[curPrompt]; const set=document.getElementById("dentoggle").checked ? p.denoised : p.raw;
  const cname=curSteerConcept, ci=CONCEPTS.indexOf(cname), cset=set[cname];
  const out=document.getElementById("steerout"); out.innerHTML="";
  const box=document.createElement("div"); box.className="baselinebox";
  const badge=document.createElement("div"); badge.className="badge"; badge.textContent="BASELINE · no steering";
  box.appendChild(badge); box.appendChild(steerCard("baseline", p.baseline, -1));
  out.appendChild(box);
  STEER_MAGS.forEach((m,mi)=>{
    const ms=m.toFixed(1);
    out.appendChild(grp("α = ±"+ms+" · ‖h‖   ·   "+cname));
    out.appendChild(steerCard("+"+ms+" · "+cname, cset["+"+ms], ci));
    out.appendChild(steerCard("−"+ms+" · "+cname, cset["-"+ms], ci));
    if(mi<STEER_MAGS.length-1) out.appendChild(sep("sepstrong"));
  });
}

// legend
const legend=document.getElementById("legend");
CONCEPTS.forEach((c,k)=>{
  const it=document.createElement("div"); it.className="item"; it.dataset.k=k;
  it.innerHTML=`<span class="sw" style="background:${COLORS[k]}"></span>${c}`;
  it.onclick=()=>{ active=(active===k?null:k);
    [...legend.children].forEach((el,j)=>el.classList.toggle("dim", active!==null && j!==active));
    renderCurrent(); };
  legend.appendChild(it);
});
// prompt selector
const sel=document.getElementById("sel");
DATA.prompts.forEach((p,i)=>{const o=document.createElement("option");o.value=i;o.textContent="Prompt "+(i+1);sel.appendChild(o);});
sel.onchange=()=>{ curPrompt=+sel.value; document.getElementById("ptext").textContent='"'+DATA.prompts[curPrompt].prompt+'"'; renderCurrent(); renderSteer(); };
document.getElementById("ptext").textContent='"'+DATA.prompts[0].prompt+'"';
// steering concept pills
const pillbox=document.getElementById("steerpills");
CONCEPTS.forEach((c,k)=>{
  const b=document.createElement("button"); b.className="pill"+(k===0?" active":""); b.dataset.c=c;
  b.innerHTML='<span class="sw" style="background:'+COLORS[k]+'"></span>'+c;
  b.onclick=()=>{ curSteerConcept=c; document.querySelectorAll(".pill").forEach(x=>x.classList.toggle("active",x.dataset.c===c)); renderSteer(); };
  pillbox.appendChild(b);
});
curSteerConcept=CONCEPTS[0];
renderCurrent(); renderSteer();
// top-level tabs
document.querySelectorAll(".toptab").forEach(b=>b.addEventListener("click",()=>{
  document.querySelectorAll(".toptab").forEach(x=>x.classList.remove("active"));
  b.classList.add("active");
  const tb=b.dataset.tab;
  document.querySelectorAll(".toppanel").forEach(p=>{p.hidden=p.dataset.tab!==tb;});
  if(tb==="analysis") renderCurrent();   // re-measure sparklines now that #out is visible
}));
document.getElementById("dentoggle").addEventListener("change",renderSteer);
// re-align on resize (debounced)
let t; window.addEventListener("resize",()=>{clearTimeout(t);t=setTimeout(renderCurrent,150);});
// denoise cosine tabs
document.querySelectorAll('.tab').forEach(b=>b.addEventListener('click',()=>{
  document.querySelectorAll('.tab').forEach(x=>x.classList.remove('active'));
  b.classList.add('active');
  const k=b.dataset.k;
  document.querySelectorAll('.panel').forEach(p=>{p.hidden = p.dataset.k!==k;});
}));
</script>
"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", choices=list(common.PRESETS), default="qwen35-4b")
    ap.add_argument("--model", default=None)
    ap.add_argument("--kind", choices=["text", "multimodal"], default=None)
    ap.add_argument("--vectors", default="data/concepts/concept_vectors.pt")
    ap.add_argument("--layer", type=int, default=None, help="default: primary_layer from vectors")
    ap.add_argument("--prompts-file", default=None, help="JSON list of prompt strings")
    ap.add_argument("--max-new-tokens", type=int, default=120)
    ap.add_argument("--denoise", default="data/concepts/denoise.pt",
                    help="denoised vectors/cosine tables from 12_denoise.py (tabs + denoised steering)")
    ap.add_argument("--steer-denoise-k", type=int, default=2,
                    help="which denoised-by-k vectors to use for the denoised steering condition")
    ap.add_argument("--steer-mags", default="0.1,0.2,0.4,0.6",
                    help="comma list of |alpha| magnitudes to sweep (both signs, all concepts)")
    ap.add_argument("--out", default="data/concepts/concept_tokens.html")
    args = ap.parse_args()

    saved = torch.load(args.vectors, weights_only=False)
    L = args.layer if args.layer is not None else int(saved["primary_layer"])
    concepts_full = saved["concepts"]                            # all (incl neutral)
    neutral_idx = saved.get("neutral_idx")
    if neutral_idx is None and "neutral" in concepts_full:
        neutral_idx = concepts_full.index("neutral")
    emo_idx = saved.get("emotional_idx") or [k for k in range(len(concepts_full)) if k != neutral_idx]
    concepts = [concepts_full[k] for k in emo_idx]               # emotional only -> token-viz lines
    emo_colors = [CONCEPT_COLORS[k] for k in emo_idx]
    chat = saved["concept_hat"][L].numpy().astype(np.float32)[emo_idx]   # [n_emo, hidden]
    grand = saved["grand_mean"][L].numpy().astype(np.float32)     # [hidden]

    prompts = NEUTRAL_PROMPTS
    if args.prompts_file:
        prompts = json.load(open(args.prompts_file))

    model_id, kind = common.resolve(args.preset, args.model, args.kind)
    print(f"Loading {model_id} ... | layer {L} | {len(concepts)} concepts")
    handler, model, tok, is_mm = common.load_model(model_id, kind)
    _, layers = common.get_decoder_layers(model)
    target = layers[L]
    dev = next(model.parameters()).device

    grab = {}
    dt = next(model.parameters()).dtype

    def read_hook(_m, _i, output):
        grab["h"] = common.layer_output_hidden(output)[0].float().cpu()

    def build_enc(prompt):
        messages = [{"role": "user", "content": prompt}]
        kw = dict(add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt")
        try:
            enc = handler.apply_chat_template(messages, enable_thinking=False, **kw)
        except TypeError:
            enc = handler.apply_chat_template(messages, **kw)
        return {k: v.to(dev) for k, v in enc.items()}

    def score_text(text):
        """Unsteered forward pass -> per-token projection onto the emotional concept vecs."""
        enc2 = tok(text, return_tensors="pt").to(dev)
        rh = target.register_forward_hook(read_hook)
        try:
            with torch.inference_mode():
                model(input_ids=enc2["input_ids"], attention_mask=enc2.get("attention_mask"))
        finally:
            rh.remove()
        H = grab["h"].numpy()
        scores = (H - grand) @ chat.T                            # [seq, n_emo]
        toks = [tok.decode([t]) for t in enc2["input_ids"][0].tolist()]
        return toks, scores

    h_col = emo_idx.index(concepts_full.index("happy"))          # happy/sad columns for the lean summary
    s_col = emo_idx.index(concepts_full.index("sad"))

    def lean_of(scores):
        return float((scores[:, h_col] - scores[:, s_col]).mean())

    # --- baseline (unsteered) generations for the token heatmap ---
    data_prompts, all_abs = [], []
    for prompt in prompts:
        enc = build_enc(prompt); n_prompt = enc["input_ids"].shape[1]
        with torch.inference_mode():
            out = model.generate(**enc, max_new_tokens=args.max_new_tokens,
                                 do_sample=False, pad_token_id=tok.eos_token_id)
        text = tok.decode(out[0, n_prompt:], skip_special_tokens=True).strip()
        toks, scores = score_text(text)
        all_abs.extend(np.abs(scores).ravel().tolist())
        data_prompts.append({"prompt": prompt, "tokens": toks,
                             "scores": [[round(float(v), 3) for v in row] for row in scores],
                             "argmax": scores.argmax(1).tolist()})
        print(f"  baseline: {prompt[:50]}... | {len(toks)} tokens")

    # --- steering: every emotional concept vector, raw and denoised (top-k neutral PCs removed) ---
    Vraw = saved["concept_vectors"][L].numpy().astype(np.float32)

    def unit(v):
        return v / (np.linalg.norm(v) + 1e-12)

    Vden = None
    if os.path.exists(args.denoise):
        Vden = torch.load(args.denoise, weights_only=False)["denoised_by_k"][str(args.steer_denoise_k)].numpy().astype(np.float32)
    dirs = {"raw": {}, "denoised": {}}
    for name in concepts:                                    # 11 emotional concepts
        ci = concepts_full.index(name)
        dirs["raw"][name] = unit(Vraw[ci])
        dirs["denoised"][name] = unit(Vden[ci]) if Vden is not None else unit(Vraw[ci])

    def make_inject(u_t, alpha):
        def hook(_m, _i, output):
            h = output[0] if isinstance(output, tuple) else output
            h = h + alpha * h.norm(dim=-1, keepdim=True) * u_t
            return (h, *output[1:]) if isinstance(output, tuple) else h
        return hook

    def gen_steered(enc, u_t, alpha):
        hd = target.register_forward_hook(make_inject(u_t, alpha))
        try:
            with torch.inference_mode():
                out = model.generate(**enc, max_new_tokens=args.max_new_tokens,
                                     do_sample=False, pad_token_id=tok.eos_token_id)
        finally:
            hd.remove()
        return tok.decode(out[0, enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()

    all_proj_abs = []

    def proj_of(scores):
        proj = [round(float(scores[:, k].mean()), 3) for k in range(scores.shape[1])]  # 11 mean projections
        all_proj_abs.extend(abs(v) for v in proj)
        return proj

    def card_from_text(text):
        toks, scores = score_text(text)
        return {"tokens": toks, "argmax": scores.argmax(1).tolist(), "proj": proj_of(scores)}

    mags = [float(x) for x in args.steer_mags.split(",")]
    steer_data = []
    for i, prompt in enumerate(prompts):
        enc = build_enc(prompt)
        bl = data_prompts[i]
        entry = {"baseline": {"tokens": bl["tokens"], "argmax": bl["argmax"],
                              "proj": proj_of(np.array(bl["scores"], dtype=np.float32))},
                 "raw": {}, "denoised": {}}
        for basis in ("raw", "denoised"):
            for name in concepts:
                entry[basis][name] = {}
                u_t = torch.tensor(dirs[basis][name]).to(dev, dt)
                for m in mags:
                    for sign, a in (("+", m), ("-", -m)):
                        entry[basis][name][f"{sign}{m:.1f}"] = card_from_text(gen_steered(enc, u_t, a))
                print(f"  steer p{i+1} | {basis} {name} done")
        steer_data.append(entry)
    proj_scale = float(np.percentile(all_proj_abs, 98)) if all_proj_abs else 1.0

    scale = float(np.percentile(all_abs, 98)) if all_abs else 1.0
    sep = saved["separability"]
    cosine_tbl = matrix_html(sep["cosine_matrix"].numpy(), concepts_full, CONCEPT_COLORS, "diverging")
    confusion_tbl = matrix_html(sep["confusion"].numpy(), concepts_full, CONCEPT_COLORS, "sequential")

    # --- concept vs PCA subspace overlap (computed at the PCA layer) ---
    PLp = int(saved["pca_neutral"]["layer"])
    ehat = saved["concept_hat"][PLp].numpy().astype(np.float64)[emo_idx]
    ehat /= np.linalg.norm(ehat, axis=1, keepdims=True) + 1e-12
    Pn = saved["pca_neutral"]["components"].numpy().astype(np.float64)
    Ps = saved["pca_superset"]["components"].numpy().astype(np.float64)
    cos2_n = (ehat @ Pn.T) ** 2
    cos2_s = (ehat @ Ps.T) ** 2
    Dh = ehat.shape[1]
    overlap_tbl = overlap_table_html(concepts, CONCEPT_COLORS, emo_idx, cos2_n, cos2_s, [1, 5, 10, 20], Dh)
    # eta^2: fraction of emotional-superset variance explained by concept identity
    Xp = saved["features"][:, PLp].numpy().astype(np.float64); labs = saved["labels"].numpy()
    gmp = Xp[np.isin(labs, emo_idx)].mean(0)
    tot = bet = 0.0
    for k in emo_idx:
        Xk = Xp[labs == k]; tot += ((Xk - gmp) ** 2).sum(); bet += len(Xk) * ((Xk.mean(0) - gmp) ** 2).sum()
    eta2 = bet / tot

    # --- denoised cosine tabs (from 12_denoise.py), if available ---
    denoise_section = ""
    if os.path.exists(args.denoise):
        d = torch.load(args.denoise, weights_only=False)
        ks_d = [k for k in d["ks"] if k != 0]        # show 1,2,5,10 (k=0 == original above)
        tabs = denoise_tabs_html(d["cosine_by_k"], ks_d, concepts_full, CONCEPT_COLORS)
        denoise_section = (
            "<h2>Denoised concept cosine similarity &mdash; top-k neutral PCs removed</h2>"
            '<div class="note">Concept vectors with the top-k <b>neutral</b> PCs (style/topic axes) projected out: '
            "v<sub>d</sub> = v &minus; &Sigma;<sub>j&lt;k</sub>(v&middot;pc<sub>j</sub>)pc<sub>j</sub>. "
            "Compare to the original cosine matrix above. Tabs switch k.</div>" + tabs)

    doc = (HTML_TEMPLATE
           .replace("__DATA__", json.dumps({"prompts": data_prompts}))
           .replace("__COLORS__", json.dumps(emo_colors))
           .replace("__CONCEPTS__", json.dumps(concepts))
           .replace("__SCALE__", f"{scale:.4f}")
           .replace("__COSINE__", cosine_tbl)
           .replace("__CONFUSION__", confusion_tbl)
           .replace("__OVERLAP__", overlap_tbl)
           .replace("__DENOISE__", denoise_section)
           .replace("__STEER__", json.dumps(steer_data))
           .replace("__STEER_MAGS__", json.dumps(mags))
           .replace("__PROJ_SCALE__", f"{proj_scale:.4f}")
           .replace("__ETA__", f"{eta2*100:.0f}")
           .replace("__PCALAYER__", str(PLp))
           .replace("__LAYER__", str(L)))
    with open(args.out, "w") as f:
        f.write(doc)
    print(f"\nWrote {len(prompts)} prompts -> {args.out}  (color scale={scale:.2f})")


if __name__ == "__main__":
    main()
