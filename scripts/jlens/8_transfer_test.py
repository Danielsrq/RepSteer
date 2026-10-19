"""Transfer test: project a REAL writeup, token by token, onto the security DiM concepts
AND decompose each token into J-space. Fuses the DiM highlight (concept_tokens style) with
the J-space hover (jspace_tokens style), on genuine text fed with context.

Resumable: each 1000-token page is processed once and cached as JSON under
data/security/transfer_cache/. Re-running with more --pages skips cached pages and only
computes new ones, then re-renders the paginated HTML from ALL cached pages.

Per page: feed [start-prefix : start+page_size] (prefix gives context + avoids sink tokens),
hook layer 17, page-mean-center, then for each of the 1000 display tokens:
  - DiM: signed projection onto the 11 security concept directions (argmax over 10 vuln
    concepts drives the tint),
  - J-space: NN-OMP decomposition into vocab concepts.

Usage:  .venv/bin/python scripts/jlens/8_transfer_test.py --pages 0
        .venv/bin/python scripts/jlens/8_transfer_test.py --pages 1-4   # resume, adds pages
"""

import argparse
import glob
import html
import json
import os
import sys
import time

import numpy as np
import torch

import jlens

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import common  # noqa: E402


def nnls(A, x, iters=250):
    if A.shape[1] == 0:
        return np.zeros(0)
    AtA, Atx = A.T @ A, A.T @ x
    lr = 1.0 / max(float(np.linalg.eigvalsh(AtA)[-1]), 1e-9)
    c = np.zeros(A.shape[1])
    for _ in range(iters):
        c = np.maximum(c - lr * (AtA @ c - Atx), 0.0)
    return c


def nn_omp(x, D, k):
    x2 = float(x @ x) + 1e-12
    r = x.copy(); S, co = [], np.zeros(0)
    for _ in range(k):
        corr = D @ r
        if S:
            corr[np.array(S)] = -1e30
        j = int(np.argmax(corr))
        if corr[j] <= 0:
            break
        S.append(j); co = nnls(D[S].T, x); r = x - D[S].T @ co
    return S, co, 1.0 - float(r @ r) / x2


def parse_pages(spec, npages):
    out = set()
    for part in spec.split(","):
        if part == "all":
            out |= set(range(npages))
        elif "-" in part:
            a, b = part.split("-"); out |= set(range(int(a), int(b) + 1))
        else:
            out.add(int(part))
    return sorted(p for p in out if 0 <= p < npages)


def process_page(m, model, layers, dim_L, js_L, Jjs, g, ch, ids, start, page_size, prefix, top_m, k, show, tok):
    """DiM highlight read at dim_L (most-separable layer); J-space decomposed at js_L
    (workspace-peak layer). Both come from one forward per page (hook both layers)."""
    dev = m.input_device
    ctx0 = max(0, start - prefix)
    seg = ids[ctx0:start + page_size]
    grab, hooks = {}, []

    def mk(key):
        def hook(_mod, _i, o):
            grab[key] = common.layer_output_hidden(o)[0].detach().float().cpu()
        return hook
    hooks.append(layers[dim_L].register_forward_hook(mk("dim")))
    if js_L != dim_L:
        hooks.append(layers[js_L].register_forward_hook(mk("js")))
    try:
        with torch.inference_mode():
            m.forward(torch.tensor([seg], device=dev))
    finally:
        for h in hooks:
            h.remove()
    off = start - ctx0
    Hdim = grab["dim"][off:]                          # DiM-layer residuals [n,d]
    Hjs = (grab["js"] if js_L != dim_L else grab["dim"])[off:]   # J-space-layer residuals
    disp_ids = ids[start:start + page_size]
    n = len(disp_ids)
    mu_dim, mu_js = Hdim.mean(0), Hjs.mean(0)
    dim = ((Hdim - mu_dim) @ ch.T).numpy()           # DiM projections at dim_L

    Hd = Hjs.to(dev).float()                          # lens cone from the J-space layer
    tops = []                                         # chunk the [n, vocab] readout to bound VRAM
    for s in range(0, len(Hd), 256):
        tops.append(m.unembed(Hd[s:s + 256] @ Jjs.T).float().topk(top_m, dim=1).indices)
    top = torch.cat(tops, 0)
    toks = []
    for i in range(n):
        ti = top[i]
        w = m._lm_head.weight[ti].float()
        V = ((w * g) @ Jjs)
        Vhat = (V / (V.norm(dim=1, keepdim=True) + 1e-12)).cpu().numpy().astype(np.float64)
        S, co, recon = nn_omp((Hjs[i] - mu_js).numpy().astype(np.float64), Vhat, k)
        js = sorted([(tok.decode([int(ti[S[j]])]), float(co[j])) for j in range(len(S)) if co[j] > 1e-3],
                    key=lambda z: -z[1])[:show]
        proj = dim[i]
        tc = int(np.argmax(proj[:10]))               # argmax over the 10 vuln concepts
        toks.append({"t": tok.decode([disp_ids[i]]),
                     "d": [round(float(x), 2) for x in proj],
                     "tc": tc, "tv": round(float(proj[tc]), 2),
                     "j": [[wd, round(c, 2)] for wd, c in js], "r": round(recon, 3)})
    return {"page": start // page_size, "start": start, "n": n, "tokens": toks}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default="qwen35-4b")
    ap.add_argument("--lens", default="data/jlens/lens_band_4b.pt")
    ap.add_argument("--vectors", default="data/security/security_vectors.pt")
    ap.add_argument("--article", default="data/jlens/cve/nftables.txt")
    ap.add_argument("--dim-layer", type=int, default=17, help="layer for DiM concept highlight (most separable)")
    ap.add_argument("--jspace-layer", type=int, default=25, help="layer for J-space decomposition (workspace peak)")
    ap.add_argument("--page-size", type=int, default=1000)
    ap.add_argument("--prefix", type=int, default=128, help="context tokens fed before each page")
    ap.add_argument("--pages", default="0", help="e.g. 0 | 1-4 | 0,2,5 | all")
    ap.add_argument("--top-m", type=int, default=256)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--show", type=int, default=8)
    ap.add_argument("--cache", default="data/security/transfer_cache")
    ap.add_argument("--tint-thresh", type=float, default=0.8,
                    help="absolute margin (top_vuln - max(benign,neutral)) needed to tint a token")
    ap.add_argument("--out", default="data/security/transfer_nftables.html")
    args = ap.parse_args()

    model_id, kind = common.resolve(args.preset)
    dim_L, js_L = args.dim_layer, args.jspace_layer
    art = os.path.splitext(os.path.basename(args.article))[0]
    tag = f"{art}_d{dim_L}j{js_L}_ps{args.page_size}pf{args.prefix}"   # cache key encodes layers + window
    os.makedirs(args.cache, exist_ok=True)
    blob = torch.load(args.vectors, weights_only=False)
    concepts = blob["concepts"]

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_id)
    ids = tok(open(args.article, encoding="utf-8").read(), add_special_tokens=False).input_ids
    npages = (len(ids) + args.page_size - 1) // args.page_size
    want = parse_pages(args.pages, npages)
    todo = [p for p in want if not os.path.exists(os.path.join(args.cache, f"{tag}_p{p:03d}.json"))]
    print(f"Article {art}: {len(ids)} tokens = {npages} pages. DiM@L{dim_L} J-space@L{js_L}. "
          f"Requested {want}; to process {todo}.")

    if todo:  # only load the full model when there's something to compute
        print(f"Loading {model_id} + lens ...")
        _, model, mtok, _ = common.load_model(model_id, kind)
        m = jlens.from_hf(model, mtok)
        _, layers = common.get_decoder_layers(model)
        lens = jlens.JacobianLens.load(args.lens)
        Jjs = lens.jacobians[js_L].to(m.input_device).float()
        g = m._final_norm.weight.detach().to(m.input_device).float()
        ch = blob["concept_hat"][dim_L].float()
        for p in todo:
            cf = os.path.join(args.cache, f"{tag}_p{p:03d}.json")
            t0 = time.perf_counter()
            res = process_page(m, model, layers, dim_L, js_L, Jjs, g, ch, ids, p * args.page_size,
                               args.page_size, args.prefix, args.top_m, args.k, args.show, mtok)
            json.dump(res, open(cf, "w"))
            print(f"  page {p}: {res['n']} tokens in {time.perf_counter()-t0:.1f}s -> {cf}")

    # --- render from ALL cached pages for this config ---
    pages = []
    for cf in sorted(glob.glob(os.path.join(args.cache, f"{tag}_p*.json"))):
        pages.append(json.load(open(cf)))
    if not pages:
        print("no pages cached yet"); return
    payload = json.dumps({"concepts": concepts, "pages": pages}).replace("<", "\\u003c")
    doc = (_TEMPLATE.replace("__DATA__", payload).replace("__ART__", html.escape(art))
           .replace("__L__", f"DiM L{dim_L} · J-space L{js_L}").replace("__THR__", str(args.tint_thresh)))
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    open(args.out, "w").write(doc)
    print(f"Rendered {len(pages)} page(s) -> {args.out}")


_TEMPLATE = r"""<!doctype html><meta charset=utf-8><title>Transfer test: __ART__</title>
<style>
body{font:15px/1.9 system-ui,sans-serif;max-width:1240px;margin:1.2rem auto;padding:0 1rem;background:#fbfbf9;color:#141414}
h1{font-size:1.15rem;margin:0 0 .2rem} .note{color:#555;font-size:.83rem;margin-bottom:.6rem}
.topbar{position:sticky;top:0;z-index:30;background:#fbfbf9;padding:5px 0 7px;border-bottom:1px solid #eae9e2;margin-bottom:.5rem}
.legend{display:flex;flex-wrap:wrap;gap:4px 12px;font-size:.76rem;margin:4px 0 0}
.legend span{white-space:nowrap} .dot{display:inline-block;width:10px;height:10px;border-radius:2px;vertical-align:middle;margin-right:3px}
.lg{cursor:pointer;user-select:none} .lg.off{opacity:.4;text-decoration:line-through}
.bar-nav{display:flex;gap:10px;align-items:center;margin:0}
.bar-nav button{font:inherit;padding:2px 10px;cursor:pointer}
.wrap{display:flex;gap:1.3rem;align-items:flex-start}
#main{flex:1;min-width:0}
#para{line-height:2.3}
.tok{display:inline-block;white-space:pre;border-radius:3px;padding:1px 0;cursor:default;transition:transform .08s}
.tok:hover{transform:scale(1.3);font-weight:600;box-shadow:0 1px 6px rgba(0,0,0,.15);position:relative;z-index:5;outline:1px solid #999}
.tok.pin{outline:2px solid #e0a}
#side{flex:0 0 320px;position:sticky;top:74px;max-height:calc(100vh - 86px);overflow:auto;border:1px solid #e2e1da;border-radius:10px;padding:11px 13px;background:#fff}
#side .hd{font-weight:700;border-bottom:1px solid #eee;padding-bottom:5px;margin-bottom:6px;word-break:break-word}
#side .sec{font-size:10px;letter-spacing:.05em;text-transform:uppercase;color:#999;margin:10px 0 4px}
#side .hint{color:#aaa;font-size:.85rem}
.row{display:flex;align-items:center;gap:5px;margin:2px 0}
.row .w{flex:0 0 92px;font-family:ui-monospace,monospace;font-size:11px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.row .barwrap{flex:1;background:#f0efe9;border-radius:3px;height:10px;overflow:hidden}
.row .bar{display:block;height:100%}
.row .co{flex:0 0 32px;text-align:right;color:#555;font-size:11px}
.cloud{display:flex;flex-wrap:wrap;gap:3px 10px;align-items:baseline;line-height:1.3}
.cloud .wd{display:inline-block}
</style>
<h1>Transfer test — __ART__ writeup · __L__</h1>
<div class=note>Real writeup, fed with context. A token is tinted by its strongest vulnerability concept
where it clearly beats both null baselines (benign, neutral). <b>Hover</b> a token (click to pin): the
right panel shows its DiM concept bars and its <b>J-space as a word cloud</b> — word size &amp; darkness
&prop; NN-OMP coefficient.</div>
<div class=topbar>
<div class=bar-nav><button id=prev>&larr; prev</button><span id=pageinfo></span><button id=next>next &rarr;</button></div>
<div class=legend id=legend></div>
</div>
<div class=wrap>
  <div id=main><div id=para></div></div>
  <aside id=side><div class=hint>Hover a token to see its J-space word cloud.</div></aside>
</div>
<script>
const DATA = __DATA__;
const C = DATA.concepts, PAGES = DATA.pages;
const COLORS = ['#e6194B','#3cb44b','#4363d8','#f58231','#911eb4','#42d4f4','#f032e6','#9A6324','#808000','#000075'];
const side=document.getElementById('side'); let pinned=null, cur=0, shownTok=null;
const BEN=C.indexOf('benign'), NEU=C.indexOf('neutral'), THR=__THR__;
const OFF_DEFAULT=['command_injection','path_traversal'];   // irrelevant here; toggle in legend
let enabled=C.map((c,i)=> i<10 && !OFF_DEFAULT.includes(c));
function nullp(t){let n=(BEN>=0?t.d[BEN]:-9); if(NEU>=0&&t.d[NEU]>n)n=t.d[NEU]; return n;}
let mmax=THR+0.1;
function recompute(){ mmax=THR+0.1;   // re-derive tint concept + margin from the ENABLED vuln set
  PAGES.forEach(p=>p.tokens.forEach(t=>{ let bi=-1,bv=-1e9;
    for(let i=0;i<10;i++){ if(enabled[i]&&t.d[i]>bv){bv=t.d[i];bi=i;} }
    t._tc=bi; t._m=(bi>=0?bv:-1e9)-nullp(t); if(t._m>mmax)mmax=t._m; })); }
function esc(s){return s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');}
function drawLegend(){ const el=document.getElementById('legend');
  el.innerHTML = C.slice(0,10).map((c,i)=>`<span class="lg${enabled[i]?'':' off'}" data-i="${i}"><span class=dot style="background:${enabled[i]?COLORS[i]:'#ccc'}"></span>${esc(c)}</span>`).join('')
    + `<span style="color:#999"><span class=dot style="background:#ccc"></span>benign/neutral = null · click a concept to toggle</span>`;
  el.querySelectorAll('.lg').forEach(s=>s.onclick=()=>{const i=+s.dataset.i; enabled[i]=!enabled[i];
    recompute(); drawLegend(); render(); if(shownTok)showSide(shownTok);}); }
function tint(t){ if(t._tc<0 || t._m<=THR || !/[A-Za-zÀ-￿]/.test(t.t)) return 'transparent';
  const a=Math.min(1,(t._m-THR)/(mmax-THR+1e-6))*0.85; const c=COLORS[t._tc];
  const r=parseInt(c.slice(1,3),16),g=parseInt(c.slice(3,5),16),b=parseInt(c.slice(5,7),16);
  return `rgba(${r},${g},${b},${a.toFixed(3)})`; }
function bars(pairs, colorFn){ const mx=Math.max(...pairs.map(p=>Math.abs(p[1])),1e-6);
  return pairs.map(([lab,val],k)=>`<div class=row><span class=w>${esc(lab)}</span>
    <span class=barwrap><span class=bar style="width:${(100*Math.abs(val)/mx).toFixed(0)}%;background:${colorFn(k,val)}"></span></span>
    <span class=co>${val.toFixed(2)}</span></div>`).join(''); }
function cloud(js){ if(!js.length) return '<span class=hint>—</span>';
  const vals=js.map(x=>x[1]), mx=Math.max(...vals), mn=Math.min(...vals);
  return '<div class=cloud>'+js.map(([w,c])=>{
    const t=mx>mn?(c-mn)/(mx-mn):1, f=(13+t*19).toFixed(0);
    const col=`rgb(${Math.round(150-100*t)},${Math.round(170-100*t)},${Math.round(205-95*t)})`;
    const word=esc(w.replace(/^\s+|\s+$/g,'')||w);
    return `<span class=wd title="coef ${c.toFixed(2)}" style="font-size:${f}px;color:${col};font-weight:${400+Math.round(t*300)}">${word}</span>`;
  }).join(' ')+'</div>'; }
function showSide(t){ shownTok=t;
  const dpairs = C.map((c,i)=>[c,i,t.d[i]]).filter(x=>x[1]>=10||enabled[x[1]]).map(x=>[x[0],x[2]]).sort((a,b)=>b[1]-a[1]).slice(0,6);
  const dimHtml = bars(dpairs, (k,v)=>{const ci=C.indexOf(dpairs[k][0]); return ci<10?(v>=0?COLORS[ci]:'#bbb'):'#999';});
  side.innerHTML=`<div class=hd>${esc(JSON.stringify(t.t))} <span style="font-weight:400;color:#888">· ${(100*t.r).toFixed(0)}% J-explained · ${t.j.length} concepts</span></div>
    <div class=sec>DiM security concepts</div>${dimHtml}
    <div class=sec>J-space word cloud (size &prop; coef)</div>${cloud(t.j)}`;
}
function render(){
  const pg=PAGES[cur], para=document.getElementById('para'); para.innerHTML='';
  document.getElementById('pageinfo').textContent=`page ${cur+1} / ${PAGES.length} (doc page ${pg.page}, tokens ${pg.start}–${pg.start+pg.n})`;
  pg.tokens.forEach(t=>{ const s=document.createElement('span'); s.className='tok'; s.textContent=t.t;
    s.style.background=tint(t);
    s.onmouseenter=()=>{if(!pinned)showSide(t);};
    s.onclick=(e)=>{e.stopPropagation();
      if(pinned===s){pinned.classList.remove('pin');pinned=null;}
      else{if(pinned)pinned.classList.remove('pin');pinned=s;s.classList.add('pin');showSide(t);}};
    para.appendChild(s); });
}
document.getElementById('prev').onclick=()=>{if(cur>0){cur--;render();}};
document.getElementById('next').onclick=()=>{if(cur<PAGES.length-1){cur++;render();}};
recompute(); drawLegend(); render();
</script>
"""


if __name__ == "__main__":
    main()
