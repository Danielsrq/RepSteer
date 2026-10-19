"""Add the per-token J-space wordcloud side-panel to data/security/v2_reader_view.html.

Reuses the reader viewer's embedded per-token data (reader scores + gold labels) and merges in, for
every token, its J-space decomposition under the NEW lens (lens_v2_L17, fit on the clean v2 corpus):
the residual is decomposed by non-negative OMP onto the transported-unembedding cone, giving the
vocabulary tokens that reconstruct it. Hovering a token fills a side panel with that wordcloud.

Forward-only (applies the lens; no backward). The heavy cost is the NN-OMP per token.

  .venv/bin/python scripts/jlens/13_v2_jspace_view.py --lens data/jlens/lens_v2_L17.pt
"""
import argparse, json, os, sys, time
import numpy as np
import torch
import jlens                                  # BEFORE adding scripts/ to the path (namespace shadow)

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import common                                 # noqa: E402
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "reft"))
from span_labels_v2 import LABELS, CONCEPTS, NARRATIVE, COLOR, HOLDOUT, ARTICLE_CONCEPT  # noqa: E402
from v2_cache import token_labels             # noqa: E402


def concept_frame(m, J17, g, h, top_m):
    r0 = h.to(J17.device).float() @ J17.T
    logits = m.unembed(r0).float()
    top = logits.topk(top_m).indices
    w = m._lm_head.weight[top].float()
    V = (w * g) @ J17
    return top.detach().cpu().numpy(), V.detach().cpu().numpy()


def nnls(A, x, iters=300):
    if A.shape[1] == 0:
        return np.zeros(0)
    AtA, Atx = A.T @ A, A.T @ x
    Lc = float(np.linalg.eigvalsh(AtA)[-1]); lr = 1.0 / max(Lc, 1e-9)
    c = np.zeros(A.shape[1])
    for _ in range(iters):
        c = np.maximum(c - lr * (AtA @ c - Atx), 0.0)
    return c


def nn_omp(x, D, k):
    x2 = float(x @ x) + 1e-12
    residual = x.copy(); S, coeffs = [], np.zeros(0)
    for _ in range(k):
        corr = D @ residual
        if S:
            corr[np.array(S)] = -1e30
        j = int(np.argmax(corr))
        if corr[j] <= 0:
            break
        S.append(j); A = D[S].T; coeffs = nnls(A, x); residual = x - A @ coeffs
    return S, coeffs, 1.0 - float(residual @ residual) / x2


def load_viewer_O(path):
    s = open(path).read()
    a = s.index("const O=") + len("const O=")
    b = s.index(", CONC=O.concepts")
    return json.loads(s[a:b].replace("\\u003c", "<"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default="qwen35-4b")
    ap.add_argument("--lens", default="data/jlens/lens_v2_L17.pt")
    ap.add_argument("--layer", type=int, default=17)
    ap.add_argument("--layers", default=None,
                    help="comma list, e.g. 17,25,27; each gets its own cloud, sized by occupancy")
    ap.add_argument("--panel", default="data/jlens/panel_v2_security.json",
                    help="occupancy panel; cloud size per layer = that layer's occupancy")
    ap.add_argument("--viewer", default="data/security/v2_reader_view.html")
    ap.add_argument("--top-m", type=int, default=256)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--show", type=int, default=8)
    ap.add_argument("--page-size", type=int, default=2000)
    ap.add_argument("--prefix", type=int, default=512)
    ap.add_argument("--out", default="data/security/v2_reader_view.html")
    args = ap.parse_args()

    O = load_viewer_O(args.viewer)
    Ls = [int(x) for x in args.layers.split(",")] if args.layers else [args.layer]
    kmap = {}
    if os.path.exists(args.panel):
        pan = {r["layer"]: r["occupancy"] for r in json.load(open(args.panel))["panel"]}
        kmap = {L: max(int(pan.get(L, args.k)), 3) for L in Ls}
    else:
        kmap = {L: args.k for L in Ls}
    print(f"layers {Ls} | cloud size per layer (= occupancy): {kmap}")
    model_id, kind = common.resolve(args.preset)
    print(f"Loading {model_id} + lens {args.lens} ...")
    _, model, tok, _ = common.load_model(model_id, kind)
    m = jlens.from_hf(model, tok)
    lens = jlens.JacobianLens.load(args.lens)
    JL = {L: lens.jacobians[L].to(m.input_device).float() for L in Ls}
    g = m._final_norm.weight.detach().to(m.input_device).float()
    _, layers = common.get_decoder_layers(model)
    grab = {}
    def mk(L):
        def hook(_m, _i, out):
            grab[L] = common.layer_output_hidden(out)[0].detach().float().cpu()
        return hook
    hks = [layers[L].register_forward_hook(mk(L)) for L in Ls]

    t0 = time.perf_counter(); done = 0
    try:
        for art in O["articles"]:
            ids, _tl = token_labels(art["name"], tok)
            npages = (len(ids) + args.page_size - 1) // args.page_size
            Hs = {L: [] for L in Ls}
            for p in range(npages):
                start = p * args.page_size; ctx0 = max(0, start - args.prefix)
                with torch.inference_mode():
                    m.forward(torch.tensor([ids[ctx0:start + args.page_size]], device=m.input_device))
                for L in Ls:
                    Hs[L].append(grab[L][start - ctx0:])      # raw display residuals, per layer
            H = {L: torch.cat(Hs[L], 0) for L in Ls}
            n = min(min(len(H[L]) for L in Ls), len(art["tokens"]))
            mu = {L: H[L][:n].mean(0) for L in Ls}
            for i in range(n):
                for L in Ls:
                    idx, V = concept_frame(m, JL[L], g, H[L][i], args.top_m)
                    Vhat = V / (np.linalg.norm(V, axis=1, keepdims=True) + 1e-12)
                    x = (H[L][i] - mu[L]).numpy().astype(np.float64)
                    S, coeffs, recon = nn_omp(x, Vhat.astype(np.float64), kmap[L])
                    words = sorted([(tok.decode([int(idx[S[j]])]), float(coeffs[j]))
                                    for j in range(len(S)) if coeffs[j] > 1e-3], key=lambda z: -z[1])[:kmap[L]]
                    art["tokens"][i][f"j{L}"] = [[w, round(c, 2)] for w, c in words]
                    art["tokens"][i][f"jr{L}"] = round(recon, 3)
                done += 1
                if done % 2000 == 0:
                    dt = time.perf_counter() - t0
                    print(f"  {done} tokens ({dt:.0f}s, {dt/done*1000:.0f} ms/tok)")
    finally:
        [h.remove() for h in hks]
    print(f"Decomposed {done} tokens in {(time.perf_counter()-t0)/60:.1f} min")
    O["lens"] = os.path.basename(args.lens)
    O["jlayers"] = Ls
    O["jocc"] = kmap
    render(O, args.out)


CSS = """
:root{__PAL__;--bg:#fbfaf7;--fg:#171613;--muted:#76746c;--line:#e3e1d8;--head:#f1efe7;--card:#fff;--acc:#1f6fb2}
@media(prefers-color-scheme:dark){:root:not([data-theme="light"]){--bg:#141412;--fg:#ecebe5;--muted:#9b998f;--line:#312f2a;--head:#1e1d1a;--card:#1a1917;--acc:#62a8e6}}
:root[data-theme="dark"]{--bg:#141412;--fg:#ecebe5;--muted:#9b998f;--line:#312f2a;--head:#1e1d1a;--card:#1a1917;--acc:#62a8e6}
*{box-sizing:border-box}
body{font:14px/1.6 system-ui,-apple-system,sans-serif;background:var(--bg);color:var(--fg);max-width:1360px;margin:0 auto;padding:1.4rem 1.1rem 4rem}
h1{font-size:1.3rem;margin:0 0 .3rem} p.note{color:var(--muted);max-width:88ch;font-size:.88rem}
.tabbar{display:flex;gap:.4rem;flex-wrap:wrap;border-bottom:1px solid var(--line);margin:.8rem 0 0;position:sticky;top:0;background:var(--bg);padding-top:.3rem;z-index:8}
.tabbar button{font:inherit;font-size:.86rem;padding:.4rem .8rem;border:1px solid var(--line);border-bottom:none;border-radius:8px 8px 0 0;background:var(--head);color:var(--muted);cursor:pointer;margin-bottom:-1px}
.tabbar button.on{background:var(--card);color:var(--fg);font-weight:600}
.ctl{position:sticky;top:2.7rem;z-index:6;background:var(--bg);border-bottom:1px solid var(--line);padding:.5rem 0;display:flex;gap:.5rem 1.1rem;flex-wrap:wrap;align-items:center;font-size:.85rem}
.ctl b{font-weight:600;cursor:pointer;white-space:nowrap}.ctl b.off{opacity:.3;text-decoration:line-through}
.ctl select{font:inherit;font-size:.84rem}
.sw{display:inline-block;width:11px;height:11px;border-radius:3px;vertical-align:middle;margin-right:4px}
.info{color:var(--muted);font-size:.82rem;margin:.3rem 0}
.layout{display:flex;gap:1rem;align-items:flex-start}
.main{flex:1;min-width:0}
.txt{white-space:pre-wrap;font-family:ui-monospace,Menlo,monospace;font-size:11.5px;line-height:1.75;background:var(--card);border:1px solid var(--line);border-radius:10px;padding:1rem;margin:.4rem 0;overflow-x:auto}
.txt span{border-radius:3px;cursor:default} .txt span.hl{outline:2px solid var(--acc);outline-offset:1px}
.side{width:300px;flex:none;position:sticky;top:5.4rem;background:var(--card);border:1px solid var(--line);border-radius:10px;padding:.8rem .9rem;font-size:.85rem}
.jsel{display:block;font-size:.8rem;color:var(--muted);margin:0 0 .5rem}
.jsel select{font:inherit;font-size:.8rem}
.side h3{margin:.1rem 0 .5rem;font-size:.82rem;color:var(--muted);text-transform:uppercase;letter-spacing:.05em}
.side .tk{font-family:ui-monospace,Menlo,monospace;font-size:1.1rem;background:var(--head);padding:.1rem .4rem;border-radius:5px;word-break:break-all}
.side .rec{color:var(--muted);font-size:.78rem;margin:.35rem 0 .5rem}
.cloud{display:flex;flex-wrap:wrap;gap:.2rem .45rem;align-items:baseline;line-height:1.45;min-height:11rem}
.cloud span{font-family:ui-monospace,Menlo,monospace;white-space:pre}
.hint{color:var(--muted);font-size:.78rem}
@media(max-width:1100px){.side{position:static;width:100%}.layout{flex-direction:column}}
"""


def render(o, out):
    import html
    E = html.escape
    pal = ";".join(f"--c{i}:{o['color'][l]}" for i, l in enumerate(o["labels"]))
    css = CSS.replace("__PAL__", pal)
    body = ["<h1>Security write-ups: v2 readers + J-space wordcloud</h1>",
            f"<p class=note>Held-out articles tinted by a chosen 5-way reader over layer-17 streaming "
            f"residuals (reader / priors / margin / colour-by controls as before). <b>Hover any token</b> "
            f"to see its J-space decomposition in the side panel — the vocabulary directions (via the "
            f"lens <code>{E(o.get('lens','?'))}</code> refit on the clean v2 corpus) that reconstruct "
            f"that token's residual, sized by coefficient. Click a legend entry to dim it.</p>"]
    leg = " ".join(f'<b data-i="{i}"><span class=sw style="background:var(--c{i})"></span>{l}</b>'
                   for i, l in enumerate(o["labels"]))
    body.append('<div class=tabbar>' + "".join(
        f'<button data-a="{i}"{" class=on" if i == 0 else ""}>{E(a["name"])}'
        f' <span class=sw style="background:var(--c{o["labels"].index(a["concept"])})"></span>'
        f'{"" if not a["holdout"] else " ·hold"}</button>' for i, a in enumerate(o["articles"])) + "</div>")
    rdr_opts = "".join(f'<option value="{r}">{n}</option>' for r, n in
                       [("lda", "LDA (full 2560-d)"), ("dim", "DiM"),
                        ("read", "LoReFT read (q_i)"), ("write", "LoReFT write (p_i)")])
    body.append('<div class=ctl><span>' + leg + '</span>'
                '<label>reader <select id=rdr>' + rdr_opts + '</select></label>'
                '<label>colour by <select id=mode><option value=pred>prediction</option>'
                '<option value=true>ground truth</option></select></label>'
                f'<label>margin <input id=mg type=range min=0 max=8 step=0.1 value="{o["margin"]}">'
                '<span id=mgv></span></label>'
                '<label title="add log class-priors"><input type=checkbox id=pri> priors</label></div>')
    body.append('<div class=info id=info></div>')
    panes = "".join(
        f'<div class="pane" data-a="{i}"{" style=display:block" if i == 0 else ""}>'
        f'<div class=txt id=txt{i}></div></div>' for i in range(len(o["articles"])))
    jl = o.get("jlayers", [])
    occ = o.get("jocc", {})
    lyr_opts = "".join(f'<option value="{L}">L{L} · occupancy {occ.get(str(L), occ.get(L,"?"))}</option>'
                       for L in jl)
    side = ('<div class=side><h3>J-space</h3>'
            f'<label class=jsel>layer <select id=jlyr>{lyr_opts}</select></label>'
            '<div id=jcloud class=cloud></div>'
            '<div id=jrec class=rec></div></div>')
    body.append(f'<div class=layout><div class=main>{panes}</div>{side}</div>')

    js = """
const O=__O__, CONC=O.concepts.map(c=>O.labels.indexOf(c)), NARR=O.narrative.map(c=>O.labels.indexOf(c));
const off=new Set(), rdr=document.getElementById('rdr'), mode=document.getElementById('mode'),
      mg=document.getElementById('mg'), mgv=document.getElementById('mgv'), info=document.getElementById('info'),
      pri=document.getElementById('pri'), LP=O.logprior,
      jrec=document.getElementById('jrec'), jcloud=document.getElementById('jcloud'),
      jlyr=document.getElementById('jlyr');
let lastTok=null;
let cur=0;
function col(i){return getComputedStyle(document.documentElement).getPropertyValue('--c'+i).trim();}
document.querySelectorAll('.tabbar b').forEach(b=>b.onclick=()=>{const i=+b.dataset.i;
  off.has(i)?off.delete(i):off.add(i); b.classList.toggle('off',off.has(i)); draw();});
document.querySelectorAll('.tabbar button').forEach(b=>b.onclick=()=>{cur=+b.dataset.a;
  document.querySelectorAll('.tabbar button').forEach(x=>x.classList.toggle('on',+x.dataset.a===cur));
  document.querySelectorAll('.pane').forEach(p=>p.style.display=(+p.dataset.a===cur)?'block':'none'); draw();});
rdr.onchange=draw; mode.onchange=draw; mg.oninput=draw; pri.onchange=draw;
if(jlyr) jlyr.onchange=()=>{ if(lastTok) showJ(lastTok); };
function winner(s,m){ let k=0; for(let j=1;j<s.length;j++) if(s[j]>s[k]) k=j;
  if(CONC.indexOf(k)>=0){ let bn=-1e9,ni=-1; for(const j of NARR) if(s[j]>bn){bn=s[j];ni=j;}
    if(ni>=0 && s[k]-bn<m) return ni; } return k; }
function showJ(tk){
  lastTok=tk;
  const L=jlyr?jlyr.value:null, j=(L!=null?tk['j'+L]:tk.j)||[], rc=(L!=null?tk['jr'+L]:tk.jr);
  jrec.textContent = rc!=null ? `L${L} · reconstructs ${(100*rc).toFixed(0)}% of the residual`:'';
  jcloud.innerHTML='';
  const mx=j.length?j[0][1]:1;
  if(!j.length){ jcloud.innerHTML='<span class=hint>hover a token</span>'; return; }
  j.forEach(([w,c])=>{ const el=document.createElement('span');
    el.textContent=w===' '?'␠':w; el.style.fontSize=(0.72+0.95*c/mx).toFixed(2)+'rem';
    el.style.opacity=(0.45+0.55*c/mx).toFixed(2); el.title=c.toFixed(3); jcloud.appendChild(el); });
}
function draw(){
  const a=O.articles[cur], box=document.getElementById('txt'+cur), m=+mg.value, r=rdr.value; mgv.textContent=m.toFixed(1);
  const truth=mode.value==='true', usep=pri.checked;
  box.innerHTML=''; let gold=0,hit=0,conc=0;
  a.tokens.forEach(tk=>{
    const s = usep ? tk[r].map((v,j)=>v+LP[j]) : tk[r];
    const pred=winner(s,m), lab=truth?tk.g:pred;
    const el=document.createElement('span'); el.textContent=tk.t;
    if(lab>=0 && !off.has(lab)){ const isC=CONC.indexOf(lab)>=0, al=isC?0.34:(O.labels[lab]==='triage'?0.14:0.07);
      el.style.background=`color-mix(in srgb, ${col(lab)} ${Math.round(al*100)}%, transparent)`; }
    el.onmouseenter=()=>{ showJ(tk);
      box.querySelectorAll('span.hl').forEach(x=>x.classList.remove('hl')); el.classList.add('hl'); };
    if(!truth && CONC.indexOf(pred)>=0) conc++;
    if(tk.g>=0){gold++; if(pred===tk.g) hit++;}
    el.title=O.labels.map((l,j)=>`${l}: ${s[j]>=0?'+':''}${s[j].toFixed(1)}`).join('\\n')
      +(tk.g>=0?`\\n— labelled: ${O.labels[tk.g]} —`:'');
    box.appendChild(el);});
  info.textContent = truth ? `${a.tokens.length} tokens · ground-truth span labels`
    : `${a.tokens.length} tokens · reader ${r} · ${conc} predicted a concept (${(100*conc/a.tokens.length).toFixed(0)}%) · `
      + (gold?`token accuracy vs labels ${(100*hit/gold).toFixed(0)}%`:'no labels');
}
draw();
"""
    js = js.replace("__O__", json.dumps(o, ensure_ascii=False).replace("<", "\\u003c"))
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    open(out, "w").write(f"<title>Security readers + J-space (v2)</title>\n<style>{css}</style>\n"
                         + "\n".join(body) + f"\n<script>{js}</script>")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
