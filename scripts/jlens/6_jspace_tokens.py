"""Interactive per-token J-space viewer.

For every token position in a set of paragraphs (security + general + other domains),
decompose the centered layer-17 residual into its J-space concepts by non-negative
orthogonal matching pursuit (same method as 5_jspace.py, but run at ALL positions).
Renders a self-contained HTML page: hover a token to magnify it and pop up a box of its
top J-space concepts (with coefficient bars); click to pin. Tokens are tinted by how much
of their (centered) residual the top-K concepts reconstruct ("verbalizability").

Forward-only + small linear algebra (NNLS on CPU) -> cheap; no backward passes.

Usage:  .venv/bin/python scripts/jlens/6_jspace_tokens.py --top-m 256 --k 10
"""

import argparse
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

PARAGRAPHS = [
    ("security",
     "A remote attacker can exploit the heap buffer overflow to overwrite a function "
     "pointer and execute arbitrary code with kernel privileges, then escalate to root, "
     "disable logging, and install a persistent backdoor on the compromised server."),
    ("general",
     "The Roman aqueducts carried fresh water across great stone arches into the heart of "
     "the imperial capital, supplying its fountains, public baths, and gardens for "
     "centuries until the network fell into disrepair during the later empire."),
    ("science",
     "Photosynthesis converts carbon dioxide and water into glucose and oxygen using light "
     "captured by chlorophyll in the chloroplasts of plant cells, forming the base of "
     "nearly every food chain and regulating the composition of the atmosphere."),
    ("finance",
     "The quarterly earnings report showed revenue rising twelve percent while operating "
     "costs declined, and management raised full-year guidance for margins and free cash "
     "flow on stronger demand in the enterprise software segment."),
    ("story",
     "She felt a sudden wave of joy and gratitude wash over her as the train pulled into "
     "the station where her family stood waiting on the crowded platform, waving beneath "
     "the pale winter sun that broke through the grey clouds overhead."),
]


def concept_frame(m, J17, g, h, top_m):
    """Top-M J-lens cone at residual h: returns (vocab_indices[M], V[M,d] numpy raw)."""
    r0 = h.to(J17.device).float() @ J17.T
    logits = m.unembed(r0).float()
    top = logits.topk(top_m).indices
    w = m._lm_head.weight[top].float()
    V = (w * g) @ J17                      # [M, d] transported-unembedding directions
    return top.detach().cpu().numpy(), V.detach().cpu().numpy()


def nnls(A, x, iters=300):
    """min_{c>=0} ||A c - x||^2 via projected gradient on the normal equations (CPU numpy)."""
    if A.shape[1] == 0:
        return np.zeros(0)
    AtA, Atx = A.T @ A, A.T @ x
    L = float(np.linalg.eigvalsh(AtA)[-1])
    lr = 1.0 / max(L, 1e-9)
    c = np.zeros(A.shape[1])
    for _ in range(iters):
        c = np.maximum(c - lr * (AtA @ c - Atx), 0.0)
    return c


def nn_omp(x, D, k):
    """Non-negative OMP over unit dictionary D[M,d]; returns (sel_idx, coeffs, recon_frac)."""
    x2 = float(x @ x) + 1e-12
    residual = x.copy()
    S, coeffs = [], np.zeros(0)
    for _ in range(k):
        corr = D @ residual
        if S:
            corr[np.array(S)] = -1e30
        j = int(np.argmax(corr))
        if corr[j] <= 0:
            break
        S.append(j)
        A = D[S].T
        coeffs = nnls(A, x)
        residual = x - A @ coeffs
    recon = 1.0 - float(residual @ residual) / x2
    return S, coeffs, recon


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default="qwen35-4b")
    ap.add_argument("--lens", default="data/jlens/lens_L17.pt")
    ap.add_argument("--layer", type=int, default=17)
    ap.add_argument("--top-m", type=int, default=256)
    ap.add_argument("--k", type=int, default=10, help="max concepts decomposed per token")
    ap.add_argument("--show", type=int, default=8, help="max concepts shown in the popup")
    ap.add_argument("--out", default="data/jlens/jspace_tokens.html")
    args = ap.parse_args()

    model_id, kind = common.resolve(args.preset)
    print(f"Loading {model_id} + lens {args.lens} ...")
    _, model, tok, _ = common.load_model(model_id, kind)
    m = jlens.from_hf(model, tok)
    lens = jlens.JacobianLens.load(args.lens)
    J17 = lens.jacobians[args.layer].to(m.input_device).float()
    g = m._final_norm.weight.detach().to(m.input_device).float()
    print(f"{lens!r}")

    # --- collect layer-17 residuals at every position of every paragraph ---
    _, layers = common.get_decoder_layers(model)
    grab = {}
    handle = layers[args.layer].register_forward_hook(
        lambda mod, i, o: grab.__setitem__("h", common.layer_output_hidden(o)[0].detach().float().cpu()))
    paras = []
    try:
        for name, text in PARAGRAPHS:
            ids = m.encode(text)[0].tolist()
            with torch.inference_mode():
                m.forward(m.encode(text))
            paras.append({"name": name, "ids": ids, "H": grab["h"].clone()})
    finally:
        handle.remove()
    mu = torch.cat([p["H"] for p in paras], 0).mean(0)   # global mean over all positions

    # --- decompose every position ---
    t0 = time.perf_counter()
    n_pos = 0
    for p in paras:
        H, ids = p["H"], p["ids"]
        toks = []
        for pos in range(H.shape[0]):
            idx, V = concept_frame(m, J17, g, H[pos], args.top_m)
            Vhat = V / (np.linalg.norm(V, axis=1, keepdims=True) + 1e-12)
            x = (H[pos] - mu).numpy().astype(np.float64)
            S, coeffs, recon = nn_omp(x, Vhat.astype(np.float64), args.k)
            concepts = sorted(
                [(tok.decode([int(idx[S[i]])]), float(coeffs[i])) for i in range(len(S)) if coeffs[i] > 1e-3],
                key=lambda z: -z[1])[:args.show]
            toks.append({"t": tok.decode([ids[pos]]), "r": round(recon, 4),
                         "c": [[w, round(c, 3)] for w, c in concepts]})
            n_pos += 1
        p["toks"] = toks
    dt = time.perf_counter() - t0
    print(f"Decomposed {n_pos} positions in {dt:.1f}s ({dt/max(n_pos,1)*1000:.0f} ms/token) -- forward-only")

    data = [{"name": p["name"], "toks": p["toks"]} for p in paras]
    payload = json.dumps(data).replace("<", "\\u003c")
    doc = _TEMPLATE.replace("__DATA__", payload).replace("__K__", str(args.k)).replace("__L__", str(args.layer))
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    open(args.out, "w").write(doc)
    print(f"Wrote {args.out}")


_TEMPLATE = r"""<!doctype html><meta charset=utf-8><title>J-space per-token viewer</title>
<style>
body{font:15px/1.7 system-ui,-apple-system,sans-serif;max-width:900px;margin:1.4rem auto;padding:0 1rem;color:#141414;background:#fbfbf9}
h1{font-size:1.25rem;margin:0 0 .2rem} .note{color:#555;font-size:.85rem;margin-bottom:1.2rem}
h2{font-size:.72rem;letter-spacing:.08em;text-transform:uppercase;color:#8a8a8a;margin:1.5rem 0 .5rem;font-weight:700}
.para{line-height:2.4}
.tok{display:inline-block;white-space:pre;border-radius:4px;padding:1px 0;cursor:default;transition:transform .08s ease, background .1s}
.tok:hover{transform:scale(1.32);font-weight:600;background:#fff2b8!important;box-shadow:0 1px 6px rgba(0,0,0,.15);position:relative;z-index:5}
.tok.pin{outline:2px solid #e0a; background:#fff2b8!important}
#box{position:fixed;display:none;z-index:50;background:#fff;border:1px solid #d9d7cc;border-radius:8px;
  box-shadow:0 6px 24px rgba(0,0,0,.18);padding:10px 12px;min-width:210px;max-width:280px;font-size:12.5px}
#box .hd{font-weight:700;margin-bottom:6px;color:#222;border-bottom:1px solid #eee;padding-bottom:5px}
#box .hd small{font-weight:400;color:#888}
.row{display:flex;align-items:center;gap:6px;margin:3px 0}
.row .w{flex:0 0 88px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;font-family:ui-monospace,monospace;font-size:11.5px}
.row .barwrap{flex:1;background:#f0efe9;border-radius:3px;height:11px;overflow:hidden}
.row .bar{display:block;height:100%;background:#2166ac;min-width:1px}
.row .co{flex:0 0 34px;text-align:right;color:#555;font-variant-numeric:tabular-nums}
.empty{color:#aaa;font-style:italic}
.legend{display:flex;gap:14px;align-items:center;font-size:.8rem;color:#555;margin-top:.3rem}
.sw{display:inline-block;width:60px;height:11px;border-radius:3px;
  background:linear-gradient(90deg,#f7f7f7,#2166ac);vertical-align:middle}
</style>
<h1>J-space per-token viewer &mdash; layer __L__</h1>
<div class=note>Every token's centered residual is decomposed into a non-negative combination of up to
__K__ J-lens concept vectors (NN-OMP). <b>Hover</b> a token to magnify it and see its top J-space concepts
with coefficient bars; <b>click</b> to pin the box. Tokens are tinted by <b>verbalizability</b> &mdash; the
fraction of the centered residual reconstructed by its top concepts.
<div class=legend><span id=lmin>·</span>&nbsp;<span class=sw></span>&nbsp;<span id=lmax>·</span>
&nbsp;<span style="color:#999">= % of the token's centered residual <i>variance</i> its top concepts reconstruct</span></div></div>
<div id=root></div>
<div id=box></div>
<script>
const DATA = __DATA__;
const box = document.getElementById('box'), root = document.getElementById('root');
let pinned = null;
// reconstruction range for tint scaling + legend units
let rmax = 1e-6, rmin = 1e9;
DATA.forEach(p=>p.toks.forEach(t=>{if(t.r>rmax)rmax=t.r; if(t.r<rmin)rmin=t.r;}));
document.getElementById('lmin').textContent = (100*rmin).toFixed(1)+'%';
document.getElementById('lmax').textContent = (100*rmax).toFixed(1)+'%';
function tint(r){const a = Math.max(0, Math.min(1, r/rmax))*0.55;
  return `rgba(33,102,172,${a.toFixed(3)})`;}
function esc(s){return s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');}
function showBox(t, el){
  if(!t.c.length){box.innerHTML=`<div class=hd>${esc(JSON.stringify(t.t))}</div><div class=empty>no concepts above threshold</div>`;}
  else{
    const cmax = Math.max(...t.c.map(c=>c[1]));
    const rows = t.c.map(([w,co])=>`<div class=row><span class=w>${esc(JSON.stringify(w))}</span>
      <span class=barwrap><span class=bar style="width:${(100*co/cmax).toFixed(0)}%"></span></span>
      <span class=co>${co.toFixed(2)}</span></div>`).join('');
    box.innerHTML=`<div class=hd>${esc(JSON.stringify(t.t))} <small>&middot; ${(100*t.r).toFixed(1)}% explained, ${t.c.length} concepts</small></div>${rows}`;
  }
  const rc = el.getBoundingClientRect();
  box.style.display='block';
  let left = rc.left, top = rc.bottom+8;
  const bw = box.offsetWidth, bh = box.offsetHeight;
  if(left+bw>window.innerWidth-8) left = window.innerWidth-bw-8;
  if(top+bh>window.innerHeight-8) top = rc.top-bh-8;
  box.style.left=Math.max(8,left)+'px'; box.style.top=Math.max(8,top)+'px';
}
DATA.forEach((p,pi)=>{
  const h=document.createElement('h2'); h.textContent=p.name; root.appendChild(h);
  const div=document.createElement('div'); div.className='para';
  p.toks.forEach((t,ti)=>{
    const s=document.createElement('span'); s.className='tok'; s.textContent=t.t;
    s.style.background=tint(t.r);
    s.onmouseenter=()=>{if(!pinned)showBox(t,s);};
    s.onmouseleave=()=>{if(!pinned)box.style.display='none';};
    s.onclick=(e)=>{e.stopPropagation();
      if(pinned===s){pinned=null;s.classList.remove('pin');box.style.display='none';}
      else{if(pinned)pinned.classList.remove('pin');pinned=s;s.classList.add('pin');showBox(t,s);}};
    div.appendChild(s);
  });
  root.appendChild(div);
});
document.body.onclick=()=>{if(pinned){pinned.classList.remove('pin');pinned=null;box.style.display='none';}};
</script>
"""


if __name__ == "__main__":
    main()
