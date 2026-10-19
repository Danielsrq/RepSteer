"""Tint the held-out security write-ups with the v2 5-label readers, with a reader selector.

Four readers, each producing a calibrated 5-way discriminant per token so they can be argmax-ed and
compared on the SAME streaming residuals:
  DiM    Gaussian discriminant with identity covariance  (h.mu_c - 0.5|mu_c|^2 + log pi_c)
  LDA    Gaussian discriminant with shared covariance    (h.Sinv mu_c - 0.5 mu_c.Sinv mu_c + log pi_c)
  read   multiclass LDA in the 20-d space spanned by all 5 LoReFT READ subspaces (q_i)
  write  multiclass LDA in the 20-d space spanned by all 5 LoReFT WRITE subspaces (p_i)
All share the same true-frequency priors, so switching readers changes only the directions, not the
calibration. A concept only wins if it beats the best narrative class by the margin.

  .venv/bin/python scripts/reft/v2_viewer.py         # -> data/security/v2_reader_view.html
"""
import argparse, collections, html, json, os, sys
import numpy as np, torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common
from span_labels_v2 import LABELS, CONCEPTS, NARRATIVE, COLOR, HOLDOUT, ARTICLE_CONCEPT
from v2_cache import token_labels

READERS = ["dim", "lda", "read", "write"]


def gaussian_disc(mu, prior, whiten=None):
    """Return (A, quad, logprior). Score = H @ A + quad [+ logprior if priors are on].
    quad = the -0.5 mu.Sinv.mu Gaussian term (always needed); logprior = log pi_c (optional, so a
    'priors off' mode can isolate directional quality from the class base rate).
    whiten=None -> identity covariance (DiM); else the precomputed Sinv (LDA)."""
    A = np.stack([(whiten @ mu[c] if whiten is not None else mu[c]) for c in LABELS], 1)  # d x 5
    quad = np.array([-0.5 * mu[c] @ (whiten @ mu[c] if whiten is not None else mu[c]) for c in LABELS])
    logp = np.array([np.log(prior[c]) for c in LABELS])
    return A.astype(np.float32), quad.astype(np.float32), logp.astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, default=17)
    ap.add_argument("--articles", default=None)
    ap.add_argument("--page-size", type=int, default=2000)
    ap.add_argument("--prefix", type=int, default=512)
    ap.add_argument("--margin", type=float, default=1.0)
    ap.add_argument("--reft-dir", default="data/reft/v2")
    ap.add_argument("--out", default="data/security/v2_reader_view.html")
    args = ap.parse_args()
    L = args.layer
    arts = args.articles.split(",") if args.articles else sorted(HOLDOUT)

    z = np.load(f"data/security/v2_resid_L{L}.npz", allow_pickle=True)
    H, lab, spl = z["H"].astype(np.float32), z["label"], z["split"]
    tr = spl == "train"
    Htr = H[tr]; ltr = lab[tr]
    mu = {c: Htr[ltr == c].mean(0) for c in LABELS}

    mid, kind = common.resolve("qwen35-4b")
    print(f"Loading {mid} | v2 viewer (4 readers) @ L{L} | {arts}")
    handler, model, tok, is_mm = common.load_model(mid, kind)
    _, layers = common.get_decoder_layers(model)
    dev = next(model.parameters()).device
    grab = {}
    hk = layers[L].register_forward_hook(
        lambda _m, _i, o: grab.__setitem__("H", common.layer_output_hidden(o)[0].float().cpu()))

    # true-frequency priors over the training articles
    tc = collections.Counter()
    for a in sorted(ARTICLE_CONCEPT):
        if a in HOLDOUT:
            continue
        _i, tl = token_labels(a, tok)
        tc.update(x for x in tl if x in LABELS)
    ntot = sum(tc.values()); prior = {c: max(tc[c], 1) / ntot for c in LABELS}
    print("priors: " + ", ".join(f"{c} {prior[c]:.3f}" for c in LABELS))

    # --- build the four readers as (A, b): scores = H @ A + b ---
    Sw = np.concatenate([Htr[ltr == c] - mu[c] for c in LABELS])
    Sw = Sw.T @ Sw / len(Sw); Sw += 1e-1 * np.trace(Sw) / Sw.shape[0] * np.eye(Sw.shape[0])
    Sinv = np.linalg.inv(Sw)
    disc = {"dim": gaussian_disc(mu, prior), "lda": gaussian_disc(mu, prior, Sinv)}

    def subspace_reader(kind_):  # kind_ in {"read","write"}
        cols = []
        for c in LABELS:
            sd = torch.load(f"{args.reft_dir}/{c}_loreft_L{L}.pt", weights_only=False,
                            map_location="cpu")["state_dict"]
            R = sd["rotate_layer"].float().numpy().T; W = sd["weight"].float().numpy()
            U, S, Qt = np.linalg.svd(W - R, full_matrices=False)
            cols.append(Qt.T if kind_ == "read" else R.T @ U)     # d x 4
        B = np.concatenate(cols, 1)                                # d x 20
        Rtr = Htr @ B                                              # project train tokens -> 20-d
        mur = {c: Rtr[ltr == c].mean(0) for c in LABELS}
        Sr = np.concatenate([Rtr[ltr == c] - mur[c] for c in LABELS])
        Sr = Sr.T @ Sr / len(Sr); Sr += 1e-1 * np.trace(Sr) / Sr.shape[0] * np.eye(Sr.shape[0])
        Sri = np.linalg.inv(Sr)
        A20, quad, logp = gaussian_disc(mur, prior, Sri)          # 20x5, 5, 5
        return (B @ A20).astype(np.float32), quad, logp           # fold projection in: d x 5
    disc["read"] = subspace_reader("read")
    disc["write"] = subspace_reader("write")

    out_arts = []
    try:
        for a in arts:
            ids, tl = token_labels(a, tok)
            npages = (len(ids) + args.page_size - 1) // args.page_size
            toks = []
            for p in range(npages):
                start = p * args.page_size; ctx0 = max(0, start - args.prefix)
                with torch.no_grad():
                    model(input_ids=torch.tensor([ids[ctx0:start + args.page_size]], device=dev))
                Hd = grab["H"][start - ctx0:].numpy(); Hd = Hd - Hd.mean(0)
                disp = ids[start:start + len(Hd)]; seg = tl[start:start + len(Hd)]
                sc = {r: (Hd @ disc[r][0] + disc[r][1]) for r in READERS}   # H@A + quad (no prior)
                for i in range(len(Hd)):
                    g = seg[i] if seg[i] in LABELS else None
                    toks.append({"t": tok.decode([disp[i]]), "g": LABELS.index(g) if g else -1,
                                 **{r: [round(float(x), 1) for x in sc[r][i]] for r in READERS}})
                print(f"  {a} page {p}: {len(Hd)} tokens")
            out_arts.append({"name": a, "holdout": a in HOLDOUT,
                             "concept": ARTICLE_CONCEPT.get(a, "?"), "tokens": toks})
    finally:
        hk.remove()
    o = {"labels": LABELS, "concepts": CONCEPTS, "narrative": NARRATIVE, "readers": READERS,
         "color": {l: COLOR[l] for l in LABELS}, "margin": args.margin, "articles": out_arts,
         "logprior": [round(float(x), 2) for x in disc["lda"][2]]}   # same priors for all readers
    render(o, args.out)


def render(o, out):
    E = html.escape
    pal = ";".join(f"--c{i}:{o['color'][l]}" for i, l in enumerate(o["labels"]))
    CSS = """
:root{%s;--bg:#fbfaf7;--fg:#171613;--muted:#76746c;--line:#e3e1d8;--head:#f1efe7;--card:#fff}
@media(prefers-color-scheme:dark){:root:not([data-theme="light"]){--bg:#141412;--fg:#ecebe5;--muted:#9b998f;--line:#312f2a;--head:#1e1d1a;--card:#1a1917}}
:root[data-theme="dark"]{--bg:#141412;--fg:#ecebe5;--muted:#9b998f;--line:#312f2a;--head:#1e1d1a;--card:#1a1917}
*{box-sizing:border-box}
body{font:14px/1.6 system-ui,-apple-system,sans-serif;background:var(--bg);color:var(--fg);max-width:1100px;margin:0 auto;padding:1.4rem 1.1rem 4rem}
h1{font-size:1.3rem;margin:0 0 .3rem} p.note{color:var(--muted);max-width:82ch;font-size:.88rem}
.tabbar{display:flex;gap:.4rem;flex-wrap:wrap;border-bottom:1px solid var(--line);margin:.8rem 0 0;position:sticky;top:0;background:var(--bg);padding-top:.3rem;z-index:8}
.tabbar button{font:inherit;font-size:.86rem;padding:.4rem .8rem;border:1px solid var(--line);border-bottom:none;border-radius:8px 8px 0 0;background:var(--head);color:var(--muted);cursor:pointer;margin-bottom:-1px}
.tabbar button.on{background:var(--card);color:var(--fg);font-weight:600}
.ctl{position:sticky;top:2.7rem;z-index:6;background:var(--bg);border-bottom:1px solid var(--line);padding:.5rem 0;display:flex;gap:.5rem 1.1rem;flex-wrap:wrap;align-items:center;font-size:.85rem}
.ctl b{font-weight:600;cursor:pointer;white-space:nowrap}.ctl b.off{opacity:.3;text-decoration:line-through}
.ctl select{font:inherit;font-size:.84rem}
.sw{display:inline-block;width:11px;height:11px;border-radius:3px;vertical-align:middle;margin-right:4px}
.info{color:var(--muted);font-size:.82rem;margin:.3rem 0}
.txt{white-space:pre-wrap;font-family:ui-monospace,Menlo,monospace;font-size:11.5px;line-height:1.7;background:var(--card);border:1px solid var(--line);border-radius:10px;padding:1rem;margin:.4rem 0;overflow-x:auto}
.txt span{border-radius:3px}
""" % pal
    body = ["<h1>Security write-ups read with the v2 5-label readers</h1>",
            "<p class=note>Held-out articles, tinted by a chosen 5-way reader over layer-17 streaming "
            "residuals. <b>reader</b> switches between DiM, full-space LDA, and the LoReFT read (q_i) "
            "and write (p_i) subspaces — same residuals, same priors, only the directions differ. A "
            "concept (overflow/uaf/race) wins only when it beats the best narrative class by the "
            "margin. <b>colour by</b> flips prediction vs the ground-truth span labels. Hover a token "
            "for the current reader's 5 scores + gold label; click a legend entry to dim it.</p>"]
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
                '<label title="add log class-priors: calibrated decision, but the 74% narrative '
                'base rate swamps every reader except full LDA"><input type=checkbox id=pri> priors</label>'
                '</div>')
    body.append('<div class=info id=info></div>')
    for i, a in enumerate(o["articles"]):
        body.append(f'<div class="pane" data-a="{i}"{" style=display:block" if i==0 else ""}>'
                    f'<div class=txt id=txt{i}></div></div>')
    js = """
const O=__O__, CONC=O.concepts.map(c=>O.labels.indexOf(c)), NARR=O.narrative.map(c=>O.labels.indexOf(c));
const off=new Set(), rdr=document.getElementById('rdr'), mode=document.getElementById('mode'),
      mg=document.getElementById('mg'), mgv=document.getElementById('mgv'), info=document.getElementById('info'),
      pri=document.getElementById('pri'), LP=O.logprior;
let cur=0;
function col(i){return getComputedStyle(document.documentElement).getPropertyValue('--c'+i).trim();}
document.querySelectorAll('.tabbar b').forEach(b=>b.onclick=()=>{const i=+b.dataset.i;
  off.has(i)?off.delete(i):off.add(i); b.classList.toggle('off',off.has(i)); draw();});
document.querySelectorAll('.tabbar button').forEach(b=>b.onclick=()=>{cur=+b.dataset.a;
  document.querySelectorAll('.tabbar button').forEach(x=>x.classList.toggle('on',+x.dataset.a===cur));
  document.querySelectorAll('.pane').forEach(p=>p.style.display=(+p.dataset.a===cur)?'block':'none'); draw();});
rdr.onchange=draw; mode.onchange=draw; mg.oninput=draw; pri.onchange=draw;
function winner(s,m){
  let k=0; for(let j=1;j<s.length;j++) if(s[j]>s[k]) k=j;
  if(CONC.indexOf(k)>=0){ let bn=-1e9,ni=-1;
    for(const j of NARR) if(s[j]>bn){bn=s[j];ni=j;}
    if(ni>=0 && s[k]-bn<m) return ni; }
  return k;
}
function draw(){
  const a=O.articles[cur], box=document.getElementById('txt'+cur), m=+mg.value, r=rdr.value; mgv.textContent=m.toFixed(1);
  const truth=mode.value==='true', usep=pri.checked;
  box.innerHTML=''; let gold=0,hit=0,conc=0;
  a.tokens.forEach(tk=>{
    const s = usep ? tk[r].map((v,j)=>v+LP[j]) : tk[r];
    const pred=winner(s,m), lab=truth?tk.g:pred;
    const el=document.createElement('span'); el.textContent=tk.t;
    if(lab>=0 && !off.has(lab)){
      const isC=CONC.indexOf(lab)>=0, al=isC?0.34:(O.labels[lab]==='triage'?0.14:0.07);
      el.style.background=`color-mix(in srgb, ${col(lab)} ${Math.round(al*100)}%, transparent)`;}
    if(!truth && CONC.indexOf(pred)>=0) conc++;
    if(tk.g>=0){gold++; if(pred===tk.g) hit++;}
    el.title=O.labels.map((l,j)=>`${l}: ${s[j]>=0?'+':''}${s[j].toFixed(1)}`).join('\\n')
      +(tk.g>=0?`\\n— labelled: ${O.labels[tk.g]} —`:'');
    box.appendChild(el);});
  info.textContent = truth
    ? `${a.tokens.length} tokens · ground-truth span labels`
    : `${a.tokens.length} tokens · reader ${r} · ${conc} predicted a concept (${(100*conc/a.tokens.length).toFixed(0)}%) · `
      + (gold?`token accuracy vs labels ${(100*hit/gold).toFixed(0)}%`:'no labels');
}
draw();
"""
    js = js.replace("__O__", json.dumps(o, ensure_ascii=False).replace("<", "\\u003c"))
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    open(out, "w").write(f"<title>Security readers (v2)</title>\n<style>{CSS}</style>\n"
                         + "\n".join(body) + f"\n<script>{js}</script>")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
