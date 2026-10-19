"""Head-to-head: DiM-steer vs LoReFT-steer (last AND all schemes) for happy / sad, 3 prompts.

Columns per prompt (neutral / happy / sad):
  baseline      plain generation
  happy DiM     additive h += alpha*||h||*happy_hat at L17, ALL positions, continuous
  happy LoReFT  the happy LoReFT -- generates BOTH schemes (last / all), a page toggle picks which
  sad DiM       additive sad steering
  sad LoReFT    the sad LoReFT (last / all)

Five selectable readers score every generation token-by-token:
  DiM             projection onto the grand-mean-centred happy / sad directions
  read q (last)   Fisher-LDA in the q_i (read) subspace of the LAST-scheme happy / sad LoReFTs
  read q (all)    ... of the ALL-scheme LoReFTs
  write p (last)  Fisher-LDA in the p_i (write) subspace of the LAST-scheme LoReFTs
  write p (all)   ... of the ALL-scheme LoReFTs
For every reader, happy is read by the HAPPY LoReFT's subspace and sad by the SAD LoReFT's.
Scores are z-scored on neutral corpus tokens; a token is tinted toward max(happy, sad) >= threshold.

  .venv/bin/python scripts/reft/steer_compare.py
"""
import html, json, os, sys
import numpy as np, torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import common, pyreft
from concept_read import lda_fit

L = 17
PROMPTS = {
    "neutral": "Write a short paragraph about a train station at midday.",
    "happy":   "Write a short paragraph about the most joyful day of your life.",
    "sad":     "Write a short paragraph about the day you lost someone you loved.",
}
ALPHA = 0.4
MAXNEW = 80
FIT_TOPICS = 22
SCHEMES = ["last", "all"]


def main():
    cv = torch.load("data/concepts/concept_vectors.pt", weights_only=False, map_location="cpu")
    concepts = cv["concepts"]; hi, si = concepts.index("happy"), concepts.index("sad")
    hat = cv["concept_hat"][L].float().numpy(); gm = cv["grand_mean"][L].float().numpy()
    dim = {"happy": hat[hi], "sad": hat[si]}
    rows = json.load(open("data/concepts/paragraphs_concepts.json"))["rows"]

    mid, kind = common.resolve("qwen35-4b")
    print(f"Loading {mid} ...")
    handler, model, tok, is_mm = common.load_model(mid, kind)
    dev = next(model.parameters()).device
    _, layers = common.get_decoder_layers(model)
    Hd = model.config.get_text_config().hidden_size
    dt = next(model.parameters()).dtype

    def resid(text):
        cap = {}
        hh = layers[L].register_forward_hook(
            lambda _m, _i, o: cap.__setitem__("h", common.layer_output_hidden(o)[0].float().cpu().numpy()))
        try:
            with torch.no_grad():
                model(input_ids=torch.tensor([tok(text, add_special_tokens=False).input_ids], device=dev))
        finally:
            hh.remove()
        return cap["h"].copy()

    # corpus residuals for the readers
    topics = sorted({r["topic"] for r in rows})[:FIT_TOPICS]
    Htr = {c: np.concatenate([resid(r["text"]) for r in rows if r["concept"] == c and r["topic"] in topics])
           for c in ["happy", "sad", "neutral"]}
    for c in Htr:
        print(f"  fit residuals {c}: {Htr[c].shape[0]} tokens")

    # LoReFT state dicts + read/write subspaces, for happy/sad x last/all
    lo = {(c, s): torch.load(f"data/reft/{c}_loreft_L{L}_{s}.pt", weights_only=False, map_location="cpu")["state_dict"]
          for c in ["happy", "sad"] for s in SCHEMES}
    QP = {}
    for (c, s), sd in lo.items():
        R = sd["rotate_layer"].float().numpy().T; W = sd["weight"].float().numpy()
        U, S, Qt = np.linalg.svd(W - R, full_matrices=False)
        sgn = np.sign((Htr[c] @ Qt.T).mean(0) - (Htr["neutral"] @ Qt.T).mean(0)); sgn[sgn == 0] = 1
        QP[(c, s)] = (Qt * sgn[:, None], R.T @ (U * sgn[None, :]))

    # readers: dim, read_last, read_all, write_last, write_all
    def build_reader(spec):
        proj = {}
        for c in ["happy", "sad"]:
            if spec == "dim":
                proj[c] = ("dim", dim[c])
            else:
                kind_, s = spec.split("_"); Qt, P = QP[(c, s)]; B = Qt.T if kind_ == "read" else P
                proj[c] = ("sub", B, lda_fit(Htr[c] @ B, Htr["neutral"] @ B))
        def raw(c, H):
            return (H - gm) @ proj[c][1] if proj[c][0] == "dim" else (H @ proj[c][1]) @ proj[c][2]
        st = {c: (raw(c, Htr["neutral"]).mean(), raw(c, Htr["neutral"]).std() + 1e-9) for c in ["happy", "sad"]}
        def score(H):
            return {c: (raw(c, H) - st[c][0]) / st[c][1] for c in ["happy", "sad"]}
        return score
    reader_specs = ["dim", "read_last", "read_all", "write_last", "write_all"]
    readers = {r: build_reader(r) for r in reader_specs}

    def render_chat(user):
        msgs = [{"role": "user", "content": user}]
        try:
            return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        except TypeError:
            return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

    def gen_plain(user, hook=None):
        ids = torch.tensor([tok(render_chat(user), add_special_tokens=False).input_ids], device=dev)
        h = layers[L].register_forward_hook(hook) if hook else None
        try:
            with torch.inference_mode():
                o = model.generate(input_ids=ids, max_new_tokens=MAXNEW, do_sample=False, pad_token_id=tok.eos_token_id)
        finally:
            if h: h.remove()
        return tok.decode(o[0, ids.shape[1]:], skip_special_tokens=True).strip()

    def dim_hook(c):
        u_t = torch.tensor(dim[c] / (np.linalg.norm(dim[c]) + 1e-9), device=dev, dtype=dt)
        def f(_m, _i, o):
            h = common.layer_output_hidden(o)
            h = h + ALPHA * h.norm(dim=-1, keepdim=True) * u_t
            return (h,) + tuple(o[1:]) if isinstance(o, tuple) else h
        return f

    reft_cache = {}
    def gen_loreft(user, c, s):
        key = (c, s)
        if key not in reft_cache:
            reft = pyreft.get_reft_model(model, pyreft.ReftConfig(representations={
                "component": f"model.language_model.layers[{L}].output", "low_rank_dimension": 4,
                "intervention": pyreft.LoreftIntervention(embed_dim=Hd, low_rank_dimension=4)}))
            reft.set_device(dev); model.eval()
            iv = list(reft.interventions.values())[0]; iv = iv[0] if isinstance(iv, (list, tuple)) else iv
            iv.load_state_dict(lo[key]); reft_cache[key] = reft
        reft = reft_cache[key]
        pi = torch.tensor([tok(render_chat(user), add_special_tokens=False).input_ids], device=dev)
        pos = [pi.shape[1] - 1] if s == "last" else list(range(pi.shape[1]))
        with torch.inference_mode():
            _, g = reft.generate({"input_ids": pi, "attention_mask": torch.ones_like(pi)},
                                 unit_locations={"sources->base": (None, [[pos]])},
                                 intervene_on_prompt=True, max_new_tokens=MAXNEW, do_sample=False,
                                 eos_token_id=tok.eos_token_id, pad_token_id=tok.eos_token_id)
        return tok.decode(g[0, pi.shape[1]:], skip_special_tokens=True).strip()

    def score_text(text):
        H = resid(text) if text else np.zeros((0, Hd), np.float32)
        ids = tok(text, add_special_tokens=False).input_ids
        toks = [{"t": tok.decode([tid])} for tid in ids[:len(H)]]
        scores, avg = {}, {}
        for rn, rf in readers.items():
            s = rf(H) if len(H) else {"happy": np.zeros(0), "sad": np.zeros(0)}
            scores[rn] = {"happy": [round(float(x), 2) for x in s["happy"]],
                          "sad": [round(float(x), 2) for x in s["sad"]]}
            avg[rn] = {"happy": round(float(np.mean(s["happy"])) if len(H) else 0.0, 2),
                       "sad": round(float(np.mean(s["sad"])) if len(H) else 0.0, 2)}
        return {"tokens": toks, "scores": scores, "avg": avg}

    conds = [("baseline", ("plain",)), ("happy DiM", ("dim", "happy")), ("happy LoReFT", ("loreft", "happy")),
             ("sad DiM", ("dim", "sad")), ("sad LoReFT", ("loreft", "sad"))]
    out = {"alpha": ALPHA, "prompts": PROMPTS, "conditions": [c[0] for c in conds],
           "readers": reader_specs, "schemes": SCHEMES, "cells": []}
    for pk, user in PROMPTS.items():
        for cname, spec in conds:
            if spec[0] == "plain":
                variants = {"-": score_text(gen_plain(user))}
            elif spec[0] == "dim":
                variants = {"-": score_text(gen_plain(user, dim_hook(spec[1])))}
            else:
                variants = {s: score_text(gen_loreft(user, spec[1], s)) for s in SCHEMES}
            out["cells"].append({"prompt": pk, "cond": cname, "loreft": spec[0] == "loreft", "variants": variants})
            tag = next(iter(variants)); a = variants[tag]["avg"]
            print(f"  [{pk:<7}|{cname:<12}|{tag}] "
                  + " ".join(f"{r}:H{a[r]['happy']:+.1f}/S{a[r]['sad']:+.1f}" for r in ["dim", "read_last", "read_all"]))
    json.dump(out, open("data/reft/steer_compare.json", "w"), ensure_ascii=False)
    render(out, "data/reft/steer_compare.html")
    print("saved data/reft/steer_compare.{json,html}")


CSS = """
:root{--bg:#fbfaf7;--fg:#171613;--muted:#76746c;--line:#e3e1d8;--head:#f1efe7;--card:#fff;--happy:#d9730d;--sad:#2a6fdb}
@media(prefers-color-scheme:dark){:root{--bg:#141412;--fg:#ecebe5;--muted:#9b998f;--line:#312f2a;--head:#1e1d1a;--card:#1a1917;--happy:#f0a04b;--sad:#6ea8ff}}
body{font:14px/1.5 system-ui,-apple-system,sans-serif;background:var(--bg);color:var(--fg);max-width:1320px;margin:0 auto;padding:1.3rem 1rem 4rem}
h1{font-size:1.3rem;margin:0 0 .3rem} p.note{color:var(--muted);max-width:94ch;font-size:.87rem}
.ctl{position:sticky;top:0;z-index:5;background:var(--bg);border-bottom:1px solid var(--line);padding:.5rem 0;display:flex;gap:1.3rem;align-items:center;flex-wrap:wrap;font-size:.9rem}
.ctl select,.ctl input{font:inherit;font-size:.85rem}
.grid{display:grid;grid-template-columns:repeat(5,1fr);gap:.6rem;margin-top:.8rem}
.hd{font-weight:600;font-size:.82rem;color:var(--muted);text-align:center;padding:.3rem}
.pl{grid-column:1/-1;font-weight:600;margin:.6rem 0 -.2rem;color:var(--muted);font-size:.85rem;border-top:1px solid var(--line);padding-top:.5rem}
.cell{background:var(--card);border:1px solid var(--line);border-radius:9px;padding:.55rem .6rem;font-size:12px;line-height:1.65}
.cell .sch{font-size:.68rem;color:var(--muted);text-transform:uppercase;letter-spacing:.04em;margin-bottom:.2rem}
.cell .txt{font-family:ui-monospace,Menlo,monospace;white-space:pre-wrap;word-break:break-word}
.cell .txt span{border-radius:3px}
.avg{margin-top:.4rem;font-size:.74rem;color:var(--muted);border-top:1px solid var(--line);padding-top:.3rem}
.avg b{font-weight:650} .h{color:var(--happy)} .s{color:var(--sad)} .spark{display:block;margin-top:.3rem}
"""
RLABEL = {"dim": "DiM vector", "read_last": "LoReFT read q (last)", "read_all": "LoReFT read q (all)",
          "write_last": "LoReFT write p (last)", "write_all": "LoReFT write p (all)"}


def render(o, path):
    ropts = "".join(f'<option value="{r}">{RLABEL[r]}</option>' for r in o["readers"])
    body = ["<h1>Steering happy / sad: DiM vs LoReFT (last & all)</h1>",
            f"<p class=note>Columns steer the row's prompt; DiM is additive α={o['alpha']} at ALL positions "
            f"(continuous), LoReFT intervenes at its scheme's prompt positions (prefill only). The "
            f"<b>LoReFT scheme</b> toggle switches the LoReFT boxes between the <i>last</i>-position and "
            f"<i>all</i>-prompt-position generations (baseline / DiM ignore it). Tokens are tinted toward "
            f"max(happy,sad) via the chosen <b>reader</b>; happy is read by the happy LoReFT's subspace, sad "
            f"by the sad LoReFT's. Sparkline = happy−sad; per-cell averages are mean happy / sad.</p>",
            '<div class=ctl>'
            f'<label>reader <select id=rdr>{ropts}</select></label>'
            '<label>LoReFT scheme <select id=sch><option value=last>last position</option>'
            '<option value=all>all prompt positions</option></select></label>'
            '<label>tint threshold <input id=thr type=range min=0 max=2 step=0.1 value=0.4><span id=thrv>0.4</span></label>'
            '</div>',
            '<div class=grid id=grid></div>']
    open(path, "w").write(
        f"<title>Steering happy/sad: DiM vs LoReFT</title>\n<style>{CSS}</style>\n" + "\n".join(body)
        + "\n<script>\nconst O=" + json.dumps(o, ensure_ascii=False).replace("<", "\\u003c") + ";\n" + JS + "\n</script>")


JS = r"""
const grid=document.getElementById('grid'), rdr=document.getElementById('rdr'),
      sch=document.getElementById('sch'), thr=document.getElementById('thr'), thrv=document.getElementById('thrv');
const CONDS=O.conditions, PK=Object.keys(O.prompts);
function cvar(v){return getComputedStyle(document.documentElement).getPropertyValue(v).trim();}
function variant(cell){ return cell.loreft ? cell.variants[sch.value] : cell.variants['-']; }
function draw(){
  const r=rdr.value, t=+thr.value; thrv.textContent=t.toFixed(1); grid.innerHTML='';
  CONDS.forEach(c=>{const h=document.createElement('div');h.className='hd';h.textContent=c;grid.appendChild(h);});
  PK.forEach(pk=>{
    const lab=document.createElement('div'); lab.className='pl';
    lab.textContent=`${pk} prompt — "${O.prompts[pk]}"`; grid.appendChild(lab);
    CONDS.forEach(cn=>{
      const cell=O.cells.find(x=>x.prompt===pk&&x.cond===cn), v=variant(cell);
      const d=document.createElement('div'); d.className='cell';
      if(cell.loreft) d.insertAdjacentHTML('beforeend',`<div class=sch>scheme: ${sch.value}</div>`);
      const box=document.createElement('div'); box.className='txt';
      const H=v.scores[r].happy, S=v.scores[r].sad;
      v.tokens.forEach((tk,i)=>{
        const hv=H[i]||0, sv=S[i]||0, win=hv>=sv?'--happy':'--sad', z=Math.max(hv,sv);
        const sp=document.createElement('span'); sp.textContent=tk.t;
        if(z>=t){ const a=Math.min(1,(z-t)/1.8)*0.6+0.12;
          sp.style.background=`color-mix(in srgb, ${cvar(win)} ${Math.round(a*100)}%, transparent)`;}
        sp.title=`happy ${hv.toFixed(2)} · sad ${sv.toFixed(2)}`; box.appendChild(sp);});
      d.appendChild(box);
      const n=v.tokens.length;
      if(n>1){ const w=Math.max(120,Math.min(240,n*2)), hh=22;
        const diff=v.tokens.map((_,i)=>(H[i]||0)-(S[i]||0)), mx=Math.max(1,...diff.map(Math.abs));
        const pts=diff.map((val,i)=>`${(i/(n-1)*w).toFixed(1)},${(hh/2-val/mx*hh/2).toFixed(1)}`).join(' ');
        d.insertAdjacentHTML('beforeend',
          `<svg class=spark width=${w} height=${hh}><line x1=0 y1=${hh/2} x2=${w} y2=${hh/2} stroke='var(--line)'/>`
          +`<polyline points='${pts}' fill=none stroke='var(--muted)' stroke-width=1.2/></svg>`);}
      const av=v.avg[r];
      d.insertAdjacentHTML('beforeend',
        `<div class=avg>avg <b class=h>happy ${av.happy>=0?'+':''}${av.happy.toFixed(2)}</b> · `
        +`<b class=s>sad ${av.sad>=0?'+':''}${av.sad.toFixed(2)}</b></div>`);
      grid.appendChild(d);});
  });
}
rdr.onchange=draw; sch.onchange=draw; thr.oninput=draw; draw();
"""


if __name__ == "__main__":
    main()
