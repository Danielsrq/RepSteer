"""Per-token sparklines of several concept LoReFTs' channels (e.g. happy / sad / angry / calm), drawn under the text.

For each LoReFT (trained by concept_read.py), V = W - R = U S Q^T gives two sets of channels:
  READ  q_i = rows of Q^T   -- what gates the intervention ("how hard would it push here")
  WRITE p_i = cols of R^T U -- the register it edits, span(R) ("how much is already present")
Both are read off a token as a dot product with h; each set also gets a Fisher-LDA combination of all r channels.
Signs are fixed (independently per set) so the LoReFT's own concept scores above neutral on TRAINING topics, and every
channel is standardised with training-topic statistics (centred on neutral tokens, scaled by the std of all training
tokens), so the held-out paragraphs shown never set the scale. Default channel: the write LDA (best concept meter).

Shows held-out topics x all 12 concepts' paragraphs (the plotted emotions + neutral are on by default, the rest are
opt-in checkboxes), plus summary tables: per-channel AUCs, cosines with the DiM vectors, mean response per concept,
and the pairwise geometry (principal cosines) between every pair of LoReFTs. Tokens are highlighted by whichever
emotion scores them highest.

Usage:  .venv/bin/python scripts/reft/emotion_sparklines.py          (writes data/reft/emotion_sparklines_L17.{html,json})
"""
import argparse, html, json, os, sys
import numpy as np, torch
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(os.path.dirname(__file__))
import common
from concept_read import auc, lda_fit

# (light theme, dark theme) per emotion slot -- well-separated hues: orange / blue / magenta / teal / olive / purple
COLORS = {0: ("#d9730d", "#f0a04b"), 1: ("#2a6fdb", "#6ea8ff"), 2: ("#b5179e", "#e05fd0"), 3: ("#12897f", "#48c9bd"),
          4: ("#7a7500", "#c9c34a"), 5: ("#6a3fb5", "#a98ae6")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, default=17)
    ap.add_argument("--scheme", default="all", choices=["all", "last"])
    ap.add_argument("--corpus", default="emotion", choices=["emotion", "security"])
    ap.add_argument("--emotions", default=None, help="concepts to plot (comma list); default depends on --corpus")
    ap.add_argument("--out", default=None, help="default data/reft/{corpus}_sparklines_L{layer}")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--from-json", action="store_true",
                    help="rebuild only the HTML from the saved .json (no model load, no recompute)")
    args = ap.parse_args()
    L = args.layer
    DEFAULTS = {"emotion": "happy,sad", "security": "uaf,command_injection,path_traversal,benign"}
    EMO = (args.emotions or DEFAULTS[args.corpus]).split(",")
    args.out = args.out or f"data/reft/{args.corpus}_sparklines_L{L}"
    if args.from_json:
        o = json.load(open(args.out + ".json"))
        open(args.out + ".html", "w").write(render_html(o))
        print(f"Rebuilt {args.out}.html from {args.out}.json (no recompute)")
        return
    for f in (args.out + ".html", args.out + ".json"):
        if os.path.exists(f) and not args.force:
            sys.exit(f"refusing to overwrite existing {f} (pass --force or a different --out)")
    # ---------------- corpus-specific layer: checkpoints, paragraphs, train/held split, DiM vectors ----------------
    # Both corpora end up as: text[key], train_keys[concept], held_keys[concept], items (what to display), chat (DiM).
    if args.corpus == "emotion":
        ck_dir, corpus_file, vec_file = "data/reft", "data/concepts/paragraphs_concepts.json", "data/concepts/concept_vectors.pt"
    else:
        ck_dir, corpus_file, vec_file = "data/reft/security", "data/security/paragraphs_security.json", "data/security/security_vectors.pt"
    cks = {e: torch.load(f"{ck_dir}/{e}_loreft_L{L}_{args.scheme}.pt", weights_only=False, map_location="cpu") for e in EMO}
    rows = json.load(open(corpus_file))["rows"]
    vf = torch.load(vec_file, weights_only=False, map_location="cpu")
    concepts, emo_idx = vf["concepts"], vf["emotional_idx"]      # emo_idx = non-baseline concepts (emotions / vuln classes)
    lab = vf["labels"].numpy(); F = vf["features"][:, L].float().numpy(); del vf

    text, train_keys, held_keys, items, groups = {}, {}, {}, [], []
    if args.corpus == "emotion":
        n_topics = 30
        topics = [r["topic"] for r in sorted((r for r in rows if r["concept"] == "happy"), key=lambda r: r["id"])]
        held_s = {ck["args"]["heldout"] for ck in cks.values()}
        assert len(held_s) == 1, "LoReFTs were trained with different held-out topics"
        held_t = [int(x) for x in held_s.pop().split(",")]
        train_t = [t for t in range(n_topics) if t not in held_t]
        key = lambda c, t: f"{t}|{c}"
        for r in rows: text[key(r["concept"], r["id"] % n_topics)] = r["text"]
        for c in concepts:
            train_keys[c] = [key(c, t) for t in train_t]; held_keys[c] = [key(c, t) for t in held_t]
        groups = [topics[t] for t in held_t]                      # one card per held-out topic, rows = concepts
        items = [{"key": key(c, t), "group": topics[t], "label": c, "concept": c} for t in held_t for c in concepts]
        is_train = np.isin(np.array([r["id"] % n_topics for r in rows]), train_t)
    else:
        # security: topics are unique per concept, so hold out ROWS (as train_security.py did) and group cards by concept
        key = lambda c, t: f"{c}|{t}"
        for r in rows: text[key(r["concept"], r["topic"])] = r["text"]
        held_names = {}
        for c in concepts:
            if c in cks:                                          # the split recorded by training
                held_names[c] = list(cks[c]["heldout_topics"])
            else:                                                 # baselines have no LoReFT -> reproduce the same split
                import random as _rnd
                pool = [r["topic"] for r in rows if r["concept"] == c]
                _rnd.Random(cks[EMO[0]]["args"]["seed"]).shuffle(pool)
                held_names[c] = pool[:cks[EMO[0]]["args"]["n_heldout"]]
            allt = [r["topic"] for r in rows if r["concept"] == c]
            held_keys[c] = [key(c, t) for t in held_names[c]]
            train_keys[c] = [key(c, t) for t in allt if t not in held_names[c]]
        groups = [c for c in concepts if held_keys[c]]             # one card per concept, rows = its held-out write-ups
        items = [{"key": k, "group": c, "label": t, "concept": c}
                 for c in groups for t, k in zip(held_names[c], held_keys[c])]
        is_train = np.array([r["topic"] not in held_names[r["concept"]] for r in rows])
    baseline = "neutral"
    cms = np.stack([F[(lab == k) & is_train].mean(0) for k in range(len(concepts))]); gm = cms[emo_idx].mean(0)
    chat = (cms - gm) / np.linalg.norm(cms - gm, axis=1, keepdims=True)   # DiM vectors, training rows only
    del F

    mid, kind = common.resolve("qwen35-4b")
    print(f"Loading {mid} ... | {args.corpus}: {EMO} LoReFTs ({args.scheme}) @ L{L}")
    handler, model, tok, is_mm = common.load_model(mid, kind)
    _, layers = common.get_decoder_layers(model)
    dev = next(model.parameters()).device
    grab = {}
    hk = layers[L].register_forward_hook(lambda _m, _i, o: grab.__setitem__("H", common.layer_output_hidden(o)[0].float().cpu()))
    resid = {}
    need = sorted({k for c in EMO + [baseline] for k in train_keys[c]} | {it["key"] for it in items})
    with torch.no_grad():
        for k in need:
            ids = tok(text[k]).input_ids
            model(input_ids=torch.tensor([ids], device=dev)); resid[k] = (ids, grab["H"].numpy())
    hk.remove()
    print(f"captured {len(resid)} paragraphs")
    stack = lambda c, split: np.concatenate([resid[k][1] for k in (train_keys if split == "train" else held_keys)[c]])

    info, scores, geom = {}, {}, {}
    Qs, Ps = {}, {}
    for e in EMO:
        sd = cks[e]["state_dict"]
        R = sd["rotate_layer"].float().numpy().T; W = sd["weight"].float().numpy()
        U, S, Qt = np.linalg.svd(W - R, full_matrices=False)
        Htr_e, Htr_n = stack(e, "train"), stack(baseline, "train")
        sgn = np.sign((Htr_e @ Qt.T).mean(0) - (Htr_n @ Qt.T).mean(0)); sgn[sgn == 0] = 1
        Qt = Qt * sgn[:, None]; U = U * sgn[None, :]; P = R.T @ U
        Qs[e], Ps[e] = Qt, P
        others = [o for o in EMO if o != e]
        # write (register) channels: p_i = cols of R^T U. Their display signs are fixed INDEPENDENTLY of the read
        # signs (so each p_i scores this emotion above neutral); the underlying pairing M = sum_i s_i p_i q_i^T is
        # untouched -- only the sign of the displayed score changes.
        sgn_w = np.sign((Htr_e @ P).mean(0) - (Htr_n @ P).mean(0)); sgn_w[sgn_w == 0] = 1
        w_read = lda_fit(Htr_e @ Qt.T, Htr_n @ Qt.T)
        w_write = lda_fit(Htr_e @ P, Htr_n @ P)
        def chans(H):
            rd, wr = H @ Qt.T, H @ P
            d = {f"q{i+1}": rd[:, i] for i in range(len(S))}
            d["read LDA"] = rd @ w_read
            d.update({f"p{i+1}": wr[:, i] * sgn_w[i] for i in range(P.shape[1])})
            d["write LDA"] = wr @ w_write
            return d
        # standardisation from training topics: centre on neutral, scale by std over (own, neutral, other emotions)
        tr_all = np.concatenate([Htr_e, Htr_n] + [stack(o, "train") for o in others])
        c_tr_all, c_tr_n = chans(tr_all), chans(Htr_n)
        norm = {k: (float(c_tr_n[k].mean()), float(c_tr_all[k].std() + 1e-9)) for k in c_tr_all}
        # channel stats
        c_tr_e = chans(Htr_e)
        c_he, c_hn = chans(stack(e, "held")), chans(stack(baseline, "held"))
        c_ho = {o: chans(stack(o, "held")) for o in others}
        ch_rows = []
        for k in list(norm):
            idx = int(k[1:]) - 1 if k[0] in "qp" and k[1:].isdigit() else None    # channel index, or None for LDA rows
            vec = (Qt[idx] if k[0] == "q" else P[:, idx] * sgn_w[idx]) if idx is not None else None
            row = {"channel": k, "kind": "write" if k[0] == "p" or k == "write LDA" else "read",
                   "sigma": float(S[idx]) if idx is not None else None,      # the same gain pairs q_i with p_i
                   "train_auc_vs_neutral": auc(c_tr_e[k], c_tr_n[k]),
                   "held_auc_vs_neutral": auc(c_he[k], c_hn[k]),
                   "held_auc_vs_other": {o: auc(c_he[k], c_ho[o][k]) for o in others},
                   "cos_dim": ({c: float(vec @ chat[concepts.index(c)]) for c in EMO} if vec is not None else None),
                   "mean_z": {c: float(np.mean([((chans(resid[kk][1])[k] - norm[k][0]) / norm[k][1]).mean() for kk in held_keys[c]]))
                              for c in concepts if held_keys[c]}}
            ch_rows.append(row)
        pick = lambda pref: max([r for r in ch_rows if r["channel"].startswith(pref)],
                                key=lambda r: r["train_auc_vs_neutral"])["channel"]
        info[e] = {"channels": ch_rows, "best": pick("q"), "best_write": pick("p"), "sigma": S.tolist(),
                   "cos_qp": [float(Qt[i] @ P[:, i]) for i in range(len(S))]}
        scores[e] = {k: {} for k in norm}
        for it in items:
            ch = chans(resid[it["key"]][1])
            for k in norm:
                scores[e][k][it["key"]] = np.round((ch[k] - norm[k][0]) / norm[k][1], 2).tolist()
        print(f"[{e}] σ {np.round(S, 2).tolist()} | best read {info[e]['best']}, best write {info[e]['best_write']} | " +
              ", ".join(f"{r['channel']} {r['held_auc_vs_neutral']:.3f}" for r in ch_rows))
    geom = {"pairs": []}
    for i, a in enumerate(EMO):                                   # every pair of LoReFTs
        for b in EMO[i + 1:]:
            pr = np.linalg.svd(Qs[a] @ Qs[b].T, compute_uv=False)
            pw = np.linalg.svd(Ps[a].T @ Ps[b], compute_uv=False)
            geom["pairs"].append({"a": a, "b": b, "principal_read": pr.tolist(), "principal_write": pw.tolist(),
                                  "cos_read": (Qs[a] @ Qs[b].T).tolist(), "cos_write": (Ps[a].T @ Ps[b]).tolist(),
                                  "dim_cos": float(chat[concepts.index(a)] @ chat[concepts.index(b)])})
            print(f"geometry {a:7} vs {b:7}: principal cosines read {np.round(pr, 2).tolist()}, "
                  f"write {np.round(pw, 2).tolist()} | DiM cos {geom['pairs'][-1]['dim_cos']:+.2f}")

    out = {"layer": L, "corpus": args.corpus, "scheme": args.scheme, "emotions": EMO, "concepts": concepts,
           "groups": groups, "items": items, "baseline": baseline, "info": info, "geometry": geom, "scores": scores,
           "tokens": {it["key"]: [tok.decode([i]) for i in resid[it["key"]][0]] for it in items}}
    json.dump(out, open(args.out + ".json", "w"), ensure_ascii=False)
    open(args.out + ".html", "w").write(render_html(out))
    print(f"Wrote {args.out}.html / .json")


CSS = """
:root{--bg:#fbfaf7;--fg:#171613;--muted:#76746c;--line:#e3e1d8;--head:#f1efe7;--card:#fff;--grid:#d9d6cb;--acc:#1f6fb2;__EL__}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--bg:#141412;--fg:#ecebe5;--muted:#9b998f;--line:#312f2a;--head:#1e1d1a;--card:#1a1917;--grid:#3a3833;--acc:#62a8e6;__ED__}}
:root[data-theme="dark"]{--bg:#141412;--fg:#ecebe5;--muted:#9b998f;--line:#312f2a;--head:#1e1d1a;--card:#1a1917;--grid:#3a3833;--acc:#62a8e6;__ED__}
body{font:14px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif;background:var(--bg);color:var(--fg);max-width:1180px;margin:0 auto;padding:1.6rem 1.1rem 3rem}
h1{font-size:1.4rem;margin:0 0 .3rem} h2{font-size:1.06rem;margin:2rem 0 .5rem} h3{font-size:.95rem;margin:.2rem 0 .5rem}
.lede,.note{color:var(--muted);max-width:960px} .note{font-size:.86rem}
.wrap{overflow-x:auto} table{border-collapse:collapse} th,td{border-bottom:1px solid var(--line);padding:5px 8px;text-align:left;vertical-align:top}
th{background:var(--head);font-size:.78rem;color:var(--muted);font-weight:600} td.num{font-variant-numeric:tabular-nums;text-align:right;white-space:nowrap}
tr.best td{font-weight:700}
.controls{position:sticky;top:0;z-index:3;background:var(--bg);border-bottom:1px solid var(--line);padding:.5rem 0;display:flex;flex-wrap:wrap;gap:.6rem 1.2rem;align-items:center;font-size:.86rem}
.controls label{display:inline-flex;gap:.3rem;align-items:center} select{font:inherit;font-size:.84rem;padding:1px 4px}
.chk{display:inline-flex;flex-wrap:wrap;gap:.2rem .6rem}
.sw{display:inline-block;width:18px;height:3px;border-radius:2px;vertical-align:middle}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:.7rem .9rem .3rem;margin:0 0 1rem}
.lab{font-weight:600;font-size:.74rem;color:var(--muted);text-transform:uppercase;letter-spacing:.05em;margin-top:.4rem}
.box{position:relative} .linetext{position:relative;white-space:normal;line-height:1.55} .flow{line-height:1.55}
.tok{border-radius:3px} svg.spark{display:block;margin:0 0 .25rem;overflow:visible}
tr.single{display:none} body.show-chan tr.single{display:table-row} tr.lda td{font-weight:600}
details{margin:.2rem 0 1rem} summary{cursor:pointer;color:var(--muted);font-size:.82rem;padding:.2rem 0}
details[open] summary{margin-bottom:.4rem}
"""

JS = """
const D = __DATA__;
const EMO = D.emotions, H = 44, PAD = 5, SCALE = 3;
// default: the write-space LDA (best concept meter in our measurements), else read LDA, else best single channel
const DEF = e => ('write LDA' in D.scores[e]) ? 'write LDA' : ('read LDA' in D.scores[e]) ? 'read LDA' : D.info[e].best;
const state = {chan: Object.fromEntries(EMO.map(e => [e, DEF(e)])), smooth: 3, tint: 'argmax',
               shown: new Set(EMO.concat([D.baseline]))};
const cvar = i => getComputedStyle(document.documentElement).getPropertyValue('--e' + i).trim();
function sm(a, w){ if (w <= 1) return a; const h = (w - 1) / 2;
  return a.map((_, i) => { let s = 0, n = 0; for (let j = i - h; j <= i + h; j++) if (j >= 0 && j < a.length){ s += a[j]; n++; } return s / n; }); }
const yMap = v => H / 2 - Math.max(-1, Math.min(1, v / SCALE)) * (H / 2 - PAD);
function svgLine(line, spans, series, width){
  const NS = 'http://www.w3.org/2000/svg', svg = document.createElementNS(NS, 'svg');
  svg.setAttribute('class', 'spark'); svg.setAttribute('width', width); svg.setAttribute('height', H);
  [0, 1, -1].forEach(g => { const l = document.createElementNS(NS, 'line'); const y = yMap(g);
    l.setAttribute('x1', 0); l.setAttribute('x2', width); l.setAttribute('y1', y); l.setAttribute('y2', y);
    l.setAttribute('stroke', 'var(--grid)'); l.setAttribute('stroke-width', g === 0 ? 1 : 0.6);
    if (g) l.setAttribute('stroke-dasharray', '2 3'); svg.appendChild(l); });
  const xs = line.map(i => spans[i].offsetLeft + spans[i].offsetWidth / 2);
  series.forEach((s, k) => { const pl = document.createElementNS(NS, 'polyline');
    pl.setAttribute('points', line.map((i, j) => xs[j].toFixed(1) + ',' + yMap(s.v[i]).toFixed(1)).join(' '));
    pl.setAttribute('fill', 'none'); pl.setAttribute('stroke', cvar(k)); pl.setAttribute('stroke-width', 1.7); svg.appendChild(pl); });
  return svg; }
function tintOf(series, i){ if (state.tint === 'none') return '';
  let k;
  if (state.tint === 'argmax'){                      // whichever emotion scores this token highest
    k = 0; for (let j = 1; j < series.length; j++) if (series[j].v[i] > series[k].v[i]) k = j;
  } else k = EMO.indexOf(state.tint);
  const z = series[k].v[i]; if (z <= 0) return '';   // below neutral on the winner -> no tint
  const a = Math.min(1, z / SCALE) * 0.55, c = cvar(k);
  return `color-mix(in srgb, ${c} ${Math.round(a * 100)}%, transparent)`; }
function para(root, toks, series){
  root.innerHTML = ''; const flow = document.createElement('div'); flow.className = 'flow'; root.appendChild(flow);
  const spans = toks.map((t, i) => { const s = document.createElement('span'); s.className = 'tok'; s.textContent = t;
    s.title = series.map(x => `${x.e} ${state.chan[x.e]}: ${x.raw[i] >= 0 ? '+' : ''}${x.raw[i].toFixed(2)}`).join('\\n');
    s.style.background = tintOf(series, i); flow.appendChild(s); return s; });
  const lines = []; let cur = null, top = null;
  spans.forEach((s, i) => { if (s.offsetTop !== top){ cur = []; lines.push(cur); top = s.offsetTop; } cur.push(i); });
  root.removeChild(flow); const W = root.clientWidth;
  lines.forEach(line => { const lt = document.createElement('div'); lt.className = 'linetext';
    line.forEach(i => lt.appendChild(spans[i])); root.appendChild(lt); root.appendChild(svgLine(line, spans, series, W)); }); }
function render(){
  const box = document.getElementById('traces'); box.innerHTML = '';
  D.groups.forEach(g => {
    const its = D.items.filter(it => it.group === g && state.shown.has(it.concept));
    if (!its.length) return;
    const card = document.createElement('div'); card.className = 'card'; box.appendChild(card);
    const h = document.createElement('h3'); h.textContent = g; card.appendChild(h);
    its.forEach(it => {
      const lab = document.createElement('div'); lab.className = 'lab'; lab.textContent = it.label; card.appendChild(lab);
      const p = document.createElement('div'); p.className = 'box'; card.appendChild(p);
      const series = EMO.map(e => { const raw = D.scores[e][state.chan[e]][it.key]; return {e, raw, v: sm(raw, state.smooth)}; });
      para(p, D.tokens[it.key], series); }); }); }
// controls
const ctl = document.getElementById('ctl');
EMO.forEach((e, k) => { const l = document.createElement('label');
  l.innerHTML = `<span class="sw" style="background:var(--e${k})"></span> ${e}`; const s = document.createElement('select');
  Object.keys(D.scores[e]).forEach(ch => { const o = document.createElement('option'); o.value = ch;
    o.textContent = ch + (ch.endsWith('LDA') ? ' (all 4)' : ch === D.info[e].best || ch === D.info[e].best_write ? ' (best single)' : ''); s.appendChild(o); });
  s.value = state.chan[e]; s.onchange = () => { state.chan[e] = s.value; render(); }; l.appendChild(s); ctl.appendChild(l); });
const sl = document.createElement('label'); sl.textContent = 'smoothing '; const ss = document.createElement('select');
[[1, 'none'], [3, '3 tokens'], [5, '5 tokens']].forEach(([v, n]) => { const o = document.createElement('option'); o.value = v; o.textContent = n; ss.appendChild(o); });
ss.value = state.smooth; ss.onchange = () => { state.smooth = +ss.value; render(); }; sl.appendChild(ss); ctl.appendChild(sl);
const tl = document.createElement('label'); tl.textContent = 'highlight tokens '; const ts = document.createElement('select');
[['argmax', 'max(' + EMO.join(', ') + ')'], ['none', 'none']].concat(EMO.map(e => [e, e + ' only'])).forEach(([v, n]) => {
  const o = document.createElement('option'); o.value = v; o.textContent = n; ts.appendChild(o); });
ts.value = state.tint; ts.onchange = () => { state.tint = ts.value; render(); }; tl.appendChild(ts); ctl.appendChild(tl);
const cl = document.createElement('label'); const ci = document.createElement('input'); ci.type = 'checkbox';
ci.onchange = () => document.body.classList.toggle('show-chan', ci.checked);
cl.appendChild(ci); cl.appendChild(document.createTextNode(' show individual channels')); ctl.appendChild(cl);
const ck = document.createElement('span'); ck.className = 'chk'; ck.innerHTML = '<span class=muted>paragraphs:</span>';
D.concepts.forEach(c => { const l = document.createElement('label'); const i = document.createElement('input'); i.type = 'checkbox';
  i.checked = state.shown.has(c); i.onchange = () => { i.checked ? state.shown.add(c) : state.shown.delete(c); render(); };
  l.appendChild(i); l.appendChild(document.createTextNode(c)); ck.appendChild(l); });
ctl.appendChild(ck);
let rt; window.addEventListener('resize', () => { clearTimeout(rt); rt = setTimeout(render, 150); });
render();
"""


def render_html(o):
    E = html.escape
    EMO, C = o["emotions"], o["concepts"]
    el = ";".join(f"--e{k}:{COLORS[k][0]}" for k in range(len(EMO)))
    ed = ";".join(f"--e{k}:{COLORS[k][1]}" for k in range(len(EMO)))
    css = CSS.replace("__EL__", el).replace("__ED__", ed)
    def zcell(v, pos="var(--acc)", neg="var(--muted)", full=1.5):
        a = min(1, abs(v) / full) * 0.55
        col = f"color-mix(in srgb, {pos if v >= 0 else neg} {round(a * 100)}%, transparent)"
        return f"<td class=num style='background:{col}'>{v:+.2f}</td>"
    tables = ""
    for k, e in enumerate(EMO):
        inf = o["info"][e]; others = [x for x in EMO if x != e]
        neg = f"var(--e{1 - k})" if len(EMO) == 2 else "var(--muted)"
        z = lambda r, c: (zcell(r["mean_z"][c], f"var(--e{k})", neg) if c in r["mean_z"] else "<td class=num>—</td>")
        # --- primary table: how each channel responds to every concept's held-out paragraphs ---
        head1 = "<tr><th>channel</th>" + "".join(f"<th>{E(c)}</th>" for c in C) + "</tr>"
        body1 = "".join(
            f"<tr class='{'lda' if r['channel'].endswith('LDA') else 'single'}"
            f"{' best' if r['channel'] in (inf['best'], inf.get('best_write')) else ''}'>"
            f"<td class=mname>{E(r['channel'])}<span class=muted> · {r.get('kind', 'read')}</span></td>"
            + "".join(z(r, c) for c in C) + "</tr>" for r in inf["channels"])
        # --- diagnostics, collapsed ---
        head2 = ("<tr><th>channel</th><th>kind</th><th>σ</th>" + "".join(f"<th>cos with<br>{E(x)} DiM</th>" for x in EMO)
                 + "<th>AUC train<br>vs neutral</th><th>AUC held-out<br>vs neutral</th>"
                 + "".join(f"<th>AUC held-out<br>vs {E(x)}</th>" for x in others) + "</tr>")
        body2 = "".join(
            f"<tr><td>{E(r['channel'])}</td><td class=muted>{r.get('kind', 'read')}</td>"
            f"<td class=num>{'—' if r['sigma'] is None else format(r['sigma'], '.2f')}</td>"
            + "".join(f"<td class=num>{r['cos_dim'][x]:+.2f}</td>" if r["cos_dim"] else "<td class=num>—</td>" for x in EMO)
            + f"<td class=num>{r['train_auc_vs_neutral']:.3f}</td><td class=num>{r['held_auc_vs_neutral']:.3f}</td>"
            + "".join(f"<td class=num>{r['held_auc_vs_other'][x]:.3f}</td>" for x in others) + "</tr>"
            for r in inf["channels"])
        tables += (f"<h3><span class=sw style='background:var(--e{k})'></span> {E(e)} LoReFT</h3>"
                   f"<div class=note>Mean standardised score on each concept's held-out paragraphs "
                   f"(0 = average <i>{E(o.get('baseline', 'neutral'))}</i> token). Bold = best single channel of its kind.</div>"
                   f"<div class=wrap><table>{head1}{body1}</table></div>"
                   f"<details><summary>diagnostics — σ, cosines with the DiM vectors, AUCs "
                   f"(cos(qᵢ,pᵢ) = {', '.join(f'{x:+.2f}' for x in inf['cos_qp'])}, −1 = pure clamp)</summary>"
                   f"<div class=wrap><table>{head2}{body2}</table></div></details>")
    g = o["geometry"]
    geo = ""
    if g and g.get("pairs"):
        rows_g = "".join(
            f"<tr><td>{E(p['a'])} &middot; {E(p['b'])}</td>"
            + "".join(f"<td class=num>{x:.2f}</td>" for x in p["principal_read"])
            + "".join(f"<td class=num>{x:.2f}</td>" for x in p["principal_write"])
            + f"<td class=num>{p['dim_cos']:+.2f}</td></tr>" for p in g["pairs"])
        r = len(g["pairs"][0]["principal_read"])
        geo = (f"<h2>How the LoReFTs relate to each other</h2>"
               f"<div class=note><b>Principal cosines</b> measure how much two 4-dimensional subspaces share: 1 = a direction in common, "
               f"0 = orthogonal (two random 4-d subspaces in 2560-d sit at ≈0.08). If one emotion were simply the negative of another, "
               f"the subspaces would coincide (principal cosines ≈ 1) while the DiM vectors would have cosine ≈ −1. "
               f"The last column is the cosine between the two concepts' DiM vectors, for reference.</div>"
               f"<div class=wrap><table><tr><th rowspan=2>pair</th><th colspan={r}>read spaces (qᵢ)</th>"
               f"<th colspan={r}>write spaces (pᵢ / span R)</th><th rowspan=2>DiM cos</th></tr>"
               f"<tr>{''.join(f'<th>{i+1}</th>' for i in range(r)) * 2}</tr>{rows_g}</table></div>")
    data = json.dumps({k: o[k] for k in ["emotions", "concepts", "groups", "items", "baseline", "info", "scores", "tokens"]},
                      ensure_ascii=False).replace("<", "\\u003c")   # keep the HTML parser inside script-data state
    sec = o.get("corpus") == "security"
    trained_on = ("turn a vulnerability-free component description into that vulnerability class's write-up"
                  if sec else "turn a tone-free prompt into its emotion's paragraph")
    return f"""<title>{'Security' if sec else 'Emotion'} LoReFT sparklines</title>
<style>{css}</style>
<h1>Reading with LoReFTs — {' vs '.join(EMO)}</h1>
<div class=lede>Qwen3.5-4B, layer {o['layer']}. Each LoReFT (rank 4, edits on {'every prompt token' if o['scheme'] == 'all' else 'the last prompt token'})
was trained to {trained_on}. Here they are used only to <b>read</b>: each channel scores a token
by qᵢ·h (the sensor, from V = W − R = UΣQᵀ) or pᵢ·h (the register, span R). Signs are fixed so each LoReFT's own concept reads positive,
and every channel is standardised on <b>training</b> text (0 = average <i>{o.get('baseline','neutral')}</i> token, 1 = one standard deviation).
All paragraphs below are held out. {'Cards group a concept’s held-out write-ups (topics are unique per concept in this corpus).' if sec else 'Cards group one topic, written in each concept’s style.'}</div>

<h2>Channels</h2>
{tables}
{geo}

<h2>Per-token sparklines <span class=muted>(held-out topics)</span></h2>
<div class=note>Solid line = zero (neutral average); dashed = ±1 sd; traces clip at ±3 sd. Hover a token for its unsmoothed values.
Token highlight: each token is coloured by whichever emotion scores it highest ({' / '.join(f'<b style="color:var(--e{k})">{E(e)}</b>' for k, e in enumerate(EMO))}),
with the shade proportional to that score; tokens below the neutral average on the winner are left uncoloured.</div>
<div class=controls id=ctl></div>
<div id=traces></div>
<script>{JS.replace('__DATA__', data)}</script>"""


if __name__ == "__main__":
    main()
