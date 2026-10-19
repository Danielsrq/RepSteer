"""The 5-label scheme: background / triage / overflow / uaf / race.

Why this replaces the 17-label scheme: with a 4B model we are not going to read a fine bug
taxonomy off the residual stream -- the previous study measured uaf-vs-other-mechanisms at 0.244
AUC (below chance) while the narrative roles read at 0.64-0.70. So we keep the narrative split,
which the model demonstrably represents, and collapse the mechanism side to the three concepts
that have enough real articles to train and hold out:

    overflow  8 articles   heap/integer/oob -- the same story at different points in the chain
    uaf       3 articles
    race      2 articles   (the positive control: it read at 0.755 full-space on held-out text)

Each article has exactly ONE security concept. Everything else is background or triage.

Labels live in data/security/span_labels_v2/{article}.json, same format as v1, and the span
indices come from the SAME segmenter (span_labels.segment), so the two schemes stay comparable.

Usage:
  .venv/bin/python scripts/reft/span_labels_v2.py --article looney_tunables          # render one
  .venv/bin/python scripts/reft/span_labels_v2.py --all --out data/security/labels_v2.html
  .venv/bin/python scripts/reft/span_labels_v2.py --article X --dump --width 300     # for labelling
"""
import argparse, glob, html, json, os, re, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from span_labels import segment

CONCEPTS = ["overflow", "uaf", "race"]
NARRATIVE = ["background", "triage"]
LABELS = CONCEPTS + NARRATIVE
COLOR = {"overflow": "#d9730d", "uaf": "#b5179e", "race": "#7a7500",
         "background": "#78716c", "triage": "#2a6fdb"}
DIR = "data/security/span_labels_v2"

# Which concept each article carries, from the audit in data/security/concept_audit.json
# (agent audit cross-checked against the v1 span labels; 17/19 agreed).
ARTICLE_CONCEPT = {
    "cve_2022_0185": "overflow", "ghostscript_28879": "overflow", "looney_tunables": "overflow",
    "netfilter_22555": "overflow", "pwnkit": "overflow", "sequoia": "overflow",
    "smbghost": "overflow", "win32k_1732": "overflow",
    "ios_voucher": "uaf", "nftables": "uaf", "shiftfs": "uaf",
    "dirtycow": "race", "regresshion": "race",
}
HOLDOUT = {"smbghost", "win32k_1732", "nftables", "regresshion"}


def load(article, d=DIR):
    f = os.path.join(d, f"{article}.json")
    if not os.path.exists(f):
        return {}
    return {int(k): v for k, v in json.load(open(f))["labels"].items()}


def labeller(article, d=DIR):
    f = os.path.join(d, f"{article}.json")
    return json.load(open(f)).get("labeller", "?") if os.path.exists(f) else "?" 


def regions(article, d=DIR):
    """Contiguous same-label runs, the unit we actually train on."""
    spans, lab = segment(open(f"data/jlens/cve/{article}.txt", encoding="utf-8").read()), load(article, d)
    out, cur = [], None
    for s in spans:
        l = lab.get(s["i"], "unlabelled")
        if cur and cur["label"] == l:
            cur["end"] = s["i"]; cur["text"] += "\n\n" + s["text"]
        else:
            if cur:
                out.append(cur)
            cur = {"article": article, "label": l, "start": s["i"], "end": s["i"], "text": s["text"]}
    if cur:
        out.append(cur)
    return out


CSS = """
:root{--bg:#fbfaf7;--fg:#171613;--muted:#76746c;--line:#e3e1d8;--head:#f1efe7;--card:#fff}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--bg:#141412;--fg:#ecebe5;
  --muted:#9b998f;--line:#312f2a;--head:#1e1d1a;--card:#1a1917}}
:root[data-theme="dark"]{--bg:#141412;--fg:#ecebe5;--muted:#9b998f;--line:#312f2a;--head:#1e1d1a;--card:#1a1917}
*{box-sizing:border-box}
body{font:14px/1.6 system-ui,-apple-system,"Segoe UI",sans-serif;background:var(--bg);color:var(--fg);
     max-width:1100px;margin:0 auto;padding:1.5rem 1.1rem 4rem}
h1{font-size:1.35rem;margin:0 0 .3rem} h2{font-size:1.02rem;margin:1.8rem 0 .4rem}
p.note{color:var(--muted);max-width:80ch;font-size:.88rem}
.ctl{position:sticky;top:0;z-index:5;background:var(--bg);border-bottom:1px solid var(--line);
     padding:.55rem 0;display:flex;gap:.5rem 1.1rem;flex-wrap:wrap;align-items:center;font-size:.85rem}
.ctl b{font-weight:600;cursor:pointer;white-space:nowrap;user-select:none}
.ctl b.off{opacity:.3;text-decoration:line-through}
.sw{display:inline-block;width:11px;height:11px;border-radius:3px;vertical-align:middle;margin-right:4px}
.art{white-space:pre-wrap;font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:12px;
     line-height:1.65;background:var(--card);border:1px solid var(--line);border-radius:10px;
     padding:1rem 1.1rem;margin:.5rem 0;overflow-x:auto}
.art span{border-radius:3px;padding:1px 0}
.art span.dim{background:transparent!important;opacity:.32}
table{border-collapse:collapse;margin:.4rem 0;font-size:.85rem}
th,td{border-bottom:1px solid var(--line);padding:4px 10px;text-align:left}
th{background:var(--head);color:var(--muted);font-weight:600}
td.n{text-align:right;font-variant-numeric:tabular-nums}
.tag{font-size:.72rem;padding:1px 7px;border-radius:99px;background:var(--head);color:var(--muted)}
.tabbar{display:flex;gap:.4rem;flex-wrap:wrap;border-bottom:1px solid var(--line);margin:.8rem 0 0;
        position:sticky;top:0;z-index:8;background:var(--bg);padding-top:.3rem}
.tabbar button{font:inherit;font-size:.88rem;padding:.45rem .85rem;border:1px solid var(--line);
        border-bottom:none;border-radius:8px 8px 0 0;background:var(--head);color:var(--muted);
        cursor:pointer;margin-bottom:-1px}
.tabbar button.on{background:var(--card);color:var(--fg);font-weight:600}
.tabbar button.on .tag{background:var(--bg)}
.pane{display:none} .pane.on{display:block}
.tabbar ~ .ctl{top:2.9rem;z-index:5}
"""


def render(arts, out, title="CVE span labels (5-label scheme)"):
    E = html.escape
    B = [f"<h1>{E(title)}</h1>",
         "<p class=note>★ = hand-labelled exemplar. Five labels. Each article carries exactly one security concept; everything "
         "else is <b>background</b> (preamble: what the target is, versions, history, references) or "
         "<b>triage</b> (what was tried: fuzzing, debugging, patch diffing, and the exploitation "
         "work). Click a label to dim it. Hover any highlighted run for its span indices.</p>"]
    leg = " ".join(f'<b data-l="{l}"><span class=sw style="background:{COLOR[l]}"></span>{l}</b>'
                   for l in LABELS)
    if len(arts) > 1:
        B.append('<div class=tabbar>' + "".join(
            f'<button data-i="{i}"{" class=on" if i == 0 else ""}>'
            f'{"★ " if "claude" in labeller(n) else ""}{E(n)}'
            f' <span class=tag>{E(ARTICLE_CONCEPT.get(n, "?"))}</span>'
            f'{" <span class=tag>hold</span>" if n in HOLDOUT else ""}</button>'
            for i, (n, _, _) in enumerate(arts)) + "</div>")
    B.append(f'<div class=ctl>{leg}<label><input type=checkbox id=only> concept only</label></div>')
    for pi, (name, spans, lab) in enumerate(arts):
        if len(arts) > 1:
            B.append(f'<div class="pane{" on" if pi == 0 else ""}" data-i="{pi}">')
        cnt, ch = {}, {}
        for s in spans:
            l = lab.get(s["i"], "unlabelled")
            cnt[l] = cnt.get(l, 0) + 1; ch[l] = ch.get(l, 0) + len(s["text"])
        tot = sum(ch.values()) or 1
        rows = "".join(
            f'<tr><td><span class=sw style="background:{COLOR.get(l,"#999")}"></span>{E(l)}</td>'
            f'<td class=n>{c}</td><td class=n>{ch[l]}</td><td class=n>{100*ch[l]/tot:.0f}%</td></tr>'
            for l, c in sorted(cnt.items(), key=lambda kv: -ch[kv[0]]))
        tag = ' <span class=tag>holdout</span>' if name in HOLDOUT else ""
        B.append(f"<h2>{E(name)}{tag} <span class=note>· concept: "
                 f"{E(ARTICLE_CONCEPT.get(name,'?'))} · {len(spans)} spans · labelled by "
                 f"{E(labeller(name))}</span></h2>")
        B.append("<table><tr><th>label</th><th>spans</th><th>chars</th><th>share</th></tr>"
                 + rows + "</table>")
        body = []
        for s in spans:
            l = lab.get(s["i"], "unlabelled")
            c = COLOR.get(l, "#999")
            a = 0.30 if l in CONCEPTS else (0.14 if l == "triage" else 0.07)
            body.append(
                f'<span data-l="{l}" title="#{s["i"]} · {E(l)}" '
                f'style="background:color-mix(in srgb,{c} {int(a*100)}%,transparent)">'
                f'{E(s["text"])}</span>')
        B.append('<div class=art>' + "\n\n".join(body) + "</div>")
        if len(arts) > 1:
            B.append("</div>")
    js = """
const off=new Set(), only=document.getElementById('only');
document.querySelectorAll('.ctl b').forEach(b=>b.onclick=()=>{
  const l=b.dataset.l; off.has(l)?off.delete(l):off.add(l); b.classList.toggle('off',off.has(l)); ap();});
only.onchange=ap;
const CONCEPTS=%s;
function ap(){document.querySelectorAll('.art span').forEach(s=>{
  const l=s.dataset.l;
  s.classList.toggle('dim', off.has(l) || (only.checked && CONCEPTS.indexOf(l)<0));});}
ap();
const tabs=[...document.querySelectorAll('.tabbar button')],
      panes=[...document.querySelectorAll('.pane')];
tabs.forEach(b=>b.onclick=()=>{
  const i=+b.dataset.i;
  tabs.forEach(t=>t.classList.toggle('on', +t.dataset.i===i));
  panes.forEach(p=>p.classList.toggle('on', +p.dataset.i===i));
  window.scrollTo(0,0);});
""" % json.dumps(CONCEPTS)
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    open(out, "w").write(f"<title>{E(title)}</title>\n<style>{CSS}</style>\n"
                         + "\n".join(B) + f"\n<script>{js}</script>")
    print(f"Wrote {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--article", default="looney_tunables")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--dump", action="store_true")
    ap.add_argument("--width", type=int, default=110)
    ap.add_argument("--labels", default=DIR)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    names = ([os.path.basename(f)[:-5] for f in glob.glob(f"{a.labels}/*.json")]
             if a.all else [a.article])
    if a.all:   # group the review by concept, holdouts last within each group
        names.sort(key=lambda n: (CONCEPTS.index(ARTICLE_CONCEPT.get(n, "overflow")),
                                  n in HOLDOUT, n))
    arts = []
    for n in names:
        spans = segment(open(f"data/jlens/cve/{n}.txt", encoding="utf-8").read())
        lab = load(n, a.labels)
        arts.append((n, spans, lab))
        if a.dump:
            for s in spans:
                print(f"{s['i']:>3} [{s['kind'][:4]}] {lab.get(s['i'],'-'):<12} "
                      f"{' '.join(s['text'].split())[:a.width]}")
            print(f"--- {n}: {len(spans)} spans, {len(lab)} labelled")
    if not a.dump:
        render(arts, a.out or f"data/security/labels_v2_{names[0]}.html")


if __name__ == "__main__":
    main()
