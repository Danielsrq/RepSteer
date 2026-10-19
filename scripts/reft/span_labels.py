"""Segment CVE write-ups into labellable spans and visualise the labels.

Segmentation rules (shared by the labeller and every consumer, so indices are stable):
  - a fenced ``` block is ONE span (never split on the blank lines inside it)
  - a markdown heading (#, ##, ...) is its own span
  - otherwise split on blank lines; for files with no blank lines (some articles are one
    line per paragraph) fall back to splitting on single newlines
Each span carries surface flags (code / heading) separately from its semantic label, so a
code listing that demonstrates an exploit is labelled `exploit` with is_code=true.

Labels live in data/security/span_labels/{article}.json as {"article", "labels": {span_index: label}}.

Usage:  .venv/bin/python scripts/reft/span_labels.py --article dirtypipe --dump      # numbered spans
        .venv/bin/python scripts/reft/span_labels.py --article dirtypipe             # render the HTML
        .venv/bin/python scripts/reft/span_labels.py --all --out data/security/span_labels.html
"""
import argparse, glob, html, json, os, re, sys

# --- label schema -------------------------------------------------------------------------------
# mechanism classes: what kind of bug the span is describing. path_traversal was dropped: no
# real write-up in the corpus exhibits it. The rest match the trained
# LoReFTs / DiM concepts; the rest are needed because real write-ups do not fit that vocabulary.
MECHANISM = ["uaf", "heap_overflow", "stack_overflow", "integer_overflow", "race_condition",
             "oob_access", "command_injection", "auth_bypass", "priv_esc",
             "uninit_memory", "other_vuln"]
# non-mechanism classes: what else write-ups are made of
NARRATIVE = ["background", "debugging", "exploit", "patch", "benign"]
LABELS = MECHANISM + NARRATIVE
COLOR = {"uaf": "#b5179e", "heap_overflow": "#d9730d", "stack_overflow": "#c2410c",
         "integer_overflow": "#a16207", "race_condition": "#7a7500", "oob_access": "#be123c",
         "command_injection": "#2a6fdb", "auth_bypass": "#6a3fb5",
         "priv_esc": "#9d174d", "uninit_memory": "#12897f", "other_vuln": "#57534e",
         "background": "#78716c", "debugging": "#94a3b8", "exploit": "#dc2626",
         "patch": "#15803d", "benign": "#16a34a"}


def segment(text):
    """Split an article into stable, labellable spans."""
    lines = text.split("\n")
    spans, buf, in_code = [], [], False
    def flush(kind="prose"):
        if buf and "".join(buf).strip():
            spans.append({"kind": kind, "text": "\n".join(buf).rstrip()})
        buf.clear()
    for ln in lines:
        if ln.lstrip().startswith("```"):
            if in_code:
                buf.append(ln); flush("code"); in_code = False
            else:
                flush(); buf.append(ln); in_code = True
            continue
        if in_code:
            buf.append(ln); continue
        if not ln.strip():
            flush(); continue
        if re.match(r"^#{1,6}\s", ln):
            flush(); buf.append(ln); flush("heading"); continue
        buf.append(ln)
    flush("code" if in_code else "prose")
    # Some articles have no blank-line paragraph breaks, so the split above yields a few enormous
    # blobs (one was 124 kB). Detect that by span size, not span count -- a couple of code fences can
    # push the count over any threshold while leaving the prose unsplit -- and fall back to one line
    # per span. Correctly-separated articles have a median prose span of ~200-300 chars.
    pros = [len(x["text"]) for x in spans if x["kind"] == "prose"]
    if len(spans) < 5 or (pros and sorted(pros)[len(pros) // 2] > 1000):
        spans = [{"kind": "heading" if re.match(r"^#{1,6}\s", l) else "prose", "text": l.rstrip()}
                 for l in lines if l.strip()]
    for i, s in enumerate(spans):
        s["i"] = i
    return spans


def load_labels(article, d="data/security/span_labels"):
    f = os.path.join(d, f"{article}.json")
    if not os.path.exists(f): return {}
    return {int(k): v for k, v in json.load(open(f))["labels"].items()}


def render(articles, out):
    """articles: list of (name, spans, labels)."""
    E = html.escape
    css = """
:root{--bg:#fbfaf7;--fg:#171613;--muted:#76746c;--line:#e3e1d8;--head:#f1efe7;--card:#fff;--acc:#1f6fb2}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--bg:#141412;--fg:#ecebe5;--muted:#9b998f;--line:#312f2a;--head:#1e1d1a;--card:#1a1917;--acc:#62a8e6}}
body{font:14px/1.6 system-ui,-apple-system,"Segoe UI",sans-serif;background:var(--bg);color:var(--fg);max-width:1100px;margin:0 auto;padding:1.5rem 1.1rem 3rem}
h1{font-size:1.35rem;margin:0 0 .3rem} h2{font-size:1.05rem;margin:1.8rem 0 .5rem}
.note{color:var(--muted);font-size:.86rem;max-width:900px}
.ctl{position:sticky;top:0;z-index:5;background:var(--bg);border-bottom:1px solid var(--line);padding:.5rem 0;display:flex;gap:.5rem 1rem;flex-wrap:wrap;align-items:center;font-size:.85rem}
.legend b{font-weight:600;margin-right:.7rem;white-space:nowrap;cursor:pointer;opacity:.95}
.legend b.off{opacity:.25;text-decoration:line-through} .sw{display:inline-block;width:11px;height:11px;border-radius:3px;vertical-align:middle;margin-right:3px}
.span{border-left:4px solid var(--line);padding:.35rem .6rem;margin:.28rem 0;border-radius:0 6px 6px 0;background:var(--card)}
.span .lab{font-size:.7rem;text-transform:uppercase;letter-spacing:.05em;color:var(--muted);font-weight:700}
.span .idx{color:var(--muted);font-size:.7rem;float:right}
.span pre{margin:.3rem 0 0;white-space:pre-wrap;font-family:ui-monospace,Menlo,monospace;font-size:11.5px;color:var(--muted);max-height:16em;overflow:auto}
.span p{margin:.15rem 0 0} .span.code{background:transparent} .hide{display:none}
table{border-collapse:collapse;margin:.4rem 0} th,td{border-bottom:1px solid var(--line);padding:4px 10px;text-align:left;font-size:.85rem}
th{background:var(--head);color:var(--muted);font-weight:600} td.n{text-align:right;font-variant-numeric:tabular-nums}
"""
    body = ["<h1>CVE write-up span labels</h1>",
            "<div class=note>Each span is one paragraph, heading or fenced code block. "
            "The <b>label</b> says what the span is about; <b>code</b> is a surface flag, so a listing that "
            "demonstrates an exploit is <i>exploit</i> + code. Click a legend entry to hide that label.</div>"]
    legend = " ".join(f'<b data-lab="{l}"><span class=sw style="background:{COLOR[l]}"></span>{l}</b>' for l in LABELS)
    body.append(f'<div class=ctl><span class="legend">{legend}</span>'
                '<label><input type=checkbox id=hidecode> hide code spans</label></div>')
    for name, spans, labels in articles:
        counts, tally = {}, {}
        for s in spans:
            l = labels.get(s["i"], "unlabelled")
            counts[l] = counts.get(l, 0) + 1
            tally[l] = tally.get(l, 0) + len(s["text"])
        rows = "".join(f"<tr><td><span class=sw style='background:{COLOR.get(l,'#999')}'></span>{E(l)}</td>"
                       f"<td class=n>{c}</td><td class=n>{tally[l]}</td></tr>"
                       for l, c in sorted(counts.items(), key=lambda kv: -kv[1]))
        body.append(f"<h2>{E(name)} <span class=note>({len(spans)} spans)</span></h2>")
        body.append(f"<table><tr><th>label</th><th>spans</th><th>chars</th></tr>{rows}</table>")
        for s in spans:
            l = labels.get(s["i"], "unlabelled"); col = COLOR.get(l, "#999")
            txt = s["text"]
            inner = (f"<pre>{E(txt[:4000])}</pre>" if s["kind"] == "code"
                     else f"<p>{'<b>' if s['kind']=='heading' else ''}{E(txt)}{'</b>' if s['kind']=='heading' else ''}</p>")
            body.append(f'<div class="span {s["kind"]}" data-lab="{l}" style="border-left-color:{col}">'
                        f'<span class=idx>#{s["i"]} · {s["kind"]}</span>'
                        f'<span class=lab style="color:{col}">{E(l)}</span>{inner}</div>')
    js = """
const off = new Set();
document.querySelectorAll('.legend b').forEach(b => b.onclick = () => {
  const l = b.dataset.lab; off.has(l) ? off.delete(l) : off.add(l); b.classList.toggle('off', off.has(l)); apply(); });
const hc = document.getElementById('hidecode');
hc.onchange = apply;
function apply(){ document.querySelectorAll('.span').forEach(s => {
  s.classList.toggle('hide', off.has(s.dataset.lab) || (hc.checked && s.classList.contains('code'))); }); }
"""
    open(out, "w").write(f"<title>CVE span labels</title>\n<style>{css}</style>\n" + "\n".join(body) + f"\n<script>{js}</script>")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--article", default="dirtypipe")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--dump", action="store_true", help="print numbered spans (for labelling)")
    ap.add_argument("--width", type=int, default=110, help="chars of each span to show in --dump")
    ap.add_argument("--labels", default="data/security/span_labels")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    names = ([os.path.basename(f)[:-4] for f in sorted(glob.glob("data/jlens/cve/*.txt"))]
             if args.all else [args.article])
    if args.all:                      # --all shows only articles that have been labelled
        names = [n for n in names if load_labels(n, args.labels)]
    arts = []
    for n in names:
        spans = segment(open(f"data/jlens/cve/{n}.txt", encoding="utf-8").read())
        labels = load_labels(n, args.labels)
        arts.append((n, spans, labels))
        if args.dump:
            for s in spans:
                first = " ".join(s["text"].split())[:args.width]
                print(f"{s['i']:>3} [{s['kind'][:4]}] {labels.get(s['i'],'-'):<16} {first}")
            print(f"--- {n}: {len(spans)} spans, {len(labels)} labelled")
    if not args.dump:
        out = args.out or f"data/security/span_labels_{names[0]}.html"
        os.makedirs(os.path.dirname(out), exist_ok=True)
        render(arts, out)
        print(f"Wrote {out}")


if __name__ == "__main__":
    main()
