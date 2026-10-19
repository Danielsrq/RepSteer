"""Render the real-CVE-corpus probe results as a single page.

Reads: data/security/cve_probe_L17.json, data/security/cve_dataset.json, the span labels.
Writes: data/security/cve_real_report.html
"""
import argparse, glob, html, json, os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from span_labels import MECHANISM, NARRATIVE
from cve_regions import HOLDOUT, all_regions

CSS = """
:root{--bg:#fbfaf7;--fg:#171613;--muted:#76746c;--line:#e3e1d8;--head:#f1efe7;--card:#fff;
      --good:#1a7f45;--bad:#b3261e;--mid:#8a6d1f}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--bg:#141412;--fg:#ecebe5;
  --muted:#9b998f;--line:#312f2a;--head:#1e1d1a;--card:#1a1917;--good:#5ec687;--bad:#ef8e86;--mid:#d4b45e}}
:root[data-theme="dark"]{--bg:#141412;--fg:#ecebe5;--muted:#9b998f;--line:#312f2a;--head:#1e1d1a;
  --card:#1a1917;--good:#5ec687;--bad:#ef8e86;--mid:#d4b45e}
*{box-sizing:border-box}
body{font:14px/1.62 system-ui,-apple-system,"Segoe UI",sans-serif;background:var(--bg);color:var(--fg);
     max-width:1180px;margin:0 auto;padding:1.6rem 1.1rem 4rem}
h1{font-size:1.45rem;margin:0 0 .25rem} h2{font-size:1.08rem;margin:2.2rem 0 .5rem}
h3{font-size:.95rem;margin:1.4rem 0 .4rem;color:var(--muted);font-weight:600}
p{max-width:78ch} .sub{color:var(--muted);margin:0 0 1.4rem}
.wrap{overflow-x:auto;-webkit-overflow-scrolling:touch}
table{border-collapse:collapse;margin:.5rem 0;font-size:.85rem;min-width:100%}
th,td{border-bottom:1px solid var(--line);padding:5px 9px;text-align:left;white-space:nowrap}
th{background:var(--head);color:var(--muted);font-weight:600;position:sticky;top:0}
td.n{text-align:right;font-variant-numeric:tabular-nums}
td.cls{font-weight:600} tr.sep td{border-top:2px solid var(--line)}
.k{display:inline-block;padding:1px 7px;border-radius:99px;font-size:.72rem;font-weight:600}
.win{background:color-mix(in srgb,var(--good) 18%,transparent);color:var(--good)}
.lose{background:color-mix(in srgb,var(--bad) 15%,transparent);color:var(--bad)}
.mid{background:color-mix(in srgb,var(--mid) 18%,transparent);color:var(--mid)}
.note{background:var(--card);border:1px solid var(--line);border-left:3px solid var(--muted);
      border-radius:0 7px 7px 0;padding:.7rem .95rem;margin:1rem 0;max-width:82ch}
.note b{font-weight:650}
ul{max-width:78ch;padding-left:1.15rem} li{margin:.3rem 0}
code{font-family:ui-monospace,Menlo,monospace;font-size:.88em;background:var(--head);
     padding:1px 5px;border-radius:4px}
.legend{color:var(--muted);font-size:.82rem;margin:.35rem 0 0}
"""


def cell(v):
    if v is None or v != v:
        return '<td class="n">–</td>'
    c = "win" if v >= 0.70 else ("lose" if v < 0.55 else "mid")
    return f'<td class="n"><span class="k {c}">{v:.3f}</span></td>'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, default=17)
    ap.add_argument("--out", default="data/security/cve_real_report.html")
    args = ap.parse_args()
    L = args.layer
    probe = json.load(open(f"data/security/cve_probe_L{L}.json"))
    sf = f"data/security/cve_probe_stream_L{L}.json"
    stream = json.load(open(sf)) if os.path.exists(sf) else None
    ds = json.load(open("data/security/cve_dataset.json"))
    per = all_regions()
    E = html.escape

    B = [f"<h1>Reading vulnerability mechanisms off real CVE write-ups</h1>",
         f"<p class=sub>Qwen3.5-4B, layer {L}. 19 published write-ups, span-labelled and collapsed "
         f"into contiguous regions. nftables + smbghost held out entirely. This replaces the "
         f"synthetic security corpus.</p>"]

    # ---------- corpus ----------
    B.append("<h2>1. The corpus</h2>")
    B.append("<p>Every example is a contiguous run of same-labelled spans from a genuine write-up, "
             "carrying the article's preceding text as context. The synthetic corpus this replaces "
             "was model-written snippets of uniform length and register; a reader fitted on those "
             "never saw text shaped like a real article.</p>")
    labs = [l for l in MECHANISM + NARRATIVE if any(r["label"] == l for v in per.values() for r in v)]
    rows = []
    for a in sorted(per):
        cnt = {}
        for r in per[a]:
            cnt[r["label"]] = cnt.get(r["label"], 0) + 1
        tag = ' <span class="k mid">holdout</span>' if a in HOLDOUT else ""
        rows.append(f"<tr><td class=cls>{E(a)}{tag}</td>" +
                    "".join(f'<td class="n">{cnt.get(l,"")}</td>' for l in labs) +
                    f'<td class="n">{len(per[a])}</td></tr>')
    tot = {l: sum(1 for v in per.values() for r in v if r["label"] == l) for l in labs}
    rows.append('<tr class=sep><td class=cls>total</td>' +
                "".join(f'<td class="n">{tot[l]}</td>' for l in labs) +
                f'<td class="n">{sum(len(v) for v in per.values())}</td></tr>')
    B.append('<div class=wrap><table><tr><th>article</th>' +
             "".join(f"<th>{E(l)}</th>" for l in labs) + "<th>regions</th></tr>" +
             "".join(rows) + "</table></div>")
    B.append(f'<p class=legend>Mechanism classes are rare by nature — roughly 5% of a write-up. '
             f'<code>path_traversal</code> was dropped (no real write-up exhibits it); '
             f'<code>stack_overflow</code>, <code>auth_bypass</code> and <code>priv_esc</code> '
             f'never reached 4 regions across 2 articles and are not modelled.</p>')

    # ---------- probes ----------
    B.append("<h2>2. Reading regions in isolation</h2>")
    B.append("<p>Every direction is fitted on token-level residuals and scored by AUC. "
             "<b>LOAO</b> refits inside each leave-one-article-out fold; <b>HOLD</b> fits on all 17 "
             "training articles and scores nftables + smbghost. The negative pool is varied from "
             "easy (plain <code>background</code> prose) to hard (<code>exploit</code> text, which "
             "is also about memory corruption, and other mechanism classes).</p>")
    B.append('<div class=note><b>Fit the direction inside the fold.</b> Fitting the '
             'difference-of-means once on all training positives and then scoring folds leaks the '
             'held-out article into the direction. For <code>uaf</code> that alone moved the '
             'leave-one-article-out AUC from 0.63 to 0.82. Every number below refits per fold.</div>')
    cols = [("dim", "DiM"), ("lda_full", "LDA (full 2560-d)"), ("rand4", "random rank-4"),
            ("cls_last_read", "LoReFT read (last)"), ("cls_last_write", "LoReFT write (last)"),
            ("cls_all_read", "LoReFT read (all)"), ("cls_all_write", "LoReFT write (all)")]
    for split, title, key in [("loao", "Leave-one-article-out (17 training articles)", "loao"),
                              ("hold", "Held out: nftables + smbghost", "holdout")]:
        B.append(f"<h3>{title}</h3>")
        rows = []
        first = True
        for c, pools in probe.items():
            for pname, v in pools.items():
                dd = v[key]
                if not dd:
                    continue
                rows.append(f'<tr{"" if not first else " class=sep"}><td class=cls>{E(c)}</td>'
                            f'<td>{E(pname)}</td>' +
                            "".join(cell(dd.get(k)) for k, _ in cols) + "</tr>")
                first = False
            first = True
        if not any("<td" in r for r in rows):
            continue
        B.append('<div class=wrap><table><tr><th>class</th><th>negatives</th>' +
                 "".join(f"<th>{E(n)}</th>" for _, n in cols) + "</tr>" + "".join(rows) + "</table></div>")
    B.append('<p class=legend>Green ≥ 0.70, amber 0.55–0.70, red &lt; 0.55. The random rank-4 '
             'control sits at 0.51–0.60 rather than 0.50 — an LDA over any 4 directions picks up '
             'some length and position artefact — so that band, not 0.5, is the floor.</p>')

    if stream:
        B.append("<h2>3. Reading the article as a stream — and why it differs</h2>")
        B.append('<div class=note><b>The numbers above do not describe the viewer.</b> Section 2 '
                 'forwards each region <i>on its own</i>. The token viewer forwards a continuous '
                 '2512-token window of the article, so every layer-17 residual carries the whole '
                 'preceding context. Same text, different distribution — and a direction fitted on '
                 'isolated regions scores at chance on the streaming page. The table below refits '
                 'and rescores everything on residuals captured exactly as the viewer captures '
                 'them (continuous pages, 512-token prefix, page-mean-centred).</div>')
        for title, key in [("Leave-one-article-out (17 training articles)", "loao"),
                           ("Held out: nftables + smbghost", "holdout")]:
            B.append(f"<h3>{title}</h3>")
            rows, first = [], True
            for c, pl in stream.items():
                for pname, v in pl.items():
                    dd = v[key]
                    if not dd:
                        continue
                    rows.append(f'<tr{" class=sep" if first else ""}><td class=cls>{E(c)}</td>'
                                f'<td>{E(pname)}</td>' +
                                "".join(cell(dd.get(k)) for k, _ in cols) + "</tr>")
                    first = False
                first = True
            if rows:
                B.append('<div class=wrap><table><tr><th>class</th><th>negatives</th>' +
                         "".join(f"<th>{E(n)}</th>" for _, n in cols) + "</tr>" +
                         "".join(rows) + "</table></div>")
        B.append('<p class=legend>Verified end-to-end: scoring the rendered viewer\'s own per-token '
                 'output against the nftables span labels gives uaf AUC 0.497 (full-space LDA) — '
                 'matching this table, not section 2.</p>')

    # ---------- findings ----------
    B.append("<h2>4. What this says</h2>")
    B.append("""<ul>
<li><b>Real data fixed the transfer failure.</b> <code>race_condition</code> reads at 0.75 (DiM)
and 0.85 (full LDA) on the held-out articles, against 0.95 when the negatives are plain prose.
The synthetic-corpus readers were at chance on the same text. It is also the class with the most
real data — 19 regions across 4 articles.</li>

<li><b>The rank-4 LoReFT is the wrong instrument for these concepts.</b> Its read and write
channels land between 0.33 and 0.73, mostly inside or below the random rank-4 band, and its read
directions are near-orthogonal to the data direction (cos(q<sub>i</sub>, DiM) ≈ 0.03–0.05). This
holds even though the intervention <i>trains</i> fine — the classify objective drives loss from
0.105 to 0.005. It learns to answer the question without routing the answer through a direction
that is linearly readable off other text.</li>

<li><b>Difference-of-means is measuring "vulnerability-flavoured text", not the mechanism.</b>
Against plain prose <code>uaf</code> DiM reads 0.63; against other mechanism classes it reads
0.455, and on the holdout 0.200 — <i>below</i> chance, meaning other mechanisms score
<i>higher</i> on the "uaf" direction than uaf does. The same inversion hits
<code>heap_overflow</code> (0.247) and <code>oob_access</code> (0.373). A single mean-difference
direction cannot separate mechanisms from each other.</li>

<li><b>The information is there; the rank-4 probe is what is missing.</b> A full-space LDA beats
DiM almost everywhere and survives hard negatives where DiM inverts — <code>uaf</code> 0.771 vs
exploit text on the holdout, <code>command_injection</code> 0.755–0.769 across every pool,
<code>uninit_memory</code> 0.805. So layer 17 does linearly encode the distinction; four
directions chosen by a steering objective are not the four that carry it.</li>

<li><b>Class count is the binding constraint, not article count.</b> The classes that work
(<code>race_condition</code> 19 regions, <code>uninit_memory</code> 7, <code>command_injection</code>
8) are the ones with regions spread over 3–4 articles. <code>oob_access</code> (5 regions) is at
chance on every pool.</li>
</ul>""")
    B.append('<div class=note><b>Caveat on the labels.</b> The spans were labelled by agents '
             'following <code>data/security/LABELLING_GUIDE.md</code>, not by a security expert, '
             'and only spot-checked. nftables and cve_2022_0185 had to be re-fetched and '
             're-labelled after their first HTML→text extraction destroyed paragraph structure — '
             'that defect alone had collapsed nftables into 6 regions, one of them 76 kB.</div>')

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    open(args.out, "w").write(
        f"<title>Reading CVE mechanisms at layer {L}</title>\n<style>{CSS}</style>\n" + "\n".join(B))
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
