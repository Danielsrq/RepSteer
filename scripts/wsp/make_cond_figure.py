"""Render the conditional-offset results: 4 clouds, d' matrix, and the model's actual replies.

Deliberately NOT t-SNE/PCA -- those are unsupervised and would mostly show nuisance variance.
The scatter uses two MEANINGFUL axes (d_animal, v_mode) and is paired with the lossless 4x4 d'
matrix so the geometry claim does not rest on a 2-D projection.

  .venv/bin/python scripts/wsp/make_cond_figure.py
"""
import html, json, os

OUT = "data/wsp/cond_figure.html"
P = json.load(open("data/wsp/cond_offset.json"))
VA = json.load(open("data/wsp/attn_vab.json"))
E = html.escape

names = P["dprime_matrix"]["names"]
mat = P["dprime_matrix"]["matrix"]
proj = P["projection"]
COL = {"b  cond/animal": "#c2410c", "a  cond/other": "#1f6fb2",
       "b' always/animal": "#7c3aed", "a' always/other": "#0f766e"}

# ---- scatter geometry ----
xs = [proj[n]["x"] for n in names]; ys = [proj[n]["y"] for n in names]
sxs = [proj[n]["sx"] for n in names]; sys_ = [proj[n]["sy"] for n in names]
x0, x1 = min(x - s for x, s in zip(xs, sxs)) - 2, max(x + s for x, s in zip(xs, sxs)) + 2
y0, y1 = min(y - s for y, s in zip(ys, sys_)) - 2, max(y + s for y, s in zip(ys, sys_)) + 2
W, Hh, PAD = 560, 420, 54


def sx(v):
    return PAD + (v - x0) / (x1 - x0) * (W - 2 * PAD)


def sy(v):
    return Hh - PAD - (v - y0) / (y1 - y0) * (Hh - 2 * PAD)


pts = []
for n in names:
    p = proj[n]
    pts.append(f'<ellipse cx="{sx(p["x"]):.1f}" cy="{sy(p["y"]):.1f}" '
               f'rx="{abs(sx(p["x"]+p["sx"])-sx(p["x"])):.1f}" '
               f'ry="{abs(sy(p["y"]+p["sy"])-sy(p["y"])):.1f}" fill="{COL[n]}" '
               f'fill-opacity=".16" stroke="{COL[n]}" stroke-opacity=".5"/>')
    pts.append(f'<circle cx="{sx(p["x"]):.1f}" cy="{sy(p["y"]):.1f}" r="5" fill="{COL[n]}"/>')
    pts.append(f'<text x="{sx(p["x"])+9:.1f}" y="{sy(p["y"])-7:.1f}" font-size="12" '
               f'font-weight="600" fill="{COL[n]}">{E(n.split()[0])}</text>')
# b <-> b' connector (the un-fitted contrast)
bp, bpp = proj["b  cond/animal"], proj["b' always/animal"]
pts.append(f'<line x1="{sx(bp["x"]):.1f}" y1="{sy(bp["y"]):.1f}" x2="{sx(bpp["x"]):.1f}" '
           f'y2="{sy(bpp["y"]):.1f}" stroke="#111" stroke-width="2" stroke-dasharray="5 3"/>')
pts.append(f'<text x="{(sx(bp["x"])+sx(bpp["x"]))/2-16:.1f}" y="{(sy(bp["y"])+sy(bpp["y"]))/2-8:.1f}" '
           f'font-size="12" font-weight="700" fill="#111">d\'={mat[0][2]:.2f}</text>')
scatter = (f'<svg viewBox="0 0 {W} {Hh}" width="100%" style="max-width:{W}px">'
           f'<line x1="{PAD}" y1="{sy(0):.1f}" x2="{W-PAD}" y2="{sy(0):.1f}" stroke="#ccc"/>'
           f'<line x1="{sx(0):.1f}" y1="{PAD}" x2="{sx(0):.1f}" y2="{Hh-PAD}" stroke="#ccc"/>'
           + "".join(pts) +
           f'<text x="{W/2:.0f}" y="{Hh-14}" font-size="12" text-anchor="middle" fill="#555">'
           f'x = d_animal  (topic)</text>'
           f'<text x="14" y="{Hh/2:.0f}" font-size="12" fill="#555" '
           f'transform="rotate(-90 14 {Hh/2:.0f})">y = v_mode  (emoji vs prose)</text></svg>')

# ---- d' matrix ----
mx = max(max(r) for r in mat)
rows = ["<tr><th></th>" + "".join(f"<th>{E(n)}</th>" for n in names) + "</tr>"]
for i, n in enumerate(names):
    tds = []
    for j, v in enumerate(mat[i]):
        if i == j:
            tds.append('<td class="diag">—</td>')
        else:
            a = v / mx
            tds.append(f'<td style="background:rgba(194,65,12,{a*.75:.2f})">{v:.2f}</td>')
    rows.append(f"<tr><th class=rh>{E(n)}</th>" + "".join(tds) + "</tr>")
dmat = "<table class=mat>" + "".join(rows) + "</table>"


def rtable(ra, ro, n=4):
    r = ["<table class=resp><tr><th>prompt</th><th>reply</th></tr>"]
    for x in ra[:n]:
        r.append(f"<tr class=an><td>{E(x['p'])}</td><td class=o>{E(x['t'][:92])}</td></tr>")
    for x in ro[:n]:
        r.append(f"<tr class=ot><td>{E(x['p'])}</td><td class=o>{E(x['t'][:92])}</td></tr>")
    return "".join(r) + "</table>"


ref = P["ref"]
refblocks = ""
for k, v in ref.items():
    refblocks += (f"<h3>{E(k)} <span class=sc>animal {100*v['animal']:.0f}% · other "
                  f"{100*v['other']:.0f}% · gap {100*(v['animal']-v['other']):.0f}</span></h3>"
                  + rtable(v["ra"], v["ro"]))
swrows = "".join(
    f"<tr><td>{r['alpha']:.2f}</td><td>{100*r['animal']:.1f}%</td><td>{100*r['other']:.1f}%</td>"
    f"<td><b>{100*r['gap']:+.1f}</b></td><td class=o>{E(r['ra'][0]['t'][:40])}</td>"
    f"<td class=o>{E(r['ro'][0]['t'][:40])}</td></tr>" for r in P["sweep"])
best = max(P["sweep"], key=lambda r: r["gap"])
vab = "".join(
    f"<tr><td>{r['write']}</td><td>{r['c']:.1f}</td><td>{100*r['animal']:.1f}%</td>"
    f"<td>{r['uniq']:.1f}</td><td>{100*r['other']:.1f}%</td><td><b>{100*r['gap']:.1f}</b></td></tr>"
    for r in VA["rows"])

doc = f"""<title>Is the conditional represented?</title>
<style>
:root{{--bg:#fbfaf7;--fg:#171613;--mut:#6f6d65;--line:#e3e1d8;--card:#fff}}
@media(prefers-color-scheme:dark){{:root:not([data-theme=light]){{--bg:#141412;--fg:#ecebe5;--mut:#9b998f;--line:#312f2a;--card:#1a1917}}}}
*{{box-sizing:border-box}}body{{font:15px/1.6 system-ui,sans-serif;background:var(--bg);color:var(--fg);max-width:1080px;margin:0 auto;padding:1.5rem 1.1rem 4rem}}
h1{{font-size:1.35rem;margin:0 0 .3rem}}h2{{font-size:1.1rem;margin:2rem 0 .5rem;border-bottom:1px solid var(--line);padding-bottom:.3rem}}
h3{{font-size:.95rem;margin:1.1rem 0 .3rem}}.sc{{font-weight:400;color:var(--mut);font-size:.85rem}}
p.n{{color:var(--mut);max-width:80ch;font-size:.9rem}}
.two{{display:flex;gap:1.4rem;flex-wrap:wrap;align-items:flex-start}}.two>div{{flex:1;min-width:330px}}
table{{border-collapse:collapse;width:100%;font-size:.85rem;margin:.4rem 0}}
th,td{{border:1px solid var(--line);padding:4px 7px;text-align:left}}
th{{background:var(--card);color:var(--mut);font-weight:600;font-size:.78rem}}
.mat td{{text-align:center;font-variant-numeric:tabular-nums}}.mat .diag{{color:var(--mut)}}
.mat .rh{{text-align:right;white-space:nowrap}}
.resp td{{vertical-align:top}}.resp .o{{font-family:ui-monospace,Menlo,monospace;font-size:13px}}
.resp tr.an td:first-child{{border-left:3px solid #c2410c}}
.resp tr.ot td:first-child{{border-left:3px solid #1f6fb2}}
.box{{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:.7rem 1rem;margin:.6rem 0}}
.warn{{border-left:3px solid #b45309}}
</style>
<h1>Is the conditional itself represented — and can it be written?</h1>
<p class=n>Qwen3.5-4B, layer 27, generation position. <b>Conditional</b> system prompt =
"reply only in emojis if the topic is about animals, otherwise reply normally".</p>

<h2>1. Four clouds — all separable</h2>
<div class=two>
<div>{scatter}
<p class=n><b>Read the dashed line.</b> <code>b</code> and <code>b'</code> are both
animal-topic AND both produce emoji — they differ <i>only</i> in whether the rule had to be
evaluated. They separate at <b>d'={mat[0][2]:.2f}</b>.</p>
<div class="box warn"><b>Caveat:</b> the y-axis is defined as <code>a' − a</code>, so
<code>a</code> and <code>a'</code> separating vertically is <i>by construction</i>, not
evidence. Only the <code>b</code>↔<code>b'</code> split is un-fitted.</div></div>
<div><b>Pairwise d′ (lossless)</b>{dmat}
<p class=n>Minimum off-diagonal d′ = {min(v for r in mat for v in r if v>0):.2f} — every pair
is separable. Note <code>b</code> sits on the <i>prose</i> side of the emoji axis (y={proj['b  cond/animal']['y']:.1f})
despite emitting emoji, and has 5× the spread of <code>b'</code> (±{proj['b  cond/animal']['sy']:.1f}
vs ±{proj["b' always/animal"]['sy']:.1f}) — consistent with a rule being evaluated per prompt
rather than a uniform mode.</p></div></div>

<h2>2. What the model actually says (prompts present)</h2>
<p class=n><span style="color:#c2410c">■</span> animal &nbsp; <span style="color:#1f6fb2">■</span> non-animal</p>
{refblocks}

<h2>3. Causal test — write the rule, remove the prompt</h2>
<p class=n><code>v_rule = mean[resid(SYS_COND+Q) − resid(Q)]</code>, added at L27 with
<b>no system prompt</b>. Target to beat: the conditional prompt's own gap of
<b>{100*(ref['conditional sysprompt']['animal']-ref['conditional sysprompt']['other']):.0f}</b>.</p>
<table><tr><th>α</th><th>animal%</th><th>other%</th><th>gap</th><th>animal sample</th><th>other sample</th></tr>{swrows}</table>
<div class="box warn"><b>It largely fails.</b> Best gap <b>{100*best['gap']:.1f}</b> at α={best['alpha']}
versus <b>70</b> for the prompt itself, and the outputs degrade into
<code>📳📳📳</code> and <code>Replyงงงง</code>. The rule is <i>readable</i> (d′ 5.73) but not
transplantable as a constant offset — see note below.</div>

<h2>4. Bootstrap: dropping the unembedding crutch</h2>
<p class=n>Attention V→O channel, write direction from measured activations
(<code>v_ab</code>, topic held constant) instead of <code>W_unembed[9008]</code>.
<b>cos(v_ab, v_emoji) = {VA['cos_vab_vemoji']:+.3f}</b> — related but far from identical.</p>
<table><tr><th>write</th><th>c</th><th>animal%</th><th>uniq</th><th>other%</th><th>gap</th></tr>{vab}</table>
<p class=n><code>v_ab</code> reaches a comparable gap with 0.0% leakage, at 2× the strength and
lower variety. The structural dependence on the unembedding <b>can be removed</b>.</p>

<h2>Why the offset fails, in one line</h2>
<div class=box><code>v_rule</code> is averaged over <b>both</b> animal and non-animal prompts, so
the part that <i>differs</i> between them — precisely the conditional — cancels. A constant
vector can only deliver the mean effect of the rule, and the rule's whole function is to
<i>modulate</i> by input. <b>Readable ≠ writable</b> for conditionals.</div>
"""
os.makedirs("data/wsp", exist_ok=True)
open(OUT, "w", encoding="utf-8").write(doc)
print(f"wrote {OUT} ({os.path.getsize(OUT)/1024:.0f} KB)")
