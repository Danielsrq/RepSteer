"""Compare J-space across model sizes in ONE page: occupancy vs relative depth, the
reconstruction curves at each model's peak layer, and a per-layer table.

Usage:  .venv/bin/python scripts/jlens/10_size_compare.py \
          --panels data/jlens/panel_qwen35-4b.json data/jlens/panel_qwen35-2b.json \
          --curves data/jlens/curve_qwen35-4b_L25.json data/jlens/curve_qwen35-2b_L20.json
"""

import argparse
import json
import os

COLORS = ["#2166ac", "#b2182b", "#1a9850", "#762a83"]


def occ_svg(series, W=520, H=300):
    pad = 46
    ys = [p["occupancy"] for s in series for p in s["panel"]]
    ymax = max(ys) * 1.12 + 1e-6

    def sx(r):
        return pad + r * (W - 2 * pad)

    def sy(v):
        return (H - pad) - v / ymax * (H - 2 * pad)
    out = [f'<svg viewBox="0 0 {W} {H}" width="100%" style="max-width:{W}px">']
    for gy in range(0, int(ymax) + 1, max(1, int(ymax) // 5)):
        out.append(f'<line x1="{pad}" y1="{sy(gy):.0f}" x2="{W-pad}" y2="{sy(gy):.0f}" stroke="#eee"/>'
                   f'<text x="{pad-6}" y="{sy(gy)+3:.0f}" font-size="10" fill="#999" text-anchor="end">{gy}</text>')
    for si, s in enumerate(series):
        c = COLORS[si % len(COLORS)]
        pts = " ".join(f"{sx(p['rel_depth']):.1f},{sy(p['occupancy']):.1f}" for p in s["panel"])
        out.append(f'<polyline points="{pts}" fill="none" stroke="{c}" stroke-width="2.5"/>')
        for p in s["panel"]:
            out.append(f'<circle cx="{sx(p["rel_depth"]):.1f}" cy="{sy(p["occupancy"]):.1f}" r="3.5" fill="{c}">'
                       f'<title>{s["preset"]} L{p["layer"]}: occ {p["occupancy"]}</title></circle>')
        out.append(f'<text x="{W-pad-4}" y="{sy(s["panel"][-1]["occupancy"])-7:.0f}" font-size="11" '
                   f'fill="{c}" text-anchor="end">{s["preset"]} (d={s["d_model"]})</text>')
    out.append(f'<text x="{W/2:.0f}" y="{H-10}" font-size="12" fill="#555" text-anchor="middle">relative depth (layer / n_layers)</text>')
    out.append(f'<text x="14" y="{H/2:.0f}" font-size="12" fill="#555" text-anchor="middle" '
               f'transform="rotate(-90 14 {H/2:.0f})">occupancy K</text></svg>')
    return "".join(out)


def recon_svg(curves, W=520, H=300):
    pad = 48
    K = max(len(c["real"]) for c in curves) - 1

    def sx(k):
        return pad + k / K * (W - 2 * pad)

    def sy(v):
        return (H - pad) - v * (H - 2 * pad)
    out = [f'<svg viewBox="0 0 {W} {H}" width="100%" style="max-width:{W}px">']
    for gy in [0, .2, .4, .6, .8, 1.0]:
        out.append(f'<line x1="{pad}" y1="{sy(gy):.0f}" x2="{W-pad}" y2="{sy(gy):.0f}" stroke="#eee"/>'
                   f'<text x="{pad-6}" y="{sy(gy)+3:.0f}" font-size="10" fill="#999" text-anchor="end">{gy:.1f}</text>')
    for ci, c in enumerate(curves):
        col = COLORS[ci % len(COLORS)]
        real, rand, occ = c["real"], c["rand"], c["occupancy"]
        out.append(f'<polyline points="{" ".join(f"{sx(k):.1f},{sy(v):.1f}" for k,v in enumerate(real))}" '
                   f'fill="none" stroke="{col}" stroke-width="2.5"/>')
        out.append(f'<polyline points="{" ".join(f"{sx(k):.1f},{sy(v):.1f}" for k,v in enumerate(rand))}" '
                   f'fill="none" stroke="{col}" stroke-width="1.3" stroke-dasharray="4 3" opacity="0.7"/>')
        out.append(f'<line x1="{sx(occ):.1f}" y1="{sy(real[occ]):.1f}" x2="{sx(occ):.1f}" y2="{H-pad}" '
                   f'stroke="{col}" stroke-width="1" stroke-dasharray="2 2" opacity="0.55"/>')
        out.append(f'<circle cx="{sx(occ):.1f}" cy="{sy(real[occ]):.1f}" r="4" fill="{col}"/>'
                   f'<text x="{sx(occ):.1f}" y="{H-pad+13:.0f}" font-size="10" fill="{col}" text-anchor="middle">K={occ}</text>')
    out.append(f'<text x="{W/2:.0f}" y="{H-10}" font-size="12" fill="#555" text-anchor="middle"># concepts subtracted (k)</text>')
    out.append(f'<text x="14" y="{H/2:.0f}" font-size="12" fill="#555" text-anchor="middle" '
               f'transform="rotate(-90 14 {H/2:.0f})">residual² / ||x||²</text></svg>')
    return "".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--panels", nargs="+", required=True)
    ap.add_argument("--curves", nargs="*", default=None, help="peak-layer recon curve JSONs")
    ap.add_argument("--out", default="data/jlens/size_compare.html")
    args = ap.parse_args()
    series = sorted([json.load(open(p)) for p in args.panels], key=lambda s: -s["d_model"])
    curves = [json.load(open(c)) for c in args.curves] if args.curves else []

    rows = ""
    for s in series:
        for p in s["panel"]:
            rows += (f"<tr><td>{s['preset']}</td><td>{s['d_model']}</td><td>{p['layer']}</td>"
                     f"<td>{p['rel_depth']}</td><td><b>{p['occupancy']}</b></td>"
                     f"<td>{100*p['excess_var']:.1f}%</td><td>{100*p['faithfulness']:.0f}%</td>"
                     f"<td>{p['occupancy']/s['d_model']*1000:.2f}</td></tr>")
    peaks = "; ".join(f"{s['preset']} peak occ={max(p['occupancy'] for p in s['panel'])} (d={s['d_model']})"
                      for s in series)
    # colored legend for the model lines
    mleg = " ".join(f'<span style="color:{COLORS[i]}">&#9644; {s["preset"]} (d={s["d_model"]})</span>'
                    for i, s in enumerate(series))
    recon_block = ""
    if curves:
        cleg = " ".join(f'<span style="color:{COLORS[i]}">&#9644; {c["preset"]} L{c["layer"]} '
                        f'(rel {c["rel_depth"]}), K={c["occupancy"]}</span>' for i, c in enumerate(curves))
        recon_block = f"""<h2>Reconstruction at each model's peak layer</h2>
<div class=leg><b>solid</b> = real J-lens frame &nbsp;·&nbsp; <b>– – dashed</b> = same-size random control
&nbsp;·&nbsp; vertical tick / dot = occupancy K. &nbsp; {cleg}</div>
{recon_svg(curves)}"""

    doc = f"""<!doctype html><meta charset=utf-8><title>J-space occupancy vs model size</title>
<style>body{{font:14px system-ui,sans-serif;max-width:900px;margin:1.5rem auto;padding:0 1rem;background:#fcfcfb;color:#141414}}
h1{{font-size:1.2rem}}h2{{font-size:1rem;margin:1.5rem 0 .3rem}} .note{{color:#555;font-size:.85rem}}
.leg{{color:#555;font-size:.82rem;margin:.3rem 0 .5rem;display:flex;gap:1rem;flex-wrap:wrap;align-items:center}}
.leg span{{white-space:nowrap}}
table{{border-collapse:collapse;margin-top:.6rem;font-size:12.5px}} td,th{{border:1px solid #e2e1da;padding:3px 8px;text-align:right}}
th{{background:#f3f2ec}} td:first-child{{text-align:left}}</style>
<h1>J-space occupancy vs model size</h1>
<div class=note>Occupancy K = concepts before NN-OMP marginal reconstruction gain falls below a same-size
random control, per layer, on an identical prompt set, compared at <b>matched relative depth</b>.
<b>d_model confound:</b> K partly tracks d_model (a bigger residual space and a weaker per-direction
random control both push K up), so read K alongside d_model and the d-normalized K·1000/d column.
Faithfulness marks the workspace boundary (jumps in the motor layers), not J-space quality. {peaks}.</div>
<h2>Occupancy vs relative depth</h2>
<div class=leg>{mleg}</div>
{occ_svg(series)}
{recon_block}
<h2>Per-layer panel</h2>
<table><tr><th>model</th><th>d</th><th>layer</th><th>rel depth</th><th>occupancy</th>
<th>excess var</th><th>faithful</th><th>K·1000/d</th></tr>{rows}</table>"""
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    open(args.out, "w").write(doc)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
