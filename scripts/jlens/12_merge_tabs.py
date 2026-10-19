"""Fold several single-page HTMLs (same generator, different layers) into one tabbed page.
Each source page's <style> is shared (identical across the set) and its content goes into a tab.

Usage:  .venv/bin/python scripts/jlens/12_merge_tabs.py --out data/jlens/jspace.html \
          --title "J-space" "layer 17" data/jlens/jspace.html "layer 25" data/jlens/jspace_L25.html
"""

import argparse
import os
import re


def split_html(h):
    """Return (style_block, content_after_style)."""
    m = re.search(r"<style>.*?</style>", h, re.S)
    style = m.group(0) if m else ""
    i = h.find("</style>")
    body = h[i + len("</style>"):] if i >= 0 else h
    return style, body


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--title", default="tabbed")
    ap.add_argument("pairs", nargs="+", help="label1 file1 label2 file2 ...")
    args = ap.parse_args()
    assert len(args.pairs) % 2 == 0, "need label/file pairs"
    items = [(args.pairs[i], args.pairs[i + 1]) for i in range(0, len(args.pairs), 2)]

    # read all BEFORE writing (out may be one of the inputs)
    loaded = [(label, open(f, encoding="utf-8").read()) for label, f in items]
    style = split_html(loaded[0][1])[0]
    tabs, panels = [], []
    for i, (label, h) in enumerate(loaded):
        _, body = split_html(h)
        tabs.append(f'<button class="tab{" active" if i == 0 else ""}" onclick="show({i})">{label}</button>')
        panels.append(f'<div class="panel" id="p{i}"{"" if i == 0 else " hidden"}>{body}</div>')

    doc = f"""<!doctype html><meta charset=utf-8><title>{args.title}</title>
{style}
<style>.tabs{{display:flex;gap:6px;margin:.8rem 0 1rem;position:sticky;top:0;background:#fcfcfb;padding:6px 0;z-index:20}}
.tab{{font:inherit;padding:5px 16px;border:1px solid #cdccc4;background:#efeee8;cursor:pointer;border-radius:6px}}
.tab.active{{background:#2166ac;color:#fff;border-color:#2166ac}}</style>
<div class="tabs">{''.join(tabs)}</div>
{''.join(panels)}
<script>function show(i){{document.querySelectorAll('.panel').forEach((p,k)=>p.hidden=(k!==i));
document.querySelectorAll('.tab').forEach((t,k)=>t.classList.toggle('active',k===i));}}</script>"""
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    open(args.out, "w").write(doc)
    print(f"Wrote {args.out} ({len(loaded)} tabs: {', '.join(l for l, _ in loaded)})")


if __name__ == "__main__":
    main()
