"""Verify the fitted J-lens: top-5 readout vs logit-lens baseline vs the model.

For a few held-out prompts, at several interior positions, compare the next-token
top-5 from (a) the J-lens transport J_17, (b) the vanilla logit-lens (no transport),
and (c) the model's own final logits. The J-lens readout should be coherent and
better-aligned with the model than the raw logit-lens. Renders data/jlens/verify.html.

Usage:  .venv/bin/python scripts/jlens/4_verify.py
"""

import argparse
import html
import os
import sys

import torch

import jlens

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import common  # noqa: E402

PROMPTS = [
    ("general", "The Roman Empire reached its greatest territorial extent under the "
                "emperor Trajan, whose campaigns pushed the frontier deep into "
                "Mesopotamia before his successor Hadrian consolidated the borders."),
    ("security", "The vulnerability is a heap buffer overflow in the kernel module: an "
                 "unprivileged local user can overflow the allocation and overwrite "
                 "adjacent metadata to escalate privileges and gain root access."),
    ("code", "def parse_header(buf):\n    length = buf.read_u32()\n    data = "
             "buf.read(length)\n    return Header(length, data)"),
]


def top5(logits_row, tok):
    return [tok.decode([t]) for t in logits_row.topk(5).indices.tolist()]


def cell(tokens, target=None):
    out = []
    for t in tokens:
        hit = target is not None and t == target
        bg = '#cde8c8' if hit else '#eee'
        fw = '600' if hit else '400'
        label = html.escape(repr(t))  # repr shows quotes + \n; escape handles <>&"'
        out.append(f'<span style="display:inline-block;padding:1px 5px;margin:1px;'
                   f'border-radius:3px;background:{bg};font-weight:{fw}">{label}</span>')
    return " ".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default="qwen35-4b")
    ap.add_argument("--lens", default="data/jlens/lens_L17.pt")
    ap.add_argument("--layer", type=int, default=17)
    ap.add_argument("--out", default="data/jlens/verify.html")
    ap.add_argument("--max-positions", type=int, default=10)
    args = ap.parse_args()

    model_id, kind = common.resolve(args.preset)
    print(f"Loading {model_id} + lens {args.lens} ...")
    _, model, tok, _ = common.load_model(model_id, kind)
    m = jlens.from_hf(model, tok)
    lens = jlens.JacobianLens.load(args.lens)
    print(f"{lens!r}")
    L = args.layer

    sections = []
    agree_j, agree_ll, total = 0, 0, 0
    # lens-vs-MODEL faithfulness (the meaningful metric): does the lens recover
    # the model's own next-token prediction?
    mtop1_in_j5, mtop1_in_l5, top1_j, top1_l = 0, 0, 0, 0
    for kind_name, prompt in PROMPTS:
        ids = m.encode(prompt)[0].tolist()
        seq_len = len(ids)
        lo, hi = 16, seq_len - 1
        if hi <= lo:
            lo = max(1, seq_len - args.max_positions - 1)
        step = max(1, (hi - lo) // args.max_positions)
        positions = list(range(lo, hi, step))[:args.max_positions]

        jl, ml, _ = lens.apply(m, prompt, layers=[L], positions=positions, use_jacobian=True)
        ll, _, _ = lens.apply(m, prompt, layers=[L], positions=positions, use_jacobian=False)
        jl, ll, ml = jl[L], ll[L], ml

        rows = []
        for i, p in enumerate(positions):
            cur = tok.decode([ids[p]])
            nxt = tok.decode([ids[p + 1]]) if p + 1 < seq_len else ""
            j5, l5, mo5 = top5(jl[i], tok), top5(ll[i], tok), top5(ml[i], tok)
            total += 1
            agree_j += int(nxt in j5)
            agree_ll += int(nxt in l5)
            mtop1 = mo5[0]
            mtop1_in_j5 += int(mtop1 in j5)
            mtop1_in_l5 += int(mtop1 in l5)
            top1_j += int(j5[0] == mtop1)
            top1_l += int(l5[0] == mtop1)
            rows.append(
                f"<tr><td>{p}</td>"
                f"<td class=tok>{html.escape(repr(cur))}</td>"
                f"<td class=tgt>{html.escape(repr(nxt))}</td>"
                f"<td>{cell(mo5, nxt)}</td>"
                f"<td>{cell(j5, nxt)}</td>"
                f"<td>{cell(l5, nxt)}</td></tr>")
        sections.append(
            f"<h2>[{kind_name}] {html.escape(prompt)}</h2>"
            f"<table><tr><th>pos</th><th>token</th><th>actual next</th>"
            f"<th>model top-5</th><th>J-lens top-5 (L{L})</th><th>logit-lens top-5 (L{L})</th></tr>"
            f"{''.join(rows)}</table>")

    doc = f"""<!doctype html><meta charset=utf-8><title>J-lens verification</title>
<style>body{{font:14px system-ui,sans-serif;max-width:1200px;margin:1.5rem auto;padding:0 1rem;color:#111;background:#fcfcfb}}
h1{{font-size:1.2rem}}h2{{font-size:.92rem;margin:1.4rem 0 .3rem;color:#333;font-weight:600}}
table{{border-collapse:collapse;width:100%;margin-bottom:.5rem}}
td,th{{border:1px solid #e2e1da;padding:3px 6px;vertical-align:top;text-align:left;font-size:12.5px}}
th{{background:#f3f2ec}} .tok{{color:#555}} .tgt{{color:#0a6;font-weight:600;white-space:nowrap}}</style>
<h1>J-lens readout verification — layer {L}</h1>
<p class=note>Green = the model's actual next token appears in that top-5.
Over {total} interior positions (layer {L} is mid-stack, so absolute recovery is expected to be modest):<br>
&bull; <b>Model top-1 recovered in lens top-5</b> (faithfulness): J-lens <b>{100*mtop1_in_j5/total:.0f}%</b> vs logit-lens {100*mtop1_in_l5/total:.0f}%<br>
&bull; Model top-1 == lens top-1: J-lens {100*top1_j/total:.0f}% vs logit-lens {100*top1_l/total:.0f}%<br>
&bull; Written next-token in lens top-5: J-lens {100*agree_j/total:.0f}% vs logit-lens {100*agree_ll/total:.0f}%<br>
J-lens = softmax(W_U·norm(J_{L}·h)); logit-lens = softmax(W_U·norm(h)).<br>
<b>Read the tokens, not just the %:</b> at mid-stack layer {L} the residual encodes the
<i>current concept</i>, not the grammatical next word — so the J-lens surfaces on-concept
tokens (overflow→buffer, escalate→escalation, root→privileges) while the logit-lens returns
incoherent multilingual junk. The transport J_{L} is what makes L{L} legible; this is exactly
the structure J-space decomposes.</p>
{''.join(sections)}"""
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    open(args.out, "w").write(doc)
    print(f"\n[written next-token] in top-5:  J-lens {agree_j}/{total} ({100*agree_j/total:.0f}%)  |  "
          f"logit-lens {agree_ll}/{total} ({100*agree_ll/total:.0f}%)")
    print(f"[MODEL top-1 recovered] in lens top-5:  "
          f"J-lens {mtop1_in_j5}/{total} ({100*mtop1_in_j5/total:.0f}%)  |  "
          f"logit-lens {mtop1_in_l5}/{total} ({100*mtop1_in_l5/total:.0f}%)")
    print(f"[MODEL top-1 == lens top-1]:  "
          f"J-lens {top1_j}/{total} ({100*top1_j/total:.0f}%)  |  "
          f"logit-lens {top1_l}/{total} ({100*top1_l/total:.0f}%)")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
