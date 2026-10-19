"""Quantify degenerate (repetitive) generation across the steering sweep, and test
whether denoising reduces it.

Reads the steered generations embedded in data/concepts/concept_tokens.html, then
for each output computes repetition metrics (no semantics, no regex):
  rep4   = 1 - distinct_4grams / total_4grams   (0 = all unique, ->1 = very repetitive)
  maxrun = highest repeat-count of any single 4-gram
  d1     = distinct unigrams / total            (low = repetitive)
A generation is flagged 'degenerate' if rep4 > --thresh.

Reports raw-vs-denoised overall, by magnitude, and PAIRED (same prompt/concept/sign/mag).

Usage:  python scripts/14_degeneration.py
"""

import argparse
import json
from collections import Counter


def metrics(text):
    w = text.split()
    if len(w) < 4:
        return {"rep4": 0.0, "maxrun": 1, "d1": 1.0, "n": len(w)}
    g = list(zip(w, w[1:], w[2:], w[3:]))
    c = Counter(g)
    return {"rep4": 1 - len(c) / len(g), "maxrun": max(c.values()),
            "d1": len(set(w)) / len(w), "n": len(w)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--html", default="data/concepts/concept_tokens.html")
    ap.add_argument("--thresh", type=float, default=0.5)
    args = ap.parse_args()

    t = open(args.html).read()
    i = t.index("STEER=") + len("STEER=")
    j = t.index("], STEER_MAGS=") + 1
    steer = json.loads(t[i:j])

    rows = []
    for pi, p in enumerate(steer):
        m = metrics("".join(p["baseline"]["tokens"]))
        rows.append({"prompt": pi, "basis": "baseline", "concept": "-", "key": "0", **m})
        for basis in ("raw", "denoised"):
            for concept, keys in p[basis].items():
                for key, card in keys.items():
                    m = metrics("".join(card["tokens"]))
                    rows.append({"prompt": pi, "basis": basis, "concept": concept, "key": key, **m})

    def summ(rs):
        n = len(rs)
        if n == 0:
            return "  (none)"
        rep = sum(r["rep4"] for r in rs) / n
        deg = sum(r["rep4"] > args.thresh for r in rs) / n
        return f"n={n:4d}  mean rep4={rep:.3f}  %degenerate={100*deg:5.1f}%"

    base = [r for r in rows if r["basis"] == "baseline"]
    raw = [r for r in rows if r["basis"] == "raw"]
    den = [r for r in rows if r["basis"] == "denoised"]
    print("=== overall ===")
    print("baseline :", summ(base))
    print("raw      :", summ(raw))
    print("denoised :", summ(den))

    print("\n=== by |alpha| magnitude (raw vs denoised) ===")
    print(f"{'mag':>5} | {'raw mean rep4':>14} {'raw %deg':>9} | {'den mean rep4':>14} {'den %deg':>9}")
    mags = sorted({r["key"].lstrip("+-") for r in raw})
    for mg in mags:
        r = [x for x in raw if x["key"].lstrip("+-") == mg]
        d = [x for x in den if x["key"].lstrip("+-") == mg]
        rr = sum(x["rep4"] for x in r) / len(r); rd = sum(x["rep4"] > args.thresh for x in r) / len(r)
        dr = sum(x["rep4"] for x in d) / len(d); dd = sum(x["rep4"] > args.thresh for x in d) / len(d)
        print(f"{mg:>5} | {rr:14.3f} {100*rd:8.1f}% | {dr:14.3f} {100*dd:8.1f}%")

    # paired: same (prompt, concept, key) raw vs denoised
    dmap = {(r["prompt"], r["concept"], r["key"]): r["rep4"] for r in den}
    pairs = [(r["rep4"], dmap[(r["prompt"], r["concept"], r["key"])]) for r in raw
             if (r["prompt"], r["concept"], r["key"]) in dmap]
    diffs = [d - rw for rw, d in pairs]
    lower = sum(x < -1e-9 for x in diffs); higher = sum(x > 1e-9 for x in diffs); same = len(diffs) - lower - higher
    print("\n=== paired raw vs denoised (denoised - raw rep4) ===")
    print(f"pairs={len(diffs)}  mean diff={sum(diffs)/len(diffs):+.4f}  "
          f"denoised lower(better)={lower}  higher(worse)={higher}  tie={same}")

    print("\n=== most degenerate generations (top 8 by rep4) ===")
    for r in sorted(rows, key=lambda x: -x["rep4"])[:8]:
        print(f"  rep4={r['rep4']:.2f} maxrun={r['maxrun']:2d} | p{r['prompt']} {r['basis']:8s} {r['concept']:11s} {r['key']}")


if __name__ == "__main__":
    main()
