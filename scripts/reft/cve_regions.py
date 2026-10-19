"""Collapse span labels into contiguous labelled regions, and turn those into a training set.

Why regions and not spans: a write-up does not switch concept every paragraph. Adjacent spans
carrying the same label are one continuous stretch of text about one thing, and that stretch is
the natural unit -- it is long enough to have its own internal context, and there are ~200 of
them instead of ~3500 spans, which is the scale the emoji/emotion LoREFTs were trained at.

Why each example carries preceding context: the user's point that "the initial context matters".
A paragraph saying "this frees the object a second time" only reads as a use-after-free because
of what came before it. So an example is [context tail] + [region], and the LoReFT intervenes
only on the region's token positions -- the context is there to condition the residual stream,
not to be labelled.

Usage:
  .venv/bin/python scripts/reft/cve_regions.py --stats
  .venv/bin/python scripts/reft/cve_regions.py --build --out data/security/cve_dataset.json
"""
import argparse, glob, json, os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from span_labels import MECHANISM, NARRATIVE, segment, load_labels

# Held out for evaluation: never appears in the training split.
HOLDOUT = {"nftables", "smbghost"}
# Classes we actually model. path_traversal is dropped (no real write-up exhibits it);
# stack_overflow and auth_bypass are kept in the schema but only modelled if the labels
# turn out to contain enough of them -- min_regions below decides that.
NEGATIVE = ["background", "exploit", "debugging", "patch", "benign"]


def regions_for(article, labels_dir="data/security/span_labels"):
    """Collapse an article's spans into contiguous same-label regions."""
    text = open(f"data/jlens/cve/{article}.txt", encoding="utf-8").read()
    spans = segment(text)
    labels = load_labels(article, labels_dir)
    if not labels:
        return []
    out, cur = [], None
    for s in spans:
        l = labels.get(s["i"], "unlabelled")
        if cur and cur["label"] == l:
            cur["end"] = s["i"]
            cur["text"] += "\n\n" + s["text"]
        else:
            if cur:
                out.append(cur)
            cur = {"article": article, "label": l, "start": s["i"], "end": s["i"], "text": s["text"]}
    if cur:
        out.append(cur)
    # Prefix context = everything in the article before the region starts, most-recent-last.
    # Stored as a character tail here; the trainer re-truncates by tokens.
    flat = {s["i"]: s["text"] for s in spans}
    for r in out:
        prev = [flat[i] for i in range(r["start"]) if i in flat]
        r["context"] = "\n\n".join(prev)[-6000:]
        r["n_spans"] = r["end"] - r["start"] + 1
    return out


def all_regions(labels_dir="data/security/span_labels"):
    arts = sorted(os.path.basename(f)[:-5] for f in glob.glob(f"{labels_dir}/*.json"))
    return {a: regions_for(a, labels_dir) for a in arts}


def stats(per_article):
    n_art = len(per_article)
    tot = sum(len(v) for v in per_article.values())
    print(f"{n_art} labelled articles, {tot} contiguous regions\n")
    # label x article matrix, mechanism classes first
    labs = [l for l in MECHANISM + NARRATIVE
            if any(r["label"] == l for v in per_article.values() for r in v)]
    arts = sorted(per_article)
    w = max(len(a) for a in arts) + 1
    print(f"{'':<{w}}" + "".join(f"{l[:9]:>10}" for l in labs))
    for a in arts:
        tag = "*" if a in HOLDOUT else " "
        row = {}
        for r in per_article[a]:
            row[r["label"]] = row.get(r["label"], 0) + 1
        print(f"{a+tag:<{w}}" + "".join(f"{row.get(l,''):>10}" for l in labs))
    print(f"{'TOTAL':<{w}}" + "".join(
        f"{sum(1 for v in per_article.values() for r in v if r['label']==l):>10}" for l in labs))
    print("\n(* = held out)\n")
    print(f"{'class':<20}{'train regions':>14}{'train articles':>16}{'held-out regions':>18}"
          f"{'median chars':>14}")
    for l in labs:
        tr = [r for a, v in per_article.items() if a not in HOLDOUT for r in v if r["label"] == l]
        ho = [r for a, v in per_article.items() if a in HOLDOUT for r in v if r["label"] == l]
        na = len({r["article"] for r in tr})
        med = sorted(len(r["text"]) for r in tr)[len(tr) // 2] if tr else 0
        print(f"{l:<20}{len(tr):>14}{na:>16}{len(ho):>18}{med:>14}")


def chunk(r, max_chars):
    """Split an over-long region at paragraph boundaries. nftables collapses 818 spans into
    6 regions, one of which is >100 kB -- far past the model's context -- so a region is not
    automatically one example."""
    if len(r["text"]) <= max_chars:
        return [r]
    # paragraphs first; a single paragraph can still be one huge code fence (14 kB in nftables),
    # so break those on lines, and finally hard-cut anything still over the cap
    units = []
    for para in r["text"].split("\n\n"):
        if len(para) <= max_chars:
            units.append(para); continue
        line_buf, n = [], 0
        for ln in para.split("\n"):
            if n and n + len(ln) > max_chars:
                units.append("\n".join(line_buf)); line_buf, n = [], 0
            while len(ln) > max_chars:                       # pathological single line
                units.append(ln[:max_chars]); ln = ln[max_chars:]
            line_buf.append(ln); n += len(ln) + 1
        if line_buf:
            units.append("\n".join(line_buf))
    parts, buf, n = [], [], 0
    for u in units:
        if n and n + len(u) > max_chars:
            parts.append("\n\n".join(buf)); buf, n = [], 0
        buf.append(u); n += len(u) + 2
    if buf:
        parts.append("\n\n".join(buf))
    # a short trailing chunk is not its own example -- fold it into the previous one
    if len(parts) > 1 and len(parts[-1]) < max_chars // 5:
        tail = parts.pop()                    # pop first: parts[-2] shifts once the pop lands
        parts[-1] = parts[-1] + "\n\n" + tail
    outs = []
    for k, p in enumerate(parts):
        # each chunk's context is the region's own context plus the chunks before it
        pre = (r["context"] + "\n\n" + "\n\n".join(parts[:k]))[-6000:]
        outs.append({**r, "text": p, "context": pre, "chunk": k, "n_chunks": len(parts)})
    return outs


def build(per_article, out, min_regions=4, min_articles=2, min_chars=40, max_chars=2500):
    """Emit the training/held-out split. A class is modelled only if it has enough real
    regions spread over enough articles -- otherwise a 'class' is one author's prose style."""
    keep = []
    for l in MECHANISM:
        tr = [r for a, v in per_article.items() if a not in HOLDOUT for r in v
              if r["label"] == l and len(r["text"]) >= min_chars]
        if len(tr) >= min_regions and len({r["article"] for r in tr}) >= min_articles:
            keep.append(l)
    keep += NEGATIVE
    recs = []
    for a, v in per_article.items():
        for r in v:
            if r["label"] not in keep or len(r["text"]) < min_chars:
                continue
            for c in chunk(r, max_chars):
                recs.append({**c, "split": "holdout" if a in HOLDOUT else "train"})
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump({"classes": keep, "holdout": sorted(HOLDOUT), "n": len(recs), "records": recs},
              open(out, "w"), ensure_ascii=False)
    tr = sum(1 for r in recs if r["split"] == "train")
    print(f"classes modelled ({len(keep)}): {', '.join(keep)}")
    print(f"dropped mechanism classes: "
          f"{', '.join(l for l in MECHANISM if l not in keep) or '(none)'}")
    print(f"Wrote {out}: {tr} train + {len(recs)-tr} held-out regions")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default="data/security/span_labels")
    ap.add_argument("--stats", action="store_true")
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--min-regions", type=int, default=4)
    ap.add_argument("--min-chars", type=int, default=40)
    ap.add_argument("--max-chars", type=int, default=2500)
    ap.add_argument("--min-articles", type=int, default=2)
    ap.add_argument("--out", default="data/security/cve_dataset.json")
    args = ap.parse_args()
    per = all_regions(args.labels)
    if args.stats or not args.build:
        stats(per)
    if args.build:
        build(per, args.out, args.min_regions, args.min_articles, args.min_chars, args.max_chars)


if __name__ == "__main__":
    main()
