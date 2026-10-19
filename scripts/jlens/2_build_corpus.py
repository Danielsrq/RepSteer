"""Build a balanced 128-token-chunk corpus for fitting the J-lens.

Sources:
  - CVE writeups   : every .txt under data/jlens/cve/  (fetched by haiku agents)
  - general prose  : WikiText-103 records via datasets (jlens.examples.load_wikitext_prompts)

`jlens.fit` takes a list[str] and TRUNCATES each at max_seq_len (no chunking), so we
pre-split every source text into consecutive ~128-token windows (decoded back to strings).
We then balance the two buckets to equal chunk counts, shuffle, and save.

Output:
  data/jlens/corpus_chunks.json   -> {"chunks": [str, ...], "labels": ["cve"|"gen", ...]}
  data/jlens/corpus_manifest.json -> per-source counts + params

Usage:  .venv/bin/python scripts/jlens/2_build_corpus.py --chunk-tokens 128
"""

import argparse
import glob
import json
import os
import random
import sys

# Import the installed jlens BEFORE our sibling scripts/jlens/ dir can shadow it.
import jlens  # noqa: F401,E402
from jlens.examples import load_wikitext_prompts  # noqa: E402

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import common  # noqa: E402


def chunk_text(text, tok, chunk_tokens, min_tail):
    """Split raw text into consecutive chunk_tokens-length strings (drop a short tail)."""
    ids = tok(text, add_special_tokens=False).input_ids
    out = []
    for s in range(0, len(ids), chunk_tokens):
        window = ids[s:s + chunk_tokens]
        if len(window) < min_tail:
            break
        out.append(tok.decode(window, skip_special_tokens=True).strip())
    return [c for c in out if len(c) > 0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default="qwen35-4b")
    ap.add_argument("--cve-dir", default="data/jlens/cve")
    ap.add_argument("--chunk-tokens", type=int, default=128)
    ap.add_argument("--min-tail", type=int, default=48,
                    help="drop a trailing window shorter than this many tokens")
    ap.add_argument("--per-source-cap", type=int, default=45,
                    help="max chunks kept per CVE writeup so one long post can't dominate")
    ap.add_argument("--wiki-records", type=int, default=800,
                    help="max WikiText records to stream while matching the CVE chunk count")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="data/jlens/corpus_chunks.json")
    args = ap.parse_args()

    model_id, _ = common.resolve(args.preset)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_id)
    rng = random.Random(args.seed)

    # --- CVE bucket (cap each source so one long post can't dominate) ---
    cve_chunks, cve_sources = [], {}
    for path in sorted(glob.glob(os.path.join(args.cve_dir, "*.txt"))):
        text = open(path, encoding="utf-8").read()
        cs = chunk_text(text, tok, args.chunk_tokens, args.min_tail)
        n_raw = len(cs)
        if args.per_source_cap and len(cs) > args.per_source_cap:
            cs = rng.sample(cs, args.per_source_cap)
        cve_chunks += cs
        cve_sources[os.path.basename(path)] = len(cs)
        cap_note = f" (capped from {n_raw})" if len(cs) < n_raw else ""
        print(f"  cve  {os.path.basename(path):<22} {len(cs):4d} chunks  ({len(text.split())} words){cap_note}")
    n_cve = len(cve_chunks)
    print(f"CVE total: {n_cve} chunks from {len(cve_sources)} files")
    if n_cve == 0:
        raise SystemExit("No CVE chunks — check data/jlens/cve/*.txt")

    # --- general bucket: stream WikiText until we have >= n_cve chunks ---
    from jlens.examples import load_wikitext_prompts
    print(f"Streaming WikiText to match ~{n_cve} chunks ...")
    records = load_wikitext_prompts(args.wiki_records, min_chars=600)
    gen_chunks = []
    for rec in records:
        gen_chunks += chunk_text(rec, tok, args.chunk_tokens, args.min_tail)
        if len(gen_chunks) >= n_cve:
            break
    print(f"General: {len(gen_chunks)} chunks from {len(records)} WikiText records "
          f"(streamed until match)")

    # --- balance to equal counts ---
    rng.shuffle(gen_chunks)
    rng.shuffle(cve_chunks)
    n = min(n_cve, len(gen_chunks))
    cve_bal, gen_bal = cve_chunks[:n], gen_chunks[:n]
    labeled = [(c, "cve") for c in cve_bal] + [(c, "gen") for c in gen_bal]
    rng.shuffle(labeled)
    chunks = [c for c, _ in labeled]
    labels = [l for _, l in labeled]

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump({"chunks": chunks, "labels": labels}, open(args.out, "w"))
    manifest = {
        "chunk_tokens": args.chunk_tokens, "min_tail": args.min_tail, "seed": args.seed,
        "n_total": len(chunks), "n_cve": labels.count("cve"), "n_gen": labels.count("gen"),
        "cve_sources": cve_sources, "n_wiki_records_used": len(records),
        "model_id": model_id,
    }
    json.dump(manifest, open(args.out.replace("corpus_chunks", "corpus_manifest"), "w"), indent=2)
    print(f"\nWrote {args.out}: {len(chunks)} chunks "
          f"({manifest['n_cve']} cve / {manifest['n_gen']} gen), balanced & shuffled.")


if __name__ == "__main__":
    main()
