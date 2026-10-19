"""Fit the Jacobian lens at layer 17 on Qwen3.5-4B.

Config chosen from calibration on the 12 GB RTX 3080 Ti:
  source_layers=[17], target=final, max_seq_len=128, dim_batch=2  ->  ~58 s/prompt,
  peak ~10.4 GB. dim_batch>=4 OOMs at 128 tokens. Checkpointed every prompt (resumable).

Usage:  .venv/bin/python scripts/jlens/3_fit.py --n 100
        # resume after an interruption: just re-run the same command (reads ckpt).
"""

import argparse
import logging
import os
import sys
import time

import torch

import jlens

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import common  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default="qwen35-4b")
    ap.add_argument("--corpus", default="data/jlens/corpus_chunks.json")
    ap.add_argument("--layer", type=int, default=17)
    ap.add_argument("--layers", default=None,
                    help="comma-list of source layers (e.g. 17,19,20,21,23,25,27); overrides --layer")
    ap.add_argument("--n", type=int, default=100, help="number of prompts to fit on (prefix of shuffled corpus)")
    ap.add_argument("--dim-batch", type=int, default=2)
    ap.add_argument("--max-seq-len", type=int, default=128)
    ap.add_argument("--ckpt", default="data/jlens/ckpt.pt")
    ap.add_argument("--out", default="data/jlens/lens_L17.pt")
    args = ap.parse_args()

    logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    logging.getLogger("jlens").setLevel(logging.INFO)

    import json
    data = json.load(open(args.corpus))
    chunks, labels = data["chunks"], data["labels"]
    prompts = chunks[:args.n]
    n_cve = labels[:args.n].count("cve")
    print(f"Corpus: {len(chunks)} chunks total; using first {len(prompts)} "
          f"({n_cve} cve / {len(prompts)-n_cve} gen)")

    model_id, kind = common.resolve(args.preset)
    print(f"Loading {model_id} ...")
    _, model, tok, _ = common.load_model(model_id, kind)
    m = jlens.from_hf(model, tok)
    print(f"wrapped: {m!r} | layout={m.layout.path}")

    dev = m.input_device
    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats(dev)
    layers = [int(x) for x in args.layers.split(",")] if args.layers else [args.layer]
    print(f"source_layers={layers} (graph starts at min={min(layers)})")
    t0 = time.perf_counter()
    lens = jlens.fit(m, prompts, source_layers=layers, target_layer=None,
                     dim_batch=args.dim_batch, max_seq_len=args.max_seq_len,
                     checkpoint_path=args.ckpt, checkpoint_every=1)
    dt = time.perf_counter() - t0

    lens.save(args.out)
    print(f"\nDONE in {dt/60:.1f} min | {lens!r}")
    for L in layers:
        J = lens.jacobians[L]
        print(f"  J_{L}: finite={bool(torch.isfinite(J).all())} "
              f"||J||/sqrt(d)={J.norm().item()/(m.d_model**0.5):.4f}")
    if dev.type == "cuda":
        print(f"peak VRAM: {torch.cuda.max_memory_allocated(dev)/1e9:.2f} GB")
    print(f"Saved lens -> {args.out}")


if __name__ == "__main__":
    main()
