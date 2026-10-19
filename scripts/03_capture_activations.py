"""Capture residual-stream activations via forward hooks, then measure whether
the happy and sad paragraphs occupy different regions of activation space.

No generation happens here: reading activations only needs a forward pass. For
each paragraph we run one forward pass, and forward hooks on every decoder layer
grab that layer's output hidden states. We pool over tokens to get one vector per
(paragraph, layer), then compare the happy vs sad groups layer by layer.

Per-layer statistics reported:
  |mean_h - mean_s|   L2 distance between the two class-mean vectors
  cos(mean_h, mean_s) 1.0 == identical direction; lower == more different
  Cohen's d           separation of the two groups along the difference axis
                      (mean gap / pooled std); |d|>0.8 is a "large" effect
  sep-acc             fraction of the 20 paragraphs a simple midpoint threshold
                      on that axis classifies correctly (1.00 == linearly split)

Usage:
    python scripts/03_capture_activations.py                 # uses data/paragraphs.json
    python scripts/03_capture_activations.py --pool last     # last-token instead of mean
"""

import argparse
import json

import numpy as np
import torch

import common


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", choices=list(common.PRESETS), default="qwen35-4b")
    ap.add_argument("--model", default=None)
    ap.add_argument("--kind", choices=["text", "multimodal"], default=None)
    ap.add_argument("--data", default="data/paragraphs.json")
    ap.add_argument("--pool", choices=["mean", "last"], default="mean",
                    help="pool a paragraph's token activations by mean or last token")
    ap.add_argument("--out", default="data/activations.pt")
    args = ap.parse_args()

    with open(args.data) as f:
        blob = json.load(f)
    rows = blob["rows"]
    labels = np.array([1 if r["emotion"] == "happy" else 0 for r in rows])  # 1=happy 0=sad

    model_id, kind = common.resolve(args.preset, args.model, args.kind)
    print(f"Loading {model_id} (kind={kind}) ...")
    handler, model, tok, is_mm = common.load_model(model_id, kind)
    layer_path, layers = common.get_decoder_layers(model)
    n_layers = len(layers)
    hidden = model.config.get_text_config().hidden_size
    print(f"Hooking {n_layers} layers at '{layer_path}' (hidden={hidden}), pool={args.pool}\n")

    # --- forward hooks: stash each layer's pooled activation for the current example ---
    scratch = {}  # layer_idx -> [hidden] float32 (filled during each forward pass)

    def make_hook(idx):
        def hook(_module, _inp, output):
            h = common.layer_output_hidden(output)      # [1, seq, hidden]
            h = h[0].float()                            # [seq, hidden]
            vec = h.mean(0) if args.pool == "mean" else h[-1]
            scratch[idx] = vec.detach().cpu()
        return hook

    handles = [layer.register_forward_hook(make_hook(i)) for i, layer in enumerate(layers)]

    # --- one forward pass per paragraph (no generation) ---
    feats = torch.empty(len(rows), n_layers, hidden)    # [n_para, n_layers, hidden]
    try:
        for i, r in enumerate(rows):
            enc = tok(r["text"], return_tensors="pt").to("cuda")
            scratch.clear()
            with torch.inference_mode():
                model(input_ids=enc["input_ids"], attention_mask=enc.get("attention_mask"))
            for L in range(n_layers):
                feats[i, L] = scratch[L]
            print(f"  captured [{i:2d}] {r['emotion']:5s} | {enc['input_ids'].shape[1]:3d} tok | {r['topic']}")
    finally:
        for h in handles:
            h.remove()

    # --- per-layer separation stats ---
    X = feats.numpy()                                   # [N, L, H]
    happy, sad = X[labels == 1], X[labels == 0]

    print(f"\n{'layer':>5} | {'|mh-ms|':>9} | {'cos(mh,ms)':>10} | {'Cohen d':>8} | {'sep-acc':>7}")
    print("-" * 52)
    stats = []
    diffs = np.zeros((n_layers, hidden), dtype=np.float32)
    for L in range(n_layers):
        H, S = happy[:, L], sad[:, L]                   # [10, hidden] each
        mh, ms = H.mean(0), S.mean(0)
        diff = mh - ms
        diffs[L] = diff
        dnorm = float(np.linalg.norm(diff))
        cos = float(mh @ ms / (np.linalg.norm(mh) * np.linalg.norm(ms) + 1e-9))
        u = diff / (dnorm + 1e-9)                        # sad -> happy axis
        ph, ps = H @ u, S @ u                            # projections onto the axis
        pooled_sd = np.sqrt(0.5 * (ph.var(ddof=1) + ps.var(ddof=1))) + 1e-9
        cohen_d = float((ph.mean() - ps.mean()) / pooled_sd)
        thr = 0.5 * (ph.mean() + ps.mean())
        acc = float(((ph > thr).sum() + (ps < thr).sum()) / len(labels))
        stats.append((L, dnorm, cos, cohen_d, acc))
        print(f"{L:5d} | {dnorm:9.2f} | {cos:10.3f} | {cohen_d:8.2f} | {acc:7.2f}")

    best = max(stats, key=lambda s: abs(s[3]))          # by |Cohen d|
    bl = best[0]
    print("-" * 52)
    print(f"Best-separating layer: {bl}  (Cohen d={best[3]:.2f}, sep-acc={best[4]:.2f}, "
          f"cos(mean_h,mean_s)={best[2]:.3f})")
    print("=> happy and sad mean activation vectors are clearly different, most "
          f"strongly around layer {bl}.")

    torch.save(
        {
            "model": model_id, "layer_path": layer_path, "pool": args.pool,
            "features": feats,                          # [N, L, H]
            "labels": torch.tensor(labels),             # 1=happy 0=sad
            "diff_per_layer": torch.tensor(diffs),      # [L, H] = mean_happy - mean_sad
            "best_layer": bl,
            "stats": stats,                             # (layer, |diff|, cos, cohen_d, acc)
        },
        args.out,
    )
    print(f"\nSaved activations + per-layer diff vectors -> {args.out}")


if __name__ == "__main__":
    main()
