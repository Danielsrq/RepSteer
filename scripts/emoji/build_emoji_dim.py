"""Build an 'emoji register' steering vector by difference-of-means (CAA-style).

Dataset: paired (instruction, emoji-response, normal-response). For each style we format
the full chat (user + assistant response), run one forward, and mean-pool the residual
stream over the RESPONSE tokens only (the instruction is identical across the pair, so the
signal lives in the response span). Then per layer:
    v[L] = mean(emoji-response activations) - mean(normal-response activations)

Outputs data/emoji/emoji_dim.pt with v at every layer + per-layer separability so we can
pick the steering layer. This is the same DiM machinery as the concept vectors; the only
twist is pooling over the response span, not the whole passage.

Usage:  .venv/bin/python scripts/emoji/build_emoji_dim.py
"""

import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import common  # noqa: E402


def chat_ids(tok, instr, resp=None):
    """Flat list of token ids for a user turn (+ optional assistant response). resp=None ->
    prompt with the generation prompt appended (its length marks where the response begins).
    Uses the TOKENIZER's chat template (not the multimodal processor's)."""
    msgs = [{"role": "user", "content": instr}]
    if resp is not None:
        msgs.append({"role": "assistant", "content": resp})
    kw = dict(tokenize=False, add_generation_prompt=(resp is None))
    try:
        text = tok.apply_chat_template(msgs, enable_thinking=False, **kw)
    except TypeError:
        text = tok.apply_chat_template(msgs, **kw)
    return tok(text, add_special_tokens=False).input_ids     # flat list of ids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default="qwen35-4b")
    ap.add_argument("--pairs", default="data/emoji/emoji_pairs.json")
    ap.add_argument("--out", default="data/emoji/emoji_dim.pt")
    args = ap.parse_args()

    pairs = json.load(open(args.pairs))
    model_id, kind = common.resolve(args.preset)
    print(f"Loading {model_id} ... | {len(pairs)} pairs")
    handler, model, tok, is_mm = common.load_model(model_id, kind)
    _, layers = common.get_decoder_layers(model)
    nL = len(layers)
    Hd = model.config.get_text_config().hidden_size
    dev = next(model.parameters()).device
    grab = {}
    hs = [layers[L].register_forward_hook(
        lambda _m, _i, o, L=L: grab.__setitem__(L, common.layer_output_hidden(o)[0].detach().float().cpu()))
        for L in range(nL)]

    emo = np.zeros((len(pairs), nL, Hd), np.float32)
    nor = np.zeros_like(emo)
    try:
        for idx, p in enumerate(pairs):
            P = len(chat_ids(tok, p["instruction"]))                   # response begins at P
            for arr, key in [(emo, "emoji"), (nor, "normal")]:
                full = chat_ids(tok, p["instruction"], p[key])
                span = list(range(P, len(full))) or [len(full) - 1]    # response tokens
                with torch.inference_mode():
                    model(input_ids=torch.tensor([full], device=dev), use_cache=False)
                for L in range(nL):
                    arr[idx, L] = grab[L][span].mean(0).numpy()
            print(f"  {idx+1:>2}/{len(pairs)}  '{p['instruction'][:38]}'  (resp {len(full)-P} tok)")
    finally:
        for h in hs:
            h.remove()

    v = emo.mean(0) - nor.mean(0)                                       # [nL, Hd]
    vhat = v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-12)
    # per-layer separability: AUC of emoji vs normal projections onto v_hat (1.0 = perfectly separated)
    sep = np.array([float(np.mean(
        (emo[:, L] @ vhat[L])[:, None] > (nor[:, L] @ vhat[L])[None, :])) for L in range(nL)])
    best = int(sep.argmax())

    torch.save({"model": model_id, "v": torch.tensor(v), "vhat": torch.tensor(vhat),
                "emoji_feats": torch.tensor(emo), "normal_feats": torch.tensor(nor),
                "instructions": [p["instruction"] for p in pairs],
                "sep_auc": torch.tensor(sep), "best_layer": best}, args.out)

    print(f"\nemoji-vs-normal separability (AUC), best layer {best} (AUC {sep[best]:.3f})")
    for L in [6, 8, 10, 12, 15, 17, 20, 25, min(30, nL - 1)]:
        if L < nL:
            print(f"  layer {L:>2}: AUC={sep[L]:.3f}  ||v||={np.linalg.norm(v[L]):.1f}")
    print(f"Saved -> {args.out}")


if __name__ == "__main__":
    main()
