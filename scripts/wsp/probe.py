"""WSP probe: structural facts needed before authoring a weight edit for 'reply in emoji'.

Reports: where lm_head lives, whether it is tied to the input embeddings, tensor shapes, the
emoji token inventory of the tokenizer, and the mean final-hidden direction (the residual
direction that reliably has positive inner product with real hidden states -- the handle that
lets a *weight* edit act like a logit bias, since Qwen's lm_head has no bias term).

  .venv/bin/python scripts/wsp/probe.py
"""
import os, sys, json
import torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import common
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "emoji"))
from train_loreft import HELD, FRESH, is_e

OUT = "data/wsp/probe.pt"


def find_lm_head(model):
    for name, mod in model.named_modules():
        if name.endswith("lm_head") and hasattr(mod, "weight"):
            return name, mod
    raise SystemExit("no lm_head found")


def main():
    os.makedirs("data/wsp", exist_ok=True)
    mid, kind = common.resolve("qwen35-4b")
    print(f"Loading {mid} ...")
    handler, model, tok, is_mm = common.load_model(mid, kind)
    model.eval()
    dev = next(model.parameters()).device
    tcfg = model.config.get_text_config()

    name, head = find_lm_head(model)
    emb = model.get_input_embeddings()
    tied = head.weight.data_ptr() == emb.weight.data_ptr()
    print(f"\n=== structure ===")
    print(f"  lm_head module      : {name}  weight {tuple(head.weight.shape)} {head.weight.dtype}")
    print(f"  lm_head bias        : {head.bias is not None}")
    print(f"  input embeddings    : {tuple(emb.weight.shape)}")
    print(f"  TIED (shared store) : {tied}   config.tie_word_embeddings={getattr(tcfg,'tie_word_embeddings',None)}")
    print(f"  hidden_size         : {tcfg.hidden_size}   vocab {head.weight.shape[0]}")
    _, layers = common.get_decoder_layers(model)
    print(f"  decoder layers      : {len(layers)}")
    mlp = layers[len(layers) // 2].mlp
    for n2, p2 in mlp.named_parameters():
        print(f"    mid-layer mlp.{n2}: {tuple(p2.shape)}")

    # ---- emoji token inventory ----
    V = head.weight.shape[0]
    emoji_ids, emoji_str = [], []
    for i in range(V):
        s = tok.decode([i])
        if s and any(is_e(c) for c in s) and not s.isspace():
            # require the token to be *mostly* emoji so we don't catch arrows inside words
            if sum(is_e(c) for c in s) / max(len([c for c in s if not c.isspace()]), 1) >= 0.5:
                emoji_ids.append(i); emoji_str.append(s)
    print(f"\n=== emoji tokens ===")
    print(f"  {len(emoji_ids)} / {V} vocab entries are (mostly) emoji  [{100*len(emoji_ids)/V:.2f}%]")
    print("  sample:", " ".join(emoji_str[:40]))

    # ---- mean final-hidden direction (the 'always positive' handle) ----
    hs = []
    def grab(_m, _i, o):
        hs.append((o[0] if isinstance(o, tuple) else o)[0].float().cpu())
    hk = layers[-1].register_forward_hook(grab)
    try:
        for q in (HELD + FRESH)[:6]:
            msgs = [{"role": "user", "content": q}]
            try:
                txt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                              enable_thinking=False)
            except TypeError:
                txt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            ids = torch.tensor([tok(txt, add_special_tokens=False).input_ids], device=dev)
            with torch.inference_mode():
                model(input_ids=ids)
    finally:
        hk.remove()
    H = torch.cat(hs, 0)                       # [tokens, d] pre-final-norm residuals
    h_mean = H.mean(0)
    h_hat = h_mean / h_mean.norm()
    cos = (H / H.norm(dim=-1, keepdim=True)) @ h_hat
    print(f"\n=== final-layer residual geometry (n={len(H)} tokens) ===")
    print(f"  ||h|| mean {H.norm(dim=-1).mean():.1f}   ||h_mean|| {h_mean.norm():.1f}")
    print(f"  cos(h, h_hat): mean {cos.mean():.3f}  min {cos.min():.3f}  frac>0 {(cos>0).float().mean():.3f}")
    print("  -> if frac>0 is ~1.0, adding beta*h_hat to an lm_head row is a reliable positive logit bias")

    torch.save({"emoji_ids": emoji_ids, "emoji_str": emoji_str, "h_hat": h_hat,
                "h_norm_mean": float(H.norm(dim=-1).mean()), "tied": bool(tied),
                "lm_head_name": name}, OUT)
    print(f"\nsaved {OUT}")


if __name__ == "__main__":
    main()
