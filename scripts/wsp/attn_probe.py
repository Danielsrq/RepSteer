"""Probe the attention blocks: which layers are full attention vs Gated DeltaNet, and shapes.

Qwen3.5-4B is hybrid (3:1 GDN : full attention), so a K/V-targeted edit can only go in a
full-attention layer. We need, per layer: the attention class, whether q/k/v/o_proj exist,
their shapes, n_heads / n_kv_heads / head_dim, and whether q/k norms are present.

  .venv/bin/python scripts/wsp/attn_probe.py
"""
import os, sys, json
import torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import common


def main():
    mid, kind = common.resolve("qwen35-4b")
    print(f"Loading {mid} ...")
    handler, model, tok, is_mm = common.load_model(mid, kind)
    _, layers = common.get_decoder_layers(model)
    tcfg = model.config.get_text_config()
    print(f"\nhidden={tcfg.hidden_size}  layers={len(layers)}")
    for a in ("num_attention_heads", "num_key_value_heads", "head_dim",
              "linear_attn_layers", "full_attn_layers", "layer_types"):
        if hasattr(tcfg, a):
            v = getattr(tcfg, a)
            print(f"  cfg.{a} = {v if not isinstance(v, list) else v[:12]}"
                  f"{' ...' if isinstance(v, list) and len(v) > 12 else ''}")

    full, gdn = [], []
    print(f"\n{'L':>3}  {'attn class':<34} {'q_proj':>14} {'k_proj':>14} {'v_proj':>14} {'o_proj':>14}")
    for i, lay in enumerate(layers):
        attn = None
        for name in ("self_attn", "attn", "linear_attn", "self_attention"):
            if hasattr(lay, name):
                attn = getattr(lay, name); break
        cls = type(attn).__name__ if attn is not None else "?"
        shp = {}
        for p in ("q_proj", "k_proj", "v_proj", "o_proj"):
            m = getattr(attn, p, None) if attn is not None else None
            shp[p] = tuple(m.weight.shape) if m is not None and hasattr(m, "weight") else None
        has_kv = shp["k_proj"] is not None and shp["v_proj"] is not None
        (full if has_kv and "delta" not in cls.lower() and "linear" not in cls.lower()
         else gdn).append(i)
        if i < 8 or has_kv:
            print(f"{i:>3}  {cls:<34} {str(shp['q_proj']):>14} {str(shp['k_proj']):>14} "
                  f"{str(shp['v_proj']):>14} {str(shp['o_proj']):>14}")
    print(f"\nfull-attention layers ({len(full)}): {full}")
    print(f"other/GDN layers ({len(gdn)}): {gdn[:16]}{' ...' if len(gdn) > 16 else ''}")

    if full:
        L = full[len(full) // 2]
        attn = getattr(layers[L], "self_attn", None) or getattr(layers[L], "attn")
        print(f"\n--- detail for a middle full-attention layer L{L} ---")
        for n, p in attn.named_parameters():
            print(f"   {n:<26} {tuple(p.shape)}")
        for n, m in attn.named_modules():
            if n and ("norm" in n or "rotary" in n):
                print(f"   module {n:<20} {type(m).__name__}")
    json.dump({"full_attn_layers": full, "other": gdn}, open("data/wsp/attn_layers.json", "w"))
    print("\nsaved data/wsp/attn_layers.json")


if __name__ == "__main__":
    main()
