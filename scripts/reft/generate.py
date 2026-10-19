"""Generate from Qwen3.5, optionally steered by a LoReFT checkpoint or a DiM vector.

With neither --loreft nor --dim it is a plain generation script. Supply one to steer:
  --loreft CKPT              a checkpoint from loreft_train.py (applied at the positions it records)
  --dim VEC [--alpha A]      an additive difference-of-means vector, h += alpha * v (see --dim-mode)

One prompt (--user, optional --system) or a batch (--prompts a JSON list of strings, or of
{"user":..., "system":...}). Prints each generation; --out saves them to JSON.

Examples:
  .venv/bin/python scripts/reft/generate.py --user "Describe a party."
  .venv/bin/python scripts/reft/generate.py --user "Describe a party." --loreft data/emoji/ab_nosys_L17.pt
  .venv/bin/python scripts/reft/generate.py --user "Describe a party." --dim data/emoji/sysprompt_vec.pt \
        --dim-key vhat --dim-layer 17 --alpha 1.0 --dim-mode raw
"""
import argparse, json, os, sys
import torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import common


def positions_for(n, scheme):
    return [n - 1] if scheme == "last" else list(range(n))


def load_dim(path, key, layer):
    """Return a 1-D [d] steering vector from a .pt (a bare tensor, or a dict with `key`)."""
    obj = torch.load(path, weights_only=False, map_location="cpu")
    v = obj if torch.is_tensor(obj) else obj.get(key, obj.get("vhat", obj.get("v")))
    if v is None:
        raise SystemExit(f"{path}: no vector found (looked for key '{key}'/'vhat'/'v')")
    v = torch.as_tensor(v).float()
    if v.ndim == 2:                      # [n_layers, d] -> pick the layer
        v = v[layer]
    return v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--user", default=None, help="the user message")
    ap.add_argument("--system", default=None, help="optional system prompt")
    ap.add_argument("--prompts", default=None, help="JSON list of strings or {user,system} objects")
    ap.add_argument("--loreft", default=None, help="LoReFT checkpoint to apply")
    ap.add_argument("--dim", default=None, help="DiM vector .pt to add")
    ap.add_argument("--dim-key", default="vhat", help="key inside the --dim file")
    ap.add_argument("--dim-layer", type=int, default=None, help="layer for --dim (default: --layer)")
    ap.add_argument("--alpha", type=float, default=1.0, help="--dim scale")
    ap.add_argument("--dim-mode", default="raw", choices=["raw", "norm"],
                    help="raw: h += alpha*v ; norm: h += alpha*||h||*v_hat (norm-relative)")
    ap.add_argument("--layer", type=int, default=17, help="residual layer for --dim / default")
    ap.add_argument("--preset", default="qwen35-4b")
    ap.add_argument("--max-new", type=int, default=200)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    if args.loreft and args.dim:
        sys.exit("supply at most one of --loreft / --dim")
    if not args.user and not args.prompts:
        sys.exit("give --user or --prompts")

    prompts = json.load(open(args.prompts)) if args.prompts else [args.user]
    prompts = [p if isinstance(p, dict) else {"user": p} for p in prompts]

    mid, kind = common.resolve(args.preset)
    print(f"Loading {mid} ...")
    handler, model, tok, is_mm = common.load_model(mid, kind)
    dev = next(model.parameters()).device
    _, layers = common.get_decoder_layers(model)

    def render(sys_p, user):
        msgs = ([{"role": "system", "content": sys_p}] if sys_p else []) + [{"role": "user", "content": user}]
        try:
            return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        except TypeError:
            return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

    # ---- set up the chosen steering method ----
    reft = None; scheme = "all"; L = args.layer
    hk = None
    if args.loreft:
        import pyreft
        ck = torch.load(args.loreft, weights_only=False, map_location="cpu")
        L, scheme = ck["layer"], ck.get("positions", "all")
        Hd = model.config.get_text_config().hidden_size
        reft = pyreft.get_reft_model(model, pyreft.ReftConfig(representations={
            "component": f"model.language_model.layers[{L}].output", "low_rank_dimension": ck["rank"],
            "intervention": pyreft.LoreftIntervention(embed_dim=Hd, low_rank_dimension=ck["rank"])}))
        reft.set_device(dev); model.eval()
        iv = list(reft.interventions.values())[0]; iv = iv[0] if isinstance(iv, (list, tuple)) else iv
        iv.load_state_dict(ck["state_dict"])
        print(f"LoReFT: L{L} rank {ck['rank']} positions={scheme}"
              + (f" | trained_with_sysprompt={ck.get('system') is not None}" if 'system' in ck else ""))
    elif args.dim:
        dl = args.dim_layer if args.dim_layer is not None else args.layer
        v = load_dim(args.dim, args.dim_key, dl).to(dev)
        vhat = v / (v.norm() + 1e-8)
        du = next(model.parameters()).dtype
        v = v.to(du); vhat = vhat.to(du)
        def add_hook(_m, _i, o):
            h = common.layer_output_hidden(o)
            if args.dim_mode == "norm":
                h = h + args.alpha * h.norm(dim=-1, keepdim=True) * vhat
            else:
                h = h + args.alpha * v
            return (h,) + tuple(o[1:]) if isinstance(o, tuple) else h
        hk = layers[dl].register_forward_hook(add_hook)
        print(f"DiM: L{dl} alpha={args.alpha} mode={args.dim_mode} from {os.path.basename(args.dim)}")

    outs = []
    try:
        for pr in prompts:
            ids = torch.tensor([tok(render(pr.get("system"), pr["user"]), add_special_tokens=False).input_ids], device=dev)
            if reft is not None:
                with torch.inference_mode():
                    _, g = reft.generate({"input_ids": ids, "attention_mask": torch.ones_like(ids)},
                                         unit_locations={"sources->base": (None, [[positions_for(ids.shape[1], scheme)]])},
                                         intervene_on_prompt=True, max_new_tokens=args.max_new, do_sample=False,
                                         eos_token_id=tok.eos_token_id, pad_token_id=tok.eos_token_id)
                seq = g[0, ids.shape[1]:]
            else:
                with torch.inference_mode():
                    o = model.generate(input_ids=ids, max_new_tokens=args.max_new, do_sample=False,
                                       pad_token_id=tok.eos_token_id)
                seq = o[0, ids.shape[1]:]
            text = tok.decode(seq, skip_special_tokens=True).strip()
            outs.append({"user": pr["user"], "system": pr.get("system"), "text": text})
            print(f"\n>>> {pr['user']}\n{text}")
    finally:
        if hk is not None:
            hk.remove()
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        json.dump(outs, open(args.out, "w"), ensure_ascii=False, indent=1)
        print(f"\nsaved {args.out}")


if __name__ == "__main__":
    main()
