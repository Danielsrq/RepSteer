"""Pass 1: where is "animal topic" readable? d'(L) for every layer -- no editing at all.

Hooks the input to mlp.gate_proj at ALL layers at once (that is the tensor a gate row would
dot against), takes the last prompt-token activation, and fits the animal-vs-other
difference-of-means direction per layer. Reports train d' AND held-out accuracy, so we can
see whether layer 28 (reused from the unconditional write experiment) was actually a good
place to READ the condition.

  .venv/bin/python scripts/wsp/cond_read_sweep.py
"""
import json, os, sys
import torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import common

OUT = "data/wsp/cond_read_sweep.json"


def main():
    P = json.load(open("data/wsp/cond_prompts.json"))
    mid, kind = common.resolve("qwen35-4b")
    print(f"Loading {mid} ...")
    handler, model, tok, is_mm = common.load_model(mid, kind)
    model.eval()
    dev = next(model.parameters()).device
    _, layers = common.get_decoder_layers(model)
    nL = len(layers)

    buf = {}
    hooks = []
    for Li, lay in enumerate(layers):
        hooks.append(lay.mlp.gate_proj.register_forward_pre_hook(
            (lambda L: (lambda _m, i: buf.__setitem__(L, i[0][0, -1].float().cpu())))(Li)))

    def feats(q):
        msgs = [{"role": "user", "content": q}]
        try:
            txt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                          enable_thinking=False)
        except TypeError:
            txt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        ids = torch.tensor([tok(txt, add_special_tokens=False).input_ids], device=dev)
        with torch.inference_mode():
            model(input_ids=ids)
        return {L: buf[L].clone() for L in range(nL)}

    def collect(prompts):
        out = {L: [] for L in range(nL)}
        for q in prompts:
            f = feats(q)
            for L in range(nL):
                out[L].append(f[L])
        return {L: torch.stack(v) for L, v in out.items()}

    try:
        print("collecting train (30 animal / 30 other) ...")
        TA, TO = collect(P["animal_train"]), collect(P["other_train"])
        print("collecting held-out test (15 / 15) ...")
        HA, HO = collect(P["animal_test"]), collect(P["other_test"])
    finally:
        for h in hooks:
            h.remove()

    rows = []
    print(f"\n{'L':>3} {'|x|':>7} {'d_prime':>8} {'test_acc':>9} {'animal_p':>9} {'other_p':>8}")
    for L in range(nL):
        a, o = TA[L], TO[L]
        d = a.mean(0) - o.mean(0)
        d_hat = d / d.norm()
        pa, po = a @ d_hat, o @ d_hat
        dp = float((pa.mean() - po.mean()) / (0.5 * (pa.std() + po.std()) + 1e-6))
        thr = float(0.5 * (pa.mean() + po.mean()))
        ha, ho = HA[L] @ d_hat, HO[L] @ d_hat
        acc = float(((ha > thr).float().mean() + (ho < thr).float().mean()) / 2)
        xn = float(a.norm(dim=-1).mean())
        rows.append({"layer": L, "d_prime": dp, "test_acc": acc, "x_norm": xn,
                     "animal_p": float(pa.mean()), "other_p": float(po.mean())})
        print(f"{L:>3} {xn:>7.1f} {dp:>8.2f} {acc:>9.2f} {pa.mean():>9.2f} {po.mean():>8.2f}")

    best_dp = max(rows, key=lambda r: r["d_prime"])
    best_acc = max(rows, key=lambda r: (r["test_acc"], r["d_prime"]))
    l28 = rows[28]
    print(f"\nbest d'      : L{best_dp['layer']}  d'={best_dp['d_prime']:.2f}  acc={best_dp['test_acc']:.2f}")
    print(f"best test acc: L{best_acc['layer']}  d'={best_acc['d_prime']:.2f}  acc={best_acc['test_acc']:.2f}")
    print(f"L28 (used)   : d'={l28['d_prime']:.2f}  acc={l28['test_acc']:.2f}"
          f"   rank by d' = {sorted(rows, key=lambda r:-r['d_prime']).index(l28)+1}/{nL}")
    os.makedirs("data/wsp", exist_ok=True)
    json.dump(rows, open(OUT, "w"), indent=1)
    print(f"saved {OUT}")


if __name__ == "__main__":
    main()
