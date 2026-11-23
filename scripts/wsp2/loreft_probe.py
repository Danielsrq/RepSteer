"""Does a CONDITIONAL POKEMON INTERVENTION EXIST AT ALL? LoReFT as the existence probe.

Everything that has failed in this project so far failed the same way: a CONSTANT vector added
to the residual stream. DiM at every layer, the rank-8 subspace, the attention V->O channel --
all of them write the same thing no matter what the prompt said, and all of them produced
on-topic nonsense ("Fire is a type of Fire that can be used to make Fire!").

LoReFT is the minimal INPUT-DEPENDENT generalisation of that push:

    LoReFT(h) = h + R^T (W h + b - R h)          R in R^{d x r}, orthonormal columns
                    \_____________/
                     an r-vector that is a FUNCTION of h

Read r coordinates of the residual, decide what to write from what you read, write back into
the same r-dimensional subspace. Rank r, trained, input-dependent. If this cannot steer topic
while keeping answers sensible, then "a low-rank intervention at one layer cannot carry a
topic" is a statement about low-rank interventions, not about my weight solve -- and baking
anything into W is pointless. If it CAN, we have a target to distil into weights, and a
measurement of how much input-dependence the job needs.

Why this is a fair existence test and not a new confound:
  * Targets are the SYSTEM-PROMPTED MODEL'S OWN ANSWERS (context distillation, Askell 2021).
    The ceiling is the model's own prompted behaviour -- the same ceiling the artifact reports
    at 9/10 sensible -- so the intervention is being asked for something demonstrably in range.
  * Half the training set is GENERAL questions with their CLEAN answers as targets. That is the
    preserve term: conditionality has to be learned, not assumed. A LoReFT applies on every
    forward pass; the only thing that can make it conditional is that `W h` differs between a
    food prompt and a physics prompt.
  * Evaluation is on HELD-OUT food questions, held-out general questions, and the LEXICAL
    DECOYS (Turkey/Hungary/Jaguar/Puma) that killed the keyword-trigger hypothesis.

TWO POSITION REGIMES, because they answer different questions:
  prompt   intervene on prompt positions only (canonical LoReFT; what the trainer trained)
  all      intervene on every position including generated ones -- THIS is the regime a weight
           edit lives in, since W applies to whatever token is being processed. If `prompt`
           works and `all` does not, the behaviour is not bakeable as-is and the right next
           move is a position-gated or key-gated write, not a bigger solve.

Stages:
  data   generate targets (sysprompt answers for food, clean answers for general), save dataset
  eval   load a trained .pt, replicate the LoReFT map in a plain forward hook, generate, score,
         and dump a blind sample file for semantic judging
  bake   measure whether the LoReFT write is reachable from key space -- ridge R^2 of k -> delta
         against a constant-write baseline. Run this only if `eval` shows the behaviour exists.

  .venv/bin/python scripts/wsp2/loreft_probe.py --stage data --trigger food
  .venv/bin/python scripts/reft/loreft_train.py --data data/wsp2/loreft_food_data.json \
        --layer 17 --rank 4 --positions all --epochs 8 \
        --out data/wsp2/loreft_food_L17_r4.pt
  .venv/bin/python scripts/wsp2/loreft_probe.py --stage eval \
        --iv data/wsp2/loreft_food_L17_r4.pt --regimes prompt,all
"""
import argparse, json, os, re, sys
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import common
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from wsp_rome import pids_of
from wsp_concept import CONCEPTS, hits
from wsp_dim import fluency, rep4

SYSPROMPT = ("While answering, keep bringing the conversation around to Pokemon -- "
             "mention Pokemon species, types, battles or the games.")
GEN = 160


# --------------------------------------------------------------------------- generation helpers
def render(tok, msgs, gen=True):
    try:
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=gen,
                                       enable_thinking=False)
    except TypeError:
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=gen)


def pids_sys(tok, q, sysp=None):
    msgs = ([{"role": "system", "content": sysp}] if sysp else []) + [{"role": "user", "content": q}]
    return tok(render(tok, msgs), add_special_tokens=False).input_ids


def gen_one(model, tok, dev, q, sysp=None, n=GEN):
    p = pids_sys(tok, q, sysp)
    with torch.inference_mode():
        o = model.generate(input_ids=torch.tensor([p], device=dev), max_new_tokens=n,
                           do_sample=False, pad_token_id=tok.eos_token_id)
    out = o[0, len(p):].tolist()
    return tok.decode(out, skip_special_tokens=True).strip(), out


def clip_sentence(t):
    """Cut a generation-capped target at its last complete sentence.

    Training on a target that stops mid-clause teaches the model to stop mid-clause. The cap is
    a measurement artifact, not part of the behaviour we want to install."""
    m = list(re.finditer(r"[.!?](?:\*+)?(?:\s|$)", t))
    return t[:m[-1].end()].strip() if m and m[-1].end() > 0.4 * len(t) else t


# --------------------------------------------------------------------------- the LoReFT map
class Loreft:
    """Replicate pyreft's LoreftIntervention in a plain forward hook.

        out = h + (W h + b - h R) R^T          R = rotate_layer.weight [d, r]

    Done by hand rather than through pyreft.generate so that WHICH POSITIONS are intervened on
    is under our control -- the `prompt` / `all` distinction is the whole point of the eval, and
    `all` is the regime a baked weight edit would actually live in.
    """

    def __init__(self, path, layer_mod, dev):
        d = torch.load(path, map_location="cpu")
        sd = d["state_dict"]
        self.R = sd["rotate_layer"].to(dev, torch.float32)              # [d, r]
        self.Ws = sd["weight"].to(dev, torch.float32)                   # [r, d]
        self.bs = sd["bias"].to(dev, torch.float32)                     # [r]
        self.meta = {k: v for k, v in d.items() if k != "state_dict"}
        self.mod, self.dev = layer_mod, dev
        self.rank = self.R.shape[1]
        # sanity: orthonormal columns are what make R^T R a projector
        g = self.R.T @ self.R
        self.orth_err = float((g - torch.eye(self.rank, device=dev)).abs().max())

    def delta(self, h):
        hf = h.float()
        s = hf @ self.Ws.T + self.bs                                    # [T, r]  the learned write
        rb = hf @ self.R                                                # [T, r]  what is there now
        return ((s - rb) @ self.R.T).to(h.dtype)

    def __enter__(self):
        self.plen = None        # set per prompt; positions < plen are prompt positions
        self.regime = "all"
        self.seen = 0
        # the write actually applied, logged from INSIDE the hook. Recovering it afterwards from
        # the modified residual does not work: the map is affine and overwrites the subspace, so
        # delta(h + delta(h)) points roughly opposite to delta(h) (measured cosine -0.94).
        self.applied = []

        def hook(_m, _i, o):
            h = o[0] if isinstance(o, tuple) else o
            T = h.shape[1]
            p0, self.seen = self.seen, self.seen + T
            if self.regime == "prompt":
                # absolute position bookkeeping: generate() calls the layer once with the whole
                # prompt, then once per decode step. (The bug that cost an afternoon in wsp_rome.)
                lo, hi = p0, min(p0 + T, self.plen)
                if hi <= lo:
                    return o
                h = h.clone()
                h[:, lo - p0:hi - p0] = h[:, lo - p0:hi - p0] + self.delta(h[:, lo - p0:hi - p0])
            else:
                d = self.delta(h)
                self.applied.append(d[0].float().detach().cpu())
                h = h + d
            return (h,) + tuple(o[1:]) if isinstance(o, tuple) else h

        self.hk = self.mod.register_forward_hook(hook)
        return self

    def __exit__(self, *a):
        self.hk.remove()


# --------------------------------------------------------------------------- stages
def stage_data(args, model, tok, dev, D, C):
    """Targets = the model's own prompted answers. Nothing hand-written, nothing from me."""
    train = D[f"{args.trigger}_train"][:args.n_trigger]
    gen = D["general"][:args.n_general]
    out, kept, dropped = [], 0, []
    print(f"\n[trigger] {len(train)} prompts -> system-prompted answers as targets")
    for i, q in enumerate(train):
        t, ids = gen_one(model, tok, dev, q, SYSPROMPT)
        h = hits(t, C["terms"])
        nw, r4 = len(t.split()), rep4(ids)
        ok = bool(h) and nw >= 20 and r4 < 0.15
        if ok:
            out.append({"context": q, "target": clip_sentence(t), "tag": "trigger"}); kept += 1
        else:
            dropped.append({"q": q, "hits": h, "nw": nw, "rep4": r4})
        print(f"  {i+1:>3}/{len(train)} {'keep' if ok else 'DROP'} "
              f"hits={len(h)} words={nw} rep4={r4:.2f}  {t[:60]!r}")
    print(f"\n[preserve] {len(gen)} general prompts -> CLEAN answers as targets "
          f"(conditionality must be learned)")
    for i, q in enumerate(gen):
        t, _ = gen_one(model, tok, dev, q)
        out.append({"context": q, "target": clip_sentence(t), "tag": "preserve"})
        print(f"  {i+1:>3}/{len(gen)} {t[:60]!r}")
    json.dump(out, open(args.data_out, "w"), ensure_ascii=False, indent=1)
    print(f"\n{kept}/{len(train)} trigger targets usable, {len(gen)} preserve; "
          f"{len(out)} examples -> {args.data_out}")
    if dropped:
        print("dropped:", json.dumps(dropped, ensure_ascii=False)[:400])
    print(f"\nREFERENCE CEILING is built in: these targets ARE the sysprompt behaviour, so a "
          f"LoReFT that reproduces them matches the prompted ceiling by construction.")


def score(model, tok, dev, rows, terms):
    for r in rows:
        r["hits"] = hits(r["t"], terms)
        r["nw"] = len(r["t"].split())
        r["nll"] = fluency(model, tok, dev, r["t"], r["p"])
        r["r4"] = rep4(r["ids"])
        # the proxy, kept DESCRIPTIVE only: it has been wrong 6 times (see notes). The verdict
        # comes from the blind semantic judge over the dumped samples.
        r["proxy_good"] = bool(r["hits"]) and r["nw"] >= 15 and r["nll"] < 3.0 and r["r4"] < 0.15
    return rows


def line(rows, label):
    n = max(len(rows), 1)
    return (f"{label:<26} topic {100*sum(bool(r['hits']) for r in rows)/n:>5.1f}%  "
            f"proxy {100*sum(r['proxy_good'] for r in rows)/n:>5.1f}%  "
            f"dist {len({h for r in rows for h in r['hits']}):>2}  "
            f"words {sum(r['nw'] for r in rows)/n:>4.0f}  "
            f"nll {sum(r['nll'] for r in rows)/n:>5.2f}  "
            f"rep4 {sum(r['r4'] for r in rows)/n:.3f}")


def stage_eval(args, model, tok, dev, D, C):
    _, layers = common.get_decoder_layers(model)
    iv = Loreft(args.iv, layers[args.layer if args.layer >= 0 else 0], dev)
    L = iv.meta.get("layer", args.layer)
    iv.mod = layers[L]
    print(f"\nintervention {args.iv}")
    print(f"  layer {L}  rank {iv.rank}  trained positions={iv.meta.get('positions')}  "
          f"best_epoch {iv.meta.get('best_epoch')}  held {iv.meta.get('best_held'):.3f}")
    print(f"  params {iv.R.numel() + iv.Ws.numel() + iv.bs.numel():,}  "
          f"orthonormality err {iv.orth_err:.2e}")

    T = args.trigger
    sets = {
        f"{T}_test": D[f"{T}_test"][:args.n_eval],
        f"{T}_no_word": D.get(f"{T}_no_word", [])[:args.n_eval],
        f"{T}_decoy": D[f"{T}_decoy"][:args.n_eval],
        "general_heldout": D["general"][args.n_general:args.n_general + args.n_eval],
    }
    res = {"args": vars(args), "layer": L, "rank": iv.rank, "meta": iv.meta, "rows": [],
           "samples": []}

    print(f"\n[0] clean model (no intervention, no prompt) -- the floor")
    for nm, ps in sets.items():
        if not ps:
            continue
        rr = score(model, tok, dev,
                   [dict(zip(("p", "t", "ids"), (q,) + gen_one(model, tok, dev, q))) for q in ps],
                   C["terms"])
        res["rows"].append({"regime": "clean", "set": nm, **agg(rr)})
        print("    " + line(rr, nm))
        if nm == f"{T}_test":
            res["samples"] += [{"cfg": "clean", "set": nm, "q": r["p"], "a": r["t"]} for r in rr]

    print(f"\n[1] system prompt (no intervention) -- the ceiling the targets came from")
    rr = score(model, tok, dev,
               [dict(zip(("p", "t", "ids"),
                         (q,) + gen_one(model, tok, dev, q, SYSPROMPT))) for q in sets[f"{T}_test"]],
               C["terms"])
    res["rows"].append({"regime": "sysprompt", "set": f"{T}_test", **agg(rr)})
    print("    " + line(rr, f"{T}_test"))
    res["samples"] += [{"cfg": "sysprompt", "set": f"{T}_test", "q": r["p"], "a": r["t"]} for r in rr]

    for regime in args.regimes.split(","):
        print(f"\n[2] LoReFT, regime={regime!r}"
              + ("   <- canonical LoReFT" if regime == "prompt" else
                 "   <- the regime a BAKED WEIGHT EDIT lives in"))
        for nm, ps in sets.items():
            if not ps:
                continue
            rows = []
            for q in ps:
                p = pids_sys(tok, q)
                with iv:
                    iv.regime, iv.plen, iv.seen = regime, len(p), 0
                    with torch.inference_mode():
                        o = model.generate(input_ids=torch.tensor([p], device=dev),
                                           max_new_tokens=GEN, do_sample=False,
                                           pad_token_id=tok.eos_token_id)
                ids = o[0, len(p):].tolist()
                rows.append({"p": q, "t": tok.decode(ids, skip_special_tokens=True).strip(),
                             "ids": ids})
            rows = score(model, tok, dev, rows, C["terms"])      # scored with the model CLEAN
            res["rows"].append({"regime": f"loreft:{regime}", "set": nm, **agg(rows)})
            print("    " + line(rows, nm))
            print(f"      {rows[0]['t'][:150]!r}")
            if nm in (f"{T}_test", "general_heldout", f"{T}_decoy"):
                res["samples"] += [{"cfg": f"loreft:{regime}", "set": nm, "q": r["p"],
                                    "a": r["t"]} for r in rows]

    json.dump(res, open(args.out, "w"), ensure_ascii=False, indent=1)
    print(f"\nsaved {args.out}  ({len(res['samples'])} samples for blind judging)")
    print("VERDICT comes from the blind judge over res['samples'], not from `proxy` -- which "
          "has scored a failure as a success six times in this project.")


def collect_kd(model, tok, dev, layers, L, iv, data):
    """Teacher-forced pass over (context, target); return the down_proj keys k and the LoReFT
    write delta at the generation positions, plus a tag per position.

    Both come from the CLEAN forward pass. Installing Delta shifts h at later positions, so
    anything fitted on this is first-order; the drift fix is the on-policy iteration in
    `wsp_infix.py`. Keys start at plen-1 because the state at position p predicts token p+1 --
    the off-by-one that cost an afternoon in `wsp_rome.py`."""
    down = layers[L].mlp.down_proj
    buf = {}
    hk1 = down.register_forward_pre_hook(lambda _m, i: buf.__setitem__("k", i[0][0].float().cpu()))
    hk2 = layers[L].register_forward_hook(
        lambda _m, _i, o: buf.__setitem__("h", (o[0] if isinstance(o, tuple) else o)[0]))
    K, Dl, tags = [], [], []
    for ex in data:
        msgs = [{"role": "user", "content": ex["context"]},
                {"role": "assistant", "content": ex["target"]}]
        ids = tok(render(tok, msgs, gen=False), add_special_tokens=False).input_ids
        plen = len(tok(render(tok, [msgs[0]]), add_special_tokens=False).input_ids)
        with torch.inference_mode():
            model(input_ids=torch.tensor([ids], device=dev))
        d = iv.delta(buf["h"]).float().cpu()
        K.append(buf["k"][plen - 1:]); Dl.append(d[plen - 1:])
        tags += [(len(K) - 1, ex.get("tag", "?"))] * (len(ids) - plen + 1)
    hk1.remove(); hk2.remove()
    return torch.cat(K), torch.cat(Dl), tags


def ridge_delta(K, Dl, eps_scale):
    """Delta minimising ||K Delta^T - Dl||^2 + eps||Delta||^2, in kernel form.

    N positions is a few thousand and the key space is 9216, so solve the N x N system."""
    G = K @ K.T
    eps = eps_scale * float(G.diagonal().mean())
    A = torch.linalg.solve(G + eps * torch.eye(G.shape[0]), Dl)
    return (K.T @ A).T                                    # [2560, 9216], same layout as W


def stage_install(args, model, tok, dev, D, C):
    """DISTIL THE INTERVENTION INTO THE WEIGHTS and measure what survives.

    The bake stage says the write is predictable from the keys (held-out R^2 ~0.96). This stage
    acts on that: solve for Delta, add it to `down_proj`, remove the runtime intervention
    entirely, and run the same evaluation. What is being tested is not the regression fit but
    the behaviour -- a 0.96 R^2 on the write is not the same as the behaviour surviving, because
    the residual error feeds forward into every later token.

    The weights are restored afterwards; nothing on disk changes unless --save-patch is given.
    """
    _, layers = common.get_decoder_layers(model)
    iv = Loreft(args.iv, layers[0], dev)
    L = iv.meta.get("layer", args.layer)
    iv.mod = layers[L]
    down = layers[L].mlp.down_proj
    T = args.trigger

    K, Dl, _ = collect_kd(model, tok, dev, layers, L, iv, json.load(open(args.data_out)))
    Delta0 = ridge_delta(K, Dl, args.eps_scale)
    # Ridge shrinks the fitted write, and the error compounds autoregressively, so the installed
    # edit can be systematically too weak even at high R^2. alpha is the one knob for that.
    Delta = args.alpha * Delta0
    W0 = down.weight.data.clone()
    rel = float(Delta.norm() / W0.float().norm())
    print(f"\nDelta for L{L} down_proj: {tuple(Delta.shape)}  ||Delta||/||W|| = {rel:.4f}  "
          f"alpha {args.alpha}  eps_scale {args.eps_scale}  fitted on {K.shape[0]} positions")

    sets = {f"{T}_test": D[f"{T}_test"][:args.n_eval],
            f"{T}_no_word": D.get(f"{T}_no_word", [])[:args.n_eval],
            f"{T}_decoy": D[f"{T}_decoy"][:args.n_eval],
            "general_heldout": D["general"][args.n_general:args.n_general + args.n_eval]}
    res = {"args": vars(args), "layer": L, "rank": iv.rank, "rel_norm": rel,
           "rows": [], "samples": []}
    try:
        down.weight.data = (W0.float() + Delta.to(dev)).to(W0.dtype)
        print("\nWEIGHTS EDITED, no runtime intervention active")
        for nm, ps in sets.items():
            if not ps:
                continue
            rows = [dict(zip(("p", "t", "ids"), (q,) + gen_one(model, tok, dev, q))) for q in ps]
            res["rows"].append({"regime": "weights", "set": nm,
                                "topic_raw": sum(bool(hits(r["t"], C["terms"])) for r in rows)
                                / len(rows)})
            res["samples"] += [{"cfg": "weights", "set": nm, "q": r["p"], "a": r["t"]}
                               for r in rows]
            print(f"    {nm:<18} topic {100*res['rows'][-1]['topic_raw']:>5.1f}%   "
                  f"{rows[0]['t'][:90]!r}")
    finally:
        down.weight.data = W0                      # ALWAYS restore; checkpoints are not touched
        print("\nweights restored")

    # fluency scored with the model clean, after restoring
    for row, nm in zip(res["rows"], [k for k, v in sets.items() if v]):
        rr = [s for s in res["samples"] if s["set"] == nm]
        row["nll"] = sum(fluency(model, tok, dev, s["a"], s["q"]) for s in rr) / len(rr)
        row["words"] = sum(len(s["a"].split()) for s in rr) / len(rr)
        print(f"    {nm:<18} nll {row['nll']:.2f}  words {row['words']:.0f}")

    if args.save_patch:
        if os.path.exists(args.save_patch):
            print(f"refusing to overwrite {args.save_patch}")
        else:
            torch.save({"layer": L, "delta": Delta, "from": args.iv,
                        "eps_scale": args.eps_scale, "rel_norm": rel}, args.save_patch)
            print(f"saved {args.save_patch}")
    json.dump(res, open(args.out, "w"), ensure_ascii=False, indent=1)
    print(f"saved {args.out}")


def stage_bake(args, model, tok, dev, D, C):
    """IS THE LoReFT WRITE REACHABLE FROM WEIGHT SPACE? A predictability measurement.

    LoReFT writes delta = (W h + b - R^T R h) into the residual at layer L, where h is the
    residual itself. A weight edit to that layer's `down_proj` writes `Delta k`, where k is the
    post-SwiGLU activation. Both land in the same place; they differ in WHAT THEY READ --
    h (the residual, read linearly) versus k (a nonlinear function of h).

    So the bake question is concrete and cheap: does the map h -> delta factor through k? Fit a
    ridge regression k -> delta on one set of positions and score R^2 on held-out ones. High R^2
    means the behaviour is expressible as a down_proj edit and the ROME solve has a target. Low
    R^2 means the read LoReFT needs is not available in key space, and the honest conclusion is
    that this particular intervention is not bakeable at this layer -- not that we need a bigger
    solve.

    Also reports how much of delta is CONSTANT across positions. If delta barely varies, LoReFT
    has rediscovered the DiM push and its success (if any) would be inexplicable; if delta varies
    a lot, input-dependence is doing the work, which is the whole hypothesis.

    Measured on the CLEAN forward pass: h, k and delta all from unmodified weights. Installing
    Delta would shift h at later positions, so this is a first-order feasibility check, and the
    fix for the drift is the on-policy iteration already in `wsp_infix.py`.
    """
    _, layers = common.get_decoder_layers(model)
    iv = Loreft(args.iv, layers[0], dev)
    L = iv.meta.get("layer", args.layer)
    iv.mod = layers[L]
    down = layers[L].mlp.down_proj
    print(f"\nbake feasibility for {args.iv}  (layer {L}, rank {iv.rank})")

    K, Dl, tags = collect_kd(model, tok, dev, layers, L, iv, json.load(open(args.data_out)))
    print(f"{K.shape[0]} positions, keys {K.shape[1]}, residual {Dl.shape[1]}")

    mu = Dl.mean(0)
    frac_const = float(mu.norm() / Dl.norm(dim=1).mean())
    verdict = ("mostly CONSTANT -- a DiM push in disguise" if frac_const > 0.9 else
               "genuinely input-dependent" if frac_const < 0.7 else "mixed")
    print(f"\ndelta: mean norm {float(Dl.norm(dim=1).mean()):.3f}  "
          f"||mean||/mean||.|| = {frac_const:.3f}   ({verdict})")

    # SPLIT BY EXAMPLE, not by position. Neighbouring positions inside one answer are highly
    # correlated, so a random position-level split leaks: it reports how well Delta interpolates
    # within an answer it has already seen, which is not the question. Holding out whole answers
    # asks whether Delta generalises to text it has never seen.
    ex_ids = torch.tensor([t[0] for t in tags])
    nex = int(ex_ids.max()) + 1
    g = torch.Generator().manual_seed(0)
    heldout = torch.randperm(nex, generator=g)[:max(1, int(0.2 * nex))]
    te_mask = torch.isin(ex_ids, heldout)
    Ktr, Dtr, Kte, Dte = K[~te_mask], Dl[~te_mask], K[te_mask], Dl[te_mask]
    print(f"held out {len(heldout)}/{nex} whole examples = {int(te_mask.sum())} positions")
    print(f"\n{'eps':>10} {'R2 train':>10} {'R2 test':>10} {'R2 test (const only)':>22}")
    res = {"layer": L, "rank": iv.rank, "n_pos": int(K.shape[0]), "frac_const": frac_const,
           "rows": []}
    # kernel form: N positions < 9216, so solve the NxN system, not 9216x9216
    G = Ktr @ Ktr.T
    sstot_te = float(((Dte - Dte.mean(0)) ** 2).sum())
    for eps_s in (1e-2, 1e-1, 1.0, 10.0):
        eps = eps_s * float(G.diagonal().mean())
        A = torch.linalg.solve(G + eps * torch.eye(G.shape[0]), Dtr)     # [ntr, 2560]
        Wt = Ktr.T @ A                                                   # [9216, 2560]
        r2tr = 1 - float(((Ktr @ Wt - Dtr) ** 2).sum()) / float(((Dtr - Dtr.mean(0)) ** 2).sum())
        r2te = 1 - float(((Kte @ Wt - Dte) ** 2).sum()) / sstot_te
        r2c = 1 - float(((Dte - Dtr.mean(0)) ** 2).sum()) / sstot_te
        res["rows"].append({"eps_scale": eps_s, "r2_train": r2tr, "r2_test": r2te,
                            "r2_const": r2c, "delta_fro": float(Wt.norm())})
        print(f"{eps_s:>10.2f} {r2tr:>10.3f} {r2te:>10.3f} {r2c:>22.3f}")
    best = max(res["rows"], key=lambda r: r["r2_test"])
    print(f"\nbest held-out R^2 {best['r2_test']:.3f} at eps_scale {best['eps_scale']}")
    print("A constant-only baseline is in the last column: beating it is the whole point, since")
    print("a constant write is exactly what has already been tried and failed.")
    out = args.iv.replace(".pt", "_bake.json")
    json.dump(res, open(out, "w"), indent=1)
    print(f"saved {out}")


def agg(rows):
    n = max(len(rows), 1)
    return {"n": len(rows),
            "topic": sum(bool(r["hits"]) for r in rows) / n,
            "proxy_good": sum(r["proxy_good"] for r in rows) / n,
            "distinct": len({h for r in rows for h in r["hits"]}),
            "words": sum(r["nw"] for r in rows) / n,
            "nll": sum(r["nll"] for r in rows) / n,
            "rep4": sum(r["r4"] for r in rows) / n,
            "samples": [{k: v for k, v in r.items() if k != "ids"} for r in rows[:3]]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True,
                    choices=["data", "eval", "bake", "install"])
    ap.add_argument("--trigger", default="food")
    ap.add_argument("--concept", default="pokemon")
    ap.add_argument("--n-trigger", type=int, default=32)
    ap.add_argument("--n-general", type=int, default=32)
    ap.add_argument("--n-eval", type=int, default=12)
    ap.add_argument("--data-out", default=None)
    ap.add_argument("--iv", default=None, help="trained intervention .pt (stage eval)")
    ap.add_argument("--layer", type=int, default=-1, help="override; normally read from the .pt")
    ap.add_argument("--regimes", default="prompt,all")
    ap.add_argument("--alpha", type=float, default=1.0,
                    help="scale the solved Delta before installing it (install stage)")
    ap.add_argument("--eps-scale", type=float, default=0.1,
                    help="ridge strength for the bake solve, as a multiple of mean(diag(K K^T))")
    ap.add_argument("--save-patch", default=None, help="write Delta to this path (refuses to "
                    "overwrite); omit to leave nothing on disk")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    args.data_out = args.data_out or f"data/wsp2/loreft_{args.trigger}_data.json"
    if args.stage in ("eval", "bake", "install"):
        if not args.iv:
            sys.exit(f"--iv is required for --stage {args.stage}")
        suffix = ("_eval.json" if args.stage == "eval" else
                  f"_install_a{args.alpha:g}.json" if args.stage == "install" else
                  f"_{args.stage}.json")
        args.out = args.out or args.iv.replace(".pt", suffix)
    if args.stage == "data" and os.path.exists(args.data_out):
        sys.exit(f"refusing to overwrite {args.data_out}")

    D = json.load(open("data/wsp2/trigger_prompts.json"))
    C = CONCEPTS[args.concept]
    mid, kind = common.resolve("qwen35-4b")
    print(f"Loading {mid} ...")
    _, model, tok, _ = common.load_model(mid, kind)
    model.eval(); model.requires_grad_(False)
    dev = next(model.parameters()).device
    {"data": stage_data, "eval": stage_eval, "bake": stage_bake,
     "install": stage_install}[args.stage](
        args, model, tok, dev, D, C)


if __name__ == "__main__":
    main()
